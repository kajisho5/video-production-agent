"""Follow the sibling Skills' releases: re-pin their contracts from the release tags, and keep the integration lock (ADR-117).

    python scripts/repin.py update [--only NAME ...] [--report report.json] [--pr-body body.md] [--si si]
        For each sibling Skill, find its latest release tag (`vX.Y.Z`, or `blender-skill-vX.Y.Z`), and when it is newer than the pin,
        replace the pin with the tag's committed `contract/contract.json` (byte for byte, no Skill code run), move every reader of
        the pin with it, classify the change with `si contract-diff OLD NEW --json` (a dependency's version range that only widened
        is additive, not si's breaking: addendum 2) and decide:
          auto   none / additive, inside the adapter's SUPPORTED_SKILL_VERSIONS gate: the pin moves, nothing else
          widen  none / additive, a new minor of a 0.x / 1.x line outside the gate: the gate takes the new `X.Y.` prefix
                 (a 0.x -> 1.0.0 promotion included: the gate takes `1.0.`, addendum 3)
          review breaking, or a major change: the pin moves, the gate does not, and the pull request asks a person to look
                 (it is labelled, and the `repin review` check stays red until a person removes the label)
        Then the lock (integration/siblings.lock.json) is written for the new pins.
    python scripts/repin.py verify --report report.json --pr-body body.md
        after `update`: run the unit job's test commands (read from .github/workflows/tests.yml) and the evals on the re-pinned tree.
        A failure makes the re-pin a `review` one (the pull request is a draft, labelled, and lists the failing tests with the lines in
        tests/ and evals/ that still name an old release); either way the result heads the pull request body
    python scripts/repin.py lock        write the lock from the pins as they are (tag and commit of each pinned version)
    python scripts/repin.py check       offline: the lock names every sibling and agrees with the pins (also a unit test)
    python scripts/repin.py clone --dest DIR [--main] [--only NAME ...] [--skip NAME ...]
        clone each sibling into DIR/<repo>: at the lock's tag, checked against the lock's commit, or (--main, the nightly canary)
        at the default branch

Standard library only, Python 3.9+. Network access is through `git` only (ls-remote, fetch, clone).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
OWNER = "kajisho5"
LOCK_PATH = Path("integration") / "siblings.lock.json"
LOCK_FORMAT = "video-agent/siblings-lock@1"
CONTRACT_IN_TAG = "contract/contract.json"
# where pin readers live; docs/ keeps the history of earlier pins and is never rewritten
SCAN_DIRS = ("src", "tests", "evals")
SCAN_SKIP = ("tests/test_repin.py",)     # this script's own tests: their versions are examples, not pins
SCAN_SUFFIXES = (".py", ".json")
TOOLS = Path("src") / "video_agent" / "tools"
PINS = Path("src") / "video_agent" / "skills" / "pins"


@dataclass(frozen=True)
class Sibling:
    name: str                         # the lock's key (the Skill's short name)
    repo: str                         # kajisho5/<repo>; also the clone directory's name
    pin_dir: str                      # repository-relative directory holding exactly one pin
    pin_prefix: str                   # pin file: <pin_prefix><version>.json
    gate_file: Optional[str]          # the adapter whose SUPPORTED_SKILL_VERSIONS gates the Skill (None: handoff only, no gate)
    tag_prefixes: Tuple[str, ...] = ("v",)
    names: Tuple[str, ...] = ()       # words that name the Skill on a line (besides repo, the directory name and its kebab form)
    owned: Tuple[str, ...] = ()       # files (repository-relative) about this Skill alone, besides the default ones

    @property
    def identifiers(self) -> Tuple[str, ...]:
        d = Path(self.pin_dir).name
        words = {self.repo, self.name, d, d.replace("_", "-"), *self.names}
        if self.pin_dir == PINS.as_posix():
            words.discard("pins")
        return tuple(sorted(words, key=len, reverse=True))

    def owned_files(self) -> Tuple[str, ...]:
        files = list(self.owned)
        if self.gate_file:
            files.append(self.gate_file)
        if self.pin_dir != PINS.as_posix():
            d = Path(self.pin_dir).name
            files += [f"tests/fake_{d}.py", f"tests/test_adapter_{d}.py", f"tests/test_{d}.py"]
        return tuple(files)


def _tool(name: str, repo: str, directory: str, **kw) -> Sibling:
    return Sibling(name=name, repo=repo, pin_dir=(TOOLS / directory).as_posix(), pin_prefix="contract_",
                   gate_file=(TOOLS / directory / "adapter.py").as_posix(), **kw)


SIBLINGS: Tuple[Sibling, ...] = (
    _tool("artifacts", "artifacts-skill", "artifacts", owned=("tests/test_documents.py",)),
    _tool("audio-production", "audio-production-skill", "audio_production"),
    _tool("color-grading", "color-grading-skill", "color_grading", owned=("tests/test_color_grading_features.py",)),
    _tool("detection", "detection-skill", "detection", owned=("tests/test_face_redaction.py",)),
    _tool("diarization", "diarization-skill", "diarization"),
    _tool("dubbing", "dubbing-skill", "dubbing"),
    _tool("imagemagick", "imagemagick-skill", "imagemagick", owned=("tests/test_image_prepare.py",)),
    _tool("ingest", "ingest-skill", "ingest", owned=("tests/test_offload.py",)),
    _tool("media-analysis", "media-analysis-skill", "media_analysis"),
    _tool("motion-graphics", "motion-graphics-skill", "motion_graphics", owned=("tests/test_motion_graphics_features.py",)),
    _tool("nle-bridge", "nle-bridge-skill", "nle_bridge", owned=("tests/test_nle.py",)),
    _tool("publish", "publish-skill", "publish"),
    _tool("qc", "qc-skill", "qc", owned=("tests/test_qc_checks.py",)),
    _tool("remotion-ffmpeg", "remotion-ffmpeg-skill", "remotion"),
    _tool("slide-sync", "slide-sync-skill", "slide_sync", owned=("tests/test_slides.py",)),
    _tool("sound-design", "sound-design-skill", "sound_design", owned=("tests/test_bgm.py",)),
    _tool("subtitle", "subtitle-skill", "subtitle", owned=("tests/test_subtitle_features.py",)),
    _tool("thumbnail", "thumbnail-skill", "thumbnail", owned=("tests/test_thumbnail_features.py", "tests/test_thumbnail_document.py")),
    _tool("transcription", "transcription-skill", "transcription", owned=("tests/test_transcribe_options.py",)),
    _tool("video-editing", "video-editing-skill", "video_editing", owned=("tests/test_video_editing_features.py",)),
    _tool("voice-enhance", "voice-enhance-skill", "voice_enhance"),
    Sibling(name="blender", repo="blender-skill", pin_dir=PINS.as_posix(), pin_prefix="blender-skill_", gate_file=None,
            tag_prefixes=("blender-skill-v", "v")),
    Sibling(name="hyperframes", repo="hyperframes-skill", pin_dir=PINS.as_posix(), pin_prefix="hyperframes-skill_",
            gate_file=(TOOLS / "hyperframes" / "adapter.py").as_posix(), owned=("tests/fake_hyperframes.py", "tests/test_html_layer.py")),
)
BY_NAME: Dict[str, Sibling] = {s.name: s for s in SIBLINGS}
# releases cut before the Skill committed contract/contract.json: the locked clone has nothing to compare, the tests check the live contract
NO_COMMITTED_CONTRACT = {("artifacts", "v0.4.0")}

VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def parse_version(text: str) -> Tuple[int, int, int]:
    m = VERSION_RE.fullmatch(text)
    if not m:
        raise ValueError(f"not a release version: {text!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def tag_version(tag: str, prefixes: Sequence[str]) -> Optional[str]:
    """The release version a tag names (`v1.2.3`, `blender-skill-v1.2.3`), or None for any other tag."""
    for p in prefixes:
        if tag.startswith(p) and VERSION_RE.fullmatch(tag[len(p):]):
            return tag[len(p):]
    return None


def latest_tag(tags: Iterable[str], prefixes: Sequence[str]) -> Optional[Tuple[str, str]]:
    """(tag, version) of the highest release version; on a tie the first prefix wins (`blender-skill-v` over `v`)."""
    best: Optional[Tuple[Tuple[int, int, int], int, str, str]] = None
    for t in tags:
        v = tag_version(t, prefixes)
        if v is None:
            continue
        rank = (parse_version(v), -next(i for i, p in enumerate(prefixes) if t.startswith(p) and t[len(p):] == v))
        if best is None or rank > best[:2]:
            best = (rank[0], rank[1], t, v)
    return (best[2], best[3]) if best else None


def current_pin(root: Path, sib: Sibling) -> Tuple[Path, str]:
    d = root / sib.pin_dir
    found = sorted(p for p in d.glob(f"{sib.pin_prefix}*.json") if VERSION_RE.fullmatch(p.name[len(sib.pin_prefix):-5]))
    if len(found) != 1:
        raise SystemExit(f"{sib.name}: expected exactly one {sib.pin_prefix}<version>.json in {sib.pin_dir}, found {[p.name for p in found]}")
    return found[0], found[0].name[len(sib.pin_prefix):-5]


# ---- the gate -----------------------------------------------------------------------------------------------------------------------

def _gate_node(source: str) -> Optional[ast.Tuple]:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SUPPORTED_SKILL_VERSIONS" for t in node.targets):
            if isinstance(node.value, ast.Tuple):
                return node.value
    return None


def read_gate(source: str) -> Tuple[str, ...]:
    node = _gate_node(source)
    if node is None:
        raise ValueError("no SUPPORTED_SKILL_VERSIONS = (...) tuple")
    return tuple(ast.literal_eval(node))


def in_gate(version: str, gate: Optional[Sequence[str]]) -> bool:
    return gate is None or any(version.startswith(p) for p in gate)


def widen_gate(source: str, prefix: str, note: str) -> str:
    """Add `prefix` at the end of the gate tuple, and `note` as a comment line just above the assignment."""
    node = _gate_node(source)
    if node is None:
        raise ValueError("no SUPPORTED_SKILL_VERSIONS = (...) tuple")
    if not node.elts:
        raise ValueError("the gate tuple is empty")
    lines = source.splitlines(keepends=True)
    last = node.elts[-1]          # insert right after the last element: a trailing comma or a `)` on its own line stays valid
    line_no, col = last.end_lineno - 1, last.end_col_offset
    line = lines[line_no]
    lines[line_no] = f'{line[:col]}, "{prefix}"{line[col:]}'
    start = next(i for i, l in enumerate(lines) if l.startswith("SUPPORTED_SKILL_VERSIONS"))
    newline = "\r\n" if lines[start].endswith("\r\n") else "\n"
    lines.insert(start, f"# {note}{newline}")
    return "".join(lines)


GATE_TESTS = ("tests/test_version_gates.py", "tests/test_release_version.py")


def _call_args(text: str, start: int) -> Optional[List[str]]:
    """The top-level arguments of the call whose `(` is at `text[start]`, read across lines; None when it never closes.
    Brackets inside string literals are skipped."""
    depth, args, current, quote = 0, [], [], ""
    for ch in text[start:]:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
            if depth == 1:
                continue
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(current).strip())
                return args
        elif ch == "," and depth == 1:
            args.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    return None


def _accepts(body: str) -> bool:
    """An `assertEqual(..., [])` call, with or without a message and on one line or several, is in `body`:
    `self.assertEqual(x.check_contract(live), [], v)` says the version is accepted."""
    for m in re.finditer(r"assertEqual\s*\(", body):
        args = _call_args(body, m.end() - 1)
        if args and "[]" in args[:2]:
            return True
    return False


def _is_refusal_loop(lines: Sequence[str], i: int) -> bool:
    """Addendum 4: the `for v in (...)` loop at line `i` is a refusal loop. That means its body (the lines indented under it)
    asserts a refusal (`assertTrue(` / `assertRaises(`), never acceptance (`assertEqual(..., [], ...)`), and the loop does not
    list the pinned release (`pinned_version(`). An acceptance loop can hold the probe `"X.Y.0"` right after the reader
    rewrite moved the pin there (motion-graphics 0.4.5 -> 1.0.0, agent #182). A loop with no body read is left alone."""
    if "pinned_version(" in lines[i]:
        return False
    indent = len(lines[i]) - len(lines[i].lstrip())
    body = []
    for line in lines[i + 1:]:
        if not line.strip():
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
    refuses = any("assertTrue(" in b or "assertRaises(" in b for b in body)
    return refuses and not _accepts("".join(body))


def move_refusal_probe(text: str, sib: Sibling, prefix: str) -> Tuple[str, int]:
    """After the gate takes `X.Y.`, the gate tests' probe `"X.Y.0"` (a version they expect refused) moves to the next minor, `"X.(Y+1).0"`:
    on a `for v in (...)` line about this Skill (named on the line or on one of the lines above it) whose loop is a refusal loop
    (`_is_refusal_loop`, addendum 4), and in a `("<repo>", "X.Y.0")` pair."""
    major, minor = (int(x) for x in prefix.rstrip(".").split("."))
    probe, moved = f'"{major}.{minor}.0"', f'"{major}.{minor + 1}.0"'
    aliases = adapter_aliases(text)
    mine = sib.identifiers + (tuple(aliases.get(Path(sib.gate_file).parent.name, ())) if sib.gate_file else ())
    pair = re.compile(r'(\(\s*"' + re.escape(sib.repo) + r'",\s*)' + re.escape(probe) + r"(\s*\))")
    lines = text.splitlines(keepends=True)
    count = 0
    for i, line in enumerate(lines):
        if probe not in line:
            continue
        out = pair.sub(lambda m: m.group(1) + moved + m.group(2), line)
        if (line.lstrip().startswith("for v in (") and any(_mentions(l, mine) for l in lines[max(0, i - CONTEXT_LINES):i + 1])
                and _is_refusal_loop(lines, i)):
            out = out.replace(probe, moved)
        if out != line:
            lines[i] = out
            count += 1
    return "".join(lines), count


# ---- the decision -------------------------------------------------------------------------------------------------------------------

@dataclass
class Decision:
    kind: str                 # auto | widen | review
    reasons: List[str] = field(default_factory=list)
    new_prefix: Optional[str] = None


def decide(old: str, new: str, classification: str, gate: Optional[Sequence[str]]) -> Decision:
    o, n = parse_version(old), parse_version(new)
    if n <= o:
        return Decision("review", [f"the tag {new} is not newer than the pin {old}"])
    reasons: List[str] = []
    # addendum 3 (Velisa ADR-031): a Skill's first 1.0.0 after 0.x is a promotion, not a break; with a none / additive diff it
    # widens the gate to `1.0.` like any minor. Every other major change (1.x -> 2.0.0, 0.x -> 1.1.0, ...) stays a review.
    promotion = o[0] == 0 and n == (1, 0, 0)
    if n[0] != o[0] and not promotion:
        reasons.append(f"major version change {old} -> {new}: the gate is not widened")
    if classification not in ("none", "additive"):
        reasons.append(f"si contract-diff: {classification}")
    if reasons:
        return Decision("review", reasons)
    if in_gate(new, gate):
        return Decision("auto", [f"si contract-diff: {classification}, {new} inside the gate"])
    if n[0] in (0, 1):
        prefix = f"{n[0]}.{n[1]}."
        return Decision("widen", [f"si contract-diff: {classification}, {new} outside the gate: the gate takes {prefix}"], prefix)
    return Decision("review", [f"{new} is outside the gate and not a 0.x / 1.x minor"])


# ---- moving the pin's readers -------------------------------------------------------------------------------------------------------

def _version_patterns(old: str) -> List[re.Pattern]:
    v = re.escape(old)
    return [re.compile(r"(?<![\w.])([\"'])" + v + r"\1"), re.compile(r"@" + v + r"(?![\w.])")]


SOURCE_TAG_RE = r"\b(v){old} tag [0-9a-f]{{7,40}}\b"


# a Skill's name counts where it is not part of a longer name: `ingest_version` names ingest, `fake-ingest-x` does not
def _mentions(line: str, words: Iterable[str]) -> bool:
    return any(re.search(r"(?<![A-Za-z0-9-])" + re.escape(w) + r"(?![A-Za-z0-9-])", line) for w in words)


NOT_A_SIBLING = ("ffmpeg-skill", "ffmpeg_skill", "FfmpegSkill", "ffmpeg")   # the engine: its versions are never a sibling's
CONTEXT_LINES = 3      # a line with no Skill name takes the Skill named on one of the 3 lines above it (`pk = next(... "qc")`)


IMPORT_RE = re.compile(r"^\s*from video_agent\.tools\.(\w+)(?:\.adapter)? import \(?([\w ,]+)\)?", re.M)
MODULE_ALIAS_RE = re.compile(r"^\s*import video_agent\.tools\.(\w+)\.adapter as (\w+)", re.M)


def adapter_aliases(text: str) -> Dict[str, Tuple[str, ...]]:
    """tools directory -> the names this file imports from that Skill's package: `from video_agent.tools.qc import adapter as qc_adapter`
    (`qc_adapter.` names qc), `from video_agent.tools.audio_production.adapter import pinned_contract as audio_contract`. A name
    imported from two Skills names neither."""
    found: Dict[str, set] = {}
    for m in IMPORT_RE.finditer(text):
        for part in m.group(2).split(","):
            words = part.split()
            if words:
                found.setdefault(m.group(1), set()).add(words[-1])          # `x as y` -> y, `x` -> x
    for m in MODULE_ALIAS_RE.finditer(text):
        found.setdefault(m.group(1), set()).add(m.group(2))
    seen: Dict[str, int] = {}
    for names in found.values():
        for n in names:
            seen[n] = seen.get(n, 0) + 1
    return {d: tuple(sorted(n for n in names if seen[n] == 1)) for d, names in found.items()}


@dataclass
class Rewrite:
    changed: Dict[str, int] = field(default_factory=dict)      # file -> lines changed
    leftovers: List[str] = field(default_factory=list)          # file:line: text, an old version literal on a line about this Skill


def rewrite_text(text: str, rel: str, sib: Sibling, others: Sequence[Sibling], old: str, new: str, new_sha: str,
                 same_version: Sequence[Sibling] = ()) -> Tuple[str, int, List[str]]:
    """Move one file's references from the old pin to the new one. Returns (text, lines changed, leftovers).
    `same_version`: the other siblings pinned at `old` too. A line that names one of them as well as this Skill is ambiguous (whose
    "0.2.3" is it?): it is left as it is and listed."""
    old_file, new_file = f"{sib.pin_prefix}{old}.json", f"{sib.pin_prefix}{new}.json"
    owned = rel in sib.owned_files() or rel.startswith(sib.pin_dir + "/") and sib.pin_dir != PINS.as_posix()
    aliases = adapter_aliases(text)        # `from video_agent.tools.qc import adapter as qc_adapter`: `qc_adapter.` names qc here
    mine = sib.identifiers + tuple(aliases.get(Path(sib.gate_file).parent.name, ()) if sib.gate_file else ())
    theirs = tuple(w for o in others for w in o.identifiers + tuple(aliases.get(Path(o.gate_file).parent.name, ()) if o.gate_file else ())
                   if w not in mine) + NOT_A_SIBLING
    unique_file = sib.pin_prefix != "contract_"             # blender-skill_<v>.json names its Skill
    vpats = _version_patterns(old)
    src_tag = re.compile(SOURCE_TAG_RE.format(old=re.escape(old)))
    lines = text.splitlines(keepends=True)
    file_mine = owned or _mentions(text, mine)
    twins = tuple(w for o in same_version for w in o.identifiers + tuple(aliases.get(Path(o.gate_file).parent.name, ()) if o.gate_file else ())
                  if w not in mine)
    count, leftovers = 0, []
    for i, line in enumerate(lines):
        if old not in line:
            continue
        about_me = _mentions(line, mine)
        about_other = _mentions(line, theirs)
        if not about_me and not about_other:
            above = lines[max(0, i - CONTEXT_LINES):i]
            near_me = any(_mentions(prev, mine) for prev in above)
            near_other = any(_mentions(prev, theirs) for prev in above)
            about_me = near_me and not near_other
        applies = about_me or (owned and not about_other)
        ambiguous = applies and bool(twins) and _mentions(line, twins)
        if ambiguous:
            leftovers.append(f"{rel}:{i + 1}: (also names a Skill pinned at {old}) {line.strip()[:140]}")
            continue
        out = line
        if old_file in out and (unique_file or applies):
            out = out.replace(old_file, new_file)
        if applies:
            out = src_tag.sub(lambda m: f"v{new} tag {new_sha[:7]}", out)
            for p in vpats:
                out = p.sub(lambda m: m.group(0).replace(old, new), out)
        if out != line:
            lines[i] = out
            count += 1
        # what is left of the old pin on a line that is not another Skill's, in a file that names this Skill: listed in the pull request
        if not about_other and file_mine and (any(p.search(out) for p in vpats) or old_file in out):
            leftovers.append(f"{rel}:{i + 1}: {out.strip()[:160]}")
    return "".join(lines), count, leftovers


def scan_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for d in SCAN_DIRS:
        for p in sorted((root / d).rglob("*")):
            if p.is_file() and p.suffix in SCAN_SUFFIXES and "__pycache__" not in p.parts:
                if p.parent.name in ("pins",) or (p.name.startswith("contract_") and p.parent.parent.name == "tools"):
                    continue       # the pins themselves
                if p.relative_to(root).as_posix() in SCAN_SKIP:
                    continue
                files.append(p)
    return files


def rewrite_readers(root: Path, sib: Sibling, old: str, new: str, new_sha: str) -> Rewrite:
    result = Rewrite()
    others = [s for s in SIBLINGS if s is not sib]
    same_version = [s for s in others if current_pin(root, s)[1] == old] if all((root / s.pin_dir).is_dir() for s in others) else []
    for p in scan_files(root):
        raw = p.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if old not in text:
            continue
        rel = p.relative_to(root).as_posix()
        out, n, left = rewrite_text(text, rel, sib, others, old, new, new_sha, same_version)
        if n:
            p.write_bytes(out.encode("utf-8"))
            result.changed[rel] = n
        result.leftovers += left
    return result


# ---- git -----------------------------------------------------------------------------------------------------------------------------

def _git(*args: str, cwd: Optional[Path] = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def repo_url(repo: str) -> str:
    return f"https://github.com/{OWNER}/{repo}"


def ls_remote_tags(repo: str) -> Dict[str, str]:
    """tag -> the commit it points at (annotated tags peeled)."""
    out: Dict[str, str] = {}
    peeled: Dict[str, str] = {}
    for line in _git("ls-remote", "--tags", repo_url(repo)).splitlines():
        sha, ref = line.split("\t")
        name = ref[len("refs/tags/"):]
        if name.endswith("^{}"):
            peeled[name[:-3]] = sha
        else:
            out[name] = sha
    out.update(peeled)
    return out


class TagFiles:
    """Read a file at a tag of a sibling repository: a shallow fetch of that tag into a bare cache, then `git show` (no Skill code runs)."""

    def __init__(self, cache: Path):
        self.cache = cache

    def show(self, repo: str, tag: str, path: str, sha: str) -> bytes:
        """The file at the commit `sha` that ls-remote gave for `tag`; refused when the fetched tag is another commit (the tag moved)."""
        bare = self.cache / repo
        if not bare.exists():
            _git("init", "-q", "--bare", str(bare))
        _git("fetch", "-q", "--depth", "1", "--no-tags", repo_url(repo), f"+refs/tags/{tag}:refs/tags/{tag}", cwd=bare)
        fetched = _git("rev-parse", f"refs/tags/{tag}^{{commit}}", cwd=bare).strip()
        if fetched != sha:
            raise TagMoved(f"{tag} is {fetched} now, ls-remote said {sha}")
        return subprocess.run(["git", "show", f"{sha}:{path}"], cwd=bare, check=True, capture_output=True).stdout


class TagMoved(RuntimeError):
    pass


def contract_identity(contract: dict) -> Optional[str]:
    """The Skill a contract says it is: `skill_id`, else `skill.id`, else `skill` (a string)."""
    s = contract.get("skill")
    return contract.get("skill_id") or (s.get("id") if isinstance(s, dict) else s)


def validate_contract(data: bytes, version: str, pinned: dict) -> Optional[str]:
    """Why these bytes cannot be the new pin (None when they can): a JSON object of the tag's version, the same Skill as the pin."""
    try:
        doc = json.loads(data)
    except ValueError as e:
        return f"not JSON ({e})"
    if not isinstance(doc, dict):
        return f"not a contract object ({type(doc).__name__})"
    if doc.get("version") != version:
        return f"declares version {doc.get('version')!r}, the tag is {version}"
    if contract_identity(doc) != contract_identity(pinned):
        return f"is {contract_identity(doc)!r}, the pin is {contract_identity(pinned)!r}"
    return None


def contract_diff(si: str, old: Path, new: Path) -> dict:
    proc = subprocess.run([si, "contract-diff", str(old), str(new), "--json"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"si contract-diff failed ({proc.returncode}): {proc.stderr.strip()}")
    diff = json.loads(proc.stdout)
    return reclassify_dependency_ranges(diff, json.loads(old.read_text(encoding="utf-8")), json.loads(new.read_text(encoding="utf-8")))


# ---- a dependency's version range that only widened (addendum 2) ---------------------------------------------------------------------
# `si contract-diff` compares `dependencies[].version_range` as an opaque string: any change is `breaking` (value-changed). A range that
# only widens (`>=2.0.0 <4.0.0` -> `>=2.0.0 <5.0.0`: the Skill now also runs with ffmpeg-skill 4.x) takes nothing from a caller of the
# Skill: every engine release it ran with before, it still runs with. Such a change is `additive` here. A narrowed, moved or unreadable
# range, or one whose dependency changed its skill_id, stays `breaking`.

_RANGE_TERM_RE = re.compile(r"^(>=|<=|>|<|==|=)?\s*(\d+(?:\.\d+){0,2})$")
_DEPENDENCY_RANGE_RE = re.compile(r"^/dependencies/(\d+)/version_range$")
Bound = Optional[Tuple[Tuple[int, int, int], bool]]   # (version, inclusive); None = unbounded


def parse_range(text: str) -> Optional[Tuple[Bound, Bound]]:
    """`>=2.0.0 <4.0.0` (space- or comma-separated comparators; `1.0` reads as 1.0.0) -> (lower, upper). None when not readable."""
    if not isinstance(text, str) or not text.strip():
        return None
    lower: Bound = None
    upper: Bound = None
    for term in re.split(r"[\s,]+", text.strip()):
        m = _RANGE_TERM_RE.match(term)
        if not m:
            return None
        parts = [int(p) for p in m.group(2).split(".")]
        v = tuple(parts + [0] * (3 - len(parts)))
        op = m.group(1) or "=="
        if op in ("==", "="):
            if lower is not None or upper is not None:
                return None
            lower, upper = (v, True), (v, True)
        elif op in (">=", ">"):
            if lower is not None:
                return None
            lower = (v, op == ">=")
        else:
            if upper is not None:
                return None
            upper = (v, op == "<=")
    return lower, upper


def range_widened(old: str, new: str) -> bool:
    """True when every version `old` accepts, `new` accepts too, and the two differ."""
    a, b = parse_range(old), parse_range(new)
    if a is None or b is None or a == b:
        return False
    (alo, ahi), (blo, bhi) = a, b
    lower_ok = blo is None or (alo is not None and (blo[0] < alo[0] or (blo[0] == alo[0] and (blo[1] or not alo[1]))))
    upper_ok = bhi is None or (ahi is not None and (bhi[0] > ahi[0] or (bhi[0] == ahi[0] and (bhi[1] or not ahi[1]))))
    return lower_ok and upper_ok


def reclassify_dependency_ranges(diff: dict, old_doc: dict, new_doc: dict) -> dict:
    """Mark each `breaking` change of `/dependencies/N/version_range` that only widened the range (same dependency skill_id) `additive`
    (rule `range-widened`), then classify again: breaking if any change still is, else additive if any is, else as si said."""
    changes = diff.get("changes")
    if not isinstance(changes, list):
        return diff
    old_deps = old_doc.get("dependencies") if isinstance(old_doc, dict) else None
    new_deps = new_doc.get("dependencies") if isinstance(new_doc, dict) else None
    touched = False
    for c in changes:
        m = _DEPENDENCY_RANGE_RE.match(str(c.get("pointer") or ""))
        if not m or c.get("effect") != "breaking" or not isinstance(old_deps, list) or not isinstance(new_deps, list):
            continue
        i = int(m.group(1))
        if i >= len(old_deps) or i >= len(new_deps) or not isinstance(old_deps[i], dict) or not isinstance(new_deps[i], dict):
            continue
        o, n = old_deps[i], new_deps[i]
        if o.get("skill_id") != n.get("skill_id") or not range_widened(o.get("version_range"), n.get("version_range")):
            continue
        c["effect"], c["rule"] = "additive", "range-widened"
        c["detail"] = f"{n.get('skill_id')} version range widened: {o.get('version_range')} -> {n.get('version_range')}"
        touched = True
    if touched:
        effects = [c.get("effect") for c in changes]
        diff["classification"] = "breaking" if "breaking" in effects else "additive" if "additive" in effects else diff.get("classification")
        diff.setdefault("notes", []).append("repin.py: a dependency version range that only widened is additive (ADR-117 addendum 2)")
    return diff


# ---- the lock ------------------------------------------------------------------------------------------------------------------------

def pinned_tag(sib: Sibling, version: str, tags: Dict[str, str]) -> Tuple[str, str]:
    for p in sib.tag_prefixes:
        if p + version in tags:
            return p + version, tags[p + version]
    raise SystemExit(f"{sib.name}: no release tag for the pinned version {version} ({', '.join(p + version for p in sib.tag_prefixes)})")


def lock_entry(root: Path, sib: Sibling, tag: str, sha: str) -> dict:
    pin, version = current_pin(root, sib)
    return {"repo": f"{OWNER}/{sib.repo}", "tag": tag, "sha": sha, "version": version, "pin": pin.relative_to(root).as_posix()}


def write_lock(root: Path, entries: Dict[str, dict]) -> None:
    doc = {"format": LOCK_FORMAT,
           "note": "Written by scripts/repin.py (ADR-117): the release each sibling Skill's pin was taken from. "
                   "Integration clones each sibling at this tag and checks the commit.",
           "siblings": {k: entries[k] for k in sorted(entries)}}
    path = root / LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def read_lock(root: Path) -> dict:
    return json.loads((root / LOCK_PATH).read_text(encoding="utf-8"))


def check_lock(root: Path) -> List[str]:
    """Offline: every sibling has a lock entry, and each entry names the pinned version, its pin file and a tag of that version."""
    problems: List[str] = []
    try:
        lock = read_lock(root)
    except (OSError, ValueError) as e:
        return [f"{LOCK_PATH}: {e}"]
    if lock.get("format") != LOCK_FORMAT:
        problems.append(f"{LOCK_PATH}: format {lock.get('format')!r}, expected {LOCK_FORMAT!r}")
    entries = lock.get("siblings", {})
    for extra in sorted(set(entries) - set(BY_NAME)):
        problems.append(f"{extra}: in the lock but not a known sibling")
    for sib in SIBLINGS:
        e = entries.get(sib.name)
        if e is None:
            problems.append(f"{sib.name}: no lock entry")
            continue
        pin, version = current_pin(root, sib)
        want = {"repo": f"{OWNER}/{sib.repo}", "version": version, "pin": pin.relative_to(root).as_posix()}
        for k, v in want.items():
            if e.get(k) != v:
                problems.append(f"{sib.name}: lock {k} {e.get(k)!r}, the pin says {v!r}")
        if tag_version(str(e.get("tag", "")), sib.tag_prefixes) != version:
            problems.append(f"{sib.name}: lock tag {e.get('tag')!r} is not a release tag of {version}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(e.get("sha", ""))):
            problems.append(f"{sib.name}: lock sha {e.get('sha')!r} is not a full commit id")
    return problems


# ---- update --------------------------------------------------------------------------------------------------------------------------

@dataclass
class Update:
    sibling: str
    old: str
    new: str
    tag: str
    sha: str
    classification: str
    decision: Decision
    changes: List[dict] = field(default_factory=list)
    readers: Dict[str, int] = field(default_factory=dict)
    leftovers: List[str] = field(default_factory=list)


def apply_update(root: Path, sib: Sibling, tag: str, sha: str, data: bytes, diff: dict, today: str) -> Update:
    old_path, old = current_pin(root, sib)
    new = tag_version(tag, sib.tag_prefixes) or ""
    classification = str(diff.get("classification"))
    gate = read_gate((root / sib.gate_file).read_text(encoding="utf-8")) if sib.gate_file else None
    decision = decide(old, new, classification, gate)
    new_path = old_path.with_name(f"{sib.pin_prefix}{new}.json")
    old_path.unlink()
    new_path.write_bytes(data)
    rw = rewrite_readers(root, sib, old, new, sha)
    if decision.kind == "widen" and sib.gate_file:
        gp = root / sib.gate_file
        note = (f"{new} ({repo_url(sib.repo)} {tag}, automatic re-pin {today}, ADR-117): si contract-diff {classification}; "
                f"the gate takes {decision.new_prefix}")
        gp.write_bytes(widen_gate(gp.read_bytes().decode("utf-8"), decision.new_prefix or "", note).encode("utf-8"))
        rw.changed[sib.gate_file] = rw.changed.get(sib.gate_file, 0) + 2
        for rel in GATE_TESTS:
            gt = root / rel
            if gt.exists():
                text, n = move_refusal_probe(gt.read_bytes().decode("utf-8"), sib, decision.new_prefix or "")
                if n:
                    gt.write_bytes(text.encode("utf-8"))
                    rw.changed[rel] = rw.changed.get(rel, 0) + n
    changes = [c for c in diff.get("changes", []) if c.get("effect") != "info"]
    return Update(sib.name, old, new, tag, sha, classification, decision, changes, rw.changed, rw.leftovers)


def is_test_line(leftover: str) -> bool:
    """A leftover (`file:line: text`) in tests/ or evals/: possibly what a test expects of the pin."""
    return leftover.startswith(("tests/", "evals/"))


def pr_body(updates: List[Update], lock_changed: bool, skipped: Sequence[str] = ()) -> str:
    review = [u for u in updates if u.decision.kind == "review"]
    out = ["Automatic re-pin of sibling Skill contracts from their release tags (`scripts/repin.py update`, ADR-117).", ""]
    if review:
        out += ["**A person must review this pull request**: " + ", ".join(f"{u.sibling} ({'; '.join(u.decision.reasons)})" for u in review)
                + ". The check `repin review` stays red while the label `repin: needs review` is on.", ""]
    expectations = [(u, l) for u in updates for l in u.leftovers if is_test_line(l)]
    if expectations:
        out += ["**Check before merging: lines in tests/ or evals/ still name an old release.** The rewrite could not place them. One that "
                "is history (a release the gate still takes, a prose note) can stay; one that is what a test expects of the pin fails the "
                "unit tests (see `scripts/repin.py verify` above) and should read the pin instead (`tests/pinned_versions.py`, ADR-117 "
                "addendum 1):", ""]
        out += [f"- {u.sibling} {u.old}: `{l}`" for u, l in expectations[:30]]
        out.append("")
    out += ["| Skill | pin | tag | commit | si contract-diff | decision |", "|---|---|---|---|---|---|"]
    for u in updates:
        out.append(f"| {u.sibling} | {u.old} → {u.new} | `{u.tag}` | `{u.sha[:7]}` | {u.classification} | {u.decision.kind}: {'; '.join(u.decision.reasons)} |")
    for u in updates:
        if u.changes:
            out += ["", f"<details><summary>{u.sibling}: {len(u.changes)} non-informational change(s)</summary>", ""]
            out += [f"- `{c.get('pointer')}` {c.get('effect')} ({c.get('rule')}): {c.get('detail')}" for c in u.changes[:50]]
            out += ["", "</details>"]
        rest = [l for l in u.leftovers if not is_test_line(l)]
        if rest:
            out += ["", f"{u.sibling}: lines outside tests/ and evals/ that still name {u.old}, left unchanged (history, or a reader the "
                    "rewrite could not place; a failing test points at the second kind):", ""]
            out += [f"- `{l}`" for l in rest[:30]]
    out += ["", f"Each new pin is the tag's committed `{CONTRACT_IN_TAG}`, byte for byte; the old pin file is replaced and its readers "
            "(adapters, agent modules, fakes, tests and evals) move with it."]
    if skipped:
        out += ["", "Not re-pinned:", ""] + [f"- {x}" for x in skipped]
    if lock_changed:
        out.append(f"`{LOCK_PATH.as_posix()}` names the new tags and commits; integration runs against them.")
    return "\n".join(out) + "\n"


def cmd_update(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    names = args.only or [s.name for s in SIBLINGS]
    cache = Path(tempfile.mkdtemp(prefix="repin-cache-"))
    files = TagFiles(cache)
    today = args.today or __import__("datetime").date.today().isoformat()
    lock = read_lock(root).get("siblings", {}) if (root / LOCK_PATH).exists() else {}
    updates: List[Update] = []
    skipped: List[str] = []
    for name in names:
        sib = BY_NAME[name]
        tags = ls_remote_tags(sib.repo)
        pin_path, pinned = current_pin(root, sib)
        found = latest_tag(tags, sib.tag_prefixes)
        if found is None:
            print(f"{name}: no release tag, pin {pinned} kept", file=sys.stderr)
            continue
        tag, version = found
        if parse_version(version) <= parse_version(pinned):
            if name not in lock or lock[name].get("version") != pinned:
                t, sha = pinned_tag(sib, pinned, tags)
                lock[name] = lock_entry(root, sib, t, sha)
            print(f"{name}: {pinned} is the latest release", file=sys.stderr)
            continue
        try:
            data = files.show(sib.repo, tag, CONTRACT_IN_TAG, tags[tag])
        except (subprocess.CalledProcessError, TagMoved) as e:
            why = str(e) if isinstance(e, TagMoved) else f"no {CONTRACT_IN_TAG} at the tag"
            print(f"{name}: {tag}: {why}; pin {pinned} kept", file=sys.stderr)
            skipped.append(f"{name} {tag}: {why}; pin {pinned} kept")
            continue
        bad = validate_contract(data, version, json.loads(pin_path.read_text(encoding="utf-8")))
        if bad:
            print(f"{name}: {tag}:{CONTRACT_IN_TAG} {bad}; pin {pinned} kept", file=sys.stderr)
            skipped.append(f"{name} {tag}: {CONTRACT_IN_TAG} {bad}; pin {pinned} kept")
            continue
        with tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False, dir=cache) as tmp:
            tmp.write(data)
        diff = contract_diff(args.si, pin_path, Path(tmp.name))
        u = apply_update(root, sib, tag, tags[tag], data, diff, today)
        lock[name] = lock_entry(root, sib, tag, tags[tag])
        updates.append(u)
        print(f"{name}: {u.old} -> {u.new} ({u.classification}, {u.decision.kind})", file=sys.stderr)
    before = (root / LOCK_PATH).read_bytes() if (root / LOCK_PATH).exists() else b""
    write_lock(root, lock)
    lock_changed = (root / LOCK_PATH).read_bytes() != before
    report = {"updates": [{"sibling": u.sibling, "old": u.old, "new": u.new, "tag": u.tag, "sha": u.sha, "classification": u.classification,
                           "decision": u.decision.kind, "reasons": u.decision.reasons, "readers": u.readers, "leftovers": u.leftovers}
                          for u in updates],
              "skipped": skipped, "review": any(u.decision.kind == "review" for u in updates), "lock_changed": lock_changed,
              "review_reasons": [f"{u.sibling} {u.old} -> {u.new}: {'; '.join(u.decision.reasons)}" for u in updates if u.decision.kind == "review"]}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.pr_body:
        Path(args.pr_body).write_text(pr_body(updates, lock_changed, skipped), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("review", "lock_changed")} | {"updated": [u.sibling for u in updates]}))
    return 0


# ---- verify: the unit tests on the re-pinned tree -------------------------------------------------------------------------------------

UNIT_RUN_RE = re.compile(r"^\s*-\s*run:\s*python -m unittest -v ((?:tests/\S+\.py\s*)+)$", re.M)
FAILED_TEST_RE = re.compile(r"^(?:FAIL|ERROR): (\w+) \(([\w.]+)\)", re.M)
FAILED_EVAL_RE = re.compile(r"^FAIL (\S+)", re.M)


def unit_commands(root: Path) -> List[List[str]]:
    """The unit job's test commands, as .github/workflows/tests.yml runs them, then the evals."""
    text = (root / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    cmds = [[sys.executable, "-m", "unittest", *m.group(1).split()] for m in UNIT_RUN_RE.finditer(text)]
    if (root / "evals" / "run.py").is_file():
        cmds.append([sys.executable, "evals/run.py"])
    return cmds


def failed_ids(cmd: Sequence[str], output: str, returncode: int) -> List[str]:
    """Test ids (`tests.test_x.Class.test_y`, both unittest spellings) or eval cases (`evals <case>`) that failed in one command."""
    if returncode == 0:
        return []
    if cmd[-1].endswith("run.py"):
        ids = [f"evals {m.group(1)}" for m in FAILED_EVAL_RE.finditer(output)]
    else:
        ids = [where if where.endswith("." + name) else f"{where}.{name}" for name, where in FAILED_TEST_RE.findall(output)]
    return ids or [f"{' '.join(Path(c).name if i == 0 else c for i, c in enumerate(cmd))}: exit {returncode}, no test id"]


def run_unit(root: Path, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> List[str]:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(root / "src"), os.environ.get("PYTHONPATH")])))
    failures: List[str] = []
    for cmd in unit_commands(root):
        r = runner(cmd, cwd=str(root), env=env, capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        failed = failed_ids(cmd, out, r.returncode)
        if failed:   # the run's log keeps why (the body lists the ids only)
            print(f"{' '.join(cmd[1:])}: exit {r.returncode}\n" + "\n".join(out.splitlines()[-200:]), file=sys.stderr)
        failures += failed
    return failures


def verify_section(failures: Sequence[str], report: dict) -> str:
    if not failures:
        return "The unit tests and the evals pass on this re-pin (`scripts/repin.py verify`).\n\n"
    out = [f"**A person must review this pull request: {len(failures)} unit test(s) or eval(s) fail on this re-pin** (`scripts/repin.py "
           "verify`). The pull request is a draft and labelled `repin: needs review`; fix the tests on this branch (a test that expects "
           "the pinned release reads it from the pin: `tests/pinned_versions.py`, ADR-117 addendum 1), then remove the label and mark it "
           "ready.", ""]
    out += [f"- `{f}`" for f in failures[:40]]
    if len(failures) > 40:
        out.append(f"- ... and {len(failures) - 40} more")
    lines = [l for u in report.get("updates", []) for l in u.get("leftovers", []) if is_test_line(l)]
    if lines:
        out += ["", "Lines in tests/ and evals/ the rewrite left on an old release (the likely cause):", ""] + [f"- `{l}`" for l in lines[:30]]
    return "\n".join(out) + "\n\n"


def cmd_verify(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    failures = run_unit(root)
    report["unit_failures"] = failures
    if failures:
        report["review"] = True
        report.setdefault("review_reasons", []).append(f"{len(failures)} unit test(s) or eval(s) fail on the re-pin: " + ", ".join(failures[:10])
                                                       + (" ..." if len(failures) > 10 else ""))
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.pr_body:
        body = Path(args.pr_body).read_text(encoding="utf-8")
        Path(args.pr_body).write_text(verify_section(failures, report) + body, encoding="utf-8")
    print(json.dumps({"review": report["review"], "unit_failures": failures}))
    return 0


def cmd_lock(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    entries = {}
    for sib in SIBLINGS:
        _, version = current_pin(root, sib)
        tag, sha = pinned_tag(sib, version, ls_remote_tags(sib.repo))
        entries[sib.name] = lock_entry(root, sib, tag, sha)
    write_lock(root, entries)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    problems = check_lock(Path(args.root).resolve())
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


def cmd_clone(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    lock = read_lock(root)["siblings"]
    dest = Path(args.dest).resolve()
    names = [s.name for s in SIBLINGS if (not args.only or s.name in args.only) and s.name not in (args.skip or [])]
    summary = ["| sibling | ref | commit | contract/contract.json |", "|---|---|---|---|"]
    for name in names:
        e, sib = lock[name], BY_NAME[name]
        target = dest / sib.repo
        if args.main:
            _git("clone", "-q", "--depth", "1", repo_url(sib.repo), str(target))
            ref = "main"
        else:
            _git("clone", "-q", "--depth", "1", "--branch", e["tag"], repo_url(sib.repo), str(target))
            ref = e["tag"]
        head = _git("rev-parse", "HEAD", cwd=target).strip()
        if not args.main and head != e["sha"]:
            raise SystemExit(f"{name}: {e['tag']} is {head}, the lock says {e['sha']} (the tag moved: re-run `scripts/repin.py lock`)")
        committed = target / CONTRACT_IN_TAG
        if not committed.is_file() and (args.main or (name, e["tag"]) in NO_COMMITTED_CONTRACT):
            same, state = True, "none committed (the pin is checked live by the tests)"
        elif not committed.is_file():
            raise SystemExit(f"{name}: {e['tag']} has no {CONTRACT_IN_TAG} to compare with the pin {e['pin']}")
        else:
            same = committed.read_bytes() == (root / e["pin"]).read_bytes()
            state = "= pin" if same else f"differs from the pin {e['version']}"
        if not args.main and not same:
            raise SystemExit(f"{name}: {e['tag']}:{CONTRACT_IN_TAG} is not byte for byte the pin {e['pin']}")
        print(f"{sib.repo} {ref} {head} contract {state}")
        summary.append(f"| {sib.repo} | {ref} | `{head[:7]}` | {state} |")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as f:
            f.write("\n".join(summary) + "\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("update")
    up.add_argument("--only", action="append", choices=sorted(BY_NAME))
    up.add_argument("--si", default="si")
    up.add_argument("--report")
    up.add_argument("--pr-body")
    up.add_argument("--today")
    ve = sub.add_parser("verify")
    ve.add_argument("--report", required=True)
    ve.add_argument("--pr-body")
    sub.add_parser("lock")
    sub.add_parser("check")
    cl = sub.add_parser("clone")
    cl.add_argument("--dest", required=True)
    cl.add_argument("--main", action="store_true")
    cl.add_argument("--only", action="append", choices=sorted(BY_NAME))
    cl.add_argument("--skip", action="append", choices=sorted(BY_NAME))
    args = ap.parse_args(argv)
    return {"update": cmd_update, "verify": cmd_verify, "lock": cmd_lock, "check": cmd_check, "clone": cmd_clone}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
