"""Paper edit from a script (ADR-139, roadmap step 28): the script's lines are the cut order of a take assembly (ADR-053).

    script file (one line per line of the script) + one input per take → each take placed on the script line its words are most like
      → the takes of each line scored as in ADR-053 (coverage against the script line) → the best take per line, in script order
      → lines with no take listed as missing; a pick whose words differ from its script line CONFIRM, the script line and the spoken words side by
        side (approve: keep the take as said; reject: the line is cut, and listed for a retake)

Vocabulary and arithmetic over recorded transcripts only (rule `script_lines@1`), no execution and no AI:

- **Requirement**: `edit.takes.script=<file>` with `edit.takes=true`: a UTF-8 text file, one script line per line (blank lines and lines starting with
  `#` are skipped), at most 500 lines of at most 500 characters.
- **Placing** (`place_takes`): each take's cleaned words (ADR-053's comparison form: folded, fillers removed, punctuation and spaces dropped) are
  compared with every script line's (difflib ratio). A take goes to the line it is most like when the ratio reaches `edit.takes.similarity`
  (ties: the earlier line); a take like no line is not used and is listed (`unplaced`). Lines keep the script's order and number.
- **Differences** (`differences`): the pick's words against its script line, word by word (character by character for CJK text), fillers and
  punctuation ignored; each replaced, missing or added run is one difference `{script, spoken}`.

Transcription-skill does not offer a script alignment yet (the roadmap's "transcription-skill's script alignment"); the placing above uses the
same comparison the take grouping already makes here. When the Skill publishes one, the placing moves there.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RULE_VERSION = "script_lines@1"
KEY = "edit.takes.script"
MAX_LINES = 500
MAX_LINE_CHARS = 500
MISSING_SUBJECT = "script.missing"
UNPLACED_SUBJECT = "script.unplaced"
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]")


class ScriptError(ValueError):
    """A script file that cannot be read as lines."""


def read_script(path: Any) -> List[str]:
    p = Path(str(path)).expanduser()
    if not p.is_file():
        raise ScriptError(f"{KEY}: no such file: {path}")
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ScriptError(f"{KEY}: {path} is not a UTF-8 text file ({type(e).__name__})") from None
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise ScriptError(f"{KEY}: {path} holds no script line")
    if len(lines) > MAX_LINES:
        raise ScriptError(f"{KEY}: {len(lines)} lines, at most {MAX_LINES}")
    long = next((n for n, ln in enumerate(lines, 1) if len(ln) > MAX_LINE_CHARS), None)
    if long is not None:
        raise ScriptError(f"{KEY}: line {long} is longer than {MAX_LINE_CHARS} characters")
    return lines


def place_takes(takes: List[Tuple[str, str]], script: List[str], threshold: float, clean: Any, similarity: Any) -> Tuple[List[Dict[str, Any]], List[int], List[Dict[str, Any]]]:
    """takes: [(asset id, cleaned text)] in input order → (lines with takes in script order, missing line numbers, unplaced takes).
    Each line is shaped like ADR-053's groups: {"line", "anchor", "takes", "similarity", "script", "script_clean"}."""
    cleaned = [clean(s) for s in script]
    placed: Dict[int, Dict[str, Any]] = {}
    unplaced: List[Dict[str, Any]] = []
    for aid, txt in takes:
        best, best_r = None, -1.0
        for n, sc in enumerate(cleaned):
            r = similarity(txt, sc)
            if r > best_r:
                best, best_r = n, r
        if best is None or best_r < threshold:
            unplaced.append({"asset_id": aid, "best_line": (best + 1) if best is not None else None, "similarity": max(best_r, 0.0)})
            continue
        ln = placed.setdefault(best, {"line": best + 1, "anchor": aid, "takes": [], "similarity": {}, "script": script[best], "script_clean": cleaned[best]})
        ln["takes"].append(aid)
        ln["similarity"][aid] = best_r
    lines = [placed[n] for n in sorted(placed)]
    missing = [n + 1 for n in range(len(script)) if n not in placed]
    return lines, missing, unplaced


def _tokens(text: str, strip_fillers: Any) -> List[str]:
    t = strip_fillers(unicodedata.normalize("NFKC", text))
    out: List[str] = []
    for w in re.findall(r"[\w']+", t):
        out += list(w) if _CJK_RE.search(w) else [w]
    return out


def differences(script_line: str, spoken: str, strip_fillers: Any) -> List[Dict[str, str]]:
    """The runs of the script line said differently: [{script, spoken}] (an empty side: missing or added words)."""
    a, b = _tokens(script_line, strip_fillers), _tokens(spoken, strip_fillers)
    out: List[Dict[str, str]] = []
    joiner = (lambda xs: "".join(xs)) if (_CJK_RE.search(script_line) or _CJK_RE.search(spoken)) else (lambda xs: " ".join(xs))
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "equal":
            out.append({"script": joiner(a[i1:i2]), "spoken": joiner(b[j1:j2])})
    return out


def report_lines(script: List[str], picks: List[Dict[str, Any]], missing: List[int], unplaced: List[Dict[str, Any]]) -> List[str]:
    """What the plan summary says: the order, the changed lines with both texts, the missing lines and the takes not used."""
    out = [f"Paper edit from the script ({len(script)} lines): {len(picks)} line(s) in script order"]
    for p in picks:
        if p.get("differences"):
            out.append(f"line {p['line']} said differently (CONFIRM: keep as said, or reject to cut it and retake): script “{p['script']}” / spoken “{p['spoken']}”")
    if missing:
        out.append("missing (no take): " + "; ".join(f"line {n} “{script[n - 1]}”" for n in missing))
    if unplaced:
        out.append("not used (like no script line): " + ", ".join(u["asset_id"] for u in unplaced))
    return out
