"""Entity memory (ADR-128): the people and proper nouns of a workspace or of one project, kept by the user and applied to every job.

    video-agent entities add / edit / remove / import / export / list / suggest  →  <workspace>/entities/global.json
                                                                                     <workspace>/entities/projects/<project>.json
      → the next plan reads the global store and the project's store (project wins for the same id), records what it used
        (request.args.entities) and applies it where the job already uses the matching Skill input:
          transcription  domain_vocabulary {speakers, terms} + corrections [{from: alias, to: canonical}]   (transcription-skill 0.13.0+)
          subtitles      the cues are the corrected transcript; `subtitle.emphasis=entities` (burn) highlights the names (subtitle-skill emphasis words)
          lower third    `motion.lower_third=<a name or an alias>` → the canonical spelling, and the entity's title when the plan gives none
          dubbing        the entities' readings in the dub's language → a dubbing/pronunciation@1 file passed as `--pronunciation` (EXPERIMENTAL)
        and says so in the plan summary ("entities: 3 entities applied to transcription vocabulary").

- **One entity**: `id` (a slug), `canonical` (the spelling to use), `type` (person | organization | product | place | term), `aliases` (other
  spellings and common mis-hearings), `title` (role, for a lower third), `pronunciation` ([{lang, reading, ipa?, accent?}]), `speaker`
  ({speaker_id?, voiceprint?}: diarization-skill's anonymous id or the path of a diarization-skill/voiceprint@1 enrollment), `note`.
- **Files**: JSON validated by video_agent/schemas/entity-store.schema.json on every read and write (a hand edit that breaks it is refused with the
  path of the bad field, never guessed around), written atomically. CSV import / export with the same fields (`|` between list items; a literal `|` or `\\` in a value is written `\\|` / `\\\\`).
  `$VIDEO_AGENT_ENTITIES_DIR` replaces `<workspace>/entities`.
- **Nothing is added automatically**: `suggest` lists candidates from a plan's transcripts (capitalised tokens that recur mid-sentence,
  low-confidence words that look like names) and the user adds the ones they want with `add`.
- **Privacy**: the names stay in the workspace. A plan sends them only to the local Skills that use them (transcription when every engine the
  request can reach runs locally, subtitle-skill, motion-graphics-skill, dubbing-skill), never to an AI provider (media/analysis.py
  scrub_for_ai drops the vocabulary from what an AI provider sees), and never through a remote recognition engine.
- **Empty store, unchanged plan**: with no entity in either store the plan records nothing and sends nothing new.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import now_iso

SCHEMA = "video-agent/entity-store@1"
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "entity-store.schema.json"   # shipped inside the package (package-data)
TYPES = ("person", "organization", "product", "place", "term")
ENV_DIR = "VIDEO_AGENT_ENTITIES_DIR"
PROJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
SPEAKER_RE = re.compile(r"^SPEAKER_\d{2,}$")
VOICEPRINT_SCHEMA = "diarization-skill/voiceprint@1"
PRONUNCIATION_SCHEMA = "dubbing/pronunciation@1"
CSV_COLUMNS = ("id", "canonical", "type", "aliases", "title", "pronunciation", "ipa", "accent", "speaker_id", "voiceprint", "note")
FIELDS = ("canonical", "type", "aliases", "title", "pronunciation", "speaker", "note")
RULE = "entities@1"
# transcription-skill's own limits (vocab.py): at most 200 terms, 500 correction rules, 100 characters a term
MAX_TERMS = 200
MAX_CORRECTIONS = 500
# dubbing-skill's pronunciation dictionary limits (pronunciation.py): a term of at most 64 characters
MAX_PRONUNCIATION_TERM = 64
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


class EntityError(ValueError):
    """A store, entity or file that is refused (nothing is written)."""


# ---- scope and files
def check_project(project: Any) -> str:
    p = str(project or "").strip()
    if not PROJECT_RE.match(p):
        raise EntityError(f"project must be 1..64 letters, digits, '.', '_' or '-' (a name, never a path), got {project!r}")
    return p


def store_root(workspace: str) -> Path:
    return Path(os.environ.get(ENV_DIR) or Path(workspace) / "entities")


def store_name(project: Optional[str]) -> str:
    return f"project:{project}" if project else "global"


def slug(text: str) -> str:
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:64].strip("-")
    if not s:   # a name in a script without a Latin transliteration here: a stable id from its text
        s = "e-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return s


def _schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_doc(doc: Any) -> List[str]:
    """Schema errors of a store document (each with the path of the field), then the rules a schema cannot state: unique ids, no spelling
    (canonical or alias, ignoring case) naming two entities, and one pronunciation a language (ignoring case, as normalize_entity)."""
    import jsonschema
    v = jsonschema.Draft202012Validator(_schema())
    errs = [_schema_error(e) for e in sorted(v.iter_errors(doc), key=lambda e: list(e.absolute_path))]
    if errs:
        return errs
    ids: Dict[str, int] = {}
    spellings: Dict[str, str] = {}
    for i, e in enumerate(doc["entities"]):
        if e["id"] in ids:
            errs.append(f"entities/{i}/id: {e['id']!r} is used by entity {ids[e['id']]} too")
        ids.setdefault(e["id"], i)
        langs: Dict[str, int] = {}
        for j, p in enumerate(e.get("pronunciation") or []):
            k = p["lang"].lower()
            if k in langs:
                errs.append(f"entities/{i}/pronunciation/{j}/lang: {p['lang']!r} repeats the language of pronunciation {langs[k]} "
                            "(one pronunciation a language)")
            langs.setdefault(k, j)
        for s in [e["canonical"]] + list(e.get("aliases") or []):
            k = s.casefold()
            if k in spellings and spellings[k] != e["id"]:
                errs.append(f"entities/{i}: {s!r} already names entity {spellings[k]!r}")
            spellings.setdefault(k, e["id"])
    return errs


def _schema_error(e: Any) -> str:
    """One schema error as `<path of the field>: <reason>`, in words for the rules whose jsonschema message names no field."""
    where = "/".join(str(p) for p in e.absolute_path) or "(top)"
    rule = list(e.schema_path)
    if rule[:1] == ["then"]:
        return "project: a project store names its project"
    if rule[:1] == ["else"]:
        return "project: a global store names no project"
    if e.validator == "pattern" and isinstance(e.instance, str) and not e.instance.strip():
        return f"{where}: is blank (only whitespace)"
    return f"{where}: {e.message}"


class EntityStore:
    """One store file: global, or one project's."""

    def __init__(self, workspace: str, project: Optional[str] = None):
        self.project = check_project(project) if project else None
        root = store_root(workspace)
        self.path = root / "projects" / f"{self.project}.json" if self.project else root / "global.json"
        self.name = store_name(self.project)

    def empty(self) -> Dict[str, Any]:
        return {"schema": SCHEMA, "scope": "project" if self.project else "global", **({"project": self.project} if self.project else {}), "entities": []}

    def load(self) -> Dict[str, Any]:
        return self.read()[0]

    def read(self) -> Tuple[Dict[str, Any], Optional[str]]:
        """The validated document and the sha256 of the very bytes it was parsed from (None when there is no file): one read, so a writer
        replacing the store in between cannot pair one version's digest with another version's entities."""
        if not self.path.exists():
            return self.empty(), None
        try:
            data = self.path.read_bytes()
            doc = json.loads(data.decode("utf-8"))
        except FileNotFoundError:
            return self.empty(), None
        except (OSError, ValueError) as e:
            raise EntityError(f"{self.path}: not a readable JSON file ({e}); fix or remove it") from e
        errs = validate_doc(doc)
        if errs:
            raise EntityError(f"{self.path} is not a valid {SCHEMA} store: " + "; ".join(errs[:10]))
        if (doc.get("project") or None) != self.project:
            raise EntityError(f"{self.path} is the store of {store_name(doc.get('project'))}, not {self.name}")
        return doc, hashlib.sha256(data).hexdigest()

    @contextmanager
    def locked(self, timeout: float = 10.0, stale: float = 120.0):
        """One writer at a time, from load to save: an exclusive lock file next to the store (os.O_EXCL works on every OS). A lock older
        than `stale` seconds is a crashed writer's and is taken over; waiting longer than `timeout` refuses the write."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + ".lock")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                break
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > stale:
                        lock.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise EntityError(f"{self.path} is being changed by another process (lock {lock.name}); try again")
                time.sleep(0.05)
        try:
            yield
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass

    def save(self, doc: Dict[str, Any]) -> None:
        doc = dict(doc, updated_at=now_iso())
        errs = validate_doc(doc)
        if errs:
            raise EntityError(f"refusing to write {self.path}: " + "; ".join(errs[:10]))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=1, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ---- CRUD
    def list(self, type_: Optional[str] = None) -> List[Dict[str, Any]]:
        return [e for e in self.load()["entities"] if type_ is None or e["type"] == type_]

    def get(self, ref: str) -> Dict[str, Any]:
        doc = self.load()
        return doc["entities"][_index(doc, ref)]

    def add(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        with self.locked():
            return self._add(raw)

    def _add(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        doc = self.load()
        e = normalize_entity(raw)
        e.setdefault("id", _free_id(doc, slug(e["canonical"])))
        if any(x["id"] == e["id"] for x in doc["entities"]):
            raise EntityError(f"entity {e['id']!r} exists in {self.name}; use `entities edit {e['id']}`")
        now = now_iso()
        e.update(added_at=now, updated_at=now)
        doc["entities"].append(_ordered(e))
        self.save(doc)
        return e

    def edit(self, ref: str, changes: Dict[str, Any], add_aliases: Iterable[str] = (), remove_aliases: Iterable[str] = ()) -> Dict[str, Any]:
        with self.locked():
            return self._edit(ref, changes, add_aliases, remove_aliases)

    def _edit(self, ref: str, changes: Dict[str, Any], add_aliases: Iterable[str] = (), remove_aliases: Iterable[str] = ()) -> Dict[str, Any]:
        doc = self.load()
        i = _index(doc, ref)
        cur = dict(doc["entities"][i])
        for k, v in changes.items():
            if k not in FIELDS:
                raise EntityError(f"unknown field {k!r} (fields: {', '.join(FIELDS)})")
            if v is None or v == "" or v == []:
                cur.pop(k, None)
            else:
                cur[k] = v
        al = list(cur.get("aliases") or [])
        for a in add_aliases:
            if a.casefold() not in {x.casefold() for x in al}:
                al.append(a)
        drop = {a.casefold() for a in remove_aliases}
        al = [a for a in al if a.casefold() not in drop]
        if al:
            cur["aliases"] = al
        else:
            cur.pop("aliases", None)
        e = normalize_entity(cur)
        e["updated_at"] = now_iso()
        doc["entities"][i] = _ordered(e)
        self.save(doc)
        return e

    def remove(self, ref: str) -> Dict[str, Any]:
        with self.locked():
            return self._remove(ref)

    def _remove(self, ref: str) -> Dict[str, Any]:
        doc = self.load()
        i = _index(doc, ref)
        gone = doc["entities"].pop(i)
        self.save(doc)
        return gone

    # ---- CSV
    def export_csv(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        w.writeheader()
        for e in self.load()["entities"]:
            w.writerow(entity_to_row(e))
        return buf.getvalue()

    def import_csv(self, text: str, replace: bool = False) -> Dict[str, Any]:
        with self.locked():
            return self._import_csv(text, replace)

    def _import_csv(self, text: str, replace: bool = False) -> Dict[str, Any]:
        """Rows by id (a row without one gets the slug of its canonical): new ids are added, existing ones take the row's fields. With
        `replace` the store becomes exactly the rows. All rows are checked before anything is written."""
        rows = []
        for r in rows_from_csv(text):
            e = normalize_entity(r)
            e.setdefault("id", slug(e["canonical"]))
            rows.append(e)
        return self._merge(rows, replace)

    def import_doc(self, doc: Any, replace: bool = False) -> Dict[str, Any]:
        """An entity store document (an `export --format json`), merged by id as a CSV is, with its entities as they are: validated against
        the schema, never re-normalized through CSV (two spellings that differ only in spacing stay two spellings)."""
        errs = validate_doc(doc) if isinstance(doc, dict) and "schema" in doc else [f"a JSON import is an entity store document ({SCHEMA})"]
        if errs:
            raise EntityError("; ".join(errs[:10]))
        rows = [{k: v for k, v in e.items() if k not in ("added_at", "updated_at")} for e in doc["entities"]]
        with self.locked():
            return self._merge(rows, replace)

    def _merge(self, rows: List[Dict[str, Any]], replace: bool) -> Dict[str, Any]:
        """Entities with ids, merged by id (under the lock); the destination document is validated before it is saved."""
        cur = self.load()
        doc = self.empty() if replace else cur
        before = {e["id"]: e for e in cur["entities"]}
        index = {e["id"]: i for i, e in enumerate(doc["entities"])}
        added, updated, unchanged = [], [], []
        now = now_iso()
        for r in rows:
            e = dict(r)
            old = before.get(e["id"])
            stamp = {"added_at": (old or {}).get("added_at") or now}
            if old is not None and _content(old) == _content(e):
                e = dict(old)
                unchanged.append(e["id"])
            else:
                e.update(stamp, updated_at=now)
                (updated if old is not None else added).append(e["id"])
            if e["id"] in index:
                doc["entities"][index[e["id"]]] = _ordered(e)
            else:
                index[e["id"]] = len(doc["entities"])
                doc["entities"].append(_ordered(e))
        removed = sorted(set(before) - set(index)) if replace else []
        self.save(doc)
        return {"store": self.name, "path": str(self.path), "added": added, "updated": updated, "unchanged": unchanged, "removed": removed}


def _content(e: Dict[str, Any]) -> Dict[str, Any]:
    return {k: e[k] for k in ("id",) + FIELDS if k in e}


def _ordered(e: Dict[str, Any]) -> Dict[str, Any]:
    order = ("id",) + FIELDS + ("added_at", "updated_at")
    return {k: e[k] for k in order if k in e}


def _index(doc: Dict[str, Any], ref: str) -> int:
    ref_s = str(ref or "").strip()
    for i, e in enumerate(doc["entities"]):
        if e["id"] == ref_s:
            return i
    hits = [i for i, e in enumerate(doc["entities"]) if ref_s.casefold() in {s.casefold() for s in [e["canonical"]] + list(e.get("aliases") or [])}]
    if len(hits) == 1:
        return hits[0]
    raise EntityError(f"no entity {ref!r} (give its id, canonical spelling or an alias; `entities list` shows them)")


def _free_id(doc: Dict[str, Any], base: str) -> str:
    taken = {e["id"] for e in doc["entities"]}
    if base not in taken:
        return base
    n = 2
    while f"{base[:60]}-{n}" in taken:
        n += 1
    return f"{base[:60]}-{n}"


def _text(v: Any, what: str, limit: int = 100) -> str:
    if not isinstance(v, str):
        raise EntityError(f"{what} must be text, got {v!r}")
    t = " ".join(v.split())
    if not t:
        raise EntityError(f"{what} is empty")
    if len(t) > limit:
        raise EntityError(f"{what} is longer than {limit} characters")
    if _CTRL.search(t):
        raise EntityError(f"{what} contains control characters")
    return t


def normalize_entity(raw: Dict[str, Any]) -> Dict[str, Any]:
    """One entity checked and tidied (whitespace collapsed, aliases unique and not the canonical spelling itself). EntityError otherwise."""
    if not isinstance(raw, dict):
        raise EntityError("an entity is an object")
    unknown = sorted(set(raw) - set(("id",) + FIELDS + ("added_at", "updated_at")))
    if unknown:
        raise EntityError(f"unknown entity field(s): {', '.join(unknown)} (fields: id, {', '.join(FIELDS)})")
    e: Dict[str, Any] = {}
    if raw.get("id") not in (None, ""):
        if not isinstance(raw["id"], str) or not ID_RE.match(raw["id"]):
            raise EntityError(f"id must be 1..64 lower-case letters, digits, '.', '_' or '-', got {raw['id']!r}")
        e["id"] = raw["id"]
    e["canonical"] = _text(raw.get("canonical"), "canonical")
    t = str(raw.get("type") or "").strip().lower()
    if t not in TYPES:
        raise EntityError(f"type must be one of {', '.join(TYPES)}, got {raw.get('type')!r}")
    e["type"] = t
    aliases: List[str] = []
    for a in raw.get("aliases") or []:
        a = _text(a, "alias")
        if a == e["canonical"] or a.casefold() in {x.casefold() for x in aliases}:
            continue
        aliases.append(a)
    if len(aliases) > 50:
        raise EntityError("at most 50 aliases an entity")
    if aliases:
        e["aliases"] = aliases
    if raw.get("title") not in (None, ""):
        e["title"] = _text(raw["title"], "title", 200)
    pron = raw.get("pronunciation")
    if pron not in (None, [], ""):
        if not isinstance(pron, list):
            raise EntityError("pronunciation is a list of {lang, reading, ipa?, accent?}")
        out = []
        for p in pron:
            if not isinstance(p, dict) or set(p) - {"lang", "reading", "ipa", "accent"}:
                raise EntityError("a pronunciation is {lang, reading, ipa?, accent?}")
            lang = str(p.get("lang") or "")
            if not re.match(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$", lang):
                raise EntityError(f"pronunciation lang must be a BCP-47 tag (en, ja, en-US, ...), got {p.get('lang')!r}")
            q: Dict[str, Any] = {"lang": lang, "reading": _text(p.get("reading"), "pronunciation reading", 128)}
            if p.get("ipa") not in (None, ""):
                q["ipa"] = _text(p["ipa"], "pronunciation ipa", 128)
                if any(c in q["ipa"] for c in "[]()/"):
                    raise EntityError("pronunciation ipa cannot contain [ ] ( ) or /")
            if p.get("accent") not in (None, ""):
                acc = p["accent"]
                if isinstance(acc, str) and acc.strip().isdigit():
                    acc = int(acc)
                if isinstance(acc, bool) or not isinstance(acc, int) or not 0 <= acc <= 64:
                    raise EntityError("pronunciation accent is the accent nucleus in morae, an integer 0..64")
                q["accent"] = acc
            out.append(q)
        if len({p["lang"].lower() for p in out}) != len(out):
            raise EntityError("one pronunciation a language")
        e["pronunciation"] = out
    sp = raw.get("speaker")
    if sp not in (None, {}, ""):
        if not isinstance(sp, dict) or set(sp) - {"speaker_id", "voiceprint"}:
            raise EntityError("speaker is {speaker_id?, voiceprint?}")
        s: Dict[str, Any] = {}
        if sp.get("speaker_id") not in (None, ""):
            if not SPEAKER_RE.match(str(sp["speaker_id"])):
                raise EntityError(f"speaker_id is diarization-skill's anonymous id (SPEAKER_00, ...), got {sp['speaker_id']!r}")
            s["speaker_id"] = str(sp["speaker_id"])
        if sp.get("voiceprint") not in (None, ""):
            s["voiceprint"] = _text(sp["voiceprint"], "voiceprint", 1024)
        if s:
            e["speaker"] = s
    if raw.get("note") not in (None, ""):
        if not isinstance(raw["note"], str) or len(raw["note"]) > 500:
            raise EntityError("note is text of at most 500 characters")
        e["note"] = raw["note"]
    for k in ("added_at", "updated_at"):
        if isinstance(raw.get(k), str):
            e[k] = raw[k]
    return e


def check_voiceprint(path: str) -> str:
    """A speaker link to an enrolled voice: an existing diarization-skill/voiceprint@1 file (its absolute path is stored)."""
    p = Path(path).expanduser()
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise EntityError(f"voiceprint {path}: not a readable JSON file ({e})") from e
    if not isinstance(doc, dict) or doc.get("schema") != VOICEPRINT_SCHEMA:
        raise EntityError(f"voiceprint {path}: not a {VOICEPRINT_SCHEMA} document (diarization-skill `enroll` writes one)")
    return str(p.resolve())


# ---- CSV rows
def _escape(value: str) -> str:
    """One list item in a CSV cell: `\\` and `|` are escaped, so `|` between items is never read as part of one."""
    return value.replace("\\", "\\\\").replace("|", "\\|")


def _join(items: Iterable[str]) -> str:
    return "|".join(_escape(x) for x in items)


def _split(cell: Any) -> List[str]:
    """The items of a CSV list cell: split at each `|` that is not escaped, then `\\|` → `|` and `\\\\` → `\\` (the reverse of _join)."""
    items, cur, esc = [], [], False
    for ch in str(cell or ""):
        if esc:
            cur.append(ch if ch in "|\\" else "\\" + ch)
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == "|":
            items.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if esc:
        cur.append("\\")
    items.append("".join(cur))
    return [x.strip() for x in items if x.strip()]


def _pairs(cell: Any, what: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in _split(cell):
        if "=" not in item:
            raise EntityError(f"CSV {what}: {item!r} is not lang=value")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def entity_to_row(e: Dict[str, Any]) -> Dict[str, str]:
    pr = e.get("pronunciation") or []
    sp = e.get("speaker") or {}
    return {"id": e["id"], "canonical": e["canonical"], "type": e["type"], "aliases": _join(e.get("aliases") or []), "title": e.get("title", ""),
            "pronunciation": _join(f"{p['lang']}={p['reading']}" for p in pr), "ipa": _join(f"{p['lang']}={p['ipa']}" for p in pr if p.get("ipa")),
            "accent": _join(f"{p['lang']}={p['accent']}" for p in pr if p.get("accent") is not None),
            "speaker_id": sp.get("speaker_id", ""), "voiceprint": sp.get("voiceprint", ""), "note": e.get("note", "")}


def rows_from_csv(text: str) -> List[Dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    cols = [c.strip() for c in reader.fieldnames or []]
    if "canonical" not in cols or "type" not in cols:
        raise EntityError(f"the CSV needs a header with at least canonical and type (columns: {', '.join(CSV_COLUMNS)})")
    unknown = sorted(set(cols) - set(CSV_COLUMNS))
    if unknown:
        raise EntityError(f"unknown CSV column(s): {', '.join(unknown)} (columns: {', '.join(CSV_COLUMNS)})")
    out: List[Dict[str, Any]] = []
    for n, row in enumerate(reader, start=2):
        if None in row:   # DictReader keeps the fields past the header under None
            raise EntityError(f"CSV line {n}: {len(row[None])} field(s) more than the header's {len(cols)} column(s) (quote a value with a comma)")
        row = {(k or "").strip(): (v or "") for k, v in row.items()}
        if not any(v.strip() for v in row.values()):
            continue
        try:
            reading, ipa, accent = _pairs(row.get("pronunciation"), "pronunciation"), _pairs(row.get("ipa"), "ipa"), _pairs(row.get("accent"), "accent")
            stray = sorted((set(ipa) | set(accent)) - set(reading))
            if stray:
                raise EntityError(f"ipa / accent for {', '.join(stray)} without a reading in that language")
            pron = [{"lang": lang, "reading": r, **({"ipa": ipa[lang]} if lang in ipa else {}), **({"accent": accent[lang]} if lang in accent else {})} for lang, r in reading.items()]
            raw = {"id": row.get("id", "").strip() or None, "canonical": row.get("canonical", ""), "type": row.get("type", ""), "aliases": _split(row.get("aliases")),
                   "title": row.get("title", "").strip() or None, "pronunciation": pron or None,
                   "speaker": {"speaker_id": row.get("speaker_id", "").strip() or None, "voiceprint": row.get("voiceprint", "").strip() or None},
                   "note": row.get("note", "") or None}
            out.append(normalize_entity(raw))
        except EntityError as e:
            raise EntityError(f"CSV line {n}: {e}") from None
    ids = [r.get("id") or slug(r["canonical"]) for r in out]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise EntityError(f"the CSV names id(s) {', '.join(dup)} more than once")
    return out


# ---- what a plan uses
def snapshot(workspace: str, project: Optional[str]) -> Optional[Dict[str, Any]]:
    """The entities a plan applies: the global store's, then the project's (the project's wins for the same id), with each store's digest.
    None when both are empty (the plan then records nothing)."""
    stores = [EntityStore(workspace)] + ([EntityStore(workspace, project)] if project else [])
    merged: Dict[str, Dict[str, Any]] = {}
    used = []
    for st in stores:
        doc, sha = st.read()
        ents = doc["entities"]
        used.append({"store": st.name, "path": str(st.path), "sha256": sha, "entities": len(ents)})
        for e in ents:
            merged[e["id"]] = dict(_content(e), store=st.name)
    if not merged:
        return None
    # a spelling the project gives another entity wins over the global one (the project's knowledge is the nearer one)
    owner: Dict[str, str] = {}
    shadowed: List[str] = []
    for e in sorted(merged.values(), key=lambda x: x["store"] == "global"):
        keep = []
        for s in [e["canonical"]] + list(e.get("aliases") or []):
            if s.casefold() in owner and owner[s.casefold()] != e["id"]:
                shadowed.append(f"{s!r} of {e['id']} (named by {owner[s.casefold()]})")
                continue
            owner[s.casefold()] = e["id"]
            keep.append(s)
        if e.get("aliases"):
            e["aliases"] = [a for a in e["aliases"] if a in keep]
    return {"rule": RULE, "stores": used, "entities": sorted(merged.values(), key=lambda x: x["id"]), **({"shadowed": shadowed} if shadowed else {})}


def _cjk(s: str) -> bool:
    return bool(_CJK.search(s))


def _owners(entities: List[Dict[str, Any]]) -> Dict[str, str]:
    """Each spelling (casefolded) → the id of the entity it names: a project store's entity before a global one, as in snapshot, so a project
    alias that is a global entity's canonical spelling names the project's entity."""
    owner: Dict[str, str] = {}
    for e in sorted(entities, key=lambda x: x.get("store", "global") == "global"):
        for s in [e["canonical"]] + list(e.get("aliases") or []):
            owner.setdefault(s.casefold(), e["id"])
    return owner


def _owns(owner: Dict[str, str], e: Dict[str, Any], spelling: str) -> bool:
    return owner.get(spelling.casefold(), e["id"]) == e["id"]


def transcription_inputs(entities: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, str]]]]:
    """transcription-skill's own inputs: domain_vocabulary {speakers: people, terms: the rest, max_terms} and corrections [{from: alias, to:
    canonical, match: word | substring (CJK)}], within the Skill's limits (the first entities by id win when there are more). A spelling another
    entity owns (a global canonical spelling a project entity has as an alias) is neither a term nor a correction source of the shadowed one."""
    owner = _owners(entities)
    people = [e["canonical"] for e in entities if e["type"] == "person" and _owns(owner, e, e["canonical"])]
    others = [e["canonical"] for e in entities if e["type"] != "person" and _owns(owner, e, e["canonical"])]
    names = (people + others)[:MAX_TERMS]
    vocab = None
    if names:
        sp = [n for n in names if n in set(people)]
        tm = [n for n in names if n not in set(people)]
        vocab = {**({"speakers": sp} if sp else {}), **({"terms": tm} if tm else {}), "max_terms": len(names)}
        if any(re.search(r"[぀-ヿ㐀-鿿]", n) for n in names):
            vocab["japanese"] = True   # Japanese names: the Skill joins the terms with "、" in the decoder prompt
    rules: List[Dict[str, str]] = []
    seen = set()
    for e in entities:
        for a in e.get("aliases") or []:
            if a == e["canonical"] or a.casefold() in seen or len(rules) >= MAX_CORRECTIONS or not _owns(owner, e, a):
                continue
            seen.add(a.casefold())
            rules.append({"from": a, "to": e["canonical"], "match": "substring" if _cjk(a) else "word"})
    return vocab, (rules or None)


def emphasis_words(entities: List[Dict[str, Any]]) -> List[str]:
    """subtitle-skill emphasis `words` entries: subtitle-skill matches one drawn token, so each name gives its capitalised (or CJK) tokens."""
    out: List[str] = []
    for e in entities:
        if e["type"] == "term":
            continue
        for tok in e["canonical"].split():
            core = tok.strip(".,;:!?\"'()[]")
            if len(core) >= 2 and (core[0].isupper() or _cjk(core)) and core.casefold() not in {w.casefold() for w in out}:
                out.append(core)
    return out


def find(entities: List[Dict[str, Any]], name: Any) -> Optional[Dict[str, Any]]:
    """The entity a spelling names, compared as the store compares spellings (ignoring case, spacing kept: a JSON import can hold `A  B`
    and `A B` as two entities): the exact spelling first, then without the surrounding whitespace, then with every run of whitespace
    collapsed, only when that names one entity."""
    if not isinstance(name, str):
        return None
    owner = _owners(entities)
    want = owner.get(name.casefold()) or owner.get(name.strip().casefold())
    if want is None:
        k = " ".join(name.split()).casefold()
        hits = {i for s, i in owner.items() if " ".join(s.split()) == k}
        want = hits.pop() if len(hits) == 1 else None
    return next((e for e in entities if e["id"] == want), None)


def pronunciation_dictionary(entities: List[Dict[str, Any]], lang: str) -> Optional[Dict[str, Any]]:
    """A dubbing/pronunciation@1 dictionary of the entities' readings for `lang` (an entry for en serves en-US; one for en-US only en-US), the
    canonical spelling and its aliases each as a term. None when no entity has a reading for the language."""
    want = str(lang or "").lower()
    owner = _owners(entities)
    entries = []
    for e in entities:
        for p in e.get("pronunciation") or []:
            pl = p["lang"].lower()
            if not (want == pl or want.startswith(pl + "-")):
                continue
            for term in [e["canonical"]] + list(e.get("aliases") or []):
                if len(term) <= MAX_PRONUNCIATION_TERM and _owns(owner, e, term):
                    entries.append({"term": term, "lang": p["lang"], "reading": p["reading"], **{k: p[k] for k in ("ipa", "accent") if k in p}})
    if not entries:
        return None
    return {"schema": PRONUNCIATION_SCHEMA, "name": "entity memory", "entries": entries}


# ---- suggestions from a transcript (the user confirms; nothing is added here)
_WORD = re.compile(r"[^\W\d_][\w'’\-]*", re.UNICODE)
_END = (".", "!", "?", ":", ";", "。", "！", "？")
_COMMON = frozenset("""i i'm i've i'll i'd okay ok yes no so and but the a an this that these those we you they he she it our my your their
thank thanks hello hi good well right now then here there today what when where why how who which let let's also just very really
monday tuesday wednesday thursday friday saturday sunday january february march april may june july august september october november
december mr mrs ms dr""".split())


def suggest(transcripts: List[Dict[str, Any]], entities: List[Dict[str, Any]], min_count: int = 2, max_probability: float = 0.6,
            limit: int = 50) -> List[Dict[str, Any]]:
    """Candidate entities in transcripts: runs of capitalised words that are not at a sentence start, seen `min_count` times or more, and
    capitalised words the engine was not sure of (word probability below `max_probability`). Known spellings are left out. A list for a
    person to confirm with `entities add`; nothing is stored."""
    known = {s.casefold() for e in entities for s in [e["canonical"]] + list(e.get("aliases") or [])}
    found: Dict[str, Dict[str, Any]] = {}

    def note(text: str, why: str, at: Any, prob: Optional[float] = None) -> None:
        k = text.casefold()
        if k in known or k in _COMMON or len(text) < 2:
            return
        c = found.setdefault(k, {"text": text, "count": 0, "reasons": [], "first_at": at, "min_probability": None})
        c["count"] += 1
        if why not in c["reasons"]:
            c["reasons"].append(why)
        if prob is not None:
            c["min_probability"] = prob if c["min_probability"] is None else min(c["min_probability"], prob)

    for tr in transcripts:
        for seg in tr.get("segments") or []:
            text = str(seg.get("text") or "")
            toks = [(m.group(0), m.start()) for m in _WORD.finditer(text)]
            run: List[str] = []
            for i, (tok, pos) in enumerate(toks):
                before = text[:pos].rstrip()
                initial = not before or before.endswith(_END)
                if tok[0].isupper() and not initial and tok.casefold() not in _COMMON:
                    run.append(tok)
                else:
                    if run:
                        note(" ".join(run), "capitalised mid-sentence", seg.get("start"))
                    run = [tok] if tok[0].isupper() and initial and i + 1 < len(toks) and toks[i + 1][0][0].isupper() and tok.casefold() not in _COMMON else []
            if run:
                note(" ".join(run), "capitalised mid-sentence", seg.get("start"))
            for w in seg.get("words") or []:
                if not isinstance(w, dict):
                    continue
                p = w.get("probability", w.get("confidence"))
                core = str(w.get("text") or w.get("word") or "").strip().strip(".,;:!?\"'()")
                if isinstance(p, (int, float)) and not isinstance(p, bool) and p < max_probability and core[:1].isupper():
                    note(core, "low confidence", w.get("start", seg.get("start")), float(p))
    out = [c for c in found.values() if c["count"] >= min_count or "low confidence" in c["reasons"]]
    out.sort(key=lambda c: (-c["count"], c["text"].casefold()))
    for c in out:
        c["suggested"] = {"canonical": c["text"], "type": "person" if len(c["text"].split()) == 2 else "term"}
        if c["min_probability"] is None:
            del c["min_probability"]
    return out[:limit]


# ---- the plan record (request.args.entities): what a plan applies, decided once and re-used by its revisions
LOWER_THIRD_KEY = "motion.lower_third"
LOWER_THIRD_TITLE_KEY = "motion.lower_third.title"
EMPHASIS_KEY = "subtitle.emphasis"
EMPHASIS_VALUE = "entities"
DUB_LANG_KEY = "dub.lang"
DUB_CUES_KEY = "dub.cues"
DUB_PRONUNCIATION_KEY = "dub.pronunciation"


def pronunciation_file(workspace: str, doc: Dict[str, Any]) -> str:
    """The dictionary written into the workspace under its content's hash (the same readings give the same file). Written to a temporary file
    beside it and moved into place, so a reader never sees half a file; a file already there is reused only when its bytes are the expected
    ones, so a later plan repairs one an earlier failure left."""
    data = (json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=1) + "\n").encode("utf-8")
    d = Path(workspace) / "entities" / "pronunciation"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"pronunciation-{hashlib.sha256(data).hexdigest()[:16]}.json"
    try:
        if p.read_bytes() == data:
            return str(p.resolve())
    except OSError:
        pass
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(d))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return str(p.resolve())


def plan_record(snap: Dict[str, Any], reqs: Dict[str, Any], workspace: str) -> Dict[str, Any]:
    """What the plan does with the entities, from the requirements it was given: requirement values it rewrites (a lower third's spelling,
    `subtitle.emphasis=entities`), requirements it adds when none is given (the lower third's title, the dub's pronunciation dictionary),
    and one line per use for the plan summary. Transcription inputs are added by the Service, which knows whether a transcript is measured."""
    ents = snap["entities"]
    rec: Dict[str, Any] = dict(snap, requirements={}, rewrites={}, applied=[], withheld=[])
    name = reqs.get(LOWER_THIRD_KEY)
    if isinstance(name, str):
        e = find(ents, name)
        if e is not None:
            if name != e["canonical"]:
                rec["rewrites"][LOWER_THIRD_KEY] = {"from": name, "to": e["canonical"], "entity": e["id"]}
            if e.get("title") and LOWER_THIRD_TITLE_KEY not in reqs:
                rec["requirements"][LOWER_THIRD_TITLE_KEY] = {"value": e["title"], "entity": e["id"]}
            rec["applied"].append({"use": "lower_third", "entities": [e["id"]],
                                   "line": "entities: 1 entity applied to the lower third ("
                                           + ", ".join(["name" + (", spelling corrected" if name != e["canonical"] else "")]
                                                       + (["title from the entity"] if LOWER_THIRD_TITLE_KEY in rec["requirements"] else [])) + ")"})
    if reqs.get(EMPHASIS_KEY) == EMPHASIS_VALUE:
        words = emphasis_words(ents)
        if words:
            rec["rewrites"][EMPHASIS_KEY] = {"from": EMPHASIS_VALUE, "to": {"mode": "words", "words": words}}
            used = [e["id"] for e in ents if e["type"] != "term"]
            rec["applied"].append({"use": "subtitle_emphasis", "entities": used, "words": len(words),
                                   "line": f"entities: {_n(len(used))} applied to subtitle emphasis ({len(words)} word(s), burn-in)"})
    if reqs.get(DUB_CUES_KEY) is not None and isinstance(reqs.get(DUB_LANG_KEY), str):
        doc = pronunciation_dictionary(ents, reqs[DUB_LANG_KEY])
        if doc is not None and DUB_PRONUNCIATION_KEY in reqs:
            rec["withheld"].append({"use": "dub_pronunciation", "reason": "dub.pronunciation is given explicitly (the user's dictionary wins)"})
        elif doc is not None:
            path = pronunciation_file(workspace, doc)
            used = sorted({e["id"] for e in ents for p in e.get("pronunciation") or [] if str(reqs[DUB_LANG_KEY]).lower() == p["lang"].lower()
                           or str(reqs[DUB_LANG_KEY]).lower().startswith(p["lang"].lower() + "-")})
            rec["requirements"][DUB_PRONUNCIATION_KEY] = {"value": path, "entity": ",".join(used)[:200]}
            rec["applied"].append({"use": "dub_pronunciation", "entities": used, "file": path, "entries": len(doc["entries"]),
                                   "line": f"entities: {_n(len(used))} applied to the dub's pronunciation dictionary ({reqs[DUB_LANG_KEY]}, "
                                           f"{len(doc['entries'])} term(s); dubbing-skill --pronunciation is EXPERIMENTAL)"})
    return rec


def revised_record(old: Optional[Dict[str, Any]], reqs: Dict[str, Any], workspace: str) -> Optional[Dict[str, Any]]:
    """A revision's record: the entities the plan recorded (never the stores' newer content) applied to the revised requirements; the
    transcription uses are the plan's own (a revision re-uses the recorded analysis)."""
    if not old:
        return None
    snap = {k: old[k] for k in ("rule", "stores", "entities", "shadowed") if k in old}
    rec = plan_record(snap, reqs, workspace)
    keep = [a for a in old.get("applied") or [] if str(a.get("use", "")).startswith("transcription")]
    rec["applied"] = keep + rec["applied"]
    rec["withheld"] = [w for w in old.get("withheld") or [] if w.get("use") == "transcription"] + rec["withheld"]
    return rec


def add_transcription(rec: Dict[str, Any], vocab: Optional[Dict[str, Any]], rules: Optional[List[Dict[str, str]]], withheld: Optional[str] = None) -> None:
    if withheld:
        rec["withheld"].append({"use": "transcription", "reason": withheld})
        return
    if vocab:
        n = len(vocab.get("speakers") or []) + len(vocab.get("terms") or [])
        rec["applied"].append({"use": "transcription_vocabulary", "count": n, "line": f"entities: {_n(n)} applied to transcription vocabulary"})
    if rules:
        rec["applied"].append({"use": "transcription_corrections", "count": len(rules),
                               "line": f"entities: {len(rules)} alias correction(s) applied to the transcript (the subtitles are cut from it)"})


def requirement_values(rec: Optional[Dict[str, Any]], reqs: Dict[str, Any]) -> Dict[str, Any]:
    """The requirement values with the record applied (rewrites of the given values, then additions for keys not given): what the plan's
    checks and analysis read. The recorded explicit requirements themselves stay as the user typed them."""
    if not rec:
        return dict(reqs)
    out = dict(reqs)
    for k, rw in (rec.get("rewrites") or {}).items():
        if out.get(k) == rw["from"]:
            out[k] = rw["to"]
    for k, add in (rec.get("requirements") or {}).items():
        out.setdefault(k, add["value"])
    return out


def _n(n: int) -> str:
    return f"{n} entit{'y' if n == 1 else 'ies'}"


USE_ORDER = ("transcription_vocabulary", "transcription_corrections", "subtitle_emphasis", "lower_third", "dub_pronunciation")


def summary_lines(rec: Optional[Dict[str, Any]]) -> List[str]:
    if not rec:
        return []
    applied = sorted(rec.get("applied") or [], key=lambda a: USE_ORDER.index(a["use"]) if a.get("use") in USE_ORDER else len(USE_ORDER))
    return [a["line"] for a in applied] + [f"entities: not applied to {w['use']}: {w['reason']}" for w in rec.get("withheld") or []]
