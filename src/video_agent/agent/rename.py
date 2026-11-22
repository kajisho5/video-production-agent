"""Smart rename of ingested media (ADR-129): the agent gathers what it knows about each clip and ingest-skill renames.

    video-agent rename preview <offloaded copy> --event E --room R --camera C [--evidence DIR ...] [--template T] [--copy-to DIR]
      → `ingest inventory <copy> --json`                    (the clips, their manifest hashes, capture times)
      → per clip, from the agent's own records (nothing is measured here):
          speaker  the entity memory (ADR-128) matched through, in this order:
                     diarization-skill `name` proposals (name-proposals@1) of the clip's speaker with most speech → an entity's spelling
                     diarization-skill turns (turns@1): the clip's main speaker id equal to an entity's speaker.speaker_id
                     a transcript of the clip (transcription-skill transcript): the one person entity named most often (2+ times, a clear lead)
          scene    the clip's place among capture-time groups: a new scene starts after a gap of more than --scene-gap seconds
      → an `ingest/rename-fields@1` file (temporary) + `ingest rename plan <copy> --fields F --set event=E ... --json`
      → the rows: old → new name, every field's value and why; nothing is written
    video-agent rename apply <same options> [--plan-id ID] (--yes | an interactive yes after the preview)
      → `ingest rename apply ... --plan-id ID --json`: ingest-skill re-hashes, renames, keeps the manifest and ASC MHL in step, writes
        the undo manifest
    video-agent rename undo <copy> [--undo FILE] (--yes | an interactive yes)  → `ingest rename undo`

Evidence files are matched to a clip by the media fingerprint they carry (sha256 of the file, equal to the manifest's when the copy was
made with sha256), else by the file name they name, else by a file name that starts with the clip's name. A speaker is only ever a
spelling from the entity memory; a proposal or a mention that names nobody in it is reported, not used.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..tools.base import ToolError
from ..tools.ingest.locate import locate_ingest
from ..tools.skill_process import CliSkill, invoke, one_json_document

TOOL = "ingest/rename"
FIELDS_SCHEMA = "ingest/rename-fields@1"
PLAN_SCHEMA = "ingest/rename-plan@1"
REPORT_SCHEMA = "ingest/rename-report@1"
INVENTORY_SCHEMA = "ingest/inventory@2"
TURNS_SCHEMA = "diarization-skill/turns@1"
NAMES_SCHEMA = "diarization-skill/name-proposals@1"
TRANSCRIPT_PREFIX = "transcription-skill/transcript"
DEFAULT_SCENE_GAP_S = 120.0
MIN_MENTIONS = 2
MAIN_SPEAKER_SHARE = 0.5         # the clip's main speaker: at least half of the speech
MAX_EVIDENCE_BYTES = 32 * 1024 * 1024
MAX_EVIDENCE_FILES = 5000
TIMEOUT_S = 6 * 3600.0           # an apply re-hashes every file of the plan


class RenameError(Exception):
    def __init__(self, code: str, message: str, doc: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.doc = doc


# ---------------------------------------------------------------- the Skill

def skill_or_error(env: Optional[Dict[str, str]] = None) -> CliSkill:
    skill = locate_ingest(env=env)
    if skill is None:
        raise RenameError("ENGINE_MISSING", "ingest-skill is not installed here (VIDEO_AGENT_INGEST_DIR or the `ingest` command): smart rename runs "
                                            "through its `ingest rename`")
    return skill


def check_rename_tool(skill: CliSkill) -> Dict[str, Any]:
    """The live contract, refused unless it offers ingest/rename (ingest-skill 1.1.0+)."""
    code, out, err = invoke(skill, ["contract", "--json"], timeout=60.0)
    if code != 0:
        raise RenameError("CONTRACT_FAILED", f"ingest contract --json failed (exit {code}): {err.strip()[-300:]}")
    try:
        c = one_json_document(out, "ingest contract")
    except ToolError as e:
        raise RenameError("CONTRACT_FAILED", str(e)) from None
    if TOOL not in {t.get("tool_id") for t in c.get("tools") or [] if isinstance(t, dict)}:
        raise RenameError("SKILL_TOO_OLD", f"ingest-skill {c.get('version')} has no {TOOL} (it arrived in 1.1.0): update ingest-skill")
    return c


def run_ingest(skill: CliSkill, argv: List[str], timeout: float = TIMEOUT_S) -> Tuple[int, Dict[str, Any]]:
    code, out, err = invoke(skill, argv, timeout=timeout)
    if code == 124:
        raise RenameError("TIMEOUT", f"ingest {' '.join(argv[:2])} exceeded {timeout:g} s")
    try:
        doc = one_json_document(out, f"ingest {argv[0]}")
    except ToolError as e:
        raise RenameError("INVALID_RESULT", f"{e} (exit {code}) {err.strip()[-300:]}".strip()) from None
    if isinstance(doc.get("error"), dict):
        raise RenameError(str(doc["error"].get("code") or "REFUSED"), str(doc["error"].get("message") or ""), doc)
    return code, doc


# ---------------------------------------------------------------- evidence

def _load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        if p.stat().st_size > MAX_EVIDENCE_BYTES:
            return None
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def collect_evidence(dirs: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """The diarization turns, name proposals and transcripts under the given folders (or files): [{kind, path, doc}], and notes."""
    found: List[Dict[str, Any]] = []
    notes: List[str] = []
    paths: List[Path] = []
    for d in dirs or ():
        p = Path(d)
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            paths += sorted(x for x in p.rglob("*.json") if x.is_file())
        else:
            notes.append(f"evidence {d}: not found")
    if len(paths) > MAX_EVIDENCE_FILES:
        notes.append(f"evidence: only the first {MAX_EVIDENCE_FILES} of {len(paths)} JSON files were read")
        paths = paths[:MAX_EVIDENCE_FILES]
    for p in paths:
        doc = _load_json(p)
        schema = str((doc or {}).get("schema") or "")
        kind = "turns" if schema == TURNS_SCHEMA else "names" if schema == NAMES_SCHEMA else "transcript" if schema.startswith(TRANSCRIPT_PREFIX) else None
        if kind:
            found.append({"kind": kind, "path": str(p), "doc": doc})
    return found, notes


def _fingerprint(doc: Dict[str, Any]) -> Optional[str]:
    src = doc.get("source") if isinstance(doc.get("source"), dict) else {}
    fp = str(src.get("fingerprint") or "")
    return fp.split(":", 1)[1].lower() if fp.startswith("sha256:") else None


def _source_name(doc: Dict[str, Any]) -> Optional[str]:
    src = doc.get("source") if isinstance(doc.get("source"), dict) else {}
    for k in ("filename", "file", "name"):
        if isinstance(src.get(k), str) and src[k]:
            return PurePosixPath(src[k].replace("\\", "/")).name
    return None


def match_evidence(clip: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{kind: {path, doc, matched_by}} for one inventory clip: by the media fingerprint, else the named source file, else the file name."""
    h = clip.get("hash") or {}
    sha = str(h.get("value") or "").lower() if h.get("algorithm") == "sha256" else None
    name = PurePosixPath(str(clip.get("path") or "")).name
    stem = PurePosixPath(name).stem
    out: Dict[str, Dict[str, Any]] = {}
    for rank, test in (("fingerprint", lambda e: sha is not None and _fingerprint(e["doc"]) == sha),
                       ("source file name", lambda e: _source_name(e["doc"]) is not None and _source_name(e["doc"]).casefold() == name.casefold()),
                       ("evidence file name", lambda e: Path(e["path"]).name.casefold().startswith((stem + ".").casefold()))):
        for e in evidence:
            if e["kind"] not in out and test(e):
                out[e["kind"]] = {"path": e["path"], "doc": e["doc"], "matched_by": rank}
    return out


def _speech_by_speaker(turns_doc: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for t in turns_doc.get("turns") or []:
        try:
            out[str(t["speaker"])] = out.get(str(t["speaker"]), 0.0) + max(0.0, float(t["end"]) - float(t["start"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def main_speaker(turns_doc: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    """(speaker id, share of the speech) when one speaker has at least MAIN_SPEAKER_SHARE of it."""
    speech = _speech_by_speaker(turns_doc)
    total = sum(speech.values())
    if total <= 0:
        return None
    sid, sec = max(speech.items(), key=lambda kv: (kv[1], kv[0]))
    share = sec / total
    return (sid, share) if share >= MAIN_SPEAKER_SHARE else None


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", " ".join(str(s).split())).casefold()


def _cjk(s: str) -> bool:
    return bool(re.search(r"[぀-ヿ㐀-鿿가-힯]", s))


def _transcript_text(doc: Dict[str, Any]) -> str:
    parts = [str(s.get("text") or "") for s in doc.get("segments") or [] if isinstance(s, dict)]
    return " ".join(parts) if any(parts) else str(doc.get("text") or "")


def mention_counts(text: str, people: List[Dict[str, Any]]) -> Dict[str, int]:
    """How often each person is named: every spelling and alias, the longest first, each stretch of text counted once
    ("Kenji Tanaka" is one mention, not also one of the alias "Tanaka")."""
    t = _norm(text)
    spellings = sorted({(_norm(x), e["id"]) for e in people for x in [e["canonical"]] + list(e.get("aliases") or []) if str(x).strip()},
                       key=lambda se: (-len(se[0]), se[0]))
    taken = [False] * len(t)
    out: Dict[str, int] = {}
    for s, eid in spellings:
        pattern = re.escape(s) if _cjk(s) else r"(?<!\w)" + re.escape(s) + r"(?!\w)"
        for m in re.finditer(pattern, t):
            if any(taken[m.start():m.end()]):
                continue
            taken[m.start():m.end()] = [True] * (m.end() - m.start())
            out[eid] = out.get(eid, 0) + 1
    return out


def _by_name(entities: List[Dict[str, Any]], name: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(name, str):
        return None
    k = _norm(name)
    for e in entities:
        if k in {_norm(s) for s in [e["canonical"]] + list(e.get("aliases") or [])} or k == _norm(e["id"]):
            return e
    return None


def speaker_for(matched: Dict[str, Dict[str, Any]], entities: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """({value, source} or None, notes) for one clip, from its evidence and the entity memory."""
    people = [e for e in entities if e.get("type") == "person"]
    notes: List[str] = []
    turns = (matched.get("turns") or {}).get("doc")
    main = main_speaker(turns) if turns else None
    names = (matched.get("names") or {}).get("doc")
    if names:
        props = [s for s in names.get("speakers") or [] if isinstance(s, dict) and s.get("status") == "proposed" and isinstance(s.get("proposal"), dict)]
        pick = None
        if main:
            pick = next((s for s in props if s.get("speaker") == main[0]), None)
        elif len(props) == 1:
            pick = props[0]
        if pick:
            prop = pick["proposal"]
            e = _by_name(people, prop.get("name")) or _by_name(people, prop.get("person_id"))
            basis = "+".join(prop.get("basis") or [])
            if e:
                share = f", {main[1]:.0%} of the speech" if main else ""
                return {"value": e["canonical"], "source": f"entity memory ({e['id']}) via diarization name proposal for {pick['speaker']} ({basis}{share}; unconfirmed: confirm in this preview)"}, notes
            notes.append(f"diarization proposed {prop.get('name')!r} for {pick['speaker']}, who is not in the entity memory: not used")
    if main:
        hits = [e for e in people if (e.get("speaker") or {}).get("speaker_id") == main[0]]
        if len(hits) == 1:
            return {"value": hits[0]["canonical"], "source": f"entity memory ({hits[0]['id']}) speaker_id {main[0]}, the clip's main speaker ({main[1]:.0%} of the speech; "
                                                              "diarization ids are per recording)"}, notes
        if len(hits) > 1:
            notes.append(f"{len(hits)} entities have speaker_id {main[0]}: none chosen")
    tr = (matched.get("transcript") or {}).get("doc")
    if tr and people:
        counts = mention_counts(_transcript_text(tr), people)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if ranked and ranked[0][1] >= MIN_MENTIONS and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
            e = next(x for x in people if x["id"] == ranked[0][0])
            return {"value": e["canonical"], "source": f"entity memory ({e['id']}) named {ranked[0][1]} times in the clip's transcript (a mention, not a voice match)"}, notes
        if ranked:
            notes.append("transcript mentions give no clear speaker (" + ", ".join(f"{k} x{v}" for k, v in ranked[:3]) + ")")
    return None, notes


def _start_end(clip: Dict[str, Any]) -> Optional[Tuple[datetime, float]]:
    wc = (clip.get("sync") or {}).get("wall_clock_start") or {}
    text = wc.get("local") or wc.get("utc")
    if not text:
        return None
    t = str(text)
    try:
        dt = datetime.fromisoformat(t[:-1] if t.endswith("Z") else t)
    except ValueError:
        try:
            dt = datetime.fromisoformat(t[:19])
        except ValueError:
            return None
    dur = clip.get("duration_s") or (clip.get("sync") or {}).get("duration_s") or 0.0
    return dt.replace(tzinfo=None), float(dur or 0.0)


def scenes(clips: List[Dict[str, Any]], gap_s: float = DEFAULT_SCENE_GAP_S) -> Dict[str, Dict[str, Any]]:
    """{clip path: {value, source}}: clips in capture order, a new scene after a pause of more than gap_s between one clip's end
    and the next one's start. Clips without a capture time get none."""
    timed = sorted(((c, se) for c in clips for se in [_start_end(c)] if se), key=lambda x: (x[1][0], x[0]["path"]))
    out: Dict[str, Dict[str, Any]] = {}
    scene = 0
    prev_end: Optional[datetime] = None
    for c, (start, dur) in timed:
        if prev_end is None:
            scene, why = 1, "the first clip in capture order"
        else:
            pause = (start - prev_end).total_seconds()
            if pause > gap_s:
                scene += 1
                why = f"a pause of {pause:.0f} s before it (more than {gap_s:g} s) starts a new scene"
            else:
                why = f"continues the scene (a pause of {max(pause, 0.0):.0f} s, at most {gap_s:g} s)"
        end = start + timedelta(seconds=dur)
        prev_end = end if prev_end is None or end > prev_end else prev_end
        out[c["path"]] = {"value": scene, "source": f"capture-time grouping: {why}"}
    return out


def fields_document(inventory: Dict[str, Any], entities: List[Dict[str, Any]], evidence: List[Dict[str, Any]], scene_gap_s: Optional[float],
                    use_scenes: bool = True) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """(ingest/rename-fields@1 document, per-clip notes, notes) for an inventory."""
    clips = [c for c in inventory.get("clips") or [] if isinstance(c, dict) and c.get("path")]
    sc = scenes(clips, DEFAULT_SCENE_GAP_S if scene_gap_s is None else scene_gap_s) if use_scenes else {}
    out: Dict[str, Dict[str, Any]] = {}
    per_clip: List[Dict[str, Any]] = []
    for c in clips:
        matched = match_evidence(c, evidence)
        f: Dict[str, Any] = {}
        sp, notes = speaker_for(matched, entities) if entities else (None, ["the entity memory is empty: no speaker names"] if matched else [])
        if sp:
            f["speaker"] = sp
        if c["path"] in sc:
            f["scene"] = sc[c["path"]]
        if f:
            out[c["path"]] = f
        per_clip.append({"clip": c["path"], "evidence": {k: {"path": v["path"], "matched_by": v["matched_by"]} for k, v in matched.items()},
                         "speaker": sp, "scene": sc.get(c["path"]), "notes": notes})
    return {"schema": FIELDS_SCHEMA, "clips": out}, per_clip, []


# ---------------------------------------------------------------- the calls

def _common_argv(o: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    if o.get("template"):
        argv += ["--template", str(o["template"])]
    for k in ("event", "room", "camera"):
        if o.get(k) not in (None, ""):
            argv += ["--set", f"{k}={o[k]}"]
    for kv in o.get("set") or []:
        argv += ["--set", str(kv)]
    if o.get("utc_offset"):
        argv += ["--utc-offset", str(o["utc_offset"])]
    if o.get("date_from_mtime"):
        argv.append("--date-from-mtime")
    if o.get("copy_to"):
        argv += ["--copy-to", str(Path(o["copy_to"]).resolve())]
    return argv


def entity_list(workspace: str, project: Optional[str], use: bool) -> List[Dict[str, Any]]:
    if not use:
        return []
    from .. import entities as ent
    snap = ent.snapshot(workspace, project)
    return list((snap or {}).get("entities") or [])


def _check_folder(folder: str) -> str:
    p = Path(folder)
    if not p.is_dir():
        raise RenameError("INPUT_MISSING", f"{folder} is not a folder")
    return str(p.resolve())


def preview(folder: str, workspace: str, options: Optional[Dict[str, Any]] = None, skill: Optional[CliSkill] = None,
            keep_fields: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """The rename plan with the agent's evidence: {plan (ingest/rename-plan@1), agent: {fields, clips, notes}}. Writes nothing
    (the fields file lives in a temporary folder for the one call)."""
    o = dict(options or {})
    root = _check_folder(folder)
    skill = skill or skill_or_error()
    check_rename_tool(skill)
    _, inv = run_ingest(skill, ["inventory", root, "--json"], timeout=TIMEOUT_S)
    if inv.get("schema") != INVENTORY_SCHEMA:
        raise RenameError("INVALID_RESULT", f"ingest inventory gave {inv.get('schema')!r}, not {INVENTORY_SCHEMA}")
    ents = entity_list(workspace, o.get("entity_project"), not o.get("no_entities"))
    evidence, notes = collect_evidence(o.get("evidence") or [])
    fields, per_clip, more = fields_document(inv, ents, evidence, o.get("scene_gap"), use_scenes=not o.get("no_scenes"))
    with tempfile.TemporaryDirectory(prefix="video-agent-rename-") as tmp:
        fpath = os.path.join(tmp, "fields.json")
        Path(fpath).write_text(json.dumps(fields, ensure_ascii=False, indent=1), encoding="utf-8")
        if keep_fields:
            keep_fields(fpath)
        _, plan = run_ingest(skill, ["rename", "plan", root, *_common_argv(o), "--fields", fpath, "--json"], timeout=TIMEOUT_S)
    if plan.get("schema") != PLAN_SCHEMA:
        raise RenameError("INVALID_RESULT", f"ingest rename plan gave {plan.get('schema')!r}, not {PLAN_SCHEMA}")
    return {"status": "PREVIEW", "plan": plan, "plan_id": plan.get("plan_id"),
            "agent": {"entities": len(ents), "evidence_files": len(evidence), "clips": per_clip, "notes": notes + more,
                      "scene_gap_s": None if o.get("no_scenes") else (o.get("scene_gap") if o.get("scene_gap") is not None else DEFAULT_SCENE_GAP_S)},
            "next": "show the user the rows; only after they agree, apply with this plan_id"}


def apply(folder: str, workspace: str, plan_id: str, options: Optional[Dict[str, Any]] = None, skill: Optional[CliSkill] = None) -> Dict[str, Any]:
    """Run exactly the previewed plan (ingest-skill refuses PLAN_CHANGED otherwise). The fields file is built again from the same
    evidence, so the same facts give the same plan."""
    o = dict(options or {})
    root = _check_folder(folder)
    skill = skill or skill_or_error()
    check_rename_tool(skill)
    _, inv = run_ingest(skill, ["inventory", root, "--json"], timeout=TIMEOUT_S)
    ents = entity_list(workspace, o.get("entity_project"), not o.get("no_entities"))
    evidence, _ = collect_evidence(o.get("evidence") or [])
    fields, _, _ = fields_document(inv, ents, evidence, o.get("scene_gap"), use_scenes=not o.get("no_scenes"))
    with tempfile.TemporaryDirectory(prefix="video-agent-rename-") as tmp:
        fpath = os.path.join(tmp, "fields.json")
        Path(fpath).write_text(json.dumps(fields, ensure_ascii=False, indent=1), encoding="utf-8")
        code, rep = run_ingest(skill, ["rename", "apply", root, *_common_argv(o), "--fields", fpath, "--plan-id", str(plan_id), "--json"])
    if rep.get("schema") != REPORT_SCHEMA:
        raise RenameError("INVALID_RESULT", f"ingest rename apply gave {rep.get('schema')!r}, not {REPORT_SCHEMA}")
    return {"status": "APPLIED" if rep.get("ok") else "FAILED", "report": rep, "exit_code": code}


def undo(folder: str, undo_file: Optional[str] = None, skill: Optional[CliSkill] = None) -> Dict[str, Any]:
    root = _check_folder(folder)
    skill = skill or skill_or_error()
    check_rename_tool(skill)
    code, rep = run_ingest(skill, ["rename", "undo", root, *(["--undo", str(Path(undo_file).resolve())] if undo_file else []), "--json"])
    if rep.get("schema") != REPORT_SCHEMA:
        raise RenameError("INVALID_RESULT", f"ingest rename undo gave {rep.get('schema')!r}, not {REPORT_SCHEMA}")
    return {"status": "UNDONE" if rep.get("ok") else "FAILED", "report": rep, "exit_code": code}


def latest_undo_file(folder: str) -> Optional[str]:
    """The undo manifest `ingest rename undo` would use (the newest applied in-place one), for a confirmation preview."""
    d = Path(folder) / "ingest-rename"
    for p in sorted(d.glob("*.json"), reverse=True) if d.is_dir() else []:
        doc = _load_json(p) or {}
        if doc.get("status") == "applied" and doc.get("mode") == "in_place":
            return str(p)
    return None


# ---------------------------------------------------------------- what the user reads

def preview_lines(result: Dict[str, Any], limit: Optional[int] = None) -> List[str]:
    plan = result["plan"]
    c = plan.get("counts") or {}
    verb = "copy" if plan.get("mode") == "copy" else "rename"
    lines = [f"{plan.get('root')}: {c.get('rename', 0) + c.get('copy', 0)} clip(s) to {verb}, {c.get('unchanged', 0)} unchanged, "
             f"{c.get('skipped', 0)} skipped ({plan.get('files_to_move', 0)} file(s) with sidecars); plan_id {plan.get('plan_id')}"]
    rows = plan.get("rows") or []
    for r in rows[:limit] if limit else rows:
        lines.append(f"  {r['from']} -> {r['to'] if r.get('files') else '(kept)'}" + (f"   [{r['reason']}]" if r.get("reason") else ""))
        for f in (r.get("files") or [])[1:]:
            lines.append(f"      {f['from']} -> {f['to']} ({f['role']})")
        for name, v in sorted((r.get("fields") or {}).items()):
            if v.get("value") not in (None, "") and name not in ("ext", "orig"):
                lines.append(f"      {name} = {v['value']}  ({v['source']})")
    if limit and len(rows) > limit:
        lines.append(f"  ... {len(rows) - limit} more row(s) in the JSON")
    for n in result.get("agent", {}).get("notes") or []:
        lines.append(f"  note: {n}")
    for w in plan.get("warnings") or []:
        lines.append(f"  warning: {w}")
    return lines


def options_from_args(a: Any) -> Dict[str, Any]:
    return {k: getattr(a, k, None) for k in ("template", "event", "room", "camera", "set", "utc_offset", "date_from_mtime", "copy_to", "evidence",
                                               "entity_project", "no_entities", "scene_gap", "no_scenes")}


def add_parser(sub: Any, fn: Callable) -> None:
    """`video-agent rename preview | apply | undo` (registered by cli.py)."""
    p = sub.add_parser("rename", help="smart rename of an offloaded copy's clips (date, event, room, camera, scene, speaker, seq) through ingest-skill; "
                                      "preview first, apply after a yes, undo (ADR-129)")
    rsub = p.add_subparsers(dest="rename_cmd", required=True)
    for name, hlp in (("preview", "old -> new name of every clip with each field's source; nothing is renamed"),
                      ("apply", "show the preview, then rename after --yes or an interactive yes"),
                      ("undo", "restore the names of the last rename (after --yes or an interactive yes)")):
        q = rsub.add_parser(name, help=hlp)
        q.add_argument("folder", help="the offloaded copy (a folder ingest offload wrote, with its ingest-manifest.json)")
        if name in ("apply", "undo"):
            q.add_argument("--yes", action="store_true", help="do it without asking (after a preview you have seen)")
        if name == "undo":
            q.add_argument("--undo", dest="undo_file", help="an ingest-rename/<time>.json undo manifest (default: the last applied one)")
            q.set_defaults(fn=fn)
            continue
        q.add_argument("--template", help="ingest-skill's template (default {date}_{event}_{room}_{camera}_{scene:S02}_{speaker:Speaker-}_{seq:04}{ext})")
        q.add_argument("--event", help="the event's name, e.g. ConfName"); q.add_argument("--room", help="the room, e.g. 'Room A'")
        q.add_argument("--camera", help="the camera's name, e.g. CamB (default: the card's reel when it is a camera reel)")
        q.add_argument("--set", action="append", metavar="FIELD=VALUE", help="another template field for every clip")
        q.add_argument("--evidence", action="append", metavar="DIR_OR_FILE",
                       help="diarization turns / name proposals and transcripts of these clips (JSON), matched by fingerprint or file name")
        q.add_argument("--entity-project", help="also the entity memory of this project (ADR-128)")
        q.add_argument("--no-entities", action="store_true", help="no speaker names from the entity memory")
        q.add_argument("--scene-gap", type=float, help=f"seconds of pause between clips that start a new scene (default {DEFAULT_SCENE_GAP_S:g})")
        q.add_argument("--no-scenes", action="store_true", help="no scene numbers")
        q.add_argument("--utc-offset", help="local time for clips that record UTC only, e.g. +09:00")
        q.add_argument("--date-from-mtime", action="store_true", help="the file time when a clip records no capture time")
        q.add_argument("--copy-to", help="renamed, verified copies in this folder instead of renaming in place")
        if name == "apply":
            q.add_argument("--plan-id", help="the plan_id a preview printed (default: the preview shown now)")
        q.set_defaults(fn=fn)


def confirm(prompt: str, stdin: Any = None, stdout: Any = None) -> bool:
    import sys
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    if not getattr(stdin, "isatty", lambda: False)():
        return False
    stdout.write(prompt)
    stdout.flush()
    return stdin.readline().strip().lower() in ("y", "yes")


def cmd_rename(args: Any, svc: Any) -> int:
    """The CLI command. Exit 0 done / previewed, 2 refused or not confirmed, 3 the Skill reported an incomplete run."""
    as_json = bool(getattr(args, "json", False))

    def emit(doc: Dict[str, Any], lines: List[str]) -> None:
        print(json.dumps(doc, indent=2, ensure_ascii=False, default=str) if as_json else "\n".join(lines))
    try:
        if args.rename_cmd == "undo":
            target = latest_undo_file(args.folder) if not args.undo_file else args.undo_file
            if not args.yes:
                msg = f"undo the rename recorded in {target or '(no applied rename found)'}"
                if not confirm(f"{msg}: rename the files back? [y/N] "):
                    emit({"status": "NOT_CONFIRMED", "undo": target, "detail": "nothing was renamed: pass --yes, or answer yes at a terminal"}, [f"{msg}: not confirmed, nothing done (pass --yes)"])
                    return 2
            r = undo(args.folder, args.undo_file)
            rep = r["report"]
            emit(r, [f"undo: {rep.get('moved', 0)} file(s) renamed back" if rep.get("ok") else "undo NOT done: " + "; ".join(f"{x.get('code')}: {x.get('path') or x.get('detail')}" for x in rep.get("failures") or [])])
            return 0 if rep.get("ok") else 3
        opts = options_from_args(args)
        res = preview(args.folder, svc.workspace, opts)
        if args.rename_cmd == "preview":
            emit(res, preview_lines(res))
            return 0
        plan_id = args.plan_id or res["plan_id"]
        if args.plan_id and args.plan_id != res["plan_id"]:
            emit({"status": "PLAN_CHANGED", "preview": res, "detail": "the folder or the options give another plan now: review the new preview"},
                 preview_lines(res) + [f"PLAN_CHANGED: the plan is now {res['plan_id']}, not {args.plan_id}; nothing was renamed"])
            return 2
        if not res["plan"].get("files_to_move"):
            emit({"status": "NOTHING_TO_DO", "preview": res}, preview_lines(res) + ["nothing to rename"])
            return 0
        if not args.yes:
            if not as_json:
                print("\n".join(preview_lines(res)))
            if not confirm(f"rename {res['plan']['files_to_move']} file(s) as shown? [y/N] "):
                emit({"status": "NOT_CONFIRMED", "preview": res, "detail": "nothing was renamed: pass --yes after reviewing the preview, or answer yes at a terminal"},
                     ["not confirmed: nothing was renamed (pass --yes)"])
                return 2
        r = apply(args.folder, svc.workspace, plan_id, opts)
        rep = r["report"]
        lines = ([f"renamed {rep.get('moved', 0)} file(s); undo manifest {rep.get('undo')}" if rep.get("mode") != "copy" else
                  f"copied {rep.get('moved', 0)} file(s) to {rep.get('copy_to')}"] if rep.get("ok") else
                 ["rename NOT done: " + "; ".join(f"{x.get('code')}: {x.get('path') or x.get('detail')}" for x in rep.get("failures") or [])])
        emit(r, lines)
        return 0 if rep.get("ok") else 3
    except RenameError as e:
        emit({"status": "REFUSED", "error": {"code": e.code, "message": str(e)}}, [f"refused ({e.code}): {e}"])
        return 2


# ---------------------------------------------------------------- MCP (mcp_permissions.classify and mcp_server.tool_argv)

MCP_OPTION_KEYS = ("template", "event", "room", "camera", "set", "evidence", "entity_project", "no_entities", "scene_gap", "no_scenes",
                   "utc_offset", "date_from_mtime", "copy_to")


def mcp_options(a: Dict[str, Any]) -> Dict[str, Any]:
    o = {k: a.get(k) for k in MCP_OPTION_KEYS}
    for k in ("set", "evidence"):
        v = o.get(k)
        o[k] = [v] if isinstance(v, str) else [str(x) for x in v or []]
    return o


def mcp_argv(name: str, a: Dict[str, Any]) -> List[str]:
    """CLI arguments for rename_preview / rename_apply / rename_undo; every value is spelled --flag=value, the folder after `--`."""
    folder = a.get("folder")
    if not folder:
        raise ValueError(f"{name}: 'folder' is required")
    if name == "rename_undo":
        return ["rename", "undo", "--yes", *([f"--undo={a['undo']}"] if a.get("undo") else []), "--", str(folder)]
    o = mcp_options(a)
    argv: List[str] = []
    for k in ("template", "event", "room", "camera", "entity_project", "scene_gap", "utc_offset", "copy_to"):
        if o.get(k) not in (None, ""):
            argv.append(f"--{k.replace('_', '-')}={o[k]}")
    argv += [f"--set={x}" for x in o["set"]] + [f"--evidence={x}" for x in o["evidence"]]
    argv += [f"--{k.replace('_', '-')}" for k in ("no_entities", "no_scenes", "date_from_mtime") if o.get(k)]
    if name == "rename_preview":
        return ["rename", "preview", *argv, "--", str(folder)]
    if name == "rename_apply":
        if not a.get("plan_id"):
            raise ValueError("rename_apply: 'plan_id' is required (the plan_id rename_preview returned)")
        return ["rename", "apply", "--yes", f"--plan-id={a['plan_id']}", *argv, "--", str(folder)]
    raise KeyError(name)


def confirmation_items(name: str, a: Dict[str, Any], workspace: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """(paths the call writes, preview items) for the confirm_token step (ADR-125): the rows that would be renamed and the plan_id,
    or the undo manifest's entries and digest. The token is bound to them, so a folder that changes after the preview needs a new one."""
    import hashlib
    folder = str(a.get("folder") or "")
    if name == "rename_undo":
        target = str(a.get("undo") or "") or latest_undo_file(folder)
        item: Dict[str, Any] = {"action": "undo_rename", "folder": os.path.abspath(folder) if folder else None, "undo": target}
        doc = _load_json(Path(target)) if target else None
        if doc and isinstance(doc.get("entries"), list):
            item.update(sha256=hashlib.sha256(Path(target).read_bytes()).hexdigest(), files=len(doc["entries"]),
                        renames=[{"from": e.get("to"), "to": e.get("from")} for e in doc["entries"][:200]],
                        detail="each file renamed by that rename goes back to its original name; the manifest and the ASC MHL history follow")
        else:
            item["detail"] = "nothing to undo found: the call will be refused"
        return ([folder] if folder else []), [item]
    o = mcp_options(a)
    writes = [str(o["copy_to"])] if o.get("copy_to") else ([folder] if folder else [])
    try:
        res = preview(folder, workspace, o)
    except (RenameError, ToolError, OSError) as exc:
        return writes, [{"action": "rename_files", "folder": os.path.abspath(folder) if folder else None, "detail": f"the preview was refused: {exc}"}]
    plan = res["plan"]
    item = {"action": "copy_renamed" if plan.get("mode") == "copy" else "rename_files", "folder": plan.get("root"), "copy_to": plan.get("copy_to"),
            "plan_id": plan.get("plan_id"), "requested_plan_id": a.get("plan_id"), "counts": plan.get("counts"),
            "files": plan.get("files_to_move"), "renames": [{"from": f["from"], "to": f["to"]} for r in plan.get("rows") or [] for f in r.get("files") or []][:500],
            "detail": ("the files are renamed in place (the manifest and ASC MHL follow; an undo manifest is written)" if plan.get("mode") != "copy"
                       else "renamed, verified copies are written to copy_to; the folder is only read")}
    if a.get("plan_id") and a.get("plan_id") != plan.get("plan_id"):
        item["detail"] += f"; the plan is now {plan.get('plan_id')}, not the requested {a.get('plan_id')}: the call will be refused (PLAN_CHANGED)"
    return writes, [item]
