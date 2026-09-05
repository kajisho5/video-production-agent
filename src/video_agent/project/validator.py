"""IR validation: JSON Schema + semantic rules + capability availability."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from ..agent.decision_engine import check_decisions
from ..agent.audio import AUDIO_ORDER, OPERATIONS as AUDIO_OPERATIONS, SKILL_OF as AUDIO_SKILL_OF
from ..agent.editing import EDIT_ORDER, OPERATIONS, SKILL_OF
from ..agent.finishing import COLOR_OPERATIONS, COLOR_ORDER, COLOR_SKILL_OF, ELEMENT_TYPES, GRAPHICS_SKILL, THUMBNAIL_FRAME_SKILL, THUMBNAIL_RENDER_SKILL
from ..agent.production_plan import validate_plan
from ..agent.qc import QC_SKILL
from ..agent.subtitles import BURN_SKILL, GENERATE_SKILL
from ..models import Event
from ..temporal import Session, classify, validate_event, validate_session
from typing import Any, Dict, List, Optional

from .ir import ProjectIR

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "project.schema.json"


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def produced_subjects(d: Dict[str, Any]) -> set:
    """Logical subjects an operation produced (the video / audio concat programmes) that later operations may reference."""
    return {op.get("output") for op in d["video"]["operations"] + d["audio"]["operations"] if op.get("type") in ("video.concat", "audio.concat") and op.get("output")}


def check_audio_operations(d: Dict[str, Any]) -> List[str]:
    """Per-type rules of the audio operations on the audio production path (ADR-030): explicit type, allowlisted parameters, an
    audio stream on the subject, one cut per subject before anything else, one concat at most, the fixed order, references that
    exist, ranges inside the source, the loudness operation carrying its references only on the audio path. Never a guess."""
    errs: List[str] = []
    assets = d["assets"]
    aops = d["audio"]["operations"]
    known: set = set(assets)
    seen_rank: Dict[str, int] = {}
    concat_seen = False
    on_audio_path: set = set()
    for op in aops:
        t = op.get("type")
        subj = op.get("asset")
        if t == "audio.loudness":
            if ("input" in op) != ("output" in op):
                errs.append("audio.loudness: input and output references go together (audio path) or are both absent (reference engine path)")
            if "input" not in op and any(k in op for k in ("tolerance_lu", "sample_rate")):
                errs.append("audio.loudness: tolerance_lu / sample_rate exist only on the audio production path")
            if "input" in op:
                on_audio_path.add(subj)
                if subj not in known:
                    errs.append(f"audio.loudness references {subj!r} before it exists")
                rank = AUDIO_ORDER.index("audio.loudness")
                if seen_rank.get(subj, -1) >= rank:
                    errs.append(f"audio.loudness on {subj} is out of the fixed audio order {AUDIO_ORDER}")
                seen_rank[subj] = rank
            continue
        if t not in AUDIO_OPERATIONS:
            errs.append(f"unknown audio operation type {t!r}")
            continue
        spec = AUDIO_OPERATIONS[t]
        extra = sorted(k for k in op if k not in ("type", "asset", "input", "inputs", "output", "segments", "timeline_duration", "temporal_scope", "decision_ids") and k not in spec["params"])
        if extra:
            errs.append(f"{t}: parameters {extra} are not in the operation's vocabulary")
        on_audio_path.add(subj)
        if t == "audio.cut":
            if subj not in assets:
                errs.append(f"audio.cut on {subj!r}: only a source asset is cut")
            elif subj in seen_rank:
                errs.append(f"audio.cut on {subj} must come before every other audio operation")
            seen_rank[subj] = -1
            dur = (assets.get(subj) or {}).get("technical", {}).get("duration") or 0.0
            rs = op.get("remove") or []
            last = -1.0
            for r in rs:
                if r[1] <= r[0] or r[0] < last:
                    errs.append(f"audio.cut remove range {r} is empty, unsorted or overlapping")
                if dur and r[1] > float(dur) + 0.01:
                    errs.append(f"audio.cut remove range {r} exceeds asset duration {float(dur):.3f}")
                last = r[1]
            if dur and sum(e - s for s, e in rs) >= float(dur) - 0.01:
                errs.append(f"audio.cut on {subj} would remove everything")
        elif t == "audio.concat":
            if concat_seen:
                errs.append("more than one audio.concat (one audio programme per plan)")
            concat_seen = True
            ins = op.get("inputs") or []
            if len(ins) < 2 or len(set(ins)) != len(ins):
                errs.append("audio.concat needs two or more distinct inputs")
            for i in ins:
                if i not in assets:
                    errs.append(f"audio.concat input {i!r} is not an asset")
            if op.get("output") != subj or subj in assets:
                errs.append("audio.concat: output must equal the operation's subject and not collide with an asset id")
            segs = op.get("segments") or []
            if segs and {s_.get("input") for s_ in segs} != set(ins):
                errs.append("audio.concat segments do not cover exactly the inputs")
            known.add(subj)
            seen_rank[subj] = -1
            continue
        else:
            if subj not in known:
                errs.append(f"{t} references {subj!r} before it exists (the audio programme exists only after audio.concat)")
            rank = AUDIO_ORDER.index(t)
            if seen_rank.get(subj, -1) >= rank:
                errs.append(f"{t} on {subj} is out of the fixed audio order {AUDIO_ORDER}")
            seen_rank[subj] = rank
            if op.get("input") is None:
                errs.append(f"{t} on {subj}: input reference missing")
            if t in ("audio.fade_in", "audio.fade_out") and not (isinstance(op.get("duration"), (int, float)) and 0 < float(op["duration"]) <= 3600):
                errs.append(f"{t}: duration {op.get('duration')!r} must be within 0..3600 s")
            if t == "audio.gain" and not (isinstance(op.get("gain_db"), (int, float)) and -60 <= float(op["gain_db"]) <= 60 and float(op["gain_db"]) != 0):
                errs.append(f"audio.gain: gain_db {op.get('gain_db')!r} must be within -60..60 and not 0")
    for subj in on_audio_path:
        if subj in assets and not ((assets[subj].get("technical") or {}).get("audio")):
            errs.append(f"audio operations on {subj}: no audio stream")
    # an asset on the audio path is never also edited as video (its deliverable is audio)
    for op in d["video"]["operations"]:
        if op.get("asset") in on_audio_path:
            errs.append(f"{op.get('type')} on {op.get('asset')}: the asset is on the audio production path (audio deliverable), video operations conflict")
    return errs


def check_video_operations(d: Dict[str, Any]) -> List[str]:
    """Per-type rules of the video operations (ADR-029): explicit type, allowlisted parameters, references that exist at that
    point of the chain, ranges inside the source, one concat at most, the fixed order, no fit + fill, image inside the allowed
    input roots. Anything outside the vocabulary is an error, never a guess."""
    errs: List[str] = []
    assets = d["assets"]
    ops = d["video"]["operations"]
    known: set = set(assets)
    order_seen: Dict[str, int] = {}
    concat_seen = False
    allowed_roots = [str(Path(r)) for r in (d.get("execution") or {}).get("allowed_inputs") or []]
    for op in ops:
        t = op.get("type")
        subj = op.get("asset")
        if t == "video.trim":
            if subj not in assets:
                continue   # reported by the caller
            dur = (assets[subj].get("technical") or {}).get("duration") or 0.0
            for k in op.get("keep") or []:
                if k[1] <= k[0]:
                    errs.append(f"video.trim keep range {k} is empty")
                if dur and k[1] > dur + 0.01:
                    errs.append(f"video.trim keep range {k} exceeds asset duration {dur:.3f}")
            continue
        if t not in OPERATIONS:
            errs.append(f"unknown video operation type {t!r}")
            continue
        spec = OPERATIONS[t]
        extra = sorted(k for k in op if k not in ("type", "asset", "input", "inputs", "output", "image", "segments", "timeline_duration", "temporal_scope", "decision_ids") and k not in spec["params"])
        if extra:
            errs.append(f"{t}: parameters {extra} are not in the operation's vocabulary")
        rank = EDIT_ORDER.index(t)
        if subj in order_seen and order_seen[subj] >= rank:
            errs.append(f"{t} on {subj} is out of the fixed operation order {EDIT_ORDER}")
        order_seen[subj] = rank
        if t == "video.concat":
            if concat_seen:
                errs.append("more than one video.concat (one programme per plan)")
            concat_seen = True
            ins = op.get("inputs") or []
            if len(ins) < 2 or len(set(ins)) != len(ins):
                errs.append("video.concat needs two or more distinct inputs")
            for i in ins:
                if i not in assets:
                    errs.append(f"video.concat input {i!r} is not an asset")
                elif not ((assets[i].get("technical") or {}).get("video")):
                    errs.append(f"video.concat input {i!r} has no video stream")
            if op.get("output") != subj:
                errs.append("video.concat: output must equal the operation's subject")
            if subj in assets:
                errs.append(f"video.concat output {subj!r} collides with an asset id")
            segs = op.get("segments") or []
            if [s_.get("input") for s_ in segs] and set(s_.get("input") for s_ in segs) != set(ins):
                errs.append("video.concat segments do not cover exactly the inputs")
            for s_ in segs:
                sr, tr = s_.get("source_range") or [0, 0], s_.get("timeline_range") or [0, 0]
                if not (sr[0] < sr[1] and tr[0] < tr[1]):
                    errs.append(f"video.concat segment {s_.get('input')}: empty range")
                sd = (assets.get(s_.get("input")) or {}).get("technical", {}).get("duration")
                if sd and sr[1] > float(sd) + 0.01:
                    errs.append(f"video.concat segment {s_.get('input')}: source range {sr} exceeds duration {sd:.3f}")
            known.add(subj)
            continue
        if subj not in known:
            errs.append(f"{t} references {subj!r} before it exists (the programme exists only after video.concat)")
        if subj in assets and not ((assets[subj].get("technical") or {}).get("video")):
            errs.append(f"{t} on {subj}: no video stream")
        if op.get("input") is None:
            errs.append(f"{t} on {subj}: input reference missing")
        if t == "video.speed":
            f = op.get("factor")
            if not isinstance(f, (int, float)) or isinstance(f, bool) or not (0.25 <= float(f) <= 4.0) or float(f) == 1.0:
                errs.append(f"video.speed factor {f!r} must be within 0.25..4 and not 1")
        if t in ("video.resize", "video.fit", "video.fill", "video.concat") and op.get("width") is not None and (int(op["width"]) % 2 or int(op["width"]) < 16):
            errs.append(f"{t}: width {op['width']} must be an even integer ≥ 16")
        if t == "video.resize" and op.get("width") is None:
            errs.append("video.resize needs width")
        if t in ("video.fit", "video.fill") and not op.get("aspect"):
            errs.append(f"{t} needs aspect")
        if t == "video.overlay":
            img = op.get("image")
            if not isinstance(img, str) or not img:
                errs.append("video.overlay needs image")
            else:
                if any(part == ".." for part in img.replace("\\", "/").split("/")):
                    errs.append("video.overlay image path contains '..'")
                if Path(img).suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    errs.append(f"video.overlay image must be PNG / JPEG: {Path(img).name}")
                if allowed_roots and not any(_under(img, r) for r in allowed_roots):
                    errs.append(f"video.overlay image is outside the allowed input roots: {Path(img).name}")
            if op.get("start") is not None and op.get("end") is not None and not float(op["start"]) < float(op["end"]):
                errs.append("video.overlay: start must be before end")
    subjects_with = {}
    for op in ops:
        subjects_with.setdefault(op.get("asset"), set()).add(op.get("type"))
    for subj, types in subjects_with.items():
        if "video.fit" in types and "video.fill" in types:
            errs.append(f"video.fit and video.fill both on {subj} (conflicting request)")
    return errs


def finishing_operations(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [op for sec in ("color", "graphics", "captions") for op in (d.get(sec) or {}).get("operations") or []]


def check_finishing_operations(d: Dict[str, Any]) -> List[str]:
    """Per-type rules of the finishing sections (ADR-031): explicit types, allowlisted parameters (the schema closes the shapes;
    this checks the references and the chain), references that exist at that point (the subject after its edits), the fixed order
    colour → graphics → captions.burn, one render / one sidecar / one burn / one thumbnail per subject, a LUT / image inside the
    allowed input roots, graphics elements inside the subject's timeline, cues inside the delivered timeline, a burn-in whose
    input is an intermediate (never a source asset), the thumbnail timestamp inside the picture."""
    errs: List[str] = []
    assets = d["assets"]
    subjects_known: set = set(assets) | produced_subjects(d)
    allowed_roots = [str(Path(r)) for r in (d.get("execution") or {}).get("allowed_inputs") or []]
    on_audio = {op["asset"] for op in d["audio"]["operations"] if op.get("type") != "audio.loudness" or "input" in op}
    stage: Dict[str, int] = {}
    seen: Dict[tuple, int] = {}
    scopes: Dict[str, Dict[str, float]] = {}

    def scope_of(subj: str) -> Optional[Dict[str, float]]:
        if subj in scopes:
            return scopes[subj]
        sc = None
        for op in d["video"]["operations"]:
            if op.get("asset") == subj and op.get("temporal_scope"):
                sc = op["temporal_scope"]
        if sc is None and subj in assets:
            dur = (assets[subj].get("technical") or {}).get("duration")
            keep = next((op["keep"] for op in d["video"]["operations"] if op.get("asset") == subj and op.get("type") == "video.trim"), None)
            if keep:
                sc = {"start": 0.0, "end": round(sum(float(e) - float(s_) for s_, e in keep), 3)}
            elif dur:
                sc = {"start": 0.0, "end": float(dur)}
        return sc

    for op in (d.get("color") or {}).get("operations") or []:
        t, subj = op.get("type"), op.get("asset")
        if t not in COLOR_OPERATIONS:
            errs.append(f"unknown color operation type {t!r}")
            continue
        rank = COLOR_ORDER.index(t)
        if stage.get(subj, -1) >= rank:
            errs.append(f"{t} on {subj} is out of the fixed finishing order (colour {COLOR_ORDER} → graphics → captions.burn)")
        stage[subj] = rank
        if subj not in subjects_known:
            errs.append(f"{t} references {subj!r} before it exists")
        if subj in on_audio:
            errs.append(f"{t} on {subj}: the asset is on the audio production path (audio deliverable)")
        if subj in assets and not ((assets[subj].get("technical") or {}).get("video")):
            errs.append(f"{t} on {subj}: no video stream")
        if t == "color.lut":
            lut = op.get("lut")
            if not isinstance(lut, str) or Path(lut).suffix.lower() != ".cube" or any(part == ".." for part in lut.replace("\\", "/").split("/")):
                errs.append("color.lut needs a .cube LUT path without traversal")
            elif allowed_roots and not any(_under(lut, r) for r in allowed_roots):
                errs.append(f"color.lut LUT is outside the allowed input roots: {Path(lut).name}")
        if t == "color.retag" and not op.get("target"):
            errs.append("color.retag needs target")
    for op in (d.get("graphics") or {}).get("operations") or []:
        t, subj = op.get("type"), op.get("asset")
        if subj not in subjects_known:
            errs.append(f"{t} references {subj!r} before it exists")
        if subj in on_audio:
            errs.append(f"{t} on {subj}: the asset is on the audio production path (audio deliverable)")
        if subj in assets and not ((assets[subj].get("technical") or {}).get("video")):
            errs.append(f"{t} on {subj}: no video stream")
        if t == "graphics.render":
            if seen.get(("render", subj)):
                errs.append(f"more than one graphics.render on {subj} (one render per subject)")
            seen[("render", subj)] = 1
            stage[subj] = 10
            sc = scope_of(subj)
            ids = [e.get("id") for e in op.get("elements") or []]
            if len(ids) != len(set(ids)):
                errs.append(f"graphics.render on {subj}: duplicate element ids")
            for e in op.get("elements") or []:
                if e.get("type") not in ELEMENT_TYPES:
                    errs.append(f"graphics.render on {subj}: unknown element type {e.get('type')!r}")
                if not (float(e.get("start", 0)) < float(e.get("end", 0))):
                    errs.append(f"graphics.render element {e.get('id')}: start must be before end")
                if sc and float(e.get("end", 0)) > float(sc["end"]) + 0.01:
                    errs.append(f"graphics.render element {e.get('id')}: ends at {e.get('end')} after the subject's timeline ({sc['end']:.3f})")
                if e.get("type") == "image_overlay":
                    img = (e.get("parameters") or {}).get("image")
                    if not isinstance(img, str) or Path(img).suffix.lower() not in (".png", ".jpg", ".jpeg") or any(part == ".." for part in img.replace("\\", "/").split("/")):
                        errs.append(f"graphics.render element {e.get('id')}: image must be a PNG / JPEG path without traversal")
                    elif allowed_roots and not any(_under(img, r) for r in allowed_roots):
                        errs.append(f"graphics.render element {e.get('id')}: image is outside the allowed input roots")
                if e.get("animation") and e.get("type") not in ("text_overlay", "image_overlay"):
                    errs.append(f"graphics.render element {e.get('id')}: a fade applies to text / image overlays only")
        elif t == "graphics.thumbnail":
            if seen.get(("thumbnail", subj)):
                errs.append(f"more than one graphics.thumbnail on {subj}")
            seen[("thumbnail", subj)] = 1
            sc = scope_of(subj)
            if sc and float(op.get("timestamp", 0)) > float(sc["end"]) + 0.01:
                errs.append(f"graphics.thumbnail on {subj}: timestamp {op.get('timestamp')} is after the picture ends ({sc['end']:.3f})")
            if op.get("text") is not None and not (op.get("width") and op.get("height")):
                errs.append(f"graphics.thumbnail on {subj}: a caption needs the canvas size (width / height)")
        else:
            errs.append(f"unknown graphics operation type {t!r}")
    for op in (d.get("captions") or {}).get("operations") or []:
        t, subj = op.get("type"), op.get("asset")
        if subj not in subjects_known:
            errs.append(f"{t} references {subj!r} before it exists")
        if subj in on_audio:
            errs.append(f"{t} on {subj}: the asset is on the audio production path (audio deliverable)")
        if t == "captions.generate":
            if seen.get(("captions", subj)):
                errs.append(f"more than one captions.generate on {subj}")
            seen[("captions", subj)] = 1
            cues = op.get("cues") or []
            if not cues:
                errs.append(f"captions.generate on {subj}: no cues (an empty sidecar is never planned)")
            sc = scope_of(subj)
            speed = next((float(x["factor"]) for x in d["video"]["operations"] if x.get("asset") == subj and x.get("type") == "video.speed"), 1.0)
            end_limit = (float(sc["end"]) / speed if sc else None)
            prev = -1.0
            for c in cues:
                if not (float(c["start"]) < float(c["end"])):
                    errs.append(f"captions.generate cue {c.get('id')}: start must be before end")
                if float(c["start"]) < prev:
                    errs.append(f"captions.generate cue {c.get('id')}: overlaps the previous cue")
                prev = float(c["end"])
                if end_limit is not None and float(c["end"]) > end_limit + 0.05:
                    errs.append(f"captions.generate cue {c.get('id')}: ends at {c['end']} after the delivered timeline ({end_limit:.3f})")
            for src in op.get("sources") or []:
                if not any(o.get("id") == src and o.get("kind") == "transcript" for o in d["analysis"].get("observations") or []):
                    errs.append(f"captions.generate on {subj}: source {src} is not a transcript observation of this project")
        elif t == "captions.burn":
            if seen.get(("burn", subj)):
                errs.append(f"more than one captions.burn on {subj}")
            seen[("burn", subj)] = 1
            if not seen.get(("captions", subj)):
                errs.append(f"captions.burn on {subj} without captions.generate")
            if op.get("input") in assets:
                errs.append(f"captions.burn on {subj}: the input is a source asset; the burn-in takes an intermediate inside the workspace")
            if op.get("sidecar") != f"{subj}_captions":
                errs.append(f"captions.burn on {subj}: sidecar reference {op.get('sidecar')!r} is not the subject's captions")
            stage[subj] = 20
        else:
            errs.append(f"unknown captions operation type {t!r}")
    return errs


def check_plan_dependencies(d: Dict[str, Any]) -> List[str]:
    """Every step input that is not a source asset must be produced by a step the consuming step (transitively) depends on:
    the plan's dependency graph, not the listing order, is what makes a multi-Skill chain executable in order (ADR-031)."""
    errs: List[str] = []
    steps = (d.get("plan") or {}).get("steps") or []
    by_id = {s["id"]: s for s in steps}
    producer = {o: s["id"] for s in steps for o in (s.get("outputs") or [])}
    assets = set(d.get("assets") or {})

    def deps(sid: str, acc: set) -> set:
        for x in by_id.get(sid, {}).get("depends_on") or []:
            if x not in acc:
                acc.add(x)
                deps(x, acc)
        return acc

    for s in steps:
        closure = deps(s["id"], set())
        for i in s.get("inputs") or []:
            if i in assets:
                continue
            p = producer.get(i)
            if p is None:
                errs.append(f"step {s['id']}: input {i!r} is produced by no step")
            elif p not in closure:
                errs.append(f"step {s['id']}: consumes {i!r} from {p} without depending on it")
    return errs


def _under(path: str, root: str) -> bool:
    try:
        p, r = Path(path).resolve(), Path(root).resolve()
    except OSError:
        return False
    return p == r or r in p.parents


def load_schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_ir(ir: ProjectIR, caps: Optional[Dict[str, Any]] = None, check_paths: bool = True, registry=None, supports=None) -> ValidationReport:
    """registry: SkillRegistry (plan steps must cite implemented skills and their declared tools); supports: callable(tool) -> bool
    from the tool router (the named tool must be executable here). Both optional so pure schema checks stay cheap."""
    rep = ValidationReport()
    try:
        import jsonschema
        validator = jsonschema.Draft202012Validator(load_schema())
        for err in sorted(validator.iter_errors(ir.doc), key=lambda e: list(e.absolute_path)):
            path = "/".join(str(p) for p in err.absolute_path) or "<root>"
            rep.errors.append(f"schema: {path}: {err.message}")
    except ImportError:
        rep.warnings.append("jsonschema not installed; schema validation skipped")
    if rep.errors:
        return rep
    d = ir.doc
    assets = d["assets"]
    ids = {x["id"] for x in d["decisions"]}
    rejected = {x["id"] for x in d["decisions"] if x["status"] == "REJECTED"}
    for op in d["video"]["operations"] + d["audio"]["operations"] + ir.finishing_operations() + d["delivery"]["targets"]:
        for did in op.get("decision_ids") or []:
            if did in rejected:
                rep.errors.append(f"{op.get('type') or 'delivery.' + op.get('id', '?')} cites REJECTED decision {did}; re-plan (revise) before rendering")
    # semantic: operations reference known assets (or the concat programme once it exists) and decisions, ranges inside durations
    rep.errors += check_video_operations(d)
    rep.errors += check_audio_operations(d)
    rep.errors += check_finishing_operations(d)
    rep.errors += check_plan_dependencies(d)
    for op in d["video"]["operations"] + d["audio"]["operations"] + finishing_operations(d):
        if op["asset"] not in assets and op["asset"] not in produced_subjects(d):
            rep.errors.append(f"operation {op['type']} references unknown asset {op['asset']}")
            continue
        for did in op.get("decision_ids") or []:
            if did not in ids:
                rep.errors.append(f"operation {op['type']} cites unknown decision {did}")
    # inferences must cite evidence that exists
    obs_ids = {o["id"] for o in d["analysis"]["observations"]} | {e["id"] for e in d["timeline"]["events"]} | {i["id"] for i in d["analysis"]["inferences"]} \
        | {c["id"] for c in d["analysis"].get("contexts") or []}
    for inf in d["analysis"]["inferences"]:
        for ev in inf["evidence"]:
            if ev not in obs_ids:
                rep.errors.append(f"inference {inf['id']} cites missing evidence {ev}")
    # decision engine invariants (evidence present and known, type, grounding, BLOCK ⇔ BLOCKED, executable citations only)
    rep.errors += check_decisions(d)
    # decisions: BLOCK is never executable; CONFIRM must be approved before render (reported as warning here)
    for dec in d["decisions"]:
        if dec["approval"] == "BLOCK":
            rep.warnings.append(f"decision {dec['id']} ({dec['subject']}) is BLOCKED: {dec['reason']}")
        elif dec["approval"] == "CONFIRM" and dec["status"] == "PROPOSED":
            rep.warnings.append(f"decision {dec['id']} ({dec['subject']}) needs confirmation before execution")
    # delivery presets / platforms must be names the tool knows (catalog-level check, no numbers duplicated)
    known_presets = {"youtube", "youtube4k", "reels", "x", "prores", "h265", "gif"}
    known_platforms = {"youtube", "shorts", "reels", "tiktok", "x", "linkedin", "broadcast", "podcast", "custom"}
    for t in d["delivery"]["targets"]:
        if t.get("preset") and t["preset"] not in known_presets:
            rep.errors.append(f"delivery target {t['id']}: unknown preset {t['preset']}")
        if t.get("platform") not in known_platforms:
            rep.errors.append(f"delivery target {t['id']}: unknown platform {t.get('platform')}")
    # ProductionPlan structure / boundaries (ids, order, dependencies, evidence, decisions, parameters, scopes, status)
    rep.errors += validate_plan(d, registry=registry)
    # plan steps: Skill → Tool consistency (the compiler takes tools from the steps, so this is the execution contract)
    if registry is not None:
        known = set(registry.names())
        for s in d["plan"]["steps"]:
            if s["skill"] not in known:
                rep.errors.append(f"plan step {s['id']} cites unknown skill {s['skill']}")
                continue
            spec = registry.get(s["skill"])
            if not spec.implemented:
                rep.errors.append(f"plan step {s['id']} cites skill {s['skill']} which is declared for phase {spec.phase} and not implemented")
            if not s.get("tool"):
                rep.errors.append(f"plan step {s['id']} ({s['skill']}) has no selected tool")
            elif s["tool"] not in spec.tools:
                rep.errors.append(f"plan step {s['id']}: tool {s['tool']} is not a declared tool of skill {s['skill']} ({', '.join(spec.tools)})")
            elif supports is not None and not supports(s["tool"]):
                rep.errors.append(f"plan step {s['id']}: no registered adapter supports {s['tool']}")
            elif registry.packages() and registry.tool(s["tool"]) is None:
                rep.errors.append(f"plan step {s['id']}: tool {s['tool']} is not declared by any registered skill package")
    # every executable operation must have a step (otherwise the compiler cannot know its tool)
    step_keys = {(s["skill"], (s.get("params") or {}).get("asset") or (s.get("params") or {}).get("target")) for s in d["plan"]["steps"]}
    for op in d["video"]["operations"]:
        if op["type"] == "video.trim" and ("silence_cleanup", op["asset"]) not in step_keys:
            rep.errors.append(f"video.trim on {op['asset']} has no plan step")
        if op["type"] in SKILL_OF and (SKILL_OF[op["type"]], op["asset"]) not in step_keys:
            rep.errors.append(f"{op['type']} on {op['asset']} has no plan step")
    for op in d["audio"]["operations"]:
        if op["type"] == "audio.loudness" and (("audio_normalize" if "input" in op else "loudness_normalization"), op["asset"]) not in step_keys:
            rep.errors.append(f"audio.loudness on {op['asset']} has no plan step")
        if op["type"] in AUDIO_SKILL_OF and op["type"] != "audio.loudness" and op["type"] not in ("audio.mono", "audio.stereo", "audio.downmix") and (AUDIO_SKILL_OF[op["type"]], op["asset"]) not in step_keys:
            rep.errors.append(f"{op['type']} on {op['asset']} has no plan step")
        if op["type"] in ("audio.mono", "audio.stereo", "audio.downmix") and (AUDIO_SKILL_OF[op["type"]], op["asset"]) not in step_keys:
            rep.errors.append(f"{op['type']} on {op['asset']} has no plan step")
    for t in d["delivery"]["targets"]:
        if t.get("preset") and ("delivery_export", t["id"]) not in step_keys:
            rep.errors.append(f"delivery target {t['id']} has no export step")
    for op in (d.get("color") or {}).get("operations") or []:
        if (COLOR_SKILL_OF.get(op["type"], "?"), op["asset"]) not in step_keys:
            rep.errors.append(f"{op['type']} on {op['asset']} has no plan step")
    for op in (d.get("graphics") or {}).get("operations") or []:
        want = GRAPHICS_SKILL if op["type"] == "graphics.render" else (THUMBNAIL_RENDER_SKILL if op.get("text") is not None else THUMBNAIL_FRAME_SKILL)
        if (want, op["asset"]) not in step_keys:
            rep.errors.append(f"{op['type']} on {op['asset']} has no plan step")
    for op in (d.get("captions") or {}).get("operations") or []:
        want = GENERATE_SKILL if op["type"] == "captions.generate" else BURN_SKILL
        if (want, op["asset"]) not in step_keys:
            rep.errors.append(f"{op['type']} on {op['asset']} has no plan step")
    qc = (d.get("qa") or {}).get("qc") or {}
    if qc.get("enabled"):
        qc_steps = [s for s in d["plan"]["steps"] if s["skill"] == QC_SKILL]
        for subj, row in (qc.get("subjects") or {}).items():
            for target in row.get("targets") or []:
                if not any((s.get("params") or {}).get("asset") == subj and (s.get("params") or {}).get("target") == target for s in qc_steps):
                    rep.errors.append(f"qa.qc: {subj}/{target} has no qc_check step")
        for sc, row in (qc.get("sidecars") or {}).items():
            if not any((s.get("params") or {}).get("artifact") == sc for s in qc_steps):
                rep.errors.append(f"qa.qc: sidecar {sc} has no qc_check step")
    # temporal layer: events are validated domain objects on existing assets, within their duration, with real evidence
    durations = {aid: (a.get("technical") or {}).get("duration") for aid, a in assets.items()}
    known_evidence = {o.get("id") for o in d["analysis"].get("observations") or []} | {x.get("id") for x in d["decisions"]} | {i.get("id") for i in d["analysis"].get("inferences") or []}
    events: Dict[str, Event] = {}
    for raw in d["timeline"].get("events") or []:
        try:
            ev = classify(Event.from_dict(raw))
        except (TypeError, ValueError) as ex:
            rep.errors.append(f"event {raw.get('id')}: {ex}")
            continue
        for err in validate_event(ev, durations, known_evidence - {ev.id}):
            rep.errors.append(f"event {ev.id}: {err}")
        events[ev.id] = ev
    for raw in d["timeline"].get("sessions") or []:
        try:
            ses = Session.from_dict(raw)
        except (TypeError, ValueError) as ex:
            rep.errors.append(f"session {raw.get('id')}: {ex}")
            continue
        for err in validate_session(ses, d["project"].get("id"), durations, events):
            rep.errors.append(f"session {ses.id}: {err}")
    # observations are measurements: their source is a tool id + version, never an AI provider; AI output is AI_GENERATED
    for o in d["analysis"].get("observations") or []:
        src = str(o.get("source") or "")
        if "@" not in src or src.startswith("ai"):
            rep.errors.append(f"observation {o.get('id')} has no tool source ({src!r}); only tool measurements may be OBSERVED")
        if o.get("provenance", "OBSERVED") != "OBSERVED":
            rep.errors.append(f"observation {o.get('id')} has provenance {o.get('provenance')!r}; observations are always OBSERVED")
    # production contexts: derived situations whose references all exist and whose ids match their content
    from ..context import ProductionContext, validate_context
    ev_index = {e["id"]: e for e in d["timeline"].get("events") or []}
    for raw in d["analysis"].get("contexts") or []:
        try:
            rep.errors += validate_context(ProductionContext.from_dict(raw), ev_index, durations, {o["id"] for o in d["analysis"].get("observations") or []},
                                           {i["id"] for i in d["analysis"].get("inferences") or []})
        except (KeyError, TypeError, ValueError) as ex:
            rep.errors.append(f"context {raw.get('id')}: malformed: {ex}")
    for i in d["analysis"].get("inferences") or []:
        if str(i.get("kind", "")).startswith("ai_recommendation:") and i.get("provenance") != "AI_GENERATED":
            rep.errors.append(f"inference {i.get('id')} from an AI provider must carry provenance AI_GENERATED")
        if not i.get("evidence"):
            rep.errors.append(f"inference {i.get('id')} cites no evidence")
    # paths
    if check_paths:
        for a in assets.values():
            if not Path(a["path"]).exists():
                rep.errors.append(f"asset {a['id']} path missing: {a['path']}")
    # capabilities: a plan must not silently depend on unavailable capabilities
    if caps is not None:
        # skill-level needs come from the registry (production skill + owning package); preset-level encoder needs below
        needed = set()
        if registry is not None:
            for st in d["plan"]["steps"]:
                if st["skill"] in registry.names():
                    needed.update(registry.get(st["skill"]).required_capabilities)
                pkg = registry.package(str(st.get("tool") or "").split("/", 1)[0])
                if pkg:
                    needed.update(pkg.capabilities)
                    ts = registry.tool(st["tool"]) if st.get("tool") else None
                    if ts:
                        needed.update(ts.required_capabilities)
        if d["video"]["operations"] or any(t.get("preset") for t in d["delivery"]["targets"]):
            needed.add("encoder:libx264")
        if any(op["type"] in SKILL_OF for op in d["video"]["operations"]):
            needed.add("video-editing")   # the editing operations exist only in video-editing-skill (ADR-029): UNKNOWN is not AVAILABLE
        if any(op["type"] in AUDIO_OPERATIONS and (op["type"] != "audio.loudness" or "input" in op) for op in d["audio"]["operations"]):
            needed.add("audio-production")   # the audio production path exists only in audio-production-skill (ADR-030)
        if d["audio"]["operations"]:
            needed.add("filter:loudnorm")
        if (d.get("color") or {}).get("operations"):
            needed.add("color-grading")     # the finishing Skills (ADR-031 / ADR-032): UNKNOWN is not AVAILABLE
        if any(op["type"] == "graphics.render" for op in (d.get("graphics") or {}).get("operations") or []):
            needed.add("motion-graphics")
        if any(op["type"] == "graphics.thumbnail" for op in (d.get("graphics") or {}).get("operations") or []):
            needed.add("thumbnail")
        if (d.get("captions") or {}).get("operations"):
            needed.add("subtitle")
        if ((d.get("qa") or {}).get("qc") or {}).get("enabled"):
            needed.add("qc")
        if any(t.get("preset") == "prores" for t in d["delivery"]["targets"]):
            needed.add("encoder:prores_ks")
        if any(t.get("preset") == "h265" for t in d["delivery"]["targets"]):
            needed.add("encoder:libx265")
        for n in sorted(needed):
            st = getattr(caps.get(n), "status", None) or (caps.get(n) or {}).get("status") if isinstance(caps.get(n), dict) else getattr(caps.get(n), "status", "UNKNOWN")
            if st == "MISSING":
                rep.errors.append(f"required capability MISSING: {n}")
            elif st in ("UNKNOWN", "DEGRADED"):
                rep.warnings.append(f"capability {n} is {st}")
    return rep
