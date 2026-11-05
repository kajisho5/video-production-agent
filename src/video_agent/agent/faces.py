"""Blur faces (ADR-090; detection-skill roadmap step 7, agent side): detection-skill's face tracks → a CONFIRM decision that blurs the measured
rectangles for their measured time spans, executed by the reference media engine. Never applied without the user's approval.

    Requirement (`--set edit.redact_faces=true`, explicit) → analysis kind `face_tracks` (detection-skill faces + tracks, a measurement)
    → Observation `face_tracks` (OBSERVED) → TRANSFORM `video.redact_faces` (approval floor CONFIRM) → IR `video.redact_faces`
    → steps `face_redact_segment` (the engine's cut) → `face_redact_blur` (its redact) → `face_redact_composite` (its broll), tools from the registry

- **Requirements** (`--set` or a brief; a refinement without the switch, an unknown key or a value out of range is refused before
  analysis): `edit.redact_faces=true`; `edit.redact_faces.mode=blur|pixelate` (redact.py's modes; delogo is not a moving region),
  `.blur_strength` / `.block_size` (redact.py's own parameters, sent only when given), `.every` (seconds between sampled frames,
  default EVERY_DEFAULT = 0.5: detection-skill's SKILL.md asks for 0.2-0.5 s for redaction, its CLI default of 1 s is for
  finding, not hiding), `.min_score` (detection-skill's own threshold, its default 0.6 when absent), `.window` (one rectangle
  per that many seconds of a track instead of one per track, detection-skill's `tracks --window`; default 0 = one per track).
- **Measurement**: `detection faces --every` on the source, then `detection tracks`: boxes linked over time into rectangles padded
  and extended by one sampling step (the Skill's documented heuristic). Faces only; a face box is never an identity.
- **Regions** (pure, `redaction_regions` / `elementary_segments`): every `redact` rectangle of every track over its own [start, end];
  the timeline is cut at every rectangle boundary into elementary segments, each with the set of rectangles active in it.
- **Decision** per video input that is measured: TRANSFORM `video.redact_faces` (risk HIGH, provenance OBSERVED, evidence the
  observation + probe, approval from `video.redact_faces.approval` with floor CONFIRM: the explicit requirement does not waive it,
  a profile cannot make it AUTO -- the blur is an irreversible change to the delivered picture and the detector misses faces).
  No track → KEEP (nothing blurred) with the reason and the sampling, so the user checks the picture themselves. BLOCK (never
  guessed): the measurement failed (the Skill's own code), the source is rotated (the boxes are in display pixels, redact.py's in
  stored pixels), the measured frame is not the probed frame, more than MAX_SEGMENTS segments (ask for a coarser `.window`).
  Without detection-skill (or FACES not ready: opencv, the YuNet file or the engine's frames missing, as its doctor says) the
  measurement is handed off: TRANSFORM (the ask, `handoff: true`) + BLOCK naming detection-skill and its install.
- **Execution** (compiler, before the input's trim: the rectangles are on the source timeline): per elementary segment, cut
  [start, end] out of the source frame-accurately, redact each active rectangle on that piece, then one broll on the source with
  every redacted piece as a cutaway at its own time (`--audio a`: the source's audio untouched; the output is exactly as long as the
  source). Everything else of the plan continues on that output.
- **What it is not**: OCR / objects (text, badges, screens) are not wired; nothing names anyone; the result needs a human check of
  the rendered video before it is published (a sampled detector misses faces between samples and small / turned-away faces)."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models import Requirement
from ..tools.detection.adapter import KIND
from .editing import EditRequirementError

SWITCH = "edit.redact_faces"
SUBJECT = "video.redact_faces"
MEASURE_SKILL = "face_detection"
SEGMENT_SKILL, BLUR_SKILL, COMPOSITE_SKILL = "face_redact_segment", "face_redact_blur", "face_redact_composite"
EXEC_SKILLS = (SEGMENT_SKILL, BLUR_SKILL, COMPOSITE_SKILL)
TOOL_NAMES = {SEGMENT_SKILL: "cut", BLUR_SKILL: "redact", COMPOSITE_SKILL: "broll"}   # the tool each skill's arguments are shaped for (ids come from the registry)
EVERY_DEFAULT = 0.5
MAX_SEGMENTS = 60          # one cut + its redacts per segment, all composited by one broll: more is a request for a coarser window
MODES = ("blur", "pixelate")
REQUIREMENT_KEYS = (SWITCH, SWITCH + ".mode", SWITCH + ".blur_strength", SWITCH + ".block_size", SWITCH + ".every", SWITCH + ".min_score", SWITCH + ".window")
APPROVAL_KEY = ("video.redact_faces.approval", "CONFIRM")
_ON = (True, "true", "yes", "on", "1")
_OFF = (False, None, "false", "no", "off", "0", "none")


def _number(v: Any, key: str, lo: float, hi: float, integer: bool = False) -> float:
    if isinstance(v, str):
        try:
            v = float(v.strip()) if not integer else int(v.strip())
        except ValueError:
            raise EditRequirementError(f"{key} must be a number from {lo:g} to {hi:g}, got {v!r}") from None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not lo <= float(v) <= hi or (integer and float(v) != int(v)):
        raise EditRequirementError(f"{key} must be {'an integer' if integer else 'a number'} from {lo:g} to {hi:g}, got {v!r}")
    return int(v) if integer else float(v)


def parse_face_requirements(values: Dict[str, Any]) -> Tuple[bool, Dict[str, float], Dict[str, Any]]:
    """(requested, the detection parameters, the redact parameters) from requirement values; EditRequirementError on anything refused."""
    given = {k: v for k, v in (values or {}).items() if k == SWITCH or k.startswith(SWITCH + ".")}
    unknown = sorted(k for k in given if k not in REQUIREMENT_KEYS)
    if unknown:
        raise EditRequirementError(f"unknown requirement key(s): {', '.join(unknown)}; known: {', '.join(REQUIREMENT_KEYS)}")
    if not given:
        return False, {}, {}
    sw = given.get(SWITCH)
    sw = sw.strip().lower() if isinstance(sw, str) else sw
    if SWITCH in given and sw not in _ON + _OFF:
        raise EditRequirementError(f"{SWITCH} is a switch (true / false), got {given[SWITCH]!r}")
    on = SWITCH in given and sw in _ON
    refinements = [k for k in given if k != SWITCH and given[k] is not None]
    if refinements and not on:
        raise EditRequirementError(f"{', '.join(sorted(refinements))} refine {SWITCH}, which is not set to true")
    if not on:
        return False, {}, {}
    detect: Dict[str, float] = {"every": _number(given.get(SWITCH + ".every", EVERY_DEFAULT), SWITCH + ".every", 0.04, 5.0)}
    if given.get(SWITCH + ".min_score") is not None:
        detect["min_score"] = _number(given[SWITCH + ".min_score"], SWITCH + ".min_score", 0.0, 1.0)
    if given.get(SWITCH + ".window") is not None:
        detect["window"] = _number(given[SWITCH + ".window"], SWITCH + ".window", 0.0, 3600.0)
    mode = str(given.get(SWITCH + ".mode") or "blur").strip().lower()
    if mode not in MODES:
        raise EditRequirementError(f"{SWITCH}.mode must be one of {', '.join(MODES)} (a face moves: delogo is for a fixed rectangle), got {given[SWITCH + '.mode']!r}")
    redact: Dict[str, Any] = {"mode": mode}
    if given.get(SWITCH + ".blur_strength") is not None:
        if mode != "blur":
            raise EditRequirementError(f"{SWITCH}.blur_strength applies to mode blur only")
        redact["blur_strength"] = _number(given[SWITCH + ".blur_strength"], SWITCH + ".blur_strength", 1, 256, integer=True)
    if given.get(SWITCH + ".block_size") is not None:
        if mode != "pixelate":
            raise EditRequirementError(f"{SWITCH}.block_size applies to mode pixelate only")
        redact["block_size"] = _number(given[SWITCH + ".block_size"], SWITCH + ".block_size", 2, 256, integer=True)
    return True, detect, redact


def check_face_keys(user_requirements: Dict[str, Any]) -> None:
    parse_face_requirements(user_requirements)


def face_kind(user_requirements: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, float]]]:
    """(analysis kind, its parameters) when the request asks to blur faces, else None."""
    on, detect, _ = parse_face_requirements(user_requirements)
    return (KIND, detect) if on else None


def redaction_regions(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every redact rectangle of every track over its own span, as the Skill measured it (start, end, x, y, width, height)."""
    out = []
    for t in tracks or []:
        for r in t.get("redact") or []:
            out.append({"start": round(float(r["start"]), 3), "end": round(float(r["end"]), 3), "x": int(r["x"]), "y": int(r["y"]),
                        "width": int(r["width"]), "height": int(r["height"])})
    return sorted(out, key=lambda r: (r["start"], r["end"], r["x"], r["y"]))


def elementary_segments(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The timeline cut at every rectangle boundary: [{start, end, rects}] for each piece where at least one rectangle is active, in time
    order, each rectangle listed once (an identical rectangle of two tracks is blurred once)."""
    cuts = sorted({r["start"] for r in regions} | {r["end"] for r in regions})
    out: List[Dict[str, Any]] = []
    for s, e in zip(cuts, cuts[1:]):
        if e - s < 0.001:
            continue
        rects: List[Dict[str, int]] = []
        for r in regions:
            if r["start"] <= s and r["end"] >= e:
                rect = {k: r[k] for k in ("x", "y", "width", "height")}
                if rect not in rects:
                    rects.append(rect)
        if rects:
            out.append({"start": round(s, 3), "end": round(e, 3), "rects": rects})
    return out


def _words(rects: List[Dict[str, int]]) -> str:
    return ", ".join(f"{r['width']}x{r['height']}+{r['x']}+{r['y']}" for r in rects)


def decide_faces(eng: Any, m: Dict[str, Requirement], analysis: Any, approval_for: Callable, probe_ids_of: Callable[[List[str]], List[str]],
                 cap_missing: Callable[[str], Optional[str]], cap_block: Callable[[str, str], Any]) -> None:
    """One decision per video input when `edit.redact_faces` is asked for (see the module docstring)."""
    r = m.get(SWITCH)
    if r is None or r.provenance == "DEFAULT":
        return
    values = {k: req.value for k, req in m.items() if k == SWITCH or k.startswith(SWITCH + ".")}
    on, detect, redact = parse_face_requirements(values)
    if not on:
        return
    reqs = [m[k] for k in sorted(values) if k in m]
    video = [a for a in analysis.assets if (a.technical or {}).get("video")]
    why = cap_missing(MEASURE_SKILL)
    if why:   # the measurement cannot run here: recorded and handed off, never dropped (ADR-051 / ADR-057)
        eng.decide(subject=SWITCH, type="TRANSFORM", decision=f"{KIND}: face boxes linked over time on every video input, then blurred",
                   reason=f"user asked for {SWITCH} ({r.source}); no tool for {MEASURE_SKILL} here, so the face detection is handed off to detection-skill",
                   confidence=1.0, evidence=[x.id for x in reqs] + probe_ids_of([a.id for a in video]), risk="LOW", approval="CONFIRM", provenance="USER",
                   params={"skill": MEASURE_SKILL, "handoff": True}, requirements=reqs)
        eng.decide(subject=SWITCH, type="BLOCK", decision=f"BLOCK: {KIND} is handed off ({MEASURE_SKILL} unavailable)",
                   reason=f"{why}; install detection-skill (and its YuNet model file) and plan again, or plan without {SWITCH} (see the plan's handoff)",
                   confidence=1.0, evidence=[f"skill:{MEASURE_SKILL}"], risk="HIGH", approval="BLOCK", provenance="SYSTEM", params={"skill": MEASURE_SKILL, "missing": ["detection"]})
        return
    if not video:
        eng.decide(subject=SUBJECT, type="BLOCK", decision="BLOCK: no video input to blur faces in", reason=f"user asked for {SWITCH}; no input has a video stream",
                   confidence=1.0, evidence=[x.id for x in reqs], risk="HIGH", approval="BLOCK", provenance="USER", params={}, requirements=reqs)
        return
    rows = [row for a in getattr(analysis, "analyses", []) or [] for row in a.get("rows") or [] if row.get("kind") == KIND]
    proposed = False
    for asset in video:
        ev = [x.id for x in reqs] + probe_ids_of([asset.id])
        failed = next((row for row in rows if row.get("asset_id") == asset.id and row.get("status") == "FAILED"), None)
        o = next((x for x in analysis.observations if x.kind == KIND and x.asset_id == asset.id), None)
        if o is None:
            err = (failed or {}).get("error") or {}
            code = err.get("skill_error") or err.get("kind") or "NOT_MEASURED"
            eng.decide(subject=SUBJECT, type="BLOCK", decision=f"BLOCK: face detection failed on {asset.id} ({code})",
                       reason=f"user asked for {SWITCH}; the measurement did not produce face tracks: {code}: {str(err.get('message') or '')[:200]} (nothing is guessed in its place)",
                       confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK", provenance="SYSTEM", params={"asset_id": asset.id, "code": code}, requirements=reqs)
            continue
        ev = [o.id] + ev
        data = o.data or {}
        rot = ((asset.technical or {}).get("video") or {}).get("rotation") or 0
        fs = data.get("frame_size") or {}
        pw, ph = ((asset.technical or {}).get("video") or {}).get("width"), ((asset.technical or {}).get("video") or {}).get("height")
        samp = (data.get("detections") or {}).get("sampling") or {}
        sampled = f"{samp.get('frames')} frame(s) sampled every {samp.get('every')} s"
        if rot not in (0, None):
            eng.decide(subject=SUBJECT, type="BLOCK", decision=f"BLOCK: {asset.id} is rotated {rot}°", reason="the face boxes are in display pixels and redact.py's rectangle in stored pixels; "
                       "a rotated source is not mapped by guessing (rotate the source first)", confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK", provenance="SYSTEM",
                       params={"asset_id": asset.id}, requirements=reqs)
            continue
        if (fs.get("width"), fs.get("height")) != (pw, ph):
            eng.decide(subject=SUBJECT, type="BLOCK", decision=f"BLOCK: measured frame {fs.get('width')}x{fs.get('height')} is not {asset.id}'s {pw}x{ph}",
                       reason="the rectangles would land elsewhere on the picture; not rescaled by guessing", confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK",
                       provenance="SYSTEM", params={"asset_id": asset.id}, requirements=reqs)
            continue
        tracks = data.get("tracks") or []
        segments = elementary_segments(redaction_regions(tracks))
        if not segments:
            eng.decide(subject=SUBJECT, type="KEEP", decision=f"no face detected in {asset.id}; nothing is blurred",
                       reason=f"detection-skill found no face in {sampled} (min score {(data.get('detections') or {}).get('parameters', {}).get('min_score')}); "
                              "a sampled detector misses small, turned-away and briefly visible faces: check the picture before publishing",
                       confidence=1.0, evidence=ev, risk="MEDIUM", approval="AUTO", provenance="OBSERVED", params={"asset_id": asset.id, "tracks": 0}, requirements=reqs)
            continue
        if len(segments) > MAX_SEGMENTS:
            eng.decide(subject=SUBJECT, type="BLOCK", decision=f"BLOCK: {len(segments)} redaction segments on {asset.id} (more than {MAX_SEGMENTS})",
                       reason=f"{len(tracks)} face track(s) give {len(segments)} time segments to cut, blur and composite; ask for a coarser {SWITCH}.window "
                              "(or none: one rectangle per track) instead of blurring piece by piece", confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK",
                       provenance="SYSTEM", params={"asset_id": asset.id, "segments": len(segments)}, requirements=reqs)
            continue
        covered = round(sum(s["end"] - s["start"] for s in segments), 3)
        eng.decide(subject=SUBJECT, type="TRANSFORM",
                   decision=f"video.redact_faces on {asset.id}: {redact['mode']} {len(tracks)} face track(s) over {covered:.2f}s in {len(segments)} segment(s)",
                   reason=f"user asked for {SWITCH} ({r.source}); detection-skill ({o.source}) linked face boxes from {sampled} into {len(tracks)} track(s); "
                          f"each rectangle is blurred only during its own span (first: {segments[0]['start']:.2f}-{segments[0]['end']:.2f}s, {_words(segments[0]['rects'])}). "
                          "Irreversible on the delivered picture and a detector misses faces: look at the rendered video before publishing",
                   confidence=1.0, evidence=ev, risk="HIGH", approval=approval_for(SUBJECT, floor="CONFIRM"), provenance="OBSERVED",
                   params={"asset_id": asset.id, "regions": segments, **redact, "tracks": len(tracks)}, requirements=reqs, serves_intent=None)
        proposed = True
    if proposed:
        for sk in EXEC_SKILLS:
            cap_block(sk, f"capability.{sk}")


IR_KEYS = ("type", "asset", "input", "output", "regions", "mode", "blur_strength", "block_size", "decision_ids", "temporal_scope")


def ir_redact_faces(asset_id: str, current: str, d: Any) -> Dict[str, Any]:
    """The IR operation of an approved / pending blur-faces decision: the source as input, its regions as decided, redact.py's parameters."""
    op: Dict[str, Any] = {"type": SUBJECT, "asset": asset_id, "input": current, "output": f"{asset_id}_redact_faces",
                          "regions": [{"start": float(s["start"]), "end": float(s["end"]), "rects": [dict(r) for r in s["rects"]]} for s in d.params["regions"]],
                          "mode": d.params.get("mode", "blur")}
    for k in ("blur_strength", "block_size"):
        if d.params.get(k) is not None:
            op[k] = int(d.params[k])
    op["decision_ids"] = [d.id]
    return op


def face_problems(op: Dict[str, Any], assets: Dict[str, Any]) -> List[str]:
    """Validator rules of a `video.redact_faces` operation: its own vocabulary only, on a video source (as its input, so it runs before the
    source is re-timed), regions in time order without overlap inside the source, each with at least one even-sized rectangle inside the probed
    frame, redact.py's mode and its one parameter for that mode."""
    errs: List[str] = []
    subj = op.get("asset")
    extra = sorted(k for k in op if k not in IR_KEYS)
    if extra:
        errs.append(f"{SUBJECT}: parameters {extra} are not in the operation's vocabulary")
    a = assets.get(subj)
    if a is None:
        return errs + [f"{SUBJECT} on {subj!r}: not an asset (faces are blurred on a source, before it is re-timed)"]
    tech = a.get("technical") or {}
    v = tech.get("video") or {}
    if not v:
        return errs + [f"{SUBJECT} on {subj}: no video stream"]
    if op.get("input") != subj or op.get("output") != f"{subj}_redact_faces":
        errs.append(f"{SUBJECT} on {subj}: input must be the source itself and output {subj}_redact_faces")
    if op.get("mode") not in MODES:
        errs.append(f"{SUBJECT} on {subj}: mode {op.get('mode')!r} is not one of {', '.join(MODES)}")
    if op.get("blur_strength") is not None and (op.get("mode") != "blur" or isinstance(op["blur_strength"], bool) or not isinstance(op["blur_strength"], int) or not 1 <= op["blur_strength"] <= 256):
        errs.append(f"{SUBJECT} on {subj}: blur_strength {op['blur_strength']!r} needs mode blur and 1..256")
    if op.get("block_size") is not None and (op.get("mode") != "pixelate" or isinstance(op["block_size"], bool) or not isinstance(op["block_size"], int) or not 2 <= op["block_size"] <= 256):
        errs.append(f"{SUBJECT} on {subj}: block_size {op['block_size']!r} needs mode pixelate and 2..256")
    regions = op.get("regions")
    if not isinstance(regions, list) or not regions or len(regions) > MAX_SEGMENTS:
        return errs + [f"{SUBJECT} on {subj}: regions must be 1..{MAX_SEGMENTS} segments"]
    dur = float(tech.get("duration") or 0.0)
    w, h = v.get("width"), v.get("height")
    prev = 0.0
    for n, reg in enumerate(regions):
        s, e = (reg or {}).get("start"), (reg or {}).get("end")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (s, e)) or not 0 <= s < e or (dur and e > dur + 0.01) or s < prev - 1e-6:
            errs.append(f"{SUBJECT} on {subj}: region {n} {s!r}..{e!r} is not inside the source after the previous one")
            continue
        prev = e
        rects = reg.get("rects")
        if not isinstance(rects, list) or not rects:
            errs.append(f"{SUBJECT} on {subj}: region {n} has no rectangle")
            continue
        for r in rects:
            ok = isinstance(r, dict) and set(r) == {"x", "y", "width", "height"} and all(isinstance(r[k], int) and not isinstance(r[k], bool) for k in r)
            if not ok or r["width"] < 2 or r["height"] < 2 or r["width"] % 2 or r["height"] % 2 or r["x"] < 0 or r["y"] < 0 \
                    or (isinstance(w, int) and r["x"] + r["width"] > w) or (isinstance(h, int) and r["y"] + r["height"] > h):
                errs.append(f"{SUBJECT} on {subj}: region {n} rectangle {r!r} is not an even-sized integer rectangle inside the {w}x{h} frame")
    return errs
