"""Timeline export (ADR-052, amending ADR-002): the IR's cut list as an editor timeline (OpenTimelineIO first; FCPXML and
CMX 3600 EDL through the same tool). Explicit requirement → Decision → ProductionPlan step → IR record → the registry-selected
tool of the `timeline_export` skill; nothing is rendered, the source media are referenced where they are.

- **Requirement** `delivery.timeline`: none (default: no timeline) | otio | fcpxml | edl.
- **What the timeline carries**: the kept source ranges of every source of the subject, in programme order: one clip per
  `video.trim` keep range (the same ranges the cut executes), or one clip of the whole source when nothing trims it; a concat
  programme lists its inputs in the decided order. Hard cuts only (the reference engine's own default is a dissolve on every
  cut; its adapter asks for none). Everything that changes the picture or the sound after the cut
  (speed, resize / fit / fill / overlay, colour, graphics, burn-in, loudness, a concat transition) is not in the timeline;
  the decision names what it leaves out.
- **Refused (BLOCK)**: the audio production path (no picture timeline), a camera switch or a comparison grid (their programme
  is not a cut list of kept ranges), a subject without a video stream."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models import Requirement
from .editing import EditRequirementError

TIMELINE_KEY = "delivery.timeline"
TIMELINE_SKILL = "timeline_export"
TIMELINE_SUBJECT = "timeline.export"
TIMELINE_FORMATS = ("otio", "fcpxml", "edl")
TIMELINE_EXTENSIONS = {"otio": "otio", "fcpxml": "fcpxml", "edl": "edl"}
_OFF = ("none", "false", "no", "off", "0")


def parse_timeline_requirement(m: Dict[str, Requirement]) -> Optional[Dict[str, Any]]:
    """→ {"format", "requirements"} when `delivery.timeline` asks for a timeline, None when absent or `none`. Any other value
    is refused (an unknown format is never guessed)."""
    r = m.get(TIMELINE_KEY)
    if r is None or r.provenance == "DEFAULT" or r.value is None or r.value is False:
        return None
    v = str(r.value).strip().lower() if isinstance(r.value, str) else r.value
    if isinstance(v, str) and v in _OFF:
        return None
    if v not in TIMELINE_FORMATS:
        raise EditRequirementError(f"{TIMELINE_KEY} must be one of none, {', '.join(TIMELINE_FORMATS)}, got {r.value!r}")
    return {"format": v, "requirements": [r]}


def timeline_clips(video_ops: List[Dict[str, Any]], sources: List[str], durations: Dict[str, float]) -> List[Dict[str, Any]]:
    """The cut list of a subject: every source (in programme order) with each range its `video.trim` keeps, or the whole source."""
    clips: List[Dict[str, Any]] = []
    for src in sources:
        trim = next((o for o in video_ops if o.get("type") == "video.trim" and o.get("asset") == src), None)
        keep = trim["keep"] if trim is not None else [[0.0, float(durations.get(src) or 0.0)]]
        for s, e in keep:
            if float(e) > float(s):
                clips.append({"asset": src, "in": round(float(s), 3), "out": round(float(e), 3)})
    return clips


def ir_timeline_export(subject: str, fmt: str, clips: List[Dict[str, Any]], decision_ids: List[str], output: str) -> Dict[str, Any]:
    return {"type": TIMELINE_SUBJECT, "asset": subject, "format": fmt, "clips": [dict(c) for c in clips], "output": output,
            "timeline_duration": round(sum(c["out"] - c["in"] for c in clips), 3), "decision_ids": list(decision_ids)}



# decisions on a subject whose effect a cut-list timeline does not carry (named on the timeline decision, never silently dropped)
_CARRIED = ("video.concat", "subtitle.generate", "thumbnail.render")


def decide_timeline(eng: Any, m: Dict[str, Requirement], analysis: Any, decisions: List[Any], subjects: List[Tuple[str, List[str]]], audio_production: bool,
                    cap_block: Callable[[str, str], Any], approval_for: Callable[..., Dict[str, Any]], probe_ids_of: Callable[[List[str]], List[str]]) -> None:
    """`delivery.timeline` → one DELIVER decision `timeline.export` per video subject (the concat programme or each asset), or a
    BLOCK where a cut-list timeline would misstate the programme. Appends to `eng.decisions`."""
    tl = parse_timeline_requirement(m)
    if tl is None:
        return
    req, fmt = tl["requirements"], tl["format"]
    ev_req = [r.id for r in req]
    live = [d for d in decisions if d.status != "REJECTED"]
    builder = next((d for d in live if d.subject in ("video.switch", "video.grid") and d.type == "TRANSFORM"), None)
    if audio_production or builder is not None:
        why = ("the audio path delivers audio only; there is no picture cut list to export" if audio_production else
               f"the {builder.subject} programme is not a sequence of kept source ranges, so a cut-list timeline would misstate it")
        eng.decide(subject=TIMELINE_SUBJECT, type="BLOCK", decision=f"BLOCK: {fmt} timeline export not possible for this programme", reason=why + " (not guessed)",
                   confidence=1.0, evidence=ev_req + probe_ids_of([a.id for a in analysis.assets]), risk="HIGH", approval="BLOCK", provenance="USER", requirements=req)
        return
    assets = {a.id: a for a in analysis.assets}
    for subject, sources in subjects:
        ev = ev_req + probe_ids_of(sources)
        if not all((assets[a].technical or {}).get("video") for a in sources if a in assets):
            eng.decide(subject=TIMELINE_SUBJECT, type="BLOCK", decision=f"BLOCK: {fmt} timeline of {subject} (no video stream)",
                       reason=f"{subject} has no video stream; an editor timeline here is a picture cut list (unsupported input, not guessed)",
                       confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": subject}, requirements=req)
            continue
        left_out = sorted({d.subject for d in live if d.params.get("asset_id") == subject and ((d.type == "TRANSFORM" and d.subject not in _CARRIED)
                                                                                                  or (d.subject == "audio.loudness" and str(d.decision).startswith("normalize")))})
        if any(d.subject == "video.concat" and d.type == "TRANSFORM" and d.params.get("transition") for d in live):
            left_out.append("video.concat transition")
        eng.decide(subject=TIMELINE_SUBJECT, type="DELIVER", decision=f"export the cut of {subject} as an {fmt} editor timeline (hard cuts, source media referenced in place)",
                   reason=f"user asked for an editor timeline ({req[0].source}); its clips are the kept source ranges the cut executes, in programme order"
                          + (f"; not in the timeline: {', '.join(left_out)}" if left_out else ""),
                   confidence=1.0, evidence=ev, risk="LOW", approval=approval_for(TIMELINE_SUBJECT, explicit=req[0]), provenance="USER",
                   params={"asset_id": subject, "format": fmt, "not_exported": left_out}, requirements=req, serves_intent=None)
    cap_block(TIMELINE_SKILL, f"capability.{TIMELINE_SKILL}")
