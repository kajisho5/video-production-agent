"""Reading the editor's timeline back (ADR-098; ECOSYSTEM_ROADMAP step 12, the second half of nle-bridge-skill's step 7 "(agent side) adapter").

The cut handed to Premiere Pro / DaVinci Resolve (ADR-089: `delivery.nle`, from the subject's `delivery.timelines` record, ADR-052) comes back as
OTIO / FCP7 XML / CMX 3600 EDL. `video-agent revise <project> --editor-timeline <file>` runs `nle-bridge diff --timeline-out` against the project
IR (the measurement: what the editor moved, trimmed, removed, added; nle-bridge-skill's own reading) and records it as one observation of kind
`nle.diff` on the subject. From that observation the next plan version gets, per source whose cut list the editor changed, one CONFIRM decision
`timeline.editor_cut` (TRANSFORM) whose kept source ranges are the editor's, and one REVIEW decision `timeline.editor_markers` for the markers the
editor added / removed / changed (the plan carries no markers: they are recorded, never executed). The planner cuts that source to exactly the
editor's ranges (frame-accurate) instead of the ranges its silence / trim / take decisions would give, so the next render, the next timeline
record and the next nle-bridge export have the editor's cuts.

What is read: the first picture track (V1) of the returned timeline, in frames at its rate, mapped back to the plan's asset ids by nle-bridge.
What is refused (nothing is guessed, the plan is not revised): a second picture track with clips, titles / generators on V1, media the plan does
not have, a speed change (record length != source length), a source whose clips are out of source order or overlap (a re-ordered cut is not a
list of kept ranges), sources re-ordered or a source removed entirely from a programme (the concat order and inputs are decisions of their own),
and the audio production path. Read but not carried, each a named warning: gaps on V1 (the render joins the kept ranges), disabled clips (left
out), transitions (hard cuts, as the agent's timeline export), audio edits that differ from the picture (the sound follows the picture), and
nle-bridge's own `not_imported` list (effects, titles the editor added)."""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

from .timeline import TIMELINE_SUBJECT

OBS_KIND = "nle.diff"
SUBJECT = "timeline.editor_cut"
MARKERS_SUBJECT = "timeline.editor_markers"
APPROVAL_KEYS = {SUBJECT: (f"{SUBJECT}.approval", "CONFIRM")}


class EditorTimelineError(ValueError):
    """The returned timeline cannot be read back as kept ranges of the plan's sources (the reason names what to change)."""


def parse_rate(value: Any) -> Fraction:
    s = str(value or "").strip()
    try:
        r = Fraction(int(s.split("/")[0]), int(s.split("/")[1])) if "/" in s else Fraction(s)
    except (ValueError, ZeroDivisionError):
        raise EditorTimelineError(f"the returned timeline's rate {value!r} is not a frame rate")
    if r <= 0:
        raise EditorTimelineError(f"the returned timeline's rate {value!r} is not a frame rate")
    return r


def to_frames(seconds: float, rate: Fraction) -> int:
    """Seconds → the nearest frame, as nle-bridge reads the plan's seconds (exact rationals, ties to even)."""
    return int(round(Fraction(str(seconds)) * rate))


def to_seconds(frames: int, rate: Fraction) -> float:
    return round(float(Fraction(int(frames)) / rate), 3)


def timeline_record(doc: Dict[str, Any], subject: Optional[str]) -> Dict[str, Any]:
    recs = [t for t in (doc.get("delivery") or {}).get("timelines") or [] if t.get("type") == TIMELINE_SUBJECT]
    if not recs:
        raise EditorTimelineError("the plan has no timeline to compare with (delivery.timelines): it was not exported to an editor; plan it with "
                                  "--set delivery.timeline=otio (and delivery.nle=export) and render first")
    if subject is None:
        if len(recs) > 1:
            raise EditorTimelineError("the plan has several timelines: pass --subject " + " | ".join(str(r.get("asset")) for r in recs))
        return recs[0]
    rec = next((r for r in recs if r.get("asset") == subject), None)
    if rec is None:
        raise EditorTimelineError(f"no timeline of subject {subject!r} in the plan (have: {', '.join(str(r.get('asset')) for r in recs)})")
    return rec


def _plan_keep(rec: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[List[float]]]]:
    order: List[str] = []
    keep: Dict[str, List[List[float]]] = {}
    for c in rec.get("clips") or []:
        a = str(c["asset"])
        if a not in keep:
            order.append(a)
            keep[a] = []
        keep[a].append([float(c["in"]), float(c["out"])])
    return order, keep


def _warn(code: str, message: str) -> Dict[str, str]:
    return {"code": code, "message": message}


def read_back(doc: Dict[str, Any], subject: str, diff: Dict[str, Any], timeline: Dict[str, Any]) -> Dict[str, Any]:
    """nle-bridge's diff + returned timeline (timeline-out) → {"rate", "plan_keep", "editor_keep" (only the sources whose cut list changed),
    "changes" (per source, from the diff), "markers", "warnings"}. Raises EditorTimelineError for what cannot be read back."""
    rec = timeline_record(doc, subject)
    order, plan_keep = _plan_keep(rec)
    rate = parse_rate(timeline.get("rate"))
    warnings: List[Dict[str, str]] = []
    video = [t for t in timeline.get("tracks") or [] if t.get("kind") == "video"]
    if not video:
        raise EditorTimelineError("the returned timeline has no picture track")
    for t in video[1:]:
        if any(it.get("enabled", True) for it in t.get("items") or []):
            raise EditorTimelineError(f"the returned timeline has clips on a second picture track ({t.get('name')}); the agent renders one cut list "
                                      "(V1): flatten the edit onto V1 and export it again")
    assets = timeline.get("assets") or {}
    items = sorted((dict(it) for it in video[0].get("items") or []), key=lambda it: int(it["record_in"]))
    seq: List[Tuple[str, int, int]] = []
    pos = 0
    for it in items:
        name = it.get("name") or it.get("id")
        if (it.get("kind") or "clip") != "clip":
            raise EditorTimelineError(f"V1 holds a {it.get('kind')} ({name}); a title or generator is not a cut of the plan's sources: remove it from V1")
        if not it.get("enabled", True):
            warnings.append(_warn("DISABLED_CLIP", f"the disabled clip {name} on V1 is left out"))
            continue
        a = str(it.get("asset"))
        if a not in plan_keep:
            path = (assets.get(a) or {}).get("path")
            raise EditorTimelineError(f"V1 clip {name} uses media the plan does not cut ({path or a}); adding media is a new plan, not a revision")
        ri, ro, si, so = (int(it[k]) for k in ("record_in", "record_out", "source_in", "source_out"))
        if ro - ri != so - si:
            raise EditorTimelineError(f"V1 clip {name} plays {so - si} source frames in {ro - ri} frames (a speed change); the agent reads cuts only")
        if ri > pos:
            warnings.append(_warn("GAP_CLOSED", f"a gap of {ri - pos} frame(s) before {name} on V1 is closed (the render joins the kept ranges)"))
        if ri < pos:
            raise EditorTimelineError(f"V1 clip {name} overlaps the clip before it")
        pos = ro
        if it.get("transition_in"):
            warnings.append(_warn("TRANSITION_NOT_READ", f"the transition into {name} is not read back (hard cut)"))
        seq.append((a, si, so))
    if not seq:
        raise EditorTimelineError("V1 of the returned timeline has no clip of the plan's sources")
    returned_order: List[str] = []
    for a, _si, _so in seq:
        if not returned_order or returned_order[-1] != a:
            if a in returned_order:
                raise EditorTimelineError(f"the clips of {a} are not together on V1 (another source is cut in between); the agent's programme plays "
                                          "each source's kept ranges in one run")
            returned_order.append(a)
    if returned_order != order:
        missing = [a for a in order if a not in returned_order]
        if missing:
            raise EditorTimelineError(f"every clip of {', '.join(missing)} was removed; dropping a source from the programme is not read back "
                                      "(reject its decision or plan without it)")
        raise EditorTimelineError(f"the sources are in another order on V1 ({' → '.join(returned_order)}, the plan: {' → '.join(order)}); the "
                                  "programme order is not read back")
    editor_frames: Dict[str, List[List[int]]] = {}
    for a, si, so in seq:
        rs = editor_frames.setdefault(a, [])
        if rs and si < rs[-1][1]:
            raise EditorTimelineError(f"the clips of {a} on V1 are out of source order or overlap at source frame {si}; a re-ordered cut is not "
                                      "a list of kept ranges")
        if rs and si == rs[-1][1]:
            rs[-1][1] = so   # a razor cut with nothing removed is one kept range
        else:
            rs.append([si, so])
    plan_frames = {a: [[to_frames(s, rate), to_frames(e, rate)] for s, e in ks] for a, ks in plan_keep.items()}
    for a in plan_frames:   # the plan's ranges, adjacent ones joined the same way
        joined: List[List[int]] = []
        for s, e in plan_frames[a]:
            if joined and s == joined[-1][1]:
                joined[-1][1] = e
            else:
                joined.append([s, e])
        plan_frames[a] = joined
    returned_keep = {a: [[to_seconds(s, rate), to_seconds(e, rate)] for s, e in rs] for a, rs in editor_frames.items()}
    editor_keep = {a: ks for a, ks in returned_keep.items() if editor_frames[a] != plan_frames.get(a)}
    for t in [t for t in timeline.get("tracks") or [] if t.get("kind") == "audio"]:
        cuts = [(str(it.get("asset")), int(it["source_in"]), int(it["source_out"])) for it in sorted(t.get("items") or [], key=lambda x: int(x["record_in"]))
                if it.get("enabled", True)]
        if cuts and cuts != seq:
            warnings.append(_warn("AUDIO_FOLLOWS_PICTURE", f"the audio edits on {t.get('name')} differ from V1 (a J / L cut, a moved or removed sound clip); "
                                                           "they are not read back: the sound is cut with the picture"))
            break
    for w in diff.get("not_imported") or []:
        warnings.append(_warn(str((w or {}).get("code") or "NOT_IMPORTED"), str((w or {}).get("message") or w)))
    changes: Dict[str, List[str]] = {}
    for c in diff.get("clips") or []:
        if c.get("track") == "V1" and c.get("status") == "changed" and c.get("asset"):
            changes.setdefault(str(c["asset"]), []).append(f"{c.get('name') or c.get('plan_id')}: {', '.join(c.get('changes') or [])}")
    for c in diff.get("removed") or []:
        if c.get("track") == "V1" and c.get("asset"):
            changes.setdefault(str(c["asset"]), []).append(f"{c.get('name') or c.get('plan_id')}: removed ({c.get('record_in')}-{c.get('record_out')})")
    for c in diff.get("added") or []:
        if c.get("track") == "V1" and c.get("asset"):
            changes.setdefault(str(c["asset"]), []).append(f"{c.get('name') or c.get('returned_id')}: added" + (f" (split from {c['split_from']})" if c.get("split_from") else ""))
    markers = diff.get("markers") or {}
    summary = diff.get("summary") or {}
    return {"subject": subject, "rate": str(timeline.get("rate")), "plan_keep": plan_keep, "editor_keep": editor_keep, "returned_keep": returned_keep,
            "editor_frames": editor_frames,
            "changes": {a: changes.get(a, []) for a in editor_keep}, "warnings": warnings,
            "markers": {k: list(markers.get(k) or []) for k in ("added", "removed", "changed")},
            "summary": {k: summary.get(k) for k in ("clips_unchanged", "clips_changed", "clips_added", "clips_removed", "cuts_moved", "cuts_added", "cuts_removed",
                                                     "markers_added", "markers_removed", "markers_changed", "duration_delta_frames", "identical")},
            "cuts": [c for c in diff.get("cuts") or [] if c.get("status") != "unchanged"]}


def readbacks(analysis: Any, subject: str) -> List[Any]:
    """The `nle.diff` observations of the subject, oldest first."""
    return [o for o in analysis.observations if o.kind == OBS_KIND and o.asset_id == subject]


def latest_readback(analysis: Any, subject: str) -> Optional[Any]:
    """The last `nle.diff` observation of the subject: its timeline is the editor's current cut (an earlier read-back is history)."""
    obs = readbacks(analysis, subject)
    return obs[-1] if obs else None


def is_noop(data: Dict[str, Any]) -> bool:
    """A read-back that changes nothing in the plan it was compared with: no source's kept ranges, no marker."""
    m = data.get("markers") or {}
    return not data.get("editor_keep") and not any(m.get(k) for k in ("added", "removed", "changed"))


def decide_editor_cut(eng: Any, analysis: Any, subjects: List[Tuple[str, List[str]]], audio_production: bool, approval_for: Any,
                      probe_ids_of: Any) -> None:
    """One CONFIRM `timeline.editor_cut` per source whose kept ranges the editor changed in any read-back of the subject, with the ranges of the
    latest read-back (the editor's current timeline), and one REVIEW `timeline.editor_markers` for the latest read-back's marker edits.
    Nothing without a read-back."""
    for subject, sources in subjects:
        allr = readbacks(analysis, subject)
        if not allr:
            continue
        o = allr[-1]
        d = o.data
        changed = {a for r in allr for a in (r.data.get("editor_keep") or {})}
        if audio_production:
            eng.decide(subject=SUBJECT, type="BLOCK", decision=f"BLOCK: the editor's timeline of {subject} on the audio production path",
                       reason="the audio production path has no picture timeline to take the editor's cuts into", confidence=1.0, evidence=[o.id],
                       risk="HIGH", approval="BLOCK", provenance="OBSERVED", params={"asset_id": subject})
            continue
        ret = (d.get("returned") or {})
        for aid in sources:
            keep = (d.get("returned_keep") or {}).get(aid) if aid in changed else None
            if keep is None:
                continue
            kept = round(sum(e - s for s, e in keep), 3)
            plan = (d.get("plan_keep") or {}).get(aid) or []
            what = "; ".join((d.get("changes") or {}).get(aid) or []) or "as in the earlier read-back"
            eng.decide(subject=SUBJECT, type="TRANSFORM",
                       decision=f"cut {aid} as the editor's timeline: {len(keep)} range(s), {kept:g}s kept (the plan: {len(plan)} range(s), "
                                f"{round(sum(e - s for s, e in plan), 3):g}s)",
                       reason=f"the editor's timeline {ret.get('file', '').rsplit('/', 1)[-1]} (sha256 {str(ret.get('sha256') or '')[:12]}, read by nle-bridge diff) "
                              f"changes the cut of {aid}: {what}. The source is cut to exactly these ranges, frame-accurate, instead of the ranges the "
                              "plan's own trim / silence / take decisions give; an editor's change is confirmed, never applied silently",
                       confidence=1.0, evidence=[r.id for r in allr if aid in (r.data.get("editor_keep") or {})][:-1] + [o.id] + probe_ids_of([aid]), risk="MEDIUM", approval=approval_for(SUBJECT, floor="CONFIRM", keys=APPROVAL_KEYS),
                       provenance="OBSERVED", params={"asset_id": aid, "subject": subject, "keep": [list(r) for r in keep], "plan_keep": [list(r) for r in plan],
                                                      "frames": ((d.get("editor_frames") or {}).get(aid) or []), "rate": d.get("rate"),
                                                      "returned_file": ret.get("file"), "returned_sha256": ret.get("sha256")},
                       alternatives=[{"decision": "keep the plan's cut", "reason": "reject this decision: the next version cuts the source as before", "cost": ""}])
        m = d.get("markers") or {}
        if any(m.get(k) for k in ("added", "removed", "changed")):
            eng.decide(subject=MARKERS_SUBJECT, type="REVIEW",
                       decision=f"review the editor's markers on {subject}: {len(m.get('added') or [])} added, {len(m.get('removed') or [])} removed, "
                                f"{len(m.get('changed') or [])} changed",
                       reason="the editor's marker edits are recorded with the read-back (nle-bridge diff); the plan carries no markers, so they are listed "
                              "for review and change nothing that is rendered", confidence=1.0, evidence=[o.id], risk="LOW", approval="CONFIRM",
                       provenance="OBSERVED", params={"asset_id": subject, "markers": {k: list(m.get(k) or []) for k in ("added", "removed", "changed")}})
