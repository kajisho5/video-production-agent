"""Timeline → ProductionContexts (deterministic segmentation of each timeline into situations).

The boundaries are the events' own start / end points: between two consecutive boundaries the set of active events is
constant, so each elementary interval is one situation. Nothing is snapped, merged by heuristics, or corrected; a
point event (no end) marks a boundary and belongs to the context that starts there. Only events that describe the
media are used (USER_DECISION events are review history, not a situation)."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from ..models import Asset, Event, Inference, Observation, TimeRange
from .model import ContextTrack, ProductionContext

SITUATION_EVENT_TYPES = ("AudioEvent", "SpeechEvent", "SpeakerEvent", "SceneEvent", "SlideEvent", "CameraEvent", "IncidentEvent", "CaptionEvent")


def _asset_of(e: Event) -> Optional[str]:
    return e.asset_id or (e.timeline_id.split(":", 1)[1] if str(e.timeline_id).startswith("asset:") else None)


def build_contexts(events: Iterable[Event], assets: Iterable[Asset], observations: Iterable[Observation] = (), inferences: Iterable[Inference] = ()) -> List[ProductionContext]:
    """One context per elementary interval per timeline, in temporal order. Empty intervals inside the asset duration
    (no event active) are contexts too: "nothing observed here" is a situation the agent must be able to name."""
    obs_ids = {o.id for o in observations}
    inf_by_event: Dict[str, List[str]] = {}
    for inf in inferences:
        for ev in inf.evidence:
            inf_by_event.setdefault(ev, []).append(inf.id)
    durations = {a.id: (a.technical or {}).get("duration") for a in assets}
    by_timeline: Dict[str, List[Event]] = {}
    for e in events:
        if e.event_type not in SITUATION_EVENT_TYPES:
            continue
        by_timeline.setdefault(e.timeline_id, []).append(e)
    out: List[ProductionContext] = []
    for tl_id in sorted(by_timeline):
        evs = by_timeline[tl_id]
        aid = _asset_of(evs[0])
        dur = durations.get(aid) if aid else None
        points = {0.0} if dur is not None else set()
        if dur is not None:
            points.add(float(dur))
        for e in evs:
            r = e.temporal_range()
            points.add(float(r.start)); points.add(float(r.stop))
        pts = sorted(points)
        for a, b in zip(pts, pts[1:]):
            if b - a <= 1e-6:
                continue
            scope = TimeRange(a, b)
            active = [e for e in evs if e.temporal_range().overlaps(scope) and not (e.temporal_range().is_point and abs(e.temporal_range().start - b) < 1e-6)]
            active.sort(key=lambda x: (x.event_type, x.subtype, float(x.range["start"]), x.id))
            tracks: Dict[str, ContextTrack] = {}
            for e in active:
                key = f"{e.event_type}/{e.subtype}"
                t = tracks.setdefault(key, ContextTrack(event_type=e.event_type, subtype=e.subtype))
                t.event_ids.append(e.id)
                if e.source and e.source not in t.sources:
                    t.sources.append(e.source)
                if e.provenance and e.provenance not in t.provenance:
                    t.provenance.append(e.provenance)
            event_ids = sorted(e.id for e in active)
            ctx = ProductionContext(id=ProductionContext.make_id(tl_id, scope.to_dict(), event_ids), timeline_id=tl_id, scope=scope.to_dict(),
                                    asset_ids=sorted({x for x in (_asset_of(e) for e in active) if x} | ({aid} if aid else set())),
                                    tracks=[tracks[k].to_dict() for k in sorted(tracks)], event_ids=event_ids,
                                    observation_ids=sorted({x for e in active for x in e.evidence if x in obs_ids}),
                                    inference_ids=sorted({i for e in active for i in inf_by_event.get(e.id, [])}))
            out.append(ctx)
    return out


def contexts_at(contexts: Iterable[ProductionContext], at: float, timeline_id: Optional[str] = None) -> List[ProductionContext]:
    """The situation at one instant (start-inclusive, end-exclusive)."""
    return [c for c in contexts if (timeline_id is None or c.timeline_id == timeline_id) and c.scope["start"] - 1e-6 <= at < c.scope["end"] - 1e-6]


def contexts_between(contexts: Iterable[ProductionContext], start: float, end: float, timeline_id: Optional[str] = None) -> List[ProductionContext]:
    rng = TimeRange(start, end)
    return [c for c in contexts if (timeline_id is None or c.timeline_id == timeline_id) and c.temporal_range().overlaps(rng)]


def contexts_to_dicts(contexts: Iterable[ProductionContext]) -> List[Dict[str, Any]]:
    return [c.to_dict() for c in contexts]
