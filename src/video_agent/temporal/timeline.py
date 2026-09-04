"""Unified timeline: events of any type on per-asset or master timelines, with one query function."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..models import Event


@dataclass
class TimelineMap:
    """How an asset timeline maps onto the master timeline: master_t = (asset_t - offset)... mirrored
    from multicam.py: source_t = (master_t - offset) * drift_ratio."""
    id: str
    asset_id: Optional[str]
    offset_seconds: float = 0.0
    drift_ratio: float = 1.0

    def to_master(self, t: float) -> float:
        return t / self.drift_ratio + self.offset_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "asset_id": self.asset_id, "offset_seconds": self.offset_seconds, "drift_ratio": self.drift_ratio}


@dataclass
class Timeline:
    timelines: Dict[str, TimelineMap] = field(default_factory=lambda: {"master": TimelineMap("master", None)})
    events: List[Event] = field(default_factory=list)

    def add_timeline(self, asset_id: str, offset: float = 0.0, drift_ratio: float = 1.0) -> TimelineMap:
        tm = TimelineMap(f"asset:{asset_id}", asset_id, offset, drift_ratio)
        self.timelines[tm.id] = tm
        return tm

    def add(self, event: Event) -> Event:
        if event.timeline_id not in self.timelines:
            raise ValueError(f"unknown timeline {event.timeline_id}")
        self.events.append(event)
        return event

    def query(self, type: Optional[str] = None, timeline_id: Optional[str] = None, between: Optional[Tuple[float, float]] = None,
              kind: Optional[str] = None, source: Optional[str] = None, min_confidence: Optional[float] = None) -> List[Event]:
        out = []
        for e in self.events:
            if type and e.type != type:
                continue
            if timeline_id and e.timeline_id != timeline_id:
                continue
            if kind and e.kind != kind:
                continue
            if source and not e.source.startswith(source):
                continue
            if min_confidence is not None and (e.confidence is None or e.confidence < min_confidence):
                continue
            if between:
                s, en = e.range["start"], e.range.get("end")
                en = s if en is None else en
                if en < between[0] or s > between[1]:
                    continue
            out.append(e)
        return sorted(out, key=lambda e: (e.range["start"], e.type))

    def to_dict(self) -> Dict[str, Any]:
        return {"timelines": {k: v.to_dict() for k, v in self.timelines.items()}, "events": [e.to_dict() for e in self.events]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Timeline":
        t = cls(timelines={}, events=[])
        for k, v in (d.get("timelines") or {}).items():
            t.timelines[k] = TimelineMap(v["id"], v.get("asset_id"), v.get("offset_seconds", 0.0), v.get("drift_ratio", 1.0))
        t.timelines.setdefault("master", TimelineMap("master", None))
        t.events = [Event.from_dict(e) for e in d.get("events") or []]
        return t
