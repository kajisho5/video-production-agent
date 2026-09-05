"""Session: a temporal grouping of assets and events (a talk, a recording, a block of a programme). A domain object built
explicitly by the system or the user — never detected automatically here, never a production plan."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..models import Asset, Event, Model, TimeRange, now_iso, stable_hash

SESSION_PROVENANCE = ("SYSTEM", "USER")


@dataclass
class Session(Model):
    id: str
    project_id: str
    name: str
    range: Dict[str, Any]                       # TimeRange.to_dict(), end required
    asset_ids: List[str] = field(default_factory=list)
    event_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: str = "SYSTEM"
    generator: str = ""
    created_at: str = field(default_factory=now_iso)

    def temporal_range(self) -> TimeRange:
        return TimeRange(self.range["start"], self.range.get("end"))

    @staticmethod
    def make_id(project_id: str, name: str, rng: Dict[str, Any], asset_ids: Iterable[str]) -> str:
        return "ses_" + stable_hash([project_id, name, round(float(rng["start"]), 6), None if rng.get("end") is None else round(float(rng["end"]), 6), sorted(asset_ids)])[:16]

    @classmethod
    def new(cls, project_id: str, name: str, rng: TimeRange, asset_ids: Iterable[str], events: Iterable[Event] = (), provenance: str = "SYSTEM",
            generator: str = "session@1.0", metadata: Optional[Dict[str, Any]] = None) -> "Session":
        r = rng.to_dict()
        ids = sorted(set(asset_ids))
        return cls(id=cls.make_id(project_id, name, r, ids), project_id=project_id, name=name, range=r, asset_ids=ids,
                   event_ids=[e.id for e in events], metadata=dict(metadata or {}), provenance=provenance, generator=generator)


def session_for_asset(project_id: str, asset: Asset, events: Iterable[Event]) -> Optional[Session]:
    """The default single-asset session: the whole asset (0..duration) with the events on its timeline. None when the
    asset has no duration (a session needs a real range; nothing is guessed)."""
    dur = (asset.technical or {}).get("duration")
    if dur is None or float(dur) <= 0:
        return None
    rng = TimeRange(0.0, float(dur))
    mine = [e for e in events if (e.asset_id or (e.timeline_id.split(":", 1)[1] if str(e.timeline_id).startswith("asset:") else None)) == asset.id
            and rng.contains(e.temporal_range())]
    return Session.new(project_id, asset.path.replace("\\", "/").split("/")[-1], rng, [asset.id], mine, metadata={"scope": "asset"})


def validate_session(s: Session, project_id: Optional[str], assets: Dict[str, Optional[float]], events: Dict[str, Event]) -> List[str]:
    """Errors for a session. Child events must lie inside the session range (they are never clipped)."""
    errs: List[str] = []
    if not s.id or not str(s.id).startswith("ses_"):
        errs.append(f"invalid session id {s.id!r}")
    if project_id is not None and s.project_id != project_id:
        errs.append(f"session {s.id} belongs to project {s.project_id!r}, not {project_id!r}")
    try:
        rng = s.temporal_range()
        if rng.is_point or rng.duration <= 0:
            errs.append("session range must have end > start")
    except (ValueError, KeyError, TypeError) as ex:
        errs.append(f"invalid session range: {ex}")
        rng = None
    if not s.asset_ids:
        errs.append("session references no asset")
    for a in s.asset_ids:
        if a not in assets:
            errs.append(f"session references unknown asset {a!r}")
        elif rng is not None and not rng.within(assets[a]):
            errs.append(f"session range exceeds asset {a} duration {assets[a]}")
    for eid in s.event_ids:
        e = events.get(eid)
        if e is None:
            errs.append(f"session references unknown event {eid!r}")
            continue
        aid = e.asset_id or (e.timeline_id.split(":", 1)[1] if str(e.timeline_id).startswith("asset:") else None)
        if aid is not None and aid not in s.asset_ids:
            errs.append(f"event {eid} is on asset {aid}, which is not part of the session")
        if rng is not None and not rng.contains(e.temporal_range()):
            errs.append(f"event {eid} ({e.range}) lies outside the session range {s.range}")
    if s.provenance not in SESSION_PROVENANCE:
        errs.append(f"invalid session provenance {s.provenance!r}")
    return errs
