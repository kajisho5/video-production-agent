"""ProductionContext: the agent's intermediate representation of the production situation on a timeline over a time
range (ADR-026).

    ProductionContext ≠ Observation (a measured fact)   ≠ Event (one temporal occurrence)   ≠ Inference (an interpretation)
    ≠ Decision (a production choice)   ≠ Session (a grouping a person or the system declares)

A context answers "what is observed here, at the same time?": for one timeline and one scope it references the events
that are active in the scope (grouped by domain type and subtype), the observations those events rest on, the assets
they belong to, and the inferences that already cite those events. It is derived deterministically from the timeline
(provenance DERIVED), copies no timestamps, changes no event, resolves no overlap, and carries no decision, tool,
command or path. Contexts are reference-centred: everything they name exists in the analysis / IR they were built from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..models import Model, TimeRange, now_iso, stable_hash

CONTEXT_BUILDER_ID = "context_builder@1.0"
CONTEXT_PROVENANCE = "DERIVED"


@dataclass
class ContextTrack(Model):
    """The events of one domain type / subtype active in the scope (references only)."""
    event_type: str                                   # SpeechEvent | AudioEvent | SceneEvent | ...
    subtype: str                                      # speech | silence | active | loudness | visual_change | ...
    event_ids: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)  # tool sources of those events ("<tool>@<version>")
    provenance: List[str] = field(default_factory=list)   # OBSERVED / DERIVED / INFERRED / AI_GENERATED / USER as recorded on the events


@dataclass
class ProductionContext(Model):
    id: str
    timeline_id: str                                  # "asset:<id>" or "master"
    scope: Dict[str, Any]                             # TimeRange.to_dict(): the situation holds for this whole range
    asset_ids: List[str] = field(default_factory=list)
    tracks: List[Dict[str, Any]] = field(default_factory=list)      # ContextTrack.to_dict() in deterministic order
    event_ids: List[str] = field(default_factory=list)              # every active event (union of the tracks)
    observation_ids: List[str] = field(default_factory=list)        # observations the active events rest on
    inference_ids: List[str] = field(default_factory=list)          # existing inferences that cite any active event
    provenance: str = CONTEXT_PROVENANCE
    generator: str = CONTEXT_BUILDER_ID
    created_at: str = field(default_factory=now_iso)

    def temporal_range(self) -> TimeRange:
        return TimeRange(self.scope["start"], self.scope.get("end"))

    @property
    def signature(self) -> str:
        """The situation as a type/subtype set (what kinds of things are happening), independent of the scope."""
        return "+".join(sorted(f"{t['event_type']}/{t['subtype']}" for t in self.tracks)) or "nothing"

    @staticmethod
    def make_id(timeline_id: str, scope: Dict[str, Any], event_ids: Iterable[str]) -> str:
        """Deterministic: same timeline + scope + active events → same id (event ids are themselves deterministic)."""
        return "ctx_" + stable_hash([timeline_id, round(float(scope["start"]), 6), None if scope.get("end") is None else round(float(scope["end"]), 6), sorted(event_ids)])[:16]


def validate_context(c: ProductionContext, events: Dict[str, Any], assets: Dict[str, Optional[float]], observations: Iterable[str], inferences: Iterable[str]) -> List[str]:
    """Errors for a context: every reference must exist, the scope must be a real range inside the asset, every active
    event must overlap the scope, and the id must be the deterministic one (a context is never edited by hand)."""
    errs: List[str] = []
    if not str(c.id).startswith("ctx_"):
        errs.append(f"invalid context id {c.id!r}")
    try:
        rng = c.temporal_range()
        if rng.is_point or rng.duration <= 0:
            errs.append(f"context {c.id}: scope must have end > start")
    except (ValueError, KeyError, TypeError) as ex:
        errs.append(f"context {c.id}: invalid scope: {ex}")
        rng = None
    if c.provenance != CONTEXT_PROVENANCE:
        errs.append(f"context {c.id}: provenance must be {CONTEXT_PROVENANCE}, got {c.provenance!r}")
    for a in c.asset_ids:
        if a not in assets:
            errs.append(f"context {c.id}: unknown asset {a!r}")
        elif rng is not None and not rng.within(assets[a]):
            errs.append(f"context {c.id}: scope exceeds asset {a} duration {assets[a]}")
    obs, infs = set(observations), set(inferences)
    for eid in c.event_ids:
        e = events.get(eid)
        if e is None:
            errs.append(f"context {c.id}: unknown event {eid!r}")
            continue
        if rng is not None:
            er = TimeRange(e["range"]["start"], e["range"].get("end")) if isinstance(e, dict) else e.temporal_range()
            if not er.overlaps(rng):
                errs.append(f"context {c.id}: event {eid} does not overlap the scope")
    if sorted({i for t in c.tracks for i in t.get("event_ids") or []}) != sorted(c.event_ids):
        errs.append(f"context {c.id}: tracks and event_ids disagree")
    for o in c.observation_ids:
        if o not in obs:
            errs.append(f"context {c.id}: unknown observation {o!r}")
    for i in c.inference_ids:
        if i not in infs:
            errs.append(f"context {c.id}: unknown inference {i!r}")
    if c.id != ProductionContext.make_id(c.timeline_id, c.scope, c.event_ids):
        errs.append(f"context {c.id}: id does not match its content")
    return errs
