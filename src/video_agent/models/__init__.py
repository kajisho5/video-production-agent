"""Core data models. Plain dataclasses; every object serialises with to_dict()/from_dict().

Design rules (docs/MASTER_SPEC.md §6-§16, docs/ARCHITECTURE_REVIEW.md §7-§10):
- Observation and Inference are distinct types; an Inference always references evidence ids.
- Requirements carry provenance so USER input is never confused with DEFAULT/PROFILE/INFERRED values.
- Decisions carry risk and approval independently of confidence, a type from the engine vocabulary and the basis they were resolved from.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PROVENANCE = ("USER", "SYSTEM", "PROFILE", "DEFAULT", "OBSERVED", "INFERRED", "AI_GENERATED")
APPROVALS = ("AUTO", "CONFIRM", "BLOCK")
RISKS = ("LOW", "MEDIUM", "HIGH")
CAPABILITY_STATUS = ("AVAILABLE", "MISSING", "DEGRADED", "UNKNOWN")
JOB_STATES = ("QUEUED", "INGESTING", "ANALYZING", "PLANNING", "WAITING_FOR_APPROVAL", "EXECUTING", "QA",
              "RECOVERY", "REVIEW", "DELIVERING", "COMPLETED", "FAILED", "BLOCKED", "CANCELLED")
ASSET_TYPES = ("CAMERA", "AUDIO", "SLIDE", "SCREEN_CAPTURE", "MUSIC", "BGM", "LOGO", "IMAGE", "GRAPHIC", "CAPTION", "UNKNOWN")
ARTIFACT_TYPES = ("MASTER", "WEB", "YOUTUBE", "SOCIAL", "ARCHIVE", "CAPTIONS", "THUMBNAIL", "REPORT", "INTERMEDIATE")
ARTIFACT_STAGES = ("working", "candidate", "approved", "final", "archive")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class Model:
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in d.items() if k in names})  # type: ignore[call-arg]


@dataclass
class Request(Model):
    raw: str
    received_at: str = field(default_factory=now_iso)
    channel: str = "cli"
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Requirement(Model):
    key: str
    value: Any
    provenance: str            # PROVENANCE
    source: str = ""           # profile name, flag name, rule id ...
    id: str = field(default_factory=lambda: new_id("req"))


@dataclass
class Intent(Model):
    primary: str
    secondary: List[str] = field(default_factory=list)
    confidence: float = 1.0
    provenance: str = "SYSTEM"
    reason: str = ""


@dataclass
class Observation(Model):
    """Evidence measured by a tool. Never overwritten by inference."""
    kind: str                  # probe | silence | loudness | scenes | sync | file
    asset_id: str
    source: str                # "ffmpeg-skill/probe@0.8.4"
    data: Dict[str, Any]
    id: str = field(default_factory=lambda: new_id("obs"))
    observed_at: str = field(default_factory=now_iso)
    analysis_id: str = ""      # AnalysisRequest that produced it (analysis provenance, ADR-019)
    analyzer: str = ""         # "<analyzer id>@<version>"
    cache_key: str = ""        # deterministic measurement identity (not the observation id)
    provenance: str = "OBSERVED"   # always OBSERVED: AI output never becomes an observation
    # ---- external Skill provenance (ADR-023): identity of the measurement as the Skill reported it, never simplified
    skill: str = ""                # producing Skill package id, e.g. "media-analysis"
    skill_version: str = ""
    tool: str = ""                 # tool id, e.g. "media-analysis/silence"
    external_id: str = ""          # the Skill's own observation id
    fingerprint: str = ""          # asset content fingerprint as the Skill measured it
    parameters: Dict[str, Any] = field(default_factory=dict)   # effective parameters recorded by the Skill
    cache: Dict[str, Any] = field(default_factory=dict)        # the Skill's cache status for this measurement (owned by the Skill)


@dataclass
class Inference(Model):
    """An interpretation of observations. Must cite evidence."""
    kind: str                  # leading_silence_unwanted | trailing_silence_unwanted | loudness_off_target ...
    asset_id: str
    statement: str
    confidence: float
    evidence: List[str]        # observation ids / event ids
    data: Dict[str, Any] = field(default_factory=dict)
    provenance: str = "INFERRED"
    id: str = field(default_factory=lambda: new_id("inf"))


@dataclass
class Alternative(Model):
    decision: str
    reason: str
    cost: str = ""


@dataclass
class Decision(Model):
    subject: str
    decision: str
    reason: str
    confidence: float
    evidence: List[str]
    risk: str                  # RISKS
    approval: str              # APPROVALS
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    provenance: str = "SYSTEM"
    status: str = "PROPOSED"   # PROPOSED | APPROVED | REJECTED | BLOCKED
    params: Dict[str, Any] = field(default_factory=dict)
    type: str = ""             # DECISION_TYPES (agent/decision_engine.py): KEEP | REMOVE | TRANSFORM | DELIVER | SKIP | REVIEW | BLOCK
    basis: Dict[str, Any] = field(default_factory=dict)   # how it was resolved: settings (policy / preference / constraint + provenance), approval, intent, requirements
    id: str = field(default_factory=lambda: new_id("dec"))


TIME_EPS = 1e-6   # seconds: float tolerance for temporal comparisons (canonical timebase is seconds as float)
# a scope end meant to cover an asset's entire duration is rounded to 3 decimals when built (e.g. planner.py's
# round(dur, 3)), which can round *up* by as much as 5e-4 s versus the raw, unrounded probe duration it is then
# checked against in within() — TIME_EPS is far too tight to absorb that and would reject a scope that is, in
# fact, the asset's own exact length. Matches project/validator.py's own independently-chosen tolerance for the
# same class of "does this range exceed the source duration" check.
DURATION_EPS = 0.01


@dataclass
class TimePoint(Model):
    """A position on an asset / master timeline, in seconds."""
    seconds: float

    def __post_init__(self) -> None:
        try:
            self.seconds = float(self.seconds)
        except (TypeError, ValueError):
            raise ValueError(f"time point must be a number of seconds, got {self.seconds!r}")
        if self.seconds != self.seconds or self.seconds < -TIME_EPS:
            raise ValueError(f"time point must be >= 0 s, got {self.seconds}")
        self.seconds = max(0.0, self.seconds)


@dataclass
class TimeRange(Model):
    """Temporal range in seconds. end=None is a point event (start only). Validated on construction: start >= 0, end >= start.
    Relations (overlaps / contains / precedes / adjacent) use TIME_EPS so float noise never flips a verdict."""
    start: float
    end: Optional[float] = None   # None = point event

    def __post_init__(self) -> None:
        self.start = TimePoint(self.start).seconds
        if self.end is not None:
            self.end = TimePoint(self.end).seconds
            if self.end < self.start - TIME_EPS:
                raise ValueError(f"temporal range end {self.end} < start {self.start}")
            self.end = max(self.end, self.start)

    @property
    def is_point(self) -> bool:
        return self.end is None

    @property
    def stop(self) -> float:
        return self.start if self.end is None else self.end

    @property
    def duration(self) -> float:
        return self.stop - self.start

    def overlaps(self, other: "TimeRange") -> bool:
        return self.start < other.stop - TIME_EPS and other.start < self.stop - TIME_EPS if not (self.is_point or other.is_point) \
            else (other.start - TIME_EPS <= self.start <= other.stop + TIME_EPS) if self.is_point else (self.start - TIME_EPS <= other.start <= self.stop + TIME_EPS)

    def contains(self, other: "TimeRange") -> bool:
        return self.start - TIME_EPS <= other.start and other.stop <= self.stop + TIME_EPS

    def precedes(self, other: "TimeRange") -> bool:
        return self.stop <= other.start + TIME_EPS

    def adjacent(self, other: "TimeRange", tolerance: float = TIME_EPS) -> bool:
        return abs(self.stop - other.start) <= tolerance or abs(other.stop - self.start) <= tolerance

    def within(self, duration: Optional[float]) -> bool:
        return duration is None or self.stop <= float(duration) + DURATION_EPS


TemporalRange = TimeRange
EVENT_PROVENANCE = ("OBSERVED", "DERIVED", "INFERRED", "AI_GENERATED", "USER")


@dataclass
class Event(Model):
    """A temporal domain occurrence on an asset (or the master) timeline. `type` is the canonical event code
    (e.g. AUDIO_SILENCE); `event_type` / `subtype` are its domain classification (AudioEvent / silence).
    Not a measurement (Observation), not an interpretation (Inference), not a production choice (Decision)."""
    type: str
    timeline_id: str
    range: Dict[str, Any]         # TimeRange.to_dict()
    source: str
    kind: str                     # OBSERVED | INFERRED | USER (schema-level class)
    confidence: Optional[float] = None
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("evt"))
    event_type: str = ""          # AudioEvent | SpeechEvent | ... (filled from `type` by temporal.events when omitted)
    subtype: str = ""
    asset_id: Optional[str] = None
    provenance: str = ""          # EVENT_PROVENANCE (filled from `kind` when omitted)
    session_id: Optional[str] = None
    generator: str = ""           # "<transformation>@<version>" for DERIVED events, tool for OBSERVED, actor for USER
    created_at: str = field(default_factory=now_iso)

    def temporal_range(self) -> TimeRange:
        return TimeRange(self.range["start"], self.range.get("end"))


@dataclass
class Asset(Model):
    path: str
    type: str = "UNKNOWN"          # ASSET_TYPES
    hash: Optional[str] = None
    technical: Dict[str, Any] = field(default_factory=dict)   # probe subset
    analysis: Dict[str, Any] = field(default_factory=dict)
    classification: Dict[str, Any] = field(default_factory=dict)  # {type, confidence, evidence}
    provenance: str = "USER"
    relationships: List[Dict[str, Any]] = field(default_factory=list)  # {kind, target, metadata}
    status: str = "ingested"
    id: str = field(default_factory=lambda: new_id("asset"))


@dataclass
class Incident(Model):
    type: str
    severity: str                  # LOW | MEDIUM | HIGH
    start: Optional[float] = None
    end: Optional[float] = None
    evidence: List[str] = field(default_factory=list)
    possible_cause: str = ""
    recommended_action: str = ""
    status: str = "OPEN"
    id: str = field(default_factory=lambda: new_id("inc"))


@dataclass
class Artifact(Model):
    path: str
    type: str                      # ARTIFACT_TYPES
    hash: Optional[str] = None
    source: List[str] = field(default_factory=list)   # asset / artifact ids
    generation: int = 0
    tool: str = ""
    tool_version: str = ""
    created_at: str = field(default_factory=now_iso)
    qa_status: str = "PENDING"      # PENDING | PASS | WARN | FAIL | UNKNOWN
    stage: str = "working"          # ARTIFACT_STAGES: working → candidate → (approved) → final → archive
    id: str = field(default_factory=lambda: new_id("art"))
    # ---- production identity and lifecycle (ADR-022); defaults keep older job records loadable
    logical_name: str = ""          # ProductionPlan output name, e.g. "<asset>_delivery_youtube"
    project_id: str = ""
    plan_id: str = ""
    plan_version: int = 1
    job_id: str = ""                # producing job; `jobs` lists every job that produced or reused it
    jobs: List[str] = field(default_factory=list)
    format: str = ""                # preset / container name
    name: str = ""                  # safe delivery file name (naming template), not the storage path
    size: Optional[int] = None
    media: Dict[str, Any] = field(default_factory=dict)      # probe subset recorded by QA
    operations: List[str] = field(default_factory=list)      # IR operation ids that produced it
    step_id: Optional[str] = None                            # ProductionStep
    decision_ids: List[str] = field(default_factory=list)
    qa: Dict[str, Any] = field(default_factory=dict)         # {pass, warn, fail, items}
    delivery_status: str = "NOT_READY"                       # NOT_READY | READY | DELIVERED | ARCHIVED (view of stage)
    delivery_history: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict) # {ir_path, plan_hash, ir_hash, provenance_path}


@dataclass
class Operation(Model):
    """Output of the compiler: one deterministic tool invocation."""
    tool: str                       # "ffmpeg-skill/cut" (selected via SkillRegistry.select_tool, never hard-coded downstream)
    args: Dict[str, Any]            # typed adapter args (never argv, never ffmpeg flags)
    inputs: List[str]               # artifact/asset ids
    outputs: List[str]              # artifact ids
    decision_ids: List[str] = field(default_factory=list)
    kind: str = "transform"         # transform | measure | qa
    idempotency_key: str = ""
    skill: str = ""                 # the skill this operation realises (for provenance / listing)
    id: str = field(default_factory=lambda: new_id("op"))


@dataclass
class ToolResult(Model):
    op_id: str
    tool: str
    ok: bool
    exit_code: int
    output: Optional[str]
    data: Dict[str, Any]
    commands: List[str]
    stderr_tail: str
    seconds: float
    attempt: int = 1
    dry_run: bool = False
