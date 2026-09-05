"""Core data models. Plain dataclasses; every object serialises with to_dict()/from_dict().

Design rules (docs/MASTER_SPEC.md §6-§16, docs/ARCHITECTURE_REVIEW.md §7-§10):
- Observation and Inference are distinct types; an Inference always references evidence ids.
- Requirements carry provenance so USER input is never confused with DEFAULT/PROFILE/INFERRED values.
- Decisions carry risk and approval independently of confidence.
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
    id: str = field(default_factory=lambda: new_id("dec"))


@dataclass
class TimeRange(Model):
    start: float
    end: Optional[float] = None   # None = point event


@dataclass
class Event(Model):
    type: str
    timeline_id: str
    range: Dict[str, Any]         # TimeRange.to_dict()
    source: str
    kind: str                     # OBSERVED | INFERRED | USER
    confidence: Optional[float] = None
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("evt"))


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
    qa_status: str = "PENDING"      # PENDING | PASS | WARN | FAIL
    stage: str = "working"          # ARTIFACT_STAGES
    id: str = field(default_factory=lambda: new_id("art"))


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
