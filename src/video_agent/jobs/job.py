"""Job state machine and workspace layout: <workspace>/jobs/<job_id>/{job.json, ir.json, plans/, ops/, artifacts/, qa/, provenance.json}."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import JOB_STATES, new_id, now_iso

TRANSITIONS = {
    "QUEUED": {"INGESTING", "CANCELLED"}, "INGESTING": {"ANALYZING", "FAILED", "CANCELLED"}, "ANALYZING": {"PLANNING", "FAILED", "CANCELLED"},
    "PLANNING": {"WAITING_FOR_APPROVAL", "EXECUTING", "BLOCKED", "FAILED", "CANCELLED"}, "WAITING_FOR_APPROVAL": {"EXECUTING", "PLANNING", "CANCELLED", "BLOCKED"},
    "EXECUTING": {"QA", "RECOVERY", "FAILED", "BLOCKED", "CANCELLED"}, "RECOVERY": {"EXECUTING", "FAILED", "BLOCKED"}, "QA": {"REVIEW", "DELIVERING", "COMPLETED", "FAILED", "PLANNING"},
    "REVIEW": {"DELIVERING", "PLANNING", "CANCELLED"}, "DELIVERING": {"COMPLETED", "FAILED"}, "COMPLETED": set(), "FAILED": {"QUEUED"}, "BLOCKED": {"PLANNING", "QUEUED"}, "CANCELLED": set(),
}


@dataclass
class Job:
    id: str = field(default_factory=lambda: new_id("job"))
    state: str = "QUEUED"
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    ir_path: str = ""
    workspace: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    completed_ops: Dict[str, str] = field(default_factory=dict)   # idempotency_key -> output
    artifacts: List[Dict[str, Any]] = field(default_factory=list)

    def transition(self, new_state: str, reason: str = "") -> None:
        if new_state not in JOB_STATES:
            raise ValueError(new_state)
        if new_state not in TRANSITIONS.get(self.state, set()):
            raise ValueError(f"illegal job transition {self.state} -> {new_state}")
        self.history.append({"from": self.state, "to": new_state, "at": now_iso(), "reason": reason})
        self.state, self.updated_at = new_state, now_iso()

    @property
    def dir(self) -> Path:
        return Path(self.workspace) / "jobs" / self.id

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in ("id", "state", "created_at", "updated_at", "ir_path", "workspace", "history", "completed_ops", "artifacts")}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Job":
        return cls(**{k: d.get(k, v) for k, v in cls().to_dict().items()})


class JobStore:
    def __init__(self, workspace: str):
        self.workspace = str(Path(workspace).resolve())
        (Path(self.workspace) / "jobs").mkdir(parents=True, exist_ok=True)

    def create(self) -> Job:
        job = Job(workspace=self.workspace)
        for sub in ("plans", "ops", "artifacts", "qa"):
            (job.dir / sub).mkdir(parents=True, exist_ok=True)
        self.save(job)
        return job

    def save(self, job: Job) -> None:
        job.dir.mkdir(parents=True, exist_ok=True)
        (job.dir / "job.json").write_text(json.dumps(job.to_dict(), indent=2) + "\n", encoding="utf-8")

    def load(self, job_id: str) -> Job:
        p = Path(self.workspace) / "jobs" / job_id / "job.json"
        return Job.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def list(self) -> List[Job]:
        out = []
        for p in sorted((Path(self.workspace) / "jobs").glob("*/job.json")):
            out.append(Job.from_dict(json.loads(p.read_text(encoding="utf-8"))))
        return out
