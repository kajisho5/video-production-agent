"""Project IR: the versioned contract between planning and deterministic execution.

Approval / rejection state lives in decisions[].status plus execution.reviews (who / when / why). Revision state
(feedback, version history, which plan version was approved) lives in the `revision` section."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import Decision, new_id, now_iso
from ..agent.production_plan import plan_status
from ..temporal.events import event_id
from .hashing import ir_hash as _ir_hash, plan_hash as _plan_hash
from .migrations import CURRENT, migrate

SECTIONS = ["schema_version", "project", "request", "requirements", "source", "assets", "analysis", "intent", "constraints", "policy", "decisions", "plan", "timeline",
            "video", "audio", "captions", "graphics", "color", "delivery", "qa", "execution", "provenance", "revision"]


class ProjectIR:
    def __init__(self, doc: Dict[str, Any]):
        self.doc = doc

    @classmethod
    def new(cls, name: str, profile: Dict[str, Any], workspace: str) -> "ProjectIR":
        d: Dict[str, Any] = {
            "schema_version": CURRENT,
            "project": {"id": new_id("proj"), "kind": "single", "name": name, "created_at": now_iso(), "profile": profile, "production": None},
            "request": {"raw": "", "received_at": now_iso(), "channel": "cli", "args": {}},
            "requirements": [], "source": {"agent_version": "0", "tool_versions": {}, "generator": "video-agent"},
            "assets": {}, "analysis": {"observations": [], "inferences": [], "strategy": "FULL_ANALYSIS", "budget": {}, "warnings": [], "tool_calls": []},
            "intent": {"primary": "inspect", "secondary": [], "confidence": 0.0, "provenance": "SYSTEM", "reason": ""},
            "constraints": [], "policy": {"effective": {}, "conflicts": []}, "decisions": [],
            "plan": {"version": 1, "steps": [], "summary": []},
            "timeline": {"timelines": {"master": {"id": "master", "asset_id": None, "offset_seconds": 0.0, "drift_ratio": 1.0}}, "events": []},
            "video": {"operations": []}, "audio": {"operations": []}, "captions": {}, "graphics": {}, "color": {},
            "delivery": {"targets": [], "naming": ""}, "qa": {"required": ["video", "audio", "delivery"], "thresholds": {"duration_tolerance_s": 0.5, "loudness_tolerance_lu": 2.0}},
            "execution": {"workspace": workspace, "dry_run": False, "allowed_inputs": [], "budgets": {}, "recovery_policy": {"max_attempts": 2}, "approvals": {}, "reviews": {}, "resume_from": None},
            "provenance": {"source_hashes": {}, "profile_version": profile.get("version", "0"), "skill_versions": {}, "tool_versions": {}, "created_by": "video-agent", "recovery": [], "runs": [], "plan_hash": "", "ir_hash": ""},
            "revision": {"feedback": [], "history": [], "approved_plan_version": None},
        }
        return cls(d)

    # ---- accessors
    @property
    def version(self) -> int:
        return int(self.doc["plan"]["version"])

    @property
    def decisions(self) -> List[Decision]:
        return [Decision.from_dict(d) for d in self.doc["decisions"]]

    def decision(self, id_: str) -> Optional[Dict[str, Any]]:
        return next((d for d in self.doc["decisions"] if d["id"] == id_), None)

    def pending_confirmations(self) -> List[Dict[str, Any]]:
        return [d for d in self.doc["decisions"] if d["approval"] == "CONFIRM" and d["status"] == "PROPOSED"]

    def blocked(self) -> List[Dict[str, Any]]:
        return [d for d in self.doc["decisions"] if d["approval"] == "BLOCK" or d["status"] == "BLOCKED"]

    def rejected(self) -> List[Dict[str, Any]]:
        return [d for d in self.doc["decisions"] if d["status"] == "REJECTED"]

    def rejected_cited(self) -> List[Dict[str, Any]]:
        """Operations / delivery targets that still cite a REJECTED decision (must be revised away before rendering)."""
        rej = {d["id"] for d in self.rejected()}
        out = []
        for op in self.doc["video"]["operations"] + self.doc["audio"]["operations"] + self.doc["delivery"]["targets"]:
            if any(x in rej for x in op.get("decision_ids") or []):
                out.append(op)
        return out

    def needs_reapproval(self) -> bool:
        """A revised plan (version > 1) is renderable only after an explicit approval of that version."""
        return self.version > 1 and self.doc["revision"].get("approved_plan_version") != self.version

    # ---- review actions
    def _record(self, d: Dict[str, Any], action: str, who: str, reason: str) -> None:
        rec = {"action": action, "by": who, "at": now_iso(), "reason": reason, "plan_version": self.version}
        self.doc["execution"].setdefault("reviews", {})[d["id"]] = rec
        if action == "APPROVED":
            self.doc["execution"].setdefault("approvals", {})[d["id"]] = {"by": who, "at": rec["at"]}
        subtype = {"APPROVED": "approved", "REJECTED": "rejected", "REVISED": "revised"}.get(action, "approved")
        rng = {"start": 0.0, "end": None}
        self.doc["timeline"]["events"].append({"id": event_id(None, "USER_DECISION", subtype, rng, who, [d["id"], f"plan_v{self.version}", rec["at"]]),
                                                "type": "USER_DECISION", "event_type": "UserDecisionEvent", "subtype": subtype, "timeline_id": "master", "range": rng,
                                                "asset_id": None, "source": who, "kind": "USER", "provenance": "USER", "confidence": None, "evidence": [d["id"]],
                                                "generator": "review@1.0", "created_at": rec["at"],
                                                "metadata": {"decision": d["id"], "subject": d["subject"], "action": action, "reason": reason, "plan_version": self.version}})

    def approve(self, ids: List[str], who: str = "user", reason: str = "") -> List[str]:
        """Approve CONFIRM decisions. When nothing remains pending or rejected-and-cited, the current plan version becomes approved."""
        done = []
        for d in self.doc["decisions"]:
            if (d["id"] in ids or "all" in ids) and d["approval"] == "CONFIRM" and d["status"] == "PROPOSED":
                d["status"] = "APPROVED"
                self._record(d, "APPROVED", who, reason)
                done.append(d["id"])
        if not self.pending_confirmations() and not self.rejected_cited() and not self.blocked():
            self.doc["revision"]["approved_plan_version"] = self.version
        self.refresh_plan_status()
        return done

    def reject(self, ids: List[str], who: str, reason: str) -> List[str]:
        """Reject decisions (AUTO or CONFIRM). Rejection invalidates any plan-version approval: the plan must be revised."""
        if not reason or not reason.strip():
            raise ValueError("a rejection reason is required")
        done = []
        for d in self.doc["decisions"]:
            if (d["id"] in ids or "all" in ids) and d["status"] in ("PROPOSED", "APPROVED") and d["approval"] != "BLOCK":
                d["status"] = "REJECTED"
                self._record(d, "REJECTED", who, reason.strip())
                done.append(d["id"])
        if done:
            self.doc["revision"]["approved_plan_version"] = None
        self.refresh_plan_status()
        return done

    def refresh_plan_status(self) -> str:
        """The ProductionPlan's status is derived from the reviews / approvals (the source of truth), never set by hand."""
        st = plan_status(self.doc)
        self.doc["plan"]["status"] = st
        decisions = {d["id"]: d for d in self.doc["decisions"]}
        from ..agent.production_plan import step_status
        for step in self.doc["plan"].get("steps") or []:
            step["status"] = step_status(step, decisions)
        return st

    # ---- hashes
    def ir_hash(self) -> str:
        return _ir_hash(self.doc)

    def plan_hash(self) -> str:
        """Hash of what will execute; unchanged by approvals/rejections (used to judge whether a previous job can be resumed)."""
        return _plan_hash(self.doc)

    def finalize_hash(self) -> str:
        self.doc["provenance"]["plan_hash"] = self.plan_hash()
        self.doc["provenance"]["ir_hash"] = self.ir_hash()
        return self.doc["provenance"]["ir_hash"]


def save_ir(ir: ProjectIR, path: str) -> str:
    ir.finalize_hash()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(ir.doc, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def load_ir(path: str) -> ProjectIR:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProjectIR(migrate(doc))


def snapshot_path(ir_path: str, version: int) -> str:
    """Where a plan version is preserved before a revision replaces it: <dir>/<stem>.v<N>.json (never overwritten)."""
    p = Path(ir_path)
    return str(p.with_name(f"{p.stem}.v{version}{p.suffix}"))
