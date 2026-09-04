"""Project IR: the versioned contract between planning and deterministic execution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import Decision, new_id, now_iso, stable_hash
from .migrations import CURRENT, migrate

SECTIONS = ["schema_version", "project", "request", "requirements", "source", "assets", "analysis", "intent", "constraints", "policy", "decisions", "plan", "timeline",
            "video", "audio", "captions", "graphics", "color", "delivery", "qa", "execution", "provenance"]


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
            "execution": {"workspace": workspace, "dry_run": False, "allowed_inputs": [], "budgets": {}, "recovery_policy": {"max_attempts": 2}, "approvals": {}},
            "provenance": {"source_hashes": {}, "profile_version": profile.get("version", "0"), "skill_versions": {}, "tool_versions": {}, "created_by": "video-agent", "recovery": [], "runs": []},
        }
        return cls(d)

    # ---- accessors
    @property
    def decisions(self) -> List[Decision]:
        return [Decision.from_dict(d) for d in self.doc["decisions"]]

    def decision(self, id_: str) -> Optional[Dict[str, Any]]:
        return next((d for d in self.doc["decisions"] if d["id"] == id_), None)

    def pending_confirmations(self) -> List[Dict[str, Any]]:
        return [d for d in self.doc["decisions"] if d["approval"] == "CONFIRM" and d["status"] == "PROPOSED"]

    def blocked(self) -> List[Dict[str, Any]]:
        return [d for d in self.doc["decisions"] if d["approval"] == "BLOCK" or d["status"] == "BLOCKED"]

    def approve(self, ids: List[str], who: str = "user") -> List[str]:
        done = []
        for d in self.doc["decisions"]:
            if d["id"] in ids or "all" in ids:
                if d["approval"] == "CONFIRM" and d["status"] == "PROPOSED":
                    d["status"] = "APPROVED"
                    self.doc["execution"].setdefault("approvals", {})[d["id"]] = {"by": who, "at": now_iso()}
                    self.doc["timeline"]["events"].append({"id": new_id("evt"), "type": "USER_DECISION", "timeline_id": "master", "range": {"start": 0.0, "end": None},
                                                            "source": who, "kind": "USER", "confidence": None, "evidence": [d["id"]], "metadata": {"decision": d["id"], "action": "APPROVED"}})
                    done.append(d["id"])
        return done

    def ir_hash(self) -> str:
        core = {k: self.doc[k] for k in ("schema_version", "assets", "decisions", "plan", "video", "audio", "delivery", "qa") if k in self.doc}
        return stable_hash(core)

    def finalize_hash(self) -> str:
        h = self.ir_hash()
        self.doc["provenance"]["ir_hash"] = h
        return h


def save_ir(ir: ProjectIR, path: str) -> str:
    ir.finalize_hash()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(ir.doc, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def load_ir(path: str) -> ProjectIR:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProjectIR(migrate(doc))
