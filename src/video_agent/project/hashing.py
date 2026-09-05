"""Content hashes of a Project IR.

plan_hash: what will be executed (assets, operations, delivery, qa). Unchanged by approvals, so it tells whether a
job's completed operations can be reused. ir_hash: plan_hash + decisions (incl. their status)."""
from __future__ import annotations

from typing import Any, Dict

from ..models import stable_hash

# schema_version is deliberately excluded: the same plan content must hash identically before and after a migration.
PLAN_SECTIONS = ("assets", "video", "audio", "captions", "graphics", "color", "delivery", "qa")   # the finishing sections (ADR-031) and the QC gate (qa.qc) change what is executed


def plan_hash(doc: Dict[str, Any]) -> str:
    return stable_hash({k: doc.get(k) for k in PLAN_SECTIONS})


def ir_hash(doc: Dict[str, Any]) -> str:
    return stable_hash({"plan": plan_hash(doc), "decisions": doc.get("decisions"), "plan_meta": doc.get("plan")})
