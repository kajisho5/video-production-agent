"""Schema migrations: {from_version: (to_version, fn)}. Phase 1 knows only 1.0."""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

CURRENT = "1.1"


def _v10_to_v11(doc: Dict[str, Any]) -> Dict[str, Any]:
    """1.1 adds provenance.plan_hash (execution content, independent of approvals) and execution.resume_from."""
    from .hashing import plan_hash
    doc.setdefault("provenance", {})
    doc["schema_version"] = "1.1"
    doc["provenance"]["plan_hash"] = plan_hash(doc)
    doc.setdefault("execution", {}).setdefault("resume_from", None)
    return doc


MIGRATIONS: Dict[str, Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = {"1.0": ("1.1", _v10_to_v11)}


def migrate(doc: Dict[str, Any]) -> Dict[str, Any]:
    v = str(doc.get("schema_version", ""))
    hops = 0
    while v != CURRENT:
        if v not in MIGRATIONS:
            raise ValueError(f"unsupported schema_version {v!r} (current {CURRENT})")
        v, fn = MIGRATIONS[v]
        doc = fn(doc)
        doc["schema_version"] = v
        hops += 1
        if hops > 20:
            raise ValueError("migration loop")
    return doc
