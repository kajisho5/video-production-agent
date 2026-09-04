"""Schema migrations: {from_version: (to_version, fn)}. Phase 1 knows only 1.0."""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

CURRENT = "1.0"
MIGRATIONS: Dict[str, Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = {}


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
