"""Provenance / audit: who, what, why, when, input, output, tool, version, decision, result, qa — linking
decisions to executed operations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..models import Operation, ToolResult, now_iso


def build_provenance(ir_doc: Dict[str, Any], ops: List[Operation], results: List[ToolResult], paths: Dict[str, str], recovery: List[Dict[str, Any]], qa: Dict[str, Any], who: str = "video-agent") -> Dict[str, Any]:
    by_op: Dict[str, List[ToolResult]] = {}
    for r in results:
        by_op.setdefault(r.op_id, []).append(r)
    entries = []
    for op in ops:
        rs = by_op.get(op.id, [])
        last = rs[-1] if rs else None
        entries.append({
            "who": who, "when": now_iso(), "what": op.tool, "skill": op.skill, "why": [ir_doc_decision_reason(ir_doc, d) for d in op.decision_ids],
            "decision": op.decision_ids, "input": [paths.get(i, i) for i in op.inputs], "output": [paths.get(o, o) for o in op.outputs],
            "tool": op.tool, "tool_version": _version_of(ir_doc["source"]["tool_versions"], op.tool), "skill_package": op.tool.split("/", 1)[0], "args": op.args, "idempotency_key": op.idempotency_key,
            "result": None if last is None else {"ok": last.ok, "exit_code": last.exit_code, "attempts": len(rs), "seconds": sum(r.seconds for r in rs), "commands": last.commands, "dry_run": last.dry_run},
        })
    return {"ir_hash": ir_doc["provenance"].get("ir_hash"), "source_hashes": ir_doc["provenance"]["source_hashes"], "profile_version": ir_doc["provenance"]["profile_version"],
            "skill_versions": ir_doc["provenance"]["skill_versions"], "tool_versions": ir_doc["source"]["tool_versions"], "operations": entries, "recovery": recovery, "qa": qa,
            "plan_hash": ir_doc["provenance"].get("plan_hash"), "ai_provider": ir_doc["provenance"].get("ai_provider"), "ai_calls": list(ir_doc["provenance"].get("ai_calls") or [])}


def _version_of(versions: Dict[str, str], tool: str) -> str:
    return str(versions.get(tool.split("/", 1)[0], ""))


def ir_doc_decision_reason(ir_doc: Dict[str, Any], decision_id: str) -> str:
    d = next((x for x in ir_doc["decisions"] if x["id"] == decision_id), None)
    return f"{d['subject']}: {d['decision']} — {d['reason']}" if d else decision_id


def write_audit(path: str, record: Dict[str, Any]) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path
