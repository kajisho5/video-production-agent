"""ProductionReceipt (docs/SPEC.md §6, ADR-040): the once-per-completed-execution roll-up
answering "what happened, why, with what tools, did it pass verification" for one Plan run.
Composed entirely from data `build_provenance()` / `Service._register_artifacts()` already
produce -- one more Artifact type (`production_receipt`), never a new source of truth."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import now_iso, stable_hash


def obligations_of(skill_observations: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """What the published video has to carry because of the engines that made it (ADR-082): every `obligations.credit` line a Skill
    reported (dubbing-skill DUB: VOICEVOX / Kokoro / Style-Bert-VITS2 voice credits, ADR-012 of that Skill), each once, in the order
    reported, with where it came from, and the watermarks embedded in the output. Empty lists when nothing obliges anything."""
    credit: List[str] = []
    sources: List[Dict[str, Any]] = []
    watermarks: List[Dict[str, Any]] = []
    for o in skill_observations or []:
        ob = ((o.get("data") or {}).get("obligations")) if isinstance(o, dict) else None
        if not isinstance(ob, dict):
            continue
        new = [str(c) for c in ob.get("credit") or [] if str(c) not in credit]
        credit += new
        if ob.get("credit") or ob.get("watermark"):
            sources.append({"kind": o.get("kind"), "source": o.get("source"), "credit": [str(c) for c in ob.get("credit") or []],
                            "credit_required": ob.get("credit_required"), "watermark": ob.get("watermark")})
        if ob.get("watermark"):
            watermarks.append(dict(ob["watermark"]) if isinstance(ob["watermark"], dict) else {"name": str(ob["watermark"])})
    return {"credit": credit, "watermarks": watermarks, "sources": sources}


def build_receipt(ir_doc: Dict[str, Any], result_status: str, failed_op: Optional[str],
                  artifacts: List[Dict[str, Any]], qa_items: Optional[List[Dict[str, Any]]],
                  skill_observations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """`artifacts`: the registered Artifact dicts for this job (empty when execution never reached registration).
    `qa_items`: this job's QAReport.items as dicts, None when execution itself did not complete (nothing measured).
    `skill_observations`: provenance's lifted Skill observations; their `obligations` (credits / watermarks, ADR-082) become `obligations`.
    Emitted once execution reaches ANY terminal Executor status (COMPLETED, FAILED, BLOCKED, CANCELLED) -- "the
    Plan finished running," per PROVENANCE.md §4, not that it fully passed. Never emitted for a Plan that was
    rejected/blocked/pending approval before execution started: nothing ran, so there is nothing to record."""
    warnings: List[str] = []
    failures: List[str] = []
    if result_status != "COMPLETED":
        failures.append(f"execution {result_status}" + (f": op {failed_op}" if failed_op else ""))
    for i in qa_items or []:
        line = f"{i['layer']}/{i['name']} ({i.get('artifact', '')}): {i['observed']!r} vs {i['expected']!r}"
        if i["status"] == "FAIL":
            failures.append(line)
        elif i["status"] == "WARN":
            warnings.append(line)
    warnings.extend(str(w) for w in (ir_doc.get("analysis") or {}).get("warnings") or [])
    body = {
        "project_id": ir_doc["project"]["id"],
        "plan_id": ir_doc["plan"].get("id", ""),
        "plan_hash": ir_doc["provenance"].get("plan_hash"),
        "ir_hash": ir_doc["provenance"].get("ir_hash"),
        "input_artifact_ids": sorted(ir_doc.get("assets") or {}),
        "output_artifact_ids": [a["id"] for a in artifacts],
        "skill_versions": ir_doc["provenance"]["skill_versions"],
        "tool_versions": ir_doc["source"]["tool_versions"],
        "decisions": [d["id"] for d in ir_doc.get("decisions") or []],
        # qc-skill's own reports are not registered as independent Artifacts anywhere in this codebase today
        # (ADR-040): their PASS/WARN/FAIL verdict is already folded into `warnings`/`failures` above via the QA
        # items that admit them (qa/checks.py's qc_items()), so this stays honestly empty rather than naming an
        # artifact id nothing actually owns.
        "qc_report_ids": [],
        "warnings": warnings,
        "failures": failures,
    }
    ob = obligations_of(skill_observations)
    if ob["credit"] or ob["watermarks"]:   # ADR-082: written only when an engine obliges something, so other receipts keep their shape
        body["obligations"] = ob
    body["id"] = "receipt_" + stable_hash(body)[:16]
    body["created_at"] = now_iso()
    return body
