"""Executor: runs compiled Operations through the adapter with finite recovery, idempotency and cancellation.
It never sees the IR's reasoning; it sees Operations and paths only."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import Operation, ToolResult, now_iso
from ..tools.base import ToolAdapter, ToolError
from .recovery import next_attempt


@dataclass
class ExecutionResult:
    results: List[ToolResult] = field(default_factory=list)
    recovery: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "COMPLETED"        # COMPLETED | FAILED | BLOCKED | CANCELLED
    failed_op: Optional[str] = None
    skipped: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "failed_op": self.failed_op, "skipped": self.skipped, "recovery": self.recovery, "results": [r.to_dict() for r in self.results]}


class Executor:
    def __init__(self, adapter: ToolAdapter, max_attempts: int = 2, timeout: Optional[float] = None, completed_keys: Optional[Dict[str, str]] = None):
        self.adapter = adapter
        self.max_attempts = max(1, max_attempts)
        self.timeout = timeout
        self.completed = dict(completed_keys or {})   # idempotency_key -> output path
        self.cancel_event = threading.Event()
        self.log: List[str] = []

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self, ops: List[Operation], paths: Dict[str, str], dry_run: bool = False) -> ExecutionResult:
        res = ExecutionResult()
        for op in ops:
            if self.cancel_event.is_set():
                res.status = "CANCELLED"
                return res
            if not self.adapter.supports(op.tool):
                res.status, res.failed_op = "BLOCKED", op.id
                res.recovery.append({"op": op.id, "class": "TOOL_MISSING", "action": "BLOCK", "reason": f"adapter does not support {op.tool}", "at": now_iso()})
                return res
            if not dry_run and op.idempotency_key and op.idempotency_key in self.completed and os.path.exists(self.completed[op.idempotency_key]):
                res.skipped.append(op.id)
                for out in op.outputs:
                    paths[out] = self.completed[op.idempotency_key]
                self.log.append(f"skip {op.id} (idempotent)")
                continue
            for out in op.outputs:
                Path(paths[out]).parent.mkdir(parents=True, exist_ok=True)
            attempt, timeout, args = 1, self.timeout, dict(op.args)
            while True:
                try:
                    r = self.adapter.run(Operation(tool=op.tool, args=args, inputs=op.inputs, outputs=op.outputs, decision_ids=op.decision_ids, kind=op.kind, id=op.id), paths, timeout=timeout, dry_run=dry_run, attempt=attempt)
                except ToolError as exc:
                    r = ToolResult(op_id=op.id, tool=op.tool, ok=False, exit_code=-1, output=None, data={}, commands=[], stderr_tail=str(exc), seconds=0.0, attempt=attempt, dry_run=dry_run)
                res.results.append(r)
                if r.ok:
                    if not dry_run and op.idempotency_key and r.output:
                        self.completed[op.idempotency_key] = r.output
                    break
                plan = next_attempt(r, attempt, self.max_attempts, timeout)
                res.recovery.append({"op": op.id, "attempt": attempt, "class": plan["class"], "action": plan["action"], "reason": plan["reason"], "at": now_iso(), "stderr": r.stderr_tail[-400:]})
                if plan["action"] != "RETRY":
                    res.status = "BLOCKED" if plan["class"] in ("TOOL_MISSING", "INPUT_MISSING", "DISK_FULL") else "FAILED"
                    res.failed_op = op.id
                    return res
                args.update(plan["args_patch"])
                timeout = plan["timeout"]
                attempt += 1
        return res
