"""Executor: runs compiled Operations through the adapter with finite recovery, idempotent skipping and
cancellation. It never sees the IR's reasoning; it sees Operations and paths only.

Idempotent skip rule: an operation is skipped only when a completed record exists for its (chained) idempotency
key AND the recorded output file still exists with the recorded size and mtime. Anything else re-runs."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..models import Operation, ToolResult, now_iso
from ..tools.base import ToolAdapter, ToolError
from .recovery import describe_failure, next_attempt


@dataclass
class ExecutionResult:
    results: List[ToolResult] = field(default_factory=list)
    recovery: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "COMPLETED"        # COMPLETED | FAILED | BLOCKED | CANCELLED
    failed_op: Optional[str] = None
    skipped: List[str] = field(default_factory=list)
    reused: Dict[str, str] = field(default_factory=dict)   # op id -> reused output path

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "failed_op": self.failed_op, "skipped": self.skipped, "reused": self.reused, "recovery": self.recovery, "results": [r.to_dict() for r in self.results]}


def output_record(path: str) -> Dict[str, Any]:
    st = os.stat(path)
    return {"output": path, "size": st.st_size, "mtime": st.st_mtime}


def record_matches(rec: Any) -> bool:
    """True when the recorded output still exists unchanged. Legacy string records (no size/mtime) never match."""
    if not isinstance(rec, dict) or not rec.get("output"):
        return False
    try:
        st = os.stat(rec["output"])
    except OSError:
        return False
    return st.st_size == rec.get("size") and abs(st.st_mtime - float(rec.get("mtime", -1))) < 1e-6


class Executor:
    def __init__(self, adapter: ToolAdapter, max_attempts: int = 2, timeout: Optional[float] = None, completed_keys: Optional[Dict[str, Any]] = None):
        self.adapter = adapter
        self.max_attempts = max(1, max_attempts)
        self.timeout = timeout
        self.completed: Dict[str, Any] = dict(completed_keys or {})   # idempotency_key -> {output, size, mtime}
        self.cancel_event = threading.Event()
        self.log: List[str] = []
        # ADR-135: called after each operation that completed ("ran") or was reused ("reused"), before the next one starts -- the job's
        # checkpoint, so a render that is killed keeps the record of every operation done before it
        self.checkpoint: Optional[Callable[[Operation, str], None]] = None
        # ADR-140: called with (op, its 0-based index, state) -- "start" before an operation runs, then "ran" / "reused" / "failed" -- for the progress
        self.progress: Optional[Callable[[Operation, int, str], None]] = None

    def _progress(self, op: Operation, index: int, state: str) -> None:
        if self.progress is not None:
            try:
                self.progress(op, index, state)
            except Exception:   # noqa: BLE001 -- reporting progress never stops a render
                pass

    def _checkpoint(self, op: Operation, state: str) -> None:
        if self.checkpoint is not None:
            self.checkpoint(op, state)

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self, ops: List[Operation], paths: Dict[str, str], dry_run: bool = False) -> ExecutionResult:
        res = ExecutionResult()
        for index, op in enumerate(ops):
            if self.cancel_event.is_set():
                res.status = "CANCELLED"
                return res
            if not self.adapter.supports(op.tool):
                res.status, res.failed_op = "BLOCKED", op.id
                res.recovery.append({"op": op.id, "tool": op.tool, "class": "TOOL_MISSING", "action": "BLOCK", "reason": f"adapter does not support {op.tool}", "at": now_iso(),
                                     "error_class": "environment", "recovery_class": "TOOL_MISSING", "code": None, "retry_helps": False,
                                     "fix": "this machine: install or update the tool, then render again"})
                return res
            # a record names one output file; an operation with several (ADR-073: voice-enhance bleed-reduce, one track per input) is re-run, not half-reused
            reusable = bool(op.idempotency_key) and not dry_run and len(op.outputs) == 1
            rec = self.completed.get(op.idempotency_key) if reusable else None
            if rec is not None and record_matches(rec):
                res.skipped.append(op.id)
                res.reused[op.id] = rec["output"]
                for out in op.outputs:
                    paths[out] = rec["output"]
                self.log.append(f"skip {op.id} (idempotent, reusing {rec['output']})")
                self._checkpoint(op, "reused")
                self._progress(op, index, "reused")
                continue
            if rec is not None:
                self.log.append(f"stale record for {op.id}: output missing or changed, re-running")
                self.completed.pop(op.idempotency_key, None)
            for out in op.outputs:
                Path(paths[out]).parent.mkdir(parents=True, exist_ok=True)
            attempt, timeout, args = 1, self.timeout, dict(op.args)
            self._progress(op, index, "start")
            while True:
                try:
                    r = self.adapter.run(Operation(tool=op.tool, args=args, inputs=op.inputs, outputs=op.outputs, decision_ids=op.decision_ids, kind=op.kind, id=op.id), paths, timeout=timeout, dry_run=dry_run, attempt=attempt)
                except KeyboardInterrupt:
                    # the adapter has already killed the tool's process group; leave intermediates, record the interruption
                    res.status, res.failed_op = "CANCELLED", op.id
                    res.recovery.append({"op": op.id, "tool": op.tool, "attempt": attempt, "class": "INTERRUPTED", "action": "CANCEL", "reason": "interrupted by user (SIGINT)",
                                         "at": now_iso(), "stderr": "", "error_class": "transient", "recovery_class": "INTERRUPTED", "code": None, "retry_helps": True,
                                         "fix": "render again (`render --resume last` reuses what finished)"})
                    return res
                except ToolError as exc:
                    r = ToolResult(op_id=op.id, tool=op.tool, ok=False, exit_code=-1, output=None, data={}, commands=[], stderr_tail=str(exc), seconds=0.0, attempt=attempt, dry_run=dry_run)
                res.results.append(r)
                if r.ok:
                    if reusable and r.output and os.path.exists(r.output):
                        self.completed[op.idempotency_key] = output_record(r.output)
                    self._checkpoint(op, "ran")
                    self._progress(op, index, "ran")
                    break
                plan = next_attempt(r, attempt, self.max_attempts, timeout)
                res.recovery.append({"op": op.id, "tool": op.tool, "attempt": attempt, "class": plan["class"], "action": plan["action"], "reason": plan["reason"], "at": now_iso(),
                                     "stderr": r.stderr_tail[-400:], **describe_failure(r, plan["class"])})   # ADR-140: error_class, code, retry_helps, fix
                if plan["action"] != "RETRY":
                    self._progress(op, index, "failed")
                    res.status = "BLOCKED" if plan["class"] in ("TOOL_MISSING", "INPUT_MISSING", "DISK_FULL", "ACTION_REQUIRED") else "FAILED"
                    res.failed_op = op.id
                    return res
                args.update(plan["args_patch"])
                timeout = plan["timeout"]
                attempt += 1
        return res
