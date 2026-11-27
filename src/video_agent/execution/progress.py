"""Render progress (ADR-140, roadmap step 24): which step runs, how far the render is, how long it still takes, and which chunk.

The weight of each compiled operation is its estimate from this machine's speed book (ADR-133: overhead + seconds per media second, a stated
default for a tool never run here). Percent is the share of the summed estimates of the operations done or reused, so it never goes down. The
ETA is the estimate still to run, scaled by how the run has gone so far (seconds taken / seconds estimated for what ran); before anything ran
it is the estimate itself. A snapshot is `video-agent/progress@1`."""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

SCHEMA = "video-agent/progress@1"


class Progress:
    def __init__(self, ops: List[Any], estimates: Dict[str, float], sink: Callable[[Dict[str, Any]], None], job_id: str = "",
                 clock: Callable[[], float] = time.monotonic):
        self.ops = ops
        self.weight = {op.id: max(0.05, float(estimates.get(op.id, 1.0))) for op in ops}
        self.total = sum(self.weight[op.id] for op in ops) or 1.0
        self.sink, self.job_id, self.clock = sink, job_id, clock
        self.t0 = clock()
        self.done_weight = 0.0
        self.ran_weight = 0.0      # estimates of the operations that actually ran (reused ones took no time)
        self.ran_seconds = 0.0
        self.started: Optional[float] = None
        self.done = 0
        self.last: Dict[str, Any] = {}

    def snapshot(self, op: Any, index: int, state: str) -> Dict[str, Any]:
        remaining = max(0.0, self.total - self.done_weight)
        ratio = (self.ran_seconds / self.ran_weight) if self.ran_weight > 0 and self.ran_seconds > 0 else 1.0
        if state == "start" and self.started is not None:
            remaining = max(0.0, remaining - min(self.weight[op.id], (self.clock() - self.started) / ratio))
        snap = {"schema": SCHEMA, "job": self.job_id, "state": state, "percent": round(min(100.0, 100.0 * self.done_weight / self.total), 1),
                "done": self.done, "total": len(self.ops), "elapsed_seconds": round(self.clock() - self.t0, 1),
                "eta_seconds": round(remaining * ratio, 1),
                "step": {"index": index + 1, "op_id": op.id, "tool": op.tool, "skill": op.skill or "", "kind": op.kind}}
        chunk = getattr(op, "chunk", None)
        if chunk is not None:
            snap["chunk"] = int(chunk)
        return snap

    def __call__(self, op: Any, index: int, state: str) -> None:
        now = self.clock()
        if state == "start":
            self.started = now
        elif state in ("ran", "reused"):
            self.done_weight += self.weight[op.id]
            self.done += 1
            if state == "ran" and self.started is not None:
                self.ran_weight += self.weight[op.id]
                self.ran_seconds += now - self.started
            self.started = None
        snap = self.snapshot(op, index, state)
        if self.last and snap["percent"] < self.last.get("percent", 0.0):   # never reported backwards
            snap["percent"] = self.last["percent"]
        self.last = snap
        self.sink(snap)

    def final(self, status: str) -> Dict[str, Any]:
        snap = dict(self.last or {"schema": SCHEMA, "job": self.job_id, "done": self.done, "total": len(self.ops)}, state=status.lower(),
                    elapsed_seconds=round(self.clock() - self.t0, 1))
        if status == "COMPLETED":
            snap.update(percent=100.0, eta_seconds=0.0)
        self.last = snap
        self.sink(snap)
        return snap
