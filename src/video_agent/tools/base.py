"""Tool adapter boundary. The agent never builds shell commands; it hands typed Operations to an adapter."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import Operation, ToolResult


class ToolError(Exception):
    pass


class ToolAdapter:
    name = "abstract"
    version = "0"

    def describe(self) -> Dict[str, Any]:
        raise NotImplementedError

    def supports(self, tool: str) -> bool:
        raise NotImplementedError

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        """Human-readable command preview for dry run. Must not execute media processing."""
        raise NotImplementedError

    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        raise NotImplementedError

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        """Run a measurement (probe / analysis / check) without artifacts. Default builds a measure Operation."""
        return self.run(Operation(tool=tool, args=args, inputs=[], outputs=[], kind="measure"), paths or {}, timeout=timeout)
