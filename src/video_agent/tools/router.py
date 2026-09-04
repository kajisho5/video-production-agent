"""ToolRouter: dispatches a tool id ("<adapter>/<tool>") to the registered adapter that supports it.

This is the only place a second adapter would be registered when another skill package becomes available.
Today exactly one adapter exists (ffmpeg-skill); the router adds no behaviour of its own."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import Operation, ToolResult
from ..skills.contract import SkillPackage
from .base import ToolAdapter, ToolError


class ToolRouter(ToolAdapter):
    name = "router"

    def __init__(self, adapters: Optional[List[ToolAdapter]] = None):
        self.adapters: List[ToolAdapter] = list(adapters or [])

    def register(self, adapter: ToolAdapter) -> None:
        self.adapters.append(adapter)

    def adapter_for(self, tool: str) -> Optional[ToolAdapter]:
        return next((a for a in self.adapters if a.supports(tool)), None)

    def supports(self, tool: str) -> bool:
        return self.adapter_for(tool) is not None

    def version_of(self, tool: str) -> str:
        a = self.adapter_for(tool)
        return getattr(a, "version", "?") if a else "?"

    @property
    def version(self) -> str:   # kept for callers that label results with the engine version
        return ", ".join(f"{a.name}@{getattr(a, 'version', '?')}" for a in self.adapters) or "none"

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "adapters": [a.describe() for a in self.adapters]}

    def packages(self) -> List[SkillPackage]:
        """Skill packages implemented by the registered adapters (identity + tools, versions as detected)."""
        return [a.package() for a in self.adapters]

    def _need(self, tool: str) -> ToolAdapter:
        a = self.adapter_for(tool)
        if a is None:
            raise ToolError(f"no registered adapter supports {tool}")
        return a

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        return self._need(op.tool).preview(op, paths)

    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        return self._need(op.tool).run(op, paths, timeout=timeout, dry_run=dry_run, attempt=attempt)

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        return self._need(tool).measure(tool, args, paths, timeout)

    def measurement_args(self, tool: str, kind: str, path: str, asset_id: str, parameters: Dict[str, Any], analysis_id: str, cache_policy: str) -> Optional[Dict[str, Any]]:
        a = self.adapter_for(tool)
        return a.measurement_args(tool, kind, path, asset_id, parameters, analysis_id, cache_policy) if a else None

    def owns_cache_for(self, tool: str) -> bool:
        a = self.adapter_for(tool)
        return bool(getattr(a, "owns_cache", False)) if a else False
