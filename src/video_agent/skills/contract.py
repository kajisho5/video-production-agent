"""Ecosystem contract: what a Skill package must declare so the agent can use it without engine-specific code.

Vocabulary (see docs/skills.md):
  Skill package  – an independent repository / capability domain (today: kajisho5/ffmpeg-skill, the Reference Skill).
  Tool           – one concrete operation the package provides ("<skill_id>/<tool>", e.g. "ffmpeg-skill/cut").
  Capability     – what the runtime environment supports (CapabilityResolver: AVAILABLE / MISSING / DEGRADED / UNKNOWN).
  Adapter        – connects the package's tools to the runtime (ToolAdapter); executes the selected tool, decides nothing.
  Production skill (SkillSpec) – what the agent knows how to accomplish (silence_cleanup …); its `tools` are candidates
                   drawn from registered packages, selected by SkillRegistry.select_tool.

This module is a typed contract only. There is no package loader, plugin manager or dynamic import: a package becomes
known when its adapter module registers a SkillPackage (one line in Service.adapter()).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolSpec:
    """A concrete operation offered by a Skill package. Engine details (flags, argv) stay inside the adapter."""
    tool_id: str                                  # "<skill_id>/<name>"
    skill_id: str                                 # owning package
    version: str = ""                             # package version the contract was verified against ("" = follows the package)
    description: str = ""
    required_capabilities: List[str] = field(default_factory=list)   # in addition to the package's own capabilities
    # execution contract
    inputs: List[str] = field(default_factory=list)     # named inputs the agent passes (media paths / artifact ids)
    produces_output: bool = False                        # writes an artifact (transform) vs. measurement only
    deterministic: bool = True
    result_keys: List[str] = field(default_factory=list)  # keys the adapter guarantees in ToolResult.data

    @property
    def kind(self) -> str:
        return "transform" if self.produces_output else "measure"

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["kind"] = self.kind
        return d


@dataclass
class SkillPackage:
    """Minimum identity a Skill repository must provide. Registered by its adapter module; never discovered dynamically."""
    skill_id: str                                 # tool id prefix, e.g. "ffmpeg-skill"
    name: str
    version: str                                  # declared / detected version ("" when not installed)
    description: str
    capabilities: List[str] = field(default_factory=list)   # runtime capabilities the package needs (CapabilityResolver names)
    tools: List[ToolSpec] = field(default_factory=list)
    repository: str = ""                          # e.g. "kajisho5/ffmpeg-skill"
    role: str = ""                                # e.g. "reference", "deterministic media processing"

    def tool_ids(self) -> List[str]:
        return [t.tool_id for t in self.tools]

    def tool(self, tool_id: str) -> Optional[ToolSpec]:
        return next((t for t in self.tools if t.tool_id == tool_id), None)

    def validate(self) -> List[str]:
        """Contract violations (empty list = valid)."""
        errs: List[str] = []
        if not self.skill_id or "/" in self.skill_id:
            errs.append(f"skill_id must be a non-empty prefix without '/': {self.skill_id!r}")
        if not self.name:
            errs.append("name is required")
        if not self.tools:
            errs.append(f"{self.skill_id}: a package must declare at least one tool")
        seen = set()
        for t in self.tools:
            if t.skill_id != self.skill_id:
                errs.append(f"{t.tool_id}: skill_id {t.skill_id!r} does not match package {self.skill_id!r}")
            if not t.tool_id.startswith(self.skill_id + "/") or t.tool_id.count("/") != 1:
                errs.append(f"{t.tool_id}: tool_id must be '{self.skill_id}/<name>'")
            if t.tool_id in seen:
                errs.append(f"{t.tool_id}: duplicate tool id")
            seen.add(t.tool_id)
        return errs

    def to_dict(self) -> Dict[str, Any]:
        return {"skill_id": self.skill_id, "name": self.name, "version": self.version, "description": self.description,
                "capabilities": list(self.capabilities), "repository": self.repository, "role": self.role, "tools": [t.to_dict() for t in self.tools]}
