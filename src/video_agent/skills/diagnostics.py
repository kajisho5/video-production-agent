"""Provides/Capability consumption diagnostic (read-only; proposed as `kajisho5/
AI-video-production-OS` `docs/ecosystem/WORK_QUEUE.md` item 8, 2026-09-06 — that
project's own exhaustive investigation of this Agent's registry found two real
tool-id drift cases and several published-but-unconsumed capabilities; see this
module's own tests for both, reproduced as fixtures).

Generic, not specific to any one Skill, Agent, or ecosystem: answers one question for
any registered `SkillPackage` plus the Capability Contract document its adapter
fetched — "does this Agent correctly recognize and consume every Capability this
Skill's own contract publishes under `provides[]`?" — without assuming any particular
Skill's naming, id scheme, or tool-id convention. Nothing here is specific to
`kajisho5/AI-video-production-OS`'s own registry, schema, or naming; it is expressed
purely in terms of this codebase's own, already-established `SkillPackage` / `ToolSpec`
/ `SkillSpec` types (`skills/contract.py`, `skills/registry.py`).

Vocabulary note (see `skills/contract.py`'s own vocabulary section): "Capability"
already names two different things in this codebase's own model — `CapabilityResolver`'s
runtime environment-availability status (AVAILABLE/MISSING/DEGRADED/UNKNOWN), and, in
the wider ecosystem a Skill package may belong to, the dotted id a Skill's `provides[]`
entry names (e.g. "measure.audio.loudness"). This module is about the second sense
only; every docstring below says "Capability id" or "published Capability," never bare
"Capability," to avoid adding a third meaning to an already-overloaded word.

This module never executes a Skill, never calls `SkillRegistry.select_tool()`, and
never imports anything from `execution/` — it only reads a Capability Contract document
(a plain dict, already fetched by the caller) and the Agent's own already-registered,
static `SkillPackage` / `SkillSpec` declarations. It is pure and side-effect-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .contract import SkillPackage
from .registry import SkillSpec

STATUSES = ("PROVIDES_VALID", "PROVIDES_MISMATCH", "CAPABILITY_UNCONSUMED", "CAPABILITY_MISSING", "UNKNOWN")


@dataclass
class ProvidesFinding:
    """One fact about how a single published Capability id (or a `SkillSpec` tool
    candidate with no matching published Capability) relates to this Agent's own
    recognition and consumption of it. Shaped like this codebase's own `Capability`
    (`capabilities/resolver.py`) and `QAItem` (`qa/checks.py`) dataclasses —
    name/status/detail/evidence — deliberately, so a reader already familiar with
    either reads this the same way; not a new, unrelated reporting shape.
    """
    capability_id: str      # "" when there is no known Capability id (a CAPABILITY_MISSING/UNKNOWN case with no provides[] match at all)
    skill_id: str
    tool_id: str
    status: str              # one of STATUSES
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"capability_id": self.capability_id, "skill_id": self.skill_id, "tool_id": self.tool_id,
                "status": self.status, "detail": self.detail, "evidence": self.evidence}


def extract_provides(contract: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A Capability Contract's `provides` list, or `[]` when absent or malformed.

    Deliberately tiny and independent of any other repository's implementation — this
    package does not depend on any other repository's registry/validation library — but
    kept to the same minimal, real-world shape (`{id, tool_id, lifecycle}` per entry)
    that every Skill contract in practice actually publishes, not an assumption unique
    to this function.
    """
    if not isinstance(contract, dict):
        return []
    provides = contract.get("provides")
    return provides if isinstance(provides, list) else []


def check_provides(package: SkillPackage, contract: Optional[Dict[str, Any]], skill_specs: List[SkillSpec]) -> List[ProvidesFinding]:
    """The core algorithm; pure, no I/O.

    `package` — this Skill's already-registered `SkillPackage` (whatever tool ids this
    Agent's own adapter/package declaration actually knows about).
    `contract` — the live (or pinned) Capability Contract document the adapter fetched,
    or `None` when unavailable (not installed, or this adapter does not expose one).
    `skill_specs` — every production `SkillSpec` that may reference this package (pass
    `SkillRegistry.all()`; this function filters internally by `package.skill_id + "/"`,
    so passing every registered `SkillSpec` for every package is always safe and simplest).

    Never guesses: when `contract` is `None`, this function returns exactly one
    `UNKNOWN` finding for the whole package rather than inferring anything about its
    individual capabilities from the Agent's own expectations alone.
    """
    if contract is None:
        return [ProvidesFinding("", package.skill_id, "", "UNKNOWN",
                                 "no Capability Contract document was available for this package (not installed, "
                                 "or its adapter does not expose a fetched contract) -- consumption cannot be verified")]

    prefix = package.skill_id + "/"
    agent_tool_ids = {t.tool_id for t in package.tools}
    consumed_tool_ids = {tid for spec in skill_specs for tid in spec.tools}
    entries = [e for e in extract_provides(contract) if isinstance(e, dict) and e.get("id") and e.get("tool_id")]

    # Some Skills expose every Capability id through one shared, generic tool id (e.g. "qc/run",
    # "color-grading/run") -- the internal operation is a request parameter, not a distinguishable
    # tool id. Tool-id-level evidence alone cannot then confirm that any *one* of those sibling
    # Capabilities specifically is consumed, only that the shared tool id as a whole is -- so such
    # findings are annotated, never silently reported with the same confidence as an unshared one.
    tool_id_to_cap_ids: Dict[str, List[str]] = {}
    for e in entries:
        tool_id_to_cap_ids.setdefault(str(e["tool_id"]), []).append(str(e["id"]))

    provided_tool_ids: Dict[str, str] = {}   # tool_id -> the first Capability id seen for it
    findings: List[ProvidesFinding] = []

    for entry in entries:
        cap_id = str(entry["id"])
        tool_id = str(entry["tool_id"])
        provided_tool_ids.setdefault(tool_id, cap_id)
        lifecycle = entry.get("lifecycle")
        siblings = [c for c in tool_id_to_cap_ids[tool_id] if c != cap_id]
        shared_note = (f" -- this tool id is shared by {len(siblings) + 1} Capability ids "
                        f"({', '.join(sorted([cap_id] + siblings))}); this finding reflects only that the shared "
                        "tool id is reachable/consumed, not that this specific Capability id is separately "
                        "requested" if siblings else "")
        evidence: Dict[str, Any] = {"lifecycle": lifecycle}
        if siblings:
            evidence["shared_tool_id_capabilities"] = sorted([cap_id] + siblings)
        if tool_id in agent_tool_ids:
            if tool_id in consumed_tool_ids:
                findings.append(ProvidesFinding(cap_id, package.skill_id, tool_id, "PROVIDES_VALID",
                                 "published under provides[], recognized by this Agent's registered package, "
                                 "and referenced by at least one production SkillSpec" + shared_note, evidence))
            else:
                findings.append(ProvidesFinding(cap_id, package.skill_id, tool_id, "CAPABILITY_UNCONSUMED",
                                 "published under provides[] and recognized by this Agent's registered package, "
                                 "but no registered SkillSpec references this tool id yet" + shared_note, evidence))
        else:
            evidence["agent_tool_ids"] = sorted(agent_tool_ids)
            findings.append(ProvidesFinding(cap_id, package.skill_id, tool_id, "PROVIDES_MISMATCH",
                             f"the Skill's own Capability Contract publishes {cap_id!r} via tool_id {tool_id!r}, but "
                             f"this Agent's registered package for {package.skill_id!r} has no ToolSpec with that "
                             "tool id -- if the Agent still reaches this operation, it does so through a tool id of "
                             "its own choosing, not one derived from this contract; verify any integration by "
                             "Capability id, never by tool-id string equality" + shared_note, evidence))

    for spec in skill_specs:
        for tool_id in spec.tools:
            if not tool_id.startswith(prefix) or tool_id in provided_tool_ids:
                continue
            if tool_id in agent_tool_ids:
                findings.append(ProvidesFinding("", package.skill_id, tool_id, "UNKNOWN",
                                 f"SkillSpec {spec.name!r} references tool id {tool_id!r}, which this Agent's "
                                 "registered package does recognize, but the Skill's own Capability Contract does "
                                 "not publish any Capability id for it under provides[] yet -- likely just not "
                                 "migrated to provides[] yet, not necessarily broken",
                                 {"skill_spec": spec.name}))
            else:
                findings.append(ProvidesFinding("", package.skill_id, tool_id, "CAPABILITY_MISSING",
                                 f"SkillSpec {spec.name!r} references tool id {tool_id!r} for package "
                                 f"{package.skill_id!r}, but neither this Agent's registered package nor the "
                                 "Skill's own Capability Contract's provides[] recognizes that tool id at all",
                                 {"skill_spec": spec.name}))
    return findings


def check_all(packages: List[SkillPackage], contracts: Dict[str, Optional[Dict[str, Any]]], skill_specs: List[SkillSpec]) -> List[ProvidesFinding]:
    """`check_provides()` for every package, keyed by `package.skill_id` in `contracts`.
    A package with no entry in `contracts` is treated the same as an explicit `None`."""
    out: List[ProvidesFinding] = []
    for pkg in packages:
        out.extend(check_provides(pkg, contracts.get(pkg.skill_id), skill_specs))
    return out
