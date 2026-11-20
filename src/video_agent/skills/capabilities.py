"""Capabilities read from the pinned contracts (ADR-118): one loader for every pin, so a production skill can name the capability it
needs and get its tool from the Skill's own `provides[]`, instead of a hand-written tool id.

    capability id (`voice.isolate`) ─→ pinned provides[] entry (tool_id `voice-enhance/isolate`, operation, lifecycle)
        ─→ SkillSpec.tools ─→ SkillRegistry.select_tool / ToolRouter: the registered adapter that supports that tool id

The pins are the files the repin bot writes (ADR-117): `tools/<dir>/contract_<version>.json` for a Skill with an adapter and
`skills/pins/<skill_id>_<version>.json` for a handed-off one. They are byte-identical copies of the Skills' release contracts and this
module only reads them. A pin is found by globbing, never by a version written here, so a re-pin moves nothing in this file.

Three things live here:

- `pinned_skills()` / `providers()` / `resolve()`: the index of every `provides[]` entry of every pin. `resolve()` raises
  `CapabilityNotProvided` when no pinned Skill provides a capability, naming the pins it read and the capability ids that share the
  same prefix, so a Skill that drops or renames a capability fails the unit suite at its re-pin with a message that says what moved.
- `CAPABILITY_SKILLS`: the production skills whose tool now comes from a pin (skill name → capability id). The remaining hand-wired
  SkillSpecs are listed in ADR-118 and migrate one package at a time.
- `contract_gate()`: the check an adapter's `check_contract` runs on a live contract: every capability the agent consumes from that
  package is still in `provides[]`, with the tool the pin names. The lifecycle is not compared (a capability leaving EXPERIMENTAL is
  reported as drift, never refused).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

_SKILLS_DIR = Path(__file__).resolve().parent
TOOLS_DIR = _SKILLS_DIR.parent / "tools"
HANDOFF_PINS_DIR = _SKILLS_DIR / "pins"

# Production skills whose tool candidate is read from a pin (ADR-118). Each capability must be provided by exactly one pinned Skill.
CAPABILITY_SKILLS: Dict[str, str] = {
    # voice-enhance-skill (ADR-073)
    "voice_isolate": "voice.isolate",
    "voice_dereverb": "voice.dereverb",
    "voice_bleed_reduce": "voice.bleed_reduce",
    # detection-skill (ADR-090, ADR-090 addendum 1, ADR-104)
    "face_detection": "vision.detect_faces",
    "face_attributes": "vision.detect_faces",
    "text_detection": "vision.detect_text",
    "shot_detection": "vision.detect_shots",
    # diarization-skill (ADR-086)
    "speaker_diarization": "speech.diarize",
}


class CapabilityNotProvided(LookupError):
    """No pinned Skill provides the capability (or more than one does and none was named)."""


@dataclass(frozen=True)
class ProvidedCapability:
    """One `provides[]` entry of a pinned contract."""
    id: str                       # capability id, e.g. "voice.isolate"
    tool_id: str                  # the tool that provides it, e.g. "voice-enhance/isolate"
    operation: str                # the operation inside that tool: the entry's own `operation` / `element_type` / `kind`, else the tool name
    lifecycle: str                # EXPERIMENTAL | STABLE | ... as the Skill states it
    skill_id: str                 # the contract's own skill id (may differ from the tool id prefix, e.g. subtitle-skill → subtitle/...)
    version: str                  # the pinned release
    pin: str                      # path of the pin, relative to the package source

    @property
    def package(self) -> str:
        """The tool id prefix: the name the agent's adapters and SkillPackages use."""
        return self.tool_id.split("/", 1)[0]

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "tool_id": self.tool_id, "operation": self.operation, "lifecycle": self.lifecycle, "skill_id": self.skill_id,
                "version": self.version, "pin": self.pin, "package": self.package}


@dataclass(frozen=True)
class PinnedSkill:
    skill_id: str
    version: str
    pin: str
    provides: Tuple[ProvidedCapability, ...]


def _skill_id(contract: Dict[str, Any]) -> str:
    """The contract's own identity: `skill_id`, else `skill.id` (imagemagick-skill, hyperframes-skill), else `id`."""
    sid = contract.get("skill_id")
    if not sid and isinstance(contract.get("skill"), dict):
        sid = contract["skill"].get("id")
    return str(sid or contract.get("id") or "")


def provides_of(contract: Any, skill_id: str = "", version: str = "", pin: str = "") -> Tuple[ProvidedCapability, ...]:
    """The well-formed `provides[]` entries of a contract (an `id` and a `tool_id`); anything else is skipped, never guessed."""
    if not isinstance(contract, dict) or not isinstance(contract.get("provides"), list):
        return ()
    out = []
    for e in contract["provides"]:
        if not isinstance(e, dict) or not e.get("id") or not e.get("tool_id"):
            continue
        tool = str(e["tool_id"])
        op = e.get("operation") or e.get("element_type") or e.get("kind") or tool.split("/", 1)[-1]
        out.append(ProvidedCapability(str(e["id"]), tool, str(op), str(e.get("lifecycle") or ""), skill_id or _skill_id(contract),
                                      version or str(contract.get("version") or ""), pin))
    return tuple(out)


def pin_files() -> List[Path]:
    """Every pinned contract: `tools/<dir>/contract_*.json` and `skills/pins/*.json`, sorted."""
    return sorted(TOOLS_DIR.glob("*/contract_*.json")) + sorted(HANDOFF_PINS_DIR.glob("*.json"))


def read_pin(path: Path) -> PinnedSkill:
    contract = json.loads(path.read_text(encoding="utf-8"))
    rel = path.relative_to(_SKILLS_DIR.parent).as_posix()
    sid, ver = _skill_id(contract), str(contract.get("version") or "")
    return PinnedSkill(sid, ver, rel, provides_of(contract, sid, ver, rel))


@lru_cache(maxsize=1)
def pinned_skills() -> Tuple[PinnedSkill, ...]:
    """Every pin, read once per process (the pins ship with the code and do not change while it runs)."""
    return tuple(read_pin(p) for p in pin_files())


def all_capabilities() -> Tuple[ProvidedCapability, ...]:
    return tuple(c for s in pinned_skills() for c in s.provides)


def providers(capability_id: str) -> List[ProvidedCapability]:
    """Every pinned provides[] entry for a capability id (empty when none)."""
    return [c for c in all_capabilities() if c.id == capability_id]


def _not_provided(capability_id: str, package: Optional[str]) -> CapabilityNotProvided:
    head = capability_id.split(".", 1)[0] + "."
    near = sorted({c.id for c in all_capabilities() if c.id.startswith(head)})
    pins = ", ".join(f"{s.skill_id} {s.version}" for s in pinned_skills())
    where = f" from package {package}" if package else ""
    hint = f"; pinned capabilities under {head!r}: {', '.join(near)}" if near else f"; no pinned capability starts with {head!r}"
    return CapabilityNotProvided(f"no pinned skill provides capability {capability_id!r}{where} (pins read: {pins}){hint}. "
                                 "Either the Skill dropped or renamed it at its last re-pin, or the capability id here is wrong; "
                                 "the pins are never edited by hand (ADR-117), so fix the capability id or wait for the Skill to publish it")


def resolve(capability_id: str, package: Optional[str] = None) -> ProvidedCapability:
    """The one pinned provides[] entry for a capability id (optionally from a given package, by tool id prefix or skill id).
    CapabilityNotProvided when none does, or when several do and no package was named."""
    found = [c for c in providers(capability_id) if package is None or package in (c.package, c.skill_id)]
    if not found:
        raise _not_provided(capability_id, package)
    if len(found) > 1:
        raise CapabilityNotProvided(f"capability {capability_id!r} is provided by more than one pinned skill "
                                    f"({', '.join(sorted(c.package for c in found))}); name the package")
    return found[0]


def skill_capability(skill: str) -> Optional[str]:
    """The capability id a production skill is read from, or None for a SkillSpec that is still hand-wired."""
    return CAPABILITY_SKILLS.get(skill)


def tool_for_skill(skill: str) -> str:
    """The tool id a capability-backed production skill runs, from its pin. KeyError for a hand-wired skill."""
    return resolve(CAPABILITY_SKILLS[skill]).tool_id


def consumed_capabilities(package: str) -> List[str]:
    """The capability ids the agent consumes from a package (its tool id prefix), sorted; empty for a package with no migrated skill."""
    return sorted({cap for cap in CAPABILITY_SKILLS.values() for c in providers(cap) if c.package == package})


def contract_gate(contract: Any, package: str, capability_ids: Optional[Iterable[str]] = None) -> List[str]:
    """What the agent needs from a live contract's provides[]: each consumed capability present, with the tool the pin names. Problems
    as strings (empty = passes). `capability_ids` defaults to every capability the agent consumes from `package` (consumed_capabilities)."""
    caps = list(capability_ids) if capability_ids is not None else consumed_capabilities(package)
    live = {c.id: c for c in provides_of(contract)}
    errs: List[str] = []
    for cap in caps:
        got = live.get(cap)
        if got is None:
            errs.append(f"provides lacks {cap}")
            continue
        pinned = [p for p in providers(cap) if p.package == package]
        if pinned and got.tool_id != pinned[0].tool_id:
            errs.append(f"provides {cap} names tool {got.tool_id}, the pin names {pinned[0].tool_id}")
    return errs


@dataclass(frozen=True)
class CapabilityRoute:
    """capability → operation → adapter: what runs a capability here."""
    capability: ProvidedCapability
    adapter: Optional[str]            # the registered adapter's name, None when no installed adapter supports the tool (a handoff)

    @property
    def tool_id(self) -> str:
        return self.capability.tool_id

    @property
    def operation(self) -> str:
        return self.capability.operation


def route(capability_id: str, adapter_for: Callable[[str], Any], package: Optional[str] = None) -> CapabilityRoute:
    """The pinned tool and operation of a capability and the adapter that would run it (`adapter_for` is ToolRouter.adapter_for).
    CapabilityNotProvided when no pin provides the capability; an adapter of None is not an error (the step is handed off, ADR-051)."""
    cap = resolve(capability_id, package)
    a = adapter_for(cap.tool_id)
    return CapabilityRoute(cap, getattr(a, "name", None) if a is not None else None)
