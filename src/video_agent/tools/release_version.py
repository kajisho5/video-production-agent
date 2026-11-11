"""ADR-108: compare a sibling Skill's live contract with its pin by range on the release version, and by equality on everything else.

A Skill writes its release version into its contract, and some Skills write it more than once. The fields are the top-level `version`
and, where a contract has them, `tools[].version`, `skill.version`, `tool_version`, `response_shape.skill.version`,
`doctor_shape.skill.version` and `versioning.version` (RELEASE_VERSION_FIELDS). A release that changes only CI, tests or the release
procedure changes those fields and nothing else.

`compare_to_pin()` does not leave those fields out of the comparison. It works in two steps:

1. Range: the live release version must be inside the adapter's SUPPORTED_SKILL_VERSIONS (the same prefix gate `check_contract`
   uses), and so must the pinned one. A version outside the range is a problem, exactly as before.
2. Content: in a copy of the live contract, every release-version field whose value is the live release version is set to the
   pinned release version. The copy is then compared with the pin by the adapter's own drift function, unchanged. A release-version
   field that holds some other value (a tool on a version of its own) is not touched, so a difference there is still drift.

So a release that differs from the pin only in its version number, inside the range, passes, and `PinComparison.version_only` says so:
the caller must report it (the integration suite prints a warning and a summary). Any other difference fails as before, and the pin
must be taken again. This module only reads; it changes neither the adapters' drift reports nor what gates a capability at run time.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

# The fields a contract uses to repeat its release version. Every pinned contract was checked: each of these, where present, holds
# exactly the top-level `version`. Several tools[].version: video-editing, media-analysis, color-grading, audio-production, motion-graphics;
# response_shape / doctor_shape / versioning: video-editing; skill.version: imagemagick, ingest, nle-bridge, publish, remotion; tool_version: artifacts.
RELEASE_VERSION_FIELDS = ("version", "tools[].version", "skill.version", "tool_version", "response_shape.skill.version",
                          "doctor_shape.skill.version", "versioning.version")


class VersionOnlyDrift(UserWarning):
    """A live Skill contract that differs from its pin only in its release version, inside the supported range (ADR-108)."""


def _slots(doc: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """(field name, the dict holding its `version` / `tool_version` key) for each RELEASE_VERSION_FIELDS entry present in `doc`."""
    if "version" in doc:
        yield "version", doc
    tools = doc.get("tools")
    if isinstance(tools, list):
        for i, t in enumerate(tools):
            if isinstance(t, dict) and "version" in t:
                yield f"tools[{i}].version", t
    if isinstance(doc.get("skill"), dict) and "version" in doc["skill"]:
        yield "skill.version", doc["skill"]
    if "tool_version" in doc:
        yield "tool_version", doc
    for shape in ("response_shape", "doctor_shape"):
        s = doc.get(shape)
        if isinstance(s, dict) and isinstance(s.get("skill"), dict) and "version" in s["skill"]:
            yield f"{shape}.skill.version", s["skill"]
    if isinstance(doc.get("versioning"), dict) and "version" in doc["versioning"]:
        yield "versioning.version", doc["versioning"]


def restate_release_version(doc: Dict[str, Any], version: str) -> Tuple[Dict[str, Any], List[str]]:
    """A deep copy of `doc` in which every release-version field that holds `doc`'s own release version holds `version` instead, and
    the names of the fields that were changed. A field holding any other value is left as it is."""
    out = copy.deepcopy(doc)
    own = out.get("version")
    if not isinstance(own, str) or not own:
        return out, []
    changed: List[str] = []
    for name, holder in list(_slots(out)):
        key = "tool_version" if name == "tool_version" else "version"
        if holder.get(key) == own:
            holder[key] = version
            changed.append(name)
    return out, changed


@dataclass
class PinComparison:
    """The result of comparing one live contract with its pin."""
    skill: str
    live_version: str
    pinned_version: str
    supported: Tuple[str, ...]
    out_of_range: List[str] = field(default_factory=list)    # a release version outside SUPPORTED_SKILL_VERSIONS (live or pinned)
    content_drift: List[str] = field(default_factory=list)   # the drift function's findings on the contract with the version restated
    restated: List[str] = field(default_factory=list)        # the release-version fields that held the live version and were restated

    @property
    def problems(self) -> List[str]:
        return self.out_of_range + self.content_drift

    @property
    def version_only(self) -> bool:
        """True when the live contract differs from the pin in its release version only, inside the range: passes, must be reported."""
        return self.live_version != self.pinned_version and not self.problems

    def describe(self) -> str:
        if self.problems:
            return f"{self.skill}: live {self.live_version} vs pin {self.pinned_version}: " + "; ".join(self.problems)
        if self.version_only:
            return (f"{self.skill}: live {self.live_version} differs from the pin {self.pinned_version} only in its version "
                    f"({', '.join(_condensed(self.restated))}), inside the supported range {' / '.join(p + 'x' for p in self.supported)}; "
                    "the pin takes it at the next re-pin")
        return f"{self.skill}: live == pin ({self.live_version})"


def _condensed(names: List[str]) -> List[str]:
    """Field names for a message: the `tools[i].version` entries as one `tools[].version (N)`."""
    tools = [n for n in names if n.startswith("tools[")]
    rest = [n for n in names if not n.startswith("tools[")]
    return rest[:1] + ([f"tools[].version ({len(tools)})"] if tools else []) + rest[1:]


def compare_to_pin(skill: str, live: Dict[str, Any], pinned: Dict[str, Any], supported: Sequence[str],
                   drift: Callable[[Dict[str, Any], Optional[Dict[str, Any]]], List[str]]) -> PinComparison:
    """Compare `live` with `pinned`: the release versions by `supported` (the adapter's SUPPORTED_SKILL_VERSIONS prefixes), the rest
    by `drift(live, pinned)` (the adapter's own drift function) on the live contract with its release version restated as the pin's."""
    supported = tuple(supported)
    lv, pv = str(live.get("version") or ""), str(pinned.get("version") or "")
    out_of_range = [f"{which} version {v!r} outside the supported range {supported}"
                    for which, v in (("live", lv), ("pinned", pv)) if not (supported and v.startswith(supported))]
    restated_live, restated = restate_release_version(live, pv) if lv != pv else (live, [])
    return PinComparison(skill=skill, live_version=lv, pinned_version=pv, supported=supported, out_of_range=out_of_range,
                         content_drift=list(drift(restated_live, pinned)), restated=restated if lv != pv else [])
