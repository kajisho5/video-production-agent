"""ffmpeg-skill as the Reference Skill package: identity + tool contract derived from the typed CATALOG.
The version is filled in by the adapter once the checkout is located; the declared package carries the
contract-verified version range instead."""
from __future__ import annotations

from ...skills.contract import SkillPackage, ToolSpec
from .catalog import CATALOG

SKILL_ID = "ffmpeg-skill"
PREFIX = SKILL_ID + "/"
SUPPORTED_RANGE = "0.8.4 <= v < 0.9"
BASE_CAPABILITIES = ["ffmpeg", "ffprobe", "ffmpeg-skill"]

_EXTRA_CAPS = {"loudness": ["filter:loudnorm"], "cut": ["encoder:libx264"], "export": ["encoder:libx264"], "fit": ["encoder:libx264"],
               "multicam": ["encoder:libx264"], "sync": ["encoder:libx264"]}

_DESCRIPTIONS = {"probe": "Inspect media (duration, streams, HDR, VFR)", "silence": "Detect silences / keep ranges", "loudness": "Measure or normalise loudness (EBU R128)",
                 "cut": "Cut / trim segments (lossless or frame-accurate)", "fit": "Fit duration / aspect", "export": "Encode with a platform preset", "check": "Platform compliance check",
                 "look": "Contact sheet", "scenes": "Scene / highlight detection", "sync": "Align two sources by audio", "multicam": "Multi-camera switch", "report": "Before/after report"}


def package(version: str = "") -> SkillPackage:
    tools = [ToolSpec(tool_id=PREFIX + name, skill_id=SKILL_ID, version=version, description=_DESCRIPTIONS.get(name, name),
                      required_capabilities=list(_EXTRA_CAPS.get(name, [])), inputs=list(spec["positional"]),
                      produces_output=bool(spec.get("produces_output")), deterministic=True, result_keys=list(spec.get("result_keys", [])))
             for name, spec in CATALOG.items()]
    return SkillPackage(skill_id=SKILL_ID, name="ffmpeg-skill", version=version, description=f"Deterministic media processing (Reference Skill, contract verified for {SUPPORTED_RANGE})",
                        capabilities=list(BASE_CAPABILITIES), tools=tools, repository="kajisho5/ffmpeg-skill", role="reference: deterministic media processing")


PACKAGE = package()
