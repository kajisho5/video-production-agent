"""Skill registry: what the system knows how to do, independent of what is installed (Capability) and
of what executes it (Tool). Phase 1 ships five skills; the registry is a plain contract, not a plugin system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillSpec:
    name: str
    version: str
    description: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    required_capabilities: List[str]
    risk_level: str                 # LOW | MEDIUM | HIGH
    deterministic: bool
    approval: str                   # default approval: AUTO | CONFIRM | BLOCK
    tools: List[str]                # ordered candidates, e.g. ["ffmpeg-skill/cut"]
    phase: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: Dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec) -> None:
        self._skills[spec.name] = spec

    def get(self, name: str) -> SkillSpec:
        return self._skills[name]

    def names(self) -> List[str]:
        return sorted(self._skills)

    def all(self) -> List[SkillSpec]:
        return [self._skills[n] for n in self.names()]

    def missing_capabilities(self, name: str, caps: Dict[str, Any]) -> List[str]:
        spec = self.get(name)
        return [c for c in spec.required_capabilities if getattr(caps.get(c), "status", "MISSING") not in ("AVAILABLE", "DEGRADED")]


def default_registry() -> SkillRegistry:
    r = SkillRegistry()
    r.register(SkillSpec("media_probe", "1.0", "Inspect media (duration, codecs, fps, HDR, audio)", {"asset": "media"}, {"observation": "probe"},
                         ["ffmpeg", "ffprobe", "ffmpeg-skill"], "LOW", True, "AUTO", ["ffmpeg-skill/probe"]))
    r.register(SkillSpec("silence_analysis", "1.0", "Detect silences (list only)", {"asset": "media"}, {"events": "AUDIO_SILENCE"},
                         ["ffmpeg", "ffmpeg-skill"], "LOW", True, "AUTO", ["ffmpeg-skill/silence"]))
    r.register(SkillSpec("loudness_analysis", "1.0", "Measure integrated loudness / true peak", {"asset": "media"}, {"observation": "loudness"},
                         ["ffmpeg", "ffmpeg-skill", "filter:loudnorm"], "LOW", True, "AUTO", ["ffmpeg-skill/loudness"]))
    r.register(SkillSpec("silence_cleanup", "1.0", "Trim technical leading/trailing silence", {"asset": "video|audio", "keep": "ranges"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill", "encoder:libx264"], "LOW", True, "AUTO", ["ffmpeg-skill/cut"]))
    r.register(SkillSpec("loudness_normalization", "1.0", "Two-pass EBU R128 normalisation", {"asset": "video|audio", "target_lufs": "float"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill", "filter:loudnorm"], "LOW", True, "AUTO", ["ffmpeg-skill/loudness"]))
    r.register(SkillSpec("delivery_export", "1.0", "Encode a delivery target with a platform preset", {"asset": "video", "preset": "str"}, {"artifact": "delivery"},
                         ["ffmpeg", "ffmpeg-skill", "encoder:libx264"], "LOW", True, "AUTO", ["ffmpeg-skill/export"]))
    r.register(SkillSpec("delivery_check", "1.0", "Platform compliance check", {"artifact": "delivery", "platform": "str"}, {"qa": "delivery"},
                         ["ffmpeg", "ffmpeg-skill"], "LOW", True, "AUTO", ["ffmpeg-skill/check"]))
    r.register(SkillSpec("visual_inspection", "1.0", "Contact sheet for human/AI review", {"artifact": "video"}, {"artifact": "THUMBNAIL"},
                         ["ffmpeg", "ffmpeg-skill"], "LOW", True, "AUTO", ["ffmpeg-skill/look"]))
    # declared, not implemented in Phase 1 (registry keeps the contract visible)
    r.register(SkillSpec("multi_source_sync", "0.1", "Align cameras/recorders by audio", {"assets": "media[]"}, {"timeline": "offsets"},
                         ["ffmpeg", "ffmpeg-skill"], "MEDIUM", True, "CONFIRM", ["ffmpeg-skill/sync", "ffmpeg-skill/multicam"], phase=2))
    r.register(SkillSpec("caption_generation", "0.1", "Transcribe and burn captions", {"asset": "video"}, {"artifact": "CAPTIONS"},
                         ["ffmpeg", "ffmpeg-skill", "filter:libass", "asr:whisper"], "MEDIUM", False, "CONFIRM", ["ffmpeg-skill/caption"], phase=3))
    r.register(SkillSpec("semantic_deletion", "0.1", "Remove content based on meaning", {"asset": "video", "ranges": "ranges"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill"], "HIGH", False, "CONFIRM", ["ffmpeg-skill/cut"], phase=4))
    return r
