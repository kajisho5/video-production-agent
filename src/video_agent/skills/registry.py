"""Skill registry: what the system knows how to do, independent of what is installed (Capability) and
of what executes it (Tool). Phase 1 ships five skills; the registry is a plain contract, not a plugin system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .contract import SkillPackage, ToolSpec

# Skills whose phase is above this are declared for the roadmap only: they are never selectable and never "available".
IMPLEMENTED_PHASE = 1


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

    @property
    def implemented(self) -> bool:
        return self.phase <= IMPLEMENTED_PHASE

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["implemented"] = self.implemented
        return d


class SkillRegistry:
    """Discovers, records and lists production skills (SkillSpec) and Skill packages (SkillPackage) and selects a tool per
    skill for the current environment. It never makes a production decision (that is agent/decision.py)."""

    def __init__(self) -> None:
        self._skills: Dict[str, SkillSpec] = {}
        self._packages: Dict[str, SkillPackage] = {}

    def register(self, spec: SkillSpec) -> None:
        self._skills[spec.name] = spec

    # ---- Skill packages (ecosystem contract)
    def register_package(self, pkg: SkillPackage) -> None:
        """Record a Skill package (identity + tool contract). Re-registering the same skill_id replaces it (e.g. once an
        adapter has detected the installed version). A package that violates the contract is refused."""
        errs = pkg.validate()
        if errs:
            raise ValueError("invalid skill package: " + "; ".join(errs))
        self._packages[pkg.skill_id] = pkg

    def packages(self) -> List[SkillPackage]:
        return [self._packages[k] for k in sorted(self._packages)]

    def package(self, skill_id: str) -> Optional[SkillPackage]:
        return self._packages.get(skill_id)

    def tool(self, tool_id: str) -> Optional[ToolSpec]:
        """ToolSpec for a tool id, from the package owning its prefix; None if no registered package declares it."""
        pkg = self._packages.get(tool_id.split("/", 1)[0])
        return pkg.tool(tool_id) if pkg else None

    def unknown_tool_candidates(self) -> List[str]:
        """Tool candidates cited by *implemented* production skills that no registered package declares (skill/tool
        confusion check). Declared future skills may cite tools that are not catalogued yet; they are never selectable."""
        return sorted({f"{sp.name}: {t}" for sp in self.all() if sp.implemented for t in sp.tools if self.tool(t) is None})

    def package_availability(self, caps: Dict[str, Any], supports: Callable[[str], bool], versions: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Per package: implemented (an adapter for it exists in this codebase, i.e. it was registered), available (an adapter
        is registered with the router for its tools and its capabilities are present), and which tools are usable now."""
        rows = []
        for pkg in self.packages():
            missing = [c for c in pkg.capabilities if getattr(caps.get(c), "status", "MISSING") not in ("AVAILABLE", "DEGRADED")]
            usable = [t.tool_id for t in pkg.tools if supports(t.tool_id)]
            available = not missing and bool(usable)
            reason = "ok" if available else ("required capability missing: " + ", ".join(missing) if missing else "no registered adapter supports its tools")
            rows.append({"skill_id": pkg.skill_id, "name": pkg.name, "version": (versions or {}).get(pkg.skill_id) or pkg.version or "-", "repository": pkg.repository,
                         "role": pkg.role, "implemented": True, "available": available, "reason": reason, "tools": pkg.tool_ids(), "usable_tools": usable,
                         "used_by": sorted(sp.name for sp in self.all() if any(t.startswith(pkg.skill_id + "/") for t in sp.tools))})
        return rows

    def get(self, name: str) -> SkillSpec:
        return self._skills[name]

    def names(self) -> List[str]:
        return sorted(self._skills)

    def all(self) -> List[SkillSpec]:
        return [self._skills[n] for n in self.names()]

    def missing_capabilities(self, name: str, caps: Dict[str, Any]) -> List[str]:
        spec = self.get(name)
        return [c for c in spec.required_capabilities if getattr(caps.get(c), "status", "MISSING") not in ("AVAILABLE", "DEGRADED")]

    def tool_missing_capabilities(self, tool_id: str, caps: Dict[str, Any]) -> List[str]:
        """Capabilities the owning package or the ToolSpec require that are known to be MISSING here. A package-level
        capability (the Skill itself, e.g. "video-editing", resolved by its doctor) counts as missing when it is MISSING or
        not resolved at all, exactly as package_availability reports it. A ToolSpec capability (encoders, filters) only
        disqualifies on an explicit MISSING: an undetected one is reported by the validator as a warning, never guessed."""
        pkg = self.package(tool_id.split("/", 1)[0])
        if pkg is None:
            return []
        ts = pkg.tool(tool_id)
        missing = {c for c in pkg.capabilities if getattr(caps.get(c), "status", "MISSING") not in ("AVAILABLE", "DEGRADED", "UNKNOWN")}
        missing |= {c for c in (ts.required_capabilities if ts else []) if getattr(caps.get(c), "status", None) == "MISSING"}
        return sorted(missing)

    def select_tool(self, name: str, caps: Dict[str, Any], supports: Callable[[str], bool]) -> Tuple[Optional[str], str]:
        """Skill → Capability check → first tool candidate that a registered adapter supports and whose package / tool
        capabilities are not known to be missing. Returns (tool id, reason). Declared-but-unimplemented skills are never
        selected."""
        spec = self.get(name)
        if not spec.implemented:
            return None, f"skill {name} is declared for phase {spec.phase} and not implemented"
        missing = self.missing_capabilities(name, caps)
        if missing:
            return None, "required capability missing: " + ", ".join(missing)
        blocked: List[str] = []
        for tool in spec.tools:
            if not supports(tool):
                continue
            tm = self.tool_missing_capabilities(tool, caps)
            if tm:
                blocked.append(f"{tool} (missing {', '.join(tm)})")
                continue
            return tool, "ok"
        if blocked:
            return None, "candidate tools blocked by missing capabilities: " + "; ".join(blocked)
        return None, "no registered adapter supports any of: " + ", ".join(spec.tools)

    def resolve_tools(self, caps: Dict[str, Any], supports: Callable[[str], bool]) -> Dict[str, str]:
        """skill name → selected tool id, for every skill that is selectable in this environment."""
        out: Dict[str, str] = {}
        for spec in self.all():
            tool, _ = self.select_tool(spec.name, caps, supports)
            if tool:
                out[spec.name] = tool
        return out

    def availability(self, caps: Dict[str, Any], supports: Callable[[str], bool]) -> List[Dict[str, Any]]:
        """Human/machine listing: AVAILABLE (tool selected) / UNAVAILABLE (capability or adapter missing) / NOT_IMPLEMENTED
        (declared for a later phase). DECLARED == NOT_IMPLEMENTED; IMPLEMENTED == `implemented`; AVAILABLE == status."""
        rows = []
        for spec in self.all():
            tool, reason = self.select_tool(spec.name, caps, supports)
            status = "AVAILABLE" if tool else ("NOT_IMPLEMENTED" if not spec.implemented else "UNAVAILABLE")
            rows.append({"skill": spec.name, "version": spec.version, "phase": spec.phase, "status": status, "tool": tool, "reason": reason,
                         "implemented": spec.implemented, "packages": sorted({t.split("/", 1)[0] for t in spec.tools}),
                         "required_capabilities": spec.required_capabilities, "risk": spec.risk_level, "approval": spec.approval})
        return rows


def default_registry() -> SkillRegistry:
    r = SkillRegistry()
    r.register(SkillSpec("media_probe", "1.0", "Inspect media (duration, codecs, fps, HDR, audio)", {"asset": "media"}, {"observation": "probe"},
                         ["ffmpeg", "ffprobe"], "LOW", True, "AUTO", ["ffmpeg-skill/probe", "media-analysis/probe"]))
    r.register(SkillSpec("silence_analysis", "1.0", "Detect silences (list only)", {"asset": "media"}, {"events": "AUDIO_SILENCE"},
                         ["ffmpeg"], "LOW", True, "AUTO", ["ffmpeg-skill/silence", "media-analysis/silence"]))
    r.register(SkillSpec("loudness_analysis", "1.0", "Measure integrated loudness / true peak", {"asset": "media"}, {"observation": "loudness"},
                         ["ffmpeg", "filter:loudnorm"], "LOW", True, "AUTO", ["ffmpeg-skill/loudness", "media-analysis/loudness"]))
    # silence_cleanup: the Reference Skill first; video-editing-skill (ADR-028) realises the same keep-ranges operation through its
    # own typed contract on top of ffmpeg-skill, so it is the second candidate (selected when ffmpeg-skill's adapter is absent
    # or when a caller reorders the candidates; never by planner / compiler code)
    r.register(SkillSpec("silence_cleanup", "1.0", "Trim technical leading/trailing silence", {"asset": "video|audio", "keep": "ranges"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill", "encoder:libx264"], "LOW", True, "AUTO", ["ffmpeg-skill/cut", "video-editing/cut"]))
    r.register(SkillSpec("loudness_normalization", "1.0", "Two-pass EBU R128 normalisation", {"asset": "video|audio", "target_lufs": "float"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill", "filter:loudnorm"], "LOW", True, "AUTO", ["ffmpeg-skill/loudness"]))
    r.register(SkillSpec("delivery_export", "1.0", "Encode a delivery target with a platform preset", {"asset": "video", "preset": "str"}, {"artifact": "delivery"},
                         ["ffmpeg", "ffmpeg-skill", "encoder:libx264"], "LOW", True, "AUTO", ["ffmpeg-skill/export"]))
    r.register(SkillSpec("delivery_check", "1.0", "Platform compliance check", {"artifact": "delivery", "platform": "str"}, {"qa": "delivery"},
                         ["ffmpeg", "ffmpeg-skill"], "LOW", True, "AUTO", ["ffmpeg-skill/check"]))
    r.register(SkillSpec("visual_inspection", "1.0", "Contact sheet for human/AI review", {"artifact": "video"}, {"artifact": "THUMBNAIL"},
                         ["ffmpeg", "ffmpeg-skill"], "LOW", True, "AUTO", ["ffmpeg-skill/look"]))
    # measurement skills only media-analysis-skill provides (external observation Skill; tool ids from its contract)
    for name, kind, tool, caps, desc in (
        ("stream_layout_analysis", "stream_layout", "media-analysis/streams", ["ffprobe", "media-analysis"], "Every stream: index, type, codec, language, dimensions, rate, channels"),
        ("video_format_analysis", "video_format", "media-analysis/video", ["ffprobe", "media-analysis"], "Resolution, fps, pixel format, colour, SAR / DAR, CFR / VFR"),
        ("audio_format_analysis", "audio_format", "media-analysis/audio", ["ffprobe", "media-analysis"], "Sample rate, channels, layout, codec, sample format, bitrate"),
        ("duration_analysis", "duration", "media-analysis/timing", ["ffprobe", "media-analysis"], "Container / stream durations and start times"),
        ("integrity_analysis", "integrity", "media-analysis/integrity", ["ffmpeg", "ffprobe", "media-analysis"], "Full decode error count, frame counts, timestamp monotonicity (PASS / WARN / FAIL)"),
        ("scene_analysis", "scene_detection", "media-analysis/scenes", ["ffmpeg", "ffprobe", "media-analysis"], "Visual cuts with score (not semantic scenes)"),
        ("timing_analysis", "timing", "media-analysis/timing", ["ffprobe", "media-analysis"], "Packet timestamps, gaps, A/V duration mismatch"),
    ):
        r.register(SkillSpec(name, "1.0", desc, {"asset": "media"}, {"observation": kind}, caps, "LOW", True, "AUTO", [tool]))
    # speech recognition (transcription-skill, external Skill; recognition only: no speaker identity, no interpretation)
    r.register(SkillSpec("speech_transcription", "1.0", "Speech → timestamped Transcript (segments, language, optional word timestamps); recognition only",
                         {"asset": "media"}, {"observation": "transcript", "events": "SPEECH"}, ["ffmpeg", "ffprobe", "transcription"], "LOW", True, "AUTO", ["transcription/transcribe"]))
    # declared, not implemented in Phase 1 (registry keeps the contract visible)
    r.register(SkillSpec("multi_source_sync", "0.1", "Align cameras/recorders by audio", {"assets": "media[]"}, {"timeline": "offsets"},
                         ["ffmpeg", "ffmpeg-skill"], "MEDIUM", True, "CONFIRM", ["ffmpeg-skill/sync", "ffmpeg-skill/multicam"], phase=2))
    r.register(SkillSpec("caption_generation", "0.1", "Transcribe and burn captions", {"asset": "video"}, {"artifact": "CAPTIONS"},
                         ["ffmpeg", "ffmpeg-skill", "filter:libass", "asr:whisper"], "MEDIUM", False, "CONFIRM", ["ffmpeg-skill/caption"], phase=3))
    r.register(SkillSpec("semantic_deletion", "0.1", "Remove content based on meaning", {"asset": "video", "ranges": "ranges"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill"], "HIGH", False, "CONFIRM", ["ffmpeg-skill/cut", "video-editing/cut"], phase=4))
    return r
