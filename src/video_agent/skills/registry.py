"""Skill registry: what the system knows how to do, independent of what is installed (Capability) and
of what executes it (Tool). The registry is a plain contract, not a plugin system: every production skill is declared here with its
tool candidates (in declared order, no ranking, no fallback) and the capabilities it needs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .contract import SkillPackage, ToolSpec

# Skills whose phase is above this are declared for the roadmap only: they are never selectable and never "available".
IMPLEMENTED_PHASE = 1

# docs/CAPABILITY_MODEL.md's "Capability collision policy", Tier 2 (OS-level default): the package each currently-declared
# multi-candidate skill resolved to before Tiers 1-3 existed (its first-declared tools[] candidate). Named explicitly here,
# in the open, instead of left as an accident of declaration order -- this is the "silent default this OS replaces with an
# explicit, provenance-recorded choice" the roadmap's Phase 3 names. A workspace `providers.json` (skills/providers.py)
# overrides these per skill; a `--set provider.<skill>=<package>` requirement overrides both (Tier 1, always wins).
DEFAULT_PROVIDERS: Dict[str, str] = {
    "media_probe": "ffmpeg-skill",
    "silence_analysis": "ffmpeg-skill",
    "loudness_analysis": "ffmpeg-skill",
    "silence_cleanup": "ffmpeg-skill",
}


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

    def select_tool(self, name: str, caps: Dict[str, Any], supports: Callable[[str], bool],
                    explicit: Optional[str] = None, default: Optional[str] = None) -> Tuple[Optional[str], str]:
        """Skill → Capability check → the tool candidate this environment supports. Returns (tool id, reason).
        Declared-but-unimplemented skills are never selected.

        When more than one candidate is usable here (a real Provider collision, docs/CAPABILITY_MODEL.md's
        "Capability collision policy"), the OS never picks silently -- it applies, in order: (1) `explicit`
        (Tier 1, a Plan-time `provider.<skill>=<package>` choice -- always wins if the package it names is
        actually usable here), (2) `default` (Tier 2, a resolved default-provider policy -- OS-level baked-in,
        overridable per workspace by `providers.json`; see skills/providers.py), (3) refusal (Tier 3 -- more
        than one Provider is AVAILABLE and neither of the above resolved it, so this fails loudly rather than
        guessing). A skill with zero or one usable candidate has nothing to choose: neither tier is consulted."""
        spec = self.get(name)
        if not spec.implemented:
            return None, f"skill {name} is declared for phase {spec.phase} and not implemented"
        missing = self.missing_capabilities(name, caps)
        if missing:
            return None, "required capability missing: " + ", ".join(missing)
        candidates: List[str] = []
        blocked: List[str] = []
        for tool in spec.tools:
            if not supports(tool):
                continue
            tool_missing = self.tool_missing_capabilities(tool, caps)
            if tool_missing:   # the package / tool contract names capabilities this environment lacks: never executed on a guess
                blocked.append(f"{tool} (missing {', '.join(tool_missing)})")
                continue
            candidates.append(tool)
        if not candidates:
            if blocked:
                return None, "required capability missing for " + "; ".join(blocked)
            return None, "no registered adapter supports any of: " + ", ".join(spec.tools)
        if len(candidates) == 1:
            return candidates[0], "ok"
        if explicit is not None:
            picked = next((t for t in candidates if t.split("/", 1)[0] == explicit), None)
            if picked:
                return picked, "ok (explicit provider)"
            known = sorted({t.split("/", 1)[0] for t in candidates})
            return None, f"explicit provider {explicit!r} for skill {name} is not among the available providers ({', '.join(known)})"
        if default is not None:
            picked = next((t for t in candidates if t.split("/", 1)[0] == default), None)
            if picked:
                return picked, "ok (default provider)"
        builtin = DEFAULT_PROVIDERS.get(name)   # unset default=: the same choice a caller that resolves default_providers() would reach anyway
        if builtin is not None:
            picked = next((t for t in candidates if t.split("/", 1)[0] == builtin), None)
            if picked:
                return picked, "ok"
        packages = sorted({t.split("/", 1)[0] for t in candidates})
        return None, (f"{len(candidates)} providers available for skill {name} ({', '.join(packages)}) and none was chosen; "
                      f"pin one with --set provider.{name}=<package>, or set a workspace providers.json default")

    def tool_missing_capabilities(self, tool_id: str, caps: Dict[str, Any]) -> List[str]:
        """Capabilities the tool's package declares (its runtime requirements, e.g. ffmpeg-skill / video-editing) that are not
        AVAILABLE / DEGRADED here, plus ToolSpec.required_capabilities the resolver knows about (present in `caps`) but reports
        missing. Names the resolver does not resolve at all are left to the package's own doctor (which the package-level
        capability reflects); they are never guessed as present or absent here. A tool no registered package declares has
        no contract to check (the validator reports that case separately)."""
        pkg = self._packages.get(tool_id.split("/", 1)[0])
        spec = pkg.tool(tool_id) if pkg else None
        if pkg is None or spec is None:
            return []
        missing = [c for c in pkg.capabilities if getattr(caps.get(c), "status", "MISSING") not in ("AVAILABLE", "DEGRADED")]
        missing += [c for c in spec.required_capabilities if c not in pkg.capabilities and c in caps and getattr(caps.get(c), "status", "MISSING") not in ("AVAILABLE", "DEGRADED")]
        return missing

    def resolve_tools(self, caps: Dict[str, Any], supports: Callable[[str], bool],
                      explicit: Optional[Dict[str, str]] = None, default: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """skill name → selected tool id, for every skill that is selectable in this environment. `explicit` / `default`:
        per-skill provider choices (Tiers 1/2 of the collision policy; see select_tool)."""
        out: Dict[str, str] = {}
        for spec in self.all():
            tool, _ = self.select_tool(spec.name, caps, supports, explicit=(explicit or {}).get(spec.name), default=(default or {}).get(spec.name))
            if tool:
                out[spec.name] = tool
        return out

    def availability(self, caps: Dict[str, Any], supports: Callable[[str], bool],
                     explicit: Optional[Dict[str, str]] = None, default: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Human/machine listing: AVAILABLE (tool selected) / UNAVAILABLE (capability or adapter missing, or an
        unresolved Provider collision -- Tier 3 refusal) / NOT_IMPLEMENTED (declared for a later phase).
        DECLARED == NOT_IMPLEMENTED; IMPLEMENTED == `implemented`; AVAILABLE == status."""
        rows = []
        for spec in self.all():
            tool, reason = self.select_tool(spec.name, caps, supports, explicit=(explicit or {}).get(spec.name), default=(default or {}).get(spec.name))
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
    r.register(SkillSpec("silence_cleanup", "1.0", "Trim technical leading/trailing silence", {"asset": "video|audio", "keep": "ranges"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill", "encoder:libx264"], "LOW", True, "AUTO", ["ffmpeg-skill/cut", "video-editing/cut"]))   # candidates in declared order; video-editing/cut (ADR-028) is selectable when its package is available
    r.register(SkillSpec("loudness_normalization", "1.0", "Two-pass EBU R128 normalisation", {"asset": "video|audio", "target_lufs": "float"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill", "filter:loudnorm"], "LOW", True, "AUTO", ["ffmpeg-skill/loudness"]))
    r.register(SkillSpec("delivery_export", "1.0", "Encode a delivery target with a platform preset", {"asset": "video", "preset": "str"}, {"artifact": "delivery"},
                         ["ffmpeg", "ffmpeg-skill", "encoder:libx264"], "LOW", True, "AUTO", ["ffmpeg-skill/export"]))
    r.register(SkillSpec("delivery_check", "1.0", "Platform compliance check", {"artifact": "delivery", "platform": "str"}, {"qa": "delivery"},
                         ["ffmpeg", "ffmpeg-skill"], "LOW", True, "AUTO", ["ffmpeg-skill/check"]))
    # editing operations only video-editing-skill provides (ADR-029): explicit requirement → Decision (CONFIRM by default) → plan step
    r.register(SkillSpec("video_concat", "1.0", "Join two or more (trimmed) inputs into one programme, in the given order", {"assets": "video[]"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill", "video-editing"], "MEDIUM", True, "CONFIRM", ["video-editing/concat"]))
    r.register(SkillSpec("video_speed", "1.0", "Change playback speed by a factor", {"asset": "video", "factor": "float"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill", "video-editing"], "MEDIUM", True, "CONFIRM", ["video-editing/speed"]))
    r.register(SkillSpec("video_resize", "1.0", "Scale to a width (aspect kept)", {"asset": "video", "width": "int"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill", "video-editing"], "LOW", True, "CONFIRM", ["video-editing/resize"]))
    r.register(SkillSpec("video_fit", "1.0", "Letterbox / pillarbox into an aspect ratio (no picture lost)", {"asset": "video", "aspect": "W:H"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill", "video-editing"], "LOW", True, "CONFIRM", ["video-editing/fit"]))
    r.register(SkillSpec("video_fill", "1.0", "Crop into an aspect ratio (picture lost at the edges)", {"asset": "video", "aspect": "W:H"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill", "video-editing"], "MEDIUM", True, "CONFIRM", ["video-editing/fill"]))
    r.register(SkillSpec("video_overlay", "1.0", "Composite a PNG / JPEG image over the picture", {"asset": "video", "image": "png|jpg"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill", "video-editing"], "MEDIUM", True, "CONFIRM", ["video-editing/overlay"]))
    # video-editing-skill's own TRIM (start/end range, distinct from silence_cleanup's multi-range CUT) is a
    # real, tested, published Capability (video.trim) with no consuming SkillSpec at all until now (found by
    # skills/diagnostics.py's ecosystem-wide run, docs/design decision kajisho5/AI-video-production-OS
    # WORK_QUEUE.md item 1/8). Declared for the roadmap only, like multi_source_sync/semantic_deletion below:
    # wiring an actual "video.trim" edit request through agent/editing.py's EDIT_OPS and
    # agent/production_plan.py's domain-parameter table is real design work (a new request verb, its exact
    # parameter contract, a risk classification) this addition deliberately does not decide unprompted.
    r.register(SkillSpec("video_trim", "0.1", "Trim to an explicit start/end time range", {"asset": "video", "start": "time", "end": "time"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill", "video-editing"], "MEDIUM", True, "CONFIRM", ["video-editing/trim"], phase=2))
    # audio production operations only audio-production-skill provides (ADR-030): the audio path of an asset (explicit `audio.production`);
    # each skill needs the Skill's own per-operation capability (from its doctor) besides the package capability
    for name, typ, desc, inputs, risk in (
        ("audio_cut", "CUT", "Remove explicit source ranges from the audio (silence decisions)", {"asset": "audio", "remove": "ranges"}, "LOW"),
        ("audio_normalize", "NORMALIZE", "Two-pass EBU R128 normalisation of the audio (target / true peak from policy)", {"asset": "audio", "target_lufs": "float"}, "LOW"),
        ("audio_gain", "GAIN", "Fixed gain in dB", {"asset": "audio", "gain_db": "float"}, "LOW"),
        ("audio_mono", "MONO", "Stereo → mono down-mix", {"asset": "audio"}, "MEDIUM"),
        ("audio_stereo", "STEREO", "Mono → stereo (duplicated)", {"asset": "audio"}, "LOW"),
        ("audio_downmix", "DOWNMIX", "5.1 / 7.1 → stereo down-mix", {"asset": "audio"}, "MEDIUM"),
        ("audio_fade_in", "FADE_IN", "Linear fade in", {"asset": "audio", "duration": "float"}, "LOW"),
        ("audio_fade_out", "FADE_OUT", "Linear fade out", {"asset": "audio", "duration": "float"}, "LOW"),
        ("audio_concat", "CONCAT", "Join two or more audio subjects in order (optional crossfade)", {"assets": "audio[]"}, "MEDIUM"),
    ):
        r.register(SkillSpec(name, "1.0", desc, inputs, {"artifact": "INTERMEDIATE"}, ["ffmpeg", "ffprobe", "ffmpeg-skill", "audio-production", f"audio-production:{typ}"],
                             risk, True, "CONFIRM", ["audio-production/run"]))
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
    # ---- Phase 3 finishing Skills (ADR-031 / ADR-032): subtitle-skill replaces the former `caption_generation` declaration (which cited
    # ffmpeg-skill/caption directly); each Skill is reached only through its own package tool and needs the package capability
    r.register(SkillSpec("subtitle_generation", "1.0", "Transcript cues → SRT / WebVTT sidecar (mapped onto the delivered timeline)", {"transcript": "observation"}, {"artifact": "CAPTIONS"},
                         ["subtitle"], "LOW", True, "CONFIRM", ["subtitle/generate"]))
    r.register(SkillSpec("subtitle_burn_in", "1.0", "Burn the subtitle document into the picture (subtitle-skill render → ffmpeg-skill/caption)", {"asset": "video", "captions": "artifact"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffprobe", "ffmpeg-skill", "subtitle", "encoder:libx264", "filter:subtitles"], "MEDIUM", True, "CONFIRM", ["subtitle/render"]))
    r.register(SkillSpec("thumbnail_frame", "1.0", "One video frame at an explicit timestamp as a PNG / JPEG", {"asset": "video", "at": "float"}, {"artifact": "THUMBNAIL"},
                         ["ffmpeg", "ffprobe", "ffmpeg-skill", "thumbnail"], "LOW", True, "CONFIRM", ["thumbnail/extract_frame"]))
    r.register(SkillSpec("thumbnail_render", "1.0", "A video frame with a caption rendered as a thumbnail document", {"asset": "video", "at": "float", "text": "str"}, {"artifact": "THUMBNAIL"},
                         ["ffmpeg", "ffprobe", "ffmpeg-skill", "thumbnail"], "LOW", True, "CONFIRM", ["thumbnail/render"]))
    for name, typ, desc, inputs, risk in (
        ("color_strip_dovi", "STRIP_DOVI", "Strip Dolby Vision side data", {"asset": "video"}, "LOW"),
        ("color_hdr_to_sdr", "HDR_TO_SDR", "Tone-map an HDR source to SDR BT.709", {"asset": "video"}, "MEDIUM"),
        ("color_primary_correction", "PRIMARY_CORRECTION", "Typed primary colour correction: exposure, contrast, saturation, white balance (temperature + tint)", {"asset": "video"}, "MEDIUM"),
        ("color_lut", "LUT_APPLY", "Apply a 3D .cube LUT", {"asset": "video", "lut": "cube"}, "MEDIUM"),
        ("color_retag", "RETAG", "Re-tag the colour metadata (bt709 / bt2020-pq / bt2020-hlg / bt601)", {"asset": "video", "target": "str"}, "LOW"),
    ):
        r.register(SkillSpec(name, "1.0", desc, inputs, {"artifact": "INTERMEDIATE"}, ["ffmpeg", "ffprobe", "ffmpeg-skill", "color-grading", f"color-grading:{typ}"], risk, True, "CONFIRM", ["color-grading/run"]))
    r.register(SkillSpec("motion_graphics", "1.0", "Render titles / lower thirds / text and image overlays onto the picture (one request per subject)", {"asset": "video", "elements": "list"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffprobe", "ffmpeg-skill", "motion-graphics"], "MEDIUM", True, "CONFIRM", ["motion-graphics/run"]))
    r.register(SkillSpec("qc_check", "1.0", "Quality control report of a deliverable (qc-skill; the final promotion gate)", {"artifact": "delivery|captions"}, {"qa": "qc"},
                         ["ffprobe", "qc"], "LOW", True, "AUTO", ["qc/check"]))
    r.register(SkillSpec("semantic_deletion", "0.1", "Remove content based on meaning", {"asset": "video", "ranges": "ranges"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill"], "HIGH", False, "CONFIRM", ["ffmpeg-skill/cut"], phase=4))
    return r
