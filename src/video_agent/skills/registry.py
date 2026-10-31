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
            usable = [t.tool_id for t in pkg.tools if supports(t.tool_id) and not self.tool_missing_capabilities(t.tool_id, caps)]
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
            return None, "required capability missing: " + ", ".join(_with_reason(c, caps) for c in missing)
        candidates: List[str] = []
        blocked: List[str] = []
        for tool in spec.tools:
            if not supports(tool):
                continue
            tool_missing = self.tool_missing_capabilities(tool, caps)
            if tool_missing:   # the package / tool contract names capabilities this environment lacks: never executed on a guess
                blocked.append(f"{tool} (missing {', '.join(_with_reason(c, caps) for c in tool_missing)})")
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


def _with_reason(cap: str, caps: Dict[str, Any]) -> str:
    """A missing capability as the selection reason names it: with the resolver's evidence["reason"] when it recorded one
    (ADR-056: the ffmpeg-skill contract's word on why that one tool cannot be used), so a handoff says which flag broke."""
    reason = (getattr(caps.get(cap), "evidence", None) or {}).get("reason")
    return f"{cap}: {reason}" if reason else cap


def _job(name: str, desc: str, inputs: Dict[str, str], outputs: Dict[str, str], package: str, tool: str, risk: str, approval: str = "CONFIRM",
         deterministic: bool = True) -> SkillSpec:
    return SkillSpec(name, "0.1", desc, inputs, outputs, [package], risk, deterministic, approval, [f"{package}/{tool}"], phase=2)


# ADR-057: one phase-2 SkillSpec per ecosystem job no adapter here runs; its package is the job's owner in ecosystem.json
# (skills/ecosystem.py). Disc authoring is four jobs because ecosystem.json gives it to four packages. ADR-066: `media_ingest` (ingest) left this
# list when its adapter landed: it is the implemented `media_offload` below (roadmap step 9, one job per PR as each adapter ships). ADR-073:
# `voice_enhancement` (voice-enhance) left it the same way: the implemented voice_isolate / voice_dereverb / voice_bleed_reduce below. ADR-083:
# `speaker_diarization` (diarization) too: the implemented measurement of the same name below (analysis kind speaker_turns).
ECOSYSTEM_JOB_SKILLS: Tuple[SkillSpec, ...] = (
    _job("dubbing", "AI narration (TTS), dubbing / voice replacement timed to the source, audio description (cloning a voice needs its owner's consent)",
         {"asset": "video", "script": "text"}, {"artifact": "INTERMEDIATE"}, "dubbing", "dub", "HIGH", deterministic=False),
    _job("sound_design", "Generate a music bed (BGM) or sound effects",
         {"brief": "text"}, {"artifact": "audio"}, "sound-design", "generate", "MEDIUM", deterministic=False),
    _job("visual_detection", "Face, text (OCR) and object detection on the picture",
         {"asset": "video|image"}, {"observation": "detections"}, "detection", "detect", "LOW", approval="AUTO", deterministic=False),
    _job("slide_sync", "Import a slide deck, detect slide changes in the recording and match each segment to its slide",
         {"asset": "video", "deck": "pdf|pptx|key"}, {"observation": "slide_segments"}, "slide-sync", "match", "LOW", approval="AUTO"),
    _job("platform_publish", "Upload a QC-passed deliverable with its metadata to a delivery platform",
         {"artifact": "delivery", "platform": "str"}, {"receipt": "upload"}, "publish", "upload", "HIGH"),
    # ADR-077: imagemagick-skill's package id is its own `skill.id` (imagemagick-skill), the prefix of its provides[].tool_id; the overlay
    # preparation is implemented (image_strip / image_prepare / image_pad below), editing a still as a deliverable of its own is not
    _job("still_image_edit", "Still-image editing with ImageMagick (resize, crop, convert, optimise, strip metadata), original never overwritten",
         {"image": "png|jpg|heic|webp"}, {"artifact": "image"}, "imagemagick-skill", "edit", "LOW"),
    _job("render_3d", "3D assets with headless Blender: inspect, convert, optimise and render",
         {"model": "gltf|glb|fbx|obj|usd|blend"}, {"artifact": "image|video|model"}, "blender", "render", "MEDIUM"),
    _job("measured_motion_graphics", "Remotion motion graphics on the footage, timed from ffmpeg-skill measurements (captions, jump cuts, beat-synced motion)",
         {"asset": "video"}, {"artifact": "video"}, "remotion-ffmpeg", "pipeline", "MEDIUM"),
    _job("html_scene_render", "Render HTML scenes to verified video with HyperFrames",
         {"scene": "html"}, {"artifact": "video"}, "hyperframes", "render", "MEDIUM"),
    _job("document_verification", "Verify an AI-generated document (script, captions, report) and issue an evidence-backed receipt",
         {"document": "file"}, {"receipt": "verification"}, "artifacts", "verify", "LOW", approval="AUTO"),
    _job("bluray_authoring", "Author a Blu-ray (BDMV): titles, chapters, menus, subtitles",
         {"asset": "video"}, {"artifact": "BDMV"}, "bdmvkit", "build", "MEDIUM"),
    _job("dvd_authoring", "Author a DVD-Video (VIDEO_TS): titles, menus, subtitles",
         {"asset": "video"}, {"artifact": "VIDEO_TS"}, "vtskit", "author", "MEDIUM"),
    _job("disc_image", "Write a UDF disc image (.iso) of an authored DVD-Video or Blu-ray folder",
         {"disc": "VIDEO_TS|BDMV"}, {"artifact": "iso"}, "udfkit", "image", "LOW"),
    _job("dvd_line21_captions", "Insert line-21 closed captions (CEA-608) into DVD MPEG-2 video without re-encoding",
         {"video": "m2v", "captions": "scc|srt"}, {"artifact": "m2v"}, "dvdcc", "caption", "MEDIUM"),
)


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
    # video-editing-skill's own TRIM (one start/end range, distinct from silence_cleanup's multi-range CUT): declared for the roadmap
    # only until ADR-062, which wires it -- `edit.trim` (an explicit in / out range) → Decision → the asset's one video.trim, executed
    # by video-editing/trim when one kept range remains (internal pauses removed inside it keep the multi-range silence_cleanup cut)
    r.register(SkillSpec("video_trim", "1.0", "Trim to an explicit start/end time range", {"asset": "video", "keep": "ranges"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill", "video-editing"], "MEDIUM", True, "CONFIRM", ["video-editing/trim"]))
    # ADR-062: video-editing-skill ROTATE (0.3.0): a clockwise turn (90 / 180 / 270) and / or a mirror
    r.register(SkillSpec("video_rotate", "1.0", "Rotate clockwise by 90 / 180 / 270 degrees and / or mirror (no scaling, no crop)", {"asset": "video", "degrees": "int", "flip": "h|v"},
                         {"artifact": "INTERMEDIATE"}, ["ffmpeg", "ffmpeg-skill", "video-editing"], "LOW", True, "CONFIRM", ["video-editing/rotate"]))
    # ADR-070: video-editing-skill IMAGE_INSERT (0.4.0, #21): a still (title card, end slate) → a silent clip at the picture's frame / rate;
    # the plan joins it before / after the finished picture with a video_concat step (video-editing/concat) on the same subject
    r.register(SkillSpec("video_image_insert", "1.0", "Turn a PNG / JPEG still into a silent clip of a set duration, joined before / after the picture",
                         {"asset": "video", "image": "png|jpg", "duration": "float"}, {"artifact": "INTERMEDIATE"}, ["ffmpeg", "ffmpeg-skill", "video-editing"], "MEDIUM", True,
                         "CONFIRM", ["video-editing/image_insert"]))
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
        # ADR-063: what the pinned contract (0.4.2) already offers and nothing planned before
        ("audio_mix", "MIX", "Sum 2..8 audio inputs (per-input gain / mute); the first input sets the length", {"assets": "audio[]", "levels": "list"}, "MEDIUM"),
        ("audio_noise_reduction", "NOISE_REDUCTION", "FFT noise reduction (strength in dB)", {"asset": "audio", "strength_db": "float"}, "MEDIUM"),
        # ADR-068: audio-production 0.4.3's speech clean-up preset (ffmpeg-skill/audio --voice <level>)
        ("audio_voice_cleanup", "VOICE_CLEANUP", "Speech clean-up preset (light | medium | strong: highpass, de-esser, FFT denoise, compression, limiter)", {"asset": "audio", "level": "light|medium|strong"}, "MEDIUM"),
        ("audio_dynamics", "DYNAMICS", "Gate → compressor → limiter (typed stage parameters)", {"asset": "audio", "stages": "object"}, "MEDIUM"),
        # ADR-072: audio-production 0.4.6's filler removal at caller-supplied word timings (the transcript's word timestamps)
        ("audio_filler_remove", "FILLER_REMOVE", "Remove filler words at the transcript's word timings, matched against a chosen lexicon (ja / en presets, own words)",
         {"asset": "audio", "words": "transcript word timings", "lexicon": "object"}, "MEDIUM"),
    ):
        r.register(SkillSpec(name, "1.0", desc, inputs, {"artifact": "INTERMEDIATE"}, ["ffmpeg", "ffprobe", "ffmpeg-skill", "audio-production", f"audio-production:{typ}"],
                             risk, True, "CONFIRM", ["audio-production/run"]))
    # ADR-054: the programme audio of a take assembly (crossfaded joins, built by audio-production) replaces the hard-cut audio of the
    # assembled picture. ffmpeg-skill/audio --replace stream-copies the video and conforms the new audio to the video's duration.
    r.register(SkillSpec("audio_mux", "1.0", "Replace a video's audio with a programme audio track (video stream-copied, audio conformed to the video duration)",
                         {"asset": "video", "audio": "audio"}, {"artifact": "INTERMEDIATE"}, ["ffmpeg", "ffprobe", "ffmpeg-skill", "encoder:aac"], "LOW", True, "AUTO", ["ffmpeg-skill/audio"]))
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
    # who spoke when (ADR-083, replacing ADR-057's phase-2 declaration of the same name): diarization-skill's anonymous exclusive speaker turns,
    # a measurement (analysis kind speaker_turns) that needs the package and its own DIARIZE readiness (`diarization:DIARIZE`, from its
    # doctor: pyannote.audio, torch and the local pipeline directory). Never a name; nothing is decided from it here.
    r.register(SkillSpec("speaker_diarization", "1.0", "Who spoke when: anonymous exclusive speaker turns (SPEAKER_nn) of a recording, overlapping speech kept apart; measurement only",
                         {"asset": "media"}, {"observation": "speaker_turns", "events": "SPEAKER"}, ["diarization", "diarization:DIARIZE"], "LOW", False, "AUTO", ["diarization/diarize"]))
    # multi-source sync measurement (ADR-035): the offset / drift of a recording relative to the reference, by ffmpeg-skill/sync (audio
    # cross-correlation in the Skill; the agent records the fact and maps the target timeline). Measurement only: no switching, no edit.
    r.register(SkillSpec("sync_analysis", "1.0", "Time offset (and clock drift) of a recording relative to the reference recording, by audio cross-correlation",
                         {"reference": "media", "second": "media"}, {"observation": "sync"}, ["ffmpeg", "ffprobe", "ffmpeg-skill"], "LOW", True, "AUTO", ["ffmpeg-skill/sync"]))
    # the beat grid of a music bed (ADR-053): ffmpeg-skill/scenes --beats, measurement only (no output). Feeds the beat snap of a
    # take assembly; the agent places a join only on a beat this measurement reports as supported.
    r.register(SkillSpec("beat_analysis", "1.0", "Beat grid of a music track (tempo, beat times, confidence, supported grid points); measurement only",
                         {"asset": "media"}, {"observation": "beats"}, ["ffmpeg", "ffprobe", "ffmpeg-skill"], "LOW", True, "AUTO", ["ffmpeg-skill/scenes"]))
    # explicit multi-camera switch (ADR-045): the production side of multi-source work. ffmpeg-skill/multicam aligns every
    # input to the reference by audio (the same correlation as sync_analysis) then cuts between them on the reference
    # timeline per the user's own switch list -- no speaker detection, no automatic cut decision.
    r.register(SkillSpec("camera_switch", "1.0", "Align 2+ cameras/recorders by audio and cut between them per an explicit switch list",
                         {"inputs": "media[]", "switch": "str"}, {"artifact": "video"}, ["ffmpeg", "ffmpeg-skill"], "HIGH", True, "CONFIRM", ["ffmpeg-skill/multicam"]))
    # ---- ADR-046 Priority A batch: five previously-undeclared ffmpeg-skill scripts, wired as real (phase 1) Skills.
    # video_grid is the third, mutually exclusive way to build PROGRAMME from 2+ video inputs (alongside video_concat /
    # camera_switch); the other four are single-source filter-style ops that compose freely on top of whichever one ran.
    r.register(SkillSpec("video_grid", "1.0", "Composite 2+ clips into one COLSxROWS comparison grid (explicit layout, no auto-detection)",
                         {"inputs": "media[]", "cols": "int", "rows": "int"}, {"artifact": "video"}, ["ffmpeg", "ffmpeg-skill"], "MEDIUM", True, "CONFIRM", ["ffmpeg-skill/grid"]))
    r.register(SkillSpec("video_redact", "1.0", "Blur / pixelate a caller-supplied pixel rectangle for the whole clip (privacy redaction); no face/plate detection",
                         {"asset": "video", "x": "float", "y": "float", "width": "int", "height": "int"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill"], "HIGH", True, "CONFIRM", ["ffmpeg-skill/redact"]))
    r.register(SkillSpec("video_deinterlace", "1.0", "Deinterlace (yadif); quality-only but re-encodes the whole file (lossy), so it is CONFIRM not AUTO",
                         {"asset": "video"}, {"artifact": "INTERMEDIATE"}, ["ffmpeg", "ffmpeg-skill"], "LOW", True, "CONFIRM", ["ffmpeg-skill/deinterlace"]))
    # ADR-079: video-editing-skill 0.4.1's CROP (it owns the pixel-rectangle crop now; it calls ffmpeg-skill crop.py itself), not crop.py directly
    r.register(SkillSpec("video_crop", "1.0", "Crop to a caller-supplied exact pixel rectangle; no auto letterbox/aspect detection (that is video_fit/video_fill)",
                         {"asset": "video", "x": "int", "y": "int", "width": "int", "height": "int"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill", "video-editing"], "MEDIUM", True, "CONFIRM", ["video-editing/crop"]))
    r.register(SkillSpec("video_stabilize", "1.0", "Stabilize shaky footage (vidstab, two-pass analysis + render)",
                         {"asset": "video"}, {"artifact": "INTERMEDIATE"}, ["ffmpeg", "ffmpeg-skill"], "MEDIUM", True, "CONFIRM", ["ffmpeg-skill/stabilize"]))
    # cropdetect is deliberately NOT a pipeline edit op (ADR-046): it is a read-only measurement (reports a crop
    # rectangle, writes no file) meant to feed video_crop's x/y/width/height, the same "measure, then a separate
    # explicit op acts on it" shape as sync_analysis feeding camera_switch above. It is not in agent/editing.py's
    # OPERATIONS / EDIT_ORDER; it is the analysis kind `crop_rectangle` (ADR-060), whose stable rectangle becomes a
    # CONFIRM decision proposing video_crop (agent/crop.py).
    r.register(SkillSpec("cropdetect_analysis", "1.0", "Measure black letterbox/pillarbox bars and report the crop rectangle that removes them (measurement only, no output file)",
                         {"asset": "video"}, {"observation": "crop_rectangle"}, ["ffmpeg", "ffmpeg-skill"], "LOW", True, "AUTO", ["ffmpeg-skill/cropdetect"]))
    # ---- Phase 3 finishing Skills (ADR-031 / ADR-032): subtitle-skill replaces the former `caption_generation` declaration (which cited
    # ffmpeg-skill/caption directly); each Skill is reached only through its own package tool and needs the package capability
    r.register(SkillSpec("subtitle_generation", "1.0", "Transcript cues → SRT / WebVTT sidecar (mapped onto the delivered timeline)", {"transcript": "observation"}, {"artifact": "CAPTIONS"},
                         ["subtitle"], "LOW", True, "CONFIRM", ["subtitle/generate"]))
    r.register(SkillSpec("subtitle_burn_in", "1.0", "Burn the subtitle document into the picture (subtitle-skill render → ffmpeg-skill/caption)", {"asset": "video", "captions": "artifact"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffprobe", "ffmpeg-skill", "subtitle", "encoder:libx264", "filter:subtitles"], "MEDIUM", True, "CONFIRM", ["subtitle/render"]))
    # ADR-063: the same render with mode "mux" (subtitle-skill owns it; ffmpeg-skill/caption --mode mux underneath): no re-encode, no subtitles filter
    r.register(SkillSpec("subtitle_mux", "1.0", "Mux the subtitle document into the video as a soft, player-toggleable track (subtitle-skill render mode mux → ffmpeg-skill/caption)",
                         {"asset": "video", "captions": "artifact"}, {"artifact": "INTERMEDIATE"}, ["ffmpeg", "ffprobe", "ffmpeg-skill", "subtitle"], "LOW", True, "CONFIRM", ["subtitle/render"]))
    # ADR-052 (amends ADR-002): the IR's cut list as an editor timeline. ffmpeg-skill/render --export-timeline reads only the
    # sources (probing their durations / rates) and writes a text document; nothing is encoded, so no encoder capability.
    r.register(SkillSpec("timeline_export", "1.0", "The cut list (kept source ranges in programme order) as an editor timeline: OpenTimelineIO, FCPXML or CMX 3600 EDL",
                         {"asset": "video", "format": "otio|fcpxml|edl"}, {"artifact": "TIMELINE"}, ["ffprobe", "ffmpeg-skill"], "LOW", True, "AUTO", ["ffmpeg-skill/render"]))
    r.register(SkillSpec("thumbnail_frame", "1.0", "One video frame at an explicit timestamp as a PNG / JPEG", {"asset": "video", "at": "float"}, {"artifact": "THUMBNAIL"},
                         ["ffmpeg", "ffprobe", "ffmpeg-skill", "thumbnail"], "LOW", True, "CONFIRM", ["thumbnail/extract_frame"]))
    r.register(SkillSpec("thumbnail_render", "1.0", "A video frame with a caption rendered as a thumbnail document", {"asset": "video", "at": "float", "text": "str"}, {"artifact": "THUMBNAIL"},
                         ["ffmpeg", "ffprobe", "ffmpeg-skill", "thumbnail"], "LOW", True, "CONFIRM", ["thumbnail/render"]))
    for name, typ, desc, inputs, risk in (
        ("color_strip_dovi", "STRIP_DOVI", "Strip Dolby Vision side data", {"asset": "video"}, "LOW"),
        ("color_hdr_to_sdr", "HDR_TO_SDR", "Tone-map an HDR source to SDR BT.709", {"asset": "video"}, "MEDIUM"),
        ("color_primary_correction", "PRIMARY_CORRECTION", "Typed primary colour correction: exposure, contrast, saturation, white balance (temperature + tint), gamma, lift / gain, levels, a curves preset", {"asset": "video"}, "MEDIUM"),
        ("color_lut", "LUT_APPLY", "Apply a 3D .cube LUT", {"asset": "video", "lut": "cube"}, "MEDIUM"),
        ("color_retag", "RETAG", "Re-tag the colour metadata (bt709 / bt2020-pq / bt2020-hlg / bt601)", {"asset": "video", "target": "str"}, "LOW"),
    ):
        r.register(SkillSpec(name, "1.0", desc, inputs, {"artifact": "INTERMEDIATE"}, ["ffmpeg", "ffprobe", "ffmpeg-skill", "color-grading", f"color-grading:{typ}"], risk, True, "CONFIRM", ["color-grading/run"]))
    r.register(SkillSpec("motion_graphics", "1.0", "Render the motion-graphics elements (titles, lower thirds, text / image / video overlays, bugs, chapters, progress, countdowns, stickers, hooks, memes, shapes) onto the picture (one request per subject)", {"asset": "video", "elements": "list"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffprobe", "ffmpeg-skill", "motion-graphics"], "MEDIUM", True, "CONFIRM", ["motion-graphics/run"]))
    # ADR-066 (roadmap step 9, replacing ADR-057's phase-2 `media_ingest`): a card folder copied by ingest-skill to the destinations the user
    # named, each copy read back; CONFIRM always (it writes the user's drives), never AUTO. The inventory of the copy becomes the new assets.
    r.register(SkillSpec("media_offload", "1.0", "Verified offload of a camera card / recorder folder to the destinations the user names: checksum read-back copies, "
                         "dual backup, ingest-manifest.json and ASC MHL per destination, clip inventory of the copy", {"source": "card|folder", "destinations": "list"},
                         {"assets": "media", "observation": "ingest_clip"}, ["ingest"], "MEDIUM", True, "CONFIRM", ["ingest/offload"]))
    # ADR-073 (roadmap step 9, voice-enhance-skill's roadmap step 8; replacing ADR-057's phase-2 `voice_enhancement`): one skill per operation,
    # each needing ffmpeg-skill (the Skill's media engine), the package and the Skill's own per-operation readiness (`voice-enhance:<TYPE>`,
    # from its doctor: the engine installed). CONFIRM always: every operation is EXPERIMENTAL in voice-enhance-skill 0.1.0.
    for name, typ, tool, desc, inputs in (
        ("voice_isolate", "ISOLATE", "voice-enhance/isolate", "AI voice isolation (DeepFilterNet3): noise and background removed, a 48 kHz mono stem; EXPERIMENTAL",
         {"asset": "audio", "atten_lim_db": "float"}),
        ("voice_dereverb", "DEREVERB", "voice-enhance/dereverb", "Dereverberation (WPE, nara_wpe): late room reverberation removed at the stream's own rate and channels; EXPERIMENTAL",
         {"asset": "audio", "taps": "int", "delay": "int", "iterations": "int"}),
        ("voice_bleed_reduce", "BLEED_REDUCE", "voice-enhance/bleed-reduce", "Mic-bleed reduction between time-aligned microphone tracks (crosstalk, no weights): one mono track per mic; EXPERIMENTAL",
         {"assets": "audio[]", "dominance_db": "float"}),
    ):
        r.register(SkillSpec(name, "1.0", desc, inputs, {"artifact": "INTERMEDIATE"}, ["ffmpeg-skill", "voice-enhance", f"voice-enhance:{typ}"], "MEDIUM", True, "CONFIRM", [tool]))
    # ADR-077 (imagemagick-skill's roadmap step 4, agent side): an `edit.overlay` image prepared before video-editing composites it -- GPS / EXIF
    # stripped (orientation baked in), resized to the user's box (fit | fill | exact), padded to that canvas -- each a new file, the user's never
    # written. Each needs the package and the Skill's own per-tool readiness (`imagemagick-skill:<TYPE>`, from its doctor's tools.<name>.usable);
    # they run as part of the overlay decision (its approval, CONFIRM by default).
    for name, typ, tool, desc, inputs in (
        ("image_strip", "STRIP", "imagemagick-skill/strip", "Remove GPS / EXIF from an overlay image, baking its EXIF orientation into the pixels (ImageMagick only)",
         {"image": "png|jpg"}),
        ("image_prepare", "RESIZE", "imagemagick-skill/resize", "Resize an overlay image to the user's box: fit (inside, no distortion), fill (cover and centre-crop) or exact",
         {"image": "png|jpg", "width": "int", "height": "int", "mode": "fit|fill|exact"}),
        ("image_pad", "PAD", "imagemagick-skill/pad", "Pad a fitted overlay image to the user's exact canvas with a given colour (or transparency)",
         {"image": "png|jpg", "width": "int", "height": "int", "color": "str", "gravity": "str"}),
    ):
        r.register(SkillSpec(name, "1.0", desc, inputs, {"artifact": "image"}, ["imagemagick-skill", f"imagemagick-skill:{typ}"], "LOW", True, "CONFIRM", [tool]))
    r.register(SkillSpec("qc_check", "1.0", "Quality control report of a deliverable (qc-skill; the final promotion gate)", {"artifact": "delivery|captions"}, {"qa": "qc"},
                         ["ffprobe", "qc"], "LOW", True, "AUTO", ["qc/check"]))
    # ---- ADR-057 (roadmap step 1): every other job of the ecosystem, declared for phase 2 so a plan that asks for it (`job.<skill>`)
    # shows an ADR-051 handoff naming the owning package and its install command (skills/ecosystem.py). No adapter here: never selected.
    # Tool ids are named after the job until an adapter catalogs the package's real tools (roadmap step 9).
    for spec in ECOSYSTEM_JOB_SKILLS:
        r.register(spec)
    r.register(SkillSpec("semantic_deletion", "0.1", "Remove content based on meaning", {"asset": "video", "ranges": "ranges"}, {"artifact": "INTERMEDIATE"},
                         ["ffmpeg", "ffmpeg-skill"], "HIGH", False, "CONFIRM", ["ffmpeg-skill/cut"], phase=4))
    return r
