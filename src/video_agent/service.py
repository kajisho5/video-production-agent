"""Orchestration service: the lifecycle glue used by the CLI (and later by a queue or UI).
Request → Requirements → Intent → Analysis → Inference → Decision → Plan → IR → Validate → Execute → QA → Report."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from .agent import build_plan, decide, extract_requirements, infer, resolve_intent
from .agent.ai_reasoning import AIReasoner, build_request, to_inferences
from .agent import brief as brief_task
from .agent.decision_engine import basis_rows
from .agent.production_plan import plan_status
from .agent.editing import PROGRAMME, delivery_subjects, parse_edit_requirements
from .agent.editing import REQUIREMENT_KEYS as _EDIT_REQUIREMENT_KEYS
from .agent.ingest import IngestRecord
from .agent.handoffs import check_job_keys
from .agent.voice import check_voice_keys
from .agent.requirements import requirement_map
from .artifacts import ArtifactError, ArtifactStore, artifact_id, delivery_name
from .audit import build_provenance, build_receipt, write_audit
from .capabilities import CapabilityResolver
from .context import ProductionContext, build_contexts, contexts_at, contexts_between, infer_from_contexts
from .execution import CompileError, Executor, compile_ir
from .execution.compiler import tool_version_of
from .jobs import Job, JobStore
from .media import MediaAnalyzer
from .temporal import session_for_asset
from .media.analysis import ANALYSIS_KINDS, CORE_KINDS, SYNC_MIN_CONFIDENCE_DEFAULT, AnalysisBudget, AnalysisRequest, normalize_strategy, targeted_kinds
from .models import Artifact, Inference, Request, new_id, now_iso
from .policy.rules import SYSTEM_CONSTRAINTS, Rule, resolve_rules
from .profiles import load_profile
from .providers import AIProvider, AIProviderError, get_provider
from .project import ProjectIR, load_ir, save_ir, validate_ir
from .project.diff import plan_diff
from .project.ir import snapshot_path
from .media.analyzer import AnalysisResult
from .qa import run_qa
from .skills import ProvidesFinding, check_all, default_registry
from .skills.ecosystem import component, install_commands, repository
from .skills.providers import default_providers, explicit_providers
from .tools import ToolRouter
from .tools.ffmpeg_skill import FfmpegSkillAdapter
from .tools.ffmpeg_skill import PACKAGE as FFMPEG_SKILL_PACKAGE
from .tools.ffmpeg_skill.adapter import PathPolicy
from .tools.ffmpeg_skill.locate import locate_ffmpeg_skill
from .tools.media_analysis import PACKAGE as MEDIA_ANALYSIS_PACKAGE, MediaAnalysisAdapter, locate_media_analysis
from .tools.transcription import PACKAGE as TRANSCRIPTION_PACKAGE, TranscriptionAdapter, locate_transcription
from .tools.video_editing import PACKAGE as VIDEO_EDITING_PACKAGE, VideoEditingAdapter, lift_observation, locate_video_editing
from .tools.audio_production import PACKAGE as AUDIO_PRODUCTION_PACKAGE, AudioProductionAdapter, locate_audio_production
from .tools.audio_production import lift_measurement as lift_audio_measurement, lift_observation as lift_audio_observation
from .tools.color_grading import PACKAGE as COLOR_GRADING_PACKAGE, ColorGradingAdapter, lift_observation as lift_color_observation, locate_color_grading
from .tools.motion_graphics import PACKAGE as MOTION_GRAPHICS_PACKAGE, MotionGraphicsAdapter, lift_observation as lift_graphics_observation, locate_motion_graphics
from .tools.qc import PACKAGE as QC_PACKAGE, QcAdapter, lift_report as lift_qc_report, locate_qc
from .tools.subtitle import PACKAGE as SUBTITLE_PACKAGE, SubtitleAdapter, lift_result as lift_subtitle_result, locate_subtitle
from .tools.thumbnail import PACKAGE as THUMBNAIL_PACKAGE, ThumbnailAdapter, lift_observation as lift_thumbnail_observation, locate_thumbnail
from .tools.ingest import PACKAGE as INGEST_PACKAGE, IngestAdapter, locate_ingest
from .tools.imagemagick import PACKAGE as IMAGEMAGICK_PACKAGE, ImageMagickAdapter, lift_observation as lift_image_observation, locate_imagemagick
from .tools.voice_enhance import PACKAGE as VOICE_ENHANCE_PACKAGE, VoiceEnhanceAdapter, lift_observation as lift_voice_observation, locate_voice_enhance
from .tools.ingest import TOOL_INVENTORY as INGEST_INVENTORY, TOOL_OFFLOAD as INGEST_OFFLOAD, TOOL_VERIFY as INGEST_VERIFY
from .agent import offload as offload_task
from .agent.finishing import COLOR_KEYS, ELEMENT_TYPES, MOTION_KEYS, parse_color_requirements, parse_motion_requirements, parse_thumbnail_requirements, qc_requested
from .agent.finishing import QC_REFERENCE_KEYS, parse_qc_reference
from .agent.finishing import THUMBNAIL_KEYS as _THUMBNAIL_REQUIREMENT_KEYS
from .agent.timeline import TIMELINE_KEY, parse_timeline_requirement
from .agent.takes import AI_INTENT as TAKE_AI_INTENT, BEAT_KIND, REQUIREMENT_KEYS as _TAKES_REQUIREMENT_KEYS, parse_takes_requirements
from .agent.subtitles import REQUIREMENT_KEYS as _SUBTITLE_REQUIREMENT_KEYS
from .agent.subtitles import parse_subtitle_requirements
from .agent.qc import WARN_PROMOTION_DEFAULT, WARN_PROMOTION_KEY, qc_verdict_of, stage_for
from .agent.audio import REQUIREMENT_KEYS as _AUDIO_REQUIREMENT_KEYS
from .agent.audio import parse_audio_requirements
from .agent.decision import APPROVAL_KEYS as _DECISION_APPROVAL_KEYS
from .agent.decision_finishing import APPROVAL_KEYS as _FINISHING_APPROVAL_KEYS
from .tools.base import ToolError


class Service:
    def __init__(self, workspace: Optional[str] = None, ffmpeg_skill_dir: Optional[str] = None, adapter=None, caps: Optional[CapabilityResolver] = None,
                 provider: Optional[AIProvider] = None, media_analysis_dir: Optional[str] = None, transcription_dir: Optional[str] = None, offline: bool = False,
                 video_editing_dir: Optional[str] = None, audio_production_dir: Optional[str] = None, subtitle_dir: Optional[str] = None, thumbnail_dir: Optional[str] = None,
                 color_grading_dir: Optional[str] = None, motion_graphics_dir: Optional[str] = None, qc_dir: Optional[str] = None, ingest_dir: Optional[str] = None):
        self.workspace = str(Path(workspace or os.environ.get("VIDEO_AGENT_WORKSPACE") or "./video-agent-work").resolve())
        self.skill_dir = ffmpeg_skill_dir or os.environ.get("VIDEO_AGENT_FFMPEG_SKILL_DIR")
        self.media_analysis_dir = media_analysis_dir or os.environ.get("VIDEO_AGENT_MEDIA_ANALYSIS_DIR")
        self.transcription_dir = transcription_dir or os.environ.get("VIDEO_AGENT_TRANSCRIPTION_DIR")
        self.video_editing_dir = video_editing_dir or os.environ.get("VIDEO_AGENT_VIDEO_EDITING_DIR")
        self.audio_production_dir = audio_production_dir or os.environ.get("VIDEO_AGENT_AUDIO_PRODUCTION_DIR")
        # Phase 3 finishing Skills (ADR-031 / ADR-032): the same rule — a checkout named by an environment variable, never imported
        self.subtitle_dir = subtitle_dir or os.environ.get("VIDEO_AGENT_SUBTITLE_DIR")
        self.thumbnail_dir = thumbnail_dir or os.environ.get("VIDEO_AGENT_THUMBNAIL_DIR")
        self.color_grading_dir = color_grading_dir or os.environ.get("VIDEO_AGENT_COLOR_GRADING_DIR")
        self.motion_graphics_dir = motion_graphics_dir or os.environ.get("VIDEO_AGENT_MOTION_GRAPHICS_DIR")
        self.qc_dir = qc_dir or os.environ.get("VIDEO_AGENT_QC_DIR")
        self.ingest_dir = ingest_dir or os.environ.get("VIDEO_AGENT_INGEST_DIR")   # ADR-066: ingest-skill's card offload (and ADR-059's URL fetch)
        self.voice_enhance_dir = os.environ.get("VIDEO_AGENT_VOICE_ENHANCE_DIR")   # ADR-073: voice-enhance-skill (isolate / dereverb / bleed-reduce)
        self.imagemagick_dir = os.environ.get("VIDEO_AGENT_IMAGEMAGICK_DIR")   # ADR-077: imagemagick-skill (overlay image resize / pad / strip)
        self.offline = bool(offline)   # hard constraint for recognition Skills: no remote engine, no model download (never loosened by a request)
        self.caps = caps or CapabilityResolver(self.skill_dir, media_analysis_dir=self.media_analysis_dir, transcription_dir=self.transcription_dir, offline=self.offline,
                                               video_editing_dir=self.video_editing_dir, audio_production_dir=self.audio_production_dir, subtitle_dir=self.subtitle_dir,
                                               thumbnail_dir=self.thumbnail_dir, color_grading_dir=self.color_grading_dir, motion_graphics_dir=self.motion_graphics_dir, qc_dir=self.qc_dir)
        self._adapter = adapter
        self.registry = default_registry()
        self.provider = provider or get_provider()   # NullProvider unless configured: the pipeline never depends on AI
        self._ai_calls: List[Dict[str, Any]] = []     # provenance.ai_calls for the project being planned
        # Reference Skill package: implemented in this codebase (tools/ffmpeg_skill) whether or not a checkout is installed.
        self.registry.register_package(FFMPEG_SKILL_PACKAGE)
        self.registry.register_package(MEDIA_ANALYSIS_PACKAGE)   # external observation Skill: adapter exists here; availability needs an installation
        self.registry.register_package(TRANSCRIPTION_PACKAGE)    # external recognition Skill (transcription-skill): same rule
        self.registry.register_package(VIDEO_EDITING_PACKAGE)    # external editing Skill (video-editing-skill, ADR-028): same rule
        self.registry.register_package(AUDIO_PRODUCTION_PACKAGE)  # external audio production Skill (audio-production-skill, ADR-030): same rule
        for pkg in (SUBTITLE_PACKAGE, THUMBNAIL_PACKAGE, COLOR_GRADING_PACKAGE, MOTION_GRAPHICS_PACKAGE, QC_PACKAGE):   # Phase 3 finishing Skills (ADR-031 / ADR-032): same rule
            self.registry.register_package(pkg)
        self.registry.register_package(INGEST_PACKAGE)   # ingest-skill's card offload (ADR-066): same rule
        self.registry.register_package(VOICE_ENHANCE_PACKAGE)   # voice-enhance-skill (ADR-073): same rule
        self.registry.register_package(IMAGEMAGICK_PACKAGE)   # imagemagick-skill (ADR-077): same rule
        if self._adapter is not None:
            self.adapter([])   # injected adapters (tests) declare their packages up front

    # ---- adapters / tools
    def adapter(self, allowed_inputs: Optional[List[str]] = None) -> ToolRouter:
        """The tool router with every adapter available in this environment: ffmpeg-skill (media engine) and, when
        installed, media-analysis-skill (observation Skill). Adapters are registered here and nowhere else."""
        if self._adapter is not None:
            return self._sync_packages(self._adapter if isinstance(self._adapter, ToolRouter) else ToolRouter([self._adapter]))
        skill = locate_ffmpeg_skill(self.skill_dir)
        policy = PathPolicy(allowed_inputs or [], self.workspace) if allowed_inputs is not None else None
        router = ToolRouter()
        if skill:   # the Reference Skill/engine: unlike every other adapter below, a broken installation here is not
            router.register(FfmpegSkillAdapter(skill, policy))   # caught -- it is the one package this codebase implements unconditionally (registered in __init__ regardless of a checkout), so failing to construct it is a real error, not a "not installed" case for doctor to report
        # Every other Skill package (ADR-038, Phase 4 item 2: capability-driven registration replacing per-Skill
        # hardcoded branches): (locate_fn, workspace/allowed_inputs, constructor). Adding a new Skill package here
        # means adding one row, not a new branch -- the two `roots` shapes below are real, deliberate differences
        # between Skills (media-analysis / transcription default to no extra root when `allowed_inputs` is None;
        # every editing/finishing Skill defaults to the workspace itself, since its output always lands there),
        # preserved exactly rather than smoothed into a false uniformity.
        ma_roots = (list(allowed_inputs) + [self.workspace]) if allowed_inputs is not None else []
        roots = (list(allowed_inputs) + [self.workspace]) if allowed_inputs is not None else [self.workspace]
        engine = str(skill.root) if skill else None
        for located, make in (
            (locate_media_analysis(self.media_analysis_dir),
             lambda sk: MediaAnalysisAdapter(sk, workspace=self.workspace, allowed_inputs=ma_roots, cache_dir=str(Path(self.workspace) / "cache" / "media-analysis"))),
            (locate_transcription(self.transcription_dir),
             lambda sk: TranscriptionAdapter(sk, workspace=str(Path(self.workspace) / "cache" / "transcription"), allowed_inputs=ma_roots, offline=self.offline)),
            (locate_video_editing(self.video_editing_dir),
             lambda sk: VideoEditingAdapter(sk, workspace=self.workspace, allowed_inputs=roots, ffmpeg_skill_dir=engine, path_policy=policy)),
            (locate_audio_production(self.audio_production_dir),
             lambda sk: AudioProductionAdapter(sk, workspace=self.workspace, allowed_inputs=roots, ffmpeg_skill_dir=engine, path_policy=policy)),
            (locate_subtitle(self.subtitle_dir), lambda sk: SubtitleAdapter(sk, workspace=self.workspace, allowed_inputs=roots, ffmpeg_skill_dir=engine, path_policy=policy)),
            (locate_thumbnail(self.thumbnail_dir), lambda sk: ThumbnailAdapter(sk, workspace=self.workspace, allowed_inputs=roots, ffmpeg_skill_dir=engine, path_policy=policy)),
            (locate_color_grading(self.color_grading_dir), lambda sk: ColorGradingAdapter(sk, workspace=self.workspace, allowed_inputs=roots, ffmpeg_skill_dir=engine, path_policy=policy)),
            (locate_motion_graphics(self.motion_graphics_dir), lambda sk: MotionGraphicsAdapter(sk, workspace=self.workspace, allowed_inputs=roots, ffmpeg_skill_dir=engine, path_policy=policy)),
            (locate_qc(self.qc_dir), lambda sk: QcAdapter(sk, workspace=self.workspace, allowed_inputs=roots, path_policy=policy)),
            (locate_ingest(self.ingest_dir), lambda sk: IngestAdapter(sk)),   # ADR-066: writes only where an approved plan names (checked per call)
            # ADR-073: always told which ffmpeg-skill to use (never its own current-directory search); paths checked per call like the other Skills
            (locate_voice_enhance(self.voice_enhance_dir), lambda sk: VoiceEnhanceAdapter(sk, workspace=self.workspace, allowed_inputs=roots, ffmpeg_skill_dir=engine, path_policy=policy)),
            # ADR-077: its scripts run as `python <checkout>/scripts/<tool>.py` (the current directory is never on their sys.path); paths checked per call
            (locate_imagemagick(self.imagemagick_dir), lambda sk: ImageMagickAdapter(sk, workspace=self.workspace, allowed_inputs=roots, path_policy=policy)),
        ):
            if located:
                try:
                    router.register(make(located))
                except ToolError:
                    pass   # incompatible / broken installation: doctor reports it; the tool stays unavailable rather than half-usable
        return self._sync_packages(router)

    def _sync_packages(self, router: ToolRouter) -> ToolRouter:
        """Every registered adapter declares its Skill package; the registry records it (with the detected version)."""
        for pkg in router.packages():
            self.registry.register_package(pkg)
        return router

    def _providers(self, user_requirements: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, str], Dict[str, str]]:
        """(explicit, default) per-skill provider choices for `SkillRegistry.select_tool()`'s collision policy
        (docs/CAPABILITY_MODEL.md): Tier 1 from `--set provider.<skill>=<package>` in this request's own
        requirements (none, for calls with no request in scope, e.g. `doctor` / `skills`), Tier 2 from this
        workspace's provider policy (OS-level default, overridden by `<workspace>/providers.json`)."""
        return explicit_providers(user_requirements or {}), default_providers(self.workspace)

    def tools_for(self, router: Optional[ToolRouter] = None, user_requirements: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Skill → tool id for this environment (capabilities + registered adapters). `user_requirements`: this
        request's own requirements, consulted only for an explicit `provider.<skill>=<package>` choice."""
        router = router or self.adapter([])
        explicit, default = self._providers(user_requirements)
        return self.registry.resolve_tools(self.caps.resolve(), router.supports, explicit=explicit, default=default)

    def skills(self) -> List[Dict[str, Any]]:
        return self.registry.availability(self.caps.resolve(), self.adapter([]).supports, default=default_providers(self.workspace))

    def handoffs(self, ir: ProjectIR) -> List[Dict[str, Any]]:
        """Plan steps no installed skill can execute here (ADR-051). Each names the job, why no tool was selected and which
        packages would do it, so the chat model the agent runs in can tell the user, install the skill, or do the step
        with what it has -- outside the OS: the agent never runs, records or verifies a handoff as its own work."""
        caps, supports = self.caps.resolve(), self.adapter([]).supports
        out = []
        for st in ir.doc["plan"]["steps"]:
            if st.get("tool") or st.get("skill") not in self.registry.names():
                continue
            spec = self.registry.get(st["skill"])
            _, reason = self.registry.select_tool(spec.name, caps, supports)
            pkgs = sorted({t.split("/", 1)[0] for t in spec.tools})
            # ADR-057: repository and install commands from the ecosystem snapshot (skills/ecosystem.py), not a guessed `-skill` suffix
            install = list(dict.fromkeys(c for p in pkgs for c in install_commands(p)))
            out.append({"step": st["id"], "skill": spec.name, "job": spec.description, "inputs": st.get("inputs", []), "params": st.get("params", {}),
                        "reason": reason, "provided_by": pkgs, "repositories": [repository(p) for p in pkgs],
                        "packages": [component(p).to_dict() if component(p) else {"package": p, "repository": repository(p)} for p in pkgs],
                        "install": install})
        return out

    def packages(self) -> List[Dict[str, Any]]:
        """Skill packages known to this codebase and whether they are usable here (ecosystem view for `video-agent skills`)."""
        router = self.adapter([])
        return self.registry.package_availability(self.caps.resolve(), router.supports, {a.name: str(getattr(a, "version", "")) for a in router.adapters})

    def check_provides(self) -> List[ProvidesFinding]:
        """Read-only provides/Capability consumption diagnostic (`video-agent skills --check-provides`,
        `skills/diagnostics.py`). For every registered Skill package: does its Capability Contract's
        `provides[]` match what this Agent's own package declares, and what a production SkillSpec
        consumes? A package that is not currently installed here (no live contract) reports UNKNOWN,
        never a guess. Purely additive: never calls `SkillRegistry.select_tool()` and never touches the
        execution path."""
        router = self.adapter([])
        contracts = {a.name: getattr(a, "contract", None) for a in router.adapters}
        return check_all(self.registry.packages(), contracts, self.registry.all())

    def _check_provider_requirements(self, user_requirements: Dict[str, Any]) -> None:
        """A `provider.<skill>=<package>` requirement is validated against the registry's own skill names right
        away: an unknown skill name would otherwise resolve to nothing and silently apply to no real skill,
        indistinguishable from a correct choice that just happens not to collide."""
        bad = sorted(name for name in explicit_providers(user_requirements) if name not in self.registry.names())
        if bad:
            raise ValueError(f"unknown skill name(s) in provider.<skill> requirement(s): {', '.join(bad)}; known skills: {', '.join(self.registry.names())}")

    def require_tools(self, tools: Dict[str, str], needed: List[str], router: ToolRouter, user_requirements: Optional[Dict[str, Any]] = None) -> None:
        """Fail early with the registry's reason when a skill this command depends on has no executable tool here."""
        missing = [n for n in needed if n not in tools]
        if missing:
            explicit, default = self._providers(user_requirements)
            reasons = "; ".join(f"{n}: {self.registry.select_tool(n, self.caps.resolve(), router.supports, explicit=explicit.get(n), default=default.get(n))[1]}" for n in missing)
            raise RuntimeError(f"required skill(s) unavailable — {reasons}. Run `video-agent doctor` / `video-agent skills`.")

    def tool_versions(self) -> Dict[str, str]:
        """Adapter name (tool id prefix) → version, for every registered adapter, plus ffmpeg and the agent itself."""
        caps = self.caps.resolve()
        out = {"ffmpeg": caps["ffmpeg"].detail if caps["ffmpeg"].status == "AVAILABLE" else "missing", "video-agent": __version__}
        for a in self.adapter([]).adapters:
            out[a.name] = str(getattr(a, "version", "?"))
        if caps["ffmpeg-skill"].status == "AVAILABLE":
            out["ffmpeg-skill"] = str(caps["ffmpeg-skill"].evidence.get("version", out.get("ffmpeg-skill", "?")))
        return out

    # ---- lifecycle
    def analyze(self, inputs: List[str], profile_name: str = "generic", request_text: str = "", user_requirements: Optional[Dict[str, Any]] = None, hash_sources: bool = True,
                strategy: Optional[str] = None, cache_policy: Optional[str] = None, use_cache: bool = True, kinds: Optional[List[str]] = None,
                params: Optional[Dict[str, Any]] = None, allowed_inputs: Optional[List[str]] = None, ingest_records: Optional[Dict[str, IngestRecord]] = None):
        """params: typed per-kind parameters (e.g. transcript: language / model / offline) merged into the AnalysisRequest;
        allowed_inputs: input roots for the measurement Skills' path policy (default: the inputs' own directories);
        ingest_records: provenance for any input the caller downloaded from a URL (cli.py's `_resolve_url_inputs`,
        issue #50 Task 1), keyed by the resolved local path -- recorded as an OBSERVED Observation on that asset,
        never treated differently by anything below this point."""
        profile = load_profile(profile_name)
        rules = resolve_rules(SYSTEM_CONSTRAINTS + profile.rules + _request_rules(user_requirements or {}))
        adapter = self.adapter(list(allowed_inputs) if allowed_inputs else [str(Path(p).resolve().parent) for p in inputs])
        tools = self.tools_for(adapter, user_requirements)
        self.require_tools(tools, ["media_probe", "silence_analysis", "loudness_analysis"] + [ANALYSIS_KINDS[k]["skill"] for k in (kinds or []) if k in ANALYSIS_KINDS and k not in CORE_KINDS], adapter, user_requirements)
        req = self.analysis_request(inputs, rules, request_text, user_requirements, profile, hash_sources, strategy=strategy, cache_policy=cache_policy, kinds=kinds, params=params)
        analyzer = MediaAnalyzer(adapter, tools=tools, silence_threshold_db=float(rules.get("silence.threshold_db", -40)), hash_sources=hash_sources,
                                 cache_dir=self.workspace if use_cache else None)
        analysis = analyzer.run(req)
        if ingest_records:
            by_path = {str(Path(a.path).resolve()): a for a in analysis.assets}
            for local_path, record in ingest_records.items():
                asset = by_path.get(str(Path(local_path).resolve()))
                if asset:
                    analysis.observations.append(record.observation(asset.id))
        return profile, rules, analysis

    def analysis_request(self, inputs: List[str], rules, request_text: str, user_requirements, profile, hash_sources: bool = True,
                         strategy: Optional[str] = None, cache_policy: Optional[str] = None, kinds: Optional[List[str]] = None, params: Optional[Dict[str, Any]] = None) -> AnalysisRequest:
        """What to observe is decided here (system / policy / requirements), never by an AI provider:
        strategy from `analysis.strategy` (policy or `--set analysis.strategy=`), kinds from the requirements under TARGETED,
        budget from `analysis.budget.*` (only enforceable items; others are refused), parameters from policy."""
        strat = strategy or (user_requirements or {}).get("analysis.strategy") or rules.get("analysis.strategy") or "FULL"
        strat = normalize_strategy(strat)
        reqs = extract_requirements(Request(raw=request_text, args={"inputs": inputs, "profile": profile.name, "requirements": user_requirements or {}}), profile, rules)
        base = targeted_kinds(reqs) if strat == "TARGETED" else list(CORE_KINDS)
        extra = [k for k in (kinds or []) if k not in base]   # explicitly requested measurements (system / user), never chosen by an AI
        kinds = base + extra
        p: Dict[str, Any] = {"threshold_db": float(rules.get("silence.threshold_db", -40)), "min_silence": 0.5, **{k: v for k, v in (params or {}).items() if v is not None}}
        if "sync" in kinds:
            p.setdefault("min_confidence", float(rules.get("sync.min_confidence", SYNC_MIN_CONFIDENCE_DEFAULT)))   # policy or the explicit default (ADR-035)
        return AnalysisRequest(inputs=list(inputs), kinds=kinds, strategy=strat, budget=AnalysisBudget.from_rules(rules),
                               cache_policy=cache_policy or ("only" if strat == "CACHED_ONLY" else "use"), params=p, hash_sources=hash_sources)

    def plan(self, inputs: List[str], profile_name: str = "generic", request_text: str = "", user_requirements: Optional[Dict[str, Any]] = None,
             project_name: Optional[str] = None, hash_sources: bool = True, strategy: Optional[str] = None, use_cache: bool = True,
             kinds: Optional[List[str]] = None, params: Optional[Dict[str, Any]] = None, ingest_records: Optional[Dict[str, IngestRecord]] = None,
             brief: str = "", brief_answer: Any = None, ai_model: str = "host-chat") -> ProjectIR:
        """brief (ADR-061): the user's brief in their own words, read into `--set` keys by the AI provider (or the chat model's
        answer `brief_answer`, host provider) or, with neither, by the rule-based reader; explicit `user_requirements` win."""
        explicit = dict(user_requirements or {})
        _check_requirement_keys(explicit)
        self._check_provider_requirements(explicit)
        self._ai_calls = []   # provenance.ai_calls of this project: the brief's call (if any) and the recommendation call share the budget
        brief_record = self.extract_brief(brief, explicit, profile_name, answer=brief_answer, ai_model=ai_model) if (brief or "").strip() else None
        user_requirements = {**brief_task.values_of(brief_record), **explicit}   # explicit wins (ADR-061)
        if _subtitles_requested(user_requirements or {}) and "transcript" not in (kinds or []):
            kinds = list(kinds or []) + ["transcript"]   # the cues come from a recognised transcript (ADR-031): the measurement a requirement needs, chosen by the system
        if _filler_removal_requested(user_requirements or {}):
            # ADR-072: filler words are removed at recognised word timings: a transcript with word timestamps, chosen by the system for the requirement
            kinds = list(kinds or []) + ([] if "transcript" in (kinds or []) else ["transcript"])
            params = dict(params or {}, word_timestamps=True)
        takes = _takes_requested(user_requirements or {})
        if takes is not None:   # ADR-053: takes are grouped by what is said in them; a named music bed's beat grid is measured on that input only
            kinds = list(kinds or []) + [k for k in ("transcript",) + ((BEAT_KIND,) if takes["music"] is not None else ()) if k not in (kinds or [])]
            if takes["music"] is not None:
                params = dict(params or {}, beat_input=takes["music"])
        profile, rules, analysis = self.analyze(inputs, profile_name, request_text, user_requirements, hash_sources, strategy=strategy, use_cache=use_cache, kinds=kinds, params=params,
                                                ingest_records=ingest_records)
        request = Request(raw=request_text, args={"inputs": inputs, "profile": profile_name, "requirements": explicit, **({"brief": brief_record} if brief_record else {})})
        ir = ProjectIR.new(project_name or Path(inputs[0]).stem, {"name": profile.name, "version": profile.version, "chain": profile.chain}, self.workspace)
        self._fill(ir, request, profile, rules, analysis, plan_version=1)
        return ir

    # ---- card offload (ADR-066): plan → CONFIRM → render → the verified copies become assets
    def plan_offload(self, source: str, user_requirements: Optional[Dict[str, Any]] = None, project_name: Optional[str] = None) -> ProjectIR:
        """The plan of a card offload: what ingest-skill's dry run and the card's inventory say (OBSERVED, nothing written), one TRANSFORM
        `ingest.offload` decision that is always CONFIRM, and one `media_offload` step. Without ingest-skill (or an ingest-skill this agent
        cannot use) the step has no tool and a BLOCK decision names the package; `handoffs()` then says how to install it. Requirements that
        cannot be planned (no destination, a destination that is not a mounted directory, ...) and refusals by ingest-skill's dry run
        (HASH_UNAVAILABLE, REEL_INVALID, ...) raise ValueError before anything is recorded."""
        from .agent.decision_engine import DecisionEngine
        from .models import Asset, Intent, Observation, Requirement
        reqs = dict(user_requirements or {})
        req = offload_task.parse_offload_requirements(source, reqs)
        router = self.adapter([])
        tools = self.tools_for(router)
        tool = tools.get(offload_task.SKILL)
        requirements = [Requirement(key=k, value=reqs[k], provenance="USER", source="cli") for k in sorted(reqs)]
        card = Asset(path=req.source, type="UNKNOWN", provenance="USER", status="source_card", classification={"kind": "card", "provenance": "USER"})
        observations: List[Observation] = []
        version = ""
        if tool:
            version = str(router.version_of(INGEST_OFFLOAD))
            dry = router.measure(INGEST_OFFLOAD, req.args(dry_run=True))
            if not dry.ok:
                err = dry.data.get("error") or {}
                raise ValueError(f"ingest-skill refused the offload ({err.get('code')}/{err.get('kind')}): {err.get('message') or dry.stderr_tail}")
            inv = router.measure(INGEST_INVENTORY, {"target": req.source, **({"reel": req.reel} if req.reel else {})})
            if not inv.ok:
                err = inv.data.get("error") or {}
                raise ValueError(f"ingest-skill could not inventory the card ({err.get('code')}/{err.get('kind')}): {err.get('message') or inv.stderr_tail}")
            observations.append(Observation(kind=offload_task.PLAN_OBS_KIND, asset_id=card.id, source=f"{INGEST_OFFLOAD}@{version}", data=offload_task.plan_summary(dry.data["summary"]),
                                            skill="ingest", skill_version=version, tool=INGEST_OFFLOAD, parameters=req.args(dry_run=True)))
            observations.append(Observation(kind=offload_task.CARD_OBS_KIND, asset_id=card.id, source=f"{INGEST_INVENTORY}@{version}", data=offload_task.card_summary(inv.data["inventory"]),
                                            skill="ingest", skill_version=version, tool=INGEST_INVENTORY, parameters={"target": req.source}))
            roots = observations[0].data["roots"]
        else:
            name = req.reel if (req.name_by == "reel" and req.reel) else Path(req.source).name   # without ingest-skill a reel from the card cannot be read
            roots = [str(Path(d) / req.project / name) if req.project else str(Path(d) / name) for d in req.destinations]
        intent = Intent(primary="ingest", provenance="USER", reason="a card folder with explicit destinations (ingest.dest)")
        eng = DecisionEngine(resolve_rules([]), intent, DecisionEngine.evidence_index(observations=observations, requirements=requirements), requirements)
        params = {k: v for k, v in {"source": req.source, "destinations": req.destinations, "project": req.project, "hash": req.hash, "name_by": req.name_by,
                                     "reel": req.reel, "roots": roots}.items() if v is not None}
        plan_obs = observations[0].data if observations else {}
        what = (f"{plan_obs.get('files')} file(s), {plan_obs.get('bytes_total')} bytes, reel {((plan_obs.get('card') or {}).get('reel') or {}).get('id')}" if observations
                else "size unknown (ingest-skill is not available to read the card)")
        dec = eng.decide(subject=offload_task.SUBJECT, type="TRANSFORM", decision=f"offload {req.source} to {len(req.destinations)} destination(s): " + ", ".join(roots),
                         reason=f"the user named the destination(s) ({offload_task.KEY_DEST}); {what}; every copy is read back ({req.hash}), nothing is overwritten"
                                + ("; " + "; ".join(req.warnings) if req.warnings else ""),
                         evidence=[r.id for r in requirements] + [o.id for o in observations], risk="MEDIUM",
                         approval={"approval": "CONFIRM", "setting": None, "notes": ["fixed CONFIRM: an offload writes the user's drives; never AUTO, no policy key lowers it (ADR-066)"]},
                         confidence=1.0, provenance="USER", params=params, requirements=requirements, serves_intent="ingest")
        if not tool:
            spec = self.registry.get(offload_task.SKILL)
            _, why = self.registry.select_tool(spec.name, self.caps.resolve(), router.supports)
            eng.decide(subject=offload_task.SUBJECT, type="BLOCK", decision="BLOCK: the offload is handed off to ingest-skill",
                       reason=f"no tool for {spec.name} here ({why}); install ingest-skill and plan again, or offload the card with it directly (see the plan's handoff)",
                       evidence=[f"skill:{spec.name}"], risk="HIGH", approval="BLOCK", confidence=1.0, provenance="SYSTEM", params={"skill": spec.name, "missing": ["ingest"]})
        ir = ProjectIR.new(project_name or Path(req.source).name, {"name": "offload", "version": "1", "chain": []}, self.workspace)
        d = ir.doc
        d["project"]["kind"] = offload_task.PROJECT_KIND
        d["request"] = Request(raw="", args={"inputs": [req.source], "profile": "offload", "requirements": reqs}).to_dict()
        d["requirements"] = [r.to_dict() for r in requirements]
        d["source"] = {"agent_version": __version__, "tool_versions": self.tool_versions(), "generator": "video-agent plan (offload)"}
        d["assets"] = {card.id: card.to_dict()}
        d["analysis"]["observations"] = [o.to_dict() for o in observations]
        d["analysis"]["warnings"] = list(req.warnings) + list((observations[1].data.get("warnings") if len(observations) > 1 else []) or [])
        d["intent"] = intent.to_dict()
        d["decisions"] = [x.to_dict() for x in eng.decisions]
        d["plan"] = {"id": new_id("plan"), "project_id": d["project"]["id"], "version": 1, "objective": "verified offload of a card to the destinations the user named",
                     "steps": [{"id": offload_task.STEP_ID, "order": 1, "skill": offload_task.SKILL, "tool": tool, "inputs": [card.id], "params": params, "outputs": [],
                                "depends_on": [], "evidence": [o.id for o in observations], "decision_ids": [dec.id], "decision_id": dec.id, "temporal_scope": None}],
                     "summary": [f"offload {req.source} → {', '.join(roots)} ({req.hash} read-back, ingest-manifest.json + ASC MHL per destination)"] + list(req.warnings)}
        d["qa"]["required"] = []
        d["execution"]["allowed_inputs"] = [req.source]
        d["execution"]["recovery_policy"]["max_attempts"] = 1   # a copy to the user's drives is never re-run automatically
        d["provenance"].update({"source_hashes": {}, "profile_version": "1", "skill_versions": {offload_task.SKILL: self.registry.get(offload_task.SKILL).version},
                                "tool_versions": d["source"]["tool_versions"]})
        ir.refresh_plan_status()
        ir.finalize_hash()
        return ir

    def _execute_offload(self, ir: ProjectIR, ir_path: str, job: Job, store: JobStore, approved: List[str], who: str, timeout: Optional[float]) -> Dict[str, Any]:
        """The approved copy (ingest offload), then the admission of the copies and their inventory as new assets (ADR-066). Never re-run
        automatically; a completed offload is not copied again by a second render (BLOCKED with the earlier job; re-plan the card for that)."""
        from .agent.offload import OP_ID, STEP_ID
        from .models import Operation
        prev = (ir.doc["execution"].get("offload") or {})
        if prev.get("status") == "COMPLETED":
            job.transition("BLOCKED", f"offload already completed in job {prev.get('job_id')}")
            store.save(job)
            return {"job": job.to_dict(), "status": "BLOCKED", "blocked": [], "approved": approved, "offload": prev,
                    "hint": f"this card was offloaded by job {prev.get('job_id')}; plan the card again to copy it once more (ingest-skill skips identical files)"}
        step = next(st for st in ir.doc["plan"]["steps"] if st["id"] == STEP_ID)
        p = step["params"]
        args = {"source": p["source"], "destinations": list(p["destinations"]), "hash": p.get("hash", "sha256"), "name_by": p.get("name_by", "folder"), "dry_run": False}
        args.update({k: p[k] for k in ("project", "reel") if p.get(k)})
        job.transition("EXECUTING")
        store.save(job)
        router = self.adapter([])
        op = Operation(tool=step["tool"], args=args, inputs=[], outputs=[], decision_ids=list(step["decision_ids"]), kind="transform", id=OP_ID)
        ex = Executor(router, max_attempts=1, timeout=timeout)
        result = ex.run([op], {})
        r = result.results[-1] if result.results else None
        data = (r.data if r is not None else {}) or {}
        if data.get("report") is not None:
            (job.dir / "offload-report.json").write_text(json.dumps(data["report"], indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        summary = data.get("summary") or {}
        rec: Dict[str, Any] = {"job_id": job.id, "at": now_iso(), "tool": op.tool, "ingest_version": str(router.version_of(op.tool)), "summary": summary,
                               "report": str(job.dir / "offload-report.json") if data.get("report") is not None else None}
        out: Dict[str, Any] = {"job": None, "status": result.status, "execution": result.to_dict(), "approved": approved, "offload": rec}
        if result.status == "COMPLETED":
            job.transition("QA", "admitting the verified copies")
            try:
                lifted = self._lift_offload(ir, router, data, args, job.id, rec)
            except (ValueError, ToolError) as e:
                rec.update({"status": "FAILED", "error": {"kind": "INVALID_OUTPUT", "message": str(e)}})
                job.transition("FAILED", f"offload copies not admitted: {e}")
                out["status"] = "FAILED"
            else:
                rec.update({"status": "COMPLETED", **lifted})
                job.transition("COMPLETED", f"offloaded to {len(summary.get('destinations') or [])} destination(s); {len(lifted['asset_ids'])} asset(s)")
        else:
            err = (data.get("error") or {})
            rec.update({"status": result.status, "error": {k: err.get(k) for k in ("code", "kind", "message", "retryable", "recovery_class", "exit_code", "details")}})
            # ingest-skill's own re-check of each destination that holds a manifest (reads only): its failure codes per file, for the person who fixes the drive
            checks = []
            for dd in ((err.get("details") or {}).get("destinations") or []):
                if not dd.get("ok") and dd.get("manifest") and Path(str(dd["manifest"])).is_file():
                    v = router.measure(INGEST_VERIFY, {"target": dd["root"]})
                    checks.append({"root": dd["root"], "verified": bool(v.data.get("verified")) if v.ok else None, "failures": v.data.get("failures") if v.ok else [],
                                   "error": None if v.ok else (v.data.get("error") or {}).get("message")})
            rec["verify"] = checks
            out["hint"] = ("fix what the error names (a full or unplugged drive, a conflicting file), then render again: ingest-skill resumes the copy "
                           "(identical files are skipped, nothing is overwritten); destinations already complete stay as they are")
            job.transition(result.status if result.status in ("FAILED", "BLOCKED", "CANCELLED") else "FAILED", f"offload {err.get('code')}/{err.get('kind')}: {str(err.get('message') or '')[:300]}")
        ir.doc["execution"]["offload"] = rec
        ir.doc["provenance"]["runs"].append({"job_id": job.id, "at": now_iso(), "status": out["status"], "who": who, "plan_hash": job.plan_hash, "plan_version": ir.version,
                                             "resumed_from": None, "skipped": []})
        save_ir(ir, ir_path)
        save_ir(ir, str(job.dir / "ir.json"))
        write_audit(str(job.dir / "provenance.json"), {"kind": "offload", "who": who, "at": now_iso(), "plan_hash": job.plan_hash, "plan_version": ir.version,
                                                       "decisions": step["decision_ids"], "reviews": ir.doc["execution"].get("reviews", {}), "offload": rec,
                                                       "results": [x.to_dict() for x in result.results]})
        (job.dir / "report.json").write_text(json.dumps({k: v for k, v in out.items() if k != "job"}, indent=2, default=str) + "\n", encoding="utf-8")
        store.save(job)
        out["job"] = job.to_dict()
        out["report"] = str(job.dir / "report.json")
        return out

    def _lift_offload(self, ir: ProjectIR, router: ToolRouter, data: Dict[str, Any], args: Dict[str, Any], job_id: str, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Admission of a completed offload and the new assets: every destination's ingest-manifest.json lists every file of the report with
        the report's hash and size; the inventory of the first destination's copy (ingest inventory, reads only) names its recordings; each
        member file is under that copy, present, of its manifest size, and -- with sha256 -- of its manifest hash when the agent's own analysis
        hashed it. The agent's analysis of the copies runs when the media engine can read them; otherwise the assets carry ingest-skill's facts."""
        from .models import Asset
        report, summary = data["report"], data["summary"]
        algo = str(report.get("hash"))
        files = {f["path"]: f for f in report.get("files") or [] if isinstance(f, dict)}
        dests = summary.get("destinations") or []
        for dd in dests:
            man = json.loads(Path(str(dd["manifest"])).read_text(encoding="utf-8"))
            entries = {e.get("path"): e for e in man.get("files") or [] if isinstance(e, dict)}
            for path, f in files.items():
                e = entries.get(path)
                if e is None or e.get(algo) != (f.get("hashes") or {}).get(algo) or e.get("size") != f.get("size"):
                    raise ValueError(f"{dd['manifest']} does not list {path} with the reported {algo} and size")
        primary = Path(str(dests[0]["root"]))
        inv_res = router.measure(INGEST_INVENTORY, {"target": str(primary)})
        if not inv_res.ok:
            err = inv_res.data.get("error") or {}
            raise ValueError(f"ingest inventory of {primary} failed ({err.get('code')}/{err.get('kind')}): {err.get('message')}")
        inv = inv_res.data["inventory"]
        rels: List[str] = []
        for c in inv.get("clips") or []:
            for m in c.get("members") or []:
                rel = str(m.get("path"))
                f = (primary / rel).resolve()
                if ".." in Path(rel).parts or not str(f).startswith(str(primary.resolve())) or not f.is_file():
                    raise ValueError(f"inventory names {rel!r}, which is not a file of the copy {primary}")
                if rel not in files or f.stat().st_size != files[rel].get("size"):
                    raise ValueError(f"{rel}: not in the offload report, or its size differs from the copied file")
                rels.append(rel)
        paths = [str((primary / rel).resolve()) for rel in rels]
        warnings: List[str] = []
        assets: List[Any] = []
        observations: List[Any] = []
        if paths:
            try:
                _, _, analysis = self.analyze(paths, "generic", allowed_inputs=[str(primary)], hash_sources=True)
                assets, observations = list(analysis.assets), list(analysis.observations)
                warnings += list(analysis.warnings or [])
            except Exception as e:  # noqa: BLE001 -- the copies are verified whether or not the media engine can read them (e.g. BRAW, no engine)
                warnings.append(f"the agent's analysis of the copies did not run ({str(e)[:300]}); the assets carry ingest-skill's facts only")
                facts = {str((primary / str(m.get('path'))).resolve()): (c, m) for c in inv.get("clips") or [] for m in c.get("members") or []}
                for pth in paths:
                    c, m = facts[pth]
                    h = (m.get("hash") or {}) if isinstance(m.get("hash"), dict) else {}
                    assets.append(Asset(path=pth, type="UNKNOWN", hash=h.get("value") if h.get("algorithm") == "sha256" else None,
                                        technical={"duration": m.get("duration_s"), "size": m.get("size")}, provenance="OBSERVED", status="ingested"))
        by_path = {str(Path(a.path).resolve()): a for a in assets}
        asset_of = {rel: by_path[str((primary / rel).resolve())].id for rel in rels}
        if algo == "sha256":
            for rel in rels:
                a = by_path[str((primary / rel).resolve())]
                want = (files[rel].get("hashes") or {}).get("sha256")
                if a.hash and a.hash != want:
                    raise ValueError(f"{rel}: the agent's sha256 {a.hash} != ingest-skill's {want} (the copy changed after it was verified)")
        for c in inv.get("clips") or []:   # a recording split across files: its members point at one another, in order
            ids = [asset_of[str(m.get("path"))] for m in c.get("members") or []]
            for i, aid in enumerate(ids):
                a = next(x for x in assets if x.id == aid)
                a.status = "ingested"
                a.relationships = list(a.relationships or []) + [{"kind": "spanned_member", "target": t, "metadata": {"clip_id": c.get("clip_id"), "index": j}}
                                                                 for j, t in enumerate(ids) if t != aid]
        version = str(rec.get("ingest_version") or "")
        clip_obs = offload_task.clip_observations(inv, report, summary, asset_of, version, {"job_id": job_id, "offload_report": rec.get("report"),
                                                                                            "started_at": report.get("started_at"), "finished_at": report.get("finished_at")})
        ir.doc["assets"].update({a.id: a.to_dict() for a in assets})
        ir.doc["analysis"]["observations"] += [o.to_dict() for o in observations] + [o.to_dict() for o in clip_obs]
        ir.doc["analysis"]["warnings"] = list(ir.doc["analysis"].get("warnings") or []) + warnings + [f"inventory: {w}" for w in inv.get("warnings") or []]
        ir.doc["provenance"]["source_hashes"].update({a.id: a.hash for a in assets})
        return {"asset_ids": [asset_of[rel] for rel in rels], "paths": paths, "clip_count": inv.get("clip_count"), "spanned_count": inv.get("spanned_count"),
                "reel": (inv.get("card") or {}).get("reel"), "observations": [o.id for o in clip_obs], "warnings": warnings,
                "next": "video-agent plan " + " ".join(paths[:3]) + (" ..." if len(paths) > 3 else "")}

    def extract_brief(self, text: str, explicit: Optional[Dict[str, Any]] = None, profile_name: str = "generic",
                      answer: Any = None, ai_model: str = "host-chat") -> Dict[str, Any]:
        """ADR-061 `requirements_extraction`: brief → validated `--set` keys. Who reads it: the chat model's `answer` (host
        provider) if given, else the configured AI provider if available, else the rule-based reader (agent/brief.parse_brief);
        an AI failure or a host that has not answered yet falls back to the rules with the reason recorded. Whoever read it,
        every key is validated like `--set` (unknown keys / invalid values refused with the reason) and explicit keys win.
        Returns the record stored in the request (`request.args.brief`) and printed by `plan --brief` / `brief`."""
        from .providers.host import HOST_BRIEF_RULES, HostProvider, load_brief_answer
        explicit = dict(explicit or {})
        text = (text or "").strip()[:brief_task.MAX_BRIEF_CHARS]
        if answer is not None:
            answered, answered_unmapped = load_brief_answer(answer)
            provider: AIProvider = HostProvider(requirements=answered, unmapped=answered_unmapped, model=ai_model)
        else:
            provider = self.provider
        request = brief_task.build_request(text, BRIEF_REQUIREMENT_KEYS, explicit)
        record: Dict[str, Any] = {"text": text, "method": "rules", "provider": None, "model": None, "fallback": None, "warnings": []}
        candidates: Optional[List[Dict[str, Any]]] = None
        unmapped: List[Dict[str, Any]] = []
        if isinstance(provider, HostProvider) and provider.requirements is None:
            record["fallback"] = "the chat model has not answered the brief yet (host provider): rule-based reading until it does"
            record["host_request"] = {"task_type": request.task_type, "rules": HOST_BRIEF_RULES, "result_shape": request.schema, "inputs": request.inputs,
                                      "context": request.context, "next": "answer as JSON, then plan again with --brief-requirements <file> (MCP: brief_requirements)"}
        elif provider.available():
            profile = load_profile(profile_name)
            rules = resolve_rules(SYSTEM_CONSTRAINTS + profile.rules + _request_rules(explicit))
            reasoner = AIReasoner(provider, max_calls=int(rules.get("analysis.budget.max_ai_calls", DEFAULT_MAX_AI_CALLS)), calls=self._ai_calls)
            try:
                resp = reasoner.ask(request)
                candidates, unmapped, record["warnings"] = brief_task.candidates_from_response(resp, text)
                record.update(method="ai", provider=resp.provider, model=resp.model)
            except AIProviderError as e:
                record["fallback"] = f"AI provider {provider.name} failed ({e.kind}): rule-based reading"
        else:
            record["fallback"] = "no AI provider configured: rule-based reading"
        if candidates is None:
            candidates, unmapped = brief_task.parse_brief(text)
        source = f"{brief_task.SOURCE_AI}:{record['provider']}/{record['model']}" if record["method"] == "ai" else brief_task.SOURCE_RULES
        provenance = "AI_GENERATED" if record["method"] == "ai" else "USER"
        accepted, refused, overridden = brief_task.validate_candidates(candidates, BRIEF_REQUIREMENT_KEYS, explicit, _check_requirement_keys)
        record.update(accepted=[{**c, "provenance": provenance, "source": source} for c in accepted.values()], refused=refused, overridden=overridden, unmapped=unmapped)
        return record

    def _reason(self, request: Request, profile, rules, analysis: AnalysisResult, suppressed: Optional[List[Dict[str, Any]]] = None, prior_ai=None,
                project_id: str = "", plan_version: int = 1):
        """Requirements → Intent → Inference → Decision → Plan. `suppressed` lists (subject, asset_id) pairs the user rejected:
        the planner must not propose them again. Returns (reqs, intent, inferences, decisions, plan, dropped)."""
        caps = self.caps.resolve()
        router = self.adapter([])
        explicit, default = self._providers((request.args or {}).get("requirements"))
        tools = self.registry.resolve_tools(caps, router.supports, explicit=explicit, default=default)
        reqs = extract_requirements(request, profile, rules)
        rm = requirement_map(reqs)
        intent = resolve_intent(reqs)
        target = rm.get("audio.loudness.target_lufs")
        inferences = infer(analysis, rules, target_lufs=float(target.value) if target else None, tolerance_lu=float(rules.get("audio.loudness.tolerance_lu", 2.0)))
        # situation understanding (ADR-026): events → ProductionContexts (derived, reference-centred) → generic deterministic inferences
        contexts = build_contexts(analysis.timeline.events, analysis.assets, analysis.observations, inferences)
        inferences += infer_from_contexts(contexts, {e.id: e for e in analysis.timeline.events}, {a.id: (a.technical or {}).get("duration") for a in analysis.assets})
        inferences += self._ai_inferences(analysis, rules, prior_ai, request_text=request.raw, intent=intent.primary,
                                          extra_intents=[TAKE_AI_INTENT] if parse_takes_requirements(rm) is not None else [])
        decisions = decide(reqs, intent, analysis, inferences, rules, caps, self.registry, tool_supports=router.supports, explicit_providers=explicit, default_providers=default)
        dropped: List[Dict[str, Any]] = []
        if suppressed:
            keys = {(s["subject"], s.get("asset_id")) for s in suppressed}
            keep = []
            for dcs in decisions:
                if (dcs.subject, dcs.params.get("asset_id")) in keys:
                    dropped.append({"subject": dcs.subject, "asset_id": dcs.params.get("asset_id"), "decision": dcs.decision})
                else:
                    keep.append(dcs)
            decisions = keep
        precision = rm.get("edit.precision")
        plan = build_plan(decisions, analysis, tools=tools, version=plan_version, frame_accurate=bool(precision and precision.value == "frame"), project_id=project_id,
                          constraints=[r.to_dict() for r in rules.all_rules if r.hard], objective=f"{intent.primary} ({profile.name})", inferences=inferences,
                          audio_production=bool(parse_audio_requirements(rm)["production"]), qc_tolerance_lu=float(rules.get("audio.loudness.tolerance_lu", 2.0)))
        return reqs, intent, inferences, decisions, plan, dropped, contexts

    def _ai_inferences(self, analysis: AnalysisResult, rules, prior_ai=None, request_text: str = "", intent: str = "", extra_intents: Optional[List[str]] = None) -> List[Inference]:
        """AI reasoning boundary. Recorded AI inferences (a revision) are reused: revising never spends AI calls. A provider
        failure is an AI-domain incident recorded in analysis.warnings and provenance.ai_calls; the plan stays deterministic.
        The user's own request text and the resolved intent go along as context (ADR-049) so the provider can weigh what
        was asked for; they inform recommendations only -- explicit requirements still come from extract_requirements()."""
        if prior_ai is not None:
            return list(prior_ai)
        if not self.provider.available():
            return []
        intents = [sp.name for sp in self.registry.all() if sp.implemented and ("artifact" in sp.outputs or "qa" in sp.outputs)]
        intents += [x for x in (extra_intents or []) if x not in intents]   # ADR-053: `take_selection` (prefer one take of a line) while takes are being picked
        reasoner = AIReasoner(self.provider, max_calls=int(rules.get("analysis.budget.max_ai_calls", DEFAULT_MAX_AI_CALLS)), calls=self._ai_calls)
        try:
            ctx = {"strategy": analysis.strategy, "intent": intent, "request_text": (request_text or "")[:2000]}
            resp = reasoner.ask(build_request(analysis, intents, ctx))
        except AIProviderError as e:
            analysis.warnings.append(f"ai: provider {self.provider.name} failed ({e.kind}); plan is deterministic only")
            return []
        infs, warns = to_inferences(resp, analysis, intents)
        analysis.warnings.extend(warns)
        return infs

    def _fill(self, ir: ProjectIR, request: Request, profile, rules, analysis: AnalysisResult, plan_version: int, suppressed=None, prior_ai=None):
        reqs, intent, inferences, decisions, plan, dropped, contexts = self._reason(request, profile, rules, analysis, suppressed, prior_ai, project_id=ir.doc["project"]["id"], plan_version=plan_version)
        d = ir.doc
        d["request"] = request.to_dict()
        d["requirements"] = [r.to_dict() for r in reqs]
        d["source"] = {"agent_version": __version__, "tool_versions": self.tool_versions(), "generator": "video-agent plan"}
        d["assets"] = {a.id: a.to_dict() for a in analysis.assets}
        d["analysis"] = {"observations": [o.to_dict() for o in analysis.observations], "inferences": [i.to_dict() for i in inferences], "strategy": analysis.strategy,
                         "budget": {**(analysis.budget or {}), "requested_strategy": rules.get("analysis.strategy")},
                         "warnings": analysis.warnings, "tool_calls": analysis.tool_calls, "analyses": analysis.analyses,
                         "contexts": [c.to_dict() for c in contexts]}   # ProductionContexts: the situation per timeline scope (references only)
        d["intent"] = intent.to_dict()
        d["constraints"] = [r.to_dict() for r in rules.all_rules if r.hard]
        d["policy"] = rules.to_dict()
        d["decisions"] = [x.to_dict() for x in decisions]
        d["plan"] = plan["plan"]   # ProductionPlan (steps / outputs / evidence / constraints / provenance); status derived below
        for a in analysis.assets:   # default temporal grouping: one session per asset (explicit domain object, not a production plan)
            ses = session_for_asset(d["project"]["id"], a, analysis.timeline.events)
            if ses:
                analysis.timeline.add_session(ses)
        d["timeline"] = analysis.timeline.to_dict()
        d["video"] = {"operations": plan["video_ops"]}
        d["audio"] = {"operations": plan["audio_ops"]}
        if plan.get("take_crossfade"):   # ADR-054: the join crossfades of a take assembly; written only when planned
            d["audio"]["take_crossfade"] = plan["take_crossfade"]
        # finishing sections (ADR-031) and the QC gate (ADR-032): written only when planned, so an unchanged request keeps its plan hash
        d["captions"] = {"operations": plan["captions_ops"]} if plan.get("captions_ops") else {}
        d["graphics"] = {"operations": plan["graphics_ops"]} if plan.get("graphics_ops") else {}
        d["color"] = {"operations": plan["color_ops"]} if plan.get("color_ops") else {}
        d["delivery"] = {"targets": plan["delivery"], "naming": profile.data.get("naming", "")}
        if plan.get("timelines"):   # ADR-052: written only when planned, so an unchanged request keeps its plan hash
            d["delivery"]["timelines"] = plan["timelines"]
        d["qa"]["thresholds"]["loudness_tolerance_lu"] = float(rules.get("audio.loudness.tolerance_lu", 2.0))
        if (plan.get("qc") or {}).get("enabled"):
            d["qa"]["qc"] = plan["qc"]
        else:
            d["qa"].pop("qc", None)
        # allowed input roots: the media inputs' directories and, for an overlay / LUT / graphics image or clip / QC reference, the file's directory (all named by the user)
        images = [op["image"] for op in plan["video_ops"] if op.get("type") in ("video.overlay", "video.image_insert") and op.get("image")]   # ADR-070: the joined still too
        images += [op["lut"] for op in plan.get("color_ops") or [] if op.get("lut")] + [op[ref] for op in plan.get("graphics_ops") or [] for ref in ("image", "video") if op.get(ref)] \
            + [op["logo"]["image"] for op in plan.get("graphics_ops") or [] if isinstance(op.get("logo"), dict) and op["logo"].get("image")]   # ADR-085: a thumbnail's logo
        images += [plan["qc"]["reference"]["path"]] if ((plan.get("qc") or {}).get("enabled") and (plan["qc"].get("reference") or {}).get("path")) else []   # ADR-058: the QC master
        d["execution"]["allowed_inputs"] = sorted({str(Path(p).resolve().parent) for p in list((request.args or {}).get("inputs", [])) + images})
        d["execution"]["recovery_policy"]["max_attempts"] = int(rules.get("execution.recovery.max_attempts", 2))
        d["provenance"].update({"source_hashes": {a.id: a.hash for a in analysis.assets}, "profile_version": profile.version,
                                "skill_versions": {s.name: s.version for s in self.registry.all() if s.phase == 1}, "tool_versions": d["source"]["tool_versions"],
                                "ai_calls": list(self._ai_calls), "ai_provider": self.provider.describe() if self.provider.available() else None})
        ir.refresh_plan_status()
        ir.finalize_hash()
        return dropped

    # ---- review workflow
    def approve(self, ir: ProjectIR, ir_path: str, ids: List[str], who: Optional[str] = None, reason: str = "") -> Dict[str, Any]:
        who = who or _default_who()
        if ir.rejected_cited():
            raise ValueError("plan cites REJECTED decisions; run `revise` before approving")
        done = ir.approve(ids, who, reason)
        save_ir(ir, ir_path)
        return {"approved": done, "pending": [d["id"] for d in ir.pending_confirmations()], "plan_version": ir.version,
                "approved_plan_version": ir.doc["revision"]["approved_plan_version"], "renderable": not ir.needs_reapproval() and not ir.pending_confirmations()}

    def reject(self, ir: ProjectIR, ir_path: str, ids: List[str], reason: str, who: Optional[str] = None) -> Dict[str, Any]:
        who = who or _default_who()
        done = ir.reject(ids, who, reason)
        if not done:
            raise ValueError("no decision matched (already rejected, blocked, or unknown id)")
        save_ir(ir, ir_path)
        return {"rejected": done, "reason": reason, "by": who, "plan_version": ir.version, "cited_operations": len(ir.rejected_cited()),
                "hint": "run `video-agent revise <project>` to produce the next plan version without the rejected operations"}

    def revise(self, ir: ProjectIR, ir_path: str, feedback: str = "", user_requirements: Optional[Dict[str, Any]] = None, who: Optional[str] = None,
               ai_recommendations: Optional[List[Dict[str, Any]]] = None, ai_model: str = "host-chat") -> Dict[str, Any]:
        """Plan v(n) → v(n+1): re-plan from the recorded analysis with the rejections and feedback applied. v(n) is snapshotted
        first and never modified; rejected decisions are carried into v(n+1) as history (status REJECTED, no operations).
        ai_recommendations (ADR-050): the answer of the chat model the agent runs in to the brief `plan --ai-brief` returned.
        It replaces the recorded AI inferences and goes through the same validation as any provider's; the recorded analysis
        keeps every asset / observation / event id the brief named, which is why this is a revision and not a new plan."""
        if ir.doc["project"].get("kind") == offload_task.PROJECT_KIND:
            raise ValueError("an offload plan is not revised: reject it, then plan the card again with the destinations you want (ADR-066)")
        who = who or _default_who()
        old = json.loads(json.dumps(ir.doc))
        _check_requirement_keys(user_requirements or {})
        self._check_provider_requirements(user_requirements or {})
        fb_ids: List[str] = []
        duplicate_feedback = False
        if feedback or user_requirements:
            same = [f for f in ir.doc["revision"]["feedback"] if f["plan_version"] == ir.version and f["text"] == (feedback or "") and (f.get("structured") or None) == (user_requirements or None)]
            if same:
                duplicate_feedback = True   # identical feedback on the same version: recorded once, not twice
                fb_ids = [same[-1]["id"]]
            else:
                fb = {"id": new_id("fb"), "plan_version": ir.version, "target": "global", "text": feedback or "", "structured": user_requirements or None, "by": who, "at": now_iso()}
                ir.doc["revision"]["feedback"].append(fb)
                fb_ids.append(fb["id"])
        # merged requirements: previous explicit ones + new structured feedback; free text is appended to the request
        prev_req = dict((old["request"].get("args") or {}).get("requirements") or {})
        prev_req.update(user_requirements or {})
        brief_record = (old["request"].get("args") or {}).get("brief")
        rejected_brief = any(d["status"] == "REJECTED" and d["subject"] == brief_task.SUBJECT for d in old["decisions"])
        if brief_record and rejected_brief and not brief_record.get("rejected"):
            brief_record = {**brief_record, "rejected": True}   # ADR-061: a rejected reading of the brief contributes no keys from v(n+1) on
        raw = (old["request"].get("raw") or "")
        if feedback:
            raw = (raw + "\n" + feedback).strip()
        profile = load_profile(old["project"]["profile"]["name"])
        rules = resolve_rules(SYSTEM_CONSTRAINTS + profile.rules + _request_rules({**brief_task.values_of(brief_record), **prev_req}))
        analysis = AnalysisResult.from_ir(old)
        rejected = [d for d in old["decisions"] if d["status"] == "REJECTED"]
        suppressed = [{"subject": d["subject"], "asset_id": (d.get("params") or {}).get("asset_id")} for d in rejected]
        request = Request(raw=raw, received_at=old["request"]["received_at"], channel=old["request"]["channel"],
                          args={**(old["request"].get("args") or {}), "requirements": prev_req, "revised_at": now_iso(), **({"brief": brief_record} if brief_record else {})})
        new_version = ir.version + 1
        fresh = ProjectIR.new(old["project"]["name"], old["project"]["profile"], self.workspace)
        fresh.doc["project"] = dict(old["project"])   # same project identity: sessions / events keep their deterministic ids across versions
        prior_ai = [Inference.from_dict(i) for i in old["analysis"].get("inferences") or [] if i.get("provenance") == "AI_GENERATED"]
        self._ai_calls = list(old["provenance"].get("ai_calls") or [])
        saved_provider, ai_warn_from = self.provider, len(analysis.warnings)
        if ai_recommendations is not None:
            from .providers.host import HostProvider
            self.provider, prior_ai = HostProvider(recommendations=ai_recommendations, model=ai_model), None
        try:
            dropped = self._fill(fresh, request, profile, rules, analysis, plan_version=new_version, suppressed=suppressed, prior_ai=prior_ai)
        finally:
            self.provider = saved_provider
        ai_report = None
        if ai_recommendations is not None:
            ai_report = {"accepted": [{"id": i["id"], "kind": i["kind"], "asset_id": i["asset_id"], "statement": i["statement"]}
                                      for i in fresh.doc["analysis"]["inferences"] if i.get("provenance") == "AI_GENERATED"],
                         "dropped": [w for w in analysis.warnings[ai_warn_from:] if str(w).startswith("ai:")], "model": ai_model}
        nd = fresh.doc
        # carry identity, history and the audit trail
        nd["project"] = old["project"]
        # ADR-034: the USER_DECISION events of every earlier version travel with the project as review history. Their evidence names
        # the decision of the version they reviewed; that decision is *not* an active v(n+1) decision (only REJECTED ones are carried,
        # nothing is re-approved), so the review records of v(n) are kept verbatim in revision.history[].reviews and the validator
        # resolves a historical citation there — never against the active decision set.
        nd["timeline"]["events"] += [e for e in old["timeline"]["events"] if e["type"] == "USER_DECISION"]
        nd["decisions"] = rejected + nd["decisions"]            # rejected decisions stay visible, with their reviews
        nd["execution"]["reviews"] = {k: v for k, v in (old["execution"].get("reviews") or {}).items() if k in {d["id"] for d in rejected}}
        nd["execution"]["approvals"] = {}
        nd["execution"]["allowed_inputs"] = sorted(set(old["execution"].get("allowed_inputs", [])) | set(nd["execution"].get("allowed_inputs") or []))   # + an overlay image's directory named in this revision
        nd["revision"] = {"feedback": list(ir.doc["revision"]["feedback"]), "history": list(old["revision"]["history"]), "approved_plan_version": None}
        nd["provenance"]["runs"] = list(old["provenance"].get("runs") or [])
        nd["provenance"]["source_hashes"] = old["provenance"]["source_hashes"]
        nd["analysis"] = {**old["analysis"], "inferences": nd["analysis"]["inferences"], "contexts": nd["analysis"].get("contexts") or []}   # same observations / analyses; the re-plan's inferences (AI ones reused) and contexts
        fresh.finalize_hash()
        diff = plan_diff(old, nd)
        applied_before = set(old["revision"]["history"][-1]["rejected_decision_ids"]) if old["revision"]["history"] else set()
        new_rejections = [d for d in rejected if d["id"] not in applied_before]
        for d in new_rejections:
            diff["summary"].insert(0, f"REJECTED {d['subject']}: {d['decision']} — {(old['execution'].get('reviews') or {}).get(d['id'], {}).get('reason', '')}")
        # a new version exists only when the plan actually changed or a new rejection was applied. Feedback that produces no
        # plan change is kept in revision.feedback (once) but never creates an empty version.
        if diff["empty"] and not new_rejections and ai_recommendations is None:   # an AI answer is always recorded as a version (ADR-050)
            if fb_ids and not duplicate_feedback:
                save_ir(ir, ir_path)   # persist the recorded feedback without a version bump
            reason = ("feedback recorded but it changed nothing in the plan" + ("; the proposal it would re-enable was rejected earlier" if dropped else "")) if fb_ids else "no new feedback or rejections since the last revision"
            return {"version": ir.version, "created": False, "diff": diff, "reason": reason, "dropped_proposals": dropped, "feedback_ids": fb_ids,
                    **({"ai": ai_report} if ai_report is not None else {})}
        snap = snapshot_path(ir_path, ir.version)
        if not Path(snap).exists():
            Path(snap).write_text(json.dumps(old, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        nd["revision"]["history"].append({"version": new_version, "from_version": ir.version, "created_at": now_iso(), "by": who, "feedback_ids": fb_ids,
                                          "rejected_decision_ids": [d["id"] for d in rejected], "rejection_reasons": {d["id"]: (old["execution"].get("reviews") or {}).get(d["id"], {}).get("reason", "") for d in rejected},
                                          "dropped_proposals": dropped, "diff": diff, "plan_hash": fresh.plan_hash(), "ir_hash_before": old["provenance"].get("ir_hash"), "snapshot": snap,
                                          "reviews": json.loads(json.dumps(old["execution"].get("reviews") or {}))})   # v(n)'s review records, verbatim (history, not approval)
        ir.doc = nd
        save_ir(ir, ir_path)
        return {"version": new_version, "created": True, "snapshot": snap, "diff": diff, "dropped_proposals": dropped, "pending": [d["id"] for d in ir.pending_confirmations()],
                "hint": "review the diff, then `video-agent approve <project> --decision all` before rendering", **({"ai": ai_report} if ai_report is not None else {})}

    def validate(self, ir: ProjectIR, check_paths: bool = True):
        return validate_ir(ir, self.caps.resolve(), check_paths=check_paths, registry=self.registry, supports=self.adapter([]).supports)

    def dry_run(self, ir: ProjectIR) -> Dict[str, Any]:
        if ir.doc["project"].get("kind") == offload_task.PROJECT_KIND:   # ADR-066: what the approved offload would run (nothing is copied)
            return self._offload_dry_run(ir)
        job_dir = str(Path(self.workspace) / "jobs" / "<job_id>")
        ops, paths = compile_ir(ir, job_dir)
        adapter = self.adapter(ir.doc["execution"].get("allowed_inputs") or None)
        previews = []
        for op in ops:
            try:
                cmd = adapter.preview(op, paths)
            except Exception as exc:  # noqa: BLE001
                cmd = [f"<cannot build command: {exc}>"]
            previews.append({"op": op.id, "tool": op.tool, "kind": op.kind, "args": op.args, "inputs": [paths.get(i, i) for i in op.inputs], "outputs": [paths.get(o, o) for o in op.outputs], "command": cmd, "decisions": op.decision_ids})
        needed = sorted({c for step in ir.doc["plan"]["steps"] for c in self.registry.get(step["skill"]).required_capabilities}) if ir.doc["plan"]["steps"] else []
        caps = self.caps.resolve()
        total = sum((a.get("technical") or {}).get("duration") or 0 for a in ir.doc["assets"].values())
        reencodes = sum(1 for op in ops if op.skill == "delivery_export") + sum(1 for op in ops if (op.skill == "silence_cleanup" and op.args.get("accurate")) or (op.skill == "video_trim" and op.args.get("precision") == "frame"))
        return {"operations": previews, "required_capabilities": {c: caps[c].status if c in caps else "UNKNOWN" for c in needed},
                "expected_outputs": [paths[o] for op in ops for o in op.outputs], "risks": [d["subject"] + ": " + d["risk"] for d in ir.doc["decisions"] if d["risk"] != "LOW"],
                "warnings": self.validate(ir).warnings + ir.doc["analysis"].get("warnings", []), "pending_confirmations": [d["id"] for d in ir.pending_confirmations()], "blocked": [d["id"] for d in ir.blocked()],
                "estimate": {"source_seconds": round(total, 1), "full_reencodes": reencodes, "note": "processing time ≈ source duration × re-encodes on a laptop CPU (see ffmpeg-skill devices.md)"}}

    def _offload_dry_run(self, ir: ProjectIR) -> Dict[str, Any]:
        from .models import Operation
        step = next(st for st in ir.doc["plan"]["steps"] if st["skill"] == offload_task.SKILL)
        p = step["params"]
        args = {"source": p["source"], "destinations": list(p["destinations"]), "hash": p.get("hash", "sha256"), "name_by": p.get("name_by", "folder"), "dry_run": False}
        args.update({k: p[k] for k in ("project", "reel") if p.get(k)})
        router = self.adapter([])
        cmd = router.preview(Operation(tool=step["tool"], args=args, inputs=[], outputs=[], kind="transform", id=offload_task.OP_ID), {}) if step.get("tool") and router.supports(step["tool"]) else ["<no tool: ingest-skill is not available here>"]
        caps = self.caps.resolve()
        plan_obs = next((o["data"] for o in ir.doc["analysis"]["observations"] if o["kind"] == offload_task.PLAN_OBS_KIND), {})
        return {"operations": [{"op": offload_task.OP_ID, "tool": step.get("tool"), "kind": "transform", "args": args, "inputs": [p["source"]], "outputs": list(p.get("roots") or []),
                                "command": cmd, "decisions": step["decision_ids"]}],
                "required_capabilities": {c: caps[c].status if c in caps else "UNKNOWN" for c in self.registry.get(offload_task.SKILL).required_capabilities},
                "expected_outputs": list(p.get("roots") or []), "risks": [d["subject"] + ": " + d["risk"] for d in ir.doc["decisions"] if d["risk"] != "LOW"],
                "warnings": self.validate(ir).warnings + ir.doc["analysis"].get("warnings", []), "pending_confirmations": [d["id"] for d in ir.pending_confirmations()],
                "blocked": [d["id"] for d in ir.blocked()],
                "estimate": {"files": plan_obs.get("files"), "bytes": plan_obs.get("bytes_total"), "note": "the card is read once per file; each destination is written and read back"}}

    def render(self, ir: ProjectIR, ir_path: str, approve: Optional[List[str]] = None, timeout: Optional[float] = None, who: Optional[str] = None,
               resume: Optional[str] = None) -> Dict[str, Any]:
        """resume: a previous job id (or "last") whose completed operations may be reused. The new job stays a new job;
        reuse is decided per operation by the chained idempotency key and by the recorded output still being intact."""
        who = who or _default_who()
        store = JobStore(self.workspace)
        prior: Optional[Job] = None
        resume_note: Optional[Dict[str, Any]] = None
        if resume:
            prior = store.latest(ir_path) if resume == "last" else store.load(resume)
            if prior is None:
                raise FileNotFoundError(f"no previous job to resume for {ir_path}")
            resume_note = {"resumed_from": prior.id, "prior_state": prior.state, "prior_plan_hash": prior.plan_hash, "plan_hash": ir.plan_hash(),
                           "plan_changed": bool(prior.plan_hash) and prior.plan_hash != ir.plan_hash(), "candidate_ops": len(prior.completed_ops)}
        job = store.create()
        job.ir_path = ir_path
        job.plan_hash = ir.plan_hash()
        job.plan_version = ir.version
        if prior is not None:
            job.resumed_from = prior.id
            job.completed_ops = dict(prior.completed_ops)
            ir.doc["execution"]["resume_from"] = prior.id
        job.transition("INGESTING", "render requested")
        job.transition("ANALYZING")
        job.transition("PLANNING")
        if ir.rejected_cited():
            reasons = {d["id"]: (ir.doc["execution"].get("reviews") or {}).get(d["id"], {}).get("reason") for d in ir.rejected()}
            job.transition("BLOCKED", "plan cites REJECTED decisions; revise before rendering")
            store.save(job)
            return {"job": job.to_dict(), "status": "BLOCKED", "rejected": reasons, "approved": [], "hint": "run `video-agent revise <project>`"}
        rep = self.validate(ir)
        if not rep.ok:
            job.transition("FAILED", "validation failed")
            store.save(job)
            return {"job": job.to_dict(), "validation": rep.to_dict(), "status": "FAILED"}
        if approve:
            approved = ir.approve(approve, who)
        else:
            approved = []
        if ir.blocked():
            job.transition("BLOCKED", "; ".join(d["reason"] for d in ir.blocked()))
            store.save(job)
            return {"job": job.to_dict(), "status": "BLOCKED", "blocked": ir.blocked(), "approved": approved}
        pending = ir.pending_confirmations()
        if not pending and ir.needs_reapproval():
            job.transition("WAITING_FOR_APPROVAL", f"plan v{ir.version} has not been approved")
            store.save(job)
            save_ir(ir, ir_path)
            return {"job": job.to_dict(), "status": "WAITING_FOR_APPROVAL", "pending": [], "approved": approved, "plan_version": ir.version,
                    "hint": f"plan v{ir.version} is a revision: approve it with `video-agent approve <project> --decision all` (or render --approve all)"}
        if pending:
            job.transition("WAITING_FOR_APPROVAL", f"{len(pending)} decision(s) need confirmation")
            store.save(job)
            save_ir(ir, ir_path)
            return {"job": job.to_dict(), "status": "WAITING_FOR_APPROVAL", "pending": pending, "approved": approved,
                    "hint": "re-run with --approve <id,...> or --approve all"}
        st = ir.refresh_plan_status()   # the ProductionPlan status is the review state; only APPROVED reaches the compiler
        if st != "APPROVED":
            job.transition("BLOCKED" if st in ("REJECTED", "BLOCKED") else "WAITING_FOR_APPROVAL", f"production plan is {st}")
            store.save(job)
            return {"job": job.to_dict(), "status": "BLOCKED" if st in ("REJECTED", "BLOCKED") else "WAITING_FOR_APPROVAL", "plan_status": st, "approved": approved}
        if ir.doc["project"].get("kind") == offload_task.PROJECT_KIND:   # ADR-066: the approved card offload, not a compiled media pipeline
            return self._execute_offload(ir, ir_path, job, store, approved, who, timeout)
        job.transition("EXECUTING")
        store.save(job)
        ops, paths = compile_ir(ir, str(job.dir))
        adapter = self.adapter(ir.doc["execution"].get("allowed_inputs") or [])
        ex = Executor(adapter, max_attempts=int(ir.doc["execution"]["recovery_policy"]["max_attempts"]), timeout=timeout, completed_keys=job.completed_ops)
        try:
            result = ex.run(ops, paths)
        finally:
            job.completed_ops = ex.completed
            store.save(job)  # whatever happens next, the job file reflects the operations that completed
        out: Dict[str, Any] = {"job": None, "status": result.status, "execution": result.to_dict(), "approved": approved, "paths": paths, "resume": resume_note}
        if result.status != "COMPLETED":
            job.transition(result.status if result.status in ("FAILED", "BLOCKED", "CANCELLED") else "FAILED", f"op {result.failed_op}")
        else:
            job.transition("QA")
            checks = {op.args["input"]: r for op in ops if op.skill == "delivery_check" for r in result.results if r.op_id == op.id}
            qc_results = {op.args["input"]: r for op in ops if op.skill == "qc_check" and op.args.get("input") for r in result.results if r.op_id == op.id}   # the QC gate's reports, per artifact (ADR-032)
            qc_packages = [(op.args, next((r for r in result.results if r.op_id == op.id), None)) for op in ops if op.skill == "qc_check" and op.args.get("artifacts")]   # ADR-065
            qa_reqs = ((ir.doc.get("request") or {}).get("args") or {}).get("requirements")
            qa_tools = self.tools_for(adapter, qa_reqs)
            self.require_tools(qa_tools, ["media_probe", "loudness_analysis", "delivery_check"], adapter, qa_reqs)
            qa = run_qa(adapter, ir.doc, paths, result.results, sheet_dir=str(job.dir / "qa"), check_by_artifact=checks, tools=qa_tools, qc_by_artifact=qc_results, qc_packages=qc_packages)
            out["qa"] = qa.to_dict()
            try:
                artifacts = self._register_artifacts(ir, ir_path, job, ops, paths, qa)
            except ArtifactError as e:
                job.transition("FAILED", f"artifact registration failed: {e}")
                store.save(job)
                out.update({"job": job.to_dict(), "status": "FAILED", "artifact_error": {"kind": e.kind, "message": str(e)}})
                return out
            job.artifacts = [a.to_dict() for a in artifacts]
            out["artifacts"] = job.artifacts
            job.transition("COMPLETED" if qa.status != "FAIL" else "REVIEW", f"qa {qa.status}")
            if qa.status == "FAIL":
                out["status"] = "REVIEW"
        prov = build_provenance(ir.doc, ops, result.results, paths, result.recovery, out.get("qa", {}), who=who)
        prov["skill_observations"] = self._skill_observations(ops, result.results)
        prov["resume"] = resume_note
        prov["skipped"] = result.skipped
        prov["reused"] = result.reused
        prov["plan_version"] = ir.version
        prov["reviews"] = ir.doc["execution"].get("reviews", {})
        ir.doc["provenance"]["runs"].append({"job_id": job.id, "at": now_iso(), "status": out["status"], "who": who, "plan_hash": job.plan_hash, "plan_version": ir.version,
                                             "resumed_from": job.resumed_from, "skipped": result.skipped})
        ir.doc["provenance"]["recovery"] = result.recovery
        save_ir(ir, ir_path)
        save_ir(ir, str(job.dir / "ir.json"))
        write_audit(str(job.dir / "provenance.json"), prov)
        # ProductionReceipt (docs/SPEC.md §6, ADR-040): emitted once execution reaches this point -- any terminal
        # Executor status, "the Plan finished running" per PROVENANCE.md §4, not that it fully passed. Never
        # reached for a Plan that was rejected/blocked/pending approval before execution started (this line is
        # after `ex.run()`, not before it) -- nothing ran yet, so there is nothing to record (I5, resolved).
        receipt_body = build_receipt(ir.doc, result.status, result.failed_op, out.get("artifacts") or [], (out.get("qa") or {}).get("items"))
        receipt_path = str(job.dir / "receipt.json")
        Path(receipt_path).write_text(json.dumps(receipt_body, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        rst = self.artifact_store()
        chk = rst.integrity(receipt_path)
        receipt_artifact = rst.register(Artifact(path=receipt_path, type="PRODUCTION_RECEIPT", hash=chk["sha256"], size=chk["size"],
                                                 id=artifact_id(ir.doc["project"]["id"], ir.doc["plan"].get("id", ""), "production_receipt", chk["sha256"]),
                                                 logical_name="production_receipt", project_id=ir.doc["project"]["id"], plan_id=ir.doc["plan"].get("id", ""),
                                                 plan_version=ir.version, job_id=job.id, jobs=[job.id],
                                                 provenance={"ir_path": ir_path, "plan_hash": ir.doc["provenance"].get("plan_hash"), "ir_hash": ir.doc["provenance"].get("ir_hash"),
                                                             "provenance_path": str(job.dir / "provenance.json"), "receipt_body_id": receipt_body["id"]}))
        out["receipt"] = receipt_body
        out["receipt_artifact_id"] = receipt_artifact.id
        (job.dir / "report.json").write_text(json.dumps({k: v for k, v in out.items() if k != "job"}, indent=2, default=str) + "\n", encoding="utf-8")
        (job.dir / "report.md").write_text(render_report_md(ir.doc, out, job.id), encoding="utf-8")
        store.save(job)
        out["job"] = job.to_dict()
        out["report"] = str(job.dir / "report.md")
        return out

    @staticmethod
    def _skill_observations(ops, results) -> List[Dict[str, Any]]:
        """OBSERVED measurements an editing Skill reported for the outputs it delivered (ADR-028): agent Observation records in
        provenance, keyed by the operation and its output artifact id; never fed back into the IR's analysis."""
        out: List[Dict[str, Any]] = []
        outputs = {o.id: (o.outputs[0] if o.outputs else o.id) for o in ops}
        ops_by_id = {o.id: o for o in ops}
        for r in results:
            if not r.ok:
                continue
            lifted = []
            if str(r.tool).startswith("video-editing/"):
                lifted = [lift_observation(r, outputs.get(r.op_id))]
            elif str(r.tool).startswith("audio-production/"):
                lifted = [lift_audio_observation(r, outputs.get(r.op_id)), lift_audio_measurement(r, outputs.get(r.op_id))]   # probe + the NORMALIZE re-measurement
            elif str(r.tool).startswith("color-grading/"):
                lifted = [lift_color_observation(r, outputs.get(r.op_id))]
            elif str(r.tool).startswith("motion-graphics/"):
                lifted = [lift_graphics_observation(r, outputs.get(r.op_id))]
            elif str(r.tool).startswith("subtitle/"):
                lifted = [lift_subtitle_result(r, outputs.get(r.op_id))]
            elif str(r.tool).startswith("voice-enhance/"):   # ADR-073: engine attribution / licences, parameters, warnings, stem facts
                lifted = [lift_voice_observation(r, outputs.get(r.op_id))]
            elif str(r.tool).startswith("thumbnail/"):
                lifted = [lift_thumbnail_observation(r, outputs.get(r.op_id))]
            elif str(r.tool).startswith("imagemagick-skill/"):   # ADR-077: the prepared overlay image (operation, backend, size, sha256)
                lifted = [lift_image_observation(r, outputs.get(r.op_id))]
            elif str(r.tool).startswith("qc/"):
                qop = ops_by_id.get(r.op_id)
                lifted = [lift_qc_report(r, (str(qop.args.get("input")) if qop is not None and qop.args.get("input") else None) or r.op_id)]
            for obs in lifted:
                if obs is not None:
                    d = obs.to_dict()
                    d["operation"] = r.op_id
                    out.append(d)
        return out

    # ---- artifacts (ADR-022): registration after QA, delivery promotion, archive
    def artifact_store(self) -> ArtifactStore:
        return ArtifactStore(self.workspace)

    def _register_artifacts(self, ir: ProjectIR, ir_path: str, job: Job, ops, paths: Dict[str, str], qa) -> List[Artifact]:
        """Every planned delivery output that the execution produced becomes a registered Artifact: identity from
        (project, plan, logical name, sha256), links to the job / operations / production step / decisions, QA status and
        the initial stage (candidate when QA is not FAIL, working otherwise). A missing or unreadable output is an error."""
        d = ir.doc
        st = self.artifact_store()
        out: List[Artifact] = []
        planned = {o["logical"]: o for o in d["plan"].get("outputs") or []}
        qc_enabled = bool(((d.get("qa") or {}).get("qc") or {}).get("enabled"))
        warn_promotion = str(next((x["params"].get("warn_promotion") for x in d["decisions"] if x["subject"] == "qc.check" and x["status"] != "REJECTED"), WARN_PROMOTION_DEFAULT) or WARN_PROMOTION_DEFAULT).upper()
        rows: List[Dict[str, Any]] = []   # (logical, type, sources, format) of every planned output the execution produced: deliverables, sidecars, thumbnails
        for subject in delivery_subjects(d):   # each asset, or the concat programme made of several sources (ADR-029)
            asset_id = subject["id"]
            for t in d["delivery"]["targets"]:
                logical = f"{asset_id}_delivery_{t['id']}"
                if logical not in paths:
                    # no preset, nothing upstream touched the subject, and no video stream to stream-copy
                    # (a genuinely untouched pure-audio subject on the audio-production path): `compiler.delivery()`
                    # recorded no path for it (an untouched source lives outside the workspace, so there is no
                    # in-workspace file yet to register as an Artifact)
                    continue
                rows.append({"logical": logical, "type": t.get("artifact_type", "MASTER"), "sources": list(subject["sources"]), "format": planned.get(logical, {}).get("format") or t.get("preset") or "", "target": t["id"], "target_decision_ids": list(t.get("decision_ids") or [])})
            for o in d["plan"].get("outputs") or []:   # finishing outputs of the subject (ADR-031): the subtitle sidecar and the thumbnail; the editor timeline (ADR-052)
                if o.get("role") in ("CAPTIONS", "THUMBNAIL", "TIMELINE") and (o.get("expected") or {}).get("source") == asset_id and o["logical"] in paths:
                    rows.append({"logical": o["logical"], "type": o["role"], "sources": list(subject["sources"]), "format": o.get("format") or "", "target": o["role"].lower()})
        for row in rows:
            logical, t = row["logical"], {"id": row["target"], "preset": row["format"], "artifact_type": row["type"]}
            if True:
                path = paths[logical]
                chk = st.integrity(path)
                if not chk["ok"]:
                    raise ArtifactError(chk["error"].split(":")[0], f"{logical}: {chk['error']} ({path})")
                items = [i for i in qa.items if i.artifact == logical]
                statuses = {i.status for i in items}
                art_qa = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else ("PASS" if items else "UNKNOWN"))
                # the gate covers deliverables and sidecars by their own reports (no admitted report → FAIL, never promoted on silence); a thumbnail is
                # covered only as a member of its deliverable set (ADR-065), a timeline not at all
                qc_verdict: Optional[str] = qc_verdict_of(items, gated=row["type"] not in ("THUMBNAIL", "TIMELINE")) if qc_enabled else None
                exp = next((o for o in ops if o.skill in ("delivery_export", "subtitle_generation", "thumbnail_frame", "thumbnail_render", "timeline_export") and logical in o.outputs), None)
                # a no-preset deliverable is an alias (compiler.delivery(): "the last processed intermediate
                # is the deliverable") — no op names `logical` itself, so also credit whichever real op's
                # output resolved to this same on-disk path, or a processed deliverable would misreport as
                # if no operation and no decision had ever touched it
                chain = [o for o in ops if logical in o.outputs or logical in o.inputs or any(paths.get(x) == path for x in o.outputs)]
                step = next((s for s in d["plan"].get("steps") or [] if logical in (s.get("outputs") or [])), None)
                probe = next((m for m in qa.measurements if str(m.get("tool", "")).endswith("/probe") and (m.get("args") or {}).get("inputs") == [path]), None)
                media = {"measured_by": probe.get("tool")} if probe else {}
                media.update({i.name: i.observed for i in items if i.layer in ("video", "audio") and i.kind != "judgement"})   # QA facts (codec / stream / duration …)
                a = Artifact(path=path, type=row["type"], hash=chk["sha256"], source=list(row["sources"]), generation=1,
                             tool=exp.tool if exp else "", tool_version=tool_version_of(d["source"]["tool_versions"], exp.tool) if exp else "",
                             capability_id=exp.skill if exp else "", provider_id=(exp.tool.split("/", 1)[0] if exp and exp.tool else ""),
                             derived_from=[s for s in row["sources"] if st.get(s) is not None],
                             qa_status=art_qa, stage=stage_for(art_qa, qc_verdict, warn_promotion),
                             id=artifact_id(d["project"]["id"], d["plan"].get("id", ""), logical, chk["sha256"]),
                             logical_name=logical, project_id=d["project"]["id"], plan_id=d["plan"].get("id", ""), plan_version=ir.version, job_id=job.id, jobs=[job.id],
                             format=row["format"], size=chk["size"], media=media,
                             name=delivery_name(d["delivery"].get("naming") or "", {"project": d["project"]["name"], "target": t["id"], "version": f"v{ir.version}", "format": row["format"], "date": str(d["project"].get("created_at", ""))[:10]}, Path(path).suffix.lstrip(".")),
                             operations=[o.id for o in chain], step_id=step["id"] if step else None,
                             decision_ids=sorted({x for o in chain for x in o.decision_ids} | set(row.get("target_decision_ids") or [])),
                             qa={"status": art_qa, "pass": sum(1 for i in items if i.status == "PASS"), "warn": sum(1 for i in items if i.status == "WARN"),
                                 "fail": sum(1 for i in items if i.status == "FAIL"), "items": [i.to_dict() for i in items],
                                 **({"qc": qc_verdict, "qc_warn_promotion": warn_promotion} if qc_enabled else {})},
                             provenance={"ir_path": ir_path, "plan_hash": ir.plan_hash(), "ir_hash": d["provenance"].get("ir_hash"), "provenance_path": str(job.dir / "provenance.json"),
                                         "planned": planned.get(logical)})
                out.append(st.register(a))
        return out

    def artifacts(self, ir: Optional[ProjectIR] = None, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        st = self.artifact_store()
        return [a.to_dict() for a in st.list(project_id=ir.doc["project"]["id"] if ir else None, job_id=job_id)]

    def artifact(self, art_id: str) -> Dict[str, Any]:
        st = self.artifact_store()
        a = st.get(art_id)
        if a is None:
            raise ArtifactError("ARTIFACT_MISSING", art_id)
        return {**a.to_dict(), "integrity": st.integrity(a.path, a.hash, a.size)}

    def promote_artifact(self, art_id: str, to: str = "final", who: Optional[str] = None, reason: str = "", channel: str = "local") -> Dict[str, Any]:
        """Delivery = promotion to `final` (recorded, no upload); archive = promotion to `archive`. The plan's current review
        state is read from the artifact's IR so a rejected / blocked plan can never deliver."""
        st = self.artifact_store()
        a = st.get(art_id)
        if a is None:
            raise ArtifactError("ARTIFACT_MISSING", art_id)
        pstatus = None
        ir_path = (a.provenance or {}).get("ir_path")
        if ir_path and os.path.exists(ir_path):
            ir = load_ir(ir_path)
            pstatus = plan_status(ir.doc)
            if ir.doc["plan"].get("id") and ir.doc["plan"]["id"] != a.plan_id and to == "final":
                pstatus = "REVIEW"   # the IR moved on to another plan version: this artifact's plan is no longer the current one
        return st.promote(art_id, to, who or _default_who(), reason, plan_status=pstatus, channel=channel).to_dict()

    def archive_artifact(self, art_id: str, who: Optional[str] = None, reason: str = "") -> Dict[str, Any]:
        return self.promote_artifact(art_id, "archive", who, reason)

    @staticmethod
    def contexts_of(doc: Dict[str, Any]) -> List[ProductionContext]:
        return [ProductionContext.from_dict(c) for c in (doc.get("analysis") or {}).get("contexts") or []]

    @staticmethod
    def explain_context(doc: Dict[str, Any], ctx_id: str) -> Dict[str, Any]:
        """Why does the agent think this situation holds? Context → tracks → events (with their timestamps as recorded) →
        observations → tools; plus the inferences that cite its events and the decisions that rest on those inferences."""
        ctx = next((c for c in (doc.get("analysis") or {}).get("contexts") or [] if c["id"] == ctx_id), None)
        if ctx is None:
            raise KeyError(ctx_id)
        events = {e["id"]: e for e in doc["timeline"].get("events") or []}
        obs = {o["id"]: o for o in doc["analysis"].get("observations") or []}
        infs = {i["id"]: i for i in doc["analysis"].get("inferences") or []}
        chain: List[Dict[str, Any]] = [{"level": 0, "kind": "context", "id": ctx["id"], "provenance": ctx.get("provenance"),
                                        "detail": f"{ctx['timeline_id']} {ctx['scope']['start']}–{ctx['scope']['end']} [{'+'.join(sorted(t['event_type'] + '/' + t['subtype'] for t in ctx['tracks'])) or 'nothing'}]"}]
        for t in ctx["tracks"]:
            chain.append({"level": 1, "kind": "track", "id": f"{t['event_type']}/{t['subtype']}", "detail": f"{len(t['event_ids'])} event(s); sources {', '.join(t.get('sources') or [])}"})
            for eid in t["event_ids"]:
                e = events.get(eid)
                if not e:
                    chain.append({"level": 2, "kind": "reference", "id": eid})
                    continue
                chain.append({"level": 2, "kind": "event", "id": eid, "provenance": e.get("provenance"), "detail": f"{e['type']} {e['range'].get('start')}–{e['range'].get('end')} (as recorded)", "source": e.get("source")})
                for oid in e.get("evidence") or []:
                    o = obs.get(oid)
                    if o:
                        chain.append({"level": 3, "kind": "observation", "id": oid, "provenance": o.get("provenance"), "detail": o.get("kind"), "source": o.get("source")})
        citing = [i for i in infs.values() if set(i.get("evidence") or []) & set(ctx["event_ids"]) or ctx["id"] in ((i.get("data") or {}).get("context_ids") or [])]
        for i in citing:
            chain.append({"level": 1, "kind": "inference", "id": i["id"], "provenance": i.get("provenance"), "detail": i.get("statement"), "generator": (i.get("data") or {}).get("generator")})
        inf_ids = {i["id"] for i in citing}
        for d in doc.get("decisions") or []:
            if set(d.get("evidence") or []) & inf_ids:
                chain.append({"level": 2, "kind": "decision", "id": d["id"], "provenance": d.get("provenance"), "detail": f"{d['subject']}: {d['decision']}", "approval": d.get("approval"), "status": d.get("status")})
        return {"context": ctx, "chain": chain, "boundary": "context → inference → decision only; a context never becomes a step, an operation or a command"}

    @staticmethod
    def explain_decision(doc: Dict[str, Any], ref: str) -> List[Dict[str, Any]]:
        """Why was this decided, on what basis, and what does it move? For each decision matching `ref` (id or subject):
        decision (type / rationale / risk / approval / status / provenance) → basis (policy / preference / constraint values with
        provenance, approval resolution, intent, requirements, risk) → evidence chain (inference → contexts → events → observations
        → asset; requirements; rules) → plan steps and IR operations that cite it (Decision → Plan → Step → IR)."""
        decs = [x for x in doc.get("decisions") or [] if x["id"] == ref or x["subject"] == ref]
        if not decs:
            raise KeyError(ref)
        infs = {i["id"]: i for i in doc["analysis"].get("inferences") or []}
        obs = {o["id"]: o for o in doc["analysis"].get("observations") or []}
        events = {e["id"]: e for e in doc["timeline"].get("events") or []}
        ctxs = {c["id"]: c for c in doc["analysis"].get("contexts") or []}
        reqs = {r["id"]: r for r in doc.get("requirements") or []}
        rules = {r["id"]: r for r in ((doc.get("policy") or {}).get("effective") or {}).values()}
        for c in (doc.get("policy") or {}).get("conflicts") or []:
            rules.setdefault(c["constraint"]["id"], c["constraint"]); rules.setdefault(c["attempted"]["id"], c["attempted"])
        reviews = (doc.get("execution") or {}).get("reviews") or {}
        out: List[Dict[str, Any]] = []
        for d in decs:
            chain: List[Dict[str, Any]] = []
            seen: set = set()

            def walk(eid: str, depth: int) -> None:
                if eid in seen:
                    return
                seen.add(eid)
                if eid in infs:
                    i = infs[eid]
                    chain.append({"level": depth, "kind": "inference", "id": eid, "provenance": i.get("provenance"), "detail": i.get("statement"), "confidence": i.get("confidence")})
                    for cid in (i.get("data") or {}).get("context_ids") or []:
                        c = ctxs.get(cid)
                        if c and cid not in seen:
                            seen.add(cid)
                            chain.append({"level": depth + 1, "kind": "context", "id": cid, "provenance": c.get("provenance"),
                                          "detail": f"{c['timeline_id']} {c['scope']['start']}–{c['scope']['end']} [{'+'.join(sorted(t['event_type'] + '/' + t['subtype'] for t in c['tracks'])) or 'nothing'}]"})
                    for x in i.get("evidence") or []:
                        walk(x, depth + 1)
                elif eid in events:
                    e = events[eid]
                    chain.append({"level": depth, "kind": "event", "id": eid, "provenance": e.get("provenance"), "detail": f"{e.get('event_type')}/{e.get('subtype')} {e['range']}", "source": e.get("source")})
                    for cid, c in ctxs.items():
                        if eid in (c.get("event_ids") or []) and cid not in seen:
                            seen.add(cid)
                            chain.append({"level": depth + 1, "kind": "context", "id": cid, "provenance": c.get("provenance"), "detail": f"{c['scope']['start']}–{c['scope']['end']}"})
                    for x in e.get("evidence") or []:
                        walk(x, depth + 1)
                elif eid in obs:
                    o = obs[eid]
                    chain.append({"level": depth, "kind": "observation", "id": eid, "provenance": o.get("provenance", "OBSERVED"), "detail": o.get("kind"), "source": o.get("source")})
                    a = (doc.get("assets") or {}).get(o.get("asset_id")) or {}
                    if o.get("asset_id") and o["asset_id"] not in seen:
                        seen.add(o["asset_id"])
                        chain.append({"level": depth + 1, "kind": "asset", "id": o["asset_id"], "detail": f"{Path(a.get('path', '')).name} sha256 {(a.get('hash') or '-')[:16]}"})
                elif eid in reqs:
                    r = reqs[eid]
                    chain.append({"level": depth, "kind": "requirement", "id": eid, "provenance": r.get("provenance"), "detail": f"{r['key']} = {json.dumps(r.get('value'), default=str)[:80]}", "source": r.get("source")})
                elif eid in rules:
                    r = rules[eid]
                    chain.append({"level": depth, "kind": str(r.get("kind", "rule")).lower(), "id": eid, "provenance": r.get("scope"), "detail": f"{r['key']} = {json.dumps(r.get('value'), default=str)[:80]}", "source": r.get("source")})
                else:
                    chain.append({"level": depth, "kind": "reference", "id": eid})

            for e in d.get("evidence") or []:
                walk(e, 1)
            steps = [s for s in (doc.get("plan") or {}).get("steps") or [] if d["id"] in (s.get("decision_ids") or [])]
            ops = [{"section": sec, "type": op.get("type") or f"delivery.{op.get('id')}", "asset": op.get("asset"), "keep": op.get("keep"), "target_lufs": op.get("target_lufs"), "preset": op.get("preset")}
                   for sec, key in (("video", "operations"), ("audio", "operations"), ("delivery", "targets")) for op in (doc.get(sec) or {}).get(key) or [] if d["id"] in (op.get("decision_ids") or [])]
            out.append({"decision": {k: d.get(k) for k in ("id", "subject", "type", "decision", "reason", "confidence", "risk", "approval", "status", "provenance", "params", "alternatives")},
                        "basis": basis_rows(d), "review": reviews.get(d["id"]), "evidence": chain,
                        "plan": {"steps": [{"id": s["id"], "skill": s["skill"], "tool": s.get("tool"), "status": s.get("status"), "params": s.get("params")} for s in steps], "operations": ops},
                        "executable": d.get("type") in ("REMOVE", "TRANSFORM", "DELIVER") and d.get("approval") != "BLOCK" and d.get("status") not in ("REJECTED", "BLOCKED"),
                        "boundary": "inference = what is happening; decision = what production should do; plan / IR = how it is executed; no command line or tool argument exists at this layer"})
        return out

    @staticmethod
    def explain_observation(doc: Dict[str, Any], obs_id: str) -> Dict[str, Any]:
        """Why does this observation exist? Observation → Skill package → tool → (engine → model for a transcript) → asset identity
        (fingerprint) → analysis request → the events derived from it. Facts only: no inference or decision is part of this chain."""
        o = next((x for x in doc["analysis"]["observations"] if x["id"] == obs_id or x.get("external_id") == obs_id), None)
        if o is None:
            raise KeyError(obs_id)
        asset = doc["assets"].get(o["asset_id"]) or {}
        analysis = next((a for a in doc["analysis"].get("analyses") or [] if a.get("analysis_id") == o.get("analysis_id")), None)
        row = next((r for r in (analysis or {}).get("rows", []) if r.get("kind") == o["kind"] and r.get("asset_id") == o["asset_id"]), None)
        chain = [{"level": 0, "kind": "observation", "id": o["id"], "provenance": o.get("provenance"), "detail": f"{o['kind']} observed_at {o.get('observed_at')}", "source": o.get("source")},
                 {"level": 1, "kind": "skill", "id": o.get("skill") or o["source"].split("/", 1)[0], "detail": f"version {o.get('skill_version') or '-'}", "source": o.get("source")},
                 {"level": 2, "kind": "tool", "id": o.get("tool") or o["source"].split("@", 1)[0], "detail": f"external id {o.get('external_id') or '-'}; cache {json.dumps(o.get('cache') or {}, sort_keys=True)}"}]
        params = o.get("parameters") or {}
        if o["kind"] == "transcript":
            tr = o.get("data") or {}
            prov = tr.get("provenance") or {}
            chain.append({"level": 3, "kind": "engine", "id": f"{tr.get('engine')}@{tr.get('engine_version')}", "detail": f"execution_mode {prov.get('execution_mode')}; requires no interpretation"})
            chain.append({"level": 4, "kind": "model", "id": str(prov.get("model")), "detail": f"model_version {prov.get('model_version') or '-'}; parameters {json.dumps(prov.get('parameters') or {}, sort_keys=True, ensure_ascii=False)}"})
            chain.append({"level": 5, "kind": "transcript", "id": str(tr.get("id")), "detail": f"language {tr.get('language')} ({tr.get('language_source')}), {len(tr.get('segments') or [])} segment(s), speaker_id always null (no diarization)"})
        elif params:
            chain.append({"level": 3, "kind": "parameters", "id": "-", "detail": json.dumps(params, sort_keys=True, ensure_ascii=False)})
        chain.append({"level": 6, "kind": "asset", "id": o["asset_id"], "detail": f"{Path(asset.get('path', '')).name} sha256 {(asset.get('hash') or '-')[:16]}; observation fingerprint {(o.get('fingerprint') or '-')[:16]}",
                      "shared_identity": bool(asset.get("hash")) and (o.get("fingerprint") or "") == asset.get("hash")})
        if analysis:
            chain.append({"level": 7, "kind": "analysis", "id": analysis["analysis_id"], "detail": f"{analysis['request']['strategy']} by {analysis.get('analyzer')}; row {json.dumps({k: row.get(k) for k in ('status', 'cache_hit', 'cache_owner')}, sort_keys=True) if row else '-'}"})
        events = [e for e in doc["timeline"]["events"] if o["id"] in (e.get("evidence") or [])]
        ev_ids = {e["id"] for e in events}
        for e in events:
            chain.append({"level": 8, "kind": "event", "id": e["id"], "provenance": e.get("provenance"), "detail": f"{e['type']} {e['range'].get('start')}–{e['range'].get('end')}", "source": e.get("source")})
        for c in (doc.get("analysis") or {}).get("contexts") or []:
            if ev_ids & set(c.get("event_ids") or []):
                chain.append({"level": 9, "kind": "context", "id": c["id"], "provenance": c.get("provenance"), "detail": f"{c['scope']['start']}–{c['scope']['end']} [{'+'.join(sorted(t['event_type'] + '/' + t['subtype'] for t in c['tracks']))}]"})
        return {"observation": o, "chain": chain, "events": [e["id"] for e in events], "asset": {"id": o["asset_id"], "path": asset.get("path"), "hash": asset.get("hash")},
                "boundary": "observation → event only; no inference, decision, plan step or command derives from this chain"}

    def explain_artifact(self, art_id: str) -> Dict[str, Any]:
        """Artifact → job → operations → production step → decisions → inferences → events → observations."""
        from .agent.production_plan import explain_step
        a = self.artifact(art_id)
        chain: Dict[str, Any] = {"artifact": a, "jobs": a["jobs"], "operations": [], "step": None}
        pp = (a.get("provenance") or {}).get("provenance_path")
        if pp and os.path.exists(pp):
            prov = json.loads(Path(pp).read_text(encoding="utf-8"))
            chain["operations"] = [e for e in prov.get("operations", []) if a["path"] in (e.get("output") or []) or (e.get("skill") == "delivery_check" and a["path"] in (e.get("input") or []))]
        ir_path = (a.get("provenance") or {}).get("ir_path")
        if ir_path and os.path.exists(ir_path) and a.get("step_id"):
            ir = load_ir(ir_path)
            if ir.doc["plan"].get("id") == a["plan_id"]:
                try:
                    chain["step"] = explain_step(ir.doc, a["step_id"])
                except KeyError:
                    chain["step"] = None
            else:
                snap = snapshot_path(ir_path, a["plan_version"])
                if snap and os.path.exists(snap):
                    try:
                        chain["step"] = explain_step(load_ir(snap).doc, a["step_id"])
                    except KeyError:
                        chain["step"] = None
        return chain

    def check(self, path: str, platform: str = "custom", qc: bool = False) -> Dict[str, Any]:
        """probe + platform compliance of a file; with `qc` also the qc-skill report (kind delivery for a video file, audio otherwise),
        admitted only when its fingerprint is the file's sha256 as this agent computes it (ADR-032)."""
        adapter = self.adapter([str(Path(path).resolve().parent)])
        tools = self.tools_for(adapter)
        self.require_tools(tools, ["media_probe", "delivery_check"], adapter)
        pr = adapter.measure(tools["media_probe"], {"inputs": [str(Path(path).resolve())]})
        ck = adapter.measure(tools["delivery_check"], {"input": str(Path(path).resolve()), "platform": platform})
        out: Dict[str, Any] = {"probe": pr.data if pr.ok else {"error": pr.stderr_tail}, "check": ck.data if ck.data else {"error": ck.stderr_tail}}
        if qc:
            from .agent.qc import admit
            from .tools.skill_process import sha256_file
            self.require_tools(tools, ["qc_check"], adapter)
            facts = pr.data.get("observation", {}).get("data", pr.data) if pr.ok and isinstance(pr.data, dict) else {}
            kind = "delivery" if (facts or {}).get("video") else "audio"
            rules = {"delivery": {"require_video": True}} if kind == "delivery" else {"audio": {"require_audio_stream": True}}
            r = adapter.measure(tools["qc_check"], {"input": str(Path(path).resolve()), "kind": kind, "rules": rules, "cache_policy": "bypass"})
            problems = admit(r.data, sha256_file(str(Path(path).resolve())), kind) if r.ok else [f"{(r.data.get('error') or {}).get('code')}: {(r.data.get('error') or {}).get('message')}"]
            out["qc"] = {"admitted": not problems, "problems": problems, "verdict": r.data.get("verdict") if r.ok and not problems else None,
                         "checks": r.data.get("checks") if r.ok else [], "findings": r.data.get("findings") if r.ok else [], "report_id": r.data.get("report_id") if r.ok else None}
        return out

    @staticmethod
    def explain_pipeline(doc: Dict[str, Any], job: Optional[Dict[str, Any]] = None, provenance: Optional[Dict[str, Any]] = None, artifacts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """The whole causal chain of a project, level by level (ADR-031): Request → Requirements → Observations → Events → Inferences →
        Decisions → ProductionPlan steps → IR operations → Skill → Capability → Tool → Execution (the job's operations) → QA → Artifacts.
        Every row carries the ids it rests on, so any link can be followed with the specific `explain --decision / --step / --artifact`."""
        req = doc.get("request") or {}
        rows: List[Dict[str, Any]] = [{"level": "request", "id": "request", "detail": (req.get("raw") or "")[:200], "args": {k: v for k, v in (req.get("args") or {}).items() if k != "inputs"},
                                       "inputs": [Path(p).name for p in (req.get("args") or {}).get("inputs") or []]}]
        for r in doc.get("requirements") or []:
            rows.append({"level": "requirement", "id": r["id"], "detail": f"{r['key']} = {json.dumps(r.get('value'), default=str)[:80]}", "provenance": r.get("provenance"), "source": r.get("source")})
        for o in (doc.get("analysis") or {}).get("observations") or []:
            rows.append({"level": "observation", "id": o["id"], "detail": f"{o.get('kind')} on {o.get('asset_id')}", "source": o.get("source"), "skill": o.get("skill"), "tool": o.get("tool"), "provenance": o.get("provenance")})
        for e in (doc.get("timeline") or {}).get("events") or []:
            rows.append({"level": "event", "id": e["id"], "detail": f"{e.get('type')} {e.get('range', {}).get('start')}–{e.get('range', {}).get('end')} on {e.get('timeline_id')}", "evidence": list(e.get("evidence") or []), "provenance": e.get("provenance") or e.get("kind")})
        for i in (doc.get("analysis") or {}).get("inferences") or []:
            rows.append({"level": "inference", "id": i["id"], "detail": i.get("statement"), "evidence": list(i.get("evidence") or []), "provenance": i.get("provenance"), "confidence": i.get("confidence")})
        for x in doc.get("decisions") or []:
            rows.append({"level": "decision", "id": x["id"], "detail": f"{x['subject']}: {x['decision']}", "type": x.get("type"), "approval": x.get("approval"), "status": x.get("status"),
                         "evidence": list(x.get("evidence") or []), "provenance": x.get("provenance")})
        steps = (doc.get("plan") or {}).get("steps") or []
        for s in steps:
            rows.append({"level": "step", "id": s["id"], "detail": f"{s['skill']} → {s.get('tool') or '-'}", "skill": s["skill"], "tool": s.get("tool"), "package": (s.get("tool") or "").split("/", 1)[0] or None,
                         "decisions": list(s.get("decision_ids") or []), "depends_on": list(s.get("depends_on") or []), "inputs": list(s.get("inputs") or []), "outputs": list(s.get("outputs") or []), "status": s.get("status")})
        for sec in ("video", "audio", "color", "graphics", "captions"):
            for op in (doc.get(sec) or {}).get("operations") or []:
                rows.append({"level": "operation", "id": f"{sec}.{op.get('type')}@{op.get('asset')}", "detail": f"{op.get('type')} on {op.get('asset')}", "section": sec, "decisions": list(op.get("decision_ids") or []),
                             "input": op.get("input") or op.get("inputs"), "output": op.get("output")})
        for t in (doc.get("delivery") or {}).get("targets") or []:
            rows.append({"level": "operation", "id": f"delivery.{t['id']}", "detail": f"deliver {t['id']} ({t.get('preset') or 'source'} / {t.get('platform')})", "section": "delivery", "decisions": list(t.get("decision_ids") or [])})
        qc = (doc.get("qa") or {}).get("qc") or {}
        if qc.get("enabled"):
            rows.append({"level": "operation", "id": "qa.qc", "detail": f"QC gate on {', '.join(sorted(qc.get('subjects') or {}))} (+ {len(qc.get('sidecars') or {})} sidecar(s))", "section": "qa", "decisions": list(qc.get("decision_ids") or [])})
        for s in steps:   # Skill → Capability → Tool: what the step needs and what package executes it (from the registry vocabulary recorded in the plan)
            rows.append({"level": "skill", "id": s["skill"], "detail": f"registry skill {s['skill']} selected {s.get('tool') or 'no tool'}", "step": s["id"], "tool": s.get("tool")})
        versions = (doc.get("source") or {}).get("tool_versions") or {}
        for pkg in sorted({(s.get("tool") or "").split("/", 1)[0] for s in steps if s.get("tool")}):
            rows.append({"level": "capability", "id": pkg, "detail": f"package {pkg} {versions.get(pkg, '?')}", "version": versions.get(pkg)})
        for tool in sorted({s.get("tool") for s in steps if s.get("tool")}):
            rows.append({"level": "tool", "id": str(tool), "detail": f"tool {tool} ({str(tool).split('/', 1)[0]} {versions.get(str(tool).split('/', 1)[0], '?')})"})
        for e in (provenance or {}).get("operations") or []:
            res = e.get("result") or {}
            rows.append({"level": "execution", "id": e.get("idempotency_key") or e.get("what"), "detail": f"{e.get('what')} {'ok' if res.get('ok') else 'FAILED' if res else 'not run'} ({res.get('seconds', 0)} s, {res.get('attempts', 0)} attempt(s))",
                         "skill": e.get("skill"), "tool": e.get("tool"), "tool_version": e.get("tool_version"), "decisions": list(e.get("decision") or []), "inputs": e.get("input"), "outputs": e.get("output"),
                         "skill_result": {k: v for k, v in (e.get("skill_result") or {}).items() if k in ("status", "operation_id", "artifact")}})
        for i in ((provenance or {}).get("qa") or {}).get("items") or []:
            rows.append({"level": "qa", "id": f"{i.get('layer')}.{i.get('name')}@{i.get('artifact')}", "detail": f"{i.get('status')} observed {json.dumps(i.get('observed'), default=str)[:60]} expected {json.dumps(i.get('expected'), default=str)[:60]}", "artifact": i.get("artifact"), "status": i.get("status")})
        for a in artifacts or []:
            rows.append({"level": "artifact", "id": a.get("id"), "detail": f"{a.get('logical_name')} {a.get('type')} sha256 {(a.get('hash') or '')[:16]} qa {a.get('qa_status')} stage {a.get('stage')}", "step": a.get("step_id"),
                         "decisions": list(a.get("decision_ids") or []), "qc": (a.get("qa") or {}).get("qc"), "stage": a.get("stage")})
        levels = ["request", "requirement", "observation", "event", "inference", "decision", "step", "operation", "skill", "capability", "tool", "execution", "qa", "artifact"]
        return {"levels": levels, "rows": rows, "counts": {lv: sum(1 for r in rows if r["level"] == lv) for lv in levels}, "job": (job or {}).get("id"),
                "boundary": "request → requirements → observations → events → inferences → decisions → plan steps → IR operations → skill → capability → tool → execution → QA → artifacts; "
                            "no level generates a command: tools are named by the registry, arguments by the compiler, commands by the Skills"}


DEFAULT_MAX_AI_CALLS = 4   # per project; policy key analysis.budget.max_ai_calls
REQUIREMENT_PREFIXES = ("edit.", "audio.", "silence.", "delivery.", "analysis.", "subtitle", "thumbnail", "color.", "motion.", "qc", "provider.", "sync.", "job.", "voice.")
# keys that match a REQUIREMENT_PREFIXES prefix but are never a real per-request lever: never-write-the-source is
# policy/rules.py's hardcoded, non-overridable sys.preserve_source CONSTRAINT (ADR-022's workspace boundary,
# structurally enforced by ArtifactStore.check_path() regardless of any requirement), not something a request can
# turn on or off. Rejected explicitly rather than silently accepted-and-ignored, since silently accepting a
# `--set` that looks like it disables a safety guarantee is worse than rejecting an unknown key.
DEAD_REQUIREMENT_KEYS = {"delivery.preserve_source"}
_FINISHING_PREFIXES = ("edit.", "audio.", "subtitle", "thumbnail", "color.", "motion.", "qc")
# Two genuinely different key surfaces share these 7 prefixes, and only the first was ever validated:
# (1) "switch" keys, read via `m.get()`/the Requirement map inside parse_edit_requirements() and its siblings below (plus
#     a handful of pre-existing keys read directly in agent/decision.py, agent/intent.py and analysis_request() above --
#     edit.trim_leading_silence / edit.trim_trailing_silence / edit.precision / audio.normalize /
#     audio.loudness.target_lufs / audio.loudness.true_peak predate ADR-029/030's own REQUIREMENT_KEYS constants). An
#     unrecognised key here is the severe failure: the whole feature silently never happens (found via real-data testing
#     with `--set subtitle.generate=true` -- the real switch is bare `subtitle`; `subtitle.generate` only names the
#     internal decision subject -- which produced no decision, no error and no subtitles, plan still APPROVED).
# (2) "policy default" keys, read via `resolve_setting(rules, "<key>", default)` / `rules.get(...)`, sourced from
#     `_request_rules()` below, which -- by design, and unlike this function -- accepts *any* key matching these 7
#     prefixes with zero validation and just lets resolve_setting() fall back to its default on a miss. This covers
#     every `<subject>.approval` key (decision.py / decision_finishing.py APPROVAL_KEYS' values, not their dict keys --
#     those are internal decision subjects, e.g. "video.concat" or "subtitle.generate", never a requirement key
#     themselves) plus thumbnail.at_ratio/qc.warn.promotion/subtitle.format's own default and the motion element
#     start/duration defaults (keyed by the internal ELEMENT_TYPES name, e.g. "motion.text_overlay.start" -- distinct
#     from the real user-facing "motion.text.start" switch key in MOTION_KEYS; a separate, minor naming inconsistency,
#     not fixed here). This surface intentionally tolerates an unrecognised key (it just keeps the default), so it must
#     never be rejected as unknown -- only enumerated here so it is not mistaken for (1) and rejected by mistake.
_MOTION_DEFAULT_KEYS = {f"motion.{typ}.{field}" for typ in ELEMENT_TYPES for field in ("start", "duration")}
_APPROVAL_REQUIREMENT_KEYS = {v[0] for v in list(_DECISION_APPROVAL_KEYS.values()) + list(_FINISHING_APPROVAL_KEYS.values()) if v[0].startswith(_FINISHING_PREFIXES)}
_POLICY_DEFAULT_KEYS = {"audio.loudness.tolerance_lu", "thumbnail.at_ratio", WARN_PROMOTION_KEY} | _MOTION_DEFAULT_KEYS | _APPROVAL_REQUIREMENT_KEYS
_SWITCH_REQUIREMENT_KEYS = frozenset(   # surface (1): what a request asks for
    {k for keys in _EDIT_REQUIREMENT_KEYS.values() for k in keys}
    | {"edit.trim_leading_silence", "edit.trim_trailing_silence", "edit.precision"}
    | set(_AUDIO_REQUIREMENT_KEYS)
    | {"audio.normalize", "audio.loudness.target_lufs", "audio.loudness.true_peak"}
    | set(_SUBTITLE_REQUIREMENT_KEYS)
    | set(COLOR_KEYS)
    | {k for keys in MOTION_KEYS.values() for k in keys}
    | set(_THUMBNAIL_REQUIREMENT_KEYS)
    | {"qc"}
    | set(QC_REFERENCE_KEYS)   # ADR-058: qc.reference_video / qc.reference_metrics / qc.min_*
    | set(_TAKES_REQUIREMENT_KEYS)
)
_FINISHING_REQUIREMENT_KEYS = _SWITCH_REQUIREMENT_KEYS | _POLICY_DEFAULT_KEYS
# ADR-061: the keys a brief (`plan --brief`) may yield, derived from the --set vocabulary above rather than listed by hand: every key
# `_check_edit_requirements` recognises as a switch / refinement (surface (1)), plus delivery.timeline -- so not (1') the policy-default
# surface (approval policies, qc.warn.promotion, tolerance / ratio defaults: a brief never sets how much review a decision gets) -- minus (2) keys whose value
# is a file path (overlay / joined still (ADR-070) / LUT / graphics image or clip / QC master: only the user names a file the agent may read) and (3) job.* and
# provider.* (handoffs and tool choice stay explicit). Everything else `--set` takes, a brief may yield.
_PATH_REQUIREMENT_KEYS = frozenset({"edit.overlay", "edit.image_insert", "color.lut", "motion.image", "motion.video", "qc.reference_video"})
BRIEF_REQUIREMENT_KEYS = frozenset((_SWITCH_REQUIREMENT_KEYS - _PATH_REQUIREMENT_KEYS) | {TIMELINE_KEY})


def _check_requirement_keys(user_requirements: Dict[str, Any]) -> None:
    """The --set gate shared by plan / revise / a brief's keys: a known prefix, not a dead key, then the per-key checks."""
    if offload_task.has_offload_keys(user_requirements or {}):   # ADR-066: the offload of a card folder is its own plan
        raise ValueError(f"{', '.join(sorted(k for k in user_requirements if k.startswith(offload_task.PREFIX)))}: ingest.* keys plan the offload of a card folder; "
                         "give the card folder as the only input (plan the edit on the verified copies afterwards)")
    bad = [k for k in (user_requirements or {}) if not k.startswith(REQUIREMENT_PREFIXES) or k in DEAD_REQUIREMENT_KEYS]
    if bad:
        raise ValueError(f"unknown requirement key(s): {', '.join(bad)}; allowed prefixes: {', '.join(REQUIREMENT_PREFIXES)}")
    _check_edit_requirements(user_requirements or {})


_SWITCH_OR_AUTO = (True, False, "auto")
_CORE_VALUE_CHECKS = {   # the six keys older than ADR-029's REQUIREMENT_KEYS constants, read directly by decision.py / intent.py
    "edit.trim_leading_silence": lambda v: v in _SWITCH_OR_AUTO, "edit.trim_trailing_silence": lambda v: v in _SWITCH_OR_AUTO,
    "audio.normalize": lambda v: v in _SWITCH_OR_AUTO, "edit.precision": lambda v: v in ("frame", "keyframe-ok"),
    "audio.loudness.target_lufs": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and -70.0 <= float(v) <= -5.0,
    "audio.loudness.true_peak": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and -9.0 <= float(v) <= 0.0,
}
_CORE_VALUE_HINTS = {"edit.trim_leading_silence": "true / false / auto", "edit.trim_trailing_silence": "true / false / auto", "audio.normalize": "true / false / auto",
                     "edit.precision": "frame / keyframe-ok", "audio.loudness.target_lufs": "a number of LUFS from -70 to -5", "audio.loudness.true_peak": "a number of dBTP from -9 to 0"}


def _check_core_values(user_requirements: Dict[str, Any]) -> None:
    """ADR-061: a value one of these keys cannot use is refused before analysis, not read as `nothing asked` (or a float() crash)."""
    for k, ok in _CORE_VALUE_CHECKS.items():
        if k in user_requirements and not ok(user_requirements[k]):
            raise ValueError(f"{k} must be {_CORE_VALUE_HINTS[k]}, got {user_requirements[k]!r}")


def _check_edit_requirements(user_requirements: Dict[str, Any]) -> None:
    """Explicit `edit.*` requirements are range-checked before any analysis runs (an invalid value is a planning error, not a
    guess and not something a later stage corrects). A key matching one of these 7 namespaces but recognised by neither the
    switch vocabulary nor the policy-default vocabulary above is refused here too -- silently doing nothing is worse than an
    error naming the real keys (`docs/decisions.md` ADR-042)."""
    from .models import Requirement
    keys = [k for k in user_requirements if k.startswith(_FINISHING_PREFIXES)]
    unknown = sorted(k for k in keys if k not in _FINISHING_REQUIREMENT_KEYS)
    unknown += sorted(k for k in user_requirements if k.startswith(TIMELINE_KEY) and k != TIMELINE_KEY)   # ADR-052: e.g. delivery.timeline.format is not a key
    if unknown:
        raise ValueError(f"unknown requirement key(s): {', '.join(unknown)} (matches a recognised namespace but is not a real key there)")
    _check_core_values(user_requirements)
    reqs = [Requirement(key=k, value=user_requirements[k], provenance="USER", source="cli") for k in keys]
    rm = requirement_map(reqs)
    parse_edit_requirements(rm)
    parse_audio_requirements(rm)
    parse_subtitle_requirements(rm)
    parse_color_requirements(rm)
    parse_motion_requirements(rm)
    parse_thumbnail_requirements(rm)
    qc_requested(rm)
    parse_qc_reference(rm)   # ADR-058: the master file exists, metrics are named, thresholds in range, the qc gate is on
    parse_takes_requirements(rm)   # ADR-053: edit.takes / its refinements, range-checked before any analysis runs
    check_job_keys(user_requirements)   # ADR-057: job.<skill> names a declared ecosystem job and is a plain switch
    check_voice_keys(user_requirements)   # ADR-073: voice.* keys known, in range, with audio.production (and audio.mix for bleed reduction)
    if TIMELINE_KEY in user_requirements:   # ADR-052: none | otio | fcpxml | edl, refused before any analysis runs
        parse_timeline_requirement(requirement_map([Requirement(key=TIMELINE_KEY, value=user_requirements[TIMELINE_KEY], provenance="USER", source="cli")]))


def _takes_requested(user_requirements: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from .models import Requirement
    reqs = [Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in user_requirements.items() if k.startswith("edit.takes")]
    return parse_takes_requirements(requirement_map(reqs))


def _filler_removal_requested(user_requirements: Dict[str, Any]) -> bool:
    """ADR-072: `audio.filler_remove` set to anything but false (the value itself was range-checked by `_check_requirement_keys`)."""
    v = (user_requirements or {}).get("audio.filler_remove")
    return v is not None and v is not False and not (isinstance(v, str) and v.strip().lower() == "false")


def _subtitles_requested(user_requirements: Dict[str, Any]) -> bool:
    from .models import Requirement
    reqs = [Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in user_requirements.items() if k.startswith("subtitle")]
    return bool(parse_subtitle_requirements(requirement_map(reqs))["enabled"])


def _default_who() -> str:
    try:
        import getpass
        return f"user:{getpass.getuser()}"
    except Exception:  # noqa: BLE001
        return "user"


def _request_rules(user_requirements: Dict[str, Any]) -> List[Rule]:
    out = []
    for k, v in user_requirements.items():
        if k.startswith(("audio.", "silence.", "delivery.", "edit.", "subtitle", "thumbnail", "color.", "motion.", "qc", "sync.")):
            out.append(Rule(f"request.{k}", "PREFERENCE", "REQUEST", k, v, "request"))
    return out


def render_report_md(doc: Dict[str, Any], out: Dict[str, Any], job_id: str) -> str:
    lines = [f"# Production report — {doc['project']['name']} (job {job_id})", "", f"Status: **{out['status']}**  profile: {doc['project']['profile']['name']} v{doc['project']['profile']['version']}", "",
             "## Plan", *[f"- {s}" for s in doc["plan"]["summary"]], "", "## Decisions"]
    for d in doc["decisions"]:
        lines.append(f"- [{d['approval']}/{d['risk']}/{d['status']}] **{d['subject']}** → {d['decision']}  \n  {d['reason']} (confidence {d['confidence']:.2f}, evidence {len(d['evidence'])})")
    qa = out.get("qa")
    if qa:
        lines += ["", f"## QA — {qa['status']}"]
        for i in qa["items"]:
            if i["status"] != "PASS":
                lines.append(f"- {i['status']} {i['layer']}/{i['name']}: {i['observed']} (expected {i['expected']}) {i.get('fix_hint', '')}")
        lines.append(f"- {sum(1 for i in qa['items'] if i['status'] == 'PASS')} checks passed, {len(qa['incidents'])} incident(s)")
        for inc in qa["incidents"]:
            lines.append(f"  - {inc['type']} [{inc['severity']}] {inc['possible_cause']} → {inc['recommended_action']}")
        for s in qa.get("sheets", []):
            lines.append(f"- Look: {s}")
    if out.get("artifacts"):
        lines += ["", "## Artifacts"]
        for a in out["artifacts"]:
            lines.append(f"- {a['type']} {a['path']} (stage {a['stage']}, qa {a['qa_status']}, sha256 {a['hash'][:12]}…)")
    ex = out.get("execution") or {}
    if out.get("resume"):
        r = out["resume"]
        lines += ["", "## Resume", f"- resumed from job {r['resumed_from']} ({r['prior_state']}), plan {'changed' if r['plan_changed'] else 'unchanged'}, reused {len(ex.get('skipped', []))} operation(s)"]
        for op_id, path in (ex.get("reused") or {}).items():
            lines.append(f"  - {op_id} ← {path}")
    if ex.get("recovery"):
        lines += ["", "## Recovery", *[f"- op {r['op']} attempt {r.get('attempt')}: {r['class']} → {r['action']} ({r['reason']})" for r in ex["recovery"]]]
    return "\n".join(lines) + "\n"
