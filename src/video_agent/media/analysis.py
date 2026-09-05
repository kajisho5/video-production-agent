"""Observation / Analysis architecture (ADR-019).

    Asset → AnalysisRequest → Analyzer → Observation → AnalysisResult (evidence) → Inference → Decision

- AnalysisKind: the measurements this codebase can actually run today (nothing declared "for later").
- AnalysisRequest: what to observe (inputs, kinds, strategy FULL / TARGETED / CACHED_ONLY, budget, cache policy, params).
- Analyzer contract: deterministic, tool-backed observation producer. It never calls an AI provider, never makes a
  decision, never builds a Project IR, never takes a command / argv from anyone; it measures through the registry-selected
  tools (ToolAdapter.measure) only.
- Observation validation: an analyzer result is stored only if it is a well-formed measurement (asset / kind / source /
  analysis id match, structured data, no credential or command material).
- ObservationCache: deterministic key = asset fingerprint + kind + analyzer id@version + tool id@version + parameters.
- AnalysisBudget: limits this code can actually enforce (calls, wall-clock seconds); anything else is UNSUPPORTED, never
  a silent no-op. Separate from the AI call budget (agent/ai_reasoning.py).
- AnalysisError kinds: the analysis failure domain, distinct from AIProviderError and from media-engine incidents.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..models import Observation, new_id, stable_hash

# ---- kinds (only what is implemented: each maps to a measurement production skill in the registry)
ANALYSIS_KINDS: Dict[str, Dict[str, Any]] = {
    "media_probe": {"skill": "media_probe", "needs_audio": False, "params": ()},
    "silence": {"skill": "silence_analysis", "needs_audio": True, "params": ("threshold_db", "min_silence")},
    "loudness": {"skill": "loudness_analysis", "needs_audio": True, "params": ()},
    # measurements only an external observation Skill provides (media-analysis-skill); parameters are the Skill's own
    "stream_layout": {"skill": "stream_layout_analysis", "needs_audio": False, "params": ()},
    "video_format": {"skill": "video_format_analysis", "needs_audio": False, "params": ("stream",)},
    "audio_format": {"skill": "audio_format_analysis", "needs_audio": True, "params": ("stream",)},
    "duration": {"skill": "duration_analysis", "needs_audio": False, "params": ()},
    "integrity": {"skill": "integrity_analysis", "needs_audio": False, "params": ("max_error_lines",)},
    "scene_detection": {"skill": "scene_analysis", "needs_audio": False, "params": ("threshold", "min_scene_duration")},
    "timing": {"skill": "timing_analysis", "needs_audio": False, "params": ("gap_factor", "av_mismatch_tolerance")},
}
CORE_KINDS = ("media_probe", "silence", "loudness")   # what FULL runs by default; other kinds are requested explicitly
STRATEGIES = ("FULL", "TARGETED", "CACHED_ONLY")
# names recorded in the Project IR (schema enum) ↔ request strategy
IR_STRATEGY = {"FULL": "FULL_ANALYSIS", "TARGETED": "TARGETED_ANALYSIS", "CACHED_ONLY": "CACHED_ONLY"}
LEGACY_STRATEGY = {"FULL_ANALYSIS": "FULL", "COARSE_ANALYSIS": "FULL", "TARGETED_ANALYSIS": "TARGETED", "CACHED_ONLY": "CACHED_ONLY"}
CACHE_POLICIES = ("use", "bypass", "only")

FAILURE_KINDS = ("ANALYZER_UNAVAILABLE", "ANALYZER_TIMEOUT", "ANALYSIS_BUDGET_EXCEEDED", "ANALYSIS_CACHE_INVALID", "ANALYSIS_INVALID_RESULT", "ANALYSIS_UNSUPPORTED")

SOURCE_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[^\s@]+$")   # "<tool>@<version>" where tool = "<package>/<name>"
_LEAK_KEYS = re.compile(r"(api[_-]?key|secret|token|password|credential|authorization|argv|command|cmd|shell)", re.I)
_LEAK_VALUES = re.compile(r"(^sk-[A-Za-z0-9]|^[A-Za-z0-9_./-]+\s+-{1,2}[A-Za-z]|^(sh|bash|cmd|powershell)\b|\$\(|`|;\s*rm\s)", re.I)   # keys, command lines, shell material


class AnalysisError(Exception):
    """Analysis failure domain. Never an AI failure, never an engine incident."""

    def __init__(self, kind: str, message: str = ""):
        if kind not in FAILURE_KINDS:
            kind = "ANALYSIS_INVALID_RESULT"
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind = kind


def normalize_strategy(value: Optional[str]) -> str:
    v = str(value or "FULL").upper()
    v = LEGACY_STRATEGY.get(v, v)
    if v not in STRATEGIES:
        raise AnalysisError("ANALYSIS_UNSUPPORTED", f"unknown analysis strategy {value!r}; allowed: {', '.join(STRATEGIES)}")
    return v


# ---- budget: only what is enforced here
@dataclass
class AnalysisBudget:
    """Limits the analyzer enforces before each tool call: number of measurement calls and wall-clock seconds spent in the
    analysis (a call already running is bounded by the tool timeout, not by this budget). Anything else that a policy names
    (max_bytes_scanned, max_duration_s, max_probe_calls …) is UNSUPPORTED and rejected instead of silently ignored."""
    max_analysis_calls: Optional[int] = None
    max_total_seconds: Optional[float] = None

    SUPPORTED = ("max_analysis_calls", "max_total_seconds")
    ALIASES = {"max_processing_time": "max_total_seconds"}   # legacy policy key, now enforced as wall-clock seconds

    @classmethod
    def from_rules(cls, rules: Any) -> "AnalysisBudget":
        """Read `analysis.budget.*` policy keys. Unsupported keys raise ANALYSIS_UNSUPPORTED (no fake budgets)."""
        b = cls()
        for rule in getattr(rules, "all_rules", []):
            key = getattr(rule, "key", "")
            if not key.startswith("analysis.budget."):
                continue
            name = cls.ALIASES.get(key[len("analysis.budget."):], key[len("analysis.budget."):])
            if name not in cls.SUPPORTED:
                raise AnalysisError("ANALYSIS_UNSUPPORTED", f"analysis budget {key} is not enforceable by this version (supported: {', '.join(cls.SUPPORTED)})")
            val = rules.get(key)
            if val is not None:
                setattr(b, name, int(val) if name == "max_analysis_calls" else float(val))
        return b

    def to_dict(self) -> Dict[str, Any]:
        return {"max_analysis_calls": self.max_analysis_calls, "max_total_seconds": self.max_total_seconds}


@dataclass
class AnalysisRequest:
    inputs: List[str]                                   # media paths (assets are identified by the analyzer's probe)
    kinds: List[str] = field(default_factory=lambda: list(CORE_KINDS))
    strategy: str = "FULL"
    budget: AnalysisBudget = field(default_factory=AnalysisBudget)
    cache_policy: str = "use"                           # use | bypass | only
    params: Dict[str, Any] = field(default_factory=dict)   # kind parameters, e.g. threshold_db / min_silence
    hash_sources: bool = True
    analysis_id: str = field(default_factory=lambda: new_id("ana"))

    def __post_init__(self) -> None:
        self.strategy = normalize_strategy(self.strategy)
        if self.strategy == "CACHED_ONLY":
            self.cache_policy = "only"
        if self.cache_policy not in CACHE_POLICIES:
            raise AnalysisError("ANALYSIS_UNSUPPORTED", f"unknown cache policy {self.cache_policy!r}")
        unknown = [k for k in self.kinds if k not in ANALYSIS_KINDS]
        if unknown:
            raise AnalysisError("ANALYSIS_UNSUPPORTED", "unsupported analysis kind(s): " + ", ".join(unknown) + f" (implemented: {', '.join(ANALYSIS_KINDS)})")
        if not self.inputs:
            raise AnalysisError("ANALYSIS_INVALID_RESULT", "an analysis request needs at least one input")
        if "media_probe" not in self.kinds:
            self.kinds = ["media_probe"] + list(self.kinds)   # every other kind depends on the probe (asset identity, audio presence)

    def kind_params(self, kind: str) -> Dict[str, Any]:
        return {k: self.params[k] for k in ANALYSIS_KINDS[kind]["params"] if k in self.params}

    def to_dict(self) -> Dict[str, Any]:
        return {"analysis_id": self.analysis_id, "inputs": [Path(p).name for p in self.inputs], "kinds": list(self.kinds), "strategy": self.strategy,
                "budget": self.budget.to_dict(), "cache_policy": self.cache_policy, "params": dict(self.params), "hash_sources": self.hash_sources}


def _num(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def loudness_facts(data: Dict[str, Any]) -> Dict[str, Any]:
    """One vocabulary for a loudness fact whichever measurement tool produced it. The Observation keeps the tool's own
    keys (facts as measured); consumers (inference, QA) read through this view.
      agent shape       : silent / lufs / true_peak / lra
      loudnorm-style raw : silent / input_i / input_tp / input_lra
      media-analysis    : integrated_lufs / true_peak_dbtp / loudness_range_lu / unmeasurable / integrated_below_absolute_gate
    `silent` is True only when the tool measured no programme level (a `silent` flag, or an integrated loudness that is
    unmeasurable or below the BS.1770 absolute gate)."""
    d = data or {}
    lufs = _num(d.get("lufs", d.get("input_i", d.get("integrated_lufs"))))
    tp = _num(d.get("true_peak", d.get("input_tp", d.get("true_peak_dbtp"))))
    lra = _num(d.get("lra", d.get("input_lra", d.get("loudness_range_lu", d.get("loudness_range")))))
    silent = bool(d.get("silent")) or ("integrated_lufs" in d and (lufs is None or bool(d.get("integrated_below_absolute_gate"))))
    return {"silent": silent, "lufs": None if silent else lufs, "true_peak": tp, "lra": lra}


def probe_facts(probe: Dict[str, Any]) -> Dict[str, Any]:
    """Asset technical facts from a probe observation, whichever measurement tool produced it: the agent's own keys
    (format / duration / size_bytes / bitrate / video / audio) or a container-based layout (container{duration, size, bitrate, format})."""
    if isinstance(probe.get("container"), dict):
        c = probe["container"]
        return {"format": c.get("format"), "duration": c.get("duration"), "size_bytes": c.get("size"), "bitrate": c.get("bitrate"),
                "video": probe.get("video"), "audio": probe.get("audio"), "subtitle_streams": probe.get("subtitle_streams", 0)}
    return {k: probe.get(k) for k in ("format", "duration", "size_bytes", "bitrate", "video", "audio", "subtitle_streams")}


def targeted_kinds(requirements: Iterable[Any]) -> List[str]:
    """TARGETED strategy: the system decides what to observe from the requirements (never the AI)."""
    rm = {getattr(r, "key", None): getattr(r, "value", None) for r in requirements}
    on = lambda k: rm.get(k) in (True, "auto", "true", "yes")   # noqa: E731
    kinds = ["media_probe"]
    if on("edit.trim_leading_silence") or on("edit.trim_trailing_silence"):
        kinds.append("silence")
    if on("audio.normalize"):
        kinds.append("loudness")
    return kinds


# ---- observation validation
def validate_observation(obs: Any, request: AnalysisRequest, asset_ids: Iterable[str], kind: Optional[str] = None) -> List[str]:
    """Errors for an analyzer result. An observation with any error is never stored or used as evidence."""
    errs: List[str] = []
    if not isinstance(obs, Observation):
        return ["result is not an Observation"]
    for f in ("id", "kind", "asset_id", "source", "data"):
        if getattr(obs, f, None) in (None, ""):
            errs.append(f"missing field {f}")
    if obs.asset_id not in set(asset_ids):
        errs.append(f"asset {obs.asset_id!r} is not part of this analysis")
    if obs.kind not in request.kinds or (kind and obs.kind != kind):
        errs.append(f"kind {obs.kind!r} was not requested" + (f" (expected {kind})" if kind else ""))
    if not SOURCE_RE.match(str(obs.source or "")) or str(obs.source).startswith("ai"):
        errs.append(f"source {obs.source!r} is not a tool measurement '<package>/<tool>@<version>'")
    if obs.analysis_id and obs.analysis_id != request.analysis_id:
        errs.append(f"analysis_id {obs.analysis_id!r} does not match request {request.analysis_id!r}")
    if getattr(obs, "provenance", "OBSERVED") != "OBSERVED":
        errs.append(f"observation provenance must be OBSERVED, got {obs.provenance!r}")
    if not isinstance(obs.data, dict):
        errs.append("data must be a structured object")
    else:
        errs += [f"data leaks {what}" for what in leak_scan(obs.data)]
    return errs


def leak_scan(data: Any, path: str = "data") -> List[str]:
    """Keys or values that look like credentials or executable material."""
    hits: List[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            if _LEAK_KEYS.search(str(k)):
                hits.append(f"key {path}.{k}")
            hits += leak_scan(v, f"{path}.{k}")
    elif isinstance(data, (list, tuple)):
        for i, v in enumerate(data):
            hits += leak_scan(v, f"{path}[{i}]")
    elif isinstance(data, str) and _LEAK_VALUES.search(data):
        hits.append(f"value at {path}")
    return hits


def safe_observation_summary(obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """What an AI provider may see of an observation: id, kind, asset, source and data with leak-like material removed.
    Returns None for anything that is not a tool measurement (AI evidence is built from real observations only)."""
    src = str(obs.get("source") or "")
    if not obs.get("id") or not SOURCE_RE.match(src) or src.startswith("ai") or obs.get("provenance", "OBSERVED") != "OBSERVED":
        return None
    return {"id": obs["id"], "kind": obs.get("kind"), "asset_id": obs.get("asset_id"), "source": src, "data": _scrub(obs.get("data") or {})}


def _scrub(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _scrub(x) for k, x in v.items() if not _LEAK_KEYS.search(str(k))}
    if isinstance(v, list):
        return [_scrub(x) for x in v]
    if isinstance(v, str) and _LEAK_VALUES.search(v):
        return "[removed]"
    return v


# ---- cache
def cache_key(fingerprint: str, kind: str, analyzer: str, tool: str, params: Dict[str, Any]) -> str:
    """Deterministic identity of a measurement: same asset content + kind + analyzer id@version + tool id@version + params."""
    return stable_hash({"fp": fingerprint, "kind": kind, "analyzer": analyzer, "tool": tool, "params": params})[:32]


class ObservationCache:
    """File-backed observation cache: <workspace>/cache/observations/<key>.json. Records carry the analyzer / tool identity
    that produced them, so a hit is auditable; a version or parameter change is a different key, never a stale reuse."""

    def __init__(self, root: Optional[str]):
        self.root = Path(root) / "cache" / "observations" if root else None
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Optional[Path]:
        return (self.root / f"{key}.json") if self.root else None

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._path(key)
        if not p or not p.exists():
            self.misses += 1
            return None
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if rec.get("cache_key") != key or not isinstance(rec.get("observation"), dict):
                raise ValueError("record does not match its key")
        except (OSError, ValueError) as e:
            self.misses += 1
            raise AnalysisError("ANALYSIS_CACHE_INVALID", f"{p.name}: {e}")
        self.hits += 1
        return rec

    def put(self, key: str, observation: Observation, produced_by: Dict[str, Any]) -> None:
        p = self._path(key)
        if not p:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"cache_key": key, "observation": observation.to_dict(), "produced_by": produced_by}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


# ---- analyzer contract
class Analyzer:
    """Deterministic observation producer. Subclasses measure through registry-selected tools only."""
    id = "abstract"
    version = "0"
    supported_kinds: Tuple[str, ...] = ()
    required_capabilities: Tuple[str, ...] = ()

    @property
    def identity(self) -> str:
        return f"{self.id}@{self.version}"

    def analyze(self, request: AnalysisRequest):
        raise NotImplementedError


class BudgetMeter:
    """Enforces AnalysisBudget before each measurement call; records usage for provenance."""

    def __init__(self, budget: AnalysisBudget):
        self.budget = budget
        self.calls = 0
        self.started = time.time()

    def check(self, what: str) -> None:
        b = self.budget
        if b.max_analysis_calls is not None and self.calls >= b.max_analysis_calls:
            raise AnalysisError("ANALYSIS_BUDGET_EXCEEDED", f"{what}: max_analysis_calls={b.max_analysis_calls} reached")
        if b.max_total_seconds is not None and (time.time() - self.started) >= b.max_total_seconds:
            raise AnalysisError("ANALYSIS_BUDGET_EXCEEDED", f"{what}: max_total_seconds={b.max_total_seconds} reached")

    def spent(self) -> None:
        self.calls += 1

    def usage(self) -> Dict[str, Any]:
        return {"calls": self.calls, "seconds": round(time.time() - self.started, 3), "limits": self.budget.to_dict(), "enforced": True,
                "unsupported": ["max_bytes_scanned", "max_duration_s", "max_probe_calls"]}
