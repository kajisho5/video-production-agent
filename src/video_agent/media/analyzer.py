"""MediaAnalyzer: the implemented Analyzer. Turns registry-selected measurement tools into Assets, Observations and
timeline Events, per AnalysisRequest (kinds, strategy, budget, cache). It never interprets; interpretation lives in
agent/inference.py. It never calls an AI provider and never executes anything but ToolAdapter.measure."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import Asset, Event, Observation, TimeRange, now_iso
from ..temporal import Timeline
from ..tools.base import ToolAdapter, ToolError
from .analysis import (ANALYSIS_KINDS, IR_STRATEGY, LEGACY_STRATEGY, AnalysisError, AnalysisRequest, Analyzer, BudgetMeter, ObservationCache,
                       cache_key, validate_observation)

STRATEGIES = ("FULL_ANALYSIS", "COARSE_ANALYSIS", "TARGETED_ANALYSIS", "CACHED_ONLY")   # names recorded in the IR


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class AnalysisResult:
    assets: List[Asset]
    observations: List[Observation]
    timeline: Timeline
    strategy: str
    warnings: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    analyses: List[Dict[str, Any]] = field(default_factory=list)   # analysis provenance (one entry per AnalysisRequest)
    budget: Dict[str, Any] = field(default_factory=dict)           # usage of the last request

    def to_dict(self) -> Dict[str, Any]:
        return {"assets": [a.to_dict() for a in self.assets], "observations": [o.to_dict() for o in self.observations],
                "timeline": self.timeline.to_dict(), "strategy": self.strategy, "warnings": self.warnings, "tool_calls": self.tool_calls,
                "analyses": self.analyses, "budget": self.budget}

    @classmethod
    def from_ir(cls, doc: Dict[str, Any]) -> "AnalysisResult":
        """Rebuild the analysis from a Project IR (assets, observations, timeline) so a revision re-plans from the same
        evidence without re-reading media. USER_DECISION events are dropped from the working timeline copy; they are
        kept in the IR itself."""
        assets = [Asset.from_dict(a) for a in doc["assets"].values()]
        obs = [Observation.from_dict(o) for o in doc["analysis"]["observations"]]
        tl = Timeline.from_dict(doc["timeline"])
        tl.events = [e for e in tl.events if e.type != "USER_DECISION"]
        return cls(assets=assets, observations=obs, timeline=tl, strategy=doc["analysis"].get("strategy", "FULL_ANALYSIS"),
                   warnings=list(doc["analysis"].get("warnings") or []), tool_calls=list(doc["analysis"].get("tool_calls") or []),
                   analyses=list(doc["analysis"].get("analyses") or []), budget=dict(doc["analysis"].get("budget") or {}))


class MediaAnalyzer(Analyzer):
    id = "media"
    version = "1.0"
    supported_kinds = tuple(ANALYSIS_KINDS)
    required_capabilities = ()   # capabilities come from the measurement skills in the registry (media_probe / silence_analysis / loudness_analysis)
    SKILLS = tuple(spec["skill"] for spec in ANALYSIS_KINDS.values())

    def __init__(self, adapter: ToolAdapter, tools: Dict[str, str], silence_threshold_db: float = -40.0, min_silence: float = 0.5, strategy: str = "FULL_ANALYSIS",
                 hash_sources: bool = True, cache_dir: Optional[str] = None):
        """`tools` is the skill → tool id map selected by SkillRegistry for this environment. The analyzer has no
        default engine: every measurement skill it uses must be present in the map."""
        if tools is None:
            raise TypeError("MediaAnalyzer needs the skill → tool map resolved by SkillRegistry (tools=None is not allowed)")
        missing = [x for x in self.SKILLS if not tools.get(x)]
        if missing:
            raise ToolError("no tool selected for skill(s): " + ", ".join(missing) + " (SkillRegistry.resolve_tools must provide them)")
        self.adapter = adapter
        self.threshold = silence_threshold_db
        self.min_silence = min_silence
        self.strategy = strategy if strategy in STRATEGIES else "FULL_ANALYSIS"
        self.hash_sources = hash_sources
        self.tools = dict(tools)
        self.cache = ObservationCache(cache_dir)
        self.measure_calls = 0   # tool calls actually made (cache hits excluded); the budget counts these

    def _tool(self, skill: str) -> str:
        return self.tools[skill]

    def _source(self, skill: str) -> str:
        tool = self._tool(skill)
        ver = self.adapter.version_of(tool) if hasattr(self.adapter, "version_of") else getattr(self.adapter, "version", "?")
        return f"{tool}@{ver}"

    # ---- backward-compatible entry: a FULL request with this analyzer's defaults
    def analyze(self, paths_or_request) -> AnalysisResult:
        if isinstance(paths_or_request, AnalysisRequest):
            return self.run(paths_or_request)
        req = AnalysisRequest(inputs=list(paths_or_request), strategy=LEGACY_STRATEGY.get(self.strategy, "FULL"),
                              params={"threshold_db": self.threshold, "min_silence": self.min_silence}, hash_sources=self.hash_sources)
        return self.run(req)

    # ---- Analyzer contract
    def run(self, req: AnalysisRequest) -> AnalysisResult:
        assets: List[Asset] = []
        obs: List[Observation] = []
        tl = Timeline()
        warnings: List[str] = []
        calls: List[Dict[str, Any]] = []
        rows: List[Dict[str, Any]] = []
        meter = BudgetMeter(req.budget)
        started = now_iso()
        params = {"threshold_db": float(req.params.get("threshold_db", self.threshold)), "min_silence": float(req.params.get("min_silence", self.min_silence))}
        for p in req.inputs:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            asset = Asset(path=str(Path(p).resolve()), provenance="USER")
            st = os.stat(p)
            if req.hash_sources:
                asset.hash = sha256_file(p)
            fp = asset.hash or f"stat:{st.st_size}:{int(st.st_mtime)}"
            # -- media_probe (asset identity; every other kind depends on it)
            o, row = self._observe(req, meter, "media_probe", asset, fp, {"inputs": [asset.path]}, {}, calls, [asset.id])
            rows.append(row)
            if o is None:
                raise AnalysisError(row["error"]["kind"] if row.get("error") else "ANALYZER_UNAVAILABLE", f"probe failed for {p}: {(row.get('error') or {}).get('message', '')}")
            probe = o.data
            asset.technical = {k: probe.get(k) for k in ("format", "duration", "size_bytes", "bitrate", "video", "audio", "subtitle_streams")}
            asset.technical["file"] = {"size": st.st_size, "mtime": st.st_mtime}  # fingerprint fallback when hashing is skipped
            asset.classification = _classify(probe)
            asset.type = asset.classification["type"]
            obs.append(o)
            tl.add_timeline(asset.id)
            dur = probe.get("duration") or 0.0
            has_audio = bool(probe.get("audio"))
            for kind in [k for k in req.kinds if k != "media_probe"]:
                if ANALYSIS_KINDS[kind]["needs_audio"] and not has_audio:
                    rows.append({"asset_id": asset.id, "kind": kind, "status": "SKIPPED", "reason": "no audio stream"})
                    continue
                if kind == "silence":
                    args, kp = {"input": asset.path, "list": True, "threshold": params["threshold_db"], "min_silence": params["min_silence"]}, params
                else:
                    args, kp = {"input": asset.path, "measure_only": True}, {}
                o, row = self._observe(req, meter, kind, asset, fp, args, kp, calls, [asset.id])
                rows.append(row)
                if o is None:
                    warnings.append(f"{kind} analysis failed for {p}: {(row.get('error') or {}).get('kind')} {(row.get('error') or {}).get('message', '')}".rstrip())
                    continue
                obs.append(o)
                self._events(kind, o, asset, dur, tl, params)
            if not has_audio:
                warnings.append(f"{p}: no audio stream; silence and loudness analysis skipped")
            assets.append(asset)
        usage = meter.usage()
        analysis = {"analysis_id": req.analysis_id, "request": req.to_dict(), "analyzer": self.identity, "started_at": started, "completed_at": now_iso(),
                    "status": "FAILED" if any(r["status"] == "FAILED" for r in rows) else "OK", "rows": rows, "budget": usage,
                    "cache": {"hits": self.cache.hits, "misses": self.cache.misses, "policy": req.cache_policy}}
        return AnalysisResult(assets=assets, observations=obs, timeline=tl, strategy=IR_STRATEGY[req.strategy], warnings=warnings, tool_calls=calls,
                              analyses=[analysis], budget=usage)

    def _observe(self, req: AnalysisRequest, meter: BudgetMeter, kind: str, asset: Asset, fp: str, args: Dict[str, Any], kparams: Dict[str, Any],
                 calls: List[Dict[str, Any]], asset_ids: List[str]):
        """One measurement: cache → budget → tool → validation. Returns (Observation | None, provenance row)."""
        skill = ANALYSIS_KINDS[kind]["skill"]
        tool = self._tool(skill)
        source = self._source(skill)
        key = cache_key(fp, kind, self.identity, source, kparams)
        row: Dict[str, Any] = {"asset_id": asset.id, "kind": kind, "tool": source, "cache_key": key, "cache_hit": False, "status": "OK"}
        if req.cache_policy in ("use", "only"):
            try:
                rec = self.cache.get(key)
            except AnalysisError as e:
                rec = None
                row["warning"] = str(e)
            if rec is not None:
                o = Observation.from_dict(rec["observation"])
                o.asset_id = asset.id           # cache identity is the asset content, the id belongs to this analysis
                o.analysis_id = req.analysis_id
                errs = validate_observation(o, req, asset_ids, kind)
                if errs:
                    row.update({"status": "FAILED", "error": {"kind": "ANALYSIS_CACHE_INVALID", "message": "; ".join(errs)}})
                    return None, row
                row["cache_hit"] = True
                row["produced_by"] = rec.get("produced_by")
                return o, row
        if req.cache_policy == "only":
            row.update({"status": "FAILED", "error": {"kind": "ANALYZER_UNAVAILABLE", "message": "CACHED_ONLY: no cached observation for this measurement"}})
            return None, row
        try:
            meter.check(f"{kind} on {asset.path}")
        except AnalysisError as e:
            row.update({"status": "FAILED", "error": {"kind": e.kind, "message": str(e)}})
            return None, row
        meter.spent()
        self.measure_calls += 1
        r = self.adapter.measure(tool, args)
        calls.append({"tool": r.tool, "ok": r.ok, "seconds": r.seconds, "kind": kind, "analysis_id": req.analysis_id})
        if not r.ok:
            row.update({"status": "FAILED", "error": {"kind": "ANALYZER_TIMEOUT" if "timeout" in (r.stderr_tail or "").lower() else "ANALYZER_UNAVAILABLE", "message": (r.stderr_tail or "")[:200]}})
            return None, row
        data = self._shape(kind, r.data, kparams)
        o = Observation(kind=kind, asset_id=asset.id, source=source, data=data, analysis_id=req.analysis_id, analyzer=self.identity, cache_key=key)
        errs = validate_observation(o, req, asset_ids, kind)
        if errs:
            row.update({"status": "FAILED", "error": {"kind": "ANALYSIS_INVALID_RESULT", "message": "; ".join(errs)}})
            return None, row
        if req.cache_policy != "bypass":
            self.cache.put(key, o, {"analyzer": self.identity, "tool": source, "params": kparams, "at": o.observed_at})
        return o, row

    @staticmethod
    def _shape(kind: str, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Structured observation data per kind (the tool's result keys the agent relies on)."""
        if kind == "media_probe":
            return dict(data)
        if kind == "silence":
            d = {k: data.get(k) for k in ("silences", "keep", "input_duration", "kept_duration", "removed_seconds", "threshold")}
            d["threshold_db"] = params.get("threshold_db")
            return d
        if kind == "loudness":
            d = {"silent": bool(data.get("silent"))}
            if not d["silent"]:
                d.update({"lufs": _f(data.get("input_i")), "true_peak": _f(data.get("input_tp")), "lra": _f(data.get("input_lra"))})
            return d
        return dict(data)

    @staticmethod
    def _events(kind: str, o: Observation, asset: Asset, dur: float, tl: Timeline, params: Dict[str, Any]) -> None:
        tid = f"asset:{asset.id}"
        if kind == "silence":
            for se in o.data.get("silences") or []:
                end = se[1] if se[1] is not None else dur
                tl.add(Event(type="AUDIO_SILENCE", timeline_id=tid, range=TimeRange(se[0], end).to_dict(), source=o.source, kind="OBSERVED",
                             confidence=None, evidence=[o.id], metadata={"threshold_db": params["threshold_db"], "runs_to_end": se[1] is None}))
            for ke in o.data.get("keep") or []:
                tl.add(Event(type="AUDIO_ACTIVE", timeline_id=tid, range=TimeRange(ke[0], ke[1]).to_dict(), source=o.source, kind="OBSERVED", evidence=[o.id]))
        elif kind == "loudness":
            tl.add(Event(type="LOUDNESS_MEASURE", timeline_id=tid, range=TimeRange(0.0, dur).to_dict(), source=o.source, kind="OBSERVED", evidence=[o.id], metadata=dict(o.data)))


def _classify(probe: Dict[str, Any]) -> Dict[str, Any]:
    """Cheap, evidence-backed classification. Conference roles (camera_a, slides...) come in Phase 2."""
    v, a = probe.get("video"), probe.get("audio")
    if v and not a:
        return {"type": "CAMERA", "confidence": 0.5, "evidence": ["video stream, no audio stream"]}
    if v:
        w, h = v.get("width") or 0, v.get("height") or 0
        if v.get("fps") and v["fps"] < 5:
            return {"type": "SCREEN_CAPTURE", "confidence": 0.4, "evidence": [f"fps {v['fps']}"]}
        if w and h and (w, h) not in ((1920, 1080), (3840, 2160), (1280, 720), (1080, 1920)) and v.get("variable_frame_rate_suspected"):
            return {"type": "SCREEN_CAPTURE", "confidence": 0.5, "evidence": [f"odd size {w}x{h}", "VFR"]}
        return {"type": "CAMERA", "confidence": 0.6, "evidence": ["video+audio streams"]}
    if a:
        return {"type": "AUDIO", "confidence": 0.9, "evidence": ["audio stream only"]}
    return {"type": "UNKNOWN", "confidence": 0.0, "evidence": []}


def _f(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
