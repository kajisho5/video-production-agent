"""MediaAnalyzer: turns ffmpeg-skill measurements into Assets, Observations and timeline Events.
It never interprets; interpretation lives in agent/inference.py."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import Asset, Event, Observation, TimeRange
from ..temporal import Timeline
from ..tools.base import ToolAdapter, ToolError

STRATEGIES = ("FULL_ANALYSIS", "COARSE_ANALYSIS", "TARGETED_ANALYSIS")


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

    def to_dict(self) -> Dict[str, Any]:
        return {"assets": [a.to_dict() for a in self.assets], "observations": [o.to_dict() for o in self.observations],
                "timeline": self.timeline.to_dict(), "strategy": self.strategy, "warnings": self.warnings, "tool_calls": self.tool_calls}

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
                   warnings=list(doc["analysis"].get("warnings") or []), tool_calls=list(doc["analysis"].get("tool_calls") or []))


class MediaAnalyzer:
    SKILLS = ("media_probe", "silence_analysis", "loudness_analysis")

    def __init__(self, adapter: ToolAdapter, tools: Dict[str, str], silence_threshold_db: float = -40.0, min_silence: float = 0.5, strategy: str = "FULL_ANALYSIS",
                 hash_sources: bool = True):
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

    def _tool(self, skill: str) -> str:
        return self.tools[skill]

    def _source(self, skill: str) -> str:
        tool = self._tool(skill)
        ver = self.adapter.version_of(tool) if hasattr(self.adapter, "version_of") else getattr(self.adapter, "version", "?")
        return f"{tool}@{ver}"

    def analyze(self, paths: List[str]) -> AnalysisResult:
        assets: List[Asset] = []
        obs: List[Observation] = []
        tl = Timeline()
        warnings: List[str] = []
        calls: List[Dict[str, Any]] = []
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            asset = Asset(path=str(Path(p).resolve()), provenance="USER")
            st = os.stat(p)
            if self.hash_sources:
                asset.hash = sha256_file(p)
            r = self.adapter.measure(self._tool("media_probe"), {"inputs": [asset.path]})
            calls.append({"tool": r.tool, "ok": r.ok, "seconds": r.seconds})
            if not r.ok:
                raise RuntimeError(f"probe failed for {p}: {r.stderr_tail}")
            probe = r.data
            asset.technical = {k: probe.get(k) for k in ("format", "duration", "size_bytes", "bitrate", "video", "audio", "subtitle_streams")}
            asset.technical["file"] = {"size": st.st_size, "mtime": st.st_mtime}  # fingerprint fallback when hashing is skipped
            asset.classification = _classify(probe)
            asset.type = asset.classification["type"]
            obs.append(Observation(kind="probe", asset_id=asset.id, source=self._source("media_probe"), data=probe))
            tl.add_timeline(asset.id)
            dur = probe.get("duration") or 0.0
            if probe.get("audio"):
                s = self.adapter.measure(self._tool("silence_analysis"), {"input": asset.path, "list": True, "threshold": self.threshold, "min_silence": self.min_silence})
                calls.append({"tool": s.tool, "ok": s.ok, "seconds": s.seconds})
                if s.ok:
                    o = Observation(kind="silence", asset_id=asset.id, source=self._source("silence_analysis"), data={k: s.data.get(k) for k in ("silences", "keep", "input_duration", "kept_duration", "removed_seconds", "threshold")})
                    o.data["threshold_db"] = self.threshold
                    obs.append(o)
                    for se in s.data.get("silences") or []:
                        end = se[1] if se[1] is not None else dur
                        tl.add(Event(type="AUDIO_SILENCE", timeline_id=f"asset:{asset.id}", range=TimeRange(se[0], end).to_dict(), source=o.source, kind="OBSERVED",
                                     confidence=None, evidence=[o.id], metadata={"threshold_db": self.threshold, "runs_to_end": se[1] is None}))
                    for ke in s.data.get("keep") or []:
                        tl.add(Event(type="AUDIO_ACTIVE", timeline_id=f"asset:{asset.id}", range=TimeRange(ke[0], ke[1]).to_dict(), source=o.source, kind="OBSERVED", evidence=[o.id]))
                else:
                    warnings.append(f"silence analysis failed for {p}: {s.stderr_tail}")
                m = self.adapter.measure(self._tool("loudness_analysis"), {"input": asset.path, "measure_only": True})
                calls.append({"tool": m.tool, "ok": m.ok, "seconds": m.seconds})
                if m.ok:
                    d = m.data
                    data = {"silent": bool(d.get("silent"))}
                    if not data["silent"]:
                        data.update({"lufs": _f(d.get("input_i")), "true_peak": _f(d.get("input_tp")), "lra": _f(d.get("input_lra"))})
                    o = Observation(kind="loudness", asset_id=asset.id, source=self._source("loudness_analysis"), data=data)
                    obs.append(o)
                    tl.add(Event(type="LOUDNESS_MEASURE", timeline_id=f"asset:{asset.id}", range=TimeRange(0.0, dur).to_dict(), source=o.source, kind="OBSERVED", evidence=[o.id], metadata=data))
                else:
                    warnings.append(f"loudness analysis failed for {p}: {m.stderr_tail}")
            else:
                warnings.append(f"{p}: no audio stream; silence and loudness analysis skipped")
            assets.append(asset)
        return AnalysisResult(assets=assets, observations=obs, timeline=tl, strategy=self.strategy, warnings=warnings, tool_calls=calls)


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
