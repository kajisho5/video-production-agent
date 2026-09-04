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
from ..tools.base import ToolAdapter

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


class MediaAnalyzer:
    def __init__(self, adapter: ToolAdapter, silence_threshold_db: float = -40.0, min_silence: float = 0.5, strategy: str = "TARGETED_ANALYSIS",
                 hash_sources: bool = True):
        self.adapter = adapter
        self.threshold = silence_threshold_db
        self.min_silence = min_silence
        self.strategy = strategy if strategy in STRATEGIES else "TARGETED_ANALYSIS"
        self.hash_sources = hash_sources
        self.src = f"ffmpeg-skill@{getattr(adapter, 'version', '?')}"

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
            if self.hash_sources:
                asset.hash = sha256_file(p)
            r = self.adapter.measure("ffmpeg-skill/probe", {"inputs": [asset.path]})
            calls.append({"tool": r.tool, "ok": r.ok, "seconds": r.seconds})
            if not r.ok:
                raise RuntimeError(f"probe failed for {p}: {r.stderr_tail}")
            probe = r.data
            asset.technical = {k: probe.get(k) for k in ("format", "duration", "size_bytes", "bitrate", "video", "audio", "subtitle_streams")}
            asset.classification = _classify(probe)
            asset.type = asset.classification["type"]
            obs.append(Observation(kind="probe", asset_id=asset.id, source=f"{self.src}/probe", data=probe))
            tl.add_timeline(asset.id)
            dur = probe.get("duration") or 0.0
            if probe.get("audio"):
                s = self.adapter.measure("ffmpeg-skill/silence", {"input": asset.path, "list": True, "threshold": self.threshold, "min_silence": self.min_silence})
                calls.append({"tool": s.tool, "ok": s.ok, "seconds": s.seconds})
                if s.ok:
                    o = Observation(kind="silence", asset_id=asset.id, source=f"{self.src}/silence", data={k: s.data.get(k) for k in ("silences", "keep", "input_duration", "kept_duration", "removed_seconds", "threshold")})
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
                m = self.adapter.measure("ffmpeg-skill/loudness", {"input": asset.path, "measure_only": True})
                calls.append({"tool": m.tool, "ok": m.ok, "seconds": m.seconds})
                if m.ok:
                    d = m.data
                    data = {"silent": bool(d.get("silent"))}
                    if not data["silent"]:
                        data.update({"lufs": _f(d.get("input_i")), "true_peak": _f(d.get("input_tp")), "lra": _f(d.get("input_lra"))})
                    o = Observation(kind="loudness", asset_id=asset.id, source=f"{self.src}/loudness", data=data)
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
