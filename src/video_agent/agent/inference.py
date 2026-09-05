"""Observations → Inferences. Each inference cites the observation/event ids it rests on and never
mutates them."""
from __future__ import annotations

from typing import Dict, List

from ..media.analysis import loudness_facts
from ..media.analyzer import AnalysisResult
from ..models import Inference
from ..policy.rules import RuleSet

EDGE_TOL = 0.05  # seconds: a silence starting within this of 0 counts as "leading"


def infer(analysis: AnalysisResult, rules: RuleSet, target_lufs: float = None, tolerance_lu: float = 2.0) -> List[Inference]:
    out: List[Inference] = []
    min_lead = float(rules.get("silence.leading.min_seconds", 1.0))
    margin = float(rules.get("silence.margin_seconds", 0.15))
    min_internal = float(rules.get("silence.internal.min_seconds", 1.0))  # keep a little air so words are not clipped (same idea as silence.py --margin)
    for asset in analysis.assets:
        dur = (asset.technical.get("duration") or 0.0)
        tl_id = f"asset:{asset.id}"
        silences = analysis.timeline.query(type="AUDIO_SILENCE", timeline_id=tl_id)
        speech = analysis.timeline.query(type="AUDIO_ACTIVE", timeline_id=tl_id)
        for ev in silences:
            s, e = ev.range["start"], ev.range["end"]
            if s <= EDGE_TOL and (e - s) >= min_lead:
                out.append(Inference(kind="leading_silence_unwanted", asset_id=asset.id, confidence=0.9 if speech else 0.5,
                                     statement=f"audio below {ev.metadata.get('threshold_db')} dB for the first {e - s:.2f}s; likely technical lead-in, not content",
                                     evidence=[ev.id] + ev.evidence, data={"start": 0.0, "end": round(max(0.0, e - margin), 3), "seconds": round(e - s, 3), "margin": margin}))
            if dur and e >= dur - EDGE_TOL and (e - s) >= min_lead:
                out.append(Inference(kind="trailing_silence_unwanted", asset_id=asset.id, confidence=0.9 if speech else 0.5,
                                     statement=f"audio below threshold for the last {e - s:.2f}s; likely technical tail",
                                     evidence=[ev.id] + ev.evidence, data={"start": round(min(dur, s + margin), 3), "end": round(dur, 3), "seconds": round(e - s, 3), "margin": margin}))
            if s > EDGE_TOL and (not dur or e < dur - EDGE_TOL) and (e - s) >= min_internal:
                out.append(Inference(kind="internal_silence_candidate", asset_id=asset.id, confidence=0.4,
                                     statement=f"{e - s:.2f}s gap at {s:.2f}s; could be a pause worth keeping (not proposed for removal in Phase 1)",
                                     evidence=[ev.id] + ev.evidence, data={"start": round(s, 3), "end": round(e, 3), "seconds": round(e - s, 3)}))
        for obs in analysis.observations:
            if obs.asset_id != asset.id or obs.kind != "loudness":
                continue
            lf = loudness_facts(obs.data)   # tool vocabulary (ffmpeg-skill / media-analysis) → one view; the fact itself is untouched
            if lf["silent"]:
                out.append(Inference(kind="audio_silent", asset_id=asset.id, confidence=0.95, statement="integrated loudness is -inf: the track carries no programme audio",
                                     evidence=[obs.id], data={}))
            elif target_lufs is not None and lf["lufs"] is not None:
                lufs = lf["lufs"]
                if lufs <= -40:
                    out.append(Inference(kind="ambience_not_programme", asset_id=asset.id, confidence=0.7,
                                         statement=f"{lufs:.1f} LUFS is room tone / near-silence level; normalising would raise noise, not content",
                                         evidence=[obs.id], data={"lufs": lufs}))
                elif abs(lufs - target_lufs) > tolerance_lu:
                    out.append(Inference(kind="loudness_off_target", asset_id=asset.id, confidence=0.95,
                                         statement=f"measured {lufs:.1f} LUFS vs target {target_lufs:g} (tolerance ±{tolerance_lu:g} LU)",
                                         evidence=[obs.id], data={"lufs": lufs, "target": target_lufs, "true_peak": lf["true_peak"]}))
    return out
