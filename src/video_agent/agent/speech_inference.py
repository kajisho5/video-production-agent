"""SpeechEvent / AudioEvent(silence) → speech inferences (deterministic, evidence-based; no AI, no speaker identity).

Input: the timeline of an analysis — SPEECH events lifted from a transcript observation (PR #13) and AUDIO_SILENCE events
from a silence measurement (whichever measurement tool the registry selected). Output: Inferences that cite those events:

- speech_activity          one per asset: speech exists in these intervals (count, seconds, coverage)
- speech_interval          one per logical speech interval: consecutive SPEECH events whose gap is at most
                           `speech.merge_gap_seconds` form one interval (an interval operation, not an interpretation
                           of content); `speaker_id` stays null
- internal_silence_removable
                           an internal measured silence that lies strictly between two speech intervals, overlaps no
                           speech, and lasts at least `silence.internal.removable_min_seconds`: a production *candidate*
                           for removal (the decision layer decides, with CONFIRM by policy)
- speech_silence_conflict  a SPEECH event overlapping a measured silence: the two measurements disagree; recorded,
                           never resolved here (timestamps are not corrected, neither event is modified)

Thresholds come from the effective policy (profile / request) with an explicit DEFAULT when a key is absent; the value
and its provenance are recorded in the inference data. Nothing here reads transcript text, identifies a speaker, or
produces an operation, a decision, a tool argument or a command.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..media.analyzer import AnalysisResult
from ..models import Event, Inference
from ..policy.rules import RuleSet

SPEECH_MERGE_GAP_KEY = "speech.merge_gap_seconds"
SPEECH_MERGE_GAP_DEFAULT = 0.5           # seconds: recognised segments closer than this are one logical speech interval
REMOVABLE_MIN_KEY = "silence.internal.removable_min_seconds"
REMOVABLE_MIN_DEFAULT = 2.0              # seconds: shorter internal pauses are never proposed for removal
MARGIN_KEY = "silence.margin_seconds"    # existing key (leading / trailing trims): air kept on both sides of a removal
MARGIN_DEFAULT = 0.15
EDGE_TOL = 0.05                          # seconds: a silence touching 0 or the end is leading / trailing, not internal (same as inference.py)
OVERLAP_TOL = 0.01                       # seconds: measurement rounding, not a conflict


def _setting(rules: RuleSet, key: str, default: float) -> Dict[str, Any]:
    """Effective value and where it came from (PROFILE / USER / SYSTEM / DEFAULT) — recorded, never implicit."""
    if key in rules.effective:
        return {"value": float(rules.get(key)), "provenance": rules.provenance(key) or "SYSTEM", "source": rules.effective[key].source}
    return {"value": float(default), "provenance": "DEFAULT", "source": "video_agent.agent.speech_inference"}


def merge_speech_events(events: List[Event], merge_gap: float) -> List[List[Event]]:
    """Group consecutive SPEECH events whose gap is at most `merge_gap` seconds. Pure interval arithmetic on the events'
    own timestamps (no shifting, no snapping); groups are returned in temporal order."""
    groups: List[List[Event]] = []
    for e in sorted(events, key=lambda x: (float(x.range["start"]), float(x.range["end"] if x.range.get("end") is not None else x.range["start"]))):
        if groups and float(e.range["start"]) - float(groups[-1][-1].range["end"]) <= merge_gap:
            groups[-1].append(e)
        else:
            groups.append([e])
    return groups


def infer_speech(analysis: AnalysisResult, rules: RuleSet) -> List[Inference]:
    out: List[Inference] = []
    gap = _setting(rules, SPEECH_MERGE_GAP_KEY, SPEECH_MERGE_GAP_DEFAULT)
    removable = _setting(rules, REMOVABLE_MIN_KEY, REMOVABLE_MIN_DEFAULT)
    margin = _setting(rules, MARGIN_KEY, MARGIN_DEFAULT)
    for asset in analysis.assets:
        tl_id = f"asset:{asset.id}"
        speech = [e for e in analysis.timeline.query(type="SPEECH", timeline_id=tl_id) if e.provenance == "OBSERVED"]
        if not speech:
            continue   # no recognition → no speech inference (never inferred from silence alone)
        if any(e.metadata.get("speaker_id") is not None for e in speech):
            raise ValueError(f"asset {asset.id}: a SpeechEvent carries a speaker id; speaker identification is not part of this system")
        dur: Optional[float] = asset.technical.get("duration")
        transcript_ids = sorted({o.id for o in analysis.observations if o.asset_id == asset.id and o.kind == "transcript"})
        # ---- logical speech intervals
        intervals: List[Inference] = []
        for group in merge_speech_events(speech, gap["value"]):
            s, e = float(group[0].range["start"]), float(group[-1].range["end"])
            confs = [x.confidence for x in group if isinstance(x.confidence, (int, float))]
            inf = Inference(kind="speech_interval", asset_id=asset.id, confidence=round(min(confs), 3) if confs and len(confs) == len(group) else 0.5,
                            statement=f"speech from {s:.2f}s to {e:.2f}s ({len(group)} recognised segment(s), gaps ≤ {gap['value']:g}s merged); who speaks is unknown",
                            evidence=[x.id for x in group] + transcript_ids,
                            data={"start": s, "end": e, "seconds": round(e - s, 3), "segments": len(group), "events": [x.id for x in group], "speaker_id": None,
                                  "merge_gap": gap, "language": group[0].metadata.get("language")})
            intervals.append(inf)
        out.extend(intervals)
        total = round(sum(i.data["seconds"] for i in intervals), 3)
        out.append(Inference(kind="speech_activity", asset_id=asset.id, confidence=round(min(i.confidence for i in intervals), 3),
                             statement=f"speech is present in {len(intervals)} interval(s), {total:.2f}s" + (f" of {dur:.2f}s ({100.0 * total / dur:.0f}%)" if dur else ""),
                             evidence=[i.id for i in intervals] + transcript_ids,
                             data={"intervals": len(intervals), "speech_seconds": total, "duration": dur, "coverage": round(total / dur, 4) if dur else None,
                                   "interval_ids": [i.id for i in intervals], "speaker_id": None}))
        # ---- measured silences on the same timeline: conflicts and removable internal pauses
        silences = analysis.timeline.query(type="AUDIO_SILENCE", timeline_id=tl_id)
        for sil in silences:
            ss = float(sil.range["start"])
            se = float(sil.range["end"]) if sil.range.get("end") is not None else (dur if dur is not None else None)
            if se is None:
                continue
            overlapping = [i for i in intervals if i.data["start"] < se - OVERLAP_TOL and i.data["end"] > ss + OVERLAP_TOL]
            if overlapping:
                ov = round(sum(min(i.data["end"], se) - max(i.data["start"], ss) for i in overlapping), 3)
                out.append(Inference(kind="speech_silence_conflict", asset_id=asset.id, confidence=0.5,
                                     statement=f"measured silence {ss:.2f}-{se:.2f}s overlaps recognised speech for {ov:.2f}s; the measurements disagree and neither is corrected",
                                     evidence=[sil.id] + [i.id for i in overlapping], data={"silence": {"start": ss, "end": se}, "overlap_seconds": ov, "intervals": [i.id for i in overlapping]}))
                continue
            internal = ss > EDGE_TOL and (dur is None or se < dur - EDGE_TOL)
            before = [i for i in intervals if i.data["end"] <= ss + OVERLAP_TOL]
            after = [i for i in intervals if i.data["start"] >= se - OVERLAP_TOL]
            if not internal or not before or not after:
                continue   # leading / trailing silences are handled by the existing inferences; a pause needs speech on both sides
            if (se - ss) < removable["value"]:
                continue
            prev_i, next_i = max(before, key=lambda i: i.data["end"]), min(after, key=lambda i: i.data["start"])
            r_start, r_end = round(ss + margin["value"], 3), round(se - margin["value"], 3)
            if r_end <= r_start:
                continue
            out.append(Inference(kind="internal_silence_removable", asset_id=asset.id, confidence=0.7,
                                 statement=f"{se - ss:.2f}s pause between speech ending {prev_i.data['end']:.2f}s and speech starting {next_i.data['start']:.2f}s "
                                           f"(≥ {removable['value']:g}s): candidate for removal, keeping {margin['value']:g}s of air on each side",
                                 evidence=[sil.id] + list(sil.evidence) + [prev_i.id, next_i.id],
                                 data={"start": r_start, "end": r_end, "seconds": round(r_end - r_start, 3), "silence": {"start": ss, "end": se, "seconds": round(se - ss, 3)},
                                       "threshold": removable, "margin": margin, "before": prev_i.id, "after": next_i.id, "speaker_id": None}))
    return out
