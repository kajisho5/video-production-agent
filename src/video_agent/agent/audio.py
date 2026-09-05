"""Audio production vocabulary (ADR-030): what the agent plans through audio-production-skill, and how audio subjects differ
from video subjects.

    Observation (probe: audio / video streams, channels, sample rate; loudness; silence) → Inference → Decision → step → IR
    audio operation → compiler → audio-production-skill

Vocabulary and arithmetic only, no execution:

- **Switch:** `audio.production` (explicit USER / PROFILE requirement) puts an asset on the audio production path: the asset's
  *audio* becomes the subject (a video container's audio track is extracted by the Skill; the picture is not delivered — a
  CONFIRM decision says so). Without the switch nothing here is planned and the existing paths are byte-identical.
- **Requirements** (`audio.<op>` keys, explicit only, range-checked here; invalid → EditRequirementError-like refusal):
  `audio.gain` (dB), `audio.fade_in` / `audio.fade_out` (seconds), `audio.channels` (mono | stereo: a *target*; the decision
  compares it with the probed channel count and yields KEEP, MONO, STEREO, DOWNMIX or BLOCK), `audio.concat` (bool, order =
  input order) + `audio.concat.crossfade` (seconds), `audio.sample_rate` (only as NORMALIZE.sample_rate — the Skill has no
  standalone resample: without a normalisation decision it is refused as BLOCK).
- **Derived from existing decisions** (never from observations directly): the silence decisions (leading / trailing /
  internal REMOVE) become one `audio.cut` (explicit remove ranges); the `audio.loudness` TRANSFORM decision (measured loudness
  off target, or the user asked) becomes the Skill's NORMALIZE — the same IR operation type, lowered by the compiler to the tool
  the plan selected.
- **Fixed order** after the cut: concat → gain → channels → fade_in → fade_out → normalize (loudness last, so the delivered
  level is the normalised one).
- **Four stages:** the Skill supports 14 operation types; the adapter lowers every declared type; the planner generates the
  seven listed here (TRIM as a cut, CUT, NORMALIZE, GAIN, FADE_IN, FADE_OUT, MONO / STEREO / DOWNMIX, CONCAT); MIX,
  SILENCE_REMOVE (the agent already subtracts margins itself), NOISE_REDUCTION and DYNAMICS are executable through the adapter
  but not planned (no measurement or requirement grounds them today).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..models import Requirement
from .editing import EditRequirementError, _bool, _number

PROGRAMME_AUDIO = "programme_audio"     # logical id of the audio concat output
TOOL = "audio-production/run"
SWITCH = "audio.production"
# IR audio operation type → production skill, Skill operation type, IR parameter allowlist (besides references)
OPERATIONS: Dict[str, Dict[str, Any]] = {
    "audio.cut":      {"skill": "audio_cut",      "type": "CUT",      "params": ("remove",), "risk": "LOW"},
    "audio.concat":   {"skill": "audio_concat",   "type": "CONCAT",   "params": ("crossfade",), "risk": "MEDIUM"},
    "audio.gain":     {"skill": "audio_gain",     "type": "GAIN",     "params": ("gain_db",), "risk": "LOW"},
    "audio.mono":     {"skill": "audio_mono",     "type": "MONO",     "params": (), "risk": "MEDIUM"},
    "audio.stereo":   {"skill": "audio_stereo",   "type": "STEREO",   "params": (), "risk": "LOW"},
    "audio.downmix":  {"skill": "audio_downmix",  "type": "DOWNMIX",  "params": (), "risk": "MEDIUM"},
    "audio.fade_in":  {"skill": "audio_fade_in",  "type": "FADE_IN",  "params": ("duration",), "risk": "LOW"},
    "audio.fade_out": {"skill": "audio_fade_out", "type": "FADE_OUT", "params": ("duration",), "risk": "LOW"},
    # audio.loudness is the existing IR type; on the audio path its skill is audio_normalize (NORMALIZE), else loudness_normalization
    "audio.loudness": {"skill": "audio_normalize", "type": "NORMALIZE", "params": ("target_lufs", "true_peak", "tolerance_lu", "sample_rate"), "risk": "LOW"},
}
SKILL_OF = {k: v["skill"] for k, v in OPERATIONS.items()}
# the order the planner / compiler apply audio operations on one subject (after the per-asset cut and the concat)
AUDIO_ORDER = ("audio.gain", "audio.mono", "audio.stereo", "audio.downmix", "audio.fade_in", "audio.fade_out", "audio.loudness")
CHANNEL_SKILLS = ("audio_mono", "audio_stereo", "audio_downmix")
REQUIREMENT_KEYS = ("audio.production", "audio.gain", "audio.fade_in", "audio.fade_out", "audio.channels", "audio.concat", "audio.concat.crossfade", "audio.sample_rate")
SAMPLE_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000)
GAIN_RANGE = (-60.0, 60.0)
FADE_RANGE = (0.001, 3600.0)
CROSSFADE_RANGE = (0.0, 30.0)
_ASSET_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def is_audio_capable(technical: Dict[str, Any]) -> bool:
    return bool((technical or {}).get("audio"))


def has_video(technical: Dict[str, Any]) -> bool:
    return bool((technical or {}).get("video"))


def audio_channels(technical: Dict[str, Any]) -> Optional[int]:
    a = (technical or {}).get("audio") or {}
    raw = a.get("channels")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def parse_audio_requirements(m: Dict[str, Requirement]) -> Dict[str, Any]:
    """Requirement map → {"production": bool, "requirements": [Requirement], "gain", "fade_in", "fade_out", "channels", "concat",
    "crossfade", "sample_rate"} (keys present only when asked). Values are range-checked; DEFAULT never switches anything on;
    an `audio.<op>` without `audio.production` is an ambiguous request and is refused."""
    out: Dict[str, Any] = {"production": False, "requirements": []}
    sw = m.get(SWITCH)
    if sw is not None and sw.provenance != "DEFAULT":
        out["production"] = _bool(sw.value, SWITCH)
        out["requirements"].append(sw)
    asked = [k for k in REQUIREMENT_KEYS[1:] if k in m and m[k].provenance != "DEFAULT"]
    if asked and not out["production"]:
        raise EditRequirementError(f"{asked[0]} is set but {SWITCH}=true is not: refusing an ambiguous request")
    if not out["production"]:
        return out
    for key in asked:
        r = m[key]
        v = r.value
        out["requirements"].append(r)
        if key == "audio.gain":
            out["gain"] = float(_number(v, key, GAIN_RANGE[0], GAIN_RANGE[1]))
            if out["gain"] == 0.0:
                raise EditRequirementError("audio.gain=0 changes nothing; leave it unset")
        elif key in ("audio.fade_in", "audio.fade_out"):
            out[key.split(".", 1)[1]] = float(_number(v, key, FADE_RANGE[0], FADE_RANGE[1]))
        elif key == "audio.channels":
            if not isinstance(v, str) or v.strip().lower() not in ("mono", "stereo"):
                raise EditRequirementError(f"audio.channels must be mono or stereo, got {v!r}")
            out["channels"] = v.strip().lower()
        elif key == "audio.concat":
            out["concat"] = _bool(v, key)
        elif key == "audio.concat.crossfade":
            out["crossfade"] = float(_number(v, key, CROSSFADE_RANGE[0], CROSSFADE_RANGE[1]))
        elif key == "audio.sample_rate":
            sr = _number(v, key, 8000, 96000)
            if int(sr) != float(sr) or int(sr) not in SAMPLE_RATES:
                raise EditRequirementError(f"audio.sample_rate must be one of {SAMPLE_RATES}, got {v!r}")
            out["sample_rate"] = int(sr)
    if "crossfade" in out and not out.get("concat"):
        raise EditRequirementError("audio.concat.crossfade needs audio.concat=true")
    return out


def channel_operation(target: str, channels: Optional[int]) -> Tuple[Optional[str], str]:
    """(IR operation type or None for keep, reason). The Skill's contract: MONO needs 2 channels, STEREO 1 or 2, DOWNMIX 5.1 / 7.1."""
    if channels is None:
        return "BLOCK", "channel count of the source is unknown"
    if target == "mono":
        if channels == 1:
            return None, "source is already mono"
        if channels == 2:
            return "audio.mono", "stereo source → mono (0.5 L + 0.5 R)"
        return "BLOCK", f"{channels}-channel source cannot be down-mixed to mono by the Skill (MONO takes a 2-channel input)"
    if target == "stereo":
        if channels == 2:
            return None, "source is already stereo"
        if channels == 1:
            return "audio.stereo", "mono source → stereo (duplicated)"
        if channels in (6, 8):
            return "audio.downmix", f"{channels}-channel source → stereo down-mix (5.1 / 7.1 weights)"
        return "BLOCK", f"{channels}-channel source has no stereo down-mix in the Skill (DOWNMIX takes 5.1 / 7.1)"
    return "BLOCK", f"unknown channel target {target!r}"


def cut_ranges(removed: List[List[float]]) -> List[List[float]]:
    """The planner's removed ranges (already margin-adjusted by the decisions) as sorted, merged, non-overlapping [start, end)."""
    rs = sorted(([round(float(s), 3), round(float(e), 3)] for s, e in removed if float(e) > float(s)), key=lambda r: r[0])
    out: List[List[float]] = []
    for r in rs:
        if out and r[0] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], r[1])
        else:
            out.append(r)
    return out


def kept_after_cut(duration: float, remove: List[List[float]]) -> float:
    return round(max(0.0, float(duration) - sum(e - s for s, e in remove)), 3)


def concat_segments(inputs: List[str], durations: Dict[str, float], crossfade: float) -> Tuple[List[Dict[str, Any]], float]:
    """Timeline of an audio concat: every input in order, each clip starting `crossfade` seconds before the previous one ends."""
    segs: List[Dict[str, Any]] = []
    t = 0.0
    for n, aid in enumerate(inputs):
        d = round(float(durations.get(aid) or 0.0), 3)
        if n > 0:
            t = max(0.0, t - crossfade)
        segs.append({"input": aid, "track": "A1", "source_range": [0.0, d], "timeline_range": [round(t, 3), round(t + d, 3)]})
        t = round(t + d, 3)
    return segs, round(t, 3)


def ir_audio_operation(op_type: str, subject: str, params: Dict[str, Any], decision_ids: List[str], scope: Optional[Dict[str, float]] = None, **refs: Any) -> Dict[str, Any]:
    spec = OPERATIONS[op_type]
    rec: Dict[str, Any] = {"type": op_type, "asset": subject}
    for k, v in refs.items():
        rec[k] = v
    for k in spec["params"]:
        if k in params and params[k] is not None:
            rec[k] = params[k]
    if scope is not None:
        rec["temporal_scope"] = {"start": round(float(scope["start"]), 3), "end": round(float(scope["end"]), 3)}
    rec["decision_ids"] = list(decision_ids)
    return rec


def audio_subjects(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Subjects that went through the audio production path (their deliverable is audio-only): subject id → {sources, duration}
    derived from the IR audio operations (cut, concat, speed-less chain). Empty when nothing was planned on that path."""
    assets = doc.get("assets") or {}
    aops = (doc.get("audio") or {}).get("operations") or []
    on_path = {op["asset"] for op in aops if op.get("type") in OPERATIONS and op.get("type") != "audio.loudness"} | \
              {op["asset"] for op in aops if op.get("type") == "audio.loudness" and op.get("input") is not None}
    out: Dict[str, Dict[str, Any]] = {}
    concat = next((op for op in aops if op.get("type") == "audio.concat"), None)
    for aid in on_path:
        if aid in assets:
            dur = float((assets[aid].get("technical") or {}).get("duration") or 0.0)
            cut = next((op for op in aops if op.get("type") == "audio.cut" and op.get("asset") == aid), None)
            if cut is not None:
                dur = kept_after_cut(dur, cut.get("remove") or [])
            out[aid] = {"sources": [aid], "duration": dur}
    if concat is not None:
        out[concat.get("output", PROGRAMME_AUDIO)] = {"sources": list(concat.get("inputs") or []), "duration": float(concat.get("timeline_duration") or 0.0)}
        for aid in concat.get("inputs") or []:
            out.pop(aid, None)   # the inputs are consumed by the programme
    for sid, row in out.items():
        chans = [audio_channels((assets.get(a) or {}).get("technical") or {}) for a in row["sources"]]
        known_chans = [c for c in chans if c is not None]
        expected: Optional[int] = max(known_chans) if chans and len(known_chans) == len(chans) else None
        for op in aops:   # a channel operation on the subject sets what QA expects
            if op.get("asset") == sid and op.get("type") in ("audio.mono", "audio.stereo", "audio.downmix"):
                expected = 1 if op["type"] == "audio.mono" else 2
        row["channels"] = expected
    return out
