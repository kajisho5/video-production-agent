"""Audio production vocabulary (ADR-030): what the agent plans through audio-production-skill, and how audio subjects differ
from video subjects.

    Observation (probe: audio / video streams, channels, sample rate; loudness; silence) → Inference → Decision → step → IR
    audio operation → compiler → audio-production-skill

Vocabulary and arithmetic only, no execution:

- **Switch:** `audio.production` (explicit USER / PROFILE requirement) puts an asset on the audio production path: the asset's
  *audio* becomes the subject (a video container's audio track is extracted by the Skill; the picture is not delivered — a
  CONFIRM decision `audio.extract` says so). The switch only *enables* the path: it is not the confirmation of the extraction.
  Only the dedicated explicit requirement `audio.extract=true` (USER) waives that CONFIRM, exactly as `audio.gain` waives
  `audio.gain.approval` — the generic switch never does. Without the switch nothing here is planned and the existing paths are
  byte-identical.
- **Requirements** (`audio.<op>` keys, explicit only, range-checked here; invalid → EditRequirementError-like refusal):
  `audio.extract` (bool: the user confirms up front that a video container is delivered as audio only), `audio.gain` (dB), `audio.fade_in` / `audio.fade_out` (seconds), `audio.channels` (mono | stereo: a *target*; the decision
  compares it with the probed channel count and yields KEEP, MONO, STEREO, DOWNMIX or BLOCK), `audio.concat` (bool, order =
  input order) + `audio.concat.crossfade` (seconds), `audio.sample_rate` (only as NORMALIZE.sample_rate — the Skill has no
  standalone resample: without a normalisation decision it is refused as BLOCK).
- **Derived from existing decisions** (never from observations directly): the silence decisions (leading / trailing /
  internal REMOVE) become one `audio.cut` (explicit remove ranges); the `audio.loudness` TRANSFORM decision (measured loudness
  off target, or the user asked) becomes the Skill's NORMALIZE — the same IR operation type, lowered by the compiler to the tool
  the plan selected.
- **Mix / noise reduction / dynamics** (ADR-063, keys named after the pinned contract's parameters, ranges from it):
  `audio.mix` (bool: every audio input, in input order, summed into one programme; the first input sets its length) +
  `audio.mix.levels` (per input: a gain in dB, `mute` or {gain_db, mute}, aligned with the inputs), NOISE_REDUCTION's
  `audio.noise_reduction.strength_db` (required by the contract; 20..60, the contract's range since audio-production 0.4.5) + `audio.noise_reduction.mode` (fft), and DYNAMICS'
  three stages `audio.dynamics.<gate|compressor|limiter>` (true: the stage with the Skill's own defaults) + `.<field>` (each stage
  field of the contract, e.g. `audio.dynamics.compressor.ratio`). The mix keeps every input on its own timeline from 0 s, so
  no silence is cut from a mixed input. Ducking: the pinned MIX has none (no sidechain parameter), so none is planned.
- **Voice clean-up** (ADR-068, audio-production-skill 0.4.3): `audio.voice_cleanup=light|medium|strong` → VOICE_CLEANUP (the engine's
  speech chain: highpass / de-esser / FFT denoise / compression / limiter by level). light / strong need engine release 1.13+; an
  older engine runs medium only and the Skill refuses the others with TOOL_ERROR `voice_level_unsupported` (never substituted).
- **Fixed order** after the cut: concat | mix → voice clean-up → noise reduction → gain → channels → dynamics → fade_in → fade_out → normalize
  (noise is estimated on the raw signal, dynamics act on the gained / laid-out signal, fades are not pumped by a compressor,
  loudness last so the delivered level is the normalised one).
- **Four stages:** the Skill supports 15 operation types (0.4.3); the adapter lowers every declared type; the planner generates the
  ones listed here (TRIM as a cut, CUT, NORMALIZE, GAIN, FADE_IN, FADE_OUT, MONO / STEREO / DOWNMIX, CONCAT, and since ADR-063
  MIX, NOISE_REDUCTION, DYNAMICS, since ADR-068 VOICE_CLEANUP); SILENCE_REMOVE (the agent already subtracts margins itself) is executable through the
  adapter but not planned.
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
    # ADR-063: the other way to build the audio programme (2..8 inputs summed; the first sets the length), and two chain operations
    "audio.mix":      {"skill": "audio_mix",      "type": "MIX",      "params": ("levels",), "risk": "MEDIUM"},
    # ADR-068: audio-production-skill 0.4.3's speech clean-up preset (ffmpeg-skill/audio --voice <level>), one typed level
    "audio.voice_cleanup": {"skill": "audio_voice_cleanup", "type": "VOICE_CLEANUP", "params": ("level",), "risk": "MEDIUM"},
    "audio.noise_reduction": {"skill": "audio_noise_reduction", "type": "NOISE_REDUCTION", "params": ("mode", "strength_db"), "risk": "MEDIUM"},
    "audio.dynamics": {"skill": "audio_dynamics", "type": "DYNAMICS", "params": ("gate", "compressor", "limiter"), "risk": "MEDIUM"},
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
AUDIO_ORDER = ("audio.voice_cleanup", "audio.noise_reduction", "audio.gain", "audio.mono", "audio.stereo", "audio.downmix", "audio.dynamics", "audio.fade_in", "audio.fade_out", "audio.loudness")
PROGRAMME_OPS = ("audio.concat", "audio.mix")   # the two ways to build PROGRAMME_AUDIO from 2+ inputs (mutually exclusive)
CHANNEL_SKILLS = ("audio_mono", "audio_stereo", "audio_downmix")
# ADR-063: the pinned contract's parameters (audio-production 0.4.3, which added VOICE_CLEANUP; tests/test_satellite_params.py checks them against it)
MAX_MIX_INPUTS = 8                                   # contract max_mix_inputs / MIX.inputs.max
MIX_GAIN_RANGE = (-60.0, 60.0)                       # MIX.levels[].gain_db
NOISE_REDUCTION_MODES = ("fft",)                     # NOISE_REDUCTION.mode (enum; only afftdn is implemented)
NOISE_REDUCTION_CONTRACT_STRENGTH = (20.0, 60.0)     # NOISE_REDUCTION.strength_db as the contract declares it (required, no default)
# 0.4.5 (audio ADR-17) raised the contract's min from 10 to 20: the FFT denoiser's noise floor `nf=-<strength>` only takes -80..-20,
# which the agent already enforced at planning time (ADR-063); the contract and what runs now agree.
NOISE_REDUCTION_STRENGTH = (20.0, 60.0)
DYNAMICS_STAGES: Dict[str, Dict[str, Tuple[float, float]]] = {   # DYNAMICS.<stage>.properties: field -> (min, max); fixed order gate → compressor → limiter
    "gate": {"threshold_db": (-60.0, 0.0), "ratio": (1.0, 9000.0), "attack_ms": (0.01, 9000.0), "release_ms": (0.01, 9000.0), "range_db": (-90.0, 0.0), "knee_db": (1.0, 8.0)},
    "compressor": {"threshold_db": (-60.0, 0.0), "ratio": (1.0, 20.0), "attack_ms": (0.01, 2000.0), "release_ms": (0.01, 9000.0), "makeup_db": (0.0, 36.0), "knee_db": (1.0, 8.0)},
    "limiter": {"ceiling_db": (-24.0, 0.0), "attack_ms": (0.1, 80.0), "release_ms": (1.0, 8000.0)},
}
# ADR-068: VOICE_CLEANUP.level (required enum, no default). light / strong need an ffmpeg-skill whose --voice takes a level (>= 1.13);
# with an older one the Skill runs only `medium` and refuses the others before any tool runs (TOOL_ERROR voice_level_unsupported).
VOICE_CLEANUP_LEVELS = ("light", "medium", "strong")
VOICE_CLEANUP_KEY = "audio.voice_cleanup"
MIX_KEYS = ("audio.mix", "audio.mix.levels")
NOISE_REDUCTION_KEYS = ("audio.noise_reduction.strength_db", "audio.noise_reduction.mode")
DYNAMICS_KEYS = tuple(k for st, fields in DYNAMICS_STAGES.items() for k in (f"audio.dynamics.{st}",) + tuple(f"audio.dynamics.{st}.{f}" for f in fields))
REQUIREMENT_KEYS = ("audio.production", "audio.extract", "audio.gain", "audio.fade_in", "audio.fade_out", "audio.channels", "audio.concat", "audio.concat.crossfade", "audio.sample_rate") \
    + MIX_KEYS + (VOICE_CLEANUP_KEY,) + NOISE_REDUCTION_KEYS + DYNAMICS_KEYS
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
    "crossfade", "sample_rate", "extract"} (keys present only when asked). Values are range-checked; DEFAULT never switches anything on;
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
        elif key == "audio.extract":
            out["extract"] = _bool(v, key)
        elif key == "audio.concat.crossfade":
            out["crossfade"] = float(_number(v, key, CROSSFADE_RANGE[0], CROSSFADE_RANGE[1]))
        elif key == "audio.sample_rate":
            sr = _number(v, key, 8000, 96000)
            if int(sr) != float(sr) or int(sr) not in SAMPLE_RATES:
                raise EditRequirementError(f"audio.sample_rate must be one of {SAMPLE_RATES}, got {v!r}")
            out["sample_rate"] = int(sr)
        elif key == VOICE_CLEANUP_KEY:   # ADR-068: the level itself is the value (the contract has no default, none is guessed)
            if not isinstance(v, str) or v.strip().lower() not in VOICE_CLEANUP_LEVELS:
                raise EditRequirementError(f"audio.voice_cleanup must be one of {VOICE_CLEANUP_LEVELS} (the Skill's speech clean-up presets), got {v!r}")
            out["voice_cleanup"] = {"level": v.strip().lower()}
        elif key in MIX_KEYS or key in NOISE_REDUCTION_KEYS or key in DYNAMICS_KEYS:
            continue   # ADR-063: read together below (each operation from all of its keys)
    if "crossfade" in out and not out.get("concat"):
        raise EditRequirementError("audio.concat.crossfade needs audio.concat=true")
    vals = {k: m[k].value for k in asked}
    if "audio.mix" in vals:
        out["mix"] = _bool(vals["audio.mix"], "audio.mix")
    if "audio.mix.levels" in vals:
        if not out.get("mix"):
            raise EditRequirementError("audio.mix.levels needs audio.mix=true")
        out["mix_levels"] = parse_mix_levels(vals["audio.mix.levels"])
    if out.get("mix") and out.get("concat"):
        raise EditRequirementError("audio.mix and audio.concat both build the audio programme (sum vs join): refusing an ambiguous request")
    nr = [k for k in NOISE_REDUCTION_KEYS if k in vals]
    if nr:
        if "audio.noise_reduction.strength_db" not in vals:
            raise EditRequirementError("audio.noise_reduction needs audio.noise_reduction.strength_db (20..60 dB): the Skill's contract has no default and none is guessed")
        p_nr: Dict[str, Any] = {"strength_db": float(_number(vals["audio.noise_reduction.strength_db"], "audio.noise_reduction.strength_db", *NOISE_REDUCTION_STRENGTH))}
        mode = vals.get("audio.noise_reduction.mode", NOISE_REDUCTION_MODES[0])   # the contract's only mode; required by it, so always sent
        if not isinstance(mode, str) or mode.strip().lower() not in NOISE_REDUCTION_MODES:
            raise EditRequirementError(f"audio.noise_reduction.mode must be one of {NOISE_REDUCTION_MODES}, got {mode!r}")
        p_nr["mode"] = mode.strip().lower()
        out["noise_reduction"] = p_nr
        out["noise_reduction_keys"] = nr
    dyn = [k for k in DYNAMICS_KEYS if k in vals]
    if dyn:
        stages: Dict[str, Dict[str, float]] = {}
        for st, fields in DYNAMICS_STAGES.items():
            base = f"audio.dynamics.{st}"
            on = _bool(vals[base], base) if base in vals else None
            given = {f: vals[f"{base}.{f}"] for f in fields if f"{base}.{f}" in vals}
            if on is False:
                if given:
                    raise EditRequirementError(f"{base}=false with {base}.* fields: refusing an ambiguous request")
                continue
            if on or given:
                stages[st] = {f: float(_number(v, f"{base}.{f}", *fields[f])) for f, v in given.items()}
        if not stages:
            raise EditRequirementError("audio.dynamics needs at least one stage (audio.dynamics.gate / .compressor / .limiter)")
        out["dynamics"] = stages
        out["dynamics_keys"] = dyn
    return out


def parse_mix_levels(v: Any) -> List[Dict[str, Any]]:
    """`audio.mix.levels`: one entry per input, in input order — a gain in dB, `mute`, or {gain_db?, mute?} — as a list or a
    comma-separated string ("0,-12,mute") → the contract's [{gain_db, mute}]. The count is checked against the inputs by the
    decision (the inputs are known only then). The first input sets the mix's length, so it is never muted here."""
    items = [x.strip() for x in v.split(",")] if isinstance(v, str) else v
    if not isinstance(items, list) or not items or len(items) > MAX_MIX_INPUTS:
        raise EditRequirementError(f"audio.mix.levels must list one level per input (1..{MAX_MIX_INPUTS}): a gain in dB, mute, or {{gain_db, mute}}, got {v!r}")
    out: List[Dict[str, Any]] = []
    for i, x in enumerate(items):
        what = f"audio.mix.levels[{i}]"
        if isinstance(x, str) and x.lower() == "mute":
            lv = {"gain_db": 0.0, "mute": True}
        elif isinstance(x, dict):
            if not x or set(x) - {"gain_db", "mute"}:
                raise EditRequirementError(f"{what} must be {{gain_db, mute}}, got {x!r}")
            lv = {"gain_db": float(_number(x.get("gain_db", 0.0), what + ".gain_db", *MIX_GAIN_RANGE)), "mute": _bool(x.get("mute", False), what + ".mute")}
        else:
            lv = {"gain_db": float(_number(x, what, *MIX_GAIN_RANGE)), "mute": False}
        out.append(lv)
    if out[0]["mute"]:
        raise EditRequirementError("audio.mix.levels: the first input sets the mix's length and cannot be muted (put another input first)")
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


def mix_segments(inputs: List[str], durations: Dict[str, float]) -> Tuple[List[Dict[str, Any]], float]:
    """Timeline of an audio mix (ADR-063): every input starts at 0 s and is summed; the output lasts as long as the first input
    (the Skill's amix duration=first), so a longer input is cut at that length and a shorter one simply ends."""
    total = round(float(durations.get(inputs[0]) or 0.0), 3)
    segs = []
    for aid in inputs:
        d = round(min(float(durations.get(aid) or 0.0), total), 3)
        segs.append({"input": aid, "track": "A1", "source_range": [0.0, d], "timeline_range": [0.0, d]})
    return segs, total


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
    concat = next((op for op in aops if op.get("type") in PROGRAMME_OPS), None)   # ADR-063: a mix builds the programme the same way
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
        if concat is not None and concat.get("type") == "audio.mix" and sid == concat.get("output", PROGRAMME_AUDIO):
            expected = chans[0] if chans else None   # a mix keeps its first input's layout (the Skill's MIX folds the others into it)
        for op in aops:   # a channel operation on the subject sets what QA expects
            if op.get("asset") == sid and op.get("type") in ("audio.mono", "audio.stereo", "audio.downmix"):
                expected = 1 if op["type"] == "audio.mono" else 2
        row["channels"] = expected
    return out
