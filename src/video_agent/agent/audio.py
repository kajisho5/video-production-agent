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
  no silence is cut from a mixed input. Ducking (ADR-072, audio-production 0.4.6 MIX.duck): only on request, with the bed and the
  amount named explicitly — `audio.mix.duck.input` (the index of the bed among the mixed inputs, 1..; input 0 is the key it ducks
  under) + `audio.mix.duck.amount_db` (both required, no default) + optional `.threshold_db` / `.attack_ms` / `.release_ms` (the
  contract's names and ranges; omitted = the engine's own defaults).
- **Filler removal** (ADR-072, audio-production 0.4.6 FILLER_REMOVE): `audio.filler_remove=ja|en|ja,en` (the Skill's lexicon presets) or
  `true` with `audio.filler_remove.words` (the user's own words; also adds to the presets) + `.keep` (words never removed) + `.pad` /
  `.merge_gap` / `.declick_ms`. No default lexicon. The word timings are the transcript's word timestamps (transcription-skill,
  requested with word timestamps when the key is set), carried through this plan's silence cut onto the cut audio's timeline; a
  segment without word timestamps is a BLOCK (never a guessed timing). It runs per input right after the cut, before the programme
  (concat) and the chain; never on a mixed input (it would move that input against the others). What is removed is matched by the
  Skill, so planned durations after it are upper bounds and QA checks the delivered length against what the Skill reports it removed.
- **Voice clean-up** (ADR-068, audio-production-skill 0.4.3): `audio.voice_cleanup=light|medium|strong` → VOICE_CLEANUP (the engine's
  speech chain: highpass / de-esser / FFT denoise / compression / limiter by level). light / strong need engine release 1.13+; an
  older engine runs medium only and the Skill refuses the others with TOOL_ERROR `voice_level_unsupported` (never substituted).
- **Fixed order** after the cut (and the filler removal of each input): concat | mix → voice clean-up → noise reduction → gain → channels → dynamics → fade_in → fade_out → normalize
  (noise is estimated on the raw signal, dynamics act on the gained / laid-out signal, fades are not pumped by a compressor,
  loudness last so the delivered level is the normalised one).
- **Four stages:** the Skill supports 16 operation types (0.4.6); the adapter lowers every declared type; the planner generates the
  ones listed here (TRIM as a cut, CUT, NORMALIZE, GAIN, FADE_IN, FADE_OUT, MONO / STEREO / DOWNMIX, CONCAT, and since ADR-063
  MIX, NOISE_REDUCTION, DYNAMICS, since ADR-068 VOICE_CLEANUP, since ADR-072 FILLER_REMOVE and MIX.duck); SILENCE_REMOVE (the agent already subtracts margins itself) is executable through the
  adapter but not planned.
"""
from __future__ import annotations

import re
import unicodedata
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
    "audio.mix":      {"skill": "audio_mix",      "type": "MIX",      "params": ("levels", "duck"), "risk": "MEDIUM"},
    # ADR-072: audio-production 0.4.6's filler removal at the transcript's word timings (per input, right after its cut)
    "audio.filler_remove": {"skill": "audio_filler_remove", "type": "FILLER_REMOVE", "params": ("words", "lexicon", "pad", "merge_gap", "declick_ms"), "risk": "MEDIUM"},
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
SOURCE_OPS = ("audio.cut", "audio.filler_remove")   # per source asset, in this order, before the programme and the AUDIO_ORDER chain
CHANNEL_SKILLS = ("audio_mono", "audio_stereo", "audio_downmix")
# ADR-073: voice-enhance-skill's operations on this path (agent/voice.py holds their vocabulary); ISOLATE and BLEED_REDUCE write mono
VOICE_OP_TYPES = ("voice.bleed_reduce", "voice.dereverb", "voice.isolate")
VOICE_MONO_TYPES = ("voice.bleed_reduce", "voice.isolate")
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
# ADR-072: MIX.duck (audio-production 0.4.6): field -> (min, max); input and amount_db are required by the contract, the rest keep ffmpeg-skill's defaults
DUCK_FIELDS: Dict[str, Tuple[float, float]] = {"input": (1, MAX_MIX_INPUTS - 1), "amount_db": (0.0, 60.0), "threshold_db": (-60.0, 0.0), "attack_ms": (0.01, 2000.0),
                                               "release_ms": (0.01, 9000.0)}
DUCK_REQUIRED = ("input", "amount_db")
DUCK_KEYS = tuple(f"audio.mix.duck.{f}" for f in DUCK_FIELDS)
# ADR-072: FILLER_REMOVE (audio-production 0.4.6): the contract's lexicon presets, list sizes / lengths, ranges, and the most words it takes
FILLER_KEY = "audio.filler_remove"
FILLER_PRESETS = ("en", "ja")
FILLER_PAD_RANGE = (0.0, 0.5)
FILLER_MERGE_GAP_RANGE = (0.0, 2.0)
FILLER_DECLICK_RANGE = (0.0, 200.0)
FILLER_LEXICON_MAX_ITEMS = 256
FILLER_LEXICON_MAX_LENGTH = 64
FILLER_WORD_MAX_LENGTH = 128
FILLER_MAX_WORDS = 100000
FILLER_KEYS = (FILLER_KEY, "audio.filler_remove.words", "audio.filler_remove.keep", "audio.filler_remove.pad", "audio.filler_remove.merge_gap", "audio.filler_remove.declick_ms")
MIX_KEYS = ("audio.mix", "audio.mix.levels") + DUCK_KEYS
NOISE_REDUCTION_KEYS = ("audio.noise_reduction.strength_db", "audio.noise_reduction.mode")
DYNAMICS_KEYS = tuple(k for st, fields in DYNAMICS_STAGES.items() for k in (f"audio.dynamics.{st}",) + tuple(f"audio.dynamics.{st}.{f}" for f in fields))
REQUIREMENT_KEYS = ("audio.production", "audio.extract", "audio.gain", "audio.fade_in", "audio.fade_out", "audio.channels", "audio.concat", "audio.concat.crossfade", "audio.sample_rate") \
    + MIX_KEYS + (VOICE_CLEANUP_KEY,) + NOISE_REDUCTION_KEYS + DYNAMICS_KEYS + FILLER_KEYS
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
        elif key in MIX_KEYS or key in NOISE_REDUCTION_KEYS or key in DYNAMICS_KEYS or key in FILLER_KEYS:
            continue   # ADR-063 / ADR-072: read together below (each operation from all of its keys)
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
    duck = {k.rsplit(".", 1)[1]: vals[k] for k in DUCK_KEYS if k in vals}
    if duck:
        out["mix_duck"] = parse_duck(duck, out.get("mix", False), out.get("mix_levels"))
        out["mix_duck_keys"] = [k for k in DUCK_KEYS if k in vals]
    fr = {k: vals[k] for k in FILLER_KEYS if k in vals}
    if fr:
        out["filler_remove"] = parse_filler_remove(fr)
        out["filler_remove_keys"] = [k for k in FILLER_KEYS if k in vals]
        if out["filler_remove"] is None:
            out.pop("filler_remove")
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


def parse_duck(given: Dict[str, Any], mix: bool, levels: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """ADR-072: `audio.mix.duck.<field>` → the contract's MIX.duck {input, amount_db, threshold_db?, attack_ms?, release_ms?}. Opt-in only:
    the bed (`input`, its index among the mixed inputs, 1..7; input 0 is the key) and the amount are both named by the user — there is no
    default bed and no default amount. Whether the index exists is checked by the decision (the inputs are known only then)."""
    if not mix:
        raise EditRequirementError("audio.mix.duck.* needs audio.mix=true (ducking lowers one mixed input under input 0)")
    missing = [f for f in DUCK_REQUIRED if f not in given]
    if missing:
        raise EditRequirementError(f"audio.mix.duck needs {' and '.join('audio.mix.duck.' + f for f in missing)}: the bed and how far it is lowered are never guessed")
    out: Dict[str, Any] = {}
    for f, v in given.items():
        lo, hi = DUCK_FIELDS[f]
        x = _number(v, f"audio.mix.duck.{f}", lo, hi)
        if f == "input":
            if float(x) != int(x):
                raise EditRequirementError(f"audio.mix.duck.input must be an input index (1..{MAX_MIX_INPUTS - 1}), got {v!r}")
            out[f] = int(x)
        else:
            out[f] = float(x)
    if out["amount_db"] == 0.0:
        raise EditRequirementError("audio.mix.duck.amount_db=0 lowers nothing; leave ducking unset")
    if levels is not None and out["input"] < len(levels) and levels[out["input"]].get("mute"):
        raise EditRequirementError(f"audio.mix.duck.input={out['input']} names a muted input: the Skill ducks an unmuted bed only")
    return {f: out[f] for f in DUCK_FIELDS if f in out}


def _filler_text_list(v: Any, key: str) -> List[str]:
    """A list of words (a JSON list or a comma-separated string), each printable (any script: えー, um), at most the contract's
    length, at most its count; duplicates dropped (first kept)."""
    items = [x.strip() for x in v.split(",")] if isinstance(v, str) else v
    if not isinstance(items, list) or not items or len(items) > FILLER_LEXICON_MAX_ITEMS:
        raise EditRequirementError(f"{key} must list 1..{FILLER_LEXICON_MAX_ITEMS} words (a list or comma-separated), got {v!r}")
    out: List[str] = []
    for i, x in enumerate(items):
        if not isinstance(x, str) or not x.strip() or len(x) > FILLER_LEXICON_MAX_LENGTH or any(unicodedata.category(c).startswith("C") for c in x):
            raise EditRequirementError(f"{key}[{i}] must be a printable word of 1..{FILLER_LEXICON_MAX_LENGTH} characters, got {x!r}")
        if x not in out:
            out.append(x)
    return out


def parse_filler_remove(vals: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """ADR-072: `audio.filler_remove` (+ refinements) → {"lexicon": {presets?, words?, keep?}, pad?, merge_gap?, declick_ms?}, or None for an
    explicit `false`. The lexicon has no default (the contract requires the caller to choose): the value names the Skill's presets
    (ja | en | ja,en) or is `true`, which needs the user's own `audio.filler_remove.words`."""
    sw = vals.get(FILLER_KEY)
    refinements = [k for k in vals if k != FILLER_KEY]
    if sw is None:
        raise EditRequirementError(f"{refinements[0]} needs {FILLER_KEY} (ja | en | ja,en, or true with {FILLER_KEY}.words)")
    presets: List[str] = []
    if isinstance(sw, bool) or (isinstance(sw, str) and sw.strip().lower() in ("true", "false")):
        on = _bool(sw, FILLER_KEY)
        if not on:
            if refinements:
                raise EditRequirementError(f"{FILLER_KEY}=false with {', '.join(refinements)}: refusing an ambiguous request")
            return None
    else:
        raw = [x.strip().lower() for x in sw.split(",")] if isinstance(sw, str) else sw
        if not isinstance(raw, list) or not raw or not all(isinstance(x, str) and x.strip().lower() in FILLER_PRESETS for x in raw):
            raise EditRequirementError(f"{FILLER_KEY} must name the Skill's lexicon presets ({' | '.join(FILLER_PRESETS)}, comma-separated) or be true with {FILLER_KEY}.words, got {sw!r}")
        presets = sorted({x.strip().lower() for x in raw})
    lexicon: Dict[str, Any] = {}
    if presets:
        lexicon["presets"] = presets
    for part in ("words", "keep"):
        k = f"{FILLER_KEY}.{part}"
        if k in vals:
            lexicon[part] = _filler_text_list(vals[k], k)
    if not lexicon.get("presets") and not lexicon.get("words"):
        raise EditRequirementError(f"{FILLER_KEY}=true needs {FILLER_KEY}.words (the fillers to remove): there is no default lexicon")
    if lexicon.get("keep") and not lexicon.get("presets") and set(lexicon["words"]) <= set(lexicon["keep"]):
        raise EditRequirementError(f"{FILLER_KEY}.keep removes every word of {FILLER_KEY}.words: the lexicon would be empty")
    out: Dict[str, Any] = {"lexicon": lexicon}
    for part, rng in (("pad", FILLER_PAD_RANGE), ("merge_gap", FILLER_MERGE_GAP_RANGE), ("declick_ms", FILLER_DECLICK_RANGE)):
        k = f"{FILLER_KEY}.{part}"
        if k in vals:
            out[part] = float(_number(vals[k], k, *rng))
    return out


def transcript_words(segments: List[Dict[str, Any]]) -> Tuple[Optional[List[Dict[str, Any]]], List[str], str]:
    """ADR-072: the word timings of one transcript (transcription-skill/transcript/0.1 segments) on the source's timeline →
    (words sorted by time, ids of segments without word timestamps, problem). Only start / end / text / confidence are kept. A
    segment whose `words` is null (no word timestamps) is reported, never filled in; a word without times or with text the contract
    does not take (control characters, more than 128 characters) is a problem (None, ..., why). Zero-length words and words whose text
    is empty are left out (nothing to cut / nothing to match)."""
    untimed: List[str] = []
    out: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments or []):
        ws = seg.get("words") if isinstance(seg, dict) else None
        if ws is None:
            untimed.append(str((seg or {}).get("id") or i) if isinstance(seg, dict) else str(i))
            continue
        for w in ws:
            if not isinstance(w, dict) or not all(isinstance(w.get(k), (int, float)) and not isinstance(w.get(k), bool) for k in ("start", "end")):
                return None, untimed, f"segment {(seg or {}).get('id') or i}: a word without start / end times"
            text = w.get("text")
            if not isinstance(text, str) or len(text) > FILLER_WORD_MAX_LENGTH or any(ord(c) < 32 for c in text):
                return None, untimed, f"segment {(seg or {}).get('id') or i}: word text {str(text)[:40]!r} is not a printable string of at most {FILLER_WORD_MAX_LENGTH} characters"
            s_, e_ = float(w["start"]), float(w["end"])
            if s_ < 0 or not e_ > s_ or not text.strip():
                continue
            rec: Dict[str, Any] = {"start": round(s_, 6), "end": round(e_, 6), "text": text}
            c = w.get("confidence")
            if isinstance(c, (int, float)) and not isinstance(c, bool) and 0.0 <= float(c) <= 1.0:
                rec["confidence"] = float(c)
            out.append(rec)
    return sorted(out, key=lambda w: (w["start"], w["end"])), untimed, ""


def words_after_cut(words: List[Dict[str, Any]], remove: List[List[float]], duration: float) -> Tuple[List[Dict[str, Any]], int]:
    """ADR-072: word timings on the source's timeline → the same words on the timeline of the source after `remove` was cut out of it
    (the Skill's CUT joins the kept ranges in order). A word not wholly inside one kept range is left out (count returned): it is cut
    (partly) already, so FILLER_REMOVE never touches it."""
    keep: List[List[float]] = []
    t = 0.0
    for s_, e_ in sorted(([float(a), float(b)] for a, b in remove), key=lambda r: r[0]):
        if s_ > t:
            keep.append([t, s_])
        t = max(t, e_)
    if t < float(duration):
        keep.append([t, float(duration)])
    out: List[Dict[str, Any]] = []
    dropped = 0
    offset = 0.0
    spans = []
    for ks, ke in keep:
        spans.append((ks, ke, offset))
        offset += ke - ks
    for w in words:
        span = next(((ks, ke, off) for ks, ke, off in spans if w["start"] >= ks - 1e-9 and w["end"] <= ke + 1e-9), None)
        if span is None:
            dropped += 1
            continue
        ks, _, off = span
        out.append(dict(w, start=round(max(0.0, off + w["start"] - ks), 6), end=round(off + w["end"] - ks, 6)))
    return out, dropped


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
              {op["asset"] for op in aops if op.get("type") == "audio.loudness" and op.get("input") is not None} | \
              {op["asset"] for op in aops if op.get("type") in VOICE_OP_TYPES}   # ADR-073 (bleed reduction's subject is the programme: its inputs are the mix's)
    mono_sources = {op["asset"] for op in aops if op.get("type") in VOICE_MONO_TYPES} | \
                   {a for op in aops if op.get("type") == "voice.bleed_reduce" for a in op.get("inputs") or []}
    out: Dict[str, Dict[str, Any]] = {}
    concat = next((op for op in aops if op.get("type") in PROGRAMME_OPS), None)   # ADR-063: a mix builds the programme the same way
    for aid in on_path:
        if aid in assets:
            dur = float((assets[aid].get("technical") or {}).get("duration") or 0.0)
            cut = next((op for op in aops if op.get("type") == "audio.cut" and op.get("asset") == aid), None)
            if cut is not None:
                dur = kept_after_cut(dur, cut.get("remove") or [])
            out[aid] = {"sources": [aid], "duration": dur}   # ADR-072: an upper bound when the source's fillers are removed (see filler_sources)
    if concat is not None:
        out[concat.get("output", PROGRAMME_AUDIO)] = {"sources": list(concat.get("inputs") or []), "duration": float(concat.get("timeline_duration") or 0.0)}
        for aid in concat.get("inputs") or []:
            out.pop(aid, None)   # the inputs are consumed by the programme
    fillers = {op["asset"] for op in aops if op.get("type") == "audio.filler_remove"}
    for sid, row in out.items():
        row["filler_sources"] = [a for a in row["sources"] if a in fillers]   # ADR-072: what the Skill removes there is known only after it ran
        chans = [1 if a in mono_sources else audio_channels((assets.get(a) or {}).get("technical") or {}) for a in row["sources"]]   # ADR-073: an isolated / bleed-reduced source is mono
        known_chans = [c for c in chans if c is not None]
        expected: Optional[int] = max(known_chans) if chans and len(known_chans) == len(chans) else None
        if concat is not None and concat.get("type") == "audio.mix" and sid == concat.get("output", PROGRAMME_AUDIO):
            expected = chans[0] if chans else None   # a mix keeps its first input's layout (the Skill's MIX folds the others into it)
        for op in aops:   # a channel operation on the subject sets what QA expects
            if op.get("asset") == sid and op.get("type") in ("audio.mono", "audio.stereo", "audio.downmix"):
                expected = 1 if op["type"] == "audio.mono" else 2
        row["channels"] = expected
    return out
