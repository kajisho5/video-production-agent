"""Voice enhancement on the audio production path (ADR-073): what the agent plans through voice-enhance-skill, and where in the
audio chain it runs.

    Requirement (`--set voice.<op>=true`, explicit only) → Decision (always CONFIRM: every op is EXPERIMENTAL) → step → IR
    audio operation `voice.<op>` → compiler → voice-enhance-skill (its own engine worker; every media step through the media engine Skill)

Vocabulary, decisions and arithmetic only, no execution:

- **Requirements** (`voice.` keys; never from a profile, a brief or a default -- the user opts in on the request, `--set`):
  `voice.isolate=true` (+ `voice.isolate.atten_lim_db`, 0 < dB <= 100: keep some background), `voice.dereverb=true` (+ `.taps` 1..50,
  `.delay` 1..20, `.iterations` 1..20, nara_wpe's own defaults otherwise), `voice.bleed_reduce=true` (+ `.dominance_db` 0..30). Each
  needs `audio.production=true` (the cleaned speech is the audio deliverable; putting a stem back under the picture is not wired yet),
  and `voice.bleed_reduce` needs `audio.mix=true` (each mixed input one time-aligned microphone). A refinement without its switch, a
  value out of range, an unknown `voice.*` key or a switch that is not true / false is refused before analysis.
- **Where in the chain** (per input, then the programme): cut → filler removal → **bleed reduction (all mixed inputs at once) →
  dereverberation → isolation** → concat | mix → voice clean-up → noise reduction → gain → channels → dynamics → fades → loudness.
  After the cut because voice-enhance's ISOLATE stem ends ~30 ms early (deep-filter -D) and audio-production's CUT refuses a range past
  the end of its input; before everything audio-production does to the level and the spectrum, because the engines model the recorded
  signal (WPE is linear prediction of the room; DeepFilterNet was trained on raw noisy speech; crosstalk needs the tracks exactly as
  recorded and aligned). Bleed reduction first, then WPE, then the non-linear mask (voice-enhance's own SKILL.md order).
- **Rules the plan checks before any engine runs** (each a BLOCK with voice-enhance-skill's own error code where it has one):
  ISOLATE needs a 48 kHz stream unless the located media engine declares `audio --sample-rate` (the Skill's RATE_REFUSAL code, never a
  private resample); an engine or model that is not installed blocks with the Skill's code (`ENGINE_UNAVAILABLE` / `MODEL_NOT_FOUND`)
  and its install line, never a download; BLEED_REDUCE needs the mix (2..8 inputs, no cut: a mixed input keeps its own timeline), one
  sample rate (RATE_REFUSAL) and distinct file names (`INVALID_ARGUMENT`: the Skill names each stem after its file).
  Without voice-enhance-skill the step has no tool and the plan's handoff names the package (ADR-051).
- **Shape of the result**: ISOLATE and BLEED_REDUCE write mono (QA expects one channel downstream unless a later channel operation says
  otherwise); ISOLATE at 48 kHz. Every op keeps the timeline (sample-aligned), ISOLATE ending up to 40 ms early, which the audio QA's
  duration tolerance absorbs."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from ..models import Requirement
from ..tools.voice_enhance.adapter import DOCTOR_CODES, ISOLATE_RATE, MAX_TRACKS, PARAMETERS, RATE_REFUSAL, TOOL_BLEED_REDUCE, TOOL_DEREVERB, TOOL_ISOLATE
from .audio import PROGRAMME_AUDIO, SWITCH as AUDIO_SWITCH
from .editing import EditRequirementError, _bool

PREFIX = "voice."
PACKAGE = "voice-enhance"
# IR operation type → production skill, voice-enhance operation type (its doctor / capability `voice-enhance:<TYPE>`), tool, parameters
OPERATIONS: Dict[str, Dict[str, Any]] = {
    "voice.bleed_reduce": {"skill": "voice_bleed_reduce", "type": "BLEED_REDUCE", "tool": TOOL_BLEED_REDUCE, "params": tuple(PARAMETERS[TOOL_BLEED_REDUCE]), "risk": "MEDIUM"},
    "voice.dereverb": {"skill": "voice_dereverb", "type": "DEREVERB", "tool": TOOL_DEREVERB, "params": tuple(PARAMETERS[TOOL_DEREVERB]), "risk": "MEDIUM"},
    "voice.isolate": {"skill": "voice_isolate", "type": "ISOLATE", "tool": TOOL_ISOLATE, "params": tuple(PARAMETERS[TOOL_ISOLATE]), "risk": "MEDIUM"},
}
VOICE_ORDER = ("voice.bleed_reduce", "voice.dereverb", "voice.isolate")   # bleed (all tracks) → WPE → mask, per input after the cut / filler removal
PER_INPUT = ("voice.dereverb", "voice.isolate")
SKILL_OF = {k: v["skill"] for k, v in OPERATIONS.items()}
MONO_OUTPUT = ("voice.bleed_reduce", "voice.isolate")
REFINEMENTS: Dict[str, Any] = {f"{op}.{name}": (op, name, spec) for op, o in OPERATIONS.items() for name, spec in PARAMETERS[o["tool"]].items()}
REQUIREMENT_KEYS = tuple(OPERATIONS) + tuple(REFINEMENTS)
# always CONFIRM (floor CONFIRM: the explicit requirement does not waive it -- every op is EXPERIMENTAL in voice-enhance-skill 0.1.0)
APPROVAL_KEYS = {op: (f"{op}.approval", "CONFIRM") for op in OPERATIONS}
ISOLATE_SHORTFALL_S = 0.04   # voice-enhance ADR-007: the isolate stem ends 1440..1920 samples (30..40 ms) before its input


def _num(v: Any, key: str, spec: Any) -> Any:
    lo, hi, integer, open_lo = spec
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            raise EditRequirementError(f"{key} must be a number, got {v!r}")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        raise EditRequirementError(f"{key} must be a finite number, got {v!r}")
    if integer and float(v) != int(v):
        raise EditRequirementError(f"{key} must be an integer, got {v!r}")
    if (v <= lo if open_lo else v < lo) or v > hi:
        raise EditRequirementError(f"{key} must be within {'(' if open_lo else '['}{lo:g}, {hi:g}] (voice-enhance-skill's own range), got {v!r}")
    return int(v) if integer else float(v)


def parse_voice_requirements(m: Dict[str, Requirement]) -> Dict[str, Any]:
    """Requirement map → {"ops": {op type: params}, "keys": {op type: [requirement keys]}, "requirements": [Requirement]}. Only USER
    requirements count, and any other provenance of a `voice.*` key is refused (the ops are opt-in per request, never a profile default
    or a brief's reading). A switch set to false asks for nothing (its refinements are then refused as ambiguous)."""
    out: Dict[str, Any] = {"ops": {}, "keys": {}, "requirements": []}
    given = {k: r for k, r in m.items() if k.startswith(PREFIX)}
    if not given:
        return out
    unknown = sorted(k for k in given if k not in REQUIREMENT_KEYS)
    if unknown:
        raise EditRequirementError(f"unknown requirement key(s): {', '.join(unknown)}; voice.* keys are {', '.join(REQUIREMENT_KEYS)}")
    not_user = sorted(k for k, r in given.items() if r.provenance != "USER")
    if not_user:
        raise EditRequirementError(f"{', '.join(not_user)}: voice enhancement is opt-in on the request (--set), never from a profile, a brief or a default")
    sw = m.get(AUDIO_SWITCH)
    production = sw is not None and sw.provenance != "DEFAULT" and _bool(sw.value, AUDIO_SWITCH)
    if not production:
        raise EditRequirementError(f"{sorted(given)[0]} needs {AUDIO_SWITCH}=true: voice enhancement runs on the audio production path "
                                   "(the cleaned speech is the audio deliverable; putting a stem back under the picture is not wired yet)")
    for op in OPERATIONS:
        refs = [k for k in REFINEMENTS if REFINEMENTS[k][0] == op and k in given]
        on = _bool(given[op].value, op) if op in given else None
        if not on:
            if refs:
                raise EditRequirementError(f"{refs[0]} needs {op}=true" if on is None else f"{op}=false with {', '.join(refs)}: refusing an ambiguous request")
            continue
        params = {REFINEMENTS[k][1]: _num(given[k].value, k, REFINEMENTS[k][2]) for k in refs}
        out["ops"][op] = params
        out["keys"][op] = [op] + refs
        out["requirements"] += [given[op]] + [given[k] for k in refs]
    if "voice.bleed_reduce" in out["ops"]:
        mix = m.get("audio.mix")
        if mix is None or mix.provenance == "DEFAULT" or not _bool(mix.value, "audio.mix"):
            raise EditRequirementError("voice.bleed_reduce needs audio.mix=true: it reduces the cross-talk between time-aligned microphone tracks, each mixed input one microphone")
    return out


def check_voice_keys(user_requirements: Dict[str, Any]) -> None:
    """The `--set` gate for `voice.*` (before any analysis runs): known keys, values in range, the switches they need."""
    if not any(k.startswith(PREFIX) for k in user_requirements or {}):
        return
    reqs = [Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in user_requirements.items() if k.startswith(PREFIX) or k in (AUDIO_SWITCH, "audio.mix")]
    parse_voice_requirements({r.key: r for r in reqs})


def ir_voice_operation(op_type: str, subject: str, params: Dict[str, Any], decision_ids: List[str], scope: Optional[Dict[str, float]] = None, **refs: Any) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"type": op_type, "asset": subject}
    rec.update(refs)
    for k in OPERATIONS[op_type]["params"]:
        if params.get(k) is not None:
            rec[k] = params[k]
    if scope is not None:
        rec["temporal_scope"] = {"start": round(float(scope["start"]), 3), "end": round(float(scope["end"]), 3)}
    rec["decision_ids"] = list(decision_ids)
    return rec


def _sample_rate(technical: Dict[str, Any]) -> Optional[int]:
    try:
        return int(((technical or {}).get("audio") or {}).get("sample_rate"))
    except (TypeError, ValueError):
        return None


def _words(params: Dict[str, Any]) -> str:
    return ", ".join(f"{k} {v:g}" for k, v in params.items()) if params else "the engine's defaults"


def decide_voice(eng: Any, vreq: Dict[str, Any], analysis: Any, audio_assets: List[Any], mix_ok: bool, mixed_inputs: List[str], caps: Dict[str, Any],
                 evidence_of_asset: Callable[[str], List[str]], approval_for: Callable[..., Dict[str, Any]], cap_block: Callable[[str, str], Any]) -> None:
    """One decision per requested op and input (BLEED_REDUCE: one for all mixed inputs). TRANSFORM with a CONFIRM floor, or BLOCK with the
    reason (and voice-enhance-skill's own error code in `params.code` where the Skill has one). Without the package the TRANSFORM stands and
    cap_block adds the BLOCK whose step the ADR-051 handoff names."""
    ops = vreq["ops"]
    pkg = caps.get(PACKAGE)
    pkg_ok = pkg is not None and getattr(pkg, "status", "") == "AVAILABLE"

    def op_cap(op: str) -> Dict[str, Any]:
        c = caps.get(f"{PACKAGE}:{OPERATIONS[op]['type']}")
        return {"status": getattr(c, "status", "MISSING") if c is not None else "MISSING", "evidence": dict(getattr(c, "evidence", {}) or {}) if c is not None else {}}

    def reqs_of(op: str) -> List[Requirement]:
        return [r for r in vreq["requirements"] if r.key == op or r.key.startswith(op + ".")]

    def block(op: str, subject: str, inputs: List[str], why: str, code: Optional[str], extra: Optional[Dict[str, Any]] = None) -> None:
        ev = [r.id for r in reqs_of(op)] + [e for a in inputs for e in evidence_of_asset(a)]
        if code in DOCTOR_CODES and extra and extra.get("from_doctor"):
            ev.append(f"capability:{PACKAGE}:{OPERATIONS[op]['type']}")
        params = {"asset_id": subject, **({"inputs": inputs} if op == "voice.bleed_reduce" else {}), **({"code": code} if code else {}),
                  **{k: v for k, v in (extra or {}).items() if k != "from_doctor"}}
        eng.decide(subject=op, type="BLOCK", decision=f"BLOCK: {op} on {subject}", reason=why, confidence=1.0, evidence=sorted(set(ev), key=ev.index), risk="HIGH", approval="BLOCK",
                   provenance="USER", params=params, requirements=reqs_of(op))

    def engine_problem(op: str) -> Optional[Dict[str, Any]]:
        """When voice-enhance-skill is installed but this op's engine is not ready: the Skill's own code, detail and install line."""
        if not pkg_ok:
            return None
        c = op_cap(op)
        if c["status"] == "AVAILABLE":
            return None
        ev = c["evidence"]
        code = str(ev.get("code") or "ENGINE_UNAVAILABLE")
        why = (f"voice-enhance-skill reports {code} for {OPERATIONS[op]['type'].lower()}: engine {ev.get('engine') or '?'} is {ev.get('engine_status') or 'not ready'}"
               + (f" ({ev['detail']})" if ev.get("detail") else "") + (f"; install: {ev['install']}" if ev.get("install") else "")
               + "; nothing is downloaded or installed without the user's go-ahead")
        return {"why": why, "code": code, "extra": {"engine": ev.get("engine"), "install": ev.get("install") or None, "from_doctor": True}}

    def transform(op: str, subject: str, inputs: List[str], what: str, reason: str) -> None:
        params = dict(ops[op])
        ev = [r.id for r in reqs_of(op)] + [e for a in inputs for e in evidence_of_asset(a)]
        eng.decide(subject=op, type="TRANSFORM", decision=what, reason=reason, confidence=1.0, evidence=sorted(set(ev), key=ev.index), risk=OPERATIONS[op]["risk"],
                   approval=approval_for(op, floor="CONFIRM", keys=APPROVAL_KEYS), provenance="USER",
                   params={"asset_id": subject, **({"inputs": inputs} if op == "voice.bleed_reduce" else {}), **params, "lifecycle": "EXPERIMENTAL"},
                   requirements=reqs_of(op), serves_intent=None)
        cap_block(OPERATIONS[op]["skill"], f"capability.{OPERATIONS[op]['skill']}")

    experimental = "voice-enhance-skill marks it EXPERIMENTAL (measured on synthetic fixtures only, not yet on real Japanese speech), so it always needs confirmation and the result should be listened to"
    if "voice.bleed_reduce" in ops:
        op = "voice.bleed_reduce"
        by_id = {a.id: a for a in audio_assets}
        ids = [a for a in mixed_inputs if a in by_id]
        rates = {a: _sample_rate(by_id[a].technical) for a in ids}
        stems = [os.path.splitext(os.path.basename(by_id[a].path))[0] for a in ids]
        problem = engine_problem(op)
        if not mix_ok or len(ids) < 2:
            block(op, PROGRAMME_AUDIO, ids, "bleed reduction runs on the time-aligned microphone tracks of the mix (audio.mix with 2.."
                  f"{MAX_TRACKS} inputs), and no mix was decided", None)
        elif len(ids) > MAX_TRACKS:
            block(op, PROGRAMME_AUDIO, ids, f"{len(ids)} tracks: voice-enhance-skill reduces cross-talk between at most {MAX_TRACKS}", "INVALID_ARGUMENT")
        elif None in rates.values() or len(set(rates.values())) != 1:
            at = ", ".join(f"{a} {rates[a] or 'unknown'} Hz" for a in ids)
            block(op, PROGRAMME_AUDIO, ids, f"the tracks are at {at}: bleed reduction needs one rate, and converting one "
                  "needs the media engine's audio --sample-rate (never a private resample)", RATE_REFUSAL, {"needs_flag": "sample_rate"})
        elif len(set(stems)) != len(stems):
            block(op, PROGRAMME_AUDIO, ids, "two tracks share a file name: voice-enhance-skill names each cleaned track after its file (<name>_a0.wav)", "INVALID_ARGUMENT")
        elif problem is not None:
            block(op, PROGRAMME_AUDIO, ids, problem["why"], problem["code"], problem["extra"])
        else:
            transform(op, PROGRAMME_AUDIO, ids, f"reduce the cross-talk between {' + '.join(ids)} ({_words(ops[op])})",
                      f"user asked for bleed reduction ({reqs_of(op)[0].source}); voice-enhance-skill's crosstalk engine (signal processing, no weights) estimates how much "
                      f"of each talker every other microphone hears and removes it linearly, on the mixed inputs as recorded (time-aligned, uncut), before the mix; "
                      f"one mono track per input; {experimental}")
    for op in PER_INPUT:
        if op not in ops:
            continue
        for a in audio_assets:
            problem = engine_problem(op)
            rate = _sample_rate(a.technical)
            if op == "voice.isolate" and rate != ISOLATE_RATE and not op_cap(op)["evidence"].get("resample"):
                block(op, a.id, [a.id], f"the audio of {a.id} is {rate or 'of unknown rate'}{' Hz' if rate else ''}: DeepFilterNet3 works at {ISOLATE_RATE} Hz, and converting "
                      f"the stream needs the media engine's audio --sample-rate, which the located engine does not declare (voice-enhance-skill refuses with "
                      f"{RATE_REFUSAL}); convert the input to 48 kHz first -- the agent never resamples privately", RATE_REFUSAL,
                      {"needs_flag": "sample_rate", "sample_rate": rate})
            elif problem is not None:
                block(op, a.id, [a.id], problem["why"], problem["code"], problem["extra"])
            elif op == "voice.dereverb":
                transform(op, a.id, [a.id], f"remove room reverberation from {a.id} ({_words(ops[op])})",
                          f"user asked for dereverberation ({reqs_of(op)[0].source}); voice-enhance-skill runs WPE (nara_wpe, signal processing, no weights) at the "
                          "stream's own rate and channels, after the cut and before every level / spectrum operation; the length is kept; " + experimental)
            else:
                transform(op, a.id, [a.id], f"isolate the voice of {a.id} ({_words(ops[op])})",
                          f"user asked for voice isolation ({reqs_of(op)[0].source}); voice-enhance-skill runs DeepFilterNet3 (local, offline) and writes a 48 kHz mono "
                          "stem that removes everything that is not speech, music under the voice included; after the cut and the dereverberation, before every "
                          "level / spectrum operation; the stem ends up to 40 ms early; " + experimental)
