"""Who spoke when (ADR-086): the `speakers.*` requirements that ask diarization-skill for anonymous speaker turns.

    Requirement (`--set speakers.diarize=true`, explicit only) → analysis kind `speaker_turns` (diarization-skill `diarize`, a measurement)
    → Observation `speaker_turns` (OBSERVED) + SpeakerEvents on the asset's timeline

Vocabulary and checks only, no execution:

- **Requirements** (`speakers.` keys; `--set` only, never a profile, a brief or a default): `speakers.diarize=true` asks for the
  measurement on every input with audio. Optional hints, diarization-skill's own: `speakers.num_speakers` (exact count, when known),
  or `speakers.min_speakers` / `speakers.max_speakers` (a range); integers >= 1, the exact count excludes the range, min <= max (the
  Skill's check_speaker_counts). `speakers.audio_stream` (0-based) picks another audio track. A refinement without the switch, an
  unknown `speakers.*` key, a switch that is not true / false or a value out of range is refused before analysis.
- **What it is not**: the ids are anonymous (`SPEAKER_00` = the first voice heard), never a name; nothing here decides an edit.
  Putting the ids into the transcript / subtitles is a separate, explicit step (roadmap step 8).
- **Without diarization-skill** (or with DIARIZE not ready: pyannote.audio, torch or the pipeline directory missing, as its doctor
  says) the measurement is not run: the plan records the ask and BLOCKs with a handoff naming diarization-skill and its install
  (agent/handoffs.py `decide_measurement_handoffs`). A measurement that ran and failed blocks with diarization-skill's own code."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..tools.diarization.adapter import KIND, PARAMETERS
from .editing import EditRequirementError

PREFIX = "speakers."
SWITCH = "speakers.diarize"
SKILL = "speaker_diarization"
REQUIREMENT_KEYS = (SWITCH,) + tuple(PREFIX + p for p in PARAMETERS)
_ON = (True, "true", "yes", "on", "1")
_OFF = (False, None, "false", "no", "off", "0", "none")


def _int(v: Any, key: str, lo: int) -> int:
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        v = int(v.strip())
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise EditRequirementError(f"{key} must be an integer >= {lo} (diarization-skill's own rule), got {v!r}")
    return v


def parse_speaker_requirements(user_requirements: Dict[str, Any]) -> Tuple[bool, Dict[str, int]]:
    """(requested, the diarize parameters) from explicit requirements; ValueError (EditRequirementError) on anything the Skill would refuse."""
    given = {k: v for k, v in (user_requirements or {}).items() if k.startswith(PREFIX)}
    unknown = sorted(k for k in given if k not in REQUIREMENT_KEYS)
    if unknown:
        raise EditRequirementError(f"unknown requirement key(s): {', '.join(unknown)}; known: {', '.join(REQUIREMENT_KEYS)}")
    if not given:
        return False, {}
    sw = given.get(SWITCH)
    sw = sw.strip().lower() if isinstance(sw, str) else sw
    if SWITCH in given and sw not in _ON + _OFF:
        raise EditRequirementError(f"{SWITCH} is a switch (true / false), got {given[SWITCH]!r}")
    on = SWITCH in given and sw in _ON
    params: Dict[str, int] = {}
    for p in PARAMETERS:
        key = PREFIX + p
        if key in given and given[key] is not None:
            if not on:
                raise EditRequirementError(f"{key} refines {SWITCH}, which is not set to true")
            params[p] = _int(given[key], key, 0 if p == "audio_stream" else 1)
    if "num_speakers" in params and ("min_speakers" in params or "max_speakers" in params):
        raise EditRequirementError("speakers.num_speakers excludes speakers.min_speakers / speakers.max_speakers (diarization-skill's rule)")
    if "min_speakers" in params and "max_speakers" in params and params["min_speakers"] > params["max_speakers"]:
        raise EditRequirementError("speakers.min_speakers must be <= speakers.max_speakers")
    return on, params


def check_speaker_keys(user_requirements: Dict[str, Any]) -> None:
    parse_speaker_requirements(user_requirements)


def speaker_kind(user_requirements: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, int]]]:
    """(analysis kind, its parameters) when the request asks for speaker turns, else None."""
    on, params = parse_speaker_requirements(user_requirements)
    return (KIND, params) if on else None
