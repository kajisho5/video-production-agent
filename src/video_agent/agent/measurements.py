"""Measurements asked for by name (roadmap step 33): text on screen, the shot table and speech regions. The ownership audit of 2026-11-12 found
these three tools already shipping in their Skills with no SkillSpec here (ADR-090 addendum 1, ADR-103 addendum 1).

    --set text.detect=true            [text.every=<s> text.lang=<jpn+eng> text.min_score=<0..1>]   → analysis kind screen_text   (detection/text)
    --set shots.detect=true           [shots.threshold=<x> shots.min_shot_frames=<n>]              → analysis kind shot_table    (detection/shots)
    --set speech_regions.detect=true                                                               → analysis kind speech_regions (media-analysis/vad)

- **Explicit only.** These keys come from `--set` (or the MCP `set`). They are never brief keys, stored preferences or profile defaults. Without
  them nothing new is measured, and the default providers do not change: the engine's `scenes` stays the fast cut list (`scene_detection`).
- **Measurements only.** Each result is an OBSERVED observation of its kind on every input with a video stream (text, shots) or an audio stream
  (speech regions). Nothing here edits, decides or names anyone: what the text says or where the shots fall is evidence for later steps.
- **Checked before analysis.** An unknown key, a refinement without its switch, a switch that is not true / false, or a value the Skill would
  refuse is refused before anything runs.
- **Without the Skill** (or its engine: Tesseract for text, PySceneDetect for shots, the doctor says which) the measurement is not run. The plan
  records the ask and BLOCKs with a handoff naming the package and how to install it (agent/handoffs.py `decide_measurement_handoffs`), the
  route `speakers.diarize` takes. A measurement that ran and failed blocks with the Skill's own code."""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from .editing import EditRequirementError

_ON = (True, "true", "yes", "on", "1")
_OFF = (False, None, "false", "no", "off", "0", "none")
LANG_RE = re.compile(r"^[a-z]{3}(?:_[a-z]{3,4})?(?:\+[a-z]{3}(?:_[a-z]{3,4})?){0,7}$")   # detection-skill's --lang: Tesseract codes joined by +
TEXT_EVERY_DEFAULT = 1.0   # one frame a second: on-screen text (a lower third, a slide) stays up longer than that


def _number(v: Any, key: str, lo: float, hi: float, lo_open: bool = False) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise EditRequirementError(f"{key} must be a number, got {v!r}")
    if isinstance(v, bool) or f != f or (f <= lo if lo_open else f < lo) or f > hi:
        raise EditRequirementError(f"{key} must be in {'(' if lo_open else '['}{lo:g}, {hi:g}], got {v!r}")
    return f


def _int(v: Any, key: str, lo: int, hi: int) -> int:
    if isinstance(v, str) and v.strip().isdigit():
        v = int(v.strip())
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise EditRequirementError(f"{key} must be an integer in [{lo}, {hi}], got {v!r}")
    return v


def _text(v: Any, key: str) -> str:
    if not isinstance(v, str) or not LANG_RE.match(v.strip()):
        raise EditRequirementError(f"{key} must be Tesseract language codes joined by + (e.g. jpn+eng, jpn_vert), got {v!r}")
    return v.strip()


# switch → (registry skill, analysis kind, what is measured, {refinement key: (analysis parameter, parser)})
MEASUREMENTS: Dict[str, Tuple[str, str, str, Dict[str, Tuple[str, Any]]]] = {
    "text.detect": ("text_detection", "screen_text", "text on screen (OCR lines with their boxes) on every input with video", {
        "text.every": ("text_every", lambda v, k: _number(v, k, 0.0, 60.0, lo_open=True)),
        "text.lang": ("text_lang", _text),
        "text.min_score": ("text_min_score", lambda v, k: _number(v, k, 0.0, 1.0)),
    }),
    "shots.detect": ("shot_detection", "shot_table", "the frame-accurate shot table on every input with video", {
        "shots.threshold": ("shot_threshold", lambda v, k: _number(v, k, 0.0, 255.0, lo_open=True)),
        "shots.min_shot_frames": ("shot_min_frames", lambda v, k: _int(v, k, 1, 100000)),
    }),
    "speech_regions.detect": ("speech_region_analysis", "speech_regions", "speech regions (a voice-activity model) on every input with audio", {}),
}
PREFIXES = tuple(sorted({k.split(".", 1)[0] + "." for k in MEASUREMENTS}))
REQUIREMENT_KEYS = tuple(sorted(set(MEASUREMENTS) | {r for m in MEASUREMENTS.values() for r in m[3]}))


def _switch(given: Dict[str, Any], key: str) -> bool:
    v = given[key].strip().lower() if isinstance(given.get(key), str) else given.get(key)
    if v not in _ON + _OFF:
        raise EditRequirementError(f"{key} is a switch (true / false), got {given[key]!r}")
    return v in _ON


def parse_measurement(user_requirements: Dict[str, Any], switch: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """(analysis kind, its parameters) when `switch` is on, else None; EditRequirementError on anything the Skill would refuse."""
    skill, kind, _, refinements = MEASUREMENTS[switch]
    prefix = switch.split(".", 1)[0] + "."
    given = {k: v for k, v in (user_requirements or {}).items() if k.startswith(prefix)}
    unknown = sorted(k for k in given if k not in REQUIREMENT_KEYS)
    if unknown:
        raise EditRequirementError(f"unknown requirement key(s): {', '.join(unknown)}; known: {', '.join(k for k in REQUIREMENT_KEYS if k.startswith(prefix))}")
    on = switch in given and _switch(given, switch)
    params: Dict[str, Any] = {}
    for key, (param, parse) in refinements.items():
        if given.get(key) is not None:
            if not on:
                raise EditRequirementError(f"{key} refines {switch}, which is not set to true")
            params[param] = parse(given[key], key)
    if on and kind == "screen_text":
        params.setdefault("text_every", TEXT_EVERY_DEFAULT)   # the agent always says how densely frames are sampled (detection-skill needs it)
    return (kind, params) if on else None


def check_measurement_keys(user_requirements: Dict[str, Any]) -> None:
    """Every `text.*` / `shots.*` / `speech_regions.*` key known and in range, refinements only with their switch (before any analysis)."""
    for switch in MEASUREMENTS:
        parse_measurement(user_requirements, switch)


def requested_kinds(user_requirements: Dict[str, Any]) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """{registry skill: (analysis kind, parameters)} for every measurement switched on."""
    out = {}
    for switch, (skill, _, _, _) in MEASUREMENTS.items():
        got = parse_measurement(user_requirements, switch)
        if got is not None:
            out[skill] = got
    return out
