"""Subtitle vocabulary (ADR-031): what the agent plans through subtitle-skill, and how a recognised Transcript becomes cues on
the *delivered* timeline.

    transcript Observation (transcription-skill: segments with start / end / text; speaker_id always null) → SpeechEvents
    → Decision `subtitle.generate` (explicit `subtitle` requirement) → ProductionPlan step → IR `captions.generate`
    (the cues, mapped onto the output timeline) → compiler → subtitle-skill `generate` (SRT / WebVTT sidecar)
    → optional Decision `subtitle.burn_in` → IR `captions.burn` → subtitle-skill `render` (the engine's caption tool)

Vocabulary and arithmetic only, no execution and no recognition:

- **Requirements** (`subtitle.*` keys, explicit USER / PROFILE only; DEFAULT never switches subtitles on): `subtitle` (true, or the
  format name), `subtitle.format` (srt | vtt), `subtitle.burn_in` (bool; the burnt-in picture is the deliverable, the sidecar is
  still produced), `subtitle.max_chars_per_line` / `subtitle.max_lines` (constraints the Skill checks; it never rewrites cues).
- **Timeline mapping** (deterministic, from IR content only): a transcript is measured on the *source*; the deliverable is the
  source after the trim (kept ranges), the concat (timeline offsets) and the speed factor. Every cue is mapped through exactly
  those operations — a cue inside a removed range is dropped, one overlapping a cut edge is clipped — so the sidecar is in sync
  with what is delivered. Nothing is re-timed by guesswork.
- **What is never done here**: speaker names (speaker_id is null and stays null), translation, rewriting, line breaking (the
  Skill's constraints report warnings; the cue text is the recognised text), burning into a source file (the burn-in input is
  always an intermediate inside the agent workspace — the Skill accepts workspace-relative inputs only).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..models import Requirement
from .editing import EditRequirementError, _bool, _number

FORMATS = ("srt", "vtt")
REQUIREMENT_KEYS = ("subtitle", "subtitle.format", "subtitle.burn_in", "subtitle.max_chars_per_line", "subtitle.max_lines")
CONSTRAINT_KEYS = ("max_chars_per_line", "max_lines")
GENERATE_TOOL = "subtitle/generate"
RENDER_TOOL = "subtitle/render"
GENERATE_SKILL = "subtitle_generation"
BURN_SKILL = "subtitle_burn_in"
MIN_CUE = 0.05          # seconds: a mapped cue shorter than this after clipping is dropped (it would not be readable)
_LANG_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def parse_subtitle_requirements(m: Dict[str, Requirement]) -> Dict[str, Any]:
    """Requirement map → {"enabled", "format" (None = policy decides), "burn_in", "constraints", "requirements"}. A refinement
    without the switch is refused (ambiguous request); values are range-checked; nothing is corrected."""
    main = m.get("subtitle")
    refinements = [m[k] for k in REQUIREMENT_KEYS[1:] if k in m and m[k].provenance != "DEFAULT"]
    out: Dict[str, Any] = {"enabled": False, "format": None, "burn_in": False, "constraints": {}, "requirements": []}
    if main is None or main.provenance == "DEFAULT":
        if refinements:
            raise EditRequirementError(f"{refinements[0].key} is set but subtitle is not: refusing an ambiguous request")
        return out
    v = main.value
    fmt: Optional[str] = None
    if isinstance(v, str) and v.strip().lower() in FORMATS:
        fmt = v.strip().lower()
        enabled = True
    else:
        enabled = _bool(v, "subtitle")
    if not enabled:
        if refinements:
            raise EditRequirementError("subtitle=false with subtitle.* refinements: refusing an ambiguous request")
        return out
    reqs = [main] + refinements
    vals = {r.key: r.value for r in reqs}
    if "subtitle.format" in vals:
        f = vals["subtitle.format"]
        if not isinstance(f, str) or f.strip().lower() not in FORMATS:
            raise EditRequirementError(f"subtitle.format must be one of {FORMATS}, got {f!r}")
        if fmt and fmt != f.strip().lower():
            raise EditRequirementError("subtitle names one format and subtitle.format another: refusing an ambiguous request")
        fmt = f.strip().lower()
    burn = _bool(vals["subtitle.burn_in"], "subtitle.burn_in") if "subtitle.burn_in" in vals else False
    cons: Dict[str, int] = {}
    if "subtitle.max_chars_per_line" in vals:
        cons["max_chars_per_line"] = int(_number(vals["subtitle.max_chars_per_line"], "subtitle.max_chars_per_line", 8, 200))
    if "subtitle.max_lines" in vals:
        cons["max_lines"] = int(_number(vals["subtitle.max_lines"], "subtitle.max_lines", 1, 4))
    return {"enabled": True, "format": fmt, "burn_in": burn, "constraints": cons, "requirements": reqs}


# ---- timeline mapping (source time → delivered time)
def kept_ranges_of(doc_video_ops: List[Dict[str, Any]], asset_id: str, duration: float) -> List[List[float]]:
    op = next((o for o in doc_video_ops if o.get("type") == "video.trim" and o.get("asset") == asset_id), None)
    if op is None:
        return [[0.0, round(float(duration), 3)]]
    return [[round(float(s), 3), round(float(e), 3)] for s, e in op["keep"]]


def map_point(t: float, keep: List[List[float]], is_end: bool) -> Optional[float]:
    """A source time onto the trimmed timeline. Inside a kept range: shifted by the removed material before it. Inside a removed
    range: a cue start snaps forward to the next kept range, a cue end snaps back to the previous one. None when nothing kept
    lies on that side (the cue lies entirely in removed material on that side)."""
    acc = 0.0
    for s, e in keep:
        if s <= t <= e:
            return round(acc + (t - s), 3)
        if t < s:
            return round(acc, 3) if not is_end else (round(acc, 3) if acc > 0 else None)
        acc += e - s
    return round(acc, 3) if is_end else None


def map_cue(start: float, end: float, keep: List[List[float]], offset: float = 0.0, speed: float = 1.0) -> Optional[Tuple[float, float]]:
    """(start, end) on the delivered timeline: trim → concat offset → speed factor; None when the cue is dropped."""
    ms, me = map_point(float(start), keep, False), map_point(float(end), keep, True)
    if ms is None or me is None:
        return None
    ms, me = (ms + offset) / speed, (me + offset) / speed
    if me - ms < MIN_CUE:
        return None
    return round(ms, 3), round(me, 3)


def cues_from_segments(segments: List[Dict[str, Any]], keep: List[List[float]], offset: float = 0.0, speed: float = 1.0, id_prefix: str = "c") -> List[Dict[str, Any]]:
    """Transcript segments → subtitle cues on the delivered timeline (ids `c0001`…, deterministic order, no overlap: a cue
    starting before the previous one ended is clipped to start there). Text is the recognised text, trimmed; empty cues and
    cues that fall into removed material are dropped. speaker is never set."""
    out: List[Dict[str, Any]] = []
    prev_end = -1.0
    for seg in sorted(segments, key=lambda s: (float(s.get("start", 0.0)), str(s.get("id")))):
        text = _CTRL_RE.sub("", str(seg.get("text") or "")).strip()
        if not text:
            continue
        mapped = map_cue(float(seg["start"]), float(seg["end"]), keep, offset, speed)
        if mapped is None:
            continue
        s, e = mapped
        if s < prev_end:
            s = prev_end
        if e - s < MIN_CUE:
            continue
        out.append({"id": f"{id_prefix}{len(out) + 1:04d}", "start": round(s, 3), "end": round(e, 3), "text": text})
        prev_end = round(e, 3)
    return out


def valid_language(lang: Any) -> str:
    s = str(lang or "").strip()
    return s if _LANG_RE.match(s) else "und"


# ---- IR records
def ir_captions_generate(subject: str, output: str, fmt: str, language: str, cues: List[Dict[str, Any]], decision_ids: List[str], sources: List[str],
                         timeline_map: Dict[str, Any], constraints: Optional[Dict[str, Any]] = None, scope: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"type": "captions.generate", "asset": subject, "output": output, "format": fmt, "language": language, "cues": list(cues),
                           "sources": list(sources), "timeline_map": dict(timeline_map)}
    if constraints:
        rec["constraints"] = {k: constraints[k] for k in CONSTRAINT_KEYS if k in constraints}
    if scope is not None:
        rec["temporal_scope"] = {"start": round(float(scope["start"]), 3), "end": round(float(scope["end"]), 3)}
    rec["decision_ids"] = list(decision_ids)
    return rec


def ir_captions_burn(subject: str, input_id: str, sidecar: str, output: str, decision_ids: List[str], scope: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"type": "captions.burn", "asset": subject, "input": input_id, "sidecar": sidecar, "output": output}
    if scope is not None:
        rec["temporal_scope"] = {"start": round(float(scope["start"]), 3), "end": round(float(scope["end"]), 3)}
    rec["decision_ids"] = list(decision_ids)
    return rec
