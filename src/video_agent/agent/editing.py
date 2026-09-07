"""Editing operations vocabulary (ADR-029): the video-editing-skill operations the agent can plan — concat, speed, resize,
fit, fill, overlay — as explicit Requirements → Decisions → ProductionPlan steps → Project IR operations.

Everything here is vocabulary and arithmetic, no execution:

- **Requirements** (`edit.<op>` keys, explicit `--set` only; no keyword pass, no inference): parsed and range-checked here.
  An invalid value is refused at planning time (ValueError) — nothing is corrected or guessed.
- **IR operations**: one record per operation with an explicit `type`, its typed parameters (allowlisted per type), the
  input / output references, the temporal scope on the timeline it produces, and the decision ids. `video.concat` is the
  only multi-source operation: it consumes the (trimmed) assets in the order given and produces the logical output
  `programme`; the later single-source operations then apply to that programme instead of to each asset.
- **Expected durations** (QA): derived from the IR only (kept ranges, concat segments, speed factor), never measured here.

The Skill executes; the Decision decides. Fit and fill both requested is a conflict (BLOCK), an asset without a video
stream cannot take a video operation (BLOCK), concat needs two or more video inputs (BLOCK). Parameters that name
commands, filters, executables or paths never exist in this vocabulary (the compiler copies allowlisted keys only).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ..models import Requirement

PROGRAMME = "programme"      # logical id of the concat output (the multi-source timeline)
# operation order after the trims: the same for the plan, the IR and the compiler (deterministic, not configurable)
EDIT_ORDER = ("video.concat", "video.speed", "video.resize", "video.fit", "video.fill", "video.overlay")
# op type → production skill / video-editing tool / IR parameter allowlist (the only keys copied into an IR op besides the references)
OPERATIONS: Dict[str, Dict[str, Any]] = {
    "video.concat":  {"skill": "video_concat",  "tool": "video-editing/concat",  "params": ("transition", "width", "height", "fps", "mode", "pad_color"), "risk": "MEDIUM"},
    "video.speed":   {"skill": "video_speed",   "tool": "video-editing/speed",   "params": ("factor",), "risk": "MEDIUM"},
    "video.resize":  {"skill": "video_resize",  "tool": "video-editing/resize",  "params": ("width", "fps"), "risk": "LOW"},
    "video.fit":     {"skill": "video_fit",     "tool": "video-editing/fit",     "params": ("aspect", "width", "pad_color", "fps"), "risk": "LOW"},
    "video.fill":    {"skill": "video_fill",    "tool": "video-editing/fill",    "params": ("aspect", "width", "fps"), "risk": "MEDIUM"},
    "video.overlay": {"skill": "video_overlay", "tool": "video-editing/overlay", "params": ("position", "margin", "scale", "opacity", "start", "end", "fade"), "risk": "MEDIUM"},
}
SKILL_OF = {k: v["skill"] for k, v in OPERATIONS.items()}
OP_OF_SKILL = {v["skill"]: k for k, v in OPERATIONS.items()}
# requirement keys: edit.<op> switches the operation on (its value is the main parameter), edit.<op>.<param> refines it
REQUIREMENT_KEYS: Dict[str, Tuple[str, ...]] = {
    "video.concat": ("edit.concat", "edit.concat.transition", "edit.concat.transition_duration", "edit.concat.width", "edit.concat.height", "edit.concat.fps", "edit.concat.mode", "edit.concat.pad_color"),
    "video.speed": ("edit.speed",),
    "video.resize": ("edit.resize", "edit.resize.fps"),
    "video.fit": ("edit.fit", "edit.fit.width", "edit.fit.pad_color", "edit.fit.fps"),
    "video.fill": ("edit.fill", "edit.fill.width", "edit.fill.fps"),
    "video.overlay": ("edit.overlay", "edit.overlay.position", "edit.overlay.margin", "edit.overlay.scale", "edit.overlay.opacity", "edit.overlay.start", "edit.overlay.end", "edit.overlay.fade"),
}
SPEED_RANGE = (0.25, 4.0)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
NAMED_POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right", "center", "top", "bottom", "left", "right")
_ASPECT_RE = re.compile(r"^[1-9][0-9]{0,3}:[1-9][0-9]{0,3}$")
_COLOR_RE = re.compile(r"^([a-z]{1,16}|0x[0-9A-Fa-f]{6})$")
# video-editing-skill's CONCAT transition.type enum (its own contract.py TRANSITIONS, pinned here since the agent never
# imports a Skill's package): "none" is deliberately excluded -- omitting `edit.concat.transition` entirely is how this
# vocabulary asks for a straight cut, not a transition value named "none" (that spelling is ffmpeg-skill CLI's own
# --transition flag, a different, lower-level vocabulary this agent never exposes directly).
_TRANSITIONS = ("fade", "dissolve", "wipeleft", "wiperight", "wipeup", "wipedown", "slideleft", "slideright",
                "circleopen", "circleclose", "fadeblack", "fadewhite", "smoothleft", "smoothright", "radial")
_TRANSITION_RE = re.compile("^(" + "|".join(_TRANSITIONS) + ")$")


class EditRequirementError(ValueError):
    """An `edit.*` requirement whose value is outside the vocabulary (refused at planning time; nothing is corrected)."""


def _bool(v: Any, key: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "yes", "on", "1"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "no", "off", "0"):
        return False
    raise EditRequirementError(f"{key} must be true or false, got {v!r}")


def _number(v: Any, key: str, lo: Optional[float] = None, hi: Optional[float] = None, allow_ratio: bool = False) -> Any:
    if isinstance(v, str) and allow_ratio and re.match(r"^[0-9]{1,6}/[0-9]{1,6}$", v):
        return v
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            raise EditRequirementError(f"{key} must be a number, got {v!r}")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        raise EditRequirementError(f"{key} must be a finite number, got {v!r}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise EditRequirementError(f"{key} must be within {lo}..{hi}, got {v!r}")
    return float(v) if isinstance(v, float) else int(v)


def _even_int(v: Any, key: str, lo: int = 16, hi: int = 8192) -> int:
    n = _number(v, key, lo, hi)
    if float(n) != int(n) or int(n) % 2:
        raise EditRequirementError(f"{key} must be an even integer between {lo} and {hi}, got {v!r}")
    return int(n)


def _token(v: Any, key: str, rx: "re.Pattern[str]", what: str) -> str:
    if not isinstance(v, str) or not rx.match(v):
        raise EditRequirementError(f"{key} must be {what}, got {v!r}")
    return v


def _fps(v: Any, key: str) -> Any:
    return _number(v, key, 1, 240, allow_ratio=True)


def parse_edit_requirements(m: Dict[str, Requirement]) -> Dict[str, Dict[str, Any]]:
    """Requirement map → {op type: {"params": {...}, "requirements": [Requirement, …]}} for the operations that were asked for.
    Only explicit USER / PROFILE requirements switch an operation on (DEFAULT never does). Values are range-checked; a
    refinement key without its operation is refused (ambiguous request). Raises EditRequirementError."""
    out: Dict[str, Dict[str, Any]] = {}
    for op, keys in REQUIREMENT_KEYS.items():
        main = m.get(keys[0])
        refinements = [m[k] for k in keys[1:] if k in m]
        if main is None or main.provenance == "DEFAULT":
            if refinements:
                raise EditRequirementError(f"{refinements[0].key} is set but {keys[0]} is not: refusing an ambiguous request")
            continue
        reqs = [main] + refinements
        vals = {r.key: r.value for r in reqs}
        p: Dict[str, Any] = {}
        if op == "video.concat":
            if not _bool(vals["edit.concat"], "edit.concat"):
                continue
            if "edit.concat.transition" in vals:
                p["transition"] = {"type": _token(vals["edit.concat.transition"], "edit.concat.transition", _TRANSITION_RE, "one of " + ", ".join(_TRANSITIONS)),
                                   "duration": _number(vals.get("edit.concat.transition_duration", 0.5), "edit.concat.transition_duration", 0.05, 5.0)}
            elif "edit.concat.transition_duration" in vals:
                raise EditRequirementError("edit.concat.transition_duration needs edit.concat.transition")
            for k in ("width", "height"):
                if f"edit.concat.{k}" in vals:
                    p[k] = _even_int(vals[f"edit.concat.{k}"], f"edit.concat.{k}")
            if "edit.concat.fps" in vals:
                p["fps"] = _fps(vals["edit.concat.fps"], "edit.concat.fps")
            if "edit.concat.mode" in vals:
                p["mode"] = _token(vals["edit.concat.mode"], "edit.concat.mode", re.compile(r"^(pad|crop)$"), "pad or crop")
            if "edit.concat.pad_color" in vals:
                p["pad_color"] = _token(vals["edit.concat.pad_color"], "edit.concat.pad_color", _COLOR_RE, "a colour name or 0xRRGGBB")
        elif op == "video.speed":
            f = _number(vals["edit.speed"], "edit.speed", SPEED_RANGE[0], SPEED_RANGE[1])
            if float(f) == 1.0:
                raise EditRequirementError("edit.speed=1 changes nothing; leave it unset")
            p["factor"] = float(f)
        elif op == "video.resize":
            p["width"] = _even_int(vals["edit.resize"], "edit.resize")
            if "edit.resize.fps" in vals:
                p["fps"] = _fps(vals["edit.resize.fps"], "edit.resize.fps")
        elif op in ("video.fit", "video.fill"):
            base = "edit.fit" if op == "video.fit" else "edit.fill"
            p["aspect"] = _token(vals[base], base, _ASPECT_RE, "an aspect ratio W:H")
            if f"{base}.width" in vals:
                p["width"] = _even_int(vals[f"{base}.width"], f"{base}.width")
            if op == "video.fit" and f"{base}.pad_color" in vals:
                p["pad_color"] = _token(vals[f"{base}.pad_color"], f"{base}.pad_color", _COLOR_RE, "a colour name or 0xRRGGBB")
            if f"{base}.fps" in vals:
                p["fps"] = _fps(vals[f"{base}.fps"], f"{base}.fps")
        elif op == "video.overlay":
            img = vals["edit.overlay"]
            if not isinstance(img, str) or not img.strip() or "\n" in img or "\x00" in img:
                raise EditRequirementError("edit.overlay must be the path of a PNG / JPEG image")
            if any(part == ".." for part in img.replace("\\", "/").split("/")):
                raise EditRequirementError("edit.overlay path contains '..' (traversal)")
            if os.path.splitext(img)[1].lower() not in IMAGE_EXTENSIONS:
                raise EditRequirementError(f"edit.overlay must be one of {IMAGE_EXTENSIONS}")
            if not os.path.isfile(img):
                raise EditRequirementError(f"edit.overlay image not found: {img}")
            p["image"] = os.path.abspath(img)
            pos = vals.get("edit.overlay.position")
            if isinstance(pos, dict):
                if set(pos) != {"x", "y"}:
                    raise EditRequirementError("edit.overlay.position must be a named position or {x, y}")
                p["position"] = {"x": int(_number(pos["x"], "edit.overlay.position.x", 0, 16384)), "y": int(_number(pos["y"], "edit.overlay.position.y", 0, 16384))}
            elif pos is not None:
                p["position"] = _token(pos, "edit.overlay.position", re.compile("^(" + "|".join(NAMED_POSITIONS) + ")$"), "one of " + ", ".join(NAMED_POSITIONS))
            for k, lo, hi in (("margin", 0, 4096), ("scale", 1, 4096)):
                if f"edit.overlay.{k}" in vals:
                    p[k] = int(_number(vals[f"edit.overlay.{k}"], f"edit.overlay.{k}", lo, hi))
                    if float(p[k]) != float(_number(vals[f"edit.overlay.{k}"], f"edit.overlay.{k}")):
                        raise EditRequirementError(f"edit.overlay.{k} must be an integer")
            if "edit.overlay.opacity" in vals:
                p["opacity"] = float(_number(vals["edit.overlay.opacity"], "edit.overlay.opacity", 0.0, 1.0))
            for k in ("start", "end", "fade"):
                if f"edit.overlay.{k}" in vals:
                    p[k] = float(_number(vals[f"edit.overlay.{k}"], f"edit.overlay.{k}", 0.0, 86400.0))
            if "start" in p and "end" in p and not p["start"] < p["end"]:
                raise EditRequirementError("edit.overlay.start must be before edit.overlay.end")
        out[op] = {"params": p, "requirements": reqs}
    return out


# ---- IR arithmetic (deterministic, from IR content only)
def kept_duration(doc_or_ops: Any, asset_id: str, source_duration: float) -> float:
    """Duration of an asset after its video.trim (sum of kept ranges), or the source duration when it is not trimmed."""
    ops = doc_or_ops["video"]["operations"] if isinstance(doc_or_ops, dict) and "video" in doc_or_ops else doc_or_ops
    for op in ops:
        if op.get("asset") == asset_id and op.get("type") == "video.trim":
            return round(sum(float(e) - float(s) for s, e in op["keep"]), 3)
    return round(float(source_duration or 0.0), 3)


def concat_segments(inputs: List[str], video_ops: List[Dict[str, Any]], durations: Dict[str, float], transition: Optional[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    """Multi-source timeline of a concat: for every input, its kept source ranges and where they land on the programme
    timeline (a transition overlaps consecutive clips by its duration). Returns (segments, programme duration)."""
    segs: List[Dict[str, Any]] = []
    t = 0.0
    overlap = float((transition or {}).get("duration") or 0.0)
    for n, aid in enumerate(inputs):
        keep = next((op["keep"] for op in video_ops if op.get("asset") == aid and op.get("type") == "video.trim"), [[0.0, float(durations.get(aid) or 0.0)]])
        if n > 0:
            t = max(0.0, t - overlap)
        for s, e in keep:
            length = round(float(e) - float(s), 3)
            segs.append({"input": aid, "track": "V1", "source_range": [round(float(s), 3), round(float(e), 3)], "timeline_range": [round(t, 3), round(t + length, 3)]})
            t = round(t + length, 3)
    return segs, round(t, 3)


def delivery_subjects(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What the plan delivers: the concat programme (one subject made of every input) or each asset. Each row carries the
    subject id (the key of `<subject>_delivery_<target>`), its source asset ids, the expected output duration derived from
    the IR (trim → concat → speed), the loudness target of its audio.loudness op, and the technical facts of its first source."""
    from .audio import audio_subjects
    assets = doc.get("assets") or {}
    vops = (doc.get("video") or {}).get("operations") or []
    aops = (doc.get("audio") or {}).get("operations") or []
    concat = next((op for op in vops if op.get("type") == "video.concat"), None)
    rows: List[Dict[str, Any]] = []
    audio_rows = audio_subjects(doc)   # subjects delivered as audio only (ADR-030): their picture is not part of the deliverable
    consumed = {s for r in audio_rows.values() for s in r["sources"]}
    for sid, r in sorted(audio_rows.items()):
        tech = dict((assets.get(r["sources"][0]) or {}).get("technical") or {})
        tech["video"] = None
        if r.get("channels") is not None:
            tech["audio"] = dict(tech.get("audio") or {}, channels=r["channels"])   # the planned layout (mono / stereo / down-mix), not the source's
        rows.append({"id": sid, "sources": list(r["sources"]), "duration": r["duration"], "technical": tech, "audio_only": True})
    if concat is not None:
        rows.append({"id": concat.get("output", PROGRAMME), "sources": list(concat.get("inputs") or []), "duration": float(concat.get("timeline_duration") or 0.0),
                     "technical": dict((assets.get((concat.get("inputs") or [""])[0]) or {}).get("technical") or {})})
        # assets that are not part of the programme (none today: concat takes every video asset) would be delivered on their own
    else:
        for aid, a in assets.items():
            if aid in consumed:
                continue
            rows.append({"id": aid, "sources": [aid], "duration": kept_duration(vops, aid, (a.get("technical") or {}).get("duration") or 0.0), "technical": dict(a.get("technical") or {})})
    for row in rows:
        row.setdefault("audio_only", False)
        dur = row["duration"]
        for op in vops:
            if op.get("asset") == row["id"] and op.get("type") == "video.speed" and op.get("factor"):
                dur = dur / float(op["factor"])
        row["duration"] = round(dur, 3)
        row["target_lufs"] = next((op.get("target_lufs") for op in aops if op.get("asset") == row["id"] and op.get("type") == "audio.loudness"), None)
        row["edits"] = [op["type"] for op in vops if op.get("asset") == row["id"] and op.get("type") != "video.trim"]
    return rows


def ir_operation(op_type: str, subject: str, params: Dict[str, Any], decision_ids: List[str], scope: Optional[Dict[str, float]] = None, **refs: Any) -> Dict[str, Any]:
    """One IR video operation record: explicit type, subject (asset or programme), allowlisted parameters, references
    (inputs / output / image), temporal scope on the produced timeline and the decision ids. Keys outside the allowlist are
    dropped here on purpose: the IR never carries free-form parameters."""
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
