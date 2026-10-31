"""Editing operations vocabulary (ADR-029): the video-editing-skill operations the agent can plan — concat, speed, resize,
fit, fill, overlay, rotate (ADR-062), an explicit trim (`edit.trim`, ADR-062) and a still image joined before / after the
picture (`edit.image_insert`, ADR-070) — as explicit Requirements → Decisions →
ProductionPlan steps → Project IR operations.

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

PROGRAMME = "programme"      # logical id of the concat (or switch / grid) output (the multi-source timeline)
# operation order after the trims: the same for the plan, the IR and the compiler (deterministic, not configurable).
# video.concat, video.switch and video.grid are alternative ways to build PROGRAMME from 2+ video inputs (mutually
# exclusive, ADR-045 / ADR-046 decide a BLOCK if more than one is requested); everything after them applies to whichever
# one actually ran. The single-source ops (ADR-046) compose freely with each other and with whichever programme was built:
# deinterlace / stabilize (quality / motion fixes) run first, then the fixed-pixel-rectangle ops (crop, then redact --
# redact's rectangle is expressed in whatever frame crop already produced, matching the order they are declared here),
# then the pre-existing speed / resize / fit / fill / overlay chain. video.rotate (ADR-062) runs after crop / redact (their
# rectangles are in source pixels, before any turn) and before speed / resize / fit / fill / overlay (a width, an aspect and an
# overlay position are meant on the upright picture).
# video.image_insert (ADR-070) is last: the still is joined to the finished picture -- after colour, graphics and burnt-in subtitles too,
# so it is not graded or captioned and every finishing time stays on the picture's own timeline (planner / compiler run it after
# finishing, before loudness); it is not part of SINGLE_SOURCE_ORDER's chain.
IMAGE_INSERT_OP = "video.image_insert"
EDIT_ORDER = ("video.concat", "video.switch", "video.grid", "video.deinterlace", "video.stabilize", "video.crop", "video.redact",
              "video.rotate", "video.speed", "video.resize", "video.fit", "video.fill", "video.overlay", IMAGE_INSERT_OP)
# the multi-source entry ops (build PROGRAMME from 2+ inputs; mutually exclusive, handled specially by the planner /
# compiler) vs. the single-source ops that apply to whatever PROGRAMME (or the untouched asset) already is
MULTI_SOURCE_OPS = ("video.concat", "video.switch", "video.grid")
SINGLE_SOURCE_ORDER = tuple(op for op in EDIT_ORDER[len(MULTI_SOURCE_OPS):] if op != IMAGE_INSERT_OP)
# op type → production skill / video-editing tool / IR parameter allowlist (the only keys copied into an IR op besides the references)
OPERATIONS: Dict[str, Dict[str, Any]] = {
    "video.concat":  {"skill": "video_concat",  "tool": "video-editing/concat",  "params": ("transition", "width", "height", "fps", "mode", "pad_color"), "risk": "MEDIUM"},
    # no "tool" here (unlike the other entries): camera_switch's sole candidate is ffmpeg-skill/multicam, and this codebase's
    # own boundary test (test_no_tool_id_literals_outside_tool_layer) refuses an "ffmpeg-skill/..." literal anywhere outside
    # tools/ or skills/ -- the registry (skills/registry.py) is the only place that names it; lower_video_edit() skips the
    # tool-pairing sanity check when "tool" is absent, since the generic per-step registry check already guarantees it.
    "video.switch":  {"skill": "camera_switch", "params": ("switch", "audio", "fix_drift"), "risk": "HIGH"},
    # ADR-046: same pattern as video.switch -- ffmpeg-skill/grid is the sole candidate, so "tool" is deliberately omitted here too.
    "video.grid":    {"skill": "video_grid", "params": ("cols", "rows", "cell_width", "cell_height", "fps", "label", "font", "font_size",
                                                          "font_color", "pad", "audio_from", "gap", "background"), "risk": "MEDIUM"},
    # ADR-046: fixed-pixel-rectangle / quality ops, all ffmpeg-skill-only (same "no tool key" pattern).
    "video.redact":      {"skill": "video_redact",      "params": ("x", "y", "width", "height", "mode", "blur_strength", "block_size", "audio_stream", "fps"), "risk": "HIGH"},
    "video.deinterlace": {"skill": "video_deinterlace",  "params": ("mode", "parity", "only_interlaced", "audio_stream"), "risk": "LOW"},
    "video.crop":        {"skill": "video_crop",         "params": ("x", "y", "width", "height", "fps"), "risk": "MEDIUM"},
    "video.stabilize":   {"skill": "video_stabilize",    "params": ("shakiness", "smoothing", "zoom", "crop_mode", "tripod"), "risk": "MEDIUM"},
    # ADR-062: video-editing-skill ROTATE (0.3.0): a clockwise turn of 90 / 180 / 270 and / or a mirror; nothing scaled or cut
    "video.rotate":  {"skill": "video_rotate",  "tool": "video-editing/rotate",  "params": ("degrees", "flip"), "risk": "LOW"},
    "video.speed":   {"skill": "video_speed",   "tool": "video-editing/speed",   "params": ("factor",), "risk": "MEDIUM"},
    "video.resize":  {"skill": "video_resize",  "tool": "video-editing/resize",  "params": ("width", "fps"), "risk": "LOW"},
    "video.fit":     {"skill": "video_fit",     "tool": "video-editing/fit",     "params": ("aspect", "width", "pad_color", "fps"), "risk": "LOW"},
    "video.fill":    {"skill": "video_fill",    "tool": "video-editing/fill",    "params": ("aspect", "width", "fps"), "risk": "MEDIUM"},
    "video.overlay": {"skill": "video_overlay", "tool": "video-editing/overlay", "params": ("position", "margin", "scale", "opacity", "start", "end", "fade"), "risk": "MEDIUM"},
    # ADR-070: video-editing-skill IMAGE_INSERT (0.4.0, #21): a still → a silent clip (duration / width / height / fps, the contract's names),
    # joined by video-editing/concat before or after the picture (`position`). width / height / fps are the picture's own (planner), never asked
    IMAGE_INSERT_OP: {"skill": "video_image_insert", "tool": "video-editing/image_insert", "params": ("duration", "position", "width", "height", "fps"), "risk": "MEDIUM"},
}
SKILL_OF = {k: v["skill"] for k, v in OPERATIONS.items()}
OP_OF_SKILL = {v["skill"]: k for k, v in OPERATIONS.items()}
# ADR-062: the explicit trim -- decision subject / IR type, requirement key, the production skill that runs it alone
TRIM_OP = "video.trim"
TRIM_KEY = "edit.trim"
TRIM_SKILL = "video_trim"
# requirement keys: edit.<op> switches the operation on (its value is the main parameter), edit.<op>.<param> refines it
REQUIREMENT_KEYS: Dict[str, Tuple[str, ...]] = {
    "video.concat": ("edit.concat", "edit.concat.transition", "edit.concat.transition_duration", "edit.concat.width", "edit.concat.height", "edit.concat.fps", "edit.concat.mode", "edit.concat.pad_color"),
    "video.switch": ("edit.switch", "edit.switch.audio", "edit.switch.fix_drift"),
    "video.grid": ("edit.grid", "edit.grid.cols", "edit.grid.rows", "edit.grid.cell_width", "edit.grid.cell_height", "edit.grid.fps", "edit.grid.label",
                   "edit.grid.font", "edit.grid.font_size", "edit.grid.font_color", "edit.grid.pad", "edit.grid.audio_from", "edit.grid.gap", "edit.grid.background"),
    # edit.redact / edit.crop: the main key IS the rectangle ({x, y, width, height}), mirroring edit.overlay's main key
    # being the image path -- there is no sensible "on/off" switch for a redaction or crop without the rectangle it acts on.
    "video.redact": ("edit.redact", "edit.redact.mode", "edit.redact.blur_strength", "edit.redact.block_size", "edit.redact.audio_stream", "edit.redact.fps"),
    "video.deinterlace": ("edit.deinterlace", "edit.deinterlace.mode", "edit.deinterlace.parity", "edit.deinterlace.only_interlaced", "edit.deinterlace.audio_stream"),
    "video.crop": ("edit.crop", "edit.crop.fps"),
    "video.stabilize": ("edit.stabilize", "edit.stabilize.shakiness", "edit.stabilize.smoothing", "edit.stabilize.zoom", "edit.stabilize.crop_mode", "edit.stabilize.tripod"),
    "video.rotate": ("edit.rotate", "edit.rotate.flip"),
    "video.speed": ("edit.speed",),
    "video.resize": ("edit.resize", "edit.resize.fps"),
    "video.fit": ("edit.fit", "edit.fit.width", "edit.fit.pad_color", "edit.fit.fps"),
    "video.fill": ("edit.fill", "edit.fill.width", "edit.fill.fps"),
    "video.overlay": ("edit.overlay", "edit.overlay.position", "edit.overlay.margin", "edit.overlay.scale", "edit.overlay.opacity", "edit.overlay.start", "edit.overlay.end", "edit.overlay.fade"),
    # ADR-062: an explicit in / out range ("START-END", seconds or mm:ss). Not an OPERATIONS entry: it narrows the asset's one
    # video.trim (the same IR record the silence decisions produce), executed by video-editing/trim when one range remains
    TRIM_OP: (TRIM_KEY,),
    # ADR-070: the main key is the still's path (like edit.overlay); duration is required; position before (default) | after
    IMAGE_INSERT_OP: ("edit.image_insert", "edit.image_insert.duration", "edit.image_insert.position"),
}
IMAGE_INSERT_DURATION = (0.01, 600.0)        # video-editing-skill IMAGE_INSERT.duration ("time 0.01..600 s")
IMAGE_INSERT_POSITIONS = ("before", "after")
IMAGE_INSERT_SKILL = "video_image_insert"
IMAGE_JOIN_SKILL = "video_concat"            # the join of the still's clip and the picture is a plain video-editing CONCAT
ROTATIONS = (90, 180, 270)   # video-editing-skill ROTATE.degrees, clockwise
FLIPS = ("h", "v")           # ROTATE.flip: mirror horizontally / vertically
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
# a switch list segment on the reference timeline: "START-END:CAM" (seconds or mm:ss, CAM = input index, 0 = reference);
# the same grammar ffmpeg-skill/multicam.py's own --switch flag takes (ADR-045). Bounds against the actual input count
# are checked in decision.py, once the number of video inputs is known; this only fixes the grammar.
_TIME_RE = r"[0-9]+(\.[0-9]+)?|[0-9]{1,2}:[0-9]{2}"
_SWITCH_SEG_RE = re.compile(r"^(?P<s>" + _TIME_RE + r")-(?P<e>" + _TIME_RE + r"):(?P<cam>[0-9]+)$")


def _switch_seconds(v: str) -> float:
    if ":" not in v:
        return float(v)
    mm, ss = v.split(":")
    return float(int(mm) * 60 + int(ss))


def switch_cams(switch: str) -> List[int]:
    """The camera indices a switch list cites (for a bounds check against the actual input count)."""
    return [int(seg.split(":")[-1]) for seg in switch.split(",")]


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


def parse_trim(v: Any) -> Dict[str, float]:
    """`edit.trim`: "START-END" (seconds or mm:ss, the switch-list time grammar) or {start, end} in seconds → {start, end}.
    The range is checked against the input's duration by the decision, once the probe is known; here only its shape and order."""
    if isinstance(v, dict):
        if set(v) != {"start", "end"}:
            raise EditRequirementError(f"{TRIM_KEY} must be \"START-END\" or {{start, end}} in seconds")
        s_, e_ = float(_number(v["start"], TRIM_KEY + ".start", 0, 86400)), float(_number(v["end"], TRIM_KEY + ".end", 0, 86400))
    else:
        mobj = re.match(r"^\s*(?P<s>" + _TIME_RE + r")\s*-\s*(?P<e>" + _TIME_RE + r")\s*$", v) if isinstance(v, str) else None
        if not mobj:
            raise EditRequirementError(f"{TRIM_KEY} must be \"START-END\" (seconds or mm:ss, e.g. 2.5-14 or 0:05-1:30), got {v!r}")
        s_, e_ = _switch_seconds(mobj.group("s")), _switch_seconds(mobj.group("e"))
    if not s_ < e_:
        raise EditRequirementError(f"{TRIM_KEY}: start {s_:g}s must be before end {e_:g}s")
    return {"start": round(s_, 3), "end": round(e_, 3)}


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
        elif op == "video.switch":
            sw = vals["edit.switch"]
            if not isinstance(sw, str) or not sw.strip():
                raise EditRequirementError("edit.switch must be a non-empty switch list \"START-END:CAM,...\" (seconds or mm:ss, CAM = input index)")
            for seg in sw.split(","):
                mobj = _SWITCH_SEG_RE.match(seg.strip())
                if not mobj:
                    raise EditRequirementError(f"edit.switch segment {seg!r} must be START-END:CAM (seconds or mm:ss, CAM a non-negative integer)")
                if not _switch_seconds(mobj.group("s")) < _switch_seconds(mobj.group("e")):
                    raise EditRequirementError(f"edit.switch segment {seg!r}: start must be before end")
            p["switch"] = sw
            if "edit.switch.audio" in vals:
                p["audio"] = int(_number(vals["edit.switch.audio"], "edit.switch.audio", 0, 63))
            if "edit.switch.fix_drift" in vals:
                p["fix_drift"] = _bool(vals["edit.switch.fix_drift"], "edit.switch.fix_drift")
        elif op == "video.grid":
            if not _bool(vals["edit.grid"], "edit.grid"):
                continue
            if "edit.grid.cols" not in vals or "edit.grid.rows" not in vals:
                raise EditRequirementError("edit.grid needs edit.grid.cols and edit.grid.rows (grid.py's own required flags)")
            p["cols"] = int(_number(vals["edit.grid.cols"], "edit.grid.cols", 1, 64))
            p["rows"] = int(_number(vals["edit.grid.rows"], "edit.grid.rows", 1, 64))
            for k in ("cell_width", "cell_height"):
                if f"edit.grid.{k}" in vals:
                    p[k] = _even_int(vals[f"edit.grid.{k}"], f"edit.grid.{k}")
            if "edit.grid.fps" in vals:
                p["fps"] = _fps(vals["edit.grid.fps"], "edit.grid.fps")
            if "edit.grid.label" in vals:
                p["label"] = _token(vals["edit.grid.label"], "edit.grid.label", re.compile(r"^(auto|none)$"), "auto or none")
            if "edit.grid.font" in vals:
                p["font"] = _token(vals["edit.grid.font"], "edit.grid.font", re.compile(r"^[A-Za-z0-9 _-]{1,64}$"), "a font family name")
            if "edit.grid.font_size" in vals:
                p["font_size"] = int(_number(vals["edit.grid.font_size"], "edit.grid.font_size", 1, 256))
            if "edit.grid.font_color" in vals:
                p["font_color"] = _token(vals["edit.grid.font_color"], "edit.grid.font_color", _COLOR_RE, "a colour name or 0xRRGGBB")
            if "edit.grid.pad" in vals:
                p["pad"] = _bool(vals["edit.grid.pad"], "edit.grid.pad")
            if "edit.grid.audio_from" in vals:
                p["audio_from"] = int(_number(vals["edit.grid.audio_from"], "edit.grid.audio_from", 0, 63))
            if "edit.grid.gap" in vals:
                p["gap"] = _even_int(vals["edit.grid.gap"], "edit.grid.gap", lo=0)
            if "edit.grid.background" in vals:
                p["background"] = _token(vals["edit.grid.background"], "edit.grid.background", _COLOR_RE, "a colour name or 0xRRGGBB")
        elif op in ("video.redact", "video.crop"):
            # the rectangle IS the main requirement value (mirrors edit.overlay's image path): {x, y, width, height} in
            # SOURCE pixels, exactly redact.py / crop.py's own contract. width/height must be even (4:2:0 chroma
            # subsampling, both scripts' own validation) -- checked here so a bad rectangle fails at planning time,
            # never silently rounded or guessed. Neither op locates anything itself (no face/plate detection, no
            # auto letterbox detection): the caller always supplies the exact rectangle (ffmpeg-skill's own non-goal).
            base = "edit.redact" if op == "video.redact" else "edit.crop"
            rect = vals[base]
            if not isinstance(rect, dict) or set(rect) != {"x", "y", "width", "height"}:
                raise EditRequirementError(f"{base} must be a rectangle {{x, y, width, height}} in source pixels")
            p["x"] = float(_number(rect["x"], f"{base}.x", 0, 16384))
            p["y"] = float(_number(rect["y"], f"{base}.y", 0, 16384))
            p["width"] = _even_int(rect["width"], f"{base}.width")
            p["height"] = _even_int(rect["height"], f"{base}.height")
            if f"{base}.fps" in vals:
                p["fps"] = _fps(vals[f"{base}.fps"], f"{base}.fps")
            if op == "video.redact":
                if "edit.redact.mode" in vals:
                    p["mode"] = _token(vals["edit.redact.mode"], "edit.redact.mode", re.compile(r"^(blur|pixelate)$"), "blur or pixelate")
                if "edit.redact.blur_strength" in vals:
                    p["blur_strength"] = int(_number(vals["edit.redact.blur_strength"], "edit.redact.blur_strength", 1, 256))
                if "edit.redact.block_size" in vals:
                    p["block_size"] = int(_number(vals["edit.redact.block_size"], "edit.redact.block_size", 1, 256))
                if "edit.redact.audio_stream" in vals:
                    p["audio_stream"] = int(_number(vals["edit.redact.audio_stream"], "edit.redact.audio_stream", 0, 63))
        elif op == "video.deinterlace":
            if not _bool(vals["edit.deinterlace"], "edit.deinterlace"):
                continue
            if "edit.deinterlace.mode" in vals:
                p["mode"] = _token(vals["edit.deinterlace.mode"], "edit.deinterlace.mode", re.compile(r"^(frame|field)$"), "frame or field")
            if "edit.deinterlace.parity" in vals:
                p["parity"] = _token(vals["edit.deinterlace.parity"], "edit.deinterlace.parity", re.compile(r"^(auto|tff|bff)$"), "auto, tff or bff")
            if "edit.deinterlace.only_interlaced" in vals:
                p["only_interlaced"] = _bool(vals["edit.deinterlace.only_interlaced"], "edit.deinterlace.only_interlaced")
            if "edit.deinterlace.audio_stream" in vals:
                p["audio_stream"] = int(_number(vals["edit.deinterlace.audio_stream"], "edit.deinterlace.audio_stream", 0, 63))
        elif op == "video.stabilize":
            if not _bool(vals["edit.stabilize"], "edit.stabilize"):
                continue
            if "edit.stabilize.shakiness" in vals:
                p["shakiness"] = int(_number(vals["edit.stabilize.shakiness"], "edit.stabilize.shakiness", 1, 10))
            if "edit.stabilize.smoothing" in vals:
                p["smoothing"] = int(_number(vals["edit.stabilize.smoothing"], "edit.stabilize.smoothing", 0, 1000))
            if "edit.stabilize.zoom" in vals:
                p["zoom"] = float(_number(vals["edit.stabilize.zoom"], "edit.stabilize.zoom", 0, 100))
            if "edit.stabilize.crop_mode" in vals:
                p["crop_mode"] = _token(vals["edit.stabilize.crop_mode"], "edit.stabilize.crop_mode", re.compile(r"^(keep|black)$"), "keep or black")
            if "edit.stabilize.tripod" in vals:
                p["tripod"] = _bool(vals["edit.stabilize.tripod"], "edit.stabilize.tripod")
        elif op == "video.rotate":
            deg = vals["edit.rotate"]
            deg = _number(deg, "edit.rotate", 0, 270)
            if float(deg) not in (0.0,) + tuple(float(r) for r in ROTATIONS):
                raise EditRequirementError(f"edit.rotate must be 90, 180 or 270 (degrees clockwise; 0 with edit.rotate.flip for a mirror only), got {vals['edit.rotate']!r}")
            if int(deg):
                p["degrees"] = int(deg)
            if "edit.rotate.flip" in vals:
                p["flip"] = _token(vals["edit.rotate.flip"], "edit.rotate.flip", re.compile("^(h|v)$"), "h (mirror left-right) or v (mirror top-bottom)")
            if not p:
                raise EditRequirementError("edit.rotate=0 without edit.rotate.flip changes nothing; leave it unset")
        elif op == TRIM_OP:
            p.update(parse_trim(vals[TRIM_KEY]))
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
        elif op == IMAGE_INSERT_OP:
            p["image"] = image_path(vals["edit.image_insert"], "edit.image_insert")
            if "edit.image_insert.duration" not in vals:
                raise EditRequirementError("edit.image_insert needs edit.image_insert.duration (seconds the still is shown; nothing is assumed)")
            p["duration"] = float(_number(vals["edit.image_insert.duration"], "edit.image_insert.duration", *IMAGE_INSERT_DURATION))
            p["position"] = _token(vals.get("edit.image_insert.position", "before"), "edit.image_insert.position",
                                   re.compile("^(" + "|".join(IMAGE_INSERT_POSITIONS) + ")$"), "before or after")
        out[op] = {"params": p, "requirements": reqs}
    return out


_IMAGE_MAGIC = ((b"\x89PNG\r\n\x1a\n", (".png",)), (b"\xff\xd8\xff", (".jpg", ".jpeg")))


def image_path(img: Any, key: str) -> str:
    """ADR-070: the still of an `edit.image_insert` must be an image asset -- an existing PNG / JPEG whose bytes say so (a video, or a
    video renamed .png, is refused at planning time; video-editing-skill refuses it again as kind_mismatch / image_undecodable)."""
    if not isinstance(img, str) or not img.strip() or "\n" in img or "\x00" in img:
        raise EditRequirementError(f"{key} must be the path of a PNG / JPEG image")
    if any(part == ".." for part in img.replace("\\", "/").split("/")):
        raise EditRequirementError(f"{key} path contains '..' (traversal)")
    ext = os.path.splitext(img)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        raise EditRequirementError(f"{key} must be an image asset ({', '.join(IMAGE_EXTENSIONS)}), got {os.path.basename(img)!r}")
    if not os.path.isfile(img):
        raise EditRequirementError(f"{key} image not found: {img}")
    try:
        with open(img, "rb") as fh:
            head = fh.read(8)
    except OSError as e:
        raise EditRequirementError(f"{key} image is not readable: {e}")
    if not any(head.startswith(magic) and ext in exts for magic, exts in _IMAGE_MAGIC):
        raise EditRequirementError(f"{key} is not a {ext[1:].upper()} image (its bytes do not start with the format's signature): {os.path.basename(img)}")
    return os.path.abspath(img)


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
    the IR (trim → concat → speed → a joined still, ADR-070), the loudness target of its audio.loudness op, and the technical facts of its first source."""
    from .audio import audio_subjects
    assets = doc.get("assets") or {}
    vops = (doc.get("video") or {}).get("operations") or []
    aops = (doc.get("audio") or {}).get("operations") or []
    concat = next((op for op in vops if op.get("type") == "video.concat"), None)
    switch = next((op for op in vops if op.get("type") == "video.switch"), None)   # ADR-045: alternative to concat, mutually exclusive
    grid = next((op for op in vops if op.get("type") == "video.grid"), None)       # ADR-046: alternative to concat/switch, mutually exclusive
    programme_op = concat or switch or grid
    rows: List[Dict[str, Any]] = []
    audio_rows = audio_subjects(doc)   # subjects delivered as audio only (ADR-030): their picture is not part of the deliverable
    consumed = {s for r in audio_rows.values() for s in r["sources"]}
    for sid, r in sorted(audio_rows.items()):
        tech = dict((assets.get(r["sources"][0]) or {}).get("technical") or {})
        tech["video"] = None
        if r.get("channels") is not None:
            tech["audio"] = dict(tech.get("audio") or {}, channels=r["channels"])   # the planned layout (mono / stereo / down-mix), not the source's
        rows.append({"id": sid, "sources": list(r["sources"]), "duration": r["duration"], "technical": tech, "audio_only": True,
                     "filler_sources": list(r.get("filler_sources") or [])})   # ADR-072: duration is an upper bound there (QA subtracts what the Skill removed)
    if programme_op is not None:
        rows.append({"id": programme_op.get("output", PROGRAMME), "sources": list(programme_op.get("inputs") or []), "duration": float(programme_op.get("timeline_duration") or 0.0),
                     "technical": dict((assets.get((programme_op.get("inputs") or [""])[0]) or {}).get("technical") or {})})
        # assets that are not part of the programme (none today: concat/switch take every video asset) would be delivered on their own
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
        # ADR-070: a still joined before / after the finished picture adds its own duration (after the speed change, which it does not take)
        dur += sum(float(op.get("duration") or 0.0) for op in vops if op.get("asset") == row["id"] and op.get("type") == IMAGE_INSERT_OP)
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
