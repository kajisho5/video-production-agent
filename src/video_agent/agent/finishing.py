"""Finishing vocabulary (ADR-031): colour (color-grading-skill), motion graphics (motion-graphics-skill) and the thumbnail
(thumbnail-skill) as explicit Requirements → Decisions → ProductionPlan steps → Project IR operations.

Everything here is vocabulary and arithmetic, no execution:

- **Colour** (`color.*`): `color.target` (bt709 | bt2020-pq | bt2020-hlg | bt601 → RETAG), `color.sdr` (true → HDR_TO_SDR; the
  decision compares it with the probe: an SDR source is KEEP, never tone-mapped by guessing), `color.lut` (a .cube file →
  LUT_APPLY; `color.lut.strength` 0..1), `color.strip_dovi` (true → STRIP_DOVI). Fixed order: strip_dovi → hdr_to_sdr → lut → retag.
- **Motion graphics** (`motion.*`): one element per type — `motion.title` (+ `.subtitle`, `.start`, `.end`), `motion.lower_third`
  (the name; + `.title`, `.start`, `.end`), `motion.text` (+ `.position`, `.start`, `.end`, `.fade`), `motion.image` (a PNG / JPEG
  path; + `.position`, `.start`, `.end`, `.fade`, `.scale_percent`). Start / end default from policy settings recorded on the
  decision (`motion.<type>.start`, `motion.<type>.duration`) and are clipped to the subject's timeline. All elements of a subject
  are rendered by one `graphics.render` operation (one Skill request, one encode).
- **Thumbnail** (`thumbnail.*`): `thumbnail` (true, or png | jpeg), `thumbnail.at` (seconds on the delivered timeline; default
  from policy `thumbnail.at_ratio` × duration, recorded), `thumbnail.text` (a caption rendered on the frame), `thumbnail.font_size`,
  `thumbnail.position` (center | top | bottom), `thumbnail.format`. Without text the frame is extracted as is (thumbnail/extract_frame);
  with text a ThumbnailDocument is rendered (thumbnail/render). The canvas is the subject's picture size after its edits.

Fixed order on a subject after the editing operations: color → graphics → captions.burn → loudness → export → check → thumbnail → qc.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ..models import Requirement
from .editing import IMAGE_EXTENSIONS, NAMED_POSITIONS, EditRequirementError, _bool, _number, _token

# ---- colour
COLOR_TOOL = "color-grading/run"
COLOR_ORDER = ("color.strip_dovi", "color.hdr_to_sdr", "color.lut", "color.retag")
COLOR_OPERATIONS: Dict[str, Dict[str, Any]] = {
    "color.strip_dovi": {"skill": "color_strip_dovi", "type": "STRIP_DOVI", "params": (), "risk": "LOW"},
    "color.hdr_to_sdr": {"skill": "color_hdr_to_sdr", "type": "HDR_TO_SDR", "params": (), "risk": "MEDIUM"},
    "color.lut":        {"skill": "color_lut",        "type": "LUT_APPLY",  "params": ("lut_strength",), "risk": "MEDIUM"},
    "color.retag":      {"skill": "color_retag",      "type": "RETAG",      "params": ("target",), "risk": "LOW"},
}
COLOR_SKILL_OF = {k: v["skill"] for k, v in COLOR_OPERATIONS.items()}
COLOR_TARGETS = ("bt709", "bt2020-pq", "bt2020-hlg", "bt601")
COLOR_KEYS = ("color.target", "color.sdr", "color.lut", "color.lut.strength", "color.strip_dovi")

# ---- motion graphics
GRAPHICS_TOOL = "motion-graphics/run"
GRAPHICS_SKILL = "motion_graphics"
ELEMENT_TYPES = ("title", "lower_third", "text_overlay", "image_overlay")
MOTION_KEYS: Dict[str, Tuple[str, ...]] = {
    "title": ("motion.title", "motion.title.subtitle", "motion.title.start", "motion.title.end"),
    "lower_third": ("motion.lower_third", "motion.lower_third.title", "motion.lower_third.start", "motion.lower_third.end"),
    "text_overlay": ("motion.text", "motion.text.position", "motion.text.start", "motion.text.end", "motion.text.fade"),
    "image_overlay": ("motion.image", "motion.image.position", "motion.image.start", "motion.image.end", "motion.image.fade", "motion.image.scale_percent"),
}
ELEMENT_DEFAULT_DURATION = {"title": 5.0, "lower_third": 6.0}    # explicit DEFAULTs of the policy keys `motion.<type>.duration` (recorded on the decision)
TEXT_MAX = {"title": 200, "lower_third": 200, "text_overlay": 500}

# ---- thumbnail
THUMBNAIL_FRAME_TOOL = "thumbnail/extract_frame"
THUMBNAIL_RENDER_TOOL = "thumbnail/render"
THUMBNAIL_FRAME_SKILL = "thumbnail_frame"
THUMBNAIL_RENDER_SKILL = "thumbnail_render"
THUMBNAIL_FORMATS = ("png", "jpeg")
THUMBNAIL_POSITIONS = ("center", "top", "bottom")
THUMBNAIL_KEYS = ("thumbnail", "thumbnail.at", "thumbnail.text", "thumbnail.format", "thumbnail.font_size", "thumbnail.position")
THUMBNAIL_FONT_SIZE_RANGE = (6, 400)
_TEXT_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _text(v: Any, key: str, max_len: int) -> str:
    if not isinstance(v, str) or not v.strip() or _TEXT_RE.search(v) or len(v) > max_len:
        raise EditRequirementError(f"{key} must be a non-empty text of at most {max_len} characters without control characters")
    return v


def _file(v: Any, key: str, exts: Tuple[str, ...]) -> str:
    if not isinstance(v, str) or not v.strip() or "\n" in v or "\x00" in v:
        raise EditRequirementError(f"{key} must be a file path")
    if any(part == ".." for part in v.replace("\\", "/").split("/")):
        raise EditRequirementError(f"{key} path contains '..' (traversal)")
    if os.path.splitext(v)[1].lower() not in exts:
        raise EditRequirementError(f"{key} must be one of {exts}")
    if not os.path.isfile(v):
        raise EditRequirementError(f"{key} file not found: {v}")
    return os.path.abspath(v)


def _position(v: Any, key: str) -> Any:
    if isinstance(v, dict):
        if set(v) != {"x", "y"}:
            raise EditRequirementError(f"{key} must be a named position or {{x, y}}")
        return {"x": int(_number(v["x"], key + ".x", -16384, 16384)), "y": int(_number(v["y"], key + ".y", -16384, 16384))}
    return _token(v, key, re.compile("^(" + "|".join(NAMED_POSITIONS) + ")$"), "one of " + ", ".join(NAMED_POSITIONS))


def parse_color_requirements(m: Dict[str, Requirement]) -> Dict[str, Dict[str, Any]]:
    """→ {IR op type: {"params", "requirements"}} for the colour operations asked for (explicit USER / PROFILE only)."""
    out: Dict[str, Dict[str, Any]] = {}
    lut = m.get("color.lut")
    if "color.lut.strength" in m and m["color.lut.strength"].provenance != "DEFAULT" and (lut is None or lut.provenance == "DEFAULT"):
        raise EditRequirementError("color.lut.strength is set but color.lut is not: refusing an ambiguous request")
    for key, op in (("color.strip_dovi", "color.strip_dovi"), ("color.sdr", "color.hdr_to_sdr"), ("color.lut", "color.lut"), ("color.target", "color.retag")):
        r = m.get(key)
        if r is None or r.provenance == "DEFAULT":
            continue
        p: Dict[str, Any] = {}
        reqs = [r]
        if op in ("color.strip_dovi", "color.hdr_to_sdr"):
            if not _bool(r.value, key):
                continue
        elif op == "color.lut":
            p["lut"] = _file(r.value, key, (".cube",))
            if "color.lut.strength" in m and m["color.lut.strength"].provenance != "DEFAULT":
                p["lut_strength"] = float(_number(m["color.lut.strength"].value, "color.lut.strength", 0.0, 1.0))
                reqs.append(m["color.lut.strength"])
        else:
            p["target"] = _token(r.value, key, re.compile("^(" + "|".join(COLOR_TARGETS) + ")$"), "one of " + ", ".join(COLOR_TARGETS))
        out[op] = {"params": p, "requirements": reqs}
    return out


def parse_motion_requirements(m: Dict[str, Requirement]) -> Dict[str, Dict[str, Any]]:
    """→ {element type: {"params" (contract parameter names; `image` is a path), "start"?, "end"?, "fade"?, "requirements"}}."""
    out: Dict[str, Dict[str, Any]] = {}
    for typ, keys in MOTION_KEYS.items():
        main = m.get(keys[0])
        refinements = [m[k] for k in keys[1:] if k in m and m[k].provenance != "DEFAULT"]
        if main is None or main.provenance == "DEFAULT":
            if refinements:
                raise EditRequirementError(f"{refinements[0].key} is set but {keys[0]} is not: refusing an ambiguous request")
            continue
        reqs = [main] + refinements
        vals = {r.key: r.value for r in reqs}
        p: Dict[str, Any] = {}
        el: Dict[str, Any] = {"params": p, "requirements": reqs}
        base = keys[0]
        if typ == "title":
            p["title"] = _text(vals[base], base, TEXT_MAX[typ])
            if f"{base}.subtitle" in vals:
                p["subtitle"] = _text(vals[f"{base}.subtitle"], f"{base}.subtitle", TEXT_MAX[typ])
        elif typ == "lower_third":
            p["name"] = _text(vals[base], base, TEXT_MAX[typ])
            if f"{base}.title" in vals:
                p["title"] = _text(vals[f"{base}.title"], f"{base}.title", TEXT_MAX[typ])
        elif typ == "text_overlay":
            p["text"] = _text(vals[base], base, TEXT_MAX[typ])
            if f"{base}.position" in vals:
                p["position"] = _position(vals[f"{base}.position"], f"{base}.position")
        else:
            p["image"] = _file(vals[base], base, IMAGE_EXTENSIONS)
            if f"{base}.position" in vals:
                p["position"] = _position(vals[f"{base}.position"], f"{base}.position")
            if f"{base}.scale_percent" in vals:
                p["scale_percent"] = float(_number(vals[f"{base}.scale_percent"], f"{base}.scale_percent", 0.1, 100.0))
        for k in ("start", "end"):
            if f"{base}.{k}" in vals:
                el[k] = float(_number(vals[f"{base}.{k}"], f"{base}.{k}", 0.0, 86400.0))
        if "start" in el and "end" in el and not el["start"] < el["end"]:
            raise EditRequirementError(f"{base}.start must be before {base}.end")
        if f"{base}.fade" in vals:
            el["fade"] = float(_number(vals[f"{base}.fade"], f"{base}.fade", 0.01, 30.0))
        out[typ] = el
    return out


def parse_thumbnail_requirements(m: Dict[str, Requirement]) -> Dict[str, Any]:
    """→ {"enabled", "format" (None = policy), "at" (None = policy ratio), "text", "font_size", "position", "requirements"}."""
    main = m.get("thumbnail")
    refinements = [m[k] for k in THUMBNAIL_KEYS[1:] if k in m and m[k].provenance != "DEFAULT"]
    out: Dict[str, Any] = {"enabled": False, "format": None, "at": None, "text": None, "font_size": None, "position": None, "requirements": []}
    if main is None or main.provenance == "DEFAULT":
        if refinements:
            raise EditRequirementError(f"{refinements[0].key} is set but thumbnail is not: refusing an ambiguous request")
        return out
    fmt: Optional[str] = None
    if isinstance(main.value, str) and main.value.strip().lower() in THUMBNAIL_FORMATS:
        fmt, enabled = main.value.strip().lower(), True
    else:
        enabled = _bool(main.value, "thumbnail")
    if not enabled:
        if refinements:
            raise EditRequirementError("thumbnail=false with thumbnail.* refinements: refusing an ambiguous request")
        return out
    reqs = [main] + refinements
    vals = {r.key: r.value for r in reqs}
    if "thumbnail.format" in vals:
        f = _token(vals["thumbnail.format"], "thumbnail.format", re.compile("^(png|jpeg)$"), "png or jpeg")
        if fmt and fmt != f:
            raise EditRequirementError("thumbnail names one format and thumbnail.format another: refusing an ambiguous request")
        fmt = f
    out.update({"enabled": True, "format": fmt, "requirements": reqs})
    if "thumbnail.at" in vals:
        out["at"] = float(_number(vals["thumbnail.at"], "thumbnail.at", 0.0, 86400.0))
    if "thumbnail.text" in vals:
        out["text"] = _text(vals["thumbnail.text"], "thumbnail.text", 2000)
    if "thumbnail.font_size" in vals:
        out["font_size"] = int(_number(vals["thumbnail.font_size"], "thumbnail.font_size", THUMBNAIL_FONT_SIZE_RANGE[0], THUMBNAIL_FONT_SIZE_RANGE[1]))
    if "thumbnail.position" in vals:
        out["position"] = _token(vals["thumbnail.position"], "thumbnail.position", re.compile("^(" + "|".join(THUMBNAIL_POSITIONS) + ")$"), "one of " + ", ".join(THUMBNAIL_POSITIONS))
    return out


def qc_requested(m: Dict[str, Requirement]) -> Optional[Requirement]:
    """The explicit `qc` requirement when it switches the QC gate on (USER / PROFILE), else None."""
    r = m.get("qc")
    if r is None or r.provenance == "DEFAULT" or not _bool(r.value, "qc"):
        return None
    return r


# ---- IR arithmetic
def picture_size(technical: Dict[str, Any], video_ops: List[Dict[str, Any]], subject: str) -> Optional[Tuple[int, int]]:
    """The subject's picture size after its editing operations (resize keeps the aspect; fit / fill change it; concat may set
    width / height). None when the source size is unknown — nothing is guessed."""
    v = (technical or {}).get("video") or {}
    w, h = v.get("width"), v.get("height")
    if not w or not h:
        return None
    w, h = int(w), int(h)
    for op in video_ops:
        if op.get("asset") != subject:
            continue
        t = op.get("type")
        if t == "video.concat" and op.get("width") and op.get("height"):
            w, h = int(op["width"]), int(op["height"])
        elif t == "video.resize" and op.get("width"):
            nw = int(op["width"])
            h = max(2, int(round(h * nw / w / 2.0)) * 2)
            w = nw
        elif t in ("video.fit", "video.fill") and op.get("aspect"):
            aw, ah = (int(x) for x in str(op["aspect"]).split(":"))
            nw = int(op.get("width") or w)
            h = max(2, int(round(nw * ah / aw / 2.0)) * 2)
            w = nw
    return w, h


def ir_color_operation(op_type: str, subject: str, params: Dict[str, Any], decision_ids: List[str], input_id: str, output: str, scope: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    spec = COLOR_OPERATIONS[op_type]
    rec: Dict[str, Any] = {"type": op_type, "asset": subject, "input": input_id, "output": output}
    if op_type == "color.lut":
        rec["lut"] = params["lut"]
    for k in spec["params"]:
        if params.get(k) is not None:
            rec[k] = params[k]
    if scope is not None:
        rec["temporal_scope"] = {"start": round(float(scope["start"]), 3), "end": round(float(scope["end"]), 3)}
    rec["decision_ids"] = list(decision_ids)
    return rec


def ir_graphics_render(subject: str, elements: List[Dict[str, Any]], decision_ids: List[str], input_id: str, output: str, scope: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"type": "graphics.render", "asset": subject, "input": input_id, "output": output, "elements": list(elements)}
    imgs = [e["parameters"]["image"] for e in elements if e.get("type") == "image_overlay" and e.get("parameters", {}).get("image")]
    if imgs:
        rec["image"] = imgs[0]
    if scope is not None:
        rec["temporal_scope"] = {"start": round(float(scope["start"]), 3), "end": round(float(scope["end"]), 3)}
    rec["decision_ids"] = list(decision_ids)
    return rec


def ir_thumbnail(subject: str, params: Dict[str, Any], decision_ids: List[str], input_id: str, output: str) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"type": "graphics.thumbnail", "asset": subject, "input": input_id, "output": output}
    for k in ("timestamp", "format", "width", "height", "text", "font_id", "font_size", "color", "position"):
        if params.get(k) is not None:
            rec[k] = params[k]
    rec["decision_ids"] = list(decision_ids)
    return rec
