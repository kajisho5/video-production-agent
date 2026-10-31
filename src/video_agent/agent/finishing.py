"""Finishing vocabulary (ADR-031): colour (color-grading-skill), motion graphics (motion-graphics-skill) and the thumbnail
(thumbnail-skill) as explicit Requirements → Decisions → ProductionPlan steps → Project IR operations.

Everything here is vocabulary and arithmetic, no execution:

- **Colour** (`color.*`): `color.target` (bt709 | bt2020-pq | bt2020-hlg | bt601 → RETAG), `color.sdr` (true → HDR_TO_SDR; the
  decision compares it with the probe: an SDR source is KEEP, never tone-mapped by guessing), `color.lut` (a .cube file →
  LUT_APPLY; `color.lut.strength` 0..1), `color.strip_dovi` (true → STRIP_DOVI), `color.exposure` / `color.contrast` /
  `color.saturation` / `color.temperature` / `color.tint` (→ PRIMARY_CORRECTION; five independent, optional, range-checked
  values — none of them is a creative "look" chosen by this agent, only numbers the caller supplied), and since
  color-grading-skill 0.4.0 the same operation's `color.gamma` / `color.lift` / `color.gain` / `color.levels_in_black` /
  `color.levels_in_white` / `color.levels_out_black` / `color.levels_out_white` / `color.curves` (a curves preset name) —
  each named after the contract parameter it sets (ADR-062). Fixed order:
  strip_dovi → hdr_to_sdr → primary_correction → lut → retag (technical correction runs on the tone-mapped picture, before
  a creative LUT; color-grading-skill's own ADR-15 documents this as guidance, not something either side enforces).
- **Motion graphics** (`motion.*`): one element per type, every element type the pinned motion-graphics contract offers (ADR-064):
  `motion.title`, `motion.lower_third` (the name), `motion.text`, `motion.image` (a PNG / JPEG path), `motion.video` (a clip, picture
  in picture / chroma key), `motion.bug`, `motion.chapter`, `motion.sticker`, `motion.hook` (their text), and the switches
  `motion.progress`, `motion.countdown`, `motion.meme` (+ `.top` / `.bottom`), `motion.shape` (+ `.width` / `.height` / `.color`, ADR-078). Every other contract parameter is
  `motion.<element>.<parameter>` with the contract's range (e.g. `motion.text.font_size`, `motion.video.chromakey`), plus `.start`,
  `.end` and, for text / image overlays, `.fade`. Start / end default from policy settings recorded on the decision
  (`motion.<type>.start`, `motion.<type>.duration`) and are clipped to the subject's timeline; an explicit placement is checked
  against the picture size after the edits. All elements of a subject are rendered by one `graphics.render` operation (one Skill
  request, one encode).
- **Thumbnail** (`thumbnail.*`): `thumbnail` (true, or png | jpeg), `thumbnail.at` (seconds on the delivered timeline; default
  from policy `thumbnail.at_ratio` × duration, recorded), `thumbnail.text` (a caption rendered on the frame), `thumbnail.font_size`,
  `thumbnail.position` (center | top | bottom), `thumbnail.format`. Without text the frame is extracted as is (thumbnail/extract_frame);
  with text a ThumbnailDocument is rendered (thumbnail/render). The canvas is the subject's picture size after its edits.

Fixed order on a subject after the editing operations: color → graphics → captions.burn → loudness → export → check → thumbnail → qc.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models import Requirement
from .editing import IMAGE_EXTENSIONS, NAMED_POSITIONS, EditRequirementError, _bool, _number, _token

# ---- colour
COLOR_TOOL = "color-grading/run"
# ADR-076: PRIMARY_CORRECTION's vocabulary (which `color.<parameter>` keys exist, each one's type, range, enum and "unchanged" default) is
# read from the pinned color-grading contract (the data file the adapter pins, read as data so the agent layer never imports a tool
# package), the way ADR-064 reads motion-graphics. A re-pin that adds or narrows a parameter changes the keys and the plan-time check with
# it; tests/test_satellite_params.py::ColorContractVocabularyTests says so, and the IR schema is checked against the same contract there.
COLOR_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "tools" / "color_grading" / "contract_0.4.0.json"
_COLOR_CONTRACT: Dict[str, Any] = json.loads(COLOR_CONTRACT_PATH.read_text(encoding="utf-8"))
PRIMARY_CORRECTION_SPEC: Dict[str, Dict[str, Any]] = next(
    o for o in _COLOR_CONTRACT["operations"] if o.get("type") == "PRIMARY_CORRECTION")["parameters"]
# parameters of the same operation that are encoding / stream selection, not colour treatment: never `color.*` keys (the encode settings
# stay the Skill's own defaults; audio_stream is chosen by the agent's audio path, not per colour operation)
PRIMARY_CORRECTION_NOT_GRADING = ("crf", "preset", "audio_stream")


def _grading_param(spec: Dict[str, Any]) -> bool:
    """A shape the plan-time check can validate exactly: a bounded number / integer, or a string enum. Anything else a future contract
    declares is left out of the vocabulary (no key, so it is never sent) until it is decided, and the vocabulary test fails to say so."""
    t = spec.get("type")
    if t in ("number", "integer"):
        return isinstance(spec.get("min"), (int, float)) and isinstance(spec.get("max"), (int, float)) and not spec.get("required")
    return t == "string" and bool(spec.get("enum")) and not spec.get("required")


COLOR_CORRECTION_PARAMS: Dict[str, Dict[str, Any]] = {
    n: s for n, s in PRIMARY_CORRECTION_SPEC.items() if n not in PRIMARY_CORRECTION_NOT_GRADING and _grading_param(s)}
COLOR_ORDER = ("color.strip_dovi", "color.hdr_to_sdr", "color.primary_correction", "color.lut", "color.retag")
COLOR_OPERATIONS: Dict[str, Dict[str, Any]] = {
    "color.strip_dovi": {"skill": "color_strip_dovi", "type": "STRIP_DOVI", "params": (), "risk": "LOW"},
    "color.hdr_to_sdr": {"skill": "color_hdr_to_sdr", "type": "HDR_TO_SDR", "params": (), "risk": "MEDIUM"},
    "color.primary_correction": {"skill": "color_primary_correction", "type": "PRIMARY_CORRECTION",
                                  "params": tuple(COLOR_CORRECTION_PARAMS), "risk": "MEDIUM"},
    "color.lut":        {"skill": "color_lut",        "type": "LUT_APPLY",  "params": ("lut_strength",), "risk": "MEDIUM"},
    "color.retag":      {"skill": "color_retag",      "type": "RETAG",      "params": ("target",), "risk": "LOW"},
}
COLOR_SKILL_OF = {k: v["skill"] for k, v in COLOR_OPERATIONS.items()}
COLOR_TARGETS = ("bt709", "bt2020-pq", "bt2020-hlg", "bt601")
COLOR_KEYS = ("color.target", "color.sdr", "color.lut", "color.lut.strength", "color.strip_dovi") + tuple(f"color.{n}" for n in COLOR_CORRECTION_PARAMS)
# key -> (lo, hi) for the numeric parameters, the contract's own min / max (exposure / contrast / saturation / temperature / tint since
# ADR-031, gamma / lift / gain / levels_* since ADR-062; hand-copied until ADR-076, the same numbers)
COLOR_CORRECTION_RANGES: Dict[str, Tuple[float, float]] = {
    f"color.{n}": (s["min"], s["max"]) for n, s in COLOR_CORRECTION_PARAMS.items() if s["type"] in ("number", "integer")}
# colorlevels points are integers in 8-bit units (the contract types them `integer`); unset ones keep the Skill's default (the contract's)
COLOR_LEVELS_DEFAULTS: Dict[str, int] = {n: int(s["default"]) for n, s in COLOR_CORRECTION_PARAMS.items() if n.startswith("levels_")}
# the curves filter's built-in presets (the contract's enum; the filter's own "none" is not one: leaving color.curves unset adds no curves term)
COLOR_CURVES_PRESETS = tuple(PRIMARY_CORRECTION_SPEC["curves"]["enum"])

# ---- motion graphics
GRAPHICS_TOOL = "motion-graphics/run"
GRAPHICS_SKILL = "motion_graphics"
# ADR-064: the vocabulary is read from the pinned motion-graphics contract (the data file the adapter pins, read as data so the agent
# layer never imports a tool package): every element type it offers, each with every parameter it declares, by the contract's names
# and ranges. A re-pin that adds an element or a parameter changes this vocabulary with it (tests/test_satellite_params.py says so).
MOTION_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "tools" / "motion_graphics" / "contract_0.2.8.json"
_MOTION_CONTRACT: Dict[str, Any] = json.loads(MOTION_CONTRACT_PATH.read_text(encoding="utf-8"))
MOTION_ELEMENT_SPECS: Dict[str, Dict[str, Any]] = {str(e["type"]): e for e in _MOTION_CONTRACT["element_types"]}
# element type → (its `--set` key, the contract parameter that key's value sets; None = the key is a plain switch: the element has no
# required text / file, e.g. progress / countdown, or needs one of several, e.g. meme's top / bottom). The four ADR-031 keys keep their names.
MOTION_BASE: Dict[str, Tuple[str, Optional[str]]] = {
    "title": ("motion.title", "title"), "lower_third": ("motion.lower_third", "name"), "text_overlay": ("motion.text", "text"),
    "image_overlay": ("motion.image", "image_path"), "video_overlay": ("motion.video", "video_path"), "bug": ("motion.bug", "title"),
    "chapter": ("motion.chapter", "title"), "progress": ("motion.progress", None), "countdown": ("motion.countdown", None),
    "sticker": ("motion.sticker", "text"), "hook": ("motion.hook", "title"), "meme": ("motion.meme", None),
    # ADR-078: motion-graphics-skill 0.2.7 (its ADR-22) implements shape, a solid rectangle; a switch whose width / height / color are required
    "shape": ("motion.shape", None)}
ELEMENT_TYPES = tuple(MOTION_BASE)
# path parameters travel in the IR under a short name (the adapter turns the artifact id back into the contract's path parameter)
MOTION_PATH_PARAMS = {"image_path": "image", "video_path": "video"}
FADE_TYPES = tuple(t for a in _MOTION_CONTRACT.get("animations") or [] if a.get("kind") == "fade" for t in a.get("applies_to") or [])
MOTION_KEYS: Dict[str, Tuple[str, ...]] = {
    typ: (base,) + tuple(f"{base}.{p}" for p in MOTION_ELEMENT_SPECS[typ]["parameters"] if p != main) + (f"{base}.start", f"{base}.end")
    + ((f"{base}.fade",) if typ in FADE_TYPES else ())
    for typ, (base, main) in MOTION_BASE.items()}
# explicit DEFAULTs of the policy keys `motion.<type>.duration` (recorded on the decision); an element without one runs to the end of the
# subject (a persistent bug / progress bar / overlay / meme). chapter / sticker / hook / countdown are moments, not whole-programme layers.
ELEMENT_DEFAULT_DURATION = {"title": 5.0, "lower_third": 6.0, "chapter": 5.0, "sticker": 4.0, "hook": 3.0, "countdown": 5.0}
MOTION_FONTS = tuple(sorted((_MOTION_CONTRACT.get("fonts") or {}).get("registry") or {}))
# motion-graphics-skill model.py COLOR_RE: a colour name or bare RRGGBB, optionally @alpha (the Skill refuses anything else)
MOTION_COLOR_RE = re.compile(r"^([A-Za-z]{2,24}|[0-9A-Fa-f]{6})(@(0(\.[0-9]+)?|1(\.0+)?))?$")
# motion-graphics-skill 0.2.4 (its ADR-20) publishes the colour names it accepts (`colors.named`: FFmpeg's own table, matched in any case)
# and marks every colour parameter that cannot take @alpha with `alpha: false` (the graphics-template colours: ffmpeg-skill/graphics
# draws them opaque, and the Skill converts a name to its RRGGBB itself). The agent accepts exactly that, read from the pinned contract:
# a name from the table or bare RRGGBB everywhere, @alpha only where the parameter's schema does not say `alpha: false` (ADR-067).
MOTION_COLOR_NAMES = frozenset(str(n).lower() for n in ((_MOTION_CONTRACT.get("colors") or {}).get("named") or {}))
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".m4v", ".webm", ".mxf", ".avi", ".ts", ".mts", ".m2ts")

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
    # color.exposure / color.contrast / color.saturation / color.temperature / color.tint: five independent, optional
    # parameters (no single "main" key the way color.lut/color.target have) -- any one of them, explicitly set, asks
    # for PRIMARY_CORRECTION; unset ones are left for the Skill's own defaults (unchanged), never guessed here.
    # ADR-062: gamma / lift / gain / levels_* / curves are more parameters of the same operation, read the same way.
    # ADR-076: the keys, types, ranges and enums are the pinned contract's (COLOR_CORRECTION_PARAMS); only explicit keys are sent.
    active = [f"color.{n}" for n in COLOR_CORRECTION_PARAMS if f"color.{n}" in m and m[f"color.{n}"].provenance != "DEFAULT"]
    if active:
        p = {}
        for k in active:
            name = k.split(".", 1)[1]
            spec = COLOR_CORRECTION_PARAMS[name]
            if spec["type"] == "string":
                enum = tuple(str(x) for x in spec["enum"])
                p[name] = _token(m[k].value, k, re.compile("^(" + "|".join(re.escape(x) for x in enum) + ")$"), "one of " + ", ".join(enum))
                continue
            lo, hi = spec["min"], spec["max"]
            v = _number(m[k].value, k, lo, hi)
            if spec["type"] == "integer":
                if float(v) != int(v):
                    unit = " (8-bit units)" if name in COLOR_LEVELS_DEFAULTS else ""
                    raise EditRequirementError(f"{k} must be an integer from {lo} to {hi}{unit}, got {m[k].value!r}")
                p[name] = int(v)
            else:
                p[name] = float(v)
        # the Skill refuses black >= white (its own colorlevels ordering check); unset points count with their defaults
        lv = {n: p.get(n, d) for n, d in COLOR_LEVELS_DEFAULTS.items()}
        for side in ("in", "out"):
            if any(f"levels_{side}_{x}" in p for x in ("black", "white")) and not lv[f"levels_{side}_black"] < lv[f"levels_{side}_white"]:
                raise EditRequirementError(f"color.levels_{side}_black ({lv[f'levels_{side}_black']}) must be below color.levels_{side}_white ({lv[f'levels_{side}_white']})")
        out["color.primary_correction"] = {"params": p, "requirements": [m[k] for k in active]}
    return out


def _motion_param(v: Any, key: str, name: str, spec: Dict[str, Any]) -> Any:
    """One element parameter checked against the pinned contract's schema for it (type, range, max_length, enum; for a colour, the
    contract's colour-name table and the parameter's `alpha: false`, ADR-067)."""
    t = spec.get("type")
    if t == "path":
        return _file(v, key, IMAGE_EXTENSIONS if name == "image_path" else VIDEO_EXTENSIONS)
    if t == "string":
        if spec.get("enum"):
            return _token(v, key, re.compile("^(" + "|".join(re.escape(x) for x in spec["enum"]) + ")$"), "one of " + ", ".join(spec["enum"]))
        return _text(v, key, int(spec.get("max_length") or 500))
    if t == "color":
        if isinstance(v, str) and len(v) > 32:
            raise EditRequirementError(f"{key} must be a colour of at most 32 characters, got {v!r}")
        opaque = spec.get("alpha") is False
        c = _token(v, key, MOTION_COLOR_RE, "a colour name or RRGGBB hex" + ("" if opaque else ", optionally @alpha") + " (e.g. white, FFCC00" + ("" if opaque else ", black@0.5") + ")")
        base, _, alpha = c.partition("@")
        if not re.fullmatch(r"[0-9A-Fa-f]{6}", base) and base.lower() not in MOTION_COLOR_NAMES:
            raise EditRequirementError(f"{key}: {base!r} is not a colour name motion-graphics-skill knows (its contract's colors.named table); give RRGGBB hex instead")
        if alpha and opaque:
            raise EditRequirementError(f"{key} takes no @alpha: motion-graphics-skill draws this colour opaque (its contract says alpha: false); give RRGGBB or a colour name")
        return c
    if t == "position":
        return _position(v, key)
    if t == "font":
        return _token(v, key, re.compile("^(" + "|".join(re.escape(x) for x in MOTION_FONTS) + ")$"), "a registry font id (" + ", ".join(MOTION_FONTS) + ")")
    if t == "boolean":
        return _bool(v, key)
    x = _number(v, key, spec.get("min"), spec.get("max"))
    if t == "integer":
        if float(x) != int(x):
            raise EditRequirementError(f"{key} must be an integer from {spec.get('min')} to {spec.get('max')}, got {v!r}")
        step = spec.get("multiple_of")
        if isinstance(step, int) and step > 1 and int(x) % step:   # ADR-078: shape's width / height are even
            raise EditRequirementError(f"{key} must be a multiple of {step} (the contract's multiple_of), got {v!r}")
        return int(x)
    return float(x)


def parse_motion_requirements(m: Dict[str, Requirement]) -> Dict[str, Dict[str, Any]]:
    """→ {element type: {"params" (contract parameter names; a path parameter as `image` / `video`), "start"?, "end"?, "fade"?, "requirements"}}.
    `motion.<element>` asks for the element (its value is the parameter MOTION_BASE names, or true for a switch element) and
    `motion.<element>.<parameter>` sets any other parameter the contract declares for it, range-checked against the contract; an
    unset parameter keeps the Skill's own default (never filled in here)."""
    out: Dict[str, Dict[str, Any]] = {}
    for typ, keys in MOTION_KEYS.items():
        base, main_param = MOTION_BASE[typ]
        spec = MOTION_ELEMENT_SPECS[typ]
        main = m.get(base)
        refinements = [m[k] for k in keys[1:] if k in m and m[k].provenance != "DEFAULT"]
        if main is None or main.provenance == "DEFAULT":
            if refinements:
                raise EditRequirementError(f"{refinements[0].key} is set but {base} is not: refusing an ambiguous request")
            continue
        if main_param is None and not _bool(main.value, base):
            if refinements:
                raise EditRequirementError(f"{base}=false with {base}.* refinements: refusing an ambiguous request")
            continue
        reqs = [main] + refinements
        vals = {r.key: r.value for r in reqs}
        p: Dict[str, Any] = {}
        el: Dict[str, Any] = {"params": p, "requirements": reqs}
        for name, pspec in spec["parameters"].items():
            key = base if name == main_param else f"{base}.{name}"
            if key not in vals:
                continue
            p[MOTION_PATH_PARAMS.get(name, name)] = _motion_param(vals[key], key, name, pspec)
        missing = [f"{base}.{n}" for n, ps in spec["parameters"].items() if ps.get("required") and n != main_param and MOTION_PATH_PARAMS.get(n, n) not in p]
        if missing:   # ADR-078: a parameter the contract requires (shape's width / height / color) is asked for, never defaulted here
            raise EditRequirementError(f"{base} needs " + " / ".join(missing) + " (required by the contract)")
        for group in (c.get("at_least_one_of") for c in spec.get("constraints") or [] if c.get("at_least_one_of")):
            if not any(n in p for n in group):
                raise EditRequirementError(f"{base} needs at least one of " + " / ".join(f"{base}.{n}" for n in group) + " (the contract's constraint)")
        for k in ("start", "end"):
            if f"{base}.{k}" in vals:
                el[k] = float(_number(vals[f"{base}.{k}"], f"{base}.{k}", 0.0, float((_MOTION_CONTRACT.get("limits") or {}).get("max_element_duration_seconds") or 86400.0)))
        if "start" in el and "end" in el and not el["start"] < el["end"]:
            raise EditRequirementError(f"{base}.start must be before {base}.end")
        if f"{base}.fade" in vals:
            el["fade"] = float(_number(vals[f"{base}.fade"], f"{base}.fade", 0.01, 30.0))
        out[typ] = el
    return out


def element_label(typ: str, params: Dict[str, Any]) -> str:
    """How a decision names the element: its text, its file, or what it is."""
    for k in ("title", "name", "text"):
        if params.get(k):
            return f"'{params[k]}'"
    for k in ("image", "video"):
        if params.get(k):
            return f"'{Path(str(params[k])).name}'"
    if typ == "meme":
        return "'" + " / ".join(str(params[k]) for k in ("top", "bottom") if params.get(k)) + "'"
    if typ == "countdown":
        return f"from {params.get('count_from', MOTION_ELEMENT_SPECS[typ]['parameters']['count_from'].get('default'))}"
    if typ == "shape":
        return f"{params.get('width')}x{params.get('height')} {params.get('color')}"
    return "bar"


def placement_problems(typ: str, params: Dict[str, Any], size: Optional[Tuple[int, int]]) -> List[str]:
    """Why an element would not land on the picture it is drawn on (ADR-064): the planned size after the subject's edits (`picture_size`),
    not the source's. Only what is certain is refused: an explicit {x, y} outside the frame, a margin at least as large as the side it is
    measured from, an overlay scaled wider than the frame, text taller than the frame. An unknown size checks nothing (never guessed)."""
    if size is None:
        return []
    w, h = size
    pspec = MOTION_ELEMENT_SPECS[typ]["parameters"]
    out: List[str] = []
    pos = params.get("position", (pspec.get("position") or {}).get("default"))
    if isinstance(pos, dict):
        if not (-w < int(pos["x"]) < w and -h < int(pos["y"]) < h):
            out.append(f"position {{x: {pos['x']}, y: {pos['y']}}} lies outside the {w}x{h} picture")
    elif isinstance(pos, str) and params.get("margin") is not None and (pspec.get("margin") or {}).get("type") == "integer":
        margin = int(params["margin"])
        for side, dim in (("left", w), ("right", w), ("top", h), ("bottom", h)):
            if side in pos and margin >= dim:
                out.append(f"margin {margin} from the {side} edge puts it outside the {w}x{h} picture")
                break
    if params.get("scale_width") is not None and int(params["scale_width"]) > w:
        out.append(f"scale_width {params['scale_width']} is wider than the {w}x{h} picture")
    if params.get("font_size") is not None and int(params["font_size"]) >= h:
        out.append(f"font_size {params['font_size']} is taller than the {w}x{h} picture")
    if typ == "shape" and params.get("width") is not None and params.get("height") is not None and (int(params["width"]) > w or int(params["height"]) > h):
        out.append(f"shape {params['width']}x{params['height']} is larger than the {w}x{h} picture")   # ADR-078
    return out


def planned_picture_size(technical: Dict[str, Any], decisions: List[Any], subject: str) -> Optional[Tuple[int, int]]:
    """`picture_size` at decision time: the subject's not-rejected editing decisions, in EDIT_ORDER, stand in for the IR operations the
    planner will build from them (their params carry the same names)."""
    from .editing import EDIT_ORDER
    decs = [d for d in decisions if d.type == "TRANSFORM" and d.status != "REJECTED" and d.subject in EDIT_ORDER and d.params.get("asset_id") == subject]
    ops = [dict(d.params, type=d.subject, asset=subject) for d in sorted(decs, key=lambda d: EDIT_ORDER.index(d.subject))]
    return picture_size(technical, ops, subject)


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


# ---- QC reference metrics (ADR-058): qc-skill 0.2.3 compares a video deliverable against a master the user names
QC_REFERENCE_KEYS = ("qc.reference_video", "qc.reference_metrics", "qc.min_vmaf", "qc.min_ssim", "qc.min_psnr_db")
QC_REFERENCE_METRICS = ("vmaf", "ssim", "psnr")   # qc-skill rules.py REFERENCE_METRICS, in its canonical order
# the threshold key of each metric → (rule field, lo, hi); unset, the Skill's own defaults apply (VMAF 93, SSIM 0.95, PSNR 31 dB)
QC_REFERENCE_MINIMA = {"vmaf": ("qc.min_vmaf", "min_vmaf", 0.0, 100.0), "ssim": ("qc.min_ssim", "min_ssim", 0.0, 1.0), "psnr": ("qc.min_psnr_db", "min_psnr_db", 0.0, 100.0)}
QC_REFERENCE_EXTENSIONS = (".mp4", ".mov", ".mkv", ".m4v", ".webm", ".mxf", ".avi", ".ts", ".mts", ".m2ts")


def parse_qc_reference(m: Dict[str, Requirement]) -> Optional[Dict[str, Any]]:
    """`qc.reference_video` (the master file) + `qc.reference_metrics` (vmaf, ssim, psnr: a comma list or a list) and optional
    `qc.min_vmaf` / `qc.min_ssim` / `qc.min_psnr_db` → {"path", "metrics", "rule": {reference_metrics, min_*}, "requirements"}.
    Opt-in and explicit: every key needs the `qc` gate, the file and the metrics are named together, and a threshold only for a
    listed metric — an ambiguous request is refused, never completed by guessing. None when none of the keys is set."""
    given = [m[k] for k in QC_REFERENCE_KEYS if k in m and m[k].provenance != "DEFAULT"]
    if not given:
        return None
    if qc_requested(m) is None:
        raise EditRequirementError(f"{', '.join(r.key for r in given)} set without the qc gate (--set qc=true): refusing an ambiguous request")
    ref, met = m.get("qc.reference_video"), m.get("qc.reference_metrics")
    if ref is None or ref.provenance == "DEFAULT" or met is None or met.provenance == "DEFAULT":
        raise EditRequirementError("qc.reference_video and qc.reference_metrics are set together (the master and what to measure against it)")
    raw = met.value if isinstance(met.value, list) else str(met.value).split(",") if isinstance(met.value, str) else None
    names = [str(x).strip().lower() for x in raw or [] if str(x).strip()]
    if not names or any(n not in QC_REFERENCE_METRICS for n in names) or len(set(names)) != len(names):
        raise EditRequirementError(f"qc.reference_metrics must name one or more of {', '.join(QC_REFERENCE_METRICS)} once each, got {met.value!r}")
    metrics = [x for x in QC_REFERENCE_METRICS if x in names]
    rule: Dict[str, Any] = {"reference_metrics": metrics}
    reqs = [ref, met]
    for metric, (key, field, lo, hi) in QC_REFERENCE_MINIMA.items():
        r = m.get(key)
        if r is None or r.provenance == "DEFAULT":
            continue
        if metric not in metrics:
            raise EditRequirementError(f"{key} is set but qc.reference_metrics does not list {metric}: refusing an ambiguous request")
        rule[field] = float(_number(r.value, key, lo, hi))
        reqs.append(r)
    return {"path": _file(ref.value, "qc.reference_video", QC_REFERENCE_EXTENSIONS), "metrics": metrics, "rule": rule, "requirements": reqs}


# ---- IR arithmetic
# ffmpeg-skill grid.py's own defaults for the flags `edit.grid.cell_width` / `.cell_height` / `.gap` leave unset (its argparse defaults)
GRID_CELL_DEFAULT = (480, 270)
GRID_GAP_DEFAULT = 0


def grid_size(op: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """The picture grid.py writes: xstack places input i at column i % cols, row i // cols, each cell cell_width x cell_height and
    `gap` apart, so the frame is the cells the inputs fill plus the gaps between them. An unfilled grid with gaps is None: grid.py
    pads a colour-converted cell by its gap but leaves an untouched one bare, so the size depends on the inputs' colour tags (not guessed)."""
    cols, rows = int(op["cols"]), int(op["rows"])
    n = len(op.get("inputs") or []) or cols * rows
    cw, ch = int(op.get("cell_width") or GRID_CELL_DEFAULT[0]), int(op.get("cell_height") or GRID_CELL_DEFAULT[1])
    gap = int(op.get("gap") or GRID_GAP_DEFAULT)
    if gap and n < cols * rows:
        return None
    used_cols, used_rows = min(cols, n), -(-n // cols)
    return used_cols * cw + (used_cols - 1) * gap, used_rows * ch + (used_rows - 1) * gap


def picture_size(technical: Dict[str, Any], video_ops: List[Dict[str, Any]], subject: str) -> Optional[Tuple[int, int]]:
    """The subject's picture size after its editing operations, applied in the IR's (fixed) operation order: concat may set
    width / height; a grid is its cells (`grid_size`); crop makes the picture its rectangle (ADR-060's approved proposal or an explicit `edit.crop`, in source
    pixels, before any turn); a 90 / 270 rotation swaps width and height; resize keeps the aspect; fit / fill change it.
    None when the source size is unknown — nothing is guessed."""
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
        elif t == "video.grid" and op.get("cols") and op.get("rows"):   # the grid's own picture, not its first input (PR #81's leftover)
            size = grid_size(op)
            if size is None:
                return None
            w, h = size
        elif t == "video.crop" and op.get("width") and op.get("height"):   # the cropped picture is the rectangle (before rotate / resize / fit)
            w, h = int(op["width"]), int(op["height"])
        elif t == "video.rotate" and op.get("degrees") in (90, 270):   # ADR-062: a quarter turn swaps width and height
            w, h = h, w
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
    """One render of every element of the subject. The files an element composites (an image overlay's picture, ADR-064: a video
    overlay's clip) are also named on the operation (`image` / `video`), the way the compiler finds the inputs it registers."""
    rec: Dict[str, Any] = {"type": "graphics.render", "asset": subject, "input": input_id, "output": output, "elements": list(elements)}
    for typ, ref in (("image_overlay", "image"), ("video_overlay", "video")):
        files = [e["parameters"][ref] for e in elements if e.get("type") == typ and e.get("parameters", {}).get(ref)]
        if files:
            rec[ref] = files[0]
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
