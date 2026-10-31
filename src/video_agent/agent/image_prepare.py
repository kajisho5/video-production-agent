"""Image preparation of an overlay asset (ADR-077; imagemagick-skill roadmap step 4, agent side): before video-editing composites an
`edit.overlay` image onto the picture, imagemagick-skill brings it to the size the user asked for.

Everything here is vocabulary and arithmetic, no execution:

- **Requirements** (`edit.overlay.prepare.*`, explicit `--set` only, refinements of `edit.overlay`): `.width` + `.height` (both, pixels) →
  imagemagick-skill `resize` to that box, `.mode` (the contract's enum fit | fill | exact; its default fit is recorded when omitted),
  `.pad` (a colour, or `none` for transparency) → `pad` to exactly width x height after a fit, `.gravity` (the contract's enum, default
  center) where the image sits on that canvas, `.strip` (true) → `strip` GPS / EXIF first, baking the EXIF orientation into the pixels so the
  box applies to the upright picture. Names, types, enums and defaults are read from the pinned imagemagick-skill contract (the data file the
  adapter pins; the agent layer never imports a tool package, as ADR-064 / ADR-076 read motion-graphics / color-grading). The agent chooses
  no size and no colour: a resize needs both sides, a pad needs a colour.
- **Plan**: one step per preparation, in the order strip → resize → pad, before the overlay step of the same subject (`image_strip`,
  `image_prepare` → `imagemagick-skill/resize`, `image_pad`), each on the previous one's output; the overlay composites the last.
- **IR**: the `video.overlay` record carries `prepare` ({width, height, mode, pad?, gravity?, strip?}); `image` stays the user's file.

A request without `edit.overlay.prepare.*` plans exactly what it planned before (no step, no IR key)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "tools" / "imagemagick" / "contract_0.8.1.json"
_CONTRACT: Dict[str, Any] = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
_TOOLS: Dict[str, Dict[str, Any]] = {str(t.get("name")): t for t in _CONTRACT.get("tools") or [] if isinstance(t, dict)}
PACKAGE = str((_CONTRACT.get("skill") or {}).get("id"))   # imagemagick-skill


def contract_parameter(tool: str, name: str) -> Dict[str, Any]:
    """One input_schema property of a pinned imagemagick-skill tool (`resize`, `pad`, `strip`)."""
    return dict(((_TOOLS[tool].get("input_schema") or {}).get("properties") or {})[name])


PREFIX = "edit.overlay.prepare."
# requirement key suffix → (imagemagick-skill tool, its parameter); the parameter's type / enum / default come from the pinned contract
KEY_PARAMETERS: Dict[str, Tuple[str, str]] = {"width": ("resize", "width"), "height": ("resize", "height"), "mode": ("resize", "mode"),
                                              "pad": ("pad", "color"), "gravity": ("pad", "gravity")}
STRIP_KEY = PREFIX + "strip"
KEYS: Tuple[str, ...] = tuple(PREFIX + k for k in KEY_PARAMETERS) + (STRIP_KEY,)
RESIZE_MODES: Tuple[str, ...] = tuple(contract_parameter("resize", "mode")["enum"])
RESIZE_MODE_DEFAULT: str = contract_parameter("resize", "mode")["default"]
GRAVITIES: Tuple[str, ...] = tuple(contract_parameter("pad", "gravity")["enum"])
GRAVITY_DEFAULT: str = contract_parameter("pad", "gravity")["default"]
SIDE_RANGE = (1, 16384)   # the agent's bound (the contract states none; resize.py itself refuses anything not positive)
# pad's colour: the subset of the contract's syntax (a name, #RRGGBB, #RRGGBBAA, rgb()/rgba(), none) the agent sends; rgb() / rgba() are not taken
COLOR_PATTERN = r"^(none|[a-z]{1,32}|#[0-9A-Fa-f]{6}|#[0-9A-Fa-f]{8})$"

# production skills (skills/registry.py) and the imagemagick-skill tool each one runs, in execution order
STRIP_SKILL, RESIZE_SKILL, PAD_SKILL = "image_strip", "image_prepare", "image_pad"
SKILL_TOOL = {STRIP_SKILL: "strip", RESIZE_SKILL: "resize", PAD_SKILL: "pad"}
ORDER = (STRIP_SKILL, RESIZE_SKILL, PAD_SKILL)
PREPARE_FIELDS = ("width", "height", "mode", "pad", "gravity", "strip")


def prepare_chain(prep: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """[(skill, tool name, tool parameters)] of an IR `prepare` record, in execution order (strip → resize → pad)."""
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    if prep.get("strip"):
        out.append((STRIP_SKILL, "strip", {}))
    if prep.get("width") is not None:
        out.append((RESIZE_SKILL, "resize", {"width": prep["width"], "height": prep.get("height"), "mode": prep.get("mode") or RESIZE_MODE_DEFAULT}))
        if prep.get("pad") is not None:
            out.append((PAD_SKILL, "pad", {"width": prep["width"], "height": prep.get("height"), "color": prep["pad"], "gravity": prep.get("gravity") or GRAVITY_DEFAULT}))
    return out


def prepared_id(subject: str, tool: str) -> str:
    """The artifact id of an overlay image after one preparation step."""
    return f"{subject}_overlay_image_{tool}"


def describe(prep: Dict[str, Any]) -> str:
    parts = []
    if prep.get("strip"):
        parts.append("GPS / EXIF stripped")
    if prep.get("width") is not None:
        parts.append(f"resized to {prep['width']}x{prep['height']} ({prep.get('mode') or RESIZE_MODE_DEFAULT})")
        if prep.get("pad") is not None:
            parts.append(f"padded to {prep['width']}x{prep['height']} with {prep['pad']} ({prep.get('gravity') or GRAVITY_DEFAULT})")
    return ", ".join(parts)


def prepare_problems(prep: Any) -> List[str]:
    """What is wrong with an IR `prepare` record (the validator's re-check of what the parser produced)."""
    if not isinstance(prep, dict):
        return ["prepare must be an object"]
    errs = [f"prepare.{k} is not in the vocabulary" for k in sorted(prep) if k not in PREPARE_FIELDS]
    w, h = prep.get("width"), prep.get("height")
    if (w is None) != (h is None):
        errs.append("prepare needs both width and height")
    for k, v in (("width", w), ("height", h)):
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or not SIDE_RANGE[0] <= v <= SIDE_RANGE[1]):
            errs.append(f"prepare.{k} {v!r} must be an integer within {SIDE_RANGE[0]}..{SIDE_RANGE[1]}")
    if prep.get("mode") is not None and prep["mode"] not in RESIZE_MODES:
        errs.append(f"prepare.mode {prep['mode']!r} must be one of {', '.join(RESIZE_MODES)}")
    if prep.get("mode") is not None and w is None:
        errs.append("prepare.mode needs width and height")
    if prep.get("pad") is not None:
        if not isinstance(prep["pad"], str) or not re.match(COLOR_PATTERN, prep["pad"]):
            errs.append(f"prepare.pad {prep['pad']!r} is not a colour this agent sends")
        if w is None or (prep.get("mode") or RESIZE_MODE_DEFAULT) != "fit":
            errs.append("prepare.pad pads a fit result to the width x height canvas (fill / exact already have that size)")
    if prep.get("gravity") is not None and (prep["gravity"] not in GRAVITIES or prep.get("pad") is None):
        errs.append(f"prepare.gravity must be one of {', '.join(GRAVITIES)} and needs pad")
    if prep.get("strip") not in (None, True):
        errs.append("prepare.strip is true or absent")
    if not prepare_chain(prep):
        errs.append("prepare names no preparation")
    return errs
