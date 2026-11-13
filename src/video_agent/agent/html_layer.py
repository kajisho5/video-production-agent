"""An animated layer rendered by hyperframes-skill and placed by motion-graphics-skill's `rendered_layer` (ADR-114; roadmap step 29; Velisa
DECISION_LOG D12, ownership decision O1: hyperframes-skill makes the material, motion-graphics-skill places it).

    `--set motion.layer=hyperframes hyperframes.template=lower-third hyperframes.values=<json>` (or `hyperframes.request=<scene request .json>`)
      → TRANSFORM `graphics.html_layer` (approval floor CONFIRM) + the rendered_layer element's TRANSFORM `graphics.rendered_layer`
      → IR `graphics.html_layer {asset, output, template?, values?, font?, request?, segments?, workers?}` and the graphics.render's element
        `rendered_layer` whose `layer` is that output (an artifact of this plan, not a file the user made)
      → steps `html_scene_layer` (hyperframes-skill/render: template → scene → render --codec prores [--segments]) then `motion_graphics`

- **Opt-in only**: nothing here runs unless `motion.layer` is the word `hyperframes`; a `hyperframes.*` key without it is refused, and so is
  `motion.layer=hyperframes` without a template or a scene request. The layer is never chosen by the agent and never a default.
- **Requirements** (`--set` only, checked before analysis): `hyperframes.template` (a template name of the Skill: lower-third, title-card, ...),
  `hyperframes.values` (the template's values: a JSON object, inline or a .json file), `hyperframes.font` (.otf / .ttf / .woff2, a template's
  font), or `hyperframes.request` (the user's own scene request .json); `hyperframes.segments` asks for IN / LOOP / OUT (`--segments`): with a
  template, `IN,LOOP,OUT` seconds that add up to the template's `duration` value (given in the values); with a scene request, `true` (the
  request's own `segments`); `hyperframes.workers` (render's capture workers, auto or 1-8).
- **Timing on the element**: `motion.layer.in_duration` / `.out_duration` / `.loop` are hyperframes-skill's segments, never the user's guess:
  with segments, in / out are the IN / OUT durations and loop is true (the LOOP part repeats to fill the element's window); without, the clip
  plays once (in 0, out 0, loop false). The render's segments[] must equal them, or the step fails (adapter).
- **Decision**: per video subject, TRANSFORM `graphics.html_layer` (risk MEDIUM, approval `graphics.html_layer.approval`, floor CONFIRM: a
  browser render of new graphics). Without hyperframes-skill (not installed, or its doctor says Node, a headless Chromium or the encoder is missing) a BLOCK
  names what is missing, and the step is handed off naming hyperframes-skill/render (ADR-051 / ADR-080), as before this ADR."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models import Requirement
from .editing import EditRequirementError, _number

PREFIX = "hyperframes."
SOURCE = "hyperframes"                 # the value of motion.layer that asks for this path
LAYER_KEY = "motion.layer"
SUBJECT = "graphics.html_layer"
SKILL = "html_scene_layer"
TOOL = "hyperframes-skill/render"
PACKAGE_ID = "hyperframes-skill"
CAPABILITY = "hyperframes-skill:RENDER"
APPROVAL_KEY = ("graphics.html_layer.approval", "CONFIRM")
REQUIREMENT_KEYS = tuple(PREFIX + k for k in ("template", "values", "font", "request", "segments", "workers"))
TIMING_KEYS = tuple(f"{LAYER_KEY}.{k}" for k in ("in_duration", "out_duration", "loop"))
PARAMS = ("template", "values", "font", "request", "segments", "workers")
SEGMENT_NAMES = ("in", "loop", "out")
LAYER_MAX_SEGMENT = 60.0               # rendered_layer's in_duration / out_duration maximum (motion-graphics contract)
TEMPLATE_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
FONT_EXTENSIONS = (".otf", ".ttf", ".woff2")
MAX_VALUES_BYTES = 65536
_ON = (True, "true", "yes", "on", "1")
_OFF = (False, "false", "no", "off", "0")


def layer_id(subject: str) -> str:
    """The artifact id of the subject's rendered layer (the html_scene_layer step's output, the rendered_layer element's `layer`)."""
    return f"{subject}_html_layer"


def asks_for_layer(values: Dict[str, Any]) -> bool:
    v = values.get(LAYER_KEY)
    return isinstance(v, str) and v.strip().lower() == SOURCE


def _file(v: Any, key: str, exts: Tuple[str, ...]) -> str:
    if not isinstance(v, str) or not v.strip() or "\n" in v or "\x00" in v:
        raise EditRequirementError(f"{key} must be a file path")
    if any(part == ".." for part in v.replace("\\", "/").split("/")):
        raise EditRequirementError(f"{key} path contains '..' (traversal)")
    if os.path.splitext(v)[1].lower() not in exts:
        raise EditRequirementError(f"{key} must be one of {', '.join(exts)}")
    if not os.path.isfile(v):
        raise EditRequirementError(f"{key} file not found: {v}")
    return os.path.abspath(v)


def _values(v: Any) -> Dict[str, Any]:
    key = PREFIX + "values"
    if isinstance(v, str) and not v.lstrip().startswith(("{", "[")):
        path = _file(v, key, (".json",))
        if os.path.getsize(path) > MAX_VALUES_BYTES:
            raise EditRequirementError(f"{key}: {os.path.basename(path)} is larger than {MAX_VALUES_BYTES} bytes")
        with open(path, encoding="utf-8") as fh:
            v = fh.read()
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError as e:
            raise EditRequirementError(f"{key} must be a JSON object of the template's values: {e}")
    if not isinstance(v, dict) or not v:
        raise EditRequirementError(f"{key} must be a non-empty JSON object of the template's values (e.g. {{\"name\": \"Ada Lovelace\"}})")
    for k, x in v.items():
        if not isinstance(k, str) or not re.match(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$", k):
            raise EditRequirementError(f"{key}: {k!r} is not a value name")
        if isinstance(x, (dict, list)) or x is None:
            raise EditRequirementError(f"{key}.{k} must be a text, a number or true / false")
        if isinstance(x, str) and (len(x) > 2000 or "\x00" in x):
            raise EditRequirementError(f"{key}.{k} is not a short text")
    return dict(v)


def _segments_given(v: Any, key: str) -> Dict[str, float]:
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",")]
        if len(parts) != 3:
            raise EditRequirementError(f"{key} takes IN,LOOP,OUT seconds (e.g. 0.5,7,0.5) or true for a scene request's own segments, got {v!r}")
        v = dict(zip(SEGMENT_NAMES, parts))
    elif isinstance(v, (list, tuple)) and len(v) == 3:
        v = dict(zip(SEGMENT_NAMES, v))
    if not isinstance(v, dict) or set(v) != set(SEGMENT_NAMES):
        raise EditRequirementError(f"{key} must be IN,LOOP,OUT seconds or {{in, loop, out}}, got {v!r}")
    out = {k: float(_number(v[k], f"{key}.{k}", 0.0, 3600.0)) for k in SEGMENT_NAMES}
    if out["loop"] <= 0:
        raise EditRequirementError(f"{key}.loop must be > 0 (the part that repeats to fill the element's window)")
    for k in ("in", "out"):
        if out[k] > LAYER_MAX_SEGMENT:
            raise EditRequirementError(f"{key}.{k} is {out[k]:g}s; motion-graphics-skill's rendered_layer takes at most {LAYER_MAX_SEGMENT:g}s for {k}_duration")
    return out


def parse_html_layer_requirements(values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The layer asked for ({template, values?, font?, segments?, workers?} or {request, segments?, workers?}) or None; EditRequirementError on
    anything refused (never corrected)."""
    values = values or {}
    given = {k: v for k, v in values.items() if k.startswith(PREFIX) and v is not None}
    unknown = sorted(k for k in given if k not in REQUIREMENT_KEYS)
    if unknown:
        raise EditRequirementError(f"unknown requirement key(s): {', '.join(unknown)}; known: {', '.join(REQUIREMENT_KEYS)}")
    if not asks_for_layer(values):
        if given:
            raise EditRequirementError(f"{', '.join(sorted(given))} render the layer that {LAYER_KEY}={SOURCE} places; {LAYER_KEY} is "
                                       + ("not set" if values.get(LAYER_KEY) is None else f"{values.get(LAYER_KEY)!r} (a file of your own)"))
        return None
    timing = sorted(k for k in TIMING_KEYS if values.get(k) is not None)
    if timing:
        raise EditRequirementError(f"{', '.join(timing)}: with {LAYER_KEY}={SOURCE} the in / out / loop timing is hyperframes-skill's segments "
                                   f"(ask for them with {PREFIX}segments), never set by hand")
    template, request = given.get(PREFIX + "template"), given.get(PREFIX + "request")
    if bool(template) == bool(request):
        raise EditRequirementError(f"{LAYER_KEY}={SOURCE} needs exactly one of {PREFIX}template (with {PREFIX}values) or {PREFIX}request (a scene request .json)")
    out: Dict[str, Any] = {}
    seg = given.get(PREFIX + "segments")
    seg_on = _value(seg)
    if template:
        if not isinstance(template, str) or not TEMPLATE_RE.match(template):
            raise EditRequirementError(f"{PREFIX}template must be a hyperframes-skill template name (e.g. lower-third, title-card), got {template!r}")
        out["template"] = template
        if PREFIX + "values" not in given:
            raise EditRequirementError(f"{PREFIX}template={template} needs {PREFIX}values (the template's values, e.g. {{\"name\": \"Ada Lovelace\"}})")
        out["values"] = _values(given[PREFIX + "values"])
        if PREFIX + "font" in given:
            out["font"] = _file(given[PREFIX + "font"], PREFIX + "font", FONT_EXTENSIONS)
        if seg is not None and seg_on not in _OFF:
            if seg_on in _ON:
                raise EditRequirementError(f"{PREFIX}segments=true reads a scene request's own segments; a template declares none: give IN,LOOP,OUT seconds "
                                           "(e.g. 0.5,7,0.5) that add up to its duration")
            s = _segments_given(seg, PREFIX + "segments")
            dur = out["values"].get("duration")
            if isinstance(dur, bool) or not isinstance(dur, (int, float)):
                raise EditRequirementError(f"{PREFIX}segments with a template needs the template's duration in {PREFIX}values (in + loop + out must add up to it)")
            if abs(sum(s.values()) - float(dur)) > 1e-6:
                raise EditRequirementError(f"{PREFIX}segments: in + loop + out = {sum(s.values()):g}s, the template's duration is {float(dur):g}s")
            out["segments"] = s
    else:
        for k in ("values", "font"):
            if PREFIX + k in given:
                raise EditRequirementError(f"{PREFIX}{k} fills a template; a scene request ({PREFIX}request) carries its own")
        out["request"] = _file(request, PREFIX + "request", (".json",))
        if seg is not None and seg_on not in _OFF:
            if seg_on not in _ON:
                raise EditRequirementError(f"{PREFIX}segments with a scene request is true (its own segments); edit the request to change them")
            try:
                with open(out["request"], encoding="utf-8") as fh:
                    declared = (json.load(fh) or {}).get("segments")
            except (OSError, ValueError, AttributeError) as e:
                raise EditRequirementError(f"{PREFIX}request: not a JSON scene request ({e})")
            if declared is None:
                raise EditRequirementError(f"{PREFIX}segments=true: {os.path.basename(out['request'])} declares no segments {{in, loop, out}}")
            out["segments"] = _segments_given(declared, f"{os.path.basename(out['request'])} segments")
    w = given.get(PREFIX + "workers")
    if w is not None:
        if isinstance(w, str) and w.strip().lower() == "auto":
            out["workers"] = "auto"
        else:
            n = _number(w, PREFIX + "workers", 1, 8)
            if float(n) != int(n):
                raise EditRequirementError(f"{PREFIX}workers must be auto or an integer 1-8, got {w!r}")
            out["workers"] = int(n)
    return out


def _value(v: Any) -> Any:
    return v.strip().lower() if isinstance(v, str) else v


def check_html_layer_keys(user_requirements: Dict[str, Any]) -> None:
    parse_html_layer_requirements(user_requirements)


def element_timing(p: Dict[str, Any]) -> Dict[str, Any]:
    """rendered_layer's in_duration / out_duration / loop from the segments asked for (hyperframes-skill's IN / LOOP / OUT); without segments
    the clip plays once from the element's start."""
    s = p.get("segments")
    if s:
        return {"in_duration": float(s["in"]), "out_duration": float(s["out"]), "loop": True}
    return {"in_duration": 0.0, "out_duration": 0.0, "loop": False}


def describe(p: Dict[str, Any]) -> str:
    what = f"template {p['template']}" if p.get("template") else f"scene request {os.path.basename(p['request'])}"
    seg = p.get("segments")
    return what + (f", IN {seg['in']:g}s / LOOP {seg['loop']:g}s / OUT {seg['out']:g}s" if seg else ", played once")


def decide_html_layer(eng: Any, p: Dict[str, Any], subject: str, ev: List[str], reqs: List[Requirement], caps: Dict[str, Any], approval_for: Callable) -> None:
    """One subject's layer render: the TRANSFORM (the ask, so the plan has its step) and, when hyperframes-skill cannot render here, a BLOCK that
    says why (not installed: the step is handed off; installed: its doctor's missing capabilities -- Node, a headless Chromium, the encoder)."""
    eng.decide(subject=SUBJECT, type="TRANSFORM", decision=f"render the motion.layer clip on {subject} with {PACKAGE_ID}: {describe(p)} (ProRes 4444 with alpha)",
               reason=f"user asked for {LAYER_KEY}={SOURCE} ({reqs[0].source}); {PACKAGE_ID} writes the scene and renders it to ProRes 4444 with alpha "
                      "(motion-graphics-skill's rendered_layer refuses a VP9 alpha WebM), verified against the scene's size / duration / fps"
                      + ("; the IN / LOOP / OUT segments give the element's in / out / loop timing" if p.get("segments") else ""),
               confidence=1.0, evidence=ev, risk="MEDIUM", approval=approval_for(SUBJECT, explicit=reqs[0], floor="CONFIRM"), provenance="USER",
               params={"asset_id": subject, **{k: p[k] for k in PARAMS if p.get(k) is not None}, "codec": "prores"}, requirements=reqs, serves_intent=None)
    pkg, render = caps.get(PACKAGE_ID), caps.get(CAPABILITY)
    if getattr(render, "status", None) == "AVAILABLE":
        return
    if pkg is None or getattr(pkg, "status", "MISSING") != "AVAILABLE":
        detail = str(getattr(pkg, "detail", "") or "")
        installed = detail.startswith("found at")
        why = (f"{PACKAGE_ID} is installed but cannot run: {detail}" if installed else
               f"{PACKAGE_ID} is not installed here ({detail or 'no checkout found'}); the step is handed off naming {TOOL} (codec prores"
               + (", --segments" if p.get("segments") else "") + ")")
        missing = [PACKAGE_ID, CAPABILITY]
    else:
        evd = getattr(render, "evidence", None) or {}
        why = f"{PACKAGE_ID}'s doctor: {evd.get('reason') or 'render is not usable here'}" + (f" ({evd['detail']})" if evd.get("detail") else "") \
            + (f"; fix: {evd['install']}" if evd.get("install") else "")
        missing = [CAPABILITY]
    eng.decide(subject=f"capability.{SKILL}", type="BLOCK", decision=f"BLOCK: the motion.layer clip on {subject} cannot be rendered here",
               reason=why + " (never rendered another way, never replaced by a file the agent picks)", confidence=1.0,
               evidence=[f"capability:{c}" for c in missing] + ev, risk="HIGH", approval="BLOCK", provenance="SYSTEM", params={"skill": SKILL, "missing": missing},
               requirements=reqs)


IR_KEYS = ("type", "asset", "output", "decision_ids", "codec") + PARAMS


def html_layer_problems(op: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    extra = sorted(k for k in op if k not in IR_KEYS)
    if extra:
        errs.append(f"{SUBJECT}: parameters {extra} are not in the operation's vocabulary")
    if bool(op.get("template")) == bool(op.get("request")):
        errs.append(f"{SUBJECT} on {op.get('asset')}: needs exactly one of template or request")
    if op.get("template") and not isinstance(op.get("values"), dict):
        errs.append(f"{SUBJECT} on {op.get('asset')}: a template needs its values")
    if op.get("output") != layer_id(str(op.get("asset"))):
        errs.append(f"{SUBJECT} on {op.get('asset')}: output must be {layer_id(str(op.get('asset')))}")
    if op.get("codec") not in (None, "prores"):
        errs.append(f"{SUBJECT} on {op.get('asset')}: codec must be prores (rendered_layer needs alpha)")
    s = op.get("segments")
    if s is not None and (not isinstance(s, dict) or set(s) != set(SEGMENT_NAMES) or not float(s.get("loop") or 0) > 0):
        errs.append(f"{SUBJECT} on {op.get('asset')}: segments must be {{in, loop, out}} with loop > 0")
    return errs
