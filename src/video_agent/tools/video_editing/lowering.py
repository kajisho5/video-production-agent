"""Lowering: the agent's typed operation arguments → one video-editing request document (ported from PR #19, ADR-028).

The agent's compiler hands the adapter typed args per tool (`video-editing/cut`: {"input", "keep" | "segments", "precision" |
"accurate", "output"}; the other operation types take the contract's own parameter names). video-editing-skill speaks in an
edit request (sources / operations with allowlisted types and typed params / outputs relative to a workspace). This module
translates meaning, never syntax: every parameter it emits is typed here and checked against the parameter allowlist the
Skill's own contract declares for that operation type, so a contract change (a renamed or removed parameter) is refused
instead of being forwarded blindly. Nothing here builds commands, argv, filters or paths outside the workspace.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..base import ToolError

REQUEST_SCHEMA = "video-editing/request@1"
PREFIX = "video-editing/"
# operation types the pinned contract declares that have no lowering here. ADR-099 lowered PUNCH_IN and CONDENSE (0.5.0), so the list is
# empty; tests pin it so a type a later contract adds still fails them until it is lowered or listed here (ADR-097 addendum 6).
NOT_LOWERED: Tuple[str, ...] = ()
# agent-side typed argument keys, per operation type; everything else is refused (never forwarded)
ARGS: Dict[str, Tuple[str, ...]] = {
    "TRIM": ("input", "output", "start", "end", "accurate", "precision"),
    # ADR-099: `segment_cache` is the request's options.segment_cache (contract.segment_cache: CUT with precision frame only)
    "CUT": ("input", "output", "keep", "segments", "accurate", "precision", "segment_cache"),
    # ADR-099: `stinger` is the artifact id of the stinger file when transition.type is stinger (it becomes a video source of the request)
    "CONCAT": ("inputs", "output", "transition", "width", "height", "fps", "mode", "pad_color", "stinger"),
    "SPEED": ("input", "output", "factor", "smooth"),
    "FIT": ("input", "output", "aspect", "width", "pad_color", "fps"),
    "FILL": ("input", "output", "aspect", "width", "fps"),
    "RESIZE": ("input", "output", "width", "fps"),
    "OVERLAY": ("input", "output", "image", "position", "margin", "scale", "opacity", "start", "end", "fade"),
    # ADR-062: the three operation types the contract has declared since 0.3.0 / 0.4.0 and this module did not lower yet
    "ROTATE": ("input", "output", "degrees", "flip"),
    "FREEZE": ("input", "output", "hold", "at"),
    "REVERSE": ("input", "output"),
    # ADR-070: video-editing-skill 0.4.0 (#21) IMAGE_INSERT: a still image (the `input`, an image source) → a silent clip of `duration`
    # seconds at width x height / fps (the contract's own parameter names); the only type whose input is an image, never a video
    "IMAGE_INSERT": ("input", "output", "duration", "width", "height", "fps"),
    # ADR-078: video-editing-skill 0.4.1 CROP (one pixel rectangle, the contract's own parameter names). Lowered so the adapter can run
    # every declared type; the planner's `video_crop` step keeps calling ffmpeg-skill crop.py (ADR-060) until a requirement moves it
    "CROP": ("input", "output", "x", "y", "width", "height"),
    # ADR-099: video-editing-skill 0.5.0 PUNCH_IN (a zoom on the caller's subject box for a time range; the frame size is kept) and CONDENSE
    # (a rule per labelled span: approved proposals are cut). CONDENSE's own `output` parameter (result | removed) is the agent's `review`
    # (true = the review file of only the cut spans), because `output` is the agent's output reference for every type.
    "PUNCH_IN": ("input", "output", "start", "end", "scale", "subject"),
    "CONDENSE": ("input", "output", "segments", "rules", "default", "review"),
}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")   # the contract's formats.image_inputs (an IMAGE_INSERT input outside them is refused here)
STILL_DURATION = (0.01, 600.0)                # the contract's IMAGE_INSERT.duration ("time 0.01..600 s")
ROTATIONS = (90, 180, 270)   # the contract's ROTATE.degrees ("90 | 180 | 270, clockwise")
FLIPS = ("h", "v")           # the contract's ROTATE.flip ("h | v")
FORBIDDEN_ARG_KEYS = ("command", "commands", "argv", "cmd", "shell", "exec", "args", "script", "binary", "executable", "executables", "env", "environment",
                      "filter", "filters", "filter_complex", "ffmpeg", "ffprobe", "api_key", "apikey", "token", "secret", "password", "credentials",
                      "workspace", "allowed_input", "allowed_inputs", "allowed_input_roots", "allowed-input", "ffmpeg_skill_dir", "ffmpeg-skill-dir", "path", "paths")
PRECISIONS = ("frame", "keyframe", "auto")   # TRIM / CUT precision (auto since 0.5.0: stream copy when the copy lands within 0.2 s, else frame)
PUNCH_SCALE = (1.0, 4.0)                     # PUNCH_IN.scale: above 1, at most 4
CONDENSE_LABEL = re.compile(r"^[a-z0-9][a-z0-9_+.-]{0,39}$")   # CONDENSE segments[].label / rules keys
CONDENSE_SPEED = (0.25, 4.0)                 # CONDENSE {speed: factor}: 1/4..4, not 1
SIZE_UNITS = ("bytes", "kb", "mb", "gb", "kib", "mib", "gib")   # outputs[].max_size: exactly one of these
SIZE_FACTORS = {"bytes": 1, "kb": 1000, "mb": 1000 ** 2, "gb": 1000 ** 3, "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3}
SIZE_RANGE = (100_000, 50 * 1000 ** 3)       # 100 kB .. 50 GB
_DECLARED_TEXT = re.compile(r"^[^\x00-\x1f\x7f]{1,200}$")   # sources[].ai_generated.generator / .licence: 1..200 printable characters
STINGER_ID = "stinger"   # ADR-099: the source id of a CONCAT's stinger file
OUTPUT_ID = "out"      # ids used inside every request this adapter builds; the response is checked against them
OPERATION_ID = "edit"
# ADR-099: request-level fields any type may carry: `ai_generated` ({artifact id: {generator, licence}} -> sources[].ai_generated of the
# sources those ids become) and `max_size` ({unit: number} -> outputs[0].max_size, a .mp4 output only)
COMMON_KEYS = ("timeout", "ai_generated", "max_size")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SEG_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def op_type(tool: str) -> str:
    """'video-editing/cut' → 'CUT'."""
    if not tool.startswith(PREFIX) or tool.count("/") != 1:
        raise ToolError(f"video-editing: not a tool of this package: {tool}")
    return tool[len(PREFIX):].upper()


def safe_id(raw: str, fallback: str = "op") -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", str(raw or ""))[:64]
    return s if s and _ID_RE.match(s) else fallback


def _num(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ToolError(f"video-editing: {what} must be a number, got {type(v).__name__}")
    if v != v or v in (float("inf"), float("-inf")) or v < 0:
        raise ToolError(f"video-editing: {what} must be a finite non-negative number")
    return round(float(v), 6)


def ranges_from_args(args: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Keep ranges from the agent's `keep` ([[s, e], …], the IR form) or `segments` ("s-e,s-e", the ffmpeg-skill/cut form the
    compiler emits today). Both mean the same thing: the source ranges to keep, in output order."""
    if args.get("keep") is not None:
        keep = args["keep"]
        if not isinstance(keep, list) or not keep:
            raise ToolError("video-editing: keep must be a non-empty list of [start, end]")
        out = []
        for i, r in enumerate(keep):
            if not isinstance(r, (list, tuple)) or len(r) != 2:
                raise ToolError(f"video-editing: keep[{i}] must be [start, end]")
            out.append((_num(r[0], f"keep[{i}].start"), _num(r[1], f"keep[{i}].end")))
    elif isinstance(args.get("segments"), str):
        out = []
        for i, part in enumerate(args["segments"].split(",")):
            m = _SEG_RE.match(part)
            if not m:
                raise ToolError(f"video-editing: segments[{i}] {part.strip()!r} is not START-END")
            out.append((_num(float(m.group(1)), f"segments[{i}].start"), _num(float(m.group(2)), f"segments[{i}].end")))
        if not out:
            raise ToolError("video-editing: segments is empty")
    else:
        raise ToolError("video-editing: CUT needs keep ranges (keep=[[s, e], …] or segments='s-e,…')")
    for s, e in out:
        if not s < e:
            raise ToolError(f"video-editing: range {s}-{e}: start must be before end")
    return out


def _precision(args: Dict[str, Any]) -> str:
    if args.get("precision") is not None:
        if args["precision"] not in PRECISIONS:
            raise ToolError("video-editing: precision must be frame, keyframe or auto")
        return str(args["precision"])
    if args.get("accurate") is not None and not isinstance(args["accurate"], bool):
        raise ToolError("video-editing: accurate must be a boolean")
    return "frame" if args.get("accurate", True) else "keyframe"   # the agent's default: frame-accurate, deterministic cuts


def _int(v: Any, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ToolError(f"video-editing: {what} must be an integer")
    return v


def _str(v: Any, what: str, pattern: str = r"^[A-Za-z0-9:_.-]{1,32}$") -> str:
    if not isinstance(v, str) or not re.match(pattern, v):
        raise ToolError(f"video-editing: {what} must be a short plain token")
    return v


def params_for(t: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Skill params for one operation type from the agent's typed args. Values are typed here; the Skill validates again."""
    p: Dict[str, Any] = {}
    if t == "TRIM":
        if args.get("start") is None or args.get("end") is None:
            raise ToolError("video-editing: TRIM needs start and end")
        p["start"], p["end"], p["precision"] = _num(args["start"], "start"), _num(args["end"], "end"), _precision(args)
    elif t == "CUT":
        p["keep"] = [{"start": s, "end": e} for s, e in ranges_from_args(args)]
        p["precision"] = _precision(args)
    elif t == "CONCAT":
        tr = args.get("transition")
        if isinstance(tr, dict) and tr.get("type") == "stinger":   # ADR-099: contract.stinger -- the file is a source, its times are the maker's
            if set(tr) - {"type", "cut_point", "cover"} or "cut_point" not in tr or "cover" not in tr:
                raise ToolError("video-editing: a stinger transition is {type: stinger, cut_point, cover: {start, end}} (the file is the `stinger` argument)")
            cov = tr["cover"]
            if not isinstance(cov, dict) or set(cov) != {"start", "end"}:
                raise ToolError("video-editing: transition.cover must be {start, end}")
            cp, cs, ce = _num(tr["cut_point"], "transition.cut_point"), _num(cov["start"], "transition.cover.start"), _num(cov["end"], "transition.cover.end")
            if not cs <= cp < ce:
                raise ToolError("video-editing: transition.cut_point must lie inside transition.cover [start, end)")
            if not isinstance(args.get("stinger"), str) or not args["stinger"]:
                raise ToolError("video-editing: a stinger transition needs the `stinger` file")
            p["transition"] = {"type": "stinger", "stinger": STINGER_ID, "cut_point": cp, "cover": {"start": cs, "end": ce}}
        elif tr is not None:
            if not isinstance(tr, dict) or set(tr) - {"type", "duration"} or "type" not in tr or "duration" not in tr:
                raise ToolError("video-editing: transition must be {type, duration}")
            p["transition"] = {"type": _str(tr["type"], "transition.type", r"^[a-z]{1,16}$"), "duration": _num(tr["duration"], "transition.duration")}
        if args.get("stinger") is not None and (p.get("transition") or {}).get("type") != "stinger":
            raise ToolError("video-editing: `stinger` goes only with a transition of type stinger")
        for k in ("width", "height"):
            if args.get(k) is not None:
                p[k] = _int(args[k], k)
        if args.get("fps") is not None:
            p["fps"] = _num(args["fps"], "fps") if isinstance(args["fps"], (int, float)) else _str(args["fps"], "fps", r"^[0-9]{1,6}/[0-9]{1,6}$")
        if args.get("mode") is not None:
            p["mode"] = _str(args["mode"], "mode", r"^(pad|crop)$")
        if args.get("pad_color") is not None:
            p["pad_color"] = _str(args["pad_color"], "pad_color", r"^([a-z]{1,16}|0x[0-9A-Fa-f]{6})$")
    elif t == "SPEED":
        if args.get("factor") is None:
            raise ToolError("video-editing: SPEED needs factor")
        p["factor"] = _num(args["factor"], "factor") if isinstance(args["factor"], (int, float)) else _str(args["factor"], "factor", r"^[0-9]{1,6}/[0-9]{1,6}$")
        if args.get("smooth") is not None:
            p["smooth"] = _str(args["smooth"], "smooth", r"^(blend|interpolate)$")
    elif t in ("FIT", "FILL", "RESIZE"):
        if t != "RESIZE":
            if args.get("aspect") is None:
                raise ToolError(f"video-editing: {t} needs aspect")
            p["aspect"] = _str(args["aspect"], "aspect", r"^[1-9][0-9]{0,3}:[1-9][0-9]{0,3}$")
        if args.get("width") is not None or t == "RESIZE":
            if args.get("width") is None:
                raise ToolError("video-editing: RESIZE needs width")
            p["width"] = _int(args["width"], "width")
        if t == "FIT" and args.get("pad_color") is not None:
            p["pad_color"] = _str(args["pad_color"], "pad_color", r"^([a-z]{1,16}|0x[0-9A-Fa-f]{6})$")
        if args.get("fps") is not None:
            p["fps"] = _num(args["fps"], "fps") if isinstance(args["fps"], (int, float)) else _str(args["fps"], "fps", r"^[0-9]{1,6}/[0-9]{1,6}$")
    elif t == "OVERLAY":
        p["image"] = "image"   # the image source id (see build_request)
        pos = args.get("position")
        if isinstance(pos, dict):
            if set(pos) != {"x", "y"}:
                raise ToolError("video-editing: position must be a named position or {x, y}")
            p["position"] = {"x": _int(pos["x"], "position.x"), "y": _int(pos["y"], "position.y")}
        elif pos is not None:
            p["position"] = _str(pos, "position", r"^[a-z-]{1,16}$")
        for k in ("margin", "scale"):
            if args.get(k) is not None:
                p[k] = _int(args[k], k)
        for k in ("opacity", "start", "end", "fade"):
            if args.get(k) is not None:
                p[k] = _num(args[k], k)
    elif t == "ROTATE":
        if args.get("degrees") is not None:
            if _int(args["degrees"], "degrees") not in ROTATIONS:
                raise ToolError(f"video-editing: degrees must be one of {ROTATIONS}")
            p["degrees"] = args["degrees"]
        if args.get("flip") is not None:
            p["flip"] = _str(args["flip"], "flip", r"^(h|v)$")
        if not p:
            raise ToolError("video-editing: ROTATE needs degrees and/or flip")
    elif t == "FREEZE":
        if args.get("hold") is None:
            raise ToolError("video-editing: FREEZE needs hold")
        p["hold"] = _num(args["hold"], "hold")
        if args.get("at") is not None:
            p["at"] = _num(args["at"], "at")
    elif t == "REVERSE":
        pass
    elif t == "IMAGE_INSERT":   # ADR-070
        if args.get("duration") is None:
            raise ToolError("video-editing: IMAGE_INSERT needs duration")
        d = _num(args["duration"], "duration")
        if not STILL_DURATION[0] <= d <= STILL_DURATION[1]:
            raise ToolError(f"video-editing: IMAGE_INSERT duration must be within {STILL_DURATION[0]:g}..{STILL_DURATION[1]:g} s")
        p["duration"] = d
        for k in ("width", "height"):
            if args.get(k) is not None:
                n = _int(args[k], k)
                if n < 2 or n % 2:
                    raise ToolError(f"video-editing: IMAGE_INSERT {k} must be an even integer")
                p[k] = n
        if args.get("fps") is not None:
            p["fps"] = _num(args["fps"], "fps") if isinstance(args["fps"], (int, float)) else _str(args["fps"], "fps", r"^[0-9]{1,6}/[0-9]{1,6}$")
    elif t == "CROP":   # ADR-078
        for k in ("x", "y", "width", "height"):
            if args.get(k) is None:
                raise ToolError(f"video-editing: CROP needs {k}")
            n = _int(args[k], k)
            if n < 0 or (k in ("width", "height") and (n < 2 or n % 2)):
                raise ToolError(f"video-editing: CROP {k} must be " + ("an even integer >= 2" if k in ("width", "height") else "an integer >= 0"))
            p[k] = n
    elif t == "PUNCH_IN":   # ADR-099
        for k in ("start", "end", "scale", "subject"):
            if args.get(k) is None:
                raise ToolError(f"video-editing: PUNCH_IN needs {k}")
        p["start"], p["end"] = _num(args["start"], "start"), _num(args["end"], "end")
        if not p["start"] < p["end"]:
            raise ToolError("video-editing: PUNCH_IN start must be before end")
        sc = _num(args["scale"], "scale")
        if not PUNCH_SCALE[0] < sc <= PUNCH_SCALE[1]:
            raise ToolError(f"video-editing: PUNCH_IN scale must be above {PUNCH_SCALE[0]:g} and at most {PUNCH_SCALE[1]:g}")
        p["scale"] = sc
        box = args["subject"]
        if not isinstance(box, dict) or set(box) != {"x", "y", "width", "height"}:
            raise ToolError("video-editing: PUNCH_IN subject must be {x, y, width, height} in pixels")
        p["subject"] = {k: _int(box[k], f"subject.{k}") for k in ("x", "y", "width", "height")}
        if p["subject"]["width"] < 1 or p["subject"]["height"] < 1 or p["subject"]["x"] < 0 or p["subject"]["y"] < 0:
            raise ToolError("video-editing: PUNCH_IN subject must have x, y >= 0 and width, height >= 1")
    elif t == "CONDENSE":   # ADR-099
        segs = args.get("segments")
        if not isinstance(segs, list) or not segs or len(segs) > 5000:
            raise ToolError("video-editing: CONDENSE needs segments: a non-empty list (at most 5000) of {start, end, label}")
        out_segs = []
        for i, sg in enumerate(segs):
            if not isinstance(sg, dict) or set(sg) != {"start", "end", "label"}:
                raise ToolError(f"video-editing: CONDENSE segments[{i}] must be {{start, end, label}}")
            s_, e_ = _num(sg["start"], f"segments[{i}].start"), _num(sg["end"], f"segments[{i}].end")
            if not s_ < e_:
                raise ToolError(f"video-editing: CONDENSE segments[{i}]: start must be before end")
            if not isinstance(sg["label"], str) or not CONDENSE_LABEL.match(sg["label"]):
                raise ToolError(f"video-editing: CONDENSE segments[{i}].label must match {CONDENSE_LABEL.pattern}")
            out_segs.append({"start": s_, "end": e_, "label": sg["label"]})
        ordered = sorted(out_segs, key=lambda x: x["start"])
        for a, b in zip(ordered, ordered[1:]):
            if b["start"] < a["end"]:
                raise ToolError(f"video-editing: CONDENSE segments {a['label']} and {b['label']} overlap")
        p["segments"] = ordered
        rules = args.get("rules")
        if not isinstance(rules, dict) or not rules:
            raise ToolError("video-editing: CONDENSE needs rules: {label: cut | keep | {speed: factor}}")
        p["rules"] = {}
        for lab in sorted(rules):
            if not isinstance(lab, str) or not CONDENSE_LABEL.match(lab):
                raise ToolError(f"video-editing: CONDENSE rules label {lab!r} must match {CONDENSE_LABEL.pattern}")
            p["rules"][lab] = _condense_rule(rules[lab], f"rules.{lab}")
        missing = sorted({x["label"] for x in ordered} - set(p["rules"]))
        if missing:
            raise ToolError(f"video-editing: CONDENSE has no rule for labels {missing}")
        if args.get("default") is not None:
            p["default"] = _condense_rule(args["default"], "default")
        if args.get("review") is not None:
            if not isinstance(args["review"], bool):
                raise ToolError("video-editing: CONDENSE review must be a boolean")
            p["output"] = "removed" if args["review"] else "result"
    else:
        raise ToolError(f"video-editing: no lowering for operation type {t}")
    return p


def _condense_rule(v: Any, what: str) -> Any:
    """CONDENSE rule: "cut" | "keep" | {"speed": factor 1/4..4, not 1}."""
    if v in ("cut", "keep"):
        return v
    if isinstance(v, dict) and set(v) == {"speed"}:
        f = _num(v["speed"], f"{what}.speed")
        if not CONDENSE_SPEED[0] <= f <= CONDENSE_SPEED[1] or f == 1.0:
            raise ToolError(f"video-editing: {what}.speed must be within {CONDENSE_SPEED[0]:g}..{CONDENSE_SPEED[1]:g} and not 1")
        return {"speed": f}
    raise ToolError(f"video-editing: {what} must be cut, keep or {{speed: factor}}")


def ai_declaration(v: Any, what: str) -> Dict[str, str]:
    """ADR-099: one source's AI-generated declaration {generator, licence} (each 1..200 printable characters). The caller declares it;
    nothing is inferred from the media."""
    if not isinstance(v, dict) or set(v) != {"generator", "licence"}:
        raise ToolError(f"video-editing: {what} must be {{generator, licence}}")
    for k in ("generator", "licence"):
        if not isinstance(v[k], str) or not v[k].strip() or not _DECLARED_TEXT.fullmatch(v[k]):
            raise ToolError(f"video-editing: {what}.{k} must be 1..200 printable characters")
    return {"generator": v["generator"], "licence": v["licence"]}


def max_size(v: Any) -> Dict[str, Any]:
    """ADR-099: outputs[].max_size {unit: number}, exactly one of SIZE_UNITS, 100 kB .. 50 GB."""
    if not isinstance(v, dict) or len(v) != 1 or next(iter(v)) not in SIZE_UNITS:
        raise ToolError(f"video-editing: max_size must be {{one of {', '.join(SIZE_UNITS)}: number}}")
    unit, n = next(iter(v.items()))
    if isinstance(n, bool) or not isinstance(n, (int, float)) or not n > 0 or n != n or n == float("inf"):
        raise ToolError("video-editing: max_size must be a positive number")
    if not SIZE_RANGE[0] <= int(n * SIZE_FACTORS[unit]) <= SIZE_RANGE[1]:
        raise ToolError("video-editing: max_size must be between 100 kB and 50 GB")
    return {unit: n}


class Lowering:
    """Builds requests for one contract: the parameter allowlist per type comes from `contract.operations`."""

    def __init__(self, contract: Dict[str, Any], workspace: Optional[str] = None):
        self.operations: Dict[str, Dict[str, Any]] = dict(contract.get("operations") or {})
        self.request_schema = str((contract.get("schemas") or {}).get("request") or REQUEST_SCHEMA)
        self.workspace = str(Path(workspace).resolve()) if workspace else None

    def supported_types(self) -> List[str]:
        """Operation types both the contract declares and this module can lower; anything else is never offered."""
        return sorted(t for t in self.operations if t in ARGS)

    def contract_only(self) -> List[str]:
        """Contract types without a lowering here (drift: the Skill grew; the adapter must be reviewed)."""
        return sorted(t for t in self.operations if t not in ARGS)

    def relative_output(self, path: str, workspace: Optional[str] = None) -> str:
        """Output path relative to the Skill's --workspace (the agent's op directory); an output elsewhere is refused."""
        ws = str(Path(workspace or self.workspace or os.getcwd()).resolve())
        try:
            rel = os.path.relpath(str(Path(path).resolve()), ws)
        except ValueError as e:   # different drive on Windows
            raise ToolError(f"video-editing: output must be inside the workspace {ws}: {e}")
        if rel == "." or rel.startswith("..") or os.path.isabs(rel):
            raise ToolError(f"video-editing: output {path} is outside the workspace {ws}")
        return rel.replace(os.sep, "/")

    def build_request(self, tool: str, args: Dict[str, Any], paths: Dict[str, str], op_id: str = "op", timeout: Optional[float] = None,
                      workspace: Optional[str] = None) -> Tuple[Dict[str, Any], str, List[str]]:
        """(request document, absolute output path, absolute input paths). `paths` maps artifact ids to filesystem paths;
        `workspace` is the Skill's --workspace the output is expressed relative to (default: the lowering's workspace)."""
        t = op_type(tool)
        if t not in self.operations:
            raise ToolError(f"video-editing: operation {t} is not declared by the installed contract (unsupported operation)")
        if t not in ARGS:
            raise ToolError(f"video-editing: operation {t} has no lowering in this adapter (contract drift: review the adapter)")
        for k in args:
            if str(k).lower() in FORBIDDEN_ARG_KEYS:
                raise ToolError(f"video-editing: refusing argument {k!r} (commands, argv, filters, executables, environment, paths policy and credentials never cross the Skill boundary)")
            if k not in ARGS[t] and k not in COMMON_KEYS:
                raise ToolError(f"video-editing: {t} does not take argument {k!r}")
        allowed = set((self.operations[t].get("parameters") or {}).keys())
        params = params_for(t, args)
        extra = sorted(set(params) - allowed)
        if extra:
            raise ToolError(f"video-editing: contract for {t} does not declare parameters {extra} (contract drift; not forwarded)")
        # inputs
        if t == "CONCAT":
            ids = args.get("inputs")
            if not isinstance(ids, list) or len(ids) < 2:
                raise ToolError("video-editing: CONCAT needs an `inputs` list of two or more")
        else:
            if not isinstance(args.get("input"), str) or not args["input"]:
                raise ToolError(f"video-editing: {t} needs an `input`")
            ids = [args["input"]]
        in_paths = [str(Path(paths.get(i, i)).resolve()) for i in ids]
        sources = [{"id": f"s{n}", "path": p} for n, p in enumerate(in_paths)]
        if t == "IMAGE_INSERT":   # ADR-070: the one input is an image source; a video (or anything else) in that slot is refused here, before the Skill
            if Path(in_paths[0]).suffix.lower() not in IMAGE_EXTENSIONS:
                raise ToolError(f"video-editing: IMAGE_INSERT input must be an image ({', '.join(IMAGE_EXTENSIONS)}), got {Path(in_paths[0]).name}")
            sources[0]["kind"] = "image"
        if t == "OVERLAY":
            img = args.get("image")
            if not isinstance(img, str) or not img:
                raise ToolError("video-editing: OVERLAY needs an `image`")
            img_path = str(Path(paths.get(img, img)).resolve())
            sources.append({"id": "image", "path": img_path, "kind": "image"})
            in_paths.append(img_path)
        if t == "CONCAT" and args.get("stinger") is not None:   # ADR-099: the stinger file is a video source (the Skill refuses an operation there)
            st_path = str(Path(paths.get(args["stinger"], args["stinger"])).resolve())
            sources.append({"id": STINGER_ID, "path": st_path, "kind": "video"})
            in_paths.append(st_path)
        if args.get("ai_generated") is not None:   # ADR-099: declared by the caller per input artifact id, never inferred
            decl = args["ai_generated"]
            if not isinstance(decl, dict) or not decl:
                raise ToolError("video-editing: ai_generated must map input ids to {generator, licence}")
            slot = {ref: f"s{n}" for n, ref in enumerate(ids)}
            if t == "CONCAT" and args.get("stinger") is not None:
                slot[args["stinger"]] = STINGER_ID
            for ref, d in decl.items():
                if ref not in slot:
                    raise ToolError(f"video-editing: ai_generated names {ref!r}, which is not a video input of this {t}")
                next(x for x in sources if x["id"] == slot[ref])["ai_generated"] = ai_declaration(d, f"ai_generated[{ref}]")
        if not isinstance(args.get("output"), str) or not args["output"]:
            raise ToolError(f"video-editing: {t} needs an `output`")
        # the agent's own path string (absolute, not canonicalised): ToolResult.output must equal what the executor / QA /
        # artifact registry key on, exactly as the ffmpeg-skill adapter reports it; the Skill compares canonical forms
        # (Windows 8.3 short names, symlinked temp directories resolve to a different spelling of the same file)
        out_path = os.path.abspath(paths.get(args["output"], args["output"]))
        operation: Dict[str, Any] = {"id": OPERATION_ID, "type": t, "params": params}
        if t == "CONCAT":
            operation["inputs"] = [s["id"] for s in sources if s["id"] != STINGER_ID]   # the stinger is named by transition.stinger, not an input slot
        else:
            operation["input"] = "s0"
        req: Dict[str, Any] = {"schema": self.request_schema,
                               "project": {"id": safe_id(op_id), "sources": sources, "operations": [operation],
                                           "outputs": [{"id": OUTPUT_ID, "operation": OPERATION_ID, "path": self.relative_output(out_path, workspace)}]},
                               "options": {"overwrite": True, "reuse": True}}
        if args.get("max_size") is not None:   # ADR-099: contract.size_limit (a .mp4 output; not with an encoding, which this adapter never sends)
            if not out_path.lower().endswith(".mp4"):
                raise ToolError("video-editing: max_size needs a .mp4 output")
            req["project"]["outputs"][0]["max_size"] = max_size(args["max_size"])
        if args.get("segment_cache") is not None:   # ADR-099: contract.segment_cache (CUT with precision frame)
            if not isinstance(args["segment_cache"], bool):
                raise ToolError("video-editing: segment_cache must be a boolean")
            if args["segment_cache"] and params.get("precision") != "frame":
                raise ToolError("video-editing: segment_cache applies to a CUT with precision frame only")
            req["options"]["segment_cache"] = args["segment_cache"]
        if timeout is not None and timeout > 0:
            req["options"]["timeout_seconds"] = int(min(max(1, timeout), 86400))
        return req, out_path, in_paths
