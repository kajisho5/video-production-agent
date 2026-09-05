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
# agent-side typed argument keys, per operation type; everything else is refused (never forwarded)
ARGS: Dict[str, Tuple[str, ...]] = {
    "TRIM": ("input", "output", "start", "end", "accurate", "precision"),
    "CUT": ("input", "output", "keep", "segments", "accurate", "precision"),
    "CONCAT": ("inputs", "output", "transition", "width", "height", "fps", "mode", "pad_color"),
    "SPEED": ("input", "output", "factor"),
    "FIT": ("input", "output", "aspect", "width", "pad_color", "fps"),
    "FILL": ("input", "output", "aspect", "width", "fps"),
    "RESIZE": ("input", "output", "width", "fps"),
    "OVERLAY": ("input", "output", "image", "position", "margin", "scale", "opacity", "start", "end", "fade"),
}
FORBIDDEN_ARG_KEYS = ("command", "commands", "argv", "cmd", "shell", "exec", "args", "script", "binary", "executable", "executables", "env", "environment",
                      "filter", "filters", "filter_complex", "ffmpeg", "ffprobe", "api_key", "apikey", "token", "secret", "password", "credentials",
                      "workspace", "allowed_input", "allowed_inputs", "allowed_input_roots", "allowed-input", "ffmpeg_skill_dir", "ffmpeg-skill-dir", "path", "paths")
OUTPUT_ID = "out"      # ids used inside every request this adapter builds; the response is checked against them
OPERATION_ID = "edit"
COMMON_KEYS = ("timeout",)
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
        if args["precision"] not in ("frame", "keyframe"):
            raise ToolError("video-editing: precision must be frame or keyframe")
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
        if tr is not None:
            if not isinstance(tr, dict) or set(tr) - {"type", "duration"} or "type" not in tr or "duration" not in tr:
                raise ToolError("video-editing: transition must be {type, duration}")
            p["transition"] = {"type": _str(tr["type"], "transition.type", r"^[a-z]{1,16}$"), "duration": _num(tr["duration"], "transition.duration")}
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
    else:
        raise ToolError(f"video-editing: no lowering for operation type {t}")
    return p


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
        if t == "OVERLAY":
            img = args.get("image")
            if not isinstance(img, str) or not img:
                raise ToolError("video-editing: OVERLAY needs an `image`")
            img_path = str(Path(paths.get(img, img)).resolve())
            sources.append({"id": "image", "path": img_path, "kind": "image"})
            in_paths.append(img_path)
        if not isinstance(args.get("output"), str) or not args["output"]:
            raise ToolError(f"video-editing: {t} needs an `output`")
        # the agent's own path string (absolute, not canonicalised): ToolResult.output must equal what the executor / QA /
        # artifact registry key on, exactly as the ffmpeg-skill adapter reports it; the Skill compares canonical forms
        # (Windows 8.3 short names, symlinked temp directories resolve to a different spelling of the same file)
        out_path = os.path.abspath(paths.get(args["output"], args["output"]))
        operation: Dict[str, Any] = {"id": OPERATION_ID, "type": t, "params": params}
        if t == "CONCAT":
            operation["inputs"] = [s["id"] for s in sources]
        else:
            operation["input"] = "s0"
        req: Dict[str, Any] = {"schema": self.request_schema,
                               "project": {"id": safe_id(op_id), "sources": sources, "operations": [operation],
                                           "outputs": [{"id": OUTPUT_ID, "operation": OPERATION_ID, "path": self.relative_output(out_path, workspace)}]},
                               "options": {"overwrite": True, "reuse": True}}
        if timeout is not None and timeout > 0:
            req["options"]["timeout_seconds"] = int(min(max(1, timeout), 86400))
        return req, out_path, in_paths
