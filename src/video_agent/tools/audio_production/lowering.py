"""Lowering: the agent's typed operation arguments → one audio-production request document (ADR-030).

The compiler hands the adapter typed args for the single contract tool `audio-production/run`:
    {"operation": <TYPE>, "input": <id> | "inputs": [<id>…], "output": <id>, "format"?: wav|flac|…, "expect"?: {…}, <typed parameters of TYPE>}
audio-production-skill speaks in an audio request (sources / tracks / operations with typed params / outputs relative to a
workspace). This module translates meaning, never syntax: every parameter it emits is checked against the parameter schema
the Skill's own contract declares for that operation type (name, type, range, enum), so a contract change is refused instead
of being forwarded blindly. Nothing here builds commands, argv, filters or paths outside the workspace."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..base import ToolError

REQUEST_SCHEMA = "audio-production/request@1"
TOOL_ID = "audio-production/run"
SOURCE_ID = "s{n}"
TRACK_ID = "t{n}"
OPERATION_ID = "edit"
OUTPUT_ID = "out"
PROJECT_ID_FALLBACK = "op"
# agent-side argument keys every operation takes (references, output format, output expectations); parameters come from the contract
COMMON_ARGS = ("operation", "input", "inputs", "output", "format", "expect")
EXPECT_KEYS = ("channels", "sample_rate", "channel_layout", "duration", "duration_tolerance")
COMMON_KEYS = ("timeout",)
# the Skill's own forbidden fields plus the agent's boundary vocabulary: never forwarded, never accepted as a parameter name
FORBIDDEN_ARG_KEYS = ("command", "commands", "argv", "cmd", "shell", "exec", "args", "script", "binary", "executable", "executables", "env", "environment", "cwd",
                      "filter", "filters", "filter_complex", "af", "ffmpeg", "ffprobe", "api_key", "apikey", "token", "secret", "password", "credentials",
                      "workspace", "allowed_input", "allowed_inputs", "allowed_input_roots", "allowed-input", "ffmpeg_skill", "ffmpeg-skill", "ffmpeg_skill_dir", "path", "paths")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RANGE_PARAMS = ("remove", "ranges")


def safe_id(raw: str, fallback: str = PROJECT_ID_FALLBACK) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]", "_", str(raw or ""))[:64]
    return s if s and _ID_RE.match(s) else fallback


def _finite(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        raise ToolError(f"audio-production: {what} must be a finite number")
    return float(v)


def _number(v: Any, spec: Dict[str, Any], what: str) -> Any:
    x = _finite(v, what)
    if spec.get("type") == "integer":
        if float(x) != int(x):
            raise ToolError(f"audio-production: {what} must be an integer")
        x = int(x)
    if "min" in spec and x < spec["min"]:
        raise ToolError(f"audio-production: {what} {x} is below the contract minimum {spec['min']}")
    if "max" in spec and x > spec["max"]:
        raise ToolError(f"audio-production: {what} {x} is above the contract maximum {spec['max']}")
    if "enum" in spec and x not in spec["enum"]:
        raise ToolError(f"audio-production: {what} {x} is not one of {spec['enum']}")
    return x


def _ranges(v: Any, what: str) -> List[Dict[str, float]]:
    """[[start, end], …] or [{start, end}, …] → the contract's [{start, end}, …]; numbers only, the Skill validates order / bounds."""
    if not isinstance(v, list) or not v:
        raise ToolError(f"audio-production: {what} must be a non-empty list of ranges")
    out = []
    for i, r in enumerate(v):
        if isinstance(r, dict) and set(r) == {"start", "end"}:
            s, e = r["start"], r["end"]
        elif isinstance(r, (list, tuple)) and len(r) == 2:
            s, e = r
        else:
            raise ToolError(f"audio-production: {what}[{i}] must be [start, end]")
        s, e = _finite(s, f"{what}[{i}].start"), _finite(e, f"{what}[{i}].end")
        if s < 0 or not s < e:
            raise ToolError(f"audio-production: {what}[{i}] must satisfy 0 <= start < end")
        out.append({"start": round(s, 6), "end": round(e, 6)})
    return out


def _value(v: Any, spec: Dict[str, Any], what: str) -> Any:
    t = spec.get("type")
    if t in ("number", "integer"):
        return _number(v, spec, what)
    if t == "string":
        if not isinstance(v, str) or not v or len(v) > int(spec.get("max_length") or 64) or "\n" in v or "\x00" in v or not re.match(r"^[A-Za-z0-9 ._:/-]+$", v):
            raise ToolError(f"audio-production: {what} must be a short plain string")
        if "enum" in spec and v not in spec["enum"]:
            raise ToolError(f"audio-production: {what} {v!r} is not one of {spec['enum']}")
        return v
    if t == "boolean":
        if not isinstance(v, bool):
            raise ToolError(f"audio-production: {what} must be a boolean")
        return v
    if t == "array":
        items = spec.get("items") or {}
        props = items.get("properties") or {}
        if set(props) == {"start", "end"}:
            return _ranges(v, what)
        if not isinstance(v, list) or len(v) > 32:
            raise ToolError(f"audio-production: {what} must be a list")
        return [_value(x, items, f"{what}[{i}]") for i, x in enumerate(v)]
    if t == "object":
        props = spec.get("properties") or {}
        if not isinstance(v, dict):
            raise ToolError(f"audio-production: {what} must be an object")
        out = {}
        for k, x in v.items():
            if not isinstance(k, str) or k.lower() in FORBIDDEN_ARG_KEYS:
                raise ToolError(f"audio-production: refusing key {k!r} under {what}")
            if k not in props:
                raise ToolError(f"audio-production: {what}.{k} is not declared by the contract")
            out[k] = _value(x, props[k], f"{what}.{k}")
        return out
    raise ToolError(f"audio-production: {what}: contract type {t!r} is not lowered by this adapter")


class Lowering:
    """Builds requests for one contract: operations, their parameter schemas, output formats and forbidden fields come from it."""

    def __init__(self, contract: Dict[str, Any], workspace: Optional[str] = None):
        self.operations: Dict[str, Dict[str, Any]] = {o["type"]: o for o in contract.get("operations") or [] if isinstance(o, dict) and o.get("type")}
        self.unsupported: List[str] = [str(u.get("type")) for u in contract.get("unsupported_operations") or [] if isinstance(u, dict) and u.get("type")]
        self.formats: Dict[str, Dict[str, Any]] = dict(contract.get("output_formats") or {})
        self.request_schema = str((contract.get("request") or {}).get("schema") or REQUEST_SCHEMA)
        self.forbidden = tuple(set(FORBIDDEN_ARG_KEYS) | set((contract.get("request") or {}).get("forbidden_fields") or []))
        self.workspace = str(Path(workspace).resolve()) if workspace else None

    def supported_types(self) -> List[str]:
        return sorted(self.operations)

    def params_for(self, t: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Typed parameters of one operation from the agent's args, validated against the contract's parameter schema."""
        spec = self.operations[t]
        schema = spec.get("parameters") or {}
        params: Dict[str, Any] = {}
        for k, v in args.items():
            if k in COMMON_ARGS or k in COMMON_KEYS:
                continue
            if str(k).lower() in self.forbidden:
                raise ToolError(f"audio-production: refusing argument {k!r} (commands, argv, filters, executables, environment, path policy and credentials never cross the Skill boundary)")
            if k not in schema:
                raise ToolError(f"audio-production: {t} does not take parameter {k!r} (contract parameters: {sorted(schema)})")
            if v is None:
                continue
            params[k] = _value(v, schema[k], f"{t}.{k}")
        for k, ps in schema.items():
            if ps.get("required") and k not in params:
                raise ToolError(f"audio-production: {t} needs parameter {k!r}")
        return params

    def relative_output(self, path: str, workspace: Optional[str] = None) -> str:
        ws = str(Path(workspace or self.workspace or os.getcwd()).resolve())
        try:
            rel = os.path.relpath(str(Path(path).resolve()), ws)
        except ValueError as e:
            raise ToolError(f"audio-production: output must be inside the workspace {ws}: {e}")
        if rel == "." or rel.startswith("..") or os.path.isabs(rel):
            raise ToolError(f"audio-production: output {path} is outside the workspace {ws}")
        return rel.replace(os.sep, "/")

    def build_request(self, args: Dict[str, Any], paths: Dict[str, str], op_id: str = "op", timeout: Optional[float] = None,
                      workspace: Optional[str] = None) -> Tuple[Dict[str, Any], str, List[str], str]:
        """(request document, absolute output path, absolute input paths, operation type)."""
        t = args.get("operation")
        if not isinstance(t, str) or not t:
            raise ToolError("audio-production: an `operation` type is required")
        if t in self.unsupported:
            raise ToolError(f"audio-production: operation {t} is declared unsupported by the installed contract (not implemented by the Skill)")
        if t not in self.operations:
            raise ToolError(f"audio-production: operation {t} is not declared by the installed contract (unsupported operation)")
        for k in args:
            if str(k).lower() in self.forbidden:
                raise ToolError(f"audio-production: refusing argument {k!r} (commands, argv, filters, executables, environment, path policy and credentials never cross the Skill boundary)")
        params = self.params_for(t, args)
        arity = self.operations[t].get("inputs") or {}
        lo, hi = int(arity.get("min", 1)), int(arity.get("max", 1))
        ids: List[str]
        if isinstance(args.get("inputs"), list):
            ids = [str(i) for i in args["inputs"]]
        elif isinstance(args.get("input"), str) and args["input"]:
            ids = [str(args["input"])]
        else:
            raise ToolError(f"audio-production: {t} needs an `input` (or `inputs`)")
        if not (lo <= len(ids) <= hi) or not all(i for i in ids):
            raise ToolError(f"audio-production: {t} takes {lo}..{hi} input(s), got {len(ids)}")
        in_paths: List[str] = [str(Path(str(paths.get(i, i))).resolve()) for i in ids]
        fmt = args.get("format") or "wav"
        if fmt not in self.formats:
            raise ToolError(f"audio-production: output format {fmt!r} is not declared by the contract ({sorted(self.formats)})")
        if not isinstance(args.get("output"), str) or not args["output"]:
            raise ToolError(f"audio-production: {t} needs an `output`")
        out_path = os.path.abspath(paths.get(args["output"], args["output"]))
        if Path(out_path).suffix.lower() != str(self.formats[fmt].get("extension") or "").lower():
            raise ToolError(f"audio-production: output extension {Path(out_path).suffix!r} does not match format {fmt!r}")
        expect: Dict[str, Any] = {}
        if args.get("expect") is not None:
            if not isinstance(args["expect"], dict):
                raise ToolError("audio-production: expect must be an object")
            for k, v in args["expect"].items():
                if k not in EXPECT_KEYS:
                    raise ToolError(f"audio-production: expect.{k} is not an output expectation")
                expect[k] = v if k == "channel_layout" else _finite(v, f"expect.{k}")
                if k in ("channels", "sample_rate"):
                    expect[k] = int(expect[k])
        sources = [{"source_id": SOURCE_ID.format(n=n), "path": p} for n, p in enumerate(in_paths)]
        tracks = [{"track_id": TRACK_ID.format(n=n), "source_id": SOURCE_ID.format(n=n)} for n in range(len(in_paths))]
        operation = {"op_id": OPERATION_ID, "type": t, "inputs": [f"track:{TRACK_ID.format(n=n)}" for n in range(len(in_paths))], "parameters": params}
        output: Dict[str, Any] = {"output_id": OUTPUT_ID, "operation": f"op:{OPERATION_ID}", "path": self.relative_output(out_path, workspace), "format": fmt, "overwrite": True}
        if expect:
            output["expect"] = expect
        req: Dict[str, Any] = {"schema": self.request_schema,
                               "project": {"project_id": safe_id(op_id), "sources": sources, "tracks": tracks, "operations": [operation], "outputs": [output]},
                               "options": {"reuse_intermediates": True}}
        if timeout is not None and timeout > 0:
            req["options"]["timeout"] = int(min(max(1, timeout), 86400))
        return req, out_path, in_paths, t
