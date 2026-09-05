"""ThumbnailAdapter: the agent's boundary to thumbnail-skill (deterministic thumbnail rendering Skill, ADR-031).

    video-production-agent ─(typed Operation)─→ ThumbnailAdapter ─({"tool", "params"} on stdin)─→ `thumbnail run - --json`
        ─(Pillow raster ops; video frames through typed ffmpeg-skill probe/look calls)─→ PNG / JPEG

Protocol (from the Skill's own contract, `thumbnail skill --json`):
    contract → `thumbnail skill --json`
    doctor   → `thumbnail doctor --json [--workspace D] [--allowed-input R]… [--ffmpeg-skill X]`
    run      → `thumbnail run - --json --workspace D --allowed-input R… [--ffmpeg-skill X] [--timeout S]` (request on stdin)
    request  → {"tool": "thumbnail/render" | "thumbnail/extract_frame", "params": {...}}   (workspace / roots / timeout ride on argv, never in params)
    response ← one JSON document {"ok": true, "tool", "result": <thumbnail-skill/response@1>}; the outer "ok" only means "dispatched",
               the tool's own verdict is result.ok / result.status; a request the transport itself refused comes back as a bare
               response document {"ok": false, "error": {code, message, retryable, details}}

The agent hands one typed operation and this adapter lowers it deterministically:
    thumbnail/extract_frame: {"input": <video id>, "timestamp": s ≥ 0, "format": png|jpeg, "jpeg_quality"?: 1..100, "output": <id>}
        → params {source: {path, timestamp}, output: {path, format, overwrite: true, jpeg_quality?}}
    thumbnail/render: {"input", "timestamp", "format", "output", "width", "height" (canvas 16..7680), "background"?: "#RRGGBB", "text"?,
        "font_id"? (contract font_ids, default sans-bold), "font_size"? (6..400, default 48), "color"? (default #FFFFFF),
        "position"?: center|top|bottom (default bottom), "jpeg_quality"?}
        → a ThumbnailDocument: document_id from the op id, one video_frame asset "frame", one image element "frame" covering the canvas,
          and, when text is given, one text element "caption" anchored by `position` with the Skill's own align vocabulary.
Every value is checked against the ranges the contract declares (canvas, font_size, text length, font_ids, output formats and
extensions); unknown parameters and forbidden keys are refused by name. The response is verified: transport envelope and tool id,
schema / skill id / version / status, the output's realpath, sha256 recomputed from the file, size, format and dimensions, and the
provenance chain (skill, version, identity, output_hash, the source's sha256). The Skill reports no commands; `commands` is [].

Identity discrepancy (documented, not patched): the Skill calls itself "thumbnail-skill" (contract skill_id / response skill.id /
provenance.skill) while its tool ids are "thumbnail/<name>". The agent package id is therefore SKILL_ID = "thumbnail" (the tool
prefix) and CONTRACT_SKILL_ID = "thumbnail-skill" is what the Skill's own documents are checked against. The contract also declares
`thumbnail/validate` (structural validation, no output); the agent does not use it. Facts, never "fixed": extract_frame delivers the
frame at ffmpeg-skill/look's default width (1280 px) regardless of the source; the Skill has no dry-run mode (a dry run here lowers
and validates the request but never invokes the Skill)."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...models import Observation, Operation, ToolResult
from ...skills.contract import SkillPackage, ToolSpec
from ..base import ToolAdapter, ToolError
from ..ffmpeg_skill.adapter import PathPolicy
from ..skill_process import (FORBIDDEN_ARG_KEYS, CliSkill, ContractError, as_dict, drift_report, error_table, failed_result, fingerprint_matches, invoke, one_json_document,
                             remove_fresh, same_file, scan_forbidden, scrub, strip_sha_prefix)
from .locate import locate_thumbnail

SKILL_ID = "thumbnail"
PREFIX = SKILL_ID + "/"
CONTRACT_SKILL_ID = "thumbnail-skill"
TOOL_RENDER = "thumbnail/render"
TOOL_EXTRACT_FRAME = "thumbnail/extract_frame"
TOOLS = (TOOL_RENDER, TOOL_EXTRACT_FRAME)
CONTRACT_SCHEMA = "thumbnail-skill/contract@1"
REQUEST_SCHEMA = "thumbnail-skill/request@1"
RESPONSE_SCHEMA = "thumbnail-skill/response@1"
DOCTOR_SCHEMA = "thumbnail-skill/doctor@1"
SUPPORTED_SKILL_VERSIONS = ("0.1.",)
ENGINE_ID = "ffmpeg-skill"
CANONICAL_INVOCATION = ["thumbnail", "run", "-", "--json"]
REQUIRED_EXECUTION_FLAGS = {"shell": False, "arbitrary_executables": False, "arbitrary_filters": False, "network": False, "input_mutation": False, "ai": False}
COMMON_ARGS = ("input", "output", "format", "timestamp", "jpeg_quality")
RENDER_ARGS = ("width", "height", "background", "text", "font_id", "font_size", "color", "position")
POSITIONS = ("center", "top", "bottom")
DEFAULTS = {"font_id": "sans-bold", "font_size": 48, "color": "#FFFFFF", "position": "bottom", "background": "#000000"}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_RANGE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\.\.(-?\d+(?:\.\d+)?)")
DRIFT_KEYS = ("schema", "skill_id", "version", "kind", "execution", "document", "output_formats", "fonts", "rendering", "ffmpeg_skill", "schema_versions", "errors",
              "provenance", "deterministic")
DRIFT_TOOL_KEYS = ("role", "produces_output", "deterministic", "writes_media", "mutates_input", "input", "output", "delegates_to")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def _range(text: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    """"16..7680" (possibly with a trailing note) → (16, 7680); the pinned default when the contract phrases it differently."""
    m = _RANGE_RE.search(str(text or ""))
    return (float(m.group(1)), float(m.group(2))) if m else default


def check_contract(contract: Any) -> List[str]:
    """Compatibility checks: schema, id, version range, kind, execution flags, canonical invocation, the two execution tools, the
    document vocabulary (schema, id pattern, forbidden fields, element types, fonts), output formats, error codes. Anything off is
    refused, never patched."""
    errs: List[str] = []
    if not isinstance(contract, dict):
        return ["contract is not an object"]
    if contract.get("schema") != CONTRACT_SCHEMA:
        errs.append(f"contract schema {contract.get('schema')!r} != {CONTRACT_SCHEMA}")
    if contract.get("skill_id") != CONTRACT_SKILL_ID or contract.get("id") != CONTRACT_SKILL_ID:
        errs.append(f"skill_id {contract.get('skill_id')!r} / id {contract.get('id')!r} != {CONTRACT_SKILL_ID}")
    ver = str(contract.get("version") or "")
    if not ver.startswith(SUPPORTED_SKILL_VERSIONS):
        errs.append(f"skill version {ver!r} not in supported range {SUPPORTED_SKILL_VERSIONS}")
    if contract.get("kind") != "execution":
        errs.append(f"kind {contract.get('kind')!r} != execution")
    ex = contract.get("execution") or {}
    if ex.get("mode") != "local":
        errs.append(f"execution.mode {ex.get('mode')!r} != local")
    for k, want in REQUIRED_EXECUTION_FLAGS.items():
        if ex.get(k) is not want:
            errs.append(f"execution.{k} must be {want!r}, contract says {ex.get(k)!r}")
    if list(ex.get("canonical_invocation") or []) != CANONICAL_INVOCATION:
        errs.append(f"canonical_invocation {ex.get('canonical_invocation')!r} != {CANONICAL_INVOCATION}")
    fs = contract.get("ffmpeg_skill") or {}
    if str(fs.get("contract_version")) != "1.0" or not {"probe", "look"} <= {str(t) for t in fs.get("tools_used") or []}:
        errs.append("ffmpeg_skill.contract_version 1.0 / tools_used probe+look missing")
    tools = {str(t.get("tool_id")): t for t in contract.get("tools") or [] if isinstance(t, dict)}
    for tid in TOOLS:
        t = tools.get(tid)
        if t is None:
            errs.append(f"contract lacks tool {tid} (declares {sorted(tools)})")
            continue
        if t.get("role") != "execution" or t.get("produces_output") is not True or t.get("deterministic") is not True or t.get("mutates_input") is not False:
            errs.append(f"tool {tid} must declare role execution, produces_output / deterministic = true, mutates_input = false")
        if not {"ffmpeg-skill/probe", "ffmpeg-skill/look"} <= {str(d) for d in t.get("delegates_to") or []}:
            errs.append(f"tool {tid} must delegate video frames to ffmpeg-skill/probe and ffmpeg-skill/look")
    for tid, t in tools.items():
        if not tid.startswith(PREFIX):
            errs.append(f"tool id {tid!r} does not carry the {PREFIX} prefix")
    doc = contract.get("document") or {}
    if doc.get("schema") != REQUEST_SCHEMA:
        errs.append(f"document.schema {doc.get('schema')!r} != {REQUEST_SCHEMA}")
    if doc.get("id_pattern") != _ID_RE.pattern:
        errs.append(f"document.id_pattern {doc.get('id_pattern')!r} != {_ID_RE.pattern}")
    forbidden = set(doc.get("forbidden_fields") or []) & set(ex.get("forbidden_request_fields") or [])
    for k in ("command", "argv", "filter", "shell", "exec", "env"):
        if k not in forbidden:
            errs.append(f"forbidden fields lack {k!r}")
    els = as_dict(doc.get("elements"))
    if sorted(str(x) for x in els.get("types") or []) != ["image", "text"]:
        errs.append(f"document.elements.types {els.get('types')!r} != ['image', 'text']")
    align = as_dict(as_dict(as_dict(els.get("text")).get("fields")).get("align"))
    if "center" not in (align.get("horizontal") or []) or not {"top", "middle", "bottom"} <= set(align.get("vertical") or []):
        errs.append("text align vocabulary lacks center / top / middle / bottom")
    if not {"image", "video_frame"} <= {str(k) for k in as_dict(doc.get("assets")).get("kinds") or []}:
        errs.append("document.assets.kinds lacks image / video_frame")
    fonts = [str(f) for f in as_dict(contract.get("fonts")).get("font_ids") or []]
    if not fonts or DEFAULTS["font_id"] not in fonts:
        errs.append(f"fonts.font_ids {fonts!r} lacks the default {DEFAULTS['font_id']!r}")
    fmts = contract.get("output_formats") or {}
    if not isinstance(fmts, dict) or not {"png", "jpeg"} <= set(fmts) or not all(isinstance(as_dict(v).get("extensions"), list) and v["extensions"] for v in fmts.values()):
        errs.append("output_formats must declare png and jpeg with extensions")
    if as_dict(contract.get("rendering")).get("engine") != "Pillow":
        errs.append(f"rendering.engine {as_dict(contract.get('rendering')).get('engine')!r} != Pillow")
    retry, exit_codes = error_table(contract)
    for c in ("INVALID_REQUEST", "INVALID_INPUT", "PATH_NOT_ALLOWED", "UNSUPPORTED_OPERATION", "UNSUPPORTED_FORMAT", "MISSING_INPUT", "INVALID_TIME_RANGE", "TOOL_ERROR",
              "OUTPUT_ERROR", "VALIDATION_ERROR", "CANCELLED", "INTERNAL_ERROR"):
        if c not in retry:
            errs.append(f"errors.codes lacks {c}")
    if not exit_codes:
        errs.append("errors.exit_codes missing")
    return errs


def contract_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(live, pinned or pinned_contract(), DRIFT_KEYS, "tools", "tool_id", DRIFT_TOOL_KEYS)


def package_from_contract(contract: Dict[str, Any]) -> SkillPackage:
    ver = str(contract.get("version") or "")
    descs = {str(t.get("tool_id")): str(t.get("description", "")) for t in contract.get("tools") or [] if isinstance(t, dict)}
    tools = [ToolSpec(tool_id=tid, skill_id=SKILL_ID, version=ver, description=descs.get(tid, ""), required_capabilities=[SKILL_ID, "ffmpeg", "ffprobe", ENGINE_ID],
                      inputs=["input", "output"], produces_output=True, deterministic=True,
                      result_keys=["operation_type", "artifact", "observation", "provenance", "source", "commands"]) for tid in TOOLS]
    return SkillPackage(skill_id=SKILL_ID, name=str(contract.get("name") or CONTRACT_SKILL_ID), version=ver, description=str(contract.get("description", ""))[:200],
                        capabilities=[SKILL_ID], tools=tools, repository="kajisho5/thumbnail-skill",
                        role="deterministic thumbnail rendering (Pillow; video frames through ffmpeg-skill/look)")


PACKAGE = package_from_contract(pinned_contract())


class ThumbnailAdapter(ToolAdapter):
    name = SKILL_ID

    def __init__(self, skill: Optional[CliSkill] = None, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None,
                 ffmpeg_skill_dir: Optional[str] = None, timeout: float = 120.0, path_policy: Optional[PathPolicy] = None):
        located = skill or locate_thumbnail()
        if not located:
            raise ToolError("thumbnail-skill not found (set VIDEO_AGENT_THUMBNAIL_DIR or install `thumbnail`)")
        self.skill: CliSkill = located
        self.workspace = str(Path(workspace).resolve()) if workspace else None
        self.allowed_inputs = [str(Path(r).resolve()) for r in (allowed_inputs or [])]
        self.ffmpeg_skill_dir = ffmpeg_skill_dir
        self.default_timeout = float(timeout)
        self.path_policy = path_policy
        self.calls = 0
        self.contract = self._fetch_contract()   # an install without Pillow cannot even print its contract: MISSING, never half-used
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("thumbnail contract incompatible: " + "; ".join(errs))
        self.version = str(self.contract["version"])
        self.tools = set(TOOLS)
        doc = self.contract["document"]
        canvas = as_dict(doc.get("canvas"))
        self.canvas_range = {"width": _range(canvas.get("width"), (16, 7680)), "height": _range(canvas.get("height"), (16, 7680))}
        tf = as_dict(as_dict(as_dict(doc.get("elements")).get("text")).get("fields"))
        self.font_size_range = _range(tf.get("font_size"), (6, 400))
        m = re.search(r"max (\d+) chars", str(tf.get("text") or ""))
        self.text_max = int(m.group(1)) if m else 2000
        self.font_ids = sorted(str(f) for f in self.contract["fonts"]["font_ids"])
        self.formats: Dict[str, List[str]] = {str(k): [str(e).lower() for e in v["extensions"]] for k, v in self.contract["output_formats"].items()}
        self.forbidden = tuple(sorted(set(FORBIDDEN_ARG_KEYS) | {str(f) for f in doc.get("forbidden_fields") or []} | {str(f) for f in self.contract["execution"].get("forbidden_request_fields") or []}))
        self.retryable, self.exit_codes = error_table(self.contract)
        self._drift: Optional[List[str]] = None

    # ---- transport
    def _invoke(self, argv: List[str], stdin: Optional[str] = None, timeout: Optional[float] = None):
        self.calls += 1
        return invoke(self.skill, argv, stdin=stdin, timeout=timeout or self.default_timeout)

    def _engine_argv(self) -> List[str]:
        return ["--ffmpeg-skill", self.ffmpeg_skill_dir] if self.ffmpeg_skill_dir else []

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._invoke(["skill", "--json"], timeout=60.0)
        if code != 0:
            raise ContractError(f"thumbnail skill --json failed (exit {code}): {err.strip()[-300:]}")
        try:
            return one_json_document(out, "thumbnail contract")
        except ToolError as e:
            raise ContractError(str(e))

    def doctor(self) -> Dict[str, Any]:
        argv = ["doctor", "--json"] + (["--workspace", self.workspace] if self.workspace else []) + [x for r in self.allowed_inputs for x in ("--allowed-input", r)] + self._engine_argv()
        code, out, err = self._invoke(argv, timeout=180.0)
        try:
            doc = one_json_document(out, "thumbnail doctor")
        except ToolError as e:
            return {"schema": DOCTOR_SCHEMA, "status": "fail", "problems": [f"doctor produced no document: {e}"], "checks": {}, "exit_code": code}
        if doc.get("schema") != DOCTOR_SCHEMA or doc.get("status") not in ("ok", "degraded", "fail"):
            return {"schema": DOCTOR_SCHEMA, "status": "fail", "problems": [f"unexpected doctor document {doc.get('schema')!r} / {doc.get('status')!r}"], "checks": doc.get("checks") or {}, "exit_code": code}
        doc["exit_code"] = code
        return doc

    @staticmethod
    def font_status(doc: Dict[str, Any]) -> Dict[str, str]:
        """Per font_id: available | missing | …, as the Skill's doctor reports it (never inferred here)."""
        fonts = as_dict(as_dict(doc.get("checks")).get("fonts"))
        return {str(f): str(as_dict(v).get("status") or "unknown") for f, v in fonts.items()}

    @staticmethod
    def engine_status(doc: Dict[str, Any]) -> str:
        """The ffmpeg-skill check status (ok | fail | missing | unknown): without it video_frame assets fail TOOL_ERROR."""
        return str(as_dict(as_dict(doc.get("checks")).get("ffmpeg_skill")).get("status") or "unknown")

    def drift(self) -> List[str]:
        if self._drift is None:
            self._drift = contract_drift(self.contract)
        return self._drift

    # ---- ToolAdapter
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "root": self.skill.describe(), "tools": sorted(self.tools), "formats": sorted(self.formats), "fonts": self.font_ids,
                "contract": self.contract.get("schema"), "drift": self.drift()}

    def package(self) -> SkillPackage:
        return package_from_contract(self.contract)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            b = self.build_request(op.tool, op.args, paths, op_id=op.id)
        except ToolError as e:
            return [f"thumbnail: refused: {e}"]
        return [" ".join(["thumbnail"] + self._argv(b, timeout=None)) + "  <<< " + json.dumps(b["request"], ensure_ascii=False)[:400]]

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        raise ToolError("thumbnail: an execution Skill has no measurement tools")

    # ---- lowering: typed args → one {"tool", "params"} request
    @staticmethod
    def _number(name: str, v: Any, lo: float, hi: float, integer: bool = False) -> Any:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
            raise ToolError(f"thumbnail: {name} must be a finite number")
        x: Any = float(v)
        if integer:
            if x != int(x):
                raise ToolError(f"thumbnail: {name} must be an integer")
            x = int(x)
        if x < lo or x > hi:
            raise ToolError(f"thumbnail: {name} {x} is outside the contract range {lo:g}..{hi:g}")
        return x

    @staticmethod
    def _color(name: str, v: Any) -> str:
        if not isinstance(v, str) or not _COLOR_RE.match(v):
            raise ToolError(f"thumbnail: {name} must be a '#RRGGBB' colour")
        return v

    def _output_spec(self, out: str, fmt: str, args: Dict[str, Any]) -> Dict[str, Any]:
        spec: Dict[str, Any] = {"path": str(Path(out).resolve()), "format": fmt, "overwrite": True}
        if "jpeg_quality" in args:
            if fmt != "jpeg":
                raise ToolError("thumbnail: jpeg_quality is only accepted for format jpeg")
            spec["jpeg_quality"] = self._number("jpeg_quality", args["jpeg_quality"], 1, 100, integer=True)
        return spec

    def _document(self, pid: str, src: str, ts: float, args: Dict[str, Any]) -> Dict[str, Any]:
        for k in ("width", "height"):
            if k not in args:
                raise ToolError(f"thumbnail: render requires {k}")
        w = self._number("width", args["width"], *self.canvas_range["width"], integer=True)
        h = self._number("height", args["height"], *self.canvas_range["height"], integer=True)
        canvas = {"width": w, "height": h, "background": self._color("background", args.get("background", DEFAULTS["background"]))}
        elements: List[Dict[str, Any]] = [{"element_id": "frame", "type": "image", "z_index": 0,
                                           "image": {"asset_id": "frame", "position": {"x": 0, "y": 0}, "size": {"width": w, "height": h}, "fit": "cover"}}]
        if "text" in args:
            text = args["text"]
            if not isinstance(text, str) or not text or len(text) > self.text_max or any(ord(c) < 32 and c != "\n" for c in text):
                raise ToolError(f"thumbnail: text must be a non-empty string of at most {self.text_max} characters (explicit newlines only)")
            font_id = args.get("font_id", DEFAULTS["font_id"])
            if font_id not in self.font_ids:
                raise ToolError(f"thumbnail: font_id {font_id!r} is not one of the contract font_ids {self.font_ids}")
            size = self._number("font_size", args.get("font_size", DEFAULTS["font_size"]), *self.font_size_range, integer=True)
            color = self._color("color", args.get("color", DEFAULTS["color"]))
            position = args.get("position", DEFAULTS["position"])
            if position not in POSITIONS:
                raise ToolError(f"thumbnail: position {position!r} is not one of {list(POSITIONS)}")
            anchor = {"center": ({"x": w / 2, "y": h / 2}, "middle"), "top": ({"x": w / 2, "y": size}, "top"), "bottom": ({"x": w / 2, "y": h - size}, "bottom")}[position]
            elements.append({"element_id": "caption", "type": "text", "z_index": 1,
                             "text": {"text": text, "font_id": font_id, "font_size": size, "color": color, "position": anchor[0], "align": {"horizontal": "center", "vertical": anchor[1]}}})
        else:
            for k in ("font_id", "font_size", "color", "position"):
                if k in args:
                    raise ToolError(f"thumbnail: {k} needs a text")
        return {"document_id": pid, "canvas": canvas, "assets": [{"asset_id": "frame", "kind": "video_frame", "path": str(Path(src).resolve()), "timestamp": ts}], "elements": elements}

    def build_request(self, tool: str, args: Dict[str, Any], paths: Dict[str, str], op_id: str = "op", timeout: Optional[float] = None) -> Dict[str, Any]:
        if tool not in self.tools:
            raise ToolError(f"thumbnail: unsupported tool {tool}")
        hit = scan_forbidden(args, self.forbidden)
        if hit:
            raise ToolError(f"thumbnail: forbidden field {hit} in the operation arguments")
        allowed = COMMON_ARGS + (RENDER_ARGS if tool == TOOL_RENDER else ())
        unknown = sorted(k for k in args if k not in allowed)
        if unknown:
            raise ToolError(f"thumbnail: parameter(s) {unknown} are not declared for {tool} (accepted: {list(allowed)})")
        src_id, out_id = str(args.get("input") or ""), str(args.get("output") or "")
        if not src_id or not out_id:
            raise ToolError("thumbnail: input and output references are required")
        src = paths.get(src_id, src_id)
        out = paths.get(out_id, out_id)
        if not os.path.isfile(src):
            raise ToolError(f"thumbnail: input not found: {src}")
        if self.path_policy is not None:
            self.path_policy.check_input(src)
            self.path_policy.check_output(out, [src])
        elif self.allowed_inputs and not any(self._under(src, r) for r in self.allowed_inputs + ([self.workspace] if self.workspace else [])):
            raise ToolError(f"thumbnail: input outside the allowed roots: {src}")
        if self.workspace and not self._under(out, self.workspace):
            raise ToolError(f"thumbnail: output outside the workspace: {out}")
        if same_file(src, out):
            raise ToolError("thumbnail: output would overwrite its input")
        fmt = str(args.get("format") or "")
        if fmt not in self.formats:
            raise ToolError(f"thumbnail: output format {fmt!r} is not one of {sorted(self.formats)}")
        if Path(out).suffix.lower() not in self.formats[fmt]:
            raise ToolError(f"thumbnail: output extension {Path(out).suffix!r} does not match format {fmt} ({self.formats[fmt]})")
        if "timestamp" not in args:
            raise ToolError("thumbnail: timestamp is required")
        ts = float(self._number("timestamp", args["timestamp"], 0, 7 * 24 * 3600))
        pid = re.sub(r"[^A-Za-z0-9._-]", "_", str(op_id))[:64] or "op"
        if not _ID_RE.match(pid):
            pid = "op"
        output = self._output_spec(out, fmt, args)
        if tool == TOOL_EXTRACT_FRAME:
            params: Dict[str, Any] = {"source": {"path": str(Path(src).resolve()), "timestamp": ts}, "output": output}
        else:
            params = {"schema": REQUEST_SCHEMA, "document": self._document(pid, src, ts, args), "output": output}
        req = {"tool": tool, "params": params}
        hit = scan_forbidden(req, tuple(f for f in self.forbidden if f not in ("path", "paths")), "request")
        if hit:
            raise ToolError(f"thumbnail: refusing to send a request carrying {hit}")
        return {"request": req, "output": out, "input": src, "tool": tool, "format": fmt, "workspace": self.workspace or str(Path(out).parent)}

    @staticmethod
    def _under(path: str, root: str) -> bool:
        try:
            p, r = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
        except OSError:
            return False
        return p == r or p.startswith(r.rstrip(os.sep) + os.sep)

    def _argv(self, b: Dict[str, Any], timeout: Optional[float]) -> List[str]:
        argv = ["run", "-", "--json", "--workspace", b["workspace"]]
        for r in list(self.allowed_inputs) + ([self.workspace] if self.workspace else []):
            argv += ["--allowed-input", r]
        argv += self._engine_argv()
        argv += ["--timeout", str(int(timeout or self.default_timeout))]
        return argv

    # ---- execution
    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        try:
            b = self.build_request(op.tool, op.args, paths, op_id=op.id, timeout=timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e), retryable=False)
        if dry_run:   # the Skill has no plan mode: a dry run is the lowered, validated request and nothing else
            data = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run", "operation_type": op.tool[len(PREFIX):], "request": b["request"], "commands": [], "warnings": []}
            return ToolResult(op.id, op.tool, True, 0, None, data, [], "", round(time.time() - t0, 3), attempt, True)
        os.makedirs(b["workspace"], exist_ok=True)
        os.makedirs(os.path.dirname(b["output"]) or ".", exist_ok=True)
        argv = self._argv(b, timeout)
        code, out, err = self._invoke(argv, stdin=json.dumps(b["request"], ensure_ascii=False), timeout=(timeout or self.default_timeout) + 5.0)
        tail = "\n".join(err.strip().splitlines()[-12:])
        secs = round(time.time() - t0, 3)
        if code == 124:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, 124, "CANCELLED", tail or "process timed out", retryable=True, details={"reason": "timeout"})
        try:
            doc = one_json_document(out, "thumbnail")
        except ToolError as e:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 9, "INVALID_RESULT", str(e), retryable=False, details={"exit_code": code})
        # the transport envelope: {"ok": true, "tool", "result"} on dispatch, a bare error response when the request itself was refused
        res = as_dict(doc.get("result")) if isinstance(doc.get("result"), dict) else None
        if res is None:
            if doc.get("ok") is False and isinstance(doc.get("error"), dict):
                res = doc
            else:
                remove_fresh(b["output"], t0)
                return self._fail(op, attempt, dry_run, t0, code if code != 0 else 9, "INVALID_RESULT", "response carries neither a result nor an error", retryable=False, details={"exit_code": code})
        elif doc.get("ok") is not True or doc.get("tool") != b["tool"]:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 9, "INVALID_RESULT", f"transport envelope ok={doc.get('ok')!r} tool={doc.get('tool')!r} is not the dispatched {b['tool']}",
                              retryable=False, details={"exit_code": code})
        if res.get("ok") is not True:
            err_doc = as_dict(res.get("error"))
            errc = str(err_doc.get("code") or "INVALID_RESULT")
            if errc not in self.retryable:
                errc, retry = "INVALID_RESULT", False
            else:
                retry = bool(err_doc["retryable"]) if isinstance(err_doc.get("retryable"), bool) else self.retryable.get(errc, False)
            details = scrub(err_doc.get("details"), self.forbidden)
            if errc == "CANCELLED" and (details.get("reason") or "") not in ("timeout", "signal"):
                details["reason"] = details.get("reason") or "signal"
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 1, errc, str(err_doc.get("message") or tail or "")[:500], retryable=bool(retry), details=details)
        errs = self._check_response(res, b)
        if errs:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 9, "INVALID_RESULT", "; ".join(errs), retryable=False, details={"exit_code": code})
        if code != 0:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code, "INVALID_RESULT", f"exit code {code} with an ok response", retryable=False)
        data = self._success_data(res, b)
        return ToolResult(op.id, op.tool, True, 0, b["output"], data, [], tail, secs, attempt, dry_run)

    def _check_response(self, res: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
        errs: List[str] = []
        out_path = b["output"]
        if res.get("schema") != RESPONSE_SCHEMA:
            errs.append(f"response schema {res.get('schema')!r}")
        sk = as_dict(res.get("skill"))
        if sk.get("id") != CONTRACT_SKILL_ID or str(sk.get("version")) != self.version:
            errs.append(f"response skill {sk!r} is not {CONTRACT_SKILL_ID}@{self.version}")
        if res.get("status") != "ok":
            errs.append(f"status {res.get('status')!r} is not ok")
        if not res.get("output") or not same_file(str(res["output"]), out_path):
            errs.append(f"output path {res.get('output')!r} is not the requested {out_path}")
        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            return errs + ["output file missing or empty"]
        ok, actual = fingerprint_matches(res.get("sha256"), out_path)
        if not ok:
            errs.append(f"sha256 {res.get('sha256')!r} != file {actual}")
        if res.get("size") != os.path.getsize(out_path):
            errs.append("size differs from the file")
        if res.get("format") != b["format"]:
            errs.append(f"format {res.get('format')!r} is not the requested {b['format']}")
        for k in ("width", "height"):
            v = res.get(k)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                errs.append(f"{k} is not a positive integer")
        if b["tool"] not in (res.get("operations") or []):
            errs.append(f"operations {res.get('operations')!r} lack {b['tool']}")
        prov = as_dict(res.get("provenance"))
        if prov.get("skill") != CONTRACT_SKILL_ID or str(prov.get("skill_version")) != self.version or strip_sha_prefix(prov.get("output_hash")) != strip_sha_prefix(res.get("sha256")) \
                or prov.get("operation") != b["tool"] or not prov.get("identity") or not prov.get("engine"):
            errs.append("provenance incomplete (skill / version / operation / engine / identity / output_hash)")
        if b["tool"] == TOOL_RENDER:
            assets = [a for a in prov.get("assets") or [] if isinstance(a, dict)]
            if not assets or not assets[0].get("sha256"):
                errs.append("provenance.assets lacks the source frame's sha256")
        elif not as_dict(prov.get("source")).get("sha256"):
            errs.append("provenance.source lacks the source's sha256")
        return errs

    def _success_data(self, res: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        prov = as_dict(res.get("provenance"))
        op_type = b["tool"][len(PREFIX):]
        sha = strip_sha_prefix(res.get("sha256"))
        if b["tool"] == TOOL_RENDER:
            a = as_dict((prov.get("assets") or [None])[0])
            source = {"sha256": strip_sha_prefix(a.get("sha256")), "timestamp": a.get("timestamp"), "duration": a.get("source_duration")}
        else:
            s = as_dict(prov.get("source"))
            source = {"sha256": strip_sha_prefix(s.get("sha256")), "timestamp": s.get("timestamp"), "duration": s.get("duration")}
        artifact = {"path": b["output"], "sha256": sha, "size": res.get("size"), "format": res.get("format"), "width": res.get("width"), "height": res.get("height"),
                    "reused": bool(res.get("reused")), "identity": prov.get("identity")}
        return {"skill": {"id": SKILL_ID, "version": self.version}, "status": "completed", "operation_type": op_type, "artifact": artifact,
                "observation": {"kind": "image.probe", "source": f"{CONTRACT_SKILL_ID}/{op_type}@{self.version}", "provenance": "OBSERVED",
                                "data": {k: artifact[k] for k in ("width", "height", "format", "size", "sha256")}},
                "provenance": {k: prov.get(k) for k in ("skill", "skill_version", "operation", "engine", "engine_version", "document_id", "identity", "reused", "assets", "source", "fonts", "output_hash")},
                "source": source, "warnings": list(res.get("warnings") or []), "commands": []}

    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, code: int, errc: str, message: str, retryable: bool, details: Optional[Dict[str, Any]] = None) -> ToolResult:
        return failed_result(op, SKILL_ID, getattr(self, "version", ""), attempt, dry_run, t0, code, errc, message, retryable, details)


def lift_observation(result: ToolResult, asset_id: Optional[str] = None) -> Optional[Observation]:
    """The Skill's OBSERVED facts about the delivered image (dimensions, format, size, sha256) as an agent Observation
    (provenance only; never fed back into analysis)."""
    obs = (result.data or {}).get("observation")
    if not isinstance(obs, dict) or obs.get("provenance") != "OBSERVED" or not isinstance(obs.get("data"), dict):
        return None
    sk = (result.data or {}).get("skill") or {}
    art = (result.data or {}).get("artifact") or {}
    return Observation(kind="image.probe", asset_id=asset_id or result.op_id, source=str(obs.get("source")), data=dict(obs["data"]), analyzer=str(obs.get("source")),
                       provenance="OBSERVED", skill=SKILL_ID, skill_version=str(sk.get("version") or ""), tool=result.tool, fingerprint=str(art.get("sha256") or ""),
                       parameters={"output": result.output, "identity": art.get("identity")})
