"""VideoEditingAdapter: the agent's boundary to video-editing-skill (deterministic editing Skill, ADR-028).

    video-production-agent ─(typed Operation)─→ VideoEditingAdapter ─(EditRequest JSON on stdin)─→ `video-editing run - --json`
        ─(typed ffmpeg-skill calls)─→ ffmpeg-skill ─→ FFmpeg

Protocol (from the Skill's own contract, `video-editing contract --json`, schema video-editing/contract@1):
    contract → `video-editing contract --json`
    doctor   → `video-editing doctor --json [--workspace D] [--allowed-input R]…`
    run      → `video-editing run - --json --workspace <dir> --allowed-input <root>…`  with an EditRequest on stdin
    response ← exactly one JSON document on stdout ({"ok": true, "status", "execution": {"operations", "outputs"}, …} or
               {"ok": false, "error": {code, message, retryable, details}}); stderr is diagnostics only.

The contract is the source of truth: tool ids (`video-editing/<operation>`), operation types, parameters, required
capabilities, error codes and schemas come from it and are never re-declared here. The adapter turns the agent's typed
Operation (input ids, output id, the operation's parameters as the contract names them) into one EditRequest, pins the
workspace and the allowed input roots itself (the agent's PathPolicy), and never forwards commands, argv, filters,
executables, environment or credentials. It runs no ffmpeg / ffprobe and imports nothing from the Skill.

Response mapping: `execution.outputs[]` (path, sha256, size, timeline, observation) → ToolResult.output / data.artifact /
data.timeline / data.observation; the per-operation provenance record (operation_id, tool, tool versions, inputs' hashes,
commands, timing) → data.operation; `commands` are recorded as provenance only — nothing here ever re-runs them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...models import Observation, Operation, ToolResult
from ...skills.contract import SkillPackage, ToolSpec
from ..base import ToolAdapter, ToolError
from ..ffmpeg_skill.adapter import PathPolicy, run_process_group
from .locate import VideoEditingSkill, locate_video_editing
from .lowering import ARGS, OPERATION_ID, OUTPUT_ID, Lowering, op_type

SKILL_ID = "video-editing"
PREFIX = SKILL_ID + "/"
CONTRACT_SCHEMA = "video-editing/contract@1"
REQUEST_SCHEMA = "video-editing/request@1"
RESPONSE_SCHEMA = "video-editing/response@1"
DOCTOR_SCHEMA = "video-editing/doctor@1"
SUPPORTED_SKILL_VERSIONS = ("0.1.",)          # 0.1.x: the contract this adapter was verified against
ENGINE_ID = "ffmpeg-skill"
# the Skill's error vocabulary (contract errors.codes) and the agent's reading of each: (retryable by default, recovery class)
ERROR_CODES: Dict[str, str] = {
    "INVALID_REQUEST": "INVALID_ARGS", "INVALID_INPUT": "INPUT_MISSING", "PATH_NOT_ALLOWED": "INPUT_MISSING", "UNSUPPORTED_OPERATION": "INVALID_ARGS",
    "UNSUPPORTED_FORMAT": "INVALID_ARGS", "MISSING_INPUT": "INPUT_MISSING", "INVALID_TIME_RANGE": "INVALID_ARGS", "DEPENDENCY_ERROR": "INVALID_ARGS",
    "TOOL_ERROR": "UNKNOWN", "OUTPUT_ERROR": "SKILL_ERROR", "VALIDATION_ERROR": "SKILL_ERROR", "CANCELLED": "TIMEOUT", "INTERNAL_ERROR": "SKILL_ERROR",
}
RESULT_KEYS = ("operation_id", "output", "probe", "commands", "provenance")
REQUIRED_EXECUTION_FLAGS = {"shell": False, "arbitrary_executables": False, "raw_ffmpeg_arguments": False, "filter_strings": False, "network": False, "ai": False}
# argument keys the agent may hand to any tool: references (resolved by the adapter) and typed parameters the contract declares
REFERENCE_KEYS = ("input", "inputs", "image", "output")
FORBIDDEN_ARG_KEYS = ("command", "commands", "argv", "cmd", "shell", "exec", "args", "script", "binary", "executable", "executables", "env", "environment",
                      "filter", "filters", "filter_complex", "ffmpeg", "ffprobe", "api_key", "apikey", "token", "secret", "password", "credentials",
                      "workspace", "allowed_input", "allowed_input_roots", "allowed-input", "ffmpeg_skill_dir", "ffmpeg-skill-dir", "path", "paths")
OUTPUT_EXTENSIONS = (".mp4", ".mov", ".mkv")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ToolError):
    """The installed Skill does not satisfy the contract this adapter was written for."""


# ---- contract -----------------------------------------------------------------------------------------------------------
def check_contract(contract: Any) -> List[str]:
    """Compatibility checks: schema, skill id, version range, execution flags (no shell / executables / raw ffmpeg / filters /
    network / AI), canonical invocation, engine, tools (ids, ownership, operation types, capabilities, result keys), error codes."""
    errs: List[str] = []
    if not isinstance(contract, dict):
        return ["contract is not an object"]
    if contract.get("schema") != CONTRACT_SCHEMA:
        errs.append(f"contract schema {contract.get('schema')!r} != {CONTRACT_SCHEMA}")
    if contract.get("skill_id") != SKILL_ID:
        errs.append(f"skill_id {contract.get('skill_id')!r} != {SKILL_ID}")
    ver = str(contract.get("version") or "")
    if not ver.startswith(SUPPORTED_SKILL_VERSIONS):
        errs.append(f"skill version {ver!r} not in supported range {SUPPORTED_SKILL_VERSIONS}")
    if contract.get("role") != "execution":
        errs.append(f"role {contract.get('role')!r} != execution")
    schemas = contract.get("schemas") or {}
    for k, want in (("request", REQUEST_SCHEMA), ("response", RESPONSE_SCHEMA), ("contract", CONTRACT_SCHEMA), ("doctor", DOCTOR_SCHEMA)):
        if schemas.get(k) != want:
            errs.append(f"schemas.{k}={schemas.get(k)!r} unsupported (expected {want})")
    ex = contract.get("execution") or {}
    if ex.get("mode") != "local_subprocess":
        errs.append(f"execution.mode {ex.get('mode')!r} != local_subprocess")
    for k, want in REQUIRED_EXECUTION_FLAGS.items():
        if ex.get(k) is not want:
            errs.append(f"execution.{k} must be {want!r}, contract says {ex.get(k)!r}")
    inv = ex.get("canonical_invocation") or []
    if not (isinstance(inv, list) and len(inv) >= 5 and inv[1:4] == ["run", "-", "--json"] and "--workspace" in inv and "--allowed-input" in inv):
        errs.append(f"canonical_invocation {inv!r} is not `run - --json --workspace … --allowed-input …`")
    eng = contract.get("engine") or {}
    if eng.get("id") != ENGINE_ID:
        errs.append(f"engine {eng.get('id')!r} != {ENGINE_ID} (this adapter integrates an ffmpeg-skill-backed Skill)")
    ops = contract.get("operations") or {}
    if not isinstance(ops, dict) or not ops:
        errs.append("no operations declared")
    tools = contract.get("tools") or []
    if not tools:
        errs.append("no tools declared")
    seen = set()
    for t in tools:
        tid = str(t.get("tool_id", ""))
        if not tid.startswith(PREFIX) or tid.count("/") != 1 or t.get("skill_id") != SKILL_ID:
            errs.append(f"tool {tid!r} does not belong to {SKILL_ID}")
        if tid in seen:
            errs.append(f"duplicate tool {tid}")
        seen.add(tid)
        op = t.get("operation_type")
        if op not in ops:
            errs.append(f"tool {tid}: operation_type {op!r} is not a declared operation")
        elif tid != PREFIX + str(op).lower():
            errs.append(f"tool {tid}: id does not match its operation type {op}")
        if str(t.get("version") or "") != ver:
            errs.append(f"tool {tid}: version {t.get('version')!r} != contract version {ver!r}")
        if t.get("produces_output") is not True or t.get("writes_media") is not True or t.get("deterministic") is not True:
            errs.append(f"tool {tid}: must declare produces_output / writes_media / deterministic = true")
        if t.get("kind") != "transform":
            errs.append(f"tool {tid}: kind {t.get('kind')!r} != transform")
        rc = t.get("required_capabilities")
        if not isinstance(rc, list) or not rc or not all(isinstance(c, str) and c for c in rc):
            errs.append(f"tool {tid}: required_capabilities must be a non-empty list of capability names")
        if list(t.get("result_keys") or []) != list(RESULT_KEYS):
            errs.append(f"tool {tid}: result_keys {t.get('result_keys')!r} != {list(RESULT_KEYS)}")
        if t.get("inputs") not in (["input"], ["inputs"]):
            errs.append(f"tool {tid}: inputs {t.get('inputs')!r} must be ['input'] or ['inputs']")
        if not isinstance(t.get("parameters"), dict):
            errs.append(f"tool {tid}: parameters must be an object naming the typed parameters")
        if not str(t.get("executed_by", "")).startswith(ENGINE_ID + "/"):
            errs.append(f"tool {tid}: executed_by {t.get('executed_by')!r} is not an {ENGINE_ID} tool")
    tool_ops = {t.get("operation_type") for t in tools}
    for op in ops:
        if op not in tool_ops:
            errs.append(f"operation {op} has no tool")
    errors = contract.get("errors") or {}
    codes = errors.get("codes")
    if not isinstance(codes, list) or set(codes) != set(ERROR_CODES):
        errs.append(f"errors.codes {codes!r} != the vocabulary this adapter maps {sorted(ERROR_CODES)}")
    retry = errors.get("retryable_default") or {}
    if not isinstance(retry, dict) or set(retry) != set(ERROR_CODES) or not all(isinstance(v, bool) for v in retry.values()):
        errs.append("errors.retryable_default must give a boolean for every code")
    shape = errors.get("shape") or {}
    if shape.get("ok") is not False or not isinstance(shape.get("error"), dict) or "retryable" not in shape["error"]:
        errs.append("errors.shape is not {ok: false, error: {code, message, retryable, details}}")
    rs = contract.get("response_shape") or {}
    if rs.get("ok") is not True or "execution" not in rs or "status" not in rs:
        errs.append("response_shape is not {ok: true, status, execution}")
    return errs


def package_from_contract(contract: Dict[str, Any]) -> SkillPackage:
    ver = str(contract.get("version") or "")
    tools = [ToolSpec(tool_id=t["tool_id"], skill_id=SKILL_ID, version=ver, description=str(t.get("description", "")),
                      required_capabilities=list(t.get("required_capabilities") or []) + [SKILL_ID],
                      inputs=list(t.get("inputs") or []) + ["output"], produces_output=True, deterministic=bool(t.get("deterministic", True)),
                      result_keys=list(t.get("result_keys") or [])) for t in contract.get("tools") or []]
    return SkillPackage(skill_id=SKILL_ID, name=str(contract.get("name") or SKILL_ID), version=ver, description=str(contract.get("description", "")),
                        capabilities=["ffmpeg", "ffprobe", "ffmpeg-skill", SKILL_ID], tools=tools, repository="kajisho5/video-editing-skill",
                        role="deterministic video editing (typed operations, executed through ffmpeg-skill)")


PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    """The contract this adapter was verified against (snapshot of `video-editing contract --json`, 0.1.0). Used for the package
    identity when the Skill is not installed; a live installation always replaces it (and is compared against it: drift)."""
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


DRIFT_KEYS = ("skill_id", "version", "schema", "schemas", "operations", "capability_names", "unsupported", "execution", "engine", "errors", "response_shape",
              "request_shape", "formats")
DRIFT_TOOL_KEYS = ("tool_id", "skill_id", "version", "operation_type", "capability", "required_capabilities", "inputs", "input_arity", "parameters",
                   "produces_output", "writes_media", "deterministic", "result_keys", "executed_by", "kind")


def contract_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    """Differences between the installed Skill's contract and the pinned one on every field the agent depends on. A non-empty
    list means the agent's expectations are stale: the adapter must be re-verified, never silently kept."""
    pinned = pinned or pinned_contract()
    out: List[str] = []
    for k in DRIFT_KEYS:
        if live.get(k) != pinned.get(k):
            out.append(f"{k}: pinned {json.dumps(pinned.get(k), sort_keys=True)[:160]} != live {json.dumps(live.get(k), sort_keys=True)[:160]}")
    lt = {t.get("tool_id"): t for t in live.get("tools") or []}
    pt = {t.get("tool_id"): t for t in pinned.get("tools") or []}
    for tid in sorted(set(lt) | set(pt)):
        if tid not in lt:
            out.append(f"tool {tid}: pinned but not in the installed contract")
        elif tid not in pt:
            out.append(f"tool {tid}: installed but not pinned")
        else:
            for k in DRIFT_TOOL_KEYS:
                if lt[tid].get(k) != pt[tid].get(k):
                    out.append(f"tool {tid}.{k}: pinned {pt[tid].get(k)!r} != live {lt[tid].get(k)!r}")
    return out


PACKAGE = package_from_contract(pinned_contract())


# ---- helpers --------------------------------------------------------------------------------------------------------------
def _one_json_document(stdout: str) -> Dict[str, Any]:
    """Exactly one JSON object on stdout; anything else (empty, text, several documents) is a protocol violation."""
    text = (stdout or "").strip()
    if not text:
        raise ToolError("video-editing: empty stdout (expected one response document)")
    try:
        doc, end = json.JSONDecoder().raw_decode(text)
    except ValueError as e:
        raise ToolError(f"video-editing: stdout is not JSON: {e}")
    if text[end:].strip():
        raise ToolError("video-editing: more than one JSON document on stdout")
    if not isinstance(doc, dict):
        raise ToolError("video-editing: response is not an object")
    return doc


def _within(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(root), os.path.normcase(path)]) == os.path.normcase(root)
    except ValueError:
        return False


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_scalar(v: Any, what: str) -> Any:
    """Typed parameter values only: bool / int / float / short string / small object / list thereof. No NUL, no newline."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        if v != v or v in (float("inf"), float("-inf")):
            raise ToolError(f"video-editing: {what} is not a finite number")
        return v
    if isinstance(v, str):
        if len(v) > 256 or "\x00" in v or "\n" in v or "\r" in v:
            raise ToolError(f"video-editing: {what} must be a short single-line string")
        return v
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if not isinstance(k, str) or k.lower() in FORBIDDEN_ARG_KEYS:
                raise ToolError(f"video-editing: refusing parameter key {k!r} under {what}")
            out[k] = _clean_scalar(x, f"{what}.{k}")
        return out
    if isinstance(v, (list, tuple)):
        if len(v) > 500:
            raise ToolError(f"video-editing: {what} has too many entries")
        return [_clean_scalar(x, f"{what}[{i}]") for i, x in enumerate(v)]
    raise ToolError(f"video-editing: {what} has an unsupported value type {type(v).__name__}")


def _ranges(v: Any, what: str) -> List[Dict[str, Any]]:
    """[[start, end], …] or [{start, end}, …] → the contract's [{start, end}, …] (numbers only; the Skill validates order / bounds)."""
    if not isinstance(v, list) or not v:
        raise ToolError(f"video-editing: {what} must be a non-empty list of ranges")
    out = []
    for i, r in enumerate(v):
        if isinstance(r, dict) and set(r) == {"start", "end"}:
            s, e = r["start"], r["end"]
        elif isinstance(r, (list, tuple)) and len(r) == 2:
            s, e = r
        else:
            raise ToolError(f"video-editing: {what}[{i}] must be [start, end]")
        for x, name in ((s, "start"), (e, "end")):
            if isinstance(x, bool) or not isinstance(x, (int, float)) or x != x or x < 0:
                raise ToolError(f"video-editing: {what}[{i}].{name} must be a non-negative number of seconds")
        out.append({"start": round(float(s), 6), "end": round(float(e), 6)})
    return out


class VideoEditingAdapter(ToolAdapter):
    name = SKILL_ID

    def __init__(self, skill: Optional[VideoEditingSkill] = None, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None,
                 contract: Optional[Dict[str, Any]] = None, timeout: float = 3600.0, ffmpeg_skill_dir: Optional[str] = None, path_policy: Optional[PathPolicy] = None):
        self.skill = skill or locate_video_editing()
        if not self.skill:
            raise ToolError("video-editing-skill not found (set VIDEO_AGENT_VIDEO_EDITING_DIR or install the `video-editing` command)")
        self.workspace = os.path.realpath(os.path.abspath(workspace)) if workspace else None   # every output lands under it (agent PathPolicy)
        self.allowed_inputs = [os.path.realpath(os.path.abspath(p)) for p in (allowed_inputs or [])]
        self.default_timeout = float(timeout)
        self.ffmpeg_skill_dir = str(Path(ffmpeg_skill_dir).resolve()) if ffmpeg_skill_dir else None   # engine location from agent config, never from a request
        self.policy = path_policy            # the agent's own PathPolicy (Service passes it); applied before the adapter's root checks
        self.calls = 0
        self.contract = contract or self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("video-editing contract incompatible: " + "; ".join(errs))
        self.version = str(self.contract["version"])
        self.tools: Dict[str, Dict[str, Any]] = {t["tool_id"]: t for t in self.contract["tools"]}
        self.retryable: Dict[str, bool] = dict(self.contract["errors"]["retryable_default"])
        self.lowering = Lowering(self.contract, self.workspace)

    # ---- process boundary (argv list only; the request travels on stdin)
    def _invoke(self, argv: List[str], stdin: Optional[str] = None, timeout: Optional[float] = None) -> "tuple[int, str, str]":
        for a in argv:
            if not isinstance(a, str) or "\x00" in a or "\n" in a:
                raise ToolError("video-editing: argv entries must be clean single-line strings")
        cmd = list(self.skill.command) + argv
        env_extra = dict(self.skill.env)
        old = {k: os.environ.get(k) for k in env_extra}
        try:
            for k, v in env_extra.items():
                os.environ[k] = v if k != "PYTHONPATH" or not old.get(k) else v + os.pathsep + old[k]
            self.calls += 1
            return run_process_group(cmd, timeout or self.default_timeout, stdin=stdin)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _engine_argv(self) -> List[str]:
        return ["--ffmpeg-skill-dir", self.ffmpeg_skill_dir] if self.ffmpeg_skill_dir else []

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._invoke(["contract", "--json"], timeout=60.0)
        if code != 0:
            raise ContractError(f"video-editing contract --json failed ({code}): {err.strip()[-300:]}")
        return _one_json_document(out)

    def doctor(self) -> Dict[str, Any]:
        """The Skill's own doctor: ffmpeg-skill location / version / tools, ffmpeg / ffprobe as ffmpeg-skill sees them, path policy."""
        argv = ["doctor", "--json"] + self._engine_argv()
        if self.workspace and os.path.isdir(self.workspace):
            argv += ["--workspace", self.workspace]
            for root in self.allowed_inputs:
                if os.path.isdir(root):
                    argv += ["--allowed-input", root]
        code, out, err = self._invoke(argv, timeout=120.0)
        doc = _one_json_document(out) if out.strip() else {"ok": False, "checks": [], "problems": [err.strip()[-300:]], "summary": err.strip()[-300:]}
        if doc.get("schema") not in (None, DOCTOR_SCHEMA) or not isinstance(doc.get("checks"), list):
            raise ContractError("unexpected doctor document")
        return doc

    def drift(self) -> List[str]:
        return contract_drift(self.contract)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "root": self.skill.describe(), "schemas": dict(self.contract.get("schemas") or {}),
                "tools": sorted(self.tools), "operations": self.lowering.supported_types(), "contract_only": self.lowering.contract_only(),
                "engine": dict(self.contract.get("engine") or {}), "workspace": self.workspace, "allowed_inputs": list(self.allowed_inputs)}

    def package(self) -> SkillPackage:
        return package_from_contract(self.contract)

    def supports(self, tool: str) -> bool:
        """Declared by the installed contract AND lowered by this adapter: a contract that grew a type this adapter cannot lower
        never becomes executable by accident (see Lowering.contract_only)."""
        return tool in self.tools and op_type(tool) in ARGS

    # ---- path boundary (the agent's PathPolicy roots; the Skill enforces the same roots again and never widens them)
    @staticmethod
    def _check_raw(raw: Any, what: str) -> None:
        """Checks on the path as the agent spelled it, before any resolution: a '..' component is refused as traversal."""
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw or "\n" in raw:
            raise ToolError(f"video-editing: {what} must be a non-empty single-line path")
        if any(part == ".." for part in raw.replace("\\", "/").split("/")):
            raise ToolError(f"video-editing: {what} path contains '..' (traversal)")

    def _check_input(self, raw: Any, what: str) -> str:
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw or "\n" in raw:
            raise ToolError(f"video-editing: {what} must be a non-empty single-line path")
        if any(part == ".." for part in raw.replace("\\", "/").split("/")):
            raise ToolError(f"video-editing: {what} contains '..' (traversal)")
        resolved = os.path.realpath(os.path.abspath(raw))
        roots = list(self.allowed_inputs) + ([self.workspace] if self.workspace else [])
        if roots and not any(_within(r, resolved) for r in roots):
            raise ToolError(f"video-editing: {what} is outside the allowed input roots: {os.path.basename(raw)}")
        if not os.path.isfile(resolved):
            raise ToolError(f"video-editing: {what} not found: {os.path.basename(raw)}")
        return resolved

    def _check_output(self, raw: Any, inputs: List[str]) -> str:
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw or "\n" in raw:
            raise ToolError("video-editing: output must be a non-empty single-line path")
        if any(part == ".." for part in raw.replace("\\", "/").split("/")):
            raise ToolError("video-editing: output path contains '..' (traversal)")
        absolute = os.path.abspath(raw)
        parent = os.path.realpath(os.path.dirname(absolute))
        resolved = os.path.join(parent, os.path.basename(absolute))
        if self.workspace and not (_within(self.workspace, resolved) and resolved != self.workspace):
            raise ToolError(f"video-editing: output is outside the workspace: {os.path.basename(raw)}")
        if os.path.splitext(resolved)[1].lower() not in OUTPUT_EXTENSIONS:
            raise ToolError(f"video-editing: output extension must be one of {OUTPUT_EXTENSIONS}")
        for i in inputs:
            if os.path.normcase(os.path.realpath(i)) == os.path.normcase(resolved):
                raise ToolError("video-editing: output would overwrite an input")
        return absolute

    # ---- request construction (typed Operation → EditRequest through the per-type lowering; nothing else crosses the boundary)
    def build_request(self, tool: str, args: Dict[str, Any], paths: Dict[str, str], op_id: str = "op", timeout: Optional[float] = None) -> Dict[str, Any]:
        """Returns {"request": EditRequest, "workspace": <op dir>, "allowed_inputs": [...], "output": <agent output path>, "inputs": [...]}.
        The output path is reported exactly as the agent's paths map spells it (absolute, not canonicalised) so ToolResult.output
        equals what the executor / QA / artifact registry key on; containment is checked on the canonical form."""
        if tool not in self.tools:
            raise ToolError(f"video-editing: unsupported tool {tool} (contract tools: {sorted(self.tools)})")
        if not self.supports(tool):
            raise ToolError(f"video-editing: tool {tool} is declared by the contract but has no lowering in this adapter (contract drift: review the adapter)")
        out_ref = args.get("output")
        if not isinstance(out_ref, str) or not out_ref:
            raise ToolError("video-editing: an output reference is required (this Skill's tools always write media)")
        for key in ("input", "image"):
            if isinstance(args.get(key), str):
                self._check_raw(paths.get(args[key], args[key]), key)
        for ref in (args.get("inputs") if isinstance(args.get("inputs"), list) else []):
            if isinstance(ref, str):
                self._check_raw(paths.get(ref, ref), "input")
        self._check_raw(paths.get(out_ref, out_ref), "output")
        out_abs = os.path.abspath(paths.get(out_ref, out_ref))
        ws = os.path.dirname(out_abs)   # the operation's own directory (created by the executor inside the agent workspace)
        req, _, in_paths = self.lowering.build_request(tool, args, paths, op_id=op_id, timeout=timeout, workspace=ws)
        for p in in_paths:
            self._check_input(p, "input")
        self._check_output(out_abs, in_paths)
        if self.policy is not None:
            for p in in_paths:
                self.policy.check_input(p)
            self.policy.check_output(out_abs, in_paths)
        req["options"]["timeout_seconds"] = max(1, min(86400, int(timeout or self.default_timeout)))
        roots = [r for r in list(self.allowed_inputs) + ([self.workspace] if self.workspace else []) if os.path.isdir(r)]
        return {"request": req, "workspace": ws, "allowed_inputs": roots, "output": out_abs, "inputs": in_paths}

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        b = self.build_request(op.tool, op.args, paths, op_id=op.id)
        argv = ["run", "-", "--json", "--workspace", b["workspace"]] + [x for r in b["allowed_inputs"] for x in ("--allowed-input", r)] + self._engine_argv()
        return [" ".join(list(self.skill.command) + argv) + "  <<< " + json.dumps(b["request"], sort_keys=True, ensure_ascii=False)]

    # ---- execution: one subprocess per operation; the response document is mapped to a ToolResult
    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        try:
            b = self.build_request(op.tool, op.args, paths, op_id=op.id, timeout=timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e), retryable=False)
        os.makedirs(b["workspace"], exist_ok=True)
        req, out_path = b["request"], b["output"]
        argv = ["plan" if dry_run else "run", "-", "--json", "--workspace", b["workspace"]] + [x for r in b["allowed_inputs"] for x in ("--allowed-input", r)] + self._engine_argv()
        if dry_run:
            argv.append("--no-preview")
        # the Skill gets the agent's timeout as its own budget (options.timeout_seconds); the process boundary allows a short grace on top
        code, out, err = self._invoke(argv, stdin=json.dumps(req, ensure_ascii=False), timeout=(timeout or self.default_timeout) + 5.0)
        tail = "\n".join(err.strip().splitlines()[-12:])
        secs = round(time.time() - t0, 3)
        if code == 124:
            self._remove_fresh(out_path, t0)
            return self._fail(op, attempt, dry_run, t0, 124, "CANCELLED", tail or "process timed out", retryable=True, details={"reason": "timeout"})
        try:
            doc = _one_json_document(out)
        except ToolError as e:
            self._remove_fresh(out_path, t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 9, "INVALID_RESULT", str(e), retryable=False, details={"exit_code": code})
        if doc.get("ok") is not True:
            err_doc = doc.get("error") if isinstance(doc.get("error"), dict) else {}
            errc = str(err_doc.get("code") or "INVALID_RESULT")
            if errc not in ERROR_CODES:
                errc, retry = "INVALID_RESULT", False
            else:
                retry = err_doc.get("retryable") if isinstance(err_doc.get("retryable"), bool) else self.retryable.get(errc, False)
            self._remove_fresh(out_path, t0)
            r = self._fail(op, attempt, dry_run, t0, code if code != 0 else 1, errc, str(err_doc.get("message") or tail or "")[:500], retryable=bool(retry),
                           details=_scrub(err_doc.get("details")), skill_execution=doc.get("execution") if isinstance(doc.get("execution"), dict) else None)
            return r
        errs = self._check_response(doc, out_path, dry_run)
        if errs:
            self._remove_fresh(out_path, t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 9, "INVALID_RESULT", "; ".join(errs), retryable=False, details={"exit_code": code})
        if code != 0:
            self._remove_fresh(out_path, t0)
            return self._fail(op, attempt, dry_run, t0, code, "INVALID_RESULT", f"exit code {code} with an ok response", retryable=False)
        data = self._success_data(doc, out_path, dry_run)
        commands = list(data.get("commands") or [])
        return ToolResult(op.id, op.tool, True, 0, None if dry_run else out_path, data, commands, tail, secs, attempt, dry_run)

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        raise ToolError("video-editing: an editing Skill has no measurement tools")

    # ---- response validation: the Skill's document, about this request, with a delivered output whose hash we can verify
    def _check_response(self, doc: Dict[str, Any], out_path: str, dry_run: bool) -> List[str]:
        errs: List[str] = []
        if doc.get("schema") not in (RESPONSE_SCHEMA, self.contract["schemas"].get("plan")):
            errs.append(f"response schema {doc.get('schema')!r}")
        sk = doc.get("skill") if isinstance(doc.get("skill"), dict) else {}
        if sk.get("id") != SKILL_ID or str(sk.get("version")) != self.version:
            errs.append(f"response skill {sk!r} is not {SKILL_ID}@{self.version}")
        status = doc.get("status")
        if dry_run:
            if status != "planned" or not isinstance(doc.get("plan"), dict) or not isinstance(doc["plan"].get("steps"), list):
                errs.append("dry-run response lacks a plan")
            return errs
        if status not in ("completed", "reused"):
            errs.append(f"status {status!r} is not completed / reused")
        ex = doc.get("execution")
        if not isinstance(ex, dict) or not isinstance(ex.get("operations"), list) or not isinstance(ex.get("outputs"), list):
            return errs + ["response carries no execution.operations / execution.outputs"]
        outs = [o for o in ex["outputs"] if isinstance(o, dict) and o.get("id") == OUTPUT_ID]
        if len(outs) != 1:
            return errs + [f"execution.outputs does not report {OUTPUT_ID} exactly once"]
        o = outs[0]
        if o.get("delivered") is not True:
            errs.append(f"output {OUTPUT_ID} not delivered")
        if os.path.normcase(os.path.realpath(str(o.get("path") or ""))) != os.path.normcase(os.path.realpath(out_path)):
            errs.append(f"output path {o.get('path')!r} is not the requested output")
        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            errs.append("output file missing or empty although the Skill reported success")
        elif not _SHA_RE.match(str(o.get("sha256") or "")):
            errs.append("output sha256 missing")
        elif _sha256(out_path) != o["sha256"]:
            errs.append("output sha256 does not match the file on disk")
        if o.get("size") is not None and os.path.isfile(out_path) and o.get("size") != os.path.getsize(out_path):
            errs.append("output size does not match the file on disk")
        obs = o.get("observation")
        if not isinstance(obs, dict) or obs.get("provenance") != "OBSERVED" or not str(obs.get("source", "")).startswith(ENGINE_ID + "/"):
            errs.append("output observation is not an OBSERVED ffmpeg-skill measurement")
        recs = [r for r in ex["operations"] if isinstance(r, dict) and r.get("operation") == OPERATION_ID]
        if len(recs) != 1:
            errs.append(f"execution.operations does not report {OPERATION_ID} exactly once")
        else:
            rec = recs[0]
            if rec.get("status") not in ("completed", "reused") or rec.get("skill") != SKILL_ID or str(rec.get("skill_version")) != self.version:
                errs.append(f"operation record status / skill mismatch: {rec.get('status')} {rec.get('skill')}@{rec.get('skill_version')}")
            if not str(rec.get("tool", "")).startswith(ENGINE_ID + "/"):
                errs.append(f"operation executed by {rec.get('tool')!r}, not an {ENGINE_ID} tool")
            if not str(rec.get("operation_id", "")).startswith("op_"):
                errs.append("operation record lacks operation_id")
            if (rec.get("output") or {}).get("sha256") != o.get("sha256"):
                errs.append("operation output hash differs from the delivered output hash")
        return errs

    def _success_data(self, doc: Dict[str, Any], out_path: str, dry_run: bool) -> Dict[str, Any]:
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": doc.get("status"), "warnings": list(doc.get("warnings") or []),
                                "engine": dict(doc.get("engine") or {}), "dry_run": dry_run}
        if dry_run:
            plan = doc["plan"]
            data["plan"] = {"steps": [{k: s.get(k) for k in ("id", "type", "tool", "operation_id", "idempotency_key", "reusable", "timeline")} for s in plan.get("steps") or []]}
            data["commands"] = [c for s in plan.get("steps") or [] for c in ((s.get("preview") or {}).get("commands") or [])]
            return data
        ex = doc["execution"]
        o = next(x for x in ex["outputs"] if x.get("id") == OUTPUT_ID)
        rec = next(x for x in ex["operations"] if x.get("operation") == OPERATION_ID)
        data["operation_id"] = rec.get("operation_id")
        data["operation"] = {k: rec.get(k) for k in ("operation_id", "type", "capability", "status", "skill", "skill_version", "tool", "tool_versions", "idempotency_key",
                                                      "parameters", "inputs", "output", "started_at", "finished_at", "seconds", "provenance")}
        data["artifact"] = {"path": out_path, "sha256": o.get("sha256"), "size": o.get("size"), "operation_id": rec.get("operation_id"), "reused": rec.get("status") == "reused"}
        data["timeline"] = o.get("timeline")
        data["observation"] = o.get("observation")
        data["commands"] = [str(c) for r in ex["operations"] if isinstance(r, dict) for c in (r.get("commands") or [])]   # provenance only, never re-run
        data["execution"] = {k: ex.get(k) for k in ("status", "started_at", "finished_at", "work_dir")}
        data["output"] = out_path
        return data

    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, code: int, errc: str, message: str, retryable: bool,
              details: Optional[Dict[str, Any]] = None, skill_execution: Optional[Dict[str, Any]] = None) -> ToolResult:
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": getattr(self, "version", "")}, "status": "failed",
                                "error": {"code": errc, "message": message, "retryable": bool(retryable), "details": details or {}, "exit_code": code,
                                          "recovery_class": ERROR_CODES.get(errc, "SKILL_ERROR")}}
        if skill_execution is not None:
            data["execution"] = {k: skill_execution.get(k) for k in ("status", "started_at", "finished_at")}
            data["commands"] = [str(c) for r in skill_execution.get("operations") or [] if isinstance(r, dict) for c in (r.get("commands") or [])]
        tail = f"video-editing [{errc}] {message}"
        return ToolResult(op.id, op.tool, False, code, None, data, list(data.get("commands") or []), tail, round(time.time() - t0, 3), attempt, dry_run)

    @staticmethod
    def _remove_fresh(path: str, t0: float) -> None:
        """A file the failed run may have left at the output path (never a retry input)."""
        try:
            if os.path.isfile(path) and os.path.getmtime(path) >= t0 - 1:
                os.remove(path)
        except OSError:
            pass


def _scrub(details: Any) -> Dict[str, Any]:
    if not isinstance(details, dict):
        return {}
    return {k: v for k, v in details.items() if isinstance(k, str) and k.lower() not in FORBIDDEN_ARG_KEYS and isinstance(v, (str, int, float, bool, list, dict, type(None)))}


def lift_observation(result: ToolResult, asset_id: Optional[str] = None) -> Optional[Observation]:
    """The Skill's OBSERVED probe of a delivered output as an agent Observation (ported from PR #19): provenance kept (skill,
    version, tool, engine source, output hash as fingerprint). None when the result carries no observation (failure, dry run).
    Recorded as provenance only; never fed back into the IR's analysis."""
    obs = (result.data or {}).get("observation")
    if not isinstance(obs, dict) or obs.get("provenance") != "OBSERVED" or not isinstance(obs.get("data"), dict):
        return None
    sk = (result.data or {}).get("skill") or {}
    art = (result.data or {}).get("artifact") or {}
    return Observation(kind=str(obs.get("kind") or "media.probe"), asset_id=asset_id or result.op_id, source=str(obs.get("source")), data=dict(obs["data"]),
                       analyzer=str(obs.get("source")), provenance="OBSERVED", skill=SKILL_ID, skill_version=str(sk.get("version") or ""), tool=result.tool,
                       fingerprint=str(art.get("sha256") or ""), parameters={"output": result.output, "operation_id": art.get("operation_id")})

