"""AudioProductionAdapter: the agent's boundary to audio-production-skill (deterministic audio production Skill, ADR-030).

    video-production-agent ─(typed Operation)─→ AudioProductionAdapter ─(audio request JSON on stdin)─→ `audio-production run - --json`
        ─(typed ffmpeg-skill calls)─→ ffmpeg-skill ─→ FFmpeg

Protocol (from the Skill's own contract, `audio-production skill --json`, schema audio-production/contract@1):
    contract → `audio-production skill --json`
    doctor   → `audio-production doctor --json [--workspace D] [--allowed-input R]… [--ffmpeg-skill X]`
    run      → `audio-production run - --json --workspace <dir> --allowed-input <root>… [--ffmpeg-skill X] [--timeout S]` with the request on stdin
    response ← exactly one JSON document on stdout ({"ok": true, "status": "ok", "plan", "results", "outputs", "tool_runs"} or
               {"ok": false, "status": "error" | "cancelled", "error": {code, message, retryable, details}}); stderr is diagnostics only.

The contract is the source of truth: the single tool id (`audio-production/run`), the operation types with their parameter
schemas and required capabilities, the unsupported operations, output formats, forbidden request fields, error codes and
schema versions come from it and are never re-declared here. The adapter turns the agent's typed Operation (operation type,
input ids, output id, typed parameters) into one request, pins the workspace and the allowed input roots itself (the agent's
PathPolicy), and never forwards commands, argv, filters, executables, environment or credentials. It runs no ffmpeg /
ffprobe and imports nothing from the Skill.

Response mapping: `outputs[out]` (path, artifact {sha256, size, duration, channels, sample_rate, codec}, segments, provenance)
→ ToolResult.output / data.artifact / data.timeline / data.observation (an OBSERVED ffmpeg-skill probe of the delivered
file) / data.provenance; `results[op:edit]` (operation_id, tool, parameters, input_hashes, measurements — the NORMALIZE
re-measurement among them —, tool_commands_observed) → data.operation; commands are recorded as provenance only.
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
from .locate import AudioProductionSkill, locate_audio_production
from .lowering import OPERATION_ID, OUTPUT_ID, TOOL_ID, Lowering

SKILL_ID = "audio-production"
PREFIX = SKILL_ID + "/"
CONTRACT_SCHEMA = "audio-production/contract@1"
REQUEST_SCHEMA = "audio-production/request@1"
RESPONSE_SCHEMA = "audio-production/response@1"
DOCTOR_SCHEMA = "audio-production/doctor@1"
SUPPORTED_SKILL_VERSIONS = ("0.1.",)          # 0.1.x: the contract this adapter was verified against
ENGINE_ID = "ffmpeg-skill"
CANONICAL_INVOCATION = ["audio-production", "run", "-", "--json"]
# the Skill's error vocabulary (contract errors.codes) and the agent's reading of each (recovery class)
ERROR_CODES: Dict[str, str] = {
    "INVALID_REQUEST": "INVALID_ARGS", "INVALID_INPUT": "INPUT_MISSING", "PATH_NOT_ALLOWED": "INPUT_MISSING", "UNSUPPORTED_OPERATION": "INVALID_ARGS",
    "UNSUPPORTED_FORMAT": "INVALID_ARGS", "MISSING_INPUT": "INPUT_MISSING", "INVALID_TIME_RANGE": "INVALID_ARGS", "INVALID_CHANNEL_LAYOUT": "INVALID_ARGS",
    "INVALID_SAMPLE_RATE": "INVALID_ARGS", "DEPENDENCY_ERROR": "INVALID_ARGS", "TOOL_ERROR": "UNKNOWN", "OUTPUT_ERROR": "SKILL_ERROR",
    "VALIDATION_ERROR": "SKILL_ERROR", "CANCELLED": "TIMEOUT", "INTERNAL_ERROR": "SKILL_ERROR",
}
REQUIRED_EXECUTION_FLAGS = {"shell": False, "arbitrary_executables": False, "arbitrary_filters": False, "network": False, "input_mutation": False, "ai": False}
SCHEMA_VERSIONS = {"contract": "1", "request": "1", "response": "1", "doctor": "1"}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ToolError):
    """The installed Skill does not satisfy the contract this adapter was written for."""


# ---- contract -----------------------------------------------------------------------------------------------------------
def check_contract(contract: Any) -> List[str]:
    """Compatibility checks: schema, skill id, version range, kind / role, execution flags (no shell / executables / filters /
    network / input mutation / AI), canonical invocation, the single tool and its ownership / operations, operation schemas,
    output formats, forbidden request fields, error codes, schema versions."""
    errs: List[str] = []
    if not isinstance(contract, dict):
        return ["contract is not an object"]
    if contract.get("schema") != CONTRACT_SCHEMA:
        errs.append(f"contract schema {contract.get('schema')!r} != {CONTRACT_SCHEMA}")
    if contract.get("skill_id") != SKILL_ID or contract.get("id") != SKILL_ID:
        errs.append(f"skill_id {contract.get('skill_id')!r} / id {contract.get('id')!r} != {SKILL_ID}")
    ver = str(contract.get("version") or "")
    if not ver.startswith(SUPPORTED_SKILL_VERSIONS):
        errs.append(f"skill version {ver!r} not in supported range {SUPPORTED_SKILL_VERSIONS}")
    if contract.get("kind") != "execution":
        errs.append(f"kind {contract.get('kind')!r} != execution")
    if "not measurement" not in str(contract.get("role", "")) or "not decision" not in str(contract.get("role", "")):
        errs.append(f"role {contract.get('role')!r} must declare an execution-only role (not measurement, not decision)")
    ex = contract.get("execution") or {}
    if ex.get("mode") != "local":
        errs.append(f"execution.mode {ex.get('mode')!r} != local")
    for k, want in REQUIRED_EXECUTION_FLAGS.items():
        if ex.get(k) is not want:
            errs.append(f"execution.{k} must be {want!r}, contract says {ex.get(k)!r}")
    if list(ex.get("canonical_invocation") or []) != CANONICAL_INVOCATION:
        errs.append(f"canonical_invocation {ex.get('canonical_invocation')!r} != {CANONICAL_INVOCATION}")
    if ex.get("stdin") != REQUEST_SCHEMA:
        errs.append(f"execution.stdin {ex.get('stdin')!r} != {REQUEST_SCHEMA}")
    if (contract.get("schema_versions") or {}) != SCHEMA_VERSIONS:
        errs.append(f"schema_versions {contract.get('schema_versions')!r} != {SCHEMA_VERSIONS}")
    fs = contract.get("ffmpeg_skill") or {}
    if str(fs.get("contract_version")) != "1.0" or not isinstance(fs.get("version_window"), dict):
        errs.append("ffmpeg_skill.contract_version / version_window missing (the Skill must declare the engine window it verified)")
    ops = {str(o.get("type")): o for o in contract.get("operations") or [] if isinstance(o, dict)}
    if not ops:
        errs.append("no operations declared")
    for t, o in ops.items():
        t = str(t)
        if not re.match(r"^[A-Z_]{2,32}$", t):
            errs.append(f"operation type {t!r} is not an upper-case identifier")
        if not isinstance(o.get("parameters"), dict) or not isinstance(o.get("inputs"), dict) or not isinstance(o.get("required_capabilities"), list):
            errs.append(f"operation {t}: parameters / inputs / required_capabilities missing")
        if not str(o.get("tool", "")).startswith(ENGINE_ID + "/"):
            errs.append(f"operation {t}: executed by {o.get('tool')!r}, not an {ENGINE_ID} tool")
        for k, ps in (o.get("parameters") or {}).items():
            if not isinstance(ps, dict) or ps.get("type") not in ("number", "integer", "string", "boolean", "array", "object"):
                errs.append(f"operation {t}: parameter {k!r} has no typed schema")
    unsupported = {str(u.get("type")) for u in contract.get("unsupported_operations") or [] if isinstance(u, dict)}
    if unsupported & {str(t) for t in ops}:
        errs.append(f"operations declared both supported and unsupported: {sorted(unsupported & {str(t) for t in ops})}")
    tools = contract.get("tools") or []
    if len(tools) != 1:
        errs.append(f"expected exactly one tool ({TOOL_ID}), contract declares {len(tools)}")
    for t in tools:
        if t.get("tool_id") != TOOL_ID or t.get("skill_id") != SKILL_ID:
            errs.append(f"tool {t.get('tool_id')!r} does not belong to {SKILL_ID} as {TOOL_ID}")
        if str(t.get("version") or "") != ver:
            errs.append(f"tool version {t.get('version')!r} != contract version {ver!r}")
        if t.get("role") != "execution" or t.get("produces_output") is not True or t.get("writes_media") is not True or t.get("deterministic") is not True or t.get("mutates_input") is not False:
            errs.append("tool must declare role execution, produces_output / writes_media / deterministic = true, mutates_input = false")
        if t.get("input_type") != REQUEST_SCHEMA or t.get("provenance") != "OBSERVED":
            errs.append(f"tool input_type {t.get('input_type')!r} / provenance {t.get('provenance')!r} unsupported")
        if sorted(str(x) for x in t.get("operations") or []) != sorted(str(x) for x in ops):
            errs.append(f"tool operations {t.get('operations')!r} != declared operations {sorted(ops)}")
        if not all(str(d).startswith(ENGINE_ID + "/") for d in t.get("delegates_to") or []) or not t.get("delegates_to"):
            errs.append("tool delegates_to must name ffmpeg-skill tools only")
    fmts = contract.get("output_formats") or {}
    if not isinstance(fmts, dict) or "wav" not in fmts or not all(isinstance(f, dict) and f.get("extension") and f.get("required_capability") for f in fmts.values()):
        errs.append("output_formats must declare wav (the intermediate format) with extensions and required capabilities")
    req = contract.get("request") or {}
    if req.get("schema") != REQUEST_SCHEMA:
        errs.append(f"request.schema {req.get('schema')!r} != {REQUEST_SCHEMA}")
    forbidden = set(req.get("forbidden_fields") or [])
    for k in ("command", "argv", "filter", "shell", "exec", "env"):
        if k not in forbidden:
            errs.append(f"request.forbidden_fields does not reject {k!r}")
    resp = contract.get("response") or {}
    if resp.get("schema") != RESPONSE_SCHEMA or (resp.get("failure") or {}).get("ok") is not False or "error" not in (resp.get("failure") or {}):
        errs.append(f"response shape unsupported: {resp!r}"[:200])
    errors = contract.get("errors") or {}
    codes = errors.get("codes")
    if not isinstance(codes, list) or set(codes) != set(ERROR_CODES):
        errs.append(f"errors.codes {codes!r} != the vocabulary this adapter maps {sorted(ERROR_CODES)}")
    retry = errors.get("retryable") or {}
    if not isinstance(retry, dict) or set(retry) != set(ERROR_CODES) or not all(isinstance(v, bool) for v in retry.values()):
        errs.append("errors.retryable must give a boolean for every code")
    exits = errors.get("exit_codes") or {}
    if not isinstance(exits, dict) or set(exits) != set(ERROR_CODES) or errors.get("success_exit_code") != 0:
        errs.append("errors.exit_codes must cover every code; success_exit_code must be 0")
    return errs


def package_from_contract(contract: Dict[str, Any]) -> SkillPackage:
    ver = str(contract.get("version") or "")
    tools = [ToolSpec(tool_id=str(t.get("tool_id") or TOOL_ID), skill_id=SKILL_ID, version=ver, description=str(t.get("description", "")),
                      required_capabilities=[SKILL_ID], inputs=["input", "inputs", "output"], produces_output=True, deterministic=bool(t.get("deterministic", True)),
                      result_keys=["operation_id", "operation", "artifact", "timeline", "observation", "provenance", "commands"]) for t in contract.get("tools") or []]
    return SkillPackage(skill_id=SKILL_ID, name=str(contract.get("name") or SKILL_ID), version=ver, description=str(contract.get("description", ""))[:200],
                        capabilities=["ffmpeg", "ffprobe", "ffmpeg-skill", SKILL_ID], tools=tools, repository="kajisho5/audio-production-skill",
                        role="deterministic audio production (typed operation graph, executed through ffmpeg-skill)")


PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    """The contract this adapter was verified against (snapshot of `audio-production skill --json`, 0.1.0)."""
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


DRIFT_KEYS = ("schema", "skill_id", "version", "kind", "tools", "unsupported_operations", "output_formats", "intermediate_format", "channel_layouts", "sample_rates",
              "execution", "ffmpeg_skill", "request", "response", "provenance", "schema_versions", "errors")
DRIFT_OPERATION_KEYS = ("type", "inputs", "parameters", "tool", "required_capabilities", "keeps_timeline", "deterministic")


def contract_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    """Differences between the installed Skill's contract and the pinned one on every field the agent depends on. A non-empty
    list means the agent's expectations are stale: the adapter must be re-verified, never silently kept."""
    pinned = pinned or pinned_contract()
    out: List[str] = []
    for k in DRIFT_KEYS:
        if live.get(k) != pinned.get(k):
            out.append(f"{k}: pinned {json.dumps(pinned.get(k), sort_keys=True)[:160]} != live {json.dumps(live.get(k), sort_keys=True)[:160]}")
    lo = {str(o.get("type")): o for o in live.get("operations") or [] if isinstance(o, dict)}
    po = {str(o.get("type")): o for o in pinned.get("operations") or [] if isinstance(o, dict)}
    for t in sorted(set(lo) | set(po)):
        if t not in lo:
            out.append(f"operation {t}: pinned but not in the installed contract")
        elif t not in po:
            out.append(f"operation {t}: installed but not pinned")
        else:
            for k in DRIFT_OPERATION_KEYS:
                if lo[t].get(k) != po[t].get(k):
                    out.append(f"operation {t}.{k}: pinned {json.dumps(po[t].get(k), sort_keys=True)[:120]} != live {json.dumps(lo[t].get(k), sort_keys=True)[:120]}")
    return out


PACKAGE = package_from_contract(pinned_contract())


# ---- helpers --------------------------------------------------------------------------------------------------------------
def _one_json_document(stdout: str) -> Dict[str, Any]:
    text = (stdout or "").strip()
    if not text:
        raise ToolError("audio-production: empty stdout (expected one response document)")
    try:
        doc, end = json.JSONDecoder().raw_decode(text)
    except ValueError as e:
        raise ToolError(f"audio-production: stdout is not JSON: {e}")
    if text[end:].strip():
        raise ToolError("audio-production: more than one JSON document on stdout")
    if not isinstance(doc, dict):
        raise ToolError("audio-production: response is not an object")
    return doc


def _as_dict(v: Any) -> Dict[str, Any]:
    return {str(k): x for k, x in v.items()} if isinstance(v, dict) else {}


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


def _scrub(details: Any, forbidden: tuple) -> Dict[str, Any]:
    if not isinstance(details, dict):
        return {}
    return {k: v for k, v in details.items() if isinstance(k, str) and k.lower() not in forbidden and isinstance(v, (str, int, float, bool, list, dict, type(None)))}


class AudioProductionAdapter(ToolAdapter):
    name = SKILL_ID

    def __init__(self, skill: Optional[AudioProductionSkill] = None, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None,
                 contract: Optional[Dict[str, Any]] = None, timeout: float = 3600.0, ffmpeg_skill_dir: Optional[str] = None, path_policy: Optional[PathPolicy] = None):
        located = skill or locate_audio_production()
        if located is None:
            raise ToolError("audio-production-skill not found (set VIDEO_AGENT_AUDIO_PRODUCTION_DIR or install the `audio-production` command)")
        self.skill: AudioProductionSkill = located
        self.workspace = os.path.realpath(os.path.abspath(workspace)) if workspace else None
        self.allowed_inputs = [os.path.realpath(os.path.abspath(p)) for p in (allowed_inputs or [])]
        self.default_timeout = float(timeout)
        self.ffmpeg_skill_dir: Optional[str] = str(Path(ffmpeg_skill_dir).resolve()) if ffmpeg_skill_dir else None   # engine location from agent config, never from a request
        self.policy = path_policy
        self.calls = 0
        self.contract = contract or self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("audio-production-skill contract incompatible: " + "; ".join(errs))
        self.version: str = str(self.contract["version"])
        self.tools = {t["tool_id"] for t in self.contract["tools"]}
        self.lowering = Lowering(self.contract, self.workspace)
        self.retryable: Dict[str, bool] = {str(k): bool(v) for k, v in ((self.contract.get("errors") or {}).get("retryable") or {}).items()}
        self._drift: Optional[List[str]] = None

    # ---- process boundary
    def _invoke(self, argv: List[str], stdin: Optional[str] = None, timeout: Optional[float] = None) -> "tuple[int, str, str]":
        self.calls += 1
        env_backup: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in self.skill.env}
        try:
            for k, v in self.skill.env.items():
                os.environ[k] = v
            return run_process_group(list(self.skill.command) + list(argv), timeout=timeout or self.default_timeout, stdin=stdin)
        finally:
            for k, old in env_backup.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old

    def _engine_argv(self) -> List[str]:
        return ["--ffmpeg-skill", self.ffmpeg_skill_dir] if self.ffmpeg_skill_dir else []

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._invoke(["skill", "--json"], timeout=60.0)
        if code != 0:
            raise ContractError(f"audio-production skill --json failed (exit {code}): {err.strip()[-300:]}")
        try:
            return _one_json_document(out)
        except ToolError as e:
            raise ContractError(f"audio-production contract: {e}")

    def doctor(self) -> Dict[str, Any]:
        """The Skill's own diagnosis (`doctor --json`): ffmpeg-skill window, ffmpeg / ffprobe, per-operation status, path policy."""
        argv = ["doctor", "--json"] + (["--workspace", self.workspace] if self.workspace else []) + [x for r in self.allowed_inputs for x in ("--allowed-input", r)] + self._engine_argv()
        code, out, err = self._invoke(argv, timeout=180.0)
        try:
            doc = _one_json_document(out)
        except ToolError as e:
            return {"schema": DOCTOR_SCHEMA, "status": "fail", "problems": [f"doctor produced no document: {e}"], "checks": {}, "exit_code": code}
        if doc.get("schema") != DOCTOR_SCHEMA or doc.get("status") not in ("ok", "degraded", "fail"):
            return {"schema": DOCTOR_SCHEMA, "status": "fail", "problems": [f"unexpected doctor document {doc.get('schema')!r} / {doc.get('status')!r}"], "checks": doc.get("checks") or {}, "exit_code": code}
        doc["exit_code"] = code
        return doc

    def drift(self) -> List[str]:
        if self._drift is None:
            self._drift = contract_drift(self.contract)
        return self._drift

    def operation_status(self, doctor: Dict[str, Any]) -> Dict[str, str]:
        """type → supported | unsupported | unknown as the Skill's doctor reports it (never guessed here)."""
        ops = (doctor.get("checks") or {}).get("operations") or {}
        return {t: str((ops.get(t) or {}).get("status") or "unknown") for t in self.lowering.supported_types()}

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "skill": self.skill.describe(), "contract": self.contract.get("schema"), "tools": sorted(self.tools),
                "operations": self.lowering.supported_types(), "unsupported": list(self.lowering.unsupported), "formats": sorted(self.lowering.formats),
                "engine": {"id": ENGINE_ID, "window": (self.contract.get("ffmpeg_skill") or {}).get("version_window")}, "workspace": self.workspace, "allowed_inputs": list(self.allowed_inputs)}

    def package(self) -> SkillPackage:
        return package_from_contract(self.contract)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    # ---- path boundary (the agent's PathPolicy roots; the Skill enforces the same roots again and never widens them)
    @staticmethod
    def _check_raw(raw: Any, what: str) -> None:
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw or "\n" in raw:
            raise ToolError(f"audio-production: {what} must be a non-empty single-line path")
        if any(part == ".." for part in raw.replace("\\", "/").split("/")):
            raise ToolError(f"audio-production: {what} path contains '..' (traversal)")

    def _check_input(self, raw: str, what: str) -> str:
        self._check_raw(raw, what)
        resolved = os.path.realpath(os.path.abspath(raw))
        roots = list(self.allowed_inputs) + ([self.workspace] if self.workspace else [])
        if roots and not any(_within(r, resolved) for r in roots):
            raise ToolError(f"audio-production: {what} is outside the allowed input roots: {os.path.basename(raw)}")
        if not os.path.isfile(resolved):
            raise ToolError(f"audio-production: {what} not found: {os.path.basename(raw)}")
        return resolved

    def _check_output(self, raw: str, inputs: List[str]) -> str:
        self._check_raw(raw, "output")
        absolute = os.path.abspath(raw)
        parent = os.path.realpath(os.path.dirname(absolute))
        resolved = os.path.join(parent, os.path.basename(absolute))
        if self.workspace and not (_within(self.workspace, resolved) and resolved != self.workspace):
            raise ToolError(f"audio-production: output is outside the workspace: {os.path.basename(raw)}")
        exts = {str(f.get("extension", "")).lower() for f in self.lowering.formats.values()}
        if os.path.splitext(resolved)[1].lower() not in exts:
            raise ToolError(f"audio-production: output extension must be one of {sorted(exts)}")
        for i in inputs:
            if os.path.normcase(os.path.realpath(i)) == os.path.normcase(resolved):
                raise ToolError("audio-production: output would overwrite an input")
        return absolute

    # ---- request construction
    def build_request(self, tool: str, args: Dict[str, Any], paths: Dict[str, str], op_id: str = "op", timeout: Optional[float] = None) -> Dict[str, Any]:
        """Returns {"request", "workspace" (the operation's own directory), "allowed_inputs", "output" (the agent's path spelling), "inputs", "type"}."""
        if tool not in self.tools:
            raise ToolError(f"audio-production: unsupported tool {tool} (contract tool: {sorted(self.tools)})")
        out_ref = args.get("output")
        if not isinstance(out_ref, str) or not out_ref:
            raise ToolError("audio-production: an output reference is required (this Skill always writes media)")
        for ref in ([args["input"]] if isinstance(args.get("input"), str) else []) + (list(args["inputs"]) if isinstance(args.get("inputs"), list) else []):
            if isinstance(ref, str):
                self._check_raw(paths.get(ref, ref), "input")
        self._check_raw(paths.get(out_ref, out_ref), "output")
        out_abs = os.path.abspath(paths.get(out_ref, out_ref))
        ws = os.path.dirname(out_abs)   # the operation's own directory (created by the executor inside the agent workspace)
        req, _, in_paths, t = self.lowering.build_request(args, paths, op_id=op_id, timeout=timeout, workspace=ws)
        for p in in_paths:
            self._check_input(p, "input")
        self._check_output(out_abs, in_paths)
        if self.policy is not None:
            for p in in_paths:
                self.policy.check_input(p)
            self.policy.check_output(out_abs, in_paths)
        req["options"]["timeout"] = max(1, min(86400, int(timeout or self.default_timeout)))
        roots = [r for r in list(self.allowed_inputs) + ([self.workspace] if self.workspace else []) if os.path.isdir(r)]
        return {"request": req, "workspace": ws, "allowed_inputs": roots, "output": out_abs, "inputs": in_paths, "type": t}

    def _argv(self, b: Dict[str, Any], dry_run: bool, timeout: Optional[float]) -> List[str]:
        return (["plan" if dry_run else "run", "-", "--json", "--workspace", b["workspace"]] + [x for r in b["allowed_inputs"] for x in ("--allowed-input", r)]
                + self._engine_argv() + ["--timeout", str(max(1, min(86400, int(timeout or self.default_timeout))))])

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        b = self.build_request(op.tool, op.args, paths, op_id=op.id)
        return [" ".join(list(self.skill.command) + self._argv(b, True, None)) + "  <<< " + json.dumps(b["request"], sort_keys=True, ensure_ascii=False)]

    # ---- execution: one subprocess per operation; the response document is mapped to a ToolResult
    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        try:
            b = self.build_request(op.tool, op.args, paths, op_id=op.id, timeout=timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e), retryable=False)
        os.makedirs(b["workspace"], exist_ok=True)
        req, out_path = b["request"], b["output"]
        argv = self._argv(b, dry_run, timeout)
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
            err_doc: Dict[str, Any] = _as_dict(doc.get("error"))
            errc = str(err_doc.get("code") or "INVALID_RESULT")
            retry: bool
            if errc not in ERROR_CODES:
                errc, retry = "INVALID_RESULT", False
            else:
                retry = bool(err_doc["retryable"]) if isinstance(err_doc.get("retryable"), bool) else self.retryable.get(errc, False)
            details = _scrub(err_doc.get("details"), self.lowering.forbidden)
            if errc == "CANCELLED" and (details.get("reason") or "") not in ("timeout", "signal"):
                details["reason"] = details.get("reason") or "signal"
            self._remove_fresh(out_path, t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 1, errc, str(err_doc.get("message") or tail or "")[:500], retryable=bool(retry), details=details,
                              skill_results=doc.get("results") if isinstance(doc.get("results"), list) else None)
        errs = self._check_response(doc, out_path, dry_run)
        if errs:
            self._remove_fresh(out_path, t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 9, "INVALID_RESULT", "; ".join(errs), retryable=False, details={"exit_code": code})
        if code != 0:
            self._remove_fresh(out_path, t0)
            return self._fail(op, attempt, dry_run, t0, code, "INVALID_RESULT", f"exit code {code} with an ok response", retryable=False)
        data = self._success_data(doc, out_path, dry_run, b["type"])
        return ToolResult(op.id, op.tool, True, 0, None if dry_run else out_path, data, list(data.get("commands") or []), tail, secs, attempt, dry_run)

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        raise ToolError("audio-production: a production Skill has no measurement tools (media-analysis-skill measures)")

    # ---- response validation
    def _check_response(self, doc: Dict[str, Any], out_path: str, dry_run: bool) -> List[str]:
        errs: List[str] = []
        if doc.get("schema") != RESPONSE_SCHEMA:
            errs.append(f"response schema {doc.get('schema')!r}")
        sk: Dict[str, Any] = _as_dict(doc.get("skill"))
        if sk.get("id") != SKILL_ID or str(sk.get("version")) != self.version:
            errs.append(f"response skill {sk!r} is not {SKILL_ID}@{self.version}")
        if doc.get("status") != "ok":
            errs.append(f"status {doc.get('status')!r} is not ok")
        if bool(doc.get("dry_run")) != bool(dry_run):
            errs.append(f"dry_run flag {doc.get('dry_run')!r} does not match the request")
        plan = doc.get("plan")
        if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list) or not isinstance(plan.get("tool_versions"), dict):
            errs.append("response carries no plan (steps / tool_versions)")
        results = doc.get("results")
        if not isinstance(results, list):
            return errs + ["response carries no results"]
        recs = [r for r in results if isinstance(r, dict) and r.get("node_id") == f"op:{OPERATION_ID}"]
        if len(recs) != 1:
            errs.append(f"results do not report op:{OPERATION_ID} exactly once")
        else:
            rec = recs[0]
            if not dry_run and rec.get("status") not in ("completed", "reused"):
                errs.append(f"operation status {rec.get('status')!r}")
            if not str(rec.get("tool", "")).startswith(ENGINE_ID + "/"):
                errs.append(f"operation executed by {rec.get('tool')!r}, not an {ENGINE_ID} tool")
            if not _SHA_RE.match(str(rec.get("operation_id") or "")):
                errs.append("operation record lacks a sha256 operation_id")
        if dry_run:
            return errs
        outs = [o for o in doc.get("outputs") or [] if isinstance(o, dict) and o.get("output_id") == OUTPUT_ID]
        if len(outs) != 1:
            return errs + [f"outputs do not report {OUTPUT_ID} exactly once"]
        o = outs[0]
        if o.get("status") != "completed":
            errs.append(f"output {OUTPUT_ID} status {o.get('status')!r}")
        if os.path.normcase(os.path.realpath(str(o.get("path") or ""))) != os.path.normcase(os.path.realpath(out_path)):
            errs.append(f"output path {o.get('path')!r} is not the requested output")
        art: Dict[str, Any] = _as_dict(o.get("artifact"))
        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            errs.append("output file missing or empty although the Skill reported success")
        elif not _SHA_RE.match(str(art.get("sha256") or "")):
            errs.append("output sha256 missing")
        elif _sha256(out_path) != art["sha256"]:
            errs.append("output sha256 does not match the file on disk")
        if art.get("size") is not None and os.path.isfile(out_path) and art.get("size") != os.path.getsize(out_path):
            errs.append("output size does not match the file on disk")
        for k in ("duration", "channels", "sample_rate", "codec"):
            if art.get(k) in (None, ""):
                errs.append(f"output artifact lacks the probed {k}")
        prov: Dict[str, Any] = _as_dict(o.get("provenance"))
        if prov.get("skill") != SKILL_ID or str(prov.get("skill_version")) != self.version or prov.get("output_hash") != art.get("sha256"):
            errs.append("output provenance does not name this Skill / version / output hash")
        if not isinstance(prov.get("operations"), list) or not prov.get("sources"):
            errs.append("output provenance lacks the operation chain / sources")
        if recs and recs[0].get("status") == "completed" and isinstance(recs[0].get("artifact"), dict) and not _SHA_RE.match(str(recs[0]["artifact"].get("sha256") or "")):
            errs.append("operation artifact lacks its sha256")
        return errs

    def _success_data(self, doc: Dict[str, Any], out_path: str, dry_run: bool, op_type: str) -> Dict[str, Any]:
        plan = doc.get("plan") or {}
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": doc.get("status"), "warnings": list(doc.get("warnings") or []),
                                "engine": dict(plan.get("tool_versions") or {}), "dry_run": dry_run, "operation_type": op_type}
        rec = next(r for r in doc["results"] if r.get("node_id") == f"op:{OPERATION_ID}")
        data["operation_id"] = rec.get("operation_id")
        data["operation"] = {k: rec.get(k) for k in ("node_id", "operation_id", "type", "tool", "required_capabilities", "status", "parameters", "inputs", "input_hashes",
                                                      "expected_duration", "seconds", "measurements")}
        data["operation"]["skill"], data["operation"]["skill_version"] = SKILL_ID, self.version
        data["operation"]["tool_versions"] = dict(plan.get("tool_versions") or {})
        if dry_run:
            data["plan"] = {"plan_id": plan.get("plan_id"), "steps": [{k: s.get(k) for k in ("node_id", "operation_id", "type", "tool", "expected_duration")} for s in plan.get("steps") or []]}
            data["commands"] = []
            return data
        o = next(x for x in doc.get("outputs") or [] if x.get("output_id") == OUTPUT_ID)
        art = dict(o.get("artifact") or {})
        data["artifact"] = {"path": out_path, "sha256": art.get("sha256"), "size": art.get("size"), "format": o.get("format"), "operation_id": rec.get("operation_id"),
                            "reused": rec.get("status") == "reused"}
        data["timeline"] = {"duration": art.get("duration"), "segments": list(o.get("segments") or [])}
        engine_ver = str((plan.get("tool_versions") or {}).get(ENGINE_ID) or "")
        data["observation"] = {"kind": "media.probe", "provenance": "OBSERVED", "source": f"{ENGINE_ID}/probe@{engine_ver}",
                               "data": {"file": out_path, "duration": art.get("duration"), "size_bytes": art.get("size"),
                                        "audio": {"codec": art.get("codec"), "channels": art.get("channels"), "sample_rate": art.get("sample_rate"), "channel_layout": art.get("channel_layout")},
                                        "video": None}}
        m: Dict[str, Any] = _as_dict(rec.get("measurements"))
        if isinstance(m.get("loudness"), dict):
            lm = m["loudness"]
            data["measurement"] = {"kind": "loudness", "provenance": "OBSERVED", "source": f"{ENGINE_ID}/loudness@{engine_ver}",
                                   "data": {"integrated_lufs": lm.get("integrated_lufs"), "true_peak_dbtp": lm.get("true_peak_dbtp"), "loudness_range_lu": lm.get("loudness_range_lu"),
                                            "target_lufs": lm.get("target_lufs"), "true_peak_db": lm.get("true_peak_db"), "measured_by": lm.get("measured_by")}}
        data["provenance"] = dict(o.get("provenance") or {})
        data["commands"] = [str(c) for r in doc["results"] if isinstance(r, dict) for c in (r.get("tool_commands_observed") or [])] + [str(c) for c in (o.get("tool_commands_observed") or [])]
        data["tool_runs"] = [{"tool": t.get("tool"), "exit_code": t.get("exit_code"), "seconds": t.get("seconds")} for t in doc.get("tool_runs") or [] if isinstance(t, dict)]
        data["output"] = out_path
        return data

    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, code: int, errc: str, message: str, retryable: bool,
              details: Optional[Dict[str, Any]] = None, skill_results: Optional[List[Dict[str, Any]]] = None) -> ToolResult:
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": getattr(self, "version", "")}, "status": "failed",
                                "error": {"code": errc, "message": message, "retryable": bool(retryable), "details": details or {}, "exit_code": code,
                                          "recovery_class": ERROR_CODES.get(errc, "SKILL_ERROR")}}
        if skill_results:
            data["results"] = [{k: r.get(k) for k in ("node_id", "type", "status", "tool")} for r in skill_results if isinstance(r, dict)]
            data["commands"] = [str(c) for r in skill_results if isinstance(r, dict) for c in (r.get("tool_commands_observed") or [])]
        tail = f"audio-production [{errc}] {message}"
        return ToolResult(op.id, op.tool, False, code, None, data, list(data.get("commands") or []), tail, round(time.time() - t0, 3), attempt, dry_run)

    @staticmethod
    def _remove_fresh(path: str, t0: float) -> None:
        try:
            if os.path.isfile(path) and os.path.getmtime(path) >= t0 - 1:
                os.remove(path)
        except OSError:
            pass


def lift_observation(result: ToolResult, asset_id: Optional[str] = None) -> Optional[Observation]:
    """The Skill's OBSERVED probe of a delivered output as an agent Observation (same shape as ADR-028). None when the result
    carries no observation (failure, dry run). Recorded as provenance only; never fed back into the IR's analysis."""
    obs = (result.data or {}).get("observation")
    if not isinstance(obs, dict) or obs.get("provenance") != "OBSERVED" or not isinstance(obs.get("data"), dict):
        return None
    sk = (result.data or {}).get("skill") or {}
    art = (result.data or {}).get("artifact") or {}
    return Observation(kind=str(obs.get("kind") or "media.probe"), asset_id=asset_id or result.op_id, source=str(obs.get("source")), data=dict(obs["data"]),
                       analyzer=str(obs.get("source")), provenance="OBSERVED", skill=SKILL_ID, skill_version=str(sk.get("version") or ""), tool=result.tool,
                       fingerprint=str(art.get("sha256") or ""), parameters={"output": result.output, "operation_id": art.get("operation_id")})


def lift_measurement(result: ToolResult, asset_id: Optional[str] = None) -> Optional[Observation]:
    """The Skill's re-measurement of its own NORMALIZE output (loudness) as an agent Observation (provenance only)."""
    m = (result.data or {}).get("measurement")
    if not isinstance(m, dict) or m.get("provenance") != "OBSERVED" or not isinstance(m.get("data"), dict):
        return None
    sk = (result.data or {}).get("skill") or {}
    art = (result.data or {}).get("artifact") or {}
    return Observation(kind=str(m.get("kind") or "loudness"), asset_id=asset_id or result.op_id, source=str(m.get("source")), data=dict(m["data"]),
                       analyzer=str(m.get("source")), provenance="OBSERVED", skill=SKILL_ID, skill_version=str(sk.get("version") or ""), tool=result.tool,
                       fingerprint=str(art.get("sha256") or ""), parameters={"output": result.output, "operation_id": art.get("operation_id")})
