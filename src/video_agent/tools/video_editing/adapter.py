"""VideoEditingAdapter: the agent's boundary to video-editing-skill (deterministic editing Skill, ADR-028).

Protocol (from the Skill's own contract, `video-editing contract --json`, schema video-editing/contract@1):
    contract → `video-editing contract --json`
    doctor   → `video-editing doctor --json [--workspace DIR] [--allowed-input ROOT]… [--ffmpeg-skill-dir DIR]`
    plan     → `video-editing plan - --json --workspace DIR --allowed-input ROOT…`   (dry run; the Skill writes nothing)
    run      → `video-editing run  - --json --workspace DIR --allowed-input ROOT…`   with an edit request on stdin
    response ← exactly one JSON document on stdout ({"ok": true, execution: {operations[], outputs[]}} or
               {"ok": false, "error": {code, message, retryable, details}}); stderr is diagnostics only.

Responsibility chain: video-production-agent (what to do) → this adapter (typed args → edit request, response → ToolResult)
→ video-editing-skill (what to edit, validated, provenance) → ffmpeg-skill (how to run FFmpeg) → FFmpeg. The adapter never
builds commands, argv, filters or executables, never forwards them, never runs ffmpeg / ffprobe / ffmpeg-skill itself, and
never imports the Skill. The `commands` the Skill reports are stored as provenance only; nothing re-executes them.
"""
from __future__ import annotations

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
from .lowering import ARGS, PREFIX, Lowering, op_type

SKILL_ID = "video-editing"
CONTRACT_SCHEMA = "video-editing/contract@1"
REQUEST_SCHEMA = "video-editing/request@1"
RESPONSE_SCHEMA = "video-editing/response@1"
PLAN_SCHEMA = "video-editing/plan@1"
DOCTOR_SCHEMA = "video-editing/doctor@1"
SUPPORTED_SKILL_VERSIONS = ("0.1.",)          # 0.1.x: the contract this adapter was verified against
ENGINE_ID = "ffmpeg-skill"                    # the only media engine the contract may name
ERROR_CODES = ("INVALID_REQUEST", "INVALID_INPUT", "PATH_NOT_ALLOWED", "UNSUPPORTED_OPERATION", "UNSUPPORTED_FORMAT", "MISSING_INPUT", "INVALID_TIME_RANGE",
               "DEPENDENCY_ERROR", "TOOL_ERROR", "OUTPUT_ERROR", "VALIDATION_ERROR", "CANCELLED", "INTERNAL_ERROR")
EXECUTION_MUST_BE_FALSE = ("shell", "arbitrary_executables", "raw_ffmpeg_arguments", "filter_strings", "network", "ai", "input_mutation")
# runtime capabilities the package needs as CapabilityResolver names (the contract's own capability_names are editing
# capabilities such as video.trim; those describe what the Skill can do, not what the host must provide)
BASE_CAPABILITIES = ["ffmpeg", "ffprobe", "ffmpeg-skill", "video-editing"]
RESULT_KEYS = ["status", "output", "sha256", "timeline", "observation", "operations", "commands", "skill", "engine"]
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ToolError):
    """The installed Skill does not satisfy the contract this adapter was written for."""


def check_contract(contract: Dict[str, Any]) -> List[str]:
    """Compatibility checks: schema / skill id / version range, role and execution guarantees, invocation, tool shape and
    ownership, operations ↔ tools, unsupported ↔ supported, error table, request / response schema ids, engine."""
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
    ex = contract.get("execution") or {}
    if ex.get("mode") != "local_subprocess":
        errs.append(f"execution.mode {ex.get('mode')!r} != local_subprocess")
    inv = ex.get("canonical_invocation") or []
    if not (len(inv) >= 3 and inv[1] == "run" and inv[2] == "-" and "--json" in inv and "--workspace" in inv):
        errs.append(f"canonical_invocation {inv!r} is not `run - --json --workspace …`")
    for k in EXECUTION_MUST_BE_FALSE:
        if ex.get(k) is not False:
            errs.append(f"execution.{k} must be false (got {ex.get(k)!r})")
    schemas = contract.get("schemas") or {}
    if schemas.get("request") != REQUEST_SCHEMA or schemas.get("response") != RESPONSE_SCHEMA or schemas.get("plan") != PLAN_SCHEMA or schemas.get("doctor") != DOCTOR_SCHEMA:
        errs.append(f"schemas {schemas!r} do not name request@1 / response@1 / plan@1 / doctor@1")
    eng = contract.get("engine") or {}
    if eng.get("id") != ENGINE_ID:
        errs.append(f"engine {eng.get('id')!r} != {ENGINE_ID}: this adapter only accepts an ffmpeg-skill backed Skill")
    tools = contract.get("tools") or []
    if not tools:
        errs.append("no tools declared")
    ops = contract.get("operations") or {}
    tool_types = set()
    for t in tools:
        tid = str(t.get("tool_id", ""))
        if not tid.startswith(PREFIX) or tid.count("/") != 1 or t.get("skill_id") != SKILL_ID:
            errs.append(f"tool {tid!r} does not belong to {SKILL_ID}")
            continue
        ot = t.get("operation_type")
        if ot != tid[len(PREFIX):].upper():
            errs.append(f"tool {tid}: operation_type {ot!r} does not match the tool id")
        tool_types.add(ot)
        if not t.get("produces_output") or not t.get("deterministic") or t.get("kind") != "transform":
            errs.append(f"tool {tid}: editing tools must be deterministic transforms that produce output")
        if not t.get("required_capabilities"):
            errs.append(f"tool {tid}: no required_capabilities")
        if not t.get("executed_by", "").startswith(ENGINE_ID + "/"):
            errs.append(f"tool {tid}: executed_by {t.get('executed_by')!r} is not an {ENGINE_ID} tool")
        if ot in ops and set(t.get("parameters") or {}) != set(ops[ot].get("parameters") or {}):
            errs.append(f"tool {tid}: parameters disagree with operations[{ot}]")
    if set(ops) != tool_types:
        errs.append(f"operations {sorted(ops)} and tools {sorted(tool_types)} disagree")
    declared = {c.get("capability") for c in contract.get("capabilities") or []}
    for u in contract.get("unsupported") or []:
        if u.get("status") != "NOT_IMPLEMENTED" or u.get("capability") in declared or u.get("type") in ops:
            errs.append(f"unsupported entry {u.get('type')!r} overlaps the supported set or lacks NOT_IMPLEMENTED")
    er = contract.get("errors") or {}
    codes = set(er.get("codes") or [])
    if codes != set(ERROR_CODES):
        errs.append(f"error codes {sorted(codes)} differ from the expected table")
    if set(er.get("exit_codes") or {}) != set(ERROR_CODES) or not all(isinstance(v, int) for v in (er.get("exit_codes") or {}).values()):
        errs.append("exit_codes do not cover every error code")
    if er.get("retryable_default", {}).get("TOOL_ERROR") is not True or er.get("retryable_default", {}).get("VALIDATION_ERROR") is not False:
        errs.append("retryable defaults changed (TOOL_ERROR must be retryable, VALIDATION_ERROR must not)")
    return errs


def package_from_contract(contract: Dict[str, Any]) -> SkillPackage:
    tools = [ToolSpec(tool_id=t["tool_id"], skill_id=SKILL_ID, version=str(t.get("version") or contract.get("version") or ""), description=t.get("description", ""),
                      required_capabilities=list(t.get("required_capabilities") or []), inputs=list(t.get("inputs") or []) + ["output"],
                      produces_output=True, deterministic=bool(t.get("deterministic", True)), result_keys=list(RESULT_KEYS))
             for t in contract.get("tools") or []]
    return SkillPackage(skill_id=SKILL_ID, name=contract.get("name") or SKILL_ID, version=str(contract.get("version") or ""), description=contract.get("description", ""),
                        capabilities=list(BASE_CAPABILITIES), tools=tools, repository=contract.get("repository", ""), role="editing: deterministic edit execution through ffmpeg-skill")


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


PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    """The contract this adapter was verified against (snapshot of `video-editing contract --json`, 0.1.0). Used for the
    package identity when the Skill is not installed; a live installation always replaces it after check_contract."""
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


PACKAGE = package_from_contract(pinned_contract())


def _scrub(details: Any) -> Dict[str, Any]:
    """Error details are data from the Skill; keep only plain scalars / lists (no nested blobs, no secrets-looking keys)."""
    if not isinstance(details, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in details.items():
        if any(s in str(k).lower() for s in ("key", "token", "secret", "password")):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[str(k)] = v if not isinstance(v, str) else v[:300]
        elif isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v):
            out[str(k)] = [x if not isinstance(x, str) else x[:120] for x in v[:20]]
    return out


class VideoEditingAdapter(ToolAdapter):
    name = SKILL_ID

    def __init__(self, skill: Optional[VideoEditingSkill] = None, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None,
                 path_policy: Optional[PathPolicy] = None, ffmpeg_skill_dir: Optional[str] = None, contract: Optional[Dict[str, Any]] = None, timeout: float = 3600.0):
        self.skill = skill or locate_video_editing()
        if not self.skill:
            raise ToolError("video-editing-skill not found (set VIDEO_AGENT_VIDEO_EDITING_DIR or install the `video-editing` command)")
        self.workspace = str(Path(workspace).resolve()) if workspace else None
        self.allowed_inputs = [str(Path(p).resolve()) for p in (allowed_inputs or [])]
        self.policy = path_policy
        self.ffmpeg_skill_dir = str(Path(ffmpeg_skill_dir).resolve()) if ffmpeg_skill_dir else None
        self.default_timeout = timeout
        self.calls = 0
        self.contract = contract or self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("video-editing contract incompatible: " + "; ".join(errs))
        self.version = str(self.contract["version"])
        self.tools: Dict[str, Dict[str, Any]] = {t["tool_id"]: t for t in self.contract["tools"]}
        self.lowering = Lowering(self.contract, self.workspace or os.getcwd())

    # ---- process boundary
    def _invoke(self, argv: List[str], stdin: Optional[str] = None, timeout: Optional[float] = None) -> "tuple[int, str, str]":
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

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._invoke(["contract", "--json"], timeout=60.0)
        if code != 0:
            raise ContractError(f"video-editing contract --json failed ({code}): {err.strip()[-300:]}")
        return _one_json_document(out)

    def _boundary_argv(self) -> List[str]:
        """The Skill's own path policy is pinned from the agent's PathPolicy: workspace and allowed roots are CLI flags the
        request document cannot override."""
        argv: List[str] = []
        if self.workspace:
            argv += ["--workspace", self.workspace]
        for root in self.allowed_inputs:
            argv += ["--allowed-input", root]
        if self.ffmpeg_skill_dir:
            argv += ["--ffmpeg-skill-dir", self.ffmpeg_skill_dir]
        return argv

    def doctor(self) -> Dict[str, Any]:
        code, out, err = self._invoke(["doctor", "--json"] + self._boundary_argv(), timeout=120.0)
        doc = _one_json_document(out) if out.strip() else {"schema": DOCTOR_SCHEMA, "ok": False, "checks": [], "problems": [err.strip()[-300:] or f"exit {code}"]}
        if doc.get("schema") != DOCTOR_SCHEMA:
            raise ContractError(f"unexpected doctor schema {doc.get('schema')!r}")
        return doc

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "root": self.skill.describe(), "contract": self.contract.get("schema"), "tools": sorted(self.tools),
                "operations": self.lowering.supported_types(), "contract_only": self.lowering.contract_only(),
                "execution": self.contract.get("execution", {}).get("mode"), "engine": (self.contract.get("engine") or {}).get("id")}

    def package(self) -> SkillPackage:
        return package_from_contract(self.contract)

    def supports(self, tool: str) -> bool:
        return tool in self.tools and op_type(tool) in ARGS

    # ---- request construction (typed args → edit request; the agent's PathPolicy is applied first)
    def build_request(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None) -> "tuple[Dict[str, Any], str, List[str]]":
        if op.tool not in self.tools:
            raise ToolError(f"video-editing: tool {op.tool} is not declared by the installed contract")
        if not self.supports(op.tool):
            raise ToolError(f"video-editing: tool {op.tool} is declared by the contract but has no lowering in this adapter")
        req, out_path, in_paths = self.lowering.build_request(op.tool, op.args, paths, op_id=op.id, timeout=timeout)
        if self.policy is not None:
            for p in in_paths:
                self.policy.check_input(p)
            self.policy.check_output(out_path, in_paths)
        return req, out_path, in_paths

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        req, _, _ = self.build_request(op, paths)
        return [" ".join(list(self.skill.command) + ["run", "-", "--json"] + self._boundary_argv()) + "  <<< " + json.dumps(req, sort_keys=True)]

    # ---- execution: one subprocess per call; the response document is mapped to ToolResult
    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        try:
            req, out_path, _ = self.build_request(op, paths, timeout=timeout)
        except ToolError as e:
            return ToolResult(op.id, op.tool, False, 2, None, {"error": {"code": "INVALID_REQUEST", "message": str(e), "retryable": False, "details": {"by": "adapter"}}, "error_kind": "INVALID_REQUEST"},
                              [], str(e), 0.0, attempt, dry_run)
        verb = "plan" if dry_run else "run"
        limit = (timeout + 15.0) if timeout else self.default_timeout   # the Skill applies options.timeout_seconds itself; the agent's guard sits above it
        code, out, err = self._invoke([verb, "-", "--json"] + self._boundary_argv(), stdin=json.dumps(req), timeout=limit)
        secs = round(time.time() - t0, 3)
        tail = "\n".join(err.strip().splitlines()[-12:])
        if code == 124:
            return ToolResult(op.id, op.tool, False, 124, None, {"error": {"code": "CANCELLED", "message": tail or "process timed out", "retryable": True, "details": {"reason": "timeout", "by": "agent"}},
                                                                 "error_kind": "CANCELLED", "request": req}, [], tail, secs, attempt, dry_run)
        try:
            doc = _one_json_document(out)
        except ToolError as e:
            return ToolResult(op.id, op.tool, False, code or 9, None, {"error": {"code": "INVALID_RESULT", "message": str(e), "retryable": False, "details": {"exit_code": code}}, "error_kind": "INVALID_RESULT",
                                                                       "request": req}, [], tail, secs, attempt, dry_run)
        if doc.get("ok") is False:
            e = doc.get("error") if isinstance(doc.get("error"), dict) else {}
            errc = str(e.get("code") or "")
            if errc not in ERROR_CODES:
                return ToolResult(op.id, op.tool, False, code or 9, None, {"error": {"code": "INVALID_RESULT", "message": f"unknown error code {errc!r}", "retryable": False, "details": {"exit_code": code}},
                                                                           "error_kind": "INVALID_RESULT", "request": req}, [], tail, secs, attempt, dry_run)
            data = {"error": {"code": errc, "message": str(e.get("message") or tail or "")[:500], "retryable": bool(e.get("retryable", False)), "details": _scrub(e.get("details"))},
                    "error_kind": errc, "status": doc.get("status"), "request": req, "skill": doc.get("skill")}
            ex = doc.get("execution") if isinstance(doc.get("execution"), dict) else {}
            ops = ex.get("operations") if isinstance(ex.get("operations"), list) else []
            data["operations"] = ops
            return ToolResult(op.id, op.tool, False, code or 1, None, data, _commands(ops), tail, secs, attempt, dry_run)
        errs = self._check_response(doc, req, out_path, dry_run)
        if errs or code != 0:
            msg = "; ".join(errs) if errs else f"exit {code} with an ok response"
            return ToolResult(op.id, op.tool, False, code or 9, None, {"error": {"code": "INVALID_RESULT", "message": msg, "retryable": False, "details": {"exit_code": code}}, "error_kind": "INVALID_RESULT",
                                                                       "request": req}, [], tail, secs, attempt, dry_run)
        if dry_run:
            data = {"status": doc.get("status"), "output": None, "dry_run": True, "plan": doc.get("plan"), "skill": doc.get("skill"), "engine": doc.get("engine"), "request": req, "warnings": doc.get("warnings") or []}
            steps = (doc.get("plan") or {}).get("steps") or []
            cmds = [c for s in steps for c in ((s.get("preview") or {}).get("commands") or [])]
            return ToolResult(op.id, op.tool, True, 0, None, data, cmds, tail, secs, attempt, dry_run)
        ex = doc["execution"]
        o = ex["outputs"][0]
        data = {"status": doc.get("status"), "output": out_path, "sha256": o.get("sha256"), "size": o.get("size"), "timeline": o.get("timeline"), "observation": o.get("observation"),
                "operations": ex.get("operations") or [], "skill": doc.get("skill"), "engine": doc.get("engine"), "reused": doc.get("status") == "reused",
                "request": req, "warnings": doc.get("warnings") or []}
        return ToolResult(op.id, op.tool, True, 0, out_path, data, _commands(ex.get("operations") or []), tail, secs, attempt, dry_run)

    def _check_response(self, doc: Dict[str, Any], req: Dict[str, Any], out_path: str, dry_run: bool) -> List[str]:
        """Success is execution + output existence + the Skill's own validation + hash + provenance; anything less is a failure."""
        errs: List[str] = []
        want = PLAN_SCHEMA if dry_run else RESPONSE_SCHEMA
        if doc.get("schema") != want:
            errs.append(f"response schema {doc.get('schema')!r} != {want}")
        sk = doc.get("skill") or {}
        if sk.get("id") != SKILL_ID:
            errs.append(f"response skill {sk.get('id')!r} != {SKILL_ID}")
        if str(sk.get("version") or "") != self.version:
            errs.append(f"response skill version {sk.get('version')!r} != contract {self.version}")
        if doc.get("ok") is not True:
            errs.append("response has no ok: true")
            return errs
        if dry_run:
            if doc.get("status") != "planned" or doc.get("dry_run") is not True or not isinstance(doc.get("plan"), dict):
                errs.append("plan response is not a dry run")
            return errs
        if doc.get("status") not in ("completed", "reused"):
            errs.append(f"status {doc.get('status')!r} is not completed / reused")
        ex = doc.get("execution")
        if not isinstance(ex, dict) or not isinstance(ex.get("outputs"), list) or len(ex["outputs"]) != 1:
            errs.append("response must carry exactly one execution output")
            return errs
        o = ex["outputs"][0]
        if o.get("delivered") is not True:
            errs.append("output not delivered")
        if os.path.normcase(str(Path(str(o.get("path") or "")).resolve())) != os.path.normcase(str(Path(out_path).resolve())):
            errs.append(f"delivered path {o.get('path')!r} is not the requested output")
        if not os.path.isfile(out_path) or os.path.getsize(out_path) <= 0:
            errs.append("output file missing or empty on disk")
        if not _SHA_RE.match(str(o.get("sha256") or "")):
            errs.append("output has no sha256")
        if not isinstance(o.get("timeline"), dict):
            errs.append("output has no timeline")
        obs = o.get("observation")
        if not isinstance(obs, dict) or obs.get("provenance") != "OBSERVED" or not str(obs.get("source") or "").startswith(ENGINE_ID + "/") or "@" not in str(obs.get("source") or "") or not isinstance(obs.get("data"), dict):
            errs.append("output observation is not an OBSERVED engine measurement")
        ops = ex.get("operations")
        if not isinstance(ops, list) or not ops:
            errs.append("no operation provenance records")
        else:
            for rec in ops:
                if rec.get("status") not in ("completed", "reused") or not rec.get("operation_id") or rec.get("skill") != SKILL_ID or not rec.get("tool", "").startswith(ENGINE_ID + "/"):
                    errs.append(f"operation record {rec.get('operation')!r} is not a completed {SKILL_ID} → {ENGINE_ID} record")
                    break
        return errs

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        raise ToolError("video-editing: an editing Skill has no measurement tools")


def _commands(ops: List[Dict[str, Any]]) -> List[str]:
    """Command lines the Skill reports it ran, flattened for provenance. Never re-executed."""
    out: List[str] = []
    for rec in ops:
        for c in rec.get("commands") or []:
            if isinstance(c, str):
                out.append(c)
    return out


def lift_observation(result: ToolResult, asset_id: Optional[str] = None) -> Optional[Observation]:
    """The Skill's OBSERVED probe of a delivered output as an agent Observation (provenance kept: skill, version, tool,
    engine source). None when the result carries no observation (failure, dry run)."""
    obs = (result.data or {}).get("observation")
    if not isinstance(obs, dict) or obs.get("provenance") != "OBSERVED" or not isinstance(obs.get("data"), dict):
        return None
    sk = (result.data or {}).get("skill") or {}
    return Observation(kind=str(obs.get("kind") or "media.probe"), asset_id=asset_id or result.op_id, source=str(obs.get("source")), data=dict(obs["data"]),
                       analyzer=str(obs.get("source")), provenance="OBSERVED", skill=SKILL_ID, skill_version=str(sk.get("version") or ""), tool=result.tool,
                       fingerprint=str((result.data or {}).get("sha256") or ""), parameters={"output": result.output})
