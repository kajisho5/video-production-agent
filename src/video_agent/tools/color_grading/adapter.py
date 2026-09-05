"""ColorGradingAdapter: the agent's boundary to color-grading-skill (deterministic colour execution Skill, ADR-031).

    video-production-agent ─(typed Operation)─→ ColorGradingAdapter ─(color-grading/request@1 on stdin)─→ `color-grading run - --json`
        ─(typed ffmpeg-skill calls)─→ ffmpeg-skill ─→ FFmpeg

Protocol (from the Skill's own contract, `color-grading skill --json`):
    contract → `color-grading skill --json`
    doctor   → `color-grading doctor --json [--workspace D] [--allowed-input R]… [--allowed-lut L]… [--ffmpeg-skill X]`
    run      → `color-grading run - --json --workspace D --allowed-input R… [--allowed-lut L]… [--ffmpeg-skill X] [--timeout S]` (request on stdin)
    response ← one JSON document: {"ok": true, "status": "ok", "plan", "results", "outputs", "tool_runs"} or {"ok": false, "error": {code, message, retryable, details}}

The agent hands one typed operation: {"operation": HDR_TO_SDR | LUT_APPLY | RETAG | STRIP_DOVI, "input": <id>, "output": <id>, "format": mp4|mov|m4v|mkv,
"lut"?: <artifact id of a .cube file>, <contract parameters by name>}. Every parameter is checked against the parameter schema the
contract declares (type / min / max / enum / required) and forbidden keys are refused by name; the request carries paths only for the
source, the LUT and the output (workspace and allowed roots go on argv). The response is verified: schema / skill / status, the
output's realpath, sha256 recomputed from the file, size, the probe facts and the provenance chain. Commands the Skill observed are
recorded as provenance only."""
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
from ..ffmpeg_skill.adapter import PathPolicy
from ..skill_process import (FORBIDDEN_ARG_KEYS, CliSkill, ContractError, as_dict, drift_report, error_table, failed_result, fingerprint_matches, invoke, one_json_document,
                             remove_fresh, same_file, scan_forbidden, scrub)
from .locate import locate_color_grading

SKILL_ID = "color-grading"
PREFIX = SKILL_ID + "/"
TOOL_ID = "color-grading/run"
CONTRACT_SCHEMA = "color-grading/contract@1"
REQUEST_SCHEMA = "color-grading/request@1"
RESPONSE_SCHEMA = "color-grading/response@1"
DOCTOR_SCHEMA = "color-grading/doctor@1"
SUPPORTED_SKILL_VERSIONS = ("0.1.",)
ENGINE_ID = "ffmpeg-skill"
CANONICAL_INVOCATION = ["color-grading", "run", "-", "--json"]
REQUIRED_EXECUTION_FLAGS = {"shell": False, "arbitrary_executables": False, "arbitrary_filters": False, "network": False, "input_mutation": False, "ai": False}
COMMON_ARGS = ("operation", "input", "output", "format", "lut", "expect")
EXPECT_KEYS = ("width", "height", "duration", "duration_tolerance", "pix_fmt")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DRIFT_KEYS = ("schema", "skill_id", "version", "kind", "tools", "unsupported_operations", "output_formats", "execution", "ffmpeg_skill", "request", "response", "provenance",
              "schema_versions", "errors", "lut", "color_space", "hdr_sdr")
DRIFT_OPERATION_KEYS = ("type", "inputs", "parameters", "tool", "required_capabilities", "changes_duration", "changes_resolution", "deterministic")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def check_contract(contract: Any) -> List[str]:
    """Compatibility checks: schema, id, version range, kind, execution flags, canonical invocation, the single tool, typed operation
    parameters, forbidden fields, error codes. Anything off is refused, never patched."""
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
    fs = contract.get("ffmpeg_skill") or {}
    if str(fs.get("contract_version")) != "1.0" or not isinstance(fs.get("version_window"), dict):
        errs.append("ffmpeg_skill.contract_version / version_window missing")
    ops = {str(o.get("type")): o for o in contract.get("operations") or [] if isinstance(o, dict)}
    if not ops:
        errs.append("no operations declared")
    for t, o in ops.items():
        if not re.match(r"^[A-Z_]{2,32}$", t):
            errs.append(f"operation type {t!r} is not an upper-case identifier")
        if not isinstance(o.get("parameters"), dict) or not isinstance(o.get("required_capabilities"), list):
            errs.append(f"operation {t}: parameters / required_capabilities missing")
        if not str(o.get("tool", "")).startswith(ENGINE_ID + "/"):
            errs.append(f"operation {t}: executed by {o.get('tool')!r}, not an {ENGINE_ID} tool")
        for k, ps in (o.get("parameters") or {}).items():
            if not isinstance(ps, dict) or ps.get("type") not in ("number", "integer", "string", "boolean"):
                errs.append(f"operation {t}: parameter {k!r} has no typed schema")
    unsupported = {str(u.get("type")) for u in contract.get("unsupported_operations") or [] if isinstance(u, dict)}
    if unsupported & set(ops):
        errs.append(f"operations declared both supported and unsupported: {sorted(unsupported & set(ops))}")
    tools = contract.get("tools") or []
    if len(tools) != 1 or tools[0].get("tool_id") != TOOL_ID or tools[0].get("skill_id") != SKILL_ID:
        errs.append(f"expected exactly one tool {TOOL_ID}, contract declares {[t.get('tool_id') for t in tools]}")
    else:
        t = tools[0]
        if t.get("role") != "execution" or t.get("produces_output") is not True or t.get("deterministic") is not True or t.get("input_type") != REQUEST_SCHEMA:
            errs.append("tool must declare role execution, produces_output / deterministic = true and the request schema as input_type")
        if sorted(str(x) for x in t.get("operations") or []) != sorted(ops):
            errs.append(f"tool operations {t.get('operations')!r} != declared operations {sorted(ops)}")
    fmts = contract.get("output_formats") or {}
    if not isinstance(fmts, dict) or "mp4" not in fmts:
        errs.append("output_formats must declare mp4")
    req = contract.get("request") or {}
    if req.get("schema") != REQUEST_SCHEMA:
        errs.append(f"request.schema {req.get('schema')!r} != {REQUEST_SCHEMA}")
    forbidden = set(req.get("forbidden_fields") or [])
    for k in ("command", "argv", "filter", "shell", "exec", "env"):
        if k not in forbidden:
            errs.append(f"request.forbidden_fields lacks {k!r}")
    retry, exit_codes = error_table(contract)
    for c in ("INVALID_REQUEST", "INVALID_INPUT", "PATH_NOT_ALLOWED", "UNSUPPORTED_OPERATION", "MISSING_INPUT", "TOOL_ERROR", "OUTPUT_ERROR", "VALIDATION_ERROR", "CANCELLED", "INTERNAL_ERROR"):
        if c not in retry:
            errs.append(f"errors.codes lacks {c}")
    if not exit_codes:
        errs.append("errors.exit_codes missing")
    return errs


def contract_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(live, pinned or pinned_contract(), DRIFT_KEYS, "operations", "type", DRIFT_OPERATION_KEYS)


def package_from_contract(contract: Dict[str, Any]) -> SkillPackage:
    ver = str(contract.get("version") or "")
    tools = [ToolSpec(tool_id=TOOL_ID, skill_id=SKILL_ID, version=ver, description=str(t.get("description", "")), required_capabilities=[SKILL_ID],
                      inputs=["input", "output", "lut"], produces_output=True, deterministic=True,
                      result_keys=["operation_id", "operation", "artifact", "observation", "provenance", "commands"]) for t in contract.get("tools") or []]
    return SkillPackage(skill_id=SKILL_ID, name=str(contract.get("name") or SKILL_ID), version=ver, description=str(contract.get("description", ""))[:200],
                        capabilities=["ffmpeg", "ffprobe", "ffmpeg-skill", SKILL_ID], tools=tools, repository="kajisho5/color-grading-skill",
                        role="deterministic colour grading execution (HDR→SDR, LUT, retag, Dolby Vision strip) through ffmpeg-skill")


PACKAGE = package_from_contract(pinned_contract())


class ColorGradingAdapter(ToolAdapter):
    name = SKILL_ID

    def __init__(self, skill: Optional[CliSkill] = None, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None,
                 ffmpeg_skill_dir: Optional[str] = None, timeout: float = 900.0, path_policy: Optional[PathPolicy] = None):
        located = skill or locate_color_grading()
        if not located:
            raise ToolError("color-grading-skill not found (set VIDEO_AGENT_COLOR_GRADING_DIR or install `color-grading`)")
        self.skill: CliSkill = located
        self.workspace = str(Path(workspace).resolve()) if workspace else None
        self.allowed_inputs = [str(Path(r).resolve()) for r in (allowed_inputs or [])]
        self.ffmpeg_skill_dir = ffmpeg_skill_dir
        self.default_timeout = float(timeout)
        self.path_policy = path_policy
        self.calls = 0
        self.contract = self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("color-grading contract incompatible: " + "; ".join(errs))
        self.version = str(self.contract["version"])
        self.operations: Dict[str, Dict[str, Any]] = {str(o["type"]): o for o in self.contract["operations"]}
        self.unsupported = sorted(str(u.get("type")) for u in self.contract.get("unsupported_operations") or [])
        self.formats = sorted(str(k) for k in (self.contract.get("output_formats") or {}))
        self.forbidden = tuple(sorted(set(FORBIDDEN_ARG_KEYS) | {str(f) for f in (self.contract.get("request") or {}).get("forbidden_fields") or []}))
        self.retryable, self.exit_codes = error_table(self.contract)
        self.tools = {TOOL_ID}
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
            raise ContractError(f"color-grading skill --json failed (exit {code}): {err.strip()[-300:]}")
        try:
            return one_json_document(out, "color-grading contract")
        except ToolError as e:
            raise ContractError(str(e))

    def doctor(self) -> Dict[str, Any]:
        argv = ["doctor", "--json"] + (["--workspace", self.workspace] if self.workspace else []) + [x for r in self.allowed_inputs for x in ("--allowed-input", r)] + self._engine_argv()
        code, out, err = self._invoke(argv, timeout=180.0)
        try:
            doc = one_json_document(out, "color-grading doctor")
        except ToolError as e:
            return {"schema": DOCTOR_SCHEMA, "status": "fail", "problems": [f"doctor produced no document: {e}"], "checks": {}, "exit_code": code}
        if doc.get("schema") != DOCTOR_SCHEMA or doc.get("status") not in ("ok", "degraded", "fail"):
            return {"schema": DOCTOR_SCHEMA, "status": "fail", "problems": [f"unexpected doctor document {doc.get('schema')!r} / {doc.get('status')!r}"], "checks": doc.get("checks") or {}, "exit_code": code}
        doc["exit_code"] = code
        return doc

    @staticmethod
    def operation_status(doc: Dict[str, Any]) -> Dict[str, str]:
        """Per operation type: supported | unsupported | unknown, as the Skill's doctor reports it (never inferred here)."""
        ops = as_dict(as_dict(doc.get("checks")).get("operations"))
        return {str(t): str(as_dict(v).get("status") or "unknown") for t, v in ops.items()}

    def drift(self) -> List[str]:
        if self._drift is None:
            self._drift = contract_drift(self.contract)
        return self._drift

    # ---- ToolAdapter
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "root": self.skill.describe(), "tools": sorted(self.tools), "operations": sorted(self.operations),
                "unsupported": self.unsupported, "formats": self.formats, "contract": self.contract.get("schema"), "drift": self.drift()}

    def package(self) -> SkillPackage:
        return package_from_contract(self.contract)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            b = self.build_request(op.tool, op.args, paths, op_id=op.id)
        except ToolError as e:
            return [f"color-grading: refused: {e}"]
        return [" ".join(["color-grading"] + self._argv(b, dry_run=True, timeout=None)) + "  <<< " + json.dumps(b["request"], ensure_ascii=False)[:400]]

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        raise ToolError("color-grading: an execution Skill has no measurement tools")

    # ---- lowering: typed args → one request document
    def params_for(self, op_type: str, args: Dict[str, Any], paths: Dict[str, str]) -> Dict[str, Any]:
        if op_type in self.unsupported:
            raise ToolError(f"color-grading: operation {op_type} is declared unsupported by the Skill contract")
        spec = self.operations.get(op_type)
        if spec is None:
            raise ToolError(f"color-grading: unknown operation type {op_type!r} (contract declares {sorted(self.operations)})")
        schema: Dict[str, Any] = spec.get("parameters") or {}
        out: Dict[str, Any] = {}
        for k, v in args.items():
            if k in COMMON_ARGS:
                continue
            if k.lower() in {f.lower() for f in self.forbidden}:
                raise ToolError(f"color-grading: parameter {k!r} is not accepted (forbidden field)")
            if k not in schema:
                raise ToolError(f"color-grading: parameter {k!r} is not declared for {op_type} (contract: {sorted(schema)})")
            out[k] = self._typed(k, v, schema[k])
        if "lut" in args:
            if "lut_path" not in schema:
                raise ToolError(f"color-grading: {op_type} takes no LUT")
            lut = paths.get(str(args["lut"]), str(args["lut"]))
            if Path(lut).suffix.lower() != ".cube" or not os.path.isfile(lut):
                raise ToolError(f"color-grading: LUT must be an existing .cube file: {lut}")
            out["lut_path"] = str(Path(lut).resolve())
        for k, ps in schema.items():
            if ps.get("required") and k not in out:
                raise ToolError(f"color-grading: {op_type} requires parameter {k!r}")
        return out

    @staticmethod
    def _typed(name: str, v: Any, ps: Dict[str, Any]) -> Any:
        t = ps.get("type")
        if t == "boolean":
            if not isinstance(v, bool):
                raise ToolError(f"color-grading: {name} must be a boolean")
            return v
        if t == "string":
            if not isinstance(v, str) or "\n" in v or "\x00" in v:
                raise ToolError(f"color-grading: {name} must be a string")
            if "enum" in ps and v not in ps["enum"]:
                raise ToolError(f"color-grading: {name} {v!r} is not one of {ps['enum']}")
            return v
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
            raise ToolError(f"color-grading: {name} must be a finite number")
        x: Any = float(v)
        if t == "integer":
            if float(x) != int(x):
                raise ToolError(f"color-grading: {name} must be an integer")
            x = int(x)
        if "min" in ps and x < ps["min"]:
            raise ToolError(f"color-grading: {name} {x} is below the contract minimum {ps['min']}")
        if "max" in ps and x > ps["max"]:
            raise ToolError(f"color-grading: {name} {x} is above the contract maximum {ps['max']}")
        return x

    def build_request(self, tool: str, args: Dict[str, Any], paths: Dict[str, str], op_id: str = "op", timeout: Optional[float] = None) -> Dict[str, Any]:
        if tool != TOOL_ID:
            raise ToolError(f"color-grading: unsupported tool {tool}")
        hit = scan_forbidden(args, self.forbidden)
        if hit:
            raise ToolError(f"color-grading: forbidden field {hit} in the operation arguments")
        op_type = str(args.get("operation") or "")
        src_id, out_id = str(args.get("input") or ""), str(args.get("output") or "")
        if not src_id or not out_id:
            raise ToolError("color-grading: input and output references are required")
        src = paths.get(src_id, src_id)
        out = paths.get(out_id, out_id)
        if not os.path.isfile(src):
            raise ToolError(f"color-grading: input not found: {src}")
        if self.path_policy is not None:
            self.path_policy.check_input(src)
            self.path_policy.check_output(out, [src])
        elif self.allowed_inputs and not any(self._under(src, r) for r in self.allowed_inputs + ([self.workspace] if self.workspace else [])):
            raise ToolError(f"color-grading: input outside the allowed roots: {src}")
        if self.workspace and not self._under(out, self.workspace):
            raise ToolError(f"color-grading: output outside the workspace: {out}")
        fmt = str(args.get("format") or Path(out).suffix.lstrip(".").lower() or "mp4")
        if fmt not in self.formats:
            raise ToolError(f"color-grading: output format {fmt!r} is not one of {self.formats}")
        if Path(out).suffix.lower().lstrip(".") != fmt or Path(src).suffix.lower().lstrip(".") != fmt:
            raise ToolError(f"color-grading: the Skill does not convert containers; source, output and format must agree ({fmt})")
        params = self.params_for(op_type, args, paths)
        expect = {k: args["expect"][k] for k in EXPECT_KEYS if isinstance(args.get("expect"), dict) and k in args["expect"]}
        pid = re.sub(r"[^A-Za-z0-9._-]", "_", str(op_id))[:64] or "op"
        if not _ID_RE.match(pid):
            pid = "op"
        req: Dict[str, Any] = {"schema": REQUEST_SCHEMA,
                               "project": {"project_id": pid, "source": {"source_id": "src", "path": str(Path(src).resolve())},
                                           "operations": [{"op_id": "edit", "type": op_type, "input": "source", "parameters": params}],
                                           "outputs": [{"output_id": "out", "operation": "op:edit", "path": str(Path(out).resolve()), "format": fmt, "overwrite": True, **({"expect": expect} if expect else {})}]}}
        if timeout:
            req["options"] = {"timeout": float(timeout)}
        hit = scan_forbidden(req, tuple(f for f in self.forbidden if f not in ("path", "paths", "workspace")), "request")
        if hit:
            raise ToolError(f"color-grading: refusing to send a request carrying {hit}")
        lut_root = str(Path(params["lut_path"]).parent) if "lut_path" in params else None
        return {"request": req, "output": out, "input": src, "type": op_type, "workspace": self.workspace or str(Path(out).parent), "lut_root": lut_root}

    @staticmethod
    def _under(path: str, root: str) -> bool:
        try:
            p, r = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
        except OSError:
            return False
        return p == r or p.startswith(r.rstrip(os.sep) + os.sep)

    def _argv(self, b: Dict[str, Any], dry_run: bool, timeout: Optional[float]) -> List[str]:
        argv = ["run", "-", "--json", "--workspace", b["workspace"]]
        roots = list(self.allowed_inputs) + ([self.workspace] if self.workspace else [])
        for r in roots:
            argv += ["--allowed-input", r]
        if b.get("lut_root"):
            argv += ["--allowed-lut", b["lut_root"]]
        argv += self._engine_argv()
        argv += ["--timeout", str(int(timeout or self.default_timeout))]
        if dry_run:
            argv.append("--dry-run")
        return argv

    # ---- execution
    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        try:
            b = self.build_request(op.tool, op.args, paths, op_id=op.id, timeout=timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e), retryable=False)
        os.makedirs(b["workspace"], exist_ok=True)
        os.makedirs(os.path.dirname(b["output"]) or ".", exist_ok=True)
        argv = self._argv(b, dry_run, timeout)
        code, out, err = self._invoke(argv, stdin=json.dumps(b["request"], ensure_ascii=False), timeout=(timeout or self.default_timeout) + 5.0)
        tail = "\n".join(err.strip().splitlines()[-12:])
        secs = round(time.time() - t0, 3)
        if code == 124:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, 124, "CANCELLED", tail or "process timed out", retryable=True, details={"reason": "timeout"})
        try:
            doc = one_json_document(out, "color-grading")
        except ToolError as e:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 9, "INVALID_RESULT", str(e), retryable=False, details={"exit_code": code})
        if doc.get("ok") is not True:
            edoc = as_dict(doc.get("error"))
            errc = str(edoc.get("code") or "INVALID_RESULT")
            if errc not in self.retryable:
                errc, retry = "INVALID_RESULT", False
            else:
                retry = bool(edoc["retryable"]) if isinstance(edoc.get("retryable"), bool) else self.retryable.get(errc, False)
            details = scrub(edoc.get("details"), self.forbidden)
            if errc == "CANCELLED" and (details.get("reason") or "") not in ("timeout", "signal"):
                details["reason"] = details.get("reason") or "signal"
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 1, errc, str(edoc.get("message") or tail or "")[:500], retryable=bool(retry), details=details)
        errs = self._check_response(doc, b["output"], dry_run)
        if errs:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 9, "INVALID_RESULT", "; ".join(errs), retryable=False, details={"exit_code": code})
        if code != 0:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code, "INVALID_RESULT", f"exit code {code} with an ok response", retryable=False)
        data = self._success_data(doc, b["output"], dry_run, b["type"])
        return ToolResult(op.id, op.tool, True, 0, None if dry_run else b["output"], data, list(data.get("commands") or []), tail, secs, attempt, dry_run)

    def _check_response(self, doc: Dict[str, Any], out_path: str, dry_run: bool) -> List[str]:
        errs: List[str] = []
        if doc.get("schema") != RESPONSE_SCHEMA:
            errs.append(f"response schema {doc.get('schema')!r}")
        sk = as_dict(doc.get("skill"))
        if sk.get("id") != SKILL_ID or str(sk.get("version")) != self.version:
            errs.append(f"response skill {sk!r} is not {SKILL_ID}@{self.version}")
        if doc.get("status") != "ok":
            errs.append(f"status {doc.get('status')!r} is not ok")
        if bool(doc.get("dry_run")) != bool(dry_run):
            errs.append("dry_run flag does not match the request")
        if not isinstance(as_dict(doc.get("plan")).get("graph"), dict):
            errs.append("plan.graph missing")
        if dry_run:
            return errs
        res = [r for r in doc.get("results") or [] if isinstance(r, dict) and r.get("node_id") == "op:edit"]
        if len(res) != 1 or res[0].get("status") not in ("completed", "reused") or not str(res[0].get("tool", "")).startswith(ENGINE_ID + "/"):
            errs.append("results lack a completed op:edit executed by an ffmpeg-skill tool")
        outs = [o for o in doc.get("outputs") or [] if isinstance(o, dict) and o.get("output_id") == "out"]
        if len(outs) != 1:
            return errs + ["outputs lack out"]
        o = outs[0]
        if o.get("status") != "completed":
            errs.append(f"output status {o.get('status')!r}")
        if not o.get("path") or not same_file(str(o["path"]), out_path):
            errs.append(f"output path {o.get('path')!r} is not the requested {out_path}")
        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            return errs + ["output file missing or empty"]
        art = as_dict(o.get("artifact"))
        ok, actual = fingerprint_matches(art.get("sha256"), out_path)
        if not ok:
            errs.append(f"artifact sha256 {art.get('sha256')!r} != file {actual}")
        if art.get("size") != os.path.getsize(out_path):
            errs.append("artifact size differs from the file")
        for k in ("duration", "width", "height", "codec"):
            if art.get(k) in (None, ""):
                errs.append(f"artifact lacks probe fact {k}")
        prov = as_dict(o.get("provenance"))
        if prov.get("skill") != SKILL_ID or str(prov.get("skill_version")) != self.version or prov.get("output_hash") != art.get("sha256") or not prov.get("operations"):
            errs.append("provenance incomplete (skill / version / output_hash / operations)")
        if not isinstance(as_dict(prov.get("tool_versions")).get(ENGINE_ID), str):
            errs.append("provenance.tool_versions lacks ffmpeg-skill")
        return errs

    def _success_data(self, doc: Dict[str, Any], out_path: str, dry_run: bool, op_type: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run" if dry_run else "completed", "operation_type": op_type,
                                "plan": {"plan_id": as_dict(doc.get("plan")).get("plan_id"), "order": as_dict(as_dict(doc.get("plan")).get("graph")).get("order")}, "warnings": list(doc.get("warnings") or [])}
        runs = [r for r in doc.get("tool_runs") or [] if isinstance(r, dict)]
        data["commands"] = [str(c) for r in (doc.get("results") or []) if isinstance(r, dict) for c in (r.get("tool_commands_observed") or [])] or [str(c) for r in runs for c in (r.get("commands_observed") or [])]
        data["tool_runs"] = [{k: r.get(k) for k in ("tool", "exit_code", "seconds")} for r in runs]
        if dry_run:
            return data
        res = next(r for r in doc["results"] if isinstance(r, dict) and r.get("node_id") == "op:edit")
        o = next(x for x in doc["outputs"] if isinstance(x, dict) and x.get("output_id") == "out")
        art, prov = as_dict(o.get("artifact")), as_dict(o.get("provenance"))
        data["operation_id"] = str(res.get("operation_id") or "")
        data["operation"] = {k: res.get(k) for k in ("operation_id", "type", "tool", "status", "parameters", "input_hash", "measurements", "required_capabilities")}
        data["artifact"] = {"path": out_path, "sha256": art.get("sha256"), "size": art.get("size"), "duration": art.get("duration"), "width": art.get("width"), "height": art.get("height"),
                            "codec": art.get("codec"), "pix_fmt": art.get("pix_fmt"), "color_space": art.get("color_space"), "color_primaries": art.get("color_primaries"),
                            "color_transfer": art.get("color_transfer"), "color_range": art.get("color_range"), "hdr": art.get("hdr"), "dolby_vision": art.get("dolby_vision"),
                            "operation_id": data["operation_id"], "reused": res.get("status") == "reused"}
        engine_ver = str(as_dict(prov.get("tool_versions")).get(ENGINE_ID) or "")
        data["observation"] = {"kind": "media.probe", "source": f"{ENGINE_ID}/probe@{engine_ver}", "provenance": "OBSERVED",
                               "data": {k: art.get(k) for k in ("duration", "width", "height", "codec", "pix_fmt", "color_space", "color_primaries", "color_transfer", "color_range", "hdr", "dolby_vision", "size", "sha256")}}
        data["provenance"] = {"skill": prov.get("skill"), "skill_version": prov.get("skill_version"), "tool_versions": prov.get("tool_versions"), "output_hash": prov.get("output_hash"),
                              "operations": prov.get("operations"), "source": prov.get("source"), "operation_id": prov.get("operation_id")}
        return data

    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, code: int, errc: str, message: str, retryable: bool, details: Optional[Dict[str, Any]] = None) -> ToolResult:
        return failed_result(op, SKILL_ID, getattr(self, "version", ""), attempt, dry_run, t0, code, errc, message, retryable, details)


def lift_observation(result: ToolResult, asset_id: Optional[str] = None) -> Optional[Observation]:
    """The Skill's OBSERVED probe of the delivered output as an agent Observation (provenance only; never fed back into analysis)."""
    obs = (result.data or {}).get("observation")
    if not isinstance(obs, dict) or obs.get("provenance") != "OBSERVED" or not isinstance(obs.get("data"), dict):
        return None
    sk = (result.data or {}).get("skill") or {}
    art = (result.data or {}).get("artifact") or {}
    return Observation(kind=str(obs.get("kind") or "media.probe"), asset_id=asset_id or result.op_id, source=str(obs.get("source")), data=dict(obs["data"]), analyzer=str(obs.get("source")),
                       provenance="OBSERVED", skill=SKILL_ID, skill_version=str(sk.get("version") or ""), tool=result.tool, fingerprint=str(art.get("sha256") or ""),
                       parameters={"output": result.output, "operation_id": art.get("operation_id")})
