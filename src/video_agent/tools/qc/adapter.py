"""QcAdapter: the agent's boundary to qc-skill (deterministic media quality control / validation Skill, ADR-031).

    video-production-agent ─(typed Operation)─→ QcAdapter ─(qc/request@1 on stdin)─→ `qc run - --json` ─→ ffprobe / ffmpeg (PATH)

Protocol (from the Skill's own contract, `qc contract --json`):
    contract → `qc contract --json`
    doctor   → `qc doctor --json [--workspace D]`                     (exit 0 ok / 2 degraded / 1 fail)
    run      → `qc run - --json --workspace D --allowed-input-root R… [--no-cache]` (request on stdin)
    response ← one JSON document: {"schema": "qc/response@1", "status": "completed", "skill", "report", "provenance", "reused", "cache"}
               or {"schema": "qc/response@1", "status": "failed", "skill", "error": {code, message, retryable, details}}

qc-skill is a measurement Skill: it writes NO media and its result is a report. The agent hands one typed operation:
{"input": <artifact id>, "kind": video|audio|subtitle|delivery, "rules"?: {video?, audio?, subtitle?, delivery?}, "parameters"?: {…},
"subtitle"?: <artifact id> (kind delivery only), "reference_video"?: <artifact id> (kind subtitle only), "cache_policy"?: use|bypass}.
Rule sections / field names / value types and parameter names come from the contract's `rules` and `parameters` blocks; forbidden keys
are refused by name; the request carries paths only for the input and its companions (workspace and allowed roots go on argv).

ADMISSION is the whole point of this adapter: a report is accepted only when the response schema, status, skill id and version, the
report's operation and kind, provenance.measurement_source == OBSERVED, the report's fingerprint == provenance fingerprint == the sha256
the adapter recomputes from the input file (companions likewise), the verdict and every check status are contract statuses, and the
report id is a qc report id. Anything else → INVALID_RESULT: the QA gate never sees a report about a different file. A FAIL or WARN
verdict is a SUCCESSFUL result (ok=True) — the verdict is evidence, never a production decision.

Contract discrepancies absorbed here (documented, never patched):
  * the contract declares operations [inspect, check, validate] but no tool ids; the agent defines qc/check and qc/inspect. `validate`
    is the same pipeline as `check` in this Skill version and is not exposed.
  * the contract carries no request.forbidden_fields; the Skill's 14 forbidden request keys (schemas.py) are pinned here.
  * the contract's errors block has no retryable map, so error_table() yields non-retryable for every code; the Skill's own
    `retryable` verdict in a failed response is preferred (TOOL_ERROR / CANCELLED are retryable in the Skill's errors.py).
  * the CLI has no --timeout flag and the request's `timeout` is parsed but unused; the process-boundary timeout is the only timeout.
  * the CLI flag is --allowed-input-root (not --allowed-input); there is no --ffmpeg-skill (ffprobe/ffmpeg come from PATH).
  * a measurement has no dry run: run(dry_run=True) validates the request and returns without invoking the Skill."""
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
from ..skill_process import (FORBIDDEN_ARG_KEYS, CliSkill, ContractError, as_dict, drift_report, error_table, failed_result, invoke, one_json_document, scan_forbidden,
                             scrub, sha256_file, strip_sha_prefix)
from .locate import locate_qc

SKILL_ID = "qc"
PREFIX = SKILL_ID + "/"
TOOL_CHECK = "qc/check"
TOOL_INSPECT = "qc/inspect"
TOOL_OPERATIONS = {TOOL_CHECK: "check", TOOL_INSPECT: "inspect"}
CONTRACT_SCHEMA = "qc/contract@1"
REQUEST_SCHEMA = "qc/request@1"
RESPONSE_SCHEMA = "qc/response@1"
DOCTOR_SCHEMA = "qc/doctor@1"
SUPPORTED_SKILL_VERSIONS = ("0.1.",)
CANONICAL_INVOCATION = ["qc", "run", "-", "--json"]
REQUIRED_EXECUTION_FLAGS = {"shell": False, "arbitrary_executables": False, "arbitrary_filters": False, "network": False, "input_mutation": False}
ALLOWED_EXECUTABLES = {"ffprobe", "ffmpeg"}
REQUIRED_KINDS = ("video", "audio", "subtitle", "delivery")
REQUIRED_STATUSES = ("PASS", "WARN", "FAIL", "UNKNOWN")
REQUIRED_ERROR_CODES = ("INVALID_REQUEST", "INVALID_INPUT", "PATH_NOT_ALLOWED", "UNSUPPORTED_OPERATION", "MISSING_INPUT", "TOOL_ERROR", "VALIDATION_ERROR", "CANCELLED", "INTERNAL_ERROR")
# the Skill's own forbidden request keys (qc_skill/schemas.py FORBIDDEN_KEYS): not declared in the contract, pinned here
SKILL_FORBIDDEN_KEYS = ("command", "commands", "argv", "args", "shell", "cmd", "cmdline", "exec", "executable", "filter", "filter_complex", "env", "environment")
COMMON_ARGS = ("input", "kind", "rules", "parameters", "subtitle", "reference_video", "cache_policy")
AGENT_CACHE_POLICIES = ("use", "bypass")   # "only" (fail unless cached) is never useful to the agent
REPORT_ID_PREFIX = "qcreport_"
_RULE_TYPE_RE = re.compile(r"^(Optional\[)?(int|float|bool|str|[A-Za-z]+Rule)(\])?$")
DRIFT_KEYS = ("schema", "skill_id", "version", "contract_version", "role", "kinds", "operations", "statuses", "checks", "rules", "parameters", "inputs", "outputs",
              "execution", "capabilities", "provenance", "identity", "cache", "errors", "deterministic", "measurements")
DRIFT_FINDING_KEYS = ("category", "default_severity")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def check_contract(contract: Any) -> List[str]:
    """Compatibility checks: schema, id, version range, role, operations, kinds, statuses, execution flags and executables,
    canonical invocation, provenance, typed rule sections, parameter names, the report output, error codes. Anything off is
    refused, never patched."""
    errs: List[str] = []
    if not isinstance(contract, dict):
        return ["contract is not an object"]
    if contract.get("schema") != CONTRACT_SCHEMA:
        errs.append(f"contract schema {contract.get('schema')!r} != {CONTRACT_SCHEMA}")
    if contract.get("skill_id") != SKILL_ID or contract.get("name") != SKILL_ID:
        errs.append(f"skill_id {contract.get('skill_id')!r} / name {contract.get('name')!r} != {SKILL_ID}")
    ver = str(contract.get("version") or "")
    if not ver.startswith(SUPPORTED_SKILL_VERSIONS):
        errs.append(f"skill version {ver!r} not in supported range {SUPPORTED_SKILL_VERSIONS}")
    if str(contract.get("contract_version")) != "1":
        errs.append(f"contract_version {contract.get('contract_version')!r} != 1")
    if contract.get("role") != "observation / validation":
        errs.append(f"role {contract.get('role')!r} != 'observation / validation'")
    if contract.get("provenance") != "OBSERVED":
        errs.append(f"provenance {contract.get('provenance')!r} != OBSERVED")
    if contract.get("deterministic") is not True:
        errs.append("deterministic must be true")
    ops = [str(o) for o in contract.get("operations") or []]
    for o in TOOL_OPERATIONS.values():
        if o not in ops:
            errs.append(f"operations lack {o!r}")
    kinds = [str(k) for k in contract.get("kinds") or []]
    for k in REQUIRED_KINDS:
        if k not in kinds:
            errs.append(f"kinds lack {k!r}")
    statuses = [str(s) for s in contract.get("statuses") or []]
    for s in REQUIRED_STATUSES:
        if s not in statuses:
            errs.append(f"statuses lack {s!r}")
    ex = contract.get("execution") or {}
    if ex.get("mode") != "local_subprocess":
        errs.append(f"execution.mode {ex.get('mode')!r} != local_subprocess")
    for k, want in REQUIRED_EXECUTION_FLAGS.items():
        if ex.get(k) is not want:
            errs.append(f"execution.{k} must be {want!r}, contract says {ex.get(k)!r}")
    if list(ex.get("canonical_invocation") or []) != CANONICAL_INVOCATION:
        errs.append(f"canonical_invocation {ex.get('canonical_invocation')!r} != {CANONICAL_INVOCATION}")
    extra = {str(x) for x in ex.get("executables") or []} - ALLOWED_EXECUTABLES
    if extra:
        errs.append(f"execution.executables names {sorted(extra)} beyond {sorted(ALLOWED_EXECUTABLES)}")
    caps = contract.get("capabilities") or {}
    if "ffprobe" not in [str(c) for c in caps.get("required") or []]:
        errs.append("capabilities.required lacks ffprobe")
    rules = contract.get("rules")
    if not isinstance(rules, dict):
        errs.append("rules block missing")
    else:
        for section in REQUIRED_KINDS:
            fields = rules.get(section)
            if not isinstance(fields, dict) or not fields:
                errs.append(f"rules.{section} missing")
                continue
            for name, spec in fields.items():
                if not isinstance(spec, dict) or not _RULE_TYPE_RE.match(str(spec.get("type", ""))):
                    errs.append(f"rules.{section}.{name} has no typed schema")
                elif spec.get("nested_rule") is not None and spec.get("nested_rule") not in rules:
                    errs.append(f"rules.{section}.{name} nests an undeclared rule {spec.get('nested_rule')!r}")
    params = contract.get("parameters")
    if not isinstance(params, list) or not all(isinstance(p, str) and re.match(r"^[a-z][a-z0-9_]{0,63}$", p) for p in params):
        errs.append("parameters must be a list of names")
    if list(contract.get("outputs") or []) != ["report"]:
        errs.append(f"outputs {contract.get('outputs')!r} != ['report'] (a measurement Skill writes no media)")
    if not isinstance(contract.get("checks"), list) or not contract.get("checks"):
        errs.append("checks missing")
    if not all(isinstance(f, dict) and f.get("code") and f.get("default_severity") in statuses for f in contract.get("findings") or []):
        errs.append("findings must declare code and a default_severity from statuses")
    cache = contract.get("cache") or {}
    for p in AGENT_CACHE_POLICIES:
        if p not in [str(x) for x in cache.get("policies") or []]:
            errs.append(f"cache.policies lacks {p!r}")
    retry, exit_codes = error_table(contract)
    for c in REQUIRED_ERROR_CODES:
        if c not in retry:
            errs.append(f"errors.codes lacks {c}")
    if not exit_codes:
        errs.append("errors.exit_codes missing")
    return errs


def contract_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(live, pinned or pinned_contract(), DRIFT_KEYS, "findings", "code", DRIFT_FINDING_KEYS)


def package_from_contract(contract: Dict[str, Any]) -> SkillPackage:
    ver = str(contract.get("version") or "")
    keys = ["verdict", "report_id", "checks", "findings", "measurements", "fingerprint", "provenance"]
    tools = [ToolSpec(tool_id=TOOL_CHECK, skill_id=SKILL_ID, version=ver, description="measure one artifact and evaluate caller-supplied rules; the verdict (PASS/WARN/FAIL/UNKNOWN) is evidence, never a decision",
                      required_capabilities=[SKILL_ID], inputs=["input", "subtitle", "reference_video"], produces_output=False, deterministic=True, result_keys=list(keys)),
             ToolSpec(tool_id=TOOL_INSPECT, skill_id=SKILL_ID, version=ver, description="measure one artifact without rules (measurements only, no checks)",
                      required_capabilities=[SKILL_ID], inputs=["input", "subtitle", "reference_video"], produces_output=False, deterministic=True, result_keys=list(keys))]
    return SkillPackage(skill_id=SKILL_ID, name=str(contract.get("name") or SKILL_ID), version=ver, description=str(contract.get("description", ""))[:200],
                        capabilities=["ffprobe", SKILL_ID], tools=tools, repository="kajisho5/qc-skill",
                        role="media quality control measurement and rule evaluation (observation; never a production decision)")


PACKAGE = package_from_contract(pinned_contract())


class QcAdapter(ToolAdapter):
    name = SKILL_ID
    owns_cache = True   # the Skill keeps its own report cache (keyed by content fingerprint); the agent records its status, never duplicates it

    def __init__(self, skill: Optional[CliSkill] = None, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None,
                 ffmpeg_skill_dir: Optional[str] = None, timeout: float = 600.0, path_policy: Optional[PathPolicy] = None):
        located = skill or locate_qc()
        if not located:
            raise ToolError("qc-skill not found (set VIDEO_AGENT_QC_DIR or install `qc`)")
        self.skill: CliSkill = located
        self.workspace = str(Path(workspace).resolve()) if workspace else None
        self.allowed_inputs = [str(Path(r).resolve()) for r in (allowed_inputs or [])]
        self.ffmpeg_skill_dir = ffmpeg_skill_dir   # accepted for the family's constructor shape; qc-skill does not use ffmpeg-skill
        self.default_timeout = float(timeout)
        self.path_policy = path_policy
        self.calls = 0
        self.contract = self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("qc contract incompatible: " + "; ".join(errs))
        self.version = str(self.contract["version"])
        self.kinds = [str(k) for k in self.contract["kinds"]]
        self.statuses = [str(s) for s in self.contract["statuses"]]
        self.rules: Dict[str, Dict[str, Any]] = {str(k): dict(v) for k, v in self.contract["rules"].items()}
        self.parameters = [str(p) for p in self.contract["parameters"]]
        self.cache_policies = [str(p) for p in (self.contract.get("cache") or {}).get("policies") or []]
        self.forbidden = tuple(sorted(set(FORBIDDEN_ARG_KEYS) | set(SKILL_FORBIDDEN_KEYS)))
        self.retryable, self.exit_codes = error_table(self.contract)
        self.tools = set(TOOL_OPERATIONS)
        self._drift: Optional[List[str]] = None

    # ---- transport
    def _invoke(self, argv: List[str], stdin: Optional[str] = None, timeout: Optional[float] = None):
        self.calls += 1
        return invoke(self.skill, argv, stdin=stdin, timeout=timeout or self.default_timeout)

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._invoke(["contract", "--json"], timeout=60.0)
        if code != 0:
            raise ContractError(f"qc contract --json failed (exit {code}): {err.strip()[-300:]}")
        try:
            return one_json_document(out, "qc contract")
        except ToolError as e:
            raise ContractError(str(e))

    def doctor(self) -> Dict[str, Any]:
        argv = ["doctor", "--json"] + (["--workspace", self.workspace] if self.workspace else [])
        code, out, err = self._invoke(argv, timeout=180.0)
        try:
            doc = one_json_document(out, "qc doctor")
        except ToolError as e:
            return {"schema": DOCTOR_SCHEMA, "status": "fail", "problems": [f"doctor produced no document: {e}"], "checks": {}, "exit_code": code}
        if doc.get("schema") != DOCTOR_SCHEMA:
            return {"schema": DOCTOR_SCHEMA, "status": "fail", "problems": [f"unexpected doctor document {doc.get('schema')!r}"], "checks": doc.get("checks") or {}, "exit_code": code}
        status = doc.get("status") if doc.get("status") in ("ok", "degraded", "fail") else {0: "ok", 2: "degraded"}.get(code, "fail")
        doc["status"] = status
        doc["exit_code"] = code
        return doc

    @staticmethod
    def capability_status(doc: Dict[str, Any]) -> Dict[str, str]:
        """Per executable / filter: the status the Skill's doctor reports (AVAILABLE | MISSING | UNKNOWN; never inferred here)."""
        checks = as_dict(doc.get("checks"))
        return {str(k): str(as_dict(v).get("status") or "UNKNOWN") for k, v in checks.items() if k in ("ffmpeg", "ffprobe") or str(k).startswith("filter:")}

    def drift(self) -> List[str]:
        if self._drift is None:
            self._drift = contract_drift(self.contract)
        return self._drift

    # ---- ToolAdapter
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "root": self.skill.describe(), "tools": sorted(self.tools), "kinds": list(self.kinds), "statuses": list(self.statuses),
                "contract": self.contract.get("schema"), "drift": self.drift()}

    def package(self) -> SkillPackage:
        return package_from_contract(self.contract)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            b = self.build_request(op.tool, op.args, paths, op_id=op.id)
        except ToolError as e:
            return [f"qc: refused: {e}"]
        return [" ".join(["qc"] + self._argv(b)) + "  <<< " + json.dumps(b["request"], ensure_ascii=False)[:400]]

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        """QA may measure directly: a measure Operation for qc/check or qc/inspect (the same admission rules apply)."""
        return self.run(Operation(tool=tool, args=args, inputs=[str(args.get("input") or "")], outputs=[], kind="measure"), paths or {}, timeout=timeout)

    # ---- lowering: typed args → one request document
    def rules_for(self, rules: Any) -> Dict[str, Any]:
        """Rule sections and fields exactly as the contract declares them (names, and values of the declared type); nested delivery
        rules are checked against the section they nest."""
        if rules is None:
            return {}
        if not isinstance(rules, dict):
            raise ToolError("qc: rules must be an object of rule sections")
        out: Dict[str, Any] = {}
        for section, fields in rules.items():
            if section not in self.rules:
                raise ToolError(f"qc: unknown rule section {section!r} (contract: {sorted(self.rules)})")
            out[str(section)] = self._rule_section(str(section), fields, f"rules.{section}")
        return out

    def _rule_section(self, section: str, fields: Any, where: str) -> Dict[str, Any]:
        if not isinstance(fields, dict):
            raise ToolError(f"qc: {where} must be an object")
        schema = self.rules[section]
        out: Dict[str, Any] = {}
        for k, v in fields.items():
            if str(k).lower() in {f.lower() for f in self.forbidden}:
                raise ToolError(f"qc: {where}.{k} is not accepted (forbidden field)")
            if k not in schema:
                raise ToolError(f"qc: {where}.{k} is not a declared rule field (contract: {sorted(schema)})")
            out[str(k)] = self._typed_rule(f"{where}.{k}", v, schema[k])
        return out

    def _typed_rule(self, name: str, v: Any, spec: Dict[str, Any]) -> Any:
        t = str(spec.get("type", ""))
        m = _RULE_TYPE_RE.match(t)
        if not m:
            raise ToolError(f"qc: {name} has no typed schema")
        optional, base = bool(m.group(1)), m.group(2)
        if v is None:
            if optional:
                return None
            raise ToolError(f"qc: {name} must be a {base}, not null")
        nested = spec.get("nested_rule")
        if nested is not None:
            return self._rule_section(str(nested), v, name)
        if base == "bool":
            if not isinstance(v, bool):
                raise ToolError(f"qc: {name} must be a boolean")
            return v
        if base == "str":
            if not isinstance(v, str) or "\n" in v or "\x00" in v or len(v) > 256:
                raise ToolError(f"qc: {name} must be a plain string")
            return v
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
            raise ToolError(f"qc: {name} must be a finite number")
        if base == "int":
            if float(v) != int(v):
                raise ToolError(f"qc: {name} must be an integer")
            return int(v)
        return float(v)

    def params_for(self, params: Any) -> Dict[str, Any]:
        if params is None:
            return {}
        if not isinstance(params, dict):
            raise ToolError("qc: parameters must be an object")
        out: Dict[str, Any] = {}
        for k, v in params.items():
            if str(k).lower() in {f.lower() for f in self.forbidden}:
                raise ToolError(f"qc: parameter {k!r} is not accepted (forbidden field)")
            if k not in self.parameters:
                raise ToolError(f"qc: parameter {k!r} is not declared by the contract (contract: {sorted(self.parameters)})")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
                raise ToolError(f"qc: parameter {k} must be a finite number")
            out[str(k)] = v
        return out

    def _resolve_input(self, what: str, ref: Any, paths: Dict[str, str]) -> str:
        rid = str(ref or "")
        if not rid:
            raise ToolError(f"qc: {what} reference is required")
        p = paths.get(rid, rid)
        if not os.path.isfile(p):
            raise ToolError(f"qc: {what} not found: {p}")
        if self.path_policy is not None:
            self.path_policy.check_input(p)
        elif self.allowed_inputs and not any(self._under(p, r) for r in self.allowed_inputs + ([self.workspace] if self.workspace else [])):
            raise ToolError(f"qc: {what} outside the allowed roots: {p}")
        return str(Path(p).resolve())

    def build_request(self, tool: str, args: Dict[str, Any], paths: Dict[str, str], op_id: str = "op", timeout: Optional[float] = None) -> Dict[str, Any]:
        operation = TOOL_OPERATIONS.get(tool)
        if operation is None:
            raise ToolError(f"qc: unsupported tool {tool}")
        hit = scan_forbidden(args, self.forbidden)
        if hit:
            raise ToolError(f"qc: forbidden field {hit} in the operation arguments")
        unknown = sorted(set(args) - set(COMMON_ARGS))
        if unknown:
            raise ToolError(f"qc: unknown arguments {unknown} (accepted: {list(COMMON_ARGS)})")
        kind = str(args.get("kind") or "")
        if kind not in self.kinds:
            raise ToolError(f"qc: kind {kind!r} is not one of {self.kinds}")
        src = self._resolve_input("input", args.get("input"), paths)
        companions: Dict[str, str] = {}
        if args.get("subtitle") is not None:
            if kind != "delivery":
                raise ToolError("qc: a subtitle companion is only accepted with kind delivery")
            companions["subtitle"] = self._resolve_input("subtitle", args["subtitle"], paths)
        if args.get("reference_video") is not None:
            if kind != "subtitle":
                raise ToolError("qc: a reference_video companion is only accepted with kind subtitle")
            companions["reference_video"] = self._resolve_input("reference_video", args["reference_video"], paths)
        rules = self.rules_for(args.get("rules"))
        params = self.params_for(args.get("parameters"))
        if operation == "inspect" and rules:
            raise ToolError("qc: inspect takes no rules (use qc/check)")
        policy = str(args.get("cache_policy") or "use")
        if policy not in AGENT_CACHE_POLICIES or policy not in self.cache_policies:
            raise ToolError(f"qc: cache_policy {policy!r} is not one of {list(AGENT_CACHE_POLICIES)}")
        rid = re.sub(r"[^A-Za-z0-9._-]", "_", str(op_id))[:64] or "op"
        req: Dict[str, Any] = {"schema": REQUEST_SCHEMA, "request_id": rid, "operation": operation, "kind": kind, "input": src, **companions, "cache_policy": policy}
        if rules:
            req["rules"] = rules
        if params:
            req["parameters"] = params
        hit = scan_forbidden(req, self.forbidden, "request")
        if hit:
            raise ToolError(f"qc: refusing to send a request carrying {hit}")
        inputs = [src] + list(companions.values())
        return {"request": req, "input": src, "companions": companions, "operation": operation, "kind": kind, "tool": tool,
                "workspace": self.workspace or str(Path(src).parent), "cache": bool(self.workspace), "input_dirs": sorted({str(Path(p).parent) for p in inputs})}

    @staticmethod
    def _under(path: str, root: str) -> bool:
        try:
            p, r = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
        except OSError:
            return False
        return p == r or p.startswith(r.rstrip(os.sep) + os.sep)

    def _argv(self, b: Dict[str, Any]) -> List[str]:
        argv = ["run", "-", "--json", "--workspace", b["workspace"]]
        roots = list(self.allowed_inputs) + ([self.workspace] if self.workspace else [])
        for r in roots or list(b["input_dirs"]):   # without configured roots the Skill is still confined to the inputs' own directories
            argv += ["--allowed-input-root", r]
        if not b["cache"]:
            argv.append("--no-cache")   # no workspace: nothing may be written next to the input
        return argv

    # ---- execution
    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        try:
            b = self.build_request(op.tool, op.args, paths, op_id=op.id, timeout=timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e), retryable=False)
        if dry_run:   # a measurement has no dry run: the request is validated, the Skill is not invoked
            data = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run", "operation_type": b["operation"], "kind": b["kind"], "verdict": None,
                    "request": b["request"], "commands": [], "warnings": []}
            return ToolResult(op.id, op.tool, True, 0, None, data, [], "", round(time.time() - t0, 3), attempt, True)
        if b["cache"]:
            os.makedirs(b["workspace"], exist_ok=True)
        argv = self._argv(b)
        code, out, err = self._invoke(argv, stdin=json.dumps(b["request"], ensure_ascii=False), timeout=(timeout or self.default_timeout) + 5.0)
        tail = "\n".join(err.strip().splitlines()[-12:])
        secs = round(time.time() - t0, 3)
        if code == 124:
            return self._fail(op, attempt, dry_run, t0, 124, "CANCELLED", tail or "process timed out", retryable=True, details={"reason": "timeout"})
        try:
            doc = one_json_document(out, "qc")
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 9, "INVALID_RESULT", str(e), retryable=False, details={"exit_code": code})
        if doc.get("status") != "completed":
            edoc = as_dict(doc.get("error"))
            errc = str(edoc.get("code") or "INVALID_RESULT")
            if errc not in self.retryable or doc.get("status") != "failed":
                errc, retry = "INVALID_RESULT", False
            else:
                retry = bool(edoc["retryable"]) if isinstance(edoc.get("retryable"), bool) else self.retryable.get(errc, False)
            details = scrub(edoc.get("details"), self.forbidden)
            if errc == "CANCELLED" and (details.get("reason") or "") not in ("timeout", "signal"):
                details["reason"] = details.get("reason") or "signal"
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else self.exit_codes.get(errc, 1), errc, str(edoc.get("message") or tail or f"status {doc.get('status')!r} without an error document")[:500],
                              retryable=bool(retry), details=details)
        errs, fp, comp = self._check_response(doc, b)
        if errs:
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 9, "INVALID_RESULT", "; ".join(errs), retryable=False, details={"exit_code": code, "failed": errs[0]})
        if code != 0:
            return self._fail(op, attempt, dry_run, t0, code, "INVALID_RESULT", f"exit code {code} with a completed response", retryable=False)
        data = self._success_data(doc, b, fp, comp)
        return ToolResult(op.id, op.tool, True, 0, None, data, [], tail, secs, attempt, dry_run)

    def _check_response(self, doc: Dict[str, Any], b: Dict[str, Any]) -> Tuple[List[str], str, Dict[str, Optional[str]]]:
        """Admission: every condition names itself; the first failure is what the failed result reports."""
        errs: List[str] = []
        fp = sha256_file(b["input"])
        comp: Dict[str, Optional[str]] = {k: sha256_file(p) for k, p in b["companions"].items()}
        if doc.get("schema") != RESPONSE_SCHEMA:
            errs.append(f"response schema {doc.get('schema')!r} != {RESPONSE_SCHEMA}")
        sk = as_dict(doc.get("skill"))
        if sk.get("id") != SKILL_ID or str(sk.get("version")) != self.version:
            errs.append(f"response skill {sk!r} is not {SKILL_ID}@{self.version}")
        rep = doc.get("report")
        if not isinstance(rep, dict):
            return errs + ["report missing"], fp, comp
        prov = as_dict(doc.get("provenance"))
        if rep.get("operation") != b["operation"]:
            errs.append(f"report.operation {rep.get('operation')!r} != requested {b['operation']!r}")
        if rep.get("kind") != b["kind"]:
            errs.append(f"report.kind {rep.get('kind')!r} != requested {b['kind']!r}")
        if prov.get("measurement_source") != "OBSERVED":
            errs.append(f"provenance.measurement_source {prov.get('measurement_source')!r} != OBSERVED")
        if prov.get("skill") != SKILL_ID or str(prov.get("skill_version")) != self.version:
            errs.append("provenance skill / skill_version differ from the Skill that answered")
        rfp, pfp = strip_sha_prefix(as_dict(rep.get("input")).get("fingerprint")), strip_sha_prefix(as_dict(prov.get("input")).get("fingerprint"))
        if rfp != fp:
            errs.append(f"report.input.fingerprint {rfp!r} != sha256 of the input {fp}")
        if pfp != fp:
            errs.append(f"provenance.input.fingerprint {pfp!r} != sha256 of the input {fp}")
        for k, digest in comp.items():
            got = strip_sha_prefix(as_dict(prov.get(f"{k}_input")).get("fingerprint"))
            if got != digest:
                errs.append(f"provenance.{k}_input.fingerprint {got!r} != sha256 of the {k} companion {digest}")
        if rep.get("overall_status") not in self.statuses:
            errs.append(f"report.overall_status {rep.get('overall_status')!r} is not one of {self.statuses}")
        checks = rep.get("checks")
        if not isinstance(checks, list):
            errs.append("report.checks is not a list")
        else:
            for c in checks:
                if not isinstance(c, dict) or c.get("status") not in self.statuses:
                    errs.append(f"check {as_dict(c).get('check_id')!r} status {as_dict(c).get('status')!r} is not one of {self.statuses}")
                    break
        if not str(rep.get("id") or "").startswith(REPORT_ID_PREFIX):
            errs.append(f"report.id {rep.get('id')!r} is not a qc report id")
        return errs, fp, comp

    def _success_data(self, doc: Dict[str, Any], b: Dict[str, Any], fp: str, comp: Dict[str, Optional[str]]) -> Dict[str, Any]:
        rep, prov = as_dict(doc.get("report")), as_dict(doc.get("provenance"))
        checks = [{"check_id": c.get("check_id"), "category": c.get("category"), "status": c.get("status"), "finding_codes": list(c.get("finding_codes") or []),
                   "measurement_ids": list(c.get("measurement_ids") or [])} for c in rep.get("checks") or [] if isinstance(c, dict)]
        findings = [scrub(f, self.forbidden) for f in rep.get("findings") or [] if isinstance(f, dict)]
        measurements = [{k: m.get(k) for k in ("id", "category", "name", "value", "unit", "source", "estimated")} for m in rep.get("measurements") or [] if isinstance(m, dict)]
        return {"skill": {"id": SKILL_ID, "version": self.version}, "status": "completed", "operation_type": b["operation"], "kind": b["kind"], "verdict": rep.get("overall_status"),
                "report_id": rep.get("id"), "checks": checks, "findings": findings, "measurements": measurements, "fingerprint": fp,
                "companions": {"subtitle": comp.get("subtitle"), "reference_video": comp.get("reference_video")},
                "provenance": {k: prov.get(k) for k in ("skill", "skill_version", "engine", "identity", "observed_at", "measurement_source")},
                "cache": {k: as_dict(doc.get("cache")).get(k) for k in ("status", "policy", "key")}, "reused": bool(doc.get("reused")), "admitted": True, "commands": [], "warnings": []}

    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, code: int, errc: str, message: str, retryable: bool, details: Optional[Dict[str, Any]] = None) -> ToolResult:
        return failed_result(op, SKILL_ID, getattr(self, "version", ""), attempt, dry_run, t0, code, errc, message, retryable, details)


def lift_report(result: ToolResult, asset_id: Optional[str] = None) -> Optional[Observation]:
    """An admitted qc report as an agent Observation (the verdict is evidence for the QA gate, never a decision)."""
    d = result.data or {}
    if not result.ok or d.get("admitted") is not True or d.get("status") != "completed" or as_dict(d.get("provenance")).get("measurement_source") != "OBSERVED":
        return None
    sk, prov = as_dict(d.get("skill")), as_dict(d.get("provenance"))
    source = f"{result.tool}@{sk.get('version', '')}"
    return Observation(kind="qc.report", asset_id=asset_id or result.op_id, source=source,
                       data={"verdict": d.get("verdict"), "report_id": d.get("report_id"), "checks": list(d.get("checks") or []), "findings": list(d.get("findings") or []),
                             "measurements": list(d.get("measurements") or []), "identity": prov.get("identity")},
                       analyzer=source, provenance="OBSERVED", skill=SKILL_ID, skill_version=str(sk.get("version") or ""), tool=result.tool, external_id=str(d.get("report_id") or ""),
                       fingerprint=str(d.get("fingerprint") or ""), parameters={"kind": d.get("kind"), "operation": d.get("operation_type"), "companions": dict(d.get("companions") or {})},
                       cache=dict(d.get("cache") or {}))
