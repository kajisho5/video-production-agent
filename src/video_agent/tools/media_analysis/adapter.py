"""MediaAnalysisAdapter: the agent's boundary to media-analysis-skill (measurement / observation Skill, ADR-023).

Protocol (from the Skill's own contract, `media-analysis contract --json`, schema media-analysis/contract@1):
    request  → `media-analysis run - --json`  with an AnalysisRequest (or batch) JSON document on stdin
    response ← exactly one response document (media-analysis/response@1) on stdout; stderr is diagnostics only
The contract is the source of truth: tools, analysis kinds, kind → tool, capabilities, versions and schemas come from
it and are never re-declared here. The adapter builds requests from the agent's typed arguments (asset id, input path,
kind, parameters, timeout, cache policy) and never forwards commands, argv, executable paths or credentials. It never
runs ffprobe / ffmpeg itself; the Skill owns its analyzers and its cache."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...models import Operation, ToolResult
from ...skills.contract import SkillPackage, ToolSpec
from ..base import ToolAdapter, ToolError
from ..ffmpeg_skill.adapter import run_process_group
from .locate import MediaAnalysisSkill, locate_media_analysis

SKILL_ID = "media-analysis"
PREFIX = SKILL_ID + "/"
CONTRACT_SCHEMA = "media-analysis/contract@1"
RESPONSE_SCHEMA = "media-analysis/response@1"
DOCTOR_SCHEMA = "media-analysis/doctor@1"
SUPPORTED_CONTRACT_VERSIONS = ("1",)
SUPPORTED_SKILL_VERSIONS = ("0.1.",)          # 0.1.x: the contract this adapter was verified against
REQUEST_KEYS = ("analysis_id", "asset_id", "input", "kind", "parameters", "timeout", "cache_policy", "output_policy")
FORBIDDEN_ARG_KEYS = ("command", "commands", "argv", "shell", "executable", "executables", "ffmpeg", "ffprobe", "api_key", "token", "secret")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ContractError(ToolError):
    """The installed Skill does not satisfy the contract this adapter was written for."""


def check_contract(contract: Dict[str, Any]) -> List[str]:
    """Compatibility checks (skill id, contract / schema versions, tools, kinds, capabilities, invocation, execution mode, provenance)."""
    errs: List[str] = []
    if contract.get("schema") != CONTRACT_SCHEMA:
        errs.append(f"contract schema {contract.get('schema')!r} != {CONTRACT_SCHEMA}")
    if contract.get("skill_id") != SKILL_ID:
        errs.append(f"skill_id {contract.get('skill_id')!r} != {SKILL_ID}")
    ver = str(contract.get("version") or "")
    if not ver.startswith(SUPPORTED_SKILL_VERSIONS):
        errs.append(f"skill version {ver!r} not in supported range {SUPPORTED_SKILL_VERSIONS}")
    sv = contract.get("schema_versions") or {}
    for k in ("contract", "request", "response", "observation"):
        if str(sv.get(k)) not in SUPPORTED_CONTRACT_VERSIONS:
            errs.append(f"schema_versions.{k}={sv.get(k)!r} unsupported")
    ex = contract.get("execution") or {}
    if ex.get("mode") != "local_subprocess":
        errs.append(f"execution.mode {ex.get('mode')!r} != local_subprocess")
    inv = ex.get("canonical_invocation") or []
    if not (len(inv) >= 2 and inv[1] == "run" and "--json" in inv):
        errs.append(f"canonical_invocation {inv!r} is not `run … --json`")
    if ex.get("ai") or ex.get("network") or ex.get("media_processing"):
        errs.append("execution declares ai / network / media_processing: not an observation Skill")
    tools = contract.get("tools") or []
    if not tools:
        errs.append("no tools declared")
    for t in tools:
        if not str(t.get("tool_id", "")).startswith(PREFIX) or t.get("skill_id") != SKILL_ID:
            errs.append(f"tool {t.get('tool_id')!r} does not belong to {SKILL_ID}")
        if t.get("produces_output") or t.get("writes_media"):
            errs.append(f"tool {t.get('tool_id')} writes output / media: not a measurement tool")
    k2t = contract.get("kind_to_tool") or {}
    kinds = contract.get("analysis_kinds") or []
    if not kinds or set(kinds) != set(k2t):
        errs.append("analysis_kinds and kind_to_tool disagree")
    tool_ids = {t.get("tool_id") for t in tools}
    for k, tool in k2t.items():
        if tool not in tool_ids:
            errs.append(f"kind {k} maps to unknown tool {tool}")
    if contract.get("provenance") != "OBSERVED":
        errs.append(f"provenance {contract.get('provenance')!r} != OBSERVED")
    if not contract.get("capability_names"):
        errs.append("no capability names declared")
    return errs


def package_from_contract(contract: Dict[str, Any]) -> SkillPackage:
    tools = [ToolSpec(tool_id=t["tool_id"], skill_id=SKILL_ID, version=str(t.get("version") or contract.get("version") or ""), description=t.get("description", ""),
                      required_capabilities=list(t.get("required_capabilities") or []), inputs=list(t.get("inputs") or []), produces_output=False, deterministic=bool(t.get("deterministic", True)),
                      result_keys=list(t.get("result_keys") or [])) for t in contract.get("tools") or []]
    return SkillPackage(skill_id=SKILL_ID, name=contract.get("name") or SKILL_ID, version=str(contract.get("version") or ""), description=contract.get("description", ""),
                        capabilities=list(contract.get("capabilities") or []), tools=tools, repository=contract.get("repository", ""), role=contract.get("role", "observation / analysis"))


def _one_json_document(stdout: str) -> Dict[str, Any]:
    """Exactly one JSON object on stdout; anything else (empty, text, several documents) is a protocol violation."""
    text = (stdout or "").strip()
    if not text:
        raise ToolError("media-analysis: empty stdout (expected one response document)")
    try:
        doc, end = json.JSONDecoder().raw_decode(text)
    except ValueError as e:
        raise ToolError(f"media-analysis: stdout is not JSON: {e}")
    if text[end:].strip():
        raise ToolError("media-analysis: more than one JSON document on stdout")
    if not isinstance(doc, dict):
        raise ToolError("media-analysis: response is not an object")
    return doc


PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    """The contract this adapter was verified against (snapshot of `media-analysis contract --json`, 0.1.0). Used for the
    package identity when the Skill is not installed; a live installation always replaces it after check_contract."""
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


PACKAGE = package_from_contract(pinned_contract())


class MediaAnalysisAdapter(ToolAdapter):
    name = SKILL_ID

    def __init__(self, skill: Optional[MediaAnalysisSkill] = None, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None,
                 cache_dir: Optional[str] = None, contract: Optional[Dict[str, Any]] = None, timeout: float = 60.0):
        self.skill = skill or locate_media_analysis()
        if not self.skill:
            raise ToolError(f"media-analysis-skill not found (set VIDEO_AGENT_MEDIA_ANALYSIS_DIR or install the `media-analysis` command)")
        self.workspace = str(Path(workspace).resolve()) if workspace else None
        self.allowed_inputs = [str(Path(p).resolve()) for p in (allowed_inputs or [])]
        self.cache_dir = cache_dir
        self.default_timeout = timeout
        self.calls = 0   # subprocesses started (never more than one per measure / batch)
        self.contract = contract or self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("media-analysis contract incompatible: " + "; ".join(errs))
        self.version = str(self.contract["version"])
        self.tools: Dict[str, Dict[str, Any]] = {t["tool_id"]: t for t in self.contract["tools"]}
        self.kind_to_tool: Dict[str, str] = dict(self.contract["kind_to_tool"])

    # ---- discovery
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
        code, out, err = self._invoke(["contract", "--json"])
        if code != 0:
            raise ContractError(f"media-analysis contract --json failed ({code}): {err.strip()[-300:]}")
        return _one_json_document(out)

    def doctor(self) -> Dict[str, Any]:
        code, out, err = self._invoke(["doctor", "--json"])
        doc = _one_json_document(out) if out.strip() else {"schema": DOCTOR_SCHEMA, "status": "error", "checks": {}, "stderr": err.strip()[-300:]}
        if doc.get("schema") != DOCTOR_SCHEMA:
            raise ContractError(f"unexpected doctor schema {doc.get('schema')!r}")
        return doc

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "root": self.skill.describe(), "contract": self.contract.get("schema"), "tools": sorted(self.tools),
                "kinds": sorted(self.kind_to_tool), "execution": self.contract.get("execution", {}).get("mode")}

    def package(self) -> SkillPackage:
        return package_from_contract(self.contract)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    owns_cache = True   # the Skill owns its cache (`--cache-dir`); the agent records cache status as provenance only

    def measurement_args(self, tool: str, kind: str, path: str, asset_id: str, parameters: Dict[str, Any], analysis_id: str, cache_policy: str) -> Optional[Dict[str, Any]]:
        allowed = ((self.tools.get(tool) or {}).get("parameters") or {}).get(kind) or {}
        return {"input": path, "asset_id": asset_id, "kind": kind, "parameters": {k: v for k, v in (parameters or {}).items() if k in allowed},
                "analysis_id": analysis_id, "cache_policy": cache_policy}

    def kinds_of(self, tool: str) -> List[str]:
        return list((self.tools.get(tool) or {}).get("kinds") or [])

    # ---- request construction (typed args → AnalysisRequest; nothing else crosses the boundary)
    def build_request(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool not in self.tools:
            raise ToolError(f"media-analysis: unknown tool {tool}")
        for k in args:
            if k.lower() in FORBIDDEN_ARG_KEYS:
                raise ToolError(f"media-analysis: refusing argument {k!r} (commands, argv, executables and credentials never cross the Skill boundary)")
        kinds = self.kinds_of(tool)
        kind = args.get("kind") or (kinds[0] if len(kinds) == 1 else None)
        if kind not in kinds:
            raise ToolError(f"media-analysis: tool {tool} needs an explicit kind from {kinds}, got {kind!r}")
        path = args.get("input") or (args.get("inputs") or [None])[0]
        if not path or not isinstance(path, str):
            raise ToolError("media-analysis: request needs an input path")
        asset_id = str(args.get("asset_id") or "asset")
        if not _ID_RE.match(asset_id):
            raise ToolError(f"media-analysis: invalid asset id {asset_id!r}")
        req: Dict[str, Any] = {"asset_id": asset_id, "input": str(Path(path).resolve()), "kind": kind}
        if args.get("analysis_id"):
            req["analysis_id"] = str(args["analysis_id"])
        params = args.get("parameters") or {k: v for k, v in args.items() if k not in ("input", "inputs", "asset_id", "analysis_id", "kind", "timeout", "cache_policy", "parameters")}
        if params:
            req["parameters"] = params
        if args.get("timeout"):
            req["timeout"] = float(args["timeout"])
        if args.get("cache_policy"):
            req["cache_policy"] = str(args["cache_policy"])
        return req

    def _engine_argv(self, dry_run: bool = False) -> List[str]:
        argv = ["run", "-", "--json"]
        if dry_run:
            argv.append("--dry-run")
        if self.cache_dir:
            argv += ["--cache-dir", self.cache_dir]
        if self.workspace:
            argv += ["--workspace", self.workspace]
        for root in self.allowed_inputs:
            argv += ["--allowed-input", root]
        return argv

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        req = self.build_request(op.tool, op.args)
        return [" ".join(list(self.skill.command) + self._engine_argv()) + "  <<< " + json.dumps(req, sort_keys=True)]

    # ---- execution: one subprocess per call, response document mapped to ToolResult
    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        req = self.build_request(op.tool, op.args)
        t0 = time.time()
        code, out, err = self._invoke(self._engine_argv(dry_run), stdin=json.dumps(req), timeout=timeout or float(req.get("timeout") or self.default_timeout))
        tail = "\n".join(err.strip().splitlines()[-12:])
        if code == 124:
            return ToolResult(op.id, op.tool, False, 124, None, {"error": {"code": "ANALYZER_TIMEOUT", "message": tail}}, [], tail, round(time.time() - t0, 3), attempt, dry_run)
        try:
            doc = _one_json_document(out)
        except ToolError as e:
            return ToolResult(op.id, op.tool, False, code or 9, None, {"error": {"code": "INVALID_RESULT", "message": str(e)}}, [], tail, round(time.time() - t0, 3), attempt, dry_run)
        errs = self._check_response(doc, req)
        if errs:
            return ToolResult(op.id, op.tool, False, code or 9, None, {"error": {"code": "INVALID_RESULT", "message": "; ".join(errs)}, "response": doc}, [], tail, round(time.time() - t0, 3), attempt, dry_run)
        result = doc["results"][0]
        ok = result.get("status") == "ok" and code == 0
        data = {"result": result, "observation": result.get("observation"), "cache": result.get("cache"), "usage": result.get("usage"), "budget": doc.get("budget"),
                "skill": doc.get("skill"), "dry_run": bool(doc.get("dry_run")), "warnings": doc.get("warnings") or []}
        if not ok:
            data["error"] = result.get("error") or doc.get("error") or {"code": "ANALYSIS_FAILED", "message": tail}
            data["error_kind"] = result.get("error_kind") or doc.get("error_kind") or data["error"].get("code")
        ops = [f"{o.get('executable')}: {o.get('purpose')}" for o in (result.get("usage") or {}).get("operations") or []]
        return ToolResult(op.id, op.tool, ok, code, None, data, ops, tail, round(time.time() - t0, 3), attempt, dry_run)

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        return self.run(Operation(tool=tool, args=args, inputs=[], outputs=[], kind="measure"), paths or {}, timeout=timeout)

    def _check_response(self, doc: Dict[str, Any], req: Dict[str, Any]) -> List[str]:
        errs: List[str] = []
        if doc.get("schema") != RESPONSE_SCHEMA:
            errs.append(f"response schema {doc.get('schema')!r} != {RESPONSE_SCHEMA}")
        sk = doc.get("skill") or {}
        if sk.get("id") != SKILL_ID:
            errs.append(f"response skill {sk.get('id')!r} != {SKILL_ID}")
        if str(sk.get("version") or "") != self.version:
            errs.append(f"response skill version {sk.get('version')!r} != contract {self.version}")
        results = doc.get("results")
        if not isinstance(results, list) or len(results) != 1:
            errs.append("response must carry exactly one result for a single request")
            return errs
        r = results[0]
        if r.get("kind") != req["kind"] or r.get("asset_id") != req["asset_id"]:
            errs.append(f"result kind / asset ({r.get('kind')}, {r.get('asset_id')}) do not match the request ({req['kind']}, {req['asset_id']})")
        if r.get("status") == "ok":
            o = r.get("observation")
            if not isinstance(o, dict):
                errs.append("ok result without observation")
            else:
                for k in ("id", "asset_id", "kind", "data", "source", "analysis_id", "observed_at", "analysis", "asset"):
                    if k not in o:
                        errs.append(f"observation misses {k}")
                if o.get("kind") != req["kind"] or o.get("asset_id") != req["asset_id"]:
                    errs.append("observation kind / asset do not match the request")
                src = str(o.get("source") or "")
                if not src.startswith(PREFIX) or "@" not in src:
                    errs.append(f"observation source {src!r} is not media-analysis/<tool>@<version>")
                if not isinstance(o.get("data"), dict):
                    errs.append("observation data is not an object")
        return errs
