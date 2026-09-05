"""TranscriptionAdapter: the agent's boundary to transcription-skill (speech recognition Skill, ADR-024).

Protocol (from the Skill's own contract, `transcription skill --json`):
    contract  → `transcription skill --json`                    one JSON document: id, version, capabilities, tools, engines (EngineSpec), schemas
    doctor    → `transcription doctor --json [--offline] [--allowed-input DIR]…`
    request   → `transcription run -`  with {"tool": "<tool id>", "params": {...}} on stdin
    response  ← exactly one JSON document on stdout: {"ok": true, "tool", "result"} or {"ok": false, "error": {code, message, details}}
The contract is the source of truth: tools, engines, execution modes, schema ids, capabilities and versions come from
it and are never re-declared here. The adapter builds requests from typed arguments (asset id, input path, language,
engine, model, word timestamps, offline, decoder parameters, budget, cache policy), pins the workspace and the allowed
input roots itself, and never forwards commands, argv, executable paths, environment or credentials. It never runs an
ASR engine, ffmpeg or ffprobe; the Skill owns recognition, its engines, its models and its transcript cache.

Recognition ends here: a Transcript is a recognition result (facts as recognised), lifted by the analyzer into an
Observation and, from there, into SpeechEvents. Nothing in this module interprets text or decides anything.
"""
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
from .locate import TranscriptionSkill, locate_transcription

CONTRACT_ID = "transcription-skill"       # the Skill's own id (contract `id`, transcript provenance.skill)
SKILL_ID = "transcription"                # package id in the agent's registry == tool id prefix (`transcription/transcribe`)
PREFIX = SKILL_ID + "/"
TRANSCRIBE_TOOL = "transcription/transcribe"
TRANSCRIPT_SCHEMA = "transcription-skill/transcript/0.1"
SPEECH_EVENT_SCHEMA = "transcription-skill/speech-event/0.1"
ENGINE_SPEC_SCHEMA = "transcription-skill/engine-spec/0.1"
SUPPORTED_SKILL_VERSIONS = ("0.2.",)      # 0.2.x: the contract this adapter was verified against
SUPPORTED_SCHEMAS = {"transcript": TRANSCRIPT_SCHEMA, "speech_event": SPEECH_EVENT_SCHEMA, "engine_spec": ENGINE_SPEC_SCHEMA}
REQUIRED_CAPABILITIES = ("speech_recognition", "engine_registry", "input_path_policy", "offline_mode")
# typed request keys the agent may set (a subset of the contract's tool input; the contract is checked to declare them)
PARAMETER_KEYS = ("language", "engine", "model", "word_timestamps", "temperature", "initial_prompt", "beam_size", "offline")
BUDGET_KEYS = ("timeout", "max_audio_seconds")
FORBIDDEN_ARG_KEYS = ("command", "commands", "argv", "cmd", "shell", "exec", "args", "script", "binary", "executable", "executables", "env",
                      "environment", "api_key", "apikey", "token", "secret", "password", "credentials", "ffmpeg", "ffprobe")
ADAPTER_OWNED_KEYS = ("workspace", "allowed_input_roots")   # pinned by the adapter; a caller cannot widen them
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_FP_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_LANG_RE = re.compile(r"^[a-z]{2,3}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ENGINE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class ContractError(ToolError):
    """The installed Skill does not satisfy the contract this adapter was written for."""


def check_contract(contract: Dict[str, Any]) -> List[str]:
    """Compatibility checks: skill id, version range, schema ids, engine contract, tool ownership and shape, capabilities."""
    errs: List[str] = []
    if not isinstance(contract, dict):
        return ["contract is not an object"]
    if contract.get("id") != CONTRACT_ID:
        errs.append(f"skill id {contract.get('id')!r} != {CONTRACT_ID}")
    ver = str(contract.get("version") or "")
    if not ver.startswith(SUPPORTED_SKILL_VERSIONS):
        errs.append(f"skill version {ver!r} not in supported range {SUPPORTED_SKILL_VERSIONS}")
    schemas = contract.get("schemas") or {}
    for k, want in SUPPORTED_SCHEMAS.items():
        if schemas.get(k) != want:
            errs.append(f"schemas.{k}={schemas.get(k)!r} unsupported (expected {want})")
    ec = contract.get("engine_contract") or {}
    if ec.get("schema") != ENGINE_SPEC_SCHEMA:
        errs.append(f"engine_contract.schema {ec.get('schema')!r} != {ENGINE_SPEC_SCHEMA}")
    modes = ec.get("execution_modes") or []
    if "local" not in modes:
        errs.append(f"engine_contract.execution_modes {modes!r} lacks 'local'")
    engines = contract.get("engines")
    if not isinstance(engines, list) or not engines:
        errs.append("no engines declared")
    else:
        for e in engines:
            if not isinstance(e, dict) or not e.get("id") or e.get("execution_mode") not in modes or not isinstance(e.get("requires_network"), bool):
                errs.append(f"engine {e.get('id') if isinstance(e, dict) else e!r}: id / execution_mode / requires_network malformed")
    tools = contract.get("tools") or []
    if not tools:
        errs.append("no tools declared")
    names = []
    for t in tools:
        name = str(t.get("name", ""))
        names.append(name)
        if not name.startswith(PREFIX):
            errs.append(f"tool {name!r} does not belong to {SKILL_ID}")
        if not isinstance(t.get("input"), dict) or not isinstance(t.get("output"), dict):
            errs.append(f"tool {name!r} lacks an input / output schema")
    if TRANSCRIBE_TOOL not in names:
        errs.append(f"{TRANSCRIBE_TOOL} not declared")
    else:
        t = next(x for x in tools if x.get("name") == TRANSCRIBE_TOOL)
        inp = t.get("input") or {}
        for k in ("input", "asset_id", "allowed_input_roots", "workspace", "offline", "cache", "budget") + PARAMETER_KEYS:
            if k not in inp:
                errs.append(f"{TRANSCRIBE_TOOL}.input lacks {k}")
        if "transcript" not in (t.get("output") or {}):
            errs.append(f"{TRANSCRIBE_TOOL}.output lacks transcript")
        for se in t.get("side_effects") or []:
            if "cache" not in str(se):
                errs.append(f"{TRANSCRIBE_TOOL} declares side effect {se!r}: not a recognition-only tool")
    caps = contract.get("capabilities") or []
    for c in REQUIRED_CAPABILITIES:
        if c not in caps:
            errs.append(f"capability {c} not declared")
    return errs


def package_from_contract(contract: Dict[str, Any]) -> SkillPackage:
    ver = str(contract.get("version") or "")
    tools = [ToolSpec(tool_id=t["name"], skill_id=SKILL_ID, version=ver, description=t.get("description", ""), required_capabilities=["ffmpeg", "ffprobe", SKILL_ID],
                      inputs=sorted((t.get("input") or {}).keys()), produces_output=False, deterministic=bool(t.get("deterministic", True)),
                      result_keys=sorted((t.get("output") or {}).keys())) for t in contract.get("tools") or []]
    return SkillPackage(skill_id=SKILL_ID, name=contract.get("name") or CONTRACT_ID, version=ver, description=contract.get("description", ""),
                        capabilities=["ffmpeg", "ffprobe"], tools=tools, repository="kajisho5/transcription-skill", role="speech recognition (recognition only)")


def _one_json_document(stdout: str) -> Dict[str, Any]:
    """Exactly one JSON object on stdout; anything else (empty, text, several documents) is a protocol violation."""
    text = (stdout or "").strip()
    if not text:
        raise ToolError("transcription: empty stdout (expected one response document)")
    try:
        doc, end = json.JSONDecoder().raw_decode(text)
    except ValueError as e:
        raise ToolError(f"transcription: stdout is not JSON: {e}")
    if text[end:].strip():
        raise ToolError("transcription: more than one JSON document on stdout")
    if not isinstance(doc, dict):
        raise ToolError("transcription: response is not an object")
    return doc


PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.2.0.json")


def pinned_contract() -> Dict[str, Any]:
    """The contract this adapter was verified against (snapshot of `transcription skill --json`, 0.2.0, engine availability
    stripped). Used for the package identity when the Skill is not installed; a live installation always replaces it."""
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


PACKAGE = package_from_contract(pinned_contract())


def _has_symlink(path: str) -> bool:
    """Any component of the absolute path is a symlink / junction (the escape case); a mere spelling difference between
    the absolute and the resolved path, e.g. a Windows short name, is not."""
    cur = path
    while True:
        if os.path.islink(cur):
            return True
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            return False
        cur = parent


def _within(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(root), os.path.normcase(path)]) == os.path.normcase(root)
    except ValueError:
        return False


class TranscriptionAdapter(ToolAdapter):
    name = SKILL_ID

    def __init__(self, skill: Optional[TranscriptionSkill] = None, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None,
                 contract: Optional[Dict[str, Any]] = None, timeout: float = 1800.0, offline: bool = False):
        self.skill = skill or locate_transcription()
        if not self.skill:
            raise ToolError("transcription-skill not found (set VIDEO_AGENT_TRANSCRIPTION_DIR or install the `transcription` command)")
        self.workspace = str(Path(workspace).resolve()) if workspace else None          # the Skill's cache / tmp live here, never elsewhere
        self.allowed_inputs = [os.path.realpath(os.path.abspath(p)) for p in (allowed_inputs or [])]
        self.default_timeout = timeout
        self.offline = bool(offline)         # adapter-level hard constraint (CLI --offline); a request may tighten it, never loosen it
        self.calls = 0                       # subprocesses started
        self.contract = contract or self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("transcription contract incompatible: " + "; ".join(errs))
        self.version = str(self.contract["version"])
        self.tools: Dict[str, Dict[str, Any]] = {t["name"]: t for t in self.contract["tools"]}
        self.engines: Dict[str, Dict[str, Any]] = {e["id"]: e for e in self.contract["engines"]}

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
        code, out, err = self._invoke(["skill", "--json"], timeout=60.0)
        if code != 0:
            raise ContractError(f"transcription skill --json failed ({code}): {err.strip()[-300:]}")
        return _one_json_document(out)

    def doctor(self, offline: Optional[bool] = None) -> Dict[str, Any]:
        """The Skill's own doctor (engines, models, ffmpeg, workspace, cache, input path policy). Paths only, never secrets."""
        argv = ["doctor", "--json"]
        if offline if offline is not None else self.offline:
            argv.append("--offline")
        if self.workspace:
            argv += ["--workspace", self.workspace]
        for root in self.allowed_inputs:
            argv += ["--allowed-input", root]
        code, out, err = self._invoke(argv, timeout=120.0)
        doc = _one_json_document(out) if out.strip() else {"ok": False, "checks": [], "summary": err.strip()[-300:]}
        if not isinstance(doc.get("checks"), list):
            raise ContractError("unexpected doctor document (no checks)")
        return doc

    def engine_status(self) -> List[Dict[str, Any]]:
        """EngineSpec rows as the contract reports them: id, version, execution_mode, requires_network, available, capabilities,
        models with their availability. No ranking, no re-interpretation."""
        return [dict(e) for e in self.contract.get("engines") or []]

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "skill": CONTRACT_ID, "version": self.version, "root": self.skill.describe(), "schemas": dict(self.contract.get("schemas") or {}),
                "tools": sorted(self.tools), "engines": sorted(self.engines), "execution_modes": list((self.contract.get("engine_contract") or {}).get("execution_modes") or []),
                "offline": self.offline, "workspace": self.workspace, "allowed_inputs": list(self.allowed_inputs)}

    def package(self) -> SkillPackage:
        return package_from_contract(self.contract)

    def supports(self, tool: str) -> bool:
        return tool == TRANSCRIBE_TOOL and tool in self.tools   # recognition only: segments / export / check are not measurement tools

    owns_cache = True   # the Skill owns the transcript cache (<workspace>/transcripts); the agent records its status as provenance only

    def measurement_args(self, tool: str, kind: str, path: str, asset_id: str, parameters: Dict[str, Any], analysis_id: str, cache_policy: str) -> Optional[Dict[str, Any]]:
        declared = (self.tools.get(tool) or {}).get("input") or {}
        params = {k: v for k, v in (parameters or {}).items() if k in PARAMETER_KEYS and k in declared}
        budget = {k: v for k, v in (parameters or {}).items() if k in BUDGET_KEYS}
        return {"input": path, "asset_id": asset_id, "kind": kind, "parameters": params, "budget": budget, "analysis_id": analysis_id, "cache_policy": cache_policy}

    # ---- input boundary (the Skill enforces it again with the same roots; the adapter refuses early and never widens)
    def check_input(self, raw: str) -> str:
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw or "\n" in raw:
            raise ToolError("transcription: input path must be a non-empty single-line string")
        if self.allowed_inputs:
            if any(part == ".." for part in raw.replace("\\", "/").split("/")):
                raise ToolError("transcription: input path contains '..' (traversal) and allowed roots are enforced")
            absolute = os.path.abspath(raw)
            resolved = os.path.realpath(absolute)
            if not any(_within(root, resolved) for root in self.allowed_inputs):
                reason = "symlink_escape" if _has_symlink(absolute) else "outside_allowed_roots"
                raise ToolError(f"transcription: input is outside the allowed roots ({reason}): {os.path.basename(raw)}")
            return resolved
        return os.path.realpath(os.path.abspath(raw))

    # ---- request construction (typed args → {"tool", "params"}; nothing else crosses the boundary)
    def build_request(self, tool: str, args: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
        if not self.supports(tool):
            raise ToolError(f"transcription: unsupported tool {tool} (recognition only: {TRANSCRIBE_TOOL})")
        for k in args:
            if str(k).lower() in FORBIDDEN_ARG_KEYS:
                raise ToolError(f"transcription: refusing argument {k!r} (commands, argv, executables, environment and credentials never cross the Skill boundary)")
            if str(k) in ADAPTER_OWNED_KEYS:
                raise ToolError(f"transcription: {k!r} is pinned by the adapter (workspace / allowed input roots come from the agent's PathPolicy, not from a request)")
        path = args.get("input") or (args.get("inputs") or [None])[0]
        if not path or not isinstance(path, str):
            raise ToolError("transcription: request needs an input path")
        resolved = self.check_input(path)
        asset_id = str(args.get("asset_id") or "asset")
        if not _ID_RE.match(asset_id):
            raise ToolError(f"transcription: invalid asset id {asset_id!r}")
        params: Dict[str, Any] = {"input": resolved, "asset_id": asset_id}
        given = dict(args.get("parameters") or {})
        given.update({k: v for k, v in args.items() if k in PARAMETER_KEYS})
        declared = (self.tools[tool].get("input") or {})
        for k, v in given.items():
            if k not in PARAMETER_KEYS or k not in declared or v is None:
                continue
            if k == "language":
                v = str(v).strip().lower()
                if v in ("", "auto"):
                    continue
                if not _LANG_RE.match(v):
                    raise ToolError(f"transcription: language must be an ISO 639 code, got {v!r}")
            elif k == "engine":
                if not _ENGINE_RE.match(str(v)):
                    raise ToolError(f"transcription: invalid engine id {v!r}")
                if str(v) not in self.engines:
                    raise ToolError(f"transcription: engine {v!r} is not declared by the Skill's contract ({sorted(self.engines)})")
            elif k == "model":
                if not _MODEL_RE.match(str(v)):
                    raise ToolError(f"transcription: model must be a model name, not a path: {v!r}")
            elif k in ("word_timestamps", "offline"):
                if not isinstance(v, bool):
                    raise ToolError(f"transcription: {k} must be a boolean")
            elif k == "beam_size":
                if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 10:
                    raise ToolError("transcription: beam_size must be an integer in [1, 10]")
            elif k == "temperature":
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= float(v) <= 1.0:
                    raise ToolError("transcription: temperature must be a number in [0, 1]")
            elif k == "initial_prompt":
                if not isinstance(v, str) or "\x00" in v or len(v) > 500:
                    raise ToolError("transcription: initial_prompt must be a short string (vocabulary hint)")
            params[k] = v
        if self.offline:
            params["offline"] = True   # the adapter's constraint can only tighten
        budget = dict(args.get("budget") or {})
        for k in BUDGET_KEYS:
            if k in args:
                budget[k] = args[k]
        budget = {k: float(v) for k, v in budget.items() if k in BUDGET_KEYS and v is not None}
        if budget:
            for k, v in budget.items():
                if v <= 0:
                    raise ToolError(f"transcription: budget.{k} must be positive")
            params["budget"] = budget
        cp = str(args.get("cache_policy") or "use")
        if cp not in ("use", "bypass", "only"):
            raise ToolError(f"transcription: unknown cache policy {cp!r}")
        if cp == "bypass":
            params["cache"] = False
        if self.workspace:
            params["workspace"] = self.workspace
        if self.allowed_inputs:
            params["allowed_input_roots"] = list(self.allowed_inputs)
        if dry_run:
            params["dry_run"] = True
        return {"tool": tool, "params": params}

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        req = self.build_request(op.tool, op.args)
        return [" ".join(list(self.skill.command) + ["run", "-"]) + "  <<< " + json.dumps(req, sort_keys=True, ensure_ascii=False)]

    # ---- execution: one subprocess per call; the response document is mapped to a ToolResult
    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        try:
            req = self.build_request(op.tool, op.args, dry_run=dry_run)
        except ToolError as e:
            return ToolResult(op.id, op.tool, False, 2, None, {"error": {"code": "INVALID_INPUT", "message": str(e)}}, [], str(e), 0.0, attempt, dry_run)
        cache_policy = str(op.args.get("cache_policy") or "use")
        if cache_policy == "only" and not dry_run:
            # the Skill has no cache-only mode: ask what would run; without a cached transcript nothing is recognised
            probe = self._call(dict(req, params=dict(req["params"], dry_run=True)), timeout, op, attempt, True, t0)
            status = ((probe.data or {}).get("result") or {}).get("cache", {}).get("status") if probe.ok else None
            if status != "hit":
                msg = "CACHED_ONLY: no cached transcript for this request" if probe.ok else (probe.data or {}).get("error", {}).get("message", "")
                return ToolResult(op.id, op.tool, False, 1, None, {"error": {"code": "CACHE_MISS", "message": msg}}, [], probe.stderr_tail, round(time.time() - t0, 3), attempt, dry_run)
        return self._call(req, timeout, op, attempt, dry_run, t0)

    def _call(self, req: Dict[str, Any], timeout: Optional[float], op: Operation, attempt: int, dry_run: bool, t0: float) -> ToolResult:
        budget = (req["params"].get("budget") or {}).get("timeout")
        code, out, err = self._invoke(["run", "-"], stdin=json.dumps(req, ensure_ascii=False), timeout=timeout or (float(budget) + 60.0 if budget else self.default_timeout))
        tail = "\n".join(err.strip().splitlines()[-12:])
        secs = round(time.time() - t0, 3)
        if code == 124:
            return ToolResult(op.id, op.tool, False, 124, None, {"error": {"code": "TRANSCRIPTION_TIMEOUT", "message": tail or "process timed out"}}, [], tail, secs, attempt, dry_run)
        try:
            doc = _one_json_document(out)
        except ToolError as e:
            return ToolResult(op.id, op.tool, False, code or 9, None, {"error": {"code": "INVALID_RESULT", "message": str(e), "exit_code": code}}, [], tail, secs, attempt, dry_run)
        if doc.get("ok") is not True:
            err_doc = doc.get("error") if isinstance(doc.get("error"), dict) else {}
            errc = str(err_doc.get("code") or "INVALID_RESULT")
            data = {"error": {"code": errc, "message": str(err_doc.get("message") or tail or "")[:500], "details": _scrub_details(err_doc.get("details")), "exit_code": code},
                    "skill": {"id": CONTRACT_ID, "version": self.version}}
            return ToolResult(op.id, op.tool, False, code or 1, None, data, [], tail, secs, attempt, dry_run)
        errs = self._check_response(doc, req)
        if errs:
            return ToolResult(op.id, op.tool, False, code or 9, None, {"error": {"code": "INVALID_RESULT", "message": "; ".join(errs), "exit_code": code}}, [], tail, secs, attempt, dry_run)
        result = doc["result"]
        data: Dict[str, Any] = {"result": result, "skill": {"id": CONTRACT_ID, "version": self.version}, "dry_run": bool(result.get("dry_run")), "warnings": list(result.get("warnings") or [])}
        if result.get("dry_run"):
            data["cache"] = dict(result.get("cache") or {}, owner=SKILL_ID)
            return ToolResult(op.id, op.tool, code == 0, code, None, data, [], tail, secs, attempt, True)
        tr = result["transcript"]
        data["transcript"] = tr
        data["cache"] = {"status": "hit" if result.get("cache_hit") else "miss", "key": result.get("cache_key"), "owner": SKILL_ID}
        if result.get("cache_hit") and tr.get("asset_id") != req["params"].get("asset_id"):
            data["cache"]["stored_asset_id"] = tr.get("asset_id")   # the first caller's id; identity = fingerprint (checked by the analyzer)
        data["engine"] = {"id": tr.get("engine"), "version": tr.get("engine_version"), "execution_mode": (tr.get("provenance") or {}).get("execution_mode"),
                          "model": (tr.get("provenance") or {}).get("model"), "model_version": (tr.get("provenance") or {}).get("model_version")}
        ops = [f"{tr['engine']}@{tr['engine_version']}: recognition ({(tr.get('provenance') or {}).get('execution_mode')})"]
        return ToolResult(op.id, op.tool, code == 0, code, None, data, ops, tail, secs, attempt, dry_run)

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        return self.run(Operation(tool=tool, args=args, inputs=[], outputs=[], kind="measure"), paths or {}, timeout=timeout)

    # ---- response validation: the document must be the Skill's, about this asset, produced by a declared engine, and a recognition fact
    def _check_response(self, doc: Dict[str, Any], req: Dict[str, Any]) -> List[str]:
        errs: List[str] = []
        if doc.get("tool") != req["tool"]:
            errs.append(f"response tool {doc.get('tool')!r} != {req['tool']}")
        result = doc.get("result")
        if not isinstance(result, dict):
            return errs + ["response carries no result object"]
        if result.get("dry_run"):
            if not isinstance(result.get("cache"), dict) or not isinstance(result.get("engine"), dict):
                errs.append("dry-run result lacks cache / engine")
            return errs
        tr = result.get("transcript")
        if not isinstance(tr, dict):
            return errs + ["ok result without transcript"]
        if not isinstance(result.get("cache_hit"), bool) or not result.get("cache_key"):
            errs.append("result lacks cache_hit / cache_key")
        # a cache hit returns the stored document unchanged (the Skill's contract): its asset_id is whatever the first caller
        # stamped; identity is then the content fingerprint, which the analyzer checks against the asset's own hash
        errs += check_transcript(tr, req["params"], self.version, self.engines, dict((self.contract.get("schemas") or {})), cached=result.get("cache_hit") is True)
        return errs


def check_transcript(tr: Dict[str, Any], params: Dict[str, Any], skill_version: str, engines: Dict[str, Dict[str, Any]], schemas: Dict[str, str], cached: bool = False) -> List[str]:
    """A Transcript document is accepted only when its identity, provenance and structure are exactly what the request and
    the contract say. Partial or inconsistent documents are refused as a whole (no partial transcript is ever a success)."""
    errs: List[str] = []
    if tr.get("schema") != schemas.get("transcript", TRANSCRIPT_SCHEMA):
        errs.append(f"transcript schema {tr.get('schema')!r} != {schemas.get('transcript', TRANSCRIPT_SCHEMA)}")
    for k in ("id", "asset_id", "language", "language_source", "duration", "segments", "source", "engine", "engine_version", "created_at", "provenance"):
        if k not in tr:
            errs.append(f"transcript misses {k}")
    if errs:
        return errs
    if tr["asset_id"] != params.get("asset_id") and not cached:
        errs.append(f"transcript asset {tr['asset_id']!r} != requested {params.get('asset_id')!r}")
    src = tr.get("source") or {}
    fp = str(src.get("fingerprint") or "")
    if not _FP_RE.match(fp):
        errs.append(f"transcript source.fingerprint {fp!r} is not sha256:<64 hex>")
    if not src.get("filename") or "/" in str(src.get("filename")) or "\\" in str(src.get("filename")):
        errs.append("transcript source.filename must be a bare file name")
    eng = str(tr.get("engine") or "")
    if eng not in engines:
        errs.append(f"transcript engine {eng!r} is not declared by the contract ({sorted(engines)})")
    if params.get("engine") and eng != params["engine"]:
        errs.append(f"transcript engine {eng!r} != requested {params['engine']!r}")
    if not tr.get("engine_version"):
        errs.append("transcript engine_version missing")
    prov = tr.get("provenance") or {}
    if not isinstance(prov, dict):
        return errs + ["transcript provenance is not an object"]
    if prov.get("skill") != CONTRACT_ID:
        errs.append(f"provenance.skill {prov.get('skill')!r} != {CONTRACT_ID}")
    if prov.get("tool") != TRANSCRIBE_TOOL:
        errs.append(f"provenance.tool {prov.get('tool')!r} != {TRANSCRIBE_TOOL}")
    if str(prov.get("skill_version") or "") != skill_version:
        errs.append(f"provenance.skill_version {prov.get('skill_version')!r} != contract {skill_version}")
    mode = prov.get("execution_mode")
    if eng in engines and mode != engines[eng].get("execution_mode"):
        errs.append(f"provenance.execution_mode {mode!r} != engine spec {engines.get(eng, {}).get('execution_mode')!r}")
    if prov.get("engine") != eng or prov.get("engine_version") != tr.get("engine_version"):
        errs.append("provenance engine identity differs from the transcript's")
    if not prov.get("model"):
        errs.append("provenance.model missing")
    if params.get("model") and prov.get("model") != params["model"]:
        errs.append(f"provenance.model {prov.get('model')!r} != requested {params['model']!r}")
    if not isinstance(prov.get("parameters"), dict) or not prov.get("cache_key"):
        errs.append("provenance.parameters / cache_key missing")
    if params.get("language") and tr.get("language") != params["language"]:
        errs.append(f"transcript language {tr.get('language')!r} != requested {params['language']!r}")
    if tr.get("language_source") not in ("requested", "detected", "unknown"):
        errs.append(f"language_source {tr.get('language_source')!r} invalid")
    segs = tr.get("segments")
    if not isinstance(segs, list):
        return errs + ["segments is not a list"]
    prev_end = -1.0
    for s in segs:
        if not isinstance(s, dict) or not all(k in s for k in ("id", "start", "end", "text")):
            errs.append("segment lacks id / start / end / text")
            break
        try:
            st, en = float(s["start"]), float(s["end"])
        except (TypeError, ValueError):
            errs.append(f"segment {s.get('id')} has non-numeric timestamps")
            break
        if not (0.0 <= st < en) or st + 0.01 < prev_end:
            errs.append(f"segment {s.get('id')} timestamps invalid or out of order ({st}, {en})")
            break
        prev_end = en
        if "speaker_id" in s and s["speaker_id"] is not None:
            errs.append(f"segment {s.get('id')} carries speaker_id {s['speaker_id']!r}: recognition never identifies speakers")
            break
        if not isinstance(s.get("text"), str):
            errs.append(f"segment {s.get('id')} text is not a string")
            break
    try:
        if float(tr.get("duration")) <= 0:
            errs.append("transcript duration must be positive")
    except (TypeError, ValueError):
        errs.append("transcript duration is not a number")
    return errs


def _scrub_details(details: Any) -> Dict[str, Any]:
    if not isinstance(details, dict):
        return {}
    bad = re.compile(r"(api[_-]?key|secret|token|password|credential|authorization|argv|command|cmd|shell)", re.I)
    return {k: v for k, v in details.items() if not bad.search(str(k)) and isinstance(v, (str, int, float, bool, list, type(None)))}
