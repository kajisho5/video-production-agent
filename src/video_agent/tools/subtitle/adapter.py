"""SubtitleAdapter: the agent's boundary to subtitle-skill (deterministic subtitle file generation and burn-in, ADR-031).

    video-production-agent ─(typed Operation)─→ SubtitleAdapter ─(subtitle request on stdin)─→ `subtitle-skill run - --json`
        ─(render only: typed ffmpeg-skill/caption call)─→ ffmpeg-skill ─→ FFmpeg

Protocol (from the Skill's own contract, `subtitle-skill contract --json`):
    contract → `subtitle-skill contract --json`
    doctor   → `subtitle-skill doctor --json` (exit 1 when healthy is false)
    run      → `subtitle-skill run - --json` (request on stdin; no workspace / allowed-input / ffmpeg-skill / timeout flags exist)
    response ← one JSON document: {"status": "ok", "skill", "skill_version", "contract_version", "operation", "output", "sha256", "size", "reused",
               "observation", "timeline", "duration_ms", "engine"?, "engine_version"?} or {"status": "error", "error": {code, message, retryable, details?}}

The agent hands one typed operation: {"operation": generate | render, "format": srt | vtt, "document_id", "language", "cues": [{id, start, end, text}],
"constraints"?, "video_duration"?, "output": <artifact id>, and for render "input": <video artifact id>, "sidecar"?: <artifact id of the generated file>}.
Cues, constraints and formats are validated here against the pinned contract before anything is sent; forbidden keys are refused by name.
The request carries the workspace (absolute) and workspace-relative output / video paths only. The response is verified: skill / version /
contract version / status / operation, the output's realpath, sha256 recomputed from the file, size, cue count and (render) the engine identity.
The agent never identifies speakers (`speaker` is never set) and never sends the sidecar file: the Skill regenerates the subtitle text
from the typed document, so the sidecar is only checked to exist under an allowed root for the planner's bookkeeping.

Contract discrepancies absorbed here (documented, never patched in either project):
    * the contract declares no tool ids and no schema id: the agent defines `subtitle/generate` and `subtitle/render` itself; the agent
      package id is "subtitle" while the contract's skill_id is "subtitle-skill" (check_contract expects exactly that string);
    * the contract lists no forbidden request keys: the ten names the Skill's security.py rejects are hard-coded below (SKILL_FORBIDDEN_KEYS)
      and merged with the agent's own vocabulary;
    * the Skill takes the workspace from the request body and the ffmpeg-skill location only from the environment
      (SUBTITLE_SKILL_FFMPEG_SKILL_DIR): the adapter passes that directory through `invoke(extra_env=…)` — a location, not a credential;
    * the CLI has no --timeout: the process boundary (timeout + 5 s, exit 124) is the only limit;
    * the Skill has no dry-run: `run(dry_run=True)` validates and lowers the request, then returns without invoking the process;
    * the Skill reports no probe of a rendered output; the agent's QA probes it itself, so data["observation"] is None on render."""
from __future__ import annotations

import json
import math
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
from .locate import locate_subtitle

SKILL_ID = "subtitle"
PREFIX = SKILL_ID + "/"
CONTRACT_SKILL_ID = "subtitle-skill"          # the Skill's own id (the agent package id is "subtitle")
TOOL_GENERATE = "subtitle/generate"
TOOL_RENDER = "subtitle/render"
CONTRACT_VERSION = "1.0.0"
SUPPORTED_SKILL_VERSIONS = ("0.1.",)
ENGINE_ID = "ffmpeg-skill"
ENGINE_DIR_ENV = "SUBTITLE_SKILL_FFMPEG_SKILL_DIR"   # the only way the Skill learns where ffmpeg-skill is (a location, not a credential)
OPERATIONS = ("generate", "render")
# subtitle-skill's security.py FORBIDDEN_KEYS (the contract JSON does not list them); refused anywhere in a request, by name
SKILL_FORBIDDEN_KEYS = ("command", "argv", "shell", "executable", "filter", "filter_complex", "vf", "af", "env", "api_key")
REQUIRED_ERROR_CODES = ("CANCELLED", "DEPENDENCY_ERROR", "INTERNAL_ERROR", "INVALID_INPUT", "INVALID_REQUEST", "INVALID_TIME_RANGE", "MISSING_INPUT", "OUTPUT_ERROR",
                        "PATH_NOT_ALLOWED", "TOOL_ERROR", "UNSUPPORTED_FORMAT", "UNSUPPORTED_OPERATION", "VALIDATION_ERROR")
REQUIRED_OUT_OF_SCOPE = ("speech_recognition", "transcription", "speaker_diarization", "ai_editing_decisions", "arbitrary_ffmpeg_execution")
COMMON_ARGS = ("operation", "format", "document_id", "language", "cues", "constraints", "video_duration", "output")
RENDER_ARGS = COMMON_ARGS + ("input", "sidecar")
CUE_KEYS = ("id", "start", "end", "text")
INTEGER_CONSTRAINTS = ("max_chars_per_line", "max_lines")
MAX_CUE_TEXT = 2000
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_LANG_RE = re.compile(r"^[A-Za-z]{2,8}(-[A-Za-z0-9]{1,8})*$")
DRIFT_KEYS = ("skill_id", "version", "contract_version", "deterministic", "capabilities", "operations", "out_of_scope", "parameters", "errors")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def check_contract(contract: Any) -> List[str]:
    """Compatibility checks: skill id, contract version, version range, determinism, the two operations and their formats, the render
    delegation to ffmpeg-skill/caption, the declared out-of-scope list, the constraint parameters and the error table. Anything off is
    refused, never patched."""
    errs: List[str] = []
    if not isinstance(contract, dict):
        return ["contract is not an object"]
    if contract.get("skill_id") != CONTRACT_SKILL_ID:
        errs.append(f"skill_id {contract.get('skill_id')!r} != {CONTRACT_SKILL_ID}")
    if str(contract.get("contract_version")) != CONTRACT_VERSION:
        errs.append(f"contract_version {contract.get('contract_version')!r} != {CONTRACT_VERSION}")
    ver = str(contract.get("version") or "")
    if not ver.startswith(SUPPORTED_SKILL_VERSIONS):
        errs.append(f"skill version {ver!r} not in supported range {SUPPORTED_SKILL_VERSIONS}")
    if contract.get("deterministic") is not True:
        errs.append(f"deterministic must be true, contract says {contract.get('deterministic')!r}")
    ops = as_dict(contract.get("operations"))
    gen, ren = as_dict(ops.get("generate")), as_dict(ops.get("render"))
    if not gen or not ren:
        errs.append(f"operations must declare generate and render, contract declares {sorted(ops)}")
    if not {"srt", "vtt"} <= {str(f) for f in gen.get("formats") or []}:
        errs.append(f"generate formats {gen.get('formats')!r} lack srt / vtt")
    if [str(f) for f in ren.get("formats") or []] != ["srt"]:
        errs.append(f"render formats {ren.get('formats')!r} != ['srt']")
    dl = as_dict(ren.get("delegates_to"))
    if dl.get("skill_id") != ENGINE_ID or dl.get("tool") != "caption":
        errs.append(f"render must delegate to {ENGINE_ID}/caption, contract says {dl!r}")
    for o in (gen, ren):
        if not isinstance(o.get("inputs"), list) or not isinstance(o.get("outputs"), list):
            errs.append("operation inputs / outputs missing")
    oos = {str(x) for x in contract.get("out_of_scope") or []}
    missing = [x for x in REQUIRED_OUT_OF_SCOPE if x not in oos]
    if missing:
        errs.append(f"out_of_scope lacks {missing}")
    params = as_dict(contract.get("parameters"))
    if not isinstance(params.get("constraints"), list) or not params.get("constraints"):
        errs.append("parameters.constraints missing")
    retry, _ = error_table(contract)
    for c in REQUIRED_ERROR_CODES:
        if c not in retry:
            errs.append(f"errors.codes lacks {c}")
    if not isinstance(as_dict(contract.get("errors")).get("non_retryable"), list):
        errs.append("errors.non_retryable must be a list")
    return errs


def contract_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(live, pinned or pinned_contract(), DRIFT_KEYS)


def package_from_contract(contract: Dict[str, Any]) -> SkillPackage:
    ver = str(contract.get("version") or "")
    ops = as_dict(contract.get("operations"))
    result_keys = ["operation_type", "artifact", "timeline", "warnings", "provenance", "commands"]
    tools = [ToolSpec(tool_id=TOOL_GENERATE, skill_id=SKILL_ID, version=ver, description=str(as_dict(ops.get("generate")).get("description", "")), required_capabilities=[SKILL_ID],
                      inputs=["output"], produces_output=True, deterministic=True, result_keys=result_keys),
             ToolSpec(tool_id=TOOL_RENDER, skill_id=SKILL_ID, version=ver, description=str(as_dict(ops.get("render")).get("description", "")),
                      required_capabilities=[SKILL_ID, "ffmpeg", "ffprobe", ENGINE_ID, "encoder:libx264", "filter:subtitles"],
                      inputs=["input", "sidecar", "output"], produces_output=True, deterministic=True, result_keys=result_keys + ["engine", "observation"])]
    return SkillPackage(skill_id=SKILL_ID, name=CONTRACT_SKILL_ID, version=ver, description="Typed subtitle document → SRT / WebVTT file, and burn-in through ffmpeg-skill/caption.",
                        capabilities=[SKILL_ID], tools=tools, repository="kajisho5/subtitle-skill",
                        role="deterministic subtitle file generation and burn-in (through ffmpeg-skill/caption)")


PACKAGE = package_from_contract(pinned_contract())


class SubtitleAdapter(ToolAdapter):
    name = SKILL_ID

    def __init__(self, skill: Optional[CliSkill] = None, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None,
                 ffmpeg_skill_dir: Optional[str] = None, timeout: float = 600.0, path_policy: Optional[PathPolicy] = None):
        located = skill or locate_subtitle()
        if located is None:
            raise ToolError("subtitle-skill not found (set VIDEO_AGENT_SUBTITLE_DIR or install `subtitle-skill`)")
        self.skill: CliSkill = located
        self.workspace = str(Path(workspace).resolve()) if workspace else None
        self.allowed_inputs = [str(Path(r).resolve()) for r in (allowed_inputs or [])]
        self.ffmpeg_skill_dir = str(Path(ffmpeg_skill_dir).resolve()) if ffmpeg_skill_dir else None
        self.default_timeout = float(timeout)
        self.path_policy = path_policy
        self.calls = 0
        self.contract = self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("subtitle-skill contract incompatible: " + "; ".join(errs))
        self.version = str(self.contract["version"])
        self.contract_version = str(self.contract["contract_version"])
        self.operations: Dict[str, Dict[str, Any]] = {str(k): as_dict(v) for k, v in as_dict(self.contract.get("operations")).items()}
        self.formats: Dict[str, List[str]] = {k: [str(f) for f in v.get("formats") or []] for k, v in self.operations.items()}
        self.constraint_keys = tuple(str(k) for k in as_dict(self.contract.get("parameters")).get("constraints") or [])
        self.forbidden = tuple(sorted(set(FORBIDDEN_ARG_KEYS) | set(SKILL_FORBIDDEN_KEYS)))
        self.retryable, self.exit_codes = error_table(self.contract)
        self.tools = {TOOL_GENERATE, TOOL_RENDER}
        self._drift: Optional[List[str]] = None

    # ---- transport
    def _invoke(self, argv: List[str], stdin: Optional[str] = None, timeout: Optional[float] = None):
        self.calls += 1
        return invoke(self.skill, argv, stdin=stdin, timeout=timeout or self.default_timeout, extra_env=self._engine_env())

    def _engine_env(self) -> Dict[str, str]:
        """ffmpeg-skill's location for the child (the Skill reads only this variable; nothing else is added to its environment)."""
        return {ENGINE_DIR_ENV: self.ffmpeg_skill_dir} if self.ffmpeg_skill_dir else {}

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._invoke(["contract", "--json"], timeout=60.0)
        if code != 0:
            raise ContractError(f"subtitle-skill contract --json failed (exit {code}): {err.strip()[-300:]}")
        try:
            return one_json_document(out, "subtitle-skill contract")
        except ToolError as e:
            raise ContractError(str(e))

    def doctor(self) -> Dict[str, Any]:
        code, out, err = self._invoke(["doctor", "--json"], timeout=180.0)
        try:
            doc = one_json_document(out, "subtitle-skill doctor")
        except ToolError as e:
            return {"skill": CONTRACT_SKILL_ID, "status": "fail", "healthy": False, "problems": [f"doctor produced no document: {e}"], "supported_operations": [], "exit_code": code}
        if doc.get("skill") != CONTRACT_SKILL_ID or not isinstance(doc.get("healthy"), bool):
            return {"skill": CONTRACT_SKILL_ID, "status": "fail", "healthy": False, "problems": [f"unexpected doctor document {doc.get('skill')!r} / {doc.get('healthy')!r}"],
                    "supported_operations": [], "exit_code": code}
        doc["status"] = "ok" if doc["healthy"] else "fail"
        doc["exit_code"] = code
        return doc

    @staticmethod
    def operation_status(doc: Dict[str, Any]) -> Dict[str, str]:
        """Per operation: supported | unsupported, as the Skill's doctor reports it in supported_operations (never inferred here)."""
        supported = {str(x) for x in doc.get("supported_operations") or []}
        return {op: "supported" if op in supported else "unsupported" for op in OPERATIONS}

    @staticmethod
    def render_formats(doc: Dict[str, Any]) -> List[str]:
        return [str(x) for x in doc.get("render_supported_formats") or []]

    def drift(self) -> List[str]:
        if self._drift is None:
            self._drift = contract_drift(self.contract)
        return self._drift

    # ---- ToolAdapter
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "root": self.skill.describe(), "tools": sorted(self.tools), "operations": sorted(self.operations),
                "formats": self.formats, "contract": self.contract_version, "drift": self.drift()}

    def package(self) -> SkillPackage:
        return package_from_contract(self.contract)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            b = self.build_request(op.tool, op.args, paths, op_id=op.id)
        except ToolError as e:
            return [f"subtitle-skill: refused: {e}"]
        return [" ".join(["subtitle-skill"] + self._argv()) + "  <<< " + json.dumps(b["request"], ensure_ascii=False)[:400]]

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        raise ToolError("subtitle-skill: an execution Skill has no measurement tools")

    # ---- lowering: typed args → one request document
    @staticmethod
    def _finite(name: str, v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ToolError(f"subtitle-skill: {name} must be a finite number")
        return float(v)

    def cues_for(self, raw: Any) -> List[Dict[str, Any]]:
        """The typed cues the agent hands over, checked here before the Skill sees them: unique ids, ordered finite times, plain text.
        `speaker` / `style` / `metadata` are never emitted (the agent identifies no speakers and styles nothing)."""
        if not isinstance(raw, list) or not raw:
            raise ToolError("subtitle-skill: cues must be a non-empty list")
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for i, c in enumerate(raw):
            if not isinstance(c, dict):
                raise ToolError(f"subtitle-skill: cue[{i}] must be an object")
            extra = sorted(set(c) - set(CUE_KEYS))
            if extra:
                raise ToolError(f"subtitle-skill: cue[{i}] carries unknown field(s) {extra} (accepted: {list(CUE_KEYS)})")
            cid = c.get("id")
            if not isinstance(cid, str) or not _ID_RE.match(cid):
                raise ToolError(f"subtitle-skill: cue[{i}] id {cid!r} is not a valid identifier")
            if cid in seen:
                raise ToolError(f"subtitle-skill: duplicate cue id {cid!r}")
            seen.add(cid)
            start, end = self._finite(f"cue {cid} start", c.get("start")), self._finite(f"cue {cid} end", c.get("end"))
            if start < 0 or end <= start:
                raise ToolError(f"subtitle-skill: cue {cid} needs 0 <= start < end (got {start}, {end})")
            text = c.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_CUE_TEXT:
                raise ToolError(f"subtitle-skill: cue {cid} text must be a non-empty string of at most {MAX_CUE_TEXT} characters")
            if any(ord(ch) < 32 and ch != "\n" for ch in text) or "\x7f" in text:
                raise ToolError(f"subtitle-skill: cue {cid} text carries control characters")
            out.append({"id": cid, "start": start, "end": end, "text": text})
        return out

    def constraints_for(self, raw: Any) -> Optional[Dict[str, Any]]:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ToolError("subtitle-skill: constraints must be an object")
        out: Dict[str, Any] = {}
        for k, v in raw.items():
            if str(k) not in self.constraint_keys:
                raise ToolError(f"subtitle-skill: constraint {k!r} is not declared by the contract ({list(self.constraint_keys)})")
            x = self._finite(f"constraints.{k}", v)
            if x <= 0:
                raise ToolError(f"subtitle-skill: constraints.{k} must be positive")
            if k in INTEGER_CONSTRAINTS:
                if x != int(x):
                    raise ToolError(f"subtitle-skill: constraints.{k} must be an integer")
                out[str(k)] = int(x)
            else:
                out[str(k)] = x
        return out or None

    def build_request(self, tool: str, args: Dict[str, Any], paths: Dict[str, str], op_id: str = "op", timeout: Optional[float] = None) -> Dict[str, Any]:
        if tool not in self.tools:
            raise ToolError(f"subtitle-skill: unsupported tool {tool}")
        if not self.workspace:
            raise ToolError("subtitle-skill: a workspace is required (the Skill accepts workspace-relative paths only)")
        hit = scan_forbidden(args, self.forbidden)
        if hit:
            raise ToolError(f"subtitle-skill: forbidden field {hit} in the operation arguments")
        op_type = "generate" if tool == TOOL_GENERATE else "render"
        if str(args.get("operation") or op_type) != op_type:
            raise ToolError(f"subtitle-skill: operation {args.get('operation')!r} does not match tool {tool}")
        allowed = RENDER_ARGS if op_type == "render" else COMMON_ARGS
        extra = sorted(set(args) - set(allowed))
        if extra:
            raise ToolError(f"subtitle-skill: unknown argument(s) {extra} for {tool} (accepted: {list(allowed)})")
        fmt = str(args.get("format") or "")
        if fmt not in self.formats.get(op_type, []):
            raise ToolError(f"subtitle-skill: format {fmt!r} is not one of {self.formats.get(op_type)} for {op_type}")
        out_id = str(args.get("output") or "")
        if not out_id:
            raise ToolError("subtitle-skill: an output reference is required")
        out = str(Path(paths.get(out_id, out_id)).resolve())
        if op_type == "generate" and Path(out).suffix.lower().lstrip(".") != fmt:
            raise ToolError(f"subtitle-skill: the output extension must match the format ({fmt}): {out}")
        src: Optional[str] = None
        if op_type == "render":
            src_id = str(args.get("input") or "")
            if not src_id:
                raise ToolError("subtitle-skill: render requires an input video reference")
            src = paths.get(src_id, src_id)
            if not os.path.isfile(src):
                raise ToolError(f"subtitle-skill: input not found: {src}")
            src = str(Path(src).resolve())
            if self.path_policy is not None:
                self.path_policy.check_input(src)
            elif self.allowed_inputs and not any(self._under(src, r) for r in self.allowed_inputs + [self.workspace]):
                raise ToolError(f"subtitle-skill: input outside the allowed roots: {src}")
            # the Skill resolves video_input inside the request workspace only; the planner guarantees the burn-in source is an intermediate there
            if not self._under(src, self.workspace):
                raise ToolError(f"subtitle-skill: the render input must live inside the workspace (the Skill takes workspace-relative paths only): {src}")
            if "sidecar" in args:
                side = paths.get(str(args["sidecar"]), str(args["sidecar"]))
                if not os.path.isfile(side):
                    raise ToolError(f"subtitle-skill: sidecar not found: {side}")
                if self.path_policy is not None:
                    self.path_policy.check_input(side)
                elif not any(self._under(side, r) for r in self.allowed_inputs + [self.workspace]):
                    raise ToolError(f"subtitle-skill: sidecar outside the allowed roots: {side}")
        if self.path_policy is not None:
            self.path_policy.check_output(out, [src] if src else [])
        if not self._under(out, self.workspace) or same_file(out, self.workspace):
            raise ToolError(f"subtitle-skill: output outside the workspace: {out}")
        if src and same_file(src, out):
            raise ToolError(f"subtitle-skill: output would overwrite its input: {out}")
        language = args.get("language")
        if not isinstance(language, str) or not _LANG_RE.match(language):
            raise ToolError(f"subtitle-skill: language {language!r} is not a language tag")
        cues = self.cues_for(args.get("cues"))
        constraints = self.constraints_for(args.get("constraints"))
        video_duration: Optional[float] = None
        if args.get("video_duration") is not None:
            video_duration = self._finite("video_duration", args["video_duration"])
            if video_duration <= 0:
                raise ToolError("subtitle-skill: video_duration must be positive")
        did = re.sub(r"[^A-Za-z0-9._-]", "_", str(args.get("document_id") or op_id))[:64]
        if not _ID_RE.match(did):
            did = "doc"
        ws_real = os.path.realpath(self.workspace)
        req: Dict[str, Any] = {"operation": op_type, "format": fmt, "workspace": self.workspace,
                               "output_path": os.path.relpath(os.path.realpath(out), ws_real).replace(os.sep, "/"),
                               "subtitle": {"id": did, "version": 1, "language": language, "cues": cues}}
        if src:
            req["video_input"] = os.path.relpath(os.path.realpath(src), ws_real).replace(os.sep, "/")
        if constraints:
            req["constraints"] = constraints
        if video_duration is not None:
            req["video_duration"] = video_duration
        hit = scan_forbidden(req, tuple(f for f in self.forbidden if f not in ("workspace",)), "request")
        if hit:
            raise ToolError(f"subtitle-skill: refusing to send a request carrying {hit}")
        return {"request": req, "output": out, "input": src, "type": op_type, "format": fmt, "workspace": self.workspace, "cue_count": len(cues)}

    @staticmethod
    def _under(path: str, root: str) -> bool:
        try:
            p, r = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
        except OSError:
            return False
        return p == r or p.startswith(r.rstrip(os.sep) + os.sep)

    @staticmethod
    def _argv() -> List[str]:
        return ["run", "-", "--json"]

    # ---- execution
    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        try:
            b = self.build_request(op.tool, op.args, paths, op_id=op.id, timeout=timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e), retryable=False)
        if dry_run:   # the Skill has no dry-run: the lowered, validated request is the preview
            data = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run", "operation_type": b["type"], "request": b["request"], "warnings": [], "commands": []}
            return ToolResult(op.id, op.tool, True, 0, None, data, [], "", round(time.time() - t0, 3), attempt, True)
        os.makedirs(os.path.dirname(b["output"]) or ".", exist_ok=True)
        code, out, err = self._invoke(self._argv(), stdin=json.dumps(b["request"], ensure_ascii=False), timeout=(timeout or self.default_timeout) + 5.0)
        tail = "\n".join(err.strip().splitlines()[-12:])
        secs = round(time.time() - t0, 3)
        if code == 124:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, 124, "CANCELLED", tail or "process timed out", retryable=True, details={"reason": "timeout"})
        try:
            doc = one_json_document(out, "subtitle-skill")
        except ToolError as e:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 9, "INVALID_RESULT", str(e), retryable=False, details={"exit_code": code})
        if doc.get("status") != "ok":
            se = as_dict(doc.get("error"))
            errc = str(se.get("code") or "INVALID_RESULT")
            if errc not in self.retryable:
                errc, retry = "INVALID_RESULT", False
            else:
                retry = bool(se["retryable"]) if isinstance(se.get("retryable"), bool) else self.retryable.get(errc, False)
            details = scrub(se.get("details"), self.forbidden)
            if errc == "CANCELLED" and (details.get("reason") or "") not in ("timeout", "signal"):
                details["reason"] = details.get("reason") or "signal"
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 1, errc, str(se.get("message") or tail or "")[:500], retryable=bool(retry), details=details)
        errs = self._check_response(doc, b)
        if errs:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 9, "INVALID_RESULT", "; ".join(errs), retryable=False, details={"exit_code": code})
        if code != 0:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code, "INVALID_RESULT", f"exit code {code} with an ok response", retryable=False)
        data = self._success_data(doc, b)
        return ToolResult(op.id, op.tool, True, 0, b["output"], data, [], tail, secs, attempt, dry_run)

    def _check_response(self, doc: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
        errs: List[str] = []
        out_path = str(b["output"])
        if doc.get("skill") != CONTRACT_SKILL_ID or str(doc.get("skill_version")) != self.version:
            errs.append(f"response skill {doc.get('skill')!r}@{doc.get('skill_version')!r} is not {CONTRACT_SKILL_ID}@{self.version}")
        if str(doc.get("contract_version")) != self.contract_version:
            errs.append(f"response contract_version {doc.get('contract_version')!r} != {self.contract_version}")
        if doc.get("operation") != b["type"]:
            errs.append(f"response operation {doc.get('operation')!r} != requested {b['type']}")
        if not doc.get("output") or not same_file(str(doc["output"]), out_path):
            errs.append(f"output path {doc.get('output')!r} is not the requested {out_path}")
        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            return errs + ["output file missing or empty"]
        ok, actual = fingerprint_matches(doc.get("sha256"), out_path)
        if not ok:
            errs.append(f"sha256 {doc.get('sha256')!r} != file {actual}")
        if doc.get("size") != os.path.getsize(out_path):
            errs.append("size differs from the file")
        if as_dict(doc.get("timeline")).get("cue_count") != b["cue_count"]:
            errs.append(f"timeline.cue_count {as_dict(doc.get('timeline')).get('cue_count')!r} != {b['cue_count']} cues sent")
        if not isinstance(doc.get("observation"), list):
            errs.append("observation list missing")
        if b["type"] == "render":
            if doc.get("engine") != ENGINE_ID or not isinstance(doc.get("engine_version"), str) or not doc.get("engine_version"):
                errs.append(f"render must report engine {ENGINE_ID} with a version, got {doc.get('engine')!r} / {doc.get('engine_version')!r}")
        return errs

    def _success_data(self, doc: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        sha = str(doc.get("sha256") or "")
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "completed", "operation_type": b["type"],
                                "artifact": {"path": b["output"], "sha256": sha, "size": doc.get("size"), "format": b["format"], "cue_count": b["cue_count"], "reused": bool(doc.get("reused"))},
                                "timeline": {"cue_count": as_dict(doc.get("timeline")).get("cue_count")},
                                "warnings": [dict(w) for w in doc.get("observation") or [] if isinstance(w, dict)],
                                "provenance": {"skill": doc.get("skill"), "skill_version": doc.get("skill_version"), "contract_version": doc.get("contract_version"), "output_hash": sha},
                                "commands": []}
        if b["type"] == "render":
            data["engine"] = {"id": str(doc.get("engine")), "version": str(doc.get("engine_version"))}
            data["observation"] = None   # the Skill reports no probe of the rendered file; the agent's QA probes it itself
        return data

    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, code: int, errc: str, message: str, retryable: bool, details: Optional[Dict[str, Any]] = None) -> ToolResult:
        return failed_result(op, SKILL_ID, getattr(self, "version", ""), attempt, dry_run, t0, code, errc, message, retryable, details)


def lift_result(result: ToolResult, asset_id: Optional[str] = None) -> Optional[Observation]:
    """The delivered subtitle artifact (cue count, format, hash, size) as an OBSERVED agent Observation for provenance; None on failure."""
    if not result.ok or result.dry_run:
        return None
    art = as_dict((result.data or {}).get("artifact"))
    if not art.get("sha256"):
        return None
    sk = as_dict((result.data or {}).get("skill"))
    src = f"{CONTRACT_SKILL_ID}/{(result.data or {}).get('operation_type')}@{sk.get('version') or ''}"
    return Observation(kind="subtitle.file", asset_id=asset_id or result.op_id, source=src,
                       data={"cue_count": art.get("cue_count"), "format": art.get("format"), "sha256": art.get("sha256"), "size": art.get("size")},
                       analyzer=src, provenance="OBSERVED", skill=SKILL_ID, skill_version=str(sk.get("version") or ""), tool=result.tool, fingerprint=str(art.get("sha256")),
                       parameters={"output": result.output})

