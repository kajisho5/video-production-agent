"""NleBridgeAdapter: the agent's boundary to nle-bridge-skill (ADR-086; nle-bridge-skill roadmap step 7, agent side: the adapter).

    video-production-agent ─(the project IR it executes, as a file)─→ NleBridgeAdapter ─(argv list, never a shell)─→
        `nle-bridge export  --json --overwrite --output=<dir> --subject=<s> [--formats=otio,fcp7,edl,srt] -- <project.json>`
        `nle-bridge conform --json --overwrite --output=<dir> --subject=<s> [--formats=…] [--zip] --ffmpeg-skill=<dir> -- <project.json>`

nle-bridge reads video-production-agent's project IR itself (its contract `inputs`: "project IR (delivery.timelines[], ADR-052)"): the
`timeline.export` record of the subject, the assets' paths and probe facts, the subject's caption cues, and the `not_exported` list of the
timeline decision (as warnings). The agent hands it the IR it is executing -- nothing is converted or re-described here.

Protocol (nle-bridge-skill's own contract, `nle-bridge contract --json`, pinned from main in contract_0.1.0.json at c0ab78f -- step 12, ffmpeg-skill
`>=2.0.0 <4.0.0` and start timecodes in relink.json; no release tag, so the pin follows main):
    export   → {ok, files[{format, path, sha256, verification}], warnings[{code, message}], timeline, plan}
    conform  → {ok, package, zip {path, sha256} | null, files, relink (nle-bridge/relink@1), manifest, warnings, timeline, plan}
    failure  → {ok: false, error {code, message, detail}}, the code fixing the exit code (its errors table)

Every value is checked before the process starts (formats from the contract's `formats`, the output directory inside the workspace, the IR
file an artifact of this job), every option is one `--flag=value` word and the IR comes after `--` (ADR-069). The result is admitted only when
every file it names is inside the output directory and its sha256 matches the file on disk. Nothing here edits a timeline or controls an NLE."""
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
from ..skill_process import (FORBIDDEN_ARG_KEYS, CliSkill, ContractError, as_dict, breaking_drift as _breaking_drift, drift_report, invoke, one_json_document,
                             scan_forbidden, scrub, sha256_file)
from .locate import PACKAGE_ID, locate_nle_bridge  # noqa: F401  (re-exported)

SKILL_ID = PACKAGE_ID
TOOL_EXPORT = "nle-bridge/export"
TOOL_CONFORM = "nle-bridge/conform"
TOOLS = (TOOL_EXPORT, TOOL_CONFORM)
COMMAND = {TOOL_EXPORT: "export", TOOL_CONFORM: "conform"}
SUPPORTED_SKILL_VERSIONS = ("0.1.",)
PROJECT_IR_INPUT = "video-production-agent project IR"
DEFAULT_TIMEOUT = 600.0
ERROR_CLASSES: Dict[str, str] = {
    "INPUT_NOT_FOUND": "INPUT_MISSING", "INVALID_ARGUMENT": "INVALID_ARGS", "INVALID_PLAN": "INVALID_ARGS", "INVALID_TIMELINE": "INVALID_ARGS",
    "INVALID_BEATS": "INVALID_ARGS", "INVALID_CAPTIONS": "INVALID_ARGS", "UNSUPPORTED_FORMAT": "INVALID_ARGS", "OUTPUT_EXISTS": "INVALID_ARGS",
    "OUTPUT_IS_INPUT": "INVALID_ARGS", "ENGINE_TOOL_MISSING": "TOOL_MISSING", "ENGINE_FAILED": "SKILL_ERROR", "ENGINE_TIMEOUT": "TIMEOUT",
    "CONTENT_NOT_EXPORTED": "SKILL_ERROR",
}
AGENT_ERRORS: Dict[str, str] = {"INVALID_REQUEST": "INVALID_ARGS", "INVALID_OUTPUT": "SKILL_ERROR", "TIMEOUT": "TIMEOUT"}
DRIFT_KEYS = ("skill_id", "version", "contract_version", "provides", "dependencies", "not_provided", "formats", "default_formats", "inputs", "timeline",
              "start_timecode", "conform", "warnings", "execution", "errors")
DRIFT_TOOL_KEYS = ("role", "writes_media", "mutates_input", "deterministic", "argv")
FATAL_DRIFT_KEYS = ("skill_id", "contract_version", "execution", "formats")
FATAL_DRIFT_TOOL_KEYS = ("role", "writes_media", "mutates_input", "argv")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def formats(contract: Optional[Dict[str, Any]] = None) -> Tuple[str, ...]:
    """The interchange formats the contract writes (otio, fcp7, edl, srt), read from it."""
    return tuple(sorted(as_dict((contract or pinned_contract()).get("formats"))))


def default_formats(contract: Optional[Dict[str, Any]] = None) -> Tuple[str, ...]:
    return tuple(str(f) for f in (contract or pinned_contract()).get("default_formats") or [])


def recovery_class(code: str) -> str:
    return ERROR_CLASSES.get(code) or AGENT_ERRORS.get(code) or "SKILL_ERROR"


def _tool(c: Dict[str, Any], tid: str) -> Dict[str, Any]:
    return next((t for t in c.get("tools") or [] if isinstance(t, dict) and t.get("tool_id") == tid), {})


def check_contract(contract: Any) -> List[str]:
    """What the adapter relies on: identity and versions; export / conform as execution tools that write no media and never touch their
    input, taking `--subject` and `-o`; the project IR among the inputs; no network, no shell, read-only inputs; formats; the error codes."""
    errs: List[str] = []
    if not isinstance(contract, dict):
        return ["contract is not an object"]
    if contract.get("skill_id") != SKILL_ID:
        errs.append(f"skill_id {contract.get('skill_id')!r} != {SKILL_ID}")
    if not str(contract.get("contract_version") or "").startswith("1."):
        errs.append(f"contract_version {contract.get('contract_version')!r} is not 1.x")
    if not str(contract.get("version") or "").startswith(SUPPORTED_SKILL_VERSIONS):
        errs.append(f"skill version {contract.get('version')!r} not in supported range {SUPPORTED_SKILL_VERSIONS}")
    for tid in TOOLS:
        t = _tool(contract, tid)
        argv = [str(a) for a in t.get("argv") or []]
        if not t or t.get("role") != "execution" or t.get("writes_media") is not False or t.get("mutates_input") is not False \
                or "-o <dir>" not in argv or "[--subject <asset>]" not in argv:
            errs.append(f"tool {tid} must be an execution tool that writes no media, never touches its input, and takes -o <dir> and --subject")
    if "[--zip]" not in [str(a) for a in _tool(contract, TOOL_CONFORM).get("argv") or []]:
        errs.append(f"tool {TOOL_CONFORM} must take [--zip]")
    if not any(PROJECT_IR_INPUT in str(i) for i in contract.get("inputs") or []):
        errs.append(f"inputs must include the {PROJECT_IR_INPUT}")
    ex = as_dict(contract.get("execution"))
    if ex.get("network") is not False or ex.get("shell") is not False or ex.get("input_is_read_only") is not True:
        errs.append("execution must declare network false, shell false and input_is_read_only true")
    if not {"otio", "fcp7", "edl"} <= set(formats(contract)):
        errs.append("formats must include otio, fcp7 and edl")
    errors = as_dict(contract.get("errors"))
    for code in ERROR_CLASSES:
        if not isinstance(as_dict(errors.get(code)).get("exit_code"), int):
            errs.append(f"errors lack {code} (with its exit_code)")
    return errs


def _op_tools(c: Dict[str, Any]) -> Dict[str, Any]:
    return dict(c, tools=[t for t in c.get("tools") or [] if isinstance(t, dict) and t.get("tool_id") in TOOLS])


def drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(_op_tools(live), _op_tools(pinned or pinned_contract()), DRIFT_KEYS, "tools", "tool_id", DRIFT_TOOL_KEYS)


def pinned_error_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    p, lv = as_dict((pinned or pinned_contract()).get("errors")), as_dict(live.get("errors"))
    return [f"errors {c}: pinned {json.dumps(p[c], sort_keys=True)} != live {json.dumps(lv.get(c), sort_keys=True)}" for c in sorted(p) if lv.get(c) != p[c]]


def breaking_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    pinned = pinned or pinned_contract()
    return _breaking_drift(_op_tools(live), _op_tools(pinned), FATAL_DRIFT_KEYS, "tools", "tool_id", FATAL_DRIFT_TOOL_KEYS) + pinned_error_drift(live, pinned)


PACKAGE = SkillPackage(
    skill_id=SKILL_ID, name="nle-bridge-skill", version="",
    description="Hand an edit to Premiere Pro / DaVinci Resolve as files: OTIO, FCP7 XML, CMX 3600 EDL and an SRT sidecar from the agent's project IR, "
                "or a conform package with a relink manifest; every loss a named warning; local, no network",
    capabilities=["nle-bridge"],
    tools=[ToolSpec(TOOL_EXPORT, SKILL_ID, description="The subject's timeline from the project IR as OTIO / FCP7 XML / EDL (+ SRT)", inputs=["project"],
                    produces_output=True, deterministic=True, result_keys=["files", "warnings", "timeline"]),
           ToolSpec(TOOL_CONFORM, SKILL_ID, description="A conform package: the timelines, the plan, a relink manifest (sha256, start timecode, ranges used) and a zip",
                    inputs=["project"], produces_output=True, deterministic=True, result_keys=["package", "zip", "files", "relink", "manifest", "warnings"])],
    repository="kajisho5/nle-bridge-skill", role="NLE interchange (timeline hand-off to Premiere Pro / DaVinci Resolve)")


class NleBridgeAdapter(ToolAdapter):
    owns_cache = False

    def __init__(self, skill: CliSkill, workspace: Optional[str] = None, ffmpeg_skill_dir: Optional[str] = None, path_policy: Optional[PathPolicy] = None,
                 timeout: float = DEFAULT_TIMEOUT):
        self.skill = skill
        self.name = SKILL_ID
        self.workspace = os.path.realpath(os.path.abspath(workspace)) if workspace else None
        self.ffmpeg_skill_dir = str(Path(ffmpeg_skill_dir).resolve()) if ffmpeg_skill_dir else None
        self.policy = path_policy
        self.default_timeout = float(timeout)
        code, out, err = invoke(self.skill, ["contract", "--json"], timeout=60.0)
        if code != 0:
            raise ContractError(f"nle-bridge contract --json failed (exit {code}): {err.strip()[-300:]}")
        self.contract = one_json_document(out, "nle-bridge contract")
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("incompatible nle-bridge-skill contract: " + "; ".join(errs))
        self.version = str(self.contract.get("version"))
        self.tools = {t: True for t in TOOLS}
        self.formats = formats(self.contract)
        self.retryable = {c: bool(as_dict(v).get("retryable")) for c, v in as_dict(self.contract.get("errors")).items()}
        self._doctor: Optional[Dict[str, Any]] = None

    def doctor(self) -> Dict[str, Any]:
        """`nle-bridge doctor --json [--ffmpeg-skill=<dir>]`: nothing is required beyond Python (ok); its notes are kept."""
        if self._doctor is None:
            code, out, err = invoke(self.skill, ["doctor", "--json"] + ([f"--ffmpeg-skill={self.ffmpeg_skill_dir}"] if self.ffmpeg_skill_dir else []), timeout=120.0)
            try:
                doc = one_json_document(out, "nle-bridge doctor")
            except ToolError as e:
                doc = {"ok": False, "problems": [f"doctor produced no document (exit {code}): {e} {err.strip()[-200:]}"]}
            self._doctor = dict(doc, status="ok" if doc.get("ok") is True and not doc.get("problems") else "fail", problems=list(doc.get("problems") or []),
                                warnings=[str(n) for n in doc.get("notes") or []])
        return self._doctor

    def drift(self) -> List[str]:
        return drift(self.contract)

    def breaking_drift(self) -> List[str]:
        return breaking_drift(self.contract)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "skill": self.skill.describe(), "tools": list(TOOLS), "formats": list(self.formats), "drift": self.drift()}

    def package(self) -> SkillPackage:
        return SkillPackage(PACKAGE.skill_id, PACKAGE.name, self.version, PACKAGE.description, list(PACKAGE.capabilities),
                            [ToolSpec(**{**t.__dict__, "version": self.version}) for t in PACKAGE.tools], PACKAGE.repository, PACKAGE.role)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        raise ToolError("nle-bridge: no measurement tools are used by the agent")

    def _within(self, root: str, p: str) -> bool:
        r, q = os.path.normcase(root), os.path.normcase(p)
        return q == r or q.startswith(r.rstrip(os.sep) + os.sep)

    def build(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None) -> Dict[str, Any]:
        if op.tool not in TOOLS:
            raise ToolError(f"{SKILL_ID} adapter does not support {op.tool}")
        args = dict(op.args)
        hit = scan_forbidden(args, FORBIDDEN_ARG_KEYS)
        if hit:
            raise ToolError(f"forbidden key {hit}")
        allowed = {"project", "subject", "formats", "output"} | ({"zip"} if op.tool == TOOL_CONFORM else set())
        unknown = sorted(k for k in args if k not in allowed)
        if unknown:
            raise ToolError(f"unknown {op.tool} argument(s): {', '.join(unknown)}")
        ref = args.get("project")
        if not isinstance(ref, str) or ref not in paths or not os.path.isfile(paths[ref]):
            raise ToolError("project must be the project IR file of this job")
        project = os.path.realpath(paths[ref])
        subject = args.get("subject")
        if not isinstance(subject, str) or not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", subject):
            raise ToolError(f"subject {subject!r} is not an id")
        fmts = list(args.get("formats") or [])
        if not fmts or any(f not in self.formats for f in fmts) or len(set(fmts)) != len(fmts):
            raise ToolError(f"formats {fmts!r} must be distinct names from {', '.join(self.formats)}")
        out_ref = args.get("output")
        if not isinstance(out_ref, str) or out_ref not in paths:
            raise ToolError("output must be an artifact id of this job")
        out_dir = os.path.realpath(os.path.abspath(paths[out_ref]))
        if self.workspace and not (self._within(self.workspace, out_dir) and out_dir != self.workspace):
            raise ToolError(f"output directory is outside the workspace: {os.path.basename(out_dir)}")
        if self._within(out_dir, project):
            raise ToolError("the output directory holds the project IR")
        if self.policy is not None:
            self.policy.check_input(project)
        t = max(1.0, min(86400.0, float(timeout or self.default_timeout)))
        argv = [COMMAND[op.tool], "--json", "--overwrite", f"--output={out_dir}", f"--subject={subject}", f"--formats={','.join(fmts)}"]
        if op.tool == TOOL_CONFORM:
            argv += (["--zip"] if args.get("zip") else []) + ([f"--ffmpeg-skill={self.ffmpeg_skill_dir}"] if self.ffmpeg_skill_dir else ["--no-probe"])
        return {"argv": argv + ["--", project], "out_dir": out_dir, "project": project, "timeout": t, "formats": fmts}

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            return [" ".join(["nle-bridge"] + self.build(op, paths)["argv"])]
        except ToolError as e:
            return [f"<refused: {e}>"]

    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, exit_code: int, code: str, message: str, retryable: bool = False,
              details: Optional[Dict[str, Any]] = None) -> ToolResult:
        data = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "failed",
                "error": {"code": code, "message": message[:1200], "retryable": bool(retryable), "details": dict(details or {}), "exit_code": exit_code,
                          "recovery_class": recovery_class(code)}}
        return ToolResult(op.id, op.tool, False, exit_code, None, data, [], f"{SKILL_ID} [{code}] {message[:600]}", round(time.time() - t0, 3), attempt, dry_run)

    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        try:
            b = self.build(op, paths, timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e))
        command = " ".join(["nle-bridge"] + b["argv"])
        if dry_run:
            return ToolResult(op.id, op.tool, True, 0, None, {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run", "commands": [command]},
                              [command], "", round(time.time() - t0, 3), attempt, True)
        code, out, err = invoke(self.skill, b["argv"], timeout=b["timeout"])
        if code == 124:
            return self._fail(op, attempt, dry_run, t0, 124, "TIMEOUT", f"nle-bridge {COMMAND[op.tool]} exceeded {b['timeout']:g}s", retryable=True, details={"reason": "timeout"})
        try:
            doc = one_json_document(out, f"nle-bridge {COMMAND[op.tool]}")
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, code or 9, "INVALID_OUTPUT", f"{e} (exit {code}) {err.strip()[-300:]}")
        if doc.get("ok") is not True:
            e = as_dict(doc.get("error"))
            c = str(e.get("code") or "INVALID_OUTPUT")
            retry = self.retryable.get(c, False) if c in ERROR_CLASSES else False
            return self._fail(op, attempt, dry_run, t0, code or 1, c, str(e.get("message") or ""), retryable=retry, details=scrub({"detail": e.get("detail")}, FORBIDDEN_ARG_KEYS))
        why = self._admit(op.tool, doc, b)
        if why:
            return self._fail(op, attempt, dry_run, t0, code or 9, "INVALID_OUTPUT", "nle-bridge result not admitted: " + "; ".join(why))
        files = [{"format": f.get("format"), "file": f.get("path"), "sha256": f.get("sha256"), "verification": f.get("verification")} for f in doc.get("files") or []]
        primary = (as_dict(doc.get("zip")).get("path") if op.tool == TOOL_CONFORM and doc.get("zip") else None) or (doc.get("manifest") if op.tool == TOOL_CONFORM else None) \
            or files[0]["file"]
        data = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "ok", "op": COMMAND[op.tool], "files": files,
                "warnings": [{"code": str(as_dict(w).get("code") or ""), "message": str(as_dict(w).get("message") or w)[:600]} for w in doc.get("warnings") or []],
                "timeline": as_dict(doc.get("timeline")), "package": doc.get("package"), "zip": doc.get("zip"),
                "relink": {k: as_dict(doc.get("relink")).get(k) for k in ("schema", "rate", "drop_frame")} if doc.get("relink") else None,
                "relink_media": [{k: as_dict(m).get(k) for k in ("name", "exists", "sha256", "start_timecode", "start_timecode_source", "fps")}
                                 for m in as_dict(doc.get("relink")).get("media") or []], "commands": [command]}
        return ToolResult(op.id, op.tool, True, 0, primary, data, [command], err.strip()[-300:], round(time.time() - t0, 3), attempt, False)

    def _admit(self, tool: str, doc: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
        errs: List[str] = []
        files = [as_dict(f) for f in doc.get("files") or []]
        if not files:
            return ["no files written"]
        timeline_formats = [f for f in b["formats"] if f != "srt"]
        if not {str(f.get("format")) for f in files} >= set(timeline_formats):
            errs.append(f"files {[f.get('format') for f in files]} lack a requested format of {timeline_formats}")
        for f in files:
            p = os.path.realpath(str(f.get("path") or ""))
            if not self._within(b["out_dir"], p):
                errs.append(f"{os.path.basename(p)} is outside the output directory")
            elif not os.path.isfile(p) or str(f.get("sha256") or "") != sha256_file(p):
                errs.append(f"{os.path.basename(p)}: missing, or its sha256 is not the one reported")
        if tool == TOOL_CONFORM:
            z = as_dict(doc.get("zip"))
            if z and (not os.path.isfile(str(z.get("path") or "")) or str(z.get("sha256") or "") != sha256_file(str(z["path"]))):
                errs.append("the package zip is missing or its sha256 is not the one reported")
            if not os.path.isfile(str(doc.get("manifest") or "")):
                errs.append("no manifest.json")
        return errs


def lift_observation(result: ToolResult, asset_id: Optional[str] = None) -> Optional[Observation]:
    """What nle-bridge reported (provenance only): the files with their sha256 and re-read verification, every loss warning verbatim, the timeline
    summary, and for a conform package its zip and the relink facts (start timecode and where it came from)."""
    data = result.data if isinstance(result.data, dict) else {}
    if data.get("status") != "ok":
        return None
    sk = as_dict(data.get("skill"))
    files = [dict(f) for f in data.get("files") or [] if isinstance(f, dict)]
    return Observation(kind=f"nle.{data.get('op')}", asset_id=asset_id or result.op_id, source=f"{result.tool}@{sk.get('version') or ''}",
                       data={k: data.get(k) for k in ("files", "warnings", "timeline", "package", "zip", "relink", "relink_media")},
                       analyzer=f"{SKILL_ID}@{sk.get('version') or ''}", provenance="OBSERVED", skill=SKILL_ID, skill_version=str(sk.get("version") or ""), tool=result.tool,
                       fingerprint=str((files[0] if files else {}).get("sha256") or ""), parameters={"files": [f.get("file") for f in files]})
