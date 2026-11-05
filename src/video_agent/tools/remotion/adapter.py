"""RemotionAdapter: the agent's boundary to remotion-ffmpeg-skill (ADR-093; remotion-ffmpeg-skill roadmap step 4, agent side: SkillSpec measured_motion ->
pipeline, CONFIRM).

    video-production-agent ─(plan step measured_motion)─→ RemotionAdapter ─(argv list, never a shell)─→
        `python <checkout>/scripts/pipeline.py <video> --project=<the user's Remotion project> --comp=<composition> [--captions=F] [--options=F]
         [--beats] [--conform] [--fast] [--aspect=A] [--fit=F] --out-dir=<job>/ops/<…>_motion --json --timeout=T`
          → analyze (ffmpeg-skill measurements) → render (Remotion, verified frame count / fps / size / audio) ; no --platform: the agent delivers

Protocol (remotion-ffmpeg-skill's own contract, `python scripts/_contract.py --json`, pinned from the v0.12.1 release tag in contract_0.12.1.json):
pipeline's inputSchema / flags / output (`timeline`, `render`, `look`, `verified`, `warnings`, `steps`), the shared output (`status`
completed | failed, `output`), the failure shape (`status: failed`, `error`, `kind` input | missing_tool | timeout | failed, exit 2 / 3 / 4 / 1).
doctor (`scripts/doctor.py --json [--project DIR]`) → `ok` and `checks[]` {check, ok, value, fix} (python, node, npm, ffmpeg-skill, ffmpeg and,
with --project, the project's Remotion install).

What the agent sends is only what the contract's inputSchema lists for pipeline, minus what the agent owns: no `platform` / `brand` / `chapters`
(delivery is the agent's), no `aspect` / `width` / `fit` (the frame stays the subject's), no `overwrite` (a fresh directory per job), no `input` list of several files (one subject's current picture). Every
value is checked before the process starts (the source and the user's files inside the allowed roots, the project a folder with a package.json, the
out dir inside the workspace, enums from the contract). The result is admitted only when it is about this request: status completed,
`verified` true, the render in the planned place on disk (`<stem>_<comp>.mp4`), its step list ending with the draw. Failures keep the Skill's
`kind` with the agent's recovery class."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...models import Operation, ToolResult
from ...skills.contract import SkillPackage, ToolSpec
from ..base import ToolAdapter, ToolError
from ..ffmpeg_skill.adapter import run_process_group
from ..skill_process import ContractError, as_dict, breaking_drift as _breaking_drift, drift_report, one_json_document, same_file
from .locate import PACKAGE_ID, RemotionSkill, locate_remotion  # noqa: F401  (re-exported)

SKILL_ID = PACKAGE_ID
PREFIX = SKILL_ID + "/"
TOOL_PIPELINE = "remotion-ffmpeg-skill/pipeline"
SUPPORTED_SKILL_VERSIONS = ("0.12.",)
# the compositions that keep the source's clock (the agent's later steps and QA expect the length it planned); Jumpcut and Highlights cut the
# picture (their length is known only after the render) and are not planned here
COMPOSITIONS = ("Captioned", "PeakPulse", "Titled", "BeatMontage", "Karaoke", "Pip")
CUTTING_COMPOSITIONS = ("Jumpcut", "Highlights")
# the inputSchema keys the agent may send; not aspect / width / fit (they change the frame the agent's later steps plan on)
SENT = ("project", "comp", "captions", "options", "beats", "conform", "fast", "font_file")
FILE_KEYS = ("captions", "options", "font_file")
# the contract's failure kinds (conventions.errors) → the agent's recovery class
ERROR_CLASSES: Dict[str, str] = {"input": "INVALID_ARGS", "missing_tool": "TOOL_MISSING", "timeout": "TIMEOUT", "failed": "SKILL_ERROR"}
AGENT_ERRORS: Dict[str, str] = {"INVALID_REQUEST": "INVALID_ARGS", "INVALID_OUTPUT": "SKILL_ERROR", "TIMEOUT": "TIMEOUT"}
DEFAULT_TIMEOUT = 1800.0   # the contract's per-step default
PROCESS_MARGIN = 300.0
DRIFT_KEYS = ("skill_id", "version", "contract_version", "provides", "dependencies", "timeline_schema", "depends_on", "roles", "output_shared", "output_failure", "conventions")
DRIFT_TOOL_KEYS = ("role", "inputSchema", "flags", "output", "dry_run")
FATAL_DRIFT_KEYS = ("skill_id", "contract_version", "output_failure", "conventions")
FATAL_DRIFT_TOOL_KEYS = ("role", "flags", "output")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.12.1.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def recovery_class(code: str) -> str:
    return ERROR_CLASSES.get(code) or AGENT_ERRORS.get(code) or "SKILL_ERROR"


def _tool(contract: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next((t for t in contract.get("tools") or [] if isinstance(t, dict) and t.get("name") == name), {})


def check_contract(contract: Any) -> List[str]:
    """What the adapter relies on, refused (never patched) when it does not hold: identity, contract and Skill version, pipeline as an orchestration
    whose inputSchema and flags carry every key the agent sends (and the composition enum it allows), whose result names the render and its
    verification, the shared failure shape and its exit codes, and doctor."""
    errs: List[str] = []
    if not isinstance(contract, dict):
        return ["contract is not an object"]
    if contract.get("skill_id") != SKILL_ID:
        errs.append(f"skill_id {contract.get('skill_id')!r} != {SKILL_ID}")
    if not str(contract.get("contract_version") or "").startswith("1."):
        errs.append(f"contract_version {contract.get('contract_version')!r} is not 1.x")
    ver = str(contract.get("version") or "")
    if not ver.startswith(SUPPORTED_SKILL_VERSIONS):
        errs.append(f"skill version {ver!r} not in supported range {SUPPORTED_SKILL_VERSIONS}")
    p = _tool(contract, "pipeline")
    if not p:
        return errs + ["contract lacks tool pipeline"]
    if p.get("role") != "orchestration":
        errs.append("pipeline must declare role orchestration")
    props = as_dict(as_dict(p.get("inputSchema")).get("properties"))
    flags = as_dict(p.get("flags"))
    missing = [k for k in SENT + ("out_dir", "timeout", "json") if k not in flags or (k not in props and k != "json")]
    if missing:
        errs.append(f"pipeline lacks {missing} in its inputSchema / flags")
    result = as_dict(as_dict(p.get("output")).get("result"))
    if not all(k in result for k in ("render", "verified", "steps")):
        errs.append("pipeline's result does not name render / verified / steps")
    fail = as_dict(contract.get("output_failure"))
    if not all(k in fail for k in ("status", "error", "kind")):
        errs.append("output_failure must carry status / error / kind")
    codes = as_dict(as_dict(contract.get("conventions")).get("errors"))
    for k in ERROR_CLASSES:
        if not isinstance(codes.get(k), int):
            errs.append(f"conventions.errors lacks {k}")
    if not _tool(contract, "doctor"):
        errs.append("contract lacks tool doctor")
    return errs


def _used(c: Dict[str, Any]) -> Dict[str, Any]:
    return dict(c, tools=[t for t in c.get("tools") or [] if isinstance(t, dict) and t.get("name") in ("pipeline", "doctor")])


def drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(_used(live), _used(pinned or pinned_contract()), DRIFT_KEYS, "tools", "name", DRIFT_TOOL_KEYS)


def breaking_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return _breaking_drift(_used(live), _used(pinned or pinned_contract()), FATAL_DRIFT_KEYS, "tools", "name", FATAL_DRIFT_TOOL_KEYS)


PACKAGE = SkillPackage(
    skill_id=SKILL_ID, name="remotion-ffmpeg-skill", version="",
    description="Motion graphics on real footage with Remotion, timed from ffmpeg-skill measurements (captions, pulses on peaks / beats, picture-in-picture, "
                "titles); every render verified against its timeline (EXPERIMENTAL; Remotion's own licence applies to companies over 3 people)",
    capabilities=["remotion-ffmpeg-skill"],
    tools=[ToolSpec(TOOL_PIPELINE, SKILL_ID, description="Measure (analyze) then draw (render) one composition over a video, verified", required_capabilities=["remotion-ffmpeg-skill:PIPELINE"],
                    inputs=["input"], produces_output=True, deterministic=False, result_keys=["render", "timeline", "look", "verified", "warnings", "steps"])],
    repository="kajisho5/remotion-ffmpeg-skill", role="measurement-driven Remotion compositions over the footage")


def render_name(input_path: str, comp: str) -> str:
    """Where pipeline.py writes the render (its own rule: `<out-dir>/<input stem>_<comp>.mp4`)."""
    return f"{Path(input_path).stem}_{comp}.mp4"


class RemotionAdapter(ToolAdapter):
    def __init__(self, skill: RemotionSkill, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None, path_policy: Any = None,
                 timeout: float = DEFAULT_TIMEOUT):
        self.skill = skill
        self.name = SKILL_ID
        self.workspace = os.path.realpath(os.path.abspath(workspace)) if workspace else None
        self.allowed_inputs = [os.path.realpath(os.path.abspath(p)) for p in (allowed_inputs or [])]
        self.policy = path_policy
        self.default_timeout = float(timeout)
        self.contract = self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("incompatible remotion-ffmpeg-skill contract: " + "; ".join(errs))
        self.version = str(self.contract.get("version"))
        self.tools = {TOOL_PIPELINE: True}
        self.enums = {k: list(v.get("enum") or []) for k, v in as_dict(as_dict(_tool(self.contract, "pipeline").get("inputSchema")).get("properties")).items()
                      if isinstance(v, dict) and v.get("enum")}
        self._doctor: Optional[Dict[str, Any]] = None

    def _call(self, script: str, argv: List[str], timeout: Optional[float]) -> Tuple[int, str, str]:
        return run_process_group(self.skill.script(script) + list(argv), timeout=timeout)

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._call("_contract", ["--json"], 60.0)
        if code != 0:
            raise ContractError(f"remotion-ffmpeg-skill _contract.py --json failed (exit {code}): {err.strip()[-300:]}")
        return one_json_document(out, "remotion-ffmpeg-skill contract")

    def doctor(self) -> Dict[str, Any]:
        """`doctor.py --json` (no project: the machine): ok when python / node / npm / ffmpeg-skill / ffmpeg all check; `status` ok or fail."""
        if self._doctor is not None:
            return self._doctor
        code, out, err = self._call("doctor", ["--json"], 300.0)
        try:
            doc = one_json_document(out, "remotion-ffmpeg-skill doctor")
        except ToolError as e:
            doc = {"ok": False, "checks": [], "problems": [f"doctor produced no document (exit {code}): {e} {err.strip()[-200:]}"]}
        bad = [c for c in doc.get("checks") or [] if isinstance(c, dict) and not c.get("ok")]
        self._doctor = dict(doc, status="ok" if doc.get("ok") is True and not bad else "fail",
                            problems=[str(p) for p in doc.get("problems") or []] + [f"{c.get('check')}: {c.get('value')} ({c.get('fix')})" for c in bad])
        return self._doctor

    def operation_status(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {"PIPELINE": "supported" if (doc or self.doctor()).get("status") == "ok" else "unavailable"}

    def operation_evidence(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        doc = doc or self.doctor()
        bad = [c for c in doc.get("checks") or [] if isinstance(c, dict) and not c.get("ok")]
        ev: Dict[str, Any] = {"ready": doc.get("status") == "ok"}
        if bad:
            ev.update(code="missing_tool", install=" ".join(str(c.get("fix") or "") for c in bad)[:600], detail="; ".join(f"{c.get('check')}: {c.get('value')}" for c in bad)[:300])
        return {"PIPELINE": ev}

    def project_status(self, project: str) -> Dict[str, Any]:
        """`doctor.py --json --project DIR`: whether that Remotion project is installed (its own check and fix line)."""
        code, out, _ = self._call("doctor", ["--json", f"--project={project}"], 300.0)
        try:
            doc = one_json_document(out, "remotion-ffmpeg-skill doctor --project")
        except ToolError as e:
            return {"ok": False, "fix": str(e)}
        c = next((c for c in doc.get("checks") or [] if isinstance(c, dict) and c.get("check") == "remotion project"), None)
        return {"ok": bool(c and c.get("ok")), "fix": (c or {}).get("fix") or "", "value": (c or {}).get("value")}

    def drift(self) -> List[str]:
        return drift(self.contract)

    def breaking_drift(self) -> List[str]:
        return breaking_drift(self.contract)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "skill": self.skill.describe(), "tools": [TOOL_PIPELINE], "drift": self.drift()}

    def package(self) -> SkillPackage:
        return SkillPackage(PACKAGE.skill_id, PACKAGE.name, self.version, PACKAGE.description, list(PACKAGE.capabilities),
                            [ToolSpec(**{**t.__dict__, "version": self.version}) for t in PACKAGE.tools], PACKAGE.repository, PACKAGE.role)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        return self._fail(Operation(tool=tool, args=args, inputs=[], outputs=[], kind="measure"), 1, False, time.time(), 2, "INVALID_REQUEST",
                          "remotion-ffmpeg-skill is a plan step (it renders), never a measurement")

    def _within(self, root: str, p: str) -> bool:
        return os.path.normcase(p) == os.path.normcase(root) or os.path.normcase(p).startswith(os.path.normcase(root).rstrip(os.sep) + os.sep)

    def _user_path(self, raw: Any, what: str, paths: Dict[str, str], directory: bool = False) -> str:
        p = paths.get(raw, raw) if isinstance(raw, str) else raw
        if not isinstance(p, str) or not p.strip() or "\x00" in p or "\n" in p:
            raise ToolError(f"{what} must be a non-empty single-line path")
        resolved = os.path.realpath(os.path.abspath(p))
        roots = self.allowed_inputs + ([self.workspace] if self.workspace else [])
        if roots and not any(self._within(r, resolved) for r in roots):
            raise ToolError(f"{what} is outside the allowed input roots: {os.path.basename(p)}")
        if (directory and not os.path.isdir(resolved)) or (not directory and not os.path.isfile(resolved)):
            raise ToolError(f"{what} not found: {os.path.basename(p)}")
        return resolved

    def build(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None) -> Dict[str, Any]:
        args = dict(op.args)
        if op.tool != TOOL_PIPELINE:
            raise ToolError(f"{SKILL_ID} adapter does not support {op.tool}")
        unknown = sorted(k for k in args if k not in SENT + ("input", "output"))
        if unknown:
            raise ToolError(f"unknown pipeline argument(s) {unknown} (the agent sends only {', '.join(SENT)}: delivery, brand and chapters are its own)")
        src = self._user_path(args.get("input"), "input", paths)
        comp = args.get("comp")
        if comp not in COMPOSITIONS:
            raise ToolError(f"comp {comp!r} is not one of {', '.join(COMPOSITIONS)}" + (" (it cuts the picture: its length is not known before the render)" if comp in CUTTING_COMPOSITIONS else ""))
        project = self._user_path(args.get("project"), "project", paths, directory=True)
        if not os.path.isfile(os.path.join(project, "package.json")):
            raise ToolError(f"project {os.path.basename(project)} is not a Remotion project (no package.json: make it with init.py)")
        out = paths.get(args.get("output"), args.get("output")) if isinstance(args.get("output"), str) else None
        if not out:
            raise ToolError("output missing")
        out = os.path.realpath(os.path.abspath(out))
        if not self.workspace or not self._within(self.workspace, out):
            raise ToolError(f"output is outside the workspace: {os.path.basename(out)}")
        if os.path.basename(out) != render_name(src, comp):
            raise ToolError(f"output {os.path.basename(out)} is not where pipeline.py writes the render ({render_name(src, comp)})")
        if self.policy is not None:
            self.policy.check_output(out, [src])
        argv = [src, f"--project={project}", f"--comp={comp}"]
        for k in FILE_KEYS:
            if args.get(k) is not None:
                argv.append(f"--{k.replace('_', '-')}={self._user_path(args[k], k, paths)}")
        for k in ("beats", "conform", "fast"):
            if args.get(k) is not None:
                if not isinstance(args[k], bool):
                    raise ToolError(f"{k} must be true / false")
                if args[k]:
                    argv.append(f"--{k}")
        t = max(1.0, min(86400.0, float(timeout or self.default_timeout)))
        argv += [f"--out-dir={os.path.dirname(out)}", "--json", f"--timeout={t:g}"]
        return {"argv": argv, "output": out, "timeout": t, "comp": comp}

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            return [" ".join(["pipeline.py"] + self.build(op, paths)["argv"])]
        except ToolError as e:
            return [f"<refused: {e}>"]

    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, exit_code: int, code: str, message: str, retryable: bool = False) -> ToolResult:
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "failed",
                                "error": {"code": code, "message": message[:1200], "retryable": bool(retryable), "exit_code": exit_code, "recovery_class": recovery_class(code)}}
        return ToolResult(op.id, op.tool, False, exit_code, None, data, [], f"{SKILL_ID} [{code}] {message[:600]}", round(time.time() - t0, 3), attempt, dry_run)

    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        try:
            b = self.build(op, paths, timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e))
        command = " ".join(["pipeline.py"] + b["argv"])
        if dry_run:
            return ToolResult(op.id, op.tool, True, 0, None, {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run", "commands": [command]},
                              [command], "", round(time.time() - t0, 3), attempt, True)
        Path(b["output"]).parent.mkdir(parents=True, exist_ok=True)
        code, out, err = self._call("pipeline", b["argv"], b["timeout"] * 3 + PROCESS_MARGIN)   # measure + draw, each within its own --timeout
        tail = "\n".join(err.strip().splitlines()[-12:])
        if code == 124:
            return self._fail(op, attempt, dry_run, t0, 124, "TIMEOUT", "remotion-ffmpeg-skill pipeline exceeded its time limit", retryable=True)
        try:
            doc = one_json_document(out, "remotion-ffmpeg-skill pipeline")
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, code if code else 9, "INVALID_OUTPUT", f"{e} (exit {code}) {tail[-300:]}".strip())
        if doc.get("status") != "completed":
            kind = str(doc.get("kind") or "failed")
            return self._fail(op, attempt, dry_run, t0, code if code else 1, kind if kind in ERROR_CLASSES else "failed", str(doc.get("error") or tail or "")[:1200],
                              retryable=kind == "timeout")
        why: List[str] = []
        if doc.get("verified") is not True:
            why.append("the render is not verified against its timeline")
        if not same_file(str(doc.get("render") or ""), b["output"]) or not same_file(str(doc.get("output") or ""), b["output"]):
            why.append(f"render {os.path.basename(str(doc.get('render') or ''))!r} is not the planned {os.path.basename(b['output'])} (no delivery was asked)")
        if not os.path.isfile(b["output"]) or os.path.getsize(b["output"]) == 0:
            why.append("the render is missing or empty although the Skill reported success")
        steps = [as_dict(s).get("step") for s in doc.get("steps") or []]
        if steps[-1:] != ["draw"]:
            why.append(f"the steps {steps} do not end with the draw")
        if why or code != 0:
            return self._fail(op, attempt, dry_run, t0, code if code else 9, "INVALID_OUTPUT", "remotion-ffmpeg-skill result not admitted: " + "; ".join(why or [f"exit {code} with a completed result"]))
        data = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "ok", "comp": b["comp"], "render": b["output"], "verified": True,
                "look": doc.get("look"), "warnings": [str(w)[:600] for w in doc.get("warnings") or []],
                "steps": [{k: as_dict(s).get(k) for k in ("step", "status")} for s in doc.get("steps") or []], "commands": [command]}
        return ToolResult(op.id, op.tool, True, 0, b["output"], data, [command], tail, round(time.time() - t0, 3), attempt, False)


def comp_ok(comp: Any) -> bool:
    return isinstance(comp, str) and re.match(r"^[A-Za-z][A-Za-z0-9]{0,63}$", comp) is not None and comp in COMPOSITIONS
