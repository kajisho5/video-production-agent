"""SoundDesignAdapter: the agent's boundary to sound-design-skill (ADR-092; sound-design-skill roadmap step 5, agent side: BGM -> MIX).

    video-production-agent ─(plan step bgm_library / bgm_generate)─→ SoundDesignAdapter ─(argv lists, never a shell)─→
        `sound-design library-pick --json --output=<job>/ops/…/bgm_bed.wav --kind=bgm [--tags=a,b] [--mood=a,b] [--file=<relative>]
         [--crossfade=S] --duration=<the mix's length> --ffmpeg-skill=<dir> --timeout=T -- <library dir>`   (no engine: the user's licensed files)
        `sound-design generate-bgm --json --output=<…>.wav --prompt=<text> [--tags=a,b] --duration=<the mix's length> [--seed=N] --offline
         --ffmpeg-skill=<dir> --timeout=T`                                        (ACE-Step 1.5, optional install, never downloaded)
    → the WAV becomes one input of audio-production-skill's MIX (the bed; ducked under input 0 when the user asks)

Protocol (sound-design-skill's own contract, `sound-design contract --json`, pinned from main in contract_0.1.0.json -- no release tag): each op
prints one ASCII JSON document (`sound-design/library-pick@1` / `sound-design/generate-bgm@1`: output {path, format, codec, sample_rate, channels,
duration, sha256}, the request, the licence as declared or the engine's terms, provenance, warnings); failure {ok: false, error: {code, message,
retryable, details}} with the code fixing the exit code. doctor → `checks.operations.{bgm, library}.ready`, never downloads.

Nothing here mixes, ducks or normalises (audio-production MIX does). Every value is checked before the process starts (the library inside the
allowed roots, the output a new WAV in the workspace, duration 1..3600 s, tags / moods plain words, a prompt of one line under 512 characters,
seed 0..2**32-1). The result is admitted only when it is about this request: schema, op, Skill version, the output the agent named, its sha256
equal to the file, its duration within the Skill's own tolerance (max(0.1 s, 1 %)) of the length asked, a usable declared licence (library) or
an engine named (generated). The Skill's error codes are kept verbatim with the agent's recovery class."""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...models import Observation, Operation, ToolResult
from ...skills.contract import SkillPackage, ToolSpec
from ..base import ToolAdapter, ToolError
from ..skill_process import (FORBIDDEN_ARG_KEYS, CliSkill, ContractError, as_dict, breaking_drift as _breaking_drift, drift_report, invoke, one_json_document,
                             same_file, scan_forbidden, scrub, sha256_file)
from .locate import PACKAGE_ID, locate_sound_design  # noqa: F401  (re-exported)

SKILL_ID = PACKAGE_ID
PREFIX = SKILL_ID + "/"
TOOL_LIBRARY_PICK, TOOL_GENERATE_BGM = "sound-design/library-pick", "sound-design/generate-bgm"
SCHEMAS = {TOOL_LIBRARY_PICK: "sound-design/library-pick@1", TOOL_GENERATE_BGM: "sound-design/generate-bgm@1"}
RESULT_OP = {TOOL_LIBRARY_PICK: "library-pick", TOOL_GENERATE_BGM: "bgm"}
COMMAND = {TOOL_LIBRARY_PICK: "library-pick", TOOL_GENERATE_BGM: "generate-bgm"}
SUPPORTED_SKILL_VERSIONS = ("0.1.",)
# the per-operation capability `sound-design:<TYPE>` (resolver) ← the doctor's checks.operations.<op>.ready
OPERATION_TYPES = {"library": "LIBRARY", "bgm": "BGM_GENERATE"}
DURATION = (1.0, 3600.0)
MAX_PROMPT = 511
WORD_RE = re.compile(r"^[^,\x00-\x1f]{1,64}$")
FILE_RE = re.compile(r"^(?!/)(?!.*(^|/)\.\.(/|$))[^\x00-\x1f]{1,255}$")
DEFAULT_TIMEOUT = 3600.0
PROCESS_MARGIN = 120.0
ERROR_CLASSES: Dict[str, str] = {
    "INVALID_ARGUMENT": "INVALID_ARGS", "INPUT_NOT_FOUND": "INPUT_MISSING", "UNSUPPORTED_MEDIA": "INVALID_ARGS", "OUTPUT_EXISTS": "INVALID_ARGS",
    "OUTPUT_IS_INPUT": "INVALID_ARGS", "FFMPEG_SKILL_UNAVAILABLE": "TOOL_MISSING", "ENGINE_UNAVAILABLE": "TOOL_MISSING", "MODEL_NOT_FOUND": "TOOL_MISSING",
    "TOOL_ERROR": "SKILL_ERROR", "ENGINE_FAILED": "SKILL_ERROR", "ENGINE_TIMEOUT": "TIMEOUT", "INVALID_RESULT": "SKILL_ERROR", "NOT_IMPLEMENTED": "TOOL_MISSING",
    "CANCELLED": "INTERRUPTED", "INTERNAL_ERROR": "SKILL_ERROR", "CONFIRMATION_REQUIRED": "INVALID_ARGS", "LICENCE_REQUIRED": "INVALID_ARGS",
}
AGENT_ERRORS: Dict[str, str] = {"INVALID_REQUEST": "INVALID_ARGS", "INVALID_OUTPUT": "SKILL_ERROR", "TIMEOUT": "TIMEOUT"}
USED_TOOLS = (TOOL_LIBRARY_PICK, TOOL_GENERATE_BGM)
DRIFT_KEYS = ("skill_id", "version", "contract_version", "provides", "dependencies", "not_provided", "handoff", "engines", "policy", "provenance", "execution", "errors")
DRIFT_TOOL_KEYS = ("role", "writes_media", "mutates_input", "deterministic", "argv", "schema", "confirmation_required")
FATAL_DRIFT_KEYS = ("skill_id", "contract_version", "execution")
FATAL_DRIFT_TOOL_KEYS = ("role", "mutates_input", "argv", "schema", "confirmation_required")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def recovery_class(code: str) -> str:
    return ERROR_CLASSES.get(code) or AGENT_ERRORS.get(code) or "SKILL_ERROR"


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def tolerance(duration: float) -> float:
    """The Skill's own duration tolerance: max(0.1 s, 1 % of the requested duration)."""
    return max(0.1, 0.01 * float(duration))


def check_contract(contract: Any) -> List[str]:
    """What the adapter relies on, refused (never patched) when it does not hold: identity, contract and Skill version, library-pick (selection)
    and generate-bgm (generation) writing a new file without touching their input and asking no confirmation, their result schemas, a shell-free
    offline execution with a read-only input, and every error code the adapter maps with its exit code."""
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
    tools = {str(t.get("tool_id")): t for t in contract.get("tools") or [] if isinstance(t, dict)}
    for tid, role in ((TOOL_LIBRARY_PICK, "selection"), (TOOL_GENERATE_BGM, "generation")):
        t = tools.get(tid)
        if t is None:
            errs.append(f"contract lacks tool {tid} (declares {sorted(tools)})")
        elif t.get("role") != role or t.get("mutates_input") is not False or t.get("writes_media") is not True or t.get("schema") != SCHEMAS[tid] \
                or t.get("confirmation_required") not in (None, False) or t.get("implemented") is not True:
            errs.append(f"tool {tid} must declare role {role}, writes_media true, mutates_input false, schema {SCHEMAS[tid]}, no confirmation, implemented")
    ex = as_dict(contract.get("execution"))
    if ex.get("shell") is not False or ex.get("network") is not False or ex.get("input_is_read_only") is not True:
        errs.append("execution must declare shell false, network false and input_is_read_only true")
    errors = as_dict(contract.get("errors"))
    for code in ERROR_CLASSES:
        if not isinstance(as_dict(errors.get(code)).get("exit_code"), int):
            errs.append(f"errors lack {code} (with its exit_code)")
    return errs


def _used(c: Dict[str, Any]) -> Dict[str, Any]:
    return dict(c, tools=[t for t in c.get("tools") or [] if isinstance(t, dict) and t.get("tool_id") in USED_TOOLS])


def drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(live, pinned or pinned_contract(), DRIFT_KEYS, "tools", "tool_id", DRIFT_TOOL_KEYS)


def pinned_error_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    p, lv = as_dict((pinned or pinned_contract()).get("errors")), as_dict(live.get("errors"))
    return [f"errors {c}: pinned {json.dumps(p[c], sort_keys=True)} != live {json.dumps(lv.get(c), sort_keys=True)}" for c in sorted(p) if lv.get(c) != p[c]]


def breaking_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    pinned = pinned or pinned_contract()
    return _breaking_drift(_used(live), _used(pinned), FATAL_DRIFT_KEYS, "tools", "tool_id", FATAL_DRIFT_TOOL_KEYS) + pinned_error_drift(live, pinned)


PACKAGE = SkillPackage(
    skill_id=SKILL_ID, name="sound-design-skill", version="",
    description="Music beds (BGM) as WAV with full provenance, for audio-production MIX: chosen, trimmed and looped from the user's own library with a "
                "declared licence (no engine), or generated locally by ACE-Step 1.5 (optional, never downloaded) (EXPERIMENTAL)",
    capabilities=["sound-design"],
    tools=[ToolSpec(TOOL_LIBRARY_PICK, SKILL_ID, description="One library BGM file fitted to a duration (declared licence, tags / mood), as a WAV",
                    required_capabilities=["sound-design:LIBRARY"], inputs=["library"], produces_output=True, deterministic=True,
                    result_keys=["outputs", "licence", "selection", "fit", "warnings"]),
           ToolSpec(TOOL_GENERATE_BGM, SKILL_ID, description="A generated music bed of a duration from a prompt (ACE-Step 1.5), as a WAV",
                    required_capabilities=["sound-design:BGM_GENERATE"], inputs=[], produces_output=True, deterministic=False,
                    result_keys=["outputs", "engine", "request", "warnings"])],
    repository="kajisho5/sound-design-skill", role="music beds for the mix (library with declared licences, or generated)")


class SoundDesignAdapter(ToolAdapter):
    def __init__(self, skill: CliSkill, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None, ffmpeg_skill_dir: Optional[str] = None,
                 path_policy: Any = None, timeout: float = DEFAULT_TIMEOUT):
        self.skill = skill
        self.name = SKILL_ID
        self.workspace = os.path.realpath(os.path.abspath(workspace)) if workspace else None
        self.allowed_inputs = [os.path.realpath(os.path.abspath(p)) for p in (allowed_inputs or [])]
        self.ffmpeg_skill_dir = os.path.realpath(os.path.abspath(ffmpeg_skill_dir)) if ffmpeg_skill_dir else None
        self.policy = path_policy
        self.default_timeout = float(timeout)
        self.contract = self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("incompatible sound-design-skill contract: " + "; ".join(errs))
        self.version = str(self.contract.get("version"))
        self.tools = {t: True for t in USED_TOOLS}
        self.retryable = {c: bool(as_dict(v).get("retryable")) for c, v in as_dict(self.contract.get("errors")).items()}
        self._doctor: Optional[Dict[str, Any]] = None

    def _call(self, argv: List[str], timeout: Optional[float]) -> Tuple[int, str, str]:
        return invoke(self.skill, argv, timeout=timeout)

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._call(["contract", "--json"], 60.0)
        if code != 0:
            raise ContractError(f"sound-design contract --json failed (exit {code}): {err.strip()[-300:]}")
        return one_json_document(out, "sound-design contract")

    def _engine_argv(self) -> List[str]:
        return [f"--ffmpeg-skill={self.ffmpeg_skill_dir}"] if self.ffmpeg_skill_dir else []

    def doctor(self) -> Dict[str, Any]:
        """`sound-design doctor --json --offline --ffmpeg-skill=<the agent's checkout>`: its own status ok / degraded / fail (an engine missing is
        degraded: library mode needs none), with checks.operations.<op>.ready per op."""
        if self._doctor is not None:
            return self._doctor
        code, out, err = self._call(["doctor", "--json", "--offline"] + self._engine_argv(), 300.0)
        try:
            doc = one_json_document(out, "sound-design doctor")
        except ToolError as e:
            doc = {"ok": False, "status": "fail", "checks": {}, "problems": [f"doctor produced no document (exit {code}): {e} {err.strip()[-200:]}"]}
        status = str(doc.get("status") or ("ok" if doc.get("ok") is True else "fail"))
        if status not in ("ok", "degraded", "fail"):
            status = "fail"
        self._doctor = dict(doc, status=status, problems=[str(p) for p in doc.get("problems") or []], warnings=[str(w) for w in doc.get("warnings") or []])
        return self._doctor

    def operation_status(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        ops = as_dict(as_dict((doc or self.doctor()).get("checks")).get("operations"))
        return {typ: ("supported" if as_dict(ops.get(op)).get("ready") is True else "unavailable") for op, typ in OPERATION_TYPES.items()}

    def operation_evidence(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Why an op is not ready, in the Skill's own words: the engine's status (ENGINE_UNAVAILABLE / MODEL_NOT_FOUND) and its install line, or
        FFMPEG_SKILL_UNAVAILABLE for library mode."""
        doc = doc or self.doctor()
        checks = as_dict(doc.get("checks"))
        ops = as_dict(checks.get("operations"))
        out: Dict[str, Dict[str, Any]] = {}
        for op, typ in OPERATION_TYPES.items():
            o = as_dict(ops.get(op))
            ev: Dict[str, Any] = {"ready": o.get("ready") is True}
            if not ev["ready"]:
                eng = str(o.get("engine") or "")
                est = str(o.get("engine_status") or "")
                ev["code"] = "FFMPEG_SKILL_UNAVAILABLE" if not eng else ("MODEL_NOT_FOUND" if "MODEL" in est.upper() else "ENGINE_UNAVAILABLE")
                eng_check = as_dict(as_dict(checks.get("engines")).get(eng)) if eng else {}
                inst = eng_check.get("install") or as_dict(checks.get("ffmpeg_skill")).get("install")
                ev["install"] = (" ".join(str(x) for x in inst) if isinstance(inst, list) else str(inst or ""))[:600]
                ev["detail"] = "; ".join(str(p) for p in doc.get("problems") or [])[:300]
            out[typ] = ev
        return out

    def drift(self) -> List[str]:
        return drift(self.contract)

    def breaking_drift(self) -> List[str]:
        return breaking_drift(self.contract)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "skill": self.skill.describe(), "tools": list(USED_TOOLS), "drift": self.drift()}

    def package(self) -> SkillPackage:
        return SkillPackage(PACKAGE.skill_id, PACKAGE.name, self.version, PACKAGE.description, list(PACKAGE.capabilities),
                            [ToolSpec(**{**t.__dict__, "version": self.version}) for t in PACKAGE.tools], PACKAGE.repository, PACKAGE.role)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        return self._fail(Operation(tool=tool, args=args, inputs=[], outputs=[], kind="measure"), 1, False, time.time(), 2, "INVALID_REQUEST",
                          "sound-design is a plan step (it writes a WAV), never a measurement")

    # ---- request construction
    def _within(self, root: str, p: str) -> bool:
        return os.path.normcase(p) == os.path.normcase(root) or os.path.normcase(p).startswith(os.path.normcase(root).rstrip(os.sep) + os.sep)

    def _library(self, raw: Any, paths: Dict[str, str]) -> str:
        p = paths.get(raw, raw) if isinstance(raw, str) else raw
        if not isinstance(p, str) or not p.strip() or "\x00" in p or "\n" in p:
            raise ToolError("library must be a non-empty single-line path")
        resolved = os.path.realpath(os.path.abspath(p))
        if self.allowed_inputs and not any(self._within(r, resolved) for r in self.allowed_inputs):
            raise ToolError(f"library is outside the allowed input roots: {os.path.basename(p)}")
        if not os.path.isfile(os.path.join(resolved, "sound-library.json")):
            raise ToolError(f"{os.path.basename(p)} is not a sound-design library (no sound-library.json: import the files with their licence first)")
        return resolved

    def _output(self, raw: Any, paths: Dict[str, str]) -> str:
        p = paths.get(raw, raw) if isinstance(raw, str) else raw
        if not isinstance(p, str) or not p.lower().endswith(".wav"):
            raise ToolError("output must be a .wav path")
        resolved = os.path.realpath(os.path.abspath(p))
        if not self.workspace or not self._within(self.workspace, resolved) or resolved == self.workspace:
            raise ToolError(f"output is outside the workspace: {os.path.basename(p)}")
        if self.policy is not None:
            self.policy.check_output(resolved, [])
        return resolved

    @staticmethod
    def _words(v: Any, name: str) -> List[str]:
        vals = v.split(",") if isinstance(v, str) else list(v or [])
        out = [str(x).strip() for x in vals if str(x).strip()]
        if any(not WORD_RE.match(x) for x in out) or len(out) > 16:
            raise ToolError(f"{name} must be up to 16 plain words (no commas or control characters)")
        return out

    def build(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None) -> Dict[str, Any]:
        args = dict(op.args)
        hit = scan_forbidden(args, FORBIDDEN_ARG_KEYS)
        if hit:
            raise ToolError(f"forbidden key {hit}")
        allowed = {TOOL_LIBRARY_PICK: ("library", "tags", "mood", "file", "crossfade", "duration", "output"),
                   TOOL_GENERATE_BGM: ("prompt", "tags", "seed", "duration", "output")}.get(op.tool)
        if allowed is None:
            raise ToolError(f"{SKILL_ID} adapter does not support {op.tool}")
        unknown = sorted(k for k in args if k not in allowed)
        if unknown:
            raise ToolError(f"unknown {COMMAND[op.tool]} argument(s): {', '.join(unknown)}")
        dur = args.get("duration")
        if not _num(dur) or not DURATION[0] <= float(dur) <= DURATION[1]:
            raise ToolError(f"duration must be {DURATION[0]:g}..{DURATION[1]:g} s, got {dur!r}")
        out = self._output(args.get("output"), paths)
        t = max(1.0, min(86400.0, float(timeout or self.default_timeout)))
        tags = self._words(args.get("tags"), "tags")
        argv = [COMMAND[op.tool], "--json", f"--output={out}"]
        library = None
        if op.tool == TOOL_LIBRARY_PICK:
            library = self._library(args.get("library"), paths)
            mood = self._words(args.get("mood"), "mood")
            argv.append("--kind=bgm")
            if tags:
                argv.append("--tags=" + ",".join(tags))
            if mood:
                argv.append("--mood=" + ",".join(mood))
            if args.get("file") is not None:
                if not isinstance(args["file"], str) or not FILE_RE.match(args["file"]):
                    raise ToolError("file must be a path inside the library (relative, no '..')")
                argv.append(f"--file={args['file']}")
            if args.get("crossfade") is not None:
                if not _num(args["crossfade"]) or not 0 <= float(args["crossfade"]) <= 10:
                    raise ToolError("crossfade must be 0..10 s")
                argv.append(f"--crossfade={float(args['crossfade']):g}")
            argv.append(f"--duration={float(dur):g}")
        else:
            prompt = args.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT or any(ord(c) < 32 for c in prompt):
                raise ToolError(f"prompt must be one line of 1..{MAX_PROMPT} characters")
            argv.append(f"--prompt={prompt.strip()}")
            if tags:
                argv.append("--tags=" + ",".join(tags))
            argv.append(f"--duration={float(dur):g}")
            if args.get("seed") is not None:
                s = args["seed"]
                if isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2 ** 32:
                    raise ToolError("seed must be an integer in [0, 2**32)")
                argv.append(f"--seed={s}")
            argv.append("--offline")
        argv += self._engine_argv() + [f"--timeout={t:g}"]
        if library:
            argv += ["--", library]
        return {"argv": argv, "output": out, "duration": float(dur), "library": library, "timeout": t}

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            return [" ".join(["sound-design"] + self.build(op, paths)["argv"])]
        except ToolError as e:
            return [f"<refused: {e}>"]

    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, exit_code: int, code: str, message: str, retryable: bool = False,
              details: Optional[Dict[str, Any]] = None) -> ToolResult:
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "failed",
                                "error": {"code": code, "message": message[:1200], "retryable": bool(retryable), "details": dict(details or {}), "exit_code": exit_code,
                                          "recovery_class": recovery_class(code)}}
        return ToolResult(op.id, op.tool, False, exit_code, None, data, [], f"{SKILL_ID} [{code}] {message[:600]}", round(time.time() - t0, 3), attempt, dry_run)

    @staticmethod
    def _remove_fresh(path: str, t0: float) -> None:
        try:
            if os.path.isfile(path) and os.path.getmtime(path) >= t0 - 1:
                os.remove(path)
        except OSError:
            pass

    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        try:
            b = self.build(op, paths, timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e))
        command = " ".join(["sound-design"] + b["argv"])
        if dry_run:
            return ToolResult(op.id, op.tool, True, 0, None, {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run", "commands": [command]},
                              [command], "", round(time.time() - t0, 3), attempt, True)
        Path(b["output"]).parent.mkdir(parents=True, exist_ok=True)
        code, out, err = self._call(b["argv"], b["timeout"] + PROCESS_MARGIN)
        tail = "\n".join(err.strip().splitlines()[-12:])
        if code == 124:
            self._remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, 124, "TIMEOUT", f"sound-design {COMMAND[op.tool]} exceeded {b['timeout'] + PROCESS_MARGIN:g}s", retryable=True)
        try:
            doc = one_json_document(out, f"sound-design {COMMAND[op.tool]}")
        except ToolError as e:
            self._remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code else 9, "INVALID_OUTPUT", f"{e} (exit {code}) {tail[-300:]}".strip())
        if doc.get("ok") is not True:
            e_doc = as_dict(doc.get("error"))
            ecode = str(e_doc.get("code") or "INTERNAL_ERROR")
            details = scrub(e_doc.get("details"), FORBIDDEN_ARG_KEYS)
            retry = (bool(e_doc["retryable"]) if isinstance(e_doc.get("retryable"), bool) else self.retryable.get(ecode, False)) if ecode in ERROR_CLASSES else False
            self._remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code else 1, ecode, str(e_doc.get("message") or tail or "")[:1200], retryable=retry, details=details)
        why = self._admit(op.tool, doc, b)
        if why or code != 0:
            self._remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code else 9, "INVALID_OUTPUT", "sound-design result not admitted: " + "; ".join(why or [f"exit code {code} with an ok result"]))
        data = self._success_data(op.tool, doc, b, command)
        return ToolResult(op.id, op.tool, True, 0, b["output"], data, [command], tail, round(time.time() - t0, 3), attempt, False)

    def _admit(self, tool: str, doc: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
        if doc.get("schema") != SCHEMAS[tool] or doc.get("op") != RESULT_OP[tool]:
            return [f"result {doc.get('schema')!r} / op {doc.get('op')!r} is not {SCHEMAS[tool]}"]
        errs: List[str] = []
        sk = as_dict(doc.get("skill"))
        if sk.get("id") != SKILL_ID or str(sk.get("version")) != self.version:
            errs.append(f"result skill {sk!r} is not {SKILL_ID}@{self.version}")
        o = as_dict(doc.get("output"))
        if not same_file(str(o.get("path") or ""), b["output"]):
            return errs + [f"output {os.path.basename(str(o.get('path') or ''))!r} is not the planned {os.path.basename(b['output'])}"]
        if not os.path.isfile(b["output"]) or os.path.getsize(b["output"]) == 0:
            return errs + ["the output is missing or empty although the Skill reported success"]
        if str(o.get("sha256") or "") != sha256_file(b["output"]):
            errs.append("the sha256 the Skill reported does not match the file on disk")
        if not _num(o.get("duration")) or abs(float(o["duration"]) - b["duration"]) > tolerance(b["duration"]):
            errs.append(f"output duration {o.get('duration')!r} is not {b['duration']:g} s within {tolerance(b['duration']):g} s")
        if not isinstance(o.get("channels"), int) or not isinstance(o.get("sample_rate"), int):
            errs.append("output channels / sample_rate missing")
        if tool == TOOL_LIBRARY_PICK:
            lic = as_dict(doc.get("licence"))
            if lic.get("usable") is not True or lic.get("status") != "declared" or not lic.get("licence"):
                errs.append("the picked file has no usable declared licence")
        elif not as_dict(doc.get("engine")).get("id"):
            errs.append("the result names no engine")
        return errs

    def _success_data(self, tool: str, doc: Dict[str, Any], b: Dict[str, Any], command: str) -> Dict[str, Any]:
        """What the agent records: the output's facts and sha256, the licence as declared (library; never cleared) or the engine's terms and the seed
        (generated), the selection / fit, and the Skill's warnings verbatim (to pass on: a non-commercial or credit term, an engine's output warning)."""
        o = as_dict(doc.get("output"))
        out = {"file": b["output"], **{k: o.get(k) for k in ("format", "codec", "sample_rate", "channels", "duration", "sha256")}}
        base: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "ok", "op": RESULT_OP[tool], "outputs": [out],
                                "warnings": [str(w)[:600] for w in doc.get("warnings") or []], "commands": [command],
                                "ai_generated": tool == TOOL_GENERATE_BGM}
        if tool == TOOL_LIBRARY_PICK:
            lic = as_dict(doc.get("licence"))
            base.update({"licence": {k: lic.get(k) for k in ("licence", "status", "declared_as", "verified", "author", "credit", "source")},
                         "selection": {"chosen": as_dict(doc.get("selection")).get("chosen"), "matched": as_dict(doc.get("selection")).get("matched")},
                         "fit": {k: as_dict(doc.get("fit")).get(k) for k in ("mode", "source", "target", "repeats", "crossfade")},
                         "source_sha256": as_dict(doc.get("source")).get("sha256")})
        else:
            req = as_dict(doc.get("request"))
            base.update({"engine": as_dict(doc.get("engine")), "request": {k: req.get(k) for k in ("prompt", "duration", "seed", "seed_source")}})
        return base


def lift_observation(result: ToolResult, asset_id: Optional[str] = None) -> Optional[Observation]:
    """What sound-design-skill reported about the bed it delivered, as an agent Observation (provenance only): the licence as declared (library) or the
    engine's terms and seed (generated), the fit, the output's facts and sha256, and the Skill's warnings verbatim (credits to keep with anything published)."""
    data = result.data if isinstance(result.data, dict) else {}
    if data.get("status") != "ok" or not data.get("outputs"):
        return None
    sk = as_dict(data.get("skill"))
    outs = [dict(o) for o in data.get("outputs") or [] if isinstance(o, dict)]
    body = {k: data.get(k) for k in ("op", "licence", "selection", "fit", "engine", "request", "warnings", "ai_generated") if data.get(k) is not None}
    body["outputs"] = [{k: v for k, v in o.items() if k != "file"} for o in outs]
    return Observation(kind=f"bgm.{data.get('op')}", asset_id=asset_id or result.op_id, source=f"{result.tool}@{sk.get('version') or ''}", data=body,
                       analyzer=f"{SKILL_ID}@{sk.get('version') or ''}", provenance="OBSERVED", skill=SKILL_ID, skill_version=str(sk.get("version") or ""),
                       tool=result.tool, fingerprint=str((outs[0] if outs else {}).get("sha256") or ""), parameters={"outputs": [o.get("file") for o in outs]})
