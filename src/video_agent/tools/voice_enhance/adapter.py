"""VoiceEnhanceAdapter: the agent's boundary to voice-enhance-skill (ADR-073; voice-enhance-skill roadmap step 8, agent side).

    video-production-agent ─(typed Operation)─→ VoiceEnhanceAdapter ─(argv list, never a shell)─→
        `voice-enhance isolate|dereverb --json --offline --overwrite --ffmpeg-skill=<dir> --output=<out.wav> [--k=v ...] -- <input>`
        `voice-enhance bleed-reduce --json --offline --overwrite --ffmpeg-skill=<dir> --output=<dir> [--dominance-db=N] -- <track> <track> ...`
          ─(its own engine worker: deep-filter / nara_wpe / crosstalk; every media step through ffmpeg-skill)─→ WAV stem(s)

Protocol (voice-enhance-skill's own contract, `voice-enhance contract --json`, pinned from main in contract_0.1.0.json -- the
repository has no release tag, so the pin follows main as ingest-skill's does, ADR-066):
    contract     → `voice-enhance contract --json`
    doctor       → `voice-enhance doctor --json --offline --ffmpeg-skill=<dir>` (voice-enhance/doctor@1: python / ffmpeg_skill / ffmpeg / engines /
                   operations.<op>.ready, never downloads)
    isolate      → voice-enhance/isolate@1: a 48 kHz mono WAV, sample-aligned (DeepFilterNet3, `deep-filter -D`), ~30 ms short at the end
    dereverb     → voice-enhance/dereverb@1: a float WAV at the input's rate and channels, sample-aligned (WPE, nara_wpe)
    bleed-reduce → voice-enhance/bleed-reduce@1: one mono float WAV per track, `<out_dir>/<input name>_a0.wav` (crosstalk, no weights)
    failure      → {"ok": false, "error": {code, message, retryable, details}}, the code fixing the exit code (its errors table)

Every value is checked before the process starts: the inputs are artifact ids resolved through the paths map to existing files inside
the allowed roots, the outputs land in the workspace and never on an input, the numbers are inside the ranges voice-enhance-skill's
own CLI checks (ADR-003 / ADR-004 / ADR-005), unknown keys and command-like keys are refused by name. Every option is one
`--flag=value` word and every position comes after `--` (ADR-069), so no value is ever read as an option. ffmpeg-skill is always
named explicitly (`--ffmpeg-skill=<the agent's checkout>`): the Skill's own fallback search includes `./vendor` and `../` of the
current directory, which the agent never lets a child use (ADR-069). The result is admitted only when it is about this request (schema,
op, Skill version, output path(s), the output hashes the Skill reports equal to the files on disk, ISOLATE's 48 kHz mono). The Skill's
error codes are kept verbatim and mapped onto the agent's recovery classes (ERROR_CLASSES)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...models import Observation, Operation, ToolResult
from ...skills.contract import SkillPackage, ToolSpec
from ..base import ToolAdapter, ToolError
from ..ffmpeg_skill.adapter import PathPolicy
from ..skill_process import (FORBIDDEN_ARG_KEYS, CliSkill, ContractError, as_dict, breaking_drift as _breaking_drift, drift_report, invoke, one_json_document,
                             same_file, scan_forbidden, scrub, sha256_file)
from .locate import PACKAGE_ID, locate_voice_enhance  # noqa: F401  (re-exported)

SKILL_ID = PACKAGE_ID
PREFIX = SKILL_ID + "/"
TOOL_ISOLATE = "voice-enhance/isolate"
TOOL_DEREVERB = "voice-enhance/dereverb"
TOOL_BLEED_REDUCE = "voice-enhance/bleed-reduce"
TOOLS = (TOOL_ISOLATE, TOOL_DEREVERB, TOOL_BLEED_REDUCE)
COMMAND = {TOOL_ISOLATE: "isolate", TOOL_DEREVERB: "dereverb", TOOL_BLEED_REDUCE: "bleed-reduce"}
SCHEMAS = {TOOL_ISOLATE: "voice-enhance/isolate@1", TOOL_DEREVERB: "voice-enhance/dereverb@1", TOOL_BLEED_REDUCE: "voice-enhance/bleed-reduce@1"}
RESULT_OP = {TOOL_ISOLATE: "isolate", TOOL_DEREVERB: "dereverb", TOOL_BLEED_REDUCE: "bleed_reduce"}
# the per-operation capability `voice-enhance:<TYPE>` (resolver) ← the doctor's `checks.operations.<op>` row, and the contract capability id
OPERATION_TYPES = {"isolate": "ISOLATE", "dereverb": "DEREVERB", "bleed_reduce": "BLEED_REDUCE"}
CAPABILITY_IDS = {"ISOLATE": "voice.isolate", "DEREVERB": "voice.dereverb", "BLEED_REDUCE": "voice.bleed_reduce"}
TOOL_OF_TYPE = {"ISOLATE": TOOL_ISOLATE, "DEREVERB": TOOL_DEREVERB, "BLEED_REDUCE": TOOL_BLEED_REDUCE}
SUPPORTED_SKILL_VERSIONS = ("0.1.",)
DOCTOR_SCHEMA = "voice-enhance/doctor@1"
ISOLATE_RATE = 48000            # ADR-003: DeepFilterNet3 works at 48 kHz; the stem is always 48 kHz mono
MAX_TRACKS = 8                  # ADR-005: 2..8 time-aligned microphone tracks
# voice-enhance-skill's own codes the planner quotes when it blocks before any engine runs (ADR-073): a stream that is not at the rate an op
# needs, with no `audio --sample-rate` in the located ffmpeg-skill; and the codes a not-ready operation would return (its doctor's verdict)
RATE_REFUSAL = "FFMPEG_SKILL_UNAVAILABLE"
DOCTOR_CODES = ("ENGINE_UNAVAILABLE", "MODEL_NOT_FOUND", "FFMPEG_SKILL_UNAVAILABLE")
# typed parameters per tool: name → (min, max, int?, min exclusive?) -- the ranges voice-enhance-skill's CLI checks itself
PARAMETERS: Dict[str, Dict[str, Tuple[float, float, bool, bool]]] = {
    TOOL_ISOLATE: {"atten_lim_db": (0.0, 100.0, False, True)},                                                        # ADR-003: 0 < N <= 100
    TOOL_DEREVERB: {"taps": (1, 50, True, False), "delay": (1, 20, True, False), "iterations": (1, 20, True, False)},   # ADR-004
    TOOL_BLEED_REDUCE: {"dominance_db": (0.0, 30.0, False, False)},                                                   # ADR-005
}
DEFAULT_TIMEOUT = 3600.0        # the Skill's own engine time limit (its --timeout default); the process gets a margin on top
PROCESS_MARGIN = 120.0
# voice-enhance-skill's error codes (its contract `errors`) → the agent's recovery class (execution/recovery.py). The code is kept verbatim;
# a missing engine / model / ffmpeg-skill capability is TOOL_MISSING (blocked: the user installs it), a bad request INVALID_ARGS, the Skill's
# own refusal of its output SKILL_ERROR. TOOL_ERROR and ENGINE_TIMEOUT keep the Skill's retryable verdict (one more attempt, the second
# with a doubled timeout); CANCELLED by a signal is never retried.
ERROR_CLASSES: Dict[str, str] = {
    "INVALID_ARGUMENT": "INVALID_ARGS", "INPUT_NOT_FOUND": "INPUT_MISSING", "UNSUPPORTED_MEDIA": "INVALID_ARGS", "OUTPUT_EXISTS": "INVALID_ARGS",
    "OUTPUT_IS_INPUT": "INVALID_ARGS", "FFMPEG_SKILL_UNAVAILABLE": "TOOL_MISSING", "ENGINE_UNAVAILABLE": "TOOL_MISSING", "MODEL_NOT_FOUND": "TOOL_MISSING",
    "TOOL_ERROR": "UNKNOWN", "ENGINE_FAILED": "SKILL_ERROR", "ENGINE_TIMEOUT": "TIMEOUT", "INVALID_RESULT": "SKILL_ERROR", "NOT_IMPLEMENTED": "SKILL_ERROR",
    "CANCELLED": "TIMEOUT", "INTERNAL_ERROR": "SKILL_ERROR",
}
# the agent's own refusals (not voice-enhance codes)
AGENT_ERRORS: Dict[str, str] = {"INVALID_REQUEST": "INVALID_ARGS", "INVALID_OUTPUT": "SKILL_ERROR", "TIMEOUT": "TIMEOUT"}
# contract drift (ADR-045 / ADR-047): every key the adapter reads is compared with the pinned contract; the fatal subset gates availability
DRIFT_KEYS = ("skill_id", "version", "contract_version", "provides", "dependencies", "not_provided", "engines", "provenance", "execution", "errors", "error_shape")
DRIFT_TOOL_KEYS = ("role", "writes_media", "mutates_input", "deterministic", "implemented", "engine", "argv", "schema", "input", "output", "result_keys")
# version: moves within SUPPORTED_SKILL_VERSIONS; provides (a lifecycle leaving EXPERIMENTAL) / dependencies / engines / not_provided / provenance:
# reported, and the engines' readiness is the doctor's verdict, not the contract's; errors: checked on every use (pinned_error_drift: a pinned
# code that vanished or changed its exit / retryable is fatal, a new code is not). What is left changes the wire or the files the adapter admits.
FATAL_DRIFT_KEYS = ("skill_id", "contract_version", "execution", "error_shape")
FATAL_DRIFT_TOOL_KEYS = ("role", "writes_media", "mutates_input", "argv", "schema", "output")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def recovery_class(code: str) -> str:
    """The agent's recovery class of a voice-enhance code or an agent code; an unknown code is SKILL_ERROR (kept verbatim, never retried blindly)."""
    return ERROR_CLASSES.get(code) or AGENT_ERRORS.get(code) or "SKILL_ERROR"


def check_contract(contract: Any) -> List[str]:
    """What the adapter relies on, refused (never patched) when it does not hold: identity, contract and Skill version, the three op
    tools as implemented transformations that write media and never touch their input, a shell-free offline execution with read-only
    inputs, the three capabilities, and every error code the adapter maps with its exit code."""
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
    for tid in TOOLS:
        t = tools.get(tid)
        if t is None:
            errs.append(f"contract lacks tool {tid} (declares {sorted(tools)})")
            continue
        if t.get("role") != "transformation" or t.get("writes_media") is not True or t.get("mutates_input") is not False or t.get("implemented") is not True:
            errs.append(f"tool {tid} must declare role transformation, writes_media true, mutates_input false, implemented true")
        if t.get("schema") != SCHEMAS[tid]:
            errs.append(f"tool {tid} schema {t.get('schema')!r} != {SCHEMAS[tid]}")
    iso = as_dict(as_dict(tools.get(TOOL_ISOLATE)).get("output"))
    if tools.get(TOOL_ISOLATE) and (iso.get("sample_rate") != ISOLATE_RATE or iso.get("channels") != 1):
        errs.append(f"tool {TOOL_ISOLATE} must write {ISOLATE_RATE} Hz mono")
    ex = as_dict(contract.get("execution"))
    if ex.get("shell") is not False or ex.get("network") is not False or ex.get("input_is_read_only") is not True:
        errs.append("execution must declare shell false, network false and input_is_read_only true")
    provided = {str(p.get("id")) for p in contract.get("provides") or [] if isinstance(p, dict)}
    for cap in CAPABILITY_IDS.values():
        if cap not in provided:
            errs.append(f"provides lacks {cap}")
    errors = as_dict(contract.get("errors"))
    for code in ERROR_CLASSES:
        if not isinstance(as_dict(errors.get(code)).get("exit_code"), int):
            errs.append(f"errors lack {code} (with its exit_code)")
    return errs


def lifecycle(contract: Dict[str, Any]) -> Dict[str, str]:
    """capability id → lifecycle (EXPERIMENTAL today) from the contract's `provides`."""
    return {str(p.get("id")): str(p.get("lifecycle")) for p in contract.get("provides") or [] if isinstance(p, dict)}


def _op_tools(c: Dict[str, Any]) -> Dict[str, Any]:
    """The contract with only the tools this adapter runs (remux / doctor / engines / contract are not called as operations)."""
    return dict(c, tools=[t for t in c.get("tools") or [] if isinstance(t, dict) and t.get("tool_id") in TOOLS])


def drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(live, pinned or pinned_contract(), DRIFT_KEYS, "tools", "tool_id", DRIFT_TOOL_KEYS)


def pinned_error_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    """A pinned error code that disappeared or changed its exit code / retryable verdict (a new code is additive: SKILL_ERROR, kept verbatim)."""
    p, lv = as_dict((pinned or pinned_contract()).get("errors")), as_dict(live.get("errors"))
    return [f"errors {c}: pinned {json.dumps(p[c], sort_keys=True)} != live {json.dumps(lv.get(c), sort_keys=True)}" for c in sorted(p) if lv.get(c) != p[c]]


def breaking_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    pinned = pinned or pinned_contract()
    return _breaking_drift(_op_tools(live), _op_tools(pinned), FATAL_DRIFT_KEYS, "tools", "tool_id", FATAL_DRIFT_TOOL_KEYS) + pinned_error_drift(live, pinned)


PACKAGE = SkillPackage(
    skill_id=SKILL_ID, name="voice-enhance-skill", version="",
    description="Speech enhancement of recordings, local and offline: voice isolation (DeepFilterNet3), dereverberation (WPE, nara_wpe) and mic-bleed "
                "reduction of time-aligned tracks (signal processing); every media step through ffmpeg-skill; all three EXPERIMENTAL",
    capabilities=["voice-enhance"],
    tools=[ToolSpec(TOOL_ISOLATE, SKILL_ID, description="Voice stem with noise and background removed (48 kHz mono WAV, sample-aligned)",
                    inputs=["input"], produces_output=True, deterministic=True, result_keys=["output", "engine", "provenance", "warnings"]),
           ToolSpec(TOOL_DEREVERB, SKILL_ID, description="Speech with room reverberation reduced (float WAV, same rate and channels, sample-aligned)",
                    inputs=["input"], produces_output=True, deterministic=True, result_keys=["output", "engine", "provenance", "warnings"]),
           ToolSpec(TOOL_BLEED_REDUCE, SKILL_ID, description="Per-microphone tracks with the other microphones' cross-talk reduced (one mono float WAV per track)",
                    inputs=["inputs"], produces_output=True, deterministic=True, result_keys=["outputs", "engine", "provenance", "warnings"])],
    repository="kajisho5/voice-enhance-skill", role="speech enhancement (AI voice isolation, dereverberation, mic-bleed reduction)")


def _number(v: Any, name: str, spec: Tuple[float, float, bool, bool]) -> str:
    lo, hi, integer, open_lo = spec
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        raise ToolError(f"{name} must be a number, got {v!r}")
    if integer and float(v) != int(v):
        raise ToolError(f"{name} must be an integer, got {v!r}")
    if (v <= lo if open_lo else v < lo) or v > hi:
        raise ToolError(f"{name} {v!r} must be within {'(' if open_lo else '['}{lo:g}, {hi:g}]")
    return str(int(v)) if integer else repr(float(v))


def stem_name(path: str) -> str:
    """The file voice-enhance bleed-reduce writes for one track given as its own file (stream 0): `<input name without extension>_a0.wav`."""
    return f"{os.path.splitext(os.path.basename(path))[0]}_a0.wav"


class VoiceEnhanceAdapter(ToolAdapter):
    owns_cache = False

    def __init__(self, skill: CliSkill, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None, ffmpeg_skill_dir: Optional[str] = None,
                 path_policy: Optional[PathPolicy] = None, timeout: float = DEFAULT_TIMEOUT):
        self.skill = skill
        self.name = SKILL_ID
        self.workspace = os.path.realpath(os.path.abspath(workspace)) if workspace else None
        self.allowed_inputs = [os.path.realpath(os.path.abspath(p)) for p in (allowed_inputs or [])]
        # the engine location from the agent's own configuration, never from a request; without one nothing runs (see module doc)
        self.ffmpeg_skill_dir: Optional[str] = str(Path(ffmpeg_skill_dir).resolve()) if ffmpeg_skill_dir else None
        self.policy = path_policy
        self.default_timeout = float(timeout)
        self.contract = self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("incompatible voice-enhance-skill contract: " + "; ".join(errs))
        self.version = str(self.contract.get("version"))
        self.tools = {t: True for t in TOOLS}
        self.retryable = {c: bool(as_dict(v).get("retryable")) for c, v in as_dict(self.contract.get("errors")).items()}
        self.lifecycle = lifecycle(self.contract)
        self._doctor: Optional[Dict[str, Any]] = None

    # ---- contract / doctor
    def _call(self, argv: List[str], timeout: Optional[float]) -> Tuple[int, str, str]:
        return invoke(self.skill, argv, timeout=timeout)

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._call(["contract", "--json"], 60.0)
        if code != 0:
            raise ContractError(f"voice-enhance contract --json failed (exit {code}): {err.strip()[-300:]}")
        return one_json_document(out, "voice-enhance contract")

    def _engine_argv(self) -> List[str]:
        return [f"--ffmpeg-skill={self.ffmpeg_skill_dir}"] if self.ffmpeg_skill_dir else []

    def doctor(self) -> Dict[str, Any]:
        """`voice-enhance doctor --json --offline --ffmpeg-skill=<dir>` with `status` as the Skill reports it (ok: ffmpeg-skill usable and every
        engine OK; degraded: an engine MISSING / INCOMPATIBLE, the Skill itself works; fail: Python or ffmpeg-skill not usable). Without the
        agent's own ffmpeg-skill it is `fail` without running the Skill (the Skill would otherwise search the current directory)."""
        if self._doctor is not None:
            return self._doctor
        if not self.ffmpeg_skill_dir:
            self._doctor = {"schema": DOCTOR_SCHEMA, "ok": False, "status": "fail", "checks": {}, "problems": ["no ffmpeg-skill located by the agent: voice-enhance-skill runs every media step through it"],
                            "warnings": []}
            return self._doctor
        code, out, err = self._call(["doctor", "--json", "--offline"] + self._engine_argv(), 300.0)
        try:
            doc = one_json_document(out, "voice-enhance doctor")
        except ToolError as e:
            doc = {"schema": DOCTOR_SCHEMA, "ok": False, "status": "fail", "checks": {}, "problems": [f"doctor produced no document (exit {code}): {e} {err.strip()[-200:]}"]}
        if doc.get("schema") != DOCTOR_SCHEMA or doc.get("status") not in ("ok", "degraded", "fail"):
            doc = dict(doc, status="fail", problems=list(doc.get("problems") or []) + [f"doctor document {doc.get('schema')!r} / status {doc.get('status')!r} not understood"])
        doc = dict(doc, problems=list(doc.get("problems") or []), warnings=list(doc.get("warnings") or []))
        self._doctor = doc
        return doc

    def operation_status(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """`voice-enhance:<TYPE>` verdicts from the doctor's `checks.operations`: supported (ready) / unavailable (its engine is not OK)."""
        ops = as_dict(as_dict((doc or self.doctor()).get("checks")).get("operations"))
        return {typ: ("supported" if as_dict(ops.get(op)).get("ready") is True else "unavailable") for op, typ in OPERATION_TYPES.items()}

    def operation_evidence(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Why each operation is (not) ready, in the Skill's own words: the engine, its status, and -- when it is not OK -- the error code the
        Skill itself would return (ENGINE_UNAVAILABLE, or MODEL_NOT_FOUND when only the DeepFilterNet3 model is missing) and its install line.
        ISOLATE also carries whether the located ffmpeg-skill declares `audio --sample-rate` (without it only 48 kHz input is accepted)."""
        doc = doc or self.doctor()
        checks = as_dict(doc.get("checks"))
        ops, engines = as_dict(checks.get("operations")), as_dict(checks.get("engines"))
        out: Dict[str, Dict[str, Any]] = {}
        for op, typ in OPERATION_TYPES.items():
            row = as_dict(ops.get(op))
            eng = as_dict(engines.get(str(row.get("engine") or "")))
            status = str(row.get("engine_status") or eng.get("status") or "UNKNOWN")
            ev: Dict[str, Any] = {"engine": row.get("engine"), "engine_status": status, "engine_version": eng.get("version"), "ready": row.get("ready") is True,
                                  "lifecycle": self.lifecycle.get(CAPABILITY_IDS[typ])}
            if row.get("ready") is not True:
                if checks.get("ffmpeg_skill") and as_dict(checks.get("ffmpeg_skill")).get("status") != "OK":
                    ev["code"] = "FFMPEG_SKILL_UNAVAILABLE"
                else:
                    ev["code"] = "MODEL_NOT_FOUND" if eng.get("missing") == "model" else "ENGINE_UNAVAILABLE"
                ev["detail"] = str(eng.get("detail") or "")[:300]
                ev["install"] = str(eng.get("install") or "")[:600]
            if typ == "ISOLATE":
                flags = as_dict(as_dict(as_dict(checks.get("ffmpeg_skill")).get("optional_flags")).get("audio"))
                ev["resample"] = flags.get("sample_rate") is True
            out[typ] = ev
        return out

    def drift(self) -> List[str]:
        return drift(self.contract)

    def breaking_drift(self) -> List[str]:
        """ADR-048: the subset of drift() that gates availability (FATAL_DRIFT_KEYS, the op tools' wire, the pinned error codes)."""
        return breaking_drift(self.contract)

    # ---- ToolAdapter
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "skill": self.skill.describe(), "tools": list(TOOLS), "lifecycle": dict(self.lifecycle),
                "ffmpeg_skill": self.ffmpeg_skill_dir, "drift": self.drift()}

    def package(self) -> SkillPackage:
        return SkillPackage(PACKAGE.skill_id, PACKAGE.name, self.version, PACKAGE.description, list(PACKAGE.capabilities),
                            [ToolSpec(**{**t.__dict__, "version": self.version}) for t in PACKAGE.tools], PACKAGE.repository, PACKAGE.role)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        raise ToolError("voice-enhance: a production Skill has no measurement tools (media-analysis-skill / ffmpeg-skill measure)")

    # ---- request construction (typed args → argv); nothing here reads a value as an option
    @staticmethod
    def _check_raw(raw: Any, what: str) -> None:
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw or "\n" in raw:
            raise ToolError(f"{what} must be a non-empty single-line path")
        if any(part == ".." for part in raw.replace("\\", "/").split("/")):
            raise ToolError(f"{what} path contains '..' (traversal)")

    def _within(self, root: str, p: str) -> bool:
        r, q = os.path.normcase(root), os.path.normcase(p)
        return q == r or q.startswith(r.rstrip(os.sep) + os.sep)

    def _input(self, ref: Any, paths: Dict[str, str]) -> str:
        if not isinstance(ref, str) or ref not in paths:
            raise ToolError(f"input {ref!r} is not an artifact id of this job")
        raw = paths[ref]
        self._check_raw(raw, "input")
        resolved = os.path.realpath(os.path.abspath(raw))
        roots = list(self.allowed_inputs) + ([self.workspace] if self.workspace else [])
        if roots and not any(self._within(r, resolved) for r in roots):
            raise ToolError(f"input is outside the allowed input roots: {os.path.basename(raw)}")
        if not os.path.isfile(resolved):
            raise ToolError(f"input not found: {os.path.basename(raw)}")
        if self.policy is not None:
            self.policy.check_input(resolved)
        return resolved

    def _output(self, ref: Any, paths: Dict[str, str], inputs: List[str], ext: Optional[str]) -> str:
        if not isinstance(ref, str) or ref not in paths:
            raise ToolError(f"output {ref!r} is not an artifact id of this job")
        raw = paths[ref]
        self._check_raw(raw, "output")
        absolute = os.path.join(os.path.realpath(os.path.dirname(os.path.abspath(raw))), os.path.basename(raw))
        if self.workspace and not (self._within(self.workspace, absolute) and absolute != self.workspace):
            raise ToolError(f"output is outside the workspace: {os.path.basename(raw)}")
        if ext and os.path.splitext(absolute)[1].lower() != ext:
            raise ToolError(f"output must be a {ext} file: {os.path.basename(raw)}")
        for i in inputs:
            if same_file(i, absolute) or os.path.normcase(i) == os.path.normcase(absolute):
                raise ToolError("output would overwrite an input")
        if self.policy is not None:
            self.policy.check_output(absolute, inputs)
        return absolute

    def build(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None) -> Dict[str, Any]:
        """{"argv", "inputs", "outputs", "out_dir"} for one operation, or ToolError naming the first value that cannot be sent."""
        if op.tool not in TOOLS:
            raise ToolError(f"{SKILL_ID} adapter does not support {op.tool}")
        if not self.ffmpeg_skill_dir:
            raise ToolError("no ffmpeg-skill located by the agent: voice-enhance-skill runs every media step through it")
        args = dict(op.args)
        hit = scan_forbidden(args, FORBIDDEN_ARG_KEYS)
        if hit:
            raise ToolError(f"forbidden key {hit}")
        allowed = set(PARAMETERS[op.tool]) | ({"inputs", "outputs"} if op.tool == TOOL_BLEED_REDUCE else {"input", "output"})
        unknown = sorted(k for k in args if k not in allowed)
        if unknown:
            raise ToolError(f"unknown {op.tool} argument(s): {', '.join(unknown)}")
        options: List[str] = []
        for name, spec in PARAMETERS[op.tool].items():
            if args.get(name) is not None:
                options.append(f"--{name.replace('_', '-')}={_number(args[name], name, spec)}")
        t = max(1.0, min(86400.0, float(timeout or self.default_timeout)))
        if op.tool == TOOL_BLEED_REDUCE:
            ins, outs = args.get("inputs"), args.get("outputs")
            if not isinstance(ins, list) or not 2 <= len(ins) <= MAX_TRACKS or not isinstance(outs, list) or len(outs) != len(ins):
                raise ToolError(f"bleed-reduce takes 2..{MAX_TRACKS} inputs and one output per input")
            in_paths = [self._input(r, paths) for r in ins]
            if len({os.path.normcase(p) for p in in_paths}) != len(in_paths):
                raise ToolError("bleed-reduce: the same track given twice")
            out_paths = [self._output(r, paths, in_paths, ".wav") for r in outs]
            out_dir = os.path.dirname(out_paths[0])
            for p, o in zip(in_paths, out_paths):
                if os.path.dirname(o) != out_dir or os.path.basename(o) != stem_name(p):
                    raise ToolError(f"bleed-reduce output {os.path.basename(o)} is not where the Skill writes the track of {os.path.basename(p)} ({stem_name(p)} in one directory)")
            if any(os.path.dirname(p) == out_dir for p in in_paths):
                raise ToolError("bleed-reduce output directory holds an input")
            argv = [COMMAND[op.tool], "--json", "--offline", "--overwrite"] + self._engine_argv() + [f"--output={out_dir}", f"--timeout={t:g}"] + options + ["--"] + in_paths
            return {"argv": argv, "inputs": in_paths, "outputs": out_paths, "out_dir": out_dir, "timeout": t}
        src = self._input(args.get("input"), paths)
        out = self._output(args.get("output"), paths, [src], ".wav")
        argv = [COMMAND[op.tool], "--json", "--offline", "--overwrite"] + self._engine_argv() + [f"--output={out}", f"--timeout={t:g}"] + options + ["--", src]
        return {"argv": argv, "inputs": [src], "outputs": [out], "out_dir": os.path.dirname(out), "timeout": t}

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            b = self.build(op, paths)
        except ToolError as e:
            return [f"<refused: {e}>"]
        return [" ".join(["voice-enhance"] + b["argv"])]

    # ---- execution
    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, exit_code: int, code: str, message: str, retryable: bool = False,
              details: Optional[Dict[str, Any]] = None) -> ToolResult:
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "failed",
                                "error": {"code": code, "message": message[:1200], "retryable": bool(retryable), "details": dict(details or {}), "exit_code": exit_code,
                                          "recovery_class": recovery_class(code)}}
        return ToolResult(op.id, op.tool, False, exit_code, None, data, [], f"{SKILL_ID} [{code}] {message[:600]}", round(time.time() - t0, 3), attempt, dry_run)

    @staticmethod
    def _remove_fresh(paths: List[str], t0: float) -> None:
        for p in paths:
            try:
                if os.path.isfile(p) and os.path.getmtime(p) >= t0 - 1:
                    os.remove(p)
            except OSError:
                pass

    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        try:
            b = self.build(op, paths, timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e))
        command = " ".join(["voice-enhance"] + b["argv"])
        if dry_run:   # the Skill has no dry run: the executor's dry run checks the request and runs nothing
            return ToolResult(op.id, op.tool, True, 0, None, {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run", "commands": [command]},
                              [command], "", round(time.time() - t0, 3), attempt, True)
        code, out, err = self._call(b["argv"], b["timeout"] + PROCESS_MARGIN)
        tail = "\n".join(err.strip().splitlines()[-12:])
        if code == 124:
            self._remove_fresh(b["outputs"], t0)
            return self._fail(op, attempt, dry_run, t0, 124, "TIMEOUT", f"voice-enhance {COMMAND[op.tool]} exceeded {b['timeout'] + PROCESS_MARGIN:g}s", retryable=True,
                              details={"reason": "timeout"})
        try:
            doc = one_json_document(out, f"voice-enhance {COMMAND[op.tool]}")
        except ToolError as e:
            self._remove_fresh(b["outputs"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code else 9, "INVALID_OUTPUT", f"{e} (exit {code}) {tail[-300:]}".strip())
        if doc.get("ok") is not True:
            e_doc = as_dict(doc.get("error"))
            ecode = str(e_doc.get("code") or "INTERNAL_ERROR")
            details = scrub(e_doc.get("details"), FORBIDDEN_ARG_KEYS)
            retry = bool(e_doc["retryable"]) if isinstance(e_doc.get("retryable"), bool) else self.retryable.get(ecode, False)
            if ecode == "CANCELLED":
                details["reason"] = "signal"   # interrupted, not timed out: a longer timeout would not help (recovery: INTERRUPTED)
            if ecode not in ERROR_CLASSES:
                retry = False
            self._remove_fresh(b["outputs"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code else 1, ecode, str(e_doc.get("message") or tail or "")[:1200], retryable=retry, details=details)
        why = self._admit(op.tool, doc, b)
        if why:
            self._remove_fresh(b["outputs"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code else 9, "INVALID_OUTPUT", "voice-enhance result not admitted: " + "; ".join(why))
        if code != 0:
            self._remove_fresh(b["outputs"], t0)
            return self._fail(op, attempt, dry_run, t0, code, "INVALID_OUTPUT", f"exit code {code} with an ok result")
        data = self._success_data(op.tool, doc, b, command)
        return ToolResult(op.id, op.tool, True, 0, b["outputs"][0], data, [command], tail, round(time.time() - t0, 3), attempt, False)

    def _admit(self, tool: str, doc: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
        """Why a result is not about this request (empty = admitted)."""
        errs: List[str] = []
        if doc.get("schema") != SCHEMAS[tool] or doc.get("op") != RESULT_OP[tool]:
            return [f"result {doc.get('schema')!r} / op {doc.get('op')!r} is not {SCHEMAS[tool]}"]
        sk = as_dict(doc.get("skill"))
        if sk.get("id") != SKILL_ID or str(sk.get("version")) != self.version:
            errs.append(f"result skill {sk!r} is not {SKILL_ID}@{self.version}")
        prov = as_dict(doc.get("provenance"))
        if tool == TOOL_BLEED_REDUCE:
            outs = [as_dict(o) for o in doc.get("outputs") or []]
            got = [str(o.get("path") or "") for o in outs]
            if len(got) != len(b["outputs"]) or not all(same_file(g, w) for g, w in zip(got, b["outputs"])):
                return errs + [f"outputs {[os.path.basename(g) for g in got]} are not the planned {[os.path.basename(w) for w in b['outputs']]}"]
            hashes = [as_dict(h).get("sha256") for h in prov.get("outputs") or []]
            facts = outs
        else:
            o = as_dict(doc.get("output"))
            if not same_file(str(o.get("path") or ""), b["outputs"][0]):
                return errs + [f"output {os.path.basename(str(o.get('path') or ''))!r} is not the planned {os.path.basename(b['outputs'][0])}"]
            hashes = [as_dict(prov.get("output")).get("sha256")]
            facts = [o]
        if len(hashes) != len(b["outputs"]):
            errs.append("provenance does not hash every output")
        for p, h in zip(b["outputs"], hashes):
            if not os.path.isfile(p) or os.path.getsize(p) == 0:
                errs.append(f"{os.path.basename(p)} missing or empty although the Skill reported success")
            elif str(h or "") != sha256_file(p):
                errs.append(f"{os.path.basename(p)}: the sha256 the Skill reported does not match the file on disk")
        for f in facts:
            if tool == TOOL_ISOLATE and (f.get("sample_rate") != ISOLATE_RATE or f.get("channels") != 1):
                errs.append(f"isolate stem is {f.get('sample_rate')} Hz / {f.get('channels')} ch, not {ISOLATE_RATE} Hz mono")
            if tool == TOOL_BLEED_REDUCE and f.get("channels") != 1:
                errs.append(f"bleed-reduce stem has {f.get('channels')} channels, not 1")
        if not as_dict(prov.get("engine")).get("id"):
            errs.append("provenance names no engine")
        return errs

    def _success_data(self, tool: str, doc: Dict[str, Any], b: Dict[str, Any], command: str) -> Dict[str, Any]:
        """What the agent records: the engine (attribution and licences, to keep with anything published), the parameters the engine ran
        with, the Skill's warnings verbatim, each output's facts and sha256, and the lifecycle (EXPERIMENTAL) -- never local paths beyond
        the outputs the agent itself named."""
        prov = as_dict(doc.get("provenance"))
        outs = [as_dict(o) for o in doc.get("outputs") or []] if tool == TOOL_BLEED_REDUCE else [as_dict(doc.get("output"))]
        hashes = [as_dict(h).get("sha256") for h in prov.get("outputs") or []] if tool == TOOL_BLEED_REDUCE else [as_dict(prov.get("output")).get("sha256")]
        outputs = [{"file": p, "sha256": h, **{k: o.get(k) for k in ("format", "codec", "sample_rate", "channels", "duration")}} for p, h, o in zip(b["outputs"], hashes, outs)]
        params = {k: v for k, v in as_dict(doc.get("parameters")).items() if k != "engine_argv"}
        cap = str(doc.get("capability") or CAPABILITY_IDS[OPERATION_TYPES[RESULT_OP[tool]]])
        return {"skill": {"id": SKILL_ID, "version": self.version}, "status": "ok", "op": RESULT_OP[tool], "capability": cap,
                "lifecycle": self.lifecycle.get(cap), "engine": as_dict(prov.get("engine")) or as_dict(doc.get("engine")), "parameters": params,
                "warnings": [str(w)[:600] for w in doc.get("warnings") or []], "outputs": outputs,
                "inputs_sha256": [as_dict(h).get("sha256") for h in prov.get("inputs") or []] if tool == TOOL_BLEED_REDUCE else [as_dict(prov.get("input")).get("sha256")],
                "ffmpeg_skill": as_dict(prov.get("ffmpeg_skill")), "network": prov.get("network"), "created_at": prov.get("created_at"), "commands": [command]}


def lift_observation(result: ToolResult, asset_id: Optional[str] = None) -> Optional[Observation]:
    """What voice-enhance-skill reported about the stem(s) it delivered, as an agent Observation (provenance only; never fed back into the
    analysis): the engine with its licences and attribution (to keep with anything published), the parameters it ran with, the Skill's
    warnings verbatim, each stem's facts and sha256, and the lifecycle (EXPERIMENTAL)."""
    data = result.data if isinstance(result.data, dict) else {}
    if data.get("status") != "ok" or not data.get("outputs"):
        return None
    sk = as_dict(data.get("skill"))
    eng = as_dict(data.get("engine"))
    outs = [dict(o) for o in data.get("outputs") or [] if isinstance(o, dict)]
    return Observation(kind=f"voice.{data.get('op')}", asset_id=asset_id or result.op_id, source=f"{result.tool}@{sk.get('version') or ''}",
                       data={"capability": data.get("capability"), "lifecycle": data.get("lifecycle"), "engine": eng, "parameters": as_dict(data.get("parameters")),
                             "warnings": list(data.get("warnings") or []), "outputs": outs, "inputs_sha256": list(data.get("inputs_sha256") or []),
                             "ffmpeg_skill": as_dict(data.get("ffmpeg_skill")), "network": data.get("network")},
                       analyzer=f"{eng.get('id') or SKILL_ID}@{eng.get('version') or ''}", provenance="OBSERVED", skill=SKILL_ID, skill_version=str(sk.get("version") or ""),
                       tool=result.tool, fingerprint=str((outs[0] if outs else {}).get("sha256") or ""), parameters={"outputs": [o.get("file") for o in outs]})
