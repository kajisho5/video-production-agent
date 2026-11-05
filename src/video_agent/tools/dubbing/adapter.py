"""DubbingAdapter: the agent's boundary to dubbing-skill's DUB (ADR-085; dubbing-skill roadmap step 9, agent side, as its ADR-012 hands it over).

    video-production-agent ─(typed Operation)─→ DubbingAdapter ─(argv list, never a shell)─→
        `dubbing dub --json --offline --overwrite --ffmpeg-skill=<dir> --output=<out.mp4> --lang=L [--k=v ...] -- <programme> <translated cues>`
          ─(its own engine worker; every media step through ffmpeg-skill)─→ the programme with its dialogue replaced

Protocol (dubbing-skill's own contract, `dubbing contract --json`, pinned from main in contract_0.1.0.json -- the repository has no
release tag, so the pin follows main as voice-enhance-skill's does, ADR-073):
    contract → `dubbing contract --json`
    doctor   → `dubbing doctor --json --offline --ffmpeg-skill=<dir>` (dubbing/doctor@1: python / ffmpeg_skill / engines /
               operations.<op>.ready, never downloads)
    dub      → dubbing/dub@1: output (path, kind, duration, audio / video duration, av_delta, tolerance), fit, mix, clips[],
               kept_original, obligations (credit[], credits[], watermark), provenance (engine, consent, output sha256), warnings
    failure  → {"ok": false, "error": {code, message, retryable, details}}, the code fixing the exit code (its errors table)

What the agent supplies and what it never does (dubbing-skill ADR-003 / ADR-009 / ADR-012):
- the translated cues are the user's file (the Skill never translates and neither does the agent); a translation note saying so goes
  with every call;
- a reference voice is passed only together with a consent record the user named (`--reference=… --consent=…`, both files inside the
  allowed input roots). The agent never writes, edits, fills in or signs a consent record, never passes a reference without one, and
  never retries a `CONSENT_REQUIRED`: that code is not retryable whatever the result says, and its recovery class blocks the job with
  the Skill's `details.reason` / `details.problems` for the user;
- `obligations.credit` of the result is carried into the ProductionReceipt (audit/receipt.py), so the published description or credits
  can name what the engine / voice requires.

Every value is checked before the process starts: the programme is an artifact id resolved through the paths map to an existing file
inside the allowed roots, the user's files exist inside those roots, the output lands in the workspace (never on an input) as a video
container the contract accepts, numbers are inside the contract's limits, enums are the contract's, unknown and command-like keys are
refused by name. Every option is one `--flag=value` word and every position comes after `--` (ADR-069). ffmpeg-skill is always named
explicitly (`--ffmpeg-skill=<the agent's checkout>`; the Skill's own fallback search includes `./vendor` and `../`). The result is
admitted only when it is about this request (schema, op, Skill version, output path, the output sha256 the Skill reports equal to the
file on disk, audio and picture within the Skill's own tolerance)."""
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
                             same_file, scan_forbidden, scrub, sha256_file)
from .locate import PACKAGE_ID, locate_dubbing  # noqa: F401  (re-exported)

SKILL_ID = PACKAGE_ID
PREFIX = SKILL_ID + "/"
TOOL_DUB = "dubbing/dub"
TOOLS = (TOOL_DUB,)
COMMAND = {TOOL_DUB: "dub"}
SCHEMAS = {TOOL_DUB: "dubbing/dub@1"}
RESULT_OP = {TOOL_DUB: "dub"}
# the per-operation capability `dubbing:<TYPE>` (resolver) ← the doctor's `checks.operations.<op>` row, and the contract capability id
OPERATION_TYPES = {"dub": "DUB"}
CAPABILITY_IDS = {"DUB": "speech.dub"}
TOOL_OF_TYPE = {"DUB": TOOL_DUB}
SUPPORTED_SKILL_VERSIONS = ("0.1.",)
DOCTOR_SCHEMA = "dubbing/doctor@1"
DOCTOR_CODES = ("ENGINE_UNAVAILABLE", "MODEL_NOT_FOUND", "FFMPEG_SKILL_UNAVAILABLE")
VIDEO_OUTPUTS = (".mp4", ".mov", ".mkv")   # the contract's `-o <out.mp4|out.mov|out.mkv|out.wav>`: the agent keeps the picture, so never .wav
CUES_EXTENSIONS = (".json", ".srt", ".vtt")  # CUES_INPUT
CONSENT_EXTENSIONS = (".json",)             # dubbing/consent@1 is a JSON file
REFERENCE_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus")
LANG_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8}){0,3}$")   # a BCP-47 tag; the engine's language list is the Skill's to check
VOICE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")      # a voice id (chatterbox built-in, Style-Bert-VITS2 MODEL[:SPEAKER[:STYLE]])
# the note every call carries: who translated the cues (the user; dubbing-skill ADR-003: translation is a caller input)
TRANSLATION_NOTE = {"provided_by": "user", "engine": None, "via": "video-production-agent dub.cues"}
CONSENT_CODE = "CONSENT_REQUIRED"
DEFAULT_TIMEOUT = 3600.0        # the Skill's own --timeout default; the process gets a margin on top
PROCESS_MARGIN = 120.0
# dubbing-skill's error codes (its contract `errors`) → the agent's recovery class (execution/recovery.py). The code is kept verbatim.
# CONSENT_REQUIRED has its own class (BLOCK, never retried: the record comes from the user, the agent never makes one); TERMS_UNVERIFIED
# and TIMING_OVERFLOW are the Skill's refusals of this request (SKILL_ERROR); a missing engine / model / ffmpeg-skill tool is TOOL_MISSING.
ERROR_CLASSES: Dict[str, str] = {
    "INVALID_ARGUMENT": "INVALID_ARGS", "INPUT_NOT_FOUND": "INPUT_MISSING", "UNSUPPORTED_MEDIA": "INVALID_ARGS", "OUTPUT_EXISTS": "INVALID_ARGS",
    "OUTPUT_IS_INPUT": "INVALID_ARGS", "TERMS_UNVERIFIED": "SKILL_ERROR", CONSENT_CODE: "CONSENT_REQUIRED", "FFMPEG_SKILL_UNAVAILABLE": "TOOL_MISSING",
    "ENGINE_UNAVAILABLE": "TOOL_MISSING", "MODEL_NOT_FOUND": "TOOL_MISSING", "TOOL_ERROR": "UNKNOWN", "ENGINE_FAILED": "SKILL_ERROR", "ENGINE_TIMEOUT": "TIMEOUT",
    "INVALID_RESULT": "SKILL_ERROR", "TIMING_OVERFLOW": "SKILL_ERROR", "NOT_IMPLEMENTED": "SKILL_ERROR", "CANCELLED": "TIMEOUT", "INTERNAL_ERROR": "SKILL_ERROR",
}
NEVER_RETRIED = (CONSENT_CODE,)   # dubbing-skill ADR-012: "not retryable"; held here whatever a result's `retryable` says
AGENT_ERRORS: Dict[str, str] = {"INVALID_REQUEST": "INVALID_ARGS", "INVALID_OUTPUT": "SKILL_ERROR", "TIMEOUT": "TIMEOUT"}
# contract drift (ADR-045 / ADR-047): every key the adapter reads is compared with the pinned contract; the fatal subset gates availability
DRIFT_KEYS = ("skill_id", "version", "contract_version", "provides", "dependencies", "inputs", "not_provided", "policy", "provenance", "execution", "errors", "error_shape")
DRIFT_TOOL_KEYS = ("role", "writes_media", "mutates_input", "deterministic", "implemented", "engine", "engines", "argv", "refused", "schema", "limits", "render", "output")
# version: moves within SUPPORTED_SKILL_VERSIONS; provides / dependencies / engines / inputs / not_provided / provenance: reported (the engines'
# readiness is the doctor's verdict); errors: checked on every use (a pinned code that vanished or changed its exit / retryable is fatal).
# policy is fatal: the consent gate (`policy.consent_record.bypass: null`) is what lets the agent pass a reference at all.
FATAL_DRIFT_KEYS = ("skill_id", "contract_version", "execution", "error_shape", "policy")
FATAL_DRIFT_TOOL_KEYS = ("role", "writes_media", "mutates_input", "argv", "schema", "limits")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def _limits(contract: Dict[str, Any]) -> Dict[str, Any]:
    t = next((t for t in contract.get("tools") or [] if isinstance(t, dict) and t.get("tool_id") == TOOL_DUB), {})
    return as_dict(t.get("limits"))


def dub_tool(contract: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    c = contract or pinned_contract()
    return next((t for t in c.get("tools") or [] if isinstance(t, dict) and t.get("tool_id") == TOOL_DUB), {})


def _range(lim: Dict[str, Any], key: str) -> Tuple[float, float]:
    r = as_dict(lim.get(key)).get("range") or []
    return float(r[0]), float(r[1])


def parameters(contract: Optional[Dict[str, Any]] = None) -> Dict[str, Tuple[float, float]]:
    """The numeric DUB options and their ranges, read from the contract's `limits` (never copied into this file)."""
    lim = _limits(contract or pinned_contract())
    return {"duck": _range(lim, "duck_db"), "voice_gain": _range(lim, "voice_gain_db"), "max_overrun": _range(lim, "max_overrun")}


def engines(contract: Optional[Dict[str, Any]] = None) -> Tuple[str, ...]:
    """The engines DUB takes (`tools[dubbing/dub].engines`): chatterbox and style_bert_vits2 at 0.1.0."""
    return tuple(str(e) for e in dub_tool(contract).get("engines") or [])


def unfit_policies(contract: Optional[Dict[str, Any]] = None) -> Tuple[str, ...]:
    """`--on-unfit fail|keep` as the contract's argv spells it."""
    for a in dub_tool(contract).get("argv") or []:
        m = re.match(r"^\[--on-unfit ([a-z|]+)\]$", str(a))
        if m:
            return tuple(m.group(1).split("|"))
    return ()


def recovery_class(code: str) -> str:
    """The agent's recovery class of a dubbing code or an agent code; an unknown code is SKILL_ERROR (kept verbatim, never retried blindly)."""
    return ERROR_CLASSES.get(code) or AGENT_ERRORS.get(code) or "SKILL_ERROR"


def check_contract(contract: Any) -> List[str]:
    """What the adapter relies on, refused (never patched) when it does not hold: identity, contract and Skill version, DUB as an implemented
    transformation that writes media and never touches its input, its limits and engines, a shell-free offline execution with read-only
    inputs, the consent gate with no bypass, the capability, and every error code the adapter maps with its exit code."""
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
    t = dub_tool(contract)
    if not t:
        errs.append(f"contract lacks tool {TOOL_DUB}")
    else:
        if t.get("role") != "transformation" or t.get("writes_media") is not True or t.get("mutates_input") is not False or t.get("implemented") is not True:
            errs.append(f"tool {TOOL_DUB} must declare role transformation, writes_media true, mutates_input false, implemented true")
        if t.get("schema") != SCHEMAS[TOOL_DUB]:
            errs.append(f"tool {TOOL_DUB} schema {t.get('schema')!r} != {SCHEMAS[TOOL_DUB]}")
        try:
            parameters(contract)
        except (TypeError, ValueError, IndexError):
            errs.append(f"tool {TOOL_DUB} limits lack duck_db / voice_gain_db / max_overrun ranges")
        if not engines(contract):
            errs.append(f"tool {TOOL_DUB} names no engines")
        if not unfit_policies(contract):
            errs.append(f"tool {TOOL_DUB} argv lacks [--on-unfit ...]")
    ex = as_dict(contract.get("execution"))
    if ex.get("shell") is not False or ex.get("network") is not False or ex.get("input_is_read_only") is not True:
        errs.append("execution must declare shell false, network false and input_is_read_only true")
    consent = as_dict(as_dict(contract.get("policy")).get("consent_record"))
    if consent.get("schema") != "dubbing/consent@1" or "bypass" not in consent or consent.get("bypass") is not None:
        errs.append("policy.consent_record must be dubbing/consent@1 with bypass null (no way to skip the consent gate)")
    provided = {str(p.get("id")) for p in contract.get("provides") or [] if isinstance(p, dict)}
    for cap in CAPABILITY_IDS.values():
        if cap not in provided:
            errs.append(f"provides lacks {cap}")
    errors = as_dict(contract.get("errors"))
    for code in ERROR_CLASSES:
        if not isinstance(as_dict(errors.get(code)).get("exit_code"), int):
            errs.append(f"errors lack {code} (with its exit_code)")
    if as_dict(errors.get(CONSENT_CODE)).get("retryable") is not False:
        errs.append(f"errors {CONSENT_CODE} must be retryable false")
    return errs


def lifecycle(contract: Dict[str, Any]) -> Dict[str, str]:
    """capability id → lifecycle (EXPERIMENTAL today) from the contract's `provides`."""
    return {str(p.get("id")): str(p.get("lifecycle")) for p in contract.get("provides") or [] if isinstance(p, dict)}


def _op_tools(c: Dict[str, Any]) -> Dict[str, Any]:
    """The contract with only the tools this adapter runs (narrate / describe / fit / doctor / engines / contract are not called as operations)."""
    return dict(c, tools=[t for t in c.get("tools") or [] if isinstance(t, dict) and t.get("tool_id") in TOOLS])


def drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(_op_tools(live), _op_tools(pinned or pinned_contract()), DRIFT_KEYS, "tools", "tool_id", DRIFT_TOOL_KEYS)


def pinned_error_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    """A pinned error code that disappeared or changed its exit code / retryable verdict (a new code is additive: SKILL_ERROR, kept verbatim)."""
    p, lv = as_dict((pinned or pinned_contract()).get("errors")), as_dict(live.get("errors"))
    return [f"errors {c}: pinned {json.dumps(p[c], sort_keys=True)} != live {json.dumps(lv.get(c), sort_keys=True)}" for c in sorted(p) if lv.get(c) != p[c]]


def breaking_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    pinned = pinned or pinned_contract()
    return _breaking_drift(_op_tools(live), _op_tools(pinned), FATAL_DRIFT_KEYS, "tools", "tool_id", FATAL_DRIFT_TOOL_KEYS) + pinned_error_drift(live, pinned)


PACKAGE = SkillPackage(
    skill_id=SKILL_ID, name="dubbing-skill", version="",
    description="AI narration, dubbing and audio description with local engines (Chatterbox, VOICEVOX, Style-Bert-VITS2, Kokoro); translation is the "
                "caller's, a reference voice needs its owner's consent record; every media step through ffmpeg-skill; EXPERIMENTAL",
    capabilities=["dubbing"],
    tools=[ToolSpec(TOOL_DUB, SKILL_ID, description="The programme with its dialogue replaced by the user's translated cues, spoken by a local engine over the "
                                                    "programme's own sound (ducked); the picture stream-copied",
                    inputs=["input", "cues"], produces_output=True, deterministic=False,
                    result_keys=["output", "fit", "mix", "clips", "kept_original", "obligations", "provenance", "warnings"])],
    repository="kajisho5/dubbing-skill", role="speech generation (narration, dubbing, audio description)")


def _number(v: Any, name: str, lo: float, hi: float) -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        raise ToolError(f"{name} must be a number, got {v!r}")
    if v < lo or v > hi:
        raise ToolError(f"{name} {v!r} must be within [{lo:g}, {hi:g}] (the dubbing contract's limits)")
    return repr(float(v))


class DubbingAdapter(ToolAdapter):
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
            raise ContractError("incompatible dubbing-skill contract: " + "; ".join(errs))
        self.version = str(self.contract.get("version"))
        self.tools = {t: True for t in TOOLS}
        self.retryable = {c: bool(as_dict(v).get("retryable")) for c, v in as_dict(self.contract.get("errors")).items()}
        self.lifecycle = lifecycle(self.contract)
        self.parameters = parameters(self.contract)
        self.engines = engines(self.contract)
        self.unfit = unfit_policies(self.contract)
        self._doctor: Optional[Dict[str, Any]] = None

    # ---- contract / doctor
    def _call(self, argv: List[str], timeout: Optional[float]) -> Tuple[int, str, str]:
        return invoke(self.skill, argv, timeout=timeout)

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._call(["contract", "--json"], 60.0)
        if code != 0:
            raise ContractError(f"dubbing contract --json failed (exit {code}): {err.strip()[-300:]}")
        return one_json_document(out, "dubbing contract")

    def _engine_argv(self) -> List[str]:
        return [f"--ffmpeg-skill={self.ffmpeg_skill_dir}"] if self.ffmpeg_skill_dir else []

    def doctor(self) -> Dict[str, Any]:
        """`dubbing doctor --json --offline --ffmpeg-skill=<dir>` with `status` as the Skill reports it (ok / degraded: an engine MISSING, the
        Skill itself works / fail). Without the agent's own ffmpeg-skill it is `fail` without running the Skill (the Skill would otherwise
        search the current directory)."""
        if self._doctor is not None:
            return self._doctor
        if not self.ffmpeg_skill_dir:
            self._doctor = {"schema": DOCTOR_SCHEMA, "ok": False, "status": "fail", "checks": {}, "problems": ["no ffmpeg-skill located by the agent: dubbing-skill runs every media step through it"],
                            "warnings": []}
            return self._doctor
        code, out, err = self._call(["doctor", "--json", "--offline"] + self._engine_argv(), 300.0)
        try:
            doc = one_json_document(out, "dubbing doctor")
        except ToolError as e:
            doc = {"schema": DOCTOR_SCHEMA, "ok": False, "status": "fail", "checks": {}, "problems": [f"doctor produced no document (exit {code}): {e} {err.strip()[-200:]}"]}
        if doc.get("schema") != DOCTOR_SCHEMA or doc.get("status") not in ("ok", "degraded", "fail"):
            doc = dict(doc, status="fail", problems=list(doc.get("problems") or []) + [f"doctor document {doc.get('schema')!r} / status {doc.get('status')!r} not understood"])
        doc = dict(doc, problems=list(doc.get("problems") or []), warnings=list(doc.get("warnings") or []))
        self._doctor = doc
        return doc

    def operation_status(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """`dubbing:<TYPE>` verdicts from the doctor's `checks.operations`: supported (ready) / unavailable (its engine is not OK)."""
        ops = as_dict(as_dict((doc or self.doctor()).get("checks")).get("operations"))
        return {typ: ("supported" if as_dict(ops.get(op)).get("ready") is True else "unavailable") for op, typ in OPERATION_TYPES.items()}

    def operation_evidence(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Why DUB is (not) ready, in the Skill's own words: its default engine, that engine's status, the alternatives the doctor names and,
        when it is not ready, the code the Skill itself would return (FFMPEG_SKILL_UNAVAILABLE, MODEL_NOT_FOUND or ENGINE_UNAVAILABLE) with the
        engine's install line. Each engine's own status is listed too, so a plan naming another engine (dub.engine) is judged on that one."""
        doc = doc or self.doctor()
        checks = as_dict(doc.get("checks"))
        ops, engs = as_dict(checks.get("operations")), as_dict(checks.get("engines"))
        out: Dict[str, Dict[str, Any]] = {}
        for op, typ in OPERATION_TYPES.items():
            row = as_dict(ops.get(op))
            eng = as_dict(engs.get(str(row.get("engine") or "")))
            ev: Dict[str, Any] = {"engine": row.get("engine"), "engine_status": str(row.get("engine_status") or eng.get("status") or "UNKNOWN"), "ready": row.get("ready") is True,
                                  "alternatives": [str(a) for a in row.get("alternatives") or []], "lifecycle": self.lifecycle.get(CAPABILITY_IDS[typ]),
                                  "engines": {str(e): {"status": str(as_dict(engs.get(e)).get("status") or "UNKNOWN"), "install": str(as_dict(engs.get(e)).get("install") or "")[:600],
                                                       "detail": str(as_dict(engs.get(e)).get("detail") or "")[:300], "missing": as_dict(engs.get(e)).get("missing"),
                                                       "voice_cloning": as_dict(engs.get(e)).get("voice_cloning")}
                                              for e in self.engines}}
            if row.get("ready") is not True:
                ev["code"] = engine_code(checks, str(row.get("engine") or ""))
                ev["detail"] = str(eng.get("detail") or "")[:300]
                ev["install"] = str(eng.get("install") or "")[:600]
            out[typ] = ev
        return out

    def drift(self) -> List[str]:
        return drift(self.contract)

    def breaking_drift(self) -> List[str]:
        """ADR-048: the subset of drift() that gates availability (FATAL_DRIFT_KEYS, the op tool's wire, the pinned error codes)."""
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
        raise ToolError("dubbing: a production Skill has no measurement tools (media-analysis-skill / ffmpeg-skill measure)")

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

    def _existing(self, raw: Any, what: str, exts: Tuple[str, ...]) -> str:
        """A file the user named (cues, source cues, reference voice, consent record): an existing file inside the allowed roots, of its kind."""
        self._check_raw(raw, what)
        resolved = os.path.realpath(os.path.abspath(raw))
        roots = list(self.allowed_inputs) + ([self.workspace] if self.workspace else [])
        if roots and not any(self._within(r, resolved) for r in roots):
            raise ToolError(f"{what} is outside the allowed input roots: {os.path.basename(raw)}")
        if not os.path.isfile(resolved):
            raise ToolError(f"{what} not found: {os.path.basename(raw)}")
        if exts and os.path.splitext(resolved)[1].lower() not in exts:
            raise ToolError(f"{what} must be one of {', '.join(exts)}: {os.path.basename(raw)}")
        if self.policy is not None:
            self.policy.check_input(resolved)
        return resolved

    def _input(self, ref: Any, paths: Dict[str, str]) -> str:
        if not isinstance(ref, str) or ref not in paths:
            raise ToolError(f"input {ref!r} is not an artifact id of this job")
        return self._existing(paths[ref], "input", ())

    def _output(self, ref: Any, paths: Dict[str, str], inputs: List[str]) -> str:
        if not isinstance(ref, str) or ref not in paths:
            raise ToolError(f"output {ref!r} is not an artifact id of this job")
        raw = paths[ref]
        self._check_raw(raw, "output")
        absolute = os.path.join(os.path.realpath(os.path.dirname(os.path.abspath(raw))), os.path.basename(raw))
        if self.workspace and not (self._within(self.workspace, absolute) and absolute != self.workspace):
            raise ToolError(f"output is outside the workspace: {os.path.basename(raw)}")
        if os.path.splitext(absolute)[1].lower() not in VIDEO_OUTPUTS:
            raise ToolError(f"output must be a video container ({', '.join(VIDEO_OUTPUTS)}): {os.path.basename(raw)}")
        for i in inputs:
            if same_file(i, absolute) or os.path.normcase(i) == os.path.normcase(absolute):
                raise ToolError("output would overwrite an input")
        if self.policy is not None:
            self.policy.check_output(absolute, inputs)
        return absolute

    def build(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None) -> Dict[str, Any]:
        """{"argv", "inputs", "outputs", "timeout"} for one DUB operation, or ToolError naming the first value that cannot be sent."""
        if op.tool not in TOOLS:
            raise ToolError(f"{SKILL_ID} adapter does not support {op.tool}")
        if not self.ffmpeg_skill_dir:
            raise ToolError("no ffmpeg-skill located by the agent: dubbing-skill runs every media step through it")
        args = dict(op.args)
        hit = scan_forbidden(args, FORBIDDEN_ARG_KEYS)
        if hit:
            raise ToolError(f"forbidden key {hit}")
        allowed = {"input", "output", "cues", "lang", "source_cues", "engine", "voice", "reference", "consent", "on_unfit"} | set(self.parameters)
        unknown = sorted(k for k in args if k not in allowed)
        if unknown:
            raise ToolError(f"unknown {op.tool} argument(s): {', '.join(unknown)}")
        lang = args.get("lang")
        if not isinstance(lang, str) or not LANG_RE.match(lang):
            raise ToolError(f"lang {lang!r} is not a BCP-47 language tag")
        src = self._input(args.get("input"), paths)
        cues = self._existing(args.get("cues"), "cues", CUES_EXTENSIONS)
        files = [src, cues]
        options = [f"--lang={lang}"]
        if args.get("source_cues") is not None:
            sc = self._existing(args["source_cues"], "source_cues", CUES_EXTENSIONS)
            files.append(sc)
            options.append(f"--source-cues={sc}")
        if args.get("engine") is not None:
            if args["engine"] not in self.engines:
                raise ToolError(f"engine {args['engine']!r} is not one DUB takes ({', '.join(self.engines)})")
            options.append(f"--engine={args['engine']}")
        if args.get("voice") is not None:
            if not isinstance(args["voice"], str) or not VOICE_RE.match(args["voice"]):
                raise ToolError(f"voice {args['voice']!r} is not a voice id")
            options.append(f"--voice={args['voice']}")
        # a reference voice only with the consent record the user named, never one without the other (dubbing-skill ADR-009 / ADR-012)
        if (args.get("reference") is None) != (args.get("consent") is None):
            raise ToolError("a reference voice is passed only together with its owner's consent record (reference + consent, both from the user)")
        if args.get("reference") is not None:
            ref = self._existing(args["reference"], "reference", REFERENCE_EXTENSIONS)
            rec = self._existing(args["consent"], "consent", CONSENT_EXTENSIONS)
            if same_file(ref, rec):
                raise ToolError("the consent record cannot be the reference voice")
            files += [ref, rec]
            options += [f"--reference={ref}", f"--consent={rec}"]
        for name, (lo, hi) in self.parameters.items():
            if args.get(name) is not None:
                options.append(f"--{name.replace('_', '-')}={_number(args[name], name, lo, hi)}")
        if args.get("on_unfit") is not None:
            if args["on_unfit"] not in self.unfit:
                raise ToolError(f"on_unfit {args['on_unfit']!r} must be one of {', '.join(self.unfit)}")
            options.append(f"--on-unfit={args['on_unfit']}")
        out = self._output(args.get("output"), paths, files)
        t = max(1.0, min(86400.0, float(timeout or self.default_timeout)))
        note = json.dumps(TRANSLATION_NOTE, sort_keys=True, separators=(",", ":"))
        argv = [COMMAND[op.tool], "--json", "--offline", "--overwrite"] + self._engine_argv() + [f"--output={out}", f"--timeout={t:g}", f"--translation-note={note}"] \
            + options + ["--", src, cues]
        return {"argv": argv, "inputs": files, "outputs": [out], "timeout": t, "reference": args.get("reference") is not None}

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            b = self.build(op, paths)
        except ToolError as e:
            return [f"<refused: {e}>"]
        return [" ".join(["dubbing"] + b["argv"])]

    # ---- execution
    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, exit_code: int, code: str, message: str, retryable: bool = False,
              details: Optional[Dict[str, Any]] = None) -> ToolResult:
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "failed",
                                "error": {"code": code, "message": message[:1200], "retryable": bool(retryable) and code not in NEVER_RETRIED, "details": dict(details or {}),
                                          "exit_code": exit_code, "recovery_class": recovery_class(code)}}
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
        command = " ".join(["dubbing"] + b["argv"])
        if dry_run:   # DUB has no dry run: the executor's dry run checks the request and runs nothing
            return ToolResult(op.id, op.tool, True, 0, None, {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run", "commands": [command]},
                              [command], "", round(time.time() - t0, 3), attempt, True)
        code, out, err = self._call(b["argv"], b["timeout"] + PROCESS_MARGIN)
        tail = "\n".join(err.strip().splitlines()[-12:])
        if code == 124:
            self._remove_fresh(b["outputs"], t0)
            return self._fail(op, attempt, dry_run, t0, 124, "TIMEOUT", f"dubbing dub exceeded {b['timeout'] + PROCESS_MARGIN:g}s", retryable=True, details={"reason": "timeout"})
        try:
            doc = one_json_document(out, "dubbing dub")
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
            if ecode not in ERROR_CLASSES or ecode in NEVER_RETRIED:
                retry = False
            if ecode == CONSENT_CODE:   # what the user has to fix; the agent does not fix it
                details["agent"] = "the consent record comes from the user: the agent never writes, edits or signs one, and does not retry"
            self._remove_fresh(b["outputs"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code else 1, ecode, str(e_doc.get("message") or tail or "")[:1200], retryable=retry, details=details)
        why = self._admit(doc, b)
        if why:
            self._remove_fresh(b["outputs"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code else 9, "INVALID_OUTPUT", "dubbing result not admitted: " + "; ".join(why))
        if code != 0:
            self._remove_fresh(b["outputs"], t0)
            return self._fail(op, attempt, dry_run, t0, code, "INVALID_OUTPUT", f"exit code {code} with an ok result")
        data = self._success_data(doc, b, command)
        return ToolResult(op.id, op.tool, True, 0, b["outputs"][0], data, [command], tail, round(time.time() - t0, 3), attempt, False)

    def _admit(self, doc: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
        """Why a result is not about this request (empty = admitted)."""
        errs: List[str] = []
        if doc.get("schema") != SCHEMAS[TOOL_DUB] or doc.get("op") != RESULT_OP[TOOL_DUB]:
            return [f"result {doc.get('schema')!r} / op {doc.get('op')!r} is not {SCHEMAS[TOOL_DUB]}"]
        sk = as_dict(doc.get("skill"))
        if sk.get("id") != SKILL_ID or str(sk.get("version")) != self.version:
            errs.append(f"result skill {sk!r} is not {SKILL_ID}@{self.version}")
        o = as_dict(doc.get("output"))
        out = b["outputs"][0]
        if not same_file(str(o.get("path") or ""), out):
            return errs + [f"output {os.path.basename(str(o.get('path') or ''))!r} is not the planned {os.path.basename(out)}"]
        if o.get("kind") != "video":
            errs.append(f"output kind {o.get('kind')!r} is not video (the agent keeps the picture)")
        prov = as_dict(doc.get("provenance"))
        if not os.path.isfile(out) or os.path.getsize(out) == 0:
            errs.append(f"{os.path.basename(out)} missing or empty although the Skill reported success")
        elif str(as_dict(prov.get("output")).get("sha256") or "") != sha256_file(out):
            errs.append(f"{os.path.basename(out)}: the sha256 the Skill reported does not match the file on disk")
        delta, tol = o.get("av_delta"), o.get("tolerance")
        if isinstance(delta, (int, float)) and isinstance(tol, (int, float)) and abs(float(delta)) > float(tol) + 1e-6:
            errs.append(f"audio and picture differ by {delta}s, more than the Skill's own tolerance {tol}s")
        if not as_dict(prov.get("engine")).get("id"):
            errs.append("provenance names no engine")
        if not isinstance(doc.get("obligations"), dict) or not isinstance(as_dict(doc.get("obligations")).get("credit"), list):
            errs.append("result carries no obligations.credit list (the credits the published video must carry)")
        if b.get("reference") and not prov.get("consent"):
            errs.append("a reference voice was passed but the provenance records no verified consent")
        if not b.get("reference") and prov.get("consent"):
            errs.append("the provenance records a consent although no reference voice was passed")
        return errs

    def _success_data(self, doc: Dict[str, Any], b: Dict[str, Any], command: str) -> Dict[str, Any]:
        """What the agent records: the engine (licences, credit, watermark), the obligations (credit[] goes into the ProductionReceipt), the fit
        (lines kept in the original), the consent summary the Skill verified (never the evidence's contents), the Skill's warnings verbatim,
        the output's facts and sha256, and the lifecycle (EXPERIMENTAL) -- never local paths beyond the output the agent itself named."""
        prov = as_dict(doc.get("provenance"))
        o = as_dict(doc.get("output"))
        ob = as_dict(doc.get("obligations"))
        fit = as_dict(doc.get("fit"))
        clips = [as_dict(c) for c in doc.get("clips") or []]
        cap = str(doc.get("capability") or CAPABILITY_IDS["DUB"])
        return {"skill": {"id": SKILL_ID, "version": self.version}, "status": "ok", "op": RESULT_OP[TOOL_DUB], "capability": cap, "lifecycle": self.lifecycle.get(cap),
                "engine": as_dict(doc.get("engine")) or as_dict(prov.get("engine")),
                "obligations": {"credit": [str(c) for c in ob.get("credit") or []], "credit_required": ob.get("credit_required"),
                                "credits": [as_dict(c) for c in ob.get("credits") or []], "watermark": ob.get("watermark")},
                "consent": prov.get("consent"), "translation": as_dict(prov.get("translation")),
                "fit": {"summary": as_dict(fit.get("summary")), "unfit": list(fit.get("unfit") or []), "render": as_dict(fit.get("render"))},
                "lines": len(clips), "placed": sum(1 for c in clips if c.get("placed")), "kept_original": list(doc.get("kept_original") or []),
                "warnings": [str(w)[:600] for w in doc.get("warnings") or []],
                "outputs": [{"file": b["outputs"][0], "sha256": as_dict(prov.get("output")).get("sha256"),
                             **{k: o.get(k) for k in ("kind", "duration", "audio_duration", "video_duration", "av_delta", "tolerance", "codec", "sample_rate", "channels")}}],
                "text_sha256": as_dict(doc.get("input")).get("text_sha256"), "ffmpeg_skill": as_dict(prov.get("ffmpeg_skill")), "network": prov.get("network"),
                "created_at": prov.get("created_at"), "commands": [command]}


def engine_code(checks: Dict[str, Any], engine: str) -> str:
    """The code dubbing-skill itself returns for an engine that is not ready: ffmpeg-skill unusable, else a model missing, else the engine."""
    if checks.get("ffmpeg_skill") and as_dict(checks.get("ffmpeg_skill")).get("status") != "OK":
        return "FFMPEG_SKILL_UNAVAILABLE"
    return "MODEL_NOT_FOUND" if as_dict(as_dict(checks.get("engines")).get(engine)).get("missing") == "model" else "ENGINE_UNAVAILABLE"


def readiness_code(evidence: Dict[str, Any], engine: Optional[str]) -> str:
    """The code dubbing-skill would return for DUB on `engine` given the resolver's `dubbing:DUB` evidence: the media engine Skill unusable (the
    evidence's own code), else that engine's model missing, else the engine itself."""
    if evidence.get("code") == "FFMPEG_SKILL_UNAVAILABLE":
        return "FFMPEG_SKILL_UNAVAILABLE"
    row = as_dict(as_dict(evidence.get("engines")).get(str(engine or "")))
    return "MODEL_NOT_FOUND" if row.get("missing") == "model" else "ENGINE_UNAVAILABLE"


def lift_observation(result: ToolResult, asset_id: Optional[str] = None) -> Optional[Observation]:
    """What dubbing-skill reported about the dubbed programme, as an agent Observation (provenance only; never fed back into the analysis):
    the engine with its licences, the obligations (credits, watermark), the consent summary, the fit, the Skill's warnings and the output."""
    data = result.data if isinstance(result.data, dict) else {}
    if data.get("status") != "ok" or not data.get("outputs"):
        return None
    sk = as_dict(data.get("skill"))
    eng = as_dict(data.get("engine"))
    outs = [dict(o) for o in data.get("outputs") or [] if isinstance(o, dict)]
    return Observation(kind=f"speech.{data.get('op')}", asset_id=asset_id or result.op_id, source=f"{result.tool}@{sk.get('version') or ''}",
                       data={"capability": data.get("capability"), "lifecycle": data.get("lifecycle"), "engine": eng, "obligations": as_dict(data.get("obligations")),
                             "consent": data.get("consent"), "translation": as_dict(data.get("translation")), "fit": as_dict(data.get("fit")),
                             "lines": data.get("lines"), "placed": data.get("placed"), "kept_original": list(data.get("kept_original") or []),
                             "warnings": list(data.get("warnings") or []), "outputs": outs, "ffmpeg_skill": as_dict(data.get("ffmpeg_skill")), "network": data.get("network")},
                       analyzer=f"{eng.get('id') or SKILL_ID}@{eng.get('version') or ''}", provenance="OBSERVED", skill=SKILL_ID, skill_version=str(sk.get("version") or ""),
                       tool=result.tool, fingerprint=str((outs[0] if outs else {}).get("sha256") or ""), parameters={"outputs": [o.get("file") for o in outs]})
