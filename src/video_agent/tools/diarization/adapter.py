"""DiarizationAdapter: the agent's boundary to diarization-skill (ADR-083; diarization-skill roadmap step 5, agent side).

    video-production-agent ─(analysis kind speaker_turns)─→ DiarizationAdapter ─(argv list, never a shell)─→
        `diarization diarize --json --overwrite --output=<workspace>/cache/diarization/<asset>.turns.json [--num-speakers=N |
         --min-speakers=N --max-speakers=N] [--audio-stream=N] --timeout=T -- <media>`
          ─(its own worker: pyannote.audio 4.x + speaker-diarization-community-1 from a local directory, offline)─→ turns JSON

Protocol (diarization-skill's own contract, `diarization contract --json`, pinned from main in contract_0.1.0.json -- the repository
has no release tag, so the pin follows main as ingest-skill's and voice-enhance-skill's do):
    contract → `diarization contract --json`
    doctor   → `diarization doctor --json` (ok, ready.{diarize, assign, check}, checks, problems; exit 3 while not ok, never downloads)
    diarize  → `diarization-skill/turns@1`: anonymous exclusive turns SPEAKER_nn on the media's own timeline, overlap kept apart,
               engine / model / revision / licence / attribution, the source fingerprint (sha256 of the file bytes)
    failure  → {"ok": false, "error": {code, message, detail}}, the code fixing the exit code (its `errors` table)

A measurement only (the contract's role `measurement`, writes_media false): the turns become an OBSERVED Observation of kind
`speaker_turns` and SpeakerEvents on the asset's timeline; nothing here names a speaker, decides an edit or touches the media.
Every value is checked before the process starts: the input lies inside the allowed roots, the turns file lands in the
workspace, the speaker counts are integers >= 1 (`--num-speakers` excludes the range, min <= max: the Skill's own rule), the
stream index an integer >= 0. Every option is one `--flag=value` word and the media path comes after `--` (ADR-069). The result is
admitted only when it is about this request: schema, tool, Skill version, the fingerprint equal to the input's sha256, turns
exclusive, ordered, inside the media and spoken by listed SPEAKER_nn ids, and the file on disk equal to the document printed.
The Skill's error codes are kept verbatim and mapped onto the agent's recovery classes (ERROR_CLASSES)."""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...models import Operation, ToolResult
from ...skills.contract import SkillPackage, ToolSpec
from ..base import ToolAdapter, ToolError
from ..skill_process import (FORBIDDEN_ARG_KEYS, CliSkill, ContractError, as_dict, breaking_drift as _breaking_drift, drift_report, invoke, one_json_document,
                             scan_forbidden, sha256_file, strip_sha_prefix)
from .locate import PACKAGE_ID, locate_diarization  # noqa: F401  (re-exported)

SKILL_ID = PACKAGE_ID
PREFIX = SKILL_ID + "/"
TOOL_DIARIZE = "diarization/diarize"
TURNS_SCHEMA = "diarization-skill/turns@1"
SUPPORTED_SKILL_VERSIONS = ("0.1.",)
KIND = "speaker_turns"                          # the analysis kind (media/analysis.py) this adapter measures
SPEAKER_RE = re.compile(r"^SPEAKER_\d{2,}$")
ORDER_TOLERANCE = 0.01                          # diarization-skill's OVERLAP_TOLERANCE between consecutive exclusive turns (its turns.py)
END_TOLERANCE = 0.01                            # a turn ends no later than the media (the Skill clamps engine overruns itself)
# the typed parameters the agent may send (diarization-skill's CLI: counts >= 1, --num-speakers excludes the range, min <= max; stream >= 0)
PARAMETERS = ("num_speakers", "min_speakers", "max_speakers", "audio_stream")
# the per-operation capability `diarization:<TYPE>` (resolver) ← the doctor's `ready.<op>`
OPERATION_TYPES = {"diarize": "DIARIZE", "assign": "ASSIGN"}
DEFAULT_TIMEOUT = 3600.0                        # the Skill's own engine time limit (its --timeout default); the process gets a margin on top
PROCESS_MARGIN = 120.0
# diarization-skill's error codes (its contract `errors`) → the agent's recovery class. A missing engine / pipeline / ffmpeg is
# TOOL_MISSING (the user installs it), a bad request INVALID_ARGS, the Skill's refusal of its own output SKILL_ERROR; ENGINE_TIMEOUT
# keeps the Skill's retryable verdict.
ERROR_CLASSES: Dict[str, str] = {
    "INPUT_NOT_FOUND": "INPUT_MISSING", "INVALID_ARGUMENT": "INVALID_ARGS", "UNSUPPORTED_MEDIA": "INVALID_ARGS", "INVALID_TRANSCRIPT": "INVALID_ARGS",
    "INVALID_TURNS": "INVALID_ARGS", "SOURCE_MISMATCH": "INVALID_ARGS", "SPEAKERS_ALREADY_SET": "INVALID_ARGS", "OUTPUT_IS_INPUT": "INVALID_ARGS",
    "OUTPUT_EXISTS": "INVALID_ARGS", "FFMPEG_NOT_FOUND": "TOOL_MISSING", "ENGINE_UNAVAILABLE": "TOOL_MISSING", "PIPELINE_NOT_FOUND": "TOOL_MISSING",
    "ENGINE_FAILED": "SKILL_ERROR", "ENGINE_TIMEOUT": "TIMEOUT", "INVALID_RESULT": "SKILL_ERROR",
}
AGENT_ERRORS: Dict[str, str] = {"INVALID_REQUEST": "INVALID_ARGS", "INVALID_OUTPUT": "SKILL_ERROR", "TIMEOUT": "TIMEOUT"}
# contract drift (ADR-045 / ADR-047): every key the adapter reads is compared with the pinned contract; the fatal subset gates availability
DRIFT_KEYS = ("skill_id", "version", "contract_version", "provides", "dependencies", "not_provided", "turns", "engines", "execution", "errors")
DRIFT_TOOL_KEYS = ("role", "writes_media", "mutates_input", "deterministic", "argv")
FATAL_DRIFT_KEYS = ("skill_id", "contract_version", "turns", "execution")
FATAL_DRIFT_TOOL_KEYS = ("role", "writes_media", "mutates_input", "argv")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def recovery_class(code: str) -> str:
    return ERROR_CLASSES.get(code) or AGENT_ERRORS.get(code) or "SKILL_ERROR"


def check_contract(contract: Any) -> List[str]:
    """What the adapter relies on, refused (never patched) when it does not hold: identity, contract and Skill version, DIARIZE as a
    measurement that writes no media and never touches its input, the turns schema, a shell-free offline execution with a read-only
    input, and every error code the adapter maps with its exit code."""
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
    t = tools.get(TOOL_DIARIZE)
    if t is None:
        errs.append(f"contract lacks tool {TOOL_DIARIZE} (declares {sorted(tools)})")
    elif t.get("role") != "measurement" or t.get("writes_media") is not False or t.get("mutates_input") is not False:
        errs.append(f"tool {TOOL_DIARIZE} must declare role measurement, writes_media false, mutates_input false")
    if as_dict(contract.get("turns")).get("schema") != TURNS_SCHEMA:
        errs.append(f"turns.schema {as_dict(contract.get('turns')).get('schema')!r} != {TURNS_SCHEMA}")
    ex = as_dict(contract.get("execution"))
    if ex.get("shell") is not False or ex.get("network") is not False or ex.get("input_is_read_only") is not True:
        errs.append("execution must declare shell false, network false and input_is_read_only true")
    errors = as_dict(contract.get("errors"))
    for code in ERROR_CLASSES:
        if not isinstance(as_dict(errors.get(code)).get("exit_code"), int):
            errs.append(f"errors lack {code} (with its exit_code)")
    return errs


def _measure_tools(c: Dict[str, Any]) -> Dict[str, Any]:
    return dict(c, tools=[t for t in c.get("tools") or [] if isinstance(t, dict) and t.get("tool_id") == TOOL_DIARIZE])


def drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(live, pinned or pinned_contract(), DRIFT_KEYS, "tools", "tool_id", DRIFT_TOOL_KEYS)


def pinned_error_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    """A pinned error code that disappeared or changed its exit code / retryable verdict (a new code is additive: SKILL_ERROR, kept verbatim)."""
    p, lv = as_dict((pinned or pinned_contract()).get("errors")), as_dict(live.get("errors"))
    return [f"errors {c}: pinned {json.dumps(p[c], sort_keys=True)} != live {json.dumps(lv.get(c), sort_keys=True)}" for c in sorted(p) if lv.get(c) != p[c]]


def breaking_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    pinned = pinned or pinned_contract()
    return _breaking_drift(_measure_tools(live), _measure_tools(pinned), FATAL_DRIFT_KEYS, "tools", "tool_id", FATAL_DRIFT_TOOL_KEYS) + pinned_error_drift(live, pinned)


PACKAGE = SkillPackage(
    skill_id=SKILL_ID, name="diarization-skill", version="",
    description="Who spoke when, local and offline (pyannote.audio 4.x, speaker-diarization-community-1 from a local directory): anonymous "
                "speaker turns SPEAKER_nn with overlapping speech kept apart; it never names anyone (EXPERIMENTAL)",
    capabilities=["diarization"],
    tools=[ToolSpec(TOOL_DIARIZE, SKILL_ID, description="Anonymous exclusive speaker turns of a recording (diarization-skill/turns@1), overlap regions, engine attribution",
                    required_capabilities=["diarization:DIARIZE"], inputs=["input"], produces_output=False, deterministic=False,
                    result_keys=["turns", "turns_file", "engine", "warnings"])],
    repository="kajisho5/diarization-skill", role="speaker diarization (who spoke when; anonymous ids, never a name)")


def _count(v: Any, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ToolError(f"{name} must be an integer >= 1, got {v!r}")
    return v


def check_parameters(params: Dict[str, Any]) -> Dict[str, int]:
    """The speaker counts and stream index as diarization-skill's own CLI checks them (its check_speaker_counts)."""
    unknown = sorted(k for k in params if k not in PARAMETERS)
    if unknown:
        raise ToolError(f"unknown diarize parameter(s): {', '.join(unknown)}")
    out: Dict[str, int] = {}
    for k in ("num_speakers", "min_speakers", "max_speakers"):
        if params.get(k) is not None:
            out[k] = _count(params[k], k)
    if "num_speakers" in out and ("min_speakers" in out or "max_speakers" in out):
        raise ToolError("num_speakers excludes min_speakers / max_speakers (diarization-skill's rule)")
    if "min_speakers" in out and "max_speakers" in out and out["min_speakers"] > out["max_speakers"]:
        raise ToolError("min_speakers must be <= max_speakers")
    if params.get("audio_stream") is not None:
        s = params["audio_stream"]
        if isinstance(s, bool) or not isinstance(s, int) or s < 0:
            raise ToolError(f"audio_stream must be an integer >= 0, got {s!r}")
        out["audio_stream"] = s
    return out


def check_turns(doc: Dict[str, Any], version: str, input_sha256: str, params: Dict[str, int]) -> List[str]:
    """Why a turns document is not about this request (empty = admitted)."""
    errs: List[str] = []
    if doc.get("schema") != TURNS_SCHEMA:
        return [f"schema {doc.get('schema')!r} != {TURNS_SCHEMA}"]
    if doc.get("tool") != TOOL_DIARIZE or doc.get("skill") != "diarization-skill" or str(doc.get("skill_version")) != version:
        errs.append(f"result tool / skill {doc.get('tool')!r} / {doc.get('skill')!r}@{doc.get('skill_version')!r} is not {TOOL_DIARIZE}@{version}")
    src = as_dict(doc.get("source"))
    if strip_sha_prefix(src.get("fingerprint")) != input_sha256:
        errs.append("source.fingerprint is not the sha256 of the input file")
    if int(src.get("audio_stream") or 0) != params.get("audio_stream", 0):
        errs.append(f"source.audio_stream {src.get('audio_stream')!r} is not the requested {params.get('audio_stream', 0)}")
    dur = src.get("media_duration")
    if not isinstance(dur, (int, float)) or isinstance(dur, bool) or not math.isfinite(dur) or dur <= 0:
        errs.append("source.media_duration missing")
        dur = None
    sent = as_dict(doc.get("parameters"))
    for k in ("num_speakers", "min_speakers", "max_speakers"):
        if sent.get(k) != params.get(k):
            errs.append(f"parameters.{k} {sent.get(k)!r} is not the requested {params.get(k)!r}")
    speakers = doc.get("speakers")
    if not isinstance(speakers, list) or any(not isinstance(s, str) or not SPEAKER_RE.match(s) for s in speakers) or len(set(speakers)) != len(speakers):
        return errs + ["speakers must be distinct anonymous SPEAKER_nn ids"]
    if doc.get("speaker_count") != len(speakers):
        errs.append("speaker_count is not the number of speakers")
    prev = None
    for t in doc.get("turns") if isinstance(doc.get("turns"), list) else [None]:
        t = as_dict(t)
        s, e = t.get("start"), t.get("end")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (s, e)) or not 0 <= s < e:
            return errs + [f"turn {t!r} has no finite 0 <= start < end"]
        if t.get("speaker") not in speakers:
            return errs + [f"turn speaker {t.get('speaker')!r} is not listed in speakers"]
        if prev is not None and s < prev - ORDER_TOLERANCE:
            return errs + ["turns overlap or are out of order (exclusive turns expected)"]
        if dur is not None and e > dur + END_TOLERANCE:
            return errs + [f"a turn ends at {e} past the media ({dur})"]
        prev = e
    if not isinstance(doc.get("turns"), list):
        errs.append("turns is not a list")
    if not as_dict(doc.get("engine")).get("id") or not as_dict(doc.get("engine")).get("attribution"):
        errs.append("engine id / attribution missing")
    return errs


class DiarizationAdapter(ToolAdapter):
    owns_cache = False   # the Skill keeps no cache: the agent's ObservationCache stores the turns like any other measurement

    def __init__(self, skill: CliSkill, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None, timeout: float = DEFAULT_TIMEOUT):
        self.skill = skill
        self.name = SKILL_ID
        self.workspace = os.path.realpath(os.path.abspath(workspace)) if workspace else None
        self.allowed_inputs = [os.path.realpath(os.path.abspath(p)) for p in (allowed_inputs or [])]
        self.default_timeout = float(timeout)
        self.contract = self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("incompatible diarization-skill contract: " + "; ".join(errs))
        self.version = str(self.contract.get("version"))
        self.tools = {TOOL_DIARIZE: True}
        self.retryable = {c: bool(as_dict(v).get("retryable")) for c, v in as_dict(self.contract.get("errors")).items()}
        self._doctor: Optional[Dict[str, Any]] = None

    # ---- contract / doctor
    def _call(self, argv: List[str], timeout: Optional[float]) -> Tuple[int, str, str]:
        return invoke(self.skill, argv, timeout=timeout)

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._call(["contract", "--json"], 60.0)
        if code != 0:
            raise ContractError(f"diarization contract --json failed (exit {code}): {err.strip()[-300:]}")
        return one_json_document(out, "diarization contract")

    def doctor(self) -> Dict[str, Any]:
        """`diarization doctor --json` (exit 3 while not ok, the document is printed either way) with `status` for the resolver: ok (every
        op ready), degraded (the Skill runs -- ASSIGN / CHECK -- but DIARIZE is not ready: pyannote.audio, torch or the pipeline
        directory missing), fail (no document, or nothing ready)."""
        if self._doctor is not None:
            return self._doctor
        code, out, err = self._call(["doctor", "--json"], 300.0)
        try:
            doc = one_json_document(out, "diarization doctor")
        except ToolError as e:
            doc = {"ok": False, "ready": {}, "checks": {}, "problems": [f"doctor produced no document (exit {code}): {e} {err.strip()[-200:]}"]}
        ready = as_dict(doc.get("ready"))
        status = "ok" if doc.get("ok") is True and all(v is True for v in ready.values()) and ready else ("degraded" if any(v is True for v in ready.values()) else "fail")
        self._doctor = dict(doc, status=status, problems=[str(p) for p in doc.get("problems") or []], warnings=[str(w) for w in doc.get("warnings") or []])
        return self._doctor

    def operation_status(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """`diarization:<TYPE>` verdicts from the doctor's `ready`: supported / unavailable."""
        ready = as_dict((doc or self.doctor()).get("ready"))
        return {typ: ("supported" if ready.get(op) is True else "unavailable") for op, typ in OPERATION_TYPES.items()}

    def operation_evidence(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Why DIARIZE is (not) ready, in the Skill's own words: the code its diarize would return (PIPELINE_NOT_FOUND when only the pipeline
        directory is missing, ENGINE_UNAVAILABLE when pyannote.audio / torch is, FFMPEG_NOT_FOUND) and the doctor's install lines."""
        doc = doc or self.doctor()
        checks = as_dict(doc.get("checks"))
        ready = as_dict(doc.get("ready"))
        out: Dict[str, Dict[str, Any]] = {}
        for op, typ in OPERATION_TYPES.items():
            ev: Dict[str, Any] = {"ready": ready.get(op) is True}
            if op == "diarize" and ready.get(op) is not True:
                missing = [k for k in ("ffmpeg", "pyannote_audio", "torch", "pipeline") if as_dict(checks.get(k)).get("status") not in (None, "OK")]
                ev["code"] = ("FFMPEG_NOT_FOUND" if "ffmpeg" in missing else "ENGINE_UNAVAILABLE" if {"pyannote_audio", "torch"} & set(missing)
                              else "PIPELINE_NOT_FOUND")
                ev["detail"] = "; ".join(str(p) for p in doc.get("problems") or [])[:300]
                lines: List[str] = []
                for k in missing:
                    inst = as_dict(checks.get(k)).get("install")
                    lines += [str(x) for x in inst] if isinstance(inst, list) else ([str(inst)] if inst else [])
                ev["install"] = " ".join(lines)[:600]
            out[typ] = ev
        return out

    def drift(self) -> List[str]:
        return drift(self.contract)

    def breaking_drift(self) -> List[str]:
        return breaking_drift(self.contract)

    # ---- ToolAdapter
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "skill": self.skill.describe(), "tools": [TOOL_DIARIZE], "drift": self.drift()}

    def package(self) -> SkillPackage:
        return SkillPackage(PACKAGE.skill_id, PACKAGE.name, self.version, PACKAGE.description, list(PACKAGE.capabilities),
                            [ToolSpec(**{**t.__dict__, "version": self.version}) for t in PACKAGE.tools], PACKAGE.repository, PACKAGE.role)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def measurement_args(self, tool: str, kind: str, path: str, asset_id: str, parameters: Dict[str, Any], analysis_id: str, cache_policy: str) -> Optional[Dict[str, Any]]:
        return {"input": path, "asset_id": asset_id, "parameters": {k: v for k, v in (parameters or {}).items() if k in PARAMETERS}}

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            b = self.build(op.args)
        except ToolError as e:
            return [f"<refused: {e}>"]
        return [" ".join(["diarization"] + b["argv"])]

    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        if op.kind != "measure":
            return self._fail(op, attempt, dry_run, time.time(), 2, "INVALID_REQUEST", "diarization runs as an analysis measurement (kind speaker_turns), never as a plan step")
        return self._measure(op, timeout, dry_run, attempt)

    # ---- request construction
    def _input(self, raw: Any) -> str:
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw or "\n" in raw:
            raise ToolError("input must be a non-empty single-line path")
        resolved = os.path.realpath(os.path.abspath(raw))
        if self.allowed_inputs and not any(os.path.normcase(resolved) == os.path.normcase(r) or os.path.normcase(resolved).startswith(os.path.normcase(r).rstrip(os.sep) + os.sep)
                                           for r in self.allowed_inputs):
            raise ToolError(f"input is outside the allowed input roots: {os.path.basename(raw)}")
        if not os.path.isfile(resolved):
            raise ToolError(f"input not found: {os.path.basename(raw)}")
        return resolved

    def build(self, args: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        hit = scan_forbidden(args, FORBIDDEN_ARG_KEYS)
        if hit:
            raise ToolError(f"forbidden key {hit}")
        unknown = sorted(k for k in args if k not in ("input", "asset_id", "parameters"))
        if unknown:
            raise ToolError(f"unknown diarize argument(s): {', '.join(unknown)}")
        if not self.workspace:
            raise ToolError("diarization needs the agent's workspace (the turns file is written there)")
        src = self._input(args.get("input"))
        params = check_parameters(dict(args.get("parameters") or {}))
        asset = str(args.get("asset_id") or "asset")
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", asset):
            raise ToolError(f"invalid asset id {asset!r}")
        out_dir = os.path.join(self.workspace, "cache", "diarization")
        out = os.path.join(out_dir, f"{asset}.turns.json")
        t = max(1.0, min(86400.0, float(timeout or self.default_timeout)))
        options = [f"--{k.replace('_', '-')}={v}" for k, v in params.items()]
        argv = ["diarize", "--json", "--overwrite", f"--output={out}", f"--timeout={t:g}"] + options + ["--", src]
        return {"argv": argv, "input": src, "output": out, "out_dir": out_dir, "params": params, "timeout": t}

    # ---- execution
    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, exit_code: int, code: str, message: str, retryable: bool = False,
              details: Optional[Dict[str, Any]] = None) -> ToolResult:
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "failed",
                                "error": {"code": code, "message": message[:1200], "retryable": bool(retryable), "details": dict(details or {}), "exit_code": exit_code,
                                          "recovery_class": recovery_class(code)}}
        return ToolResult(op.id, op.tool, False, exit_code, None, data, [], f"{SKILL_ID} [{code}] {message[:600]}", round(time.time() - t0, 3), attempt, dry_run)

    def _measure(self, op: Operation, timeout: Optional[float], dry_run: bool, attempt: int) -> ToolResult:
        t0 = time.time()
        if op.tool != TOOL_DIARIZE:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", f"{SKILL_ID} adapter does not support {op.tool}")
        try:
            b = self.build(op.args, timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e))
        command = " ".join(["diarization"] + b["argv"])
        if dry_run:
            return ToolResult(op.id, op.tool, True, 0, None, {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run", "commands": [command]},
                              [command], "", round(time.time() - t0, 3), attempt, True)
        os.makedirs(b["out_dir"], exist_ok=True)
        code, out, err = self._call(b["argv"], b["timeout"] + PROCESS_MARGIN)
        tail = "\n".join(err.strip().splitlines()[-12:])
        if code == 124:
            return self._fail(op, attempt, dry_run, t0, 124, "TIMEOUT", f"diarization diarize exceeded {b['timeout'] + PROCESS_MARGIN:g}s", retryable=True)
        try:
            doc = one_json_document(out, "diarization diarize")
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, code if code else 9, "INVALID_OUTPUT", f"{e} (exit {code}) {tail[-300:]}".strip())
        if doc.get("ok") is not True:
            e_doc = as_dict(doc.get("error"))
            ecode = str(e_doc.get("code") or "ENGINE_FAILED")
            detail = e_doc.get("detail")
            details = {"detail": detail if isinstance(detail, (str, int, float, bool)) or detail is None else json.dumps(detail, default=str)[:300]}
            retry = self.retryable.get(ecode, False) if ecode in ERROR_CLASSES else False
            return self._fail(op, attempt, dry_run, t0, code if code else 1, ecode, str(e_doc.get("message") or tail or "")[:1200], retryable=retry, details=details)
        turns = {k: v for k, v in doc.items() if k not in ("ok", "turns_file")}
        why = check_turns(turns, self.version, sha256_file(b["input"]), b["params"])
        try:
            on_disk = json.loads(Path(b["output"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            on_disk = None
        if on_disk != turns:
            why.append("the turns file on disk is not the document the Skill printed")
        if why:
            return self._fail(op, attempt, dry_run, t0, code if code else 9, "INVALID_OUTPUT", "diarization result not admitted: " + "; ".join(why))
        if code != 0:
            return self._fail(op, attempt, dry_run, t0, code, "INVALID_OUTPUT", f"exit code {code} with an ok result")
        data = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "ok", "turns": turns, "turns_file": b["output"],
                "engine": as_dict(turns.get("engine")), "warnings": [str(w)[:600] for w in turns.get("warnings") or []], "commands": [command]}
        return ToolResult(op.id, op.tool, True, 0, None, data, [command], tail, round(time.time() - t0, 3), attempt, False)


def turns_facts(data: Dict[str, Any]) -> Dict[str, Any]:
    """The observation data of a `speaker_turns` measurement: the turns document as diarization-skill wrote it (no path, no command line,
    no credential appears in it, the Skill's own rule), plus the turns file the agent keeps in its workspace for later merges."""
    return dict(as_dict(data.get("turns")))
