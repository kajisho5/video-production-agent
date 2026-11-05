"""DetectionAdapter: the agent's boundary to detection-skill (ADR-090; detection-skill roadmap step 7, agent side: blur faces -> redact).

    video-production-agent ─(analysis kind face_tracks)─→ DetectionAdapter ─(argv lists, never a shell)─→
        `detection faces --json --overwrite --output=<workspace>/cache/detection/<asset>.faces.json --every=S [--min-score=X]
         [--ffmpeg-skill=<dir>] --timeout=T -- <video>`        (YuNet on frames ffmpeg-skill `look` extracts; boxes, never identities)
        `detection tracks --json --overwrite --output=<workspace>/cache/detection/<asset>.tracks.json [--window=S] -- <faces.json>`
                                                              (boxes linked over time into redact-ready rectangles, a documented heuristic)

Protocol (detection-skill's own contract, `detection contract --json`, pinned from main in contract_0.1.0.json -- the repository has no
release tag, so the pin follows main as diarization / ingest / voice-enhance do):
    contract → `detection contract --json`
    doctor   → `detection doctor --json` (ok, ready.{faces, text, objects, video_frames, tracks, check}, checks with install lines,
               problems; exit 3 while not ok, never downloads a model)
    faces    → `detection-skill/detections@1` (kind face, the sampled frame times, boxes in source pixels, labels always null, the model's
               sha256 on every box, the source fingerprint = sha256 of the file bytes)
    tracks   → `detection-skill/tracks@1` (per track: start / end, the boxes, and `redact` rectangles {start, end, x, y, width, height}
               with even width / height inside the frame, padded and extended by one sampling step)
    failure  → {"ok": false, "error": {code, message, detail}}, the code fixing the exit code (its `errors` table)

A measurement only (faces: role `measurement`, tracks: `transformation` of the detections document; neither writes media): the result
becomes an OBSERVED Observation of kind `face_tracks`. Nothing here blurs, decides or names anyone: the redaction is a CONFIRM decision
(agent/faces.py) executed by ffmpeg-skill cut / redact / broll. Every value is checked before a process starts: the input lies inside
the allowed roots, the files land in the workspace, `every` is in (0, 60] seconds, `min_score` in [0, 1], `window` >= 0. Every option
is one `--flag=value` word and the input path comes after `--`. The result is admitted only when it is about this request: schemas,
tools, Skill version, the fingerprint equal to the input's sha256, face labels null, the sampling and min_score requested, the tracks
made from exactly the detections file written (its sha256), every redact rectangle even-sized inside the frame and inside the media,
and each file on disk equal to the document printed. The Skill's error codes are kept verbatim with the agent's recovery class."""
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
from .locate import PACKAGE_ID, locate_detection  # noqa: F401  (re-exported)

SKILL_ID = PACKAGE_ID
PREFIX = SKILL_ID + "/"
TOOL_FACES = "detection/faces"
TOOL_TRACKS = "detection/tracks"
DETECTIONS_SCHEMA = "detection-skill/detections@1"
TRACKS_SCHEMA = "detection-skill/tracks@1"
SUPPORTED_SKILL_VERSIONS = ("0.1.",)
KIND = "face_tracks"                            # the analysis kind (media/analysis.py) this adapter measures
# the typed parameters the agent may send: faces --every / --min-score, tracks --window (the Skill's own CLI ranges)
PARAMETERS = ("every", "min_score", "window")
MAX_EVERY = 60.0
MAX_WINDOW = 3600.0
# identity keys the Skill never writes on a face (its validator's rule); the agent refuses a result that carries one
IDENTITY_KEYS = ("identity", "name", "person_id", "person", "embedding", "face_id", "recognized_as")
# the per-operation capability `detection:<TYPE>` (resolver) ← the doctor's `ready`; FACES on a video also needs ffmpeg-skill frames
OPERATION_TYPES = {"faces": "FACES"}
DEFAULT_TIMEOUT = 3600.0                        # the Skill's own engine time limit (its --timeout default); the process gets a margin on top
PROCESS_MARGIN = 120.0
# detection-skill's error codes (its contract `errors`) → the agent's recovery class. A missing engine / model / ffmpeg-skill is TOOL_MISSING
# (the user installs it), a bad request INVALID_ARGS, a frame or engine failure SKILL_ERROR; ENGINE_TIMEOUT keeps the Skill's retryable verdict.
ERROR_CLASSES: Dict[str, str] = {
    "INPUT_NOT_FOUND": "INPUT_MISSING", "INVALID_ARGUMENT": "INVALID_ARGS", "UNSUPPORTED_MEDIA": "INVALID_ARGS", "INVALID_DETECTIONS": "INVALID_ARGS",
    "OUTPUT_IS_INPUT": "INVALID_ARGS", "OUTPUT_EXISTS": "INVALID_ARGS", "FFMPEG_SKILL_NOT_FOUND": "TOOL_MISSING", "ENGINE_UNAVAILABLE": "TOOL_MISSING",
    "MODEL_NOT_FOUND": "TOOL_MISSING", "LANGUAGE_NOT_FOUND": "TOOL_MISSING", "FRAME_EXTRACTION_FAILED": "SKILL_ERROR", "ENGINE_FAILED": "SKILL_ERROR",
    "ENGINE_TIMEOUT": "TIMEOUT", "INVALID_RESULT": "SKILL_ERROR",
}
AGENT_ERRORS: Dict[str, str] = {"INVALID_REQUEST": "INVALID_ARGS", "INVALID_OUTPUT": "SKILL_ERROR", "TIMEOUT": "TIMEOUT"}
# contract drift (ADR-045 / ADR-047): every key the adapter reads is compared with the pinned contract; the fatal subset gates availability
DRIFT_KEYS = ("skill_id", "version", "contract_version", "provides", "dependencies", "not_provided", "detections", "tracks", "defaults", "execution", "errors")
DRIFT_TOOL_KEYS = ("role", "writes_media", "mutates_input", "deterministic", "argv")
FATAL_DRIFT_KEYS = ("skill_id", "contract_version", "tracks", "execution")
FATAL_DRIFT_TOOL_KEYS = ("role", "writes_media", "mutates_input", "argv")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def recovery_class(code: str) -> str:
    return ERROR_CLASSES.get(code) or AGENT_ERRORS.get(code) or "SKILL_ERROR"


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_contract(contract: Any) -> List[str]:
    """What the adapter relies on, refused (never patched) when it does not hold: identity, contract and Skill version, FACES as a
    measurement and TRACKS as a transformation that write no media and never touch their input, both schemas, a shell-free offline
    execution with a read-only input, and every error code the adapter maps with its exit code."""
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
    for tid, role in ((TOOL_FACES, "measurement"), (TOOL_TRACKS, "transformation")):
        t = tools.get(tid)
        if t is None:
            errs.append(f"contract lacks tool {tid} (declares {sorted(tools)})")
        elif t.get("role") != role or t.get("writes_media") is not False or t.get("mutates_input") is not False:
            errs.append(f"tool {tid} must declare role {role}, writes_media false, mutates_input false")
    if as_dict(contract.get("detections")).get("schema") != DETECTIONS_SCHEMA:
        errs.append(f"detections.schema {as_dict(contract.get('detections')).get('schema')!r} != {DETECTIONS_SCHEMA}")
    tr = as_dict(contract.get("tracks"))
    if tr.get("schema") != TRACKS_SCHEMA or tr.get("identity") is not False:
        errs.append(f"tracks must declare schema {TRACKS_SCHEMA} and identity false")
    ex = as_dict(contract.get("execution"))
    if ex.get("shell") is not False or ex.get("network") is not False or ex.get("input_is_read_only") is not True:
        errs.append("execution must declare shell false, network false and input_is_read_only true")
    errors = as_dict(contract.get("errors"))
    for code in ERROR_CLASSES:
        if not isinstance(as_dict(errors.get(code)).get("exit_code"), int):
            errs.append(f"errors lack {code} (with its exit_code)")
    return errs


def _used_tools(c: Dict[str, Any]) -> Dict[str, Any]:
    return dict(c, tools=[t for t in c.get("tools") or [] if isinstance(t, dict) and t.get("tool_id") in (TOOL_FACES, TOOL_TRACKS)])


def drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(live, pinned or pinned_contract(), DRIFT_KEYS, "tools", "tool_id", DRIFT_TOOL_KEYS)


def pinned_error_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    """A pinned error code that disappeared or changed its exit code / retryable verdict (a new code is additive: SKILL_ERROR, kept verbatim)."""
    p, lv = as_dict((pinned or pinned_contract()).get("errors")), as_dict(live.get("errors"))
    return [f"errors {c}: pinned {json.dumps(p[c], sort_keys=True)} != live {json.dumps(lv.get(c), sort_keys=True)}" for c in sorted(p) if lv.get(c) != p[c]]


def breaking_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    pinned = pinned or pinned_contract()
    return _breaking_drift(_used_tools(live), _used_tools(pinned), FATAL_DRIFT_KEYS, "tools", "tool_id", FATAL_DRIFT_TOOL_KEYS) + pinned_error_drift(live, pinned)


PACKAGE = SkillPackage(
    skill_id=SKILL_ID, name="detection-skill", version="",
    description="Faces, text (OCR) and objects as measured boxes in time, local and offline (YuNet, Tesseract, YOLOX on CPU, frames from ffmpeg-skill); "
                "boxes linked over time into redact-ready rectangles. It never identifies anyone (EXPERIMENTAL)",
    capabilities=["detection"],
    tools=[ToolSpec(TOOL_FACES, SKILL_ID, description="Face boxes on sampled frames (detection-skill/detections@1, labels always null) linked into redact rectangles "
                                                       "(detection-skill/tracks@1)", required_capabilities=["detection:FACES"], inputs=["input"], produces_output=False,
                    deterministic=False, result_keys=["detections", "tracks", "detections_file", "tracks_file", "engine", "warnings"])],
    repository="kajisho5/detection-skill", role="face / text / object detection (boxes in time, never an identity)")


def check_parameters(params: Dict[str, Any]) -> Dict[str, float]:
    """`every` (faces --every, seconds), `min_score` (faces --min-score) and `window` (tracks --window) as detection-skill's own CLI checks them."""
    unknown = sorted(k for k in params if k not in PARAMETERS)
    if unknown:
        raise ToolError(f"unknown detection parameter(s): {', '.join(unknown)}")
    out: Dict[str, float] = {}
    if params.get("every") is None:
        raise ToolError("every (the sampling interval in seconds) is required: the agent always says how densely frames are sampled")
    if not _num(params["every"]) or not 0 < float(params["every"]) <= MAX_EVERY:
        raise ToolError(f"every must be a number of seconds in (0, {MAX_EVERY:g}], got {params['every']!r}")
    out["every"] = float(params["every"])
    if params.get("min_score") is not None:
        if not _num(params["min_score"]) or not 0.0 <= float(params["min_score"]) <= 1.0:
            raise ToolError(f"min_score must be in [0, 1], got {params['min_score']!r}")
        out["min_score"] = float(params["min_score"])
    if params.get("window") is not None:
        if not _num(params["window"]) or not 0.0 <= float(params["window"]) <= MAX_WINDOW:
            raise ToolError(f"window must be a number of seconds in [0, {MAX_WINDOW:g}], got {params['window']!r}")
        out["window"] = float(params["window"])
    return out


def _face_problems(doc: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    frames = doc.get("frames")
    if not isinstance(frames, list):
        return ["frames is not a list"]
    prev = None
    for fr in frames:
        fr = as_dict(fr)
        t = fr.get("t")
        if not _num(t) or (prev is not None and t <= prev):
            return errs + [f"frame time {t!r} is not a strictly increasing number"]
        prev = t
        for d in fr.get("detections") or []:
            d = as_dict(d)
            if d.get("kind") != "face":
                return errs + [f"a detection of kind {d.get('kind')!r} in a face document"]
            if d.get("label") is not None or any(k in d for k in IDENTITY_KEYS):
                return errs + ["a face carries a label or an identity key (a face box is never an identity)"]
    return errs


def check_detections(doc: Dict[str, Any], version: str, input_sha256: str, params: Dict[str, float]) -> List[str]:
    """Why a faces document is not about this request (empty = admitted)."""
    if doc.get("schema") != DETECTIONS_SCHEMA:
        return [f"schema {doc.get('schema')!r} != {DETECTIONS_SCHEMA}"]
    errs: List[str] = []
    if doc.get("kind") != "face" or doc.get("tool") != TOOL_FACES or doc.get("skill") != "detection-skill" or str(doc.get("skill_version")) != version:
        errs.append(f"result kind / tool / skill {doc.get('kind')!r} / {doc.get('tool')!r} / {doc.get('skill')!r}@{doc.get('skill_version')!r} is not face / {TOOL_FACES}@{version}")
    src = as_dict(doc.get("source"))
    if strip_sha_prefix(src.get("fingerprint")) != input_sha256:
        errs.append("source.fingerprint is not the sha256 of the input file")
    if src.get("media_type") != "video" or not _num(src.get("duration")) or float(src["duration"]) <= 0:
        errs.append("source is not a video with a duration")
    samp = as_dict(doc.get("sampling"))
    if samp.get("method") != "every" or not _num(samp.get("every")) or abs(float(samp["every"]) - params["every"]) > 1e-6:
        errs.append(f"sampling {samp.get('method')!r} every {samp.get('every')!r} is not the requested every {params['every']:g}")
    if "min_score" in params and (not _num(as_dict(doc.get("parameters")).get("min_score")) or abs(float(doc["parameters"]["min_score"]) - params["min_score"]) > 1e-6):
        errs.append(f"parameters.min_score {as_dict(doc.get('parameters')).get('min_score')!r} is not the requested {params['min_score']:g}")
    fs = as_dict(doc.get("frame_size"))
    if not all(isinstance(fs.get(k), int) and fs[k] > 0 for k in ("width", "height")):
        errs.append("frame_size missing")
    eng = as_dict(doc.get("engine"))
    if not eng.get("id") or not eng.get("model_sha256"):
        errs.append("engine id / model_sha256 missing")
    return errs + _face_problems(doc)


def check_tracks(doc: Dict[str, Any], version: str, detections_sha256: str, faces: Dict[str, Any], params: Dict[str, float]) -> List[str]:
    """Why a tracks document is not the linking of exactly the faces document the agent wrote (empty = admitted)."""
    if doc.get("schema") != TRACKS_SCHEMA:
        return [f"schema {doc.get('schema')!r} != {TRACKS_SCHEMA}"]
    errs: List[str] = []
    if doc.get("kind") != "face" or doc.get("skill") != "detection-skill" or str(doc.get("skill_version")) != version:
        errs.append(f"tracks kind / skill {doc.get('kind')!r} / {doc.get('skill')!r}@{doc.get('skill_version')!r} is not face / detection-skill@{version}")
    if strip_sha_prefix(as_dict(doc.get("detections")).get("fingerprint")) != detections_sha256:
        errs.append("detections.fingerprint is not the sha256 of the faces file the agent wrote")
    if as_dict(doc.get("method")).get("identity") is not False:
        errs.append("method.identity must be false")
    if doc.get("frame_size") != faces.get("frame_size") or strip_sha_prefix(as_dict(doc.get("source")).get("fingerprint")) != strip_sha_prefix(as_dict(faces.get("source")).get("fingerprint")):
        errs.append("tracks name another source or frame size than the faces document")
    if "window" in params and (not _num(as_dict(doc.get("parameters")).get("window")) or abs(float(doc["parameters"]["window"]) - params["window"]) > 1e-6):
        errs.append(f"parameters.window {as_dict(doc.get('parameters')).get('window')!r} is not the requested {params['window']:g}")
    w, h = as_dict(doc.get("frame_size")).get("width"), as_dict(doc.get("frame_size")).get("height")
    dur = as_dict(doc.get("source")).get("duration")
    tracks = doc.get("tracks")
    if not isinstance(tracks, list):
        return errs + ["tracks is not a list"]
    for t in tracks:
        t = as_dict(t)
        if t.get("label") is not None or any(k in t for k in IDENTITY_KEYS):
            return errs + [f"track {t.get('id')!r} carries a label or an identity key"]
        rects = t.get("redact")
        if not isinstance(rects, list) or not rects:
            return errs + [f"track {t.get('id')!r} has no redact rectangle"]
        for r in rects:
            r = as_dict(r)
            if not all(isinstance(r.get(k), int) and not isinstance(r.get(k), bool) for k in ("x", "y", "width", "height")) or not (_num(r.get("start")) and _num(r.get("end"))):
                return errs + [f"track {t.get('id')!r}: a redact rectangle without integer x / y / width / height and a start / end"]
            if r["width"] < 2 or r["height"] < 2 or r["width"] % 2 or r["height"] % 2 or r["x"] < 0 or r["y"] < 0 \
                    or not isinstance(w, int) or not isinstance(h, int) or r["x"] + r["width"] > w or r["y"] + r["height"] > h:
                return errs + [f"track {t.get('id')!r}: redact rectangle {r} is not even-sized inside the {w}x{h} frame"]
            if not 0 <= float(r["start"]) < float(r["end"]) or (_num(dur) and float(r["end"]) > float(dur) + 0.001):
                return errs + [f"track {t.get('id')!r}: redact span {r.get('start')}..{r.get('end')} is not inside the media"]
    return errs


class DetectionAdapter(ToolAdapter):
    owns_cache = False   # the Skill keeps no cache: the agent's ObservationCache stores the result like any other measurement

    def __init__(self, skill: CliSkill, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None, ffmpeg_skill_dir: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT):
        self.skill = skill
        self.name = SKILL_ID
        self.workspace = os.path.realpath(os.path.abspath(workspace)) if workspace else None
        self.allowed_inputs = [os.path.realpath(os.path.abspath(p)) for p in (allowed_inputs or [])]
        self.ffmpeg_skill_dir = os.path.realpath(os.path.abspath(ffmpeg_skill_dir)) if ffmpeg_skill_dir else None   # always named: never the Skill's own search
        self.default_timeout = float(timeout)
        self.contract = self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("incompatible detection-skill contract: " + "; ".join(errs))
        self.version = str(self.contract.get("version"))
        self.tools = {TOOL_FACES: True}
        self.retryable = {c: bool(as_dict(v).get("retryable")) for c, v in as_dict(self.contract.get("errors")).items()}
        self._doctor: Optional[Dict[str, Any]] = None

    # ---- contract / doctor
    def _call(self, argv: List[str], timeout: Optional[float]) -> Tuple[int, str, str]:
        return invoke(self.skill, argv, timeout=timeout)

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._call(["contract", "--json"], 60.0)
        if code != 0:
            raise ContractError(f"detection contract --json failed (exit {code}): {err.strip()[-300:]}")
        return one_json_document(out, "detection contract")

    def doctor(self) -> Dict[str, Any]:
        """`detection doctor --json` (exit 3 while not ok, the document is printed either way) with `status` for the resolver: ok (every
        engine ready), degraded (the Skill runs -- tracks / check at least -- but an engine, model or ffmpeg-skill is missing), fail (no
        document, or nothing ready). ffmpeg-skill is always the agent's own checkout."""
        if self._doctor is not None:
            return self._doctor
        argv = ["doctor", "--json"] + ([f"--ffmpeg-skill={self.ffmpeg_skill_dir}"] if self.ffmpeg_skill_dir else [])
        code, out, err = self._call(argv, 300.0)
        try:
            doc = one_json_document(out, "detection doctor")
        except ToolError as e:
            doc = {"ok": False, "ready": {}, "checks": {}, "problems": [f"doctor produced no document (exit {code}): {e} {err.strip()[-200:]}"]}
        ready = as_dict(doc.get("ready"))
        status = "ok" if doc.get("ok") is True and ready and all(v is True for v in ready.values()) else ("degraded" if any(v is True for v in ready.values()) else "fail")
        self._doctor = dict(doc, status=status, problems=[str(p) for p in doc.get("problems") or []], warnings=[str(w) for w in doc.get("warnings") or []])
        return self._doctor

    def operation_status(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """`detection:FACES` from the doctor: faces on a video need YuNet (opencv + the model file) and ffmpeg-skill's frames."""
        ready = as_dict((doc or self.doctor()).get("ready"))
        return {"FACES": "supported" if ready.get("faces") is True and ready.get("video_frames") is True else "unavailable"}

    def operation_evidence(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Why FACES is (not) ready, in the Skill's own words: the code its faces command would return (FFMPEG_SKILL_NOT_FOUND when
        frames cannot be extracted, ENGINE_UNAVAILABLE without opencv / numpy, MODEL_NOT_FOUND without the YuNet file) and the doctor's install lines."""
        doc = doc or self.doctor()
        checks = as_dict(doc.get("checks"))
        ready = as_dict(doc.get("ready"))
        ev: Dict[str, Any] = {"ready": ready.get("faces") is True and ready.get("video_frames") is True}
        if not ev["ready"]:
            missing = [k for k in ("ffmpeg_skill", "opencv", "numpy", "face_model") if as_dict(checks.get(k)).get("status") not in (None, "OK")]
            ev["code"] = ("FFMPEG_SKILL_NOT_FOUND" if "ffmpeg_skill" in missing or ready.get("video_frames") is not True else
                          "ENGINE_UNAVAILABLE" if {"opencv", "numpy"} & set(missing) else "MODEL_NOT_FOUND")
            ev["detail"] = "; ".join(str(p) for p in doc.get("problems") or [])[:300]
            lines: List[str] = []
            for k in missing:
                inst = as_dict(checks.get(k)).get("install")
                lines += [str(x) for x in inst] if isinstance(inst, list) else ([str(inst)] if inst else [])
            ev["install"] = " ".join(lines)[:600]
        return {"FACES": ev}

    def drift(self) -> List[str]:
        return drift(self.contract)

    def breaking_drift(self) -> List[str]:
        return breaking_drift(self.contract)

    # ---- ToolAdapter
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "skill": self.skill.describe(), "tools": [TOOL_FACES], "drift": self.drift()}

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
        return [" ".join(["detection"] + b["faces_argv"]), " ".join(["detection"] + b["tracks_argv"])]

    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        if op.kind != "measure":
            return self._fail(op, attempt, dry_run, time.time(), 2, "INVALID_REQUEST", "detection runs as an analysis measurement (kind face_tracks), never as a plan step")
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
            raise ToolError(f"unknown detection argument(s): {', '.join(unknown)}")
        if not self.workspace:
            raise ToolError("detection needs the agent's workspace (the detections and tracks files are written there)")
        src = self._input(args.get("input"))
        params = check_parameters(dict(args.get("parameters") or {}))
        asset = str(args.get("asset_id") or "asset")
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", asset):
            raise ToolError(f"invalid asset id {asset!r}")
        out_dir = os.path.join(self.workspace, "cache", "detection")
        faces_out, tracks_out = os.path.join(out_dir, f"{asset}.faces.json"), os.path.join(out_dir, f"{asset}.tracks.json")
        t = max(1.0, min(86400.0, float(timeout or self.default_timeout)))
        faces = ["faces", "--json", "--overwrite", f"--output={faces_out}", f"--every={params['every']:g}"]
        if "min_score" in params:
            faces.append(f"--min-score={params['min_score']:g}")
        if self.ffmpeg_skill_dir:
            faces.append(f"--ffmpeg-skill={self.ffmpeg_skill_dir}")
        faces += [f"--timeout={t:g}", "--", src]
        tracks = ["tracks", "--json", "--overwrite", f"--output={tracks_out}"] + ([f"--window={params['window']:g}"] if "window" in params else []) + ["--", faces_out]
        return {"faces_argv": faces, "tracks_argv": tracks, "input": src, "faces_out": faces_out, "tracks_out": tracks_out, "out_dir": out_dir, "params": params, "timeout": t}

    # ---- execution
    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, exit_code: int, code: str, message: str, retryable: bool = False,
              details: Optional[Dict[str, Any]] = None) -> ToolResult:
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "failed",
                                "error": {"code": code, "message": message[:1200], "retryable": bool(retryable), "details": dict(details or {}), "exit_code": exit_code,
                                          "recovery_class": recovery_class(code)}}
        return ToolResult(op.id, op.tool, False, exit_code, None, data, [], f"{SKILL_ID} [{code}] {message[:600]}", round(time.time() - t0, 3), attempt, dry_run)

    def _step(self, op: Operation, attempt: int, t0: float, argv: List[str], timeout: float, who: str) -> Tuple[Optional[Dict[str, Any]], Optional[ToolResult], str]:
        """One Skill call → (its ok document, None, stderr tail) or (None, the failed result)."""
        code, out, err = self._call(argv, timeout + PROCESS_MARGIN)
        tail = "\n".join(err.strip().splitlines()[-12:])
        if code == 124:
            return None, self._fail(op, attempt, False, t0, 124, "TIMEOUT", f"detection {who} exceeded {timeout + PROCESS_MARGIN:g}s", retryable=True), tail
        try:
            doc = one_json_document(out, f"detection {who}")
        except ToolError as e:
            return None, self._fail(op, attempt, False, t0, code if code else 9, "INVALID_OUTPUT", f"{e} (exit {code}) {tail[-300:]}".strip()), tail
        if doc.get("ok") is not True:
            e_doc = as_dict(doc.get("error"))
            ecode = str(e_doc.get("code") or "ENGINE_FAILED")
            detail = e_doc.get("detail")
            details = {"detail": detail if isinstance(detail, (str, int, float, bool)) or detail is None else json.dumps(detail, default=str)[:300]}
            retry = self.retryable.get(ecode, False) if ecode in ERROR_CLASSES else False
            return None, self._fail(op, attempt, False, t0, code if code else 1, ecode, str(e_doc.get("message") or tail or "")[:1200], retryable=retry, details=details), tail
        if code != 0:
            return None, self._fail(op, attempt, False, t0, code, "INVALID_OUTPUT", f"detection {who}: exit code {code} with an ok result"), tail
        return doc, None, tail

    def _measure(self, op: Operation, timeout: Optional[float], dry_run: bool, attempt: int) -> ToolResult:
        t0 = time.time()
        if op.tool != TOOL_FACES:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", f"{SKILL_ID} adapter does not support {op.tool}")
        try:
            b = self.build(op.args, timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e))
        commands = [" ".join(["detection"] + b["faces_argv"]), " ".join(["detection"] + b["tracks_argv"])]
        if dry_run:
            return ToolResult(op.id, op.tool, True, 0, None, {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run", "commands": commands},
                              commands, "", round(time.time() - t0, 3), attempt, True)
        os.makedirs(b["out_dir"], exist_ok=True)
        for stale in (b["faces_out"], b["tracks_out"]):   # an earlier run's file must never pass for this run's (the on-disk check below)
            if os.path.exists(stale):
                os.remove(stale)
        doc, failed, tail = self._step(op, attempt, t0, b["faces_argv"], b["timeout"], "faces")
        if failed is not None:
            return failed
        faces = {k: v for k, v in doc.items() if k not in ("ok", "detections_file")}
        why = check_detections(faces, self.version, sha256_file(b["input"]), b["params"])
        if _read_json(b["faces_out"]) != faces:
            why.append("the faces file on disk is not the document the Skill printed")
        if why:
            return self._fail(op, attempt, dry_run, t0, 9, "INVALID_OUTPUT", "detection faces result not admitted: " + "; ".join(why))
        doc, failed, tail2 = self._step(op, attempt, t0, b["tracks_argv"], 300.0, "tracks")
        if failed is not None:
            return failed
        tracks = {k: v for k, v in doc.items() if k not in ("ok", "tracks_file")}
        why = check_tracks(tracks, self.version, sha256_file(b["faces_out"]), faces, b["params"])
        if _read_json(b["tracks_out"]) != tracks:
            why.append("the tracks file on disk is not the document the Skill printed")
        if why:
            return self._fail(op, attempt, dry_run, t0, 9, "INVALID_OUTPUT", "detection tracks result not admitted: " + "; ".join(why))
        data = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "ok", "detections": detection_summary(faces), "tracks": tracks,
                "detections_file": b["faces_out"], "tracks_file": b["tracks_out"], "engine": as_dict(faces.get("engine")),
                "warnings": [str(w)[:600] for w in list(faces.get("warnings") or []) + list(tracks.get("warnings") or [])], "commands": commands}
        return ToolResult(op.id, op.tool, True, 0, None, data, commands, (tail + "\n" + tail2).strip(), round(time.time() - t0, 3), attempt, False)


def _read_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def detection_summary(faces: Dict[str, Any]) -> Dict[str, Any]:
    """What the observation keeps of the faces document besides the tracks (the boxes stay in the file the agent wrote): how frames were
    sampled, the counts, the engine and model with its sha256, the parameters, the frame size and the source."""
    samp = as_dict(faces.get("sampling"))
    return {"sampling": {k: samp.get(k) for k in ("method", "every", "frames_tool", "ffmpeg_skill_version")} | {"frames": len(samp.get("times") or [])},
            "counts": as_dict(faces.get("counts")), "engine": {k: as_dict(faces.get("engine")).get(k) for k in ("id", "version", "model", "model_sha256", "model_license", "engine_license")},
            "parameters": as_dict(faces.get("parameters")), "frame_size": as_dict(faces.get("frame_size")), "source": as_dict(faces.get("source"))}
