"""SlideSyncAdapter: the agent's boundary to slide-sync-skill (ADR-091; slide-sync-skill roadmap step 6, agent side: the adapter).

    video-production-agent ─(analysis kind slide_timeline)─→ SlideSyncAdapter ─(argv lists, never a shell)─→
        `slide-sync detect --json --overwrite --output=<workspace>/cache/slide-sync/<asset>.timeline.json [--threshold=X] [--min-dwell=S]
         --ffmpeg-skill=<dir> --timeout=T -- <video>`                         (measured picture cuts held at least min_dwell seconds)
      and, when the user named the speaker's deck:
        `slide-sync import --json --overwrite --output=<workspace>/cache/slide-sync/<asset>.deck -- <deck>`   (one PNG + text per page)
        `slide-sync match --json --overwrite --output=<…>.match.json --deck=<…>.deck --timeline=<…>.timeline.json --ffmpeg-skill=<dir>
         --timeout=T -- <video>`                                               (which page is on screen in each slide, or null with the reason)

Protocol (slide-sync-skill's own contract, `slide-sync contract --json`, pinned from main in contract_0.1.0.json -- the repository has no
release tag): detect → `slide-sync/timeline@1` (slides {index, start, end, duration}, changes, the detector's threshold / min_dwell and
engine version), import → `slide-sync/deck@1` (pages with their own text layer, the deck's sha256), match → `slide-sync/match@1`
(segments with page / confidence / margin / view / reason, page null when no page is clearly best). doctor → engine (ffmpeg-skill) ok and
import readiness (pypdfium2 for PDF, LibreOffice for pptx / key / odp). Failure → {"ok": false, "error": {code, message, detail}}.

A measurement only: nothing here writes media the plan delivers (import's page PNGs land in the agent's cache), cuts, names a slide or
decides an edit. Every value is checked before a process starts (input and deck inside the allowed roots, files in the workspace,
threshold in (0, 100], min_dwell > 0, a deck extension the Skill imports); every option is one `--flag=value` word and the input comes
after `--`. A result is admitted only when it is about this request: schemas, the Skill version, the requested threshold / min_dwell,
the timeline's own video, slides indexed 1..n in time order inside the media; the match covering exactly those slides, pages inside the
imported deck, the deck's sha256 = the file the user named; each file on disk equal to the document printed."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...models import Operation, ToolResult
from ...skills.contract import SkillPackage, ToolSpec
from ..base import ToolAdapter, ToolError
from ..skill_process import (FORBIDDEN_ARG_KEYS, CliSkill, ContractError, as_dict, breaking_drift as _breaking_drift, drift_report, invoke, one_json_document,
                             scan_forbidden, sha256_file)
from .locate import PACKAGE_ID, locate_slide_sync  # noqa: F401  (re-exported)

SKILL_ID = PACKAGE_ID
PREFIX = SKILL_ID + "/"
TOOL_DETECT, TOOL_IMPORT, TOOL_MATCH = "slide-sync/detect", "slide-sync/import", "slide-sync/match"
TIMELINE_SCHEMA, DECK_SCHEMA, MATCH_SCHEMA = "slide-sync/timeline@1", "slide-sync/deck@1", "slide-sync/match@1"
SUPPORTED_SKILL_VERSIONS = ("0.1.",)
KIND = "slide_timeline"                         # the analysis kind (media/analysis.py) this adapter measures
PARAMETERS = ("threshold", "min_dwell", "deck")
DECK_EXTENSIONS = (".pdf", ".pptx", ".ppt", ".key", ".odp")   # the contract's import argv; all but PDF need LibreOffice
END_TOLERANCE = 0.05
# the per-operation capability `slide-sync:<TYPE>` (resolver) ← the doctor: DETECT / MATCH need the engine, IMPORT_PDF pypdfium2, IMPORT_OFFICE also LibreOffice
OPERATION_TYPES = ("DETECT", "MATCH", "IMPORT_PDF", "IMPORT_OFFICE")
DEFAULT_TIMEOUT = 3600.0
PROCESS_MARGIN = 120.0
ERROR_CLASSES: Dict[str, str] = {
    "INPUT_NOT_FOUND": "INPUT_MISSING", "INVALID_ARGUMENT": "INVALID_ARGS", "INVALID_TIMELINE": "INVALID_ARGS", "OUTPUT_IS_INPUT": "INVALID_ARGS",
    "INVALID_DECK": "INVALID_ARGS", "INVALID_IMAGE": "INVALID_ARGS", "DEPENDENCY_MISSING": "TOOL_MISSING", "ENGINE_NOT_FOUND": "TOOL_MISSING",
    "ENGINE_TOOL_MISSING": "TOOL_MISSING", "ENGINE_FAILED": "SKILL_ERROR", "ENGINE_TIMEOUT": "TIMEOUT", "CONVERT_FAILED": "SKILL_ERROR", "CONVERT_TIMEOUT": "TIMEOUT",
}
AGENT_ERRORS: Dict[str, str] = {"INVALID_REQUEST": "INVALID_ARGS", "INVALID_OUTPUT": "SKILL_ERROR", "TIMEOUT": "TIMEOUT"}
USED_TOOLS = (TOOL_DETECT, TOOL_IMPORT, TOOL_MATCH)
DRIFT_KEYS = ("skill_id", "version", "contract_version", "provides", "dependencies", "not_provided", "timeline", "match", "deck", "execution", "errors")
DRIFT_TOOL_KEYS = ("role", "writes_media", "mutates_input", "deterministic", "argv")
FATAL_DRIFT_KEYS = ("skill_id", "contract_version", "execution")
FATAL_DRIFT_TOOL_KEYS = ("role", "mutates_input", "argv")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def recovery_class(code: str) -> str:
    return ERROR_CLASSES.get(code) or AGENT_ERRORS.get(code) or "SKILL_ERROR"


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_contract(contract: Any) -> List[str]:
    """What the adapter relies on, refused (never patched) when it does not hold: identity, contract and Skill version, detect and match as
    measurements that write no media, import as a measurement that never touches its input, the three schemas, a shell-free offline
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
    for tid in USED_TOOLS:
        t = tools.get(tid)
        if t is None:
            errs.append(f"contract lacks tool {tid} (declares {sorted(tools)})")
        elif t.get("role") != "measurement" or t.get("mutates_input") is not False or (tid != TOOL_IMPORT and t.get("writes_media") is not False):
            errs.append(f"tool {tid} must declare role measurement, mutates_input false" + ("" if tid == TOOL_IMPORT else ", writes_media false"))
    for key, schema in (("timeline", TIMELINE_SCHEMA), ("match", MATCH_SCHEMA), ("deck", DECK_SCHEMA)):
        if as_dict(contract.get(key)).get("schema") != schema:
            errs.append(f"{key}.schema {as_dict(contract.get(key)).get('schema')!r} != {schema}")
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
    out = _breaking_drift(_used(live), _used(pinned), FATAL_DRIFT_KEYS, "tools", "tool_id", FATAL_DRIFT_TOOL_KEYS) + pinned_error_drift(live, pinned)
    for key in ("timeline", "match", "deck"):   # the schemas the results are read by
        if as_dict(live.get(key)).get("schema") != as_dict(pinned.get(key)).get("schema"):
            out.append(f"{key}.schema: pinned {as_dict(pinned.get(key)).get('schema')!r} != live {as_dict(live.get(key)).get('schema')!r}")
    return out


PACKAGE = SkillPackage(
    skill_id=SKILL_ID, name="slide-sync-skill", version="",
    description="Where the slides change in a talk recording (measured picture cuts held a minimum time, through ffmpeg-skill), and which page of the "
                "speaker's deck is on screen in each slide (or not matched); local, no network (EXPERIMENTAL)",
    capabilities=["slide-sync"],
    tools=[ToolSpec(TOOL_DETECT, SKILL_ID, description="Slide timeline of a recording (slide-sync/timeline@1), and with a deck its page match (slide-sync/match@1)",
                    required_capabilities=["slide-sync:DETECT"], inputs=["input"], produces_output=False, deterministic=True,
                    result_keys=["timeline", "match", "deck", "timeline_file", "match_file", "warnings"])],
    repository="kajisho5/slide-sync-skill", role="slide changes and deck pages of a talk recording (measured, never interpreted)")


def check_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """threshold (0, 100], min_dwell > 0 (detect's own checks), deck a single-line path with an extension import takes."""
    unknown = sorted(k for k in params if k not in PARAMETERS)
    if unknown:
        raise ToolError(f"unknown slide-sync parameter(s): {', '.join(unknown)}")
    out: Dict[str, Any] = {}
    if params.get("threshold") is not None:
        if not _num(params["threshold"]) or not 0 < float(params["threshold"]) <= 100:
            raise ToolError(f"threshold must be in (0, 100], got {params['threshold']!r}")
        out["threshold"] = float(params["threshold"])
    if params.get("min_dwell") is not None:
        if not _num(params["min_dwell"]) or not 0 < float(params["min_dwell"]) <= 3600:
            raise ToolError(f"min_dwell must be a number of seconds in (0, 3600], got {params['min_dwell']!r}")
        out["min_dwell"] = float(params["min_dwell"])
    if params.get("deck") is not None:
        d = params["deck"]
        if not isinstance(d, str) or not d.strip() or "\x00" in d or "\n" in d or Path(d).suffix.lower() not in DECK_EXTENSIONS:
            raise ToolError(f"deck must be a path to a {' / '.join(DECK_EXTENSIONS)} file")
        out["deck"] = d
    return out


def check_timeline(doc: Dict[str, Any], version: str, video: str, params: Dict[str, Any]) -> List[str]:
    """Why a timeline is not this request's (empty = admitted)."""
    if doc.get("schema") != TIMELINE_SCHEMA:
        return [f"schema {doc.get('schema')!r} != {TIMELINE_SCHEMA}"]
    errs: List[str] = []
    det = as_dict(doc.get("detector"))
    if det.get("skill") != SKILL_ID or str(det.get("skill_version")) != version:
        errs.append(f"detector {det.get('skill')!r}@{det.get('skill_version')!r} is not {SKILL_ID}@{version}")
    for k in ("threshold", "min_dwell"):
        if k in params and (not _num(det.get(k)) or abs(float(det[k]) - params[k]) > 1e-9):
            errs.append(f"detector.{k} {det.get(k)!r} is not the requested {params[k]:g}")
    src = as_dict(doc.get("source"))
    if os.path.normcase(os.path.realpath(str(src.get("path") or ""))) != os.path.normcase(os.path.realpath(video)):
        errs.append("source.path is not the input video")
    dur = src.get("duration")
    if not _num(dur) or dur <= 0:
        return errs + ["source.duration missing"]
    slides = doc.get("slides")
    if not isinstance(slides, list) or not slides or doc.get("slide_count") != len(slides):
        return errs + ["slides missing or slide_count wrong"]
    prev = 0.0
    for n, s in enumerate(slides, start=1):
        s = as_dict(s)
        if s.get("index") != n or not (_num(s.get("start")) and _num(s.get("end"))) or not prev - 0.001 <= s["start"] < s["end"] or s["end"] > dur + END_TOLERANCE:
            return errs + [f"slide {n} {s!r} is not the next slide in time order inside the media"]
        prev = s["end"]
    if doc.get("provenance") != "OBSERVED":
        errs.append("provenance is not OBSERVED")
    return errs


def check_match(doc: Dict[str, Any], version: str, timeline: Dict[str, Any], deck: Dict[str, Any], deck_sha256: str) -> List[str]:
    """Why a match / deck pair is not about this timeline and the deck the user named (empty = admitted)."""
    if deck.get("schema") != DECK_SCHEMA or doc.get("schema") != MATCH_SCHEMA:
        return [f"schemas {deck.get('schema')!r} / {doc.get('schema')!r} are not {DECK_SCHEMA} / {MATCH_SCHEMA}"]
    errs: List[str] = []
    if as_dict(deck.get("source")).get("sha256") != deck_sha256:
        errs.append("deck.source.sha256 is not the sha256 of the deck the user named")
    pages = deck.get("pages") if isinstance(deck.get("pages"), list) else []
    if deck.get("page_count") != len(pages) or not pages:
        errs.append("deck page_count is not the number of pages")
    m = as_dict(doc.get("matcher"))
    if m.get("skill") != SKILL_ID or str(m.get("skill_version")) != version:
        errs.append(f"matcher {m.get('skill')!r}@{m.get('skill_version')!r} is not {SKILL_ID}@{version}")
    if as_dict(doc.get("deck")).get("page_count") != len(pages):
        errs.append("the match names another deck (page count)")
    segs = doc.get("segments") if isinstance(doc.get("segments"), list) else []
    slides = timeline.get("slides") or []
    if len(segs) != len(slides) or doc.get("segment_count") != len(segs):
        return errs + [f"{len(segs)} matched segment(s) for {len(slides)} slide(s)"]
    for s, sl in zip(segs, slides):
        s = as_dict(s)
        if (s.get("index"), s.get("start"), s.get("end")) != (sl.get("index"), sl.get("start"), sl.get("end")):
            return errs + [f"segment {s.get('index')!r} is not slide {sl.get('index')}'s span"]
        page = s.get("page")
        if page is not None and (isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= len(pages)):
            return errs + [f"segment {s.get('index')}: page {page!r} is not a page of the deck"]
        if page is None and not s.get("reason"):
            return errs + [f"segment {s.get('index')}: no page and no reason"]
    return errs


class SlideSyncAdapter(ToolAdapter):
    owns_cache = False

    def __init__(self, skill: CliSkill, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None, ffmpeg_skill_dir: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT):
        self.skill = skill
        self.name = SKILL_ID
        self.workspace = os.path.realpath(os.path.abspath(workspace)) if workspace else None
        self.allowed_inputs = [os.path.realpath(os.path.abspath(p)) for p in (allowed_inputs or [])]
        self.ffmpeg_skill_dir = os.path.realpath(os.path.abspath(ffmpeg_skill_dir)) if ffmpeg_skill_dir else None
        self.default_timeout = float(timeout)
        self.contract = self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("incompatible slide-sync-skill contract: " + "; ".join(errs))
        self.version = str(self.contract.get("version"))
        self.tools = {TOOL_DETECT: True}
        self.retryable = {c: bool(as_dict(v).get("retryable")) for c, v in as_dict(self.contract.get("errors")).items()}
        self._doctor: Optional[Dict[str, Any]] = None

    def _call(self, argv: List[str], timeout: Optional[float]) -> Tuple[int, str, str]:
        return invoke(self.skill, argv, timeout=timeout)

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._call(["contract", "--json"], 60.0)
        if code != 0:
            raise ContractError(f"slide-sync contract --json failed (exit {code}): {err.strip()[-300:]}")
        return one_json_document(out, "slide-sync contract")

    def doctor(self) -> Dict[str, Any]:
        """`slide-sync doctor --json --ffmpeg-skill=<the agent's checkout>` with `status`: ok (engine ready and PDF import ready), degraded (the
        engine is ready, import is not: detect / match run, a deck cannot be imported), fail (no document, or the engine not ready)."""
        if self._doctor is not None:
            return self._doctor
        argv = ["doctor", "--json"] + ([f"--ffmpeg-skill={self.ffmpeg_skill_dir}"] if self.ffmpeg_skill_dir else [])
        code, out, err = self._call(argv, 300.0)
        try:
            doc = one_json_document(out, "slide-sync doctor")
        except ToolError as e:
            doc = {"ok": False, "engine": {}, "import": {}, "problems": [f"doctor produced no document (exit {code}): {e} {err.strip()[-200:]}"]}
        engine_ok = as_dict(doc.get("engine")).get("ok") is True and doc.get("ok") is True
        pdf_ok = as_dict(as_dict(doc.get("import")).get("ready")).get("pdf") is True
        status = "ok" if engine_ok and pdf_ok else ("degraded" if engine_ok else "fail")
        self._doctor = dict(doc, status=status, problems=[str(p) for p in doc.get("problems") or []], warnings=[str(n) for n in doc.get("notes") or []])
        return self._doctor

    def operation_status(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        doc = doc or self.doctor()
        engine_ok = as_dict(doc.get("engine")).get("ok") is True and doc.get("ok") is True
        ready = as_dict(as_dict(doc.get("import")).get("ready"))
        ok = {"DETECT": engine_ok, "MATCH": engine_ok, "IMPORT_PDF": ready.get("pdf") is True, "IMPORT_OFFICE": ready.get("pptx_key_odp") is True}
        return {typ: ("supported" if ok[typ] else "unavailable") for typ in OPERATION_TYPES}

    def operation_evidence(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Why an operation is not ready, in the Skill's own words: ENGINE_NOT_FOUND / ENGINE_TOOL_MISSING for detect / match, DEPENDENCY_MISSING with
        the install line (pypdfium2, LibreOffice) for an import."""
        doc = doc or self.doctor()
        st = self.operation_status(doc)
        imp = as_dict(doc.get("import"))
        out: Dict[str, Dict[str, Any]] = {}
        for typ in OPERATION_TYPES:
            ev: Dict[str, Any] = {"ready": st[typ] == "supported"}
            if not ev["ready"]:
                if typ in ("DETECT", "MATCH"):
                    ev["code"] = "ENGINE_TOOL_MISSING" if as_dict(doc.get("engine")).get("found") else "ENGINE_NOT_FOUND"
                    ev["detail"] = "; ".join(str(p) for p in doc.get("problems") or [])[:300]
                else:
                    ev["code"] = "DEPENDENCY_MISSING"
                    lines = [str(as_dict(imp.get("pdf")).get("install") or "")] + ([str(as_dict(imp.get("libreoffice")).get("install") or "")] if typ == "IMPORT_OFFICE" else [])
                    ev["install"] = " ".join(x for x in lines if x and x != "None")[:600]
            out[typ] = ev
        return out

    def drift(self) -> List[str]:
        return drift(self.contract)

    def breaking_drift(self) -> List[str]:
        return breaking_drift(self.contract)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "skill": self.skill.describe(), "tools": [TOOL_DETECT], "drift": self.drift()}

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
        return [" ".join(["slide-sync"] + argv) for argv in b["calls"]]

    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        if op.kind != "measure":
            return self._fail(op, attempt, dry_run, time.time(), 2, "INVALID_REQUEST", "slide-sync runs as an analysis measurement (kind slide_timeline), never as a plan step")
        return self._measure(op, timeout, dry_run, attempt)

    def _path(self, raw: Any, what: str) -> str:
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw or "\n" in raw:
            raise ToolError(f"{what} must be a non-empty single-line path")
        resolved = os.path.realpath(os.path.abspath(raw))
        if self.allowed_inputs and not any(os.path.normcase(resolved) == os.path.normcase(r) or os.path.normcase(resolved).startswith(os.path.normcase(r).rstrip(os.sep) + os.sep)
                                           for r in self.allowed_inputs):
            raise ToolError(f"{what} is outside the allowed input roots: {os.path.basename(raw)}")
        if not os.path.isfile(resolved):
            raise ToolError(f"{what} not found: {os.path.basename(raw)}")
        return resolved

    def build(self, args: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        hit = scan_forbidden(args, FORBIDDEN_ARG_KEYS)
        if hit:
            raise ToolError(f"forbidden key {hit}")
        unknown = sorted(k for k in args if k not in ("input", "asset_id", "parameters"))
        if unknown:
            raise ToolError(f"unknown slide-sync argument(s): {', '.join(unknown)}")
        if not self.workspace:
            raise ToolError("slide-sync needs the agent's workspace (its files are written there)")
        src = self._path(args.get("input"), "input")
        params = check_parameters(dict(args.get("parameters") or {}))
        deck = self._path(params["deck"], "deck") if "deck" in params else None
        asset = str(args.get("asset_id") or "asset")
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", asset):
            raise ToolError(f"invalid asset id {asset!r}")
        out_dir = os.path.join(self.workspace, "cache", "slide-sync")
        files = {"timeline": os.path.join(out_dir, f"{asset}.timeline.json"), "deck": os.path.join(out_dir, f"{asset}.deck"), "match": os.path.join(out_dir, f"{asset}.match.json")}
        t = max(1.0, min(86400.0, float(timeout or self.default_timeout)))
        engine = [f"--ffmpeg-skill={self.ffmpeg_skill_dir}"] if self.ffmpeg_skill_dir else []
        detect = ["detect", "--json", "--overwrite", f"--output={files['timeline']}"] + [f"--{k.replace('_', '-')}={params[k]:g}" for k in ("threshold", "min_dwell") if k in params] \
            + engine + [f"--timeout={t:g}", "--", src]
        calls = [detect]
        if deck:
            calls.append(["import", "--json", "--overwrite", f"--output={files['deck']}", "--", deck])
            calls.append(["match", "--json", "--overwrite", f"--output={files['match']}", f"--deck={files['deck']}", f"--timeline={files['timeline']}"] + engine
                         + [f"--timeout={t:g}", "--", src])
        return {"calls": calls, "input": src, "deck": deck, "files": files, "out_dir": out_dir, "params": params, "timeout": t}

    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, exit_code: int, code: str, message: str, retryable: bool = False,
              details: Optional[Dict[str, Any]] = None) -> ToolResult:
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "failed",
                                "error": {"code": code, "message": message[:1200], "retryable": bool(retryable), "details": dict(details or {}), "exit_code": exit_code,
                                          "recovery_class": recovery_class(code)}}
        return ToolResult(op.id, op.tool, False, exit_code, None, data, [], f"{SKILL_ID} [{code}] {message[:600]}", round(time.time() - t0, 3), attempt, dry_run)

    def _step(self, op: Operation, attempt: int, t0: float, argv: List[str], timeout: float) -> Tuple[Optional[Dict[str, Any]], Optional[ToolResult]]:
        code, out, err = self._call(argv, timeout + PROCESS_MARGIN)
        tail = "\n".join(err.strip().splitlines()[-12:])
        who = f"slide-sync {argv[0]}"
        if code == 124:
            return None, self._fail(op, attempt, False, t0, 124, "TIMEOUT", f"{who} exceeded {timeout + PROCESS_MARGIN:g}s", retryable=True)
        try:
            doc = one_json_document(out, who)
        except ToolError as e:
            return None, self._fail(op, attempt, False, t0, code if code else 9, "INVALID_OUTPUT", f"{e} (exit {code}) {tail[-300:]}".strip())
        if doc.get("ok") is not True:
            e_doc = as_dict(doc.get("error"))
            ecode = str(e_doc.get("code") or "ENGINE_FAILED")
            detail = e_doc.get("detail")
            details = {"detail": detail if isinstance(detail, (str, int, float, bool)) or detail is None else json.dumps(detail, default=str)[:300]}
            retry = self.retryable.get(ecode, False) if ecode in ERROR_CLASSES else False
            return None, self._fail(op, attempt, False, t0, code if code else 1, ecode, str(e_doc.get("message") or tail or "")[:1200], retryable=retry, details=details)
        if code != 0:
            return None, self._fail(op, attempt, False, t0, code, "INVALID_OUTPUT", f"{who}: exit code {code} with an ok result")
        return doc, None

    def _measure(self, op: Operation, timeout: Optional[float], dry_run: bool, attempt: int) -> ToolResult:
        t0 = time.time()
        if op.tool != TOOL_DETECT:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", f"{SKILL_ID} adapter does not support {op.tool}")
        try:
            b = self.build(op.args, timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e))
        commands = [" ".join(["slide-sync"] + argv) for argv in b["calls"]]
        if dry_run:
            return ToolResult(op.id, op.tool, True, 0, None, {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run", "commands": commands},
                              commands, "", round(time.time() - t0, 3), attempt, True)
        os.makedirs(b["out_dir"], exist_ok=True)
        for k, p in b["files"].items():   # an earlier run's files must never pass for this run's
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.remove(p)
        doc, failed = self._step(op, attempt, t0, b["calls"][0], b["timeout"])
        if failed is not None:
            return failed
        timeline = {k: v for k, v in doc.items() if k not in ("ok", "timeline_file")}
        why = check_timeline(timeline, self.version, b["input"], b["params"])
        if _read_json(b["files"]["timeline"]) != timeline:
            why.append("the timeline file on disk is not the document the Skill printed")
        if why:
            return self._fail(op, attempt, dry_run, t0, 9, "INVALID_OUTPUT", "slide-sync detect result not admitted: " + "; ".join(why))
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "ok", "timeline": timeline, "timeline_file": b["files"]["timeline"],
                                "warnings": [], "commands": commands}
        if b["deck"]:
            doc, failed = self._step(op, attempt, t0, b["calls"][1], 600.0)
            if failed is not None:
                return failed
            deck = {k: v for k, v in doc.items() if k not in ("ok", "output_dir")}
            doc, failed = self._step(op, attempt, t0, b["calls"][2], b["timeout"])
            if failed is not None:
                return failed
            match = {k: v for k, v in doc.items() if k not in ("ok", "match_file")}
            why = check_match(match, self.version, timeline, deck, sha256_file(b["deck"]))
            if _read_json(b["files"]["match"]) != match:
                why.append("the match file on disk is not the document the Skill printed")
            if _read_json(os.path.join(b["files"]["deck"], "deck.json")) != deck:
                why.append("the deck manifest on disk is not the document the Skill printed")
            if why:
                return self._fail(op, attempt, dry_run, t0, 9, "INVALID_OUTPUT", "slide-sync match result not admitted: " + "; ".join(why))
            data.update({"match": match, "deck": deck_summary(deck), "match_file": b["files"]["match"], "deck_dir": b["files"]["deck"]})
            data["warnings"] = [str(w)[:600] for w in match.get("warnings") or []]
        return ToolResult(op.id, op.tool, True, 0, None, data, commands, "", round(time.time() - t0, 3), attempt, False)


def _read_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def deck_summary(deck: Dict[str, Any]) -> Dict[str, Any]:
    """What the observation keeps of the imported deck: its format and sha256, the page count, each page's size and how many characters its
    own text layer has (the text itself stays in the deck directory the agent wrote), and the renderer."""
    src = as_dict(deck.get("source"))
    return {"format": src.get("format"), "sha256": src.get("sha256"), "converted_by": src.get("converted_by"), "page_count": deck.get("page_count"),
            "pages": [{k: as_dict(p).get(k) for k in ("index", "width_px", "height_px", "chars")} for p in deck.get("pages") or []],
            "renderer": {k: as_dict(deck.get("renderer")).get(k) for k in ("engine", "dpi", "skill_version")}}
