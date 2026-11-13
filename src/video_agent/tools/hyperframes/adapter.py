"""HyperframesAdapter: the agent's boundary to hyperframes-skill for one plan step, an animated layer with alpha for motion.layer (ADR-114;
Velisa DECISION_LOG D12 / O1: hyperframes-skill makes the material, motion-graphics-skill's `rendered_layer` places it).

    video-production-agent ─(plan step html_scene_layer)─→ HyperframesAdapter ─(argv lists, never a shell)─→
        [template <name> --values <job>/values.json [--font F] --output <job>/request.json --json --overwrite]   (a template and its values)
        scene <request> --output <job>/scene --json [--overwrite]                                               (or the user's own scene request)
        render <job>/scene --output <job>/<stem>_layer.mov --codec prores [--segments] [--workers N] [--ffmpeg-skill DIR] --json --overwrite --timeout T
          → a ProRes 4444 .mov with straight alpha (motion-graphics-skill's rendered_layer refuses a VP9 alpha WebM), and with --segments the
            IN / LOOP / OUT pieces, their frame counts and the loop seam measured

Protocol (hyperframes-skill's own contract, `node bin/hyperframes-skill.mjs contract --json`, pinned in skills/pins/hyperframes-skill_<version>.json,
the same pin ADR-080's handoff reads): every tool prints one JSON document (`status` completed | failed, `verified`, `verification[]`, `commands[]`;
failed: `error {kind, code, message, hint}` with exit 1 / 124 / 127). doctor (`doctor --json`) → `capabilities{}` and `tools.<tool>.usable` yes | no
| unknown with what is `missing`.

The agent sends only the codec it needs (prores), `--segments` when IN / LOOP / OUT was asked for, and the user's template / values / font /
scene request. Every value is checked before a process starts (the user's files inside the allowed roots, the output inside the workspace and
named `.mov`). The result is admitted only when each tool reports `completed` and `verified`, the render is the planned file, ProRes with
straight alpha, not FAILED by ffmpeg-skill's cross-check, and -- with segments -- the rendered segments[] are the planned IN / LOOP / OUT (the
durations the motion.layer element was planned with) and the loop seam measured clean. Failures keep the Skill's kind with the agent's recovery
class."""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...models import Operation, ToolResult
from ...skills.contract import SkillPackage, ToolSpec
from ..base import ToolAdapter, ToolError
from ..ffmpeg_skill.adapter import run_process_group
from ..skill_process import ContractError, as_dict, breaking_drift as _breaking_drift, drift_report, one_json_document, same_file, sha256_file
from .locate import PACKAGE_ID, HyperframesSkill, locate_hyperframes  # noqa: F401  (re-exported)

SKILL_ID = PACKAGE_ID
TOOL_RENDER = "hyperframes-skill/render"
# 0.8.0 (render --segments, --codec png, probe --alpha) and 0.8.1 (hyperframes-skill #22, docs only: the contract differs in `version`); never 0.9 / 1.x
SUPPORTED_SKILL_VERSIONS = ("0.8.",)
PIN_VERSION = "0.8.1"
PINNED_CONTRACT_PATH = Path(__file__).resolve().parents[2] / "skills" / "pins" / f"hyperframes-skill_{PIN_VERSION}.json"
CODEC = "prores"            # ProRes 4444 with alpha: what rendered_layer reads (a VP9 alpha WebM decodes without its alpha there)
SEGMENT_NAMES = ("in", "loop", "out")
FONT_EXTENSIONS = (".otf", ".ttf", ".woff2")
TEMPLATE_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
SENT = ("template", "values", "font", "request", "segments", "workers")
# the contract's error kinds (error.kinds) → the agent's recovery class
ERROR_CLASSES: Dict[str, str] = {"input": "INVALID_ARGS", "missing_tool": "TOOL_MISSING", "timeout": "TIMEOUT", "render": "SKILL_ERROR", "encode": "SKILL_ERROR",
                                 "verification": "SKILL_ERROR", "interrupted": "SKILL_ERROR", "internal": "SKILL_ERROR"}
AGENT_ERRORS: Dict[str, str] = {"INVALID_REQUEST": "INVALID_ARGS", "INVALID_OUTPUT": "SKILL_ERROR", "TIMEOUT": "TIMEOUT"}
DEFAULT_TIMEOUT = 1800.0    # render's own default (--timeout)
PROCESS_MARGIN = 120.0
SEGMENT_TOLERANCE = 1e-3    # seconds; segments are whole frames on both sides
DRIFT_KEYS = ("skill_id", "version", "contract_version", "provides", "error", "exit_codes", "execution")
DRIFT_TOOL_KEYS = ("input_schema", "output_schema", "capabilities")
FATAL_DRIFT_KEYS = ("skill_id", "contract_version", "error", "exit_codes")
FATAL_DRIFT_TOOL_KEYS = ("input_schema",)
USED_TOOLS = ("hyperframes-skill/template", "hyperframes-skill/scene", TOOL_RENDER, "hyperframes-skill/doctor")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def recovery_class(code: str) -> str:
    return ERROR_CLASSES.get(code) or AGENT_ERRORS.get(code) or "SKILL_ERROR"


def _tool(contract: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
    return next((t for t in contract.get("tools") or [] if isinstance(t, dict) and t.get("id") == tool_id), {})


def _props(contract: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
    return as_dict(as_dict(_tool(contract, tool_id).get("input_schema")).get("properties"))


def check_contract(contract: Any) -> List[str]:
    """What the adapter relies on, refused (never patched) when it does not hold: identity, the 0.8 line, the four tools with every argument the
    agent sends (render's codec enum with prores, --segments), and the failure kinds / exit codes."""
    errs: List[str] = []
    if not isinstance(contract, dict):
        return ["contract is not an object"]
    if contract.get("skill_id") != SKILL_ID:
        errs.append(f"skill_id {contract.get('skill_id')!r} != {SKILL_ID}")
    ver = str(contract.get("version") or "")
    if not ver.startswith(SUPPORTED_SKILL_VERSIONS):
        errs.append(f"skill version {ver!r} not in supported range {SUPPORTED_SKILL_VERSIONS}")
    if TOOL_RENDER not in [p.get("tool_id") for p in contract.get("provides") or [] if isinstance(p, dict)]:
        errs.append(f"provides[] has no {TOOL_RENDER}")
    need = {"hyperframes-skill/template": ("name", "values", "font", "output", "overwrite", "json"),
            "hyperframes-skill/scene": ("request", "output", "overwrite", "json"),
            TOOL_RENDER: ("scene_dir", "output", "codec", "segments", "workers", "ffmpegSkill", "overwrite", "timeout", "json"),
            "hyperframes-skill/doctor": ("json",)}
    for tid, keys in need.items():
        if not _tool(contract, tid):
            errs.append(f"contract lacks tool {tid}")
            continue
        missing = [k for k in keys if k not in _props(contract, tid)]
        if missing:
            errs.append(f"{tid} lacks {missing} in its input_schema")
    codecs = as_dict(_props(contract, TOOL_RENDER).get("codec")).get("enum") or []
    if CODEC not in codecs:
        errs.append(f"{TOOL_RENDER} codec enum {codecs} has no {CODEC}")
    kinds = as_dict(as_dict(contract.get("error")).get("kinds"))
    for k in ("input", "missing_tool", "timeout"):
        if k not in kinds:
            errs.append(f"error.kinds lacks {k}")
    codes = as_dict(contract.get("exit_codes"))
    for k in ("ok", "failure", "missing_tool", "timeout"):
        if not isinstance(codes.get(k), int):
            errs.append(f"exit_codes lacks {k}")
    return errs


def _used(c: Dict[str, Any]) -> Dict[str, Any]:
    return dict(c, tools=[t for t in c.get("tools") or [] if isinstance(t, dict) and t.get("id") in USED_TOOLS])


def drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(_used(live), _used(pinned or pinned_contract()), DRIFT_KEYS, "tools", "id", DRIFT_TOOL_KEYS)


def breaking_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return _breaking_drift(_used(live), _used(pinned or pinned_contract()), FATAL_DRIFT_KEYS, "tools", "id", FATAL_DRIFT_TOOL_KEYS)


PACKAGE = SkillPackage(
    skill_id=SKILL_ID, name="hyperframes-skill", version="",
    description="HTML scenes rendered to verified video with HyperFrames (headless Chromium + ffmpeg): templates and scene requests, ProRes 4444 with alpha, "
                "IN / LOOP / OUT segments (EXPERIMENTAL)",
    capabilities=["hyperframes-skill"],
    tools=[ToolSpec(TOOL_RENDER, SKILL_ID, description="Fill a template (or take a scene request), write the scene and render it to ProRes 4444 with alpha, verified",
                    required_capabilities=["hyperframes-skill:RENDER"], inputs=[], produces_output=True, deterministic=False,
                    result_keys=["layer", "expected", "alpha", "segments", "loop_seam", "ffmpeg_skill", "hyperframes"])],
    repository="kajisho5/hyperframes-skill", role="animated overlays (lower thirds, titles) as alpha layers for motion-graphics-skill's rendered_layer")


def planned_segments(segments: Any) -> Optional[Dict[str, float]]:
    """{in, loop, out} seconds or None; ToolError when the shape is wrong."""
    if segments in (None, False):
        return None
    if not isinstance(segments, dict) or set(segments) != set(SEGMENT_NAMES):
        raise ToolError(f"segments must be {{in, loop, out}} seconds, got {segments!r}")
    out: Dict[str, float] = {}
    for k in SEGMENT_NAMES:
        v = segments[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ToolError(f"segments.{k} must be a number of seconds >= 0, got {v!r}")
        out[k] = float(v)
    if out["loop"] <= 0:
        raise ToolError("segments.loop must be > 0")
    return out


class HyperframesAdapter(ToolAdapter):
    def __init__(self, skill: HyperframesSkill, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None, path_policy: Any = None,
                 ffmpeg_skill_dir: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT):
        self.skill = skill
        self.name = SKILL_ID
        if not skill.node or not (os.path.isfile(skill.node) or shutil.which(skill.node)):
            raise ContractError("Node.js is not on PATH (hyperframes-skill runs under Node 22+; set VIDEO_AGENT_NODE or install Node)")
        self.workspace = os.path.realpath(os.path.abspath(workspace)) if workspace else None
        self.allowed_inputs = [os.path.realpath(os.path.abspath(p)) for p in (allowed_inputs or [])]
        self.policy = path_policy
        self.ffmpeg_skill_dir = ffmpeg_skill_dir
        self.default_timeout = float(timeout)
        self.contract = self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("incompatible hyperframes-skill contract: " + "; ".join(errs))
        self.version = str(self.contract.get("version"))
        self.tools = {TOOL_RENDER: True}
        self._doctor: Optional[Dict[str, Any]] = None

    def _call(self, tool: str, argv: List[str], timeout: Optional[float]) -> Tuple[int, str, str]:
        return run_process_group(self.skill.command(tool) + list(argv), timeout=timeout)

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._call("contract", ["--json"], 120.0)
        if code != 0:
            raise ContractError(f"hyperframes-skill contract --json failed (exit {code}): {err.strip()[-300:]} (are its npm dependencies installed? `npm ci` in the checkout)")
        return one_json_document(out, "hyperframes-skill contract")

    def doctor(self) -> Dict[str, Any]:
        """`doctor --json`, normalised: ok when template, scene and render are usable; degraded when template / scene are but render is not (a
        headless Chromium or ffmpeg missing: the scene can be written, not rendered); fail otherwise. `problems` names each missing capability."""
        if self._doctor is not None:
            return self._doctor
        code, out, err = self._call("doctor", ["--json"], 300.0)
        try:
            doc = one_json_document(out, "hyperframes-skill doctor")
        except ToolError as e:
            doc = {"tools": {}, "capabilities": {}, "problems": [f"doctor produced no document (exit {code}): {e} {err.strip()[-200:]}"]}
        tools = as_dict(doc.get("tools"))
        usable = {t: as_dict(tools.get(t)).get("usable") == "yes" for t in ("template", "scene", "render")}
        caps = as_dict(doc.get("capabilities"))
        missing = sorted({str(m) for t in ("template", "scene", "render") for m in as_dict(tools.get(t)).get("missing") or []})
        unknown = sorted({str(m) for t in ("template", "scene", "render") for m in as_dict(tools.get(t)).get("unknown") or []})
        problems = [str(p) for p in doc.get("problems") or []] + [f"{m}: {as_dict(caps.get(m)).get('status', 'missing')} ({as_dict(caps.get(m)).get('detail') or 'not found'})"
                                                                  for m in missing + unknown]
        status = "ok" if all(usable.values()) else ("degraded" if usable["template"] and usable["scene"] else "fail")
        self._doctor = dict(doc, status=status, render_missing=missing + unknown, problems=problems, skill_status=doc.get("status"))
        return self._doctor

    def operation_status(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {"RENDER": "supported" if (doc or self.doctor()).get("status") == "ok" else "unavailable"}

    def operation_evidence(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        doc = doc or self.doctor()
        ev: Dict[str, Any] = {"ready": doc.get("status") == "ok"}
        if doc.get("status") != "ok":
            miss = list(doc.get("render_missing") or [])
            ev.update(code="missing_tool", reason=("render cannot run here: " + ", ".join(miss) + " missing" if miss else "render is not usable here (doctor)"),
                      detail="; ".join(doc.get("problems") or [])[:400],
                      install=("`npx hyperframes browser ensure` (chrome-headless-shell) or HYPERFRAMES_BROWSER_PATH" if "chromium:headless" in miss else
                               "Node.js 22+, `npm ci` in the hyperframes-skill checkout, ffmpeg 5+ (see `hyperframes-skill doctor`)"))
        return {"RENDER": ev}

    def drift(self) -> List[str]:
        return drift(self.contract)

    def breaking_drift(self) -> List[str]:
        return breaking_drift(self.contract)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "skill": self.skill.describe(), "tools": [TOOL_RENDER], "drift": self.drift()}

    def package(self) -> SkillPackage:
        return SkillPackage(PACKAGE.skill_id, PACKAGE.name, self.version, PACKAGE.description, list(PACKAGE.capabilities),
                            [ToolSpec(**{**t.__dict__, "version": self.version}) for t in PACKAGE.tools], PACKAGE.repository, PACKAGE.role)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        return self._fail(Operation(tool=tool, args=args, inputs=[], outputs=[], kind="measure"), 1, False, time.time(), 2, "INVALID_REQUEST",
                          "hyperframes-skill is a plan step (it renders), never a measurement")

    def _within(self, root: str, p: str) -> bool:
        return os.path.normcase(p) == os.path.normcase(root) or os.path.normcase(p).startswith(os.path.normcase(root).rstrip(os.sep) + os.sep)

    def _user_file(self, raw: Any, what: str, paths: Dict[str, str], exts: Tuple[str, ...]) -> str:
        p = paths.get(raw, raw) if isinstance(raw, str) else raw
        if not isinstance(p, str) or not p.strip() or "\x00" in p or "\n" in p:
            raise ToolError(f"{what} must be a non-empty single-line path")
        resolved = os.path.realpath(os.path.abspath(p))
        roots = self.allowed_inputs + ([self.workspace] if self.workspace else [])
        if roots and not any(self._within(r, resolved) for r in roots):
            raise ToolError(f"{what} is outside the allowed input roots: {os.path.basename(p)}")
        if not os.path.isfile(resolved):
            raise ToolError(f"{what} not found: {os.path.basename(p)}")
        if Path(resolved).suffix.lower() not in exts:
            raise ToolError(f"{what} must be one of {', '.join(exts)}: {os.path.basename(p)}")
        return resolved

    def build(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None) -> Dict[str, Any]:
        """The checked plan of the call: which of template / scene / render run with which argv. Nothing runs here."""
        args = dict(op.args)
        if op.tool != TOOL_RENDER:
            raise ToolError(f"{SKILL_ID} adapter does not support {op.tool}")
        unknown = sorted(k for k in args if k not in SENT + ("output",))
        if unknown:
            raise ToolError(f"unknown argument(s) {unknown} (the agent sends only {', '.join(SENT)}; the codec is always {CODEC})")
        template, request = args.get("template"), args.get("request")
        if bool(template) == bool(request):
            raise ToolError("give exactly one of template (with values) or request (a scene request JSON)")
        segments = planned_segments(args.get("segments"))
        out = paths.get(args.get("output"), args.get("output")) if isinstance(args.get("output"), str) else None
        if not out:
            raise ToolError("output missing")
        out = os.path.realpath(os.path.abspath(out))
        if not self.workspace or not self._within(self.workspace, out):
            raise ToolError(f"output is outside the workspace: {os.path.basename(out)}")
        if Path(out).suffix.lower() != ".mov":
            raise ToolError(f"output must be a .mov (ProRes 4444 with alpha): {os.path.basename(out)}")
        work = os.path.dirname(out)
        stem = Path(out).stem
        t = max(1.0, min(86400.0, float(timeout or self.default_timeout)))
        steps: List[Tuple[str, List[str]]] = []
        files: Dict[str, Any] = {}
        if template:
            if not isinstance(template, str) or not TEMPLATE_RE.match(template):
                raise ToolError(f"template must be a template name (lower case, digits, '-'), got {template!r}")
            values = args.get("values") or {}
            if not isinstance(values, dict) or any(not isinstance(k, str) or isinstance(v, (dict, list)) for k, v in values.items()):
                raise ToolError("values must be an object of plain values (text, number, true / false)")
            req_path = os.path.join(work, f"{stem}.request.json")
            files["values"] = (os.path.join(work, f"{stem}.values.json"), values)
            argv = [template, f"--values={files['values'][0]}", f"--output={req_path}", "--json", "--overwrite"]
            if args.get("font") is not None:
                argv.append(f"--font={self._user_file(args['font'], 'font', paths, FONT_EXTENSIONS)}")
            steps.append(("template", argv))
        else:
            if args.get("values") or args.get("font") is not None:
                raise ToolError("values / font fill a template; a scene request carries its own")
            req_path = self._user_file(request, "request", paths, (".json",))
        scene_dir = os.path.join(work, f"{stem}.scene")
        steps.append(("scene", [req_path, f"--output={scene_dir}", "--json"]))
        rargv = [scene_dir, f"--output={out}", f"--codec={CODEC}"]
        if segments:
            rargv.append("--segments")
        w = args.get("workers")
        if w is not None:
            if w != "auto" and (isinstance(w, bool) or not isinstance(w, int) or not 1 <= w <= 8):
                raise ToolError(f"workers must be auto or 1-8, got {w!r}")
            rargv.append(f"--workers={w}")
        if self.ffmpeg_skill_dir:
            rargv.append(f"--ffmpeg-skill={self.ffmpeg_skill_dir}")
        rargv += ["--json", "--overwrite", f"--timeout={t:g}"]
        steps.append(("render", rargv))
        if self.policy is not None:
            self.policy.check_output(out, [req_path] if request else [])
        return {"steps": steps, "output": out, "timeout": t, "segments": segments, "segments_inject": bool(template and segments), "request": req_path,
                "scene": scene_dir, "files": files, "template": template, "values": args.get("values") if template else None}

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            return [" ".join(["hyperframes-skill", name] + argv) for name, argv in self.build(op, paths)["steps"]]
        except ToolError as e:
            return [f"<refused: {e}>"]

    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, exit_code: int, code: str, message: str, retryable: bool = False,
              commands: Optional[List[str]] = None) -> ToolResult:
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "failed",
                                "error": {"code": code, "message": message[:1200], "retryable": bool(retryable), "exit_code": exit_code, "recovery_class": recovery_class(code)}}
        return ToolResult(op.id, op.tool, False, exit_code, None, data, list(commands or []), f"{SKILL_ID} [{code}] {message[:600]}", round(time.time() - t0, 3), attempt, dry_run)

    def _segments_problems(self, planned: Optional[Dict[str, float]], doc: Dict[str, Any]) -> List[str]:
        """The rendered segments[] against the planned IN / LOOP / OUT (the durations the motion.layer element carries)."""
        if planned is None:
            return []
        why: List[str] = []
        got = {str(as_dict(s).get("name")): as_dict(s) for s in doc.get("segments") or []}
        for name in SEGMENT_NAMES:
            s = got.get(name)
            if planned[name] == 0 and (s is None or float(s.get("duration") or 0) == 0):
                continue
            if s is None:
                why.append(f"segment {name} was not rendered")
                continue
            if abs(float(s.get("duration") or 0) - planned[name]) > SEGMENT_TOLERANCE:
                why.append(f"segment {name} is {s.get('duration')}s, planned {planned[name]:g}s")
            if s.get("output") and s.get("verified") is not True:
                why.append(f"segment {name} is not verified")
        seam = as_dict(doc.get("loop_seam"))
        if seam.get("ok") is not True:
            why.append(f"the loop seam is not clean (loop_seam {json.dumps(seam, sort_keys=True)[:160]})")
        return why

    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        try:
            b = self.build(op, paths, timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e))
        commands = [" ".join(["hyperframes-skill", name] + argv) for name, argv in b["steps"]]
        if dry_run:
            return ToolResult(op.id, op.tool, True, 0, None, {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run", "commands": commands},
                              commands, "", round(time.time() - t0, 3), attempt, True)
        Path(b["output"]).parent.mkdir(parents=True, exist_ok=True)
        if b["files"].get("values"):
            vpath, values = b["files"]["values"]
            Path(vpath).write_text(json.dumps(values, ensure_ascii=False, sort_keys=True), encoding="utf-8")   # the agent's own intermediate, never the user's file
        if os.path.isdir(b["scene"]):
            shutil.rmtree(b["scene"])   # the scene of an earlier attempt of this step (its own directory in the job)
        docs: Dict[str, Dict[str, Any]] = {}
        tail = ""
        for name, argv in b["steps"]:
            if name == "scene" and b["segments_inject"]:
                try:   # the template's request with the segments the user asked for (the request file the template just wrote, in the job)
                    req = json.loads(Path(b["request"]).read_text(encoding="utf-8"))
                    req["segments"] = dict(b["segments"])
                    Path(b["request"]).write_text(json.dumps(req, ensure_ascii=False, sort_keys=True), encoding="utf-8")
                except (OSError, ValueError) as e:
                    return self._fail(op, attempt, dry_run, t0, 9, "INVALID_OUTPUT", f"the template's request could not be read back: {e}", commands=commands)
            budget = b["timeout"] + PROCESS_MARGIN if name == "render" else 300.0
            code, out, err = self._call(name, argv, budget * (4 if name == "render" and b["segments"] else 1))
            tail = "\n".join(err.strip().splitlines()[-12:])
            if code == 124 and not out.strip():
                return self._fail(op, attempt, dry_run, t0, 124, "TIMEOUT", f"hyperframes-skill {name} exceeded its time limit", retryable=True, commands=commands)
            try:
                doc = one_json_document(out, f"hyperframes-skill {name}")
            except ToolError as e:
                return self._fail(op, attempt, dry_run, t0, code if code else 9, "INVALID_OUTPUT", f"{e} (exit {code}) {tail[-300:]}".strip(), commands=commands)
            if doc.get("status") != "completed":
                e = as_dict(doc.get("error"))
                kind = str(e.get("kind") or "internal")
                msg = f"{name}: {e.get('message') or tail or 'failed'}" + (f" (hint: {e['hint']})" if e.get("hint") else "")
                return self._fail(op, attempt, dry_run, t0, code if code else 1, kind if kind in ERROR_CLASSES else "internal", msg, retryable=kind == "timeout", commands=commands)
            if doc.get("verified") is not True or code != 0:
                bad = [v for v in doc.get("verification") or [] if isinstance(v, dict) and v.get("ok") is False]
                return self._fail(op, attempt, dry_run, t0, code if code else 9, "INVALID_OUTPUT",
                                  f"{name} result not admitted: not verified ({json.dumps(bad, sort_keys=True)[:300]})", commands=commands)
            docs[name] = doc
        r = docs["render"]
        why: List[str] = []
        if not same_file(str(r.get("output") or ""), b["output"]):
            why.append(f"render {os.path.basename(str(r.get('output') or ''))!r} is not the planned {os.path.basename(b['output'])}")
        if not os.path.isfile(b["output"]) or os.path.getsize(b["output"]) == 0:
            why.append("the render is missing or empty although the Skill reported success")
        if as_dict(r.get("expected")).get("codec") != CODEC:
            why.append(f"codec {as_dict(r.get('expected')).get('codec')!r} is not {CODEC}")
        alpha = as_dict(r.get("alpha"))
        if alpha.get("mode") != "straight" or alpha.get("channel") is False:
            why.append(f"the render has no alpha plane (alpha {json.dumps(alpha, sort_keys=True)[:120]}); rendered_layer needs one")
        fs = as_dict(r.get("ffmpeg_skill"))
        if fs.get("status") == "FAILED":
            why.append(f"ffmpeg-skill's cross-check FAILED: {fs.get('reason') or ''}")
        why += self._segments_problems(b["segments"], r)
        if why:
            return self._fail(op, attempt, dry_run, t0, 9, "INVALID_OUTPUT", "hyperframes-skill result not admitted: " + "; ".join(why), commands=commands)
        seg = b["segments"]
        hf = as_dict(r.get("hyperframes"))
        nested = [c for d in docs.values() for c in d.get("commands") or [] if isinstance(c, dict)]
        data = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "ok", "codec": CODEC, "layer": b["output"], "sha256": sha256_file(b["output"]),
                "template": b["template"], "values": b["values"], "request": None if b["template"] else {"path": b["request"], "sha256": sha256_file(b["request"])},
                "expected": r.get("expected"), "alpha": {k: alpha.get(k) for k in ("mode", "channel")},
                "hyperframes": {"version": hf.get("version"), "chromium": as_dict(hf.get("chromium")).get("source")},
                "ffmpeg_skill": {k: fs.get(k) for k in ("status", "reason") if fs.get(k) is not None},
                "segments": [{k: as_dict(s).get(k) for k in ("name", "start", "duration", "frames", "verified")} for s in r.get("segments") or []] if seg else None,
                "loop_seam": {k: as_dict(r.get("loop_seam")).get(k) for k in ("measured", "frames", "wrap_diff", "max_step_diff", "tolerance", "ok")} if seg else None,
                # what the motion.layer element was planned with, read from these segments (rendered_layer's in_duration / out_duration / loop)
                "layer_timing": ({"in_duration": seg["in"], "out_duration": seg["out"], "loop": True} if seg else {"in_duration": 0.0, "out_duration": 0.0, "loop": False}),
                "commands": commands, "tool_commands": [{"program": c.get("program"), "argv": [str(a) for a in c.get("argv") or []][:40], "exit_code": c.get("exit_code")}
                                                       for c in nested][:60]}
        return ToolResult(op.id, op.tool, True, 0, b["output"], data, commands, tail, round(time.time() - t0, 3), attempt, False)
