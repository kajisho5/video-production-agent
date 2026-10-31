"""ImageMagickAdapter: the agent's boundary to imagemagick-skill (ADR-077; imagemagick-skill roadmap step 4, agent side).

    video-production-agent ─(typed Operation)─→ ImageMagickAdapter ─(argv list, never a shell)─→
        `python <checkout>/scripts/resize.py --output=<out.png> --width=W --height=H --mode=fit|fill|exact --overwrite --json --timeout=T -- <image>`
        `python <checkout>/scripts/pad.py --output=<out.png> --width=W --height=H --color=C --gravity=G --overwrite --json --timeout=T -- <image>`
        `python <checkout>/scripts/strip.py --output=<out.png> --overwrite --json --timeout=T -- <image>`
          ─(ImageMagick 7 `magick` / ImageMagick 6 `convert`, or macOS `sips` for resize / pad)─→ a new image file

Protocol (imagemagick-skill's own contract, `python scripts/_contract.py contract --json`, pinned in contract_0.8.1.json = the v0.8.1
release; main differs from the tag only in CI files):
    contract → `_contract.py contract --json` (contract_version 1.0; each tool's input_schema is derived from its own argparse parser)
    doctor   → `_contract.py doctor --json` (backends found, `tools.<name>.usable` per tool; never downloads anything)
    a tool   → exactly one JSON object: `{"ok": true, ...output_schema}`, or `{"ok": false, "reason": ...}` (exit 1), or the timeout shape
               `{"ok": false, "code": "timeout", "timeout": N}` (exit 124, ImageMagick's whole process group killed, a half-written output removed)

Every value is checked before the process starts: the input is an artifact id resolved through the paths map to an existing PNG / JPEG
inside the allowed roots, the output lands in the workspace, never on the input, and has an image extension; every parameter is one the
live contract declares for that tool (by the flag spelling its `cli` lists), with its type and enum; unknown keys and command-like keys
are refused by name. Every option is one `--flag=value` word and the input comes after `--` (ADR-069). The result is admitted only when
it is about this request (the input and output it names, the size it promises: exact for fill / exact / pad, inside the box and touching
it for fit; no GPS after strip), the output is a non-empty file and the input's bytes are unchanged; the agent hashes the output itself
(the Skill reports no hash)."""
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
from ..ffmpeg_skill.adapter import PathPolicy, run_process_group
from ..skill_process import FORBIDDEN_ARG_KEYS, ContractError, as_dict, breaking_drift as _breaking_drift, drift_report, one_json_document, same_file, scan_forbidden, sha256_file
from .locate import PACKAGE_ID, ImageMagickSkill, locate_imagemagick  # noqa: F401  (re-exported)

SKILL_ID = PACKAGE_ID
PREFIX = SKILL_ID + "/"
TOOL_RESIZE = "imagemagick-skill/resize"
TOOL_PAD = "imagemagick-skill/pad"
TOOL_STRIP = "imagemagick-skill/strip"
TOOLS = (TOOL_RESIZE, TOOL_PAD, TOOL_STRIP)
SCRIPT = {TOOL_RESIZE: "resize", TOOL_PAD: "pad", TOOL_STRIP: "strip"}
# the per-tool capability `imagemagick-skill:<TYPE>` (resolver) ← the doctor's `tools.<name>.usable`, and the contract capability id (provides[])
OPERATION_TYPES = {"resize": "RESIZE", "pad": "PAD", "strip": "STRIP"}
CAPABILITY_IDS = {"RESIZE": "image.resize", "PAD": "image.pad", "STRIP": "image.strip"}
TOOL_OF_TYPE = {"RESIZE": TOOL_RESIZE, "PAD": TOOL_PAD, "STRIP": TOOL_STRIP}
SUPPORTED_SKILL_VERSIONS = ("0.8.",)
# the parameters the adapter sends per tool (input_schema property names), besides the references and the boundary flags below. Their type /
# enum / flag spelling are read from the live contract on every call; the pinned contract fixes what the agent's vocabulary was written for.
PARAMETERS: Dict[str, Tuple[str, ...]] = {TOOL_RESIZE: ("width", "height", "mode"), TOOL_PAD: ("width", "height", "color", "gravity"), TOOL_STRIP: ()}
REQUIRED: Dict[str, Tuple[str, ...]] = {TOOL_RESIZE: ("width", "height"), TOOL_PAD: ("width", "height", "color"), TOOL_STRIP: ()}
BOUNDARY_FLAGS = ("input", "output", "overwrite", "json", "timeout")   # always sent (input positional after `--`); dry_run never
# output_schema keys the adapter reads to admit a result
RESULT_KEYS: Dict[str, Tuple[str, ...]] = {TOOL_RESIZE: ("input", "output", "mode", "requested", "actual", "backend"),
                                           TOOL_PAD: ("input", "output", "mode", "color", "actual", "backend"),
                                           TOOL_STRIP: ("input", "output", "has_gps", "backend")}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
MAX_SIDE = 16384                  # the agent's own bound on a pixel size it sends (the contract states none)
COLOR_RE = re.compile(r"^(none|[a-z]{1,32}|#[0-9A-Fa-f]{6}|#[0-9A-Fa-f]{8})$")   # the subset of pad's colour syntax the agent sends (no rgb() / rgba())
DEFAULT_TIMEOUT = 300.0           # the Skill's own --timeout default; the process gets a margin on top
PROCESS_MARGIN = 30.0
# the agent's codes (imagemagick-skill reports a reason, and `code: "timeout"` only for its timeout) → recovery class (execution/recovery.py)
ERROR_CLASSES: Dict[str, str] = {"INVALID_REQUEST": "INVALID_ARGS", "SKILL_FAILED": "SKILL_ERROR", "INVALID_OUTPUT": "SKILL_ERROR", "INPUT_CHANGED": "SKILL_ERROR",
                                 "TIMEOUT": "TIMEOUT"}
# contract drift (ADR-045 / ADR-047): every key the adapter reads is compared with the pinned contract; the fatal subset gates availability
DRIFT_KEYS = ("name", "version", "contract_version", "skill", "requirements", "execution", "result_shapes", "provides", "rules")
DRIFT_TOOL_KEYS = ("id", "script", "description", "role", "backends", "requires_magick", "writes_output", "supports_json", "supports_dry_run", "output_optional", "mutates_input",
                   "verify", "input_schema", "output_schema", "required_args", "optional_args")
# version (within SUPPORTED_SKILL_VERSIONS), provides' lifecycle, rules, requirements, backends, descriptions and parameters the adapter does not
# send: reported only. What is left changes the wire or what the adapter admits: identity, the execution guarantees, the result shapes, the
# three tools' script / output / input-mutation flags, and (parameter_drift) every parameter the adapter sends and every result key it reads.
FATAL_DRIFT_KEYS = ("contract_version", "execution", "result_shapes")
FATAL_DRIFT_TOOL_KEYS = ("id", "script", "writes_output", "supports_json", "mutates_input")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.8.1.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def recovery_class(code: str) -> str:
    return ERROR_CLASSES.get(code, "SKILL_ERROR")


def _tools(contract: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """tool id → contract tool entry (the contract names each tool `imagemagick-skill/<name>` in `id`)."""
    return {str(t.get("id")): t for t in contract.get("tools") or [] if isinstance(t, dict)}


def tool_schema(contract: Dict[str, Any], tool: str) -> Dict[str, Any]:
    return as_dict(as_dict(_tools(contract).get(tool)).get("input_schema"))


def _flag(prop: Dict[str, Any], name: str) -> Optional[str]:
    """The long flag a property is spelled with (`--width`), or None when the contract does not list it."""
    cli = prop.get("cli")
    want = "--" + name.replace("_", "-")
    return want if isinstance(cli, list) and want in cli else None


def check_contract(contract: Any) -> List[str]:
    """What the adapter relies on, refused (never patched) when it does not hold: identity, contract and Skill version, a shell-free offline
    execution that never writes its input, the three tools as JSON-speaking scripts that write a new file, every parameter and boundary
    flag the adapter sends (with the spelling it sends), every result key it reads, the three capabilities, and the timeout result shape."""
    errs: List[str] = []
    if not isinstance(contract, dict):
        return ["contract is not an object"]
    skill = as_dict(contract.get("skill"))
    if skill.get("id") != SKILL_ID:
        errs.append(f"skill.id {skill.get('id')!r} != {SKILL_ID}")
    if not str(contract.get("contract_version") or "").startswith("1."):
        errs.append(f"contract_version {contract.get('contract_version')!r} is not 1.x")
    ver = str(skill.get("version") or contract.get("version") or "")
    if not ver.startswith(SUPPORTED_SKILL_VERSIONS):
        errs.append(f"skill version {ver!r} not in supported range {SUPPORTED_SKILL_VERSIONS}")
    ex = as_dict(contract.get("execution"))
    if ex.get("shell") is not False or ex.get("network") is not False or ex.get("input_mutation") is not False or ex.get("arbitrary_executables") is not False:
        errs.append("execution must declare shell, network, input_mutation and arbitrary_executables false")
    tools = _tools(contract)
    for tid in TOOLS:
        t = tools.get(tid)
        if t is None:
            errs.append(f"contract lacks tool {tid}")
            continue
        if t.get("script") != f"{SCRIPT[tid]}.py" or t.get("writes_output") is not True or t.get("mutates_input") is not False or t.get("supports_json") is not True:
            errs.append(f"tool {tid} must be script {SCRIPT[tid]}.py with writes_output true, mutates_input false, supports_json true")
        props = as_dict(as_dict(t.get("input_schema")).get("properties"))
        for name in PARAMETERS[tid] + BOUNDARY_FLAGS:
            p = as_dict(props.get(name))
            if not p:
                errs.append(f"tool {tid} lacks parameter {name}")
            elif name == "input":
                if p.get("cli") != "positional":
                    errs.append(f"tool {tid} input is not positional")
            elif _flag(p, name) is None:
                errs.append(f"tool {tid} parameter {name} is not spelled --{name.replace('_', '-')}")
        required = set(as_dict(t.get("input_schema")).get("required") or [])
        unsent = sorted(required - set(PARAMETERS[tid] + BOUNDARY_FLAGS))
        if unsent:
            errs.append(f"tool {tid} requires {unsent}, which the agent does not send")
        out_keys = set(as_dict(as_dict(t.get("output_schema")).get("properties")))
        missing = [k for k in RESULT_KEYS[tid] if k not in out_keys]
        if missing:
            errs.append(f"tool {tid} output lacks {missing}")
    provided = {(str(p.get("id")), str(p.get("tool_id"))) for p in contract.get("provides") or [] if isinstance(p, dict)}
    for typ, cap in CAPABILITY_IDS.items():
        if (cap, TOOL_OF_TYPE[typ]) not in provided:
            errs.append(f"provides lacks {cap} -> {TOOL_OF_TYPE[typ]}")
    timeout = as_dict(as_dict(contract.get("result_shapes")).get("timeout"))
    if as_dict(timeout.get("code")).get("enum") != ["timeout"]:
        errs.append("result_shapes.timeout must declare code \"timeout\"")
    return errs


def lifecycle(contract: Dict[str, Any]) -> Dict[str, str]:
    """capability id → lifecycle (EXPERIMENTAL in 0.8.1) from the contract's `provides`."""
    return {str(p.get("id")): str(p.get("lifecycle")) for p in contract.get("provides") or [] if isinstance(p, dict)}


def _op_tools(c: Dict[str, Any]) -> Dict[str, Any]:
    """The contract with only the tools this adapter runs (the other 16 tools are not called)."""
    return dict(c, tools=[t for t in c.get("tools") or [] if isinstance(t, dict) and t.get("id") in TOOLS])


def drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    """Every difference on a field the adapter reads (top level, and the three tools it runs), for operators."""
    return drift_report(_op_tools(live), _op_tools(pinned or pinned_contract()), DRIFT_KEYS, "tools", "id", DRIFT_TOOL_KEYS)


def parameter_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    """Fatal: a parameter the adapter sends whose type / flag spelling / enum changed or vanished, a newly required parameter it does not
    send, or a result key it reads that vanished. A changed description or a new optional parameter is not (reported by drift())."""
    pinned = pinned or pinned_contract()
    out: List[str] = []
    for tid in TOOLS:
        lp = as_dict(tool_schema(live, tid).get("properties"))
        pp = as_dict(tool_schema(pinned, tid).get("properties"))
        for name in PARAMETERS[tid] + BOUNDARY_FLAGS:
            a = {k: as_dict(lp.get(name)).get(k) for k in ("type", "cli", "enum")} if name in lp else None
            b = {k: as_dict(pp.get(name)).get(k) for k in ("type", "cli", "enum")} if name in pp else None
            if a != b:
                out.append(f"tools {tid}.input_schema.{name}: pinned {json.dumps(b, sort_keys=True)} != live {json.dumps(a, sort_keys=True)}")
        req = set(tool_schema(live, tid).get("required") or []) - set(PARAMETERS[tid] + BOUNDARY_FLAGS)
        if req:
            out.append(f"tools {tid}: live requires {sorted(req)}, which the agent does not send")
        lo = as_dict(as_dict(as_dict(_tools(live).get(tid)).get("output_schema")).get("properties"))
        gone = [k for k in RESULT_KEYS[tid] if k not in lo]
        if gone:
            out.append(f"tools {tid}.output_schema: live lacks {gone}")
    return out


def breaking_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    pinned = pinned or pinned_contract()
    out = _breaking_drift(_op_tools(live), _op_tools(pinned), FATAL_DRIFT_KEYS, "tools", "id", FATAL_DRIFT_TOOL_KEYS) + parameter_drift(live, pinned)
    if as_dict(live.get("skill")).get("id") != as_dict(pinned.get("skill")).get("id"):
        out.append(f"skill.id: pinned {as_dict(pinned.get('skill')).get('id')!r} != live {as_dict(live.get('skill')).get('id')!r}")
    return out


PACKAGE = SkillPackage(
    skill_id=SKILL_ID, name="imagemagick-skill", version="",
    description="Local, deterministic still-image editing through ImageMagick (macOS sips for a subset): resize (fit / fill / exact), pad to a canvas "
                "with a given colour, strip GPS / EXIF; the input is never overwritten, every output is read back",
    capabilities=["imagemagick-skill"],
    tools=[ToolSpec(TOOL_RESIZE, SKILL_ID, description="Resize an image to a caller-given box: fit (inside, no distortion), fill (cover and centre-crop) or exact",
                    inputs=["input"], produces_output=True, deterministic=True, result_keys=list(RESULT_KEYS[TOOL_RESIZE])),
           ToolSpec(TOOL_PAD, SKILL_ID, description="Pad an image to an exact canvas with a caller-given colour (or transparency), never scaling or cropping it",
                    inputs=["input"], produces_output=True, deterministic=True, result_keys=list(RESULT_KEYS[TOOL_PAD])),
           ToolSpec(TOOL_STRIP, SKILL_ID, description="Remove GPS / EXIF metadata, baking the EXIF orientation into the pixels first (ImageMagick only)",
                    inputs=["input"], produces_output=True, deterministic=True, result_keys=list(RESULT_KEYS[TOOL_STRIP]))],
    repository="kajisho5/imagemagick-skill", role="still-image preparation (resize / pad / strip of overlay assets)")


def _int_value(v: Any, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or float(v) != int(v):
        raise ToolError(f"{name} must be an integer, got {v!r}")
    if not 1 <= int(v) <= MAX_SIDE:
        raise ToolError(f"{name} {v!r} must be within 1..{MAX_SIDE}")
    return int(v)


class ImageMagickAdapter(ToolAdapter):
    owns_cache = False

    def __init__(self, skill: ImageMagickSkill, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None,
                 path_policy: Optional[PathPolicy] = None, timeout: float = DEFAULT_TIMEOUT):
        self.skill = skill
        self.name = SKILL_ID
        self.workspace = os.path.realpath(os.path.abspath(workspace)) if workspace else None
        self.allowed_inputs = [os.path.realpath(os.path.abspath(p)) for p in (allowed_inputs or [])]
        self.policy = path_policy
        self.default_timeout = float(timeout)
        self.contract = self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("incompatible imagemagick-skill contract: " + "; ".join(errs))
        self.version = str(as_dict(self.contract.get("skill")).get("version") or self.contract.get("version"))
        self.tools = {t: True for t in TOOLS}
        self.lifecycle = lifecycle(self.contract)
        self._doctor: Optional[Dict[str, Any]] = None

    # ---- contract / doctor
    def _call(self, script: str, argv: List[str], timeout: Optional[float]) -> Tuple[int, str, str]:
        return run_process_group(self.skill.script(script) + list(argv), timeout=timeout)

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._call("_contract", ["contract", "--json"], 60.0)
        if code != 0:
            raise ContractError(f"imagemagick-skill contract --json failed (exit {code}): {err.strip()[-300:]}")
        return one_json_document(out, "imagemagick-skill contract")

    def doctor(self) -> Dict[str, Any]:
        """`_contract.py doctor --json` with a `status` the resolver reads: ok (resize, pad and strip all usable here), degraded (some of them:
        sips covers resize / pad but never strip), fail (none, or no document)."""
        if self._doctor is not None:
            return self._doctor
        code, out, err = self._call("_contract", ["doctor", "--json"], 120.0)
        try:
            doc = one_json_document(out, "imagemagick-skill doctor")
        except ToolError as e:
            doc = {"ok": False, "tools": {}, "problems": [f"doctor produced no document (exit {code}): {e} {err.strip()[-200:]}"]}
        usable = self.operation_status(doc)
        n = sum(1 for v in usable.values() if v == "supported")
        status = "ok" if n == len(usable) else ("degraded" if n else "fail")
        problems = list(doc.get("problems") or []) + [f"{TOOL_OF_TYPE[t]} is not usable here (doctor tools.{SCRIPT[TOOL_OF_TYPE[t]]}.usable)" for t, v in usable.items() if v != "supported"]
        self._doctor = dict(doc, status=status, problems=problems, warnings=list(doc.get("warnings") or []))
        return self._doctor

    def operation_status(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """`imagemagick-skill:<TYPE>` verdicts from the doctor's `tools.<name>.usable`: supported / unavailable."""
        tools = as_dict((doc if doc is not None else self.doctor()).get("tools"))
        return {typ: ("supported" if as_dict(tools.get(name)).get("usable") is True else "unavailable") for name, typ in OPERATION_TYPES.items()}

    def operation_evidence(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Which backend would run each tool (magick / sips) and, when none can, what to install, in the Skill's own words."""
        doc = doc if doc is not None else self.doctor()
        tools, backends = as_dict(doc.get("tools")), as_dict(doc.get("backends"))
        out: Dict[str, Dict[str, Any]] = {}
        for name, typ in OPERATION_TYPES.items():
            row = as_dict(tools.get(name))
            ev: Dict[str, Any] = {"backend": row.get("backend"), "usable": row.get("usable") is True, "lifecycle": self.lifecycle.get(CAPABILITY_IDS[typ]),
                                  "magick": as_dict(backends.get("magick")).get("kind")}
            if row.get("usable") is not True:
                ev["code"] = "BACKEND_UNAVAILABLE"
                ev["detail"] = str(row.get("reason") or row.get("fix") or "no ImageMagick (magick, or ImageMagick 6 convert / identify) on PATH")[:300]
            out[typ] = ev
        return out

    def drift(self) -> List[str]:
        return drift(self.contract)

    def breaking_drift(self) -> List[str]:
        return breaking_drift(self.contract)

    # ---- ToolAdapter
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "skill": self.skill.describe(), "tools": list(TOOLS), "lifecycle": dict(self.lifecycle), "drift": self.drift()}

    def package(self) -> SkillPackage:
        return SkillPackage(PACKAGE.skill_id, PACKAGE.name, self.version, PACKAGE.description, list(PACKAGE.capabilities),
                            [ToolSpec(**{**t.__dict__, "version": self.version}) for t in PACKAGE.tools], PACKAGE.repository, PACKAGE.role)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        raise ToolError("imagemagick-skill: the agent uses it for preparation only (resize / pad / strip), never as a measurement")

    # ---- request construction (typed args → argv); nothing here reads a value as an option
    @staticmethod
    def _check_raw(raw: Any, what: str) -> None:
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw or "\n" in raw:
            raise ToolError(f"{what} must be a non-empty single-line path")
        if any(part == ".." for part in raw.replace("\\", "/").split("/")):
            raise ToolError(f"{what} path contains '..' (traversal)")

    @staticmethod
    def _within(root: str, p: str) -> bool:
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
        if os.path.splitext(resolved)[1].lower() not in IMAGE_EXTENSIONS:
            raise ToolError(f"input must be a PNG / JPEG image: {os.path.basename(raw)}")
        if not os.path.isfile(resolved):
            raise ToolError(f"input not found: {os.path.basename(raw)}")
        if self.policy is not None:
            self.policy.check_input(resolved)
        return resolved

    def _output(self, ref: Any, paths: Dict[str, str], src: str) -> str:
        if not isinstance(ref, str) or ref not in paths:
            raise ToolError(f"output {ref!r} is not an artifact id of this job")
        raw = paths[ref]
        self._check_raw(raw, "output")
        absolute = os.path.join(os.path.realpath(os.path.dirname(os.path.abspath(raw))), os.path.basename(raw))
        if self.workspace and not (self._within(self.workspace, absolute) and absolute != self.workspace):
            raise ToolError(f"output is outside the workspace: {os.path.basename(raw)}")
        if os.path.splitext(absolute)[1].lower() not in IMAGE_EXTENSIONS:
            raise ToolError(f"output must be a .png / .jpg file: {os.path.basename(raw)}")
        if same_file(src, absolute) or os.path.normcase(src) == os.path.normcase(absolute):
            raise ToolError("output would overwrite the input")
        if self.policy is not None:
            self.policy.check_output(absolute, [src])
        return absolute

    def _options(self, tool: str, args: Dict[str, Any]) -> List[str]:
        """`--flag=value` words for the tool's parameters, each checked against the live contract's schema (type, enum, flag spelling)."""
        props = as_dict(tool_schema(self.contract, tool).get("properties"))
        out: List[str] = []
        for name in PARAMETERS[tool]:
            v = args.get(name)
            if v is None:
                if name in REQUIRED[tool]:
                    raise ToolError(f"{tool} needs {name}")
                continue
            p = as_dict(props.get(name))
            flag = _flag(p, name)
            if flag is None:
                raise ToolError(f"the installed contract has no --{name} for {tool}")
            if name in ("width", "height"):
                value = str(_int_value(v, name))   # pad declares them as strings of pixels; resize as integers: an integer either way
            elif p.get("enum"):
                if not isinstance(v, str) or v not in p["enum"]:
                    raise ToolError(f"{name} must be one of {', '.join(map(str, p['enum']))}, got {v!r}")
                value = v
            elif name == "color":
                if not isinstance(v, str) or not COLOR_RE.match(v):
                    raise ToolError(f"color must be none, a colour name or #RRGGBB / #RRGGBBAA, got {v!r}")
                value = v
            else:
                raise ToolError(f"{tool} parameter {name} has no check in this adapter")
            out.append(f"{flag}={value}")
        return out

    def build(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None) -> Dict[str, Any]:
        """{"argv", "input", "output", "timeout"} for one operation, or ToolError naming the first value that cannot be sent."""
        if op.tool not in TOOLS:
            raise ToolError(f"{SKILL_ID} adapter does not support {op.tool}")
        args = dict(op.args)
        hit = scan_forbidden(args, FORBIDDEN_ARG_KEYS)
        if hit:
            raise ToolError(f"forbidden key {hit}")
        unknown = sorted(k for k in args if k not in set(PARAMETERS[op.tool]) | {"input", "output"})
        if unknown:
            raise ToolError(f"unknown {op.tool} argument(s): {', '.join(unknown)}")
        src = self._input(args.get("input"), paths)
        out = self._output(args.get("output"), paths, src)
        options = self._options(op.tool, args)
        t = max(1.0, min(3600.0, float(timeout or self.default_timeout)))
        argv = [f"--output={out}"] + options + ["--overwrite", "--json", f"--timeout={t:g}", "--", src]
        return {"argv": argv, "input": src, "output": out, "timeout": t, "args": {k: args[k] for k in PARAMETERS[op.tool] if args.get(k) is not None}}

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            b = self.build(op, paths)
        except ToolError as e:
            return [f"<refused: {e}>"]
        return [" ".join([f"scripts/{SCRIPT[op.tool]}.py"] + b["argv"])]

    # ---- execution
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
        command = " ".join([f"scripts/{SCRIPT[op.tool]}.py"] + b["argv"])
        if dry_run:   # the executor's dry run checks the request and runs nothing (the Skill's own --dry-run is not needed for that)
            return ToolResult(op.id, op.tool, True, 0, None, {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run", "commands": [command]},
                              [command], "", round(time.time() - t0, 3), attempt, True)
        before = sha256_file(b["input"])
        code, out, err = self._call(SCRIPT[op.tool], b["argv"], b["timeout"] + PROCESS_MARGIN)
        tail = "\n".join(err.strip().splitlines()[-12:])
        try:
            doc: Optional[Dict[str, Any]] = one_json_document(out, f"imagemagick-skill {SCRIPT[op.tool]}")
        except ToolError as e:
            doc = None
            parse_error = str(e)
        if code == 124 or (doc is not None and doc.get("code") == "timeout"):
            self._remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, 124, "TIMEOUT", f"imagemagick-skill {SCRIPT[op.tool]} exceeded {b['timeout']:g}s", retryable=True,
                              details={"reason": "timeout"})
        if doc is None:
            self._remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code else 9, "INVALID_OUTPUT", f"{parse_error} (exit {code}) {tail[-300:]}".strip())
        if sha256_file(b["input"]) != before:
            self._remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code else 9, "INPUT_CHANGED", "the input image changed during the call (the Skill promises it never writes its input)")
        if doc.get("ok") is not True:
            self._remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code else 1, "SKILL_FAILED", str(doc.get("reason") or tail or "")[:1200],
                              details={k: v for k, v in doc.items() if k not in ("ok", "reason") and isinstance(v, (str, int, float, bool))})
        why = self._admit(op.tool, doc, b)
        if why:
            self._remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code else 9, "INVALID_OUTPUT", "imagemagick-skill result not admitted: " + "; ".join(why))
        if code != 0:
            self._remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code, "INVALID_OUTPUT", f"exit code {code} with an ok result")
        data = self._success_data(op.tool, doc, b, before, command)
        return ToolResult(op.id, op.tool, True, 0, b["output"], data, [command], tail, round(time.time() - t0, 3), attempt, False)

    def _admit(self, tool: str, doc: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
        """Why a result is not about this request (empty = admitted)."""
        errs: List[str] = []
        if not same_file(str(doc.get("input") or ""), b["input"]):
            errs.append(f"input {os.path.basename(str(doc.get('input') or ''))!r} is not the planned {os.path.basename(b['input'])}")
        if not same_file(str(doc.get("output") or ""), b["output"]):
            return errs + [f"output {os.path.basename(str(doc.get('output') or ''))!r} is not the planned {os.path.basename(b['output'])}"]
        if not os.path.isfile(b["output"]) or os.path.getsize(b["output"]) == 0:
            return errs + [f"{os.path.basename(b['output'])} missing or empty although the Skill reported success"]
        a = b["args"]
        actual = as_dict(doc.get("actual"))
        size = (actual.get("width"), actual.get("height"))
        if tool == TOOL_RESIZE:
            w, h = int(a["width"]), int(a["height"])
            mode = a.get("mode") or doc.get("mode")
            if a.get("mode") is not None and doc.get("mode") != a["mode"]:
                errs.append(f"mode {doc.get('mode')!r} is not the requested {a['mode']!r}")
            if as_dict(doc.get("requested")) != {"width": w, "height": h}:
                errs.append(f"requested {doc.get('requested')!r} is not {w}x{h}")
            if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in size):
                errs.append(f"actual size {actual!r} is not measured")
            elif mode == "fit":
                if size[0] > w or size[1] > h or (abs(size[0] - w) > 1 and abs(size[1] - h) > 1):
                    errs.append(f"fit result {size[0]}x{size[1]} does not fit {w}x{h} touching the box")
            elif size != (w, h):
                errs.append(f"{mode} result {size[0]}x{size[1]} is not {w}x{h}")
        elif tool == TOOL_PAD:
            if size != (int(a["width"]), int(a["height"])):
                errs.append(f"pad result {size[0]}x{size[1]} is not the {a['width']}x{a['height']} canvas")
            if doc.get("mode") != "canvas":
                errs.append(f"pad mode {doc.get('mode')!r} is not canvas")
        elif doc.get("has_gps") is not False:
            errs.append(f"strip result reports has_gps {doc.get('has_gps')!r}")
        if doc.get("backend") not in ("magick", "sips"):
            errs.append(f"backend {doc.get('backend')!r} is not magick / sips")
        return errs

    def _success_data(self, tool: str, doc: Dict[str, Any], b: Dict[str, Any], input_sha: str, command: str) -> Dict[str, Any]:
        """What the agent records: the operation, its parameters, the backend, the sizes the Skill measured, and the output's sha256 (hashed
        here: the Skill reports none) -- never a local path beyond the output the agent itself named."""
        name = SCRIPT[tool]
        cap = CAPABILITY_IDS[OPERATION_TYPES[name]]
        actual = as_dict(doc.get("actual"))
        facts = {k: doc[k] for k in ("mode", "requested", "original", "offset", "color", "has_gps") if k in doc}
        return {"skill": {"id": SKILL_ID, "version": self.version}, "status": "ok", "op": name, "capability": cap, "lifecycle": self.lifecycle.get(cap),
                "backend": doc.get("backend"), "parameters": dict(b["args"]), "result": facts,
                "output": {"file": b["output"], "sha256": sha256_file(b["output"]), "width": actual.get("width"), "height": actual.get("height"),
                           "format": os.path.splitext(b["output"])[1].lstrip(".").lower()},
                "input_sha256": input_sha, "commands": [command]}


def lift_observation(result: ToolResult, asset_id: Optional[str] = None) -> Optional[Observation]:
    """What imagemagick-skill reported about the image it prepared, as an agent Observation (provenance only; never fed back into the
    analysis): operation, parameters, backend, measured size, and the input / output sha256."""
    data = result.data if isinstance(result.data, dict) else {}
    if data.get("status") != "ok" or not data.get("output"):
        return None
    sk = as_dict(data.get("skill"))
    out = dict(as_dict(data.get("output")))
    return Observation(kind=f"image.{data.get('op')}", asset_id=asset_id or result.op_id, source=f"{result.tool}@{sk.get('version') or ''}",
                       data={"capability": data.get("capability"), "lifecycle": data.get("lifecycle"), "backend": data.get("backend"),
                             "parameters": as_dict(data.get("parameters")), "result": as_dict(data.get("result")), "output": out,
                             "input_sha256": data.get("input_sha256")},
                       analyzer=f"{SKILL_ID}@{sk.get('version') or ''}", provenance="OBSERVED", skill=SKILL_ID, skill_version=str(sk.get("version") or ""),
                       tool=result.tool, fingerprint=str(out.get("sha256") or ""), parameters={"output": out.get("file")})
