"""MotionGraphicsAdapter: the agent's boundary to motion-graphics-skill (deterministic motion-graphics execution Skill, ADR-031).

    video-production-agent ─(typed Operation)─→ MotionGraphicsAdapter ─(motion-graphics/request@1 on stdin)─→ `motion-graphics run - --json`
        ─(typed ffmpeg-skill calls: probe / graphics / overlay)─→ ffmpeg-skill ─→ FFmpeg

Protocol (from the Skill's own contract, `motion-graphics skill --json`):
    contract → `motion-graphics skill --json`
    doctor   → `motion-graphics doctor --json [--workspace D] [--allowed-input R]… [--ffmpeg-skill X]`
    run      → `motion-graphics run - --json --workspace D --allowed-input R… [--ffmpeg-skill X] [--timeout S] [--dry-run]` (request on stdin)
    response ← one JSON document: {"ok": true, "status": "ok", "output", "timeline", "operations", "engine", "provenance"} or
               {"ok": false, "error": {code, message, retryable, details}}

The agent hands one typed operation: {"input": <video id>, "output": <artifact id>, "elements": [{id, type, start, end, parameters,
animation?}], "crf"?, "preset"?}. Every element is checked against the element-type parameter schema the contract declares
(type / min / max / max_length / required, named positions, the font registry, the fade animation and its applicability) and
forbidden keys are refused by name; the request carries paths only for the video, the output and an image asset (workspace and
allowed roots go on argv). The response is verified: schema / skill / status / dry_run, the output's realpath, sha256 recomputed
from the file, size, the probe facts, the timeline, the per-operation records and the provenance chain. Commands the Skill
observed are recorded as provenance only.

Contract discrepancies absorbed here (documented, never patched into either project):
- `image_overlay.image_path` is a `path` parameter. The agent never accepts a path from a plan: the element carries "image": <artifact
  id>, resolved through `paths`, required to be an existing .png/.jpg/.jpeg under the allowed roots / workspace, and emitted as
  `image_path` (absolute). "image_path" itself is refused from the agent.
- `font` parameters are `{font_id}` | `{font_file}` objects on the wire. The agent accepts only a registry font id (`fonts.registry`)
  as a string and sends `{"font_id": …}`; a custom font file is not accepted from the agent (it would be a path from a plan).
- The x264 preset enum is not published by the contract (request.shape shows "medium" only); the Skill's own list is pinned here
  as PRESETS and re-checked by the Skill.
- `color` values are validated by the Skill (named colour / RRGGBB hex with optional @alpha); the agent only refuses shapes that
  cannot be a colour (non-empty string ≤ 32 chars of [A-Za-z0-9#@.])."""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...models import Observation, Operation, ToolResult
from ...skills.contract import SkillPackage, ToolSpec
from ..base import ToolAdapter, ToolError
from ..ffmpeg_skill.adapter import PathPolicy
from ..skill_process import (FORBIDDEN_ARG_KEYS, CliSkill, ContractError, as_dict, drift_report, error_table, failed_result, fingerprint_matches, invoke, one_json_document,
                             remove_fresh, same_file, scan_forbidden, scrub)
from .locate import locate_motion_graphics

SKILL_ID = "motion-graphics"
PREFIX = SKILL_ID + "/"
TOOL_ID = "motion-graphics/run"
CONTRACT_SCHEMA = "motion-graphics/contract@1"
REQUEST_SCHEMA = "motion-graphics/request@1"
RESPONSE_SCHEMA = "motion-graphics/response@1"
DOCTOR_SCHEMA = "motion-graphics/doctor@1"
SUPPORTED_SKILL_VERSIONS = ("0.1.",)
ENGINE_ID = "ffmpeg-skill"
CANONICAL_INVOCATION = ["motion-graphics", "run", "-", "--json"]
REQUIRED_EXECUTION_FLAGS = {"shell": False, "arbitrary_executables": False, "arbitrary_filters": False, "network": False, "input_mutation": False, "ai": False}
COMMON_ARGS = ("input", "output", "elements", "crf", "preset")
ELEMENT_KEYS = ("id", "type", "start", "end", "parameters", "animation")
PARAMETER_TYPES = ("string", "integer", "number", "boolean", "color", "position", "font", "path")
PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")
OUTPUT_EXTENSION = ".mp4"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_COLOR_SHAPE_RE = re.compile(r"^[A-Za-z0-9#@.]+$")
DRIFT_KEYS = ("schema", "skill_id", "version", "kind", "tools", "unsupported_element_types", "animations", "unsupported_animations", "positions", "fonts", "output_formats",
              "limits", "timeline", "execution", "ffmpeg_skill", "request", "response", "provenance", "work_dir", "image_formats", "schema_versions", "errors")
DRIFT_ELEMENT_KEYS = ("type", "tool", "animation", "parameters", "required_capabilities", "deterministic")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def check_contract(contract: Any) -> List[str]:
    """Compatibility checks: schema, id, version range, kind, execution flags, canonical invocation, the single tool, typed element
    parameters, the fade animation, positions, the font registry, limits, forbidden fields, error codes. Anything off is refused, never patched."""
    errs: List[str] = []
    if not isinstance(contract, dict):
        return ["contract is not an object"]
    if contract.get("schema") != CONTRACT_SCHEMA:
        errs.append(f"contract schema {contract.get('schema')!r} != {CONTRACT_SCHEMA}")
    if contract.get("skill_id") != SKILL_ID or contract.get("id") != SKILL_ID:
        errs.append(f"skill_id {contract.get('skill_id')!r} / id {contract.get('id')!r} != {SKILL_ID}")
    ver = str(contract.get("version") or "")
    if not ver.startswith(SUPPORTED_SKILL_VERSIONS):
        errs.append(f"skill version {ver!r} not in supported range {SUPPORTED_SKILL_VERSIONS}")
    if contract.get("kind") != "execution":
        errs.append(f"kind {contract.get('kind')!r} != execution")
    ex = contract.get("execution") or {}
    if ex.get("mode") != "local":
        errs.append(f"execution.mode {ex.get('mode')!r} != local")
    for k, want in REQUIRED_EXECUTION_FLAGS.items():
        if ex.get(k) is not want:
            errs.append(f"execution.{k} must be {want!r}, contract says {ex.get(k)!r}")
    if list(ex.get("canonical_invocation") or []) != CANONICAL_INVOCATION:
        errs.append(f"canonical_invocation {ex.get('canonical_invocation')!r} != {CANONICAL_INVOCATION}")
    if ex.get("stdin") != REQUEST_SCHEMA:
        errs.append(f"execution.stdin {ex.get('stdin')!r} != {REQUEST_SCHEMA}")
    fs = contract.get("ffmpeg_skill") or {}
    if str(fs.get("contract_version")) != "1.0" or not isinstance(fs.get("version_window"), dict):
        errs.append("ffmpeg_skill.contract_version / version_window missing")
    types = {str(o.get("type")): o for o in contract.get("element_types") or [] if isinstance(o, dict)}
    if not types:
        errs.append("no element_types declared")
    for t, o in types.items():
        if not re.match(r"^[a-z_]{2,32}$", t):
            errs.append(f"element type {t!r} is not a lower-case identifier")
        if not isinstance(o.get("parameters"), dict) or not isinstance(o.get("required_capabilities"), list):
            errs.append(f"element type {t}: parameters / required_capabilities missing")
        if not str(o.get("tool", "")).startswith(ENGINE_ID + "/"):
            errs.append(f"element type {t}: executed by {o.get('tool')!r}, not an {ENGINE_ID} tool")
        for k, ps in (o.get("parameters") or {}).items():
            if not isinstance(ps, dict) or ps.get("type") not in PARAMETER_TYPES:
                errs.append(f"element type {t}: parameter {k!r} has no typed schema")
    unsupported = {str(u.get("type")) for u in contract.get("unsupported_element_types") or [] if isinstance(u, dict)}
    if unsupported & set(types):
        errs.append(f"element types declared both supported and unsupported: {sorted(unsupported & set(types))}")
    anims = {str(a.get("kind")): a for a in contract.get("animations") or [] if isinstance(a, dict)}
    fade = anims.get("fade")
    if not fade or not isinstance(as_dict(fade.get("parameters")).get("duration"), dict) or not isinstance(fade.get("applies_to"), list):
        errs.append("animations must declare fade with a duration parameter and applies_to")
    else:
        for t in fade["applies_to"]:
            if str(t) not in types:
                errs.append(f"animation fade applies to an undeclared element type {t!r}")
    pos = contract.get("positions") or {}
    if not isinstance(pos.get("named"), list) or not pos["named"] or "explicit" not in pos:
        errs.append("positions.named / positions.explicit missing")
    fonts = contract.get("fonts") or {}
    if not isinstance(fonts.get("registry"), dict) or not fonts["registry"] or fonts.get("default_font_id") not in fonts["registry"]:
        errs.append("fonts.registry / default_font_id missing")
    lim = contract.get("limits") or {}
    if not isinstance(lim.get("max_elements"), int) or lim["max_elements"] < 1 or not isinstance(lim.get("max_element_duration_seconds"), (int, float)):
        errs.append("limits.max_elements / max_element_duration_seconds missing")
    tools = contract.get("tools") or []
    if len(tools) != 1 or tools[0].get("tool_id") != TOOL_ID or tools[0].get("skill_id") != SKILL_ID:
        errs.append(f"expected exactly one tool {TOOL_ID}, contract declares {[t.get('tool_id') for t in tools]}")
    else:
        t0 = tools[0]
        if t0.get("role") != "execution" or t0.get("produces_output") is not True or t0.get("deterministic") is not True or t0.get("input_type") != REQUEST_SCHEMA:
            errs.append("tool must declare role execution, produces_output / deterministic = true and the request schema as input_type")
        if sorted(str(x) for x in t0.get("element_types") or []) != sorted(types):
            errs.append(f"tool element_types {t0.get('element_types')!r} != declared element types {sorted(types)}")
    if not isinstance(contract.get("output_formats"), dict) or not contract["output_formats"].get("video_codec"):
        errs.append("output_formats must declare video_codec")
    req = contract.get("request") or {}
    if req.get("schema") != REQUEST_SCHEMA:
        errs.append(f"request.schema {req.get('schema')!r} != {REQUEST_SCHEMA}")
    if not req.get("id_pattern"):
        errs.append("request.id_pattern missing")
    forbidden = set(req.get("forbidden_fields") or [])
    for k in ("command", "argv", "filter", "shell", "exec", "env"):
        if k not in forbidden:
            errs.append(f"request.forbidden_fields lacks {k!r}")
    retry, exit_codes = error_table(contract)
    for c in ("INVALID_REQUEST", "INVALID_INPUT", "PATH_NOT_ALLOWED", "UNSUPPORTED_OPERATION", "UNSUPPORTED_FORMAT", "INVALID_TIME_RANGE", "DEPENDENCY_ERROR", "MISSING_INPUT",
              "TOOL_ERROR", "OUTPUT_ERROR", "VALIDATION_ERROR", "CANCELLED", "INTERNAL_ERROR"):
        if c not in retry:
            errs.append(f"errors.codes lacks {c}")
    if not exit_codes:
        errs.append("errors.exit_codes missing")
    return errs


def contract_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(live, pinned or pinned_contract(), DRIFT_KEYS, "element_types", "type", DRIFT_ELEMENT_KEYS)


def package_from_contract(contract: Dict[str, Any]) -> SkillPackage:
    ver = str(contract.get("version") or "")
    tools = [ToolSpec(tool_id=TOOL_ID, skill_id=SKILL_ID, version=ver, description=str(t.get("description", "")), required_capabilities=[SKILL_ID],
                      inputs=["input", "output", "image"], produces_output=True, deterministic=True,
                      result_keys=["operation_type", "artifact", "timeline", "operations", "engine", "observation", "provenance", "commands"]) for t in contract.get("tools") or []]
    return SkillPackage(skill_id=SKILL_ID, name=str(contract.get("name") or SKILL_ID), version=ver, description=str(contract.get("description", ""))[:200],
                        capabilities=["ffmpeg", "ffprobe", "ffmpeg-skill", SKILL_ID], tools=tools, repository="kajisho5/motion-graphics-skill",
                        role="deterministic motion graphics rendering (titles, lower thirds, text / image overlays) through ffmpeg-skill")


PACKAGE = package_from_contract(pinned_contract())


class MotionGraphicsAdapter(ToolAdapter):
    name = SKILL_ID

    def __init__(self, skill: Optional[CliSkill] = None, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None,
                 ffmpeg_skill_dir: Optional[str] = None, timeout: float = 600.0, path_policy: Optional[PathPolicy] = None):
        located = skill or locate_motion_graphics()
        if not located:
            raise ToolError("motion-graphics-skill not found (set VIDEO_AGENT_MOTION_GRAPHICS_DIR or install `motion-graphics`)")
        self.skill: CliSkill = located
        self.workspace = str(Path(workspace).resolve()) if workspace else None
        self.allowed_inputs = [str(Path(r).resolve()) for r in (allowed_inputs or [])]
        self.ffmpeg_skill_dir = ffmpeg_skill_dir
        self.default_timeout = float(timeout)
        self.path_policy = path_policy
        self.calls = 0
        self.contract = self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("motion-graphics contract incompatible: " + "; ".join(errs))
        self.version = str(self.contract["version"])
        self.element_types: Dict[str, Dict[str, Any]] = {str(o["type"]): o for o in self.contract["element_types"]}
        self.unsupported = sorted(str(u.get("type")) for u in self.contract.get("unsupported_element_types") or [])
        self.animations: Dict[str, Dict[str, Any]] = {str(a["kind"]): a for a in self.contract["animations"]}
        self.positions = [str(p) for p in self.contract["positions"]["named"]]
        self.fonts = sorted(str(f) for f in self.contract["fonts"]["registry"])
        self.image_extensions = tuple(str(e).lower() for e in as_dict(self.contract.get("image_formats")).get("allowed_extensions") or (".png", ".jpg", ".jpeg"))
        self.max_elements = int(self.contract["limits"]["max_elements"])
        self.max_duration = float(self.contract["limits"]["max_element_duration_seconds"])
        self.id_re = re.compile(str(self.contract["request"]["id_pattern"]))
        self.forbidden = tuple(sorted(set(FORBIDDEN_ARG_KEYS) | {str(f) for f in (self.contract.get("request") or {}).get("forbidden_fields") or []}))
        self.retryable, self.exit_codes = error_table(self.contract)
        self.tools = {TOOL_ID}
        self._drift: Optional[List[str]] = None

    # ---- transport
    def _invoke(self, argv: List[str], stdin: Optional[str] = None, timeout: Optional[float] = None):
        self.calls += 1
        return invoke(self.skill, argv, stdin=stdin, timeout=timeout or self.default_timeout)

    def _engine_argv(self) -> List[str]:
        return ["--ffmpeg-skill", self.ffmpeg_skill_dir] if self.ffmpeg_skill_dir else []

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._invoke(["skill", "--json"], timeout=60.0)
        if code != 0:
            raise ContractError(f"motion-graphics skill --json failed (exit {code}): {err.strip()[-300:]}")
        try:
            return one_json_document(out, "motion-graphics contract")
        except ToolError as e:
            raise ContractError(str(e))

    def doctor(self) -> Dict[str, Any]:
        argv = ["doctor", "--json"] + (["--workspace", self.workspace] if self.workspace else []) + [x for r in self.allowed_inputs for x in ("--allowed-input", r)] + self._engine_argv()
        code, out, err = self._invoke(argv, timeout=180.0)
        try:
            doc = one_json_document(out, "motion-graphics doctor")
        except ToolError as e:
            return {"schema": DOCTOR_SCHEMA, "status": "fail", "problems": [f"doctor produced no document: {e}"], "checks": {}, "exit_code": code}
        if doc.get("schema") != DOCTOR_SCHEMA or doc.get("status") not in ("ok", "degraded", "fail"):
            return {"schema": DOCTOR_SCHEMA, "status": "fail", "problems": [f"unexpected doctor document {doc.get('schema')!r} / {doc.get('status')!r}"], "checks": doc.get("checks") or {}, "exit_code": code}
        doc["exit_code"] = code
        return doc

    @staticmethod
    def element_status(doc: Dict[str, Any]) -> Dict[str, str]:
        """Per element type: supported | unknown | unavailable, as the Skill's doctor reports it (never upgraded here: on a machine
        where ffmpeg-skill's doctor does not classify drawbox / overlay / color / scale, title / lower_third / image_overlay stay unknown)."""
        types = as_dict(as_dict(doc.get("checks")).get("element_types"))
        return {str(t): str(as_dict(v).get("status") or "unknown") for t, v in types.items()}

    @staticmethod
    def font_status(doc: Dict[str, Any]) -> Dict[str, str]:
        """Per registry font id: the status the Skill's doctor reports (fc-list detection); never inferred here."""
        fonts = as_dict(as_dict(as_dict(doc.get("checks")).get("fonts")).get("fonts"))
        return {str(f): str(v or "unknown") for f, v in fonts.items()}

    def drift(self) -> List[str]:
        if self._drift is None:
            self._drift = contract_drift(self.contract)
        return self._drift

    # ---- ToolAdapter
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "root": self.skill.describe(), "tools": sorted(self.tools), "element_types": sorted(self.element_types),
                "unsupported": self.unsupported, "animations": sorted(self.animations), "fonts": self.fonts, "contract": self.contract.get("schema"), "drift": self.drift()}

    def package(self) -> SkillPackage:
        return package_from_contract(self.contract)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            b = self.build_request(op.tool, op.args, paths, op_id=op.id)
        except ToolError as e:
            return [f"motion-graphics: refused: {e}"]
        return [" ".join(["motion-graphics"] + self._argv(b, dry_run=True, timeout=None)) + "  <<< " + json.dumps(b["request"], ensure_ascii=False)[:400]]

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        raise ToolError("motion-graphics: an execution Skill has no measurement tools")

    # ---- lowering: typed elements → contract-shaped elements
    def element_for(self, raw: Any, index: int, paths: Dict[str, str]) -> Dict[str, Any]:
        where = f"elements[{index}]"
        if not isinstance(raw, dict):
            raise ToolError(f"motion-graphics: {where} must be an object")
        unknown = sorted(set(raw) - set(ELEMENT_KEYS))
        if unknown:
            raise ToolError(f"motion-graphics: {where} has unknown field(s) {unknown}")
        for k in ("id", "type", "start", "end"):
            if k not in raw:
                raise ToolError(f"motion-graphics: {where} requires {k!r}")
        el_id = raw["id"]
        if not isinstance(el_id, str) or not self.id_re.match(el_id):
            raise ToolError(f"motion-graphics: {where}.id must match {self.id_re.pattern}")
        et = str(raw["type"])
        if et in self.unsupported:
            raise ToolError(f"motion-graphics: element type {et} is declared unsupported by the Skill contract")
        spec = self.element_types.get(et)
        if spec is None:
            raise ToolError(f"motion-graphics: unknown element type {et!r} (contract declares {sorted(self.element_types)})")
        start, end = self._finite(raw["start"], f"{where}.start"), self._finite(raw["end"], f"{where}.end")
        if start < 0 or end <= start or end > self.max_duration:
            raise ToolError(f"motion-graphics: {where} needs 0 <= start < end <= {self.max_duration} (got {start}..{end})")
        params = self.params_for(et, raw.get("parameters") or {}, paths, where)
        out: Dict[str, Any] = {"id": el_id, "type": et, "start": start, "end": end, "parameters": params}
        if "animation" in raw and raw["animation"] is not None:
            out["animation"] = self.animation_for(et, raw["animation"], where)
        return out

    def params_for(self, et: str, params: Any, paths: Dict[str, str], where: str) -> Dict[str, Any]:
        if not isinstance(params, dict):
            raise ToolError(f"motion-graphics: {where}.parameters must be an object")
        schema: Dict[str, Any] = self.element_types[et].get("parameters") or {}
        fb = {f.lower() for f in self.forbidden}
        out: Dict[str, Any] = {}
        for k, v in params.items():
            if str(k).lower() in fb:
                raise ToolError(f"motion-graphics: {where}: parameter {k!r} is not accepted (forbidden field)")
            if k == "image" and "image_path" in schema:
                image = paths.get(str(v), str(v))
                if Path(image).suffix.lower() not in self.image_extensions or not os.path.isfile(image):
                    raise ToolError(f"motion-graphics: {where}: image must be an existing {'/'.join(self.image_extensions)} file: {image}")
                self._check_input(image, what=f"{where} image")
                out["image_path"] = str(Path(image).resolve())
                continue
            if k not in schema or str(schema[k].get("type")) == "path":
                raise ToolError(f"motion-graphics: {where}: parameter {k!r} is not declared for {et} (contract: {sorted(x for x in schema if schema[x].get('type') != 'path')})")
            out[k] = self._typed(f"{where}.parameters.{k}", v, schema[k])
        for k, ps in schema.items():
            if ps.get("required") and k not in out:
                raise ToolError(f"motion-graphics: {where}: {et} requires parameter {'image' if k == 'image_path' else k!r}")
        return out

    def animation_for(self, et: str, raw: Any, where: str) -> Dict[str, Any]:
        if not isinstance(raw, dict) or set(raw) != {"kind", "parameters"} or not isinstance(raw.get("parameters"), dict):
            raise ToolError(f"motion-graphics: {where}.animation must be {{kind, parameters}}")
        kind = str(raw["kind"])
        spec = self.animations.get(kind)
        if spec is None:
            raise ToolError(f"motion-graphics: {where}.animation kind {kind!r} is not implemented (contract declares {sorted(self.animations)})")
        if et not in [str(t) for t in spec.get("applies_to") or []]:
            raise ToolError(f"motion-graphics: {where}.animation {kind} does not apply to {et} (only {spec.get('applies_to')})")
        aschema: Dict[str, Any] = spec.get("parameters") or {}
        out: Dict[str, Any] = {}
        for k, v in raw["parameters"].items():
            if k not in aschema:
                raise ToolError(f"motion-graphics: {where}.animation parameter {k!r} is not declared for {kind}")
            out[k] = self._typed(f"{where}.animation.parameters.{k}", v, aschema[k])
        for k, ps in aschema.items():
            if ps.get("required") and k not in out:
                raise ToolError(f"motion-graphics: {where}.animation {kind} requires {k!r}")
        return {"kind": kind, "parameters": out}

    @staticmethod
    def _finite(v: Any, name: str) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ToolError(f"motion-graphics: {name} must be a finite number")
        return float(v)

    def _typed(self, name: str, v: Any, ps: Dict[str, Any]) -> Any:
        t = ps.get("type")
        if t == "boolean":
            if not isinstance(v, bool):
                raise ToolError(f"motion-graphics: {name} must be a boolean")
            return v
        if t in ("string", "color", "font"):
            if not isinstance(v, str) or "\n" in v or "\x00" in v or (t != "string" and not v):
                raise ToolError(f"motion-graphics: {name} must be a {'string' if t == 'string' else 'non-empty string'}")
            if t == "string":
                if "max_length" in ps and len(v) > ps["max_length"]:
                    raise ToolError(f"motion-graphics: {name} is longer than {ps['max_length']} characters")
                if "enum" in ps and v not in ps["enum"]:
                    raise ToolError(f"motion-graphics: {name} {v!r} is not one of {ps['enum']}")
                return v
            if t == "color":
                if len(v) > 32 or not _COLOR_SHAPE_RE.match(v):
                    raise ToolError(f"motion-graphics: {name} {v!r} cannot be a colour (named colour or RRGGBB hex, optionally @alpha)")
                return v
            if v not in self.fonts:
                raise ToolError(f"motion-graphics: {name} {v!r} is not a registry font id {self.fonts} (custom font files are not accepted from the agent)")
            return {"font_id": v}
        if t == "position":
            if isinstance(v, str):
                if v not in self.positions:
                    raise ToolError(f"motion-graphics: {name} {v!r} is not one of {self.positions} or {{x, y}}")
                return v
            if isinstance(v, dict) and set(v) == {"x", "y"} and all(isinstance(v[a], int) and not isinstance(v[a], bool) for a in ("x", "y")):
                return {"x": v["x"], "y": v["y"]}
            raise ToolError(f"motion-graphics: {name} must be a named position or {{x: int, y: int}}")
        x: Any = self._finite(v, name)
        if t == "integer":
            if x != int(x):
                raise ToolError(f"motion-graphics: {name} must be an integer")
            x = int(x)
        if "min" in ps and x < ps["min"]:
            raise ToolError(f"motion-graphics: {name} {x} is below the contract minimum {ps['min']}")
        if "max" in ps and x > ps["max"]:
            raise ToolError(f"motion-graphics: {name} {x} is above the contract maximum {ps['max']}")
        return x

    def _check_input(self, path: str, what: str) -> None:
        if not os.path.isfile(path):
            raise ToolError(f"motion-graphics: {what} not found: {path}")
        if self.path_policy is not None:
            self.path_policy.check_input(path)
        elif self.allowed_inputs and not any(self._under(path, r) for r in self.allowed_inputs + ([self.workspace] if self.workspace else [])):
            raise ToolError(f"motion-graphics: {what} outside the allowed roots: {path}")

    def build_request(self, tool: str, args: Dict[str, Any], paths: Dict[str, str], op_id: str = "op", timeout: Optional[float] = None) -> Dict[str, Any]:
        if tool != TOOL_ID:
            raise ToolError(f"motion-graphics: unsupported tool {tool}")
        hit = scan_forbidden(args, self.forbidden)
        if hit:
            raise ToolError(f"motion-graphics: forbidden field {hit} in the operation arguments")
        unknown = sorted(set(args) - set(COMMON_ARGS))
        if unknown:
            raise ToolError(f"motion-graphics: unknown argument(s) {unknown} (accepted: {list(COMMON_ARGS)})")
        src_id, out_id = str(args.get("input") or ""), str(args.get("output") or "")
        if not src_id or not out_id:
            raise ToolError("motion-graphics: input and output references are required")
        src = paths.get(src_id, src_id)
        out = paths.get(out_id, out_id)
        self._check_input(src, what="input")
        if self.path_policy is not None:
            self.path_policy.check_output(out, [src])
        if self.workspace and not self._under(out, self.workspace):
            raise ToolError(f"motion-graphics: output outside the workspace: {out}")
        if Path(out).suffix.lower() != OUTPUT_EXTENSION:
            raise ToolError(f"motion-graphics: output must be a {OUTPUT_EXTENSION} file: {out}")
        if same_file(src, out):
            raise ToolError("motion-graphics: output would overwrite its input")
        raw_elements = args.get("elements")
        if not isinstance(raw_elements, list) or not raw_elements:
            raise ToolError("motion-graphics: elements must be a non-empty list")
        if len(raw_elements) > self.max_elements:
            raise ToolError(f"motion-graphics: more than {self.max_elements} elements")
        elements = [self.element_for(e, i, paths) for i, e in enumerate(raw_elements)]
        ids = [e["id"] for e in elements]
        if len(set(ids)) != len(ids):
            raise ToolError(f"motion-graphics: duplicate element id(s) {sorted({i for i in ids if ids.count(i) > 1})}")
        options: Dict[str, Any] = {}
        if "crf" in args:
            crf = args["crf"]
            if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= 51:
                raise ToolError("motion-graphics: crf must be an integer within [0, 51]")
            options["crf"] = crf
        if "preset" in args:
            if args["preset"] not in PRESETS:
                raise ToolError(f"motion-graphics: preset must be one of {PRESETS}")
            options["preset"] = str(args["preset"])
        req: Dict[str, Any] = {"schema": REQUEST_SCHEMA, "video": {"path": str(Path(src).resolve())}, "output": {"path": str(Path(out).resolve()), "overwrite": True}, "elements": elements}
        if options:
            req["options"] = options
        hit = scan_forbidden(req, tuple(f for f in self.forbidden if f not in ("path", "paths", "workspace")), "request")
        if hit:
            raise ToolError(f"motion-graphics: refusing to send a request carrying {hit}")
        return {"request": req, "output": out, "input": src, "element_ids": ids, "workspace": self.workspace or str(Path(out).parent)}

    @staticmethod
    def _under(path: str, root: str) -> bool:
        try:
            p, r = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
        except OSError:
            return False
        return p == r or p.startswith(r.rstrip(os.sep) + os.sep)

    def _argv(self, b: Dict[str, Any], dry_run: bool, timeout: Optional[float]) -> List[str]:
        argv = ["run", "-", "--json", "--workspace", b["workspace"]]
        roots = list(self.allowed_inputs) + ([self.workspace] if self.workspace else [])
        for r in roots:
            argv += ["--allowed-input", r]
        argv += self._engine_argv()
        argv += ["--timeout", str(int(timeout or self.default_timeout))]
        if dry_run:
            argv.append("--dry-run")
        return argv

    # ---- execution
    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        try:
            b = self.build_request(op.tool, op.args, paths, op_id=op.id, timeout=timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e), retryable=False)
        os.makedirs(b["workspace"], exist_ok=True)
        os.makedirs(os.path.dirname(b["output"]) or ".", exist_ok=True)
        argv = self._argv(b, dry_run, timeout)
        code, out, err = self._invoke(argv, stdin=json.dumps(b["request"], ensure_ascii=False), timeout=(timeout or self.default_timeout) + 5.0)
        tail = "\n".join(err.strip().splitlines()[-12:])
        secs = round(time.time() - t0, 3)
        if code == 124:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, 124, "CANCELLED", tail or "process timed out", retryable=True, details={"reason": "timeout"})
        try:
            doc = one_json_document(out, "motion-graphics")
        except ToolError as e:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 9, "INVALID_RESULT", str(e), retryable=False, details={"exit_code": code})
        if doc.get("ok") is not True:
            ed = as_dict(doc.get("error"))
            errc = str(ed.get("code") or "INVALID_RESULT")
            if errc not in self.retryable:
                errc, retry = "INVALID_RESULT", False
            else:
                retry = bool(ed["retryable"]) if isinstance(ed.get("retryable"), bool) else self.retryable.get(errc, False)
            details = scrub(ed.get("details"), self.forbidden)
            if errc == "CANCELLED" and (details.get("reason") or "") not in ("timeout", "signal"):
                details["reason"] = details.get("reason") or "signal"
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 1, errc, str(ed.get("message") or tail or "")[:500], retryable=bool(retry), details=details)
        errs = self._check_response(doc, b["output"], dry_run, b["element_ids"])
        if errs:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code if code != 0 else 9, "INVALID_RESULT", "; ".join(errs), retryable=False, details={"exit_code": code})
        if code != 0:
            remove_fresh(b["output"], t0)
            return self._fail(op, attempt, dry_run, t0, code, "INVALID_RESULT", f"exit code {code} with an ok response", retryable=False)
        data = self._success_data(doc, b["output"], dry_run)
        return ToolResult(op.id, op.tool, True, 0, None if dry_run else b["output"], data, list(data.get("commands") or []), tail, secs, attempt, dry_run)

    def _check_response(self, doc: Dict[str, Any], out_path: str, dry_run: bool, element_ids: List[str]) -> List[str]:
        errs: List[str] = []
        if doc.get("schema") != RESPONSE_SCHEMA:
            errs.append(f"response schema {doc.get('schema')!r}")
        sk = as_dict(doc.get("skill"))
        if sk.get("id") != SKILL_ID or str(sk.get("version")) != self.version:
            errs.append(f"response skill {sk!r} is not {SKILL_ID}@{self.version}")
        if doc.get("status") != "ok":
            errs.append(f"status {doc.get('status')!r} is not ok")
        if bool(doc.get("dry_run")) != bool(dry_run):
            errs.append("dry_run flag does not match the request")
        if dry_run:
            plan = as_dict(doc.get("plan"))
            if not plan.get("document_id") or not isinstance(plan.get("timeline"), list):
                errs.append("plan.document_id / plan.timeline missing")
            elif {str(as_dict(t).get("id")) for t in plan["timeline"]} != set(element_ids):
                errs.append("plan.timeline does not carry every element")
            if not same_file(str(as_dict(plan.get("output")).get("path") or ""), out_path):
                errs.append(f"plan.output.path {as_dict(plan.get('output')).get('path')!r} is not the requested {out_path}")
            return errs
        o = as_dict(doc.get("output"))
        if not o.get("path") or not same_file(str(o["path"]), out_path):
            errs.append(f"output path {o.get('path')!r} is not the requested {out_path}")
        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            return errs + ["output file missing or empty"]
        ok, actual = fingerprint_matches(o.get("sha256"), out_path)
        if not ok:
            errs.append(f"output sha256 {o.get('sha256')!r} != file {actual}")
        if o.get("size") != os.path.getsize(out_path):
            errs.append("output size differs from the file")
        for k in ("duration", "width", "height"):
            if o.get(k) in (None, ""):
                errs.append(f"output lacks probe fact {k}")
        tl = [as_dict(t) for t in doc.get("timeline") or [] if isinstance(t, dict)]
        if {str(t.get("id")) for t in tl} != set(element_ids):
            errs.append("timeline does not carry every element")
        ops = [as_dict(r) for r in doc.get("operations") or [] if isinstance(r, dict)]
        if not ops or any(r.get("status") not in ("rendered", "reused") or not str(r.get("tool", "")).startswith(ENGINE_ID + "/") for r in ops):
            errs.append("operations lack a rendered/reused record per element executed by an ffmpeg-skill tool")
        if not isinstance(as_dict(doc.get("engine")).get(ENGINE_ID), str):
            errs.append("engine lacks ffmpeg-skill")
        prov = as_dict(doc.get("provenance"))
        if prov.get("output_hash") != o.get("sha256") or not as_dict(prov.get("video")).get("sha256") or not prov.get("operations"):
            errs.append("provenance incomplete (output_hash / video.sha256 / operations)")
        return errs

    def _success_data(self, doc: Dict[str, Any], out_path: str, dry_run: bool) -> Dict[str, Any]:
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run" if dry_run else "completed", "operation_type": "GRAPHICS",
                                "warnings": list(doc.get("warnings") or [])}
        if dry_run:
            plan = as_dict(doc.get("plan"))
            data["plan"] = {"document_id": plan.get("document_id"), "video": plan.get("video"), "timeline": plan.get("timeline")}
            data["timeline"] = [{k: as_dict(t).get(k) for k in ("id", "type", "start", "end")} for t in plan.get("timeline") or []]
            data["commands"] = []
            return data
        o, prov = as_dict(doc.get("output")), as_dict(doc.get("provenance"))
        ops = [as_dict(r) for r in doc.get("operations") or [] if isinstance(r, dict)]
        data["commands"] = [str(c) for r in ops for c in (r.get("tool_commands_observed") or [])]
        data["artifact"] = {"path": out_path, "sha256": o.get("sha256"), "size": o.get("size"), "duration": o.get("duration"), "width": o.get("width"), "height": o.get("height"),
                            "reused": bool(doc.get("reused"))}
        data["timeline"] = [{k: as_dict(t).get(k) for k in ("id", "type", "start", "end")} for t in doc.get("timeline") or []]
        data["operations"] = [{k: r.get(k) for k in ("element_id", "type", "tool", "status", "operation_id", "parameters", "input_hashes", "output_hash")} for r in ops]
        data["engine"] = dict(as_dict(doc.get("engine")))
        engine_ver = str(data["engine"].get(ENGINE_ID) or "")
        data["observation"] = {"kind": "media.probe", "source": f"{ENGINE_ID}/probe@{engine_ver}", "provenance": "OBSERVED",
                               "data": {k: o.get(k) for k in ("duration", "width", "height", "size", "sha256")}}
        data["provenance"] = {"document_id": prov.get("document_id"), "video": prov.get("video"), "assets": prov.get("assets"), "fonts": prov.get("fonts"),
                              "operations": prov.get("operations"), "output_hash": prov.get("output_hash")}
        return data

    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, code: int, errc: str, message: str, retryable: bool, details: Optional[Dict[str, Any]] = None) -> ToolResult:
        return failed_result(op, SKILL_ID, getattr(self, "version", ""), attempt, dry_run, t0, code, errc, message, retryable, details)


def lift_observation(result: ToolResult, asset_id: Optional[str] = None) -> Optional[Observation]:
    """The Skill's OBSERVED probe of the delivered output as an agent Observation (provenance only; never fed back into analysis)."""
    obs = (result.data or {}).get("observation")
    if not isinstance(obs, dict) or obs.get("provenance") != "OBSERVED" or not isinstance(obs.get("data"), dict):
        return None
    sk = (result.data or {}).get("skill") or {}
    art = (result.data or {}).get("artifact") or {}
    prov = (result.data or {}).get("provenance") or {}
    return Observation(kind=str(obs.get("kind") or "media.probe"), asset_id=asset_id or result.op_id, source=str(obs.get("source")), data=dict(obs["data"]), analyzer=str(obs.get("source")),
                       provenance="OBSERVED", skill=SKILL_ID, skill_version=str(sk.get("version") or ""), tool=result.tool, fingerprint=str(art.get("sha256") or ""),
                       parameters={"output": result.output, "document_id": prov.get("document_id")})
