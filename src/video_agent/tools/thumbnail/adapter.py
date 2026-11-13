"""ThumbnailAdapter: the agent's boundary to thumbnail-skill (deterministic thumbnail rendering Skill, ADR-031).

    video-production-agent ─(typed Operation)─→ ThumbnailAdapter ─({"tool", "params"} on stdin)─→ `thumbnail run - --json`
        ─(Pillow raster ops; video frames through typed ffmpeg-skill probe/look calls)─→ PNG / JPEG

Protocol (from the Skill's own contract, `thumbnail skill --json`):
    contract → `thumbnail skill --json`
    doctor   → `thumbnail doctor --json [--workspace D] [--allowed-input R]… [--ffmpeg-skill X]`
    run      → `thumbnail run - --json --workspace D --allowed-input R… [--ffmpeg-skill X] [--timeout S]` (request on stdin)
    request  → {"tool": "thumbnail/render" | "thumbnail/extract_frame", "params": {...}}   (workspace / roots / timeout ride on argv, never in params)
    response ← one JSON document {"ok": true, "tool", "result": <thumbnail-skill/response@1>}; the outer "ok" only means "dispatched",
               the tool's own verdict is result.ok / result.status; a request the transport itself refused comes back as a bare
               response document {"ok": false, "error": {code, message, retryable, details}}

The agent hands one typed operation and this adapter lowers it deterministically:
    thumbnail/extract_frame: {"input": <video id>, "timestamp": s ≥ 0, "format": png|jpeg, "jpeg_quality"?: 1..100, "output": <id>}
        → params {source: {path, timestamp}, output: {path, format, overwrite: true, jpeg_quality?}}
    thumbnail/render: {"input", "timestamp", "format", "output", "width", "height" (canvas 16..7680), "background"?: "#RRGGBB", "text"?,
        "font_id"? (contract font_ids, default sans-bold), "font_size"? (6..400, default 48), "color"? (default #FFFFFF),
        "position"?: center|top|bottom (default bottom), "jpeg_quality"?}
        → a ThumbnailDocument: document_id from the op id, one video_frame asset "frame", one image element "frame" covering the canvas,
          and, when text is given, one text element "caption" anchored by `position` with the Skill's own align vocabulary.
Every value is checked against the ranges the contract declares (canvas, font_size, text length, font_ids, output formats and
extensions); unknown parameters and forbidden keys are refused by name. The response is verified: transport envelope and tool id,
schema / skill id / version / status, the output's realpath, sha256 recomputed from the file, size, format and dimensions, and the
provenance chain (skill, version, identity, output_hash, the source's sha256). The Skill reports no commands; `commands` is [].

ADR-104 (thumbnail-skill 0.1.12 .. 0.1.15): the same document, rendered by the tools the plan names --
    thumbnail/render_variants: render's arguments + {"variants": {"count" 2..12, "outputs": [ids of variants 2..N], "frames"?: [seconds],
        "phone_width"?, "min_text_height"?, "min_contrast"?}}; `output` is variant 1 (the document as given) and must be named <stem>-v1<ext>
        → params {document, output {path: <stem><ext>}, variants {count, frame_element "frame", frames?, phone_width?, min_text_height_px?,
          min_contrast_ratio?, qc_skill?}}; every variant file is checked (path <stem>-v<k>, sha256, size, format) and its phone-size verdict kept.
    thumbnail/render_layers: render's arguments + {"layers": {"output": <id of layers.json>}} → params {document, output, layers {directory,
        overwrite}}; every layer file and the manifest are read back (inside the directory, sha256, the Skill's pixel-equal verification).
    thumbnail/render_batch: render's caption / texts + {"batch": {"csv", "csv_sha256", "encoding", "delimiter", "name", "min_font_size"?,
        "targets": [{preset, width, height, logo?}], "outputs": [{row, preset, output: id}]}} → params {csv {path, encoding, delimiter},
        targets [{preset, document, text_fit?}], output {directory, name, format, overwrite}}; every (row, preset) file is the one planned.
    "avoid_faces": {"faces": [[x, y, w, h]], "frame_size": {width, height}, "margin"?} on a document render: thumbnail/place_text is called
        first with the same document, and only the position / alignment it chose is copied onto the caption (the rest of the document
        stays the agent's own); the placement is reported.
    thumbnail/choose_frame (a measurement, `measure()`): {"input": <video>, "detections_path", "detections_sha256"} → the ranking of the
        detection-skill face document's sampled frames, admitted when it is about that video (fingerprint) and that file (sha256).

Identity discrepancy (documented, not patched): the Skill calls itself "thumbnail-skill" (contract skill_id / response skill.id /
provenance.skill) while its tool ids are "thumbnail/<name>". The agent package id is therefore SKILL_ID = "thumbnail" (the tool
prefix) and CONTRACT_SKILL_ID = "thumbnail-skill" is what the Skill's own documents are checked against. The contract also declares
`thumbnail/validate` (structural validation, no output) and `thumbnail/generate_background` (optional AI background); the agent
uses neither (NOT_USED). Facts, never "fixed": extract_frame delivers the
frame at ffmpeg-skill/look's default width (1280 px) regardless of the source; the Skill has no dry-run mode (a dry run here lowers
and validates the request but never invokes the Skill)."""
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
from ..skill_process import (ENGINE_CONTRACT_VERSIONS, FORBIDDEN_ARG_KEYS, CliSkill, ContractError, as_dict, breaking_drift, drift_report, error_table, failed_result, fingerprint_matches, invoke, one_json_document,
                             remove_fresh, same_file, scan_forbidden, scrub, strip_sha_prefix)
from .locate import locate_thumbnail

SKILL_ID = "thumbnail"
PREFIX = SKILL_ID + "/"
CONTRACT_SKILL_ID = "thumbnail-skill"
TOOL_RENDER = "thumbnail/render"
TOOL_EXTRACT_FRAME = "thumbnail/extract_frame"
# ADR-104: the tools of 0.1.12 .. 0.1.15, wired into the plan
TOOL_RENDER_BATCH = "thumbnail/render_batch"
TOOL_RENDER_VARIANTS = "thumbnail/render_variants"
TOOL_RENDER_LAYERS = "thumbnail/render_layers"
TOOL_CHOOSE_FRAME = "thumbnail/choose_frame"
TOOL_PLACE_TEXT = "thumbnail/place_text"
EXECUTION_TOOLS = (TOOL_RENDER, TOOL_EXTRACT_FRAME, TOOL_RENDER_BATCH, TOOL_RENDER_VARIANTS, TOOL_RENDER_LAYERS)
DOCUMENT_TOOLS = (TOOL_RENDER, TOOL_RENDER_VARIANTS, TOOL_RENDER_LAYERS)   # one ThumbnailDocument (+ its frame / texts / logo)
PURE_TOOLS = {TOOL_CHOOSE_FRAME: "ranking", TOOL_PLACE_TEXT: "layout"}       # arithmetic over measured facts: no file written
TOOLS = EXECUTION_TOOLS + tuple(PURE_TOOLS)
NOT_USED = ("thumbnail/generate_background", "thumbnail/validate")       # every other tool of the pinned contract (a test keeps this list exact)
LAYERS_MANIFEST = "layers.json"
LAYERS_SCHEMA = "thumbnail-skill/layers@1"
DETECTIONS_SCHEMA = "detection-skill/detections@1"
MAX_VARIANTS = 12
BATCH_DELIMITERS = {",": ",", ";": ";", "tab": "\t"}
BATCH_ENCODINGS = ("utf-8", "cp932")
CONTRACT_SCHEMA = "thumbnail-skill/contract@1"
REQUEST_SCHEMA = "thumbnail-skill/request@1"
RESPONSE_SCHEMA = "thumbnail-skill/response@1"
DOCTOR_SCHEMA = "thumbnail-skill/doctor@1"
SUPPORTED_SKILL_VERSIONS = ("0.1.",)
# 0.1.8 (CI changes only): version-only; the pin is re-taken from main (ADR-083).
ENGINE_ID = "ffmpeg-skill"
CANONICAL_INVOCATION = ["thumbnail", "run", "-", "--json"]
REQUIRED_EXECUTION_FLAGS = {"shell": False, "arbitrary_executables": False, "arbitrary_filters": False, "network": False, "input_mutation": False, "ai": False}
COMMON_ARGS = ("input", "output", "format", "timestamp", "jpeg_quality")
RENDER_ARGS = ("width", "height", "background", "text", "font_id", "font_size", "color", "position", "texts", "logo",   # ADR-088: more texts, a logo
               "avoid_faces")   # ADR-104: thumbnail/place_text before the render
TOOL_ARGS = {TOOL_EXTRACT_FRAME: COMMON_ARGS, TOOL_RENDER: COMMON_ARGS + RENDER_ARGS, TOOL_RENDER_VARIANTS: COMMON_ARGS + RENDER_ARGS + ("variants",),
             TOOL_RENDER_LAYERS: COMMON_ARGS + RENDER_ARGS + ("layers",),
             TOOL_RENDER_BATCH: ("input", "format", "timestamp", "jpeg_quality", "background", "text", "font_id", "font_size", "color", "position", "texts", "batch")}
MAX_TEXTS = 8                  # an agent-side bound on extra text elements (the agent's vocabulary asks for at most 3)
LOGO_EXTENSIONS = (".png", ".jpg", ".jpeg")
POSITIONS = ("center", "top", "bottom")
DEFAULTS = {"font_id": "sans-bold", "font_size": 48, "color": "#FFFFFF", "position": "bottom", "background": "#000000"}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_RANGE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\.\.(-?\d+(?:\.\d+)?)")
DRIFT_KEYS = ("schema", "skill_id", "version", "kind", "execution", "document", "output_formats", "fonts", "rendering", "ffmpeg_skill", "schema_versions", "errors",
              "provenance", "deterministic",
              "platform_presets", "batch", "frame_choice", "variants", "layered_output")   # ADR-104: the blocks the new tools' requests are built from
DRIFT_TOOL_KEYS = ("role", "produces_output", "deterministic", "writes_media", "mutates_input", "input", "output", "delegates_to")
# 0.1.9 (Dependabot CI bump), 0.1.10 / 0.1.11 (roadmap docs PRs, thumbnail-skill #24): version-only; re-pinned from the v0.1.11 tag (ADR-097).
# 0.1.12 .. 0.1.15 (thumbnail-skill #25 .. #28): new tools render_batch / choose_frame / place_text / render_variants / render_layers and their
# blocks; validate / extract_frame / render / generate_background unchanged. Re-pinned from the v0.1.15 tag (ADR-097 addendum 6). 0.1.16 (ADR-107): version-only, re-pinned from the tag.
# 0.1.17 (docs only) and 0.1.18 (thumbnail-skill #31; ADR-107 addendum 28): a top-level `kit` block (render --kit, options.kit); otherwise unchanged.
# The agent sends no kit. Re-pinned from the v0.1.18 tag.
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.18.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def _range(text: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    """"16..7680" (possibly with a trailing note) → (16, 7680); the pinned default when the contract phrases it differently."""
    m = _RANGE_RE.search(str(text or ""))
    return (float(m.group(1)), float(m.group(2))) if m else default


def check_contract(contract: Any) -> List[str]:
    """Compatibility checks: schema, id, version range, kind, execution flags, canonical invocation, the two execution tools, the
    document vocabulary (schema, id pattern, forbidden fields, element types, fonts), output formats, error codes. Anything off is
    refused, never patched."""
    errs: List[str] = []
    if not isinstance(contract, dict):
        return ["contract is not an object"]
    if contract.get("schema") != CONTRACT_SCHEMA:
        errs.append(f"contract schema {contract.get('schema')!r} != {CONTRACT_SCHEMA}")
    if contract.get("skill_id") != CONTRACT_SKILL_ID or contract.get("id") != CONTRACT_SKILL_ID:
        errs.append(f"skill_id {contract.get('skill_id')!r} / id {contract.get('id')!r} != {CONTRACT_SKILL_ID}")
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
    fs = contract.get("ffmpeg_skill") or {}
    if str(fs.get("contract_version")) not in ENGINE_CONTRACT_VERSIONS or not {"probe", "look"} <= {str(t) for t in fs.get("tools_used") or []}:
        errs.append(f"ffmpeg_skill.contract_version {ENGINE_CONTRACT_VERSIONS} / tools_used probe+look missing")
    tools = {str(t.get("tool_id")): t for t in contract.get("tools") or [] if isinstance(t, dict)}
    for tid in EXECUTION_TOOLS:
        t = tools.get(tid)
        if t is None:
            errs.append(f"contract lacks tool {tid} (declares {sorted(tools)})")
            continue
        if t.get("role") != "execution" or t.get("produces_output") is not True or t.get("deterministic") is not True or t.get("mutates_input") is not False:
            errs.append(f"tool {tid} must declare role execution, produces_output / deterministic = true, mutates_input = false")
        if not {"ffmpeg-skill/probe", "ffmpeg-skill/look"} <= {str(d) for d in t.get("delegates_to") or []}:
            errs.append(f"tool {tid} must delegate video frames to ffmpeg-skill/probe and ffmpeg-skill/look")
    for tid, role in PURE_TOOLS.items():   # ADR-104: ranking / layout read facts and write nothing
        t = tools.get(tid)
        if t is None:
            errs.append(f"contract lacks tool {tid} (declares {sorted(tools)})")
        elif t.get("role") != role or t.get("produces_output") is not False or t.get("writes_media") is not False or t.get("mutates_input") is not False \
                or t.get("deterministic") is not True:
            errs.append(f"tool {tid} must declare role {role}, produces_output / writes_media / mutates_input = false, deterministic = true")
    lo = as_dict(contract.get("layered_output"))
    if as_dict(lo.get("manifest")).get("file") != LAYERS_MANIFEST or as_dict(lo.get("manifest")).get("schema") != LAYERS_SCHEMA:
        errs.append(f"layered_output.manifest must be {LAYERS_MANIFEST} ({LAYERS_SCHEMA})")
    presets = as_dict(as_dict(contract.get("platform_presets")).get("presets"))
    if not presets or not all(isinstance(as_dict(v).get("width"), int) and isinstance(as_dict(v).get("height"), int) and as_dict(v).get("formats") for v in presets.values()):
        errs.append("platform_presets.presets must give each preset's width, height and formats")
    if as_dict(as_dict(contract.get("variants"))).get("max_variants") != MAX_VARIANTS:
        errs.append(f"variants.max_variants {as_dict(contract.get('variants')).get('max_variants')!r} != {MAX_VARIANTS}")
    for tid, t in tools.items():
        if not tid.startswith(PREFIX):
            errs.append(f"tool id {tid!r} does not carry the {PREFIX} prefix")
    doc = contract.get("document") or {}
    if doc.get("schema") != REQUEST_SCHEMA:
        errs.append(f"document.schema {doc.get('schema')!r} != {REQUEST_SCHEMA}")
    if doc.get("id_pattern") != _ID_RE.pattern:
        errs.append(f"document.id_pattern {doc.get('id_pattern')!r} != {_ID_RE.pattern}")
    forbidden = set(doc.get("forbidden_fields") or []) & set(ex.get("forbidden_request_fields") or [])
    for k in ("command", "argv", "filter", "shell", "exec", "env"):
        if k not in forbidden:
            errs.append(f"forbidden fields lack {k!r}")
    els = as_dict(doc.get("elements"))
    if sorted(str(x) for x in els.get("types") or []) != ["image", "text"]:
        errs.append(f"document.elements.types {els.get('types')!r} != ['image', 'text']")
    align = as_dict(as_dict(as_dict(els.get("text")).get("fields")).get("align"))
    if "center" not in (align.get("horizontal") or []) or not {"top", "middle", "bottom"} <= set(align.get("vertical") or []):
        errs.append("text align vocabulary lacks center / top / middle / bottom")
    if not {"image", "video_frame"} <= {str(k) for k in as_dict(doc.get("assets")).get("kinds") or []}:
        errs.append("document.assets.kinds lacks image / video_frame")
    fonts = [str(f) for f in as_dict(contract.get("fonts")).get("font_ids") or []]
    if not fonts or DEFAULTS["font_id"] not in fonts:
        errs.append(f"fonts.font_ids {fonts!r} lacks the default {DEFAULTS['font_id']!r}")
    fmts = contract.get("output_formats") or {}
    if not isinstance(fmts, dict) or not {"png", "jpeg"} <= set(fmts) or not all(isinstance(as_dict(v).get("extensions"), list) and v["extensions"] for v in fmts.values()):
        errs.append("output_formats must declare png and jpeg with extensions")
    if as_dict(contract.get("rendering")).get("engine") != "Pillow":
        errs.append(f"rendering.engine {as_dict(contract.get('rendering')).get('engine')!r} != Pillow")
    retry, exit_codes = error_table(contract)
    for c in ("INVALID_REQUEST", "INVALID_INPUT", "PATH_NOT_ALLOWED", "UNSUPPORTED_OPERATION", "UNSUPPORTED_FORMAT", "MISSING_INPUT", "INVALID_TIME_RANGE", "TOOL_ERROR",
              "OUTPUT_ERROR", "VALIDATION_ERROR", "CANCELLED", "INTERNAL_ERROR"):
        if c not in retry:
            errs.append(f"errors.codes lacks {c}")
    if not exit_codes:
        errs.append("errors.exit_codes missing")
    return errs


def contract_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(live, pinned or pinned_contract(), DRIFT_KEYS, "tools", "tool_id", DRIFT_TOOL_KEYS)


# ADR-048: `version` moves within the range check_contract() accepts and `ffmpeg_skill` is the Skill's own
# engine-compatibility business (capability detection, gated by its doctor); everything else stays fatal.
FATAL_DRIFT_KEYS = tuple(k for k in DRIFT_KEYS if k not in ("version", "ffmpeg_skill"))


def contract_breaking_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return breaking_drift(live, pinned or pinned_contract(), FATAL_DRIFT_KEYS, "tools", "tool_id", DRIFT_TOOL_KEYS)


def package_from_contract(contract: Dict[str, Any]) -> SkillPackage:
    ver = str(contract.get("version") or "")
    descs = {str(t.get("tool_id")): str(t.get("description", "")) for t in contract.get("tools") or [] if isinstance(t, dict)}
    tools = [ToolSpec(tool_id=tid, skill_id=SKILL_ID, version=ver, description=descs.get(tid, ""), required_capabilities=[SKILL_ID, "ffmpeg", "ffprobe", ENGINE_ID],
                      inputs=["input", "output"], produces_output=True, deterministic=True,
                      result_keys=["operation_type", "artifact", "observation", "provenance", "source", "commands"]) for tid in EXECUTION_TOOLS]
    tools += [ToolSpec(tool_id=tid, skill_id=SKILL_ID, version=ver, description=descs.get(tid, ""), required_capabilities=[SKILL_ID], inputs=["input"],
                       produces_output=False, deterministic=True, result_keys=["ranking" if tid == TOOL_CHOOSE_FRAME else "placement", "provenance", "commands"])
              for tid in PURE_TOOLS]
    return SkillPackage(skill_id=SKILL_ID, name=str(contract.get("name") or CONTRACT_SKILL_ID), version=ver, description=str(contract.get("description", ""))[:200],
                        capabilities=[SKILL_ID], tools=tools, repository="kajisho5/thumbnail-skill",
                        role="deterministic thumbnail rendering (Pillow; video frames through ffmpeg-skill/look)")


PACKAGE = package_from_contract(pinned_contract())


class ThumbnailAdapter(ToolAdapter):
    name = SKILL_ID

    def __init__(self, skill: Optional[CliSkill] = None, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None,
                 ffmpeg_skill_dir: Optional[str] = None, timeout: float = 120.0, path_policy: Optional[PathPolicy] = None, qc_skill_dir: Optional[str] = None):
        located = skill or locate_thumbnail()
        if not located:
            raise ToolError("thumbnail-skill not found (set VIDEO_AGENT_THUMBNAIL_DIR or install `thumbnail`)")
        self.skill: CliSkill = located
        self.workspace = str(Path(workspace).resolve()) if workspace else None
        self.allowed_inputs = [str(Path(r).resolve()) for r in (allowed_inputs or [])]
        self.ffmpeg_skill_dir = ffmpeg_skill_dir
        self.qc_skill_dir = str(Path(qc_skill_dir).resolve()) if qc_skill_dir else None   # ADR-104: render_variants measures contrast through qc-skill
        self.default_timeout = float(timeout)
        self.path_policy = path_policy
        self.calls = 0
        self.contract = self._fetch_contract()   # an install without Pillow cannot even print its contract: MISSING, never half-used
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("thumbnail contract incompatible: " + "; ".join(errs))
        self.version = str(self.contract["version"])
        self.tools = set(TOOLS)
        doc = self.contract["document"]
        canvas = as_dict(doc.get("canvas"))
        self.canvas_range = {"width": _range(canvas.get("width"), (16, 7680)), "height": _range(canvas.get("height"), (16, 7680))}
        tf = as_dict(as_dict(as_dict(doc.get("elements")).get("text")).get("fields"))
        self.font_size_range = _range(tf.get("font_size"), (6, 400))
        m = re.search(r"max (\d+) chars", str(tf.get("text") or ""))
        self.text_max = int(m.group(1)) if m else 2000
        self.font_ids = sorted(str(f) for f in self.contract["fonts"]["font_ids"])
        self.formats: Dict[str, List[str]] = {str(k): [str(e).lower() for e in v["extensions"]] for k, v in self.contract["output_formats"].items()}
        self.forbidden = tuple(sorted(set(FORBIDDEN_ARG_KEYS) | {str(f) for f in doc.get("forbidden_fields") or []} | {str(f) for f in self.contract["execution"].get("forbidden_request_fields") or []}))
        self.retryable, self.exit_codes = error_table(self.contract)
        self.presets = {str(k): as_dict(v) for k, v in as_dict(as_dict(self.contract.get("platform_presets")).get("presets")).items()}
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
            raise ContractError(f"thumbnail skill --json failed (exit {code}): {err.strip()[-300:]}")
        try:
            return one_json_document(out, "thumbnail contract")
        except ToolError as e:
            raise ContractError(str(e))

    def doctor(self) -> Dict[str, Any]:
        argv = ["doctor", "--json"] + (["--workspace", self.workspace] if self.workspace else []) + [x for r in self.allowed_inputs for x in ("--allowed-input", r)] + self._engine_argv()
        code, out, err = self._invoke(argv, timeout=180.0)
        try:
            doc = one_json_document(out, "thumbnail doctor")
        except ToolError as e:
            return {"schema": DOCTOR_SCHEMA, "status": "fail", "problems": [f"doctor produced no document: {e}"], "checks": {}, "exit_code": code}
        if doc.get("schema") != DOCTOR_SCHEMA or doc.get("status") not in ("ok", "degraded", "fail"):
            return {"schema": DOCTOR_SCHEMA, "status": "fail", "problems": [f"unexpected doctor document {doc.get('schema')!r} / {doc.get('status')!r}"], "checks": doc.get("checks") or {}, "exit_code": code}
        doc["exit_code"] = code
        return doc

    @staticmethod
    def font_status(doc: Dict[str, Any]) -> Dict[str, str]:
        """Per font_id: available | missing | …, as the Skill's doctor reports it (never inferred here)."""
        fonts = as_dict(as_dict(doc.get("checks")).get("fonts"))
        return {str(f): str(as_dict(v).get("status") or "unknown") for f, v in fonts.items()}

    @staticmethod
    def engine_status(doc: Dict[str, Any]) -> str:
        """The ffmpeg-skill check status (ok | fail | missing | unknown): without it video_frame assets fail TOOL_ERROR."""
        return str(as_dict(as_dict(doc.get("checks")).get("ffmpeg_skill")).get("status") or "unknown")

    def drift(self) -> List[str]:
        if self._drift is None:
            self._drift = contract_drift(self.contract)
        return self._drift

    def breaking_drift(self) -> List[str]:
        """ADR-048: the subset of drift() that gates availability (see FATAL_DRIFT_KEYS)."""
        return contract_breaking_drift(self.contract)

    # ---- ToolAdapter
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "root": self.skill.describe(), "tools": sorted(self.tools), "formats": sorted(self.formats), "fonts": self.font_ids,
                "contract": self.contract.get("schema"), "drift": self.drift()}

    def package(self) -> SkillPackage:
        return package_from_contract(self.contract)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            b = self.build_request(op.tool, op.args, paths, op_id=op.id)
        except ToolError as e:
            return [f"thumbnail: refused: {e}"]
        return [" ".join(["thumbnail"] + self._argv(b, timeout=None)) + "  <<< " + json.dumps(b["request"], ensure_ascii=False)[:400]]


    # ---- lowering: typed args → one {"tool", "params"} request
    @staticmethod
    def _number(name: str, v: Any, lo: float, hi: float, integer: bool = False) -> Any:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
            raise ToolError(f"thumbnail: {name} must be a finite number")
        x: Any = float(v)
        if integer:
            if x != int(x):
                raise ToolError(f"thumbnail: {name} must be an integer")
            x = int(x)
        if x < lo or x > hi:
            raise ToolError(f"thumbnail: {name} {x} is outside the contract range {lo:g}..{hi:g}")
        return x

    @staticmethod
    def _color(name: str, v: Any) -> str:
        if not isinstance(v, str) or not _COLOR_RE.match(v):
            raise ToolError(f"thumbnail: {name} must be a '#RRGGBB' colour")
        return v

    def _output_spec(self, out: str, fmt: str, args: Dict[str, Any]) -> Dict[str, Any]:
        spec: Dict[str, Any] = {"path": str(Path(out).resolve()), "format": fmt, "overwrite": True}
        if "jpeg_quality" in args:
            if fmt != "jpeg":
                raise ToolError("thumbnail: jpeg_quality is only accepted for format jpeg")
            spec["jpeg_quality"] = self._number("jpeg_quality", args["jpeg_quality"], 1, 100, integer=True)
        return spec

    def _document(self, pid: str, src: str, ts: float, args: Dict[str, Any]) -> Dict[str, Any]:
        for k in ("width", "height"):
            if k not in args:
                raise ToolError(f"thumbnail: render requires {k}")
        w = self._number("width", args["width"], *self.canvas_range["width"], integer=True)
        h = self._number("height", args["height"], *self.canvas_range["height"], integer=True)
        canvas = {"width": w, "height": h, "background": self._color("background", args.get("background", DEFAULTS["background"]))}
        elements: List[Dict[str, Any]] = [{"element_id": "frame", "type": "image", "z_index": 0,
                                           "image": {"asset_id": "frame", "position": {"x": 0, "y": 0}, "size": {"width": w, "height": h}, "fit": "cover"}}]
        if "text" in args:
            text = args["text"]
            if not isinstance(text, str) or not text or len(text) > self.text_max or any(ord(c) < 32 and c != "\n" for c in text):
                raise ToolError(f"thumbnail: text must be a non-empty string of at most {self.text_max} characters (explicit newlines only)")
            font_id = args.get("font_id", DEFAULTS["font_id"])
            if font_id not in self.font_ids:
                raise ToolError(f"thumbnail: font_id {font_id!r} is not one of the contract font_ids {self.font_ids}")
            size = self._number("font_size", args.get("font_size", DEFAULTS["font_size"]), *self.font_size_range, integer=True)
            color = self._color("color", args.get("color", DEFAULTS["color"]))
            position = args.get("position", DEFAULTS["position"])
            if position not in POSITIONS:
                raise ToolError(f"thumbnail: position {position!r} is not one of {list(POSITIONS)}")
            anchor = {"center": ({"x": w / 2, "y": h / 2}, "middle"), "top": ({"x": w / 2, "y": size}, "top"), "bottom": ({"x": w / 2, "y": h - size}, "bottom")}[position]
            elements.append({"element_id": "caption", "type": "text", "z_index": 1,
                             "text": {"text": text, "font_id": font_id, "font_size": size, "color": color, "position": anchor[0], "align": {"horizontal": "center", "vertical": anchor[1]}}})
        elif not (args.get("texts") or args.get("logo")):
            for k in ("font_id", "font_size", "color", "position"):
                if k in args:
                    raise ToolError(f"thumbnail: {k} needs a text")
        assets: List[Dict[str, Any]] = [{"asset_id": "frame", "kind": "video_frame", "path": str(Path(src).resolve()), "timestamp": ts}]
        # ADR-088: the full ThumbnailDocument -- further text elements (each its own position / size / colour) and a logo image asset
        texts = args.get("texts") or []
        if not isinstance(texts, list) or len(texts) > MAX_TEXTS:
            raise ToolError(f"thumbnail: texts must be a list of at most {MAX_TEXTS} text elements")
        for n, t in enumerate(texts, start=2):
            if not isinstance(t, dict) or sorted(t) != sorted(k for k in ("text", "font_size", "color", "position") if k in t) or "text" not in t:
                raise ToolError("thumbnail: each of texts is {text, font_size?, color?, position?}")
            text = t["text"]
            if not isinstance(text, str) or not text or len(text) > self.text_max or any(ord(c) < 32 and c != "\n" for c in text):
                raise ToolError(f"thumbnail: text {n} must be a non-empty string of at most {self.text_max} characters (explicit newlines only)")
            size = self._number(f"font_size of text {n}", t.get("font_size", DEFAULTS["font_size"]), *self.font_size_range, integer=True)
            position = t.get("position", DEFAULTS["position"])
            if position not in POSITIONS:
                raise ToolError(f"thumbnail: position {position!r} of text {n} is not one of {list(POSITIONS)}")
            anchor = {"center": ({"x": w / 2, "y": h / 2}, "middle"), "top": ({"x": w / 2, "y": size}, "top"), "bottom": ({"x": w / 2, "y": h - size}, "bottom")}[position]
            elements.append({"element_id": f"text{n}", "type": "text", "z_index": n,
                             "text": {"text": text, "font_id": DEFAULTS["font_id"], "font_size": size, "color": self._color(f"color of text {n}", t.get("color", DEFAULTS["color"])),
                                      "position": anchor[0], "align": {"horizontal": "center", "vertical": anchor[1]}}})
        logo = args.get("logo")
        if logo is not None:
            if not isinstance(logo, dict) or set(logo) - {"image", "x", "y", "width", "height", "opacity"} or not {"image", "x", "y", "width", "height"} <= set(logo):
                raise ToolError("thumbnail: logo is {image, x, y, width, height, opacity?}")
            img = logo["image"]
            if not isinstance(img, str) or Path(img).suffix.lower() not in LOGO_EXTENSIONS or not os.path.isfile(img):
                raise ToolError(f"thumbnail: logo image must be an existing PNG / JPEG file: {Path(str(img)).name}")
            if self.path_policy is not None:
                self.path_policy.check_input(img)
            elif self.allowed_inputs and not any(self._under(img, r) for r in self.allowed_inputs + ([self.workspace] if self.workspace else [])):
                raise ToolError(f"thumbnail: logo image outside the allowed roots: {img}")
            lw = self._number("logo width", logo["width"], 1, 7680, integer=True)
            lh = self._number("logo height", logo["height"], 1, 7680, integer=True)
            lx = self._number("logo x", logo["x"], -7680, 7680, integer=True)
            ly = self._number("logo y", logo["y"], -7680, 7680, integer=True)
            assets.append({"asset_id": "logo", "kind": "image", "path": str(Path(img).resolve())})
            elements.append({"element_id": "logo", "type": "image", "z_index": 100,
                             "image": {"asset_id": "logo", "position": {"x": lx, "y": ly}, "size": {"width": lw, "height": lh}, "fit": "contain",
                                       "opacity": self._number("logo opacity", logo.get("opacity", 1.0), 0.0, 1.0)}})
        return {"document_id": pid, "canvas": canvas, "assets": assets, "elements": elements}

    def _check_input(self, path: str, what: str) -> None:
        if self.path_policy is not None:
            self.path_policy.check_input(path)
        elif self.allowed_inputs and not any(self._under(path, r) for r in self.allowed_inputs + ([self.workspace] if self.workspace else [])):
            raise ToolError(f"thumbnail: {what} outside the allowed roots: {path}")

    def _check_output(self, out: str, src: str, fmt: str) -> None:
        if self.path_policy is not None:
            self.path_policy.check_output(out, [src])
        if self.workspace and not self._under(out, self.workspace):
            raise ToolError(f"thumbnail: output outside the workspace: {out}")
        if same_file(src, out):
            raise ToolError("thumbnail: output would overwrite its input")
        if Path(out).suffix.lower() not in self.formats[fmt]:
            raise ToolError(f"thumbnail: output extension {Path(out).suffix!r} does not match format {fmt} ({self.formats[fmt]})")

    def build_request(self, tool: str, args: Dict[str, Any], paths: Dict[str, str], op_id: str = "op", timeout: Optional[float] = None) -> Dict[str, Any]:
        if tool not in self.tools:
            raise ToolError(f"thumbnail: unsupported tool {tool}")
        if tool in PURE_TOOLS:
            raise ToolError(f"thumbnail: {tool} is not a plan operation (choose_frame is a measurement, place_text runs inside a render)")
        hit = scan_forbidden(args, self.forbidden)
        if hit:
            raise ToolError(f"thumbnail: forbidden field {hit} in the operation arguments")
        allowed = TOOL_ARGS[tool]
        unknown = sorted(k for k in args if k not in allowed)
        if unknown:
            raise ToolError(f"thumbnail: parameter(s) {unknown} are not declared for {tool} (accepted: {list(allowed)})")
        src_id = str(args.get("input") or "")
        if not src_id or (tool != TOOL_RENDER_BATCH and not args.get("output")):
            raise ToolError("thumbnail: input and output references are required")
        src = paths.get(src_id, src_id)
        if not os.path.isfile(src):
            raise ToolError(f"thumbnail: input not found: {src}")
        self._check_input(src, "input")
        fmt = str(args.get("format") or "")
        if fmt not in self.formats:
            raise ToolError(f"thumbnail: output format {fmt!r} is not one of {sorted(self.formats)}")
        if "timestamp" not in args:
            raise ToolError("thumbnail: timestamp is required")
        ts = float(self._number("timestamp", args["timestamp"], 0, 7 * 24 * 3600))
        pid = re.sub(r"[^A-Za-z0-9._-]", "_", str(op_id))[:64] or "op"
        if not _ID_RE.match(pid):
            pid = "op"
        if tool == TOOL_RENDER_BATCH:
            return self._batch_request(args, paths, src, fmt, ts, pid)
        out_id = str(args.get("output"))
        out = paths.get(out_id, out_id)
        self._check_output(out, src, fmt)
        output = self._output_spec(out, fmt, args)
        b: Dict[str, Any] = {"output": out, "input": src, "tool": tool, "format": fmt, "workspace": self.workspace or str(Path(out).parent), "outputs": [out]}
        if tool == TOOL_EXTRACT_FRAME:
            params: Dict[str, Any] = {"source": {"path": str(Path(src).resolve()), "timestamp": ts}, "output": output}
        else:
            doc = self._document(pid, src, ts, args)
            params = {"schema": REQUEST_SCHEMA, "document": doc, "output": output}
            if tool == TOOL_RENDER_VARIANTS:
                params["variants"] = self._variants(args, out, paths, src, fmt, b)
                params["output"] = dict(output, path=b["variant_stem"])
            elif tool == TOOL_RENDER_LAYERS:
                params["layers"] = self._layers(args, out, paths, src, b)
            if "avoid_faces" in args:
                b["place"] = {"tool": TOOL_PLACE_TEXT, "params": self._place_params(args["avoid_faces"], doc)}
        req = {"tool": tool, "params": params}
        hit = scan_forbidden(req, tuple(f for f in self.forbidden if f not in ("path", "paths")), "request")
        if hit:
            raise ToolError(f"thumbnail: refusing to send a request carrying {hit}")
        b["request"] = req
        return b

    # ---- ADR-104: the arguments of the new tools
    def _place_params(self, af: Any, doc: Dict[str, Any]) -> Dict[str, Any]:
        """thumbnail/place_text for the caption over the frame element: the face boxes (in frame_size pixels) and the margin, as planned."""
        if not isinstance(af, dict) or set(af) - {"faces", "frame_size", "margin"} or not {"faces", "frame_size"} <= set(af):
            raise ToolError("thumbnail: avoid_faces is {faces: [[x, y, w, h]], frame_size: {width, height}, margin?}")
        if not any(e.get("element_id") == "caption" for e in doc["elements"]):
            raise ToolError("thumbnail: avoid_faces moves the caption: it needs a text")
        faces = af["faces"]
        if not isinstance(faces, list) or len(faces) > 64:
            raise ToolError("thumbnail: avoid_faces.faces must be a list of at most 64 boxes")
        boxes = []
        for n, f in enumerate(faces):
            if not isinstance(f, list) or len(f) != 4:
                raise ToolError(f"thumbnail: avoid_faces.faces[{n}] must be [x, y, w, h]")
            x, y = (self._number(f"face {n} x", v, -100000, 100000) for v in f[:2])
            w, h = (self._number(f"face {n} size", v, 0.001, 100000) for v in f[2:])
            boxes.append([x, y, w, h])
        fs = af["frame_size"]
        if not isinstance(fs, dict) or set(fs) != {"width", "height"}:
            raise ToolError("thumbnail: avoid_faces.frame_size is {width, height}")
        params: Dict[str, Any] = {"document": json.loads(json.dumps(doc)), "text_element": "caption", "image_element": "frame", "faces": boxes,
                                  "frame_size": {k: self._number(f"frame_size.{k}", fs[k], 1, 100000, integer=True) for k in ("width", "height")}}
        if "margin" in af:
            params["margin"] = self._number("avoid_faces.margin", af["margin"], 0, 1000, integer=True)
        return params

    def _variants(self, args: Dict[str, Any], out: str, paths: Dict[str, str], src: str, fmt: str, b: Dict[str, Any]) -> Dict[str, Any]:
        v = args.get("variants")
        if not isinstance(v, dict) or set(v) - {"count", "outputs", "frames", "phone_width", "min_text_height", "min_contrast"} or not {"count", "outputs"} <= set(v):
            raise ToolError("thumbnail: variants is {count, outputs, frames?, phone_width?, min_text_height?, min_contrast?}")
        if "text" not in args:
            raise ToolError("thumbnail: variants vary the caption: they need a text")
        n = self._number("variants.count", v["count"], 2, MAX_VARIANTS, integer=True)
        ext = Path(out).suffix
        if not Path(out).stem.endswith("-v1"):
            raise ToolError(f"thumbnail: variant 1 must be written as <stem>-v1{ext} (the Skill's own naming), got {Path(out).name}")
        stem = str(Path(out).with_name(Path(out).stem[:-3] + ext))
        ids = v["outputs"]
        if not isinstance(ids, list) or len(ids) != n - 1:
            raise ToolError(f"thumbnail: variants.outputs must name variants 2..{n}")
        for k, oid in enumerate(ids, start=2):
            p = paths.get(str(oid), str(oid))
            if os.path.realpath(p) != os.path.realpath(str(Path(stem).with_name(f"{Path(stem).stem}-v{k}{ext}"))):
                raise ToolError(f"thumbnail: variant {k} must be written as {Path(stem).stem}-v{k}{ext}, the plan names {Path(p).name}")
            self._check_output(p, src, fmt)
            b["outputs"].append(p)
        b["variant_stem"] = str(Path(stem).resolve())
        b["variant_count"] = n
        params: Dict[str, Any] = {"count": n}
        if v.get("frames"):
            frames = v["frames"]
            if not isinstance(frames, list) or len(frames) > MAX_VARIANTS - 1:
                raise ToolError(f"thumbnail: variants.frames must be a list of at most {MAX_VARIANTS - 1} timestamps")
            params["frame_element"] = "frame"
            params["frames"] = [{"name": f"at-{self._number('variants.frames', t, 0, 7 * 24 * 3600):g}s", "asset": {"kind": "video_frame", "path": str(Path(src).resolve()), "timestamp": float(t)}}
                                for t in frames]
        for k, key, lo, hi, integer in (("phone_width", "phone_width", 100, 4000, True), ("min_text_height", "min_text_height_px", 0, 400, False),
                                        ("min_contrast", "min_contrast_ratio", 1, 21, False)):
            if v.get(k) is not None:
                params[key] = self._number(f"variants.{k}", v[k], lo, hi, integer=integer)
        if self.qc_skill_dir:
            params["qc_skill"] = self.qc_skill_dir
        return params

    def _layers(self, args: Dict[str, Any], out: str, paths: Dict[str, str], src: str, b: Dict[str, Any]) -> Dict[str, Any]:
        lv = args.get("layers")
        if not isinstance(lv, dict) or set(lv) != {"output"}:
            raise ToolError("thumbnail: layers is {output: <the layers.json reference>}")
        manifest = paths.get(str(lv["output"]), str(lv["output"]))
        if Path(manifest).name != LAYERS_MANIFEST:
            raise ToolError(f"thumbnail: the layers manifest is {LAYERS_MANIFEST} (the Skill's own name), the plan names {Path(manifest).name}")
        directory = str(Path(manifest).resolve().parent)
        if self.workspace and not self._under(directory, self.workspace):
            raise ToolError(f"thumbnail: the layers directory is outside the workspace: {directory}")
        if self._under(out, directory) or self._under(src, directory):
            raise ToolError("thumbnail: the layers directory must hold the layers only (not the thumbnail or its input)")
        b["layers_dir"], b["manifest"] = directory, str(Path(manifest).resolve())
        b["outputs"].append(b["manifest"])
        return {"directory": directory, "overwrite": True}

    def _batch_request(self, args: Dict[str, Any], paths: Dict[str, str], src: str, fmt: str, ts: float, pid: str) -> Dict[str, Any]:
        bt = args.get("batch")
        keys = {"csv", "csv_sha256", "encoding", "delimiter", "name", "min_font_size", "targets", "outputs"}
        if not isinstance(bt, dict) or set(bt) - keys or not {"csv", "csv_sha256", "targets", "outputs"} <= set(bt):
            raise ToolError(f"thumbnail: batch is {{{', '.join(sorted(keys))}}} (csv, csv_sha256, targets, outputs required)")
        csv_path = str(bt["csv"])
        if not os.path.isfile(csv_path):
            raise ToolError(f"thumbnail: the batch table is missing: {Path(csv_path).name}")
        self._check_input(csv_path, "batch table")
        from ..skill_process import sha256_file
        if sha256_file(csv_path) != strip_sha_prefix(bt["csv_sha256"]):
            raise ToolError(f"thumbnail: the batch table {Path(csv_path).name} changed since the plan (sha256): plan again")
        enc = bt.get("encoding", "utf-8")
        if enc not in BATCH_ENCODINGS:
            raise ToolError(f"thumbnail: batch.encoding {enc!r} is not one of {list(BATCH_ENCODINGS)}")
        delim = bt.get("delimiter", ",")
        if delim not in BATCH_DELIMITERS:
            raise ToolError(f"thumbnail: batch.delimiter {delim!r} is not one of {list(BATCH_DELIMITERS)}")
        name = bt.get("name", "{{row}}-{{preset}}")
        if not isinstance(name, str) or not name or len(name) > 200:
            raise ToolError("thumbnail: batch.name must be a non-empty name template")
        targets = bt["targets"]
        if not isinstance(targets, list) or not 1 <= len(targets) <= 4:
            raise ToolError("thumbnail: batch.targets must list 1..4 platform presets")
        mfs = self._number("batch.min_font_size", bt["min_font_size"], *self.font_size_range, integer=True) if bt.get("min_font_size") is not None else None
        reqs_targets: List[Dict[str, Any]] = []
        sizes: Dict[str, Tuple[int, int]] = {}
        for t in targets:
            if not isinstance(t, dict) or set(t) - {"preset", "width", "height", "logo"} or not {"preset", "width", "height"} <= set(t):
                raise ToolError("thumbnail: each batch target is {preset, width, height, logo?}")
            preset = str(t["preset"])
            p = self.presets.get(preset)
            if p is None:
                raise ToolError(f"thumbnail: batch preset {preset!r} is not one of the contract's {sorted(self.presets)}")
            if (t["width"], t["height"]) != (p.get("width"), p.get("height")):
                raise ToolError(f"thumbnail: batch target {preset} must be laid out on the preset's {p.get('width')}x{p.get('height')} canvas")
            if fmt not in (p.get("formats") or []):
                raise ToolError(f"thumbnail: preset {preset} accepts {p.get('formats')}, not {fmt}")
            if preset in sizes:
                raise ToolError(f"thumbnail: batch preset {preset} is listed twice")
            sizes[preset] = (int(t["width"]), int(t["height"]))
            targs = {k: v for k, v in args.items() if k in ("background", "text", "font_id", "font_size", "color", "position", "texts")}
            targs.update({"width": t["width"], "height": t["height"]})
            if t.get("logo") is not None:
                targs["logo"] = t["logo"]
            doc = self._document(f"{pid[:50]}-{preset}"[:64], src, ts, targs)
            entry: Dict[str, Any] = {"preset": preset, "document": doc}
            if mfs is not None:
                entry["text_fit"] = {e["element_id"]: {"max_width": max(1, int(t["width"] * 0.9)), "min_font_size": int(mfs)} for e in doc["elements"] if e["type"] == "text"}
            reqs_targets.append(entry)
        outs = bt["outputs"]
        if not isinstance(outs, list) or not outs:
            raise ToolError("thumbnail: batch.outputs must list every (row, preset) file")
        expect: Dict[Tuple[int, str], str] = {}
        for o in outs:
            if not isinstance(o, dict) or set(o) != {"row", "preset", "output"} or o["preset"] not in sizes or isinstance(o["row"], bool) or not isinstance(o["row"], int) or o["row"] < 1:
                raise ToolError("thumbnail: each batch output is {row >= 1, preset of a target, output}")
            path = paths.get(str(o["output"]), str(o["output"]))
            self._check_output(path, src, fmt)
            expect[(o["row"], o["preset"])] = str(Path(path).resolve())
        rows = sorted({r for r, _p in expect})
        if rows != list(range(1, len(rows) + 1)) or len(expect) != len(rows) * len(sizes):
            raise ToolError("thumbnail: batch.outputs must name one file per row and preset")
        dirs = {str(Path(p).parent) for p in expect.values()}
        if len(dirs) != 1:
            raise ToolError("thumbnail: every batch file lands in one directory")
        directory = dirs.pop()
        output: Dict[str, Any] = {"directory": directory, "name": name, "format": fmt, "overwrite": True}
        if "jpeg_quality" in args:
            if fmt != "jpeg":
                raise ToolError("thumbnail: jpeg_quality is only accepted for format jpeg")
            output["jpeg_quality"] = self._number("jpeg_quality", args["jpeg_quality"], 1, 100, integer=True)
        params = {"csv": {"path": str(Path(csv_path).resolve()), "encoding": enc, "delimiter": BATCH_DELIMITERS[delim]}, "targets": reqs_targets, "output": output}
        req = {"tool": TOOL_RENDER_BATCH, "params": params}
        hit = scan_forbidden(req, tuple(f for f in self.forbidden if f not in ("path", "paths")), "request")
        if hit:
            raise ToolError(f"thumbnail: refusing to send a request carrying {hit}")
        first = expect[(1, reqs_targets[0]["preset"])]
        return {"request": req, "output": first, "input": src, "tool": TOOL_RENDER_BATCH, "format": fmt, "workspace": self.workspace or directory,
                "outputs": list(expect.values()), "batch_expect": expect, "batch_sizes": sizes, "csv_sha256": strip_sha_prefix(bt["csv_sha256"])}

    @staticmethod
    def _under(path: str, root: str) -> bool:
        try:
            p, r = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
        except OSError:
            return False
        return p == r or p.startswith(r.rstrip(os.sep) + os.sep)

    def _argv(self, b: Dict[str, Any], timeout: Optional[float]) -> List[str]:
        argv = ["run", "-", "--json", "--workspace", b["workspace"]]
        for r in list(self.allowed_inputs) + ([self.workspace] if self.workspace else []):
            argv += ["--allowed-input", r]
        argv += self._engine_argv()
        argv += ["--timeout", str(int(timeout or self.default_timeout))]
        return argv

    # ---- execution
    def _cleanup(self, b: Dict[str, Any], t0: float) -> None:
        for p in b.get("outputs") or []:
            remove_fresh(p, t0)
        if b.get("layers_dir") and os.path.isdir(b["layers_dir"]):
            for f in Path(b["layers_dir"]).glob("[0-9][0-9]-*.png"):
                remove_fresh(str(f), t0)

    def _dispatch(self, op: Operation, attempt: int, t0: float, request: Dict[str, Any], b: Dict[str, Any], timeout: Optional[float]):
        """One `thumbnail run -` call → (the tool's ok response, None, stderr tail) or (None, the failed result, tail)."""
        argv = self._argv(b, timeout)
        code, out, err = self._invoke(argv, stdin=json.dumps(request, ensure_ascii=False), timeout=(timeout or self.default_timeout) + 5.0)
        tail = "\n".join(err.strip().splitlines()[-12:])
        if code == 124:
            self._cleanup(b, t0)
            return None, self._fail(op, attempt, False, t0, 124, "CANCELLED", tail or "process timed out", retryable=True, details={"reason": "timeout"}), tail
        try:
            doc = one_json_document(out, "thumbnail")
        except ToolError as e:
            self._cleanup(b, t0)
            return None, self._fail(op, attempt, False, t0, code if code != 0 else 9, "INVALID_RESULT", str(e), retryable=False, details={"exit_code": code}), tail
        # the transport envelope: {"ok": true, "tool", "result"} on dispatch, a bare error response when the request itself was refused
        res = as_dict(doc.get("result")) if isinstance(doc.get("result"), dict) else None
        if res is None:
            if doc.get("ok") is False and isinstance(doc.get("error"), dict):
                res = doc
            else:
                self._cleanup(b, t0)
                return None, self._fail(op, attempt, False, t0, code if code != 0 else 9, "INVALID_RESULT", "response carries neither a result nor an error", retryable=False,
                                        details={"exit_code": code}), tail
        elif doc.get("ok") is not True or doc.get("tool") != request["tool"]:
            self._cleanup(b, t0)
            return None, self._fail(op, attempt, False, t0, code if code != 0 else 9, "INVALID_RESULT",
                                    f"transport envelope ok={doc.get('ok')!r} tool={doc.get('tool')!r} is not the dispatched {request['tool']}", retryable=False, details={"exit_code": code}), tail
        if res.get("ok") is not True:
            err_doc = as_dict(res.get("error"))
            errc = str(err_doc.get("code") or "INVALID_RESULT")
            if errc not in self.retryable:
                errc, retry = "INVALID_RESULT", False
            else:
                retry = bool(err_doc["retryable"]) if isinstance(err_doc.get("retryable"), bool) else self.retryable.get(errc, False)
            details = scrub(err_doc.get("details"), self.forbidden)
            if errc == "CANCELLED" and (details.get("reason") or "") not in ("timeout", "signal"):
                details["reason"] = details.get("reason") or "signal"
            self._cleanup(b, t0)
            return None, self._fail(op, attempt, False, t0, code if code != 0 else 1, errc, str(err_doc.get("message") or tail or "")[:500], retryable=bool(retry), details=details), tail
        if code != 0:
            self._cleanup(b, t0)
            return None, self._fail(op, attempt, False, t0, code, "INVALID_RESULT", f"exit code {code} with an ok response", retryable=False), tail
        return res, None, tail

    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        if op.kind == "measure":
            return self.measure(op.tool, op.args, paths, timeout)
        try:
            b = self.build_request(op.tool, op.args, paths, op_id=op.id, timeout=timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e), retryable=False)
        if dry_run:   # the Skill has no plan mode: a dry run is the lowered, validated request and nothing else
            data = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run", "operation_type": op.tool[len(PREFIX):], "request": b["request"], "commands": [], "warnings": []}
            if b.get("place"):
                data["place_text"] = b["place"]
            return ToolResult(op.id, op.tool, True, 0, None, data, [], "", round(time.time() - t0, 3), attempt, True)
        os.makedirs(b["workspace"], exist_ok=True)
        for p in b["outputs"]:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        placement = None
        if b.get("place"):   # ADR-104: thumbnail/place_text first; only the position / alignment it chose is taken over
            res, failed, _tail = self._dispatch(op, attempt, t0, b["place"], b, timeout)
            if failed is not None:
                return failed
            errs, placement = self._check_placement(res, b["place"]["params"])
            if errs:
                return self._fail(op, attempt, dry_run, t0, 9, "INVALID_RESULT", "; ".join(errs), retryable=False)
            for e in b["request"]["params"]["document"]["elements"]:
                if e["element_id"] == "caption":
                    e["text"]["position"], e["text"]["align"] = dict(placement["position"]), dict(placement["align"])
        res, failed, tail = self._dispatch(op, attempt, t0, b["request"], b, timeout)
        if failed is not None:
            return failed
        errs = self._check_response(res, b)
        if errs:
            self._cleanup(b, t0)
            return self._fail(op, attempt, dry_run, t0, 9, "INVALID_RESULT", "; ".join(errs), retryable=False)
        data = self._success_data(res, b)
        if placement is not None:
            data["placement"] = placement
            data["request_document"] = b["request"]["params"]["document"]
        return ToolResult(op.id, op.tool, True, 0, b["output"], data, [], tail, round(time.time() - t0, 3), attempt, dry_run)

    @staticmethod
    def _check_placement(res: Dict[str, Any], sent: Dict[str, Any]) -> Tuple[List[str], Optional[Dict[str, Any]]]:
        """place_text's answer is about this request: a placement of the caption over the frame, a chosen candidate that is clear, a numeric position
        and the Skill's own alignment vocabulary."""
        errs: List[str] = []
        if res.get("schema") != RESPONSE_SCHEMA or res.get("status") != "ok" or TOOL_PLACE_TEXT not in (res.get("operations") or []):
            errs.append("place_text response schema / status / operations")
        pl = as_dict(res.get("placement"))
        pos, al = as_dict(pl.get("position")), as_dict(pl.get("align"))
        if pl.get("text_element") != sent["text_element"] or pl.get("image_element") != sent["image_element"]:
            errs.append("place_text placed another element than the caption over the frame")
        if not all(isinstance(pos.get(k), (int, float)) and not isinstance(pos.get(k), bool) for k in ("x", "y")):
            errs.append("place_text gave no numeric position")
        if al.get("horizontal") not in ("left", "center", "right") or al.get("vertical") not in ("top", "middle", "bottom"):
            errs.append("place_text gave an alignment outside the contract's vocabulary")
        tried = [as_dict(t) for t in pl.get("tried") or []]
        if not tried or tried[-1].get("name") != pl.get("chosen") or tried[-1].get("clear") is not True:
            errs.append("place_text's chosen position is not a clear candidate it tried")
        if errs:
            return errs, None
        return [], {"chosen": pl.get("chosen"), "position": {"x": pos["x"], "y": pos["y"]}, "align": {"horizontal": al["horizontal"], "vertical": al["vertical"]},
                    "text_box": pl.get("text_box"), "face_boxes": pl.get("face_boxes"), "margin": pl.get("margin"), "tried": [t.get("name") for t in tried]}

    def _check_file(self, label: str, path: str, reported: Dict[str, Any], fmt: str) -> List[str]:
        errs: List[str] = []
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return [f"{label} file missing or empty"]
        ok, actual = fingerprint_matches(reported.get("sha256"), path)
        if not ok:
            errs.append(f"{label}: sha256 {reported.get('sha256')!r} != file {actual}")
        if reported.get("size") != os.path.getsize(path):
            errs.append(f"{label}: size differs from the file")
        if reported.get("format") != fmt:
            errs.append(f"{label}: format {reported.get('format')!r} is not the requested {fmt}")
        for k in ("width", "height"):
            v = reported.get(k)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                errs.append(f"{label}: {k} is not a positive integer")
        return errs

    def _check_response(self, res: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
        errs: List[str] = []
        out_path, tool = b["output"], b["tool"]
        if res.get("schema") != RESPONSE_SCHEMA:
            errs.append(f"response schema {res.get('schema')!r}")
        sk = as_dict(res.get("skill"))
        if sk.get("id") != CONTRACT_SKILL_ID or str(sk.get("version")) != self.version:
            errs.append(f"response skill {sk!r} is not {CONTRACT_SKILL_ID}@{self.version}")
        if res.get("status") != "ok":
            errs.append(f"status {res.get('status')!r} is not ok")
        if tool not in (res.get("operations") or []):
            errs.append(f"operations {res.get('operations')!r} lack {tool}")
        prov = as_dict(res.get("provenance"))
        if prov.get("skill") != CONTRACT_SKILL_ID or str(prov.get("skill_version")) != self.version or not prov.get("engine"):
            errs.append("provenance incomplete (skill / version / engine)")
        if tool == TOOL_RENDER_VARIANTS:
            return errs + self._check_variants(res, b, prov)
        if tool == TOOL_RENDER_BATCH:
            return errs + self._check_batch(res, b, prov)
        if not res.get("output") or not same_file(str(res["output"]), out_path):
            errs.append(f"output path {res.get('output')!r} is not the requested {out_path}")
        errs += self._check_file("output", out_path, res, b["format"])
        if errs and any("missing or empty" in e for e in errs):
            return errs
        if strip_sha_prefix(prov.get("output_hash")) != strip_sha_prefix(res.get("sha256")) or prov.get("operation") != (TOOL_RENDER if tool == TOOL_RENDER_LAYERS else tool) \
                or not prov.get("identity"):
            errs.append("provenance incomplete (operation / identity / output_hash)")
        if tool in DOCUMENT_TOOLS:
            assets = [a for a in prov.get("assets") or [] if isinstance(a, dict)]
            if not assets or not assets[0].get("sha256"):
                errs.append("provenance.assets lacks the source frame's sha256")
        elif not as_dict(prov.get("source")).get("sha256"):
            errs.append("provenance.source lacks the source's sha256")
        if tool == TOOL_RENDER_LAYERS:
            errs += self._check_layers(res, b, prov)
        return errs

    def _check_layers(self, res: Dict[str, Any], b: Dict[str, Any], prov: Dict[str, Any]) -> List[str]:
        """Every layer file the manifest names lies in the planned directory with the sha256 it states, the manifest on disk is the one reported, and
        the Skill says it rebuilt the thumbnail from the files read back, pixel for pixel."""
        from ..skill_process import sha256_file
        errs: List[str] = []
        lv = as_dict(res.get("layers"))
        d, manifest = b["layers_dir"], b["manifest"]
        if not lv.get("directory") or os.path.realpath(str(lv["directory"])) != os.path.realpath(d) or not lv.get("manifest") or os.path.realpath(str(lv["manifest"])) != os.path.realpath(manifest):
            return [f"layers directory / manifest {lv.get('directory')!r} / {lv.get('manifest')!r} are not the planned {d}"]
        if as_dict(lv.get("verified")).get("pixel_equal") is not True:
            errs.append("the Skill did not verify that the layers rebuild the thumbnail")
        try:
            on_disk = json.loads(Path(manifest).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return errs + ["the layers manifest is missing or not JSON"]
        if on_disk.get("schema") != LAYERS_SCHEMA or on_disk.get("layers") != lv.get("layers"):
            errs.append("the layers manifest on disk is not the one reported")
        if strip_sha_prefix(prov.get("layers_manifest_sha256")) != sha256_file(manifest):
            errs.append("provenance.layers_manifest_sha256 is not the manifest's sha256")
        files = [as_dict(x) for x in lv.get("layers") or [] if as_dict(x).get("file")]
        if not files or lv.get("count") != len(files) or as_dict(lv["layers"][0]).get("role") != "background":
            errs.append("the layer list is empty, miscounted or does not start with the background")
        for x in files:
            f = os.path.join(d, str(x["file"]))
            if os.path.dirname(os.path.realpath(f)) != os.path.realpath(d) or not os.path.isfile(f):
                errs.append(f"layer {x.get('file')!r} is not a file in the layers directory")
            elif strip_sha_prefix(x.get("sha256")) != sha256_file(f):
                errs.append(f"layer {x.get('file')!r}: sha256 differs from the file")
        return errs

    def _check_variants(self, res: Dict[str, Any], b: Dict[str, Any], prov: Dict[str, Any]) -> List[str]:
        errs: List[str] = []
        vs = [as_dict(v) for v in res.get("variants") or []]
        n = b["variant_count"]
        if [v.get("variant") for v in vs] != list(range(1, n + 1)) or res.get("count") != n:
            return [f"variants {[v.get('variant') for v in vs]!r} are not 1..{n}"]
        if prov.get("operation") != TOOL_RENDER_VARIANTS:
            errs.append("provenance.operation is not thumbnail/render_variants")
        for v, want in zip(vs, b["outputs"]):
            k = v["variant"]
            if not v.get("output") or os.path.realpath(str(v["output"])) != os.path.realpath(want):
                errs.append(f"variant {k}: output {v.get('output')!r} is not the planned {Path(want).name}")
                continue
            errs += self._check_file(f"variant {k}", want, v, b["format"])
            ss = as_dict(v.get("small_size"))
            if ss.get("status") not in ("pass", "fail", "unknown") or not isinstance(as_dict(v.get("choices")), dict) or not v.get("choices"):
                errs.append(f"variant {k}: no phone-size verdict or no documented choices")
        flagged = res.get("flagged")
        if not isinstance(flagged, list) or sorted(flagged) != sorted(v["variant"] for v in vs if as_dict(v.get("small_size")).get("status") == "fail"):
            errs.append("flagged is not the variants whose phone-size check failed")
        return errs

    def _check_batch(self, res: Dict[str, Any], b: Dict[str, Any], prov: Dict[str, Any]) -> List[str]:
        errs: List[str] = []
        items = [as_dict(i) for i in res.get("items") or []]
        expect = b["batch_expect"]
        if res.get("count") != len(expect) or len(items) != len(expect):
            return [f"batch wrote {len(items)} file(s), the plan names {len(expect)}"]
        if prov.get("operation") != TOOL_RENDER_BATCH or strip_sha_prefix(as_dict(prov.get("source")).get("sha256")) != b["csv_sha256"]:
            errs.append("provenance is not render_batch of the planned table (sha256)")
        seen = set()
        for it in items:
            key = (it.get("row"), it.get("preset"))
            want = expect.get(key)  # type: ignore[arg-type]
            if want is None or key in seen:
                errs.append(f"batch item {key!r} is not a planned (row, preset) or repeats one")
                continue
            seen.add(key)
            if not it.get("output") or os.path.realpath(str(it["output"])) != os.path.realpath(want):
                errs.append(f"batch item {key!r}: output {it.get('output')!r} is not the planned {Path(want).name}")
                continue
            errs += self._check_file(f"batch item {key!r}", want, it, b["format"])
            if (it.get("width"), it.get("height")) != b["batch_sizes"][key[1]]:
                errs.append(f"batch item {key!r}: {it.get('width')}x{it.get('height')} is not the preset's canvas")
            if as_dict(it.get("preset_check")).get("ok") is not True:
                errs.append(f"batch item {key!r}: the preset check did not pass")
        return errs

    def _success_data(self, res: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        prov = as_dict(res.get("provenance"))
        tool = b["tool"]
        op_type = tool[len(PREFIX):]
        extra: Dict[str, Any] = {}
        if tool == TOOL_RENDER_VARIANTS:
            vs = [as_dict(v) for v in res.get("variants") or []]
            main = vs[0]
            extra["variants"] = [{"variant": v["variant"], "output": v["output"], "sha256": strip_sha_prefix(v.get("sha256")), "size": v.get("size"), "choices": v.get("choices"),
                                  "description": v.get("description"), "small_size": {k: as_dict(v.get("small_size")).get(k) for k in ("status", "ok", "phone_width", "min_text_height_px",
                                                                                                                                   "min_contrast_ratio", "texts")}} for v in vs]
            extra["flagged"] = list(res.get("flagged") or [])
            extra["axes"] = res.get("axes")
            source = {"sha256": None, "timestamp": None, "duration": None}
        elif tool == TOOL_RENDER_BATCH:
            items = [as_dict(i) for i in res.get("items") or []]
            main = next(i for i in items if os.path.realpath(str(i.get("output"))) == os.path.realpath(b["output"]))
            extra["items"] = [{k: i.get(k) for k in ("row", "preset", "output", "size", "width", "height", "fitted", "preset_check")} | {"sha256": strip_sha_prefix(i.get("sha256"))}
                              for i in items]
            extra["table"] = {"sha256": b["csv_sha256"], "rows": res.get("rows"), "targets": res.get("targets")}
            source = {"sha256": None, "timestamp": None, "duration": None}
        else:
            main = res
            if tool in DOCUMENT_TOOLS:
                a = as_dict((prov.get("assets") or [None])[0])
                source = {"sha256": strip_sha_prefix(a.get("sha256")), "timestamp": a.get("timestamp"), "duration": a.get("source_duration")}
            else:
                s = as_dict(prov.get("source"))
                source = {"sha256": strip_sha_prefix(s.get("sha256")), "timestamp": s.get("timestamp"), "duration": s.get("duration")}
            if tool == TOOL_RENDER_LAYERS:
                lv = as_dict(res.get("layers"))
                extra["layers"] = {"directory": lv.get("directory"), "manifest": lv.get("manifest"), "count": lv.get("count"), "verified": lv.get("verified"),
                                   "manifest_sha256": strip_sha_prefix(prov.get("layers_manifest_sha256")),
                                   "files": [{k: as_dict(x).get(k) for k in ("order", "role", "element_id", "file", "position", "size", "sha256")} for x in lv.get("layers") or []]}
        sha = strip_sha_prefix(main.get("sha256"))
        artifact = {"path": b["output"], "sha256": sha, "size": main.get("size"), "format": main.get("format"), "width": main.get("width"), "height": main.get("height"),
                    "reused": bool(main.get("reused")), "identity": main.get("identity") or prov.get("identity")}
        return {"skill": {"id": SKILL_ID, "version": self.version}, "status": "completed", "operation_type": op_type, "artifact": artifact,
                "observation": {"kind": "image.probe", "source": f"{CONTRACT_SKILL_ID}/{op_type}@{self.version}", "provenance": "OBSERVED",
                                "data": {k: artifact[k] for k in ("width", "height", "format", "size", "sha256")}},
                "provenance": {k: prov.get(k) for k in ("skill", "skill_version", "operation", "engine", "engine_version", "document_id", "identity", "reused", "assets", "source", "fonts",
                                                        "output_hash", "outputs", "rows", "targets", "qc_skill", "layers_manifest_sha256") if k in prov},
                "source": source, "warnings": list(res.get("warnings") or []), "commands": [], **extra}

    # ---- ADR-104: thumbnail/choose_frame, a measurement (analysis kind frame_ranking)
    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        from ..skill_process import sha256_file
        t0 = time.time()
        op = Operation(tool=tool, args=dict(args or {}), inputs=[], outputs=[], kind="measure")
        if tool != TOOL_CHOOSE_FRAME:
            return self._fail(op, 1, False, t0, 2, "INVALID_REQUEST", f"thumbnail: {tool} is not a measurement (only {TOOL_CHOOSE_FRAME} is)", retryable=False)
        try:
            hit = scan_forbidden(args, self.forbidden)
            if hit:
                raise ToolError(f"thumbnail: forbidden field {hit} in the measurement arguments")
            unknown = sorted(k for k in args if k not in ("input", "detections_path", "detections_sha256"))
            if unknown or not {"input", "detections_path", "detections_sha256"} <= set(args):
                raise ToolError("thumbnail: choose_frame takes {input, detections_path, detections_sha256}")
            src, det = str(args["input"]), str(args["detections_path"])
            for p, what in ((src, "video"), (det, "detections file")):
                if not os.path.isfile(p):
                    raise ToolError(f"thumbnail: {what} not found: {Path(p).name}")
                self._check_input(p, what)
            if sha256_file(det) != strip_sha_prefix(args["detections_sha256"]):
                raise ToolError("thumbnail: the detections file changed since it was measured (sha256)")
        except ToolError as e:
            return self._fail(op, 1, False, t0, 2, "INVALID_REQUEST", str(e), retryable=False)
        b = {"workspace": self.workspace or str(Path(det).parent), "outputs": []}
        request = {"tool": TOOL_CHOOSE_FRAME, "params": {"detections_path": str(Path(det).resolve()), "video": str(Path(src).resolve())}}
        res, failed, tail = self._dispatch(op, 1, t0, request, b, timeout)
        if failed is not None:
            return failed
        errs = self._check_ranking(res, sha256_file(src))
        if errs:
            return self._fail(op, 1, False, t0, 9, "INVALID_RESULT", "; ".join(errs), retryable=False)

        def entry(e: Dict[str, Any]) -> Dict[str, Any]:
            return {k: e.get(k) for k in ("rank", "index", "t", "score", "parts", "missing", "eyes_closed", "faces", "main_face", "main_face_box")}
        ranking = [entry(as_dict(e)) for e in res.get("ranking") or []]
        data = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "ok", "tool": TOOL_CHOOSE_FRAME, "ranking": ranking,
                "chosen": entry(as_dict(res["chosen"])) if isinstance(res.get("chosen"), dict) else None, "frame_size": dict(as_dict(res.get("frame_size"))),
                "method": as_dict(res.get("method")), "source": as_dict(res.get("source")), "fingerprint": strip_sha_prefix(as_dict(res.get("video")).get("fingerprint")),
                "detections_sha256": strip_sha_prefix(args["detections_sha256"]), "warnings": list(res.get("warnings") or []), "commands": []}
        return ToolResult(op.id, tool, True, 0, None, data, [], tail, round(time.time() - t0, 3), 1, False)

    def _check_ranking(self, res: Dict[str, Any], video_sha256: str) -> List[str]:
        errs: List[str] = []
        if res.get("schema") != RESPONSE_SCHEMA or res.get("status") != "ok" or as_dict(res.get("skill")).get("id") != CONTRACT_SKILL_ID \
                or str(as_dict(res.get("skill")).get("version")) != self.version or TOOL_CHOOSE_FRAME not in (res.get("operations") or []):
            errs.append("choose_frame response schema / skill / status / operations")
        if strip_sha_prefix(as_dict(res.get("video")).get("fingerprint")) != video_sha256 or strip_sha_prefix(as_dict(res.get("source")).get("fingerprint")) != video_sha256:
            errs.append("the ranking is not about this video (fingerprint)")
        if as_dict(res.get("source")).get("detections") != DETECTIONS_SCHEMA or as_dict(res.get("method")).get("heuristic") is not True:
            errs.append("the ranking does not name its detections schema or its heuristic method")
        fs = as_dict(res.get("frame_size"))
        if not all(isinstance(fs.get(k), int) and not isinstance(fs.get(k), bool) and fs[k] > 0 for k in ("width", "height")):
            errs.append("frame_size missing")
        ranking = res.get("ranking")
        if not isinstance(ranking, list):
            return errs + ["ranking is not a list"]
        ranked = []
        for e in ranking:
            e = as_dict(e)
            faces = e.get("faces")
            if not isinstance(faces, list) or not all(isinstance(f, list) and len(f) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in f) for f in faces):
                return errs + ["a ranked frame without [x, y, w, h] face boxes"]
            if e.get("rank") is not None:
                if not isinstance(e.get("t"), (int, float)) or isinstance(e.get("t"), bool) or not isinstance(e.get("score"), (int, float)) or not isinstance(e.get("eyes_closed"), bool) or not faces:
                    return errs + ["a ranked frame without a time, a score, faces and an eyes_closed verdict"]
                ranked.append(e)
        if [e["rank"] for e in ranked] != list(range(1, len(ranked) + 1)):
            errs.append("ranks are not 1..n in order")
        if ranked and as_dict(res.get("chosen")).get("t") != ranked[0]["t"]:
            errs.append("chosen is not the first ranked frame")
        return errs

    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, code: int, errc: str, message: str, retryable: bool, details: Optional[Dict[str, Any]] = None) -> ToolResult:
        return failed_result(op, SKILL_ID, getattr(self, "version", ""), attempt, dry_run, t0, code, errc, message, retryable, details)


def lift_observation(result: ToolResult, asset_id: Optional[str] = None) -> Optional[Observation]:
    """The Skill's OBSERVED facts about the delivered image (dimensions, format, size, sha256) as an agent Observation
    (provenance only; never fed back into analysis)."""
    obs = (result.data or {}).get("observation")
    if not isinstance(obs, dict) or obs.get("provenance") != "OBSERVED" or not isinstance(obs.get("data"), dict):
        return None
    sk = (result.data or {}).get("skill") or {}
    art = (result.data or {}).get("artifact") or {}
    return Observation(kind="image.probe", asset_id=asset_id or result.op_id, source=str(obs.get("source")), data=dict(obs["data"]), analyzer=str(obs.get("source")),
                       provenance="OBSERVED", skill=SKILL_ID, skill_version=str(sk.get("version") or ""), tool=result.tool, fingerprint=str(art.get("sha256") or ""),
                       parameters={"output": result.output, "identity": art.get("identity")})
