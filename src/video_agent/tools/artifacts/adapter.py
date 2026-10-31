"""ArtifactsAdapter: the agent's boundary to artifacts-skill (ADR-095; artifacts-skill roadmap step 5, agent side: adapter + SkillSpec document_verify).

    video-production-agent ─(plan step document_verify, QA)─→ ArtifactsAdapter ─(argv lists, never a shell)─→
        `artifacts-skill verify --json --policy={"forbid_placeholder_text": true} [--policy-preset=NAME] -- <document>`

Protocol (artifacts-skill's own contract, `artifacts-skill contract --json`, pinned from main in contract_0.3.0.json -- the repository has no
release tag):
    contract → `artifacts-skill contract --json` (skill_id artifacts-skill, contract_version 1.x, tools[] by `name`, provides[] by tool_id,
               exit_codes {ok 0, fail 1, input 2, capability 3, security 4, internal 5}, input_mutation_policy never)
    doctor   → `artifacts-skill doctor --json` ({tool_version, capabilities: {<id>: {status available | missing | not_required, detail}}})
    verify   → structural verification, read-only and writing no file (`side_effects`): {kind structural, status, checks: [{id, name, status,
               message, evidence}]}, statuses pass | warn | fail | unknown | not_checked | skipped; exit 0 (pass / warn), 1 (fail), 3 (unknown)
    failure  → {"error": {code, category, message, remediation, evidence}}, the category fixing the exit code

The policy always forbids leftover generation placeholders (`forbid_placeholder_text`: "Lorem ipsum", "Click to add title", "TODO", ...), so a
document still carrying one is a FAIL check `leftover_placeholder_text`; a named preset (`policy_preset`) adds its own checks. Nothing here
edits, renders or judges a layout: the verdict is the Skill's structural checks, kept by id. Every value is checked before a process starts
(the document is a file inside the allowed roots, the preset one of the Skill's names); the document's sha256 is taken before and after the
run (the Skill never mutates its input; a changed file is refused). The result is admitted only when it is a structural verification with a
known status and checks that each carry an id and a known status; the Skill's error codes are kept verbatim with the agent's recovery class."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...models import Operation, ToolResult
from ...skills.contract import SkillPackage, ToolSpec
from ..base import ToolAdapter, ToolError
from ..skill_process import CliSkill, ContractError, as_dict, breaking_drift as _breaking_drift, drift_report, invoke, one_json_document, sha256_file
from .locate import PACKAGE_ID, locate_artifacts  # noqa: F401  (re-exported)

SKILL_ID = PACKAGE_ID
PREFIX = SKILL_ID + "/"
TOOL_VERIFY = "artifacts-skill/verify"
SUPPORTED_SKILL_VERSIONS = ("0.3.",)
STATUSES = ("pass", "warn", "fail", "unknown", "not_checked", "skipped")
PLACEHOLDER_CHECK = "leftover_placeholder_text"
POLICY = {"forbid_placeholder_text": True}
# artifacts-skill's named policies (src/artifact_skill/policies.py PRESETS on main); the Skill refuses any other name itself
PRESETS = ("print-a4", "print-letter", "slides-16x9", "spreadsheet-no-cached-errors", "web-no-external")
DEFAULT_TIMEOUT = 600.0
# the error categories (artifacts-skill core/errors.py) → the agent's recovery class; a missing backend is TOOL_MISSING (the user installs it)
CATEGORY_CLASSES: Dict[str, str] = {"input": "INVALID_ARGS", "capability": "TOOL_MISSING", "security": "INVALID_ARGS", "execution": "SKILL_ERROR",
                                    "verification": "SKILL_ERROR", "internal": "SKILL_ERROR"}
CODE_CLASSES: Dict[str, str] = {"ARTIFACT_INPUT_NOT_FOUND": "INPUT_MISSING", "ARTIFACT_TYPE_UNSUPPORTED": "INVALID_ARGS", "ARTIFACT_CAPABILITY_MISSING": "TOOL_MISSING"}
AGENT_ERRORS: Dict[str, str] = {"INVALID_REQUEST": "INVALID_ARGS", "INVALID_OUTPUT": "SKILL_ERROR", "TIMEOUT": "TIMEOUT", "INPUT_CHANGED": "SKILL_ERROR"}
EXIT_CODES = {"ok": 0, "fail": 1, "input": 2, "capability": 3, "security": 4, "internal": 5}
# contract drift (ADR-045 / ADR-047): every key the adapter reads; the fatal subset gates availability
DRIFT_KEYS = ("skill_id", "version", "contract_version", "provides", "not_provided", "exit_codes", "network_policy", "input_mutation_policy")
DRIFT_TOOL_KEYS = ("input_schema", "output_schema", "side_effects", "errors", "artifact_types")
FATAL_DRIFT_KEYS = ("skill_id", "contract_version", "exit_codes", "input_mutation_policy")
FATAL_DRIFT_TOOL_KEYS = ("input_schema", "side_effects")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.3.0.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def recovery_class(code: str, category: Optional[str] = None) -> str:
    return CODE_CLASSES.get(code) or AGENT_ERRORS.get(code) or CATEGORY_CLASSES.get(str(category or "")) or "SKILL_ERROR"


def _tool(contract: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next((t for t in contract.get("tools") or [] if isinstance(t, dict) and t.get("name") == name), {})


def check_contract(contract: Any) -> List[str]:
    """What the adapter relies on, refused (never patched) when it does not hold: identity, contract and Skill version, `verify` provided
    under artifacts-skill/verify taking input / policy / policy_preset, read-only and writing no file, the exit codes it reads, and an
    input-mutation policy of never."""
    if not isinstance(contract, dict):
        return ["contract is not an object"]
    errs: List[str] = []
    if contract.get("skill_id") != SKILL_ID:
        errs.append(f"skill_id {contract.get('skill_id')!r} != {SKILL_ID}")
    if not str(contract.get("contract_version") or "").startswith("1."):
        errs.append(f"contract_version {contract.get('contract_version')!r} is not 1.x")
    ver = str(contract.get("version") or "")
    if not ver.startswith(SUPPORTED_SKILL_VERSIONS):
        errs.append(f"skill version {ver!r} not in supported range {SUPPORTED_SKILL_VERSIONS}")
    if TOOL_VERIFY not in [as_dict(p).get("tool_id") for p in contract.get("provides") or []]:
        errs.append(f"provides lacks {TOOL_VERIFY}")
    v = _tool(contract, "verify")
    props = as_dict(as_dict(v.get("input_schema")).get("properties"))
    if not v:
        errs.append("contract lacks tool verify")
    elif any(k not in props for k in ("input", "policy", "policy_preset")):
        errs.append(f"verify input_schema must take input, policy and policy_preset (has {sorted(props)})")
    se = as_dict(v.get("side_effects"))
    if v and (se.get("mutates_input") is not False or se.get("writes_files") is not False):
        errs.append("verify must declare side_effects mutates_input false and writes_files false")
    codes = as_dict(contract.get("exit_codes"))
    if any(codes.get(k) != n for k, n in EXIT_CODES.items()):
        errs.append(f"exit_codes {json.dumps(codes, sort_keys=True)} do not keep {json.dumps(EXIT_CODES, sort_keys=True)}")
    if contract.get("input_mutation_policy") != "never":
        errs.append(f"input_mutation_policy {contract.get('input_mutation_policy')!r} is not never")
    return errs


def _used(c: Dict[str, Any]) -> Dict[str, Any]:
    return dict(c, tools=[t for t in c.get("tools") or [] if isinstance(t, dict) and t.get("name") == "verify"])


def drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(_used(live), _used(pinned or pinned_contract()), DRIFT_KEYS, "tools", "name", DRIFT_TOOL_KEYS)


def breaking_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return _breaking_drift(_used(live), _used(pinned or pinned_contract()), FATAL_DRIFT_KEYS, "tools", "name", FATAL_DRIFT_TOOL_KEYS)


def check_verification(doc: Any) -> List[str]:
    """The verify result as the agent admits it: a structural verification with a known status and checks each carrying an id and a known status."""
    if not isinstance(doc, dict):
        return ["the result is not an object"]
    errs: List[str] = []
    if doc.get("kind") != "structural":
        errs.append(f"kind {doc.get('kind')!r} is not structural")
    if doc.get("status") not in STATUSES:
        errs.append(f"status {doc.get('status')!r} is not one of {', '.join(STATUSES)}")
    checks = doc.get("checks")
    if not isinstance(checks, list) or not checks:
        return errs + ["no checks"]
    for c in checks:
        c = as_dict(c)
        if not isinstance(c.get("id"), str) or not c["id"] or c.get("status") not in STATUSES:
            errs.append(f"check {c.get('id')!r} has no id or an unknown status {c.get('status')!r}")
    return errs


PACKAGE = SkillPackage(
    skill_id=SKILL_ID, name="artifacts-skill", version="",
    description="Structural verification of a generated document (PDF, PPTX, DOCX, XLSX, Markdown, HTML, SVG, CSV, EPUB, images): page / slide facts, "
                "encryption, scripts, leftover placeholder text; every check PASS / WARN / FAIL / UNKNOWN / NOT_CHECKED / SKIPPED, read-only, local",
    capabilities=["artifacts-skill"],
    tools=[ToolSpec(TOOL_VERIFY, SKILL_ID, description="Structural checks of one document against a policy that forbids leftover placeholder text (read-only, writes no file)",
                    required_capabilities=["artifacts-skill:VERIFY"], inputs=["input"], produces_output=False, deterministic=True,
                    result_keys=["status", "checks", "sha256", "document"])],
    repository="kajisho5/artifacts-skill", role="verification of generated documents (structural checks with evidence)")


class ArtifactsAdapter(ToolAdapter):
    def __init__(self, skill: CliSkill, allowed_inputs: Optional[List[str]] = None, timeout: float = DEFAULT_TIMEOUT):
        self.skill = skill
        self.name = SKILL_ID
        self.allowed_inputs = [os.path.realpath(os.path.abspath(p)) for p in (allowed_inputs or [])]
        self.default_timeout = float(timeout)
        code, out, err = invoke(skill, ["contract", "--json"], timeout=60.0)
        if code != 0:
            raise ContractError(f"artifacts-skill contract --json failed (exit {code}): {err.strip()[-300:]}")
        self.contract = one_json_document(out, "artifacts-skill contract")
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("incompatible artifacts-skill contract: " + "; ".join(errs))
        self.version = str(self.contract.get("version"))
        self.tools = {TOOL_VERIFY: True}
        self._doctor: Optional[Dict[str, Any]] = None

    def doctor(self) -> Dict[str, Any]:
        """`doctor --json`: ok when at least one format's structural verification is available (`<format>.structural`); the formats that
        are not are listed as warnings with the Skill's own detail (a document of such a format fails its step with the Skill's error)."""
        if self._doctor is not None:
            return self._doctor
        code, out, err = invoke(self.skill, ["doctor", "--json"], timeout=120.0)
        try:
            doc = one_json_document(out, "artifacts-skill doctor")
        except ToolError as e:
            self._doctor = {"status": "fail", "capabilities": {}, "problems": [f"doctor produced no document (exit {code}): {e} {err.strip()[-200:]}"], "warnings": []}
            return self._doctor
        caps = {k: as_dict(v) for k, v in as_dict(doc.get("capabilities")).items()}
        structural = {k[:-len(".structural")]: v.get("status") for k, v in caps.items() if k.endswith(".structural")}
        ready = sorted(f for f, st in structural.items() if st == "available")
        self._doctor = dict(doc, status="ok" if ready else "fail", formats=ready,
                            problems=[] if ready else ["no format's structural verification is available"],
                            warnings=[f"{f}.structural: {caps[f + '.structural'].get('detail')}" for f, st in sorted(structural.items()) if st != "available"])
        return self._doctor

    def operation_status(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        return {"VERIFY": "supported" if (doc or self.doctor()).get("status") == "ok" else "unavailable"}

    def operation_evidence(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        doc = doc or self.doctor()
        ev: Dict[str, Any] = {"ready": doc.get("status") == "ok", "formats": list(doc.get("formats") or [])}
        if doc.get("status") != "ok":
            ev.update(code="ARTIFACT_CAPABILITY_MISSING", install="pip install -e 'artifacts-skill[all]'", detail="; ".join(doc.get("problems") or [])[:300])
        return {"VERIFY": ev}

    def drift(self) -> List[str]:
        return drift(self.contract)

    def breaking_drift(self) -> List[str]:
        return breaking_drift(self.contract)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "skill": self.skill.describe(), "tools": [TOOL_VERIFY], "drift": self.drift()}

    def package(self) -> SkillPackage:
        return SkillPackage(PACKAGE.skill_id, PACKAGE.name, self.version, PACKAGE.description, list(PACKAGE.capabilities),
                            [ToolSpec(**{**t.__dict__, "version": self.version}) for t in PACKAGE.tools], PACKAGE.repository, PACKAGE.role)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        return self.run(Operation(tool=tool, args=args, inputs=[], outputs=[], kind="measure"), paths or {}, timeout=timeout)

    def _within(self, root: str, p: str) -> bool:
        return os.path.normcase(p) == os.path.normcase(root) or os.path.normcase(p).startswith(os.path.normcase(root).rstrip(os.sep) + os.sep)

    def build(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None) -> Dict[str, Any]:
        if op.tool != TOOL_VERIFY:
            raise ToolError(f"{SKILL_ID} adapter does not support {op.tool}")
        args = dict(op.args)
        unknown = sorted(k for k in args if k not in ("input", "preset"))
        if unknown:
            raise ToolError(f"unknown verify argument(s) {unknown} (the agent sends the document and an optional preset; the policy is its own)")
        raw = args.get("input")
        p = paths.get(raw, raw) if isinstance(raw, str) else raw
        if not isinstance(p, str) or not p.strip() or "\x00" in p or "\n" in p:
            raise ToolError("input must be a non-empty single-line path")
        doc = os.path.realpath(os.path.abspath(p))
        if self.allowed_inputs and not any(self._within(r, doc) for r in self.allowed_inputs):
            raise ToolError(f"the document is outside the allowed input roots: {os.path.basename(p)}")
        if not os.path.isfile(doc):
            raise ToolError(f"the document is not a file: {os.path.basename(p)}")
        argv = ["verify", "--json", "--policy=" + json.dumps(POLICY, sort_keys=True)]
        if args.get("preset") is not None:
            if args["preset"] not in PRESETS:
                raise ToolError(f"preset must be one of {', '.join(PRESETS)}, got {args['preset']!r}")
            argv.append(f"--policy-preset={args['preset']}")
        argv += ["--", doc]
        return {"argv": argv, "document": doc, "timeout": max(1.0, min(3600.0, float(timeout or self.default_timeout)))}

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            return [" ".join(["artifacts-skill"] + self.build(op, paths)["argv"])]
        except ToolError as e:
            return [f"<refused: {e}>"]

    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, exit_code: int, code: str, message: str, category: Optional[str] = None,
              retryable: bool = False, extra: Optional[Dict[str, Any]] = None) -> ToolResult:
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "failed", **(extra or {}),
                                "error": {"code": code, "message": message[:1200], "retryable": bool(retryable), "exit_code": exit_code,
                                          "recovery_class": recovery_class(code, category), **({"category": category} if category else {})}}
        return ToolResult(op.id, op.tool, False, exit_code, None, data, [], f"{SKILL_ID} [{code}] {message[:600]}", round(time.time() - t0, 3), attempt, dry_run)

    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        try:
            b = self.build(op, paths, timeout)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e))
        command = " ".join(["artifacts-skill"] + b["argv"][:-1] + [os.path.basename(b["document"])])
        if dry_run:
            return ToolResult(op.id, op.tool, True, 0, None, {"skill": {"id": SKILL_ID, "version": self.version}, "status": "dry_run", "commands": [command]},
                              [command], "", round(time.time() - t0, 3), attempt, True)
        before = sha256_file(b["document"])
        code, out, err = invoke(self.skill, b["argv"], timeout=b["timeout"])
        tail = "\n".join(err.strip().splitlines()[-12:])
        name = os.path.basename(b["document"])
        if code == 124:
            return self._fail(op, attempt, dry_run, t0, 124, "TIMEOUT", "artifacts-skill verify exceeded its time limit", retryable=True)
        try:
            doc = one_json_document(out, "artifacts-skill verify")
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, code if code else 9, "INVALID_OUTPUT", f"{e} (exit {code}) {tail[-300:]}".strip())
        if sha256_file(b["document"]) != before:
            return self._fail(op, attempt, dry_run, t0, 9, "INPUT_CHANGED", f"{name} changed while it was verified (the Skill never mutates its input)")
        if isinstance(doc.get("error"), dict):
            e = as_dict(doc["error"])
            msg = str(e.get("message") or "") + (f" ({e.get('remediation')})" if e.get("remediation") else "")
            return self._fail(op, attempt, dry_run, t0, code if code else 1, str(e.get("code") or "SKILL_ERROR"), msg, category=str(e.get("category") or ""),
                              extra={"document": name, "sha256": before})
        why = check_verification(doc)
        want = {"pass": 0, "warn": 0, "fail": 1, "unknown": 3}.get(str(doc.get("status")), 0)
        if not why and code != want:
            why.append(f"exit {code} does not match status {doc.get('status')} (expected exit {want})")
        if why:
            return self._fail(op, attempt, dry_run, t0, code if code else 9, "INVALID_OUTPUT", "artifacts-skill result not admitted: " + "; ".join(why))
        checks = [{k: as_dict(c).get(k) for k in ("id", "name", "status") + (("message",) if as_dict(c).get("status") != "pass" else ())} for c in doc["checks"]]
        data = {"skill": {"id": SKILL_ID, "version": self.version}, "document": name, "sha256": before, "status": doc["status"], "checks": checks,
                "placeholders": next((as_dict(as_dict(c).get("evidence")).get("markers") for c in doc["checks"] if as_dict(c).get("id") == PLACEHOLDER_CHECK), None),
                "preset": op.args.get("preset"), "commands": [command]}
        # a FAIL / UNKNOWN verdict is a measured answer, not a failed step: QA gates on it (the step itself succeeded)
        return ToolResult(op.id, op.tool, True, 0, None, data, [command], tail, round(time.time() - t0, 3), attempt, False)
