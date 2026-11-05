"""PublishAdapter: the agent's boundary to publish-skill (ADR-087; publish-skill roadmap step 6, agent side: SkillSpec publish_upload).

    video-production-agent ─(a QC-passed, delivered artifact + the plan's approved metadata)─→ PublishAdapter ─(argv list, never a shell)─→
        `publish plan --json --platform=P --ffmpeg-skill=<dir> [--title=… --description=… --tag=… --privacy=… --thumbnail=… --caption=… --caption-language=…] -- <file>`
        `publish upload --json --platform=P [metadata] -- <file>`                      (dry run: no network, prints the confirm_token)
        `publish upload --json --platform=P [metadata] --confirm=<token> -- <file>`    (the upload: only with the token a human typed)

Protocol (publish-skill's own contract, `publish contract --json`, pinned from main in contract_0.1.0.json -- the repository has no release
tag, so the pin follows main as dubbing-skill's does, ADR-085):
    platforms  → `platforms.registered.<id>` (implemented, metadata_fields) and `tools[publish/upload].uploaders.<id>`: only a platform whose
                 uploader exists on the pinned main is offered; a platform added later is refused until the pin is renewed
    limits     → `upload_limits.platforms.<id>["privacy.values"]` (the privacy each platform takes; X: public only)
    plan       → publish/plan@1: verdict PASS / WARN / FAIL / INCOMPLETE with rows (offline)
    dry run    → ok, dry_run true, uploaded false, network_used false, confirm_token, input (path, size, sha256), metadata, credentials (present / absent)
    upload     → ok, uploaded true, result (video_id / post_id, url, privacy, warnings); errors {code, message, retryable, details}
    confirmation → `confirmation.flag` --confirm, the token binds platform, deliverable sha256 / size and every metadata field;
                 never from the environment or a config file

Publishing cannot be undone. The adapter therefore never runs `upload --confirm` as a pipeline operation (run() refuses it): render runs the
offline plan and the dry run only, and the upload is a separate call (Service.publish) with a token a human typed, sent once and never
retried -- whatever `retryable` says (an interrupted upload may have published; the Skill's SKILL.md says to check before any retry).
Credentials stay where publish-skill reads them (the environment or the OS keychain); the agent never reads, passes or prints one."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...models import Operation, ToolResult
from ...skills.contract import SkillPackage, ToolSpec
from ..base import ToolAdapter, ToolError
from ..ffmpeg_skill.adapter import PathPolicy
from ..skill_process import (FORBIDDEN_ARG_KEYS, CliSkill, ContractError, as_dict, breaking_drift as _breaking_drift, drift_report, invoke, one_json_document,
                             scan_forbidden, scrub, sha256_file)
from .locate import PACKAGE_ID, locate_publish  # noqa: F401  (re-exported)

SKILL_ID = PACKAGE_ID
PREFIX = SKILL_ID + "/"
TOOL_PLAN = "publish/plan"
TOOL_UPLOAD = "publish/upload"
TOOLS = (TOOL_UPLOAD,)          # the plan step's tool; publish/plan is called by the same adapter as the upload's precondition
PLAN_SCHEMA = "publish/plan@1"
DOCTOR_SCHEMA = "publish/doctor@1"
SUPPORTED_SKILL_VERSIONS = ("0.1.",)
CONFIRM_FLAG = "--confirm"
TOKEN_RE = re.compile(r"^confirm-[0-9a-f]{16}$")
PLAN_OK = ("PASS", "WARN")      # publish-skill: FAIL / INCOMPLETE → do not upload
LANG_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8}){0,3}$")
MAX_TEXT = 10000                # an agent-side bound on any one metadata string (the platform limits are the Skill's)
DEFAULT_TIMEOUT = 300.0         # plan / dry run (offline)
UPLOAD_TIMEOUT = 6 * 3600.0     # an upload of a large file
# publish-skill's error codes (its contract `errors`) → the agent's recovery class. An upload is never retried by the agent (see module doc),
# so the retryable codes are recorded as the Skill reports them and the human decides.
ERROR_CLASSES: Dict[str, str] = {
    "INVALID_ARGUMENT": "INVALID_ARGS", "INPUT_NOT_FOUND": "INPUT_MISSING", "UNKNOWN_PLATFORM": "INVALID_ARGS", "CONFIRM_MISMATCH": "INVALID_ARGS",
    "METADATA_INVALID": "INVALID_ARGS", "CREDENTIALS_MISSING": "TOOL_MISSING", "KEYCHAIN_UNAVAILABLE": "TOOL_MISSING", "KEYCHAIN_ERROR": "TOOL_MISSING",
    "AUTH_FAILED": "TOOL_MISSING", "QUOTA_EXCEEDED": "SKILL_ERROR", "UPLOAD_LIMIT_EXCEEDED": "SKILL_ERROR", "UPLOAD_INTERRUPTED": "SKILL_ERROR",
    "UPLOAD_REJECTED": "SKILL_ERROR", "UPLOAD_PARTIAL": "SKILL_ERROR", "NOT_IMPLEMENTED": "SKILL_ERROR", "CANCELLED": "INTERRUPTED", "INTERNAL_ERROR": "SKILL_ERROR",
}
AGENT_ERRORS: Dict[str, str] = {"INVALID_REQUEST": "INVALID_ARGS", "INVALID_OUTPUT": "SKILL_ERROR", "TIMEOUT": "TIMEOUT"}
# contract drift (ADR-045 / ADR-047): what the adapter reads; a platform or limit added on main is reported, not fatal
DRIFT_KEYS = ("skill_id", "version", "contract_version", "provides", "dependencies", "not_provided", "platforms", "upload_limits", "confirmation", "credentials",
              "execution", "errors", "error_shape")
DRIFT_TOOL_KEYS = ("role", "network", "writes_media", "mutates_input", "implemented", "schema", "argv", "uploaders", "verdicts", "row_statuses")
FATAL_DRIFT_KEYS = ("skill_id", "contract_version", "confirmation", "error_shape")
FATAL_DRIFT_TOOL_KEYS = ("role", "network", "writes_media", "mutates_input", "schema", "argv", "verdicts")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def _tool(contract: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
    return next((t for t in contract.get("tools") or [] if isinstance(t, dict) and t.get("tool_id") == tool_id), {})


def platforms(contract: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """id → {name, metadata_fields, privacy} for every platform the contract registers as implemented AND whose uploader exists
    (`tools[publish/upload].uploaders`). Read from the contract, never listed here."""
    c = contract or pinned_contract()
    reg = as_dict(as_dict(c.get("platforms")).get("registered"))
    up = as_dict(_tool(c, TOOL_UPLOAD).get("uploaders"))
    lim = as_dict(as_dict(c.get("upload_limits")).get("platforms"))
    out: Dict[str, Dict[str, Any]] = {}
    for pid, p in sorted(reg.items()):
        p = as_dict(p)
        if p.get("implemented") is True and up.get(pid) is True:
            privacy = as_dict(as_dict(lim.get(pid)).get("privacy.values")).get("value")
            out[str(pid)] = {"name": p.get("name") or pid, "metadata_fields": [str(f) for f in p.get("metadata_fields") or []],
                             "privacy": [str(v) for v in privacy] if isinstance(privacy, list) else [], "credential_env": [str(v) for v in p.get("credential_env") or []]}
    return out


def privacy_values(contract: Optional[Dict[str, Any]] = None) -> Tuple[str, ...]:
    """`--privacy private|unlisted|public` as the upload argv spells it."""
    for a in _tool(contract or pinned_contract(), TOOL_UPLOAD).get("argv") or []:
        m = re.match(r"^\[--privacy ([a-z|]+)\]$", str(a))
        if m:
            return tuple(m.group(1).split("|"))
    return ()


def recovery_class(code: str) -> str:
    return ERROR_CLASSES.get(code) or AGENT_ERRORS.get(code) or "SKILL_ERROR"


def check_contract(contract: Any) -> List[str]:
    """What the adapter relies on, refused (never patched): identity and versions; plan offline (measurement, no network, its schema and
    verdicts); upload an execution tool whose network use needs `--confirm <token>`; the confirmation never taken from the environment or a
    config; a shell-free execution with read-only inputs; credentials never from files or flags; at least one implemented platform; every
    error code the adapter maps."""
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
    pl, up = _tool(contract, TOOL_PLAN), _tool(contract, TOOL_UPLOAD)
    if not pl or pl.get("network") is not False or pl.get("writes_media") is not False or pl.get("schema") != PLAN_SCHEMA or pl.get("implemented") is not True \
            or not set(PLAN_OK) <= set(pl.get("verdicts") or []):
        errs.append(f"tool {TOOL_PLAN} must be implemented, offline, write nothing, schema {PLAN_SCHEMA}, verdicts incl. {PLAN_OK}")
    if not up or up.get("role") != "execution" or up.get("writes_media") is not False or up.get("mutates_input") is not False or up.get("implemented") is not True \
            or f"[{CONFIRM_FLAG} <token>]" not in [str(a) for a in up.get("argv") or []]:
        errs.append(f"tool {TOOL_UPLOAD} must be an implemented execution tool that writes no media and takes [{CONFIRM_FLAG} <token>]")
    conf = as_dict(contract.get("confirmation"))
    if conf.get("flag") != CONFIRM_FLAG or conf.get("from_environment_or_config") is not False or TOOL_UPLOAD not in (conf.get("required_for") or []):
        errs.append(f"confirmation must require {CONFIRM_FLAG} for {TOOL_UPLOAD} and never come from the environment or a config")
    ex = as_dict(contract.get("execution"))
    if ex.get("shell") is not False or ex.get("input_is_read_only") is not True or str(ex.get("network_requires") or "") != f"{CONFIRM_FLAG} <token>":
        errs.append(f"execution must declare shell false, input_is_read_only true and network only with {CONFIRM_FLAG} <token>")
    if as_dict(contract.get("credentials")).get("from_files_or_flags") is not False or as_dict(contract.get("credentials")).get("printed") is not False:
        errs.append("credentials must never come from files or flags, and never be printed")
    if not platforms(contract):
        errs.append("no platform is implemented with an uploader")
    if not privacy_values(contract):
        errs.append(f"tool {TOOL_UPLOAD} argv lacks [--privacy ...]")
    errors = as_dict(contract.get("errors"))
    for code in ERROR_CLASSES:
        if not isinstance(as_dict(errors.get(code)).get("exit_code"), int):
            errs.append(f"errors lack {code} (with its exit_code)")
    return errs


def lifecycle(contract: Dict[str, Any]) -> Dict[str, str]:
    return {str(p.get("id")): str(p.get("lifecycle")) for p in contract.get("provides") or [] if isinstance(p, dict)}


def _op_tools(c: Dict[str, Any]) -> Dict[str, Any]:
    return dict(c, tools=[t for t in c.get("tools") or [] if isinstance(t, dict) and t.get("tool_id") in (TOOL_PLAN, TOOL_UPLOAD)])


def drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(_op_tools(live), _op_tools(pinned or pinned_contract()), DRIFT_KEYS, "tools", "tool_id", DRIFT_TOOL_KEYS)


def pinned_error_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    p, lv = as_dict((pinned or pinned_contract()).get("errors")), as_dict(live.get("errors"))
    return [f"errors {c}: pinned {json.dumps(p[c], sort_keys=True)} != live {json.dumps(lv.get(c), sort_keys=True)}" for c in sorted(p) if lv.get(c) != p[c]]


def breaking_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    """The fatal subset; additionally a pinned platform whose uploader disappeared (a plan for it could no longer run)."""
    pinned = pinned or pinned_contract()
    gone = sorted(set(platforms(pinned)) - set(platforms(live)))
    return _breaking_drift(_op_tools(live), _op_tools(pinned), FATAL_DRIFT_KEYS, "tools", "tool_id", FATAL_DRIFT_TOOL_KEYS) + pinned_error_drift(live, pinned) \
        + ([f"platforms no longer implemented: {', '.join(gone)}"] if gone else [])


PACKAGE = SkillPackage(
    skill_id=SKILL_ID, name="publish-skill", version="",
    description="Upload a finished, QC-passed deliverable with its metadata to a delivery platform (the platforms its contract implements); offline plan "
                "and dry run first, the upload only with the confirmation token a human approved; credentials from the environment or the OS keychain only",
    capabilities=["publish"],
    tools=[ToolSpec(TOOL_UPLOAD, SKILL_ID, description="Upload one deliverable (+ title / description / tags / privacy / thumbnail / captions as the platform takes "
                                                       "them) after QC and a human's confirmation token; dry run without the token",
                    inputs=["deliverable"], produces_output=False, deterministic=False, result_keys=["dry_run", "uploaded", "confirm_token", "input", "metadata", "result"])],
    repository="kajisho5/publish-skill", role="delivery (platform upload)")


class PublishAdapter(ToolAdapter):
    owns_cache = False

    def __init__(self, skill: CliSkill, workspace: Optional[str] = None, allowed_inputs: Optional[List[str]] = None, ffmpeg_skill_dir: Optional[str] = None,
                 path_policy: Optional[PathPolicy] = None):
        self.skill = skill
        self.name = SKILL_ID
        self.workspace = os.path.realpath(os.path.abspath(workspace)) if workspace else None
        self.allowed_inputs = [os.path.realpath(os.path.abspath(p)) for p in (allowed_inputs or [])]
        self.ffmpeg_skill_dir: Optional[str] = str(Path(ffmpeg_skill_dir).resolve()) if ffmpeg_skill_dir else None
        self.policy = path_policy
        self.contract = self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("incompatible publish-skill contract: " + "; ".join(errs))
        self.version = str(self.contract.get("version"))
        self.tools = {TOOL_UPLOAD: True}
        self.retryable = {c: bool(as_dict(v).get("retryable")) for c, v in as_dict(self.contract.get("errors")).items()}
        self.lifecycle = lifecycle(self.contract)
        self.platforms = platforms(self.contract)
        self._doctor: Optional[Dict[str, Any]] = None

    # ---- contract / doctor
    def _call(self, argv: List[str], timeout: Optional[float]) -> Tuple[int, str, str]:
        return invoke(self.skill, argv, timeout=timeout)

    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = self._call(["contract", "--json"], 60.0)
        if code != 0:
            raise ContractError(f"publish contract --json failed (exit {code}): {err.strip()[-300:]}")
        return one_json_document(out, "publish contract")

    def doctor(self) -> Dict[str, Any]:
        """`publish doctor --json --offline [--ffmpeg-skill=<dir>]`: which platforms are implemented and whether their credentials are present
        (never their values)."""
        if self._doctor is not None:
            return self._doctor
        code, out, err = self._call(["doctor", "--json", "--offline"] + ([f"--ffmpeg-skill={self.ffmpeg_skill_dir}"] if self.ffmpeg_skill_dir else []), 120.0)
        try:
            doc = one_json_document(out, "publish doctor")
        except ToolError as e:
            doc = {"schema": DOCTOR_SCHEMA, "ok": False, "status": "fail", "checks": {}, "problems": [f"doctor produced no document (exit {code}): {e} {err.strip()[-200:]}"]}
        if doc.get("schema") != DOCTOR_SCHEMA or doc.get("status") not in ("ok", "degraded", "fail"):
            doc = dict(doc, status="fail", problems=list(doc.get("problems") or []) + [f"doctor document {doc.get('schema')!r} / status {doc.get('status')!r} not understood"])
        self._doctor = dict(doc, problems=list(doc.get("problems") or []), warnings=list(doc.get("warnings") or []))
        return self._doctor

    def platform_status(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """`publish:<platform>` verdicts: supported when the pinned contract implements it and the doctor lists it implemented (credentials are
        reported, not required here: they are the user's, checked by the Skill at upload time)."""
        rows = as_dict(as_dict((doc or self.doctor()).get("checks")).get("platforms"))
        return {pid.upper(): ("supported" if as_dict(rows.get(pid)).get("implemented") is True else "unavailable") for pid in self.platforms}

    def platform_evidence(self, doc: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        rows = as_dict(as_dict((doc or self.doctor()).get("checks")).get("platforms"))
        out: Dict[str, Dict[str, Any]] = {}
        for pid, p in self.platforms.items():
            cred = as_dict(as_dict(rows.get(pid)).get("credentials"))
            out[pid.upper()] = {"platform": pid, "metadata_fields": list(p["metadata_fields"]), "privacy": list(p["privacy"]),
                                "credentials_complete": cred.get("complete") is True, "credential_env": list(p["credential_env"]), "lifecycle": self.lifecycle.get("publish.upload")}
        return out

    def drift(self) -> List[str]:
        return drift(self.contract)

    def breaking_drift(self) -> List[str]:
        return breaking_drift(self.contract)

    # ---- ToolAdapter
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "skill": self.skill.describe(), "tools": list(TOOLS), "platforms": sorted(self.platforms), "drift": self.drift()}

    def package(self) -> SkillPackage:
        return SkillPackage(PACKAGE.skill_id, PACKAGE.name, self.version, PACKAGE.description, list(PACKAGE.capabilities),
                            [ToolSpec(**{**t.__dict__, "version": self.version}) for t in PACKAGE.tools], PACKAGE.repository, PACKAGE.role)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        raise ToolError("publish: no measurement tools")

    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        """Never part of a compiled pipeline: an upload cannot be undone, so it runs only through `publish()` with a human's token."""
        return ToolResult(op.id, op.tool, False, 2, None, {"skill": {"id": SKILL_ID, "version": self.version}, "status": "failed",
                                                           "error": {"code": "INVALID_REQUEST", "message": "publish/upload is never run as a pipeline operation",
                                                                     "retryable": False, "details": {}, "exit_code": 2, "recovery_class": "INVALID_ARGS"}},
                          [], "publish/upload is never run as a pipeline operation", 0.0, attempt, dry_run)

    # ---- request construction
    def _within(self, root: str, p: str) -> bool:
        r, q = os.path.normcase(root), os.path.normcase(p)
        return q == r or q.startswith(r.rstrip(os.sep) + os.sep)

    def _file(self, raw: Any, what: str) -> str:
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw or "\n" in raw:
            raise ToolError(f"{what} must be a non-empty single-line path")
        if any(part == ".." for part in raw.replace("\\", "/").split("/")):
            raise ToolError(f"{what} path contains '..' (traversal)")
        resolved = os.path.realpath(os.path.abspath(raw))
        roots = list(self.allowed_inputs) + ([self.workspace] if self.workspace else [])
        if roots and not any(self._within(r, resolved) for r in roots):
            raise ToolError(f"{what} is outside the workspace and the allowed input roots: {os.path.basename(raw)}")
        if not os.path.isfile(resolved):
            raise ToolError(f"{what} not found: {os.path.basename(raw)}")
        if self.policy is not None:
            self.policy.check_input(resolved)
        return resolved

    @staticmethod
    def _text(v: Any, name: str) -> str:
        if not isinstance(v, str) or "\x00" in v or len(v) > MAX_TEXT:
            raise ToolError(f"{name} must be text of at most {MAX_TEXT} characters")
        return v

    def metadata_argv(self, platform: str, meta: Dict[str, Any]) -> List[str]:
        """The metadata options, each one `--flag=value` word, for the fields the platform takes (others refused here, as the Skill would)."""
        if platform not in self.platforms:
            raise ToolError(f"platform {platform!r} is not implemented by publish-skill {self.version} (implemented: {', '.join(sorted(self.platforms))})")
        hit = scan_forbidden(meta, FORBIDDEN_ARG_KEYS)
        if hit:
            raise ToolError(f"forbidden key {hit}")
        allowed = {"title", "description", "tags", "privacy", "thumbnail", "captions"}
        unknown = sorted(k for k in meta if k not in allowed)
        if unknown:
            raise ToolError(f"unknown metadata field(s): {', '.join(unknown)}")
        fields = set(self.platforms[platform]["metadata_fields"])
        given = {k for k, v in meta.items() if v not in (None, [], "")}
        for k in sorted(given):
            key = "tags" if k == "tags" else k
            if key not in fields:
                raise ToolError(f"{platform} does not take {k} (it takes {', '.join(sorted(fields))})")
        argv: List[str] = []
        if meta.get("title") not in (None, ""):
            argv.append(f"--title={self._text(meta['title'], 'title')}")
        if meta.get("description") not in (None, ""):
            argv.append(f"--description={self._text(meta['description'], 'description')}")
        for t in meta.get("tags") or []:
            argv.append(f"--tag={self._text(t, 'tag')}")
        if meta.get("privacy") is not None:
            if meta["privacy"] not in privacy_values(self.contract) or (self.platforms[platform]["privacy"] and meta["privacy"] not in self.platforms[platform]["privacy"]):
                raise ToolError(f"privacy {meta['privacy']!r} is not one {platform} takes ({', '.join(self.platforms[platform]['privacy'] or privacy_values(self.contract))})")
            argv.append(f"--privacy={meta['privacy']}")
        if meta.get("thumbnail"):
            argv.append(f"--thumbnail={self._file(meta['thumbnail'], 'thumbnail')}")
        for c in meta.get("captions") or []:
            c = as_dict(c)
            if not LANG_RE.match(str(c.get("language") or "")):
                raise ToolError(f"caption language {c.get('language')!r} is not a BCP-47 tag")
            argv += [f"--caption={self._file(c.get('path'), 'caption')}", f"--caption-language={c['language']}"]
        return argv

    # ---- calls
    def _run(self, argv: List[str], timeout: float, what: str) -> Dict[str, Any]:
        """One call → the Skill's document, or {"ok": False, "error": {...}} with the agent's recovery class (never raised)."""
        code, out, err = self._call(argv, timeout)
        if code == 124:
            return {"ok": False, "error": {"code": "TIMEOUT", "message": f"publish {what} exceeded {timeout:g}s", "retryable": False, "details": {}, "exit_code": 124,
                                           "recovery_class": "TIMEOUT"}}
        try:
            doc = one_json_document(out, f"publish {what}")
        except ToolError as e:
            return {"ok": False, "error": {"code": "INVALID_OUTPUT", "message": f"{e} (exit {code}) {err.strip()[-300:]}", "retryable": False, "details": {}, "exit_code": code,
                                           "recovery_class": "SKILL_ERROR"}}
        if isinstance(doc.get("error"), dict):
            e = as_dict(doc["error"])
            c = str(e.get("code") or "INTERNAL_ERROR")
            doc = dict(doc, ok=False, error={"code": c, "message": str(e.get("message") or "")[:1200], "retryable": bool(e.get("retryable")) if isinstance(e.get("retryable"), bool) else self.retryable.get(c, False),
                                             "details": scrub(e.get("details"), FORBIDDEN_ARG_KEYS), "exit_code": code, "recovery_class": recovery_class(c)})
        return doc

    def plan(self, deliverable: str, platform: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """`publish plan` (offline): the verdict and its rows; nothing uploaded."""
        f = self._file(deliverable, "deliverable")
        argv = ["plan", "--json", f"--platform={platform}"] + ([f"--ffmpeg-skill={self.ffmpeg_skill_dir}"] if self.ffmpeg_skill_dir else []) + self.metadata_argv(platform, meta) + ["--", f]
        doc = self._run(argv, DEFAULT_TIMEOUT, "plan")
        if doc.get("error") is None and (doc.get("schema") != PLAN_SCHEMA or doc.get("uploaded") is not False or doc.get("network_used") is not False):
            return {"ok": False, "error": {"code": "INVALID_OUTPUT", "message": "publish plan result is not publish/plan@1 without network use", "retryable": False, "details": {},
                                           "exit_code": 9, "recovery_class": "SKILL_ERROR"}}
        return doc

    def dry_run(self, deliverable: str, platform: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """`publish upload` without a token: the confirm_token for exactly this file, metadata and platform; no network, nothing uploaded. The
        reported input sha256 must be the file's own."""
        f = self._file(deliverable, "deliverable")
        doc = self._run(["upload", "--json", f"--platform={platform}"] + self.metadata_argv(platform, meta) + ["--", f], DEFAULT_TIMEOUT, "upload (dry run)")
        if doc.get("error") is None:
            why = []
            if doc.get("dry_run") is not True or doc.get("uploaded") is not False or doc.get("network_used") is not False:
                why.append("the dry run must not upload or use the network")
            if not TOKEN_RE.match(str(doc.get("confirm_token") or "")):
                why.append("no confirm_token")
            if str(as_dict(doc.get("input")).get("sha256") or "") != sha256_file(f):
                why.append("the dry run's input sha256 is not the deliverable's")
            if why:
                return {"ok": False, "error": {"code": "INVALID_OUTPUT", "message": "; ".join(why), "retryable": False, "details": {}, "exit_code": 9, "recovery_class": "SKILL_ERROR"}}
        return doc

    def upload(self, deliverable: str, platform: str, meta: Dict[str, Any], token: str) -> Dict[str, Any]:
        """`publish upload --confirm=<token>`: the only network use. Sent once; the caller never retries it."""
        if not isinstance(token, str) or not TOKEN_RE.match(token):
            raise ToolError("the confirmation token must be the confirm-… token of the dry run")
        f = self._file(deliverable, "deliverable")
        t0 = time.time()
        doc = self._run(["upload", "--json", f"--platform={platform}"] + self.metadata_argv(platform, meta) + [f"{CONFIRM_FLAG}={token}", "--", f], UPLOAD_TIMEOUT, "upload")
        doc = dict(doc, seconds=round(time.time() - t0, 3))
        if doc.get("error") is None and (doc.get("uploaded") is not True or doc.get("confirm_token") != token or not isinstance(doc.get("result"), dict)):
            return dict(doc, ok=False, error={"code": "INVALID_OUTPUT", "message": "an upload result without uploaded / its token / result", "retryable": False, "details": {},
                                              "exit_code": 9, "recovery_class": "SKILL_ERROR"})
        return doc
