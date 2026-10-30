"""IngestAdapter: the agent's boundary to ingest-skill's verified card offload (ADR-066, roadmap step 9; ingest-skill roadmap step 6).

    video-production-agent ─(typed Operation)─→ IngestAdapter ─(argv list, never a shell)─→ `ingest offload <card> --dest A --dest B ... --json`
        ─(each file of the card read once, written to every destination, each copy read back)─→ <dest>/<project>/<card folder>/...
          + ingest-manifest.json + an ASC MHL generation (ascmhl/) per destination

Protocol (ingest-skill's own contract, `ingest contract --json`, pinned in contract_0.1.0.json):
    contract  → `ingest contract --json`
    doctor    → `ingest doctor --json`  (ingest/doctor@1: python / sha256 / c4 / xxhash / ffprobe / yt-dlp checks, hash_algorithms)
    offload   → `ingest offload <source> --dest D [--dest D ...] [--project P] --hash sha256|xxh64 --name-by folder|reel [--reel R] [--dry-run] --json`
                exit 0 + ingest/offload-report@1 (ok), exit 3 + the same report with ok false (a conflict, a failed copy, a damaged ASC MHL
                history), exit 2 + {"ok": false, "error": {code, message}} (refused, nothing written)
    inventory → `ingest inventory <folder> [--reel R] --json` → ingest/inventory@2 (reads only; one clip per recording)
    verify    → `ingest verify <folder> --json` → ingest/verify-report@1 (reads only; exit 3 with the report when it is not ok)

The agent hands one typed operation (`ingest/offload`: source, destinations, project, hash, name_by, reel, dry_run) or a measurement
(`ingest/inventory` / `ingest/verify`: target, reel). Every value is checked before the process starts: absolute paths, the card a
directory, every destination an existing directory (an unmounted drive's mount point is never created on the system disk), no
destination twice or inside the card, the hash one the contract's manifest declares, names in ingest-skill's own safe-name pattern.
The report is admitted only when it is about this card, these destinations (each copy's folder under the destination it was asked
for, in order) and this hash. ingest-skill's error codes are kept verbatim and mapped onto the agent's failure kinds and recovery
classes (ERROR_KINDS); an incomplete report keeps every destination's counts, its failed and conflicting files and its problems."""
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
from ..skill_process import (FORBIDDEN_ARG_KEYS, CliSkill, ContractError, as_dict, breaking_drift as _breaking_drift, drift_report, invoke, one_json_document,
                             same_file, scan_forbidden, scrub)
from .locate import PACKAGE_ID, locate_ingest  # noqa: F401  (re-exported)

SKILL_ID = PACKAGE_ID
PREFIX = SKILL_ID + "/"
TOOL_OFFLOAD = "ingest/offload"
TOOL_INVENTORY = "ingest/inventory"
TOOL_VERIFY = "ingest/verify"
TOOLS = (TOOL_OFFLOAD, TOOL_INVENTORY, TOOL_VERIFY)
SUPPORTED_SKILL_VERSIONS = ("0.1.",)
REPORT_SCHEMA = "ingest/offload-report@1"
INVENTORY_SCHEMA = "ingest/inventory@2"
VERIFY_SCHEMA = "ingest/verify-report@1"
DOCTOR_SCHEMA = "ingest/doctor@1"
MANIFEST_SCHEMA = "ingest/manifest@1"
MANIFEST_NAME = "ingest-manifest.json"
NAME_BY = ("folder", "reel")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")   # ingest-skill's reel pattern (its ADR-003); used for --project too, so it is one folder name
OFFLOAD_ARGS = ("source", "destinations", "project", "hash", "name_by", "reel", "dry_run")
MEASURE_ARGS = ("target", "reel")
DEFAULT_TIMEOUT = None          # a card is as big as it is: no default bound on the copy (render --timeout sets one)
MEASURE_TIMEOUT = 1800.0        # inventory (ffprobe per clip) / verify (re-reads a copy)
# ingest-skill's error codes (its contract `errors`) → (the agent's failure kind, recovery class). The code itself is kept verbatim in
# the result; the kind is the agent's word for it; the class decides recovery (execution/recovery.py: TOOL_MISSING / INPUT_MISSING /
# INVALID_ARGS block, SKILL_ERROR blocks without a retry). Nothing an offload reports is retried automatically: re-running a copy to the
# user's drives is the user's call (ingest-skill resumes it: identical files are skipped, nothing is overwritten).
ERROR_KINDS: Dict[str, Tuple[str, str]] = {
    "SOURCE_NOT_FOUND": ("SOURCE_MISSING", "INPUT_MISSING"),
    "NO_DESTINATION": ("INVALID_REQUEST", "INVALID_ARGS"),
    "DUPLICATE_DESTINATION": ("INVALID_REQUEST", "INVALID_ARGS"),
    "DESTINATION_OVERLAPS_SOURCE": ("INVALID_REQUEST", "INVALID_ARGS"),
    "REEL_INVALID": ("INVALID_REQUEST", "INVALID_ARGS"),
    "NAME_INVALID": ("INVALID_REQUEST", "INVALID_ARGS"),
    "URL_INVALID": ("INVALID_REQUEST", "INVALID_ARGS"),
    "HASH_UNAVAILABLE": ("ENGINE_MISSING", "TOOL_MISSING"),
    "FETCH_UNAVAILABLE": ("ENGINE_MISSING", "TOOL_MISSING"),
    "MANIFEST_NOT_FOUND": ("INCOMPLETE", "SKILL_ERROR"),
    "MANIFEST_INVALID": ("INCOMPLETE", "SKILL_ERROR"),
    "FETCH_FAILED": ("INCOMPLETE", "SKILL_ERROR"),
    "INCOMPLETE": ("INCOMPLETE", "SKILL_ERROR"),
    "SOURCE_CHANGED": ("SOURCE_CHANGED", "SKILL_ERROR"),
    "VERIFY_MISMATCH": ("VERIFY_FAILED", "SKILL_ERROR"),
    "FILE_MISSING": ("INCOMPLETE", "SKILL_ERROR"),
    "MHL_INVALID": ("HISTORY_DAMAGED", "SKILL_ERROR"),
}
# the agent's own refusals / admission failures (not ingest-skill codes)
AGENT_ERRORS: Dict[str, Tuple[str, str]] = {"INVALID_REQUEST": ("INVALID_REQUEST", "INVALID_ARGS"), "INVALID_RESULT": ("INVALID_OUTPUT", "SKILL_ERROR"),
                                            "TIMEOUT": ("TIMEOUT", "TIMEOUT")}
# the order a report with several problems is named in (the first is the result's kind; all are kept in details.kinds)
KIND_ORDER = ("HISTORY_DAMAGED", "SOURCE_CHANGED", "VERIFY_FAILED", "CONFLICT", "PARTIAL", "COPY_FAILED", "INCOMPLETE")
FAILURE_KINDS = tuple(dict.fromkeys([k for k, _ in ERROR_KINDS.values()] + [k for k, _ in AGENT_ERRORS.values()] + list(KIND_ORDER)))
# contract drift (ADR-045 / ADR-047): every key the adapter reads is compared with the pinned contract; the fatal subset gates availability
DRIFT_KEYS = ("skill_id", "version", "contract_version", "execution", "manifest", "asc_mhl", "inventory", "errors", "capabilities", "provides", "not_provided", "fetch")
DRIFT_TOOL_KEYS = ("role", "writes_media", "mutates_input", "deterministic", "network", "argv")
# version: moves within SUPPORTED_SKILL_VERSIONS; capabilities / provides / not_provided / asc_mhl / fetch: read for reporting only (the URL
# fetch path, agent/ingest.py, checks its own schemas); execution / errors: checked on every use (check_contract, pinned_error_drift), so
# an added error code or network tool does not turn the offload off; what is left changes the wire or the documents the adapter admits.
FATAL_DRIFT_KEYS = ("skill_id", "contract_version", "manifest", "inventory")
FATAL_DRIFT_TOOL_KEYS = ("role", "writes_media", "mutates_input", "argv")
PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_0.1.0.json")


def pinned_contract() -> Dict[str, Any]:
    return json.loads(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"))


def kind_of(code: str) -> Tuple[str, str]:
    """(failure kind, recovery class) of an ingest-skill code or an agent code; an unknown code is INCOMPLETE / SKILL_ERROR (kept verbatim)."""
    return ERROR_KINDS.get(code) or AGENT_ERRORS.get(code) or ("INCOMPLETE", "SKILL_ERROR")


def check_contract(contract: Any) -> List[str]:
    """What the adapter relies on, refused (never patched) when it does not hold: identity, contract and Skill version, the three
    tools with their roles, a shell-free execution that never overwrites and never writes the source, the manifest / inventory
    schemas, and every error code the adapter maps with its exit code."""
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
    want = {TOOL_OFFLOAD: ("execution", True), TOOL_INVENTORY: ("measurement", False), TOOL_VERIFY: ("measurement", False)}
    for tid, (role, writes) in want.items():
        t = tools.get(tid)
        if t is None:
            errs.append(f"contract lacks tool {tid} (declares {sorted(tools)})")
            continue
        if t.get("role") != role or t.get("writes_media") is not writes or t.get("mutates_input") is not False or t.get("network") is not False:
            errs.append(f"tool {tid} must declare role {role}, writes_media {writes}, mutates_input false, network false")
    ex = as_dict(contract.get("execution"))
    if ex.get("shell") is not False or ex.get("source_is_read_only") is not True or not str(ex.get("overwrites") or "").startswith("never"):
        errs.append("execution must declare shell false, source_is_read_only true and overwrites never")
    if TOOL_OFFLOAD in (ex.get("network_tools") or []):
        errs.append("execution.network_tools includes ingest/offload (an offload is local)")
    man = as_dict(contract.get("manifest"))
    if man.get("name") != MANIFEST_NAME or man.get("schema") != MANIFEST_SCHEMA or "sha256" not in (man.get("hash") or []):
        errs.append(f"manifest must be {MANIFEST_NAME} / {MANIFEST_SCHEMA} with sha256")
    if as_dict(contract.get("inventory")).get("schema") != INVENTORY_SCHEMA:
        errs.append(f"inventory.schema {as_dict(contract.get('inventory')).get('schema')!r} != {INVENTORY_SCHEMA}")
    errors = as_dict(contract.get("errors"))
    for code in ERROR_KINDS:
        if code not in errors:
            errs.append(f"errors lack {code}")
    return errs


def drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    return drift_report(live, pinned or pinned_contract(), DRIFT_KEYS, "tools", "tool_id", DRIFT_TOOL_KEYS)


def _offload_tools(c: Dict[str, Any]) -> Dict[str, Any]:
    """The contract with only the tools this adapter uses (a change to `ingest/fetch` is the URL path's business, ADR-059)."""
    return dict(c, tools=[t for t in c.get("tools") or [] if isinstance(t, dict) and t.get("tool_id") in TOOLS])


def pinned_error_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    """A pinned error code that disappeared or changed its exit code / retryable verdict (a new code is additive: mapped as INCOMPLETE)."""
    p, lv = as_dict((pinned or pinned_contract()).get("errors")), as_dict(live.get("errors"))
    return [f"errors {c}: pinned {json.dumps(p[c], sort_keys=True)} != live {json.dumps(lv.get(c), sort_keys=True)}" for c in sorted(p) if lv.get(c) != p[c]]


def breaking_drift(live: Dict[str, Any], pinned: Optional[Dict[str, Any]] = None) -> List[str]:
    pinned = pinned or pinned_contract()
    return _breaking_drift(_offload_tools(live), _offload_tools(pinned), FATAL_DRIFT_KEYS, "tools", "tool_id", FATAL_DRIFT_TOOL_KEYS) + pinned_error_drift(live, pinned)


PACKAGE = SkillPackage(
    skill_id=SKILL_ID, name="ingest-skill", version="",
    description="Verified offload of camera cards / recorders: a sha256 (or xxh64) read-back of every copy, never an overwrite, ingest-manifest.json "
                "and an ASC MHL history per destination, card-structure-aware clip inventory",
    capabilities=["ingest"],
    tools=[ToolSpec(TOOL_OFFLOAD, SKILL_ID, description="Copy every file of a card folder to each destination, verified by read-back (writes the user's drives)",
                    inputs=["source"], produces_output=True, deterministic=True, result_keys=["report"]),
           ToolSpec(TOOL_INVENTORY, SKILL_ID, description="Clip list of a card or an offloaded folder: one entry per recording, spanned members in order, sidecars, reel",
                    inputs=["target"], produces_output=False, deterministic=True, result_keys=["inventory"]),
           ToolSpec(TOOL_VERIFY, SKILL_ID, description="Re-hash an offloaded folder against its manifest and ASC MHL history",
                    inputs=["target"], produces_output=False, deterministic=True, result_keys=["verify"])],
    repository="kajisho5/ingest-skill", role="verified media offload (DIT)")


def _is_abs_dir(p: Any) -> bool:
    return isinstance(p, str) and bool(p) and os.path.isabs(p) and os.path.isdir(p)


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def offload_argv(args: Dict[str, Any], hashes: Tuple[str, ...] = ("sha256", "xxh64")) -> List[str]:
    """The argv of `ingest offload` for typed args, or ToolError naming the first value that cannot be sent."""
    unknown = sorted(k for k in args if k not in OFFLOAD_ARGS)
    if unknown:
        raise ToolError(f"unknown offload argument(s): {', '.join(unknown)}")
    hit = scan_forbidden(args, FORBIDDEN_ARG_KEYS[:-2])   # "path" / "paths" are not keys here; everything command-like is refused by name
    if hit:
        raise ToolError(f"forbidden key {hit}")
    src = args.get("source")
    if not _is_abs_dir(src):
        raise ToolError(f"source {src!r} is not an absolute path to a directory (the card folder)")
    dests = args.get("destinations")
    if not isinstance(dests, list) or not dests:
        raise ToolError("destinations must be a non-empty list (offload copies only where it is told)")
    resolved: List[Path] = []
    for d in dests:
        if not _is_abs_dir(d):
            raise ToolError(f"destination {d!r} is not an absolute path to an existing directory (a drive that is not mounted is never created)")
        r = Path(d).resolve()
        if r in resolved:
            raise ToolError(f"destination {d} given twice")
        s = Path(src).resolve()
        if _inside(r, s) or _inside(s, r):
            raise ToolError(f"destination {d} and the card {src} contain one another")
        resolved.append(r)
    h = args.get("hash", "sha256")
    if h not in hashes:
        raise ToolError(f"hash {h!r} is not one of {', '.join(hashes)}")
    nb = args.get("name_by", "folder")
    if nb not in NAME_BY:
        raise ToolError(f"name_by {nb!r} is not one of {', '.join(NAME_BY)}")
    for k in ("project", "reel"):
        v = args.get(k)
        if v is not None and (not isinstance(v, str) or not NAME_RE.match(v)):
            raise ToolError(f"{k} {v!r} must be one folder name: 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit")
    if not isinstance(args.get("dry_run", False), bool):
        raise ToolError("dry_run must be true or false")
    argv = ["offload", str(Path(src).resolve())]
    for r in resolved:
        argv += ["--dest", str(r)]
    if args.get("project"):
        argv += ["--project", args["project"]]
    argv += ["--hash", h, "--name-by", nb]
    if args.get("reel"):
        argv += ["--reel", args["reel"]]
    if args.get("dry_run"):
        argv.append("--dry-run")
    return argv + ["--json"]


def planned_roots(args: Dict[str, Any], name: str) -> List[str]:
    """<dest>/<project>/<name> per destination, in order: where ingest-skill puts the card folder (its offload())."""
    out = []
    for d in args["destinations"]:
        base = Path(d).resolve()
        out.append(str((base / args["project"] / name) if args.get("project") else (base / name)))
    return out


def admit_report(doc: Dict[str, Any], args: Dict[str, Any]) -> List[str]:
    """Why an offload report is not about this request (empty = admitted): schema, mode, card, hash, and every destination's copy
    folder under the destination asked for, in the same order, named after the card the way --name-by asked."""
    errs: List[str] = []
    if doc.get("schema") != REPORT_SCHEMA:
        return [f"report schema {doc.get('schema')!r} != {REPORT_SCHEMA}"]
    if bool(doc.get("dry_run")) != bool(args.get("dry_run")):
        errs.append(f"report dry_run {doc.get('dry_run')!r} but {'a dry run' if args.get('dry_run') else 'a copy'} was asked")
    if not same_file(str(doc.get("source") or ""), args["source"]):
        errs.append(f"report is about {doc.get('source')!r}, not the card {args['source']}")
    if doc.get("hash") != args.get("hash", "sha256"):
        errs.append(f"report hash {doc.get('hash')!r} != {args.get('hash', 'sha256')}")
    dests = doc.get("destinations")
    if not isinstance(dests, list) or len(dests) != len(args["destinations"]) or not all(isinstance(d, dict) for d in dests):
        return errs + [f"report names {len(dests) if isinstance(dests, list) else 'no'} destination(s), {len(args['destinations'])} asked"]
    reel = as_dict(as_dict(doc.get("card")).get("reel"))
    name = str(reel.get("id") or "") if args.get("name_by") == "reel" else Path(args["source"]).resolve().name
    for want, d in zip(planned_roots(args, name), dests):
        if not same_file(str(d.get("root") or ""), want) and os.path.normcase(os.path.abspath(str(d.get("root") or ""))) != os.path.normcase(want):
            errs.append(f"destination root {d.get('root')!r} != the planned {want}")
    return errs


def destination_summary(d: Dict[str, Any], files: List[Dict[str, Any]], read_manifest: bool = True) -> Dict[str, Any]:
    """One destination of a report, as the agent records it: counts, manifest / ASC MHL generation, problems (with ingest-skill's
    codes), warnings, and the files that failed there (from that destination's manifest `failed` list, which ingest-skill writes
    per destination) and the conflicting files (never overwritten)."""
    out = {k: d.get(k) for k in ("root", "copied", "skipped_identical", "conflicts", "failed", "mhl_failed", "manifest", "mhl")}
    out["problems"] = [{"code": str(p.get("code") or ""), "detail": str(p.get("detail") or "")[:600]} for p in d.get("problems") or [] if isinstance(p, dict)]
    out["warnings"] = [str(w)[:400] for w in d.get("warnings") or []]
    failed_files: List[Dict[str, Any]] = []
    if read_manifest and d.get("failed") and d.get("manifest"):
        try:
            man = json.loads(Path(str(d["manifest"])).read_text(encoding="utf-8"))
            failed_files = [{"path": str(e.get("path")), "size": e.get("size"), "detail": str(e.get("detail") or "")[:600]} for e in man.get("failed") or [] if isinstance(e, dict)]
        except (OSError, ValueError, AttributeError):
            failed_files = []
    out["failed_files"] = failed_files
    out["conflict_files"] = [str(f.get("path")) for f in files if f.get("status") == "conflict"] if d.get("conflicts") else []
    out["ok"] = not d.get("failed") and not d.get("conflicts") and not d.get("mhl_failed") and not out["problems"]
    return out


def classify_report(dests: List[Dict[str, Any]]) -> List[str]:
    """The agent's failure kinds of an incomplete report, most serious first (KIND_ORDER): damaged ASC MHL history, conflicts, a
    destination that failed while another is complete (PARTIAL) or every destination failing (COPY_FAILED), and the kind of every
    other problem code. Never read from free text: only counts and ingest-skill's codes."""
    kinds = set()
    for d in dests:
        if d.get("mhl_failed"):
            kinds.add("HISTORY_DAMAGED")
        for p in d.get("problems") or []:
            kinds.add(kind_of(p["code"])[0])
        if d.get("conflicts"):
            kinds.add("CONFLICT")
    failed = [d for d in dests if d.get("failed")]
    if failed:
        kinds.add("PARTIAL" if len(failed) < len(dests) else "COPY_FAILED")
    return [k for k in KIND_ORDER if k in kinds] + sorted(k for k in kinds if k not in KIND_ORDER) or ["INCOMPLETE"]


def verify_failures(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """`ingest verify` failures with the agent's kind beside ingest-skill's code (VERIFY_MISMATCH → VERIFY_FAILED, FILE_MISSING /
    INCOMPLETE → INCOMPLETE, a damaged ASC MHL history → HISTORY_DAMAGED); expected and actual hashes kept."""
    out = []
    for f in doc.get("failures") or []:
        if isinstance(f, dict):
            code = str(f.get("code") or "")
            out.append(dict(f, kind=kind_of(code)[0] if code else "HISTORY_DAMAGED"))
    return out


class IngestAdapter(ToolAdapter):
    owns_cache = False

    def __init__(self, skill: CliSkill, timeout: Optional[float] = DEFAULT_TIMEOUT, measure_timeout: float = MEASURE_TIMEOUT):
        self.skill = skill
        self.timeout = timeout
        self.measure_timeout = measure_timeout
        self.name = SKILL_ID
        self.contract = self._fetch_contract()
        errs = check_contract(self.contract)
        if errs:
            raise ContractError("incompatible ingest-skill contract: " + "; ".join(errs))
        self.version = str(self.contract.get("version"))
        self.tools = {t: True for t in TOOLS}
        self.retryable = {c: bool(as_dict(v).get("retryable")) for c, v in as_dict(self.contract.get("errors")).items()}
        self.hashes = tuple(str(h) for h in as_dict(self.contract.get("manifest")).get("hash") or ["sha256"])
        self._doctor: Optional[Dict[str, Any]] = None

    # ---- contract / doctor
    def _fetch_contract(self) -> Dict[str, Any]:
        code, out, err = invoke(self.skill, ["contract", "--json"], timeout=60.0)
        if code != 0:
            raise ContractError(f"ingest contract --json failed (exit {code}): {err.strip()[-300:]}")
        return one_json_document(out, "ingest contract")

    def doctor(self) -> Dict[str, Any]:
        """`ingest doctor --json`, with `status`: ok (every check ok), degraded (usable; an optional part such as xxhash, ffprobe or
        yt-dlp is unavailable), fail (a required check failed or no document)."""
        if self._doctor is not None:
            return self._doctor
        code, out, err = invoke(self.skill, ["doctor", "--json"], timeout=120.0)
        try:
            doc = one_json_document(out, "ingest doctor")
        except ToolError as e:
            doc = {"schema": DOCTOR_SCHEMA, "ok": False, "checks": {}, "problems": [f"doctor produced no document (exit {code}): {e} {err.strip()[-200:]}"]}
        checks = as_dict(doc.get("checks"))
        if doc.get("schema") != DOCTOR_SCHEMA or not doc.get("ok"):
            status = "fail"
        else:
            status = "ok" if all(as_dict(c).get("status") == "ok" for c in checks.values()) else "degraded"
        doc = dict(doc, status=status, problems=list(doc.get("problems") or []) + [f"{k}: {as_dict(c).get('detail') or as_dict(c).get('status')}" for k, c in checks.items() if as_dict(c).get("status") == "fail"],
                   warnings=[f"{k}: {as_dict(c).get('detail') or as_dict(c).get('status')}" for k, c in checks.items() if as_dict(c).get("status") not in ("ok", "fail")])
        self._doctor = doc
        return doc

    def hash_available(self, h: str) -> bool:
        return as_dict(self.doctor().get("hash_algorithms")).get(h) == "ok"

    def drift(self) -> List[str]:
        return drift(self.contract)

    def breaking_drift(self) -> List[str]:
        """ADR-048: the subset of drift() that gates availability (FATAL_DRIFT_KEYS, the offload tools' wire, the pinned error codes)."""
        return breaking_drift(self.contract)

    # ---- ToolAdapter
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "skill": self.skill.describe(), "tools": list(TOOLS), "hashes": list(self.hashes), "drift": self.drift()}

    def package(self) -> SkillPackage:
        return SkillPackage(PACKAGE.skill_id, PACKAGE.name, self.version, PACKAGE.description, list(PACKAGE.capabilities),
                            [ToolSpec(**{**t.__dict__, "version": self.version}) for t in PACKAGE.tools], PACKAGE.repository, PACKAGE.role)

    def supports(self, tool: str) -> bool:
        return tool in self.tools

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        try:
            argv = offload_argv(op.args, self.hashes) if op.tool == TOOL_OFFLOAD else self._measure_argv(op.tool, op.args)
        except ToolError as e:
            return [f"<refused: {e}>"]
        return [" ".join(["ingest"] + argv)]

    def _measure_argv(self, tool: str, args: Dict[str, Any]) -> List[str]:
        unknown = sorted(k for k in args if k not in MEASURE_ARGS)
        if unknown:
            raise ToolError(f"unknown {tool} argument(s): {', '.join(unknown)}")
        if not _is_abs_dir(args.get("target")):
            raise ToolError(f"target {args.get('target')!r} is not an absolute path to a directory")
        argv = ["inventory" if tool == TOOL_INVENTORY else "verify", str(Path(args["target"]).resolve())]
        if args.get("reel") is not None:
            if tool != TOOL_INVENTORY or not isinstance(args["reel"], str) or not NAME_RE.match(args["reel"]):
                raise ToolError(f"reel {args['reel']!r} is not a reel id (inventory only)")
            argv += ["--reel", args["reel"]]
        return argv + ["--json"]

    def _fail(self, op: Operation, attempt: int, dry_run: bool, t0: float, exit_code: int, code: str, message: str, details: Optional[Dict[str, Any]] = None,
              extra: Optional[Dict[str, Any]] = None) -> ToolResult:
        kind, cls = kind_of(code)
        det = dict(details or {})
        det.setdefault("kind", kind)
        det.setdefault("kinds", [det["kind"]])
        retry = self.retryable.get(code, False)
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, "status": "failed",
                                "error": {"code": code, "message": message[:1200], "retryable": bool(retry), "kind": det["kind"], "details": det, "exit_code": exit_code,
                                          "recovery_class": cls}}
        data.update(extra or {})
        return ToolResult(op.id, op.tool, False, exit_code, None, data, [], f"{SKILL_ID} [{code}/{det['kind']}] {message[:600]}", round(time.time() - t0, 3), attempt, dry_run)

    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        t0 = time.time()
        if op.tool == TOOL_OFFLOAD:
            return self._offload(op, timeout, dry_run, attempt, t0)
        if op.tool in (TOOL_INVENTORY, TOOL_VERIFY):
            return self._measure(op, timeout, dry_run, attempt, t0)
        raise ToolError(f"{SKILL_ID} adapter does not support {op.tool}")

    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        """inventory / verify, or an offload with dry_run true (lists what would be copied; writes nothing). A copy is never a measurement."""
        if tool == TOOL_OFFLOAD and args.get("dry_run") is not True:
            raise ToolError("ingest/offload is measured only as a dry run; the copy is an approved operation")
        return self.run(Operation(tool=tool, args=args, inputs=[], outputs=[], kind="measure"), paths or {}, timeout=timeout)

    def _call(self, argv: List[str], timeout: Optional[float]) -> Tuple[int, str, str]:
        return invoke(self.skill, argv, timeout=timeout)

    def _offload(self, op: Operation, timeout: Optional[float], dry_run: bool, attempt: int, t0: float) -> ToolResult:
        args = dict(op.args)
        try:
            argv = offload_argv(args, self.hashes)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e))
        if dry_run and not args.get("dry_run"):   # the executor's dry run never copies: it is ingest-skill's own --dry-run
            argv.insert(-1, "--dry-run")
            args["dry_run"] = True
        code, out, err = self._call(argv, timeout if timeout is not None else self.timeout)
        if code == 124:
            return self._fail(op, attempt, dry_run, t0, code, "TIMEOUT", f"ingest offload exceeded {timeout or self.timeout}s; re-running resumes it (identical files are skipped)")
        try:
            doc = one_json_document(out, "ingest offload")
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, code, "INVALID_RESULT", f"{e} (exit {code}) {err.strip()[-300:]}".strip())
        if isinstance(doc.get("error"), dict):
            ecode = str(doc["error"].get("code") or "INCOMPLETE")
            return self._fail(op, attempt, dry_run, t0, code, ecode, str(doc["error"].get("message") or ""),
                              {"skill_error": scrub(doc["error"], FORBIDDEN_ARG_KEYS[:-2])})
        why = admit_report(doc, args)
        if why:
            return self._fail(op, attempt, dry_run, t0, code, "INVALID_RESULT", "offload report not admitted: " + "; ".join(why))
        files = [f for f in doc.get("files") or [] if isinstance(f, dict)]
        dests = [destination_summary(d, files, read_manifest=not args.get("dry_run")) for d in doc["destinations"]]
        summary = {"ok": bool(doc.get("ok")), "dry_run": bool(doc.get("dry_run")), "source": doc.get("source"), "hash": doc.get("hash"), "bytes_total": doc.get("bytes_total"),
                   "files": len(files), "skipped": len(doc.get("skipped") or []), "card": doc.get("card"), "destinations": dests,
                   "started_at": doc.get("started_at"), "finished_at": doc.get("finished_at")}
        data = {"skill": {"id": SKILL_ID, "version": self.version}, "report": doc, "summary": summary}
        if doc.get("ok") is True and code == 0:
            if not args.get("dry_run"):
                missing = [d["root"] for d in dests if not d.get("manifest") or not Path(str(d["manifest"])).is_file()]
                if missing:
                    return self._fail(op, attempt, dry_run, t0, code, "INVALID_RESULT", f"report ok but no {MANIFEST_NAME} at {', '.join(missing)}", extra=data)
            return ToolResult(op.id, op.tool, True, 0, None, dict(data, status="ok"), [], "", round(time.time() - t0, 3), attempt, dry_run)
        kinds = classify_report(dests)
        codes = sorted({p["code"] for d in dests for p in d["problems"]})
        bad = [d for d in dests if not d["ok"]]
        msg = "; ".join(f"{d['root']}: copied {d['copied']}, identical {d['skipped_identical']}, conflicts {d['conflicts']}, failed {d['failed']}, mhl_failed {d['mhl_failed']}"
                        + (f" ({', '.join(p['code'] for p in d['problems'])})" if d["problems"] else "") for d in bad)
        complete = [d["root"] for d in dests if d["ok"]]
        return self._fail(op, attempt, dry_run, t0, code or 3, "INCOMPLETE", f"ingest offload incomplete ({', '.join(kinds)}): {msg}"
                          + (f"; complete and verified: {', '.join(complete)}" if complete else ""),
                          {"kind": kinds[0], "kinds": kinds, "codes": codes, "destinations": dests, "complete": complete, "incomplete": [d["root"] for d in bad]}, extra=data)

    def _measure(self, op: Operation, timeout: Optional[float], dry_run: bool, attempt: int, t0: float) -> ToolResult:
        try:
            argv = self._measure_argv(op.tool, op.args)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, 2, "INVALID_REQUEST", str(e))
        code, out, err = self._call(argv, timeout if timeout is not None else self.measure_timeout)
        if code == 124:
            return self._fail(op, attempt, dry_run, t0, code, "TIMEOUT", f"{op.tool} exceeded {timeout or self.measure_timeout}s")
        try:
            doc = one_json_document(out, op.tool)
        except ToolError as e:
            return self._fail(op, attempt, dry_run, t0, code, "INVALID_RESULT", f"{e} (exit {code}) {err.strip()[-300:]}".strip())
        if isinstance(doc.get("error"), dict):
            return self._fail(op, attempt, dry_run, t0, code, str(doc["error"].get("code") or "INCOMPLETE"), str(doc["error"].get("message") or ""),
                              {"skill_error": scrub(doc["error"], FORBIDDEN_ARG_KEYS[:-2])})
        want = INVENTORY_SCHEMA if op.tool == TOOL_INVENTORY else VERIFY_SCHEMA
        if doc.get("schema") != want or not same_file(str(doc.get("root") or ""), op.args["target"]):
            return self._fail(op, attempt, dry_run, t0, code, "INVALID_RESULT", f"{op.tool}: document {doc.get('schema')!r} about {doc.get('root')!r} is not a {want} of {op.args['target']}")
        key = "inventory" if op.tool == TOOL_INVENTORY else "verify"
        data: Dict[str, Any] = {"skill": {"id": SKILL_ID, "version": self.version}, key: doc, "status": "ok"}
        if op.tool == TOOL_VERIFY:
            data["failures"] = verify_failures(doc)
            data["verified"] = bool(doc.get("ok"))
        return ToolResult(op.id, op.tool, True, 0, None, data, [], "", round(time.time() - t0, 3), attempt, dry_run)
