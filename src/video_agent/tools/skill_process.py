"""Shared transport for external Skills that speak JSON over a CLI (ADR-031): one process per call, argv as a list, the
request on stdin, exactly one JSON document on stdout, the Skill's typed error {code, message, retryable} on failure.

Everything here is boundary mechanics only: locating a checkout, invoking the CLI in its own process group with a timeout,
reading one document, recomputing a file hash, and the generic pieces every adapter of this family shares (contract drift,
error → recovery class, the failed ToolResult shape). No adapter here decides anything, builds a command from a request,
or imports a Skill's package. The five earlier adapters (ffmpeg-skill / media-analysis / transcription / video-editing /
audio-production) keep their own code; this module exists so the Phase 3 adapters do not copy it a sixth time."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..models import Operation, ToolResult
from .base import ToolError
from .ffmpeg_skill.adapter import run_process_group


class ContractError(ToolError):
    """The installed Skill does not satisfy the contract the adapter was written for (never patched, never guessed)."""


@dataclass
class CliSkill:
    """A located Skill: argv prefix (a checkout run as `python -m <module>` or a console script) plus the child environment
    it needs (PYTHONPATH for a checkout, never credentials)."""
    skill_id: str
    command: List[str]
    root: Optional[Path]
    env: Dict[str, str]

    def describe(self) -> str:
        return str(self.root) if self.root else self.command[0]


def locate_cli_skill(skill_id: str, module: str, package_dir: str, console: str, env_var: str, explicit: Optional[str] = None,
                     env: Optional[Mapping[str, str]] = None, checkout_names: Tuple[str, ...] = ()) -> Optional[CliSkill]:
    """Find a checkout (<dir>/src/<package_dir>/cli.py or __init__.py, run with PYTHONPATH=<dir>/src) or a console script on
    PATH. Order: explicit dir, the env var, ~/.claude/skills/<name>, ./vendor/<name>, ../<name>, then PATH."""
    env_map: Mapping[str, str] = os.environ if env is None else env
    cands: List[Path] = []
    if explicit:
        cands.append(Path(explicit))
    if env_map.get(env_var):
        cands.append(Path(env_map[env_var]))
    for name in checkout_names or (skill_id,):
        cands += [Path.home() / ".claude" / "skills" / name, Path.cwd() / "vendor" / name, Path.cwd().parent / name]
    for c in cands:
        pkg = c / "src" / package_dir
        if (pkg / "cli.py").is_file() or (pkg / "__init__.py").is_file():
            return CliSkill(skill_id, [sys.executable, "-m", module], c, {"PYTHONPATH": str(c / "src")})
    exe = shutil.which(console, path=env_map.get("PATH"))
    if exe:
        return CliSkill(skill_id, [exe], None, {})
    return None


def invoke(skill: CliSkill, argv: List[str], stdin: Optional[str] = None, timeout: Optional[float] = None, extra_env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    """Run the Skill CLI once (process group, timeout → exit 124). The child's extra environment is the checkout's PYTHONPATH
    and, where a Skill only takes its engine location from the environment, that directory; nothing else is added."""
    env_backup: Dict[str, Optional[str]] = {}
    merged = dict(skill.env)
    merged.update(extra_env or {})
    try:
        for k, v in merged.items():
            env_backup[k] = os.environ.get(k)
            os.environ[k] = v
        return run_process_group(list(skill.command) + list(argv), timeout=timeout, stdin=stdin)
    finally:
        for k, old in env_backup.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def one_json_document(stdout: str, who: str) -> Dict[str, Any]:
    """stdout must carry exactly one JSON object (a Skill that prints two documents, text or nothing is not trusted)."""
    text = (stdout or "").strip()
    if not text:
        raise ToolError(f"{who}: empty stdout (expected one response document)")
    try:
        doc, end = json.JSONDecoder().raw_decode(text)
    except ValueError:
        raise ToolError(f"{who}: stdout is not JSON: {text[:120]!r}")
    if text[end:].strip():
        raise ToolError(f"{who}: more than one document on stdout")
    if not isinstance(doc, dict):
        raise ToolError(f"{who}: response is not an object")
    return doc


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_sha_prefix(v: Any) -> str:
    s = str(v or "")
    return s[7:] if s.startswith("sha256:") else s


def same_file(a: str, b: str) -> bool:
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except OSError:
        return False


def remove_fresh(path: Optional[str], t0: float) -> None:
    """Remove an output the failed call may have half-written (only a file newer than the call started)."""
    if not path:
        return
    try:
        if os.path.isfile(path) and os.path.getmtime(path) >= t0 - 1:
            os.remove(path)
    except OSError:
        pass


# the agent's boundary vocabulary: never forwarded as a request key, never accepted as a parameter name (each Skill adds its own forbidden fields)
FORBIDDEN_ARG_KEYS = ("command", "commands", "argv", "cmd", "cmdline", "shell", "exec", "args", "script", "binary", "executable", "executables", "env", "environment", "cwd",
                      "filter", "filters", "filter_complex", "vf", "af", "ffmpeg", "ffprobe", "api_key", "apikey", "token", "secret", "password", "credentials", "eval",
                      "expression", "html", "javascript", "css", "workspace", "allowed_input", "allowed_inputs", "allowed_input_roots", "allowed-input", "allowed_lut",
                      "ffmpeg_skill", "ffmpeg-skill", "ffmpeg_skill_dir", "path", "paths")


def scan_forbidden(obj: Any, forbidden: Tuple[str, ...], where: str = "args") -> Optional[str]:
    """First forbidden key anywhere in a document (by name, case-insensitive), or None."""
    fb = {f.lower() for f in forbidden}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in fb:
                return f"{where}.{k}"
            hit = scan_forbidden(v, forbidden, f"{where}.{k}")
            if hit:
                return hit
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hit = scan_forbidden(v, forbidden, f"{where}[{i}]")
            if hit:
                return hit
    return None


def scrub(obj: Any, forbidden: Tuple[str, ...]) -> Dict[str, Any]:
    """Error details the Skill returned, minus anything that looks like a command or credential (recorded for humans only)."""
    if not isinstance(obj, dict):
        return {}
    fb = {f.lower() for f in forbidden}
    return {str(k): (v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(v, default=str)[:300]) for k, v in obj.items() if str(k).lower() not in fb}


def drift_report(live: Dict[str, Any], pinned: Dict[str, Any], keys: Tuple[str, ...], list_key: Optional[str] = None, item_id: str = "type",
                 item_keys: Tuple[str, ...] = ()) -> List[str]:
    """Differences between the installed contract and the pinned one on the fields the agent depends on (top-level keys, and
    per item of one list keyed by `item_id`). Non-empty = the adapter's expectations are stale: reported, never silently kept."""
    out: List[str] = []
    for k in keys:
        if live.get(k) != pinned.get(k):
            out.append(f"{k}: pinned {json.dumps(pinned.get(k), sort_keys=True, default=str)[:160]} != live {json.dumps(live.get(k), sort_keys=True, default=str)[:160]}")
    if list_key:
        lo = {str(o.get(item_id)): o for o in live.get(list_key) or [] if isinstance(o, dict)}
        po = {str(o.get(item_id)): o for o in pinned.get(list_key) or [] if isinstance(o, dict)}
        for t in sorted(set(lo) | set(po)):
            if t not in lo:
                out.append(f"{list_key} {t}: pinned but not in the installed contract")
            elif t not in po:
                out.append(f"{list_key} {t}: installed but not pinned")
            else:
                for k in item_keys:
                    if lo[t].get(k) != po[t].get(k):
                        out.append(f"{list_key} {t}.{k}: pinned {json.dumps(po[t].get(k), sort_keys=True, default=str)[:120]} != live {json.dumps(lo[t].get(k), sort_keys=True, default=str)[:120]}")
    return out


def error_table(contract: Dict[str, Any]) -> Tuple[Dict[str, bool], Dict[str, int]]:
    """(code → retryable, code → exit code) from a contract's `errors` block. Two shapes exist in the ecosystem: a `retryable`
    map (audio-production family) or a `non_retryable` list (subtitle-skill); both are read, nothing is assumed."""
    errs = contract.get("errors") or {}
    codes = [str(c) for c in errs.get("codes") or []]
    if isinstance(errs.get("retryable"), dict):
        retry = {c: bool(errs["retryable"].get(c, False)) for c in codes}
    elif isinstance(errs.get("non_retryable"), list):
        nr = {str(x) for x in errs["non_retryable"]}
        retry = {c: c not in nr for c in codes}
    else:
        retry = {c: False for c in codes}
    exit_codes = {str(k): int(v) for k, v in (errs.get("exit_codes") or {}).items()} if isinstance(errs.get("exit_codes"), dict) else {}
    return retry, exit_codes


# the agent's reading of the ecosystem's shared error vocabulary (recovery class); a Skill's own retryable verdict is kept beside it
RECOVERY_CLASS: Dict[str, str] = {
    "INVALID_REQUEST": "INVALID_ARGS", "UNSUPPORTED_OPERATION": "INVALID_ARGS", "UNSUPPORTED_FORMAT": "INVALID_ARGS", "INVALID_TIME_RANGE": "INVALID_ARGS",
    "DEPENDENCY_ERROR": "INVALID_ARGS", "INVALID_INPUT": "INPUT_MISSING", "PATH_NOT_ALLOWED": "INPUT_MISSING", "MISSING_INPUT": "INPUT_MISSING",
    "TOOL_ERROR": "UNKNOWN", "CANCELLED": "TIMEOUT", "OUTPUT_ERROR": "SKILL_ERROR", "VALIDATION_ERROR": "SKILL_ERROR", "INTERNAL_ERROR": "SKILL_ERROR",
    "INVALID_RESULT": "SKILL_ERROR",
}


def failed_result(op: Operation, skill_id: str, version: str, attempt: int, dry_run: bool, t0: float, code: int, errc: str, message: str, retryable: bool,
                  details: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, Any]] = None) -> ToolResult:
    """The failed ToolResult every adapter of this family returns: the Skill's code / message / retryable verdict, the exit code,
    and the agent's recovery class (recovery.py prefers this structured error over stderr heuristics)."""
    data: Dict[str, Any] = {"skill": {"id": skill_id, "version": version}, "status": "failed",
                            "error": {"code": errc, "message": message, "retryable": bool(retryable), "details": details or {}, "exit_code": code,
                                      "recovery_class": RECOVERY_CLASS.get(errc, "SKILL_ERROR")}}
    if extra:
        data.update(extra)
    return ToolResult(op.id, op.tool, False, code, None, data, list(data.get("commands") or []), f"{skill_id} [{errc}] {message}", round(time.time() - t0, 3), attempt, dry_run)


def as_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def fingerprint_matches(reported: Any, path: str) -> Tuple[bool, str]:
    """Recompute the sha256 of `path` and compare it with what the Skill reported (with or without the `sha256:` prefix)."""
    actual = sha256_file(path)
    return strip_sha_prefix(reported) == actual, actual
