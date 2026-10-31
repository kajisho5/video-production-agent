"""Contract-backed check of the hand-maintained CATALOG against the installed ffmpeg-skill (ADR-056).

The catalog (catalog.py) is the agent's pin of ffmpeg-skill: the flags it emits per script. ffmpeg-skill publishes its
own machine-readable contract since 0.9.0, generated from each script's argparse parser:

    python3 scripts/_contract.py --json --static    # contract_version, and per tool input_schema.properties[*].cli
    python3 scripts/_contract.py doctor --json      # per tool usable / missing / fix on this machine

At locate time the agent reads both once per checkout and asks, per catalogued tool: is the tool still published, does every
flag the agent can emit for it appear in its input_schema (with the same arity: a catalogued switch is a boolean there, a
catalogued value flag is not), do the catalogued positionals lead its positional list, and does doctor say it can run
here. A tool that fails any of these is unavailable with the reason (capability `ffmpeg-skill:<tool>` MISSING: the
plan names it in its handoff and doctor prints it), so a renamed flag never reaches argparse at run time. Only that tool
is affected. A contract the agent cannot read as 1.x (another major, another skill id, no document) refuses the whole
package: none of its facts can be trusted.

Fatal / non-fatal (ADR-047's split, applied to this Skill): there is no pinned contract file for ffmpeg-skill because the
catalog is the pin and the contract grows every release (42 tools, descriptions, diagnostics). Fatal: contract_version
major, skill id, a catalogued tool / flag / positional / switch arity. Everything else in the contract is ignored here.
A checkout without scripts/_contract.py (ffmpeg-skill < 0.9.0, still inside SUPPORTED_MIN) is `unchecked`: the catalog
is used as before (verified against 0.8.4 --help) and doctor says so."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..discovery import child_env
from .catalog import CATALOG, FLAG_ALIASES, QUALITY_FLAG

SKILL_ID = "ffmpeg-skill"
CONTRACT_SCRIPT = "_contract.py"
CONTRACT_ARGS = ["--json", "--static"]
DOCTOR_ARGS = ["doctor", "--json"]
SUPPORTED_CONTRACT_MAJOR = 1
# flags the adapter appends to every command (adapter.command): --json always, --dry-run for a dry run
RUNNER_FLAGS = ("--json", "--dry-run")
CHECKED, UNCHECKED, REFUSED = "checked", "unchecked", "refused"


def flag_for(script: str, key: str) -> str:
    """The CLI spelling the adapter emits for a catalogued key (adapter.build_argv's rule, without the crf choice)."""
    return FLAG_ALIASES.get((script, key)) or FLAG_ALIASES.get(("*", key)) or "--" + key.replace("_", "-")


@dataclass
class CatalogCheck:
    """Result of comparing CATALOG with one checkout's contract (+ doctor). `tool_reason(name)` is None when the tool is usable."""
    status: str                                   # checked | unchecked | refused
    reason: str = ""                              # package level: why refused / unchecked
    contract_version: Optional[str] = None
    skill_version: Optional[str] = None
    unavailable: Dict[str, str] = field(default_factory=dict)   # catalogued tool -> why it cannot be used
    doctor: Optional[Dict[str, Any]] = None      # {ok, missing, missing_optional, unknown, tools: {name: {usable, missing, fix}}} (non-yes tools only)
    notes: List[str] = field(default_factory=list)

    def tool_reason(self, name: str) -> Optional[str]:
        if self.status == REFUSED:
            return self.reason
        return self.unavailable.get(name)

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "contract_version": self.contract_version, "skill_version": self.skill_version,
                "unavailable": dict(self.unavailable), "doctor": self.doctor, "notes": list(self.notes)}


def _contract_major(v: Any) -> Optional[int]:
    try:
        return int(str(v).split(".", 1)[0])
    except ValueError:
        return None


def check_catalog(contract: Any, doctor: Optional[Dict[str, Any]] = None, scripts: Optional[List[str]] = None) -> CatalogCheck:
    """Pure: CATALOG against a `_contract.py --json --static` document (and optionally a `doctor --json` document).
    `scripts`: the checkout's public scripts; a catalogued tool whose script is absent is left to supports() (not a contract fact)."""
    if not isinstance(contract, dict):
        return CatalogCheck(REFUSED, "ffmpeg-skill contract is not a JSON object")
    cv = contract.get("contract_version")
    skill = contract.get("skill") if isinstance(contract.get("skill"), dict) else {}
    chk = CatalogCheck(CHECKED, contract_version=None if cv is None else str(cv), skill_version=skill.get("version"))
    if _contract_major(cv) != SUPPORTED_CONTRACT_MAJOR:
        chk.status, chk.reason = REFUSED, f"ffmpeg-skill contract_version {cv!r} is not {SUPPORTED_CONTRACT_MAJOR}.x (the catalog check reads the 1.x shape only)"
        return chk
    if skill.get("id") != SKILL_ID:
        chk.status, chk.reason = REFUSED, f"contract skill.id {skill.get('id')!r} != {SKILL_ID!r}"
        return chk
    tools = {str(t.get("name")): t for t in contract.get("tools") or [] if isinstance(t, dict)}
    for name, spec in CATALOG.items():
        if scripts is not None and name not in scripts:
            continue
        why = _tool_problems(name, spec, tools.get(name))
        if why:
            chk.unavailable[name] = "ffmpeg-skill contract: " + "; ".join(why)
    if doctor is not None:
        _apply_doctor(chk, doctor, scripts)
    return chk


def _tool_problems(name: str, spec: Dict[str, Any], tool: Optional[Dict[str, Any]]) -> List[str]:
    if tool is None:
        return [f"tool {name!r} is not published"]
    schema = tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {}
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    by_flag: Dict[str, Dict[str, Any]] = {}
    for p in props.values():
        if isinstance(p, dict) and isinstance(p.get("cli"), list):
            for f in p["cli"]:
                by_flag[str(f)] = p
    out: List[str] = []
    for key, typ in spec["flags"].items():
        flag = flag_for(name, key)
        if key == "crf":   # ADR-048: --quality where the script offers it, else --crf; either satisfies the catalog
            flag = QUALITY_FLAG if QUALITY_FLAG in by_flag else flag
        p = by_flag.get(flag)
        if p is None:
            out.append(f"{name}.py has no {flag} (catalog key {key!r})")
        elif (typ == "bool") != (p.get("type") == "boolean"):
            out.append(f"{name}.py {flag} is {'a switch' if p.get('type') == 'boolean' else 'a value flag'} but the catalog uses it as {'a switch' if typ == 'bool' else 'a value flag'}")
    for flag in RUNNER_FLAGS:
        if flag not in by_flag:
            out.append(f"{name}.py has no {flag}")
    positional = [str(x) for x in schema.get("positional") or []]
    want = list(spec["positional"])
    if positional[:len(want)] != want:
        out.append(f"{name}.py positionals {positional} do not start with the catalog's {want}")
    else:
        required = set(schema.get("required") or [])
        extra = [p for p in positional[len(want):] if p in required]
        if extra:
            out.append(f"{name}.py requires positionals the catalog does not pass: {extra}")
    return out


def _apply_doctor(chk: CatalogCheck, doctor: Dict[str, Any], scripts: Optional[List[str]]) -> None:
    """Surface doctor: per catalogued tool, `usable: no` makes it unavailable with doctor's own missing list and fix; `unknown`
    is recorded (detection failed, not proven missing) and never blocks."""
    per = doctor.get("tools") if isinstance(doctor.get("tools"), dict) else {}
    summary: Dict[str, Any] = {"ok": doctor.get("ok"), "missing": list(doctor.get("missing") or []), "missing_optional": list(doctor.get("missing_optional") or []),
                               "unknown": list(doctor.get("unknown") or []), "tools": {}}
    for name in CATALOG:
        if scripts is not None and name not in scripts:
            continue
        t = per.get(name)
        if not isinstance(t, dict) or t.get("usable") == "yes":
            continue
        summary["tools"][name] = {k: t.get(k) for k in ("usable", "missing", "unknown", "fix") if t.get(k) is not None}
        if t.get("usable") == "no":
            msg = f"ffmpeg-skill doctor: missing {', '.join(map(str, t.get('missing') or [])) or 'a required capability'}" + (f" (fix: {t['fix']})" if t.get("fix") else "")
            chk.unavailable[name] = (chk.unavailable[name] + "; " + msg) if name in chk.unavailable else msg
    chk.doctor = summary


def _run_json(cmd: List[str], timeout: float) -> Tuple[Optional[Any], str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace", env=child_env())
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"{type(e).__name__}: {e}"
    try:
        return json.loads(p.stdout), ""
    except ValueError:
        return None, f"exit {p.returncode}, no JSON document on stdout: {(p.stderr or p.stdout).strip()[-200:]}"


def read_contract(root: Path, python: Optional[str] = None) -> Tuple[Optional[Any], str]:
    """(`_contract.py --json --static` document, error). No environment detection: the contract alone."""
    return _run_json([python or sys.executable, str(Path(root) / "scripts" / CONTRACT_SCRIPT)] + CONTRACT_ARGS, timeout=120.0)


def read_doctor(root: Path, python: Optional[str] = None) -> Tuple[Optional[Any], str]:
    """(`_contract.py doctor --json` document, error). doctor exits 1 (a required capability missing) or 2 (unknown) and
    still prints its document, so the exit code is not read."""
    return _run_json([python or sys.executable, str(Path(root) / "scripts" / CONTRACT_SCRIPT)] + DOCTOR_ARGS, timeout=180.0)


def read_catalog_check(root: Path, scripts: Optional[List[str]] = None, python: Optional[str] = None) -> CatalogCheck:
    """Run the checkout's contract (and doctor) and check the catalog. Never raises: every failure is a status with a reason."""
    if not (Path(root) / "scripts" / CONTRACT_SCRIPT).exists():
        return CatalogCheck(UNCHECKED, f"no scripts/{CONTRACT_SCRIPT} in this checkout (ffmpeg-skill < 0.9.0): the catalog is used unchecked")
    contract, err = read_contract(root, python)
    if contract is None:
        return CatalogCheck(REFUSED, f"`_contract.py --json --static` gave no contract ({err})")
    doc, derr = read_doctor(root, python)
    chk = check_catalog(contract, doc if isinstance(doc, dict) else None, scripts)
    if chk.status == CHECKED and not isinstance(doc, dict):
        chk.notes.append(f"`_contract.py doctor --json` gave no document ({derr}); per-tool usability not surfaced")
    return chk


_CACHE: Dict[Tuple[str, str], CatalogCheck] = {}


def catalog_check(root: Path, version: str, scripts: Optional[List[str]] = None) -> CatalogCheck:
    """Cached per (checkout, version): the contract is static for a checkout, doctor is machine state read once per process."""
    key = (str(Path(root).resolve()), str(version))
    if key not in _CACHE:
        _CACHE[key] = read_catalog_check(Path(root), scripts)
    return _CACHE[key]
