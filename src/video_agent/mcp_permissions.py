"""MCP permission scopes and the confirmation step before destructive or outward-facing calls (ADR-125).

Every MCP tool has a permission class (TOOL_CLASSES). A call's effective class is worked out from its arguments and the files it
would touch (classify): `plan` is `write`, but `destructive` when it would replace an existing plan; `render` is `write`, `read` as a
dry run, and `publish` when its card offload copies onto drives outside the allowed roots. The server runs a call only when its class
is among the enabled scopes (default read + write). A `destructive` or `publish` call, even when enabled, runs in two steps: the first
call returns a preview and a confirmation token; the action runs only when the same call is repeated with that token. The token is
single use, expires after a short time, and is bound to a digest of the call and of the preview (the files' current contents), so it
cannot authorise a different call or a call whose target changed after the preview.

Every path a writing call writes (the project file, the new plan) must resolve, symlinks included, inside an allowed root (the
workspace, plus any --allow-root).

Classes:
  read         reads only: probe, analyse, explain, validate, dry runs, reports
  write        writes new files (or appends a version to the agent's own plan / preference store) inside the allowed roots
  destructive  replaces or deletes an existing file
  publish      outward-facing: uploads or publishes, writes outside the allowed roots, or spends money / API quota (a paid AI provider)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

READ, WRITE, DESTRUCTIVE, PUBLISH = "read", "write", "destructive", "publish"
SCOPES: Tuple[str, ...] = (READ, WRITE, DESTRUCTIVE, PUBLISH)
DEFAULT_SCOPES: Tuple[str, ...] = (READ, WRITE)
CONFIRMED_CLASSES = (DESTRUCTIVE, PUBLISH)   # these always need the preview + token step
DEFAULT_CONFIRM_TTL_S = 300.0

ENV_SCOPES = "VIDEO_AGENT_MCP_SCOPES"
ENV_ROOTS = "VIDEO_AGENT_MCP_ROOTS"            # os.pathsep separated
ENV_CONFIG = "VIDEO_AGENT_MCP_CONFIG"          # a JSON file: {"scopes": [...], "allowed_roots": [...], "confirm_ttl_s": 300}
ENV_TTL = "VIDEO_AGENT_MCP_CONFIRM_TTL"
ENV_AI_PROVIDER = "VIDEO_AGENT_AI_PROVIDER"
FREE_AI_PROVIDERS = ("", "null", "none", "host")

# The declared class of every MCP tool: the class of its ordinary call. A tool missing here is refused (fail closed), and a test
# fails when a listed tool has no class. ESCALATES names the higher classes a call of the tool can reach (see classify).
TOOL_CLASSES: Dict[str, str] = {
    "doctor": READ, "skills": READ, "brief": READ, "corrections": READ, "validate": READ, "explain": READ, "diff": READ,
    "check": READ, "artifacts": READ,
    "plan": WRITE, "revise": WRITE, "approve": WRITE, "reject": WRITE, "preferences": WRITE, "render": WRITE, "transcript_edit": WRITE,
    "selects": WRITE,
}
ESCALATES: Dict[str, Tuple[str, ...]] = {"plan": (DESTRUCTIVE, PUBLISH), "revise": (PUBLISH,), "render": (PUBLISH,), "transcript_edit": (DESTRUCTIVE,), "selects": (DESTRUCTIVE,)}


class PermissionConfigError(ValueError):
    pass


def enable_hint(cls: str) -> str:
    want = ",".join(s for s in SCOPES if s in (READ, WRITE) or s == cls)
    return f"start the server with --scopes {want} (or {ENV_SCOPES}={want}, or \"scopes\" in the {ENV_CONFIG} file)"


def parse_scopes(value: Any) -> Tuple[str, ...]:
    items = value if isinstance(value, (list, tuple)) else str(value or "").replace(" ", ",").split(",")
    out: List[str] = []
    for raw in items:
        s = str(raw).strip().lower()
        if not s:
            continue
        if s == "all":
            return SCOPES
        if s not in SCOPES:
            raise PermissionConfigError(f"unknown MCP scope {raw!r}: the scopes are {', '.join(SCOPES)} (or all)")
        if s not in out:
            out.append(s)
    return tuple(out)


@dataclass
class Permissions:
    scopes: Tuple[str, ...] = DEFAULT_SCOPES
    roots: Tuple[str, ...] = ()           # real paths
    confirm_ttl_s: float = DEFAULT_CONFIRM_TTL_S
    paid_ai_provider: Optional[str] = None   # the server's VIDEO_AGENT_AI_PROVIDER when it is a paid one

    def enabled(self, cls: str) -> bool:
        return cls in self.scopes

    def inside(self, path: str) -> bool:
        return any(_within(real_path(path), r) for r in self.roots)


def real_path(path: str) -> str:
    """The path with every symlink and `..` resolved (as far as it exists), case-normalised where the OS ignores case. `~` is not
    expanded: the CLI child opens a tool argument literally (relative to the same working directory), so the path checked here must
    be read the same way (a `~/x` argument is `<cwd>/~/x` for both)."""
    return os.path.normcase(os.path.realpath(os.path.abspath(str(path))))


def _within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:   # different drives on Windows
        return False


def load_permissions(workspace: str, scopes: Optional[str] = None, roots: Optional[Iterable[str]] = None, config: Optional[str] = None,
                     confirm_ttl_s: Optional[float] = None, env: Optional[Mapping[str, str]] = None) -> Permissions:
    """The server's permissions. Each setting comes from the command line, else the environment, else the config file, else the
    default (scopes read + write, the workspace as the only root, a 300 s token lifetime). The workspace is always a root."""
    env = os.environ if env is None else env
    cfg: Dict[str, Any] = {}
    cfg_path = config or env.get(ENV_CONFIG)
    if cfg_path:
        try:
            cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PermissionConfigError(f"cannot read the MCP config file {cfg_path}: {exc}") from exc
        if not isinstance(cfg, dict):
            raise PermissionConfigError(f"the MCP config file {cfg_path} must hold a JSON object")
        unknown = set(cfg) - {"scopes", "allowed_roots", "confirm_ttl_s"}
        if unknown:
            raise PermissionConfigError(f"unknown key(s) in the MCP config file {cfg_path}: {', '.join(sorted(unknown))}")
    if scopes is not None:
        sc = parse_scopes(scopes)
    elif env.get(ENV_SCOPES) is not None:
        sc = parse_scopes(env.get(ENV_SCOPES))
    elif "scopes" in cfg:
        sc = parse_scopes(cfg["scopes"])
    else:
        sc = DEFAULT_SCOPES
    extra: List[str] = list(roots or [])
    if not extra and env.get(ENV_ROOTS):
        extra = [r for r in str(env[ENV_ROOTS]).split(os.pathsep) if r]
    if not extra and cfg.get("allowed_roots"):
        extra = [str(r) for r in cfg["allowed_roots"]]
    for r in extra:
        if not os.path.isabs(os.path.expanduser(r)):
            raise PermissionConfigError(f"allowed root {r!r} is not an absolute path")
    all_roots = tuple(dict.fromkeys(real_path(os.path.expanduser(r)) for r in [workspace, *extra]))
    ttl_raw: Any = confirm_ttl_s if confirm_ttl_s is not None else env.get(ENV_TTL) or cfg.get("confirm_ttl_s") or DEFAULT_CONFIRM_TTL_S
    try:
        ttl = float(ttl_raw)
    except (TypeError, ValueError) as exc:
        raise PermissionConfigError(f"confirmation token lifetime {ttl_raw!r} is not a number of seconds") from exc
    if not 1 <= ttl <= 3600:
        raise PermissionConfigError(f"confirmation token lifetime {ttl:g} s is outside 1..3600 s")
    provider = str(env.get(ENV_AI_PROVIDER) or "").strip().lower()
    return Permissions(scopes=sc, roots=all_roots, confirm_ttl_s=ttl, paid_ai_provider=None if provider in FREE_AI_PROVIDERS else provider)


@dataclass
class Call:
    """What one tool call would do: its effective class, the paths it writes, and (destructive / publish) the preview."""
    tool: str
    cls: str
    writes: List[str] = field(default_factory=list)
    preview: List[Dict[str, Any]] = field(default_factory=list)
    spends: bool = False


def _rank(cls: str) -> int:
    return SCOPES.index(cls)


def _file_facts(path: str) -> Dict[str, Any]:
    facts: Dict[str, Any] = {"path": os.path.abspath(path), "resolved": real_path(path), "symlink": os.path.islink(path)}
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        facts["unreadable"] = str(exc)
        return facts
    facts.update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
    try:
        doc = json.loads(data.decode("utf-8"))
        plan = doc.get("plan") or {}
        decisions = doc.get("decisions") or []
        facts["plan"] = {"project": (doc.get("project") or {}).get("name"), "version": plan.get("version"), "status": plan.get("status"),
                         "approved_decisions": sum(1 for d in decisions if d.get("status") == "APPROVED"), "decisions": len(decisions)}
    except (ValueError, AttributeError, UnicodeDecodeError):
        pass
    return facts


def _offload_destinations(project: str) -> Optional[Dict[str, Any]]:
    try:
        doc = json.loads(Path(project).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    from .agent import offload
    for st in (doc.get("plan") or {}).get("steps") or []:
        if st.get("id") == offload.STEP_ID or st.get("skill") == offload.SKILL:
            p = st.get("params") or {}
            return {"source": p.get("source"), "destinations": [str(d) for d in p.get("destinations") or p.get("roots") or []]}
    return None


def classify(name: str, args: Mapping[str, Any], perms: Permissions, workspace: str) -> Call:
    """The effective class of a call. Raises KeyError for a tool without a class (the server refuses it)."""
    base = TOOL_CLASSES[name]
    call = Call(name, base)

    def raise_to(cls: str, item: Optional[Dict[str, Any]] = None) -> None:
        if _rank(cls) > _rank(call.cls):
            call.cls = cls
        if item:
            call.preview.append(item)

    def spend(why: str) -> None:
        if perms.paid_ai_provider:
            call.spends = True
            raise_to(PUBLISH, {"action": "spend", "provider": perms.paid_ai_provider,
                               "detail": f"{why}: the server's {ENV_AI_PROVIDER}={perms.paid_ai_provider} is a paid API (tokens are billed)"})

    project = args.get("project")
    if name == "plan":
        inputs = args.get("inputs")
        inputs = [inputs] if isinstance(inputs, str) else [str(i) for i in inputs or []]
        if inputs:
            from .project.ir import default_plan_path
            out = default_plan_path(workspace, inputs, str(args.get("profile") or "generic"))
            call.writes.append(out)
            if os.path.lexists(out):
                raise_to(DESTRUCTIVE, {"action": "overwrite", **_file_facts(out),
                                       "detail": "a plan of this input and profile exists; planning again replaces it with its versions and approvals "
                                                 "(revise continues it instead)"})
            from .agent.offload import is_card_input
            if not args.get("ai_brief", True) and not any(is_card_input(i) for i in inputs):
                spend("plan without ai_brief asks the configured AI provider")
    elif name == "revise":
        if project:
            call.writes.append(str(project))
        if args.get("ai_recommendations") is None:
            spend("revise without ai_recommendations asks the configured AI provider")
            if call.spends and project and os.path.isfile(str(project)):   # bind the token to the plan the spend is about
                call.preview.append({"action": "project_state", **_file_facts(str(project))})
    elif name in ("approve", "reject"):
        if project:
            call.writes.append(str(project))
    elif name == "render":
        if args.get("dry_run"):
            call.cls = READ
        elif project:
            call.writes.append(str(project))
            off = _offload_destinations(str(project))
            outside = [d for d in (off or {}).get("destinations") or [] if not perms.inside(d)]
            if outside:
                raise_to(PUBLISH, {"action": "copy_outside_roots", "source": off.get("source"), "destinations": outside,
                                   "detail": "the card offload copies the card onto these folders, outside the server's allowed roots"})
    elif name == "transcript_edit":
        source = args.get("input")
        if source:
            stem = Path(str(source)).stem
            out = str(args.get("output") or Path(workspace) / "transcript_edits" / f"{stem}.transcript-edit.json")
            preview_dir = str(Path(workspace) / "transcript_edits" / f"{stem}.preview")   # video-editing ripple's outputs (overwritten)
            call.writes += [out, preview_dir]
            if os.path.lexists(out) or os.path.lexists(preview_dir):
                item: Dict[str, Any] = {"action": "overwrite", "preview_dir": os.path.abspath(preview_dir),
                                        "detail": "a transcript edit of this input exists; a new one replaces the file (a plan's edit.transcript_edit may point "
                                                  "to it) and the preview files"}
                if os.path.lexists(out):
                    item.update(_file_facts(out))
                raise_to(DESTRUCTIVE, item)
    elif name == "selects":
        out = str(args.get("output") or Path(workspace) / "selects" / "selects.json")
        call.writes.append(out)
        if os.path.lexists(out):
            raise_to(DESTRUCTIVE, {"action": "overwrite", **_file_facts(out),
                                   "detail": "a selects file exists here; a new batch replaces it (a plan's edit.selects may point to it)"})
    elif name == "preferences":
        if (args.get("action") or "list") in ("list", "show", "diff"):
            call.cls = READ
    elif name == "check":
        if args.get("qc"):
            call.cls = WRITE   # qc-skill stores its report in the workspace
    return call


def call_digest(tool: str, args: Mapping[str, Any], call: Call) -> str:
    body = json.dumps({"tool": tool, "arguments": args, "class": call.cls, "writes": [real_path(p) for p in call.writes], "preview": call.preview},
                      sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class ConfirmationTokens:
    """Single-use, short-lived tokens, each bound to the digest of one previewed call. Kept in the server's memory only."""

    def __init__(self, ttl_s: float, clock: Callable[[], float] = time.monotonic):
        self.ttl_s, self.clock = ttl_s, clock
        self._tokens: Dict[str, Tuple[str, float]] = {}

    def mint(self, digest: str) -> str:
        now = self.clock()
        self._tokens = {t: v for t, v in self._tokens.items() if v[1] > now}   # forget expired ones
        token = secrets.token_urlsafe(24)
        self._tokens[token] = (digest, now + self.ttl_s)
        return token

    def redeem(self, token: str, digest: str) -> Optional[str]:
        """None when the token authorises this call; otherwise the refusal code. A presented token is spent either way."""
        entry = self._tokens.pop(str(token), None)
        if entry is None:
            return "CONFIRM_TOKEN_UNKNOWN"
        if self.clock() >= entry[1]:
            return "CONFIRM_TOKEN_EXPIRED"
        if not hmac.compare_digest(entry[0], digest):
            return "CONFIRM_TOKEN_MISMATCH"
        return None
