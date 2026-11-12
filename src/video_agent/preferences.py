"""Feedback scopes and the preference store (ADR-111).

    revise --feedback / --set  →  revision.feedback[] {scope: this_video | channel | all_videos, scope_status, proposed_scope}
      → (a person, explicitly) promote  →  PreferenceStore (channel:<id> or global) new version {rules, who, when, from which feedback}
      → the next plan of that channel reads the stores' current versions as PREFERENCE requirements (explicit --set and the brief win)
        and records the versions it used (request.args.preferences); list / diff / rollback through the CLI and MCP

- **Scope.** Every feedback item records a scope. The default is `this_video`: it changes this project's next version only. A wider
  scope is never inferred: the caller says it (`--scope`), and when an AI proposes it (`scope_by=ai`) the item stays `this_video`
  with `proposed_scope` and `scope_status: candidate` until a person confirms it at promotion. `channel` needs the project's
  `channel_id` (a channel, brand or series name the user gives at plan / revise).
- **Promotion is a separate, recorded step.** One feedback event never rewrites a permanent preference by itself (MASTER_SPEC
  §28): only `promote` writes a store, only for a `channel` / `all_videos` item (a candidate needs its scope confirmed), only for
  its structured `--set` keys (free text has no rule to store), each key checked like `--set` and limited to the keys a brief may
  set (no file paths, no job / provider keys). The item records where and when it was promoted and by whom; it is promoted once.
- **Versioned store.** `<workspace>/preferences/global.json` and `<workspace>/preferences/channels/<channel_id>.json`
  (`$VIDEO_AGENT_PREFERENCES_DIR` replaces `<workspace>/preferences`). Each store keeps every version: version 0 is empty; a
  promotion or a rollback adds version N+1 with the full rule set, the action, who, when and why. Rollback never deletes: it adds a
  version whose rules are those of the target version. Written atomically (a temporary file, then a rename).
- **Plans.** A new plan of a channel uses the global store's current rules, then the channel's (channel wins), then the brief, then
  explicit `--set` (explicit wins). The rules become Requirements with provenance PREFERENCE (never the user's explicit word for
  this video, so they never waive a CONFIRM). The versions and rules used are recorded in `request.args.preferences`; a revision
  re-uses them, so a later promotion or rollback changes the next plan, never a recorded one.
- **Corrections per minute.** Feedback items given on a plan version ÷ that version's delivered minutes (the planned deliverable
  durations). Reported per version with the change from the previous version and the channel's recent average (the latest version
  of up to 10 other projects of the channel, from `<channel>.corrections.json`). A report only: nothing acts on it.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import now_iso

SCHEMA = "video-agent/preference-store@1"
SCOPES = ("this_video", "channel", "all_videos")
DEFAULT_SCOPE = "this_video"
SCOPE_BY = ("user", "ai")
CHANNEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
RECENT_PROJECTS = 10
ENV_DIR = "VIDEO_AGENT_PREFERENCES_DIR"


class PreferenceError(ValueError):
    """A scope, promotion, store or rollback request that is refused (nothing is written)."""


def check_channel(channel: Any) -> str:
    c = str(channel or "").strip()
    if not CHANNEL_RE.match(c):
        raise PreferenceError(f"channel must be 1..64 letters, digits, '.', '_' or '-' (a name, never a path), got {channel!r}")
    return c


def check_scope(scope: Any) -> str:
    s = str(scope or DEFAULT_SCOPE).strip().lower()
    if s not in SCOPES:
        raise PreferenceError(f"scope must be one of {' | '.join(SCOPES)}, got {scope!r}")
    return s


def feedback_scope(scope: Optional[str], scope_by: Optional[str], channel_id: Optional[str]) -> Dict[str, Any]:
    """The scope fields of one feedback item. A wider scope from the user is confirmed; from an AI it is a candidate (the item stays
    this_video until a person confirms it). `channel` needs the project's channel."""
    s = check_scope(scope)
    by = str(scope_by or "user").strip().lower()
    if by not in SCOPE_BY:
        raise PreferenceError(f"scope_by must be one of {' | '.join(SCOPE_BY)}, got {scope_by!r}")
    if s == "channel" and not channel_id:
        raise PreferenceError("scope channel needs the project's channel (plan or revise with --channel <name>)")
    if scope is None:
        return {"scope": DEFAULT_SCOPE, "scope_source": "default", "scope_status": "confirmed"}
    if by == "ai" and s != DEFAULT_SCOPE:
        return {"scope": DEFAULT_SCOPE, "proposed_scope": s, "scope_source": "ai", "scope_status": "candidate"}
    return {"scope": s, "scope_source": by, "scope_status": "confirmed"}


# ---- the store --------------------------------------------------------------------------------------------------------------
def store_root(workspace: str) -> Path:
    return Path(os.environ.get(ENV_DIR) or (Path(workspace) / "preferences")).resolve()


class PreferenceStore:
    """One store (global, or one channel's). `doc()` is the whole versioned document; version 0 is the empty rule set."""

    def __init__(self, workspace: str, channel: Optional[str] = None):
        self.channel = check_channel(channel) if channel is not None else None
        root = store_root(workspace)
        self.path = root / "global.json" if self.channel is None else root / "channels" / f"{self.channel}.json"
        self.name = "global" if self.channel is None else f"channel:{self.channel}"

    def doc(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return {"schema": SCHEMA, "store": self.name, "channel_id": self.channel, "current": 0,
                    "versions": [{"version": 0, "action": "init", "at": None, "by": None, "reason": "empty store", "rules": {}}]}
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise PreferenceError(f"preference store {self.path} cannot be read: {e}")
        if d.get("schema") != SCHEMA or d.get("store") != self.name or not isinstance(d.get("versions"), list):
            raise PreferenceError(f"{self.path} is not a {SCHEMA} document for {self.name}")
        return d

    def version(self, n: Optional[int] = None) -> Dict[str, Any]:
        d = self.doc()
        n = d["current"] if n is None else int(n)
        v = next((x for x in d["versions"] if x["version"] == n), None)
        if v is None:
            raise PreferenceError(f"{self.name} has no version {n} (versions 0..{d['versions'][-1]['version']})")
        return v

    def current(self) -> Tuple[int, Dict[str, Any]]:
        v = self.version()
        return v["version"], {k: r["value"] for k, r in v["rules"].items()}

    def _write(self, d: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(d, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _append(self, rules: Dict[str, Any], action: str, by: str, reason: str, source: Dict[str, Any]) -> Dict[str, Any]:
        d = self.doc()
        n = d["versions"][-1]["version"] + 1
        v = {"version": n, "action": action, "at": now_iso(), "by": by, "reason": reason, "source": source, "rules": rules, "previous": d["current"]}
        d["versions"].append(v)
        d["current"] = n
        self._write(d)
        return v

    def promote(self, values: Dict[str, Any], by: str, reason: str, source: Dict[str, Any]) -> Dict[str, Any]:
        """A new version: the current rules plus these keys (a key already there takes the new value)."""
        cur = dict(self.version()["rules"])
        at = now_iso()
        for k, v in values.items():
            cur[k] = {"value": v, "by": by, "at": at, **{x: source.get(x) for x in ("project_id", "feedback_id", "plan_version")}}
        return self._append(cur, "promote", by, reason, source)

    def rollback(self, to: int, by: str, reason: str) -> Dict[str, Any]:
        """A new version whose rules are those of version `to` (nothing is deleted)."""
        target = self.version(to)
        return self._append(json.loads(json.dumps(target["rules"])), "rollback", by, reason or f"roll back to version {to}", {"rollback_to": int(to)})

    def history(self) -> List[Dict[str, Any]]:
        d = self.doc()
        return [{"version": v["version"], "action": v["action"], "at": v["at"], "by": v["by"], "reason": v.get("reason"), "keys": sorted(v["rules"]),
                 "current": v["version"] == d["current"], **({"source": v["source"]} if v.get("source") else {})} for v in d["versions"]]

    def diff(self, a: int, b: int) -> Dict[str, Any]:
        ra, rb = self.version(a)["rules"], self.version(b)["rules"]
        return {"store": self.name, "from": int(a), "to": int(b),
                "added": {k: rb[k]["value"] for k in sorted(set(rb) - set(ra))},
                "removed": {k: ra[k]["value"] for k in sorted(set(ra) - set(rb))},
                "changed": {k: {"before": ra[k]["value"], "after": rb[k]["value"]} for k in sorted(set(ra) & set(rb)) if ra[k]["value"] != rb[k]["value"]}}


def stores_for(workspace: str, channel: Optional[str]) -> List[PreferenceStore]:
    return [PreferenceStore(workspace)] + ([PreferenceStore(workspace, channel)] if channel else [])


def snapshot(workspace: str, channel: Optional[str]) -> Dict[str, Any]:
    """What a new plan records in request.args.preferences: each store's current version and rules (global first, channel wins)."""
    out: Dict[str, Any] = {"stores": [], "values": {}}
    for st in stores_for(workspace, channel):
        n, rules = st.current()
        out["stores"].append({"store": st.name, "version": n, "rules": rules})
        out["values"].update(rules)
    return out


def source_of(prefs: Dict[str, Any], key: str) -> str:
    """`preferences:<store>@v<N>` of the store whose rule set this key (the last one, as in `snapshot`)."""
    src = ""
    for s in (prefs or {}).get("stores") or []:
        if key in (s.get("rules") or {}):
            src = f"preferences:{s['store']}@v{s['version']}"
    return src


# ---- corrections per minute ----------------------------------------------------------------------------------------------------
def corrections_log(workspace: str, channel: str) -> Path:
    return store_root(workspace) / "channels" / f"{check_channel(channel)}.corrections.json"


def record_corrections(workspace: str, channel: str, project_id: str, project_name: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Upsert this project's per-version rows into the channel's log; returns the log's rows."""
    p = corrections_log(workspace, channel)
    try:
        log = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []
    except (OSError, ValueError):
        log = []
    log = [r for r in log if isinstance(r, dict) and r.get("project_id") != project_id]
    log += [dict(r, project_id=project_id, project_name=project_name, at=now_iso()) for r in rows]
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(log, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    return log


def recent_average(log: List[Dict[str, Any]], project_id: str) -> Optional[Dict[str, Any]]:
    """Mean corrections per minute of the latest version of up to RECENT_PROJECTS other projects (log order = most recent last)."""
    latest: Dict[str, Dict[str, Any]] = {}
    for r in log:
        if r.get("project_id") == project_id or r.get("cpm") is None:
            continue
        if r["project_id"] not in latest or r["plan_version"] >= latest[r["project_id"]]["plan_version"]:
            latest[r["project_id"]] = r
    rows = sorted(latest.values(), key=lambda r: str(r.get("at") or ""))[-RECENT_PROJECTS:]
    if not rows:
        return None
    return {"projects": len(rows), "cpm": round(sum(r["cpm"] for r in rows) / len(rows), 3)}
