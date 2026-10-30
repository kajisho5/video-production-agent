"""URL ingestion (video-production-agent issue #50, Task 1; ADR-059): `plan` / `analyze` accept a URL as well as a local
file path. A URL is downloaded once, up front, into the workspace before anything else runs; from that point on the
pipeline sees only a local file, exactly like any other input -- cli.py rewrites the input list before analysis ever
starts, so nothing downstream (probing, requirements, decisions, the compiler, execution) ever special-cases a URL.

URL download (URL素材取得) belongs to ingest-skill (ecosystem.json, owner decision 2026-10-30). The agent no longer runs
yt-dlp itself: it hands the URL to `ingest fetch <url> --dest <workspace>/downloads --json` (an argv list, never a
shell), located like every other Skill CLI (a checkout through VIDEO_AGENT_INGEST_DIR / a sibling `ingest-skill`, or
the `ingest` console script). ingest-skill (its ADR-005) asks yt-dlp for metadata first, adds live_from_start only
when yt-dlp itself says the stream is still live, downloads into a staging folder, then offloads that folder like a
card into <dest>/<extractor>-<id>/ with a read-back, ingest-manifest.json and ASC MHL. Its "fetch" record keeps the
source and page URLs with credentials removed; cookies, headers and the signed media URL are never stored.

What the agent keeps: the IngestRecord (provenance of the download, an OBSERVED Observation on the asset it became),
taken from ingest-skill's report -- and the sha256 it recomputes itself from the delivered file, which must match the
report's (a report about a different file is never admitted). yt-dlp's availability comes from `ingest doctor --json`
(`checks["yt-dlp"]`). Without ingest-skill (or without its `fetch` extra) a URL input is not a crash: cli.py prints an
ADR-051 handoff naming ingest-skill and how to install it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from ..models import Observation, now_iso
from ..skills.ecosystem import component, install_commands, repository
from ..tools.base import ToolError
from ..tools.skill_process import CliSkill, invoke, locate_cli_skill, one_json_document, sha256_file

ENV_DIR = "VIDEO_AGENT_INGEST_DIR"      # an ingest-skill checkout (<dir>/src/ingest_skill), checked before PATH
PACKAGE = "ingest"                      # the ecosystem package id (skills/ecosystem.py)
TOOL = "ingest/fetch"
FETCH_SCHEMA = "ingest/fetch@1"
REPORT_SCHEMA = "ingest/fetch-report@1"
DOCTOR_SCHEMA = "ingest/doctor@1"
DOWNLOAD_TIMEOUT = 3600.0               # ingest-skill has no overall timeout of its own (socket timeout 60 s): the process bound
DOWNLOAD_SUBDIR = "downloads"
FAILURE_KINDS = ("ENGINE_MISSING", "URL_INVALID", "DOWNLOAD_FAILED", "INCOMPLETE", "TIMEOUT", "INVALID_OUTPUT")
# ingest-skill's error codes (its contract.ERRORS) → the agent's failure kinds; any other code is DOWNLOAD_FAILED, named
ERROR_KINDS = {"FETCH_UNAVAILABLE": "ENGINE_MISSING", "URL_INVALID": "URL_INVALID", "FETCH_FAILED": "DOWNLOAD_FAILED"}


def install_hint() -> str:
    return "install ingest-skill with its fetch extra: " + " / ".join(c for c in install_commands(PACKAGE) if not c.startswith("or "))


class IngestError(Exception):
    """Ingestion failure domain (mirrors media.analysis.AnalysisError): a download failure is never a decision and
    never an execution incident -- it is refused before analysis, the same as a missing local file."""

    def __init__(self, kind: str, message: str = ""):
        if kind not in FAILURE_KINDS:
            kind = "DOWNLOAD_FAILED"
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind = kind


def is_url(value: str) -> bool:
    """A URL input: scheme + host, never inferred from "this doesn't exist as a local path" -- a mistyped local
    path must still fail as a missing file (the media engine's own probing step), never silently attempt a
    network fetch."""
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def locate_ingest(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[CliSkill]:
    """An ingest-skill checkout (run as `python -m ingest_skill.cli` with PYTHONPATH=<dir>/src) or the `ingest` console
    script; the same order as every other Skill CLI (tools/skill_process.locate_cli_skill)."""
    return locate_cli_skill(PACKAGE, "ingest_skill.cli", "ingest_skill", "ingest", ENV_DIR, explicit, env, ("ingest-skill",))


def ingest_doctor(skill: CliSkill, timeout: float = 120.0) -> Dict[str, Any]:
    """`ingest doctor --json` (exit 0 ok / 3 a required check failed; a document either way)."""
    code, out, err = invoke(skill, ["doctor", "--json"], timeout=timeout)
    try:
        doc = one_json_document(out, "ingest doctor")
    except ToolError as e:
        return {"schema": DOCTOR_SCHEMA, "ok": False, "checks": {}, "problem": f"doctor produced no document (exit {code}): {e} {err.strip()[-200:]}".strip()}
    if doc.get("schema") != DOCTOR_SCHEMA:
        return {"schema": DOCTOR_SCHEMA, "ok": False, "checks": {}, "problem": f"unexpected doctor document {doc.get('schema')!r}"}
    return doc


def yt_dlp_capability(env: Optional[Dict[str, str]] = None, skill: Optional[CliSkill] = None) -> Tuple[str, str, Dict[str, Any]]:
    """(status, detail, evidence) of URL download, as ingest-skill's own doctor reports it (`checks["yt-dlp"]`): AVAILABLE with
    the yt-dlp version, else MISSING with the reason (ingest-skill absent, its fetch extra not installed, no doctor document)."""
    skill = skill or locate_ingest(env=env)
    if skill is None:
        return "MISSING", f"ingest-skill not found (set {ENV_DIR} or install `ingest`): URL inputs are handed over; {install_hint()}", {"via": "ingest-skill", "path": None, "version": None}
    doc = ingest_doctor(skill)
    checks = doc.get("checks") if isinstance(doc.get("checks"), dict) else {}
    chk = checks.get("yt-dlp")
    ev: Dict[str, Any] = {"via": "ingest-skill", "path": skill.describe(), "ingest_version": doc.get("version"), "version": None}
    if not isinstance(chk, dict):
        return "MISSING", f"ingest-skill doctor reports no yt-dlp check ({doc.get('problem') or 'an ingest-skill without fetch'}); {install_hint()}", ev
    ev["version"] = chk.get("version")
    if chk.get("status") == "ok" and chk.get("version"):
        return "AVAILABLE", str(chk["version"]), ev
    return "MISSING", f"ingest-skill: {chk.get('detail') or chk.get('status')}", ev


@dataclass
class IngestRecord:
    """Provenance for one downloaded source -- the same discipline already applied to every other artifact in this
    system (who / what / when / hash). Turned into an OBSERVED Observation (kind "ingest") on the asset the download
    became, once analysis has resolved the local file into an Asset. Taken from ingest-skill's fetch report: the URLs
    are the redacted ones (credentials and token-like parameter values removed by ingest-skill)."""
    source_url: str
    resolved_url: str
    video_id: str
    title: str
    path: str
    sha256: str
    yt_dlp_version: str
    is_live: bool
    was_live: bool
    live_from_start: bool
    duration: Optional[float] = None
    downloaded_at: str = field(default_factory=now_iso)
    extractor: Optional[str] = None
    format_id: Optional[str] = None
    redacted: List[str] = field(default_factory=list)   # names (never values) of what ingest-skill removed from the URLs
    manifest: Optional[str] = None      # the destination's ingest-manifest.json (ingest verify / inventory work on its folder)
    ingest_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"source_url": self.source_url, "resolved_url": self.resolved_url, "video_id": self.video_id, "title": self.title,
                "path": self.path, "sha256": self.sha256, "yt_dlp_version": self.yt_dlp_version, "is_live": self.is_live,
                "was_live": self.was_live, "live_from_start": self.live_from_start, "duration": self.duration, "downloaded_at": self.downloaded_at,
                "extractor": self.extractor, "format_id": self.format_id, "redacted": list(self.redacted), "manifest": self.manifest,
                "ingest_version": self.ingest_version}

    def observation(self, asset_id: str) -> Observation:
        """The OBSERVED record of this download: same source shape as any other tool measurement,
        `<package>/<tool>@<version>` (media/analysis.py's SOURCE_RE), never an AI provider, never simplified."""
        ver = self.ingest_version or "unknown"
        return Observation(kind="ingest", asset_id=asset_id, source=f"{TOOL}@{ver}", data=self.to_dict(),
                           provenance="OBSERVED", skill=PACKAGE, skill_version=ver, tool=TOOL,
                           external_id=self.video_id, fingerprint=self.sha256)


def _ingest_version(skill: CliSkill) -> str:
    code, out, _ = invoke(skill, ["contract", "--json"], timeout=60.0)
    try:
        return str(one_json_document(out, "ingest contract").get("version") or "") if code == 0 else ""
    except ToolError:
        return ""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def fetch_url(url: str, workspace: str, skill: Optional[CliSkill] = None, timeout: float = DOWNLOAD_TIMEOUT,
              env: Optional[Dict[str, str]] = None) -> IngestRecord:
    """Download `url` through `ingest fetch <url> --dest <workspace>/downloads --json`. Always inside the workspace, so the
    existing workspace-boundary policy (the media engine adapter's PathPolicy: inputs from an allowed root, outputs inside
    the workspace) admits the result with no special case. The file is <workspace>/downloads/<extractor>-<id>/<id>.<ext>.
    Errors: ingest-skill missing or without its fetch extra → ENGINE_MISSING (cli.py hands the URL over, ADR-051);
    URL_INVALID → URL_INVALID; FETCH_FAILED → DOWNLOAD_FAILED; a report with ok false → INCOMPLETE (the download is kept
    in ingest-skill's staging folder, named in the message); a report the agent cannot admit → INVALID_OUTPUT."""
    skill = skill or locate_ingest(env=env)
    if skill is None:
        raise IngestError("ENGINE_MISSING", f"ingest-skill not found (set {ENV_DIR} or install `ingest`); {install_hint()}")
    dest = (Path(workspace) / DOWNLOAD_SUBDIR).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    code, out, err = invoke(skill, ["fetch", url, "--dest", str(dest), "--json"], timeout=timeout)
    if code == 124:
        raise IngestError("TIMEOUT", f"ingest fetch exceeded {timeout}s")
    try:
        doc = one_json_document(out, "ingest fetch")
    except ToolError as e:
        raise IngestError("INVALID_OUTPUT", f"{e} (exit {code}) {err.strip()[-300:]}".strip())
    if isinstance(doc.get("error"), dict):
        ecode = str(doc["error"].get("code") or "")
        kind = ERROR_KINDS.get(ecode, "DOWNLOAD_FAILED")
        raise IngestError(kind, f"ingest fetch {ecode}: {str(doc['error'].get('message') or '')[:800]}" + (f"; {install_hint()}" if kind == "ENGINE_MISSING" else ""))
    if doc.get("schema") != REPORT_SCHEMA:
        raise IngestError("INVALID_OUTPUT", f"ingest fetch report schema {doc.get('schema')!r} != {REPORT_SCHEMA}")
    if doc.get("ok") is not True:
        staging = doc.get("staging") if isinstance(doc.get("staging"), dict) else {}
        problems = [f"{p.get('code')}: {p.get('detail')}" for d in doc.get("destinations") or [] if isinstance(d, dict) for p in d.get("problems") or [] if isinstance(p, dict)]
        raise IngestError("INCOMPLETE", f"ingest fetch did not verify every copy (exit {code}); the download is kept in {staging.get('path')}"
                                        + (f" -- {'; '.join(problems)[:600]}" if problems else ""))
    return _record(doc, dest, _ingest_version(skill))


def _record(doc: Dict[str, Any], dest: Path, ingest_version: str) -> IngestRecord:
    """Admission of the fetch report: the "fetch" record, a destination under <workspace>/downloads, its first file present
    there, and the sha256 the agent recomputes equal to the one ingest-skill verified."""
    f = doc.get("fetch")
    dests = doc.get("destinations")
    if not isinstance(f, dict) or f.get("schema") != FETCH_SCHEMA:
        raise IngestError("INVALID_OUTPUT", f"the fetch report has no {FETCH_SCHEMA} record")
    if not isinstance(dests, list) or not dests or not isinstance(dests[0], dict) or not dests[0].get("root"):
        raise IngestError("INVALID_OUTPUT", "the fetch report names no destination")
    files = [str(x) for x in f.get("files") or []]
    if not files:
        raise IngestError("INVALID_OUTPUT", "the fetch record lists no downloaded file")
    path = (Path(str(dests[0]["root"])) / files[0]).resolve()
    if not _inside(path, dest) or not path.is_file():
        raise IngestError("INVALID_OUTPUT", f"the downloaded file {path} is not a file under {dest}")
    entry = next((x for x in doc.get("files") or [] if isinstance(x, dict) and x.get("path") == files[0]), None) or {}
    reported = str((entry.get("hashes") or {}).get("sha256") or entry.get("sha256") or "")
    digest = sha256_file(str(path))
    if reported != digest:
        raise IngestError("INVALID_OUTPUT", f"sha256 of {path.name} is {digest}; the fetch report says {reported or 'nothing'}")
    downloader = f.get("downloader") if isinstance(f.get("downloader"), dict) else {}
    dur = f.get("duration")
    return IngestRecord(source_url=str(f.get("source_url") or ""), resolved_url=str(f.get("webpage_url") or f.get("source_url") or ""),
                        video_id=str(f.get("id") or ""), title=str(f.get("title") or ""), path=str(path), sha256=digest,
                        yt_dlp_version=str(downloader.get("version") or ""), is_live=bool(f.get("is_live")), was_live=bool(f.get("was_live")),
                        live_from_start=bool(f.get("live_from_start")), duration=float(dur) if isinstance(dur, (int, float)) and not isinstance(dur, bool) else None,
                        downloaded_at=str(f.get("fetched_at") or now_iso()), extractor=f.get("extractor"), format_id=f.get("format_id"),
                        redacted=[str(x) for x in f.get("redacted") or []], manifest=dests[0].get("manifest"), ingest_version=ingest_version)


def url_fetch_handoff(urls: List[str], reason: str, workspace: str) -> List[Dict[str, Any]]:
    """ADR-051 handoffs for URL inputs nothing installed here can download: the job, why, the owner (ingest-skill) and how to
    install it -- the same shape as Service.handoffs(). The agent does not download the URL any other way."""
    c = component(PACKAGE)
    dest = str(Path(workspace) / DOWNLOAD_SUBDIR)
    return [{"step": f"fetch_input_{i + 1}", "skill": "url_fetch", "job": "Download one video URL into the workspace, verified, before analysis (ingest fetch)",
             "inputs": [u], "params": {"dest": dest}, "reason": reason, "provided_by": [PACKAGE], "repositories": [repository(PACKAGE)],
             "packages": [c.to_dict() if c else {"package": PACKAGE, "repository": repository(PACKAGE)}], "install": install_commands(PACKAGE)}
            for i, u in enumerate(urls)]
