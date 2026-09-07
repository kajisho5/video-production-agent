"""URL ingestion (video-production-agent issue #50, Task 1): `plan` / `analyze` accept a URL as well as a local
file path. A URL is downloaded once, up front, into the workspace before anything else runs; from that point on the
pipeline sees only a local file, exactly like any other input -- cli.py rewrites the input list before analysis ever
starts, so nothing downstream (probing, requirements, decisions, the compiler, execution) ever special-cases a URL.

yt-dlp is a bare external engine dependency, the same shape as the other command-line engines this agent locates
without a Skill contract (ARCHITECTURE_REVIEW SS1.7): no JSON stdin/stdout protocol, no pinned version window --
just a versioned binary, located on PATH or via an explicit override, its version reported in provenance and its
availability reported by `doctor` (capabilities/resolver.py) the same way those other bare engines are.

Live vs. VOD is decided from yt-dlp's own `--dump-json` metadata, never guessed from the URL shape: `is_live` true
means the stream is still running and needs `--live-from-start` to capture it from the beginning; `was_live` true
with `is_live` false means the stream already ended and downloads exactly like any ordinary video. Getting this
backwards either truncates an in-progress stream to "from now on" or hangs waiting on a stream that already ended.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from ..media.analyzer import sha256_file
from ..models import Observation, now_iso

ENV_YTDLP = "VIDEO_AGENT_YTDLP"          # explicit path to the yt-dlp executable (checked before PATH)
PROBE_TIMEOUT = 60.0                     # --dump-json fetches metadata only, never media bytes
DOWNLOAD_TIMEOUT = 3600.0
DOWNLOAD_SUBDIR = "downloads"
# video+audio merged into one file; yt-dlp falls back to a progressive single-file stream itself when a site (or a
# livestream in progress) has no separate adaptive streams to merge. This is yt-dlp's own selector vocabulary, not
# an ffmpeg flag -- the agent never speaks ffmpeg to anything but ffmpeg-skill.
FORMAT_SELECTOR = "bestvideo*+bestaudio/best"
FAILURE_KINDS = ("ENGINE_MISSING", "PROBE_FAILED", "DOWNLOAD_FAILED", "TIMEOUT", "INVALID_OUTPUT")


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


@dataclass
class YtDlp:
    command: List[str]     # argv prefix: ["yt-dlp"], an explicit path, or the ENV_YTDLP override
    version: str

    def describe(self) -> str:
        return self.command[0]


def locate_yt_dlp(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[YtDlp]:
    """explicit -> VIDEO_AGENT_YTDLP -> PATH -- the same override order the other external engines resolve their
    own location in (tools/*/locate.py), but with no checkout to find: like those other bare binaries, yt-dlp is a
    single versioned executable, confirmed by actually running `--version` (a name on PATH that fails to run is
    not AVAILABLE)."""
    env = os.environ if env is None else env
    exe = explicit or env.get(ENV_YTDLP) or shutil.which("yt-dlp", path=env.get("PATH"))
    if not exe:
        return None
    try:
        p = subprocess.run([exe, "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0 or not p.stdout.strip():
        return None
    return YtDlp(command=[exe], version=p.stdout.strip().splitlines()[-1].strip())


@dataclass
class IngestRecord:
    """Provenance for one downloaded source -- the same discipline already applied to every other artifact in this
    system (who / what / when / hash). Turned into an OBSERVED Observation (kind "ingest") on the asset the download
    became, once analysis has resolved the local file into an Asset."""
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

    def to_dict(self) -> Dict[str, Any]:
        return {"source_url": self.source_url, "resolved_url": self.resolved_url, "video_id": self.video_id, "title": self.title,
                "path": self.path, "sha256": self.sha256, "yt_dlp_version": self.yt_dlp_version, "is_live": self.is_live,
                "was_live": self.was_live, "live_from_start": self.live_from_start, "duration": self.duration, "downloaded_at": self.downloaded_at}

    def observation(self, asset_id: str) -> Observation:
        """The OBSERVED record of this download: same source shape as any other tool measurement,
        `<package>/<tool>@<version>` (media/analysis.py's SOURCE_RE), never an AI provider, never simplified."""
        return Observation(kind="ingest", asset_id=asset_id, source=f"yt-dlp/download@{self.yt_dlp_version}", data=self.to_dict(),
                            provenance="OBSERVED", skill="yt-dlp", skill_version=self.yt_dlp_version, tool="yt-dlp/download",
                            external_id=self.video_id, fingerprint=self.sha256)


def probe_url(url: str, yt_dlp: YtDlp, timeout: float = PROBE_TIMEOUT) -> Dict[str, Any]:
    """yt-dlp's own metadata for `url`, never guessed: `is_live` / `was_live` decide the live-vs-VOD branch,
    id / title / duration / webpage_url feed provenance. No media bytes are fetched here."""
    argv = yt_dlp.command + ["--dump-json", "--no-warnings", "--no-playlist", url]
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise IngestError("TIMEOUT", f"yt-dlp --dump-json exceeded {timeout}s for {url}")
    if p.returncode != 0 or not p.stdout.strip():
        raise IngestError("PROBE_FAILED", (p.stderr or "yt-dlp produced no metadata").strip()[-800:])
    try:
        return json.loads(p.stdout.splitlines()[0])
    except (ValueError, IndexError) as exc:
        raise IngestError("PROBE_FAILED", f"yt-dlp --dump-json did not return valid JSON: {exc}")


def download_url(url: str, workspace: str, yt_dlp: Optional[YtDlp] = None, timeout: float = DOWNLOAD_TIMEOUT,
                  env: Optional[Dict[str, str]] = None) -> IngestRecord:
    """Download `url` (video+audio merged into one file) into <workspace>/downloads/. Always inside the workspace,
    so the existing workspace-boundary policy (the media engine adapter's PathPolicy: inputs from an allowed root,
    outputs inside the workspace) admits the result with no special case -- the workspace itself is always an
    allowed root. Chooses `--live-from-start` only when yt-dlp's own metadata says the stream is still live; an
    already-ended stream (`was_live`, `is_live` false) downloads exactly like any ordinary video."""
    yt_dlp = yt_dlp or locate_yt_dlp(env=env)
    if yt_dlp is None:
        raise IngestError("ENGINE_MISSING", f"yt-dlp not found (set {ENV_YTDLP} or install it on PATH)")
    meta = probe_url(url, yt_dlp, timeout=min(timeout, PROBE_TIMEOUT))
    is_live, was_live = bool(meta.get("is_live")), bool(meta.get("was_live"))
    video_id = str(meta.get("id") or "")
    if not video_id:
        raise IngestError("PROBE_FAILED", f"yt-dlp metadata for {url} has no video id")
    dest_dir = Path(workspace) / DOWNLOAD_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_template = dest_dir / f"{video_id}.%(ext)s"
    argv = yt_dlp.command + ["--no-playlist", "-f", FORMAT_SELECTOR, "--merge-output-format", "mp4",
                              "-o", str(out_template), "--print", "after_move:filepath"]
    if is_live:
        argv.append("--live-from-start")   # in-progress stream: capture from its beginning, not from "now"
    argv.append(url)
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise IngestError("TIMEOUT", f"yt-dlp download exceeded {timeout}s for {url}")
    if p.returncode != 0:
        raise IngestError("DOWNLOAD_FAILED", (p.stderr or "yt-dlp exited non-zero with no stderr").strip()[-800:])
    lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    out_path = lines[-1] if lines else ""
    if not out_path or not Path(out_path).is_file():
        raise IngestError("INVALID_OUTPUT", f"yt-dlp reported no downloaded file for {url}")
    resolved = str(Path(out_path).resolve())
    return IngestRecord(source_url=url, resolved_url=str(meta.get("webpage_url") or url), video_id=video_id,
                         title=str(meta.get("title") or ""), path=resolved, sha256=sha256_file(resolved),
                         yt_dlp_version=yt_dlp.version, is_live=is_live, was_live=was_live, live_from_start=is_live,
                         duration=meta.get("duration"))
