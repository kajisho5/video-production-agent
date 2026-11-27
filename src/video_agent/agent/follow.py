"""Following a recording that is still being written (ADR-136; roadmap step 23): `video-agent follow <recording>`.

    every --interval s:  media-analysis-skill follow <recording> --kind silence --state <workspace>/follow/<id>/<kind>.state.json --json
    until it is complete (--final, --complete-marker, or --settle seconds with size and mtime unchanged)
    then:                the plan is made on the finished file (and rendered with --render); `--deadline-after 30m` is counted from the
                         moment the completion was detected

media-analysis-skill measures only what was added since its last look and keeps the result in its state file (its ADR-041). The agent
runs it, says how far the recording has got, and decides nothing until the file is complete. Matroska / WebM, MPEG-TS and fragmented MP4
can be read while they grow. A plain MP4 / MOV writes its index at the end, so it cannot be read until the recorder closes it: the agent
says so once and keeps waiting (the completion rule still applies to the file's size and mtime).

The Skill is driven through its command line (`follow` is a workflow command, not a contract@1 tool). Without the Skill the follow is
refused with how to install it; a Skill older than 0.3.0 (no `follow`) is refused as such.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import deadline as deadline_task

RULE = "follow_recording@1"
KINDS = ("silence", "scene_detection")
DEFAULT_INTERVAL = 10.0
DEFAULT_SETTLE = 30.0
MIN_INTERVAL = 0.2
PLAIN_MP4 = (".mp4", ".mov", ".m4v")
INSTALL_HINT = ("install media-analysis-skill 0.3.0 or later (set VIDEO_AGENT_MEDIA_ANALYSIS_DIR to a checkout, or put the `media-analysis` "
                "command on PATH); without it, wait until the recording has ended and run `video-agent plan` on the file")


class FollowError(RuntimeError):
    def __init__(self, message: str, code: str = "FOLLOW_FAILED"):
        super().__init__(message)
        self.code = code


def parse_after(text: Optional[str]) -> Optional[float]:
    """`--deadline-after 30m`: a duration after the recording ends (not a clock time)."""
    if text is None:
        return None
    t = str(text).strip()
    if ":" in t:
        raise deadline_task.DeadlineError(f"--deadline-after {text!r}: a duration after the recording ends (30m, 1h, 90s); for a clock time use plan --deadline HH:MM")
    return deadline_task.parse_deadline(t)


def check_options(interval: float, settle: float, kind: str) -> None:
    if not isinstance(interval, (int, float)) or interval < MIN_INTERVAL:
        raise FollowError(f"--interval must be at least {MIN_INTERVAL:g} s", "INVALID_REQUEST")
    if not isinstance(settle, (int, float)) or settle < 0:
        raise FollowError("--settle must be 0 or more seconds", "INVALID_REQUEST")
    if kind not in KINDS:
        raise FollowError(f"--kind must be one of {', '.join(KINDS)} (the kinds media-analysis-skill measures on a growing file)", "INVALID_REQUEST")


def state_dir(workspace: str, recording: str) -> Path:
    """<workspace>/follow/<stem>-<12 hex of the resolved path>: one follow state per recording."""
    p = str(Path(recording).resolve())
    return Path(workspace) / "follow" / f"{Path(p).stem}-{hashlib.sha256(p.encode('utf-8')).hexdigest()[:12]}"


def _iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t))


def _hms(s: Optional[float]) -> str:
    if s is None:
        return "?"
    s = max(0, int(round(s)))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def argv_for(recording: str, kind: str, state: str, workspace: str, settle: float, marker: Optional[str], final: bool) -> List[str]:
    argv = ["follow", recording, "--kind", kind, "--state", state, "--settle", f"{float(settle):g}", "--workspace", workspace,
            "--allowed-input", str(Path(recording).resolve().parent), "--json"]
    if marker:
        argv += ["--complete-marker", marker]
    if final:
        argv.append("--final")
    return argv


def _document(code: int, out: str, err: str) -> Dict[str, Any]:
    doc: Any = None
    for text in (out, (out.strip().splitlines() or [""])[-1]):
        try:
            doc = json.loads(text)
            break
        except ValueError:
            continue
    if code != 0 or not isinstance(doc, dict) or doc.get("status") != "ok":
        if "invalid choice: 'follow'" in (err or "") or "invalid choice: 'follow'" in (out or ""):
            raise FollowError("the installed media-analysis-skill has no `follow` (0.3.0 or later measures a growing file); " + INSTALL_HINT, "SKILL_TOO_OLD")
        e = (doc or {}).get("error") if isinstance(doc, dict) else None
        msg = (e.get("message") if isinstance(e, dict) else None) or (err or out or "").strip()[-400:] or f"exit {code}"
        raise FollowError(f"media-analysis follow failed: {msg}", (e.get("code") if isinstance(e, dict) else None) or "SKILL_FAILED")
    if doc.get("schema") != "media-analysis/follow@1":
        raise FollowError(f"media-analysis follow answered {doc.get('schema')!r}, not media-analysis/follow@1", "INVALID_OUTPUT")
    return doc


def follow(run: Optional[Callable[[List[str], float], "tuple[int, str, str]"]], recording: str, workspace: str, *, interval: float = DEFAULT_INTERVAL,
           settle: float = DEFAULT_SETTLE, marker: Optional[str] = None, final: bool = False, kind: str = "silence", max_wait: Optional[float] = None,
           once: bool = False, emit: Callable[[str], None] = print, clock: Callable[[], float] = time.time,
           sleep: Callable[[float], None] = time.sleep, timeout: float = 900.0) -> Dict[str, Any]:
    """Look at the recording every `interval` s until media-analysis-skill says it is complete (or once, with `once`). Returns the
    follow record (`follow_recording@1`). `run(argv, timeout)` runs media-analysis-skill; None means it is not installed."""
    if run is None:
        raise FollowError("media-analysis-skill is not installed; " + INSTALL_HINT, "SKILL_MISSING")
    check_options(interval, settle, kind)
    path = str(Path(recording).resolve())
    if not os.path.isfile(path):
        raise FollowError(f"{recording}: no such file (start the recording first, or give the file the recorder is writing)", "INPUT_MISSING")
    sdir = state_dir(workspace, path)
    sdir.mkdir(parents=True, exist_ok=True)
    state = str(sdir / f"{kind}.state.json")
    started = clock()
    ticks: List[Dict[str, Any]] = []
    said_unreadable = False
    last_size: Optional[int] = None
    while True:
        try:
            size = os.stat(path).st_size
        except OSError:
            raise FollowError(f"{recording} disappeared while it was being followed", "INPUT_MISSING")
        left = None if max_wait is None else max_wait - (clock() - started)
        if left is not None and left <= 0 and ticks:
            raise _timed_out(recording, max_wait, sdir)
        try:   # a look never runs past --max-wait by much: its timeout is what is left (at least 5 s, so a short budget can still look)
            doc = _document(*run(argv_for(path, kind, state, workspace, settle, marker, final), timeout if left is None else max(5.0, min(timeout, left))))
        except FollowError as e:
            if e.code != "UNSUPPORTED_FORMAT" or final:
                raise
            # the completion rule fired (no growth for --settle s, or the marker) on a file that cannot be opened yet: a writer that
            # buffers, or a plain MP4 / MOV not yet closed. Not complete; looked at again (--max-wait ends it if it never opens)
            doc = {"readable": False, "complete": False, "completion": {"reason": None}, "committed_until": None,
                   "unreadable": {"reason": "it has stopped growing but cannot be opened as media yet"}}
        now = clock()
        comp = doc.get("completion") or {}
        tick = {"at": round(now - started, 3), "size": size, "grew_bytes": None if last_size is None else size - last_size, "readable": bool(doc.get("readable")),
                "committed_until": doc.get("committed_until"), "complete": bool(doc.get("complete")), "reason": comp.get("reason"),
                "unchanged_for": comp.get("unchanged_for"), "restarted": doc.get("restarted")}
        ticks.append(tick)
        last_size = size
        if not doc.get("readable"):
            if not said_unreadable:
                plain = Path(path).suffix.lower() in PLAIN_MP4
                why = (doc.get("unreadable") or {}).get("reason") or "the file cannot be opened yet"
                emit(f"  {Path(path).name}: not readable while it is written ({why})."
                     + (" A plain MP4 / MOV writes its index (moov) when the recorder closes it, so nothing can be measured before then; "
                        "record to MKV, MPEG-TS or fragmented MP4 to follow a recording as it grows. Waiting for the file to close." if plain else " Waiting."))
                said_unreadable = True
        else:
            grew = "" if tick["grew_bytes"] is None else f", +{tick['grew_bytes'] / 1e6:.1f} MB since the last look"
            emit(f"  [{_hms(tick['at'])}] measured to {_hms(tick['committed_until'])}{grew}"
                 + (f"; unchanged for {float(tick['unchanged_for'] or 0):.0f} of {settle:g} s" if tick["grew_bytes"] == 0 and not tick["complete"] else "")
                 + (f" (measured again from the start: {doc['restarted']})" if doc.get("restarted") else ""))
        if doc.get("complete"):
            data = doc.get("data") or {}
            try:
                mtime = os.stat(path).st_mtime
            except OSError:
                mtime = None
            rec = {"rule": RULE, "recording": path, "kind": kind, "state": state, "interval": float(interval), "settle": float(settle),
                   "marker": marker, "complete": True, "reason": comp.get("reason"), "detected_at": _iso(now), "detected_epoch": round(now, 3),
                   "last_written_at": _iso(mtime) if mtime else None, "duration": data.get("duration"), "looks": len(ticks),
                   "skill_runs": doc.get("run_count"), "measured_seconds": doc.get("measured_seconds"), "unreadable_while_growing": said_unreadable,
                   "media_analysis": (doc.get("skill") or {}).get("version"), "followed_seconds": round(now - started, 3)}
            emit(f"  complete ({rec['reason']}) at {rec['detected_at']}: {_hms(rec['duration'])} recorded")
            (sdir / "follow.json").write_text(json.dumps(dict(rec, ticks=ticks), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return rec
        if once:
            return {"rule": RULE, "recording": path, "kind": kind, "state": state, "complete": False, "readable": tick["readable"],
                    "committed_until": tick["committed_until"], "looks": 1, "unreadable_while_growing": said_unreadable}
        if max_wait is not None and now - started >= max_wait:
            raise _timed_out(recording, max_wait, sdir)
        sleep(float(interval) if max_wait is None else min(float(interval), max(0.0, max_wait - (now - started))))


def _timed_out(recording: str, max_wait: float, sdir: Path) -> FollowError:
    return FollowError(f"{recording} was not complete after {_hms(max_wait)} (--max-wait); the follow state is kept in {sdir}, "
                       "so running follow again continues from it", "TIMEOUT")


def deadline_text(after_seconds: float, detected_epoch: float, now: Optional[float] = None) -> str:
    """The budget left at plan time for a deadline `after_seconds` after the completion was detected, as a duration plan --deadline takes.
    Already past: one second (the plan is then at risk, and says so)."""
    now = time.time() if now is None else now
    left = detected_epoch + after_seconds - now
    return f"{max(1, int(left))}s"
