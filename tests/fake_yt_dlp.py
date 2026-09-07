"""Fake yt-dlp binary for agent/ingest.py tests: speaks just enough of the real CLI surface (`--version`,
`--dump-json`, and a download invocation ending in `--print after_move:filepath <url>`) to test locate / probe /
download without any network access. Controlled via FAKE_YTDLP_MODE = ok | live | was_live | probe_fail |
bad_json | no_id | download_fail | no_output | hang. FAKE_YTDLP_CALLS, if set, logs one line per invocation
(the argv actually received) so a test can assert `--live-from-start` was (or was not) passed. Never touches the
network, never runs a real download. Test double only."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MODE = os.environ.get("FAKE_YTDLP_MODE", "ok")
VERSION = os.environ.get("FAKE_YTDLP_VERSION", "2026.08.19")
CALLS_LOG = os.environ.get("FAKE_YTDLP_CALLS")
VIDEO_ID = "vid0000001"


def _log(argv):
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\n")


def _meta(url: str) -> dict:
    is_live = MODE == "live"
    was_live = MODE in ("was_live", "download_fail", "no_output")
    return {"id": "" if MODE == "no_id" else VIDEO_ID, "title": "フェイク taped session", "webpage_url": url,
            "duration": None if is_live else 42.5, "is_live": is_live, "was_live": was_live}


def main() -> int:
    argv = sys.argv[1:]
    _log(argv)
    if MODE == "hang":
        time.sleep(30)
        return 0
    if argv[:1] == ["--version"]:
        print(VERSION)
        return 0
    if "--dump-json" in argv:
        if MODE == "probe_fail":
            print("ERROR: fake probe failure", file=sys.stderr)
            return 1
        if MODE == "bad_json":
            print("not json at all")
            return 0
        print(json.dumps(_meta(argv[-1]), ensure_ascii=False))
        return 0
    # download invocation
    if MODE == "download_fail":
        print("ERROR: fake download failure (sign in required)", file=sys.stderr)
        return 1
    url = argv[-1]
    if MODE == "no_output":
        return 0   # exit 0 but print nothing: caller must treat this as a failure, not guess a path
    out_template = argv[argv.index("-o") + 1]
    out_path = out_template.replace("%(ext)s", "mp4")
    Path(out_path).write_bytes(b"FAKE-MP4-BYTES" + os.urandom(32))
    print(out_path)
    return 0


def install(bin_dir) -> str:
    """Write an executable `yt-dlp` shim into `bin_dir` (a shebang'd script that runs this module) and return its
    path -- so a test can point `locate_yt_dlp`'s `explicit` / `VIDEO_AGENT_YTDLP` / PATH at a real, runnable
    executable without a real yt-dlp installation."""
    bin_dir = Path(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    exe = bin_dir / "yt-dlp"
    exe.write_text(f"#!/usr/bin/env python3\nimport runpy\nrunpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')\n", encoding="utf-8")
    exe.chmod(0o755)
    return str(exe)


if __name__ == "__main__":
    sys.exit(main())
