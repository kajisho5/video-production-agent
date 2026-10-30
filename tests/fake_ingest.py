"""Fake ingest-skill process for agent/ingest.py tests (ADR-059): speaks the slice of the real `ingest` CLI the agent uses --
`contract --json`, `doctor --json` and `fetch <url> --dest D --json` -- with the report shapes of ingest-skill's ADR-005
(`ingest/fetch-report@1` with the `ingest/fetch@1` record, destinations, files with hashes, staging). `fetch` writes a small
file to <dest>/<extractor>-<id>/<id>.mp4 and an ingest-manifest.json next to it; it never touches the network and never runs
yt-dlp. Controlled via FAKE_INGEST_MODE:

  ok | live | was_live | unavailable (no fetch extra: FETCH_UNAVAILABLE, doctor yt-dlp unavailable) | url_invalid |
  fetch_failed | incomplete (a conflict: ok false, staging kept) | bad_json | no_files | outside | hash_mismatch | wrong_schema |
  no_doctor_check | hang

FAKE_INGEST_CALLS: a file every invocation's argv is appended to (one JSON list per line). Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

MODE = os.environ.get("FAKE_INGEST_MODE", "ok")
CALLS_LOG = os.environ.get("FAKE_INGEST_CALLS")
VERSION = "0.1.0"
YT_DLP_VERSION = "2026.10.26"
VIDEO_ID = "vid0000001"
EXIT = {"URL_INVALID": 2, "FETCH_UNAVAILABLE": 2, "FETCH_FAILED": 3, "INCOMPLETE": 3}


def _log(argv) -> None:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\n")


def _emit(doc) -> None:
    print(json.dumps(doc, ensure_ascii=False, indent=2))


def _error(code: str, message: str) -> int:
    print(json.dumps({"ok": False, "error": {"code": code, "message": message}}, ensure_ascii=False))
    return EXIT[code]


def _redact(url: str):
    """The real Skill's rule in miniature: userinfo and token-like parameter values go, the names are reported."""
    removed = []
    scheme, rest = url.split("://", 1)
    host, sep, tail = rest.partition("/")
    if "@" in host:
        host = host.split("@", 1)[1]
        removed.append("userinfo")
    path, q, query = tail.partition("?")
    parts = []
    for kv in query.split("&") if query else []:
        k, eq, v = kv.partition("=")
        if eq and v and any(t in k.lower() for t in ("token", "sig", "key")):
            parts.append(f"{k}=REDACTED")
            removed.append(k)
        else:
            parts.append(kv)
    return f"{scheme}://{host}{sep}{path}" + (("?" + "&".join(parts)) if q else ""), removed


def fetch(url: str, dest: str) -> int:
    if not url.startswith(("http://", "https://")):
        return _error("URL_INVALID", f"{url}: fetch takes one http:// or https:// URL")
    if MODE == "url_invalid":
        return _error("URL_INVALID", f"{url}: fetch takes one http:// or https:// URL")
    if MODE == "unavailable":
        return _error("FETCH_UNAVAILABLE", "ingest fetch needs the optional yt-dlp package, which is not installed: pip install 'ingest-skill[fetch]'")
    if MODE == "fetch_failed":
        return _error("FETCH_FAILED", "yt-dlp: ERROR: [generic] Unable to download webpage: HTTP Error 404 (sign in required)")
    if MODE == "hang":
        time.sleep(30)
    if MODE == "bad_json":
        print("downloading... done")
        return 0
    shown, removed = _redact(url)
    name = f"generic-{VIDEO_ID}"
    root = Path(dest).resolve() / name
    root.mkdir(parents=True, exist_ok=True)
    media = root / f"{VIDEO_ID}.mp4"
    payload = b"FAKE-MP4-BYTES" + url.encode("utf-8")
    if MODE == "incomplete" and media.exists():
        payload += b"-different"
    else:
        media.write_bytes(payload)
    digest = hashlib.sha256(payload if MODE != "incomplete" else media.read_bytes()).hexdigest()
    reported = "0" * 64 if MODE == "hash_mismatch" else digest
    is_live = MODE == "live"
    files = [] if MODE == "no_files" else (["../../outside.mp4"] if MODE == "outside" else [media.name])
    record = {"schema": "ingest/fetch@1", "source_url": shown, "webpage_url": shown, "redacted": sorted(set(removed)), "extractor": "generic",
              "extractor_key": "Generic", "downloader": {"name": "yt-dlp", "version": YT_DLP_VERSION}, "format_selector": "bestvideo*+bestaudio/best",
              "format_id": "mp4", "id": VIDEO_ID, "title": "フェイク taped session", "duration": None if is_live else 42.5, "is_live": is_live,
              "was_live": MODE in ("was_live", "live") and not is_live, "live_from_start": is_live, "files": files,
              "started_at": "2026-10-30T00:00:00Z", "fetched_at": "2026-10-30T00:00:05Z"}
    manifest = root / "ingest-manifest.json"
    manifest.write_text(json.dumps({"schema": "ingest/manifest@1", "fetch": record}, ensure_ascii=False), encoding="utf-8")
    ok = MODE != "incomplete"
    staging = str(Path(dest).resolve().parent / "tmp-staging" / "ingest-fetch-fake")
    doc = {"schema": "ingest/fetch-report@1" if MODE != "wrong_schema" else "ingest/offload-report@1", "ok": ok, "dry_run": False, "source": staging + "/" + name,
           "hash": "sha256", "bytes_total": len(payload), "card": {"reel": {"id": name, "source": "folder"}, "name_by": "folder"},
           "destinations": [{"root": str(root), "copied": 1 if ok else 0, "skipped_identical": 0, "conflicts": 0 if ok else 1, "failed": 0, "manifest": str(manifest),
                             "mhl": None, "mhl_failed": 0, "problems": [] if ok else [{"code": "CONFLICT", "detail": f"{media.name} exists with different content"}], "warnings": []}],
           "files": [{"path": media.name, "size": len(payload), "sha256": reported, "mtime": None, "status": "copied" if ok else "conflict", "detail": "",
                      "hashes": {"sha256": reported, "c4": "c4fake"}}],
           "skipped": [], "started_at": "2026-10-30T00:00:00Z", "finished_at": "2026-10-30T00:00:06Z", "fetch": record,
           "staging": {"path": staging, "kept": not ok, "detail": "kept" if not ok else "removed: every destination is verified"}, "warnings": []}
    _emit(doc)
    return 0 if ok else EXIT["INCOMPLETE"]


def main() -> int:
    argv = sys.argv[1:]
    _log(argv)
    if argv[:1] == ["--version"]:
        print(f"ingest-skill {VERSION}")
        return 0
    if argv[:1] == ["contract"]:
        _emit({"skill_id": "ingest", "skill": {"id": "ingest", "package": "ingest-skill", "version": VERSION}, "version": VERSION})
        return 0
    if argv[:1] == ["doctor"]:
        checks = {"python": {"status": "ok"}, "sha256": {"status": "ok"}}
        if MODE != "no_doctor_check":
            checks["yt-dlp"] = ({"status": "unavailable", "version": None, "detail": "optional package not installed: ingest fetch is UNAVAILABLE (refused, nothing written); pip install 'ingest-skill[fetch]'"}
                                if MODE == "unavailable" else {"status": "ok", "version": YT_DLP_VERSION, "detail": f"yt-dlp {YT_DLP_VERSION}: ingest fetch available"})
        _emit({"schema": "ingest/doctor@1", "ok": True, "checks": checks, "version": VERSION})
        return 0
    if argv[:1] == ["fetch"] and "--dest" in argv and "--json" in argv:
        return fetch(argv[1], argv[argv.index("--dest") + 1])
    print("usage: ingest {contract,doctor,fetch} ...", file=sys.stderr)
    return 2


def checkout(root) -> str:
    """A fake ingest-skill checkout (<root>/src/ingest_skill/cli.py running this module), for VIDEO_AGENT_INGEST_DIR."""
    pkg = Path(root) / "src" / "ingest_skill"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "cli.py").write_text(f"import runpy\nrunpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')\n", encoding="utf-8")
    return str(root)


def install(bin_dir) -> str:
    """Write an executable `ingest` console-script shim into `bin_dir` (PATH lookup), like the real package's entry point."""
    bin_dir = Path(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":   # no shebangs on Windows: a .cmd launcher, which shutil.which finds through PATHEXT
        exe = bin_dir / "ingest.cmd"
        exe.write_text(f'@"{sys.executable}" "{Path(__file__).resolve()}" %*\r\n', encoding="utf-8")
        return str(exe)
    exe = bin_dir / "ingest"
    exe.write_text(f"#!{sys.executable}\nimport runpy\nrunpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')\n", encoding="utf-8")
    exe.chmod(0o755)
    return str(exe)


if __name__ == "__main__":
    sys.exit(main())
