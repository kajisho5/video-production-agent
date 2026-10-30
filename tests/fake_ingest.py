"""Fake ingest-skill process for agent/ingest.py tests (ADR-059): speaks the slice of the real `ingest` CLI the agent uses --
`contract --json`, `doctor --json` and `fetch <url> --dest D --json` -- with the report shapes of ingest-skill's ADR-005
(`ingest/fetch-report@1` with the `ingest/fetch@1` record, destinations, files with hashes, staging). `fetch` writes a small
file to <dest>/<extractor>-<id>/<id>.mp4 and an ingest-manifest.json next to it; it never touches the network and never runs
yt-dlp. Controlled via FAKE_INGEST_MODE:

  ok | live | was_live | unavailable (no fetch extra: FETCH_UNAVAILABLE, doctor yt-dlp unavailable) | url_invalid |
  fetch_failed | incomplete (a conflict: ok false, staging kept) | bad_json | no_files | outside | hash_mismatch | wrong_schema |
  no_doctor_check | hang

FAKE_INGEST_CALLS: a file every invocation's argv is appended to (one JSON list per line). Test double only.

ADR-066 (card offload) adds `offload <card> --dest D [--dest D ...] [--project P] [--hash H] [--name-by folder|reel] [--reel R] [--dry-run] --json`,
`inventory <folder> [--reel R] --json` and `verify <folder> --json` in the shapes of ingest-skill main (ingest/offload-report@1,
ingest/inventory@2, ingest/verify-report@1, its manifest and its refusals), and `contract --json` returns the pinned contract
(src/video_agent/tools/ingest/contract_0.1.0.json). The copy is real (files are written to <dest>/<project>/<name>/...); the card
structure is read in miniature: every .mp4 / .mxf / .mov / .mts / .braw file is a recording, except that XDCAM EX clip folders
`CLPR/<take>_<nn>/<take>_<nn>.MP4` of one take are one spanned recording in <nn> order, and `<name>M01.XML` beside a recording is its
sidecar. Controlled via:

  FAKE_INGEST_OFFLOAD: ok | conflict (the first file exists with other content on the first destination) | partial (the second
    destination fails every file) | all_failed | mhl_invalid (the first destination's ASC MHL history is damaged) | source_changed
    (a SOURCE_CHANGED problem) | wrong_root | no_manifest | manifest_mismatch | bad_json | hang | inventory_outside | inventory_fails |
    refuse:<CODE> (the copy is refused with that ingest-skill error code; the dry run still answers)
  FAKE_INGEST_CONTRACT: pinned | drift_version (0.1.9: non-fatal) | drift_fatal (offload argv changed) | drift_error (a pinned
    error code's exit code changed) | new_error (an extra error code: non-fatal) | incompatible (version 0.2.0)
  FAKE_INGEST_XXHASH: "1" makes --hash xxh64 available (otherwise HASH_UNAVAILABLE, like ingest-skill without its extra)"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

MODE = os.environ.get("FAKE_INGEST_MODE", "ok")
CALLS_LOG = os.environ.get("FAKE_INGEST_CALLS")
VERSION = "0.1.0"
YT_DLP_VERSION = "2026.10.26"
VIDEO_ID = "vid0000001"
EXIT = {"URL_INVALID": 2, "FETCH_UNAVAILABLE": 2, "FETCH_FAILED": 3, "INCOMPLETE": 3, "SOURCE_NOT_FOUND": 2, "HASH_UNAVAILABLE": 2, "REEL_INVALID": 2,
        "NO_DESTINATION": 2, "DUPLICATE_DESTINATION": 2, "DESTINATION_OVERLAPS_SOURCE": 2, "MANIFEST_NOT_FOUND": 2}
OFFLOAD_MODE = os.environ.get("FAKE_INGEST_OFFLOAD", "ok")
CONTRACT_MODE = os.environ.get("FAKE_INGEST_CONTRACT", "pinned")
PINNED = Path(__file__).resolve().parents[1] / "src" / "video_agent" / "tools" / "ingest" / "contract_0.1.0.json"
MEDIA = (".mp4", ".mxf", ".mov", ".mts", ".braw")
JUNK = (".DS_Store", "Thumbs.db")
REEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


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


def _contract() -> dict:
    c = json.loads(PINNED.read_text(encoding="utf-8"))
    if CONTRACT_MODE == "drift_version":
        c["version"] = c["skill"]["version"] = "0.1.9"
    elif CONTRACT_MODE == "incompatible":
        c["version"] = c["skill"]["version"] = "0.2.0"
    elif CONTRACT_MODE == "drift_fatal":
        for t in c["tools"]:
            if t["tool_id"] == "ingest/offload":
                t["argv"] = t["argv"] + ["[--flatten]"]
    elif CONTRACT_MODE == "drift_error":
        c["errors"]["VERIFY_MISMATCH"] = {"exit_code": 4, "retryable": True}
    elif CONTRACT_MODE == "new_error":
        c["errors"]["DISK_FULL"] = {"exit_code": 3, "retryable": True}
    return c


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _scan(src: Path):
    out = []
    for dirpath, dirnames, filenames in os.walk(src):
        d = Path(dirpath)
        dirnames[:] = sorted(n for n in dirnames if not (d == src and n == "ascmhl"))
        for n in sorted(filenames):
            if n in JUNK or n.startswith("._") or (d == src and n == "ingest-manifest.json"):
                continue
            out.append((d / n).relative_to(src).as_posix())
    return out


def _opt(argv, name, many=False):
    vals = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == name]
    return vals if many else (vals[-1] if vals else None)


def _fingerprint(src: Path, rels) -> str:
    return "CARD-" + hashlib.sha256("\n".join(f"{r}\t{(src / r).stat().st_size}" for r in sorted(rels)).encode()).hexdigest()[:8].upper()


def offload(argv) -> int:
    src = Path(argv[1])
    dests = _opt(argv, "--dest", many=True)
    project, algo, name_by, reel = _opt(argv, "--project"), _opt(argv, "--hash") or "sha256", _opt(argv, "--name-by") or "folder", _opt(argv, "--reel")
    dry = "--dry-run" in argv
    if algo == "xxh64" and os.environ.get("FAKE_INGEST_XXHASH") != "1":
        return _error("HASH_UNAVAILABLE", "xxh64 needs the optional xxhash package, which is not installed: pip install 'ingest-skill[xxhash]'")
    if not dests:
        return _error("NO_DESTINATION", "at least one --dest is required")
    if reel is not None and not REEL_RE.match(reel):
        return _error("REEL_INVALID", f"reel {reel!r}: use 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit")
    if not src.is_dir():
        return _error("SOURCE_NOT_FOUND", f"source {src} is not a directory")
    if OFFLOAD_MODE.startswith("refuse:") and not dry:   # ingest-skill's own refusal with that code (e.g. a reel it rejects, the card unplugged)
        code = OFFLOAD_MODE.split(":", 1)[1]
        return _error(code, f"fake refusal {code}")
    if OFFLOAD_MODE == "hang" and not dry:
        time.sleep(30)
    if OFFLOAD_MODE == "bad_json" and not dry:
        print("copying...")
        return 0
    src = src.resolve()
    rels = _scan(src)
    reel_info = {"id": reel, "source": "operator", "fingerprint": _fingerprint(src, rels)} if reel else {"id": _fingerprint(src, rels), "source": "fingerprint", "fingerprint": _fingerprint(src, rels)}
    name = reel_info["id"] if name_by == "reel" else src.name
    roots = [Path(d).resolve() / project / name if project else Path(d).resolve() / name for d in dests]
    files = [{"path": r, "size": (src / r).stat().st_size, "sha256": None, "mtime": "2026-10-30T09:00:00Z", "status": "planned", "detail": "", "hashes": {}} for r in rels]
    out_dests = []
    for i, root in enumerate(roots):
        dd = {"root": str(root), "copied": 0, "skipped_identical": 0, "conflicts": 0, "failed": 0, "manifest": None, "mhl": None, "mhl_failed": 0, "problems": [], "warnings": []}
        out_dests.append(dd)
        if dry:
            continue
        root.mkdir(parents=True, exist_ok=True)
        entries, failed = [], []
        for f in files:
            digest = _sha(src / f["path"])
            f["hashes"] = {algo: digest, "c4": "c4fake"}
            f["sha256"] = digest if algo == "sha256" else None
            dst = root / f["path"]
            fails_here = (OFFLOAD_MODE == "partial" and i == 1) or OFFLOAD_MODE == "all_failed"
            if fails_here:
                dd["failed"] += 1
                f["status"] = "failed"
                f["detail"] = f"{dst}: [Errno 28] No space left on device"
                failed.append({"path": f["path"], "size": f["size"], "detail": f["detail"]})
                continue
            if OFFLOAD_MODE == "conflict" and i == 0 and f is files[0]:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(b"other content")
                dd["conflicts"] += 1
                f["status"] = "conflict"
                f["detail"] = f"{dst} exists with different content; not overwritten"
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / f["path"], dst)
            dd["copied"] += 1
            if f["status"] == "planned":
                f["status"] = "copied"
            entries.append({"path": f["path"], "size": f["size"], algo: ("0" * 64 if OFFLOAD_MODE == "manifest_mismatch" and f is files[0] else digest), "mtime": f["mtime"]})
        if OFFLOAD_MODE != "no_manifest":
            doc = {"schema": "ingest/manifest@1", "tool": f"ingest-skill {VERSION}", "source": str(src), "created_at": "2026-10-30T09:00:01Z", "hash": algo, "files": entries,
                   "card": {"reel": reel_info["id"], "reel_source": reel_info["source"]}}
            if failed:
                doc["failed"] = failed
            (root / "ingest-manifest.json").write_text(json.dumps(doc), encoding="utf-8")
            dd["manifest"] = str(root / "ingest-manifest.json")
        if entries:
            mhl = root / "ascmhl" / f"0001_{name}_2026-10-30_090001Z.mhl"
            mhl.parent.mkdir(parents=True, exist_ok=True)
            mhl.write_text("<hashlist version=\"2.0\"/>", encoding="utf-8")
            dd["mhl"] = str(mhl)
        if OFFLOAD_MODE == "mhl_invalid" and i == 0:
            dd["problems"].append({"code": "MHL_INVALID", "detail": f"{root}/ascmhl/ascmhl_chain.xml: generation 1 hash does not match"})
            dd["mhl_failed"] = 1
        if OFFLOAD_MODE == "source_changed" and i == 0:
            dd["problems"].append({"code": "SOURCE_CHANGED", "detail": f"{src / rels[0]} changed while it was being copied"})
    if OFFLOAD_MODE == "wrong_root" and not dry:
        out_dests[-1]["root"] = str(Path(dests[-1]).resolve() / "elsewhere")
    ok = all(d["failed"] == 0 and d["conflicts"] == 0 and d["mhl_failed"] == 0 and not d["problems"] for d in out_dests)
    _emit({"schema": "ingest/offload-report@1", "ok": ok, "dry_run": dry, "source": str(src), "hash": algo, "bytes_total": sum(f["size"] for f in files),
           "card": {"reel": reel_info, "name_by": name_by}, "destinations": out_dests, "files": files, "skipped": [],
           "started_at": "2026-10-30T09:00:00Z", "finished_at": "2026-10-30T09:00:02Z"})
    return 0 if ok else EXIT["INCOMPLETE"]


def _fake_facts(p: Path):
    try:
        raw = p.read_bytes()
        if raw.startswith(b'{"fake"'):
            d = json.loads(raw.decode())
            return {"probed": True, "duration_s": d.get("duration"), "container": "mov,mp4,m4a,3gp,3g2,mj2", "video_codec": "h264" if d.get("video", True) else None,
                    "width": 1280, "height": 720, "frame_rate": "30/1", "audio_codec": "aac", "audio_channels": d.get("channels", 2), "sample_rate": "48000",
                    "timecode": None, "creation_time": None}
    except (OSError, ValueError):
        pass
    return {"probed": False, "duration_s": None}


def inventory(argv) -> int:
    root = Path(argv[1])
    if not root.is_dir():
        return _error("SOURCE_NOT_FOUND", f"{root} is not a directory")
    if OFFLOAD_MODE == "inventory_fails":
        return _error("MANIFEST_INVALID", f"{root}/ingest-manifest.json is not an ingest manifest")
    root = root.resolve()
    rels = _scan(root)
    hashes, algo, recorded = {}, None, None
    man = root / "ingest-manifest.json"
    if man.is_file():
        doc = json.loads(man.read_text(encoding="utf-8"))
        algo = doc["hash"]
        hashes = {e["path"]: e[algo] for e in doc["files"]}
        recorded = doc.get("card")
    groups = {}
    for r in rels:
        p = Path(r)
        if p.suffix.lower() not in MEDIA:
            continue
        m = re.match(r"^(.+)_(\d\d)$", p.stem)
        key = m.group(1) if (m and "CLPR" in p.parts) else p.with_suffix("").as_posix()
        groups.setdefault(key, []).append(r)
    reel = _opt(argv, "--reel")
    if reel:
        reel_info = {"id": reel, "source": "operator", "fingerprint": _fingerprint(root, rels)}
    elif recorded:
        reel_info = {"id": recorded["reel"], "source": "manifest", "recorded_source": recorded.get("reel_source"), "fingerprint": _fingerprint(root, rels)}
    else:
        reel_info = {"id": _fingerprint(root, rels), "source": "fingerprint", "fingerprint": _fingerprint(root, rels)}
    clips = []
    for key in sorted(groups):
        members = []
        for r in sorted(groups[key]):
            facts = _fake_facts(root / r)
            members.append({"path": r, "size": (root / r).stat().st_size, "hash": {"algorithm": algo, "value": hashes[r]} if r in hashes else None,
                            "probed": facts["probed"], "duration_s": facts.get("duration_s")})
        if OFFLOAD_MODE == "inventory_outside":
            members[0]["path"] = "../escape.mp4"
        first = _fake_facts(root / groups[key][0])
        side = Path(members[0]["path"]).with_suffix("").as_posix() + "M01.XML"   # POSIX-relative, like ingest-skill's inventory
        sidecars = [{"path": side, "kind": "sony_nonrealtimemeta", "parsed": True, "metadata": {"device": {"manufacturer": "Sony", "model": "PMW-EX1"}}}] if side in rels else []
        spanned = len(members) > 1
        dur = round(sum(m["duration_s"] for m in members), 3) if all(m["probed"] and m["duration_s"] is not None for m in members) else None
        clips.append({"clip_id": Path(key).name, "structure": "sony_xdcam_ex" if "CLPR" in Path(members[0]["path"]).parts else "generic", "path": members[0]["path"],
                      "spanned": spanned, "incomplete": False, "size": sum(m["size"] for m in members),
                      "sha256": None if spanned or algo != "sha256" else (members[0]["hash"] or {}).get("value"), "hash": None if spanned else members[0]["hash"],
                      "members": members, "sidecars": sidecars, "proxies": [], "related": [], "metadata": sidecars[0]["metadata"] if sidecars else {},
                      **dict(first, duration_s=dur, probed=all(m["probed"] for m in members))})
    _emit({"schema": "ingest/inventory@2", "root": str(root), "card": {"reel": reel_info, "structures": [{"id": "sony_xdcam_ex", "root": "BPAV"}] if any(c["structure"] == "sony_xdcam_ex" for c in clips) else [],
                                                                      "index_files": [r for r in rels if r.upper().endswith("MEDIAPRO.XML")]},
           "clips": clips, "clip_count": len(clips), "file_count": sum(len(c["members"]) for c in clips), "spanned_count": sum(1 for c in clips if c["spanned"]),
           "total_duration_s": round(sum(c["duration_s"] or 0 for c in clips), 3), "warnings": [], "fetch": None, "probe": "ffprobe", "generated_at": "2026-10-30T09:00:03Z"})
    return 0


def verify(argv) -> int:
    root = Path(argv[1]).resolve()
    man = root / "ingest-manifest.json"
    if not man.is_file():
        return _error("MANIFEST_NOT_FOUND", f"no ingest-manifest.json and no ASC MHL history (ascmhl/ascmhl_chain.xml) at {root}")
    doc = json.loads(man.read_text(encoding="utf-8"))
    algo = doc["hash"]
    out = {"schema": "ingest/verify-report@1", "root": str(root), "hash": algo, "checked": 0, "ok_files": 0, "missing": [], "changed": [], "unlisted": [], "copy_failed": [],
           "failures": [], "warnings": [], "mhl": None}
    for e in doc["files"]:
        out["checked"] += 1
        f = root / e["path"]
        if not f.is_file():
            out["missing"].append(e["path"])
            out["failures"].append({"code": "FILE_MISSING", "path": e["path"]})
        elif _sha(f) != e[algo]:
            out["changed"].append(e["path"])
            out["failures"].append({"code": "VERIFY_MISMATCH", "path": e["path"], "hash": algo, "expected": e[algo], "actual": _sha(f), "against": "ingest-manifest.json"})
        else:
            out["ok_files"] += 1
    for e in doc.get("failed") or []:
        out["copy_failed"].append(e["path"])
        out["failures"].append({"code": "INCOMPLETE", "path": e["path"], "against": "ingest-manifest.json", "detail": f"the offload failed to copy this file here: {e.get('detail')}"})
    out["ok"] = not out["missing"] and not out["changed"] and not out["copy_failed"]
    _emit(out)
    return 0 if out["ok"] else 3


def main() -> int:
    argv = sys.argv[1:]
    _log(argv)
    if argv[:1] == ["--version"]:
        print(f"ingest-skill {VERSION}")
        return 0
    if argv[:1] == ["contract"]:
        _emit(_contract())
        return 0
    if argv[:1] == ["doctor"]:
        checks = {"python": {"status": "ok"}, "sha256": {"status": "ok"}}
        if MODE != "no_doctor_check":
            checks["yt-dlp"] = ({"status": "unavailable", "version": None, "detail": "optional package not installed: ingest fetch is UNAVAILABLE (refused, nothing written); pip install 'ingest-skill[fetch]'"}
                                if MODE == "unavailable" else {"status": "ok", "version": YT_DLP_VERSION, "detail": f"yt-dlp {YT_DLP_VERSION}: ingest fetch available"})
        xx = os.environ.get("FAKE_INGEST_XXHASH") == "1"
        checks["xxhash"] = {"status": "ok" if xx else "unavailable", "detail": "xxh64 / xxh3 / xxh128 available" if xx else "optional package not installed: pip install 'ingest-skill[xxhash]'"}
        _emit({"schema": "ingest/doctor@1", "ok": True, "checks": checks, "hash_algorithms": {"sha256": "ok", "xxh64": "ok" if xx else "unavailable"}, "version": VERSION})
        return 0
    if argv[:1] == ["fetch"] and "--dest" in argv and "--json" in argv:
        return fetch(argv[1], argv[argv.index("--dest") + 1])
    if argv[:1] == ["offload"] and "--json" in argv:
        return offload(argv)
    if argv[:1] == ["inventory"] and "--json" in argv:
        return inventory(argv)
    if argv[:1] == ["verify"] and "--json" in argv:
        return verify(argv)
    print("usage: ingest {contract,doctor,fetch,offload,inventory,verify} ...", file=sys.stderr)
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
