"""Fake publish-skill process for the ADR-087 tests: speaks the slice of the real `publish` CLI the agent uses -- `contract --json` (the pinned
src/video_agent/tools/publish/contract_0.1.0.json), `doctor --json --offline` (publish/doctor@1 with checks.platforms) and `plan` / `upload` with
`--flag=value` options and the deliverable after `--`, in the result shapes of publish-skill main (publish/plan@1; the dry run with its
confirm_token = the real formula over platform, sha256, size and the canonical metadata; the upload result; its error document). Nothing touches
a network: an upload with the right token writes a receipt line to FAKE_PUB_UPLOADS instead.
Controlled via:

  FAKE_PUB_VERDICT: PASS (default) | WARN | FAIL | INCOMPLETE (the plan's verdict)
  FAKE_PUB_MODE: ok | fail:<CODE> (upload --confirm refuses with that publish code) | dry_wrong_sha (the dry run reports another sha256)
  FAKE_PUB_CONTRACT: pinned | drift_platform (a new platform appears: non-fatal) | drift_confirm (confirmation from config: fatal) | gone_platform (x's uploader removed: fatal)
  FAKE_PUB_CALLS: a file every invocation's argv is appended to (one JSON list per line)
  FAKE_PUB_UPLOADS: a file each completed upload appends one JSON line to. Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

PINNED = Path(__file__).resolve().parents[1] / "src" / "video_agent" / "tools" / "publish" / "contract_0.1.0.json"
CALLS_LOG = os.environ.get("FAKE_PUB_CALLS")


def _log(argv) -> None:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\n")


def _emit(doc) -> None:
    sys.stdout.write(json.dumps(doc) + "\n")


def _contract() -> dict:
    c = json.loads(PINNED.read_text(encoding="utf-8"))
    mode = os.environ.get("FAKE_PUB_CONTRACT", "pinned")
    if mode == "drift_platform":
        c["platforms"]["registered"]["linkedin"] = dict(c["platforms"]["registered"]["x"], id="linkedin", name="LinkedIn")
        next(t for t in c["tools"] if t["tool_id"] == "publish/upload")["uploaders"]["linkedin"] = True
    elif mode == "drift_confirm":
        c["confirmation"]["from_environment_or_config"] = True
    elif mode == "gone_platform":
        next(t for t in c["tools"] if t["tool_id"] == "publish/upload")["uploaders"]["x"] = False
    return c


def _error(code: str, message: str, details=None) -> int:
    errs = _contract()["errors"]
    _emit({"ok": False, "skill": {"id": "publish", "version": "0.1.0"}, "error": {"code": code, "message": message, "retryable": bool((errs.get(code) or {}).get("retryable")),
                                                                                   "details": details or {}}})
    return int((errs.get(code) or {}).get("exit_code", 7))


def _sha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _parse(argv):
    if "--" not in argv:
        return None, None, None
    opts, pos = argv[:argv.index("--")], argv[argv.index("--") + 1:]
    kv, tags, caps, langs = {}, [], [], []
    for a in opts:
        if a in ("--json", "--offline"):
            continue
        if not a.startswith("--") or "=" not in a:
            return None, None, None
        k, v = a[2:].split("=", 1)
        if k == "tag":
            tags.append(v)
        elif k == "caption":
            caps.append(v)
        elif k == "caption-language":
            langs.append(v)
        else:
            kv[k] = v
    md = {"title": kv.get("title"), "description": kv.get("description"), "tags": tags, "privacy": kv.get("privacy", "private"),
          "thumbnail": {"path": kv["thumbnail"], "sha256": _sha(kv["thumbnail"]), "size": os.path.getsize(kv["thumbnail"])} if kv.get("thumbnail") else None,
          "captions": [{"path": c, "sha256": _sha(c), "size": os.path.getsize(c), "language": l} for c, l in zip(caps, langs)]}
    return kv, pos, md


def _token(platform: str, sha: str, size: int, md: dict) -> str:
    canon = json.dumps({k: (v if k not in ("thumbnail", "captions") else (None if v is None else ([{x: c[x] for x in ("sha256", "size", "language")} for c in v] if k == "captions"
                                                                                                   else {x: v[x] for x in ("sha256", "size")})))
                        for k, v in md.items()}, sort_keys=True, ensure_ascii=False)
    return "confirm-" + hashlib.sha256(f"publish/upload\n{platform}\n{sha}\n{size}\n{canon}".encode("utf-8")).hexdigest()[:16]


def main() -> int:
    argv = sys.argv[1:]
    _log(argv)
    c = _contract()
    if argv[:1] == ["contract"]:
        _emit(c)
        return 0
    if argv[:1] == ["doctor"]:
        reg = c["platforms"]["registered"]
        _emit({"schema": "publish/doctor@1", "ok": True, "status": "degraded", "skill": {"id": "publish", "version": c["version"]},
               "checks": {"platforms": {pid: dict(p, credentials={"complete": pid == "youtube", "variables": {v: "absent" for v in p.get("credential_env") or []}}) for pid, p in reg.items()}},
               "problems": [], "warnings": []})
        return 0
    if argv[:1] not in (["plan"], ["upload"]):
        print("usage: publish {contract,doctor,plan,upload}", file=sys.stderr)
        return 2
    kv, pos, md = _parse(argv[1:])
    if kv is None or len(pos) != 1:
        return _error("INVALID_ARGUMENT", f"the fake expects --flag=value options and one deliverable after --: {argv}")
    f, platform = pos[0], kv.get("platform")
    if platform not in c["platforms"]["registered"]:
        return _error("UNKNOWN_PLATFORM", f"unknown platform {platform}")
    if not os.path.isfile(f):
        return _error("INPUT_NOT_FOUND", f"not a regular file: {f}")
    item = {"path": os.path.abspath(f), "size": os.path.getsize(f), "sha256": _sha(f)}
    base = {"skill": {"id": "publish", "version": c["version"]}, "platform": {"id": platform, "name": c["platforms"]["registered"][platform]["name"]}, "input": item, "metadata": md}
    if argv[0] == "plan":
        v = os.environ.get("FAKE_PUB_VERDICT", "PASS")
        rows = [{"check": "video.max_bytes", "status": "PASS", "value": item["size"], "limit": 1, "owner": "publish"}]
        if v != "PASS":
            rows.append({"check": "title.max_chars", "status": {"WARN": "WARN", "FAIL": "FAIL", "INCOMPLETE": "UNCHECKED"}[v], "value": 999, "limit": 100, "owner": "publish", "fix": "shorten the title"})
        _emit(dict(base, schema="publish/plan@1", ok=v in ("PASS", "WARN"), verdict=v, checks=rows, uploaded=False, network_used=False, next_step="fake"))
        return 0 if v in ("PASS", "WARN") else 1
    token = _token(platform, item["sha256"], item["size"], md)
    if "confirm" not in kv:
        if os.environ.get("FAKE_PUB_MODE") == "dry_wrong_sha":
            item = dict(item, sha256="0" * 64)
        _emit(dict(base, input=item, ok=True, dry_run=True, uploaded=False, network_used=False, confirm_token=token,
                   credentials={"source": None, "complete": False, "variables": {"PUBLISH_YOUTUBE_REFRESH_TOKEN": "absent"}}, next_step="fake"))
        return 0
    if kv["confirm"] != token:
        return _error("CONFIRM_MISMATCH", "the --confirm token does not match; nothing was uploaded", {"uploaded": False})
    mode = os.environ.get("FAKE_PUB_MODE", "ok")
    if mode.startswith("fail:"):
        return _error(mode.split(":", 1)[1], "fake upload refused", {"uploaded": False})
    if os.environ.get("FAKE_PUB_UPLOADS"):
        with open(os.environ["FAKE_PUB_UPLOADS"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"platform": platform, "sha256": item["sha256"], "metadata": md}) + "\n")
    _emit(dict(base, ok=True, dry_run=False, uploaded=True, network_used=True, confirm_token=token,
               result={"video_id": "fakevid123", "url": "https://example.invalid/watch?v=fakevid123", "privacy": md["privacy"], "warnings": []}))
    return 0


def checkout(root) -> str:
    """A fake publish-skill checkout (<root>/src/publish_skill/cli.py running this module), for VIDEO_AGENT_PUBLISH_DIR."""
    pkg = Path(root) / "src" / "publish_skill"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "cli.py").write_text(f"import runpy\nrunpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')\n", encoding="utf-8")
    return str(root)


if __name__ == "__main__":
    sys.exit(main())
