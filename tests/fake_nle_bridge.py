"""Fake nle-bridge-skill process for the ADR-086 tests: `contract --json` (the pinned src/video_agent/tools/nle_bridge/contract_0.1.0.json),
`doctor --json` and `export` / `conform` with `--flag=value` options and the project IR after `--`, in the result shapes of nle-bridge-skill main.
It reads the project IR as the real Skill does (the subject's `delivery.timelines` record, else INVALID_PLAN) and writes one small file per
format. Controlled via FAKE_NLE_MODE: ok | fail:<CODE> | bad_hash | outside; FAKE_NLE_CALLS: a file each argv is appended to. Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

PINNED = Path(__file__).resolve().parents[1] / "src" / "video_agent" / "tools" / "nle_bridge" / "contract_0.1.0.json"
EXT = {"otio": ".otio", "fcp7": ".xml", "edl": ".edl", "srt": ".srt"}


def emit(doc) -> None:
    sys.stdout.write(json.dumps(doc) + "\n")


def fail(code: str, message: str) -> int:
    c = json.loads(PINNED.read_text(encoding="utf-8"))
    emit({"ok": False, "error": {"code": code, "message": message, "detail": None}})
    return int(c["errors"].get(code, {}).get("exit_code", 1))


def sha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    argv = sys.argv[1:]
    if os.environ.get("FAKE_NLE_CALLS"):
        with open(os.environ["FAKE_NLE_CALLS"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\n")
    if argv[:1] == ["contract"]:
        sys.stdout.write(PINNED.read_text(encoding="utf-8"))
        return 0
    if argv[:1] == ["doctor"]:
        emit({"ok": True, "skill": "nle-bridge-skill", "version": "0.1.0", "problems": [], "notes": ["fake"]})
        return 0
    if argv[:1] not in (["export"], ["conform"]) or "--" not in argv:
        return fail("INVALID_ARGUMENT", f"usage: {argv}")
    opts, pos = argv[1:argv.index("--")], argv[argv.index("--") + 1:]
    kv = dict(a[2:].split("=", 1) for a in opts if a.startswith("--") and "=" in a)
    if any(a.startswith("-") and "=" not in a and a not in ("--json", "--overwrite", "--zip", "--no-probe") for a in opts) or len(pos) != 1:
        return fail("INVALID_ARGUMENT", f"unexpected arguments {argv}")
    mode = os.environ.get("FAKE_NLE_MODE", "ok")
    if mode.startswith("fail:"):
        return fail(mode.split(":", 1)[1], "fake refusal")
    doc = json.loads(Path(pos[0]).read_text(encoding="utf-8"))
    rec = next((t for t in (doc.get("delivery") or {}).get("timelines") or [] if t.get("asset") == kv.get("subject")), None)
    if rec is None:
        return fail("INVALID_PLAN", "the project IR has no delivery.timelines for the subject")
    out = kv["output"]
    os.makedirs(out, exist_ok=True)
    files = []
    for f in kv["formats"].split(","):
        p = os.path.join(out, f"tl{EXT[f]}")
        Path(p).write_text(json.dumps({"format": f, "clips": rec["clips"]}), encoding="utf-8")
        files.append({"format": f, "path": "/elsewhere/tl.otio" if mode == "outside" else p, "sha256": "0" * 64 if mode == "bad_hash" else sha(p), "verification": None})
    warns = [{"code": "NOT_IN_TIMELINE", "message": f"the agent's {w} is not in the timeline"} for d in doc.get("decisions") or []
             if d.get("subject") == "timeline.export" for w in (d.get("params") or {}).get("not_exported") or []]
    res = {"ok": True, "files": files, "warnings": warns, "timeline": {"name": "tl", "duration": rec.get("timeline_duration")}, "plan": pos[0]}
    if argv[0] == "conform":
        man = os.path.join(out, "manifest.json")
        Path(man).write_text("{}", encoding="utf-8")
        z = None
        if "--zip" in opts:
            zp = out.rstrip("/") + ".zip"
            Path(zp).write_bytes(b"PK fake")
            z = {"path": zp, "sha256": sha(zp)}
        res.update(package=out, zip=z, manifest=man, relink={"schema": "nle-bridge/relink@1", "rate": "25", "drop_frame": False,
                                                              "media": [{"name": "talk.mp4", "exists": True, "sha256": "a" * 64, "start_timecode": "00:00:00:00",
                                                                         "start_timecode_source": None}]})
    emit(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
