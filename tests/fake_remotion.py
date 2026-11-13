"""Fake remotion-ffmpeg-skill checkout for the ADR-093 tests: copied as <dir>/scripts/_contract.py, doctor.py and pipeline.py, it acts by the name it
runs under -- `_contract.py --json` (the pinned src/video_agent/tools/remotion/contract_0.14.0.json), `doctor.py --json [--project=DIR]` (checks
python / node / npm / ffmpeg-skill / ffmpeg, and the project's node_modules/remotion), and `pipeline.py <video> --project=P --comp=C ... --out-dir=D
--json --timeout=T` in the result shape of remotion-ffmpeg-skill v0.14.0 (status, output, timeline, render, look, verified, warnings, steps). The render
is a self-describing fake media file (tests/pipeline_harness.py) as long as the input, written to D/<stem>_<comp>.mp4. Nothing renders.
Controlled via:

  FAKE_RF_READY: all (default) | nonode
  FAKE_RF_MODE: ok | fail:<kind> | unverified | elsewhere
  FAKE_RF_CALLS: a file every invocation's argv is appended to (one JSON list per line, the script name first). Test double only."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MODE = os.environ.get("FAKE_RF_MODE", "ok")
READY = os.environ.get("FAKE_RF_READY", "all")
PINNED = os.environ.get("FAKE_RF_PIN", "")
NAME = Path(sys.argv[0]).stem


def emit(doc):
    sys.stdout.write(json.dumps(doc) + "\n")


def opts(argv):
    o, pos = {}, []
    for a in argv:
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            o[k] = v
        elif a.startswith("--"):
            o[a[2:]] = True
        else:
            pos.append(a)
    return o, pos


def main(argv):
    if os.environ.get("FAKE_RF_CALLS"):
        with open(os.environ["FAKE_RF_CALLS"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps([NAME] + argv) + "\n")
    if NAME == "_contract":
        emit(json.loads(Path(PINNED).read_text(encoding="utf-8")))
        return 0
    o, pos = opts(argv)
    if NAME == "doctor":
        checks = [{"check": "python", "ok": True, "value": "3.11", "fix": ""}, {"check": "node", "ok": READY != "nonode", "value": None if READY == "nonode" else "v22",
                   "fix": "install Node.js 18+ (https://nodejs.org)" if READY == "nonode" else ""},
                  {"check": "npm", "ok": True, "value": "10", "fix": ""}, {"check": "ffmpeg-skill", "ok": True, "value": "fake", "fix": ""}, {"check": "ffmpeg", "ok": True, "value": "6.1", "fix": ""}]
        if o.get("project"):
            ok = os.path.isfile(os.path.join(o["project"], "node_modules", "remotion", "package.json"))
            checks.append({"check": "remotion project", "ok": ok, "value": o["project"], "fix": "" if ok else f"cd {o['project']} && npm install"})
        emit({"status": "completed", "ok": all(c["ok"] for c in checks), "version": "0.14.0", "checks": checks, "ffmpeg_skill_doctor": None})
        return 0
    if NAME == "pipeline":
        if MODE.startswith("fail:"):
            kind = MODE.split(":", 1)[1]
            emit({"status": "failed", "error": f"refused (fake {kind})", "kind": kind, "dry_run": False})
            return {"input": 2, "missing_tool": 3, "timeout": 4}.get(kind, 1)
        src = pos[0]
        try:
            meta = json.loads(Path(src).read_bytes().decode())
        except ValueError:
            meta = {"duration": 10.0}
        out_dir = Path(o["out-dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        render = out_dir / f"{Path(src).stem}_{o['comp']}.mp4"
        render.write_bytes(json.dumps({"fake": True, "duration": meta.get("duration", 10.0), "lufs": meta.get("lufs", -11.0), "remotion": o["comp"]}).encode())
        shown = str(render) + (".elsewhere.mp4" if MODE == "elsewhere" else "")
        emit({"status": "completed", "dry_run": False, "output": shown, "timeline": str(out_dir / f"{Path(src).stem}.timeline.json"), "render": shown,
              "deliverable": None, "look": str(out_dir / "sheet.png"), "verified": MODE != "unverified", "warnings": [],
              "steps": [{"step": "measure", "status": "completed", "output": "t.json", "command": "analyze.py"}, {"step": "draw", "status": "completed", "output": str(render), "command": "render.py"}]})
        return 0
    emit({"status": "failed", "error": f"unknown script {NAME}", "kind": "input"})
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
