"""Fake hyperframes-skill checkout for the ADR-114 tests: copied as <dir>/bin/hyperframes-skill.mjs (run by Python, not Node: the adapter only
needs an argv), it acts by its first argument -- `contract --json` (the pinned skills/pins/hyperframes-skill_<version>.json), `doctor --json`
(capabilities and tools.<tool>.usable), `template <name> --values=F --output=R --json`, `scene <request> --output=D --json` and
`render <scene> --output=O --codec=C [--segments] --json`, in the result shapes of hyperframes-skill v0.9.0 (status, verified, verification[],
commands[], expected, alpha, hyperframes, ffmpeg_skill, segments[], loop_seam). The render is a small file; nothing renders.
Controlled via:

  FAKE_HF_READY: all (default) | nochromium
  FAKE_HF_MODE: ok | fail:<kind> | noalpha | shortseg | seam | unverified | elsewhere
  FAKE_HF_CALLS: a file every invocation's argv is appended to (one JSON list per line, the tool first). Test double only."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MODE = os.environ.get("FAKE_HF_MODE", "ok")
READY = os.environ.get("FAKE_HF_READY", "all")
PINNED = os.environ.get("FAKE_HF_PIN", "")
KINDS = {"input": ("INPUT_INVALID", 1), "missing_tool": ("DEPENDENCY_MISSING", 127), "render": ("RENDER_FAILED", 1), "timeout": ("TIMEOUT", 124)}


def emit(doc):
    sys.stdout.write(json.dumps(doc, indent=2) + "\n")


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


def done(output, **kw):
    return dict({"status": "completed", "exit_code": 0, "output": output, "dry_run": False, "verified": True, "verification": [{"step": "exists", "ok": True}],
                 "commands": []}, **kw)


def segments_of(req):
    s = req.get("segments")
    if not s:
        return None
    out, start = [], 0.0
    for name in ("in", "loop", "out"):
        d = float(s[name])
        out.append({"name": name, "start": start, "duration": s[name], "start_frame": int(round(start * 30)), "frames": int(round(d * 30))})
        start += d
    return out


def main(argv):
    tool, rest = (argv[0], argv[1:]) if argv else ("", [])
    if os.environ.get("FAKE_HF_CALLS"):
        with open(os.environ["FAKE_HF_CALLS"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps([tool] + rest) + "\n")
    if tool == "contract":
        emit(json.loads(Path(PINNED).read_text(encoding="utf-8")))
        return 0
    o, pos = opts(rest)
    if tool == "doctor":
        chromium = READY != "nochromium"
        caps = {k: {"status": "available", "detail": k} for k in ("node", "hyperframes", "ffmpeg", "ffprobe", "encoder:libx264", "encoder:prores_ks", "disk:tmp")}
        caps["chromium:headless"] = {"status": "available" if chromium else "missing", "detail": "fake" if chromium else "no Chromium found in any searched location"}
        render = {"usable": "yes"} if chromium else {"usable": "no", "missing": ["chromium:headless"]}
        emit({"status": "completed" if chromium else "failed", "ok": chromium, "skill": {"id": "hyperframes-skill", "version": "0.9.1"}, "missing": [] if chromium else ["chromium:headless"],
              "capabilities": caps, "tools": {"doctor": {"usable": "yes"}, "template": {"usable": "yes"}, "scene": {"usable": "yes"}, "render": render}})
        return 0 if chromium else 1
    if MODE.startswith("fail:") and tool == "render":
        kind = MODE.split(":", 1)[1]
        code, exit_code = KINDS.get(kind, ("INTERNAL_ERROR", 1))
        emit({"status": "failed", "exit_code": exit_code, "error": {"kind": kind, "code": code, "message": f"fake {kind}", "retryable": False, "hint": "fake hint"},
              "dry_run": False, "verified": False, "commands": []})
        return exit_code
    if tool == "template":
        values = json.loads(Path(o["values"]).read_text(encoding="utf-8"))
        req = {"scene_version": 1, "id": pos[0], "width": 1920, "height": 1080, "duration": values.get("duration", 8), "fps": 30, "background": "transparent",
               "layers": [{"type": "text", "id": "name", "text": values.get("name", "")}]}
        Path(o["output"]).write_text(json.dumps(req), encoding="utf-8")
        emit(done(o["output"], template=pos[0], values=values, request=req))
        return 0
    if tool == "scene":
        req = json.loads(Path(pos[0]).read_text(encoding="utf-8"))
        d = Path(o["output"])
        d.mkdir(parents=True, exist_ok=False)
        (d / "index.html").write_text("<html></html>", encoding="utf-8")
        (d / "request.json").write_text(json.dumps(req), encoding="utf-8")
        emit(done(str(d), html=str(d / "index.html"), segments=segments_of(req), commands=[{"program": "node", "argv": ["node", "hyperframes", "lint", str(d)], "exit_code": 0}]))
        return 0
    if tool == "render":
        req = json.loads((Path(pos[0]) / "request.json").read_text(encoding="utf-8"))
        out = Path(o["output"])
        if out.exists() and not o.get("overwrite"):
            emit({"status": "failed", "exit_code": 1, "error": {"kind": "input", "code": "INPUT_INVALID", "message": "output exists"}, "verified": False})
            return 1
        out.write_bytes(b"fake prores 4444\n")
        segs = segments_of(req) if o.get("segments") else None
        if segs:
            for s in segs:
                p = out.with_name(f"{out.stem}.{s['name']}{out.suffix}")
                p.write_bytes(b"fake segment\n")
                s.update(output=str(p), verified=True, verification=[], probe={})
            if MODE == "shortseg":
                segs[1]["duration"] = float(segs[1]["duration"]) - 1.0
        doc = done(str(out) if MODE != "elsewhere" else str(out.with_name("other.mov")),
                   expected={"width": 1920, "height": 1080, "duration": req["duration"], "fps": 30, "frames": int(round(float(req["duration"]) * 30)), "codec": o.get("codec")},
                   alpha={"mode": "none" if MODE == "noalpha" else "straight", "channel": MODE != "noalpha"},
                   hyperframes={"version": "0.8.61", "chromium": {"path": "/fake/chrome", "source": "fake"}},
                   ffmpeg_skill={"status": "PASSED", "tool": "ffmpeg-skill/probe"},
                   commands=[{"program": "node", "argv": ["node", "--require", "argv-recorder.cjs", "hyperframes", "render", pos[0]], "exit_code": 0},
                             {"program": "ffmpeg", "argv": ["ffmpeg", "-i", "frames"], "exit_code": 0}])
        if MODE == "unverified":
            doc["verified"] = False
            doc["verification"] = [{"step": "duration", "ok": False, "expected": 4, "actual": 3.9}]
        if segs is not None:
            doc["segments"] = segs
            doc["loop_seam"] = {"measured": True, "frames": segs[1]["frames"], "wrap_diff": 9 if MODE == "seam" else 0, "max_step_diff": 0, "tolerance": 1,
                                "ok": MODE != "seam"}
        emit(doc)
        return 0
    emit({"status": "failed", "exit_code": 2, "error": {"kind": "input", "code": "INPUT_INVALID", "message": f"unknown tool {tool}"}})
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
