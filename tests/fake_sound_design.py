"""Fake sound-design-skill process for the ADR-092 tests: speaks the slice of the real `sound-design` CLI the agent uses -- `contract --json` (the pinned
src/video_agent/tools/sound_design/contract_0.3.0.json), `doctor --json --offline` (status, checks.operations.{bgm, library}.ready with the engine's
install line), `library-pick --json --output=F --kind=bgm [--tags=..] [--mood=..] [--file=..] [--crossfade=S] --duration=D ... -- <library>` and
`generate-bgm --json --output=F --prompt=P [--tags=..] --duration=D [--seed=N] --offline ...` in the result shapes of sound-design-skill main
(sound-design/library-pick@1, sound-design/generate-bgm@1). The WAV written is a self-describing fake media file (tests/pipeline_harness.py) of the
duration asked, audio only, so the fake engine and the fake audio-production MIX read it. Nothing runs an engine.
Controlled via:

  FAKE_SD_READY: library (default: library ready, ACE-Step missing) | all | none
  FAKE_SD_MODE: ok | fail:<CODE> | short (0.5 s short of the duration) | bad_sha | other_output | no_licence
  FAKE_SD_CALLS: a file every invocation's argv is appended to (one JSON list per line). Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

MODE = os.environ.get("FAKE_SD_MODE", "ok")
READY = os.environ.get("FAKE_SD_READY", "library")
CALLS_LOG = os.environ.get("FAKE_SD_CALLS")
PINNED = Path(__file__).resolve().parents[1] / "src" / "video_agent" / "tools" / "sound_design" / "contract_0.3.0.json"


def _emit(doc) -> None:
    sys.stdout.write(json.dumps(doc) + "\n")


def _contract() -> dict:
    return json.loads(PINNED.read_text(encoding="utf-8"))


def _fail(code: str, message: str, details=None) -> int:
    c = _contract()["errors"].get(code, {})
    _emit({"ok": False, "error": {"code": code, "message": message, "retryable": bool(c.get("retryable")), "details": details or {}}})
    return int(c.get("exit_code", 1))


def _opts(argv):
    opts, pos, after = {}, [], False
    for a in argv:
        if after:
            pos.append(a)
        elif a == "--":
            after = True
        elif a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            opts[k] = v
        elif a.startswith("--"):
            opts[a[2:]] = True
        else:
            pos.append(a)
    return opts, pos


def _doctor() -> int:
    lib_ok, bgm_ok = READY in ("library", "all"), READY == "all"
    ops = {"bgm": {"implemented": True, "engine": "acestep", "engine_status": "OK" if bgm_ok else "MISSING", "ready": bgm_ok},
           "sfx": {"implemented": True, "engine": "ezaudio", "engine_status": "MISSING", "ready": False, "confirmation_required": True},
           "library": {"implemented": True, "engine": None, "engine_status": None, "ready": lib_ok}}
    checks = {"operations": ops, "engines": {"acestep": {"status": "OK" if bgm_ok else "MISSING", "install": None if bgm_ok else "git clone https://github.com/ace-step/ACE-Step-1.5 ... (fake)"}},
              "ffmpeg_skill": {"status": "OK" if lib_ok else "MISSING", "install": None if lib_ok else "set SOUND_DESIGN_FFMPEG_SKILL_DIR (fake)"}}
    status = "ok" if lib_ok and bgm_ok else ("degraded" if lib_ok else "fail")
    _emit({"schema": "sound-design/doctor@1", "ok": status == "ok", "status": status, "skill": {"id": "sound-design", "version": _contract()["version"]}, "offline": True,
           "checks": checks, "unavailable_engines": [] if bgm_ok else ["acestep"], "problems": [] if lib_ok else ["ffmpeg-skill missing"], "warnings": []})
    return 0 if status != "fail" else 3


def _write(out: str, duration: float) -> dict:
    meta = {"fake": True, "duration": duration, "video": False, "channels": 2, "sample_rate": 48000, "lufs": -24.0, "name": os.path.basename(out)}
    Path(out).write_bytes(json.dumps(meta).encode())
    return meta


def _result(schema: str, op: str, out: str, duration: float, extra: dict) -> dict:
    measured = duration - (0.5 if MODE == "short" else 0.0)
    _write(out, measured)
    sha = ("0" * 64) if MODE == "bad_sha" else hashlib.sha256(Path(out).read_bytes()).hexdigest()
    path = out + ".other.wav" if MODE == "other_output" else out
    return {"schema": schema, "ok": True, "op": op, "skill": {"id": "sound-design", "version": _contract()["version"]},
            "output": {"path": path, "format": "wav", "codec": "pcm_s16le", "sample_rate": 48000, "channels": 2, "duration": measured, "sha256": sha},
            "provenance": {"skill": {"id": "sound-design"}, "output": {"sha256": sha}}, "warnings": [], **extra}


def _pick(argv) -> int:
    opts, pos = _opts(argv)
    if READY == "none":
        return _fail("FFMPEG_SKILL_UNAVAILABLE", "no ffmpeg-skill (fake)")
    if MODE.startswith("fail:"):
        return _fail(MODE.split(":", 1)[1], "refused (fake)")
    lib = pos[0]
    manifest = json.loads((Path(lib) / "sound-library.json").read_text(encoding="utf-8"))
    files = sorted(manifest.get("files") or {})
    chosen = opts.get("file") or (files[0] if files else "bed.wav")
    lic = {"licence": None if MODE == "no_licence" else "CC0-1.0", "status": "unknown" if MODE == "no_licence" else "declared", "declared_as": "CC0-1.0",
           "verified": False, "usable": MODE != "no_licence", "author": None, "credit": None, "source": None}
    dur = float(opts["duration"])
    doc = _result("sound-design/library-pick@1", "library-pick", opts["output"], dur,
                  {"capability": "audio.library_import", "request": {"kind": opts.get("kind"), "tags": [t for t in str(opts.get("tags") or "").split(",") if t], "duration": dur, "file": opts.get("file")},
                   "selection": {"chosen": chosen, "matched": [t for t in str(opts.get("tags") or "").split(",") if t]}, "source": {"file": chosen, "sha256": "a" * 64},
                   "fit": {"source": 6.0, "target": dur, "mode": "loop", "repeats": 3, "crossfade": float(opts.get("crossfade", 0.5))}, "licence": lic, "engine": None})
    _emit(doc)
    return 0


def _generate(argv) -> int:
    opts, _ = _opts(argv)
    if READY != "all":
        return _fail("ENGINE_UNAVAILABLE", "ACE-Step is not installed (fake)", {"install": "git clone https://github.com/ace-step/ACE-Step-1.5 ... (fake)"})
    if MODE == "fail:INSUFFICIENT_VRAM":   # 0.3.0 (its ADR-016): a model the GPU cannot hold, refused before any work, with the fallback named
        return _fail("INSUFFICIENT_VRAM", "model acestep-v15-xl-turbo needs at least 20 GB of GPU memory; measured 12.0 GB; use the default model "
                     "acestep-v15-turbo (fake)", {"model": "acestep-v15-xl-turbo", "min_vram_gb": 20, "measured_gb": 12.0, "default_model": "acestep-v15-turbo"})
    if MODE.startswith("fail:"):
        return _fail(MODE.split(":", 1)[1], "refused (fake)")
    dur = float(opts["duration"])
    seed = int(opts.get("seed", 1234))
    model = opts.get("model") or "acestep-v15-turbo"   # 0.3.0: request.model names the model that ran (the default when none is named)
    doc = _result("sound-design/generate-bgm@1", "bgm", opts["output"], dur,
                  {"capability": "audio.bgm_generate", "request": {"prompt": opts.get("prompt"), "duration": dur, "seed": seed, "seed_source": "given" if "seed" in opts else "drawn",
                                                                  "model": model},
                   "engine": {"id": "acestep", "version": "1.5", "model": "acestep-v15-turbo", "code_license": "MIT", "weights_license": "MIT", "output_use": "commercial use allowed (vendor claim)",
                              "lm_status": "off"}})
    _emit(doc)
    return 0


def main(argv) -> int:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\n")
    cmd = argv[:1]
    if cmd == ["contract"]:
        _emit(_contract())
        return 0
    if cmd == ["doctor"]:
        return _doctor()
    if cmd == ["library-pick"]:
        return _pick(argv[1:])
    if cmd == ["generate-bgm"]:
        return _generate(argv[1:])
    return _fail("INVALID_ARGUMENT", f"unsupported command {cmd}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
