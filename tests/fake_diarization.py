"""Fake diarization-skill process for the ADR-083 tests: speaks the slice of the real `diarization` CLI the agent uses -- `contract --json`
(the pinned src/video_agent/tools/diarization/contract_0.1.0.json), `doctor --json` (ok / ready / checks / problems, exit 3 while not ok)
and `diarize --json --overwrite --output=F [--num-speakers=N ...] -- <media>` in the result shape of diarization-skill main
(diarization-skill/turns@1 with the sha256 of the input file, printed with ok / turns_file and written to --output). The media are the
self-describing fake files of tests/pipeline_harness.py: the turns are their `turns` fact ([[start, end, "SPEAKER_00"], ...]) or, without
one, two voices splitting the file in half. Nothing runs an engine.
Controlled via:

  FAKE_DIA_READY: all (default) | none (pyannote.audio, torch and the pipeline missing) | pipeline (only the pipeline directory missing)
  FAKE_DIA_MODE: ok | fail:<CODE> | bad_fingerprint | overlap | wrong_params | not_written | bad_json
  FAKE_DIA_CONTRACT: pinned | drift_version (0.1.9: non-fatal) | drift_fatal (diarize argv changed) | incompatible (0.2.0)
  FAKE_DIA_CALLS: a file every invocation's argv is appended to (one JSON list per line). Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

MODE = os.environ.get("FAKE_DIA_MODE", "ok")
READY = os.environ.get("FAKE_DIA_READY", "all")
CONTRACT_MODE = os.environ.get("FAKE_DIA_CONTRACT", "pinned")
CALLS_LOG = os.environ.get("FAKE_DIA_CALLS")
PINNED = Path(__file__).resolve().parents[1] / "src" / "video_agent" / "tools" / "diarization" / "contract_0.1.0.json"


def _emit(doc) -> None:
    sys.stdout.write(json.dumps(doc) + "\n")


def _contract() -> dict:
    c = json.loads(PINNED.read_text(encoding="utf-8"))
    if CONTRACT_MODE == "drift_version":
        c["version"] = "0.1.9"
    elif CONTRACT_MODE == "drift_fatal":
        next(t for t in c["tools"] if t["tool_id"] == "diarization/diarize")["argv"].append("--strict")
    elif CONTRACT_MODE == "incompatible":
        c["version"] = "0.2.0"
    return c


def _fail(code: str, message: str) -> int:
    _emit({"ok": False, "error": {"code": code, "message": message, "detail": None}})
    return int(_contract()["errors"].get(code, {}).get("exit_code", 1))


def _doctor() -> int:
    eng_ok = READY in ("all", "pipeline")
    pipe_ok = READY == "all"
    checks = {"python": {"status": "OK"}, "ffmpeg": {"status": "OK", "install": None},
              "pyannote_audio": {"status": "OK" if eng_ok else "MISSING", "install": None if eng_ok else "pip install 'diarization-skill[pyannote]'"},
              "torch": {"status": "OK" if eng_ok else "MISSING", "install": None if eng_ok else "installed with pyannote.audio"},
              "pipeline": {"status": "OK" if pipe_ok else "MISSING", "install": None if pipe_ok else ["1. accept the conditions", "3. export DIARIZATION_PIPELINE_DIR=..."]}}
    problems = [f"{k}: MISSING" for k, v in checks.items() if v["status"] != "OK"]
    ok = not problems
    _emit({"ok": ok, "skill": "diarization-skill", "version": _contract()["version"], "ready": {"diarize": ok, "assign": True, "check": True}, "checks": checks,
           "problems": problems, "network": "never used at run time"})
    return 0 if ok else 3


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


def _diarize(argv) -> int:
    opts, pos = _opts(argv)
    if READY != "all":
        return _fail("PIPELINE_NOT_FOUND" if READY == "pipeline" else "ENGINE_UNAVAILABLE", "not ready (fake)")
    if MODE.startswith("fail:"):
        return _fail(MODE.split(":", 1)[1], "refused (fake)")
    if MODE == "bad_json":
        sys.stdout.write("not json\n")
        return 0
    media = pos[0]
    raw = Path(media).read_bytes()
    try:
        facts = json.loads(raw.decode())
    except ValueError:
        facts = {}
    dur = float(facts.get("duration") or 10.0)
    turns = [{"start": float(s), "end": float(e), "speaker": str(sp)} for s, e, sp in facts.get("turns") or [[0.0, dur / 2, "SPEAKER_00"], [dur / 2, dur, "SPEAKER_01"]]]
    if MODE == "overlap" and len(turns) > 1:
        turns[1]["start"] = turns[0]["end"] - 1.0
    speakers = []
    for t in turns:
        if t["speaker"] not in speakers:
            speakers.append(t["speaker"])
    n = lambda k: int(opts[k]) if k in opts else None  # noqa: E731
    params = {"num_speakers": n("num-speakers"), "min_speakers": n("min-speakers"), "max_speakers": n("max-speakers")}
    if MODE == "wrong_params":
        params["num_speakers"] = 7
    fp = "sha256:" + (("0" * 64) if MODE == "bad_fingerprint" else hashlib.sha256(raw).hexdigest())
    doc = {"schema": "diarization-skill/turns@1",
           "source": {"filename": os.path.basename(media), "fingerprint": fp, "size_bytes": len(raw), "media_duration": dur, "has_video": bool(facts.get("video", True)),
                      "audio_stream": int(opts.get("audio-stream", 0))},
           "engine": {"id": "pyannote", "version": "4.0.1", "model": "pyannote/speaker-diarization-community-1", "revision": "abc123", "license": "CC-BY-4.0",
                      "attribution": "pyannote/speaker-diarization-community-1 by pyannoteAI, licensed CC-BY-4.0; run with pyannote.audio (MIT)", "execution_mode": "local", "device": "cpu"},
           "parameters": params, "speakers": speakers, "speaker_count": len(speakers), "speech_seconds": round(sum(t["end"] - t["start"] for t in turns), 3), "turns": turns,
           "overlap": {"available": True, "turns": turns, "regions": [], "total_seconds": 0.0},
           "audio_extraction": {"tool": "ffmpeg", "recipe": {"channels": 1, "sample_rate": 16000, "codec": "pcm_s16le", "container": "wav"}, "audio_stream_index": 0,
                                "stream": "0:a:0", "engine_seconds": 0.1},
           "skill": "diarization-skill", "skill_version": _contract()["version"], "tool": "diarization/diarize", "created_at": "2026-10-31T00:00:00Z",
           "processing_seconds": 0.1, "warnings": []}
    out = opts.get("output")
    if out and MODE != "not_written":
        Path(out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    _emit(dict(doc, ok=True, turns_file=out))
    return 0


def main(argv) -> int:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\n")
    if argv[:1] == ["contract"]:
        _emit(_contract())
        return 0
    if argv[:1] == ["doctor"]:
        return _doctor()
    if argv[:1] == ["diarize"]:
        return _diarize(argv[1:])
    return _fail("INVALID_ARGUMENT", f"unsupported command {argv[:1]}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
