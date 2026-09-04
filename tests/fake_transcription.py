"""Fake transcription-skill process for adapter protocol tests: speaks the real transport (`skill --json`, `doctor --json`,
`run -` with {"tool","params"} on stdin → {"ok","tool","result"} on stdout) with canned data, and misbehaves on request via
FAKE_TS_MODE = ok | empty | text | two_docs | wrong_schema | wrong_skill | wrong_version | wrong_engine | wrong_asset |
bad_source | no_transcript | invalid_provenance | speaker_set | timeout_error | hang | crash | nonzero | model_unavailable |
engine_unavailable | bad_segments. FAKE_TS_CACHE=hit reports a cache hit. Never runs an ASR engine or ffmpeg. Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE.parent / "src" / "video_agent" / "tools" / "transcription" / "contract_0.2.0.json").read_text(encoding="utf-8"))
MODE = os.environ.get("FAKE_TS_MODE", "ok")
CALLS_LOG = os.environ.get("FAKE_TS_CALLS")
ENGINE = {"id": "faster_whisper", "version": "1.2.1-fake", "execution_mode": "local", "requires_network": False, "deterministic": True, "available": True,
          "unavailable_reason": None, "capabilities": ["local_execution", "local_model", "model_download", "word_timestamps", "language_detection"],
          "supported_languages": ["ja", "en"], "supported_models": ["tiny", "base", "small"], "default_model": "base",
          "models": [{"model": "tiny", "status": "MISSING", "availability": "MODEL_DOWNLOAD_REQUIRED"}, {"model": "base", "status": "AVAILABLE", "availability": "MODEL_AVAILABLE"},
                     {"model": "small", "status": "MISSING", "availability": "MODEL_DOWNLOAD_REQUIRED"}],
          "description": "fake"}


def log(kind: str) -> None:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(kind + "\n")


def _err(code: str, message: str, details: dict = None) -> int:
    print(json.dumps({"ok": False, "error": {"code": code, "message": message, "details": details or {}}}))
    return 2 if code in ("INVALID_INPUT", "FILE_NOT_FOUND") else 1


def _within(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:
        return False


def transcript(params: dict, fp: str) -> dict:
    lang = params.get("language") or "ja"
    model = params.get("model") or "base"
    words = [{"start": 0.5, "end": 0.9, "text": "本日", "confidence": 0.9}, {"start": 0.9, "end": 1.4, "text": "の", "confidence": 0.8}] if params.get("word_timestamps") else None
    segs = [{"id": "seg_0001", "start": 0.5, "end": 3.2, "text": "本日の公園を始めます", "raw_text": "本日の公園を始めます", "confidence": 0.72, "words": words, "speaker_id": None},
            {"id": "seg_0002", "start": 4.0, "end": 8.4, "text": "会場の音教設備についてご説明します", "raw_text": "会場の音教設備についてご説明します", "confidence": 0.68, "words": None, "speaker_id": None}]
    if os.environ.get("FAKE_TS_SEGMENTS"):   # [[start, end, text], ...] → canned segments with a fixed confidence (temporal-structure tests)
        segs = [{"id": f"seg_{i:04d}", "start": float(a), "end": float(b), "text": t, "raw_text": t, "confidence": 0.7, "words": None, "speaker_id": None}
                for i, (a, b, t) in enumerate(json.loads(os.environ["FAKE_TS_SEGMENTS"]), 1)]
    tid = "tr_" + hashlib.sha256((fp + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:12]
    prov_params = {"language": params.get("language"), "word_timestamps": bool(params.get("word_timestamps")), "temperature": params.get("temperature", 0.0),
                   "initial_prompt": params.get("initial_prompt"), "beam_size": params.get("beam_size", 5)}
    return {"schema": "transcription-skill/transcript/0.1", "id": tid, "asset_id": params["asset_id"], "language": lang, "language_source": "requested" if params.get("language") else "detected",
            "language_confidence": None if params.get("language") else 0.93, "duration": 9.606, "segments": segs,
            "source": {"filename": os.path.basename(params["input"]), "fingerprint": "sha256:" + fp, "size_bytes": 64, "media_duration": 9.606, "audio_channels": 1, "sample_rate": 16000,
                       "container": "wav", "has_video": False},
            "engine": ENGINE["id"], "engine_version": ENGINE["version"], "created_at": "2026-09-04T10:00:00Z",
            "provenance": {"engine": ENGINE["id"], "engine_version": ENGINE["version"], "execution_mode": "local", "model": model, "model_version": "fake-model-rev",
                           "parameters": prov_params, "parameters_hash": "p" * 64, "cache_key": "c" * 64, "created_at": "2026-09-04T10:00:00Z", "processing_seconds": 0.01,
                           "skill_version": CONTRACT["version"], "skill": "transcription-skill", "tool": "transcription/transcribe",
                           "language_detection": None if params.get("language") else {"candidate": lang, "probability": 0.93, "min_probability": 0.5},
                           "audio_extraction": {"tool": "ffmpeg", "recipe": {"channels": 1, "sample_rate": 16000, "codec": "pcm_s16le", "container": "wav"}, "engine_seconds": 0.0}},
            "warnings": []}


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["skill"]:
        log("skill")
        c = json.loads(json.dumps(CONTRACT))
        c["engines"] = [dict(ENGINE)]
        if MODE == "engine_unavailable":
            c["engines"][0].update(available=False, version=None, models=[], unavailable_reason="faster-whisper is not installed")
        if MODE == "wrong_version":
            c["version"] = "9.0.0"
        if MODE == "wrong_schema":
            c["schemas"]["transcript"] = "transcription-skill/transcript/0.9"
        if MODE == "wrong_skill":
            c["id"] = "other-skill"
        print(json.dumps(c, ensure_ascii=False))
        return 0
    if argv[:1] == ["doctor"]:
        log("doctor")
        offline = "--offline" in argv
        roots = [argv[i + 1] for i, a in enumerate(argv) if a == "--allowed-input"]
        engine_ok = MODE != "engine_unavailable"
        checks = [{"check": "skill", "status": "AVAILABLE", "detail": f"transcription-skill {CONTRACT['version']}"}, {"check": "ffmpeg", "status": "AVAILABLE", "detail": "fake"},
                  {"check": "ffprobe", "status": "AVAILABLE", "detail": "fake"},
                  {"check": "engine:faster_whisper", "status": "AVAILABLE" if engine_ok else "MISSING", "detail": "fake", "default": True, "execution_mode": "local"},
                  {"check": "model:faster_whisper:base", "status": "MISSING" if MODE == "model_unavailable" else "AVAILABLE", "detail": "fake"},
                  {"check": "input path policy", "status": "AVAILABLE", "detail": "fake", "mode": "allowed_roots" if roots else "unrestricted", "allowed_roots": roots}]
        ok = engine_ok and MODE != "model_unavailable"
        print(json.dumps({"ok": ok, "offline": offline, "checks": checks, "summary": "ready" if ok else "not ready: see MISSING rows"}))
        return 0 if ok else 1
    if argv[:1] == ["run"]:
        log("run")
        try:
            req = json.loads(sys.stdin.read())
        except ValueError:
            return _err("INVALID_INPUT", "stdin is not valid JSON")
        if MODE == "hang":
            time.sleep(30)
        if MODE == "crash":
            print("boom", file=sys.stderr)
            return 8
        if MODE == "empty":
            return 0
        if MODE == "text":
            print("not json at all")
            return 0
        if set(req) - {"tool", "params"} or req.get("tool") not in [t["name"] for t in CONTRACT["tools"]]:
            return _err("INVALID_INPUT", "bad request envelope")
        p = req.get("params") or {}
        for k in p:
            if k.lower() in ("command", "argv", "cmd", "shell", "exec", "args", "script", "binary", "api_key", "token", "secret", "password", "credentials", "env"):
                return _err("INVALID_INPUT", f"key '{k}' is not allowed")
        raw = p.get("input")
        if not isinstance(raw, str) or not raw:
            return _err("INVALID_INPUT", "'input' must be a non-empty path string")
        roots = [os.path.realpath(r) for r in (p.get("allowed_input_roots") or [])]
        if roots:
            if any(part == ".." for part in raw.replace("\\", "/").split("/")):
                return _err("INVALID_INPUT", "input path contains '..'", {"reason": "traversal"})
            resolved = os.path.realpath(os.path.abspath(raw))
            if not any(_within(r, resolved) for r in roots):
                reason = "symlink_escape" if os.path.islink(os.path.abspath(raw)) else "outside_allowed_roots"
                return _err("INVALID_INPUT", "input is outside the allowed roots", {"reason": reason})
        if not os.path.isfile(raw):
            return _err("FILE_NOT_FOUND", "no such input")
        if MODE == "engine_unavailable":
            return _err("ENGINE_UNAVAILABLE", "faster-whisper is not installed", {"engine": "faster_whisper", "reason": "engine_not_installed"})
        if MODE == "model_unavailable" or (p.get("offline") and p.get("model") not in (None, "base")):
            return _err("MODEL_UNAVAILABLE", "model not on this machine and cannot be fetched (offline)", {"engine": "faster_whisper", "model": p.get("model") or "base", "availability": "MODEL_MISSING", "offline": bool(p.get("offline"))})
        if MODE == "nonzero":
            return _err("TRANSCRIPTION_FAILED", "engine crashed", {"exit": 1})
        if MODE == "timeout_error":
            return _err("TRANSCRIPTION_TIMEOUT", "engine exceeded budget.timeout", {"timeout": (p.get("budget") or {}).get("timeout")})
        fp = hashlib.sha256(Path(raw).read_bytes()).hexdigest()
        hit = os.environ.get("FAKE_TS_CACHE") == "hit"
        if p.get("dry_run"):
            res = {"dry_run": True, "input": {"filename": os.path.basename(raw), "fingerprint": "sha256:" + fp, "duration": 9.606}, "engine": {"id": ENGINE["id"], "version": ENGINE["version"], "execution_mode": "local"},
                   "model": {"model": p.get("model") or "base", "availability": "MODEL_AVAILABLE"}, "cache": {"status": "hit" if hit else "miss", "key": "c" * 64}, "would_run": not hit,
                   "path_policy": {"mode": "allowed_roots" if roots else "unrestricted", "allowed_roots": roots}}
            print(json.dumps({"ok": True, "tool": req["tool"], "result": res}, ensure_ascii=False))
            return 0
        tr = transcript(p, fp)
        if MODE == "wrong_engine":
            tr["engine"] = tr["provenance"]["engine"] = "cloud_asr"
        if MODE == "wrong_asset":
            tr["asset_id"] = "asset_other"
        if MODE == "bad_source":
            tr["source"]["fingerprint"] = "md5:abc"
        if MODE == "invalid_provenance":
            tr["provenance"]["skill"] = "ai-provider"
            tr["provenance"]["execution_mode"] = "remote"
        if MODE == "speaker_set":
            tr["segments"][0]["speaker_id"] = "spk_1"
        if MODE == "bad_segments":
            tr["segments"][1]["end"] = 3.0
        if hit:
            tr["asset_id"] = "asset_first_caller"   # like the real Skill: a cache hit returns the stored document unchanged
        res = {"transcript": tr, "cache_hit": hit, "cache_key": "c" * 64, "warnings": []}
        if MODE == "no_transcript":
            res.pop("transcript")
        doc = {"ok": True, "tool": req["tool"], "result": res}
        print(json.dumps(doc, ensure_ascii=False))
        if MODE == "two_docs":
            print(json.dumps(doc, ensure_ascii=False))
        return 0
    print("unknown", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
