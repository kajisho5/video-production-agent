"""Fake transcription-skill process for adapter protocol tests: speaks the real transport (`skill --json`, `doctor --json`,
`run -` with {"tool","params"} on stdin → {"ok","tool","result"} on stdout) with canned data, and misbehaves on request via
FAKE_TS_MODE = ok | empty | text | two_docs | wrong_schema | wrong_skill | wrong_version | wrong_engine | wrong_asset |
bad_source | no_transcript | invalid_provenance | speaker_set | timeout_error | hang | crash | nonzero | model_unavailable |
engine_unavailable | bad_segments | speaker_unknown | speaker_mismatch. FAKE_TS_CACHE=hit reports a cache hit. Model distil-large-v3.5 is
English only, as in transcription-skill 0.10.0: another language or none is refused (language_unsupported / language_detection_unsupported). FAKE_TS_RENDERED (JSON segments / words)
is what a rendered fake file (which carries no speech) is heard to say (ADR-110). With `speaker_turns` (a
diarization-skill/turns@1 document, ADR-096) the segments get speaker_id by the Skill's own ASSIGN rule (longest overlap with the exclusive
turns; none or a tie: null) and the transcript a `speaker_assignment` record, as transcription-skill 0.8.0 merges after recognition.
Never runs an ASR engine or ffmpeg. Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE.parent / "src" / "video_agent" / "tools" / "transcription" / "contract_0.11.0.json").read_text(encoding="utf-8"))
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


# transcription-skill 0.10.0: models limited to some languages (faster_whisper's English-only distil-large-v3.5)
MODEL_LANGUAGES = {"distil-large-v3.5": ("en",)}


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
    try:   # a self-describing fake media file may carry its own segments [[start, end, text], ...] (ADR-053 takes: one transcript per take)
        meta = json.loads(Path(params["input"]).read_bytes().decode("utf-8"))
    except (OSError, ValueError, KeyError, UnicodeDecodeError):
        meta = None
    side = Path(str(params["input"]) + ".segments.json")   # a real media file may carry its segments in a sidecar (ADR-054 real take assembly)
    if meta is None and side.is_file():
        meta = {"fake": True, "segments": json.loads(side.read_text(encoding="utf-8"))}
    words_side = Path(str(params["input"]) + ".words.json")   # word timings [[start, end, text], ...] for a real media file (ADR-072 filler removal)
    if isinstance(meta, dict) and words_side.is_file():
        meta = dict(meta, words=json.loads(words_side.read_text(encoding="utf-8")))
    if isinstance(meta, dict) and meta.get("fake") and not isinstance(meta.get("segments"), list) and os.environ.get("FAKE_TS_RENDERED"):
        # ADR-110: what is "heard" in a rendered file (the fake engine writes no speech): {"segments": [[start, end, text]], "words": [[start, end, text(, confidence)]]}
        meta = dict(meta, **json.loads(os.environ["FAKE_TS_RENDERED"]))
    if isinstance(meta, dict) and meta.get("fake") and isinstance(meta.get("segments"), list):
        # ADR-072: with word timestamps requested, a segment carries the words of `words` [[start, end, text], ...] that fall inside it (none → null)
        timed = meta.get("words") if params.get("word_timestamps") and isinstance(meta.get("words"), list) else None
        segs = [{"id": f"seg_{i:04d}", "start": float(a), "end": float(b), "text": t, "raw_text": t, "confidence": 0.7,
                 "words": ([{"start": float(w[0]), "end": float(w[1]), "text": w[2], "confidence": float(w[3]) if len(w) > 3 else 0.9} for w in timed
                            if float(a) <= float(w[0]) and float(w[1]) <= float(b)] or None)
                 if timed is not None else None, "speaker_id": None}
                for i, (a, b, t) in enumerate(meta["segments"], 1)]
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
        if p.get("model") in MODEL_LANGUAGES:   # 0.10.0 (its ADR-040): a model limited to some languages, refused before any audio work
            langs, lang = MODEL_LANGUAGES[p["model"]], p.get("language")
            if lang is None:
                return _err("INVALID_INPUT", f"model {p['model']} of engine faster_whisper recognizes only {', '.join(langs)} and does not detect the language; pass 'language' explicitly",
                            {"engine": "faster_whisper", "model": p["model"], "reason": "language_detection_unsupported", "supported": list(langs)})
            if lang not in langs:
                return _err("INVALID_INPUT", f"model {p['model']} of engine faster_whisper does not recognize language {lang!r}",
                            {"engine": "faster_whisper", "model": p["model"], "reason": "language_unsupported", "supported": list(langs)})
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
        turns = p.get("speaker_turns")
        if turns is not None:
            if not isinstance(turns, dict) or turns.get("schema") != "diarization-skill/turns@1":
                return _err("INVALID_INPUT", "speaker turns schema must be diarization-skill/turns@1", {"reason": "TURNS_SCHEMA"})
            if turns["source"]["fingerprint"] != tr["source"]["fingerprint"]:
                return _err("INVALID_INPUT", "transcript and speaker turns were made from different files", {"reason": "SOURCE_MISMATCH"})
            per = {}
            for seg in tr["segments"]:
                ov = {}
                for t in turns["turns"]:
                    o = min(seg["end"], t["end"]) - max(seg["start"], t["start"])
                    if o > 0:
                        ov[t["speaker"]] = ov.get(t["speaker"], 0.0) + o
                best = sorted(ov.items(), key=lambda kv: -kv[1])
                seg["speaker_id"] = best[0][0] if best and (len(best) == 1 or best[0][1] > best[1][1] + 1e-9) else None
                for w in seg.get("words") or []:
                    w["speaker_id"] = seg["speaker_id"]
                if seg["speaker_id"]:
                    per[seg["speaker_id"]] = per.get(seg["speaker_id"], 0) + 1
            if MODE == "speaker_unknown":
                tr["segments"][0]["speaker_id"] = "SPEAKER_42"
            tr["speaker_assignment"] = {"skill": "transcription-skill", "skill_version": CONTRACT["version"], "tool": "transcription/speakers", "created_at": "2026-10-31T00:00:00Z",
                                        "method": "max_overlap_with_exclusive_turns",
                                        "rule": "per word (else per segment): speaker with the longest overlap with the exclusive turns; zero overlap or a tie -> null. "
                                                "Segment with words: speaker with the largest assigned word duration; tie -> null.", "speaker_ids": "anonymous (SPEAKER_nn) from diarization-skill; they name nobody",
                                        "turns_schema": "diarization-skill/turns@1", "source_fingerprint": turns["source"]["fingerprint"] if MODE != "speaker_mismatch" else "sha256:" + "0" * 64,
                                        "source_match": MODE != "speaker_mismatch", "speakers": list(turns["speakers"]), "segments_per_speaker": per,
                                        "counts": {"segments": len(tr["segments"]), "assigned": sum(per.values()), "unassigned": len(tr["segments"]) - sum(per.values())},
                                        "overlap_available": False, "overlap_segment_ids": []}
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
