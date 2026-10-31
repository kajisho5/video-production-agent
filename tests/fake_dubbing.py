"""Fake dubbing-skill process for the ADR-085 tests: speaks the slice of the real `dubbing` CLI the agent uses -- `contract --json` (the pinned
src/video_agent/tools/dubbing/contract_0.1.0.json), `doctor --json --offline --ffmpeg-skill=DIR` (dubbing/doctor@1 with engines and
operations.<op>.ready) and `dub` with `--flag=value` options and the programme + cues after `--`, in the result shape of dubbing-skill main
(dubbing/dub@1: output, fit, mix, clips, kept_original, obligations, provenance with the output sha256 and the consent summary; its error document).
The media are the self-describing fake files of tests/fake_adapter.py: the output is such a file (the input's facts, louder dialogue). Nothing
runs an engine. The consent check stands in for dubbing-skill ADR-009: a record is accepted only when it is a JSON object with "fake_valid": true
and "reference_sha256" equal to the reference's sha256; anything else is CONSENT_REQUIRED (record_invalid / reference_mismatch).
Controlled via:

  FAKE_DUB_ENGINES: all (every engine OK, default) | none | a comma list of engines that are MISSING (chatterbox, style_bert_vits2, kokoro, voicevox)
  FAKE_DUB_MODE: ok | fail:<CODE> (the op refuses with that dubbing code) | bad_hash | wrong_path | bad_json | consent_retryable (CONSENT_REQUIRED
    claiming retryable: true) | no_obligations
  FAKE_DUB_CONTRACT: pinned | drift_version (0.1.9: non-fatal) | drift_fatal (dub argv changed) | bypass (policy.consent_record.bypass set) |
    incompatible (version 0.2.0)
  FAKE_DUB_CALLS: a file every invocation's argv is appended to (one JSON list per line). Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

MODE = os.environ.get("FAKE_DUB_MODE", "ok")
ENGINES = os.environ.get("FAKE_DUB_ENGINES", "all")
CONTRACT_MODE = os.environ.get("FAKE_DUB_CONTRACT", "pinned")
CALLS_LOG = os.environ.get("FAKE_DUB_CALLS")
PINNED = Path(__file__).resolve().parents[1] / "src" / "video_agent" / "tools" / "dubbing" / "contract_0.1.0.json"
CREDITS = {"style_bert_vits2": "JVNV corpus voice jvnv-F1-jp (CC BY-SA 4.0)", "chatterbox": None}
CLONES = {"chatterbox": True, "style_bert_vits2": True, "kokoro": False, "voicevox": False}


def _log(argv) -> None:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\n")


def _emit(doc) -> None:
    sys.stdout.write(json.dumps(doc) + "\n")


def _contract() -> dict:
    c = json.loads(PINNED.read_text(encoding="utf-8"))
    if CONTRACT_MODE == "drift_version":
        c["version"] = "0.1.9"
    elif CONTRACT_MODE == "drift_fatal":
        next(t for t in c["tools"] if t["tool_id"] == "dubbing/dub")["argv"].append("[--strength N]")
    elif CONTRACT_MODE == "bypass":
        c["policy"]["consent_record"]["bypass"] = "--trust-me"
    elif CONTRACT_MODE == "incompatible":
        c["version"] = "0.2.0"
    return c


def _error(code: str, message: str, details=None, retryable=None) -> int:
    errs = _contract()["errors"]
    r = bool((errs.get(code) or {}).get("retryable")) if retryable is None else retryable
    _emit({"ok": False, "skill": {"id": "dubbing", "version": _contract()["version"]}, "error": {"code": code, "message": message, "retryable": r, "details": details or {}}})
    return int((errs.get(code) or {}).get("exit_code", 7))


def _missing() -> set:
    return set() if ENGINES == "all" else set(CLONES) if ENGINES == "none" else {e.strip() for e in ENGINES.split(",")}


def doctor(argv) -> int:
    miss = _missing()
    engines = {e: {"id": e, "status": "MISSING" if e in miss else "OK", "version": None if e in miss else "0.1.7", "missing": "package" if e in miss else None,
                   "detail": f"{e} is not installed" if e in miss else "fake", "install": f"install {e} (fake install line)", "voice_cloning": CLONES[e],
                   "credit": CREDITS.get(e)} for e in CLONES}
    ops = {"dub": {"implemented": True, "engine": "chatterbox", "engine_status": engines["chatterbox"]["status"], "alternatives": ["style_bert_vits2"],
                   "ready": engines["chatterbox"]["status"] == "OK"}}
    fs = {"directory": next((a.split("=", 1)[1] for a in argv if a.startswith("--ffmpeg-skill=")), None), "version": "2.22.5-fake", "status": "OK"}
    unavailable = [e for e, r in engines.items() if r["status"] != "OK"]
    _emit({"schema": "dubbing/doctor@1", "ok": True, "status": "degraded" if unavailable else "ok", "skill": {"id": "dubbing", "version": _contract()["version"]},
           "offline": "--offline" in argv, "checks": {"python": {"status": "OK"}, "ffmpeg_skill": fs, "engines": engines, "operations": ops},
           "unavailable_engines": unavailable, "problems": [f"engine {e}: MISSING" for e in unavailable], "warnings": []})
    return 0


def _read(path: str) -> dict:
    raw = Path(path).read_bytes()
    try:
        return json.loads(raw.decode()) if raw.startswith(b'{"fake"') else {}
    except ValueError:
        return {}


def _sha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_dub(argv) -> int:
    if "--" not in argv:
        return _error("INVALID_ARGUMENT", "the fake expects the programme and the cues after --")
    opts, pos = argv[:argv.index("--")], argv[argv.index("--") + 1:]
    if len(pos) != 2:
        return _error("INVALID_ARGUMENT", f"dub takes <programme> <cues>, got {pos}")
    if any(a.startswith("-") and "=" not in a and a not in ("--json", "--offline", "--overwrite") for a in opts):
        return _error("INVALID_ARGUMENT", f"unexpected option form in {opts}")
    kv = dict(a[2:].split("=", 1) for a in opts if a.startswith("--") and "=" in a)
    src, cues = pos
    out = kv.get("output")
    if MODE.startswith("fail:"):
        code = MODE.split(":", 1)[1]
        return _error(code, f"fake dub refused with {code}", {"install": "fake install line"} if code in ("ENGINE_UNAVAILABLE", "MODEL_NOT_FOUND") else {})
    if MODE == "bad_json":
        sys.stdout.write("not json\n")
        return 0
    for p in (src, cues):
        if not os.path.isfile(p):
            return _error("INPUT_NOT_FOUND", f"{os.path.basename(p)} does not exist")
    consent = None
    if "reference" in kv or "consent" in kv:   # dubbing-skill ADR-009: the record first, then the clip's hash; before any engine work
        if "consent" not in kv:
            return _error("CONSENT_REQUIRED", "a reference voice needs a consent record", {"reason": "no_record", "problems": ["no --consent"], "bypass": None})
        try:
            rec = json.loads(Path(kv["consent"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rec = None
        if not isinstance(rec, dict) or rec.get("fake_valid") is not True:
            return _error("CONSENT_REQUIRED", "the consent record does not check out", {"reason": "record_invalid", "problems": ["evidence[0]: sha256 mismatch"], "bypass": None},
                          retryable=True if MODE == "consent_retryable" else None)
        if rec.get("reference_sha256") != _sha(kv["reference"]):
            return _error("CONSENT_REQUIRED", "the reference is not the clip the consent covers", {"reason": "reference_mismatch", "problems": ["reference.sha256"], "bypass": None})
        consent = {"record_sha256": _sha(kv["consent"]), "voice_owner": rec.get("voice_owner"), "verified_for": {"op": "dub", "engine": kv.get("engine", "chatterbox")}}
    engine = kv.get("engine", "chatterbox")
    if engine in _missing():
        return _error("ENGINE_UNAVAILABLE", f"engine {engine} is not available")
    facts = _read(src)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Path(out).write_bytes(json.dumps({**{k: v for k, v in facts.items() if k != "name"}, "fake": True, "lufs": -18.0, "op": "dub", "name": os.path.basename(out)}).encode())
    dur = float(facts.get("duration", 16.0))
    credit = [CREDITS[engine]] if CREDITS.get(engine) else []
    watermark = {"name": "Resemble AI Perth", "clips": 2, "detected": 2} if engine == "chatterbox" else None
    eng = {"id": engine, "version": "0.1.7", "model": "fake", "code_license": "MIT", "weights_license": "MIT", "credit": CREDITS.get(engine),
           "credit_required": bool(CREDITS.get(engine)), "watermark": "Resemble AI Perth" if engine == "chatterbox" else None}
    prov = {"skill": {"id": "dubbing", "version": _contract()["version"]}, "op": "dub", "engine": eng, "voice": kv.get("voice"),
            "translation": json.loads(kv["translation-note"]) if kv.get("translation-note") else {"provided_by": "caller", "engine": None},
            "watermark": watermark, "credits": [{"credit": c} for c in credit], "consent": consent,
            "output": {"sha256": "0" * 64 if MODE == "bad_hash" else _sha(out)}, "parameters": {"lang": kv.get("lang")}, "ffmpeg_skill": {"version": "2.22.5-fake"},
            "network": False, "created_at": "2026-10-31T00:00:00Z"}
    doc = {"schema": "dubbing/dub@1", "ok": True, "op": "dub", "capability": "speech.dub", "skill": {"id": "dubbing", "version": _contract()["version"]},
           "output": {"path": out + ".x" if MODE == "wrong_path" else out, "kind": "video", "duration": dur, "audio_duration": dur, "video_duration": dur, "av_delta": 0.0,
                      "expected_duration": dur, "tolerance": 0.16, "codec": "aac", "sample_rate": 48000, "channels": 2},
           "input": {"path": src, "cues": cues, "lines": 2, "lang": kv.get("lang"), "text_sha256": "b" * 64},
           "engine": eng, "fit": {"summary": {"fit": 2, "unfit": 0}, "unfit": [], "render": {"tempo": 1.0}}, "mix": {"duck_db": float(kv.get("duck", 12))},
           "clips": [{"index": 0, "placed": True, "status": "fit"}, {"index": 1, "placed": True, "status": "fit"}], "kept_original": [],
           "obligations": {"credit": credit, "credit_required": bool(credit), "credits": [{"credit": c} for c in credit], "watermark": watermark},
           "provenance": prov, "warnings": ["fake: nothing was spoken"]}
    if MODE == "no_obligations":
        doc.pop("obligations")
    _emit(doc)
    return 0


def main() -> int:
    argv = sys.argv[1:]
    _log(argv)
    if argv[:1] == ["contract"]:
        _emit(_contract())
        return 0
    if argv[:1] == ["doctor"]:
        return doctor(argv)
    if argv[:1] == ["dub"] and "--json" in argv:
        return run_dub(argv[1:])
    print("usage: dubbing {contract,doctor,dub} ...", file=sys.stderr)
    return 2


def checkout(root) -> str:
    """A fake dubbing-skill checkout (<root>/src/dubbing_skill/cli.py running this module), for VIDEO_AGENT_DUBBING_DIR."""
    pkg = Path(root) / "src" / "dubbing_skill"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "cli.py").write_text(f"import runpy\nrunpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')\n", encoding="utf-8")
    return str(root)


if __name__ == "__main__":
    sys.exit(main())
