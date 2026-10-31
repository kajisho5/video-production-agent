"""Fake voice-enhance-skill process for the ADR-073 tests: speaks the slice of the real `voice-enhance` CLI the agent uses -- `contract --json`
(the pinned src/video_agent/tools/voice_enhance/contract_0.1.0.json), `doctor --json --offline --ffmpeg-skill=DIR` (voice-enhance/doctor@1 with
engines and operations.<op>.ready) and `isolate` / `dereverb` / `bleed-reduce` with `--flag=value` options and the inputs after `--`, in the
result shapes of voice-enhance-skill main (voice-enhance/<op>@1, its provenance block with sha256 of every input and output, its error
document). The media are the self-describing fake files of tests/fake_adapter.py: an output is such a file (isolate: 48 kHz mono, 30 ms
shorter; dereverb: the input's rate and channels; bleed-reduce: one mono file per track named <input name>_a0.wav). Nothing runs an engine.
Controlled via:

  FAKE_VOICE_ENGINES: all (every engine OK, default) | none | a comma list of engines that are MISSING (deepfilternet, nara_wpe, crosstalk) |
    model (deep-filter installed, DeepFilterNet3 model missing: MODEL_NOT_FOUND)
  FAKE_VOICE_RESAMPLE: "1" makes the doctor report ffmpeg-skill's optional `audio --sample-rate` (isolate then takes any rate)
  FAKE_VOICE_MODE: ok | fail:<CODE> (the op refuses with that voice-enhance code) | bad_hash | wrong_path | wrong_rate | bad_json | hang
  FAKE_VOICE_CONTRACT: pinned | drift_version (0.1.9: non-fatal) | drift_lifecycle (isolate STABLE: non-fatal) | drift_fatal (isolate argv changed) |
    drift_error (a pinned error code's exit code changed) | incompatible (version 0.2.0)
  FAKE_VOICE_CALLS: a file every invocation's argv is appended to (one JSON list per line). Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

MODE = os.environ.get("FAKE_VOICE_MODE", "ok")
ENGINES = os.environ.get("FAKE_VOICE_ENGINES", "all")
CONTRACT_MODE = os.environ.get("FAKE_VOICE_CONTRACT", "pinned")
CALLS_LOG = os.environ.get("FAKE_VOICE_CALLS")
PINNED = Path(__file__).resolve().parents[1] / "src" / "video_agent" / "tools" / "voice_enhance" / "contract_0.1.0.json"
OP_ENGINE = {"isolate": "deepfilternet", "dereverb": "nara_wpe", "bleed_reduce": "crosstalk"}


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
    elif CONTRACT_MODE == "drift_lifecycle":
        c["provides"][0]["lifecycle"] = "STABLE"
    elif CONTRACT_MODE == "drift_fatal":
        next(t for t in c["tools"] if t["tool_id"] == "voice-enhance/isolate")["argv"].append("--strength N")
    elif CONTRACT_MODE == "drift_error":
        c["errors"]["ENGINE_FAILED"]["exit_code"] = 9
    elif CONTRACT_MODE == "incompatible":
        c["version"] = "0.2.0"
    return c


def _error(code: str, message: str, details=None) -> int:
    errs = _contract()["errors"]
    _emit({"ok": False, "error": {"code": code, "message": message, "retryable": bool((errs.get(code) or {}).get("retryable")), "details": details or {}}})
    return int((errs.get(code) or {}).get("exit_code", 7))


def _engines() -> dict:
    reg = _contract()["engines"]["registered"]
    missing = set() if ENGINES in ("all", "model") else set(reg) if ENGINES == "none" else {e.strip() for e in ENGINES.split(",")}
    out = {}
    for eid, e in reg.items():
        row = dict(e)
        if eid in missing:
            row.update(status="MISSING", version=None, missing="binary" if eid == "deepfilternet" else "package",
                       detail="deep-filter is not on PATH" if eid == "deepfilternet" else f"{eid} is not installed")
        elif eid == "deepfilternet" and ENGINES == "model":
            row.update(status="MISSING", version="0.5.6", missing="model", detail="VOICE_ENHANCE_DFN_MODEL is not set", install=e.get("install"))
        else:
            row.update(status="OK", version={"deepfilternet": "0.5.6", "nara_wpe": "0.0.11", "crosstalk": "0.1.0"}[eid], install=None, detail="fake")
        out[eid] = row
    return out


def doctor(argv) -> int:
    engines = _engines()
    ops = {op: {"implemented": True, "engine": eng, "engine_status": engines[eng]["status"], "ready": engines[eng]["status"] == "OK"} for op, eng in OP_ENGINE.items()}
    resample = os.environ.get("FAKE_VOICE_RESAMPLE") == "1"
    fs = {"directory": next((a.split("=", 1)[1] for a in argv if a.startswith("--ffmpeg-skill=")), None), "version": "2.22.5-fake", "contract_version": "1.0",
          "problems": [], "warnings": [], "optional_flags": {"audio": {"sample_rate": resample}}, "status": "OK"}
    unavailable = [e for e, r in engines.items() if r["status"] != "OK"]
    _emit({"schema": "voice-enhance/doctor@1", "ok": True, "status": "degraded" if unavailable else "ok", "skill": {"id": "voice-enhance", "version": _contract()["version"]},
           "offline": "--offline" in argv, "checks": {"python": {"status": "OK"}, "ffmpeg_skill": fs, "engines": engines, "operations": ops,
                                                       "helpers": {"remux": {"implemented": True, "ready": True}}},
           "unavailable_engines": unavailable, "problems": [f"engine {e}: MISSING" for e in unavailable], "warnings": []})
    return 0


def _read(path: str) -> dict:
    raw = Path(path).read_bytes()
    try:
        meta = json.loads(raw.decode()) if raw.startswith(b'{"fake"') else {}
    except ValueError:
        meta = {}
    return {"duration": float(meta.get("duration", 16.0)), "channels": int(meta.get("channels", 2)), "sample_rate": int(meta.get("sample_rate", 48000)),
            "lufs": float(meta.get("lufs", -16.0)), "video": bool(meta.get("video", True))}


def _sha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _write(path: str, facts: dict, tag: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Path(path).write_bytes(json.dumps({"fake": True, "duration": facts["duration"], "lufs": facts["lufs"], "video": False, "channels": facts["channels"],
                                       "sample_rate": facts["sample_rate"], "op": tag}).encode())


def _engine_doc(op: str) -> dict:
    eid = OP_ENGINE[op]
    e = _contract()["engines"]["registered"][eid]
    return {"id": eid, "version": _engines()[eid]["version"], "model": "DeepFilterNet3_onnx.tar.gz" if eid == "deepfilternet" else None,
            "model_sha256": "a" * 64 if eid == "deepfilternet" else None, "code_license": e["code_license"], "weights_license": e["weights_license"],
            "attribution": e["attribution"], "execution_mode": "local", "deterministic": True}


def run_op(cmd: str, argv) -> int:
    op = cmd.replace("-", "_")
    if "--" not in argv:
        return _error("INVALID_ARGUMENT", "the fake expects the inputs after --")
    opts, inputs = argv[:argv.index("--")], argv[argv.index("--") + 1:]
    kv = dict(a[2:].split("=", 1) for a in opts if a.startswith("--") and "=" in a)
    if any(a.startswith("-") and "=" not in a and a not in ("--json", "--offline", "--overwrite") for a in opts):
        return _error("INVALID_ARGUMENT", f"unexpected option form in {opts}")
    out = kv.get("output")
    if MODE == "hang":
        time.sleep(30)
    if MODE.startswith("fail:"):
        code = MODE.split(":", 1)[1]
        return _error(code, f"fake {cmd} refused with {code}", {"install": "fake install line"} if code in ("ENGINE_UNAVAILABLE", "MODEL_NOT_FOUND") else {})
    if MODE == "bad_json":
        sys.stdout.write("not json\n")
        return 0
    if _engines()[OP_ENGINE[op]]["status"] != "OK":
        return _error("MODEL_NOT_FOUND" if ENGINES == "model" and op == "isolate" else "ENGINE_UNAVAILABLE", f"engine {OP_ENGINE[op]} is not available")
    for p in inputs:
        if not os.path.isfile(p):
            return _error("INPUT_NOT_FOUND", f"{os.path.basename(p)} does not exist")
    facts_in = [_read(p) for p in inputs]
    params = {k.replace("-", "_"): float(v) for k, v in kv.items() if k not in ("output", "ffmpeg-skill", "timeout")}
    base = {"ok": True, "op": op, "capability": f"voice.{op}", "skill": {"id": "voice-enhance", "version": _contract()["version"]}, "steps": [], "warnings": []}
    if op == "bleed_reduce":
        if not 2 <= len(inputs) <= 8:
            return _error("INVALID_ARGUMENT", "bleed-reduce takes 2..8 tracks")
        rate = facts_in[0]["sample_rate"]
        if any(f["sample_rate"] != rate for f in facts_in):
            return _error("FFMPEG_SKILL_UNAVAILABLE", "tracks at different sample rates", {"needs_flag": "sample_rate"})
        dsts = [os.path.join(out, f"{os.path.splitext(os.path.basename(p))[0]}_a0.wav") for p in inputs]
        outs = []
        for i, (p, d, f) in enumerate(zip(inputs, dsts, facts_in)):
            g = dict(f, channels=1)
            _write(d, g, f"bleed{i}")
            if MODE == "wrong_path":
                d = d + ".x"
            outs.append({"path": d, "format": "wav", "codec": "pcm_f32le", "sample_rate": rate, "channels": 1, "duration": g["duration"], "track": i})
        prov = {"skill": base["skill"], "op": op, "engine": _engine_doc(op), "inputs": [{"sha256": _sha(p)} for p in inputs],
                "outputs": [{"sha256": "0" * 64 if MODE == "bad_hash" else _sha(d)} for d in dsts], "parameters": params, "ffmpeg_skill": {"version": "2.22.5-fake", "contract_version": "1.0"},
                "network": False, "created_at": "2026-10-31T00:00:00Z"}
        _emit({**base, "schema": "voice-enhance/bleed-reduce@1", "output_dir": out, "outputs": outs, "inputs": [{"path": p, "track": i} for i, p in enumerate(inputs)],
               "engine": {k: _engine_doc(op)[k] for k in ("id", "version", "model")}, "parameters": dict(params, tracks=len(inputs), coupling_db=[[None, -8.2], [-8.4, None]][:len(inputs)]),
               "provenance": prov, "warnings": ["fake: nothing was measured"]})
        return 0
    f = facts_in[0]
    if op == "isolate":
        if f["sample_rate"] != 48000 and os.environ.get("FAKE_VOICE_RESAMPLE") != "1":
            return _error("FFMPEG_SKILL_UNAVAILABLE", f"the stream is {f['sample_rate']} Hz; isolate needs 48000 Hz and ffmpeg-skill has no audio --sample-rate", {"needs_flag": "sample_rate"})
        g = dict(f, duration=round(max(0.0, f["duration"] - 0.03), 3), channels=1, sample_rate=16000 if MODE == "wrong_rate" else 48000)
        warnings = ["the stream has 2 channels; folded down to mono"] if f["channels"] > 1 else []
    else:
        g = dict(f)
        warnings = []
    _write(out, g, op)
    o = {"path": out + ".x" if MODE == "wrong_path" else out, "format": "wav", "codec": "pcm_s16le" if op == "isolate" else "pcm_f32le", "sample_rate": g["sample_rate"],
         "channels": g["channels"], "duration": g["duration"]}
    prov = {"skill": base["skill"], "op": op, "engine": _engine_doc(op), "input": {"sha256": _sha(inputs[0])}, "output": {"sha256": "0" * 64 if MODE == "bad_hash" else _sha(out)},
            "parameters": params, "ffmpeg_skill": {"version": "2.22.5-fake", "contract_version": "1.0"}, "network": False, "created_at": "2026-10-31T00:00:00Z"}
    _emit({**base, "schema": f"voice-enhance/{op}@1", "output": o, "input": {"path": inputs[0], "duration": f["duration"], "audio_stream": 0},
           "engine": {k: _engine_doc(op)[k] for k in ("id", "version", "model", "model_sha256")}, "parameters": dict(params, engine_argv=["deep-filter", "<input>"]),
           "provenance": prov, "warnings": warnings})
    return 0


def main() -> int:
    argv = sys.argv[1:]
    _log(argv)
    if argv[:1] in (["contract"], ["skill"]):
        _emit(_contract())
        return 0
    if argv[:1] == ["doctor"]:
        return doctor(argv)
    if argv[:1] and argv[0] in ("isolate", "dereverb", "bleed-reduce") and "--json" in argv:
        return run_op(argv[0], argv[1:])
    print("usage: voice-enhance {contract,doctor,isolate,dereverb,bleed-reduce} ...", file=sys.stderr)
    return 2


def checkout(root) -> str:
    """A fake voice-enhance-skill checkout (<root>/src/voice_enhance_skill/cli.py running this module), for VIDEO_AGENT_VOICE_ENHANCE_DIR."""
    pkg = Path(root) / "src" / "voice_enhance_skill"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "cli.py").write_text(f"import runpy\nrunpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')\n", encoding="utf-8")
    return str(root)


if __name__ == "__main__":
    sys.exit(main())
