"""Fake audio-production-skill process for adapter boundary tests: speaks the real transport (`skill --json`, `doctor --json`,
`plan|run - --json --workspace D --allowed-input R… [--ffmpeg-skill X] [--timeout S]` with an audio request on stdin → one
response document on stdout) with canned data, enforces the request shape and the path policy like the real Skill, writes a
small self-describing output file (the format tests/fake_adapter.py probes), and misbehaves on request via FAKE_AP_MODE:

  ok | reused | tool_error | tool_error_final | output_missing | validation_error | cancelled | timeout | hang | malformed | empty |
  two_docs | text | nonzero_ok | wrong_schema | wrong_skill | wrong_version | bad_contract | contract_fail | contract_drift |
  hash_mismatch | unknown_code | internal_error | doctor_fail | doctor_degraded | no_provenance

FAKE_AP_CALLS: a file every invocation (subcommand + argv) is appended to (security tests read it). Never runs ffmpeg or
ffmpeg-skill, never imports audio_production. Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE.parent / "src" / "video_agent" / "tools" / "audio_production" / "contract_0.1.0.json").read_text(encoding="utf-8"))
MODE = os.environ.get("FAKE_AP_MODE", "ok")
CALLS_LOG = os.environ.get("FAKE_AP_CALLS")
EXIT = CONTRACT["errors"]["exit_codes"]
RETRY = CONTRACT["errors"]["retryable"]
OPS = {o["type"]: o for o in CONTRACT["operations"]}
UNSUPPORTED = {u["type"] for u in CONTRACT["unsupported_operations"]}
FORBIDDEN = set(CONTRACT["request"]["forbidden_fields"])
RESPONSE = "audio-production/response@1"


def log(kind: str, argv) -> None:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"cmd": kind, "argv": list(argv), "cwd": os.getcwd(), "env_video": {k: v for k, v in os.environ.items() if k.startswith("VIDEO_")}}) + "\n")


def emit(doc) -> None:
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def err(code: str, message: str, details=None, retryable=None, dry_run=False) -> int:
    emit({"schema": RESPONSE, "skill": {"id": "audio-production", "version": CONTRACT["version"]}, "ok": False, "status": "cancelled" if code == "CANCELLED" else "error",
          "dry_run": dry_run, "error": {"code": code, "message": message, "retryable": RETRY[code] if retryable is None else retryable, "details": details or {}}, "warnings": []})
    return EXIT[code]


def within(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(root), os.path.normcase(path)]) == os.path.normcase(root)
    except ValueError:
        return False


def flag_values(argv, name):
    return [argv[i + 1] for i, a in enumerate(argv) if a == name and i + 1 < len(argv)]


def scan(obj, where):
    """Forbidden field names anywhere in the document (the real Skill rejects them by name)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN:
                return f"{where}.{k}"
            hit = scan(v, f"{where}.{k}")
            if hit:
                return hit
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hit = scan(v, f"{where}[{i}]")
            if hit:
                return hit
    return None


def in_dur(path: str) -> float:
    raw = Path(path).read_bytes()
    if raw.startswith(b'{"fake"'):
        try:
            return float(json.loads(raw.decode()).get("duration") or 16.0)
        except (ValueError, TypeError):
            return 16.0
    return 16.0


def in_channels(path: str) -> int:
    raw = Path(path).read_bytes()
    if raw.startswith(b'{"fake"'):
        try:
            return int(json.loads(raw.decode()).get("channels") or 2)
        except (ValueError, TypeError):
            return 2
    return 2


def main() -> int:
    argv = sys.argv[1:]
    cmd = argv[0] if argv else ""
    log(cmd, argv)
    if cmd in ("contract", "skill"):
        c = json.loads(json.dumps(CONTRACT))
        if MODE == "wrong_schema":
            c["schema"] = "audio-production/contract@2"
        if MODE == "wrong_skill":
            c["skill_id"] = "other-skill"; c["id"] = "other-skill"
        if MODE == "wrong_version":
            c["version"] = "9.0.0"; c["tools"][0]["version"] = "9.0.0"
        if MODE == "bad_contract":
            c["execution"]["shell"] = True
        if MODE == "contract_fail":
            sys.stderr.write("boom\n")
            return 1
        if MODE == "contract_drift":   # a compatible newer contract: one more operation (the adapter must notice, never silently keep 0.1.0 expectations)
            c["operations"].append(dict(c["operations"][0], type="RESAMPLE", parameters={"sample_rate": {"type": "integer", "required": True}}))
            c["unsupported_operations"] = [u for u in c["unsupported_operations"] if u["type"] != "RESAMPLE"]
            c["tools"][0]["operations"] = sorted(c["tools"][0]["operations"] + ["RESAMPLE"])
        emit(c)
        return 0
    if cmd == "doctor":
        status = {"doctor_fail": "fail", "doctor_degraded": "degraded"}.get(MODE, "ok")
        ok = status != "fail"
        ops = {}
        for t in OPS:
            st = "supported" if ok else "unsupported"
            if MODE == "doctor_degraded" and t == "NOISE_REDUCTION":
                st = "unsupported"
            if t in ("GAIN", "FADE_IN", "FADE_OUT", "MONO", "STEREO", "DOWNMIX", "MIX") and ok:
                st = "unknown"   # like the real Skill: ffmpeg-skill's doctor does not probe the core filters
            ops[t] = {"status": st, "tool": OPS[t]["tool"], "required_capabilities": OPS[t]["required_capabilities"], "missing": [] if st != "unsupported" else ["ffmpeg-skill"], "unknown": []}
        emit({"schema": "audio-production/doctor@1", "skill": {"id": "audio-production", "version": CONTRACT["version"]}, "status": status,
              "checks": {"python": {"status": "ok"}, "ffmpeg_skill": {"status": "ok" if ok else "missing", "directory": (flag_values(argv, "--ffmpeg-skill") or [None])[0], "version": "0.9.1-fake"},
                         "ffmpeg": {"status": "ok" if ok else "missing", "version": "fake"}, "ffprobe": {"status": "ok" if ok else "missing", "version": "fake"},
                         "capabilities": {}, "operations": ops, "path_policy": {"status": "ok", "workspace": (flag_values(argv, "--workspace") or [None])[0], "allowed_input_roots": flag_values(argv, "--allowed-input")}},
              "unavailable_operations": [t for t, o in ops.items() if o["status"] == "unsupported"], "problems": [] if ok else ["ffmpeg-skill: not found"], "warnings": [], "secrets_shown": False})
        return 0 if ok else 1
    if cmd not in ("run", "plan", "validate"):
        sys.stderr.write("unknown command\n")
        return 2
    dry_run = cmd == "plan" or "--dry-run" in argv
    if argv[1:2] != ["-"] or "--json" not in argv:
        return err("INVALID_REQUEST", "request must come from stdin with --json")
    ws_list = flag_values(argv, "--workspace")
    if not ws_list:
        return err("INVALID_REQUEST", "--workspace is required")
    ws = os.path.realpath(ws_list[0])
    if not os.path.isdir(ws):
        return err("PATH_NOT_ALLOWED", "workspace is not a directory", {"reason": "workspace_missing"})
    roots = [os.path.realpath(r) for r in flag_values(argv, "--allowed-input")] or None
    try:
        req = json.loads(sys.stdin.read())
    except ValueError:
        return err("INVALID_REQUEST", "request is not valid JSON")
    if MODE == "hang":
        time.sleep(60)
    if MODE == "malformed":
        sys.stdout.write("{not json\n")
        return 0
    if MODE == "empty":
        return 0
    if MODE == "text":
        sys.stdout.write("done, no JSON here\n")
        return 1
    if MODE == "internal_error":
        return err("INTERNAL_ERROR", "TypeError: bug")
    if MODE == "unknown_code":
        emit({"ok": False, "error": {"code": "WEIRD", "message": "?", "retryable": True, "details": {}}})
        return 1
    # ---- request validation (shape, forbidden fields, ids, references, parameters, formats), like the real Skill
    if not isinstance(req, dict) or req.get("schema") != CONTRACT["request"]["schema"] or set(req) - {"schema", "project", "options"}:
        return err("INVALID_REQUEST", "unsupported request shape / schema", {"field": "schema"})
    hit = scan(req, "request")
    if hit:
        return err("INVALID_REQUEST", f"forbidden field {hit}", {"field": hit})
    proj = req.get("project") or {}
    if set(proj) - {"project_id", "sources", "tracks", "operations", "outputs"}:
        return err("INVALID_REQUEST", "unknown project field")
    sources = {}
    for s in proj.get("sources") or []:
        if set(s) != {"source_id", "path"}:
            return err("INVALID_REQUEST", "source fields")
        p = s["path"]
        if not os.path.isabs(p):
            p = os.path.join(ws, p)
        resolved = os.path.realpath(p)
        if roots and not any(within(r, resolved) for r in roots):
            return err("PATH_NOT_ALLOWED", "input outside allowed roots", {"reason": "outside_allowed_roots"})
        if not os.path.isfile(resolved):
            return err("INVALID_INPUT", f"source {s['source_id']!r} not found", {"reason": "not_found"})
        sources[s["source_id"]] = resolved
    tracks = {t["track_id"]: t for t in proj.get("tracks") or []}
    for t in tracks.values():
        if t.get("source_id") not in sources:
            return err("MISSING_INPUT", "track references an unknown source")
    ops = proj.get("operations") or []
    if len(ops) != 1:
        return err("INVALID_REQUEST", "the fake handles one operation")
    op = ops[0]
    if op.get("type") in UNSUPPORTED:
        return err("UNSUPPORTED_OPERATION", f"{op['type']} is not implemented", {"type": op["type"]})
    if op.get("type") not in OPS:
        return err("UNSUPPORTED_OPERATION", f"{op.get('type')} is unknown", {"type": op.get("type")})
    params = op.get("parameters") or {}
    schema = OPS[op["type"]]["parameters"]
    if set(params) - set(schema):
        return err("INVALID_REQUEST", f"unknown parameters {sorted(set(params) - set(schema))}")
    for k, ps in schema.items():
        if ps.get("required") and k not in params:
            return err("INVALID_REQUEST", f"{op['type']} needs {k}")
        if k in params and ps.get("type") in ("number", "integer"):
            v = params[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or ("min" in ps and v < ps["min"]) or ("max" in ps and v > ps["max"]):
                return err("INVALID_REQUEST", f"{k} out of range")
    inputs = op.get("inputs") or []
    arity = OPS[op["type"]]["inputs"]
    if not (arity["min"] <= len(inputs) <= arity["max"]):
        return err("INVALID_REQUEST", f"{op['type']} takes {arity['min']}..{arity['max']} inputs")
    in_paths = []
    for ref in inputs:
        if not ref.startswith("track:") or ref[6:] not in tracks:
            return err("MISSING_INPUT", f"unknown input {ref!r}")
        in_paths.append(sources[tracks[ref[6:]]["source_id"]])
    outs = proj.get("outputs") or []
    if len(outs) != 1 or outs[0].get("operation") != f"op:{op.get('op_id')}":
        return err("DEPENDENCY_ERROR", "outputs must reference the operation")
    o = outs[0]
    fmt = o.get("format")
    if fmt not in CONTRACT["output_formats"]:
        return err("UNSUPPORTED_FORMAT", "format")
    raw_out = o.get("path", "")
    if any(part == ".." for part in Path(raw_out).parts):
        return err("PATH_NOT_ALLOWED", "traversal", {"reason": "traversal"})
    out_path = os.path.realpath(os.path.join(ws, raw_out)) if not os.path.isabs(raw_out) else os.path.realpath(raw_out)
    if not within(ws, out_path):
        return err("PATH_NOT_ALLOWED", "output must be inside the workspace", {"reason": "outside_workspace"})
    if not out_path.lower().endswith(CONTRACT["output_formats"][fmt]["extension"]):
        return err("UNSUPPORTED_FORMAT", "extension mismatch")
    if any(os.path.normcase(v) == os.path.normcase(out_path) for v in sources.values()):
        return err("OUTPUT_ERROR", "output would overwrite an input", {"reason": "input_output_collision"})
    if os.path.exists(out_path) and not o.get("overwrite"):
        return err("OUTPUT_ERROR", "output already exists", {"reason": "exists"})
    # ---- timeline arithmetic
    t = op["type"]
    durs = [in_dur(p) for p in in_paths]
    if t == "TRIM":
        dur = round(float(params["end"]) - float(params["start"]), 3)
    elif t in ("CUT", "SILENCE_REMOVE"):
        key = "remove" if t == "CUT" else "ranges"
        rs = params.get(key) or []
        for r in rs:
            if not (isinstance(r, dict) and isinstance(r.get("start"), (int, float)) and isinstance(r.get("end"), (int, float))):
                return err("INVALID_REQUEST", "ranges must be {start, end}")
            if not r["start"] < r["end"]:
                return err("INVALID_TIME_RANGE", "start must be before end")
            if r["end"] > durs[0] + 0.01:
                return err("INVALID_TIME_RANGE", f"range beyond the media duration {durs[0]:.3f}s")
        dur = round(durs[0] - sum(r["end"] - r["start"] for r in rs), 3)
        if dur <= 0:
            return err("INVALID_TIME_RANGE", "nothing would remain")
    elif t == "CONCAT":
        cf = float(params.get("crossfade") or 0.0)
        dur = round(sum(durs) - cf * (len(durs) - 1), 3)
    elif t == "MIX":
        dur = durs[0]
    else:
        dur = durs[0]
    if t in ("FADE_IN", "FADE_OUT") and float(params["duration"]) > dur + 0.001:
        return err("INVALID_TIME_RANGE", "fade longer than the media")
    channels = max(in_channels(p) for p in in_paths)
    if t == "MONO":
        if channels != 2:
            return err("INVALID_CHANNEL_LAYOUT", "MONO takes a 2-channel input", {"channels": channels})
        channels = 1
    elif t == "STEREO":
        if channels not in (1, 2):
            return err("INVALID_CHANNEL_LAYOUT", "STEREO takes a 1- or 2-channel input", {"channels": channels})
        channels = 2
    elif t == "DOWNMIX":
        if channels not in (6, 8):
            return err("INVALID_CHANNEL_LAYOUT", "DOWNMIX takes 5.1 / 7.1", {"channels": channels})
        channels = 2
    expect = o.get("expect") or {}
    if "channels" in expect and expect["channels"] != channels:
        return err("VALIDATION_ERROR", f"output has {channels} channel(s), expected {expect['channels']}", {"reason": "channel_mismatch"})
    if "duration" in expect and abs(float(expect["duration"]) - dur) > float(expect.get("duration_tolerance", 0.1)):
        return err("VALIDATION_ERROR", "duration mismatch", {"reason": "duration_mismatch"})
    src_hashes = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in in_paths]
    op_id = hashlib.sha256(json.dumps({"type": t, "params": params, "inputs": src_hashes, "tool_versions": {"ffmpeg-skill": "0.9.1-fake"}}, sort_keys=True).encode()).hexdigest()
    tool = OPS[t]["tool"]
    versions = {"ffmpeg-skill": "0.9.1-fake", "ffmpeg-skill_contract": "1.0", "ffmpeg": "fake", "ffprobe": "fake"}
    base = {"schema": RESPONSE, "skill": {"id": "audio-production", "version": CONTRACT["version"]}, "warnings": []}
    plan = {"plan_id": "p" * 64, "project_id": proj.get("project_id"), "work_dir": os.path.join(ws, ".audio-production", str(proj.get("project_id"))),
            "graph": {"order": [f"track:{tr}" for tr in tracks] + [f"op:{op['op_id']}"]},
            "steps": [{"node_id": f"op:{op['op_id']}", "operation_id": op_id, "type": t, "tool": tool, "inputs": inputs, "parameters": params, "expected_duration": dur}],
            "outputs": [{"output_id": o["output_id"], "node_id": f"op:{op['op_id']}", "path": out_path, "format": fmt, "expected_duration": dur}],
            "required_capabilities": OPS[t]["required_capabilities"], "tool_versions": versions, "intermediate_format": "wav", "duration_tolerance": 0.1}
    segments = [{"timeline": {"start": 0.0, "end": dur}, "source_id": tracks[inputs[0][6:]]["source_id"], "source": {"start": 0.0, "end": dur}, "input_index": 0}]
    record = {"node_id": f"op:{op['op_id']}", "operation_id": op_id, "type": t, "implicit": False, "tool": tool, "required_capabilities": OPS[t]["required_capabilities"], "inputs": inputs,
              "parameters": params, "status": "planned", "expected_duration": dur, "input_hashes": src_hashes, "seconds": 0.0, "segments": segments, "measurements": {}, "tool_commands_observed": []}
    if dry_run:
        emit({**base, "ok": True, "status": "ok", "dry_run": True, "plan": plan, "results": [record], "outputs": plan["outputs"]})
        return 0
    if MODE == "cancelled":
        return err("CANCELLED", "interrupted", {"reason": "signal"})
    if MODE == "timeout":
        return err("CANCELLED", "operation exceeded timeout", {"reason": "timeout"})
    if MODE in ("tool_error", "tool_error_final"):
        rec = dict(record, status="failed", error={"code": "TOOL_ERROR", "message": "ffmpeg-skill/audio failed", "retryable": MODE == "tool_error", "details": {"reason": "tool_failed"}})
        emit({**base, "ok": False, "status": "error", "dry_run": False, "plan": plan, "results": [rec], "outputs": [], "tool_runs": [],
              "error": {"code": "TOOL_ERROR", "message": "ffmpeg-skill/audio failed", "retryable": MODE == "tool_error", "details": {"reason": "tool_failed"}}})
        return EXIT["TOOL_ERROR"]
    if MODE == "validation_error":
        return err("VALIDATION_ERROR", "NORMALIZE: measured -19.2 LUFS is outside the tolerance of -16.0", {"reason": "loudness_out_of_tolerance"})
    # ---- write the output (bytes derived from the request so identical requests give identical hashes)
    payload = json.dumps({"fake": True, "duration": dur, "lufs": float(params["target_lufs"]) if t == "NORMALIZE" else -16.0, "channels": channels, "video": False, "op": op_id[:16], "src": sorted(src_hashes)}).encode()
    reused = MODE == "reused" and os.path.isfile(out_path)
    if MODE != "output_missing":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Path(out_path).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if MODE == "hash_mismatch":
        digest = "0" * 64
    artifact = {"path": out_path, "duration": dur, "channels": channels, "channel_layout": None, "sample_rate": 48000, "codec": CONTRACT["output_formats"][fmt]["codec"], "size": len(payload), "sha256": digest}
    record.update({"status": "reused" if reused else "completed", "seconds": 0.01, "artifact": dict(artifact, path=os.path.join(plan["work_dir"], op_id[:16] + ".wav")),
                   "tool_commands_observed": [f"{tool} {op_id[:16]} (fake, provenance only)"]})
    if t == "NORMALIZE":
        record["measurements"] = {"loudness": {"measured_by": "ffmpeg-skill/loudness --measure-only", "target_lufs": params["target_lufs"], "true_peak_db": params["true_peak_db"],
                                               "integrated_lufs": float(params["target_lufs"]) - 0.1, "true_peak_dbtp": float(params["true_peak_db"]) - 0.5, "loudness_range_lu": 3.0}}
    out_doc = {"output_id": o["output_id"], "status": "completed", "path": out_path, "format": fmt, "artifact": artifact, "segments": segments, "seconds": 0.01,
               "tool_commands_observed": [f"ffmpeg-skill/audio export (fake, provenance only)"],
               "provenance": {"skill": "audio-production", "skill_version": CONTRACT["version"], "tool": "ffmpeg-skill/audio", "tool_versions": versions, "output_hash": digest,
                              "node_id": f"op:{op['op_id']}", "operation_id": op_id,
                              "operations": [{"node_id": f"op:{op['op_id']}", "operation_id": op_id, "type": t, "status": record["status"], "tool": tool, "parameters": params, "input_hashes": src_hashes, "output_hash": digest}],
                              "sources": {sid: {"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(), "size": os.path.getsize(p)} for sid, p in sources.items()}}}
    if MODE == "no_provenance":
        out_doc.pop("provenance")
    doc = {**base, "ok": True, "status": "ok", "dry_run": False, "plan": plan, "results": [record], "outputs": [out_doc],
           "tool_runs": [{"tool": "ffmpeg-skill/probe", "exit_code": 0, "seconds": 0.01, "commands_observed": []}]}
    emit(doc)
    if MODE == "two_docs":
        emit(doc)
    return 3 if MODE == "nonzero_ok" else 0


if __name__ == "__main__":
    sys.exit(main())
