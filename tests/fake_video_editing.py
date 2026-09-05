"""Fake video-editing-skill process for adapter boundary tests: speaks the real transport (`contract --json`, `doctor --json`,
`plan|run - --json --workspace D --allowed-input R… [--ffmpeg-skill-dir X]` with an EditRequest on stdin → one response
document on stdout) with canned data, enforces the request shape and the path policy like the real Skill, writes a small
output file, and misbehaves on request via FAKE_VE_MODE:

  ok | reused | tool_error | tool_error_final | output_missing | validation_error | cancelled | timeout | hang | malformed | empty |
  two_docs | text | nonzero_ok | wrong_schema | wrong_skill | wrong_version | bad_contract | contract_fail | hash_mismatch |
  no_observation | unknown_code | internal_error | doctor_fail

FAKE_VE_CALLS: a file every invocation (subcommand + argv) is appended to (security tests read it). Never runs ffmpeg or
ffmpeg-skill, never imports video_editing_skill. Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE.parent / "src" / "video_agent" / "tools" / "video_editing" / "contract_0.1.0.json").read_text(encoding="utf-8"))
MODE = os.environ.get("FAKE_VE_MODE", "ok")
CALLS_LOG = os.environ.get("FAKE_VE_CALLS")
EXIT = CONTRACT["errors"]["exit_codes"]
RETRY = CONTRACT["errors"]["retryable_default"]
OPS = CONTRACT["operations"]


def log(kind: str, argv) -> None:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"cmd": kind, "argv": list(argv), "cwd": os.getcwd(), "env_video": {k: v for k, v in os.environ.items() if k.startswith("VIDEO_")}}) + "\n")


def emit(doc) -> None:
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def err(code: str, message: str, details=None, retryable=None) -> int:
    emit({"ok": False, "error": {"code": code, "message": message, "retryable": RETRY[code] if retryable is None else retryable, "details": details or {}}})
    return EXIT[code]


def within(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(root), os.path.normcase(path)]) == os.path.normcase(root)
    except ValueError:
        return False


def flag_values(argv, name):
    return [argv[i + 1] for i, a in enumerate(argv) if a == name and i + 1 < len(argv)]


def main() -> int:
    argv = sys.argv[1:]
    cmd = argv[0] if argv else ""
    log(cmd, argv)
    if cmd in ("contract", "skill"):
        c = json.loads(json.dumps(CONTRACT))
        if MODE == "wrong_schema":
            c["schema"] = "video-editing/contract@2"
        if MODE == "wrong_skill":
            c["skill_id"] = "other-skill"
        if MODE == "wrong_version":
            c["version"] = "9.0.0"
        if MODE == "bad_contract":
            c["execution"]["shell"] = True
        if MODE == "contract_fail":
            sys.stderr.write("boom\n")
            return 1
        if MODE == "contract_drift":   # a compatible newer contract: one more operation / tool (the adapter must notice, never silently keep 0.1.0 expectations)
            c["operations"]["CROP"] = {"capability": "video.crop", "tool": "ffmpeg-skill/crop", "arity": "one", "parameters": {}}
            c["tools"].append(dict(c["tools"][0], tool_id="video-editing/crop", operation_type="CROP", capability="video.crop", result_keys=list(CONTRACT["tools"][0]["result_keys"])))
        emit(c)
        return 0
    if cmd == "doctor":
        ok = MODE != "doctor_fail"
        checks = [{"check": "python", "status": "AVAILABLE"}, {"check": "skill", "status": "AVAILABLE", "id": "video-editing", "version": CONTRACT["version"]},
                  {"check": "ffmpeg-skill", "status": "AVAILABLE" if ok else "MISSING", "version": "0.9.0-fake", "root": flag_values(argv, "--ffmpeg-skill-dir")[:1]},
                  {"check": "ffmpeg", "status": "AVAILABLE" if ok else "MISSING", "version": "fake", "source": "ffmpeg-skill doctor"},
                  {"check": "ffprobe", "status": "AVAILABLE" if ok else "MISSING", "version": "fake", "source": "ffmpeg-skill doctor"},
                  {"check": "path_policy", "status": "AVAILABLE" if flag_values(argv, "--workspace") else "UNKNOWN", "workspace": flag_values(argv, "--workspace")[:1],
                   "allowed_input_roots": flag_values(argv, "--allowed-input")}]
        emit({"schema": "video-editing/doctor@1", "ok": ok, "skill": {"id": "video-editing", "version": CONTRACT["version"]}, "checks": checks,
              "problems": [] if ok else ["ffmpeg-skill not found"], "summary": "ready to edit" if ok else "not ready: ffmpeg-skill not found", "secrets_shown": False})
        return 0 if ok else 1
    if cmd not in ("run", "plan", "validate"):
        sys.stderr.write("unknown command\n")
        return 2
    if argv[1:2] != ["-"] or "--json" not in argv:
        return err("INVALID_REQUEST", "request must come from stdin with --json")
    ws_list = flag_values(argv, "--workspace")
    if not ws_list:
        return err("INVALID_REQUEST", "--workspace is required")
    ws = os.path.realpath(ws_list[0])
    if not os.path.isdir(ws):
        return err("PATH_NOT_ALLOWED", "workspace is not an existing directory", {"reason": "workspace_missing"})
    roots = [os.path.realpath(r) for r in flag_values(argv, "--allowed-input")] or [ws]
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
    # ---- request validation (a subset of the real Skill's, same codes)
    if not isinstance(req, dict) or req.get("schema") != CONTRACT["schemas"]["request"] or set(req) - {"schema", "project", "options"}:
        return err("INVALID_REQUEST", "request.schema / keys")
    proj = req.get("project") or {}
    forbidden = ("command", "cmd", "argv", "args", "shell", "exec", "executable", "script", "filter", "filter_complex", "ffmpeg", "binary", "env")

    def scan(obj, what):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() in forbidden:
                    raise ValueError(f"{what}: key {k!r} is not accepted")
                scan(v, f"{what}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan(v, f"{what}[{i}]")
    try:
        scan(req, "request")
    except ValueError as e:
        return err("INVALID_REQUEST", str(e), {"reason": "forbidden_key"})
    sources = {}
    for s in proj.get("sources") or []:
        raw = s.get("path", "")
        if any(p == ".." for p in raw.replace("\\", "/").split("/")):
            return err("PATH_NOT_ALLOWED", "path contains '..'", {"reason": "traversal"})
        absolute = os.path.abspath(raw if os.path.isabs(raw) else os.path.join(ws, raw))
        resolved = os.path.realpath(absolute)
        if not any(within(r, resolved) for r in roots):
            escaped = any(within(r, os.path.normpath(absolute)) for r in roots)
            return err("PATH_NOT_ALLOWED", "input is outside the allowed input roots", {"reason": "symlink_escape" if escaped else "outside_allowed_roots"})
        if not os.path.isfile(resolved):
            return err("MISSING_INPUT", "input not found")
        ext = os.path.splitext(resolved)[1].lower()
        if s.get("kind", "video") == "video" and ext not in CONTRACT["formats"]["video_inputs"]:
            return err("UNSUPPORTED_FORMAT", f"extension {ext} is not supported")
        sources[s["id"]] = resolved
    ops = proj.get("operations") or []
    if len(ops) != 1:
        return err("INVALID_REQUEST", "the fake handles one operation")
    op = ops[0]
    if op.get("type") not in OPS:
        return err("UNSUPPORTED_OPERATION", f"{op.get('type')} is not implemented", {"type": op.get("type"), "supported": sorted(OPS)})
    params = op.get("params") or {}
    if set(params) - set(OPS[op["type"]]["parameters"]):
        return err("INVALID_REQUEST", f"unknown parameters {sorted(set(params) - set(OPS[op['type']]['parameters']))}")
    if op["type"] in ("CUT", "TRIM") and params.get("precision", "frame") not in ("frame", "keyframe"):
        return err("INVALID_REQUEST", "precision: must be one of ['frame', 'keyframe']")
    if op["type"] in ("CUT", "TRIM"):
        ranges = params.get("keep") if op["type"] == "CUT" else [{"start": params.get("start"), "end": params.get("end")}]
        for r in ranges or []:
            if not (isinstance(r, dict) and isinstance(r.get("start"), (int, float)) and isinstance(r.get("end"), (int, float))):
                return err("INVALID_REQUEST", "keep ranges must be {start, end}")
            if not r["start"] < r["end"]:
                return err("INVALID_TIME_RANGE", "start must be before end")
            if r["end"] > 16.0 + 0.1:
                return err("INVALID_TIME_RANGE", f"end {r['end']}s is beyond the input duration 16.000s")
    inputs = op.get("inputs") or [op.get("input")]
    for ref in inputs:
        if ref not in sources:
            return err("DEPENDENCY_ERROR", f"unknown input {ref!r}", {"reason": "unknown_reference"})
    outs = proj.get("outputs") or []
    if len(outs) != 1 or outs[0].get("operation") != op.get("id"):
        return err("DEPENDENCY_ERROR", "outputs must reference the operation")
    raw_out = outs[0].get("path", "")
    if os.path.isabs(raw_out):
        return err("PATH_NOT_ALLOWED", "output paths are relative to the workspace", {"reason": "absolute_output"})
    out_path = os.path.realpath(os.path.join(ws, raw_out))
    if not within(ws, out_path):
        return err("PATH_NOT_ALLOWED", "output must be inside the workspace", {"reason": "workspace_escape"})
    if os.path.splitext(out_path)[1].lower() not in CONTRACT["formats"]["outputs"]:
        return err("UNSUPPORTED_FORMAT", "output extension")
    if any(os.path.normcase(v) == os.path.normcase(out_path) for v in sources.values()):
        return err("PATH_NOT_ALLOWED", "output would overwrite an input", {"reason": "overwrite_input"})
    if os.path.exists(out_path) and not (req.get("options") or {}).get("overwrite"):
        return err("PATH_NOT_ALLOWED", "output already exists", {"reason": "exists"})
    if cmd == "validate":
        emit({"ok": True, "schema": CONTRACT["schemas"]["response"], "skill": {"id": "video-editing", "version": CONTRACT["version"]}, "status": "valid", "command": "validate", "project": proj, "warnings": []})
        return 0
    op_id = "op_" + hashlib.sha256(json.dumps({"type": op["type"], "params": params, "inputs": [hashlib.sha256(Path(sources[i]).read_bytes()).hexdigest() for i in inputs]}, sort_keys=True).encode()).hexdigest()[:16]
    tool = OPS[op["type"]]["tool"]
    commands = [f"{tool} {op_id} (fake, provenance only)"]
    base = {"schema": CONTRACT["schemas"]["response"], "skill": {"id": "video-editing", "version": CONTRACT["version"]}, "project": {"id": proj.get("id"), "operations": [{"id": op["id"], "operation_id": op_id, "type": op["type"]}]},
            "warnings": []}
    if cmd == "plan":
        emit({"ok": True, **base, "status": "planned", "command": "plan", "schema": CONTRACT["schemas"]["plan"], "dry_run": True, "engine": {"ffmpeg-skill": "0.9.0-fake"},
              "plan": {"work_dir": os.path.join(ws, ".video-editing", "work"), "steps": [{"id": op["id"], "type": op["type"], "tool": tool, "operation_id": op_id, "idempotency_key": "k" * 64,
                                                                                         "reusable": False, "timeline": None, "preview": {"ok": True, "commands": commands, "note": None}}]}})
        return 0
    if MODE == "cancelled":
        return err("CANCELLED", "interrupted", {"reason": "signal"})
    if MODE == "timeout":
        return err("CANCELLED", "operation exceeded timeout", {"reason": "timeout"})
    if MODE in ("tool_error", "tool_error_final"):
        record = {"operation": op["id"], "operation_id": op_id, "type": op["type"], "status": "failed", "skill": "video-editing", "skill_version": CONTRACT["version"], "tool": tool,
                  "commands": commands, "error": {"code": "TOOL_ERROR", "message": "ffmpeg failed", "retryable": MODE == "tool_error", "details": {}}}
        emit({"ok": False, "error": record["error"], **base, "status": "failed", "execution": {"status": "failed", "operations": [record], "outputs": [{"id": outs[0]["id"], "operation": op["id"], "path": out_path, "delivered": False}]}})
        return EXIT["TOOL_ERROR"]
    if MODE == "validation_error":
        return err("VALIDATION_ERROR", "output duration 0.500s differs from the expected 2.000s", {"observed": 0.5, "expected": 2.0})
    if MODE == "nonzero_ok":
        pass
    # ---- write the output (bytes derived from the request so identical requests give identical hashes)
    def in_dur(ref: str) -> float:
        """Duration of a fake input (its self-describing payload) or a fixed 16 s for a real / opaque file."""
        raw = Path(sources[ref]).read_bytes()
        if raw.startswith(b'{"fake"'):
            try:
                return float(json.loads(raw.decode()).get("duration") or 16.0)
            except (ValueError, TypeError):
                return 16.0
        return 16.0
    if op["type"] == "CUT":
        dur = round(sum(float(r["end"]) - float(r["start"]) for r in params["keep"]), 3)
    elif op["type"] == "TRIM":
        dur = round(float(params["end"]) - float(params["start"]), 3)
    elif op["type"] == "CONCAT":
        tr = float((params.get("transition") or {}).get("duration") or 0.0)
        dur = round(sum(in_dur(i) for i in inputs) - tr * (len(inputs) - 1), 3)
    elif op["type"] == "SPEED":
        f = params["factor"]
        f = (float(f.split("/")[0]) / float(f.split("/")[1])) if isinstance(f, str) else float(f)
        dur = round(in_dur(inputs[0]) / f, 3)
    else:
        dur = round(in_dur(inputs[0]), 3)   # FIT / FILL / RESIZE / OVERLAY keep the duration
    # "fake" first: the same self-describing format tests/fake_adapter.py writes, so the fake ffmpeg-skill adapter can probe the output
    payload = json.dumps({"fake": True, "duration": dur, "lufs": -16.0, "op": op_id, "src": sorted(hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources.values())}).encode()   # "fake" first (key order is fixed, not sorted)
    reused = MODE == "reused" and os.path.isfile(out_path)
    if MODE != "output_missing":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Path(out_path).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if MODE == "hash_mismatch":
        digest = "0" * 64
    probe = {"file": out_path, "duration": dur, "size_bytes": len(payload), "video": {"codec": "h264", "width": 640, "height": 360, "fps": 30.0}, "audio": {"codec": "aac"}}
    observation = None if MODE == "no_observation" else {"kind": "media.probe", "provenance": "OBSERVED", "source": "ffmpeg-skill/probe@0.9.0-fake", "data": probe}
    timeline = {"duration_known": True, "duration": {"seconds": f"{dur:.6f}", "rational": f"{int(round(dur * 1000))}/1000"},
                "tracks": [{"id": "V1", "kind": "video", "segments": [{"source": inputs[0], "source_range": {"start": {"seconds": "0.000000"}, "end": {"seconds": f"{dur:.6f}"}}, "speed": "1/1"}]}]}
    record = {"operation": op["id"], "operation_id": op_id, "type": op["type"], "capability": OPS[op["type"]]["capability"], "status": "reused" if reused else "completed",
              "skill": "video-editing", "skill_version": CONTRACT["version"], "tool": tool, "tool_versions": {"ffmpeg-skill": "0.9.0-fake", "ffmpeg": "fake", "ffprobe": "fake"},
              "idempotency_key": "k" * 64, "parameters": params, "inputs": [{"ref": i, "kind": "source", "sha256": hashlib.sha256(Path(sources[i]).read_bytes()).hexdigest()} for i in inputs],
              "output": {"path": os.path.join(ws, ".video-editing", "work", op_id + ".mp4"), "sha256": digest}, "probe": probe, "commands": [] if reused else commands,
              "started_at": "2026-09-05T00:00:00Z", "finished_at": "2026-09-05T00:00:01Z", "seconds": 0.01, "provenance": "OBSERVED"}
    delivered = MODE != "output_missing"
    out_doc = {"id": outs[0]["id"], "operation": op["id"], "path": out_path, "delivered": delivered}
    if delivered:
        out_doc.update({"sha256": digest, "size": len(payload), "timeline": timeline, "observation": observation})
    doc = {"ok": True, **base, "status": "reused" if reused else "completed", "command": "run", "engine": {"ffmpeg-skill": "0.9.0-fake", "ffmpeg": "fake", "ffprobe": "fake"},
           "execution": {"status": "completed", "started_at": "2026-09-05T00:00:00Z", "finished_at": "2026-09-05T00:00:01Z", "work_dir": os.path.join(ws, ".video-editing", "work"),
                         "operations": [record], "outputs": [out_doc]}}
    emit(doc)
    if MODE == "two_docs":
        emit(doc)
    return 3 if MODE == "nonzero_ok" else 0


if __name__ == "__main__":
    sys.exit(main())
