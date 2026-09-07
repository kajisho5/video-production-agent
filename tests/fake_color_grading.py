"""Fake color-grading-skill process for adapter boundary tests: speaks the real transport (`skill --json`, `doctor --json`,
`plan|run - --json --workspace D --allowed-input R… [--allowed-lut L]… [--ffmpeg-skill X] [--timeout S] [--dry-run]` with a colour
request on stdin → one response document on stdout) with canned data, enforces the request shape, the operation parameter schemas
and the path policy like the real Skill, writes a small self-describing output file (the format tests/fake_adapter.py probes),
and misbehaves on request via FAKE_CG_MODE:

  ok | reused | tool_error | tool_error_final | output_missing | validation_error | cancelled | timeout (hangs ~30 s) | hang |
  malformed | empty | two_docs | text | nonzero_ok | wrong_schema | wrong_skill | wrong_version | bad_contract | contract_fail |
  contract_drift | hash_mismatch | unknown_code | internal_error | doctor_fail | doctor_degraded | no_provenance | not_hdr (HDR_TO_SDR
  on an SDR source → TOOL_ERROR from the engine, like the real Skill without force)

FAKE_CG_CALLS: a file every invocation (subcommand + argv) is appended to. Never runs ffmpeg or ffmpeg-skill, never imports
color_grading. Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE.parent / "src" / "video_agent" / "tools" / "color_grading" / "contract_0.2.0.json").read_text(encoding="utf-8"))
MODE = os.environ.get("FAKE_CG_MODE", "ok")
CALLS_LOG = os.environ.get("FAKE_CG_CALLS")
EXIT = CONTRACT["errors"]["exit_codes"]
RETRY = CONTRACT["errors"]["retryable"]
OPS = {o["type"]: o for o in CONTRACT["operations"]}
UNSUPPORTED = {u["type"] for u in CONTRACT["unsupported_operations"]}
FORBIDDEN = set(CONTRACT["request"]["forbidden_fields"])
ID_RE = re.compile(CONTRACT["request"]["id_pattern"])
RESPONSE = "color-grading/response@1"
TOOL_VERSIONS = {"ffmpeg-skill": "0.9.2-fake", "ffmpeg-skill_contract": "1.0", "ffmpeg": "fake", "ffprobe": "fake"}


def log(kind: str, argv) -> None:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"cmd": kind, "argv": list(argv), "cwd": os.getcwd(), "env_video": {k: v for k, v in os.environ.items() if k.startswith("VIDEO_")}}) + "\n")


def emit(doc) -> None:
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def err(code: str, message: str, details=None, retryable=None, dry_run=False) -> int:
    emit({"schema": RESPONSE, "skill": {"id": "color-grading", "version": CONTRACT["version"]}, "ok": False, "status": "cancelled" if code == "CANCELLED" else "error",
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


def probe(path: str) -> dict:
    raw = Path(path).read_bytes()
    meta = {"duration": 16.0, "width": 640, "height": 360, "hdr": False, "video": True, "channels": 2}
    if raw.startswith(b'{"fake"'):
        try:
            meta.update({k: v for k, v in json.loads(raw.decode()).items() if k != "fake"})
        except (ValueError, TypeError):
            pass
    return meta


def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    argv = sys.argv[1:]
    cmd = argv[0] if argv else ""
    log(cmd, argv)
    if cmd in ("contract", "skill"):
        c = json.loads(json.dumps(CONTRACT))
        if MODE == "wrong_schema":
            c["schema"] = "color-grading/contract@2"
        if MODE == "wrong_skill":
            c["skill_id"] = "other-skill"; c["id"] = "other-skill"
        if MODE == "wrong_version":
            c["version"] = "9.0.0"; c["tools"][0]["version"] = "9.0.0"
        if MODE == "bad_contract":
            c["execution"]["shell"] = True
        if MODE == "contract_fail":
            sys.stderr.write("boom\n")
            return 1
        if MODE == "contract_drift":   # a compatible newer contract with one more operation: reported as drift, never silently kept
            c["operations"].append(dict(c["operations"][2], type="EXPOSURE", parameters={"stops": {"type": "number", "required": True, "min": -5, "max": 5}}))
            c["unsupported_operations"] = [u for u in c["unsupported_operations"] if u["type"] != "EXPOSURE"]
            c["tools"][0]["operations"] = sorted(c["tools"][0]["operations"] + ["EXPOSURE"])
        emit(c)
        return 0
    if cmd == "doctor":
        status = {"doctor_fail": "fail", "doctor_degraded": "degraded"}.get(MODE, "ok")
        ok = status != "fail"
        ops = {}
        for t, spec in OPS.items():
            st = "supported" if ok else "unavailable"
            if MODE == "doctor_degraded" and t == "LUT_APPLY":
                st = "unavailable"
            ops[t] = {"status": st, "tool": spec["tool"], "required_capabilities": spec["required_capabilities"], "missing": [] if st != "unavailable" else ["filter:lut3d"], "unknown": []}
        emit({"schema": "color-grading/doctor@1", "skill": {"id": "color-grading", "version": CONTRACT["version"]}, "status": status,
              "checks": {"python": {"status": "ok"}, "ffmpeg_skill": {"status": "ok" if ok else "missing", "directory": (flag_values(argv, "--ffmpeg-skill") or [None])[0], "version": "0.9.1-fake", "contract_version": "1.0"},
                         "ffmpeg": {"status": "ok" if ok else "missing"}, "ffprobe": {"status": "ok" if ok else "missing"}, "capabilities": {}, "operations": ops,
                         "unsupported_operations": {u: "" for u in UNSUPPORTED}, "output_formats": list(CONTRACT["output_formats"]),
                         "path_policy": {"status": "ok", "workspace": (flag_values(argv, "--workspace") or [None])[0], "allowed_input_roots": flag_values(argv, "--allowed-input"), "allowed_lut_roots": flag_values(argv, "--allowed-lut")}},
              "unavailable_operations": [t for t, o in ops.items() if o["status"] == "unavailable"], "problems": [] if ok else ["ffmpeg-skill: not found"], "warnings": [], "secrets_shown": False})
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
    lut_roots = [os.path.realpath(r) for r in flag_values(argv, "--allowed-lut")] or roots
    try:
        req = json.loads(sys.stdin.read())
    except ValueError:
        return err("INVALID_REQUEST", "request is not valid JSON")
    if MODE in ("hang", "timeout"):
        time.sleep(60 if MODE == "hang" else 30)
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
    hit = scan(req, "request")
    if hit:
        return err("INVALID_REQUEST", f"forbidden field {hit}", {"field": hit})
    if not isinstance(req, dict) or req.get("schema") != CONTRACT["request"]["schema"] or set(req) - {"schema", "project", "options"} or "project" not in req:
        return err("INVALID_REQUEST", "unsupported request shape / schema", {"field": "schema"})
    pj = req["project"]
    if not isinstance(pj, dict) or set(pj) != {"project_id", "source", "operations", "outputs"} or not ID_RE.match(str(pj["project_id"])):
        return err("INVALID_REQUEST", "project shape")
    src_raw = pj["source"].get("path")
    src = os.path.realpath(src_raw) if isinstance(src_raw, str) else ""
    if roots and not any(within(r, src) for r in roots):
        return err("PATH_NOT_ALLOWED", "source outside allowed roots", {"reason": "outside_allowed_roots"})
    if not os.path.isfile(src):
        return err("INVALID_INPUT", "source not found", {"reason": "not_found"})
    ops = pj["operations"]
    if not isinstance(ops, list) or not ops or len(ops) > CONTRACT["request"]["max_operations"]:
        return err("INVALID_REQUEST", "operations must be a non-empty array")
    meta = probe(src)
    results = []
    for i, op in enumerate(ops):
        where = f"operations[{i}]"
        if not isinstance(op, dict) or set(op) != {"op_id", "type", "input", "parameters"} or not ID_RE.match(str(op["op_id"])):
            return err("INVALID_REQUEST", f"{where}: operation shape")
        t = op["type"]
        if t in UNSUPPORTED:
            return err("UNSUPPORTED_OPERATION", f"{where}.type {t!r} is not implemented", {"type": t})
        if t not in OPS:
            return err("UNSUPPORTED_OPERATION", f"{where}.type {t!r} is unknown", {"type": t})
        schema = OPS[t]["parameters"]
        params = op["parameters"]
        if not isinstance(params, dict) or set(params) - set(schema):
            return err("INVALID_REQUEST", f"{where}: unknown parameters", {"field": f"{where}.parameters"})
        for name, spec in schema.items():
            if spec.get("required") and name not in params:
                return err("INVALID_REQUEST", f"{where}: {t} needs {name}")
            if name not in params:
                continue
            v = params[name]
            if spec["type"] == "string":
                if not isinstance(v, str) or ("enum" in spec and v not in spec["enum"]):
                    return err("INVALID_REQUEST", f"{where}.{name} invalid")
            elif spec["type"] == "boolean":
                if not isinstance(v, bool):
                    return err("INVALID_REQUEST", f"{where}.{name} must be boolean")
            else:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or ("min" in spec and v < spec["min"]) or ("max" in spec and v > spec["max"]):
                    return err("INVALID_REQUEST", f"{where}.{name} out of range")
        if t == "LUT_APPLY":
            lut = os.path.realpath(str(params["lut_path"]))
            if lut_roots and not any(within(r, lut) for r in lut_roots):
                return err("PATH_NOT_ALLOWED", "LUT outside allowed roots", {"reason": "outside_allowed_lut_roots"})
            if not os.path.isfile(lut) or Path(lut).suffix.lower() != ".cube":
                return err("INVALID_INPUT", "LUT not found or not a .cube", {"reason": "lut"})
        if t == "HDR_TO_SDR" and (MODE == "not_hdr" or not meta.get("hdr")) and not params.get("force"):
            return err("TOOL_ERROR", "ffmpeg-skill/color --to-sdr failed: source is not tagged HDR (use force to tone-map anyway)", {"reason": "not_hdr"})   # like the real Skill: the engine refuses, reported as TOOL_ERROR
        results.append({"node_id": f"op:{op['op_id']}", "operation_id": hashlib.sha256(json.dumps([t, params, sha(src)], sort_keys=True).encode()).hexdigest(), "type": t,
                        "tool": OPS[t]["tool"], "required_capabilities": OPS[t]["required_capabilities"], "input": op["input"], "parameters": params,
                        "status": "reused" if MODE == "reused" else "completed", "input_hash": sha(src), "seconds": 0.01, "tool_commands_observed": ["fake ffmpeg -i in -vf x out"], "measurements": {}})
    outs = pj["outputs"]
    if not isinstance(outs, list) or len(outs) != 1:
        return err("INVALID_REQUEST", "exactly one output is expected by this fake")
    o = outs[0]
    if set(o) - {"output_id", "operation", "path", "format", "overwrite", "expect"} or o.get("format") not in CONTRACT["output_formats"]:
        return err("UNSUPPORTED_FORMAT", "output shape / format")
    out_path = os.path.realpath(str(o["path"]))
    if not within(ws, out_path):
        return err("PATH_NOT_ALLOWED", "output must be inside the workspace", {"reason": "outside_workspace"})
    if Path(out_path).suffix.lstrip(".").lower() != o["format"] or Path(src).suffix.lstrip(".").lower() != o["format"]:
        return err("UNSUPPORTED_FORMAT", "the container is never converted", {"reason": "container"})
    plan = {"plan_id": hashlib.sha256(json.dumps(pj, sort_keys=True).encode()).hexdigest(), "project_id": pj["project_id"], "work_dir": os.path.join(ws, ".color-grading", pj["project_id"]),
            "graph": {"order": ["source"] + [r["node_id"] for r in results], "nodes": [], "outputs": {o["output_id"]: o["operation"]}},
            "source": {"path": src, "sha256": sha(src), "size": os.path.getsize(src), "duration": meta["duration"], "has_video": True, "has_audio": bool(meta.get("channels")), "width": meta["width"], "height": meta["height"], "hdr": bool(meta.get("hdr"))}}
    base = {"schema": RESPONSE, "skill": {"id": "color-grading", "version": CONTRACT["version"]}, "ok": True, "warnings": [], "plan": plan}
    if dry_run:
        emit({**base, "status": "ok", "dry_run": True, "results": [], "outputs": [], "tool_runs": []})
        return 0
    if MODE in ("tool_error", "tool_error_final"):
        return err("TOOL_ERROR", "ffmpeg exited 1", {"reason": "tool_failed", "argv": ["never", "forwarded"]}, retryable=MODE == "tool_error")
    if MODE == "validation_error":
        return err("VALIDATION_ERROR", "output colour tags do not match the request", {"reason": "verification"})
    if MODE == "cancelled":
        return err("CANCELLED", "interrupted", {"reason": "signal"})
    last = results[-1]
    payload = {"fake": True, "duration": meta["duration"], "video": True, "channels": meta.get("channels", 2), "hdr": False if last["type"] == "HDR_TO_SDR" else bool(meta.get("hdr")),
               "color": last["type"].lower(), "lufs": meta.get("lufs", -11.0)}
    if MODE != "output_missing":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Path(out_path).write_bytes(json.dumps(payload).encode())
    digest = sha(out_path) if os.path.exists(out_path) else "0" * 64
    if MODE == "hash_mismatch":
        digest = "1" * 64
    art = {"path": out_path, "duration": meta["duration"], "width": meta["width"], "height": meta["height"], "pix_fmt": "yuv420p", "codec": "h264",
           "color_space": "bt709", "color_primaries": "bt709", "color_transfer": "bt709", "color_range": "tv", "hdr": payload["hdr"], "dolby_vision": False,
           "size": os.path.getsize(out_path) if os.path.exists(out_path) else 0, "sha256": digest}
    prov = {"skill": "color-grading", "skill_version": CONTRACT["version"], "tool_versions": TOOL_VERSIONS, "output_hash": digest, "node_id": last["node_id"], "operation_id": last["operation_id"],
            "operations": [{k: r.get(k) for k in ("node_id", "operation_id", "type", "status", "tool", "parameters", "input_hash")} | {"output_hash": digest} for r in results], "source": {"sha256": sha(src)}}
    if MODE == "no_provenance":
        prov = {}
    doc = {**base, "status": "ok", "dry_run": False, "results": results,
           "outputs": [{"output_id": o["output_id"], "status": "completed", "path": out_path, "format": o["format"], "artifact": art, "seconds": 0.02, "tool_commands_observed": [], "provenance": prov}],
           "tool_runs": [{"tool": "ffmpeg-skill/probe", "exit_code": 0, "seconds": 0.01, "commands_observed": []}, {"tool": "ffmpeg-skill/color", "exit_code": 0, "seconds": 0.01, "commands_observed": ["fake ffmpeg -i in -vf x out"]}]}
    emit(doc)
    if MODE == "two_docs":
        emit(doc)
    return 1 if MODE == "nonzero_ok" else 0


if __name__ == "__main__":
    sys.exit(main())
