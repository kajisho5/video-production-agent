"""Fake video-editing-skill process for adapter protocol tests: speaks the contract@1 / response@1 protocol
(contract --json, doctor --json, plan - --json, run - --json) with canned data, writes a small output file and can
misbehave on request via FAKE_VE_MODE = ok | reused | empty | text | two_docs | wrong_schema | wrong_skill | wrong_version |
wrong_version_contract | wrong_schema_contract | no_output | wrong_path | no_sha | no_observation | bad_observation |
not_delivered | exit_nonzero_ok | unknown_code | error:<CODE> | hang | crash. Never runs ffmpeg. Test double only.
FAKE_VE_REQUEST=<file> records every request document received; FAKE_VE_CALLS=<file> records the argv of every call."""
from __future__ import annotations

import hashlib
from fractions import Fraction
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE.parent / "src" / "video_agent" / "tools" / "video_editing" / "contract_0.1.0.json").read_text(encoding="utf-8"))
MODE = os.environ.get("FAKE_VE_MODE", "ok")
VERSION = CONTRACT["version"]
EXIT = CONTRACT["errors"]["exit_codes"]
RETRY = CONTRACT["errors"]["retryable_default"]


def log(path_env: str, text: str) -> None:
    p = os.environ.get(path_env)
    if p:
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def flags(argv, name):
    return [argv[i + 1] for i, a in enumerate(argv) if a == name]


def fake_duration(op):
    """Duration of the fake output: the kept ranges of a CUT / TRIM, 2.5 s otherwise (mirrors tests/fake_adapter.py)."""
    params = op.get("params") or {}
    if op.get("type") == "CUT" and params.get("keep"):
        return round(sum(float(r["end"]) - float(r["start"]) for r in params["keep"]), 3)
    if op.get("type") == "TRIM" and params.get("start") is not None and params.get("end") is not None:
        return round(float(params["end"]) - float(params["start"]), 3)
    return 2.5


def error(code: str, message: str, details=None):
    print(json.dumps({"ok": False, "error": {"code": code, "message": message, "retryable": RETRY.get(code, False), "details": details or {}}, "schema": "video-editing/response@1",
                      "skill": {"id": "video-editing", "version": VERSION}, "status": "failed"}))
    return EXIT.get(code, 1)


def main() -> int:
    argv = sys.argv[1:]
    log("FAKE_VE_CALLS", json.dumps(argv))
    if argv[:1] in (["contract"], ["skill"]):
        c = json.loads(json.dumps(CONTRACT))
        if MODE == "wrong_version_contract":
            c["version"] = "9.0.0"
        if MODE == "wrong_schema_contract":
            c["schema"] = "video-editing/contract@2"
        print(json.dumps(c))
        return 0
    if argv[:1] == ["doctor"]:
        ok = MODE != "doctor_not_ready"
        print(json.dumps({"schema": "video-editing/doctor@1", "ok": ok, "skill": {"id": "video-editing", "version": VERSION},
                          "checks": [{"check": "ffmpeg-skill", "status": "AVAILABLE" if ok else "MISSING", "version": "0.9.0"}, {"check": "ffmpeg", "status": "AVAILABLE", "version": "fake"},
                                     {"check": "path_policy", "status": "AVAILABLE", "workspace": flag(argv, "--workspace"), "allowed_input_roots": flags(argv, "--allowed-input")}],
                          "problems": [] if ok else ["ffmpeg-skill not found"], "summary": "ready to edit" if ok else "not ready", "secrets_shown": False}))
        return 0 if ok else 1
    if argv[:2] not in (["run", "-"], ["plan", "-"]):
        print("unknown", file=sys.stderr)
        return 2
    raw = sys.stdin.read()
    log("FAKE_VE_REQUEST", raw)
    try:
        req = json.loads(raw)
    except ValueError:
        return error("INVALID_REQUEST", "request is not valid JSON")
    workspace = flag(argv, "--workspace")
    if not workspace:
        return error("INVALID_REQUEST", "--workspace is required")
    if MODE == "hang":
        time.sleep(30)
    if MODE == "crash":
        print("boom", file=sys.stderr)
        return 7
    if MODE == "empty":
        return 0
    if MODE == "text":
        print("not json at all")
        return 0
    if MODE.startswith("error:"):
        code = MODE.split(":", 1)[1]
        return error(code, f"fake {code}", {"reason": "timeout"} if code == "CANCELLED" else {"tool": "cut"})
    if MODE == "unknown_code":
        print(json.dumps({"ok": False, "error": {"code": "SOMETHING_ELSE", "message": "x", "retryable": True, "details": {}}}))
        return 1
    project = req.get("project") or {}
    op = (project.get("operations") or [{}])[0]
    if op.get("type") not in CONTRACT["operations"]:
        return error("UNSUPPORTED_OPERATION", f"{op.get('type')} is not implemented", {"type": op.get("type")})
    for key in ("command", "argv", "shell", "filter"):
        if key in json.dumps(req):
            return error("INVALID_REQUEST", f"key {key!r} is not accepted", {"reason": "forbidden_key"})
    out_rel = (project.get("outputs") or [{}])[0].get("path", "")
    if os.path.isabs(out_rel) or out_rel.startswith(("/", "\\")) or ".." in out_rel.split("/"):
        return error("PATH_NOT_ALLOWED", "output paths are relative to the workspace", {"reason": "absolute_output"})
    out_abs = os.path.join(workspace, out_rel.replace("/", os.sep))
    skill_doc = {"id": "video-editing", "version": VERSION}
    if argv[0] == "plan":
        print(json.dumps({"ok": True, "schema": "video-editing/plan@1", "skill": skill_doc, "status": "planned", "command": "plan", "dry_run": True,
                          "engine": {"ffmpeg-skill": "0.9.0", "ffmpeg": "fake", "ffprobe": "fake"},
                          "plan": {"work_dir": os.path.join(workspace, ".video-editing", "work"), "steps": [{"operation": "edit", "tool": "ffmpeg-skill/cut", "preview": {"ok": True, "commands": ["ffmpeg -i fake"]}}]},
                          "project": project, "warnings": []}))
        return 0
    duration = fake_duration(op)
    if MODE != "no_output":
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)
        with open(out_abs, "wb") as fh:
            # the same self-describing format tests/fake_adapter.py writes, so the agent's fake ffmpeg-skill adapter can probe it
            fh.write(json.dumps({"fake": True, "duration": duration, "lufs": -16.0, "op": op}).encode())   # "fake" first: that prefix is the marker
    sha = hashlib.sha256(open(out_abs, "rb").read()).hexdigest() if os.path.exists(out_abs) else "0" * 64
    status = "reused" if MODE == "reused" else "completed"
    delivered_path = out_abs if MODE != "wrong_path" else os.path.join(workspace, "elsewhere.mp4")
    output = {"id": "out", "operation": op.get("id"), "path": delivered_path, "delivered": MODE != "not_delivered",
              "sha256": None if MODE == "no_sha" else sha, "size": os.path.getsize(out_abs) if os.path.exists(out_abs) else 0,
              "timeline": {"duration_known": True, "duration": {"seconds": f"{duration:.6f}", "rational": str(Fraction(duration).limit_denominator(1000))}, "tracks": [{"id": "V1", "kind": "video", "segments": []}]},
              "observation": {"kind": "media.probe", "provenance": "OBSERVED", "source": "ffmpeg-skill/probe@0.9.0",
                              "data": {"duration": duration, "video": {"codec": "h264", "width": 320, "height": 180, "fps": 30.0}, "audio": None}}}
    if MODE == "no_observation":
        output.pop("observation")
    if MODE == "bad_observation":
        output["observation"] = {"kind": "media.probe", "provenance": "INFERRED", "source": "ai:model", "data": {}}
    rec = {"operation": op.get("id"), "operation_id": "op_" + sha[:16], "type": op.get("type"), "capability": "video." + str(op.get("type")).lower(), "status": status,
           "skill": "video-editing", "skill_version": VERSION, "tool": "ffmpeg-skill/cut", "tool_versions": {"ffmpeg-skill": "0.9.0", "ffmpeg": "fake", "ffprobe": "fake"},
           "idempotency_key": "k" * 64, "parameters": op.get("params"), "inputs": [{"ref": s["id"], "kind": "source", "sha256": "a" * 64} for s in project.get("sources") or []],
           "output": {"path": out_abs, "sha256": sha}, "probe": output["observation"]["data"] if "observation" in output else None,
           "commands": ["/usr/bin/ffmpeg -hide_banner -i fake -c:v libx264 " + out_abs], "started_at": "2026-09-05T00:00:00Z", "finished_at": "2026-09-05T00:00:01Z", "seconds": 0.5, "provenance": "OBSERVED"}
    doc = {"ok": True, "schema": "video-editing/response@1", "skill": skill_doc, "status": status, "command": "run",
           "engine": {"ffmpeg-skill": "0.9.0", "ffmpeg": "fake", "ffprobe": "fake"},
           "execution": {"status": status, "started_at": "2026-09-05T00:00:00Z", "finished_at": "2026-09-05T00:00:01Z", "work_dir": os.path.join(workspace, ".video-editing", "work"),
                         "operations": [rec], "outputs": [output]},
           "project": project, "warnings": []}
    if MODE == "wrong_schema":
        doc["schema"] = "video-editing/response@2"
    if MODE == "wrong_skill":
        doc["skill"]["id"] = "other"
    if MODE == "wrong_version":
        doc["skill"]["version"] = "9.9.9"
    print(json.dumps(doc))
    if MODE == "two_docs":
        print(json.dumps(doc))
    return 3 if MODE == "exit_nonzero_ok" else 0


if __name__ == "__main__":
    sys.exit(main())
