"""Fake subtitle-skill process for adapter boundary tests: speaks the real transport (`contract --json`, `doctor --json`,
`run - --json` with a subtitle request on stdin → one response document on stdout, workspace taken from the request body,
ffmpeg-skill located only through SUBTITLE_SKILL_FFMPEG_SKILL_DIR) with canned data, enforces the request shape, forbidden keys
and the workspace-relative path policy like the real Skill, writes a real SRT / WebVTT text file for generate and a small
self-describing media file (the format tests/fake_adapter.py probes) for render, and misbehaves on request via FAKE_SUBTITLE_MODE:

  ok | reused | tool_error | output_missing | hash_mismatch | malformed | empty | two_docs | text | nonzero_ok | wrong_schema | wrong_skill |
  wrong_version | wrong_operation | bad_contract | contract_fail | contract_drift | doctor_fail | timeout | cancelled | validation_error |
  dependency_error | internal_error | unknown_code | cue_count_mismatch | vtt_render_unsupported | warning_observation | no_engine

FAKE_SUBTITLE_CALLS: a file every invocation (subcommand + argv + cwd + VIDEO_* env) is appended to (security tests read it).
Never runs ffmpeg or ffmpeg-skill, never imports subtitle_skill. Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE.parent / "src" / "video_agent" / "tools" / "subtitle" / "contract_0.1.0.json").read_text(encoding="utf-8"))
MODE = os.environ.get("FAKE_SUBTITLE_MODE", "ok")
CALLS_LOG = os.environ.get("FAKE_SUBTITLE_CALLS")
NON_RETRYABLE = set(CONTRACT["errors"]["non_retryable"])
FORBIDDEN = {"command", "argv", "shell", "executable", "filter", "filter_complex", "vf", "af", "env", "api_key"}
CUE_FIELDS = {"id", "start", "end", "text", "speaker", "style", "metadata"}
CONSTRAINT_FIELDS = set(CONTRACT["parameters"]["constraints"])
ENGINE_DIR_ENV = "SUBTITLE_SKILL_FFMPEG_SKILL_DIR"


def log(kind: str, argv) -> None:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"cmd": kind, "argv": list(argv), "cwd": os.getcwd(), "env_video": {k: v for k, v in os.environ.items() if k.startswith("VIDEO_")},
                                 "engine_dir": os.environ.get(ENGINE_DIR_ENV)}) + "\n")


def emit(doc) -> None:
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def err(code: str, message: str, details=None, retryable=None) -> int:
    e = {"code": code, "message": message, "retryable": (code not in NON_RETRYABLE) if retryable is None else retryable}
    if details is not None:
        e["details"] = details
    emit({"status": "error", "error": e})
    return 1


def within(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(root), os.path.normcase(path)]) == os.path.normcase(root)
    except ValueError:
        return False


def scan(obj, where):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in FORBIDDEN:
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


def rel_path(raw) -> str | None:
    if not isinstance(raw, str) or not raw or "\x00" in raw or os.path.isabs(raw) or raw.startswith(("/", "\\")):
        return None
    if any(p == ".." for p in raw.replace("\\", "/").split("/")):
        return None
    return raw


def in_dur(path: str) -> float:
    raw = Path(path).read_bytes()
    if raw.startswith(b'{"fake"'):
        try:
            return float(json.loads(raw.decode()).get("duration") or 16.0)
        except (ValueError, TypeError):
            return 16.0
    return 16.0


def ts(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def render_text(cues, fmt: str) -> str:
    cues = sorted(cues, key=lambda c: (c["start"], c["end"], c["id"]))
    if fmt == "srt":
        return "".join(f"{i}\n{ts(c['start'], ',')} --> {ts(c['end'], ',')}\n{c['text']}\n\n" for i, c in enumerate(cues, 1))
    return "WEBVTT\n\n" + "".join(f"{c['id']}\n{ts(c['start'], '.')} --> {ts(c['end'], '.')}\n{c['text']}\n\n" for c in cues)


def main() -> int:
    argv = sys.argv[1:]
    cmd = argv[0] if argv else ""
    log(cmd, argv)
    if cmd == "contract":
        c = json.loads(json.dumps(CONTRACT))
        if MODE == "wrong_schema":
            c["contract_version"] = "2.0.0"
        if MODE == "wrong_skill":
            c["skill_id"] = "other-skill"
        if MODE == "wrong_version":
            c["version"] = "9.0.0"
        if MODE == "bad_contract":
            c["deterministic"] = False
        if MODE == "contract_fail":
            sys.stderr.write("boom\n")
            return 1
        if MODE == "contract_drift":   # a compatible newer contract: one more generate format (the adapter must notice, never silently keep 0.1.0 expectations)
            c["operations"]["generate"]["formats"] = ["srt", "vtt", "ass"]
            c["capabilities"]["formats"] = ["srt", "vtt", "ass"]
        emit(c)
        return 0
    if cmd == "doctor":
        engine = os.environ.get(ENGINE_DIR_ENV)
        healthy = MODE != "doctor_fail" and bool(engine) and os.path.isdir(engine or "")
        emit({"contract_version": CONTRACT["contract_version"], "skill": "subtitle-skill", "version": CONTRACT["version"], "healthy": healthy,
              "dependencies": {"ffmpeg-skill": {"available": healthy, "resolved_path": engine if healthy else None,
                                                "capabilities": {"checked": healthy, "missing": [], "unknown": [], "detail": None} if healthy else None}},
              "problems": [] if healthy else [{"code": "DEPENDENCY_ERROR", "message": "ffmpeg-skill install not found; 'render' operation is unavailable"}],
              "supported_operations": ["generate", "render"] if healthy else ["generate"], "supported_formats": ["srt", "vtt"], "render_supported_formats": ["srt"]})
        return 0 if healthy else 1
    if cmd != "run":
        sys.stderr.write("unknown command\n")
        return 2
    if argv[1:2] != ["-"] or "--json" not in argv or len(argv) != 3:
        sys.stderr.write("the real CLI takes exactly `run - --json`\n")
        return 2
    try:
        req = json.loads(sys.stdin.read())
    except ValueError:
        err("INVALID_REQUEST", "request is not valid JSON")
        return 2
    if MODE == "timeout":
        time.sleep(30)
    if MODE == "malformed":
        sys.stdout.write("{not json\n")
        return 0
    if MODE == "empty":
        return 0
    if MODE == "text":
        sys.stdout.write("done, no JSON here\n")
        return 1
    if MODE == "internal_error":
        return err("INTERNAL_ERROR", "TypeError: bug", retryable=False)   # the real CLI hard-codes retryable: false on an unexpected exception
    if MODE == "unknown_code":
        emit({"status": "error", "error": {"code": "WEIRD", "message": "?", "retryable": True}})
        return 1
    # ---- request validation like the real Skill (security screen, operation, format, workspace, paths, typed document)
    hit = scan(req, "request")
    if hit:
        return err("INVALID_REQUEST", f"forbidden field '{hit}' is not allowed in a subtitle-skill request")
    if not isinstance(req, dict):
        return err("INVALID_REQUEST", "request must be a JSON object")
    op = req.get("operation")
    if op not in ("generate", "render"):
        return err("UNSUPPORTED_OPERATION", f"unsupported operation: {op!r}")
    fmt = req.get("format")
    if fmt not in CONTRACT["operations"]["generate"]["formats"]:
        return err("UNSUPPORTED_FORMAT", f"unsupported format: {fmt!r}")
    ws = req.get("workspace")
    if not isinstance(ws, str) or not ws or not os.path.isabs(ws):
        return err("INVALID_REQUEST", "request.workspace (absolute path) is required")
    ws = os.path.realpath(ws)
    if not os.path.isdir(ws):
        return err("PATH_NOT_ALLOWED", "workspace is not a directory")
    out_rel = rel_path(req.get("output_path"))
    if not req.get("output_path"):
        return err("MISSING_INPUT", "request.output_path is required")
    if out_rel is None:
        return err("PATH_NOT_ALLOWED", f"absolute paths / traversal are not allowed: {req.get('output_path')!r}")
    doc = req.get("subtitle")
    if doc is None:
        return err("MISSING_INPUT", "request.subtitle is required")
    if not isinstance(doc, dict) or set(doc) - {"id", "version", "language", "cues", "metadata"}:
        return err("INVALID_INPUT", "unknown document field(s)")
    for k in ("id", "language", "cues"):
        if k not in doc:
            return err("MISSING_INPUT", f"subtitle document missing field: {k}")
    cues = doc["cues"]
    if not isinstance(cues, list) or not cues:
        return err("MISSING_INPUT", "document.cues must be a non-empty array")
    seen = set()
    for c in cues:
        if not isinstance(c, dict) or set(c) - CUE_FIELDS:
            return err("INVALID_INPUT", "cue: unknown field(s)")
        for k in ("id", "start", "end", "text"):
            if k not in c:
                return err("MISSING_INPUT", f"cue missing required field: {k}")
        if not isinstance(c["start"], (int, float)) or not isinstance(c["end"], (int, float)):
            return err("INVALID_TIME_RANGE", f"cue {c['id']}: start/end must be numeric")
        if c["start"] < 0 or c["end"] <= c["start"]:
            return err("INVALID_TIME_RANGE", f"cue {c['id']}: end <= start")
        if not isinstance(c["text"], str):
            return err("INVALID_INPUT", f"cue {c['id']}: text must be a string")
        if c["id"] in seen:
            return err("VALIDATION_ERROR", f"duplicate cue id: {c['id']}")
        seen.add(c["id"])
    constraints = req.get("constraints")
    if constraints is not None and (not isinstance(constraints, dict) or set(constraints) - CONSTRAINT_FIELDS):
        return err("INVALID_INPUT", "unknown constraint field(s)")
    vd = req.get("video_duration")
    if vd is not None and (not isinstance(vd, (int, float)) or vd <= 0):
        return err("INVALID_INPUT", "video_duration must be a positive number")
    if MODE == "validation_error" or any(c["text"].strip() == "" for c in cues):
        return err("VALIDATION_ERROR", "subtitle document has fatal validation issues", {"issues": [{"severity": "error", "code": "EMPTY_TEXT", "cue_id": cues[0]["id"]}]})
    issues = [{"severity": "warning", "code": "TOO_MANY_LINES", "message": "3 lines exceeds max_lines=2", "cue_id": cues[0]["id"]}] if MODE == "warning_observation" else []
    for c in cues:
        if vd is not None and c["end"] > vd:
            issues.append({"severity": "warning", "code": "CUE_BEYOND_VIDEO_END", "message": f"cue ends after video_duration={vd}", "cue_id": c["id"]})
    out_path = os.path.realpath(os.path.join(ws, out_rel))
    if not within(ws, out_path):
        return err("PATH_NOT_ALLOWED", "path escapes workspace root")
    engine = None
    if op == "render":
        if fmt != "srt":
            return err("UNSUPPORTED_FORMAT", f"render only supports format='srt' (ffmpeg-skill/caption burns SRT or ASS, never {fmt!r})")
        vi = req.get("video_input")
        if not vi:
            return err("MISSING_INPUT", "request.video_input is required for the render operation")
        vi_rel = rel_path(vi)
        if vi_rel is None:
            return err("PATH_NOT_ALLOWED", f"absolute paths / traversal are not allowed: {vi!r}")
        video = os.path.realpath(os.path.join(ws, vi_rel))
        if not os.path.isfile(video):
            return err("MISSING_INPUT", f"input not found: {vi!r}")
        if not within(ws, video):
            return err("PATH_NOT_ALLOWED", "path escapes workspace root")
        engine_dir = os.environ.get(ENGINE_DIR_ENV)
        if MODE == "dependency_error" or not engine_dir or not os.path.isdir(engine_dir):
            return err("DEPENDENCY_ERROR", f"ffmpeg-skill install not found (checked {ENGINE_DIR_ENV})")
        if MODE == "tool_error":
            return err("TOOL_ERROR", "ffmpeg-skill/caption failed (ffmpeg): fake encoder failure")
        engine = ("ffmpeg-skill", "0.9.1-fake")
        payload = json.dumps({"fake": True, "duration": in_dur(video), "video": True, "channels": 2, "captions": True, "cues": len(cues)}).encode()
    else:
        if MODE == "tool_error":
            return err("OUTPUT_ERROR", "output file is missing or empty after execution")
        payload = render_text(cues, fmt).encode("utf-8")
    if MODE == "cancelled":
        return err("CANCELLED", "interrupted")
    reused = MODE == "reused" and os.path.isfile(out_path)
    if MODE != "output_missing":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Path(out_path).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if MODE == "hash_mismatch":
        digest = "0" * 64
    resp = {"status": "ok", "skill": "other-skill" if MODE == "wrong_skill" else "subtitle-skill", "skill_version": "9.9.9" if MODE == "wrong_version" else CONTRACT["version"],
            "contract_version": CONTRACT["contract_version"], "operation": ("generate" if op == "render" else "render") if MODE == "wrong_operation" else op,
            "output": out_path, "sha256": digest, "size": len(payload), "reused": reused, "observation": issues,
            "timeline": {"cue_count": len(cues) + 1 if MODE == "cue_count_mismatch" else len(cues)}, "duration_ms": 1.0}
    if engine and MODE != "no_engine":
        resp["engine"], resp["engine_version"] = engine
    emit(resp)
    if MODE == "two_docs":
        emit(resp)
    return 3 if MODE == "nonzero_ok" else 0


if __name__ == "__main__":
    sys.exit(main())
