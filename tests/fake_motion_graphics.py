"""Fake motion-graphics-skill process for adapter boundary tests: speaks the real transport (`skill --json`, `doctor --json`,
`plan|run - --json --workspace D --allowed-input R… [--ffmpeg-skill X] [--timeout S] [--dry-run]` with a graphics request on
stdin → one response document on stdout) with canned data, enforces the request shape, the element schemas and the path policy
like the real Skill, writes a small self-describing output file (the format tests/fake_adapter.py probes), and misbehaves on
request via FAKE_MG_MODE:

  ok | reused | tool_error | tool_error_final | output_missing | validation_error | cancelled | skill_timeout | timeout (hangs ~30 s) |
  hang | malformed | empty | two_docs | text | nonzero_ok | wrong_schema | wrong_skill | wrong_version | bad_contract | contract_fail |
  contract_drift | hash_mismatch | unknown_code | internal_error | doctor_fail | doctor_degraded | no_provenance |
  duplicate_ids (DEPENDENCY_ERROR) | timeline_mismatch (the timeline omits an element) | unknown_font (MISSING_INPUT)

FAKE_MG_CALLS: a file every invocation (subcommand + argv) is appended to (security tests read it). Never runs ffmpeg or
ffmpeg-skill, never imports motion_graphics. Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE.parent / "src" / "video_agent" / "tools" / "motion_graphics" / "contract_0.1.0.json").read_text(encoding="utf-8"))
MODE = os.environ.get("FAKE_MG_MODE", "ok")
CALLS_LOG = os.environ.get("FAKE_MG_CALLS")
EXIT = CONTRACT["errors"]["exit_codes"]
RETRY = CONTRACT["errors"]["retryable"]
TYPES = {o["type"]: o for o in CONTRACT["element_types"]}
UNSUPPORTED = {u["type"] for u in CONTRACT["unsupported_element_types"]}
ANIMS = {a["kind"]: a for a in CONTRACT["animations"]}
FONTS = CONTRACT["fonts"]["registry"]
FORBIDDEN = set(CONTRACT["request"]["forbidden_fields"])
ID_RE = re.compile(CONTRACT["request"]["id_pattern"])
COLOR_RE = re.compile(r"^(#?[0-9A-Fa-f]{6}|[A-Za-z]+)(@(0(\.\d+)?|1(\.0+)?))?$")
RESPONSE = "motion-graphics/response@1"
ENGINE = {"ffmpeg-skill": "0.9.1-fake", "ffmpeg-skill_contract": "1.0", "ffmpeg": "fake", "ffprobe": "fake"}


def log(kind: str, argv) -> None:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"cmd": kind, "argv": list(argv), "cwd": os.getcwd(), "env_video": {k: v for k, v in os.environ.items() if k.startswith("VIDEO_")}}) + "\n")


def emit(doc) -> None:
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def err(code: str, message: str, details=None, retryable=None, dry_run=False) -> int:
    emit({"schema": RESPONSE, "skill": {"id": "motion-graphics", "version": CONTRACT["version"]}, "ok": False, "status": "cancelled" if code == "CANCELLED" else "error",
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


def probe(path: str):
    raw = Path(path).read_bytes()
    if raw.startswith(b'{"fake"'):
        try:
            d = json.loads(raw.decode())
            return float(d.get("duration") or 16.0), int(d.get("width") or 640), int(d.get("height") or 360)
        except (ValueError, TypeError):
            pass
    return 16.0, 640, 360


def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_param(name, raw, spec, where):
    t = spec["type"]
    if t == "string":
        if not isinstance(raw, str) or (spec.get("max_length") and len(raw) > spec["max_length"]):
            return err("INVALID_REQUEST", f"{where}.{name} must be a string (max {spec.get('max_length')})", {"field": f"{where}.{name}"})
    elif t == "color":
        if not isinstance(raw, str) or not COLOR_RE.match(raw):
            return err("INVALID_REQUEST", f"{where}.{name} must be a named color or RRGGBB hex", {"field": f"{where}.{name}"})
    elif t == "position":
        if isinstance(raw, str):
            if raw not in CONTRACT["positions"]["named"]:
                return err("INVALID_REQUEST", f"{where}.{name} unknown position", {"field": f"{where}.{name}"})
        elif not (isinstance(raw, dict) and set(raw) == {"x", "y"} and all(isinstance(raw[a], int) and not isinstance(raw[a], bool) for a in ("x", "y"))):
            return err("INVALID_REQUEST", f"{where}.{name} must be a name or {{x,y}}", {"field": f"{where}.{name}"})
    elif t == "font":
        if not isinstance(raw, dict) or set(raw) not in ({"font_id"}, {"font_file"}):
            return err("INVALID_REQUEST", f"{where}.{name} must be {{font_id}} or {{font_file}}", {"field": f"{where}.{name}"})
    elif t == "path":
        if not isinstance(raw, str) or not raw:
            return err("INVALID_REQUEST", f"{where}.{name} must be a path string", {"field": f"{where}.{name}"})
    elif t == "boolean":
        if not isinstance(raw, bool):
            return err("INVALID_REQUEST", f"{where}.{name} must be a boolean", {"field": f"{where}.{name}"})
    else:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or (t == "integer" and raw != int(raw)):
            return err("INVALID_REQUEST", f"{where}.{name} must be a {t}", {"field": f"{where}.{name}"})
        if ("min" in spec and raw < spec["min"]) or ("max" in spec and raw > spec["max"]):
            return err("INVALID_REQUEST", f"{where}.{name} out of range", {"field": f"{where}.{name}"})
    return None


def main() -> int:
    argv = sys.argv[1:]
    cmd = argv[0] if argv else ""
    log(cmd, argv)
    if cmd in ("contract", "skill"):
        c = json.loads(json.dumps(CONTRACT))
        if MODE == "wrong_schema":
            c["schema"] = "motion-graphics/contract@2"
        if MODE == "wrong_skill":
            c["skill_id"] = "other-skill"; c["id"] = "other-skill"
        if MODE == "wrong_version":
            c["version"] = "9.0.0"; c["tools"][0]["version"] = "9.0.0"
        if MODE == "bad_contract":
            c["execution"]["shell"] = True
        if MODE == "contract_fail":
            sys.stderr.write("boom\n")
            return 1
        if MODE == "contract_drift":   # a compatible newer contract: one more element type (the adapter must notice, never silently keep 0.1.0 expectations)
            c["element_types"].append(dict(c["element_types"][0], type="watermark", parameters={"title": {"type": "string", "required": True, "max_length": 200}}))
            c["unsupported_element_types"] = [u for u in c["unsupported_element_types"] if u["type"] != "watermark"]
            c["tools"][0]["element_types"] = sorted(c["tools"][0]["element_types"] + ["watermark"])
        emit(c)
        return 0
    if cmd == "doctor":
        status = {"doctor_fail": "fail", "doctor_degraded": "degraded"}.get(MODE, "ok")
        ok = status != "fail"
        types = {}
        for t, spec in TYPES.items():
            st = "supported" if ok else "unavailable"
            if MODE == "doctor_degraded" and t == "image_overlay":
                st = "unavailable"
            if t in ("title", "lower_third", "image_overlay") and ok and st == "supported":
                st = "unknown"   # like the real Skill: ffmpeg-skill's doctor does not classify drawbox / overlay / color / scale
            types[t] = {"status": st, "tool": spec["tool"], "required_capabilities": spec["required_capabilities"], "missing": [] if st != "unavailable" else ["ffmpeg-skill"], "unknown": []}
        emit({"schema": "motion-graphics/doctor@1", "skill": {"id": "motion-graphics", "version": CONTRACT["version"]}, "status": status,
              "checks": {"python": {"status": "ok"}, "ffmpeg_skill": {"status": "ok" if ok else "missing", "directory": (flag_values(argv, "--ffmpeg-skill") or [None])[0], "version": "0.9.1-fake", "contract_version": "1.0"},
                         "ffmpeg": {"status": "ok" if ok else "missing", "version": "fake"}, "ffprobe": {"status": "ok" if ok else "missing", "version": "fake"},
                         "capabilities": {}, "filter_detection": {"status": "ok"}, "element_types": types, "unsupported_element_types": {u: "" for u in UNSUPPORTED},
                         "animations": {"fade": {"status": "unknown", "applies_to": ANIMS["fade"]["applies_to"]}}, "unsupported_animations": {},
                         "fonts": {"detection": "fake", "fonts": {f: ("supported" if ok else "unavailable") for f in FONTS}},
                         "path_policy": {"status": "ok", "workspace": (flag_values(argv, "--workspace") or [None])[0], "allowed_input_roots": flag_values(argv, "--allowed-input")}},
              "unavailable_element_types": [t for t, o in types.items() if o["status"] == "unavailable"], "problems": [] if ok else ["ffmpeg-skill: not found"], "warnings": [], "secrets_shown": False})
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
    # ---- request validation (shape, forbidden fields, elements, parameters, paths), like the real Skill
    hit = scan(req, "document")
    if hit:
        return err("INVALID_REQUEST", f"forbidden field {hit}", {"field": hit})
    if not isinstance(req, dict) or req.get("schema") != CONTRACT["request"]["schema"] or set(req) - {"schema", "video", "output", "elements", "options"} or not {"video", "output", "elements"} <= set(req):
        return err("INVALID_REQUEST", "unsupported request shape / schema", {"field": "schema"})
    if set(req["video"]) != {"path"} or not set(req["output"]) <= {"path", "overwrite"} or "path" not in req["output"]:
        return err("INVALID_REQUEST", "video / output shape")
    elements = req["elements"]
    if not isinstance(elements, list) or not elements:
        return err("INVALID_REQUEST", "elements must be a non-empty array", {"field": "elements"})
    if len(elements) > CONTRACT["limits"]["max_elements"]:
        return err("INVALID_REQUEST", "too many elements", {"field": "elements"})
    opts = req.get("options") or {}
    if set(opts) - {"reuse_intermediates", "crf", "preset"}:
        return err("INVALID_REQUEST", "unknown option")
    crf = opts.get("crf", 18)
    if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= 51:
        return err("INVALID_REQUEST", "crf out of range", {"field": "document.options.crf"})

    def resolve_input(p: str, what: str):
        resolved = os.path.realpath(p if os.path.isabs(p) else os.path.join(ws, p))
        if roots and not any(within(r, resolved) for r in roots):
            return None, err("PATH_NOT_ALLOWED", f"{what} outside allowed roots", {"reason": "outside_allowed_roots"})
        if not os.path.isfile(resolved):
            return None, err("INVALID_INPUT", f"{what} not found", {"reason": "not_found"})
        return resolved, None

    video, rc = resolve_input(req["video"]["path"], "document.video.path")
    if rc is not None:
        return rc
    seen = {}
    parsed = []
    assets, fonts = {}, {}
    for i, el in enumerate(elements):
        where = f"elements[{i}]"
        if not isinstance(el, dict) or set(el) - {"id", "type", "start", "end", "parameters", "animation"} or not {"id", "type", "start", "end", "parameters"} <= set(el):
            return err("INVALID_REQUEST", f"{where}: element shape", {"field": where})
        if not isinstance(el["id"], str) or not ID_RE.match(el["id"]):
            return err("INVALID_REQUEST", f"{where}.id", {"field": f"{where}.id"})
        t = el["type"]
        if t in UNSUPPORTED:
            return err("UNSUPPORTED_OPERATION", f"{where}.type {t!r} is not implemented", {"type": t})
        if t not in TYPES:
            return err("UNSUPPORTED_OPERATION", f"{where}.type {t!r} is unknown", {"type": t})
        s, e = el["start"], el["end"]
        for v in (s, e):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
                return err("INVALID_REQUEST", f"{where}.start/end must be finite numbers")
        if s < 0 or e <= s:
            return err("INVALID_TIME_RANGE", f"{where}: needs 0 <= start < end", {"field": f"{where}.end"})
        schema = TYPES[t]["parameters"]
        params = el["parameters"]
        if not isinstance(params, dict) or set(params) - set(schema):
            return err("INVALID_REQUEST", f"{where}: unknown parameters", {"field": f"{where}.parameters"})
        for name, spec in schema.items():
            if spec.get("required") and name not in params:
                return err("INVALID_REQUEST", f"{where}: {t} needs {name}", {"field": f"{where}.parameters.{name}"})
            if name in params:
                rc = check_param(name, params[name], spec, f"{where}.parameters")
                if rc is not None:
                    return rc
        full = {name: spec["default"] for name, spec in schema.items() if "default" in spec}
        full.update(params)
        if "font" in full:
            f = full["font"]
            if "font_id" in f and (f["font_id"] not in FONTS or MODE == "unknown_font"):
                return err("MISSING_INPUT", f"{where}.parameters.font: unknown font_id {f['font_id']!r}", {"field": f"{where}.parameters.font", "font_id": f["font_id"]})
            if "font_file" in f:
                ff, rc = resolve_input(f["font_file"], f"{where}.parameters.font.font_file")
                if rc is not None:
                    return rc
                fonts[el["id"]] = {"kind": "file", "font_file_hash": sha(ff)}
            else:
                fonts[el["id"]] = {"kind": "system", "font_id": f["font_id"], "font_name": FONTS[f["font_id"]]}
        elif "font" in schema:
            fid = CONTRACT["fonts"]["default_font_id"]
            if MODE == "unknown_font":
                return err("MISSING_INPUT", f"{where}.parameters.font: unknown font_id {fid!r}", {"field": f"{where}.parameters.font", "font_id": fid})
            fonts[el["id"]] = {"kind": "system", "font_id": fid, "font_name": FONTS[fid]}
        if t == "image_overlay":
            img, rc = resolve_input(full["image_path"], f"{where}.parameters.image_path")
            if rc is not None:
                return rc
            if Path(img).suffix.lower() not in CONTRACT["image_formats"]["allowed_extensions"]:
                return err("UNSUPPORTED_FORMAT", f"{where}.parameters.image_path extension", {"element_id": el["id"]})
            assets[el["id"]] = {"path": img, "sha256": sha(img), "size": os.path.getsize(img)}
            full["image_path"] = {"sha256": assets[el["id"]]["sha256"], "size": assets[el["id"]]["size"]}
        anim = None
        if "animation" in el and el["animation"] is not None:
            a = el["animation"]
            if not isinstance(a, dict) or set(a) != {"kind", "parameters"} or a["kind"] not in ANIMS:
                return err("UNSUPPORTED_OPERATION", f"{where}.animation.kind", {"field": f"{where}.animation"})
            if t not in ANIMS[a["kind"]]["applies_to"]:
                return err("INVALID_REQUEST", f"{where}.animation does not apply to {t}", {"field": f"{where}.animation"})
            aschema = ANIMS[a["kind"]]["parameters"]
            if not isinstance(a["parameters"], dict) or set(a["parameters"]) - set(aschema):
                return err("INVALID_REQUEST", f"{where}.animation.parameters", {"field": f"{where}.animation"})
            for name, spec in aschema.items():
                if spec.get("required") and name not in a["parameters"]:
                    return err("INVALID_REQUEST", f"{where}.animation needs {name}")
                rc = check_param(name, a["parameters"][name], spec, f"{where}.animation.parameters")
                if rc is not None:
                    return rc
            anim = a
        if el["id"] in seen or MODE == "duplicate_ids":
            return err("DEPENDENCY_ERROR", f"duplicate element id {el['id']!r}", {"field": "elements", "id": el["id"]})
        seen[el["id"]] = i
        parsed.append({"id": el["id"], "type": t, "start": float(s), "end": float(e), "parameters": full, "animation": anim})
    parsed.sort(key=lambda x: (x["start"], x["id"]))
    raw_out = req["output"]["path"]
    if any(part == ".." for part in Path(raw_out).parts):
        return err("PATH_NOT_ALLOWED", "traversal", {"reason": "traversal"})
    out_path = os.path.realpath(os.path.join(ws, raw_out)) if not os.path.isabs(raw_out) else os.path.realpath(raw_out)
    if not within(ws, out_path):
        return err("PATH_NOT_ALLOWED", "output must be inside the workspace", {"reason": "outside_workspace"})
    if os.path.normcase(out_path) == os.path.normcase(video):
        return err("OUTPUT_ERROR", "output may not be the input", {"reason": "output_is_input"})
    if os.path.exists(out_path) and not req["output"].get("overwrite"):
        return err("OUTPUT_ERROR", "output already exists", {"reason": "exists"})
    dur, width, height = probe(video)
    video_sha = sha(video)
    doc_id = hashlib.sha256(json.dumps({"video": video_sha, "elements": parsed}, sort_keys=True, default=str).encode()).hexdigest()[:16]
    base = {"schema": RESPONSE, "skill": {"id": "motion-graphics", "version": CONTRACT["version"]}, "warnings": []}
    video_meta = {"path": video, "sha256": video_sha, "duration": dur, "width": width, "height": height}
    if dry_run:
        tl = [dict(p, index=i, tool=TYPES[p["type"]]["tool"], **({"font": fonts[p["id"]]} if p["id"] in fonts else {})) for i, p in enumerate(parsed)]
        emit({**base, "ok": True, "status": "ok", "dry_run": True, "plan": {"document_id": doc_id, "video": video_meta, "output": {"path": out_path}, "timeline": tl}})
        return 0
    if MODE == "cancelled":
        return err("CANCELLED", "interrupted", {"reason": "signal"})
    if MODE == "skill_timeout":
        return err("CANCELLED", "operation exceeded timeout", {"reason": "timeout"})
    if MODE in ("tool_error", "tool_error_final"):
        return err("TOOL_ERROR", "ffmpeg-skill/overlay failed", {"reason": "tool_failed", "element_id": parsed[0]["id"]}, retryable=MODE == "tool_error")
    if MODE == "validation_error":
        return err("VALIDATION_ERROR", "output duration 2.0 differs from the source 3.0 by more than 0.25 s", {"reason": "duration_mismatch"})
    # ---- write the output (bytes derived from the request so identical requests give identical hashes)
    payload = json.dumps({"fake": True, "duration": dur, "video": True, "channels": 2, "graphics": [p["id"] for p in parsed], "width": width, "height": height, "doc": doc_id}).encode()
    reused = MODE == "reused" and os.path.isfile(out_path)
    if MODE != "output_missing":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Path(out_path).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if MODE == "hash_mismatch":
        digest = "0" * 64
    ops = []
    prev = video_sha
    for i, p in enumerate(parsed):
        op_id = hashlib.sha256(json.dumps({"type": p["type"], "params": p["parameters"], "start": p["start"], "end": p["end"], "animation": p["animation"], "prev": prev, "tool_versions": ENGINE}, sort_keys=True, default=str).encode()).hexdigest()
        out_hash = digest if i == len(parsed) - 1 else hashlib.sha256((op_id + "stage").encode()).hexdigest()
        rec = {"index": i, "element_id": p["id"], "type": p["type"], "tool": TYPES[p["type"]]["tool"], "status": "reused" if (reused or (MODE == "reused" and i < len(parsed) - 1)) else "rendered",
               "operation_id": op_id, "parameters": dict(p["parameters"], **({"font": fonts[p["id"]]} if p["id"] in fonts else {})), "input_hashes": [prev], "output_hash": out_hash, "seconds": 0.01,
               "tool_commands_observed": [f"{TYPES[p['type']]['tool']} {op_id[:16]} (fake, provenance only)"]}
        ops.append(rec)
        prev = op_id
    timeline = [{"id": p["id"], "type": p["type"], "start": p["start"], "end": p["end"]} for p in parsed]
    if MODE == "timeline_mismatch":
        timeline = timeline[1:]
    doc = {**base, "ok": True, "status": "ok", "dry_run": False,
           "output": {"path": out_path, "sha256": digest, "size": len(payload), "duration": dur, "width": width, "height": height},
           "timeline": timeline, "operations": ops, "reused": bool(reused), "engine": dict(ENGINE),
           "provenance": {"document_id": doc_id, "video": video_meta, "assets": assets, "fonts": fonts, "operations": ops, "output_hash": digest}}
    if MODE == "no_provenance":
        doc.pop("provenance")
    emit(doc)
    if MODE == "two_docs":
        emit(doc)
    return 3 if MODE == "nonzero_ok" else 0


if __name__ == "__main__":
    sys.exit(main())
