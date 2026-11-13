"""Fake qc-skill process for adapter boundary tests: speaks the real transport (`contract --json`, `doctor --json [--workspace D]`,
`run - --json --workspace D --allowed-input-root R… [--no-cache]` with a qc request on stdin → one response document on stdout) with
canned data, enforces the request shape, the forbidden keys and the path policy like the real Skill, measures nothing (an input that
starts with b'{"fake"' is parsed for duration / video / channels / lufs, anything else is stat only), writes NO media, and misbehaves on
request via FAKE_QC_MODE:

  ok | reused | tool_error | tool_error_final | output_missing | no_report | hash_mismatch | fingerprint_mismatch | malformed | empty |
  two_docs | text | nonzero_ok | wrong_schema | wrong_skill | wrong_version | bad_contract | contract_fail | contract_drift |
  doctor_fail | doctor_degraded | timeout | hang | cancelled | validation_error | internal_error | unknown_code |
  verdict_fail | verdict_warn | wrong_kind | wrong_operation | not_observed | status_failed_ok | bad_status | bad_report_id |
  package_fail (kind delivery_package: the last member's extension does not match its rule) | package_fingerprint (a member's
  fingerprint is not its file's)

FAKE_QC_CALLS: a file every invocation (subcommand + argv) is appended to (security tests read it). FAKE_QC_REQUESTS: a file every run
request document is appended to. Never runs ffprobe / ffmpeg, never imports qc_skill. Test double only.

The opt-in stinger (1.2.0) and loop-seam (1.3.0) checks run only when their rule is in the request, like the real Skill: the cover check from
the rule itself, the opacity from FAKE_QC_STINGER (ok | not_opaque | unknown), the cut-frame difference from FAKE_QC_CUT_DIFF (default 0.0,
judged only with max_cut_frame_difference), the seam facts from FAKE_QC_SEAM_CLICK / _LEVEL / _BAND (defaults 1.0 / 0.3 / 1.0) or
FAKE_QC_SEAM=unknown. A stinger `file` is resolved like an input (MISSING_INPUT / PATH_NOT_ALLOWED) and fingerprinted (provenance.stinger_input)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE.parent / "src" / "video_agent" / "tools" / "qc" / "contract_1.4.0.json").read_text(encoding="utf-8"))
MODE = os.environ.get("FAKE_QC_MODE", "ok")
CALLS_LOG = os.environ.get("FAKE_QC_CALLS")
EXIT = CONTRACT["errors"]["exit_codes"]
RETRY = {"TOOL_ERROR": True, "CANCELLED": True}   # the Skill's errors.py verdicts (the contract carries no retryable map)
FORBIDDEN = {"command", "commands", "argv", "args", "shell", "cmd", "cmdline", "exec", "executable", "filter", "filter_complex", "env", "environment"}
TOP_LEVEL = {"schema", "request_id", "operation", "kind", "input", "subtitle", "reference_video", "parameters", "rules", "cache_policy", "timeout", "artifacts"}
RESPONSE = "qc/response@1"
VERSION = CONTRACT["version"]


def log(kind: str, argv) -> None:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"cmd": kind, "argv": list(argv), "cwd": os.getcwd(), "env_video": {k: v for k, v in os.environ.items() if k.startswith("VIDEO_")}}) + "\n")


def emit(doc) -> None:
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def err(code: str, message: str, details=None, retryable=None) -> int:
    emit({"schema": RESPONSE, "status": "failed", "skill": {"id": "qc", "version": VERSION},
          "error": {"code": code, "message": message, "retryable": RETRY.get(code, False) if retryable is None else retryable, "details": details or {}}})
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


def facts(path: str) -> dict:
    raw = Path(path).read_bytes()
    out = {"duration": 3.0, "video": True, "channels": 2, "lufs": -16.0, "size": len(raw)}
    if raw.startswith(b'{"fake"'):
        try:
            d = json.loads(raw.decode())
            out.update({k: d[k] for k in ("duration", "video", "channels", "lufs") if k in d})
        except (ValueError, TypeError):
            pass
    return out


def check_rules(section: str, data, where: str):
    schema = CONTRACT["rules"].get(section)
    if schema is None:
        return err("INVALID_REQUEST", f"unknown rule sections: ['{section}']")
    if not isinstance(data, dict):
        return err("INVALID_REQUEST", f"rule payload for {where} must be an object")
    unknown = sorted(set(data) - set(schema))
    if unknown:
        return err("INVALID_REQUEST", f"unknown fields for {where}: {unknown}")
    for k, v in data.items():
        nested = schema[k].get("nested_rule")
        if nested and v is not None:
            for i, item in enumerate(v if str(schema[k].get("type", "")).startswith(("List[", "Optional[List[")) else [v]):
                bad = check_rules(nested, item, f"{where}.{k}" + (f"[{i}]" if isinstance(v, list) else ""))
                if bad:
                    return bad
    return None


def resolve(raw, roots, what):
    if not isinstance(raw, str) or not raw:
        return None, err("MISSING_INPUT", f"request.{what} must be a non-empty string")
    if any(part == ".." for part in Path(raw).parts):
        return None, err("PATH_NOT_ALLOWED", "path traversal ('..') is not allowed", {"path": raw})
    p = os.path.realpath(raw)
    if not os.path.exists(p):
        return None, err("MISSING_INPUT", "input file does not exist", {"path": raw})
    if not os.path.isfile(p):
        return None, err("INVALID_INPUT", "input path is not a regular file", {"path": raw})
    if roots and not any(within(r, p) for r in roots):
        return None, err("PATH_NOT_ALLOWED", "input is outside the allowed input roots", {"path": raw})
    return p, None


def measurement(mid: str, value, unit=None, source="ffprobe"):
    cat, name = mid.split(".", 1)
    m = {"id": mid, "category": cat, "name": name, "value": value, "source": source, "estimated": False}
    if unit:
        m["unit"] = unit
    return m


def main() -> int:
    argv = sys.argv[1:]
    cmd = argv[0] if argv else ""
    log(cmd, argv)
    if cmd == "contract":
        c = json.loads(json.dumps(CONTRACT))
        if MODE == "wrong_schema":
            c["schema"] = "qc/contract@2"
        if MODE == "wrong_skill":
            c["skill_id"] = "other-skill"; c["name"] = "other-skill"
        if MODE == "wrong_version":
            c["version"] = "9.0.0"
        if MODE == "bad_contract":
            c["execution"]["shell"] = True
        if MODE == "contract_fail":
            sys.stderr.write("boom\n")
            return 1
        if MODE == "contract_drift":   # a compatible newer contract: one more finding and one more parameter (the adapter must notice)
            c["findings"].append({"category": "video", "code": "VIDEO_HDR_MISMATCH", "default_severity": "WARN"})
            c["parameters"].append("hdr_probe_frames")
        emit(c)
        return 0
    if cmd == "doctor":
        status = {"doctor_fail": "fail", "doctor_degraded": "degraded"}.get(MODE, "ok")
        ok = status != "fail"
        checks = {"contract": {"checks": len(CONTRACT["checks"]), "schema": CONTRACT["schema"], "status": "AVAILABLE"},
                  "ffprobe": {"path": "/usr/bin/ffprobe", "status": "AVAILABLE" if ok else "MISSING", "version": "fake"},
                  "ffmpeg": {"path": "/usr/bin/ffmpeg", "status": "AVAILABLE" if status == "ok" else "MISSING", "version": "fake"},
                  "path_policy": {"status": "AVAILABLE", "workspace": (flag_values(argv, "--workspace") or [os.getcwd()])[0], "workspace_exists": True},
                  "python": {"implementation": "CPython", "status": "AVAILABLE", "version": "fake"}}
        for f in CONTRACT["capabilities"]["optional"]:
            if f.startswith("filter:"):
                checks[f] = {"status": "AVAILABLE" if status == "ok" else "MISSING"}
        emit({"schema": "qc/doctor@1", "skill": {"id": "qc", "version": VERSION}, "status": status, "checks": checks,
              "problems": [] if ok else ["ffprobe: not found"], "unavailable_tools": [] if status == "ok" else ["ffmpeg"]})
        return {"ok": 0, "degraded": 2}.get(status, 1)
    if cmd != "run":
        sys.stderr.write("unknown command\n")
        return 2
    if argv[1:2] != ["-"] or "--json" not in argv:
        return err("INVALID_REQUEST", "request must come from stdin with --json")
    ws = os.path.realpath((flag_values(argv, "--workspace") or [os.getcwd()])[0])
    roots = [os.path.realpath(r) for r in flag_values(argv, "--allowed-input-root")]
    try:
        req = json.loads(sys.stdin.read())
    except ValueError:
        return err("INVALID_REQUEST", "request is not valid JSON")
    if os.environ.get("FAKE_QC_REQUESTS"):
        with open(os.environ["FAKE_QC_REQUESTS"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(req, ensure_ascii=False) + "\n")
    if MODE in ("hang", "timeout"):
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
        return err("INTERNAL_ERROR", "TypeError: bug")
    if MODE == "unknown_code":
        emit({"schema": RESPONSE, "status": "failed", "skill": {"id": "qc", "version": VERSION}, "error": {"code": "WEIRD", "message": "?", "retryable": True, "details": {}}})
        return 1
    # ---- request validation like the real Skill (schemas.py)
    if not isinstance(req, dict):
        return err("INVALID_REQUEST", "request must be a JSON object")
    hit = scan(req, "request")
    if hit:
        return err("INVALID_REQUEST", f"request contains a forbidden key: {hit}")
    if set(req) - TOP_LEVEL:
        return err("INVALID_REQUEST", f"unknown top-level request keys: {sorted(set(req) - TOP_LEVEL)}")
    operation, kind = req.get("operation"), req.get("kind")
    if operation not in CONTRACT["operations"]:
        return err("UNSUPPORTED_OPERATION", "operation must be one of ['check', 'inspect', 'validate']", {"got": operation})
    if kind not in CONTRACT["kinds"]:
        return err("INVALID_REQUEST", "kind must be one of ['audio', 'delivery', 'subtitle', 'video']", {"got": kind})
    params = req.get("parameters", {})
    if not isinstance(params, dict):
        return err("INVALID_REQUEST", "request.parameters must be an object")
    if set(params) - set(CONTRACT["parameters"]):
        return err("INVALID_REQUEST", f"unknown parameter keys: {sorted(set(params) - set(CONTRACT['parameters']))}")
    rules = req.get("rules", {})
    if not isinstance(rules, dict):
        return err("INVALID_REQUEST", "request.rules must be an object")
    for section, data in rules.items():
        bad = check_rules(section, data, section)
        if bad:
            return bad
    if req.get("cache_policy", "use") not in CONTRACT["cache"]["policies"]:
        return err("INVALID_REQUEST", "cache_policy must be one of ['bypass', 'only', 'use']")
    if kind == "delivery_package" or req.get("artifacts") is not None:
        return run_package(req, roots, operation, kind, params, rules, argv)
    src, bad = resolve(req.get("input"), roots, "input")
    if bad:
        return bad
    companions = {}
    vrule = (rules.get("video") if kind == "video" else (rules.get("delivery") or {}).get("video") if kind == "delivery" else None) or {}
    metrics = [m for m in ("vmaf", "ssim", "psnr") if m in (vrule.get("reference_metrics") or [])]   # canonical order, like the Skill
    for k in ("subtitle", "reference_video"):
        if k == "reference_video" and kind != "subtitle" and not metrics:
            continue   # qc-skill ADR-021: kind video / delivery read the reference only when the video rule opts in
        if req.get(k) is not None:
            p, bad = resolve(req[k], roots, k)
            if bad:
                return bad
            companions[k] = p
    stinger = (vrule.get("stinger") or None) if kind in ("video", "delivery") else None
    if stinger and stinger.get("file") is not None:   # qc-skill ADR-024: the stinger is a second input, resolved and fingerprinted like `input`
        p, bad = resolve(stinger["file"], roots, "rules.video.stinger.file")
        if bad:
            return bad
        companions["stinger"] = p
    arule = (rules.get("audio") if kind == "audio" else (rules.get("delivery") or {}).get("audio") if kind == "delivery" else None) or {}
    seam_rule = arule.get("loop_seam") or None
    if MODE == "cancelled":
        return err("CANCELLED", "interrupted", {"reason": "signal"})
    if MODE in ("tool_error", "tool_error_final"):
        return err("TOOL_ERROR", "ffprobe failed", {"reason": "tool_failed", "argv": ["never", "forwarded"]}, retryable=MODE == "tool_error")
    if MODE == "validation_error":
        return err("VALIDATION_ERROR", "cache_policy is 'only' but no cached report was found")
    # ---- canned measurements (nothing is probed)
    f = facts(src)
    digest = hashlib.sha256(Path(src).read_bytes()).hexdigest()
    ms = [measurement("container.format_name", "mov,mp4,m4a,3gp,3g2,mj2"), measurement("container.duration_sec", f["duration"], "sec"), measurement("container.size_bytes", f["size"], "bytes", "OBSERVED")]
    if kind in ("video", "delivery"):
        ms += [measurement("video.stream_present", bool(f["video"])), measurement("video.width", 640, "px"), measurement("video.height", 360, "px"), measurement("video.frame_rate", 25.0, "fps"),
               measurement("video.decode_error_count", 0, None, "ffmpeg:decode")]
    if kind in ("audio", "delivery"):
        ms += [measurement("audio.stream_present", f["channels"] > 0), measurement("audio.channels", f["channels"]), measurement("audio.sample_rate", 48000, "Hz"),
               measurement("audio.integrated_loudness_lufs", f["lufs"], "LUFS", "ffmpeg:ebur128"), measurement("audio.leading_silence_sec", 0.0, "sec", "ffmpeg:silencedetect")]
    ref_ids = {"vmaf": "video.vmaf", "ssim": "video.ssim", "psnr": "video.psnr_db"}
    if metrics:
        ms += [measurement("video.reference_alignment", "aligned" if "reference_video" in companions else "not_performed", None, "ffmpeg")]
        if "reference_video" in companions:
            ms += [measurement(ref_ids[m], {"vmaf": 96.5, "ssim": 0.99, "psnr": 42.0}[m], "dB" if m == "psnr" else None, f"ffmpeg:{'libvmaf' if m == 'vmaf' else m}") for m in metrics]
    if kind == "subtitle" or "subtitle" in companions:
        ms += [measurement("subtitle.exists", True, None, "OBSERVED"), measurement("subtitle.format", "srt", None, "parser"), measurement("subtitle.cue_count", 2, None, "parser")]
    if kind == "delivery":
        ms += [measurement("delivery.extension", Path(src).suffix.lstrip("."), None, "OBSERVED")]
    prefixes = {"video": ("video.",), "audio": ("audio.",), "subtitle": ("subtitle.",), "delivery": ("video.", "audio.", "subtitle.", "delivery.")}[kind]
    checks, findings = [], []
    opt_in = {"video.vmaf_meets_minimum", "video.ssim_meets_minimum", "video.psnr_meets_minimum"} | {c for c in CONTRACT["checks"] if c.startswith(("video.stinger_", "audio.loop_seam_"))}
    if operation in ("check", "validate"):
        opted = {f"video.{m}_meets_minimum" for m in metrics}
        for cid in CONTRACT["checks"]:
            if cid.startswith(prefixes) and (cid not in opt_in or cid in opted):
                checks.append({"check_id": cid, "category": cid.split(".")[0], "status": "PASS", "measurement_ids": [m["id"] for m in ms if m["category"] == cid.split(".")[0]][:2], "finding_codes": [], "evidence": {}})
        if stinger:
            stinger_checks(stinger, checks, findings, ms)
        if seam_rule:
            seam_checks(seam_rule, f["duration"], checks, findings, ms)
        if MODE == "verdict_fail" or (kind in ("video", "delivery") and not f["video"]):
            checks[0].update(status="FAIL", finding_codes=["VIDEO_STREAM_MISSING"])
            findings.append({"code": "VIDEO_STREAM_MISSING", "category": "video", "severity": "FAIL", "message": "no video stream", "check_id": checks[0]["check_id"], "argv": ["scrubbed"]})
        if MODE == "verdict_warn":
            c = next((c for c in checks if c["category"] == "audio"), checks[0])
            c.update(status="WARN", finding_codes=["AUDIO_LEADING_SILENCE_EXCEEDED"])
            findings.append({"code": "AUDIO_LEADING_SILENCE_EXCEEDED", "category": "audio", "severity": "WARN", "message": "leading silence 2.5 s > 1.0 s", "check_id": c["check_id"]})
    overall = "FAIL" if any(c["status"] == "FAIL" for c in checks) else "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS"
    if MODE == "bad_status":
        overall = "MAYBE"
    identity = hashlib.sha256(json.dumps({"fp": digest, "kind": kind, "op": operation, "params": params, "rules": rules}, sort_keys=True).encode()).hexdigest()
    reported = "0" * 64 if MODE in ("hash_mismatch", "fingerprint_mismatch") else digest
    prov = {"skill": "qc", "skill_version": VERSION, "operation": "inspect" if MODE == "wrong_operation" else operation, "engine": {"ffmpeg_version": "fake", "ffprobe_version": "fake"},
            "input": {"fingerprint": reported, "size_bytes": f["size"]}, "identity": identity, "observed_at": "2026-09-05T00:00:00Z",
            "measurement_source": "ESTIMATED" if MODE == "not_observed" else "OBSERVED"}
    for k, p in companions.items():
        prov[f"{k}_input"] = {"fingerprint": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
    rep_kind = ("audio" if kind != "audio" else "video") if MODE == "wrong_kind" else kind
    report = {"id": ("report_" if MODE == "bad_report_id" else "qcreport_") + identity[:16], "version": "1", "operation": prov["operation"], "kind": rep_kind,
              "input": {"kind": rep_kind, "fingerprint": reported, "size_bytes": f["size"]}, "overall_status": overall, "checks": checks, "measurements": ms, "findings": findings, "provenance": prov}
    cache_status = "disabled" if "--no-cache" in argv else ("hit" if MODE == "reused" else "miss")
    doc = {"schema": RESPONSE, "status": "failed" if MODE == "status_failed_ok" else "completed", "skill": {"id": "qc", "version": VERSION}, "report": report, "provenance": prov,
           "reused": MODE == "reused", "cache": {"status": cache_status, "policy": req.get("cache_policy", "use"), "key": identity}}
    if MODE in ("no_report", "output_missing"):
        doc.pop("report")
    if "--no-cache" not in argv and MODE == "ok":
        os.makedirs(os.path.join(ws, ".qc-cache"), exist_ok=True)   # the real Skill keeps its cache under the workspace
    emit(doc)
    if MODE == "two_docs":
        emit(doc)
    return 3 if MODE == "nonzero_ok" else 0


def check(cid, status, codes=(), reason=None, evidence=None):
    c = {"check_id": cid, "category": cid.split(".")[0], "status": status, "measurement_ids": [], "finding_codes": list(codes), "evidence": evidence or {}}
    if reason:
        c["reason"] = reason
    return c


def stinger_checks(rule, checks, findings, ms) -> None:
    """qc-skill rules.py _evaluate_stinger, with canned measurements."""
    cp = float(rule["cut_point"])
    if rule.get("cover"):
        inside = float(rule["cover"]["start"]) <= cp < float(rule["cover"]["end"])
        checks.append(check("video.stinger_cut_point_in_cover", "PASS" if inside else "FAIL", [] if inside else ["VIDEO_STINGER_CUT_POINT_OUTSIDE_COVER"]))
        if not inside:
            findings.append({"code": "VIDEO_STINGER_CUT_POINT_OUTSIDE_COVER", "category": "video", "severity": "FAIL", "timestamp": cp,
                             "message": f"{cp:.3f}s the stated cut point is outside the stated cover window [{rule['cover']['start']}, {rule['cover']['end']})s"})
    mode = os.environ.get("FAKE_QC_STINGER", "ok")
    alpha = {"measured": False, "reason": "the stream's pixel format yuv420p has no alpha channel"} if mode == "unknown" else \
        {"measured": True, "frame": int(round(cp * 25)), "pixels": 2304, "pixels_below": 12 if mode == "not_opaque" else 0, "min_alpha_255": 200 if mode == "not_opaque" else 255}
    ms.append(measurement("video.stinger_cut_point_alpha", alpha, None, "ffmpeg:extractplanes"))
    if mode == "unknown":
        checks.append(check("video.stinger_opaque_at_cut_point", "UNKNOWN", reason=alpha["reason"]))
    else:
        bad = mode == "not_opaque"
        checks.append(check("video.stinger_opaque_at_cut_point", "FAIL" if bad else "PASS", ["VIDEO_STINGER_NOT_OPAQUE"] if bad else []))
        if bad:
            findings.append({"code": "VIDEO_STINGER_NOT_OPAQUE", "category": "video", "severity": "FAIL", "timestamp": cp,
                             "message": f"{cp:.3f}s frame {alpha['frame']} of the stinger: 12 of 2304 pixels have alpha below 253/255 (lowest 200/255)"})
    if rule.get("cut_at"):
        diff = float(os.environ.get("FAKE_QC_CUT_DIFF", "0.0"))
        cuts = [{"cut_at": t, "frame": int(round(t * 25)), "difference": diff, "measured": True} for t in rule["cut_at"]]
        ms.append(measurement("video.stinger_cut_frames", {"cuts": cuts}, None, "ffmpeg:rawvideo"))
        if rule.get("max_cut_frame_difference") is not None:
            over = [c for c in cuts if c["difference"] > float(rule["max_cut_frame_difference"])]
            checks.append(check("video.stinger_cut_frame_matches", "FAIL" if over else "PASS", ["VIDEO_STINGER_CUT_FRAME_MISMATCH"] if over else []))
            if over:
                findings.append({"code": "VIDEO_STINGER_CUT_FRAME_MISMATCH", "category": "video", "severity": "FAIL", "timestamp": over[0]["cut_at"],
                                 "message": f"{len(over)} cut frame(s) do not show the stinger's frame at its cut point (limit {rule['max_cut_frame_difference']})"})


def seam_checks(rule, duration, checks, findings, ms) -> None:
    """qc-skill rules.py loop-seam checks, with canned facts per seam."""
    unknown = os.environ.get("FAKE_QC_SEAM") == "unknown"
    vals = {"click": float(os.environ.get("FAKE_QC_SEAM_CLICK", "1.0")), "level": float(os.environ.get("FAKE_QC_SEAM_LEVEL", "0.3")), "band": float(os.environ.get("FAKE_QC_SEAM_BAND", "1.0"))}
    seams = [{"seam": t, "time_sec": t} for t in rule.get("seams") or []] + ([{"seam": "wrap", "time_sec": duration}] if rule.get("include_wrap", True) else [])
    ms.append(measurement("audio.loop_seams", {"measured": True, "window_sec": rule.get("window_sec"), "seams": seams}, None, "ffmpeg:rawaudio"))
    for field, cid, code, key in (("max_click_ratio", "audio.loop_seam_click_within_limit", "AUDIO_LOOP_SEAM_CLICK", "click"),
                                  ("max_level_diff_db", "audio.loop_seam_level_within_limit", "AUDIO_LOOP_SEAM_LEVEL_MISMATCH", "level"),
                                  ("max_band_diff_db", "audio.loop_seam_spectrum_within_limit", "AUDIO_LOOP_SEAM_SPECTRUM_MISMATCH", "band")):
        if rule.get(field) is None:
            continue
        if unknown:
            checks.append(check(cid, "UNKNOWN", reason="seam 2.0: within window_sec of the start of the file"))
            continue
        bad = vals[key] > float(rule[field])
        checks.append(check(cid, "FAIL" if bad else "PASS", [code] if bad else []))
        if bad and seams:
            findings.append({"code": code, "category": "audio", "severity": "FAIL", "timestamp": seams[0]["time_sec"],
                             "message": f"{len(seams)} seam(s) over the limit: {key} {vals[key]} > {rule[field]}"})


def run_package(req, roots, operation, kind, params, rules, argv) -> int:
    """kind delivery_package like the real Skill (engine.py _run_delivery_package / rules.py evaluate_delivery_package): N named artifacts,
    each present / sized / of its expected extension, the dependencies between them; nothing is probed."""
    arts = req.get("artifacts")
    if kind != "delivery_package" or req.get("input") is not None or not isinstance(arts, list) or not arts:
        return err("INVALID_REQUEST", "kind=delivery_package requires a non-empty request.artifacts list (and no input)")
    found = {}
    for a in arts:
        if not isinstance(a, dict) or set(a) - {"artifact_id", "artifact_type", "path"} or a.get("artifact_type") not in CONTRACT["delivery_package"]["artifact_types"]:
            return err("INVALID_REQUEST", "artifact entry shape / type")
        p, bad = resolve(a.get("path"), roots, "artifacts.path")
        if bad:
            return bad
        found[a["artifact_id"]] = (a["artifact_type"], p)
    rule = rules.get("delivery_package") or {}
    checks, findings, ms = [], [], []
    for aid, (_t, p) in found.items():
        ms += [dict(measurement("delivery_package.artifact_present", True, None, "OBSERVED"), artifact_id=aid),
               dict(measurement("delivery_package.artifact_extension", Path(p).suffix.lower().lstrip("."), None, "OBSERVED"), artifact_id=aid)]
    last = list(found)[-1]
    for ar in rule.get("artifacts") or []:
        aid = ar["artifact_id"]
        checks.append({"check_id": "delivery_package.artifact_present", "category": "delivery_package", "status": "PASS" if aid in found else "FAIL", "measurement_ids": [], "finding_codes": [], "artifact_id": aid})
        if aid in found and ar.get("min_size_bytes") is not None:
            checks.append({"check_id": "delivery_package.artifact_size_within_limit", "category": "delivery_package", "status": "PASS" if os.path.getsize(found[aid][1]) >= ar["min_size_bytes"] else "FAIL",
                           "measurement_ids": [], "finding_codes": [], "artifact_id": aid})
        if aid in found and ar.get("expected_extension") is not None:
            ok = Path(found[aid][1]).suffix.lower().lstrip(".") == ar["expected_extension"] and not (MODE == "package_fail" and aid == last)
            checks.append({"check_id": "delivery_package.artifact_extension_matches_expected", "category": "delivery_package", "status": "PASS" if ok else "FAIL",
                           "measurement_ids": [], "finding_codes": [] if ok else ["DELIVERY_PACKAGE_ARTIFACT_EXTENSION_MISMATCH"], "artifact_id": aid})
            if not ok:
                findings.append({"code": "DELIVERY_PACKAGE_ARTIFACT_EXTENSION_MISMATCH", "category": "delivery_package", "severity": "FAIL", "message": f"{aid} extension", "artifact_id": aid})
    for dep in (rule.get("cross_artifact") or {}).get("dependencies") or []:
        if dep["artifact_id"] in found:
            checks.append({"check_id": "delivery_package.dependency_satisfied", "category": "delivery_package", "status": "PASS" if dep["requires_artifact_id"] in found else "FAIL",
                           "measurement_ids": [], "finding_codes": []})
    overall = "FAIL" if any(c["status"] == "FAIL" for c in checks) else "PASS"
    fps = {aid: hashlib.sha256(Path(p).read_bytes()).hexdigest() for aid, (_t, p) in found.items()}
    if MODE == "package_fingerprint":
        fps[last] = "0" * 64
    identity = hashlib.sha256(json.dumps({"fps": fps, "kind": kind, "op": operation, "params": params, "rules": rules}, sort_keys=True).encode()).hexdigest()
    prov = {"skill": "qc", "skill_version": VERSION, "operation": operation, "engine": {"ffmpeg_version": "fake", "ffprobe_version": "fake"},
            "artifacts": [{"artifact_id": aid, "artifact_type": t, "present": True, "fingerprint": fps[aid], "size_bytes": os.path.getsize(p)} for aid, (t, p) in found.items()],
            "identity": identity, "observed_at": "2026-09-05T00:00:00Z", "measurement_source": "OBSERVED"}
    report = {"id": "qcreport_" + identity[:16], "version": "1", "operation": operation, "kind": kind, "input": {"kind": kind, "artifacts": list(found)},
              "overall_status": overall, "checks": checks, "measurements": ms, "findings": findings, "provenance": prov}
    emit({"schema": RESPONSE, "status": "completed", "skill": {"id": "qc", "version": VERSION}, "report": report, "provenance": prov, "reused": False,
          "cache": {"status": "disabled" if "--no-cache" in argv else "miss", "policy": req.get("cache_policy", "use"), "key": identity}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
