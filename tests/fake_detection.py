"""Fake detection-skill process for the ADR-090 tests: speaks the slice of the real `detection` CLI the agent uses -- `contract --json`
(the pinned src/video_agent/tools/detection/contract_0.1.0.json), `doctor --json` (ok / ready / checks / problems, exit 3 while not ok),
`faces --json --overwrite --output=F --every=S [--min-score=X] [--ffmpeg-skill=D] --timeout=T -- <video>` (detection-skill/detections@1 with
the sha256 of the input, printed with ok / detections_file and written to --output) and `tracks --json --overwrite --output=F [--window=S]
-- <faces.json>` (detection-skill/tracks@1 made from that file, with its sha256). The media are the self-describing fake files of
tests/pipeline_harness.py (1280x720 like the fake engine's probe): their `faces` fact [{"start", "end", "box": [x, y, w, h]}] says which
box is on screen when; the sampled frames see it, the tracks link equal boxes into one track whose redact rectangle is the box made even,
extended by one sampling step. Nothing runs an engine.
Controlled via:

  FAKE_DET_READY: all (default) | model (the YuNet file missing) | frames (ffmpeg-skill unusable)
  FAKE_DET_MODE: ok | fail:<CODE> | bad_fingerprint | label | wrong_every | odd_rect | tracks_other_file | not_written
  FAKE_DET_CONTRACT: pinned | drift_version (0.1.9: non-fatal) | drift_fatal (tracks argv changed) | incompatible (0.2.0)
  FAKE_DET_CALLS: a file every invocation's argv is appended to (one JSON list per line). Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

MODE = os.environ.get("FAKE_DET_MODE", "ok")
READY = os.environ.get("FAKE_DET_READY", "all")
CONTRACT_MODE = os.environ.get("FAKE_DET_CONTRACT", "pinned")
CALLS_LOG = os.environ.get("FAKE_DET_CALLS")
PINNED = Path(__file__).resolve().parents[1] / "src" / "video_agent" / "tools" / "detection" / "contract_0.1.0.json"
W, H = 1280, 720


def _emit(doc) -> None:
    sys.stdout.write(json.dumps(doc) + "\n")


def _contract() -> dict:
    c = json.loads(PINNED.read_text(encoding="utf-8"))
    if CONTRACT_MODE == "drift_version":
        c["version"] = "0.1.9"
    elif CONTRACT_MODE == "drift_fatal":
        next(t for t in c["tools"] if t["tool_id"] == "detection/tracks")["argv"].append("--strict")
    elif CONTRACT_MODE == "incompatible":
        c["version"] = "0.2.0"
    return c


def _fail(code: str, message: str) -> int:
    _emit({"ok": False, "error": {"code": code, "message": message, "detail": None}})
    return int(_contract()["errors"].get(code, {}).get("exit_code", 1))


def _doctor(argv) -> int:
    model_ok, frames_ok = READY != "model", READY != "frames"
    checks = {"python": {"status": "OK"}, "opencv": {"status": "OK", "install": None}, "numpy": {"status": "OK", "install": None},
              "ffmpeg_skill": {"status": "OK" if frames_ok else "INCOMPATIBLE", "install": None if frames_ok else "ffmpeg-skill 2.x or 3.x with a working ffmpeg"},
              "face_model": {"status": "OK" if model_ok else "MISSING",
                             "install": None if model_ok else "download face_detection_yunet_2026may.onnx (MIT) from https://github.com/opencv/opencv_zoo and set $DETECTION_MODELS_DIR"},
              "object_model": {"status": "MISSING", "install": "download a YOLOX ONNX file"}, "tesseract": {"status": "OK"}}
    problems = [f"{k}: {v['status']}" for k, v in checks.items() if v["status"] != "OK"]
    ready = {"faces": model_ok, "text": True, "objects": False, "video_frames": frames_ok, "tracks": True, "check": True}
    _emit({"ok": all(ready.values()), "skill": "detection-skill", "version": _contract()["version"], "ready": ready, "checks": checks, "problems": problems,
           "network": "never used at run time", "argv_ffmpeg_skill": any(a.startswith("--ffmpeg-skill=") for a in argv)})
    return 0 if all(ready.values()) else 3


def _opts(argv):
    opts, pos, after = {}, [], False
    for a in argv:
        if after:
            pos.append(a)
        elif a == "--":
            after = True
        elif a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            opts[k] = v
        elif a.startswith("--"):
            opts[a[2:]] = True
        else:
            pos.append(a)
    return opts, pos


def _engine():
    return {"id": "yunet", "kind": "face", "version": "4.10.0", "runtime": "opencv", "model": "face_detection_yunet_2026may.onnx",
            "model_sha256": "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0", "model_license": "MIT", "engine_license": "Apache-2.0",
            "files": [], "execution_mode": "local", "engine_seconds": 0.1, "details": {}}


def _faces(argv) -> int:
    opts, pos = _opts(argv)
    if READY == "model":
        return _fail("MODEL_NOT_FOUND", "no YuNet model (fake)")
    if READY == "frames":
        return _fail("FFMPEG_SKILL_NOT_FOUND", "ffmpeg-skill unusable (fake)")
    if MODE.startswith("fail:"):
        return _fail(MODE.split(":", 1)[1], "refused (fake)")
    media = pos[0]
    raw = Path(media).read_bytes()
    try:
        facts = json.loads(raw.decode())
    except ValueError:
        facts = {}
    dur = float(facts.get("duration") or 10.0)
    every = float(opts["every"])
    times, t = [], 0.0
    while t < dur - 1e-9:
        times.append(round(t, 3))
        t += every
    eng = _engine()
    min_score = float(opts.get("min-score", 0.6))
    frames = []
    for i, t in enumerate(times):
        dets = []
        for f in facts.get("faces") or []:
            if float(f["start"]) <= t <= float(f["end"]):
                x, y, w, h = [float(v) for v in f["box"]]
                d = {"t": t, "kind": "face", "box": [x, y, w, h], "box_norm": [round(x / W, 5), round(y / H, 5), round(w / W, 5), round(h / H, 5)], "score": 0.91,
                     "label": None, "landmarks": [], "engine": eng["id"], "model": eng["model"], "model_sha256": eng["model_sha256"]}
                if MODE == "label":
                    d["label"] = "someone"
                dets.append(d)
        frames.append({"index": i, "t": t, "detections": dets})
    fp = "sha256:" + (("0" * 64) if MODE == "bad_fingerprint" else hashlib.sha256(raw).hexdigest())
    doc = {"schema": "detection-skill/detections@1", "kind": "face",
           "source": {"filename": os.path.basename(media), "fingerprint": fp, "size_bytes": len(raw), "media_type": "video", "duration": dur, "width": W, "height": H, "fps": 30.0},
           "frame_size": {"width": W, "height": H},
           "sampling": {"method": "every", "every": every + (0.5 if MODE == "wrong_every" else 0.0), "times": times, "scene_points": [], "frames_tool": "ffmpeg-skill/look",
                        "ffmpeg_skill_version": "3.1.1", "extracted_size": {"width": W, "height": H}},
           "engine": eng, "parameters": {"min_score": min_score}, "frames": frames,
           "counts": {"frames": len(frames), "frames_with_detections": sum(1 for f in frames if f["detections"]), "detections": sum(len(f["detections"]) for f in frames)},
           "skill": "detection-skill", "skill_version": _contract()["version"], "tool": "detection/faces", "created_at": "2026-10-31T00:00:00Z", "processing_seconds": 0.1, "warnings": []}
    out = opts.get("output")
    if out and MODE != "not_written":
        Path(out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    _emit(dict(doc, ok=True, detections_file=out))
    return 0


def _even_out(x0: float, y0: float, x1: float, y1: float):
    xa, ya = int(x0) - int(x0) % 2, int(y0) - int(y0) % 2
    xb, yb = int(-(-x1 // 1)), int(-(-y1 // 1))
    xb += (xb - xa) % 2
    yb += (yb - ya) % 2
    return max(0, xa), max(0, ya), min(W, xb), min(H, yb)


def _tracks(argv) -> int:
    opts, pos = _opts(argv)
    data = Path(pos[0]).read_bytes()
    doc = json.loads(data.decode())
    step = float(doc["sampling"]["every"])
    dur = float(doc["source"]["duration"])
    by_box = {}
    for fr in doc["frames"]:
        for d in fr["detections"]:
            by_box.setdefault(tuple(d["box"]), []).append({"t": d["t"], "box": d["box"], "score": d["score"]})
    tracks = []
    for n, (box, boxes) in enumerate(sorted(by_box.items(), key=lambda kv: kv[1][0]["t"]), start=1):
        x, y, w, h = box
        xa, ya, xb, yb = _even_out(x, y, x + w, y + h)
        rect = {"start": round(max(0.0, boxes[0]["t"] - step), 3), "end": round(min(dur, boxes[-1]["t"] + step), 3), "x": xa, "y": ya, "width": xb - xa, "height": yb - ya}
        if MODE == "odd_rect":
            rect["width"] += 1
        tracks.append({"id": f"track_{n:03d}", "label": None, "start": boxes[0]["t"], "end": boxes[-1]["t"], "hits": len(boxes), "mean_score": 0.91, "max_score": 0.91,
                       "boxes": boxes, "redact": [rect]})
    fp = "sha256:" + (("1" * 64) if MODE == "tracks_other_file" else hashlib.sha256(data).hexdigest())
    window = float(opts.get("window", 0.0))
    out_doc = {"schema": "detection-skill/tracks@1", "kind": "face", "source": doc["source"], "frame_size": doc["frame_size"],
               "detections": {"file": os.path.basename(pos[0]), "fingerprint": fp, "tool": "detection/faces", "engine": "yunet", "model": doc["engine"]["model"],
                              "model_sha256": doc["engine"]["model_sha256"], "sampling_method": "every", "sampling_step": step},
               "method": {"name": "greedy-iou", "heuristic": True, "identity": False, "note": "heuristic: boxes linked over time by IoU; a track id is not an identity (redact spans are extended by `extend` on each side)"},
               "parameters": {"min_iou": 0.3, "max_shift": 1.0, "max_gap": max(2.5 * step, 0.5), "pad": 0.15, "extend": step, "window": window, "min_hits": 1, "same_label": False},
               "tracks": tracks, "counts": {"tracks": len(tracks), "boxes_linked": sum(t["hits"] for t in tracks), "boxes_dropped": 0, "redact_rects": len(tracks)},
               "skill": "detection-skill", "skill_version": _contract()["version"], "tool": "detection/tracks", "created_at": "2026-10-31T00:00:00Z", "warnings": [] if tracks else ["no tracks"]}
    out = opts.get("output")
    Path(out).write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")
    _emit(dict(out_doc, ok=True, tracks_file=out))
    return 0


def main(argv) -> int:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\n")
    if argv[:1] == ["contract"]:
        _emit(_contract())
        return 0
    if argv[:1] == ["doctor"]:
        return _doctor(argv[1:])
    if argv[:1] == ["faces"]:
        return _faces(argv[1:])
    if argv[:1] == ["tracks"]:
        return _tracks(argv[1:])
    return _fail("INVALID_ARGUMENT", f"unsupported command {argv[:1]}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
