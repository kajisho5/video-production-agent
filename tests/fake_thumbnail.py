"""Fake thumbnail-skill process for adapter boundary tests: speaks the real transport (`skill --json`, `doctor --json`,
`run - --json --workspace D --allowed-input R… [--ffmpeg-skill X] [--timeout S]` with {"tool", "params"} on stdin → one
{"ok", "tool", "result"} document on stdout, exit code from the nested result), enforces the request shape, forbidden keys,
contract ranges and the path policy like the real Skill, writes a real (tiny, constant) PNG or JPEG output, and misbehaves on
request via FAKE_THUMBNAIL_MODE:

  ok | reused | tool_error | tool_error_final | output_missing | validation_error | cancelled | timeout | hang | malformed | empty |
  two_docs | text | nonzero_ok | wrong_schema | wrong_skill | wrong_version | bad_contract | contract_fail | contract_drift |
  hash_mismatch | wrong_size | unknown_code | internal_error | doctor_fail | doctor_degraded | no_provenance | outer_not_ok |
  wrong_tool | pillow_missing

Reported dimensions: the canvas for render, 1280x720 for extract_frame (the real Skill delivers ffmpeg-skill/look's default
width regardless of the source). An input whose bytes start with b'{"fake"' is parsed for its duration (default 16 s).
FAKE_THUMBNAIL_CALLS: a file every invocation is appended to (security tests read it). Never runs ffmpeg or ffmpeg-skill,
never imports thumbnail_skill or Pillow. Test double only."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE.parent / "src" / "video_agent" / "tools" / "thumbnail" / "contract_0.1.18.json").read_text(encoding="utf-8"))
MODE = os.environ.get("FAKE_THUMBNAIL_MODE", "ok")
CALLS_LOG = os.environ.get("FAKE_THUMBNAIL_CALLS")
EXIT = CONTRACT["errors"]["exit_codes"]
RETRY = CONTRACT["errors"]["retryable"]
TOOLS = [t["tool_id"] for t in CONTRACT["tools"]]
FORBIDDEN = set(CONTRACT["document"]["forbidden_fields"])
FONT_IDS = CONTRACT["fonts"]["font_ids"]
FORMATS = {k: tuple(v["extensions"]) for k, v in CONTRACT["output_formats"].items()}
SKILL = {"id": CONTRACT["skill_id"], "version": CONTRACT["version"]}
RESPONSE = "thumbnail-skill/response@1"
ID_RE = re.compile(CONTRACT["document"]["id_pattern"])
COLOR_RE = re.compile(r"^#(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$")
PNG_1x1 = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC")
JPEG_1x1 = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2NjY2NjY2Nj"
    "Y2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2P/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEI"
    "I0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo"
    "6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNE"
    "RUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDz+iiigD//2Q==")


class Fail(Exception):
    def __init__(self, code: str, message: str, details=None, retryable=None):
        super().__init__(message)
        self.code, self.message, self.details, self.retryable = code, message, details or {}, retryable


def log(kind: str, argv) -> None:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"cmd": kind, "argv": list(argv), "cwd": os.getcwd(), "env_video": {k: v for k, v in os.environ.items() if k.startswith("VIDEO_")}}) + "\n")


def emit(doc) -> None:
    sys.stdout.write(json.dumps(doc, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def error_result(f: Fail) -> dict:
    return {"schema": RESPONSE, "skill": dict(SKILL), "ok": False, "status": "cancelled" if f.code == "CANCELLED" else "error",
            "error": {"code": f.code, "message": f.message, "retryable": RETRY[f.code] if f.retryable is None else f.retryable, "details": f.details}, "warnings": []}


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
            if str(k).lower() in FORBIDDEN:
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


def obj(value, where, allowed, required):
    if not isinstance(value, dict):
        raise Fail("INVALID_REQUEST", f"{where} must be an object", {"field": where})
    unknown = sorted(k for k in value if k not in allowed)
    if unknown:
        raise Fail("INVALID_REQUEST", f"{where}: unknown field(s) {unknown}", {"field": where, "unknown": unknown})
    missing = [k for k in required if k not in value]
    if missing:
        raise Fail("INVALID_REQUEST", f"{where}: missing required field(s) {missing}", {"field": where})
    return value


def num(value, where, lo, hi, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Fail("INVALID_REQUEST", f"{where} must be a number", {"field": where})
    if integer and float(value) != int(value):
        raise Fail("INVALID_REQUEST", f"{where} must be an integer", {"field": where})
    if value < lo or value > hi:
        raise Fail("INVALID_REQUEST", f"{where} must be within [{lo}, {hi}]", {"field": where})
    return value


def color(value, where):
    if not isinstance(value, str) or not COLOR_RE.match(value):
        raise Fail("INVALID_REQUEST", f"{where} must be a colour", {"field": where})
    return value


def parse_output(value, ws: str, sources):
    d = obj(value, "output", ("path", "format", "overwrite", "jpeg_quality"), ("path", "format"))
    fmt = d["format"]
    if fmt not in FORMATS:
        raise Fail("UNSUPPORTED_FORMAT", f"output.format {fmt!r} is not supported", {"field": "output.format"})
    raw = d["path"]
    if not isinstance(raw, str) or not raw:
        raise Fail("INVALID_REQUEST", "output.path must be a non-empty string")
    if not raw.lower().endswith(FORMATS[fmt]):
        raise Fail("UNSUPPORTED_FORMAT", f"output.path must end with one of {FORMATS[fmt]}", {"field": "output.path"})
    if "jpeg_quality" in d:
        if fmt != "jpeg":
            raise Fail("INVALID_REQUEST", "output.jpeg_quality is only accepted for jpeg")
        num(d["jpeg_quality"], "output.jpeg_quality", 1, 100, integer=True)
    if any(part == ".." for part in Path(raw).parts):
        raise Fail("PATH_NOT_ALLOWED", "traversal", {"reason": "traversal"})
    out = os.path.realpath(raw if os.path.isabs(raw) else os.path.join(ws, raw))
    if not within(ws, out):
        raise Fail("PATH_NOT_ALLOWED", "output is outside the workspace", {"reason": "outside_workspace"})
    if any(os.path.normcase(s) == os.path.normcase(out) for s in sources):
        raise Fail("OUTPUT_ERROR", "output would overwrite an input", {"reason": "input_output_collision"})
    if os.path.exists(out) and not d.get("overwrite"):
        raise Fail("OUTPUT_ERROR", "output already exists", {"reason": "exists"})
    return out, fmt


def resolve_input(raw, ws: str, roots, what: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise Fail("INVALID_REQUEST", f"{what}.path must be a non-empty string")
    p = os.path.realpath(raw if os.path.isabs(raw) else os.path.join(ws, raw))
    if not os.path.isfile(p):
        raise Fail("INVALID_INPUT", f"{what} not found: {raw}", {"reason": "not_found"})
    if roots and not any(within(r, p) for r in roots):
        raise Fail("PATH_NOT_ALLOWED", f"{what} is outside the allowed input roots", {"reason": "outside_allowed_roots"})
    return p


def timestamp(raw, where):
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise Fail("INVALID_TIME_RANGE", f"{where} must be a finite number", {"field": where})
    if raw < 0:
        raise Fail("INVALID_TIME_RANGE", f"{where} must not be negative", {"field": where})
    return float(raw)


def check_frame(path: str, ts: float, what: str) -> float:
    dur = in_dur(path)
    if ts > dur:
        raise Fail("INVALID_TIME_RANGE", f"{what}: timestamp {ts}s is beyond the source duration ({dur:.3f}s)", {"timestamp": ts, "duration": dur})
    return dur


def parse_document(value, ws, roots):
    d = obj(value, "document", ("document_id", "canvas", "assets", "elements", "metadata"), ("document_id", "canvas", "elements"))
    if not isinstance(d["document_id"], str) or not ID_RE.match(d["document_id"]):
        raise Fail("INVALID_REQUEST", "document.document_id must match the id pattern")
    c = obj(d["canvas"], "document.canvas", ("width", "height", "background"), ("width", "height"))
    w, h = num(c["width"], "canvas.width", 16, 7680, True), num(c["height"], "canvas.height", 16, 7680, True)
    color(c.get("background", "#000000"), "canvas.background")
    assets = {}
    for i, a in enumerate(d.get("assets") or []):
        a = obj(a, f"assets[{i}]", ("asset_id", "kind", "path", "timestamp"), ("asset_id", "kind", "path"))
        if a["kind"] not in CONTRACT["document"]["assets"]["kinds"]:
            raise Fail("INVALID_REQUEST", "asset kind")
        path = resolve_input(a["path"], ws, roots, f"asset {a['asset_id']!r}")
        ident = {"asset_id": a["asset_id"], "kind": a["kind"], "path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        if a["kind"] == "video_frame":
            if "timestamp" not in a:
                raise Fail("INVALID_REQUEST", "video_frame needs a timestamp")
            ts = timestamp(a["timestamp"], f"assets[{i}].timestamp")
            ident["timestamp"], ident["source_duration"] = ts, check_frame(path, ts, f"asset {a['asset_id']!r}")
        assets[a["asset_id"]] = ident
    fonts = []
    if not isinstance(d["elements"], list) or not d["elements"]:
        raise Fail("INVALID_REQUEST", "document.elements must be a non-empty array")
    for i, e in enumerate(d["elements"]):
        e = obj(e, f"elements[{i}]", ("element_id", "type", "z_index", "image", "text"), ("element_id", "type"))
        if e["type"] == "image":
            im = obj(e.get("image"), f"elements[{i}].image", ("asset_id", "position", "size", "fit", "crop", "opacity", "rotation"), ("asset_id", "position", "size"))
            if im["asset_id"] not in assets:
                raise Fail("MISSING_INPUT", f"asset {im['asset_id']!r} is not declared")
            if im.get("fit", "cover") not in CONTRACT["document"]["elements"]["image"]["fields"]["fit"]:
                raise Fail("INVALID_REQUEST", "fit")
        elif e["type"] == "text":
            t = obj(e.get("text"), f"elements[{i}].text", ("text", "font_id", "font_size", "color", "position", "align", "line_spacing", "opacity", "background", "stroke", "shadow"),
                    ("text", "font_id", "font_size", "color", "position"))
            if not isinstance(t["text"], str) or not t["text"] or len(t["text"]) > 2000:
                raise Fail("INVALID_REQUEST", "text length")
            if t["font_id"] not in FONT_IDS:
                raise Fail("MISSING_INPUT", f"font_id {t['font_id']!r} is not registered", {"font_id": t["font_id"]})
            num(t["font_size"], "font_size", 6, 400, True)
            color(t["color"], "text.color")
            al = obj(t.get("align") or {}, "align", ("horizontal", "vertical"), ())
            if al.get("horizontal", "left") not in ("left", "center", "right") or al.get("vertical", "top") not in ("top", "middle", "bottom"):
                raise Fail("INVALID_REQUEST", "align vocabulary")
            fonts.append({"font_id": t["font_id"], "font_name": t["font_id"], "path": f"/fake/fonts/{t['font_id']}.ttf", "index": 0, "font_file_hash": "sha256:" + "f" * 64})
        else:
            raise Fail("INVALID_REQUEST", "element type")
    return d["document_id"], int(w), int(h), assets, fonts


# ---- ADR-104: the 0.1.12 .. 0.1.15 tools, with the real Skill's request shapes and response fields (tiny constant images, a simple text box)
def write_image(out: str, fmt: str) -> tuple:
    payload = PNG_1x1 if fmt == "png" else JPEG_1x1
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Path(out).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    return len(payload), "0" * 64 if MODE == "hash_mismatch" else digest


def base_prov(tool: str) -> dict:
    return {"skill": SKILL["id"], "skill_version": SKILL["version"], "operation": tool, "engine": "Pillow", "engine_version": "12.3.0-fake"}


def text_box(t: dict) -> list:
    """A stand-in for renderer.text_box: about 0.6 em per character, one line, anchored by the element's alignment."""
    w, h = 0.6 * t["font_size"] * len(t["text"]), float(t["font_size"])
    al = t.get("align") or {}
    x = t["position"]["x"] - {"left": 0, "center": w / 2, "right": w}[al.get("horizontal", "left")]
    y = t["position"]["y"] - {"top": 0, "middle": h / 2, "bottom": h}[al.get("vertical", "top")]
    return [x, y, x + w, y + h]


def choose_frame(params, ws, roots) -> dict:
    p = obj(params, "params", ("detections", "detections_path", "video", "measures", "weights", "eyes_closed_below", "options"), ())
    path = resolve_input(p["detections_path"], ws, roots, "detections_path")
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != "detection-skill/detections@1" or doc.get("kind") != "face":
        raise Fail("INVALID_REQUEST", "detections must be face detections")
    faces_any = [d for fr in doc["frames"] for d in fr["detections"]]
    if faces_any and not any(isinstance(d.get("attributes"), dict) for d in faces_any):
        raise Fail("INVALID_REQUEST", "the faces carry no attributes: run detection faces --attributes", {"reason": "no_attributes"})
    video = None
    if "video" in p:
        v = resolve_input(p["video"], ws, roots, "video")
        fp = "sha256:" + hashlib.sha256(Path(v).read_bytes()).hexdigest()
        if doc["source"].get("fingerprint") != fp:
            raise Fail("INVALID_INPUT", "video is not the file these detections were made from", {"reason": "detections_source_mismatch"})
        video = {"path": v, "fingerprint": "sha256:" + ("1" * 64 if MODE == "rank_other_video" else fp[7:])}
    ranked, empty = [], []
    for pos, fr in enumerate(doc["frames"]):
        boxes = [list(d["box"]) for d in fr["detections"]]
        e = {"index": fr.get("index", pos), "t": fr["t"], "faces": boxes}
        if not boxes:
            empty.append(dict(e, rank=None, score=None, parts=None, missing=[], eyes_closed=None, main_face=None))
            continue
        main = max(range(len(boxes)), key=lambda j: (boxes[j][2] * boxes[j][3], -j))
        a = fr["detections"][main].get("attributes") or {}
        parts = {"eyes_open": a.get("eyes_open"), "sharpness": a["sharpness"] / (a["sharpness"] + 100) if a.get("sharpness") is not None else None, "exposure": None,
                 "frontal": a.get("frontal"), "size": min(1.0, a["size"] / 0.4) if a.get("size") is not None else None}
        w = {"eyes_open": 0.3, "sharpness": 0.25, "exposure": 0.15, "frontal": 0.15, "size": 0.15}
        ranked.append(dict(e, score=round(sum(w[k] * (v or 0.0) for k, v in parts.items()), 4), parts=parts, missing=[k for k, v in parts.items() if v is None],
                           sources={}, eyes_closed=a.get("eyes_open") is not None and a["eyes_open"] < 0.5, main_face=main, main_face_box=boxes[main], _o=pos))
    ranked.sort(key=lambda e: (e["eyes_closed"], -e["score"], e["_o"]))
    for n, e in enumerate(ranked, start=1):
        e["rank"] = n
        del e["_o"]
    body = {"ranking": ranked + empty, "chosen": ranked[0] if ranked else None, "frame_size": dict(doc["frame_size"]),
            "method": {"weights": {}, "heuristic": True, "order": "eyes open first, then score"},
            "source": {"detections": "detection-skill/detections@1", "fingerprint": doc["source"].get("fingerprint"), "frames": len(doc["frames"])},
            "operations": ["thumbnail/choose_frame"], "provenance": {"skill": SKILL["id"], "skill_version": SKILL["version"], "operation": "thumbnail/choose_frame"}}
    if video:
        body["video"] = video
    return body


def place_text(params, ws, roots) -> dict:
    p = obj(params, "params", ("document", "text_element", "image_element", "faces", "frame_size", "candidates", "margin", "options"), ("document", "text_element", "image_element", "faces"))
    doc_id, w, h, assets, fonts = parse_document(p["document"], ws, roots)
    els = {e["element_id"]: e for e in p["document"]["elements"]}
    t = (els.get(p["text_element"]) or {}).get("text")
    im = (els.get(p["image_element"]) or {}).get("image")
    if t is None or im is None:
        raise Fail("INVALID_REQUEST", "text_element / image_element")
    if assets[im["asset_id"]]["kind"] == "video_frame" and "frame_size" not in p:
        raise Fail("INVALID_REQUEST", "frame_size is required", {"field": "frame_size"})
    fs = p.get("frame_size") or {"width": w, "height": h}
    sx, sy = im["size"]["width"] / fs["width"], im["size"]["height"] / fs["height"]
    m = int(p.get("margin", 16))
    faces = [[f[0] * sx - m, f[1] * sy - m, (f[0] + f[2]) * sx + m, (f[1] + f[3]) * sy + m] for f in p["faces"]]
    mm = round(0.05 * min(w, h))
    cands = [("bottom-center", w / 2, h - mm, "center", "bottom"), ("top-center", w / 2, mm, "center", "top"), ("bottom-left", mm, h - mm, "left", "bottom"),
             ("top-left", mm, mm, "left", "top")]
    tried, chosen = [], None
    for name, x, y, ha, va in cands:
        t2 = dict(t, position={"x": x, "y": y}, align={"horizontal": ha, "vertical": va})
        b = text_box(t2)
        inside = b[0] >= 0 and b[1] >= 0 and b[2] <= w and b[3] <= h
        hits = [i for i, f in enumerate(faces) if b[0] < f[2] and f[0] < b[2] and b[1] < f[3] and f[1] < b[3]]
        ok = inside and not hits and MODE != "no_clear"
        tried.append({"name": name, "text_box": b, "inside_canvas": inside, "overlaps_faces": hits, "clear": ok})
        if ok:
            chosen = (name, {"x": x, "y": y}, {"horizontal": ha, "vertical": va})
            break
    if chosen is None:
        raise Fail("VALIDATION_ERROR", "no candidate position keeps the text clear of every face", {"reason": "no_position_clear_of_faces", "tried": tried})
    t.update(position=chosen[1], align=chosen[2])
    return {"document": p["document"], "placement": {"text_element": p["text_element"], "image_element": p["image_element"], "chosen": chosen[0], "position": chosen[1],
                                                     "align": chosen[2], "text_box": tried[-1]["text_box"], "face_boxes": faces, "margin": m, "tried": tried},
            "operations": ["thumbnail/place_text"], "provenance": {"skill": SKILL["id"], "skill_version": SKILL["version"], "operation": "thumbnail/place_text"}}


def render_layers(params, ws, roots) -> dict:
    p = obj(params, "params", ("schema", "document", "output", "options", "layers"), ("document", "output", "layers"))
    lv = obj(p["layers"], "layers", ("directory", "overwrite"), ("directory",))
    d = os.path.realpath(lv["directory"] if os.path.isabs(lv["directory"]) else os.path.join(ws, lv["directory"]))
    if not within(ws, d):
        raise Fail("PATH_NOT_ALLOWED", "layers.directory outside the workspace", {"reason": "outside_workspace"})
    res = render_one("thumbnail/render", {k: v for k, v in p.items() if k != "layers"}, ws, roots)
    os.makedirs(d, exist_ok=True)
    layers = []
    for order, (role, eid) in enumerate([("background", None)] + [("image" if e["type"] == "image" else "text", e["element_id"]) for e in p["document"]["elements"]]):
        name = f"{order:02d}-{'background' if eid is None else eid}.png"
        Path(d, name).write_bytes(PNG_1x1)
        layers.append({"order": order, "role": role, "element_id": eid, "type": None, "z_index": None, "file": name, "position": {"x": 0, "y": 0}, "size": {"width": 1, "height": 1},
                       "mode": "RGBA", "sha256": "sha256:" + hashlib.sha256(PNG_1x1).hexdigest()})
    if MODE == "layers_tampered":
        Path(d, layers[-1]["file"]).write_bytes(PNG_1x1 + b"x")
    manifest = {"schema": "thumbnail-skill/layers@1", "canvas": {"width": res["width"], "height": res["height"]}, "layers": layers, "verified": {"pixel_equal": True}}
    Path(d, "layers.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    res["layers"] = {"directory": d, "manifest": os.path.join(d, "layers.json"), "count": len(layers), "layers": layers, "verified": {"pixel_equal": True}}
    res["operations"] = ["thumbnail/render", "thumbnail/render_layers"]
    res["provenance"]["layers_manifest_sha256"] = "sha256:" + hashlib.sha256(Path(d, "layers.json").read_bytes()).hexdigest()
    return res


def render_variants(params, ws, roots) -> dict:
    p = obj(params, "params", ("schema", "document", "output", "options", "variants"), ("document", "output", "variants"))
    v = obj(p["variants"], "variants", ("count", "text_element", "layouts", "palettes", "frame_element", "frames", "phone_width", "min_text_height_px", "min_contrast_ratio", "qc_skill"), ())
    n = num(v.get("count"), "variants.count", 1, 12, True)
    if not any(e["type"] == "text" for e in p["document"]["elements"]):
        raise Fail("INVALID_REQUEST", "the document has no text element to vary")
    if "frames" in v and "frame_element" not in v:
        raise Fail("INVALID_REQUEST", "variants.frames needs variants.frame_element")
    for f in v.get("frames") or []:
        a = obj(f, "frame", ("name", "asset"), ("name", "asset"))
        resolve_input(a["asset"]["path"], ws, roots, "variant frame")
    base = Path(p["output"]["path"])
    out = []
    for k in range(1, n + 1):
        body = dict(p, output=dict(p["output"], path=str(base.with_name(f"{base.stem}-v{k}{base.suffix}"))))
        body.pop("variants")
        r = render_one("thumbnail/render", body, ws, roots)
        fail = MODE == "variants_flag" and k == 2
        out.append({"variant": k, "output": r["output"], "format": r["format"], "width": r["width"], "height": r["height"], "size": r["size"], "sha256": r["sha256"], "reused": False,
                    "preset_check": None, "choices": {"layout": "as-given" if k == 1 else f"layout-{k}", "palette": "as-given" if k == 1 else f"palette-{k}", "frame": "as-given"},
                    "description": f"variant {k}", "small_size": {"status": "fail" if fail else "pass", "ok": not fail, "phone_width": v.get("phone_width", 320), "min_text_height_px": 12,
                                                                 "min_contrast_ratio": 4.5, "texts": [{"element_id": "caption", "findings": [{"code": "text_too_small"}] if fail else []}],
                                                                 "qc_skill": v.get("qc_skill")}, "identity": r["provenance"]["identity"]})
    return {"variants": out, "count": n, "flagged": [x["variant"] for x in out if x["small_size"]["status"] == "fail"], "text_element": "caption", "axes": {},
            "operations": ["thumbnail/render_variants"], "provenance": dict(base_prov("thumbnail/render_variants"), outputs=[{"variant": x["variant"], "output_hash": x["sha256"]} for x in out],
                                                                              qc_skill=v.get("qc_skill"))}


def render_batch(params, ws, roots) -> dict:
    import csv as _csv
    import io as _io
    p = obj(params, "params", ("csv", "rows", "targets", "output", "options"), ("targets", "output"))
    c = obj(p["csv"], "csv", ("path", "encoding", "delimiter"), ("path",))
    path = resolve_input(c["path"], ws, roots, "csv.path")
    raw = Path(path).read_bytes()
    table = [r for r in _csv.reader(_io.StringIO(raw.decode("utf-8-sig" if c.get("encoding", "utf-8") == "utf-8" else c["encoding"]), newline=""), delimiter=c.get("delimiter", ",")) if r]
    rows = [dict(zip(table[0], r)) for r in table[1:]]
    o = obj(p["output"], "output", ("directory", "name", "format", "overwrite", "jpeg_quality"), ("directory", "format"))
    items = []
    for n_, row in enumerate(rows, start=1):
        for t in p["targets"]:
            obj(t, "target", ("preset", "document", "text_fit"), ("preset", "document"))
            pr = CONTRACT["platform_presets"]["presets"][t["preset"]]
            vals = dict(row, row=str(n_), preset=t["preset"])
            doc = json.loads(json.dumps(t["document"]))
            for e in doc["elements"]:
                if e["type"] == "text":
                    e["text"]["text"] = re.sub(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", lambda m: vals[m.group(1)], e["text"]["text"])
            name = re.sub(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", lambda m: vals[m.group(1)], o.get("name", "{{row}}-{{preset}}"))
            ext = FORMATS[o["format"]][0]
            r = render_one("thumbnail/render", {"document": doc, "output": {"path": os.path.join(o["directory"], name + ext), "format": o["format"], "overwrite": True}}, ws, roots)
            items.append({"row": n_, "preset": t["preset"], "output": r["output"], "format": r["format"], "width": pr["width"], "height": pr["height"], "size": r["size"],
                          "sha256": r["sha256"], "reused": False, "preset_check": {"name": t["preset"], "ok": True}, "fitted": {k: {"font_size": 40} for k in t.get("text_fit") or {}},
                          "identity": r["provenance"]["identity"]})
    return {"items": items, "count": len(items), "rows": len(rows), "targets": [t["preset"] for t in p["targets"]], "operations": ["thumbnail/render_batch"],
            "provenance": dict(base_prov("thumbnail/render_batch"), source={"path": path, "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}, rows=len(rows))}


def render(tool: str, params, ws: str, roots) -> dict:
    """The tool's own response document (never the transport envelope)."""
    if MODE == "internal_error":
        raise Fail("INTERNAL_ERROR", "TypeError: bug")
    if not isinstance(params, dict):
        raise Fail("INVALID_REQUEST", "params must be a JSON object")
    hit = scan({k: v for k, v in params.items() if k != "detections"}, "params")
    if hit:
        raise Fail("INVALID_REQUEST", f"field {hit} is not accepted", {"field": hit, "reason": "forbidden_field"})
    extra = {"thumbnail/choose_frame": choose_frame, "thumbnail/place_text": place_text, "thumbnail/render_layers": render_layers,
             "thumbnail/render_variants": render_variants, "thumbnail/render_batch": render_batch}
    if tool in extra:
        doc = {"schema": RESPONSE, "skill": dict(SKILL), "ok": True, "status": "ok", **extra[tool](params, ws, roots), "warnings": []}
        return doc
    return render_one(tool, params, ws, roots)


def render_one(tool: str, params, ws: str, roots) -> dict:
    if tool == "thumbnail/render":
        p = obj(params, "params", ("schema", "document", "output", "options"), ("document", "output"))
        if "schema" in p and p["schema"] != CONTRACT["document"]["schema"]:
            raise Fail("INVALID_REQUEST", "unsupported request schema", {"field": "schema"})
        doc_id, w, h, assets, fonts = parse_document(p["document"], ws, roots)
        out, fmt = parse_output(p["output"], ws, [a["path"] for a in assets.values()])
        ident_src = {"document_id": doc_id, "assets": {k: {x: y for x, y in v.items() if x != "path"} for k, v in assets.items()}, "fmt": fmt}
        prov_extra = {"engine": "Pillow", "engine_version": "12.3.0-fake", "document_id": doc_id, "assets": list(assets.values()), "fonts": fonts}
        duration = None
    elif tool == "thumbnail/extract_frame":
        p = obj(params, "params", ("source", "output", "options"), ("source", "output"))
        s = obj(p["source"], "source", ("path", "timestamp"), ("path", "timestamp"))
        ts = timestamp(s["timestamp"], "source.timestamp")
        src = resolve_input(s["path"], ws, roots, "source")
        duration = check_frame(src, ts, "source")
        out, fmt = parse_output(p["output"], ws, [src])
        sha = hashlib.sha256(Path(src).read_bytes()).hexdigest()
        ident_src = {"source_sha256": sha, "timestamp": ts, "fmt": fmt}
        prov_extra = {"engine": "ffmpeg-skill/look", "source": {"path": src, "sha256": "sha256:" + sha, "timestamp": ts, "duration": duration}}
        w, h = 1280, 720
    else:
        raise Fail("UNSUPPORTED_OPERATION", f"unknown tool {tool!r}", {"tool": tool})
    if "options" in p:
        obj(p["options"], "options", ("allowed_input_roots", "workspace", "reuse", "timeout", "ffmpeg_skill"), ())
    if MODE == "cancelled":
        raise Fail("CANCELLED", "interrupted", {"reason": "signal"})
    if MODE in ("tool_error", "tool_error_final"):
        raise Fail("TOOL_ERROR", "ffmpeg-skill/look failed", {"reason": "tool_failed", "command": "never forwarded"}, retryable=MODE == "tool_error")
    if MODE == "validation_error":
        raise Fail("VALIDATION_ERROR", "output size 1x1 does not match the canvas", {"reason": "size_mismatch"})
    if MODE == "unknown_code":
        return {"schema": RESPONSE, "skill": dict(SKILL), "ok": False, "status": "error", "error": {"code": "WEIRD", "message": "?", "retryable": True, "details": {}}, "warnings": []}
    payload = PNG_1x1 if fmt == "png" else JPEG_1x1
    reused = MODE == "reused" and os.path.isfile(out)
    if MODE != "output_missing":
        os.makedirs(os.path.dirname(out), exist_ok=True)
        Path(out).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if MODE == "hash_mismatch":
        digest = "0" * 64
    identity = hashlib.sha256(json.dumps({"kind": tool, "skill": SKILL, **ident_src}, sort_keys=True).encode()).hexdigest()
    prov = {"skill": SKILL["id"], "skill_version": SKILL["version"], "operation": tool, **prov_extra, "identity": identity, "reused": reused, "output_hash": "sha256:" + digest}
    doc = {"schema": RESPONSE, "skill": dict(SKILL), "ok": True, "status": "ok", "output": out, "format": fmt, "width": w, "height": h,
           "size": len(payload) + (1 if MODE == "wrong_size" else 0), "sha256": "sha256:" + digest, "reused": reused, "operations": [tool], "provenance": prov, "warnings": []}
    if MODE == "no_provenance":
        doc.pop("provenance")
    return doc


def main() -> int:
    argv = sys.argv[1:]
    cmd = argv[0] if argv else ""
    log(cmd, argv)
    if MODE == "pillow_missing":   # the real Skill imports Pillow at module import time: nothing works, not even the contract
        sys.stderr.write("Traceback (most recent call last):\n  File \"thumbnail_skill/executor.py\", line 23, in <module>\n    import PIL\nModuleNotFoundError: No module named 'PIL'\n")
        return 1
    if cmd in ("contract", "skill"):
        c = json.loads(json.dumps(CONTRACT))
        if MODE == "wrong_schema":
            c["schema"] = "thumbnail-skill/contract@2"
        if MODE == "wrong_skill":
            c["skill_id"] = "other-skill"; c["id"] = "other-skill"
        if MODE == "wrong_version":
            c["version"] = "9.0.0"
        if MODE == "bad_contract":
            c["execution"]["shell"] = True
        if MODE == "contract_fail":
            sys.stderr.write("boom\n")
            return 1
        if MODE == "contract_drift":   # a compatible newer contract: one more registered font (the adapter must notice, never silently keep the pinned expectations)
            c["fonts"]["font_ids"] = sorted(c["fonts"]["font_ids"] + ["emoji"])
            c["fonts"]["registry"]["emoji"] = {"display_name": "Emoji", "role": "colour emoji"}
        emit(c)
        return 0
    if cmd == "doctor":
        status = {"doctor_fail": "fail", "doctor_degraded": "degraded"}.get(MODE, "ok")
        ok = status != "fail"
        fonts = {f: {"font_id": f, "display_name": f, "status": "missing" if (MODE == "doctor_degraded" and f == "cjk") else "available", "path": f"/fake/fonts/{f}.ttf", "index": 0, "sha256": "sha256:" + "f" * 64} for f in FONT_IDS}
        engine = {"status": "ok", "directory": (flag_values(argv, "--ffmpeg-skill") or [None])[0], "version": "0.9.1-fake", "contract_version": "1.0", "problems": []} if ok else \
            {"status": "fail", "directory": (flag_values(argv, "--ffmpeg-skill") or [None])[0], "version": None, "contract_version": None, "problems": ["ffmpeg not found"]}
        ws = (flag_values(argv, "--workspace") or [None])[0]
        emit({"schema": "thumbnail-skill/doctor@1", "skill": dict(SKILL), "status": status, "ok": ok,
              "checks": {"python": {"status": "ok", "version": "3.11.0", "implementation": "CPython", "platform": "Linux"}, "pillow": {"status": "ok", "version": "12.3.0-fake"}, "fonts": fonts,
                         "ffmpeg_skill": engine, "element_types": ["image", "text"], "output_formats": {f: {"status": "ok", "extensions": list(e)} for f, e in FORMATS.items()},
                         "path_policy": {"mode": "allowed_roots" if flag_values(argv, "--allowed-input") else "unrestricted", "workspace": ws, "allowed_input_roots": flag_values(argv, "--allowed-input") or None, "status": "ok"}},
              "problems": [] if ok else ["ffmpeg-skill: ffmpeg not found"], "warnings": ["font_id(s) with no resolvable file on this machine: ['cjk']"] if MODE == "doctor_degraded" else [], "secrets_shown": False})
        return 0 if ok else 1
    if cmd != "run":
        sys.stderr.write("unknown command\n")
        return 2
    if argv[1:2] != ["-"]:
        emit(error_result(Fail("INVALID_REQUEST", "run takes '-' and reads one JSON request from stdin")))
        return EXIT["INVALID_REQUEST"]
    ws = os.path.realpath((flag_values(argv, "--workspace") or [os.getcwd()])[0])
    if not os.path.isdir(ws):
        emit(error_result(Fail("PATH_NOT_ALLOWED", "workspace is not a directory", {"reason": "workspace_missing"})))
        return EXIT["PATH_NOT_ALLOWED"]
    roots = [os.path.realpath(r) for r in flag_values(argv, "--allowed-input")] or None
    try:
        req = json.loads(sys.stdin.read())
    except ValueError:
        emit(error_result(Fail("INVALID_REQUEST", "request document is not valid JSON")))
        return EXIT["INVALID_REQUEST"]
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
    # ---- transport envelope, like run_request in the real Skill
    if not isinstance(req, dict) or set(req) - {"tool", "params"}:
        emit(error_result(Fail("INVALID_REQUEST", "request must be a JSON object with 'tool' and 'params'")))
        return EXIT["INVALID_REQUEST"]
    tool = req.get("tool")
    if tool not in TOOLS:
        emit(error_result(Fail("INVALID_REQUEST", f"'tool' must be one of {TOOLS}", {"tools": TOOLS})))
        return EXIT["INVALID_REQUEST"]
    if MODE == "outer_not_ok":
        emit({"ok": False, "tool": tool, "result": error_result(Fail("INTERNAL_ERROR", "dispatch failed"))})
        return EXIT["INTERNAL_ERROR"]
    try:
        result = render(tool, req.get("params"), ws, roots)
    except Fail as f:
        result = error_result(f)
    if MODE == "wrong_schema":
        result["schema"] = "thumbnail-skill/response@2"
    if MODE == "wrong_skill":
        result["skill"] = {"id": "other-skill", "version": SKILL["version"]}
    if MODE == "wrong_version":
        result["skill"] = {"id": SKILL["id"], "version": "9.0.0"}
    reported_tool = "thumbnail/validate" if MODE == "wrong_tool" else tool
    doc = {"ok": True, "tool": reported_tool, "result": result}
    emit(doc)
    if MODE == "two_docs":
        emit(doc)
    if result.get("ok") is False:
        return EXIT.get(result["error"]["code"], EXIT["INTERNAL_ERROR"])
    return 3 if MODE == "nonzero_ok" else 0


if __name__ == "__main__":
    sys.exit(main())
