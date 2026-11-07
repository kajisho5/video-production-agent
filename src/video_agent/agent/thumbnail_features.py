"""thumbnail-skill 0.1.12 .. 0.1.15 in the plan (ADR-104): the frame chosen by a face ranking, the caption moved clear of faces, A/B variants,
layered output and a table-driven batch per platform preset. Explicit requirements only (`--set` / a profile; the table path never from a brief);
the default thumbnail (one frame, optional caption / texts / logo, ADR-031 / ADR-088) is unchanged.

- **Frame by faces** (`thumbnail.at=faces`, + `thumbnail.faces.every` seconds between sampled frames, default FACES_EVERY_DEFAULT, and
  `thumbnail.faces.min_score`): detection-skill measures each picture source's faces with their attributes (analysis kind `face_attributes`),
  thumbnail-skill choose_frame ranks those frames (kind `frame_ranking`: eyes open first, then its stated weighted score). The planner takes
  the first ranked frame (over every source of the subject, in the Skill's own order) whose time survives the edit, on the delivered timeline.
  No ranked frame with a face → BLOCK (never a guessed frame); no detection-skill → the measurement is handed off (BLOCK naming it).
- **Caption clear of faces** (`thumbnail.avoid_faces=true`, + `.margin` px): thumbnail-skill place_text moves the caption to the first of its
  candidate positions whose painted box stays on the canvas and clears every face box of the chosen frame; refused when none is clear. Needs
  `thumbnail.text` and `thumbnail.at=faces` (the face boxes are those of the ranked frame) and chooses the position itself (`thumbnail.position`
  is refused with it). The boxes are mapped from the source frame to the picture only when the edit kept the picture's geometry (no crop / turn
  / punch in / fit / fill / stabilize / grid / switch, the source's aspect ratio); otherwise BLOCK.
- **Variants** (`thumbnail.variants=N`, 2..12, + `.frames` (more timestamps on the delivered timeline), `.phone_width`, `.min_text_height`,
  `.min_contrast`): thumbnail-skill render_variants writes N documented variants of the captioned thumbnail; variant 1 is the document as given
  and stays the plan's thumbnail (publish, QC package); variants 2..N are further THUMBNAIL artifacts. A variant below the phone-size check is
  reported (QA WARN), never dropped; the agent never picks a winner.
- **Layers** (`thumbnail.layers=true`): thumbnail-skill render_layers also writes every composited layer as a PNG with a manifest
  (THUMBNAIL_LAYERS artifact) that rebuilds the thumbnail pixel for pixel.
- **Batch** (`thumbnail.batch=<table.csv>` + `.presets` (1..4 of the contract's platform presets, required), `.encoding` utf-8 (default) | cp932
  (opt-in, a CSV saved by Japanese Excel), `.delimiter` , | ; | tab, `.min_font_size` (shrink each text to 90 % of the canvas width, never
  below this), `.name` (file name template, default {{row}}-{{preset}})): one thumbnail per table row and preset from the planned document,
  laid out on each preset's canvas; `{{column}}` placeholders in thumbnail.text / text2..4. Replaces the single thumbnail (publish with
  `publish.thumbnail=true` is refused: several files). The table is read at plan time (its rows name the files) and bound by its sha256.

variants / layers / batch exclude each other (one thumbnail-skill tool per thumbnail); batch excludes avoid_faces (one face layout per canvas)."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .editing import EditRequirementError, _bool, _number, _token

THUMBNAIL_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "tools" / "thumbnail" / "contract_0.1.15.json"
_CONTRACT: Dict[str, Any] = json.loads(THUMBNAIL_CONTRACT_PATH.read_text(encoding="utf-8"))
PRESETS: Dict[str, Dict[str, Any]] = dict(_CONTRACT["platform_presets"]["presets"])
_BATCH = _CONTRACT["batch"]
BATCH_LIMITS = dict(_BATCH["limits"])                    # rows / targets / cell_chars
BATCH_MAX_BYTES = int(_BATCH["csv"]["max_bytes"])
BATCH_ENCODINGS = tuple(_BATCH["csv"]["encodings"])      # utf-8 (default), cp932 (opt-in)
BATCH_DELIMITERS = {",": ",", ";": ";", "tab": "\t"}
BATCH_BUILTINS = tuple(_BATCH["placeholders"]["built_in"])   # row, preset
BATCH_NAME_DEFAULT = str(_BATCH["output_name"]["default"])
BATCH_NAME_RE = re.compile(_BATCH["output_name"]["pattern"])
BATCH_FIT_WIDTH = 0.9                                    # text_fit max_width as a share of the canvas width
PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
MAX_VARIANTS = int(_CONTRACT["variants"]["max_variants"])
FACES_EVERY_DEFAULT = 1.0     # detection-skill's own CLI default for finding (ADR-090 uses 0.5 for hiding)
FRAME_SKILLS = ("face_attributes", "thumbnail_frame_ranking")
# the edits after which a box measured on the source frame is no longer where the picture shows it
GEOMETRY_SUBJECTS = ("video.switch", "video.grid", "video.stabilize", "video.crop", "video.rotate", "video.punch_in", "video.fit", "video.fill")

FACE_KEYS = ("thumbnail.faces.every", "thumbnail.faces.min_score")
AVOID_KEYS = ("thumbnail.avoid_faces", "thumbnail.avoid_faces.margin")
VARIANT_KEYS = ("thumbnail.variants", "thumbnail.variants.frames", "thumbnail.variants.phone_width", "thumbnail.variants.min_text_height", "thumbnail.variants.min_contrast")
LAYER_KEYS = ("thumbnail.layers",)
BATCH_KEYS = ("thumbnail.batch", "thumbnail.batch.presets", "thumbnail.batch.encoding", "thumbnail.batch.delimiter", "thumbnail.batch.min_font_size", "thumbnail.batch.name")
FEATURE_KEYS = FACE_KEYS + AVOID_KEYS + VARIANT_KEYS + LAYER_KEYS + BATCH_KEYS
PATH_KEYS = ("thumbnail.batch",)   # a file the user names: never read from a brief


def _list(v: Any, key: str) -> List[Any]:
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return [v]
    raise EditRequirementError(f"{key} must be a list (or a comma-separated text), got {v!r}")


def read_batch_table(path: str, encoding: str, delimiter: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """The table thumbnail-skill's render_batch reads, read the same way (its contract's `batch.csv`): header row of unique non-empty names,
    blank lines skipped, every line as many fields as the header, at most the contract's rows / bytes / cell length, no column named as a
    built-in placeholder. Read at plan time because its rows name the files the plan produces; the Skill checks it again before writing."""
    if not os.path.isfile(path):
        raise EditRequirementError(f"thumbnail.batch: table not found: {os.path.basename(path)}")
    if os.path.getsize(path) > BATCH_MAX_BYTES:
        raise EditRequirementError(f"thumbnail.batch: the table is larger than {BATCH_MAX_BYTES} bytes")
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig" if encoding == "utf-8" else encoding)
    except UnicodeDecodeError:
        raise EditRequirementError(f"thumbnail.batch: the table is not valid {encoding}" + (" (a CSV saved by Japanese Excel is usually cp932: set thumbnail.batch.encoding=cp932)"
                                                                                             if encoding == "utf-8" else ""))
    try:
        table = [r for r in csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True) if r]
    except csv.Error as e:
        raise EditRequirementError(f"thumbnail.batch: the table is malformed: {e}")
    if not table:
        raise EditRequirementError("thumbnail.batch: the table has no header row")
    header = [h.strip() for h in table[0]]
    if any(not h for h in header) or len(set(header)) != len(header):
        raise EditRequirementError(f"thumbnail.batch: the header must have unique, non-empty column names, got {header}")
    reserved = [h for h in header if h in BATCH_BUILTINS]
    if reserved:
        raise EditRequirementError(f"thumbnail.batch: column {reserved[0]!r} is reserved (a built-in placeholder)")
    rows: List[Dict[str, str]] = []
    for n, r in enumerate(table[1:], start=2):
        if len(r) != len(header):
            raise EditRequirementError(f"thumbnail.batch: line {n} has {len(r)} field(s), the header has {len(header)}")
        if any(len(c) > int(BATCH_LIMITS["cell_chars"]) for c in r):
            raise EditRequirementError(f"thumbnail.batch: a cell on line {n} is longer than {BATCH_LIMITS['cell_chars']} characters")
        rows.append(dict(zip(header, r)))
    if not rows:
        raise EditRequirementError("thumbnail.batch: the table has no rows")
    if len(rows) > int(BATCH_LIMITS["rows"]):
        raise EditRequirementError(f"thumbnail.batch: the table has {len(rows)} rows; at most {BATCH_LIMITS['rows']} per batch")
    return header, rows


def _fill(template: str, values: Dict[str, str], key: str) -> str:
    def one(m: "re.Match[str]") -> str:
        if m.group(1) not in values:
            raise EditRequirementError(f"{key}: placeholder {{{{{m.group(1)}}}}} names no column of the table (columns: {sorted(k for k in values if k not in BATCH_BUILTINS)})")
        return values[m.group(1)]
    return PLACEHOLDER_RE.sub(one, template)


def parse_thumbnail_features(vals: Dict[str, Any], out: Dict[str, Any], fmt: Optional[str]) -> None:
    """The ADR-104 keys of an enabled thumbnail request (`vals`: key → value of the explicit requirements) into `out` (parse_thumbnail_requirements'
    result, its caption / texts already parsed): at_faces, faces, avoid_faces, variants, layers, batch. Anything ambiguous is refused."""
    at = vals.get("thumbnail.at")
    out["at_faces"] = isinstance(at, str) and at.strip().lower() == "faces"
    given = [k for k in FACE_KEYS if k in vals]
    avoid = _bool(vals["thumbnail.avoid_faces"], "thumbnail.avoid_faces") if "thumbnail.avoid_faces" in vals else False
    if "thumbnail.avoid_faces.margin" in vals and not avoid:
        raise EditRequirementError("thumbnail.avoid_faces.margin is set but thumbnail.avoid_faces is not: refusing an ambiguous request")
    if given and not out["at_faces"]:
        raise EditRequirementError(f"{given[0]} samples the faces of thumbnail.at=faces, which is not set: refusing an ambiguous request")
    if out["at_faces"]:
        f: Dict[str, Any] = {"every": float(_number(vals["thumbnail.faces.every"], "thumbnail.faces.every", 0.05, 60.0)) if "thumbnail.faces.every" in vals else FACES_EVERY_DEFAULT}
        if "thumbnail.faces.min_score" in vals:
            f["min_score"] = float(_number(vals["thumbnail.faces.min_score"], "thumbnail.faces.min_score", 0.0, 1.0))
        out["faces"] = f
    if avoid:
        if out.get("text") is None:
            raise EditRequirementError("thumbnail.avoid_faces moves the caption: it needs thumbnail.text")
        if not out["at_faces"]:
            raise EditRequirementError("thumbnail.avoid_faces uses the face boxes of the ranked frame: it needs thumbnail.at=faces")
        if "thumbnail.position" in vals:
            raise EditRequirementError("thumbnail.avoid_faces chooses the caption's position: thumbnail.position is refused with it")
        out["avoid_faces"] = {"margin": int(_number(vals["thumbnail.avoid_faces.margin"], "thumbnail.avoid_faces.margin", 0, 1000))} if "thumbnail.avoid_faces.margin" in vals else {}
    # ---- variants
    vk = [k for k in VARIANT_KEYS[1:] if k in vals]
    if "thumbnail.variants" not in vals:
        if vk:
            raise EditRequirementError(f"{vk[0]} is set but thumbnail.variants is not: refusing an ambiguous request")
    else:
        n = _number(vals["thumbnail.variants"], "thumbnail.variants", 2, MAX_VARIANTS)
        if float(n) != int(n):
            raise EditRequirementError(f"thumbnail.variants must be a whole number of variants (2..{MAX_VARIANTS})")
        if out.get("text") is None:
            raise EditRequirementError("thumbnail.variants vary the caption's layout and palette: they need thumbnail.text")
        v: Dict[str, Any] = {"count": int(n)}
        if "thumbnail.variants.frames" in vals:
            frames = [float(_number(x, "thumbnail.variants.frames", 0.0, 86400.0)) for x in _list(vals["thumbnail.variants.frames"], "thumbnail.variants.frames")]
            if not 1 <= len(frames) <= MAX_VARIANTS - 1:
                raise EditRequirementError(f"thumbnail.variants.frames lists 1..{MAX_VARIANTS - 1} timestamps")
            v["frames"] = frames
        for k, name, lo, hi in (("thumbnail.variants.phone_width", "phone_width", 100, 4000), ("thumbnail.variants.min_text_height", "min_text_height", 0, 400),
                                ("thumbnail.variants.min_contrast", "min_contrast", 1, 21)):
            if k in vals:
                x = _number(vals[k], k, lo, hi)
                v[name] = int(x) if name == "phone_width" else float(x)
        out["variants"] = v
    if "thumbnail.layers" in vals and _bool(vals["thumbnail.layers"], "thumbnail.layers"):
        out["layers"] = True
    # ---- batch
    bk = [k for k in BATCH_KEYS[1:] if k in vals]
    texts = ([("thumbnail.text", out["text"])] if out.get("text") is not None else []) + [(f"thumbnail.text{n}", t["text"]) for n, t in enumerate(out.get("texts") or [], start=2)]
    placeholders = sorted({m for _k, t in texts for m in PLACEHOLDER_RE.findall(t)})
    if "thumbnail.batch" not in vals:
        if bk:
            raise EditRequirementError(f"{bk[0]} is set but thumbnail.batch is not: refusing an ambiguous request")
        if placeholders:
            raise EditRequirementError(f"{{{{{placeholders[0]}}}}} is a table placeholder: it needs thumbnail.batch (the table)")
    else:
        path = vals["thumbnail.batch"]
        if not isinstance(path, str) or not path.strip() or any(part == ".." for part in path.replace("\\", "/").split("/")):
            raise EditRequirementError("thumbnail.batch must be a path to a table (.csv) without '..'")
        if "thumbnail.batch.presets" not in vals:
            raise EditRequirementError("thumbnail.batch needs thumbnail.batch.presets (the platforms, e.g. youtube,youtube-shorts)")
        presets = [str(p) for p in _list(vals["thumbnail.batch.presets"], "thumbnail.batch.presets")]
        if not 1 <= len(presets) <= int(BATCH_LIMITS["targets"]) or len(set(presets)) != len(presets):
            raise EditRequirementError(f"thumbnail.batch.presets lists 1..{BATCH_LIMITS['targets']} different presets")
        for p in presets:
            if p not in PRESETS:
                raise EditRequirementError(f"thumbnail.batch.presets: {p!r} is not one of the contract's presets {sorted(PRESETS)}")
            if fmt is not None and fmt not in PRESETS[p]["formats"]:
                raise EditRequirementError(f"thumbnail.batch.presets: {p} accepts {PRESETS[p]['formats']}, not {fmt} (set thumbnail.format)")
        enc = _token(vals.get("thumbnail.batch.encoding", "utf-8"), "thumbnail.batch.encoding", re.compile("^(" + "|".join(BATCH_ENCODINGS) + ")$"), " or ".join(BATCH_ENCODINGS))
        delim = vals.get("thumbnail.batch.delimiter", ",")
        if delim not in BATCH_DELIMITERS:
            raise EditRequirementError(f"thumbnail.batch.delimiter must be one of {list(BATCH_DELIMITERS)}, got {delim!r}")
        name = vals.get("thumbnail.batch.name", BATCH_NAME_DEFAULT)
        if not isinstance(name, str) or not name.strip():
            raise EditRequirementError("thumbnail.batch.name must be a file name template")
        columns, rows = read_batch_table(path, enc, BATCH_DELIMITERS[delim])
        if len(rows) * len(presets) > int(BATCH_LIMITS["rows"]):
            raise EditRequirementError(f"thumbnail.batch: {len(rows)} rows x {len(presets)} presets is more than {BATCH_LIMITS['rows']} files")
        files: List[Dict[str, Any]] = []
        seen: Dict[str, Tuple[int, str]] = {}
        for k, t in texts:
            _fill(t, {**{c: "" for c in columns}, **{b: "" for b in BATCH_BUILTINS}}, k)
        for n_, row in enumerate(rows, start=1):
            for p in presets:
                fname = _fill(name, {**row, "row": str(n_), "preset": p}, "thumbnail.batch.name")
                if not BATCH_NAME_RE.match(fname):
                    raise EditRequirementError(f"thumbnail.batch.name gives {fname!r} for row {n_}, not a safe file name ([A-Za-z0-9._-], starting with a letter or digit)")
                if fname in seen:
                    raise EditRequirementError(f"thumbnail.batch.name gives {fname!r} twice (rows {seen[fname][0]} and {n_})")
                seen[fname] = (n_, p)
                files.append({"row": n_, "preset": p, "name": fname})
        b: Dict[str, Any] = {"csv": str(Path(path).resolve()), "csv_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "encoding": enc, "delimiter": delim,
                             "presets": presets, "name": name, "rows": len(rows), "columns": columns, "files": files}
        if "thumbnail.batch.min_font_size" in vals:
            b["min_font_size"] = int(_number(vals["thumbnail.batch.min_font_size"], "thumbnail.batch.min_font_size", 6, 400))
        out["batch"] = b
    chosen = [k for k in ("variants", "layers", "batch") if out.get(k)]
    if len(chosen) > 1:
        raise EditRequirementError(f"thumbnail.{chosen[0]} and thumbnail.{chosen[1]} are different thumbnail-skill tools: ask for one")
    if out.get("batch") and out.get("avoid_faces") is not None:
        raise EditRequirementError("thumbnail.avoid_faces places one caption on one canvas: it does not apply to a batch over several presets")


def inside_keep(t: float, keep: List[List[float]]) -> Optional[float]:
    """A source time inside a kept range → its time on the trimmed timeline; None when the edit removed it (never snapped to a neighbour)."""
    acc = 0.0
    for s, e in keep:
        if s <= t <= e:
            return round(acc + (t - s), 3)
        acc += e - s
    return None


def choose_ranked_frame(rankings: List[Tuple[str, Any]], timeline_of: Any) -> Optional[Dict[str, Any]]:
    """The thumbnail frame from thumbnail-skill's rankings of the subject's sources (`rankings`: (source asset id, frame_ranking observation)
    in programme order): every ranked frame with a face, in the Skill's own order (eyes open first, then score high first) with the source
    order and the rank breaking ties; the first whose time `timeline_of(source, t)` keeps on the delivered timeline (None when removed)."""
    cands: List[Tuple[Any, ...]] = []
    for n, (src, obs) in enumerate(rankings):
        for e in (obs.data or {}).get("ranking") or []:
            if e.get("rank") is None or e.get("t") is None or not e.get("faces"):
                continue
            cands.append((bool(e.get("eyes_closed")), -float(e.get("score") or 0.0), n, int(e["rank"]), src, obs, e))
    for _ec, _s, _n, _r, src, obs, e in sorted(cands, key=lambda c: c[:4]):
        at = timeline_of(src, float(e["t"]))
        if at is not None:
            return {"source": src, "observation": obs.id, "source_time": float(e["t"]), "timestamp": at, "rank": int(e["rank"]), "score": e.get("score"),
                    "eyes_closed": bool(e.get("eyes_closed")), "parts": e.get("parts"), "missing": e.get("missing"), "faces": [list(f) for f in e.get("faces") or []],
                    "frame_size": dict((obs.data or {}).get("frame_size") or {})}
    return None


def faces_on_picture(choice: Dict[str, Any], picture: Tuple[int, int]) -> Optional[List[List[float]]]:
    """The chosen frame's face boxes from the source frame's pixels onto the picture's: a uniform scale only (the same aspect ratio within
    half a percent); None when the picture is not a scaled source frame."""
    fs = choice.get("frame_size") or {}
    fw, fh = fs.get("width"), fs.get("height")
    if not fw or not fh or not picture[0] or not picture[1]:
        return None
    sx, sy = picture[0] / float(fw), picture[1] / float(fh)
    if abs(sx - sy) > 0.005 * max(sx, sy):
        return None
    return [[round(b[0] * sx, 2), round(b[1] * sy, 2), round(b[2] * sx, 2), round(b[3] * sy, 2)] for b in choice.get("faces") or []]


def thumbnail_faces_kinds(reqs: Dict[str, Any]) -> Optional[Tuple[List[str], Dict[str, float]]]:
    """(analysis kinds, their parameters) when the request asks for the thumbnail frame by faces (thumbnail.at=faces); the keys were range-checked
    by the --set gate (parse_thumbnail_requirements) before this is asked."""
    at = reqs.get("thumbnail.at")
    if reqs.get("thumbnail") in (None, False, "false", "no", "off", "0") or not (isinstance(at, str) and at.strip().lower() == "faces"):
        return None
    p: Dict[str, float] = {"frame_every": float(reqs.get("thumbnail.faces.every", FACES_EVERY_DEFAULT))}
    if reqs.get("thumbnail.faces.min_score") is not None:
        p["frame_min_score"] = float(reqs["thumbnail.faces.min_score"])
    return ["face_attributes", "frame_ranking"], p
