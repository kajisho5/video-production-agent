"""Chunked render of a long programme (ADR-134; roadmap step 23): the delivery encode runs chunk by chunk, so a render that stops in chunk 3
resumes at chunk 3 (`render --resume`) instead of encoding the whole programme again.

    programme ─┬─ ffmpeg-skill/cut --start a1 --end a2 --accurate ─ ffmpeg-skill/export --preset P ─┐
               ├─ ffmpeg-skill/cut --start a2 --end a3 --accurate ─ ffmpeg-skill/export --preset P ─┼─ ffmpeg-skill/join --transition none --copy ─ deliverable
               └─ ...                                                                              ─┘

- **Which plans**: one video programme from one source (no concat, camera switch, grid or audio programme), whose delivery target exports
  with a platform or mastering preset (not `copy`: a stream copy has nothing to render), at a constant frame rate the probe measured, and
  with no operation that changes the rate. A programme of `LONG_PROGRAMME_SECONDS` or more is chunked by default; `plan --chunk-seconds N`
  chunks a shorter one too, and `plan --chunk-seconds 0` never chunks. A long plan that cannot be chunked says why in the warnings.
- **Boundaries** are whole frames of the source's rate (`rate`, N/D): boundary k is frame round(k × chunk_seconds × rate), written as its
  time rounded down to the microsecond, so the frame at the boundary starts the next chunk and is never in both. The last chunk runs to the
  end of the programme (`end: null`), so a programme a little longer or shorter than planned loses nothing.
- **Each chunk** is cut frame-accurately (one re-encode at the engine's cut quality, the same generation as a trimmed plan) and exported with
  the target's preset; the exported chunks share one encoding, so they are joined by stream copy (ffmpeg-skill `join --copy`, which checks
  that they match and refuses a join that would gap or shift the sound). Nothing is re-encoded after the join.
- **Resume**: every chunk operation has its own chained idempotency key (the chunk's range is in its arguments), and the job records each
  completed operation as soon as it is done (the checkpoint). A resumed render reuses the chunks whose outputs are intact and starts at the
  first unfinished one.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional

RULE = "chunked_render@1"
LONG_PROGRAMME_SECONDS = 1800.0   # 30 min, the long-input threshold of ADR-132
CHUNK_SECONDS = 600.0             # 10 min: a stop costs at most one chunk's encode
MIN_CHUNK_SECONDS = 10.0
MAX_CHUNK_SECONDS = 7200.0
MAX_CHUNKS = 200
CHUNK_TOOLS = ("cut", "join")     # the engine tools besides the target's export; the join needs `--copy`
MULTI_PROGRAMME_OPS = ("video.concat", "video.switch", "video.grid")


class ChunkError(ValueError):
    pass


def check_chunk_seconds(value: Any) -> Optional[float]:
    """`plan --chunk-seconds`: None (default: chunk a long programme), 0 (never), or MIN..MAX seconds."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ChunkError(f"chunk seconds must be a number, got {value!r}")
    if v == 0:
        return 0.0
    if not MIN_CHUNK_SECONDS <= v <= MAX_CHUNK_SECONDS or math.isnan(v):
        raise ChunkError(f"chunk seconds must be 0 (never chunk) or {MIN_CHUNK_SECONDS:g}-{MAX_CHUNK_SECONDS:g}, got {value!r}")
    return v


def rate_of(video: Optional[Dict[str, Any]]) -> Optional[Fraction]:
    """The source's frame rate as a rational (N/D with D up to 1001, as video-editing turns it into one), or None when not measured."""
    fps = (video or {}).get("fps")
    if isinstance(fps, str) and "/" in fps:
        try:
            n, d = fps.split("/", 1)
            fps = float(n) / float(d)
        except (ValueError, ZeroDivisionError):
            return None
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or not fps or fps <= 0 or math.isnan(float(fps)):
        return None
    return Fraction(float(fps)).limit_denominator(1001)


def _time_of(frame: int, rate: Fraction) -> float:
    """The frame's start time, rounded down to the microsecond (so it never lands after the frame)."""
    return math.floor(Fraction(frame) / rate * 1_000_000) / 1_000_000


def boundaries(programme_seconds: float, chunk_seconds: float, rate: Fraction) -> List[Dict[str, Any]]:
    """[{index, start, end, start_frame}] covering the programme; the last chunk's end is None (to the end)."""
    total_frames = int(math.floor(Fraction(programme_seconds).limit_denominator(1_000_000) * rate + Fraction(1, 1000)))
    starts: List[int] = []
    k = 0
    while True:
        f = int(round(Fraction(k) * Fraction(chunk_seconds).limit_denominator(1_000_000) * rate))
        if k and f >= total_frames:
            break
        starts.append(f)
        k += 1
    # a last chunk shorter than a quarter of a chunk joins the one before it
    if len(starts) > 1 and (total_frames - starts[-1]) < Fraction(chunk_seconds).limit_denominator(1_000_000) * rate / 4:
        starts.pop()
    out = []
    for i, f in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else None
        out.append({"index": i + 1, "start": _time_of(f, rate), "end": _time_of(nxt, rate) if nxt is not None else None, "start_frame": f})
    return out


def plan_chunks(doc: Dict[str, Any], chunk_seconds: Optional[float], supports: Callable[[str], bool]) -> Dict[str, Any]:
    """{"record": <the IR's chunked_render object> | None, "warning": str | None}. `chunk_seconds`: None = default (long programmes only),
    0 = never. `supports(tool)` says whether the engine can run a tool (the join only when it offers `--copy`)."""
    explicit = chunk_seconds is not None
    if chunk_seconds == 0:
        return {"record": None, "warning": None}
    exports = [s for s in (doc.get("plan") or {}).get("steps") or [] if s.get("skill") == "delivery_export" and s.get("outputs")]
    longest = max([float((s.get("temporal_scope") or {}).get("end") or 0.0) for s in exports] or [0.0])
    if not explicit and longest < LONG_PROGRAMME_SECONDS:
        return {"record": None, "warning": None}

    def no(reason: str) -> Dict[str, Any]:
        return {"record": None, "warning": f"render not chunked: {reason}; the delivery is encoded in one pass, and a stopped render starts that encode again"}
    vops = (doc.get("video") or {}).get("operations") or []
    if any(op.get("type") in MULTI_PROGRAMME_OPS for op in vops):
        return no("the programme is joined from several inputs (concat, camera switch or grid)")
    if any(op.get("type") in ("audio.concat", "audio.mix") for op in (doc.get("audio") or {}).get("operations") or []):
        return no("the programme's sound is a joined or mixed audio programme")
    videos = {aid: a for aid, a in (doc.get("assets") or {}).items() if ((a.get("technical") or {}).get("video"))}
    if len(videos) != 1:
        return no(f"chunking takes one video source ({len(videos)} in this plan)")
    subject, asset = next(iter(videos.items()))
    tech = asset.get("technical") or {}
    if (tech.get("video") or {}).get("variable_frame_rate_suspected"):
        return no("the source's frame rate is variable, so chunk boundaries cannot be whole frames")
    rate = rate_of(tech.get("video"))
    if rate is None:
        return no("the source's frame rate was not measured")
    if any(op.get("asset") == subject and op.get("fps") is not None for op in vops):
        return no("an operation changes the frame rate")
    targets = [s for s in exports if (s.get("params") or {}).get("preset") not in (None, "copy") and str(s["outputs"][0]).startswith(f"{subject}_delivery_")]
    if not targets:
        return no("no delivery target re-encodes with a preset (a stream copy has nothing to render)")
    tools = {}
    for s in targets:
        prefix = str(s.get("tool") or "").split("/", 1)[0]
        for name in CHUNK_TOOLS:
            tool = f"{prefix}/{name}"
            if not supports(tool):
                return no(f"the engine cannot run {tool}" + (" with --copy" if name == "join" else ""))
            tools[name] = tool
    size = float(chunk_seconds) if explicit else CHUNK_SECONDS
    programme = longest if longest > 0 else float(tech.get("duration") or 0.0)
    if programme <= 0:
        return no("the programme's duration is not known")
    ranges = boundaries(programme, size, rate)
    if len(ranges) < 2:
        return no(f"the programme ({programme:.1f} s) fits in one chunk of {size:g} s")
    if len(ranges) > MAX_CHUNKS:
        return no(f"{len(ranges)} chunks of {size:g} s is more than {MAX_CHUNKS}; choose longer chunks (plan --chunk-seconds)")
    record = {"rule": RULE, "subject": subject, "rate": f"{rate.numerator}/{rate.denominator}", "chunk_seconds": size,
              "programme_seconds": round(programme, 3), "targets": [s["params"]["target"] for s in targets], "tools": tools,
              "chunks": ranges, "reason": ("chunk length asked for (plan --chunk-seconds)" if explicit else
                                           f"a programme of {LONG_PROGRAMME_SECONDS / 60:.0f} min or more is rendered in {size / 60:g}-minute chunks")}
    return {"record": record, "warning": None}


def summary_line(rec: Dict[str, Any]) -> str:
    return (f"Render {', '.join(rec['targets'])} in {len(rec['chunks'])} chunks of {rec['chunk_seconds']:g} s on whole frames ({rec['rate']} fps), "
            "joined by stream copy; render --resume restarts at the first unfinished chunk")


def progress(rec: Dict[str, Any], ops: List[Any], done: Dict[str, str]) -> Dict[str, Any]:
    """The chunk state of a job: per chunk, its operations and whether each is done (`ran` / `reused`) or not yet.
    `done` maps op id → "ran" | "reused"."""
    rows: Dict[int, Dict[str, Any]] = {c["index"]: {"index": c["index"], "start": c["start"], "end": c["end"], "ops": [], "status": "pending"} for c in rec["chunks"]}
    join_ops = []
    for op in ops:
        c = getattr(op, "chunk", None)
        if c in rows:
            rows[c]["ops"].append({"op": op.id, "tool": op.tool, "output": op.outputs[0] if op.outputs else None, "state": done.get(op.id, "pending")})
        elif getattr(op, "chunk_join", False):
            join_ops.append({"op": op.id, "tool": op.tool, "output": op.outputs[0] if op.outputs else None, "state": done.get(op.id, "pending")})
    for r in rows.values():
        states = [o["state"] for o in r["ops"]]
        if states and all(s == "reused" for s in states):
            r["status"] = "reused"
        elif states and all(s in ("ran", "reused") for s in states):
            r["status"] = "done"
        elif any(s in ("ran", "reused") for s in states):
            r["status"] = "partial"
    chunks = [rows[k] for k in sorted(rows)]
    first = next((c["index"] for c in chunks if c["status"] not in ("reused",)), None)
    return {"rule": rec["rule"], "chunks": chunks, "join": join_ops, "total": len(chunks),
            "reused": sum(1 for c in chunks if c["status"] == "reused"), "done": sum(1 for c in chunks if c["status"] in ("done", "reused")),
            "started_at": first}
