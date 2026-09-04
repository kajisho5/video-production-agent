"""Compiler: Project IR → ordered Operations with typed adapter args. Fixed Phase 1 order per asset:
trim → loudness → export → check. Paths for intermediates are decided here; no ffmpeg flags appear here."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..models import Operation, stable_hash
from ..project.ir import ProjectIR


def compile_ir(ir: ProjectIR, job_dir: str, tool_version: str = "") -> Tuple[List[Operation], Dict[str, str]]:
    """Returns (operations, paths) where paths maps artifact ids to filesystem paths."""
    d = ir.doc
    ops: List[Operation] = []
    paths: Dict[str, str] = {}
    job = Path(job_dir)
    frame_accurate = any(r.get("key") == "edit.precision" and r.get("value") == "frame" for r in d["requirements"])
    for asset_id, asset in d["assets"].items():
        paths[asset_id] = asset["path"]
        current = asset_id
        src_hash = asset.get("hash") or ""
        gen = 0
        for op in d["video"]["operations"]:
            if op["asset"] != asset_id or op["type"] != "video.trim":
                continue
            gen += 1
            out_id = f"{asset_id}_trim"
            paths[out_id] = str(job / "ops" / f"{gen:02d}_trim" / f"{Path(asset['path']).stem}_trim.mp4")
            segs = ",".join(f"{s:.3f}-{e:.3f}" for s, e in op["keep"])
            args: Dict[str, Any] = {"input": current, "segments": segs, "output": out_id}
            if frame_accurate:
                args["accurate"] = True
            o = Operation(tool="ffmpeg-skill/cut", args=args, inputs=[current], outputs=[out_id], decision_ids=list(op.get("decision_ids") or []))
            o.idempotency_key = stable_hash([src_hash, o.tool, args, tool_version])
            ops.append(o)
            current = out_id
        for op in d["audio"]["operations"]:
            if op["asset"] != asset_id or op["type"] != "audio.loudness":
                continue
            gen += 1
            out_id = f"{asset_id}_loudnorm"
            paths[out_id] = str(job / "ops" / f"{gen:02d}_loudness" / f"{Path(asset['path']).stem}_loudnorm.mp4")
            args = {"input": current, "lufs": op["target_lufs"], "tp": op["true_peak"], "output": out_id}
            o = Operation(tool="ffmpeg-skill/loudness", args=args, inputs=[current], outputs=[out_id], decision_ids=list(op.get("decision_ids") or []))
            o.idempotency_key = stable_hash([src_hash, o.tool, args, tool_version])
            ops.append(o)
            current = out_id
        for t in d["delivery"]["targets"]:
            art_id = f"{asset_id}_delivery_{t['id']}"
            if t.get("preset"):
                gen += 1
                ext = {"prores": "mov", "gif": "gif"}.get(t["preset"], "mp4")
                paths[art_id] = str(job / "artifacts" / f"{Path(asset['path']).stem}_{t['id']}.{ext}")
                args = {"input": current, "preset": t["preset"], "output": art_id}
                o = Operation(tool="ffmpeg-skill/export", args=args, inputs=[current], outputs=[art_id], decision_ids=list(t.get("decision_ids") or []))
                o.idempotency_key = stable_hash([src_hash, o.tool, args, tool_version])
                ops.append(o)
                q = Operation(tool="ffmpeg-skill/check", args={"input": art_id, "platform": t.get("platform", "custom")}, inputs=[art_id], outputs=[], decision_ids=list(t.get("decision_ids") or []), kind="qa")
                ops.append(q)
            elif current != asset_id:
                # generic profile: the last processed intermediate is the deliverable (no re-encode)
                paths[art_id] = paths[current]
    return ops, paths
