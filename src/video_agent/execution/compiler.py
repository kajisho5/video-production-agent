"""Compiler: Project IR → ordered Operations with typed adapter args. Fixed Phase 1 order per asset:
trim → loudness → export → check. Paths for intermediates are decided here; no ffmpeg flags appear here.

Idempotency keys are chained: key(op) = H(source fingerprint, tool, args, tool version, keys of the ops that
produced its inputs). Changing the trim therefore changes the loudness key too, so a resumed job can never reuse
an output whose upstream changed. Operation ids are deterministic (H(tool, args, inputs)) so provenance and
job records can be matched across compiles."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models import Operation, stable_hash
from ..project.ir import ProjectIR


def tool_version_of(versions: Dict[str, str], tool: str) -> str:
    """Version of the adapter that owns a tool id ("<adapter>/<tool>")."""
    return str(versions.get(tool.split("/", 1)[0], ""))


def _op_id(tool: str, args: Dict[str, Any], inputs: List[str]) -> str:
    return "op_" + stable_hash([tool, args, inputs])[:12]


def source_fingerprint(asset: Dict[str, Any]) -> str:
    """sha256 when the analyzer hashed the file; otherwise size+mtime (weaker, but still detects replaced sources)."""
    if asset.get("hash"):
        return "sha256:" + asset["hash"]
    f = (asset.get("technical") or {}).get("file") or {}
    if f.get("size") is not None and f.get("mtime") is not None:
        return f"stat:{f['size']}:{f['mtime']}"
    return "unknown:" + asset.get("path", "")


class CompileError(ValueError):
    pass


def _step_tools(d: Dict[str, Any]) -> Dict[Tuple[str, str], str]:
    """(skill, asset-or-target key) → tool id, from plan.steps. The compiler never chooses tools itself."""
    out: Dict[Tuple[str, str], str] = {}
    for s in d["plan"]["steps"]:
        p = s.get("params") or {}
        key = p.get("asset") or p.get("target") or ""
        if s.get("tool"):
            out[(s["skill"], key)] = s["tool"]
    return out


def compile_ir(ir: ProjectIR, job_dir: str, tool_versions: Optional[Dict[str, str]] = None) -> Tuple[List[Operation], Dict[str, str]]:
    """Returns (operations, paths) where paths maps artifact ids to filesystem paths. `tool_versions` maps an adapter
    name (the tool id prefix, e.g. "ffmpeg-skill") to its version; it defaults to the IR's recorded source.tool_versions."""
    d = ir.doc
    versions = tool_versions if tool_versions is not None else (d.get("source", {}).get("tool_versions") or {})
    step_tools = _step_tools(d)

    def tool_for(skill: str, key: str) -> str:
        t = step_tools.get((skill, key))
        if not t:
            raise CompileError(f"plan has no tool for skill {skill} ({key}); the plan must name a selected tool for every operation")
        return t
    ops: List[Operation] = []
    paths: Dict[str, str] = {}
    keys: Dict[str, str] = {}      # artifact id -> idempotency key of the op that produces it ("" for sources)
    job = Path(job_dir)

    def add(tool: str, args: Dict[str, Any], inputs: List[str], outputs: List[str], decision_ids: List[str], fp: str, kind: str = "transform", skill: str = "") -> Operation:
        o = Operation(tool=tool, args=args, inputs=inputs, outputs=outputs, decision_ids=decision_ids, kind=kind, skill=skill, id=_op_id(tool, args, inputs))
        if outputs:
            o.idempotency_key = stable_hash([fp, tool, args, tool_version_of(versions, tool), [keys.get(i, "") for i in inputs]])
            for out in outputs:
                keys[out] = o.idempotency_key
        ops.append(o)
        return o

    for idx, (asset_id, asset) in enumerate(d["assets"].items(), start=1):
        paths[asset_id] = asset["path"]
        keys[asset_id] = ""
        fp = source_fingerprint(asset)
        stem = f"{idx:02d}_{Path(asset['path']).stem}"  # index keeps two inputs with the same file name apart
        current = asset_id
        gen = 0
        for op in d["video"]["operations"]:
            if op["asset"] != asset_id or op["type"] != "video.trim":
                continue
            gen += 1
            out_id = f"{asset_id}_trim"
            paths[out_id] = str(job / "ops" / f"{stem}_{gen:02d}_trim" / f"{stem}_trim.mp4")
            segs = ",".join(f"{s:.3f}-{e:.3f}" for s, e in op["keep"])
            args: Dict[str, Any] = {"input": current, "segments": segs, "output": out_id}
            if op.get("accurate"):
                args["accurate"] = True   # frame-accurate cut is part of the plan content (hashed, diffed), not a side channel
            add(tool_for("silence_cleanup", asset_id), args, [current], [out_id], list(op.get("decision_ids") or []), fp, skill="silence_cleanup")
            current = out_id
        for op in d["audio"]["operations"]:
            if op["asset"] != asset_id or op["type"] != "audio.loudness":
                continue
            gen += 1
            out_id = f"{asset_id}_loudnorm"
            paths[out_id] = str(job / "ops" / f"{stem}_{gen:02d}_loudness" / f"{stem}_loudnorm.mp4")
            args = {"input": current, "lufs": op["target_lufs"], "tp": op["true_peak"], "output": out_id}
            add(tool_for("loudness_normalization", asset_id), args, [current], [out_id], list(op.get("decision_ids") or []), fp, skill="loudness_normalization")
            current = out_id
        for t in d["delivery"]["targets"]:
            art_id = f"{asset_id}_delivery_{t['id']}"
            if t.get("preset"):
                ext = {"prores": "mov", "gif": "gif"}.get(t["preset"], "mp4")
                paths[art_id] = str(job / "artifacts" / f"{stem}_{t['id']}.{ext}")
                args = {"input": current, "preset": t["preset"], "output": art_id}
                add(tool_for("delivery_export", t["id"]), args, [current], [art_id], list(t.get("decision_ids") or []), fp, skill="delivery_export")
                add(tool_for("delivery_check", t["id"]), {"input": art_id, "platform": t.get("platform", "custom")}, [art_id], [], list(t.get("decision_ids") or []), fp, kind="qa", skill="delivery_check")
            elif current != asset_id:
                # generic profile: the last processed intermediate is the deliverable (no re-encode)
                paths[art_id] = paths[current]
    return ops, paths
