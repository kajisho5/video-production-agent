"""Compiler: Project IR → ordered Operations with typed adapter args. Fixed Phase 1 order per asset:
trim → loudness → export → check. Paths for intermediates are decided here; no engine flags appear here.

Idempotency keys are chained: key(op) = H(source fingerprint, tool, args, tool version, keys of the ops that
produced its inputs). Changing the trim therefore changes the loudness key too, so a resumed job can never reuse
an output whose upstream changed. Operation ids are deterministic (H(tool, args, inputs)) so provenance and
job records can be matched across compiles."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..agent.audio import AUDIO_ORDER, OPERATIONS as AUDIO_OPERATIONS, PROGRAMME_AUDIO, TOOL as AUDIO_TOOL
from ..agent.editing import EDIT_ORDER, OPERATIONS, PROGRAMME
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


def lower_video_trim(tool: str, op: Dict[str, Any], current: str, out_id: str) -> Dict[str, Any]:
    """IR `video.trim` (keep ranges on one asset) → the typed arguments of the tool the plan selected. The meaning is the
    same for every tool: keep exactly these ranges, in this order, everything else is removed; `accurate` is the plan's
    precision requirement. The reference cut tool takes them as `segments` (+ `accurate`); video-editing/cut (ADR-028) as
    the contract's CUT parameters `keep` [{start, end}] and `precision` frame | keyframe. The tool itself comes from the
    plan (registry-selected); the compiler never chooses an engine and never falls back to another one."""
    keep = [[round(float(s), 3), round(float(e), 3)] for s, e in op["keep"]]
    if tool == "video-editing/cut":
        return {"input": current, "keep": keep, "precision": "frame" if op.get("accurate") else "keyframe", "output": out_id}
    # the reference catalog shape (any adapter implementing the typed `cut` catalog): unchanged since Phase 1
    args: Dict[str, Any] = {"input": current, "segments": ",".join(f"{s:.3f}-{e:.3f}" for s, e in keep), "output": out_id}
    if op.get("accurate"):
        args["accurate"] = True   # frame-accurate cut is part of the plan content (hashed, diffed), not a side channel
    return args


def lower_video_edit(tool: str, op: Dict[str, Any], current: str, out_id: str, extra_refs: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """IR editing operation (ADR-029: video.speed / resize / fit / fill / overlay / concat) → the typed arguments of the selected
    video-editing tool. Only the operation's allowlisted parameters are copied, by name; references become artifact ids the
    adapter resolves through the paths map. The compiler never invents a parameter, a filter or a command, and refuses a
    tool that does not belong to the operation (the plan names the tool; the compiler only checks the pairing)."""
    spec = OPERATIONS[op["type"]]
    if tool != spec["tool"]:
        raise CompileError(f"{op['type']} cannot be executed by {tool}; the only tool of this operation is {spec['tool']}")
    args: Dict[str, Any] = {}
    if op["type"] == "video.concat":
        args["inputs"] = list(current if isinstance(current, list) else [current])
    else:
        args["input"] = current
    for k in spec["params"]:
        if op.get(k) is not None:
            args[k] = op[k]
    for k, v in (extra_refs or {}).items():
        args[k] = v
    args["output"] = out_id
    return args


def lower_audio_loudness(tool: str, op: Dict[str, Any], current: str, out_id: str) -> Dict[str, Any]:
    """IR `audio.loudness` → the typed arguments of the tool the plan selected: the reference loudness tool takes {input, lufs,
    tp, output}; audio-production/run (ADR-030) takes the contract's NORMALIZE parameters (target_lufs, true_peak_db, the
    tolerance the Skill re-measures against, an optional output sample rate). The compiler never chooses the engine."""
    if tool == AUDIO_TOOL:
        args: Dict[str, Any] = {"operation": "NORMALIZE", "input": current, "target_lufs": float(op["target_lufs"]), "true_peak_db": float(op["true_peak"]), "output": out_id}
        if op.get("tolerance_lu") is not None:
            args["tolerance_lufs"] = float(op["tolerance_lu"])
        if op.get("sample_rate") is not None:
            args["sample_rate"] = int(op["sample_rate"])
        return args
    return {"input": current, "lufs": op["target_lufs"], "tp": op["true_peak"], "output": out_id}


def lower_audio_op(tool: str, op: Dict[str, Any], current: Any, out_id: str) -> Dict[str, Any]:
    """IR audio operation (audio.cut / concat / gain / mono / stereo / downmix / fade_in / fade_out) → audio-production/run
    arguments: the contract's operation type plus the allowlisted parameters copied by name (remove ranges as [[s, e], …])."""
    spec = AUDIO_OPERATIONS[op["type"]]
    if tool != AUDIO_TOOL:
        raise CompileError(f"{op['type']} cannot be executed by {tool}; the only tool of this operation is {AUDIO_TOOL}")
    args: Dict[str, Any] = {"operation": spec["type"]}
    if op["type"] == "audio.concat":
        args["inputs"] = list(current if isinstance(current, list) else [current])
    else:
        args["input"] = current
    for k in spec["params"]:
        if op.get(k) is not None:
            args[k] = op[k]
    args["output"] = out_id
    return args


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
    name (the tool id prefix) to its version; it defaults to the IR's recorded source.tool_versions."""
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

    concat = next((op for op in d["video"]["operations"] if op["type"] == "video.concat"), None)
    audio_concat = next((op for op in d["audio"]["operations"] if op["type"] == "audio.concat"), None)
    state: Dict[str, Dict[str, Any]] = {}   # subject → {"current", "gen", "stem", "fp"}

    def edits(subject: str) -> None:
        """The single-source editing operations on a subject, chained in IR order (the IR is already in the fixed order)."""
        st = state[subject]
        for op in d["video"]["operations"]:
            if op["asset"] != subject or op["type"] not in EDIT_ORDER[1:]:
                continue
            name = op["type"].split(".", 1)[1]
            st["gen"] += 1
            out_id = f"{subject}_{name}"
            paths[out_id] = str(job / "ops" / f"{st['stem']}_{st['gen']:02d}_{name}" / f"{st['stem']}_{name}.mp4")
            skill = OPERATIONS[op["type"]]["skill"]
            tool = tool_for(skill, subject)
            refs: Dict[str, str] = {}
            inputs = [st["current"]]
            if op["type"] == "video.overlay":
                img_id = f"{subject}_{name}_image"
                paths[img_id] = op["image"]
                keys[img_id] = ""
                refs["image"] = img_id
                inputs.append(img_id)
            args = lower_video_edit(tool, op, st["current"], out_id, refs)
            add(tool, args, inputs, [out_id], list(op.get("decision_ids") or []), st["fp"], skill=skill)
            st["current"] = out_id

    def loudness(subject: str) -> None:
        st = state[subject]
        for op in d["audio"]["operations"]:
            if op["asset"] != subject or op["type"] != "audio.loudness":
                continue
            st["gen"] += 1
            out_id = f"{subject}_loudnorm"
            skill = "audio_normalize" if "input" in op else "loudness_normalization"
            tool = tool_for(skill, subject)
            paths[out_id] = str(job / "ops" / f"{st['stem']}_{st['gen']:02d}_loudness" / f"{st['stem']}_loudnorm.{'wav' if tool == AUDIO_TOOL else 'mp4'}")
            args = lower_audio_loudness(tool, op, st["current"], out_id)
            add(tool, args, [st["current"]], [out_id], list(op.get("decision_ids") or []), st["fp"], skill=skill)
            st["current"] = out_id

    def audio_edits(subject: str) -> None:
        """Audio production operations on a subject (gain → channels → fades), chained in IR order; audio-production outputs are WAV."""
        st = state[subject]
        for op in d["audio"]["operations"]:
            if op["asset"] != subject or op["type"] not in AUDIO_ORDER or op["type"] == "audio.loudness":
                continue
            name = op["type"].split(".", 1)[1]
            st["gen"] += 1
            out_id = f"{subject}_{name}"
            paths[out_id] = str(job / "ops" / f"{st['stem']}_{st['gen']:02d}_{name}" / f"{st['stem']}_{name}.wav")
            skill = AUDIO_OPERATIONS[op["type"]]["skill"]
            tool = tool_for(skill, subject)
            add(tool, lower_audio_op(tool, op, st["current"], out_id), [st["current"]], [out_id], list(op.get("decision_ids") or []), st["fp"], skill=skill)
            st["current"] = out_id

    def delivery(subject: str) -> None:
        st = state[subject]
        for t in d["delivery"]["targets"]:
            art_id = f"{subject}_delivery_{t['id']}"
            if t.get("preset"):
                ext = {"prores": "mov", "gif": "gif"}.get(t["preset"], "mp4")
                paths[art_id] = str(job / "artifacts" / f"{st['stem']}_{t['id']}.{ext}")
                args = {"input": st["current"], "preset": t["preset"], "output": art_id}
                add(tool_for("delivery_export", t["id"]), args, [st["current"]], [art_id], list(t.get("decision_ids") or []), st["fp"], skill="delivery_export")
                add(tool_for("delivery_check", t["id"]), {"input": art_id, "platform": t.get("platform", "custom")}, [art_id], [], list(t.get("decision_ids") or []), st["fp"], kind="qa", skill="delivery_check")
            elif st["current"] != subject:
                # generic profile: the last processed intermediate is the deliverable (no re-encode)
                paths[art_id] = paths[st["current"]]

    for idx, (asset_id, asset) in enumerate(d["assets"].items(), start=1):
        paths[asset_id] = asset["path"]
        keys[asset_id] = ""
        fp = source_fingerprint(asset)
        stem = f"{idx:02d}_{Path(asset['path']).stem}"  # index keeps two inputs with the same file name apart
        state[asset_id] = {"current": asset_id, "gen": 0, "stem": stem, "fp": fp}
        st = state[asset_id]
        for op in d["video"]["operations"]:
            if op["asset"] != asset_id or op["type"] != "video.trim":
                continue
            st["gen"] += 1
            out_id = f"{asset_id}_trim"
            paths[out_id] = str(job / "ops" / f"{stem}_{st['gen']:02d}_trim" / f"{stem}_trim.mp4")
            tool = tool_for("silence_cleanup", asset_id)
            args = lower_video_trim(tool, op, st["current"], out_id)
            add(tool, args, [st["current"]], [out_id], list(op.get("decision_ids") or []), fp, skill="silence_cleanup")
            st["current"] = out_id
        cut = next((op for op in d["audio"]["operations"] if op["type"] == "audio.cut" and op["asset"] == asset_id), None)
        if cut is not None:   # audio production path (ADR-030): the silence decisions as one audio.cut; audio outputs are WAV
            st["gen"] += 1
            out_id = f"{asset_id}_cut"
            paths[out_id] = str(job / "ops" / f"{stem}_{st['gen']:02d}_cut" / f"{stem}_cut.wav")
            tool = tool_for("audio_cut", asset_id)
            add(tool, lower_audio_op(tool, cut, st["current"], out_id), [st["current"]], [out_id], list(cut.get("decision_ids") or []), fp, skill="audio_cut")
            st["current"] = out_id
        on_audio = cut is not None or any(op["asset"] == asset_id and (op["type"] != "audio.loudness" or "input" in op) for op in d["audio"]["operations"]) \
            or asset_id in (audio_concat.get("inputs") or [] if audio_concat else [])
        if on_audio:
            if audio_concat is None:
                audio_edits(asset_id)
                loudness(asset_id)
                delivery(asset_id)
        elif concat is None:
            edits(asset_id)
            loudness(asset_id)
            delivery(asset_id)
    if audio_concat is not None:
        subject = audio_concat.get("output") or PROGRAMME_AUDIO
        inputs = [state[a]["current"] for a in audio_concat["inputs"]]
        fp = "concat:" + stable_hash([source_fingerprint(d["assets"][a]) for a in audio_concat["inputs"]])[:16]
        state[subject] = {"current": subject, "gen": 1, "stem": subject, "fp": fp}
        paths[subject] = str(job / "ops" / f"{subject}_01_concat" / f"{subject}.wav")
        tool = tool_for("audio_concat", subject)
        add(tool, lower_audio_op(tool, audio_concat, inputs, subject), inputs, [subject], list(audio_concat.get("decision_ids") or []), fp, skill="audio_concat")
        audio_edits(subject)
        loudness(subject)
        delivery(subject)
    if concat is not None:
        # the multi-source timeline: every (trimmed) input in the decided order → one programme; the chain continues on it
        subject = concat.get("output") or PROGRAMME
        inputs = [state[a]["current"] for a in concat["inputs"]]
        fp = "concat:" + stable_hash([source_fingerprint(d["assets"][a]) for a in concat["inputs"]])[:16]
        state[subject] = {"current": subject, "gen": 1, "stem": subject, "fp": fp}
        paths[subject] = str(job / "ops" / f"{subject}_01_concat" / f"{subject}.mp4")
        tool = tool_for("video_concat", subject)
        args = lower_video_edit(tool, concat, inputs, subject)
        add(tool, args, inputs, [subject], list(concat.get("decision_ids") or []), fp, skill="video_concat")
        edits(subject)
        loudness(subject)
        delivery(subject)
    return ops, paths
