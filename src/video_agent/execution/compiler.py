"""Compiler: Project IR → ordered Operations with typed adapter args. Fixed order per subject:
trim → concat → edits → colour → graphics → captions (sidecar, burn-in or soft track) → a joined still (ADR-070) → loudness → export → check → timeline → thumbnail → QC gate.
Paths for intermediates are decided here; no engine flags appear here.

Idempotency keys are chained: key(op) = H(source fingerprint, tool, args, tool version, keys of the ops that
produced its inputs). Changing the trim therefore changes the loudness key too, so a resumed job can never reuse
an output whose upstream changed. Operation ids are deterministic (H(tool, args, inputs)) so provenance and
job records can be matched across compiles."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..agent.audio import AUDIO_ORDER, OPERATIONS as AUDIO_OPERATIONS, PROGRAMME_AUDIO, PROGRAMME_OPS as AUDIO_PROGRAMME_OPS, TOOL as AUDIO_TOOL
from ..agent.editing import IMAGE_INSERT_OP, IMAGE_INSERT_SKILL, IMAGE_JOIN_SKILL, OPERATIONS, PROGRAMME, SINGLE_SOURCE_ORDER, TRIM_SKILL, delivery_subjects
from ..agent.finishing import COLOR_OPERATIONS, COLOR_ORDER, COLOR_TOOL, GRAPHICS_SKILL, GRAPHICS_TOOL, THUMBNAIL_FRAME_SKILL, THUMBNAIL_FRAME_TOOL, THUMBNAIL_RENDER_SKILL, THUMBNAIL_RENDER_TOOL
from ..agent.qc import PACKAGE_KIND, PACKAGE_TYPES, QC_SKILL, QC_TOOL, package_rules, rules_for_subject, sidecar_rules
from ..agent.subtitles import BURN_SKILL, GENERATE_SKILL, GENERATE_TOOL, MUX_SKILL, RENDER_TOOL
from ..agent.takes import AUDIO_MUX_SKILL
from ..agent.timeline import TIMELINE_EXTENSIONS, TIMELINE_SKILL, TIMELINE_SUBJECT
from ..agent.nle import MODES as NLE_MODES
from ..agent.image_prepare import prepare_chain, prepared_id
from ..agent.voice import OPERATIONS as VOICE_OPERATIONS, PER_INPUT as VOICE_PER_INPUT
from ..tools.voice_enhance.adapter import stem_name as voice_stem_name
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


def image_fingerprint(path: str) -> str:
    """ADR-077: an overlay image's identity for the keys of its preparation: the path with its size and mtime (a replaced file changes it)."""
    try:
        s = os.stat(path)
        return f"image:{path}:{s.st_size}:{int(s.st_mtime)}"
    except OSError:
        return "image:" + path


class CompileError(ValueError):
    pass


PROJECT_IR = "project_ir"   # ADR-086: the artifact id of the project IR snapshot nle-bridge reads (<job>/nle/project.json, written by the service)


def lower_video_trim(tool: str, op: Dict[str, Any], current: str, out_id: str) -> Dict[str, Any]:
    """IR `video.trim` (keep ranges on one asset) → the typed arguments of the tool the plan selected. The meaning is the
    same for every tool: keep exactly these ranges, in this order, everything else is removed; `accurate` is the plan's
    precision requirement. The reference cut tool takes them as `segments` (+ `accurate`); video-editing/cut (ADR-028) as
    the contract's CUT parameters `keep` [{start, end}] and `precision` frame | keyframe. The tool itself comes from the
    plan (registry-selected); the compiler never chooses an engine and never falls back to another one."""
    keep = [[round(float(s), 3), round(float(e), 3)] for s, e in op["keep"]]
    if tool == "video-editing/trim":   # ADR-062: the contract's TRIM takes one range as start / end
        if len(keep) != 1:
            raise CompileError(f"video-editing/trim keeps exactly one range; this video.trim keeps {len(keep)}")
        return {"input": current, "start": keep[0][0], "end": keep[0][1], "precision": "frame" if op.get("accurate") else "keyframe", "output": out_id}
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
    if "tool" in spec and tool != spec["tool"]:
        raise CompileError(f"{op['type']} cannot be executed by {tool}; the only tool of this operation is {spec['tool']}")
    args: Dict[str, Any] = {}
    if op["type"] in ("video.concat", "video.switch", "video.grid"):
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


def lower_image_prepare(tool: str, tool_name: str, params: Dict[str, Any], image_id: str, out_id: str) -> Dict[str, Any]:
    """ADR-077: one preparation of an overlay image → imagemagick-skill/<tool>'s typed arguments (the IR's values by the contract's names;
    references are artifact ids). The plan names the tool; the compiler only checks it is the one for this preparation."""
    if tool.split("/", 1)[-1] != tool_name:
        raise CompileError(f"the overlay image's {tool_name} cannot be executed by {tool}")
    return {"input": image_id, **{k: v for k, v in params.items() if v is not None}, "output": out_id}


def lower_image_insert(tool: str, join_tool: str, op: Dict[str, Any], image_id: str, still_id: str, current: str, out_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ADR-070: IR `video.image_insert` → (video-editing/image_insert args, video-editing/concat args). The still becomes a clip at the
    operation's duration / width / height / fps (the contract's IMAGE_INSERT names); the join takes the clip and the picture in the
    decided order at the same frame and rate, so the picture is not rescaled. Nothing else is copied."""
    spec = OPERATIONS[IMAGE_INSERT_OP]
    if tool != spec["tool"]:
        raise CompileError(f"{IMAGE_INSERT_OP} cannot be executed by {tool}; the only tool of this operation is {spec['tool']}")
    if join_tool != OPERATIONS["video.concat"]["tool"]:
        raise CompileError(f"the join of a {IMAGE_INSERT_OP} cannot be executed by {join_tool}; it is {OPERATIONS['video.concat']['tool']}")
    if op.get("position") not in ("before", "after"):
        raise CompileError(f"{IMAGE_INSERT_OP}: position {op.get('position')!r} is not before / after")
    frame = {k: op[k] for k in ("width", "height", "fps") if op.get(k) is not None}
    still = {"input": image_id, "duration": float(op["duration"]), **frame, "output": still_id}
    join = {"inputs": [still_id, current] if op["position"] == "before" else [current, still_id], **frame, "output": out_id}
    return still, join


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
    """IR audio operation (audio.cut / filler_remove / concat / mix / noise_reduction / gain / mono / stereo / downmix / dynamics / fade_in / fade_out)
    → audio-production/run arguments: the contract's operation type plus the allowlisted parameters copied by name (remove ranges as
    [[s, e], …]; MIX levels / duck, NOISE_REDUCTION mode / strength_db, the DYNAMICS stage objects, FILLER_REMOVE words / lexicon exactly as
    the IR holds them)."""
    spec = AUDIO_OPERATIONS[op["type"]]
    if tool != AUDIO_TOOL:
        raise CompileError(f"{op['type']} cannot be executed by {tool}; the only tool of this operation is {AUDIO_TOOL}")
    args: Dict[str, Any] = {"operation": spec["type"]}
    if op["type"] in AUDIO_PROGRAMME_OPS:   # concat / mix (ADR-063) take the inputs in order
        args["inputs"] = list(current if isinstance(current, list) else [current])
    else:
        args["input"] = current
    for k in spec["params"]:
        if op.get(k) is not None:
            args[k] = op[k]
    args["output"] = out_id
    return args


def lower_voice_op(tool: str, op: Dict[str, Any], current: Any, out_id: Any) -> Dict[str, Any]:
    """IR voice operation (ADR-073: voice.isolate / voice.dereverb on one input, voice.bleed_reduce on every mixed input at once) → the
    voice-enhance tool's typed arguments: references plus the allowlisted parameters copied by name. The tool is the plan's; each
    operation has exactly one."""
    spec = VOICE_OPERATIONS[op["type"]]
    if tool != spec["tool"]:
        raise CompileError(f"{op['type']} cannot be executed by {tool}; the only tool of this operation is {spec['tool']}")
    args: Dict[str, Any] = {"inputs": list(current), "outputs": list(out_id)} if op["type"] == "voice.bleed_reduce" else {"input": current, "output": out_id}
    for k in spec["params"]:
        if op.get(k) is not None:
            args[k] = op[k]
    return args


def lower_color_op(tool: str, op: Dict[str, Any], current: str, out_id: str, lut_id: Optional[str] = None) -> Dict[str, Any]:
    """IR colour operation → color-grading/run arguments: the contract's operation type plus the allowlisted parameters by name;
    a LUT travels as an artifact id the adapter resolves through the paths map (ADR-031)."""
    spec = COLOR_OPERATIONS[op["type"]]
    if tool != COLOR_TOOL:
        raise CompileError(f"{op['type']} cannot be executed by {tool}; the only tool of this operation is {COLOR_TOOL}")
    args: Dict[str, Any] = {"operation": spec["type"], "input": current, "output": out_id, "format": "mp4"}
    for k in spec["params"]:
        if op.get(k) is not None:
            args[k] = op[k]
    if lut_id:
        args["lut"] = lut_id
    return args


def lower_graphics_render(tool: str, op: Dict[str, Any], current: str, out_id: str, image_id: Optional[str] = None, video_id: Optional[str] = None) -> Dict[str, Any]:
    """IR graphics.render → motion-graphics/run arguments: the element list as planned (image / video overlay paths become artifact ids)."""
    if tool != GRAPHICS_TOOL:
        raise CompileError(f"graphics.render cannot be executed by {tool}; the only tool of this operation is {GRAPHICS_TOOL}")
    elements = []
    for e in op.get("elements") or []:
        el = {"id": e["id"], "type": e["type"], "start": float(e["start"]), "end": float(e["end"]), "parameters": {k: v for k, v in (e.get("parameters") or {}).items() if k not in ("image", "video")}}
        if e["type"] == "image_overlay" and image_id:
            el["parameters"]["image"] = image_id
        if e["type"] == "video_overlay" and video_id:
            el["parameters"]["video"] = video_id
        if e.get("animation"):
            el["animation"] = {"kind": e["animation"]["kind"], "parameters": dict(e["animation"]["parameters"])}
        elements.append(el)
    return {"input": current, "output": out_id, "elements": elements}


def lower_thumbnail(tool: str, op: Dict[str, Any], current: str, out_id: str) -> Dict[str, Any]:
    """IR graphics.thumbnail → thumbnail/extract_frame (a plain frame) or thumbnail/render (frame + caption) arguments."""
    want = THUMBNAIL_RENDER_TOOL if op.get("text") is not None else THUMBNAIL_FRAME_TOOL
    if tool != want:
        raise CompileError(f"graphics.thumbnail cannot be executed by {tool}; this operation needs {want}")
    args: Dict[str, Any] = {"input": current, "timestamp": float(op["timestamp"]), "format": op["format"], "output": out_id}
    if want == THUMBNAIL_RENDER_TOOL:
        for k in ("width", "height", "text", "font_id", "font_size", "color", "position"):
            if op.get(k) is not None:
                args[k] = op[k]
    return args


def lower_captions(tool: str, op: Dict[str, Any], current: Optional[str], out_id: str, sidecar_id: Optional[str] = None) -> Dict[str, Any]:
    """IR captions.generate / captions.burn / captions.mux → subtitle/generate / subtitle/render arguments (the same document; the
    burn-in and the mux (ADR-063, render `mode: "mux"`) also name the video input and the sidecar it was generated from)."""
    if op["type"] == "captions.generate":
        if tool != GENERATE_TOOL:
            raise CompileError(f"captions.generate cannot be executed by {tool}; the only tool of this operation is {GENERATE_TOOL}")
        args: Dict[str, Any] = {"operation": "generate", "format": op["format"], "document_id": f"{op['asset']}_captions", "language": op["language"], "cues": list(op["cues"]), "output": out_id}
        if op.get("constraints"):
            args["constraints"] = dict(op["constraints"])
        if op.get("temporal_scope"):
            args["video_duration"] = float(op["temporal_scope"]["end"])
        return args
    if tool != RENDER_TOOL:
        raise CompileError(f"{op['type']} cannot be executed by {tool}; the only tool of this operation is {RENDER_TOOL}")
    args = {"operation": "render", "input": current, "sidecar": sidecar_id, "format": "srt", "document_id": f"{op['asset']}_captions", "language": op["_language"], "cues": list(op["_cues"]), "output": out_id}
    if op["type"] == "captions.mux":   # ADR-063: subtitle-skill render's own `mode` parameter (a soft track; nothing re-encoded)
        args["mode"] = "mux"
    return args


def lower_qc_check(tool: str, kind: str, input_id: str, rules: Dict[str, Any], subtitle_id: Optional[str] = None, reference_id: Optional[str] = None) -> Dict[str, Any]:
    if tool != QC_TOOL:
        raise CompileError(f"the QC gate cannot be executed by {tool}; the only tool of this operation is {QC_TOOL}")
    args: Dict[str, Any] = {"input": input_id, "kind": kind, "rules": dict(rules), "cache_policy": "bypass"}
    if subtitle_id:
        args["subtitle"] = subtitle_id
    if reference_id:
        args["reference_video"] = reference_id
    return args


def lower_qc_package(tool: str, members: List[Dict[str, Any]], rules: Dict[str, Any]) -> Dict[str, Any]:
    """ADR-065: the deliverable set → qc/check kind delivery_package; each member names its artifact (the id the report speaks of), its
    contract type and the logical input the adapter resolves to the file."""
    if tool != QC_TOOL:
        raise CompileError(f"the QC gate cannot be executed by {tool}; the only tool of this operation is {QC_TOOL}")
    return {"kind": PACKAGE_KIND, "artifacts": [{"artifact_id": m["artifact_id"], "artifact_type": m["artifact_type"], "input": m["input"]} for m in members],
            "rules": dict(rules), "cache_policy": "bypass"}


def lower_timeline_export(tool: str, op: Dict[str, Any], project_id: str, out_id: str) -> Dict[str, Any]:
    """IR `timeline.export` (ADR-052) → the timeline tool's typed arguments: the clips as records (source asset id + in / out
    seconds) the adapter writes into the tool's project document inside the workspace, that document's path id, and the
    timeline output id (its extension, chosen by the compiler from the format, selects the writer). The tool comes from the
    plan (registry-selected, single candidate); like the other reference-engine-only operations there is no pairing to check."""
    return {"project": project_id, "clips": [{"input": c["asset"], "in": float(c["in"]), "out": float(c["out"])} for c in op["clips"]],
            "output": out_id, "overwrite": True}


def lower_take_crossfade(tool: str, kind: str, rec: Dict[str, Any], out_id: str) -> Dict[str, Any]:
    """ADR-054 take-assembly crossfade records → typed tool arguments: a span is an audio-production CUT of the source (the ranges
    outside the span removed), a join is an audio-production CONCAT (with the crossfade, or 0 between runs), the mux is the
    plan's `audio_mux` tool (registry-selected, single candidate: the picture input, the replacing audio, the output; like the
    timeline export there is no pairing to check here)."""
    if kind == "mux":
        return {"input": rec["video"], "replace": rec["audio"], "output": out_id}
    if tool != AUDIO_TOOL:
        raise CompileError(f"the take-assembly audio {kind} cannot be executed by {tool}; the only tool of this operation is {AUDIO_TOOL}")
    if kind == "span":
        return {"operation": "CUT", "input": rec["asset"], "remove": [[float(x), float(y)] for x, y in rec["remove"]], "output": out_id}
    return {"operation": "CONCAT", "inputs": list(rec["inputs"]), "crossfade": float(rec["crossfade"]), "output": out_id}


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
    switch = next((op for op in d["video"]["operations"] if op["type"] == "video.switch"), None)   # ADR-045: alternative to concat, mutually exclusive
    grid = next((op for op in d["video"]["operations"] if op["type"] == "video.grid"), None)        # ADR-046: alternative to concat/switch, mutually exclusive
    audio_concat = next((op for op in d["audio"]["operations"] if op["type"] in AUDIO_PROGRAMME_OPS), None)   # the audio programme: a concat, or a mix (ADR-063)
    state: Dict[str, Dict[str, Any]] = {}   # subject → {"current", "gen", "stem", "fp"}

    def edits(subject: str) -> None:
        """The single-source editing operations on a subject, chained in IR order (the IR is already in the fixed order)."""
        st = state[subject]
        for op in d["video"]["operations"]:
            if op["asset"] != subject or op["type"] not in SINGLE_SOURCE_ORDER:
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
                for sk, tool_name, tparams in prepare_chain(op.get("prepare") or {}):   # ADR-077: strip → resize → pad, each a new PNG in the job
                    st["gen"] += 1
                    prep_id = prepared_id(subject, tool_name)
                    paths[prep_id] = str(job / "ops" / f"{st['stem']}_{st['gen']:02d}_overlay_{tool_name}" / f"{st['stem']}_overlay_{tool_name}.png")
                    ptool = tool_for(sk, subject)
                    add(ptool, lower_image_prepare(ptool, tool_name, tparams, img_id, prep_id), [img_id], [prep_id], list(op.get("decision_ids") or []),
                        image_fingerprint(op["image"]), skill=sk)
                    img_id = prep_id
                refs["image"] = img_id
                inputs.append(img_id)
            args = lower_video_edit(tool, op, st["current"], out_id, refs)
            add(tool, args, inputs, [out_id], list(op.get("decision_ids") or []), st["fp"], skill=skill)
            st["current"] = out_id

    def inserts(subject: str) -> None:
        """ADR-070: a still joined to the finished picture (after the finishing, before loudness): image_insert, then concat."""
        st = state[subject]
        for op in d["video"]["operations"]:
            if op["asset"] != subject or op["type"] != IMAGE_INSERT_OP:
                continue
            st["gen"] += 1
            still_id, out_id, img_id = f"{subject}_still", f"{subject}_image_insert", f"{subject}_image_insert_image"
            paths[img_id] = op["image"]
            keys[img_id] = ""
            paths[still_id] = str(job / "ops" / f"{st['stem']}_{st['gen']:02d}_still" / f"{st['stem']}_still.mp4")
            st["gen"] += 1
            paths[out_id] = str(job / "ops" / f"{st['stem']}_{st['gen']:02d}_image_insert" / f"{st['stem']}_image_insert.mp4")
            tool, join_tool = tool_for(IMAGE_INSERT_SKILL, subject), tool_for(IMAGE_JOIN_SKILL, subject)
            still, join = lower_image_insert(tool, join_tool, op, img_id, still_id, st["current"], out_id)
            ids = list(op.get("decision_ids") or [])
            add(tool, still, [img_id], [still_id], ids, st["fp"], skill=IMAGE_INSERT_SKILL)
            add(join_tool, join, list(join["inputs"]), [out_id], ids, st["fp"], skill=IMAGE_JOIN_SKILL)
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

    def finishing(subject: str) -> None:
        """Colour → graphics → captions (sidecar, then the burn-in) on the subject's current picture, in IR order (ADR-031)."""
        st = state[subject]
        for op in (d.get("color") or {}).get("operations") or []:
            if op["asset"] != subject or op["type"] not in COLOR_ORDER:
                continue
            name = op["type"].split(".", 1)[1]
            st["gen"] += 1
            out_id = f"{subject}_{name}"
            paths[out_id] = str(job / "ops" / f"{st['stem']}_{st['gen']:02d}_{name}" / f"{st['stem']}_{name}.mp4")
            skill = COLOR_OPERATIONS[op["type"]]["skill"]
            tool = tool_for(skill, subject)
            inputs = [st["current"]]
            lut_id = None
            if op["type"] == "color.lut":
                lut_id = f"{subject}_{name}_lut"
                paths[lut_id] = op["lut"]
                keys[lut_id] = ""
                inputs.append(lut_id)
            add(tool, lower_color_op(tool, op, st["current"], out_id, lut_id), inputs, [out_id], list(op.get("decision_ids") or []), st["fp"], skill=skill)
            st["current"] = out_id
        for op in (d.get("graphics") or {}).get("operations") or []:
            if op["asset"] != subject or op["type"] != "graphics.render":
                continue
            st["gen"] += 1
            out_id = f"{subject}_graphics"
            paths[out_id] = str(job / "ops" / f"{st['stem']}_{st['gen']:02d}_graphics" / f"{st['stem']}_graphics.mp4")
            tool = tool_for(GRAPHICS_SKILL, subject)
            inputs = [st["current"]]
            refs: Dict[str, Optional[str]] = {"image": None, "video": None}
            for ref in refs:   # the composited files (named by the user) are inputs by path, like a LUT
                if op.get(ref):
                    refs[ref] = f"{subject}_graphics_{ref}"
                    paths[refs[ref]] = op[ref]
                    keys[refs[ref]] = ""
                    inputs.append(refs[ref])
            add(tool, lower_graphics_render(tool, op, st["current"], out_id, refs["image"], refs["video"]), inputs, [out_id], list(op.get("decision_ids") or []), st["fp"], skill=GRAPHICS_SKILL)
            st["current"] = out_id
        gen = next((op for op in (d.get("captions") or {}).get("operations") or [] if op["asset"] == subject and op["type"] == "captions.generate"), None)
        if gen is not None:
            sc_id = gen.get("output") or f"{subject}_captions"
            paths[sc_id] = str(job / "artifacts" / f"{st['stem']}_captions.{gen['format']}")
            tool = tool_for(GENERATE_SKILL, subject)
            add(tool, lower_captions(tool, gen, None, sc_id), [], [sc_id], list(gen.get("decision_ids") or []), st["fp"], skill=GENERATE_SKILL)
            burn = next((op for op in (d.get("captions") or {}).get("operations") or [] if op["asset"] == subject and op["type"] == "captions.burn"), None)
            if burn is not None:
                st["gen"] += 1
                out_id = burn.get("output") or f"{subject}_burn"
                paths[out_id] = str(job / "ops" / f"{st['stem']}_{st['gen']:02d}_burn" / f"{st['stem']}_burn.mp4")
                tool = tool_for(BURN_SKILL, subject)
                b = dict(burn, _language=gen["language"], _cues=gen["cues"])
                add(tool, lower_captions(tool, b, st["current"], out_id, sc_id), [st["current"], sc_id], [out_id], list(burn.get("decision_ids") or []), st["fp"], skill=BURN_SKILL)
                st["current"] = out_id
            mux = next((op for op in (d.get("captions") or {}).get("operations") or [] if op["asset"] == subject and op["type"] == "captions.mux"), None)
            if mux is not None:   # ADR-063: the soft track; the container stays mp4 (mov_text, chosen by ffmpeg-skill/caption from the extension)
                st["gen"] += 1
                out_id = mux.get("output") or f"{subject}_subs"
                paths[out_id] = str(job / "ops" / f"{st['stem']}_{st['gen']:02d}_subs" / f"{st['stem']}_subs.mp4")
                tool = tool_for(MUX_SKILL, subject)
                x = dict(mux, _language=gen["language"], _cues=gen["cues"])
                add(tool, lower_captions(tool, x, st["current"], out_id, sc_id), [st["current"], sc_id], [out_id], list(mux.get("decision_ids") or []), st["fp"], skill=MUX_SKILL)
                st["current"] = out_id
        st["picture"] = st["current"]

    def thumbnail(subject: str) -> None:
        st = state[subject]
        for op in (d.get("graphics") or {}).get("operations") or []:
            if op["asset"] != subject or op["type"] != "graphics.thumbnail":
                continue
            out_id = op.get("output") or f"{subject}_thumbnail"
            ext = "png" if op["format"] == "png" else "jpg"
            paths[out_id] = str(job / "artifacts" / f"{st['stem']}_thumbnail.{ext}")
            skill = THUMBNAIL_RENDER_SKILL if op.get("text") is not None else THUMBNAIL_FRAME_SKILL
            tool = tool_for(skill, subject)
            src = st.get("picture") or st["current"]
            add(tool, lower_thumbnail(tool, op, src, out_id), [src], [out_id], list(op.get("decision_ids") or []), st["fp"], skill=skill)

    def timeline(subject: str) -> None:
        """The subject's cut list as an editor timeline (ADR-052): reads the sources only, never an intermediate, so it depends on
        no other operation; the render project it needs is an intermediate of its own under ops/."""
        st = state[subject]
        for op in (d.get("delivery") or {}).get("timelines") or []:
            if op["asset"] != subject or op["type"] != TIMELINE_SUBJECT:
                continue
            out_id = op.get("output") or f"{subject}_timeline"
            paths[out_id] = str(job / "artifacts" / f"{st['stem']}_timeline.{TIMELINE_EXTENSIONS[op['format']]}")
            project_id = f"{out_id}_project"
            paths[project_id] = str(job / "ops" / f"{st['stem']}_timeline" / f"{st['stem']}_timeline_project.json")
            keys[project_id] = ""
            tool = tool_for(TIMELINE_SKILL, subject)
            inputs = list(dict.fromkeys(c["asset"] for c in op["clips"]))
            add(tool, lower_timeline_export(tool, op, project_id, out_id), inputs, [out_id], list(op.get("decision_ids") or []), st["fp"], skill=TIMELINE_SKILL)

    def nle(subject: str) -> None:
        """ADR-086: nle-bridge export / conform of the subject from the project IR this job executes (the service writes it as
        <job>/nle/project.json before running); the files land in one directory under artifacts/, next to the deliverables."""
        st = state[subject]
        for op in (d.get("delivery") or {}).get("nle") or []:
            if op["asset"] != subject:
                continue
            mode = op["type"].split(".", 1)[1]
            skill, want = NLE_MODES[mode]
            tool = tool_for(skill, subject)
            if tool != want:
                raise CompileError(f"{op['type']} cannot be executed by {tool}; its tool is {want}")
            paths[PROJECT_IR] = str(job / "nle" / "project.json")
            keys[PROJECT_IR] = ""
            out_id = op.get("output") or f"{subject}_nle"
            paths[out_id] = str(job / "artifacts" / f"{st['stem']}_nle_{mode}")
            args = {"project": PROJECT_IR, "subject": subject, "formats": list(op["formats"]), "output": out_id, **({"zip": bool(op.get("zip"))} if mode == "conform" else {})}
            add(tool, args, [PROJECT_IR], [out_id], list(op.get("decision_ids") or []), st["fp"] + "|ir:" + stable_hash(d.get("delivery", {}).get("timelines") or [])[:16], skill=skill)

    def qc_gate(subject: str) -> None:
        """One qc/check per delivery artifact of the subject and one for its subtitle sidecar (ADR-032): kind qa, no output, never reused."""
        qc = (d.get("qa") or {}).get("qc") or {}
        if not qc.get("enabled"):
            return
        st = state[subject]
        row = next((r for r in delivery_subjects(d) if r["id"] == subject), None)
        tol = float((d["qa"].get("thresholds") or {}).get("loudness_tolerance_lu", 2.0))
        for t in d["delivery"]["targets"]:
            art_id = f"{subject}_delivery_{t['id']}"
            # a no-preset target is never platform-checked (no delivery_check op), and a processed or genuinely-
            # untouched-but-audio-only subject has no dedicated delivery_export either (compiler.delivery() only
            # aliases or, for audio-only, does nothing) — gate directly against the subject's own current media
            # instead in those cases, same real bytes as the deliverable. A genuinely untouched subject with a
            # video stream does get a real delivery_export (the stream-copy materialization above), so `art_id`
            # is already in `paths` there and this picks it up automatically; `agent/planner.py`'s `qc_steps()`
            # plans the matching step/tool selection for every one of these cases.
            check_input = art_id if art_id in paths else subject
            if paths.get(check_input) is None:
                continue
            spec = rules_for_subject(row, t, d, tol) if row else {"kind": "delivery", "rules": {}}
            tool = tool_for(QC_SKILL, subject)
            ref_id = None
            if spec.get("reference"):   # ADR-058: the user's master, an input by path like a LUT (never produced by the plan)
                ref_id = "qc_reference"
                paths[ref_id] = spec["reference"]
                keys[ref_id] = ""
            add(tool, lower_qc_check(tool, spec["kind"], check_input, spec["rules"], reference_id=ref_id), [check_input] + ([ref_id] if ref_id else []), [],
                list(qc.get("decision_ids") or []), st["fp"], kind="qa", skill=QC_SKILL)
        for sc_id, srow in (qc.get("sidecars") or {}).items():
            if srow.get("subject") != subject or sc_id not in paths:
                continue
            spec = sidecar_rules()
            ref = srow.get("reference") if srow.get("reference") in paths else None
            tool = tool_for(QC_SKILL, subject)
            add(tool, lower_qc_check(tool, spec["kind"], sc_id, spec["rules"], reference_id=ref), [sc_id] + ([ref] if ref else []), [], list(qc.get("decision_ids") or []), st["fp"], kind="qa", skill=QC_SKILL)
        pkg = (qc.get("packages") or {}).get(subject)
        for t in d["delivery"]["targets"] if pkg else []:   # ADR-065: each target's deliverable set, checked together
            if t["id"] not in pkg.get("targets") or []:
                continue
            art_id = f"{subject}_delivery_{t['id']}"
            check_input = art_id if art_id in paths else subject
            members = [{"artifact_id": art_id, "artifact_type": PACKAGE_TYPES["audio" if row and row.get("audio_only") else "delivery"], "input": check_input}]
            members += [{"artifact_id": pkg[role], "artifact_type": PACKAGE_TYPES[role], "input": pkg[role]} for role in ("captions", "thumbnail") if pkg.get(role) in paths]
            if paths.get(check_input) is None or len(members) < 2:
                continue
            spec = package_rules([dict(m, extension=Path(paths[m["input"]]).suffix) for m in members])
            tool = tool_for(QC_SKILL, subject)
            add(tool, lower_qc_package(tool, members, spec["rules"]), [m["input"] for m in members], [], list(qc.get("decision_ids") or []), st["fp"], kind="qa", skill=QC_SKILL)

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
            elif st["current"] not in d["assets"]:
                # generic profile: the last processed intermediate is the deliverable (no re-encode). Checking
                # "not a raw source asset" rather than "!= subject" matters for a concat/audio_concat programme:
                # its own subject id *is* the id of a real, already-produced (in-workspace) op output from the
                # moment it's created, so `st["current"] != subject` never fires and this alias — and therefore
                # the deliverable's own Artifact registration and QC gate — would never fire either, even though
                # real work already produced it (unlike an untouched single source, whose subject id names the
                # external, unregistrable original asset the whole way through).
                paths[art_id] = paths[st["current"]]
            elif (d["assets"][st["current"]].get("technical") or {}).get("video"):
                # generic profile, genuinely untouched: st["current"] is still a raw source asset, which
                # ArtifactStore.check_path() (ADR-022) refuses to register directly since it lives outside the
                # workspace. Materialize it with a real stream copy (ffmpeg-skill export.py --preset copy) instead
                # of aliasing the external path — same bytes, a real in-workspace file. Requires a video stream
                # (export.py dies without one); `agent/planner.py`'s `delivery_steps()` plans the matching
                # delivery_export step/tool selection for exactly this case, gated on the same condition.
                ext = Path(paths[st["current"]]).suffix.lstrip(".").lower() or "mp4"
                paths[art_id] = str(job / "artifacts" / f"{st['stem']}_{t['id']}.{ext}")
                args = {"input": st["current"], "preset": "copy", "output": art_id}
                add(tool_for("delivery_export", t["id"]), args, [st["current"]], [art_id], list(t.get("decision_ids") or []), st["fp"], skill="delivery_export")

    bleed = next((op for op in d["audio"]["operations"] if op["type"] == "voice.bleed_reduce"), None)
    if bleed is not None:
        # ADR-073: one voice-enhance call for every mixed input as recorded (a mixed input carries no cut); each cleaned track is written as
        # <input name>_a0.wav into one directory of the job, which is where the Skill puts it
        ins, outs = list(bleed["inputs"]), list(bleed["outputs"])
        out_dir = job / "ops" / "00_bleed_reduce"
        for a, o in zip(ins, outs):
            paths[a] = d["assets"][a]["path"]
            keys.setdefault(a, "")
            paths[o] = str(out_dir / voice_stem_name(d["assets"][a]["path"]))
        tool = tool_for(VOICE_OPERATIONS["voice.bleed_reduce"]["skill"], PROGRAMME_AUDIO)
        add(tool, lower_voice_op(tool, bleed, ins, outs), ins, outs, list(bleed.get("decision_ids") or []),
            "bleed:" + stable_hash([source_fingerprint(d["assets"][a]) for a in ins])[:16], skill=VOICE_OPERATIONS["voice.bleed_reduce"]["skill"])
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
            trim_skill = TRIM_SKILL if (TRIM_SKILL, asset_id) in step_tools else "silence_cleanup"   # ADR-062: the plan's step names which skill runs it
            tool = tool_for(trim_skill, asset_id)
            args = lower_video_trim(tool, op, st["current"], out_id)
            add(tool, args, [st["current"]], [out_id], list(op.get("decision_ids") or []), fp, skill=trim_skill)
            st["current"] = out_id
        cut = next((op for op in d["audio"]["operations"] if op["type"] == "audio.cut" and op["asset"] == asset_id), None)
        if cut is not None:   # audio production path (ADR-030): the silence decisions as one audio.cut; audio outputs are WAV
            st["gen"] += 1
            out_id = f"{asset_id}_cut"
            paths[out_id] = str(job / "ops" / f"{stem}_{st['gen']:02d}_cut" / f"{stem}_cut.wav")
            tool = tool_for("audio_cut", asset_id)
            add(tool, lower_audio_op(tool, cut, st["current"], out_id), [st["current"]], [out_id], list(cut.get("decision_ids") or []), fp, skill="audio_cut")
            st["current"] = out_id
        filler = next((op for op in d["audio"]["operations"] if op["type"] == "audio.filler_remove" and op["asset"] == asset_id), None)
        if filler is not None:   # ADR-072: the filler words at their timings on the cut audio (the planner carried them through the cut)
            st["gen"] += 1
            out_id = f"{asset_id}_filler_remove"
            paths[out_id] = str(job / "ops" / f"{stem}_{st['gen']:02d}_filler_remove" / f"{stem}_filler_remove.wav")
            tool = tool_for("audio_filler_remove", asset_id)
            add(tool, lower_audio_op(tool, filler, st["current"], out_id), [st["current"]], [out_id], list(filler.get("decision_ids") or []), fp, skill="audio_filler_remove")
            st["current"] = out_id
        if bleed is not None and asset_id in bleed["inputs"]:
            st["current"] = bleed["outputs"][bleed["inputs"].index(asset_id)]
        for vt in VOICE_PER_INPUT:   # ADR-073: dereverberation, then isolation, after the cut / filler removal (float / 48 kHz mono WAV)
            vop = next((op for op in d["audio"]["operations"] if op["type"] == vt and op["asset"] == asset_id), None)
            if vop is None:
                continue
            name, skill = vt.split(".", 1)[1], VOICE_OPERATIONS[vt]["skill"]
            st["gen"] += 1
            out_id = f"{asset_id}_{name}"
            paths[out_id] = str(job / "ops" / f"{stem}_{st['gen']:02d}_{name}" / f"{stem}_{name}.wav")
            tool = tool_for(skill, asset_id)
            add(tool, lower_voice_op(tool, vop, st["current"], out_id), [st["current"]], [out_id], list(vop.get("decision_ids") or []), fp, skill=skill)
            st["current"] = out_id
        on_audio = cut is not None or filler is not None or any(op["asset"] == asset_id and (op["type"] != "audio.loudness" or "input" in op) for op in d["audio"]["operations"]) \
            or asset_id in (audio_concat.get("inputs") or [] if audio_concat else [])
        if on_audio:
            if audio_concat is None:
                audio_edits(asset_id)
                loudness(asset_id)
                delivery(asset_id)
        elif concat is None and switch is None and grid is None:
            edits(asset_id)
            finishing(asset_id)
            inserts(asset_id)
            loudness(asset_id)
            delivery(asset_id)
            timeline(asset_id)
            nle(asset_id)
            thumbnail(asset_id)
            qc_gate(asset_id)
    if audio_concat is not None:
        subject = audio_concat.get("output") or PROGRAMME_AUDIO
        inputs = [state[a]["current"] for a in audio_concat["inputs"]]
        fp = "concat:" + stable_hash([source_fingerprint(d["assets"][a]) for a in audio_concat["inputs"]])[:16]
        state[subject] = {"current": subject, "gen": 1, "stem": subject, "fp": fp}
        name = audio_concat["type"].split(".", 1)[1]
        paths[subject] = str(job / "ops" / f"{subject}_01_{name}" / f"{subject}.wav")
        skill = AUDIO_OPERATIONS[audio_concat["type"]]["skill"]
        tool = tool_for(skill, subject)
        add(tool, lower_audio_op(tool, audio_concat, inputs, subject), inputs, [subject], list(audio_concat.get("decision_ids") or []), fp, skill=skill)
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
        xf = (d["audio"].get("take_crossfade") or None)
        if xf is not None:
            # ADR-054: the programme audio with crossfaded joins, built beside the picture and muxed under it; the chain continues on the mux
            xids = list(xf.get("decision_ids") or [])
            for sp in xf["spans"]:
                if sp["output"] == sp["asset"]:
                    continue
                a_st = state[sp["asset"]]
                paths[sp["output"]] = str(job / "ops" / f"{a_st['stem']}_xfade_audio" / f"{a_st['stem']}_xfade_audio.wav")
                tool = tool_for("audio_cut", sp["asset"])
                add(tool, lower_take_crossfade(tool, "span", sp, sp["output"]), [sp["asset"]], [sp["output"]], xids, a_st["fp"], skill="audio_cut")
            for c in xf["concats"]:
                paths[c["output"]] = str(job / "ops" / f"{subject}_xfade" / f"{c['output']}.wav")
                tool = tool_for("audio_concat", c["output"])
                add(tool, lower_take_crossfade(tool, "concat", c, c["output"]), list(c["inputs"]), [c["output"]], xids, fp, skill="audio_concat")
            mux = dict(xf["mux"], video=state[subject]["current"])
            state[subject]["gen"] += 1
            paths[mux["output"]] = str(job / "ops" / f"{subject}_{state[subject]['gen']:02d}_xfade" / f"{subject}_xfade.mp4")
            tool = tool_for(AUDIO_MUX_SKILL, subject)
            add(tool, lower_take_crossfade(tool, "mux", mux, mux["output"]), [mux["video"], mux["audio"]], [mux["output"]], xids, fp, skill=AUDIO_MUX_SKILL)
            state[subject]["current"] = mux["output"]
        edits(subject)
        finishing(subject)
        inserts(subject)
        loudness(subject)
        delivery(subject)
        timeline(subject)
        nle(subject)
        thumbnail(subject)
        qc_gate(subject)
    if switch is not None:
        # ADR-045: multicam aligns every (trimmed) input by audio then cuts between them per the decided switch list → one programme
        subject = switch.get("output") or PROGRAMME
        inputs = [state[a]["current"] for a in switch["inputs"]]
        fp = "switch:" + stable_hash([source_fingerprint(d["assets"][a]) for a in switch["inputs"]])[:16]
        state[subject] = {"current": subject, "gen": 1, "stem": subject, "fp": fp}
        paths[subject] = str(job / "ops" / f"{subject}_01_switch" / f"{subject}.mp4")
        tool = tool_for("camera_switch", subject)
        args = lower_video_edit(tool, switch, inputs, subject)
        add(tool, args, inputs, [subject], list(switch.get("decision_ids") or []), fp, skill="camera_switch")
        edits(subject)
        finishing(subject)
        inserts(subject)
        loudness(subject)
        delivery(subject)
        thumbnail(subject)
        qc_gate(subject)
    if grid is not None:
        # ADR-046: grid.py composites every (trimmed) input side by side into one COLSxROWS programme
        subject = grid.get("output") or PROGRAMME
        inputs = [state[a]["current"] for a in grid["inputs"]]
        fp = "grid:" + stable_hash([source_fingerprint(d["assets"][a]) for a in grid["inputs"]])[:16]
        state[subject] = {"current": subject, "gen": 1, "stem": subject, "fp": fp}
        paths[subject] = str(job / "ops" / f"{subject}_01_grid" / f"{subject}.mp4")
        tool = tool_for("video_grid", subject)
        args = lower_video_edit(tool, grid, inputs, subject)
        add(tool, args, inputs, [subject], list(grid.get("decision_ids") or []), fp, skill="video_grid")
        edits(subject)
        finishing(subject)
        inserts(subject)
        loudness(subject)
        delivery(subject)
        thumbnail(subject)
        qc_gate(subject)
    return ops, paths
