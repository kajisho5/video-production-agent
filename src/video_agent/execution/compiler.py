"""Compiler: Project IR → ordered Operations with typed adapter args. Fixed order per subject:
trim → concat → edits → colour → graphics → captions (sidecar, burn-in) → loudness → export → check → thumbnail → QC gate.
Paths for intermediates are decided here; no engine flags appear here.

Idempotency keys are chained: key(op) = H(source fingerprint, tool, args, tool version, keys of the ops that
produced its inputs). Changing the trim therefore changes the loudness key too, so a resumed job can never reuse
an output whose upstream changed. Operation ids are deterministic (H(tool, args, inputs)) so provenance and
job records can be matched across compiles."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..agent.audio import AUDIO_ORDER, OPERATIONS as AUDIO_OPERATIONS, PROGRAMME_AUDIO, TOOL as AUDIO_TOOL
from ..agent.editing import EDIT_ORDER, OPERATIONS, PROGRAMME, delivery_subjects
from ..agent.finishing import COLOR_OPERATIONS, COLOR_ORDER, COLOR_TOOL, GRAPHICS_SKILL, GRAPHICS_TOOL, THUMBNAIL_FRAME_SKILL, THUMBNAIL_FRAME_TOOL, THUMBNAIL_RENDER_SKILL, THUMBNAIL_RENDER_TOOL
from ..agent.qc import QC_SKILL, QC_TOOL, rules_for_subject, sidecar_rules
from ..agent.subtitles import BURN_SKILL, GENERATE_SKILL, GENERATE_TOOL, RENDER_TOOL
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


def lower_graphics_render(tool: str, op: Dict[str, Any], current: str, out_id: str, image_id: Optional[str] = None) -> Dict[str, Any]:
    """IR graphics.render → motion-graphics/run arguments: the element list as planned (image paths become artifact ids)."""
    if tool != GRAPHICS_TOOL:
        raise CompileError(f"graphics.render cannot be executed by {tool}; the only tool of this operation is {GRAPHICS_TOOL}")
    elements = []
    for e in op.get("elements") or []:
        el = {"id": e["id"], "type": e["type"], "start": float(e["start"]), "end": float(e["end"]), "parameters": {k: v for k, v in (e.get("parameters") or {}).items() if k != "image"}}
        if e["type"] == "image_overlay" and image_id:
            el["parameters"]["image"] = image_id
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
    """IR captions.generate / captions.burn → subtitle/generate / subtitle/render arguments (the same document; the burn-in also
    names the video input and the sidecar it was generated from)."""
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
        raise CompileError(f"captions.burn cannot be executed by {tool}; the only tool of this operation is {RENDER_TOOL}")
    return {"operation": "render", "input": current, "sidecar": sidecar_id, "format": "srt", "document_id": f"{op['asset']}_captions", "language": op["_language"], "cues": list(op["_cues"]), "output": out_id}


def lower_qc_check(tool: str, kind: str, input_id: str, rules: Dict[str, Any], subtitle_id: Optional[str] = None, reference_id: Optional[str] = None) -> Dict[str, Any]:
    if tool != QC_TOOL:
        raise CompileError(f"the QC gate cannot be executed by {tool}; the only tool of this operation is {QC_TOOL}")
    args: Dict[str, Any] = {"input": input_id, "kind": kind, "rules": dict(rules), "cache_policy": "bypass"}
    if subtitle_id:
        args["subtitle"] = subtitle_id
    if reference_id:
        args["reference_video"] = reference_id
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
            image_id = None
            if op.get("image"):
                image_id = f"{subject}_graphics_image"
                paths[image_id] = op["image"]
                keys[image_id] = ""
                inputs.append(image_id)
            add(tool, lower_graphics_render(tool, op, st["current"], out_id, image_id), inputs, [out_id], list(op.get("decision_ids") or []), st["fp"], skill=GRAPHICS_SKILL)
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
            add(tool, lower_qc_check(tool, spec["kind"], check_input, spec["rules"]), [check_input], [], list(qc.get("decision_ids") or []), st["fp"], kind="qa", skill=QC_SKILL)
        for sc_id, srow in (qc.get("sidecars") or {}).items():
            if srow.get("subject") != subject or sc_id not in paths:
                continue
            spec = sidecar_rules()
            ref = srow.get("reference") if srow.get("reference") in paths else None
            tool = tool_for(QC_SKILL, subject)
            add(tool, lower_qc_check(tool, spec["kind"], sc_id, spec["rules"], reference_id=ref), [sc_id] + ([ref] if ref else []), [], list(qc.get("decision_ids") or []), st["fp"], kind="qa", skill=QC_SKILL)

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
            finishing(asset_id)
            loudness(asset_id)
            delivery(asset_id)
            thumbnail(asset_id)
            qc_gate(asset_id)
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
        finishing(subject)
        loudness(subject)
        delivery(subject)
        thumbnail(subject)
        qc_gate(subject)
    return ops, paths
