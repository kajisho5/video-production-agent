"""Deterministic production planner: Decisions (with their event / observation evidence) → ProductionPlan (what to
produce, in which order, on which inputs, with which registry-selected skill / tool) + IR sections (the execution
contract the compiler lowers). The planner never emits tool arguments, never executes, never takes tool ids from anyone
but the registry map it is given (ADR-021)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..media.analyzer import AnalysisResult
from ..models import Decision, Inference, now_iso
from .audio import AUDIO_ORDER, OPERATIONS as AUDIO_OPERATIONS, PROGRAMME_AUDIO, concat_segments as audio_concat_segments, cut_ranges, has_video, ir_audio_operation, is_audio_capable, kept_after_cut
from .editing import EDIT_ORDER, OPERATIONS, PROGRAMME, concat_segments, ir_operation
from .finishing import (COLOR_OPERATIONS, COLOR_ORDER, ELEMENT_TYPES, GRAPHICS_SKILL, THUMBNAIL_FRAME_SKILL, THUMBNAIL_RENDER_SKILL, ir_color_operation, ir_graphics_render,
                        ir_thumbnail, picture_size)
from .production_plan import PLANNER_ID, ProductionPlan, ProductionStep
from .qc import QC_SKILL, rules_for_subject, sidecar_rules
from .subtitles import BURN_SKILL, GENERATE_SKILL, cues_from_segments, ir_captions_burn, ir_captions_generate, kept_ranges_of


def build_plan(decisions: List[Decision], analysis: AnalysisResult, tools: Dict[str, str], version: int = 1, frame_accurate: bool = False,
               project_id: str = "", constraints: Optional[List[Dict[str, Any]]] = None, objective: str = "", inferences: Optional[List[Inference]] = None,
               audio_production: bool = False, qc_tolerance_lu: float = 2.0) -> Dict[str, Any]:
    """Decisions (+ the events / observations they rest on) → ProductionPlan + IR sections.
    `tools` is the skill → tool map produced by SkillRegistry.resolve_tools for this environment; it is the only
    source of tool ids here (the planner has no default engine). A step whose skill is absent from the map is emitted
    with tool=None and the plan is marked BLOCKED so the validator / decide() refuse it explicitly.
    Returns {"plan": ProductionPlan.to_dict(), "steps", "summary", "video_ops", "audio_ops", "delivery", "captions_ops", "graphics_ops", "color_ops", "qc"}
    (ADR-031 / ADR-032: the finishing sections and the QC gate)."""
    if tools is None:
        raise TypeError("build_plan needs the skill → tool map resolved by SkillRegistry (tools=None is not allowed)")
    no_tool: List[str] = []
    inf_map = {i.id: i for i in (inferences or [])}
    dec_map = {d.id: d for d in decisions}

    def tool_for(skill: str):
        t = tools.get(skill)
        if not t and skill not in no_tool:
            no_tool.append(skill)
        return t or None

    def evidence_of(dec_ids: List[str]) -> List[str]:
        """Decision evidence expanded through inferences to the events / observations behind them (deterministic order)."""
        out: List[str] = []
        for did in dec_ids:
            for ev in dec_map[did].evidence:
                out.append(ev)
                if ev in inf_map:
                    out.extend(inf_map[ev].evidence)
        return sorted(set(out))

    steps: List[ProductionStep] = []
    video_ops: List[Dict[str, Any]] = []
    audio_ops: List[Dict[str, Any]] = []
    captions_ops: List[Dict[str, Any]] = []
    graphics_ops: List[Dict[str, Any]] = []
    color_ops: List[Dict[str, Any]] = []
    qc_plan: Dict[str, Any] = {"enabled": False, "decision_ids": [], "subjects": {}, "sidecars": {}}
    delivery: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    summary: List[str] = []
    blocked = [d for d in decisions if d.approval == "BLOCK"]
    order = 0
    durations = {a.id: float(a.technical.get("duration") or 0.0) for a in analysis.assets}
    concat_dec = next((d for d in decisions if d.subject == "video.concat" and d.type == "TRANSFORM" and d.status != "REJECTED"), None)
    # audio production path (ADR-030): explicit `audio.production` puts every asset with audio on it (its audio is the subject, delivered as audio)
    audio_subjects = {a.id for a in analysis.assets if is_audio_capable(a.technical)} if audio_production else set()
    audio_concat_dec = next((d for d in decisions if d.subject == "audio.concat" and d.type == "TRANSFORM" and d.status != "REJECTED"), None) if audio_production else None
    # ADR-033: a video container on the audio path is delivered as audio only under its `audio.extract` decision. Every audio operation of
    # that subject cites it (a REJECTED extraction is rejected-cited → BLOCKED; a pending CONFIRM keeps the steps PROPOSED), and once the user
    # rejected it (revise drops the proposal) nothing is planned for the asset on the audio path — the picture is never quietly dropped.
    extract_of: Dict[str, Optional[Decision]] = {a.id: next((d for d in decisions if d.subject == "audio.extract" and d.type == "TRANSFORM" and d.status != "REJECTED"
                                                              and d.params.get("asset_id") == a.id), None)
                                                 for a in analysis.assets if a.id in audio_subjects and has_video(a.technical)}
    extract_refused = {a for a, d in extract_of.items() if d is None}
    if audio_concat_dec is not None and any(a in extract_refused for a in audio_concat_dec.params.get("inputs") or []):
        summary.append("audio concat not planned: the extraction of a video input was rejected")
        audio_concat_dec = None

    def extract_ids(subject: str) -> List[str]:
        """The `audio.extract` decision id(s) an audio operation on `subject` additionally cites (programme: those of its inputs)."""
        if subject == PROGRAMME_AUDIO:
            inputs: List[str] = list(audio_concat_dec.params.get("inputs") or []) if audio_concat_dec is not None else []
            return [x.id for x in (extract_of.get(a) for a in inputs) if x is not None]
        d = extract_of.get(subject)
        return [d.id] if d is not None else []
    current_of: Dict[str, str] = {}      # subject → latest logical output
    last_of: Dict[str, str] = {}
    scope_of: Dict[str, Dict[str, float]] = {}   # subject → temporal scope on the timeline the subject's current output has
    sidecar_of: Dict[str, str] = {}      # subject → logical id of its subtitle sidecar (captions.generate output)
    picture_of: Dict[str, str] = {}      # subject → the logical output the thumbnail is taken from (the finished picture before loudness / export)
    techn = {a.id: a.technical for a in analysis.assets}
    raw_asset_ids = set(techn)   # subject ids that name a raw source asset (as opposed to a produced intermediate/programme)

    def decided(subject: str, subj: str) -> List[Decision]:
        return [x for x in decisions if x.subject == subj and x.type == "TRANSFORM" and x.status != "REJECTED" and x.params.get("asset_id") == subject]

    def finishing_steps(subject: str, sources: List[str]) -> None:
        """Colour → motion graphics → subtitle sidecar (+ burn-in) on the subject's current picture (ADR-031). Each decision becomes one
        step and one IR operation; the sidecar's cues are the transcript segments mapped through this plan's trim / concat / speed."""
        nonlocal order
        for op_type in COLOR_ORDER:
            for d in decided(subject, op_type):
                spec = COLOR_OPERATIONS[op_type]
                params = {k: v for k, v in d.params.items() if k in spec["params"] or k == "lut"}
                out_id = f"{subject}_{op_type.split('.', 1)[1]}"
                scope = dict(scope_of.get(subject) or {"start": 0.0, "end": durations.get(subject, 0.0)})
                color_ops.append(ir_color_operation(op_type, subject, params, [d.id], current_of[subject], out_id, scope=scope))
                order += 1
                st = ProductionStep(id=f"step_{op_type.split('.', 1)[1]}_{subject}", order=order, skill=spec["skill"], tool=tool_for(spec["skill"]), inputs=[current_of[subject]],
                                    params={"asset": subject, **params}, outputs=[out_id], depends_on=[last_of[subject]] if last_of.get(subject) else [], evidence=evidence_of([d.id]),
                                    decision_ids=[d.id], decision_id=d.id, temporal_scope=scope)
                steps.append(st)
                current_of[subject], last_of[subject] = out_id, st.id
                summary.append(d.decision)
        g_decs = [d for typ in ELEMENT_TYPES for d in decided(subject, f"graphics.{typ}")]
        if g_decs:
            scope = dict(scope_of.get(subject) or {"start": 0.0, "end": durations.get(subject, 0.0)})
            elements = []
            for n, d in enumerate(g_decs):
                end = d.params.get("end")
                el: Dict[str, Any] = {"id": f"el{n + 1}_{d.params['type']}", "type": d.params["type"], "start": float(d.params["start"]),
                                      "end": round(float(end), 3) if end is not None else round(scope["end"], 3), "parameters": dict(d.params.get("parameters") or {})}
                if d.params.get("fade") is not None:
                    el["animation"] = {"kind": "fade", "parameters": {"duration": float(d.params["fade"])}}
                elements.append(el)
            out_id = f"{subject}_graphics"
            graphics_ops.append(ir_graphics_render(subject, elements, [d.id for d in g_decs], current_of[subject], out_id, scope=scope))
            order += 1
            st = ProductionStep(id=f"step_graphics_{subject}", order=order, skill=GRAPHICS_SKILL, tool=tool_for(GRAPHICS_SKILL), inputs=[current_of[subject]],
                                params={"asset": subject, "elements": [e["id"] for e in elements]}, outputs=[out_id], depends_on=[last_of[subject]] if last_of.get(subject) else [],
                                evidence=evidence_of([d.id for d in g_decs]), decision_ids=[d.id for d in g_decs], decision_id=g_decs[0].id, temporal_scope=scope)
            steps.append(st)
            current_of[subject], last_of[subject] = out_id, st.id
            summary.append(f"Render {len(elements)} graphics element(s) on {subject}")
        for d in decided(subject, "subtitle.generate"):
            scope = dict(scope_of.get(subject) or {"start": 0.0, "end": durations.get(subject, 0.0)})
            speed = next((float(op["factor"]) for op in video_ops if op.get("asset") == subject and op.get("type") == "video.speed"), 1.0)
            concat = next((op for op in video_ops if op.get("type") == "video.concat" and op.get("output") == subject), None)
            cues: List[Dict[str, Any]] = []
            tmap: Dict[str, Any] = {"speed": speed, "inputs": {}}
            for n, src in enumerate(sources):
                keep = kept_ranges_of(video_ops, src, durations.get(src, 0.0))
                offset = 0.0
                if concat is not None:   # where the input's first kept range lands on the programme timeline (map_point already removed the trimmed material)
                    seg = next((s_ for s_ in concat.get("segments") or [] if s_.get("input") == src), None)
                    offset = round(float(seg["timeline_range"][0]), 3) if seg else 0.0
                segs = [seg_ for o in analysis.observations if o.kind == "transcript" and o.asset_id == src for seg_ in ((o.data or {}).get("segments") or [])]
                cues += cues_from_segments(segs, keep, offset=offset, speed=speed, id_prefix=f"c{n + 1}_" if len(sources) > 1 else "c")
                tmap["inputs"][src] = {"keep": keep, "offset": offset}
            cues.sort(key=lambda c: (c["start"], c["id"]))
            out_id = f"{subject}_captions"
            captions_ops.append(ir_captions_generate(subject, out_id, d.params["format"], d.params["language"], cues, [d.id], list(d.params.get("transcript_ids") or []), tmap,
                                                     constraints=d.params.get("constraints"), scope=scope))
            order += 1
            st = ProductionStep(id=f"step_captions_{subject}", order=order, skill=GENERATE_SKILL, tool=tool_for(GENERATE_SKILL), inputs=[],
                                params={"asset": subject, "format": d.params["format"], "language": d.params["language"], "cues": len(cues)}, outputs=[out_id],
                                depends_on=[last_of[subject]] if last_of.get(subject) else [], evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id, temporal_scope=scope)
            steps.append(st)
            sidecar_of[subject] = out_id
            outputs.append({"role": "CAPTIONS", "logical": out_id, "format": d.params["format"], "expected": {"cues": len(cues), "language": d.params["language"], "source": subject}})
            summary.append(f"Subtitles for {subject}: {len(cues)} cue(s) ({d.params['format']}, {d.params['language']}) on the delivered timeline")
            for b in decided(subject, "subtitle.burn_in"):
                burn_out = f"{subject}_burn"
                captions_ops.append(ir_captions_burn(subject, current_of[subject], out_id, burn_out, [b.id], scope=scope))
                order += 1
                bst = ProductionStep(id=f"step_burn_{subject}", order=order, skill=BURN_SKILL, tool=tool_for(BURN_SKILL), inputs=[current_of[subject], out_id], params={"asset": subject},
                                     outputs=[burn_out], depends_on=[x for x in [last_of.get(subject), st.id] if x], evidence=evidence_of([b.id]),
                                     decision_ids=[b.id], decision_id=b.id, temporal_scope=scope)
                steps.append(bst)
                current_of[subject], last_of[subject] = burn_out, bst.id
                summary.append(b.decision)
        picture_of[subject] = current_of[subject]

    def thumbnail_steps(subject: str, sources: List[str]) -> None:
        nonlocal order
        for d in decided(subject, "thumbnail.render"):
            scope = scope_of.get(subject) or {"start": 0.0, "end": durations.get(subject, 0.0)}
            at = d.params.get("at")
            if at is None:
                at = round(float(scope["end"]) * float(d.params.get("at_ratio") or 0.0), 3)
            size = picture_size(techn.get(sources[0]) or {}, video_ops, subject) if sources else None
            params: Dict[str, Any] = {"timestamp": min(float(at), max(0.0, float(scope["end"]) - 0.001)), "format": d.params["format"]}
            skill = str(d.params.get("skill") or THUMBNAIL_FRAME_SKILL)
            if skill == THUMBNAIL_RENDER_SKILL:
                params.update({"text": d.params["text"], "font_id": d.params.get("font_id", "sans-bold"), "font_size": int(d.params.get("font_size") or 48), "color": d.params.get("color", "#FFFFFF"),
                               "position": d.params.get("position", "bottom"), "width": size[0] if size else None, "height": size[1] if size else None})
            out_id = f"{subject}_thumbnail"
            src = picture_of.get(subject) or current_of[subject]
            graphics_ops.append(ir_thumbnail(subject, params, [d.id], src, out_id))
            order += 1
            dep = next((s_.id for s_ in steps if src in s_.outputs), None)
            st = ProductionStep(id=f"step_thumbnail_{subject}", order=order, skill=skill, tool=tool_for(skill), inputs=[src],
                                params={"asset": subject, "at": params["timestamp"], "format": params["format"], **({"text": params["text"]} if "text" in params else {})}, outputs=[out_id],
                                depends_on=[dep] if dep else [], evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id)
            steps.append(st)
            outputs.append({"role": "THUMBNAIL", "logical": out_id, "format": params["format"], "expected": {"timestamp": params["timestamp"], "source": subject}})
            summary.append(d.decision)

    def qc_steps(subject: str, tolerance_lu: float) -> None:
        """One qc_check step per delivery artifact of the subject (after its check), and one for the subtitle sidecar (ADR-032)."""
        nonlocal order
        for d in [x for x in decisions if x.subject == "qc.check" and x.type == "DELIVER" and x.status != "REJECTED" and x.params.get("asset_id") == subject]:
            qc_plan["enabled"] = True
            qc_plan["decision_ids"].append(d.id)
            preset_targets = set()
            for exp in [s_ for s_ in steps if s_.skill == "delivery_export" and (s_.params or {}).get("target") and s_.outputs and s_.outputs[0].startswith(f"{subject}_delivery_")]:
                target = exp.params["target"]
                preset_targets.add(target)
                art = exp.outputs[0]
                chk = next((s_.id for s_ in steps if s_.skill == "delivery_check" and (s_.params or {}).get("target") == target and art in s_.inputs), exp.id)
                order += 1
                st = ProductionStep(id=f"step_qc_{target}_{subject}", order=order, skill=QC_SKILL, tool=tool_for(QC_SKILL), inputs=[art], params={"asset": subject, "target": target, "kind": "delivery", "artifact": art},
                                    outputs=[], depends_on=[chk], evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id)
                steps.append(st)
                qc_plan["subjects"][subject] = {"kind": "delivery", "targets": sorted(set(list((qc_plan["subjects"].get(subject) or {}).get("targets") or []) + [target]))}
            # no-preset targets: no delivery_export step exists (compiler.delivery() aliases the deliverable
            # to the subject's own current media instead, no re-encode), so gate directly against that real
            # media — same bytes as the deliverable, nothing to wait on but the last real edit (ADR-032's
            # own "each deliverable" promise still holds; only the discovery of *which* op stands in differs)
            for t in delivery:
                target = t["id"]
                if t.get("preset") or target in preset_targets:
                    continue
                art = current_of[subject]
                order += 1
                st = ProductionStep(id=f"step_qc_{target}_{subject}", order=order, skill=QC_SKILL, tool=tool_for(QC_SKILL), inputs=[art], params={"asset": subject, "target": target, "kind": "delivery", "artifact": art},
                                    outputs=[], depends_on=[last_of[subject]] if last_of.get(subject) else [], evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id)
                steps.append(st)
                qc_plan["subjects"][subject] = {"kind": "delivery", "targets": sorted(set(list((qc_plan["subjects"].get(subject) or {}).get("targets") or []) + [target]))}
            if subject in sidecar_of:
                sc = sidecar_of[subject]
                ref = next((s_.outputs[0] for s_ in steps if s_.skill == "delivery_export" and s_.outputs and s_.outputs[0].startswith(f"{subject}_delivery_")), None)
                order += 1
                st = ProductionStep(id=f"step_qc_captions_{subject}", order=order, skill=QC_SKILL, tool=tool_for(QC_SKILL), inputs=[sc] + ([ref] if ref else []),
                                    params={"asset": subject, "target": "captions", "kind": "subtitle", "artifact": sc}, outputs=[],
                                    depends_on=[x for x in [next((s_.id for s_ in steps if sc in s_.outputs), None), next((s_.id for s_ in steps if ref and ref in s_.outputs), None)] if x],
                                    evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id)
                steps.append(st)
                qc_plan["sidecars"][sc] = {"kind": "subtitle", "reference": ref, "subject": subject}
            summary.append(d.decision)

    def edit_steps(subject: str, sources: List[str]) -> None:
        """The single-source editing operations decided for `subject` (video.speed → resize → fit / fill → overlay), chained on its
        current output. Each becomes one plan step and one IR operation with allowlisted parameters; nothing is inferred."""
        nonlocal order
        for op_type in EDIT_ORDER[1:]:
            d = next((x for x in decisions if x.subject == op_type and x.type == "TRANSFORM" and x.status != "REJECTED" and x.params.get("asset_id") == subject), None)
            if d is None:
                continue
            spec = OPERATIONS[op_type]
            params = {k: v for k, v in d.params.items() if k in spec["params"]}
            out_id = f"{subject}_{op_type.split('.', 1)[1]}"
            scope = dict(scope_of.get(subject) or {"start": 0.0, "end": durations.get(subject, 0.0)})
            refs: Dict[str, Any] = {"input": current_of[subject], "output": out_id}
            if op_type == "video.speed":
                scope = {"start": 0.0, "end": round((scope["end"] - scope["start"]) / float(params["factor"]), 3)}
            if op_type == "video.overlay":
                refs["image"] = d.params["image"]
            video_ops.append(ir_operation(op_type, subject, params, [d.id], scope=scope, **refs))
            order += 1
            st = ProductionStep(id=f"step_{op_type.split('.', 1)[1]}_{subject}", order=order, skill=spec["skill"], tool=tool_for(spec["skill"]), inputs=[current_of[subject]],
                                params={"asset": subject, **params, **({"image": d.params["image"]} if op_type == "video.overlay" else {})}, outputs=[out_id],
                                depends_on=[last_of[subject]] if last_of.get(subject) else [], evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id,
                                temporal_scope=scope)
            steps.append(st)
            current_of[subject], last_of[subject], scope_of[subject] = out_id, st.id, scope
            summary.append(d.decision)

    def loudness_steps(subject: str, audio_path: bool = False) -> None:
        """The audio.loudness decision of a subject → one audio.loudness IR operation. On the audio production path the step's skill
        is audio_normalize (audio-production-skill NORMALIZE, with the tolerance the Skill re-measures against); otherwise the existing
        loudness_normalization (the reference engine's loudness tool). Same decision, same IR type; the compiler lowers by the plan's tool."""
        nonlocal order
        dur = (scope_of.get(subject) or {}).get("end") or durations.get(subject) or 0.0
        for d in decisions:
            if d.subject == "audio.loudness" and d.params.get("asset_id") == subject and d.decision.startswith("normalize"):
                op: Dict[str, Any] = {"type": "audio.loudness", "asset": subject, "target_lufs": d.params["target_lufs"], "true_peak": d.params["true_peak"]}
                params = {"asset": subject, "target_lufs": d.params["target_lufs"], "true_peak": d.params["true_peak"]}
                skill = "loudness_normalization"
                if audio_path:
                    skill = "audio_normalize"
                    op.update({"input": current_of[subject], "output": f"{subject}_loudnorm"})
                    for k in ("tolerance_lu", "sample_rate"):
                        if d.params.get(k) is not None:
                            op[k] = d.params[k]; params[k] = d.params[k]
                    op["temporal_scope"] = {"start": 0.0, "end": round(dur, 3)}
                ids = [d.id] + (extract_ids(subject) if audio_path else [])
                op["decision_ids"] = ids
                audio_ops.append(op)
                order += 1
                st = ProductionStep(id=f"step_loudness_{subject}", order=order, skill=skill, tool=tool_for(skill), inputs=[current_of[subject]],
                                    params=params, outputs=[f"{subject}_loudnorm"],
                                    depends_on=[last_of[subject]] if last_of.get(subject) else [], evidence=evidence_of(ids), decision_ids=ids, decision_id=d.id,
                                    temporal_scope={"start": 0.0, "end": round(dur, 3)} if dur else None)
                steps.append(st)
                current_of[subject], last_of[subject] = st.outputs[0], st.id
                summary.append(f"Normalise audio to {d.params['target_lufs']:g} LUFS / {d.params['true_peak']:g} dBTP" + (" (audio-production)" if audio_path else ""))

    def audio_steps(subject: str) -> None:
        """Audio production operations decided for `subject` (gain → channels → fade in → fade out), chained on its current audio output."""
        nonlocal order
        for op_type in AUDIO_ORDER:
            if op_type == "audio.loudness":
                continue
            if op_type in ("audio.mono", "audio.stereo", "audio.downmix"):
                d = next((x for x in decisions if x.subject == "audio.channels" and x.type == "TRANSFORM" and x.status != "REJECTED" and x.params.get("asset_id") == subject and x.params.get("operation") == op_type), None)
            else:
                d = next((x for x in decisions if x.subject == op_type and x.type == "TRANSFORM" and x.status != "REJECTED" and x.params.get("asset_id") == subject), None)
            if d is None:
                continue
            spec = AUDIO_OPERATIONS[op_type]
            params = {k: v for k, v in d.params.items() if k in spec["params"]}
            out_id = f"{subject}_{op_type.split('.', 1)[1]}"
            scope = dict(scope_of.get(subject) or {"start": 0.0, "end": durations.get(subject, 0.0)})
            ids = [d.id] + extract_ids(subject)
            audio_ops.append(ir_audio_operation(op_type, subject, params, ids, scope=scope, input=current_of[subject], output=out_id))
            order += 1
            st = ProductionStep(id=f"step_{op_type.split('.', 1)[1]}_{subject}", order=order, skill=spec["skill"], tool=tool_for(spec["skill"]), inputs=[current_of[subject]],
                                params={"asset": subject, **params}, outputs=[out_id], depends_on=[last_of[subject]] if last_of.get(subject) else [],
                                evidence=evidence_of(ids), decision_ids=ids, decision_id=d.id, temporal_scope=scope)
            steps.append(st)
            current_of[subject], last_of[subject], scope_of[subject] = out_id, st.id, scope
            summary.append(d.decision)

    def delivery_steps(subject: str, first: bool, single: bool) -> None:
        nonlocal order
        dur = (scope_of.get(subject) or {}).get("end") or durations.get(subject) or 0.0
        current, last_step = current_of[subject], last_of.get(subject)
        ex_ids = extract_ids(subject) if (subject in audio_subjects or subject == PROGRAMME_AUDIO) else []
        for d in decisions:
            if d.subject.startswith("delivery."):
                t = d.params
                ids = [d.id] + ex_ids
                if not any(x["id"] == t["id"] for x in delivery):
                    delivery.append({"id": t["id"], "preset": t.get("preset"), "platform": t.get("platform", "custom"), "artifact_type": t.get("artifact_type", "MASTER"), "decision_ids": ids})
                art = f"{subject}_delivery_{t['id']}"
                if t.get("preset"):
                    order += 1
                    exp = ProductionStep(id=f"step_export_{t['id']}" if single else f"step_export_{t['id']}_{subject}", order=order, skill="delivery_export",
                                         tool=tool_for("delivery_export"), inputs=[current], params={"preset": t["preset"], "target": t["id"]}, outputs=[art],
                                         depends_on=[last_step] if last_step else [], evidence=evidence_of(ids), decision_ids=ids, decision_id=d.id,
                                         temporal_scope={"start": 0.0, "end": round(dur, 3)} if dur else None)
                    order += 1
                    chk = ProductionStep(id=f"step_check_{t['id']}" if single else f"step_check_{t['id']}_{subject}", order=order, skill="delivery_check",
                                         tool=tool_for("delivery_check"), inputs=[art], params={"platform": t.get("platform", "custom"), "target": t["id"]}, outputs=[],
                                         depends_on=[exp.id], evidence=evidence_of(ids), decision_ids=ids, decision_id=d.id)
                    steps.extend([exp, chk])
                    outputs.append({"role": t.get("artifact_type", "MASTER"), "logical": art, "format": t["preset"], "expected": {"platform": t.get("platform", "custom"), "source": subject}})
                    if first:
                        summary.append(f"Export '{t['id']}' with preset {t['preset']} and check against {t.get('platform', 'custom')} spec")
                elif current in raw_asset_ids and techn.get(subject, {}).get("video"):
                    # generic profile, genuinely untouched (no edit/loudness step ever ran): `current` still names
                    # the raw source asset, which ArtifactStore.check_path() (ADR-022) refuses to register directly
                    # since it lives outside the workspace. Materialize it with one real stream copy (ffmpeg-skill
                    # export.py --preset copy) instead of the alias below — same bytes, but now a real in-workspace
                    # file the deliverable's Artifact can point at. Requires a video stream (export.py dies without
                    # one), so a genuinely untouched pure-audio subject on the audio-production path still falls
                    # through to the alias branch unchanged.
                    order += 1
                    exp = ProductionStep(id=f"step_export_{t['id']}" if single else f"step_export_{t['id']}_{subject}", order=order, skill="delivery_export",
                                         tool=tool_for("delivery_export"), inputs=[current], params={"preset": "copy", "target": t["id"]}, outputs=[art],
                                         depends_on=[last_step] if last_step else [], evidence=evidence_of(ids), decision_ids=ids, decision_id=d.id,
                                         temporal_scope={"start": 0.0, "end": round(dur, 3)} if dur else None)
                    steps.append(exp)
                    outputs.append({"role": t.get("artifact_type", "MASTER"), "logical": art, "format": "source", "expected": {"platform": t.get("platform", "custom"), "source": subject}})
                    if first:
                        summary.append(f"Deliver '{t['id']}' as a stream copy of the untouched source (no platform preset)")
                else:
                    outputs.append({"role": t.get("artifact_type", "MASTER"), "logical": art, "format": "source", "expected": {"platform": t.get("platform", "custom"), "source": subject}})
                    if first:
                        summary.append(f"Deliver '{t['id']}' as processed (no platform preset)")

    for asset in analysis.assets:
        if asset.id in extract_refused:
            summary.append(f"{asset.path.split('/')[-1]}: audio extraction was rejected; nothing is planned for it on the audio path")
            continue
        dur = asset.technical.get("duration") or 0.0
        start, end = 0.0, dur
        dec_ids: List[str] = []
        removed: List[List[float]] = []
        for d in decisions:
            if d.params.get("asset_id") != asset.id:
                continue
            if d.subject == "silence.leading":
                start, dec_ids = max(start, d.params["end"]), dec_ids + [d.id]
                removed.append([0.0, round(d.params["end"], 3)])
            if d.subject == "silence.trailing":
                end, dec_ids = min(end, d.params["start"]), dec_ids + [d.id]
                removed.append([round(d.params["start"], 3), round(dur, 3)])
        internal = sorted(([round(d.params["start"], 3), round(d.params["end"], 3)], d.id) for d in decisions
                          if d.params.get("asset_id") == asset.id and d.subject.startswith("silence.internal.") and d.decision.startswith("remove") and d.status != "REJECTED")
        current = asset.id
        last_step: Optional[str] = None
        if (dec_ids or internal) and end > start:
            keep = [[round(start, 3), round(end, 3)]]
            for (rs, re_), did in internal:   # cut each removal out of the kept ranges; the decision carries the range, the planner only subtracts
                nxt: List[List[float]] = []
                for ks, ke in keep:
                    if re_ <= ks or rs >= ke:
                        nxt.append([ks, ke])
                        continue
                    if rs > ks:
                        nxt.append([ks, rs])
                    if re_ < ke:
                        nxt.append([re_, ke])
                keep = nxt
                removed.append([rs, re_])
                dec_ids.append(did)
            keep = [k for k in keep if k[1] > k[0]]
            order += 1
            kept_total = sum(e - s for s, e in keep)
            if asset.id in audio_subjects:
                # audio production path: the same silence decisions as one audio.cut (explicit remove ranges; the Skill joins the remainder)
                remove = cut_ranges(removed)
                cut_ids = dec_ids + extract_ids(asset.id)
                audio_ops.append(ir_audio_operation("audio.cut", asset.id, {"remove": remove}, cut_ids, scope={"start": 0.0, "end": kept_after_cut(dur, remove)}, input=current, output=f"{asset.id}_cut"))
                st = ProductionStep(id=f"step_cut_{asset.id}", order=order, skill="audio_cut", tool=tool_for("audio_cut"), inputs=[current],
                                    params={"asset": asset.id, "remove": remove}, outputs=[f"{asset.id}_cut"], depends_on=[], evidence=evidence_of(cut_ids), decision_ids=cut_ids,
                                    decision_id=dec_ids[0], temporal_scope={"start": 0.0, "end": kept_after_cut(dur, remove)})
                scope_of[asset.id] = {"start": 0.0, "end": kept_after_cut(dur, remove)}
            else:
                video_ops.append({"type": "video.trim", "asset": asset.id, "keep": keep, "accurate": bool(frame_accurate), "decision_ids": dec_ids})
                st = ProductionStep(id=f"step_trim_{asset.id}", order=order, skill="silence_cleanup", tool=tool_for("silence_cleanup"), inputs=[current],
                                    params={"asset": asset.id, "keep": keep, "removed": removed, "accurate": bool(frame_accurate)}, outputs=[f"{asset.id}_trim"],
                                    depends_on=[], evidence=evidence_of(dec_ids), decision_ids=dec_ids, decision_id=dec_ids[0],
                                    temporal_scope={"start": keep[0][0], "end": keep[-1][1]})
                scope_of[asset.id] = {"start": 0.0, "end": round(kept_total, 3)}
            steps.append(st)
            current, last_step = st.outputs[0], st.id
            summary.append(f"Trim {asset.path.split('/')[-1]} to {start:.2f}-{end:.2f}s (removes {dur - kept_total:.2f}s of silence"
                           + (f", {len(internal)} internal pause(s) pending confirmation" if internal else "") + (", audio-production cut" if asset.id in audio_subjects else "") + ")")
        current_of[asset.id] = current
        if last_step:
            last_of[asset.id] = last_step
        if asset.id in audio_subjects:
            if audio_concat_dec is None:
                audio_steps(asset.id)
                loudness_steps(asset.id, audio_path=True)
                delivery_steps(asset.id, first=asset is analysis.assets[0], single=len(analysis.assets) == 1)
        elif concat_dec is None:
            edit_steps(asset.id, [asset.id])
            finishing_steps(asset.id, [asset.id])
            loudness_steps(asset.id)
            delivery_steps(asset.id, first=asset is analysis.assets[0], single=len(analysis.assets) == 1)
            thumbnail_steps(asset.id, [asset.id])
            qc_steps(asset.id, qc_tolerance_lu)
    if audio_concat_dec is not None:
        # ---- audio programme: the cut audio of every input, in the decided order; later audio operations apply to it
        inputs = list(audio_concat_dec.params["inputs"])
        cf = float(audio_concat_dec.params.get("crossfade") or 0.0)
        segments, total = audio_concat_segments(inputs, {a: (scope_of.get(a) or {}).get("end") or durations.get(a, 0.0) for a in inputs}, cf)
        scope = {"start": 0.0, "end": total}
        cat_ids = [audio_concat_dec.id] + extract_ids(PROGRAMME_AUDIO)
        audio_ops.append(ir_audio_operation("audio.concat", PROGRAMME_AUDIO, {"crossfade": cf}, cat_ids, scope=scope, inputs=inputs, output=PROGRAMME_AUDIO, segments=segments, timeline_duration=total))
        order += 1
        st = ProductionStep(id=f"step_concat_{PROGRAMME_AUDIO}", order=order, skill="audio_concat", tool=tool_for("audio_concat"), inputs=[current_of[a] for a in inputs],
                            params={"asset": PROGRAMME_AUDIO, "inputs": inputs, "crossfade": cf}, outputs=[PROGRAMME_AUDIO],
                            depends_on=[last_of[a] for a in inputs if last_of.get(a)], evidence=evidence_of(cat_ids), decision_ids=cat_ids, decision_id=audio_concat_dec.id,
                            temporal_scope=scope)
        steps.append(st)
        current_of[PROGRAMME_AUDIO], last_of[PROGRAMME_AUDIO], scope_of[PROGRAMME_AUDIO] = PROGRAMME_AUDIO, st.id, scope
        durations[PROGRAMME_AUDIO] = total
        summary.append(f"Join the audio of {' + '.join(inputs)} into one programme ({total:.2f}s)")
        audio_steps(PROGRAMME_AUDIO)
        loudness_steps(PROGRAMME_AUDIO, audio_path=True)
        delivery_steps(PROGRAMME_AUDIO, first=True, single=True)
    if concat_dec is not None:
        # ---- multi-source timeline: the trimmed inputs, in the decided order, become one programme; later operations apply to it
        inputs = list(concat_dec.params["inputs"])
        params = {k: v for k, v in concat_dec.params.items() if k in OPERATIONS["video.concat"]["params"]}
        segments, total = concat_segments(inputs, video_ops, durations, params.get("transition"))
        scope = {"start": 0.0, "end": total}
        video_ops.append(ir_operation("video.concat", PROGRAMME, params, [concat_dec.id], scope=scope, inputs=inputs, output=PROGRAMME, segments=segments, timeline_duration=total))
        order += 1
        st = ProductionStep(id=f"step_concat_{PROGRAMME}", order=order, skill="video_concat", tool=tool_for("video_concat"), inputs=[current_of[a] for a in inputs],
                            params={"asset": PROGRAMME, "inputs": inputs, **params}, outputs=[PROGRAMME],
                            depends_on=[last_of[a] for a in inputs if last_of.get(a)], evidence=evidence_of([concat_dec.id]), decision_ids=[concat_dec.id], decision_id=concat_dec.id,
                            temporal_scope=scope)
        steps.append(st)
        current_of[PROGRAMME], last_of[PROGRAMME], scope_of[PROGRAMME] = PROGRAMME, st.id, scope
        durations[PROGRAMME] = total
        summary.append(f"Join {' + '.join(inputs)} into one programme ({total:.2f}s)")
        edit_steps(PROGRAMME, inputs)
        finishing_steps(PROGRAMME, inputs)
        loudness_steps(PROGRAMME)
        delivery_steps(PROGRAMME, first=True, single=True)
        thumbnail_steps(PROGRAMME, inputs)
        qc_steps(PROGRAMME, qc_tolerance_lu)
    if blocked:
        summary.append("BLOCKED: " + "; ".join(d.reason for d in blocked))
    if no_tool:
        summary.append("BLOCKED: no executable tool selected for skill(s) " + ", ".join(no_tool) + " (see `video-agent skills`)")
    if not steps:
        summary.append("Nothing to do: no technical clean-up needed and no delivery preset requested")
    step_dicts = [s.to_dict() for s in steps]
    dec_ids_all = sorted({i for s in steps for i in s.decision_ids})
    event_ids = sorted({e for s in steps for e in s.evidence if e.startswith("evt_")})
    cons = list(constraints or [])
    plan = ProductionPlan(id=ProductionPlan.make_id(project_id, version, step_dicts, cons), project_id=project_id, version=version, status="DRAFT",
                          objective=objective, inputs=[a.id for a in analysis.assets], steps=step_dicts, outputs=outputs, decisions=dec_ids_all, events=event_ids,
                          constraints=[{"id": c.get("id"), "key": c.get("key"), "value": c.get("value")} for c in cons],
                          provenance={"generator": PLANNER_ID, "created_at": now_iso(), "decision_ids": dec_ids_all, "event_ids": event_ids,
                                      "evidence": sorted({e for s in steps for e in s.evidence})}, summary=summary)
    return {"plan": plan.to_dict(), "version": version, "steps": step_dicts, "summary": summary, "video_ops": video_ops, "audio_ops": audio_ops, "delivery": delivery,
            "captions_ops": captions_ops, "graphics_ops": graphics_ops, "color_ops": color_ops, "qc": qc_plan}
