"""Deterministic production planner: Decisions (with their event / observation evidence) → ProductionPlan (what to
produce, in which order, on which inputs, with which registry-selected skill / tool) + IR sections (the execution
contract the compiler lowers). The planner never emits tool arguments, never executes, never takes tool ids from anyone
but the registry map it is given (ADR-021)."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..media.analyzer import AnalysisResult
from ..models import Decision, Inference, now_iso
from .audio import (AUDIO_ORDER, OPERATIONS as AUDIO_OPERATIONS, PROGRAMME_AUDIO, PROGRAMME_OPS as AUDIO_PROGRAMME_OPS, concat_segments as audio_concat_segments, cut_ranges, has_video,
                    ir_audio_operation, is_audio_capable, kept_after_cut, mix_segments, transcript_words, words_after_cut)
from .editing import (IMAGE_INSERT_OP, IMAGE_INSERT_SKILL, IMAGE_JOIN_SKILL, OPERATIONS, PROGRAMME, SINGLE_SOURCE_ORDER, TRIM_OP, TRIM_SKILL, concat_segments, ir_operation,
                      kept_duration)
from .image_insert import still_frame
from .finishing import (COLOR_OPERATIONS, COLOR_ORDER, ELEMENT_TYPES, GRAPHICS_SKILL, THUMBNAIL_FRAME_SKILL, THUMBNAIL_RENDER_SKILL, ir_color_operation, ir_graphics_render,
                        ir_thumbnail, picture_size)
from .handoffs import JOB_PREFIX
from .production_plan import PLANNER_ID, ProductionPlan, ProductionStep
from .qc import QC_SKILL, rules_for_subject, sidecar_rules
from .subtitles import BURN_SKILL, GENERATE_SKILL, MUX_SKILL, cues_from_segments, ir_captions_burn, ir_captions_generate, kept_ranges_of
from .takes import AUDIO_MUX_SKILL, CROSSFADE_SUBJECT, PICK_SUBJECT, SNAP_SUBJECT
from .timeline import TIMELINE_SKILL, TIMELINE_SUBJECT, ir_timeline_export, timeline_clips
from .image_prepare import PACKAGE as IMAGE_PACKAGE, describe as describe_prepare, prepare_chain, prepared_id
from .voice import OPERATIONS as VOICE_OPERATIONS, PER_INPUT as VOICE_PER_INPUT, ir_voice_operation
from .dubbing import OP as DUB_OP, PARAMS as DUB_PARAMS, SKILL as DUB_SKILL, ir_dub_operation
from .publish import SKILL as PUBLISH_SKILL, SUBJECT as PUBLISH_SUBJECT


def build_plan(decisions: List[Decision], analysis: AnalysisResult, tools: Dict[str, str], version: int = 1, frame_accurate: bool = False,
               project_id: str = "", constraints: Optional[List[Dict[str, Any]]] = None, objective: str = "", inferences: Optional[List[Inference]] = None,
               audio_production: bool = False, qc_tolerance_lu: float = 2.0) -> Dict[str, Any]:
    """Decisions (+ the events / observations they rest on) → ProductionPlan + IR sections.
    `tools` is the skill → tool map produced by SkillRegistry.resolve_tools for this environment; it is the only
    source of tool ids here (the planner has no default engine). A step whose skill is absent from the map is emitted
    with tool=None and the plan is marked BLOCKED so the validator / decide() refuse it explicitly.
    Returns {"plan": ProductionPlan.to_dict(), "steps", "summary", "video_ops", "audio_ops", "delivery", "captions_ops", "graphics_ops", "color_ops", "qc", "timelines"}
    (ADR-031 / ADR-032: the finishing sections and the QC gate; ADR-052: the editor timeline exports)."""
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
    speech_ops: List[Dict[str, Any]] = []   # ADR-085: the `speech` section (dubbing-skill DUB), written only when planned
    qc_plan: Dict[str, Any] = {"enabled": False, "decision_ids": [], "subjects": {}, "sidecars": {}}
    delivery: List[Dict[str, Any]] = []
    timelines: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    summary: List[str] = []
    blocked = [d for d in decisions if d.approval == "BLOCK"]
    order = 0
    durations = {a.id: float(a.technical.get("duration") or 0.0) for a in analysis.assets}
    concat_dec = next((d for d in decisions if d.subject == "video.concat" and d.type == "TRANSFORM" and d.status != "REJECTED"), None)
    switch_dec = next((d for d in decisions if d.subject == "video.switch" and d.type == "TRANSFORM" and d.status != "REJECTED"), None)   # ADR-045: alternative to concat_dec, mutually exclusive
    grid_dec = next((d for d in decisions if d.subject == "video.grid" and d.type == "TRANSFORM" and d.status != "REJECTED"), None)         # ADR-046: alternative to concat_dec/switch_dec, mutually exclusive
    # audio production path (ADR-030): explicit `audio.production` puts every asset with audio on it (its audio is the subject, delivered as audio)
    audio_subjects = {a.id for a in analysis.assets if is_audio_capable(a.technical)} if audio_production else set()
    # the decision that builds the audio programme: audio.concat (ADR-030) or audio.mix (ADR-063), never both
    audio_concat_dec = next((d for d in decisions if d.subject in AUDIO_PROGRAMME_OPS and d.type == "TRANSFORM" and d.status != "REJECTED"), None) if audio_production else None
    # ADR-033: a video container on the audio path is delivered as audio only under its `audio.extract` decision. Every audio operation of
    # that subject cites it (a REJECTED extraction is rejected-cited → BLOCKED; a pending CONFIRM keeps the steps PROPOSED), and once the user
    # rejected it (revise drops the proposal) nothing is planned for the asset on the audio path — the picture is never quietly dropped.
    extract_of: Dict[str, Optional[Decision]] = {a.id: next((d for d in decisions if d.subject == "audio.extract" and d.type == "TRANSFORM" and d.status != "REJECTED"
                                                              and d.params.get("asset_id") == a.id), None)
                                                 for a in analysis.assets if a.id in audio_subjects and has_video(a.technical)}
    extract_refused = {a for a, d in extract_of.items() if d is None}
    if audio_concat_dec is not None and any(a in extract_refused for a in audio_concat_dec.params.get("inputs") or []):
        summary.append(f"{audio_concat_dec.subject} not planned: the extraction of a video input was rejected")
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
            for x in decided(subject, "subtitle.mux"):   # ADR-063: the same render, mode mux — a soft track in the deliverable
                mux_out = f"{subject}_subs"
                captions_ops.append(ir_captions_burn(subject, current_of[subject], out_id, mux_out, [x.id], scope=scope, op_type="captions.mux"))
                order += 1
                mst = ProductionStep(id=f"step_mux_{subject}", order=order, skill=MUX_SKILL, tool=tool_for(MUX_SKILL), inputs=[current_of[subject], out_id], params={"asset": subject},
                                     outputs=[mux_out], depends_on=[x_ for x_ in [last_of.get(subject), st.id] if x_], evidence=evidence_of([x.id]),
                                     decision_ids=[x.id], decision_id=x.id, temporal_scope=scope)
                steps.append(mst)
                current_of[subject], last_of[subject] = mux_out, mst.id
                summary.append(x.decision)
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

    def timeline_steps(subject: str, sources: List[str]) -> None:
        """The subject's cut list as an editor timeline (ADR-052): the kept source ranges of `sources` in programme order. The step
        reads the sources themselves (the timeline references them), so it waits on no other step."""
        nonlocal order
        for d in [x for x in decisions if x.subject == TIMELINE_SUBJECT and x.type == "DELIVER" and x.status != "REJECTED" and x.params.get("asset_id") == subject]:
            clips = timeline_clips(video_ops, sources, durations)
            out_id = f"{subject}_timeline"
            # the cut list rests on the decisions that made it (the trims' silence decisions, the concat): cited too, so a rejected cut
            # makes the timeline rejected-cited as well and a pending one keeps it PROPOSED
            cut_ids = [x for op in video_ops if op.get("type") == "video.trim" and op.get("asset") in sources for x in op.get("decision_ids") or []]
            cut_ids += [concat_dec.id] if (concat_dec is not None and subject == PROGRAMME) else []
            ids = [d.id] + [x for x in dict.fromkeys(cut_ids) if x != d.id]
            rec = ir_timeline_export(subject, d.params["format"], clips, ids, out_id)
            timelines.append(rec)
            order += 1
            steps.append(ProductionStep(id=f"step_timeline_{subject}", order=order, skill=TIMELINE_SKILL, tool=tool_for(TIMELINE_SKILL), inputs=list(dict.fromkeys(sources)),
                                        params={"asset": subject, "format": d.params["format"], "clips": len(clips)}, outputs=[out_id], depends_on=[],
                                        evidence=evidence_of(ids), decision_ids=ids, decision_id=d.id))
            outputs.append({"role": "TIMELINE", "logical": out_id, "format": d.params["format"],
                            "expected": {"clips": len(clips), "duration": rec["timeline_duration"], "source": subject}})
            summary.append(f"Export the cut of {subject} as an {d.params['format']} timeline ({len(clips)} clip(s), {rec['timeline_duration']:.2f}s)")

    def qc_steps(subject: str, tolerance_lu: float) -> None:
        """One qc_check step per delivery artifact of the subject (after its check), and one for the subtitle sidecar (ADR-032)."""
        nonlocal order
        for d in [x for x in decisions if x.subject == "qc.check" and x.type == "DELIVER" and x.status != "REJECTED" and x.params.get("asset_id") == subject]:
            qc_plan["enabled"] = True
            qc_plan["decision_ids"].append(d.id)
            if d.params.get("reference"):   # ADR-058: the master the video deliverables are compared against (written only when asked)
                qc_plan["reference"] = dict(d.params["reference"])
            preset_targets = set()
            gated: List[Any] = []   # (target, deliverable, its qc step) — ADR-065: the set each one heads
            for exp in [s_ for s_ in steps if s_.skill == "delivery_export" and (s_.params or {}).get("target") and s_.outputs and s_.outputs[0].startswith(f"{subject}_delivery_")]:
                target = exp.params["target"]
                preset_targets.add(target)
                art = exp.outputs[0]
                chk = next((s_.id for s_ in steps if s_.skill == "delivery_check" and (s_.params or {}).get("target") == target and art in s_.inputs), exp.id)
                order += 1
                st = ProductionStep(id=f"step_qc_{target}_{subject}", order=order, skill=QC_SKILL, tool=tool_for(QC_SKILL), inputs=[art], params={"asset": subject, "target": target, "kind": "delivery", "artifact": art},
                                    outputs=[], depends_on=[chk], evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id)
                steps.append(st)
                gated.append((target, art, st.id))
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
                gated.append((target, art, st.id))
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
            # ADR-065: the deliverable set of each target checked together (qc-skill kind delivery_package) when the deliverable has
            # company — its subtitle sidecar and / or thumbnail; a lone deliverable is already covered by its own report
            sc_id = sidecar_of.get(subject)
            th_id = next((o for s_ in steps if s_.skill in (THUMBNAIL_FRAME_SKILL, THUMBNAIL_RENDER_SKILL) and (s_.params or {}).get("asset") == subject for o in s_.outputs), None)
            extra = [x for x in (sc_id, th_id) if x]
            for target, art, qc_step in gated if extra else []:
                order += 1
                deps = [qc_step] + [x for x in (next((s_.id for s_ in steps if s_.id == f"step_qc_captions_{subject}"), None),
                                                 next((s_.id for s_ in steps if th_id and th_id in s_.outputs), None)) if x]
                steps.append(ProductionStep(id=f"step_qc_package_{target}_{subject}", order=order, skill=QC_SKILL, tool=tool_for(QC_SKILL), inputs=[art] + extra,
                                            params={"asset": subject, "target": target, "kind": "delivery_package", "artifact": art}, outputs=[], depends_on=deps,
                                            evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id))
            if extra and gated:
                qc_plan.setdefault("packages", {})[subject] = {"targets": sorted({t for t, _a, _s in gated}), "captions": sc_id, "thumbnail": th_id}
            summary.append(d.decision)

    publish_rec: Dict[str, Any] = {}   # ADR-087: delivery.publish, written only when planned

    def publish_steps(subject: str) -> None:
        """ADR-087: the `publish.upload` decision of a subject → one step after every QC step of the subject (the upload comes after QC). No IR
        operation: the compiler never runs it; render makes the Skill's offline plan and dry run, `video-agent publish` the upload."""
        nonlocal order
        for d in [x for x in decisions if x.subject == PUBLISH_SUBJECT and x.type == "DELIVER" and x.status != "REJECTED" and x.params.get("asset_id") == subject]:
            target = d.params["target"]
            art = f"{subject}_delivery_{target}"
            qc_ids = [s_.id for s_ in steps if s_.skill == QC_SKILL and (s_.params or {}).get("asset") == subject]
            meta = {k: d.params[k] for k in ("title", "description", "tags", "privacy") if d.params.get(k) is not None}
            attach = {k: bool(d.params.get(k)) for k in ("thumbnail", "captions")}
            publish_rec.update({"type": "publish.upload", "asset": subject, "target": target, "artifact": art, "platform": d.params["platform"], **meta, **attach,
                                **({"caption_language": d.params["caption_language"]} if d.params.get("caption_language") else {}), "decision_ids": [d.id]})
            order += 1
            steps.append(ProductionStep(id=f"step_publish_{subject}", order=order, skill=PUBLISH_SKILL, tool=tool_for(PUBLISH_SKILL), inputs=[art],
                                        params={"asset": subject, "target": target, "platform": d.params["platform"], **meta, **attach}, outputs=[],
                                        depends_on=qc_ids, evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id))
            summary.append(d.decision + " -- after QC; the upload needs a human's confirmation token (video-agent publish)")

    def edit_steps(subject: str, sources: List[str]) -> None:
        """The single-source editing operations decided for `subject` (video.speed → resize → fit / fill → overlay), chained on its
        current output. Each becomes one plan step and one IR operation with allowlisted parameters; nothing is inferred."""
        nonlocal order
        for op_type in SINGLE_SOURCE_ORDER:
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
            prep_last: Optional[str] = None
            if op_type == "video.overlay":
                refs["image"] = d.params["image"]
                if d.params.get("prepare"):   # ADR-077: the image prepared by imagemagick-skill (strip → resize → pad) before the overlay composites it
                    refs["prepare"] = dict(d.params["prepare"])
                    prev: Optional[str] = None
                    for sk, tool_name, tparams in prepare_chain(d.params["prepare"]):
                        order += 1
                        pst = ProductionStep(id=f"step_{sk}_{subject}", order=order, skill=sk, tool=tool_for(sk), inputs=[prev] if prev else [],
                                             params={"asset": subject, "image": d.params["image"], **tparams}, outputs=[prepared_id(subject, tool_name)],
                                             depends_on=[prep_last] if prep_last else [], evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id)
                        steps.append(pst)   # no scope on the subject's timeline: the image is its own source
                        prev, prep_last = prepared_id(subject, tool_name), pst.id
                    summary.append(f"Prepare the overlay image {os.path.basename(d.params['image'])}: {describe_prepare(d.params['prepare'])} ({IMAGE_PACKAGE})")
            video_ops.append(ir_operation(op_type, subject, params, [d.id], scope=scope, **refs))
            order += 1
            st = ProductionStep(id=f"step_{op_type.split('.', 1)[1]}_{subject}", order=order, skill=spec["skill"], tool=tool_for(spec["skill"]), inputs=[current_of[subject]],
                                params={"asset": subject, **params, **({"image": d.params["image"]} if op_type == "video.overlay" else {})}, outputs=[out_id],
                                depends_on=[x for x in (last_of.get(subject), prep_last) if x], evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id,
                                temporal_scope=scope)
            steps.append(st)
            current_of[subject], last_of[subject], scope_of[subject] = out_id, st.id, scope
            summary.append(d.decision)

    def dub_steps(subject: str) -> None:
        """ADR-085: the `speech.dub` decision of a subject → one step (dubbing-skill DUB) and one IR `speech` operation on the subject's picture after
        its edits and before the finishing: the programme with its dialogue replaced (video stream-copied), on the same timeline."""
        nonlocal order
        for d in decided(subject, DUB_OP):
            out_id = f"{subject}_dub"
            scope = dict(scope_of.get(subject) or {"start": 0.0, "end": durations.get(subject, 0.0)})
            params = {k: d.params[k] for k in DUB_PARAMS if d.params.get(k) is not None}
            speech_ops.append(ir_dub_operation(subject, params, [d.id], current_of[subject], out_id, scope))
            order += 1
            st = ProductionStep(id=f"step_dub_{subject}", order=order, skill=DUB_SKILL, tool=tool_for(DUB_SKILL), inputs=[current_of[subject]],
                                params={"asset": subject, **params}, outputs=[out_id], depends_on=[last_of[subject]] if last_of.get(subject) else [],
                                evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id, temporal_scope=scope)
            steps.append(st)
            current_of[subject], last_of[subject], scope_of[subject] = out_id, st.id, scope
            summary.append(d.decision + " -- EXPERIMENTAL, needs confirmation")

    def image_insert_steps(subject: str, sources: List[str]) -> None:
        """ADR-070: the still of a `video.image_insert` decision → a silent clip at the picture's own frame / rate (video_image_insert),
        then the join of that clip and the finished picture in the decided order (video_concat). One IR operation records both; it runs
        after the finishing (colour / graphics / burnt-in subtitles) and before loudness, so the subject's timeline grows by the still."""
        nonlocal order
        for d in decided(subject, IMAGE_INSERT_OP):
            frame, _why = still_frame(techn.get(sources[0]) or {}, video_ops, subject) if sources else (None, "")
            w, h, fps = frame if frame is not None else (d.params["width"], d.params["height"], d.params["fps"])   # the plan's own ops first; the decision's frame otherwise
            dur = float(d.params["duration"])
            scope = dict(scope_of.get(subject) or {"start": 0.0, "end": durations.get(subject, 0.0)})
            total = round(float(scope["end"]) - float(scope["start"]) + dur, 3)
            still_id, out_id = f"{subject}_still", f"{subject}_image_insert"
            inputs = [still_id, current_of[subject]] if d.params["position"] == "before" else [current_of[subject], still_id]
            params = {"duration": dur, "position": d.params["position"], "width": w, "height": h, "fps": fps}
            video_ops.append(ir_operation(IMAGE_INSERT_OP, subject, params, [d.id], scope={"start": 0.0, "end": total}, input=current_of[subject], output=out_id,
                                          image=d.params["image"], inputs=inputs))
            order += 1
            ist = ProductionStep(id=f"step_image_insert_{subject}", order=order, skill=IMAGE_INSERT_SKILL, tool=tool_for(IMAGE_INSERT_SKILL), inputs=[],
                                 params={"asset": subject, "image": d.params["image"], **params}, outputs=[still_id], depends_on=[], evidence=evidence_of([d.id]),
                                 decision_ids=[d.id], decision_id=d.id)   # no scope on the subject's timeline: the still is its own source
            steps.append(ist)
            order += 1
            jst = ProductionStep(id=f"step_image_join_{subject}", order=order, skill=IMAGE_JOIN_SKILL, tool=tool_for(IMAGE_JOIN_SKILL), inputs=list(inputs),
                                 params={"asset": subject, "inputs": list(inputs), "width": w, "height": h, "fps": fps}, outputs=[out_id],
                                 depends_on=[x for x in (ist.id, last_of.get(subject)) if x], evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id,
                                 temporal_scope={"start": 0.0, "end": total})
            steps.append(jst)
            current_of[subject], last_of[subject], scope_of[subject] = out_id, jst.id, {"start": 0.0, "end": total}
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

    # ADR-073: voice-enhance-skill on the audio path -- bleed reduction once for all mixed inputs (as recorded), then per input
    # dereverberation and isolation, each right after that input's cut / filler removal and before the programme and the chain
    voice_decs = {op: [d for d in decisions if d.subject == op and d.type == "TRANSFORM" and d.status != "REJECTED"] for op in VOICE_OPERATIONS} if audio_production else {}
    bleed_dec = (voice_decs.get("voice.bleed_reduce") or [None])[0]
    if bleed_dec is not None and (audio_concat_dec is None or audio_concat_dec.subject != "audio.mix" or list(bleed_dec.params.get("inputs") or []) != list(audio_concat_dec.params.get("inputs") or [])):
        summary.append("voice.bleed_reduce not planned: it runs on exactly the inputs of the decided audio mix")
        bleed_dec = None
    bleed_step: List[str] = []

    def voice_steps(asset_id: str, current: str, last_step: Optional[str]) -> "tuple[str, Optional[str]]":
        """The voice-enhance steps of one input on the audio path, chained on its current output; returns the new (current, last step)."""
        nonlocal order
        if bleed_dec is not None and asset_id in (bleed_dec.params.get("inputs") or []):
            if not bleed_step:
                ins = list(bleed_dec.params["inputs"])
                ids = [bleed_dec.id] + [x for a in ins for x in extract_ids(a)]
                outs = [f"{a}_bleed_reduce" for a in ins]
                params = {k: bleed_dec.params[k] for k in VOICE_OPERATIONS["voice.bleed_reduce"]["params"] if bleed_dec.params.get(k) is not None}
                audio_ops.append(ir_voice_operation("voice.bleed_reduce", PROGRAMME_AUDIO, params, ids, inputs=ins, outputs=outs))
                order += 1
                st = ProductionStep(id=f"step_bleed_reduce_{PROGRAMME_AUDIO}", order=order, skill=VOICE_OPERATIONS["voice.bleed_reduce"]["skill"],
                                    tool=tool_for(VOICE_OPERATIONS["voice.bleed_reduce"]["skill"]), inputs=ins, params={"asset": PROGRAMME_AUDIO, "inputs": ins, **params},
                                    outputs=outs, depends_on=[], evidence=evidence_of(ids), decision_ids=ids, decision_id=bleed_dec.id)
                steps.append(st)
                bleed_step.append(st.id)
                summary.append(bleed_dec.decision + " -- EXPERIMENTAL, needs confirmation")
            current, last_step = f"{asset_id}_bleed_reduce", bleed_step[0]
        for op in VOICE_PER_INPUT:
            d = next((x for x in voice_decs.get(op) or [] if x.params.get("asset_id") == asset_id), None)
            if d is None:
                continue
            spec = VOICE_OPERATIONS[op]
            name = op.split(".", 1)[1]
            out_id = f"{asset_id}_{name}"
            scope = dict(scope_of.get(asset_id) or {"start": 0.0, "end": round(durations.get(asset_id, 0.0), 3)})
            ids = [d.id] + extract_ids(asset_id)
            params = {k: d.params[k] for k in spec["params"] if d.params.get(k) is not None}
            audio_ops.append(ir_voice_operation(op, asset_id, params, ids, scope=scope, input=current, output=out_id))
            order += 1
            st = ProductionStep(id=f"step_{name}_{asset_id}", order=order, skill=spec["skill"], tool=tool_for(spec["skill"]), inputs=[current],
                                params={"asset": asset_id, **params}, outputs=[out_id], depends_on=[last_step] if last_step else [], evidence=evidence_of(ids),
                                decision_ids=ids, decision_id=d.id, temporal_scope=scope)
            steps.append(st)
            current, last_step = out_id, st.id
            scope_of[asset_id] = scope
            summary.append(d.decision + " -- EXPERIMENTAL, needs confirmation")
        return current, last_step

    def audio_steps(subject: str) -> None:
        """Audio production operations decided for `subject` (noise reduction → gain → channels → dynamics → fade in → fade out), chained on
        its current audio output."""
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

    # take selection (ADR-053): every pick decision lists its line's takes; the takes not picked and the music bed get no step. A pick's
    # trim is its decided keep range (speech span + handles, out point on the beat when snapped), cut frame-accurately when snapped.
    pick_decs = [d for d in decisions if d.subject == PICK_SUBJECT and d.type == "TRANSFORM"]
    snap_dec = next((d for d in decisions if d.subject == SNAP_SUBJECT and d.type == "TRANSFORM" and d.status != "REJECTED"), None)
    # ADR-054: the join crossfades of the assembly; every crossfaded pick is cut frame-accurately (its audio is cut sample-accurately
    # around the same in / out points, so a keyframe cut would put the picture off its sound)
    xf_dec = next((d for d in decisions if d.subject == CROSSFADE_SUBJECT and d.type == "TRANSFORM" and d.status != "REJECTED"), None) if concat_dec is not None else None
    take_crossfade: Optional[Dict[str, Any]] = None
    take_left_out = ({t["asset_id"] for d in pick_decs for t in d.params.get("takes") or []} - {d.params["asset_id"] for d in pick_decs}) \
        | {d.params["music_bed"] for d in pick_decs if d.params.get("music_bed")}
    for asset in analysis.assets:
        if asset.id in take_left_out:
            summary.append(f"{asset.path.split('/')[-1]}: not used (a take that was not picked, or the music bed)")
            continue
        if asset.id in extract_refused:
            summary.append(f"{asset.path.split('/')[-1]}: audio extraction was rejected; nothing is planned for it on the audio path")
            continue
        dur = asset.technical.get("duration") or 0.0
        start, end = 0.0, dur
        dec_ids: List[str] = []
        removed: List[List[float]] = []
        explicit_trim = False   # ADR-062: the user's own in / out range (edit.trim) is part of this trim
        for d in decisions:
            if d.params.get("asset_id") != asset.id:
                continue
            if d.subject == TRIM_OP and d.type == "TRANSFORM":
                start, end = max(start, float(d.params["start"])), min(end, float(d.params["end"]))
                dec_ids, explicit_trim = dec_ids + [d.id], True
                removed += [r for r in ([0.0, round(float(d.params["start"]), 3)], [round(float(d.params["end"]), 3), round(dur, 3)]) if r[1] > r[0]]
            if d.subject == "silence.leading":
                start, dec_ids = max(start, d.params["end"]), dec_ids + [d.id]
                removed.append([0.0, round(d.params["end"], 3)])
            if d.subject == "silence.trailing":
                end, dec_ids = min(end, d.params["start"]), dec_ids + [d.id]
                removed.append([round(d.params["start"], 3), round(dur, 3)])
            if d.subject == PICK_SUBJECT and d.type == "TRANSFORM":
                start, end = max(start, float(d.params["keep"][0])), min(end, float(d.params["keep"][1]))
                dec_ids = dec_ids + [d.id] + ([snap_dec.id] if snap_dec is not None else []) + ([xf_dec.id] if xf_dec is not None else [])
                removed += [r for r in ([0.0, round(start, 3)], [round(end, 3), round(dur, 3)]) if r[1] > r[0]]
        internal = sorted(([round(d.params["start"], 3), round(d.params["end"], 3)], d.id) for d in decisions
                          if d.params.get("asset_id") == asset.id and d.subject.startswith("silence.internal.") and d.decision.startswith("remove") and d.status != "REJECTED")
        current = asset.id
        last_step: Optional[str] = None
        audio_remove: List[List[float]] = []   # ADR-072: what this plan cuts out of the input's audio (the filler words' times follow it)
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
                audio_remove = remove
                cut_ids = dec_ids + extract_ids(asset.id)
                audio_ops.append(ir_audio_operation("audio.cut", asset.id, {"remove": remove}, cut_ids, scope={"start": 0.0, "end": kept_after_cut(dur, remove)}, input=current, output=f"{asset.id}_cut"))
                st = ProductionStep(id=f"step_cut_{asset.id}", order=order, skill="audio_cut", tool=tool_for("audio_cut"), inputs=[current],
                                    params={"asset": asset.id, "remove": remove}, outputs=[f"{asset.id}_cut"], depends_on=[], evidence=evidence_of(cut_ids), decision_ids=cut_ids,
                                    decision_id=dec_ids[0], temporal_scope={"start": 0.0, "end": kept_after_cut(dur, remove)})
                scope_of[asset.id] = {"start": 0.0, "end": kept_after_cut(dur, remove)}
            else:
                # a beat-snapped join must land on its frame, not a keyframe; a crossfaded pick's picture must start where its audio is cut
                # ADR-062: the user's in / out points are cut where they were given (frame-accurate); one kept range is video-editing TRIM,
                # several (internal pauses removed inside it) stay the multi-range cut of silence_cleanup
                accurate = bool(frame_accurate or explicit_trim or (snap_dec is not None and snap_dec.id in dec_ids) or (xf_dec is not None and xf_dec.id in dec_ids))
                trim_skill = TRIM_SKILL if explicit_trim and len(keep) == 1 else "silence_cleanup"
                video_ops.append({"type": "video.trim", "asset": asset.id, "keep": keep, "accurate": accurate, "decision_ids": dec_ids})
                st = ProductionStep(id=f"step_trim_{asset.id}", order=order, skill=trim_skill, tool=tool_for(trim_skill), inputs=[current],
                                    params={"asset": asset.id, "keep": keep, "removed": removed, "accurate": accurate}, outputs=[f"{asset.id}_trim"],
                                    depends_on=[], evidence=evidence_of(dec_ids), decision_ids=dec_ids, decision_id=dec_ids[0],
                                    temporal_scope={"start": keep[0][0], "end": keep[-1][1]})
                scope_of[asset.id] = {"start": 0.0, "end": round(kept_total, 3)}
            steps.append(st)
            current, last_step = st.outputs[0], st.id
            summary.append(f"Trim {asset.path.split('/')[-1]} to {start:.2f}-{end:.2f}s (removes {dur - kept_total:.2f}s" + (" outside the requested in / out range" if explicit_trim else " of silence")
                           + (f", {len(internal)} internal pause(s) pending confirmation" if internal else "") + (", audio-production cut" if asset.id in audio_subjects else "") + ")")
        fr_dec = next((d for d in decisions if d.subject == "audio.filler_remove" and d.type == "TRANSFORM" and d.status != "REJECTED" and d.params.get("asset_id") == asset.id), None) \
            if asset.id in audio_subjects else None
        if fr_dec is not None:
            # ADR-072: the transcript's word timings carried onto the cut audio's timeline; the Skill matches the lexicon and joins the rest
            tr = next((o for o in analysis.observations if o.id == fr_dec.params.get("transcript_id")), None)
            words, _, _ = transcript_words(((tr.data or {}).get("segments") or []) if tr is not None else [])
            words, dropped = words_after_cut(words or [], audio_remove, dur)
            if not words:
                summary.append(f"{asset.path.split('/')[-1]}: no recognised word is left after the cut; no filler removal")
            else:
                fr_ids = [fr_dec.id] + extract_ids(asset.id)
                scope = dict(scope_of.get(asset.id) or {"start": 0.0, "end": round(dur, 3)})
                p_fr = {"words": words, "lexicon": fr_dec.params["lexicon"], **{k: fr_dec.params[k] for k in ("pad", "merge_gap", "declick_ms") if k in fr_dec.params}}
                out_id = f"{asset.id}_filler_remove"
                audio_ops.append(ir_audio_operation("audio.filler_remove", asset.id, p_fr, fr_ids, scope=scope, input=current, output=out_id))
                order += 1
                st = ProductionStep(id=f"step_filler_remove_{asset.id}", order=order, skill="audio_filler_remove", tool=tool_for("audio_filler_remove"), inputs=[current],
                                    params={"asset": asset.id, "lexicon": p_fr["lexicon"], **{k: p_fr[k] for k in ("pad", "merge_gap", "declick_ms") if k in p_fr}, "word_count": len(words)},
                                    outputs=[out_id], depends_on=[last_step] if last_step else [], evidence=evidence_of(fr_ids), decision_ids=fr_ids, decision_id=fr_dec.id,
                                    temporal_scope=scope)
                steps.append(st)
                current, last_step = out_id, st.id
                summary.append(f"Remove filler words from {asset.path.split('/')[-1]} at {len(words)} recognised word timing(s)" + (f" ({dropped} word(s) inside the cut left out)" if dropped else "")
                               + f"; at most {scope['end']:.2f}s remain (what is removed is matched by audio-production)")
        if asset.id in audio_subjects and voice_decs:
            current, last_step = voice_steps(asset.id, current, last_step)
        current_of[asset.id] = current
        if last_step:
            last_of[asset.id] = last_step
        if asset.id in audio_subjects:
            if audio_concat_dec is None:
                audio_steps(asset.id)
                loudness_steps(asset.id, audio_path=True)
                delivery_steps(asset.id, first=asset is analysis.assets[0], single=len(analysis.assets) == 1)
        elif concat_dec is None and switch_dec is None and grid_dec is None:
            edit_steps(asset.id, [asset.id])
            dub_steps(asset.id)
            finishing_steps(asset.id, [asset.id])
            image_insert_steps(asset.id, [asset.id])
            loudness_steps(asset.id)
            delivery_steps(asset.id, first=asset is analysis.assets[0], single=len(analysis.assets) == 1)
            timeline_steps(asset.id, [asset.id])
            thumbnail_steps(asset.id, [asset.id])
            qc_steps(asset.id, qc_tolerance_lu)
            publish_steps(asset.id)
    def crossfade_steps(xd: Decision, concat_step: str, total: float) -> Dict[str, Any]:
        """ADR-054: each pick's audio span (its picture range + the join handles) cut out of its source (audio-production CUT), the spans
        of every run of crossfaded joins joined with the crossfade (CONCAT), the runs joined without one, and the result muxed under the
        assembled picture (skill audio_mux). The picture and its cut list are not touched. Returns the IR record."""
        nonlocal order
        xp = xd.params
        ids = [xd.id]
        span_out: Dict[str, str] = {}
        step_of: Dict[str, str] = {}
        spans: List[Dict[str, Any]] = []
        for sp in xp["spans"]:
            a = sp["asset_id"]
            dur = durations.get(a, 0.0)
            remove = [[round(x, 6), round(y, 6)] for x, y in ([0.0, sp["audio"][0]], [sp["audio"][1], dur]) if y - x > 1e-6]
            rec = {"asset": a, "video": list(sp["video"]), "audio": list(sp["audio"]), "in_handle": sp["in_handle"], "out_handle": sp["out_handle"],
                   "programme_start": sp["programme_start"], "remove": remove}
            if remove:
                out_id = f"{a}_xfade_audio"
                order += 1
                st_ = ProductionStep(id=f"step_xfade_audio_{a}", order=order, skill="audio_cut", tool=tool_for("audio_cut"), inputs=[a], params={"asset": a, "remove": remove},
                                     outputs=[out_id], depends_on=[], evidence=evidence_of(ids), decision_ids=ids, decision_id=xd.id,
                                     temporal_scope={"start": round(sp["audio"][0], 6), "end": round(sp["audio"][1], 6)})
                steps.append(st_)
                span_out[a], step_of[out_id] = out_id, st_.id
                rec["output"] = out_id
            else:
                span_out[a] = a   # the whole source is the span: CONCAT reads the source itself
                rec["output"] = a
            spans.append(rec)
        final_id = f"{PROGRAMME}_xfade_audio"
        concats: List[Dict[str, Any]] = []
        runs = [list(r) for r in xp["runs"]]
        elements: List[str] = []
        for k, run in enumerate(runs):
            ins = [span_out[xp["spans"][i]["asset_id"]] for i in run]
            if len(run) == 1:
                elements.append(ins[0])
                continue
            out_id = final_id if len(runs) == 1 else f"{PROGRAMME}_xfade_run{k + 1}"
            concats.append({"output": out_id, "inputs": ins, "crossfade": float(xp["crossfade"])})
            elements.append(out_id)
        if len(elements) >= 2:
            concats.append({"output": final_id, "inputs": elements, "crossfade": 0.0})
        for c in concats:
            order += 1
            st_ = ProductionStep(id=f"step_xfade_concat_{c['output']}", order=order, skill="audio_concat", tool=tool_for("audio_concat"), inputs=list(c["inputs"]),
                                 params={"asset": c["output"], "inputs": list(c["inputs"]), "crossfade": c["crossfade"]}, outputs=[c["output"]],
                                 depends_on=[step_of[i] for i in c["inputs"] if i in step_of], evidence=evidence_of(ids), decision_ids=ids, decision_id=xd.id)
            steps.append(st_)
            step_of[c["output"]] = st_.id
        mux_out = f"{PROGRAMME}_xfade"
        order += 1
        st_ = ProductionStep(id=f"step_xfade_mux_{PROGRAMME}", order=order, skill=AUDIO_MUX_SKILL, tool=tool_for(AUDIO_MUX_SKILL), inputs=[current_of[PROGRAMME], final_id],
                             params={"asset": PROGRAMME, "video": current_of[PROGRAMME], "audio": final_id}, outputs=[mux_out],
                             depends_on=[concat_step, step_of[final_id]], evidence=evidence_of(ids), decision_ids=ids, decision_id=xd.id,
                             temporal_scope={"start": 0.0, "end": round(total, 3)})
        steps.append(st_)
        current_of[PROGRAMME], last_of[PROGRAMME] = mux_out, st_.id
        summary.append(f"Crossfade the audio at {xp['crossfaded']} of {len(xp['joins'])} join(s) ({float(xp['crossfade']):g}s; picture cuts unchanged, audio muxed back in sync)")
        return {"type": CROSSFADE_SUBJECT, "asset": PROGRAMME, "crossfade": float(xp["crossfade"]), "spans": spans, "joins": [dict(j) for j in xp["joins"]], "concats": concats,
                "audio": final_id, "mux": {"video": PROGRAMME, "audio": final_id, "output": mux_out}, "timeline_duration": round(total, 6), "decision_ids": ids}

    if audio_concat_dec is not None:
        # ---- audio programme: the cut audio of every input, in the decided order (concat), or every input summed from 0 s (mix, ADR-063);
        # later audio operations apply to it
        inputs = list(audio_concat_dec.params["inputs"])
        lengths = {a: (scope_of.get(a) or {}).get("end") or durations.get(a, 0.0) for a in inputs}
        op_type = audio_concat_dec.subject
        name = op_type.split(".", 1)[1]
        if op_type == "audio.mix":
            segments, total = mix_segments(inputs, lengths)
            p_ = {k: audio_concat_dec.params[k] for k in ("levels", "duck") if audio_concat_dec.params.get(k)}   # ADR-072: MIX.duck rides with the levels
        else:
            cf = float(audio_concat_dec.params.get("crossfade") or 0.0)
            segments, total = audio_concat_segments(inputs, lengths, cf)
            p_ = {"crossfade": cf}
        scope = {"start": 0.0, "end": total}
        cat_ids = [audio_concat_dec.id] + extract_ids(PROGRAMME_AUDIO)
        audio_ops.append(ir_audio_operation(op_type, PROGRAMME_AUDIO, p_, cat_ids, scope=scope, inputs=inputs, output=PROGRAMME_AUDIO, segments=segments, timeline_duration=total))
        order += 1
        skill = AUDIO_OPERATIONS[op_type]["skill"]
        st = ProductionStep(id=f"step_{name}_{PROGRAMME_AUDIO}", order=order, skill=skill, tool=tool_for(skill), inputs=[current_of[a] for a in inputs],
                            params={"asset": PROGRAMME_AUDIO, "inputs": inputs, **p_}, outputs=[PROGRAMME_AUDIO],
                            depends_on=[last_of[a] for a in inputs if last_of.get(a)], evidence=evidence_of(cat_ids), decision_ids=cat_ids, decision_id=audio_concat_dec.id,
                            temporal_scope=scope)
        steps.append(st)
        current_of[PROGRAMME_AUDIO], last_of[PROGRAMME_AUDIO], scope_of[PROGRAMME_AUDIO] = PROGRAMME_AUDIO, st.id, scope
        durations[PROGRAMME_AUDIO] = total
        summary.append(f"Mix the audio of {' + '.join(inputs)} into one programme ({total:.2f}s, as long as {inputs[0]})" if op_type == "audio.mix"
                       else f"Join the audio of {' + '.join(inputs)} into one programme ({total:.2f}s)")
        audio_steps(PROGRAMME_AUDIO)
        loudness_steps(PROGRAMME_AUDIO, audio_path=True)
        delivery_steps(PROGRAMME_AUDIO, first=True, single=True)
    if concat_dec is not None:
        # ---- multi-source timeline: the trimmed inputs, in the decided order, become one programme; later operations apply to it
        inputs = list(concat_dec.params["inputs"])
        params = {k: v for k, v in concat_dec.params.items() if k in OPERATIONS["video.concat"]["params"]}
        segments, total = concat_segments(inputs, video_ops, durations, params.get("transition"))
        scope = {"start": 0.0, "end": total}
        # a concat assembled from take picks (ADR-053) cites them too: a rejected pick makes the assembly rejected-cited, a pending one keeps it PROPOSED
        known_ids = {x.id for x in decisions}
        cat_ids = [concat_dec.id] + [x for x in concat_dec.params.get("picks") or [] if x in known_ids]
        video_ops.append(ir_operation("video.concat", PROGRAMME, params, cat_ids, scope=scope, inputs=inputs, output=PROGRAMME, segments=segments, timeline_duration=total))
        order += 1
        st = ProductionStep(id=f"step_concat_{PROGRAMME}", order=order, skill="video_concat", tool=tool_for("video_concat"), inputs=[current_of[a] for a in inputs],
                            params={"asset": PROGRAMME, "inputs": inputs, **params}, outputs=[PROGRAMME],
                            depends_on=[last_of[a] for a in inputs if last_of.get(a)], evidence=evidence_of(cat_ids), decision_ids=cat_ids, decision_id=concat_dec.id,
                            temporal_scope=scope)
        steps.append(st)
        current_of[PROGRAMME], last_of[PROGRAMME], scope_of[PROGRAMME] = PROGRAMME, st.id, scope
        durations[PROGRAMME] = total
        summary.append(f"Join {' + '.join(inputs)} into one programme ({total:.2f}s)")
        if xf_dec is not None:
            take_crossfade = crossfade_steps(xf_dec, st.id, total)
        edit_steps(PROGRAMME, inputs)
        dub_steps(PROGRAMME)
        finishing_steps(PROGRAMME, inputs)
        image_insert_steps(PROGRAMME, inputs)
        loudness_steps(PROGRAMME)
        delivery_steps(PROGRAMME, first=True, single=True)
        timeline_steps(PROGRAMME, inputs)
        thumbnail_steps(PROGRAMME, inputs)
        qc_steps(PROGRAMME, qc_tolerance_lu)
        publish_steps(PROGRAMME)
    if switch_dec is not None:
        # ---- explicit camera switch (ADR-045): unlike concat, the programme's duration is the (trimmed) reference input's
        # own duration, not a sum of segments -- multicam cuts between already-aligned sources on the reference timeline.
        inputs = list(switch_dec.params["inputs"])
        params = {k: v for k, v in switch_dec.params.items() if k in OPERATIONS["video.switch"]["params"]}
        total = kept_duration(video_ops, inputs[0], durations.get(inputs[0], 0.0))
        scope = {"start": 0.0, "end": total}
        video_ops.append(ir_operation("video.switch", PROGRAMME, params, [switch_dec.id], scope=scope, inputs=inputs, output=PROGRAMME, timeline_duration=total))
        order += 1
        st = ProductionStep(id=f"step_switch_{PROGRAMME}", order=order, skill="camera_switch", tool=tool_for("camera_switch"), inputs=[current_of[a] for a in inputs],
                            params={"asset": PROGRAMME, "inputs": inputs, **params}, outputs=[PROGRAMME],
                            depends_on=[last_of[a] for a in inputs if last_of.get(a)], evidence=evidence_of([switch_dec.id]), decision_ids=[switch_dec.id], decision_id=switch_dec.id,
                            temporal_scope=scope)
        steps.append(st)
        current_of[PROGRAMME], last_of[PROGRAMME], scope_of[PROGRAMME] = PROGRAMME, st.id, scope
        durations[PROGRAMME] = total
        summary.append(f"Switch {' + '.join(inputs)} into one programme per {params['switch']} ({total:.2f}s)")
        edit_steps(PROGRAMME, inputs)
        dub_steps(PROGRAMME)
        finishing_steps(PROGRAMME, inputs)
        image_insert_steps(PROGRAMME, inputs)
        loudness_steps(PROGRAMME)
        delivery_steps(PROGRAMME, first=True, single=True)
        thumbnail_steps(PROGRAMME, inputs)
        qc_steps(PROGRAMME, qc_tolerance_lu)
        publish_steps(PROGRAMME)
    if grid_dec is not None:
        # ---- explicit comparison grid (ADR-046): unlike concat, the programme's duration is the shortest (trimmed) input's own
        # duration by default -- grid.py runs only as long as its shortest clip unless --pad holds every shorter cell's last frame
        # out to the longest clip's duration.
        inputs = list(grid_dec.params["inputs"])
        params = {k: v for k, v in grid_dec.params.items() if k in OPERATIONS["video.grid"]["params"]}
        kept = [kept_duration(video_ops, a, durations.get(a, 0.0)) for a in inputs]
        total = max(kept) if params.get("pad") else min(kept)
        scope = {"start": 0.0, "end": total}
        video_ops.append(ir_operation("video.grid", PROGRAMME, params, [grid_dec.id], scope=scope, inputs=inputs, output=PROGRAMME, timeline_duration=total))
        order += 1
        st = ProductionStep(id=f"step_grid_{PROGRAMME}", order=order, skill="video_grid", tool=tool_for("video_grid"), inputs=[current_of[a] for a in inputs],
                            params={"asset": PROGRAMME, "inputs": inputs, **params}, outputs=[PROGRAMME],
                            depends_on=[last_of[a] for a in inputs if last_of.get(a)], evidence=evidence_of([grid_dec.id]), decision_ids=[grid_dec.id], decision_id=grid_dec.id,
                            temporal_scope=scope)
        steps.append(st)
        current_of[PROGRAMME], last_of[PROGRAMME], scope_of[PROGRAMME] = PROGRAMME, st.id, scope
        durations[PROGRAMME] = total
        summary.append(f"Composite {' + '.join(inputs)} into a {params['cols']}x{params['rows']} grid programme ({total:.2f}s)")
        edit_steps(PROGRAMME, inputs)
        dub_steps(PROGRAMME)
        finishing_steps(PROGRAMME, inputs)
        image_insert_steps(PROGRAMME, inputs)
        loudness_steps(PROGRAMME)
        delivery_steps(PROGRAMME, first=True, single=True)
        thumbnail_steps(PROGRAMME, inputs)
        qc_steps(PROGRAMME, qc_tolerance_lu)
        publish_steps(PROGRAMME)
    # ---- ecosystem jobs handed off (ADR-057): one step per requested job, on the request's assets, with no tool (BLOCKED; the
    # plan's handoff names the owning package and its install command). It waits on nothing and produces nothing the agent tracks.
    # ADR-086: a measurement an explicit requirement asked for that no tool can run here (`handoff: true`) takes the same step
    for d in [x for x in decisions if (x.subject.startswith(JOB_PREFIX) or x.params.get("handoff") is True) and x.type == "TRANSFORM" and x.status != "REJECTED"]:
        skill = d.params["skill"]
        order += 1
        steps.append(ProductionStep(id=f"step_job_{skill}", order=order, skill=skill, tool=tool_for(skill), inputs=[a.id for a in analysis.assets],
                                    params=dict(d.params.get("args") or {}), outputs=[], depends_on=[], evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id))
        summary.append(f"Hand off {skill} ({d.subject}): " + ("not runnable here" if d.params.get("handoff") else "no adapter here") + "; see the plan's handoff")
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
    return {"plan": plan.to_dict(), "version": version, "steps": step_dicts, "summary": summary, "video_ops": video_ops, "audio_ops": audio_ops, "speech_ops": speech_ops, "publish": dict(publish_rec) if publish_rec else None, "delivery": delivery,
            "captions_ops": captions_ops, "graphics_ops": graphics_ops, "color_ops": color_ops, "qc": qc_plan, "timelines": timelines, "take_crossfade": take_crossfade}
