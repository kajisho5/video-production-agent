"""Deterministic production planner: Decisions (with their event / observation evidence) → ProductionPlan (what to
produce, in which order, on which inputs, with which registry-selected skill / tool) + IR sections (the execution
contract the compiler lowers). The planner never emits tool arguments, never executes, never takes tool ids from anyone
but the registry map it is given (ADR-021)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..media.analyzer import AnalysisResult
from ..models import Decision, Inference, now_iso
from .production_plan import PLANNER_ID, ProductionPlan, ProductionStep


def build_plan(decisions: List[Decision], analysis: AnalysisResult, tools: Dict[str, str], version: int = 1, frame_accurate: bool = False,
               project_id: str = "", constraints: Optional[List[Dict[str, Any]]] = None, objective: str = "", inferences: Optional[List[Inference]] = None) -> Dict[str, Any]:
    """Decisions (+ the events / observations they rest on) → ProductionPlan + IR sections.
    `tools` is the skill → tool map produced by SkillRegistry.resolve_tools for this environment; it is the only
    source of tool ids here (the planner has no default engine). A step whose skill is absent from the map is emitted
    with tool=None and the plan is marked BLOCKED so the validator / decide() refuse it explicitly.
    Returns {"plan": ProductionPlan.to_dict(), "steps", "summary", "video_ops", "audio_ops", "delivery"}."""
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
    delivery: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    summary: List[str] = []
    blocked = [d for d in decisions if d.approval == "BLOCK"]
    order = 0
    for asset in analysis.assets:
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
        current = asset.id
        last_step: Optional[str] = None
        if dec_ids and end > start:
            keep = [[round(start, 3), round(end, 3)]]
            video_ops.append({"type": "video.trim", "asset": asset.id, "keep": keep, "accurate": bool(frame_accurate), "decision_ids": dec_ids})
            order += 1
            st = ProductionStep(id=f"step_trim_{asset.id}", order=order, skill="silence_cleanup", tool=tool_for("silence_cleanup"), inputs=[current],
                                params={"asset": asset.id, "keep": keep, "removed": removed, "accurate": bool(frame_accurate)}, outputs=[f"{asset.id}_trim"],
                                depends_on=[], evidence=evidence_of(dec_ids), decision_ids=dec_ids, decision_id=dec_ids[0],
                                temporal_scope={"start": keep[0][0], "end": keep[0][1]})
            steps.append(st)
            current, last_step = st.outputs[0], st.id
            summary.append(f"Trim {asset.path.split('/')[-1]} to {start:.2f}-{end:.2f}s (removes {dur - (end - start):.2f}s of technical silence)")
        for d in decisions:
            if d.subject == "audio.loudness" and d.params.get("asset_id") == asset.id and d.decision.startswith("normalize"):
                audio_ops.append({"type": "audio.loudness", "asset": asset.id, "target_lufs": d.params["target_lufs"], "true_peak": d.params["true_peak"], "decision_ids": [d.id]})
                order += 1
                st = ProductionStep(id=f"step_loudness_{asset.id}", order=order, skill="loudness_normalization", tool=tool_for("loudness_normalization"), inputs=[current],
                                    params={"asset": asset.id, "target_lufs": d.params["target_lufs"], "true_peak": d.params["true_peak"]}, outputs=[f"{asset.id}_loudnorm"],
                                    depends_on=[last_step] if last_step else [], evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id,
                                    temporal_scope={"start": 0.0, "end": round(dur, 3)} if dur else None)
                steps.append(st)
                current, last_step = st.outputs[0], st.id
                summary.append(f"Normalise audio to {d.params['target_lufs']:g} LUFS / {d.params['true_peak']:g} dBTP")
        for d in decisions:
            if d.subject.startswith("delivery."):
                t = d.params
                if not any(x["id"] == t["id"] for x in delivery):
                    delivery.append({"id": t["id"], "preset": t.get("preset"), "platform": t.get("platform", "custom"), "artifact_type": t.get("artifact_type", "MASTER"), "decision_ids": [d.id]})
                art = f"{asset.id}_delivery_{t['id']}"
                if t.get("preset"):
                    order += 1
                    exp = ProductionStep(id=f"step_export_{t['id']}" if len(analysis.assets) == 1 else f"step_export_{t['id']}_{asset.id}", order=order, skill="delivery_export",
                                         tool=tool_for("delivery_export"), inputs=[current], params={"preset": t["preset"], "target": t["id"]}, outputs=[art],
                                         depends_on=[last_step] if last_step else [], evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id,
                                         temporal_scope={"start": 0.0, "end": round(dur, 3)} if dur else None)
                    order += 1
                    chk = ProductionStep(id=f"step_check_{t['id']}" if len(analysis.assets) == 1 else f"step_check_{t['id']}_{asset.id}", order=order, skill="delivery_check",
                                         tool=tool_for("delivery_check"), inputs=[art], params={"platform": t.get("platform", "custom"), "target": t["id"]}, outputs=[],
                                         depends_on=[exp.id], evidence=evidence_of([d.id]), decision_ids=[d.id], decision_id=d.id)
                    steps += [exp, chk]
                    outputs.append({"role": t.get("artifact_type", "MASTER"), "logical": art, "format": t["preset"], "expected": {"platform": t.get("platform", "custom"), "source": asset.id}})
                    if asset is analysis.assets[0]:
                        summary.append(f"Export '{t['id']}' with preset {t['preset']} and check against {t.get('platform', 'custom')} spec")
                else:
                    outputs.append({"role": t.get("artifact_type", "MASTER"), "logical": art, "format": "source", "expected": {"platform": t.get("platform", "custom"), "source": asset.id}})
                    if asset is analysis.assets[0]:
                        summary.append(f"Deliver '{t['id']}' as processed (no platform preset)")
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
    return {"plan": plan.to_dict(), "version": version, "steps": step_dicts, "summary": summary, "video_ops": video_ops, "audio_ops": audio_ops, "delivery": delivery}
