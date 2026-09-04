"""Decisions → ProductionPlan (human readable) + IR sections (machine readable). The planner never emits
tool arguments; it emits IR operations that the compiler lowers."""
from __future__ import annotations

from typing import Any, Dict, List

from ..media.analyzer import AnalysisResult
from ..models import Decision


DEFAULT_TOOLS = {"silence_cleanup": "ffmpeg-skill/cut", "loudness_normalization": "ffmpeg-skill/loudness", "delivery_export": "ffmpeg-skill/export", "delivery_check": "ffmpeg-skill/check"}


def build_plan(decisions: List[Decision], analysis: AnalysisResult, version: int = 1, frame_accurate: bool = False, tools: Dict[str, str] = None) -> Dict[str, Any]:
    """`tools` is the skill → tool map produced by SkillRegistry.resolve_tools for this environment. A step whose skill
    has no selectable tool is still emitted (with tool=None) so the validator reports it; decide() blocks such plans."""
    tools = tools if tools is not None else DEFAULT_TOOLS
    steps: List[Dict[str, Any]] = []
    video_ops: List[Dict[str, Any]] = []
    audio_ops: List[Dict[str, Any]] = []
    delivery: List[Dict[str, Any]] = []
    summary: List[str] = []
    blocked = [d for d in decisions if d.approval == "BLOCK"]
    for asset in analysis.assets:
        dur = asset.technical.get("duration") or 0.0
        start, end = 0.0, dur
        dec_ids = []
        for d in decisions:
            if d.params.get("asset_id") != asset.id:
                continue
            if d.subject == "silence.leading":
                start, dec_ids = max(start, d.params["end"]), dec_ids + [d.id]
            if d.subject == "silence.trailing":
                end, dec_ids = min(end, d.params["start"]), dec_ids + [d.id]
        if dec_ids and end > start:
            video_ops.append({"type": "video.trim", "asset": asset.id, "keep": [[round(start, 3), round(end, 3)]], "accurate": bool(frame_accurate), "decision_ids": dec_ids})
            steps.append({"id": f"step_trim_{asset.id}", "skill": "silence_cleanup", "tool": tools.get("silence_cleanup"), "decision_ids": dec_ids, "params": {"asset": asset.id, "keep": [[round(start, 3), round(end, 3)]]}})
            summary.append(f"Trim {asset.path.split('/')[-1]} to {start:.2f}-{end:.2f}s (removes {dur - (end - start):.2f}s of technical silence)")
        for d in decisions:
            if d.subject == "audio.loudness" and d.params.get("asset_id") == asset.id and d.decision.startswith("normalize"):
                audio_ops.append({"type": "audio.loudness", "asset": asset.id, "target_lufs": d.params["target_lufs"], "true_peak": d.params["true_peak"], "decision_ids": [d.id]})
                steps.append({"id": f"step_loudness_{asset.id}", "skill": "loudness_normalization", "tool": tools.get("loudness_normalization"), "decision_ids": [d.id], "params": {"asset": asset.id, "target_lufs": d.params["target_lufs"], "true_peak": d.params["true_peak"]}})
                summary.append(f"Normalise audio to {d.params['target_lufs']:g} LUFS / {d.params['true_peak']:g} dBTP")
    for d in decisions:
        if d.subject.startswith("delivery."):
            t = d.params
            delivery.append({"id": t["id"], "preset": t.get("preset"), "platform": t.get("platform", "custom"), "artifact_type": t.get("artifact_type", "MASTER"), "decision_ids": [d.id]})
            if t.get("preset"):
                steps.append({"id": f"step_export_{t['id']}", "skill": "delivery_export", "tool": tools.get("delivery_export"), "decision_ids": [d.id], "params": {"preset": t["preset"], "target": t["id"]}})
                steps.append({"id": f"step_check_{t['id']}", "skill": "delivery_check", "tool": tools.get("delivery_check"), "decision_ids": [d.id], "params": {"platform": t.get("platform", "custom"), "target": t["id"]}})
                summary.append(f"Export '{t['id']}' with preset {t['preset']} and check against {t.get('platform', 'custom')} spec")
            else:
                summary.append(f"Deliver '{t['id']}' as processed (no platform preset)")
    if blocked:
        summary.append("BLOCKED: " + "; ".join(d.reason for d in blocked))
    if not steps:
        summary.append("Nothing to do: no technical clean-up needed and no delivery preset requested")
    return {"version": version, "steps": steps, "summary": summary, "video_ops": video_ops, "audio_ops": audio_ops, "delivery": delivery}
