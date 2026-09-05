"""Decision engine: Requirements + Observations + Inferences + Policy + Constraints + Capabilities + Skills → Decisions.
Risk and approval are set independently from confidence (MASTER_SPEC §16)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..capabilities.resolver import Capability
from ..media.analyzer import AnalysisResult
from ..models import Alternative, Decision, Inference, Intent, Requirement
from ..policy.rules import RuleSet
from ..skills.registry import SkillRegistry
from .ai_reasoning import AI_KIND_PREFIX
from .requirements import requirement_map

# production-skill intent → decision subjects that would already execute it (measured path)
INTENT_SUBJECTS = {"silence_cleanup": ("silence.leading", "silence.trailing"), "loudness_normalization": ("audio.loudness",),
                   "delivery_export": ("delivery.",), "delivery_check": ("delivery.",)}


def decide(reqs: List[Requirement], intent: Intent, analysis: AnalysisResult, inferences: List[Inference], rules: RuleSet,
           caps: Dict[str, Capability], registry: SkillRegistry, tool_supports=None) -> List[Decision]:
    """tool_supports: callable(tool id) -> bool from the tool router; when given, a skill whose tools no adapter supports is BLOCKED too."""
    m = requirement_map(reqs)
    decs: List[Decision] = []
    by_asset: Dict[str, List[Inference]] = {}
    for inf in inferences:
        by_asset.setdefault(inf.asset_id, []).append(inf)

    # policy conflicts surface as decisions requiring confirmation
    for c in rules.conflicts:
        decs.append(Decision(subject=f"policy.{c.key}", decision=f"keep constraint {c.constraint.value}", reason=f"request/preference {c.attempted.value} conflicts with constraint {c.constraint.id}; constraints are never overridden silently",
                             confidence=1.0, evidence=[c.constraint.id, c.attempted.id], risk="MEDIUM", approval="CONFIRM", provenance="SYSTEM"))

    def cap_block(skill: str, subject: str) -> Optional[Decision]:
        missing = registry.missing_capabilities(skill, caps)
        if missing:
            d = Decision(subject=subject, decision=f"BLOCK: skill {skill} unavailable", reason=f"required capability missing: {', '.join(missing)}", confidence=1.0,
                         evidence=[f"capability:{x}" for x in missing], risk="HIGH", approval="BLOCK", status="BLOCKED", provenance="SYSTEM", params={"skill": skill, "missing": missing})
            decs.append(d)
            return d
        if tool_supports is not None:
            tool, reason = registry.select_tool(skill, caps, tool_supports)
            if tool is None:
                d = Decision(subject=subject, decision=f"BLOCK: skill {skill} has no executable tool", reason=reason, confidence=1.0,
                             evidence=[f"skill:{skill}"], risk="HIGH", approval="BLOCK", status="BLOCKED", provenance="SYSTEM", params={"skill": skill, "missing": []})
                decs.append(d)
                return d
        return None

    for asset in analysis.assets:
        infs = by_asset.get(asset.id, [])
        dur = asset.technical.get("duration") or 0.0
        # ---- silence trimming
        want_lead = m.get("edit.trim_leading_silence")
        want_tail = m.get("edit.trim_trailing_silence")
        keep_start, keep_end = 0.0, dur
        trim_evidence: List[str] = []
        lead = next((i for i in infs if i.kind == "leading_silence_unwanted"), None)
        tail = next((i for i in infs if i.kind == "trailing_silence_unwanted"), None)
        conflicts = [i for i in infs if i.kind == "speech_silence_conflict"]

        def disputed(start: float, end: float) -> List[Inference]:
            """Speech / silence conflicts overlapping a trim range: recognised speech inside a 'technical' silence. The trim then
            needs confirmation whatever the policy says (safe side); nothing is corrected."""
            return [c for c in conflicts if c.data["silence"]["start"] < end and c.data["silence"]["end"] > start]

        if lead and want_lead and want_lead.value in (True, "auto"):
            approval = str(rules.get("silence.leading.approval", "AUTO"))
            if want_lead.provenance == "USER" and approval == "CONFIRM":
                approval = "AUTO"  # the user asked explicitly; confirmation would be redundant
            lead_conf = disputed(0.0, lead.data["end"])
            if lead_conf and approval == "AUTO":
                approval = "CONFIRM"
            keep_start = lead.data["end"]
            trim_evidence += [lead.id] + lead.evidence
            decs.append(Decision(subject="silence.leading", decision=f"trim 0.000-{keep_start:.3f}s", reason=lead.statement + f"; policy silence.leading.approval={rules.get('silence.leading.approval', 'AUTO')}"
                                 + ("; recognised speech overlaps this silence, so the trim needs confirmation" if lead_conf else ""),
                                 confidence=lead.confidence, evidence=[lead.id] + lead.evidence + [c.id for c in lead_conf], risk="MEDIUM" if lead_conf else "LOW", approval=approval,
                                 alternatives=[Alternative("keep", "leave the lead-in untouched", "output keeps the silent seconds").to_dict()],
                                 provenance="USER" if want_lead.provenance == "USER" else "INFERRED", params={"asset_id": asset.id, "start": 0.0, "end": keep_start}))
        if tail and want_tail and want_tail.value in (True, "auto"):
            approval = str(rules.get("silence.trailing.approval", "AUTO"))
            if want_tail.provenance == "USER" and approval == "CONFIRM":
                approval = "AUTO"
            tail_conf = disputed(tail.data["start"], dur)
            if tail_conf and approval == "AUTO":
                approval = "CONFIRM"
            keep_end = tail.data["start"]
            trim_evidence += [tail.id] + tail.evidence
            decs.append(Decision(subject="silence.trailing", decision=f"trim {keep_end:.3f}-{dur:.3f}s", reason=tail.statement
                                 + ("; recognised speech overlaps this silence, so the trim needs confirmation" if tail_conf else ""),
                                 confidence=tail.confidence, evidence=[tail.id] + tail.evidence + [c.id for c in tail_conf], risk="MEDIUM" if tail_conf else "LOW", approval=approval,
                                 alternatives=[Alternative("keep", "leave the tail untouched").to_dict()],
                                 provenance="USER" if want_tail.provenance == "USER" else "INFERRED", params={"asset_id": asset.id, "start": keep_end, "end": dur}))
        # ---- speech (recognition evidence): continuity is kept as a fact-backed decision; long internal pauses between speech
        # become removal *candidates*. Approval comes from policy (`silence.internal.approval`, generic / conference: CONFIRM) and is
        # never lowered below CONFIRM here: "speech was recognised" ≠ "this pause must go". Who speaks is never part of it.
        activity = next((i for i in infs if i.kind == "speech_activity"), None)
        if activity:
            decs.append(Decision(subject="speech.continuity", decision=f"keep all {activity.data['intervals']} speech interval(s)", reason=activity.statement + "; recognised speech is never removed by an automatic edit",
                                 confidence=activity.confidence, evidence=[activity.id] + activity.evidence, risk="LOW", approval="AUTO", provenance="INFERRED",
                                 params={"asset_id": asset.id, "intervals": activity.data["intervals"], "speech_seconds": activity.data["speech_seconds"]}))
        removable = [i for i in infs if i.kind == "internal_silence_removable"]
        covered = {(i.data["silence"]["start"], i.data["silence"]["end"]) for i in removable}
        internal_policy = str(rules.get("silence.internal.approval", "CONFIRM"))
        for inf in removable:
            approval = "BLOCK" if internal_policy.startswith("BLOCK") else "CONFIRM"
            decs.append(Decision(subject=f"silence.internal.{inf.data['silence']['start']:.3f}-{inf.data['silence']['end']:.3f}", decision=f"remove {inf.data['start']:.3f}-{inf.data['end']:.3f}s (candidate)",
                                 reason=inf.statement + f"; policy silence.internal.approval={internal_policy}: a pause between speech is content-adjacent and needs confirmation",
                                 confidence=inf.confidence, evidence=[inf.id] + inf.evidence, risk="MEDIUM", approval=approval, status="BLOCKED" if approval == "BLOCK" else "PROPOSED",
                                 alternatives=[Alternative("keep", "leave the pause untouched", "output keeps the silent seconds").to_dict()], provenance="INFERRED",
                                 params={"asset_id": asset.id, "start": inf.data["start"], "end": inf.data["end"], "seconds": inf.data["seconds"]}))
        for inf in (i for i in infs if i.kind == "internal_silence_candidate"):
            if (inf.data.get("start"), inf.data.get("end")) in covered:
                continue   # the same pause is a removal candidate above (one decision per pause)
            decs.append(Decision(subject="silence.internal", decision="keep", reason=inf.statement + "; internal gaps are content-adjacent, removal needs CONFIRM and is not proposed automatically",
                                 confidence=inf.confidence, evidence=[inf.id] + inf.evidence, risk="MEDIUM", approval="AUTO", provenance="INFERRED", params={"asset_id": asset.id, **inf.data}))
        for inf in (i for i in infs if i.kind == "speech_silence_conflict"):
            decs.append(Decision(subject=f"silence.conflict.{inf.data['silence']['start']:.3f}-{inf.data['silence']['end']:.3f}", decision="keep", reason=inf.statement + "; a disputed interval is never edited automatically",
                                 confidence=inf.confidence, evidence=[inf.id] + inf.evidence, risk="MEDIUM", approval="AUTO", provenance="INFERRED",
                                 params={"asset_id": asset.id, **inf.data["silence"]}))
        if keep_start > 0 or keep_end < dur or removable:
            cap_block("silence_cleanup", "capability.silence_cleanup")
        # ---- loudness
        want_norm = m.get("audio.normalize")
        target = m.get("audio.loudness.target_lufs")
        off = next((i for i in infs if i.kind == "loudness_off_target"), None)
        amb = next((i for i in infs if i.kind == "ambience_not_programme"), None)
        silent = next((i for i in infs if i.kind == "audio_silent"), None)
        if want_norm and want_norm.value in (True, "auto") and target is not None and asset.technical.get("audio"):
            if silent:
                decs.append(Decision(subject="audio.loudness", decision="skip", reason=silent.statement, confidence=silent.confidence, evidence=[silent.id] + silent.evidence, risk="LOW", approval="AUTO", provenance="INFERRED"))
            elif amb and want_norm.provenance != "USER":
                decs.append(Decision(subject="audio.loudness", decision="skip", reason=amb.statement, confidence=amb.confidence, evidence=[amb.id] + amb.evidence, risk="MEDIUM", approval="AUTO",
                                     alternatives=[Alternative(f"normalize to {target.value:g} LUFS", "force normalisation", "raises noise floor").to_dict()], provenance="INFERRED"))
            elif off or want_norm.provenance == "USER":
                tp = m.get("audio.loudness.true_peak")
                ev = ([off.id] + off.evidence) if off else [target.id]
                decs.append(Decision(subject="audio.loudness", decision=f"normalize to {float(target.value):g} LUFS / {float(tp.value) if tp else -1:g} dBTP",
                                     reason=(off.statement if off else "user asked for normalisation") + f"; target from {target.provenance.lower()} ({target.source})",
                                     confidence=off.confidence if off else 1.0, evidence=ev, risk="LOW", approval="AUTO", provenance="USER" if want_norm.provenance == "USER" else target.provenance,
                                     params={"asset_id": asset.id, "target_lufs": float(target.value), "true_peak": float(tp.value) if tp else -1.0}))
                cap_block("loudness_normalization", "capability.loudness_normalization")
            else:
                decs.append(Decision(subject="audio.loudness", decision="keep", reason=f"measured loudness within tolerance of {float(target.value):g} LUFS", confidence=0.95,
                                     evidence=[o.id for o in analysis.observations if o.asset_id == asset.id and o.kind == "loudness"], risk="LOW", approval="AUTO", provenance="OBSERVED"))
        # ---- VFR / HDR facts that change tool behaviour are surfaced as decisions too
        v = (asset.technical.get("video") or {})
        probe_ids = [o.id for o in analysis.observations if o.asset_id == asset.id and o.kind in ("media_probe", "probe")]
        if v.get("variable_frame_rate_suspected"):
            decs.append(Decision(subject="video.vfr", decision="frame-accurate cuts, conform to CFR", reason="probe reports variable frame rate; lossless (stream-copy) cuts are unreliable, so trims are compiled as frame-accurate",
                                 confidence=0.9, evidence=probe_ids, risk="LOW", approval="AUTO", provenance="OBSERVED"))
        if v.get("hdr"):
            decs.append(Decision(subject="video.hdr", decision=f"keep HDR ({v.get('hdr_format')}) through intermediates; SDR presets warn", reason="probe reports HDR; intermediates keep HDR on re-encode, platform presets output SDR without tone mapping",
                                 confidence=0.9, evidence=probe_ids, risk="MEDIUM", approval="CONFIRM", provenance="OBSERVED",
                                 alternatives=[Alternative("tone-map to SDR first", "correct colours on SDR-only platforms", "one extra re-encode").to_dict()]))
    # ---- delivery
    targets = m.get("delivery.targets")
    for t in (targets.value if targets else []):
        if t.get("preset"):
            decs.append(Decision(subject=f"delivery.{t['id']}", decision=f"export preset '{t['preset']}', check platform '{t.get('platform', 'custom')}'",
                                 reason=f"delivery target from {targets.provenance.lower()} ({targets.source}); codec/container follow the preset (format-level, mechanical)",
                                 confidence=1.0, evidence=[targets.id], risk="LOW", approval="AUTO", provenance=targets.provenance, params=dict(t)))
            cap_block("delivery_export", f"capability.delivery_export.{t['id']}")
        else:
            decs.append(Decision(subject=f"delivery.{t['id']}", decision="deliver the processed file without re-encoding to a platform preset",
                                 reason="profile target has no preset (generic: keep source format)", confidence=1.0, evidence=[targets.id], risk="LOW", approval="AUTO", provenance=targets.provenance, params=dict(t)))
    # ---- AI recommendations (provenance AI_GENERATED): proposals only. A recommendation that a measured decision already
    # covers becomes extra evidence on that decision (its confidence / risk / approval are untouched). Anything else is a
    # review item: approval from policy (default CONFIRM), risk from the skill registry, never executable by itself.
    for inf in inferences:
        if not inf.kind.startswith(AI_KIND_PREFIX):
            continue
        intent = str(inf.data.get("intent") or inf.kind[len(AI_KIND_PREFIX):])
        subjects = INTENT_SUBJECTS.get(intent, ())
        covered = [d for d in decs if d.status != "BLOCKED" and any(d.subject == sub or (sub.endswith(".") and d.subject.startswith(sub)) for sub in subjects)
                   and (d.params.get("asset_id") in (None, inf.asset_id))]
        if covered:
            for d in covered:
                if inf.id not in d.evidence:
                    d.evidence.append(inf.id)
            continue
        spec = registry.get(intent) if intent in registry.names() else None
        decs.append(Decision(subject=f"ai.{intent}", decision=f"review: AI recommends {intent}",
                             reason=inf.statement + "; no measurement supports automatic execution and AI output is not an execution authority",
                             confidence=inf.confidence, evidence=[inf.id] + inf.evidence, risk=spec.risk_level if spec else "HIGH",
                             approval=str(rules.get("ai.recommendation.approval", "CONFIRM")), provenance="AI_GENERATED",
                             params={"asset_id": inf.asset_id, "intent": intent, "executable": False, "ai_params": inf.data.get("params") or {}}))
    return decs
