"""Decision engine (domain layer): Requirements + Observations + Inferences + Policy + Constraints + Capabilities + Skills → Decisions.
Risk and approval are set independently from confidence (MASTER_SPEC §16).

This module says *which* decision a situation calls for (silence, speech, loudness, delivery, capabilities, AI review items).
Every decision is constructed through `decision_engine.DecisionEngine`, which enforces the generic invariants (evidence
mandatory, grounding for executable types, approval resolved from policy with provenance and a safe default, BLOCK ⇔
BLOCKED, no executable material) and records the basis on the decision (policy / preference / constraint values with
provenance, approval resolution, intent, requirements) for `explain --decision`."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..capabilities.resolver import Capability
from ..media.analyzer import AnalysisResult
from ..models import Alternative, Decision, Inference, Intent, Requirement
from ..policy.rules import RuleSet
from ..skills.registry import SkillRegistry
from .ai_reasoning import AI_KIND_PREFIX
from .decision_engine import DecisionEngine, raise_approval, resolve_approval, resolve_setting
from .requirements import requirement_map

# production-skill intent → decision subjects that would already execute it (measured path)
INTENT_SUBJECTS = {"silence_cleanup": ("silence.leading", "silence.trailing"), "loudness_normalization": ("audio.loudness",),
                   "delivery_export": ("delivery.",), "delivery_check": ("delivery.",)}
# user / profile intent (agent/intent.py vocabulary) → decision subjects that serve it (recorded in basis.intent)
SERVING_SUBJECTS = {"clean_and_deliver": ("silence.leading", "silence.trailing", "silence.internal.", "audio.loudness", "delivery."),
                    "clean_only": ("silence.leading", "silence.trailing", "silence.internal.", "audio.loudness"),
                    "inspect_and_clean": ("silence.leading", "silence.trailing", "audio.loudness"),
                    "normalize_audio": ("audio.loudness",), "cleanup_silence": ("silence.leading", "silence.trailing", "silence.internal.")}
# approval policy keys and their explicit DEFAULT when a profile / request says nothing (recorded with provenance DEFAULT)
APPROVAL_KEYS = {"silence.leading": ("silence.leading.approval", "AUTO"), "silence.trailing": ("silence.trailing.approval", "AUTO"),
                 "silence.internal": ("silence.internal.approval", "CONFIRM"), "audio.loudness": ("audio.loudness.approval", "AUTO"),
                 "delivery.export": ("delivery.export.approval", "AUTO"), "video.vfr": ("video.vfr.approval", "AUTO"),
                 "video.hdr": ("video.hdr.approval", "CONFIRM"), "ai.recommendation": ("ai.recommendation.approval", "CONFIRM")}


def _serves(intent: Intent, subject: str) -> Optional[str]:
    """Which intent (primary first, then secondary) this subject serves; None when the decision is not what was asked for."""
    for name in [intent.primary] + list(intent.secondary):
        if any(subject == s or (s.endswith(".") and subject.startswith(s)) for s in SERVING_SUBJECTS.get(name, ())):
            return name
    return None


def decide(reqs: List[Requirement], intent: Intent, analysis: AnalysisResult, inferences: List[Inference], rules: RuleSet,
           caps: Dict[str, Capability], registry: SkillRegistry, tool_supports=None) -> List[Decision]:
    """tool_supports: callable(tool id) -> bool from the tool router; when given, a skill whose tools no adapter supports is BLOCKED too."""
    m = requirement_map(reqs)
    known = DecisionEngine.evidence_index(analysis.observations, analysis.timeline.events, inferences, reqs, rules, ai_prefix=AI_KIND_PREFIX)
    eng = DecisionEngine(rules, intent, known, reqs)
    decs = eng.decisions
    by_asset: Dict[str, List[Inference]] = {}
    for inf in inferences:
        by_asset.setdefault(inf.asset_id, []).append(inf)

    def approval_for(subject_key: str, explicit: Optional[Requirement] = None, floor: Optional[str] = None) -> Dict[str, Any]:
        key, default = APPROVAL_KEYS[subject_key]
        return resolve_approval(rules, key, default, floor=floor, explicit=explicit)

    # policy conflicts surface as decisions requiring confirmation (a constraint is never overridden silently)
    for c in rules.conflicts:
        eng.decide(subject=f"policy.{c.key}", type="KEEP", decision=f"keep constraint {c.constraint.value}",
                   reason=f"request/preference {c.attempted.value} conflicts with constraint {c.constraint.id}; constraints are never overridden silently",
                   confidence=1.0, evidence=[c.constraint.id, c.attempted.id], risk="MEDIUM", approval="CONFIRM", provenance="SYSTEM",
                   settings=[resolve_setting(rules, c.key, c.constraint.value)])

    def cap_block(skill: str, subject: str) -> Optional[Decision]:
        missing = registry.missing_capabilities(skill, caps)
        if missing:
            return eng.decide(subject=subject, type="BLOCK", decision=f"BLOCK: skill {skill} unavailable", reason=f"required capability missing: {', '.join(missing)}", confidence=1.0,
                              evidence=[f"capability:{x}" for x in missing], risk="HIGH", approval="BLOCK", provenance="SYSTEM", params={"skill": skill, "missing": missing})
        if tool_supports is not None:
            tool, reason = registry.select_tool(skill, caps, tool_supports)
            if tool is None:
                return eng.decide(subject=subject, type="BLOCK", decision=f"BLOCK: skill {skill} has no executable tool", reason=reason, confidence=1.0,
                                  evidence=[f"skill:{skill}"], risk="HIGH", approval="BLOCK", provenance="SYSTEM", params={"skill": skill, "missing": []})
        return None

    for asset in analysis.assets:
        infs = by_asset.get(asset.id, [])
        dur = asset.technical.get("duration") or 0.0
        # ---- silence trimming
        want_lead = m.get("edit.trim_leading_silence")
        want_tail = m.get("edit.trim_trailing_silence")
        keep_start, keep_end = 0.0, dur
        lead = next((i for i in infs if i.kind == "leading_silence_unwanted"), None)
        tail = next((i for i in infs if i.kind == "trailing_silence_unwanted"), None)
        conflicts = [i for i in infs if i.kind == "speech_silence_conflict"]

        def disputed(start: float, end: float) -> List[Inference]:
            """Speech / silence conflicts overlapping a trim range: recognised speech inside a 'technical' silence. The trim then
            needs confirmation whatever the policy says (safe side); nothing is corrected."""
            return [c for c in conflicts if c.data["silence"]["start"] < end and c.data["silence"]["end"] > start]

        if lead and want_lead and want_lead.value in (True, "auto"):
            appr = approval_for("silence.leading", explicit=want_lead)
            lead_conf = disputed(0.0, lead.data["end"])
            if lead_conf:
                raise_approval(appr, "CONFIRM", "recognised speech overlaps this silence")
            keep_start = lead.data["end"]
            eng.decide(subject="silence.leading", type="REMOVE", decision=f"trim 0.000-{keep_start:.3f}s",
                       reason=lead.statement + f"; policy silence.leading.approval={appr['setting']['value']}" + ("; recognised speech overlaps this silence, so the trim needs confirmation" if lead_conf else ""),
                       confidence=lead.confidence, evidence=[lead.id] + lead.evidence + [c.id for c in lead_conf], risk="MEDIUM" if lead_conf else "LOW", approval=appr,
                       alternatives=[Alternative("keep", "leave the lead-in untouched", "output keeps the silent seconds").to_dict()],
                       provenance="USER" if want_lead.provenance == "USER" else "INFERRED", params={"asset_id": asset.id, "start": 0.0, "end": keep_start},
                       settings=[resolve_setting(rules, "silence.leading.min_seconds", 1.0), resolve_setting(rules, "silence.margin_seconds", lead.data.get("margin"))],
                       requirements=[want_lead], serves_intent=_serves(intent, "silence.leading"))
        if tail and want_tail and want_tail.value in (True, "auto"):
            appr = approval_for("silence.trailing", explicit=want_tail)
            tail_conf = disputed(tail.data["start"], dur)
            if tail_conf:
                raise_approval(appr, "CONFIRM", "recognised speech overlaps this silence")
            keep_end = tail.data["start"]
            eng.decide(subject="silence.trailing", type="REMOVE", decision=f"trim {keep_end:.3f}-{dur:.3f}s",
                       reason=tail.statement + ("; recognised speech overlaps this silence, so the trim needs confirmation" if tail_conf else ""),
                       confidence=tail.confidence, evidence=[tail.id] + tail.evidence + [c.id for c in tail_conf], risk="MEDIUM" if tail_conf else "LOW", approval=appr,
                       alternatives=[Alternative("keep", "leave the tail untouched").to_dict()],
                       provenance="USER" if want_tail.provenance == "USER" else "INFERRED", params={"asset_id": asset.id, "start": keep_end, "end": dur},
                       settings=[resolve_setting(rules, "silence.margin_seconds", tail.data.get("margin"))],
                       requirements=[want_tail], serves_intent=_serves(intent, "silence.trailing"))
        # ---- speech (recognition evidence): continuity is kept as a fact-backed decision; long internal pauses between speech
        # become removal *candidates*. Approval comes from policy (`silence.internal.approval`, generic / conference: CONFIRM) with a
        # CONFIRM floor: "speech was recognised" ≠ "this pause must go". Who speaks is never part of it.
        activity = next((i for i in infs if i.kind == "speech_activity"), None)
        if activity:
            eng.decide(subject="speech.continuity", type="KEEP", decision=f"keep all {activity.data['intervals']} speech interval(s)", reason=activity.statement + "; recognised speech is never removed by an automatic edit",
                       confidence=activity.confidence, evidence=[activity.id] + activity.evidence, risk="LOW", approval="AUTO", provenance="INFERRED",
                       params={"asset_id": asset.id, "intervals": activity.data["intervals"], "speech_seconds": activity.data["speech_seconds"]})
        removable = [i for i in infs if i.kind == "internal_silence_removable"]
        covered = {(i.data["silence"]["start"], i.data["silence"]["end"]) for i in removable}
        for inf in removable:
            appr = approval_for("silence.internal", floor="CONFIRM")
            eng.decide(subject=f"silence.internal.{inf.data['silence']['start']:.3f}-{inf.data['silence']['end']:.3f}", type="REMOVE",
                       decision=f"remove {inf.data['start']:.3f}-{inf.data['end']:.3f}s (candidate)",
                       reason=inf.statement + f"; policy silence.internal.approval={appr['setting']['value']}: a pause between speech is content-adjacent and needs confirmation",
                       confidence=inf.confidence, evidence=[inf.id] + inf.evidence, risk="MEDIUM", approval=appr,
                       alternatives=[Alternative("keep", "leave the pause untouched", "output keeps the silent seconds").to_dict()], provenance="INFERRED",
                       params={"asset_id": asset.id, "start": inf.data["start"], "end": inf.data["end"], "seconds": inf.data["seconds"]},
                       settings=[{**inf.data["threshold"], "key": "silence.internal.removable_min_seconds", "kind": None if inf.data["threshold"]["provenance"] == "DEFAULT" else rules.effective["silence.internal.removable_min_seconds"].kind,
                                  "rule_id": None if inf.data["threshold"]["provenance"] == "DEFAULT" else rules.effective["silence.internal.removable_min_seconds"].id, "hard": False},
                                 {**inf.data["margin"], "key": "silence.margin_seconds", "kind": None if inf.data["margin"]["provenance"] == "DEFAULT" else rules.effective["silence.margin_seconds"].kind,
                                  "rule_id": None if inf.data["margin"]["provenance"] == "DEFAULT" else rules.effective["silence.margin_seconds"].id, "hard": False}],
                       serves_intent=_serves(intent, "silence.internal."))
        for inf in (i for i in infs if i.kind == "internal_silence_candidate"):
            if (inf.data.get("start"), inf.data.get("end")) in covered:
                continue   # the same pause is a removal candidate above (one decision per pause)
            eng.decide(subject="silence.internal", type="KEEP", decision="keep", reason=inf.statement + "; internal gaps are content-adjacent, removal needs CONFIRM and is not proposed automatically",
                       confidence=inf.confidence, evidence=[inf.id] + inf.evidence, risk="MEDIUM", approval="AUTO", provenance="INFERRED", params={"asset_id": asset.id, **inf.data},
                       settings=[resolve_setting(rules, "silence.internal.approval", APPROVAL_KEYS["silence.internal"][1])])
        for inf in (i for i in infs if i.kind == "speech_silence_conflict"):
            eng.decide(subject=f"silence.conflict.{inf.data['silence']['start']:.3f}-{inf.data['silence']['end']:.3f}", type="KEEP", decision="keep",
                       reason=inf.statement + "; a disputed interval is never edited automatically",
                       confidence=inf.confidence, evidence=[inf.id] + inf.evidence, risk="MEDIUM", approval="AUTO", provenance="INFERRED",
                       params={"asset_id": asset.id, **inf.data["silence"]})
        if keep_start > 0 or keep_end < dur or removable:
            cap_block("silence_cleanup", "capability.silence_cleanup")
        # ---- loudness
        want_norm = m.get("audio.normalize")
        target = m.get("audio.loudness.target_lufs")
        off = next((i for i in infs if i.kind == "loudness_off_target"), None)
        amb = next((i for i in infs if i.kind == "ambience_not_programme"), None)
        silent = next((i for i in infs if i.kind == "audio_silent"), None)
        loud_obs = [o.id for o in analysis.observations if o.asset_id == asset.id and o.kind == "loudness"]
        tol = resolve_setting(rules, "audio.loudness.tolerance_lu", 2.0)
        if want_norm and want_norm.value in (True, "auto") and target is not None and asset.technical.get("audio"):
            if silent:
                eng.decide(subject="audio.loudness", type="SKIP", decision="skip", reason=silent.statement, confidence=silent.confidence, evidence=[silent.id] + silent.evidence, risk="LOW", approval="AUTO",
                           provenance="INFERRED", requirements=[want_norm, target], serves_intent=_serves(intent, "audio.loudness"))
            elif amb and want_norm.provenance != "USER":
                eng.decide(subject="audio.loudness", type="SKIP", decision="skip", reason=amb.statement, confidence=amb.confidence, evidence=[amb.id] + amb.evidence, risk="MEDIUM", approval="AUTO",
                           alternatives=[Alternative(f"normalize to {target.value:g} LUFS", "force normalisation", "raises noise floor").to_dict()], provenance="INFERRED",
                           requirements=[want_norm, target], serves_intent=_serves(intent, "audio.loudness"))
            elif off or want_norm.provenance == "USER":
                tp = m.get("audio.loudness.true_peak")
                ev = ([off.id] + off.evidence) if off else [target.id] + loud_obs
                eng.decide(subject="audio.loudness", type="TRANSFORM", decision=f"normalize to {float(target.value):g} LUFS / {float(tp.value) if tp else -1:g} dBTP",
                           reason=(off.statement if off else "user asked for normalisation") + f"; target from {target.provenance.lower()} ({target.source})",
                           confidence=off.confidence if off else 1.0, evidence=ev, risk="LOW", approval=approval_for("audio.loudness"),
                           provenance="USER" if want_norm.provenance == "USER" else target.provenance,
                           params={"asset_id": asset.id, "target_lufs": float(target.value), "true_peak": float(tp.value) if tp else -1.0},
                           settings=[tol], requirements=[want_norm, target] + ([tp] if tp else []), serves_intent=_serves(intent, "audio.loudness"))
                cap_block("loudness_normalization", "capability.loudness_normalization")
            elif loud_obs:   # no measurement (analysis failed) → no claim about loudness; the analysis warning records the failure
                eng.decide(subject="audio.loudness", type="KEEP", decision="keep", reason=f"measured loudness within tolerance of {float(target.value):g} LUFS", confidence=0.95,
                           evidence=loud_obs, risk="LOW", approval="AUTO", provenance="OBSERVED", settings=[tol], requirements=[want_norm, target], serves_intent=_serves(intent, "audio.loudness"))
        # ---- VFR / HDR facts that change tool behaviour are surfaced as decisions too
        v = (asset.technical.get("video") or {})
        probe_ids = [o.id for o in analysis.observations if o.asset_id == asset.id and o.kind in ("media_probe", "probe")]
        if v.get("variable_frame_rate_suspected"):
            eng.decide(subject="video.vfr", type="TRANSFORM", decision="frame-accurate cuts, conform to CFR", reason="probe reports variable frame rate; lossless (stream-copy) cuts are unreliable, so trims are compiled as frame-accurate",
                       confidence=0.9, evidence=probe_ids, risk="LOW", approval=approval_for("video.vfr"), provenance="OBSERVED")
        if v.get("hdr"):
            eng.decide(subject="video.hdr", type="KEEP", decision=f"keep HDR ({v.get('hdr_format')}) through intermediates; SDR presets warn", reason="probe reports HDR; intermediates keep HDR on re-encode, platform presets output SDR without tone mapping",
                       confidence=0.9, evidence=probe_ids, risk="MEDIUM", approval=approval_for("video.hdr"), provenance="OBSERVED",
                       alternatives=[Alternative("tone-map to SDR first", "correct colours on SDR-only platforms", "one extra re-encode").to_dict()])
    # ---- delivery
    targets = m.get("delivery.targets")
    for t in (targets.value if targets else []):
        if t.get("preset"):
            eng.decide(subject=f"delivery.{t['id']}", type="DELIVER", decision=f"export preset '{t['preset']}', check platform '{t.get('platform', 'custom')}'",
                       reason=f"delivery target from {targets.provenance.lower()} ({targets.source}); codec/container follow the preset (format-level, mechanical)",
                       confidence=1.0, evidence=[targets.id], risk="LOW", approval=approval_for("delivery.export"), provenance=targets.provenance, params=dict(t),
                       requirements=[targets], serves_intent=_serves(intent, f"delivery.{t['id']}"))
            cap_block("delivery_export", f"capability.delivery_export.{t['id']}")
        else:
            eng.decide(subject=f"delivery.{t['id']}", type="DELIVER", decision="deliver the processed file without re-encoding to a platform preset",
                       reason="profile target has no preset (generic: keep source format)", confidence=1.0, evidence=[targets.id], risk="LOW", approval="AUTO", provenance=targets.provenance, params=dict(t),
                       requirements=[targets], serves_intent=_serves(intent, f"delivery.{t['id']}"))
    # ---- AI recommendations (provenance AI_GENERATED): proposals only. A recommendation that a measured decision already
    # covers becomes extra evidence on that decision (its confidence / risk / approval are untouched). Anything else is a
    # review item: approval from policy (default CONFIRM), risk from the skill registry, never executable by itself.
    for inf in inferences:
        if not inf.kind.startswith(AI_KIND_PREFIX):
            continue
        ai_intent = str(inf.data.get("intent") or inf.kind[len(AI_KIND_PREFIX):])
        subjects = INTENT_SUBJECTS.get(ai_intent, ())
        covered_decs = [d for d in decs if d.status != "BLOCKED" and any(d.subject == sub or (sub.endswith(".") and d.subject.startswith(sub)) for sub in subjects)
                        and (d.params.get("asset_id") in (None, inf.asset_id))]
        if covered_decs:
            for d in covered_decs:
                if inf.id not in d.evidence:
                    d.evidence.append(inf.id)
                    d.basis["evidence_classes"] = sorted(set(d.basis.get("evidence_classes", [])) | {"ai"})
            continue
        spec = registry.get(ai_intent) if ai_intent in registry.names() else None
        eng.decide(subject=f"ai.{ai_intent}", type="REVIEW", decision=f"review: AI recommends {ai_intent}",
                   reason=inf.statement + "; no measurement supports automatic execution and AI output is not an execution authority",
                   confidence=inf.confidence, evidence=[inf.id] + inf.evidence, risk=spec.risk_level if spec else "HIGH",
                   approval=approval_for("ai.recommendation"), provenance="AI_GENERATED",
                   params={"asset_id": inf.asset_id, "intent": ai_intent, "executable": False, "ai_params": inf.data.get("params") or {}})
    return decs
