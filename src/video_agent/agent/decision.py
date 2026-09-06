"""Decision engine (domain layer): Requirements + Observations + Inferences + Policy + Constraints + Capabilities + Skills → Decisions.
Risk and approval are set independently from confidence (MASTER_SPEC §16).

This module says *which* decision a situation calls for (silence, speech, loudness, delivery, capabilities, AI review items).
Every decision is constructed through `decision_engine.DecisionEngine`, which enforces the generic invariants (evidence
mandatory, grounding for executable types, approval resolved from policy with provenance and a safe default, BLOCK ⇔
BLOCKED, no executable material) and records the basis on the decision (policy / preference / constraint values with
provenance, approval resolution, intent, requirements) for `explain --decision`."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..capabilities.resolver import Capability
from ..media.analyzer import AnalysisResult
from ..models import Alternative, Decision, Inference, Intent, Requirement
from ..policy.rules import RuleSet
from ..skills.registry import SkillRegistry
from .ai_reasoning import AI_KIND_PREFIX
from .decision_engine import DecisionEngine, raise_approval, resolve_approval, resolve_setting
from .audio import OPERATIONS as AUDIO_OPERATIONS, PROGRAMME_AUDIO, SWITCH as AUDIO_SWITCH, audio_channels, channel_operation, has_video, is_audio_capable, parse_audio_requirements
from .decision_finishing import APPROVAL_KEYS as FINISHING_APPROVAL_KEYS, decide_finishing
from .editing import EDIT_ORDER, OPERATIONS, PROGRAMME, parse_edit_requirements
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
                 "video.hdr": ("video.hdr.approval", "CONFIRM"), "ai.recommendation": ("ai.recommendation.approval", "CONFIRM"),
                 # editing operations (ADR-029): CONFIRM unless the profile / request says otherwise; an explicit USER requirement waives it
                 "video.concat": ("video.concat.approval", "CONFIRM"), "video.speed": ("video.speed.approval", "CONFIRM"), "video.resize": ("video.resize.approval", "CONFIRM"),
                 "video.fit": ("video.fit.approval", "CONFIRM"), "video.fill": ("video.fill.approval", "CONFIRM"), "video.overlay": ("video.overlay.approval", "CONFIRM"),
                 # audio production path (ADR-030): CONFIRM by default (the picture of a video container is not delivered); explicit USER requirement waives it.
                 # audio.extract is waived only by its own explicit requirement `audio.extract=true` — never by the generic `audio.production` switch (ADR-033)
                 "audio.extract": ("audio.extract.approval", "CONFIRM"), "audio.gain": ("audio.gain.approval", "CONFIRM"), "audio.channels": ("audio.channels.approval", "CONFIRM"),
                 "audio.fade_in": ("audio.fade_in.approval", "CONFIRM"), "audio.fade_out": ("audio.fade_out.approval", "CONFIRM"), "audio.concat": ("audio.concat.approval", "CONFIRM")}


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

    def approval_for(subject_key: str, explicit: Optional[Requirement] = None, floor: Optional[str] = None, keys: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        key, default = (keys or APPROVAL_KEYS)[subject_key]
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

    # ---- editing operations (ADR-029): explicit `edit.*` requirements only; an invalid value is refused here (EditRequirementError)
    edits = parse_edit_requirements(m)
    video_assets = [a for a in analysis.assets if a.technical.get("video")]

    def probe_ids_of(asset_ids: List[str]) -> List[str]:
        return [o.id for o in analysis.observations if o.asset_id in asset_ids and o.kind in ("media_probe", "probe")]

    # ---- audio production path (ADR-030): explicit `audio.production` + `audio.*` requirements; the audio of the asset becomes the subject
    audio = parse_audio_requirements(m)
    audio_assets = [a for a in analysis.assets if is_audio_capable(a.technical)] if audio["production"] else []
    audio_concat_ok = False
    if audio["production"]:
        sw = m[AUDIO_SWITCH]
        a_ev = [r.id for r in audio["requirements"]]
        if edits:
            eng.decide(subject="audio.production", type="BLOCK", decision="BLOCK: audio production and video editing requested together",
                       reason="the audio path delivers audio only; edit.* operations edit the picture — a conflicting request is never resolved by guessing",
                       confidence=1.0, evidence=a_ev + [r.id for e in edits.values() for r in e["requirements"]], risk="HIGH", approval="BLOCK", provenance="USER", requirements=audio["requirements"])
        targets_req = m.get("delivery.targets")
        if targets_req and any(t.get("preset") for t in (targets_req.value or [])):
            eng.decide(subject="audio.production", type="BLOCK", decision="BLOCK: audio deliverable with a video platform preset",
                       reason="the delivery targets of this profile export with a video preset; an audio-only deliverable cannot take it (use a profile without presets)",
                       confidence=1.0, evidence=a_ev + [targets_req.id], risk="HIGH", approval="BLOCK", provenance="USER", requirements=audio["requirements"] + [targets_req])
        for a in analysis.assets:
            if not is_audio_capable(a.technical):
                eng.decide(subject="audio.production", type="BLOCK", decision=f"BLOCK: {a.id} has no audio stream", reason="audio production needs an audio stream (unsupported input, not guessed)",
                           confidence=1.0, evidence=a_ev + probe_ids_of([a.id]), risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": a.id}, requirements=audio["requirements"])
        if audio.get("concat"):
            if len(audio_assets) < 2:
                eng.decide(subject="audio.concat", type="BLOCK", decision="BLOCK: audio concat needs two or more inputs with audio", reason=f"{len(audio_assets)} input(s) carry audio",
                           confidence=1.0, evidence=a_ev + probe_ids_of([a.id for a in analysis.assets]), risk="HIGH", approval="BLOCK", provenance="USER", requirements=audio["requirements"])
            else:
                audio_concat_ok = True
                cf = float(audio.get("crossfade") or 0.0)
                eng.decide(subject="audio.concat", type="TRANSFORM", decision="audio concat " + " + ".join(a.id for a in audio_assets) + f" → {PROGRAMME_AUDIO}" + (f" (crossfade {cf:g}s)" if cf else ""),
                           reason=f"user asked to join the audio of the inputs in the given order ({sw.source}); the cut inputs become one programme",
                           confidence=1.0, evidence=a_ev + probe_ids_of([a.id for a in audio_assets]), risk=AUDIO_OPERATIONS["audio.concat"]["risk"],
                           approval=approval_for("audio.concat", explicit=m["audio.concat"]),   # the concat's own explicit requirement; the switch enables the path only (ADR-033 / ADR-034)
                           provenance="USER", params={"asset_id": PROGRAMME_AUDIO, "inputs": [a.id for a in audio_assets], "crossfade": cf}, requirements=audio["requirements"], serves_intent=None)
                cap_block("audio_concat", "capability.audio_concat")
    concat_ok = False
    if "video.concat" in edits:
        req = edits["video.concat"]["requirements"]
        ev = [r.id for r in req] + probe_ids_of([a.id for a in analysis.assets])
        if len(video_assets) < 2 or len(video_assets) != len(analysis.assets):
            eng.decide(subject="video.concat", type="BLOCK", decision="BLOCK: concat needs two or more inputs with a video stream",
                       reason=f"{len(video_assets)} of {len(analysis.assets)} input(s) carry a video stream; an ambiguous or unsupported multi-source request is never guessed",
                       confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK", provenance="USER", params={"inputs": [a.id for a in analysis.assets]}, requirements=req)
        else:
            p = edits["video.concat"]["params"]
            concat_ok = True
            eng.decide(subject="video.concat", type="TRANSFORM", decision="concat " + " + ".join(a.id for a in video_assets) + f" → {PROGRAMME}" + (f" ({p['transition']['type']} {p['transition']['duration']:g}s)" if p.get("transition") else ""),
                       reason=f"user asked to join the inputs in the given order ({req[0].source}); the trimmed inputs become one programme timeline",
                       confidence=1.0, evidence=ev, risk=OPERATIONS["video.concat"]["risk"], approval=approval_for("video.concat", explicit=req[0]), provenance="USER",
                       params={"asset_id": PROGRAMME, "inputs": [a.id for a in video_assets], **p}, requirements=req, serves_intent=None)
            cap_block("video_concat", "capability.video_concat")
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
            cap_block("audio_cut" if asset in audio_assets else "silence_cleanup", "capability.audio_cut" if asset in audio_assets else "capability.silence_cleanup")
        # ---- loudness
        want_norm = m.get("audio.normalize")
        target = m.get("audio.loudness.target_lufs")
        off = next((i for i in infs if i.kind == "loudness_off_target"), None)
        amb = next((i for i in infs if i.kind == "ambience_not_programme"), None)
        silent = next((i for i in infs if i.kind == "audio_silent"), None)
        loud_obs = [o.id for o in analysis.observations if o.asset_id == asset.id and o.kind == "loudness"]
        tol = resolve_setting(rules, "audio.loudness.tolerance_lu", 2.0)
        if want_norm and want_norm.value in (True, "auto") and target is not None and not asset.technical.get("audio"):
            eng.decide(subject="audio.loudness", type="SKIP", decision="skip", reason=f"{asset.id} has no audio stream; loudness normalization needs one (unsupported input, not guessed)",
                       confidence=1.0, evidence=[target.id] + probe_ids_of([asset.id]), risk="LOW", approval="AUTO", provenance="USER" if want_norm.provenance == "USER" else target.provenance,
                       params={"asset_id": asset.id}, requirements=[want_norm, target], serves_intent=_serves(intent, "audio.loudness"))
        elif want_norm and want_norm.value in (True, "auto") and target is not None and asset.technical.get("audio"):
            if silent:
                eng.decide(subject="audio.loudness", type="SKIP", decision="skip", reason=silent.statement, confidence=silent.confidence, evidence=[silent.id] + silent.evidence, risk="LOW", approval="AUTO",
                           provenance="INFERRED", requirements=[want_norm, target], serves_intent=_serves(intent, "audio.loudness"))
            elif amb and want_norm.provenance != "USER":
                eng.decide(subject="audio.loudness", type="SKIP", decision="skip", reason=amb.statement, confidence=amb.confidence, evidence=[amb.id] + amb.evidence, risk="MEDIUM", approval="AUTO",
                           alternatives=[Alternative(f"normalize to {target.value:g} LUFS", "force normalisation", "raises noise floor").to_dict()], provenance="INFERRED",
                           requirements=[want_norm, target], serves_intent=_serves(intent, "audio.loudness"))
            elif concat_ok or (audio_concat_ok and asset in audio_assets):
                pass   # the joined programme is normalised once (decision below); a per-input normalisation would be undone by the join
            elif off or want_norm.provenance == "USER":
                tp = m.get("audio.loudness.true_peak")
                ev = ([off.id] + off.evidence) if off else [target.id] + loud_obs
                on_audio_path = asset in audio_assets
                extra = ({"tolerance_lu": float(tol["value"])} | ({"sample_rate": audio["sample_rate"]} if audio.get("sample_rate") else {})) if on_audio_path else {}
                eng.decide(subject="audio.loudness", type="TRANSFORM", decision=f"normalize to {float(target.value):g} LUFS / {float(tp.value) if tp else -1:g} dBTP",
                           reason=(off.statement if off else "user asked for normalisation") + f"; target from {target.provenance.lower()} ({target.source})" + ("; audio production path: the Skill re-measures its output against the tolerance" if on_audio_path else ""),
                           confidence=off.confidence if off else 1.0, evidence=ev + ([r.id for r in audio["requirements"]] if on_audio_path else []), risk="LOW", approval=approval_for("audio.loudness"),
                           provenance="USER" if want_norm.provenance == "USER" else target.provenance,
                           params={"asset_id": asset.id, "target_lufs": float(target.value), "true_peak": float(tp.value) if tp else -1.0, **extra},
                           settings=[tol], requirements=[want_norm, target] + ([tp] if tp else []) + (audio["requirements"] if on_audio_path else []), serves_intent=_serves(intent, "audio.loudness"))
                cap_block("audio_normalize" if on_audio_path else "loudness_normalization", "capability.audio_normalize" if on_audio_path else "capability.loudness_normalization")
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
    # ---- loudness of the concat programme (one decision for the joined output; evidence: every input's measurement)
    if concat_ok:
        want_norm = m.get("audio.normalize")
        target = m.get("audio.loudness.target_lufs")
        offs = [i for i in inferences if i.kind == "loudness_off_target" and i.asset_id in {a.id for a in video_assets}]
        loud_obs = [o.id for o in analysis.observations if o.kind == "loudness" and o.asset_id in {a.id for a in video_assets}]
        if want_norm and want_norm.value in (True, "auto") and target is not None and all(a.technical.get("audio") for a in video_assets) and (offs or want_norm.provenance == "USER") and (loud_obs or offs):
            tp = m.get("audio.loudness.true_peak")
            eng.decide(subject="audio.loudness", type="TRANSFORM", decision=f"normalize {PROGRAMME} to {float(target.value):g} LUFS / {float(tp.value) if tp else -1:g} dBTP",
                       reason=("; ".join(i.statement for i in offs) if offs else "user asked for normalisation") + f"; applied once to the joined programme; target from {target.provenance.lower()} ({target.source})",
                       confidence=min(i.confidence for i in offs) if offs else 1.0, evidence=[i.id for i in offs] + [e for i in offs for e in i.evidence] + loud_obs, risk="LOW",
                       approval=approval_for("audio.loudness"), provenance="USER" if want_norm.provenance == "USER" else target.provenance,
                       params={"asset_id": PROGRAMME, "target_lufs": float(target.value), "true_peak": float(tp.value) if tp else -1.0},
                       settings=[resolve_setting(rules, "audio.loudness.tolerance_lu", 2.0)], requirements=[want_norm, target] + ([tp] if tp else []), serves_intent=_serves(intent, "audio.loudness"))
            cap_block("loudness_normalization", "capability.loudness_normalization")
    # ---- audio production operations on the audio programme (concat) or on each audio subject (ADR-030)
    if audio["production"] and audio_assets:
        sw = m[AUDIO_SWITCH]
        a_ev = [r.id for r in audio["requirements"]]
        if audio_concat_ok:
            want_norm, target = m.get("audio.normalize"), m.get("audio.loudness.target_lufs")
            offs = [i for i in inferences if i.kind == "loudness_off_target" and i.asset_id in {a.id for a in audio_assets}]
            loud_obs = [o.id for o in analysis.observations if o.kind == "loudness" and o.asset_id in {a.id for a in audio_assets}]
            if want_norm and want_norm.value in (True, "auto") and target is not None and (offs or want_norm.provenance == "USER") and (loud_obs or offs):
                tp = m.get("audio.loudness.true_peak")
                tol = resolve_setting(rules, "audio.loudness.tolerance_lu", 2.0)
                eng.decide(subject="audio.loudness", type="TRANSFORM", decision=f"normalize {PROGRAMME_AUDIO} to {float(target.value):g} LUFS / {float(tp.value) if tp else -1:g} dBTP",
                           reason=("; ".join(i.statement for i in offs) if offs else "user asked for normalisation") + f"; applied once to the joined audio programme; target from {target.provenance.lower()} ({target.source})",
                           confidence=min(i.confidence for i in offs) if offs else 1.0, evidence=[i.id for i in offs] + [e for i in offs for e in i.evidence] + loud_obs + a_ev, risk="LOW",
                           approval=approval_for("audio.loudness"), provenance="USER" if want_norm.provenance == "USER" else target.provenance,
                           params={"asset_id": PROGRAMME_AUDIO, "target_lufs": float(target.value), "true_peak": float(tp.value) if tp else -1.0, "tolerance_lu": float(tol["value"]),
                                   **({"sample_rate": audio["sample_rate"]} if audio.get("sample_rate") else {})},
                           settings=[tol], requirements=[want_norm, target] + ([tp] if tp else []) + audio["requirements"], serves_intent=_serves(intent, "audio.loudness"))
                cap_block("audio_normalize", "capability.audio_normalize")
        a_subjects = [(PROGRAMME_AUDIO, [a.id for a in audio_assets])] if audio_concat_ok else [(a.id, [a.id]) for a in audio_assets]
        normalised = {d.params.get("asset_id") for d in decs if d.subject == "audio.loudness" and d.type == "TRANSFORM"}
        for subject, sources in a_subjects:
            src = [a for a in analysis.assets if a.id in sources]
            ev = a_ev + probe_ids_of(sources)
            planned_ops: List[str] = []
            if audio.get("gain") is not None:
                eng.decide(subject="audio.gain", type="TRANSFORM", decision=f"gain {audio['gain']:+g} dB on {subject}", reason=f"user asked for a fixed gain ({sw.source})", confidence=1.0, evidence=ev,
                           risk=AUDIO_OPERATIONS["audio.gain"]["risk"], approval=approval_for("audio.gain", explicit=m["audio.gain"]), provenance="USER",
                           params={"asset_id": subject, "gain_db": audio["gain"]}, requirements=audio["requirements"], serves_intent=None)
                planned_ops.append("audio_gain")
            if audio.get("channels"):
                chans = [c for c in (audio_channels(a.technical) for a in src) if c is not None]
                ch: Optional[int] = max(chans) if chans and len(chans) == len(src) else None   # a concat programme carries the widest input's layout (Skill contract)
                op_type, why = channel_operation(audio["channels"], ch)
                if op_type == "BLOCK":
                    eng.decide(subject="audio.channels", type="BLOCK", decision=f"BLOCK: {audio['channels']} on {subject}", reason=why + "; a channel layout the Skill cannot produce is never guessed",
                               confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": subject, "layout": audio["channels"]}, requirements=audio["requirements"])
                elif op_type is None:
                    eng.decide(subject="audio.channels", type="KEEP", decision=f"keep {audio['channels']} on {subject}", reason=why + " (probe); no operation needed", confidence=1.0, evidence=ev,
                               risk="LOW", approval="AUTO", provenance="OBSERVED", params={"asset_id": subject, "layout": audio["channels"], "channels": ch}, requirements=audio["requirements"])
                else:
                    eng.decide(subject="audio.channels", type="TRANSFORM", decision=f"{op_type} on {subject}: {why}", reason=f"user asked for {audio['channels']} ({sw.source}); the probe reports {ch} channel(s)",
                               confidence=1.0, evidence=ev, risk=AUDIO_OPERATIONS[op_type]["risk"], approval=approval_for("audio.channels", explicit=m["audio.channels"]), provenance="USER",
                               params={"asset_id": subject, "layout": audio["channels"], "operation": op_type, "channels": ch}, requirements=audio["requirements"], serves_intent=None)
                    planned_ops.append(AUDIO_OPERATIONS[op_type]["skill"])
            for key in ("fade_in", "fade_out"):
                if audio.get(key) is not None:
                    eng.decide(subject=f"audio.{key}", type="TRANSFORM", decision=f"{key.replace('_', ' ')} {audio[key]:g}s on {subject}", reason=f"user asked for a {key.replace('_', ' ')} ({sw.source})",
                               confidence=1.0, evidence=ev, risk="LOW", approval=approval_for(f"audio.{key}", explicit=m[f"audio.{key}"]), provenance="USER",
                               params={"asset_id": subject, "duration": audio[key]}, requirements=audio["requirements"], serves_intent=None)
                    planned_ops.append(f"audio_{key}")
            if audio.get("sample_rate") and subject not in normalised:
                eng.decide(subject="audio.sample_rate", type="BLOCK", decision=f"BLOCK: resample to {audio['sample_rate']} Hz on {subject}",
                           reason="the Skill has no standalone resample; a sample rate is only applied by a loudness normalisation, and none was decided for this subject",
                           confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": subject, "sample_rate": audio["sample_rate"]}, requirements=audio["requirements"])
            for a in src:
                if has_video(a.technical):
                    # the switch enables the path; it is not the confirmation of the extraction. Only the dedicated `audio.extract=true` requirement
                    # is "the user asked for exactly this" (ADR-033) — the generic switch is never passed as `explicit`
                    extract_req = m["audio.extract"] if audio.get("extract") else None
                    eng.decide(subject="audio.extract", type="TRANSFORM", decision=f"deliver the audio track of {a.id} only",
                               reason=f"audio production was asked for ({sw.source}); the picture is not delivered on this path"
                               + (f"; the user confirmed the extraction up front (audio.extract from {extract_req.source})" if extract_req is not None else "; delivering audio only needs confirmation"),
                               confidence=1.0, evidence=a_ev + probe_ids_of([a.id]), risk="MEDIUM", approval=approval_for("audio.extract", explicit=extract_req), provenance="USER",
                               params={"asset_id": a.id}, requirements=audio["requirements"], serves_intent=None)
            cut_planned = any(d.subject in ("silence.leading", "silence.trailing") or (d.subject.startswith("silence.internal.") and d.decision.startswith("remove")) for d in decs if d.params.get("asset_id") in sources and d.status != "REJECTED")
            if not planned_ops and subject not in normalised and not cut_planned and not audio_concat_ok and any(has_video(a.technical) for a in src):
                eng.decide(subject="audio.production", type="BLOCK", decision=f"BLOCK: nothing to do on the audio path for {subject}",
                           reason="a video container on the audio path needs at least one audio operation (cut / normalise / gain / channels / fade); none was decided", confidence=1.0,
                           evidence=ev, risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": subject}, requirements=audio["requirements"])
            for sk in planned_ops:
                cap_block(sk, f"capability.{sk}")
    # ---- single-source editing operations on the programme (concat) or on each video asset, in the fixed order
    if "video.fit" in edits and "video.fill" in edits:
        req = edits["video.fit"]["requirements"] + edits["video.fill"]["requirements"]
        eng.decide(subject="video.fit", type="BLOCK", decision="BLOCK: fit and fill both requested", reason="edit.fit (letterbox) and edit.fill (crop) contradict each other; a conflicting request is never resolved by guessing",
                   confidence=1.0, evidence=[r.id for r in req], risk="HIGH", approval="BLOCK", provenance="USER", params={"ops": ["video.fit", "video.fill"]}, requirements=req)
        edits = {k: v for k, v in edits.items() if k not in ("video.fit", "video.fill")}
    subjects = [(PROGRAMME, [a.id for a in video_assets])] if concat_ok else [(a.id, [a.id]) for a in analysis.assets]
    for op in EDIT_ORDER[1:]:
        if op not in edits:
            continue
        req = edits[op]["requirements"]
        p = edits[op]["params"]
        for subject, sources in subjects:
            src = [a for a in analysis.assets if a.id in sources]
            ev = [r.id for r in req] + probe_ids_of(sources)
            if not all(a.technical.get("video") for a in src):
                eng.decide(subject=op, type="BLOCK", decision=f"BLOCK: {op} on {subject} (no video stream)", reason=f"{subject} has no video stream; {op} is a video operation (unsupported input, not guessed)",
                           confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": subject}, requirements=req)
                continue
            words = ", ".join(f"{k}={v}" for k, v in p.items() if k != "image") or "defaults"
            eng.decide(subject=op, type="TRANSFORM", decision=f"{op} on {subject}: {words}" + (f" (image {Path(p['image']).name})" if op == "video.overlay" else ""),
                       reason=f"user asked for {op} ({req[0].source}); applied to {'the joined programme' if subject == PROGRAMME else 'the input'} after the trims",
                       confidence=1.0, evidence=ev, risk=OPERATIONS[op]["risk"], approval=approval_for(op, explicit=req[0]), provenance="USER",
                       params={"asset_id": subject, **p}, requirements=req, serves_intent=None)
        cap_block(OPERATIONS[op]["skill"], f"capability.{OPERATIONS[op]['skill']}")
    # ---- finishing (ADR-031 / ADR-032): subtitles, colour, motion graphics, thumbnail, QC gate — on the programme or each asset, after the edits
    def has_edit(subject: str) -> bool:
        srcs = [a.id for a in video_assets] if subject == PROGRAMME else [subject]
        return concat_ok or any(d.type in ("TRANSFORM", "REMOVE") and d.status != "REJECTED" and d.subject in OPERATIONS and d.params.get("asset_id") == subject for d in decs) \
            or any(d.type == "REMOVE" and d.status != "REJECTED" and d.subject.startswith("silence.") and d.params.get("asset_id") in srcs for d in decs)
    decide_finishing(eng, m, analysis, rules, caps, cap_block, approval_for, probe_ids_of, subjects, bool(audio["production"]), has_edit)
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
