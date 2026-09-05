"""Finishing decisions (ADR-031 / ADR-032): subtitles, colour, motion graphics, thumbnail and the QC gate — the same Decision
Engine, the same invariants (evidence mandatory, requirement + measured fact as grounding, approval from policy with an explicit
USER request waiving CONFIRM, BLOCK for anything that cannot be done without guessing). Nothing here creates a plan step; the
planner turns these decisions into steps and IR operations, and the compiler lowers them to the Skills' typed requests.

Boundaries kept on purpose:
- subtitles come from a *recognised* transcript Observation (and its SpeechEvents) of every source of the subject; a subject
  without a transcript is BLOCKED (with the hint to run the recognition), never subtitled from silence or by AI;
- speaker identity is never inferred (speaker_id stays null);
- an HDR → SDR request on an SDR source is a KEEP (nothing is tone-mapped by guessing), a picture operation on an audio
  deliverable is a BLOCK, a burn-in whose input would be the untouched source is a BLOCK (subtitle-skill takes inputs inside the
  workspace only; the source is never copied or rewritten);
- QC is a measurement: its decision is AUTO and the gate policy (`qc.warn.promotion`) is recorded on it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..capabilities.resolver import Capability
from ..media.analyzer import AnalysisResult
from ..models import Decision, Requirement
from ..policy.rules import RuleSet
from .decision_engine import DecisionEngine, resolve_setting
from .finishing import (COLOR_OPERATIONS, COLOR_ORDER, ELEMENT_DEFAULT_DURATION, ELEMENT_TYPES, GRAPHICS_SKILL, THUMBNAIL_FRAME_SKILL, THUMBNAIL_RENDER_SKILL,
                        parse_color_requirements, parse_motion_requirements, parse_thumbnail_requirements, qc_requested)
from .qc import QC_SKILL, WARN_PROMOTION_DEFAULT, WARN_PROMOTION_KEY
from .subtitles import BURN_SKILL, GENERATE_SKILL, parse_subtitle_requirements, valid_language

APPROVAL_KEYS = {"subtitle.generate": ("subtitle.generate.approval", "CONFIRM"), "subtitle.burn_in": ("subtitle.burn_in.approval", "CONFIRM"),
                 "thumbnail.render": ("thumbnail.render.approval", "CONFIRM"), "qc.check": ("qc.check.approval", "AUTO"),
                 "color.strip_dovi": ("color.strip_dovi.approval", "CONFIRM"), "color.hdr_to_sdr": ("color.hdr_to_sdr.approval", "CONFIRM"),
                 "color.primary_correction": ("color.primary_correction.approval", "CONFIRM"),
                 "color.lut": ("color.lut.approval", "CONFIRM"), "color.retag": ("color.retag.approval", "CONFIRM"),
                 "graphics.title": ("graphics.title.approval", "CONFIRM"), "graphics.lower_third": ("graphics.lower_third.approval", "CONFIRM"),
                 "graphics.text_overlay": ("graphics.text_overlay.approval", "CONFIRM"), "graphics.image_overlay": ("graphics.image_overlay.approval", "CONFIRM")}
FINISHING_SUBJECTS = ("subtitle.", "thumbnail.", "color.", "graphics.", "qc.")


def decide_finishing(eng: DecisionEngine, m: Dict[str, Requirement], analysis: AnalysisResult, rules: RuleSet, caps: Dict[str, Capability],
                     cap_block: Callable[[str, str], Optional[Decision]], approval_for: Callable[..., Dict[str, Any]], probe_ids_of: Callable[[List[str]], List[str]],
                     subjects: List[Tuple[str, List[str]]], audio_production: bool, has_edit: Callable[[str], bool]) -> None:
    """`subjects`: (subject id, source asset ids) of the video path (the concat programme or each asset). `has_edit(subject)` says
    whether an earlier decision already produces an intermediate for the subject (trim / concat / edit / colour / graphics) —
    the burn-in needs one. Appends to `eng.decisions`."""
    sub = parse_subtitle_requirements(m)
    cols = parse_color_requirements(m)
    mots = parse_motion_requirements(m)
    th = parse_thumbnail_requirements(m)
    qc = qc_requested(m)
    wanted = sub["enabled"] or bool(cols) or bool(mots) or th["enabled"]
    if audio_production and wanted:
        reqs = sub["requirements"] + [r for c in cols.values() for r in c["requirements"]] + [r for e in mots.values() for r in e["requirements"]] + th["requirements"]
        eng.decide(subject="audio.production", type="BLOCK", decision="BLOCK: audio production and picture finishing requested together",
                   reason="the audio path delivers audio only; subtitles / colour / graphics / thumbnail finish the picture — a conflicting request is never resolved by guessing",
                   confidence=1.0, evidence=[r.id for r in reqs] + [m["audio.production"].id], risk="HIGH", approval="BLOCK", provenance="USER", requirements=reqs)
        return
    assets = {a.id: a for a in analysis.assets}

    def video_ok(subject: str, sources: List[str], op: str, reqs: List[Requirement]) -> bool:
        if all((assets[a].technical or {}).get("video") for a in sources if a in assets):
            return True
        eng.decide(subject=op, type="BLOCK", decision=f"BLOCK: {op} on {subject} (no video stream)", reason=f"{subject} has no video stream; {op} finishes the picture (unsupported input, not guessed)",
                   confidence=1.0, evidence=[r.id for r in reqs] + probe_ids_of(sources), risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": subject}, requirements=reqs)
        return False

    finished: Dict[str, bool] = {s: has_edit(s) for s, _ in subjects}
    # ---- colour (fixed order; the Skill executes one operation per request)
    for op in COLOR_ORDER:
        if op not in cols:
            continue
        req, p = cols[op]["requirements"], cols[op]["params"]
        spec = COLOR_OPERATIONS[op]
        for subject, sources in subjects:
            ev = [r.id for r in req] + probe_ids_of(sources)
            if not video_ok(subject, sources, op, req):
                continue
            v = ((assets[sources[0]].technical or {}).get("video") or {}) if sources[0] in assets else {}
            if op == "color.hdr_to_sdr" and not v.get("hdr"):
                eng.decide(subject=op, type="KEEP", decision=f"keep {subject} as SDR", reason=f"the probe reports no HDR transfer on {sources[0]} ({v.get('color_transfer') or 'unknown transfer'}); nothing is tone-mapped by guessing",
                           confidence=1.0, evidence=ev, risk="LOW", approval="AUTO", provenance="OBSERVED", params={"asset_id": subject}, requirements=req)
                continue
            words = ", ".join(f"{k}={Path(v_).name if k == 'lut' else v_}" for k, v_ in p.items()) or "no parameters"
            eng.decide(subject=op, type="TRANSFORM", decision=f"{op} on {subject}: {words}", reason=f"user asked for {op} ({req[0].source}); applied to {'the joined programme' if len(sources) > 1 else 'the input'} after the edits",
                       confidence=1.0, evidence=ev, risk=spec["risk"], approval=approval_for(op, explicit=req[0], keys=APPROVAL_KEYS), provenance="USER",
                       params={"asset_id": subject, **p}, requirements=req, serves_intent=None)
            finished[subject] = True
        cap_block(spec["skill"], f"capability.{spec['skill']}")
    # ---- motion graphics (one decision per element type; one render per subject in the plan)
    for typ in ELEMENT_TYPES:
        if typ not in mots:
            continue
        el = mots[typ]
        req = el["requirements"]
        subj_key = f"graphics.{typ}"
        for subject, sources in subjects:
            ev = [r.id for r in req] + probe_ids_of(sources)
            if not video_ok(subject, sources, subj_key, req):
                continue
            cap = caps.get(f"motion-graphics:{typ}")
            if cap is None or getattr(cap, "status", "MISSING") not in ("AVAILABLE", "DEGRADED"):
                eng.decide(subject=f"capability.motion_graphics:{typ}", type="BLOCK", decision=f"BLOCK: motion-graphics element {typ} unavailable",
                           reason=f"capability motion-graphics:{typ} is {getattr(cap, 'status', 'MISSING')}: the Skill's doctor could not confirm the filters it needs (never guessed)",
                           confidence=1.0, evidence=[f"capability:motion-graphics:{typ}"] + [r.id for r in req], risk="HIGH", approval="BLOCK", provenance="SYSTEM",
                           params={"skill": GRAPHICS_SKILL, "missing": [f"motion-graphics:{typ}"]}, requirements=req)
                continue
            settings: List[Dict[str, Any]] = []
            start = el.get("start")
            if start is None:
                st = resolve_setting(rules, f"motion.{typ}.start", 0.0)
                start, settings = float(st["value"]), settings + [st]
            end = el.get("end")
            if end is None and typ in ELEMENT_DEFAULT_DURATION:
                du = resolve_setting(rules, f"motion.{typ}.duration", ELEMENT_DEFAULT_DURATION[typ])
                end, settings = round(start + float(du["value"]), 3), settings + [du]
            gparams: Dict[str, Any] = {"asset_id": subject, "type": typ, "start": round(float(start), 3), "end": (round(float(end), 3) if end is not None else None), "parameters": dict(el["params"])}
            if el.get("fade") is not None:
                gparams["fade"] = float(el["fade"])
            label = el["params"].get("title") or el["params"].get("name") or el["params"].get("text") or Path(el["params"].get("image", "")).name
            eng.decide(subject=subj_key, type="TRANSFORM", decision=f"{typ} '{label}' on {subject} from {gparams['start']:g}s to {'the end' if end is None else f'{end:g}s'}",
                       reason=f"user asked for a {typ.replace('_', ' ')} ({req[0].source}); rendered on the finished picture" + (" (start / duration from policy defaults)" if settings else ""),
                       confidence=1.0, evidence=ev, risk="MEDIUM", approval=approval_for(subj_key, explicit=req[0], keys=APPROVAL_KEYS), provenance="USER", params=gparams,
                       settings=settings, requirements=req, serves_intent=None)
            finished[subject] = True
    if mots:
        cap_block(GRAPHICS_SKILL, f"capability.{GRAPHICS_SKILL}")
    # ---- subtitles: generate (sidecar) and optional burn-in, from the recognised transcripts of the subject's sources
    if sub["enabled"]:
        req = sub["requirements"]
        fmt_setting = None
        fmt = sub["format"]
        if fmt is None:
            fmt_setting = resolve_setting(rules, "subtitle.format", "srt")
            fmt = str(fmt_setting["value"]).lower() if str(fmt_setting["value"]).lower() in ("srt", "vtt") else "srt"
        for subject, sources in subjects:
            if not video_ok(subject, sources, "subtitle.generate", req):
                continue
            transcripts = [o for o in analysis.observations if o.kind == "transcript" and o.asset_id in sources]
            have = {o.asset_id for o in transcripts}
            missing = [a for a in sources if a not in have]
            if missing:
                eng.decide(subject="subtitle.generate", type="BLOCK", decision=f"BLOCK: no transcript for {', '.join(missing)}",
                           reason="subtitles are cues of a recognised transcript; none was measured for these inputs (run the plan with `--kind transcript` and a transcription engine); nothing is subtitled by guessing",
                           confidence=1.0, evidence=[r.id for r in req] + probe_ids_of(sources), risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": subject, "missing": missing}, requirements=req)
                continue
            speech = [e.id for e in analysis.timeline.events if e.type == "SPEECH" and e.timeline_id in {f"asset:{a}" for a in sources}]
            segs = sum(len((o.data or {}).get("segments") or []) for o in transcripts)
            ev = [r.id for r in req] + sorted(o.id for o in transcripts) + speech[:64]
            if segs == 0:
                eng.decide(subject="subtitle.generate", type="KEEP", decision=f"no subtitles for {subject}: nothing recognised", reason="the transcripts carry no segment; an empty subtitle file is not produced",
                           confidence=1.0, evidence=ev, risk="LOW", approval="AUTO", provenance="OBSERVED", params={"asset_id": subject}, requirements=req)
                continue
            lang = valid_language((transcripts[0].data or {}).get("language"))
            eng.decide(subject="subtitle.generate", type="TRANSFORM", decision=f"subtitles ({fmt}) for {subject}: {segs} recognised segment(s) → cues on the delivered timeline",
                       reason=f"user asked for subtitles ({req[0].source}); cues are the recognised text of the transcript(s), re-timed through the trims / concat / speed of the plan; who speaks is unknown",
                       confidence=1.0, evidence=ev, risk="LOW",
                       approval=approval_for("subtitle.generate", explicit=req[0], keys=APPROVAL_KEYS), provenance="USER",
                       params={"asset_id": subject, "format": fmt, "language": lang, "sources": list(sources), "transcript_ids": sorted(o.id for o in transcripts), "segments": segs,
                               **({"constraints": sub["constraints"]} if sub["constraints"] else {})},
                       settings=[fmt_setting] if fmt_setting else [], requirements=req, serves_intent=None)
            if sub["burn_in"]:
                br = m["subtitle.burn_in"]
                if not finished.get(subject):
                    eng.decide(subject="subtitle.burn_in", type="BLOCK", decision=f"BLOCK: burn-in on {subject} without an intermediate",
                               reason="subtitle-skill burns into inputs inside the agent workspace only; without a trim / concat / edit / colour / graphics step the input would be the untouched source (never rewritten)",
                               confidence=1.0, evidence=ev + [br.id], risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": subject}, requirements=req + [br])
                else:
                    eng.decide(subject="subtitle.burn_in", type="TRANSFORM", decision=f"burn the subtitles into the picture of {subject}", reason=f"user asked for a burn-in ({br.source}); the sidecar is still produced",
                               confidence=1.0, evidence=ev + [br.id], risk="MEDIUM", approval=approval_for("subtitle.burn_in", explicit=br, keys=APPROVAL_KEYS), provenance="USER",
                               params={"asset_id": subject, "format": "srt"}, requirements=req + [br], serves_intent=None)
        cap_block(GENERATE_SKILL, f"capability.{GENERATE_SKILL}")
        if sub["burn_in"]:
            cap_block(BURN_SKILL, f"capability.{BURN_SKILL}")
    # ---- thumbnail
    if th["enabled"]:
        req = th["requirements"]
        tsettings: List[Dict[str, Any]] = []
        fmt = th["format"]
        if fmt is None:
            s_ = resolve_setting(rules, "thumbnail.format", "png")
            fmt, tsettings = (str(s_["value"]).lower() if str(s_["value"]).lower() in ("png", "jpeg") else "png"), tsettings + [s_]
        ratio = None
        if th["at"] is None:
            r_ = resolve_setting(rules, "thumbnail.at_ratio", 0.5)
            ratio, tsettings = max(0.0, min(1.0, float(r_["value"]))), tsettings + [r_]
        font_size = th["font_size"]
        position = th["position"]
        if th["text"] is not None:
            if font_size is None:
                f_ = resolve_setting(rules, "thumbnail.font_size", 48)
                font_size, tsettings = int(f_["value"]), tsettings + [f_]
            if position is None:
                p_ = resolve_setting(rules, "thumbnail.position", "bottom")
                position, tsettings = (str(p_["value"]) if str(p_["value"]) in ("center", "top", "bottom") else "bottom"), tsettings + [p_]
        skill = THUMBNAIL_RENDER_SKILL if th["text"] is not None else THUMBNAIL_FRAME_SKILL
        for subject, sources in subjects:
            if not video_ok(subject, sources, "thumbnail.render", req):
                continue
            params: Dict[str, Any] = {"asset_id": subject, "format": fmt, "at": th["at"], "at_ratio": ratio, "skill": skill}
            if th["text"] is not None:
                params.update({"text": th["text"], "font_size": font_size, "position": position, "font_id": "sans-bold", "color": "#FFFFFF"})
            where = f"at {th['at']:g}s" if th["at"] is not None else f"at {ratio:g} of the delivered timeline"
            eng.decide(subject="thumbnail.render", type="TRANSFORM", decision=f"thumbnail ({fmt}) of {subject} {where}" + (" with a caption" if th["text"] is not None else ""),
                       reason=f"user asked for a thumbnail ({req[0].source}); the frame is taken from the finished picture" + (" (position from policy defaults)" if ratio is not None else ""),
                       confidence=1.0, evidence=[r.id for r in req] + probe_ids_of(sources), risk="LOW", approval=approval_for("thumbnail.render", explicit=req[0], keys=APPROVAL_KEYS),
                       provenance="USER", params=params, settings=tsettings, requirements=req, serves_intent=None)
        cap_block(skill, f"capability.{skill}")
    # ---- QC gate (a measurement: AUTO; the WARN promotion policy is recorded on the decision)
    if qc is not None:
        wp = resolve_setting(rules, WARN_PROMOTION_KEY, WARN_PROMOTION_DEFAULT)
        for subject, sources in subjects:
            eng.decide(subject="qc.check", type="DELIVER", decision=f"QC gate on the deliverables of {subject}: PASS → ready, WARN → {str(wp['value']).upper()}, FAIL → never delivered",
                       reason=f"user asked for the QC gate ({qc.source}); qc-skill measures each deliverable (and the subtitle sidecar) and the report is admitted only for the file the agent hashed itself",
                       confidence=1.0, evidence=[qc.id] + probe_ids_of(sources), risk="LOW", approval=approval_for("qc.check", explicit=qc, keys=APPROVAL_KEYS), provenance="USER",
                       params={"asset_id": subject, "warn_promotion": str(wp["value"]).upper()}, settings=[wp], requirements=[qc], serves_intent=None)
        cap_block(QC_SKILL, f"capability.{QC_SKILL}")
