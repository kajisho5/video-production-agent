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

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..capabilities.resolver import Capability, export_keeps_subtitles
from ..media.analyzer import AnalysisResult
from ..models import Decision, Requirement
from ..policy.rules import RuleSet
from .decision_engine import DecisionEngine, resolve_setting
from ..tools.color_grading.adapter import OPERATION_NEEDS as COLOR_REMEDY
from .editing import EditRequirementError
from .finishing import (COLOR_OPERATIONS, COLOR_ORDER, parse_motion_options, ELEMENT_DEFAULT_DURATION, ELEMENT_TYPES, GRAPHICS_SKILL,
                        element_label, parse_color_requirements, parse_motion_requirements, parse_thumbnail_requirements, parse_qc_reference, placement_problems,
                        planned_picture_size, qc_requested)
from .html_layer import (APPROVAL_KEY as HTML_LAYER_APPROVAL_KEY, SUBJECT as HTML_LAYER_SUBJECT, decide_html_layer, element_timing, layer_id as html_layer_id,
                         parse_html_layer_requirements)
from .qc import QC_SKILL, WARN_PROMOTION_DEFAULT, WARN_PROMOTION_KEY
from .finishing import thumbnail_skill_of
from .thumbnail_features import FRAME_SKILLS, GEOMETRY_SUBJECTS


def same_aspect(frame: Dict[str, Any], size: Tuple[int, int]) -> bool:
    """ADR-104: the measured frame and the picture have the same aspect ratio (within half a percent): its face boxes scale onto the picture."""
    fw, fh = frame.get("width"), frame.get("height")
    return bool(fw and fh and size[0] and size[1]) and abs(fw / float(fh) - size[0] / float(size[1])) <= 0.005 * (size[0] / float(size[1]))
from .subtitles import BURN_SKILL, CAPTION_PROFILES, GENERATE_SKILL, MUX_SKILL, REGIONAL_PROFILES, parse_subtitle_requirements, valid_language

# ADR-071: the engine's export keeps subtitle streams from release 2.22.4 (engine #380) for every preset but gif; older engines (measured
# in SubtitleMuxRealTests) and an engine whose version cannot be read keep the ADR-063 BLOCK of a soft track with an export preset.
EXPORT_PRESETS_DROPPING_SUBTITLES = ("gif",)


APPROVAL_KEYS = {"subtitle.generate": ("subtitle.generate.approval", "CONFIRM"), "subtitle.burn_in": ("subtitle.burn_in.approval", "CONFIRM"),
                 "subtitle.mux": ("subtitle.mux.approval", "CONFIRM"),   # ADR-063: a soft subtitle track in the deliverable
                 "thumbnail.render": ("thumbnail.render.approval", "CONFIRM"), "qc.check": ("qc.check.approval", "AUTO"),
                 "color.strip_dovi": ("color.strip_dovi.approval", "CONFIRM"), "color.hdr_to_sdr": ("color.hdr_to_sdr.approval", "CONFIRM"),
                 "color.primary_correction": ("color.primary_correction.approval", "CONFIRM"),
                 "color.lut": ("color.lut.approval", "CONFIRM"), "color.retag": ("color.retag.approval", "CONFIRM"),
                 # ADR-100: CONFIRM like every colour operation; the explicit requirement waives it
                 "color.ocio": ("color.ocio.approval", "CONFIRM"), "color.match_camera": ("color.match_camera.approval", "CONFIRM"),
                 "color.legalize": ("color.legalize.approval", "CONFIRM"),
                 # one approval key per motion-graphics element type (ADR-031; every element type of the contract since ADR-064)
                 **{f"graphics.{t}": (f"graphics.{t}.approval", "CONFIRM") for t in ELEMENT_TYPES},
                 # ADR-114: the rendered layer's clip rendered by hyperframes-skill (a browser render of new graphics: floor CONFIRM)
                 HTML_LAYER_SUBJECT: HTML_LAYER_APPROVAL_KEY}
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
    mopts = parse_motion_options(m)   # ADR-102
    html_layer = parse_html_layer_requirements({k: r.value for k, r in m.items() if r.provenance != "DEFAULT"})   # ADR-114: motion.layer=hyperframes
    th = parse_thumbnail_requirements(m)
    qc = qc_requested(m)
    qc_ref = parse_qc_reference(m)
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
            words = ", ".join(f"{k}={Path(v_).name if k in ('lut', 'reference_path') or (k == 'config' and not str(v_).startswith('ocio://')) else v_}" for k, v_ in p.items()) or "no parameters"
            eng.decide(subject=op, type="TRANSFORM", decision=f"{op} on {subject}: {words}", reason=f"user asked for {op} ({req[0].source}); applied to {'the joined programme' if len(sources) > 1 else 'the input'} after the edits",
                       confidence=1.0, evidence=ev, risk=spec["risk"], approval=approval_for(op, explicit=req[0], keys=APPROVAL_KEYS), provenance="USER",
                       params={"asset_id": subject, **p}, requirements=req, serves_intent=None)
            finished[subject] = True
        cap = caps.get(f"color-grading:{spec['type']}") if spec["type"] in COLOR_REMEDY else None
        if spec["type"] in COLOR_REMEDY and (cap is None or getattr(cap, "status", "MISSING") != "AVAILABLE") and caps.get("color-grading") is not None \
                and getattr(caps["color-grading"], "status", "MISSING") == "AVAILABLE":
            # ADR-100: the package is here but its doctor reports this operation unavailable: BLOCK with what it needs (never run another way)
            eng.decide(subject=f"capability.{spec['skill']}", type="BLOCK", decision=f"BLOCK: {spec['type']} unavailable here",
                       reason=f"color-grading-skill's doctor reports {spec['type']} {getattr(cap, 'detail', '') or 'unavailable'}; {COLOR_REMEDY[spec['type']]}",
                       confidence=1.0, evidence=[r.id for r in req] + [f"capability:color-grading:{spec['type']}"], risk="HIGH", approval="BLOCK", provenance="SYSTEM",
                       params={"skill": spec["skill"], "missing": [f"color-grading:{spec['type']}"]}, requirements=req)
        else:
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
            # ADR-064: an explicit placement must land on the picture the element is drawn on -- the size after this subject's edits
            # (crop / rotate / resize / fit / grid ...), not the source's. What is certainly off the frame is BLOCKED, never moved
            size = planned_picture_size((assets[sources[0]].technical or {}) if sources and sources[0] in assets else {}, eng.decisions, subject)
            off = placement_problems(typ, el["params"], size)
            if off:
                eng.decide(subject=subj_key, type="BLOCK", decision=f"BLOCK: {typ} on {subject} would be drawn outside the picture",
                           reason="; ".join(off) + " (the picture after the edits; an element is never moved or clipped by guessing)",
                           confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK", provenance="USER",
                           params={"asset_id": subject, "type": typ, "picture": list(size) if size else None}, requirements=req)
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
            for k in ("style", "fit"):   # ADR-102: a style preset / text fit for this element
                if el.get(k) is not None:
                    gparams[k] = el[k]
            if typ == "logo" and not mopts["options"].get("themes"):
                # ADR-102: the logo element draws the active theme's logo; without a theme there is no logo to draw (never a file guessed)
                eng.decide(subject=subj_key, type="BLOCK", decision=f"BLOCK: logo on {subject} without a theme", reason="the logo element draws the active theme's logo "
                           "(motion.themes / motion.theme name the theme file and its logo); none is given", confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK",
                           provenance="USER", params={"asset_id": subject, "type": typ}, requirements=req)
                continue
            if typ == "rendered_layer" and html_layer is not None:   # ADR-114: the clip is rendered here by hyperframes-skill, its timing is the segments'
                hreqs = [r for k, r in sorted(m.items()) if (k.startswith("hyperframes.") or k == "motion.layer") and r.provenance != "DEFAULT"]
                decide_html_layer(eng, html_layer, subject, [r.id for r in hreqs] + probe_ids_of(sources), hreqs, caps,
                                  lambda subj, explicit=None, floor=None: approval_for(subj, explicit=explicit, floor=floor, keys=APPROVAL_KEYS))
                gparams["parameters"] = {**gparams["parameters"], "layer": html_layer_id(subject), **element_timing(html_layer)}
                gparams["layer_source"] = "hyperframes-skill"
            label = element_label(typ, el["params"]) if not (typ == "rendered_layer" and html_layer is not None) else "'rendered by hyperframes-skill'"
            eng.decide(subject=subj_key, type="TRANSFORM", decision=f"{typ} {label} on {subject} from {gparams['start']:g}s to {'the end' if end is None else f'{end:g}s'}",
                       reason=f"user asked for a {typ.replace('_', ' ')} ({req[0].source}); rendered on the finished picture" + (f" ({size[0]}x{size[1]})" if size else "")
                       + (" (start / duration from policy defaults)" if settings else ""),
                       confidence=1.0, evidence=ev, risk="MEDIUM", approval=approval_for(subj_key, explicit=req[0], keys=APPROVAL_KEYS), provenance="USER", params=gparams,
                       settings=settings, requirements=req, serves_intent=None)
            finished[subject] = True
    if mots:
        cap_block(GRAPHICS_SKILL, f"capability.{GRAPHICS_SKILL}")
        if mopts["options"]:   # ADR-102: the document-level styling of every render (style / brand kit / theme / reserved regions), one record per subject
            for subject, sources in subjects:
                words = ", ".join(f"{k}={sorted(v) if k == 'themes' else v}" for k, v in mopts["options"].items())
                eng.decide(subject="graphics.options", type="TRANSFORM", decision=f"motion-graphics styling on {subject}: {words}",
                           reason="user asked for these render options (" + ", ".join(r.key for r in mopts["requirements"]) + "); an element's own parameters and style win over them",
                           confidence=1.0, evidence=[r.id for r in mopts["requirements"]] + probe_ids_of(sources), risk="LOW", approval="AUTO", provenance="USER",
                           params={"asset_id": subject, "options": dict(mopts["options"])}, requirements=mopts["requirements"], serves_intent=None)
    elif mopts["options"]:
        raise EditRequirementError(", ".join(r.key for r in mopts["requirements"]) + " style motion-graphics elements, and none is asked for (motion.*)")
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
            opts = dict(sub.get("cues") or {})
            rp = opts.get("regional_profile")
            if rp and lang.split("-")[0].lower() not in [str(x).lower() for x in REGIONAL_PROFILES[rp].get("languages") or []]:
                # ADR-101: a regional profile is for its own languages; it is never applied to another one (and never chosen from the language)
                eng.decide(subject="subtitle.generate", type="BLOCK", decision=f"BLOCK: subtitle.regional_profile={rp} on {lang} subtitles",
                           reason=f"{rp} is for {REGIONAL_PROFILES[rp].get('languages')} documents; the recognised language of {subject} is {lang}",
                           confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": subject, "language": lang}, requirements=req)
                continue
            if opts.get("words"):
                # ADR-101: the shorts caption profile splits cues by their word timings; it needs them on every recognised segment, and a vertical picture
                untimed = [o.id for o in transcripts if any(seg.get("words") is None for seg in (o.data or {}).get("segments") or [])]
                size = planned_picture_size((assets[sources[0]].technical or {}) if sources and sources[0] in assets else {}, eng.decisions, subject)
                why = (f"subtitle.caption_profile={(sub.get('render') or {}).get('caption_profile')} splits cues by their words, and transcript(s) {', '.join(untimed)} have no word "
                       "timestamps (measure with word timestamps)" if untimed else
                       f"subtitle.caption_profile={(sub.get('render') or {}).get('caption_profile')} is for a vertical picture; {subject} is {size[0]}x{size[1]}" if size and not size[1] > size[0]
                       else None)
                if why is not None:
                    eng.decide(subject="subtitle.burn_in", type="BLOCK", decision=f"BLOCK: shorts captions on {subject}", reason=why, confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK",
                               provenance="USER", params={"asset_id": subject}, requirements=req)
                    continue
            eng.decide(subject="subtitle.generate", type="TRANSFORM", decision=f"subtitles ({fmt}) for {subject}: {segs} recognised segment(s) → cues on the delivered timeline",
                       reason=f"user asked for subtitles ({req[0].source}); cues are the recognised text of the transcript(s), re-timed through the trims / concat / speed of the plan; who speaks is unknown",
                       confidence=1.0, evidence=ev, risk="LOW",
                       approval=approval_for("subtitle.generate", explicit=req[0], keys=APPROVAL_KEYS), provenance="USER",
                       params={"asset_id": subject, "format": fmt, "language": lang, "sources": list(sources), "transcript_ids": sorted(o.id for o in transcripts), "segments": segs,
                               **({"constraints": sub["constraints"]} if sub["constraints"] else {}), **opts},
                       settings=[fmt_setting] if fmt_setting else [], requirements=req, serves_intent=None)
            if sub["burn_in"]:
                br = m["subtitle.burn_in"] if "subtitle.burn_in" in m else m["subtitle.mode"]   # ADR-063: subtitle.mode=burn asks for the same
                if not finished.get(subject):
                    eng.decide(subject="subtitle.burn_in", type="BLOCK", decision=f"BLOCK: burn-in on {subject} without an intermediate",
                               reason="subtitle-skill burns into inputs inside the agent workspace only; without a trim / concat / edit / colour / graphics step the input would be the untouched source (never rewritten)",
                               confidence=1.0, evidence=ev + [br.id], risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": subject}, requirements=req + [br])
                else:
                    eng.decide(subject="subtitle.burn_in", type="TRANSFORM", decision=f"burn the subtitles into the picture of {subject}", reason=f"user asked for a burn-in ({br.source}); the sidecar is still produced",
                               confidence=1.0, evidence=ev + [br.id], risk="MEDIUM", approval=approval_for("subtitle.burn_in", explicit=br, keys=APPROVAL_KEYS), provenance="USER",
                               params={"asset_id": subject, "format": "srt", **dict(sub.get("render") or {})}, requirements=req + [br], serves_intent=None)
            if sub["mux"]:   # ADR-063: subtitle-skill render mode "mux" — the cues as a soft track; picture and audio stream-copied
                mr = m["subtitle.mode"]
                targets_req = m.get("delivery.targets")
                presets = sorted({str(t.get("preset")) for t in ((targets_req.value if targets_req else None) or []) if isinstance(t, dict) and t.get("preset")})
                # ADR-071: an engine whose export keeps subtitle streams (release 2.22.4+) carries the track through every preset but gif
                dropping = [p_ for p_ in presets if p_ in EXPORT_PRESETS_DROPPING_SUBTITLES or not export_keeps_subtitles(caps)]
                if dropping:
                    # measured on the real Skills (tests/test_integration.py::SubtitleMuxRealTests): before engine release 2.22.4 the platform export
                    # re-encodes the picture and the audio only and leaves every subtitle stream behind, so the track would never reach the deliverable
                    eng.decide(subject="subtitle.mux", type="BLOCK", decision=f"BLOCK: soft subtitle track on {subject} with an export preset ({', '.join(dropping)})",
                               reason=("the gif export holds no subtitle stream" if all(p_ in EXPORT_PRESETS_DROPPING_SUBTITLES for p_ in dropping) else
                                       "the delivery export of this engine (a release older than 2.22.4, or its version unknown) writes the picture and the audio only and drops "
                                       "subtitle streams") + ": a muxed track would not reach the deliverable (use a profile without a preset, e.g. generic, or a burn-in, "
                                      "or engine release 2.22.4+); handed over to the engine, not worked around here",
                               confidence=1.0, evidence=ev + [mr.id, targets_req.id], risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": subject, "presets": dropping},
                               requirements=req + [mr, targets_req])
                elif not finished.get(subject):
                    eng.decide(subject="subtitle.mux", type="BLOCK", decision=f"BLOCK: soft subtitle track on {subject} without an intermediate",
                               reason="subtitle-skill renders (burns or muxes) inputs inside the agent workspace only; without a trim / concat / edit / colour / graphics step the input would be the untouched source (never rewritten)",
                               confidence=1.0, evidence=ev + [mr.id], risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": subject}, requirements=req + [mr])
                else:
                    eng.decide(subject="subtitle.mux", type="TRANSFORM", decision=f"mux the subtitles into {subject} as a soft (player-toggleable) track",
                               reason=f"user asked for a soft subtitle track ({mr.source}); the picture and the audio are stream-copied, the sidecar is still produced",
                               confidence=1.0, evidence=ev + [mr.id], risk="LOW", approval=approval_for("subtitle.mux", explicit=mr, keys=APPROVAL_KEYS), provenance="USER",
                               params={"asset_id": subject, "format": "srt", "mode": "mux", **{k: v for k, v in (sub.get("render") or {}).items() if k != "caption_profile"}},
                               requirements=req + [mr], serves_intent=None)
        cap_block(GENERATE_SKILL, f"capability.{GENERATE_SKILL}")
        if sub["burn_in"]:
            cap_block(BURN_SKILL, f"capability.{BURN_SKILL}")
        if sub["mux"]:
            cap_block(MUX_SKILL, f"capability.{MUX_SKILL}")
    # ---- thumbnail
    if th["enabled"]:
        req = th["requirements"]
        tsettings: List[Dict[str, Any]] = []
        fmt = th["format"]
        if fmt is None:
            s_ = resolve_setting(rules, "thumbnail.format", "png")
            fmt, tsettings = (str(s_["value"]).lower() if str(s_["value"]).lower() in ("png", "jpeg") else "png"), tsettings + [s_]
        ratio = None
        if th["at"] is None and not th.get("at_faces"):   # ADR-104: thumbnail.at=faces takes the frame from the face ranking instead
            r_ = resolve_setting(rules, "thumbnail.at_ratio", 0.5)
            ratio, tsettings = max(0.0, min(1.0, float(r_["value"]))), tsettings + [r_]
        font_size = th["font_size"]
        position = th["position"]
        if th["text"] is not None or th.get("texts"):   # ADR-088: the extra texts take the caption's size / position when they do not name their own
            if font_size is None:
                f_ = resolve_setting(rules, "thumbnail.font_size", 48)
                font_size, tsettings = int(f_["value"]), tsettings + [f_]
            if position is None:
                p_ = resolve_setting(rules, "thumbnail.position", "bottom")
                position, tsettings = (str(p_["value"]) if str(p_["value"]) in ("center", "top", "bottom") else "bottom"), tsettings + [p_]
        full = bool(th.get("texts") or th.get("logo"))   # ADR-088: more texts and / or a logo → the full ThumbnailDocument
        # ADR-104: variants / layers / batch name their own thumbnail-skill tool (thumbnail_skill_of reads the same keys the IR operation carries)
        skill = thumbnail_skill_of({"text": th["text"], "texts": th.get("texts"), "logo": th.get("logo"), "variants": th.get("variants"), "layers": th.get("layers"),
                                    "batch": th.get("batch")})
        # ADR-104: without detection-skill's faces (or the ranking tool) the frame cannot be chosen: the capability BLOCKs say what is missing
        frame_blocked = bool(th.get("at_faces")) and any([cap_block(sk, f"capability.{sk}") is not None for sk in FRAME_SKILLS])
        for subject, sources in subjects:
            if not video_ok(subject, sources, "thumbnail.render", req) or frame_blocked:
                continue
            params: Dict[str, Any] = {"asset_id": subject, "format": fmt, "at": th["at"], "at_ratio": ratio, "skill": skill}
            fc_ev: List[str] = []
            if th.get("at_faces"):   # ADR-104: the frame from thumbnail-skill's ranking of each source's measured faces
                ranks = {o.asset_id: o for o in analysis.observations if o.kind == "frame_ranking" and o.asset_id in sources}
                attrs = [o.id for o in analysis.observations if o.kind == "face_attributes" and o.asset_id in sources]
                pictured = [a for a in sources if a in assets and (assets[a].technical or {}).get("video")]
                why = None
                if any(a not in ranks for a in pictured):
                    why = f"the faces of {', '.join(a for a in pictured if a not in ranks)} were not measured and ranked (see the analysis rows)"
                elif not any(e.get("rank") is not None and e.get("faces") for o in ranks.values() for e in (o.data or {}).get("ranking") or []):
                    why = "no sampled frame of the picture shows a face: name the frame with thumbnail.at"
                if why is not None:
                    eng.decide(subject="thumbnail.render", type="BLOCK", decision=f"BLOCK: no frame of {subject} to rank for the thumbnail", reason=why, confidence=1.0,
                               evidence=[r.id for r in req] + attrs + [o.id for o in ranks.values()], risk="MEDIUM", approval="BLOCK", provenance="OBSERVED", requirements=req)
                    continue
                params["frame_choice"] = {"observations": [ranks[a].id for a in pictured if a in ranks], "every": th["faces"]["every"]}
                fc_ev = attrs + [ranks[a].id for a in pictured if a in ranks]
                if th.get("avoid_faces") is not None:
                    bent = sorted({d.subject for d in eng.decisions if d.type == "TRANSFORM" and d.status != "REJECTED" and d.subject in GEOMETRY_SUBJECTS
                                   and d.params.get("asset_id") in [subject] + list(sources)})
                    size = planned_picture_size((assets[pictured[0]].technical if pictured else {}) or {}, eng.decisions, subject)
                    off = [a for a in pictured if size and not same_aspect((ranks[a].data or {}).get("frame_size") or {}, size)]
                    if bent or off or not size:
                        eng.decide(subject="thumbnail.render", type="BLOCK", decision=f"BLOCK: the caption of {subject} cannot be placed clear of measured faces",
                                   reason=("the faces were measured on the source frame and " + (f"{', '.join(bent)} changes the picture's geometry" if bent else
                                           f"the picture's aspect ratio is not that of {', '.join(off)}" if off else "the picture size is unknown")
                                           + ": the boxes would not be where the picture shows them (set thumbnail.position instead of thumbnail.avoid_faces)"),
                                   confidence=1.0, evidence=[r.id for r in req] + fc_ev, risk="MEDIUM", approval="BLOCK", provenance="OBSERVED", requirements=req)
                        continue
                    params["avoid_faces"] = dict(th["avoid_faces"])
            for k in ("variants", "layers", "batch"):   # ADR-104
                if th.get(k):
                    params[k] = json.loads(json.dumps(th[k]))
            if th["text"] is not None:
                params.update({"text": th["text"], "font_size": font_size, "position": position, "font_id": "sans-bold", "color": th.get("color") or "#FFFFFF"})
            elif full:
                params.update({"font_size": font_size, "position": position})
            if th.get("texts"):
                params["texts"] = [dict(t) for t in th["texts"]]
            if th.get("logo"):
                params["logo"] = dict(th["logo"])
            where = (f"at {th['at']:g}s" if th["at"] is not None else "at the best-ranked face frame (thumbnail-skill choose_frame) that the edit keeps" if th.get("at_faces")
                     else f"at {ratio:g} of the delivered timeline")
            extras = ([f"{1 + len(th['texts'])} text element(s)"] if th.get("texts") else []) + ([f"logo {Path(th['logo']['image']).name}"] if th.get("logo") else []) \
                + (["caption placed clear of the faces (place_text)"] if params.get("avoid_faces") is not None else []) \
                + ([f"{th['variants']['count']} A/B variants"] if th.get("variants") else []) + (["layered PNGs + manifest"] if th.get("layers") else []) \
                + ([f"batch of {len(th['batch']['files'])} file(s): {th['batch']['rows']} row(s) of {Path(th['batch']['csv']).name} x {', '.join(th['batch']['presets'])}"]
                   if th.get("batch") else [])
            eng.decide(subject="thumbnail.render", type="TRANSFORM", decision=f"thumbnail ({fmt}) of {subject} {where}" + (" with a caption" if th["text"] is not None else "")
                       + (f" ({', '.join(extras)})" if extras else ""),
                       reason=f"user asked for a thumbnail ({req[0].source}); the frame is taken from the finished picture" + (" (position from policy defaults)" if ratio is not None else ""),
                       confidence=1.0, evidence=[r.id for r in req] + probe_ids_of(sources) + fc_ev, risk="LOW",
                       approval=approval_for("thumbnail.render", explicit=req[0], keys=APPROVAL_KEYS) if not fc_ev else "CONFIRM",   # a ranking suggests; the user decides
                       provenance="USER", params=params, settings=tsettings, requirements=req, serves_intent=None)
        cap_block(skill, f"capability.{skill}")
    # ---- QC gate (a measurement: AUTO; the WARN promotion policy is recorded on the decision)
    if qc is not None:
        wp = resolve_setting(rules, WARN_PROMOTION_KEY, WARN_PROMOTION_DEFAULT)
        ref_reqs = qc_ref["requirements"] if qc_ref else []
        ref_text = f"; {', '.join(m_.upper() for m_ in qc_ref['metrics'])} of each video deliverable against {Path(qc_ref['path']).name}" if qc_ref else ""
        for subject, sources in subjects:
            params: Dict[str, Any] = {"asset_id": subject, "warn_promotion": str(wp["value"]).upper()}
            if qc_ref:   # ADR-058: full-reference metrics, opt-in; the rule is what qc-skill's VideoRule takes
                params["reference"] = {"path": qc_ref["path"], "rule": dict(qc_ref["rule"])}
            eng.decide(subject="qc.check", type="DELIVER", decision=f"QC gate on the deliverables of {subject}: PASS → ready, WARN → {str(wp['value']).upper()}, FAIL → never delivered{ref_text}",
                       reason=f"user asked for the QC gate ({qc.source}); qc-skill measures each deliverable (and the subtitle sidecar), and the deliverable together with its sidecar / thumbnail as one set (ADR-065); a report is admitted only for the files the agent hashed itself",
                       confidence=1.0, evidence=[qc.id] + [r.id for r in ref_reqs] + probe_ids_of(sources), risk="LOW", approval=approval_for("qc.check", explicit=qc, keys=APPROVAL_KEYS), provenance="USER",
                       params=params, settings=[wp], requirements=[qc] + ref_reqs, serves_intent=None)
        cap_block(QC_SKILL, f"capability.{QC_SKILL}")
