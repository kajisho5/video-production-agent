"""Handing the cut to an NLE through nle-bridge-skill (ADR-086; nle-bridge-skill roadmap step 7, agent side): the agent's project IR, as it is
executed, goes to `nle-bridge export` (OTIO / FCP7 XML for Premiere / CMX 3600 EDL / SRT) or `nle-bridge conform` (those plus a relink manifest
and a zip). nle-bridge reads the IR's `delivery.timelines[]` record of the subject (ADR-052), so this needs `delivery.timeline` planned.

- **Requirements** (`--set`, USER or PROFILE): `delivery.nle` = export | conform (none / false: nothing), `delivery.nle.formats` (a comma list of
  the contract's formats, default its `default_formats`), `delivery.nle.zip` (conform only, default true).
- **Decision** `timeline.nle` (DELIVER, AUTO by default like the timeline export: files next to the deliverables, no media written, the
  sources referenced where they are; policy key `timeline.nle.approval`). BLOCK without a planned timeline of the subject. Without
  nle-bridge-skill the step has no tool and the plan's handoff names the package (ADR-051).
- **What is not done here**: reading the editor's changes back (`nle-bridge diff --timeline-out` as a new plan version) and live NLE
  control -- follow-ups."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models import Requirement
from ..tools.nle_bridge.adapter import TOOL_CONFORM, TOOL_EXPORT, default_formats, formats as contract_formats
from .editing import EditRequirementError, _bool
from .timeline import TIMELINE_SUBJECT

KEY = "delivery.nle"
REQUIREMENT_KEYS = (KEY, "delivery.nle.formats", "delivery.nle.zip")
SUBJECT = "timeline.nle"
MODES = {"export": ("nle_export", TOOL_EXPORT), "conform": ("nle_conform", TOOL_CONFORM)}
SKILLS = tuple(s for s, _t in MODES.values())
APPROVAL_KEYS = {SUBJECT: (f"{SUBJECT}.approval", "AUTO")}
_OFF = ("none", "false", "no", "off", "0")


def parse_nle_requirements(m: Dict[str, Requirement]) -> Optional[Dict[str, Any]]:
    """→ {"mode", "formats", "zip", "requirements"} or None. Refinements without `delivery.nle`, an unknown mode or format, and `zip` on an
    export are refused (never guessed)."""
    r = m.get(KEY)
    refs = [m[k] for k in REQUIREMENT_KEYS[1:] if k in m and m[k].provenance != "DEFAULT"]
    v = None if r is None or r.provenance == "DEFAULT" else r.value
    if v is None or v is False or (isinstance(v, str) and v.strip().lower() in _OFF):
        if refs:
            raise EditRequirementError(f"{refs[0].key} is set but {KEY} is not: refusing an ambiguous request")
        return None
    mode = str(v).strip().lower()
    if mode not in MODES:
        raise EditRequirementError(f"{KEY} must be one of none, {', '.join(MODES)}, got {v!r}")
    fmts: List[str] = list(default_formats())
    if "delivery.nle.formats" in m and m["delivery.nle.formats"].provenance != "DEFAULT":
        raw = m["delivery.nle.formats"].value
        fmts = [f.strip().lower() for f in (raw if isinstance(raw, list) else str(raw).split(",")) if str(f).strip()]
        bad = [f for f in fmts if f not in contract_formats()]
        if not fmts or bad or len(set(fmts)) != len(fmts):
            raise EditRequirementError(f"delivery.nle.formats must be distinct names from {', '.join(contract_formats())}, got {raw!r}")
        if fmts == ["srt"]:
            raise EditRequirementError("delivery.nle.formats needs at least one timeline format (otio, fcp7, edl); srt travels next to them")
    zip_ = True if mode == "conform" else False
    if "delivery.nle.zip" in m and m["delivery.nle.zip"].provenance != "DEFAULT":
        if mode != "conform":
            raise EditRequirementError("delivery.nle.zip is a conform package option: it needs delivery.nle=conform")
        zip_ = _bool(m["delivery.nle.zip"].value, "delivery.nle.zip")
    return {"mode": mode, "formats": fmts, "zip": zip_, "requirements": [r] + refs}


def check_nle_keys(user_requirements: Dict[str, Any]) -> None:
    if not any(k == KEY or k.startswith(KEY + ".") for k in user_requirements or {}):
        return
    unknown = sorted(k for k in user_requirements if (k == KEY or k.startswith(KEY + ".")) and k not in REQUIREMENT_KEYS)
    if unknown:
        raise EditRequirementError(f"unknown requirement key(s): {', '.join(unknown)}; delivery.nle keys are {', '.join(REQUIREMENT_KEYS)}")
    parse_nle_requirements({k: Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in user_requirements.items() if k in REQUIREMENT_KEYS})


def ir_nle(subject: str, mode: str, fmts: List[str], zip_: bool, decision_ids: List[str], output: str) -> Dict[str, Any]:
    return {"type": f"nle.{mode}", "asset": subject, "formats": list(fmts), **({"zip": bool(zip_)} if mode == "conform" else {}), "output": output,
            "decision_ids": list(decision_ids)}


def decide_nle(eng: Any, m: Dict[str, Requirement], decs: List[Any], subjects: List[Tuple[str, List[str]]], approval_for: Callable[..., Dict[str, Any]],
               probe_ids_of: Callable[[List[str]], List[str]], cap_block: Callable[[str, str], Any]) -> None:
    p = parse_nle_requirements(m)
    if p is None:
        return
    req = p["requirements"]
    skill = MODES[p["mode"]][0]
    live = [d for d in decs if d.status != "REJECTED"]
    for subject, sources in subjects:
        ev = [r.id for r in req] + probe_ids_of(sources)
        tl = next((d for d in live if d.subject == TIMELINE_SUBJECT and d.type == "DELIVER" and d.params.get("asset_id") == subject), None)
        if tl is None:
            eng.decide(subject=SUBJECT, type="BLOCK", decision=f"BLOCK: nle-bridge {p['mode']} of {subject}",
                       reason="nle-bridge reads the project IR's timeline of the subject (delivery.timelines, ADR-052), and none is planned: add --set "
                              "delivery.timeline=otio (or fcpxml / edl), or the timeline itself is blocked for this programme", confidence=1.0, evidence=ev, risk="HIGH",
                       approval="BLOCK", provenance="USER", params={"asset_id": subject}, requirements=req)
            continue
        eng.decide(subject=SUBJECT, type="DELIVER",
                   decision=f"hand the cut of {subject} to an NLE with nle-bridge {p['mode']} ({', '.join(p['formats'])}" + (", zipped" if p["zip"] else "") + ")",
                   reason=f"user asked for NLE files ({req[0].source}); nle-bridge-skill reads this plan's project IR (the subject's timeline record, its assets "
                          "and caption cues) and writes the interchange files next to the deliverables; no media is written and the sources are referenced in "
                          "place; every item a format cannot carry is a named warning" + ("; the conform package adds a relink manifest (sha256, start timecode, "
                                                                                          "ranges used)" if p["mode"] == "conform" else ""),
                   confidence=1.0, evidence=ev, risk="LOW", approval=approval_for(SUBJECT, explicit=req[0], keys=APPROVAL_KEYS), provenance=req[0].provenance,
                   params={"asset_id": subject, "mode": p["mode"], "formats": list(p["formats"]), "zip": p["zip"]}, requirements=req, serves_intent=None)
    cap_block(skill, f"capability.{skill}")
