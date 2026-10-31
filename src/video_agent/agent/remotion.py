"""Measured motion graphics (ADR-093; the Remotion Skill's roadmap step 4, agent side: SkillSpec measured_motion -> pipeline, CONFIRM): one Remotion
composition over the subject's picture, drawn by the Remotion Skill's pipeline from the media Skill's measurements, after the edits and before the
finishing.

    Requirement (`--set remotion.comp=<composition> remotion.project=<the user's Remotion project>`) → TRANSFORM `video.measured_motion` (approval floor
    CONFIRM) → IR `video.measured_motion {asset, input, output, comp, project, captions?, options?, font_file?, beats?, conform?, fast?, aspect?, fit?, width?}`
    → step `measured_motion` (the Skill's pipeline tool: analyze → render, verified; no delivery: the agent delivers)

- **Requirements** (`--set` only; a refinement without the composition, an unknown key or a value outside the contract is refused before analysis):
  `remotion.comp` -- one of the compositions that keep the source's clock (Captioned, PeakPulse, Titled, BeatMontage, Karaoke, Pip); Jumpcut and
  Highlights cut the picture to a length known only after the render and are refused (the agent's own silence cut is the jump cut it plans);
  `remotion.project` -- a Remotion project made by the Skill's init.py (a folder with package.json; the agent never runs `npm install`);
  `remotion.captions` / `.options` / `.font_file` (files the user names), `.beats` / `.conform` / `.fast` (switches). The contract's `aspect` /
  `width` / `fit` are not taken: they change the frame the agent's later steps plan on (reshape with edit.fit / edit.fill / edit.resize).
- **Decision**: on the programme or each video input, TRANSFORM `video.measured_motion` (risk MEDIUM, approval `video.measured_motion.approval` with
  floor CONFIRM: a browser render of new graphics over the picture, under Remotion's own licence -- free for individuals and companies of up to 3
  people). BLOCK: the project's Remotion is not installed (`node_modules/remotion`, the file its doctor reads: run `npm install` there), the subject
  has no video. Without the
  Skill (or Node / the media Skill missing, as its doctor says) the step is handed off naming the Skill (PACKAGE_ID)."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models import Requirement
from ..tools.remotion.adapter import COMPOSITIONS, CUTTING_COMPOSITIONS, PACKAGE_ID
from .editing import EditRequirementError

PREFIX = "remotion."
SWITCH = "remotion.comp"
SUBJECT = "video.measured_motion"
SKILL = "measured_motion"
FILE_KEYS = ("captions", "options", "font_file")
BOOL_KEYS = ("beats", "conform", "fast")
REQUIREMENT_KEYS = (SWITCH, "remotion.project") + tuple(PREFIX + k for k in FILE_KEYS + BOOL_KEYS)
APPROVAL_KEY = ("video.measured_motion.approval", "CONFIRM")
PARAMS = ("comp", "project") + FILE_KEYS + BOOL_KEYS
_ON = (True, "true", "yes", "on", "1")
_OFF = (False, "false", "no", "off", "0")


def parse_remotion_requirements(values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The composition asked for ({comp, project, ...}) or None; EditRequirementError on anything refused."""
    given = {k: v for k, v in (values or {}).items() if k.startswith(PREFIX) and v is not None}
    unknown = sorted(k for k in given if k not in REQUIREMENT_KEYS)
    frame = [k for k in unknown if k in ("remotion.aspect", "remotion.width", "remotion.fit")]
    if frame:
        raise EditRequirementError(f"{', '.join(frame)}: the composition is drawn in the subject's own frame (the agent's later steps plan on it); "
                                   "reshape with edit.fit / edit.fill / edit.resize or a delivery preset")
    if unknown:
        raise EditRequirementError(f"unknown requirement key(s): {', '.join(unknown)}; known: {', '.join(REQUIREMENT_KEYS)}")
    if not given:
        return None
    comp = given.get(SWITCH)
    if comp is None:
        raise EditRequirementError(f"{', '.join(sorted(given))} refine {SWITCH}, which is not set")
    if comp in CUTTING_COMPOSITIONS:
        raise EditRequirementError(f"{SWITCH}={comp} cuts the picture to a length known only after the render; the agent plans the jump cut itself "
                                   f"(silence removal) -- use one of {', '.join(COMPOSITIONS)}")
    if comp not in COMPOSITIONS:
        raise EditRequirementError(f"{SWITCH} must be one of {', '.join(COMPOSITIONS)}, got {comp!r}")
    out: Dict[str, Any] = {"comp": comp}
    proj = given.get("remotion.project")
    if not isinstance(proj, str) or not os.path.isfile(os.path.join(proj, "package.json")):
        raise EditRequirementError(f"remotion.project must name a Remotion project made by {PACKAGE_ID}'s init.py (a folder with package.json); "
                                   "the agent never creates one or runs npm install")
    out["project"] = os.path.abspath(proj)
    for k in FILE_KEYS:
        if PREFIX + k in given:
            if not isinstance(given[PREFIX + k], str) or not os.path.isfile(given[PREFIX + k]):
                raise EditRequirementError(f"{PREFIX + k} must be an existing file, got {given[PREFIX + k]!r}")
            out[k] = os.path.abspath(given[PREFIX + k])
    for k in BOOL_KEYS:
        if PREFIX + k in given:
            v = given[PREFIX + k]
            v = v.strip().lower() if isinstance(v, str) else v
            if v not in _ON + _OFF:
                raise EditRequirementError(f"{PREFIX + k} is a switch (true / false), got {given[PREFIX + k]!r}")
            if v in _ON:
                out[k] = True
    return out


def check_remotion_keys(user_requirements: Dict[str, Any]) -> None:
    parse_remotion_requirements(user_requirements)


def decide_remotion(eng: Any, m: Dict[str, Requirement], subjects: List[Tuple[str, List[str]]], analysis: Any, approval_for: Callable,
                    probe_ids_of: Callable[[List[str]], List[str]], cap_block: Callable[[str, str], Any]) -> None:
    reqs = [r for k, r in sorted(m.items()) if k.startswith(PREFIX) and r.provenance == "USER"]
    if not reqs:
        return
    p = parse_remotion_requirements({r.key: r.value for r in reqs})
    if p is None:
        return
    installed = os.path.isfile(os.path.join(p["project"], "node_modules", "remotion", "package.json"))   # the file the Skill's doctor reads
    words = ", ".join(f"{k} {os.path.basename(str(v)) if k in FILE_KEYS + ('project',) else v}" for k, v in p.items() if k != "comp")
    for subject, sources in subjects:
        src = [a for a in analysis.assets if a.id in sources]
        ev = [r.id for r in reqs] + probe_ids_of(sources)
        if not src or not all((a.technical or {}).get("video") for a in src):
            eng.decide(subject=SUBJECT, type="BLOCK", decision=f"BLOCK: {SUBJECT} on {subject} (no video stream)", reason=f"{subject} has no video stream to draw on",
                       confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": subject}, requirements=reqs)
            continue
        if not installed:
            eng.decide(subject=SUBJECT, type="BLOCK", decision=f"BLOCK: the Remotion project {os.path.basename(p['project'])} is not installed",
                       reason=f"node_modules/remotion is missing in {os.path.basename(p['project'])}: run `cd {p['project']} && npm install` (Remotion's browser downloads on "
                              "its first render), then plan again; the agent never installs it", confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK",
                       provenance="USER", params={"asset_id": subject}, requirements=reqs)
            continue
        eng.decide(subject=SUBJECT, type="TRANSFORM", decision=f"{SUBJECT} on {subject}: Remotion {p['comp']} ({words})",
                   reason=f"user asked for the {p['comp']} composition ({reqs[0].source}); {PACKAGE_ID} measures the picture and draws it, "
                          "verifying the render against its timeline; it runs after the edits and before the finishing. Remotion's licence: free for individuals and "
                          "companies of up to 3 people, a company licence beyond",
                   confidence=1.0, evidence=ev, risk="MEDIUM", approval=approval_for(SUBJECT, floor="CONFIRM"), provenance="USER",
                   params={"asset_id": subject, **p}, requirements=reqs, serves_intent=None)
    cap_block(SKILL, f"capability.{SKILL}")


IR_KEYS = ("type", "asset", "input", "output", "decision_ids", "temporal_scope") + PARAMS


def remotion_problems(op: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    extra = sorted(k for k in op if k not in IR_KEYS)
    if extra:
        errs.append(f"{SUBJECT}: parameters {extra} are not in the operation's vocabulary")
    if op.get("comp") not in COMPOSITIONS:
        errs.append(f"{SUBJECT}: comp {op.get('comp')!r} is not one of {', '.join(COMPOSITIONS)}")
    if not op.get("project") or not op.get("input") or op.get("output") != f"{op.get('asset')}_measured_motion":
        errs.append(f"{SUBJECT}: needs project, input and output {op.get('asset')}_measured_motion")
    return errs
