"""Slide changes of a talk recording (ADR-091; slide-sync-skill roadmap step 6, agent side): the `slides.*` requirements that ask slide-sync-skill
for the slide timeline and, with the speaker's deck, which page is on screen in each slide.

    Requirement (`--set slides.detect=true`, + `slides.deck=<pdf|pptx|key|odp>`) → analysis kind `slide_timeline` (slide-sync detect [→ import →
    match], a measurement) → Observation `slide_timeline` (OBSERVED) + SlideEvents on the asset's timeline

- **Requirements** (a refinement without the switch, an unknown `slides.*` key or a value out of range is refused before analysis):
  `slides.detect=true`; `slides.threshold` (detect's scdet cut score, (0, 100], the Skill's default 8 when absent) and `slides.min_dwell`
  (seconds a slide must stay, > 0, default 2); `slides.deck=<file>` (the speaker's deck: PDF, or pptx / ppt / key / odp through LibreOffice;
  `--set` only, never from a brief: only the user names a file the agent reads).
- **What it is not**: the timeline is measured picture cuts, not an understanding of the slides; a page is reported only where slide-sync
  matched it (confidence and margin), otherwise null with the Skill's reason -- never filled in from the neighbours or the deck order.
  Nothing here decides an edit (chapters / inserts are separate, explicit steps not wired yet).
- **Without slide-sync-skill** (or its engine not ready) the measurement is not run: the plan records the ask and BLOCKs with a handoff
  naming slide-sync-skill and its install. A measurement that ran and failed blocks with slide-sync's own code (DEPENDENCY_MISSING with
  the install line when a deck cannot be imported: pypdfium2, or LibreOffice for pptx / key / odp)."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models import Requirement
from ..tools.slide_sync.adapter import DECK_EXTENSIONS, KIND
from .editing import EditRequirementError

PREFIX = "slides."
SWITCH = "slides.detect"
DECK_KEY = "slides.deck"
SKILL = "slide_sync"
REQUIREMENT_KEYS = (SWITCH, "slides.threshold", "slides.min_dwell", DECK_KEY)
_ON = (True, "true", "yes", "on", "1")
_OFF = (False, None, "false", "no", "off", "0", "none")


def _number(v: Any, key: str, lo: float, hi: float, lo_open: bool = True) -> float:
    if isinstance(v, str):
        try:
            v = float(v.strip())
        except ValueError:
            raise EditRequirementError(f"{key} must be a number, got {v!r}") from None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) > hi or (float(v) <= lo if lo_open else float(v) < lo):
        raise EditRequirementError(f"{key} must be a number in ({lo:g}, {hi:g}] (slide-sync's own range), got {v!r}")
    return float(v)


def parse_slide_requirements(user_requirements: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """(requested, the measurement parameters) from explicit requirements; EditRequirementError on anything slide-sync would refuse."""
    given = {k: v for k, v in (user_requirements or {}).items() if k.startswith(PREFIX)}
    unknown = sorted(k for k in given if k not in REQUIREMENT_KEYS)
    if unknown:
        raise EditRequirementError(f"unknown requirement key(s): {', '.join(unknown)}; known: {', '.join(REQUIREMENT_KEYS)}")
    if not given:
        return False, {}
    sw = given.get(SWITCH)
    sw = sw.strip().lower() if isinstance(sw, str) else sw
    if SWITCH in given and sw not in _ON + _OFF:
        raise EditRequirementError(f"{SWITCH} is a switch (true / false), got {given[SWITCH]!r}")
    on = SWITCH in given and sw in _ON
    refinements = sorted(k for k in given if k != SWITCH and given[k] is not None)
    if refinements and not on:
        raise EditRequirementError(f"{', '.join(refinements)} refine {SWITCH}, which is not set to true")
    if not on:
        return False, {}
    params: Dict[str, Any] = {}
    if given.get("slides.threshold") is not None:
        params["threshold"] = _number(given["slides.threshold"], "slides.threshold", 0.0, 100.0)
    if given.get("slides.min_dwell") is not None:
        params["min_dwell"] = _number(given["slides.min_dwell"], "slides.min_dwell", 0.0, 3600.0)
    if given.get(DECK_KEY) is not None:
        deck = given[DECK_KEY]
        if not isinstance(deck, str) or not deck.strip() or Path(deck).suffix.lower() not in DECK_EXTENSIONS:
            raise EditRequirementError(f"{DECK_KEY} must be a {' / '.join(DECK_EXTENSIONS)} file, got {deck!r}")
        if not os.path.isfile(deck):
            raise EditRequirementError(f"{DECK_KEY}: no such file: {os.path.basename(deck)}")
        params["deck"] = os.path.abspath(deck)
    return True, params


def check_slide_keys(user_requirements: Dict[str, Any]) -> None:
    parse_slide_requirements(user_requirements)


def slide_kind(user_requirements: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """(analysis kind, its parameters) when the request asks for the slide timeline, else None."""
    on, params = parse_slide_requirements(user_requirements)
    return (KIND, params) if on else None


def decide_slides(eng: Any, m: Dict[str, Requirement], analysis: Any, probe_ids_of: Callable[[List[str]], List[str]], cap_missing: Callable[[str], Optional[str]]) -> None:
    """A slide timeline that cannot be measured here → the ask + a BLOCK naming why (handed off); one that ran and failed on a video input →
    a BLOCK with slide-sync's own code. A measurement that ran is the observation itself: nothing to decide."""
    r = m.get(SWITCH)
    if r is None or r.provenance != "USER":
        return
    values = {k: req.value for k, req in m.items() if k.startswith(PREFIX)}
    on, _params = parse_slide_requirements(values)
    if not on:
        return
    reqs = [m[k] for k in sorted(values) if k in m]
    video = [a.id for a in analysis.assets if (a.technical or {}).get("video")]
    why = cap_missing(SKILL)
    if why:
        eng.decide(subject=SWITCH, type="TRANSFORM", decision=f"{KIND}: the slide changes of every video input" + (" and the deck page on screen in each" if DECK_KEY in values else ""),
                   reason=f"user asked for {SWITCH} ({r.source}); no tool for {SKILL} here, so the measurement is handed off to slide-sync-skill",
                   confidence=1.0, evidence=[x.id for x in reqs] + probe_ids_of(video), risk="LOW", approval="CONFIRM", provenance="USER",
                   params={"skill": SKILL, "handoff": True}, requirements=reqs)
        eng.decide(subject=SWITCH, type="BLOCK", decision=f"BLOCK: {KIND} is handed off ({SKILL} unavailable)",
                   reason=f"{why}; install slide-sync-skill and plan again, or plan without {SWITCH} (see the plan's handoff)",
                   confidence=1.0, evidence=[f"skill:{SKILL}"], risk="HIGH", approval="BLOCK", provenance="SYSTEM", params={"skill": SKILL, "missing": ["slide-sync"]})
        return
    rows = [row for a in getattr(analysis, "analyses", []) or [] for row in a.get("rows") or [] if row.get("kind") == KIND]
    for row in rows:
        if row.get("status") != "FAILED":
            continue
        err = row.get("error") or {}
        code = err.get("skill_error") or err.get("kind")
        eng.decide(subject=SWITCH, type="BLOCK", decision=f"BLOCK: {KIND} failed on {row.get('asset_id')} ({code})",
                   reason=f"user asked for {SWITCH}; the measurement failed: {code}: {str(err.get('message') or '')[:300]} (nothing is guessed in its place)",
                   confidence=1.0, evidence=[x.id for x in reqs] + probe_ids_of([row.get("asset_id")]), risk="HIGH", approval="BLOCK", provenance="SYSTEM",
                   params={"skill": SKILL, "asset_id": row.get("asset_id"), "code": code}, requirements=reqs)
