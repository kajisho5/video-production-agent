"""Request → Requirements. Phase 1 has no LLM: requirements come from structured CLI arguments (USER), the
profile (PROFILE), and defaults (DEFAULT). A small keyword pass over free text produces USER requirements
only for unambiguous phrases; anything else is left for the user to state explicitly. A second, narrower
pass (NUMERIC_KEYWORDS) extracts an explicit numeric target from an otherwise-unambiguous phrase (e.g. "-16
LUFS") -- without it, "normalize loudness to -16 LUFS" only set the boolean intent (audio.normalize=True)
and silently fell back to the profile's own default target, which can make an explicitly-requested
normalization look like "nothing to do" if the source happens to already be within tolerance of that
unrelated default (found via a real `video-agent plan --request` run, not assumed)."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from ..models import Request, Requirement
from ..policy.rules import RuleSet
from ..profiles.loader import Profile

KEYWORDS = [
    (re.compile(r"(remove|trim|cut)\s+(the\s+)?(leading|opening|initial)\s+silence|先頭の無音|冒頭の無音", re.I), "edit.trim_leading_silence", True),
    (re.compile(r"(remove|trim|cut)\s+(the\s+)?(trailing|ending)\s+silence|末尾の無音", re.I), "edit.trim_trailing_silence", True),
    (re.compile(r"normali[sz]e|loudness|LUFS|音量.*(揃|正規化)", re.I), "audio.normalize", True),
    (re.compile(r"\byoutube\b", re.I), "delivery.platform", "youtube"),
    (re.compile(r"frame.?accurate|フレーム単位", re.I), "edit.precision", "frame"),
]

# Same "only unambiguous phrases" discipline as KEYWORDS above, but the captured group becomes the
# requirement's value (cast by the third element) instead of a fixed True/string.
NUMERIC_KEYWORDS = [
    (re.compile(r"(-?\d+(?:\.\d+)?)\s*LUFS", re.I), "audio.loudness.target_lufs", float),
    (re.compile(r"(-?\d+(?:\.\d+)?)\s*dB\s*TP", re.I), "audio.loudness.true_peak", float),
]


def extract_requirements(request: Request, profile: Profile, rules: RuleSet) -> List[Requirement]:
    reqs: List[Requirement] = []
    args = request.args or {}
    # explicit CLI arguments are USER requirements
    for key, value in (args.get("requirements") or {}).items():
        reqs.append(Requirement(key=key, value=value, provenance="USER", source="cli"))
    # keyword pass on the raw text (only unambiguous phrases)
    seen = {r.key for r in reqs}
    for rx, key, value in KEYWORDS:
        if request.raw and rx.search(request.raw) and key not in seen:
            reqs.append(Requirement(key=key, value=value, provenance="USER", source="request-text"))
            seen.add(key)
    # numeric target extraction (e.g. "-16 LUFS"): takes priority over the profile/rules default
    # for the same key, since seen already includes it once matched here
    for rx, key, cast in NUMERIC_KEYWORDS:
        if request.raw and key not in seen:
            m = rx.search(request.raw)
            if m:
                reqs.append(Requirement(key=key, value=cast(m.group(1)), provenance="USER", source="request-text"))
                seen.add(key)
    # profile-level requirements
    reqs.append(Requirement(key="delivery.targets", value=profile.delivery_targets, provenance="PROFILE", source=f"profile:{profile.name}"))
    for key in ("audio.loudness.target_lufs", "audio.loudness.true_peak", "silence.leading.min_seconds", "silence.threshold_db"):
        if key not in seen and key in rules.effective:
            reqs.append(Requirement(key=key, value=rules.get(key), provenance=rules.provenance(key) or "PROFILE", source=rules.effective[key].source))
            seen.add(key)
    # defaults
    defaults: Dict[str, Any] = {"edit.trim_leading_silence": "auto", "edit.trim_trailing_silence": "auto", "audio.normalize": "auto", "edit.precision": "keyframe-ok",
                                "delivery.preserve_source": True}
    for k, v in defaults.items():
        if k not in seen:
            reqs.append(Requirement(key=k, value=v, provenance="DEFAULT", source="defaults"))
    return reqs


def requirement_map(reqs: List[Requirement]) -> Dict[str, Requirement]:
    out: Dict[str, Requirement] = {}
    order = {"USER": 3, "PROFILE": 2, "SYSTEM": 2, "DEFAULT": 1, "OBSERVED": 0, "INFERRED": 0, "AI_GENERATED": 0}
    for r in reqs:
        cur = out.get(r.key)
        if cur is None or order.get(r.provenance, 0) >= order.get(cur.provenance, 0):
            out[r.key] = r
    return out
