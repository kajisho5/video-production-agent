"""Requirements → Intent (rule based in Phase 1)."""
from __future__ import annotations

from typing import Dict, List

from ..models import Intent, Requirement
from .requirements import requirement_map


def resolve_intent(reqs: List[Requirement]) -> Intent:
    m = requirement_map(reqs)
    targets = m.get("delivery.targets")
    has_export = bool(targets and any(t.get("preset") for t in targets.value))
    wants_edit = any(m.get(k) and m[k].value is True for k in ("edit.trim_leading_silence", "edit.trim_trailing_silence", "audio.normalize"))
    secondary: List[str] = []
    if m.get("audio.normalize") and m["audio.normalize"].value in (True, "auto"):
        secondary.append("normalize_audio")
    if m.get("edit.trim_leading_silence") and m["edit.trim_leading_silence"].value in (True, "auto"):
        secondary.append("cleanup_silence")
    if m.get("audio.automix") and m["audio.automix"].value not in (False, "false", "no", "off", "0"):   # ADR-134: the clip's microphones mixed
        secondary.append("automix_microphones")
    if m.get("audio.language_tracks") and m["audio.language_tracks"].value in (True, "true", "yes", "on", "1"):
        secondary.append("language_tracks")
    if has_export:
        primary, reason = "clean_and_deliver", "profile defines delivery targets with presets"
    elif wants_edit:
        primary, reason = "clean_only", "edits requested, no delivery preset"
    else:
        primary, reason = "inspect_and_clean", "no explicit edit or delivery; only clearly technical clean-up will be proposed"
    prov = "USER" if any(r.provenance == "USER" for r in reqs if r.key.startswith(("edit.", "audio.", "delivery."))) else "SYSTEM"
    return Intent(primary=primary, secondary=secondary, confidence=1.0 if prov == "USER" else 0.7, provenance=prov, reason=reason)
