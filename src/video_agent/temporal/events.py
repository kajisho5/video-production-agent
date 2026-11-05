"""Event type system and the deterministic Observation → Event transformation (ADR-020).

Event ≠ Observation ≠ Inference ≠ Decision. Events exist on the time axis; they are built from validated observations
(OBSERVED), by deterministic transformation (DERIVED), from user actions (USER), or recorded as interpretations
(INFERRED / AI_GENERATED). An AI provider can never produce an OBSERVED event.

Defining an event type here does not mean it is detected: only the codes in IMPLEMENTED_CODES are ever generated, from
the observation kinds that are implemented (silence, loudness, transcript, speaker_turns since ADR-086, slide_timeline since ADR-091). Camera / scene /
caption types are schema only until an analyzer or skill exists for them. A SpeechEvent says "speech with this transcript exists in this
interval"; it never says who speaks (speaker_id stays null unless the transcript records the speaker merge an explicit requirement asked for,
ADR-096: then an anonymous SPEAKER_nn) and never becomes a command.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..models import EVENT_PROVENANCE, Asset, Event, Observation, TimeRange, stable_hash

TRANSFORM_VERSION = "1.0"   # observation → event transformation; part of derived-event identity

# domain type → allowed subtypes
EVENT_TYPES: Dict[str, Tuple[str, ...]] = {
    "AudioEvent": ("silence", "active", "loudness", "dropout", "clipping"),
    "SpeechEvent": ("speech",),
    "SpeakerEvent": ("speaker",),
    "SceneEvent": ("visual_change",),
    "SlideEvent": ("slide",),
    "CameraEvent": ("camera",),
    "IncidentEvent": ("black_frame", "freeze", "audio_dropout", "corrupted_frame"),
    "CaptionEvent": ("caption",),
    "UserDecisionEvent": ("approved", "rejected", "revised"),
}
# canonical event code ↔ (domain type, subtype). Codes are what the IR schema, inference queries and tests use.
EVENT_CODES: Dict[str, Tuple[str, str]] = {
    "AUDIO_SILENCE": ("AudioEvent", "silence"), "AUDIO_ACTIVE": ("AudioEvent", "active"), "LOUDNESS_MEASURE": ("AudioEvent", "loudness"),
    "AUDIO_DROPOUT": ("AudioEvent", "dropout"), "AUDIO_CLIPPING": ("AudioEvent", "clipping"),
    "SPEECH": ("SpeechEvent", "speech"), "SPEAKER": ("SpeakerEvent", "speaker"), "SCENE_CHANGE": ("SceneEvent", "visual_change"),
    "SLIDE": ("SlideEvent", "slide"), "CAMERA": ("CameraEvent", "camera"), "CAPTION": ("CaptionEvent", "caption"),
    "INCIDENT_BLACK_FRAME": ("IncidentEvent", "black_frame"), "INCIDENT_FREEZE": ("IncidentEvent", "freeze"),
    "INCIDENT_AUDIO_DROPOUT": ("IncidentEvent", "audio_dropout"), "INCIDENT_CORRUPTED_FRAME": ("IncidentEvent", "corrupted_frame"),
    "USER_DECISION": ("UserDecisionEvent", "approved"),   # subtype refined from metadata.action
}
IMPLEMENTED_CODES = ("AUDIO_SILENCE", "AUDIO_ACTIVE", "LOUDNESS_MEASURE", "SPEECH", "SPEAKER", "SLIDE", "USER_DECISION")   # the only codes this codebase generates
KIND_FOR_PROVENANCE = {"OBSERVED": "OBSERVED", "DERIVED": "OBSERVED", "INFERRED": "INFERRED", "AI_GENERATED": "INFERRED", "USER": "USER"}


def event_id(asset_id: Optional[str], code: str, subtype: str, rng: Dict[str, Any], source: str, evidence: Iterable[str]) -> str:
    """Deterministic identity: same asset + type + subtype + range + source + evidence → same id. Not an observation id,
    analysis id, cache key, operation id or job id."""
    return "evt_" + stable_hash([asset_id, code, subtype, round(float(rng["start"]), 6), None if rng.get("end") is None else round(float(rng["end"]), 6), source, sorted(evidence)])[:16]


def classify(event: Event) -> Event:
    """Fill event_type / subtype / provenance / asset_id from the canonical code, kind and timeline (idempotent)."""
    if event.type in EVENT_CODES:
        et, st = EVENT_CODES[event.type]
        event.event_type = event.event_type or et
        if event.type == "USER_DECISION":
            action = str((event.metadata or {}).get("action", "")).lower()
            event.subtype = event.subtype or {"approved": "approved", "rejected": "rejected", "revised": "revised"}.get(action, st)
        else:
            event.subtype = event.subtype or st
    if not event.provenance:
        event.provenance = {"OBSERVED": "OBSERVED", "INFERRED": "INFERRED", "USER": "USER"}.get(event.kind, "")
    if event.asset_id is None and str(event.timeline_id).startswith("asset:"):
        event.asset_id = event.timeline_id.split(":", 1)[1]
    return event


def events_from_observation(obs: Observation, asset: Asset) -> List[Event]:
    """Deterministic Observation → Event transformation for the implemented observation kinds. Only a validated tool
    measurement (provenance OBSERVED, source '<tool>@<version>') may become an OBSERVED event; anything else is refused.
    media_probe is container / scalar information and yields no event."""
    if getattr(obs, "provenance", "OBSERVED") != "OBSERVED" or "@" not in str(obs.source or "") or str(obs.source).startswith("ai"):
        raise ValueError(f"only tool measurements become OBSERVED events (observation {obs.id}: provenance={getattr(obs, 'provenance', None)!r}, source={obs.source!r})")
    if obs.asset_id != asset.id:
        raise ValueError(f"observation {obs.id} belongs to asset {obs.asset_id}, not {asset.id}")
    dur = (asset.technical or {}).get("duration")
    tid = f"asset:{asset.id}"
    gen = f"observation_to_event@{TRANSFORM_VERSION}"
    out: List[Event] = []

    def mk(code: str, rng: TimeRange, metadata: Dict[str, Any]) -> Event:
        r = rng.to_dict()
        e = Event(type=code, timeline_id=tid, range=r, source=obs.source, kind="OBSERVED", confidence=None, evidence=[obs.id], metadata=metadata,
                  id=event_id(asset.id, code, EVENT_CODES[code][1], r, obs.source, [obs.id]), asset_id=asset.id, provenance="OBSERVED", generator=gen)
        return classify(e)

    if obs.kind == "silence":
        if isinstance(obs.data.get("segments"), list):   # measurement Skill layout: classified segments
            for seg in obs.data["segments"]:
                end = seg.get("end") if seg.get("end") is not None else dur
                out.append(mk("AUDIO_SILENCE", TimeRange(seg["start"], end), {"threshold_db": obs.data.get("threshold_db"), "runs_to_end": bool(seg.get("runs_to_end")), "position": seg.get("type")}))
            prev = 0.0
            for seg in sorted(obs.data["segments"], key=lambda x: x["start"]):
                if seg["start"] > prev:
                    out.append(mk("AUDIO_ACTIVE", TimeRange(prev, seg["start"]), {}))
                prev = max(prev, seg.get("end") if seg.get("end") is not None else (dur or prev))
            if dur is not None and prev < dur:
                out.append(mk("AUDIO_ACTIVE", TimeRange(prev, dur), {}))
        else:
            for se in obs.data.get("silences") or []:
                end = se[1] if se[1] is not None else dur
                out.append(mk("AUDIO_SILENCE", TimeRange(se[0], end), {"threshold_db": obs.data.get("threshold_db"), "runs_to_end": se[1] is None}))
            for ke in obs.data.get("keep") or []:
                out.append(mk("AUDIO_ACTIVE", TimeRange(ke[0], ke[1]), {}))
    elif obs.kind == "loudness":
        if dur is not None:   # the integrated measurement covers the whole programme; without a duration there is no range to place it on
            out.append(mk("LOUDNESS_MEASURE", TimeRange(0.0, dur), dict(obs.data)))
    elif obs.kind == "transcript":
        # one SpeechEvent per recognised segment: interval + text as recognised. No merge, no importance, no edit point. A speaker id only
        # when the transcript records transcription-skill's merge of diarization-skill's turns (ADR-096): an anonymous SPEAKER_nn of those turns
        tr = obs.data
        sa = tr.get("speaker_assignment") if isinstance(tr.get("speaker_assignment"), dict) else None
        for seg in tr.get("segments") or []:
            if seg.get("speaker_id") is not None and (sa is None or seg["speaker_id"] not in (sa.get("speakers") or [])):
                raise ValueError(f"transcript segment {seg.get('id')} carries a speaker id {seg['speaker_id']!r} no recorded speaker merge gave it; recognition never identifies speakers")
            e = mk("SPEECH", TimeRange(float(seg["start"]), float(seg["end"])),
                   {"text": seg.get("text", ""), "language": tr.get("language"), "language_source": tr.get("language_source"), "segment_id": seg.get("id"),
                    "transcript_id": tr.get("id"), "engine": tr.get("engine"), "speaker_id": seg.get("speaker_id") if sa is not None else None, "words": len(seg["words"]) if isinstance(seg.get("words"), list) else None})
            e.confidence = seg.get("confidence") if isinstance(seg.get("confidence"), (int, float)) else None
            out.append(e)
    elif obs.kind == "speaker_turns":
        # ADR-086: one SpeakerEvent per exclusive turn of diarization-skill (an anonymous voice id, never a person); overlap regions are
        # kept in the observation, not as events (exclusive turns give one speaker at a time, the Skill's own convention)
        doc = obs.data
        for t in doc.get("turns") or []:
            out.append(mk("SPEAKER", TimeRange(float(t["start"]), float(t["end"])),
                          {"speaker_id": t.get("speaker"), "speaker_count": doc.get("speaker_count"), "anonymous": True, "model": (doc.get("engine") or {}).get("model")}))
    elif obs.kind == "slide_timeline":
        # ADR-091: one SlideEvent per measured slide (a picture held at least min_dwell seconds); the deck page only where slide-sync matched one
        segs = {s.get("index"): s for s in ((obs.data.get("match") or {}).get("segments") or [])}
        for s in obs.data.get("slides") or []:
            mseg = segs.get(s.get("index")) or {}
            e = mk("SLIDE", TimeRange(float(s["start"]), float(s["end"])), {"slide": s.get("index"), "page": mseg.get("page"), "matched": mseg.get("page") is not None if mseg else None,
                                                                            "reason": mseg.get("reason")})
            e.confidence = mseg.get("confidence") if isinstance(mseg.get("confidence"), (int, float)) else None
            out.append(e)
    return out


def sort_key(e: Event) -> Tuple[float, float, str, str]:
    """Stable temporal ordering: start, end (point = start), canonical type, id."""
    s = float(e.range["start"])
    en = e.range.get("end")
    return (s, s if en is None else float(en), e.type, e.id)


def sort_events(events: Iterable[Event]) -> List[Event]:
    return sorted(events, key=sort_key)


def overlaps(a: Event, b: Event) -> bool:
    return a.temporal_range().overlaps(b.temporal_range())


def contains(a: Event, b: Event) -> bool:
    return a.temporal_range().contains(b.temporal_range())


def precedes(a: Event, b: Event) -> bool:
    return a.temporal_range().precedes(b.temporal_range())


def adjacent(a: Event, b: Event, tolerance: float = 1e-6) -> bool:
    return a.temporal_range().adjacent(b.temporal_range(), tolerance)


def validate_event(e: Event, assets: Dict[str, Optional[float]], known_evidence: Optional[Iterable[str]] = None) -> List[str]:
    """Errors for an event. `assets` maps asset id → duration (None when unknown: bounds are not checked, never guessed)."""
    from ..media.analysis import leak_scan   # local import: media imports temporal
    errs: List[str] = []
    if not e.id or not str(e.id).startswith("evt_"):
        errs.append(f"invalid event id {e.id!r}")
    if e.type not in EVENT_CODES:
        errs.append(f"unknown event type {e.type!r}")
    else:
        et, _ = EVENT_CODES[e.type]
        if e.event_type and e.event_type != et:
            errs.append(f"event_type {e.event_type!r} does not match code {e.type}")
        if e.subtype and e.subtype not in EVENT_TYPES.get(e.event_type or et, ()):
            errs.append(f"subtype {e.subtype!r} is not valid for {e.event_type or et}")
    try:
        rng = e.temporal_range()
    except (ValueError, KeyError, TypeError) as ex:
        errs.append(f"invalid temporal range: {ex}")
        rng = None
    if e.timeline_id != "master":
        aid = e.asset_id or (e.timeline_id.split(":", 1)[1] if str(e.timeline_id).startswith("asset:") else None)
        if aid not in assets:
            errs.append(f"event references unknown asset {aid!r}")
        elif rng is not None and not rng.within(assets[aid]):
            errs.append(f"event range {rng.start}-{rng.stop} exceeds asset duration {assets[aid]}")
    if e.provenance and e.provenance not in EVENT_PROVENANCE:
        errs.append(f"invalid provenance {e.provenance!r}")
    if e.kind not in ("OBSERVED", "INFERRED", "USER"):
        errs.append(f"invalid kind {e.kind!r}")
    if e.provenance and KIND_FOR_PROVENANCE.get(e.provenance) != e.kind:
        errs.append(f"provenance {e.provenance} cannot be recorded as kind {e.kind} (an AI or inferred event is never OBSERVED)")
    if e.kind == "OBSERVED":
        if "@" not in str(e.source or "") or str(e.source).startswith("ai"):
            errs.append(f"OBSERVED event needs a tool source '<tool>@<version>', got {e.source!r}")
        if not e.evidence:
            errs.append("OBSERVED event must cite observation evidence")
    if known_evidence is not None:
        missing = [x for x in e.evidence if x not in set(known_evidence)]
        if missing:
            errs.append("evidence not found: " + ", ".join(missing))
    if e.confidence is not None and not (0.0 <= float(e.confidence) <= 1.0):
        errs.append(f"confidence {e.confidence} outside 0..1")
    errs += [f"metadata leaks {what}" for what in leak_scan(e.metadata or {}, "metadata")]
    if leak_scan({"source": e.source}):
        errs.append("source looks like a command or credential")
    return errs


def safe_event_summary(e: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """What an AI provider may see of an event: identity, classification, range, provenance and evidence ids, scrubbed
    metadata. AI_GENERATED events are never offered back as evidence."""
    from ..media.analysis import scrub_for_ai
    if not e.get("id") or e.get("provenance") == "AI_GENERATED" or e.get("kind") not in ("OBSERVED", "INFERRED", "USER"):
        return None
    return {"id": e["id"], "type": e.get("type"), "event_type": e.get("event_type"), "subtype": e.get("subtype"), "asset_id": e.get("asset_id"),
            "timeline_id": e.get("timeline_id"), "range": e.get("range"), "provenance": e.get("provenance") or e.get("kind"), "evidence": list(e.get("evidence") or []),
            "metadata": scrub_for_ai(e.get("metadata") or {})}
