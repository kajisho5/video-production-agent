"""Long video → a set of vertical clips (ADR-138, roadmap step 25): the AI names each clip by its first and last transcript segment, with a hook
and the words it quotes as the reason; the agent takes the times from the transcript, a person confirms each clip, and each confirmed clip is
planned and rendered as a project of its own: trimmed, filled to 9:16, captions burned in, a hook card, the platform's preset and, when asked,
published.

    transcript observation → numbered segments [index, start, end, text]
      → AI recommendation (intent `short_clips`, params clips = [{first_segment, last_segment, hook, quote}, ...])
      → one `shorts.clip` decision per clip (CONFIRM floor) with its source range, hook and quote
      → IR `delivery.shorts` (the set: platform, preset, length limits, the clips)
      → render: per APPROVED clip a child project (edit.trim, edit.fill 9:16 at 1080, subtitle burned with the shorts caption profile,
        motion.hook, the platform's preset), rendered; its deliverable and status are listed under the parent's `shorts`

Vocabulary, rules and arithmetic only, no execution and no AI call:

- **Requirements** (explicit `--set`): `shorts.count` (1..10) asks for the set; `shorts.platform` (shorts | tiktok | reels, default shorts: the
  engine's vertical presets); `shorts.max_seconds` (5..the platform's limit, default 60) and `shorts.min_seconds` (3..max, default 10);
  `shorts.publish=true` plans the upload of each clip (publish-skill, ADR-087, a confirmation token per clip as always) with
  `shorts.privacy` (default private; Reels only takes public, so it must be named).
- **Rule `short_clips@1`** (`derive_clips`): at most `count` clips; each names segments first <= last inside the transcript; clips do not
  overlap and are in time order; the hook is 1..60 printable characters; the quote is 1..200 characters found word for word (spacing and case
  folded) in the clip's own segments. A clip runs from its first segment's start less 0.15 s to its last segment's end plus 0.15 s, clamped to
  the source, and lasts min..max seconds. A set that breaks the rule is refused (a KEEP decision with the reason), never corrected.
- **Not planned** (BLOCK with the reason): more than one input, the audio production path, an input without a picture.
- **Reframe**: the clip is filled to 9:16 from the centre (video-editing FILL). Following the speaker (detection-skill's reframe path) needs an
  engine that crops along a path; not done here.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence

from ..media.analysis import _scrub
from ..models import Requirement
from .chapters import numbered_segments
from .editing import EditRequirementError, _bool, _number

RULE_VERSION = "short_clips@1"
PREFIX = "shorts."
COUNT_KEY, PLATFORM_KEY, MAX_KEY, MIN_KEY, PUBLISH_KEY, PRIVACY_KEY = ("shorts.count", "shorts.platform", "shorts.max_seconds", "shorts.min_seconds",
                                                                       "shorts.publish", "shorts.privacy")
REQUIREMENT_KEYS = (COUNT_KEY, PLATFORM_KEY, MAX_KEY, MIN_KEY, PUBLISH_KEY, PRIVACY_KEY)
COUNT_RANGE = (1, 10)
# the engine's vertical presets (1080x1920) and their duration limits (ffmpeg-skill _platforms.py, as of its upload-limit table);
# the publish-skill platform each one uploads to
PLATFORMS = {"shorts": {"max_seconds": 180.0, "publish": "youtube"}, "tiktok": {"max_seconds": 600.0, "publish": "tiktok"},
             "reels": {"max_seconds": 90.0, "publish": "reels"}}
DEFAULT_PLATFORM = "shorts"
MAX_DEFAULT = 60.0
MIN_DEFAULT = 10.0
HANDLE = 0.15
MAX_HOOK = 60
MAX_QUOTE = 200
SUBJECT = "shorts.clip"        # one decision per clip: subject shorts.clip.<n> (a rejection names one clip)
SET_SUBJECT = "shorts.set"
AI_INTENT = "short_clips"
APPROVAL_KEY = (f"{SUBJECT}.approval", "CONFIRM")   # one policy key for every clip
MAX_CONTEXT_SEGMENTS = 4000
FRAME = {"aspect": "9:16", "width": 1080}
CAPTION_PROFILE = "shorts"


def is_clip(subject: Any) -> bool:
    return isinstance(subject, str) and subject.startswith(SUBJECT + ".") and subject[len(SUBJECT) + 1:].isdigit()


class ClipError(ValueError):
    """A clip set that breaks short_clips@1: refused, never corrected."""


# ---- requirements ------------------------------------------------------------------------------------------------------
def parse_shorts_requirements(m: Dict[str, Requirement]) -> Optional[Dict[str, Any]]:
    given = {k: r for k, r in m.items() if k.startswith(PREFIX) and r.provenance != "DEFAULT"}
    if not given:
        return None
    unknown = sorted(k for k in given if k not in REQUIREMENT_KEYS)
    if unknown:
        raise EditRequirementError(f"unknown requirement key(s): {', '.join(unknown)}; shorts.* keys are {', '.join(REQUIREMENT_KEYS)}")
    if COUNT_KEY not in given:
        raise EditRequirementError(f"{sorted(given)[0]} needs {COUNT_KEY} (how many clips, {COUNT_RANGE[0]}..{COUNT_RANGE[1]})")
    count = _number(given[COUNT_KEY].value, COUNT_KEY, *COUNT_RANGE)
    if float(count) != int(float(count)):
        raise EditRequirementError(f"{COUNT_KEY} must be a whole number")
    platform = str(given[PLATFORM_KEY].value) if PLATFORM_KEY in given else DEFAULT_PLATFORM
    if platform not in PLATFORMS:
        raise EditRequirementError(f"{PLATFORM_KEY} must be one of {', '.join(PLATFORMS)}, got {platform!r}")
    limit = PLATFORMS[platform]["max_seconds"]
    max_s = float(_number(given[MAX_KEY].value, MAX_KEY, 5.0, limit)) if MAX_KEY in given else min(MAX_DEFAULT, limit)
    min_s = float(_number(given[MIN_KEY].value, MIN_KEY, 3.0, max_s)) if MIN_KEY in given else min(MIN_DEFAULT, max_s)
    publish = _bool(given[PUBLISH_KEY].value, PUBLISH_KEY) if PUBLISH_KEY in given else False
    privacy = str(given[PRIVACY_KEY].value) if PRIVACY_KEY in given else None
    if privacy is not None and not publish:
        raise EditRequirementError(f"{PRIVACY_KEY} is set but {PUBLISH_KEY} is not: refusing an ambiguous request")
    return {"count": int(float(count)), "platform": platform, "preset": platform, "max_seconds": max_s, "min_seconds": min_s, "publish": publish,
            "privacy": privacy, "requirements": [given[k] for k in REQUIREMENT_KEYS if k in given]}


def requested(user_requirements: Dict[str, Any]) -> bool:
    return COUNT_KEY in (user_requirements or {})


def check_keys(user_requirements: Dict[str, Any]) -> None:
    given = {k: v for k, v in (user_requirements or {}).items() if k.startswith(PREFIX)}
    if given:
        parse_shorts_requirements({k: Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in given.items()})


# ---- the AI's context ----------------------------------------------------------------------------------------------------
def ai_context(analysis: Any, req: Dict[str, Any]) -> Dict[str, Any]:
    out: List[Dict[str, Any]] = []
    for o in getattr(analysis, "observations", []) or []:
        if o.kind != "transcript":
            continue
        segs = numbered_segments((o.data or {}).get("segments") or [])
        if not segs:
            continue
        rec = {"asset_id": o.asset_id, "transcript_id": o.id, "segment_count": len(segs),
               "segments": [[s["index"], round(s["start"], 1), round(s["end"], 1), s["text"][:200]] for s in segs[:MAX_CONTEXT_SEGMENTS]]}
        if len(segs) > MAX_CONTEXT_SEGMENTS:
            rec["truncated_after"] = MAX_CONTEXT_SEGMENTS
        out.append(rec)
    signals = [{"observation_id": o.id, "asset_id": o.asset_id, "data": _scrub(o.data or {})} for o in getattr(analysis, "observations", []) or []
               if o.kind == "highlight_signals"]
    return {"shorts": {"intent": AI_INTENT, "rule": RULE_VERSION, "count": req["count"], "platform": req["platform"], "min_seconds": req["min_seconds"],
                       "max_seconds": req["max_seconds"], "transcripts": out, "highlight_signals": signals,
                       "params": {"clips": [{"first_segment": "<index>", "last_segment": "<index>", "hook": "<hook title>", "quote": "<words of the clip>"}]},
                       "instructions": (f"For intent short_clips give params {{clips: [...]}} with at most {req['count']} clips for vertical short videos: the moments "
                                        "that stand on their own and would make someone watch. Each clip names the 0-based index of its first and last segment "
                                        "(transcripts[].segments: [index, start, end, text]), a hook title of at most 60 characters in the talk's language, and a quote: "
                                        f"words copied exactly from those segments that show why the clip works. A clip lasts {req['min_seconds']:g}-{req['max_seconds']:g} s; "
                                        "clips do not overlap and are in time order. Never give seconds; the agent takes them from the transcript. Cite the "
                                        "transcript_id (and any highlight_signals observation you used) in evidence. One short_clips recommendation.")}}


# ---- the rule ------------------------------------------------------------------------------------------------------------
def _fold(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def _printable(v: Any, what: str, limit: int, n: int) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ClipError(f"clip {n}: the {what} must be text")
    if any(unicodedata.category(c).startswith("C") for c in v):
        raise ClipError(f"clip {n}: the {what} holds a control character")
    t = " ".join(v.split())
    if len(t) > limit:
        raise ClipError(f"clip {n}: the {what} is {len(t)} characters, at most {limit}")
    return t


def derive_clips(params: Any, segments: List[Dict[str, Any]], duration: float, req: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(params, dict) or set(params) != {"clips"}:
        raise ClipError("params must be exactly {clips: [{first_segment, last_segment, hook, quote}]}: segment numbers, never times")
    raw = params["clips"]
    if not isinstance(raw, list) or not raw:
        raise ClipError("no clip was named")
    if len(raw) > req["count"]:
        raise ClipError(f"{len(raw)} clips named, {req['count']} asked for")
    if duration is None or not duration > 0:
        raise ClipError("the source has no measured duration")
    out: List[Dict[str, Any]] = []
    for n, c in enumerate(raw, 1):
        if not isinstance(c, dict) or set(c) != {"first_segment", "last_segment", "hook", "quote"}:
            raise ClipError(f"clip {n} must be exactly {{first_segment, last_segment, hook, quote}}")
        a, b = c["first_segment"], c["last_segment"]
        for v in (a, b):
            if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < len(segments):
                raise ClipError(f"clip {n}: segment {v!r} is not one of 0..{len(segments) - 1}")
        if b < a:
            raise ClipError(f"clip {n}: last segment {b} comes before first segment {a}")
        hook = _printable(c["hook"], "hook", MAX_HOOK, n)
        quote = _printable(c["quote"], "quote", MAX_QUOTE, n)
        said = " ".join(s["text"] for s in segments[a:b + 1])
        if _fold(quote) not in _fold(said):
            raise ClipError(f"clip {n}: the quote {quote!r} is not words of segments {a}..{b}")
        start = round(max(0.0, float(segments[a]["start"]) - HANDLE), 3)
        end = round(min(float(duration), float(segments[b]["end"]) + HANDLE), 3)
        if out and start < out[-1]["end"] - 1e-9:
            raise ClipError(f"clip {n} starts at {start:g}s, inside clip {n - 1} (to {out[-1]['end']:g}s): clips do not overlap and are in time order")
        length = end - start
        if not req["min_seconds"] - 1e-9 <= length <= req["max_seconds"] + 1e-9:
            raise ClipError(f"clip {n} lasts {length:.1f}s, outside {req['min_seconds']:g}-{req['max_seconds']:g}s")
        out.append({"index": n, "first_segment": a, "last_segment": b, "start": start, "end": end, "seconds": round(length, 3), "hook": hook, "quote": quote})
    return out


def child_requirements(clip: Dict[str, Any], rec: Dict[str, Any]) -> Dict[str, Any]:
    """The `--set` keys of the project that renders one clip: the cut, the vertical frame, captions burned in, the hook card and the upload."""
    out: Dict[str, Any] = {"edit.trim": f"{clip['start']:.3f}-{clip['end']:.3f}", "edit.fill": FRAME["aspect"], "edit.fill.width": FRAME["width"],
                           "subtitle": True, "subtitle.burn_in": True, "subtitle.caption_profile": CAPTION_PROFILE, "motion.hook": clip["hook"],
                           # the clip is what the person approved: no further silence trim inside it
                           "edit.trim_leading_silence": False, "edit.trim_trailing_silence": False}
    if rec.get("publish"):
        out.update({"qc": True, "publish.platform": PLATFORMS[rec["platform"]]["publish"], "publish.title": clip["hook"]})
        if rec.get("privacy"):
            out["publish.privacy"] = rec["privacy"]
    return out


def child_profile(base: Dict[str, Any], platform: str) -> Dict[str, Any]:
    """The profile of a clip's project: the given profile with one delivery target, the platform's vertical preset."""
    prof = dict(base)
    prof.update(name=f"shorts-{platform}", delivery={"targets": [{"id": platform, "preset": platform, "platform": platform, "artifact_type": "MASTER"}]})
    return prof


# ---- decision ------------------------------------------------------------------------------------------------------------
def decide_shorts(eng: Any, m: Dict[str, Requirement], analysis: Any, inferences: List[Any], approval_for: Any, ai_prefix: str) -> Optional[Dict[str, Any]]:
    req = parse_shorts_requirements(m)
    if req is None:
        return None
    reqs = req["requirements"]
    req_ids = [r.id for r in reqs]
    base = {"rule": RULE_VERSION, **{k: req[k] for k in ("count", "platform", "preset", "min_seconds", "max_seconds", "publish", "privacy")}}

    def block(why: str, asset_id: Optional[str] = None) -> Dict[str, Any]:
        eng.decide(subject=SET_SUBJECT, type="BLOCK", decision="BLOCK: a set of vertical clips", reason=why + "; no clip is planned", confidence=1.0, evidence=req_ids,
                   risk="MEDIUM", approval="BLOCK", provenance="USER", params={**base, "asset_id": asset_id}, requirements=reqs)
        return req
    sw = m.get("audio.production")
    if sw is not None and sw.provenance != "DEFAULT" and _bool(sw.value, "audio.production"):
        return block("the audio production path delivers audio; vertical clips are pictures")
    if len(analysis.assets) != 1:
        return block(f"{len(analysis.assets)} inputs: a set of clips is cut from one talk in one input")
    asset = analysis.assets[0]
    if not (asset.technical or {}).get("video"):
        return block("the input has no picture", asset.id)
    tr = next((o for o in analysis.observations if o.kind == "transcript" and o.asset_id == asset.id), None)
    recs = [i for i in inferences if i.kind == ai_prefix + AI_INTENT and i.asset_id == asset.id]
    if not recs:
        why = ("no transcript of the input (speech recognition did not run or failed)" if tr is None else
               "no AI clip recommendation was given (no provider, the provider failed, or the host has not answered the brief yet: revise with --ai-recommendations)")
        eng.decide(subject=SET_SUBJECT, type="KEEP", decision="no clips yet: no AI clip recommendation", reason=f"{COUNT_KEY} asks the AI for clips; {why}, so no clip is planned",
                   confidence=1.0, evidence=req_ids + ([tr.id] if tr is not None else []), risk="LOW", approval="AUTO", provenance="USER", params={**base, "asset_id": asset.id},
                   requirements=reqs)
        return req
    ev = [i.id for i in recs] + req_ids + ([tr.id] if tr is not None else [])
    why = None
    clips: List[Dict[str, Any]] = []
    if len(recs) > 1:
        why = f"{len(recs)} short_clips recommendations: which one is meant is not guessed"
    elif tr is None:
        why = "no transcript: segment numbers cannot be read"
    else:
        try:
            segs = numbered_segments((tr.data or {}).get("segments") or [])
            duration = (asset.technical or {}).get("duration")
            clips = derive_clips((recs[0].data or {}).get("params"), segs, duration if duration is not None else (tr.data or {}).get("duration"), req)
        except ClipError as e:
            why = str(e)
    if why is not None:
        eng.decide(subject=SET_SUBJECT, type="KEEP", decision="no clips: the AI's clip set was refused", reason=f"refused ({RULE_VERSION}): {why}; a clip set is never corrected",
                   confidence=1.0, evidence=ev, risk="LOW", approval="AUTO", provenance="AI_GENERATED", requirements=reqs,
                   params=_scrub({**base, "asset_id": asset.id, "refused": why, "ai_params": (recs[0].data or {}).get("params") if len(recs) == 1 else None}))
        return req
    inf = recs[0]
    for c in clips:
        eng.decide(subject=f"{SUBJECT}.{c['index']}", type="TRANSFORM",
                   decision=f"clip {c['index']}: {c['start']:.2f}-{c['end']:.2f}s ({c['seconds']:.1f}s) as a 9:16 {req['platform']} clip, hook “{c['hook']}”, "
                            f"because “{c['quote']}”",
                   reason=f"{inf.statement}; the AI named the clip's segments {c['first_segment']}-{c['last_segment']} ({inf.data.get('provider')}/{inf.data.get('model')}) and "
                          f"quoted its words; the agent took the times from the transcript with {RULE_VERSION}; the clip is cut, filled to 9:16 from the centre, captioned "
                          "(burned in) and given the hook card in a project of its own, only after this is confirmed",
                   confidence=inf.confidence, evidence=ev, risk="MEDIUM", approval=approval_for(SUBJECT, floor="CONFIRM", keys={SUBJECT: APPROVAL_KEY}),
                   provenance="AI_GENERATED", params={**base, "asset_id": asset.id, "transcript_id": tr.id, "inference_id": inf.id, **c}, requirements=reqs)
    return req


def shorts_record(decisions: Sequence[Any]) -> Optional[Dict[str, Any]]:
    """IR `delivery.shorts` from the clip decisions that are not rejected (None: no clip)."""
    live = [d for d in decisions if is_clip(d.subject) and d.type == "TRANSFORM" and d.status != "REJECTED"]
    if not live:
        return None
    p0 = live[0].params
    keys = ("index", "first_segment", "last_segment", "start", "end", "seconds", "hook", "quote")
    return {"type": "delivery.shorts", "asset": p0["asset_id"], "rule": RULE_VERSION, **{k: p0[k] for k in ("platform", "preset", "min_seconds", "max_seconds", "publish", "privacy")},
            "transcript_id": p0.get("transcript_id"), "clips": [dict({k: d.params[k] for k in keys}, decision_ids=[d.id]) for d in live]}


def check_shorts(doc: Dict[str, Any]) -> List[str]:
    rec = (doc.get("delivery") or {}).get("shorts")
    if not rec:
        return []
    errs: List[str] = []
    if rec.get("platform") not in PLATFORMS:
        errs.append(f"delivery.shorts platform {rec.get('platform')!r} is not one of {', '.join(PLATFORMS)}")
    decs = {d["id"]: d for d in doc.get("decisions") or []}
    last_end = -1.0
    for c in rec.get("clips") or []:
        length = float(c["end"]) - float(c["start"])
        if not float(rec["min_seconds"]) - 1e-6 <= length <= float(rec["max_seconds"]) + 1e-6:
            errs.append(f"delivery.shorts clip {c['index']} lasts {length:.3f}s, outside {rec['min_seconds']:g}-{rec['max_seconds']:g}s")
        if float(c["start"]) < last_end - 1e-6:
            errs.append(f"delivery.shorts clip {c['index']} overlaps the clip before")
        last_end = float(c["end"])
        cited = [decs.get(i) for i in c.get("decision_ids") or []]
        if not cited or any(d is None or not is_clip(d.get("subject")) for d in cited):
            errs.append(f"delivery.shorts clip {c['index']} must cite its {SUBJECT} decision")
        if not re.match(r"^\S", str(c.get("hook") or "")):
            errs.append(f"delivery.shorts clip {c['index']} has no hook")
    return errs
