"""Chapters from the transcript (ADR-137, roadmap step 26): the AI names where the topic changes by segment number, the agent writes them.

    transcript observation → numbered segments [index, start, text] (the whole transcript, time order)
      → AI recommendation (intent `transcript_chapters`, params chapters = [{segment, title}, ...])
      → `chapters.transcript` decision (CONFIRM floor) with each chapter's source start, title and its first words quoted from the transcript
      → planner: the starts on the programme timeline (through the subject's trim), `delivery.chapters` in the IR, one `container_chapters`
        step per preset target, and the YouTube description timestamps when the plan publishes to YouTube
      → compiler: the engine's chapter writer on the exported file (streams copied), which then is the deliverable

Vocabulary, rules and arithmetic only, no execution and no AI call:

- **Requirements** (explicit `--set`, or a brief): `delivery.chapters=true` asks for them; `delivery.chapters.min_seconds` (10..600, default 10:
  YouTube's shortest chapter, publish-skill's `chapters.min_length_s`) is the shortest chapter accepted.
- **What the AI reads** (`ai_context`): per transcript, its segments as [index, start (0.1 s), text (at most 200 characters)]. It answers with
  segment numbers and titles, never times: the agent takes each start from the recorded transcript.
- **Rule `transcript_chapters@1`** (`derive_chapters`): the cited segments strictly increase and exist; there are at least two chapters; each
  title is 1..100 printable characters. Chapter 1 starts at 0 s whatever segment it cites (a chapter list starts at the start of the programme,
  YouTube's 0:00); chapter k > 1 starts at its segment's start. Each chapter is at least min_seconds long, the last one against the duration.
  The evidence of each chapter is the first words of its segment (at most 6 words or 40 characters), quoted. A list that breaks a rule is refused
  (a KEEP decision with the reason), never corrected.
- **Not planned** (a BLOCK with the reason): more than one video input, the audio production path, a slide timeline asked for (slides.detect:
  the deck's pages are the chapters there), or an edit that moves time other than a trim (speed, a joined still, takes, selects, a joined,
  switched or grid programme).
- **On the programme** (`to_programme`): a trim keeps source ranges; a source time maps to the kept seconds before it. A chapter whose start
  was removed starts where the next kept range starts. A chapter that becomes shorter than min_seconds after the trim joins the one before it
  and is listed under `dropped` with the reason (the planner says so in the summary).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..media.analysis import _scrub
from ..models import Requirement
from .editing import EditRequirementError, _bool, _number

RULE_VERSION = "transcript_chapters@1"
KEY = "delivery.chapters"
MIN_KEY = "delivery.chapters.min_seconds"
REQUIREMENT_KEYS = (KEY, MIN_KEY)
MIN_DEFAULT = 10.0        # YouTube: the minimum length of a chapter is 10 seconds (publish-skill contract, chapters.min_length_s)
MIN_RANGE = (10.0, 600.0)
SUBJECT = "chapters.transcript"
SKILL = "container_chapters"
TOOL_NAME = "metadata"
AI_INTENT = "transcript_chapters"
APPROVAL_KEY = (f"{SUBJECT}.approval", "CONFIRM")
MAX_CONTEXT_SEGMENTS = 4000
MAX_SEGMENT_TEXT = 200
MAX_TITLE = 100
FIRST_WORDS = 6
FIRST_CHARS = 40
YOUTUBE_MIN_TIMESTAMPS = 3   # publish-skill contract (YouTube): at least three timestamps in ascending order
CHAPTER_EXTENSIONS = (".mp4", ".m4v", ".mov", ".mkv", ".webm")   # containers ffmpeg-skill metadata writes chapters into
# requirement keys whose edits move the programme's time against the source in a way other than a trim
TIME_MOVING_KEYS = ("edit.speed", "edit.image_insert", "edit.takes", "edit.selects", "edit.concat", "edit.switch", "edit.grid")
SLIDES_KEY = "slides.detect"


class ChapterError(ValueError):
    """A chapter list that breaks transcript_chapters@1: refused, never corrected."""


# ---- requirements ------------------------------------------------------------------------------------------------------
def parse_chapter_requirements(m: Dict[str, Requirement]) -> Optional[Dict[str, Any]]:
    """→ {"min_seconds", "min_provenance", "requirements"} when `delivery.chapters` is on, else None. A refinement without the switch is refused."""
    main = m.get(KEY)
    refine = m.get(MIN_KEY) if m.get(MIN_KEY) is not None and m[MIN_KEY].provenance != "DEFAULT" else None
    if main is None or main.provenance == "DEFAULT" or not _bool(main.value, KEY):
        if refine is not None:
            raise EditRequirementError(f"{MIN_KEY} is set but {KEY} is not: refusing an ambiguous request")
        return None
    out = {"min_seconds": MIN_DEFAULT, "min_provenance": "DEFAULT", "requirements": [main] + ([refine] if refine is not None else [])}
    if refine is not None:
        out["min_seconds"] = float(_number(refine.value, MIN_KEY, *MIN_RANGE))
        out["min_provenance"] = refine.provenance
    return out


def requested(user_requirements: Dict[str, Any]) -> bool:
    v = (user_requirements or {}).get(KEY)
    return v is not None and v is not False and not (isinstance(v, str) and v.strip().lower() in ("false", "no", "off", "0"))


def check_keys(user_requirements: Dict[str, Any]) -> None:
    """The `--set` gate: known keys, a switch, the minimum in range (before any analysis runs)."""
    given = {k: v for k, v in (user_requirements or {}).items() if k == KEY or k.startswith(KEY + ".")}
    if not given:
        return
    unknown = sorted(k for k in given if k not in REQUIREMENT_KEYS)
    if unknown:
        raise ValueError(f"unknown requirement key(s): {', '.join(unknown)}; {KEY}.* keys are {', '.join(REQUIREMENT_KEYS)}")
    parse_chapter_requirements({k: Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in given.items()})


# ---- the transcript ------------------------------------------------------------------------------------------------------
def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float("inf"), float("-inf")) else None


def numbered_segments(segments: Sequence[Any]) -> List[Dict[str, Any]]:
    """The transcript's segments with text and times, numbered 0.. in document order (time order: check_transcript refuses others)."""
    out: List[Dict[str, Any]] = []
    for s in segments or []:
        if not isinstance(s, dict):
            continue
        start, end, text = _num(s.get("start")), _num(s.get("end")), s.get("text")
        if start is None or end is None or not isinstance(text, str) or not text.strip():
            continue
        out.append({"index": len(out), "id": s.get("id"), "start": round(start, 3), "end": round(end, 3), "text": text.strip()})
    return out


def first_words(text: str) -> str:
    """The words a chapter starts with, as quoted evidence: at most FIRST_WORDS words, at most FIRST_CHARS characters (no spaces in CJK text)."""
    words = text.split()
    out = " ".join(words[:FIRST_WORDS]) if len(words) > 1 else text
    return out if len(out) <= FIRST_CHARS else out[:FIRST_CHARS].rstrip() + "…"


def ai_context(analysis: Any, min_seconds: Optional[float] = None) -> Dict[str, Any]:
    """What the AI reads to name chapters: per transcript, its numbered segments as [index, start (0.1 s), text] and the duration."""
    out: List[Dict[str, Any]] = []
    durations = {a.id: (a.technical or {}).get("duration") for a in getattr(analysis, "assets", []) or []}
    for o in getattr(analysis, "observations", []) or []:
        if o.kind != "transcript":
            continue
        segs = numbered_segments((o.data or {}).get("segments") or [])
        if not segs:
            continue
        rec = {"asset_id": o.asset_id, "transcript_id": o.id, "duration": durations.get(o.asset_id), "segment_count": len(segs),
               "segments": [[s["index"], round(s["start"], 1), s["text"][:MAX_SEGMENT_TEXT]] for s in segs[:MAX_CONTEXT_SEGMENTS]]}
        if len(segs) > MAX_CONTEXT_SEGMENTS:
            rec["truncated_after"] = MAX_CONTEXT_SEGMENTS
        out.append(rec)
    return {"chapters": {"intent": AI_INTENT, "rule": RULE_VERSION, "transcripts": out, "params": {"chapters": [{"segment": "<segment index>", "title": "<chapter title>"}]},
                         "instructions": ("For intent transcript_chapters give params {chapters: [{segment, title}]}: one chapter where the topic of the talk changes, "
                                          "naming the 0-based index (transcripts[].segments) of the segment it starts with, in increasing order, and a short title "
                                          "(at most 100 characters) in the language of the talk. The first chapter starts at the start of the programme. "
                                          "Each chapter must last at least min_seconds. Never give seconds or times; the agent takes them from the transcript. "
                                          "Cite the transcript_id in evidence. One transcript_chapters recommendation per asset."),
                         "min_seconds": float(min_seconds or MIN_DEFAULT)}}


# ---- the rule ------------------------------------------------------------------------------------------------------------
def _title(v: Any, n: int) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ChapterError(f"chapter {n + 1}: the title must be text")
    if any(unicodedata.category(c).startswith("C") for c in v):   # a newline would start another chapter in the chapter file
        raise ChapterError(f"chapter {n + 1}: the title holds a control character")
    t = " ".join(v.split())
    if len(t) > MAX_TITLE:
        raise ChapterError(f"chapter {n + 1}: the title is {len(t)} characters, at most {MAX_TITLE}")
    return t


def derive_chapters(params: Any, segments: List[Dict[str, Any]], duration: float, min_seconds: float = MIN_DEFAULT) -> Dict[str, Any]:
    """An AI recommendation's params + the numbered segments → {chapters: [{index, segment, start, title, first_words}], duration, rule, min_seconds}.
    Times are the source's (seconds). Raises ChapterError on any broken rule."""
    if not isinstance(params, dict) or set(params) != {"chapters"}:
        raise ChapterError("params must be exactly {chapters: [{segment, title}]}: segment numbers and titles, never times")
    raw = params["chapters"]
    if not isinstance(raw, list) or len(raw) < 2:
        raise ChapterError("at least two chapters are needed (one chapter is no chapter list)")
    if duration is None or not duration > 0:
        raise ChapterError("the source has no measured duration")
    out: List[Dict[str, Any]] = []
    last = -1
    for n, c in enumerate(raw):
        if not isinstance(c, dict) or set(c) != {"segment", "title"}:
            raise ChapterError(f"chapter {n + 1} must be exactly {{segment, title}}")
        seg = c["segment"]
        if not isinstance(seg, int) or isinstance(seg, bool) or not 0 <= seg < len(segments):
            raise ChapterError(f"chapter {n + 1}: segment {seg!r} is not one of 0..{len(segments) - 1}")
        if seg <= last:
            raise ChapterError(f"chapter {n + 1}: segment {seg} does not come after segment {last}")
        last = seg
        s = segments[seg]
        start = 0.0 if n == 0 else float(s["start"])
        if start >= duration:
            raise ChapterError(f"chapter {n + 1} starts at {start:g}s, at or after the end ({duration:g}s)")
        out.append({"index": n + 1, "segment": seg, "segment_id": s.get("id"), "start": round(start, 3), "title": _title(c["title"], n),
                    "first_words": first_words(s["text"])})
    for n, c in enumerate(out):
        end = out[n + 1]["start"] if n + 1 < len(out) else float(duration)
        if end - c["start"] < min_seconds - 1e-9:
            raise ChapterError(f"chapter {c['index']} ({c['title']!r}) lasts {end - c['start']:.1f}s, under the {min_seconds:g}s minimum")
    return {"chapters": out, "duration": round(float(duration), 3), "rule": RULE_VERSION, "min_seconds": float(min_seconds)}


def to_programme(t: float, keep: Optional[Sequence[Sequence[float]]]) -> Tuple[Optional[float], bool]:
    """A source time on the programme timeline of a trim that keeps `keep` (None: not trimmed). (time, moved): moved when t was removed and the
    time is where the next kept range starts; (None, True) when nothing is kept after t."""
    if not keep:
        return round(float(t), 3), False
    acc = 0.0
    for a, b in sorted([float(x), float(y)] for x, y in keep):
        if t < a - 1e-9:
            return round(acc, 3), True
        if t < b - 1e-9:
            return round(acc + (t - a), 3), False
        acc += b - a
    return None, True


def on_programme(rec: Dict[str, Any], keep: Optional[Sequence[Sequence[float]]], programme_seconds: float) -> Dict[str, Any]:
    """The decided chapters (source times) on the programme timeline → {chapters, dropped, programme_seconds}. Chapter 1 stays at 0."""
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    min_s = float(rec.get("min_seconds") or MIN_DEFAULT)
    for c in rec["chapters"]:
        t, moved = (0.0, False) if c["index"] == 1 else to_programme(float(c["start"]), keep)
        if t is None or t >= programme_seconds - 1e-9:
            dropped.append({"index": c["index"], "title": c["title"], "reason": "its start lies after the last kept range of the trim"})
            continue
        kept.append(dict(c, source_start=c["start"], start=t, **({"moved": "its start was cut; it starts where the next kept range starts"} if moved else {})))
    merged: List[Dict[str, Any]] = []
    for n, c in enumerate(kept):
        nxt = next((x["start"] for x in kept[n + 1:]), programme_seconds)
        if merged and nxt - c["start"] < min_s - 1e-9:
            dropped.append({"index": c["index"], "title": c["title"], "reason": f"{nxt - c['start']:.1f}s after the trim, under the {min_s:g}s minimum: joined to the chapter before"})
            continue
        merged.append(c)
    # the last kept chapter may now be short against the end: join it to the one before (never chapter 1)
    while len(merged) > 1 and programme_seconds - merged[-1]["start"] < min_s - 1e-9:
        c = merged.pop()
        dropped.append({"index": c["index"], "title": c["title"], "reason": f"{programme_seconds - c['start']:.1f}s after the trim, under the {min_s:g}s minimum: joined to the chapter before"})
    for n, c in enumerate(merged):
        c["index"] = n + 1
    return {"chapters": merged, "dropped": dropped, "programme_seconds": round(float(programme_seconds), 3)}


def timestamp(seconds: float) -> str:
    """YouTube's description timestamp: M:SS under an hour, H:MM:SS from an hour (whole seconds, rounded down)."""
    s = int(seconds + 1e-6)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def description_lines(chapters: List[Dict[str, Any]]) -> List[str]:
    return [f"{timestamp(c['start'])} {c['title']}" for c in chapters]


def chapters_file_text(chapters: List[Dict[str, Any]]) -> str:
    """The engine's chapter-file format: one `TIME TITLE` per line (seconds to the millisecond)."""
    return "".join(f"{float(c['start']):.3f} {c['title']}\n" for c in chapters)


def with_description(description: Optional[str], chapters: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """The YouTube description with the chapter timestamps after the user's text → (description, None), or (unchanged, why not)."""
    if len(chapters) < YOUTUBE_MIN_TIMESTAMPS:
        return description, f"{len(chapters)} chapter(s): a YouTube description needs at least {YOUTUBE_MIN_TIMESTAMPS} timestamps, so none are written there"
    block = "\n".join(description_lines(chapters))
    if description and re.search(r"(?m)^\s*(?:[-*•]\s*)?\d{1,2}:\d{2}(?::\d{2})?\s", description):
        return description, "the description already holds timestamp lines; the chapters are not added a second time"
    return (description.rstrip() + "\n\n" + block) if description else block, None


# ---- decision ------------------------------------------------------------------------------------------------------------
def _transcript_of(analysis: Any, asset_id: str) -> Optional[Any]:
    return next((o for o in analysis.observations if o.kind == "transcript" and o.asset_id == asset_id), None)


def decide_chapters(eng: Any, m: Dict[str, Requirement], analysis: Any, inferences: List[Any], approval_for: Any, ai_prefix: str, cap_block: Any) -> Optional[Dict[str, Any]]:
    """`delivery.chapters` + the AI's `transcript_chapters` recommendation → one `chapters.transcript` decision: TRANSFORM (CONFIRM floor) with
    the chapters on the source, KEEP with the reason a list was refused or none was given, or BLOCK when the plan cannot carry them."""
    req = parse_chapter_requirements(m)
    if req is None:
        return None
    reqs = req["requirements"]
    req_ids = [r.id for r in reqs]
    base = {"rule": RULE_VERSION, "min_seconds": req["min_seconds"], "min_provenance": req["min_provenance"]}
    videos = [a for a in analysis.assets if (a.technical or {}).get("video")]

    def block(why: str, asset_id: Optional[str] = None) -> None:
        eng.decide(subject=SUBJECT, type="BLOCK", decision="BLOCK: chapters from the transcript", reason=why + "; no chapters are written", confidence=1.0, evidence=req_ids,
                   risk="MEDIUM", approval="BLOCK", provenance="USER", params={**base, "asset_id": asset_id}, requirements=reqs)

    sw = m.get("audio.production")
    if sw is not None and sw.provenance != "DEFAULT" and _bool(sw.value, "audio.production"):
        return block("the audio production path delivers audio, and container chapters are written into the video deliverable") or req
    if len(videos) != 1 or len(analysis.assets) != 1:
        return block(f"{len(analysis.assets)} inputs: chapters from a transcript are planned for one talk in one input") or req
    asset = videos[0]
    slides = m.get(SLIDES_KEY)
    if slides is not None and slides.provenance != "DEFAULT" and _bool(slides.value, SLIDES_KEY):
        return block("the slide timeline is asked for (slides.detect): with slides, the deck's pages are the chapters, and transcript chapters are for talks without "
                     "slides", asset.id) or req
    moving = [k for k in TIME_MOVING_KEYS if k in m and m[k].provenance != "DEFAULT" and m[k].value not in (False, None, "false", "off", "no", "0")]
    if moving:
        return block(f"{', '.join(moving)} moves the programme's time against the source in a way other than a trim, so the transcript's times are not the "
                     "programme's", asset.id) or req
    if cap_block(SKILL, f"capability.{SKILL}") is not None:
        return req
    tr = _transcript_of(analysis, asset.id)
    recs = [i for i in inferences if i.kind == ai_prefix + AI_INTENT and i.asset_id == asset.id]
    if not recs:
        why = ("no transcript of the input (speech recognition did not run or failed)" if tr is None else
               "no AI chapter recommendation was given (no provider, the provider failed, or the host has not answered the brief yet: revise with --ai-recommendations)")
        eng.decide(subject=SUBJECT, type="KEEP", decision="no chapters yet: no AI chapter recommendation", reason=f"{KEY} asks the AI where the topic changes; {why}, so no chapters are written",
                   confidence=1.0, evidence=req_ids + ([tr.id] if tr is not None else []), risk="LOW", approval="AUTO", provenance="USER",
                   params={**base, "asset_id": asset.id}, requirements=reqs)
        return req
    ev = [i.id for i in recs] + req_ids + ([tr.id] if tr is not None else [])
    why = None
    derived: Dict[str, Any] = {}
    if len(recs) > 1:
        why = f"{len(recs)} transcript_chapters recommendations for {asset.id}: which one is meant is not guessed"
    elif tr is None:
        why = f"no transcript of {asset.id}: segment numbers cannot be read"
    else:
        segs = numbered_segments((tr.data or {}).get("segments") or [])
        try:
            if len(segs) > MAX_CONTEXT_SEGMENTS and any(isinstance(c, dict) and isinstance(c.get("segment"), int) and c["segment"] >= MAX_CONTEXT_SEGMENTS
                                                        for c in ((recs[0].data or {}).get("params") or {}).get("chapters") or []):
                raise ChapterError(f"the AI was shown only segments 0..{MAX_CONTEXT_SEGMENTS - 1}: a chapter past them names a segment it did not see")
            duration = (asset.technical or {}).get("duration")
            derived = derive_chapters((recs[0].data or {}).get("params"), segs, duration if duration is not None else (tr.data or {}).get("duration"), req["min_seconds"])
        except ChapterError as e:
            why = str(e)
    if why is not None:
        eng.decide(subject=SUBJECT, type="KEEP", decision=f"no chapters for {asset.id}: the AI's chapter list was refused", reason=f"refused ({RULE_VERSION}): {why}; a chapter list is never corrected",
                   confidence=1.0, evidence=ev, risk="LOW", approval="AUTO", provenance="AI_GENERATED", requirements=reqs,
                   params=_scrub({**base, "asset_id": asset.id, "refused": why, "ai_params": (recs[0].data or {}).get("params") if len(recs) == 1 else None,
                                  **({"transcript_id": tr.id} if tr is not None else {})}))
        return req
    inf = recs[0]
    quoted = "; ".join(f"{timestamp(c['start'])} {c['title']} (“{c['first_words']}”)" for c in derived["chapters"])
    eng.decide(subject=SUBJECT, type="TRANSFORM", decision=f"write {len(derived['chapters'])} chapters into {asset.id}'s deliverable: {quoted}",
               reason=f"{inf.statement}; the AI named the segments where the topic changes ({inf.data.get('provider')}/{inf.data.get('model')}), the agent took their starts from the "
                      f"transcript with {RULE_VERSION} (chapter 1 at 0:00, each at least {req['min_seconds']:g}s); chapters an AI chose are confirmed before they are written",
               confidence=inf.confidence, evidence=ev, risk="LOW", approval=approval_for(SUBJECT, floor="CONFIRM", keys={SUBJECT: APPROVAL_KEY}), provenance="AI_GENERATED",
               params={**base, "asset_id": asset.id, "transcript_id": tr.id, "inference_id": inf.id, **derived}, requirements=reqs)
    return req


# ---- validation ------------------------------------------------------------------------------------------------------------
def check_chapters(doc: Dict[str, Any]) -> List[str]:
    """`delivery.chapters`: a planned subject and preset targets, chapters from 0 in increasing order, each at least min_seconds on the programme,
    titles a chapter file can carry, and an approved-or-pending chapters decision cited."""
    rec = (doc.get("delivery") or {}).get("chapters")
    if not rec:
        return []
    errs: List[str] = []
    targets = {t["id"]: t for t in (doc.get("delivery") or {}).get("targets") or []}
    for t in rec.get("targets") or []:
        if t not in targets or not targets[t].get("preset"):
            errs.append(f"delivery.chapters target {t!r} is not a preset delivery target of this plan")
    chs = rec.get("chapters") or []
    total = float(rec.get("programme_seconds") or 0.0)
    min_s = float(rec.get("min_seconds") or MIN_DEFAULT)
    if len(chs) < 2:
        errs.append("delivery.chapters holds fewer than two chapters")
    if chs and float(chs[0].get("start", -1)) != 0.0:
        errs.append("delivery.chapters: chapter 1 must start at 0")
    for n, c in enumerate(chs):
        end = float(chs[n + 1]["start"]) if n + 1 < len(chs) else total
        if end - float(c["start"]) < min_s - 1e-6:
            errs.append(f"delivery.chapters: chapter {n + 1} lasts {end - float(c['start']):.3f}s, under {min_s:g}s")
        try:
            _title(c.get("title"), n)
        except ChapterError as e:
            errs.append(f"delivery.chapters: {e}")
    decs = {d["id"]: d for d in doc.get("decisions") or []}
    cited = [decs.get(i) for i in rec.get("decision_ids") or []]
    if not cited or any(d is None or d.get("subject") != SUBJECT or d.get("type") != "TRANSFORM" for d in cited):
        errs.append(f"delivery.chapters must cite its {SUBJECT} decision")
    return errs
