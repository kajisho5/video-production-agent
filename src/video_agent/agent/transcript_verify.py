"""Post-render transcript check (ADR-110): listen to the delivered file again and compare what is heard with the words the plan kept.

    plan: `verify.transcript=true` → IR `qa.transcript_verify` (language / engine / model / unit / min_confidence, rule)
    render: QA → the registry-selected `speech_transcription` tool re-transcribes each deliverable (word timestamps)
          → heard words aligned to the planned words (the source transcript's words the trims / concat / speed keep, mapped onto
            the delivered timeline) → a "listen and check here" list in the report: missing, duplicated, unexpected, low confidence

Everything here is arithmetic over recorded documents, no execution and no AI. The check never fails a job, never changes a QA
status or an artifact stage and never edits anything: a person listens at the times it names and decides.

- **Planned words** (`planned_words`): for each source of a deliverable, the numbered words of its transcript (ADR-109 numbering)
  whose start and end both lie in one kept range of that source (the IR's `video.trim` keep, or the complement of `audio.cut`'s
  remove), mapped like the subtitle cues (trim → concat offset → speed; a still joined before the picture shifts everything by its
  length). Each keeps its source asset, index and source times. A word the plan cut is not planned.
- **Normalisation** (`normalise`): Unicode NFKC, then case folding, then every character of a punctuation (P*), separator (Z*) or
  control / format (C*) category removed. Words that are empty after it ("—", "、") are left out on both sides. Unit `word` (the
  default, any language): one token per recognised word. Unit `char` (opt-in, `verify.transcript.unit=char`, meant for scripts
  without spaces such as Japanese, where two recognitions split the same speech into different words): one token per character of
  the normalised word, each with its word's times.
- **Alignment**: `difflib.SequenceMatcher` over the two token lists (autojunk off; deterministic). Each non-equal opcode is one
  item: planned tokens that were not heard → `missing` (with the planned word indices, delivered and source times); heard tokens
  that were not planned → `duplicated` when every token of the chunk is heard more often in the whole file than the plan has it,
  else `unexpected` (delivered times, source times where the delivered time maps back into a kept range). Heard words below
  `min_confidence` (default 0.5) form `low_confidence` spans (consecutive ones merged). Items are capped at MAX_ITEMS per
  deliverable; the counts are always complete.
- **Not run** (a warning and a provenance row, never a crash): no tool for `speech_transcription` here, the Skill failing or timing
  out, a deliverable without audio, a source without a transcript with word timings (nothing planned to compare with).
"""
from __future__ import annotations

import difflib
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .word_cut import WordRangeError, number_words

RULE_VERSION = "transcript_verify@1"
PREFIX = "verify."
KEY = "verify.transcript"
LANGUAGE_KEY = "verify.transcript.language"
ENGINE_KEY = "verify.transcript.engine"
MODEL_KEY = "verify.transcript.model"
MIN_CONFIDENCE_KEY = "verify.transcript.min_confidence"
UNIT_KEY = "verify.transcript.unit"
KEYS = (KEY, LANGUAGE_KEY, ENGINE_KEY, MODEL_KEY, MIN_CONFIDENCE_KEY, UNIT_KEY)
UNITS = ("word", "char")
MIN_CONFIDENCE_DEFAULT = 0.5
MAX_ITEMS = 200
SKILL = "speech_transcription"
KINDS = ("missing", "duplicated", "unexpected", "low_confidence")


def _true(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "yes", "on", "1"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "no", "off", "0"):
        return False
    return None


def parse_verify_requirements(reqs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """`verify.*` --set keys → the IR's `qa.transcript_verify` (None when the check is not asked for). Unknown keys, a refinement
    without the switch, a language that is not an ISO 639 code, an engine / model that is not a plain name, a confidence outside 0..1
    and an unknown unit are refused (ValueError)."""
    keys = [k for k in (reqs or {}) if k.startswith(PREFIX)]
    unknown = sorted(k for k in keys if k not in KEYS)
    if unknown:
        raise ValueError(f"unknown requirement key(s): {', '.join(unknown)} (verify.* keys: {', '.join(KEYS)})")
    if not keys:
        return None
    sw = _true(reqs.get(KEY)) if KEY in reqs else None
    if KEY in reqs and sw is None:
        raise ValueError(f"{KEY} must be true or false, got {reqs[KEY]!r}")
    refinements = [k for k in keys if k != KEY]
    if not sw:
        if refinements:
            raise ValueError(f"{refinements[0]} is set but {KEY} is not: refusing an ambiguous request")
        return None
    out: Dict[str, Any] = {"enabled": True, "rule": RULE_VERSION, "unit": "word", "min_confidence": MIN_CONFIDENCE_DEFAULT}
    if LANGUAGE_KEY in reqs:
        v = str(reqs[LANGUAGE_KEY]).strip().lower()
        if not (2 <= len(v) <= 3 and v.isascii() and v.isalpha()):
            raise ValueError(f"{LANGUAGE_KEY} must be an ISO 639 language code (en, ja, ...), got {reqs[LANGUAGE_KEY]!r}")
        out["language"] = v
    for key, name in ((ENGINE_KEY, "engine"), (MODEL_KEY, "model")):
        if key in reqs:
            v = str(reqs[key]).strip()
            if not v or len(v) > 64 or not all(c.isalnum() or c in "._-" for c in v):
                raise ValueError(f"{key} must be a plain {name} name, got {reqs[key]!r}")
            out[name] = v
    if MIN_CONFIDENCE_KEY in reqs:
        try:
            c = float(reqs[MIN_CONFIDENCE_KEY])
        except (TypeError, ValueError):
            raise ValueError(f"{MIN_CONFIDENCE_KEY} must be a number in 0..1, got {reqs[MIN_CONFIDENCE_KEY]!r}")
        if isinstance(reqs[MIN_CONFIDENCE_KEY], bool) or not 0.0 <= c <= 1.0:
            raise ValueError(f"{MIN_CONFIDENCE_KEY} must be a number in 0..1, got {reqs[MIN_CONFIDENCE_KEY]!r}")
        out["min_confidence"] = c
    if UNIT_KEY in reqs:
        u = str(reqs[UNIT_KEY]).strip().lower()
        if u not in UNITS:
            raise ValueError(f"{UNIT_KEY} must be one of {' | '.join(UNITS)}, got {reqs[UNIT_KEY]!r}")
        out["unit"] = u
    return out


# ---- text ------------------------------------------------------------------------------------------------------------------
def normalise(text: Any) -> str:
    """NFKC → case fold → drop punctuation (P*), separators / spaces (Z*) and control / format characters (C*)."""
    t = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(c for c in t if not unicodedata.category(c).startswith(("P", "Z", "C")))


def tokens(words: Sequence[Dict[str, Any]], unit: str = "word") -> List[Dict[str, Any]]:
    """Words → alignment tokens {"norm", "w"} (w: the word the token came from). Empty words are left out."""
    out: List[Dict[str, Any]] = []
    for w in words:
        n = normalise(w.get("text"))
        if not n:
            continue
        if unit == "char":
            out += [{"norm": c, "w": w} for c in n]
        else:
            out.append({"norm": n, "w": w})
    return out


# ---- timelines ---------------------------------------------------------------------------------------------------------------
def _keep_of(doc: Dict[str, Any], asset_id: str, duration: float) -> List[List[float]]:
    v = next((o for o in (doc.get("video") or {}).get("operations") or [] if o.get("type") == "video.trim" and o.get("asset") == asset_id), None)
    if v is not None:
        return [[float(s), float(e)] for s, e in v["keep"]]
    a = next((o for o in (doc.get("audio") or {}).get("operations") or [] if o.get("type") == "audio.cut" and o.get("asset") == asset_id), None)
    if a is not None:
        rem = sorted([float(s), float(e)] for s, e in (a.get("remove") or (a.get("params") or {}).get("remove") or []))
        keep, t = [], 0.0
        for s, e in rem:
            if s > t:
                keep.append([t, s])
            t = max(t, e)
        if t < duration:
            keep.append([t, duration])
        return keep
    return [[0.0, float(duration)]]


def timeline_of(doc: Dict[str, Any], subject: Dict[str, Any]) -> Dict[str, Any]:
    """How a deliverable's timeline is made from its sources: {"speed", "lead", "inputs": {asset: {"keep", "offset"}}, "notes"}."""
    ops = (doc.get("video") or {}).get("operations") or []
    aops = (doc.get("audio") or {}).get("operations") or []
    sid = subject["id"]
    speed = next((float(op["factor"]) for op in ops if op.get("asset") == sid and op.get("type") == "video.speed" and op.get("factor")), 1.0)
    concat = next((op for op in ops + aops if op.get("type") in ("video.concat", "audio.concat") and op.get("output") == sid), None)
    lead = sum(float(op.get("duration") or 0.0) for op in ops if op.get("type") == "video.image_insert" and op.get("asset") == sid and op.get("position") == "before")
    notes: List[str] = []
    if any(op.get("type") == "audio.filler_remove" for op in aops):
        notes.append("filler words were removed by audio-production (FILLER_REMOVE): the words it matched are expected to be missing")
    if any(op.get("type") in ("video.switch", "video.grid") and op.get("output") == sid for op in ops) or any(op.get("type") == "audio.mix" and op.get("asset") == sid for op in aops):
        notes.append("a camera switch / grid / mix programme: the planned words follow the first source only")
    assets = doc.get("assets") or {}
    inputs: Dict[str, Dict[str, Any]] = {}
    acc = 0.0
    for src in subject.get("sources") or []:
        dur = float(((assets.get(src) or {}).get("technical") or {}).get("duration") or 0.0)
        keep = _keep_of(doc, src, dur)
        offset = acc
        if concat is not None:
            seg = next((s for s in concat.get("segments") or [] if s.get("input") == src), None)
            if seg and seg.get("timeline_range"):
                offset = float(seg["timeline_range"][0])
        inputs[src] = {"keep": keep, "offset": offset}
        acc = offset + sum(e - s for s, e in keep)
        if concat is None:
            break
    return {"speed": speed, "lead": lead, "inputs": inputs, "notes": notes}


def _to_output(t: float, keep: List[List[float]], offset: float, speed: float, lead: float) -> Optional[float]:
    acc = 0.0
    for s, e in keep:
        if s - 1e-9 <= t <= e + 1e-9:
            return round(lead + (offset + acc + (t - s)) / speed, 3)
        acc += e - s
    return None


def to_source(t_out: float, tl: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A delivered time back onto a source (asset, seconds, kept range), or None (inside a still, past the end, between inputs)."""
    x = (float(t_out) - tl["lead"]) * tl["speed"]
    for src, m in tl["inputs"].items():
        acc = m["offset"]
        for n, (s, e) in enumerate(m["keep"]):
            if acc - 1e-9 <= x <= acc + (e - s) + 1e-9:
                return {"asset_id": src, "seconds": round(s + (x - acc), 3), "range": n}
            acc += e - s
    return None


def source_span(span: Dict[str, float], tl: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A delivered span back onto one source, only when both ends lie in the same kept range (a span across a cut is not one place)."""
    if tl is None:
        return None
    a, b = to_source(span["start"], tl), to_source(span["end"], tl)
    if not a or not b or a["asset_id"] != b["asset_id"] or a["range"] != b["range"]:
        return None
    return {"asset_id": a["asset_id"], "start": a["seconds"], "end": b["seconds"]}


def planned_words(doc: Dict[str, Any], subject: Dict[str, Any], observations: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """The words the plan keeps for one deliverable, on its timeline (sorted) → (words, timeline, problems)."""
    tl = timeline_of(doc, subject)
    out: List[Dict[str, Any]] = []
    problems: List[str] = []
    for src, m in tl["inputs"].items():
        tr = next((o for o in observations if o.get("kind") == "transcript" and o.get("asset_id") == src
                   and any(isinstance(s, dict) and isinstance(s.get("words"), list) for s in (o.get("data") or {}).get("segments") or [])), None)
        if tr is None:
            problems.append(f"{src}: no transcript with word timings in the plan, so there is nothing planned to compare with")
            continue
        try:
            words, _ = number_words((tr.get("data") or {}).get("segments") or [])
        except WordRangeError as e:
            problems.append(f"{src}: {e}")
            continue
        for w in words:
            if w["start"] is None or w["end"] is None:
                continue
            os_ = _to_output(w["start"], m["keep"], m["offset"], tl["speed"], tl["lead"])
            oe = _to_output(w["end"], m["keep"], m["offset"], tl["speed"], tl["lead"])
            inside = any(s - 1e-9 <= w["start"] and w["end"] <= e + 1e-9 for s, e in m["keep"])
            if os_ is None or oe is None or not inside:
                continue
            out.append({"text": w["text"], "start": os_, "end": oe, "source": {"asset_id": src, "index": w["index"], "start": w["start"], "end": w["end"]},
                        "transcript_id": tr.get("id")})
    out.sort(key=lambda w: (w["start"], w["end"]))
    return out, tl, problems


def heard_words(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The re-transcription's timed words, numbered by their order (the Skill's own `index` is not needed here and never fails the check)."""
    segs = [dict(s, words=[{k: v for k, v in w.items() if k != "index"} for w in s["words"] if isinstance(w, dict)]) if isinstance(s, dict) and isinstance(s.get("words"), list) else s
            for s in (transcript or {}).get("segments") or []]
    words, _ = number_words(segs)
    return [w for w in words if w["start"] is not None and w["end"] is not None]


# ---- comparison --------------------------------------------------------------------------------------------------------------
def _span(toks: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    return {"start": min(t["w"]["start"] for t in toks), "end": max(t["w"]["end"] for t in toks)}


def _text(toks: Sequence[Dict[str, Any]], unit: str) -> str:
    seen: List[Any] = []
    for t in toks:
        if not seen or seen[-1] is not t["w"]:
            seen.append(t["w"])
    return ("" if unit == "char" else " ").join(str(w.get("text") or "").strip() for w in seen)


def compare(planned: Sequence[Dict[str, Any]], heard: Sequence[Dict[str, Any]], unit: str = "word", min_confidence: float = MIN_CONFIDENCE_DEFAULT,
            timeline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Align the heard words to the planned words → {"counts", "items", "matched", "planned_tokens", "heard_tokens"}."""
    p, h = tokens(planned, unit), tokens(heard, unit)
    pc, hc = Counter(t["norm"] for t in p), Counter(t["norm"] for t in h)
    sm = difflib.SequenceMatcher(None, [t["norm"] for t in p], [t["norm"] for t in h], autojunk=False)
    items: List[Dict[str, Any]] = []
    matched = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            matched += i2 - i1
            continue
        if i2 > i1:   # planned, not heard
            toks = p[i1:i2]
            srcs = [t["w"]["source"] for t in toks]
            items.append({"kind": "missing", "text": _text(toks, unit), "output": _span(toks),
                          "source": {"asset_id": srcs[0]["asset_id"], "start": srcs[0]["start"], "end": srcs[-1]["end"],
                                     "word_indices": sorted({s["index"] for s in srcs})} if all(s["asset_id"] == srcs[0]["asset_id"] for s in srcs) else None})
        if j2 > j1:   # heard, not planned
            toks = h[j1:j2]
            dup = all(pc[t["norm"]] > 0 and hc[t["norm"]] > pc[t["norm"]] for t in toks)
            span = _span(toks)
            src = source_span(span, timeline)
            items.append({"kind": "duplicated" if dup else "unexpected", "text": _text(toks, unit), "output": span, "source": src,
                          "heard_word_indices": sorted({t["w"]["index"] for t in toks})})
    low: List[List[Dict[str, Any]]] = []
    for w in heard:
        c = w.get("confidence")
        if c is not None and c < min_confidence and normalise(w.get("text")):
            if low and low[-1][-1]["index"] == w["index"] - 1:
                low[-1].append(w)
            else:
                low.append([w])
    for run in low:
        span = {"start": run[0]["start"], "end": run[-1]["end"]}
        src = source_span(span, timeline)
        items.append({"kind": "low_confidence", "text": " ".join(str(w.get("text") or "").strip() for w in run), "output": span, "source": src,
                      "confidence": round(min(float(w["confidence"]) for w in run), 3), "heard_word_indices": [w["index"] for w in run]})
    items.sort(key=lambda it: (it["output"]["start"], KINDS.index(it["kind"])))
    counts = {k: sum(1 for it in items if it["kind"] == k) for k in KINDS}
    return {"counts": counts, "items": items, "matched": matched, "planned_tokens": len(p), "heard_tokens": len(h)}


def verify_deliverable(doc: Dict[str, Any], subject: Dict[str, Any], transcript: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """One deliverable: planned words vs the re-transcription → the report entry (status CHECK_HERE / CLEAN / NOT_COMPARED)."""
    planned, tl, problems = planned_words(doc, subject, (doc.get("analysis") or {}).get("observations") or [])
    heard = heard_words(transcript)
    rep: Dict[str, Any] = {"subject": subject["id"], "rule": RULE_VERSION, "unit": cfg.get("unit", "word"), "min_confidence": cfg.get("min_confidence", MIN_CONFIDENCE_DEFAULT),
                           "language": transcript.get("language"), "engine": transcript.get("engine"), "engine_version": transcript.get("engine_version"),
                           "model": (transcript.get("provenance") or {}).get("model"), "transcript_id": transcript.get("id"),
                           "planned_words": len(planned), "heard_words": len(heard), "notes": list(tl["notes"]) + problems}
    planned_langs = sorted({str((o.get("data") or {}).get("language")) for o in (doc.get("analysis") or {}).get("observations") or []
                            if o.get("kind") == "transcript" and o.get("asset_id") in tl["inputs"] and (o.get("data") or {}).get("language")})
    if transcript.get("language") and planned_langs and transcript.get("language") not in planned_langs:
        rep["notes"].append(f"the delivered file was recognised as {transcript.get('language')!r} but the plan's transcript is {', '.join(planned_langs)}: most words will differ")
    if not planned and problems:
        rep.update(status="NOT_COMPARED", counts={k: 0 for k in KINDS}, items=[])
        return rep
    if not heard and any(isinstance(s, dict) and str(s.get("text") or "").strip() for s in transcript.get("segments") or []):
        rep["notes"].append("the delivered file's speech was recognised without word timings, so nothing could be compared word by word")
        rep.update(status="NOT_COMPARED", counts={k: 0 for k in KINDS}, items=[])
        return rep
    res = compare(planned, heard, rep["unit"], float(rep["min_confidence"]), tl)
    items = res["items"]
    # a source that could not be compared is a place to check too
    rep.update(status="CHECK_HERE" if items or problems else "CLEAN", counts=res["counts"], matched_tokens=res["matched"], planned_tokens=res["planned_tokens"],
               heard_tokens=res["heard_tokens"], items=items[:MAX_ITEMS])
    if len(items) > MAX_ITEMS:
        rep["truncated"] = {"shown": MAX_ITEMS, "total": len(items)}
    return rep


def summary_line(rep: Dict[str, Any]) -> str:
    c = rep.get("counts") or {}
    return (f"{rep.get('subject')}: {rep.get('status')} — {c.get('missing', 0)} missing, {c.get('duplicated', 0)} duplicated, {c.get('unexpected', 0)} unexpected, "
            f"{c.get('low_confidence', 0)} low-confidence span(s)")
