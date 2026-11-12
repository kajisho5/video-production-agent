"""Word-index plans (ADR-109): the AI names the words to keep or cut by their number in the transcript, never by a time.

    transcript observation (word timings) → numbered words (0-based, the whole transcript, time order)
      → AI recommendation (intent `word_cut`, params keep_words | cut_words = [[first_index, last_index], ...])
      → `words.cut` decision (CONFIRM floor) with the word ranges, the seconds derived from them and the rule version
      → removed ranges of the asset's trim step → Project IR → compiler

Everything here is vocabulary and arithmetic over recorded observations, no execution and no AI call:

- **Numbering** (`number_words`): every word of every segment that carries words, in document order (segments in order, words in
  order within a segment; `check_transcript` refuses out-of-order segments, so this is time order). transcription-skill may write
  each word's own `index`: when every word has one and they are exactly 0..N-1, that numbering is used (ordered by it); when no
  word has one the same numbering is derived from the order; a mix, a duplicate or a gap is refused (the indices the AI read would
  not be the ones the agent converts). Extra word fields (`pos`, an interpolated-timing flag) are carried, never required.
- **Conversion** (`derive_ranges`, rule `word_ranges@1`): the kept words are the union of `keep_words`, or every word not in
  `cut_words`; consecutive kept indices form runs. A run [a, b] keeps `start(a) - pre_roll` .. `end(b) + post_roll`, clamped
  (1) at the midpoint of the gap to a neighbouring word that is not kept (where the two words overlap in time: at the kept
  word's own boundary), so a kept range never takes in a word that was not kept, and (2) to the media, 0 .. duration. Seconds
  are rounded to the millisecond. `keep_words`: everything outside the kept ranges is removed (lead-in and tail included).
  `cut_words`: only the gaps around the cut words are removed (a cut that starts at word 0 removes from 0 s, one that ends at the
  last word removes to the end); the lead-in and the tail stay otherwise.
- **Refusals** (`WordRangeError`, never corrected): a range that is not two whole numbers, out of 0..N-1, reversed or overlapping
  another, a word at a range boundary (or its neighbour) without times, a word that ends after the media, nothing left to keep,
  and derived ranges that overlap (word timings that run into each other). An AI recommendation that is refused becomes a KEEP
  decision carrying the reason: nothing is cut, the plan still runs, and the refusal is visible in the IR.
- **Determinism**: the decision records the word ranges, the derived keep / remove seconds, pre / post roll with their provenance,
  the duration, the numbering source and `rule`; `check_word_cuts` (run by `validate_plan`) derives the seconds again from the
  recorded transcript and refuses an IR whose recorded seconds differ, so the same plan always produces the same cut list.
  Revisions reuse the recorded AI inference (no AI call) and derive the same seconds unless the user changed the roll.

Defaults: pre_roll 0.10 s, post_roll 0.20 s (chosen here, not taken from a published source: a recogniser's word start sits on the
onset, so a short lead keeps the consonant's attack; its word end tends to sit where the voiced part stops, before the release
and the room's decay, so the tail gets twice the lead). Both are clamped by the neighbour rule above, so a long roll can never
reach into a word that is cut.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..media.analysis import _scrub
from ..models import Requirement
from .editing import EditRequirementError, _bool, _number

RULE_VERSION = "word_ranges@1"
KEY = "edit.word_cut"
PRE_ROLL_KEY = "edit.word_cut.pre_roll"
POST_ROLL_KEY = "edit.word_cut.post_roll"
REQUIREMENT_KEYS = (KEY, PRE_ROLL_KEY, POST_ROLL_KEY)
PRE_ROLL_DEFAULT = 0.10
POST_ROLL_DEFAULT = 0.20
ROLL_RANGE = (0.0, 1.0)
SUBJECT = "words.cut"
AI_INTENT = "word_cut"
# removing recognised speech the AI chose is a semantic deletion: the profile's policy for it (conference: BLOCK), CONFIRM at least
APPROVAL_KEY = ("edit.semantic_deletion.approval", "CONFIRM")
RISK = "HIGH"
MODES = {"keep_words": "keep", "cut_words": "cut"}
# word fields transcription-skill may add that mark a timing it interpolated rather than measured (carried as `interpolated`)
INTERPOLATED_KEYS = ("interpolated",)
MAX_CONTEXT_WORDS = 20000   # the most words one asset's index sent to the AI may hold (a longer transcript is sent cut off, and said so)


class WordRangeError(ValueError):
    """Word ranges (or the transcript under them) that cannot be converted: refused, never corrected."""


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float("inf"), float("-inf")) else None


# ---- requirements ------------------------------------------------------------------------------------------------------
def parse_word_cut_requirements(m: Dict[str, Requirement]) -> Optional[Dict[str, Any]]:
    """→ {"pre_roll", "post_roll", "pre_roll_provenance", "post_roll_provenance", "requirements"} when `edit.word_cut` is on, else None.
    A refinement without the switch, or a roll outside 0..1 s, is refused (EditRequirementError)."""
    main = m.get(KEY)
    refinements = [m[k] for k in REQUIREMENT_KEYS[1:] if k in m and m[k].provenance != "DEFAULT"]
    if main is None or main.provenance == "DEFAULT" or not _bool(main.value, KEY):
        if refinements:
            raise EditRequirementError(f"{refinements[0].key} is set but {KEY} is not: refusing an ambiguous request")
        return None
    vals = {r.key: r for r in refinements}
    out: Dict[str, Any] = {"pre_roll": PRE_ROLL_DEFAULT, "post_roll": POST_ROLL_DEFAULT, "pre_roll_provenance": "DEFAULT", "post_roll_provenance": "DEFAULT",
                           "requirements": [main] + refinements}
    for key, name in ((PRE_ROLL_KEY, "pre_roll"), (POST_ROLL_KEY, "post_roll")):
        if key in vals:
            out[name] = float(_number(vals[key].value, key, *ROLL_RANGE))
            out[f"{name}_provenance"] = vals[key].provenance
    return out


def requested(user_requirements: Dict[str, Any]) -> bool:
    v = (user_requirements or {}).get(KEY)
    return v is not None and v is not False and not (isinstance(v, str) and v.strip().lower() in ("false", "no", "off", "0"))


# ---- numbering ---------------------------------------------------------------------------------------------------------
def number_words(segments: Sequence[Any]) -> Tuple[List[Dict[str, Any]], str]:
    """The words of a transcript, numbered → (words, index_source "transcript" | "derived"). Each word: index, start, end (None when the
    transcript gives no number), text, segment_id, and confidence / pos / interpolated when present. A segment whose `words` is null
    has no numbered words (no word timestamps). Raises WordRangeError for word indices that are not exactly 0..N-1 on every word."""
    words: List[Dict[str, Any]] = []
    given: List[Any] = []
    for si, seg in enumerate(segments or []):
        if not isinstance(seg, dict) or not isinstance(seg.get("words"), list):
            continue
        for w in seg["words"]:
            if not isinstance(w, dict):
                raise WordRangeError(f"segment {seg.get('id') or si}: a word that is not an object")
            rec: Dict[str, Any] = {"index": len(words), "start": _num(w.get("start")), "end": _num(w.get("end")), "text": str(w.get("text") or ""),
                                   "segment_id": str(seg.get("id") or si)}
            c = _num(w.get("confidence"))
            if c is not None and 0.0 <= c <= 1.0:
                rec["confidence"] = c
            if isinstance(w.get("pos"), str):
                rec["pos"] = w["pos"]
            if any(w.get(k) is True for k in INTERPOLATED_KEYS):
                rec["interpolated"] = True
            words.append(rec)
            given.append(w.get("index"))
    have = [isinstance(g, int) and not isinstance(g, bool) for g in given]
    if not any(have):
        return words, "derived"
    if not all(have) or sorted(given) != list(range(len(words))):
        raise WordRangeError("the transcript's word indices are not exactly 0..N-1 on every word (missing on some words, duplicated or with a gap)")
    by_index = sorted(zip(given, words), key=lambda p: p[0])
    return [dict(w, index=i) for i, w in by_index], "transcript"


def word_index_of(observation_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    return number_words((observation_data or {}).get("segments") or [])


# ---- conversion --------------------------------------------------------------------------------------------------------
def check_word_ranges(ranges: Any, count: int, name: str = "word range") -> List[List[int]]:
    """[[first, last], ...] → the same ranges sorted by first index. Refused: not a non-empty list of two whole numbers each, out of
    0..count-1, reversed, overlapping another range."""
    if not isinstance(ranges, list) or not ranges:
        raise WordRangeError(f"{name}s must be a non-empty list of [first_index, last_index]")
    out: List[List[int]] = []
    for r in ranges:
        if not (isinstance(r, (list, tuple)) and len(r) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in r)):
            raise WordRangeError(f"{name} {r!r} is not [first_index, last_index] of two whole numbers (word indices, never seconds)")
        a, b = int(r[0]), int(r[1])
        if a > b:
            raise WordRangeError(f"{name} [{a}, {b}] is reversed (first index after last)")
        if a < 0 or b >= count:
            raise WordRangeError(f"{name} [{a}, {b}] is out of bounds: the transcript has words 0..{count - 1}" if count else f"{name} [{a}, {b}]: the transcript has no words")
        out.append([a, b])
    out.sort()
    for p, q in zip(out, out[1:]):
        if q[0] <= p[1]:
            raise WordRangeError(f"{name}s [{p[0]}, {p[1]}] and [{q[0]}, {q[1]}] overlap")
    return out


def _runs(indices: Sequence[int]) -> List[List[int]]:
    runs: List[List[int]] = []
    for i in indices:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return runs


def _timed(w: Dict[str, Any], why: str) -> Tuple[float, float]:
    s, e = w.get("start"), w.get("end")
    if s is None or e is None:
        raise WordRangeError(f"word {w['index']} ({w.get('text')!r}) has no start / end time, and {why}")
    if s < 0 or e < s:
        raise WordRangeError(f"word {w['index']} ({w.get('text')!r}) has invalid times {s}..{e}")
    return s, e


def derive_ranges(words: List[Dict[str, Any]], word_ranges: Any, mode: str, duration: float, pre_roll: float = PRE_ROLL_DEFAULT,
                  post_roll: float = POST_ROLL_DEFAULT) -> Dict[str, Any]:
    """Word ranges → time ranges on the source (rule word_ranges@1, see the module doc). Returns {rule, mode, word_ranges (sorted),
    word_count, kept_words (runs), keep, remove, words_removed, pre_roll, post_roll, duration, interpolated_boundaries}."""
    if mode not in ("keep", "cut"):
        raise WordRangeError(f"mode must be keep or cut, got {mode!r}")
    dur = _num(duration)
    if dur is None or dur <= 0:
        raise WordRangeError(f"the media duration is unknown ({duration!r}): word ranges are never converted without it")
    for name, v in (("pre_roll", pre_roll), ("post_roll", post_roll)):
        if _num(v) is None or not ROLL_RANGE[0] <= float(v) <= ROLL_RANGE[1]:
            raise WordRangeError(f"{name} must be within {ROLL_RANGE[0]}..{ROLL_RANGE[1]} s, got {v!r}")
    n = len(words)
    ranges = check_word_ranges(word_ranges, n, "keep range" if mode == "keep" else "cut range")
    chosen = {i for a, b in ranges for i in range(a, b + 1)}
    kept = sorted(chosen) if mode == "keep" else [i for i in range(n) if i not in chosen]
    if not kept:
        raise WordRangeError("the word ranges keep no word: nothing would be left")
    runs = _runs(kept)
    keep: List[List[float]] = []
    interpolated: List[int] = []
    for a, b in runs:
        sa, _ = _timed(words[a], f"it starts kept range [{a}, {b}]")
        _, eb = _timed(words[b], f"it ends kept range [{a}, {b}]")
        start, end = sa - float(pre_roll), eb + float(post_roll)
        touched = [a, b]
        if a > 0:
            _, pe = _timed(words[a - 1], f"it is the word before kept range [{a}, {b}]")
            start = max(start, (pe + sa) / 2.0 if pe <= sa else sa)
            touched.append(a - 1)
        if b < n - 1:
            ns, _ = _timed(words[b + 1], f"it is the word after kept range [{a}, {b}]")
            end = min(end, (eb + ns) / 2.0 if eb <= ns else eb)
            touched.append(b + 1)
        if eb > dur + 1e-6:
            raise WordRangeError(f"word {b} ends at {eb}s, after the media ({dur}s)")
        start, end = round(max(0.0, start), 3), round(min(dur, end), 3)
        if not end > start:
            raise WordRangeError(f"kept range [{a}, {b}] has no length ({start}..{end}s)")
        keep.append([start, end])
        interpolated += [i for i in touched if words[i].get("interpolated")]
    for p, q in zip(keep, keep[1:]):
        if q[0] < p[1]:
            raise WordRangeError(f"the derived ranges {p} and {q} overlap: the word timings around them run into each other")
    d3 = round(dur, 3)
    if mode == "keep":
        edges = [0.0] + [x for r in keep for x in r] + [d3]
        remove = [[edges[i], edges[i + 1]] for i in range(0, len(edges), 2)]
    else:
        remove = [[keep[i][1], keep[i + 1][0]] for i in range(len(keep) - 1)]
        if runs[0][0] > 0:
            remove.insert(0, [0.0, keep[0][0]])
        if runs[-1][1] < n - 1:
            remove.append([keep[-1][1], d3])
    remove = [[round(s, 3), round(e, 3)] for s, e in remove if e - s > 1e-9]
    if mode == "cut":   # what stays is the media less the removed gaps (the lead-in and tail too, unless a cut reaches them)
        edges = [0.0] + [x for r in remove for x in r] + [d3]
        keep = [[edges[i], edges[i + 1]] for i in range(0, len(edges), 2) if edges[i + 1] - edges[i] > 1e-9]
    return {"rule": RULE_VERSION, "mode": mode, "word_ranges": ranges, "word_count": n, "kept_words": runs, "keep": keep, "remove": remove,
            "words_removed": n - len(kept), "pre_roll": float(pre_roll), "post_roll": float(post_roll), "duration": d3,
            "interpolated_boundaries": sorted(set(interpolated))}


def recommendation_ranges(params: Any) -> Tuple[str, Any]:
    """An AI recommendation's params → (mode, raw ranges). Exactly one of keep_words / cut_words and nothing else (no seconds)."""
    if not isinstance(params, dict):
        raise WordRangeError("params must be an object with keep_words or cut_words")
    given = [k for k in MODES if k in params]
    other = sorted(k for k in params if k not in MODES)
    if other:
        raise WordRangeError(f"params carry {', '.join(other)}: a word_cut recommendation names word ranges only (keep_words or cut_words), never times")
    if len(given) != 1:
        raise WordRangeError("params must have exactly one of keep_words / cut_words")
    return MODES[given[0]], params[given[0]]


# ---- AI context ----------------------------------------------------------------------------------------------------------
def ai_context(analysis: Any) -> Dict[str, Any]:
    """What the AI reads to name word ranges: per transcript with word timings, its numbered words as [index, text] (no times)."""
    out: List[Dict[str, Any]] = []
    for o in getattr(analysis, "observations", []) or []:
        if o.kind != "transcript":
            continue
        try:
            words, src = word_index_of(o.data or {})
        except WordRangeError:
            continue
        if not words:
            continue
        rec = {"asset_id": o.asset_id, "transcript_id": o.id, "word_count": len(words), "index_source": src,
               "words": [[w["index"], w["text"]] for w in words[:MAX_CONTEXT_WORDS]]}
        if len(words) > MAX_CONTEXT_WORDS:
            rec["truncated_after"] = MAX_CONTEXT_WORDS
        out.append(rec)
    return {"word_cut": {"intent": AI_INTENT, "rule": RULE_VERSION, "transcripts": out,
                         "params": {"keep_words": [["<first index>", "<last index>"]], "cut_words": [["<first index>", "<last index>"]]},
                         "instructions": ("For intent word_cut give params with exactly one of keep_words (the words that stay) or cut_words (the words that go): "
                                          "inclusive [first_index, last_index] pairs of the 0-based word numbers in transcripts[].words, sorted, not overlapping. "
                                          "Never give seconds or times; the agent converts the word numbers. Cite the transcript_id in evidence. "
                                          "One word_cut recommendation per asset.")}}


# ---- decision ------------------------------------------------------------------------------------------------------------
def _transcript_of(analysis: Any, asset_id: str) -> Optional[Any]:
    for o in analysis.observations:
        if o.kind == "transcript" and o.asset_id == asset_id and any(isinstance(s, dict) and isinstance(s.get("words"), list) for s in (o.data or {}).get("segments") or []):
            return o
    return None


def decide_word_cut(eng: Any, m: Dict[str, Requirement], analysis: Any, inferences: List[Any], approval_for: Any, ai_prefix: str) -> Optional[Dict[str, Any]]:
    """`edit.word_cut` + the AI's `word_cut` recommendations → one `words.cut` decision per recommended asset: TRANSFORM (CONFIRM floor,
    the profile's semantic-deletion policy) with the derived ranges, or KEEP with the reason the ranges were refused. Without any
    recommendation, one KEEP record says nothing is cut yet. Returns the parsed requirement (None when not asked for)."""
    req = parse_word_cut_requirements(m)
    if req is None:
        return None
    reqs = req["requirements"]
    req_ids = [r.id for r in reqs]
    recs = [i for i in inferences if i.kind == ai_prefix + AI_INTENT]
    roll = {"pre_roll": req["pre_roll"], "post_roll": req["post_roll"], "pre_roll_provenance": req["pre_roll_provenance"], "post_roll_provenance": req["post_roll_provenance"]}
    if not recs:
        eng.decide(subject=SUBJECT, type="KEEP", decision="keep every word: no AI word-range recommendation",
                   reason=f"{KEY} asks the AI to name words to keep or cut; no recommendation was given (no provider, the provider failed, or the host has not answered "
                          "the brief yet: revise with --ai-recommendations), so nothing is cut",
                   confidence=1.0, evidence=req_ids, risk="LOW", approval="AUTO", provenance="USER", params={"asset_id": None, "rule": RULE_VERSION, **roll}, requirements=reqs)
        return req
    by_asset: Dict[str, List[Any]] = {}
    for inf in recs:
        by_asset.setdefault(inf.asset_id, []).append(inf)
    durations = {a.id: (a.technical or {}).get("duration") for a in analysis.assets}
    for asset_id in sorted(by_asset):
        infs = by_asset[asset_id]
        tr = _transcript_of(analysis, asset_id)
        base = {"asset_id": asset_id, "rule": RULE_VERSION, **roll}
        ev = [i.id for i in infs] + req_ids + ([tr.id] if tr is not None else [])
        why = None
        derived: Dict[str, Any] = {}
        index_source = None
        mode, raw = None, None
        if len(infs) > 1:
            why = f"{len(infs)} word_cut recommendations for {asset_id}: which one is meant is not guessed"
        elif tr is None:
            why = f"no transcript of {asset_id} with word timings: word ranges cannot be converted"
        else:
            try:
                mode, raw = recommendation_ranges((infs[0].data or {}).get("params"))
                words, index_source = word_index_of(tr.data or {})
                duration = durations.get(asset_id)
                derived = derive_ranges(words, raw, mode, duration if duration is not None else (tr.data or {}).get("duration"), req["pre_roll"], req["post_roll"])
            except WordRangeError as e:
                why = str(e)
        if why is not None:
            eng.decide(subject=SUBJECT, type="KEEP", decision=f"keep {asset_id} as it is: the AI's word ranges were refused",
                       reason=f"refused ({RULE_VERSION}): {why}; word ranges are never corrected, so nothing is cut", confidence=1.0, evidence=ev, risk="LOW",
                       approval="AUTO", provenance="AI_GENERATED", requirements=reqs,
                       params=_scrub({**base, "refused": why, "ai_params": (infs[0].data or {}).get("params") if len(infs) == 1 else None,
                                      **({"transcript_id": tr.id} if tr is not None else {})}))   # the AI's text is recorded, never interpreted
            continue
        inf = infs[0]
        removed_s = sum(e - s for s, e in derived["remove"])
        eng.decide(subject=SUBJECT, type="TRANSFORM",
                   decision=f"cut {derived['words_removed']} of {derived['word_count']} word(s) from {asset_id} ({mode} words {', '.join(f'{a}-{b}' for a, b in derived['word_ranges'])})",
                   reason=f"{inf.statement}; the AI named word ranges ({inf.data.get('provider')}/{inf.data.get('model')}), the agent converted them with {RULE_VERSION} "
                          f"(pre-roll {req['pre_roll']:g}s, post-roll {req['post_roll']:g}s, clamped at the gap to a word that is not kept and to the media): "
                          f"{removed_s:.3f}s removed; removing speech an AI chose always needs confirmation",
                   confidence=inf.confidence, evidence=ev, risk=RISK, approval=approval_for(SUBJECT, floor="CONFIRM", keys={SUBJECT: APPROVAL_KEY}),
                   provenance="AI_GENERATED", params={**base, "transcript_id": tr.id, "index_source": index_source, "inference_id": inf.id, **derived}, requirements=reqs)
    return req


# ---- validation ------------------------------------------------------------------------------------------------------------
def check_word_cuts(doc: Dict[str, Any]) -> List[str]:
    """IR check: every `words.cut` TRANSFORM derives the same seconds again from its recorded transcript, word ranges, rolls and rule,
    and the asset's trim step removes them. A different rule version, a missing transcript or a different result is an error."""
    errs: List[str] = []
    obs = {o.get("id"): o for o in (doc.get("analysis") or {}).get("observations") or []}
    steps = (doc.get("plan") or {}).get("steps") or []
    for d in doc.get("decisions") or []:
        if d.get("subject") != SUBJECT or d.get("type") != "TRANSFORM":
            continue
        p = d.get("params") or {}
        if p.get("rule") != RULE_VERSION:
            errs.append(f"decision {d['id']}: word-range rule {p.get('rule')!r} is not this agent's ({RULE_VERSION})")
            continue
        tr = obs.get(p.get("transcript_id"))
        if tr is None or tr.get("kind") != "transcript" or tr.get("asset_id") != p.get("asset_id"):
            errs.append(f"decision {d['id']}: transcript {p.get('transcript_id')!r} of {p.get('asset_id')!r} is not recorded")
            continue
        try:
            words, _ = word_index_of(tr.get("data") or {})
            again = derive_ranges(words, p.get("word_ranges"), p.get("mode"), p.get("duration"), p.get("pre_roll"), p.get("post_roll"))
        except WordRangeError as e:
            errs.append(f"decision {d['id']}: recorded word ranges are refused by {RULE_VERSION}: {e}")
            continue
        if again["keep"] != p.get("keep") or again["remove"] != p.get("remove"):
            errs.append(f"decision {d['id']}: recorded seconds differ from what {RULE_VERSION} derives from the recorded word ranges")
            continue
        if d.get("status") == "REJECTED":
            continue
        st = next((s for s in steps if d["id"] in (s.get("decision_ids") or [])), None)
        if st is None:
            errs.append(f"decision {d['id']}: no plan step cuts the word ranges")
            continue
        removed = (st.get("params") or {}).get("removed") or (st.get("params") or {}).get("remove") or []
        missing = [r for r in p["remove"] if not any(float(a) <= r[0] + 1e-6 and r[1] <= float(b) + 1e-6 for a, b in _merged(removed))]
        if missing:
            errs.append(f"decision {d['id']}: step {st['id']} does not remove {missing}")
    return errs


def _merged(ranges: Sequence[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for s, e in sorted([float(a), float(b)] for a, b in ranges):
        if out and s <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
