"""Take selection and beat-synced assembly (ADR-053): several takes of the same lines → one programme of the best take of
each line, in line order, with the joins on a measured beat grid when a music bed is named.

    transcript observations (transcription-skill) → groups of takes (text similarity) → scores (measured facts only) →
    one `takes.pick` decision per line → `video.concat` of the picks → (music bed) `takes.beat_snap` on the joins

Everything here is vocabulary and arithmetic over recorded observations, no execution and no AI:

- **Requirements** (explicit `--set` only): `edit.takes` (true: every video input is a take of some line), `edit.takes.similarity`
  (0.3..1.0, default 0.6: two takes are the same line when the similarity of their cleaned transcript text reaches it),
  `edit.takes.music` (input index of the music bed: its beat grid is measured by the registry-selected `beat_analysis` tool (scenes --beats) and the joins are
  snapped to it; the bed itself is not a take), `edit.takes.snap_max_shift` (seconds, 0.01..2.0, default 0.5: how far an out point
  may move to reach a beat).
- **Grouping** (deterministic): text = the take's transcript segments joined, NFKC-folded, lower-cased, fillers removed, every
  non-word character dropped. Similarity = difflib.SequenceMatcher ratio (autojunk off) of two such strings. Takes are visited
  in input order; a take joins the existing line whose first take it is most similar to when that ratio reaches the threshold
  (ties: the earlier line), else it opens a new line. Line order = the order lines first appear in the inputs.
- **Scores** (measured facts only; 100 minus penalties; every factor keeps the observation it was read from and its value):
  fillers (8 each, from the transcript), restarts (15 each: a word / 1-3-word run, or for CJK text a 3-10 character run, said
  twice in a row, fillers ignored), coverage (50 × the share of the line's representative take — the most similar to the others —
  that this take does not say), clipping (30 when the measured true peak is at or above -0.1 dBTP), pauses (4 per second of
  measured silence inside the take's speech span), loudness (2 per LU the measured integrated loudness lies more than 3 LU from
  the line's median). Sharpness, exposure and faces are not scored: no connected adapter measures them (named on every pick).
- **Pick**: highest score, ties to the earlier input. An AI recommendation (intent `take_selection`, validated by
  agent/ai_reasoning.py: it must cite recorded observations) naming a take of the line overrides the ranking; the scores stay as
  measured, the reason names both, and the pick then needs confirmation. The AI never adds or changes a measurement.
- **Assembly**: each pick keeps its speech span plus a 0.15 s handle (clamped to the source), the picks are joined in line order
  (`video.concat`, the same operation and plan path as `edit.concat`); the non-picked takes and the music bed get no step.
- **Beat snap** (music bed named): the bed is laid from programme time 0. Each pick's out point (= the join after it) moves to
  the nearest supported beat of the measured grid within `snap_max_shift`, never earlier than 0.05 s after its last word and never
  past its source; the out points are computed in programme time (the engine's `cut --snap beats` snaps source times, which is
  not where a join lands in a multi-take programme). A grid that is not usable or below 0.5 confidence is refused (BLOCK)."""
from __future__ import annotations

import difflib
import re
import statistics
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models import Requirement
from ..media.analysis import loudness_facts
from .editing import PROGRAMME, EditRequirementError, _bool, _number

TAKES_KEY = "edit.takes"
REQUIREMENT_KEYS = ("edit.takes", "edit.takes.similarity", "edit.takes.music", "edit.takes.snap_max_shift", "edit.takes.crossfade")
PICK_SUBJECT = "takes.pick"
GROUP_SUBJECT = "takes.group"          # BLOCKs about the take set as a whole (a take without speech, a conflicting request)
SNAP_SUBJECT = "takes.beat_snap"
CROSSFADE_SUBJECT = "takes.crossfade"  # audio crossfades at the joins of the assembly (the picture keeps its hard cuts)
AI_INTENT = "take_selection"           # the intent an AI recommendation names to prefer one take of a line
BEAT_KIND = "beats"                    # the analysis kind of the music bed's beat grid (ffmpeg-skill scenes --beats)
SIMILARITY_DEFAULT = 0.6
SIMILARITY_RANGE = (0.3, 1.0)
SNAP_SHIFT_DEFAULT = 0.5
SNAP_SHIFT_RANGE = (0.01, 2.0)
HANDLE_SECONDS = 0.15                  # kept before the first word and after the last word of a pick
MIN_TAIL_SECONDS = 0.05                # a snapped out point never lands closer than this after the last word
# Crossfade at each join (seconds). A chosen default, not taken from a published source: a butt join of two non-zero waveforms is
# a step in the signal, heard as a click; any short ramp removes the step, and 10 ms (one period of 100 Hz, well under the length
# of a spoken syllable) is long enough to do that without audibly blending two words. 0 turns the crossfades off (hard-cut audio).
CROSSFADE_DEFAULT = 0.010
CROSSFADE_RANGE = (0.0, 0.1)
# Planned A/V offset the validator accepts at every pick (seconds). Far below what a viewer can notice (ITU-R BT.1359-1: detectability
# thresholds about +45 ms sound-before-picture / -125 ms sound-after-picture); it only absorbs the IR's rounding.
SYNC_TOLERANCE = 0.001
AUDIO_MUX_SKILL = "audio_mux"          # the programme audio replaces the hard-cut audio of the assembled picture (ffmpeg-skill/audio --replace)
CONCAT_MAX_INPUTS = 32                 # audio-production CONCAT takes 2..32 inputs
BEAT_MIN_CONFIDENCE = 0.5              # ffmpeg-skill's own default refusal threshold for snapping (cut.py --min-confidence)
CLIP_TRUE_PEAK_DBTP = -0.1             # a measured true peak at or above this counts as clipped
LOUDNESS_SPREAD_LU = 3.0               # loudness differences within this from the line's median are not penalised
WEIGHTS = {"fillers": 8.0, "restarts": 15.0, "coverage": 50.0, "clipping": 30.0, "pauses": 4.0, "loudness": 2.0}
RANKING = "score (high first), then input order"
UNMEASURED = [{"factor": "sharpness", "reason": "no connected adapter measures sharpness (media-analysis-skill 0.1 has no such tool)"},
              {"factor": "exposure", "reason": "no connected adapter measures exposure (media-analysis-skill 0.1 has no such tool)"},
              {"factor": "faces", "reason": "no face-detection adapter is connected (detection-skill is not wired into the agent)"}]
# fillers: longest first so a longer form is consumed before its prefix. Short Japanese words that are also ordinary words
# (あの, その, まあ) are not fillers here; only their drawn-out forms are.
FILLERS_CJK = ("えーっと", "えーと", "えっと", "あのー", "そのー", "うーん", "えー", "んー")
FILLERS_WORD = ("umm", "um", "uhh", "uh", "erm", "er", "ah", "hmm")
_WORD_FILLER_RE = re.compile(r"\b(" + "|".join(FILLERS_WORD) + r")\b")
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]")


# ---- requirements ------------------------------------------------------------------------------------------------------
def parse_takes_requirements(m: Dict[str, Requirement]) -> Optional[Dict[str, Any]]:
    """→ {"similarity", "music" (input index or None), "snap_max_shift", "requirements"} when `edit.takes` is on, None otherwise.
    A refinement without `edit.takes`, or a value out of range, is refused (EditRequirementError); nothing is corrected."""
    main = m.get(TAKES_KEY)
    refinements = [m[k] for k in REQUIREMENT_KEYS[1:] if k in m and m[k].provenance != "DEFAULT"]
    if main is None or main.provenance == "DEFAULT" or not _bool(main.value, TAKES_KEY):
        if refinements:
            raise EditRequirementError(f"{refinements[0].key} is set but {TAKES_KEY} is not: refusing an ambiguous request")
        return None
    vals = {r.key: r.value for r in refinements}
    out: Dict[str, Any] = {"similarity": SIMILARITY_DEFAULT, "music": None, "snap_max_shift": SNAP_SHIFT_DEFAULT, "crossfade": CROSSFADE_DEFAULT, "crossfade_explicit": None,
                           "requirements": [main] + refinements}
    if "edit.takes.similarity" in vals:
        out["similarity"] = float(_number(vals["edit.takes.similarity"], "edit.takes.similarity", *SIMILARITY_RANGE))
    if "edit.takes.music" in vals:
        v = _number(vals["edit.takes.music"], "edit.takes.music", 0, 999)
        if float(v) != int(v):
            raise EditRequirementError(f"edit.takes.music must be an input index (0, 1, ...), got {vals['edit.takes.music']!r}")
        out["music"] = int(v)
    if "edit.takes.snap_max_shift" in vals:
        if out["music"] is None:
            raise EditRequirementError("edit.takes.snap_max_shift needs edit.takes.music (the beat grid comes from the music bed)")
        out["snap_max_shift"] = float(_number(vals["edit.takes.snap_max_shift"], "edit.takes.snap_max_shift", *SNAP_SHIFT_RANGE))
    if "edit.takes.crossfade" in vals:
        out["crossfade"] = float(_number(vals["edit.takes.crossfade"], "edit.takes.crossfade", *CROSSFADE_RANGE))
        out["crossfade_explicit"] = next(r for r in refinements if r.key == "edit.takes.crossfade")
    return out


# ---- text ----------------------------------------------------------------------------------------------------------------
def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).lower()


def count_fillers(text: str) -> int:
    t = _fold(text)
    n = len(_WORD_FILLER_RE.findall(t))
    t = _WORD_FILLER_RE.sub(" ", t)
    for f in FILLERS_CJK:
        n += t.count(f)
        t = t.replace(f, " ")
    return n


def strip_fillers(text: str) -> str:
    t = _WORD_FILLER_RE.sub(" ", _fold(text))
    for f in FILLERS_CJK:
        t = t.replace(f, " ")
    return t


def clean_text(text: str) -> str:
    """The comparison form of a take's words: folded, fillers removed, every non-word character (spaces, punctuation) dropped."""
    return re.sub(r"[\W_]+", "", strip_fillers(text))


def similarity(a: str, b: str) -> float:
    """Ratio of two cleaned texts (difflib, autojunk off); 0.0 when either is empty."""
    if not a or not b:
        return 0.0
    return round(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 4)


def coverage(take: str, line: str) -> float:
    """Share of the line's representative text that this take says (matching characters / its length)."""
    if not line:
        return 0.0
    m = difflib.SequenceMatcher(None, take, line, autojunk=False)
    return round(min(1.0, sum(b.size for b in m.get_matching_blocks()) / len(line)), 4)


def count_restarts(text: str) -> int:
    """Restarts: something said twice in a row (fillers ignored). Word text: a 1-3 word run repeated immediately ("the the",
    "I think I think"). CJK text (no word spaces): a 3-10 character run repeated immediately ("本日の本日の")."""
    t = strip_fillers(text)
    if _CJK_RE.search(t):
        s = re.sub(r"[\W_]+", "", t)
        n, i = 0, 0
        while i < len(s):
            hit = next((L for L in range(10, 2, -1) if i + 2 * L <= len(s) and s[i:i + L] == s[i + L:i + 2 * L]), 0)
            if hit:
                n, i = n + 1, i + hit
            else:
                i += 1
        return n
    words = re.findall(r"[^\W_]+(?:'[^\W_]+)?", t)
    n, i = 0, 0
    while i < len(words):
        hit = next((k for k in (3, 2, 1) if i + 2 * k <= len(words) and words[i:i + k] == words[i + k:i + 2 * k]), 0)
        if hit:
            n, i = n + 1, i + hit
        else:
            i += 1
    return n


# ---- measured facts of one take -------------------------------------------------------------------------------------------
def transcript_of(observations: List[Any], asset_id: str) -> Optional[Any]:
    return next((o for o in observations if o.asset_id == asset_id and o.kind == "transcript"), None)


def speech_of(transcript: Any) -> Tuple[str, List[Dict[str, Any]]]:
    segs = [s for s in (transcript.data.get("segments") or []) if isinstance(s, dict) and str(s.get("text") or "").strip()
            and s.get("start") is not None and s.get("end") is not None and float(s["end"]) > float(s["start"])]
    segs.sort(key=lambda s: (float(s["start"]), float(s["end"])))
    return " ".join(str(s["text"]).strip() for s in segs), segs


def _silence_ranges(data: Dict[str, Any], dur: float) -> List[Tuple[float, float]]:
    """Silence ranges from either measurement layout (the engine's `silences` pairs / media-analysis `segments`)."""
    out: List[Tuple[float, float]] = []
    for s in data.get("silences") or []:
        if isinstance(s, (list, tuple)) and len(s) >= 2 and s[0] is not None:
            out.append((float(s[0]), float(s[1]) if s[1] is not None else dur))
        elif isinstance(s, dict) and s.get("start") is not None:
            out.append((float(s["start"]), float(s["end"]) if s.get("end") is not None else dur))
    for s in data.get("segments") or []:
        if isinstance(s, dict) and s.get("start") is not None:
            out.append((float(s["start"]), float(s["end"]) if s.get("end") is not None else dur))
    return out


def group_takes(texts: List[Tuple[str, str]], threshold: float) -> List[Dict[str, Any]]:
    """texts: [(asset id, cleaned text)] in input order → [{"line", "takes": [asset ids], "similarity": {asset: ratio to the line's
    first take}}] in line order (the order lines first appear)."""
    lines: List[Dict[str, Any]] = []
    for aid, txt in texts:
        best, best_r = None, -1.0
        for ln in lines:
            r = similarity(txt, ln["anchor_text"])
            if r >= threshold and r > best_r:
                best, best_r = ln, r
        if best is None:
            lines.append({"line": len(lines) + 1, "anchor": aid, "anchor_text": txt, "takes": [aid], "similarity": {aid: 1.0}})
        else:
            best["takes"].append(aid)
            best["similarity"][aid] = best_r
    return lines


def score_takes(line: Dict[str, Any], facts: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Scores of every take of a line, best first ({RANKING}). `facts[asset]` = {"index", "raw", "clean", "transcript" (obs id),
    "speech": [s, e], "loudness" (obs id or None), "lufs", "true_peak", "silent", "silence" (obs id or None), "silences": [(s, e)]}."""
    takes = line["takes"]
    # the line's representative text: the take most similar to all the others (ties: the earlier take)
    rep = max(takes, key=lambda a: (sum(similarity(facts[a]["clean"], facts[b]["clean"]) for b in takes if b != a), -facts[a]["index"]))
    lufs = [facts[a]["lufs"] for a in takes if facts[a].get("lufs") is not None]
    median = statistics.median(lufs) if lufs else None
    out: List[Dict[str, Any]] = []
    for a in takes:
        f = facts[a]
        rows: List[Dict[str, Any]] = []
        tid = [f["transcript"]]
        n = count_fillers(f["raw"])
        rows.append({"factor": "fillers", "value": n, "penalty": WEIGHTS["fillers"] * n, "evidence": tid, "rule": f"{WEIGHTS['fillers']:g} per filler word in the transcript"})
        n = count_restarts(f["raw"])
        rows.append({"factor": "restarts", "value": n, "penalty": WEIGHTS["restarts"] * n, "evidence": tid, "rule": f"{WEIGHTS['restarts']:g} per run said twice in a row"})
        cov = coverage(f["clean"], facts[rep]["clean"])
        rows.append({"factor": "coverage", "value": cov, "penalty": round(WEIGHTS["coverage"] * (1.0 - cov), 2), "evidence": sorted({f["transcript"], facts[rep]["transcript"]}),
                     "rule": f"{WEIGHTS['coverage']:g} x the share of the line ({rep}'s text) this take does not say"})
        if f.get("loudness") and f.get("true_peak") is not None:
            clipped = float(f["true_peak"]) >= CLIP_TRUE_PEAK_DBTP
            rows.append({"factor": "clipping", "value": round(float(f["true_peak"]), 2), "penalty": WEIGHTS["clipping"] if clipped else 0.0, "evidence": [f["loudness"]],
                         "rule": f"{WEIGHTS['clipping']:g} when the measured true peak is at or above {CLIP_TRUE_PEAK_DBTP:g} dBTP"})
        if f.get("loudness") and f.get("lufs") is not None and median is not None:
            dev = abs(float(f["lufs"]) - median)
            rows.append({"factor": "loudness", "value": round(float(f["lufs"]), 2), "penalty": round(WEIGHTS["loudness"] * max(0.0, dev - LOUDNESS_SPREAD_LU), 2), "evidence": [f["loudness"]],
                         "rule": f"{WEIGHTS['loudness']:g} per LU beyond {LOUDNESS_SPREAD_LU:g} LU from the line's median ({median:g} LUFS)"})
        if f.get("silence"):
            s0, s1 = f["speech"]
            inside = round(sum(max(0.0, min(e, s1) - max(s, s0)) for s, e in f["silences"]), 3)
            rows.append({"factor": "pauses", "value": inside, "penalty": round(WEIGHTS["pauses"] * inside, 2), "evidence": [f["silence"]],
                         "rule": f"{WEIGHTS['pauses']:g} per second of measured silence inside the speech span"})
        score = round(100.0 - sum(r["penalty"] for r in rows), 2)
        out.append({"asset_id": a, "input_index": f["index"], "score": score, "similarity": line["similarity"].get(a), "factors": rows,
                    "unmeasured": [u["factor"] for u in UNMEASURED]})
    out.sort(key=lambda t: (-t["score"], t["input_index"]))
    for n, t in enumerate(out, 1):
        t["rank"] = n
    return out


def keep_of(speech: Tuple[float, float], duration: float) -> List[float]:
    return [round(max(0.0, speech[0] - HANDLE_SECONDS), 3), round(min(duration, speech[1] + HANDLE_SECONDS), 3)]


def snap_joins(picks: List[Dict[str, Any]], beats: List[float], max_shift: float) -> List[Dict[str, Any]]:
    """picks: [{"asset_id", "keep": [s, e], "speech_end", "duration"}] in programme order. Each out point moves (in programme time,
    the bed laid from 0) to the nearest beat within `max_shift`, no earlier than MIN_TAIL_SECONDS after the last word and no later
    than the source's end; ties go to the earlier beat. Returns one join record per pick (its keep updated in place)."""
    grid = sorted(float(b) for b in beats)
    t = 0.0
    out: List[Dict[str, Any]] = []
    for p in picks:
        s, e = float(p["keep"][0]), float(p["keep"][1])
        at = t + (e - s)
        lo = max(at - max_shift, t + (float(p["speech_end"]) + MIN_TAIL_SECONDS - s))
        hi = min(at + max_shift, t + (float(p["duration"]) - s))
        cands = [(abs(b - at), b, i) for i, b in enumerate(grid) if lo - 1e-9 <= b <= hi + 1e-9]
        rec: Dict[str, Any] = {"asset_id": p["asset_id"], "from": round(at, 3), "out_before": round(e, 3)}
        if cands:
            _, b, i = min(cands)
            e2 = round(s + (b - t), 3)
            rec.update({"to": round(b, 3), "beat_index": i, "delta": round(b - at, 3), "snapped": True, "out_after": e2})
            p["keep"] = [round(s, 3), e2]
            at = b
        else:
            rec.update({"to": round(at, 3), "beat_index": None, "delta": 0.0, "snapped": False, "out_after": round(e, 3),
                        "why": f"no supported beat within {max_shift:g}s that keeps the last word and stays inside the source"})
        out.append(rec)
        t = at
    return out


def plan_crossfades(picks: List[Dict[str, Any]], crossfade: float) -> Dict[str, Any]:
    """The audio of a hard-cut assembly with a crossfade at each join, in sync with the picture. picks: [{"asset_id", "keep": [s, e]
    (the picture's cut list, final: after the beat snap), "duration" (source)}] in programme order.

    audio-production CONCAT starts each clip `crossfade` seconds before the previous one ends, so n clips come out
    (n-1)·crossfade shorter than their sum. To keep the picture's cut points and the programme length, every join gets that
    overlap as audio handles: the outgoing pick's audio runs `out_handle` past its picture out point, the incoming pick's
    audio starts `in_handle` before its picture in point, out_handle + in_handle = crossfade (centred: half each; when one
    source has less room, the other side takes the rest). Then each pick's audio starts exactly `in_handle` before its picture
    starts in the programme, and the audio ends with the picture. A join where the two sources together have less than
    `crossfade` of room (or a pick not longer than the crossfade) keeps a hard cut, with the reason; the picks between two hard
    cuts form one run (one CONCAT with the crossfade), and the runs are joined by a CONCAT without one.
    Returns {"crossfade", "joins": [...], "spans": [...], "runs": [[pick index, ...]], "crossfaded", "programme_duration", "audio_duration"}."""
    d = float(crossfade)
    n = len(picks)
    vid = [(float(p["keep"][0]), float(p["keep"][1])) for p in picks]
    hin, hout = [0.0] * n, [0.0] * n
    joins: List[Dict[str, Any]] = []
    t = 0.0
    starts: List[float] = []
    for i, (s, e) in enumerate(vid):
        starts.append(round(t, 6))
        t += e - s
    total = round(t, 6)
    for i in range(n - 1):
        a, b = picks[i], picks[i + 1]
        room_out = round(float(a["duration"]) - vid[i][1], 6)
        room_in = round(vid[i + 1][0], 6)
        at = starts[i + 1]
        rec: Dict[str, Any] = {"after": a["asset_id"], "before": b["asset_id"], "at": round(at, 6), "room_out": room_out, "room_in": room_in}
        short = [p["asset_id"] for p, (s, e) in ((a, vid[i]), (b, vid[i + 1])) if e - s <= d + 1e-9]
        if short:
            why = f"{', '.join(short)} is not longer than the {d:g}s crossfade"
        elif room_out + room_in + 1e-9 < d:
            why = (f"no room for a {d:g}s crossfade: {a['asset_id']} has {room_out:g}s of source after its out point and "
                   f"{b['asset_id']} {room_in:g}s before its in point")
        else:
            why = ""
        if why:
            rec.update({"crossfade": 0.0, "out_handle": 0.0, "in_handle": 0.0, "crossfaded": False, "why": why + "; this join keeps a hard cut"})
        else:
            ho = round(min(room_out, max(d / 2.0, d - room_in)), 6)
            hi = round(d - ho, 6)
            hout[i], hin[i + 1] = ho, hi
            rec.update({"crossfade": d, "out_handle": ho, "in_handle": hi, "crossfaded": True, "centred": abs(ho - hi) <= 1e-6,
                        "window": [round(at - hi, 6), round(at + ho, 6)]})
        joins.append(rec)
    spans = [{"asset_id": p["asset_id"], "video": [round(vid[i][0], 6), round(vid[i][1], 6)], "audio": [round(vid[i][0] - hin[i], 6), round(vid[i][1] + hout[i], 6)],
              "in_handle": hin[i], "out_handle": hout[i], "programme_start": starts[i]} for i, p in enumerate(picks)]
    runs: List[List[int]] = [[0]] if n else []
    for i, j in enumerate(joins):
        if j["crossfaded"]:
            runs[-1].append(i + 1)
        else:
            runs.append([i + 1])
    audio = round(sum(sp["audio"][1] - sp["audio"][0] for sp in spans) - sum(j["crossfade"] for j in joins), 6)
    return {"crossfade": d, "joins": joins, "spans": spans, "runs": runs, "crossfaded": sum(1 for j in joins if j["crossfaded"]),
            "programme_duration": total, "audio_duration": audio}


def audio_offsets(spans: List[Dict[str, Any]], joins: List[Dict[str, Any]]) -> List[float]:
    """Where each pick's audio starts in the programme when the spans are concatenated with the joins' crossfades (the CONCAT
    arithmetic of audio-production: each clip starts `crossfade` before the previous one ends)."""
    out: List[float] = []
    t = 0.0
    for i, sp in enumerate(spans):
        out.append(round(t, 6))
        t += float(sp["audio"][1]) - float(sp["audio"][0]) - (float(joins[i]["crossfade"]) if i < len(joins) else 0.0)
    return out


# ---- decisions ------------------------------------------------------------------------------------------------------------
def decide_takes(eng: Any, m: Dict[str, Requirement], analysis: Any, inferences: List[Any], edits: Dict[str, Any], approval_for: Callable[..., Dict[str, Any]],
                 probe_ids_of: Callable[[List[str]], List[str]], cap_block: Callable[[str, str], Any], ai_prefix: str,
                 cap_missing: Optional[Callable[[str], Optional[str]]] = None) -> Optional[Dict[str, Any]]:
    """`edit.takes` → groups, scores, one `takes.pick` per line, the `video.concat` of the picks (and `takes.beat_snap` with a music
    bed). Returns None when not requested, else {"ok", "picks": [asset ids in line order], "excluded": {asset ids with no step}}.
    Appends to `eng.decisions`. A conflicting or unsupported request is a BLOCK, never a guess."""
    tk = parse_takes_requirements(m)
    if tk is None:
        return None
    reqs = tk["requirements"]
    ev_req = [r.id for r in reqs]
    assets = list(analysis.assets)
    all_ids = [a.id for a in assets]
    res: Dict[str, Any] = {"ok": False, "picks": [], "excluded": set()}

    def block(subject: str, what: str, why: str, extra: Optional[List[str]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        eng.decide(subject=subject, type="BLOCK", decision=f"BLOCK: {what}", reason=why + " (not guessed)", confidence=1.0, evidence=ev_req + (extra or probe_ids_of(all_ids)),
                   risk="HIGH", approval="BLOCK", provenance="USER", params=params or {}, requirements=reqs)
        return res

    other = [k for k in ("video.concat", "video.switch", "video.grid") if k in edits]
    if other:
        return block(GROUP_SUBJECT, f"take selection and {', '.join(other)} requested together",
                     "edit.takes builds the programme from the picks itself; another programme-building operation would compete with it")
    music = None
    if tk["music"] is not None:
        if tk["music"] >= len(assets):
            raise EditRequirementError(f"edit.takes.music is input index {tk['music']}, but only {len(assets)} input(s) were given")
        music = assets[tk["music"]]
    takes = [a for a in assets if a is not music]
    no_video = [a.id for a in takes if not (a.technical or {}).get("video")]
    if no_video:
        return block(GROUP_SUBJECT, f"take(s) without a video stream: {', '.join(no_video)}", "every input except the music bed is a take of a line and needs a picture")
    facts: Dict[str, Dict[str, Any]] = {}
    silent_takes: List[str] = []
    for n, a in enumerate(assets):
        if a is music:
            continue
        tr = transcript_of(analysis.observations, a.id)
        raw, segs = speech_of(tr) if tr is not None else ("", [])
        if tr is None or not segs or not clean_text(raw):
            silent_takes.append(a.id)
            continue
        dur = float((a.technical or {}).get("duration") or 0.0)
        loud = next((o for o in analysis.observations if o.asset_id == a.id and o.kind == "loudness"), None)
        lf = loudness_facts(loud.data) if loud is not None else {}
        sil = next((o for o in analysis.observations if o.asset_id == a.id and o.kind == "silence"), None)
        facts[a.id] = {"index": n, "raw": raw, "clean": clean_text(raw), "transcript": tr.id, "speech": [float(segs[0]["start"]), float(segs[-1]["end"])], "duration": dur,
                       "loudness": loud.id if loud is not None else None, "lufs": lf.get("lufs"), "true_peak": lf.get("true_peak"),
                       "silence": sil.id if sil is not None else None, "silences": _silence_ranges(sil.data, dur) if sil is not None else []}
    if silent_takes:
        return block(GROUP_SUBJECT, f"no recognised speech in take(s) {', '.join(silent_takes)}",
                     "a take is grouped by what is said in it; without a transcript of it, it cannot be placed on a line",
                     extra=probe_ids_of(silent_takes) + [o.id for o in analysis.observations if o.asset_id in silent_takes and o.kind == "transcript"])
    lines = group_takes([(a.id, facts[a.id]["clean"]) for a in takes], tk["similarity"])
    ai = [i for i in inferences if i.kind == ai_prefix + AI_INTENT]
    picks: List[Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]] = []
    for ln in lines:
        ranked = score_takes(ln, facts)
        det = ranked[0]["asset_id"]
        named = sorted({i.asset_id for i in ai if i.asset_id in ln["takes"]})
        override = next((i for i in ai if i.asset_id in ln["takes"]), None) if len(named) == 1 and named[0] != det else None
        pick = override.asset_id if override is not None else det
        f = facts[pick]
        rec = {"line": ln["line"], "pick": pick, "asset_id": pick, "deterministic": det, "override": override, "ranked": ranked, "keep": keep_of(f["speech"], f["duration"]),
               "speech_end": f["speech"][1], "duration": f["duration"], "ambiguous_ai": len(named) > 1}
        picks.append((ln, rec, ranked))
    joins: List[Dict[str, Any]] = []
    beat_obs = None
    if music is not None:
        beat_obs = next((o for o in analysis.observations if o.asset_id == music.id and o.kind == BEAT_KIND), None)
        grid = (beat_obs.data if beat_obs is not None else {}) or {}
        conf = grid.get("confidence")
        if beat_obs is None:
            row = next((r for an in analysis.analyses for r in an.get("rows") or [] if r.get("asset_id") == music.id and r.get("kind") == BEAT_KIND), None)
            why = ((row or {}).get("error") or {}).get("message") or (row or {}).get("reason") or "no beat measurement was recorded"
            return block(SNAP_SUBJECT, f"no measured beat grid for the music bed {music.id}", f"the joins cannot be put on beats without a measured grid ({why})",
                         params={"asset_id": PROGRAMME, "music": music.id})
        if not grid.get("usable", True) or conf is None or float(conf) < BEAT_MIN_CONFIDENCE or not grid.get("supported_beats"):
            return block(SNAP_SUBJECT, f"the beat grid of {music.id} is not usable (confidence {conf})",
                         f"a grid below {BEAT_MIN_CONFIDENCE:g} confidence or without supported beats would put the joins on times nothing in the music supports",
                         extra=[beat_obs.id], params={"asset_id": PROGRAMME, "music": music.id})
        joins = snap_joins([rec for _, rec, _ in picks], [float(b) for b in grid["supported_beats"]], tk["snap_max_shift"])
    # ---- one pick per line
    pick_ids: List[str] = []
    for ln, rec, ranked in picks:
        ev = ev_req + sorted({e for t in ranked for r in t["factors"] for e in r["evidence"]}) + probe_ids_of(ln["takes"])
        by = {t["asset_id"]: t for t in ranked}
        ov = rec["override"]
        appr = approval_for(PICK_SUBJECT, floor="CONFIRM" if ov is not None else None)
        params: Dict[str, Any] = {"asset_id": rec["pick"], "line": rec["line"], "takes": ranked, "deterministic_pick": rec["deterministic"], "ranking": RANKING,
                                  "keep": list(rec["keep"]), "speech": [round(x, 3) for x in facts[rec["pick"]]["speech"]], "similarity_threshold": tk["similarity"],
                                  "unmeasured": [dict(u) for u in UNMEASURED], "music_bed": music.id if music is not None else None,
                                  "text": facts[ln["anchor"]]["raw"][:200]}
        why = (f"line {rec['line']}: {len(ln['takes'])} take(s) grouped by transcript similarity >= {tk['similarity']:g}; "
               f"{rec['pick']} scores {by[rec['pick']]['score']:g}" + "".join(f", {t['asset_id']} {t['score']:g}" for t in ranked if t["asset_id"] != rec["pick"])
               + " (fillers / restarts / coverage from the transcript, clipping / loudness / pauses from the audio measurements; sharpness, exposure and faces are not measured here)")
        if ov is not None:
            params["ai"] = {"inference": ov.id, "statement": ov.statement, "provider": ov.data.get("provider"), "model": ov.data.get("model"), "overrides": rec["deterministic"]}
            why += f"; AI ({ov.data.get('provider')}/{ov.data.get('model')}) preferred {rec['pick']} over the measured best {rec['deterministic']}: {ov.statement} -- the scores are unchanged, so the pick needs confirmation"
            ev = ev + [ov.id]
        elif rec["ambiguous_ai"]:
            why += "; AI recommendations named more than one take of this line, so the measured ranking stands"
        snapped = next((j for j in joins if j["asset_id"] == rec["pick"]), None)
        if snapped is not None:
            params["keep_unsnapped"] = [rec["keep"][0], snapped["out_before"]]
        d = eng.decide(subject=PICK_SUBJECT, type="TRANSFORM", decision=f"use {rec['pick']} for line {rec['line']} (score {by[rec['pick']]['score']:g} of {len(ranked)} take(s))",
                       reason=why, confidence=1.0 if ov is None else float(ov.confidence), evidence=ev, risk="MEDIUM", approval=appr,
                       alternatives=[{"decision": f"use {t['asset_id']}", "reason": f"score {t['score']:g} (rank {t['rank']})", "cost": ""} for t in ranked if t["asset_id"] != rec["pick"]],
                       provenance="AI_GENERATED" if ov is not None else "INFERRED", params=params, requirements=reqs, serves_intent=None)
        pick_ids.append(d.id)
        res["picks"].append(rec["pick"])
        res["excluded"] |= {a for a in ln["takes"] if a != rec["pick"]}
    if music is not None:
        res["excluded"].add(music.id)
    # ---- the assembly: the picks in line order (the same concat operation and plan path as edit.concat)
    if len(res["picks"]) >= 2:
        eng.decide(subject="video.concat", type="TRANSFORM", decision="concat " + " + ".join(res["picks"]) + f" → {PROGRAMME} (the picked take of each line, in line order)",
                   reason=f"user asked to assemble the takes ({reqs[0].source}); the picks of {len(res['picks'])} line(s) become one programme timeline",
                   confidence=1.0, evidence=ev_req + probe_ids_of(res["picks"]), risk="MEDIUM", approval=approval_for("video.concat", explicit=reqs[0]), provenance="USER",
                   params={"asset_id": PROGRAMME, "inputs": list(res["picks"]), "picks": pick_ids}, requirements=reqs, serves_intent=None)
        cap_block("video_concat", "capability.video_concat")
    if music is not None:
        music_req = next(r for r in reqs if r.key == "edit.takes.music")
        n_snap = sum(1 for j in joins if j["snapped"])
        g = beat_obs.data
        eng.decide(subject=SNAP_SUBJECT, type="TRANSFORM", decision=f"put {n_snap} of {len(joins)} out point(s) on the measured beat grid of {music.id} ({float(g['tempo_bpm']):g} BPM)",
                   reason=f"user named a music bed ({music_req.source}); its beat grid was measured (confidence {float(g['confidence']):g}); each out point moves to the nearest supported "
                          f"beat within {tk['snap_max_shift']:g}s in programme time, never into the last word"
                          + ("" if n_snap == len(joins) else f"; {len(joins) - n_snap} out point(s) had no beat in reach and stay where the speech ends"),
                   confidence=float(g["confidence"]), evidence=ev_req + [beat_obs.id], risk="LOW", approval=approval_for(SNAP_SUBJECT, explicit=music_req), provenance="USER",
                   params={"asset_id": PROGRAMME, "music": music.id, "tempo_bpm": float(g["tempo_bpm"]), "interval": g.get("interval"), "confidence": float(g["confidence"]),
                           "max_shift": tk["snap_max_shift"], "joins": joins, "snapped": n_snap},
                   requirements=reqs, serves_intent=None)
    if len(res["picks"]) >= 2 and tk["crossfade"] > 0:
        decide_crossfades(eng, tk, [rec for _, rec, _ in picks], pick_ids, ev_req + probe_ids_of(res["picks"]), approval_for, cap_block, cap_missing)
    res["ok"] = True
    return res


def decide_crossfades(eng: Any, tk: Dict[str, Any], recs: List[Dict[str, Any]], pick_ids: List[str], evidence: List[str], approval_for: Callable[..., Dict[str, Any]],
                      cap_block: Callable[[str, str], Any], cap_missing: Optional[Callable[[str], Optional[str]]]) -> None:
    """`takes.crossfade`: the audio crossfades at the joins of the assembly (plan_crossfades). TRANSFORM when at least one join
    gets one; KEEP (hard-cut audio, nothing planned) when no join has room, or when the default crossfade cannot run here because
    audio-production / the mux tool is unavailable. An explicit `edit.takes.crossfade` that cannot run is a BLOCK (never dropped)."""
    explicit = tk.get("crossfade_explicit")
    reqs = tk["requirements"]
    xf = plan_crossfades([{"asset_id": r["pick"], "keep": list(r["keep"]), "duration": r["duration"]} for r in recs], tk["crossfade"])
    params: Dict[str, Any] = {"asset_id": PROGRAMME, "crossfade": xf["crossfade"], "default": explicit is None, "joins": xf["joins"], "spans": xf["spans"], "runs": xf["runs"],
                              "crossfaded": xf["crossfaded"], "programme_duration": xf["programme_duration"], "audio_duration": xf["audio_duration"],
                              "sync_tolerance": SYNC_TOLERANCE, "picks": list(pick_ids), "curve": "the audio-production CONCAT crossfade (its own default curve)"}
    source = f"edit.takes.crossfade={xf['crossfade']:g}s ({explicit.source})" if explicit is not None else f"the default {CROSSFADE_DEFAULT:g}s click-free crossfade (edit.takes.crossfade)"
    hard = [j for j in xf["joins"] if not j["crossfaded"]]
    hard_note = "".join(f"; join {j['after']} → {j['before']}: {j['why']}" for j in hard)
    unavailable = [(sk, why) for sk, why in ((sk, cap_missing(sk) if cap_missing else None) for sk in ("audio_cut", "audio_concat", AUDIO_MUX_SKILL)) if why]
    too_many = len(recs) > CONCAT_MAX_INPUTS
    if xf["crossfaded"] == 0 or too_many or (unavailable and explicit is None):
        if too_many:
            why = f"{len(recs)} picks: audio-production CONCAT joins at most {CONCAT_MAX_INPUTS} inputs, so the joins keep hard cuts"
        elif xf["crossfaded"] == 0:
            why = f"no join has room for {source}" + hard_note
        else:
            why = f"{source} needs " + "; ".join(f"{sk} ({w})" for sk, w in unavailable) + ", which is unavailable here, so the joins keep hard-cut audio (set edit.takes.crossfade=0 to say so explicitly)"
        params.update({"crossfaded": 0, "unavailable": [sk for sk, _ in unavailable]})
        eng.decide(subject=CROSSFADE_SUBJECT, type="KEEP", decision=f"keep hard-cut audio at all {len(xf['joins'])} join(s)", reason=why + " (not guessed)", confidence=1.0,
                   evidence=evidence, risk="LOW", approval="AUTO", provenance="USER" if explicit is not None else "SYSTEM", params=params, requirements=reqs, serves_intent=None)
        return
    eng.decide(subject=CROSSFADE_SUBJECT, type="TRANSFORM",
               decision=f"crossfade the audio at {xf['crossfaded']} of {len(xf['joins'])} join(s) ({xf['crossfade']:g}s, picture cuts unchanged)",
               reason=(f"{source}: each crossfaded join takes its overlap from audio handles (the outgoing take's audio continues past its out point, the incoming "
                       f"take's starts before its in point), so the picture's cut points and the programme length ({xf['programme_duration']:g}s) stay as cut and every pick's "
                       f"audio stays in sync with its picture; the programme audio replaces the hard-cut audio of the assembled picture" + hard_note),
               confidence=1.0, evidence=evidence, risk="LOW", approval=approval_for(CROSSFADE_SUBJECT, explicit=explicit if explicit is not None else reqs[0]),
               provenance="USER" if explicit is not None else "SYSTEM", params=params, requirements=reqs, serves_intent=None)
    for sk in ("audio_cut", "audio_concat", AUDIO_MUX_SKILL):
        cap_block(sk, f"capability.{sk}")
