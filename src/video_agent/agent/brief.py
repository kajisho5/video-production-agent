"""Brief → requirements (AI task `requirements_extraction`, ADR-061).

A brief is what the user wants in their own words ("60 s vertical, captions, -14 LUFS", 「縦型で字幕、60秒、-14LUFS」). This
module turns it into the same requirement keys `--set` takes, and nothing else:

- **Allowed keys** are not listed here. The service passes the set it derives from the code that validates `--set`
  (`service.BRIEF_REQUIREMENT_KEYS`: the keys `_check_edit_requirements` recognises by name, minus policy / approval keys,
  file-path keys and `job.*`). A key outside it is refused with the reason, never guessed into a near key.
- **Values** go through the same checks as `--set` (the `check` callable the service passes: the same requirement parsers,
  combined with the explicit `--set` of the request so a conflict such as fit + fill is refused too). A value that fails is
  refused with the parser's own message.
- **Two extractors, one validation.** With an AI provider (the Claude provider, ADR-049, or the chat model through the host
  provider / MCP, ADR-050) the model reads the brief and returns `{key, value, phrase}` items; each must quote a `phrase` that
  occurs in the brief (the brief is its evidence, like an observation id for a recommendation, ADR-018) or it is dropped. With
  no provider (or a provider failure, or a host that has not answered yet) the deterministic `parse_brief` below runs. Both
  results are validated by `validate_candidates` the same way.
- **Explicit wins.** A key also given with `--set` keeps the explicit value; the brief's value is recorded as `overridden`.
- **Provenance.** Requirements from the rule-based parser are `USER` (source `brief:rules`), like the request-text keyword pass
  (ADR-005: an unambiguous phrase of the user's own); from an AI they are `AI_GENERATED` (source `brief:ai:<provider>/<model>`).
  The decision engine raises every executable decision that rests on an AI_GENERATED requirement to at least CONFIRM, and the
  brief itself is recorded as the decision `requirements.brief` (REVIEW / CONFIRM for an AI reading, KEEP / AUTO for the rules)
  citing the brief requirement and every requirement taken from it.

Rule-based coverage (`parse_brief`, English and Japanese, after NFKC so full-width digits and colons work):
  duration   "60 s", "60 sec", "90 seconds", "2 min", "60-second", 「60秒」「2分」「1分30秒」 → recognised, but there is no
             `--set` key for a target length, so it is reported as `unmapped` with that reason (never applied)
  aspect     "vertical" / "portrait" / "9:16" / 「縦」「縦型」「縦長」 → edit.fit=9:16; "horizontal" / "landscape" / "16:9" /
             「横型」「横長」 → edit.fit=16:9; "square" / "1:1" / 「正方形」 → edit.fit=1:1; also 4:5 and 4:3 written as ratios.
             fit (letterbox, nothing cut) is chosen over fill (crop); two different aspects → unmapped (conflict)
  captions   "caption(s)" / "subtitle(s)" / "subtitled" / 「字幕」「キャプション」 → subtitle=true; "burned-in" / "burn in" /
             "open captions" / "hardcoded" / 「焼き付け」「焼き込み」 → subtitle.burn_in=true; "soft subtitles" / "subtitle track" /
             「ソフト字幕」「字幕トラック」 → subtitle.mode=mux (ADR-063; burn-in and a soft track together → unmapped); "srt" / "vtt" → subtitle.format;
             "no captions" / "without subtitles" / 「字幕なし」「字幕不要」 → nothing
  loudness   "-14 LUFS" / "LUFS -14" / 「ラウドネス -14」 → audio.loudness.target_lufs (and audio.normalize=true); "-1 dBTP" →
             audio.loudness.true_peak; a LUFS figure without a minus sign is ambiguous → unmapped
  keywords   the request-text keywords of agent/requirements.py (trim the leading / trailing silence, normalise, frame accurate)
Not covered (left to `--set` or to an AI provider): platform names that imply an aspect ("Shorts", "TikTok", 「ショート」),
「テロップ」 (on-screen text, not necessarily captions), colour, graphics, thumbnails, takes, and anything else.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..media.analysis import _scrub
from ..models import Requirement
from ..providers import AIRequest, AIResponse
from .requirements import KEYWORDS

TASK = "requirements_extraction"
BRIEF_KEY = "brief"                       # the Requirement that carries the brief's text (the evidence of what it yields)
SUBJECT = "requirements.brief"            # the decision that records the reading of the brief
SOURCE_RULES = "brief:rules"
SOURCE_AI = "brief:ai"
MAX_BRIEF_CHARS = 2000
RESULT_SCHEMA = {"requirements": [{"key": "<one of context.allowed_keys>", "value": "<the value as `--set key=value` takes it: JSON scalar or text>",
                                   "phrase": "<the words of the brief this comes from, quoted exactly>", "confidence": "0..1"}],
                 "unmapped": [{"phrase": "<words of the brief no allowed key expresses>", "reason": "<why>"}]}
NO_DURATION_KEY = "no requirement key sets a target length (no `--set` key trims or fits the programme to a duration): recorded, not applied"


def normalize(text: str) -> str:
    """NFKC (full-width digits / colons / letters) and the minus-like dashes → '-'."""
    t = unicodedata.normalize("NFKC", text or "")
    return t.replace("−", "-").replace("–", "-").replace("‒", "-").replace("－", "-")


def coerce(value: Any) -> Any:
    """The `--set key=value` reading: a string that parses as JSON becomes that value (true / -14 / "srt"), else stays text."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


# ---- rule-based extraction ---------------------------------------------------------------------------------------------
_NUM = r"(\d+(?:\.\d+)?)"
_DURATION = [
    (re.compile(r"(\d+)\s*分\s*(\d+)\s*秒"), lambda m: int(m.group(1)) * 60 + int(m.group(2))),
    (re.compile(_NUM + r"\s*(?:-\s*)?(?:s|sec|secs|second|seconds)\b", re.I), lambda m: float(m.group(1))),
    (re.compile(_NUM + r"\s*(?:-\s*)?(?:min|mins|minute|minutes)\b", re.I), lambda m: float(m.group(1)) * 60),
    (re.compile(_NUM + r"\s*秒"), lambda m: float(m.group(1))),
    (re.compile(_NUM + r"\s*分(?!\s*\d+\s*秒)"), lambda m: float(m.group(1)) * 60),
]
_ASPECTS = [
    (re.compile(r"\b(?:vertical|portrait)\b|\b9\s*[:x×/]\s*16\b|縦(?!横)", re.I), "9:16"),
    (re.compile(r"\b(?:horizontal|landscape|widescreen)\b|\b16\s*[:x×/]\s*9\b|横(?:型|長|向き)", re.I), "16:9"),
    (re.compile(r"\bsquare\b|\b1\s*:\s*1\b|正方形", re.I), "1:1"),
    (re.compile(r"\b4\s*:\s*5\b"), "4:5"),
    (re.compile(r"\b4\s*:\s*3\b"), "4:3"),
]
_CAPTIONS = re.compile(r"\b(?:closed\s+)?(?:captions?|captioned|subtitles?|subtitled)\b|字幕|キャプション", re.I)
_NO_CAPTIONS = re.compile(r"\b(?:no|without)\s+(?:closed\s+)?(?:captions?|subtitles?)\b|字幕\s*(?:なし|無し|不要|は?いらない)", re.I)
_BURN_IN = re.compile(r"\bburn(?:ed|t)?[- ]?in\b|\bopen\s+captions?\b|\bhard[- ]?(?:coded|subs?)\b|焼き付け|焼き込み|焼きこみ", re.I)
_SOFT_SUBS = re.compile(r"\bsoft[- ]?(?:subtitles?|subs?|captions?)\b|\b(?:subtitle|caption)\s+track\b|ソフト字幕|字幕トラック", re.I)
_SUB_FORMAT = [(re.compile(r"\bsrt\b", re.I), "srt"), (re.compile(r"\b(?:web)?vtt\b", re.I), "vtt")]
_LUFS = [re.compile(r"(-\s*\d+(?:\.\d+)?)\s*LUFS", re.I), re.compile(r"LUFS\s*[:=]?\s*(-\s*\d+(?:\.\d+)?)", re.I),
         re.compile(r"ラウドネス\s*[:=]?\s*(-\s*\d+(?:\.\d+)?)")]
_LUFS_UNSIGNED = re.compile(r"(?<![-\d.])(\d+(?:\.\d+)?)\s*LUFS", re.I)
_TRUE_PEAK = re.compile(r"(-\s*\d+(?:\.\d+)?)\s*dB\s*TP", re.I)


def parse_brief(text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic reading of a brief: (candidates [{key, value, phrase}], unmapped [{phrase, reason}]). Only the phrases in
    the module docstring are read; everything else is left alone (an empty result is a valid result)."""
    t = normalize(text)
    out: List[Dict[str, Any]] = []
    unmapped: List[Dict[str, Any]] = []

    def add(key: str, value: Any, phrase: str) -> None:
        if not any(c["key"] == key for c in out):
            out.append({"key": key, "value": value, "phrase": phrase.strip()})

    # duration: recognised and reported, never applied (no key for it)
    taken: List[Tuple[int, int]] = []
    for rx, seconds in _DURATION:
        for m in rx.finditer(t):
            if any(a <= m.start() < b for a, b in taken):
                continue
            taken.append((m.start(), m.end()))
            unmapped.append({"phrase": m.group(0).strip(), "seconds": seconds(m), "reason": NO_DURATION_KEY})
    # aspect → edit.fit (letterbox; nothing of the picture is cut)
    found = [(rx.search(t), ratio) for rx, ratio in _ASPECTS]
    found = [(m, r) for m, r in found if m]
    ratios = sorted({r for _, r in found})
    if len(ratios) == 1:
        add("edit.fit", ratios[0], found[0][0].group(0))
    elif len(ratios) > 1:
        unmapped.append({"phrase": ", ".join(m.group(0) for m, _ in found), "reason": f"conflicting aspect ratios {ratios}: not guessed"})
    # captions → subtitle (+ burn-in / format)
    cap = _CAPTIONS.search(t)
    if cap and not _NO_CAPTIONS.search(t):
        add("subtitle", True, cap.group(0))
        burn, soft = _BURN_IN.search(t), _SOFT_SUBS.search(t)
        if burn and soft:
            unmapped.append({"phrase": f"{burn.group(0)}, {soft.group(0)}", "reason": "burnt-in and soft subtitles both named: which one is meant is not guessed"})
        elif burn:
            add("subtitle.burn_in", True, burn.group(0))
        elif soft:
            add("subtitle.mode", "mux", soft.group(0))
        for rx, fmt in _SUB_FORMAT:
            m = rx.search(t)
            if m:
                add("subtitle.format", fmt, m.group(0))
                break
    # loudness target / true peak
    for rx in _LUFS:
        m = rx.search(t)
        if m:
            add("audio.loudness.target_lufs", float(m.group(1).replace(" ", "")), m.group(0))
            add("audio.normalize", True, m.group(0))
            break
    else:
        m = _LUFS_UNSIGNED.search(t)
        if m:
            unmapped.append({"phrase": m.group(0).strip(), "reason": "a LUFS target without a minus sign is ambiguous: write it as -N LUFS"})
    m = _TRUE_PEAK.search(t)
    if m:
        add("audio.loudness.true_peak", float(m.group(1).replace(" ", "")), m.group(0))
    # the request-text keywords (agent/requirements.py), same unambiguous phrases
    for rx, key, value in KEYWORDS:
        m = rx.search(t)
        if m and key != "delivery.platform":
            add(key, value, m.group(0))
    return out, unmapped


# ---- AI extraction -----------------------------------------------------------------------------------------------------
def build_request(text: str, allowed_keys: Iterable[str], explicit: Optional[Dict[str, Any]] = None) -> AIRequest:
    """What a provider sees: the brief (cut to MAX_BRIEF_CHARS), the allowed keys and which keys the user already set
    explicitly (those keep their explicit value whatever the answer says)."""
    return AIRequest(task_type=TASK, inputs={"brief": (text or "")[:MAX_BRIEF_CHARS]}, schema=RESULT_SCHEMA,
                     context={"allowed_keys": sorted(allowed_keys), "explicit_keys": sorted(explicit or {})})


def candidates_from_response(resp: AIResponse, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """(candidates, unmapped, warnings). An item must be an object with a key and a quoted phrase that occurs in the brief
    (after NFKC, case-insensitive); otherwise it is dropped with a warning, never repaired."""
    res = resp.result if isinstance(resp.result, dict) else {}
    items = res.get("requirements")
    if not isinstance(items, list):
        return [], [], [f"ai: response has no requirements list (task {resp.task_type})"]
    hay = normalize(text).casefold()
    out: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict) or not str(it.get("key") or "").strip():
            warnings.append(f"ai: requirement {i} is not an object with a key; dropped")
            continue
        phrase = str(it.get("phrase") or "").strip()
        if not phrase or normalize(phrase).casefold() not in hay:
            warnings.append(f"ai: requirement {i} ({it.get('key')}) quotes no phrase of the brief; dropped")
            continue
        try:
            conf = min(1.0, max(0.0, float(it.get("confidence", resp.confidence))))
        except (TypeError, ValueError):
            conf = 0.0
        out.append({"key": str(it["key"]).strip(), "value": coerce(it.get("value")), "phrase": phrase[:200], "confidence": conf})
    unmapped = [{"phrase": str(u.get("phrase") or "")[:200], "reason": str(u.get("reason") or "")[:300]}
                for u in (res.get("unmapped") or []) if isinstance(u, dict)]
    return out, unmapped, warnings


# ---- validation (both extractors) --------------------------------------------------------------------------------------
def validate_candidates(candidates: List[Dict[str, Any]], allowed: Iterable[str], explicit: Dict[str, Any],
                        check: Callable[[Dict[str, Any]], None]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(accepted {key: candidate}, refused [{key, value, phrase, reason}], overridden [{key, value, explicit}]).
    Keys are taken in key order (a switch before its refinements: `edit.fit` < `edit.fit.width`), each checked with `check`
    on the explicit requirements plus everything accepted so far, so a value is refused for exactly the reason `--set`
    would refuse it, alone or in combination."""
    allowed = set(allowed)
    accepted: Dict[str, Dict[str, Any]] = {}
    refused: List[Dict[str, Any]] = []
    overridden: List[Dict[str, Any]] = []
    seen = set()
    for c in sorted(candidates, key=lambda c: c["key"]):
        key, value = c["key"], c["value"]
        if key in seen:
            refused.append({**c, "reason": f"{key} is given twice by the brief: only the first is used"})
            continue
        seen.add(key)
        if key not in allowed:
            refused.append({**c, "reason": f"unknown requirement key {key!r}: a brief may only yield keys `--set` takes (and not policy, file-path or job keys)"})
            continue
        if key in explicit:
            overridden.append({"key": key, "value": value, "explicit": explicit[key], "phrase": c.get("phrase", "")})
            continue
        trial = {**explicit, **{k: a["value"] for k, a in accepted.items()}, key: value}
        try:
            check(trial)
        except ValueError as e:
            refused.append({**c, "reason": f"invalid value: {e}"})
            continue
        accepted[key] = c
    return accepted, refused, overridden


def requirements_of(record: Dict[str, Any], explicit_keys: Iterable[str]) -> List[Requirement]:
    """The brief's Requirements for extract_requirements: the brief text itself (USER, the evidence) and every accepted key
    the explicit requirements do not already set. A brief whose reading was rejected (revise) yields only its text."""
    if not record or not record.get("text"):
        return []
    out = [Requirement(key=BRIEF_KEY, value=record["text"], provenance="USER", source="cli:--brief")]
    skip = set(explicit_keys)
    for a in [] if record.get("rejected") else record.get("accepted") or []:
        if a["key"] not in skip:
            out.append(Requirement(key=a["key"], value=a["value"], provenance=a["provenance"], source=a["source"]))
    return out


def values_of(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """{key: value} the brief contributes (empty once its reading is rejected)."""
    if not record or record.get("rejected"):
        return {}
    return {a["key"]: a["value"] for a in record.get("accepted") or []}


def decide_brief(eng: Any, reqs: List[Requirement]) -> None:
    """Record the reading of the brief as a decision citing the brief and every requirement taken from it. An AI reading is a
    REVIEW item the user confirms (CONFIRM); the rule-based reading is a KEEP record (AUTO, it creates nothing by itself)."""
    brief = next((r for r in reqs if r.key == BRIEF_KEY), None)
    if brief is None:
        return
    taken = [r for r in reqs if r.source.startswith(("brief:rules", "brief:ai"))]
    ai = any(r.provenance == "AI_GENERATED" for r in taken)
    keys = ", ".join(f"{r.key}={json.dumps(r.value, ensure_ascii=False)}" for r in taken) or "nothing"
    params = _scrub({"method": "ai" if ai else "rules", "requirements": [{"id": r.id, "key": r.key, "value": r.value, "provenance": r.provenance, "source": r.source} for r in taken]})
    eng.decide(subject=SUBJECT, type="REVIEW" if ai else "KEEP", decision=f"brief read as {keys}",
               reason=("an AI read the brief; every key was checked like --set and quotes the brief, but the reading is the AI's: confirm it "
                       "(decisions that rest on these keys are CONFIRM)") if ai else
                      "the rule-based reading of the brief (unambiguous phrases only); every key was checked like --set",
               confidence=1.0, evidence=[brief.id] + [r.id for r in taken], risk="LOW", approval="CONFIRM" if ai else "AUTO",
               provenance="AI_GENERATED" if ai else "USER", params=params, requirements=[brief] + taken)
