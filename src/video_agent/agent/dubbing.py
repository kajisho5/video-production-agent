"""Dubbing the programme's dialogue (ADR-085; dubbing-skill roadmap step 9, agent side): dubbing-skill's DUB, called as its ADR-012 hands it over.

    Requirement (`--set dub.cues=<translated cues> dub.lang=<L>`, explicit only) → Decision `speech.dub` (always CONFIRM: EXPERIMENTAL, a
    synthetic voice replaces a person's words) → step `speech_dub` → IR audio operation `speech.dub` → compiler → dubbing-skill `dub`
    (its own engine worker; every media step through the media engine Skill)

Vocabulary, decisions and checks only, no execution:

- **Requirements** (`dub.` keys; never from a profile, a brief or a default -- the user opts in on the request, `--set`):
  `dub.cues` (required: the user's translated cues, .json / .srt / .vtt, one per source segment on the programme's timeline -- nobody here
  translates, dubbing-skill ADR-003) and `dub.lang` (required, BCP-47); optional `dub.source_cues` (the cues the translation is of, checked
  by the Skill to be the same segments), `dub.engine` / `dub.voice`, `dub.duck` / `dub.voice_gain` / `dub.max_overrun` (the contract's
  limits), `dub.on_unfit` (fail | keep), and `dub.reference` + `dub.consent` (a reference voice, only together with its owner's consent
  record -- both files the user names; the agent never writes, edits or signs a record, and a reference without one is refused here).
  Every range and enum is read from the pinned dubbing contract (tools/dubbing/contract_0.1.0.json), never copied.
- **Where in the chain**: on the subject's picture after its edits (trim / concat / speed / ... -- the cues are on that timeline) and
  before colour / graphics / subtitles, a joined still, loudness and the deliverables: the dub output is the programme with its audio
  replaced (video stream-copied), so everything after it -- the loudness normalisation included -- works on the dubbed sound, and a
  joined still (ADR-070) cannot shift the cues.
- **Rules the plan checks before any engine runs** (each a BLOCK, with dubbing-skill's own error code where it has one): a picture to dub
  (not the audio production path, which delivers no picture: not wired); one programme (one cue file names one timeline); DUB's
  engine ready in dubbing-skill's doctor (`ENGINE_UNAVAILABLE` / `MODEL_NOT_FOUND`, or the Skill's code for its media engine, with its install line,
  never a download); a reference voice only on an engine that clones (the doctor's `voice_cloning`). Without dubbing-skill the step has
  no tool and the plan's handoff names the package (ADR-051).
- **What the result carries**: `obligations.credit` goes into the ProductionReceipt (audit/receipt.py) with the engine's watermark, and a
  `CONSENT_REQUIRED` at render is reported with the Skill's reason and problems and never retried (execution/recovery.py)."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models import Requirement
from ..tools.dubbing.adapter import (CONSENT_EXTENSIONS, CUES_EXTENSIONS, DOCTOR_CODES, LANG_RE, REFERENCE_EXTENSIONS, VOICE_RE, readiness_code, engines as contract_engines,
                                     parameters as contract_parameters, unfit_policies)
from .audio import SWITCH as AUDIO_SWITCH
from .editing import EditRequirementError, _bool

PREFIX = "dub."
PACKAGE = "dubbing"
OP = "speech.dub"
SKILL = "speech_dub"
TYPE = "DUB"
CAP_TYPED = f"{PACKAGE}:{TYPE}"
FILES = {"dub.cues": CUES_EXTENSIONS, "dub.source_cues": CUES_EXTENSIONS, "dub.reference": REFERENCE_EXTENSIONS, "dub.consent": CONSENT_EXTENSIONS}
NUMBERS = {f"dub.{k}": k for k in contract_parameters()}   # dub.duck / dub.voice_gain / dub.max_overrun → the contract's DUB option
REQUIREMENT_KEYS = ("dub.cues", "dub.lang", "dub.source_cues", "dub.engine", "dub.voice", "dub.on_unfit", "dub.reference", "dub.consent") + tuple(NUMBERS)
# the IR operation's parameters (beside type / asset / input / output / temporal_scope / decision_ids): what the compiler copies by name
PARAMS = ("cues", "lang", "source_cues", "engine", "voice", "on_unfit", "reference", "consent") + tuple(NUMBERS.values())
# always CONFIRM (floor CONFIRM: the explicit requirement does not waive it -- DUB is EXPERIMENTAL and a synthetic voice replaces someone's words)
APPROVAL_KEYS = {OP: (f"{OP}.approval", "CONFIRM")}


def _number(v: Any, key: str, lo: float, hi: float) -> float:
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            raise EditRequirementError(f"{key} must be a number, got {v!r}")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        raise EditRequirementError(f"{key} must be a finite number, got {v!r}")
    if not lo <= float(v) <= hi:
        raise EditRequirementError(f"{key} must be within [{lo:g}, {hi:g}] (dubbing-skill's own limits), got {v!r}")
    return float(v)


def _file(v: Any, key: str, exts: Tuple[str, ...]) -> str:
    if not isinstance(v, str) or not v.strip() or "\x00" in v or "\n" in v:
        raise EditRequirementError(f"{key} must be a path to a file")
    if any(part == ".." for part in v.replace("\\", "/").split("/")):
        raise EditRequirementError(f"{key} path contains '..'")
    p = os.path.realpath(os.path.abspath(v))
    if os.path.splitext(p)[1].lower() not in exts:
        raise EditRequirementError(f"{key} must be one of {', '.join(exts)}: {os.path.basename(v)}")
    if not os.path.isfile(p):
        raise EditRequirementError(f"{key}: no such file: {os.path.basename(v)}")
    return p


def parse_dub_requirements(m: Dict[str, Requirement]) -> Optional[Dict[str, Any]]:
    """Requirement map → {"params": {...}, "requirements": [Requirement]} or None when nothing is asked. Only USER requirements count; any
    other provenance of a `dub.*` key is refused (dubbing is opt-in per request, never a profile default or a brief's reading)."""
    given = {k: r for k, r in m.items() if k.startswith(PREFIX)}
    if not given:
        return None
    unknown = sorted(k for k in given if k not in REQUIREMENT_KEYS)
    if unknown:
        raise EditRequirementError(f"unknown requirement key(s): {', '.join(unknown)}; dub.* keys are {', '.join(REQUIREMENT_KEYS)}")
    not_user = sorted(k for k, r in given.items() if r.provenance != "USER")
    if not_user:
        raise EditRequirementError(f"{', '.join(not_user)}: dubbing is opt-in on the request (--set), never from a profile, a brief or a default")
    for k in ("dub.cues", "dub.lang"):
        if k not in given:
            raise EditRequirementError(f"{sorted(given)[0]} needs dub.cues (the translated cues, from the user: nothing here translates) and dub.lang; {k} is missing")
    lang = given["dub.lang"].value
    if not isinstance(lang, str) or not LANG_RE.match(lang):
        raise EditRequirementError(f"dub.lang {lang!r} is not a BCP-47 language tag")
    params: Dict[str, Any] = {"lang": lang}
    for k, exts in FILES.items():
        if k in given:
            params[k.split(".", 1)[1]] = _file(given[k].value, k, exts)
    if ("reference" in params) != ("consent" in params):
        raise EditRequirementError("dub.reference needs dub.consent (the voice owner's consent record, dubbing/consent@1, from the user) and dub.consent needs "
                                   "dub.reference: a reference voice is never passed without the record, and the agent never writes one")
    if "reference" in params and os.path.normcase(params["reference"]) == os.path.normcase(params["consent"]):
        raise EditRequirementError("dub.consent cannot be the reference voice itself")
    if "dub.engine" in given:
        eng = given["dub.engine"].value
        if eng not in contract_engines():
            raise EditRequirementError(f"dub.engine {eng!r} must be one of {', '.join(contract_engines())} (dubbing-skill's DUB engines)")
        params["engine"] = eng
    if "dub.voice" in given:
        v = given["dub.voice"].value
        if not isinstance(v, str) or not VOICE_RE.match(v):
            raise EditRequirementError(f"dub.voice {v!r} is not a voice id")
        params["voice"] = v
    if "dub.on_unfit" in given:
        u = given["dub.on_unfit"].value
        if u not in unfit_policies():
            raise EditRequirementError(f"dub.on_unfit {u!r} must be one of {', '.join(unfit_policies())}")
        params["on_unfit"] = u
    rng = contract_parameters()
    for k, name in NUMBERS.items():
        if k in given:
            params[name] = _number(given[k].value, k, *rng[name])
    return {"params": params, "requirements": [given[k] for k in REQUIREMENT_KEYS if k in given]}


def check_dub_keys(user_requirements: Dict[str, Any]) -> None:
    """The `--set` gate for `dub.*` (before any analysis runs): known keys, files that exist, values in range, reference and consent together."""
    if not any(k.startswith(PREFIX) for k in user_requirements or {}):
        return
    parse_dub_requirements({k: Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in user_requirements.items() if k.startswith(PREFIX)})


def ir_dub_operation(subject: str, params: Dict[str, Any], decision_ids: List[str], current: str, out_id: str, scope: Dict[str, float]) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"type": OP, "asset": subject, "input": current, "output": out_id}
    for k in PARAMS:
        if params.get(k) is not None:
            rec[k] = params[k]
    rec["temporal_scope"] = {"start": round(float(scope["start"]), 3), "end": round(float(scope["end"]), 3)}
    rec["decision_ids"] = list(decision_ids)
    return rec


def user_files(params: Dict[str, Any]) -> List[str]:
    """The files of a dub the user named (cues, source cues, reference voice, consent record): their directories become allowed input roots."""
    return [params[k] for k in ("cues", "source_cues", "reference", "consent") if params.get(k)]


def _words(p: Dict[str, Any]) -> str:
    bits = [f"cues {os.path.basename(p['cues'])} ({p['lang']})"]
    if p.get("engine") or p.get("voice"):
        bits.append("voice " + "/".join(str(x) for x in (p.get("engine"), p.get("voice")) if x))
    if p.get("reference"):
        bits.append(f"reference voice {os.path.basename(p['reference'])} with its consent record {os.path.basename(p['consent'])}")
    return ", ".join(bits)


def decide_dub(eng: Any, m: Dict[str, Requirement], analysis: Any, subjects: List[Tuple[str, List[str]]], caps: Dict[str, Any],
               approval_for: Callable[..., Dict[str, Any]], probe_ids_of: Callable[[List[str]], List[str]], cap_block: Callable[[str, str], Any]) -> None:
    """One `speech.dub` decision: TRANSFORM with a CONFIRM floor, or BLOCK with the reason (and dubbing-skill's own code in `params.code` where
    it has one). Without the package the TRANSFORM stands and cap_block adds the BLOCK whose step the ADR-051 handoff names."""
    dreq = parse_dub_requirements(m)
    if dreq is None:
        return
    p, req = dreq["params"], dreq["requirements"]
    pkg = caps.get(PACKAGE)
    pkg_ok = pkg is not None and getattr(pkg, "status", "") == "AVAILABLE"
    sw = m.get(AUDIO_SWITCH)
    audio_production = sw is not None and sw.provenance != "DEFAULT" and _bool(sw.value, AUDIO_SWITCH)

    def block(subject: str, sources: List[str], why: str, code: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        ev = [r.id for r in req] + probe_ids_of(sources) + ([f"capability:{CAP_TYPED}"] if code in DOCTOR_CODES and (extra or {}).get("from_doctor") else [])
        eng.decide(subject=OP, type="BLOCK", decision=f"BLOCK: {OP} on {subject}", reason=why, confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK", provenance="USER",
                   params={"asset_id": subject, **({"code": code} if code else {}), **{k: v for k, v in (extra or {}).items() if k != "from_doctor"}}, requirements=req)

    if audio_production:
        block(subjects[0][0] if subjects else "programme", [], "dubbing replaces the dialogue under the picture; the audio production path delivers no picture "
              "(dubbing the audio deliverable is not wired)")
        return
    if len(subjects) != 1:
        block(subjects[0][0] if subjects else "programme", [s for _, srcs in subjects for s in srcs],
              f"{len(subjects)} separate deliverables: one translated cue file is on one programme's timeline (join the inputs into one programme, or dub them one per request)")
        return
    subject, sources = subjects[0]
    src = [a for a in analysis.assets if a.id in sources]
    if not src or not all(a.technical.get("video") for a in src):
        block(subject, sources, f"{subject} has no video stream: DUB puts the dubbed speech back under the picture")
        return
    if not any((a.technical or {}).get("audio") for a in src):
        block(subject, sources, f"{subject} has no audio stream: DUB speaks over the programme's own sound, ducked", "UNSUPPORTED_MEDIA")
        return
    if pkg_ok:
        c = caps.get(CAP_TYPED)
        ev = dict(getattr(c, "evidence", {}) or {}) if c is not None else {}
        engine = p.get("engine") or ev.get("engine")
        row = (ev.get("engines") or {}).get(engine) or {}
        status = row.get("status") if p.get("engine") else ev.get("engine_status")
        if p.get("reference") and row and row.get("voice_cloning") is False:
            block(subject, sources, f"engine {engine} cannot clone a voice: a reference voice is refused (dubbing-skill: INVALID_ARGUMENT, whatever the consent record says)",
                  "INVALID_ARGUMENT", {"engine": engine})
            return
        if status != "OK":
            code = readiness_code(ev, engine)
            install = row.get("install") or ev.get("install") or ""
            detail = row.get("detail") or ev.get("detail") or ""
            alt = [a for a in ev.get("alternatives") or [] if (ev.get("engines") or {}).get(a, {}).get("status") == "OK"]
            block(subject, sources, f"dubbing-skill reports {code} for dub: engine {engine or '?'} is {status or 'not ready'}" + (f" ({detail})" if detail else "")
                  + (f"; install: {install}" if install else "") + (f"; ready instead: {', '.join(alt)} (dub.engine)" if alt else "")
                  + "; nothing is downloaded or installed without the user's go-ahead", code, {"engine": engine, "from_doctor": True})   # the install line stays in the reason (a command line is never a parameter)
            return
    ev = [r.id for r in req] + probe_ids_of(sources)
    eng.decide(subject=OP, type="TRANSFORM", decision=f"dub the dialogue of {subject}: {_words(p)}",
               reason=f"user asked for dubbing ({req[0].source}) with translated cues they supplied (dubbing-skill never translates and neither does the agent); "
                      "dubbing-skill speaks each line with a local engine, fitted to its source segment at tempo 1.0, over the programme's own sound lowered while the line "
                      "is spoken, and puts it back under the picture (video stream-copied); after the edits (the cues are on that timeline) and before colour, graphics, "
                      "subtitles and loudness; the engine's credit and watermark obligations go into the ProductionReceipt; "
                      + ("the reference voice is used only with the consent record the user named, checked by the Skill before any engine work (no bypass); "
                         if p.get("reference") else "")
                      + "dubbing-skill marks DUB EXPERIMENTAL, so it always needs confirmation and the result should be listened to",
               confidence=1.0, evidence=ev, risk="HIGH", approval=approval_for(OP, floor="CONFIRM", keys=APPROVAL_KEYS), provenance="USER",
               params={"asset_id": subject, **p, "lifecycle": "EXPERIMENTAL"}, requirements=req, serves_intent=None)
    cap_block(SKILL, f"capability.{SKILL}")
