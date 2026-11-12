"""Opt-in transcription options in plans (ADR-113): which of transcription-skill's engines and models recognises the plan's sources, and
whether Japanese words are segmented by a dictionary. Each is sent only when the user sets it with `--set`, and only to a contract that
declares it. A combination the Skill would refuse is refused before anything runs, with the Skill's own reasons.

    --set transcribe.engine=<engine id>                 e.g. qwen3_asr, speechcpp: one of the installed contract's engines
    --set transcribe.model=<model name>                 e.g. parakeet-tdt-0.6b-v3, distil-large-v3.5: a name, never a path
    --set transcribe.word_segmentation=engine|dictionary   dictionary: Japanese words from a dictionary (transcription-skill 0.9.0+)

- **Explicit only.** These are not brief keys and never stored preferences. Which recogniser and model run is the user's choice, and so is
  the language of the speech (a model may recognise only some languages). The agent chooses none of them.
- **Defaults unchanged.** Without the keys no engine, model or segmentation is sent, so the Skill picks its default engine and model as
  before, and the plan records nothing new.
- **Before any analysis** (Service.plan), in this order:
  1. each key and value (the --set gate, `parse_transcribe_requirements`);
  2. the plan measures a transcript: something in it recognises speech. Otherwise the options would do nothing (`check_plan`);
  3. dictionary segmentation's rule, as transcription-skill states it: language ja (`dictionary_segmentation_language`) and word
     timestamps (`word_timestamps_required`), which the plan measures only for a requirement that uses the words' timings;
  4. the installed contract declares each option: the transcribe tool's input names it, the engine is one of the contract's engines, and
     dictionary segmentation is a declared capability (`check_contract_declares`);
  5. the Skill's own dry run of the request the analysis will send. It runs every check a run makes before any audio work, such as a model's
     language limit (`language_unsupported` / `language_detection_unsupported`), and recognises nothing. A refused request (INVALID_INPUT, or a
     model the engine does not know) refuses the plan with the Skill's code, reason and message, and the adapter's hint
     (`preflight_refusal`). Anything else (an engine or a model not installed here, an input without audio, the input path policy, a
     time-out) is left to the analysis, which records it as it always has.
- **Provenance**: IR `provenance.transcription_options` and the job's provenance.json record the options, where they came from, the
  dry run's verdict and, per source, the engine / model (and segmentation) the Skill reports it used.
- **Revisions** re-use the recorded analysis, so a revision cannot change these options: plan again.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

PREFIX = "transcribe."
ENGINE_KEY = "transcribe.engine"
MODEL_KEY = "transcribe.model"
WORD_SEGMENTATION_KEY = "transcribe.word_segmentation"
KEYS = (ENGINE_KEY, MODEL_KEY, WORD_SEGMENTATION_KEY)
OPTION_OF = {ENGINE_KEY: "engine", MODEL_KEY: "model", WORD_SEGMENTATION_KEY: "word_segmentation"}
WORD_SEGMENTATIONS = ("engine", "dictionary")       # transcription-skill's vocabulary; engine is its default
DICTIONARY = "dictionary"
DICTIONARY_LANGUAGE = "ja"                          # transcription-skill: dictionary segmentation is for Japanese only
DICTIONARY_CAPABILITY = "dictionary_word_segmentation"
RULE = "transcribe_options@1"
SKILL = "speech_transcription"
# the same name rules as the transcription adapter (an engine id, a model name: never a path)
_ENGINE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# the requirements that measure word timestamps (what dictionary segmentation maps onto dictionary words)
WORD_TIMING_KEYS = ("edit.word_cut", "verify.transcript", "audio.filler_remove", "subtitle.caption_profile")
# transcription-skill's input path policy reasons: about where the files are, not about the options, so left to the analysis
PATH_REASONS = ("traversal", "outside_allowed_roots", "symlink_escape", "not_regular_file", "root_not_directory", "workspace_escape", "would_overwrite_input")


def parse_transcribe_requirements(reqs: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """`transcribe.*` keys → {engine?, model?, word_segmentation?}, or None when none is set. Unknown keys and values that are not an engine id,
    a model name or one of transcription-skill's segmentation modes are refused (ValueError)."""
    keys = [k for k in (reqs or {}) if str(k).startswith(PREFIX)]
    unknown = sorted(k for k in keys if k not in KEYS)
    if unknown:
        raise ValueError(f"unknown requirement key(s): {', '.join(unknown)} (transcribe.* keys: {', '.join(KEYS)})")
    if not keys:
        return None
    out: Dict[str, str] = {}
    for k in KEYS:
        if k not in reqs:
            continue
        v = reqs[k]
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{k} must be a name, got {v!r}")
        v = v.strip()
        if k == ENGINE_KEY and not _ENGINE_RE.match(v):
            raise ValueError(f"{k} must be an engine id of transcription-skill (letters, digits, '_' and '-'), got {v!r}")
        if k == MODEL_KEY and not _MODEL_RE.match(v):
            raise ValueError(f"{k} must be a model name, never a path (letters, digits, '.', '_' and '-'), got {v!r}")
        if k == WORD_SEGMENTATION_KEY:
            v = v.lower()
            if v not in WORD_SEGMENTATIONS:
                raise ValueError(f"{k} must be one of {' | '.join(WORD_SEGMENTATIONS)} (transcription-skill's modes), got {reqs[k]!r}")
        out[OPTION_OF[k]] = v
    return out


def check_plan(opts: Dict[str, str], kinds: List[str], params: Dict[str, Any]) -> None:
    """What the options need from the plan itself (ValueError): a transcript the plan measures, no other value for the same parameter, and
    dictionary segmentation's rule as transcription-skill states it (language ja, word timestamps)."""
    if "transcript" not in (kinds or []):
        raise ValueError(f"{', '.join(PREFIX + k for k in opts)} choose(s) how speech is recognised, and nothing in this plan recognises speech "
                         "(subtitles, edit.word_cut, audio.filler_remove, edit.takes, verify.transcript, speakers.assign or --kind transcript): refusing an option that does nothing")
    for k in ("engine", "model", "word_segmentation"):
        given = (params or {}).get(k)
        if k in opts and given is not None and given != opts[k]:
            raise ValueError(f"{PREFIX}{k}={opts[k]} and the {k} parameter {given!r} disagree: give one of them")
    if opts.get("word_segmentation") == DICTIONARY:
        lang = str((params or {}).get("language") or "").strip().lower()
        if lang != DICTIONARY_LANGUAGE:
            raise ValueError(f"{WORD_SEGMENTATION_KEY}=dictionary is for Japanese only: transcription-skill refuses it without language ja "
                             f"(dictionary_segmentation_language; got {lang or 'no language, i.e. detection'}); plan with --language ja, or leave the key out")
        if (params or {}).get("word_timestamps") is not True:
            raise ValueError(f"{WORD_SEGMENTATION_KEY}=dictionary maps the recognised word timings onto dictionary words, and transcription-skill refuses it "
                             f"without word timestamps (word_timestamps_required). This plan does not measure them: they come with a requirement that uses "
                             f"the words' timings ({', '.join(WORD_TIMING_KEYS)})")


def check_contract_declares(opts: Dict[str, str], contract: Dict[str, Any], tool: str) -> None:
    """Each option only to a contract that declares it (ValueError): the transcribe tool's input names it, the engine is one of the contract's
    engines, and dictionary segmentation is a declared capability."""
    version = str(contract.get("version") or "?")
    declared = next((t.get("input") or {} for t in contract.get("tools") or [] if t.get("name") == tool), {})
    for k in opts:
        if k not in declared:
            raise ValueError(f"{PREFIX}{k}: the installed transcription-skill {version} does not take {k} (its {tool} input does not declare it)"
                             + ("; dictionary segmentation needs transcription-skill 0.9.0 or later" if k == "word_segmentation" else ""))
    engines = [str(e.get("id")) for e in contract.get("engines") or [] if isinstance(e, dict)]
    if "engine" in opts and opts["engine"] not in engines:
        raise ValueError(f"{ENGINE_KEY}={opts['engine']}: the installed transcription-skill {version} declares no such engine (it declares {', '.join(engines) or 'none'})")
    if opts.get("word_segmentation") == DICTIONARY and DICTIONARY_CAPABILITY not in (contract.get("capabilities") or []):
        raise ValueError(f"{WORD_SEGMENTATION_KEY}=dictionary: the installed transcription-skill {version} does not declare the {DICTIONARY_CAPABILITY} capability")


def preflight_refusal(error: Dict[str, Any]) -> Optional[str]:
    """The plan's refusal for a failed dry run, or None when the failure is left to the analysis. Refused: the Skill (or the adapter, before any
    process) refusing the request itself (INVALID_INPUT, except the input path policy's reasons), and a model the engine does not know
    (MODEL_UNAVAILABLE, availability MODEL_UNKNOWN). The message keeps the Skill's code, reason and message, and the adapter's hint."""
    details = error.get("details") if isinstance(error.get("details"), dict) else {}
    code = str(error.get("code") or "")
    if code == "INVALID_INPUT" and details.get("reason") in PATH_REASONS:
        return None
    if not (code == "INVALID_INPUT" or (code == "MODEL_UNAVAILABLE" and details.get("availability") == "MODEL_UNKNOWN")):
        return None
    why = code + (f" {details['reason']}" if details.get("reason") else "") + (f" {details['availability']}" if code == "MODEL_UNAVAILABLE" else "")
    msg = f"transcription-skill refuses this transcription request before anything runs ({why}): {str(error.get('message') or '')[:400]}"
    if error.get("hint"):
        msg += f" -- {str(error['hint'])[:400]}"
    return msg


def measured(analysis_rows: List[Dict[str, Any]], observations: List[Any], opts: Dict[str, str]) -> List[Dict[str, Any]]:
    """Per source, what the Skill reports it used (engine, model, and the segmentation when one was asked for), or why there is no transcript."""
    by_asset = {o.asset_id: o for o in observations if getattr(o, "kind", None) == "transcript"}
    out: List[Dict[str, Any]] = []
    for row in analysis_rows:
        if row.get("kind") != "transcript":
            continue
        o = by_asset.get(row.get("asset_id"))
        if o is None:
            err = row.get("error") or {}
            out.append({"asset_id": row.get("asset_id"), "status": row.get("status"), "error": {k: err[k] for k in ("kind", "skill_error", "recovery_class") if err.get(k)}})
            continue
        p = o.parameters or {}
        rec: Dict[str, Any] = {"asset_id": o.asset_id, "status": "OK", "engine": p.get("engine"), "model": p.get("model")}
        if "word_segmentation" in opts:   # transcription-skill records the mode only when it is not its default
            rec["word_segmentation"] = p.get("word_segmentation") or "engine"
        out.append(rec)
    return out


def check_revision(recorded: Dict[str, Any], given: Dict[str, Any]) -> None:
    """A revision re-uses the recorded analysis: a transcribe.* key it sets must be the value the plan recorded (ValueError otherwise)."""
    for k in sorted(k for k in (given or {}) if str(k).startswith(PREFIX)):
        if (recorded or {}).get(k) != given[k]:
            raise ValueError(f"{k} chooses how the sources are recognised, which happens when a plan is made; a revision re-uses the recorded "
                             f"analysis: plan again with --set {k}={given[k]}")
