"""Long inputs (ADR-132; roadmap step 23, media-analysis-skill step 11): how a plan measures the transcript of a recording of 30 min or more.

    media-analysis-skill speech_regions (media-analysis/vad)  →  transcription-skill transcribe(chunk_seconds=600, speech_regions=<the regions>)

transcription-skill 0.11.2 cuts a long recording into chunks at pauses and recognises them one by one, each cached as soon as it is done, so a
stop does not start the recognition over (its ADR-042). Its cuts land in the longest pause between two of media-analysis-skill's speech
regions near each target, else at the quietest point there. This module decides only whether a source is chunked and with what. Neither
Skill's work is done here.

- **Long**: the probed duration is at least `LONG_INPUT_SECONDS`. Shorter inputs are transcribed in one pass, as before.
- **Chunked** when the plan names the speech's language (`plan --language`; transcription-skill recognises each chunk on its own, so it
  refuses chunking without one) and the installed contract takes `chunk_seconds`. Otherwise the input is transcribed in one pass and the
  plan says why (`reason`).
- **Speech regions** are measured first when the registry selected a tool for them (`speech_region_analysis`). Regions the model did not
  measure (UNKNOWN without media-analysis-skill's optional extra `vad`) are not sent, and the Skill falls back to its level cut, as the
  plan says. Only the regions' time ranges go to transcribe, inline (never a path); the observation id is recorded in the transcript's
  parameters.
- Recorded per source in the analysis row (`longform`): duration, chunked or not and why, chunk seconds, where the cuts come from.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

LONG_INPUT_SECONDS = 1800.0     # 30 min: below this one pass is quick enough and a stop costs little
CHUNK_SECONDS = 600             # transcription-skill's own example for 2-8 h recordings (10 min chunks)
RULE = "longform_transcript@1"


def plan_source(duration: float, language: Optional[str], declares: Callable[[str], bool], regions_tool: bool) -> Optional[Dict[str, Any]]:
    """None for a short input; otherwise {rule, duration, chunked, chunk_seconds?, cuts_from, reason?}."""
    if not duration or duration < LONG_INPUT_SECONDS:
        return None
    out: Dict[str, Any] = {"rule": RULE, "duration": round(float(duration), 3), "long_input_seconds": LONG_INPUT_SECONDS, "chunked": False}
    lang = str(language or "").strip().lower()
    if not declares("chunk_seconds"):
        out["reason"] = "the installed transcription-skill does not chunk long recordings (0.11.2 or later); transcribed in one pass"
        return out
    if lang in ("", "auto"):
        out["reason"] = "chunked recognition needs the speech's language (plan --language); transcribed in one pass"
        return out
    out.update(chunked=True, chunk_seconds=CHUNK_SECONDS)
    if not declares("speech_regions"):
        out["cuts_from"] = "level"
        out["cuts_reason"] = "the installed transcription-skill takes no speech_regions"
    elif not regions_tool:
        out["cuts_from"] = "level"
        out["cuts_reason"] = "no speech_regions tool selected here (media-analysis-skill)"
    else:
        out["cuts_from"] = "speech_regions"
    return out


def regions_of(observation_data: Any) -> Optional[List[Dict[str, float]]]:
    """The measured regions' time ranges, or None when the model did not measure them (status UNKNOWN, no `regions`)."""
    regions = observation_data.get("regions") if isinstance(observation_data, dict) else None
    if not isinstance(regions, list) or (isinstance(observation_data, dict) and str(observation_data.get("status") or "").upper() == "UNKNOWN"):
        return None
    out = []
    for r in regions:
        if isinstance(r, dict) and isinstance(r.get("start"), (int, float)) and isinstance(r.get("end"), (int, float)) and 0 <= r["start"] <= r["end"]:
            out.append({"start": float(r["start"]), "end": float(r["end"])})
    return out


def transcribe_params(plan: Dict[str, Any], regions: Optional[List[Dict[str, float]]]) -> Dict[str, Any]:
    """The transcribe inputs for a chunked source: chunk_seconds, and the regions when they were measured."""
    params: Dict[str, Any] = {"chunk_seconds": plan["chunk_seconds"]}
    if plan.get("cuts_from") == "speech_regions":
        if regions is None:
            plan["cuts_from"] = "level"
            plan["cuts_reason"] = "speech regions not measured (media-analysis-skill's optional extra vad is missing, or the measurement failed)"
        else:
            params["speech_regions"] = {"kind": "speech_regions", "data": {"regions": regions}}
    return params
