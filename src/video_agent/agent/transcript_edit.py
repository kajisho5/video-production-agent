"""Editing by transcript (ADR-123, Velisa ADR-032): the user picks sentences or words, the agent shows what goes, a yes cuts it.

    the user's selection (word index ranges / segment ids, remove | keep)
      → transcription-skill `transcription/select`: the removed spans and the gap around each (rule transcript-selection@1)
      → video-editing-skill `ripple`: each cut placed in its gap, on a measured silence, on a frame (rule ripple@1); the JSON EDL and
        the transcript / subtitles retimed as a preview
      → the transcript-edit file (`video-agent/transcript-edit@1`) and the list shown to the user: which sentences, by whom, how many seconds
      → `--set edit.transcript_edit=<file>` → a `transcript.ripple` decision (CONFIRM at least: removing speech is a semantic deletion)
      → the asset's trim runs on video-editing-skill CONDENSE with the frame-exact cuts → after the render, `ripple` retimes the transcript
        and subtitles to the cuts the plan actually made

The selection is the user's: the agent never chooses the words here (the AI path is `edit.word_cut`, ADR-109). Nothing is cut before the
user approves the decision. Everything in this module is vocabulary, file checks and arithmetic over recorded documents.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

KEY = "edit.transcript_edit"
SUBJECT = "transcript.ripple"
DOC_SCHEMA = "video-agent/transcript-edit@1"
SELECT_RULE = "transcript-selection@1"
RIPPLE_RULE = "ripple@1"
LABEL = "transcript"
APPROVAL_KEY = ("edit.semantic_deletion.approval", "CONFIRM")   # the profile's policy for removing speech (conference: BLOCK), CONFIRM at least
RISK = "HIGH"
MODES = ("remove", "keep")
_RANGE = re.compile(r"^\s*(\d{1,7})\s*(?:-\s*(\d{1,7}))?\s*$")
_SEG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class TranscriptEditError(ValueError):
    """A selection, a transcript-edit file or a plan that cannot carry it: refused, never corrected."""


# ---- the selection the user gives -------------------------------------------------------------------------------------
def parse_word_ranges(text: Any) -> List[List[int]]:
    """'12-40,55' or [[12, 40], [55, 55]] -> [[12, 40], [55, 55]] (inclusive word indices)."""
    if isinstance(text, list):
        out = text
    elif isinstance(text, str):
        out = []
        for part in [p for p in text.split(",") if p.strip()]:
            m = _RANGE.match(part)
            if not m:
                raise TranscriptEditError(f"word range {part.strip()!r} is not N or N-M (word indices)")
            out.append([int(m.group(1)), int(m.group(2) or m.group(1))])
    else:
        raise TranscriptEditError("word ranges must be 'N-M,...' or [[first, last], ...]")
    if not out or not all(isinstance(r, list) and len(r) == 2 and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in r) and r[0] <= r[1]
                          for r in out):
        raise TranscriptEditError("word ranges must be forward [first, last] pairs of word indices")
    return [list(r) for r in out]


def parse_segment_ids(text: Any) -> List[str]:
    ids = [x.strip() for x in text.split(",") if x.strip()] if isinstance(text, str) else text
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and _SEG.match(x) for x in ids):
        raise TranscriptEditError("segments must be segment ids ('seg_0003,seg_0007' or a list)")
    return list(ids)


# ---- the file the preview writes and the plan reads --------------------------------------------------------------------
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_document(source: str, transcript_path: str, subtitles: Sequence[str], selection: Dict[str, Any], ripple: Dict[str, Any]) -> Dict[str, Any]:
    """The transcript-edit file: the user's selection, what transcription-skill and video-editing-skill made of it, and the list the user
    approves (per cut: its sentences, speakers, text and frame-exact seconds)."""
    sentences = {s["segment_id"]: s for s in selection.get("sentences") or [] if isinstance(s, dict) and s.get("segment_id")}
    cuts = []
    for c in ripple.get("cuts") or []:
        segs = [s for s in c.get("segment_ids") or [] if s in sentences]
        cuts.append({"id": c["id"], "start": float(c["source_in"]["seconds"]), "end": float(c["source_out"]["seconds"]),
                     "frames": [c["source_in"]["frame"], c["source_out"]["frame"]], "seconds": float(c["seconds"]), "text": c.get("text") or "",
                     "speaker_ids": list(c.get("speaker_ids") or []),
                     "sentences": [{k: sentences[s].get(k) for k in ("segment_id", "speaker_id", "removed", "removed_text", "text")} for s in segs]})
    return {
        "schema": DOC_SCHEMA,
        "applied": False,
        "source": {"path": os.path.abspath(source), "sha256": _sha256(source)},
        "transcript": {"path": os.path.abspath(transcript_path), "sha256": _sha256(transcript_path), "id": selection.get("transcript_id")},
        "subtitles": [{"path": os.path.abspath(p), "sha256": _sha256(p)} for p in subtitles],
        "selection": {"mode": selection.get("mode"), "overlap": selection.get("overlap"), "words": (selection.get("selection") or {}).get("words"),
                      "segments": (selection.get("selection") or {}).get("segments"), "rule": SELECT_RULE},
        "select_result": selection,
        "ripple": {k: ripple.get(k) for k in ("rule", "rate", "margins", "silence", "source", "cuts", "dropped", "removed", "record", "edl", "transcript",
                                               "subtitles", "warnings", "skill")},
        "cuts": cuts,
        "summary": {"cuts": len(cuts), "removed_seconds": float((ripple.get("removed") or {}).get("seconds") or 0.0),
                    "removed_frames": (ripple.get("removed") or {}).get("frames"), "rate": ripple.get("rate"),
                    "duration_before": float((ripple.get("source") or {}).get("seconds") or 0.0),
                    "duration_after": float((ripple.get("record") or {}).get("seconds") or 0.0),
                    "sentences_whole": sum(1 for s in sentences.values() if s.get("removed") == "whole"),
                    "sentences_part": sum(1 for s in sentences.values() if s.get("removed") == "part"),
                    "speakers": sorted({x for c in cuts for x in c["speaker_ids"]})},
    }


def approval_lines(doc: Dict[str, Any], max_text: int = 120) -> List[str]:
    """What the user is shown before anything is cut: each cut with its seconds, speakers and the sentences it takes, then the total."""
    lines = []
    for c in doc.get("cuts") or []:
        who = ", ".join(c.get("speaker_ids") or []) or "-"
        text = (c.get("text") or "").replace("\n", " ")
        text = text if len(text) <= max_text else text[: max_text - 1] + "…"
        whole = [s["segment_id"] for s in c.get("sentences") or [] if s.get("removed") == "whole"]
        part = [s["segment_id"] for s in c.get("sentences") or [] if s.get("removed") == "part"]
        where = (f"whole {', '.join(whole)}" if whole else "") + ("; " if whole and part else "") + (f"part of {', '.join(part)}" if part else "")
        lines.append(f"{c['id']}  {c['start']:9.3f}-{c['end']:9.3f}s  {c['seconds']:7.3f}s  [{who}]  {where or 'no words (lead-in / tail)'}  \"{text}\"")
    s = doc.get("summary") or {}
    lines.append(f"total: {s.get('cuts', 0)} cut(s), {float(s.get('removed_seconds') or 0):.3f}s ({s.get('removed_frames')} frames at {s.get('rate')}); "
                 f"{float(s.get('duration_before') or 0):.3f}s -> {float(s.get('duration_after') or 0):.3f}s; "
                 f"{s.get('sentences_whole', 0)} whole and {s.get('sentences_part', 0)} partial sentence(s)")
    return lines


def parse_requirement(path: Any) -> Dict[str, Any]:
    """`edit.transcript_edit`: the transcript-edit file the preview wrote. Refused when it is not one, says it was applied, or its source /
    transcript changed since (sha256). Returns the decision params: {document, sha256, source, transcript, subtitles, spans, cuts, summary}."""
    if not isinstance(path, str) or not path.strip() or "\n" in path or "\x00" in path or any(x == ".." for x in path.replace("\\", "/").split("/")):
        raise TranscriptEditError(f"{KEY} must be the path of the file `video-agent transcript-edit` wrote")
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        doc = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as e:
        raise TranscriptEditError(f"{KEY}: cannot read the transcript-edit file: {e}")
    errs = check_document(doc)
    if errs:
        raise TranscriptEditError(f"{KEY} is not a transcript-edit file: " + "; ".join(errs[:3]))
    for what in ("source", "transcript"):
        p = doc[what]["path"]
        if not os.path.isfile(p) or _sha256(p) != doc[what]["sha256"]:
            raise TranscriptEditError(f"{KEY}: the {what} {os.path.basename(p)} is missing or changed since the preview; run `video-agent transcript-edit` again")
    for sub in doc.get("subtitles") or []:
        if not os.path.isfile(sub["path"]) or _sha256(sub["path"]) != sub["sha256"]:
            raise TranscriptEditError(f"{KEY}: the subtitles {os.path.basename(sub['path'])} are missing or changed since the preview")
    spans = [{"id": c["id"], "start": round(float(c["start"]), 6), "end": round(float(c["end"]), 6), "label": LABEL} for c in doc["cuts"]]
    return {"document": os.path.abspath(path), "sha256": hashlib.sha256(raw).hexdigest(), "source": dict(doc["source"]), "transcript": dict(doc["transcript"]),
            "subtitles": [dict(s) for s in doc.get("subtitles") or []], "spans": spans, "selection": dict(doc["selection"]),
            "cuts": [{k: c.get(k) for k in ("id", "start", "end", "frames", "seconds", "text", "speaker_ids", "sentences")} for c in doc["cuts"]],
            "summary": dict(doc["summary"]), "rules": [SELECT_RULE, RIPPLE_RULE], "rate": doc["summary"].get("rate")}


def check_document(doc: Any) -> List[str]:
    errs: List[str] = []
    if not isinstance(doc, dict) or doc.get("schema") != DOC_SCHEMA:
        return [f"schema is not {DOC_SCHEMA}"]
    if doc.get("applied") is not False:
        errs.append("applied must be false")
    for what in ("source", "transcript"):
        v = doc.get(what)
        if not isinstance(v, dict) or not isinstance(v.get("path"), str) or not re.match(r"^[0-9a-f]{64}$", str(v.get("sha256") or "")):
            errs.append(f"{what} must name a path and its sha256")
    if (doc.get("selection") or {}).get("mode") not in MODES:
        errs.append("selection.mode must be remove or keep")
    if ((doc.get("ripple") or {}).get("rule") or {}).get("version") != RIPPLE_RULE:
        errs.append(f"ripple.rule must be {RIPPLE_RULE}")
    cuts = doc.get("cuts")
    if not isinstance(cuts, list) or not cuts:
        errs.append("no cuts")
    else:
        last = -1.0
        for c in cuts:
            ok = isinstance(c, dict) and all(isinstance(c.get(k), (int, float)) and not isinstance(c.get(k), bool) for k in ("start", "end", "seconds")) \
                and last <= c["start"] < c["end"] and isinstance(c.get("id"), str)
            if not ok:
                errs.append(f"cut {c.get('id') if isinstance(c, dict) else c!r} is not a forward range in order")
                break
            last = c["end"]
    if not isinstance(doc.get("summary"), dict):
        errs.append("no summary")
    return errs


# ---- the decision ------------------------------------------------------------------------------------------------------
def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)


OTHER_RETIMING = ("video.speed", "video.freeze", "video.reverse", "video.concat", "video.switch", "video.grid", "takes.pick", "editor.cut")


def decide_transcript_edit(eng: Any, edit: Dict[str, Any], analysis: Any, approval_for: Any, probe_ids_of: Any, cap_block: Any, condense_skill: str) -> None:
    """`edit.transcript_edit` → one `transcript.ripple` TRANSFORM on the source the preview was made for (matched by sha256, then path), CONFIRM
    at least, or a BLOCK naming why not. Cuts past the source's end are refused, never clamped."""
    req = edit["requirements"]
    p = edit["params"]
    ev = [r.id for r in req] + probe_ids_of([a.id for a in analysis.assets])
    videos = [a for a in analysis.assets if (a.technical or {}).get("video")]
    src = p["source"]
    target = next((a for a in videos if (a.hash and a.hash == src["sha256"]) or _same_file(a.path, src["path"])), None)
    why = None
    if target is None:
        why = f"the transcript edit was previewed on {os.path.basename(src['path'])}, which is not a video input of this plan (matched by sha256 / path)"
    else:
        dur = float((target.technical or {}).get("duration") or 0.0)
        late = [c["id"] for c in p["cuts"] if dur and c["end"] > dur + 0.001]
        if late:
            why = f"cut(s) {', '.join(late)} end after {target.id} ({dur:.3f}s): a cut is never clamped"
    s = p["summary"]
    words = "; ".join(f"{c['id']} {c['start']:.3f}-{c['end']:.3f}s ({c['seconds']:.3f}s) \"{(c.get('text') or '')[:60]}\"" for c in p["cuts"])
    if why is not None:
        eng.decide(subject=SUBJECT, type="BLOCK", decision=f"BLOCK: {KEY} ({words})", reason=why, confidence=1.0, evidence=ev, risk=RISK, approval="BLOCK",
                   provenance="USER", params={"asset_id": target.id if target is not None else None, **p}, requirements=req)
        return
    eng.decide(subject=SUBJECT, type="TRANSFORM",
               decision=f"remove {s.get('cuts')} range(s) of speech the user selected from {target.id}: {float(s.get('removed_seconds') or 0):.3f}s "
                        f"({s.get('sentences_whole', 0)} whole and {s.get('sentences_part', 0)} partial sentence(s))",
               reason=f"the user selected these words / sentences ({p['selection'].get('mode')}); transcription-skill placed them ({SELECT_RULE}) and "
                      f"video-editing-skill put each cut in its gap, on silence and on a frame ({RIPPLE_RULE}): {words}. Cut by video-editing-skill CONDENSE "
                      "before every other edit; removing speech always needs the user's yes",
               confidence=1.0, evidence=ev, risk=RISK, approval=approval_for(SUBJECT, floor="CONFIRM", keys={SUBJECT: APPROVAL_KEY}), provenance="USER",
               params={"asset_id": target.id, **p}, requirements=req, serves_intent=None)
    cap_block(condense_skill, f"capability.{condense_skill}")


# ---- after the render ----------------------------------------------------------------------------------------------------
def removed_ranges(keep: Sequence[Sequence[float]], duration: float, rate: Any = None) -> List[Dict[str, float]]:
    """The complement of a trim's kept ranges in [0, duration]: the plain spans `ripple` retimes through (each edge its nearest frame). With the
    source's rate ("30000/1001"), each edge is first moved to the start of the first frame at or after it: the frames the engine's cut keeps are
    those that start in [start, end), so a range written to the millisecond (a silence trim) is retimed as it was cut, and a frame-exact one is
    unchanged."""
    from fractions import Fraction
    import math
    r: Optional[Fraction] = None
    if isinstance(rate, str) and re.match(r"^\d+(/\d+)?$", rate):
        r = Fraction(rate)
    def snap(t: float) -> float:
        if r is None or r <= 0:
            return round(t, 6)
        f = Fraction(repr(t)) * r
        n = math.ceil(f - Fraction(1, 1000))   # the 6-decimal text of a frame start sits within a thousandth of a frame of it
        return round(float(Fraction(n) / r), 6)
    out: List[Dict[str, float]] = []
    t = 0.0
    for a, b in sorted([float(x), float(y)] for x, y in keep):
        if a > t + 1e-9:
            out.append({"start": snap(t), "end": snap(a)})
        t = max(t, b)
    if duration > t + 1e-9:
        out.append({"start": snap(t), "end": round(duration, 6)})
    return [sp for sp in out if sp["end"] > sp["start"]]


def retime_target(doc: Dict[str, Any], decision: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The IR trim of the decision's asset when the delivered programme is that asset's timeline after the trim alone; else the reason the
    retimed files would not match the delivery (another source joined, a speed change, a camera switch ...)."""
    asset = (decision.get("params") or {}).get("asset_id")
    ops = (doc.get("video") or {}).get("operations") or []
    trim = next((o for o in ops if o.get("type") == "video.trim" and o.get("asset") == asset), None)
    if trim is None:
        return None, f"no trim of {asset} in the plan"
    others = sorted({o.get("type") for o in ops if o.get("type") in OTHER_RETIMING or (o.get("type") == "video.trim" and o.get("asset") != asset)})
    if others:
        return None, f"the delivered programme is not {asset}'s timeline after the cut alone ({', '.join(others)}): the retimed transcript and subtitles would not match it"
    return trim, None
