"""Subtitle vocabulary (ADR-031): what the agent plans through subtitle-skill, and how a recognised Transcript becomes cues on
the *delivered* timeline.

    transcript Observation (transcription-skill: segments with start / end / text; speaker_id always null) → SpeechEvents
    → Decision `subtitle.generate` (explicit `subtitle` requirement) → ProductionPlan step → IR `captions.generate`
    (the cues, mapped onto the output timeline) → compiler → subtitle-skill `generate` (SRT / WebVTT sidecar)
    → optional Decision `subtitle.burn_in` → IR `captions.burn` → subtitle-skill `render` (the engine's caption tool)
    → or Decision `subtitle.mux` (ADR-063) → IR `captions.mux` → subtitle-skill `render` with `mode: "mux"` (a soft, player-toggleable
      subtitle track; the picture and the audio are stream-copied)

Vocabulary and arithmetic only, no execution and no recognition:

- **Requirements** (`subtitle.*` keys, explicit USER / PROFILE only; DEFAULT never switches subtitles on): `subtitle` (true, or the
  format name), `subtitle.format` (srt | vtt), `subtitle.burn_in` (bool; the burnt-in picture is the deliverable, the sidecar is
  still produced), `subtitle.mode` (ADR-063, subtitle-skill render's own `mode` parameter: `mux` adds the cues as a soft subtitle
  track of the deliverable, `burn` is the same as `subtitle.burn_in=true`), `subtitle.max_chars_per_line` / `subtitle.max_lines`
  (constraints the Skill checks; it never rewrites cues).
- **Timeline mapping** (deterministic, from IR content only): a transcript is measured on the *source*; the deliverable is the
  source after the trim (kept ranges), the concat (timeline offsets) and the speed factor. Every cue is mapped through exactly
  those operations — a cue inside a removed range is dropped, one overlapping a cut edge is clipped — so the sidecar is in sync
  with what is delivered. Nothing is re-timed by guesswork.
- **What is never done here**: speaker names (a cue carries an anonymous SPEAKER_nn only when the transcript records the speaker merge
  `speakers.assign=true` asked for, ADR-096; subtitle-skill writes it as "SPEAKER_00: ..."), translation, rewriting, line breaking (the
  Skill's constraints report warnings; the cue text is the recognised text), burning into a source file (the burn-in input is
  always an intermediate inside the agent workspace — the Skill accepts workspace-relative inputs only).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models import Requirement
from .editing import EditRequirementError, _bool, _number

# ADR-101: the pinned subtitle-skill contract, read as data (the agent layer imports no tool package): generate's formats, the regional and
# caption profiles and the placement fields it declares. TTML (IMSC 1.1 Text) and EBU-TT-D are sidecar formats only: render (burn / mux)
# stays SRT through the engine's caption tool, from the same cues.
SUBTITLE_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "tools" / "subtitle" / "contract_0.8.0.json"
_SUB_CONTRACT: Dict[str, Any] = json.loads(SUBTITLE_CONTRACT_PATH.read_text(encoding="utf-8"))
FORMATS = tuple(str(f) for f in _SUB_CONTRACT["operations"]["generate"]["formats"])   # srt, vtt, ttml, ebu-tt-d
SIDECAR_EXTENSIONS = {"srt": "srt", "vtt": "vtt", "ttml": "ttml", "ebu-tt-d": "ttml"}   # the sidecar file's extension per format
TTML_FORMATS = ("ttml", "ebu-tt-d")
QC_FORMATS = ("srt", "vtt")   # what the QC gate's subtitle check reads (qc-skill formats.subtitle_formats); a TTML sidecar carries subtitle-skill's conformance instead
REGIONAL_PROFILES: Dict[str, Dict[str, Any]] = {str(k): dict(v) for k, v in (_SUB_CONTRACT["parameters"].get("regional_profiles") or {}).items()}
CAPTION_PROFILES: Dict[str, Dict[str, Any]] = {str(k): dict(v) for k, v in (_SUB_CONTRACT["parameters"].get("caption_profiles") or {}).items()}
PLACEMENT_SETTINGS = ("min_hold", "max_lines", "min_score")   # placement fields besides the regions (`avoid`); faces / reframe are not wired (ADR-101)
SPEAKER_RE = re.compile(r"^SPEAKER_\d{2,}$")   # ADR-096: diarization-skill's anonymous ids, the only speaker a cue may carry
# ADR-101: the opt-in render / cue inputs subtitle-skill 0.5.0 .. 0.8.0 added; regional conventions (ja-broadcast), ruby and vertical writing are
# never applied unless asked for -- the default stays the international one
NEW_KEYS = ("subtitle.regional_profile", "subtitle.caption_profile", "subtitle.placement.avoid") + tuple(f"subtitle.placement.{k}" for k in PLACEMENT_SETTINGS) \
    + ("subtitle.ruby", "subtitle.vertical")
REQUIREMENT_KEYS = ("subtitle", "subtitle.format", "subtitle.burn_in", "subtitle.mode", "subtitle.max_chars_per_line", "subtitle.max_lines") + NEW_KEYS
# subtitle-skill render's `mode` (contract capabilities.render_modes, pinned 0.2.0): burn into the picture, or mux a soft track. Muxing is
# subtitle-skill's job (ecosystem.json: 字幕生成・検証 / 字幕焼き込み; the contract's render delegates to ffmpeg-skill/caption --mode mux),
# so the agent asks subtitle-skill for it and never calls the engine's caption tool itself (ADR-063)
RENDER_MODES = ("burn", "mux")
MUX_SKILL = "subtitle_mux"
CONSTRAINT_KEYS = ("max_chars_per_line", "max_lines")
GENERATE_TOOL = "subtitle/generate"
RENDER_TOOL = "subtitle/render"
GENERATE_SKILL = "subtitle_generation"
BURN_SKILL = "subtitle_burn_in"
MIN_CUE = 0.05          # seconds: a mapped cue shorter than this after clipping is dropped (it would not be readable)
_LANG_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def parse_subtitle_requirements(m: Dict[str, Requirement]) -> Dict[str, Any]:
    """Requirement map → {"enabled", "format" (None = policy decides), "burn_in", "constraints", "requirements"}. A refinement
    without the switch is refused (ambiguous request); values are range-checked; nothing is corrected."""
    main = m.get("subtitle")
    refinements = [m[k] for k in REQUIREMENT_KEYS[1:] if k in m and m[k].provenance != "DEFAULT"]
    out: Dict[str, Any] = {"enabled": False, "format": None, "burn_in": False, "mux": False, "constraints": {}, "requirements": [], "render": {}, "cues": {}}
    if main is None or main.provenance == "DEFAULT":
        if refinements:
            raise EditRequirementError(f"{refinements[0].key} is set but subtitle is not: refusing an ambiguous request")
        return out
    v = main.value
    fmt: Optional[str] = None
    if isinstance(v, str) and v.strip().lower() in FORMATS:
        fmt = v.strip().lower()
        enabled = True
    else:
        enabled = _bool(v, "subtitle")
    if not enabled:
        if refinements:
            raise EditRequirementError("subtitle=false with subtitle.* refinements: refusing an ambiguous request")
        return out
    reqs = [main] + refinements
    vals = {r.key: r.value for r in reqs}
    if "subtitle.format" in vals:
        f = vals["subtitle.format"]
        if not isinstance(f, str) or f.strip().lower() not in FORMATS:
            raise EditRequirementError(f"subtitle.format must be one of {FORMATS}, got {f!r}")
        if fmt and fmt != f.strip().lower():
            raise EditRequirementError("subtitle names one format and subtitle.format another: refusing an ambiguous request")
        fmt = f.strip().lower()
    burn = _bool(vals["subtitle.burn_in"], "subtitle.burn_in") if "subtitle.burn_in" in vals else False
    mode: Optional[str] = None
    if "subtitle.mode" in vals:
        mv = vals["subtitle.mode"]
        if not isinstance(mv, str) or mv.strip().lower() not in RENDER_MODES:
            raise EditRequirementError(f"subtitle.mode must be one of {RENDER_MODES} (subtitle-skill render's mode), got {mv!r}")
        mode = mv.strip().lower()
        if "subtitle.burn_in" in vals and burn != (mode == "burn"):
            raise EditRequirementError(f"subtitle.burn_in={str(burn).lower()} and subtitle.mode={mode} contradict each other: refusing an ambiguous request")
        burn = burn or mode == "burn"
    cons: Dict[str, int] = {}
    if "subtitle.max_chars_per_line" in vals:
        cons["max_chars_per_line"] = int(_number(vals["subtitle.max_chars_per_line"], "subtitle.max_chars_per_line", 8, 200))
    if "subtitle.max_lines" in vals:
        cons["max_lines"] = int(_number(vals["subtitle.max_lines"], "subtitle.max_lines", 1, 4))
    render, cue_opts = _parse_new_options(vals, fmt, burn, mode == "mux")
    return {"enabled": True, "format": fmt, "burn_in": burn, "mux": mode == "mux", "constraints": cons, "requirements": reqs, "render": render, "cues": cue_opts}


def _parse_new_options(vals: Dict[str, Any], fmt: Optional[str], burn: bool, mux: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ADR-101: → (render options {regional_profile?, caption_profile?, placement?}, cue options {regional_profile?, ruby?, vertical?}). Every one is
    opt-in; nothing here is assumed from the language or the platform."""
    render: Dict[str, Any] = {}
    cue: Dict[str, Any] = {}
    if "subtitle.regional_profile" in vals:
        rp = vals["subtitle.regional_profile"]
        if not isinstance(rp, str) or rp not in REGIONAL_PROFILES:
            raise EditRequirementError(f"subtitle.regional_profile must be one of {sorted(REGIONAL_PROFILES)} (opt-in regional conventions), got {rp!r}")
        cue["regional_profile"] = rp
        if burn or mux:
            render["regional_profile"] = rp
    if "subtitle.caption_profile" in vals:
        cp = vals["subtitle.caption_profile"]
        if not isinstance(cp, str) or cp not in CAPTION_PROFILES:
            raise EditRequirementError(f"subtitle.caption_profile must be one of {sorted(CAPTION_PROFILES)}, got {cp!r}")
        if not burn:
            raise EditRequirementError(f"subtitle.caption_profile={cp} is a burn-in style ({CAPTION_PROFILES[cp].get('for')}): it needs subtitle.burn_in=true")
        render["caption_profile"] = cp
        cue["words"] = True
    placement: Dict[str, Any] = {}
    if "subtitle.placement.avoid" in vals:
        regions = vals["subtitle.placement.avoid"]
        regions = [regions] if isinstance(regions, dict) else regions
        if not isinstance(regions, list) or not regions or len(regions) > 32:
            raise EditRequirementError("subtitle.placement.avoid must be a list (at most 32) of regions {name, box: [x, y, w, h] fractions of the frame, start?, end?}")
        out_r = []
        for i, r in enumerate(regions):
            key = f"subtitle.placement.avoid[{i}]"
            if not isinstance(r, dict) or not {"name", "box"} <= set(r) or set(r) - {"name", "box", "start", "end"}:
                raise EditRequirementError(f"{key} must be {{name, box: [x, y, w, h], start?, end?}}")
            if not isinstance(r["name"], str) or not r["name"].strip() or len(r["name"]) > 64 or "\n" in r["name"]:
                raise EditRequirementError(f"{key}.name must be a short non-empty name")
            box = r["box"]
            if not isinstance(box, list) or len(box) != 4:
                raise EditRequirementError(f"{key}.box must be [x, y, w, h] as fractions of the frame")
            x, y, w, h = (float(_number(v, f"{key}.box", 0.0, 1.0)) for v in box)
            if w <= 0 or h <= 0 or x + w > 1.0 + 1e-9 or y + h > 1.0 + 1e-9:
                raise EditRequirementError(f"{key}.box must lie inside the frame with a positive size")
            rec: Dict[str, Any] = {"name": r["name"], "box": [x, y, w, h]}
            for k in ("start", "end"):
                if r.get(k) is not None:
                    rec[k] = float(_number(r[k], f"{key}.{k}", 0.0, 86400.0))
            if "start" in rec and "end" in rec and not rec["end"] > rec["start"]:
                raise EditRequirementError(f"{key}: end must be after start")
            out_r.append(rec)
        placement["avoid"] = out_r
    for k, lo, hi, integer in (("min_hold", 0.0, 60.0, False), ("max_lines", 1, 4, True), ("min_score", 0.0, 1.0, False)):
        if f"subtitle.placement.{k}" in vals:
            if "avoid" not in placement:
                raise EditRequirementError(f"subtitle.placement.{k} refines subtitle.placement.avoid, which is not set: refusing an ambiguous request")
            v = _number(vals[f"subtitle.placement.{k}"], f"subtitle.placement.{k}", lo, hi)
            if integer and float(v) != int(v):
                raise EditRequirementError(f"subtitle.placement.{k} must be an integer")
            placement[k] = int(v) if integer else float(v)
    if placement:
        if not (burn or mux):
            raise EditRequirementError("subtitle.placement.* places the cues of a burn-in or a soft track: it needs subtitle.burn_in=true or subtitle.mode=mux")
        render["placement"] = placement
    if "subtitle.ruby" in vals:
        rb = vals["subtitle.ruby"]
        if not isinstance(rb, dict) or not rb or len(rb) > 500:
            raise EditRequirementError("subtitle.ruby must be a reading per base text {base: reading} (at most 500)")
        for b, r in rb.items():
            if not isinstance(b, str) or not b.strip() or not isinstance(r, str) or not r.strip() or "\n" in b or "\n" in r or len(b) > 64 or len(r) > 64:
                raise EditRequirementError(f"subtitle.ruby {b!r}: base and reading must be short single-line text")
        cue["ruby"] = {str(b): str(r) for b, r in rb.items()}
    if "subtitle.vertical" in vals and _bool(vals["subtitle.vertical"], "subtitle.vertical"):
        if burn or mux:
            raise EditRequirementError("subtitle.vertical (top-to-bottom writing) is a sidecar style: subtitle-skill does not burn it and a soft SRT track cannot carry it")
        if (fmt or "srt") not in ("ttml", "vtt"):
            raise EditRequirementError("subtitle.vertical needs subtitle.format=ttml or vtt (the formats that carry vertical writing)")
        cue["vertical"] = True
    return render, cue


def annotate_cues(cues: List[Dict[str, Any]], ruby: Optional[Dict[str, str]] = None, vertical: bool = False,
                  words: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """ADR-101: the user's readings as cue `ruby` (each base, longest first, at its occurrences in text order, never overlapping), `style.vertical`,
    and the word timings that fall inside each cue (already on the delivered timeline) as cue `words`. Nothing is guessed: a base not in the
    text is not annotated, a cue without words gets none."""
    out = []
    bases = sorted((ruby or {}), key=lambda b: (-len(b), b))
    for c in cues:
        c = dict(c)
        if bases:
            spans: List[Tuple[int, int, str]] = []
            for b in bases:
                i = c["text"].find(b)
                while i >= 0:
                    if not any(i < e and i + len(b) > s for s, e, _ in spans):
                        spans.append((i, i + len(b), b))
                    i = c["text"].find(b, i + len(b))
            if spans:
                c["ruby"] = [{"base": b, "text": ruby[b]} for _s, _e, b in sorted(spans)]
        if vertical:
            c["style"] = {"vertical": True}
        if words is not None:
            ws, prev = [], c["start"]
            for w in words:
                s_, e_ = max(float(w["start"]), c["start"]), min(float(w["end"]), c["end"])
                if e_ - s_ >= 0.001 and s_ >= prev - 1e-6:
                    ws.append({"text": w["text"].strip(), "start": round(s_, 3), "end": round(e_, 3)})
                    prev = round(e_, 3)
            if ws:
                c["words"] = ws
        out.append(c)
    return out


def words_on_timeline(words: List[Dict[str, Any]], keep: List[List[float]], offset: float = 0.0, speed: float = 1.0) -> List[Dict[str, Any]]:
    """ADR-101: transcript word timings (source time) mapped through the same trim / concat / speed as the cues; a word in removed material is dropped."""
    out = []
    for w in words:
        m = map_cue(float(w["start"]), float(w["end"]), keep, offset, speed) if float(w["end"]) - float(w["start"]) >= MIN_CUE else None
        if m is None:
            s_, e_ = map_point(float(w["start"]), keep, False), map_point(float(w["end"]), keep, True)
            if s_ is None or e_ is None or not e_ > s_:
                continue
            m = (round((s_ + offset) / speed, 3), round((e_ + offset) / speed, 3))
        if str(w.get("text") or "").strip():
            out.append({"text": str(w["text"]), "start": m[0], "end": m[1]})
    return sorted(out, key=lambda w: (w["start"], w["end"]))


# ---- timeline mapping (source time → delivered time)
def kept_ranges_of(doc_video_ops: List[Dict[str, Any]], asset_id: str, duration: float) -> List[List[float]]:
    op = next((o for o in doc_video_ops if o.get("type") == "video.trim" and o.get("asset") == asset_id), None)
    if op is None:
        return [[0.0, round(float(duration), 3)]]
    return [[round(float(s), 3), round(float(e), 3)] for s, e in op["keep"]]


def map_point(t: float, keep: List[List[float]], is_end: bool) -> Optional[float]:
    """A source time onto the trimmed timeline. Inside a kept range: shifted by the removed material before it. Inside a removed
    range: a cue start snaps forward to the next kept range, a cue end snaps back to the previous one. None when nothing kept
    lies on that side (the cue lies entirely in removed material on that side)."""
    acc = 0.0
    for s, e in keep:
        if s <= t <= e:
            return round(acc + (t - s), 3)
        if t < s:
            return round(acc, 3) if not is_end else (round(acc, 3) if acc > 0 else None)
        acc += e - s
    return round(acc, 3) if is_end else None


def map_cue(start: float, end: float, keep: List[List[float]], offset: float = 0.0, speed: float = 1.0) -> Optional[Tuple[float, float]]:
    """(start, end) on the delivered timeline: trim → concat offset → speed factor; None when the cue is dropped."""
    ms, me = map_point(float(start), keep, False), map_point(float(end), keep, True)
    if ms is None or me is None:
        return None
    ms, me = (ms + offset) / speed, (me + offset) / speed
    if me - ms < MIN_CUE:
        return None
    return round(ms, 3), round(me, 3)


def cues_from_segments(segments: List[Dict[str, Any]], keep: List[List[float]], offset: float = 0.0, speed: float = 1.0, id_prefix: str = "c",
                       speakers: bool = False) -> List[Dict[str, Any]]:
    """Transcript segments → subtitle cues on the delivered timeline (ids `c0001`…, deterministic order, no overlap: a cue
    starting before the previous one ended is clipped to start there). Text is the recognised text, trimmed; empty cues and
    cues that fall into removed material are dropped. `speaker` is set only when `speakers` (ADR-096: the transcript records the
    speaker merge an explicit requirement asked for) and the segment carries an anonymous SPEAKER_nn id; never a name."""
    out: List[Dict[str, Any]] = []
    prev_end = -1.0
    for seg in sorted(segments, key=lambda s: (float(s.get("start", 0.0)), str(s.get("id")))):
        text = _CTRL_RE.sub("", str(seg.get("text") or "")).strip()
        if not text:
            continue
        mapped = map_cue(float(seg["start"]), float(seg["end"]), keep, offset, speed)
        if mapped is None:
            continue
        s, e = mapped
        if s < prev_end:
            s = prev_end
        if e - s < MIN_CUE:
            continue
        cue = {"id": f"{id_prefix}{len(out) + 1:04d}", "start": round(s, 3), "end": round(e, 3), "text": text}
        if speakers and isinstance(seg.get("speaker_id"), str) and SPEAKER_RE.match(seg["speaker_id"]):
            cue["speaker"] = seg["speaker_id"]
        out.append(cue)
        prev_end = round(e, 3)
    return out


def cues_from_transcripts(transcripts: List[Any], keep: List[List[float]], offset: float = 0.0, speed: float = 1.0, id_prefix: str = "c") -> List[Dict[str, Any]]:
    """The cues of one input's transcript observation(s). ADR-096: they carry the segments' anonymous speaker ids only when every one of
    them records transcription-skill's speaker merge (`speaker_assignment`, asked for by speakers.assign); otherwise no speaker at all."""
    segs = [s for o in transcripts for s in ((o.data or {}).get("segments") or [])]
    merged = bool(transcripts) and all(isinstance((o.data or {}).get("speaker_assignment"), dict) for o in transcripts)
    return cues_from_segments(segs, keep, offset=offset, speed=speed, id_prefix=id_prefix, speakers=merged)


def valid_language(lang: Any) -> str:
    s = str(lang or "").strip()
    return s if _LANG_RE.match(s) else "und"


# ---- IR records
def ir_captions_generate(subject: str, output: str, fmt: str, language: str, cues: List[Dict[str, Any]], decision_ids: List[str], sources: List[str],
                         timeline_map: Dict[str, Any], constraints: Optional[Dict[str, Any]] = None, scope: Optional[Dict[str, float]] = None,
                         regional_profile: Optional[str] = None) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"type": "captions.generate", "asset": subject, "output": output, "format": fmt, "language": language, "cues": list(cues),
                           "sources": list(sources), "timeline_map": dict(timeline_map)}
    if regional_profile:   # ADR-101: opt-in regional conventions (speaker colours, line limits) for the sidecar
        rec["regional_profile"] = regional_profile
    if constraints:
        rec["constraints"] = {k: constraints[k] for k in CONSTRAINT_KEYS if k in constraints}
    if scope is not None:
        rec["temporal_scope"] = {"start": round(float(scope["start"]), 3), "end": round(float(scope["end"]), 3)}
    rec["decision_ids"] = list(decision_ids)
    return rec


def ir_captions_burn(subject: str, input_id: str, sidecar: str, output: str, decision_ids: List[str], scope: Optional[Dict[str, float]] = None,
                     op_type: str = "captions.burn", render: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """IR captions.burn, or (ADR-063) captions.mux: the same references; the compiler asks subtitle-skill render for the mode. ADR-101: the
    render options the user asked for (placement, regional_profile, caption_profile) ride on the record."""
    rec: Dict[str, Any] = {"type": op_type, "asset": subject, "input": input_id, "sidecar": sidecar, "output": output}
    for k in ("placement", "regional_profile", "caption_profile"):
        if (render or {}).get(k):
            rec[k] = render[k]
    if scope is not None:
        rec["temporal_scope"] = {"start": round(float(scope["start"]), 3), "end": round(float(scope["end"]), 3)}
    rec["decision_ids"] = list(decision_ids)
    return rec
