"""IR validation: JSON Schema + semantic rules + capability availability."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from ..agent.decision_engine import check_decisions
from ..agent.audio import (AUDIO_ORDER, DUCK_FIELDS, DUCK_REQUIRED, DYNAMICS_STAGES, FILLER_DECLICK_RANGE, FILLER_LEXICON_MAX_ITEMS, FILLER_LEXICON_MAX_LENGTH,
                           FILLER_MAX_WORDS, FILLER_MERGE_GAP_RANGE, FILLER_PAD_RANGE, FILLER_PRESETS, FILLER_WORD_MAX_LENGTH, MAX_MIX_INPUTS, MIX_GAIN_RANGE, NOISE_REDUCTION_MODES, NOISE_REDUCTION_STRENGTH, OPERATIONS as AUDIO_OPERATIONS, VOICE_CLEANUP_LEVELS,
                           PROGRAMME_AUDIO, PROGRAMME_OPS as AUDIO_PROGRAMME_OPS, SKILL_OF as AUDIO_SKILL_OF)
from ..agent.editing import EDIT_ORDER, IMAGE_INSERT_DURATION, IMAGE_INSERT_OP, IMAGE_INSERT_POSITIONS, OPERATIONS, SKILL_OF, switch_cams
from ..agent.finishing import (COLOR_OPERATIONS, COLOR_ORDER, COLOR_SKILL_OF, ELEMENT_TYPES, FADE_TYPES, GRAPHICS_SKILL, THUMBNAIL_FRAME_SKILL, THUMBNAIL_RENDER_SKILL,
                               VIDEO_EXTENSIONS, picture_size, placement_problems)
from ..agent.production_plan import validate_plan
from ..agent.qc import QC_SKILL
from ..agent.subtitles import BURN_SKILL, GENERATE_SKILL, MUX_SKILL
from ..agent.takes import BEAT_KIND, CROSSFADE_SUBJECT, PICK_SUBJECT, SNAP_SUBJECT, SYNC_TOLERANCE
from ..agent.timeline import TIMELINE_FORMATS, TIMELINE_SKILL, TIMELINE_SUBJECT
from ..agent.image_prepare import prepare_chain, prepare_problems
from ..agent.voice import OPERATIONS as VOICE_OPERATIONS, VOICE_ORDER
from ..tools.voice_enhance.adapter import MAX_TRACKS as VOICE_MAX_TRACKS, PARAMETERS as VOICE_PARAMETERS
from ..models import Event
from ..temporal import Session, classify, validate_event, validate_session
from typing import Any, Dict, List, Optional, Tuple

from .ir import ProjectIR

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "project.schema.json"


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def produced_subjects(d: Dict[str, Any]) -> set:
    """Logical subjects an operation produced (the video / audio concat programmes) that later operations may reference."""
    return {op.get("output") for op in d["video"]["operations"] + d["audio"]["operations"] if op.get("type") in ("video.concat", "video.switch", "video.grid", "audio.concat", "audio.mix") and op.get("output")}


def check_audio_operations(d: Dict[str, Any]) -> List[str]:
    """Per-type rules of the audio operations on the audio production path (ADR-030): explicit type, allowlisted parameters, an
    audio stream on the subject, one cut per subject before anything else, one concat at most, the fixed order, references that
    exist, ranges inside the source, the loudness operation carrying its references only on the audio path. Never a guess."""
    errs: List[str] = []
    assets = d["assets"]
    aops = d["audio"]["operations"]
    known: set = set(assets)
    seen_rank: Dict[str, float] = {}
    concat_seen = False
    programme = next((op for op in aops if op.get("type") in AUDIO_PROGRAMME_OPS), None)
    on_audio_path: set = set()
    for op in aops:
        t = op.get("type")
        subj = op.get("asset")
        if t == "audio.loudness":
            if ("input" in op) != ("output" in op):
                errs.append("audio.loudness: input and output references go together (audio path) or are both absent (reference engine path)")
            if "input" not in op and any(k in op for k in ("tolerance_lu", "sample_rate")):
                errs.append("audio.loudness: tolerance_lu / sample_rate exist only on the audio production path")
            if "input" in op:
                on_audio_path.add(subj)
                if subj not in known:
                    errs.append(f"audio.loudness references {subj!r} before it exists")
                rank = AUDIO_ORDER.index("audio.loudness")
                if seen_rank.get(subj, -1) >= rank:
                    errs.append(f"audio.loudness on {subj} is out of the fixed audio order {AUDIO_ORDER}")
                seen_rank[subj] = rank
            continue
        if t in VOICE_OPERATIONS:   # ADR-073: voice-enhance-skill, per input after its cut / filler removal (bleed reduction: the mix's inputs as recorded)
            errs.extend(_voice_problems(op, assets, programme, seen_rank, on_audio_path))
            continue
        if t not in AUDIO_OPERATIONS:
            errs.append(f"unknown audio operation type {t!r}")
            continue
        spec = AUDIO_OPERATIONS[t]
        extra = sorted(k for k in op if k not in ("type", "asset", "input", "inputs", "output", "segments", "timeline_duration", "temporal_scope", "decision_ids") and k not in spec["params"])
        if extra:
            errs.append(f"{t}: parameters {extra} are not in the operation's vocabulary")
        on_audio_path.add(subj)
        if t == "audio.cut":
            if subj not in assets:
                errs.append(f"audio.cut on {subj!r}: only a source asset is cut")
            elif subj in seen_rank:
                errs.append(f"audio.cut on {subj} must come before every other audio operation")
            seen_rank[subj] = -1
            dur = (assets.get(subj) or {}).get("technical", {}).get("duration") or 0.0
            rs = op.get("remove") or []
            last = -1.0
            for r in rs:
                if r[1] <= r[0] or r[0] < last:
                    errs.append(f"audio.cut remove range {r} is empty, unsorted or overlapping")
                if dur and r[1] > float(dur) + 0.01:
                    errs.append(f"audio.cut remove range {r} exceeds asset duration {float(dur):.3f}")
                last = r[1]
            if dur and sum(e - s for s, e in rs) >= float(dur) - 0.01:
                errs.append(f"audio.cut on {subj} would remove everything")
        elif t == "audio.filler_remove":   # ADR-072: a source asset, after its cut and before everything else; never a mixed input
            if subj not in assets:
                errs.append(f"audio.filler_remove on {subj!r}: only a source asset's words are removed")
            elif seen_rank.get(subj, -1) != -1:   # nothing but its cut before it (a second filler removal sees -0.5)
                errs.append(f"audio.filler_remove on {subj} must come right after its audio.cut, before every other audio operation")
            seen_rank[subj] = -0.5
            if op.get("input") is None or op.get("output") is None:
                errs.append(f"audio.filler_remove on {subj}: input / output reference missing")
            if programme is not None and programme.get("type") == "audio.mix" and subj in (programme.get("inputs") or []):
                errs.append(f"audio.filler_remove on {subj}: a mixed input keeps its own timeline (removing words would move it against the other inputs)")
            errs.extend(_filler_problems(op, subj))
        elif t in AUDIO_PROGRAMME_OPS:   # audio.concat (ADR-030) or audio.mix (ADR-063): one audio programme per plan
            if concat_seen:
                errs.append(f"more than one audio programme ({t}; one audio.concat or audio.mix per plan)")
            concat_seen = True
            ins = op.get("inputs") or []
            if len(ins) < 2 or len(set(ins)) != len(ins):
                errs.append(f"{t} needs two or more distinct inputs")
            for i in ins:
                if i not in assets:
                    errs.append(f"{t} input {i!r} is not an asset")
            if op.get("output") != subj or subj in assets:
                errs.append(f"{t}: output must equal the operation's subject and not collide with an asset id")
            segs = op.get("segments") or []
            if segs and {s_.get("input") for s_ in segs} != set(ins):
                errs.append(f"{t} segments do not cover exactly the inputs")
            if t == "audio.mix":
                if len(ins) > MAX_MIX_INPUTS:
                    errs.append(f"audio.mix takes at most {MAX_MIX_INPUTS} inputs, got {len(ins)}")
                lv = op.get("levels")
                if lv is not None:
                    if not isinstance(lv, list) or len(lv) != len(ins):
                        errs.append("audio.mix levels must list one {gain_db, mute} per input, in input order")
                    else:
                        for n, x in enumerate(lv):
                            if not (isinstance(x, dict) and set(x) <= {"gain_db", "mute"} and isinstance(x.get("gain_db", 0.0), (int, float)) and not isinstance(x.get("gain_db", 0.0), bool)
                                    and MIX_GAIN_RANGE[0] <= float(x.get("gain_db", 0.0)) <= MIX_GAIN_RANGE[1] and isinstance(x.get("mute", False), bool)):
                                errs.append(f"audio.mix levels[{n}] {x!r} must be {{gain_db -60..60, mute}}")
                        if lv and isinstance(lv[0], dict) and lv[0].get("mute"):
                            errs.append("audio.mix: the first input sets the mix's length and cannot be muted")
                dk = op.get("duck")
                if dk is not None:   # ADR-072: MIX.duck, the bed by index (1..inputs-1) and the contract's ranges
                    if not isinstance(dk, dict) or set(dk) - set(DUCK_FIELDS) or any(f not in dk for f in DUCK_REQUIRED):
                        errs.append(f"audio.mix duck {dk!r} must be {{input, amount_db, threshold_db?, attack_ms?, release_ms?}}")
                    else:
                        for f, v in dk.items():
                            lo, hi = DUCK_FIELDS[f]
                            if not (isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= float(v) <= hi):
                                errs.append(f"audio.mix duck.{f} {v!r} must be within {lo:g}..{hi:g}")
                        di = dk.get("input")
                        if isinstance(di, int) and not isinstance(di, bool) and not 1 <= di < len(ins):
                            errs.append(f"audio.mix duck.input {di} names no bed among inputs 1..{len(ins) - 1} (input 0 is the key)")
                        elif isinstance(lv, list) and isinstance(di, int) and di < len(lv) and isinstance(lv[di], dict) and lv[di].get("mute"):
                            errs.append(f"audio.mix duck.input {di} is muted: the ducked bed must be unmuted")
            known.add(subj)
            seen_rank[subj] = -1
            continue
        else:
            if subj not in known:
                errs.append(f"{t} references {subj!r} before it exists (the audio programme exists only after audio.concat / audio.mix)")
            rank = AUDIO_ORDER.index(t)
            if seen_rank.get(subj, -1) >= rank:
                errs.append(f"{t} on {subj} is out of the fixed audio order {AUDIO_ORDER}")
            seen_rank[subj] = rank
            if op.get("input") is None:
                errs.append(f"{t} on {subj}: input reference missing")
            if t in ("audio.fade_in", "audio.fade_out") and not (isinstance(op.get("duration"), (int, float)) and 0 < float(op["duration"]) <= 3600):
                errs.append(f"{t}: duration {op.get('duration')!r} must be within 0..3600 s")
            if t == "audio.gain" and not (isinstance(op.get("gain_db"), (int, float)) and -60 <= float(op["gain_db"]) <= 60 and float(op["gain_db"]) != 0):
                errs.append(f"audio.gain: gain_db {op.get('gain_db')!r} must be within -60..60 and not 0")
            if t == "audio.voice_cleanup" and op.get("level") not in VOICE_CLEANUP_LEVELS:   # ADR-068: the contract's required enum
                errs.append(f"audio.voice_cleanup: level {op.get('level')!r} must be one of {VOICE_CLEANUP_LEVELS}")
            if t == "audio.noise_reduction":   # ADR-063: the contract's required pair
                if op.get("mode") not in NOISE_REDUCTION_MODES:
                    errs.append(f"audio.noise_reduction: mode {op.get('mode')!r} must be one of {NOISE_REDUCTION_MODES}")
                sd = op.get("strength_db")
                if not (isinstance(sd, (int, float)) and not isinstance(sd, bool) and NOISE_REDUCTION_STRENGTH[0] <= float(sd) <= NOISE_REDUCTION_STRENGTH[1]):
                    errs.append(f"audio.noise_reduction: strength_db {sd!r} must be within {NOISE_REDUCTION_STRENGTH[0]:g}..{NOISE_REDUCTION_STRENGTH[1]:g}")
            if t == "audio.dynamics":   # ADR-063: at least one stage, each field inside the contract's range
                stages = [st for st in DYNAMICS_STAGES if op.get(st) is not None]
                if not stages:
                    errs.append("audio.dynamics needs at least one of gate, compressor, limiter")
                for st in stages:
                    if not isinstance(op[st], dict):
                        errs.append(f"audio.dynamics {st} must be an object")
                        continue
                    for k, v in op[st].items():
                        rng = DYNAMICS_STAGES[st].get(k)
                        if rng is None:
                            errs.append(f"audio.dynamics {st}.{k} is not a field of the stage")
                        elif not (isinstance(v, (int, float)) and not isinstance(v, bool) and rng[0] <= float(v) <= rng[1]):
                            errs.append(f"audio.dynamics {st}.{k} {v!r} must be within {rng[0]:g}..{rng[1]:g}")
    for subj in on_audio_path:
        if subj in assets and not ((assets[subj].get("technical") or {}).get("audio")):
            errs.append(f"audio operations on {subj}: no audio stream")
    # an asset on the audio path is never also edited as video (its deliverable is audio)
    for op in d["video"]["operations"]:
        if op.get("asset") in on_audio_path:
            errs.append(f"{op.get('type')} on {op.get('asset')}: the asset is on the audio production path (audio deliverable), video operations conflict")
    return errs


def _voice_problems(op: Dict[str, Any], assets: Dict[str, Any], programme: Optional[Dict[str, Any]], seen_rank: Dict[str, float], on_audio_path: set) -> List[str]:
    """ADR-073: a voice.* record: its vocabulary, the parameters inside voice-enhance-skill's ranges, its place in the chain (after the input's
    cut / filler removal, before every audio-production operation: bleed reduction → dereverberation → isolation), and for bleed reduction
    the inputs of the decided mix exactly, uncut, one `<asset>_bleed_reduce` output per input."""
    errs: List[str] = []
    t, subj = op.get("type"), op.get("asset")
    spec = VOICE_OPERATIONS[t]
    refs = ("inputs", "outputs") if t == "voice.bleed_reduce" else ("input", "output", "temporal_scope")
    extra = sorted(k for k in op if k not in ("type", "asset", "decision_ids") + refs and k not in spec["params"])
    if extra:
        errs.append(f"{t}: parameters {extra} are not in the operation's vocabulary")
    for k, (lo, hi, integer, open_lo) in VOICE_PARAMETERS[spec["tool"]].items():
        v = op.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and float(v) != int(v)) or (v <= lo if open_lo else v < lo) or v > hi):
            errs.append(f"{t} on {subj}: {k} {v!r} must be within {'(' if open_lo else '['}{lo:g}, {hi:g}]")
    rank = -0.5 + 0.1 * (VOICE_ORDER.index(t) + 1)   # cut -1 < filler removal -0.5 < bleed -0.4 < dereverb -0.3 < isolate -0.2 < the AUDIO_ORDER chain (0..)
    if t == "voice.bleed_reduce":
        ins, outs = list(op.get("inputs") or []), list(op.get("outputs") or [])
        mix_in = list((programme or {}).get("inputs") or []) if (programme or {}).get("type") == "audio.mix" else None
        if subj != PROGRAMME_AUDIO or mix_in is None or ins != mix_in:
            errs.append(f"voice.bleed_reduce must reduce the cross-talk between exactly the inputs of the audio mix (subject {PROGRAMME_AUDIO})")
        if not 2 <= len(ins) <= VOICE_MAX_TRACKS or len(set(ins)) != len(ins) or any(a not in assets for a in ins):
            errs.append(f"voice.bleed_reduce takes 2..{VOICE_MAX_TRACKS} distinct source assets")
        if outs != [f"{a}_bleed_reduce" for a in ins]:
            errs.append("voice.bleed_reduce must name one <asset>_bleed_reduce output per input, in input order")
        for a in ins:
            if a in seen_rank:
                errs.append(f"voice.bleed_reduce on {a}: a track is cleaned as recorded (time-aligned), before anything else on it")
            seen_rank[a] = rank
            on_audio_path.add(a)
        return errs
    if subj not in assets:
        errs.append(f"{t} on {subj!r}: only a source asset's speech is enhanced (per input, before the programme)")
    elif seen_rank.get(subj, -1) >= rank:
        errs.append(f"{t} on {subj} must come after its cut / filler removal / bleed reduction and before every other audio operation ({' → '.join(VOICE_ORDER)})")
    seen_rank[subj] = rank
    on_audio_path.add(subj)
    if op.get("input") is None or op.get("output") is None:
        errs.append(f"{t} on {subj}: input / output reference missing")
    return errs


def _filler_problems(op: Dict[str, Any], subj: Any) -> List[str]:
    """ADR-072: an audio.filler_remove record against the contract's FILLER_REMOVE: timed words in order inside the operation's scope,
    printable text, a non-empty lexicon of the contract's presets / printable words, pad / merge_gap / declick_ms in range."""
    errs: List[str] = []
    words = op.get("words")
    end = float(((op.get("temporal_scope") or {}).get("end")) or 0.0)
    if not isinstance(words, list) or not words or len(words) > FILLER_MAX_WORDS:
        errs.append(f"audio.filler_remove on {subj}: words must be 1..{FILLER_MAX_WORDS} timed words")
        words = []
    last = -1.0
    for n, w in enumerate(words):
        ok = (isinstance(w, dict) and set(w) <= {"start", "end", "text", "confidence"} and all(isinstance(w.get(k), (int, float)) and not isinstance(w.get(k), bool) for k in ("start", "end"))
              and isinstance(w.get("text"), str) and 0 < len(w["text"]) <= FILLER_WORD_MAX_LENGTH and not any(ord(c) < 32 for c in w["text"]))
        if not ok:
            errs.append(f"audio.filler_remove on {subj}: words[{n}] {str(w)[:80]} must be {{start, end, text, confidence?}} with printable text")
            break
        if not 0 <= float(w["start"]) < float(w["end"]) or float(w["start"]) < last or (end and float(w["end"]) > end + 0.01):
            errs.append(f"audio.filler_remove on {subj}: words[{n}] {w['start']}-{w['end']} is empty, out of order or past the audio it cuts ({end:.3f}s)")
            break
        if "confidence" in w and not (isinstance(w["confidence"], (int, float)) and 0 <= float(w["confidence"]) <= 1):
            errs.append(f"audio.filler_remove on {subj}: words[{n}].confidence must be within 0..1")
            break
        last = float(w["start"])
    lex = op.get("lexicon")
    if not isinstance(lex, dict) or set(lex) - {"presets", "words", "keep"} or not (lex.get("presets") or lex.get("words")):
        errs.append(f"audio.filler_remove on {subj}: lexicon {lex!r} must name presets and / or words (there is no default)")
    else:
        if not set(lex.get("presets") or []) <= set(FILLER_PRESETS):
            errs.append(f"audio.filler_remove on {subj}: lexicon presets {lex.get('presets')!r} must be among {FILLER_PRESETS}")
        for part in ("words", "keep"):
            v = lex.get(part)
            if v is not None and not (isinstance(v, list) and len(v) <= FILLER_LEXICON_MAX_ITEMS
                                      and all(isinstance(x, str) and 0 < len(x) <= FILLER_LEXICON_MAX_LENGTH and not any(ord(c) < 32 for c in x) for x in v)):
                errs.append(f"audio.filler_remove on {subj}: lexicon {part} must be at most {FILLER_LEXICON_MAX_ITEMS} printable words of 1..{FILLER_LEXICON_MAX_LENGTH} characters")
    for k, (lo, hi) in (("pad", FILLER_PAD_RANGE), ("merge_gap", FILLER_MERGE_GAP_RANGE), ("declick_ms", FILLER_DECLICK_RANGE)):
        v = op.get(k)
        if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= float(v) <= hi):
            errs.append(f"audio.filler_remove on {subj}: {k} {v!r} must be within {lo:g}..{hi:g}")
    return errs


def check_video_operations(d: Dict[str, Any]) -> List[str]:
    """Per-type rules of the video operations (ADR-029): explicit type, allowlisted parameters, references that exist at that
    point of the chain, ranges inside the source, one concat at most, the fixed order, no fit + fill, image inside the allowed
    input roots. Anything outside the vocabulary is an error, never a guess."""
    errs: List[str] = []
    assets = d["assets"]
    ops = d["video"]["operations"]
    known: set = set(assets)
    order_seen: Dict[str, int] = {}
    concat_seen = False
    allowed_roots = [str(Path(r)) for r in (d.get("execution") or {}).get("allowed_inputs") or []]
    for op in ops:
        t = op.get("type")
        subj = op.get("asset")
        if "prepare" in op and t != "video.overlay":   # ADR-077: only an overlay's image is prepared
            errs.append(f"{t}: parameters ['prepare'] are not in the operation's vocabulary (only video.overlay prepares its image)")
        if t == "video.trim":
            if subj not in assets:
                continue   # reported by the caller
            dur = (assets[subj].get("technical") or {}).get("duration") or 0.0
            for k in op.get("keep") or []:
                if k[1] <= k[0]:
                    errs.append(f"video.trim keep range {k} is empty")
                if dur and k[1] > dur + 0.01:
                    errs.append(f"video.trim keep range {k} exceeds asset duration {dur:.3f}")
            continue
        if t not in OPERATIONS:
            errs.append(f"unknown video operation type {t!r}")
            continue
        spec = OPERATIONS[t]
        extra = sorted(k for k in op if k not in ("type", "asset", "input", "inputs", "output", "image", "segments", "timeline_duration", "temporal_scope", "decision_ids")
                       and k not in spec["params"] and not (k == "prepare" and t == "video.overlay"))   # ADR-077: checked below
        if extra:
            errs.append(f"{t}: parameters {extra} are not in the operation's vocabulary")
        rank = EDIT_ORDER.index(t)
        if subj in order_seen and order_seen[subj] >= rank:
            errs.append(f"{t} on {subj} is out of the fixed operation order {EDIT_ORDER}")
        order_seen[subj] = rank
        if t in ("video.concat", "video.switch", "video.grid"):
            if concat_seen:
                errs.append("more than one multi-source programme operation (video.concat / video.switch / video.grid, one programme per plan)")
            concat_seen = True
            ins = op.get("inputs") or []
            if len(ins) < 2 or len(set(ins)) != len(ins):
                errs.append(f"{t} needs two or more distinct inputs")
            for i in ins:
                if i not in assets:
                    errs.append(f"{t} input {i!r} is not an asset")
                elif not ((assets[i].get("technical") or {}).get("video")):
                    errs.append(f"{t} input {i!r} has no video stream")
            if op.get("output") != subj:
                errs.append(f"{t}: output must equal the operation's subject")
            if subj in assets:
                errs.append(f"{t} output {subj!r} collides with an asset id")
            if t == "video.concat":
                segs = op.get("segments") or []
                if [s_.get("input") for s_ in segs] and set(s_.get("input") for s_ in segs) != set(ins):
                    errs.append("video.concat segments do not cover exactly the inputs")
                for s_ in segs:
                    sr, tr = s_.get("source_range") or [0, 0], s_.get("timeline_range") or [0, 0]
                    if not (sr[0] < sr[1] and tr[0] < tr[1]):
                        errs.append(f"video.concat segment {s_.get('input')}: empty range")
                    sd = (assets.get(s_.get("input")) or {}).get("technical", {}).get("duration")
                    if sd and sr[1] > float(sd) + 0.01:
                        errs.append(f"video.concat segment {s_.get('input')}: source range {sr} exceeds duration {sd:.3f}")
            elif t == "video.switch":   # ADR-045: the switch/audio indices must name one of the given inputs (0 = reference)
                bad = [c for c in switch_cams(op.get("switch") or "") if c >= len(ins)] + ([op["audio"]] if op.get("audio") is not None and op["audio"] >= len(ins) else [])
                if bad:
                    errs.append(f"video.switch references camera index {max(bad)}, only {len(ins)} input(s) given")
            else:   # video.grid (ADR-046): cols*rows must have room for every input, audio_from must name one of them
                cols, rows = op.get("cols"), op.get("rows")
                if not cols or not rows:
                    errs.append("video.grid needs cols and rows")
                elif int(cols) * int(rows) < len(ins):
                    errs.append(f"video.grid {cols}x{rows} has no room for {len(ins)} input(s)")
                if op.get("audio_from") is not None and op["audio_from"] >= len(ins):
                    errs.append(f"video.grid audio_from references input index {op['audio_from']}, only {len(ins)} input(s) given")
            known.add(subj)
            continue
        if subj not in known:
            errs.append(f"{t} references {subj!r} before it exists (the programme exists only after video.concat / video.switch / video.grid)")
        if subj in assets and not ((assets[subj].get("technical") or {}).get("video")):
            errs.append(f"{t} on {subj}: no video stream")
        if op.get("input") is None:
            errs.append(f"{t} on {subj}: input reference missing")
        if t == "video.speed":
            f = op.get("factor")
            if not isinstance(f, (int, float)) or isinstance(f, bool) or not (0.25 <= float(f) <= 4.0) or float(f) == 1.0:
                errs.append(f"video.speed factor {f!r} must be within 0.25..4 and not 1")
        if t in ("video.resize", "video.fit", "video.fill", "video.concat", "video.crop", "video.redact") and op.get("width") is not None and (int(op["width"]) % 2 or int(op["width"]) < 16):
            errs.append(f"{t}: width {op['width']} must be an even integer ≥ 16")
        if t in ("video.crop", "video.redact") and op.get("height") is not None and (int(op["height"]) % 2 or int(op["height"]) < 16):
            errs.append(f"{t}: height {op['height']} must be an even integer ≥ 16")
        if t in ("video.crop", "video.redact") and (op.get("x") is None or op.get("y") is None or op.get("width") is None or op.get("height") is None):
            errs.append(f"{t} needs x, y, width and height (an explicit rectangle in source pixels; neither op locates anything itself)")
        if t == "video.rotate":   # ADR-062: video-editing ROTATE takes a clockwise turn and / or a mirror, at least one
            if op.get("degrees") is not None and op["degrees"] not in (90, 180, 270):
                errs.append(f"video.rotate degrees {op['degrees']!r} must be 90, 180 or 270")
            if op.get("flip") is not None and op["flip"] not in ("h", "v"):
                errs.append(f"video.rotate flip {op['flip']!r} must be h or v")
            if op.get("degrees") is None and op.get("flip") is None:
                errs.append("video.rotate needs degrees and / or flip")
        if t == "video.resize" and op.get("width") is None:
            errs.append("video.resize needs width")
        if t in ("video.fit", "video.fill") and not op.get("aspect"):
            errs.append(f"{t} needs aspect")
        if t == "video.overlay":
            img = op.get("image")
            if not isinstance(img, str) or not img:
                errs.append("video.overlay needs image")
            else:
                if any(part == ".." for part in img.replace("\\", "/").split("/")):
                    errs.append("video.overlay image path contains '..'")
                if Path(img).suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    errs.append(f"video.overlay image must be PNG / JPEG: {Path(img).name}")
                if allowed_roots and not any(_under(img, r) for r in allowed_roots):
                    errs.append(f"video.overlay image is outside the allowed input roots: {Path(img).name}")
            if op.get("start") is not None and op.get("end") is not None and not float(op["start"]) < float(op["end"]):
                errs.append("video.overlay: start must be before end")
            if "prepare" in op:   # ADR-077: the image's preparation by imagemagick-skill, in the vocabulary read from its pinned contract
                errs.extend(f"video.overlay on {subj}: {e}" for e in prepare_problems(op["prepare"]))
        if t == IMAGE_INSERT_OP:   # ADR-070: an image still, its duration and frame, joined before / after the picture it names as input
            img = op.get("image")
            if not isinstance(img, str) or not img:
                errs.append(f"{t} needs image")
            else:
                if any(part == ".." for part in img.replace("\\", "/").split("/")):
                    errs.append(f"{t} image path contains '..'")
                if Path(img).suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    errs.append(f"{t} image must be PNG / JPEG: {Path(img).name}")
                if allowed_roots and not any(_under(img, r) for r in allowed_roots):
                    errs.append(f"{t} image is outside the allowed input roots: {Path(img).name}")
            dur = op.get("duration")
            if isinstance(dur, bool) or not isinstance(dur, (int, float)) or not IMAGE_INSERT_DURATION[0] <= float(dur) <= IMAGE_INSERT_DURATION[1]:
                errs.append(f"{t} duration {dur!r} must be within {IMAGE_INSERT_DURATION[0]:g}..{IMAGE_INSERT_DURATION[1]:g} s")
            if op.get("position") not in IMAGE_INSERT_POSITIONS:
                errs.append(f"{t} position {op.get('position')!r} must be before or after")
            for k in ("width", "height"):
                v = op.get(k)
                if isinstance(v, bool) or not isinstance(v, int) or v % 2 or v < 16:
                    errs.append(f"{t} {k} {v!r} must be an even integer ≥ 16 (the picture's own)")
            if op.get("fps") is None:
                errs.append(f"{t} needs fps (the picture's own)")
            ins = op.get("inputs") or []
            want = op.get("input")
            if len(ins) != 2 or want not in ins or (op.get("position") in IMAGE_INSERT_POSITIONS and ins.index(want) != (1 if op["position"] == "before" else 0)):
                errs.append(f"{t} inputs {ins!r} must be the still and its input, in the order of position {op.get('position')!r}")
    subjects_with = {}
    for op in ops:
        subjects_with.setdefault(op.get("asset"), set()).add(op.get("type"))
    for subj, types in subjects_with.items():
        if "video.fit" in types and "video.fill" in types:
            errs.append(f"video.fit and video.fill both on {subj} (conflicting request)")
    return errs


def finishing_operations(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [op for sec in ("color", "graphics", "captions") for op in (d.get(sec) or {}).get("operations") or []]


def check_finishing_operations(d: Dict[str, Any]) -> List[str]:
    """Per-type rules of the finishing sections (ADR-031): explicit types, allowlisted parameters (the schema closes the shapes;
    this checks the references and the chain), references that exist at that point (the subject after its edits), the fixed order
    colour → graphics → captions.burn, one render / one sidecar / one burn / one thumbnail per subject, a LUT / image inside the
    allowed input roots, graphics elements inside the subject's timeline, cues inside the delivered timeline, a burn-in whose
    input is an intermediate (never a source asset), the thumbnail timestamp inside the picture."""
    errs: List[str] = []
    assets = d["assets"]
    subjects_known: set = set(assets) | produced_subjects(d)
    allowed_roots = [str(Path(r)) for r in (d.get("execution") or {}).get("allowed_inputs") or []]
    on_audio = {op["asset"] for op in d["audio"]["operations"] if op.get("type") != "audio.loudness" or "input" in op}
    stage: Dict[str, int] = {}
    seen: Dict[tuple, int] = {}
    scopes: Dict[str, Dict[str, float]] = {}

    def scope_of(subj: str) -> Optional[Dict[str, float]]:
        if subj in scopes:
            return scopes[subj]
        sc = None
        for op in d["video"]["operations"]:
            if op.get("asset") == subj and op.get("temporal_scope"):
                sc = op["temporal_scope"]
        if sc is None and subj in assets:
            dur = (assets[subj].get("technical") or {}).get("duration")
            keep = next((op["keep"] for op in d["video"]["operations"] if op.get("asset") == subj and op.get("type") == "video.trim"), None)
            if keep:
                sc = {"start": 0.0, "end": round(sum(float(e) - float(s_) for s_, e in keep), 3)}
            elif dur:
                sc = {"start": 0.0, "end": float(dur)}
        return sc

    def planned_size(subj: str) -> Optional[Tuple[int, int]]:
        """The subject's picture after its edits, from the IR itself: an asset's own probe, or a programme's first input's."""
        src = subj if subj in assets else next((op["inputs"][0] for op in d["video"]["operations"] if op.get("output") == subj and op.get("inputs")), None)
        tech = (assets.get(src) or {}).get("technical") or {} if src else {}
        return picture_size(tech, d["video"]["operations"], subj)

    for op in (d.get("color") or {}).get("operations") or []:
        t, subj = op.get("type"), op.get("asset")
        if t not in COLOR_OPERATIONS:
            errs.append(f"unknown color operation type {t!r}")
            continue
        rank = COLOR_ORDER.index(t)
        if stage.get(subj, -1) >= rank:
            errs.append(f"{t} on {subj} is out of the fixed finishing order (colour {COLOR_ORDER} → graphics → captions.burn)")
        stage[subj] = rank
        if subj not in subjects_known:
            errs.append(f"{t} references {subj!r} before it exists")
        if subj in on_audio:
            errs.append(f"{t} on {subj}: the asset is on the audio production path (audio deliverable)")
        if subj in assets and not ((assets[subj].get("technical") or {}).get("video")):
            errs.append(f"{t} on {subj}: no video stream")
        if t == "color.lut":
            lut = op.get("lut")
            if not isinstance(lut, str) or Path(lut).suffix.lower() != ".cube" or any(part == ".." for part in lut.replace("\\", "/").split("/")):
                errs.append("color.lut needs a .cube LUT path without traversal")
            elif allowed_roots and not any(_under(lut, r) for r in allowed_roots):
                errs.append(f"color.lut LUT is outside the allowed input roots: {Path(lut).name}")
        if t == "color.retag" and not op.get("target"):
            errs.append("color.retag needs target")
    for op in (d.get("graphics") or {}).get("operations") or []:
        t, subj = op.get("type"), op.get("asset")
        if subj not in subjects_known:
            errs.append(f"{t} references {subj!r} before it exists")
        if subj in on_audio:
            errs.append(f"{t} on {subj}: the asset is on the audio production path (audio deliverable)")
        if subj in assets and not ((assets[subj].get("technical") or {}).get("video")):
            errs.append(f"{t} on {subj}: no video stream")
        if t == "graphics.render":
            if seen.get(("render", subj)):
                errs.append(f"more than one graphics.render on {subj} (one render per subject)")
            seen[("render", subj)] = 1
            stage[subj] = 10
            sc = scope_of(subj)
            ids = [e.get("id") for e in op.get("elements") or []]
            if len(ids) != len(set(ids)):
                errs.append(f"graphics.render on {subj}: duplicate element ids")
            for e in op.get("elements") or []:
                if e.get("type") not in ELEMENT_TYPES:
                    errs.append(f"graphics.render on {subj}: unknown element type {e.get('type')!r}")
                if not (float(e.get("start", 0)) < float(e.get("end", 0))):
                    errs.append(f"graphics.render element {e.get('id')}: start must be before end")
                if sc and float(e.get("end", 0)) > float(sc["end"]) + 0.01:
                    errs.append(f"graphics.render element {e.get('id')}: ends at {e.get('end')} after the subject's timeline ({sc['end']:.3f})")
                for typ, ref, exts, what in (("image_overlay", "image", (".png", ".jpg", ".jpeg"), "a PNG / JPEG"), ("video_overlay", "video", VIDEO_EXTENSIONS, "a video")):
                    if e.get("type") != typ:
                        continue
                    f = (e.get("parameters") or {}).get(ref)
                    if not isinstance(f, str) or Path(f).suffix.lower() not in exts or any(part == ".." for part in f.replace("\\", "/").split("/")):
                        errs.append(f"graphics.render element {e.get('id')}: {ref} must be {what} path without traversal")
                    elif allowed_roots and not any(_under(f, r) for r in allowed_roots):
                        errs.append(f"graphics.render element {e.get('id')}: {ref} is outside the allowed input roots")
                if e.get("animation") and e.get("type") not in FADE_TYPES:
                    errs.append(f"graphics.render element {e.get('id')}: a fade applies to {' / '.join(FADE_TYPES)} only")
                if e.get("type") in ELEMENT_TYPES:   # ADR-064: an explicit placement inside the picture the plan delivers to this point
                    for why in placement_problems(str(e["type"]), dict(e.get("parameters") or {}), planned_size(subj)):
                        errs.append(f"graphics.render element {e.get('id')}: {why}")
        elif t == "graphics.thumbnail":
            if seen.get(("thumbnail", subj)):
                errs.append(f"more than one graphics.thumbnail on {subj}")
            seen[("thumbnail", subj)] = 1
            sc = scope_of(subj)
            if sc and float(op.get("timestamp", 0)) > float(sc["end"]) + 0.01:
                errs.append(f"graphics.thumbnail on {subj}: timestamp {op.get('timestamp')} is after the picture ends ({sc['end']:.3f})")
            if (op.get("text") is not None or op.get("texts") or op.get("logo")) and not (op.get("width") and op.get("height")):
                errs.append(f"graphics.thumbnail on {subj}: a caption / text / logo needs the canvas size (width / height)")
            lg = op.get("logo")
            if isinstance(lg, dict):   # ADR-085: the logo image is the user's file, inside the allowed input roots, and its box touches the canvas
                img = str(lg.get("image") or "")
                if any(part == ".." for part in img.replace("\\", "/").split("/")) or Path(img).suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    errs.append(f"graphics.thumbnail on {subj}: the logo must be a PNG / JPEG path without '..'")
                elif allowed_roots and not any(_under(img, r) for r in allowed_roots):
                    errs.append(f"graphics.thumbnail on {subj}: the logo image is outside the allowed input roots: {Path(img).name}")
                if op.get("width") and op.get("height") and not (int(lg.get("x", 0)) < int(op["width"]) and int(lg.get("y", 0)) < int(op["height"])
                                                                 and int(lg.get("x", 0)) + int(lg.get("width", 0)) > 0 and int(lg.get("y", 0)) + int(lg.get("height", 0)) > 0):
                    errs.append(f"graphics.thumbnail on {subj}: the logo box lies outside the canvas")
        else:
            errs.append(f"unknown graphics operation type {t!r}")
    for op in (d.get("captions") or {}).get("operations") or []:
        t, subj = op.get("type"), op.get("asset")
        if subj not in subjects_known:
            errs.append(f"{t} references {subj!r} before it exists")
        if subj in on_audio:
            errs.append(f"{t} on {subj}: the asset is on the audio production path (audio deliverable)")
        if t == "captions.generate":
            if seen.get(("captions", subj)):
                errs.append(f"more than one captions.generate on {subj}")
            seen[("captions", subj)] = 1
            cues = op.get("cues") or []
            if not cues:
                errs.append(f"captions.generate on {subj}: no cues (an empty sidecar is never planned)")
            sc = scope_of(subj)
            speed = next((float(x["factor"]) for x in d["video"]["operations"] if x.get("asset") == subj and x.get("type") == "video.speed"), 1.0)
            end_limit = (float(sc["end"]) / speed if sc else None)
            prev = -1.0
            for c in cues:
                if not (float(c["start"]) < float(c["end"])):
                    errs.append(f"captions.generate cue {c.get('id')}: start must be before end")
                if float(c["start"]) < prev:
                    errs.append(f"captions.generate cue {c.get('id')}: overlaps the previous cue")
                prev = float(c["end"])
                if end_limit is not None and float(c["end"]) > end_limit + 0.05:
                    errs.append(f"captions.generate cue {c.get('id')}: ends at {c['end']} after the delivered timeline ({end_limit:.3f})")
            for src in op.get("sources") or []:
                if not any(o.get("id") == src and o.get("kind") == "transcript" for o in d["analysis"].get("observations") or []):
                    errs.append(f"captions.generate on {subj}: source {src} is not a transcript observation of this project")
        elif t in ("captions.burn", "captions.mux"):   # ADR-063: the mux is the same render (mode mux); one of the two per subject
            if seen.get(("burn", subj)):
                errs.append(f"more than one captions.burn / captions.mux on {subj}")
            seen[("burn", subj)] = 1
            if not seen.get(("captions", subj)):
                errs.append(f"{t} on {subj} without captions.generate")
            if op.get("input") in assets:
                errs.append(f"{t} on {subj}: the input is a source asset; the render takes an intermediate inside the workspace")
            if op.get("sidecar") != f"{subj}_captions":
                errs.append(f"{t} on {subj}: sidecar reference {op.get('sidecar')!r} is not the subject's captions")
            stage[subj] = 20
        else:
            errs.append(f"unknown captions operation type {t!r}")
    return errs


def check_timeline_exports(d: Dict[str, Any]) -> List[str]:
    """ADR-052: one timeline per subject, a known format, clips on known source assets (the timeline references the sources,
    never an intermediate) with forward ranges inside each source's duration, and the recorded duration their sum."""
    errs: List[str] = []
    assets = d.get("assets") or {}
    subjects = set(assets) | produced_subjects(d)
    seen: set = set()
    for op in (d.get("delivery") or {}).get("timelines") or []:
        subj = op.get("asset")
        if op.get("type") != TIMELINE_SUBJECT:
            errs.append(f"unknown timeline operation type {op.get('type')!r}")
            continue
        if subj not in subjects:
            errs.append(f"timeline.export references unknown subject {subj}")
        if subj in seen:
            errs.append(f"more than one timeline.export on {subj}")
        seen.add(subj)
        if op.get("format") not in TIMELINE_FORMATS:
            errs.append(f"timeline.export on {subj}: unknown format {op.get('format')!r}")
        clips = op.get("clips") or []
        if not clips:
            errs.append(f"timeline.export on {subj}: no clips")
        for n, c in enumerate(clips):
            a = assets.get(c.get("asset"))
            if a is None:
                errs.append(f"timeline.export on {subj}: clip {n} names {c.get('asset')!r}, which is not a source asset")
                continue
            dur = (a.get("technical") or {}).get("duration")
            try:
                s_, e_ = float(c["in"]), float(c["out"])
            except (KeyError, TypeError, ValueError):
                errs.append(f"timeline.export on {subj}: clip {n} needs numeric in / out")
                continue
            if not 0.0 <= s_ < e_ or (dur is not None and e_ > float(dur) + 0.001):
                errs.append(f"timeline.export on {subj}: clip {n} range {s_:g}..{e_:g} is not inside its source (0..{dur})")
        if clips and abs(sum(float(c.get("out", 0)) - float(c.get("in", 0)) for c in clips) - float(op.get("timeline_duration") or 0.0)) > 0.002:
            errs.append(f"timeline.export on {subj}: timeline_duration does not match its clips")
    return errs


def check_take_assembly(d: Dict[str, Any]) -> List[str]:
    """ADR-053: every pick keeps a forward range inside its source and lists its own take among the scored takes; a concat assembled
    from picks joins exactly those picks in line order; every snapped join lies on a supported beat of the music bed's recorded beat
    measurement, and the trims really end there in programme time (the cut list, not only the decision, is on the beat)."""
    errs: List[str] = []
    assets = d.get("assets") or {}
    decs = {x["id"]: x for x in d.get("decisions") or []}
    picks = [x for x in decs.values() if x.get("subject") == PICK_SUBJECT and x.get("type") == "TRANSFORM"]
    for x in picks:
        p = x.get("params") or {}
        a = assets.get(p.get("asset_id"))
        if a is None:
            errs.append(f"takes.pick {x['id']} names {p.get('asset_id')!r}, which is not a source asset")
            continue
        dur = (a.get("technical") or {}).get("duration")
        k = p.get("keep") or []
        if len(k) != 2 or not 0.0 <= float(k[0]) < float(k[1]) or (dur is not None and float(k[1]) > float(dur) + 0.001):
            errs.append(f"takes.pick {x['id']}: keep {k} is not a forward range inside its source (0..{dur})")
        if p.get("asset_id") not in [t.get("asset_id") for t in p.get("takes") or []]:
            errs.append(f"takes.pick {x['id']}: the picked take is not among its scored takes")
        if any("score" not in t or not t.get("factors") for t in p.get("takes") or []):
            errs.append(f"takes.pick {x['id']}: a take lacks its score / factors")
    ops = (d.get("video") or {}).get("operations") or []
    for c in (x for x in decs.values() if x.get("subject") == "video.concat" and (x.get("params") or {}).get("picks")):
        want = [decs[i]["params"]["asset_id"] for i in c["params"]["picks"] if i in decs]
        op = next((o for o in ops if o.get("type") == "video.concat" and c["id"] in (o.get("decision_ids") or [])), None)
        if op is not None and list(op.get("inputs") or []) != want:
            errs.append(f"video.concat {c['id']}: inputs {op.get('inputs')} are not the picks in line order {want}")
    for sn in (x for x in decs.values() if x.get("subject") == SNAP_SUBJECT and x.get("type") == "TRANSFORM"):
        p = sn.get("params") or {}
        obs = next((o for o in (d.get("analysis") or {}).get("observations") or [] if o.get("asset_id") == p.get("music") and o.get("kind") == BEAT_KIND), None)
        if obs is None:
            errs.append(f"takes.beat_snap {sn['id']}: no recorded beat measurement of the music bed {p.get('music')}")
            continue
        beats = [float(b) for b in (obs.get("data") or {}).get("supported_beats") or []]
        t = 0.0
        for j in p.get("joins") or []:
            trim = next((o for o in ops if o.get("type") == "video.trim" and o.get("asset") == j.get("asset_id")), None)
            if trim is not None:
                t += sum(float(e) - float(s_) for s_, e in trim["keep"])
            if not j.get("snapped"):
                continue
            if not any(abs(float(j["to"]) - b) <= 0.001 for b in beats):
                errs.append(f"takes.beat_snap {sn['id']}: join after {j.get('asset_id')} at {j.get('to')}s is not a supported beat of the measurement")
            elif trim is not None and abs(t - float(j["to"])) > 0.002:
                errs.append(f"takes.beat_snap {sn['id']}: the cut list ends {j.get('asset_id')} at {t:.3f}s in the programme, not on its beat {j.get('to')}s")
    return errs + check_take_crossfade(d)


def check_take_crossfade(d: Dict[str, Any]) -> List[str]:
    """ADR-054: the join crossfades of a take assembly keep the picture and stay in sync with it. Rebuilt from the IR itself (the
    picture's cut list = the video.trim ranges, the audio = the recorded CUT spans and CONCAT joins), never from the decision's
    own arithmetic: the picture of every pick is exactly its trim (frame-accurate); its audio span is that range plus the join
    handles, inside the source; at each join out_handle + in_handle = that join's crossfade; replaying the CONCATs puts every
    pick's audio at its picture's programme time (within SYNC_TOLERANCE) and the programme audio ends with the picture; the mux
    puts that audio under the assembled picture."""
    errs: List[str] = []
    decs = {x["id"]: x for x in d.get("decisions") or []}
    rec = (d.get("audio") or {}).get("take_crossfade")
    live = [x for x in decs.values() if x.get("subject") == CROSSFADE_SUBJECT and x.get("type") == "TRANSFORM" and x.get("status") != "REJECTED"]
    ops = (d.get("video") or {}).get("operations") or []
    concat = next((o for o in ops if o.get("type") == "video.concat"), None)
    if rec is None:
        if live and concat is not None and any(x["id"] in (concat.get("decision_ids") or []) for x in decs.values() if x.get("subject") == PICK_SUBJECT):
            errs.append(f"takes.crossfade {live[0]['id']} is decided but the plan has no crossfade record (the joins would keep hard-cut audio)")
        return errs
    if not live or not set(x["id"] for x in live) & set(rec.get("decision_ids") or []):
        return errs + ["take_crossfade: the record cites no live takes.crossfade decision"]
    if concat is None:
        return errs + ["take_crossfade: there is no assembled picture (video.concat) to put the audio under"]
    assets = d.get("assets") or {}
    spans = rec.get("spans") or []
    joins = rec.get("joins") or []
    cf = float(rec.get("crossfade") or 0.0)
    if [sp.get("asset") for sp in spans] != list(concat.get("inputs") or []):
        errs.append(f"take_crossfade: spans {[sp.get('asset') for sp in spans]} are not the assembly's inputs {concat.get('inputs')}")
        return errs
    if len(joins) != len(spans) - 1:
        return errs + [f"take_crossfade: {len(joins)} join(s) for {len(spans)} pick(s)"]
    t = 0.0
    starts: List[float] = []
    for n, sp in enumerate(spans):
        a = sp["asset"]
        trim = next((o for o in ops if o.get("type") == "video.trim" and o.get("asset") == a), None)
        dur = float(((assets.get(a) or {}).get("technical") or {}).get("duration") or 0.0)
        keep = [[float(x), float(y)] for x, y in (trim or {}).get("keep") or []] or [[0.0, dur]]
        if len(keep) != 1 or any(abs(float(u) - v) > 0.001 for u, v in zip(sp["video"], keep[0])):
            errs.append(f"take_crossfade: the picture of {a} is {keep}, the span's picture range {sp['video']} (the cut list must not move)")
        if trim is not None and not trim.get("accurate"):
            errs.append(f"take_crossfade: {a} is cut at keyframes; a crossfaded pick's picture must be cut frame-accurately to stay on its audio")
        starts.append(t)
        t += keep[0][1] - keep[0][0]
        ai, ao = float(sp["audio"][0]), float(sp["audio"][1])
        if not (-1e-9 <= ai < ao <= dur + 1e-6):
            errs.append(f"take_crossfade: audio span {sp['audio']} of {a} is not inside its source (0..{dur:g})")
        if abs((float(sp["video"][0]) - ai) - float(sp.get("in_handle", 0))) > 1e-6 or abs((ao - float(sp["video"][1])) - float(sp.get("out_handle", 0))) > 1e-6 \
                or float(sp.get("in_handle", 0)) < 0 or float(sp.get("out_handle", 0)) < 0:
            errs.append(f"take_crossfade: audio span {sp['audio']} of {a} is not its picture range {sp['video']} plus its handles")
        want_remove = [[x, y] for x, y in ([0.0, ai], [ao, dur]) if y - x > 1e-6]
        if sp.get("output") != a and [[round(float(x), 6), round(float(y), 6)] for x, y in sp.get("remove") or []] != [[round(x, 6), round(y, 6)] for x, y in want_remove]:
            errs.append(f"take_crossfade: the CUT of {a} removes {sp.get('remove')}, not everything outside its span {sp['audio']}")
    if spans and (float(spans[0].get("in_handle", 0)) > 1e-9 or float(spans[-1].get("out_handle", 0)) > 1e-9):
        errs.append("take_crossfade: the programme audio must start with the first picture and end with the last (no handle outside the programme)")
    for n, j in enumerate(joins):
        c = float(j.get("crossfade") or 0.0)
        if c not in (0.0, cf):
            errs.append(f"take_crossfade: join {n + 1} has a {c:g}s crossfade, not {cf:g}s or a hard cut")
        if abs(float(spans[n].get("out_handle", 0)) + float(spans[n + 1].get("in_handle", 0)) - c) > 1e-6:
            errs.append(f"take_crossfade: join {n + 1} ({j.get('after')} → {j.get('before')}): handles {spans[n].get('out_handle')} + {spans[n + 1].get('in_handle')} "
                        f"do not make its {c:g}s overlap, so the audio after it would drift off the picture")
        if not j.get("crossfaded") and not j.get("why"):
            errs.append(f"take_crossfade: hard-cut join {n + 1} does not say why")
    # replay the CONCATs (audio-production: each input starts `crossfade` before the previous one ends) → where each pick's audio lands
    placed: Dict[str, Tuple[float, List[Tuple[int, float]]]] = {sp["output"]: (float(sp["audio"][1]) - float(sp["audio"][0]), [(n, 0.0)]) for n, sp in enumerate(spans)}
    for c in rec.get("concats") or []:
        x = float(c.get("crossfade") or 0.0)
        off, members = 0.0, []
        ins = list(c.get("inputs") or [])
        if any(i not in placed for i in ins) or len(ins) < 2:
            errs.append(f"take_crossfade: concat {c.get('output')} joins {ins}, which are not earlier spans / joins")
            return errs
        for k, i in enumerate(ins):
            length, mem = placed.pop(i)
            if x > 0 and length <= x + 1e-9:
                errs.append(f"take_crossfade: concat {c.get('output')} input {i} ({length:.3f}s) is not longer than its {x:g}s crossfade")
            members += [(n, off + o) for n, o in mem]
            off += length - (x if k < len(ins) - 1 else 0.0)
        placed[c["output"]] = (off, members)
    if len(placed) != 1 or rec.get("audio") not in placed:
        return errs + [f"take_crossfade: the joins do not end in one programme audio {rec.get('audio')!r} ({sorted(placed)})"]
    length, members = placed[rec["audio"]]
    total = float(concat.get("timeline_duration") or t)
    if [n for n, _ in members] != list(range(len(spans))):
        errs.append("take_crossfade: the programme audio does not hold the picks in programme order")
    for n, off in members:
        drift = off + float(spans[n].get("in_handle", 0)) - starts[n]
        if abs(drift) > SYNC_TOLERANCE:
            errs.append(f"take_crossfade: the audio of {spans[n]['asset']} lands {drift * 1000:+.1f} ms off its picture (A/V sync tolerance {SYNC_TOLERANCE * 1000:g} ms)")
    if abs(length - total) > SYNC_TOLERANCE or abs(t - total) > SYNC_TOLERANCE or abs(float(rec.get("timeline_duration") or 0.0) - total) > SYNC_TOLERANCE:
        errs.append(f"take_crossfade: programme audio {length:.4f}s / cut list {t:.4f}s / record {rec.get('timeline_duration')} do not match the assembled picture {total:.4f}s")
    mux = rec.get("mux") or {}
    if mux.get("video") != concat.get("output") or mux.get("audio") != rec.get("audio") or not mux.get("output"):
        errs.append(f"take_crossfade: the mux puts {mux.get('audio')!r} under {mux.get('video')!r}, not the programme audio under the assembled picture {concat.get('output')!r}")
    return errs


def check_plan_dependencies(d: Dict[str, Any]) -> List[str]:
    """Every step input that is not a source asset must be produced by a step the consuming step (transitively) depends on:
    the plan's dependency graph, not the listing order, is what makes a multi-Skill chain executable in order (ADR-031)."""
    errs: List[str] = []
    steps = (d.get("plan") or {}).get("steps") or []
    by_id = {s["id"]: s for s in steps}
    producer = {o: s["id"] for s in steps for o in (s.get("outputs") or [])}
    assets = set(d.get("assets") or {})

    def deps(sid: str, acc: set) -> set:
        for x in by_id.get(sid, {}).get("depends_on") or []:
            if x not in acc:
                acc.add(x)
                deps(x, acc)
        return acc

    for s in steps:
        closure = deps(s["id"], set())
        for i in s.get("inputs") or []:
            if i in assets:
                continue
            p = producer.get(i)
            if p is None:
                errs.append(f"step {s['id']}: input {i!r} is produced by no step")
            elif p not in closure:
                errs.append(f"step {s['id']}: consumes {i!r} from {p} without depending on it")
    return errs


def _under(path: str, root: str) -> bool:
    try:
        p, r = Path(path).resolve(), Path(root).resolve()
    except OSError:
        return False
    return p == r or r in p.parents


def load_schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def historical_decision_ids(doc: Dict[str, Any]) -> set:
    """Decision ids reviewed in an earlier plan version, as recorded by `revise` (revision.history[].reviews and
    rejected_decision_ids). They are history: a USER_DECISION event may cite them, an operation or a step never may."""
    out: set = set()
    for h in (doc.get("revision") or {}).get("history") or []:
        out |= set((h.get("reviews") or {}).keys())
        out |= set(h.get("rejected_decision_ids") or [])
    return out


def validate_ir(ir: ProjectIR, caps: Optional[Dict[str, Any]] = None, check_paths: bool = True, registry=None, supports=None) -> ValidationReport:
    """registry: SkillRegistry (plan steps must cite implemented skills and their declared tools); supports: callable(tool) -> bool
    from the tool router (the named tool must be executable here). Both optional so pure schema checks stay cheap."""
    rep = ValidationReport()
    try:
        import jsonschema
        validator = jsonschema.Draft202012Validator(load_schema())
        for err in sorted(validator.iter_errors(ir.doc), key=lambda e: list(e.absolute_path)):
            path = "/".join(str(p) for p in err.absolute_path) or "<root>"
            rep.errors.append(f"schema: {path}: {err.message}")
    except ImportError:
        rep.warnings.append("jsonschema not installed; schema validation skipped")
    if rep.errors:
        return rep
    d = ir.doc
    assets = d["assets"]
    ids = {x["id"] for x in d["decisions"]}
    rejected = {x["id"] for x in d["decisions"] if x["status"] == "REJECTED"}
    timelines = list(d["delivery"].get("timelines") or [])
    xf_rec = [d["audio"]["take_crossfade"]] if d["audio"].get("take_crossfade") else []   # ADR-054
    for op in d["video"]["operations"] + d["audio"]["operations"] + ir.finishing_operations() + d["delivery"]["targets"] + timelines + xf_rec:
        for did in op.get("decision_ids") or []:
            if did in rejected:
                rep.errors.append(f"{op.get('type') or 'delivery.' + op.get('id', '?')} cites REJECTED decision {did}; re-plan (revise) before rendering")
    # semantic: operations reference known assets (or the concat programme once it exists) and decisions, ranges inside durations
    rep.errors += check_video_operations(d)
    rep.errors += check_audio_operations(d)
    rep.errors += check_finishing_operations(d)
    rep.errors += check_timeline_exports(d)
    rep.errors += check_take_assembly(d)
    rep.errors += check_plan_dependencies(d)
    for op in timelines:
        for did in op.get("decision_ids") or []:
            if did not in ids:
                rep.errors.append(f"operation {op.get('type')} cites unknown decision {did}")
    for op in d["video"]["operations"] + d["audio"]["operations"] + finishing_operations(d):
        if op["asset"] not in assets and op["asset"] not in produced_subjects(d):
            rep.errors.append(f"operation {op['type']} references unknown asset {op['asset']}")
            continue
        for did in op.get("decision_ids") or []:
            if did not in ids:
                rep.errors.append(f"operation {op['type']} cites unknown decision {did}")
    # inferences must cite evidence that exists
    obs_ids = {o["id"] for o in d["analysis"]["observations"]} | {e["id"] for e in d["timeline"]["events"]} | {i["id"] for i in d["analysis"]["inferences"]} \
        | {c["id"] for c in d["analysis"].get("contexts") or []}
    for inf in d["analysis"]["inferences"]:
        for ev in inf["evidence"]:
            if ev not in obs_ids:
                rep.errors.append(f"inference {inf['id']} cites missing evidence {ev}")
    # decision engine invariants (evidence present and known, type, grounding, BLOCK ⇔ BLOCKED, executable citations only)
    rep.errors += check_decisions(d)
    # decisions: BLOCK is never executable; CONFIRM must be approved before render (reported as warning here)
    for dec in d["decisions"]:
        if dec["approval"] == "BLOCK":
            rep.warnings.append(f"decision {dec['id']} ({dec['subject']}) is BLOCKED: {dec['reason']}")
        elif dec["approval"] == "CONFIRM" and dec["status"] == "PROPOSED":
            rep.warnings.append(f"decision {dec['id']} ({dec['subject']}) needs confirmation before execution")
    # delivery presets / platforms must be names the tool knows (catalog-level check, no numbers duplicated)
    known_presets = {"youtube", "youtube4k", "reels", "x", "prores", "h265", "gif"}
    known_platforms = {"youtube", "shorts", "reels", "tiktok", "x", "linkedin", "broadcast", "podcast", "custom"}
    for t in d["delivery"]["targets"]:
        if t.get("preset") and t["preset"] not in known_presets:
            rep.errors.append(f"delivery target {t['id']}: unknown preset {t['preset']}")
        if t.get("platform") not in known_platforms:
            rep.errors.append(f"delivery target {t['id']}: unknown platform {t.get('platform')}")
    # ProductionPlan structure / boundaries (ids, order, dependencies, evidence, decisions, parameters, scopes, status)
    rep.errors += validate_plan(d, registry=registry)
    # plan steps: Skill → Tool consistency (the compiler takes tools from the steps, so this is the execution contract)
    if registry is not None:
        known = set(registry.names())
        for s in d["plan"]["steps"]:
            if s["skill"] not in known:
                rep.errors.append(f"plan step {s['id']} cites unknown skill {s['skill']}")
                continue
            spec = registry.get(s["skill"])
            if not spec.implemented:
                rep.errors.append(f"plan step {s['id']} cites skill {s['skill']} which is declared for phase {spec.phase} and not implemented")
            if not s.get("tool"):
                rep.errors.append(f"plan step {s['id']} ({s['skill']}) has no selected tool")
            elif s["tool"] not in spec.tools:
                rep.errors.append(f"plan step {s['id']}: tool {s['tool']} is not a declared tool of skill {s['skill']} ({', '.join(spec.tools)})")
            elif supports is not None and not supports(s["tool"]):
                rep.errors.append(f"plan step {s['id']}: no registered adapter supports {s['tool']}")
            elif registry.packages() and registry.tool(s["tool"]) is None:
                rep.errors.append(f"plan step {s['id']}: tool {s['tool']} is not declared by any registered skill package")
    # every executable operation must have a step (otherwise the compiler cannot know its tool)
    step_keys = {(s["skill"], (s.get("params") or {}).get("asset") or (s.get("params") or {}).get("target")) for s in d["plan"]["steps"]}
    for op in d["video"]["operations"]:
        if op["type"] == "video.trim" and ("silence_cleanup", op["asset"]) not in step_keys and ("video_trim", op["asset"]) not in step_keys:   # ADR-062: either skill runs it
            rep.errors.append(f"video.trim on {op['asset']} has no plan step")
        if op["type"] == "video.overlay" and isinstance(op.get("prepare"), dict):   # ADR-077: every preparation of the image has its step
            for sk, _tool, _params in prepare_chain(op["prepare"]):
                if (sk, op["asset"]) not in step_keys:
                    rep.errors.append(f"the overlay image's {sk} on {op['asset']} has no plan step")
        if op["type"] in SKILL_OF and (SKILL_OF[op["type"]], op["asset"]) not in step_keys:
            rep.errors.append(f"{op['type']} on {op['asset']} has no plan step")
    for op in d["audio"]["operations"]:
        if op["type"] == "audio.loudness" and (("audio_normalize" if "input" in op else "loudness_normalization"), op["asset"]) not in step_keys:
            rep.errors.append(f"audio.loudness on {op['asset']} has no plan step")
        if op["type"] in AUDIO_SKILL_OF and op["type"] != "audio.loudness" and op["type"] not in ("audio.mono", "audio.stereo", "audio.downmix") and (AUDIO_SKILL_OF[op["type"]], op["asset"]) not in step_keys:
            rep.errors.append(f"{op['type']} on {op['asset']} has no plan step")
        if op["type"] in ("audio.mono", "audio.stereo", "audio.downmix") and (AUDIO_SKILL_OF[op["type"]], op["asset"]) not in step_keys:
            rep.errors.append(f"{op['type']} on {op['asset']} has no plan step")
        if op["type"] in VOICE_OPERATIONS and (VOICE_OPERATIONS[op["type"]]["skill"], op["asset"]) not in step_keys:   # ADR-073
            rep.errors.append(f"{op['type']} on {op['asset']} has no plan step")
    for t in d["delivery"]["targets"]:
        if t.get("preset") and ("delivery_export", t["id"]) not in step_keys:
            rep.errors.append(f"delivery target {t['id']} has no export step")
    for op in (d.get("color") or {}).get("operations") or []:
        if (COLOR_SKILL_OF.get(op["type"], "?"), op["asset"]) not in step_keys:
            rep.errors.append(f"{op['type']} on {op['asset']} has no plan step")
    for op in (d.get("graphics") or {}).get("operations") or []:
        want = GRAPHICS_SKILL if op["type"] == "graphics.render" else (THUMBNAIL_RENDER_SKILL if op.get("text") is not None else THUMBNAIL_FRAME_SKILL)
        if (want, op["asset"]) not in step_keys:
            rep.errors.append(f"{op['type']} on {op['asset']} has no plan step")
    for op in (d.get("captions") or {}).get("operations") or []:
        want = GENERATE_SKILL if op["type"] == "captions.generate" else MUX_SKILL if op["type"] == "captions.mux" else BURN_SKILL
        if (want, op["asset"]) not in step_keys:
            rep.errors.append(f"{op['type']} on {op['asset']} has no plan step")
    for op in timelines:
        if (TIMELINE_SKILL, op.get("asset")) not in step_keys:
            rep.errors.append(f"timeline.export on {op.get('asset')} has no plan step")
    qc = (d.get("qa") or {}).get("qc") or {}
    if qc.get("enabled"):
        qc_steps = [s for s in d["plan"]["steps"] if s["skill"] == QC_SKILL]
        for subj, row in (qc.get("subjects") or {}).items():
            for target in row.get("targets") or []:
                if not any((s.get("params") or {}).get("asset") == subj and (s.get("params") or {}).get("target") == target for s in qc_steps):
                    rep.errors.append(f"qa.qc: {subj}/{target} has no qc_check step")
        for sc, row in (qc.get("sidecars") or {}).items():
            if not any((s.get("params") or {}).get("artifact") == sc for s in qc_steps):
                rep.errors.append(f"qa.qc: sidecar {sc} has no qc_check step")
        outputs = {o.get("logical"): o for o in d["plan"].get("outputs") or []}
        for subj, row in (qc.get("packages") or {}).items():   # ADR-065: a deliverable set names the subject's own sidecar / thumbnail, and has its steps
            for role, want in (("captions", "CAPTIONS"), ("thumbnail", "THUMBNAIL")):
                ref = row.get(role)
                if ref is not None and ((outputs.get(ref) or {}).get("role") != want or ((outputs.get(ref) or {}).get("expected") or {}).get("source") != subj):
                    rep.errors.append(f"qa.qc: package of {subj} names {ref} as its {role}, which is not a planned {want} output of {subj}")
            if not (row.get("captions") or row.get("thumbnail")):
                rep.errors.append(f"qa.qc: package of {subj} has nothing beside the deliverable")
            for target in row.get("targets") or []:
                if not any((s.get("params") or {}).get("asset") == subj and (s.get("params") or {}).get("target") == target and (s.get("params") or {}).get("kind") == "delivery_package"
                           for s in qc_steps):
                    rep.errors.append(f"qa.qc: package {subj}/{target} has no qc_check step")
    # temporal layer: events are validated domain objects on existing assets, within their duration, with real evidence
    durations = {aid: (a.get("technical") or {}).get("duration") for aid, a in assets.items()}
    known_evidence = {o.get("id") for o in d["analysis"].get("observations") or []} | {x.get("id") for x in d["decisions"]} | {i.get("id") for i in d["analysis"].get("inferences") or []}
    # ADR-034: a USER_DECISION event carried from an earlier plan version cites the decision *of that version*; it is resolved against
    # the review history (revision.history[].reviews / rejected_decision_ids), never promoted to an active decision of this version
    historical = historical_decision_ids(d)
    current_version = int((d.get("plan") or {}).get("version") or 1)
    events: Dict[str, Event] = {}
    for raw in d["timeline"].get("events") or []:
        try:
            ev = classify(Event.from_dict(raw))
        except (TypeError, ValueError) as ex:
            rep.errors.append(f"event {raw.get('id')}: {ex}")
            continue
        known = known_evidence
        if ev.type == "USER_DECISION":
            reviewed_version = (ev.metadata or {}).get("plan_version")
            if isinstance(reviewed_version, int) and reviewed_version < current_version:
                known = known_evidence | historical
            elif any(x not in known_evidence and x in historical for x in ev.evidence):
                rep.errors.append(f"event {ev.id}: cites a decision of an earlier plan version ({', '.join(ev.evidence)}) but is not recorded as that version's review")
        for err in validate_event(ev, durations, known - {ev.id}):
            rep.errors.append(f"event {ev.id}: {err}")
        events[ev.id] = ev
    for raw in d["timeline"].get("sessions") or []:
        try:
            ses = Session.from_dict(raw)
        except (TypeError, ValueError) as ex:
            rep.errors.append(f"session {raw.get('id')}: {ex}")
            continue
        for err in validate_session(ses, d["project"].get("id"), durations, events):
            rep.errors.append(f"session {ses.id}: {err}")
    # observations are measurements: their source is a tool id + version, never an AI provider; AI output is AI_GENERATED
    for o in d["analysis"].get("observations") or []:
        src = str(o.get("source") or "")
        if "@" not in src or src.startswith("ai"):
            rep.errors.append(f"observation {o.get('id')} has no tool source ({src!r}); only tool measurements may be OBSERVED")
        if o.get("provenance", "OBSERVED") != "OBSERVED":
            rep.errors.append(f"observation {o.get('id')} has provenance {o.get('provenance')!r}; observations are always OBSERVED")
    # production contexts: derived situations whose references all exist and whose ids match their content
    from ..context import ProductionContext, validate_context
    ev_index = {e["id"]: e for e in d["timeline"].get("events") or []}
    for raw in d["analysis"].get("contexts") or []:
        try:
            rep.errors += validate_context(ProductionContext.from_dict(raw), ev_index, durations, {o["id"] for o in d["analysis"].get("observations") or []},
                                           {i["id"] for i in d["analysis"].get("inferences") or []})
        except (KeyError, TypeError, ValueError) as ex:
            rep.errors.append(f"context {raw.get('id')}: malformed: {ex}")
    for i in d["analysis"].get("inferences") or []:
        if str(i.get("kind", "")).startswith("ai_recommendation:") and i.get("provenance") != "AI_GENERATED":
            rep.errors.append(f"inference {i.get('id')} from an AI provider must carry provenance AI_GENERATED")
        if not i.get("evidence"):
            rep.errors.append(f"inference {i.get('id')} cites no evidence")
    # paths
    if check_paths:
        for a in assets.values():
            if not Path(a["path"]).exists():
                rep.errors.append(f"asset {a['id']} path missing: {a['path']}")
    # capabilities: a plan must not silently depend on unavailable capabilities
    if caps is not None:
        # skill-level needs come from the registry (production skill + owning package); preset-level encoder needs below
        needed = set()
        if registry is not None:
            for st in d["plan"]["steps"]:
                if st["skill"] in registry.names():
                    needed.update(registry.get(st["skill"]).required_capabilities)
                pkg = registry.package(str(st.get("tool") or "").split("/", 1)[0])
                if pkg:
                    needed.update(pkg.capabilities)
                    ts = registry.tool(st["tool"]) if st.get("tool") else None
                    if ts:
                        needed.update(ts.required_capabilities)
        if d["video"]["operations"] or any(t.get("preset") for t in d["delivery"]["targets"]):
            needed.add("encoder:libx264")
        _REFERENCE_ENGINE_ONLY_OPS = ("video.switch", "video.grid", "video.redact", "video.deinterlace", "video.stabilize")   # ADR-045 / ADR-046: no "tool" key in OPERATIONS (video.crop moved to video-editing, ADR-079)
        if any(op["type"] in SKILL_OF and op["type"] not in _REFERENCE_ENGINE_ONLY_OPS for op in d["video"]["operations"]):
            needed.add("video-editing")   # the editing operations exist only in video-editing-skill (ADR-029): UNKNOWN is not AVAILABLE
                                           # (the reference-engine-only ops above are the exception -- their own capability need already came from the registry above)
        if any(op["type"] in AUDIO_OPERATIONS and (op["type"] != "audio.loudness" or "input" in op) for op in d["audio"]["operations"]):
            needed.add("audio-production")   # the audio production path exists only in audio-production-skill (ADR-030)
        if any(op.get("prepare") for op in d["video"]["operations"] if op["type"] == "video.overlay"):
            needed.add("imagemagick-skill")   # ADR-077: the overlay image's preparation exists only in imagemagick-skill
        if any(op["type"] in VOICE_OPERATIONS for op in d["audio"]["operations"]):
            needed.add("voice-enhance")   # ADR-073: speech enhancement exists only in voice-enhance-skill
        if d["audio"]["operations"]:
            needed.add("filter:loudnorm")
        if (d.get("color") or {}).get("operations"):
            needed.add("color-grading")     # the finishing Skills (ADR-031 / ADR-032): UNKNOWN is not AVAILABLE
        if any(op["type"] == "graphics.render" for op in (d.get("graphics") or {}).get("operations") or []):
            needed.add("motion-graphics")
        if any(op["type"] == "graphics.thumbnail" for op in (d.get("graphics") or {}).get("operations") or []):
            needed.add("thumbnail")
        if (d.get("captions") or {}).get("operations"):
            needed.add("subtitle")
        if ((d.get("qa") or {}).get("qc") or {}).get("enabled"):
            needed.add("qc")
        if any(t.get("preset") == "prores" for t in d["delivery"]["targets"]):
            needed.add("encoder:prores_ks")
        if any(t.get("preset") == "h265" for t in d["delivery"]["targets"]):
            needed.add("encoder:libx265")
        for n in sorted(needed):
            st = getattr(caps.get(n), "status", None) or (caps.get(n) or {}).get("status") if isinstance(caps.get(n), dict) else getattr(caps.get(n), "status", "UNKNOWN")
            if st == "MISSING":
                rep.errors.append(f"required capability MISSING: {n}")
            elif st in ("UNKNOWN", "DEGRADED"):
                rep.warnings.append(f"capability {n} is {st}")
    return rep
