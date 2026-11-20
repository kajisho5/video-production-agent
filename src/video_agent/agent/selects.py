"""Batch selects (ADR-124, Velisa ADR-033): the usable shots of 1-50 clips, each kept or rejected with its reasons, shown before anything renders.

    clips (a folder or a list, 1-50)
      → per clip, the measuring Skills (nothing is measured here):
          media-analysis-skill `scene_detection` (shot boundaries), `picture_quality` (blur, luma, camera jitter per window),
          `silence`, `speech_regions` (UNKNOWN without its optional extra); qc-skill `inspect` (black and frozen frames)
      → rule `selects@1` (this module): per shot, the facts in its span, flags, reject reasons, a score; then the selection
        (source order; `min_score`, `max_duration` by score)
      → the selects file (`video-agent/selects@1`) and the lines the user approves (every shot, why, the total duration)
      → `--set edit.selects=<file>` → one `selects.keep` decision per clip with kept shots (CONFIRM at least) and a `video.concat`
        of those clips in source order → each clip trimmed frame-accurately to its kept shots, joined by the existing assembly path

Everything here is vocabulary, file checks and arithmetic over recorded measurements. The thresholds are this rule's defaults, chosen
here and not taken from a standard; each is an option and is written into the file.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

KEY = "edit.selects"
SUBJECT = "selects.keep"
DOC_SCHEMA = "video-agent/selects@1"
RULE = "selects@1"
APPROVAL_KEY = ("selects.keep.approval", "CONFIRM")   # leaving shots out is an editorial choice: CONFIRM at least
RISK = "MEDIUM"
MAX_CLIPS = 50
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".m4v", ".avi", ".mxf", ".webm", ".mts", ".m2ts", ".mpg", ".mpeg")

DEFAULTS: Dict[str, Any] = {
    "min_shot": 1.0,            # seconds: a shorter shot is rejected (too short to use)
    "black_share": 0.5,         # share of the shot covered by black frames (qc-skill) at which it is rejected
    "freeze_share": 0.5,        # share covered by frozen frames (qc-skill) at which it is rejected
    "blur_ratio": 1.3,          # soft: the shot's median blur >= ratio x the batch's reference ...
    "blur_percentile": 25.0,    # ... the reference being this percentile of the shots' medians (blurdetect depends on the content)
    "max_blur": None,           # or an absolute blurdetect value the user gives (soft when above it)
    "under_luma": 40.0,         # mean luma (8-bit) below this: under-exposed
    "over_luma": 210.0,         # mean luma above this: over-exposed
    "max_jitter": 1.0,          # median camera jitter, % of the width, above this: shaky ...
    "min_motion_agreement": 0.6,  # ... judged only when this share of the picture blocks agreed with the measured shift (else: not judged)
    "dead_air_share": 0.9,      # silent over at least this share of the shot (or no speech when speech was measured): dead air
    "reject_dead_air": False,   # dead air is flagged; it rejects only when asked (silent B-roll is usable)
    "min_score": None,          # reject usable shots scoring below this
    "max_duration": None,       # seconds: keep the best-scored usable shots that fit, then source order
}
LIMITS = {"min_shot": (0.0, 60.0), "black_share": (0.01, 1.0), "freeze_share": (0.01, 1.0), "blur_ratio": (1.0, 10.0), "blur_percentile": (0.0, 100.0),
          "max_blur": (0.1, 1000.0), "under_luma": (0.0, 255.0), "over_luma": (0.0, 255.0), "max_jitter": (0.01, 100.0), "min_motion_agreement": (0.0, 1.0), "dead_air_share": (0.01, 1.0),
          "min_score": (0.0, 100.0), "max_duration": (0.1, 86400.0)}
PENALTIES = {"motion_not_judged": 0.0, "soft": 30.0, "under_exposed": 25.0, "over_exposed": 25.0, "shaky": 25.0, "dead_air": 10.0, "black": 40.0, "frozen": 40.0, "short": 20.0}
ORDER = "source"            # the rendered sequence: input order, then time (the assembly path joins each clip once; ADR-124)


class SelectsError(ValueError):
    """Inputs, options or a selects file that cannot be used: refused, never corrected."""


# ---- inputs and options ---------------------------------------------------------------------------------------------
def expand_inputs(paths: Sequence[str]) -> List[str]:
    """A folder (its video files, by name, not recursive) or files, in the order given; 1-50 clips, each once."""
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(os.path.join(p, n) for n in os.listdir(p) if n.lower().endswith(VIDEO_EXTENSIONS) and os.path.isfile(os.path.join(p, n)))
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise SelectsError(f"selects: not a file or folder: {p}")
    seen, uniq = set(), []
    for p in out:
        a = os.path.abspath(p)
        if a in seen:
            raise SelectsError(f"selects: {os.path.basename(p)} is given twice")
        seen.add(a)
        uniq.append(a)
    if not uniq:
        raise SelectsError("selects: no video clips found")
    if len(uniq) > MAX_CLIPS:
        raise SelectsError(f"selects: {len(uniq)} clips; at most {MAX_CLIPS} per batch")
    return uniq


def options(given: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The rule's defaults with the user's values; unknown names and out-of-range values are refused."""
    o = dict(DEFAULTS)
    for k, v in (given or {}).items():
        if v is None:
            continue
        if k not in DEFAULTS:
            raise SelectsError(f"selects: unknown option {k!r}")
        if k == "reject_dead_air":
            if not isinstance(v, bool):
                raise SelectsError("selects: reject_dead_air must be true or false")
            o[k] = v
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise SelectsError(f"selects: {k} must be a number")
        lo, hi = LIMITS[k]
        if not lo <= float(v) <= hi:
            raise SelectsError(f"selects: {k} must be within {lo:g}..{hi:g}")
        o[k] = float(v)
    if o["under_luma"] >= o["over_luma"]:
        raise SelectsError("selects: under_luma must be below over_luma")
    return o


# ---- facts per shot ---------------------------------------------------------------------------------------------------
def covered(a: float, b: float, spans: Sequence[Dict[str, Any]], end_default: float) -> float:
    """Seconds of [a, b) covered by the spans (an open end is the clip's end); overlapping spans are counted once."""
    iv = sorted((max(a, float(s["start"])), min(b, float(s["end"]) if s.get("end") is not None else end_default)) for s in spans if s.get("start") is not None)
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def _median(v: List[float]) -> Optional[float]:
    v = sorted(v)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def _windows_in(windows: Sequence[Dict[str, Any]], a: float, b: float) -> List[Dict[str, Any]]:
    """The picture windows of a shot: those whose middle lies in it, else the one with the largest overlap."""
    inside = [w for w in windows if a <= (float(w["start"]) + float(w["end"])) / 2 < b]
    if inside:
        return inside
    best = max(windows, key=lambda w: min(b, float(w["end"])) - max(a, float(w["start"])), default=None)
    return [best] if best is not None and min(b, float(best["end"])) > max(a, float(best["start"])) else []


def shot_facts(clip: Dict[str, Any], a: float, b: float) -> Dict[str, Any]:
    """The measured facts inside one shot [a, b) of a clip; None where nothing was measured."""
    dur = float(clip["duration"])
    ws = _windows_in(clip.get("picture") or [], a, b)
    def med(key: str, nd: int = 3) -> Optional[float]:
        m = _median([float(w[key]) for w in ws if w.get(key) is not None])
        return None if m is None else round(m, nd)
    span = max(b - a, 1e-9)
    speech = clip.get("speech")
    silence = clip.get("silence")
    return {"blur": med("blur"), "luma_mean": med("luma_mean", 2), "luma_p10": med("luma_p10", 1), "luma_p90": med("luma_p90", 1),
            "jitter": med("jitter"), "speed": med("speed", 2), "motion_agreement": med("motion_agreement", 2), "picture_windows": len(ws),
            "black_share": round(covered(a, b, clip.get("black") or [], dur) / span, 3),
            "freeze_share": round(covered(a, b, clip.get("freeze") or [], dur) / span, 3),
            "silence_share": None if silence is None else round(covered(a, b, silence, dur) / span, 3),
            "speech_share": None if speech is None else round(covered(a, b, speech, dur) / span, 3),
            "has_audio": bool(clip.get("has_audio", True))}


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def evaluate(clips: List[Dict[str, Any]], opts: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Every shot of every clip with its facts, flags, reject reasons and score (rule selects@1). Returns (shots, batch facts)."""
    shots: List[Dict[str, Any]] = []
    for ci, clip in enumerate(clips):
        for si, sc in enumerate(clip["shots"]):
            a, b = float(sc["start"]), float(sc["end"])
            shots.append({"clip": ci, "clip_name": os.path.basename(clip["path"]), "id": f"c{ci + 1:02d}s{si + 1:02d}", "index": si,
                          "start": round(a, 6), "end": round(b, 6), "duration": round(b - a, 3), "facts": shot_facts(clip, a, b)})
    blurs = [s["facts"]["blur"] for s in shots if s["facts"]["blur"] is not None and s["facts"]["black_share"] < opts["black_share"]]
    ref = percentile(blurs, opts["blur_percentile"])
    soft_at = None
    if opts.get("max_blur") is not None:
        soft_at = float(opts["max_blur"])
    elif ref is not None and len(blurs) >= 2:
        soft_at = round(ref * opts["blur_ratio"], 3)
    batch = {"blur_reference": None if ref is None else round(ref, 3), "soft_at": soft_at, "shots_with_blur": len(blurs),
             "soft_rule": "max_blur" if opts.get("max_blur") is not None else f"{opts['blur_ratio']:g} x the {opts['blur_percentile']:g}th percentile of the shots' median blur"
             if soft_at is not None else "not applied: fewer than 2 shots with a blur value and no max_blur"}
    for s in shots:
        f = s["facts"]
        flags: List[Dict[str, Any]] = []
        def flag(code: str, reject: bool, text: str, **values: Any) -> None:
            flags.append({"code": code, "reject": reject, "reason": text, "values": values})
        if f["black_share"] > 0:
            flag("black", f["black_share"] >= opts["black_share"], f"black frames over {f['black_share']:.0%} of the shot", black_share=f["black_share"])
        if f["freeze_share"] > 0:
            flag("frozen", f["freeze_share"] >= opts["freeze_share"], f"frozen frames over {f['freeze_share']:.0%} of the shot", freeze_share=f["freeze_share"])
        if f["blur"] is not None and soft_at is not None and f["blur"] >= soft_at:
            flag("soft", True, f"soft focus / blurred (blur {f['blur']:g} >= {soft_at:g})", blur=f["blur"], soft_at=soft_at)
        if f["luma_mean"] is not None and f["luma_mean"] < opts["under_luma"] and not (f["black_share"] >= opts["black_share"]):
            flag("under_exposed", True, f"under-exposed (mean luma {f['luma_mean']:g} < {opts['under_luma']:g})", luma_mean=f["luma_mean"])
        if f["luma_mean"] is not None and f["luma_mean"] > opts["over_luma"]:
            flag("over_exposed", True, f"over-exposed (mean luma {f['luma_mean']:g} > {opts['over_luma']:g})", luma_mean=f["luma_mean"])
        judged = f["jitter"] is not None and (f.get("motion_agreement") is None or f["motion_agreement"] >= opts["min_motion_agreement"])
        if f["jitter"] is not None and f["jitter"] > opts["max_jitter"]:
            if judged:
                flag("shaky", True, f"shaky (camera jitter {f['jitter']:g}% of the width > {opts['max_jitter']:g}%)", jitter=f["jitter"],
                     motion_agreement=f.get("motion_agreement"))
            else:
                flag("motion_not_judged", False, f"camera motion not judged (jitter {f['jitter']:g}%, but only {f['motion_agreement']:g} of the picture "
                     f"blocks agreed: a subject may fill the frame)", jitter=f["jitter"], motion_agreement=f.get("motion_agreement"))
        dead = []
        if not f["has_audio"]:
            dead.append("no audio track")
        elif f["silence_share"] is not None and f["silence_share"] >= opts["dead_air_share"]:
            dead.append(f"silent over {f['silence_share']:.0%}")
        if f["speech_share"] is not None and f["speech_share"] == 0 and f["has_audio"]:
            dead.append("no speech")
        if dead:
            flag("dead_air", bool(opts["reject_dead_air"]), "dead air (" + ", ".join(dead) + ")", silence_share=f["silence_share"], speech_share=f["speech_share"])
        if s["duration"] < opts["min_shot"]:
            flag("short", True, f"shorter than {opts['min_shot']:g}s ({s['duration']:g}s)", duration=s["duration"])
        score = 100.0
        for fl in flags:
            share = fl["values"].get("black_share") or fl["values"].get("freeze_share")
            score -= PENALTIES[fl["code"]] * (share if fl["code"] in ("black", "frozen") and share is not None else 1.0)
        s["flags"] = flags
        s["score"] = round(max(0.0, score), 1)
        s["steady"] = None if not judged else f["jitter"] <= opts["max_jitter"]
        s["rejected_by"] = [fl["reason"] for fl in flags if fl["reject"]]
    return shots, batch


def select(shots: List[Dict[str, Any]], opts: Dict[str, Any]) -> Dict[str, Any]:
    """Keep or reject each shot; the kept ones in source order. `min_score` rejects low scores; `max_duration` keeps the best-scored usable shots
    that fit (ties: source order), then puts them back in source order."""
    usable = []
    for s in shots:
        reasons = list(s["rejected_by"])
        if not reasons and opts.get("min_score") is not None and s["score"] < float(opts["min_score"]):
            reasons.append(f"score {s['score']:g} below min_score {float(opts['min_score']):g}")
        s["kept"], s["reasons"] = not reasons, reasons
        if not reasons:
            usable.append(s)
    cap = opts.get("max_duration")
    if cap is not None:
        total = 0.0
        for s in sorted(usable, key=lambda x: (-x["score"], x["clip"], x["start"])):
            if total + s["duration"] <= float(cap) + 1e-9:
                total += s["duration"]
            else:
                s["kept"] = False
                s["reasons"] = [f"left out by the duration cap ({float(cap):g}s; score {s['score']:g})"]
    kept = [s for s in shots if s["kept"]]
    for s in shots:
        if s["kept"]:
            s["reasons"] = ["usable" + ("" if not s["flags"] else ": " + "; ".join(fl["reason"] for fl in s["flags"]) + " (flag only)")]
    return {"kept": [s["id"] for s in kept], "kept_count": len(kept), "rejected_count": len(shots) - len(kept),
            "duration": round(sum(s["duration"] for s in kept), 3), "order": ORDER}


# ---- the file the preview writes and the plan reads --------------------------------------------------------------------
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_document(clips: List[Dict[str, Any]], opts: Dict[str, Any], shots: List[Dict[str, Any]], batch: Dict[str, Any], sel: Dict[str, Any],
                   skills: Dict[str, Any]) -> Dict[str, Any]:
    by_clip: Dict[int, List[Dict[str, Any]]] = {}
    for s in shots:
        by_clip.setdefault(s["clip"], []).append(s)
    out_clips = []
    for ci, c in enumerate(clips):
        cs = by_clip.get(ci, [])
        out_clips.append({"index": ci, "path": os.path.abspath(c["path"]), "sha256": c["sha256"], "duration": c["duration"], "rate": c.get("rate"),
                          "has_audio": bool(c.get("has_audio", True)), "observations": dict(c.get("observations") or {}),
                          "unmeasured": list(c.get("unmeasured") or []),
                          "keep": [[s["start"], s["end"]] for s in cs if s["kept"]],
                          "shots": [{k: s[k] for k in ("id", "index", "start", "end", "duration", "facts", "flags", "score", "steady", "kept", "reasons")} for s in cs]})
    return {"schema": DOC_SCHEMA, "applied": False, "rule": RULE, "options": dict(opts), "batch": batch, "skills": skills, "order": ORDER,
            "clips": out_clips,
            "summary": {"clips": len(clips), "clips_used": sum(1 for c in out_clips if c["keep"]), "shots": len(shots), **sel,
                        "duration_before": round(sum(float(c["duration"]) for c in clips), 3)}}


def approval_lines(doc: Dict[str, Any]) -> List[str]:
    """What the user sees before anything renders: every shot (kept or why not), per clip, then the total."""
    lines = []
    for c in doc.get("clips") or []:
        lines.append(f"{os.path.basename(c['path'])}  ({float(c['duration']):.2f}s, {len(c['shots'])} shot(s))"
                     + (f"  [not measured: {', '.join(c['unmeasured'])}]" if c.get("unmeasured") else ""))
        for s in c["shots"]:
            mark = "KEEP  " if s["kept"] else "REJECT"
            f = s["facts"]
            facts = ", ".join(x for x in (f"blur {f['blur']:g}" if f.get("blur") is not None else "",
                                          f"luma {f['luma_mean']:g}" if f.get("luma_mean") is not None else "",
                                          f"jitter {f['jitter']:g}%" if f.get("jitter") is not None else "") if x)
            lines.append(f"  {mark} {s['id']}  {s['start']:8.3f}-{s['end']:8.3f}s  {s['duration']:6.2f}s  score {s['score']:5.1f}  "
                         f"{'steady' if s.get('steady') else 'shaky' if s.get('steady') is False else '-'}  ({facts})  {'; '.join(s['reasons'])}")
    sm = doc.get("summary") or {}
    lines.append(f"total: {sm.get('kept_count', 0)} of {sm.get('shots', 0)} shot(s) kept from {sm.get('clips_used', 0)} of {sm.get('clips', 0)} clip(s), "
                 f"{float(sm.get('duration') or 0):.2f}s (from {float(sm.get('duration_before') or 0):.2f}s), in source order")
    return lines


def check_document(doc: Any) -> List[str]:
    errs: List[str] = []
    if not isinstance(doc, dict) or doc.get("schema") != DOC_SCHEMA:
        return [f"schema is not {DOC_SCHEMA}"]
    if doc.get("applied") is not False:
        errs.append("applied must be false")
    if doc.get("rule") != RULE:
        errs.append(f"rule must be {RULE}")
    clips = doc.get("clips")
    if not isinstance(clips, list) or not clips or len(clips) > MAX_CLIPS:
        return errs + [f"clips must be a list of 1-{MAX_CLIPS}"]
    for c in clips:
        if not isinstance(c, dict) or not isinstance(c.get("path"), str) or not re.match(r"^[0-9a-f]{64}$", str(c.get("sha256") or "")):
            errs.append("each clip must name a path and its sha256")
            break
        dur = c.get("duration")
        last = -1.0
        for k in c.get("keep") or []:
            if not (isinstance(k, list) and len(k) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in k)
                    and last <= k[0] < k[1] and (not isinstance(dur, (int, float)) or k[1] <= float(dur) + 0.001)):
                errs.append(f"{os.path.basename(c['path'])}: kept ranges must be forward, in order and inside the clip")
                break
            last = k[1]
    if not any(c.get("keep") for c in clips if isinstance(c, dict)):
        errs.append("no shot is kept")
    return errs


def parse_requirement(path: Any) -> Dict[str, Any]:
    """`edit.selects`: the selects file the preview wrote. Refused when it is not one, says it was applied, or a clip changed since (sha256)."""
    if not isinstance(path, str) or not path.strip() or "\n" in path or "\x00" in path or any(x == ".." for x in path.replace("\\", "/").split("/")):
        raise SelectsError(f"{KEY} must be the path of the file `video-agent selects` wrote")
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        doc = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as e:
        raise SelectsError(f"{KEY}: cannot read the selects file: {e}")
    errs = check_document(doc)
    if errs:
        raise SelectsError(f"{KEY} is not a usable selects file: " + "; ".join(errs[:3]))
    for c in doc["clips"]:
        if not os.path.isfile(c["path"]) or sha256_of(c["path"]) != c["sha256"]:
            raise SelectsError(f"{KEY}: the clip {os.path.basename(c['path'])} is missing or changed since the preview; run `video-agent selects` again")
    clips = [{"index": c["index"], "path": c["path"], "sha256": c["sha256"], "duration": c["duration"], "rate": c.get("rate"), "keep": [list(k) for k in c.get("keep") or []],
              "kept": [s["id"] for s in c["shots"] if s.get("kept")], "rejected": {s["id"]: s.get("reasons") for s in c["shots"] if not s.get("kept")}}
             for c in doc["clips"]]
    return {"document": os.path.abspath(path), "sha256": hashlib.sha256(raw).hexdigest(), "rule": RULE, "options": dict(doc.get("options") or {}),
            "clips": clips, "summary": dict(doc.get("summary") or {})}


def merge_ranges(keep: Sequence[Sequence[float]], gap: float = 0.0005) -> List[List[float]]:
    """Kept shots that touch (the next starts where the previous ends) become one range: nothing lies between them to cut."""
    out: List[List[float]] = []
    for a, b in sorted([float(x), float(y)] for x, y in keep):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], round(b, 6))
        else:
            out.append([round(a, 6), round(b, 6)])
    return out


# ---- the decisions -----------------------------------------------------------------------------------------------------
def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)


CONFLICTS = ("video.concat", "video.switch", "video.grid")


def decide_selects(eng: Any, edit: Dict[str, Any], analysis: Any, edits: Dict[str, Any], approval_for: Any, probe_ids_of: Any, cap_block: Any,
                   programme: str, trim_skill: str) -> Dict[str, Any]:
    """`edit.selects` → one `selects.keep` TRANSFORM per clip with kept shots (its kept ranges, CONFIRM at least) and, with two or more such clips,
    a `video.concat` of them in source order that cites them. Clips of the plan with no kept shot are left out. A clip of the file that is not an
    input of the plan, a kept range past a clip's end, or another programme-building edit is a BLOCK naming why. Returns {used, left_out, ok}."""
    req = edit["requirements"]
    p = edit["params"]
    ev = [r.id for r in req] + probe_ids_of([a.id for a in analysis.assets])
    res: Dict[str, Any] = {"used": [], "left_out": set(), "ok": False}
    videos = [a for a in analysis.assets if (a.technical or {}).get("video")]
    matched: List[Tuple[Any, Dict[str, Any]]] = []
    why = None
    other = [k for k in CONFLICTS if k in edits]
    if other:
        why = f"{KEY} builds the programme from the kept shots; {', '.join(other)} builds it another way, and which one was meant is not guessed"
    for c in p["clips"]:
        if why is not None:
            break
        a = next((x for x in videos if (x.hash and x.hash == c["sha256"]) or _same_file(x.path, c["path"])), None)
        if a is None:
            if c["keep"]:
                why = f"{os.path.basename(c['path'])} has kept shots in the selects file but is not a video input of this plan (matched by sha256 / path)"
            continue
        dur = float((a.technical or {}).get("duration") or 0.0)
        late = [k for k in c["keep"] if dur and float(k[1]) > dur + 0.001]
        if late:
            why = f"{a.id}: a kept range ends at {float(late[0][1]):.3f}s, after the clip ({dur:.3f}s); a range is never clamped"
        matched.append((a, c))
    if why is not None:
        eng.decide(subject=SUBJECT, type="BLOCK", decision=f"BLOCK: {KEY}", reason=why + " (not guessed)", confidence=1.0, evidence=ev, risk=RISK, approval="BLOCK",
                   provenance="USER", params={"document": p["document"], "rule": RULE}, requirements=req)
        return res
    matched.sort(key=lambda ac: ac[1]["index"])
    ids: List[str] = []
    for a, c in matched:
        if not c["keep"]:
            res["left_out"].add(a.id)
            continue
        kept_s = sum(float(k[1]) - float(k[0]) for k in c["keep"])
        rejected = "; ".join(f"{sid}: {', '.join(r or [])}" for sid, r in list(c["rejected"].items())[:8])
        d = eng.decide(subject=SUBJECT, type="TRANSFORM",
                       decision=f"keep {len(c['keep'])} shot(s) of {a.id} ({kept_s:.3f}s of {float(c['duration']):.3f}s): " + ", ".join(c["kept"]),
                       reason=f"the user previewed the selects ({RULE}: measured blur / exposure / jitter / black / frozen / silence / speech per shot) and asked to "
                              f"assemble them ({req[0].source}); left out: {rejected or 'nothing'}. Leaving shots out needs the user's yes",
                       confidence=1.0, evidence=ev, risk=RISK, approval=approval_for(SUBJECT, floor="CONFIRM", keys={SUBJECT: APPROVAL_KEY}), provenance="USER",
                       params={"asset_id": a.id, "keep": merge_ranges(c["keep"]), "shots": [list(k) for k in c["keep"]], "kept": c["kept"],
                               "rejected": c["rejected"], "document": p["document"], "sha256": p["sha256"], "rule": RULE, "clip_index": c["index"]},
                       requirements=req, serves_intent=None)
        ids.append(d.id)
        res["used"].append(a.id)
    unlisted = [a.id for a in analysis.assets if a.id not in {x.id for x, _ in matched}]
    res["left_out"] |= set(unlisted)
    if not res["used"]:
        eng.decide(subject=SUBJECT, type="BLOCK", decision=f"BLOCK: {KEY} keeps no shot of this plan's inputs", reason="nothing would be left to deliver (not guessed)",
                   confidence=1.0, evidence=ev, risk=RISK, approval="BLOCK", provenance="USER", params={"document": p["document"], "rule": RULE}, requirements=req)
        return res
    cap_block(trim_skill, f"capability.{trim_skill}")
    if len(res["used"]) >= 2:
        eng.decide(subject="video.concat", type="TRANSFORM", decision="concat " + " + ".join(res["used"]) + f" → {programme} (the kept shots of each clip, in source order)",
                   reason=f"user asked to assemble the selects ({req[0].source}); each clip is cut to its kept shots, the clips are joined in input order",
                   confidence=1.0, evidence=ev, risk="MEDIUM", approval=approval_for("video.concat", explicit=req[0]), provenance="USER",
                   params={"asset_id": programme, "inputs": list(res["used"]), "picks": ids}, requirements=req, serves_intent=None)
        cap_block("video_concat", "capability.video_concat")
    res["ok"] = True
    return res


# ---- the measurements (the Skills measure; this only asks and reads) ----------------------------------------------------
MA_KINDS = {"media_probe": "media-analysis/probe", "scene_detection": "media-analysis/scenes", "picture_quality": "media-analysis/picture",
            "silence": "media-analysis/silence", "speech_regions": "media-analysis/vad"}
QC_INSPECT = "qc/inspect"


def _rate_of(probe: Dict[str, Any]) -> Optional[str]:
    v = probe.get("video") or {}
    r = v.get("frame_rate") or v.get("r_frame_rate") or v.get("avg_frame_rate")
    if isinstance(r, str) and re.match(r"^\d+(/\d+)?$", r):
        return r
    f = v.get("fps")
    return f"{f:g}" if isinstance(f, (int, float)) and f > 0 else None


def measure_clip(ma: Any, qc: Any, path: str, index: int, timeout: Optional[float] = None) -> Dict[str, Any]:
    """One clip's facts from media-analysis-skill (probe, scene_detection, picture_quality, silence, speech_regions) and qc-skill inspect (black and
    frozen frames). A measurement that cannot be taken is listed in `unmeasured` with the Skill's reason; it is never guessed. A clip whose shot
    boundaries or picture cannot be measured is refused (SelectsError)."""
    aid = f"clip{index + 1:02d}"
    obs: Dict[str, str] = {}
    unmeasured: List[str] = []
    data: Dict[str, Any] = {}
    for kind, tool in MA_KINDS.items():
        if tool not in getattr(ma, "tools", {}):
            if kind in ("scene_detection", "picture_quality", "media_probe"):
                raise SelectsError(f"selects: the installed media-analysis-skill has no {tool} ({kind}); picture_quality needs media-analysis-skill 0.2.0 or later")
            unmeasured.append(f"{kind} (no {tool} in the installed media-analysis-skill)")
            continue
        r = ma.measure(tool, {"input": path, "asset_id": aid, "kind": kind}, timeout=timeout)
        o = (r.data or {}).get("observation") if r.ok else None
        if o is None:
            err = (r.data or {}).get("error") or {}
            if kind in ("scene_detection", "picture_quality", "media_probe"):
                raise SelectsError(f"selects: {os.path.basename(path)}: {kind} failed: {err.get('code')}: {str(err.get('message') or '')[:200]}")
            data[kind] = None
            unmeasured.append(f"{kind} ({err.get('code') or 'failed'})")
            continue
        obs[kind] = o.get("id")
        data[kind] = o.get("data") or {}
    probe = data["media_probe"]
    has_audio = bool(probe.get("audio"))
    dur = float(data["scene_detection"].get("duration") or data["picture_quality"].get("duration") or (probe.get("container") or {}).get("duration") or 0.0)
    if dur <= 0:
        raise SelectsError(f"selects: {os.path.basename(path)} has no measured duration")
    scenes = [{"start": float(s["start"]), "end": float(s["end"] if s.get("end") is not None else dur)} for s in data["scene_detection"].get("scenes") or []]
    if not scenes:
        scenes = [{"start": 0.0, "end": dur}]
    silence = None
    if not has_audio:
        silence = [{"start": 0.0, "end": dur}]
    elif data.get("silence") is not None:
        silence = [{"start": s["start"], "end": s.get("end")} for s in data["silence"].get("segments") or []]
    speech = None
    sr = data.get("speech_regions")
    if has_audio and sr is not None:
        if sr.get("status") == "MEASURED":
            speech = [{"start": r["start"], "end": r["end"]} for r in sr.get("regions") or []]
        else:
            unmeasured.append(f"speech_regions ({sr.get('reason') or sr.get('status')}; dead air is judged from silence alone)")
    black: List[Dict[str, Any]] = []
    freeze: List[Dict[str, Any]] = []
    if qc is None:
        unmeasured.append("black / frozen frames (qc-skill is not installed)")
    else:
        q = qc.measure(QC_INSPECT, {"input": path, "kind": "video"}, timeout=timeout)
        if not q.ok:
            unmeasured.append(f"black / frozen frames (qc-skill inspect failed: {str((q.data or {}).get('error') or q.stderr)[:160]})")
        else:
            ms = {m.get("id"): m for m in (q.data or {}).get("measurements") or []}
            black = [s for s in (ms.get("video.black_segments") or {}).get("value") or [] if isinstance(s, dict)]
            freeze = [s for s in (ms.get("video.freeze_segments") or {}).get("value") or [] if isinstance(s, dict)]
            obs["qc_inspect"] = (q.data or {}).get("report_id") or "qc-inspect"
    return {"path": os.path.abspath(path), "sha256": sha256_of(path), "duration": round(dur, 6), "rate": _rate_of(probe), "has_audio": has_audio,
            "shots": scenes, "picture": data["picture_quality"].get("windows") or [], "black": black, "freeze": freeze, "silence": silence, "speech": speech,
            "observations": obs, "unmeasured": unmeasured}


def preview(clips: List[Dict[str, Any]], opts: Dict[str, Any], skills: Dict[str, Any]) -> Dict[str, Any]:
    """Rule selects@1 over measured clips → the selects document."""
    shots, batch = evaluate(clips, opts)
    sel = select(shots, opts)
    return build_document(clips, opts, shots, batch, sel, skills)
