"""Opt-in qc-skill checks of what the agent itself joined (ADR-112): a stinger's cut point after a stinger CONCAT, and the seams of a music
bed the agent looped. Both are measurements after the render, in QA; neither decides or edits anything.

    --set qc.stinger=true [qc.stinger.max_cut_frame_difference=N]      (needs edit.concat.stinger)        → IR qa.stinger_check
    --set qc.loop_seam.max_click_ratio=R | .max_level_diff_db=D | .max_band_diff_db=B [.window_sec=W]   (needs bgm.library) → IR qa.loop_seam_check
    render → QA (qa/checks.py) → qc-skill `check` → QA items (layer qc) on the deliverables → report

- **Stinger** (qc-skill 1.2.0, its ADR-024): once per deliverable of the stinger programme, `kind: video` on the delivered file with
  `rules.video.stinger` = the concat's own `transition` (`cut_point`, `cover`: the stinger maker's statement, as the plan recorded it), `file`
  = the stinger (the plan's path, inside the allowed input roots) and `cut_at` = video-editing's `stinger.joins[].cut_at` from the concat's
  result, mapped onto the delivered timeline (a still joined before shifts it, a programme speed change scales it). The cover and opacity
  checks are fixed rules. `max_cut_frame_difference` is policy: qc-skill gives it no default (the difference a correct cut leaves is the
  delivery's own encoding loss, and this agent may grade, caption or overlay the picture after the join), so it is sent only when the user
  sets it, and without it the cut-frame check does not run (the measured differences are still reported).
- **Loop seam** (qc-skill 1.3.0, its ADR-025): the bed library-pick fitted by looping (`fit.mode` loop, `repeats` copies of a `source`
  seconds file joined with a `crossfade`): `kind: audio` on the bed with `rules.audio.loop_seam` = `seams` at every multiple of the loop
  period (`source - crossfade`) inside the bed, `include_wrap: false` (the bed is played once under the programme, so its end never runs
  into its start) and the limits the user set. No limit has a default (qc-skill: none is a standard), so the check runs only when one is
  set; `window_sec` is needed by the level / spectrum limits and only by them. A bed that was trimmed or used whole has no seam: nothing runs.
"""
from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Any, Dict, List, Optional

from .bgm import BGM_INPUT
from .editing import EditRequirementError, _bool, _number

STINGER_KEY = "qc.stinger"
STINGER_MAX_KEY = "qc.stinger.max_cut_frame_difference"
LOOP_PREFIX = "qc.loop_seam."
# the loop-seam limits: key → (qc rule field, lo, hi); each runs its own check (qc-skill ADR-025: exactly the limit passes)
LOOP_LIMITS = {"qc.loop_seam.max_click_ratio": ("max_click_ratio", 0.0, 1000.0),
               "qc.loop_seam.max_level_diff_db": ("max_level_diff_db", 0.0, 120.0),
               "qc.loop_seam.max_band_diff_db": ("max_band_diff_db", 0.0, 120.0)}
LOOP_WINDOW_KEY = "qc.loop_seam.window_sec"
WINDOW_RANGE = (0.01, 60.0)   # qc-skill: window_sec > 0 and at most 60
NEEDS_WINDOW = ("qc.loop_seam.max_level_diff_db", "qc.loop_seam.max_band_diff_db")
REQUIREMENT_KEYS = (STINGER_KEY, STINGER_MAX_KEY) + tuple(LOOP_LIMITS) + (LOOP_WINDOW_KEY,)
STINGER_RULE = "qc_stinger@1"
LOOP_RULE = "qc_loop_seam@1"
STINGER_CHECKS = ("video.stinger_cut_point_in_cover", "video.stinger_opaque_at_cut_point", "video.stinger_cut_frame_matches")
LOOP_CHECKS = ("audio.loop_seam_click_within_limit", "audio.loop_seam_level_within_limit", "audio.loop_seam_spectrum_within_limit")
STINGER_FINDINGS = ("VIDEO_STINGER_CUT_POINT_OUTSIDE_COVER", "VIDEO_STINGER_NOT_OPAQUE", "VIDEO_STINGER_CUT_FRAME_MISMATCH")
LOOP_FINDINGS = ("AUDIO_LOOP_SEAM_CLICK", "AUDIO_LOOP_SEAM_LEVEL_MISMATCH", "AUDIO_LOOP_SEAM_SPECTRUM_MISMATCH")
MAX_SEAMS = 4096   # the qc adapter's list bound (MAX_RULE_ITEMS)


def check_qc_check_values(values: Dict[str, Any]) -> None:
    """Each key's own value (the --set gate, on any set of keys: a plan's, a revision's new ones, a brief's): a switch is true / false, every
    number finite and in range. How the keys go together is parse_qc_check_requirements' (on the effective keys)."""
    for k, v in (values or {}).items():
        if v is None or k not in REQUIREMENT_KEYS:
            continue
        if k == STINGER_KEY:
            _bool(v, k)
        elif k == STINGER_MAX_KEY:
            _number(v, k, 0.0, 255.0)
        elif k in LOOP_LIMITS:
            _number(v, k, LOOP_LIMITS[k][1], LOOP_LIMITS[k][2])
        else:
            _number(v, k, *WINDOW_RANGE)


def parse_qc_check_requirements(values: Dict[str, Any]) -> Dict[str, Optional[Dict[str, Any]]]:
    """{"stinger": IR qa.stinger_check | None, "loop_seam": IR qa.loop_seam_check | None} from the effective requirements (a plan's, or a
    revision's merged with the earlier ones); EditRequirementError on a value out of range, a refinement without its switch, a limit that needs
    a window without one (or a window without such a limit), or a check of something the plan does not join (a stinger CONCAT, a library bed)."""
    check_qc_check_values(values)
    given = {k: v for k, v in (values or {}).items() if k in REQUIREMENT_KEYS and v is not None}
    out: Dict[str, Optional[Dict[str, Any]]] = {"stinger": None, "loop_seam": None}
    on = _bool(given[STINGER_KEY], STINGER_KEY) if STINGER_KEY in given else False
    if STINGER_MAX_KEY in given and not on:
        raise EditRequirementError(f"{STINGER_MAX_KEY} refines the stinger check: set {STINGER_KEY}=true with it")
    if on:
        if not (values or {}).get("edit.concat.stinger"):
            raise EditRequirementError(f"{STINGER_KEY}=true checks a stinger join: it needs edit.concat.stinger (a CONCAT with a stinger transition)")
        rec: Dict[str, Any] = {"rule": STINGER_RULE}
        if STINGER_MAX_KEY in given:
            rec["max_cut_frame_difference"] = float(_number(given[STINGER_MAX_KEY], STINGER_MAX_KEY, 0.0, 255.0))
        out["stinger"] = rec
    limits = {k: v for k, v in given.items() if k in LOOP_LIMITS}
    if LOOP_WINDOW_KEY in given and not any(k in limits for k in NEEDS_WINDOW):
        raise EditRequirementError(f"{LOOP_WINDOW_KEY} is the level / spectrum window: set {' or '.join(NEEDS_WINDOW)} with it (a window alone checks nothing)")
    if limits:
        missing = [k for k in NEEDS_WINDOW if k in limits and LOOP_WINDOW_KEY not in given]
        if missing:
            raise EditRequirementError(f"{', '.join(missing)} compare(s) {LOOP_WINDOW_KEY} of audio on each side of a seam: set {LOOP_WINDOW_KEY} too "
                                       "(qc-skill has no default window)")
        if not (values or {}).get("bgm.library"):
            raise EditRequirementError("qc.loop_seam.* checks the seams of a music bed the agent loops: it needs bgm.library (library-pick loops a short "
                                       "loopable file; a generated bed is never looped)")
        rec = {"rule": LOOP_RULE}
        for key, (field, lo, hi) in LOOP_LIMITS.items():
            if key in limits:
                rec[field] = float(_number(limits[key], key, lo, hi))
        if LOOP_WINDOW_KEY in given:
            rec["window_sec"] = float(_number(given[LOOP_WINDOW_KEY], LOOP_WINDOW_KEY, *WINDOW_RANGE))
        out["loop_seam"] = rec
    return out


def check_problems(doc: Dict[str, Any]) -> List[str]:
    """Validator rules of the two IR records: their own vocabulary and ranges, and something in the plan for them to check."""
    errs: List[str] = []
    qa = doc.get("qa") or {}
    st = qa.get("stinger_check")
    if st is not None:
        if not isinstance(st, dict) or st.get("rule") != STINGER_RULE or set(st) - {"rule", "max_cut_frame_difference"}:
            errs.append(f"qa.stinger_check: must be {{rule: {STINGER_RULE}, max_cut_frame_difference?}}")
        elif "max_cut_frame_difference" in st and not _in(st["max_cut_frame_difference"], 0.0, 255.0):
            errs.append("qa.stinger_check: max_cut_frame_difference must be a number in 0..255")
        if not any(op.get("type") == "video.concat" and op.get("stinger") and (op.get("transition") or {}).get("type") == "stinger"
                   for op in (doc.get("video") or {}).get("operations") or []):
            errs.append("qa.stinger_check: the plan has no CONCAT with a stinger transition to check")
    ls = qa.get("loop_seam_check")
    if ls is not None:
        fields = {f: (lo, hi) for f, lo, hi in LOOP_LIMITS.values()}
        if not isinstance(ls, dict) or ls.get("rule") != LOOP_RULE or set(ls) - set(fields) - {"rule", "window_sec"} or not set(ls) & set(fields):
            errs.append(f"qa.loop_seam_check: must be {{rule: {LOOP_RULE}, at least one of {', '.join(fields)}, window_sec?}}")
        else:
            for f, (lo, hi) in fields.items():
                if f in ls and not _in(ls[f], lo, hi):
                    errs.append(f"qa.loop_seam_check: {f} must be a number in {lo:g}..{hi:g}")
            if "window_sec" in ls and not _in(ls["window_sec"], *WINDOW_RANGE):
                errs.append(f"qa.loop_seam_check: window_sec must be a number in {WINDOW_RANGE[0]:g}..{WINDOW_RANGE[1]:g}")
            if any(f in ls for f in ("max_level_diff_db", "max_band_diff_db")) != ("window_sec" in ls):
                errs.append("qa.loop_seam_check: window_sec goes with max_level_diff_db / max_band_diff_db, and only with them")
        if not any(op.get("type") == "audio.bgm" and op.get("mode") == "library" for op in (doc.get("audio") or {}).get("operations") or []):
            errs.append("qa.loop_seam_check: the plan has no library bed (audio.bgm mode library) to check")
    return errs


def _in(v: Any, lo: float, hi: float) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and lo <= float(v) <= hi


# ---- runtime: what the render reported → the qc request -----------------------------------------------------------------------------
def stinger_concat(doc: Dict[str, Any], subject_id: str) -> Optional[Dict[str, Any]]:
    """The CONCAT with a stinger transition whose output is `subject_id` (the programme), or None."""
    return next((op for op in (doc.get("video") or {}).get("operations") or []
                 if op.get("type") == "video.concat" and op.get("stinger") and (op.get("transition") or {}).get("type") == "stinger"
                 and (op.get("output") or "programme") == subject_id), None)


def seconds_of(v: Any) -> Optional[float]:
    """video-editing's time ({"seconds": "2.000000", "rational": "2/1"}) or a plain number as float seconds; None when unreadable."""
    if isinstance(v, dict):
        r = v.get("rational")
        if isinstance(r, str) and re.match(r"^\d+/[1-9]\d*$", r):
            return float(Fraction(r))
        v = v.get("seconds")
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        return None
    return float(v)


def joins_cut_at(data: Any) -> Optional[List[float]]:
    """The cut of every join as video-editing reported it for the CONCAT (`stinger.joins[].cut_at`, on the CONCAT's output timeline), or
    None when the result carries no joins (the Skill did not know every length, or the step was reused without a result)."""
    st = (((data or {}).get("reported") or {}).get("stinger") or {}) if isinstance(data, dict) else {}
    joins = st.get("joins") if isinstance(st, dict) else None
    if not isinstance(joins, list) or not joins:
        return None
    out = [seconds_of((j or {}).get("cut_at") if isinstance(j, dict) else None) for j in joins]
    return None if any(t is None for t in out) else [float(t) for t in out if t is not None]


def delivered_times(times: List[float], speed: float, lead: float) -> List[float]:
    """Programme times → the delivered file's timeline: a still joined before shifts them by its length (`lead`), a speed change divides them."""
    return [round(lead + t / (speed or 1.0), 6) for t in times]


def stinger_rules(concat: Dict[str, Any], cut_at: Optional[List[float]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """`rules` of the qc check: the stinger maker's statement as the plan recorded it, the stinger file, the delivered cuts, the user's limit."""
    tr = concat.get("transition") or {}
    rule: Dict[str, Any] = {"cut_point": float(tr["cut_point"]), "file": str(concat["stinger"])}
    if isinstance(tr.get("cover"), dict):
        rule["cover"] = {"start": float(tr["cover"]["start"]), "end": float(tr["cover"]["end"])}
    if cut_at:
        rule["cut_at"] = list(cut_at)
        if cfg.get("max_cut_frame_difference") is not None:
            rule["max_cut_frame_difference"] = float(cfg["max_cut_frame_difference"])
    return {"video": {"stinger": rule}}


def bed_mix_subject(doc: Dict[str, Any]) -> Optional[str]:
    """The subject whose mix takes the bed as an input (its deliverables carry the bed), or None."""
    mix = next((op for op in (doc.get("audio") or {}).get("operations") or [] if op.get("type") == "audio.mix" and BGM_INPUT in (op.get("inputs") or [])), None)
    return str(mix.get("asset")) if mix and mix.get("asset") else None


def loop_seams(fit: Any, duration: Optional[float] = None) -> Optional[List[float]]:
    """Every multiple of the loop period (`source - crossfade`) inside the bed when library-pick looped it, or None when it did not loop
    (trimmed or used whole) or the fit is unreadable. The bed is `target` (or `duration`) seconds long."""
    if not isinstance(fit, dict) or fit.get("mode") != "loop":
        return None
    source, xfade, repeats = seconds_of(fit.get("source")), seconds_of(fit.get("crossfade") or 0.0), fit.get("repeats")
    length = seconds_of(fit.get("target")) or (float(duration) if isinstance(duration, (int, float)) and duration > 0 else None)
    if source is None or xfade is None or length is None or isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        return None
    period = source - xfade
    if period <= 0:
        return None
    seams = [round(k * period, 6) for k in range(1, repeats) if k * period < length - 1e-6]
    return seams[:MAX_SEAMS] or None


def loop_rules(seams: List[float], cfg: Dict[str, Any]) -> Dict[str, Any]:
    rule: Dict[str, Any] = {"seams": list(seams), "include_wrap": False}
    for f in ("max_click_ratio", "max_level_diff_db", "max_band_diff_db", "window_sec"):
        if cfg.get(f) is not None:
            rule[f] = float(cfg[f])
    return {"audio": {"loop_seam": rule}}
