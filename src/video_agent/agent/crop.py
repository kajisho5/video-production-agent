"""Black-bar crop proposal (ADR-060, ECOSYSTEM_ROADMAP step 7): the `crop_rectangle` measurement → a CONFIRM decision that
proposes `video.crop` with the measured rectangle. Never applied without the user's approval.

- **Measurement** (analysis kind `crop_rectangle`, requested explicitly: `plan --kind crop_rectangle`): the
  registry-selected cropdetect tool (`cropdetect.py`), which samples `--samples` windows spread across the file
  (`--seconds` of footage in all), reads the cropdetect filter per frame, and reports the most common rectangle, the share of readings that agree with it
  (`confidence`) and how many windows it could not decode. The agent asks for `--round 2` (media/analysis.py CROP_TOOL_ARGS): the active picture to the
  nearest even pixel (the crop needs even sizes) instead of cropdetect's default multiple of 16, which cuts up to 14 px
  into the picture. The agent never reads frames itself.
- **Stable rectangle** (all of): a rectangle was reported; at least `MIN_WINDOWS` windows decoded; `confidence` ≥
  `MIN_CONFIDENCE` (one window in five disagreeing is already below it); after rounding the rectangle inward to even
  offsets and sizes it lies inside the frame and is at least 16 px each way; and it removes at least `MIN_BAR_FRACTION`
  of the width or the height (a few edge pixels are noise, not bars).
- **Decision**: a stable rectangle smaller than the frame → TRANSFORM `video.crop` on that asset, approval CONFIRM
  (floor: a profile cannot make it AUTO; the measurement is the agent's, not the user's request), provenance OBSERVED,
  evidence the measurement and the probe. Approved, the planner's existing `video.crop` step runs the registry-selected crop
  tool (video-editing-skill's CROP since ADR-079) with the rectangle; rejected, `revise` drops it. Anything else → KEEP (full frame) with the reason, no operation.
- **Not proposed at all**: an explicit `edit.crop` (the user's rectangle wins), a programme built from several inputs
  (concat / switch / grid / takes: one input's bars do not describe the programme), the audio production path (no
  picture is delivered)."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

CROP_KIND = "crop_rectangle"
CROP_SUBJECT = "video.crop"
CROP_SKILL = "video_crop"
MIN_CONFIDENCE = 0.9      # share of per-frame readings across every sampled window that agree with the reported rectangle
MIN_WINDOWS = 3           # decoded windows the rectangle must rest on
MIN_BAR_FRACTION = 0.01   # the bars must remove at least this share of the width or of the height
MIN_SIDE = 16             # the validator: width and height at least 16 px


def crop_facts(data: Dict[str, Any]) -> Dict[str, Any]:
    """One vocabulary for a crop_rectangle observation as cropdetect.py reports it (the tool's keys, kept as measured)."""
    d = data or {}
    c = d.get("crop") if isinstance(d.get("crop"), dict) else None
    rect = None
    if c is not None and all(isinstance(c.get(k), (int, float)) and not isinstance(c.get(k), bool) for k in ("x", "y", "width", "height")):
        rect = {k: int(c[k]) for k in ("x", "y", "width", "height")}
    windows = d.get("windows")
    failed = d.get("windows_failed")
    return {"rect": rect, "source_width": _int(d.get("source_width")), "source_height": _int(d.get("source_height")),
            "confidence": _num(d.get("confidence")), "windows": _int(windows), "windows_failed": _int(failed) or 0}


def even_inward(rect: Dict[str, int]) -> Dict[str, int]:
    """Round a rectangle to even offsets and sizes without growing it: the left / top edges move right / down to the next even
    pixel, the right / bottom edges move left / up to the previous one. The result never includes a pixel the measurement
    called a bar."""
    x0, y0 = rect["x"] + rect["x"] % 2, rect["y"] + rect["y"] % 2
    x1, y1 = rect["x"] + rect["width"], rect["y"] + rect["height"]
    x1, y1 = x1 - x1 % 2, y1 - y1 % 2
    return {"x": x0, "y": y0, "width": max(0, x1 - x0), "height": max(0, y1 - y0)}


def stable_rectangle(facts: Dict[str, Any]) -> Tuple[Optional[Dict[str, int]], str]:
    """(rectangle to propose, why) or (None, why not). Pure: the stability rule of ADR-060 on the measured facts."""
    sw, sh = facts.get("source_width"), facts.get("source_height")
    if not sw or not sh:
        return None, "the measurement does not report the frame size"
    if facts.get("rect") is None:
        return None, "cropdetect reported no rectangle (no readings: an all-black or undecodable sample)"
    windows = facts.get("windows")
    decoded = (windows - facts.get("windows_failed", 0)) if windows is not None else None
    if decoded is not None and decoded < MIN_WINDOWS:
        return None, f"only {decoded} of {windows} sampled windows decoded (needs {MIN_WINDOWS})"
    conf = facts.get("confidence")
    if conf is None or conf < MIN_CONFIDENCE:
        return None, f"the rectangle is not stable across the sampled windows (confidence {conf if conf is not None else 'missing'} < {MIN_CONFIDENCE:g})"
    r = even_inward(facts["rect"])
    if r["x"] < 0 or r["y"] < 0 or r["x"] + r["width"] > sw or r["y"] + r["height"] > sh:
        return None, f"the rectangle {_words(r)} does not lie inside the {sw}x{sh} frame"
    if r["width"] < MIN_SIDE or r["height"] < MIN_SIDE:
        return None, f"the rectangle {_words(r)} is smaller than {MIN_SIDE} px"
    if (sw - r["width"]) < MIN_BAR_FRACTION * sw and (sh - r["height"]) < MIN_BAR_FRACTION * sh:
        return None, f"no black bars: the active picture {_words(r)} is the {sw}x{sh} frame" + ("" if (r["width"], r["height"]) == (sw, sh) else f" within {MIN_BAR_FRACTION:.0%}")
    return r, f"stable across the sampled windows (confidence {conf:g}" + (f", {decoded} of {windows} windows decoded" if decoded is not None else "") + ")"


def decide_crop(eng, analysis, subjects: List[Tuple[str, List[str]]], edits: Dict[str, Any], skip: Optional[str], approval_for: Callable,
                probe_ids_of: Callable[[List[str]], List[str]], cap_block: Callable) -> None:
    """One decision per measured asset that is its own subject: a CONFIRM proposal of `video.crop` with the stable rectangle,
    or KEEP with the reason it is not proposed. `skip`: why no single input is a subject of its own (a programme / the audio
    path); then nothing is decided, the measurement stays an observation."""
    measured = [o for o in analysis.observations if o.kind == CROP_KIND]
    if not measured or CROP_SUBJECT in edits or skip:
        return
    own = {subject for subject, sources in subjects if sources == [subject]}
    proposed = False
    for o in measured:
        if o.asset_id not in own:
            continue
        asset = next((a for a in analysis.assets if a.id == o.asset_id), None)
        if asset is None or not (asset.technical or {}).get("video"):
            continue
        facts = crop_facts(o.data)
        rect, why = stable_rectangle(facts)
        ev = [o.id] + probe_ids_of([o.asset_id])
        frame = f"{facts['source_width']}x{facts['source_height']}"
        if rect is None:
            eng.decide(subject=CROP_SUBJECT, type="KEEP", decision=f"keep the full {frame} frame of {o.asset_id}", reason=f"crop not proposed: {why}",
                       confidence=1.0, evidence=ev, risk="LOW", approval="AUTO", provenance="OBSERVED", params={"asset_id": o.asset_id, "measured": facts["rect"]})
            continue
        eng.decide(subject=CROP_SUBJECT, type="TRANSFORM", decision=f"video.crop on {o.asset_id}: crop the black bars to {_words(rect)} (from {frame})",
                   reason=f"measured black bars ({o.source}): the active picture is {_words(rect)} of the {frame} frame, {why}; "
                          "proposed, not applied: look at the frame first (letterboxing can be the intended presentation)",
                   confidence=float(facts["confidence"] or 0.0), evidence=ev, risk="MEDIUM", approval=approval_for(CROP_SUBJECT, floor="CONFIRM"), provenance="OBSERVED",
                   params={"asset_id": o.asset_id, **rect}, serves_intent=None)
        proposed = True
    if proposed:
        cap_block(CROP_SKILL, f"capability.{CROP_SKILL}")


def _words(r: Dict[str, int]) -> str:
    return f"{r['width']}x{r['height']}+{r['x']}+{r['y']}"


def _num(v: Any) -> Optional[float]:
    try:
        return None if v is None or isinstance(v, bool) else float(v)
    except (TypeError, ValueError):
        return None


def _int(v: Any) -> Optional[int]:
    f = _num(v)
    return None if f is None else int(f)
