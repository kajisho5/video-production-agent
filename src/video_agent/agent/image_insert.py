"""A still image joined before or after the finished picture (ADR-070): video-editing-skill IMAGE_INSERT (0.4.0, #21) turns the
still into a silent clip, video-editing CONCAT joins it to the picture. Requirement `edit.image_insert` (the image path) with
`edit.image_insert.duration` and `edit.image_insert.position` (before | after) → Decision `video.image_insert` → two plan steps
(`video_image_insert`, then the join `video_concat`) → one IR video operation `video.image_insert` → the compiler's two tool calls.

The still takes the picture's own frame and rate, so the join changes nothing in the picture: width / height are the subject's
size after its edits (`picture_size`) and fps its rate (the last edit that sets one, else the probe). Both are computed here from
the plan's own operations, never asked and never guessed: an unknown or odd size, or an unknown rate, is a BLOCK.

The join runs after colour, graphics and burnt-in subtitles (the still is not graded or captioned, and every finishing time stays
on the picture's own timeline) and before loudness. An editor timeline export names it among what a cut list does not carry (ADR-052's
`not_exported`), as for every other transform. What a joined still would silently break is a BLOCK instead:
- a still before the picture with a subtitle sidecar (every cue would be early by the still's duration),
- a soft subtitle track (`subtitle.mode=mux`; the join re-encodes and nothing here carries the track over),
- a QC reference master (the deliverable would no longer line up with it),
- the audio production path (no picture is delivered).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from .editing import EDIT_ORDER, IMAGE_INSERT_OP, IMAGE_INSERT_SKILL, IMAGE_JOIN_SKILL, OPERATIONS
from .finishing import picture_size

FPS_KEYS_OPS = ("video.concat", "video.grid", "video.crop", "video.redact", "video.resize", "video.fit", "video.fill")   # ops whose `fps` sets the output rate


def still_frame(technical: Dict[str, Any], video_ops: List[Dict[str, Any]], subject: str) -> Tuple[Optional[Tuple[int, int, Any]], str]:
    """((width, height, fps), "") for the still of `subject`, or (None, why) when the frame cannot be stated exactly."""
    size = picture_size(technical, video_ops, subject)
    if size is None:
        return None, "the picture size is unknown (the probe reports no width / height)"
    w, h = size
    if w % 2 or h % 2 or w < 16 or h < 16:
        return None, f"the picture is {w}x{h}; IMAGE_INSERT and CONCAT take even sizes of 16 px or more"
    fps: Any = None
    for op in video_ops:
        if op.get("asset") == subject and op.get("type") in FPS_KEYS_OPS and op.get("fps") is not None:
            fps = op["fps"]
    if fps is None:
        f = ((technical or {}).get("video") or {}).get("fps")
        if isinstance(f, (int, float)) and not isinstance(f, bool) and 1 <= float(f) <= 240:
            fps = round(float(f), 3)
    if fps is None:
        return None, "the picture's frame rate is unknown (the probe reports none)"
    return (int(w), int(h), fps), ""


def _decided_ops(decs: List[Any], subject: str) -> List[Dict[str, Any]]:
    """The subject's planned editing operations as IR-like records (decision params), in the fixed order: what `picture_size` reads."""
    ops = [dict({k: v for k, v in d.params.items() if k != "asset_id"}, type=d.subject, asset=subject) for d in decs
           if d.type == "TRANSFORM" and d.status != "REJECTED" and d.subject in OPERATIONS and d.subject != IMAGE_INSERT_OP and d.params.get("asset_id") == subject]
    return sorted(ops, key=lambda o: EDIT_ORDER.index(o["type"]))


def decide_image_insert(eng: Any, decs: List[Any], m: Dict[str, Any], analysis: Any, subjects: List[Tuple[str, List[str]]], edit: Dict[str, Any],
                        audio_production: bool, approval_for: Any, probe_ids_of: Any, cap_block: Any) -> None:
    req = edit["requirements"]
    p = edit["params"]
    for subject, sources in subjects:
        src = [a for a in analysis.assets if a.id in sources]
        ev = [r.id for r in req] + probe_ids_of(sources)

        def block(why: str) -> None:
            eng.decide(subject=IMAGE_INSERT_OP, type="BLOCK", decision=f"BLOCK: {IMAGE_INSERT_OP} on {subject}", reason=why + "; a still is never joined by guessing",
                       confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": subject}, requirements=req)
        if audio_production:
            block("the audio production path delivers no picture to join a still to")
            continue
        if not src or not all(a.technical.get("video") for a in src):
            block(f"{subject} has no video stream")
            continue
        live = lambda subj: [d for d in decs if d.subject == subj and d.type == "TRANSFORM" and d.status != "REJECTED" and d.params.get("asset_id") == subject]  # noqa: E731
        if live("subtitle.mux"):
            block("a soft subtitle track (subtitle.mode=mux) would not survive the join, which re-encodes the programme")
            continue
        if p["position"] == "before" and live("subtitle.generate"):
            block(f"every cue of the subtitle sidecar would be {p['duration']:g}s early once the still is joined before the picture (use position=after or no sidecar)")
            continue
        if m.get("qc.reference_video") is not None and m["qc.reference_video"].provenance != "DEFAULT":
            block("the deliverable would no longer line up with the QC reference master")
            continue
        frame, why = still_frame(src[0].technical, _decided_ops(decs, subject), subject)
        if frame is None:
            block(why)
            continue
        w, h, fps = frame
        eng.decide(subject=IMAGE_INSERT_OP, type="TRANSFORM",
                   decision=f"{IMAGE_INSERT_OP} on {subject}: {p['duration']:g}s still {p['position']} the picture ({w}x{h} @ {fps} fps, image {os.path.basename(p['image'])})",
                   reason=f"user asked for a still {p['position']} the picture ({req[0].source}); it takes the picture's own frame and rate, is joined after the "
                          "finishing (not graded or captioned) and has no audio (the join inserts silence under it)",
                   confidence=1.0, evidence=ev, risk=OPERATIONS[IMAGE_INSERT_OP]["risk"], approval=approval_for(IMAGE_INSERT_OP, explicit=req[0]), provenance="USER",
                   params={"asset_id": subject, "image": p["image"], "duration": p["duration"], "position": p["position"], "width": w, "height": h, "fps": fps},
                   requirements=req, serves_intent=None)
    cap_block(IMAGE_INSERT_SKILL, f"capability.{IMAGE_INSERT_SKILL}")
    cap_block(IMAGE_JOIN_SKILL, f"capability.{IMAGE_JOIN_SKILL}")
