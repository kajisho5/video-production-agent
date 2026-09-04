"""Generic production inference over ProductionContexts (deterministic, evidence-based, domain-neutral).

What can be derived mechanically from "which event types are active when", for any event type and any asset:
- source_activity     per (timeline, event type/subtype): the union of the scopes where that kind of event is active
- source_inactivity   per (timeline, event type/subtype): the scopes inside the asset where no event of that kind is active
- transition          a boundary between two consecutive contexts whose situation (type/subtype set) differs
- conflict            two active events whose types are declared mutually exclusive (EXCLUSIVE_PAIRS) overlap: recorded,
                      never resolved (which one is right is not decided here)

Nothing here reads content, names a speaker, chooses a source, proposes an edit, or reads policy: those belong to the
domain inference (agent/speech_inference.py), the decision engine and the planner. An AI provider could later produce
inferences of these kinds only with provenance AI_GENERATED (validated by the existing reasoning boundary); the
generator recorded on every inference here tells the two apart."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..models import Event, Inference
from .model import ProductionContext

CONTEXT_INFERENCE_ID = "context_inference@1.0"
GENERIC_KINDS = ("source_activity", "source_inactivity", "transition", "conflict")
# event codes that cannot be true of the same interval; a listed pair overlapping is a measurement disagreement
# AUDIO_SILENCE / AUDIO_ACTIVE are not listed: the silence tool's keep ranges include a margin of air inside the silence by design
EXCLUSIVE_PAIRS: Tuple[Tuple[str, str], ...] = (("AUDIO_SILENCE", "SPEECH"),)
WHOLE_ASSET_SUBTYPES = ("loudness",)   # measurements that cover the programme by definition: activity of these says nothing about time


def _merge(ranges: List[List[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def infer_from_contexts(contexts: Iterable[ProductionContext], events: Dict[str, Event], durations: Dict[str, Optional[float]]) -> List[Inference]:
    ctxs = sorted(contexts, key=lambda c: (c.timeline_id, float(c.scope["start"]), float(c.scope["end"])))
    out: List[Inference] = []
    by_tl: Dict[str, List[ProductionContext]] = {}
    for c in ctxs:
        by_tl.setdefault(c.timeline_id, []).append(c)
    for tl_id, cs in by_tl.items():
        aid = cs[0].asset_ids[0] if cs[0].asset_ids else None
        dur = durations.get(aid) if aid else None
        # ---- activity / inactivity per track kind
        kinds: Dict[str, Dict[str, Any]] = {}
        for c in cs:
            for t in c.tracks:
                key = f"{t['event_type']}/{t['subtype']}"
                k = kinds.setdefault(key, {"ranges": [], "events": set(), "contexts": [], "event_type": t["event_type"], "subtype": t["subtype"]})
                k["ranges"].append([float(c.scope["start"]), float(c.scope["end"])])
                k["events"].update(t["event_ids"])
                k["contexts"].append(c.id)
        for key in sorted(kinds):
            k = kinds[key]
            if k["subtype"] in WHOLE_ASSET_SUBTYPES:
                continue
            active = _merge(k["ranges"])
            total = round(sum(e - s for s, e in active), 3)
            ev_ids = sorted(k["events"])
            out.append(Inference(kind="source_activity", asset_id=aid or "", confidence=1.0,
                                 statement=f"{key} is active in {len(active)} interval(s), {total:.2f}s" + (f" of {dur:.2f}s" if dur else "") + " on this timeline",
                                 evidence=ev_ids, data={"timeline_id": tl_id, "event_type": k["event_type"], "subtype": k["subtype"], "intervals": active, "seconds": total,
                                                        "coverage": round(total / dur, 4) if dur else None, "context_ids": k["contexts"], "generator": CONTEXT_INFERENCE_ID}))
            if dur is not None:
                gaps: List[List[float]] = []
                cursor = 0.0
                for s, e in active:
                    if s - cursor > 1e-6:
                        gaps.append([round(cursor, 6), round(s, 6)])
                    cursor = max(cursor, e)
                if float(dur) - cursor > 1e-6:
                    gaps.append([round(cursor, 6), round(float(dur), 6)])
                if gaps:
                    gap_ctx = [c.id for c in cs if any(c.scope["start"] < g[1] - 1e-6 and c.scope["end"] > g[0] + 1e-6 for g in gaps)]
                    out.append(Inference(kind="source_inactivity", asset_id=aid or "", confidence=1.0,
                                         statement=f"no {key} event in {len(gaps)} interval(s), {sum(e - s for s, e in gaps):.2f}s: nothing of that kind was observed there",
                                         evidence=ev_ids, data={"timeline_id": tl_id, "event_type": k["event_type"], "subtype": k["subtype"], "intervals": gaps,
                                                                "context_ids": gap_ctx, "generator": CONTEXT_INFERENCE_ID}))
        # ---- transitions: consecutive contexts with a different situation
        for prev, nxt in zip(cs, cs[1:]):
            if prev.signature == nxt.signature or abs(float(prev.scope["end"]) - float(nxt.scope["start"])) > 1e-6:
                continue
            started = sorted(set(nxt.event_ids) - set(prev.event_ids))
            ended = sorted(set(prev.event_ids) - set(nxt.event_ids))
            out.append(Inference(kind="transition", asset_id=aid or "", confidence=1.0,
                                 statement=f"at {float(nxt.scope['start']):.3f}s the situation changes from [{prev.signature}] to [{nxt.signature}]",
                                 evidence=started + ended, data={"timeline_id": tl_id, "at": float(nxt.scope["start"]), "from_context": prev.id, "to_context": nxt.id,
                                                                  "from": prev.signature, "to": nxt.signature, "started": started, "ended": ended, "generator": CONTEXT_INFERENCE_ID}))
        # ---- conflicts: exclusive event codes active together (recorded per pair once, over the union of their overlap)
        seen: set = set()
        for c in cs:
            codes = {events[e].type: e for e in c.event_ids if e in events}
            for a, b in EXCLUSIVE_PAIRS:
                if a in codes and b in codes and (codes[a], codes[b]) not in seen:
                    seen.add((codes[a], codes[b]))
                    ea, eb = events[codes[a]], events[codes[b]]
                    ov = [max(float(ea.range["start"]), float(eb.range["start"])), min(float(ea.range["end"]), float(eb.range["end"]))]
                    out.append(Inference(kind="conflict", asset_id=aid or "", confidence=1.0,
                                         statement=f"{a} and {b} overlap for {ov[1] - ov[0]:.2f}s at {ov[0]:.2f}s; the two measurements disagree and neither is preferred here",
                                         evidence=[ea.id, eb.id], data={"timeline_id": tl_id, "codes": [a, b], "overlap": ov, "sources": [ea.source, eb.source],
                                                                        "context_ids": [x.id for x in cs if codes[a] in x.event_ids and codes[b] in x.event_ids], "generator": CONTEXT_INFERENCE_ID}))
    return out
