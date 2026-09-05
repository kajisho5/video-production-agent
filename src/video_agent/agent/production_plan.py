"""Production Plan: the deterministic bridge between Decisions / Events and the Project IR (ADR-021).

    Observation → Event → Inference / Decision → ProductionPlan → Project IR → Compiler → Tool → Execution → QA

A ProductionPlan says which production steps happen, in what order, on what inputs, with which evidence, realised by
which registry-selected skill / tool. It is not an execution contract (that is the IR's video / audio / delivery
sections, compiled by execution/compiler.py) and not a decision (agent/decision.py). The planner is deterministic:
the same project, decisions, events and constraints produce the same plan identity. Nothing here executes anything,
selects a tool outside SkillRegistry, or accepts a command / argv from a decision, an event or an AI response.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..models import Model, TimeRange, now_iso, stable_hash

PLAN_STATUSES = ("DRAFT", "REVIEW", "APPROVED", "REJECTED", "BLOCKED")
STEP_STATUSES = ("PROPOSED", "APPROVED", "REJECTED", "BLOCKED")
PLANNER_ID = "production_planner@1.0"
# parameters a step may carry (domain vocabulary): anything else is dropped by the planner and refused by the validator
STEP_PARAMETERS = {"silence_cleanup": ("asset", "keep", "removed", "accurate"), "loudness_normalization": ("asset", "target_lufs", "true_peak"),
                   "delivery_export": ("preset", "target"), "delivery_check": ("platform", "target"),
                   # editing operations (ADR-029): the IR parameter allowlist of each operation plus the subject / references
                   "video_concat": ("asset", "inputs", "transition", "width", "height", "fps", "mode", "pad_color"), "video_speed": ("asset", "factor"),
                   "video_resize": ("asset", "width", "fps"), "video_fit": ("asset", "aspect", "width", "pad_color", "fps"), "video_fill": ("asset", "aspect", "width", "fps"),
                   "video_overlay": ("asset", "image", "position", "margin", "scale", "opacity", "start", "end", "fade"),
                   # audio production path (ADR-030): the subject's audio through audio-production-skill
                   "audio_cut": ("asset", "remove"), "audio_normalize": ("asset", "target_lufs", "true_peak", "tolerance_lu", "sample_rate"), "audio_gain": ("asset", "gain_db"),
                   "audio_mono": ("asset",), "audio_stereo": ("asset",), "audio_downmix": ("asset",), "audio_fade_in": ("asset", "duration"), "audio_fade_out": ("asset", "duration"),
                   "audio_concat": ("asset", "inputs", "crossfade"),
                   # finishing (ADR-031): colour / graphics / subtitles / thumbnail, and the QC gate (ADR-032)
                   "color_strip_dovi": ("asset",), "color_hdr_to_sdr": ("asset",), "color_lut": ("asset", "lut", "lut_strength"), "color_retag": ("asset", "target"),
                   "color_primary_correction": ("asset", "exposure", "contrast", "saturation", "temperature", "tint"),
                   "motion_graphics": ("asset", "elements"), "subtitle_generation": ("asset", "format", "language", "cues"), "subtitle_burn_in": ("asset",),
                   "thumbnail_frame": ("asset", "at", "format"), "thumbnail_render": ("asset", "at", "format", "text"), "qc_check": ("asset", "target", "kind", "artifact")}


@dataclass
class ProductionStep(Model):
    id: str
    order: int
    skill: str                                    # production skill (intent) — registry vocabulary
    tool: Optional[str]                           # registry-selected tool id; None = not executable here (validator / decide() block it)
    inputs: List[str] = field(default_factory=list)      # asset ids / logical artifact names consumed
    params: Dict[str, Any] = field(default_factory=dict)  # domain parameters (STEP_PARAMETERS), never tool argv
    outputs: List[str] = field(default_factory=list)     # logical artifact names produced (compiler decides paths)
    depends_on: List[str] = field(default_factory=list)  # step ids
    evidence: List[str] = field(default_factory=list)    # inference / event / observation ids behind the step's decisions
    decision_ids: List[str] = field(default_factory=list)
    decision_id: Optional[str] = None             # primary decision
    temporal_scope: Optional[Dict[str, Any]] = None   # TimeRange.to_dict() on the input asset
    status: str = "PROPOSED"                      # derived from the decisions (STEP_STATUSES)


@dataclass
class ProductionPlan(Model):
    id: str
    project_id: str
    version: int
    status: str                                   # PLAN_STATUSES (derived from the IR's review state, see plan_status)
    objective: str
    inputs: List[str]                             # asset ids
    steps: List[Dict[str, Any]]                   # ProductionStep.to_dict() in deterministic order
    outputs: List[Dict[str, Any]]                 # planned artifacts: role / logical name / format / expected
    decisions: List[str]                          # decision ids the plan rests on
    events: List[str]                             # event ids cited as evidence
    constraints: List[Dict[str, Any]]             # hard rules in force (id / key / value) — limits, not decisions
    provenance: Dict[str, Any]
    summary: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)

    @staticmethod
    def make_id(project_id: str, version: int, steps: Iterable[Dict[str, Any]], constraints: Iterable[Dict[str, Any]]) -> str:
        content = [[s["id"], s["skill"], s.get("params"), s.get("inputs"), s.get("depends_on")] for s in steps]
        return "plan_" + stable_hash([project_id, version, content, [c.get("id") for c in constraints]])[:16]


# ---- dependency graph
def topological_order(steps: List[Dict[str, Any]]) -> List[str]:
    """Deterministic Kahn ordering (ties broken by declared order, then id). Raises ValueError on a missing dependency or a cycle."""
    ids = {s["id"] for s in steps}
    deps = {s["id"]: list(s.get("depends_on") or []) for s in steps}
    for sid, ds in deps.items():
        missing = [d for d in ds if d not in ids]
        if missing:
            raise ValueError(f"step {sid} depends on unknown step(s): {', '.join(missing)}")
    rank = {s["id"]: (int(s.get("order", 0)), s["id"]) for s in steps}
    remaining = dict(deps)
    out: List[str] = []
    while remaining:
        ready = sorted([sid for sid, ds in remaining.items() if all(d in out for d in ds)], key=lambda x: rank[x])
        if not ready:
            raise ValueError("dependency cycle among steps: " + ", ".join(sorted(remaining)))
        out.append(ready[0])
        del remaining[ready[0]]
    return out


# ---- status derivation (the IR's reviews / approvals stay the source of truth; nothing here approves anything)
def step_status(step: Dict[str, Any], decisions: Dict[str, Dict[str, Any]]) -> str:
    ds = [decisions[i] for i in step.get("decision_ids") or [] if i in decisions]
    if any(d["status"] == "BLOCKED" or d["approval"] == "BLOCK" for d in ds) or not step.get("tool"):
        return "BLOCKED"
    if any(d["status"] == "REJECTED" for d in ds):
        return "REJECTED"
    if all(d["status"] == "APPROVED" or d["approval"] == "AUTO" for d in ds):
        return "APPROVED"
    return "PROPOSED"


def plan_status(doc: Dict[str, Any]) -> str:
    """DRAFT: no steps yet (nothing to run). REJECTED: a step cites a rejected decision. BLOCKED: a step cannot execute
    here (BLOCK decision / no tool) or a BLOCK decision is in force (ADR-029: a refused request blocks the plan even without a step). REVIEW: a CONFIRM decision is pending or the version needs re-approval.
    APPROVED: every step's decisions are approved (explicitly, or AUTO by policy) and the version is not awaiting review."""
    decisions = {d["id"]: d for d in doc.get("decisions") or []}
    steps = (doc.get("plan") or {}).get("steps") or []
    statuses = [step_status(s, decisions) for s in steps]
    if "REJECTED" in statuses:
        return "REJECTED"
    if "BLOCKED" in statuses or any(d["approval"] == "BLOCK" and d["status"] == "BLOCKED" for d in decisions.values()):
        return "BLOCKED"   # a BLOCK decision in force blocks the whole plan, whether or not a step cites it (a refused request is never partially run)
    if not steps:
        return "DRAFT"
    version = int(doc["plan"].get("version", 1))
    approved_version = (doc.get("revision") or {}).get("approved_plan_version")
    pending = [d for d in decisions.values() if d["approval"] == "CONFIRM" and d["status"] == "PROPOSED"]
    if pending or (version > 1 and approved_version != version and any(d["status"] != "APPROVED" for d in decisions.values() if d["approval"] == "CONFIRM")):
        return "REVIEW"
    if version > 1 and approved_version != version:
        return "REVIEW"
    return "APPROVED"


def executable_steps(doc: Dict[str, Any]) -> List[str]:
    """Step ids whose decisions allow execution (partial approval: only these reach the compiler through the IR)."""
    decisions = {d["id"]: d for d in doc.get("decisions") or []}
    return [s["id"] for s in (doc.get("plan") or {}).get("steps") or [] if step_status(s, decisions) == "APPROVED"]


def subject_duration(doc: Dict[str, Any], subject: str, step: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Length of the timeline a step's temporal scope refers to: the source duration of an asset, the concat programme's
    timeline (a derived scope from the IR operation, not a source), and — for the speed step and every step after it — the
    length changed by the video.speed factor (the scope of a step is on the timeline its output has)."""
    ops = (doc.get("video") or {}).get("operations") or []
    aops = (doc.get("audio") or {}).get("operations") or []
    assets = doc.get("assets") or {}
    steps = (doc.get("plan") or {}).get("steps") or []
    if subject in assets:
        dur = ((assets[subject].get("technical") or {}).get("duration"))
    else:
        dur = next((op.get("timeline_duration") for op in ops + aops if op.get("type") in ("video.concat", "audio.concat") and op.get("output") == subject), None)
    if dur is None:
        return None
    speed = next((op for op in ops if op.get("type") == "video.speed" and op.get("asset") == subject and op.get("factor")), None)
    if speed is not None:
        speed_step = next((s for s in steps if s.get("skill") == "video_speed" and (s.get("params") or {}).get("asset") == subject), None)
        after = step is None or speed_step is None or int(step.get("order", 0)) >= int(speed_step.get("order", 0))
        if after:
            dur = float(dur) / float(speed["factor"])
    return float(dur)


# ---- validation
def validate_plan(doc: Dict[str, Any], registry=None, supports=None) -> List[str]:
    """Structural / boundary errors of the plan recorded in an IR (the IR validator adds skill / tool / capability checks)."""
    from ..media.analysis import leak_scan
    errs: List[str] = []
    plan = doc.get("plan") or {}
    steps = plan.get("steps") or []
    assets = doc.get("assets") or {}
    decisions = {d["id"]: d for d in doc.get("decisions") or []}
    known_evidence = ({o["id"] for o in (doc.get("analysis") or {}).get("observations") or []} | {i["id"] for i in (doc.get("analysis") or {}).get("inferences") or []}
                      | {e["id"] for e in (doc.get("timeline") or {}).get("events") or []} | set(decisions) | {r["id"] for r in doc.get("requirements") or []})
    if plan.get("id") and not str(plan["id"]).startswith("plan_"):
        errs.append(f"invalid plan id {plan['id']!r}")
    if plan.get("project_id") and plan["project_id"] != (doc.get("project") or {}).get("id"):
        errs.append("plan belongs to another project")
    if plan.get("status") and plan["status"] not in PLAN_STATUSES:
        errs.append(f"invalid plan status {plan['status']!r}")
    if plan.get("status") and plan["status"] != plan_status(doc):
        errs.append(f"plan status {plan['status']} does not match the review state ({plan_status(doc)})")
    ids = [s["id"] for s in steps]
    if len(ids) != len(set(ids)):
        errs.append("step ids are not unique")
    try:
        if topological_order(steps) != [s["id"] for s in steps]:
            errs.append("steps are not recorded in deterministic dependency order")
    except ValueError as ex:
        errs.append(str(ex))
    logical_outputs: Set[str] = set()
    for s in steps:
        for f in ("id", "skill", "decision_ids"):
            if f not in s:
                errs.append(f"step {s.get('id')} misses {f}")
        if s.get("status") and s["status"] not in STEP_STATUSES:
            errs.append(f"step {s['id']}: invalid status {s['status']!r}")
        if s.get("status") and s["status"] != step_status(s, decisions):
            errs.append(f"step {s['id']}: status {s['status']} does not match its decisions ({step_status(s, decisions)})")
        for i in s.get("inputs") or []:
            if i not in assets and i not in logical_outputs:
                errs.append(f"step {s['id']}: input {i!r} is neither an asset nor an earlier output")
        logical_outputs.update(s.get("outputs") or [])
        for dcs in s.get("decision_ids") or []:
            if dcs not in decisions:
                errs.append(f"step {s['id']}: unknown decision {dcs}")
        if s.get("decision_id") and s["decision_id"] not in (s.get("decision_ids") or []):
            errs.append(f"step {s['id']}: primary decision is not among its decisions")
        for ev in s.get("evidence") or []:
            if ev not in known_evidence:
                errs.append(f"step {s['id']}: evidence {ev} not found")
        allowed = STEP_PARAMETERS.get(s.get("skill"), ())
        for k in (s.get("params") or {}):
            if k not in allowed:
                errs.append(f"step {s['id']}: parameter {k!r} is not a domain parameter of {s.get('skill')}")
        errs += [f"step {s['id']}: params leak {w}" for w in leak_scan(s.get("params") or {}, "params")]
        scope = s.get("temporal_scope")
        if scope is not None:
            try:
                rng = TimeRange(scope["start"], scope.get("end"))
                aid = (s.get("params") or {}).get("asset")
                dur = subject_duration(doc, aid, s) if aid else None
                if not rng.within(dur):
                    errs.append(f"step {s['id']}: temporal scope {scope} exceeds asset duration {dur}")
            except (ValueError, KeyError, TypeError) as ex:
                errs.append(f"step {s['id']}: invalid temporal scope: {ex}")
    for o in plan.get("outputs") or []:
        for f in ("role", "logical", "format"):
            if f not in o:
                errs.append(f"planned output misses {f}")
    if registry is not None:
        for s in steps:
            if s.get("skill") in registry.names() and s.get("tool") and s["tool"] not in registry.get(s["skill"]).tools:
                errs.append(f"step {s['id']}: tool {s['tool']} does not belong to skill {s['skill']}")
    return errs


# ---- explainability
def explain_step(doc: Dict[str, Any], step_id: str) -> Dict[str, Any]:
    """Evidence chain of a step: decisions → inferences (incl. AI provenance) → events → observations → tool sources."""
    steps = {s["id"]: s for s in (doc.get("plan") or {}).get("steps") or []}
    if step_id not in steps:
        raise KeyError(step_id)
    s = steps[step_id]
    decisions = {d["id"]: d for d in doc.get("decisions") or []}
    infs = {i["id"]: i for i in doc["analysis"].get("inferences") or []}
    obs = {o["id"]: o for o in doc["analysis"].get("observations") or []}
    events = {e["id"]: e for e in doc["timeline"].get("events") or []}
    ai_calls = doc.get("provenance", {}).get("ai_calls") or []
    chain: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def walk(eid: str, depth: int) -> None:
        if eid in seen:
            return
        seen.add(eid)
        if eid in infs:
            i = infs[eid]
            row = {"level": depth, "kind": "inference", "id": eid, "detail": i.get("statement"), "provenance": i.get("provenance"), "confidence": i.get("confidence")}
            if (i.get("data") or {}).get("context_ids"):
                row["contexts"] = list(i["data"]["context_ids"])   # the situations this inference was derived from (explain --context)
            if i.get("provenance") == "AI_GENERATED":
                row["ai"] = {"provider": i["data"].get("provider"), "model": i["data"].get("model"), "response_hash": i["data"].get("response_hash"),
                             "call": next((c for c in ai_calls if c.get("response_hash") == i["data"].get("response_hash")), None)}
            chain.append(row)
            for x in i.get("evidence") or []:
                walk(x, depth + 1)
        elif eid in events:
            e = events[eid]
            chain.append({"level": depth, "kind": "event", "id": eid, "detail": f"{e.get('event_type')}/{e.get('subtype')} {e['range']}", "provenance": e.get("provenance") or e.get("kind"), "source": e.get("source")})
            for x in e.get("evidence") or []:
                walk(x, depth + 1)
        elif eid in obs:
            o = obs[eid]
            chain.append({"level": depth, "kind": "observation", "id": eid, "detail": o.get("kind"), "provenance": o.get("provenance", "OBSERVED"), "source": o.get("source"), "analysis_id": o.get("analysis_id")})
        elif eid in decisions:
            chain.append({"level": depth, "kind": "decision", "id": eid, "detail": decisions[eid].get("decision"), "provenance": decisions[eid].get("provenance")})
        else:
            chain.append({"level": depth, "kind": "reference", "id": eid})

    for did in s.get("decision_ids") or []:
        d = decisions.get(did)
        if not d:
            continue
        chain.append({"level": 0, "kind": "decision", "id": did, "detail": f"{d['subject']}: {d['decision']} — {d['reason']}", "provenance": d.get("provenance"),
                      "approval": d["approval"], "risk": d["risk"], "status": d["status"]})
        for x in d.get("evidence") or []:
            walk(x, 1)
    return {"step": s, "chain": chain}
