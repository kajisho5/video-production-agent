"""Production Decision Engine core: the generic, tool-independent, domain-independent rules every Decision obeys.

    Inference (what is happening) + Policy / Preference / Constraint + Intent + Risk → Decision (what production should do)

The engine does not know silence, speech, loudness or delivery. The domain code in `decision.py` states *which* decision
a situation calls for; this module states *how* a decision is allowed to exist:

- **Evidence is mandatory.** Every decision cites evidence that exists in this analysis (observation / event /
  inference / requirement / rule / capability). A decision without evidence is refused at construction, not at review.
- **Grounding for anything that changes media or output.** A REMOVE / TRANSFORM / DELIVER decision must rest on a
  measured fact (observation, event, non-AI inference) or on a requirement of the request (what was asked, including
  the profile's completion of the request). Preference / policy rules alone, an intent alone, or AI text alone never
  ground such a decision: they can only shape parameters, approval and risk, or produce a REVIEW item.
- **Approval is resolved from policy with a safe default.** `resolve_approval` reads the effective rule for a key,
  records where it came from (USER / PROFILE / SYSTEM / DEFAULT — the existing RuleSet precedence, unchanged), maps an
  unknown value to CONFIRM, honours BLOCK, applies a floor (CONFIRM for content-adjacent edits) and never lowers BLOCK.
  The one downward step that exists — CONFIRM → AUTO when the user asked for exactly this edit explicitly — is
  recorded as a note and never applies to a CONSTRAINT.
- **Confidence ≠ risk ≠ approval.** Confidence is inherited from the inference; risk and approval are set by policy and
  by the kind of change, never derived from confidence.
- **BLOCK and REJECTED are never executable**; a BLOCK approval always carries status BLOCKED.
- **Nothing executable is interpreted.** Decision parameters are scanned for command / argv / shell / credential
  material and refused; a decision names ranges, targets and levels, never how a tool is invoked.
- **Provenance of the basis is recorded on the decision** (`basis`): the settings consulted with value, kind, scope
  provenance and rule id, the approval resolution with its notes, the intent served, the requirements consulted and the
  classes of evidence — so `explain --decision` can show policy / preference / constraint / intent / evidence.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..media.analysis import _scrub, leak_scan
from ..models import APPROVALS, RISKS, Decision, Inference, Intent, Requirement
from ..policy.rules import Rule, RuleSet

ENGINE_ID = "decision_engine@1.0"
# decision vocabulary (what production should do): deliberately small, the existing decisions all fit it
DECISION_TYPES = ("KEEP", "REMOVE", "TRANSFORM", "DELIVER", "SKIP", "REVIEW", "BLOCK")
EXECUTABLE_TYPES = ("REMOVE", "TRANSFORM", "DELIVER")   # the only types a plan step / IR operation may cite
APPROVAL_DEFAULT = "CONFIRM"                            # what an unknown / ambiguous policy resolves to
BASIS_SOURCE = "video_agent.agent.decision_engine"
# evidence classes; "fact" classes ground an executable decision, the request does too, rules / ai never do
GROUNDING = ("observation", "event", "inference", "requirement")
EVIDENCE_PREFIXES = ("capability:", "skill:")           # environment facts cited by BLOCK decisions (no id record in the IR)


class DecisionError(ValueError):
    """A decision that violates an engine invariant (no evidence, ungrounded, unknown type / approval, executable material)."""


def resolve_setting(rules: RuleSet, key: str, default: Any, source: str = BASIS_SOURCE) -> Dict[str, Any]:
    """Effective value of a policy / preference / constraint key with its provenance. The RuleSet precedence is not touched
    here: what it resolved is reported; an absent key yields the explicit DEFAULT the caller passed (never an implicit one)."""
    r: Optional[Rule] = rules.effective.get(key)
    if r is None:
        return {"key": key, "value": default, "kind": None, "provenance": "DEFAULT", "source": source, "rule_id": None, "hard": False}
    return {"key": key, "value": r.value, "kind": r.kind, "provenance": rules.provenance(key) or "SYSTEM", "source": r.source, "rule_id": r.id, "hard": r.hard}


def resolve_approval(rules: RuleSet, key: str, default: str = APPROVAL_DEFAULT, floor: Optional[str] = None,
                     explicit: Optional[Requirement] = None) -> Dict[str, Any]:
    """Approval for a subject from policy key `key`:
    - BLOCK* → BLOCK (a value like BLOCK_UNLESS_EXPLICIT is BLOCK here: no extra rule is invented for its suffix)
    - AUTO / CONFIRM → as stated; any other value → CONFIRM (unknown policy is never AUTO)
    - `explicit`: a USER requirement asking for exactly this edit waives CONFIRM (the request is the confirmation) unless
      the rule is a CONSTRAINT; BLOCK is never waived
    - `floor`: the approval is raised to at least this level (never lowered)
    Returns {"approval", "setting", "notes"}; the caller stores it in the decision basis."""
    if default not in APPROVALS:
        raise DecisionError(f"approval default {default!r} for {key} is not one of {APPROVALS}")
    s = resolve_setting(rules, key, default)
    raw = str(s["value"]).upper()
    notes: List[str] = []
    if raw.startswith("BLOCK"):
        approval = "BLOCK"
        if raw != "BLOCK":
            notes.append(f"{s['value']} is applied as BLOCK (no implicit exception)")
    elif raw in ("AUTO", "CONFIRM"):
        approval = raw
    else:
        approval = "CONFIRM"
        notes.append(f"unknown approval value {s['value']!r} for {key}: CONFIRM (safe default)")
    if explicit is not None and explicit.provenance == "USER" and approval == "CONFIRM":
        if s["hard"]:
            notes.append(f"CONFIRM kept: {key} is a CONSTRAINT ({s['rule_id']}); the explicit request does not waive it")
        else:
            approval = "AUTO"
            notes.append(f"CONFIRM waived: the user asked for this explicitly ({explicit.key} from {explicit.source})")
    if floor is not None:
        if floor not in APPROVALS:
            raise DecisionError(f"approval floor {floor!r} is not one of {APPROVALS}")
        if APPROVALS.index(approval) < APPROVALS.index(floor):
            notes.append(f"raised {approval} → {floor}: floor for this kind of decision")
            approval = floor
    return {"approval": approval, "setting": s, "notes": notes}


def raise_approval(resolved: Dict[str, Any], to: str, why: str) -> Dict[str, Any]:
    """Raise a resolved approval (never lower it), recording why."""
    if to not in APPROVALS:
        raise DecisionError(f"approval {to!r} is not one of {APPROVALS}")
    if APPROVALS.index(resolved["approval"]) < APPROVALS.index(to):
        resolved["notes"].append(f"raised {resolved['approval']} → {to}: {why}")
        resolved["approval"] = to
    return resolved


class DecisionEngine:
    """Constructs decisions under the invariants above for one reasoning pass (one RuleSet, one Intent, one evidence set)."""

    def __init__(self, rules: RuleSet, intent: Intent, known: Dict[str, str], requirements: Optional[Iterable[Requirement]] = None):
        """`known`: evidence id → class (observation | event | inference | ai | requirement | rule | context)."""
        self.rules, self.intent, self.known = rules, intent, dict(known)
        self.requirements = {r.id: r for r in (requirements or [])}
        self.decisions: List[Decision] = []

    @staticmethod
    def evidence_index(observations=(), events=(), inferences: Sequence[Inference] = (), requirements: Sequence[Requirement] = (), rules: Optional[RuleSet] = None,
                       contexts=(), ai_prefix: str = "") -> Dict[str, str]:
        known: Dict[str, str] = {}
        for o in observations:
            known[o.id] = "observation"
        for e in events:
            known[e.id] = "event"
        for i in inferences:
            known[i.id] = "ai" if (i.provenance == "AI_GENERATED" or (ai_prefix and i.kind.startswith(ai_prefix))) else "inference"
        for r in requirements:
            known[r.id] = "requirement"
        for r in (rules.all_rules if rules else []):
            known[r.id] = "rule"
        for c in contexts:
            known[c.id] = "context"
        return known

    def classes_of(self, evidence: Iterable[str]) -> List[str]:
        out = []
        for e in evidence:
            if e in self.known:
                out.append(self.known[e])
            elif e.startswith(EVIDENCE_PREFIXES):
                out.append(e.split(":", 1)[0])
            else:
                raise DecisionError(f"evidence {e!r} is not an observation, event, inference, requirement, rule or capability of this analysis")
        return sorted(set(out))

    def decide(self, subject: str, type: str, decision: str, reason: str, evidence: List[str], risk: str, approval: Any, confidence: float,
               provenance: str, params: Optional[Dict[str, Any]] = None, alternatives: Optional[List[Dict[str, Any]]] = None, status: Optional[str] = None,
               settings: Optional[List[Dict[str, Any]]] = None, requirements: Optional[List[Requirement]] = None, serves_intent: Optional[str] = None) -> Decision:
        """`approval`: a string (AUTO/CONFIRM/BLOCK, for no-op decisions and blocks) or a `resolve_approval` result."""
        if type not in DECISION_TYPES:
            raise DecisionError(f"decision {subject}: type {type!r} is not one of {DECISION_TYPES}")
        if risk not in RISKS:
            raise DecisionError(f"decision {subject}: risk {risk!r} is not one of {RISKS}")
        resolved = approval if isinstance(approval, dict) else {"approval": approval, "setting": None,
                                                                "notes": [f"fixed {approval} for a {type} decision (no policy key consulted; it creates no operation)" if type not in EXECUTABLE_TYPES else f"fixed {approval}"]}
        appr = resolved["approval"]
        if appr not in APPROVALS:
            raise DecisionError(f"decision {subject}: approval {appr!r} is not one of {APPROVALS}")
        ev: List[str] = []
        for e in evidence:
            if e not in ev:
                ev.append(e)
        if not ev:
            raise DecisionError(f"decision {subject}: no evidence; a decision without evidence is not made")
        classes = self.classes_of(ev)
        if type in EXECUTABLE_TYPES and not any(c in GROUNDING for c in classes):
            raise DecisionError(f"decision {subject}: {type} needs a measured fact or a requirement as evidence, got only {classes} "
                                "(preference, intent or AI output alone never ground an executable decision)")
        if classes == ["ai"] and type != "REVIEW":
            raise DecisionError(f"decision {subject}: AI output alone yields a REVIEW item, not a {type} decision")
        params = dict(params or {})
        leaks = leak_scan({"subject": subject, "params": params, "decision": decision})
        scrubbed: List[str] = []
        if leaks and type == "REVIEW" and not leak_scan({"subject": subject, "decision": decision}):
            params, scrubbed = _scrub(params), leaks   # AI-proposed parameters are never interpreted: the material is dropped, the fact recorded
        elif leaks:
            raise DecisionError(f"decision {subject}: executable / credential material refused ({'; '.join(leaks)})")
        if appr == "BLOCK":
            status = "BLOCKED"
        elif status is None:
            status = "PROPOSED"
        if status == "BLOCKED" and appr != "BLOCK":
            raise DecisionError(f"decision {subject}: status BLOCKED needs approval BLOCK")
        if type == "REVIEW":
            params.setdefault("executable", False)
        req_ids = [r.id for r in (requirements or []) if r is not None]
        for rid in req_ids:
            if rid not in self.known:
                raise DecisionError(f"decision {subject}: requirement {rid} is not part of this request")
        basis = {"engine": ENGINE_ID, "type": type, "evidence_classes": classes,
                 "settings": [dict(s) for s in (settings or [])] + ([resolved["setting"]] if resolved.get("setting") else []),
                 "approval": {"resolved": appr, "key": (resolved.get("setting") or {}).get("key"), "provenance": (resolved.get("setting") or {}).get("provenance", "SYSTEM"),
                              "notes": list(resolved.get("notes") or [])},
                 "intent": {"primary": self.intent.primary, "secondary": list(self.intent.secondary), "provenance": self.intent.provenance,
                            "served": serves_intent or None},   # the intent this decision serves (None: a fact-backed / safety decision, not what was asked)
                 "requirements": [{"id": r.id, "key": r.key, "value": r.value, "provenance": r.provenance, "source": r.source} for r in (requirements or []) if r is not None],
                 "risk": {"level": risk, "independent_of_confidence": True}}
        if scrubbed:
            basis["approval"]["notes"].append("executable / credential-looking material removed from proposed params: " + "; ".join(scrubbed))
        d = Decision(subject=subject, decision=decision, reason=reason, confidence=float(confidence), evidence=ev, risk=risk, approval=appr,
                     alternatives=list(alternatives or []), provenance=provenance, status=status, params=params, type=type, basis=basis)
        self.decisions.append(d)
        return d


# ---- invariants on a recorded IR (validator) ----------------------------------------------------------------------------
def check_decisions(doc: Dict[str, Any]) -> List[str]:
    """Engine invariants re-checked on a Project IR: evidence present and known, type known, BLOCK ⇔ BLOCKED, an operation /
    step cites executable types only, AI-only evidence stays REVIEW / non-executable. (A step citing a BLOCKED or REJECTED
    decision is valid IR that the plan status keeps from executing — not an error here.)"""
    errs: List[str] = []
    an = doc.get("analysis") or {}
    known: Dict[str, str] = {}
    for o in an.get("observations") or []:
        known[o["id"]] = "observation"
    for e in (doc.get("timeline") or {}).get("events") or []:
        known[e["id"]] = "event"
    for i in an.get("inferences") or []:
        known[i["id"]] = "ai" if i.get("provenance") == "AI_GENERATED" else "inference"
    for r in doc.get("requirements") or []:
        known[r["id"]] = "requirement"
    for r in ((doc.get("policy") or {}).get("effective") or {}).values():
        known[r["id"]] = "rule"
    for c in ((doc.get("policy") or {}).get("conflicts") or []):
        known[c["constraint"]["id"]] = "rule"
        known[c["attempted"]["id"]] = "rule"
    for c in an.get("contexts") or []:
        known[c["id"]] = "context"
    decisions = {d["id"]: d for d in doc.get("decisions") or []}
    for d in decisions.values():
        tag = f"decision {d['id']} ({d['subject']})"
        if d.get("type") not in DECISION_TYPES:
            errs.append(f"{tag}: type {d.get('type')!r} is not one of {DECISION_TYPES}")
        if not d.get("evidence"):
            errs.append(f"{tag}: has no evidence")
        classes = set()
        for e in d.get("evidence") or []:
            if e in known:
                classes.add(known[e])
            elif str(e).startswith(EVIDENCE_PREFIXES):
                classes.add(str(e).split(":", 1)[0])
            elif d.get("status") == "REJECTED":
                continue   # review history carried from an earlier plan version: its evidence lived in that version (snapshotted), no operation cites it
            else:
                errs.append(f"{tag}: cites unknown evidence {e}")
        if d.get("type") in EXECUTABLE_TYPES and classes and not classes & set(GROUNDING):
            errs.append(f"{tag}: {d['type']} rests on {sorted(classes)} only (no measured fact or requirement)")
        if classes == {"ai"} and d.get("type") != "REVIEW":
            errs.append(f"{tag}: AI-only evidence must be a REVIEW item")
        # the recorded approval must be the one the engine resolved (basis): an IR whose `approval` was edited from CONFIRM to AUTO
        # after planning is refused instead of executed (ADR-033); an APPROVED status needs the review record that produced it
        res = ((d.get("basis") or {}).get("approval") or {}).get("resolved")
        if res in APPROVALS and d.get("approval") in APPROVALS and APPROVALS.index(d["approval"]) < APPROVALS.index(res):
            errs.append(f"{tag}: approval {d.get('approval')!r} differs from the resolved approval {res!r} in its basis (never lowered after planning)")
        if d.get("status") == "APPROVED" and d.get("approval") == "CONFIRM":
            rec = ((doc.get("execution") or {}).get("reviews") or {}).get(d["id"]) or {}
            if rec.get("action") != "APPROVED":
                errs.append(f"{tag}: status APPROVED without an APPROVED review record")
        if (d.get("approval") == "BLOCK") != (d.get("status") == "BLOCKED"):
            errs.append(f"{tag}: approval {d.get('approval')} with status {d.get('status')} (BLOCK ⇔ BLOCKED)")
        if leak_scan({"params": d.get("params") or {}, "decision": d.get("decision")}):
            errs.append(f"{tag}: params carry executable / credential material")
    cited = [(op.get("type") or f"delivery.{op.get('id')}", did) for op in (doc.get("video") or {}).get("operations", []) + (doc.get("audio") or {}).get("operations", []) + (doc.get("delivery") or {}).get("targets", [])
             for did in op.get("decision_ids") or []]
    cited += [(f"step {s.get('id')}", did) for s in (doc.get("plan") or {}).get("steps") or [] for did in s.get("decision_ids") or []]
    for what, did in cited:
        d = decisions.get(did)
        if d is None:
            continue   # unknown ids are reported by the IR validator
        if d.get("type") not in EXECUTABLE_TYPES:
            errs.append(f"{what} cites decision {did} of type {d.get('type')}; only {EXECUTABLE_TYPES} may be executed")
        # a BLOCKED / REJECTED citation is not an IR error: plan_status / step_status derive BLOCKED / REJECTED from it and the executor refuses
    return errs


def basis_rows(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flat rows of a decision's basis for explain: policy / preference / constraint settings, approval, intent, requirements."""
    b = d.get("basis") or {}
    rows: List[Dict[str, Any]] = []
    for s in b.get("settings") or []:
        rows.append({"kind": (s.get("kind") or "default").lower(), "key": s.get("key"), "value": s.get("value"), "provenance": s.get("provenance"),
                     "source": s.get("source"), "rule_id": s.get("rule_id"), "hard": bool(s.get("hard"))})
    a = b.get("approval") or {}
    rows.append({"kind": "approval", "key": a.get("key"), "value": a.get("resolved", d.get("approval")), "provenance": a.get("provenance"), "notes": a.get("notes") or []})
    it = b.get("intent") or {}
    rows.append({"kind": "intent", "key": it.get("primary"), "value": f"serves {it['served']}" if it.get("served") else "not what was asked (fact-backed / safety decision)", "provenance": it.get("provenance")})
    for r in b.get("requirements") or []:
        rows.append({"kind": "requirement", "key": r.get("key"), "value": r.get("value"), "provenance": r.get("provenance"), "source": r.get("source"), "id": r.get("id")})
    rows.append({"kind": "risk", "key": "risk", "value": d.get("risk"), "provenance": "SYSTEM", "notes": ["set by policy and kind of change, independent of confidence"]})
    return rows
