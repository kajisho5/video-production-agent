"""QC vocabulary (ADR-032): qc-skill as the measurement behind the final promotion gate.

    explicit `qc` requirement → Decision `qc.check` (AUTO: a measurement) → ProductionPlan step `qc_check` per deliverable →
    IR `qa.qc` → compiler → qc-skill `check` (kind delivery | audio | subtitle) → admission (fingerprint == the agent's own sha256
    of the artifact, OBSERVED, schema / skill / kind as requested) → QA items (layer "qc") → artifact stage

The rules a deliverable is checked against derive from the IR only (what the plan promised): the delivered stream layout, the
loudness target of a normalisation, the planned picture size after a resize, the sidecar's presence. Nothing here measures, and
the agent's own QA checks (duration / streams / loudness / true peak) stay in force — the Skill's verdict is an additional gate,
never a replacement.

Gate: PASS → the candidate becomes READY (artifact stage `approved`); WARN → stays a candidate, its promotion needs the review
the policy key `qc.warn.promotion` asks for (CONFIRM by default; AUTO in a profile promotes WARN too); FAIL → the artifact is
registered `working` and can never be promoted to final (the existing QA FAIL gate); a report that is not admitted is a FAIL.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

QC_TOOL = "qc/check"
QC_SKILL = "qc_check"
KINDS = ("video", "audio", "subtitle", "delivery")
STATUSES = ("PASS", "WARN", "FAIL", "UNKNOWN")
WARN_PROMOTION_KEY = "qc.warn.promotion"
WARN_PROMOTION_DEFAULT = "CONFIRM"


def rules_for_subject(subject: Dict[str, Any], target: Dict[str, Any], doc: Dict[str, Any], tolerance_lu: float) -> Dict[str, Any]:
    """{"kind", "rules"} for one deliverable, from the IR: an audio deliverable is checked as audio; a video deliverable as
    delivery (video + audio, the sidecar when it exists). Expected values come from the plan (loudness target, planned width)."""
    tech = subject.get("technical") or {}
    audio_rule: Dict[str, Any] = {"require_audio_stream": bool(tech.get("audio"))}
    if subject.get("target_lufs") is not None:
        audio_rule["integrated_loudness_target_lufs"] = float(subject["target_lufs"])
        audio_rule["integrated_loudness_tolerance_lu"] = float(tolerance_lu)
        tp = next((op.get("true_peak") for op in (doc.get("audio") or {}).get("operations") or [] if op.get("asset") == subject["id"] and op.get("type") == "audio.loudness"), None)
        if tp is not None:
            audio_rule["max_true_peak_dbfs"] = float(tp)
    if subject.get("audio_only"):
        if subject.get("technical", {}).get("audio", {}).get("channels"):
            audio_rule["expected_channels"] = int(subject["technical"]["audio"]["channels"])
        return {"kind": "audio", "rules": {"audio": audio_rule}}
    video_rule: Dict[str, Any] = {}
    width = next((op.get("width") for op in reversed((doc.get("video") or {}).get("operations") or []) if op.get("asset") == subject["id"] and op.get("width")), None)
    if width and target.get("preset") in (None, ""):
        video_rule["expected_width"] = int(width)   # a platform preset may scale; without one the planned width is what is delivered
    rules: Dict[str, Any] = {"require_video": True, "require_audio": bool(tech.get("audio")), "audio": audio_rule}
    if video_rule:
        rules["video"] = video_rule
    return {"kind": "delivery", "rules": {"delivery": rules}}   # the Skill's rule sections: delivery nests the video / audio / subtitle rules


def sidecar_rules() -> Dict[str, Any]:
    return {"kind": "subtitle", "rules": {"subtitle": {"require_subtitle": True}}}


def admit(data: Dict[str, Any], expected_sha256: Optional[str], expected_kind: str) -> List[str]:
    """Why a qc result must not gate promotion (empty = admitted): the adapter already verified the response; QA re-checks the
    fingerprint against the hash it computed itself and the kind it asked for, so a report about another file never counts."""
    errs: List[str] = []
    if not isinstance(data, dict) or data.get("admitted") is not True:
        errs.append("report not admitted by the adapter")
    if expected_sha256 and str(data.get("fingerprint") or "") != expected_sha256:
        errs.append(f"report fingerprint {str(data.get('fingerprint'))[:12]} != artifact sha256 {expected_sha256[:12]}")
    if data.get("kind") != expected_kind:
        errs.append(f"report kind {data.get('kind')!r} != requested {expected_kind!r}")
    if data.get("verdict") not in STATUSES:
        errs.append(f"verdict {data.get('verdict')!r} unknown")
    if data.get("provenance", {}).get("measurement_source") != "OBSERVED":
        errs.append("measurement_source is not OBSERVED")
    return errs


def worst(*statuses: str) -> str:
    order = {"FAIL": 3, "WARN": 2, "PASS": 1, "UNKNOWN": 0, "SKIP": 0}
    return max(statuses, key=lambda s: order.get(s, 0)) if statuses else "UNKNOWN"


def stage_for(qa_status: str, qc_verdict: Optional[str], warn_promotion: str) -> str:
    """Artifact stage at registration: FAIL anywhere → working; QC PASS with agent QA PASS / WARN → approved (READY); WARN → approved
    only when the policy says AUTO, else candidate; no QC → candidate (the existing rule)."""
    if qa_status == "FAIL" or qc_verdict == "FAIL":
        return "working"
    if qc_verdict is None or qc_verdict == "UNKNOWN":
        return "candidate"
    if qc_verdict == "PASS" and qa_status in ("PASS", "WARN"):
        return "approved"
    if qc_verdict == "WARN" and warn_promotion == "AUTO":
        return "approved"
    return "candidate"
