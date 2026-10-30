"""QC vocabulary (ADR-032): qc-skill as the measurement behind the final promotion gate.

    explicit `qc` requirement → Decision `qc.check` (AUTO: a measurement) → ProductionPlan step `qc_check` per deliverable →
    IR `qa.qc` → compiler → qc-skill `check` (kind delivery | audio | subtitle) → admission (fingerprint == the agent's own sha256
    of the artifact, OBSERVED, schema / skill / kind as requested) → QA items (layer "qc") → artifact stage

The rules a deliverable is checked against derive from the IR only (what the plan promised): the delivered stream layout, the
loudness target of a normalisation, the planned picture width after the edits (`picture_size`), the sidecar's presence. Nothing here measures, and
the agent's own QA checks (duration / streams / loudness / true peak) stay in force — the Skill's verdict is an additional gate,
never a replacement.

ADR-065: with the gate on, the deliverable set of each target (the deliverable + its subtitle sidecar and / or thumbnail) is also checked
together (kind delivery_package: every member present, non-empty, of its planned extension, depending on the deliverable); that verdict
gates every member beside its own report.

Gate: PASS → the candidate becomes READY (artifact stage `approved`); WARN → stays a candidate, its promotion needs the review
the policy key `qc.warn.promotion` asks for (CONFIRM by default; AUTO in a profile promotes WARN too); FAIL → the artifact is
registered `working` and can never be promoted to final (the existing QA FAIL gate); a report that is not admitted is a FAIL.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .finishing import picture_size

QC_TOOL = "qc/check"
QC_SKILL = "qc_check"
KINDS = ("video", "audio", "subtitle", "delivery")
STATUSES = ("PASS", "WARN", "FAIL", "UNKNOWN")
WARN_PROMOTION_KEY = "qc.warn.promotion"
WARN_PROMOTION_DEFAULT = "CONFIRM"


def rules_for_subject(subject: Dict[str, Any], target: Dict[str, Any], doc: Dict[str, Any], tolerance_lu: float) -> Dict[str, Any]:
    """{"kind", "rules"} for one deliverable, from the IR: an audio deliverable is checked as audio; a video deliverable as
    delivery (video + audio, the sidecar when it exists). Expected values come from the plan (loudness target, planned width).
    ADR-058: when the plan names a QC reference (`qa.qc.reference`), a video deliverable's video rule carries its reference_metrics /
    min_* and the result has "reference" (the master's path) for the compiler to pass as the reference_video companion."""
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
    # the planned picture width: the size after the subject's edits in their order (grid → crop → rotate → resize → fit / fill, `picture_size`),
    # stated only when an edit changes the size. Before, the last op carrying a "width" was read, so a quarter turn after a crop / resize /
    # fit (or a rotation alone) expected the unturned width
    vops = (doc.get("video") or {}).get("operations") or []
    resized = any(op.get("asset") == subject["id"] and (op.get("width") or op.get("type") == "video.grid" or (op.get("type") == "video.rotate" and op.get("degrees") in (90, 270)))
                  for op in vops)
    size = picture_size(tech, vops, subject["id"]) if resized else None
    if size and target.get("preset") in (None, ""):
        video_rule["expected_width"] = int(size[0])   # a platform preset may scale; without one the planned width is what is delivered
    reference = ((doc.get("qa") or {}).get("qc") or {}).get("reference") or {}
    if reference.get("path") and (reference.get("rule") or {}).get("reference_metrics"):
        video_rule.update(reference["rule"])
    rules: Dict[str, Any] = {"require_video": True, "require_audio": bool(tech.get("audio")), "audio": audio_rule}
    if video_rule:
        rules["video"] = video_rule
    out: Dict[str, Any] = {"kind": "delivery", "rules": {"delivery": rules}}   # the Skill's rule sections: delivery nests the video / audio / subtitle rules
    if "reference_metrics" in video_rule:
        out["reference"] = str(reference["path"])
    return out


# ADR-065: the deliverable set of one target checked together (qc-skill kind delivery_package). The agent's artifact roles → the
# contract's artifact types; the deliverable is what the rest of the set depends on.
PACKAGE_KIND = "delivery_package"
PACKAGE_TYPES = {"delivery": "video", "audio": "audio", "captions": "subtitle", "thumbnail": "thumbnail"}


def package_rules(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    """{"kind": delivery_package, "rules"} for a deliverable set: `members` are {"artifact_id", "artifact_type", "extension"}, the
    deliverable first. Every member is required, non-empty and of the extension the plan wrote it with; every other member needs the
    deliverable (a sidecar / thumbnail without its video is not a delivery). Durations are not compared: a subtitle's measured duration
    is its last cue's end, not the programme's (the sidecar's own qc check already compares it with the video, ADR-032)."""
    main = members[0]["artifact_id"]
    arts = [{"artifact_id": m["artifact_id"], "artifact_type": m["artifact_type"], "required": True, "min_size_bytes": 1,
             **({"expected_extension": str(m["extension"]).lower().lstrip(".")} if m.get("extension") else {})} for m in members]
    deps = [{"artifact_id": m["artifact_id"], "requires_artifact_id": main} for m in members[1:]]
    return {"kind": PACKAGE_KIND, "rules": {PACKAGE_KIND: {"artifacts": arts, **({"cross_artifact": {"dependencies": deps}} if deps else {})}}}


def admit_package(data: Dict[str, Any], expected: Dict[str, Optional[str]]) -> List[str]:
    """`admit` for a package report: every member's fingerprint as the adapter recorded it equals QA's own sha256 of that file."""
    errs: List[str] = []
    if not isinstance(data, dict) or data.get("admitted") is not True:
        errs.append("report not admitted by the adapter")
    got = (data or {}).get("artifacts") or {}
    for aid, digest in expected.items():
        if not digest or got.get(aid) != digest:
            errs.append(f"package member {aid}: report fingerprint {str(got.get(aid))[:12]} != artifact sha256 {str(digest)[:12]}")
    if (data or {}).get("kind") != PACKAGE_KIND:
        errs.append(f"report kind {(data or {}).get('kind')!r} != requested {PACKAGE_KIND!r}")
    if (data or {}).get("verdict") not in STATUSES:
        errs.append(f"verdict {(data or {}).get('verdict')!r} unknown")
    if ((data or {}).get("provenance") or {}).get("measurement_source") != "OBSERVED":
        errs.append("measurement_source is not OBSERVED")
    return errs


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


def qc_verdict_of(items: List[Any], gated: bool) -> Optional[str]:
    """The QC gate's verdict for one artifact from its QA items: its own report (`verdict`, when the artifact is `gated`: deliverables and
    sidecars, ADR-032) and the report of the deliverable set it belongs to (`delivery_package`, ADR-065), the worse of the two. A gated
    artifact without an admitted report of its own is FAIL (never promoted on silence); an artifact the gate does not cover by itself (a
    thumbnail) takes the package's verdict, or None when it is in no package."""
    own = next((i for i in items if i.layer == "qc" and i.name == "verdict"), None)
    pkg = next((i for i in items if i.layer == "qc" and i.name == PACKAGE_KIND), None)
    verdicts: List[str] = []
    if gated:
        verdicts.append(str(own.observed) if own is not None and str(own.observed) in ("PASS", "WARN", "FAIL") else "FAIL")
    if pkg is not None:
        verdicts.append(str(pkg.observed) if str(pkg.observed) in ("PASS", "WARN", "FAIL") else "FAIL")
    return worst(*verdicts) if verdicts else None


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
