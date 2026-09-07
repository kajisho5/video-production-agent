"""QA after execution: video/audio facts vs expectations, delivery compliance via the selected delivery_check tool,
incidents for FAILs. Rendering success is not production success."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional

from ..agent.editing import delivery_subjects
from ..agent.qc import admit
from ..artifacts import artifact_id
from ..media.analysis import loudness_facts, probe_facts
from ..models import Incident, ToolResult
from ..tools.base import ToolAdapter, ToolError


@dataclass
class QAItem:
    layer: str          # video | audio | delivery | visual
    name: str
    status: str         # PASS | WARN | FAIL | SKIP
    observed: Any
    expected: Any
    kind: str = "format"
    fix_hint: str = ""
    artifact: str = ""
    # ---- QC_ARCHITECTURE.md §5 (ADR-039, Phase 5): `threshold_source` says whether `expected` came from this
    # Plan's own declared intent (kept ranges / concat timeline / speed factor for duration, the plan's own
    # loudness-normalization target) or from a fixed, context-free rule (everything else here) -- the doc's own
    # framing, not a new kind of question, just the existing threshold's provenance made visible on the finding.
    # `subject_artifact_id` names which registered Artifact this item verified (computed the same way
    # `artifacts.artifact_id()` will when `_register_artifacts()` runs, so it resolves even though QA runs first).
    threshold_source: str = "rule"    # "plan" | "rule"
    subject_artifact_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class QAReport:
    items: List[QAItem] = field(default_factory=list)
    incidents: List[Incident] = field(default_factory=list)
    sheets: List[str] = field(default_factory=list)
    measurements: List[Dict[str, Any]] = field(default_factory=list)   # every tool call QA made, for provenance

    @property
    def status(self) -> str:
        st = {i.status for i in self.items}
        return "FAIL" if "FAIL" in st else ("WARN" if "WARN" in st else "PASS")

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "items": [i.to_dict() for i in self.items], "incidents": [i.to_dict() for i in self.incidents], "sheets": self.sheets, "measurements": self.measurements}


REQUIRED_SKILLS = ("media_probe", "loudness_analysis", "delivery_check")   # visual_inspection is optional (contact sheet only)


def run_qa(adapter: ToolAdapter, ir_doc: Dict[str, Any], paths: Dict[str, str], results: List[ToolResult], tools: Dict[str, str], sheet_dir: Optional[str] = None,
           check_by_artifact: Optional[Dict[str, ToolResult]] = None, qc_by_artifact: Optional[Dict[str, ToolResult]] = None) -> QAReport:
    """check_by_artifact maps artifact id -> the check ToolResult the executor already produced, so QA does not measure twice.
    qc_by_artifact maps artifact id -> the qc/check ToolResult of the QC gate (ADR-032): admitted only when its fingerprint is the
    sha256 QA computes itself; its verdict is an additional gate beside the agent's own checks, never a replacement.
    tools: skill → tool id map selected by SkillRegistry for this environment. QA has no default engine; a missing
    measurement skill is an explicit error, never a silent fallback."""
    if tools is None:
        raise TypeError("run_qa needs the skill → tool map resolved by SkillRegistry (tools=None is not allowed)")
    missing = [x for x in REQUIRED_SKILLS if not tools.get(x)]
    if missing:
        raise ToolError("no tool selected for skill(s): " + ", ".join(missing) + " (SkillRegistry.resolve_tools must provide them)")
    rep = QAReport()
    check_by_artifact = check_by_artifact or {}
    qc_by_artifact = qc_by_artifact or {}
    qc_enabled = bool(((ir_doc.get("qa") or {}).get("qc") or {}).get("enabled"))

    def qc_items(art: str, path: str, expected_kind: str, qc_key: Optional[str] = None) -> None:
        """The QC gate's verdict for an artifact as QA items (layer qc): admission first (fingerprint == QA's own sha256 of the file,
        the kind QA asked for, OBSERVED), then the verdict and every failing / warning check by name. `qc_key` is the id the qc op was
        actually compiled against when it differs from `art` (a no-preset target gates the subject's own media directly, ADR-032)."""
        if not qc_enabled:
            return
        r = qc_by_artifact.get(qc_key or art)
        if r is None:
            rep.items.append(QAItem("qc", "verdict", "FAIL", "no report", "an admitted qc report", kind="judgement", artifact=art, fix_hint="the QC gate was planned but no report exists for this artifact"))
            return
        digest = _sha256(path)
        data = r.data if isinstance(r.data, dict) else {}
        if not r.ok:
            err = (data.get("error") or {})
            rep.items.append(QAItem("qc", "verdict", "FAIL", f"{err.get('code')}: {err.get('message')}", "an admitted qc report", kind="judgement", artifact=art))
            return
        problems = admit(data, digest, expected_kind)
        if problems:
            rep.items.append(QAItem("qc", "verdict", "FAIL", "; ".join(problems), "an admitted qc report", kind="judgement", artifact=art, fix_hint="the report does not describe this file as QA measured it"))
            return
        verdict = str(data.get("verdict"))
        rep.items.append(QAItem("qc", "verdict", "PASS" if verdict == "PASS" else ("WARN" if verdict == "WARN" else "FAIL"), verdict, "PASS", kind="judgement", artifact=art,
                                fix_hint="" if verdict == "PASS" else "see the qc findings"))
        for c in data.get("checks") or []:
            if c.get("status") in ("WARN", "FAIL"):
                rep.items.append(QAItem("qc", str(c.get("check_id")), str(c["status"]), ", ".join(str(x) for x in c.get("finding_codes") or []) or c["status"], "PASS", kind="judgement", artifact=art))
        if verdict == "FAIL":
            rep.incidents.append(Incident(type="DELIVERY_SPEC_FAILURE", severity="HIGH", evidence=[art], possible_cause="qc-skill reports a failing check", recommended_action="review the qc findings; the artifact is not deliverable"))

    def measure(tool: str, args: Dict[str, Any], kind: Optional[str] = None, artifact: Optional[str] = None) -> ToolResult:
        """One measurement through the adapter boundary. A measurement Skill that shapes its own request (media-analysis:
        asset id / kind / declared parameters) gets the typed arguments through `measurement_args`; QA never builds argv."""
        shaped = None
        if kind and artifact and hasattr(adapter, "measurement_args"):
            path = args.get("input") or (args.get("inputs") or [None])[0]
            shaped = adapter.measurement_args(tool, kind, path, artifact, {}, f"qa-{artifact}", "use")
        r = adapter.measure(tool, shaped if shaped is not None else args)
        rep.measurements.append({"tool": tool, "args": shaped if shaped is not None else args, "ok": r.ok, "exit_code": r.exit_code, "seconds": r.seconds, "commands": r.commands})
        return r

    def facts(r: ToolResult) -> Dict[str, Any]:
        """The measured facts: an external Skill's observation data, or the engine's result data."""
        d = r.data if isinstance(r.data, dict) else {}
        o = d.get("observation")
        return dict(o.get("data") or {}) if isinstance(o, dict) and isinstance(o.get("data"), dict) else d
    th = ir_doc["qa"].get("thresholds") or {}
    dur_tol = float(th.get("duration_tolerance_s", 0.5))
    lu_tol = float(th.get("loudness_tolerance_lu", 2.0))
    required = set(ir_doc["qa"]["required"])
    for subject in delivery_subjects(ir_doc):   # each asset, or the concat programme (ADR-029) whose expectations derive from the IR chain
        asset_id, asset = subject["id"], {"technical": subject["technical"]}
        kept = subject["duration"]        # expected output duration: kept ranges → concat timeline → speed factor (from the IR, not measured)
        target_lufs = subject["target_lufs"]
        for t in ir_doc["delivery"]["targets"]:
            art = f"{asset_id}_delivery_{t['id']}"
            # a no-preset target has no alias path unless something upstream changed the subject; when the
            # user asked for the QC gate specifically, `execution/compiler.py`'s `qc_gate()` still measured
            # the subject's own (unchanged) media directly, so QA must look there too or a real, already-admitted
            # qc report would go unsurfaced. Outside of qc=true this stays exactly as narrow as before — no
            # extra probing of a deliverable nobody asked to verify.
            qc_key = art if art in paths else (asset_id if not t.get("preset") and qc_enabled else None)
            path = paths.get(art) or (paths.get(asset_id) if not t.get("preset") and qc_enabled else None)
            if not path:
                continue
            start_idx = len(rep.items)   # QC_ARCHITECTURE.md §5.1: every item below verifies this one subject artifact
            try:   # a missing / unreadable path is the probe's own FAIL item below to report, never a crash here
                subject_artifact_id = artifact_id(ir_doc["project"]["id"], ir_doc["plan"].get("id", ""), art, _sha256(path))
            except OSError:
                subject_artifact_id = ""
            pr = measure(tools["media_probe"], {"inputs": [path]}, kind="media_probe", artifact=art)
            if not pr.ok:
                rep.items.append(QAItem("video", "probe", "FAIL", pr.stderr_tail, "readable file", artifact=art, subject_artifact_id=subject_artifact_id))
                rep.incidents.append(Incident(type="CORRUPTED_FRAME", severity="HIGH", evidence=[art], possible_cause="output unreadable by the probe tool", recommended_action="re-run the export; inspect the tool log"))
                continue
            p = probe_facts(facts(pr))
            v, a = p.get("video") or {}, p.get("audio") or {}
            if "video" in required and subject.get("audio_only"):
                got = p.get("duration") or 0.0   # an audio deliverable: duration is checked, the picture facts are not expected
                ok = abs(got - kept) <= dur_tol
                rep.items.append(QAItem("video", "duration", "PASS" if ok else "FAIL", round(got, 3), f"{kept:.3f} ± {dur_tol}", kind="judgement", artifact=art,
                                        fix_hint="" if ok else "the audio cut landed differently from the planned ranges", threshold_source="plan"))
                if not ok:
                    rep.incidents.append(Incident(type="DURATION_MISMATCH", severity="MEDIUM", start=min(got, kept), end=max(got, kept), evidence=[art],
                                                  possible_cause="output duration differs from the planned kept duration", recommended_action="review the cut decision"))
                rep.items.append(QAItem("video", "audio_only", "PASS" if not v else "FAIL", v.get("codec") if v else "none", "no video stream (audio deliverable)", artifact=art))
            elif "video" in required:
                got = p.get("duration") or 0.0
                ok = abs(got - kept) <= dur_tol
                rep.items.append(QAItem("video", "duration", "PASS" if ok else "FAIL", round(got, 3), f"{kept:.3f} ± {dur_tol}", kind="judgement", artifact=art,
                                        fix_hint="" if ok else "trim landed on a keyframe or the export trimmed to a platform maximum", threshold_source="plan"))
                if not ok:
                    rep.incidents.append(Incident(type="DURATION_MISMATCH", severity="MEDIUM", start=min(got, kept), end=max(got, kept), evidence=[art],
                                                  possible_cause="output duration differs from the planned kept duration (keyframe snap, platform maximum, or a lost segment)",
                                                  recommended_action="review the trim decision or use frame-accurate cuts"))
                rep.items.append(QAItem("video", "video_stream", "PASS" if v else "FAIL", v.get("codec"), "present", artifact=art))
                sv = (asset.get("technical") or {}).get("video") or {}
                if v and sv:
                    tone_mapped = any(op.get("asset") == asset_id and op.get("type") == "color.hdr_to_sdr" for op in (ir_doc.get("color") or {}).get("operations") or [])
                    if sv.get("hdr") and not v.get("hdr") and not t.get("preset") and not tone_mapped:
                        rep.items.append(QAItem("video", "hdr_preserved", "WARN", v.get("hdr_format"), sv.get("hdr_format"), artifact=art, fix_hint="colour flattened; decide HDR vs SDR explicitly"))
                    elif sv.get("hdr") and tone_mapped:
                        rep.items.append(QAItem("video", "sdr", "PASS" if not v.get("hdr") else "FAIL", v.get("hdr_format") or "SDR", "SDR (colour decision)", kind="judgement", artifact=art))
                    if v.get("variable_frame_rate_suspected"):
                        rep.items.append(QAItem("video", "cfr", "WARN", "VFR suspected", "constant frame rate", artifact=art))
                    fps_src, fps_out = sv.get("fps") or 0, v.get("fps") or 0
                    if fps_src and fps_out and fps_out < fps_src - 0.5:
                        rep.items.append(QAItem("video", "fps", "WARN", fps_out, f"{fps_src} (source)", kind="judgement", artifact=art, fix_hint="preset reduced the frame rate"))
            if "audio" in required and (asset.get("technical") or {}).get("audio"):
                rep.items.append(QAItem("audio", "audio_stream", "PASS" if a else "FAIL", a.get("codec"), "present", artifact=art))
                if not a:
                    rep.incidents.append(Incident(type="MISSING_CHANNEL", severity="HIGH", evidence=[art], possible_cause="audio stream lost in processing", recommended_action="check the export preset and the source audio track"))
                if a:
                    sa = (asset.get("technical") or {}).get("audio") or {}
                    if sa.get("channels") and a.get("channels") and a["channels"] < min(2, sa["channels"]):
                        rep.items.append(QAItem("audio", "channels", "WARN", a["channels"], sa["channels"], artifact=art))
                    m = measure(tools["loudness_analysis"], {"input": path, "measure_only": True}, kind="loudness", artifact=art)
                    lf = loudness_facts(facts(m)) if m.ok else {}
                    if m.ok and not lf["silent"]:
                        lufs, tp = lf["lufs"], lf["true_peak"]
                        if target_lufs is not None and lufs is not None:
                            ok = abs(lufs - target_lufs) <= lu_tol
                            rep.items.append(QAItem("audio", "loudness", "PASS" if ok else "FAIL", lufs, f"{target_lufs:g} ± {lu_tol}", kind="judgement", artifact=art, threshold_source="plan"))
                            if not ok:
                                rep.incidents.append(Incident(type="LOUDNESS_FAILURE", severity="MEDIUM", evidence=[art], possible_cause="export re-encoded audio after normalisation or the platform trimmed the file",
                                                              recommended_action="normalise after export or check the preset's audio chain"))
                        else:
                            rep.items.append(QAItem("audio", "loudness", "PASS", lufs, "measured only", artifact=art))
                        if tp is not None and tp > 0:
                            rep.items.append(QAItem("audio", "true_peak", "FAIL", tp, "<= 0 dBTP", artifact=art))
                            rep.incidents.append(Incident(type="CLIPPING", severity="HIGH", evidence=[art], possible_cause="true peak above 0 dBTP", recommended_action="apply loudness normalisation with a true-peak ceiling"))
                    elif m.ok:
                        rep.items.append(QAItem("audio", "silence", "WARN", "silent", "programme audio", artifact=art))
                        rep.incidents.append(Incident(type="UNEXPECTED_SILENCE", severity="HIGH", start=0.0, end=p.get("duration"), evidence=[art], possible_cause="output audio is silent", recommended_action="verify the source track and the audio mapping"))
            if "delivery" in required and t.get("preset"):
                cr = check_by_artifact.get(art)
                if cr is None or not cr.data.get("checks"):
                    cr = measure(tools["delivery_check"], {"input": path, "platform": t.get("platform", "custom")})
                if cr and cr.data.get("checks"):
                    for row in cr.data["checks"]:
                        rep.items.append(QAItem("delivery", row["check"], row["status"], row["value"], row["expected"], kind=row.get("kind", "format"), fix_hint=row.get("fix", ""), artifact=art))
                        if row["status"] == "FAIL":
                            rep.incidents.append(Incident(type=_incident_type(row["check"]), severity="MEDIUM" if row.get("kind") == "judgement" else "LOW", evidence=[art],
                                                          possible_cause=f"{row['check']}={row['value']} vs {row['expected']}", recommended_action=row.get("fix", "")))
                else:
                    rep.items.append(QAItem("delivery", "check", "WARN", "no result", "check.py output", artifact=art))
            qc_items(art, path, "audio" if subject.get("audio_only") else "delivery", qc_key=qc_key)
            for i in rep.items[start_idx:]:
                i.subject_artifact_id = subject_artifact_id
            if sheet_dir and v:
                sheet = f"{sheet_dir}/{art}_sheet.png"
                if not tools.get("visual_inspection"):
                    continue
                lk = measure(tools["visual_inspection"], {"input": path, "tiles": "4x2", "width": 1280, "output": sheet})
                if lk.ok:
                    rep.sheets.append(sheet)
                    rep.items.append(QAItem("visual", "contact_sheet", "PASS", sheet, "generated for human review", artifact=art))
        # finishing outputs of the subject (ADR-031): the subtitle sidecar (cue count as planned, well-formed) and the thumbnail (a non-empty image of the planned format)
        for o in (ir_doc.get("plan") or {}).get("outputs") or []:
            if (o.get("expected") or {}).get("source") != asset_id or o.get("role") not in ("CAPTIONS", "THUMBNAIL"):
                continue
            art = o["logical"]
            path = paths.get(art)
            if not path or not os.path.isfile(path):
                rep.items.append(QAItem("delivery", "exists", "FAIL", "missing", o["role"].lower() + " file", artifact=art))
                continue
            size = os.path.getsize(path)
            if o["role"] == "CAPTIONS":
                text = _read_text(path)
                n = _cue_count(text, o.get("format") or "")
                want = int((o.get("expected") or {}).get("cues") or 0)
                rep.items.append(QAItem("delivery", "cues", "PASS" if n == want else "FAIL", n, want, kind="judgement", artifact=art, fix_hint="" if n == want else "the sidecar does not carry the planned cues"))
                rep.items.append(QAItem("delivery", "format", "PASS" if _looks_like(text, o.get("format") or "") else "FAIL", o.get("format"), "well-formed " + str(o.get("format")), artifact=art))
                qc_items(art, path, "subtitle")
            else:
                ok = size > 0 and _image_magic(path, o.get("format") or "")
                rep.items.append(QAItem("delivery", "image", "PASS" if ok else "FAIL", f"{size} bytes", f"non-empty {o.get('format')}", artifact=art))
    return rep


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text(path: str) -> str:
    try:
        return open(path, "r", encoding="utf-8-sig", errors="replace").read()
    except OSError:
        return ""


def _cue_count(text: str, fmt: str) -> int:
    import re
    return len(re.findall(r"^\s*\d{2}:\d{2}:\d{2}[,.]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[,.]\d{3}", text, flags=re.M))


def _looks_like(text: str, fmt: str) -> bool:
    if fmt == "vtt":
        return text.lstrip().startswith("WEBVTT")
    return bool(text.strip()) and "-->" in text and not text.lstrip().startswith("WEBVTT")


def _image_magic(path: str, fmt: str) -> bool:
    try:
        head = open(path, "rb").read(8)
    except OSError:
        return False
    return head.startswith(b"\x89PNG") if fmt == "png" else head[:2] == b"\xff\xd8"


def _incident_type(check: str) -> str:
    return {"aspect": "WRONG_ASPECT", "fps": "WRONG_FPS", "colour": "WRONG_COLOR", "loudness": "LOUDNESS_FAILURE", "true peak": "CLIPPING", "audio": "MISSING_CHANNEL"}.get(check, "DELIVERY_SPEC_FAILURE")


def _f(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
