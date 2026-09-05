"""QA after execution: video/audio facts vs expectations, delivery compliance via the selected delivery_check tool,
incidents for FAILs. Rendering success is not production success."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agent.editing import delivery_subjects
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
           check_by_artifact: Optional[Dict[str, ToolResult]] = None) -> QAReport:
    """check_by_artifact maps artifact id -> the check ToolResult the executor already produced, so QA does not measure twice.
    tools: skill → tool id map selected by SkillRegistry for this environment. QA has no default engine; a missing
    measurement skill is an explicit error, never a silent fallback."""
    if tools is None:
        raise TypeError("run_qa needs the skill → tool map resolved by SkillRegistry (tools=None is not allowed)")
    missing = [x for x in REQUIRED_SKILLS if not tools.get(x)]
    if missing:
        raise ToolError("no tool selected for skill(s): " + ", ".join(missing) + " (SkillRegistry.resolve_tools must provide them)")
    rep = QAReport()
    check_by_artifact = check_by_artifact or {}

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
            path = paths.get(art)
            if not path:
                continue
            pr = measure(tools["media_probe"], {"inputs": [path]}, kind="media_probe", artifact=art)
            if not pr.ok:
                rep.items.append(QAItem("video", "probe", "FAIL", pr.stderr_tail, "readable file", artifact=art))
                rep.incidents.append(Incident(type="CORRUPTED_FRAME", severity="HIGH", evidence=[art], possible_cause="output unreadable by the probe tool", recommended_action="re-run the export; inspect the tool log"))
                continue
            p = probe_facts(facts(pr))
            v, a = p.get("video") or {}, p.get("audio") or {}
            if "video" in required:
                got = p.get("duration") or 0.0
                ok = abs(got - kept) <= dur_tol
                rep.items.append(QAItem("video", "duration", "PASS" if ok else "FAIL", round(got, 3), f"{kept:.3f} ± {dur_tol}", kind="judgement", artifact=art,
                                        fix_hint="" if ok else "trim landed on a keyframe or the export trimmed to a platform maximum"))
                if not ok:
                    rep.incidents.append(Incident(type="DURATION_MISMATCH", severity="MEDIUM", start=min(got, kept), end=max(got, kept), evidence=[art],
                                                  possible_cause="output duration differs from the planned kept duration (keyframe snap, platform maximum, or a lost segment)",
                                                  recommended_action="review the trim decision or use frame-accurate cuts"))
                rep.items.append(QAItem("video", "video_stream", "PASS" if v else "FAIL", v.get("codec"), "present", artifact=art))
                sv = (asset.get("technical") or {}).get("video") or {}
                if v and sv:
                    if sv.get("hdr") and not v.get("hdr") and not t.get("preset"):
                        rep.items.append(QAItem("video", "hdr_preserved", "WARN", v.get("hdr_format"), sv.get("hdr_format"), artifact=art, fix_hint="colour flattened; decide HDR vs SDR explicitly"))
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
                            rep.items.append(QAItem("audio", "loudness", "PASS" if ok else "FAIL", lufs, f"{target_lufs:g} ± {lu_tol}", kind="judgement", artifact=art))
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
            if sheet_dir and v:
                sheet = f"{sheet_dir}/{art}_sheet.png"
                if not tools.get("visual_inspection"):
                    continue
                lk = measure(tools["visual_inspection"], {"input": path, "tiles": "4x2", "width": 1280, "output": sheet})
                if lk.ok:
                    rep.sheets.append(sheet)
                    rep.items.append(QAItem("visual", "contact_sheet", "PASS", sheet, "generated for human review", artifact=art))
    return rep


def _incident_type(check: str) -> str:
    return {"aspect": "WRONG_ASPECT", "fps": "WRONG_FPS", "colour": "WRONG_COLOR", "loudness": "LOUDNESS_FAILURE", "true peak": "CLIPPING", "audio": "MISSING_CHANNEL"}.get(check, "DELIVERY_SPEC_FAILURE")


def _f(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
