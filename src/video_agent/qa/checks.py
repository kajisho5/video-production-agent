"""QA after execution: video/audio facts vs expectations, delivery compliance via ffmpeg-skill check.py,
incidents for FAILs. Rendering success is not production success."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models import Incident, ToolResult
from ..tools.base import ToolAdapter


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


def run_qa(adapter: ToolAdapter, ir_doc: Dict[str, Any], paths: Dict[str, str], results: List[ToolResult], sheet_dir: Optional[str] = None,
           check_by_artifact: Optional[Dict[str, ToolResult]] = None) -> QAReport:
    """check_by_artifact maps artifact id -> the check.py ToolResult the executor already produced, so QA does not measure twice."""
    rep = QAReport()
    check_by_artifact = check_by_artifact or {}

    def measure(tool: str, args: Dict[str, Any]) -> ToolResult:
        r = adapter.measure(tool, args)
        rep.measurements.append({"tool": tool, "args": args, "ok": r.ok, "exit_code": r.exit_code, "seconds": r.seconds, "commands": r.commands})
        return r
    th = ir_doc["qa"].get("thresholds") or {}
    dur_tol = float(th.get("duration_tolerance_s", 0.5))
    lu_tol = float(th.get("loudness_tolerance_lu", 2.0))
    required = set(ir_doc["qa"]["required"])
    for asset_id, asset in ir_doc["assets"].items():
        src_dur = (asset.get("technical") or {}).get("duration") or 0.0
        kept = src_dur
        for op in ir_doc["video"]["operations"]:
            if op["asset"] == asset_id and op["type"] == "video.trim":
                kept = sum(e - s for s, e in op["keep"])
        target_lufs = next((op["target_lufs"] for op in ir_doc["audio"]["operations"] if op["asset"] == asset_id), None)
        for t in ir_doc["delivery"]["targets"]:
            art = f"{asset_id}_delivery_{t['id']}"
            path = paths.get(art)
            if not path:
                continue
            pr = measure("ffmpeg-skill/probe", {"inputs": [path]})
            if not pr.ok:
                rep.items.append(QAItem("video", "probe", "FAIL", pr.stderr_tail, "readable file", artifact=art))
                rep.incidents.append(Incident(type="CORRUPTED_FRAME", severity="HIGH", evidence=[art], possible_cause="output unreadable by ffprobe", recommended_action="re-run the export; inspect the tool log"))
                continue
            p = pr.data
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
                    m = measure("ffmpeg-skill/loudness", {"input": path, "measure_only": True})
                    if m.ok and not m.data.get("silent"):
                        lufs, tp = _f(m.data.get("input_i")), _f(m.data.get("input_tp"))
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
                    cr = measure("ffmpeg-skill/check", {"input": path, "platform": t.get("platform", "custom")})
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
                lk = measure("ffmpeg-skill/look", {"input": path, "tiles": "4x2", "width": 1280, "output": sheet})
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
