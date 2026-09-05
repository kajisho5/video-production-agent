"""Test harness for the integrated production pipeline (ADR-031 / ADR-032): a Service whose router carries the fake reference
engine (tests/fake_adapter.py), the fake transcription process and the five fake finishing Skills (subtitle / thumbnail /
color-grading / motion-graphics / qc), with the capabilities their registry skills need. No ffmpeg, no real Skill."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter  # noqa: E402
from video_agent.capabilities.resolver import Capability  # noqa: E402
from video_agent.service import Service  # noqa: E402
from video_agent.tools import ToolRouter  # noqa: E402
from video_agent.tools.color_grading import ColorGradingAdapter  # noqa: E402
from video_agent.tools.motion_graphics import MotionGraphicsAdapter  # noqa: E402
from video_agent.tools.qc import QcAdapter  # noqa: E402
from video_agent.tools.skill_process import CliSkill  # noqa: E402
from video_agent.tools.subtitle import SubtitleAdapter  # noqa: E402
from video_agent.tools.thumbnail import ThumbnailAdapter  # noqa: E402
from video_agent.tools.transcription import TranscriptionAdapter, TranscriptionSkill  # noqa: E402
from video_agent.tools.video_editing import VideoEditingAdapter, VideoEditingSkill  # noqa: E402

HERE = Path(__file__).resolve().parent
FAKES = {"subtitle": HERE / "fake_subtitle.py", "thumbnail": HERE / "fake_thumbnail.py", "color-grading": HERE / "fake_color_grading.py",
         "motion-graphics": HERE / "fake_motion_graphics.py", "qc": HERE / "fake_qc.py", "transcription": HERE / "fake_transcription.py", "video-editing": HERE / "fake_video_editing.py"}
MODE_VARS = ("FAKE_SUBTITLE_MODE", "FAKE_THUMBNAIL_MODE", "FAKE_CG_MODE", "FAKE_MG_MODE", "FAKE_QC_MODE", "FAKE_TS_MODE", "FAKE_TS_SEGMENTS", "FAKE_VE_MODE",
             "FAKE_SUBTITLE_CALLS", "FAKE_THUMBNAIL_CALLS", "FAKE_CG_CALLS", "FAKE_MG_CALLS", "FAKE_QC_CALLS", "FAKE_TS_CALLS", "FAKE_VE_CALLS")
BASE_CAPS = ["python", "ffmpeg", "ffprobe", "ffmpeg-skill", "encoder:libx264", "encoder:libx265", "encoder:prores_ks", "encoder:aac", "filter:loudnorm", "filter:subtitles", "font:cjk-ja"]
FINISHING_CAPS = ["subtitle", "thumbnail", "color-grading", "motion-graphics", "qc", "transcription", "video-editing", "filter:xfade", "filter:acrossfade"] \
    + [f"color-grading:{t}" for t in ("HDR_TO_SDR", "LUT_APPLY", "RETAG", "STRIP_DOVI")] + [f"motion-graphics:{t}" for t in ("title", "lower_third", "text_overlay", "image_overlay")]


class PipelineCaps:
    """AVAILABLE for everything the pipeline needs unless named in `missing`; `unknown` names come back UNKNOWN (never selectable)."""

    def __init__(self, missing=(), unknown=(), extra=()):
        self.missing, self.unknown, self.extra = set(missing), set(unknown), list(extra)

    def resolve(self, refresh=False):
        out = {}
        for n in BASE_CAPS + FINISHING_CAPS + self.extra:
            st = "MISSING" if n in self.missing else ("UNKNOWN" if n in self.unknown else "AVAILABLE")
            out[n] = Capability(n, st, "fake", {"version": "0.8.4-fake"} if n == "ffmpeg-skill" else {})
        return out


def clear_modes() -> None:
    for k in MODE_VARS:
        os.environ.pop(k, None)


def fake_media(tmp: str, name: str = "talk.mp4", duration: float = 16.0, video: bool = True, channels: int = 2, hdr: bool = False, lufs: float = -11.0) -> str:
    """A self-describing fake media file every fake process and the fake engine probe consistently."""
    p = Path(tmp) / "src" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(json.dumps({"fake": True, "duration": duration, "lufs": lufs, "video": video, "channels": channels, "hdr": hdr, "name": name}).encode())
    return str(p)


def pipeline_service(tmp: str, caps: Optional[Any] = None, engine: Optional[FakeAdapter] = None, transcription: bool = True, skills: Optional[List[str]] = None,
                     allowed: Optional[List[str]] = None, **engine_kw) -> Service:
    """A Service with the fake engine + fake Skills. `skills` restricts which finishing fakes are registered (default: all five)."""
    ws = str(Path(tmp) / "ws")
    os.makedirs(ws, exist_ok=True)
    roots = list(allowed or [str(Path(tmp) / "src")]) + [ws]
    adapters: List[Any] = [engine or FakeAdapter(**engine_kw)]
    if transcription:
        adapters.append(TranscriptionAdapter(TranscriptionSkill([sys.executable, str(FAKES["transcription"])], None, {}), workspace=str(Path(ws) / "cache" / "transcription")))
    want = skills if skills is not None else ["subtitle", "thumbnail", "color-grading", "motion-graphics", "qc", "video-editing"]

    def cli(name: str) -> CliSkill:
        return CliSkill(name, [sys.executable, str(FAKES[name])], None, {})
    if "subtitle" in want:
        adapters.append(SubtitleAdapter(cli("subtitle"), workspace=ws, allowed_inputs=roots, ffmpeg_skill_dir=tmp))
    if "thumbnail" in want:
        adapters.append(ThumbnailAdapter(cli("thumbnail"), workspace=ws, allowed_inputs=roots, ffmpeg_skill_dir=tmp))
    if "color-grading" in want:
        adapters.append(ColorGradingAdapter(cli("color-grading"), workspace=ws, allowed_inputs=roots, ffmpeg_skill_dir=tmp))
    if "motion-graphics" in want:
        adapters.append(MotionGraphicsAdapter(cli("motion-graphics"), workspace=ws, allowed_inputs=roots, ffmpeg_skill_dir=tmp))
    if "qc" in want:
        adapters.append(QcAdapter(cli("qc"), workspace=ws, allowed_inputs=roots))
    if "video-editing" in want:
        adapters.append(VideoEditingAdapter(VideoEditingSkill([sys.executable, str(FAKES["video-editing"])], None, {}), workspace=ws, allowed_inputs=roots, ffmpeg_skill_dir=tmp))
    return Service(workspace=ws, adapter=ToolRouter(adapters), caps=caps or PipelineCaps())


def plan_and_render(svc: Service, inputs: List[str], reqs: Dict[str, Any], profile: str = "youtube", kinds: Optional[List[str]] = None, approve: bool = True,
                    name: str = "p", **plan_kw) -> Dict[str, Any]:
    """plan → validate → save → render (approving every CONFIRM) → {"ir", "path", "out", "validation"}."""
    from video_agent.project import load_ir, save_ir
    ir = svc.plan(inputs, profile, user_requirements=reqs, kinds=kinds, params={"language": "ja"}, **plan_kw)
    rep = svc.validate(ir)
    path = str(Path(svc.workspace) / "plans" / f"{name}.project.json")
    save_ir(ir, path)
    out = svc.render(load_ir(path), path, approve=["all"] if approve else None)
    return {"ir": load_ir(path), "path": path, "out": out, "validation": rep}
