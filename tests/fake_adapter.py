"""FakeAdapter: canned ffmpeg-skill responses so unit tests run without ffmpeg."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from video_agent.models import Operation, ToolResult
from video_agent.tools.base import ToolAdapter


def probe_doc(path: str, duration: float = 16.0, audio: bool = True, vfr: bool = False, hdr: bool = False) -> Dict[str, Any]:
    return {"file": path, "format": "mov,mp4,m4a,3gp,3g2,mj2", "duration": duration, "size_bytes": 1000, "bitrate": 500, "subtitle_streams": 0,
            "video": {"codec": "h264", "width": 1280, "height": 720, "fps": 30.0, "pix_fmt": "yuv420p", "hdr": hdr, "hdr_format": "HLG" if hdr else None, "rotation": 0, "variable_frame_rate_suspected": vfr,
                      "color_primaries": "bt709", "color_transfer": "bt709", "bit_depth": 8},
            "audio": {"codec": "aac", "channels": 2, "sample_rate": 48000} if audio else None}


class FakeAdapter(ToolAdapter):
    name = "ffmpeg-skill"
    version = "0.8.4-fake"

    def __init__(self, duration: float = 16.0, silences: Optional[List[List[Optional[float]]]] = None, lufs: float = -11.0, fail_tools: Optional[Dict[str, int]] = None, audio: bool = True, vfr: bool = False, hdr: bool = False):
        self.duration, self.audio, self.vfr, self.hdr = duration, audio, vfr, hdr
        self.silences = silences if silences is not None else [[0.0, 3.0], [13.7, None]]
        self.lufs = lufs
        self.fail_tools = dict(fail_tools or {})   # tool -> number of failures before success
        self.calls: List[Operation] = []

    def describe(self):
        return {"name": self.name, "version": self.version}

    def supports(self, tool: str) -> bool:
        return tool.startswith("ffmpeg-skill/")

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        return [f"fake {op.tool} {op.args}"]

    def measure(self, tool: str, args: Dict[str, Any], paths=None, timeout=None) -> ToolResult:
        return self.run(Operation(tool=tool, args=args, inputs=[], outputs=[], kind="measure"), paths or {})

    def run(self, op: Operation, paths: Dict[str, str], timeout=None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        self.calls.append(op)
        script = op.tool.split("/")[1]
        if self.fail_tools.get(op.tool, 0) > 0:
            self.fail_tools[op.tool] -= 1
            return ToolResult(op.id, op.tool, False, 1, None, {}, [], "error: command failed (1): ffmpeg\nConversion failed!", 0.1, attempt, dry_run)
        out = paths.get(op.args.get("output", ""), op.args.get("output")) if op.args.get("output") else None
        if out and not dry_run:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"fake")
        if script == "probe":
            return ToolResult(op.id, op.tool, True, 0, None, probe_doc(op.args["inputs"][0], self.duration, self.audio, self.vfr, self.hdr), [], "", 0.1, attempt, dry_run)
        if script == "silence":
            keep = [[max(0.0, (s[1] or self.duration) - 0.15) if i == 0 and s[0] == 0 else 0.0, 0.0] for i, s in enumerate(self.silences[:1])]
            data = {"silences": self.silences, "keep": [[2.85, 13.85]], "input_duration": self.duration, "kept_duration": 11.0, "removed_seconds": 5.0}
            return ToolResult(op.id, op.tool, True, 0, out, data, [], "", 0.1, attempt, dry_run)
        if script == "loudness":
            if op.args.get("measure_only"):
                return ToolResult(op.id, op.tool, True, 0, None, {"input_i": str(self.lufs), "input_tp": "-5.0", "input_lra": "6.0", "input_thresh": "-21", "target_offset": "0"}, [], "", 0.1, attempt, dry_run)
            self.lufs = float(op.args["lufs"])
            return ToolResult(op.id, op.tool, True, 0, out, {"output": out, "commands": ["ffmpeg loudnorm"]}, ["ffmpeg loudnorm"], "", 0.2, attempt, dry_run)
        if script in ("cut", "export", "fit"):
            if script == "cut":
                segs = [tuple(float(x) for x in s.split("-")) for s in op.args["segments"].split(",")]
                self.duration = sum(e - s for s, e in segs)
            return ToolResult(op.id, op.tool, True, 0, out, {"output": out, "commands": [f"ffmpeg {script}"], "probe": probe_doc(out or "", self.duration)}, [f"ffmpeg {script}"], "", 0.2, attempt, dry_run)
        if script == "check":
            rows = [{"check": "duration", "status": "PASS", "value": f"{self.duration}s", "expected": "any", "fix": "", "kind": "judgement"},
                    {"check": "video codec", "status": "PASS", "value": "h264", "expected": "h264", "fix": "", "kind": "format"}]
            return ToolResult(op.id, op.tool, True, 0, None, {"platform": op.args["platform"], "checks": rows, "failed": 0, "warnings": 0, "ok": True}, [], "", 0.1, attempt, dry_run)
        if script == "look":
            if out and not dry_run:
                Path(out).write_bytes(b"png")
            return ToolResult(op.id, op.tool, True, 0, out, {"outputs": [out]}, [], "", 0.1, attempt, dry_run)
        return ToolResult(op.id, op.tool, False, 2, None, {}, [], f"error: unknown tool {script}", 0.0, attempt, dry_run)
