"""FakeAdapter: canned ffmpeg-skill responses so unit tests run without ffmpeg."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from video_agent.models import Operation, ToolResult
from video_agent.skills.contract import SkillPackage, ToolSpec
from video_agent.tools.base import ToolAdapter
from video_agent.tools.ffmpeg_skill.catalog import CATALOG


def probe_doc(path: str, duration: float = 16.0, audio: bool = True, vfr: bool = False, hdr: bool = False) -> Dict[str, Any]:
    return {"file": path, "format": "mov,mp4,m4a,3gp,3g2,mj2", "duration": duration, "size_bytes": 1000, "bitrate": 500, "subtitle_streams": 0,
            "video": {"codec": "h264", "width": 1280, "height": 720, "fps": 30.0, "pix_fmt": "yuv420p", "hdr": hdr, "hdr_format": "HLG" if hdr else None, "rotation": 0, "variable_frame_rate_suspected": vfr,
                      "color_primaries": "bt709", "color_transfer": "bt709", "bit_depth": 8},
            "audio": {"codec": "aac", "channels": 2, "sample_rate": 48000} if audio else None}


def _read_fake(path: str) -> Optional[Dict[str, Any]]:
    """Fake outputs carry their own metadata so a fresh adapter instance (e.g. on resume) probes them consistently."""
    try:
        import json
        raw = Path(path).read_bytes()
        if raw.startswith(b'{"fake"'):
            return json.loads(raw.decode())
    except (OSError, ValueError):
        return None
    return None


def _write_fake(path: str, meta: Dict[str, Any]) -> None:
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(json.dumps({"fake": True, **meta}).encode())


class FakeAdapter(ToolAdapter):
    name = "ffmpeg-skill"
    version = "0.8.4-fake"
    ALIASES: Dict[str, str] = {}   # subclasses standing in for another engine map their tool names onto the fake scripts

    def __init__(self, duration: float = 16.0, silences: Optional[List[List[Optional[float]]]] = None, lufs: float = -11.0, fail_tools: Optional[Dict[str, int]] = None, audio: bool = True, vfr: bool = False, hdr: bool = False,
                 true_peak: float = -5.0):
        self.duration, self.audio, self.vfr, self.hdr = duration, audio, vfr, hdr
        self.true_peak = true_peak   # reported by loudness --measure-only; > 0 dBTP makes QA FAIL (clipping)
        self.silences = silences if silences is not None else [[0.0, 3.0], [13.7, None]]
        self.lufs = lufs
        self.fail_tools = dict(fail_tools or {})   # tool -> number of failures before success
        self.calls: List[Operation] = []

    def describe(self):
        return {"name": self.name, "version": self.version}

    TOOLS: Optional[List[str]] = None   # tool names this fake package declares; default: the ffmpeg-skill catalog plus aliases

    def package(self) -> SkillPackage:
        names = self.TOOLS if self.TOOLS is not None else list(CATALOG) + list(self.ALIASES)
        return SkillPackage(skill_id=self.name, name=self.name, version=self.version, description="fake engine (tests only)",
                            capabilities=[], tools=[ToolSpec(tool_id=f"{self.name}/{n}", skill_id=self.name, version=self.version, produces_output=n not in ("probe", "check", "silence")) for n in names])

    def supports(self, tool: str) -> bool:
        return tool.startswith("ffmpeg-skill/")

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        return [f"fake {op.tool} {op.args}"]

    def measure(self, tool: str, args: Dict[str, Any], paths=None, timeout=None) -> ToolResult:
        return self.run(Operation(tool=tool, args=args, inputs=[], outputs=[], kind="measure"), paths or {})

    def run(self, op: Operation, paths: Dict[str, str], timeout=None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        self.calls.append(op)
        script = self.ALIASES.get(op.tool.split("/")[1], op.tool.split("/")[1])
        if self.fail_tools.get(op.tool, 0) > 0:
            self.fail_tools[op.tool] -= 1
            return ToolResult(op.id, op.tool, False, 1, None, {}, [], "error: command failed (1): ffmpeg\nConversion failed!", 0.1, attempt, dry_run)
        out = paths.get(op.args.get("output", ""), op.args.get("output")) if op.args.get("output") else None
        src = paths.get(op.args.get("input", ""), op.args.get("input")) if op.args.get("input") else None
        in_meta = _read_fake(src) if src else None
        in_dur = in_meta["duration"] if in_meta else self.duration
        if script == "probe":
            meta = _read_fake(op.args["inputs"][0])
            dur = meta["duration"] if meta else self.duration
            return ToolResult(op.id, op.tool, True, 0, None, probe_doc(op.args["inputs"][0], dur, self.audio, self.vfr, self.hdr), [], "", 0.1, attempt, dry_run)
        if script == "silence":
            keep = [[max(0.0, (s[1] or self.duration) - 0.15) if i == 0 and s[0] == 0 else 0.0, 0.0] for i, s in enumerate(self.silences[:1])]
            data = {"silences": self.silences, "keep": [[2.85, 13.85]], "input_duration": self.duration, "kept_duration": 11.0, "removed_seconds": 5.0}
            return ToolResult(op.id, op.tool, True, 0, out, data, [], "", 0.1, attempt, dry_run)
        if script == "loudness":
            if op.args.get("measure_only"):
                lufs = in_meta.get("lufs", self.lufs) if in_meta else self.lufs
                return ToolResult(op.id, op.tool, True, 0, None, {"input_i": str(lufs), "input_tp": str(self.true_peak), "input_lra": "6.0", "input_thresh": "-21", "target_offset": "0"}, [], "", 0.1, attempt, dry_run)
            self.lufs = float(op.args["lufs"])
            if out and not dry_run:
                _write_fake(out, {"duration": in_dur, "lufs": self.lufs})
            return ToolResult(op.id, op.tool, True, 0, out, {"output": out, "commands": ["ffmpeg loudnorm"]}, ["ffmpeg loudnorm"], "", 0.2, attempt, dry_run)
        if script in ("cut", "export", "fit"):
            dur = in_dur
            if script == "cut":
                segs = [tuple(float(x) for x in s.split("-")) for s in op.args["segments"].split(",")]
                dur = sum(e - s for s, e in segs)
            self.duration = dur
            if out and not dry_run:
                _write_fake(out, {"duration": dur, "lufs": in_meta.get("lufs", self.lufs) if in_meta else self.lufs})
            return ToolResult(op.id, op.tool, True, 0, out, {"output": out, "commands": [f"ffmpeg {script}"], "probe": probe_doc(out or "", dur)}, [f"ffmpeg {script}"], "", 0.2, attempt, dry_run)
        if script == "check":
            rows = [{"check": "duration", "status": "PASS", "value": f"{in_dur}s", "expected": "any", "fix": "", "kind": "judgement"},
                    {"check": "video codec", "status": "PASS", "value": "h264", "expected": "h264", "fix": "", "kind": "format"}]
            return ToolResult(op.id, op.tool, True, 0, None, {"platform": op.args["platform"], "checks": rows, "failed": 0, "warnings": 0, "ok": True}, [], "", 0.1, attempt, dry_run)
        if script == "look":
            if out and not dry_run:
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                Path(out).write_bytes(b"png")
            return ToolResult(op.id, op.tool, True, 0, out, {"outputs": [out]}, [], "", 0.1, attempt, dry_run)
        return ToolResult(op.id, op.tool, False, 2, None, {}, [], f"error: unknown tool {script}", 0.0, attempt, dry_run)
