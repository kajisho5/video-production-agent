"""ThumbnailAdapter boundary tests against the fake thumbnail-skill process (tests/fake_thumbnail.py), plus an optional
real-Skill class (VIDEO_AGENT_THUMBNAIL_DIR; VIDEO_AGENT_THUMBNAIL_PYTHON selects an interpreter with Pillow)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from video_agent.models import Operation  # noqa: E402
from video_agent.tools.base import ToolError  # noqa: E402
from video_agent.tools.ffmpeg_skill.adapter import PathPolicy  # noqa: E402
from video_agent.tools.skill_process import CliSkill  # noqa: E402
from video_agent.tools.thumbnail import (DRIFT_KEYS, PACKAGE, PREFIX, SKILL_ID, TOOL_EXTRACT_FRAME, TOOL_RENDER, ContractError, ThumbnailAdapter,  # noqa: E402
                                         check_contract, contract_drift, lift_observation, locate_thumbnail, package_from_contract, pinned_contract)

FAKE = Path(__file__).resolve().parent / "fake_thumbnail.py"
MODE = "FAKE_THUMBNAIL_MODE"


def fake_skill() -> CliSkill:
    return CliSkill("thumbnail", [sys.executable, str(FAKE)], None, {})


def set_mode(mode: str) -> None:
    os.environ[MODE] = mode


class FakeBase(unittest.TestCase):
    def setUp(self):
        set_mode("ok")
        self.tmp = tempfile.mkdtemp()
        self.ws = os.path.join(self.tmp, "ws")
        self.src_dir = os.path.join(self.tmp, "src")
        os.makedirs(self.ws)
        os.makedirs(self.src_dir)
        self.video = os.path.join(self.src_dir, "clip.mp4")
        Path(self.video).write_bytes(json.dumps({"fake": True, "duration": 12.0, "video": True, "channels": 2}).encode())
        self.calls = os.path.join(self.tmp, "calls.jsonl")
        os.environ["FAKE_THUMBNAIL_CALLS"] = self.calls
        self.adapter = ThumbnailAdapter(fake_skill(), workspace=self.ws, allowed_inputs=[self.src_dir], ffmpeg_skill_dir="/opt/fake-ffmpeg-skill", timeout=20.0)

    def tearDown(self):
        os.environ.pop(MODE, None)
        os.environ.pop("FAKE_THUMBNAIL_CALLS", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def paths(self, out="thumb.png"):
        return {"clip": self.video, "out": os.path.join(self.ws, "out", out)}

    def frame_op(self, **extra) -> Operation:
        args = {"input": "clip", "timestamp": 2.0, "format": "png", "output": "out"}
        args.update(extra)
        return Operation(tool=TOOL_EXTRACT_FRAME, args=args, inputs=["clip"], outputs=["out"], id="op-frame-1")

    def render_op(self, **extra) -> Operation:
        args = {"input": "clip", "timestamp": 1.0, "format": "png", "output": "out", "width": 640, "height": 360, "text": "Hello\nThumb"}
        args.update(extra)
        return Operation(tool=TOOL_RENDER, args=args, inputs=["clip"], outputs=["out"], id="op render:1")

    def logged(self):
        return [json.loads(line) for line in Path(self.calls).read_text(encoding="utf-8").splitlines()]


class ContractTests(FakeBase):
    def test_discovery(self):
        a = self.adapter
        self.assertEqual(a.version, "0.1.0")
        self.assertEqual(a.tools, {TOOL_RENDER, TOOL_EXTRACT_FRAME})
        self.assertEqual(a.drift(), [])
        self.assertEqual(a.font_ids, ["cjk", "mono", "sans", "sans-bold", "serif"])
        self.assertEqual(a.canvas_range["width"], (16.0, 7680.0))
        self.assertEqual(a.font_size_range, (6.0, 400.0))
        self.assertEqual(a.text_max, 2000)
        self.assertTrue(a.retryable["TOOL_ERROR"] and a.retryable["CANCELLED"] and not a.retryable["INVALID_REQUEST"])
        self.assertEqual(a.exit_codes["CANCELLED"], 15)
        self.assertTrue(a.supports(TOOL_RENDER) and not a.supports("thumbnail/validate"))
        d = a.describe()
        self.assertEqual(d["contract"], "thumbnail-skill/contract@1")
        self.assertIn("command", a.forbidden)
        self.assertIn("workspace", a.forbidden)

    def test_package(self):
        pkg = self.adapter.package()
        self.assertEqual(pkg.validate(), [])
        self.assertEqual(pkg.skill_id, SKILL_ID)
        self.assertEqual(pkg.tool_ids(), [TOOL_RENDER, TOOL_EXTRACT_FRAME])
        self.assertEqual(pkg.repository, "kajisho5/thumbnail-skill")
        self.assertEqual(pkg.capabilities, ["thumbnail"])
        t = pkg.tool(TOOL_RENDER)
        self.assertTrue(t.produces_output)
        self.assertEqual(t.inputs, ["input", "output"])
        self.assertEqual(t.required_capabilities, ["thumbnail", "ffmpeg", "ffprobe", "ffmpeg-skill"])
        self.assertEqual(PACKAGE.version, "0.1.0")
        self.assertEqual(package_from_contract(pinned_contract()).tool_ids(), PACKAGE.tool_ids())
        self.assertEqual(PREFIX, "thumbnail/")

    def test_pinned_contract_accepted(self):
        self.assertEqual(check_contract(pinned_contract()), [])
        self.assertEqual(contract_drift(pinned_contract()), [])
        self.assertIn("fonts", DRIFT_KEYS)

    def test_refusals(self):
        for mode, frag in (("wrong_schema", "contract schema"), ("wrong_skill", "skill_id"), ("wrong_version", "version"), ("bad_contract", "execution.shell"),
                           ("contract_fail", "exit 1"), ("pillow_missing", "No module named 'PIL'")):
            set_mode(mode)
            with self.assertRaises(ContractError) as cm:
                ThumbnailAdapter(fake_skill(), workspace=self.ws)
            self.assertIn(frag, str(cm.exception), mode)

    def test_check_contract_details(self):
        c = pinned_contract()
        c["tools"] = [t for t in c["tools"] if t["tool_id"] != TOOL_EXTRACT_FRAME]
        self.assertTrue(any("lacks tool thumbnail/extract_frame" in e for e in check_contract(c)))
        c = pinned_contract()
        c["fonts"]["font_ids"] = ["serif"]
        self.assertTrue(any("sans-bold" in e for e in check_contract(c)))
        c = pinned_contract()
        c["output_formats"].pop("jpeg")
        self.assertTrue(any("png and jpeg" in e for e in check_contract(c)))
        c = pinned_contract()
        c["document"]["forbidden_fields"].remove("command")
        self.assertTrue(any("lack 'command'" in e for e in check_contract(c)))
        self.assertEqual(check_contract("nope"), ["contract is not an object"])

    def test_drift_detected(self):
        set_mode("contract_drift")
        a = ThumbnailAdapter(fake_skill(), workspace=self.ws)
        self.assertEqual(check_contract(a.contract), [])
        self.assertTrue(any(d.startswith("fonts:") for d in a.drift()), a.drift())
        live = pinned_contract()
        live["tools"].append(dict(live["tools"][1], tool_id="thumbnail/sprite"))
        live["tools"][1]["produces_output"] = False
        rep = contract_drift(live)
        self.assertTrue(any("thumbnail/sprite: installed but not pinned" in d for d in rep))
        self.assertTrue(any("thumbnail/render.produces_output" in d for d in rep))

    def test_locate(self):
        self.assertIsNone(locate_thumbnail(env={"PATH": self.tmp}))
        root = os.path.join(self.tmp, "thumbnail-skill")
        os.makedirs(os.path.join(root, "src", "thumbnail_skill"))
        Path(root, "src", "thumbnail_skill", "cli.py").write_text("", encoding="utf-8")
        s = locate_thumbnail(env={"VIDEO_AGENT_THUMBNAIL_DIR": root, "PATH": self.tmp})
        self.assertEqual(s.command, [sys.executable, "-m", "thumbnail_skill.cli"])
        self.assertEqual(s.env, {"PYTHONPATH": os.path.join(root, "src")})

    def test_doctor(self):
        doc = self.adapter.doctor()
        self.assertEqual((doc["schema"], doc["status"], doc["exit_code"]), ("thumbnail-skill/doctor@1", "ok", 0))
        self.assertEqual(self.adapter.font_status(doc)["sans-bold"], "available")
        self.assertEqual(self.adapter.engine_status(doc), "ok")
        self.assertEqual(doc["checks"]["ffmpeg_skill"]["directory"], "/opt/fake-ffmpeg-skill")
        self.assertEqual(doc["checks"]["path_policy"]["allowed_input_roots"], [self.src_dir])
        set_mode("doctor_degraded")
        doc = self.adapter.doctor()
        self.assertEqual(doc["status"], "degraded")
        self.assertEqual(self.adapter.font_status(doc)["cjk"], "missing")
        set_mode("doctor_fail")
        doc = self.adapter.doctor()
        self.assertEqual((doc["status"], doc["exit_code"]), ("fail", 1))
        self.assertEqual(self.adapter.engine_status(doc), "fail")
        set_mode("pillow_missing")
        doc = self.adapter.doctor()
        self.assertEqual((doc["status"], doc["exit_code"]), ("fail", 1))
        self.assertIn("no document", doc["problems"][0])


class RequestTests(FakeBase):
    def test_extract_frame_lowering(self):
        b = self.adapter.build_request(TOOL_EXTRACT_FRAME, {"input": "clip", "timestamp": 2, "format": "jpeg", "jpeg_quality": 85, "output": "out"}, self.paths("f.jpg"), op_id="op1")
        req = b["request"]
        self.assertEqual(req["tool"], TOOL_EXTRACT_FRAME)
        self.assertEqual(req["params"], {"source": {"path": os.path.realpath(self.video), "timestamp": 2.0},
                                         "output": {"path": os.path.join(self.ws, "out", "f.jpg"), "format": "jpeg", "overwrite": True, "jpeg_quality": 85}})
        self.assertEqual(b["workspace"], self.ws)
        self.assertNotIn("options", req["params"])

    def test_render_lowering(self):
        args = {"input": "clip", "timestamp": 1.5, "format": "png", "output": "out", "width": 640, "height": 360, "background": "#202020", "text": "Hi", "font_id": "serif",
                "font_size": 40, "color": "#FF0000", "position": "top"}
        b = self.adapter.build_request(TOOL_RENDER, args, self.paths(), op_id="op render:1")
        p = b["request"]["params"]
        self.assertEqual(p["schema"], "thumbnail-skill/request@1")
        d = p["document"]
        self.assertEqual(d["document_id"], "op_render_1")
        self.assertEqual(d["canvas"], {"width": 640, "height": 360, "background": "#202020"})
        self.assertEqual(d["assets"], [{"asset_id": "frame", "kind": "video_frame", "path": os.path.realpath(self.video), "timestamp": 1.5}])
        self.assertEqual(d["elements"][0], {"element_id": "frame", "type": "image", "z_index": 0, "image": {"asset_id": "frame", "position": {"x": 0, "y": 0}, "size": {"width": 640, "height": 360}, "fit": "cover"}})
        self.assertEqual(d["elements"][1]["text"], {"text": "Hi", "font_id": "serif", "font_size": 40, "color": "#FF0000", "position": {"x": 320, "y": 40}, "align": {"horizontal": "center", "vertical": "top"}})
        self.assertEqual(p["output"], {"path": os.path.join(self.ws, "out", "thumb.png"), "format": "png", "overwrite": True})
        # defaults and the other anchors
        d = self.adapter.build_request(TOOL_RENDER, {"input": "clip", "timestamp": 0, "format": "png", "output": "out", "width": 100, "height": 50, "text": "x"}, self.paths())["request"]["params"]["document"]
        self.assertEqual(d["elements"][1]["text"], {"text": "x", "font_id": "sans-bold", "font_size": 48, "color": "#FFFFFF", "position": {"x": 50, "y": 2}, "align": {"horizontal": "center", "vertical": "bottom"}})
        self.assertEqual(d["canvas"]["background"], "#000000")
        d = self.adapter.build_request(TOOL_RENDER, {"input": "clip", "timestamp": 0, "format": "png", "output": "out", "width": 100, "height": 50, "text": "x", "position": "center"}, self.paths())["request"]["params"]["document"]
        self.assertEqual(d["elements"][1]["text"]["position"], {"x": 50, "y": 25})
        self.assertEqual(d["elements"][1]["text"]["align"], {"horizontal": "center", "vertical": "middle"})
        # no text → no caption element
        d = self.adapter.build_request(TOOL_RENDER, {"input": "clip", "timestamp": 0, "format": "png", "output": "out", "width": 100, "height": 50}, self.paths())["request"]["params"]["document"]
        self.assertEqual(len(d["elements"]), 1)

    def refuse(self, tool, args, frag, paths=None, adapter=None):
        with self.assertRaises(ToolError) as cm:
            (adapter or self.adapter).build_request(tool, args, paths or self.paths())
        self.assertIn(frag, str(cm.exception))

    def test_refusals(self):
        base = {"input": "clip", "timestamp": 1.0, "format": "png", "output": "out"}
        self.refuse("thumbnail/validate", base, "unsupported tool")
        self.refuse(TOOL_EXTRACT_FRAME, dict(base, command="rm -rf /"), "forbidden field args.command")
        self.refuse(TOOL_EXTRACT_FRAME, dict(base, filter="scale"), "forbidden field")
        self.refuse(TOOL_EXTRACT_FRAME, dict(base, width=100), "not declared for thumbnail/extract_frame")
        self.refuse(TOOL_RENDER, dict(base, width=640, height=360, crop=1), "not declared for thumbnail/render")
        self.refuse(TOOL_EXTRACT_FRAME, {"input": "clip", "timestamp": 1.0, "format": "png"}, "input and output references are required")
        self.refuse(TOOL_EXTRACT_FRAME, dict(base, input="missing"), "input not found")
        self.refuse(TOOL_EXTRACT_FRAME, dict(base, format="gif"), "not one of ['jpeg', 'png']")
        self.refuse(TOOL_EXTRACT_FRAME, dict(base, format="jpeg"), "does not match format jpeg")
        self.refuse(TOOL_EXTRACT_FRAME, dict(base, jpeg_quality=50), "only accepted for format jpeg")
        self.refuse(TOOL_EXTRACT_FRAME, dict(base, format="jpeg", jpeg_quality=101), "outside the contract range 1..100", self.paths("f.jpg"))
        self.refuse(TOOL_EXTRACT_FRAME, dict(base, timestamp=-1), "outside the contract range")
        self.refuse(TOOL_EXTRACT_FRAME, dict(base, timestamp="1"), "finite number")
        self.refuse(TOOL_EXTRACT_FRAME, dict(base, timestamp=float("nan")), "finite number")
        self.refuse(TOOL_EXTRACT_FRAME, {k: v for k, v in base.items() if k != "timestamp"}, "timestamp is required")
        self.refuse(TOOL_RENDER, base, "render requires width")
        self.refuse(TOOL_RENDER, dict(base, width=8, height=360), "outside the contract range 16..7680")
        self.refuse(TOOL_RENDER, dict(base, width=640, height=8000), "outside the contract range 16..7680")
        self.refuse(TOOL_RENDER, dict(base, width=640.5, height=360), "must be an integer")
        self.refuse(TOOL_RENDER, dict(base, width=640, height=360, background="red"), "'#RRGGBB'")
        self.refuse(TOOL_RENDER, dict(base, width=640, height=360, text=""), "non-empty string")
        self.refuse(TOOL_RENDER, dict(base, width=640, height=360, text="x" * 2001), "at most 2000")
        self.refuse(TOOL_RENDER, dict(base, width=640, height=360, text="a\tb"), "explicit newlines only")
        self.refuse(TOOL_RENDER, dict(base, width=640, height=360, text="x", font_id="comic"), "not one of the contract font_ids")
        self.refuse(TOOL_RENDER, dict(base, width=640, height=360, text="x", font_size=5), "outside the contract range 6..400")
        self.refuse(TOOL_RENDER, dict(base, width=640, height=360, text="x", color="#FFF"), "'#RRGGBB'")
        self.refuse(TOOL_RENDER, dict(base, width=640, height=360, text="x", position="left"), "not one of ['center', 'top', 'bottom']")
        self.refuse(TOOL_RENDER, dict(base, width=640, height=360, font_size=20), "needs a text")

    def test_path_refusals(self):
        base = {"input": "clip", "timestamp": 1.0, "format": "png", "output": "out"}
        self.refuse(TOOL_EXTRACT_FRAME, base, "output outside the workspace", {"clip": self.video, "out": os.path.join(self.tmp, "elsewhere.png")})
        outside = os.path.join(self.tmp, "outside.mp4")
        Path(outside).write_bytes(b'{"fake": true, "duration": 5}')
        self.refuse(TOOL_EXTRACT_FRAME, base, "input outside the allowed roots", {"clip": outside, "out": os.path.join(self.ws, "x.png")})
        # an input inside the workspace is fine
        inside = os.path.join(self.ws, "in.mp4")
        Path(inside).write_bytes(b'{"fake": true, "duration": 5}')
        self.adapter.build_request(TOOL_EXTRACT_FRAME, base, {"clip": inside, "out": os.path.join(self.ws, "x.png")})
        self.refuse(TOOL_EXTRACT_FRAME, dict(base, format="jpeg"), "overwrite its input", {"clip": inside, "out": inside})
        # the same rules through a PathPolicy
        pol = ThumbnailAdapter(fake_skill(), workspace=self.ws, path_policy=PathPolicy([self.src_dir], self.ws))
        self.refuse(TOOL_EXTRACT_FRAME, base, "input outside allowed roots", {"clip": outside, "out": os.path.join(self.ws, "x.png")}, adapter=pol)
        self.refuse(TOOL_EXTRACT_FRAME, base, "output outside workspace", {"clip": self.video, "out": os.path.join(self.tmp, "x.png")}, adapter=pol)

    def test_preview_and_dry_run(self):
        lines = self.adapter.preview(self.frame_op(), self.paths())
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("thumbnail run - --json --workspace "))
        self.assertIn("--ffmpeg-skill /opt/fake-ffmpeg-skill", lines[0])
        self.assertIn('"tool": "thumbnail/extract_frame"', lines[0])
        self.assertEqual(self.adapter.preview(self.frame_op(width=1), self.paths()), ["thumbnail: refused: thumbnail: parameter(s) ['width'] are not declared for thumbnail/extract_frame (accepted: ['input', 'output', 'format', 'timestamp', 'jpeg_quality'])"])
        calls = self.adapter.calls
        r = self.adapter.run(self.frame_op(), self.paths(), dry_run=True)
        self.assertTrue(r.ok and r.dry_run and r.output is None)
        self.assertEqual(r.data["status"], "dry_run")
        self.assertEqual(r.data["request"]["tool"], TOOL_EXTRACT_FRAME)
        self.assertEqual(self.adapter.calls, calls)   # the Skill has no plan mode: never invoked for a dry run
        self.assertFalse(os.path.exists(self.paths()["out"]))
        with self.assertRaises(ToolError):
            self.adapter.measure(TOOL_RENDER, {})


class SuccessTests(FakeBase):
    def test_extract_frame(self):
        r = self.adapter.run(self.frame_op(format="jpeg", jpeg_quality=80), self.paths("frame.jpg"), timeout=15)
        self.assertTrue(r.ok, r.data)
        self.assertEqual(r.exit_code, 0)
        self.assertEqual(r.output, self.paths("frame.jpg")["out"])
        self.assertTrue(Path(r.output).read_bytes().startswith(b"\xff\xd8\xff"))
        d = r.data
        self.assertEqual(d["skill"], {"id": "thumbnail", "version": "0.1.0"})
        self.assertEqual((d["status"], d["operation_type"]), ("completed", "extract_frame"))
        art = d["artifact"]
        self.assertEqual(art["sha256"], hashlib.sha256(Path(r.output).read_bytes()).hexdigest())
        self.assertEqual(art["size"], os.path.getsize(r.output))
        self.assertEqual((art["format"], art["width"], art["height"], art["reused"]), ("jpeg", 1280, 720, False))
        self.assertTrue(art["identity"])
        self.assertEqual(d["source"]["sha256"], hashlib.sha256(Path(self.video).read_bytes()).hexdigest())
        self.assertEqual((d["source"]["timestamp"], d["source"]["duration"]), (2.0, 12.0))
        self.assertEqual(d["provenance"]["engine"], "ffmpeg-skill/look")
        self.assertEqual(d["provenance"]["operation"], TOOL_EXTRACT_FRAME)
        self.assertEqual(d["commands"], [])
        self.assertEqual(r.commands, [])
        self.assertEqual(d["warnings"], [])
        obs = lift_observation(r, "thumb-asset")
        self.assertEqual((obs.kind, obs.asset_id, obs.source, obs.provenance, obs.skill, obs.tool), ("image.probe", "thumb-asset", "thumbnail-skill/extract_frame@0.1.0", "OBSERVED", "thumbnail", TOOL_EXTRACT_FRAME))
        self.assertEqual(obs.data, {"width": 1280, "height": 720, "format": "jpeg", "size": art["size"], "sha256": art["sha256"]})
        self.assertEqual(obs.fingerprint, art["sha256"])
        argv = self.logged()[-1]["argv"]
        self.assertEqual(argv[-2:], ["--timeout", "15"])

    def test_render(self):
        r = self.adapter.run(self.render_op(), self.paths())
        self.assertTrue(r.ok, r.data)
        self.assertTrue(Path(r.output).read_bytes().startswith(b"\x89PNG"))
        d = r.data
        self.assertEqual(d["operation_type"], "render")
        self.assertEqual((d["artifact"]["width"], d["artifact"]["height"], d["artifact"]["format"]), (640, 360, "png"))
        self.assertEqual(d["provenance"]["engine"], "Pillow")
        self.assertEqual(d["provenance"]["document_id"], "op_render_1")
        self.assertEqual(d["provenance"]["assets"][0]["kind"], "video_frame")
        self.assertEqual(d["provenance"]["fonts"][0]["font_id"], "sans-bold")
        self.assertEqual(d["source"]["sha256"], hashlib.sha256(Path(self.video).read_bytes()).hexdigest())
        self.assertEqual(d["source"]["timestamp"], 1.0)
        self.assertEqual(lift_observation(r).source, "thumbnail-skill/render@0.1.0")
        self.assertIsNone(lift_observation(self.adapter.run(self.frame_op(timestamp=99.0), self.paths())))

    def test_reused(self):
        self.assertTrue(self.adapter.run(self.frame_op(), self.paths()).ok)
        set_mode("reused")
        r = self.adapter.run(self.frame_op(), self.paths())
        self.assertTrue(r.ok)
        self.assertTrue(r.data["artifact"]["reused"])


class ErrorTests(FakeBase):
    def check(self, mode, code, retryable, recovery, op=None, exit_code=None, frag=None):
        set_mode(mode)
        r = self.adapter.run(op or self.frame_op(), self.paths())
        self.assertFalse(r.ok, mode)
        e = r.data["error"]
        self.assertEqual(e["code"], code, (mode, e))
        self.assertEqual(e["retryable"], retryable, mode)
        self.assertEqual(e["recovery_class"], recovery, mode)
        self.assertEqual(e["exit_code"], r.exit_code)
        if exit_code is not None:
            self.assertEqual(r.exit_code, exit_code, mode)
        if frag:
            self.assertIn(frag, e["message"], mode)
        self.assertEqual(r.data["skill"]["id"], "thumbnail")
        self.assertFalse(os.path.exists(self.paths()["out"]), mode)
        return r

    def test_skill_errors(self):
        self.check("tool_error", "TOOL_ERROR", True, "UNKNOWN", exit_code=12)
        r = self.check("tool_error_final", "TOOL_ERROR", False, "UNKNOWN")
        self.assertNotIn("command", r.data["error"]["details"])   # details scrubbed
        self.assertEqual(r.data["error"]["details"]["reason"], "tool_failed")
        self.check("validation_error", "VALIDATION_ERROR", False, "SKILL_ERROR", exit_code=14)
        self.check("internal_error", "INTERNAL_ERROR", False, "SKILL_ERROR", exit_code=16)
        r = self.check("cancelled", "CANCELLED", True, "TIMEOUT", exit_code=15)
        self.assertEqual(r.data["error"]["details"]["reason"], "signal")
        self.check("unknown_code", "INVALID_RESULT", False, "SKILL_ERROR")
        # the Skill's own validation, through the fake: timestamp beyond the duration, a request the transport refuses
        self.check("ok", "INVALID_TIME_RANGE", False, "INVALID_ARGS", op=self.frame_op(timestamp=13.0), exit_code=8, frag="beyond the source duration")
        r = self.check("ok", "INVALID_REQUEST", False, "INVALID_ARGS", op=self.frame_op(input="nope"), exit_code=2, frag="input not found")
        self.assertEqual(self.adapter.calls, 8)   # refused before any process was spawned (1 contract + 7 runs so far)

    def test_invalid_results(self):
        self.check("output_missing", "INVALID_RESULT", False, "SKILL_ERROR", frag="output file missing")
        self.check("hash_mismatch", "INVALID_RESULT", False, "SKILL_ERROR", frag="sha256")
        self.check("wrong_size", "INVALID_RESULT", False, "SKILL_ERROR", frag="size differs")
        self.check("malformed", "INVALID_RESULT", False, "SKILL_ERROR", exit_code=9, frag="not JSON")
        self.check("empty", "INVALID_RESULT", False, "SKILL_ERROR", frag="empty stdout")
        self.check("two_docs", "INVALID_RESULT", False, "SKILL_ERROR", frag="more than one document")
        self.check("text", "INVALID_RESULT", False, "SKILL_ERROR", exit_code=1)
        self.check("nonzero_ok", "INVALID_RESULT", False, "SKILL_ERROR", exit_code=3, frag="exit code 3 with an ok response")
        self.check("wrong_schema", "INVALID_RESULT", False, "SKILL_ERROR", frag="response schema")
        self.check("wrong_skill", "INVALID_RESULT", False, "SKILL_ERROR", frag="is not thumbnail-skill@0.1.0")
        self.check("wrong_version", "INVALID_RESULT", False, "SKILL_ERROR", frag="is not thumbnail-skill@0.1.0")
        self.check("no_provenance", "INVALID_RESULT", False, "SKILL_ERROR", frag="provenance incomplete")
        self.check("outer_not_ok", "INVALID_RESULT", False, "SKILL_ERROR", frag="transport envelope ok=False")
        self.check("wrong_tool", "INVALID_RESULT", False, "SKILL_ERROR", frag="tool='thumbnail/validate' is not the dispatched thumbnail/extract_frame")
        self.check("pillow_missing", "INVALID_RESULT", False, "SKILL_ERROR", exit_code=1, frag="empty stdout")

    def test_timeout(self):
        set_mode("timeout")
        r = self.adapter.run(self.frame_op(), self.paths(), timeout=1)
        self.assertFalse(r.ok)
        self.assertEqual((r.exit_code, r.data["error"]["code"], r.data["error"]["retryable"], r.data["error"]["recovery_class"]), (124, "CANCELLED", True, "TIMEOUT"))
        self.assertEqual(r.data["error"]["details"]["reason"], "timeout")
        self.assertLess(r.seconds, 20)

    def test_failed_result_shape(self):
        set_mode("tool_error")
        r = self.adapter.run(self.frame_op(), self.paths(), attempt=2)
        self.assertEqual(r.attempt, 2)
        self.assertEqual(sorted(r.data["error"]), ["code", "details", "exit_code", "message", "recovery_class", "retryable"])
        self.assertEqual(r.data["status"], "failed")


class SecurityTests(FakeBase):
    def test_argv_and_env(self):
        inherited = {k for k in os.environ if k.startswith("VIDEO_")}
        r = self.adapter.run(self.render_op(text="x; rm -rf /", font_id="mono"), self.paths())
        self.assertTrue(r.ok, r.data)
        calls = self.logged()
        self.assertEqual([c["cmd"] for c in calls], ["skill", "run"])
        run = calls[-1]
        self.assertIsInstance(run["argv"], list)
        self.assertEqual(run["argv"][:3], ["run", "-", "--json"])
        self.assertIn("--workspace", run["argv"])
        self.assertEqual(run["argv"][run["argv"].index("--workspace") + 1], self.ws)
        self.assertEqual([run["argv"][i + 1] for i, a in enumerate(run["argv"]) if a == "--allowed-input"], [self.src_dir, self.ws])
        joined = " ".join(run["argv"])
        for key in ("clip.mp4", "thumb.png", "rm -rf", "font_id", "timestamp", "width", "mono", "Hello"):
            self.assertNotIn(key, joined, key)   # request content never travels on argv
        self.assertTrue(set(run["env_video"]) <= inherited, run["env_video"])   # the adapter adds nothing to the child's VIDEO_* environment (no secrets, no engine paths: argv carries them)
        self.assertNotIn("VIDEO_AGENT_FFMPEG_SKILL_DIR", set(run["env_video"]) - inherited)
        # the request that was sent carries no forbidden key anywhere
        b = self.adapter.build_request(TOOL_RENDER, self.render_op().args, self.paths(), op_id="x")
        from video_agent.tools.skill_process import scan_forbidden
        self.assertIsNone(scan_forbidden(b["request"], tuple(f for f in self.adapter.forbidden if f not in ("path", "paths"))))
        self.assertIsNone(scan_forbidden(b["request"], ("command", "argv", "shell", "exec", "env", "filter", "workspace", "ffmpeg_skill", "allowed_input_roots")))

    def test_forbidden_never_forwarded(self):
        for key in ("command", "argv", "shell", "exec", "env", "filter_complex", "vf", "api_key", "workspace", "ffmpeg_skill", "path"):
            set_mode("ok")
            r = self.adapter.run(self.frame_op(**{key: "x"}), self.paths())
            self.assertFalse(r.ok, key)
            self.assertEqual(r.data["error"]["code"], "INVALID_REQUEST", key)
            self.assertIn(key, r.data["error"]["message"])
        self.assertEqual([c["cmd"] for c in self.logged()], ["skill"])   # none of those reached the Skill
        # nested forbidden keys are found too
        r = self.adapter.run(self.frame_op(jpeg_quality={"cmd": "x"}), self.paths())
        self.assertIn("forbidden field args.jpeg_quality.cmd", r.data["error"]["message"])


@unittest.skipUnless(os.environ.get("VIDEO_AGENT_THUMBNAIL_DIR"), "real thumbnail-skill: set VIDEO_AGENT_THUMBNAIL_DIR (and VIDEO_AGENT_THUMBNAIL_PYTHON for an interpreter with Pillow)")
class RealSkillTests(unittest.TestCase):
    """Runs the installed Skill. Video frames need ffmpeg and ffmpeg-skill (VIDEO_AGENT_FFMPEG_SKILL_DIR); the Skill needs Pillow
    in the interpreter that runs it (VIDEO_AGENT_THUMBNAIL_PYTHON, else sys.executable) — skipped, never faked, when absent."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.ws = os.path.join(cls.tmp, "ws")
        os.makedirs(cls.ws)
        root = Path(os.environ["VIDEO_AGENT_THUMBNAIL_DIR"]).resolve()
        py = os.environ.get("VIDEO_AGENT_THUMBNAIL_PYTHON") or sys.executable
        skill = CliSkill("thumbnail", [py, "-m", "thumbnail_skill.cli"], root, {"PYTHONPATH": str(root / "src")})
        try:
            cls.adapter = ThumbnailAdapter(skill, workspace=cls.ws, ffmpeg_skill_dir=os.environ.get("VIDEO_AGENT_FFMPEG_SKILL_DIR"))
        except ContractError as e:
            raise unittest.SkipTest(f"thumbnail-skill not usable here: {e}")
        cls.video = os.path.join(cls.ws, "src.mp4")
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "3",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", cls.video], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def need_video(self):
        if not os.path.isfile(self.video):
            self.skipTest("ffmpeg not available for the fixture")
        if self.adapter.engine_status(self.adapter.doctor()) != "ok":
            self.skipTest("ffmpeg-skill not usable (set VIDEO_AGENT_FFMPEG_SKILL_DIR)")

    def test_contract(self):
        self.assertEqual(self.adapter.version[:4], "0.1.")
        self.assertEqual(self.adapter.drift(), [])
        doc = self.adapter.doctor()
        self.assertEqual(doc["schema"], "thumbnail-skill/doctor@1")
        self.assertIn(doc["status"], ("ok", "degraded", "fail"))

    def test_extract_frame(self):
        self.need_video()
        op = Operation(tool=TOOL_EXTRACT_FRAME, args={"input": "v", "timestamp": 2.0, "format": "jpeg", "jpeg_quality": 85, "output": "o"}, inputs=["v"], outputs=["o"])
        r = self.adapter.run(op, {"v": self.video, "o": os.path.join(self.ws, "out", "frame.jpg")})
        self.assertTrue(r.ok, r.data)
        self.assertEqual(r.data["artifact"]["width"], 1280)   # ffmpeg-skill/look's default width: a fact, not adjusted
        self.assertEqual(r.data["artifact"]["sha256"], hashlib.sha256(Path(r.output).read_bytes()).hexdigest())
        self.assertEqual(r.data["source"]["duration"], 3.0)
        r2 = self.adapter.run(op, {"v": self.video, "o": os.path.join(self.ws, "out", "frame.jpg")})
        self.assertTrue(r2.ok and r2.data["artifact"]["reused"])
        bad = self.adapter.run(Operation(tool=TOOL_EXTRACT_FRAME, args={"input": "v", "timestamp": 9.0, "format": "png", "output": "o"}, inputs=["v"], outputs=["o"]),
                               {"v": self.video, "o": os.path.join(self.ws, "out", "late.png")})
        self.assertEqual((bad.ok, bad.data["error"]["code"], bad.exit_code), (False, "INVALID_TIME_RANGE", 8))

    def test_render(self):
        self.need_video()
        if self.adapter.font_status(self.adapter.doctor()).get("sans-bold") != "available":
            self.skipTest("font sans-bold not available")
        op = Operation(tool=TOOL_RENDER, args={"input": "v", "timestamp": 1.0, "format": "png", "output": "o", "width": 640, "height": 360, "background": "#202020", "text": "Hello\nThumb"},
                       inputs=["v"], outputs=["o"], id="real-render")
        r = self.adapter.run(op, {"v": self.video, "o": os.path.join(self.ws, "out", "thumb.png")})
        self.assertTrue(r.ok, r.data)
        self.assertEqual((r.data["artifact"]["width"], r.data["artifact"]["height"]), (640, 360))
        self.assertEqual(r.data["provenance"]["engine"], "Pillow")
        self.assertEqual(r.data["provenance"]["document_id"], "real-render")
        self.assertTrue(lift_observation(r).data["sha256"])


if __name__ == "__main__":
    unittest.main()
