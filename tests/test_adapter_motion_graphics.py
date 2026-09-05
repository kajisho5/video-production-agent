"""MotionGraphicsAdapter boundary tests against the fake motion-graphics-skill process (tests/fake_motion_graphics.py): contract
discovery and refusals, drift detection, request building, success mapping, every error mode, timeout, security. A real-Skill
class runs only when VIDEO_AGENT_MOTION_GRAPHICS_DIR is set."""
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
from video_agent.tools.motion_graphics import (ENV_DIR, PACKAGE, SKILL_ID, TOOL_ID, ContractError, MotionGraphicsAdapter, check_contract,  # noqa: E402
                                               contract_drift, lift_observation, locate_motion_graphics, pinned_contract)
from video_agent.tools.skill_process import CliSkill, sha256_file  # noqa: E402

FAKE = Path(__file__).resolve().parent / "fake_motion_graphics.py"


def fake_skill() -> CliSkill:
    return CliSkill(SKILL_ID, [sys.executable, str(FAKE)], None, {})


def write_fake_media(path: str, duration: float = 3.0) -> str:
    Path(path).write_bytes(json.dumps({"fake": True, "duration": duration, "video": True, "channels": 2, "width": 640, "height": 360}).encode())
    return path


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.ws = os.path.join(self.tmp, "ws")
        self.src_dir = os.path.join(self.tmp, "src")
        os.makedirs(self.ws)
        os.makedirs(self.src_dir)
        self.video = write_fake_media(os.path.join(self.src_dir, "in.mp4"))
        self.logo = os.path.join(self.src_dir, "logo.png")
        Path(self.logo).write_bytes(b"\x89PNG fake")
        self.calls = os.path.join(self.tmp, "calls.jsonl")
        self._env = dict(os.environ)
        os.environ["FAKE_MG_CALLS"] = self.calls
        os.environ.pop("FAKE_MG_MODE", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def mode(self, m: str):
        os.environ["FAKE_MG_MODE"] = m

    def adapter(self, **kw) -> MotionGraphicsAdapter:
        kw.setdefault("workspace", self.ws)
        kw.setdefault("allowed_inputs", [self.src_dir])
        kw.setdefault("ffmpeg_skill_dir", "/opt/fake-ffmpeg-skill")
        return MotionGraphicsAdapter(fake_skill(), **kw)

    def paths(self, out="out/graded.mp4"):
        return {"vid": self.video, "logo": self.logo, "out": os.path.join(self.ws, out)}

    @staticmethod
    def elements(extra=None):
        els = [{"id": "t1", "type": "text_overlay", "start": 0.2, "end": 2.5, "parameters": {"text": "Hello", "position": "bottom"}, "animation": {"kind": "fade", "parameters": {"duration": 0.3}}},
               {"id": "title", "type": "title", "start": 0.0, "end": 1.5, "parameters": {"title": "Opening", "subtitle": "Day 1", "primary_color": "#1E3A8A"}}]
        return els + list(extra or [])

    def op(self, args=None, **over):
        a = {"input": "vid", "output": "out", "elements": self.elements()}
        a.update(args or {})
        a.update(over)
        return Operation(tool=TOOL_ID, args=a, inputs=["vid"], outputs=["out"], id="op-mg-1")

    def logged(self):
        return [json.loads(line) for line in Path(self.calls).read_text(encoding="utf-8").splitlines() if line.strip()]


class ContractTests(Base):
    def test_discovery(self):
        a = self.adapter()
        self.assertEqual(a.version, "0.1.0")
        self.assertEqual(a.tools, {TOOL_ID})
        self.assertTrue(a.supports(TOOL_ID))
        self.assertFalse(a.supports("motion-graphics/other"))
        self.assertEqual(sorted(a.element_types), ["image_overlay", "lower_third", "text_overlay", "title"])
        self.assertIn("shape", a.unsupported)
        self.assertEqual(a.drift(), [])
        self.assertEqual(check_contract(pinned_contract()), [])
        self.assertEqual(contract_drift(pinned_contract()), [])
        pkg = a.package()
        self.assertEqual(pkg.skill_id, SKILL_ID)
        self.assertEqual(pkg.tool_ids(), [TOOL_ID])
        self.assertEqual(pkg.repository, "kajisho5/motion-graphics-skill")
        self.assertIn("motion-graphics", pkg.capabilities)
        self.assertEqual(PACKAGE.version, "0.1.0")
        self.assertEqual(PACKAGE.tool(TOOL_ID).inputs, ["input", "output", "image"])
        self.assertTrue(PACKAGE.tool(TOOL_ID).produces_output)
        self.assertTrue(a.retryable["TOOL_ERROR"]); self.assertTrue(a.retryable["CANCELLED"]); self.assertFalse(a.retryable["INVALID_REQUEST"])
        self.assertEqual(a.exit_codes["CANCELLED"], 15)
        d = a.describe()
        self.assertEqual(d["drift"], [])
        self.assertIn("system:dejavu-sans", d["fonts"])

    def test_refusals(self):
        for m, frag in (("wrong_schema", "schema"), ("wrong_skill", "skill_id"), ("wrong_version", "version"), ("bad_contract", "execution.shell"), ("contract_fail", "failed")):
            self.mode(m)
            with self.assertRaises(ContractError) as cm:
                self.adapter()
            self.assertIn(frag, str(cm.exception), m)

    def test_drift_detected(self):
        self.mode("contract_drift")
        a = self.adapter()
        self.assertTrue(any("chapter" in d for d in a.drift()), a.drift())
        self.assertIn("chapter", a.element_types)

    def test_doctor(self):
        a = self.adapter()
        d = a.doctor()
        self.assertEqual(d["status"], "ok")
        st = MotionGraphicsAdapter.element_status(d)
        self.assertEqual(st["text_overlay"], "supported")
        self.assertEqual(st["title"], "unknown")   # never upgraded by the adapter
        self.assertEqual(MotionGraphicsAdapter.font_status(d)["system:dejavu-sans"], "supported")
        argv = self.logged()[-1]["argv"]
        self.assertEqual(argv[:2], ["doctor", "--json"])
        self.assertIn("--ffmpeg-skill", argv)
        self.mode("doctor_fail")
        d = a.doctor()
        self.assertEqual(d["status"], "fail")
        self.assertEqual(d["exit_code"], 1)
        self.mode("doctor_degraded")
        self.assertEqual(MotionGraphicsAdapter.element_status(a.doctor())["image_overlay"], "unavailable")

    def test_locate_env(self):
        self.assertIsNone(locate_motion_graphics(env={"PATH": ""}))
        root = os.path.join(self.tmp, "motion-graphics-skill")
        os.makedirs(os.path.join(root, "src", "motion_graphics"))
        Path(root, "src", "motion_graphics", "cli.py").write_text("", encoding="utf-8")
        s = locate_motion_graphics(env={ENV_DIR: root, "PATH": ""})
        self.assertIsNotNone(s)
        self.assertEqual(s.command[1:], ["-m", "motion_graphics.cli"])
        self.assertEqual(s.env["PYTHONPATH"], os.path.join(root, "src"))


class RequestTests(Base):
    def test_build(self):
        a = self.adapter()
        b = a.build_request(TOOL_ID, self.op().args, self.paths(), op_id="op-1", timeout=30)
        r = b["request"]
        self.assertEqual(r["schema"], "motion-graphics/request@1")
        self.assertEqual(r["video"]["path"], str(Path(self.video).resolve()))
        self.assertTrue(r["output"]["path"].endswith("graded.mp4"))
        self.assertTrue(r["output"]["overwrite"])
        self.assertEqual([e["id"] for e in r["elements"]], ["t1", "title"])
        self.assertEqual(r["elements"][0]["animation"], {"kind": "fade", "parameters": {"duration": 0.3}})
        self.assertNotIn("options", r)
        self.assertEqual(b["element_ids"], ["t1", "title"])
        argv = a._argv(b, dry_run=True, timeout=30)
        self.assertEqual(argv[:3], ["run", "-", "--json"])
        self.assertIn("--workspace", argv); self.assertIn("--allowed-input", argv); self.assertIn("--ffmpeg-skill", argv)
        self.assertEqual(argv[argv.index("--timeout") + 1], "30")
        self.assertEqual(argv[-1], "--dry-run")

    def test_options_font_position_image(self):
        a = self.adapter()
        els = [{"id": "i1", "type": "image_overlay", "start": 0, "end": 2, "parameters": {"image": "logo", "position": {"x": 10, "y": -20}, "scale_percent": 25.0}},
               {"id": "t2", "type": "text_overlay", "start": 0, "end": 1, "parameters": {"text": "x", "font": "system:dejavu-serif", "box": True, "box_color": "black@0.5"}}]
        r = a.build_request(TOOL_ID, {"input": "vid", "output": "out", "elements": els, "crf": 20, "preset": "fast"}, self.paths())["request"]
        self.assertEqual(r["options"], {"crf": 20, "preset": "fast"})
        self.assertEqual(r["elements"][0]["parameters"]["image_path"], str(Path(self.logo).resolve()))
        self.assertNotIn("image", r["elements"][0]["parameters"])
        self.assertEqual(r["elements"][0]["parameters"]["position"], {"x": 10, "y": -20})
        self.assertEqual(r["elements"][1]["parameters"]["font"], {"font_id": "system:dejavu-serif"})

    def test_refused(self):
        a = self.adapter()
        p = self.paths()

        def bad(args, frag, **kw):
            with self.assertRaises(ToolError) as cm:
                a.build_request(TOOL_ID, args, kw.get("paths", p))
            self.assertIn(frag, str(cm.exception))

        el = lambda **o: [dict({"id": "e", "type": "text_overlay", "start": 0, "end": 1, "parameters": {"text": "x"}}, **o)]  # noqa: E731
        base = {"input": "vid", "output": "out"}
        bad(dict(base, elements=self.elements(), filter="x"), "forbidden field")
        bad(dict(base, elements=el(parameters={"text": "x", "vf": "y"})), "forbidden field")
        bad(dict(base, elements=el(parameters={"text": "x", "command": ["ls"]})), "forbidden field")
        bad(dict(base, elements=self.elements(), timeout=5), "unknown argument")
        bad(dict(base, elements=el(parameters={"text": "x", "wobble": 1})), "not declared")
        bad(dict(base, elements=el(parameters={"text": "x", "image_path": self.logo})), "not declared")
        bad(dict(base, elements=el(parameters={})), "requires parameter 'text'")
        bad(dict(base, elements=el(parameters={"text": "x" * 501})), "longer than 500")
        bad(dict(base, elements=el(parameters={"text": "x", "font_size": 4})), "below the contract minimum")
        bad(dict(base, elements=el(parameters={"text": "x", "font_size": 301})), "above the contract maximum")
        bad(dict(base, elements=el(parameters={"text": "x", "font_size": 12.5})), "must be an integer")
        bad(dict(base, elements=el(parameters={"text": "x", "box": "yes"})), "must be a boolean")
        bad(dict(base, elements=el(parameters={"text": "x", "font_color": "red; rm -rf"})), "cannot be a colour")
        bad(dict(base, elements=el(parameters={"text": "x", "font_color": ""})), "non-empty")
        bad(dict(base, elements=el(parameters={"text": "x", "position": "middle"})), "not one of")
        bad(dict(base, elements=el(parameters={"text": "x", "position": {"x": 1.5, "y": 2}})), "named position")
        bad(dict(base, elements=el(parameters={"text": "x", "font": "/usr/share/fonts/x.ttf"})), "registry font id")
        bad(dict(base, elements=el(parameters={"text": "x", "font": {"font_file": "/x.ttf"}})), "non-empty string")
        bad(dict(base, elements=el(type="title", parameters={"title": "T"}, animation={"kind": "fade", "parameters": {"duration": 0.5}})), "does not apply to title")
        bad(dict(base, elements=el(animation={"kind": "slide", "parameters": {}})), "not implemented")
        bad(dict(base, elements=el(animation={"kind": "fade", "parameters": {"duration": 31}})), "above the contract maximum")
        bad(dict(base, elements=el(animation={"kind": "fade", "parameters": {}})), "requires 'duration'")
        bad(dict(base, elements=el(animation={"kind": "fade", "parameters": {"duration": 1, "ease": "in"}})), "not declared for fade")
        bad(dict(base, elements=el(start=1, end=1)), "0 <= start < end")
        bad(dict(base, elements=el(start=-1, end=1)), "0 <= start < end")
        bad(dict(base, elements=el(start=0, end=1e9)), "0 <= start < end")
        bad(dict(base, elements=el(start=float("nan"), end=1)), "finite number")
        bad(dict(base, elements=el(id="bad id!")), "must match")
        bad(dict(base, elements=el(type="shape")), "declared unsupported")
        bad(dict(base, elements=el(type="banner")), "unknown element type")
        bad(dict(base, elements=el(extra=1)), "unknown field")
        bad(dict(base, elements=el() + el()), "duplicate element id")
        bad(dict(base, elements=[]), "non-empty list")
        bad(dict(base, elements=el() * 65), "more than 64")
        bad(dict(base, elements=[{"id": "i", "type": "image_overlay", "start": 0, "end": 1, "parameters": {}}]), "requires parameter 'image'")
        bad(dict(base, elements=[{"id": "i", "type": "image_overlay", "start": 0, "end": 1, "parameters": {"image": "vid"}}]), "image must be an existing")
        bad(dict(base, elements=self.elements(), crf=52), "crf must be")
        bad(dict(base, elements=self.elements(), crf=True), "crf must be")
        bad(dict(base, elements=self.elements(), preset="turbo"), "preset must be")
        bad(dict(base, elements=self.elements()), "outside the workspace", paths=dict(p, out=os.path.join(self.tmp, "elsewhere.mp4")))
        bad(dict(base, elements=self.elements()), "must be a .mp4", paths=dict(p, out=os.path.join(self.ws, "o.mov")))
        outside = write_fake_media(os.path.join(self.tmp, "outside.mp4"))
        bad(dict(base, elements=self.elements()), "outside the allowed roots", paths=dict(p, vid=outside))
        bad(dict(base, elements=self.elements()), "not found", paths=dict(p, vid=os.path.join(self.src_dir, "nope.mp4")))
        outside_png = os.path.join(self.tmp, "outside.png")
        Path(outside_png).write_bytes(b"png")
        bad(dict(base, elements=[{"id": "i", "type": "image_overlay", "start": 0, "end": 1, "parameters": {"image": "logo"}}]), "outside the allowed roots", paths=dict(p, logo=outside_png))
        bad(dict(base, output="vid", elements=self.elements()), "outside the workspace")
        with self.assertRaises(ToolError):
            a.build_request("motion-graphics/other", dict(base, elements=self.elements()), p)

    def test_path_policy(self):
        a = self.adapter(allowed_inputs=[], path_policy=PathPolicy([self.src_dir], self.ws))
        a.build_request(TOOL_ID, self.op().args, self.paths())
        outside = write_fake_media(os.path.join(self.tmp, "o.mp4"))
        with self.assertRaises(ToolError):
            a.build_request(TOOL_ID, self.op().args, dict(self.paths(), vid=outside))
        with self.assertRaises(ToolError):
            a.build_request(TOOL_ID, self.op().args, dict(self.paths(), out=os.path.join(self.tmp, "x.mp4")))

    def test_preview_and_measure(self):
        a = self.adapter()
        lines = a.preview(self.op(), self.paths())
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("motion-graphics run - --json"))
        self.assertIn("--dry-run", lines[0])
        self.assertIn("refused", a.preview(self.op(elements=[]), self.paths())[0])
        with self.assertRaises(ToolError):
            a.measure(TOOL_ID, {})
        self.assertEqual(a.calls, 1)


class SuccessTests(Base):
    def test_run(self):
        a = self.adapter()
        r = a.run(self.op(), self.paths(), timeout=20)
        self.assertTrue(r.ok, r.data)
        self.assertEqual(r.exit_code, 0)
        self.assertEqual(r.output, self.paths()["out"])
        self.assertTrue(os.path.isfile(r.output))
        d = r.data
        self.assertEqual(d["skill"], {"id": SKILL_ID, "version": "0.1.0"})
        self.assertEqual(d["status"], "completed")
        self.assertEqual(d["operation_type"], "GRAPHICS")
        self.assertEqual(d["artifact"]["sha256"], sha256_file(r.output))
        self.assertEqual(d["artifact"]["size"], os.path.getsize(r.output))
        self.assertEqual(d["artifact"]["duration"], 3.0)
        self.assertFalse(d["artifact"]["reused"])
        self.assertEqual({t["id"] for t in d["timeline"]}, {"t1", "title"})
        self.assertEqual([o["element_id"] for o in d["operations"]], ["title", "t1"])   # (start, id) order
        for o in d["operations"]:
            self.assertEqual(o["status"], "rendered")
            self.assertTrue(o["tool"].startswith("ffmpeg-skill/"))
            self.assertEqual(set(o), {"element_id", "type", "tool", "status", "operation_id", "parameters", "input_hashes", "output_hash"})
        self.assertEqual(d["engine"]["ffmpeg-skill"], "0.9.1-fake")
        self.assertEqual(d["provenance"]["output_hash"], d["artifact"]["sha256"])
        self.assertEqual(d["provenance"]["video"]["sha256"], sha256_file(self.video))
        self.assertEqual(d["observation"]["source"], "ffmpeg-skill/probe@0.9.1-fake")
        self.assertEqual(d["observation"]["provenance"], "OBSERVED")
        self.assertEqual(d["observation"]["data"]["width"], 640)
        self.assertEqual(len(d["commands"]), 2)
        self.assertEqual(r.commands, d["commands"])
        self.assertEqual(d["warnings"], [])
        obs = lift_observation(r, "art-out")
        self.assertIsNotNone(obs)
        self.assertEqual(obs.asset_id, "art-out")
        self.assertEqual(obs.skill, SKILL_ID)
        self.assertEqual(obs.fingerprint, d["artifact"]["sha256"])
        self.assertEqual(obs.parameters["document_id"], d["provenance"]["document_id"])
        self.assertEqual(json.loads(Path(r.output).read_bytes())["graphics"], ["title", "t1"])

    def test_dry_run(self):
        a = self.adapter()
        r = a.run(self.op(), self.paths(), dry_run=True)
        self.assertTrue(r.ok, r.data)
        self.assertTrue(r.dry_run)
        self.assertIsNone(r.output)
        self.assertFalse(os.path.exists(self.paths()["out"]))
        self.assertEqual(r.data["status"], "dry_run")
        self.assertEqual({t["id"] for t in r.data["timeline"]}, {"t1", "title"})
        self.assertTrue(r.data["plan"]["document_id"])
        self.assertIn("--dry-run", self.logged()[-1]["argv"])
        self.assertIsNone(lift_observation(r))

    def test_reused_and_image(self):
        a = self.adapter()
        els = self.elements([{"id": "logo", "type": "image_overlay", "start": 0.5, "end": 2.0, "parameters": {"image": "logo", "opacity": 0.8}}])
        r = a.run(self.op(elements=els), self.paths())
        self.assertTrue(r.ok, r.data)
        self.assertEqual(r.data["provenance"]["assets"]["logo"]["sha256"], sha256_file(self.logo))
        self.mode("reused")
        r2 = a.run(self.op(elements=els), self.paths())
        self.assertTrue(r2.ok, r2.data)
        self.assertTrue(r2.data["artifact"]["reused"])
        self.assertEqual(r2.data["artifact"]["sha256"], r.data["artifact"]["sha256"])


class ErrorTests(Base):
    def check(self, mode, code, retryable, rc, removed=True, exit_code=None, timeout=None):
        self.mode(mode)
        a = self.adapter()
        r = a.run(self.op(), self.paths(), timeout=timeout)
        self.assertFalse(r.ok, mode)
        e = r.data["error"]
        self.assertEqual(e["code"], code, (mode, e))
        self.assertIs(e["retryable"], retryable, mode)
        self.assertEqual(e["recovery_class"], rc, mode)
        self.assertEqual(r.data["status"], "failed")
        self.assertEqual(r.data["skill"]["version"], "0.1.0")
        self.assertEqual(e["exit_code"], r.exit_code)
        if exit_code is not None:
            self.assertEqual(r.exit_code, exit_code, mode)
        if removed:
            self.assertFalse(os.path.exists(self.paths()["out"]), mode)
        return r

    def test_skill_errors(self):
        self.check("tool_error", "TOOL_ERROR", True, "UNKNOWN", exit_code=12)
        self.check("tool_error_final", "TOOL_ERROR", False, "UNKNOWN", exit_code=12)
        self.check("validation_error", "VALIDATION_ERROR", False, "SKILL_ERROR", exit_code=14)
        r = self.check("cancelled", "CANCELLED", True, "TIMEOUT", exit_code=15)
        self.assertEqual(r.data["error"]["details"]["reason"], "signal")
        r = self.check("skill_timeout", "CANCELLED", True, "TIMEOUT", exit_code=15)
        self.assertEqual(r.data["error"]["details"]["reason"], "timeout")
        self.check("internal_error", "INTERNAL_ERROR", False, "SKILL_ERROR", exit_code=16)
        self.check("duplicate_ids", "DEPENDENCY_ERROR", False, "INVALID_ARGS", exit_code=11)
        r = self.check("unknown_font", "MISSING_INPUT", False, "INPUT_MISSING", exit_code=7)
        self.assertEqual(r.data["error"]["details"]["font_id"], "system:dejavu-sans")

    def test_invalid_results(self):
        for mode in ("malformed", "empty", "two_docs", "text", "nonzero_ok", "unknown_code", "output_missing", "hash_mismatch", "no_provenance", "timeline_mismatch"):
            self.check(mode, "INVALID_RESULT", False, "SKILL_ERROR")
            self.tearDown(); self.setUp()
        r = self.check("hash_mismatch", "INVALID_RESULT", False, "SKILL_ERROR")
        self.assertIn("sha256", r.data["error"]["message"])
        r = self.check("nonzero_ok", "INVALID_RESULT", False, "SKILL_ERROR", exit_code=3)
        self.assertIn("exit code 3", r.data["error"]["message"])

    def test_timeout(self):
        r = self.check("timeout", "CANCELLED", True, "TIMEOUT", exit_code=124, timeout=1)
        self.assertEqual(r.data["error"]["details"]["reason"], "timeout")
        self.assertLess(r.seconds, 20)

    def test_build_failure_is_invalid_request(self):
        a = self.adapter()
        r = a.run(self.op(elements=[]), self.paths())
        self.assertFalse(r.ok)
        self.assertEqual(r.data["error"]["code"], "INVALID_REQUEST")
        self.assertEqual(r.data["error"]["recovery_class"], "INVALID_ARGS")
        self.assertEqual(r.exit_code, 2)
        self.assertEqual(a.calls, 1)   # the contract fetch only: nothing was sent to the Skill

    def test_unknown_font_id_refused_locally(self):
        a = self.adapter()
        r = a.run(self.op(elements=[{"id": "e", "type": "text_overlay", "start": 0, "end": 1, "parameters": {"text": "x", "font": "system:comic"}}]), self.paths())
        self.assertEqual(r.data["error"]["code"], "INVALID_REQUEST")


class SecurityTests(Base):
    def test_argv_and_request_hygiene(self):
        os.environ["VIDEO_AGENT_SECRET_TOKEN"] = "do-not-leak"
        a = self.adapter()
        r = a.run(self.op(), self.paths())
        self.assertTrue(r.ok, r.data)
        calls = self.logged()
        self.assertEqual([c["cmd"] for c in calls], ["skill", "run"])
        run = calls[-1]
        self.assertIsInstance(run["argv"], list)
        for tok in run["argv"]:
            self.assertNotIn("Hello", tok); self.assertNotIn("Opening", tok); self.assertNotIn("elements", tok)
        self.assertEqual(run["argv"][:3], ["run", "-", "--json"])
        self.assertEqual(run["argv"][run["argv"].index("--ffmpeg-skill") + 1], "/opt/fake-ffmpeg-skill")
        self.assertNotIn("VIDEO_AGENT_MOTION_GRAPHICS_DIR", run["env_video"])
        self.assertEqual({k for k in run["env_video"] if "SECRET" in k}, {"VIDEO_AGENT_SECRET_TOKEN"})   # the parent's environment is inherited as-is, never added to
        b = a.build_request(TOOL_ID, self.op().args, self.paths())
        text = json.dumps(b["request"]).lower()
        for k in a.forbidden:
            if k not in ("path", "paths", "workspace"):
                self.assertNotIn(f'"{k}"', text, k)
        for k in ("command", "argv", "filter", "shell", "exec", "env", "cwd", "script"):
            self.assertIn(k, a.forbidden)
        self.assertNotIn("api_key", text)

    def test_never_shell(self):
        a = self.adapter()
        r = a.run(self.op(elements=[{"id": "e", "type": "text_overlay", "start": 0, "end": 1, "parameters": {"text": "$(touch pwned); `id`"}}]), self.paths())
        self.assertTrue(r.ok, r.data)
        self.assertFalse(os.path.exists("pwned"))
        self.assertEqual(self.logged()[-1]["argv"][1], "-")


@unittest.skipUnless(os.environ.get(ENV_DIR), f"set {ENV_DIR} to run against the real motion-graphics-skill")
class RealSkillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.ws = os.path.join(self.tmp, "ws")
        os.makedirs(self.ws)
        self.video = os.path.join(self.tmp, "src", "bars.mp4")
        os.makedirs(os.path.dirname(self.video))
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not on PATH")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "3",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", self.video], check=True)
        self.logo = os.path.join(self.tmp, "src", "logo.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:size=64x64", "-frames:v", "1", self.logo], check=True)
        self.skill = locate_motion_graphics()
        self.assertIsNotNone(self.skill)
        self.adapter = MotionGraphicsAdapter(self.skill, workspace=self.ws, allowed_inputs=[os.path.dirname(self.video)], ffmpeg_skill_dir=os.environ.get("VIDEO_AGENT_FFMPEG_SKILL_DIR"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_real(self):
        self.assertEqual(self.adapter.drift(), [])
        d = self.adapter.doctor()
        self.assertIn(d["status"], ("ok", "degraded"))
        st = MotionGraphicsAdapter.element_status(d)
        self.assertEqual(st["text_overlay"], "supported")
        paths = {"vid": self.video, "logo": self.logo, "out": os.path.join(self.ws, "out", "mg.mp4")}
        els = [{"id": "t1", "type": "text_overlay", "start": 0.2, "end": 2.5, "parameters": {"text": "Hello MG", "position": "bottom"}, "animation": {"kind": "fade", "parameters": {"duration": 0.3}}},
               {"id": "logo", "type": "image_overlay", "start": 0.0, "end": 3.0, "parameters": {"image": "logo", "position": "top-right", "scale_width": 32}}]
        op = Operation(tool=TOOL_ID, args={"input": "vid", "output": "out", "elements": els, "crf": 28, "preset": "ultrafast"}, inputs=["vid"], outputs=["out"], id="op-real")
        r = self.adapter.run(op, paths, dry_run=True)
        self.assertTrue(r.ok, r.data)
        r = self.adapter.run(op, paths, timeout=300)
        self.assertTrue(r.ok, r.data)
        self.assertEqual(r.data["artifact"]["sha256"], sha256_file(r.output))
        self.assertEqual(r.data["artifact"]["width"], 640)
        self.assertAlmostEqual(float(r.data["artifact"]["duration"]), 3.0, delta=0.3)
        self.assertEqual({o["element_id"] for o in r.data["operations"]}, {"t1", "logo"})
        self.assertTrue(all(o["status"] in ("rendered", "reused") for o in r.data["operations"]))
        self.assertEqual(r.data["provenance"]["assets"]["logo"]["sha256"], hashlib.sha256(Path(self.logo).read_bytes()).hexdigest())
        self.assertIsNotNone(lift_observation(r))
        bad = Operation(tool=TOOL_ID, args={"input": "vid", "output": "out", "elements": [dict(els[0], parameters={"text": "x", "font_color": "notacolour123"})]}, inputs=["vid"], outputs=["out"], id="op-bad")
        rb = self.adapter.run(bad, paths)
        self.assertFalse(rb.ok)
        self.assertEqual(rb.data["error"]["code"], "INVALID_REQUEST")


if __name__ == "__main__":
    unittest.main()
