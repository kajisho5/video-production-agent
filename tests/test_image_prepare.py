"""ADR-077 (imagemagick-skill's roadmap step 4, agent side): an `edit.overlay` image prepared by imagemagick-skill -- GPS / EXIF stripped,
resized to the user's box, padded to that canvas -- before video-editing composites it. Against a fake imagemagick-skill checkout
(tests/fake_imagemagick.py, launched exactly as the real scripts are) that speaks the pinned 0.8.1 contract, its doctor and its result
shapes, with the fake media engine and the fake video-editing process. No ImageMagick, ffmpeg or network."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter  # noqa: E402
import fake_imagemagick as fim  # noqa: E402
from video_agent.agent import image_prepare as ip  # noqa: E402
from video_agent.agent.editing import EditRequirementError, parse_edit_requirements  # noqa: E402
from video_agent.capabilities.resolver import Capability, CapabilityResolver  # noqa: E402
from video_agent.execution.compiler import CompileError, compile_ir, lower_image_prepare  # noqa: E402
from video_agent.models import Operation, Requirement  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.project.ir import ProjectIR  # noqa: E402
from video_agent.project.validator import validate_ir  # noqa: E402
from video_agent.service import BRIEF_REQUIREMENT_KEYS, Service, _check_requirement_keys  # noqa: E402
from video_agent.tools import ToolRouter  # noqa: E402
from video_agent.tools.imagemagick import (FATAL_DRIFT_KEYS, TOOL_PAD, TOOL_RESIZE, TOOL_STRIP, ContractError, ImageMagickAdapter, breaking_drift,  # noqa: E402
                                           check_contract, drift, locate_imagemagick, pinned_contract, recovery_class)
from video_agent.tools.imagemagick.adapter import PINNED_CONTRACT_PATH  # noqa: E402
from video_agent.tools.video_editing import VideoEditingAdapter, VideoEditingSkill  # noqa: E402

MODES = ("FAKE_IM_MODE", "FAKE_IM_BACKEND", "FAKE_IM_CONTRACT", "FAKE_IM_CALLS", "VIDEO_AGENT_IMAGEMAGICK_DIR", "FAKE_VE_MODE", "FAKE_VE_CALLS")
FAKE_VE = str(Path(__file__).resolve().parent / "fake_video_editing.py")
IM_CAPS = ["imagemagick-skill", "imagemagick-skill:RESIZE", "imagemagick-skill:PAD", "imagemagick-skill:STRIP"]
PREP = {"edit.overlay.prepare.width": 320, "edit.overlay.prepare.height": 180}


class Caps:
    def __init__(self, extra=(), missing=()):
        self.extra, self.missing = list(extra), set(missing)

    def resolve(self, refresh=False):
        names = ["python", "ffmpeg", "ffprobe", "ffmpeg-skill", "encoder:libx264", "encoder:aac", "filter:loudnorm", "font:cjk-ja", "video-editing"] + self.extra
        return {n: Capability(n, "MISSING" if n in self.missing else "AVAILABLE", "fake", {"version": "0.8.4-fake"} if n == "ffmpeg-skill" else {}) for n in names}


def reqs(d):
    return {k: Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in d.items()}


class _Base(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in MODES}
        for k in MODES:
            os.environ.pop(k, None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.log = str(Path(self.tmp) / "im_calls.log")
        os.environ["FAKE_IM_CALLS"] = self.log
        self.root = fim.make_checkout(str(Path(self.tmp) / "imagemagick-skill"))
        self.src = Path(self.tmp) / "src"
        self.src.mkdir(exist_ok=True)
        self.png = fim.image(str(self.src / "logo.png"), 1200, 600)
        self.video = str(self.src / "a.mp4")
        Path(self.video).write_bytes(b"\x00" * 64)
        self.ws = str(Path(self.tmp) / "ws")
        os.makedirs(self.ws, exist_ok=True)

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def adapter(self, **kw):
        return ImageMagickAdapter(locate_imagemagick(self.root), workspace=self.ws, allowed_inputs=[str(self.src)], **kw)

    def calls(self, scripts=("resize", "pad", "strip")):
        lines = Path(self.log).read_text(encoding="utf-8").splitlines() if os.path.exists(self.log) else []
        return [c for c in (json.loads(x) for x in lines) if c[0] in scripts]

    def service(self, installed=True, backend_caps=None):
        ve = VideoEditingAdapter(VideoEditingSkill([sys.executable, FAKE_VE], None, {}), workspace=self.ws, allowed_inputs=[str(self.src)], ffmpeg_skill_dir=self.tmp)
        router = [FakeAdapter(), ve] + ([self.adapter()] if installed else [])
        return Service(workspace=self.ws, adapter=ToolRouter(router), caps=Caps(extra=(backend_caps if backend_caps is not None else IM_CAPS) if installed else []))


class ContractTests(_Base):
    def test_pin_shape_and_contract_checks(self):
        text = PINNED_CONTRACT_PATH.read_text(encoding="utf-8")
        c = json.loads(text)
        self.assertEqual(text, json.dumps(c, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", "one line, keys sorted (the ingest / voice-enhance pin format)")
        self.assertEqual((c["skill"]["id"], c["skill"]["version"], c["contract_version"]), ("imagemagick-skill", "0.8.1", "1.0"))
        self.assertEqual(check_contract(c), [])
        self.assertEqual(drift(c), [])
        self.assertEqual(breaking_drift(c), [])
        bad = json.loads(text)
        bad["execution"]["shell"] = True
        next(t for t in bad["tools"] if t["name"] == "strip")["mutates_input"] = True
        bad["provides"] = [p for p in bad["provides"] if p["id"] != "image.pad"]
        errs = check_contract(bad)
        self.assertTrue(any("execution" in e for e in errs) and any("imagemagick-skill/strip" in e for e in errs) and any("image.pad" in e for e in errs), errs)
        self.assertIn("execution", FATAL_DRIFT_KEYS)

    def test_drift_classification(self):
        """A version inside 0.8.x or a reworded description is reported, not fatal; a renamed flag, a narrowed enum or a newly required
        parameter the agent does not send is fatal (the tool would be refused by argparse or misread); 0.9.0 is refused outright."""
        for mode, fatal in (("drift_version", False), ("drift_description", False), ("drift_flag", True), ("drift_enum", True), ("drift_required", True)):
            os.environ["FAKE_IM_CONTRACT"] = mode
            live = fim._contract()
            self.assertTrue(drift(live), mode)
            self.assertEqual(bool(breaking_drift(live)), fatal, (mode, breaking_drift(live)))
        os.environ["FAKE_IM_CONTRACT"] = "drift_flag"
        with self.assertRaises(ContractError):   # a flag the adapter sends is spelled differently: incompatible, never patched
            self.adapter()
        os.environ["FAKE_IM_CONTRACT"] = "incompatible"
        with self.assertRaisesRegex(ContractError, "not in supported range"):
            self.adapter()
        os.environ["FAKE_IM_CONTRACT"] = "drift_version"
        ad = self.adapter()
        self.assertTrue(ad.drift())
        self.assertEqual(ad.breaking_drift(), [])

    def test_locate_checkout_and_env(self):
        self.assertEqual(str(locate_imagemagick(self.root).root), self.root)
        self.assertEqual(locate_imagemagick(self.root).script("resize")[1], str(Path(self.root) / "scripts" / "resize.py"))
        self.assertEqual(str(locate_imagemagick(env={"VIDEO_AGENT_IMAGEMAGICK_DIR": self.root}).root), self.root)
        self.assertIsNone(locate_imagemagick(env={"VIDEO_AGENT_IMAGEMAGICK_DIR": str(self.src)}), "the env var is authoritative: no checkout there is the answer")
        cwd = os.getcwd()
        try:   # never the current directory (ADR-069): a checkout-looking `scripts/` there is not a Skill
            os.chdir(self.root)
            os.makedirs(Path(self.tmp) / "home", exist_ok=True)
            from unittest import mock
            with mock.patch("video_agent.tools.imagemagick.locate.default_skill_dirs", return_value=[Path(self.tmp) / "home" / "nope"]):
                self.assertIsNone(locate_imagemagick(env={}))
        finally:
            os.chdir(cwd)


class CapabilityTests(_Base):
    def caps(self, backend):
        os.environ["FAKE_IM_BACKEND"] = backend
        caps: dict = {}
        CapabilityResolver._finishing_skill(CapabilityResolver.__new__(CapabilityResolver), caps, "imagemagick-skill", locate_imagemagick(self.root), lambda _sk: self.adapter(),
                                            "VIDEO_AGENT_IMAGEMAGICK_DIR", "imagemagick-skill", "npx imagemagick-skill",
                                            per_item=lambda ad, doc: (ad.operation_status(doc), {}), item_evidence=lambda ad, doc: ad.operation_evidence(doc))
        return caps

    def test_doctor_verdicts_per_tool(self):
        caps = self.caps("magick")
        self.assertEqual([caps[c].status for c in IM_CAPS], ["AVAILABLE"] * 4)
        self.assertEqual(caps["imagemagick-skill:RESIZE"].evidence["backend"], "magick")
        caps = self.caps("sips")   # sips resizes and pads, never strips: the package works, strip does not
        self.assertEqual((caps["imagemagick-skill"].status, caps["imagemagick-skill:RESIZE"].status, caps["imagemagick-skill:STRIP"].status), ("AVAILABLE", "AVAILABLE", "MISSING"))
        self.assertEqual(caps["imagemagick-skill:STRIP"].evidence["code"], "BACKEND_UNAVAILABLE")
        self.assertIn("ImageMagick", caps["imagemagick-skill:STRIP"].detail)
        caps = self.caps("none")
        self.assertEqual([caps[c].status for c in IM_CAPS], ["MISSING"] * 4)
        os.environ["FAKE_IM_CONTRACT"] = "drift_enum"
        self.assertEqual(self.caps("magick")["imagemagick-skill"].status, "MISSING", "a fatal drift never leaves the package AVAILABLE")


class AdapterTests(_Base):
    def run_op(self, tool, args, ad=None, **kw):
        paths = {"img": self.png, "out": str(Path(self.ws) / "ops" / "x" / "out.png"), "out2": str(Path(self.ws) / "ops" / "y" / "out2.png")}
        return (ad or self.adapter()).run(Operation(tool=tool, args=args, inputs=["img"], outputs=["out"], id="op_t"), paths, **kw)

    def test_argv_is_flag_equals_value_and_the_input_after_double_dash(self):
        r = self.run_op(TOOL_RESIZE, {"input": "img", "output": "out", "width": 320, "height": 180, "mode": "fit"})
        self.assertTrue(r.ok, r.data)
        [c] = self.calls()
        self.assertEqual(c[0], "resize")
        pos = c.index("--")
        self.assertEqual(c[pos + 1:], [self.png])
        self.assertTrue(all(a.startswith("--") for a in c[1:pos]))
        self.assertIn("--width=320", c); self.assertIn("--height=180", c); self.assertIn("--mode=fit", c); self.assertIn("--json", c); self.assertIn("--overwrite", c)
        self.assertTrue(any(a.startswith("--output=") and a.endswith("out.png") for a in c))
        self.assertTrue(any(a.startswith("--timeout=") for a in c))
        self.run_op(TOOL_PAD, {"input": "img", "output": "out", "width": 320, "height": 180, "color": "#000000AA", "gravity": "southeast"})
        self.assertIn("--color=#000000AA", self.calls()[-1]); self.assertIn("--gravity=southeast", self.calls()[-1])
        self.run_op(TOOL_STRIP, {"input": "img", "output": "out"})
        self.assertEqual(self.calls()[-1][0], "strip")

    def test_refusals_before_the_process_starts(self):
        outside = fim.image(str(Path(self.tmp) / "elsewhere" / "x.png"), 10, 10)
        cases = [
            (TOOL_RESIZE, {"input": "img", "output": "out", "width": 320}, "needs height"),
            (TOOL_RESIZE, {"input": "img", "output": "out", "width": 320.5, "height": 180}, "integer"),
            (TOOL_RESIZE, {"input": "img", "output": "out", "width": 0, "height": 180}, "within"),
            (TOOL_RESIZE, {"input": "img", "output": "out", "width": True, "height": 180}, "integer"),
            (TOOL_RESIZE, {"input": "img", "output": "out", "width": 10, "height": 10, "mode": "stretch"}, "one of fit, fill, exact"),
            (TOOL_PAD, {"input": "img", "output": "out", "width": 10, "height": 10, "color": "rgb(1,2,3)"}, "color"),
            (TOOL_PAD, {"input": "img", "output": "out", "width": 10, "height": 10, "color": "--shell"}, "color"),
            (TOOL_PAD, {"input": "img", "output": "out", "width": 10, "height": 10, "color": "red", "gravity": "middle"}, "one of"),
            (TOOL_RESIZE, {"input": "img", "output": "out", "width": 10, "height": 10, "command": "magick x"}, "forbidden key"),
            (TOOL_RESIZE, {"input": "img", "output": "out", "width": 10, "height": 10, "quality": 90}, "unknown"),
            (TOOL_STRIP, {"input": "nope", "output": "out"}, "not an artifact id"),
            (TOOL_STRIP, {"input": "img", "output": "img"}, "outside the workspace"),
            ("imagemagick-skill/optimize", {"input": "img", "output": "out"}, "does not support"),
        ]
        for tool, args, why in cases:
            r = self.run_op(tool, args)
            self.assertFalse(r.ok, (tool, args))
            self.assertEqual((r.data["error"]["code"], r.data["error"]["recovery_class"]), ("INVALID_REQUEST", "INVALID_ARGS"))
            self.assertIn(why, r.data["error"]["message"], (args, r.data["error"]["message"]))
        ad = self.adapter()
        for paths, why in (({"img": outside, "out": str(Path(self.ws) / "o.png")}, "outside the allowed input roots"),
                           ({"img": str(self.src / ".." / "src" / "logo.png"), "out": str(Path(self.ws) / "o.png")}, "traversal"),
                           ({"img": self.video, "out": str(Path(self.ws) / "o.png")}, "PNG / JPEG"),
                           ({"img": self.png, "out": str(Path(self.ws) / "o.mp4")}, ".png / .jpg"),
                           ({"img": self.png, "out": str(Path(self.tmp) / "o.png")}, "outside the workspace")):
            r = ad.run(Operation(tool=TOOL_STRIP, args={"input": "img", "output": "out"}, inputs=["img"], outputs=["out"]), paths)
            self.assertIn(why, r.data["error"]["message"])
        self.assertEqual(self.calls(), [], "nothing reached the Skill")

    def test_run_admit_and_record(self):
        r = self.run_op(TOOL_RESIZE, {"input": "img", "output": "out", "width": 320, "height": 320, "mode": "fit"})
        self.assertTrue(r.ok, r.data)
        self.assertEqual((r.data["op"], r.data["capability"], r.data["lifecycle"], r.data["backend"]), ("resize", "image.resize", "EXPERIMENTAL", "magick"))
        self.assertEqual((r.data["output"]["width"], r.data["output"]["height"]), (320, 160), "fit: 1200x600 inside 320x320")
        import hashlib
        self.assertEqual(r.data["output"]["sha256"], hashlib.sha256(Path(r.output).read_bytes()).hexdigest())
        self.assertEqual(r.data["input_sha256"], hashlib.sha256(Path(self.png).read_bytes()).hexdigest())
        self.assertEqual(r.data["parameters"], {"width": 320, "height": 320, "mode": "fit"})
        from video_agent.tools.imagemagick import lift_observation
        obs = lift_observation(r, "a")
        self.assertEqual((obs.kind, obs.provenance, obs.skill, obs.skill_version, obs.fingerprint), ("image.resize", "OBSERVED", "imagemagick-skill", "0.8.1", r.data["output"]["sha256"]))
        dry = self.run_op(TOOL_STRIP, {"input": "img", "output": "out2"}, dry_run=True)
        self.assertTrue(dry.ok and dry.dry_run)
        self.assertEqual(len(self.calls()), 1, "the dry run checks the request and runs nothing")
        self.assertFalse(os.path.exists(str(Path(self.ws) / "ops" / "y" / "out2.png")))

    def test_failures_and_unadmitted_results(self):
        out = str(Path(self.ws) / "ops" / "x" / "out.png")
        for mode, code, cls, retry in (("fail", "SKILL_FAILED", "SKILL_ERROR", False), ("timeout", "TIMEOUT", "TIMEOUT", True), ("wrong_size", "INVALID_OUTPUT", "SKILL_ERROR", False),
                                       ("wrong_output", "INVALID_OUTPUT", "SKILL_ERROR", False), ("bad_json", "INVALID_OUTPUT", "SKILL_ERROR", False),
                                       ("touch_input", "INPUT_CHANGED", "SKILL_ERROR", False), ("exit_nonzero", "INVALID_OUTPUT", "SKILL_ERROR", False)):
            os.environ["FAKE_IM_MODE"] = mode
            fim.image(self.png, 1200, 600)
            r = self.run_op(TOOL_RESIZE, {"input": "img", "output": "out", "width": 320, "height": 180, "mode": "exact"})
            self.assertFalse(r.ok, mode)
            self.assertEqual((r.data["error"]["code"], r.data["error"]["recovery_class"], r.data["error"]["retryable"]), (code, cls, retry), (mode, r.data["error"]))
            self.assertFalse(os.path.exists(out), f"{mode}: a refused output is removed")
        os.environ["FAKE_IM_MODE"] = "gps"
        fim.image(self.png, 1200, 600)
        r = self.run_op(TOOL_STRIP, {"input": "img", "output": "out"})
        self.assertEqual(r.data["error"]["code"], "INVALID_OUTPUT"); self.assertIn("has_gps", r.data["error"]["message"])
        self.assertEqual(recovery_class("SOMETHING_NEW"), "SKILL_ERROR")

    def test_children_never_import_from_the_current_directory(self):
        """ADR-074 for the script form: the scripts run as `python <checkout>/scripts/<tool>.py`, so the folder the agent runs in is not on
        their sys.path; a planted json.py / runpy.py there is never imported (it would write a marker)."""
        work = Path(self.tmp) / "untrusted"
        work.mkdir()
        marker = Path(self.tmp) / "planted_ran"
        for mod in ("json", "runpy", "_common"):
            (work / f"{mod}.py").write_text(f"open({str(marker)!r}, 'a').write({mod!r})\n", encoding="utf-8")
        cwd = os.getcwd()
        try:
            os.chdir(work)
            ad = self.adapter()
            r = self.run_op(TOOL_RESIZE, {"input": "img", "output": "out", "width": 32, "height": 32, "mode": "fill"}, ad=ad)
        finally:
            os.chdir(cwd)
        self.assertTrue(r.ok, r.data)
        self.assertFalse(marker.exists(), marker.read_text() if marker.exists() else "")
        # the control: run from that folder with the current directory on sys.path (python -c), the planted json is what gets imported
        subprocess.run([sys.executable, "-c", "import json"], cwd=str(work), check=True)
        self.assertTrue(marker.exists(), "control: the planted module is importable from the current directory")


class RequirementTests(_Base):
    def test_vocabulary_is_the_pinned_contract(self):
        c = pinned_contract()
        tools = {t["name"]: t for t in c["tools"]}
        mode = tools["resize"]["input_schema"]["properties"]["mode"]
        grav = tools["pad"]["input_schema"]["properties"]["gravity"]
        self.assertEqual((ip.RESIZE_MODES, ip.RESIZE_MODE_DEFAULT), (tuple(mode["enum"]), mode["default"]))
        self.assertEqual((ip.GRAVITIES, ip.GRAVITY_DEFAULT), (tuple(grav["enum"]), grav["default"]))
        self.assertEqual(ip.CONTRACT_PATH.resolve(), PINNED_CONTRACT_PATH.resolve(), "the agent reads the file the adapter pins")
        for key, (tool, param) in ip.KEY_PARAMETERS.items():
            self.assertIn(param, tools[tool]["input_schema"]["properties"], key)
        self.assertEqual(tools["resize"]["input_schema"]["properties"]["width"]["type"], "integer")
        schema = json.loads((Path(__file__).resolve().parents[1] / "schemas" / "project.schema.json").read_text(encoding="utf-8"))["$defs"]["video_op"]["properties"]["prepare"]
        self.assertEqual((schema["properties"]["mode"]["enum"], schema["properties"]["gravity"]["enum"]), (list(mode["enum"]), list(grav["enum"])), "the IR schema follows the contract")

    def test_parse_and_refusals(self):
        def parse(**kw):
            return parse_edit_requirements(reqs({"edit.overlay": self.png, **kw}))["video.overlay"]["params"]
        self.assertNotIn("prepare", parse(), "no prepare key, no preparation")
        self.assertEqual(parse(**PREP)["prepare"], {"width": 320, "height": 180, "mode": "fit"}, "the contract's default mode is recorded")
        self.assertEqual(parse(**{**PREP, "edit.overlay.prepare.pad": "none", "edit.overlay.prepare.strip": "true"})["prepare"],
                         {"width": 320, "height": 180, "mode": "fit", "pad": "none", "gravity": "center", "strip": True})
        self.assertEqual(parse(**{"edit.overlay.prepare.strip": True})["prepare"], {"strip": True})
        self.assertNotIn("prepare", parse(**{"edit.overlay.prepare.strip": False}))
        self.assertEqual(parse(**{**PREP, "edit.overlay.prepare.width": "640", "edit.overlay.prepare.mode": "fill"})["prepare"], {"width": 640, "height": 180, "mode": "fill"})
        for kw, why in (({"edit.overlay.prepare.width": 320}, "go together"), ({**PREP, "edit.overlay.prepare.width": 320.5}, "integer"),
                        ({**PREP, "edit.overlay.prepare.height": 0}, "within"), ({**PREP, "edit.overlay.prepare.mode": "stretch"}, "one of fit, fill, exact"),
                        ({"edit.overlay.prepare.mode": "fit"}, "needs"), ({**PREP, "edit.overlay.prepare.mode": "fill", "edit.overlay.prepare.pad": "black"}, "mode fit"),
                        ({"edit.overlay.prepare.pad": "black"}, "mode fit"), ({**PREP, "edit.overlay.prepare.pad": "rgb(0,0,0)"}, "#RRGGBB"),
                        ({**PREP, "edit.overlay.prepare.gravity": "north"}, "needs edit.overlay.prepare.pad"),
                        ({**PREP, "edit.overlay.prepare.pad": "red", "edit.overlay.prepare.gravity": "middle"}, "one of"),
                        ({"edit.overlay.prepare.strip": "maybe"}, "true or false")):
            with self.assertRaisesRegex(EditRequirementError, why):
                parse(**kw)
        with self.assertRaisesRegex(EditRequirementError, "ambiguous"):
            parse_edit_requirements(reqs(PREP))
        _check_requirement_keys({"edit.overlay": self.png, **PREP, "edit.overlay.prepare.pad": "white", "edit.overlay.prepare.gravity": "north"})
        with self.assertRaises(ValueError):
            _check_requirement_keys({"edit.overlay": self.png, "edit.overlay.prepare.colour": "red"})
        self.assertIn("edit.overlay.prepare.width", BRIEF_REQUIREMENT_KEYS)
        self.assertNotIn("edit.overlay", BRIEF_REQUIREMENT_KEYS, "the image itself is only ever named by the user")

    def test_chain_and_lowering(self):
        self.assertEqual([s for s, _t, _p in ip.prepare_chain({"width": 1, "height": 2, "mode": "fit", "pad": "red", "gravity": "north", "strip": True})],
                         ["image_strip", "image_prepare", "image_pad"], "strip first (orientation baked in), then resize, then pad")
        self.assertEqual(lower_image_prepare(TOOL_RESIZE, "resize", {"width": 1, "height": 2, "mode": "fit"}, "i", "o"), {"input": "i", "width": 1, "height": 2, "mode": "fit", "output": "o"})
        with self.assertRaises(CompileError):
            lower_image_prepare(TOOL_PAD, "resize", {}, "i", "o")
        self.assertEqual(ip.prepare_problems({"width": 1, "height": 2, "mode": "fit"}), [])
        self.assertTrue(ip.prepare_problems({"width": 1}))
        self.assertTrue(ip.prepare_problems({"width": 1, "height": 2, "mode": "fill", "pad": "red"}))
        self.assertTrue(ip.prepare_problems({"strip": False}))
        self.assertTrue(ip.prepare_problems({"argv": ["x"], "strip": True}))


class PipelineTests(_Base):
    def plan(self, svc, extra):
        return svc.plan([self.video], "generic", user_requirements={"edit.overlay": self.png, "edit.overlay.position": "top-right", **extra})

    def test_plan_selects_imagemagick_resize_for_an_overlay_that_needs_resizing(self):
        """Done when (imagemagick-skill roadmap step 4): the plan step of an overlay asked at a size selects imagemagick-skill/resize."""
        svc = self.service()
        ir = self.plan(svc, PREP)
        d = ir.doc
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "image_prepare")
        self.assertEqual(step["tool"], "imagemagick-skill/resize")
        self.assertEqual(step["params"], {"asset": step["params"]["asset"], "image": os.path.abspath(self.png), "width": 320, "height": 180, "mode": "fit"})
        overlay = next(s for s in d["plan"]["steps"] if s["skill"] == "video_overlay")
        self.assertIn(step["id"], overlay["depends_on"])
        self.assertEqual(step["decision_id"], overlay["decision_id"], "the preparation is part of the overlay decision (same request, same approval)")
        op = next(o for o in d["video"]["operations"] if o["type"] == "video.overlay")
        self.assertEqual((op["image"], op["prepare"]), (os.path.abspath(self.png), {"width": 320, "height": 180, "mode": "fit"}))
        dec = next(x for x in d["decisions"] if x["subject"] == "video.overlay")
        self.assertIn("resized to 320x180 (fit)", dec["decision"]); self.assertIn("imagemagick-skill", dec["reason"])
        self.assertEqual(svc.validate(ir).errors, [])
        self.assertEqual(svc.handoffs(ir), [])

    def test_no_prepare_keys_plans_exactly_as_before(self):
        svc = self.service()
        d = self.plan(svc, {}).doc
        self.assertFalse([s for s in d["plan"]["steps"] if s["skill"].startswith("image_")])
        self.assertNotIn("prepare", next(o for o in d["video"]["operations"] if o["type"] == "video.overlay"))
        plain = self.service(installed=False).plan([self.video], "generic", user_requirements={"edit.overlay": self.png, "edit.overlay.position": "top-right"}).doc
        self.assertEqual([s["skill"] for s in plain["plan"]["steps"]], [s["skill"] for s in d["plan"]["steps"]], "imagemagick-skill installed or not changes nothing")
        self.assertEqual(plain["plan"]["status"], d["plan"]["status"])

    def test_strip_resize_pad_then_overlay_end_to_end(self):
        svc = self.service()
        ir = self.plan(svc, {**PREP, "edit.overlay.prepare.pad": "none", "edit.overlay.prepare.gravity": "east", "edit.overlay.prepare.strip": True})
        d = ir.doc
        self.assertEqual([s["skill"] for s in d["plan"]["steps"] if s["skill"].startswith(("image_", "video_overlay"))], ["image_strip", "image_prepare", "image_pad", "video_overlay"])
        self.assertEqual([s["tool"] for s in d["plan"]["steps"] if s["skill"].startswith("image_")], [TOOL_STRIP, TOOL_RESIZE, TOOL_PAD])
        self.assertEqual(svc.validate(ir).errors, [])
        p = str(Path(self.tmp) / "p.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        self.assertEqual([c[0] for c in self.calls()], ["strip", "resize", "pad"])
        self.assertIn("--gravity=east", self.calls()[2]); self.assertIn("--color=none", self.calls()[2])
        self.assertEqual(self.calls()[0][-1], os.path.abspath(self.png), "strip reads the user's file")
        self.assertNotIn(os.path.abspath(self.png), [c[-1] for c in self.calls()[1:]], "resize and pad read the previous step's output")
        tools = [r["tool"] for r in out["execution"]["results"]]
        self.assertEqual(tools[tools.index(TOOL_STRIP):tools.index("video-editing/overlay") + 1], [TOOL_STRIP, TOOL_RESIZE, TOOL_PAD, "video-editing/overlay"])
        pad = next(r for r in out["execution"]["results"] if r["tool"] == TOOL_PAD)
        self.assertEqual((pad["data"]["output"]["width"], pad["data"]["output"]["height"]), (320, 180))
        prov = json.loads((Path(self.ws) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        rows = [e for e in prov["operations"] if e["tool"].startswith("imagemagick-skill/") or e["tool"] == "video-editing/overlay"]
        self.assertEqual([e["skill"] for e in rows], ["image_strip", "image_prepare", "image_pad", "video_overlay"])
        self.assertTrue(all(e["decision"] for e in rows))
        self.assertIn(pad["output"], rows[-1]["input"], "the overlay composites the prepared image, not the user's file")
        self.assertNotIn(os.path.abspath(self.png), rows[-1]["input"])
        kinds = [o["kind"] for o in prov.get("skill_observations") or []]
        self.assertEqual([k for k in kinds if k.startswith("image.")], ["image.strip", "image.resize", "image.pad"])
        self.assertEqual(Path(self.png).read_bytes()[:8], fim.SIG, "the user's image is untouched")
        self.assertEqual(fim.size_of(self.png), (1200, 600))
        # resume: the prepared images are reused by the agent's idempotency keys, nothing runs again
        n = len(self.calls())
        out2 = svc.render(load_ir(p), p, resume=out["job"]["id"])
        self.assertEqual(out2["execution"]["status"], "COMPLETED")
        self.assertEqual(len(self.calls()), n)

    def test_without_imagemagick_the_step_is_handed_off(self):
        svc = self.service(installed=False)
        ir = self.plan(svc, {**PREP, "edit.overlay.prepare.strip": True})
        d = ir.doc
        steps = {s["skill"]: s for s in d["plan"]["steps"]}
        self.assertEqual([(steps[k]["tool"], steps[k]["status"]) for k in ("image_strip", "image_prepare")], [(None, "BLOCKED")] * 2)
        self.assertEqual(d["plan"]["status"], "BLOCKED")
        hs = {h["skill"]: h for h in svc.handoffs(ir)}
        self.assertEqual(sorted(hs), ["image_prepare", "image_strip"])
        h = hs["image_prepare"]
        self.assertEqual((h["provided_by"], h["repositories"]), (["imagemagick-skill"], ["imagemagick-skill"]))
        self.assertIn("npx imagemagick-skill", h["install"])
        self.assertIn("/plugin install imagemagick-skill@velisa", h["install"])
        self.assertIn("imagemagick-skill", h["reason"])
        self.assertTrue(any(b["subject"] == "capability.image_prepare" for b in ir.blocked()))

    def test_a_tool_this_machine_cannot_run_is_handed_off_alone(self):
        """sips (no ImageMagick): resize runs, strip is BLOCKED with the Skill's own reason; the plan never strips with something else."""
        svc = self.service(backend_caps=["imagemagick-skill", "imagemagick-skill:RESIZE", "imagemagick-skill:PAD"])
        ir = self.plan(svc, {**PREP, "edit.overlay.prepare.strip": True})
        steps = {s["skill"]: s for s in ir.doc["plan"]["steps"]}
        self.assertEqual((steps["image_prepare"]["tool"], steps["image_strip"]["tool"]), (TOOL_RESIZE, None))
        self.assertEqual([h["skill"] for h in svc.handoffs(ir)], ["image_strip"])

    def test_compiler_chain_and_validator(self):
        svc = self.service()
        ir = self.plan(svc, {**PREP, "edit.overlay.prepare.pad": "black", "edit.overlay.prepare.strip": True})
        ops, paths = compile_ir(ir, str(Path(self.tmp) / "job"))
        im = [o for o in ops if o.tool.startswith("imagemagick-skill/")]
        ov = next(o for o in ops if o.tool == "video-editing/overlay")
        subject = ov.args["input"].rsplit("_", 1)[0]
        self.assertEqual([o.tool for o in im], [TOOL_STRIP, TOOL_RESIZE, TOOL_PAD])
        self.assertEqual(im[0].args, {"input": f"{subject}_overlay_image", "output": f"{subject}_overlay_image_strip"})
        self.assertEqual(im[1].args, {"input": f"{subject}_overlay_image_strip", "width": 320, "height": 180, "mode": "fit", "output": f"{subject}_overlay_image_resize"})
        self.assertEqual(im[2].args, {"input": f"{subject}_overlay_image_resize", "width": 320, "height": 180, "color": "black", "gravity": "center", "output": f"{subject}_overlay_image_pad"})
        self.assertEqual(ov.args["image"], f"{subject}_overlay_image_pad")
        self.assertEqual(paths[f"{subject}_overlay_image"], os.path.abspath(self.png))
        self.assertTrue(all(paths[o.outputs[0]].endswith(".png") and paths[o.outputs[0]].startswith(str(Path(self.tmp) / "job" / "ops")) for o in im))
        self.assertEqual(len({o.idempotency_key for o in im}), 3)
        self.assertEqual([o.idempotency_key for o in compile_ir(ir, str(Path(self.tmp) / "job"))[0]], [o.idempotency_key for o in ops], "deterministic")
        fim.image(self.png, 1000, 600)   # the user replaced the image: every preparation key changes
        os.utime(self.png, (1, 1))
        self.assertNotEqual([o.idempotency_key for o in compile_ir(ir, str(Path(self.tmp) / "job"))[0] if o.tool.startswith("imagemagick-skill/")], [o.idempotency_key for o in im])
        base = json.loads(json.dumps(ir.doc))

        def errors(fn):
            doc = json.loads(json.dumps(base))
            fn(doc)
            return validate_ir(ProjectIR(doc), check_paths=False, registry=svc.registry).errors

        def overlay(doc):
            return next(o for o in doc["video"]["operations"] if o["type"] == "video.overlay")
        self.assertEqual(errors(lambda doc: None), [])
        self.assertTrue(any("mode" in e for e in errors(lambda doc: overlay(doc)["prepare"].update(mode="stretch"))))
        self.assertTrue(any("additional properties" in e.lower() or "vocabulary" in e for e in errors(lambda doc: overlay(doc)["prepare"].update(command="magick"))))
        self.assertTrue(any("pad" in e for e in errors(lambda doc: overlay(doc)["prepare"].update(mode="exact"))))
        self.assertTrue(any("has no plan step" in e for e in errors(lambda doc: doc["plan"]["steps"].remove(next(s for s in doc["plan"]["steps"] if s["skill"] == "image_pad")))))
        self.assertTrue(any("vocabulary" in e or "additional properties" in e.lower()
                            for e in errors(lambda doc: next(o for o in doc["video"]["operations"] if o["type"] == "video.trim").update(prepare={"strip": True}))))
        self.assertTrue(any("not a domain parameter" in e for e in errors(lambda doc: next(s for s in doc["plan"]["steps"] if s["skill"] == "image_prepare")["params"].update(argv=["x"]))))
        rep = validate_ir(ProjectIR(json.loads(json.dumps(base))), Caps(extra=IM_CAPS, missing={"imagemagick-skill"}).resolve(), check_paths=False, registry=svc.registry)
        self.assertIn("required capability MISSING: imagemagick-skill", rep.errors, "a plan that prepares an image depends on imagemagick-skill")


if __name__ == "__main__":
    unittest.main()
