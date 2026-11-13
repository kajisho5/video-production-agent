"""ADR-114 (roadmap step 29): `motion.layer=hyperframes` -- hyperframes-skill renders the clip (a template + values or a scene request → scene →
render --codec prores [--segments]) and motion-graphics-skill's rendered_layer places it, with in / out / loop from the segments. Against a fake
checkout (tests/fake_hyperframes.py) that speaks the v0.10.0 contract, doctor and result shapes, with the fake media engine and the fake
motion-graphics-skill. No Node, Chromium or ffmpeg. `HtmlLayerRealSkillTests` runs the real hyperframes-skill when VIDEO_AGENT_HYPERFRAMES_DIR
names a checkout (the render itself only when its doctor finds a headless Chromium)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import PipelineCaps, clear_modes, fake_media, pipeline_service  # noqa: E402

from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.html_layer import element_timing, parse_html_layer_requirements  # noqa: E402
from video_agent.capabilities.resolver import CapabilityResolver  # noqa: E402
from video_agent.models import Operation  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.service import Service, _check_requirement_keys  # noqa: E402
from video_agent.skills.handoff_contracts import PINS, load_pin  # noqa: E402
from video_agent.tools.hyperframes import (HyperframesAdapter, HyperframesSkill, PINNED_CONTRACT_PATH, ContractError, breaking_drift, check_contract,  # noqa: E402
                                           drift, locate_hyperframes, pinned_contract, recovery_class)

FAKE = Path(__file__).resolve().parent / "fake_hyperframes.py"
MODES = ("FAKE_HF_MODE", "FAKE_HF_READY", "FAKE_HF_CALLS", "FAKE_HF_PIN", "VIDEO_AGENT_HYPERFRAMES_DIR")
VALUES = {"name": "Ada Lovelace", "affiliation": "Analytical Engines", "duration": 4}


def fake_checkout(tmp: str) -> HyperframesSkill:
    d = Path(tmp) / "hyperframes-skill"
    (d / "bin").mkdir(parents=True, exist_ok=True)
    (d / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(FAKE, d / "bin" / "hyperframes-skill.mjs")
    (d / "scripts" / "render.mjs").write_text("// fake\n", encoding="utf-8")
    return HyperframesSkill(d, sys.executable)   # Python runs the fake: the adapter only needs an argv


def hf_adapter(tmp: str) -> HyperframesAdapter:
    ws = str(Path(tmp) / "ws")
    os.makedirs(ws, exist_ok=True)
    return HyperframesAdapter(fake_checkout(tmp), workspace=ws, allowed_inputs=[str(Path(tmp) / "src"), ws])


def hf_capabilities(tmp: str) -> dict:
    caps: dict = {}
    CapabilityResolver._finishing_skill(CapabilityResolver.__new__(CapabilityResolver), caps, "hyperframes-skill", fake_checkout(tmp), lambda _sk: hf_adapter(tmp),
                                        "VIDEO_AGENT_HYPERFRAMES_DIR", "hyperframes-skill", "npm ci", per_item=lambda ad, doc: (ad.operation_status(doc), {}),
                                        item_evidence=lambda ad, doc: ad.operation_evidence(doc))
    return caps


class HfCaps(PipelineCaps):
    def __init__(self, hf: dict, **kw):
        super().__init__(**kw)
        self.hf = hf

    def resolve(self, refresh=False):
        out = super().resolve(refresh)
        out.update(self.hf)
        return out


def hf_service(tmp: str, installed: bool = True):
    svc = pipeline_service(tmp, caps=HfCaps(hf_capabilities(tmp) if installed else {}), transcription=False)
    if installed:
        svc._adapter.register(hf_adapter(tmp))
        svc.adapter([])
    return svc


def layer_reqs(**extra):
    return dict({"motion.layer": "hyperframes", "hyperframes.template": "lower-third", "hyperframes.values": json.dumps(VALUES),
                 "hyperframes.segments": "0.5,3,0.5", "motion.layer.position": "bottom-left", "motion.layer.start": 1, "motion.layer.end": 9}, **extra)


class _Base(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self._env = {k: os.environ.get(k) for k in MODES}
        for k in MODES:
            os.environ.pop(k, None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.log = str(Path(self.tmp) / "hf_calls.log")
        os.environ["FAKE_HF_CALLS"] = self.log
        os.environ["FAKE_HF_PIN"] = str(PINNED_CONTRACT_PATH)
        (Path(self.tmp) / "src").mkdir(exist_ok=True)

    def tearDown(self):
        clear_modes()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def hf_calls(self, tool=None):
        if not os.path.exists(self.log):
            return []
        rows = [json.loads(x) for x in Path(self.log).read_text(encoding="utf-8").splitlines()]
        return [r for r in rows if tool is None or r[0] == tool]

    def request_file(self, segments=True) -> str:
        p = Path(self.tmp) / "src" / "strap.json"
        req = {"scene_version": 1, "id": "strap", "width": 1920, "height": 1080, "duration": 5, "fps": 30, "background": "transparent", "layers": []}
        if segments:
            req["segments"] = {"in": 1, "loop": 3, "out": 1}
        p.write_text(json.dumps(req), encoding="utf-8")
        return str(p)


class ContractTests(_Base):
    def test_pin_is_0_10_0_and_the_handoff_reads_the_same_file(self):
        c = pinned_contract()
        self.assertEqual((c["skill_id"], c["version"]), ("hyperframes-skill", "0.10.0"))
        self.assertEqual(check_contract(c), [])
        self.assertEqual(PINS["html_scene_render"].pin_file, PINNED_CONTRACT_PATH.name)
        self.assertEqual((drift(c), breaking_drift(c)), ([], []))

    def test_gate_is_the_0_8_0_9_and_0_10_lines(self):
        for v, ok in (("0.8.0", True), ("0.8.1", True), ("0.8.9", True), ("0.9.0", True), ("0.9.4", True), ("0.10.0", True), ("0.10.3", True), ("0.11.0", False),
                      ("1.0.0", False), ("0.7.0", False)):
            with self.subTest(version=v):
                c = dict(pinned_contract(), version=v)
                self.assertEqual(check_contract(c) == [], ok, check_contract(c))

    def test_what_the_adapter_relies_on_is_checked(self):
        c = json.loads(json.dumps(pinned_contract()))
        render = next(t for t in c["tools"] if t["id"] == "hyperframes-skill/render")
        render["input_schema"]["properties"]["codec"]["enum"] = ["h264", "vp9"]
        del render["input_schema"]["properties"]["segments"]
        errs = check_contract(c)
        self.assertTrue(any("prores" in e for e in errs) and any("segments" in e for e in errs), errs)

    def test_locate_needs_the_cli_and_names_a_missing_node(self):
        self.assertIsNone(locate_hyperframes(explicit=self.tmp, env={}))
        sk = fake_checkout(self.tmp)
        self.assertEqual(locate_hyperframes(explicit=str(sk.root), env={}, node=sys.executable).root, sk.root)
        with self.assertRaisesRegex(ContractError, "Node.js is not on PATH"):
            HyperframesAdapter(HyperframesSkill(sk.root, ""), workspace=self.tmp)


class CapabilityTests(_Base):
    def test_doctor(self):
        caps = hf_capabilities(self.tmp)
        self.assertEqual((caps["hyperframes-skill"].status, caps["hyperframes-skill:RENDER"].status), ("AVAILABLE", "AVAILABLE"))
        os.environ["FAKE_HF_READY"] = "nochromium"
        caps = hf_capabilities(self.tmp)
        self.assertEqual((caps["hyperframes-skill"].status, caps["hyperframes-skill:RENDER"].status), ("AVAILABLE", "MISSING"),
                         "the scene can still be written: the package is found, its render is not usable")
        self.assertIn("chromium:headless", caps["hyperframes-skill:RENDER"].evidence["reason"])
        self.assertIn("browser ensure", caps["hyperframes-skill:RENDER"].evidence["install"])


class AdapterTests(_Base):
    def run_op(self, args):
        ad = hf_adapter(self.tmp)
        out = str(Path(ad.workspace) / "jobs" / "j" / "ops" / "talk_html_layer" / "talk_layer.mov")
        paths = {"out": out}
        for k in ("request", "font"):
            if k in args:
                paths[f"ref_{k}"] = args[k]
                args = dict(args, **{k: f"ref_{k}"})
        return ad, out, ad.run(Operation(tool="hyperframes-skill/render", args=dict(args, output="out"), inputs=[], outputs=["out"]), paths)

    def test_template_with_segments(self):
        ad, out, r = self.run_op({"template": "lower-third", "values": VALUES, "segments": {"in": 0.5, "loop": 3, "out": 0.5}})
        self.assertTrue(r.ok, r.data)
        [tpl], [scene], [render] = self.hf_calls("template"), self.hf_calls("scene"), self.hf_calls("render")
        work = os.path.dirname(out)
        self.assertEqual(tpl[1:], ["lower-third", f"--values={os.path.join(work, 'talk_layer.values.json')}", f"--output={os.path.join(work, 'talk_layer.request.json')}", "--json", "--overwrite"])
        self.assertEqual(scene[1:], [f"{os.path.join(work, 'talk_layer.request.json')}", f"--output={os.path.join(work, 'talk_layer.scene')}", "--json"])
        self.assertEqual(render[1:], [f"{os.path.join(work, 'talk_layer.scene')}", f"--output={out}", "--codec=prores", "--segments", "--json", "--overwrite", "--timeout=1800"])
        req = json.loads(Path(work, "talk_layer.request.json").read_text(encoding="utf-8"))
        self.assertEqual(req["segments"], {"in": 0.5, "loop": 3.0, "out": 0.5}, "the segments asked for are added to the template's request")
        d = r.data
        self.assertEqual((d["codec"], d["alpha"]["mode"], d["layer"], d["ffmpeg_skill"]["status"]), ("prores", "straight", out, "PASSED"))
        self.assertEqual([(s["name"], s["duration"]) for s in d["segments"]], [("in", 0.5), ("loop", 3.0), ("out", 0.5)])
        self.assertEqual(d["layer_timing"], {"in_duration": 0.5, "out_duration": 0.5, "loop": True})
        self.assertEqual((d["template"], d["values"], d["hyperframes"]["version"]), ("lower-third", VALUES, "0.8.61"))
        self.assertEqual(len(d["sha256"]), 64)
        self.assertTrue(any(c["program"] == "ffmpeg" for c in d["tool_commands"]), "the Skill's own argv record is kept (provenance)")

    def test_scene_request_without_segments(self):
        req = self.request_file(segments=False)
        _, out, r = self.run_op({"request": req, "workers": 1})
        self.assertTrue(r.ok, r.data)
        self.assertEqual(self.hf_calls("template"), [])
        [render] = self.hf_calls("render")
        self.assertNotIn("--segments", render)
        self.assertIn("--workers=1", render)
        self.assertEqual((r.data["segments"], r.data["layer_timing"]["loop"], r.data["request"]["path"]), (None, False, req))

    def test_refused_before_a_process(self):
        req = self.request_file()
        for args in ({"template": "lower-third", "values": VALUES, "codec": "vp9"}, {"template": "Lower Third!", "values": VALUES}, {"template": "lower-third", "request": req},
                     {}, {"request": req, "values": VALUES}, {"request": "/etc/passwd"}, {"template": "x", "values": {"a": {"nested": 1}}},
                     {"template": "x", "values": VALUES, "segments": {"in": 1, "loop": 0, "out": 1}}, {"request": req, "workers": 99}):
            with self.subTest(args=args):
                _, _, r = self.run_op(args)
                self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_REQUEST"))
        self.assertEqual(self.hf_calls("render"), [])

    def test_results_refused(self):
        for mode, why in (("noalpha", "no alpha"), ("shortseg", "segment loop is 2.0s, planned 3s"), ("seam", "loop seam"), ("unverified", "not verified"),
                          ("elsewhere", "not the planned")):
            with self.subTest(mode=mode):
                os.environ["FAKE_HF_MODE"] = mode
                _, _, r = self.run_op({"template": "lower-third", "values": VALUES, "segments": {"in": 0.5, "loop": 3, "out": 0.5}})
                self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_OUTPUT"))
                self.assertIn(why, r.data["error"]["message"])
        os.environ["FAKE_HF_MODE"] = "fail:missing_tool"
        _, _, r = self.run_op({"template": "lower-third", "values": VALUES})
        self.assertEqual((r.ok, r.data["error"]["code"], r.data["error"]["recovery_class"], r.exit_code), (False, "missing_tool", "TOOL_MISSING", 127))
        self.assertIn("fake hint", r.data["error"]["message"])
        self.assertEqual(recovery_class("input"), "INVALID_ARGS")


class RequirementTests(_Base):
    def test_vocabulary(self):
        p = parse_html_layer_requirements(layer_reqs())
        self.assertEqual(p, {"template": "lower-third", "values": VALUES, "segments": {"in": 0.5, "loop": 3.0, "out": 0.5}})
        self.assertEqual(element_timing(p), {"in_duration": 0.5, "out_duration": 0.5, "loop": True})
        vf = Path(self.tmp) / "src" / "values.json"
        vf.write_text(json.dumps(VALUES), encoding="utf-8")
        self.assertEqual(parse_html_layer_requirements({"motion.layer": "hyperframes", "hyperframes.template": "title-card", "hyperframes.values": str(vf)})["values"], VALUES)
        req = self.request_file()
        self.assertEqual(parse_html_layer_requirements({"motion.layer": "hyperframes", "hyperframes.request": req, "hyperframes.segments": True, "hyperframes.workers": "1"}),
                         {"request": req, "segments": {"in": 1.0, "loop": 3.0, "out": 1.0}, "workers": 1})
        self.assertEqual(element_timing({"template": "x"}), {"in_duration": 0.0, "out_duration": 0.0, "loop": False})
        self.assertIsNone(parse_html_layer_requirements({"motion.title": "Hello"}), "never a default")

    def test_refusals(self):
        req = self.request_file()
        for reqs, why in (({"hyperframes.template": "lower-third"}, "motion.layer is not set"),
                          ({"motion.layer": str(Path(self.tmp) / "src" / "x.mov"), "hyperframes.template": "lower-third"}, "a file of your own"),
                          ({"motion.layer": "hyperframes"}, "exactly one of"),
                          ({"motion.layer": "hyperframes", "hyperframes.template": "lower-third", "hyperframes.request": req}, "exactly one of"),
                          ({"motion.layer": "hyperframes", "hyperframes.template": "lower-third"}, "needs hyperframes.values"),
                          (layer_reqs(**{"motion.layer.in_duration": 0.5}), "hyperframes-skill's segments"),
                          (layer_reqs(**{"motion.layer.loop": False}), "never set by hand"),
                          (layer_reqs(**{"hyperframes.segments": "true"}), "a template declares none"),
                          (layer_reqs(**{"hyperframes.segments": "1,3,1"}), "the template's duration is 4s"),
                          (layer_reqs(**{"hyperframes.segments": "0.5,0,3.5"}), "loop must be > 0"),
                          (layer_reqs(**{"hyperframes.values": "{\"name\": \"Ada\"}"}), "needs the template's duration"),
                          (layer_reqs(**{"hyperframes.values": "[1]"}), "JSON object"),
                          (layer_reqs(**{"hyperframes.codec": "vp9"}), "unknown requirement key"),
                          ({"motion.layer": "hyperframes", "hyperframes.request": req, "hyperframes.values": "{\"a\": 1}"}, "fills a template"),
                          ({"motion.layer": "hyperframes", "hyperframes.request": req, "hyperframes.segments": "1,3,1"}, "is true"),
                          ({"motion.layer": "hyperframes", "hyperframes.request": self.request_file(segments=False), "hyperframes.segments": True}, "declares no segments"),
                          ({"motion.layer": "hyperframes", "hyperframes.request": req, "hyperframes.workers": 12}, "within 1..8")):
            with self.subTest(reqs=reqs):
                with self.assertRaises(EditRequirementError) as cm:
                    parse_html_layer_requirements(reqs)
                self.assertIn(why, str(cm.exception))
                with self.assertRaises(ValueError):
                    _check_requirement_keys(reqs)
        _check_requirement_keys(layer_reqs())   # the motion.layer refinements and the hyperframes keys are known --set keys


class PlanTests(_Base):
    def plan(self, reqs=None, installed=True, **extra):
        self.svc = hf_service(self.tmp, installed=installed)
        self.media = fake_media(self.tmp, duration=16.0)
        return self.svc.plan([self.media], "generic", user_requirements=reqs or layer_reqs(**extra))

    def test_rendered_then_placed_with_the_segments_timing(self):
        ir = self.plan()
        d = ir.doc
        [dec] = [x for x in d["decisions"] if x["subject"] == "graphics.html_layer"]
        self.assertEqual((dec["type"], dec["approval"], dec["basis"]["approval"]["key"]), ("TRANSFORM", "CONFIRM", "graphics.html_layer.approval"))
        skills = [s["skill"] for s in d["plan"]["steps"]]
        self.assertLess(skills.index("html_scene_layer"), skills.index("motion_graphics"))
        hstep = next(s for s in d["plan"]["steps"] if s["skill"] == "html_scene_layer")
        gstep = next(s for s in d["plan"]["steps"] if s["skill"] == "motion_graphics")
        subj = list(d["assets"])[0]
        self.assertEqual((hstep["tool"], hstep["outputs"], hstep["params"]["codec"]), ("hyperframes-skill/render", [f"{subj}_html_layer"], "prores"))
        self.assertIn(hstep["id"], gstep["depends_on"])
        self.assertIn(f"{subj}_html_layer", gstep["inputs"])
        ops = {op["type"]: op for op in d["graphics"]["operations"]}
        el = next(e for e in ops["graphics.render"]["elements"] if e["type"] == "rendered_layer")
        self.assertEqual({k: el["parameters"][k] for k in ("layer", "in_duration", "out_duration", "loop", "position")},
                         {"layer": f"{subj}_html_layer", "in_duration": 0.5, "out_duration": 0.5, "loop": True, "position": "bottom-left"})
        self.assertEqual((ops["graphics.render"]["layer_source"], "layer" in ops["graphics.render"]), (f"{subj}_html_layer", False))
        self.assertEqual(ops["graphics.html_layer"]["segments"], {"in": 0.5, "loop": 3.0, "out": 0.5})
        self.assertEqual(self.svc.validate(ir).errors, [])
        self.assertNotIn(str(Path.cwd()), d["execution"]["allowed_inputs"], "the layer artifact id is not a user file")
        p = str(Path(self.tmp) / "p.json"); save_ir(ir, p)
        out = self.svc.render(load_ir(p), p)
        self.assertNotEqual(out["status"], "COMPLETED")
        self.assertEqual(self.hf_calls("render"), [], "nothing is rendered before the approval")
        out = self.svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED", out)
        [render] = self.hf_calls("render")
        self.assertIn("--codec=prores", render)
        hres = next(r for r in out["execution"]["results"] if r["tool"] == "hyperframes-skill/render")
        self.assertEqual((hres["data"]["skill"], hres["data"]["layer_timing"]), ({"id": "hyperframes-skill", "version": "0.10.0"}, {"in_duration": 0.5, "out_duration": 0.5, "loop": True}))
        prov = json.loads((Path(self.svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        row = next(e for e in prov["operations"] if e["tool"] == "motion-graphics/run")
        sent = next(e for e in row["args"]["elements"] if e["type"] == "rendered_layer")
        self.assertEqual((sent["parameters"]["layer"], sent["parameters"]["in_duration"], sent["parameters"]["loop"]), (f"{subj}_html_layer", 0.5, True))
        self.assertTrue(any(e["tool"] == "hyperframes-skill/render" for e in prov["operations"]), "the render is in the job's provenance")

    def test_scene_request_and_no_segments(self):
        req = self.request_file(segments=False)
        ir = self.plan({"motion.layer": "hyperframes", "hyperframes.request": req, "motion.layer.start": 2})
        d = ir.doc
        el = next(e for op in d["graphics"]["operations"] if op["type"] == "graphics.render" for e in op["elements"])
        self.assertEqual({k: el["parameters"][k] for k in ("in_duration", "out_duration", "loop")}, {"in_duration": 0.0, "out_duration": 0.0, "loop": False})
        self.assertIn(str(Path(req).parent), d["execution"]["allowed_inputs"])
        hstep = next(s for s in d["plan"]["steps"] if s["skill"] == "html_scene_layer")
        self.assertEqual(hstep["params"]["request"], "strap.json", "the user's file by name in the plan")
        self.assertEqual(self.svc.validate(ir).errors, [])

    def test_never_by_default(self):
        ir = self.plan({"motion.lower_third": "Ada Lovelace"})
        self.assertNotIn("html_scene_layer", [s["skill"] for s in ir.doc["plan"]["steps"]])
        self.assertFalse([x for x in ir.doc["decisions"] if x["subject"] == "graphics.html_layer"])

    def test_without_the_skill_the_step_is_handed_off(self):
        ir = self.plan(installed=False)
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        [blk] = [x for x in ir.doc["decisions"] if x["subject"] == "capability.html_scene_layer"]
        self.assertIn("not installed", blk["reason"])
        [h] = [h for h in self.svc.handoffs(ir) if h["skill"] == "html_scene_layer"]
        self.assertEqual(h["provided_by"], ["hyperframes-skill"])
        c = h["contract"]
        self.assertEqual((c["tool"], c["version"], c["args"]["codec"], c["args"]["segments"]), ("hyperframes-skill/render", "0.10.0", "prores", True))
        self.assertIn("motion.layer=<file>", c["next"])
        self.assertIn("--segments", c["steps"][-1])

    def test_doctor_without_chromium_blocks_by_name(self):
        os.environ["FAKE_HF_READY"] = "nochromium"
        ir = self.plan()
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        [blk] = [x for x in ir.doc["decisions"] if x["subject"] == "capability.html_scene_layer"]
        self.assertIn("chromium:headless", blk["reason"])
        self.assertIn("browser ensure", blk["reason"])
        self.assertTrue([h for h in self.svc.handoffs(ir) if h["skill"] == "html_scene_layer"])


def _real_checkout():
    d = os.environ.get("VIDEO_AGENT_HYPERFRAMES_DIR")
    return locate_hyperframes(explicit=d) if d else None


@unittest.skipUnless(_real_checkout() is not None and shutil.which("node"), "needs a hyperframes-skill checkout (VIDEO_AGENT_HYPERFRAMES_DIR) and Node")
class HtmlLayerRealSkillTests(unittest.TestCase):
    """The real hyperframes-skill: its live contract passes the adapter's checks against the pin; its doctor decides whether the step can run (no
    headless Chromium: the plan is BLOCKED naming it); with a Chromium, the lower-third template rendered with IN / LOOP / OUT is ProRes 4444 with
    alpha, and -- with the real motion-graphics-skill and ffmpeg-skill -- placed over a clip by rendered_layer."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = os.path.realpath(tempfile.mkdtemp(prefix="va_hf_"))
        cls.ws = str(Path(cls.tmp) / "ws")
        os.makedirs(cls.ws)
        cls.ad = HyperframesAdapter(_real_checkout(), workspace=cls.ws, allowed_inputs=[cls.tmp])
        cls.doc = cls.ad.doctor()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_live_contract_against_the_pin(self):
        self.assertEqual(check_contract(self.ad.contract), [])
        self.assertEqual(self.ad.breaking_drift(), [], "re-pin skills/pins/hyperframes-skill_<version>.json")
        self.assertEqual(self.ad.version, load_pin(PINS["html_scene_render"].pin_file)["version"])

    def test_doctor_decides_and_a_missing_chromium_is_named(self):
        self.assertIn(self.doc["status"], ("ok", "degraded", "fail"))
        ev = self.ad.operation_evidence(self.doc)["RENDER"]
        if self.doc["status"] != "ok":
            self.assertTrue(ev["reason"], ev)
            self.assertFalse(ev["ready"])
        else:
            self.assertTrue(ev["ready"])

    def _render_or_skip(self):
        if self.doc["status"] != "ok":
            self.skipTest("hyperframes-skill's doctor: render is not usable here (" + "; ".join(self.doc.get("problems") or [])[:200] + ")")

    def test_lower_third_with_segments_is_prores_with_alpha(self):
        self._render_or_skip()
        out = str(Path(self.ws) / "jobs" / "real" / "ops" / "talk_html_layer" / "talk_layer.mov")
        r = self.ad.run(Operation(tool="hyperframes-skill/render", args={"template": "lower-third", "values": VALUES, "segments": {"in": 0.5, "loop": 3, "out": 0.5},
                                                                         "workers": 1, "output": "out"}, inputs=[], outputs=["out"]), {"out": out})
        self.assertTrue(r.ok, r.data)
        self.assertEqual([(s["name"], float(s["duration"]), s["frames"]) for s in r.data["segments"]], [("in", 0.5, 15), ("loop", 3.0, 90), ("out", 0.5, 15)])
        self.assertTrue(r.data["loop_seam"]["ok"])
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt", "-of", "json", out],
                            capture_output=True, text=True, check=True)
        s = json.loads(pr.stdout)["streams"][0]
        self.assertEqual(s["codec_name"], "prores")
        self.assertIn("a", s["pix_fmt"].replace("yuv", ""), s)   # yuva444p10le / yuva444p12le: an alpha plane

    def test_plan_renders_and_places_the_lower_third(self):
        self._render_or_skip()
        from video_agent.tools.ffmpeg_skill import locate_ffmpeg_skill
        from video_agent.tools.motion_graphics import locate_motion_graphics
        if not (shutil.which("ffmpeg") and locate_ffmpeg_skill() and locate_motion_graphics()):
            self.skipTest("needs ffmpeg, ffmpeg-skill and motion-graphics-skill")
        src = Path(self.tmp) / "src"
        src.mkdir(exist_ok=True)
        media = str(src / "talk.mp4")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=8", "-f", "lavfi", "-i",
                        "sine=frequency=440:d=8", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", media], check=True)
        svc = Service(workspace=str(Path(self.tmp) / "plan_ws"))
        caps = svc.caps.resolve()
        if getattr(caps.get("hyperframes-skill:RENDER"), "status", None) != "AVAILABLE" or getattr(caps.get("motion-graphics"), "status", None) != "AVAILABLE":
            self.skipTest("hyperframes-skill render or motion-graphics unavailable to the Service")
        ir = svc.plan([media], "generic", user_requirements=dict(layer_reqs(**{"hyperframes.workers": 1, "motion.layer.end": 7, "motion.layer.scale_width": 1280})))
        self.assertEqual(ir.doc["plan"]["status"], "REVIEW", ir.doc["plan"]["summary"])   # the layer render waits for a confirmation (floor CONFIRM)
        self.assertEqual(svc.validate(ir).errors, [])
        p = str(Path(self.tmp) / "plan.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", json.dumps([r for r in out["execution"]["results"] if not r["ok"]])[:1500])
        res = next(r for r in out["execution"]["results"] if r["tool"] == "motion-graphics/run")
        self.assertEqual(sorted(o["type"] for o in res["data"]["operations"]), ["rendered_layer"])


if __name__ == "__main__":
    unittest.main()
