"""ADR-080: handed-off jobs whose Skill contract is pinned (skills/handoff_contracts.py). The SkillSpec cites the Skill's own tool id,
`job.<skill>.<name>` parameters are checked against the pin before analysis, and the handoff names the tool, its arguments by the
contract's names, what is still missing and where the output goes next. The plan stays BLOCKED: nothing is run.

The live checks (`HandoffPinLiveTests`) compare each pin with a sibling checkout's contract when its VIDEO_AGENT_*_DIR is set (CI's
integration job); without one they skip."""
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
from test_unit import FakeCaps, fake_media  # noqa: E402

from video_agent.service import Service  # noqa: E402
from video_agent.skills.handoff_contracts import PINS, check_pin_against, load_pin  # noqa: E402
from video_agent.skills.registry import default_registry  # noqa: E402


def _svc():
    tmp = tempfile.mkdtemp()
    return tmp, Service(workspace=tmp, adapter=FakeAdapter(), caps=FakeCaps())


def _handoff(svc, ir, skill):
    [h] = [h for h in svc.handoffs(ir) if h["skill"] == skill]
    return h


class PinnedSkillSpecTests(unittest.TestCase):
    def test_each_pinned_job_cites_the_tool_its_pin_provides(self):
        reg = default_registry()
        for skill, p in PINS.items():
            spec = reg.get(skill)
            self.assertEqual(spec.tools, [p.tool], skill)
            self.assertEqual(spec.required_capabilities, [p.package], skill)
            self.assertEqual((spec.phase, spec.approval), (2, "CONFIRM"), f"{skill}: still a handoff, and asked for with confirmation")
            c = load_pin(p.pin_file)
            self.assertEqual(c["skill_id"], p.package)
            self.assertEqual(c["version"], p.version)
            self.assertIn(p.tool, [x["tool_id"] for x in c["provides"]])
            self.assertEqual(check_pin_against(c, p), [], f"{skill}: the pin satisfies what the handoff reads from it")

    def test_pinned_vocabulary_is_read_from_the_pin(self):
        self.assertEqual(PINS["render_3d"].param("preset").choices(load_pin(PINS["render_3d"].pin_file)), ("thumbnail", "square", "720p", "1080p", "4k", "vertical"))
        self.assertEqual(PINS["html_scene_render"].param("codec").choices(load_pin(PINS["html_scene_render"].pin_file)), ("h264", "vp9", "prores"))

    def test_hyperframes_pin_is_0_6_2_with_the_argv_recorder(self):
        """hyperframes-skill 0.6.2 renamed execution.argv_shims to execution.argv_recorder (its D10): the pin is that contract."""
        ex = load_pin(PINS["html_scene_render"].pin_file)["execution"]
        self.assertIn("argv_recorder", ex)
        self.assertNotIn("argv_shims", ex)
        self.assertIs(ex["shell"], False)

    def test_a_stale_pin_is_named(self):
        p = PINS["html_scene_render"]
        live = json.loads(json.dumps(load_pin(p.pin_file)))
        live["provides"] = [x for x in live["provides"] if x["tool_id"] != p.tool]
        for t in live["tools"]:
            if t.get("id") == p.tool:
                t["input_schema"]["properties"]["codec"]["enum"] = ["h264"]
        errs = check_pin_against(live, p)
        self.assertTrue(any("provides[] has no hyperframes-skill/render" in e for e in errs), errs)
        self.assertTrue(any(e.startswith("codec: pinned") for e in errs), errs)


class Render3dHandoffTests(unittest.TestCase):
    def test_parameters_reach_the_step_and_the_handoff(self):
        tmp, svc = _svc()
        model = Path(tmp) / "robot.glb"
        model.write_bytes(b"glTF")
        ir = svc.plan([fake_media(tmp)], "youtube", user_requirements={"job.render_3d": True, "job.render_3d.model": str(model),
                                                                        "job.render_3d.preset": "1080p", "job.render_3d.transparent": "true"})
        step = next(s for s in ir.doc["plan"]["steps"] if s["skill"] == "render_3d")
        self.assertEqual((step["tool"], step["status"]), (None, "BLOCKED"))
        self.assertEqual(step["params"], {"model": str(model.resolve()), "preset": "1080p", "transparent": True})
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        asked = next(d for d in ir.doc["decisions"] if d["id"] == step["decision_id"])
        self.assertEqual(asked["approval"], "CONFIRM")
        self.assertIn("blender-skill/render, contract 0.31.0 pinned", asked["reason"])
        keys = {r["id"]: r["key"] for r in ir.doc["requirements"]}
        self.assertEqual({keys[e] for e in asked["evidence"] if e in keys},
                         {"job.render_3d", "job.render_3d.model", "job.render_3d.preset", "job.render_3d.transparent"})
        h = _handoff(svc, ir, "render_3d")
        self.assertEqual(h["provided_by"], ["blender-skill"])
        k = h["contract"]
        self.assertEqual((k["package"], k["version"], k["tool"], k["capability"], k["lifecycle"]), ("blender-skill", "0.31.0", "blender-skill/render", "3d.render", "EXPERIMENTAL"))
        self.assertEqual(k["args"], {"file": str(model.resolve()), "--resolution-preset": "1080p", "--transparent": True})
        self.assertEqual(k["missing"], [])
        self.assertIn("ffmpeg-skill/overlay", k["next"])
        self.assertIn("alpha", k["next"])
        self.assertIn("Blender 4.2+", k["note"])
        self.assertEqual(model.read_bytes(), b"glTF", "the model is never touched")
        rep = svc.validate(ir)
        self.assertFalse([e for e in rep.errors if "domain parameter" in e or "params leak" in e], rep.errors)

    def test_switch_alone_names_what_is_missing(self):
        tmp, svc = _svc()
        ir = svc.plan([fake_media(tmp)], "youtube", user_requirements={"job.render_3d": True})
        k = _handoff(svc, ir, "render_3d")["contract"]
        self.assertEqual(k["args"], {})
        self.assertEqual(len(k["missing"]), 1)
        self.assertTrue(k["missing"][0].startswith("job.render_3d.model"))
        self.assertIn("B-roll", k["next"])
        self.assertNotIn("alpha", k["next"])

    def test_bad_parameters_are_refused_before_analysis(self):
        tmp, svc = _svc()
        model = Path(tmp) / "robot.glb"
        model.write_bytes(b"glTF")
        text = Path(tmp) / "notes.txt"
        text.write_text("x")
        on = {"job.render_3d": True}
        for extra, msg in (({"job.render_3d.preset": "8k"}, "must be one of thumbnail, square, 720p, 1080p, 4k, vertical"),
                           ({"job.render_3d.model": str(Path(tmp) / "missing.glb")}, "no such file"),
                           ({"job.render_3d.model": str(text)}, "is not a file blender-skill reads"),
                           ({"job.render_3d.transparent": "maybe"}, "is a switch"),
                           ({"job.render_3d.samples": "64"}, "unknown parameter 'job.render_3d.samples'")):
            with self.assertRaisesRegex(Exception, msg):
                svc.plan([fake_media(tmp)], "youtube", user_requirements={**on, **extra})
        with self.assertRaisesRegex(Exception, "needs job.render_3d=true"):
            svc.plan([fake_media(tmp)], "youtube", user_requirements={"job.render_3d.model": str(model)})
        with self.assertRaisesRegex(Exception, "takes no parameters"):
            svc.plan([fake_media(tmp)], "youtube", user_requirements={"job.dubbing": True, "job.dubbing.voice": "x"})


class HtmlSceneHandoffTests(unittest.TestCase):
    def _scene(self, tmp):
        scene = Path(tmp) / "title-card"
        scene.mkdir()
        (scene / "index.html").write_text("<div data-width='1920' data-height='1080' data-duration='3'></div>", encoding="utf-8")
        return scene

    def test_prores_goes_to_overlay_h264_to_concat(self):
        tmp, svc = _svc()
        scene = self._scene(tmp)
        ir = svc.plan([fake_media(tmp)], "youtube", user_requirements={"job.html_scene_render": True, "job.html_scene_render.scene": str(scene),
                                                                        "job.html_scene_render.codec": "prores"})
        k = _handoff(svc, ir, "html_scene_render")["contract"]
        self.assertEqual((k["package"], k["version"], k["tool"]), ("hyperframes-skill", "0.6.2", "hyperframes-skill/render"))
        self.assertEqual(k["args"], {"scene_dir": str(scene.resolve()), "codec": "prores"})
        self.assertEqual(k["missing"], [])
        self.assertIn("ffmpeg-skill/overlay", k["next"])
        self.assertIn("argv", k["note"])
        ir2 = svc.plan([fake_media(tmp)], "youtube", user_requirements={"job.html_scene_render": True, "job.html_scene_render.scene": str(scene)})
        k2 = _handoff(svc, ir2, "html_scene_render")["contract"]
        self.assertIn("video-editing/concat", k2["next"])
        self.assertEqual(next(s for s in ir2.doc["plan"]["steps"] if s["skill"] == "html_scene_render")["params"], {"scene": str(scene.resolve())})

    def test_scene_without_index_html_or_unknown_codec_is_refused(self):
        tmp, svc = _svc()
        empty = Path(tmp) / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(Exception, "has no index.html"):
            svc.plan([fake_media(tmp)], "youtube", user_requirements={"job.html_scene_render": True, "job.html_scene_render.scene": str(empty)})
        with self.assertRaisesRegex(Exception, "must be one of h264, vp9, prores"):
            svc.plan([fake_media(tmp)], "youtube", user_requirements={"job.html_scene_render": True, "job.html_scene_render.codec": "av1"})


def _live_contract(skill: str):
    """(contract, None) from the sibling checkout its VIDEO_AGENT_*_DIR names, or (None, why not)."""
    if skill == "render_3d":
        d = os.environ.get("VIDEO_AGENT_BLENDER_DIR")
        argv = [sys.executable, "-I", str(Path(d or ".") / "scripts" / "_contract.py"), "--json"]
    elif skill == "html_scene_render":
        d = os.environ.get("VIDEO_AGENT_HYPERFRAMES_DIR")
        argv = ["node", str(Path(d or ".") / "bin" / "hyperframes-skill.mjs"), "contract", "--json"]
    else:
        return None, f"no live check for {skill}"
    if not d:
        return None, f"no checkout named for {skill}"
    out = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise AssertionError(f"{skill} contract failed: {out.stderr[-400:]}")
    return json.loads(out.stdout), None


class HandoffPinLiveTests(unittest.TestCase):
    """Each pin against its Skill's main (CI clones it): identity, the tool in provides[], the enums the parameters accept."""

    def _check(self, skill):
        live, why = _live_contract(skill)
        if live is None:
            self.skipTest(why)
        p = PINS[skill]
        self.assertEqual(check_pin_against(live, p), [], f"{skill}: re-pin skills/pins/{p.pin_file}")

    def test_blender_skill(self):
        self._check("render_3d")

    def test_hyperframes_skill(self):
        self._check("html_scene_render")


if __name__ == "__main__":
    unittest.main()
