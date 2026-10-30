"""What the satellites already offer, part 1 (ADR-062, ECOSYSTEM_ROADMAP step 6): color-grading-skill 0.4.0's PRIMARY_CORRECTION
gamma / lift / gain / levels / curves, and video-editing-skill's TRIM (an explicit in / out range) and ROTATE (lowered here together
with FREEZE / REVERSE). Each new parameter: a `--set` key checked by the shared validator (and therefore a brief key, ADR-061), a
decision, an IR record, a plan step and the compiled Skill request, on the fake engine and the fake Skills (tests/test_integration.py
runs trim / rotate and the colour parameters on the real Skills)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter  # noqa: E402
from pipeline_harness import clear_modes, pipeline_service, plan_and_render  # noqa: E402
from pipeline_harness import fake_media as pipeline_media  # noqa: E402
from test_unit import FakeCaps, fake_media, make_service  # noqa: E402
from video_agent.agent.editing import EditRequirementError, parse_edit_requirements  # noqa: E402
from video_agent.agent.finishing import COLOR_CURVES_PRESETS, parse_color_requirements, picture_size  # noqa: E402
from video_agent.agent.requirements import requirement_map  # noqa: E402
from video_agent.execution.compiler import CompileError, lower_color_op, lower_video_trim  # noqa: E402
from video_agent.models import Requirement  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.service import BRIEF_REQUIREMENT_KEYS, _check_requirement_keys  # noqa: E402
from video_agent.tools.color_grading.adapter import pinned_contract as color_contract  # noqa: E402
from video_agent.tools.video_editing.adapter import pinned_contract as editing_contract  # noqa: E402
from video_agent.tools.video_editing.lowering import ARGS, Lowering, params_for  # noqa: E402

FAKE_VE = str(Path(__file__).resolve().parent / "fake_video_editing.py")
NEW_COLOR = {"color.gamma": 1.2, "color.lift": 0.05, "color.gain": -0.1, "color.levels_in_black": 16, "color.levels_in_white": 235,
             "color.levels_out_black": 4, "color.levels_out_white": 250, "color.curves": "medium_contrast"}


def reqs(d):
    return requirement_map([Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in d.items()])


class ColorParameterTests(unittest.TestCase):
    """color.gamma / .lift / .gain / .levels_* / .curves → the PRIMARY_CORRECTION parameters of the same names."""

    def test_keys_match_the_pinned_contract(self):
        """Every new key names a PRIMARY_CORRECTION parameter of the pinned 0.4.0 contract, with the contract's own range / enum."""
        op = next(o for o in color_contract()["operations"] if o["type"] == "PRIMARY_CORRECTION")
        got = parse_color_requirements(reqs(NEW_COLOR))["color.primary_correction"]["params"]
        self.assertEqual(sorted(got), sorted(k.split(".", 1)[1] for k in NEW_COLOR))
        for name, v in got.items():
            spec = op["parameters"][name]
            if "enum" in spec:
                self.assertIn(v, spec["enum"]); self.assertEqual(tuple(spec["enum"]), COLOR_CURVES_PRESETS)
            else:
                self.assertTrue(spec["min"] <= v <= spec["max"], name)
                self.assertIsInstance(v, int if spec["type"] == "integer" else float, name)
        self.assertEqual(got["levels_in_black"], 16); self.assertEqual(got["curves"], "medium_contrast")
        # one operation for the old and the new parameters together
        both = parse_color_requirements(reqs({"color.exposure": 0.5, "color.gamma": 0.8}))
        self.assertEqual(both["color.primary_correction"]["params"], {"exposure": 0.5, "gamma": 0.8})

    def test_refusals(self):
        for bad, msg in (({"color.gamma": 0.05}, "within"), ({"color.gamma": 11}, "within"), ({"color.lift": -1.5}, "within"), ({"color.gain": "much"}, "number"),
                         ({"color.levels_in_black": 16.5}, "integer"), ({"color.levels_out_white": 300}, "within"),
                         ({"color.levels_in_black": 200, "color.levels_in_white": 100}, "below"), ({"color.levels_in_black": 255}, "below"),
                         ({"color.levels_out_white": 0}, "below"), ({"color.curves": "none"}, "one of"), ({"color.curves": "sepia"}, "one of")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                parse_color_requirements(reqs(bad))
            self.assertIn(msg, str(cm.exception), str(bad))
            with self.assertRaises(ValueError, msg=str(bad)):
                _check_requirement_keys(bad)   # the same refusal before any analysis runs (the --set gate, ADR-042 / ADR-061)
        with self.assertRaises(ValueError) as cm:
            _check_requirement_keys({"color.levels": 16})
        self.assertIn("unknown requirement key", str(cm.exception))

    def test_set_and_brief_keys(self):
        """The --set gate accepts every new key, and a brief may yield them (BRIEF_REQUIREMENT_KEYS is derived, not listed)."""
        _check_requirement_keys(dict(NEW_COLOR))
        for k in list(NEW_COLOR) + ["edit.trim", "edit.rotate", "edit.rotate.flip"]:
            self.assertIn(k, BRIEF_REQUIREMENT_KEYS, k)

    def test_compiled_request_carries_the_parameters_by_name(self):
        op = {"type": "color.primary_correction", "asset": "a", "gamma": 1.2, "levels_in_black": 16, "curves": "vintage", "unknown": 1}
        args = lower_color_op("color-grading/run", op, "a_trim", "a_pc")
        self.assertEqual(args, {"operation": "PRIMARY_CORRECTION", "input": "a_trim", "output": "a_pc", "format": "mp4", "gamma": 1.2, "levels_in_black": 16, "curves": "vintage"})

    def test_pipeline_decision_ir_step_and_skill_request(self):
        clear_modes()
        tmp = os.path.realpath(tempfile.mkdtemp())
        a = pipeline_media(tmp, "a.mp4")
        svc = pipeline_service(tmp)
        r = plan_and_render(svc, [a], dict(NEW_COLOR), name="colour")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        dec = next(x for x in d["decisions"] if x["subject"] == "color.primary_correction")
        self.assertEqual((dec["type"], dec["provenance"], dec["approval"]), ("TRANSFORM", "USER", "AUTO"))
        self.assertEqual(sorted(x["key"] for x in d["requirements"] if x["id"] in dec["evidence"]), sorted(NEW_COLOR))
        op = d["color"]["operations"][0]
        self.assertEqual({k.split(".", 1)[1]: op.get(k.split(".", 1)[1]) for k in NEW_COLOR}, {k.split(".", 1)[1]: v for k, v in NEW_COLOR.items()})
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "color_primary_correction")
        self.assertEqual({k: v for k, v in step["params"].items() if k != "asset"}, {k.split(".", 1)[1]: v for k, v in NEW_COLOR.items()})
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = next(x for x in out["execution"]["results"] if x["tool"] == "color-grading/run")
        for k, v in NEW_COLOR.items():
            self.assertEqual(res["data"]["operation"]["parameters"][k.split(".", 1)[1]], v, k)
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        row = next(e for e in prov["operations"] if e["tool"] == "color-grading/run")
        self.assertEqual(row["args"]["operation"], "PRIMARY_CORRECTION"); self.assertEqual(row["args"]["curves"], "medium_contrast")
        self.assertEqual(row["decision"], [dec["id"]])
        clear_modes()

    def test_tampered_ir_out_of_range_is_refused_by_the_schema(self):
        clear_modes()
        tmp = os.path.realpath(tempfile.mkdtemp())
        svc = pipeline_service(tmp)
        ir = svc.plan([pipeline_media(tmp, "a.mp4")], "youtube", user_requirements={"color.gamma": 1.2})
        ir.doc["color"]["operations"][0]["gamma"] = 50.0
        ir.doc["color"]["operations"][0]["curves"] = "sepia"
        errs = " ".join(svc.validate(ir).errors)
        self.assertIn("gamma", errs); self.assertIn("curves", errs)
        clear_modes()


class LoweringTests(unittest.TestCase):
    """video-editing lowering of the contract's ROTATE / FREEZE / REVERSE (and the compiler's TRIM lowering)."""

    def test_every_declared_type_is_lowered(self):
        lw = Lowering(editing_contract())
        self.assertEqual(lw.contract_only(), [])
        self.assertTrue({"ROTATE", "FREEZE", "REVERSE"} <= set(lw.supported_types()))
        for t in ("ROTATE", "FREEZE", "REVERSE"):   # the agent-side keys stay inside what the contract declares
            self.assertTrue(set(ARGS[t]) - {"input", "output"} <= set(editing_contract()["operations"][t]["parameters"]), t)

    def test_params_and_refusals(self):
        self.assertEqual(params_for("ROTATE", {"degrees": 90}), {"degrees": 90})
        self.assertEqual(params_for("ROTATE", {"flip": "v"}), {"flip": "v"})
        self.assertEqual(params_for("FREEZE", {"hold": 1.5, "at": 2}), {"hold": 1.5, "at": 2.0})
        self.assertEqual(params_for("REVERSE", {}), {})
        from video_agent.tools.base import ToolError
        for t, args in (("ROTATE", {}), ("ROTATE", {"degrees": 45}), ("ROTATE", {"degrees": "90"}), ("ROTATE", {"flip": "x"}), ("FREEZE", {}), ("FREEZE", {"hold": -1})):
            with self.assertRaises(ToolError, msg=f"{t} {args}"):
                params_for(t, args)

    def test_build_request(self):
        tmp = tempfile.mkdtemp()
        lw = Lowering(editing_contract(), workspace=tmp)
        req, out, ins = lw.build_request("video-editing/rotate", {"input": "a", "output": "b", "degrees": 270, "flip": "h"}, {"a": __file__, "b": str(Path(tmp) / "o" / "b.mp4")})
        op = req["project"]["operations"][0]
        self.assertEqual((op["type"], op["params"], op["input"]), ("ROTATE", {"degrees": 270, "flip": "h"}, "s0"))
        from video_agent.tools.base import ToolError
        with self.assertRaises(ToolError):
            lw.build_request("video-editing/rotate", {"input": "a", "output": "b", "degrees": 90, "filter": "transpose=1"}, {"a": __file__, "b": str(Path(tmp) / "b.mp4")})

    def test_trim_lowering(self):
        op = {"type": "video.trim", "asset": "a", "keep": [[2.0, 10.5]], "accurate": True}
        self.assertEqual(lower_video_trim("video-editing/trim", op, "a", "a_trim"), {"input": "a", "start": 2.0, "end": 10.5, "precision": "frame", "output": "a_trim"})
        with self.assertRaises(CompileError):
            lower_video_trim("video-editing/trim", dict(op, keep=[[0.0, 2.0], [3.0, 5.0]]), "a", "a_trim")
        # the same IR record on the other tools keeps its existing shapes
        self.assertEqual(lower_video_trim("video-editing/cut", op, "a", "a_trim")["keep"], [[2.0, 10.5]])
        self.assertEqual(lower_video_trim("ffmpeg-skill/cut", op, "a", "a_trim")["segments"], "2.000-10.500")


class TrimRotateTests(unittest.TestCase):
    """edit.trim / edit.rotate (+ .flip): requirement → decision → IR → plan step → video-editing request, on the fake video-editing-skill."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.a = fake_media(self.tmp, "a.mp4")
        self.b = fake_media(self.tmp, "b.mp4")
        os.environ.pop("FAKE_VE_MODE", None)

    def _svc(self, caps=None):
        from video_agent.tools import ToolRouter
        from video_agent.tools.video_editing import VideoEditingAdapter, VideoEditingSkill
        ad = VideoEditingAdapter(VideoEditingSkill([sys.executable, FAKE_VE], None, {}), workspace=self.tmp, allowed_inputs=[str(Path(self.a).parent)], ffmpeg_skill_dir=self.tmp)
        return make_service(self.tmp, caps=caps or FakeCaps(extra=["video-editing", "encoder:aac"]), adapter=ToolRouter([FakeAdapter(), ad]))

    def _render(self, svc, ir, name):
        p = str(Path(self.tmp) / f"{name}.json"); save_ir(ir, p)
        return svc.render(load_ir(p), p, approve=["all"])

    def test_requirement_vocabulary(self):
        def parse(**kv):
            return {k: v["params"] for k, v in parse_edit_requirements(reqs(kv)).items()}
        self.assertEqual(parse(**{"edit.trim": "2.5-14"}), {"video.trim": {"start": 2.5, "end": 14.0}})
        self.assertEqual(parse(**{"edit.trim": "0:05-1:30"}), {"video.trim": {"start": 5.0, "end": 90.0}})
        self.assertEqual(parse(**{"edit.trim": {"start": 1, "end": 3}}), {"video.trim": {"start": 1.0, "end": 3.0}})
        self.assertEqual(parse(**{"edit.rotate": 90}), {"video.rotate": {"degrees": 90}})
        self.assertEqual(parse(**{"edit.rotate": "270", "edit.rotate.flip": "v"}), {"video.rotate": {"degrees": 270, "flip": "v"}})
        self.assertEqual(parse(**{"edit.rotate": 0, "edit.rotate.flip": "h"}), {"video.rotate": {"flip": "h"}}, "a mirror alone")
        for bad, msg in (({"edit.trim": "5-2"}, "before"), ({"edit.trim": "abc"}, "START-END"), ({"edit.trim": {"start": 1}}, "START-END"), ({"edit.trim": 5}, "START-END"),
                         ({"edit.rotate": 45}, "90, 180 or 270"), ({"edit.rotate": 360}, "within"), ({"edit.rotate": 0}, "changes nothing"),
                         ({"edit.rotate.flip": "h"}, "ambiguous"), ({"edit.rotate": 90, "edit.rotate.flip": "x"}, "h (mirror")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                parse(**bad)
            self.assertIn(msg, str(cm.exception), str(bad))
            with self.assertRaises(ValueError, msg=str(bad)):
                _check_requirement_keys(bad)

    def test_trim_decision_plan_ir_and_request(self):
        svc = self._svc()
        ir = svc.plan([self.a], "generic", user_requirements={"edit.trim": "4-12"})
        d = ir.doc
        aid = list(d["assets"])[0]
        self.assertEqual((d["plan"]["status"], svc.validate(ir).errors), ("APPROVED", []))
        dec = next(x for x in d["decisions"] if x["subject"] == "video.trim")
        self.assertEqual((dec["type"], dec["approval"], dec["provenance"], dec["params"]), ("TRANSFORM", "AUTO", "USER", {"asset_id": aid, "start": 4.0, "end": 12.0}))
        self.assertEqual(dec["basis"]["approval"]["key"], "video.trim.approval")
        trim = next(op for op in d["video"]["operations"] if op["type"] == "video.trim")
        self.assertEqual((trim["keep"], trim["accurate"]), ([[4.0, 12.0]], True), "the user's in / out points, cut where they were given")
        self.assertIn(dec["id"], trim["decision_ids"])
        step = next(s for s in d["plan"]["steps"] if s["id"] == f"step_trim_{aid}")
        self.assertEqual((step["skill"], step["tool"]), ("video_trim", "video-editing/trim"))
        out = self._render(svc, ir, "trim")
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = [r for r in out["execution"]["results"] if r["tool"].startswith("video-editing/")]
        self.assertEqual([r["tool"] for r in res], ["video-editing/trim"])
        prov = json.loads((Path(self.tmp) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        row = next(e for e in prov["operations"] if e["tool"] == "video-editing/trim")
        self.assertEqual((row["skill"], sorted(row["args"])), ("video_trim", ["end", "input", "output", "precision", "start"]))
        self.assertEqual((row["args"]["start"], row["args"]["end"], row["args"]["precision"]), (4.0, 12.0, "frame"))
        self.assertIn(dec["id"], row["decision"])
        dur = next(i for i in out["qa"]["items"] if i["name"] == "duration")
        self.assertEqual(dur["status"], "PASS"); self.assertAlmostEqual(float(dur["observed"]), 8.0, delta=0.01)

    def test_trim_narrows_the_silence_trim(self):
        """Silence decisions still apply inside the range: the kept range is their intersection, one video.trim, one step."""
        svc = self._svc()
        ir = svc.plan([self.a], "generic", user_requirements={"edit.trim": "1-15"})   # the fake's silences end at 2.85 s and start at 13.85 s
        d = ir.doc
        trims = [op for op in d["video"]["operations"] if op["type"] == "video.trim"]
        self.assertEqual(len(trims), 1); self.assertEqual(trims[0]["keep"], [[2.85, 13.85]])
        subjects = {x["subject"] for x in d["decisions"] if x["id"] in trims[0]["decision_ids"]}
        self.assertEqual(subjects, {"video.trim", "silence.leading", "silence.trailing"})
        self.assertEqual([s["skill"] for s in d["plan"]["steps"] if s["id"].startswith("step_trim_")], ["video_trim"])

    def test_trim_blocks(self):
        svc = self._svc()
        ir = svc.plan([self.a, self.b], "generic", user_requirements={"edit.trim": "1-5"})
        blk = [x for x in ir.doc["decisions"] if x["approval"] == "BLOCK"]
        self.assertEqual([x["subject"] for x in blk], ["video.trim"]); self.assertIn("2 inputs", blk[0]["reason"])
        ir = svc.plan([self.a], "generic", user_requirements={"edit.trim": "1-30"})
        blk = [x for x in ir.doc["decisions"] if x["approval"] == "BLOCK"]
        self.assertEqual([x["subject"] for x in blk], ["video.trim"]); self.assertIn("not clamped", blk[0]["reason"])
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        self.assertFalse(any(s["skill"] == "video_trim" for s in ir.doc["plan"]["steps"]))
        # without video-editing the explicit trim is not silently done by another engine
        svc2 = self._svc(caps=FakeCaps(extra=["encoder:aac"]))
        ir = svc2.plan([self.a], "generic", user_requirements={"edit.trim": "1-5"})
        self.assertIn("capability.video_trim", {x["subject"] for x in ir.doc["decisions"] if x["approval"] == "BLOCK"})

    def test_rotate_decision_plan_ir_and_request(self):
        svc = self._svc()
        ir = svc.plan([self.a], "generic", user_requirements={"edit.rotate": 90, "edit.rotate.flip": "h", "edit.fit": "9:16"})
        d = ir.doc
        aid = list(d["assets"])[0]
        self.assertEqual((d["plan"]["status"], svc.validate(ir).errors), ("APPROVED", []))
        self.assertEqual([op["type"] for op in d["video"]["operations"] if op["type"] != "video.trim"], ["video.rotate", "video.fit"], "rotate before fit (EDIT_ORDER)")
        rot = next(op for op in d["video"]["operations"] if op["type"] == "video.rotate")
        self.assertEqual((rot["degrees"], rot["flip"], rot["input"], rot["output"]), (90, "h", f"{aid}_trim", f"{aid}_rotate"))
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "video_rotate")
        self.assertEqual((step["tool"], step["params"]), ("video-editing/rotate", {"asset": aid, "degrees": 90, "flip": "h"}))
        dec = next(x for x in d["decisions"] if x["subject"] == "video.rotate")
        self.assertEqual((dec["type"], dec["approval"], dec["basis"]["approval"]["key"]), ("TRANSFORM", "AUTO", "video.rotate.approval"))
        out = self._render(svc, ir, "rotate")
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        self.assertEqual([r["tool"] for r in out["execution"]["results"] if r["tool"].startswith("video-editing/")], ["video-editing/rotate", "video-editing/fit"])
        prov = json.loads((Path(self.tmp) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        row = next(e for e in prov["operations"] if e["tool"] == "video-editing/rotate")
        self.assertEqual({k: row["args"][k] for k in ("degrees", "flip")}, {"degrees": 90, "flip": "h"}); self.assertEqual(row["decision"], [dec["id"]])

    def test_picture_size_and_validator(self):
        tech = {"video": {"width": 1280, "height": 720}}
        self.assertEqual(picture_size(tech, [{"type": "video.rotate", "asset": "a", "degrees": 90}], "a"), (720, 1280))
        self.assertEqual(picture_size(tech, [{"type": "video.rotate", "asset": "a", "degrees": 180}], "a"), (1280, 720))
        self.assertEqual(picture_size(tech, [{"type": "video.rotate", "asset": "a", "flip": "h"}], "a"), (1280, 720))
        svc = self._svc()
        ir = svc.plan([self.a], "generic", user_requirements={"edit.rotate": 180})
        rot = next(op for op in ir.doc["video"]["operations"] if op["type"] == "video.rotate")
        rot["degrees"] = 45
        self.assertTrue(any("degrees" in e for e in svc.validate(ir).errors))
        del rot["degrees"]
        self.assertTrue(any("needs degrees and / or flip" in e for e in svc.validate(ir).errors))


if __name__ == "__main__":
    unittest.main()
