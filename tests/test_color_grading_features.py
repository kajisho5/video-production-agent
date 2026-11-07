"""ADR-100: color-grading-skill 0.5.0's OCIO_TRANSFORM (`color.ocio`), MATCH_CAMERA (`color.match_camera`) and LEGALIZE (`color.legalize`)
wired end to end on the fake engine and the fake color-grading-skill: vocabulary read from the pinned contract, the doctor's per-operation
status gating each one (BLOCK with what it needs), the IR order, the plan step, the request (the reference video through the input roots, a
.ocio file through the LUT roots) and the HDR tone map left at the Skill's default. tests/test_adapter_color_grading.py runs the real Skill."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import PipelineCaps, clear_modes, fake_media, pipeline_service, plan_and_render  # noqa: E402
from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.finishing import COLOR_OPERATIONS, COLOR_ORDER, parse_color_requirements  # noqa: E402
from video_agent.agent.requirements import requirement_map  # noqa: E402
from video_agent.models import Requirement  # noqa: E402
from video_agent.tools.color_grading.adapter import pinned_contract  # noqa: E402

NEW = ["color-grading:OCIO_TRANSFORM", "color-grading:MATCH_CAMERA", "color-grading:LEGALIZE"]


def reqs(d):
    return requirement_map([Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in d.items()])


class VocabularyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ref = fake_media(self.tmp, "camera_a.mp4")
        self.cfg = str(Path(self.tmp) / "show.ocio")
        Path(self.cfg).write_text("ocio_profile_version: 2\n", encoding="utf-8")

    def test_parameters_are_the_contracts(self):
        ops = {o["type"]: o for o in pinned_contract()["operations"]}
        for op, typ in (("color.ocio", "OCIO_TRANSFORM"), ("color.match_camera", "MATCH_CAMERA"), ("color.legalize", "LEGALIZE")):
            self.assertEqual(COLOR_OPERATIONS[op]["type"], typ)
            self.assertEqual(set(COLOR_OPERATIONS[op]["params"]), set(ops[typ]["parameters"]) - {"crf", "preset", "audio_stream"}, "encode settings are never keys")
        self.assertEqual(COLOR_ORDER, ("color.strip_dovi", "color.hdr_to_sdr", "color.ocio", "color.match_camera", "color.primary_correction", "color.lut", "color.legalize", "color.retag"))

    def test_parse(self):
        got = parse_color_requirements(reqs({"color.ocio": "S-Log3 S-Gamut3.Cine", "color.ocio.display": "Rec.1886 Rec.709 - Display", "color.ocio.view": "ACES 2.0 - SDR 100 nits (Rec.709)",
                                             "color.ocio.lut_size": 129}))["color.ocio"]["params"]
        self.assertEqual(got, {"display": "Rec.1886 Rec.709 - Display", "view": "ACES 2.0 - SDR 100 nits (Rec.709)", "lut_size": 129, "source_space": "S-Log3 S-Gamut3.Cine"})
        got = parse_color_requirements(reqs({"color.ocio": "ACEScct", "color.ocio.destination_space": "Linear Rec.709", "color.ocio.output_tags": "bt709", "color.ocio.config": self.cfg}))
        self.assertEqual(got["color.ocio"]["params"]["config"], os.path.abspath(self.cfg))
        got = parse_color_requirements(reqs({"color.match_camera": self.ref, "color.match_camera.model": "matrix", "color.match_camera.at": 2.5,
                                             "color.match_camera.patches": "0.1,0.1,0.2,0.2;0.5,0.5,0.1,0.1/0.4,0.5,0.1,0.1"}))["color.match_camera"]["params"]
        self.assertEqual((got["reference_path"], got["model"], got["at"]), (os.path.abspath(self.ref), "matrix", 2.5))
        self.assertEqual(parse_color_requirements(reqs({"color.legalize": True}))["color.legalize"]["params"], {})
        self.assertEqual(parse_color_requirements(reqs({"color.legalize": "ebu-r103", "color.legalize.knee": 0.9}))["color.legalize"]["params"], {"knee": 0.9, "profile": "ebu-r103"})
        self.assertNotIn("color.legalize", parse_color_requirements(reqs({"color.legalize": False})))

    def test_refusals(self):
        for bad, msg in (({"color.ocio": "ACEScct"}, "destination_space, or"), ({"color.ocio": "ACEScct", "color.ocio.display": "X"}, "destination_space, or"),
                         ({"color.ocio": "ACEScct", "color.ocio.destination_space": "Some Space"}, "output_tags is required"),
                         ({"color.ocio": "ACEScct", "color.ocio.destination_space": "Camera Rec.709", "color.ocio.lut_size": 30}, "one of"),
                         ({"color.ocio": "ACEScct", "color.ocio.destination_space": "Camera Rec.709", "color.ocio.config": "ocio://../x"}, "built-in"),
                         ({"color.ocio": "ACEScct", "color.ocio.destination_space": "Camera Rec.709", "color.sdr": True}, "both colour-space conversions"),
                         ({"color.ocio.view": "v"}, "ambiguous"), ({"color.match_camera": "/nope.mp4"}, "not found"),
                         ({"color.match_camera": None, "color.match_camera.model": "x"}, "must be a file path"),
                         ({"color.legalize": "bt601"}, "one of"), ({"color.legalize": True, "color.legalize.knee": 0.5}, "within"),
                         ({"color.legalize.knee": 0.9}, "ambiguous")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                parse_color_requirements(reqs(bad))
            self.assertIn(msg, str(cm.exception))


class PlanTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = tempfile.mkdtemp()
        self.a = fake_media(self.tmp, "a.mp4")
        self.ref = fake_media(self.tmp, "camera_a.mp4")
        self.cfg = str(Path(self.tmp) / "src" / "show.ocio")
        Path(self.cfg).write_text("ocio_profile_version: 2\n", encoding="utf-8")

    def tearDown(self):
        clear_modes()

    def test_all_three_end_to_end_in_order(self):
        svc = pipeline_service(self.tmp, caps=PipelineCaps(extra=NEW))
        r = plan_and_render(svc, [self.a], {"color.ocio": "ACEScct", "color.ocio.destination_space": "Camera Rec.709", "color.ocio.config": self.cfg,
                                            "color.match_camera": self.ref, "color.match_camera.at": 1.0, "color.legalize": "bt709", "color.target": "bt709"}, name="c1")
        d = r["ir"].doc
        self.assertEqual((d["plan"]["status"], r["validation"].errors), ("APPROVED", []), d["plan"]["summary"])
        self.assertEqual([op["type"] for op in d["color"]["operations"]], ["color.ocio", "color.match_camera", "color.legalize", "color.retag"])
        steps = {s["skill"]: s for s in d["plan"]["steps"]}
        self.assertTrue(all(steps[k]["tool"] == "color-grading/run" for k in ("color_ocio", "color_match_camera", "color_legalize")))
        self.assertIn(str(Path(self.ref).resolve().parent), d["execution"]["allowed_inputs"])
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        rows = {e["skill"]: e for e in prov["operations"] if e["tool"] == "color-grading/run"}
        self.assertEqual(rows["color_ocio"]["args"]["operation"], "OCIO_TRANSFORM")
        self.assertEqual((rows["color_match_camera"]["args"]["operation"], rows["color_match_camera"]["args"]["reference_path"]), ("MATCH_CAMERA", os.path.abspath(self.ref)))
        self.assertEqual((rows["color_legalize"]["args"]["operation"], rows["color_legalize"]["args"]["profile"]), ("LEGALIZE", "bt709"))
        for k in ("crf", "preset", "audio_stream"):
            self.assertNotIn(k, rows["color_ocio"]["args"])

    def test_doctor_per_operation_status_blocks_with_what_it_needs(self):
        svc = pipeline_service(self.tmp, caps=PipelineCaps(extra=["color-grading:MATCH_CAMERA", "color-grading:LEGALIZE"], missing=["color-grading:OCIO_TRANSFORM"]))
        ir = svc.plan([self.a], "youtube", user_requirements={"color.ocio": "ACEScct", "color.ocio.destination_space": "Camera Rec.709", "color.legalize": True})
        blk = next(x for x in ir.doc["decisions"] if x["subject"] == "capability.color_ocio")
        self.assertEqual(blk["approval"], "BLOCK"); self.assertIn("PyOpenColorIO", blk["reason"])
        self.assertFalse(any(x["subject"] == "capability.color_legalize" for x in ir.doc["decisions"]), "LEGALIZE is supported here")
        svc2 = pipeline_service(tempfile.mkdtemp(), caps=PipelineCaps(missing=["color-grading:MATCH_CAMERA", "color-grading:LEGALIZE"]))
        ir2 = svc2.plan([self.a], "youtube", user_requirements={"color.legalize": True})
        blk = next(x for x in ir2.doc["decisions"] if x["subject"] == "capability.color_legalize")
        self.assertIn("3.1.6", blk["reason"]); self.assertEqual(ir2.doc["plan"]["status"], "BLOCKED")

    def test_hdr_tone_map_keeps_the_skills_default(self):
        hdr = fake_media(self.tmp, "hlg.mp4", hdr=True)
        svc = pipeline_service(self.tmp, caps=PipelineCaps(extra=NEW), hdr=True)   # the fake engine probes it as HLG
        r = plan_and_render(svc, [hdr], {"color.sdr": True, "color.legalize": True}, name="c2")
        d = r["ir"].doc
        tm = next(op for op in d["color"]["operations"] if op["type"] == "color.hdr_to_sdr")
        self.assertNotIn("tonemap", tm, "the agent sends no tone-map curve: the Skill's default is kept")
        self.assertEqual([op["type"] for op in d["color"]["operations"]], ["color.hdr_to_sdr", "color.legalize"])
        prov = json.loads((Path(svc.workspace) / "jobs" / r["out"]["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        row = next(e for e in prov["operations"] if e["skill"] == "color_hdr_to_sdr")
        self.assertNotIn("tonemap", row["args"])


if __name__ == "__main__":
    unittest.main()
