"""FREEZE / REVERSE as plan steps (ADR-141, roadmap step 17): `edit.freeze` (seconds held, `edit.freeze.at`) and `edit.reverse` each become one
decision (the explicit request approves it, as for edit.speed), one plan step and one IR operation run by video-editing-skill FREEZE / REVERSE; nothing is planned without the request.
Fake engine + fake video-editing; no media is decoded."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import clear_modes, fake_media, pipeline_service  # noqa: E402
from video_agent.agent.editing import EditRequirementError, parse_edit_requirements  # noqa: E402
from video_agent.agent.requirements import requirement_map  # noqa: E402
from video_agent.models import Requirement  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402


def reqmap(**kv):
    return requirement_map([Requirement(key=k.replace("__", "."), value=v, provenance="USER", source="cli") for k, v in kv.items()])


class RequirementTests(unittest.TestCase):
    def test_keys(self):
        self.assertEqual(parse_edit_requirements(reqmap(edit__freeze=2, edit__freeze__at="3.5"))["video.freeze"]["params"], {"hold": 2.0, "at": 3.5})
        self.assertEqual(parse_edit_requirements(reqmap(edit__freeze="1.5"))["video.freeze"]["params"], {"hold": 1.5})
        self.assertIn("video.reverse", parse_edit_requirements(reqmap(edit__reverse=True)))
        self.assertNotIn("video.reverse", parse_edit_requirements(reqmap(edit__reverse=False)))
        for bad in ({"edit__freeze": 0}, {"edit__freeze": 61}, {"edit__freeze__at": 1}, {"edit__reverse": "sometimes"}, {"edit__freeze": 1, "edit__freeze__at": -1}):
            with self.assertRaises(EditRequirementError, msg=str(bad)):
                parse_edit_requirements(reqmap(**bad))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.src = fake_media(self.tmp, "clip.mp4", duration=10.0, silences=[])

    def plan(self, reqs):
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self.src], "youtube", user_requirements=reqs)
        path = str(Path(svc.workspace) / "plans" / "p.project.json")
        save_ir(ir, path)
        return svc, load_ir(path), path

    def test_a_step_per_key_confirmed_then_rendered(self):
        svc, ir, path = self.plan({"edit.reverse": True, "edit.freeze": 2.0, "edit.freeze.at": 4.0})
        self.assertEqual(svc.validate(ir).errors, [])
        doc = ir.doc
        decs = {d["subject"]: d for d in doc["decisions"] if d["subject"] in ("video.reverse", "video.freeze")}
        # the explicit request is the approval, as for edit.speed (policy video.<op>.approval = CONFIRM still asks a person)
        self.assertEqual({k: (d["type"], d["approval"]) for k, d in decs.items()}, {"video.reverse": ("TRANSFORM", "AUTO"), "video.freeze": ("TRANSFORM", "AUTO")})
        steps = [s for s in doc["plan"]["steps"] if s["skill"] in ("video_reverse", "video_freeze")]
        self.assertEqual([(s["skill"], s["tool"]) for s in steps], [("video_reverse", "video-editing/reverse"), ("video_freeze", "video-editing/freeze")], "reverse, then freeze")
        fr = next(op for op in doc["video"]["operations"] if op["type"] == "video.freeze")
        self.assertEqual((fr["hold"], fr["at"]), (2.0, 4.0))
        self.assertEqual(steps[1]["temporal_scope"]["end"], 12.0, "the hold lengthens the picture")
        exp = next(s for s in doc["plan"]["steps"] if s["skill"] == "delivery_export")
        self.assertEqual(exp["temporal_scope"]["end"], 12.0)
        out = svc.render(load_ir(path), path, approve=["all"])
        self.assertIn(out["status"], ("COMPLETED", "REVIEW"), out.get("execution", {}).get("failed_op"))
        tools = [r["tool"] for r in out["execution"]["results"]]
        self.assertLess(tools.index("video-editing/reverse"), tools.index("video-editing/freeze"))

    def test_nothing_without_the_request(self):
        _, ir, _ = self.plan({})
        self.assertFalse([s for s in ir.doc["plan"]["steps"] if s["skill"] in ("video_reverse", "video_freeze")])

    def test_chapters_are_blocked_by_a_freeze(self):
        _, ir, _ = self.plan({"edit.freeze": 1.0, "delivery.chapters": True})
        d = next(x for x in ir.doc["decisions"] if x["subject"] == "chapters.transcript")
        self.assertEqual(d["type"], "BLOCK")


if __name__ == "__main__":
    unittest.main()
