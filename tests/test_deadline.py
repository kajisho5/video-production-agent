"""Deadline mode (撮って出し; ADR-133, roadmap step 22): `plan --deadline`. The estimate of every compiled step comes from this machine's speed
book (written by render); optional finishing that would miss the deadline is dropped and the plan says so; a plan whose required steps alone
miss it is marked at risk at plan time. Against the fake engine and fake Skills (tests/pipeline_harness.py)."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import fake_media, pipeline_service  # noqa: E402

from video_agent.agent import deadline as dl  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402


class ParseTests(unittest.TestCase):
    def test_durations_and_clock_times(self):
        self.assertEqual(dl.parse_deadline("30m"), 1800.0)
        self.assertEqual(dl.parse_deadline("1h30m"), 5400.0)
        self.assertEqual(dl.parse_deadline("90s"), 90.0)
        self.assertEqual(dl.parse_deadline("10"), 600.0)
        now = time.mktime((2026, 11, 26, 12, 0, 0, 0, 0, -1))
        self.assertEqual(dl.parse_deadline("12:30", now=now), 1800.0)
        self.assertEqual(dl.parse_deadline("11:00", now=now), 23 * 3600.0, "a time already past today is tomorrow's")
        for bad in ("", "soon", "0m", "25:00", "-5m"):
            with self.subTest(bad=bad), self.assertRaises(dl.DeadlineError):
                dl.parse_deadline(bad)

    def test_speed_book_fit(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        book = dl.SpeedBook(tmp)
        self.assertIsNone(book.rate("m", "t"))
        book.record("m", "t", 3.0, 10.0)
        self.assertEqual(book.rate("m", "t"), (0.0, 0.3, 1))
        book.record("m", "t", 5.0, 20.0)       # 1 s overhead + 0.2 s per media second
        a, b, n = book.rate("m", "t")
        self.assertEqual((round(a, 6), round(b, 6), n), (1.0, 0.2, 2))
        book.save()
        self.assertEqual(dl.SpeedBook(tmp).runs("m", "t"), book.runs("m", "t"))
        for _ in range(dl.KEEP_RUNS + 5):
            book.record("m", "t", 1.0, 1.0)
        self.assertEqual(len(book.runs("m", "t")), dl.KEEP_RUNS)
        self.assertNotIn("node", json.dumps(book.doc))


class PlanTests(unittest.TestCase):
    REQS = {"color.target": "bt709", "thumbnail": True, "qc": True}

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.src = fake_media(self.tmp, "keynote.mp4", duration=600.0)
        self.svc = pipeline_service(self.tmp)

    def plan(self, deadline, reqs=None):
        ir = self.svc.plan([self.src], "youtube", user_requirements=dict(reqs or self.REQS), params={"language": "ja"})
        rec = self.svc.apply_deadline(ir, deadline)
        return ir, rec

    def seed(self, color_rate, other_rate=0.05):
        """Earlier runs on this machine: colour grading at `color_rate` seconds per media second, every other tool of the plan at `other_rate`."""
        ir = self.svc.plan([self.src], "youtube", user_requirements=dict(self.REQS), params={"language": "ja"})
        with tempfile.TemporaryDirectory() as d:
            from video_agent.execution import compile_ir
            tools = {op.tool for op in compile_ir(ir, d)[0]}
        book = dl.SpeedBook(self.svc.workspace)
        for tool in tools:
            book.record(dl.machine_key(self.svc._hw_encoder()), tool, (color_rate if tool == "color-grading/run" else other_rate) * 600.0, 600.0)
        book.save()

    def test_a_ten_minute_budget_drops_the_colour_step_and_says_so(self):
        """Done when (step 22): a fixture with a 10-minute budget drops the colour step and says so."""
        self.seed(1.2)   # colour grading ran at 1.2x real time here before: 720 s for this 600 s programme
        ir, rec = self.plan("10m")
        self.assertEqual(ir.doc["color"]["operations"], [])
        [drop] = rec["dropped"]
        self.assertEqual((drop["group"], drop["operations"], drop["estimated_seconds"]), ("color", ["color.retag"], 720.0))
        self.assertIn("of the 600 s to the deadline", drop["reason"])
        self.assertEqual(rec["status"], "fits")
        self.assertLessEqual(rec["estimated_seconds"], 600.0)
        self.assertTrue(any("dropped color" in s for s in ir.doc["plan"]["summary"]), ir.doc["plan"]["summary"])
        self.assertFalse([s for s in rec["steps"] if s["group"] == "color"])
        self.assertEqual(rec["defaults"], [])
        rep = self.svc.validate(ir)
        self.assertEqual(rep.errors, [])
        path = str(Path(self.svc.workspace) / "plans" / "dl.project.json")
        save_ir(ir, path)
        out = self.svc.render(load_ir(path), path, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED")
        self.assertNotIn("color-grading/run", [r["tool"] for r in out["execution"]["results"]])
        self.assertIn("thumbnail/extract_frame", [r["tool"] for r in out["execution"]["results"]], "the thumbnail is a deliverable, not finishing")

    def test_a_budget_that_fits_keeps_everything(self):
        self.seed(0.1)
        ir, rec = self.plan("30m")
        self.assertEqual((rec["status"], rec["dropped"]), ("fits", []))
        self.assertEqual([op["type"] for op in ir.doc["color"]["operations"]], ["color.retag"])

    def test_without_a_speed_book_the_estimate_is_a_stated_default(self):
        _, rec = self.plan("2h")
        self.assertTrue(rec["defaults"] and all(s["basis"] == "default" for s in rec["steps"]))
        self.assertEqual(rec["status"], "fits")

    def test_required_steps_past_the_deadline_are_at_risk_at_plan_time(self):
        ir, rec = self.plan("20s")
        self.assertEqual(rec["status"], "at_risk")
        self.assertIn("past the deadline", rec["warning"])
        self.assertTrue(any(w.startswith("deadline: ") for w in ir.doc["analysis"]["warnings"]))
        self.assertEqual(self.svc.validate(ir).errors, [])

    def test_render_writes_the_speed_book_and_the_next_estimate_is_measured(self):
        ir = self.svc.plan([self.src], "youtube", user_requirements=dict(self.REQS), params={"language": "ja"})
        path = str(Path(self.svc.workspace) / "plans" / "first.project.json")
        save_ir(ir, path)
        out = self.svc.render(load_ir(path), path, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED")
        book = dl.SpeedBook(self.svc.workspace)
        machine = dl.machine_key(self.svc._hw_encoder())
        ran = {r["tool"] for r in out["execution"]["results"] if r["ok"]}
        self.assertEqual(set((book.doc["machines"].get(machine) or {})), ran)
        _, rec = self.plan("30m")
        self.assertEqual(rec["defaults"], [])
        self.assertTrue(all(s["basis"] == "measured" for s in rec["steps"]))


if __name__ == "__main__":
    unittest.main()
