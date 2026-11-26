"""Chunked render of a long programme (ADR-134, roadmap step 23): the delivery export runs chunk by chunk (frame-accurate cut + export per
chunk, joined by stream copy), the job records every finished operation as it goes (the checkpoint), and `render --resume` restarts at the
first unfinished chunk. Against the fake engine and fake Skills (tests/pipeline_harness.py); the real-engine run with a kill in chunk 3 is
tests/test_integration.py::ChunkedRenderRealTests."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter  # noqa: E402
from pipeline_harness import fake_media, pipeline_service  # noqa: E402

from video_agent.agent import chunks as ck  # noqa: E402
from video_agent.agent import deadline as dl  # noqa: E402
from video_agent.execution import compile_ir  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402


class BoundaryTests(unittest.TestCase):
    def test_whole_frames_and_the_last_chunk_runs_to_the_end(self):
        got = ck.boundaries(7200.0, 600.0, Fraction(30))
        self.assertEqual(len(got), 12)
        self.assertEqual([c["start_frame"] for c in got[:3]], [0, 18000, 36000])
        self.assertEqual((got[2]["start"], got[2]["end"], got[-1]["end"]), (1200.0, 1800.0, None))
        for a, b in zip(got, got[1:]):
            self.assertEqual(a["end"], b["start"], "chunks touch: no frame is in two chunks or in none")

    def test_ntsc_rate_boundaries_land_on_frames_rounded_down(self):
        rate = Fraction(30000, 1001)
        got = ck.boundaries(3600.0, 600.0, rate)
        for c in got[1:]:
            exact = Fraction(c["start_frame"]) / rate
            self.assertLessEqual(c["start"], float(exact))
            self.assertLess(float(exact) - c["start"], 1e-6)
            self.assertEqual(c["start_frame"], round(Fraction(c["index"] - 1) * 600 * rate))

    def test_a_short_tail_joins_the_chunk_before(self):
        self.assertEqual(len(ck.boundaries(1210.0, 600.0, Fraction(25))), 2, "a 10 s tail is not a chunk of its own")
        self.assertEqual(len(ck.boundaries(1400.0, 600.0, Fraction(25))), 3)

    def test_chunk_seconds_option(self):
        self.assertIsNone(ck.check_chunk_seconds(None))
        self.assertEqual(ck.check_chunk_seconds(0), 0.0)
        self.assertEqual(ck.check_chunk_seconds("300"), 300.0)
        for bad in ("x", 5, -1, 10_000, float("nan")):
            with self.subTest(bad=bad), self.assertRaises(ck.ChunkError):
                ck.check_chunk_seconds(bad)

    def test_rate_of(self):
        self.assertEqual(ck.rate_of({"fps": 29.97002997}), Fraction(30000, 1001))
        self.assertEqual(ck.rate_of({"fps": "25/1"}), Fraction(25))
        self.assertIsNone(ck.rate_of({"fps": None}))
        self.assertIsNone(ck.rate_of(None))


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def plan(self, duration=7200.0, profile="youtube", engine=None, **kw):
        src = fake_media(self.tmp, "allday.mp4", duration=duration, silences=[])
        svc = pipeline_service(self.tmp, engine=engine)
        return svc, svc.plan([src], profile, **kw)

    def test_a_two_hour_programme_is_chunked_by_default(self):
        _, ir = self.plan()
        ch = ir.doc["chunked_render"]
        self.assertEqual((ch["rule"], ch["rate"], ch["chunk_seconds"], len(ch["chunks"]), ch["targets"]), ("chunked_render@1", "30/1", 600.0, 12, ["youtube"]))
        self.assertEqual(ch["tools"], {"cut": "ffmpeg-skill/cut", "join": "ffmpeg-skill/join"})
        self.assertTrue(any("12 chunks" in line for line in ir.doc["plan"]["summary"]))
        self.assertIn("chunked_render", json.dumps(ir.doc))

    def test_compiled_ops_per_chunk_then_one_copy_join(self):
        _, ir = self.plan()
        with tempfile.TemporaryDirectory() as d:
            ops, paths = compile_ir(ir, d)
        cuts = [o for o in ops if o.skill == "render_chunk"]
        exports = [o for o in ops if o.skill == "delivery_export"]
        joins = [o for o in ops if o.chunk_join]
        self.assertEqual((len(cuts), len(exports), len(joins)), (12, 12, 1))
        self.assertEqual((cuts[2].args["start"], cuts[2].args["end"], cuts[2].args["accurate"], cuts[2].chunk), ("1200.000000", "1800.000000", True, 3))
        self.assertNotIn("end", cuts[-1].args)
        self.assertEqual(exports[2].args["input"], cuts[2].outputs[0])
        self.assertEqual((joins[0].args["transition"], joins[0].args["copy"], joins[0].args["inputs"]), ("none", True, [e.outputs[0] for e in exports]))
        self.assertTrue(joins[0].outputs[0].endswith("_delivery_youtube"))
        self.assertEqual(len({o.idempotency_key for o in cuts + exports}), 24, "every chunk operation has its own key")
        self.assertEqual(cuts[0].media_seconds, 600.0)
        self.assertTrue(paths[joins[0].outputs[0]].endswith(".mp4") and "artifacts" in Path(paths[joins[0].outputs[0]]).parts)

    def test_a_short_programme_is_not_chunked_and_unchanged(self):
        svc, ir = self.plan(duration=600.0)
        self.assertNotIn("chunked_render", ir.doc)
        self.assertFalse(any("chunk" in w for w in ir.doc["analysis"]["warnings"]))
        with tempfile.TemporaryDirectory() as d:
            ops, _ = compile_ir(ir, d)
        self.assertEqual([o.skill for o in ops if o.skill.startswith("render_chunk")], [])
        self.assertNotIn("chunk_seconds", ir.doc["request"]["args"])

    def test_chunk_seconds_asked_for_on_a_short_programme(self):
        _, ir = self.plan(duration=600.0, chunk_seconds=120)
        ch = ir.doc["chunked_render"]
        self.assertEqual((len(ch["chunks"]), ch["chunk_seconds"]), (5, 120.0))
        self.assertIn("asked for", ch["reason"])

    def test_chunk_seconds_zero_never_chunks(self):
        _, ir = self.plan(chunk_seconds=0)
        self.assertNotIn("chunked_render", ir.doc)
        self.assertFalse(any("chunk" in w for w in ir.doc["analysis"]["warnings"]))

    def test_an_engine_without_join_copy_is_not_chunked_and_says_why(self):
        class NoJoin(FakeAdapter):
            def supports(self, tool):
                return tool.startswith("ffmpeg-skill/") and tool != "ffmpeg-skill/join"
        _, ir = self.plan(engine=NoJoin())
        self.assertNotIn("chunked_render", ir.doc)
        self.assertTrue(any("render not chunked: the engine cannot run ffmpeg-skill/join with --copy" in w for w in ir.doc["analysis"]["warnings"]), ir.doc["analysis"]["warnings"])

    def test_a_stream_copy_delivery_is_not_chunked(self):
        _, ir = self.plan(profile="generic")
        self.assertNotIn("chunked_render", ir.doc)
        self.assertTrue(any("render not chunked: no delivery target re-encodes" in w for w in ir.doc["analysis"]["warnings"]))

    def test_variable_frame_rate_is_not_chunked(self):
        _, ir = self.plan(engine=FakeAdapter(vfr=True))
        self.assertNotIn("chunked_render", ir.doc)
        self.assertTrue(any("variable" in w for w in ir.doc["analysis"]["warnings"] if w.startswith("render not chunked")))

    def test_bad_chunk_seconds_refused_before_analysis(self):
        with self.assertRaises(ValueError):
            self.plan(chunk_seconds=3)

    def test_the_plan_hash_includes_chunking(self):
        _, a = self.plan(chunk_seconds=600)
        _, b = self.plan(chunk_seconds=300)
        self.assertNotEqual(a.plan_hash(), b.plan_hash())

    def test_validates_against_the_schema(self):
        svc, ir = self.plan()
        p = str(Path(self.tmp) / "ws" / "p.project.json")
        save_ir(ir, p)
        rep = svc.validate(load_ir(p))
        self.assertTrue(rep.ok, rep.to_dict())


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.src = fake_media(self.tmp, "allday.mp4", duration=7200.0, silences=[])

    def planned(self, engine):
        svc = pipeline_service(self.tmp, engine=engine)
        ir = svc.plan([self.src], "youtube")
        p = str(Path(svc.workspace) / "plans" / "allday.project.json")
        save_ir(ir, p)
        return svc, p

    def test_render_writes_the_chunk_state_and_the_joined_deliverable(self):
        svc, p = self.planned(FakeAdapter(silences=[]))
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED", out["execution"])
        self.assertEqual((out["chunks"]["total"], out["chunks"]["done"], out["chunks"]["reused"], out["chunks"]["started_at"]), (12, 12, 0, 1))
        state = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "chunks.json").read_text(encoding="utf-8"))
        self.assertEqual([c["status"] for c in state["chunks"]], ["done"] * 12)
        art = next(k for k in out["paths"] if k.endswith("_delivery_youtube"))
        meta = json.loads(Path(out["paths"][art]).read_bytes())
        self.assertEqual((meta["joined"], meta["duration"]), (12, 7200.0))
        self.assertIn(out["qa"]["status"], ("PASS", "WARN"))

    def test_a_failure_in_chunk_3_then_resume_starts_at_chunk_3(self):
        class FailThird(FakeAdapter):
            """The third cut fails for good once (a stopped render); the resumed render's engine is healthy."""
            def __init__(self, fail=True, **kw):
                super().__init__(**kw)
                self.fail = fail

            def run(self, op, paths, timeout=None, dry_run=False, attempt=1):
                if self.fail and op.tool == "ffmpeg-skill/cut" and op.args.get("start") == "1200.000000":
                    from video_agent.models import ToolResult
                    self.calls.append(op)
                    return ToolResult(op.id, op.tool, False, 1, None, {}, [], "error: input file is missing: No such file", 0.1, attempt, dry_run)
                return super().run(op, paths, timeout=timeout, dry_run=dry_run, attempt=attempt)
        broken = FailThird(silences=[])
        svc, p = self.planned(broken)
        first = svc.render(load_ir(p), p, approve=["all"])
        self.assertNotEqual(first["status"], "COMPLETED")
        self.assertEqual([c["status"] for c in first["chunks"]["chunks"][:3]], ["done", "done", "pending"])
        job = json.loads((Path(svc.workspace) / "jobs" / first["job"]["id"] / "job.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(job["completed_ops"]), 4, "the checkpoint kept chunks 1-2 (and the loudness step)")
        healthy = FailThird(fail=False, silences=[])
        svc2 = pipeline_service(self.tmp, engine=healthy)
        out = svc2.render(load_ir(p), p, approve=["all"], resume="last")
        self.assertEqual(out["status"], "COMPLETED", out["execution"])
        self.assertEqual((out["chunks"]["reused"], out["chunks"]["started_at"]), (2, 3))
        ran_cuts = [o.args["start"] for o in healthy.calls if o.tool == "ffmpeg-skill/cut"]
        self.assertEqual(ran_cuts[0], "1200.000000", "the resumed render starts at chunk 3")
        self.assertNotIn("0.000000", ran_cuts)
        self.assertNotIn("600.000000", ran_cuts)

    def test_checkpoint_saves_the_job_after_every_operation(self):
        saved = []
        svc, p = self.planned(FakeAdapter(silences=[]))
        from video_agent.jobs import job as jobmod
        orig = jobmod.JobStore.save

        def spy(self_, j):
            saved.append(len(j.completed_ops))
            return orig(self_, j)
        jobmod.JobStore.save = spy
        try:
            out = svc.render(load_ir(p), p, approve=["all"])
        finally:
            jobmod.JobStore.save = orig
        self.assertEqual(out["status"], "COMPLETED")
        grows = sorted(set(saved))
        self.assertGreaterEqual(len(grows), 25, "one save per finished operation (loudness, 12 cuts, 12 exports, the join)")

    def test_speed_book_records_chunks_with_their_own_media_seconds(self):
        svc, p = self.planned(FakeAdapter(silences=[]))
        svc.render(load_ir(p), p, approve=["all"])
        book = dl.SpeedBook(svc.workspace)
        runs = book.runs(dl.machine_key(svc._hw_encoder()), "ffmpeg-skill/cut")
        self.assertTrue(runs)
        self.assertTrue(all(abs(float(r[1] if isinstance(r, (list, tuple)) else r.get("media_seconds")) - 600.0) < 1e-6 for r in runs), runs)


if __name__ == "__main__":
    unittest.main()
