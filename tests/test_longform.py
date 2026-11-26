"""Long inputs (ADR-132; roadmap step 23, media-analysis-skill step 11): a plan that measures the transcript of a recording of 30 min or more
measures media-analysis-skill's speech regions first and hands them to transcription-skill's chunked recognition. Against the fake engine, the fake
media-analysis process and the fake transcription process (which logs each request's params)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter  # noqa: E402
from pipeline_harness import FAKES, fake_media  # noqa: E402
from test_unit import FakeCaps  # noqa: E402

from video_agent.media import longform  # noqa: E402
from video_agent.service import Service  # noqa: E402
from video_agent.tools import ToolRouter  # noqa: E402
from video_agent.tools.base import ToolError  # noqa: E402
from video_agent.tools.media_analysis import MediaAnalysisAdapter, MediaAnalysisSkill  # noqa: E402
from video_agent.tools.transcription import TranscriptionAdapter, TranscriptionSkill  # noqa: E402

FAKE_MA = str(Path(__file__).resolve().parent / "fake_media_analysis.py")


class RuleTests(unittest.TestCase):
    def test_short_inputs_are_left_alone(self):
        self.assertIsNone(longform.plan_source(1799.0, "en", lambda k: True, True))
        self.assertIsNone(longform.plan_source(0.0, "en", lambda k: True, True))

    def test_long_input_is_chunked_with_regions(self):
        p = longform.plan_source(7200.0, "EN", lambda k: True, True)
        self.assertEqual((p["chunked"], p["chunk_seconds"], p["cuts_from"]), (True, 600, "speech_regions"))
        params = longform.transcribe_params(p, [{"start": 1.0, "end": 2.0}])
        self.assertEqual(params, {"chunk_seconds": 600, "speech_regions": {"kind": "speech_regions", "data": {"regions": [{"start": 1.0, "end": 2.0}]}}})

    def test_why_not_chunked(self):
        self.assertIn("language", longform.plan_source(7200.0, None, lambda k: True, True)["reason"])
        self.assertIn("language", longform.plan_source(7200.0, "auto", lambda k: True, True)["reason"])
        self.assertIn("0.11.2", longform.plan_source(7200.0, "en", lambda k: False, True)["reason"])
        self.assertEqual(longform.plan_source(7200.0, "en", lambda k: k != "speech_regions", True)["cuts_from"], "level")
        no_tool = longform.plan_source(7200.0, "en", lambda k: True, False)
        self.assertEqual((no_tool["chunked"], no_tool["cuts_from"]), (True, "level"))

    def test_unmeasured_regions_fall_back_to_the_level_cut(self):
        self.assertIsNone(longform.regions_of({"status": "UNKNOWN", "reason": "extra vad missing"}))
        p = longform.plan_source(7200.0, "en", lambda k: True, True)
        self.assertEqual(longform.transcribe_params(p, None), {"chunk_seconds": 600})
        self.assertEqual(p["cuts_from"], "level")
        self.assertIn("vad", p["cuts_reason"])

    def test_adapter_checks_the_chunking_inputs(self):
        check = TranscriptionAdapter.check_longform
        declared = {"chunk_seconds": "", "parallel": "", "speech_regions": ""}
        self.assertEqual(check({"chunk_seconds": 600}, declared, "en"), {"chunk_seconds": 600})
        for given, lang, why in (({"chunk_seconds": 10}, "en", "[30, 3600]"), ({"chunk_seconds": 600}, None, "needs a language"),
                                 ({"parallel": 2}, "en", "need chunk_seconds"), ({"chunk_seconds": 600, "parallel": 9}, "en", "[1, 8]"),
                                 ({"chunk_seconds": 600, "speech_regions": {"data": {"regions": [{"start": 2, "end": 1}]}}}, "en", "data.regions"),
                                 ({"chunk_seconds": 600, "speech_regions": "/tmp/regions.json"}, "en", "data.regions")):
            with self.subTest(given=given), self.assertRaises(ToolError) as cm:
                check(given, declared, lang)
            self.assertIn(why, str(cm.exception))
        with self.assertRaises(ToolError):
            check({"chunk_seconds": 600}, {}, "en")


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.log = str(Path(self.tmp) / "ts.log")
        os.environ["FAKE_TS_REQUESTS"] = self.log
        self.addCleanup(os.environ.pop, "FAKE_TS_REQUESTS", None)

    def service(self, media_analysis=True):
        src = str(Path(self.tmp) / "src")
        ws = str(Path(self.tmp) / "ws")
        os.makedirs(ws, exist_ok=True)
        adapters = [FakeAdapter(), TranscriptionAdapter(TranscriptionSkill([sys.executable, str(FAKES["transcription"])], None, {}), workspace=str(Path(ws) / "cache" / "tr"))]
        if media_analysis:
            adapters.append(MediaAnalysisAdapter(MediaAnalysisSkill([sys.executable, FAKE_MA], None, {}), workspace=self.tmp, allowed_inputs=[src]))
        return Service(workspace=ws, adapter=ToolRouter(adapters), caps=FakeCaps(extra=["transcription"] + (["media-analysis"] if media_analysis else [])))

    def recognitions(self):
        rows = [json.loads(x) for x in Path(self.log).read_text(encoding="utf-8").splitlines()] if os.path.exists(self.log) else []
        return [r for r in rows if not r.get("dry_run") and "input" in r]

    def plan(self, svc, duration, language="en"):
        src = fake_media(self.tmp, "keynote.mp4", duration=duration, segments=[[0.5, 3.0, "Welcome to the keynote."]])
        return svc.plan([src], "generic", kinds=["transcript"], params={"language": language} if language else {}, use_cache=False)

    def rows(self, ir, kind):
        return [r for a in ir.doc["analysis"]["analyses"] for r in a["rows"] if r["kind"] == kind]

    def test_a_two_hour_talk_is_chunked_at_the_measured_speech_regions(self):
        ir = self.plan(self.service(), 7200.0)
        [regions] = [o for o in ir.doc["analysis"]["observations"] if o["kind"] == "speech_regions"]
        [tr] = [o for o in ir.doc["analysis"]["observations"] if o["kind"] == "transcript"]
        [req] = self.recognitions()
        self.assertEqual(req["chunk_seconds"], 600)
        self.assertEqual(req["speech_regions"]["data"]["regions"], [{"start": 3.0, "end": 9.5}, {"start": 10.2, "end": 15.0}])
        self.assertEqual(tr["parameters"]["speech_regions"], regions["id"], "the transcript names the regions its cuts come from, not a copy")
        [row] = self.rows(ir, "transcript")
        self.assertEqual((row["longform"]["chunked"], row["longform"]["cuts_from"], row["longform"]["duration"]), (True, "speech_regions", 7200.0))
        order = [r["kind"] for r in ir.doc["analysis"]["analyses"][0]["rows"]]
        self.assertLess(order.index("speech_regions"), order.index("transcript"))

    def test_without_media_analysis_the_skill_cuts_by_level(self):
        ir = self.plan(self.service(media_analysis=False), 7200.0)
        [req] = self.recognitions()
        self.assertEqual(req["chunk_seconds"], 600)
        self.assertNotIn("speech_regions", req)
        [row] = self.rows(ir, "transcript")
        self.assertEqual(row["longform"]["cuts_from"], "level")

    def test_without_a_language_one_pass_and_the_plan_says_why(self):
        ir = self.plan(self.service(), 7200.0, language=None)
        [req] = self.recognitions()
        self.assertNotIn("chunk_seconds", req)
        self.assertFalse([o for o in ir.doc["analysis"]["observations"] if o["kind"] == "speech_regions"])
        self.assertTrue(any("plan --language" in w for w in ir.doc["analysis"]["warnings"]), ir.doc["analysis"]["warnings"])

    def test_a_short_input_is_unchanged(self):
        ir = self.plan(self.service(), 600.0)
        [req] = self.recognitions()
        self.assertNotIn("chunk_seconds", req)
        [row] = self.rows(ir, "transcript")
        self.assertNotIn("longform", row)


if __name__ == "__main__":
    unittest.main()
