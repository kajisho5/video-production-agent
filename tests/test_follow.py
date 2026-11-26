"""Following a recording still being written (ADR-136, roadmap step 23): `video-agent follow` runs media-analysis-skill `follow` every
interval until the file is complete, says how far it has got, then plans it; `--deadline-after` counts from the moment the completion was
detected. Against a fake media-analysis-skill process (tests/fake_media_analysis.py) and the fake engine (tests/pipeline_harness.py); the
real Skill on a file ffmpeg is writing is tests/test_integration.py::FollowRecordingRealTests."""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline_harness import fake_media, pipeline_service  # noqa: E402

from video_agent import cli  # noqa: E402
from video_agent.agent import follow as fw  # noqa: E402
from video_agent.agent.deadline import DeadlineError  # noqa: E402
from video_agent.tools.media_analysis.locate import MediaAnalysisSkill  # noqa: E402

FAKE = MediaAnalysisSkill([sys.executable, str(HERE / "fake_media_analysis.py")], None, {})


class Growing:
    """A fake recording that grows by one minute at every sleep, and stops after `looks` sleeps."""

    def __init__(self, path: str, looks: int, moov_at_end: bool = False):
        self.path, self.left, self.moov = path, looks, moov_at_end
        self.minutes = 1
        self.write(True)

    def write(self, growing: bool) -> None:
        Path(self.path).write_text(json.dumps({"fake": True, "duration": 60.0 * self.minutes, "growing": growing, "moov_at_end": self.moov, "video": True}), encoding="utf-8")

    def sleep(self, _secs: float) -> None:
        if self.left > 0:
            self.left -= 1
            self.minutes += 1
            self.write(self.left > 0)


class FollowLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.svc = pipeline_service(self.tmp)
        self.rec = str(Path(self.tmp) / "src" / "room-a.mkv")
        Path(self.rec).parent.mkdir(parents=True, exist_ok=True)

    def follow(self, grow: Growing, **kw):
        lines = []
        with mock.patch("video_agent.service.locate_media_analysis", return_value=FAKE):
            rec = self.svc.follow_recording(self.rec, interval=0.2, settle=0, emit=lines.append, sleep=grow.sleep, **kw)
        return rec, lines

    def test_follows_until_settled_and_reports_progress(self):
        grow = Growing(self.rec, looks=3)
        rec, lines = self.follow(grow)
        self.assertTrue(rec["complete"])
        self.assertEqual((rec["rule"], rec["reason"], rec["duration"], rec["looks"]), ("follow_recording@1", "settled", 240.0, 4))
        self.assertEqual(sum(1 for ln in lines if "measured to" in ln), 4, lines)
        self.assertIn("complete (settled)", lines[-1])
        saved = json.loads((fw.state_dir(self.svc.workspace, self.rec) / "follow.json").read_text(encoding="utf-8"))
        self.assertEqual(len(saved["ticks"]), 4)
        self.assertTrue(saved["ticks"][1]["grew_bytes"] is not None)

    def test_complete_marker(self):
        grow = Growing(self.rec, looks=50)
        marker = str(Path(self.tmp) / "src" / "room-a.done")
        orig = grow.sleep

        def sleep(s):
            orig(s)
            if grow.minutes == 3:
                Path(marker).write_text("", encoding="utf-8")
        grow.sleep = sleep
        rec, _ = self.follow(grow, marker=marker)
        self.assertEqual((rec["reason"], rec["looks"]), ("marker", 3))

    def test_once_reports_a_growing_file(self):
        grow = Growing(self.rec, looks=5)
        rec, lines = self.follow(grow, once=True)
        self.assertFalse(rec["complete"])
        self.assertEqual(rec["looks"], 1)
        self.assertTrue(any("measured to" in ln for ln in lines))

    def test_plain_mp4_says_why_once_and_waits_for_it_to_close(self):
        self.rec = str(Path(self.tmp) / "src" / "room-a.mp4")
        grow = Growing(self.rec, looks=3, moov_at_end=True)
        rec, lines = self.follow(grow)
        self.assertTrue(rec["complete"] and rec["unreadable_while_growing"])
        said = [ln for ln in lines if "not readable" in ln]
        self.assertEqual(len(said), 1, lines)
        self.assertIn("moov", said[0])
        self.assertIn("MKV, MPEG-TS or fragmented MP4", said[0])

    def test_max_wait(self):
        grow = Growing(self.rec, looks=1000)
        t = [0.0]

        def clock():
            t[0] += 60.0
            return t[0]
        with mock.patch("video_agent.service.locate_media_analysis", return_value=FAKE), self.assertRaises(fw.FollowError) as cm:
            self.svc.follow_recording(self.rec, interval=0.2, settle=0, emit=lambda _l: None, sleep=grow.sleep, max_wait=300, clock=clock)
        self.assertEqual(cm.exception.code, "TIMEOUT")
        self.assertIn("follow state is kept", str(cm.exception))

    def test_without_media_analysis(self):
        Growing(self.rec, looks=0)
        with mock.patch("video_agent.service.locate_media_analysis", return_value=None), self.assertRaises(fw.FollowError) as cm:
            self.svc.follow_recording(self.rec, emit=lambda _l: None)
        self.assertEqual(cm.exception.code, "SKILL_MISSING")
        self.assertIn("VIDEO_AGENT_MEDIA_ANALYSIS_DIR", str(cm.exception))
        self.assertIn("video-agent plan", str(cm.exception))

    def test_a_skill_without_follow(self):
        Growing(self.rec, looks=0)
        with mock.patch.dict(os.environ, {"FAKE_MA_MODE": "no_follow"}), mock.patch("video_agent.service.locate_media_analysis", return_value=FAKE), \
                self.assertRaises(fw.FollowError) as cm:
            self.svc.follow_recording(self.rec, emit=lambda _l: None)
        self.assertEqual(cm.exception.code, "SKILL_TOO_OLD")

    def test_refusals(self):
        with self.assertRaises(fw.FollowError):
            fw.follow(lambda a, t: (0, "", ""), str(Path(self.tmp) / "none.mkv"), self.svc.workspace)
        Growing(self.rec, looks=0)
        for kw in ({"interval": 0.0}, {"settle": -1}, {"kind": "loudness"}):
            with self.subTest(kw=kw), self.assertRaises(fw.FollowError):
                fw.follow(lambda a, t: (0, "", ""), self.rec, self.svc.workspace, **kw)

    def test_argv(self):
        argv = fw.argv_for("/r/a.mkv", "silence", "/w/s.json", "/w", 30, "/r/a.done", False)
        self.assertEqual(argv[:6], ["follow", "/r/a.mkv", "--kind", "silence", "--state", "/w/s.json"])
        self.assertIn("--json", argv)
        self.assertEqual(argv[argv.index("--complete-marker") + 1], "/r/a.done")
        self.assertEqual(argv[argv.index("--allowed-input") + 1], str(Path("/r").resolve()))


class DeadlineAfterTests(unittest.TestCase):
    def test_parse_after(self):
        self.assertEqual(fw.parse_after("30m"), 1800.0)
        self.assertIsNone(fw.parse_after(None))
        with self.assertRaises(DeadlineError):
            fw.parse_after("12:30")

    def test_counted_from_the_detection(self):
        self.assertEqual(fw.deadline_text(1800, 1000.0, now=1300.0), "1500s")
        self.assertEqual(fw.deadline_text(60, 1000.0, now=5000.0), "1s", "already past: the plan is at risk")

    def test_plan_with_a_deadline_after_the_end(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        svc = pipeline_service(tmp)
        src = fake_media(tmp, "keynote.mp4", duration=600.0)
        ir = svc.plan([src], "youtube")
        rec = {"detected_epoch": 1000.0, "detected_at": "2026-11-27T10:00:00+0900", "reason": "settled"}
        dl = svc.apply_deadline_after(ir, "30m", rec, now=1600.0)
        self.assertEqual(dl["budget_seconds"], 1200.0, "30 min after the end, 10 min of it already spent")
        self.assertEqual(dl["deadline"], "30m after the recording ended")
        self.assertEqual(dl["relative_to"], {"event": "recording_complete", "detected_at": rec["detected_at"], "after_seconds": 1800.0, "reason": "settled"})
        self.assertIs(ir.doc["deadline"], dl)
        from video_agent.project import load_ir, save_ir
        p = str(Path(tmp) / "ws" / "p.json")
        save_ir(ir, p)
        self.assertTrue(svc.validate(load_ir(p)).ok)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.svc = pipeline_service(self.tmp)
        self.rec = str(Path(self.tmp) / "src" / "room-a.mkv")
        Path(self.rec).parent.mkdir(parents=True, exist_ok=True)

    def run_cli(self, argv, grow):
        out, err = io.StringIO(), io.StringIO()
        real = self.svc.follow_recording

        def follow(recording, **kw):
            kw["sleep"] = grow.sleep
            return real(recording, **kw)
        with mock.patch("video_agent.service.locate_media_analysis", return_value=FAKE), mock.patch.object(self.svc, "follow_recording", follow), \
                redirect_stdout(out), redirect_stderr(err):
            args = cli.build_parser().parse_args(argv)
            code = args.fn(args, self.svc)
        return code, out.getvalue(), err.getvalue()

    def test_follow_then_plan_with_a_deadline_after_the_end(self):
        grow = Growing(self.rec, looks=2)
        code, out, err = self.run_cli(["--json", "follow", self.rec, "--profile", "youtube", "--interval", "0.2", "--settle", "0", "--deadline-after", "30m"], grow)
        self.assertEqual(code, 0, err)
        doc = json.loads(out)
        self.assertEqual(doc["status"], "PLANNED")
        self.assertEqual(doc["follow"]["reason"], "settled")
        self.assertEqual(doc["deadline"]["relative_to"]["after_seconds"], 1800.0)
        self.assertLessEqual(doc["deadline"]["budget_seconds"], 1800.0)
        self.assertTrue(Path(doc["project"]).exists())
        saved = json.loads(Path(doc["project"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["provenance"]["follow"]["recording"], str(Path(self.rec).resolve()))
        self.assertIn("measured to", err, "progress goes to stderr under --json")

    def test_text_progress_and_render(self):
        grow = Growing(self.rec, looks=1)
        code, out, _ = self.run_cli(["follow", self.rec, "--profile", "youtube", "--interval", "0.2", "--settle", "0", "--render", "--approve", "all"], grow)
        self.assertIn("Following", out)
        self.assertIn("complete (settled)", out)
        self.assertIn("Project IR:", out)
        self.assertIn("Job ", out)
        self.assertIn(code, (0, 4))

    def test_bad_deadline_after_refused_before_waiting(self):
        grow = Growing(self.rec, looks=1)
        code, _, err = self.run_cli(["follow", self.rec, "--deadline-after", "10:30"], grow)
        self.assertEqual(code, 2)
        self.assertIn("clock time", err)

    def test_missing_skill_exit_code(self):
        Growing(self.rec, looks=0)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("video_agent.service.locate_media_analysis", return_value=None), redirect_stdout(out), redirect_stderr(err):
            args = cli.build_parser().parse_args(["follow", self.rec])
            code = args.fn(args, self.svc)
        self.assertEqual(code, 3)
        self.assertIn("SKILL_MISSING", err.getvalue())


if __name__ == "__main__":
    unittest.main()
