"""ADR-048: ffmpeg-skill 2.x nests loudness figures under `measured`, and a loudness observation without a figure
is never reported as "within tolerance"."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter  # noqa: E402
from test_unit import FakeCaps, fake_media  # noqa: E402

from video_agent.media.analyzer import MediaAnalyzer  # noqa: E402
from video_agent.service import Service  # noqa: E402


class ShapedLoudnessAdapter(FakeAdapter):
    """FakeAdapter whose loudness --measure-only answers in a given shape."""

    def __init__(self, measured, **kw):
        super().__init__(**kw)
        self.measured = measured

    def run(self, op, paths, timeout=None, dry_run=False, attempt=1):
        r = super().run(op, paths, timeout=timeout, dry_run=dry_run, attempt=attempt)
        if op.tool.endswith("/loudness") and op.args.get("measure_only"):
            r.data = self.measured
        return r


def loudness_decision(adapter):
    tmp = tempfile.mkdtemp()
    ir = Service(workspace=tmp, adapter=adapter, caps=FakeCaps()).plan([fake_media(tmp)], "youtube")
    obs = next(o for o in ir.doc["analysis"]["observations"] if o["kind"] == "loudness")
    dec = next(d for d in ir.doc["decisions"] if d["subject"] == "audio.loudness")
    return obs, dec


class LoudnessMeasuredTests(unittest.TestCase):
    def test_shape_reads_both_layouts(self):
        top = {"input_i": "-21.8", "input_tp": "-17.7", "input_lra": "0.1"}
        for data in (top, {"status": "completed", "measured": top}):
            with self.subTest(nested="measured" in data):
                d = MediaAnalyzer._shape("loudness", data, {})
                self.assertEqual((d["lufs"], d["true_peak"], d["lra"]), (-21.8, -17.7, 0.1))

    def test_shared_reader_unwraps_measured(self):
        from video_agent.media.analysis import loudness_facts
        lf = loudness_facts({"status": "completed", "measured": {"input_i": "-16.1", "input_tp": "-2.0", "input_lra": "4.0"}})
        self.assertEqual((lf["lufs"], lf["true_peak"], lf["lra"], lf["silent"]), (-16.1, -2.0, 4.0, False))
        self.assertIsNone(loudness_facts({"status": "completed", "measured": {}})["lufs"])

    def test_nested_figure_drives_normalisation(self):
        obs, dec = loudness_decision(ShapedLoudnessAdapter({"status": "completed", "measured": {"input_i": "-22.0", "input_tp": "-6.0", "input_lra": "5.0"}}))
        self.assertEqual(obs["data"]["lufs"], -22.0)
        self.assertEqual(dec["type"], "TRANSFORM", "-22 LUFS is off a -14 target: normalise")

    def test_unreadable_figure_is_not_within_tolerance(self):
        obs, dec = loudness_decision(ShapedLoudnessAdapter({"status": "completed", "measured": {}}))
        self.assertIsNone(obs["data"]["lufs"])
        self.assertNotIn("within tolerance", dec["reason"])
        self.assertEqual(dec["approval"], "CONFIRM", "a human sees that the loudness was never measured")


if __name__ == "__main__":
    unittest.main(verbosity=2)
