"""ADR-116 (roadmap step 34): tools/loudness_compare.py, the loudnorm / ebur128 comparison of the two loudness paths.

ParserTests run anywhere (no ffmpeg). RealComparisonTests run the script on short fixtures against real ffmpeg-skill and
audio-production-skill checkouts (VIDEO_AGENT_FFMPEG_SKILL_DIR / VIDEO_AGENT_AUDIO_PRODUCTION_DIR) and skip without them.
"""
from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import loudness_compare as lc  # noqa: E402
from tmp_dirs import tmp_dir  # noqa: E402

EBUR128_TAIL = """[Parsed_ebur128_0 @ 0x55] Summary:

  Integrated loudness:
    I:         -21.8 LUFS
    Threshold: -31.8 LUFS

  Loudness range:
    LRA:        20.0 LU
    Threshold: -41.8 LUFS
    LRA low:   -41.8 LUFS
    LRA high:  -21.8 LUFS

  True peak:
    Peak:      -18.1 dBFS
"""

LOUDNORM_TAIL = """[Parsed_loudnorm_0 @ 0x56]
{
\t"input_i" : "-20.61",
\t"input_tp" : "-9.02",
\t"input_lra" : "6.80",
\t"input_thresh" : "-31.10",
\t"output_i" : "-14.00",
\t"target_offset" : "0.01"
}
"""


class ParserTests(unittest.TestCase):
    def test_ebur128_summary(self):
        self.assertEqual(lc.parse_ebur128("frame lines...\n" + EBUR128_TAIL), {"i": -21.8, "tp": -18.1, "lra": 20.0})

    def test_ebur128_reads_the_last_summary_and_inf(self):
        first = EBUR128_TAIL.replace("-21.8 LUFS\n    Threshold: -31.8", "-5.0 LUFS\n    Threshold: -31.8")
        silent = EBUR128_TAIL.replace("I:         -21.8", "I:         -inf").replace("Peak:      -18.1", "Peak:      -inf")
        self.assertEqual(lc.parse_ebur128(first + silent), {"i": None, "tp": None, "lra": 20.0})
        with self.assertRaises(ValueError):
            lc.parse_ebur128("no summary here")

    def test_loudnorm_json(self):
        self.assertEqual(lc.parse_loudnorm("Input #0 ...\n" + LOUDNORM_TAIL), {"i": -20.61, "tp": -9.02, "lra": 6.8})
        self.assertEqual(lc.parse_loudnorm(LOUDNORM_TAIL.replace('"-20.61"', '"-inf"'))["i"], None)
        with self.assertRaises(ValueError):
            lc.parse_loudnorm("nothing")

    def test_the_four_kinds_of_material(self):
        self.assertEqual(set(lc.FIXTURES), {"speech", "music", "quiet", "dynamic"})
        self.assertEqual(lc.TARGETS, ((-14.0, -1.0), (-23.0, -1.0)))

    def test_table_names_a_refusal(self):
        res = {"ffmpeg": "ffmpeg x", "ffmpeg_skill": "3", "audio_production": "1", "tolerance_lu": 1.0,
               "inputs": {"speech": {"measured": {"loudnorm": {"i": -20.0, "tp": -9.0, "lra": 6.0}, "ebur128": {"i": -19.5, "tp": -9.0, "lra": 6.5}}}},
               "rows": [{"fixture": "speech", "target_lufs": -14.0, "true_peak_db": -1.0, "same_samples": False,
                         "paths": {"ffmpeg-skill": {"ok": False, "seconds": 1.0, "error": "VALIDATION_ERROR: out of reach"},
                                   "audio-production": {"ok": False, "seconds": 2.0, "error": "VALIDATION_ERROR: out of reach"}}}]}
        text = lc.table(res)
        self.assertIn("| speech | -20.0 / -19.5 | -9.0 / -9.0 | 6.0 / 6.5 |", text)
        self.assertIn("refused: VALIDATION_ERROR: out of reach", text)


def _dir(env, sibling):
    for cand in (os.environ.get(env), str(lc.REPO.parent / sibling)):
        if cand and (Path(cand) / ("scripts/loudness.py" if sibling == "ffmpeg-skill" else "src/audio_production")).exists():
            return Path(cand)
    return None


FFS = _dir("VIDEO_AGENT_FFMPEG_SKILL_DIR", "ffmpeg-skill")
APS = _dir("VIDEO_AGENT_AUDIO_PRODUCTION_DIR", "audio-production-skill")


@unittest.skipUnless(shutil.which("ffmpeg") and FFS and APS, "needs ffmpeg and ffmpeg-skill / audio-production-skill checkouts")
class RealComparisonTests(unittest.TestCase):
    """The measured claims ADR-116 rests on, on 10 s fixtures at -14 LUFS / -1 dBTP."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tmp_dir(cls, "loudness-compare-test-")   # ADR-115: removed whatever happens
        cls.result = lc.compare(FFS, APS, Path(cls.tmp), seconds=10.0, targets=((-14.0, -1.0),), fixtures=["speech", "music", "dynamic"])
        cls.rows = {r["fixture"]: r for r in cls.result["rows"]}

    def test_both_paths_write_the_same_samples(self):
        for name, r in self.rows.items():
            with self.subTest(name):
                self.assertTrue(all(e["ok"] for e in r["paths"].values()), r["paths"])
                self.assertTrue(r["same_samples"], "LOUDNESS_SET runs ffmpeg-skill loudness with the agent's argv")

    def test_on_target_by_both_meters(self):
        for name, r in self.rows.items():
            for path, e in r["paths"].items():
                with self.subTest(f"{name} {path}"):
                    self.assertTrue(all(e["on_target"].values()), e)

    def test_meters_agree_on_steady_music_and_true_peak(self):
        m = self.rows["music"]["paths"]["ffmpeg-skill"]["meters_differ"]
        self.assertLessEqual(abs(m["i"]), 0.1)
        for name, r in self.rows.items():
            with self.subTest(name):
                self.assertLessEqual(abs(r["paths"]["ffmpeg-skill"]["meters_differ"]["tp"]), 0.1)

    def test_dynamic_material_falls_back_to_dynamic_mode(self):
        self.assertGreater(self.result["inputs"]["dynamic"]["measured"]["loudnorm"]["lra"], 11.0)
        mode = self.rows["dynamic"]["paths"]["ffmpeg-skill"]["reported"]["loudnorm_mode"]
        if mode is None:
            self.skipTest("this ffmpeg-skill does not report loudnorm_mode (2.x)")
        self.assertEqual(mode, "dynamic")


if __name__ == "__main__":
    unittest.main()
