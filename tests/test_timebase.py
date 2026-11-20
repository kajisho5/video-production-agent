"""The Velisa timebase spec's conformance vectors (velisa/timebase@1, ADR-120), run against the agent's own functions.

tests/fixtures/velisa/timebase-vectors.json is copied unchanged from the Velisa repository at the commit recorded in
tests/fixtures/velisa/provenance.json. Every vector whose `fn` this Skill offers must give the stated result or error
code; the timecode, seek, join and parse_seconds functions are not offered here (the agent writes no timecode, and its
concat timeline is covered by tests/test_concat_whole_frames.py), so their vectors are counted and skipped, never weakened.
"""
import hashlib
import json
import os
import unittest
from fractions import Fraction

from video_agent.agent import timebase
from video_agent.agent.editing import concat_fps, concat_rate, programme_rate, same_probed_rate, whole_frame_length
from video_agent.agent.timebase import RateError

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures", "velisa")
OFFERED = {
    "parse_rate": timebase.parse_rate, "rate_str": timebase.rate_str, "nominal": timebase.nominal,
    "same_rate": timebase.same_rate, "frames_at": timebase.frames_at, "seconds_of": timebase.seconds_of,
    "format_seconds": timebase.format_seconds, "transition_frames": timebase.transition_frames,
}
NOT_OFFERED = {"parse_seconds", "seek_seconds", "join_positions", "timecode_from_frames", "frames_from_timecode",
               "timecode_to_seconds", "seconds_to_timecode"}


def _out(value):
    if isinstance(value, Fraction):
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    return value


def _load():
    with open(os.path.join(FIXTURES, "timebase-vectors.json"), "rb") as fh:
        raw = fh.read()
    return raw, json.loads(raw.decode("utf-8"))


class VectorFileTests(unittest.TestCase):
    def test_the_copy_is_the_recorded_velisa_file(self):
        raw, doc = _load()
        with open(os.path.join(FIXTURES, "provenance.json"), encoding="utf-8") as fh:
            rec = json.load(fh)["files"]["timebase-vectors.json"]
        self.assertEqual(rec["repository"], "kajisho5/Velisa")
        self.assertRegex(rec["commit"], r"^[0-9a-f]{40}$")
        # a Windows checkout may turn the line ends into CRLF; the recorded hash is of the file as Velisa has it (LF)
        self.assertEqual(hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(), rec["sha256"], "the vectors were edited after they were copied")
        self.assertEqual(doc["spec"], rec["spec"])

    def test_every_function_is_either_offered_or_listed_as_not_offered(self):
        _raw, doc = _load()
        fns = {v["fn"] for v in doc["vectors"]}
        self.assertEqual(fns - set(OFFERED) - NOT_OFFERED, set(), "a new function in the spec: offer it or list it")


class ConformanceTests(unittest.TestCase):
    def test_vectors(self):
        _raw, doc = _load()
        ran = 0
        for vec in doc["vectors"]:
            if vec["fn"] not in OFFERED:
                continue
            with self.subTest(vec["id"]):
                fn = OFFERED[vec["fn"]]
                if vec.get("optional") and vec["fn"] == "parse_rate" and str(vec["args"]["value"]).strip().lower()[-1:] in ("p", "i"):
                    with self.assertRaises(RateError):   # the optional p / i suffix is not offered: refused, never misread
                        fn(**vec["args"])
                    continue
                ran += 1
                if "error" in vec:
                    with self.assertRaises(RateError) as cm:
                        fn(**vec["args"])
                    self.assertEqual(cm.exception.code, vec["error"])
                else:
                    self.assertEqual(_out(fn(**vec["args"])), vec["expect"])
        self.assertGreaterEqual(ran, 70)


class AgentReadsRatesBySpecTests(unittest.TestCase):
    """video-production-agent#167: concat_rate read 29.97 as 2997/100, params.fps was the probe's 3-decimal float, and rates
    were compared within 0.02 fps."""

    def test_concat_rate_is_ntsc_not_2997_over_100(self):
        self.assertEqual(concat_rate({}, {"video": {"fps": 29.97}}), Fraction(30000, 1001))
        self.assertEqual(concat_rate({}, {"video": {"fps": 59.94}}), Fraction(60000, 1001))
        self.assertEqual(concat_rate({"fps": "30000/1001"}, None), Fraction(30000, 1001), "an N/D params.fps was not read at all")
        self.assertEqual(concat_rate({"fps": "2997/100"}, None), Fraction(2997, 100), "an explicit N/D is taken as written")
        self.assertIsNone(concat_rate({}, {"video": {"fps": 0}}))

    def test_params_fps_is_written_exactly(self):
        self.assertEqual(concat_fps(29.97), "30000/1001")
        self.assertEqual(concat_fps(25.0), 25)
        techn = {"gen": {"video": {"width": 1280, "height": 720, "fps": 24.0}}, "cam": {"video": {"width": 1920, "height": 1080, "fps": 29.97}}}
        self.assertEqual(programme_rate(["gen", "cam"], techn, {"gen"}), Fraction(30000, 1001))

    def test_rates_compare_exactly(self):
        self.assertTrue(same_probed_rate(29.97, Fraction(30000, 1001)))
        self.assertFalse(same_probed_rate(29.98, Fraction(30000, 1001)), "0.01 off was the same rate under the 0.02 tolerance")
        self.assertFalse(same_probed_rate(29.97, 30))
        # a 29.98 fps input in a 29.97 edit is another rate: rounded to the nearest frame (conformed), not down
        rate = Fraction(30000, 1001)
        self.assertEqual(whole_frame_length(Fraction(2), rate, 29.98, False), Fraction(60) / rate)
        self.assertEqual(whole_frame_length(Fraction(2), rate, 29.97, False), Fraction(59) / rate)


if __name__ == "__main__":
    unittest.main()
