"""ADR-115: the planned concat timeline is the one video-editing-skill delivers (pinned contract 0.8.0, contract.whole_frames).

The expected numbers are worked out by hand from the contract's rules (and, for the first case, measured on the real Skill:
tests/test_integration.py ConcatWholeFramesRealTests), not from the code under test."""
from __future__ import annotations

import json
import random
import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from video_agent.agent.editing import concat_rate, concat_segments, input_rate, joined_transition, programme_rate, whole_frame_length  # noqa: E402
from video_agent.agent.takes import align_to_frames  # noqa: E402

PIN = Path(__file__).resolve().parents[1] / "src" / "video_agent" / "tools" / "video_editing" / "contract_0.8.0.json"


def tech(fps, vfr=False):
    return {"video": {"fps": fps, "variable_frame_rate_suspected": vfr}}


def trim(aid, *keep):
    return {"type": "video.trim", "asset": aid, "keep": [list(k) for k in keep]}


class ConcatWholeFramesTests(unittest.TestCase):
    def plan(self, inputs, ops=(), durations=None, transition=None, params=None, techn=None):
        techn = techn or {a: tech(30.0) for a in inputs}
        return concat_segments(inputs, list(ops), durations or {}, transition, rate=concat_rate(params or {}, techn.get(inputs[0])),
                               input_rates={a: input_rate(techn.get(a)) for a in inputs})

    def starts(self, segs):
        firsts = {}
        for s in segs:
            firsts.setdefault(s["input"], s["timeline_range"][0])
        return list(firsts.values())

    def test_the_rules_are_the_pinned_contracts(self):
        """The rules implemented here are the ones the pinned contract states (a re-pin that changes them fails here)."""
        wf = json.loads(PIN.read_text(encoding="utf-8"))["whole_frames"]
        rules = " ".join(wf["rule"])
        self.assertIn("params.fps, else the first input's frame rate", wf["rate"])
        self.assertIn("its planned length rounded down to whole frames (one within 1 ms of the next frame keeps it)", rules)
        self.assertIn("an input at another rate, or variable-rate: its planned length rounded to the nearest frame of the rate", rules)
        self.assertIn("its duration rounded to the nearest frame of the rate (a half up, at least one frame)", rules)
        self.assertIn("the timeline places every clip at that length", rules)

    def test_xfade_overlaps_by_whole_frames_reproduction_of_the_finding(self):
        """Three 4 s inputs at 30 fps with a 0.25 s fade: 0.25 s is 7.5 frames, joined as 8 (0.2667 s). Before ADR-115 the plan put
        the clips at 3.75 / 7.5 and the programme at 11.5 s; video-editing-skill 0.8.0 delivered 3.7333 / 7.4667 and 11.4667 s
        (344 frames), so every later clip, cue and transcript offset was a frame early per join."""
        ids = ["a", "b", "c"]
        segs, total = self.plan(ids, durations={a: 4.0 for a in ids}, transition={"type": "fade", "duration": 0.25})
        self.assertEqual(self.starts(segs), [0.0, 3.733, 7.467])
        self.assertEqual(total, 11.467)
        self.assertEqual(round(total * 30), 344)
        self.assertEqual([s["source_range"] for s in segs], [[0.0, 4.0]] * 3)

    def test_without_a_known_rate_the_requested_lengths_are_used(self):
        """No rate (no params.fps, no fps on the first input): the Skill joins as it is, and so does the plan."""
        segs, total = concat_segments(["a", "b"], [], {"a": 4.0, "b": 4.0}, {"type": "fade", "duration": 0.25})
        self.assertEqual((self.starts(segs), total), ([0.0, 3.75], 7.75))

    def test_an_input_at_the_rate_is_rounded_down(self):
        """2.01 s at 30 fps is 60.3 frames: joined as 60 (2.0 s), losing the part frame at the end of its kept range."""
        segs, total = self.plan(["a", "b"], [trim("a", (0.5, 2.51)), trim("b", (0.5, 2.51))])
        self.assertEqual(segs[0], {"input": "a", "track": "V1", "source_range": [0.5, 2.5], "timeline_range": [0.0, 2.0]})
        self.assertEqual(segs[1]["timeline_range"], [2.0, 4.0])
        self.assertEqual(total, 4.0)

    def test_within_one_ms_of_the_next_frame_keeps_it(self):
        """120 frames at 29.97 fps (the rate as the Skill reads 29.97: 2997/100) last 4.004004 s; a probe of 4.004 s keeps them."""
        techn = {a: tech(29.97) for a in "ab"}
        segs, total = self.plan(["a", "b"], durations={"a": 4.004, "b": 4.004}, techn=techn)
        self.assertEqual(concat_rate({}, techn["a"]), Fraction(2997, 100))
        self.assertEqual(self.starts(segs), [0.0, 4.004])
        self.assertEqual(total, 8.008)

    def test_an_input_at_another_rate_is_rounded_to_the_nearest_frame_and_held(self):
        """A 25 fps input of 2.02 s in a 30 fps edit: 60.6 frames, conformed to 61 (2.0333 s); it holds its last instant, so its
        segment is longer on the timeline than in the source."""
        techn = {"a": tech(30.0), "b": tech(25.0), "c": tech(30.0)}
        segs, total = self.plan(["a", "b", "c"], durations={"a": 2.0, "b": 2.02, "c": 2.0}, techn=techn)
        self.assertEqual(segs[1]["source_range"], [0.0, 2.02])
        self.assertEqual(segs[1]["timeline_range"], [2.0, 4.033])
        self.assertEqual(segs[2]["timeline_range"], [4.033, 6.033])
        self.assertEqual(total, 6.033)

    def test_a_variable_rate_input_is_conformed_too(self):
        """A variable-rate input at the edit's nominal rate is conformed (nearest frame), not cut (rounded down)."""
        techn = {"a": tech(30.0), "b": tech(30.0, vfr=True)}
        _, total = self.plan(["a", "b"], durations={"a": 2.0, "b": 2.02}, techn=techn)   # 60.6 frames → 61
        self.assertEqual(total, round(2.0 + 61 / 30, 3))
        techn["b"] = tech(30.0)
        _, total = self.plan(["a", "b"], durations={"a": 2.0, "b": 2.02}, techn=techn)   # at the rate → 60
        self.assertEqual(total, 4.0)

    def test_params_fps_names_the_rate(self):
        """params.fps wins over the first input's rate: two 30 fps inputs of 2 s joined at 25 fps (conformed, 50 frames each)."""
        techn = {a: tech(30.0) for a in "ab"}
        self.assertEqual(concat_rate({"fps": 25}, techn["a"]), Fraction(25))
        segs, total = self.plan(["a", "b"], durations={"a": 2.01, "b": 2.01}, params={"fps": 25}, techn=techn)   # 50.25 → 50 frames
        self.assertEqual((self.starts(segs), total), ([0.0, 2.0], 4.0))

    def test_the_part_frame_comes_off_the_last_kept_range(self):
        """Kept ranges 1.0 + 1.0 + 0.02 s at 30 fps: 2.02 s is 60.6 frames, joined as 60; the 0.02 s range lies wholly in the lost
        part frame and is not joined; the next input starts at 2.0 s."""
        segs, total = self.plan(["a", "b"], [trim("a", (0, 1), (2, 3), (5, 5.02))], durations={"b": 1.0})
        a = [s for s in segs if s["input"] == "a"]
        self.assertEqual([s["timeline_range"] for s in a], [[0.0, 1.0], [1.0, 2.0]])
        self.assertEqual(self.starts(segs), [0.0, 2.0])
        self.assertEqual(total, 3.0)
        segs, _ = self.plan(["a", "b"], [trim("a", (0, 1), (2, 3.02))], durations={"b": 1.0})
        self.assertEqual(segs[1]["source_range"], [2.0, 3.0], "a part frame cut from the middle of the last range")

    def test_transition_rounding_half_up_and_at_least_a_frame(self):
        self.assertEqual(joined_transition(0.25, Fraction(30)), Fraction(8, 30))       # 7.5 frames → 8
        self.assertEqual(joined_transition(0.01, Fraction(30)), Fraction(1, 30))       # 0.3 frames → at least 1
        self.assertEqual(joined_transition(0.5, Fraction(2997, 100)), Fraction(15) / Fraction(2997, 100))   # 14.985 → 15
        self.assertEqual(joined_transition(0.5, Fraction(25)), Fraction(13, 25))       # 12.5 frames → 13 (a half rounds up)
        self.assertEqual(joined_transition(0.48, Fraction(25)), Fraction(12, 25))      # whole already

    def test_a_stinger_butt_joins_at_whole_frames(self):
        segs, total = self.plan(["a", "b"], durations={"a": 2.01, "b": 2.01},
                                transition={"type": "stinger", "cut_point": 0.5, "cover": {"start": 0.3, "end": 0.7}})
        self.assertEqual((self.starts(segs), total), ([0.0, 2.0], 4.0))

    def test_every_clip_starts_on_a_frame_and_nothing_accumulates(self):
        """Fifty inputs of random lengths, rates and a 29.97 fps edit with a 0.5 s fade: every input starts within the IR's
        millisecond of a frame, and the programme's length is the exact whole-frame sum (no drift of rounding to ms per clip)."""
        rnd = random.Random(115)
        rate = Fraction(2997, 100)
        ids = [f"i{n}" for n in range(50)]
        durations = {a: round(rnd.uniform(1.5, 9.0), 3) for a in ids}
        techn = {a: tech(rnd.choice([29.97, 29.97, 25.0, 30.0]), vfr=rnd.random() < 0.1) for a in ids}
        techn[ids[0]] = tech(29.97)
        segs, total = self.plan(ids, durations=durations, transition={"type": "dissolve", "duration": 0.5}, techn=techn)
        for t in self.starts(segs):
            self.assertLessEqual(abs(t * rate - round(t * rate)), 0.0005 * float(rate) + 1e-9, t)
        exact = sum(whole_frame_length(Fraction(str(durations[a])), rate, *input_rate(techn[a])) for a in ids) - 49 * joined_transition(0.5, rate)
        self.assertLessEqual(abs(total - float(exact)), 0.0005)
        self.assertEqual(round(exact * rate), exact * rate, "a whole number of frames")


class TakePicksOnFramesTests(unittest.TestCase):
    """ADR-115: a take assembly's out points on the programme's frame grid, so the Skill's whole-frame join is a no-op and the
    crossfaded audio / beats planned on the picks stay with the delivered picture."""

    def picks(self, *rows):
        return [{"asset_id": f"p{n}", "keep": [s, e], "duration": d} for n, (s, e, d) in enumerate(rows)]

    def test_nearest_frame_of_the_programme_position_so_nothing_adds_up(self):
        """Ten picks of 1.0167 s (30.5 frames each) at 30 fps: rounding each pick would add half a frame ten times; the programme
        position is rounded instead, so the picks alternate 31 / 30 frames and the end stays within half a frame of 10.167 s."""
        ps = self.picks(*[(1.0, 2.0167, 10.0)] * 10)
        moves = align_to_frames(ps, Fraction(30))
        lengths = [round((p["keep"][1] - p["keep"][0]) * 30) for p in ps]
        self.assertEqual(sorted(set(lengths)), [30, 31])
        self.assertEqual(sum(lengths), round(10.167 * 30))
        self.assertEqual(moves["p9"]["programme_out"], round(sum(lengths) / 30, 3))
        segs, total = concat_segments([p["asset_id"] for p in ps], [{"type": "video.trim", "asset": p["asset_id"], "keep": [p["keep"]]} for p in ps], {},
                                      None, rate=Fraction(30), input_rates={p["asset_id"]: (30.0, False) for p in ps})
        self.assertEqual(total, moves["p9"]["programme_out"], "the Skill's whole-frame join changes nothing any more")

    def test_the_millisecond_cut_list_adds_up_to_the_frame_timeline(self):
        """Ten picks of 1.017 s at 30 fps (31 / 30 frames): the cut list the IR keeps (milliseconds) adds up to the concat timeline
        within the take validator's 1 ms (SYNC_TOLERANCE); rounding each pick's length on its own was 1.67 ms off."""
        from video_agent.agent.takes import SYNC_TOLERANCE
        ps = self.picks(*[(1.0, 2.017, 10.0)] * 10)
        align_to_frames(ps, Fraction(30))
        cut_list = sum(p["keep"][1] - p["keep"][0] for p in ps)
        _, total = concat_segments([p["asset_id"] for p in ps], [{"type": "video.trim", "asset": p["asset_id"], "keep": [p["keep"]]} for p in ps], {},
                                   None, rate=Fraction(30), input_rates={p["asset_id"]: (30.0, False) for p in ps})
        self.assertEqual(round(total * 30), 305)
        self.assertLessEqual(abs(cut_list - total), SYNC_TOLERANCE)
        for n in range(1, 11):   # and at every join, not only the end
            self.assertLessEqual(abs(sum(p["keep"][1] - p["keep"][0] for p in ps[:n]) - round(round(sum(31 - k % 2 for k in range(n)) / 30, 6), 3)), SYNC_TOLERANCE)

    def test_one_rate_rule_with_an_ai_generated_first_pick(self):
        """An AI-generated first input does not decide the rate (ADR-099): the first declared-native input's does, for the concat
        and the take alignment alike."""
        techn = {"ai": {"video": {"width": 1280, "height": 720, "fps": 24.0}}, "cam": {"video": {"width": 1920, "height": 1080, "fps": 30.0}}}
        self.assertEqual(programme_rate(["ai", "cam"], techn, {"ai"}), Fraction(30))
        self.assertEqual(programme_rate(["ai", "cam"], techn, set()), Fraction(24))
        self.assertEqual(programme_rate(["ai", "cam"], techn, {"ai"}, {"fps": 25}), Fraction(25))

    def test_never_past_the_source_and_at_least_a_frame(self):
        ps = self.picks((0.35, 6.0, 6.0), (0.0, 0.01, 6.0))   # 169.5 frames would need 6.0167 s of a 6 s source; 0.3 frames
        align_to_frames(ps, Fraction(30))
        self.assertEqual(ps[0]["keep"], [0.35, round(0.35 + 169 / 30, 3)])
        self.assertEqual(round((ps[1]["keep"][1] - ps[1]["keep"][0]) * 30), 1)

    def test_whole_frames_already_and_no_rate_move_nothing(self):
        ps = self.picks((0.55, 4.25, 6.0), (0.4, 6.0, 6.0))
        moves = align_to_frames(ps, Fraction(30))
        self.assertEqual([p["keep"] for p in ps], [[0.55, 4.25], [0.4, 6.0]])
        self.assertFalse(any(m["moved"] for m in moves.values()))
        ps = self.picks((0.0, 1.01, 6.0))
        self.assertEqual(align_to_frames(ps, None), {})
        self.assertEqual(ps[0]["keep"], [0.0, 1.01])


if __name__ == "__main__":
    unittest.main()
