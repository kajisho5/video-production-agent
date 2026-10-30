"""Take selection and beat-synced assembly (ADR-053): `edit.takes` → transcripts grouped into lines → scores from measured facts →
one `takes.pick` per line → `video.concat` of the picks in line order → (music bed) out points on the measured beat grid →
the cut list that the timeline export (ADR-052) and the render use. Fake engine + fake transcription here; the real
ffmpeg-skill `scenes --beats` shape is pinned in tests/test_integration.py::BeatGridRealTests."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import AUDIO_PRODUCTION_CAPS, PipelineCaps, clear_modes, fake_media, pipeline_service, plan_and_render  # noqa: E402
from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.requirements import requirement_map  # noqa: E402
from video_agent.agent.takes import (CROSSFADE_DEFAULT, HANDLE_SECONDS, UNMEASURED, audio_offsets, clean_text, count_fillers, count_restarts, coverage,  # noqa: E402
                                     group_takes, parse_takes_requirements, plan_crossfades, score_takes, similarity, snap_joins)
from video_agent.project.validator import check_take_crossfade  # noqa: E402
from video_agent.models import Requirement  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.project.validator import validate_ir  # noqa: E402
from video_agent.providers import AIProvider, AIRequest, AIResponse, AIUsage  # noqa: E402
from video_agent.tools.ffmpeg_skill.adapter import FfmpegSkillAdapter, PathPolicy  # noqa: E402
from video_agent.tools.ffmpeg_skill.locate import FfmpegSkill  # noqa: E402

L1 = "本日はお集まりいただきありがとうございます"
L2 = "これから新しい製品についてご説明します"
# three takes of line 1 (a restart, a clipped one, a clean one) and three of line 2 (a filler, a clean one, a cut-off one)
TAKES = [
    ("l1_take1.mp4", {"segments": [[0.8, 2.0, "本日はお集まり"], [2.2, 4.4, L1]], "silences": [[0.0, 0.7], [4.6, None]]}),
    ("l1_take2.mp4", {"segments": [[0.6, 4.0, L1]], "silences": [[0.0, 0.5], [4.2, None]], "true_peak": 0.4}),
    ("l1_take3.mp4", {"segments": [[0.7, 4.1, L1]], "silences": [[0.0, 0.6], [4.3, None]]}),
    ("l2_take1.mp4", {"segments": [[0.5, 4.0, "えーと、" + L2]], "silences": [[0.0, 0.4], [4.2, None]]}),
    ("l2_take2.mp4", {"segments": [[0.5, 3.7, L2]], "silences": [[0.0, 0.4], [3.9, None]]}),
    ("l2_take3.mp4", {"segments": [[0.5, 2.2, "これから新しい製品"]], "silences": [[0.0, 0.4], [2.4, None]]}),
]


def reqmap(**kv):
    return requirement_map([Requirement(key=k.replace("__", "."), value=v, provenance="USER", source="cli") for k, v in kv.items()])


def by_subject(doc, subject):
    return [d for d in doc["decisions"] if d["subject"] == subject]


def names(doc):
    return {aid: Path(a["path"]).name for aid, a in doc["assets"].items()}


class TextAndScoreTests(unittest.TestCase):
    """Pure functions: the deterministic text measures, grouping, scoring and the beat snap arithmetic."""

    def test_fillers_restarts_and_clean_text(self):
        self.assertEqual(count_fillers("えーと、本日は、えー、お集まり"), 2)
        self.assertEqual(count_fillers("Um, so, uh, we start"), 2)
        self.assertEqual(count_fillers("あの店で、まあまあ"), 0, "short words that are ordinary words are not fillers")
        self.assertEqual(clean_text("えーと、本日は お集まり！"), "本日はお集まり")
        self.assertEqual(clean_text("Um, Hello, World."), "helloworld")
        self.assertEqual(count_restarts("本日はお集まり 本日はお集まりいただき"), 1)
        self.assertEqual(count_restarts("本日の、えー、本日の講演"), 1, "a filler between the repeat does not hide it")
        self.assertEqual(count_restarts(L1), 0)
        self.assertEqual(count_restarts("the the plan"), 1)
        self.assertEqual(count_restarts("I think I think we should go"), 1)
        self.assertEqual(count_restarts("we should go now"), 0)

    def test_similarity_and_coverage(self):
        self.assertEqual(similarity(clean_text(L1), clean_text("えー、" + L1)), 1.0, "fillers do not separate takes of a line")
        self.assertLess(similarity(clean_text(L1), clean_text(L2)), 0.6)
        self.assertEqual(similarity("", "x"), 0.0)
        self.assertEqual(coverage(clean_text(L2), clean_text(L2)), 1.0)
        self.assertAlmostEqual(coverage(clean_text("これから新しい製品"), clean_text(L2)), 9 / 19, places=3)

    def test_grouping_is_deterministic_and_in_first_appearance_order(self):
        texts = [("b1", clean_text(L2)), ("a1", clean_text(L1)), ("b2", clean_text("えーと" + L2)), ("a2", clean_text(L1 + "ね"))]
        lines = group_takes(texts, 0.6)
        self.assertEqual([ln["takes"] for ln in lines], [["b1", "b2"], ["a1", "a2"]])
        self.assertEqual(group_takes(texts, 0.6), lines)
        self.assertEqual(len(group_takes(texts, 1.0)), 3, "a strict threshold keeps the variant on its own line")

    def _facts(self, **over):
        base = {"index": 0, "raw": L1, "clean": clean_text(L1), "transcript": "obs_t", "speech": [0.5, 4.0], "duration": 6.0, "loudness": "obs_l", "lufs": -16.0,
                "true_peak": -3.0, "silence": "obs_s", "silences": [(0.0, 0.4), (4.2, 6.0)]}
        base.update(over)
        return base

    def test_scores_keep_their_evidence_and_rank_ties_by_input_order(self):
        facts = {"a": self._facts(index=0, transcript="ta", loudness="la", silence="sa"),
                 "b": self._facts(index=1, transcript="tb", loudness="lb", silence="sb", true_peak=0.0),        # clipped
                 "c": self._facts(index=2, transcript="tc", loudness="lc", silence="sc", lufs=-24.0, silences=[(1.0, 2.5)]),   # 8 LU quiet, 1.5 s pause inside speech
                 "d": self._facts(index=3, transcript="td", loudness="ld", silence="sd")}
        ranked = score_takes({"takes": ["a", "b", "c", "d"], "similarity": {k: 1.0 for k in "abcd"}}, facts)
        self.assertEqual([t["asset_id"] for t in ranked], ["a", "d", "c", "b"])
        self.assertEqual([t["rank"] for t in ranked], [1, 2, 3, 4])
        b = next(t for t in ranked if t["asset_id"] == "b")
        clip = next(f for f in b["factors"] if f["factor"] == "clipping")
        self.assertEqual((clip["value"], clip["penalty"], clip["evidence"]), (0.0, 30.0, ["lb"]))
        c = {f["factor"]: f for f in next(t for t in ranked if t["asset_id"] == "c")["factors"]}
        self.assertEqual((c["pauses"]["value"], c["pauses"]["penalty"], c["pauses"]["evidence"]), (1.5, 6.0, ["sc"]))
        self.assertEqual((c["loudness"]["value"], c["loudness"]["penalty"]), (-24.0, 10.0), "2 per LU beyond 3 LU from the line's median (-16)")
        for t in ranked:
            self.assertEqual(t["unmeasured"], ["sharpness", "exposure", "faces"])
            self.assertTrue(all(f["evidence"] for f in t["factors"]))

    def test_a_take_without_audio_measurements_is_scored_on_the_transcript_only(self):
        facts = {"a": self._facts(loudness=None, lufs=None, true_peak=None, silence=None, silences=[])}
        ranked = score_takes({"takes": ["a"], "similarity": {"a": 1.0}}, facts)
        self.assertEqual([f["factor"] for f in ranked[0]["factors"]], ["fillers", "restarts", "coverage"], "no measurement, no factor (never a guessed value)")

    def test_snap_moves_out_points_to_beats_in_programme_time(self):
        picks = [{"asset_id": "a", "keep": [0.5, 4.2], "speech_end": 3.8, "duration": 6.0},    # programme 0..3.7
                 {"asset_id": "b", "keep": [0.3, 3.8], "speech_end": 3.65, "duration": 3.9}]   # starts where a ends
        beats = [i * 0.5 for i in range(30)]
        joins = snap_joins(picks, beats, 0.5)
        self.assertEqual([(j["from"], j["to"], j["snapped"]) for j in joins], [(3.7, 3.5, True), (7.0, 7.0, True)])
        self.assertEqual(picks[0]["keep"], [0.5, 4.0])
        self.assertEqual(picks[1]["keep"], [0.3, 3.8])

    def test_snap_never_cuts_a_word_or_runs_past_the_source(self):
        # the nearest beat (3.5) would cut 0.25 s of speech: the next one in reach (4.0) is taken instead
        p = [{"asset_id": "a", "keep": [0.0, 3.7], "speech_end": 3.6, "duration": 4.2}]
        self.assertEqual(snap_joins(p, [3.5, 4.0], 0.5)[0]["to"], 4.0)
        # no beat between the last word and the end of the source: unsnapped, recorded with why
        p = [{"asset_id": "a", "keep": [0.0, 3.7], "speech_end": 3.6, "duration": 3.8}]
        j = snap_joins(p, [3.5, 4.0], 0.5)[0]
        self.assertFalse(j["snapped"])
        self.assertEqual(p[0]["keep"], [0.0, 3.7])
        self.assertIn("no supported beat", j["why"])
        # equidistant beats: the earlier one
        p = [{"asset_id": "a", "keep": [0.0, 3.75], "speech_end": 3.0, "duration": 6.0}]
        self.assertEqual(snap_joins(p, [3.5, 4.0], 0.5)[0]["to"], 3.5)

    def test_requirements(self):
        self.assertIsNone(parse_takes_requirements(reqmap()))
        self.assertIsNone(parse_takes_requirements(reqmap(edit__takes=False)))
        t = parse_takes_requirements(reqmap(edit__takes=True))
        self.assertEqual((t["similarity"], t["music"], t["snap_max_shift"]), (0.6, None, 0.5))
        t = parse_takes_requirements(reqmap(edit__takes="true", edit__takes__music="2", edit__takes__snap_max_shift=0.25, edit__takes__similarity=0.8))
        self.assertEqual((t["similarity"], t["music"], t["snap_max_shift"]), (0.8, 2, 0.25))
        for bad, msg in (({"edit__takes__similarity": 0.7}, "is set but edit.takes is not"),
                         ({"edit__takes": True, "edit__takes__similarity": 0.1}, "within 0.3..1.0"),
                         ({"edit__takes": True, "edit__takes__music": 1.5}, "input index"),
                         ({"edit__takes": True, "edit__takes__snap_max_shift": 0.3}, "needs edit.takes.music"),
                         ({"edit__takes": "maybe"}, "true or false")):
            with self.assertRaisesRegex(EditRequirementError, msg):
                parse_takes_requirements(reqmap(**bad))


class TakesPipelineTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.takes = [fake_media(self.tmp, n, duration=6.0, **facts) for n, facts in TAKES]

    def tearDown(self):
        clear_modes()

    def plan(self, inputs, reqs, svc=None):
        svc = svc or pipeline_service(self.tmp)
        ir = svc.plan(inputs, "youtube", user_requirements=reqs, params={"language": "ja"})
        return svc, ir

    def test_best_take_per_line_assembled_in_line_order(self):
        svc, ir = self.plan(self.takes, {"edit.takes": True})
        d = ir.doc
        self.assertEqual(svc.validate(ir).errors, [])
        nm = names(d)
        picks = sorted(by_subject(d, "takes.pick"), key=lambda x: x["params"]["line"])
        self.assertEqual([nm[p["params"]["asset_id"]] for p in picks], ["l1_take3.mp4", "l2_take2.mp4"])
        self.assertEqual([[nm[t["asset_id"]] for t in p["params"]["takes"]] for p in picks],
                         [["l1_take3.mp4", "l1_take1.mp4", "l1_take2.mp4"], ["l2_take2.mp4", "l2_take1.mp4", "l2_take3.mp4"]], "ranked best first")
        p1 = {nm[t["asset_id"]]: t for t in picks[0]["params"]["takes"]}
        self.assertEqual((p1["l1_take3.mp4"]["score"], p1["l1_take1.mp4"]["score"], p1["l1_take2.mp4"]["score"]), (100.0, 85.0, 70.0))
        for p in picks:
            self.assertEqual((p["type"], p["approval"], p["provenance"]), ("TRANSFORM", "CONFIRM", "INFERRED"))
            self.assertEqual(p["params"]["unmeasured"], UNMEASURED)
            self.assertIn("sharpness, exposure and faces are not measured", p["reason"])
            cited = {e for t in p["params"]["takes"] for f in t["factors"] for e in f["evidence"]}
            self.assertTrue(cited <= set(p["evidence"]), "every measurement a score rests on is evidence of the pick")
            kinds = {o["kind"] for o in d["analysis"]["observations"] if o["id"] in cited}
            self.assertEqual(kinds, {"transcript", "loudness", "silence"})
        # the pick keeps its speech span plus the handles; the other takes get no decision and no step
        self.assertEqual(picks[0]["params"]["keep"], [round(0.7 - HANDLE_SECONDS, 3), round(4.1 + HANDLE_SECONDS, 3)])
        unused = {a for a, n in nm.items() if n not in ("l1_take3.mp4", "l2_take2.mp4")}
        self.assertFalse([x for x in d["decisions"] if x["params"].get("asset_id") in unused])
        self.assertFalse([s for s in d["plan"]["steps"] if set(s["inputs"]) & unused])
        concat = next(op for op in d["video"]["operations"] if op["type"] == "video.concat")
        self.assertEqual([nm[a] for a in concat["inputs"]], ["l1_take3.mp4", "l2_take2.mp4"])
        self.assertTrue({p["id"] for p in picks} <= set(concat["decision_ids"]), "the assembly cites the picks")
        trims = [op for op in d["video"]["operations"] if op["type"] == "video.trim"]
        self.assertEqual([op["keep"] for op in trims], [[p["params"]["keep"]] for p in picks])
        self.assertFalse(any(op["accurate"] for op in trims), "no beat snap: the usual cut precision")
        self.assertFalse(by_subject(d, "silence.leading"), "a pick's own range replaces the silence trims")

    def test_interleaved_takes_group_by_what_is_said(self):
        order = [self.takes[i] for i in (3, 0, 4, 1, 5, 2)]   # l2_take1, l1_take1, l2_take2, l1_take2, l2_take3, l1_take3
        _, ir = self.plan(order, {"edit.takes": True})
        nm = names(ir.doc)
        picks = sorted(by_subject(ir.doc, "takes.pick"), key=lambda x: x["params"]["line"])
        self.assertEqual([nm[p["params"]["asset_id"]] for p in picks], ["l2_take2.mp4", "l1_take3.mp4"], "line order = the order lines first appear")

    def test_render_and_timeline_follow_the_picks(self):
        svc = pipeline_service(self.tmp)
        r = plan_and_render(svc, self.takes, {"edit.takes": True, "delivery.timeline": "otio"})
        self.assertEqual(r["validation"].errors, [])
        self.assertEqual((r["out"]["execution"]["status"], r["out"]["qa"]["status"]), ("COMPLETED", "PASS"))
        d = r["ir"].doc
        nm = names(d)
        tl = d["delivery"]["timelines"][0]
        self.assertEqual([[nm[c["asset"]], c["in"], c["out"]] for c in tl["clips"]], [["l1_take3.mp4", 0.55, 4.25], ["l2_take2.mp4", 0.35, 3.85]])
        picks = {p["id"] for p in by_subject(d, "takes.pick")}
        self.assertTrue(picks <= set(tl["decision_ids"]), "the timeline rests on the picks")
        self.assertEqual({a["type"] for a in r["out"]["artifacts"]}, {"YOUTUBE", "TIMELINE"})

    def test_beat_snapped_joins_land_on_measured_beats(self):
        music = fake_media(self.tmp, "bed.mp4", duration=20.0, beat_interval=0.5, beat_phase=0.25, beat_confidence=0.96)
        svc, ir = self.plan(self.takes + [music], {"edit.takes": True, "edit.takes.music": 6, "delivery.timeline": "otio"})
        d = ir.doc
        self.assertEqual(svc.validate(ir).errors, [])
        snap = by_subject(d, "takes.beat_snap")[0]
        beats = next(o for o in d["analysis"]["observations"] if o["kind"] == "beats")
        self.assertEqual(beats["asset_id"], snap["params"]["music"])
        self.assertIn(beats["id"], snap["evidence"])
        self.assertEqual((snap["type"], snap["approval"], snap["params"]["snapped"], snap["params"]["tempo_bpm"]), ("TRANSFORM", "AUTO", 2, 120.0))
        self.assertEqual(len([o for o in d["analysis"]["observations"] if o["kind"] == "beats"]), 1, "the grid is measured on the music bed only")
        # the cut list ends every pick on its beat in programme time
        trims = [op for op in d["video"]["operations"] if op["type"] == "video.trim"]
        t = 0.0
        for op, j in zip(trims, snap["params"]["joins"]):
            t += op["keep"][0][1] - op["keep"][0][0]
            self.assertAlmostEqual(t, j["to"], places=3)
            self.assertIn(j["to"], beats["data"]["supported_beats"])
            self.assertTrue(op["accurate"], "a beat-snapped cut is frame-accurate")
            self.assertIn(snap["id"], op["decision_ids"])
        self.assertNotIn(snap["params"]["music"], [c["asset"] for c in d["delivery"]["timelines"][0]["clips"]], "the bed is not a clip")
        # a cut list moved off its beat after planning is refused
        bad = json.loads(json.dumps(d))
        tr = next(op for op in bad["video"]["operations"] if op["type"] == "video.trim")
        tr["keep"] = [[tr["keep"][0][0], round(tr["keep"][0][1] - 0.1, 3)]]
        ir.doc = bad
        self.assertTrue(any("not on its beat" in e for e in svc.validate(ir).errors))
        bad2 = json.loads(json.dumps(d))
        next(x for x in bad2["decisions"] if x["subject"] == "takes.beat_snap")["params"]["joins"][0]["to"] = 3.8
        ir.doc = bad2
        self.assertTrue(any("is not a supported beat" in e for e in svc.validate(ir).errors))

    def test_beat_grid_unmeasurable_or_unusable_blocks(self):
        bed = fake_media(self.tmp, "bed.wav", duration=20.0, video=False)   # scenes --beats refuses an input without a video stream
        _, ir = self.plan(self.takes[:3] + [bed], {"edit.takes": True, "edit.takes.music": 3})
        b = by_subject(ir.doc, "takes.beat_snap")[0]
        self.assertEqual((b["type"], b["status"]), ("BLOCK", "BLOCKED"))
        self.assertIn("input has no video stream", b["reason"])
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        weak = fake_media(self.tmp, "weak.mp4", duration=20.0, beat_confidence=0.3)
        _, ir = self.plan(self.takes[:3] + [weak], {"edit.takes": True, "edit.takes.music": 3})
        b = by_subject(ir.doc, "takes.beat_snap")[0]
        self.assertEqual(b["type"], "BLOCK")
        self.assertIn("not usable", b["decision"])

    def test_conflicts_and_unsupported_takes_block(self):
        _, ir = self.plan(self.takes[:2], {"edit.takes": True, "edit.concat": True})
        self.assertEqual([(x["subject"], x["type"]) for x in ir.doc["decisions"] if x["type"] == "BLOCK"], [("takes.group", "BLOCK")])
        self.assertFalse(by_subject(ir.doc, "video.concat"), "the competing concat is not decided as well")
        mute = fake_media(self.tmp, "mute.mp4", duration=6.0, segments=[], silences=[[0.0, None]])
        _, ir = self.plan(self.takes[:2] + [mute], {"edit.takes": True})
        b = by_subject(ir.doc, "takes.group")[0]
        self.assertEqual(b["type"], "BLOCK")
        self.assertIn("no recognised speech", b["decision"])
        with self.assertRaisesRegex(ValueError, "edit.takes.music is input index 9"):
            self.plan(self.takes[:2], {"edit.takes": True, "edit.takes.music": 9})
        with self.assertRaisesRegex(ValueError, "is set but edit.takes is not"):
            self.plan(self.takes[:2], {"edit.takes.similarity": 0.7})

    def test_single_line_delivers_the_pick_alone(self):
        svc, ir = self.plan(self.takes[:3], {"edit.takes": True})
        d = ir.doc
        self.assertEqual(svc.validate(ir).errors, [])
        self.assertFalse(by_subject(d, "video.concat"))
        nm = names(d)
        steps = [s for s in d["plan"]["steps"] if s["skill"] == "silence_cleanup"]
        self.assertEqual([nm[s["inputs"][0]] for s in steps], ["l1_take3.mp4"])
        self.assertEqual({s["skill"] for s in d["plan"]["steps"]} & {"delivery_export"}, {"delivery_export"})

    def test_rejected_pick_rejects_the_assembly(self):
        svc, ir = self.plan(self.takes, {"edit.takes": True})
        path = str(Path(self.tmp) / "p.json"); save_ir(ir, path)
        pick = by_subject(ir.doc, "takes.pick")[0]
        svc.reject(load_ir(path), path, [pick["id"]], "prefer another take")
        rep = svc.validate(load_ir(path))
        self.assertTrue(any("cites REJECTED decision" in e and "video.concat" in e for e in rep.errors), rep.errors)


L3 = "最後にご質問をお受けいたします"
# three lines for the crossfades (ADR-054): line 2 runs to the end of its source and line 3 starts at 0, so join 2 has no room for handles
XF_TAKES = [("l1_take1.mp4", {"segments": [[0.7, 4.1, L1]], "silences": [[0.0, 0.6], [4.3, None]]}),
            ("l2_take1.mp4", {"segments": [[0.5, 5.95, L2]], "silences": [[0.0, 0.4]]}),
            ("l3_take1.mp4", {"segments": [[0.1, 3.0, L3]], "silences": [[3.2, None]]})]


class CrossfadeArithmeticTests(unittest.TestCase):
    """ADR-054: handles at each join so audio-production's CONCAT overlap does not shorten the programme or move the audio off the picture."""

    def picks(self, *rows):
        return [{"asset_id": a, "keep": [s, e], "duration": dur} for a, s, e, dur in rows]

    def check_sync(self, xf):
        offs = audio_offsets(xf["spans"], xf["joins"])
        for sp, off in zip(xf["spans"], offs):
            self.assertAlmostEqual(off + sp["in_handle"], sp["programme_start"], places=6, msg="every pick's audio starts with its picture")
        self.assertAlmostEqual(xf["audio_duration"], xf["programme_duration"], places=6, msg="the programme audio ends with the picture")

    def test_centred_handles_keep_the_picture_and_the_length(self):
        xf = plan_crossfades(self.picks(("a", 0.5, 4.0, 6.0), ("b", 0.4, 3.0, 6.0), ("c", 1.0, 2.0, 6.0)), 0.01)
        self.assertEqual(xf["crossfaded"], 2)
        self.assertEqual([sp["video"] for sp in xf["spans"]], [[0.5, 4.0], [0.4, 3.0], [1.0, 2.0]], "the picture's cut points do not move")
        self.assertEqual([sp["audio"] for sp in xf["spans"]], [[0.5, 4.005], [0.395, 3.005], [0.995, 2.0]])
        self.assertTrue(all(j["centred"] for j in xf["joins"]))
        self.assertEqual([j["window"] for j in xf["joins"]], [[3.495, 3.505], [6.095, 6.105]], "each crossfade is centred on its cut")
        self.assertEqual(xf["runs"], [[0, 1, 2]])
        self.assertEqual(xf["programme_duration"], 7.1)
        self.check_sync(xf)

    def test_one_sided_room_moves_the_overlap_to_the_other_side(self):
        # b starts at 0.0 of its source: the whole overlap comes from a's tail (still in sync, no longer centred)
        xf = plan_crossfades(self.picks(("a", 0.5, 4.0, 6.0), ("b", 0.0, 3.0, 6.0)), 0.02)
        j = xf["joins"][0]
        self.assertEqual((j["crossfaded"], j["out_handle"], j["in_handle"], j["centred"]), (True, 0.02, 0.0, False))
        xf2 = plan_crossfades(self.picks(("a", 0.5, 5.995, 6.0), ("b", 1.0, 3.0, 6.0)), 0.02)
        self.assertEqual((xf2["joins"][0]["out_handle"], xf2["joins"][0]["in_handle"]), (0.005, 0.015))
        self.check_sync(xf)
        self.check_sync(xf2)

    def test_a_join_without_room_keeps_a_hard_cut_and_says_why(self):
        xf = plan_crossfades(self.picks(("a", 0.5, 4.0, 6.0), ("b", 0.4, 6.0, 6.0), ("c", 0.0, 2.0, 6.0)), 0.01)
        self.assertEqual([j["crossfaded"] for j in xf["joins"]], [True, False])
        self.assertIn("no room for a 0.01s crossfade", xf["joins"][1]["why"])
        self.assertIn("keeps a hard cut", xf["joins"][1]["why"])
        self.assertEqual((xf["spans"][1]["out_handle"], xf["spans"][2]["in_handle"]), (0.0, 0.0))
        self.assertEqual(xf["runs"], [[0, 1], [2]])
        self.check_sync(xf)
        short = plan_crossfades(self.picks(("a", 0.5, 4.0, 6.0), ("b", 1.0, 1.005, 6.0)), 0.01)
        self.assertFalse(short["joins"][0]["crossfaded"])
        self.assertIn("not longer than the 0.01s crossfade", short["joins"][0]["why"])

    def test_requirement(self):
        t = parse_takes_requirements(reqmap(edit__takes=True))
        self.assertEqual((t["crossfade"], t["crossfade_explicit"]), (CROSSFADE_DEFAULT, None))
        self.assertEqual(CROSSFADE_DEFAULT, 0.01)
        t = parse_takes_requirements(reqmap(edit__takes=True, edit__takes__crossfade="0.02"))
        self.assertEqual((t["crossfade"], t["crossfade_explicit"].key), (0.02, "edit.takes.crossfade"))
        self.assertEqual(parse_takes_requirements(reqmap(edit__takes=True, edit__takes__crossfade=0))["crossfade"], 0.0)
        for bad, msg in (({"edit__takes__crossfade": 0.01}, "is set but edit.takes is not"), ({"edit__takes": True, "edit__takes__crossfade": 0.2}, "within 0.0..0.1"),
                         ({"edit__takes": True, "edit__takes__crossfade": -0.01}, "within 0.0..0.1")):
            with self.assertRaisesRegex(EditRequirementError, msg):
                parse_takes_requirements(reqmap(**bad))


class TakeCrossfadePipelineTests(unittest.TestCase):
    """ADR-054 on the fake engine + fake Skills: decision → spans / joins / mux steps → IR record → validator sync check → render."""

    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.takes = [fake_media(self.tmp, n, duration=6.0, **facts) for n, facts in XF_TAKES]

    def tearDown(self):
        clear_modes()

    def svc(self, audio_production=True):
        return pipeline_service(self.tmp, caps=PipelineCaps(extra=AUDIO_PRODUCTION_CAPS if audio_production else ()))

    def plan(self, reqs, inputs=None, audio_production=True):
        svc = self.svc(audio_production)
        return svc, svc.plan(inputs or self.takes, "youtube", user_requirements=reqs, params={"language": "ja"})

    def test_default_crossfade_plans_spans_joins_and_a_mux_without_moving_a_cut(self):
        svc, ir = self.plan({"edit.takes": True})
        d = ir.doc
        self.assertEqual(svc.validate(ir).errors, [])
        nm = names(d)
        x = by_subject(d, "takes.crossfade")[0]
        self.assertEqual((x["type"], x["approval"], x["provenance"], x["risk"]), ("TRANSFORM", "AUTO", "SYSTEM", "LOW"))
        self.assertEqual(x["params"]["crossfade"], 0.01)
        self.assertEqual([j["crossfaded"] for j in x["params"]["joins"]], [True, False])
        rec = d["audio"]["take_crossfade"]
        self.assertEqual([c["inputs"] for c in rec["concats"]], [[rec["spans"][0]["output"], rec["spans"][1]["output"]], ["programme_xfade_run1", rec["spans"][2]["output"]]])
        self.assertEqual([nm[sp["asset"]] for sp in rec["spans"]], ["l1_take1.mp4", "l2_take1.mp4", "l3_take1.mp4"])
        self.assertEqual([c["crossfade"] for c in rec["concats"]], [0.01, 0.0], "the crossfaded run, then the runs joined at the hard cut")
        # the picture is exactly what the picks decided, cut frame-accurately; the same request without crossfades has the same cut list
        _, ir0 = self.plan({"edit.takes": True, "edit.takes.crossfade": 0})
        keeps = lambda doc: [op["keep"] for op in doc["video"]["operations"] if op["type"] == "video.trim"]
        self.assertEqual(keeps(d), keeps(ir0.doc))
        self.assertFalse(by_subject(ir0.doc, "takes.crossfade"), "crossfade 0: nothing decided, hard-cut audio")
        self.assertNotIn("take_crossfade", ir0.doc["audio"])
        self.assertTrue(all(op["accurate"] for op in d["video"]["operations"] if op["type"] == "video.trim"))
        self.assertFalse(any(op["accurate"] for op in ir0.doc["video"]["operations"] if op["type"] == "video.trim"))
        # steps: three audio spans, two joins, the mux on the assembled picture; later steps continue on the mux output
        steps = [(s["skill"], s["inputs"], s["outputs"]) for s in d["plan"]["steps"]]
        self.assertEqual([s[0] for s in steps], ["silence_cleanup"] * 3 + ["video_concat", "audio_cut", "audio_cut", "audio_cut", "audio_concat", "audio_concat", "audio_mux",
                                                                          "loudness_normalization", "delivery_export", "delivery_check"])
        mux = next(s for s in d["plan"]["steps"] if s["skill"] == "audio_mux")
        self.assertEqual((mux["inputs"], mux["outputs"], mux["tool"]), (["programme", "programme_xfade_audio"], ["programme_xfade"], "ffmpeg-skill/audio"))
        self.assertEqual(next(s for s in d["plan"]["steps"] if s["skill"] == "loudness_normalization")["inputs"], ["programme_xfade"])
        self.assertEqual(check_take_crossfade(d), [])

    def test_validator_refuses_audio_that_would_drift_or_a_moved_cut(self):
        svc, ir = self.plan({"edit.takes": True})
        d = ir.doc

        def errs(mutate):
            bad = json.loads(json.dumps(d))
            mutate(bad)
            ir.doc = bad
            return svc.validate(ir).errors

        def handle(b):
            b["audio"]["take_crossfade"]["spans"][1]["in_handle"] = 0.0
            b["audio"]["take_crossfade"]["spans"][1]["audio"][0] = b["audio"]["take_crossfade"]["spans"][1]["video"][0]
        self.assertTrue(any("do not make its 0.01s overlap" in e for e in errs(handle)))
        self.assertTrue(any("ms off its picture" in e for e in errs(handle)), "the replayed CONCAT puts the next pick's audio off its picture")

        def moved(b):
            tr = next(op for op in b["video"]["operations"] if op["type"] == "video.trim")
            tr["keep"] = [[tr["keep"][0][0], round(tr["keep"][0][1] - 0.04, 3)]]
        self.assertTrue(any("the cut list must not move" in e for e in errs(moved)))

        def keyframe(b):
            next(op for op in b["video"]["operations"] if op["type"] == "video.trim")["accurate"] = False
        self.assertTrue(any("cut frame-accurately" in e for e in errs(keyframe)))

        def mux(b):
            b["audio"]["take_crossfade"]["mux"]["audio"] = b["audio"]["take_crossfade"]["spans"][0]["output"]
        self.assertTrue(any("the mux puts" in e for e in errs(mux)))

        def crossfade(b):
            b["audio"]["take_crossfade"]["concats"][0]["crossfade"] = 0.02
        self.assertTrue(any("ms off its picture" in e or "do not match the assembled picture" in e for e in errs(crossfade)))

        def dropped(b):
            del b["audio"]["take_crossfade"]
        self.assertTrue(any("no crossfade record" in e for e in errs(dropped)))

    def test_render_runs_the_audio_chain_and_delivers_the_muxed_programme(self):
        svc = self.svc()
        r = plan_and_render(svc, self.takes, {"edit.takes": True, "delivery.timeline": "otio"})
        self.assertEqual(r["validation"].errors, [])
        self.assertEqual((r["out"]["execution"]["status"], r["out"]["qa"]["status"]), ("COMPLETED", "PASS"))
        tools = [x["tool"] for x in r["out"]["execution"]["results"]]
        self.assertEqual(tools.count("audio-production/run"), 5)
        self.assertLess(tools.index("video-editing/concat"), tools.index("ffmpeg-skill/audio"))
        self.assertLess(tools.index("ffmpeg-skill/audio"), tools.index("ffmpeg-skill/loudness"))
        prov = json.loads((Path(svc.workspace) / "jobs" / r["out"]["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        xid = by_subject(r["ir"].doc, "takes.crossfade")[0]["id"]
        cat = [e for e in prov["operations"] if e["tool"] == "audio-production/run" and e["args"].get("operation") == "CONCAT"]
        self.assertEqual([e["args"]["crossfade"] for e in cat], [0.01, 0.0])
        self.assertTrue(all(xid in e["decision"] for e in cat), "every audio join records the crossfade decision it executes")
        mux = next(e for e in prov["operations"] if e["tool"] == "ffmpeg-skill/audio")
        self.assertEqual(set(mux["args"]), {"input", "replace", "output"})
        # the timeline export is the picture's cut list (hard cuts), unchanged by the audio crossfades
        nm = names(r["ir"].doc)
        tl = r["ir"].doc["delivery"]["timelines"][0]
        self.assertEqual([[nm[c["asset"]], c["in"], c["out"]] for c in tl["clips"]], [["l1_take1.mp4", 0.55, 4.25], ["l2_take1.mp4", 0.35, 6.0], ["l3_take1.mp4", 0.0, 3.15]])

    def test_beat_snapped_joins_keep_their_beats_under_the_crossfades(self):
        music = fake_media(self.tmp, "bed.mp4", duration=20.0, beat_interval=0.5, beat_phase=0.25, beat_confidence=0.96)
        reqs = {"edit.takes": True, "edit.takes.music": 3}
        svc, ir = self.plan(reqs, self.takes + [music])
        _, ir0 = self.plan(dict(reqs, **{"edit.takes.crossfade": 0}), self.takes + [music])
        self.assertEqual(svc.validate(ir).errors, [])
        keeps = lambda doc: [op["keep"] for op in doc["video"]["operations"] if op["type"] == "video.trim"]
        self.assertEqual(keeps(ir.doc), keeps(ir0.doc), "the snapped cut points are the same with and without crossfades")
        snap = by_subject(ir.doc, "takes.beat_snap")[0]
        rec = ir.doc["audio"]["take_crossfade"]
        self.assertEqual([j["at"] for j in rec["joins"]], [j["to"] for j in snap["params"]["joins"]][:-1], "each crossfade sits on its beat-snapped cut")

    def test_no_room_anywhere_or_no_audio_production(self):
        # two picks whose only join has no room: KEEP with the reason, no audio chain
        _, ir = self.plan({"edit.takes": True}, self.takes[1:])
        x = by_subject(ir.doc, "takes.crossfade")[0]
        self.assertEqual((x["type"], x["params"]["crossfaded"]), ("KEEP", 0))
        self.assertIn("no join has room", x["reason"])
        self.assertNotIn("take_crossfade", ir.doc["audio"])
        # the default without audio-production here: KEEP (hard-cut audio, the reason names what is missing); explicit: BLOCK
        _, ir = self.plan({"edit.takes": True}, audio_production=False)
        x = by_subject(ir.doc, "takes.crossfade")[0]
        self.assertEqual((x["type"], x["params"]["unavailable"]), ("KEEP", ["audio_cut", "audio_concat"]))
        self.assertIn("set edit.takes.crossfade=0", x["reason"])
        self.assertFalse(ir.blocked())
        _, ir = self.plan({"edit.takes": True, "edit.takes.crossfade": 0.02}, audio_production=False)
        self.assertEqual(sorted(b["subject"] for b in ir.blocked()), ["capability.audio_concat", "capability.audio_cut"])

    def test_rejected_crossfade_is_rejected_cited(self):
        svc, ir = self.plan({"edit.takes": True})
        path = str(Path(self.tmp) / "x.json"); save_ir(ir, path)
        x = by_subject(ir.doc, "takes.crossfade")[0]
        svc.reject(load_ir(path), path, [x["id"]], "keep the hard cuts")
        rep = svc.validate(load_ir(path))
        self.assertTrue(any("cites REJECTED decision" in e and "takes.crossfade" in e for e in rep.errors), rep.errors)


class _TakePreference(AIProvider):
    """A provider that prefers one named take of a line, citing that take's recorded transcript (or a fabricated id)."""
    name = "fake"
    model = "fake-takes-1"

    def __init__(self, prefer, fabricate=False, also=None):
        self.prefer, self.fabricate, self.also = prefer, fabricate, also or []

    def available(self):
        return True

    def complete(self, request: AIRequest, timeout=None) -> AIResponse:
        recs = []
        for nm in [self.prefer] + self.also:
            asset = next(a for a in request.inputs["assets"] if a["name"] == nm)
            ev = ["obs_fabricated"] if self.fabricate else [o["id"] for o in request.inputs["observations"] if o["asset_id"] == asset["id"] and o["kind"] == "transcript"]
            recs.append({"intent": "take_selection", "asset_id": asset["id"], "statement": f"{nm} has the warmer delivery", "confidence": 0.7, "evidence": ev, "params": {}})
        return AIResponse(task_type=request.task_type, result={"recommendations": recs}, confidence=0.7, provider=self.name, model=self.model, evidence=[], reasoning="",
                          usage=AIUsage(input_tokens=1, output_tokens=1), latency_s=0.0, metadata={})


class TakesAITests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.takes = [fake_media(self.tmp, n, duration=6.0, **facts) for n, facts in TAKES[:3]]

    def plan_with(self, provider):
        svc = pipeline_service(self.tmp)
        svc.provider = provider
        ir = svc.plan(self.takes, "youtube", user_requirements={"edit.takes": True}, params={"language": "ja"})
        return svc, ir

    def test_ai_preference_overrides_the_ranking_with_scores_kept(self):
        svc, ir = self.plan_with(_TakePreference("l1_take1.mp4"))
        d = ir.doc
        self.assertEqual(svc.validate(ir).errors, [])
        nm = names(d)
        p = by_subject(d, "takes.pick")[0]
        self.assertEqual(nm[p["params"]["asset_id"]], "l1_take1.mp4")
        self.assertEqual(nm[p["params"]["deterministic_pick"]], "l1_take3.mp4")
        self.assertEqual((p["provenance"], p["approval"]), ("AI_GENERATED", "CONFIRM"))
        self.assertEqual({nm[t["asset_id"]]: t["score"] for t in p["params"]["takes"]}, {"l1_take3.mp4": 100.0, "l1_take1.mp4": 85.0, "l1_take2.mp4": 70.0},
                         "the AI changes the pick, never a measurement")
        self.assertEqual(p["params"]["ai"]["provider"], "fake")
        self.assertIn(p["params"]["ai"]["inference"], p["evidence"])
        self.assertIn("preferred", p["reason"])
        self.assertFalse(by_subject(d, "ai.take_selection"), "the recommendation is covered by the pick it changed")
        self.assertEqual(d["provenance"]["ai_calls"][0]["provider"], "fake")

    def test_ai_without_recorded_evidence_or_ambiguous_changes_nothing(self):
        _, ir = self.plan_with(_TakePreference("l1_take1.mp4", fabricate=True))
        nm = names(ir.doc)
        p = by_subject(ir.doc, "takes.pick")[0]
        self.assertEqual((nm[p["params"]["asset_id"]], p["provenance"]), ("l1_take3.mp4", "INFERRED"))
        self.assertTrue(any("cites no existing observation" in w for w in ir.doc["analysis"]["warnings"]))
        _, ir = self.plan_with(_TakePreference("l1_take1.mp4", also=["l1_take2.mp4"]))
        p = by_subject(ir.doc, "takes.pick")[0]
        self.assertEqual(names(ir.doc)[p["params"]["asset_id"]], "l1_take3.mp4")
        self.assertIn("more than one take", p["reason"])
        self.assertEqual(len(by_subject(ir.doc, "ai.take_selection")), 2, "each unused preference stays a review item")


class BeatMeasurementAdapterTests(unittest.TestCase):
    """The reference adapter's side of the beat measurement: a typed flag, never free text."""

    def test_scenes_beats_argv(self):
        tmp = os.path.realpath(tempfile.mkdtemp())
        root = Path(tmp) / "ffmpeg-skill"
        (root / "scripts").mkdir(parents=True)
        (root / "scripts" / "scenes.py").write_text("print('{}')\n", encoding="utf-8")
        src_dir = Path(tmp) / "src"; src_dir.mkdir()
        src = str(src_dir / "bed.mp4"); Path(src).write_bytes(b"x")
        ad = FfmpegSkillAdapter(FfmpegSkill(root, "2.21.0", ["scenes"]), PathPolicy([str(src_dir)], tmp))
        self.assertEqual(ad.build_argv("ffmpeg-skill/scenes", {"input": src, "beats": True}, {}), [src, "--beats"])
        self.assertTrue(ad.supports("ffmpeg-skill/scenes"))


if __name__ == "__main__":
    unittest.main()
