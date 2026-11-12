"""Word-index plans (ADR-109): the transcript's words are numbered, the AI names the words to keep / cut as [first_index, last_index]
ranges (never seconds), and the agent converts them with rule word_ranges@1 (pre / post roll, clamped at the gap to a word that is
not kept and to the media). Invalid ranges are refused, never corrected; the IR records the word ranges, the derived seconds and the
rule, and validate_plan derives them again. Fake engine + fake transcription + a fake AI provider; no media is decoded."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import clear_modes, fake_media, pipeline_service  # noqa: E402
from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.requirements import requirement_map  # noqa: E402
from video_agent.agent.word_cut import (POST_ROLL_DEFAULT, PRE_ROLL_DEFAULT, RULE_VERSION, WordRangeError, check_word_cuts, derive_ranges,  # noqa: E402
                                        number_words, parse_word_cut_requirements, recommendation_ranges)
from video_agent.models import Requirement  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.providers import AIProvider, AIRequest, AIResponse, AIUsage  # noqa: E402

EN = [[0.5, 0.8, "Hello"], [0.9, 1.2, "um"], [1.3, 1.7, "world"], [2.5, 2.9, "this"], [3.0, 3.3, "is"], [3.4, 3.9, "a"], [4.0, 4.6, "test"]]
JA = [[0.4, 0.7, "本日"], [0.7, 0.8, "は"], [0.9, 1.3, "えー"], [1.4, 1.9, "お集まり"], [1.9, 2.4, "いただき"], [2.4, 3.2, "ありがとうございます"]]


def words_of(rows, **extra):
    return [dict({"index": i, "start": s, "end": e, "text": t}, **extra) for i, (s, e, t) in enumerate(rows)]


def segs(rows, split=None, **wextra):
    """One segment per run of rows (split: indices where a new segment starts); words in the transcript-skill shape."""
    split = sorted(set([0] + list(split or [])))
    out = []
    for k, a in enumerate(split):
        b = split[k + 1] if k + 1 < len(split) else len(rows)
        ws = [dict({"start": s, "end": e, "text": t, "confidence": 0.9}, **wextra) for s, e, t in rows[a:b]]
        out.append({"id": f"seg_{k + 1:04d}", "start": rows[a][0], "end": rows[b - 1][1], "text": " ".join(r[2] for r in rows[a:b]), "words": ws})
    return out


def reqmap(**kv):
    return requirement_map([Requirement(key=k.replace("__", "."), value=v, provenance="USER", source="cli") for k, v in kv.items()])


class NumberingTests(unittest.TestCase):
    def test_numbering_is_zero_based_over_the_whole_transcript_in_order(self):
        words, src = number_words(segs(EN, split=[3]))
        self.assertEqual(src, "derived")
        self.assertEqual([w["index"] for w in words], list(range(7)))
        self.assertEqual([w["text"] for w in words], [r[2] for r in EN])
        self.assertEqual([w["segment_id"] for w in words], ["seg_0001"] * 3 + ["seg_0002"] * 4, "numbering continues across segments")
        self.assertEqual(words[0]["confidence"], 0.9)

    def test_segments_without_word_timestamps_have_no_numbered_words(self):
        s = segs(EN, split=[3])
        s.insert(1, {"id": "seg_x", "start": 1.8, "end": 2.4, "text": "untimed", "words": None})
        words, _ = number_words(s)
        self.assertEqual(len(words), 7)

    def test_the_transcripts_own_index_is_used_when_every_word_has_one(self):
        s = segs(EN)
        for i, w in enumerate(s[0]["words"]):
            w["index"] = i
        words, src = number_words(s)
        self.assertEqual((src, [w["index"] for w in words]), ("transcript", list(range(7))))
        s[0]["words"][0]["index"], s[0]["words"][1]["index"] = 1, 0   # indices out of the words' order: the AI would read other words than are cut
        with self.assertRaises(WordRangeError):
            number_words(s)

    def test_indices_on_some_words_only_or_with_a_gap_are_refused(self):
        s = segs(EN)
        s[0]["words"][0]["index"] = 0
        with self.assertRaises(WordRangeError):
            number_words(s)
        for i, w in enumerate(s[0]["words"]):
            w["index"] = i * 2
        with self.assertRaises(WordRangeError):
            number_words(s)
        for i, w in enumerate(s[0]["words"]):
            w["index"] = True if i == 0 else i   # present but not a whole number: refused, not silently re-derived
        with self.assertRaises(WordRangeError):
            number_words(s)
        for i, w in enumerate(s[0]["words"]):
            w["index"] = str(i)
        with self.assertRaises(WordRangeError):
            number_words(s)

    def test_new_optional_word_fields_are_carried_and_not_required(self):
        s = segs(JA, pos="noun", timing="interpolated")   # transcription-skill 0.9.0 dictionary words
        words, _ = number_words(s)
        self.assertTrue(all(w["pos"] == "noun" and w["timing"] == "interpolated" and w["interpolated"] for w in words))
        measured, _ = number_words(segs(JA, timing="measured"))
        self.assertNotIn("interpolated", measured[0])
        self.assertNotIn("timing", number_words(segs(JA))[0][0], "a transcript older than 0.9.0 has no timing key")

    def test_a_0_9_0_transcript_with_null_word_segments_numbers_contiguously(self):
        s = segs(EN, split=[3])
        s.insert(1, {"id": "seg_x", "start": 1.8, "end": 2.4, "text": "untimed", "words": None})
        n = 0
        for seg in s:
            for x in seg["words"] or []:
                x["index"] = n
                n += 1
        words, src = number_words(s)
        self.assertEqual((src, [x["index"] for x in words]), ("transcript", list(range(7))), "segments with words: null consume no numbers")


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.words = words_of(EN)

    def test_defaults_are_the_documented_rolls(self):
        self.assertEqual((PRE_ROLL_DEFAULT, POST_ROLL_DEFAULT, RULE_VERSION), (0.1, 0.2, "word_ranges@1"))

    def test_pre_and_post_roll_when_the_neighbours_are_far(self):
        words = words_of([[1.0, 1.5, "a"], [3.0, 3.5, "b"], [6.0, 6.5, "c"]])
        r = derive_ranges(words, [[1, 1]], "keep", 10.0)
        self.assertEqual(r["keep"], [[2.9, 3.7]])
        self.assertEqual(r["remove"], [[0.0, 2.9], [3.7, 10.0]])
        r = derive_ranges(words, [[1, 1]], "keep", 10.0, pre_roll=0.3, post_roll=0.0)
        self.assertEqual(r["keep"], [[2.7, 3.5]])

    def test_a_kept_range_is_clamped_at_the_gap_midpoint_to_a_word_that_is_not_kept(self):
        r = derive_ranges(self.words, [[1, 1]], "cut", 6.0)   # cut "um" (0.9-1.2) between Hello (..0.8) and world (1.3..)
        self.assertEqual(r["kept_words"], [[0, 0], [2, 6]])
        self.assertEqual(r["remove"], [[0.85, 1.25]], "0.8+0.2 and 1.3-0.1 would swallow 'um'; both stop at the gap midpoints")
        self.assertEqual(r["keep"], [[0.0, 0.85], [1.25, 6.0]], "cut mode keeps the lead-in and the tail")
        self.assertEqual(r["words_removed"], 1)

    def test_overlapping_word_timings_are_never_cut_in_part(self):
        words = words_of([[0.5, 1.0, "a"], [0.9, 1.4, "b"], [1.3, 2.0, "c"], [2.5, 3.0, "d"]])
        r = derive_ranges(words, [[3, 3]], "cut", 4.0)
        self.assertEqual(r["remove"], [[2.2, 4.0]])
        with self.assertRaisesRegex(WordRangeError, "overlaps a kept word"):
            derive_ranges(words, [[0, 1]], "keep", 4.0)   # c (cut) overlaps b (kept) in time
        with self.assertRaisesRegex(WordRangeError, "overlaps a kept word"):
            derive_ranges(words, [[1, 1]], "cut", 4.0)   # b overlaps both a and c: it cannot be cut whole, and it is never cut in part

    def test_speech_without_word_timings_is_never_removed(self):
        words = words_of([[0.5, 0.8, "Hello"], [2.0, 2.3, "world"]])
        with self.assertRaisesRegex(WordRangeError, "no word timings"):
            derive_ranges(words, [[0, 0]], "cut", 3.0, untimed=[[0.9, 1.6]])
        with self.assertRaisesRegex(WordRangeError, "no word timings"):
            derive_ranges(words, [[1, 1]], "cut", 3.0, untimed=[[0.9, 1.6]])   # the gap clamp does not see untimed speech: refused, not cut into
        self.assertEqual(derive_ranges(words, [[1, 1]], "cut", 3.0, untimed=[[0.1, 0.3]])["remove"], [[1.0, 3.0]])

    def test_removed_words_carry_their_text(self):
        r = derive_ranges(words_of(EN), [[1, 1], [5, 6]], "cut", 6.0)
        self.assertEqual(r["removed_words"], [[1, 1, "um"], [5, 6, "a test"]])

    def test_media_bounds_clamp_the_rolls(self):
        words = words_of([[0.05, 0.5, "a"], [0.6, 0.95, "b"]])
        r = derive_ranges(words, [[0, 1]], "keep", 1.0)
        self.assertEqual(r["keep"], [[0.0, 1.0]])
        self.assertEqual(r["remove"], [])

    def test_keep_mode_removes_lead_in_gaps_and_tail(self):
        r = derive_ranges(self.words, [[0, 0], [2, 6]], "keep", 6.0)
        self.assertEqual(r["keep"], [[0.4, 0.85], [1.25, 4.8]])
        self.assertEqual(r["remove"], [[0.0, 0.4], [0.85, 1.25], [4.8, 6.0]])

    def test_cut_mode_at_the_first_or_last_word_cuts_to_the_media_edge(self):
        r = derive_ranges(self.words, [[0, 1]], "cut", 6.0)
        self.assertEqual(r["remove"], [[0.0, 1.25]])
        r = derive_ranges(self.words, [[5, 6]], "cut", 6.0)
        self.assertEqual(r["remove"], [[3.35, 6.0]])

    def test_adjacent_ranges_form_one_run(self):
        a = derive_ranges(self.words, [[0, 2], [3, 6]], "keep", 6.0)
        b = derive_ranges(self.words, [[0, 6]], "keep", 6.0)
        self.assertEqual((a["keep"], a["kept_words"]), (b["keep"], b["kept_words"]))
        self.assertEqual(a["word_ranges"], [[0, 2], [3, 6]], "the ranges are recorded as given (sorted)")

    def test_refusals_are_never_corrected(self):
        cases = {
            "out of bounds": [[5, 7]], "negative": [[-1, 2]], "reversed": [[4, 2]], "overlapping": [[0, 3], [3, 5]],
            "seconds": [[0.5, 1.2]], "bool": [[True, 2]], "text": [["0", "2"]], "triple": [[0, 1, 2]], "empty": [],
        }
        for why, ranges in cases.items():
            with self.subTest(why), self.assertRaises(WordRangeError):
                derive_ranges(self.words, ranges, "cut", 6.0)
        with self.assertRaisesRegex(WordRangeError, "keep no word"):
            derive_ranges(self.words, [[0, 6]], "cut", 6.0)
        with self.assertRaisesRegex(WordRangeError, "duration"):
            derive_ranges(self.words, [[1, 1]], "cut", None)
        with self.assertRaisesRegex(WordRangeError, "after the media"):
            derive_ranges(self.words, [[0, 6]], "keep", 4.0)
        untimed = [dict(w) for w in self.words]
        untimed[2]["start"] = None
        with self.assertRaisesRegex(WordRangeError, "no start / end"):
            derive_ranges(untimed, [[1, 1]], "cut", 6.0)
        with self.assertRaises(WordRangeError):
            derive_ranges(self.words, [[1, 1]], "cut", 6.0, pre_roll=2.0)

    def test_a_word_without_times_inside_a_run_does_not_matter(self):
        w = [dict(x) for x in self.words]
        w[4]["start"] = w[4]["end"] = None
        self.assertEqual(derive_ranges(w, [[1, 1]], "cut", 6.0)["remove"], [[0.85, 1.25]])

    def test_derived_ranges_that_overlap_are_refused(self):
        words = words_of([[0.5, 2.0, "a"], [0.6, 0.7, "b"], [0.8, 1.0, "c"]])   # timings that run into each other
        with self.assertRaisesRegex(WordRangeError, "overlap"):
            derive_ranges(words, [[1, 1]], "cut", 3.0)

    def test_deterministic_and_order_independent(self):
        a = derive_ranges(self.words, [[3, 4], [0, 1]], "keep", 6.0)
        b = derive_ranges(self.words, [[0, 1], [3, 4]], "keep", 6.0)
        self.assertEqual(a, b)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(derive_ranges(words_of(EN), [[0, 1], [3, 4]], "keep", 6.0), sort_keys=True))

    def test_interpolated_timings_at_a_boundary_are_reported(self):
        w = [dict(x) for x in self.words]
        w[2]["interpolated"] = True
        self.assertEqual(derive_ranges(w, [[1, 1]], "cut", 6.0)["interpolated_boundaries"], [2])

    def test_japanese_words(self):
        words, _ = number_words(segs(JA))
        r = derive_ranges(words, [[2, 2]], "cut", 4.0)   # えー between は (..0.8) and お集まり (1.4..)
        self.assertEqual(r["remove"], [[0.85, 1.35]])

    def test_recommendation_params_name_word_ranges_only(self):
        self.assertEqual(recommendation_ranges({"cut_words": [[1, 2]]}), ("cut", [[1, 2]]))
        self.assertEqual(recommendation_ranges({"keep_words": [[0, 2]]}), ("keep", [[0, 2]]))
        for bad in ({"cut_words": [[1, 2]], "start": 0.5}, {"keep_words": [], "cut_words": []}, {}, {"remove": [[0.5, 1.0]]}, [[1, 2]]):
            with self.subTest(bad=bad), self.assertRaises(WordRangeError):
                recommendation_ranges(bad)


class RequirementTests(unittest.TestCase):
    def test_switch_rolls_and_refusals(self):
        self.assertIsNone(parse_word_cut_requirements(reqmap()))
        r = parse_word_cut_requirements(reqmap(edit__word_cut=True))
        self.assertEqual((r["pre_roll"], r["post_roll"], r["pre_roll_provenance"]), (0.1, 0.2, "DEFAULT"))
        r = parse_word_cut_requirements(reqmap(edit__word_cut="true", edit__word_cut__pre_roll="0.05", edit__word_cut__post_roll=0.3))
        self.assertEqual((r["pre_roll"], r["post_roll"], r["post_roll_provenance"]), (0.05, 0.3, "USER"))
        with self.assertRaises(EditRequirementError):
            parse_word_cut_requirements(reqmap(edit__word_cut__pre_roll=0.1))
        with self.assertRaises(EditRequirementError):
            parse_word_cut_requirements(reqmap(edit__word_cut=True, edit__word_cut__post_roll=1.5))
        with self.assertRaises(EditRequirementError):
            parse_word_cut_requirements(reqmap(edit__word_cut="maybe"))

    def test_the_set_gate_knows_the_keys(self):
        from video_agent.service import _check_requirement_keys
        _check_requirement_keys({"edit.word_cut": True, "edit.word_cut.pre_roll": 0.05})
        with self.assertRaises(ValueError):
            _check_requirement_keys({"edit.word_cut": True, "edit.word_cut.roll": 0.05})
        with self.assertRaises(ValueError):
            _check_requirement_keys({"edit.word_cut.post_roll": 0.05})


class _WordRanges(AIProvider):
    """Recommends word ranges for the first transcript it is shown (or several / bad ones on request)."""
    name = "fake"
    model = "fake-words-1"

    def __init__(self, params, copies=1):
        self.params, self.copies, self.requests = params, copies, []

    def available(self):
        return True

    def complete(self, request: AIRequest, timeout=None) -> AIResponse:
        self.requests.append(request)
        tr = next(o for o in request.inputs["observations"] if o["kind"] == "transcript")
        recs = [{"intent": "word_cut", "asset_id": tr["asset_id"], "statement": "drop the hesitation", "confidence": 0.8, "evidence": [tr["id"]], "params": self.params}] * self.copies
        return AIResponse(task_type=request.task_type, result={"recommendations": recs}, confidence=0.8, provider=self.name, model=self.model, evidence=[], reasoning="",
                          usage=AIUsage(input_tokens=1, output_tokens=1), latency_s=0.0, metadata={})


class PipelineTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.src = fake_media(self.tmp, "talk.mp4", duration=6.0, segments=[[0.5, 1.7, "Hello um world"], [2.5, 4.6, "this is a test"]], words=EN,
                              silences=[[0.0, 0.4], [4.8, None]])

    def plan(self, provider, reqs=None, profile="youtube"):
        svc = pipeline_service(self.tmp)
        svc.provider = provider
        ir = svc.plan([self.src], profile, user_requirements=dict({"edit.word_cut": True}, **(reqs or {})), params={"language": "en"})
        path = str(Path(svc.workspace) / "plans" / "p.project.json")
        save_ir(ir, path)
        return svc, load_ir(path), path

    @staticmethod
    def word_cuts(doc):
        return [d for d in doc["decisions"] if d["subject"] == "words.cut"]

    @staticmethod
    def trim(doc):
        return next(s for s in doc["plan"]["steps"] if s["id"].startswith("step_trim_"))

    def test_the_ai_sees_numbered_words_and_its_ranges_become_a_confirm_cut(self):
        prov = _WordRanges({"cut_words": [[1, 1]]})
        svc, ir, _ = self.plan(prov)
        ctx = prov.requests[0].context["word_cut"]
        self.assertEqual(ctx["transcripts"][0]["words"][:3], [[0, "Hello"], [1, "um"], [2, "world"]], "indices and text, no times")
        self.assertEqual(ctx["rule"], RULE_VERSION)
        self.assertIn("never give seconds", ctx["instructions"].lower())
        self.assertEqual(svc.validate(ir).errors, [])
        (d,) = self.word_cuts(ir.doc)
        self.assertEqual((d["type"], d["approval"], d["risk"], d["provenance"], d["status"]), ("TRANSFORM", "CONFIRM", "HIGH", "AI_GENERATED", "PROPOSED"))
        p = d["params"]
        self.assertEqual((p["mode"], p["word_ranges"], p["remove"], p["rule"], p["index_source"]), ("cut", [[1, 1]], [[0.85, 1.25]], RULE_VERSION, "derived"))
        self.assertEqual((p["pre_roll"], p["post_roll"]), (0.1, 0.2))
        st = self.trim(ir.doc)
        self.assertIn(d["id"], st["decision_ids"])
        self.assertIn([0.85, 1.25], st["params"]["removed"])
        self.assertNotIn([0.85, 1.25], [k for k in st["params"]["keep"]])
        self.assertTrue(all(not (a < 1.25 and b > 0.85) for a, b in st["params"]["keep"]), "no kept range covers the cut")
        self.assertEqual(ir.doc["plan"]["status"], "REVIEW", "the AI's cut waits for a person")

    def test_render_waits_and_a_rejected_cut_is_not_executed(self):
        svc, ir, path = self.plan(_WordRanges({"cut_words": [[1, 1]]}))
        out = svc.render(ir, path)
        self.assertEqual(out["status"], "WAITING_FOR_APPROVAL")
        did = self.word_cuts(ir.doc)[0]["id"]
        svc.reject(load_ir(path), path, [did], reason="keep the um")
        rev = svc.revise(load_ir(path), path)
        doc = load_ir(path).doc
        self.assertTrue(rev["created"])
        self.assertFalse([d for d in self.word_cuts(doc) if d["status"] != "REJECTED"])
        self.assertFalse(any([0.85, 1.25] == r for s in doc["plan"]["steps"] for r in (s.get("params") or {}).get("removed") or []))

    def test_revise_reuses_the_recorded_ranges_and_derives_the_same_seconds(self):
        prov = _WordRanges({"keep_words": [[0, 0], [2, 6]]})
        svc, ir, path = self.plan(prov)
        first = self.word_cuts(ir.doc)[0]["params"]
        calls = len(ir.doc["provenance"]["ai_calls"])
        svc.revise(load_ir(path), path, feedback="same again")
        doc = load_ir(path).doc
        again = self.word_cuts(doc)[0]["params"]
        self.assertEqual((again["keep"], again["remove"], again["word_ranges"]), (first["keep"], first["remove"], first["word_ranges"]))
        self.assertEqual(len(prov.requests), 1, "a revision spends no AI call")
        self.assertEqual(len(doc["provenance"]["ai_calls"]), calls)
        svc.revise(load_ir(path), path, user_requirements={"edit.word_cut": True, "edit.word_cut.pre_roll": 0.0})
        third = self.word_cuts(load_ir(path).doc)[0]["params"]
        self.assertEqual(third["word_ranges"], first["word_ranges"])
        self.assertEqual(third["keep"][0], [0.5, 0.85], "a new roll derives new seconds from the same word ranges")
        self.assertEqual(third["pre_roll_provenance"], "USER")

    def test_tampered_seconds_or_rule_are_validation_errors(self):
        svc, ir, _ = self.plan(_WordRanges({"cut_words": [[1, 1]]}))
        d = self.word_cuts(ir.doc)[0]
        d["params"]["remove"] = [[0.7, 1.3]]
        self.assertTrue(any("differ" in e for e in check_word_cuts(ir.doc)))
        self.assertTrue(any("differ" in e for e in svc.validate(ir).errors))
        d["params"]["rule"] = "word_ranges@0"
        self.assertTrue(any("not this agent's" in e for e in check_word_cuts(ir.doc)))

    def test_invalid_ranges_are_refused_and_nothing_is_cut(self):
        for params, why in (({"cut_words": [[3, 1]]}, "reversed"), ({"cut_words": [[0, 9]]}, "out of bounds"), ({"cut_words": [[0, 2], [2, 3]]}, "overlap"),
                            ({"cut_words": [[1, 1]], "start": 0.9, "end": 1.2}, "never times")):
            with self.subTest(why=why):
                svc, ir, _ = self.plan(_WordRanges(params))
                (d,) = self.word_cuts(ir.doc)
                self.assertEqual((d["type"], d["approval"]), ("KEEP", "AUTO"))
                self.assertIn(why, d["params"]["refused"])
                self.assertFalse(any(x["subject"] == "ai.word_cut" for x in ir.doc["decisions"]), "the refusal is the record, not a second review item")
                st = self.trim(ir.doc)
                self.assertNotIn(d["id"], st["decision_ids"])
                self.assertEqual(svc.validate(ir).errors, [])

    def test_two_recommendations_for_one_asset_are_not_guessed_between(self):
        _, ir, _ = self.plan(_WordRanges({"cut_words": [[1, 1]]}, copies=2))
        (d,) = self.word_cuts(ir.doc)
        self.assertEqual(d["type"], "KEEP")
        self.assertIn("not guessed", d["params"]["refused"])

    def test_without_a_provider_the_requirement_is_recorded_and_nothing_is_cut(self):
        from video_agent.providers import NullProvider
        svc, ir, _ = self.plan(NullProvider())
        (d,) = self.word_cuts(ir.doc)
        self.assertEqual((d["type"], d["approval"]), ("KEEP", "AUTO"))
        self.assertIn("no AI word-range recommendation", d["decision"])
        self.assertEqual(svc.validate(ir).errors, [])

    def test_a_kept_range_outside_the_other_cuts_is_refused(self):
        _, ir, _ = self.plan(_WordRanges({"keep_words": [[0, 0]]}), reqs={"edit.trim": "2-5"})
        (d,) = self.word_cuts(ir.doc)
        self.assertEqual(d["type"], "KEEP")
        self.assertIn("outside the range the other cuts keep", d["params"]["refused"])

    def test_words_the_ai_was_not_shown_are_not_decided_about(self):
        from video_agent.agent import word_cut
        old = word_cut.MAX_CONTEXT_WORDS
        word_cut.MAX_CONTEXT_WORDS = 3
        try:
            for params in ({"keep_words": [[0, 2]]}, {"cut_words": [[4, 4]]}):
                with self.subTest(params=params):
                    prov = _WordRanges(params)
                    _, ir, _ = self.plan(prov)
                    self.assertEqual(prov.requests[0].context["word_cut"]["transcripts"][0]["truncated_after"], 3)
                    (d,) = self.word_cuts(ir.doc)
                    self.assertEqual(d["type"], "KEEP")
                    self.assertIn("shown only words", d["params"]["refused"])
            _, ir, _ = self.plan(_WordRanges({"cut_words": [[1, 1]]}))
            self.assertEqual(self.word_cuts(ir.doc)[0]["type"], "TRANSFORM", "a cut among the words it saw stands")
        finally:
            word_cut.MAX_CONTEXT_WORDS = old

    def test_the_trim_is_frame_accurate_and_the_validator_reads_the_executed_keep(self):
        svc, ir, _ = self.plan(_WordRanges({"cut_words": [[1, 1]]}))
        self.assertTrue(self.trim(ir.doc)["params"]["accurate"])
        op = next(o for o in ir.doc["video"]["operations"] if o["type"] == "video.trim")
        op["keep"] = [[0.0, 4.95]]   # an IR whose executed trim keeps the cut words
        self.assertTrue(any("keeps" in e for e in check_word_cuts(ir.doc)))

    def test_conference_policy_blocks_an_ai_chosen_deletion(self):
        _, ir, _ = self.plan(_WordRanges({"cut_words": [[1, 1]]}), profile="conference")
        d = next(x for x in self.word_cuts(ir.doc) if x["type"] == "TRANSFORM")
        self.assertEqual((d["approval"], d["status"]), ("BLOCK", "BLOCKED"))
        self.assertFalse([x for x in ir.doc["decisions"] if x["subject"] == "ai.word_cut"], "the blocked cut is the only record of the recommendation")

    def test_without_the_requirement_a_word_cut_recommendation_is_dropped(self):
        svc = pipeline_service(self.tmp)
        svc.provider = _WordRanges({"cut_words": [[1, 1]]})
        ir = svc.plan([self.src], "youtube", user_requirements={}, kinds=["transcript"], params={"language": "en", "word_timestamps": True})
        self.assertFalse(self.word_cuts(ir.doc))
        self.assertTrue(any("word_cut" in w and "not a registered" in w for w in ir.doc["analysis"]["warnings"]))


if __name__ == "__main__":
    unittest.main()
