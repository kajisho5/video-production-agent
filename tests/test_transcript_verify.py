"""Post-render transcript check (ADR-110): `verify.transcript=true` re-transcribes each deliverable and aligns the heard words to the
words the plan kept → a "listen and check here" list (missing, duplicated, unexpected, low confidence) with delivered and source
times. It never fails a job, changes a QA status or edits anything. Fake engine + fake transcription; no media is decoded."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import clear_modes, fake_media, pipeline_service  # noqa: E402
from video_agent.agent.transcript_verify import (MIN_CONFIDENCE_DEFAULT, RULE_VERSION, compare, normalise, parse_verify_requirements, planned_words,  # noqa: E402
                                                 timeline_of, to_source, tokens)
from video_agent.project import load_ir, save_ir  # noqa: E402

EN = [[0.5, 0.8, "Hello"], [0.9, 1.2, "um"], [1.3, 1.7, "world"], [2.5, 2.9, "this"], [3.0, 3.3, "is"], [3.4, 3.9, "a"], [4.0, 4.6, "test."]]


def w(rows, conf=None):
    out = []
    for i, r in enumerate(rows):
        d = {"index": i, "start": r[0], "end": r[1], "text": r[2]}
        if len(r) > 3:
            d["confidence"] = r[3]
        elif conf is not None:
            d["confidence"] = conf
        out.append(d)
    return out


def planned(rows):
    return [dict(x, source={"asset_id": "a", "index": x["index"], "start": x["start"], "end": x["end"]}) for x in w(rows)]


class NormaliseAndTokenTests(unittest.TestCase):
    def test_nfkc_casefold_punctuation_and_spaces(self):
        self.assertEqual(normalise("Ｈｅｌｌｏ,"), "hello")
        self.assertEqual(normalise("World!"), "world")
        self.assertEqual(normalise("「ありがとう。」"), "ありがとう")
        self.assertEqual(normalise("ｶﾀｶﾅ"), "カタカナ", "half-width kana are folded by NFKC")
        self.assertEqual(normalise("Straße"), "strasse", "case folding, not lower()")
        self.assertEqual(normalise(" — "), "")

    def test_word_and_char_units(self):
        ws = w([[0, 1, "本日は"], [1, 2, "、"], [2, 3, "晴天"]])
        self.assertEqual([t["norm"] for t in tokens(ws)], ["本日は", "晴天"], "a punctuation-only word is left out")
        self.assertEqual([t["norm"] for t in tokens(ws, "char")], ["本", "日", "は", "晴", "天"])


class CompareTests(unittest.TestCase):
    def test_identical_speech_is_clean(self):
        res = compare(planned(EN), w(EN, conf=0.9))
        self.assertEqual(res["items"], [])
        self.assertEqual(res["matched"], 7)

    def test_missing_duplicated_unexpected_and_low_confidence_in_english(self):
        heard = w([[0.5, 0.8, "hello"], [0.9, 1.2, "um", 0.9], [2.5, 2.9, "this", 0.9], [2.9, 3.2, "this", 0.9], [3.3, 3.6, "is", 0.9], [3.7, 3.9, "banana", 0.9],
                   [4.0, 4.3, "a", 0.3], [4.3, 4.9, "test", 0.2]], conf=0.9)
        res = compare(planned(EN), heard)
        kinds = [(i["kind"], i["text"]) for i in res["items"]]
        self.assertIn(("missing", "world"), kinds)
        self.assertIn(("duplicated", "this"), kinds)
        self.assertIn(("unexpected", "banana"), kinds)
        self.assertIn(("low_confidence", "a test"), kinds, "consecutive low-confidence words are one span")
        miss = next(i for i in res["items"] if i["kind"] == "missing")
        self.assertEqual((miss["output"], miss["source"]["word_indices"], miss["source"]["start"]), ({"start": 1.3, "end": 1.7}, [2], 1.3))
        self.assertEqual(res["counts"], {"missing": 1, "duplicated": 1, "unexpected": 1, "low_confidence": 1})
        self.assertEqual([i["output"]["start"] for i in res["items"]], sorted(i["output"]["start"] for i in res["items"]), "items in delivered-time order")

    def test_a_word_heard_as_often_as_planned_is_unexpected_not_duplicated(self):
        res = compare(planned([[0, 1, "a"], [1, 2, "b"]]), w([[0, 1, "b"], [1, 2, "a"]], conf=0.9))
        self.assertNotIn("duplicated", {i["kind"] for i in res["items"]})

    def test_japanese_word_and_char_alignment(self):
        plan = planned([[0.4, 0.7, "本日"], [0.7, 0.8, "は"], [1.4, 1.9, "お集まり"], [1.9, 2.4, "いただき"], [2.4, 3.2, "ありがとうございます"]])
        heard = w([[0.4, 0.8, "本日は"], [1.4, 2.4, "お集まりいただき"], [2.4, 3.2, "ありがとうございます。"]], conf=0.9)
        by_word = compare(plan, heard, "word")
        self.assertTrue(by_word["items"], "two recognitions split the same Japanese speech into different words")
        by_char = compare(plan, heard, "char")
        self.assertEqual(by_char["items"], [], "character alignment (opt-in) sees the same speech")
        heard2 = w([[0.4, 0.8, "本日は"], [2.4, 3.2, "ありがとうございます"]], conf=0.9)
        res = compare(plan, heard2, "char")
        (miss,) = [i for i in res["items"] if i["kind"] == "missing"]
        self.assertEqual((miss["text"], miss["source"]["word_indices"]), ("お集まりいただき", [2, 3]))
        self.assertEqual(miss["output"], {"start": 1.4, "end": 2.4})

    def test_deterministic(self):
        heard = w([[0.5, 0.8, "hello"], [2.5, 2.9, "this"]], conf=0.9)
        self.assertEqual(json.dumps(compare(planned(EN), heard), sort_keys=True), json.dumps(compare(planned(EN), heard), sort_keys=True))


class TimelineTests(unittest.TestCase):
    def doc(self, video_ops=(), audio_ops=()):
        return {"assets": {"a": {"technical": {"duration": 10.0}}, "b": {"technical": {"duration": 6.0}}},
                "video": {"operations": list(video_ops)}, "audio": {"operations": list(audio_ops)}}

    def test_trim_concat_speed_and_a_still_before(self):
        ops = [{"type": "video.trim", "asset": "a", "keep": [[1.0, 3.0], [5.0, 9.0]]}, {"type": "video.trim", "asset": "b", "keep": [[0.0, 6.0]]},
               {"type": "video.concat", "output": "programme", "segments": [{"input": "a", "timeline_range": [0.0, 6.0]}, {"input": "b", "timeline_range": [6.0, 12.0]}]},
               {"type": "video.speed", "asset": "programme", "factor": 2.0}, {"type": "video.image_insert", "asset": "programme", "position": "before", "duration": 1.0}]
        doc = self.doc(ops)
        tl = timeline_of(doc, {"id": "programme", "sources": ["a", "b"]})
        self.assertEqual((tl["speed"], tl["lead"], tl["inputs"]["b"]["offset"]), (2.0, 1.0, 6.0))
        self.assertEqual(to_source(1.0 + 3.5 / 2.0, tl), {"asset_id": "a", "seconds": 6.5})
        self.assertEqual(to_source(1.0 + 7.0 / 2.0, tl), {"asset_id": "b", "seconds": 1.0})
        self.assertIsNone(to_source(0.5, tl), "inside the still")
        obs = [{"id": "ta", "kind": "transcript", "asset_id": "a", "data": {"segments": [{"id": "s", "words": [{"start": 1.5, "end": 2.0, "text": "kept"},
                                                                                                                     {"start": 3.2, "end": 3.8, "text": "cut"},
                                                                                                                     {"start": 2.8, "end": 3.4, "text": "straddles"},
                                                                                                                     {"start": 6.0, "end": 6.5, "text": "late"}]}]}},
               {"id": "tb", "kind": "transcript", "asset_id": "b", "data": {"segments": [{"id": "s", "words": [{"start": 0.5, "end": 1.0, "text": "bee"}]}]}}]
        words, _, problems = planned_words(doc, {"id": "programme", "sources": ["a", "b"]}, obs)
        self.assertEqual(problems, [])
        self.assertEqual([(x["text"], x["start"], x["end"], x["source"]["index"]) for x in words],
                         [("kept", 1.25, 1.5, 0), ("late", 2.5, 2.75, 3), ("bee", 4.25, 4.5, 0)])

    def test_audio_cut_and_a_source_without_word_timings(self):
        doc = self.doc(audio_ops=[{"type": "audio.cut", "asset": "a", "params": {"remove": [[0.0, 2.0]]}}])
        tl = timeline_of(doc, {"id": "a", "sources": ["a"]})
        self.assertEqual(tl["inputs"]["a"]["keep"], [[2.0, 10.0]])
        _, _, problems = planned_words(doc, {"id": "a", "sources": ["a"]}, [{"id": "t", "kind": "transcript", "asset_id": "a", "data": {"segments": [{"id": "s", "words": None}]}}])
        self.assertTrue(problems and "no transcript with word timings" in problems[0])


class RequirementTests(unittest.TestCase):
    def test_keys_and_refusals(self):
        self.assertIsNone(parse_verify_requirements({}))
        self.assertIsNone(parse_verify_requirements({"verify.transcript": False}))
        cfg = parse_verify_requirements({"verify.transcript": "true"})
        self.assertEqual(cfg, {"enabled": True, "rule": RULE_VERSION, "unit": "word", "min_confidence": MIN_CONFIDENCE_DEFAULT}, "off by default; international defaults")
        cfg = parse_verify_requirements({"verify.transcript": True, "verify.transcript.language": "JA", "verify.transcript.unit": "char", "verify.transcript.min_confidence": "0.6",
                                         "verify.transcript.engine": "faster_whisper", "verify.transcript.model": "small"})
        self.assertEqual((cfg["language"], cfg["unit"], cfg["min_confidence"], cfg["engine"], cfg["model"]), ("ja", "char", 0.6, "faster_whisper", "small"))
        for bad in ({"verify.transcript.unit": "char"}, {"verify.transcript": "maybe"}, {"verify.transcript": True, "verify.transcript.unit": "syllable"},
                    {"verify.transcript": True, "verify.transcript.min_confidence": 2}, {"verify.transcript": True, "verify.transcript.language": "japanese"},
                    {"verify.transcript": True, "verify.transcript.model": "../m"}, {"verify.transcripts": True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_verify_requirements(bad)

    def test_the_set_gate_knows_the_prefix(self):
        from video_agent.service import _check_requirement_keys
        _check_requirement_keys({"verify.transcript": True})
        with self.assertRaises(ValueError):
            _check_requirement_keys({"verify.other": True})


class RenderTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.src = fake_media(self.tmp, "talk.mp4", duration=6.0, segments=[[0.5, 1.7, "Hello um world"], [2.5, 4.6, "this is a test."]], words=EN,
                              silences=[[0.0, 0.4], [4.8, None]])

    def tearDown(self):
        clear_modes()

    def render(self, reqs, heard=None, svc_render=None, mode=None):
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self.src], "youtube", user_requirements=reqs, params={"language": "en"})
        path = str(Path(svc.workspace) / "plans" / "p.project.json")
        save_ir(ir, path)
        if heard is not None:
            os.environ["FAKE_TS_RENDERED"] = json.dumps(heard)
        if mode:
            os.environ["FAKE_TS_MODE"] = mode
        out = (svc_render or svc).render(load_ir(path), path, approve=["all"])
        return out, load_ir(path).doc, path

    HEARD = {"segments": [[0.5, 4.9, "hello um this this is banana a test"]],
             "words": [[0.5, 0.8, "hello"], [0.9, 1.2, "um"], [2.5, 2.9, "this"], [2.9, 3.2, "this"], [3.3, 3.6, "is"], [3.6, 3.9, "banana"], [4.0, 4.3, "a"],
                       [4.3, 4.9, "test", 0.2]]}

    def test_off_by_default(self):
        out, doc, _ = self.render({}, heard=self.HEARD)
        self.assertNotIn("transcript_verify", doc["qa"])
        self.assertNotIn("transcript_verification", out["qa"])

    def test_the_list_of_places_to_listen_to_is_in_the_report_and_never_a_verdict(self):
        base, _, _ = self.render({}, heard=self.HEARD)
        out, doc, path = self.render({"verify.transcript": True}, heard=self.HEARD)
        self.assertEqual(doc["qa"]["transcript_verify"]["rule"], RULE_VERSION)
        tv = out["qa"]["transcript_verification"]
        self.assertEqual(tv["status"], "CHECK_HERE")
        self.assertEqual(tv["counts"], {"missing": 1, "duplicated": 1, "unexpected": 1, "low_confidence": 1})
        (dv,) = tv["deliverables"]
        miss = next(i for i in dv["items"] if i["kind"] == "missing")
        self.assertEqual((miss["text"], miss["output"]["start"], miss["source"]["word_indices"]), ("world", 1.3, [2]))
        unexp = next(i for i in dv["items"] if i["kind"] == "unexpected")
        self.assertEqual(unexp["source"]["start"], 3.6, "a delivered time inside a kept range maps back onto the source")
        self.assertEqual((dv["language"], dv["engine"]), ("en", "faster_whisper"))
        self.assertEqual(tv["measurements"][0]["parameters"]["word_timestamps"], True)
        self.assertEqual(out["status"], base["status"], "the check never changes the job's outcome")
        self.assertEqual(out["qa"]["status"], base["qa"]["status"])
        self.assertEqual([a["stage"] for a in out["artifacts"]], [a["stage"] for a in base["artifacts"]])
        self.assertEqual(doc["provenance"]["runs"][-1]["transcript_verification"]["status"], "CHECK_HERE")
        prov = json.loads((Path(out["report"]).parent / "provenance.json").read_text(encoding="utf-8"))
        self.assertTrue(prov["transcript_verification"]["measurements"][0]["ok"])
        md = Path(out["report"]).read_text(encoding="utf-8")
        self.assertIn("Listen and check", md)
        self.assertIn("missing at 1.30-1.70s: 'world'", md)

    def test_clean_when_everything_planned_is_heard(self):
        heard = {"segments": [[0.5, 4.6, "x"]], "words": [list(r) for r in EN]}
        out, _, _ = self.render({"verify.transcript": True}, heard=heard)
        self.assertEqual(out["qa"]["transcript_verification"]["status"], "CLEAN")

    def test_without_the_transcription_skill_a_warning_and_a_provenance_row(self):
        svc2 = pipeline_service(self.tmp, transcription=False)
        out, doc, _ = self.render({"verify.transcript": True}, heard=self.HEARD, svc_render=svc2)
        tv = out["qa"]["transcript_verification"]
        self.assertEqual(tv["status"], "NOT_RUN")
        self.assertTrue(tv["warnings"] and "not re-transcribed" in tv["warnings"][0])
        self.assertEqual(tv["measurements"][0]["ok"], False)
        self.assertEqual(out["status"], "COMPLETED")

    def test_a_failing_skill_is_a_warning_not_a_crash(self):
        out, _, _ = self.render({"verify.transcript": True}, heard=self.HEARD, mode="crash")
        tv = out["qa"]["transcript_verification"]
        self.assertEqual((tv["status"], tv["measurements"][0]["ok"]), ("NOT_RUN", False))
        self.assertIn("re-transcription failed", tv["warnings"][0])
        self.assertEqual(out["status"], "COMPLETED")

    def test_another_language_is_named(self):
        out, _, _ = self.render({"verify.transcript": True, "verify.transcript.language": "ja"}, heard=self.HEARD)
        dv = out["qa"]["transcript_verification"]["deliverables"][0]
        self.assertEqual(dv["language"], "ja")
        self.assertTrue(any("recognised as 'ja'" in n for n in dv["notes"]))

    def test_cut_words_are_not_expected(self):
        from video_agent.providers import AIProvider, AIResponse, AIUsage

        class Cut(AIProvider):
            name, model = "fake", "fake-1"

            def available(self):
                return True

            def complete(self, request, timeout=None):
                tr = next(o for o in request.inputs["observations"] if o["kind"] == "transcript")
                return AIResponse(task_type=request.task_type, result={"recommendations": [{"intent": "word_cut", "asset_id": tr["asset_id"], "statement": "x", "confidence": 0.8,
                                                                                            "evidence": [tr["id"]], "params": {"cut_words": [[1, 1]]}}]},
                                  confidence=0.8, provider="fake", model="fake-1", evidence=[], reasoning="", usage=AIUsage(), latency_s=0.0, metadata={})
        svc = pipeline_service(self.tmp)
        svc.provider = Cut()
        ir = svc.plan([self.src], "youtube", user_requirements={"edit.word_cut": True, "verify.transcript": True}, params={"language": "en"})
        path = str(Path(svc.workspace) / "plans" / "c.project.json")
        save_ir(ir, path)
        # the delivered file says everything but "um", 0.4 s earlier from "world" on (the cut 0.85-1.25 is gone)
        heard = {"segments": [[0.5, 4.2, "x"]], "words": [[0.5, 0.8, "Hello"], [0.9, 1.3, "world"], [2.1, 2.5, "this"], [2.6, 2.9, "is"], [3.0, 3.5, "a"], [3.6, 4.2, "test"]]}
        os.environ["FAKE_TS_RENDERED"] = json.dumps(heard)
        out = svc.render(load_ir(path), path, approve=["all"])
        tv = out["qa"]["transcript_verification"]
        self.assertEqual(tv["status"], "CLEAN", tv)


if __name__ == "__main__":
    unittest.main()
