"""Chapters from the transcript (ADR-137, roadmap step 26): the AI names the segments where the topic changes, the agent takes their starts from
the transcript (rule transcript_chapters@1), the chapters wait for a person (CONFIRM), go onto the programme timeline through the trim and are
written into the exported file by ffmpeg-skill metadata; with a YouTube publish they become the description's timestamps. Fake engine + fake
transcription + a fake provider; no media is decoded."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import _read_fake  # noqa: E402
from pipeline_harness import clear_modes, fake_media, pipeline_service  # noqa: E402
from video_agent.agent import chapters as ch  # noqa: E402
from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.requirements import requirement_map  # noqa: E402
from video_agent.models import Operation, Requirement  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.providers import AIProvider, AIRequest, AIResponse, AIUsage  # noqa: E402
from video_agent.tools.base import ToolError  # noqa: E402
from video_agent.tools.ffmpeg_skill.adapter import FfmpegSkillAdapter, PathPolicy  # noqa: E402

# three topics in a 90 s talk: the opening (0-30), the venue's audio (30-60), questions (60-90)
SEGMENTS = [[0.4, 9.5, "Good morning everyone and welcome to the session"], [10.0, 28.0, "Today we look at how the hall was built"],
            [31.0, 44.0, "Let us turn to the sound system of the venue"], [45.0, 58.0, "Each speaker has its own delay line"],
            [61.0, 74.0, "Now we take questions from the floor"], [75.0, 89.0, "Thank you all for coming today"]]
THREE = {"chapters": [{"segment": 0, "title": "Welcome"}, {"segment": 2, "title": "The sound system"}, {"segment": 4, "title": "Questions"}]}


def reqmap(**kv):
    return requirement_map([Requirement(key=k.replace("__", "."), value=v, provenance="USER", source="cli") for k, v in kv.items()])


class _Chapters(AIProvider):
    """Recommends chapters for the first transcript it is shown (or several / bad ones on request)."""
    name = "fake"
    model = "fake-chapters-1"

    def __init__(self, params, copies=1):
        self.params, self.copies, self.requests = params, copies, []

    def available(self):
        return True

    def complete(self, request: AIRequest, timeout=None) -> AIResponse:
        self.requests.append(request)
        tr = next(o for o in request.inputs["observations"] if o["kind"] == "transcript")
        recs = [{"intent": "transcript_chapters", "asset_id": tr["asset_id"], "statement": "three topics", "confidence": 0.8, "evidence": [tr["id"]], "params": self.params}] * self.copies
        return AIResponse(task_type=request.task_type, result={"recommendations": recs}, confidence=0.8, provider=self.name, model=self.model, evidence=[], reasoning="",
                          usage=AIUsage(input_tokens=1, output_tokens=1), latency_s=0.0, metadata={})


SEGS = ch.numbered_segments([{"id": f"seg_{i:04d}", "start": a, "end": b, "text": t} for i, (a, b, t) in enumerate(SEGMENTS, 1)])


class RuleTests(unittest.TestCase):
    def test_three_topics_give_three_chapters_from_zero(self):
        got = ch.derive_chapters(THREE, SEGS, 90.0)
        self.assertEqual([(c["start"], c["title"]) for c in got["chapters"]], [(0.0, "Welcome"), (31.0, "The sound system"), (61.0, "Questions")])
        self.assertEqual([c["first_words"] for c in got["chapters"]], ["Good morning everyone and welcome to", "Let us turn to the sound", "Now we take questions from the"])
        self.assertEqual(got["rule"], ch.RULE_VERSION)

    def test_chapter_one_starts_at_zero_whatever_segment_it_cites(self):
        got = ch.derive_chapters({"chapters": [{"segment": 1, "title": "A"}, {"segment": 4, "title": "B"}]}, SEGS, 90.0)
        self.assertEqual(got["chapters"][0]["start"], 0.0)

    def test_refusals_are_never_corrected(self):
        bad = [
            {"chapters": [{"segment": 0, "title": "only one"}]},
            {"chapters": [{"segment": 0, "title": "A"}, {"segment": 0, "title": "B"}]},
            {"chapters": [{"segment": 2, "title": "A"}, {"segment": 1, "title": "B"}]},
            {"chapters": [{"segment": 0, "title": "A"}, {"segment": 9, "title": "B"}]},
            {"chapters": [{"segment": 0, "title": "A"}, {"segment": 1, "title": "B"}]},   # chapter 1 is 10.0 s: fine; see below for 5 s
            {"chapters": [{"segment": 0, "title": "A"}, {"segment": 5, "title": "B"}], "start": 3},
            {"chapters": [{"segment": 0, "title": "A", "start": 0.0}, {"segment": 5, "title": "B"}]},
            {"chapters": [{"segment": 0, "title": ""}, {"segment": 5, "title": "B"}]},
            {"chapters": [{"segment": 0, "title": "A\nB"}, {"segment": 5, "title": "B"}]},
            {"chapters": [{"segment": 0, "title": "x" * 101}, {"segment": 5, "title": "B"}]},
            {"chapters": [{"segment": True, "title": "A"}, {"segment": 5, "title": "B"}]},
        ]
        for n, params in enumerate(bad):
            if n == 4:
                self.assertEqual(len(ch.derive_chapters(params, SEGS, 90.0)["chapters"]), 2)
                continue
            with self.assertRaises(ch.ChapterError, msg=str(params)):
                ch.derive_chapters(params, SEGS, 90.0)

    def test_a_chapter_under_the_minimum_is_refused(self):
        with self.assertRaises(ch.ChapterError) as e:
            ch.derive_chapters({"chapters": [{"segment": 0, "title": "A"}, {"segment": 5, "title": "B"}]}, SEGS, 80.0)   # the last one is 5 s
        self.assertIn("under the 10s minimum", str(e.exception))
        with self.assertRaises(ch.ChapterError):
            ch.derive_chapters(THREE, SEGS, 90.0, min_seconds=31.0)

    def test_japanese_first_words_are_characters(self):
        self.assertEqual(ch.first_words("本日は会場の音響設備についてご説明いたします。まず最初に"), "本日は会場の音響設備についてご説明いたします。まず最初に")
        self.assertTrue(ch.first_words("あ" * 60).endswith("…"))

    def test_on_the_programme_through_a_trim(self):
        rec = ch.derive_chapters(THREE, SEGS, 90.0)
        got = ch.on_programme(rec, [[0.0, 20.0], [35.0, 90.0]], 75.0)   # 20-35 removed: chapter 2 started at 31 inside the cut
        self.assertEqual([c["start"] for c in got["chapters"]], [0.0, 20.0, 46.0])
        self.assertIn("moved", got["chapters"][1])
        self.assertEqual(got["chapters"][2]["source_start"], 61.0)
        self.assertEqual(got["dropped"], [])
        short = ch.on_programme(rec, [[0.0, 64.0]], 64.0)   # questions start at 61: 3 s left on the programme
        self.assertEqual([c["title"] for c in short["chapters"]], ["Welcome", "The sound system"])
        self.assertEqual(short["dropped"][0]["title"], "Questions")
        self.assertEqual(ch.on_programme(rec, None, 90.0)["chapters"][1]["start"], 31.0)
        lead = ch.on_programme(rec, [[25.0, 90.0]], 65.0)   # chapter 2 starts 6 s in: chapter 1 takes it in and stays at 0
        self.assertEqual([(c["start"], c["title"]) for c in lead["chapters"]], [(0.0, "Welcome"), (36.0, "Questions")])
        self.assertEqual([d["title"] for d in lead["dropped"]], ["The sound system"])

    def test_timestamps_and_the_description(self):
        self.assertEqual([ch.timestamp(x) for x in (0, 9.99, 61.0, 3600, 3725.4)], ["0:00", "0:09", "1:01", "1:00:00", "1:02:05"])
        rec = ch.derive_chapters(THREE, SEGS, 90.0)["chapters"]
        desc, why = ch.with_description("Session recording.", rec)
        self.assertIsNone(why)
        self.assertEqual(desc, "Session recording.\n\n0:00 Welcome\n0:31 The sound system\n1:01 Questions")
        self.assertEqual(ch.with_description(None, rec[:2])[0], None)
        self.assertIn("at least 3", ch.with_description(None, rec[:2])[1])
        self.assertIn("already", ch.with_description("0:00 Intro\n1:00 Talk", rec)[1])
        self.assertEqual(ch.chapters_file_text(rec), "0.000 Welcome\n31.000 The sound system\n61.000 Questions\n")

    def test_requirements(self):
        self.assertIsNone(ch.parse_chapter_requirements(reqmap()))
        self.assertEqual(ch.parse_chapter_requirements(reqmap(delivery__chapters=True))["min_seconds"], 10.0)
        self.assertEqual(ch.parse_chapter_requirements(reqmap(delivery__chapters="true", delivery__chapters__min_seconds="30"))["min_seconds"], 30.0)
        for bad in ({"delivery__chapters__min_seconds": 30}, {"delivery__chapters": True, "delivery__chapters__min_seconds": 5}, {"delivery__chapters": "maybe"}):
            with self.assertRaises(EditRequirementError):
                ch.parse_chapter_requirements(reqmap(**bad))
        with self.assertRaises(ValueError):
            ch.check_keys({"delivery.chapters.count": 3})


class AdapterTests(unittest.TestCase):
    def test_the_chapter_file_is_written_by_the_adapter_and_never_on_the_command_line(self):
        text = FfmpegSkillAdapter.chapter_text("ffmpeg-skill/metadata", [{"start": 0, "title": "Welcome"}, {"start": 31.5, "title": "会場の音響"}])
        self.assertEqual(text, "0.000 Welcome\n31.500 会場の音響\n")
        for bad in ([], [{"start": 0, "title": "a\nb"}], [{"start": 5, "title": "a"}, {"start": 5, "title": "b"}], [{"start": 0, "title": "a", "x": 1}], "0 a"):
            with self.assertRaises(ToolError):
                FfmpegSkillAdapter.chapter_text("ffmpeg-skill/metadata", bad)

    def test_argv_and_the_workspace_boundary(self):
        tmp = os.path.realpath(tempfile.mkdtemp())
        root = Path(tmp) / "fs"
        (root / "scripts").mkdir(parents=True)
        (root / "scripts" / "metadata.py").write_text("", encoding="utf-8")

        class Skill:
            version = "4.20.1"
            scripts = ["metadata"]

            def __init__(self):
                self.root = root

            def script(self, n):
                return root / "scripts" / f"{n}.py"

            def catalog_check(self):
                class C:
                    def tool_reason(self, n):
                        return None
                return C()

            def offers_flag(self, s, f):
                return True
        ws = Path(tmp) / "ws"
        ws.mkdir()
        ad = FfmpegSkillAdapter(skill=Skill(), path_policy=PathPolicy([tmp], str(ws)))
        paths = {"in": str(ws / "a.mp4"), "txt": str(ws / "op" / "chapters.txt"), "out": str(ws / "b.mp4")}
        args = {"input": "in", "chapters": "txt", "chapter_list": [{"start": 0.0, "title": "A"}, {"start": 12.0, "title": "B"}], "output": "out"}
        argv = ad.build_argv("ffmpeg-skill/metadata", args, paths)
        self.assertEqual(argv, [paths["in"], "--chapters", paths["txt"], "-o", paths["out"]])
        self.assertEqual(ad.chapter_document("ffmpeg-skill/metadata", args, paths), (paths["txt"], "0.000 A\n12.000 B\n"))
        with self.assertRaises(ToolError):   # the chapter file lands inside the workspace only
            ad.chapter_document("ffmpeg-skill/metadata", dict(args, chapters=str(Path(tmp) / "elsewhere.txt")), paths)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.src = fake_media(self.tmp, "talk.mp4", duration=90.0, segments=SEGMENTS, silences=[])

    def plan(self, provider, reqs=None, profile="youtube"):
        svc = pipeline_service(self.tmp)
        svc.provider = provider
        ir = svc.plan([self.src], profile, user_requirements=dict({"delivery.chapters": True}, **(reqs or {})), params={"language": "en"})
        path = str(Path(svc.workspace) / "plans" / "p.project.json")
        save_ir(ir, path)
        return svc, load_ir(path), path

    @staticmethod
    def decision(doc):
        return [d for d in doc["decisions"] if d["subject"] == ch.SUBJECT]

    def test_fixture_with_three_topics_gives_three_chapters_from_zero_each_at_least_10s(self):
        prov = _Chapters(THREE)
        svc, ir, path = self.plan(prov)
        ctx = prov.requests[0].context["chapters"]
        self.assertEqual(ctx["transcripts"][0]["segments"][2], [2, 31.0, "Let us turn to the sound system of the venue"], "index, start, text")
        self.assertIn("never give seconds", ctx["instructions"].lower())
        self.assertIn("transcript_chapters", prov.requests[0].context["allowed_intents"])
        self.assertEqual(svc.validate(ir).errors, [])
        (d,) = self.decision(ir.doc)
        self.assertEqual((d["type"], d["approval"], d["provenance"], d["status"]), ("TRANSFORM", "CONFIRM", "AI_GENERATED", "PROPOSED"))
        self.assertIn("“Let us turn to the sound”", d["decision"], "the first words are quoted as evidence")
        rec = ir.doc["delivery"]["chapters"]
        self.assertEqual([c["title"] for c in rec["chapters"]], ["Welcome", "The sound system", "Questions"])
        starts = [c["start"] for c in rec["chapters"]]
        self.assertEqual(starts[0], 0.0)
        ends = starts[1:] + [rec["programme_seconds"]]
        self.assertTrue(all(e - s >= 10.0 for s, e in zip(starts, ends)), (starts, rec["programme_seconds"]))
        step = next(s for s in ir.doc["plan"]["steps"] if s["skill"] == ch.SKILL)
        self.assertEqual(step["tool"], "ffmpeg-skill/metadata")
        chk = next(s for s in ir.doc["plan"]["steps"] if s["skill"] == "delivery_check")
        self.assertEqual(chk["depends_on"], [step["id"]])
        self.assertEqual(ir.doc["plan"]["status"], "REVIEW")
        out = svc.render(load_ir(path), path)
        self.assertEqual(out["status"], "WAITING_FOR_APPROVAL", "the chapters wait for a person")
        out = svc.render(load_ir(path), path, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED", out.get("execution", {}).get("failed_op"))
        (art,) = [a for a in out["artifacts"] if a["logical_name"].endswith("_delivery_youtube")]
        written = _read_fake(art["path"])
        self.assertEqual([c["title"] for c in written["chapters"]], ["Welcome", "The sound system", "Questions"])
        self.assertEqual([c["start"] for c in written["chapters"]], starts)
        tools = [r["tool"] for r in out["execution"]["results"]]
        self.assertEqual(tools[tools.index("ffmpeg-skill/export") + 1:tools.index("ffmpeg-skill/export") + 3], ["ffmpeg-skill/metadata", "ffmpeg-skill/check"],
                         "export, then the chapters, then the platform check of the file with chapters")

    def test_the_youtube_description_carries_the_timestamps(self):
        from test_publish import MODES, pub_service
        for k in MODES:
            os.environ.pop(k, None)
        svc = pub_service(self.tmp)
        svc.provider = _Chapters(THREE)
        ir = svc.plan([self.src], "youtube", user_requirements={"delivery.chapters": True, "qc": True, "publish.platform": "youtube", "publish.title": "Keynote",
                                                                 "publish.description": "Recorded live."}, params={"language": "en"})
        pub = ir.doc["delivery"]["publish"]
        starts = [ch.timestamp(c["start"]) for c in ir.doc["delivery"]["chapters"]["chapters"]]
        self.assertEqual(pub["description"], "Recorded live.\n\n" + "\n".join(f"{t} {n}" for t, n in zip(starts, ["Welcome", "The sound system", "Questions"])))
        self.assertEqual(svc.validate(ir).errors, [])

    def test_rejected_chapters_are_not_written(self):
        svc, ir, path = self.plan(_Chapters(THREE))
        did = self.decision(ir.doc)[0]["id"]
        svc.reject(load_ir(path), path, [did], reason="no chapters")
        svc.revise(load_ir(path), path)
        doc = load_ir(path).doc
        self.assertNotIn("chapters", doc["delivery"])
        self.assertFalse([s for s in doc["plan"]["steps"] if s["skill"] == ch.SKILL])

    def test_a_refused_list_is_recorded_and_nothing_is_written(self):
        svc, ir, _ = self.plan(_Chapters({"chapters": [{"segment": 0, "title": "A"}, {"segment": 5, "title": "B"}], "times": [0, 75]}))
        (d,) = self.decision(ir.doc)
        self.assertEqual(d["type"], "KEEP")
        self.assertIn("never times", d["params"]["refused"])
        self.assertNotIn("chapters", ir.doc["delivery"])
        self.assertEqual(svc.validate(ir).errors, [])

    def test_the_chat_model_answers_the_brief_and_revise_writes_its_chapters(self):
        from video_agent.providers.host import HostProvider
        host = HostProvider()
        svc, ir, path = self.plan(host)
        brief = host.brief()
        self.assertIn("chapters", brief["context"], "the brief carries the numbered segments")
        tr = brief["context"]["chapters"]["transcripts"][0]
        self.assertEqual(self.decision(ir.doc)[0]["type"], "KEEP")
        answer = [{"intent": "transcript_chapters", "asset_id": tr["asset_id"], "statement": "three topics", "confidence": 0.9, "evidence": [tr["transcript_id"]], "params": THREE}]
        rev = svc.revise(load_ir(path), path, ai_recommendations=answer)
        self.assertTrue(rev["created"])
        doc = load_ir(path).doc
        self.assertEqual([c["title"] for c in doc["delivery"]["chapters"]["chapters"]], ["Welcome", "The sound system", "Questions"])
        self.assertEqual([d["approval"] for d in self.decision(doc)], ["CONFIRM"])

    def test_without_a_provider_nothing_is_written_and_the_record_says_why(self):
        from video_agent.providers.base import NullProvider
        _, ir, _ = self.plan(NullProvider())
        (d,) = self.decision(ir.doc)
        self.assertEqual((d["type"], d["approval"]), ("KEEP", "AUTO"))
        self.assertIn("--ai-recommendations", d["reason"])

    def test_two_recommendations_are_not_guessed_between(self):
        _, ir, _ = self.plan(_Chapters(THREE, copies=2))
        self.assertIn("not guessed", self.decision(ir.doc)[0]["params"]["refused"])

    def test_with_slides_or_a_time_moving_edit_the_chapters_are_blocked(self):
        for reqs, word in (({"edit.speed": 1.5}, "edit.speed"), ({"audio.production": True}, "audio")):
            _, ir, _ = self.plan(_Chapters(THREE), reqs)
            d = self.decision(ir.doc)[0]
            self.assertEqual(d["type"], "BLOCK", reqs)
            self.assertIn(word, d["reason"])

    def test_a_trim_moves_the_chapters_onto_the_programme(self):
        svc, ir, _ = self.plan(_Chapters(THREE), {"edit.trim": "5-90"})
        rec = ir.doc["delivery"]["chapters"]
        self.assertEqual([c["start"] for c in rec["chapters"]], [0.0, 26.0, 56.0], "the trim removed the first 5 s")
        self.assertEqual(svc.validate(ir).errors, [])

    def test_tampered_chapters_are_validation_errors(self):
        svc, ir, _ = self.plan(_Chapters(THREE))
        ir.doc["delivery"]["chapters"]["chapters"][1]["start"] = 3.0
        self.assertTrue(any("under 10s" in e for e in svc.validate(ir).errors))
        ir.doc["delivery"]["chapters"]["chapters"][1]["start"] = 32.0   # still 10 s or longer, but not what was decided
        self.assertTrue(any("differ from the chapters.transcript decision" in e for e in svc.validate(ir).errors))

    def test_without_the_requirement_a_chapter_recommendation_is_dropped(self):
        svc = pipeline_service(self.tmp)
        svc.provider = _Chapters(THREE)
        ir = svc.plan([self.src], "youtube", user_requirements={"subtitle": True}, params={"language": "en"})
        self.assertFalse(self.decision(ir.doc))
        self.assertFalse([d for d in ir.doc["decisions"] if d["subject"] == "ai.transcript_chapters"])
        self.assertNotIn("chapters", ir.doc["delivery"])

    def test_a_plan_without_chapters_is_unchanged(self):
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self.src], "youtube", user_requirements={}, params={"language": "en"})
        self.assertNotIn("chapters", ir.doc["delivery"])
        self.assertFalse([s for s in ir.doc["plan"]["steps"] if s["skill"] == ch.SKILL])


if __name__ == "__main__":
    unittest.main()
