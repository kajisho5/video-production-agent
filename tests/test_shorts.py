"""A long talk → a set of vertical clips (ADR-138, roadmap step 25): the AI names each clip by its first and last transcript segment with a hook
and a quote, the agent takes the times from the transcript (rule short_clips@1), each clip waits for a person (CONFIRM), and each approved clip
is planned and rendered as a project of its own (trim, 9:16 fill at 1080, captions burned in, hook card, the platform's preset). Fake engine +
fake Skills + a fake provider; no media is decoded."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import clear_modes, fake_media, pipeline_service  # noqa: E402
from video_agent.agent import shorts as sh  # noqa: E402
from video_agent.agent.chapters import numbered_segments  # noqa: E402
from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.requirements import requirement_map  # noqa: E402
from video_agent.models import Requirement  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.providers import AIProvider, AIRequest, AIResponse, AIUsage  # noqa: E402

# a 30-minute talk: one 20 s segment every 30 s
SEGMENTS = [[30.0 * k + 2.0, 30.0 * k + 22.0, f"point {k} of the talk about the hall"] for k in range(60)]
SEGMENTS[10][2] = "this is the moment the whole room went silent"
SEGMENTS[11][2] = "and then the lights came back on stage"
SEGMENTS[30][2] = "nobody expected the second speaker to sing"
SEGMENTS[50][2] = "the one number you should remember is forty two"
THREE = {"clips": [{"first_segment": 10, "last_segment": 11, "hook": "The room went silent", "quote": "the whole room went silent"},
                   {"first_segment": 30, "last_segment": 30, "hook": "Then she sang", "quote": "second speaker to sing"},
                   {"first_segment": 50, "last_segment": 50, "hook": "Remember 42", "quote": "the one number you should remember"}]}


def words_of(segments):
    out = []
    for a, b, t in segments:
        ws = t.split()
        step = (b - a) / len(ws)
        out += [[round(a + i * step, 3), round(a + (i + 1) * step - 0.01, 3), w] for i, w in enumerate(ws)]
    return out


SEGS = numbered_segments([{"id": f"seg_{i:04d}", "start": a, "end": b, "text": t} for i, (a, b, t) in enumerate(SEGMENTS, 1)])
REQ = {"count": 3, "platform": "shorts", "preset": "shorts", "max_seconds": 60.0, "min_seconds": 10.0, "publish": False, "privacy": None}


def reqmap(**kv):
    return requirement_map([Requirement(key=k.replace("__", "."), value=v, provenance="USER", source="cli") for k, v in kv.items()])


class _Clips(AIProvider):
    name = "fake"
    model = "fake-clips-1"

    def __init__(self, params, copies=1):
        self.params, self.copies, self.requests = params, copies, []

    def available(self):
        return True

    def complete(self, request: AIRequest, timeout=None) -> AIResponse:
        self.requests.append(request)
        tr = next(o for o in request.inputs["observations"] if o["kind"] == "transcript")
        recs = [{"intent": "short_clips", "asset_id": tr["asset_id"], "statement": "three moments", "confidence": 0.8, "evidence": [tr["id"]], "params": self.params}] * self.copies
        return AIResponse(task_type=request.task_type, result={"recommendations": recs}, confidence=0.8, provider=self.name, model=self.model, usage=AIUsage())


class RuleTests(unittest.TestCase):
    def test_three_clips_with_their_times_hooks_and_quotes(self):
        got = sh.derive_clips(THREE, SEGS, 1800.0, REQ)
        self.assertEqual([(c["start"], c["end"]) for c in got], [(301.85, 352.15), (901.85, 922.15), (1501.85, 1522.15)])
        self.assertTrue(all(c["seconds"] <= 60.0 for c in got))
        self.assertEqual([c["hook"] for c in got], ["The room went silent", "Then she sang", "Remember 42"])

    def test_refusals_are_never_corrected(self):
        def one(**kw):
            c = dict(THREE["clips"][0], **kw)
            return {"clips": [c]}
        bad = [one(quote="words nobody said"), one(first_segment=12, last_segment=11), one(first_segment=99), one(hook=""), one(hook="x" * 61),
               one(hook="a\nb"), one(last_segment=14), one(first_segment=10, last_segment=10, quote="moment"),   # 20.3 s: fine, see below
               {"clips": THREE["clips"] + [THREE["clips"][0]]}, {"clips": [THREE["clips"][1], THREE["clips"][0]]}, {"clips": []},
               {"clips": THREE["clips"], "times": [1, 2]}, {"clips": [dict(THREE["clips"][0], start=300)]}]
        for n, params in enumerate(bad):
            if n == 7:
                self.assertEqual(len(sh.derive_clips(params, SEGS, 1800.0, REQ)), 1)
                continue
            with self.assertRaises(sh.ClipError, msg=str(params)[:200]):
                sh.derive_clips(params, SEGS, 1800.0, REQ)

    def test_quote_matching_folds_case_and_spacing(self):
        got = sh.derive_clips({"clips": [dict(THREE["clips"][0], quote="The  WHOLE room")]}, SEGS, 1800.0, REQ)
        self.assertEqual(got[0]["quote"], "The WHOLE room")

    def test_requirements(self):
        self.assertIsNone(sh.parse_shorts_requirements(reqmap()))
        r = sh.parse_shorts_requirements(reqmap(shorts__count=3))
        self.assertEqual((r["count"], r["platform"], r["max_seconds"], r["min_seconds"]), (3, "shorts", 60.0, 10.0))
        self.assertEqual(sh.parse_shorts_requirements(reqmap(shorts__count="2", shorts__platform="reels", shorts__max_seconds=90))["max_seconds"], 90.0)
        for bad in ({"shorts__platform": "tiktok"}, {"shorts__count": 0}, {"shorts__count": 2.5}, {"shorts__count": 3, "shorts__platform": "vine"},
                    {"shorts__count": 3, "shorts__platform": "reels", "shorts__max_seconds": 120}, {"shorts__count": 3, "shorts__privacy": "public"},
                    {"shorts__count": 3, "shorts__size": 2}):
            with self.assertRaises(EditRequirementError, msg=str(bad)):
                sh.parse_shorts_requirements(reqmap(**bad))

    def test_the_child_project_keys(self):
        c = sh.derive_clips(THREE, SEGS, 1800.0, REQ)[0]
        keys = sh.child_requirements(c, dict(REQ, publish=True, privacy="public", platform="reels"))
        self.assertEqual(keys["edit.trim"], "301.850-352.150")
        self.assertEqual((keys["edit.fill"], keys["edit.fill.width"], keys["subtitle.burn_in"], keys["motion.hook"]), ("9:16", 1080, True, "The room went silent"))
        self.assertEqual((keys["publish.platform"], keys["publish.privacy"], keys["publish.title"]), ("reels", "public", "The room went silent"))
        self.assertNotIn("publish.platform", sh.child_requirements(c, REQ))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.src = fake_media(self.tmp, "keynote.mp4", duration=1800.0, segments=SEGMENTS, words=words_of(SEGMENTS), silences=[])

    def plan(self, provider, reqs=None, profile="generic"):
        svc = pipeline_service(self.tmp)
        svc.provider = provider
        ir = svc.plan([self.src], profile, user_requirements=dict({"shorts.count": 3}, **(reqs or {})), params={"language": "en"}, chunk_seconds=0)
        path = str(Path(svc.workspace) / "plans" / "keynote.project.json")
        save_ir(ir, path)
        return svc, load_ir(path), path

    @staticmethod
    def clips(doc):
        return [d for d in doc["decisions"] if d["subject"].startswith(sh.SUBJECT + ".")]

    def test_a_30_minute_talk_yields_3_clips_of_at_most_60s_each_confirmed_then_rendered(self):
        prov = _Clips(THREE)
        svc, ir, path = self.plan(prov)
        ctx = prov.requests[0].context["shorts"]
        self.assertEqual(ctx["transcripts"][0]["segments"][10], [10, 302.0, 322.0, "this is the moment the whole room went silent"])
        self.assertIn("short_clips", prov.requests[0].context["allowed_intents"])
        self.assertEqual(svc.validate(ir).errors, [])
        decs = self.clips(ir.doc)
        self.assertEqual([(d["type"], d["approval"], d["status"]) for d in decs], [("TRANSFORM", "CONFIRM", "PROPOSED")] * 3)
        self.assertTrue(all("“" + q + "”" in d["decision"] for d, q in zip(decs, ["the whole room went silent", "second speaker to sing",
                                                                                              "the one number you should remember"])), "each reason is quoted")
        rec = ir.doc["delivery"]["shorts"]
        self.assertEqual(len(rec["clips"]), 3)
        self.assertTrue(all(c["end"] - c["start"] <= 60.0 for c in rec["clips"]))
        out = svc.render(load_ir(path), path)
        self.assertEqual(out["status"], "WAITING_FOR_APPROVAL")
        self.assertNotIn("shorts", out)
        out = svc.render(load_ir(path), path, approve=["all"])
        self.assertIn(out["status"], ("COMPLETED", "REVIEW"), out.get("execution", {}).get("failed_op"))
        sets = out["shorts"]
        self.assertEqual((sets["rendered"], sets["of"]), (3, 3), sets)
        for row, c in zip(sets["clips"], rec["clips"]):
            child = load_ir(row["project"]).doc
            req = {r["key"]: r["value"] for r in child["requirements"] if r["provenance"] == "USER"}
            self.assertEqual(req["edit.trim"], f"{c['start']:.3f}-{c['end']:.3f}")
            self.assertEqual((req["edit.fill"], req["edit.fill.width"], req["subtitle.burn_in"], req["motion.hook"]), ("9:16", 1080, True, c["hook"]))
            self.assertEqual([t["preset"] for t in child["delivery"]["targets"]], ["shorts"])
            skills = [s["skill"] for s in child["plan"]["steps"]]
            self.assertIn("subtitle_burn_in", skills)
            self.assertIn("video_fill", skills)
            self.assertEqual(child["provenance"]["shorts_parent"]["decision_id"], c["decision_ids"][0])
            self.assertTrue(row["artifact"] and Path(row["artifact"]).is_file(), row)

    def test_only_approved_clips_are_rendered(self):
        svc, ir, path = self.plan(_Clips(THREE))
        ids = [d["id"] for d in self.clips(ir.doc)]
        svc.reject(load_ir(path), path, [ids[1]], reason="not this one")
        svc.revise(load_ir(path), path)
        doc = load_ir(path).doc
        self.assertEqual([c["hook"] for c in doc["delivery"]["shorts"]["clips"]], ["The room went silent", "Remember 42"])
        ir = load_ir(path)
        ir.approve([d["id"] for d in self.clips(ir.doc) if d["status"] == "PROPOSED"][:1])
        save_ir(ir, path)
        out = svc.render_shorts(load_ir(path), path)
        self.assertEqual([r["status"] for r in out["clips"]][1], "WAITING_FOR_APPROVAL")
        self.assertEqual(out["rendered"], 1)

    def test_a_refused_set_is_recorded_and_no_clip_is_planned(self):
        svc, ir, _ = self.plan(_Clips({"clips": [dict(THREE["clips"][0], quote="never said")]}))
        (d,) = [x for x in ir.doc["decisions"] if x["subject"] == sh.SET_SUBJECT]
        self.assertEqual(d["type"], "KEEP")
        self.assertIn("not words of segments", d["params"]["refused"])
        self.assertNotIn("shorts", ir.doc["delivery"])
        self.assertEqual(svc.validate(ir).errors, [])

    def test_without_a_provider_nothing_is_planned_and_the_record_says_why(self):
        from video_agent.providers.base import NullProvider
        _, ir, _ = self.plan(NullProvider())
        (d,) = [x for x in ir.doc["decisions"] if x["subject"] == sh.SET_SUBJECT]
        self.assertIn("--ai-recommendations", d["reason"])

    def test_tampered_clip_is_a_validation_error(self):
        svc, ir, _ = self.plan(_Clips(THREE))
        ir.doc["delivery"]["shorts"]["clips"][0]["end"] = ir.doc["delivery"]["shorts"]["clips"][0]["start"] + 90
        self.assertTrue(any("outside" in e for e in svc.validate(ir).errors))

    def test_without_the_requirement_a_clip_recommendation_is_dropped(self):
        svc = pipeline_service(self.tmp)
        svc.provider = _Clips(THREE)
        ir = svc.plan([self.src], "generic", user_requirements={"subtitle": True}, params={"language": "en"}, chunk_seconds=0)
        self.assertFalse([d for d in ir.doc["decisions"] if d["subject"].startswith("shorts.") or d["subject"] == "ai.short_clips"])


if __name__ == "__main__":
    unittest.main()
