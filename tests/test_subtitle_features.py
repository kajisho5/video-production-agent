"""ADR-101: subtitle-skill 0.5.0 .. 0.8.0's opt-in inputs wired end to end on the fake engine and the fake subtitle-skill: the TTML / EBU-TT-D
sidecar formats (checked by the Skill's conformance; QC reads srt / vtt only), render `placement` (caller-named regions), `regional_profile`
(ja-broadcast, only when asked and only for Japanese), `caption_profile` shorts (word timings of a word-timestamped transcript on a vertical
picture), cue `ruby` from the user's readings and style `vertical`. Without these keys nothing changes (international default)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import PipelineCaps, clear_modes, fake_media, pipeline_service, plan_and_render  # noqa: E402
from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.requirements import requirement_map  # noqa: E402
from video_agent.agent.subtitles import FORMATS, annotate_cues, parse_subtitle_requirements, words_on_timeline  # noqa: E402
from video_agent.models import Requirement  # noqa: E402

SEGS = [[0.5, 3.2, "本日の講演を始めます"], [4.0, 8.4, "会場の音響設備についてご説明します"]]
WORDS = [[0.5, 1.2, "本日の"], [1.2, 2.0, "講演を"], [2.0, 3.2, "始めます"], [4.0, 5.5, "会場の"], [5.5, 7.0, "音響設備に"], [7.0, 8.4, "ついて"]]


def reqs(d):
    return requirement_map([Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in d.items()])


class VocabularyTests(unittest.TestCase):
    def test_formats_are_the_contracts(self):
        self.assertEqual(FORMATS, ("srt", "vtt", "ttml", "ebu-tt-d"))
        self.assertEqual(parse_subtitle_requirements(reqs({"subtitle": "ebu-tt-d"}))["format"], "ebu-tt-d")
        base = parse_subtitle_requirements(reqs({"subtitle": True}))
        self.assertEqual((base["render"], base["cues"]), ({}, {}), "nothing regional / vertical / ruby unless asked (international default)")

    def test_options_and_refusals(self):
        p = parse_subtitle_requirements(reqs({"subtitle": True, "subtitle.burn_in": True, "subtitle.regional_profile": "ja-broadcast", "subtitle.ruby": {"講演": "こうえん"},
                                              "subtitle.placement.avoid": [{"name": "logo", "box": [0.7, 0.8, 0.3, 0.2], "start": 0, "end": 5}], "subtitle.placement.min_hold": 1.5}))
        self.assertEqual(p["render"], {"regional_profile": "ja-broadcast", "placement": {"avoid": [{"name": "logo", "box": [0.7, 0.8, 0.3, 0.2], "start": 0.0, "end": 5.0}], "min_hold": 1.5}})
        self.assertEqual(p["cues"], {"regional_profile": "ja-broadcast", "ruby": {"講演": "こうえん"}})
        self.assertEqual(parse_subtitle_requirements(reqs({"subtitle": "ttml", "subtitle.vertical": True}))["cues"], {"vertical": True})
        for bad, msg in (({"subtitle": True, "subtitle.regional_profile": "us-broadcast"}, "one of"),
                         ({"subtitle": True, "subtitle.caption_profile": "shorts"}, "needs subtitle.burn_in"),
                         ({"subtitle": True, "subtitle.placement.avoid": [{"name": "x", "box": [0, 0, 0.5, 0.5]}]}, "burn-in or a soft track"),
                         ({"subtitle": True, "subtitle.burn_in": True, "subtitle.placement.avoid": [{"name": "x", "box": [0.8, 0, 0.5, 0.5]}]}, "inside the frame"),
                         ({"subtitle": True, "subtitle.burn_in": True, "subtitle.placement.min_hold": 1}, "ambiguous"),
                         ({"subtitle": "srt", "subtitle.vertical": True}, "ttml or vtt"), ({"subtitle": "ttml", "subtitle.vertical": True, "subtitle.burn_in": True}, "sidecar style"),
                         ({"subtitle": True, "subtitle.ruby": {"講演": ""}}, "single-line"), ({"subtitle.ruby": {"a": "b"}}, "ambiguous")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                parse_subtitle_requirements(reqs(bad))
            self.assertIn(msg, str(cm.exception))

    def test_annotation_and_words(self):
        cues = [{"id": "c0001", "start": 0.0, "end": 2.0, "text": "東京の講演と講演会"}]
        out = annotate_cues(cues, ruby={"講演": "こうえん", "講演会": "こうえんかい", "大阪": "おおさか"}, vertical=True,
                            words=[{"text": "東京の", "start": 0.0, "end": 0.5}, {"text": "講演と", "start": 0.5, "end": 1.0}, {"text": "講演会", "start": 1.0, "end": 2.5}])
        self.assertEqual(out[0]["ruby"], [{"base": "講演", "text": "こうえん"}, {"base": "講演会", "text": "こうえんかい"}], "longest base first, never overlapping, in text order")
        self.assertEqual(out[0]["style"], {"vertical": True})
        self.assertEqual(out[0]["words"][-1], {"text": "講演会", "start": 1.0, "end": 2.0}, "a word is clipped to its cue")
        self.assertNotIn("ruby", cues[0], "the input cues are not changed")
        mapped = words_on_timeline([{"start": 1.0, "end": 1.5, "text": "a"}, {"start": 3.0, "end": 3.5, "text": "cut"}, {"start": 5.0, "end": 5.6, "text": "b"}], [[0.0, 2.0], [4.0, 6.0]])
        self.assertEqual(mapped, [{"text": "a", "start": 1.0, "end": 1.5}, {"text": "b", "start": 3.0, "end": 3.6}])


class PlanTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = tempfile.mkdtemp()
        self.a = fake_media(self.tmp, "a.mp4", segments=SEGS, words=WORDS)

    def tearDown(self):
        clear_modes()

    def _prov(self, svc, out):
        return json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))

    def test_ttml_sidecar_with_ruby_and_vertical(self):
        svc = pipeline_service(self.tmp, caps=PipelineCaps())
        r = plan_and_render(svc, [self.a], {"subtitle": "ttml", "subtitle.vertical": True, "subtitle.ruby": {"講演": "こうえん"}, "qc": True}, profile="generic", name="t1")
        d = r["ir"].doc
        self.assertEqual((d["plan"]["status"], r["validation"].errors), ("APPROVED", []), d["plan"]["summary"])
        gen = next(op for op in d["captions"]["operations"] if op["type"] == "captions.generate")
        self.assertEqual(gen["format"], "ttml")
        self.assertTrue(all(c.get("style") == {"vertical": True} for c in gen["cues"]))
        self.assertEqual(gen["cues"][0]["ruby"], [{"base": "講演", "text": "こうえん"}])
        self.assertFalse(any(s["id"].startswith("step_qc_captions_") for s in d["plan"]["steps"]), "qc-skill reads srt / vtt; the TTML conformance check stands for it")
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        cap = [i for i in out["qa"]["items"] if i.get("name") in ("cues", "format") and i["status"] != "PASS"]
        self.assertEqual(cap, [], "the agent's QA reads the TTML sidecar (its <p> cues, its tt root)")
        res = next(x for x in out["execution"]["results"] if x["tool"] == "subtitle/generate")
        self.assertTrue(res["output"].endswith(".ttml")); self.assertIs(res["data"]["conformance"]["valid"], True)
        # a sidecar the Skill reports as not conforming is not admitted
        clear_modes()
        import os
        os.environ["FAKE_SUBTITLE_MODE"] = "nonconformant"
        r2 = plan_and_render(pipeline_service(tempfile.mkdtemp(), caps=PipelineCaps()), [self.a], {"subtitle": "ebu-tt-d"}, profile="generic", name="t1b")
        self.assertNotEqual(r2["out"]["execution"]["status"], "COMPLETED")

    def test_burn_with_placement_and_regional_profile(self):
        svc = pipeline_service(self.tmp, caps=PipelineCaps())
        avoid = [{"name": "logo", "box": [0.0, 0.8, 1.0, 0.2]}]
        r = plan_and_render(svc, [self.a], {"subtitle": True, "subtitle.burn_in": True, "subtitle.regional_profile": "ja-broadcast", "subtitle.placement.avoid": avoid,
                                            "subtitle.placement.max_lines": 2}, name="t2")
        d = r["ir"].doc
        self.assertEqual((d["plan"]["status"], r["validation"].errors), ("APPROVED", []), d["plan"]["summary"])
        burn = next(op for op in d["captions"]["operations"] if op["type"] == "captions.burn")
        self.assertEqual((burn["regional_profile"], burn["placement"]["avoid"][0]["name"], burn["placement"]["max_lines"]), ("ja-broadcast", "logo", 2))
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        rows = {e["tool"]: e for e in self._prov(svc, out)["operations"] if e["tool"].startswith("subtitle/")}
        self.assertEqual(rows["subtitle/render"]["args"]["regional_profile"], "ja-broadcast")
        self.assertEqual(rows["subtitle/render"]["args"]["placement"]["avoid"][0]["box"], [0.0, 0.8, 1.0, 0.2])
        self.assertEqual(rows["subtitle/generate"]["args"]["regional_profile"], "ja-broadcast")

    def test_regional_profile_never_applied_to_another_language(self):
        svc = pipeline_service(self.tmp, caps=PipelineCaps())
        ir = svc.plan([self.a], "youtube", user_requirements={"subtitle": True, "subtitle.regional_profile": "ja-broadcast"}, params={"language": "en"})
        blk = next(x for x in ir.doc["decisions"] if x["subject"] == "subtitle.generate")
        self.assertEqual(blk["approval"], "BLOCK"); self.assertIn("ja", blk["reason"])

    def test_shorts_captions_need_words_and_a_vertical_picture(self):
        svc = pipeline_service(self.tmp, caps=PipelineCaps())
        r = plan_and_render(svc, [self.a], {"subtitle": True, "subtitle.burn_in": True, "subtitle.caption_profile": "shorts", "edit.fill": "9:16"}, name="t3")
        d = r["ir"].doc
        self.assertEqual((d["plan"]["status"], r["validation"].errors), ("APPROVED", []), d["plan"]["summary"])
        gen = next(op for op in d["captions"]["operations"] if op["type"] == "captions.generate")
        self.assertTrue(all(c.get("words") for c in gen["cues"]), gen["cues"])
        for c in gen["cues"]:   # the recognised words, re-timed with the cue through the plan's trim, each inside its cue and in order
            self.assertTrue(all(w["text"] in {t for _a, _b, t in WORDS} and c["start"] <= w["start"] < w["end"] <= c["end"] for w in c["words"]), c)
            self.assertEqual([w["start"] for w in c["words"]], sorted(w["start"] for w in c["words"]))
        burn = next(op for op in d["captions"]["operations"] if op["type"] == "captions.burn")
        self.assertEqual(burn["caption_profile"], "shorts")
        self.assertEqual(r["out"]["execution"]["status"], "COMPLETED", r["out"]["execution"])
        # a landscape picture is refused before execution
        ir = svc.plan([self.a], "youtube", user_requirements={"subtitle": True, "subtitle.burn_in": True, "subtitle.caption_profile": "shorts", "edit.speed": 2},
                      params={"language": "ja"})
        blk = next(x for x in ir.doc["decisions"] if x["subject"] == "subtitle.burn_in" and x["approval"] == "BLOCK")
        self.assertIn("vertical", blk["reason"])

    def test_publish_takes_srt_or_vtt_captions(self):
        svc = pipeline_service(self.tmp, caps=PipelineCaps(extra=["publish"]))
        ir = svc.plan([self.a], "youtube", user_requirements={"subtitle": "ttml", "publish.platform": "youtube", "publish.title": "T", "publish.captions": True, "qc": True},
                      params={"language": "ja"})
        blk = [x for x in ir.doc["decisions"] if x["subject"].startswith("publish") and x["approval"] == "BLOCK"]
        self.assertTrue(blk and "srt or vtt" in blk[0]["reason"], [x["reason"] for x in blk])


if __name__ == "__main__":
    unittest.main()
