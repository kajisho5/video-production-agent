"""Batch selects (ADR-124, Velisa ADR-033): rule selects@1 over measured shots, the selects file, the plan step and the render path."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import clear_modes, fake_media, pipeline_service, plan_and_render  # noqa: E402
from video_agent.agent import selects as sel  # noqa: E402
from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.mcp_server import TOOLS, tool_argv  # noqa: E402


def window(start, blur=5.0, luma=125.0, jitter=0.1, agreement=0.95):
    return {"start": float(start), "end": float(start) + 1.0, "frames": 5, "blur": blur, "luma_mean": luma, "luma_p10": 40.0, "luma_p90": 210.0,
            "jitter": jitter, "speed": 1.0, "motion_agreement": agreement}


def clip(path, shots, dur=6.0, sha="0" * 64, **kw):
    """A measured clip: `kw` per-second picture overrides as {second: {...}}, plus black / freeze / silence / speech spans."""
    over = kw.pop("windows", {})
    pic = [window(i, **over.get(i, {})) for i in range(int(dur))]
    base = {"path": path, "sha256": sha, "duration": dur, "rate": "25", "has_audio": True, "shots": [{"start": a, "end": b} for a, b in shots],
            "picture": pic, "black": [], "freeze": [], "silence": [], "speech": [{"start": 0.0, "end": dur}], "observations": {}, "unmeasured": []}
    base.update(kw)
    return base


def soft(n):
    return {i: {"blur": 14.0} for i in n}


class RuleTests(unittest.TestCase):
    def run_rule(self, clips, **opts):
        o = sel.options(opts)
        return sel.preview(clips, o, {})

    def shots(self, doc):
        return {s["id"]: s for c in doc["clips"] for s in c["shots"]}

    def test_each_flag_rejects_with_its_reason_and_the_rest_stays_in_source_order(self):
        clips = [clip("/a.mp4", [(0, 3), (3, 6)], windows=soft([3, 4, 5])),                       # shot 2 soft
                 clip("/b.mp4", [(0, 6)], windows={i: {"jitter": 2.5} for i in range(6)}),        # shaky
                 clip("/c.mp4", [(0, 6)], black=[{"start": 0.0, "end": 6.0}], windows={i: {"luma": 16.0, "blur": None} for i in range(6)}),
                 clip("/d.mp4", [(0, 6)], silence=[{"start": 0.0, "end": None}], speech=[]),     # dead air: flagged, kept
                 clip("/e.mp4", [(0, 6)], windows={i: {"luma": 230.0} for i in range(6)}),        # over-exposed
                 clip("/f.mp4", [(0, 0.5), (0.5, 6)], freeze=[{"start": 0.5, "end": 6.0}])]       # short, then frozen
        doc = self.run_rule(clips)
        s = self.shots(doc)
        kept = [k for k, v in s.items() if v["kept"]]
        self.assertEqual(kept, ["c01s01", "c04s01"], "source order")
        codes = {k: [f["code"] for f in v["flags"] if f["reject"]] for k, v in s.items()}
        self.assertEqual(codes["c01s02"], ["soft"])
        self.assertEqual(codes["c02s01"], ["shaky"])
        self.assertEqual(codes["c03s01"], ["black"], "a black shot is not also called under-exposed")
        self.assertEqual(codes["c05s01"], ["over_exposed"])
        self.assertEqual(codes["c06s01"], ["short"])
        self.assertEqual(codes["c06s02"], ["frozen"])
        dead = next(f for f in s["c04s01"]["flags"] if f["code"] == "dead_air")
        self.assertFalse(dead["reject"])
        self.assertIn("silent over 100%", dead["reason"])
        self.assertIn("no speech", dead["reason"])
        self.assertEqual(s["c04s01"]["score"], 90.0)
        self.assertEqual(doc["summary"]["duration"], 9.0)
        self.assertEqual(doc["clips"][0]["keep"], [[0.0, 3.0]])
        self.assertTrue(any("REJECT c01s02" in ln and "soft focus" in ln for ln in sel.approval_lines(doc)))
        self.assertIn("in source order", sel.approval_lines(doc)[-1])
        # dead air rejects only when asked
        doc2 = self.run_rule(clips, reject_dead_air=True)
        self.assertFalse(self.shots(doc2)["c04s01"]["kept"])

    def test_soft_is_relative_to_the_batch_unless_max_blur_is_given(self):
        clips = [clip("/a.mp4", [(0, 6)]), clip("/b.mp4", [(0, 6)], windows={i: {"blur": 6.0} for i in range(6)}),
                 clip("/c.mp4", [(0, 6)], windows={i: {"blur": 8.0} for i in range(6)})]
        doc = self.run_rule(clips)
        self.assertEqual(doc["batch"]["blur_reference"], 5.5)                  # 25th percentile of 5, 6, 8
        self.assertEqual([s["kept"] for c in doc["clips"] for s in c["shots"]], [True, True, False])   # soft at 7.15
        doc = self.run_rule(clips, max_blur=5.5)
        self.assertEqual([s["kept"] for c in doc["clips"] for s in c["shots"]], [True, False, False])
        one = self.run_rule([clip("/a.mp4", [(0, 6)], windows=soft(range(6)))])
        self.assertTrue(one["clips"][0]["shots"][0]["kept"], "no reference in a batch of one shot: soft is not judged")
        self.assertIn("not applied", one["batch"]["soft_rule"])

    def test_low_motion_agreement_is_not_judged_as_shake(self):
        doc = self.run_rule([clip("/a.mp4", [(0, 6)], windows={i: {"jitter": 2.5, "agreement": 0.4} for i in range(6)})])
        s = doc["clips"][0]["shots"][0]
        self.assertTrue(s["kept"])
        self.assertIsNone(s["steady"])
        self.assertEqual([f["code"] for f in s["flags"]], ["motion_not_judged"])

    def test_duration_cap_keeps_the_best_scores_then_source_order_and_min_score(self):
        clips = [clip("/a.mp4", [(0, 3), (3, 6)], silence=[{"start": 0.0, "end": 3.0}], speech=[{"start": 3.0, "end": 6.0}]),   # 90, 100
                 clip("/b.mp4", [(0, 4)], dur=4.0)]                                                                              # 100
        doc = self.run_rule(clips, max_duration=7.0)
        s = self.shots(doc)
        self.assertEqual([k for k, v in s.items() if v["kept"]], ["c01s02", "c02s01"])
        self.assertIn("duration cap", s["c01s01"]["reasons"][0])
        self.assertEqual(doc["summary"]["duration"], 7.0)
        doc = self.run_rule(clips, min_score=95)
        self.assertIn("below min_score", self.shots(doc)["c01s01"]["reasons"][0])

    def test_options_and_inputs_are_checked(self):
        for bad in ({"max_jitter": -1}, {"nope": 1}, {"min_score": "x"}, {"under_luma": 200, "over_luma": 100}, {"reject_dead_air": 1}):
            with self.assertRaises(sel.SelectsError):
                sel.options(bad)
        tmp = tempfile.mkdtemp()
        for i in range(3):
            Path(tmp, f"c{i}.mp4").write_bytes(b"x")
        Path(tmp, "notes.txt").write_text("x")
        self.assertEqual([os.path.basename(p) for p in sel.expand_inputs([tmp])], ["c0.mp4", "c1.mp4", "c2.mp4"])
        with self.assertRaises(sel.SelectsError):
            sel.expand_inputs([tmp, os.path.join(tmp, "c0.mp4")])           # given twice
        many = tempfile.mkdtemp()
        for i in range(51):
            Path(many, f"c{i:02d}.mov").write_bytes(b"x")
        with self.assertRaises(sel.SelectsError):
            sel.expand_inputs([many])
        with self.assertRaises(sel.SelectsError):
            sel.expand_inputs([os.path.join(tmp, "missing.mp4")])

    def test_touching_kept_shots_are_one_range(self):
        self.assertEqual(sel.merge_ranges([[3, 6], [0, 3], [7, 8]]), [[0.0, 6.0], [7.0, 8.0]])


class SelectsPipelineTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.a = fake_media(self.tmp, "a.mp4", duration=6.0, silences=[])
        self.b = fake_media(self.tmp, "b.mp4", duration=6.0, silences=[])
        self.c = fake_media(self.tmp, "c.mp4", duration=6.0, silences=[])

    def tearDown(self):
        clear_modes()

    def write_doc(self, clips, name="selects.json", **opts):
        doc = sel.preview(clips, sel.options(opts), {"media-analysis": "test"})
        p = Path(self.tmp) / "ws" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc), encoding="utf-8")
        return str(p), doc

    def clips(self):
        return [clip(self.a, [(0, 2), (2, 4), (4, 6)], sha=sel.sha256_of(self.a), windows=soft([2, 3])),   # keeps 0-2 and 4-6
                clip(self.b, [(0, 6)], sha=sel.sha256_of(self.b), windows={i: {"jitter": 3.0} for i in range(6)}),   # nothing kept
                clip(self.c, [(0, 3), (3, 6)], sha=sel.sha256_of(self.c))]

    def test_plan_cuts_each_clip_to_its_kept_shots_and_joins_them_in_source_order(self):
        path, _ = self.write_doc(self.clips())
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self.a, self.b, self.c], "youtube", user_requirements={"edit.selects": path}, params={"language": "ja"})
        d = ir.doc
        self.assertEqual(svc.validate(ir).errors, [])
        nm = {a["id"]: Path(a["path"]).name for a in (d["assets"].values() if isinstance(d["assets"], dict) else d["assets"])}
        keeps = [x for x in d["decisions"] if x["subject"] == sel.SUBJECT]
        self.assertEqual([(nm[x["params"]["asset_id"]], x["approval"], x["type"]) for x in keeps], [("a.mp4", "CONFIRM", "TRANSFORM"), ("c.mp4", "CONFIRM", "TRANSFORM")])
        self.assertEqual(keeps[0]["params"]["keep"], [[0.0, 2.0], [4.0, 6.0]])
        self.assertEqual(keeps[1]["params"]["keep"], [[0.0, 6.0]], "touching shots are one range")
        self.assertIn("c01s02", keeps[0]["params"]["rejected"])
        trims = {nm[op["asset"]]: op for op in d["video"]["operations"] if op["type"] == "video.trim"}
        self.assertEqual(trims["a.mp4"]["keep"], [[0.0, 2.0], [4.0, 6.0]])
        self.assertTrue(trims["a.mp4"]["accurate"], "the kept shots are cut on their frames")
        self.assertNotIn("b.mp4", trims)
        concat = next(op for op in d["video"]["operations"] if op["type"] == "video.concat")
        self.assertEqual([nm[a] for a in concat["inputs"]], ["a.mp4", "c.mp4"])
        self.assertTrue({x["id"] for x in keeps} <= set(concat["decision_ids"]), "the assembly cites the selects")
        bid = next(k for k, v in nm.items() if v == "b.mp4")
        self.assertFalse([s for s in d["plan"]["steps"] if bid in s["inputs"]], "a clip with no kept shot gets no step")
        self.assertFalse([x for x in d["decisions"] if x["subject"].startswith("silence.")], "a selected clip has no silence trims of its own")

    def test_nothing_renders_before_the_yes_and_the_render_follows_the_selects(self):
        path, _ = self.write_doc(self.clips())
        svc = pipeline_service(self.tmp)
        r = plan_and_render(svc, [self.a, self.b, self.c], {"edit.selects": path}, approve=False)
        self.assertNotEqual(r["out"].get("execution", {}).get("status"), "COMPLETED")
        r = plan_and_render(svc, [self.a, self.b, self.c], {"edit.selects": path}, name="q")
        self.assertEqual(r["validation"].errors, [])
        self.assertEqual((r["out"]["execution"]["status"], r["out"]["qa"]["status"]), ("COMPLETED", "PASS"))

    def test_rejecting_a_selects_decision_rejects_the_assembly(self):
        path, _ = self.write_doc(self.clips())
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self.a, self.b, self.c], "youtube", user_requirements={"edit.selects": path}, params={"language": "ja"})
        concat = next(x for x in ir.doc["decisions"] if x["subject"] == "video.concat")
        self.assertTrue(set(concat["params"]["picks"]) == {x["id"] for x in ir.doc["decisions"] if x["subject"] == sel.SUBJECT})

    def test_a_changed_clip_or_a_bad_file_is_refused(self):
        path, _ = self.write_doc(self.clips())
        Path(self.c).write_bytes(Path(self.c).read_bytes() + b" ")
        with self.assertRaises(sel.SelectsError):
            sel.parse_requirement(path)
        svc = pipeline_service(self.tmp)
        with self.assertRaises((EditRequirementError, ValueError)):
            svc.plan([self.a, self.c], "youtube", user_requirements={"edit.selects": path}, params={"language": "ja"})
        for bad in ("", "../x.json", str(Path(self.tmp) / "missing.json")):
            with self.assertRaises(sel.SelectsError):
                sel.parse_requirement(bad)
        doc = json.loads(Path(path).read_text())
        doc["applied"] = True
        Path(path).write_text(json.dumps(doc))
        with self.assertRaises(sel.SelectsError):
            sel.parse_requirement(path)

    def test_conflicts_and_missing_clips_block(self):
        path, _ = self.write_doc(self.clips())
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self.a, self.b, self.c], "youtube", user_requirements={"edit.selects": path, "edit.concat": True}, params={"language": "ja"})
        self.assertEqual([x["type"] for x in ir.doc["decisions"] if x["subject"] == sel.SUBJECT], ["BLOCK"])
        ir = svc.plan([self.a, self.b], "youtube", user_requirements={"edit.selects": path}, params={"language": "ja"})
        blk = [x for x in ir.doc["decisions"] if x["subject"] == sel.SUBJECT]
        self.assertEqual([x["type"] for x in blk], ["BLOCK"])
        self.assertIn("c.mp4", blk[0]["reason"])

    def test_one_clip_is_delivered_alone(self):
        path, _ = self.write_doc([clip(self.a, [(0, 2), (2, 6)], sha=sel.sha256_of(self.a), windows=soft([0, 1]))], name="one.json")
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self.a], "youtube", user_requirements={"edit.selects": path}, params={"language": "ja"})
        self.assertFalse([x for x in ir.doc["decisions"] if x["subject"] == "video.concat"])
        trim = next(op for op in ir.doc["video"]["operations"] if op["type"] == "video.trim")
        self.assertEqual(trim["keep"], [[2.0, 6.0]])


class McpTests(unittest.TestCase):
    def test_the_tool_is_a_preview_with_data_only_arguments(self):
        tool = next(t for t in TOOLS if t["name"] == "selects")
        self.assertIn("Nothing is rendered", tool["description"])
        argv = tool_argv("selects", {"inputs": ["--output=/etc/x", "b.mov"], "max_duration": 30, "reject_dead_air": True}, "/tmp")
        self.assertEqual(argv, ["selects", "--max-duration=30", "--reject-dead-air", "--", "--output=/etc/x", "b.mov"])
        with self.assertRaises(ValueError):
            tool_argv("selects", {"inputs": []}, "/tmp")


class MeasureClipTests(unittest.TestCase):
    """measure_clip only reads the Skills' observations; a missing optional measurement is listed, never guessed."""

    class Result:
        def __init__(self, ok, data):
            self.ok, self.data, self.stderr = ok, data, ""

    def adapters(self, speech_status="MEASURED", has_audio=True):
        R = self.Result
        obs = {"media_probe": {"container": {"duration": 6.0}, "video": {"fps": 25.0}, "audio": {"codec": "pcm"} if has_audio else None},
               "scene_detection": {"duration": 6.0, "scenes": [{"start": 0.0, "end": 2.5}, {"start": 2.5, "end": None}]},
               "picture_quality": {"duration": 6.0, "windows": [window(i) for i in range(6)]},
               "silence": {"segments": [{"start": 5.0, "end": 6.0, "type": "trailing"}]},
               "speech_regions": {"status": speech_status, "reason": "optional extra vad is not installed", "regions": [{"start": 0.5, "end": 4.0}]}}

        class MA:
            tools = {t: {} for t in sel.MA_KINDS.values()}

            def measure(self, tool, args, timeout=None):
                return R(True, {"observation": {"id": "obs_" + args["kind"], "data": obs[args["kind"]]}})

        class QC:
            def measure(self, tool, args, timeout=None):
                return R(True, {"report_id": "qc1", "measurements": [{"id": "video.black_segments", "value": [{"start": 0.0, "end": 0.5, "duration": 0.5}]},
                                                                     {"id": "video.freeze_segments", "value": [{"start": 5.5, "end": None, "duration": None}]}]})
        return MA(), QC()

    def test_facts_come_from_the_observations(self):
        f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        f.write(b"x")
        f.close()
        ma, qc = self.adapters()
        c = sel.measure_clip(ma, qc, f.name, 0)
        self.assertEqual(c["shots"], [{"start": 0.0, "end": 2.5}, {"start": 2.5, "end": 6.0}])
        self.assertEqual((c["black"][0]["end"], c["freeze"][0]["end"]), (0.5, None))
        self.assertEqual(c["speech"], [{"start": 0.5, "end": 4.0}])
        self.assertEqual(set(c["observations"]), {"media_probe", "scene_detection", "picture_quality", "silence", "speech_regions", "qc_inspect"})
        facts = sel.shot_facts(c, 2.5, 6.0)
        self.assertEqual((facts["freeze_share"], facts["black_share"]), (round(0.5 / 3.5, 3), 0.0))
        ma, qc = self.adapters(speech_status="UNKNOWN")
        c = sel.measure_clip(ma, qc, f.name, 0)
        self.assertIsNone(c["speech"])
        self.assertTrue(any(u.startswith("speech_regions") for u in c["unmeasured"]))
        c = sel.measure_clip(ma, None, f.name, 0)
        self.assertTrue(any("qc-skill is not installed" in u for u in c["unmeasured"]))
        ma.tools = {k: v for k, v in ma.tools.items() if k != "media-analysis/picture"}
        with self.assertRaises(sel.SelectsError):
            sel.measure_clip(ma, qc, f.name, 0)


if __name__ == "__main__":
    unittest.main()
