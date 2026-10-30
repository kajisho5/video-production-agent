"""Timeline export (ADR-052): `delivery.timeline` → timeline.export decision → timeline_export step → IR delivery.timelines →
compiler → ffmpeg-skill/render --export-timeline (fake engine here; tests/test_integration.py runs the real render.py and reads
the .otio back with opentimelineio) → TIMELINE Artifact with sha256 and provenance."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import PipelineCaps, clear_modes, fake_media, pipeline_service, plan_and_render  # noqa: E402
from video_agent.execution import compile_ir  # noqa: E402
from video_agent.tools.base import ToolError  # noqa: E402
from video_agent.tools.ffmpeg_skill.adapter import FfmpegSkillAdapter, PathPolicy  # noqa: E402
from video_agent.tools.ffmpeg_skill.locate import FfmpegSkill  # noqa: E402


def skills_of(doc):
    return [s["skill"] for s in doc["plan"]["steps"]]


def timeline_decisions(doc):
    return [d for d in doc["decisions"] if d["subject"] == "timeline.export"]


class TimelinePlanTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.a = fake_media(self.tmp, "a.mp4")
        self.b = fake_media(self.tmp, "b.mp4")

    def tearDown(self):
        clear_modes()

    def test_absent_or_none_plans_nothing(self):
        svc = pipeline_service(self.tmp)
        for reqs in ({}, {"delivery.timeline": "none"}, {"delivery.timeline": False}):
            d = svc.plan([self.a], "youtube", user_requirements=reqs).doc
            self.assertNotIn("timeline_export", skills_of(d))
            self.assertEqual(timeline_decisions(d), [])
            self.assertNotIn("timelines", d["delivery"], "an unchanged request keeps the delivery section (and the plan hash) as before")

    def test_unknown_value_or_key_refused_before_analysis(self):
        svc = pipeline_service(self.tmp)
        with self.assertRaisesRegex(ValueError, "delivery.timeline must be one of none, otio, fcpxml, edl"):
            svc.plan([self.a], "youtube", user_requirements={"delivery.timeline": "aaf"})
        with self.assertRaisesRegex(ValueError, "unknown requirement key.*delivery.timeline.format"):
            svc.plan([self.a], "youtube", user_requirements={"delivery.timeline.format": "otio"})

    def test_trimmed_source_becomes_its_keep_ranges(self):
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self.a], "youtube", user_requirements={"delivery.timeline": "OTIO"})
        d = ir.doc
        self.assertEqual(svc.validate(ir).errors, [])
        trim = next(op for op in d["video"]["operations"] if op["type"] == "video.trim")
        (tl,) = d["delivery"]["timelines"]
        aid = next(iter(d["assets"]))
        self.assertEqual(tl["format"], "otio")
        self.assertEqual([[c["in"], c["out"]] for c in tl["clips"]], trim["keep"], "the timeline carries exactly the ranges the cut executes")
        self.assertTrue(all(c["asset"] == aid for c in tl["clips"]))
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "timeline_export")
        self.assertEqual(step["tool"], "ffmpeg-skill/render")
        self.assertEqual((step["inputs"], step["depends_on"], step["params"]), ([aid], [], {"asset": aid, "format": "otio", "clips": len(tl["clips"])}))
        (dec,) = timeline_decisions(d)
        self.assertEqual((dec["type"], dec["approval"], dec["provenance"]), ("DELIVER", "AUTO", "USER"))
        self.assertIn("audio.loudness", dec["params"]["not_exported"], "what the timeline leaves out is named on the decision")
        out = next(o for o in d["plan"]["outputs"] if o["role"] == "TIMELINE")
        self.assertEqual(out["expected"], {"clips": len(tl["clips"]), "duration": tl["timeline_duration"], "source": aid})

    def test_untouched_source_is_one_clip_of_its_whole_duration(self):
        svc = pipeline_service(self.tmp, silences=[])
        d = svc.plan([self.a], "generic", user_requirements={"delivery.timeline": "edl"}).doc
        self.assertEqual([op for op in d["video"]["operations"] if op["type"] == "video.trim"], [])
        (tl,) = d["delivery"]["timelines"]
        self.assertEqual([[c["in"], c["out"]] for c in tl["clips"]], [[0.0, 16.0]])

    def test_concat_lists_inputs_in_order_and_names_a_dropped_transition(self):
        svc = pipeline_service(self.tmp)
        d = svc.plan([self.a, self.b], "generic", user_requirements={"edit.concat": True, "edit.concat.transition": "fade", "delivery.timeline": "fcpxml"}).doc
        (tl,) = d["delivery"]["timelines"]
        concat = next(op for op in d["video"]["operations"] if op["type"] == "video.concat")
        self.assertEqual(tl["asset"], "programme")
        self.assertEqual([c["asset"] for c in tl["clips"]], concat["inputs"])
        (dec,) = timeline_decisions(d)
        self.assertIn("video.concat transition", dec["params"]["not_exported"])
        self.assertIn("not in the timeline", dec["reason"])

    def test_programmes_that_are_not_cut_lists_block(self):
        svc = pipeline_service(self.tmp)
        d = svc.plan([self.a, self.b], "generic", user_requirements={"edit.switch": "0-5:0,5-16:1", "delivery.timeline": "otio"}).doc
        self.assertEqual([x["type"] for x in timeline_decisions(d)], ["BLOCK"])
        self.assertEqual(d["plan"]["status"], "BLOCKED")
        self.assertNotIn("timelines", d["delivery"])
        w = fake_media(self.tmp, "v.wav", video=False)
        d = svc.plan([w], "generic", user_requirements={"delivery.timeline": "otio"}).doc
        self.assertEqual([x["decision"] for x in timeline_decisions(d)], [f"BLOCK: otio timeline of {next(iter(d['assets']))} (no video stream)"])
        d = svc.plan([w], "generic", user_requirements={"audio.production": True, "audio.gain": -3, "delivery.timeline": "otio"}).doc
        self.assertIn("audio path", next(x for x in timeline_decisions(d))["reason"])

    def test_missing_engine_capability_blocks(self):
        svc = pipeline_service(self.tmp, caps=PipelineCaps(missing=["ffmpeg-skill"]))
        d = svc.plan([self.a], "youtube", user_requirements={"delivery.timeline": "otio"}).doc
        self.assertIn("capability.timeline_export", {x["subject"] for x in d["decisions"] if x["approval"] == "BLOCK"})

    def test_compiler_paths_and_typed_args(self):
        svc = pipeline_service(self.tmp)
        for fmt, ext in (("otio", ".otio"), ("fcpxml", ".fcpxml"), ("edl", ".edl")):
            ir = svc.plan([self.a], "youtube", user_requirements={"delivery.timeline": fmt})
            job = str(Path(self.tmp) / "job")
            ops, paths = compile_ir(ir, job)
            (op,) = [o for o in ops if o.skill == "timeline_export"]
            out_id = op.outputs[0]
            self.assertEqual(op.tool, "ffmpeg-skill/render")
            self.assertTrue(paths[out_id].startswith(str(Path(job) / "artifacts")) and paths[out_id].endswith(ext))
            self.assertTrue(paths[op.args["project"]].startswith(str(Path(job) / "ops")))
            self.assertEqual(set(op.args), {"project", "clips", "output", "overwrite"})
            tl = ir.doc["delivery"]["timelines"][0]
            self.assertEqual(op.args["clips"], [{"input": c["asset"], "in": c["in"], "out": c["out"]} for c in tl["clips"]])
            self.assertEqual(op.inputs, [next(iter(ir.doc["assets"]))], "the timeline reads the source, never an intermediate")

    def test_render_registers_a_timeline_artifact(self):
        svc = pipeline_service(self.tmp)
        r = plan_and_render(svc, [self.a], {"delivery.timeline": "otio"}, name="tl")
        out, d = r["out"], r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        self.assertEqual(out["execution"]["status"], "COMPLETED")
        (art,) = [a for a in out["artifacts"] if a["type"] == "TIMELINE"]
        self.assertTrue(art["path"].startswith(os.path.realpath(svc.workspace)) or art["path"].startswith(svc.workspace))
        self.assertEqual(art["hash"], hashlib.sha256(Path(art["path"]).read_bytes()).hexdigest())
        self.assertEqual((art["format"], art["tool"], art["capability_id"]), ("otio", "ffmpeg-skill/render", "timeline_export"))
        self.assertEqual(art["step_id"], f"step_timeline_{next(iter(d['assets']))}")
        trim = next(op for op in d["video"]["operations"] if op["type"] == "video.trim")
        self.assertEqual(art["decision_ids"], sorted([timeline_decisions(d)[0]["id"]] + trim["decision_ids"]), "the timeline cites its own decision and the cut's")
        self.assertEqual(art["qa_status"], "PASS")
        items = {i["name"]: i for i in out["qa"]["items"] if i["artifact"] == art["logical_name"]}
        self.assertEqual((items["clips"]["status"], items["clips"]["observed"]), ("PASS", len(d["delivery"]["timelines"][0]["clips"])))
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        (e,) = [x for x in prov["operations"] if x["skill"] == "timeline_export"]
        blob = json.dumps(e).lower()
        for leak in ('"argv"', '"command"', "--export-timeline", "render.py"):
            self.assertNotIn(leak, blob)

    def test_rejected_decision_is_cited_until_revised(self):
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self.a], "youtube", user_requirements={"delivery.timeline": "otio"})
        ir.reject([timeline_decisions(ir.doc)[0]["id"]], "user", "no timeline after all")
        self.assertEqual([op["type"] for op in ir.rejected_cited()], ["timeline.export"])
        ir = svc.plan([self.a], "youtube", user_requirements={"delivery.timeline": "otio"})
        trim = next(op for op in ir.doc["video"]["operations"] if op["type"] == "video.trim")
        self.assertEqual(ir.doc["delivery"]["timelines"][0]["decision_ids"][1:], trim["decision_ids"])
        ir.reject(trim["decision_ids"][:1], "user", "keep the opening")
        self.assertEqual(sorted(op["type"] for op in ir.rejected_cited()), ["timeline.export", "video.trim"], "a rejected cut invalidates the timeline built from it")

    def test_validator_refuses_a_tampered_timeline(self):
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self.a], "youtube", user_requirements={"delivery.timeline": "otio"})
        tl = ir.doc["delivery"]["timelines"][0]
        tl["clips"][0]["out"] = 99.0
        errs = svc.validate(ir).errors
        self.assertTrue(any("not inside its source" in e for e in errs), errs)
        tl["clips"][0]["out"] = ir.doc["delivery"]["timelines"][0]["clips"][0]["in"] + 1.0
        tl["format"] = "aaf"
        errs = svc.validate(ir).errors
        # with jsonschema installed the schema's enum refuses it first; without it the validator's own check does
        self.assertTrue(any("unknown format 'aaf'" in e or "'aaf' is not one of" in e for e in errs), errs)
        tl["format"] = "otio"
        ir.doc["plan"]["steps"] = [s for s in ir.doc["plan"]["steps"] if s["skill"] != "timeline_export"]
        self.assertTrue(any("timeline.export on" in e and "has no plan step" in e for e in svc.validate(ir).errors))


_FAKE_RENDER = '''import argparse, json, sys
ap = argparse.ArgumentParser()
ap.add_argument("project")
ap.add_argument("--export-timeline")
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--json", action="store_true")
a = ap.parse_args()
proj = json.load(open(a.project))
open(a.export_timeline, "w").write(json.dumps({"OTIO_SCHEMA": "Timeline.1", "argv": sys.argv[1:], "project": proj}))
print(json.dumps({"status": "completed", "output": a.export_timeline, "commands": [], "timeline": {"clips": len(proj["clips"])}}))
'''


class TimelineAdapterTests(unittest.TestCase):
    """The reference adapter's side (no ffmpeg): typed clips → a project document in the workspace, never onto the command line."""

    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        root = Path(self.tmp) / "ffmpeg-skill"
        (root / "scripts").mkdir(parents=True)
        (root / "scripts" / "render.py").write_text(_FAKE_RENDER, encoding="utf-8")
        self.src_dir = Path(self.tmp) / "src"; self.src_dir.mkdir()
        self.src = str(self.src_dir / "a.mp4"); Path(self.src).write_bytes(b"x")
        self.ws = Path(self.tmp) / "ws"; self.ws.mkdir()
        self.ad = FfmpegSkillAdapter(FfmpegSkill(root, "2.21.0", ["render"]), PathPolicy([str(self.src_dir)], str(self.ws)))
        self.paths = {"A": self.src, "P": str(self.ws / "ops" / "p.json"), "T": str(self.ws / "artifacts" / "t.otio")}
        self.args = {"project": "P", "clips": [{"input": "A", "in": 1.0, "out": 3.5}, {"input": "A", "in": 5, "out": 9}], "output": "T", "overwrite": True}

    def test_argv_and_project_document(self):
        argv = self.ad.build_argv("ffmpeg-skill/render", self.args, self.paths)
        self.assertEqual(argv, [self.paths["P"], "--export-timeline", self.paths["T"], "--overwrite"])
        path, doc = self.ad.project_document("ffmpeg-skill/render", self.args, self.paths)
        self.assertEqual(path, self.paths["P"])
        self.assertEqual(doc, {"clips": [{"src": self.src, "in": 1.0, "out": 3.5}, {"src": self.src, "in": 5.0, "out": 9.0}], "transition": {"type": "none"}})

    def test_run_writes_the_project_then_exports(self):
        from video_agent.models import Operation
        (self.ws / "artifacts").mkdir()
        r = self.ad.run(Operation(tool="ffmpeg-skill/render", args=self.args, inputs=["A"], outputs=["T"]), self.paths)
        self.assertTrue(r.ok, r.stderr_tail)
        self.assertEqual(r.output, self.paths["T"])
        written = json.loads(Path(self.paths["T"]).read_text())
        self.assertEqual(written["project"]["transition"], {"type": "none"})
        self.assertEqual(written["argv"], [self.paths["P"], "--export-timeline", self.paths["T"], "--overwrite", "--json"])

    def test_policy_and_shape_refusals(self):
        outside = str(Path(self.tmp) / "elsewhere.mp4")
        for clips, msg in (([{"input": outside, "in": 0, "out": 1}], "outside allowed roots"), ([{"input": "A", "in": 3, "out": 1}], "0 <= in < out"),
                           ([{"input": "A", "in": 0, "out": 1, "filter": "x"}], "exactly"), ([], "non-empty")):
            with self.assertRaisesRegex(ToolError, msg):
                self.ad.build_argv("ffmpeg-skill/render", dict(self.args, clips=clips), self.paths)
        with self.assertRaisesRegex(ToolError, "outside workspace"):
            self.ad.project_document("ffmpeg-skill/render", self.args, dict(self.paths, P=str(self.src_dir / "p.json")))
        with self.assertRaisesRegex(ToolError, "not in the catalog"):
            self.ad.build_argv("ffmpeg-skill/render", dict(self.args, template="tiktok"), self.paths)

    def test_supported_only_where_help_offers_the_flag(self):
        self.assertTrue(self.ad.supports("ffmpeg-skill/render"))
        old = Path(self.tmp) / "old-skill"
        (old / "scripts").mkdir(parents=True)
        (old / "scripts" / "render.py").write_text("import argparse; argparse.ArgumentParser().parse_args()\n", encoding="utf-8")
        self.assertFalse(FfmpegSkillAdapter(FfmpegSkill(old, "0.9.0", ["render"])).supports("ffmpeg-skill/render"))


if __name__ == "__main__":
    unittest.main()
