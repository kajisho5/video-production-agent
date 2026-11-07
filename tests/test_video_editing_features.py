"""ADR-099: video-editing-skill 0.5.0's features wired end to end on the fake engine and the fake video-editing-skill -- PUNCH_IN
(`edit.punch_in`), cut proposals (`video-agent propose`) applied only by id through CONDENSE (`edit.condense` + `.approve`), TRIM / CUT
precision auto (`edit.precision=auto`), the segment cache (`edit.segment_cache`), the CONCAT stinger (`edit.concat.stinger`), AI-generated
inputs (`edit.ai_generated`) and the lowering of outputs[].max_size. tests/test_integration.py runs them on the real Skill."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter  # noqa: E402
from test_unit import FakeCaps, fake_media, make_service  # noqa: E402
from video_agent.agent.editing import EditRequirementError, parse_edit_requirements, subtract_spans  # noqa: E402
from video_agent.agent.requirements import requirement_map  # noqa: E402
from video_agent.execution.compiler import CompileError, lower_video_trim  # noqa: E402
from video_agent.models import Requirement  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.service import _check_requirement_keys  # noqa: E402
from video_agent.tools.base import ToolError  # noqa: E402
from video_agent.tools.video_editing.adapter import check_proposals, pinned_contract  # noqa: E402
from video_agent.tools.video_editing.lowering import NOT_LOWERED, Lowering  # noqa: E402

FAKE_VE = str(Path(__file__).resolve().parent / "fake_video_editing.py")


def reqs(d):
    return requirement_map([Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in d.items()])


class LoweringTests(unittest.TestCase):
    """The request each new feature becomes, built against the pinned 0.5.0 contract (its parameter allowlist per type)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lw = Lowering(pinned_contract(), self.tmp)
        self.src = str(Path(self.tmp) / "a.mp4")
        self.out = str(Path(self.tmp) / "op" / "out.mp4")

    def build(self, tool, args, paths=None):
        return self.lw.build_request(tool, dict(args, output=self.out), dict(paths or {}), workspace=str(Path(self.tmp) / "op"))[0]

    def test_nothing_declared_is_left_unlowered(self):
        self.assertEqual(NOT_LOWERED, ())
        self.assertEqual(self.lw.contract_only(), [])
        self.assertIn("PUNCH_IN", self.lw.supported_types()); self.assertIn("CONDENSE", self.lw.supported_types())

    def test_punch_in(self):
        req = self.build("video-editing/punch_in", {"input": self.src, "start": 2, "end": 4.5, "scale": 2, "subject": {"x": 600, "y": 200, "width": 160, "height": 160}})
        op = req["project"]["operations"][0]
        self.assertEqual((op["type"], op["input"]), ("PUNCH_IN", "s0"))
        self.assertEqual(op["params"], {"start": 2.0, "end": 4.5, "scale": 2.0, "subject": {"x": 600, "y": 200, "width": 160, "height": 160}})
        for bad, msg in (({"scale": 1}, "above 1"), ({"scale": 4.5}, "at most 4"), ({"start": 5, "end": 4}, "before end"),
                         ({"subject": {"x": 1, "y": 1, "width": 10}}, "subject"), ({"subject": {"x": -1, "y": 0, "width": 4, "height": 4}}, ">= 0")):
            args = dict({"input": self.src, "start": 2, "end": 4.5, "scale": 2, "subject": {"x": 600, "y": 200, "width": 160, "height": 160}}, **bad)
            with self.assertRaises(ToolError) as cm:
                self.build("video-editing/punch_in", args)
            self.assertIn(msg, str(cm.exception))

    def test_condense(self):
        segs = [{"start": 0, "end": 4, "label": "keep"}, {"start": 4, "end": 6, "label": "silence"}, {"start": 6, "end": 16, "label": "keep"}]
        req = self.build("video-editing/condense", {"input": self.src, "segments": segs, "rules": {"keep": "keep", "silence": {"speed": 2}}, "default": "cut", "review": True})
        p = req["project"]["operations"][0]["params"]
        self.assertEqual((p["rules"], p["default"], p["output"]), ({"keep": "keep", "silence": {"speed": 2.0}}, "cut", "removed"))
        self.assertEqual([s["label"] for s in p["segments"]], ["keep", "silence", "keep"])
        for args, msg in (({"segments": [{"start": 0, "end": 5, "label": "a"}, {"start": 4, "end": 6, "label": "b"}], "rules": {"a": "cut", "b": "cut"}}, "overlap"),
                          ({"segments": segs, "rules": {"keep": "keep"}}, "no rule"), ({"segments": [{"start": 0, "end": 1, "label": "Bad"}], "rules": {"Bad": "cut"}}, "label"),
                          ({"segments": segs, "rules": {"keep": "keep", "silence": {"speed": 1}}}, "not 1"), ({"segments": [], "rules": {"a": "cut"}}, "non-empty")):
            with self.assertRaises(ToolError) as cm:
                self.build("video-editing/condense", dict(args, input=self.src))
            self.assertIn(msg, str(cm.exception))

    def test_precision_auto_and_segment_cache(self):
        req = self.build("video-editing/cut", {"input": self.src, "keep": [[0, 2], [3, 5]], "precision": "auto"})
        self.assertEqual(req["project"]["operations"][0]["params"]["precision"], "auto"); self.assertNotIn("segment_cache", req["options"])
        req = self.build("video-editing/trim", {"input": self.src, "start": 1, "end": 3, "precision": "auto"})
        self.assertEqual(req["project"]["operations"][0]["params"]["precision"], "auto")
        req = self.build("video-editing/cut", {"input": self.src, "keep": [[0, 2]], "precision": "frame", "segment_cache": True})
        self.assertIs(req["options"]["segment_cache"], True)
        with self.assertRaises(ToolError):   # the cache renders frame-precision cuts only (contract.segment_cache.applies_to)
            self.build("video-editing/cut", {"input": self.src, "keep": [[0, 2]], "precision": "auto", "segment_cache": True})
        with self.assertRaises(ToolError):
            self.build("video-editing/trim", {"input": self.src, "start": 1, "end": 3, "segment_cache": True})
        with self.assertRaises(ToolError):
            self.build("video-editing/cut", {"input": self.src, "keep": [[0, 2]], "precision": "lossless"})

    def test_precision_smart(self):
        """ADR-106: 0.6.0's TRIM / CUT precision smart (contract.smart_cut) is lowered as asked; the segment cache stays frame-only."""
        self.assertIn("smart", pinned_contract()["smart_cut"]["request_field"])
        req = self.build("video-editing/cut", {"input": self.src, "keep": [[0, 2], [3, 5]], "precision": "smart"})
        self.assertEqual(req["project"]["operations"][0]["params"]["precision"], "smart"); self.assertNotIn("segment_cache", req["options"])
        req = self.build("video-editing/trim", {"input": self.src, "start": 1, "end": 3, "precision": "smart"})
        self.assertEqual(req["project"]["operations"][0]["params"]["precision"], "smart")
        with self.assertRaises(ToolError):   # contract.smart_cut.not_applied: options.segment_cache (frame precision only)
            self.build("video-editing/cut", {"input": self.src, "keep": [[0, 2]], "precision": "smart", "segment_cache": True})

    def test_stinger(self):
        st = str(Path(self.tmp) / "stinger.mov")
        req = self.build("video-editing/concat", {"inputs": ["a", "b"], "transition": {"type": "stinger", "cut_point": 0.5, "cover": {"start": 0.4, "end": 0.8}}, "stinger": "st"},
                         {"a": self.src, "b": str(Path(self.tmp) / "b.mp4"), "st": st})
        op = req["project"]["operations"][0]
        self.assertEqual(op["inputs"], ["s0", "s1"], "the stinger is not an input slot")
        self.assertEqual(op["params"]["transition"], {"type": "stinger", "stinger": "stinger", "cut_point": 0.5, "cover": {"start": 0.4, "end": 0.8}})
        self.assertEqual(req["project"]["sources"][-1], {"id": "stinger", "path": str(Path(st).resolve()), "kind": "video"})
        for args, msg in (({"transition": {"type": "stinger", "cut_point": 0.9, "cover": {"start": 0.4, "end": 0.8}}, "stinger": "st"}, "inside"),
                          ({"transition": {"type": "stinger", "cut_point": 0.5, "cover": {"start": 0.4, "end": 0.8}}}, "stinger` file"),
                          ({"transition": {"type": "fade", "duration": 1}, "stinger": "st"}, "only with")):
            with self.assertRaises(ToolError) as cm:
                self.build("video-editing/concat", dict(args, inputs=["a", "b"]), {"a": self.src, "b": self.src, "st": st})
            self.assertIn(msg, str(cm.exception))

    def test_ai_generated_and_max_size(self):
        req = self.build("video-editing/concat", {"inputs": ["a", "b"], "ai_generated": {"b": {"generator": "a video model", "licence": "CC BY 4.0"}}, "max_size": {"mb": 25}},
                         {"a": self.src, "b": str(Path(self.tmp) / "b.mp4")})
        srcs = req["project"]["sources"]
        self.assertNotIn("ai_generated", srcs[0]); self.assertEqual(srcs[1]["ai_generated"], {"generator": "a video model", "licence": "CC BY 4.0"})
        self.assertEqual(req["project"]["outputs"][0]["max_size"], {"mb": 25})
        for args, msg in (({"ai_generated": {"zz": {"generator": "g", "licence": "l"}}}, "not a video input"), ({"ai_generated": {"a": {"generator": "g"}}}, "generator, licence"),
                          ({"ai_generated": {"a": {"generator": "g\n", "licence": "l"}}}, "printable"), ({"max_size": {"mb": 0.01}}, "100 kB"),
                          ({"max_size": {"tb": 1}}, "one of"), ({"max_size": {"mb": 25, "kb": 1}}, "one of")):
            with self.assertRaises(ToolError) as cm:
                self.build("video-editing/trim", dict(args, input="a", start=0, end=1), {"a": self.src})
            self.assertIn(msg, str(cm.exception))
        with self.assertRaises(ToolError):   # contract.size_limit: a .mp4 output only
            self.lw.build_request("video-editing/trim", {"input": self.src, "start": 0, "end": 1, "max_size": {"mb": 25}, "output": str(Path(self.tmp) / "op" / "o.mov")}, {},
                                  workspace=str(Path(self.tmp) / "op"))

    def test_compiler_trim_lowering(self):
        op = {"type": "video.trim", "asset": "a", "keep": [[0, 4], [6, 16]], "accurate": True, "condense": [{"start": 4, "end": 6, "label": "silence"}]}
        args = lower_video_trim("video-editing/condense", op, "a", "a_trim")
        self.assertEqual(args["segments"], [{"start": 0, "end": 4, "label": "keep"}, {"start": 4.0, "end": 6.0, "label": "silence"}, {"start": 6, "end": 16, "label": "keep"}])
        self.assertEqual((args["rules"], args["default"]), ({"keep": "keep", "silence": "cut"}, "cut"))
        with self.assertRaises(CompileError):
            lower_video_trim("video-editing/cut", op, "a", "a_trim")
        auto = {"type": "video.trim", "asset": "a", "keep": [[1, 5]], "accurate": False, "precision": "auto"}
        self.assertEqual(lower_video_trim("video-editing/cut", auto, "a", "o")["precision"], "auto")
        self.assertEqual(lower_video_trim("video-editing/trim", auto, "a", "o")["precision"], "auto")
        with self.assertRaises(CompileError):   # ffmpeg-skill's cut has no auto: never silently another precision
            lower_video_trim("ffmpeg-skill/cut", auto, "a", "o")
        smart = dict(auto, accurate=True, precision="smart")   # ADR-106: smart is video-editing-skill's too
        self.assertEqual(lower_video_trim("video-editing/trim", smart, "a", "o")["precision"], "smart")
        with self.assertRaises(CompileError):
            lower_video_trim("ffmpeg-skill/cut", smart, "a", "o")
        cached = {"type": "video.trim", "asset": "a", "keep": [[1, 5]], "accurate": True, "precision": "frame", "segment_cache": True}
        self.assertIs(lower_video_trim("video-editing/cut", cached, "a", "o")["segment_cache"], True)

    def test_subtract_spans(self):
        keep, removed = subtract_spans([[0.0, 10.0], [12.0, 20.0]], [{"start": 2, "end": 3, "label": "retake"}, {"start": 2.5, "end": 4, "label": "silence"},
                                                                    {"start": 9, "end": 13, "label": "static"}, {"start": 30, "end": 31, "label": "retake"}])
        self.assertEqual(keep, [[0.0, 2.0], [4.0, 9.0], [13.0, 20.0]])
        self.assertEqual(removed, [{"start": 2, "end": 4, "label": "retake"}, {"start": 9, "end": 10, "label": "static"}, {"start": 12, "end": 13, "label": "static"}])


class ProposalsTests(unittest.TestCase):
    def test_check_proposals(self):
        good = {"schema": "video-editing/proposals@1", "applied": False, "skill": {"id": "video-editing", "version": "0.5.0"},
                "proposals": [{"id": "p1", "kind": "span", "label": "silence", "action": "cut", "start": {"seconds": "4.000000"}, "end": {"seconds": "6.0"}, "reason": "r", "confidence": 1}]}
        self.assertEqual(check_proposals(good, "0.5.0"), [])
        self.assertIn("applied", check_proposals(dict(good, applied=True))[0])
        twice = dict(good, proposals=good["proposals"] * 2)
        self.assertTrue(any("repeated" in e for e in check_proposals(twice)))
        back = dict(good, proposals=[dict(good["proposals"][0], start=7)])
        self.assertTrue(any("before end" in e for e in check_proposals(back)))
        self.assertTrue(check_proposals({"schema": "x"}))


class PlanTests(unittest.TestCase):
    """Requirement → decision → IR → plan step → request on the fake engine and the fake video-editing-skill."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.a = fake_media(self.tmp, "a.mp4")
        self.b = fake_media(self.tmp, "b.mp4")
        Path(self.b).write_bytes(b"\x01" * 64)   # another file (the sha256 tells the inputs apart)
        os.environ.pop("FAKE_VE_MODE", None)

    def _svc(self, caps=None):
        from video_agent.tools import ToolRouter
        from video_agent.tools.video_editing import VideoEditingAdapter, VideoEditingSkill
        ad = VideoEditingAdapter(VideoEditingSkill([sys.executable, FAKE_VE], None, {}), workspace=self.tmp, allowed_inputs=[str(Path(self.a).parent)], ffmpeg_skill_dir=self.tmp)
        return make_service(self.tmp, caps=caps or FakeCaps(extra=["video-editing", "encoder:aac"]), adapter=ToolRouter([FakeAdapter(), ad]))

    def _render(self, svc, ir, name):
        p = str(Path(self.tmp) / f"{name}.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        prov = json.loads((Path(self.tmp) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        return out, prov

    def test_requirement_vocabulary_and_refusals(self):
        box = {"x": 600, "y": 200, "width": 160, "height": 160}
        got = parse_edit_requirements(reqs({"edit.punch_in": "2-4.5", "edit.punch_in.scale": 2, "edit.punch_in.subject": box}))["video.punch_in"]["params"]
        self.assertEqual(got, {"start": 2.0, "end": 4.5, "scale": 2.0, "subject": box})
        for bad, msg in (({"edit.punch_in": "2-4"}, "needs edit.punch_in.scale"), ({"edit.punch_in": "2-4", "edit.punch_in.scale": 1, "edit.punch_in.subject": box}, "above 1"),
                         ({"edit.punch_in": "4-2", "edit.punch_in.scale": 2, "edit.punch_in.subject": box}, "before end"),
                         ({"edit.punch_in": "2-4", "edit.punch_in.scale": 2, "edit.punch_in.subject": {"x": 1.5, "y": 0, "width": 4, "height": 4}}, "whole number"),
                         ({"edit.punch_in.scale": 2}, "ambiguous"), ({"edit.condense.approve": "p1"}, "ambiguous"),
                         ({"edit.concat": True, "edit.concat.stinger": self.a}, "cut_point"),
                         ({"edit.concat": True, "edit.concat.stinger": self.a, "edit.concat.stinger.cut_point": 2, "edit.concat.stinger.cover": "0.2-1"}, "inside"),
                         ({"edit.concat": True, "edit.concat.stinger": self.a, "edit.concat.stinger.cut_point": 0.5, "edit.concat.stinger.cover": "0.2-1",
                           "edit.concat.transition": "fade"}, "leave edit.concat.transition"),
                         ({"edit.ai_generated": [{"input": self.a, "generator": "g"}]}, "input, generator, licence"),
                         ({"edit.ai_generated": [{"input": self.a, "generator": "g", "licence": "l"}, {"input": self.a, "generator": "h", "licence": "l"}]}, "twice")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                parse_edit_requirements(reqs(bad))
            self.assertIn(msg, str(cm.exception))
        _check_requirement_keys({"edit.precision": "auto", "edit.segment_cache": False})
        _check_requirement_keys({"edit.precision": "smart"})   # ADR-106: video-editing-skill 0.6.0
        with self.assertRaises(ValueError):
            _check_requirement_keys({"edit.precision": "lossless"})
        with self.assertRaises(ValueError):
            _check_requirement_keys({"edit.precision": "smart", "edit.segment_cache": True})
        with self.assertRaises(ValueError) as cm:
            _check_requirement_keys({"edit.precision": "auto", "edit.segment_cache": True})
        self.assertIn("frame-precision", str(cm.exception))

    def test_punch_in_end_to_end(self):
        svc = self._svc()
        box = {"x": 600, "y": 200, "width": 160, "height": 160}
        ir = svc.plan([self.a], "generic", user_requirements={"edit.punch_in": "2-4.5", "edit.punch_in.scale": 2, "edit.punch_in.subject": box, "edit.fit": "9:16"})
        d = ir.doc
        aid = list(d["assets"])[0]
        self.assertEqual((d["plan"]["status"], svc.validate(ir).errors), ("APPROVED", []), d["plan"]["summary"])
        self.assertEqual([op["type"] for op in d["video"]["operations"] if op["type"] != "video.trim"], ["video.punch_in", "video.fit"], "punch in before fit (EDIT_ORDER)")
        pi = next(op for op in d["video"]["operations"] if op["type"] == "video.punch_in")
        self.assertEqual({k: pi[k] for k in ("start", "end", "scale", "subject")}, {"start": 2.0, "end": 4.5, "scale": 2.0, "subject": box})
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "video_punch_in")
        self.assertEqual(step["tool"], "video-editing/punch_in")
        dec = next(x for x in d["decisions"] if x["subject"] == "video.punch_in")
        self.assertEqual((dec["type"], dec["basis"]["approval"]["key"]), ("TRANSFORM", "video.punch_in.approval"))
        out, prov = self._render(svc, ir, "punch")
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        row = next(e for e in prov["operations"] if e["tool"] == "video-editing/punch_in")
        self.assertEqual({k: row["args"][k] for k in ("start", "end", "scale", "subject")}, {"start": 2.0, "end": 4.5, "scale": 2.0, "subject": box})
        # past the end of the picture it zooms: the validator refuses, nothing is clamped
        ir2 = svc.plan([self.a], "generic", user_requirements={"edit.punch_in": "2-40", "edit.punch_in.scale": 2, "edit.punch_in.subject": box})
        self.assertTrue(any("after the end" in e for e in svc.validate(ir2).errors))

    def test_propose_then_condense_only_the_approved_ids(self):
        svc = self._svc()
        res = svc.propose(self.a, silence={"min_seconds": 0.6})
        doc = res["document"]
        self.assertEqual((doc["applied"], [p["id"] for p in doc["proposals"]]), (False, ["p1"]))
        self.assertTrue(Path(res["path"]).is_file())
        ws = os.path.realpath(self.tmp)   # both sides resolved the same way (Windows 8.3 short names vs long names)
        self.assertEqual(os.path.commonpath([os.path.realpath(res["path"]), ws]), ws, "written inside the workspace")
        # nothing was cut by proposing; a plan without the approval cuts nothing of it
        with self.assertRaises(ValueError) as cm:
            svc.plan([self.a], "generic", user_requirements={"edit.condense": res["path"]})
        self.assertIn("edit.condense.approve", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            svc.plan([self.a], "generic", user_requirements={"edit.condense": res["path"], "edit.condense.approve": "p9"})
        self.assertIn("p9", str(cm.exception))
        ir = svc.plan([self.a], "generic", user_requirements={"edit.condense": res["path"], "edit.condense.approve": "p1"})
        d = ir.doc
        aid = list(d["assets"])[0]
        self.assertEqual((d["plan"]["status"], svc.validate(ir).errors), ("APPROVED", []), d["plan"]["summary"])
        trim = next(op for op in d["video"]["operations"] if op["type"] == "video.trim")
        self.assertEqual(trim["condense"], [{"start": 4.0, "end": 6.0, "label": "silence"}])
        self.assertFalse(any(s_ < 6.0 and e_ > 4.0 for s_, e_ in trim["keep"]), trim["keep"])
        self.assertTrue(trim["accurate"])
        step = next(s for s in d["plan"]["steps"] if s.get("outputs") == [f"{aid}_trim"])
        self.assertEqual((step["skill"], step["tool"]), ("video_condense", "video-editing/condense"))
        dec = next(x for x in d["decisions"] if x["subject"] == "video.condense")
        self.assertEqual((dec["type"], dec["params"]["approve"]), ("TRANSFORM", ["p1"])); self.assertIn(dec["id"], trim["decision_ids"])
        out, prov = self._render(svc, ir, "condense")
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        row = next(e for e in prov["operations"] if e["tool"] == "video-editing/condense")
        self.assertEqual((row["args"]["default"], row["args"]["rules"]["silence"], row["args"]["rules"]["keep"]), ("cut", "cut", "keep"))
        self.assertIn({"start": 4.0, "end": 6.0, "label": "silence"}, row["args"]["segments"])
        # proposals measured on another file are never applied to this one
        ir3 = svc.plan([self.b], "generic", user_requirements={"edit.condense": res["path"], "edit.condense.approve": "p1"})
        blk = next(x for x in ir3.doc["decisions"] if x["subject"] == "video.condense")
        self.assertEqual(blk["approval"], "BLOCK"); self.assertIn("not a video input", blk["reason"])

    def test_precision_auto_and_segment_cache(self):
        svc = self._svc()
        ir = svc.plan([self.a], "generic", user_requirements={"edit.trim": "1-9", "edit.precision": "auto"})
        trim = next(op for op in ir.doc["video"]["operations"] if op["type"] == "video.trim")
        self.assertEqual((trim["precision"], trim["accurate"]), ("auto", False))
        step = next(s for s in ir.doc["plan"]["steps"] if s["skill"] in ("video_trim", "video_cut"))
        self.assertEqual(step["tool"], f"video-editing/{'trim' if len(trim['keep']) == 1 else 'cut'}")
        self.assertEqual(svc.validate(ir).errors, [])
        out, prov = self._render(svc, ir, "auto")
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        self.assertEqual(next(e for e in prov["operations"] if e["tool"].startswith("video-editing/"))["args"]["precision"], "auto")
        # without an explicit precision nothing changes: the default trim keeps its engine
        ir0 = svc.plan([self.a], "generic", user_requirements={})
        self.assertFalse(any("precision" in op for op in ir0.doc["video"]["operations"]))
        ir2 = svc.plan([self.a], "generic", user_requirements={"edit.trim": "1-9", "edit.segment_cache": True})
        trim = next(op for op in ir2.doc["video"]["operations"] if op["type"] == "video.trim")
        self.assertEqual((trim["precision"], trim["segment_cache"], trim["accurate"]), ("frame", True, True))
        step = next(s for s in ir2.doc["plan"]["steps"] if s["skill"] == "video_cut")
        self.assertEqual(step["tool"], "video-editing/cut")
        out, prov = self._render(svc, ir2, "cache")
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = next(r for r in out["execution"]["results"] if r["tool"] == "video-editing/cut")
        self.assertEqual(res["data"]["reported"]["segments"]["join"], "stream_copy")

    def test_precision_smart_only_when_asked(self):
        """ADR-106: edit.precision=smart runs the asset's trim as video-editing TRIM / CUT precision smart (frame-accurate, so the plan keeps
        accurate true); the Skill's cut report (cuts.smart, cuts.fallbacks) and plan.steps[].smart are carried as reported. An ffmpeg-skill
        without cut --smart: the Skill falls back to --accurate and says so (engine_without_smart, a warning); the render still completes."""
        svc = self._svc()
        ir = svc.plan([self.a], "generic", user_requirements={"edit.trim": "1-9", "edit.precision": "smart"})
        trim = next(op for op in ir.doc["video"]["operations"] if op["type"] == "video.trim")
        self.assertEqual((trim["precision"], trim["accurate"]), ("smart", True))
        step = next(s for s in ir.doc["plan"]["steps"] if s["skill"] in ("video_trim", "video_cut"))
        self.assertEqual((step["tool"], step["params"]["precision"]), ("video-editing/trim", "smart"))
        self.assertEqual(svc.validate(ir).errors, [])
        out, prov = self._render(svc, ir, "smart")
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        self.assertEqual(next(e for e in prov["operations"] if e["tool"].startswith("video-editing/"))["args"]["precision"], "smart")
        res = next(r for r in out["execution"]["results"] if r["tool"] == "video-editing/trim")
        cuts = res["data"]["reported"]["cuts"]
        self.assertEqual((cuts["requested_precision"], cuts["mode"], cuts["smart"]["applied"], cuts["fallbacks"]), ("smart", "smart", True, []))
        os.environ["FAKE_VE_MODE"] = "no_smart"
        try:
            out, _ = self._render(svc, ir, "smart_fallback")
        finally:
            os.environ.pop("FAKE_VE_MODE", None)
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = next(r for r in out["execution"]["results"] if r["tool"] == "video-editing/trim")
        self.assertEqual((res["data"]["reported"]["cuts"]["mode"], res["data"]["reported"]["cuts"]["smart"]["reason_code"]), ("accurate", "engine_without_smart"))
        self.assertTrue(any("precision smart" in w for w in res["data"]["warnings"]))
        # the dry run carries plan.steps[].smart (the engine flag the step will use)
        from video_agent.models import Operation
        ad = svc.adapter().adapter_for("video-editing/trim")
        dr = ad.run(Operation(tool="video-editing/trim", args={"input": "a", "start": 1, "end": 3, "precision": "smart", "output": "o"}, inputs=["a"], outputs=["o"], id="op_dry"),
                    {"a": self.a, "o": str(Path(self.tmp) / "dry" / "o.mp4")}, timeout=30, dry_run=True)
        self.assertTrue(dr.ok, dr.data)
        self.assertEqual(dr.data["plan"]["steps"][0]["smart"]["engine_flag"], "--smart")
        # without an explicit precision nothing changes (the default never picks smart)
        ir0 = svc.plan([self.a], "generic", user_requirements={"edit.trim": "1-9"})
        self.assertNotEqual(next(op for op in ir0.doc["video"]["operations"] if op["type"] == "video.trim").get("precision"), "smart")

    def test_stinger_and_ai_generated_concat(self):
        svc = self._svc()
        st = Path(self.tmp) / "src" / "stinger.mov"
        st.write_bytes(b"\x02" * 64)
        ir = svc.plan([self.a, self.b], "generic", user_requirements={
            "edit.concat": True, "edit.concat.stinger": str(st), "edit.concat.stinger.cut_point": 0.5, "edit.concat.stinger.cover": "0.4-0.8",
            "edit.ai_generated": [{"input": self.b, "generator": "a video model", "licence": "CC BY 4.0"}]})
        d = ir.doc
        self.assertEqual((d["plan"]["status"], svc.validate(ir).errors), ("APPROVED", []), d["plan"]["summary"])
        a_id, b_id = list(d["assets"])
        cat = next(op for op in d["video"]["operations"] if op["type"] == "video.concat")
        self.assertEqual(cat["transition"], {"type": "stinger", "cut_point": 0.5, "cover": {"start": 0.4, "end": 0.8}})
        self.assertEqual(cat["stinger"], os.path.abspath(str(st)), "the user's path as given, made absolute (like an overlay image)")
        self.assertEqual((cat["width"], cat["height"]), (1280, 720), "the generated input is conformed to the other input's frame")
        segs = cat["segments"]
        self.assertEqual(segs[1]["timeline_range"][0], segs[0]["timeline_range"][1], "a stinger butt-joins the clips (no overlap)")
        self.assertEqual(d["video"]["ai_generated"][b_id]["generator"], "a video model")
        self.assertIn(str(st.resolve().parent), d["execution"]["allowed_inputs"])
        dec = next(x for x in d["decisions"] if x["subject"] == "media.ai_generated")
        self.assertEqual((dec["type"], dec["approval"], dec["params"]["asset_id"]), ("KEEP", "AUTO", b_id))
        out, prov = self._render(svc, ir, "stinger")
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        row = next(e for e in prov["operations"] if e["tool"] == "video-editing/concat")
        self.assertEqual(row["args"]["transition"]["type"], "stinger"); self.assertIn("stinger", row["args"])
        self.assertEqual(list(row["args"]["ai_generated"].values()), [{"generator": "a video model", "licence": "CC BY 4.0"}])
        res = next(r for r in out["execution"]["results"] if r["tool"] == "video-editing/concat")
        self.assertEqual(res["data"]["marks"][0]["mark"], "AI_GENERATED")
        # a declaration naming a file that is not an input is refused, never dropped
        with self.assertRaises(ValueError):
            svc.plan([self.a], "generic", user_requirements={"edit.ai_generated": [{"input": str(st), "generator": "g", "licence": "l"}]})


if __name__ == "__main__":
    unittest.main()
