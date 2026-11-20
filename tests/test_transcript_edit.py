"""Editing by transcript (ADR-123, Velisa ADR-032): the user's words / sentences -> transcription-skill select -> video-editing-skill ripple ->
the transcript-edit file and the lines the user approves -> `edit.transcript_edit` -> a CONFIRM `transcript.ripple` decision cut by CONDENSE ->
after the render, the transcript and subtitles retimed to the cuts the plan made. Fake engine + fake transcription + fake video-editing."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import clear_modes, fake_media, pipeline_service  # noqa: E402
from video_agent import mcp_server  # noqa: E402
from video_agent.agent import transcript_edit as te  # noqa: E402
from video_agent.agent.editing import EditRequirementError, parse_edit_requirements  # noqa: E402
from video_agent.agent.requirements import requirement_map  # noqa: E402
from video_agent.models import Requirement  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.tools.base import ToolError  # noqa: E402
from video_agent.tools.transcription import adapter as tr_adapter  # noqa: E402
from video_agent.tools.video_editing import adapter as ve_adapter  # noqa: E402

WORDS = [("We", 0.5, 0.7), ("measured", 0.8, 1.3), ("um", 1.5, 1.7), ("the", 2.0, 2.1), ("noise.", 2.15, 2.6),
         ("Sorry,", 4.0, 4.4), ("wrong", 4.5, 4.9), ("slide.", 5.0, 5.5), ("Now", 6.5, 6.8), ("results.", 6.9, 7.6)]


def transcript_doc(duration=16.0):
    segs, k = [], 0
    for sid, spk, (a, b) in (("seg_0001", "SPEAKER_00", (0, 5)), ("seg_0002", "SPEAKER_01", (5, 8)), ("seg_0003", "SPEAKER_00", (8, 10))):
        ws = [{"start": s, "end": e, "text": t, "confidence": 0.9, "index": k + i, "speaker_id": spk} for i, (t, s, e) in enumerate(WORDS[a:b])]
        k += len(ws)
        text = " ".join(w["text"] for w in ws)
        segs.append({"id": sid, "start": ws[0]["start"], "end": ws[-1]["end"], "text": text, "raw_text": text, "confidence": 0.9, "words": ws, "speaker_id": spk})
    return {"schema": "transcription-skill/transcript/0.1", "id": "tr_talk", "asset_id": "talk", "language": "en", "duration": duration, "segments": segs,
            "source": {"filename": "a.mp4", "media_duration": duration}, "warnings": []}


def reqs(d):
    return requirement_map([Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in d.items()])


class SelectionVocabularyTests(unittest.TestCase):
    def test_word_ranges_and_segment_ids(self):
        self.assertEqual(te.parse_word_ranges("12-40, 55"), [[12, 40], [55, 55]])
        self.assertEqual(te.parse_word_ranges([[1, 2]]), [[1, 2]])
        self.assertEqual(te.parse_segment_ids("seg_0003,seg_0007"), ["seg_0003", "seg_0007"])
        for bad in ("12-", "a-b", "40-12", "", [[2, 1]], [[True, 2]], 5):
            with self.assertRaises(te.TranscriptEditError, msg=repr(bad)):
                te.parse_word_ranges(bad)
        for bad in ("", "seg 1", ["../x"], 3):
            with self.assertRaises(te.TranscriptEditError, msg=repr(bad)):
                te.parse_segment_ids(bad)

    def test_removed_ranges_follow_the_cut_frames(self):
        keep = [[0.0, 1.4014], [1.9019, 2.8028], [6.4064, 7.75]]
        got = te.removed_ranges(keep, 8.975633, "30000/1001")
        self.assertEqual(got, [{"start": 1.4014, "end": 1.9019}, {"start": 2.8028, "end": 6.4064}, {"start": 7.774433, "end": 8.975633}],
                         "frame-exact edges unchanged; 7.75 s (a millisecond trim) is where the cut keeps frame 232, so the removal starts at frame 233")
        self.assertEqual(te.removed_ranges([[0.0, 2.0]], 2.0, "25"), [])
        self.assertEqual(te.removed_ranges([[1.0, 2.0]], 3.0), [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}])

    def test_mcp_tool_is_a_preview_through_the_cli(self):
        names = [t["name"] for t in mcp_server.TOOLS]
        self.assertIn("transcript_edit", names)
        tool = next(t for t in mcp_server.TOOLS if t["name"] == "transcript_edit")
        self.assertIn("Nothing is cut", tool["description"])
        self.assertIn("only after their yes", tool["description"])
        argv = mcp_server.tool_argv("transcript_edit", {"input": "a.mp4", "transcript": "t.json", "mode": "keep", "segments": "seg_0002", "subtitles": ["a.srt"],
                                                         "include_overlap": True}, "/tmp")
        self.assertEqual(argv, ["transcript-edit", "--transcript=t.json", "--keep-segments=seg_0002", "--include-overlap", "--subtitles=a.srt", "--", "a.mp4"])
        for bad in ({"input": "a.mp4", "transcript": "t.json", "mode": "cut", "words": "1"}, {"input": "a.mp4", "transcript": "t.json", "mode": "remove"},
                    {"transcript": "t.json", "mode": "remove", "words": "1"}):
            with self.assertRaises(ValueError):
                mcp_server.tool_argv("transcript_edit", bad, "/tmp")


class _Stub:
    """An adapter object with the real methods and a canned process answer."""

    def __init__(self, cls, contract, answer, **attrs):
        self.obj = cls.__new__(cls)
        self.obj.contract = contract
        self.obj.version = contract["version"]
        self.obj.tools = ({t["name"]: t for t in contract["tools"]} if cls is tr_adapter.TranscriptionAdapter else {t["tool_id"]: t for t in contract["tools"]})
        self.obj.allowed_inputs, self.obj.workspace, self.obj.default_timeout, self.obj.policy, self.obj.ffmpeg_skill_dir = [], attrs.get("workspace"), 30.0, None, None
        self.sent = []

        def invoke(argv, stdin=None, timeout=None):
            self.sent.append((argv, json.loads(stdin) if stdin else None))
            return 0, json.dumps(answer), ""
        self.obj._invoke = invoke


class AdapterTests(unittest.TestCase):
    def test_select_needs_the_tool_and_checks_the_result(self):
        c = tr_adapter.pinned_contract()
        tr = transcript_doc()
        res = {"rule": {"version": "transcript-selection@1"}, "mode": "remove", "transcript_id": "tr_talk", "applied": False,
               "spans": [{"id": "r1", "start": 1.5, "end": 1.7, "in": {"lo": 1.3, "hi": 1.5}, "out": {"lo": 1.7, "hi": 2.0}}]}
        if "transcription/select" not in [t["name"] for t in c["tools"]]:
            with self.assertRaises(ToolError):
                _Stub(tr_adapter.TranscriptionAdapter, c, {}).obj.select(tr, "remove", words=[[2, 2]])
            c = dict(c, tools=c["tools"] + [{"name": "transcription/select", "input": {}, "output": {}}])
        st = _Stub(tr_adapter.TranscriptionAdapter, c, {"ok": True, "tool": "transcription/select", "result": res})
        self.assertEqual(st.obj.select(tr, "remove", words=[[2, 2]], segments=["seg_0002"])["spans"][0]["id"], "r1")
        self.assertEqual(st.sent[0][1]["params"]["words"], [[2, 2]])
        for bad in (dict(res, applied=True), dict(res, transcript_id="other"), dict(res, spans=[dict(res["spans"][0], out={"lo": 1.0, "hi": 0.5})])):
            with self.assertRaises(ToolError):
                _Stub(tr_adapter.TranscriptionAdapter, c, {"ok": True, "tool": "transcription/select", "result": bad}).obj.select(tr, "remove", words=[[2, 2]])
        refused = {"ok": False, "error": {"code": "INVALID_INPUT", "message": "overlap", "details": {"reason": "selection_overlaps_kept_speech"}}}
        with self.assertRaises(ToolError) as cm:
            _Stub(tr_adapter.TranscriptionAdapter, c, refused).obj.select(tr, "remove", words=[[2, 2]])
        self.assertIn("selection_overlaps_kept_speech", str(cm.exception))
        with self.assertRaises(ToolError):
            st.obj.select(tr, "delete", words=[[2, 2]])

    def test_check_ripple(self):
        doc = {"schema": "video-editing/ripple-result@1", "skill": {"version": "0.9.0"}, "applied": False, "rule": {"version": "ripple@1"},
               "source": {"frames": 250}, "cuts": [{"id": "c1", "source_in": {"frame": 35}, "source_out": {"frame": 48}}],
               "events": [{"record_in": {"frame": 0}, "record_out": {"frame": 35}}, {"record_in": {"frame": 35}, "record_out": {"frame": 237}}],
               "record": {"frames": 237}, "edit_request": {"project": {"operations": [{"type": "CONDENSE"}]}}}
        self.assertEqual(ve_adapter.check_ripple(doc, "0.9.0"), [])
        self.assertTrue(ve_adapter.check_ripple(dict(doc, applied=True)))
        self.assertTrue(ve_adapter.check_ripple(dict(doc, cuts=[{"id": "c1", "source_in": {"frame": 48}, "source_out": {"frame": 35}}])))
        self.assertTrue(ve_adapter.check_ripple(dict(doc, record={"frames": 1})))
        self.assertTrue(ve_adapter.check_ripple(dict(doc, edit_request={"project": {"operations": []}})))
        self.assertTrue(ve_adapter.check_ripple(doc, "0.8.8"))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self.a = fake_media(self.tmp, "a.mp4", duration=16.0)
        self.tr = str(Path(self.tmp) / "src" / "a.transcript.json")
        Path(self.tr).write_text(json.dumps(transcript_doc()), encoding="utf-8")
        self.srt = str(Path(self.tmp) / "src" / "a.srt")
        Path(self.srt).write_text("1\n00:00:00,500 --> 00:00:02,600\nWe measured um the noise.\n", encoding="utf-8")
        self.svc = pipeline_service(self.tmp)

    def test_preview_then_plan_then_render(self):
        res = self.svc.transcript_edit(self.a, self.tr, "remove", words=[[2, 2]], segments=["seg_0002"], subtitles=[self.srt])
        doc = res["document"]
        self.assertEqual((doc["schema"], doc["applied"]), (te.DOC_SCHEMA, False))
        self.assertEqual(te.check_document(doc), [])
        self.assertEqual([c["frames"] for c in doc["cuts"]], [[35, 48], [70, 160]])
        self.assertEqual(doc["cuts"][1]["speaker_ids"], ["SPEAKER_01"])
        self.assertEqual(doc["cuts"][1]["sentences"][0]["removed"], "whole")
        self.assertTrue(any("total: 2 cut(s), 4.120s" in line for line in res["lines"]), res["lines"])
        self.assertTrue(any("SPEAKER_01" in line and "whole seg_0002" in line for line in res["lines"]), res["lines"])
        self.assertIn("only after their yes", res["next"])
        # nothing is cut by the preview: the plan carries a CONFIRM decision, and rendering waits for it
        ir = self.svc.plan([self.a], "generic", user_requirements={te.KEY: res["path"]})
        d = ir.doc
        dec = next(x for x in d["decisions"] if x["subject"] == te.SUBJECT)
        self.assertEqual((dec["type"], dec["approval"], dec["risk"]), ("TRANSFORM", "CONFIRM", "HIGH"))
        self.assertEqual([s["label"] for s in dec["params"]["spans"]], ["transcript", "transcript"])
        trim = next(op for op in d["video"]["operations"] if op["type"] == "video.trim")
        for cs, ce in ((1.4, 1.92), (2.8, 6.4)):   # the frame-exact cuts are never kept (the fake engine's leading silence 0-2.85 s is trimmed too)
            self.assertFalse(any(ks < ce and cs < ke for ks, ke in trim["keep"]), trim["keep"])
        self.assertEqual(trim["condense"], [{"start": 2.85, "end": 6.4, "label": "transcript"}], "the part of the second cut the other trims left")
        self.assertIn(dec["id"], trim["decision_ids"])
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "video_condense")
        self.assertEqual(step["tool"], "video-editing/condense")
        self.assertEqual(self.svc.validate(ir).errors, [])
        p = str(Path(self.tmp) / "te.json")
        save_ir(ir, p)
        waiting = self.svc.render(load_ir(p), p)
        self.assertEqual(waiting["status"], "WAITING_FOR_APPROVAL")
        out = self.svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        rt = out["transcript_edit"]
        self.assertEqual(rt["warnings"], [])
        item = rt["items"][0]
        self.assertEqual(item["decision"], dec["id"])
        self.assertTrue(Path(item["edl"]).is_file() and Path(item["transcript"]["path"]).is_file() and Path(item["subtitles"][0]["path"]).is_file())
        self.assertIn(str(Path(self.tmp)), item["edl"])

    def test_requirement_refusals(self):
        res = self.svc.transcript_edit(self.a, self.tr, "remove", segments=["seg_0002"])
        got = parse_edit_requirements(reqs({te.KEY: res["path"]}))[te.SUBJECT]["params"]
        self.assertEqual(got["rules"], ["transcript-selection@1", "ripple@1"])
        for bad in ("../x.json", str(Path(self.tmp) / "missing.json")):
            with self.assertRaises(EditRequirementError):
                parse_edit_requirements(reqs({te.KEY: bad}))
        doc = json.loads(Path(res["path"]).read_text(encoding="utf-8"))
        for change in ({"applied": True}, {"schema": "other@1"}, {"cuts": []}):
            p = str(Path(self.tmp) / "bad.json")
            Path(p).write_text(json.dumps(dict(doc, **change)), encoding="utf-8")
            with self.assertRaises(EditRequirementError):
                parse_edit_requirements(reqs({te.KEY: p}))
        Path(self.tr).write_text(json.dumps(transcript_doc(duration=15.0)), encoding="utf-8")   # the transcript changed since the preview
        with self.assertRaises(EditRequirementError) as cm:
            parse_edit_requirements(reqs({te.KEY: res["path"]}))
        self.assertIn("changed since the preview", str(cm.exception))

    def test_another_input_is_blocked(self):
        res = self.svc.transcript_edit(self.a, self.tr, "remove", segments=["seg_0002"])
        b = fake_media(self.tmp, "b.mp4", duration=16.0)
        Path(b).write_bytes(b"\x01" * 64)
        ir = self.svc.plan([b], "generic", user_requirements={te.KEY: res["path"]})
        blk = next(x for x in ir.doc["decisions"] if x["subject"] == te.SUBJECT)
        self.assertEqual(blk["approval"], "BLOCK")
        self.assertIn("not a video input", blk["reason"])

    def test_preview_refusals(self):
        with self.assertRaises(ValueError):
            self.svc.transcript_edit(self.a, self.tr, "remove")
        with self.assertRaises(ValueError):
            self.svc.transcript_edit(self.a, str(Path(self.tmp) / "none.json"), "remove", words=[[1, 1]])
        with self.assertRaises(ValueError):
            self.svc.transcript_edit(self.a, self.tr, "remove", words=[[0, 9]])   # nothing kept: the Skill refuses, the agent passes it on


if __name__ == "__main__":
    unittest.main()
