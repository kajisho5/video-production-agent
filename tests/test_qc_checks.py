"""ADR-112: the opt-in qc-skill checks of what the agent joined -- a stinger's cut point after a stinger CONCAT (qc-skill 1.2.0) and the seams
of a music bed the agent looped (qc-skill 1.3.0) -- against the fake qc / video-editing / sound-design / audio-production processes and the
fake engine: the requirement keys and their refusals, the IR records and the plan hash, the validator, the qc adapter's stinger file
(resolved under the path policy, admitted by its fingerprint), the requests QA sends (the concat's own cut times on the delivered timeline,
seams at multiples of the loop period, no limit unless set), the QA items (each finding, UNKNOWN with its reason, refusals, no qc-skill)
and the plan diff. No ffmpeg, no real Skill (tests/test_adapter_qc.py's QcRealSkillTests runs both on the real one)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import PipelineCaps, clear_modes, fake_media, pipeline_service  # noqa: E402
from test_bgm import AP_MIX, SdCaps, library, sd_adapter, sd_capabilities  # noqa: E402

from video_agent.agent import qc_checks  # noqa: E402
from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.models import Operation  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.service import BRIEF_REQUIREMENT_KEYS, _check_requirement_keys  # noqa: E402
from video_agent.tools.base import ToolError  # noqa: E402
from video_agent.tools.qc import QcAdapter  # noqa: E402
from video_agent.tools.skill_process import CliSkill  # noqa: E402

FAKE_QC = str(Path(__file__).resolve().parent / "fake_qc.py")
ENV = ("FAKE_QC_STINGER", "FAKE_QC_CUT_DIFF", "FAKE_QC_SEAM", "FAKE_QC_SEAM_CLICK", "FAKE_QC_SEAM_LEVEL", "FAKE_QC_SEAM_BAND", "FAKE_QC_REQUESTS",
       "FAKE_SD_MODE", "FAKE_SD_READY", "FAKE_SD_CALLS")


def sha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class _Base(unittest.TestCase):
    def setUp(self):
        clear_modes()
        for k in ENV:
            os.environ.pop(k, None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.requests = str(Path(self.tmp) / "qc_requests.log")
        os.environ["FAKE_QC_REQUESTS"] = self.requests

    def tearDown(self):
        clear_modes()
        for k in ENV:
            os.environ.pop(k, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def qc_requests(self):
        return [json.loads(x) for x in Path(self.requests).read_text(encoding="utf-8").splitlines()] if os.path.exists(self.requests) else []

    def qc_items(self, out, prefix=""):
        return [i for i in out["qa"]["items"] if i["layer"] == "qc" and i["name"].startswith(prefix)]


class RequirementTests(_Base):
    def test_keys_and_refusals(self):
        st = str(Path(self.tmp) / "stinger.mov")
        self.assertEqual(qc_checks.parse_qc_check_requirements({}), {"stinger": None, "loop_seam": None})
        got = qc_checks.parse_qc_check_requirements({"qc.stinger": "true", "qc.stinger.max_cut_frame_difference": "4", "edit.concat.stinger": st})
        self.assertEqual(got["stinger"], {"rule": "qc_stinger@1", "max_cut_frame_difference": 4.0})
        self.assertEqual(qc_checks.parse_qc_check_requirements({"qc.stinger": True, "edit.concat.stinger": st})["stinger"], {"rule": "qc_stinger@1"},
                         "no default for max_cut_frame_difference: the cut-frame check is not asked for")
        self.assertIsNone(qc_checks.parse_qc_check_requirements({"qc.stinger": False, "edit.concat.stinger": st})["stinger"])
        got = qc_checks.parse_qc_check_requirements({"qc.loop_seam.max_click_ratio": 3, "qc.loop_seam.max_level_diff_db": "1.5", "qc.loop_seam.window_sec": 1, "bgm.library": "/x"})
        self.assertEqual(got["loop_seam"], {"rule": "qc_loop_seam@1", "max_click_ratio": 3.0, "max_level_diff_db": 1.5, "window_sec": 1.0})
        for bad, msg in (({"qc.stinger": True}, "edit.concat.stinger"), ({"qc.stinger.max_cut_frame_difference": 3, "edit.concat.stinger": st}, "qc.stinger=true"),
                         ({"qc.stinger": True, "qc.stinger.max_cut_frame_difference": 300, "edit.concat.stinger": st}, "within"),
                         ({"qc.stinger": "maybe", "edit.concat.stinger": st}, "true or false"),
                         ({"qc.loop_seam.max_click_ratio": 3}, "bgm.library"), ({"qc.loop_seam.max_level_diff_db": 1, "bgm.library": "/x"}, "window_sec"),
                         ({"qc.loop_seam.window_sec": 1, "bgm.library": "/x"}, "a window alone"), ({"qc.loop_seam.max_click_ratio": 3, "qc.loop_seam.window_sec": 1, "bgm.library": "/x"}, "a window alone"),
                         ({"qc.loop_seam.max_click_ratio": -1, "bgm.library": "/x"}, "within"), ({"qc.loop_seam.max_click_ratio": "nan", "bgm.library": "/x"}, "finite"),
                         ({"qc.loop_seam.max_band_diff_db": 6, "qc.loop_seam.window_sec": 0, "bgm.library": "/x"}, "within")):
            with self.subTest(bad=bad):
                with self.assertRaises(EditRequirementError) as cm:
                    qc_checks.parse_qc_check_requirements(bad)
                self.assertIn(msg, str(cm.exception))

    def test_the_set_gate_knows_the_keys_and_a_brief_never_sets_them(self):
        lib = library(self.tmp)
        _check_requirement_keys({"audio.production": True, "audio.mix": True, "bgm.library": lib, "qc.loop_seam.max_click_ratio": 3})
        _check_requirement_keys({"qc.loop_seam.max_level_diff_db": 2})   # the gate checks each value; how keys go together is checked on the effective keys
        for bad in ({"qc.loop_seam.max_click": 3}, {"qc.loop_seam.max_click_ratio": 5000}, {"qc.stinger": "sometimes"}, {"qc.loop_seam.window_sec": 0}):
            with self.assertRaises(ValueError, msg=bad):
                _check_requirement_keys(bad)
        svc = pipeline_service(self.tmp, transcription=False)
        voice = fake_media(self.tmp, name="v.wav", video=False)
        with self.assertRaises(ValueError) as cm:   # refused before any analysis: nothing to check
            svc.plan([voice], "generic", user_requirements={"qc.loop_seam.max_click_ratio": 3})
        self.assertIn("bgm.library", str(cm.exception))
        self.assertFalse(set(qc_checks.REQUIREMENT_KEYS) & set(BRIEF_REQUIREMENT_KEYS), "QA limits are policy: only an explicit --set states them")

    def test_runtime_helpers(self):
        self.assertEqual(qc_checks.seconds_of({"seconds": "2.000000", "rational": "1001/500"}), 2.002)
        self.assertEqual(qc_checks.seconds_of({"seconds": "1.500000"}), 1.5)
        self.assertIsNone(qc_checks.seconds_of({"seconds": "x"})); self.assertIsNone(qc_checks.seconds_of(float("nan"))); self.assertIsNone(qc_checks.seconds_of(True))
        self.assertEqual(qc_checks.joins_cut_at({"reported": {"stinger": {"joins": [{"cut_at": {"seconds": "4.000000"}}, {"cut_at": {"seconds": "9.5"}}]}}}), [4.0, 9.5])
        self.assertIsNone(qc_checks.joins_cut_at({"reported": {"stinger": {"joins": None}}}))
        self.assertIsNone(qc_checks.joins_cut_at({"reported": {"stinger": {"joins": [{"cut_at": None}]}}}))
        self.assertEqual(qc_checks.delivered_times([4.0, 9.0], 2.0, 3.0), [5.0, 7.5], "a 3 s still before, then twice as fast")
        fit = {"mode": "loop", "source": 6.0, "crossfade": 0.5, "repeats": 4, "target": 20.0}
        self.assertEqual(qc_checks.loop_seams(fit), [5.5, 11.0, 16.5])
        self.assertEqual(qc_checks.loop_seams(dict(fit, target=11.0)), [5.5], "a seam at the bed's very end is not inside it")
        self.assertEqual(qc_checks.loop_seams(dict(fit, crossfade=0.0, repeats=2, target=None), 12.0), [6.0])
        self.assertEqual(qc_checks.loop_seams(dict(fit, target=5.0)), [], "a loop shorter than one period has no seam inside it")
        for not_looped in ({"mode": "trim", "source": 30.0, "target": 16.0}, {"mode": "none"}, None):
            self.assertIsNone(qc_checks.loop_seams(not_looped), not_looped)
        for unreadable in (dict(fit, repeats=1), dict(fit, crossfade=6.0), dict(fit, source=None), dict(fit, repeats="3"), dict(fit, target=None)):
            with self.assertRaises(ValueError, msg=unreadable):   # the seams are unknown, which is not "no seam"
                qc_checks.loop_seams(unreadable)
        self.assertEqual(qc_checks.loop_rules([5.5], {"max_click_ratio": 3, "rule": "qc_loop_seam@1"}),
                         {"audio": {"loop_seam": {"seams": [5.5], "include_wrap": False, "max_click_ratio": 3.0}}})


class AdapterTests(_Base):
    """The qc adapter's stinger `file`: resolved like an input (inside the allowed roots), sent as an absolute path, admitted only when the
    report's provenance.stinger_input carries that file's sha256."""

    def setUp(self):
        super().setUp()
        self.src_dir = str(Path(self.tmp) / "src"); os.makedirs(self.src_dir)
        self.video = fake_media(self.tmp, "delivered.mp4")
        self.stinger = str(Path(self.src_dir) / "stinger.mov"); Path(self.stinger).write_bytes(b"\x02" * 64)
        self.ad = QcAdapter(CliSkill("qc", [sys.executable, FAKE_QC], None, {}), workspace=str(Path(self.tmp) / "ws"), allowed_inputs=[self.src_dir])

    def run_check(self, rules, kind="video"):
        return self.ad.run(Operation(tool="qc/check", args={"input": "v", "kind": kind, "rules": rules}, inputs=["v"], outputs=[], kind="qa", id="op_st"),
                           {"v": self.video, "st": self.stinger})

    def test_stinger_file_resolved_sent_and_admitted(self):
        rules = {"video": {"stinger": {"cut_point": 0.5, "cover": {"start": 0.4, "end": 0.8}, "file": "st", "cut_at": [4.0], "max_cut_frame_difference": 5}}}
        r = self.run_check(rules)
        self.assertTrue(r.ok, r.data)
        [req] = self.qc_requests()
        self.assertEqual(req["rules"]["video"]["stinger"]["file"], self.stinger, "the artifact id becomes the resolved absolute path")
        self.assertEqual(r.data["companions"]["stinger"], sha(self.stinger))
        self.assertEqual({c["check_id"]: c["status"] for c in r.data["checks"] if c["check_id"].startswith("video.stinger_")},
                         {"video.stinger_cut_point_in_cover": "PASS", "video.stinger_opaque_at_cut_point": "PASS", "video.stinger_cut_frame_matches": "PASS"})
        os.environ["FAKE_QC_STINGER"] = "unknown"
        r = self.run_check(rules)
        c = next(c for c in r.data["checks"] if c["check_id"] == "video.stinger_opaque_at_cut_point")
        self.assertEqual((c["status"], c["reason"]), ("UNKNOWN", "the stream's pixel format yuv420p has no alpha channel"), "the Skill's reason is kept")

    def test_stinger_file_refusals(self):
        outside = str(Path(self.tmp) / "elsewhere" / "stinger.mov"); os.makedirs(os.path.dirname(outside)); Path(outside).write_bytes(b"\x03" * 8)
        for rules, msg in (({"video": {"stinger": {"cut_point": 0.5, "file": outside}}}, "outside the allowed roots"),
                           ({"video": {"stinger": {"cut_point": 0.5, "file": str(Path(self.src_dir) / "none.mov")}}}, "not found")):
            r = self.run_check(rules)
            self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_REQUEST")); self.assertIn(msg, r.data["error"]["message"])
        r = self.run_check({"video": {"stinger": {"cut_point": 0.5, "file": "st"}}}, kind="audio")
        self.assertIn("own video rule", r.data["error"]["message"])
        r = self.run_check({"delivery": {"video": {"stinger": {"cut_point": 0.5, "file": "st"}}}}, kind="video")
        self.assertIn("own video rule", r.data["error"]["message"], "a stinger file the Skill would not read is never sent unresolved")
        with self.assertRaises(ToolError):
            self.ad.build_request("qc/check", {"kind": "delivery_package", "artifacts": [{"artifact_id": "v", "artifact_type": "video", "input": "v"}],
                                               "rules": {"delivery_package": {"artifacts": [{"artifact_id": "v", "video": {"stinger": {"cut_point": 0.5, "file": "st"}}}]}}},
                                  {"v": self.video, "st": self.stinger})
        self.assertEqual(self.qc_requests(), [], "every refusal comes before a process")

    def test_a_report_without_the_stinger_fingerprint_is_not_admitted(self):
        os.environ["FAKE_QC_MODE"] = "hash_mismatch"
        r = self.run_check({"video": {"stinger": {"cut_point": 0.5, "file": "st"}}})
        self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_RESULT"))
        b = self.ad.build_request("qc/check", {"input": "v", "kind": "delivery", "rules": {"delivery": {"video": {"stinger": {"cut_point": 0.5, "file": "st"}}}}},
                                  {"v": self.video, "st": self.stinger})
        self.assertEqual((b["rule_files"], b["request"]["rules"]["delivery"]["video"]["stinger"]["file"]), ({"stinger": self.stinger}, self.stinger))
        self.assertIn(self.src_dir, b["input_dirs"])


class StingerPipelineTests(_Base):
    def setUp(self):
        super().setUp()
        self.a = fake_media(self.tmp, "a.mp4")
        self.b = fake_media(self.tmp, "b.mp4", take="b")
        self.st = Path(self.tmp) / "src" / "stinger.mov"
        self.st.write_bytes(b"\x02" * 64)

    def reqs(self, **extra):
        return dict({"edit.concat": True, "edit.concat.stinger": str(self.st), "edit.concat.stinger.cut_point": 0.5, "edit.concat.stinger.cover": "0.4-0.8"}, **extra)

    def render(self, reqs, svc=None, name="p"):
        svc = svc or pipeline_service(self.tmp, transcription=False)
        ir = svc.plan([self.a, self.b], "youtube", user_requirements=reqs)
        p = str(Path(self.tmp) / f"{name}.json"); save_ir(ir, p)
        return svc, ir, p, svc.render(load_ir(p), p, approve=["all"])

    def test_off_by_default(self):
        svc, ir, _p, out = self.render(self.reqs())
        self.assertNotIn("stinger_check", ir.doc["qa"])
        self.assertEqual(self.qc_items(out, "video.stinger"), [])
        self.assertFalse(any("stinger" in json.dumps(r.get("rules") or {}) for r in self.qc_requests()))

    def test_one_check_per_deliverable_with_the_concats_own_cut_times(self):
        svc, ir, _p, out = self.render(self.reqs(**{"qc.stinger": True, "qc.stinger.max_cut_frame_difference": 5}))
        d = ir.doc
        self.assertEqual(d["qa"]["stinger_check"], {"rule": "qc_stinger@1", "max_cut_frame_difference": 5.0})
        self.assertEqual(svc.validate(ir).errors, [])
        self.assertEqual(out["status"], "COMPLETED", out.get("qa"))
        res = next(r for r in out["execution"]["results"] if r["tool"] == "video-editing/concat")
        reported = [float(j["cut_at"]["seconds"]) for j in res["data"]["reported"]["stinger"]["joins"]]
        [req] = [r for r in self.qc_requests() if "stinger" in json.dumps(r.get("rules") or {})]
        rule = req["rules"]["video"]["stinger"]
        self.assertEqual((req["kind"], rule["cut_point"], rule["cover"], rule["file"], rule["cut_at"], rule["max_cut_frame_difference"]),
                         ("video", 0.5, {"start": 0.4, "end": 0.8}, str(self.st.resolve()), reported, 5.0))
        self.assertTrue(req["input"].endswith("programme_youtube.mp4"), "the delivered file, not the concat's intermediate")
        items = {i["name"]: i for i in self.qc_items(out, "video.stinger")}
        self.assertEqual({k: v["status"] for k, v in items.items()},
                         {"video.stinger_cut_point_in_cover": "PASS", "video.stinger_opaque_at_cut_point": "PASS", "video.stinger_cut_frame_matches": "PASS"})
        self.assertTrue(all(i["artifact"] == "programme_delivery_youtube" for i in items.values()))
        art = next(a for a in out["artifacts"] if a["logical_name"] == "programme_delivery_youtube")
        self.assertIn("video.stinger_opaque_at_cut_point", [i["name"] for i in art["qa"]["items"]])

    def test_no_limit_no_cut_frame_check(self):
        _svc, _ir, _p, out = self.render(self.reqs(**{"qc.stinger": True}))
        [req] = [r for r in self.qc_requests() if "stinger" in json.dumps(r.get("rules") or {})]
        self.assertNotIn("max_cut_frame_difference", req["rules"]["video"]["stinger"], "no default: the sub-check is not run")
        self.assertIn("cut_at", req["rules"]["video"]["stinger"], "the cut frames are still measured and reported")
        self.assertNotIn("video.stinger_cut_frame_matches", [i["name"] for i in self.qc_items(out)])

    def test_findings_and_unknown_reach_qa(self):
        for env, name, status, code in ((("FAKE_QC_STINGER", "not_opaque"),), "video.stinger_opaque_at_cut_point", "FAIL", "VIDEO_STINGER_NOT_OPAQUE"), \
                                       ((("FAKE_QC_CUT_DIFF", "40.3"),), "video.stinger_cut_frame_matches", "FAIL", "VIDEO_STINGER_CUT_FRAME_MISMATCH"), \
                                       ((("FAKE_QC_STINGER", "unknown"),), "video.stinger_opaque_at_cut_point", "WARN", "UNKNOWN: the stream's pixel format yuv420p has no alpha channel"):
            with self.subTest(code=code):
                for k, v in env:
                    os.environ[k] = v
                _svc, _ir, _p, out = self.render(self.reqs(**{"qc.stinger": True, "qc.stinger.max_cut_frame_difference": 5}), name=code[:12])
                for k, _v in env:
                    os.environ.pop(k)
                item = next(i for i in self.qc_items(out) if i["name"] == name)
                self.assertEqual(item["status"], status); self.assertIn(code, item["observed"])
                if status == "FAIL":
                    self.assertEqual(out["status"], "REVIEW", "a failing stinger check is a QA FAIL: the job waits for a person, nothing is re-edited")
                    self.assertEqual(next(a for a in out["artifacts"] if a["logical_name"] == "programme_delivery_youtube")["stage"], "working")

    def test_the_cover_statement_contradicting_itself(self):
        """The plan refuses a cut point outside its cover; a tampered IR that gets one past the plan reaches qc-skill as it is and FAILs there."""
        svc = pipeline_service(self.tmp, transcription=False)
        ir = svc.plan([self.a, self.b], "youtube", user_requirements=self.reqs(**{"qc.stinger": True}))
        cat = next(op for op in ir.doc["video"]["operations"] if op["type"] == "video.concat")
        cat["transition"]["cover"] = {"start": 0.6, "end": 0.8}
        from video_agent.qa.checks import stinger_items
        from video_agent.agent.editing import delivery_subjects
        subject = next(s for s in delivery_subjects(ir.doc) if s["id"] == "programme")
        ad = QcAdapter(CliSkill("qc", [sys.executable, FAKE_QC], None, {}), workspace=str(Path(self.tmp) / "ws"), allowed_inputs=[str(self.st.parent)])
        items = stinger_items(ir.doc, subject, "art", self.a, {}, [], {"rule": "qc_stinger@1"}, {"qc_check": "qc/check"},
                              lambda tool, args: ad.measure(tool, args))
        item = next(i for i in items if i.name == "video.stinger_cut_point_in_cover")
        self.assertEqual(item.status, "FAIL"); self.assertIn("VIDEO_STINGER_CUT_POINT_OUTSIDE_COVER", item.observed)

    def test_refusals_and_missing_pieces_are_warnings_never_silence(self):
        # no joins in the concat's report: the cut-frame check cannot be sent, and says so; the other two still run
        os.environ["FAKE_VE_MODE"] = "no_joins"
        _svc, _ir, _p, out = self.render(self.reqs(**{"qc.stinger": True, "qc.stinger.max_cut_frame_difference": 5}), name="nojoins")
        os.environ.pop("FAKE_VE_MODE")
        items = {i["name"]: i for i in self.qc_items(out, "video.stinger")}
        self.assertEqual(items["video.stinger_cut_frame_matches"]["status"], "WARN"); self.assertIn("no stinger joins", items["video.stinger_cut_frame_matches"]["observed"])
        self.assertEqual(items["video.stinger_opaque_at_cut_point"]["status"], "PASS")
        [req] = [r for r in self.qc_requests() if "stinger" in json.dumps(r.get("rules") or {})]
        self.assertNotIn("cut_at", req["rules"]["video"]["stinger"])
        # the Skill refuses (a qc-skill whose own roots do not hold the stinger): a WARN with the reason, the job completes
        os.environ["FAKE_QC_MODE"] = "tool_error_final"
        _svc, _ir, _p, out = self.render(self.reqs(**{"qc.stinger": True}), name="refused")
        os.environ.pop("FAKE_QC_MODE")
        [item] = self.qc_items(out, "stinger")
        self.assertEqual(item["status"], "WARN"); self.assertIn("TOOL_ERROR", item["observed"])
        # no qc-skill here at all
        svc = pipeline_service(self.tmp, transcription=False, caps=PipelineCaps(missing=["qc"]))
        _svc, _ir, _p, out = self.render(self.reqs(**{"qc.stinger": True}), svc=svc, name="noqc")
        [item] = self.qc_items(out, "stinger")
        self.assertEqual(item["status"], "WARN"); self.assertIn("not installed", item["observed"])
        self.assertEqual(out["status"], "COMPLETED")

    def test_validator_and_diff(self):
        svc = pipeline_service(self.tmp, transcription=False)
        ir = svc.plan([self.a, self.b], "youtube", user_requirements=self.reqs(**{"qc.stinger": True}))
        base_hash = svc.plan([self.a, self.b], "youtube", user_requirements=self.reqs()).doc["provenance"]["plan_hash"]
        self.assertNotEqual(ir.doc["provenance"]["plan_hash"], base_hash, "the check is part of the plan")
        p = str(Path(self.tmp) / "v.json"); save_ir(ir, p)
        out = svc.revise(load_ir(p), p, user_requirements={"qc.stinger.max_cut_frame_difference": 3})
        self.assertIn("QC stinger check: limits changed", out["diff"]["summary"])
        self.assertEqual(load_ir(p).doc["qa"]["stinger_check"]["max_cut_frame_difference"], 3.0)
        bad = load_ir(p)
        bad.doc["qa"]["stinger_check"]["max_cut_frame_difference"] = 999
        self.assertTrue(any("0..255" in e for e in svc.validate(bad).errors))
        bad.doc["qa"]["stinger_check"] = {"rule": "qc_stinger@1"}
        bad.doc["video"]["operations"] = [op for op in bad.doc["video"]["operations"] if op["type"] != "video.concat"]
        self.assertTrue(any("no CONCAT with a stinger" in e for e in svc.validate(bad).errors))


class LoopSeamPipelineTests(_Base):
    def setUp(self):
        super().setUp()
        os.environ["FAKE_SD_CALLS"] = str(Path(self.tmp) / "sd_calls.log")
        self.lib = library(self.tmp)
        self.voice = fake_media(self.tmp, name="voice.wav", video=False, duration=16.0)

    def svc(self, qc=True):
        caps = SdCaps(sd_capabilities(lambda: sd_adapter(self.tmp)), extra=AP_MIX, missing=() if qc else ("qc",))
        svc = pipeline_service(self.tmp, caps=caps, transcription=False)
        svc._adapter.register(sd_adapter(self.tmp))
        svc.adapter([])
        return svc

    def render(self, extra, qc=True, name="p"):
        svc = self.svc(qc)
        ir = svc.plan([self.voice], "generic", user_requirements=dict({"audio.production": True, "audio.mix": True, "bgm.library": self.lib}, **extra))
        p = str(Path(self.tmp) / f"{name}.json"); save_ir(ir, p)
        return svc, ir, svc.render(load_ir(p), p, approve=["all"])

    def seam_requests(self):
        return [r for r in self.qc_requests() if "loop_seam" in json.dumps(r.get("rules") or {})]

    def test_off_by_default(self):
        _svc, ir, out = self.render({})
        self.assertNotIn("loop_seam_check", ir.doc["qa"])
        self.assertEqual((self.seam_requests(), self.qc_items(out, "audio.loop_seam")), ([], []))

    def test_seams_at_multiples_of_the_period_and_only_the_limits_set(self):
        svc, ir, out = self.render({"qc.loop_seam.max_click_ratio": 3, "qc.loop_seam.max_band_diff_db": 6, "qc.loop_seam.window_sec": 1})
        self.assertEqual(ir.doc["qa"]["loop_seam_check"], {"rule": "qc_loop_seam@1", "max_click_ratio": 3.0, "max_band_diff_db": 6.0, "window_sec": 1.0})
        self.assertEqual(svc.validate(ir).errors, [])
        self.assertEqual(out["status"], "COMPLETED", out.get("qa"))
        pick = next(r for r in out["execution"]["results"] if r["tool"] == "sound-design/library-pick")
        fit = pick["data"]["fit"]
        [req] = self.seam_requests()
        period = fit["source"] - fit["crossfade"]
        self.assertEqual(req["rules"]["audio"]["loop_seam"], {"seams": [round(k * period, 6) for k in range(1, fit["repeats"]) if k * period < 16.0], "include_wrap": False,
                                                            "max_click_ratio": 3.0, "max_band_diff_db": 6.0, "window_sec": 1.0})
        self.assertEqual((req["kind"], os.path.basename(req["input"])), ("audio", "bgm_bed.wav"))
        items = {i["name"]: i for i in self.qc_items(out, "audio.loop_seam")}
        self.assertEqual({k: v["status"] for k, v in items.items()}, {"audio.loop_seam_click_within_limit": "PASS", "audio.loop_seam_spectrum_within_limit": "PASS"},
                         "the level check was not asked for, so it is not run")
        self.assertEqual({i["artifact"] for i in items.values()}, {"programme_audio_delivery_main"}, "on the deliverable that carries the bed")

    def test_findings_and_unknown(self):
        for env, name, status, code in ((("FAKE_QC_SEAM_CLICK", "12.2"),), "audio.loop_seam_click_within_limit", "FAIL", "AUDIO_LOOP_SEAM_CLICK"), \
                                       ((("FAKE_QC_SEAM_LEVEL", "4"),), "audio.loop_seam_level_within_limit", "FAIL", "AUDIO_LOOP_SEAM_LEVEL_MISMATCH"), \
                                       ((("FAKE_QC_SEAM_BAND", "20"),), "audio.loop_seam_spectrum_within_limit", "FAIL", "AUDIO_LOOP_SEAM_SPECTRUM_MISMATCH"), \
                                       ((("FAKE_QC_SEAM", "unknown"),), "audio.loop_seam_click_within_limit", "WARN", "UNKNOWN: seam 2.0"):
            with self.subTest(code=code):
                for k, v in env:
                    os.environ[k] = v
                _svc, _ir, out = self.render({"qc.loop_seam.max_click_ratio": 3, "qc.loop_seam.max_level_diff_db": 1.5, "qc.loop_seam.max_band_diff_db": 6,
                                              "qc.loop_seam.window_sec": 1}, name=code[:14])
                for k, _v in env:
                    os.environ.pop(k)
                item = next(i for i in self.qc_items(out) if i["name"] == name)
                self.assertEqual(item["status"], status); self.assertIn(code, item["observed"])
                if status == "FAIL":
                    self.assertEqual(out["status"], "REVIEW")

    def test_a_bed_that_was_not_looped_and_no_qc_skill(self):
        from video_agent.qa.checks import loop_seam_items
        from video_agent.models import ToolResult
        bed = str(Path(self.tmp) / "bed.wav"); Path(bed).write_bytes(b"x")
        r = ToolResult("op", "sound-design/library-pick", True, 0, bed, {"op": "library-pick", "fit": {"mode": "trim", "source": 30.0, "target": 16.0}, "outputs": [{"duration": 16.0}]}, [], "", 0.1)
        cfg = {"rule": "qc_loop_seam@1", "max_click_ratio": 3.0}
        [item] = loop_seam_items("art", {"bgm_bed": bed}, [r], cfg, {"qc_check": "qc/check"}, None, {})
        self.assertEqual((item.status, item.name), ("SKIP", "loop_seam")); self.assertIn("did not loop", item.observed)
        r.data["fit"] = {"mode": "loop", "source": 6.0, "repeats": "many"}
        [item] = loop_seam_items("art", {"bgm_bed": bed}, [r], cfg, {"qc_check": "qc/check"}, None, {})
        self.assertEqual(item.status, "WARN", "a loop whose fit cannot be read is not checked, and says so"); self.assertIn("cannot be read", item.observed)
        r.data["fit"] = {"mode": "loop", "source": 6.0, "crossfade": 0.5, "repeats": 3, "target": 16.0}
        os.remove(bed)
        [item] = loop_seam_items("art", {"bgm_bed": bed}, [r], cfg, {"qc_check": "qc/check"}, None, {})
        self.assertEqual(item.status, "WARN", "a bed that cannot be read is a WARN, never a crash of QA"); self.assertIn("not checked", item.observed)
        _svc, _ir, out = self.render({"qc.loop_seam.max_click_ratio": 3}, qc=False, name="noqc")
        [item] = self.qc_items(out, "loop_seam")
        self.assertEqual(item["status"], "WARN"); self.assertIn("not installed", item["observed"])
        self.assertEqual(out["status"], "COMPLETED")

    def test_validator_and_diff(self):
        svc = self.svc()
        ir = svc.plan([self.voice], "generic", user_requirements={"audio.production": True, "audio.mix": True, "bgm.library": self.lib, "qc.loop_seam.max_click_ratio": 3})
        p = str(Path(self.tmp) / "v.json"); save_ir(ir, p)
        out = svc.revise(load_ir(p), p, user_requirements={"qc.loop_seam.max_click_ratio": 2})
        self.assertIn("QC loop-seam check: limits changed", out["diff"]["summary"])
        bad = load_ir(p)
        for tamper, msg in ((lambda q: q.update(window_sec=1.0), "only with them"), (lambda q: q.pop("max_click_ratio"), "at least one"),
                            (lambda q: q.update(max_click_ratio=-1), "0..1000")):
            doc = json.loads(json.dumps(bad.doc))
            tamper(doc["qa"]["loop_seam_check"])
            self.assertTrue(any(msg in e for e in qc_checks.check_problems(doc)), (msg, qc_checks.check_problems(doc)))


if __name__ == "__main__":
    unittest.main()
