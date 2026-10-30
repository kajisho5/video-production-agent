"""Black-bar crop proposal (ADR-060, ECOSYSTEM_ROADMAP step 7): analysis kind `crop_rectangle` (ffmpeg-skill cropdetect.py,
fake engine here; tests/test_integration.py measures a real letterboxed clip) → OBSERVED observation → a CONFIRM decision
proposing `video.crop` with the measured rectangle → the existing video_crop step (ffmpeg-skill/crop) once approved."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter, _write_fake  # noqa: E402
from pipeline_harness import clear_modes, pipeline_service  # noqa: E402
from pipeline_harness import fake_media as pipeline_media  # noqa: E402
from test_unit import make_service  # noqa: E402
from video_agent.agent.crop import (CROP_KIND, MIN_CONFIDENCE, crop_facts, decide_crop, even_inward,  # noqa: E402
                                    stable_rectangle)
from video_agent.agent.decision_engine import DecisionEngine, resolve_approval  # noqa: E402
from video_agent.agent.editing import delivery_subjects  # noqa: E402
from video_agent.agent.finishing import picture_size  # noqa: E402
from video_agent.agent.qc import rules_for_subject  # noqa: E402
from video_agent.media.analysis import CROP_TOOL_ARGS  # noqa: E402
from video_agent.models import Asset, Intent, Observation  # noqa: E402
from video_agent.policy.rules import Rule, resolve_rules  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402

LETTERBOX = {"width": 1280, "height": 536, "x": 0, "y": 92}   # 2.39:1 picture in a 1280x720 frame


def measured(crop=LETTERBOX, confidence=1.0, windows=5, windows_failed=0, sw=1280, sh=720):
    """A cropdetect.py --json measurement as the observation keeps it (fake numbers, real key set)."""
    d = {"source_width": sw, "source_height": sh, "crop": crop, "windows": windows, "windows_failed": windows_failed}
    if crop is not None:
        d["confidence"] = confidence
    return d


class StabilityRuleTests(unittest.TestCase):
    """The pure rule on fake measurements: which measurements yield a rectangle to propose."""

    def test_stable_bars_propose_the_measured_rectangle(self):
        rect, why = stable_rectangle(crop_facts(measured()))
        self.assertEqual(rect, LETTERBOX)
        self.assertIn("stable", why)

    def test_pillarbox_is_proposed_too(self):
        rect, _ = stable_rectangle(crop_facts(measured(crop={"width": 960, "height": 720, "x": 160, "y": 0})))
        self.assertEqual(rect, {"width": 960, "height": 720, "x": 160, "y": 0})

    def test_unstable_rectangle_is_not_proposed(self):
        """One window in five disagreeing (a full-frame span inside a letterboxed film) is not a stable rectangle."""
        for conf in (0.5, 0.8, MIN_CONFIDENCE - 0.001):
            rect, why = stable_rectangle(crop_facts(measured(confidence=conf)))
            self.assertIsNone(rect, conf)
            self.assertIn("not stable", why)
        self.assertIsNotNone(stable_rectangle(crop_facts(measured(confidence=MIN_CONFIDENCE)))[0])

    def test_no_bars_is_not_proposed(self):
        rect, why = stable_rectangle(crop_facts(measured(crop={"width": 1280, "height": 720, "x": 0, "y": 0})))
        self.assertIsNone(rect)
        self.assertIn("no black bars", why)
        rect, why = stable_rectangle(crop_facts(measured(crop={"width": 1280, "height": 714, "x": 0, "y": 2})))   # 6 px of 720: under 1 %, noise
        self.assertIsNone(rect)
        self.assertIn("within 1%", why)

    def test_no_reading_or_too_few_windows_is_not_proposed(self):
        self.assertIn("no rectangle", stable_rectangle(crop_facts(measured(crop=None)))[1])
        rect, why = stable_rectangle(crop_facts(measured(windows=5, windows_failed=3)))
        self.assertIsNone(rect)
        self.assertIn("only 2 of 5", why)
        self.assertIsNone(stable_rectangle(crop_facts(measured(sw=None)))[0], "without the frame size nothing is proposed")

    def test_rectangle_outside_the_frame_or_too_small_is_not_proposed(self):
        self.assertIn("inside", stable_rectangle(crop_facts(measured(crop={"width": 1280, "height": 700, "x": 0, "y": 92})))[1])
        self.assertIn("smaller than 16", stable_rectangle(crop_facts(measured(crop={"width": 1280, "height": 8, "x": 0, "y": 300})))[1])

    def test_even_rounding_goes_inward(self):
        """Odd offsets / sizes are rounded so the rectangle never grows into a bar: left / top edges move in, right / bottom edges move in."""
        self.assertEqual(even_inward({"x": 1, "y": 137, "width": 1918, "height": 806}), {"x": 2, "y": 138, "width": 1916, "height": 804})
        self.assertEqual(even_inward({"x": 0, "y": 140, "width": 1920, "height": 800}), {"x": 0, "y": 140, "width": 1920, "height": 800})
        rect, _ = stable_rectangle(crop_facts(measured(crop={"width": 1279, "height": 537, "x": 1, "y": 91}, sw=1282)))
        self.assertEqual(rect, {"x": 2, "y": 92, "width": 1278, "height": 536})
        self.assertTrue(all(v % 2 == 0 for v in rect.values()))

    def test_approval_floor_is_confirm_even_when_policy_says_auto(self):
        """Never auto-applied: a policy / profile asking for AUTO on video.crop is raised back to CONFIRM for a measured proposal."""
        rules = resolve_rules([Rule("profile.crop", "PREFERENCE", "PROFILE", "video.crop.approval", "AUTO", "test")])
        asset = Asset(path="/x/a.mp4", provenance="USER", technical={"video": {"width": 1280, "height": 720}})
        obs = Observation(kind=CROP_KIND, asset_id=asset.id, source="ffmpeg-skill/cropdetect@2.22.1", data=measured())
        eng = DecisionEngine(rules, Intent(primary="clean_and_deliver"), {obs.id: "observation"})

        class _An:
            observations, assets = [obs], [asset]
        decide_crop(eng, _An, [(asset.id, [asset.id])], {}, None,
                    lambda subj, explicit=None, floor=None: resolve_approval(rules, "video.crop.approval", "CONFIRM", floor=floor, explicit=explicit),
                    lambda ids: [], lambda *a: None)
        d = eng.decisions[0]
        self.assertEqual((d.subject, d.type, d.approval), ("video.crop", "TRANSFORM", "CONFIRM"))
        self.assertTrue(any("floor" in n for n in d.basis["approval"]["notes"]), d.basis["approval"])


class CropPlanTests(unittest.TestCase):
    """Through the Service on the fake engine: measurement → observation → decision → plan step, approval, rejection."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def media(self, name="film.mp4", **facts):
        p = str(Path(self.tmp) / "src" / name)
        _write_fake(p, {"duration": 16.0, **facts})
        return p

    def plan(self, inputs, requirements=None, kinds=("crop_rectangle",), engine=None):
        self.engine = engine or FakeAdapter()
        self.svc = make_service(self.tmp, adapter=self.engine)
        return self.svc.plan(list(inputs), "generic", user_requirements=requirements, kinds=list(kinds))

    @staticmethod
    def crop_decisions(doc):
        return [d for d in doc["decisions"] if d["subject"] == "video.crop"]

    def test_letterboxed_input_proposes_the_measured_rectangle(self):
        ir = self.plan([self.media(crop=LETTERBOX)])
        d = ir.doc
        aid = next(iter(d["assets"]))
        obs = next(o for o in d["analysis"]["observations"] if o["kind"] == CROP_KIND)
        self.assertEqual((obs["provenance"], obs["tool"], obs["parameters"]), ("OBSERVED", "ffmpeg-skill/cropdetect", CROP_TOOL_ARGS))
        self.assertEqual(obs["data"]["crop"], LETTERBOX)
        self.assertNotIn("file", obs["data"], "no path in the measured facts")
        [dec] = self.crop_decisions(d)
        self.assertEqual((dec["type"], dec["approval"], dec["status"], dec["provenance"], dec["risk"]), ("TRANSFORM", "CONFIRM", "PROPOSED", "OBSERVED", "MEDIUM"))
        self.assertEqual(dec["params"], {"asset_id": aid, **LETTERBOX})
        self.assertIn(obs["id"], dec["evidence"])
        self.assertEqual(dec["basis"]["approval"]["key"], "video.crop.approval")
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "video_crop")
        self.assertEqual((step["tool"], step["decision_id"]), ("ffmpeg-skill/crop", dec["id"]))
        self.assertEqual({k: step["params"][k] for k in ("x", "y", "width", "height")}, LETTERBOX)
        op = next(o for o in d["video"]["operations"] if o["type"] == "video.crop")
        self.assertEqual({k: op[k] for k in ("x", "y", "width", "height")}, LETTERBOX)
        self.assertNotEqual(d["plan"]["status"], "APPROVED", "a measured crop waits for the user's approval")
        self.assertEqual(self.svc.validate(ir).errors, [])
        call = next(c for c in self.engine.calls if c.tool == "ffmpeg-skill/cropdetect")
        self.assertEqual({k: call.args[k] for k in CROP_TOOL_ARGS}, CROP_TOOL_ARGS, "the agent's sampling: 5 windows, even rounding")

    def test_never_applied_without_approval(self):
        ir = self.plan([self.media(crop=LETTERBOX)])
        p = str(Path(self.tmp) / "p.json"); save_ir(ir, p)
        out = self.svc.render(load_ir(p), p)
        self.assertNotEqual(out["status"], "COMPLETED", out)
        self.assertFalse(any(c.tool == "ffmpeg-skill/crop" for c in self.engine.calls), "nothing is cropped before the approval")

    def test_approved_proposal_runs_crop_with_the_rectangle(self):
        ir = self.plan([self.media(crop=LETTERBOX)])
        dec = self.crop_decisions(ir.doc)[0]
        p = str(Path(self.tmp) / "p.json"); save_ir(ir, p)
        self.svc.approve(load_ir(p), p, [dec["id"]])
        out = self.svc.render(load_ir(p), p)
        self.assertEqual(out["status"], "COMPLETED", out)
        call = next(c for c in self.engine.calls if c.tool == "ffmpeg-skill/crop")
        self.assertEqual({k: int(call.args[k]) for k in ("x", "y", "width", "height")}, LETTERBOX)
        self.assertIn(dec["id"], out["artifacts"][0]["decision_ids"])

    def test_rejected_proposal_is_dropped_on_revise(self):
        ir = self.plan([self.media(crop=LETTERBOX)])
        dec = self.crop_decisions(ir.doc)[0]
        p = str(Path(self.tmp) / "p.json"); save_ir(ir, p)
        self.svc.reject(load_ir(p), p, [dec["id"]], reason="the letterbox is the intended presentation")
        self.svc.revise(load_ir(p), p)
        v2 = load_ir(p).doc
        self.assertFalse([d for d in self.crop_decisions(v2) if d["status"] != "REJECTED"])
        self.assertNotIn("video_crop", [s["skill"] for s in v2["plan"]["steps"]])

    def test_unstable_measurement_proposes_nothing(self):
        ir = self.plan([self.media(crop=LETTERBOX, crop_confidence=0.5)])
        [dec] = self.crop_decisions(ir.doc)
        self.assertEqual((dec["type"], dec["approval"]), ("KEEP", "AUTO"))
        self.assertIn("not stable", dec["reason"])
        self.assertNotIn("video_crop", [s["skill"] for s in ir.doc["plan"]["steps"]])
        self.assertEqual(ir.doc["plan"]["status"], "APPROVED")

    def test_no_bars_proposes_nothing(self):
        for facts in ({}, {"crop": None}):
            ir = self.plan([self.media(**facts)])
            [dec] = self.crop_decisions(ir.doc)
            self.assertEqual(dec["type"], "KEEP", facts)
            self.assertNotIn("video_crop", [s["skill"] for s in ir.doc["plan"]["steps"]])
            self.assertEqual(self.svc.validate(ir).errors, [])

    def test_not_measured_unless_asked(self):
        ir = self.plan([self.media(crop=LETTERBOX)], kinds=())
        self.assertFalse(self.crop_decisions(ir.doc))
        self.assertFalse(any(c.tool == "ffmpeg-skill/cropdetect" for c in self.engine.calls))

    def test_explicit_edit_crop_wins(self):
        rect = {"x": 0, "y": 0, "width": 640, "height": 360}
        ir = self.plan([self.media(crop=LETTERBOX)], requirements={"edit.crop": rect})
        [dec] = self.crop_decisions(ir.doc)
        self.assertEqual((dec["provenance"], dec["approval"]), ("USER", "AUTO"))
        self.assertEqual({k: dec["params"][k] for k in rect}, {k: float(v) if k in ("x", "y") else v for k, v in rect.items()})

    def test_programme_of_several_inputs_proposes_nothing(self):
        ir = self.plan([self.media("a.mp4", crop=LETTERBOX), self.media("b.mp4", crop=LETTERBOX)], requirements={"edit.concat": True})
        self.assertFalse(self.crop_decisions(ir.doc))
        self.assertEqual(len([o for o in ir.doc["analysis"]["observations"] if o["kind"] == CROP_KIND]), 2, "the measurements stay on record")

    def test_each_input_is_proposed_on_its_own(self):
        ir = self.plan([self.media("a.mp4", crop=LETTERBOX), self.media("b.mp4")])
        decs = self.crop_decisions(ir.doc)
        self.assertEqual(sorted(d["type"] for d in decs), ["KEEP", "TRANSFORM"])
        self.assertEqual(len([s for s in ir.doc["plan"]["steps"] if s["skill"] == "video_crop"]), 1)
        self.assertEqual(self.svc.validate(ir).errors, [])

    def test_input_without_video_is_skipped(self):
        ir = self.plan([self.media("a.wav", video=False)])
        row = next(r for r in ir.doc["analysis"]["analyses"][0]["rows"] if r["kind"] == CROP_KIND)
        self.assertEqual((row["status"], row["reason"]), ("SKIPPED", "no video stream"))
        self.assertFalse(self.crop_decisions(ir.doc))


class CroppedPictureSizeTests(unittest.TestCase):
    """Found during PR #80: `picture_size` ignored `video.crop`, so a step sized from the picture after a crop (the thumbnail canvas,
    the QC gate's expected width) got the pre-crop size. The size now follows the IR's operation order: crop → rotate → resize → fit / fill."""

    def setUp(self):
        clear_modes()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        clear_modes()

    def test_picture_size_follows_the_operation_order(self):
        tech = {"video": {"width": 1280, "height": 720}}
        crop = {"type": "video.crop", "asset": "a", **LETTERBOX}
        self.assertEqual(picture_size(tech, [crop], "a"), (1280, 536))
        self.assertEqual(picture_size(tech, [crop, {"type": "video.rotate", "asset": "a", "degrees": 90}], "a"), (536, 1280), "the cropped picture is turned")
        self.assertEqual(picture_size(tech, [crop, {"type": "video.resize", "asset": "a", "width": 640}], "a"), (640, 268), "resize keeps the cropped aspect")
        self.assertEqual(picture_size(tech, [crop, {"type": "video.fit", "asset": "a", "aspect": "16:9"}], "a"), (1280, 720), "fit letterboxes the cropped picture back")
        self.assertEqual(picture_size(tech, [dict(crop, asset="b")], "a"), (1280, 720), "another subject's crop does not count")

    def _plan(self, reqs, kinds=("crop_rectangle",)):
        self.svc = pipeline_service(self.tmp)
        return self.svc.plan([pipeline_media(self.tmp, "film.mp4", crop=LETTERBOX)], "generic", user_requirements=reqs, kinds=list(kinds))

    def test_thumbnail_canvas_after_an_approved_crop_proposal(self):
        """ADR-060's proposal, approved: the thumbnail is taken from the cropped picture, so its canvas is the rectangle's size."""
        ir = self._plan({"thumbnail": True, "thumbnail.text": "Film"})
        dec = next(d for d in ir.doc["decisions"] if d["subject"] == "video.crop")
        p = str(Path(self.tmp) / "p.json"); save_ir(ir, p)
        self.svc.approve(load_ir(p), p, [dec["id"]])
        d = load_ir(p).doc
        th = next(op for op in d["graphics"]["operations"] if op["type"] == "graphics.thumbnail")
        self.assertEqual(th["input"], next(op["output"] for op in d["video"]["operations"] if op["type"] == "video.crop"), "taken from the cropped picture")
        self.assertEqual((th["width"], th["height"]), (1280, 536))
        self.assertEqual(self.svc.validate(load_ir(p)).errors, [])

    def test_thumbnail_canvas_after_an_explicit_crop_and_rotate(self):
        rect = {"x": 0, "y": 0, "width": 640, "height": 360}
        ir = self._plan({"edit.crop": rect, "edit.rotate": 90, "thumbnail": True, "thumbnail.text": "Film"}, kinds=())
        th = next(op for op in ir.doc["graphics"]["operations"] if op["type"] == "graphics.thumbnail")
        self.assertEqual((th["width"], th["height"]), (360, 640), "crop first (source pixels), then the quarter turn")

    def test_qc_expected_width_after_crop_and_rotate(self):
        """The other consumer of the planned size: the QC gate's expected width of a deliverable without a platform preset."""
        ir = self._plan({"edit.crop": {"x": 0, "y": 0, "width": 640, "height": 360}, "edit.rotate": 90, "qc": True}, kinds=())
        d = ir.doc
        row = next(r for r in delivery_subjects(d) if not r["audio_only"])
        spec = rules_for_subject(row, d["delivery"]["targets"][0], d, 2.0)
        self.assertEqual(spec["rules"]["delivery"]["video"]["expected_width"], 360, "the turned crop is 360 wide, not the crop's 640")
        ir = self._plan({"edit.rotate": 90, "qc": True}, kinds=())
        row = next(r for r in delivery_subjects(ir.doc) if not r["audio_only"])
        self.assertEqual(rules_for_subject(row, ir.doc["delivery"]["targets"][0], ir.doc, 2.0)["rules"]["delivery"]["video"]["expected_width"], 720,
                         "a quarter turn alone changes the delivered width too")


if __name__ == "__main__":
    unittest.main()
