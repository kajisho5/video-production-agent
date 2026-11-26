"""ADR-130: the reference pacing route (agent/pacing.py) against a fake engine: preview by default, apply only with the
preview's confirm_token, the same protected ranges at every stage, publication only after the independent gate."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from video_agent.agent.pacing import route

PLAN = {"contract": "pacing-cut-plan@1", "cut_count": 1, "keep": [{"start_ms": 0, "end_ms": 900}]}


class FakeEngine:
    def __init__(self, confidence="measured", bgm=False, verified=True, plans=None):
        self.calls, self.meta, self.verified = [], dict(confidence=confidence, bgm_suspected=bgm, warnings=[]), verified
        self.plans = list(plans or [])

    def check_output(self, target, inputs):
        self.calls.append(("check_output", target))

    def extract(self, refs, output):
        self.calls.append(("extract", tuple(refs)))
        return {"contract": "pacing-profile@1", "probe_meta": self.meta}

    def _plan(self):
        return self.plans.pop(0) if self.plans else PLAN

    def edit(self, source, profile, protect, output, preview=False):
        self.calls.append(("preview" if preview else "edit", profile, protect))
        if not preview:
            Path(output).write_bytes(b"media")
        return {"cut_plan": self._plan(), "verified": not preview}

    def verify(self, source, profile, protect, output):
        self.calls.append(("verify", profile, protect))
        return {"verified": self.verified, "cut_plan": PLAN, "checks": {"pauses": self.verified}}


class PacingRouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        for name in ("talk.mkv", "ref.mkv"):
            (self.ws/name).write_bytes(b"x")
        self.keep = self.ws/"keep.json"
        self.keep.write_text(json.dumps([{"start_ms": 100, "end_ms": 200, "reason": "word ending"}]))
        self.src, self.ref = str(self.ws/"talk.mkv"), str(self.ws/"ref.mkv")

    def run_route(self, engine, **kw):
        return route(str(self.ws), engine, self.src, "out.mp4", **kw)

    def stages(self, engine):
        return [c[0] for c in engine.calls if c[0] != "check_output"]

    def test_preview_is_the_default_and_renders_nothing(self):
        engine = FakeEngine()
        result = self.run_route(engine, references=[self.ref], protect=str(self.keep))
        self.assertEqual(result["status"], "PREVIEW")
        self.assertEqual(result["profile_used"], "measured")
        self.assertEqual(self.stages(engine), ["extract", "preview"])
        self.assertRegex(result["confirm_token"], "^[0-9a-f]{64}$")
        self.assertFalse((self.ws/"out.mp4").exists())

    def test_apply_needs_the_preview_token(self):
        engine = FakeEngine()
        with self.assertRaisesRegex(ValueError, "confirm_token"):
            self.run_route(engine, profile="youtube-talk", apply=True)
        self.assertEqual(engine.calls, [])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.run_route(engine, profile="youtube-talk", apply=True, confirm="0" * 64)
        self.assertNotIn("edit", self.stages(engine))
        self.assertFalse((self.ws/"out.mp4").exists())

    def test_changed_measurement_after_preview_is_refused(self):
        token = self.run_route(FakeEngine(), profile="youtube-talk")["confirm_token"]
        engine = FakeEngine(plans=[dict(PLAN, cut_count=2)])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.run_route(engine, profile="youtube-talk", apply=True, confirm=token)
        self.assertNotIn("edit", self.stages(engine))

    def test_apply_passes_the_same_ranges_and_publishes_after_the_gate(self):
        token = self.run_route(FakeEngine(), references=[self.ref], protect=str(self.keep))["confirm_token"]
        engine = FakeEngine()
        result = self.run_route(engine, references=[self.ref], protect=str(self.keep), apply=True, confirm=token)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(self.stages(engine), ["extract", "preview", "edit", "verify"])
        self.assertEqual({c[2] for c in engine.calls if c[0] in ("preview", "edit", "verify")}, {str(self.keep.resolve())})
        self.assertEqual((self.ws/"out.mp4").read_bytes(), b"media")

    def test_failed_gate_publishes_nothing(self):
        token = self.run_route(FakeEngine(), profile="youtube-talk")["confirm_token"]
        with self.assertRaisesRegex(ValueError, "independent"):
            self.run_route(FakeEngine(verified=False), profile="youtube-talk", apply=True, confirm=token)
        self.assertEqual(sorted(os.listdir(self.ws)), ["keep.json", "ref.mkv", "talk.mkv"])   # no output, no staged files

    def test_uncertain_reference_stops_unless_a_fallback_is_given(self):
        for engine in (FakeEngine(confidence="low"), FakeEngine(bgm=True)):
            result = self.run_route(engine, references=[self.ref], apply=True, confirm="t")
            self.assertEqual(result["status"], "NEEDS_REVIEW")
            self.assertEqual(self.stages(engine), ["extract"])
        result = self.run_route(FakeEngine(confidence="low"), references=[self.ref], profile="podcast-natural")
        self.assertEqual((result["status"], result["profile_used"]), ("PREVIEW", "podcast-natural"))


if __name__ == "__main__":
    unittest.main()
