"""ADR-108: a sibling Skill's live contract against its pin, release version by range and everything else by equality.

The fixtures are the real pins with a release made by hand the way the 2026-11-11 releases were made (every field that repeats the
release version changed, nothing else), so the expected outcome does not depend on the code under test to build its input.
"""
from __future__ import annotations

import copy
import importlib
import io
import os
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pin_check  # noqa: E402
from pin_check import assert_live_matches_pin  # noqa: E402
from video_agent.tools.release_version import RELEASE_VERSION_FIELDS, VersionOnlyDrift, compare_to_pin, restate_release_version  # noqa: E402


# (skill, adapter module, drift function name): every adapter whose integration test compares the live contract with the pin
ADAPTERS = (("artifacts-skill", "artifacts", "drift"), ("audio-production-skill", "audio_production", "contract_drift"),
            ("color-grading-skill", "color_grading", "contract_drift"), ("detection-skill", "detection", "drift"),
            ("diarization-skill", "diarization", "drift"), ("dubbing-skill", "dubbing", "drift"), ("imagemagick-skill", "imagemagick", "drift"),
            ("ingest-skill", "ingest", "drift"), ("motion-graphics-skill", "motion_graphics", "contract_drift"),
            ("nle-bridge-skill", "nle_bridge", "drift"), ("publish-skill", "publish", "drift"), ("qc-skill", "qc", "contract_drift"),
            ("remotion-ffmpeg-skill", "remotion", "drift"), ("slide-sync-skill", "slide_sync", "drift"), ("sound-design-skill", "sound_design", "drift"),
            ("subtitle-skill", "subtitle", "contract_drift"), ("thumbnail-skill", "thumbnail", "contract_drift"),
            ("video-editing-skill", "video_editing", "contract_drift"), ("voice-enhance-skill", "voice_enhance", "drift"))


def adapters():
    """(skill, pinned contract, SUPPORTED_SKILL_VERSIONS, drift function) per ADAPTERS row."""
    out = []
    for skill, mod, fn in ADAPTERS:
        m = importlib.import_module(f"video_agent.tools.{mod}.adapter")
        assert (getattr(m, "contract_drift", None) or m.drift) is getattr(m, fn), f"pin_check would not use {mod}.{fn}"
        out.append((skill, m.pinned_contract(), m.SUPPORTED_SKILL_VERSIONS, getattr(m, fn)))
    return out


def adapter(skill):
    return next((p, sup, d) for s, p, sup, d in adapters() if s == skill)


def by_hand(contract, new):
    """A release of `contract` that changes only its version number to `new`: every field listed in RELEASE_VERSION_FIELDS that holds the
    release version, set field by field here (the code under test is not used)."""
    c, old = copy.deepcopy(contract), contract["version"]
    c["version"] = new
    for t in c.get("tools") or []:
        if isinstance(t, dict) and t.get("version") == old:
            t["version"] = new
    if isinstance(c.get("skill"), dict) and c["skill"].get("version") == old:
        c["skill"]["version"] = new
    if c.get("tool_version") == old:
        c["tool_version"] = new
    for shape in ("response_shape", "doctor_shape"):
        if isinstance(c.get(shape), dict) and isinstance(c[shape].get("skill"), dict) and c[shape]["skill"].get("version") == old:
            c[shape]["skill"]["version"] = new
    if isinstance(c.get("versioning"), dict) and c["versioning"].get("version") == old:
        c["versioning"]["version"] = new
    return c


def next_patch(v):
    a, b, c = v.split(".")[:3]
    return f"{a}.{b}.{int(c) + 1}"


PROBE = unittest.TestCase()   # a TestCase to hand to assert_live_matches_pin, so a failed check is an AssertionError this test can observe


def check(skill, live):
    """The integration suite's check of `live` against the pin of `skill`'s adapter."""
    mod = next(m for s, m, _ in ADAPTERS if s == skill)
    return assert_live_matches_pin(PROBE, skill, live, importlib.import_module(f"video_agent.tools.{mod}.adapter"))


class VersionOnlyDriftTests(unittest.TestCase):
    def setUp(self):
        pin_check.REPORTED.clear()

    def tearDown(self):
        pin_check.REPORTED.clear()

    def test_version_only_release_passes_with_a_warning_for_every_adapter(self):
        for skill, pinned, supported, drift in adapters():
            with self.subTest(skill=skill):
                live = by_hand(pinned, next_patch(pinned["version"]))
                self.assertTrue(drift(live, pinned), "the adapter's own drift report still names the version difference")
                with self.assertWarns(VersionOnlyDrift) as w:
                    c = check(skill, live)
                self.assertTrue(c.version_only)
                self.assertEqual((c.out_of_range, c.content_drift), ([], []))
                self.assertIn(next_patch(pinned["version"]), str(w.warning)); self.assertIn("only in its version", str(w.warning))
                self.assertIn((skill, live["version"], pinned["version"]), pin_check.REPORTED)

    def test_the_fields_restated_are_the_ones_a_real_release_moved(self):
        """video-editing 0.6.0 -> 0.6.2 (2026-11-11) moved version, every tools[].version, response_shape.skill.version,
        doctor_shape.skill.version and versioning.version; color-grading 1.0.0 -> 1.0.1 moved version and tools[0].version."""
        pinned = adapter("video-editing-skill")[0]
        c = compare_to_pin("video-editing-skill", by_hand(pinned, "0.7.3"), pinned, *adapter("video-editing-skill")[1:])
        self.assertEqual(sorted(c.restated), sorted(["version", "response_shape.skill.version", "doctor_shape.skill.version", "versioning.version"]
                                                     + [f"tools[{i}].version" for i in range(len(pinned["tools"]))]))
        pinned = adapter("color-grading-skill")[0]
        c = compare_to_pin("color-grading-skill", by_hand(pinned, "1.1.2"), pinned, *adapter("color-grading-skill")[1:])
        self.assertEqual(c.restated, ["version", "tools[0].version"])
        for skill, field in (("imagemagick-skill", "skill.version"), ("artifacts-skill", "tool_version")):
            pinned = adapter(skill)[0]
            c = compare_to_pin(skill, by_hand(pinned, next_patch(pinned["version"])), pinned, *adapter(skill)[1:])
            self.assertIn(field, c.restated); self.assertTrue(c.version_only)
        self.assertEqual(len(RELEASE_VERSION_FIELDS), 7)

    def test_equal_contract_passes_without_a_warning(self):
        for skill, pinned, supported, drift in adapters():
            with self.subTest(skill=skill), warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                c = check(skill, copy.deepcopy(pinned))
                self.assertFalse(c.version_only); self.assertEqual(c.restated, [])
                self.assertEqual([x for x in seen if issubclass(x.category, VersionOnlyDrift)], [])
        self.assertEqual(pin_check.REPORTED, {})

    def test_content_change_fails_with_or_without_a_version_change(self):
        pinned, supported, drift = adapter("video-editing-skill")
        changed = by_hand(pinned, "0.6.3")
        changed["tools"][0]["parameters"] = {**changed["tools"][0].get("parameters", {}), "extra": {"type": "string"}}
        c = compare_to_pin("video-editing-skill", changed, pinned, supported, drift)
        self.assertFalse(c.version_only); self.assertTrue(any("parameters" in d for d in c.content_drift), c.content_drift)
        with self.assertRaises(AssertionError):
            check("video-editing-skill", changed)
        same_version = copy.deepcopy(pinned); same_version["errors"] = {**same_version["errors"], "codes": list(same_version["errors"]["codes"]) + ["NEW_CODE"]}
        with self.assertRaises(AssertionError):
            check("video-editing-skill", same_version)
        for skill, pinned, supported, drift in adapters():   # every adapter: a top-level drift key changed alongside the version
            with self.subTest(skill=skill):
                live = by_hand(pinned, next_patch(pinned["version"]))
                live["skill_id" if "skill_id" in pinned else "schema" if "schema" in pinned else "name"] = "something-else"
                self.assertTrue(compare_to_pin(skill, live, pinned, supported, drift).content_drift)
        self.assertEqual(pin_check.REPORTED, {})

    def test_a_version_field_that_does_not_repeat_the_release_is_content(self):
        """Only a field holding the release version is restated: a tool on a version of its own, or a response_shape naming another
        version, is compared as it is and fails."""
        pinned, supported, drift = adapter("video-editing-skill")
        live = by_hand(pinned, "0.6.3"); live["tools"][2]["version"] = "0.5.0"
        c = compare_to_pin("video-editing-skill", live, pinned, supported, drift)
        self.assertNotIn("tools[2].version", c.restated); self.assertTrue(any(".version" in d for d in c.content_drift), c.content_drift)
        live = copy.deepcopy(pinned); live["tools"][1]["version"] = "0.6.3"   # the release did not move, one tool's label did
        self.assertTrue(compare_to_pin("video-editing-skill", live, pinned, supported, drift).content_drift)

    def test_version_outside_the_gate_fails(self):
        for skill, new in (("video-editing-skill", "0.8.0"), ("video-editing-skill", "1.0.0"), ("color-grading-skill", "1.2.0"),
                           ("color-grading-skill", "2.0.0"), ("subtitle-skill", "0.11.0"), ("qc-skill", "1.5.0"), ("thumbnail-skill", "0.2.0")):
            pinned, supported, drift = adapter(skill)
            with self.subTest(skill=skill, version=new):
                c = compare_to_pin(skill, by_hand(pinned, new), pinned, supported, drift)
                self.assertEqual(c.content_drift, [], "the content is the pin's: only the range refuses it")
                self.assertTrue(c.out_of_range); self.assertFalse(c.version_only)
                with self.assertRaises(AssertionError):
                    check(skill, by_hand(pinned, new))
        pinned, supported, drift = adapter("qc-skill")
        self.assertTrue(compare_to_pin("qc-skill", pinned, pinned, ("9.",), drift).out_of_range, "a pin outside the gate is refused too")
        self.assertTrue(compare_to_pin("qc-skill", pinned, pinned, (), drift).out_of_range, "an empty gate accepts nothing")
        self.assertEqual(pin_check.REPORTED, {})

    def test_restate_leaves_the_input_alone(self):
        pinned = adapter("video-editing-skill")[0]
        live = by_hand(pinned, "0.7.3"); before = copy.deepcopy(live)
        out, changed = restate_release_version(live, "0.7.0")
        self.assertEqual(live, before); self.assertEqual(out, pinned); self.assertTrue(changed)
        self.assertEqual(restate_release_version({"name": "x"}, "1.0.0"), ({"name": "x"}, []))

    def test_report_once_per_release_with_an_annotation_and_a_job_summary(self):
        pinned, supported, drift = adapter("thumbnail-skill")
        live = by_hand(pinned, next_patch(pinned["version"]))
        with tempfile.TemporaryDirectory() as tmp:
            summary = os.path.join(tmp, "summary.md")
            out = io.StringIO()
            with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": summary}), redirect_stdout(out), \
                    warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                for _ in range(2):
                    check("thumbnail-skill", live)
                pin_check._summary()
            self.assertEqual(len([x for x in seen if issubclass(x.category, VersionOnlyDrift)]), 2, "a warning at every check")
            self.assertEqual(out.getvalue().count("::warning title=Version-only contract drift (thumbnail-skill)::"), 1, "one annotation per release")
            text = Path(summary).read_text(encoding="utf-8")
            self.assertIn("version-only contract drift: 1 Skill(s)", text); self.assertIn(live["version"], text)


if __name__ == "__main__":
    unittest.main()
