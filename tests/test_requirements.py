"""agent/requirements.py: Request -> Requirement extraction. Covers the boolean KEYWORDS pass
(existing behavior, kept as a regression guard) and the numeric NUMERIC_KEYWORDS pass (this
session's fix: an explicit target named in free text, e.g. "-16 LUFS", must actually become the
USER requirement's value -- not just flip a boolean intent flag while silently keeping whatever
default target the profile/rules would otherwise supply)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from video_agent.agent.requirements import extract_requirements, requirement_map  # noqa: E402
from video_agent.models import Request  # noqa: E402
from video_agent.policy.rules import Rule, resolve_rules  # noqa: E402
from video_agent.profiles import load_profile  # noqa: E402


def make_request(raw: str, requirements: dict | None = None) -> Request:
    return Request(raw=raw, args={"requirements": requirements or {}})


class NumericRequirementExtractionTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile("generic")
        self.rules = resolve_rules(self.profile.rules)

    def test_lufs_target_named_in_text_becomes_the_user_requirement_value(self):
        reqs = requirement_map(extract_requirements(make_request("normalize loudness to -16 LUFS"), self.profile, self.rules))
        self.assertEqual((reqs["audio.normalize"].value, reqs["audio.normalize"].provenance), (True, "USER"))
        self.assertEqual((reqs["audio.loudness.target_lufs"].value, reqs["audio.loudness.target_lufs"].provenance), (-16.0, "USER"))

    def test_true_peak_target_named_in_text_becomes_the_user_requirement_value(self):
        reqs = requirement_map(extract_requirements(make_request("normalize to -18 LUFS, -1.5 dBTP"), self.profile, self.rules))
        self.assertEqual(reqs["audio.loudness.target_lufs"].value, -18.0)
        self.assertEqual((reqs["audio.loudness.true_peak"].value, reqs["audio.loudness.true_peak"].provenance), (-1.5, "USER"))

    def test_normalize_keyword_alone_falls_back_to_profile_default_unchanged(self):
        # Regression guard: no explicit number in the text -> still resolves to whatever the
        # profile/rules already provide (nothing, for the "generic" profile), exactly as before
        # this session's fix -- never a USER-provenance value invented from thin air.
        reqs = requirement_map(extract_requirements(make_request("please normalize the audio"), self.profile, self.rules))
        self.assertEqual(reqs["audio.normalize"].value, True)
        self.assertNotIn("audio.loudness.target_lufs", reqs)

    def test_explicit_cli_set_takes_priority_over_a_number_also_named_in_the_text(self):
        # If both --set and a number-in-text disagree, the explicit structured CLI value wins
        # (it is added to `seen` before the text passes run) -- never silently overridden.
        reqs = requirement_map(extract_requirements(make_request("normalize to -16 LUFS", requirements={"audio.loudness.target_lufs": -20}), self.profile, self.rules))
        self.assertEqual((reqs["audio.loudness.target_lufs"].value, reqs["audio.loudness.target_lufs"].source), (-20, "cli"))

    def test_no_number_and_no_normalize_keyword_leaves_target_lufs_non_user(self):
        reqs = requirement_map(extract_requirements(make_request("trim the leading silence"), self.profile, self.rules))
        self.assertEqual(reqs["edit.trim_leading_silence"].value, True)
        self.assertNotIn("audio.loudness.target_lufs", {k for k, r in reqs.items() if r.provenance == "USER"})

    def test_a_bare_number_with_no_unit_is_never_treated_as_a_loudness_target(self):
        # "-16" alone (e.g. part of an unrelated phrase) must not be misread as a LUFS target --
        # only an unambiguous "<number> LUFS" phrase should ever produce this USER requirement.
        reqs = requirement_map(extract_requirements(make_request("trim -16 frames from the start"), self.profile, self.rules))
        self.assertNotEqual(reqs.get("audio.loudness.target_lufs", type("_", (), {"provenance": None})).provenance, "USER")

    def test_empty_request_text_produces_no_user_requirements_from_either_pass(self):
        reqs = extract_requirements(make_request(""), self.profile, self.rules)
        self.assertFalse(any(r.provenance == "USER" and r.source == "request-text" for r in reqs))


if __name__ == "__main__":
    unittest.main()
