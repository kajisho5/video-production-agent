"""Tests for `skills/diagnostics.py` (the provides/Capability consumption diagnostic,
`video-agent skills --check-provides`).

Two layers: synthetic unit tests exercise every status the algorithm can produce in
isolation (including the shared-tool-id annotation), independent of any real Skill.
The "RealData" tests below use real `provides[]` excerpts captured from each Skill's
actual, merged `main` (2026-09-06) as literal fixtures -- no network access, matching
this test suite's own existing convention (`tests/fake_qc.py` and friends) -- combined
with this Agent's *real*, currently-registered `PACKAGE`/`default_registry()` data, so
they double as regression tests for the two real tool-id drift cases (`qc-skill`,
`subtitle-skill`) and the shared-tool-id granularity limit (`color-grading-skill`)
found by `kajisho5/AI-video-production-OS`'s own exhaustive investigation
(`docs/ecosystem/WORK_QUEUE.md` item 1, 2026-09-06).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from video_agent.skills import SkillPackage, SkillSpec, ToolSpec  # noqa: E402
from video_agent.skills.diagnostics import check_all, check_provides, extract_provides  # noqa: E402


def pkg(skill_id: str, tool_ids) -> SkillPackage:
    return SkillPackage(skill_id=skill_id, name=skill_id, version="1.0", description="",
                        tools=[ToolSpec(tool_id=t, skill_id=skill_id) for t in tool_ids])


def spec(name: str, tools) -> SkillSpec:
    return SkillSpec(name, "1.0", "", {}, {}, [], "LOW", True, "AUTO", list(tools))


def provides(*entries) -> dict:
    """entries: (capability_id, tool_id[, lifecycle]) tuples -> a minimal Capability Contract dict."""
    return {"provides": [{"id": e[0], "tool_id": e[1], "lifecycle": e[2] if len(e) > 2 else "EXPERIMENTAL"} for e in entries]}


class ExtractProvidesTests(unittest.TestCase):
    def test_none_contract_is_empty(self):
        self.assertEqual(extract_provides(None), [])

    def test_missing_provides_key_is_empty(self):
        self.assertEqual(extract_provides({"id": "x"}), [])

    def test_non_list_provides_is_empty(self):
        self.assertEqual(extract_provides({"provides": "not-a-list"}), [])

    def test_real_shape_passes_through(self):
        doc = provides(("edit.trim", "video-editing/trim"))
        self.assertEqual(extract_provides(doc), doc["provides"])


class CheckProvidesStatusTests(unittest.TestCase):
    """One synthetic case per status, in isolation."""

    def test_unknown_when_contract_is_none(self):
        findings = check_provides(pkg("acme", ["acme/run"]), None, [])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].status, "UNKNOWN")
        self.assertEqual(findings[0].capability_id, "")

    def test_provides_valid_when_tool_id_matches_and_is_consumed(self):
        p = pkg("acme", ["acme/trim"])
        specs = [spec("acme_trim", ["acme/trim"])]
        findings = check_provides(p, provides(("acme.trim", "acme/trim")), specs)
        self.assertEqual(len(findings), 1)
        f = findings[0]
        self.assertEqual((f.status, f.capability_id, f.tool_id), ("PROVIDES_VALID", "acme.trim", "acme/trim"))
        self.assertNotIn("shared_tool_id_capabilities", f.evidence)

    def test_capability_unconsumed_when_no_skillspec_references_it(self):
        p = pkg("acme", ["acme/trim"])
        findings = check_provides(p, provides(("acme.trim", "acme/trim")), [])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].status, "CAPABILITY_UNCONSUMED")

    def test_provides_mismatch_when_agent_package_lacks_the_tool_id(self):
        """The real qc-skill / subtitle-skill shape, synthetically: the contract's tool_id is not
        one this Agent's registered package declares at all (a self-declared tool id elsewhere)."""
        p = pkg("acme", ["acme/check"])   # Agent invented "acme/check", not "acme/run"
        findings = check_provides(p, provides(("acme.measure", "acme/run")), [spec("acme_measure", ["acme/check"])])
        mismatch = [f for f in findings if f.status == "PROVIDES_MISMATCH"]
        self.assertEqual(len(mismatch), 1)
        self.assertEqual(mismatch[0].capability_id, "acme.measure")
        self.assertIn("acme/check", mismatch[0].evidence["agent_tool_ids"])

    def test_capability_missing_when_skillspec_references_a_tool_id_nobody_knows(self):
        p = pkg("acme", ["acme/trim"])
        findings = check_provides(p, provides(("acme.trim", "acme/trim")), [spec("acme_ghost", ["acme/ghost"])])
        missing = [f for f in findings if f.status == "CAPABILITY_MISSING"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].tool_id, "acme/ghost")
        self.assertEqual(missing[0].capability_id, "")

    def test_unknown_when_skillspec_tool_is_real_but_not_yet_published(self):
        """The Agent's package already knows about a real tool the contract just hasn't put under
        provides[] yet -- informational, not necessarily broken."""
        p = pkg("acme", ["acme/trim", "acme/fade"])
        findings = check_provides(p, provides(("acme.trim", "acme/trim")), [spec("acme_fade", ["acme/fade"])])
        unknown = [f for f in findings if f.status == "UNKNOWN" and f.tool_id == "acme/fade"]
        self.assertEqual(len(unknown), 1)

    def test_shared_tool_id_is_annotated_not_silently_confident(self):
        """Two Capability ids behind one generic tool id (qc-skill / color-grading-skill's real
        shape): both are still PROVIDES_VALID at the tool-id level, but each finding must name its
        siblings rather than implying the specific Capability id was individually confirmed."""
        p = pkg("acme", ["acme/run"])
        specs = [spec("acme_a", ["acme/run"])]
        findings = check_provides(p, provides(("acme.a", "acme/run"), ("acme.b", "acme/run")), specs)
        self.assertEqual({f.status for f in findings}, {"PROVIDES_VALID"})
        for f in findings:
            self.assertEqual(f.evidence["shared_tool_id_capabilities"], ["acme.a", "acme.b"])
            self.assertIn("shared", f.detail)

    def test_malformed_entries_are_skipped_not_crashed_on(self):
        doc = {"provides": ["not-a-dict", {"id": "acme.trim"}, {"tool_id": "acme/trim"}, None]}
        findings = check_provides(pkg("acme", ["acme/trim"]), doc, [])
        self.assertEqual(findings, [])   # every entry is missing id or tool_id or is not a dict


class CheckAllTests(unittest.TestCase):
    def test_dispatches_per_package_and_defaults_missing_contract_to_none(self):
        p1, p2 = pkg("acme", ["acme/trim"]), pkg("beta", ["beta/run"])
        contracts = {"acme": provides(("acme.trim", "acme/trim"))}   # "beta" has no entry at all
        findings = check_all([p1, p2], contracts, [spec("acme_trim", ["acme/trim"])])
        by_skill = {f.skill_id: f.status for f in findings}
        self.assertEqual(by_skill["acme"], "PROVIDES_VALID")
        self.assertEqual(by_skill["beta"], "UNKNOWN")


class RealDataRegressionTests(unittest.TestCase):
    """Real provides[] excerpts captured 2026-09-06 from each Skill's actual merged `main`,
    combined with this Agent's real, currently-registered PACKAGE / default_registry() data."""

    def test_real_qc_skill_all_ten_capabilities_are_self_declared_mismatch(self):
        from video_agent.skills import default_registry
        from video_agent.tools.qc import PACKAGE as QC_PACKAGE

        qc_provides = provides(
            ("measure.audio.channel_layout", "qc/run"), ("measure.audio.clipping_and_dynamics", "qc/run"),
            ("measure.audio.integrity", "qc/run"), ("measure.audio.loudness", "qc/run"),
            ("measure.audio.silence", "qc/run"), ("measure.delivery.integrity", "qc/run"),
            ("measure.subtitle.timing", "qc/run"), ("measure.video.black_frame", "qc/run"),
            ("measure.video.format", "qc/run"), ("measure.video.freeze", "qc/run"),
        )
        findings = check_provides(QC_PACKAGE, qc_provides, default_registry().all())
        mismatches = {f.capability_id for f in findings if f.status == "PROVIDES_MISMATCH"}
        self.assertEqual(len(mismatches), 10, findings)
        self.assertIn("measure.audio.loudness", mismatches)
        # confirms WORK_QUEUE item 1's finding: real tool_id is "qc/run" everywhere, never "qc/check"
        self.assertTrue(all(f.evidence.get("agent_tool_ids") == ["qc/check", "qc/inspect"] for f in findings if f.status == "PROVIDES_MISMATCH"))

    def test_real_subtitle_skill_both_capabilities_are_self_declared_mismatch(self):
        from video_agent.skills import default_registry
        from video_agent.tools.subtitle import PACKAGE as SUBTITLE_PACKAGE

        subtitle_provides = provides(("subtitle.generate", "subtitle-skill/generate"), ("subtitle.render", "subtitle-skill/render"))
        findings = check_provides(SUBTITLE_PACKAGE, subtitle_provides, default_registry().all())
        mismatches = {f.capability_id for f in findings if f.status == "PROVIDES_MISMATCH"}
        self.assertEqual(mismatches, {"subtitle.generate", "subtitle.render"})

    def test_real_color_grading_skill_shared_tool_id_all_valid_but_annotated(self):
        """color-grading-skill really does publish all 5 Capability ids through the one generic
        tool id "color-grading/run", and 4 of the Agent's 5 color_* SkillSpecs really do reference
        that same tool id -- so tool-id-level evidence alone reports all 5 as PROVIDES_VALID, but
        every one of them must carry the shared-tool-id annotation, since this diagnostic cannot
        confirm from tool_id alone that e.g. `color.primary_correction` specifically -- which has
        no color_primary_correction SkillSpec of its own in default_registry() -- is really
        requested, only that the shared tool id some sibling capability uses is."""
        from video_agent.skills import default_registry
        from video_agent.tools.color_grading import PACKAGE as COLOR_GRADING_PACKAGE

        color_provides = provides(
            ("color.hdr_to_sdr", "color-grading/run"), ("color.lut_apply", "color-grading/run"),
            ("color.primary_correction", "color-grading/run"), ("color.retag", "color-grading/run"),
            ("color.strip_dovi", "color-grading/run"),
        )
        findings = check_provides(COLOR_GRADING_PACKAGE, color_provides, default_registry().all())
        self.assertEqual({f.status for f in findings}, {"PROVIDES_VALID"})
        self.assertEqual(len(findings), 5)
        for f in findings:
            self.assertEqual(len(f.evidence["shared_tool_id_capabilities"]), 5)
        primary = next(f for f in findings if f.capability_id == "color.primary_correction")
        self.assertIn("not that this specific Capability id is separately requested", primary.detail)


if __name__ == "__main__":
    unittest.main()
