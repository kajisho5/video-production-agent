"""ADR-103: media-analysis-skill 0.1.13's editing-style profile as the analysis kind `edit_style` (tool media-analysis/style), on the fake
media-analysis process: requested with `--kind edit_style`, lifted as an OBSERVED Observation with the Skill's provenance, its parameters the
Skill's own. tests/test_integration.py measures it with the real Skill."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter  # noqa: E402
from test_unit import FakeCaps, fake_media, make_service  # noqa: E402
from video_agent.media import ANALYSIS_KINDS, AnalysisRequest  # noqa: E402
from video_agent.tools.media_analysis.adapter import pinned_contract  # noqa: E402

FAKE_MA = str(Path(__file__).resolve().parent / "fake_media_analysis.py")


class EditStyleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp, "ref.mp4")

    def test_kind_parameters_are_the_contracts(self):
        tool = next(t for t in pinned_contract()["tools"] if t["tool_id"] == "media-analysis/style")
        self.assertEqual(tool["kinds"], ["edit_style"])
        self.assertEqual(set(ANALYSIS_KINDS["edit_style"]["params"]), set(tool["parameters"]["edit_style"]))
        self.assertEqual(AnalysisRequest(inputs=[self.src], kinds=["edit_style"]).kinds, ["media_probe", "edit_style"])

    def test_measured_through_media_analysis(self):
        from video_agent.tools import ToolRouter
        from video_agent.tools.media_analysis import MediaAnalysisAdapter, MediaAnalysisSkill
        ad = MediaAnalysisAdapter(MediaAnalysisSkill([sys.executable, FAKE_MA], None, {}), workspace=self.tmp, allowed_inputs=[str(Path(self.src).parent)])
        svc = make_service(self.tmp, caps=FakeCaps(extra=["media-analysis", "filter:scdet", "filter:edgedetect"]), adapter=ToolRouter([FakeAdapter(), ad]))
        self.assertEqual(svc.tools_for().get("edit_style_analysis"), "media-analysis/style")
        _, _, an = svc.analyze([self.src], "generic", kinds=["edit_style"])
        o = next(x for x in an.observations if x.kind == "edit_style")
        self.assertEqual((o.provenance, o.skill, o.tool), ("OBSERVED", "media-analysis", "media-analysis/style"))
        self.assertEqual(o.data["video"]["cut_count"], 1)
        # without the Skill's filters the kind is not selectable (never measured another way)
        svc2 = make_service(self.tmp, caps=FakeCaps(extra=["media-analysis"], missing=["filter:edgedetect"]), adapter=ToolRouter([FakeAdapter(), ad]))
        self.assertIsNone(svc2.tools_for().get("edit_style_analysis"))


if __name__ == "__main__":
    unittest.main()
