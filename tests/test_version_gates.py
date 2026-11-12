"""The satellites now report their release version (single-sourced from pyproject.toml) instead of a stale
constant: motion-graphics 0.2.x, transcription 0.7.x, subtitle 0.2.x / 0.3.x, qc 0.2.x (qc-skill #22). Their contracts are otherwise the pinned ones, so the
version gates accept those ranges and nothing else.

    python -m unittest tests/test_version_gates.py
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from video_agent.tools.audio_production import adapter as ap  # noqa: E402
from video_agent.tools.color_grading import adapter as cg  # noqa: E402
from video_agent.tools.diarization import adapter as dia  # noqa: E402
from video_agent.tools.motion_graphics import adapter as mg  # noqa: E402
from video_agent.tools.qc import adapter as qc  # noqa: E402
from video_agent.tools.subtitle import adapter as st  # noqa: E402
from video_agent.tools.transcription import adapter as tr  # noqa: E402
from video_agent.tools.video_editing import adapter as ve  # noqa: E402


def with_version(contract, version):
    c = dict(contract, version=version)
    if isinstance(c.get("tools"), list):
        c["tools"] = [dict(t, version=version) if isinstance(t, dict) and "version" in t else t for t in c["tools"]]
    return c


class VersionGateTests(unittest.TestCase):
    def test_motion_graphics_release_versions(self):
        self.assertEqual(mg.pinned_contract()["version"], "0.2.19")   # ADR-067: motion-graphics-skill #30 (colors.named, alpha: false); ADR-083: 0.2.8 widens its ffmpeg-skill range; ADR-097: 0.2.9 / 0.2.10 version-only; addendum 6: 0.2.16; ADR-106: 0.2.17 (layer_export); ADR-107: 0.2.18 / 0.2.19 version-only
        for v in ("0.1.0", "0.2.0", "0.2.1", "0.2.4", "0.2.9", "0.2.10", "0.2.16", "0.2.17", "0.2.18", "0.2.19"):
            live = with_version(mg.pinned_contract(), v)
            self.assertEqual(mg.check_contract(live), [], v)
            self.assertEqual(mg.contract_breaking_drift(live), [], v)
        self.assertTrue(mg.check_contract(with_version(mg.pinned_contract(), "0.3.0")))

    def test_transcription_release_versions(self):
        # 0.8.0 (ADR-078): optional engine whisperx appended, nothing the adapter sends changed; 0.9.0 (ADR-107 addendum 2): word index,
        # optional dictionary words (pos / timing) and the opt-in word_segmentation input, all additions
        for v in ("0.2.0", "0.7.3", "0.8.0", "0.8.2", "0.9.0", "0.9.3"):
            self.assertEqual(tr.check_contract(with_version(tr.pinned_contract(), v)), [], v)
        self.assertEqual(tr.pinned_contract()["version"], "0.9.0")
        for v in ("0.3.0", "0.10.0", "1.0.0"):
            self.assertTrue(tr.check_contract(with_version(tr.pinned_contract(), v)), v)

    def test_qc_release_versions(self):
        """qc-skill #22: the contract reports the pyproject version (0.2.1, 0.2.2) instead of a hard-coded 0.1.0; #23 (reference metrics) is 0.2.3."""
        self.assertEqual(qc.pinned_contract()["version"], "1.1.0")   # ADR-078: qc-skill #25 (per-operation tool_ids); ADR-081: 0.2.6 is version-only; ADR-097: 1.0.0 is version-only; ADR-107: 1.0.1 is version-only; ADR-107 addendum 2: 1.1.0 adds opt-in subtitle checks
        for v in ("0.1.0", "0.2.1", "0.2.2", "0.2.3", "0.2.6", "0.2.9", "1.0.0", "1.0.1", "1.0.3", "1.1.0", "1.1.1"):
            live = with_version(qc.pinned_contract(), v)
            self.assertEqual(qc.check_contract(live), [], v)
            self.assertEqual(qc.contract_breaking_drift(live), [], v)
        for v in ("0.3.0", "1.2.0", "2.0.0"):
            self.assertTrue(qc.check_contract(with_version(qc.pinned_contract(), v)), v)

    def test_subtitle_release_versions(self):
        # subtitle-skill 0.2.0 (#19, ja/zh/ko locale profiles) reports its release version; the contract is otherwise the 0.1.x one
        # ADR-097 addendum 6: 0.5.0 .. 0.8.0 add opt-in render / generate inputs and formats; the request the adapter builds is unchanged, so the gate takes 0.5.x .. 0.8.x
        # ADR-107: 0.9.0 adds render's optional fonts_dir (burn only), 0.9.1 is version-only; re-pinned from the v0.9.1 tag and the gate takes 0.9.x
        self.assertEqual(st.pinned_contract()["version"], "0.9.1")   # ADR-078: subtitle-skill 0.3.0 adds translate / mux, 0.4.0 two translate engines; ADR-083: 0.4.1 version-only; ADR-097: 0.4.2 / 0.4.3 version-only
        for v in ("0.1.5", "0.2.0", "0.2.1", "0.2.4", "0.3.0", "0.4.0", "0.4.3", "0.5.0", "0.6.0", "0.7.0", "0.8.0", "0.8.3", "0.9.0", "0.9.1", "0.9.2"):
            live = with_version(st.pinned_contract(), v)
            self.assertEqual(st.check_contract(live), [], v)
            self.assertEqual(st.contract_breaking_drift(live), [], v)
        for v in ("0.10.0", "1.0.0"):
            self.assertTrue(st.check_contract(with_version(st.pinned_contract(), v)), v)
        # a real 0.8.x contract has no fonts_dir: inside the version range, but its capabilities / operations / parameters differ from the pin (fatal drift)
        old = with_version(st.pinned_contract(), "0.8.0")
        old["capabilities"] = {k: v for k, v in old["capabilities"].items() if k != "fonts_dir"}
        old["parameters"] = dict(old["parameters"], render=[p for p in old["parameters"]["render"] if p != "fonts_dir"])
        old["operations"] = dict(old["operations"], render=dict(old["operations"]["render"], inputs=[i for i in old["operations"]["render"]["inputs"] if i != "fonts_dir?"],
                                                                outputs=[o for o in old["operations"]["render"]["outputs"] if o != "fonts_dir?"]))
        self.assertEqual(st.check_contract(old), [])
        self.assertEqual({d.split(":")[0] for d in st.contract_breaking_drift(old)}, {"capabilities", "operations", "parameters"})

    def test_diarization_versions(self):
        # ADR-097 addendum 5: diarization-skill main reports 0.3.0 (0.2.0 enroll / name, 0.3.0 --chunk-seconds; both opt-in, the diarize
        # wire the adapter sends is unchanged), so the gate takes 0.2.x and 0.3.x besides 0.1.x
        self.assertEqual(dia.pinned_contract()["version"], "0.3.0")
        for v in ("0.1.0", "0.2.0", "0.3.0", "0.3.4"):
            live = with_version(dia.pinned_contract(), v)
            self.assertEqual(dia.check_contract(live), [], v)
            self.assertEqual(dia.breaking_drift(live), [], v)
        for v in ("0.4.0", "1.0.0"):
            self.assertTrue(dia.check_contract(with_version(dia.pinned_contract(), v)), v)

    def test_video_editing_versions(self):
        # ADR-097 addendum 6: video-editing-skill 0.5.0 bumps contract_version to "5.0" for the opt-in operations / fields 0.4.8 .. 0.4.18 added;
        # the request the adapter builds is unchanged, so the gate takes 0.5.x besides 0.1.x .. 0.4.x
        # ADR-106: 0.6.0 bumps contract_version to "6.0" for TRIM / CUT precision smart (opt-in); the gate takes 0.6.x, 0.7.0 stays out
        # ADR-107: 0.6.1 / 0.6.2 are version-only (the version is also in response_shape / doctor_shape / versioning, so the pin is re-taken from the tag)
        # ADR-107 addendum 4: 0.7.0 (contract_version still 6.0) measures the stinger's cover at its cut point; the request is unchanged
        self.assertEqual((ve.pinned_contract()["version"], ve.pinned_contract()["contract_version"]), ("0.7.0", "6.0"))
        self.assertEqual({ve.pinned_contract()[k]["skill"]["version"] for k in ("response_shape", "doctor_shape")} | {ve.pinned_contract()["versioning"]["version"]}, {"0.7.0"})
        for v in ("0.4.0", "0.4.18", "0.5.0", "0.5.3", "0.6.0", "0.6.1", "0.6.2", "0.6.4", "0.7.0", "0.7.2"):
            live = with_version(ve.pinned_contract(), v)
            self.assertEqual(ve.check_contract(live), [], v)
            self.assertEqual(ve.contract_breaking_drift(live), [], v)
        for v in ("0.8.0", "1.0.0"):
            self.assertTrue(ve.check_contract(with_version(ve.pinned_contract(), v)), v)

    def test_video_editing_release_label_in_response_shape_is_not_fatal(self):
        """ADR-107 (review): a real video-editing contract writes its release version in five places (version, tools[].version, response_shape,
        doctor_shape, versioning). An installed release inside the gate that differs from the pin only there (0.6.0, 0.6.1, a later 0.6.x) must not
        become MISSING; the difference is still reported, and any other change to response_shape stays fatal."""
        import json

        def release(v):   # the pin as that release prints it: every place the version is written
            c = with_version(json.loads(json.dumps(ve.pinned_contract())), v)
            c["response_shape"]["skill"]["version"] = c["doctor_shape"]["skill"]["version"] = c["versioning"]["version"] = v
            return c
        for v in ("0.6.0", "0.6.1", "0.6.2", "0.7.0", "0.7.1"):
            live = release(v)
            self.assertEqual(ve.check_contract(live), [], v)
            self.assertEqual(ve.contract_breaking_drift(live), [], v)
            self.assertEqual(bool(ve.contract_drift(live)), v != "0.7.0", "the full drift report still names the difference")
        self.assertTrue(any(d.startswith("response_shape:") for d in ve.contract_drift(release("0.6.1"))))
        for change in (lambda c: c["response_shape"].update(status="completed"), lambda c: c["response_shape"]["skill"].update(id="other"),
                       lambda c: c["response_shape"].update(schema="video-editing/response@2"), lambda c: c["response_shape"].pop("execution")):
            live = release("0.6.1")
            change(live)
            self.assertTrue(any(d.startswith("response_shape:") for d in ve.contract_breaking_drift(live)), live["response_shape"])

    def test_color_grading_versions(self):
        # ADR-097 addendum 6: color-grading-skill 0.5.0 (OCIO_TRANSFORM / MATCH_CAMERA / LEGALIZE, the alpha guard; pinned blocks unchanged)
        # ADR-106: 0.6.0 (LEGALIZE clamp and measured shares, keep_alpha for LUT_APPLY / PRIMARY_CORRECTION; additive, pinned blocks unchanged)
        # ADR-106 addendum 1: 0.6.1 rewords hdr_sdr.tone_mapping_selection (the transfer-based default of 0.5.1, written down); re-pinned
        # ADR-107: 1.0.0 was an unintended version number for the 0.6 line (the 0.6.1 contract, version only), so the gate takes 1.0.x and 0.6.x of the
        # same contract; 1.0 is not a new contract generation, and 0.7 / 1.1 / 2.0 stay out. Addendum 1: the 1.0 line is kept; 1.0.1 is version-only
        self.assertEqual(cg.pinned_contract()["version"], "1.0.1")   # ADR-105: 0.5.1, HLG tone map defaults to mobius
        for v in ("0.2.0", "0.4.0", "0.5.0", "0.5.2", "0.6.0", "0.6.1", "0.6.2", "0.6.3", "1.0.0", "1.0.1"):
            live = with_version(cg.pinned_contract(), v)
            self.assertEqual(cg.check_contract(live), [], v)
            self.assertEqual(cg.contract_breaking_drift(live), [], v)
        for v in ("0.7.0", "1.1.0", "2.0.0", "10.0.0"):
            self.assertTrue(cg.check_contract(with_version(cg.pinned_contract(), v)), v)

    def test_audio_production_versions(self):
        # ADR-107: 1.0.0 was an unintended version number for the 0.4 line (the 0.4.10 contract, version only), so the gate takes 1.0.x and 0.4.x of the
        # same contract; 1.0 is not a new contract generation, and 0.5 / 1.1 / 2.0 stay out. Addendum 1: the 1.0 line is kept; 1.0.1 is version-only
        self.assertEqual(ap.pinned_contract()["version"], "1.0.1")
        for v in ("0.1.0", "0.2.0", "0.3.0", "0.4.0", "0.4.10", "0.4.11", "0.4.12", "1.0.0", "1.0.1"):
            live = with_version(ap.pinned_contract(), v)
            self.assertEqual(ap.check_contract(live), [], v)
            self.assertEqual(ap.contract_breaking_drift(live), [], v)
        for v in ("0.5.0", "1.1.0", "2.0.0", "10.0.0"):
            self.assertTrue(ap.check_contract(with_version(ap.pinned_contract(), v)), v)


if __name__ == "__main__":
    unittest.main(verbosity=2)
