"""The satellites now report their release version (single-sourced from pyproject.toml) instead of a stale
constant: motion-graphics 0.2.x, transcription 0.7.x, subtitle 0.2.x. Their contracts are otherwise the pinned ones, so the
version gates accept those ranges and nothing else.

    python -m unittest tests/test_version_gates.py
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from video_agent.tools.motion_graphics import adapter as mg  # noqa: E402
from video_agent.tools.subtitle import adapter as st  # noqa: E402
from video_agent.tools.transcription import adapter as tr  # noqa: E402


def with_version(contract, version):
    c = dict(contract, version=version)
    if isinstance(c.get("tools"), list):
        c["tools"] = [dict(t, version=version) if isinstance(t, dict) and "version" in t else t for t in c["tools"]]
    return c


class VersionGateTests(unittest.TestCase):
    def test_motion_graphics_release_versions(self):
        for v in ("0.1.0", "0.2.0", "0.2.1"):
            live = with_version(mg.pinned_contract(), v)
            self.assertEqual(mg.check_contract(live), [], v)
            self.assertEqual(mg.contract_breaking_drift(live), [], v)
        self.assertTrue(mg.check_contract(with_version(mg.pinned_contract(), "0.3.0")))

    def test_transcription_release_versions(self):
        for v in ("0.2.0", "0.7.3"):
            self.assertEqual(tr.check_contract(with_version(tr.pinned_contract(), v)), [], v)
        for v in ("0.3.0", "0.8.0"):
            self.assertTrue(tr.check_contract(with_version(tr.pinned_contract(), v)), v)

    def test_subtitle_release_versions(self):
        # subtitle-skill 0.2.0 (#19, ja/zh/ko locale profiles) reports its release version; the contract is otherwise the 0.1.x one
        for v in ("0.1.5", "0.2.0", "0.2.4"):
            live = with_version(st.pinned_contract(), v)
            self.assertEqual(st.check_contract(live), [], v)
            self.assertEqual(st.contract_breaking_drift(live), [], v)
        self.assertTrue(st.check_contract(with_version(st.pinned_contract(), "0.3.0")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
