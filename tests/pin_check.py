"""The integration suite's check of a sibling Skill's live contract against its pin (ADR-108).

    from video_agent.tools.video_editing import adapter as ve
    assert_live_matches_pin(self, "video-editing-skill", ad.contract, ve)

(the adapter module gives the pin, its SUPPORTED_SKILL_VERSIONS gate and its drift function, `contract_drift` or `drift`)
fails when the live release version is outside the adapter's gate, or when the live contract differs from the pin in anything but its
release version (the adapter's own drift function, on the live contract with its release version restated as the pin's: see
video_agent.tools.release_version). A difference in the release version alone, inside the gate, passes and is reported, never silently:
a VersionOnlyDrift warning at the test, a GitHub Actions warning annotation on the run, and a summary at the end of the run (also in the
job summary on GitHub Actions). That release is taken into the pin at the next re-pin.
"""
from __future__ import annotations

import atexit
import os
import sys
import warnings
from types import ModuleType
from typing import Any, Dict

from video_agent.tools.release_version import PinComparison, VersionOnlyDrift, compare_to_pin

REPORTED: Dict[tuple, PinComparison] = {}   # (skill, live version, pinned version) -> comparison: each version-only drift once per run


def _summary() -> None:
    if not REPORTED:
        return
    rows = list(REPORTED.values())
    head = f"version-only contract drift: {len(rows)} Skill(s) released a version whose contract differs from the pin only in its version number"
    print("\n" + head + ":\n" + "\n".join("  " + c.describe() for c in rows), file=sys.stderr, flush=True)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"### {head}\n\n" + "".join(f"- {c.describe()}\n" for c in rows) + "\n")
        except OSError:
            pass


atexit.register(_summary)


def report_version_only(c: PinComparison) -> None:
    """Make a version-only drift visible: a warning every time, an annotation and a summary row once per Skill and version pair."""
    warnings.warn(c.describe(), VersionOnlyDrift, stacklevel=3)
    key = (c.skill, c.live_version, c.pinned_version)
    if key in REPORTED:
        return
    REPORTED[key] = c
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::warning title=Version-only contract drift ({c.skill})::{c.describe()}", flush=True)


def assert_live_matches_pin(tc: Any, skill: str, live: Dict[str, Any], adapter: ModuleType) -> PinComparison:
    """`live` (the installed Skill's contract) against `adapter`'s pin; see the module docstring."""
    drift = getattr(adapter, "contract_drift", None) or adapter.drift
    c = compare_to_pin(skill, live, adapter.pinned_contract(), adapter.SUPPORTED_SKILL_VERSIONS, drift)
    tc.assertEqual(c.out_of_range, [], f"{skill}: release version outside the adapter's gate: re-verify the adapter, then widen the gate and re-pin")
    tc.assertEqual(c.content_drift, [], f"{skill}: the installed contract differs from the pin beyond its version number: re-verify the adapter (and re-pin)")
    if c.version_only:
        report_version_only(c)
    return c
