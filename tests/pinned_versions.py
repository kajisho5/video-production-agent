"""The release each sibling Skill is pinned at, read from its pin file's name (ADR-117 addendum 1).

Tests compare what an adapter, a fake or a handoff reports with `pinned_version("<sibling>")` instead of quoting the release, so an
automatic re-pin (scripts/repin.py) moves them without rewriting a line. The name is the one `scripts/repin.py` and
`integration/siblings.lock.json` use ("video-editing", "hyperframes", "blender", ...). The version comes from the file name
(`contract_<version>.json`, `<package>_<version>.json`), not from the contract inside it, so a test that compares the contract's own
`version` with it still checks the pin against itself.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]


def _repin():
    mod = sys.modules.get("repin")
    if mod is None or not hasattr(mod, "current_pin"):
        spec = importlib.util.spec_from_file_location("repin", ROOT / "scripts" / "repin.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["repin"] = mod
        spec.loader.exec_module(mod)
    return mod


def pin_path(name: str) -> Path:
    """The one pin file of the sibling `name` (the repin / lock name)."""
    rp = _repin()
    return rp.current_pin(ROOT, rp.BY_NAME[name])[0]


def pinned_version(name: str) -> str:
    rp = _repin()
    return rp.current_pin(ROOT, rp.BY_NAME[name])[1]


def version_tuple(version: str) -> Tuple[int, int, int]:
    return _repin().parse_version(version)
