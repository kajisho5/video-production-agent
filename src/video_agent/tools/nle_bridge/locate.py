"""Locate an installed nle-bridge-skill (kajisho5/nle-bridge-skill): a checkout (<dir>/src/nle_bridge, run as `python -m nle_bridge.cli` with
PYTHONPATH=<dir>/src, without the current directory on sys.path) or the console script `nle-bridge`. The agent never imports it (ADR-086)."""
from __future__ import annotations

from typing import Dict, Optional

from ..skill_process import CliSkill, locate_cli_skill

ENV_DIR = "VIDEO_AGENT_NLE_BRIDGE_DIR"   # a nle-bridge-skill checkout (<dir>/src/nle_bridge), checked before PATH
PACKAGE_ID = "nle-bridge"                # the ecosystem package id, the contract's skill_id and the tool id prefix


def locate_nle_bridge(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[CliSkill]:
    """A nle-bridge-skill checkout or the `nle-bridge` console script; the same order as every other Skill CLI (skill_process.locate_cli_skill)."""
    return locate_cli_skill(PACKAGE_ID, "nle_bridge.cli", "nle_bridge", "nle-bridge", ENV_DIR, explicit, env, ("nle-bridge-skill",))
