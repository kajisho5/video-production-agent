"""Locate an installed dubbing-skill (kajisho5/dubbing-skill): a checkout (<dir>/src/dubbing_skill, run as `python -m dubbing_skill.cli`
with PYTHONPATH=<dir>/src, without the current directory on sys.path) or the console script `dubbing`. The agent never imports it (ADR-085)."""
from __future__ import annotations

from typing import Dict, Optional

from ..skill_process import CliSkill, locate_cli_skill

ENV_DIR = "VIDEO_AGENT_DUBBING_DIR"   # a dubbing-skill checkout (<dir>/src/dubbing_skill), checked before PATH
PACKAGE_ID = "dubbing"                # the ecosystem package id (skills/ecosystem.py), the contract's skill_id and the tool id prefix


def locate_dubbing(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[CliSkill]:
    """A dubbing-skill checkout or the `dubbing` console script; the same order as every other Skill CLI (skill_process.locate_cli_skill)."""
    return locate_cli_skill(PACKAGE_ID, "dubbing_skill.cli", "dubbing_skill", "dubbing", ENV_DIR, explicit, env, ("dubbing-skill",))
