"""Locate an installed sound-design-skill (kajisho5/sound-design-skill): a checkout (<dir>/src/sound_design_skill, run as
`python -m sound_design_skill.cli` without the current directory on sys.path, PYTHONPATH=<dir>/src) or the console script `sound-design`."""
from __future__ import annotations

from typing import Dict, Optional

from ..skill_process import CliSkill, locate_cli_skill

ENV_DIR = "VIDEO_AGENT_SOUND_DESIGN_DIR"   # a sound-design-skill checkout (<dir>/src/sound_design_skill), checked before PATH
PACKAGE_ID = "sound-design"                # the ecosystem package id (skills/ecosystem.py), the contract's skill_id and the tool id prefix


def locate_sound_design(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[CliSkill]:
    """A sound-design-skill checkout or the `sound-design` console script; the same order as every other Skill CLI (skill_process.locate_cli_skill)."""
    return locate_cli_skill(PACKAGE_ID, "sound_design_skill.cli", "sound_design_skill", "sound-design", ENV_DIR, explicit, env, ("sound-design-skill",))
