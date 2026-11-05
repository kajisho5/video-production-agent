"""Locate an installed detection-skill (kajisho5/detection-skill): a checkout (<dir>/src/detection_skill, run as
`python -m detection_skill.cli` without the current directory on sys.path, PYTHONPATH=<dir>/src) or the console script `detection`.
The agent never imports it."""
from __future__ import annotations

from typing import Dict, Optional

from ..skill_process import CliSkill, locate_cli_skill

ENV_DIR = "VIDEO_AGENT_DETECTION_DIR"   # a detection-skill checkout (<dir>/src/detection_skill), checked before PATH
PACKAGE_ID = "detection"                # the ecosystem package id (skills/ecosystem.py), the contract's skill_id and the tool id prefix


def locate_detection(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[CliSkill]:
    """A detection-skill checkout or the `detection` console script; the same order as every other Skill CLI (skill_process.locate_cli_skill)."""
    return locate_cli_skill(PACKAGE_ID, "detection_skill.cli", "detection_skill", "detection", ENV_DIR, explicit, env, ("detection-skill",))
