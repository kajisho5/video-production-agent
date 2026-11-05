"""Locate an installed diarization-skill (kajisho5/diarization-skill): a checkout (<dir>/src/diarization_skill, run as
`python -m diarization_skill.cli` with PYTHONPATH=<dir>/src) or the console script `diarization`. The agent never imports it (ADR-086)."""
from __future__ import annotations

from typing import Dict, Optional

from ..skill_process import CliSkill, locate_cli_skill

ENV_DIR = "VIDEO_AGENT_DIARIZATION_DIR"   # a diarization-skill checkout (<dir>/src/diarization_skill), checked before PATH
PACKAGE_ID = "diarization"                # the ecosystem package id (skills/ecosystem.py), the contract's skill_id and the tool id prefix


def locate_diarization(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[CliSkill]:
    """A diarization-skill checkout or the `diarization` console script; the same order as every other Skill CLI (skill_process.locate_cli_skill)."""
    return locate_cli_skill(PACKAGE_ID, "diarization_skill.cli", "diarization_skill", "diarization", ENV_DIR, explicit, env, ("diarization-skill",))
