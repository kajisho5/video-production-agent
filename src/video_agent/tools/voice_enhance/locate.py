"""Locate an installed voice-enhance-skill (kajisho5/voice-enhance-skill): a checkout (<dir>/src/voice_enhance_skill, run as
`python -m voice_enhance_skill.cli` with PYTHONPATH=<dir>/src) or the console script `voice-enhance`. The agent never imports it (ADR-073)."""
from __future__ import annotations

from typing import Dict, Optional

from ..skill_process import CliSkill, locate_cli_skill

ENV_DIR = "VIDEO_AGENT_VOICE_ENHANCE_DIR"   # a voice-enhance-skill checkout (<dir>/src/voice_enhance_skill), checked before PATH
PACKAGE_ID = "voice-enhance"                # the ecosystem package id (skills/ecosystem.py) and the tool id prefix


def locate_voice_enhance(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[CliSkill]:
    """A voice-enhance-skill checkout or the `voice-enhance` console script; the same order as every other Skill CLI (skill_process.locate_cli_skill)."""
    return locate_cli_skill(PACKAGE_ID, "voice_enhance_skill.cli", "voice_enhance_skill", "voice-enhance", ENV_DIR, explicit, env, ("voice-enhance-skill",))
