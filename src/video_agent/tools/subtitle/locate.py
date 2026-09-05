"""Locate an installed subtitle-skill (kajisho5/subtitle-skill): a checkout (<dir>/src/subtitle_skill, run as
`python -m subtitle_skill` with PYTHONPATH=<dir>/src) or the console script `subtitle-skill`. The agent never imports it."""
from __future__ import annotations

from typing import Dict, Optional

from ..skill_process import CliSkill, locate_cli_skill

ENV_DIR = "VIDEO_AGENT_SUBTITLE_DIR"


def locate_subtitle(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[CliSkill]:
    return locate_cli_skill("subtitle", "subtitle_skill", "subtitle_skill", "subtitle-skill", ENV_DIR, explicit, env, ("subtitle-skill",))
