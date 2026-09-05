"""Locate an installed color-grading-skill (kajisho5/color-grading-skill): a checkout (<dir>/src/color_grading, run as
`python -m color_grading.cli` with PYTHONPATH=<dir>/src) or the console script `color-grading`. The agent never imports it."""
from __future__ import annotations

from typing import Dict, Optional

from ..skill_process import CliSkill, locate_cli_skill

ENV_DIR = "VIDEO_AGENT_COLOR_GRADING_DIR"


def locate_color_grading(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[CliSkill]:
    return locate_cli_skill("color-grading", "color_grading.cli", "color_grading", "color-grading", ENV_DIR, explicit, env, ("color-grading-skill",))
