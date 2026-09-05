"""Locate an installed motion-graphics-skill (kajisho5/motion-graphics-skill): a checkout (<dir>/src/motion_graphics, run as
`python -m motion_graphics.cli` with PYTHONPATH=<dir>/src) or the console script `motion-graphics`. The agent never imports it."""
from __future__ import annotations

from typing import Dict, Optional

from ..skill_process import CliSkill, locate_cli_skill

ENV_DIR = "VIDEO_AGENT_MOTION_GRAPHICS_DIR"


def locate_motion_graphics(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[CliSkill]:
    return locate_cli_skill("motion-graphics", "motion_graphics.cli", "motion_graphics", "motion-graphics", ENV_DIR, explicit, env, ("motion-graphics-skill",))
