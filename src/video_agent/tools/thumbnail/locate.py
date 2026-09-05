"""Locate an installed thumbnail-skill (kajisho5/thumbnail-skill): a checkout (<dir>/src/thumbnail_skill, run as
`python -m thumbnail_skill.cli` with PYTHONPATH=<dir>/src; the package has no __main__) or the console script `thumbnail`.
The agent never imports it. The Skill needs Pillow in the interpreter that runs it: a checkout found here is only a
candidate, the adapter's contract fetch decides whether the install is usable."""
from __future__ import annotations

from typing import Dict, Optional

from ..skill_process import CliSkill, locate_cli_skill

ENV_DIR = "VIDEO_AGENT_THUMBNAIL_DIR"


def locate_thumbnail(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[CliSkill]:
    return locate_cli_skill("thumbnail", "thumbnail_skill.cli", "thumbnail_skill", "thumbnail", ENV_DIR, explicit, env, ("thumbnail-skill",))
