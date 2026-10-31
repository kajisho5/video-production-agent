"""Locate an installed publish-skill (kajisho5/publish-skill): a checkout (<dir>/src/publish_skill, run as `python -m publish_skill.cli` with
PYTHONPATH=<dir>/src, without the current directory on sys.path) or the console script `publish`. The agent never imports it (ADR-083)."""
from __future__ import annotations

from typing import Dict, Optional

from ..skill_process import CliSkill, locate_cli_skill

ENV_DIR = "VIDEO_AGENT_PUBLISH_DIR"   # a publish-skill checkout (<dir>/src/publish_skill), checked before PATH
PACKAGE_ID = "publish"                # the ecosystem package id (skills/ecosystem.py), the contract's skill_id and the tool id prefix


def locate_publish(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[CliSkill]:
    """A publish-skill checkout or the `publish` console script; the same order as every other Skill CLI (skill_process.locate_cli_skill)."""
    return locate_cli_skill(PACKAGE_ID, "publish_skill.cli", "publish_skill", "publish", ENV_DIR, explicit, env, ("publish-skill",))
