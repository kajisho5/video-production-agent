"""Locate an installed slide-sync-skill (kajisho5/slide-sync-skill): a checkout (<dir>/src/slide_sync, run as `python -m slide_sync.cli`
without the current directory on sys.path, PYTHONPATH=<dir>/src) or the console script `slide-sync`. The agent never imports it."""
from __future__ import annotations

from typing import Dict, Optional

from ..skill_process import CliSkill, locate_cli_skill

ENV_DIR = "VIDEO_AGENT_SLIDE_SYNC_DIR"   # a slide-sync-skill checkout (<dir>/src/slide_sync), checked before PATH
PACKAGE_ID = "slide-sync"                # the ecosystem package id (skills/ecosystem.py), the contract's skill_id and the tool id prefix


def locate_slide_sync(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[CliSkill]:
    """A slide-sync-skill checkout or the `slide-sync` console script; the same order as every other Skill CLI (skill_process.locate_cli_skill)."""
    return locate_cli_skill(PACKAGE_ID, "slide_sync.cli", "slide_sync", "slide-sync", ENV_DIR, explicit, env, ("slide-sync-skill",))
