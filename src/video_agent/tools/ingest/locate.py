"""Locate an installed ingest-skill (kajisho5/ingest-skill): a checkout (<dir>/src/ingest_skill, run as `python -m ingest_skill.cli`
with PYTHONPATH=<dir>/src) or the console script `ingest`. The agent never imports it. Shared by the card offload adapter
(ADR-066) and the URL fetch path (agent/ingest.py, ADR-059)."""
from __future__ import annotations

from typing import Dict, Optional

from ..skill_process import CliSkill, locate_cli_skill

ENV_DIR = "VIDEO_AGENT_INGEST_DIR"      # an ingest-skill checkout (<dir>/src/ingest_skill), checked before PATH
PACKAGE_ID = "ingest"                   # the ecosystem package id (skills/ecosystem.py) and the tool id prefix


def locate_ingest(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[CliSkill]:
    """An ingest-skill checkout or the `ingest` console script; the same order as every other Skill CLI (skill_process.locate_cli_skill)."""
    return locate_cli_skill(PACKAGE_ID, "ingest_skill.cli", "ingest_skill", "ingest", ENV_DIR, explicit, env, ("ingest-skill",))
