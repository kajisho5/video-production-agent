"""Locate an installed artifacts-skill (kajisho5/artifacts-skill): a checkout (<dir>/src/artifact_skill, run as
`python -m artifact_skill.cli.main` without the current directory on sys.path, PYTHONPATH=<dir>/src) or the console script
`artifacts-skill`. The agent never imports it."""
from __future__ import annotations

from typing import Dict, Optional

from ..skill_process import CliSkill, locate_cli_skill

ENV_DIR = "VIDEO_AGENT_ARTIFACTS_DIR"   # an artifacts-skill checkout (<dir>/src/artifact_skill), checked before PATH
PACKAGE_ID = "artifacts-skill"          # the contract's skill_id and the prefix of its provides[].tool_id (ADR-077's rule), the ecosystem package id


def locate_artifacts(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[CliSkill]:
    """An artifacts-skill checkout or the `artifacts-skill` console script; the same order as every other Skill CLI (skill_process.locate_cli_skill)."""
    return locate_cli_skill(PACKAGE_ID, "artifact_skill.cli.main", "artifact_skill", "artifacts-skill", ENV_DIR, explicit, env, ("artifacts-skill",))
