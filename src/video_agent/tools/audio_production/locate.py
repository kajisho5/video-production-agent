"""Locate an installed audio-production-skill (kajisho5/audio-production-skill). Two shapes are accepted: a source checkout
(<dir>/src/audio_production/cli.py, run as `python -m audio_production.cli` with PYTHONPATH=<dir>/src) or the installed
console script `audio-production` on PATH. The agent never imports the package: the CLI is the boundary."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

ENV_DIR = "VIDEO_AGENT_AUDIO_PRODUCTION_DIR"


@dataclass
class AudioProductionSkill:
    command: List[str]            # argv prefix, e.g. [python, "-m", "audio_production.cli"] or ["audio-production"]
    root: Optional[Path]          # checkout root when run from source
    env: Dict[str, str]           # extra environment for the child (PYTHONPATH for a checkout); never credentials

    def describe(self) -> str:
        return str(self.root) if self.root else self.command[0]


def _from_checkout(root: Path) -> Optional[AudioProductionSkill]:
    cli = root / "src" / "audio_production" / "cli.py"
    if cli.is_file():
        return AudioProductionSkill([sys.executable, "-m", "audio_production.cli"], root, {"PYTHONPATH": str(root / "src")})
    return None


def locate_audio_production(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[AudioProductionSkill]:
    env_map: Mapping[str, str] = os.environ if env is None else env
    cands: List[Path] = []
    if explicit:
        cands.append(Path(explicit))
    if env_map.get(ENV_DIR):
        cands.append(Path(env_map[ENV_DIR]))
    cands += [Path.home() / ".claude" / "skills" / "audio-production-skill", Path.cwd() / "vendor" / "audio-production-skill", Path.cwd().parent / "audio-production-skill"]
    for c in cands:
        found = _from_checkout(c)
        if found:
            return found
    exe = shutil.which("audio-production", path=env_map.get("PATH"))
    if exe:
        return AudioProductionSkill([exe], None, {})
    return None
