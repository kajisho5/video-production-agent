"""Locate an installed transcription-skill (kajisho5/transcription-skill). Two shapes are accepted: a source checkout
(<dir>/src/transcription_skill/cli.py, run as `python -m transcription_skill.cli` with PYTHONPATH=<dir>/src) or the
installed console script `transcription` on PATH. The agent never imports the package and never runs an ASR engine."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ENV_DIR = "VIDEO_AGENT_TRANSCRIPTION_DIR"


@dataclass
class TranscriptionSkill:
    command: List[str]            # argv prefix, e.g. [python, "-m", "transcription_skill.cli"] or ["transcription"]
    root: Optional[Path]          # checkout root when run from source
    env: Dict[str, str]           # extra environment for the child (PYTHONPATH for a checkout); never credentials

    def describe(self) -> str:
        return str(self.root) if self.root else self.command[0]


def _from_checkout(root: Path) -> Optional[TranscriptionSkill]:
    cli = root / "src" / "transcription_skill" / "cli.py"
    if cli.is_file():
        return TranscriptionSkill([sys.executable, "-m", "transcription_skill.cli"], root, {"PYTHONPATH": str(root / "src")})
    return None


def locate_transcription(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[TranscriptionSkill]:
    env = os.environ if env is None else env
    cands: List[Path] = []
    if explicit:
        cands.append(Path(explicit))
    if env.get(ENV_DIR):
        cands.append(Path(env[ENV_DIR]))
    cands += [Path.home() / ".claude" / "skills" / "transcription-skill", Path.cwd() / "vendor" / "transcription-skill", Path.cwd().parent / "transcription-skill"]
    for c in cands:
        found = _from_checkout(c)
        if found:
            return found
    exe = shutil.which("transcription", path=env.get("PATH"))
    if exe:
        return TranscriptionSkill([exe], None, {})
    return None
