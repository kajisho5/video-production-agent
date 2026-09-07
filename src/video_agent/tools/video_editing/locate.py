"""Locate an installed video-editing-skill (kajisho5/video-editing-skill). Two shapes are accepted: a source checkout
(<dir>/src/video_editing_skill/cli.py, run as `python -m video_editing_skill.cli` with PYTHONPATH=<dir>/src) or the
installed console script `video-editing` on PATH. The agent never imports the package: the CLI is the boundary."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ENV_DIR = "VIDEO_AGENT_VIDEO_EDITING_DIR"


@dataclass
class VideoEditingSkill:
    command: List[str]            # argv prefix, e.g. [python, "-m", "video_editing_skill.cli"] or ["video-editing"]
    root: Optional[Path]          # checkout root when run from source
    env: Dict[str, str]           # extra environment for the child (PYTHONPATH for a checkout); never credentials

    def describe(self) -> str:
        return str(self.root) if self.root else self.command[0]


def _from_checkout(root: Path) -> Optional[VideoEditingSkill]:
    cli = root / "src" / "video_editing_skill" / "cli.py"
    if cli.is_file():
        return VideoEditingSkill([sys.executable, "-m", "video_editing_skill.cli"], root, {"PYTHONPATH": str(root / "src")})
    return None


def locate_video_editing(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[VideoEditingSkill]:
    env = os.environ if env is None else env
    authoritative: List[Path] = []
    if explicit:
        authoritative.append(Path(explicit))
    if env.get(ENV_DIR):
        authoritative.append(Path(env[ENV_DIR]))
    # An explicit dir or the env var names exactly where to look; if there is no checkout there, that is the
    # answer (MISSING), not a cue to keep guessing sibling directories that happen to exist here too.
    cands = authoritative or [Path.home() / ".claude" / "skills" / "video-editing-skill", Path.cwd() / "vendor" / "video-editing-skill", Path.cwd().parent / "video-editing-skill"]
    for c in cands:
        found = _from_checkout(c)
        if found:
            return found
    exe = shutil.which("video-editing", path=env.get("PATH"))
    if exe:
        return VideoEditingSkill([exe], None, {})
    return None
