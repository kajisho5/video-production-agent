"""Locate an installed remotion-ffmpeg-skill (kajisho5/remotion-ffmpeg-skill, ADR-093): a checkout whose tools are the scripts `<dir>/scripts/<tool>.py`
(what `npx remotion-ffmpeg-skill` copies to `~/.claude/skills/remotion-ffmpeg-skill`). The agent never imports it. Every tool runs as
`python <dir>/scripts/<tool>.py ...` (the script's own directory first on sys.path, never the current directory: ADR-074 / ADR-077's form)."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional

from ..discovery import default_skill_dirs

ENV_DIR = "VIDEO_AGENT_REMOTION_DIR"     # a remotion-ffmpeg-skill checkout (<dir>/scripts/_contract.py, pipeline.py)
PACKAGE_ID = "remotion-ffmpeg-skill"     # the Skill's own skill_id and the prefix of its provides[].tool_id (remotion-ffmpeg-skill/pipeline)
REPO_NAME = "remotion-ffmpeg-skill"


@dataclass
class RemotionSkill:
    root: Path
    python: str

    def script(self, name: str) -> List[str]:
        return [self.python, str(self.root / "scripts" / f"{name}.py")]

    def describe(self) -> str:
        return str(self.root)


def _is_checkout(d: Path) -> bool:
    return (d / "scripts" / "_contract.py").is_file() and (d / "scripts" / "pipeline.py").is_file()


def locate_remotion(explicit: Optional[str] = None, env: Optional[Mapping[str, str]] = None, python: Optional[str] = None) -> Optional[RemotionSkill]:
    """An explicit dir, then VIDEO_AGENT_REMOTION_DIR (either is authoritative), then ~/.claude/skills/remotion-ffmpeg-skill and the sibling of the
    agent's own source checkout. Never the current directory."""
    env_map: Mapping[str, str] = os.environ if env is None else env
    named = [Path(p) for p in (explicit, env_map.get(ENV_DIR)) if p]
    for c in named or default_skill_dirs(REPO_NAME):
        if _is_checkout(c):
            return RemotionSkill(c.resolve(), python or sys.executable)
    return None
