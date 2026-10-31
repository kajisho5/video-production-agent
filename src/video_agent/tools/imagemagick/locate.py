"""Locate an installed imagemagick-skill (kajisho5/imagemagick-skill, ADR-077): a checkout whose tools are the scripts
`<dir>/scripts/<tool>.py` (what `npx imagemagick-skill` copies to `~/.claude/skills/imagemagick-skill`). The agent never imports it.

Every tool runs as `python <dir>/scripts/<tool>.py ...`, the same launch form as ffmpeg-skill's scripts: Python puts the script's own
directory, not the current directory, first on `sys.path`, so a `json.py` or `_common.py` in the folder the agent was started in is
never imported by the Skill (ADR-074's analysis of the script form; ADR-069: no current-directory search either)."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional

from ..discovery import default_skill_dirs

ENV_DIR = "VIDEO_AGENT_IMAGEMAGICK_DIR"   # an imagemagick-skill checkout (<dir>/scripts/_contract.py)
PACKAGE_ID = "imagemagick-skill"          # the Skill's own `skill.id` and the prefix of its `provides[].tool_id` (imagemagick-skill/resize)
REPO_NAME = "imagemagick-skill"


@dataclass
class ImageMagickSkill:
    """A located checkout: its root and the interpreter that runs its scripts (the agent's own, never a shell)."""
    root: Path
    python: str

    def script(self, name: str) -> List[str]:
        """The argv prefix of one script (`_contract`, `resize`, `pad`, `strip`, ...): `[python, <root>/scripts/<name>.py]`."""
        return [self.python, str(self.root / "scripts" / f"{name}.py")]

    def describe(self) -> str:
        return str(self.root)


def _is_checkout(d: Path) -> bool:
    return (d / "scripts" / "_contract.py").is_file() and (d / "scripts" / "resize.py").is_file()


def locate_imagemagick(explicit: Optional[str] = None, env: Optional[Mapping[str, str]] = None, python: Optional[str] = None) -> Optional[ImageMagickSkill]:
    """An explicit dir, then VIDEO_AGENT_IMAGEMAGICK_DIR (either is authoritative: no checkout there is the answer), then
    ~/.claude/skills/imagemagick-skill and the sibling of the agent's own source checkout. Never the current directory (ADR-069)."""
    env_map: Mapping[str, str] = os.environ if env is None else env
    named = [Path(p) for p in (explicit, env_map.get(ENV_DIR)) if p]
    for c in named or default_skill_dirs(REPO_NAME):
        if _is_checkout(c):
            return ImageMagickSkill(c.resolve(), python or sys.executable)
    return None
