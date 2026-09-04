"""Locate an installed media-analysis-skill (kajisho5/media-analysis-skill). Two shapes are accepted: a source checkout
(<dir>/src/media_analysis/cli.py, run as `python -m media_analysis.cli` with PYTHONPATH=<dir>/src) or the installed
console script `media-analysis` on PATH. The agent never imports the package."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ENV_DIR = "VIDEO_AGENT_MEDIA_ANALYSIS_DIR"


@dataclass
class MediaAnalysisSkill:
    command: List[str]            # argv prefix, e.g. [python, "-m", "media_analysis.cli"] or ["media-analysis"]
    root: Optional[Path]          # checkout root when run from source
    env: Dict[str, str]           # extra environment for the child (PYTHONPATH for a checkout)

    def describe(self) -> str:
        return str(self.root) if self.root else self.command[0]


def _from_checkout(root: Path) -> Optional[MediaAnalysisSkill]:
    cli = root / "src" / "media_analysis" / "cli.py"
    if cli.is_file():
        return MediaAnalysisSkill([sys.executable, "-m", "media_analysis.cli"], root, {"PYTHONPATH": str(root / "src")})
    return None


def locate_media_analysis(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[MediaAnalysisSkill]:
    env = os.environ if env is None else env
    cands: List[Path] = []
    if explicit:
        cands.append(Path(explicit))
    if env.get(ENV_DIR):
        cands.append(Path(env[ENV_DIR]))
    cands += [Path.home() / ".claude" / "skills" / "media-analysis-skill", Path.cwd() / "vendor" / "media-analysis-skill", Path.cwd().parent / "media-analysis-skill"]
    for c in cands:
        found = _from_checkout(c)
        if found:
            return found
    exe = shutil.which("media-analysis")
    if exe:
        return MediaAnalysisSkill([exe], None, {})
    return None
