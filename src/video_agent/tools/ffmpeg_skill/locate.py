"""Find an ffmpeg-skill checkout/installation. Search order: explicit dir, VIDEO_AGENT_FFMPEG_SKILL_DIR,
~/.claude/skills/ffmpeg-skill, ./vendor/ffmpeg-skill, ../ffmpeg-skill."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

SUPPORTED_MIN = (0, 8, 4)
# 0.9.0 (2026-09-04): machine-readable contract / doctor added, `--json` results gain "status"; "no script changed its media
# behaviour" (CHANGELOG). The full integration suite passes on 0.9.0, so 0.9.x is accepted; 0.10 is not verified.
SUPPORTED_MAX_EXCLUSIVE = (0, 10, 0)


@dataclass
class FfmpegSkill:
    root: Path
    version: str
    scripts: List[str]

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    def script(self, name: str) -> Path:
        return self.scripts_dir / f"{name}.py"

    def version_supported(self) -> bool:
        try:
            v = tuple(int(x) for x in self.version.split(".")[:3])
        except ValueError:
            return False
        return SUPPORTED_MIN <= v < SUPPORTED_MAX_EXCLUSIVE


def _candidate(p: Path) -> Optional[FfmpegSkill]:
    if not (p / "scripts" / "probe.py").exists():
        return None
    version = "unknown"
    pj = p / "package.json"
    if pj.exists():
        try:
            version = json.loads(pj.read_text(encoding="utf-8")).get("version", "unknown")
        except ValueError:
            pass
    scripts = sorted(x.stem for x in (p / "scripts").glob("*.py") if not x.name.startswith("_"))
    return FfmpegSkill(p.resolve(), version, scripts)


def locate_ffmpeg_skill(explicit: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Optional[FfmpegSkill]:
    env = os.environ if env is None else env
    cands: List[Path] = []
    if explicit:
        cands.append(Path(explicit))
    if env.get("VIDEO_AGENT_FFMPEG_SKILL_DIR"):
        cands.append(Path(env["VIDEO_AGENT_FFMPEG_SKILL_DIR"]))
    cands += [Path.home() / ".claude" / "skills" / "ffmpeg-skill", Path.cwd() / "vendor" / "ffmpeg-skill", Path.cwd().parent / "ffmpeg-skill"]
    for c in cands:
        found = _candidate(c)
        if found:
            return found
    return None
