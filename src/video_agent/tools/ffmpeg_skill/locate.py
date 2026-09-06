"""Find an ffmpeg-skill checkout/installation. Search order: explicit dir, VIDEO_AGENT_FFMPEG_SKILL_DIR,
~/.claude/skills/ffmpeg-skill, ./vendor/ffmpeg-skill, ../ffmpeg-skill."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

SUPPORTED_MIN = (0, 8, 4)
# 0.10.0 (2026-09-06): per-tool `doctor --json` usable fields, `contract --json` reencodes_video/reencodes_audio,
# join.py's audio-less multi-clip filtergraph-index fix, no other script changed its media behaviour (CHANGELOG).
# The full real-Skill integration suite (tests/test_integration.py, all 9 Skills, no mocks) passes on 0.10.0, so
# 0.9.x-0.10.x are accepted; 0.11 is not verified. Widening this needs a verified integration run, not a silent edit.
SUPPORTED_MAX_EXCLUSIVE = (0, 11, 0)


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
    authoritative: List[Path] = []
    if explicit:
        authoritative.append(Path(explicit))
    if env.get("VIDEO_AGENT_FFMPEG_SKILL_DIR"):
        authoritative.append(Path(env["VIDEO_AGENT_FFMPEG_SKILL_DIR"]))
    # An explicit dir or the env var is the caller naming exactly where to look; if that candidate has no
    # checkout there, that is the answer (MISSING), not a cue to go on guessing sibling directories that
    # happen to exist here too -- a bad override must fail loudly, never resolve to some other checkout.
    cands = authoritative or [Path.home() / ".claude" / "skills" / "ffmpeg-skill", Path.cwd() / "vendor" / "ffmpeg-skill", Path.cwd().parent / "ffmpeg-skill"]
    for c in cands:
        found = _candidate(c)
        if found:
            return found
    return None
