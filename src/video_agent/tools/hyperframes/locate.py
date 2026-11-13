"""Locate an installed hyperframes-skill (kajisho5/hyperframes-skill, ADR-114): a checkout with `bin/hyperframes-skill.mjs` (its CLI entrypoint, the
contract's `skill.entrypoints.cli`) and its npm dependencies installed (`npm ci`). The agent never imports it: every tool runs as
`node <dir>/bin/hyperframes-skill.mjs <tool> ...`, an argv list, never a shell. Node is found on PATH (or named by VIDEO_AGENT_NODE)."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional

from ..discovery import default_skill_dirs

ENV_DIR = "VIDEO_AGENT_HYPERFRAMES_DIR"   # a hyperframes-skill checkout (<dir>/bin/hyperframes-skill.mjs); the same variable ADR-080's live pin check reads
ENV_NODE = "VIDEO_AGENT_NODE"             # the node executable, when it is not the one on PATH
PACKAGE_ID = "hyperframes-skill"          # the Skill's own skill_id and the prefix of its provides[].tool_id (hyperframes-skill/render)
REPO_NAME = "hyperframes-skill"


@dataclass
class HyperframesSkill:
    root: Path
    node: str

    def command(self, tool: str) -> List[str]:
        return [self.node, str(self.root / "bin" / "hyperframes-skill.mjs"), tool]

    def describe(self) -> str:
        return str(self.root)


def _is_checkout(d: Path) -> bool:
    return (d / "bin" / "hyperframes-skill.mjs").is_file() and (d / "scripts" / "render.mjs").is_file()


def locate_hyperframes(explicit: Optional[str] = None, env: Optional[Mapping[str, str]] = None, node: Optional[str] = None) -> Optional[HyperframesSkill]:
    """An explicit dir, then VIDEO_AGENT_HYPERFRAMES_DIR (either is authoritative), then ~/.claude/skills/hyperframes-skill and the sibling of the agent's
    own source checkout. Never the current directory. None without a checkout. A checkout found without Node keeps `node` empty: the adapter then
    refuses it by name ("Node.js is not on PATH"), so doctor says what is missing instead of "not installed"."""
    env_map: Mapping[str, str] = os.environ if env is None else env
    exe = node or env_map.get(ENV_NODE) or shutil.which("node") or ""
    named = [Path(p) for p in (explicit, env_map.get(ENV_DIR)) if p]
    for c in named or default_skill_dirs(REPO_NAME):
        if _is_checkout(c):
            return HyperframesSkill(c.resolve(), exe)
    return None
