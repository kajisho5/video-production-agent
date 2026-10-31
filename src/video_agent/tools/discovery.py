"""Where a Skill checkout is looked for when nobody named one, and the environment a Skill child process gets (ADR-069).

Discovery never depends on the current directory: starting the agent inside a folder (an unpacked project, a media card,
a download) must not make that folder's `vendor/<skill>` or its parent's `<skill>` the code that runs. When no explicit
directory or `VIDEO_AGENT_*_DIR` is given, the defaults are `~/.claude/skills/<name>` and, only when this package runs
from its own source checkout (`<root>/src/video_agent` with a `pyproject.toml` or `.git` at `<root>`), the sibling
`<root>/../<name>` -- the ecosystem's development layout and the CI integration job, wherever the command is started.

A Skill child process gets the agent's environment minus the AI provider credentials the agent itself reads
(`ANTHROPIC_*`, `OPENAI_*`): no Skill calls a model on the agent's behalf, so none needs them."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional

# environment variables of the AI providers the agent talks to (providers/claude.py, capabilities/resolver.py AI_ENV)
SECRET_ENV_PREFIXES = ("ANTHROPIC_", "OPENAI_")


def source_checkout_root() -> Optional[Path]:
    """The repository root when this package is imported from its source checkout (an editable install or PYTHONPATH=src),
    else None (a wheel in site-packages has no siblings to offer)."""
    here = Path(__file__).resolve()
    pkg = here.parents[1]
    root = pkg.parent.parent
    if pkg.name == "video_agent" and pkg.parent.name == "src" and ((root / "pyproject.toml").is_file() or (root / ".git").exists()):
        return root
    return None


def _home() -> Path:
    return Path.home()


def default_skill_dirs(name: str) -> List[Path]:
    """Guessed locations of a Skill checkout, in order, when neither an explicit dir nor the env var names one."""
    out = [_home() / ".claude" / "skills" / name]
    root = source_checkout_root()
    if root is not None:
        out.append(root.parent / name)
    return out


def is_secret_env(key: str) -> bool:
    return key.upper().startswith(SECRET_ENV_PREFIXES)


def child_env(extra: Optional[Mapping[str, str]] = None, base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment for a Skill (or engine) child process: `base` (default os.environ) without provider credentials, plus `extra`."""
    env = {k: v for k, v in (os.environ if base is None else base).items() if not is_secret_env(k)}
    env.update(extra or {})
    return env
