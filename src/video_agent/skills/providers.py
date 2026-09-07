"""Provider selection: which package a Skill with more than one usable tool candidate resolves to
(docs/CAPABILITY_MODEL.md's "Capability collision policy", Tiers 1 and 2; Tier 3 -- refusal -- lives entirely
inside `SkillRegistry.select_tool()` and needs nothing from here). Two independent choices, both optional and
both keyed by production skill name (`media_probe`, `silence_analysis`, ...), the same names `SkillRegistry`
already uses -- this module never invents a separate OS-level Capability-id namespace, since `select_tool()`
resolves per skill name, not per Capability id.

Tier 1 (explicit, Plan-time, always wins): a `provider.<skill>=<package>` requirement, the same `--set key=value`
mechanism every other requirement already uses (agent/requirements.py). Tier 2 (default-provider policy): this
module's own OS-level `DEFAULT_PROVIDERS` (skills/registry.py -- the package each collision silently resolved to
before this module existed), overridden per skill by a flat `{"skill_name": "package"}` object in a `providers.json`
file at the workspace root, when one exists. A skill with zero or one real tool candidate never consults either:
there is nothing to choose (`select_tool()`'s own short-circuit, not this module's concern)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .registry import DEFAULT_PROVIDERS

PROVIDER_REQUIREMENT_PREFIX = "provider."


def explicit_providers(user_requirements: Dict[str, Any]) -> Dict[str, str]:
    """{skill_name: package} from `--set provider.<skill_name>=<package>` requirement keys. Any other requirement
    key is left untouched -- this only ever reads keys under PROVIDER_REQUIREMENT_PREFIX."""
    out: Dict[str, str] = {}
    for key, value in (user_requirements or {}).items():
        if key.startswith(PROVIDER_REQUIREMENT_PREFIX) and len(key) > len(PROVIDER_REQUIREMENT_PREFIX):
            out[key[len(PROVIDER_REQUIREMENT_PREFIX):]] = str(value)
    return out


def default_providers(workspace: str) -> Dict[str, str]:
    """OS-level DEFAULT_PROVIDERS, overridden per skill by `<workspace>/providers.json` when present. A malformed
    file is a configuration error, raised loudly here rather than silently ignored or guessed at."""
    out = dict(DEFAULT_PROVIDERS)
    path = Path(workspace) / "providers.json"
    if not path.exists():
        return out
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"invalid workspace provider policy {path}: {e}") from e
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError(f"invalid workspace provider policy {path}: expected a flat {{\"skill_name\": \"package\"}} object")
    out.update(data)
    return out
