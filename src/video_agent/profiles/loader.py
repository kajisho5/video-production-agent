"""Production profiles: JSON files in profiles/ with single inheritance via "extends"."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..policy.rules import Rule

ROOT = Path(__file__).resolve().parents[3]


def profiles_dir() -> Path:
    return ROOT / "profiles"


@dataclass
class Profile:
    name: str
    version: str
    data: Dict[str, Any]
    rules: List[Rule] = field(default_factory=list)
    chain: List[str] = field(default_factory=list)

    @property
    def delivery_targets(self) -> List[Dict[str, Any]]:
        return list((self.data.get("delivery") or {}).get("targets") or [])

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "chain": self.chain}


def load_profile(name: str, directory: Optional[Path] = None, _seen: Optional[List[str]] = None) -> Profile:
    directory = directory or profiles_dir()
    _seen = _seen or []
    if name in _seen:
        raise ValueError(f"profile inheritance loop: {' -> '.join(_seen + [name])}")
    path = directory / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"profile not found: {name} ({path})")
    data = json.loads(path.read_text(encoding="utf-8"))
    rules: List[Rule] = []
    chain: List[str] = []
    merged: Dict[str, Any] = {}
    if data.get("extends"):
        base = load_profile(data["extends"], directory, _seen + [name])
        rules += base.rules
        chain += base.chain
        merged.update(base.data)
    for k, v in data.items():
        if k != "rules":
            merged[k] = v
    for r in data.get("rules") or []:
        rules.append(Rule(r["id"], r.get("kind", "POLICY"), "PROFILE", r["key"], r["value"], f"profile:{name}"))
    chain.append(name)
    return Profile(name=name, version=str(data.get("version", "0")), data=merged, rules=rules, chain=chain)
