"""Policy / Preference / Constraint. Precedence GLOBAL → ORGANIZATION → EVENT → PROJECT → PROFILE → REQUEST;
CONSTRAINTs are never overridden: a conflicting lower-precedence rule becomes a Conflict for the decision engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCOPES = ["GLOBAL", "ORGANIZATION", "EVENT", "PROJECT", "PROFILE", "REQUEST"]
KINDS = ("POLICY", "PREFERENCE", "CONSTRAINT")


@dataclass
class Rule:
    id: str
    kind: str
    scope: str
    key: str
    value: Any
    source: str = ""

    @property
    def hard(self) -> bool:
        return self.kind == "CONSTRAINT"

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "scope": self.scope, "key": self.key, "value": self.value, "source": self.source, "hard": self.hard}


@dataclass
class Conflict:
    key: str
    constraint: Rule
    attempted: Rule

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "constraint": self.constraint.to_dict(), "attempted": self.attempted.to_dict()}


@dataclass
class RuleSet:
    effective: Dict[str, Rule] = field(default_factory=dict)
    conflicts: List[Conflict] = field(default_factory=list)
    all_rules: List[Rule] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        r = self.effective.get(key)
        return default if r is None else r.value

    def provenance(self, key: str) -> Optional[str]:
        r = self.effective.get(key)
        if not r:
            return None
        return {"REQUEST": "USER", "PROFILE": "PROFILE"}.get(r.scope, "SYSTEM")

    def to_dict(self) -> Dict[str, Any]:
        return {"effective": {k: v.to_dict() for k, v in self.effective.items()}, "conflicts": [c.to_dict() for c in self.conflicts]}


def resolve_rules(rules: List[Rule]) -> RuleSet:
    rs = RuleSet(all_rules=list(rules))
    ordered = sorted(rules, key=lambda r: SCOPES.index(r.scope) if r.scope in SCOPES else -1)
    for r in ordered:
        cur = rs.effective.get(r.key)
        if cur is not None and cur.hard and not r.hard and cur.value != r.value:
            rs.conflicts.append(Conflict(r.key, cur, r))
            continue
        rs.effective[r.key] = r
    return rs


# Constraints every job carries regardless of profile (MASTER_SPEC §4, §43).
SYSTEM_CONSTRAINTS = [
    Rule("sys.preserve_source", "CONSTRAINT", "GLOBAL", "workspace.preserve_source", True, "system"),
    Rule("sys.workspace_boundary", "CONSTRAINT", "GLOBAL", "workspace.boundary", True, "system"),
    Rule("sys.no_raw_shell", "CONSTRAINT", "GLOBAL", "execution.no_raw_shell", True, "system"),
    Rule("sys.max_attempts", "POLICY", "GLOBAL", "execution.recovery.max_attempts", 2, "system"),
]
