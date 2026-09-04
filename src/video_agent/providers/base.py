"""AI provider boundary. Phase 1 ships only NullProvider: the deterministic pipeline must work without any
AI API, and no core media function may depend on one (MASTER_SPEC §42)."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


class AIProvider:
    name = "abstract"

    def available(self) -> bool:
        return False

    def extract_requirements(self, text: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return structured requirement candidates (provenance AI_GENERATED) or None."""
        return None


class NullProvider(AIProvider):
    name = "null"


def get_provider(name: Optional[str] = None) -> AIProvider:
    name = name or os.environ.get("VIDEO_AGENT_AI_PROVIDER", "null")
    # Future: "anthropic", "openai", "local" adapters. They must stay optional.
    return NullProvider()
