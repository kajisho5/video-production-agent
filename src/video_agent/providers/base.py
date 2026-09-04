"""AI Provider boundary (MASTER_SPEC §42, ADR-018).

A provider connects the agent to a model. It receives an AIRequest, returns a structured AIResponse with its own
identity (provider, model), usage and latency, and nothing else. It never selects skills or tools, never sees the
tool layer, never executes anything. The deterministic pipeline must work with NullProvider (no AI at all).

AI output is untrusted input: the reasoning layer (agent/ai_reasoning.py) validates it against the system-defined
structure before anything downstream sees it. No credentials are ever placed in requests, responses or provenance.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

TASK_TYPES = ("production_recommendation", "requirements_extraction")
FAILURE_KINDS = ("TIMEOUT", "RATE_LIMIT", "MALFORMED", "UNAVAILABLE", "AUTH", "BUDGET")


class AIProviderError(Exception):
    """AI failure domain, distinct from tool / engine failures (never a media-engine incident)."""

    def __init__(self, kind: str, message: str = ""):
        if kind not in FAILURE_KINDS:
            kind = "UNAVAILABLE"
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind = kind


@dataclass
class AIUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_estimate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "cost_estimate": self.cost_estimate}


@dataclass
class AIRequest:
    """What the agent asks. `inputs` carries system-produced evidence summaries (observation / event ids and data),
    never media, never credentials. `schema` describes the structured result the agent expects back."""
    task_type: str
    inputs: Dict[str, Any]
    schema: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)   # profile name, intent, allowed vocabularies

    def __post_init__(self) -> None:
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"unknown AI task type {self.task_type!r}; allowed: {', '.join(TASK_TYPES)}")

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps({"task_type": self.task_type, "inputs": self.inputs, "schema": self.schema, "context": self.context},
                                         sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class AIResponse:
    """Structured reasoning result. `result` follows the request schema; `reasoning` is a short decision reason
    (not a chain of thought); `evidence` are ids of system-produced observations / events the result rests on."""
    task_type: str
    result: Dict[str, Any]
    confidence: float
    provider: str
    model: str
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""
    usage: AIUsage = field(default_factory=AIUsage)
    latency_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)   # provider-specific, non-secret (e.g. finish reason)

    def response_hash(self) -> str:
        return hashlib.sha256(json.dumps({"task_type": self.task_type, "result": self.result, "confidence": self.confidence,
                                          "evidence": self.evidence, "reasoning": self.reasoning}, sort_keys=True, default=str).encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {"task_type": self.task_type, "result": self.result, "confidence": self.confidence, "provider": self.provider, "model": self.model,
                "evidence": list(self.evidence), "reasoning": self.reasoning, "usage": self.usage.to_dict(), "latency_s": self.latency_s,
                "metadata": dict(self.metadata), "response_hash": self.response_hash()}


class AIProvider:
    """Provider contract: identity + one structured completion call. Subclasses wrap a model API; they raise
    AIProviderError (TIMEOUT / RATE_LIMIT / MALFORMED / UNAVAILABLE / AUTH) instead of returning partial data."""
    name = "abstract"
    model = ""

    def available(self) -> bool:
        return False

    def describe(self) -> Dict[str, Any]:
        """Identity for doctor / provenance. Must never include credentials."""
        return {"provider": self.name, "model": self.model, "available": self.available()}

    def complete(self, request: AIRequest, timeout: Optional[float] = None) -> AIResponse:
        raise AIProviderError("UNAVAILABLE", f"provider {self.name} cannot complete requests")


class NullProvider(AIProvider):
    """No AI. The pipeline stays fully deterministic."""
    name = "null"
    model = "none"


def get_provider(name: Optional[str] = None) -> AIProvider:
    """Provider by name (default $VIDEO_AGENT_AI_PROVIDER, else null). Real providers (anthropic / openai / gemini /
    local) are future adapters behind this contract; none is bundled, so any other name falls back to NullProvider."""
    name = (name or os.environ.get("VIDEO_AGENT_AI_PROVIDER", "null")).lower()
    return NullProvider()
