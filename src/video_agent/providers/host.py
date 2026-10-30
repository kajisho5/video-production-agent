"""The host provider (ADR-050): the AI model of the chat the agent runs inside -- Claude Code, Claude Desktop, any MCP
client -- is the reasoning step, with no API key and no extra cost.

The host cannot be called from here, so reasoning is two steps of the same plan:
  1. brief: `plan --ai-brief` plans deterministically and returns the request a provider would get (the evidence
     summary, the allowed intents, the result shape and the rules), for the chat model to read;
  2. recommend: `revise <project> --ai-recommendations FILE` writes the next plan version with the chat model's
     answer (a revise, because the recorded analysis keeps the ids the brief cited), which goes through the same
     agent/ai_reasoning.py validation as any provider's (cited ids must exist, unknown intents and forbidden params
     are dropped, policy decides risk and approval). The answer is always recorded as a new version.
The MCP server (mcp_server.py) exposes the same two steps as the `plan` and `revise` tools.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import AIProvider, AIRequest, AIResponse, AIUsage

HOST_RULES = (
    "Recommend production intents for these assets from the measured facts. Only use intents from context.allowed_intents "
    "and asset ids from inputs.assets. Every recommendation must cite, in evidence, ids of observations or events from "
    "the inputs; one without a cited id is dropped. Do not invent measurements, and do not put commands, tool names, "
    "file paths or code anywhere. params are optional intent parameters. Weigh context.request_text (what the user "
    "asked for). Return {\"recommendations\": [{\"intent\", \"asset_id\", \"statement\", \"confidence\" (0-1), "
    "\"evidence\": [ids], \"params\": {}}]}; an empty list is a valid answer.")


class HostProvider(AIProvider):
    name = "host"

    def __init__(self, recommendations: Optional[List[Dict[str, Any]]] = None, model: str = "host-chat", reasoning: str = ""):
        self.recommendations = recommendations   # None: brief step (no answer yet); a list: the chat model's answer
        self.model = model
        self.reasoning = reasoning
        self.requests: List[AIRequest] = []

    def available(self) -> bool:
        return True

    def describe(self) -> Dict[str, Any]:
        return {"provider": self.name, "model": self.model, "available": True, "step": "brief" if self.recommendations is None else "recommend"}

    def complete(self, request: AIRequest, timeout: Optional[float] = None) -> AIResponse:
        self.requests.append(request)
        recs = list(self.recommendations or [])
        return AIResponse(task_type=request.task_type, result={"recommendations": recs}, confidence=1.0 if recs else 0.0, provider=self.name, model=self.model,
                          reasoning=self.reasoning[:1000], usage=AIUsage(), metadata={"step": "brief" if self.recommendations is None else "recommend"})

    def brief(self) -> Optional[Dict[str, Any]]:
        """What the chat model needs to answer: the last request this plan built, plus the rules and the result shape."""
        if not self.requests:
            return None
        r = self.requests[-1]
        return {"task_type": r.task_type, "rules": HOST_RULES, "result_shape": r.schema, "inputs": r.inputs, "context": r.context,
                "next": "write the answer as JSON, then revise the project with --ai-recommendations <file>"}


def load_recommendations(doc: Any) -> List[Dict[str, Any]]:
    """Accept either a bare list or {"recommendations": [...]} (the shape the brief asks for)."""
    recs = doc.get("recommendations") if isinstance(doc, dict) else doc
    if not isinstance(recs, list):
        raise ValueError("recommendations must be a list, or an object with a recommendations list")
    return [r for r in recs if isinstance(r, dict)]
