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

The brief (ADR-061, task `requirements_extraction`) needs no recorded ids, so its two steps are two plans (or the cheap
`brief` command): `plan --brief TEXT --ai-brief` returns `host_request` (the brief, the allowed keys, the result shape and
HOST_BRIEF_RULES) and reads the brief with the rules meanwhile; `plan --brief TEXT --brief-requirements FILE` validates the chat
model's answer like any provider's (allowed keys only, a quoted phrase of the brief, the same checks as `--set`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import AIProvider, AIRequest, AIResponse, AIUsage

HOST_RULES = (
    "Recommend production intents for these assets from the measured facts. Only use intents from context.allowed_intents "
    "and asset ids from inputs.assets. Every recommendation must cite, in evidence, ids of observations or events from "
    "the inputs; one without a cited id is dropped. Do not invent measurements, and do not put commands, tool names, "
    "file paths or code anywhere. params are optional intent parameters. Weigh context.request_text (what the user "
    "asked for). Return {\"recommendations\": [{\"intent\", \"asset_id\", \"statement\", \"confidence\" (0-1), "
    "\"evidence\": [ids], \"params\": {}}]}; an empty list is a valid answer.")

HOST_BRIEF_RULES = (
    "Read the user's brief (inputs.brief, any language) into production requirements. Use only keys from context.allowed_keys; "
    "any other key is refused. value is what `--set key=value` would take (true / false, a number, or text such as 9:16). "
    "phrase must quote, exactly, the words of the brief the requirement comes from; one without such a phrase is dropped. "
    "Keys in context.explicit_keys were set by the user and keep their value. Every value is checked like --set and refused "
    "with the reason when invalid. Do not guess what the brief does not say; put what no allowed key expresses (for example "
    "a target length) in unmapped with the reason. Return {\"requirements\": [{\"key\", \"value\", \"phrase\", \"confidence\" (0-1)}], "
    "\"unmapped\": [{\"phrase\", \"reason\"}]}.")


class HostProvider(AIProvider):
    name = "host"

    def __init__(self, recommendations: Optional[List[Dict[str, Any]]] = None, model: str = "host-chat", reasoning: str = "",
                 requirements: Optional[List[Dict[str, Any]]] = None, unmapped: Optional[List[Dict[str, Any]]] = None):
        self.recommendations = recommendations   # None: brief step (no answer yet); a list: the chat model's answer
        self.requirements = requirements         # requirements_extraction (ADR-061): None until the chat model answered the brief
        self.unmapped = unmapped or []
        self.model = model
        self.reasoning = reasoning
        self.requests: List[AIRequest] = []

    def available(self) -> bool:
        return True

    def describe(self) -> Dict[str, Any]:
        return {"provider": self.name, "model": self.model, "available": True, "step": "brief" if self.recommendations is None else "recommend"}

    def complete(self, request: AIRequest, timeout: Optional[float] = None) -> AIResponse:
        self.requests.append(request)
        if request.task_type == "requirements_extraction":
            reqs = list(self.requirements or [])
            return AIResponse(task_type=request.task_type, result={"requirements": reqs, "unmapped": list(self.unmapped)}, confidence=1.0 if reqs else 0.0,
                              provider=self.name, model=self.model, reasoning=self.reasoning[:1000], usage=AIUsage(), metadata={"step": "answer"})
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


def load_brief_answer(doc: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The chat model's answer to a brief: a bare list of requirements or {"requirements": [...], "unmapped": [...]}."""
    reqs = doc.get("requirements") if isinstance(doc, dict) else doc
    if not isinstance(reqs, list):
        raise ValueError("the brief answer must be a list of requirements, or an object with a requirements list")
    unmapped = doc.get("unmapped") if isinstance(doc, dict) and isinstance(doc.get("unmapped"), list) else []
    return [r for r in reqs if isinstance(r, dict)], [u for u in unmapped if isinstance(u, dict)]


def load_recommendations(doc: Any) -> List[Dict[str, Any]]:
    """Accept either a bare list or {"recommendations": [...]} (the shape the brief asks for)."""
    recs = doc.get("recommendations") if isinstance(doc, dict) else doc
    if not isinstance(recs, list):
        raise ValueError("recommendations must be a list, or an object with a recommendations list")
    return [r for r in recs if isinstance(r, dict)]
