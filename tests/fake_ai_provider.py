"""FakeAIProvider: canned structured responses so the AI boundary is tested without a network or an API key.
Deliberately supports hostile responses (tool ids, argv, shell commands, fabricated observations) so the tests
can prove they are ignored."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from video_agent.providers import AIProvider, AIProviderError, AIRequest, AIResponse, AIUsage


class FakeAIProvider(AIProvider):
    name = "fake"
    model = "fake-model-1"

    def __init__(self, recommendations: Optional[List[Dict[str, Any]]] = None, fail: Optional[str] = None, api_key: str = "sk-FAKE-SECRET-DO-NOT-STORE",
                 raw_result: Optional[Dict[str, Any]] = None, confidence: float = 0.8, reasoning: str = "silence at the head is a technical lead-in",
                 intent: Optional[str] = None, params: Optional[Dict[str, Any]] = None, extra: Optional[List[Dict[str, Any]]] = None):
        self.recommendations = recommendations
        self.intent, self.params, self.extra = intent, params or {}, extra or []   # dynamic: cite the request's own evidence ids
        self.raw_result = raw_result
        self.fail = fail                      # AIProviderError kind to raise, or "crash" for an unexpected exception
        self.api_key = api_key                # simulates a configured credential that must never leak into provenance
        self.confidence = confidence
        self.reasoning = reasoning
        self.requests: List[AIRequest] = []

    def available(self) -> bool:
        return True

    def complete(self, request: AIRequest, timeout: Optional[float] = None) -> AIResponse:
        self.requests.append(request)
        if self.fail == "crash":
            raise RuntimeError("provider bug")
        if self.fail:
            raise AIProviderError(self.fail, "simulated")
        recs = list(self.recommendations or [])
        if self.intent:
            recs += recommend_from_request(request, self.intent, params=self.params) + [dict(e, asset_id=request.inputs["assets"][0]["id"], evidence=[request.inputs["observations"][0]["id"]]) for e in self.extra]
        result = self.raw_result if self.raw_result is not None else {"recommendations": recs}
        return AIResponse(task_type=request.task_type, result=result, confidence=self.confidence, provider=self.name, model=self.model,
                          evidence=[], reasoning=self.reasoning, usage=AIUsage(input_tokens=120, output_tokens=40), latency_s=0.01, metadata={"finish": "stop"})


def recommend_from_request(request: AIRequest, intent: str = "silence_cleanup", **extra) -> List[Dict[str, Any]]:
    """Build a well-formed recommendation citing observation ids from the request the agent actually sent."""
    asset = request.inputs["assets"][0]
    ev = [o["id"] for o in request.inputs["observations"] if o["asset_id"] == asset["id"]][:2]
    return [{"intent": intent, "asset_id": asset["id"], "statement": f"AI recommends {intent}", "confidence": 0.9, "evidence": ev, "params": {}, **extra}]


def recommend_from_analysis(analysis, intent: str = "silence_cleanup", **extra) -> List[Dict[str, Any]]:
    """Static recommendation citing a given analysis' observation ids (for direct to_inferences tests)."""
    asset = analysis.assets[0]
    ev = [o.id for o in analysis.observations if o.asset_id == asset.id][:2]
    return [{"intent": intent, "asset_id": asset.id, "statement": f"AI recommends {intent}", "confidence": 0.9, "evidence": ev, "params": {}, **extra}]
