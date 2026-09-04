"""AI reasoning boundary: Observations → (AI Provider) → validated, evidence-bound Inferences (provenance AI_GENERATED).

What this module guarantees (ADR-018):
- The provider only ever sees system-produced evidence summaries (ids + measured data), never media or credentials.
- Whatever comes back is untrusted: only the fields of the system-defined result structure are read. Tool ids,
  argv, commands, IR fragments or anything else in the response are ignored.
- An AI recommendation must cite existing observation / event ids; otherwise it is dropped. It can never create an
  Observation (OBSERVED is reserved for measurements).
- The recommendation vocabulary is the registry's production-skill names. The AI names an intent; the registry,
  capabilities and the router decide whether and how it can be executed. Risk / approval come from policy, not from
  the response.
- Every call is budgeted (max_ai_calls) and recorded in provenance.ai_calls (provider, model, task, hash, usage,
  latency, outcome). One attempt per call, no retry.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from ..media.analysis import safe_observation_summary
from ..media.analyzer import AnalysisResult
from ..models import Inference, now_iso
from ..providers import AIProvider, AIProviderError, AIRequest, AIResponse

AI_KIND_PREFIX = "ai_recommendation:"
RESULT_SCHEMA = {"recommendations": [{"intent": "<production skill name>", "asset_id": "<asset id>", "statement": "<one sentence>",
                                      "confidence": "0..1", "evidence": ["<observation / event id>"], "params": {}}]}


def build_request(analysis: AnalysisResult, skills: List[str], context: Optional[Dict[str, Any]] = None) -> AIRequest:
    """Evidence summary for the provider: assets (id, type, technical facts), observations (id, kind, data) and timeline
    events (id, type, range). Paths are reduced to basenames; nothing executable or secret is included."""
    inputs = {
        "assets": [{"id": a.id, "name": a.path.replace("\\", "/").split("/")[-1], "type": a.type, "technical": a.technical} for a in analysis.assets],
        "observations": [x for x in (safe_observation_summary(o.to_dict()) for o in analysis.observations) if x],   # real tool measurements only, scrubbed
        "events": [{"id": e.id, "type": e.type, "timeline_id": e.timeline_id, "range": e.range, "metadata": e.metadata} for e in analysis.timeline.events],
    }
    ctx = {"allowed_intents": sorted(skills), **(context or {})}
    return AIRequest(task_type="production_recommendation", inputs=inputs, schema=RESULT_SCHEMA, context=ctx)


def to_inferences(response: AIResponse, analysis: AnalysisResult, skills: List[str]) -> Tuple[List[Inference], List[str]]:
    """Validate a provider response into Inferences. Returns (inferences, warnings about dropped items)."""
    known_ids = {o.id for o in analysis.observations if getattr(o, "provenance", "OBSERVED") == "OBSERVED"} | {e.id for e in analysis.timeline.events}
    asset_ids = {a.id for a in analysis.assets}
    out: List[Inference] = []
    warnings: List[str] = []
    recs = response.result.get("recommendations") if isinstance(response.result, dict) else None
    if not isinstance(recs, list):
        return out, [f"ai: response has no recommendations list (task {response.task_type})"]
    for i, rec in enumerate(recs):
        if not isinstance(rec, dict):
            warnings.append(f"ai: recommendation {i} is not an object; dropped")
            continue
        intent = str(rec.get("intent") or "")
        if intent not in skills:
            warnings.append(f"ai: recommendation {i} names intent {intent!r} which is not a registered production skill; dropped")
            continue
        asset_id = str(rec.get("asset_id") or "")
        if asset_id not in asset_ids:
            warnings.append(f"ai: recommendation {i} ({intent}) cites unknown asset {asset_id!r}; dropped")
            continue
        evidence = [str(e) for e in (rec.get("evidence") or []) if str(e) in known_ids]
        if not evidence:
            warnings.append(f"ai: recommendation {i} ({intent}) cites no existing observation or event; dropped")
            continue
        try:
            conf = min(1.0, max(0.0, float(rec.get("confidence", response.confidence))))
        except (TypeError, ValueError):
            conf = 0.0
        params = rec.get("params") if isinstance(rec.get("params"), dict) else {}
        params = {k: v for k, v in params.items() if k not in ("tool", "tools", "argv", "command", "commands", "shell", "risk", "approval", "provenance")}
        out.append(Inference(kind=AI_KIND_PREFIX + intent, asset_id=asset_id, statement=str(rec.get("statement") or f"AI recommends {intent}")[:500],
                             confidence=conf, evidence=evidence, provenance="AI_GENERATED",
                             data={"intent": intent, "params": params, "provider": response.provider, "model": response.model, "response_hash": response.response_hash()}))
    return out, warnings


class AIReasoner:
    """Budgeted, audited access to a provider. `calls` is the provenance.ai_calls log (shared with the IR)."""

    def __init__(self, provider: AIProvider, max_calls: int = 0, calls: Optional[List[Dict[str, Any]]] = None, timeout: Optional[float] = 60.0):
        self.provider = provider
        self.max_calls = int(max_calls)
        self.calls: List[Dict[str, Any]] = calls if calls is not None else []
        self.timeout = timeout

    @property
    def used(self) -> int:
        return len(self.calls)

    def ask(self, request: AIRequest) -> AIResponse:
        """One provider call: budget check → single attempt → provenance entry (also on failure). Never retries."""
        if self.used >= self.max_calls:
            self._log(request, None, 0.0, error=("BUDGET", f"max_ai_calls={self.max_calls} reached"))
            raise AIProviderError("BUDGET", f"max_ai_calls={self.max_calls} reached ({self.used} used)")
        t0 = time.time()
        try:
            resp = self.provider.complete(request, timeout=self.timeout)
        except AIProviderError as e:
            self._log(request, None, time.time() - t0, error=(e.kind, str(e)))
            raise
        except Exception as e:  # a provider bug is still an AI-domain failure, never an engine incident
            self._log(request, None, time.time() - t0, error=("MALFORMED", f"{type(e).__name__}: {e}"))
            raise AIProviderError("MALFORMED", f"{type(e).__name__}: {e}")
        if not isinstance(resp, AIResponse) or resp.task_type != request.task_type or not isinstance(resp.result, dict):
            self._log(request, None, time.time() - t0, error=("MALFORMED", "response is not a structured AIResponse for this task"))
            raise AIProviderError("MALFORMED", "response is not a structured AIResponse for this task")
        self._log(request, resp, time.time() - t0)
        return resp

    def _log(self, request: AIRequest, resp: Optional[AIResponse], latency: float, error: Optional[Tuple[str, str]] = None) -> None:
        self.calls.append({"at": now_iso(), "provider": self.provider.name, "model": getattr(self.provider, "model", ""), "task_type": request.task_type,
                           "request_fingerprint": request.fingerprint(), "ok": resp is not None, "response_hash": resp.response_hash() if resp else None,
                           "confidence": resp.confidence if resp else None, "usage": resp.usage.to_dict() if resp else None, "latency_s": round(latency, 3),
                           "error": {"kind": error[0], "message": error[1][:200]} if error else None})
