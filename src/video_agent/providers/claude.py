"""Anthropic (Claude) provider behind the AIProvider contract (ADR-049).

One structured completion per request, nothing else: the request's system-produced evidence summary goes in, a JSON
object constrained by a JSON schema comes out, and the reasoning layer (agent/ai_reasoning.py) still validates every
field of it -- this provider only guarantees the shape. It never sees media, tools, argv or credentials, and keeps
the "one attempt per call, no retry" rule of the reasoning layer (max_retries=0): a failure is an AIProviderError,
never partial data.

Configuration: VIDEO_AGENT_AI_PROVIDER=anthropic, VIDEO_AGENT_AI_MODEL (default claude-opus-5-5),
VIDEO_AGENT_AI_EFFORT (default medium). Credentials are whatever the `anthropic` SDK resolves from the environment
(ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, an `ant auth login` profile); they are never read, stored or logged here.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import AIProvider, AIProviderError, AIRequest, AIResponse, AIUsage

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"   # `fallbacks: "default"`: a refused request is re-run on a fallback model

SYSTEM_PROMPT = """You are the reasoning step of a video production agent. You receive measured facts about media assets \
(assets, observations, timeline events, each with an id) produced by deterministic tools, plus a list of allowed \
production intents and, in context, what the user asked for (request_text, possibly empty or in Japanese) and the \
intent the agent resolved from it. Propose which production intents are worth applying and why, favouring what the \
user asked for.

Rules:
- Only use intents from context.allowed_intents and asset ids from inputs.assets.
- Every recommendation must cite, in evidence, ids of observations or events from the inputs that support it. A \
recommendation you cannot support with a cited id is not worth making.
- Do not invent measurements. Do not write commands, tool names, file paths or code.
- params are optional intent parameters as key/value strings; leave them empty when unsure.
- confidence is your calibrated probability (0 to 1) that the recommendation is right for this material.
- reasoning is one or two plain sentences explaining the overall decision, not a step-by-step transcript.
- If nothing is worth recommending, return an empty recommendations list."""

BRIEF_SYSTEM_PROMPT = """You read a video production brief -- what the user wants, in their own words, possibly in Japanese -- \
into the agent's requirement keys (the keys of its `--set key=value` option).

Rules:
- Use only keys from context.allowed_keys. value is what `--set key=value` takes: true / false, a number, or text such as 9:16, \
written as a string (it is parsed like the command line).
- phrase quotes, exactly, the words of the brief the requirement comes from. A requirement without such a phrase is dropped.
- Keys in context.explicit_keys were set explicitly by the user and keep their value whatever you return.
- Do not guess what the brief does not say. Put what no allowed key can express (for example a target length) in unmapped \
with a short reason.
- Every value is validated afterwards exactly like the command line; an invalid one is refused, not repaired.
- confidence is your calibrated probability (0 to 1) that the requirement is what the user meant.
- reasoning is one or two plain sentences."""

_PARAMS_SCHEMA = {"type": "array", "items": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                                             "required": ["key", "value"], "additionalProperties": False}}

# production_recommendation: the shape agent/ai_reasoning.RESULT_SCHEMA describes, in JSON-schema form. params is a
# list of key/value pairs because a structured-output object cannot have open-ended keys; it is folded back to a dict.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {"type": "array", "items": {"type": "object", "properties": {
            "intent": {"type": "string"}, "asset_id": {"type": "string"}, "statement": {"type": "string"},
            "confidence": {"type": "number"}, "evidence": {"type": "array", "items": {"type": "string"}}, "params": _PARAMS_SCHEMA},
            "required": ["intent", "asset_id", "statement", "confidence", "evidence", "params"], "additionalProperties": False}},
        "confidence": {"type": "number"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["recommendations", "confidence", "evidence", "reasoning"],
    "additionalProperties": False,
}


def brief_response_schema(allowed_keys: Any) -> Dict[str, Any]:
    """requirements_extraction (ADR-061): `key` is an enum of the allowed keys the request carries, so the model cannot even
    name another key (the validation in agent/brief.py still refuses one, whoever produced it)."""
    keys = sorted({str(k) for k in (allowed_keys or [])}) or ["none"]
    return {
        "type": "object",
        "properties": {
            "requirements": {"type": "array", "items": {"type": "object", "properties": {
                "key": {"type": "string", "enum": keys}, "value": {"type": "string"}, "phrase": {"type": "string"}, "confidence": {"type": "number"}},
                "required": ["key", "value", "phrase", "confidence"], "additionalProperties": False}},
            "unmapped": {"type": "array", "items": {"type": "object", "properties": {"phrase": {"type": "string"}, "reason": {"type": "string"}},
                                                    "required": ["phrase", "reason"], "additionalProperties": False}},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["requirements", "unmapped", "confidence", "reasoning"],
        "additionalProperties": False,
    }


def _credentials_configured(env: Dict[str, str]) -> bool:
    """Whether the SDK has something to authenticate with. Checks presence only, never reads a value."""
    if env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN") or env.get("ANTHROPIC_PROFILE"):
        return True
    home = Path(env["HOME"]) if env.get("HOME") else Path.home()
    return (home / ".config" / "anthropic").is_dir()


def _fold_params(rec: Dict[str, Any]) -> Dict[str, Any]:
    pairs = rec.get("params")
    if isinstance(pairs, list):
        rec = dict(rec, params={str(p.get("key")): p.get("value") for p in pairs if isinstance(p, dict) and p.get("key")})
    return rec


class AnthropicProvider(AIProvider):
    name = "anthropic"

    def __init__(self, model: Optional[str] = None, effort: Optional[str] = None, client: Any = None, env: Optional[Dict[str, str]] = None):
        self.env = dict(os.environ if env is None else env)
        self.model = model or self.env.get("VIDEO_AGENT_AI_MODEL") or DEFAULT_MODEL
        effort = (effort or self.env.get("VIDEO_AGENT_AI_EFFORT") or DEFAULT_EFFORT).lower()
        if effort not in EFFORTS:
            raise ValueError(f"VIDEO_AGENT_AI_EFFORT {effort!r} is not one of {', '.join(EFFORTS)}")
        self.effort = effort
        self._client = client

    def available(self) -> bool:
        if self._client is not None:
            return True
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return _credentials_configured(self.env)

    def describe(self) -> Dict[str, Any]:
        return {"provider": self.name, "model": self.model, "effort": self.effort, "available": self.available()}

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise AIProviderError("UNAVAILABLE", "the anthropic package is not installed (pip install 'video-production-agent[ai]')") from e
            self._client = anthropic.Anthropic(max_retries=0)   # one attempt per call (ai_reasoning's rule)
        return self._client

    def complete(self, request: AIRequest, timeout: Optional[float] = None) -> AIResponse:
        brief = request.task_type == "requirements_extraction"   # ADR-061; the other task is production_recommendation (ADR-049)
        system = BRIEF_SYSTEM_PROMPT if brief else SYSTEM_PROMPT
        schema = brief_response_schema(request.context.get("allowed_keys")) if brief else RESPONSE_SCHEMA
        client = self._get_client()
        payload = json.dumps({"inputs": request.inputs, "context": request.context, "expected_result": request.schema}, ensure_ascii=False, sort_keys=True, default=str)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": payload}],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        t0 = time.monotonic()
        message = self._call(client, kwargs)
        latency = time.monotonic() - t0
        stop = getattr(message, "stop_reason", None)
        if stop == "refusal":
            details = getattr(message, "stop_details", None)
            raise AIProviderError("UNAVAILABLE", f"model declined the request (category {getattr(details, 'category', None)!r})")
        if stop == "max_tokens":
            raise AIProviderError("MALFORMED", f"response cut off at max_tokens={MAX_TOKENS}")
        text = next((getattr(b, "text", "") for b in getattr(message, "content", []) or [] if getattr(b, "type", "") == "text"), "")
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as e:
            raise AIProviderError("MALFORMED", f"response is not JSON: {e}") from e
        listed = "requirements" if brief else "recommendations"
        if not isinstance(data, dict) or not isinstance(data.get(listed), list):
            raise AIProviderError("MALFORMED", f"response has no {listed} list")
        try:
            confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        usage = getattr(message, "usage", None)
        served_by = str(getattr(message, "model", "") or self.model)
        result = ({"requirements": data["requirements"], "unmapped": data.get("unmapped") or []} if brief else
                  {"recommendations": [_fold_params(r) if isinstance(r, dict) else r for r in data["recommendations"]]})
        return AIResponse(task_type=request.task_type, result=result,
                          confidence=confidence, provider=self.name, model=served_by,
                          evidence=[str(e) for e in data.get("evidence") or []], reasoning=str(data.get("reasoning") or "")[:1000],
                          usage=AIUsage(input_tokens=int(getattr(usage, "input_tokens", 0) or 0), output_tokens=int(getattr(usage, "output_tokens", 0) or 0)),
                          latency_s=round(latency, 3),
                          metadata={"stop_reason": stop, "requested_model": self.model, "effort": self.effort,
                                    "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
                                    "request_id": getattr(message, "_request_id", None)})

    @staticmethod
    def _call(client: Any, kwargs: Dict[str, Any]) -> Any:
        try:
            import anthropic
        except ImportError:   # only an injected client gets here; anything it raises is an unavailable provider
            try:
                return client.beta.messages.create(**kwargs)
            except Exception as e:  # noqa: BLE001
                raise AIProviderError("UNAVAILABLE", type(e).__name__) from e
        try:
            return client.beta.messages.create(**kwargs)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise AIProviderError("AUTH", type(e).__name__) from e
        except anthropic.RateLimitError as e:
            raise AIProviderError("RATE_LIMIT", type(e).__name__) from e
        except anthropic.APITimeoutError as e:
            raise AIProviderError("TIMEOUT", type(e).__name__) from e
        except anthropic.APIConnectionError as e:
            raise AIProviderError("UNAVAILABLE", type(e).__name__) from e
        except anthropic.APIStatusError as e:
            raise AIProviderError("UNAVAILABLE", f"{type(e).__name__} {e.status_code}") from e
