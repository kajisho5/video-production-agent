"""Anthropic (Claude) provider, ADR-049: request shape, response validation, failure mapping and the full
plan pipeline -- all with an injected fake client, so no network and no credentials are needed."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter  # noqa: E402
from test_unit import FakeCaps, fake_media  # noqa: E402

from video_agent.providers import AIProviderError, AIRequest, get_provider  # noqa: E402
from video_agent.providers.claude import DEFAULT_MODEL, FALLBACK_BETA, RESPONSE_SCHEMA, AnthropicProvider  # noqa: E402
from video_agent.service import Service  # noqa: E402

try:
    import anthropic
except ImportError:  # the `ai` extra is optional
    anthropic = None


def message(payload, stop_reason="end_turn", model=DEFAULT_MODEL):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)], stop_reason=stop_reason,
                           stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None, model=model, _request_id="req_test",
                           usage=SimpleNamespace(input_tokens=1200, output_tokens=300, cache_read_input_tokens=0))


class FakeClient:
    """Records every create() call; answers with `respond(kwargs)` or raises `error`."""

    def __init__(self, respond=None, error=None):
        self.calls = []
        self.respond, self.error = respond, error
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.respond(kwargs)


def cite_first_observation(intent):
    """A well-formed answer citing ids from the request the provider actually sent."""
    def respond(kwargs):
        body = json.loads(kwargs["messages"][0]["content"])
        asset = body["inputs"]["assets"][0]["id"]
        obs = [o["id"] for o in body["inputs"]["observations"] if o.get("asset_id") == asset][:2]
        return message({"recommendations": [{"intent": intent, "asset_id": asset, "statement": f"{intent} fits the measured silence", "confidence": 0.9,
                                             "evidence": obs, "params": [{"key": "keep", "value": "x"}, {"key": "command", "value": "rm -rf /"}]}],
                        "confidence": 0.85, "evidence": obs, "reasoning": "Long silent lead-in measured at the head."})
    return respond


def request():
    return AIRequest(task_type="production_recommendation", inputs={"assets": [{"id": "a1"}], "observations": [{"id": "o1", "asset_id": "a1"}]},
                     schema={"recommendations": []}, context={"allowed_intents": ["silence_cleanup"], "request_text": "冒頭の無音を詰めて"})


class AnthropicProviderTests(unittest.TestCase):
    def test_get_provider_selects_it_and_null_stays_default(self):
        import os
        old = os.environ.get("VIDEO_AGENT_AI_PROVIDER")
        try:
            os.environ["VIDEO_AGENT_AI_PROVIDER"] = "anthropic"
            self.assertIsInstance(get_provider(), AnthropicProvider)
            os.environ["VIDEO_AGENT_AI_PROVIDER"] = "null"
            self.assertEqual(get_provider().name, "null")
        finally:
            if old is None:
                os.environ.pop("VIDEO_AGENT_AI_PROVIDER", None)
            else:
                os.environ["VIDEO_AGENT_AI_PROVIDER"] = old

    def test_available_needs_credentials_and_never_reads_them(self):
        home = tempfile.mkdtemp()
        p = AnthropicProvider(env={"HOME": home})
        self.assertFalse(p.available(), "no key, token or profile: not available")
        self.assertEqual(AnthropicProvider(env={"HOME": home, "ANTHROPIC_API_KEY": "sk-SECRET"}).available(), anthropic is not None)
        self.assertTrue(AnthropicProvider(client=FakeClient()).available())
        self.assertNotIn("sk-", json.dumps(AnthropicProvider(env={"ANTHROPIC_API_KEY": "sk-SECRET"}).describe()))
        self.assertEqual(p.describe()["model"], DEFAULT_MODEL)
        with self.assertRaises(ValueError):
            AnthropicProvider(env={"VIDEO_AGENT_AI_EFFORT": "turbo"})

    def test_request_shape(self):
        client = FakeClient(cite_first_observation("silence_cleanup"))
        AnthropicProvider(client=client, env={}).complete(request(), timeout=30)
        kw = client.calls[0]
        self.assertEqual(kw["model"], "claude-opus-5-5")
        self.assertEqual(kw["output_config"]["effort"], "medium")
        self.assertEqual(kw["output_config"]["format"], {"type": "json_schema", "schema": RESPONSE_SCHEMA})
        self.assertEqual((kw["betas"], kw["fallbacks"]), ([FALLBACK_BETA], "default"))
        self.assertEqual(kw["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("thinking", kw, "Claude Opus 5.5 always thinks adaptively; effort is the control")
        self.assertEqual(kw["timeout"], 30)
        body = json.loads(kw["messages"][0]["content"])
        self.assertEqual(body["context"]["request_text"], "冒頭の無音を詰めて")

    def test_response_is_validated_and_params_folded(self):
        client = FakeClient(cite_first_observation("silence_cleanup"))
        resp = AnthropicProvider(client=client, env={}).complete(request())
        rec = resp.result["recommendations"][0]
        self.assertEqual(rec["params"], {"keep": "x", "command": "rm -rf /"}, "folded here; ai_reasoning drops the forbidden keys")
        self.assertEqual((resp.provider, resp.model, resp.confidence), ("anthropic", DEFAULT_MODEL, 0.85))
        self.assertEqual(resp.usage.to_dict()["input_tokens"], 1200)
        self.assertEqual(resp.metadata["request_id"], "req_test")

    def test_failures_map_to_provider_errors(self):
        for answer, kind in ((message("{}", stop_reason="refusal"), "UNAVAILABLE"), (message("{\"recommendations\": [", stop_reason="max_tokens"), "MALFORMED"),
                             (message("not json"), "MALFORMED"), (message({"confidence": 1}), "MALFORMED")):
            with self.subTest(kind=kind, stop=answer.stop_reason):
                with self.assertRaises(AIProviderError) as cm:
                    AnthropicProvider(client=FakeClient(lambda kw, a=answer: a), env={}).complete(request())
                self.assertEqual(cm.exception.kind, kind)
        with self.assertRaises(AIProviderError):
            AnthropicProvider(client=FakeClient(), env={}).complete(AIRequest(task_type="requirements_extraction", inputs={}))

    @unittest.skipIf(anthropic is None, "the `ai` extra (anthropic SDK) is not installed")
    def test_sdk_errors_map_to_failure_kinds(self):
        def err(cls):
            e = cls.__new__(cls)
            e.status_code = 500
            return e
        for cls, kind in ((anthropic.AuthenticationError, "AUTH"), (anthropic.PermissionDeniedError, "AUTH"), (anthropic.RateLimitError, "RATE_LIMIT"),
                          (anthropic.APITimeoutError, "TIMEOUT"), (anthropic.APIConnectionError, "UNAVAILABLE"), (anthropic.InternalServerError, "UNAVAILABLE")):
            with self.subTest(cls=cls.__name__):
                with self.assertRaises(AIProviderError) as cm:
                    AnthropicProvider(client=FakeClient(error=err(cls)), env={}).complete(request())
                self.assertEqual(cm.exception.kind, kind)

    def test_full_plan_pipeline_with_claude(self):
        tmp = tempfile.mkdtemp()
        src = fake_media(tmp)
        client = FakeClient(cite_first_observation("silence_cleanup"))
        svc = Service(workspace=tmp, adapter=FakeAdapter(), caps=FakeCaps(), provider=AnthropicProvider(client=client, env={}))
        ir = svc.plan([src], "youtube", request_text="冒頭の無音を詰めて")
        self.assertEqual(len(client.calls), 1)
        body = json.loads(client.calls[0]["messages"][0]["content"])
        self.assertEqual(body["context"]["request_text"], "冒頭の無音を詰めて")
        self.assertIn("silence_cleanup", body["context"]["allowed_intents"])
        ai = [i for i in ir.doc["analysis"]["inferences"] if i["provenance"] == "AI_GENERATED"]
        self.assertTrue(ai, "Claude's cited recommendation became an AI_GENERATED inference")
        self.assertNotIn("command", ai[0]["data"]["params"], "forbidden params are dropped by the reasoning layer")
        self.assertEqual(ai[0]["data"]["provider"], "anthropic")
        self.assertTrue(svc.validate(ir).ok)
        calls = ir.doc["provenance"]["ai_calls"]
        self.assertEqual(calls[0]["provider"], "anthropic")
        self.assertNotIn("sk-", json.dumps(ir.doc))


if __name__ == "__main__":
    unittest.main(verbosity=2)
