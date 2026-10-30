"""Brief → requirements (requirements_extraction), ADR-061: the rule-based reader (English and Japanese), the AI path through a
fake provider, the Claude provider and the host provider, refusal of unknown keys and invalid values, explicit `--set`
precedence, the decision that records the reading, and rejecting that reading in a revision."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_ai_provider import FakeAIProvider  # noqa: E402
from pipeline_harness import fake_media, pipeline_service  # noqa: E402

from video_agent.agent import brief as brief_task  # noqa: E402
from video_agent.mcp_server import TOOLS, tool_argv  # noqa: E402
from video_agent.project import save_ir  # noqa: E402
from video_agent.providers import AIRequest  # noqa: E402
from video_agent.providers.claude import AnthropicProvider  # noqa: E402
from video_agent.providers.host import HostProvider  # noqa: E402
from video_agent.service import BRIEF_REQUIREMENT_KEYS, Service, _check_requirement_keys  # noqa: E402

DONE_WHEN = "60 s vertical, captions, -14 LUFS"


def read(text, **explicit):
    """The rule-based reading (no provider), as the service records it."""
    return Service(workspace=tempfile.mkdtemp()).extract_brief(text, explicit)


def keys(rec):
    return {a["key"]: a["value"] for a in rec["accepted"]}


def plan(brief, provider=None, **explicit):
    tmp = tempfile.mkdtemp()
    svc = pipeline_service(tmp)
    if provider is not None:
        svc.provider = provider
    ir = svc.plan([fake_media(tmp)], "youtube", user_requirements=explicit or None, params={"language": "ja"}, brief=brief)
    return svc, ir


def decision(ir, subject):
    return next(d for d in ir.doc["decisions"] if d["subject"] == subject)


class RuleBasedReaderTests(unittest.TestCase):
    def test_done_when_example_yields_valid_keys(self):
        rec = read(DONE_WHEN)
        self.assertEqual(rec["method"], "rules")
        self.assertIn("no AI provider", rec["fallback"])
        self.assertEqual(keys(rec), {"edit.fit": "9:16", "subtitle": True, "audio.loudness.target_lufs": -14.0, "audio.normalize": True})
        _check_requirement_keys(keys(rec))   # exactly what `--set` accepts
        self.assertTrue(set(keys(rec)) <= BRIEF_REQUIREMENT_KEYS)
        self.assertEqual([(u["phrase"], u["seconds"]) for u in rec["unmapped"]], [("60 s", 60.0)])
        self.assertIn("no requirement key sets a target length", rec["unmapped"][0]["reason"])
        self.assertEqual(rec["refused"], [])

    def test_english_phrases(self):
        cases = {
            "portrait 9:16 with burned-in subtitles as vtt": {"edit.fit": "9:16", "subtitle": True, "subtitle.burn_in": True, "subtitle.format": "vtt"},
            "landscape, normalize to -16 LUFS, -1.5 dBTP": {"edit.fit": "16:9", "audio.loudness.target_lufs": -16.0, "audio.normalize": True, "audio.loudness.true_peak": -1.5},
            "square 1:1, trim the leading silence, frame accurate": {"edit.fit": "1:1", "edit.trim_leading_silence": True, "edit.precision": "frame"},
            "a 4:5 cut for the feed, no captions": {"edit.fit": "4:5"},
            "LUFS -23 please": {"audio.loudness.target_lufs": -23.0, "audio.normalize": True},
        }
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertEqual(keys(read(text)), want)

    def test_japanese_phrases(self):
        cases = {
            "縦型で字幕つき、60秒、-14LUFS": ({"edit.fit": "9:16", "subtitle": True, "audio.loudness.target_lufs": -14.0, "audio.normalize": True}, [60.0]),
            "１分３０秒の横長動画。字幕は焼き付けで": ({"edit.fit": "16:9", "subtitle": True, "subtitle.burn_in": True}, [90.0]),
            "正方形、キャプション付き、ラウドネス −14": ({"edit.fit": "1:1", "subtitle": True, "audio.loudness.target_lufs": -14.0, "audio.normalize": True}, []),
            "縦長、字幕なし、2分": ({"edit.fit": "9:16"}, [120.0]),
            "縦横比はそのまま、先頭の無音を詰めて": ({"edit.trim_leading_silence": True}, []),
        }
        for text, (want, durations) in cases.items():
            with self.subTest(text=text):
                rec = read(text)
                self.assertEqual(keys(rec), want)
                self.assertEqual([u["seconds"] for u in rec["unmapped"] if "seconds" in u], durations)

    def test_ambiguity_is_reported_not_guessed(self):
        rec = read("vertical and landscape versions, 14 LUFS")
        self.assertNotIn("edit.fit", keys(rec))
        self.assertNotIn("audio.loudness.target_lufs", keys(rec))
        reasons = " ".join(u["reason"] for u in rec["unmapped"])
        self.assertIn("conflicting aspect ratios", reasons)
        self.assertIn("without a minus sign", reasons)
        self.assertEqual(keys(read("make it nice")), {})


class ValidationTests(unittest.TestCase):
    def check(self, cands, **explicit):
        return brief_task.validate_candidates(cands, BRIEF_REQUIREMENT_KEYS, explicit, _check_requirement_keys)

    def test_allowed_keys_come_from_the_set_vocabulary(self):
        for k in ("edit.fit", "subtitle", "audio.loudness.target_lufs", "delivery.timeline", "edit.takes", "qc"):
            self.assertIn(k, BRIEF_REQUIREMENT_KEYS)
        for k in ("video.fit.approval", "subtitle.burn_in.approval", "qc.warn.promotion", "edit.overlay", "color.lut", "qc.reference_video",
                  "job.dubbing", "provider.silence_cleanup", "delivery.duration", "brief"):
            self.assertNotIn(k, BRIEF_REQUIREMENT_KEYS)

    def test_unknown_keys_are_refused_with_the_reason(self):
        acc, ref, _ = self.check([{"key": "delivery.duration", "value": 60, "phrase": "60 s"}, {"key": "video.fit.approval", "value": "AUTO", "phrase": "vertical"},
                                  {"key": "edit.fit", "value": "9:16", "phrase": "vertical"}])
        self.assertEqual(list(acc), ["edit.fit"])
        self.assertEqual(sorted(r["key"] for r in ref), ["delivery.duration", "video.fit.approval"])
        self.assertTrue(all("unknown requirement key" in r["reason"] for r in ref))

    def test_invalid_values_are_refused_like_set(self):
        acc, ref, _ = self.check([{"key": "edit.fit", "value": "tall", "phrase": "vertical"}, {"key": "audio.loudness.target_lufs", "value": 14, "phrase": "14 LUFS"},
                                  {"key": "subtitle.format", "value": "ass", "phrase": "ass"}, {"key": "audio.normalize", "value": "yes please", "phrase": "loud"}])
        self.assertEqual(acc, {})
        why = {r["key"]: r["reason"] for r in ref}
        self.assertIn("aspect ratio", why["edit.fit"])
        self.assertIn("-70 to -5", why["audio.loudness.target_lufs"])
        self.assertIn("subtitle", why["subtitle.format"])   # a refinement without its switch, or an unknown format: refused either way
        self.assertIn("true / false / auto", why["audio.normalize"])

    def test_combination_with_explicit_set_is_checked(self):
        acc, ref, over = self.check([{"key": "edit.fit", "value": "9:16", "phrase": "vertical"}, {"key": "subtitle.burn_in", "value": True, "phrase": "burned-in"}],
                                    **{"edit.fit": "16:9", "subtitle": False})
        self.assertEqual(acc, {})
        self.assertEqual([o["key"] for o in over], ["edit.fit"])
        self.assertIn("subtitle=false", ref[0]["reason"])

    def test_explicit_values_the_set_gate_refuses(self):
        with self.assertRaises(ValueError):
            Service(workspace=tempfile.mkdtemp()).plan(["x.mp4"], user_requirements={"audio.loudness.target_lufs": "loud"})


class AiPathTests(unittest.TestCase):
    ANSWER = [{"key": "edit.fit", "value": "9:16", "phrase": "vertical", "confidence": 0.9}, {"key": "subtitle", "value": "true", "phrase": "captions"},
              {"key": "audio.loudness.target_lufs", "value": "-14", "phrase": "-14 LUFS"}, {"key": "delivery.duration", "value": "60", "phrase": "60 s"},
              {"key": "edit.speed", "value": "2", "phrase": "twice as fast"}, {"key": "audio.normalize", "value": "maybe", "phrase": "LUFS"}]

    def test_fake_provider_keys_are_validated_and_recorded(self):
        prov = FakeAIProvider(brief_requirements=self.ANSWER, brief_unmapped=[{"phrase": "60 s", "reason": "no duration key"}])
        svc, ir = plan(DONE_WHEN, prov)
        rec = ir.doc["request"]["args"]["brief"]
        self.assertEqual((rec["method"], rec["provider"], rec["model"]), ("ai", "fake", "fake-model-1"))
        self.assertEqual(keys(rec), {"edit.fit": "9:16", "subtitle": True, "audio.loudness.target_lufs": -14})
        self.assertEqual(sorted(r["key"] for r in rec["refused"]), ["audio.normalize", "delivery.duration"], "unknown key and invalid value")
        self.assertIn("quotes no phrase of the brief", rec["warnings"][0], "a phrase the brief does not contain is dropped")
        self.assertEqual(rec["unmapped"], [{"phrase": "60 s", "reason": "no duration key"}])
        self.assertEqual(prov.requests[0].task_type, "requirements_extraction")
        self.assertEqual(set(prov.requests[0].context["allowed_keys"]), set(BRIEF_REQUIREMENT_KEYS))
        req = {r["key"]: r for r in ir.doc["requirements"]}
        self.assertEqual((req["edit.fit"]["provenance"], req["edit.fit"]["source"]), ("AI_GENERATED", "brief:ai:fake/fake-model-1"))
        self.assertEqual(req["brief"]["value"], DONE_WHEN)
        d = decision(ir, "requirements.brief")
        self.assertEqual((d["type"], d["approval"], d["provenance"]), ("REVIEW", "CONFIRM", "AI_GENERATED"))
        self.assertIn(req["brief"]["id"], d["evidence"], "the brief is the evidence of its reading")
        self.assertIn(req["edit.fit"]["id"], d["evidence"])
        loud = decision(ir, "audio.loudness")
        self.assertEqual(loud["approval"], "CONFIRM", "an AI-read requirement never yields AUTO")
        self.assertTrue(any("an AI read from the brief" in n for n in loud["basis"]["approval"]["notes"]))
        self.assertEqual(decision(ir, "video.fit")["approval"], "CONFIRM")
        self.assertEqual([c["task_type"] for c in ir.doc["provenance"]["ai_calls"]], ["requirements_extraction", "production_recommendation"])
        self.assertEqual(ir.doc["plan"]["status"], "REVIEW")
        self.assertTrue(svc.validate(ir).ok, svc.validate(ir).errors)

    def test_rules_reading_is_the_users_word(self):
        svc, ir = plan(DONE_WHEN)
        d = decision(ir, "requirements.brief")
        self.assertEqual((d["type"], d["approval"], d["provenance"]), ("KEEP", "AUTO", "USER"))
        self.assertEqual({r["key"]: r["source"] for r in ir.doc["requirements"] if r["source"].startswith("brief")},
                         {"edit.fit": "brief:rules", "subtitle": "brief:rules", "audio.loudness.target_lufs": "brief:rules", "audio.normalize": "brief:rules"})
        self.assertTrue(any(s["skill"] == "subtitle_generation" for s in ir.doc["plan"]["steps"]), "captions → the subtitle steps (transcript measured)")
        self.assertTrue(any(op.get("type") == "video.fit" and op.get("aspect") == "9:16" for op in ir.doc["video"]["operations"]))
        self.assertTrue(svc.validate(ir).ok, svc.validate(ir).errors)

    def test_explicit_set_wins_over_the_brief(self):
        prov = FakeAIProvider(brief_requirements=self.ANSWER[:3])
        svc, ir = plan(DONE_WHEN, prov, **{"edit.fit": "16:9"})
        rec = ir.doc["request"]["args"]["brief"]
        self.assertEqual(rec["overridden"], [{"key": "edit.fit", "value": "9:16", "explicit": "16:9", "phrase": "vertical"}])
        fit = [r for r in ir.doc["requirements"] if r["key"] == "edit.fit"]
        self.assertEqual([(r["value"], r["provenance"], r["source"]) for r in fit], [("16:9", "USER", "cli")])
        self.assertEqual(decision(ir, "video.fit")["approval"], "AUTO", "the explicit request still waives CONFIRM")
        self.assertTrue(any(op.get("aspect") == "16:9" for op in ir.doc["video"]["operations"] if op.get("type") == "video.fit"))

    def test_provider_failure_falls_back_to_the_rules(self):
        svc, ir = plan(DONE_WHEN, FakeAIProvider(fail="RATE_LIMIT"))
        rec = ir.doc["request"]["args"]["brief"]
        self.assertEqual(rec["method"], "rules")
        self.assertIn("RATE_LIMIT", rec["fallback"])
        self.assertEqual(keys(rec)["edit.fit"], "9:16")
        self.assertEqual(ir.doc["provenance"]["ai_calls"][0]["error"]["kind"], "RATE_LIMIT")

    def test_rejected_reading_is_dropped_in_the_revision(self):
        svc, ir = plan(DONE_WHEN, FakeAIProvider(brief_requirements=self.ANSWER[:3]))
        path = str(Path(svc.workspace) / "plans" / "b.project.json")
        save_ir(ir, path)
        svc.reject(ir, path, [decision(ir, "requirements.brief")["id"]], reason="not what I meant")
        out = svc.revise(ir, path)
        self.assertTrue(out["created"])
        self.assertTrue(ir.doc["request"]["args"]["brief"]["rejected"])
        self.assertFalse([r for r in ir.doc["requirements"] if r["source"].startswith("brief:")])
        self.assertFalse([op for op in ir.doc["video"]["operations"] if op.get("type") == "video.fit"])
        self.assertTrue(svc.validate(ir).ok, svc.validate(ir).errors)


class ProviderTests(unittest.TestCase):
    def test_claude_provider_constrains_keys_to_the_allowed_set(self):
        sent = []

        def respond(kw):
            sent.append(kw)
            body = {"requirements": [{"key": "edit.fit", "value": "9:16", "phrase": "vertical", "confidence": 0.9}], "unmapped": [{"phrase": "60 s", "reason": "no key"}],
                    "confidence": 0.9, "reasoning": "vertical means 9:16"}
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(body))], stop_reason="end_turn", model="claude-opus-5-5", _request_id="r",
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0))
        client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: respond(kw))))
        req = brief_task.build_request(DONE_WHEN, BRIEF_REQUIREMENT_KEYS)
        resp = AnthropicProvider(client=client, env={}).complete(req)
        schema = sent[0]["output_config"]["format"]["schema"]
        self.assertEqual(set(schema["properties"]["requirements"]["items"]["properties"]["key"]["enum"]), set(BRIEF_REQUIREMENT_KEYS))
        self.assertIn("brief", sent[0]["system"][0]["text"])
        self.assertEqual(resp.result["requirements"][0]["key"], "edit.fit")
        self.assertEqual(resp.result["unmapped"], [{"phrase": "60 s", "reason": "no key"}])

    def test_host_provider_brief_then_answer(self):
        svc = Service(workspace=tempfile.mkdtemp(), provider=HostProvider(model="claude-code"))
        rec = svc.extract_brief(DONE_WHEN)
        self.assertEqual(rec["method"], "rules")
        self.assertIn("has not answered", rec["fallback"])
        self.assertEqual(rec["host_request"]["task_type"], "requirements_extraction")
        self.assertIn("allowed_keys", rec["host_request"]["context"])
        answer = {"requirements": [{"key": "edit.fit", "value": "9:16", "phrase": "vertical"}, {"key": "edit.vertical", "value": "true", "phrase": "vertical"}],
                  "unmapped": [{"phrase": "60 s", "reason": "no duration key"}]}
        rec = svc.extract_brief(DONE_WHEN, answer=answer, ai_model="claude-code")
        self.assertEqual((rec["method"], rec["provider"], rec["model"]), ("ai", "host", "claude-code"))
        self.assertEqual(keys(rec), {"edit.fit": "9:16"})
        self.assertEqual([r["key"] for r in rec["refused"]], ["edit.vertical"])

    def test_mcp_tools(self):
        self.assertIn("brief", [t["name"] for t in TOOLS])
        tmp = tempfile.mkdtemp()
        self.assertEqual(tool_argv("brief", {"text": DONE_WHEN}, tmp), ["brief", DONE_WHEN, "--ai-brief"])
        argv = tool_argv("plan", {"inputs": ["a.mp4"], "brief": DONE_WHEN, "brief_requirements": {"requirements": []}, "ai_model": "claude-code"}, tmp)
        self.assertEqual(argv[argv.index("--brief") + 1], DONE_WHEN)
        self.assertEqual(json.loads(Path(argv[argv.index("--brief-requirements") + 1]).read_text(encoding="utf-8")), {"requirements": []})

    def test_ai_request_shape(self):
        r = brief_task.build_request("x" * 5000, ["edit.fit"], {"subtitle": True})
        self.assertIsInstance(r, AIRequest)
        self.assertEqual(len(r.inputs["brief"]), brief_task.MAX_BRIEF_CHARS)
        self.assertEqual(r.context, {"allowed_keys": ["edit.fit"], "explicit_keys": ["subtitle"]})


if __name__ == "__main__":
    unittest.main()
