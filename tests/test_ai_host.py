"""Host provider and MCP server, ADR-050: the chat model the agent runs in (Claude Code, Claude Desktop) is the
reasoning step -- `plan --ai-brief` then `revise --ai-recommendations` -- with the same validation as any provider."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter  # noqa: E402
from test_unit import FakeCaps, fake_media  # noqa: E402

from video_agent.mcp_server import TOOLS, Server, tool_argv  # noqa: E402
from video_agent.project import save_ir  # noqa: E402
from video_agent.providers.host import HostProvider, load_recommendations  # noqa: E402
from video_agent.service import Service  # noqa: E402

SRC = str(Path(__file__).resolve().parents[1] / "src")


def brief_plan(tmp):
    host = HostProvider(model="claude-code")
    svc = Service(workspace=tmp, adapter=FakeAdapter(), caps=FakeCaps(), provider=host)
    ir = svc.plan([fake_media(tmp)], "youtube", request_text="冒頭の無音を詰めて")
    path = str(Path(tmp) / "plans" / "talk.project.json")
    save_ir(ir, path)
    return svc, ir, path, host.brief()


class HostProviderTests(unittest.TestCase):
    def test_brief_carries_inputs_rules_and_allowed_intents(self):
        _, ir, _, brief = brief_plan(tempfile.mkdtemp())
        self.assertIn("silence_cleanup", brief["context"]["allowed_intents"])
        self.assertEqual(brief["context"]["request_text"], "冒頭の無音を詰めて")
        self.assertIn("evidence", brief["rules"])
        self.assertTrue(brief["inputs"]["assets"])
        self.assertFalse([i for i in ir.doc["analysis"]["inferences"] if i["provenance"] == "AI_GENERATED"], "the brief step recommends nothing")

    def test_answer_is_validated_and_recorded_as_a_new_version(self):
        tmp = tempfile.mkdtemp()
        svc, ir, path, brief = brief_plan(tmp)
        asset = brief["inputs"]["assets"][0]["id"]
        cite = (brief["inputs"].get("observations") or brief["inputs"].get("events"))[0]["id"]
        recs = load_recommendations({"recommendations": [
            {"intent": "silence_cleanup", "asset_id": asset, "statement": "leading silence", "confidence": 0.8, "evidence": [cite], "params": {"command": "rm -rf /"}},
            {"intent": "silence_cleanup", "asset_id": asset, "statement": "uncited", "confidence": 0.9, "evidence": []},
            {"intent": "rm_rf", "asset_id": asset, "statement": "x", "confidence": 1, "evidence": [cite]}]})
        before = svc.provider
        out = svc.revise(ir, path, ai_recommendations=recs, ai_model="claude-code")
        self.assertTrue(out["created"], "an AI answer is a new version even when the plan does not change")
        self.assertEqual(len(out["ai"]["accepted"]), 1)
        self.assertEqual(len(out["ai"]["dropped"]), 2)
        ai = [i for i in ir.doc["analysis"]["inferences"] if i["provenance"] == "AI_GENERATED"]
        self.assertEqual([(i["data"]["provider"], i["data"]["model"]) for i in ai], [("host", "claude-code")])
        self.assertNotIn("command", ai[0]["data"]["params"])
        self.assertIs(svc.provider, before, "the service's own provider is restored after the answer")
        self.assertTrue(svc.validate(ir).ok)

    def test_load_recommendations_shapes(self):
        self.assertEqual(load_recommendations([{"intent": "a"}, 3]), [{"intent": "a"}])
        self.assertEqual(load_recommendations({"recommendations": []}), [])
        with self.assertRaises(ValueError):
            load_recommendations({"recs": []})


class McpServerTests(unittest.TestCase):
    def test_tool_argv(self):
        tmp = tempfile.mkdtemp()
        self.assertEqual(tool_argv("plan", {"inputs": ["a.mp4"], "profile": "youtube", "request": "詰めて", "set": {"audio.loudness.target_lufs": -16}}, tmp),
                         ["plan", "a.mp4", "--profile", "youtube", "--request", "詰めて", "--set", "audio.loudness.target_lufs=-16", "--ai-brief"])
        self.assertNotIn("--ai-brief", tool_argv("plan", {"inputs": "a.mp4", "ai_brief": False}, tmp))
        argv = tool_argv("revise", {"project": "p.json", "ai_recommendations": {"recommendations": []}, "ai_model": "claude-desktop"}, tmp)
        f = argv[argv.index("--ai-recommendations") + 1]
        self.assertEqual(json.loads(Path(f).read_text(encoding="utf-8")), {"recommendations": []})
        self.assertEqual(tool_argv("render", {"project": "p.json", "dry_run": True}, tmp), ["render", "p.json", "--dry-run"])
        with self.assertRaises(ValueError):
            tool_argv("approve", {"project": "p.json"}, tmp)
        self.assertNotIn("approve", json.dumps(TOOLS[[t["name"] for t in TOOLS].index("render")]["inputSchema"]), "render cannot approve inline")

    def test_every_listed_tool_maps_to_a_command(self):
        tmp = tempfile.mkdtemp()
        full = {"inputs": ["a"], "project": "p", "decision": "all", "reason": "r", "output": "o"}
        for t in TOOLS:
            tool_argv(t["name"], full, tmp)

    def test_stdio_session(self):
        lines = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                 {"jsonrpc": "2.0", "method": "notifications/initialized"},
                 {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                 {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "approve", "arguments": {"project": "p"}}},
                 {"jsonrpc": "2.0", "id": 4, "method": "nope"}]
        out = io.StringIO()
        Server().serve(io.StringIO("\n".join(json.dumps(m) for m in lines) + "\nnot json\n"), out)
        replies = [json.loads(x) for x in out.getvalue().splitlines()]
        self.assertEqual([r.get("id") for r in replies], [1, 2, 3, 4, None])
        self.assertEqual(replies[0]["result"]["serverInfo"]["name"], "video-agent")
        self.assertIn("plan", [t["name"] for t in replies[1]["result"]["tools"]])
        self.assertTrue(replies[2]["result"]["isError"])
        self.assertEqual((replies[3]["error"]["code"], replies[4]["error"]["code"]), (-32601, -32700))

    def test_tool_call_runs_the_cli(self):
        env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""))
        msgs = "\n".join(json.dumps(m) for m in [{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "explain", "arguments": {"project": "/nonexistent/p.json"}}}])
        p = subprocess.run([sys.executable, "-m", "video_agent.mcp_server", "--workspace", tempfile.mkdtemp()], input=msgs + "\n",
                           capture_output=True, text=True, env=env, timeout=120)
        reply = json.loads(p.stdout.splitlines()[0])
        self.assertTrue(reply["result"]["isError"], "a missing project is reported as a tool error, not a protocol error")
        self.assertIn("exit code", reply["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
