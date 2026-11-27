"""Progress and error classes (ADR-140, roadmap step 24): every render reports which step runs, how far it is (weighted by the speed book's
estimates), the ETA and the chunk, in <job>/progress.json, `status --json` and MCP notifications/progress; every failure carries one class
(input / environment / engine / policy / transient), the tool's own code and whether a retry helps. Fake engine; no media is decoded."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter  # noqa: E402
from pipeline_harness import clear_modes, fake_media, pipeline_service  # noqa: E402
from video_agent.execution.progress import Progress  # noqa: E402
from video_agent.execution.recovery import ERROR_CLASS_OF, ERROR_CLASSES, RECOVERY_TABLE, classify_error, describe_failure  # noqa: E402
from video_agent.mcp_server import Server  # noqa: E402
from video_agent.models import Operation, ToolResult  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402


def result(exit_code=1, stderr="", data=None, tool="ffmpeg-skill/export"):
    return ToolResult(op_id="op_x", tool=tool, ok=False, exit_code=exit_code, output=None, data=data or {}, commands=[], stderr_tail=stderr, seconds=0.1, attempt=1, dry_run=False)


# one failure of each class, as the tools report them
FIXTURES = {
    "input": result(stderr="error: input not found: /media/talk.mp4"),
    "environment": result(exit_code=127, stderr="ffmpeg was not found on PATH"),
    "engine": result(stderr="error: command failed (1): ffmpeg\nConversion failed!"),
    "policy": result(tool="publish/upload", data={"error": {"code": "TERMS_NOT_ACCEPTED", "recovery_class": "ACTION_REQUIRED", "retryable": False}}),
    "transient": result(exit_code=124, stderr="timeout after 60s"),
}


class ClassTests(unittest.TestCase):
    def test_one_failure_of_each_class(self):
        for cls, r in FIXTURES.items():
            got = describe_failure(r, classify_error(r))
            self.assertEqual(got["error_class"], cls, (cls, got))
        self.assertEqual(describe_failure(FIXTURES["policy"], classify_error(FIXTURES["policy"]))["code"], "TERMS_NOT_ACCEPTED", "the tool's own code is kept")
        self.assertTrue(describe_failure(FIXTURES["transient"], "TIMEOUT")["retry_helps"])
        self.assertFalse(describe_failure(FIXTURES["input"], "INPUT_MISSING")["retry_helps"])
        self.assertEqual(sorted({describe_failure(r, classify_error(r))["error_class"] for r in FIXTURES.values()}), sorted(ERROR_CLASSES))

    def test_every_recovery_class_has_an_error_class(self):
        self.assertEqual(sorted(set(RECOVERY_TABLE) - set(ERROR_CLASS_OF)), [])
        self.assertTrue(set(ERROR_CLASS_OF.values()) <= set(ERROR_CLASSES))


class ProgressTests(unittest.TestCase):
    def test_weighted_monotonic_with_eta_and_chunk(self):
        ops = [Operation(tool="ffmpeg-skill/cut", args={}, inputs=[], outputs=["a"], id=f"op{n}") for n in range(3)]
        ops[1].chunk = 2
        t = [0.0]
        snaps = []
        p = Progress(ops, {"op0": 10.0, "op1": 30.0, "op2": 60.0}, snaps.append, job_id="job_1", clock=lambda: t[0])
        p(ops[0], 0, "start"); t[0] = 20.0; p(ops[0], 0, "ran")   # took twice the estimate
        p(ops[1], 1, "reused")
        p(ops[2], 2, "start"); t[0] = 50.0
        self.assertEqual([s["percent"] for s in snaps], [0.0, 10.0, 40.0, 40.0])
        self.assertEqual(snaps[1]["eta_seconds"], 180.0, "90 s still to run at the rate seen so far (2x)")
        self.assertEqual(snaps[2]["chunk"], 2)
        self.assertEqual(snaps[3]["step"]["index"], 3)
        final = p.final("COMPLETED")
        self.assertEqual((final["percent"], final["eta_seconds"], final["state"]), (100.0, 0.0, "completed"))


class RenderTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.src = fake_media(self.tmp, "talk.mp4", duration=40.0, silences=[])

    def render(self, engine=None, chunk_seconds=None):
        svc = pipeline_service(self.tmp, engine=engine)
        snaps = []
        svc.on_progress.append(snaps.append)
        ir = svc.plan([self.src], "youtube", user_requirements={}, chunk_seconds=chunk_seconds)
        path = str(Path(svc.workspace) / "plans" / "p.project.json")
        save_ir(ir, path)
        return svc, svc.render(load_ir(path), path, approve=["all"]), snaps

    def test_progress_is_monotonic_and_reaches_100(self):
        svc, out, snaps = self.render()
        self.assertEqual(out["status"], "COMPLETED")
        pct = [s["percent"] for s in snaps]
        self.assertEqual(pct, sorted(pct), "never backwards")
        self.assertEqual(pct[-1], 100.0)
        self.assertEqual([s["step"]["index"] for s in snaps if s["state"] == "start"], list(range(1, len(out["execution"]["results"]) + 1)))
        st = svc.status("last")
        self.assertEqual((st["state"], st["progress"]["percent"], st["progress"]["schema"]), ("COMPLETED", 100.0, "video-agent/progress@1"))
        self.assertEqual(out["progress"]["percent"], 100.0)

    def test_a_chunked_render_names_the_chunk(self):
        _, out, snaps = self.render(chunk_seconds=10)
        self.assertEqual(out["status"], "COMPLETED")
        self.assertEqual(sorted({s["chunk"] for s in snaps if "chunk" in s}), [1, 2, 3, 4])

    def test_a_failed_render_carries_its_error_class(self):
        svc, out, _ = self.render(engine=FakeAdapter(fail_tools={"ffmpeg-skill/export": 5}))
        self.assertEqual(out["status"], "FAILED")
        e = out["error"]
        self.assertEqual((e["error_class"], e["recovery_class"], e["tool"]), ("engine", "ENCODER_FAILED", "ffmpeg-skill/export"))
        st = svc.status("last")
        self.assertEqual(st["error"]["error_class"], "engine")
        self.assertLess(st["progress"]["percent"], 100.0)

    def test_a_blocked_plan_is_a_policy_error_in_status(self):
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self.src], "youtube", user_requirements={"delivery.chapters": True}, chunk_seconds=0)   # no AI: KEEP, nothing pending
        ir.doc["decisions"].append(dict(ir.doc["decisions"][0], id="dec_block", status="BLOCKED", approval="BLOCK", type="BLOCK", subject="policy.test"))
        path = str(Path(svc.workspace) / "plans" / "b.project.json")
        save_ir(ir, path)
        out = svc.render(load_ir(path), path, approve=["all"])
        self.assertIn(out["status"], ("BLOCKED", "FAILED"))
        st = svc.status("last")
        self.assertIn(st["error"]["error_class"], ("policy", "input"))


class McpProgressTests(unittest.TestCase):
    def test_progress_lines_become_notifications(self):
        srv = Server(workspace=tempfile.mkdtemp())
        buf = io.StringIO()
        srv._out = buf
        lines = [json.dumps({"video_agent_progress": {"percent": p, "total": 3, "step": {"index": i, "tool": "ffmpeg-skill/export"}, "eta_seconds": 9.0 - i}})
                 for i, p in ((1, 0.0), (2, 40.0), (3, 30.0), (3, 100.0))]
        code = "import sys\n" + "".join(f"sys.stderr.write({ln!r} + '\\n')\n" for ln in lines) + "sys.stderr.write('a plain line\\n')\nprint('{\"status\": \"COMPLETED\"}')\n"
        cp = srv._run_with_progress([sys.executable, "-c", code], None, "tok-1")
        notes = [json.loads(x) for x in buf.getvalue().splitlines()]
        self.assertEqual([n["method"] for n in notes], ["notifications/progress"] * 4)
        self.assertEqual([n["params"]["progress"] for n in notes], [0.0, 40.0, 40.0, 100.0], "never lower than the last one sent")
        self.assertTrue(all(n["params"]["progressToken"] == "tok-1" and n["params"]["total"] == 100 for n in notes))
        self.assertIn("a plain line", cp.stderr)
        self.assertIn("COMPLETED", cp.stdout)

    def test_status_is_a_read_tool(self):
        from video_agent.mcp_permissions import TOOL_CLASSES
        from video_agent.mcp_server import TOOLS, tool_argv
        self.assertEqual(TOOL_CLASSES["status"], "read")
        self.assertIn("status", [t["name"] for t in TOOLS])
        self.assertEqual(tool_argv("status", {}, "/tmp")[-2:], ["--", "last"])


if __name__ == "__main__":
    unittest.main()
