"""ADR-124: MCP permission scopes, the preview + confirm_token step before destructive / publish calls, and the allowed roots.

The CLI child is replaced by a mock of subprocess.run: these tests are about what the server lets through, not what the CLI does."""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from video_agent.mcp_permissions import (DEFAULT_SCOPES, ESCALATES, SCOPES, TOOL_CLASSES, PermissionConfigError, load_permissions,  # noqa: E402
                                         parse_scopes)
from video_agent.mcp_server import TOOLS, Server, listed_tools, main  # noqa: E402

OK = subprocess.CompletedProcess([], 0, "{}", "")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def body(res):
    return json.loads(res["content"][0]["text"])


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ws = self.tmp / "ws"
        (self.ws / "plans").mkdir(parents=True)
        self.media = self.tmp / "talk.mp4"
        self.media.write_bytes(b"not really media")
        self.outside = self.tmp / "outside"
        self.outside.mkdir()
        self.clock = Clock()
        self.env = {}

    def server(self, scopes=None, roots=None, ttl=None, env=None):
        perms = load_permissions(str(self.ws), scopes=scopes, roots=roots, confirm_ttl_s=ttl, env=env if env is not None else self.env)
        return Server(workspace=str(self.ws), permissions=perms, clock=self.clock)

    def call(self, server, name, args):
        with mock.patch("video_agent.mcp_server.subprocess.run", return_value=OK) as run:
            res = server.call_tool(name, args)
        return res, run

    def existing_plan(self, profile="generic"):
        p = self.ws / "plans" / f"talk.{profile}.project.json"
        p.write_text(json.dumps({"project": {"name": "talk"}, "plan": {"version": 3, "status": "APPROVED"},
                                 "decisions": [{"id": "d1", "status": "APPROVED"}, {"id": "d2", "status": "PENDING"}]}), encoding="utf-8")
        return p

    def plan_args(self, **kw):
        return {"inputs": [str(self.media)], **kw}


class ClassificationTests(unittest.TestCase):
    def test_every_tool_has_a_class(self):
        """A tool added to TOOLS (e.g. by another PR) without a permission class fails here, and is refused at call time."""
        names = {t["name"] for t in TOOLS}
        self.assertEqual(sorted(names - set(TOOL_CLASSES)), [], "tools without a permission class: add them to mcp_permissions.TOOL_CLASSES")
        self.assertEqual(sorted(set(TOOL_CLASSES) - names), [], "classes for tools that do not exist")
        for name, cls in TOOL_CLASSES.items():
            self.assertIn(cls, SCOPES, name)
        for name, esc in ESCALATES.items():
            self.assertIn(name, TOOL_CLASSES)
            self.assertTrue(all(SCOPES.index(c) > SCOPES.index(TOOL_CLASSES[name]) for c in esc), name)

    def test_unclassified_tool_is_refused(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        server = Server(workspace=tmp, permissions=load_permissions(tmp, scopes="all", env={}))
        with mock.patch.dict(TOOL_CLASSES, clear=False):
            del TOOL_CLASSES["doctor"]
            with mock.patch("video_agent.mcp_server.subprocess.run", return_value=OK) as run:
                res = server.call_tool("doctor", {})
            listed = {t["name"]: t for t in listed_tools(server.permissions)}
        self.assertTrue(res["isError"])
        self.assertEqual(body(res)["error"], "NO_PERMISSION_CLASS")
        run.assert_not_called()
        self.assertIn("no permission class", listed["doctor"]["description"])


class ConfigTests(_Base):
    def test_default_is_read_and_write(self):
        perms = load_permissions(str(self.ws), env={})
        self.assertEqual(perms.scopes, DEFAULT_SCOPES)
        self.assertEqual(perms.scopes, ("read", "write"))
        self.assertEqual(perms.roots, (os.path.normcase(os.path.realpath(self.ws)),))

    def test_precedence_cli_env_file(self):
        cfg = self.tmp / "mcp.json"
        cfg.write_text(json.dumps({"scopes": ["read"], "allowed_roots": [str(self.outside)], "confirm_ttl_s": 60}), encoding="utf-8")
        p = load_permissions(str(self.ws), config=str(cfg), env={})
        self.assertEqual((p.scopes, p.confirm_ttl_s), (("read",), 60.0))
        self.assertTrue(p.inside(str(self.outside / "x")))
        p = load_permissions(str(self.ws), env={"VIDEO_AGENT_MCP_CONFIG": str(cfg), "VIDEO_AGENT_MCP_SCOPES": "read,write,destructive"})
        self.assertEqual(p.scopes, ("read", "write", "destructive"))
        p = load_permissions(str(self.ws), scopes="all", env={"VIDEO_AGENT_MCP_SCOPES": "read"})
        self.assertEqual(p.scopes, SCOPES)

    def test_bad_settings_are_refused(self):
        with self.assertRaises(PermissionConfigError):
            parse_scopes("read,delete")
        with self.assertRaises(PermissionConfigError):
            load_permissions(str(self.ws), roots=["relative/dir"], env={})
        with self.assertRaises(PermissionConfigError):
            load_permissions(str(self.ws), confirm_ttl_s=0, env={})
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps({"scopes": ["read"], "allow_everything": True}), encoding="utf-8")
        with self.assertRaises(PermissionConfigError):
            load_permissions(str(self.ws), config=str(bad), env={})
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            main(["--workspace", str(self.ws), "--scopes", "everything"])

    def test_tools_list_shows_class_and_disabled(self):
        listed = {t["name"]: t for t in listed_tools(load_permissions(str(self.ws), env={}))}
        self.assertTrue(listed["doctor"]["annotations"]["readOnlyHint"])
        self.assertIn("confirm_token", listed["plan"]["inputSchema"]["properties"])
        self.assertNotIn("confirm_token", listed["doctor"]["inputSchema"]["properties"])
        self.assertTrue(listed["plan"]["annotations"]["destructiveHint"])
        self.assertIn("destructive / publish disabled on this server", listed["plan"]["description"])
        self.assertNotIn("confirm_token", {p for t in TOOLS for p in t["inputSchema"].get("properties", {})}, "TOOLS itself is not changed")


class ScopeTests(_Base):
    def assert_disabled(self, res, run, cls):
        self.assertTrue(res["isError"], res)
        b = body(res)
        self.assertEqual((b["error"], b["class"]), ("SCOPE_DISABLED", cls))
        self.assertIn("--scopes", b["enable_with"])
        self.assertIn(cls, b["enable_with"])
        run.assert_not_called()

    def test_read_scope_blocks_and_allows(self):
        server = self.server(scopes="write")
        for name, args in (("doctor", {}), ("explain", {"project": "p"}), ("render", {"project": "p", "dry_run": True}),
                           ("preferences", {"action": "list"})):
            res, run = self.call(server, name, args)
            self.assert_disabled(res, run, "read")
        res, run = self.call(self.server(scopes="read"), "doctor", {})
        self.assertFalse(res["isError"])
        run.assert_called_once()

    def test_write_scope_blocks(self):
        server = self.server(scopes="read")
        proj = str(self.ws / "plans" / "p.project.json")
        for name, args in (("plan", self.plan_args()), ("approve", {"project": proj, "decision": "all"}),
                           ("reject", {"project": proj, "decision": "d1", "reason": "no"}), ("revise", {"project": proj, "ai_recommendations": []}),
                           ("render", {"project": proj}), ("preferences", {"action": "promote", "project": proj, "feedback": "f1"}),
                           ("check", {"output": str(self.media), "qc": True})):
            res, run = self.call(server, name, args)
            self.assert_disabled(res, run, "write")
        res, run = self.call(self.server(), "approve", {"project": proj, "decision": "all"})
        self.assertFalse(res["isError"], res)
        run.assert_called_once()

    def test_destructive_scope_blocks_overwrite(self):
        self.existing_plan()
        res, run = self.call(self.server(), "plan", self.plan_args())
        self.assert_disabled(res, run, "destructive")
        self.assertEqual(body(res)["would_do"][0]["action"], "overwrite")
        # a different profile is a new file: write, runs
        res, run = self.call(self.server(), "plan", self.plan_args(profile="youtube"))
        self.assertFalse(res["isError"], res)
        run.assert_called_once()

    def test_destructive_scope_blocks_transcript_edit_overwrite(self):
        args = {"input": str(self.media), "transcript": str(self.tmp / "t.json"), "mode": "remove", "segments": "seg_0001"}
        res, run = self.call(self.server(), "transcript_edit", args)
        self.assertFalse(res["isError"], res)
        run.assert_called_once()
        existing = self.ws / "transcript_edits" / "talk.transcript-edit.json"
        existing.parent.mkdir()
        existing.write_text("{}", encoding="utf-8")
        res, run = self.call(self.server(), "transcript_edit", args)
        self.assert_disabled(res, run, "destructive")
        res, run = self.call(self.server(), "transcript_edit", {**args, "output": str(self.outside / "te.json")})
        self.assertEqual(body(res)["error"], "PATH_OUTSIDE_ROOTS")
        run.assert_not_called()

    def test_publish_scope_blocks_offload_outside_roots(self):
        proj = self.ws / "plans" / "card.offload.project.json"
        proj.write_text(json.dumps({"plan": {"steps": [{"id": "step_media_offload", "skill": "media_offload",
                                                        "params": {"source": str(self.tmp / "card"), "destinations": [str(self.outside)]}}]}}), encoding="utf-8")
        res, run = self.call(self.server(scopes="read,write,destructive"), "render", {"project": str(proj)})
        self.assert_disabled(res, run, "publish")
        # inside an allowed root, the copy is an ordinary write
        res, run = self.call(self.server(roots=[str(self.outside)]), "render", {"project": str(proj)})
        self.assertFalse(res["isError"], res)
        run.assert_called_once()

    def test_publish_scope_blocks_paid_ai(self):
        env = {"VIDEO_AGENT_AI_PROVIDER": "anthropic"}
        proj = str(self.ws / "plans" / "p.project.json")
        server = self.server(env=env)
        res, run = self.call(server, "revise", {"project": proj, "feedback": "shorter"})
        self.assert_disabled(res, run, "publish")
        res, run = self.call(server, "plan", self.plan_args(ai_brief=False))
        self.assert_disabled(res, run, "publish")
        # the chat model as the AI (ai_brief, ai_recommendations) costs nothing, and the child cannot reach the paid provider
        with mock.patch.dict(os.environ, env):
            res, run = self.call(server, "revise", {"project": proj, "ai_recommendations": {"recommendations": []}})
        self.assertFalse(res["isError"], res)
        self.assertEqual(run.call_args[1]["env"]["VIDEO_AGENT_AI_PROVIDER"], "null")


class ConfirmationTests(_Base):
    def preview(self, server, name, args):
        res, run = self.call(server, name, args)
        run.assert_not_called()
        self.assertFalse(res["isError"], res)
        b = body(res)
        self.assertEqual(b["status"], "CONFIRMATION_REQUIRED")
        return b

    def test_preview_then_confirm_once(self):
        target = self.existing_plan()
        server = self.server(scopes="read,write,destructive")
        b = self.preview(server, "plan", self.plan_args())
        item = b["preview"][0]
        self.assertEqual((item["action"], item["plan"]["version"], item["plan"]["approved_decisions"]), ("overwrite", 3, 1))
        self.assertEqual(item["resolved"], os.path.normcase(os.path.realpath(target)))
        res, run = self.call(server, "plan", self.plan_args(confirm_token=b["confirm_token"]))
        self.assertFalse(res["isError"], res)
        run.assert_called_once()
        self.assertNotIn("confirm_token", " ".join(run.call_args[0][0]), "the token never reaches the CLI")
        res, run = self.call(server, "plan", self.plan_args(confirm_token=b["confirm_token"]))
        self.assertEqual(body(res)["error"], "CONFIRM_TOKEN_UNKNOWN")
        run.assert_not_called()

    def test_token_expires(self):
        self.existing_plan()
        server = self.server(scopes="read,write,destructive", ttl=60)
        b = self.preview(server, "plan", self.plan_args())
        self.assertEqual(b["expires_in_s"], 60)
        self.clock.t += 61
        res, run = self.call(server, "plan", self.plan_args(confirm_token=b["confirm_token"]))
        self.assertEqual(body(res)["error"], "CONFIRM_TOKEN_EXPIRED")
        run.assert_not_called()

    def test_token_is_bound_to_its_request(self):
        self.existing_plan()
        self.existing_plan("youtube")
        server = self.server(scopes="read,write,destructive")
        b = self.preview(server, "plan", self.plan_args())
        res, run = self.call(server, "plan", self.plan_args(profile="youtube", confirm_token=b["confirm_token"]))
        self.assertEqual(body(res)["error"], "CONFIRM_TOKEN_MISMATCH")
        run.assert_not_called()
        # spent by the refused attempt: not usable for the original either
        res, run = self.call(server, "plan", self.plan_args(confirm_token=b["confirm_token"]))
        self.assertEqual(body(res)["error"], "CONFIRM_TOKEN_UNKNOWN")
        # a token from another server instance, or a made-up one
        res, run = self.call(server, "plan", self.plan_args(confirm_token="x" * 32))
        self.assertEqual(body(res)["error"], "CONFIRM_TOKEN_UNKNOWN")
        run.assert_not_called()

    def test_token_refused_when_the_file_changed_after_preview(self):
        target = self.existing_plan()
        server = self.server(scopes="read,write,destructive")
        b = self.preview(server, "plan", self.plan_args())
        target.write_text(json.dumps({"plan": {"version": 4}}), encoding="utf-8")
        res, run = self.call(server, "plan", self.plan_args(confirm_token=b["confirm_token"]))
        self.assertEqual(body(res)["error"], "CONFIRM_TOKEN_MISMATCH")
        run.assert_not_called()

    def test_token_for_one_tool_is_refused_by_another(self):
        self.existing_plan()
        env = {"VIDEO_AGENT_AI_PROVIDER": "anthropic"}
        server = self.server(scopes="all", env=env)
        b = self.preview(server, "plan", self.plan_args())
        proj = str(self.ws / "plans" / "talk.generic.project.json")
        res, run = self.call(server, "revise", {"project": proj, "confirm_token": b["confirm_token"]})
        self.assertEqual(body(res)["error"], "CONFIRM_TOKEN_MISMATCH")
        run.assert_not_called()

    def test_publish_preview_lists_the_destinations_and_spend(self):
        proj = self.ws / "plans" / "card.offload.project.json"
        proj.write_text(json.dumps({"plan": {"steps": [{"id": "step_media_offload", "params": {"source": "/card", "destinations": [str(self.outside)]}}]}}),
                        encoding="utf-8")
        server = self.server(scopes="all", env={"VIDEO_AGENT_AI_PROVIDER": "anthropic"})
        b = self.preview(server, "render", {"project": str(proj)})
        self.assertEqual((b["class"], b["preview"][0]["action"], b["preview"][0]["destinations"]), ("publish", "copy_outside_roots", [str(self.outside)]))
        b = self.preview(server, "revise", {"project": str(proj)})
        self.assertEqual((b["preview"][0]["action"], b["preview"][0]["provider"]), ("spend", "anthropic"))
        with mock.patch.dict(os.environ, {"VIDEO_AGENT_AI_PROVIDER": "anthropic"}):
            res, run = self.call(server, "revise", {"project": str(proj), "confirm_token": b["confirm_token"]})
        self.assertFalse(res["isError"], res)
        self.assertNotIn("env", run.call_args[1], "a confirmed spending call keeps the configured provider")


class RootTests(_Base):
    def assert_outside(self, res, run):
        self.assertTrue(res["isError"], res)
        self.assertEqual(body(res)["error"], "PATH_OUTSIDE_ROOTS")
        run.assert_not_called()

    def test_traversal_is_refused(self):
        server = self.server(scopes="all")
        evil = str(self.ws / "plans" / ".." / ".." / "outside" / "p.project.json")
        res, run = self.call(server, "approve", {"project": evil, "decision": "all"})
        self.assert_outside(res, run)
        res, run = self.call(server, "plan", self.plan_args(profile="../../../../outside/x"))
        self.assert_outside(res, run)
        res, run = self.call(server, "render", {"project": str(self.outside / "p.project.json")})
        self.assert_outside(res, run)

    def test_tilde_is_read_as_the_child_reads_it(self):
        """The CLI child opens `~/...` literally (under its working directory), so the check must not expand it to a home inside a root."""
        with mock.patch.dict(os.environ, {"HOME": str(self.tmp), "USERPROFILE": str(self.tmp)}):
            res, run = self.call(self.server(scopes="all"), "approve", {"project": "~/ws/plans/p.project.json", "decision": "all"})
        self.assert_outside(res, run)

    def test_paid_revise_token_is_bound_to_the_project_state(self):
        proj = self.ws / "plans" / "p.project.json"
        proj.write_text(json.dumps({"plan": {"version": 1}}), encoding="utf-8")
        server = self.server(scopes="all", env={"VIDEO_AGENT_AI_PROVIDER": "anthropic"})
        res, _ = self.call(server, "revise", {"project": str(proj)})
        tok = body(res)["confirm_token"]
        proj.write_text(json.dumps({"plan": {"version": 2}}), encoding="utf-8")
        res, run = self.call(server, "revise", {"project": str(proj), "confirm_token": tok})
        self.assertEqual(body(res)["error"], "CONFIRM_TOKEN_MISMATCH")
        run.assert_not_called()

    def test_symlink_escapes_are_refused(self):
        server = self.server(scopes="all")
        target = self.outside / "real.project.json"
        target.write_text("{}", encoding="utf-8")
        link = self.ws / "plans" / "link.project.json"
        dirlink = self.ws / "escape"
        try:
            os.symlink(target, link)
            os.symlink(self.outside, dirlink, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"no symlinks here: {exc}")
        res, run = self.call(server, "approve", {"project": str(link), "decision": "all"})
        self.assert_outside(res, run)
        self.assertEqual(body(res)["paths"][0]["resolved"], os.path.realpath(target))
        res, run = self.call(server, "reject", {"project": str(dirlink / "new.project.json"), "decision": "d", "reason": "r"})
        self.assert_outside(res, run)
        # the plans folder itself a symlink out of the workspace: plan's own output is refused
        shutil.rmtree(self.ws / "plans")
        os.symlink(self.outside, self.ws / "plans", target_is_directory=True)
        res, run = self.call(server, "plan", self.plan_args())
        self.assert_outside(res, run)

    def test_inside_roots_runs(self):
        extra = self.tmp / "projects"
        extra.mkdir()
        res, run = self.call(self.server(roots=[str(extra)]), "approve", {"project": str(extra / "p.project.json"), "decision": "all"})
        self.assertFalse(res["isError"], res)
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
