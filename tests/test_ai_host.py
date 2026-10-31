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
from video_agent.tools.discovery import MODULE_BOOTSTRAP, python_module_command  # noqa: E402

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


class HandoffTests(unittest.TestCase):
    """ADR-051: a step no installed skill can run is reported as a handoff naming the job and who provides it."""

    def test_step_without_a_tool_is_a_handoff(self):
        tmp = tempfile.mkdtemp()
        svc = Service(workspace=tmp, adapter=FakeAdapter(), caps=FakeCaps())
        ir = svc.plan([fake_media(tmp)], "youtube", request_text="冒頭の無音を詰めて")
        self.assertEqual(svc.handoffs(ir), [], "every step has a tool here")
        step = next(s for s in ir.doc["plan"]["steps"] if s["skill"] == "silence_cleanup")
        step["tool"] = None
        svc.caps = FakeCaps(missing=["ffmpeg"])
        [h] = svc.handoffs(ir)
        self.assertEqual((h["step"], h["skill"]), (step["id"], "silence_cleanup"))
        self.assertIn("ffmpeg", h["reason"])
        self.assertIn("video-editing", h["provided_by"])
        self.assertIn("/plugin install video-editing-skill@ai-video-production-os", h["install"])
        self.assertIn("/plugin install ffmpeg-skill@ai-video-production-os", h["install"])



# ADR-057 (roadmap step 1): job -> (owning package, repository, the install command the handoff must name). Written by hand from
# AI-video-production-OS ecosystem.json and each repository's README, independently of skills/ecosystem.py.
_GH = "https://github.com/kajisho5/"
ECOSYSTEM_JOBS = {
    # media_ingest (ingest) left this table with its adapter (ADR-066): the card offload is the implemented skill media_offload;
    # voice_enhancement (voice-enhance) left it the same way (ADR-073): the implemented skills voice_isolate / voice_dereverb / voice_bleed_reduce
    "dubbing": ("dubbing", "dubbing-skill", f"none yet: dubbing-skill is a scaffold repository with no released package ({_GH}dubbing-skill)"),
    "sound_design": ("sound-design", "sound-design-skill", f"none yet: sound-design-skill is a scaffold repository with no released package ({_GH}sound-design-skill)"),
    "speaker_diarization": ("diarization", "diarization-skill", f"git clone {_GH}diarization-skill && pip install -e 'diarization-skill[pyannote]'"),
    "visual_detection": ("detection", "detection-skill", f"git clone {_GH}detection-skill && pip install -e 'detection-skill[all]'"),
    "slide_sync": ("slide-sync", "slide-sync-skill", f"git clone {_GH}slide-sync-skill && pip install -e 'slide-sync-skill[pdf]'"),
    "platform_publish": ("publish", "publish-skill", f"none yet: publish-skill is a scaffold repository with no released package ({_GH}publish-skill)"),
    "still_image_edit": ("imagemagick", "imagemagick-skill", "npx imagemagick-skill"),
    "render_3d": ("blender", "blender-skill", "npx blender-skill"),
    "measured_motion_graphics": ("remotion-ffmpeg", "remotion-ffmpeg-skill", "npx remotion-ffmpeg-skill"),
    "html_scene_render": ("hyperframes", "hyperframes-skill", f"git clone {_GH}hyperframes-skill && cd hyperframes-skill && npm ci"),
    "document_verification": ("artifacts", "artifacts-skill", f"git clone {_GH}artifacts-skill && pip install -e 'artifacts-skill[all]'"),
    "bluray_authoring": ("bdmvkit", "bdmvkit", "pip install bdmvkit"),
    "dvd_authoring": ("vtskit", "vtskit", "pip install vtskit"),
    "disc_image": ("udfkit", "udfkit", "pip install udfkit"),
    "dvd_line21_captions": ("dvdcc", "dvdcc", "pip install dvdcc"),
}
# ecosystem.json `plugin: true` (listed in the Claude Code marketplace) / `optional: true` (./install.sh --with-optional)
PLUGINS = {"ingest-skill", "diarization-skill", "detection-skill", "slide-sync-skill", "imagemagick-skill", "blender-skill", "remotion-ffmpeg-skill",
           "hyperframes-skill", "artifacts-skill"}
OPTIONAL = {"diarization-skill", "detection-skill", "bdmvkit", "vtskit", "udfkit", "dvdcc"}


class EcosystemJobHandoffTests(unittest.TestCase):
    """ADR-057: a plan asking for a job another package of the ecosystem owns (`job.<skill>=true`) shows a BLOCKED step and a
    handoff naming that package, its repository and its install command. One test per job."""

    def assert_handoff(self, skill):
        package, repo, command = ECOSYSTEM_JOBS[skill]
        tmp = tempfile.mkdtemp()
        svc = Service(workspace=tmp, adapter=FakeAdapter(), caps=FakeCaps())
        ir = svc.plan([fake_media(tmp)], "youtube", user_requirements={f"job.{skill}": True})
        step = next(s for s in ir.doc["plan"]["steps"] if s["skill"] == skill)
        self.assertEqual((step["id"], step["tool"], step["status"]), (f"step_job_{skill}", None, "BLOCKED"))
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        asked = next(d for d in ir.doc["decisions"] if d["id"] == step["decision_id"])
        self.assertEqual((asked["subject"], asked["type"], asked["provenance"]), (f"job.{skill}", "TRANSFORM", "USER"))
        self.assertTrue(any(d["subject"] == f"job.{skill}" and d["approval"] == "BLOCK" for d in ir.doc["decisions"]))
        [h] = [h for h in svc.handoffs(ir) if h["skill"] == skill]
        self.assertEqual(h["step"], step["id"])
        self.assertIn("declared for phase 2 and not implemented", h["reason"])
        self.assertEqual((h["provided_by"], h["repositories"]), ([package], [repo]))
        self.assertIn(command, h["install"])
        self.assertEqual(f"/plugin install {repo}@ai-video-production-os" in h["install"], repo in PLUGINS)
        if not command.startswith("none yet"):
            self.assertIn("or ./install.sh" + (" --with-optional" if repo in OPTIONAL else "") + " from AI-video-production-OS (packages and CLIs)", h["install"])
        else:
            self.assertEqual(h["install"], [command], "nothing released: no command that cannot work")
        self.assertEqual(svc.registry.get(skill).phase, 2)
        self.assertEqual(svc.registry.get(skill).tools[0].split("/", 1)[0], package)

    def test_media_ingest_moved_to_the_offload_adapter(self):
        """ADR-066: `job.media_ingest` is no longer a handoff switch; it is refused with how the implemented offload is asked for, and a
        card planned without ingest-skill still hands the step over naming ingest-skill and its install command."""
        tmp = tempfile.mkdtemp()
        svc = Service(workspace=tmp, adapter=FakeAdapter(), caps=FakeCaps())
        with self.assertRaises(ValueError) as cm:
            svc.plan([fake_media(tmp)], "youtube", user_requirements={"job.media_ingest": True})
        self.assertIn("ingest.dest", str(cm.exception))
        self.assertNotIn("media_ingest", svc.registry.names())
        card = Path(tmp) / "CARD"; (card / "CLIP").mkdir(parents=True); (card / "CLIP" / "C0001.MP4").write_bytes(b"x")
        dest = Path(tmp) / "SSD1"; dest.mkdir()
        ir = svc.plan_offload(str(card), {"ingest.dest": str(dest)})
        [h] = svc.handoffs(ir)
        self.assertEqual((h["skill"], h["provided_by"], h["repositories"]), ("media_offload", ["ingest"], ["ingest-skill"]))
        self.assertIn(f"git clone {_GH}ingest-skill && pip install -e 'ingest-skill[fetch]'", h["install"])
        self.assertIn("/plugin install ingest-skill@ai-video-production-os", h["install"])
        self.assertEqual(list(dest.iterdir()), [], "nothing copied")

    def test_voice_enhancement_moved_to_the_voice_enhance_adapter(self):
        """ADR-073: `job.voice_enhancement` is no longer a handoff switch; it is refused with how the implemented ops are asked for, and a plan
        asking for one without voice-enhance-skill still hands the step over naming voice-enhance-skill and its install command."""
        tmp = tempfile.mkdtemp()
        svc = Service(workspace=tmp, adapter=FakeAdapter(), caps=FakeCaps(extra=["audio-production", "audio-production:CUT"]))
        with self.assertRaises(ValueError) as cm:
            svc.plan([fake_media(tmp)], "youtube", user_requirements={"job.voice_enhancement": True})
        self.assertIn("voice.isolate=true", str(cm.exception))
        self.assertNotIn("voice_enhancement", svc.registry.names())
        ir = svc.plan([fake_media(tmp)], "generic", user_requirements={"audio.production": True, "voice.dereverb": True})
        [h] = [h for h in svc.handoffs(ir) if h["skill"] == "voice_dereverb"]
        self.assertEqual((h["provided_by"], h["repositories"]), (["voice-enhance"], ["voice-enhance-skill"]))
        self.assertIn(f"git clone {_GH}voice-enhance-skill && pip install -e 'voice-enhance-skill'", h["install"])
        self.assertNotIn("/plugin install voice-enhance-skill@ai-video-production-os", h["install"], "not a marketplace plugin (ecosystem.json)")
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")

    def test_dubbing(self):
        self.assert_handoff("dubbing")

    def test_sound_design(self):
        self.assert_handoff("sound_design")

    def test_speaker_diarization(self):
        self.assert_handoff("speaker_diarization")

    def test_visual_detection(self):
        self.assert_handoff("visual_detection")

    def test_slide_sync(self):
        self.assert_handoff("slide_sync")

    def test_platform_publish(self):
        self.assert_handoff("platform_publish")

    def test_still_image_edit(self):
        self.assert_handoff("still_image_edit")

    def test_render_3d(self):
        self.assert_handoff("render_3d")

    def test_measured_motion_graphics(self):
        self.assert_handoff("measured_motion_graphics")

    def test_html_scene_render(self):
        self.assert_handoff("html_scene_render")

    def test_document_verification(self):
        self.assert_handoff("document_verification")

    def test_bluray_authoring(self):
        self.assert_handoff("bluray_authoring")

    def test_dvd_authoring(self):
        self.assert_handoff("dvd_authoring")

    def test_disc_image(self):
        self.assert_handoff("disc_image")

    def test_dvd_line21_captions(self):
        self.assert_handoff("dvd_line21_captions")

    def test_every_declared_job_is_covered(self):
        from video_agent.skills.registry import ECOSYSTEM_JOB_SKILLS
        self.assertEqual({s.name for s in ECOSYSTEM_JOB_SKILLS}, set(ECOSYSTEM_JOBS))

    def test_several_jobs_one_plan_and_nothing_without_the_switch(self):
        tmp = tempfile.mkdtemp()
        svc = Service(workspace=tmp, adapter=FakeAdapter(), caps=FakeCaps())
        ir = svc.plan([fake_media(tmp)], "youtube", user_requirements={"job.dubbing": True, "job.disc_image": "true", "job.render_3d": False})
        self.assertEqual(sorted(h["skill"] for h in svc.handoffs(ir)), ["disc_image", "dubbing"])
        plain = svc.plan([fake_media(tmp)], "youtube")
        self.assertFalse([s for s in plain.doc["plan"]["steps"] if s["id"].startswith("step_job_")])
        self.assertEqual(svc.handoffs(plain), [])

    def test_unknown_job_or_value_is_refused_before_analysis(self):
        tmp = tempfile.mkdtemp()
        svc = Service(workspace=tmp, adapter=FakeAdapter(), caps=FakeCaps())
        with self.assertRaisesRegex(Exception, "unknown job 'job.teleport'"):
            svc.plan([fake_media(tmp)], "youtube", user_requirements={"job.teleport": True})
        with self.assertRaisesRegex(Exception, "is a switch"):
            svc.plan([fake_media(tmp)], "youtube", user_requirements={"job.dubbing": "into English"})

    def test_snapshot_matches_the_os_ecosystem_json(self):
        """skills/ecosystem.py against a sibling AI-video-production-OS checkout's ecosystem.json, when there is one."""
        from video_agent.skills.ecosystem import COMPONENTS
        manifest = Path(__file__).resolve().parents[2] / "AI-video-production-OS" / "ecosystem.json"
        if not manifest.is_file():
            self.skipTest("no AI-video-production-OS checkout next to this repository")
        rows = {c["repo"]: c for c in json.loads(manifest.read_text(encoding="utf-8"))["components"]}
        for c in COMPONENTS.values():
            row = rows.get(c.repo)
            self.assertIsNotNone(row, f"{c.repo} is not in ecosystem.json")
            self.assertEqual((c.department, c.install, c.plugin, c.optional, list(c.owns)),
                             (row["department"], row["install"], bool(row.get("plugin")), bool(row.get("optional")), row["owns"]), c.repo)


class McpServerTests(unittest.TestCase):
    def test_tool_argv(self):
        tmp = tempfile.mkdtemp()
        self.assertEqual(tool_argv("plan", {"inputs": ["a.mp4"], "profile": "youtube", "request": "詰めて", "set": {"audio.loudness.target_lufs": -16}}, tmp),
                         ["plan", "--profile=youtube", "--request=詰めて", "--set=audio.loudness.target_lufs=-16", "--ai-brief", "--", "a.mp4"])
        self.assertNotIn("--ai-brief", tool_argv("plan", {"inputs": "a.mp4", "ai_brief": False}, tmp))
        argv = tool_argv("revise", {"project": "p.json", "ai_recommendations": {"recommendations": []}, "ai_model": "claude-desktop"}, tmp)
        f = next(a[len("--ai-recommendations="):] for a in argv if a.startswith("--ai-recommendations="))
        self.assertEqual(json.loads(Path(f).read_text(encoding="utf-8")), {"recommendations": []})
        self.assertEqual(tool_argv("render", {"project": "p.json", "dry_run": True}, tmp), ["render", "--dry-run", "--", "p.json"])
        with self.assertRaises(ValueError):
            tool_argv("approve", {"project": "p.json"}, tmp)
        self.assertNotIn("approve", json.dumps(TOOLS[[t["name"] for t in TOOLS].index("render")]["inputSchema"]), "render cannot approve inline")

    def test_values_are_never_options(self):
        """ADR-069: a value that looks like an option (`--output=...`) stays data: positionals follow `--`, options are `--flag=value`."""
        tmp = tempfile.mkdtemp()
        evil = "--output=" + str(Path(tmp) / "elsewhere.json")
        argv = tool_argv("plan", {"inputs": ["a.mp4", evil, "-o"], "profile": "--no-hash", "set": {"-k": "--v"}, "kinds": ["--allow-source-dir"]}, tmp)
        sep = argv.index("--")
        self.assertEqual(argv[sep + 1:], ["a.mp4", evil, "-o"])
        self.assertTrue(all(a.startswith("--") and "=" in a for a in argv[1:sep] if a != "--ai-brief"), argv)
        from video_agent.cli import build_parser
        ns = build_parser().parse_args(["--json", *argv])
        self.assertEqual(ns.inputs, ["a.mp4", evil, "-o"])
        self.assertEqual((ns.profile, ns.set, ns.no_hash, getattr(ns, "output", None)), ("--no-hash", ["-k=--v"], False, None))
        for name, args, key in (("brief", {"text": "--set=x=1"}, "text"), ("render", {"project": "--dry-run"}, "project"),
                                ("check", {"output": "--qc"}, "output"), ("approve", {"project": "--by=x", "decision": "--reason=r"}, "project"),
                                ("artifacts", {"project": "--job=j"}, "project")):
            ns = build_parser().parse_args(["--json", *tool_argv(name, args, tmp)])
            self.assertEqual(getattr(ns, key), args[key], name)
        self.assertFalse(getattr(ns, "job", None))

    def test_injected_option_does_not_write_outside(self):
        """ADR-069: through the server, an `inputs` element `--output=<path>` reaches the CLI as an input file name, never as the
        write target. The command the server launches is captured and parsed by the real CLI parser (independent of whether
        ffmpeg is installed), then the call is made for real and nothing is written outside."""
        from unittest import mock
        from video_agent.cli import build_parser
        tmp = tempfile.mkdtemp()
        target = Path(tmp) / "outside" / "planted.project.json"
        target.parent.mkdir()
        args = {"inputs": [fake_media(tmp), f"--output={target}", "--allow-source-dir", "--no-hash"], "ai_brief": False}
        server = Server(workspace=str(Path(tmp) / "ws"))
        with mock.patch("video_agent.mcp_server.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "")) as run:
            server.call_tool("plan", args)
        cmd = run.call_args[0][0]
        prefix = python_module_command("video_agent.cli", sys.executable)
        self.assertEqual(cmd[:len(prefix)], prefix)
        ns = build_parser().parse_args(cmd[len(prefix):])
        self.assertEqual(ns.inputs, args["inputs"], "every element is an input, including the option-looking ones")
        self.assertIsNone(getattr(ns, "output", None))
        self.assertFalse(ns.no_hash)
        self.assertFalse(getattr(ns, "allow_source_dir", None))
        res = server.call_tool("plan", args)
        self.assertTrue(res["isError"])
        self.assertFalse(target.exists())

    def test_every_listed_tool_maps_to_a_command(self):
        tmp = tempfile.mkdtemp()
        full = {"inputs": ["a"], "project": "p", "decision": "all", "reason": "r", "output": "o", "text": "t"}
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


class CwdIndependentAgentLaunchTests(unittest.TestCase):
    """ADR-075 (ADR-074's follow-up): the MCP server starts `video-agent` itself without the current directory on the
    child's import path. An MCP host (Claude Desktop, Claude Code) starts the server in whatever folder it likes, and the
    child inherits it; with the old `python -m video_agent.cli` a `json.py` or a `video_agent/` package planted there ran
    as the agent. Each planted module appends its name to a marker file outside that folder when imported."""

    def setUp(self):
        import shutil
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.untrusted = self.tmp / "untrusted"   # the host's working directory (an unpacked project, a download)
        self.marker = self.tmp / "imported.txt"
        planted = self.untrusted / "video_agent"
        planted.mkdir(parents=True)
        note = "with open({path!r}, 'a') as _f:\n    _f.write({name!r} + '\\n')\n"
        (self.untrusted / "json.py").write_text(note.format(path=str(self.marker), name="json") + "def dumps(*a, **k):\n    return 'PLANTED'\n", encoding="utf-8")
        for name in ("__init__", "cli", "__main__"):
            (planted / f"{name}.py").write_text(note.format(path=str(self.marker), name=f"video_agent.{name}") + "print('PLANTED')\n", encoding="utf-8")
        self._old_cwd = os.getcwd()
        os.chdir(self.untrusted)
        self.addCleanup(os.chdir, self._old_cwd)
        old = os.environ.get("PYTHONPATH")   # the real package, as an install would provide it (CI also runs `pip install -e .`)
        os.environ["PYTHONPATH"] = SRC + (os.pathsep + old if old else "")
        self.addCleanup(lambda: os.environ.pop("PYTHONPATH", None) if old is None else os.environ.__setitem__("PYTHONPATH", old))
        self.ws = str(self.tmp / "ws")

    def imported(self):
        return self.marker.read_text(encoding="utf-8").split() if self.marker.exists() else []

    def test_planted_modules_run_under_plain_dash_m(self):
        """The control: the planting is effective -- the old launch runs the planted package instead of the agent."""
        p = subprocess.run([sys.executable, "-m", "video_agent.cli", "--json", "--workspace", self.ws, "artifacts"],
                           capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
        self.assertIn("video_agent.__init__", self.imported())
        self.assertIn("PLANTED", p.stdout)

    def test_server_launch_shape(self):
        from unittest import mock
        server = Server(workspace=self.ws, python="py")
        with mock.patch("video_agent.mcp_server.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "{}", "")) as run:
            server.call_tool("artifacts", {})
        cmd = run.call_args[0][0]
        self.assertEqual(cmd, ["py", "-c", MODULE_BOOTSTRAP, "video_agent.cli", "--json", "--workspace", self.ws, "artifacts", "--"])
        self.assertNotIn("-m", cmd)
        self.assertIsNone(run.call_args[1].get("cwd"), "the working directory is inherited as before (relative paths resolve the same way)")

    def test_server_child_does_not_import_from_the_current_directory(self):
        """The real call: the child is the installed agent with the stdlib json, run from the planted directory."""
        res = Server(workspace=self.ws).call_tool("artifacts", {})
        text = res["content"][0]["text"]
        self.assertEqual(self.imported(), [], "nothing from the current directory was imported")
        self.assertNotIn("PLANTED", text)
        self.assertFalse(res["isError"], text)
        self.assertIsInstance(json.loads(text), (dict, list), "the real CLI's --json output")

    def test_relative_paths_still_resolve_against_the_working_directory(self):
        """A relative project path in a tool call is resolved by the child against the inherited working directory."""
        res = Server(workspace=self.ws).call_tool("explain", {"project": "missing.project.json"})
        text = res["content"][0]["text"]
        self.assertEqual(self.imported(), [])
        self.assertTrue(res["isError"], text)
        self.assertIn("missing.project.json", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
