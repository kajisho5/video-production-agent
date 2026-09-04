"""Integration + real-media tests: need ffmpeg and an ffmpeg-skill checkout (VIDEO_AGENT_FFMPEG_SKILL_DIR)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_ai_provider import FakeAIProvider  # noqa: E402
from video_agent.project import load_ir, save_ir
from video_agent.service import Service
from video_agent.tools.ffmpeg_skill.locate import locate_ffmpeg_skill
from video_agent.tools.ffmpeg_skill.catalog import CATALOG

TONE = "0.5*sin(2*PI*440*t)*between(t\\,3\\,14)*gt(sin(2*PI*0.7*t)\\,-0.6)"


def make_media(path: str, hdr: bool = False, vfr: bool = False) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30", "-f", "lavfi", "-i", f"aevalsrc='{TONE}':s=48000", "-t", "16"]
    if vfr:
        cmd += ["-vf", "select='gt(random(1)\\,0.3)'", "-fps_mode", "vfr"]
    if hdr:
        cmd += ["-vf", "format=yuv420p10le", "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:log-level=error", "-tag:v", "hvc1"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", "128k", path]
    subprocess.run(cmd, check=True)


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill(), "needs ffmpeg and ffmpeg-skill")
class RealMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="va_int_")
        cls.src = str(Path(cls.tmp) / "src" / "talk.mp4")
        Path(cls.src).parent.mkdir(parents=True)
        make_media(cls.src)
        cls.ws = str(Path(cls.tmp) / "ws")

    def test_end_to_end_youtube(self):
        svc = Service(workspace=self.ws)
        ir = svc.plan([self.src], "youtube")
        d = ir.doc
        sil = [o for o in d["analysis"]["observations"] if o["kind"] == "silence"][0]
        self.assertAlmostEqual(sil["data"]["silences"][0][1], 3.0, delta=0.2)
        self.assertEqual(len(d["video"]["operations"]), 1)
        self.assertEqual(len(d["audio"]["operations"]), 1)
        self.assertTrue(d["provenance"]["source_hashes"])
        ir_path = str(Path(self.ws) / "p.json")
        save_ir(ir, ir_path)
        rep = svc.validate(ir)
        self.assertTrue(rep.ok, rep.errors)
        out = svc.render(load_ir(ir_path), ir_path, timeout=600)
        self.assertEqual(out["status"], "COMPLETED", out.get("execution"))
        self.assertEqual(out["qa"]["status"], "PASS", [i for i in out["qa"]["items"] if i["status"] != "PASS"])
        art = out["artifacts"][0]
        self.assertTrue(os.path.exists(art["path"]))
        self.assertEqual(art["stage"], "candidate")
        chk = svc.check(art["path"], "youtube")
        self.assertTrue(chk["check"]["ok"])
        self.assertAlmostEqual(chk["probe"]["duration"], 11.0, delta=0.6)
        # the source was never touched
        self.assertEqual(ir.doc["provenance"]["source_hashes"][list(ir.doc["assets"])[0]], __import__("video_agent.media.analyzer", fromlist=["sha256_file"]).sha256_file(self.src))
        # provenance links every executed operation to its commands
        prov = json.loads((Path(self.ws) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        self.assertTrue(all(e["result"]["ok"] for e in prov["operations"]))
        self.assertTrue(any("ffmpeg" in c for e in prov["operations"] for c in (e["result"]["commands"] or [])))

    def test_reference_skill_contract_on_real_runtime(self):
        """Ecosystem contract with the real Reference Skill: registry package → ToolSpec → router → FfmpegSkillAdapter →
        ffmpeg-skill script → FFmpeg. Measurement (probe, loudness) and transform (cut) tools, selected by the registry."""
        from video_agent.tools.ffmpeg_skill import FfmpegSkillAdapter
        svc = Service(workspace=self.ws)
        router = svc.adapter([str(Path(self.src).parent)])
        pkg = svc.registry.package("ffmpeg-skill")
        self.assertEqual(pkg.version, router.version_of("ffmpeg-skill/probe"), "registry carries the detected version")
        self.assertIs(router.adapter_for("ffmpeg-skill/cut").__class__, FfmpegSkillAdapter)
        rows = {r["skill_id"]: r for r in svc.packages()}
        self.assertTrue(rows["ffmpeg-skill"]["available"])
        self.assertTrue({"ffmpeg-skill/probe", "ffmpeg-skill/cut", "ffmpeg-skill/loudness"} <= set(rows["ffmpeg-skill"]["usable_tools"]))
        tools = svc.tools_for(router)
        for skill in ("media_probe", "loudness_analysis", "silence_cleanup"):
            self.assertEqual(svc.registry.tool(tools[skill]).skill_id, "ffmpeg-skill")
        pr = router.measure(tools["media_probe"], {"inputs": [self.src]})
        self.assertTrue(pr.ok and set(svc.registry.tool(tools["media_probe"]).result_keys) <= set(pr.data), pr.data.keys())
        lm = router.measure(tools["loudness_analysis"], {"input": self.src, "measure_only": True})
        self.assertTrue(lm.ok and "input_i" in lm.data)
        from video_agent.models import Operation
        from video_agent.tools import ToolError
        outside = str(Path(self.tmp) / "contract_cut.mp4")
        with self.assertRaises(ToolError):   # adapter contract: outputs stay inside the workspace, sources are never overwritten
            router.run(Operation(tool=tools["silence_cleanup"], args={"input": self.src, "segments": "3-8", "output": outside}, inputs=[self.src], outputs=[outside]), {}, timeout=300)
        out = str(Path(self.ws) / "contract_cut.mp4")
        Path(self.ws).mkdir(parents=True, exist_ok=True)
        op = Operation(tool=tools["silence_cleanup"], args={"input": self.src, "segments": "3-8", "output": out}, inputs=[self.src], outputs=[out], skill="silence_cleanup")
        r = router.run(op, {}, timeout=300)
        self.assertTrue(r.ok, r.stderr_tail)
        self.assertTrue(os.path.exists(out) and os.path.exists(self.src), "source preserved, artifact produced")
        self.assertEqual(r.tool, tools["silence_cleanup"])

    def test_ai_recommendation_to_real_production_without_ai_commands(self):
        """FakeAIProvider → recommendation → AI_GENERATED inference → decision evidence → SkillRegistry → ffmpeg-skill tool →
        adapter → real media → QA → provenance. The provider never sees or emits a command; every executed command comes
        from the adapter's typed catalog."""
        prov = FakeAIProvider(intent="silence_cleanup", params={"tool": "ffmpeg-skill/cut", "argv": ["ffmpeg", "-y", "-i", "x"], "command": "ffmpeg -i in out"})
        svc = Service(workspace=self.ws, provider=prov)
        ir = svc.plan([self.src], "youtube")
        ai = [i for i in ir.doc["analysis"]["inferences"] if i["provenance"] == "AI_GENERATED"]
        self.assertEqual(len(ai), 1)
        self.assertEqual(set(ai[0]["data"]["params"]), set(), "tool / argv / command stripped from the AI params")
        lead = next(d for d in ir.doc["decisions"] if d["subject"] == "silence.leading")
        self.assertIn(ai[0]["id"], lead["evidence"])
        self.assertEqual({st["skill"]: st["tool"] for st in ir.doc["plan"]["steps"]}["silence_cleanup"], "ffmpeg-skill/cut")
        ir_path = str(Path(self.ws) / "ai.json")
        save_ir(ir, ir_path)
        self.assertTrue(svc.validate(ir).ok)
        out = svc.render(load_ir(ir_path), ir_path, timeout=600)
        self.assertEqual(out["status"], "COMPLETED", out.get("execution"))
        self.assertEqual(out["qa"]["status"], "PASS", [i for i in out["qa"]["items"] if i["status"] != "PASS"])
        prov_doc = json.loads((Path(self.ws) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        self.assertEqual(prov_doc["ai_calls"][0]["provider"], "fake")
        self.assertEqual(len(prov_doc["ai_calls"][0]["response_hash"]), 64)
        cut = next(e for e in prov_doc["operations"] if e["skill"] == "silence_cleanup")
        cmd = " ".join(cut["result"]["commands"])
        start = lead["params"]["end"]
        self.assertIn(f"-ss {start:.3f}", cmd, "the executed command follows the measured decision, reported by the engine itself")
        self.assertNotIn(" -i x", cmd, "the AI's argv never reached execution")
        self.assertNotIn("in out", cmd)
        self.assertTrue(cut["args"]["segments"].startswith(f"{start:.3f}-"), "typed adapter args come from the IR trim, not from the AI")
        self.assertNotIn("argv", json.dumps(cut["args"]))
        for text in (Path(ir_path).read_text(), json.dumps(prov_doc)):
            self.assertNotIn(prov.api_key, text)

    def test_resume_reuses_real_intermediates(self):
        svc = Service(workspace=self.ws)
        ir = svc.plan([self.src], "youtube", hash_sources=False)
        ir_path = str(Path(self.ws) / "resume.json")
        save_ir(ir, ir_path)
        first = svc.render(load_ir(ir_path), ir_path, timeout=600)
        self.assertEqual(first["status"], "COMPLETED")
        # simulate a failure after the intermediates: drop the delivery artifact, keep ops/
        os.remove(first["artifacts"][0]["path"])
        second = Service(workspace=self.ws).render(load_ir(ir_path), ir_path, timeout=600, resume=first["job"]["id"])
        self.assertEqual(second["status"], "COMPLETED", second["execution"])
        self.assertEqual(second["qa"]["status"], "PASS")
        self.assertEqual(len(second["execution"]["skipped"]), 2, "cut and loudness reused from the first job's ops/")
        self.assertTrue(all(first["job"]["id"] in p for p in second["execution"]["reused"].values()))
        self.assertTrue(os.path.exists(second["artifacts"][0]["path"]))
        self.assertIn(second["job"]["id"], second["artifacts"][0]["path"], "the new artifact lands in the new job")
        # same IR twice with resume: only check.py runs among the tool operations
        third = Service(workspace=self.ws).render(load_ir(ir_path), ir_path, timeout=600, resume="last")
        self.assertEqual(third["status"], "COMPLETED")
        self.assertEqual(len(third["execution"]["skipped"]), 3)
        self.assertEqual([r["tool"] for r in third["execution"]["results"]], ["ffmpeg-skill/check"])

    def test_reject_revise_approve_render_real_media(self):
        ws = str(Path(self.tmp) / "ws_rev")
        svc = Service(workspace=ws)
        ir = svc.plan([self.src], "conference", hash_sources=False)
        ir_path = str(Path(ws) / "conf.json")
        save_ir(ir, ir_path)
        lead = next(d for d in ir.doc["decisions"] if d["subject"] == "silence.leading")
        self.assertEqual(svc.reject(load_ir(ir_path), ir_path, [lead["id"]], reason="chair introduction")["rejected"], [lead["id"]])
        self.assertEqual(svc.render(load_ir(ir_path), ir_path)["status"], "BLOCKED")
        out = svc.revise(load_ir(ir_path), ir_path)
        self.assertTrue(out["created"])
        self.assertTrue(any(l.startswith("VIDEO") for l in out["diff"]["summary"]), out["diff"]["summary"])
        self.assertEqual(svc.render(load_ir(ir_path), ir_path)["status"], "WAITING_FOR_APPROVAL")
        self.assertTrue(svc.approve(load_ir(ir_path), ir_path, ["all"])["renderable"])
        res = svc.render(load_ir(ir_path), ir_path, timeout=600)
        self.assertEqual(res["status"], "COMPLETED", res.get("execution"))
        tools = [r["tool"] for r in res["execution"]["results"]]
        self.assertNotIn("ffmpeg-skill/cut", tools)
        self.assertIn("ffmpeg-skill/loudness", tools)
        # the delivered file keeps the full 16 s: the rejected trim really did not run
        self.assertAlmostEqual(svc.check(res["artifacts"][0]["path"])["probe"]["duration"], 16.0, delta=0.3)
        self.assertEqual(res["qa"]["status"], "PASS", [i for i in res["qa"]["items"] if i["status"] != "PASS"])
        self.assertTrue(Path(str(Path(ir_path).with_name("conf.v1.json"))).exists())

    @unittest.skipIf(os.name == "nt", "process-group check uses ps")
    def test_timeout_kills_the_whole_process_group(self):
        svc = Service(workspace=self.ws)
        ir = svc.plan([self.src], "youtube", hash_sources=False)
        ir_path = str(Path(self.ws) / "timeout.json")
        save_ir(ir, ir_path)
        out = svc.render(load_ir(ir_path), ir_path, timeout=1.0)
        self.assertEqual(out["status"], "FAILED")
        self.assertTrue(all(r["class"] == "TIMEOUT" for r in out["execution"]["recovery"]), out["execution"]["recovery"])
        job_dir = Path(self.ws) / "jobs" / out["job"]["id"]
        import time
        time.sleep(0.5)
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        self.assertNotIn(str(job_dir), ps, "an ffmpeg writing into the job dir survived the timeout")
        partial = list((job_dir / "ops").rglob("*.mp4"))
        self.assertEqual(partial, [], "half-written outputs must not be left for a retry to collide with")

    def test_hdr_and_vfr_are_observed_and_decided(self):
        hdr = str(Path(self.tmp) / "src" / "hdr.mp4")
        vfr = str(Path(self.tmp) / "src" / "vfr.mp4")
        make_media(hdr, hdr=True)
        make_media(vfr, vfr=True)
        svc = Service(workspace=self.ws)
        ir = svc.plan([hdr, vfr], "generic", hash_sources=False)
        subjects = [x["subject"] for x in ir.doc["decisions"]]
        self.assertIn("video.hdr", subjects)
        self.assertIn("video.vfr", subjects)
        hdr_dec = next(x for x in ir.doc["decisions"] if x["subject"] == "video.hdr")
        self.assertEqual(hdr_dec["approval"], "CONFIRM")

    def test_cli_round_trip(self):
        env = dict(os.environ, VIDEO_AGENT_WORKSPACE=self.ws)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "--json", "plan", self.src, "--profile", "youtube", "--no-hash"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        proj = json.loads(r.stdout)["project"]
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "--json", "render", proj, "--dry-run"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(json.loads(r.stdout)["operations"]), 4)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "explain", proj], capture_output=True, text=True, env=env)
        self.assertIn("evidence", r.stdout)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "doctor"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0)


@unittest.skipUnless(locate_ffmpeg_skill(), "needs ffmpeg-skill")
class FfmpegSkillContractTests(unittest.TestCase):
    """Contract tests: pin the ffmpeg-skill CLI surface the adapter relies on."""

    def test_version_and_scripts(self):
        skill = locate_ffmpeg_skill()
        self.assertTrue(skill.version_supported(), f"ffmpeg-skill {skill.version} outside the supported range")
        for name in CATALOG:
            self.assertIn(name, skill.scripts, f"script {name}.py missing")

    def test_revision_workflow_invokes_no_tool(self):
        """Contract: reject / revise / approve / diff never spawn ffmpeg-skill (they are pure IR transformations)."""
        import shutil as _sh
        skill = locate_ffmpeg_skill()
        broken = Path(tempfile.mkdtemp()) / "skill"
        (broken / "scripts").mkdir(parents=True)
        for name in skill.scripts:
            (broken / "scripts" / f"{name}.py").write_text("import sys; sys.exit(99)")
        (broken / "package.json").write_text('{"version": "0.8.4"}')
        ws = tempfile.mkdtemp()
        src = str(Path(ws) / "in.mp4")
        make_media(src)
        good = Service(workspace=ws)
        ir = good.plan([src], "conference", hash_sources=False)
        p = str(Path(ws) / "c.json")
        save_ir(ir, p)
        bad = Service(workspace=ws, ffmpeg_skill_dir=str(broken))
        lead = next(d for d in ir.doc["decisions"] if d["subject"] == "silence.leading")
        bad.reject(load_ir(p), p, [lead["id"]], reason="x")
        self.assertTrue(bad.revise(load_ir(p), p)["created"])
        self.assertTrue(bad.approve(load_ir(p), p, ["all"])["renderable"])

    def test_help_declares_every_catalog_flag(self):
        skill = locate_ffmpeg_skill()
        from video_agent.tools.ffmpeg_skill.catalog import FLAG_ALIASES
        for name, spec in CATALOG.items():
            out = subprocess.run([sys.executable, str(skill.script(name)), "--help"], capture_output=True, text=True).stdout
            for flag in spec["flags"]:
                cli = FLAG_ALIASES.get((name, flag)) or FLAG_ALIASES.get(("*", flag)) or "--" + flag.replace("_", "-")
                self.assertIn(cli, out, f"{name}.py --help does not mention {cli}")
            for common in ("--dry-run", "--json"):
                self.assertIn(common, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill(), "needs ffmpeg and ffmpeg-skill")
class NoAudioSourceTests(unittest.TestCase):
    """A video-only source (real-world: DJI drones, some screen captures) must analyze, plan and render without audio steps."""

    def test_video_only_source(self):
        tmp = tempfile.mkdtemp()
        src = str(Path(tmp) / "src" / "silent.mp4")
        Path(src).parent.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30", "-t", "6", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", src], check=True)
        ws = str(Path(tmp) / "ws")
        svc = Service(workspace=ws)
        ir = svc.plan([src], "youtube", hash_sources=False)
        self.assertIsNone(ir.doc["assets"][list(ir.doc["assets"])[0]]["technical"]["audio"])
        self.assertEqual(ir.doc["audio"]["operations"], [])
        self.assertEqual(ir.doc["video"]["operations"], [])
        self.assertEqual([s["skill"] for s in ir.doc["plan"]["steps"]], ["delivery_export", "delivery_check"])
        self.assertTrue(any("no audio stream" in w for w in ir.doc["analysis"]["warnings"]))
        p = str(Path(ws) / "s.json")
        save_ir(ir, p)
        out = svc.render(load_ir(p), p, timeout=600)
        self.assertEqual(out["status"], "COMPLETED", out.get("execution"))
        self.assertIn(out["qa"]["status"], ("PASS", "WARN"))
        self.assertFalse([i for i in out["qa"]["items"] if i["status"] == "FAIL"])
        env = dict(os.environ, VIDEO_AGENT_WORKSPACE=ws)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "analyze", src, "--no-hash"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("no audio", r.stdout, "analyze must still print the asset line for video-only sources")
