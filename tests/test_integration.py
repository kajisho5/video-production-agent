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
