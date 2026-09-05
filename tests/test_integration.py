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
from video_agent.tools.media_analysis import locate_media_analysis
from video_agent.tools.transcription import locate_transcription

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

    def test_observation_cache_skips_the_analyzer_on_the_second_run(self):
        """Real media: first plan measures (probe / silence / loudness), second plan in the same workspace is served from the
        observation cache with zero measurement calls, and the evidence (observation ids, decisions, plan hash) is identical.
        Then the full chain with an AI recommendation runs on the cached evidence."""
        ws = str(Path(self.tmp) / "ws_cache")
        svc = Service(workspace=ws)
        ir1 = svc.plan([self.src], "youtube")
        a1 = ir1.doc["analysis"]["analyses"][0]
        self.assertEqual((a1["budget"]["calls"], a1["cache"]["hits"], a1["cache"]["misses"]), (3, 0, 3))
        self.assertTrue(all(o["source"].endswith("@" + svc.registry.package("ffmpeg-skill").version) for o in ir1.doc["analysis"]["observations"]))
        svc2 = Service(workspace=ws, provider=FakeAIProvider(intent="silence_cleanup"))
        ir2 = svc2.plan([self.src], "youtube")
        a2 = ir2.doc["analysis"]["analyses"][0]
        self.assertEqual((a2["budget"]["calls"], a2["cache"]["hits"]), (0, 3), "second run: analyzer not executed")
        self.assertEqual([o["id"] for o in ir1.doc["analysis"]["observations"]], [o["id"] for o in ir2.doc["analysis"]["observations"]])
        same = lambda ir: ([(o["kind"], o["data"]) for o in ir.doc["analysis"]["observations"]], [(d["subject"], d["decision"]) for d in ir.doc["decisions"]])  # noqa: E731
        self.assertEqual(same(ir1), same(ir2), "identical evidence and decisions from cached observations (asset ids differ per plan by design)")
        self.assertNotEqual(a1["analysis_id"], a2["analysis_id"])
        ai = [i for i in ir2.doc["analysis"]["inferences"] if i["provenance"] == "AI_GENERATED"]
        self.assertEqual(len(ai), 1)
        self.assertTrue(set(ai[0]["evidence"]) <= {o["id"] for o in ir2.doc["analysis"]["observations"]} | {e["id"] for e in ir2.doc["timeline"]["events"]})
        ir_path = str(Path(ws) / "cached.json")
        save_ir(ir2, ir_path)
        self.assertTrue(svc2.validate(ir2).ok)
        out = svc2.render(load_ir(ir_path), ir_path, timeout=600)
        self.assertEqual(out["status"], "COMPLETED", out.get("execution"))
        self.assertEqual(out["qa"]["status"], "PASS")
        # CACHED_ONLY is honoured on real media too
        ir3 = Service(workspace=ws).plan([self.src], "youtube", strategy="CACHED_ONLY")
        self.assertEqual((ir3.doc["analysis"]["strategy"], ir3.doc["analysis"]["budget"]["calls"]), ("CACHED_ONLY", 0))

    def test_temporal_events_and_sessions_on_real_media(self):
        """Real media → analysis → observations → deterministic Observation → Event transformation → session → IR → CLI.
        The 3 s lead-in silence becomes an AudioEvent(silence) with the silence observation as evidence."""
        from video_agent.temporal import events_from_observation, sort_key
        svc = Service(workspace=self.ws)
        ir = svc.plan([self.src], "youtube")
        d = ir.doc
        events = d["timeline"]["events"]
        sil_obs = next(o for o in d["analysis"]["observations"] if o["kind"] == "silence")
        lead = [e for e in events if e["type"] == "AUDIO_SILENCE" and e["range"]["start"] == 0.0]
        self.assertEqual(len(lead), 1)
        self.assertAlmostEqual(lead[0]["range"]["end"], 3.0, delta=0.2)
        self.assertEqual((lead[0]["event_type"], lead[0]["subtype"], lead[0]["provenance"], lead[0]["evidence"]), ("AudioEvent", "silence", "OBSERVED", [sil_obs["id"]]))
        self.assertTrue(lead[0]["source"].endswith("@" + svc.registry.package("ffmpeg-skill").version))
        self.assertEqual([e["id"] for e in events], [e["id"] for e in sorted(events, key=lambda e: (e["range"]["start"], e["range"]["end"] if e["range"]["end"] is not None else e["range"]["start"], e["type"], e["id"]))], "canonical order")
        self.assertEqual(len(d["timeline"]["sessions"]), 1)
        ses = d["timeline"]["sessions"][0]
        self.assertEqual((ses["project_id"], ses["asset_ids"], ses["range"]["start"]), (d["project"]["id"], list(d["assets"]), 0.0))
        self.assertAlmostEqual(ses["range"]["end"], 16.0, delta=0.2)
        self.assertEqual(sorted(ses["event_ids"]), sorted(e["id"] for e in events))
        self.assertTrue(svc.validate(ir).ok, svc.validate(ir).errors)
        # regenerating from the recorded observation gives the same identities (idempotent across runs)
        from video_agent.models import Asset, Observation
        asset = Asset.from_dict(d["assets"][ses["asset_ids"][0]])
        regen = events_from_observation(Observation.from_dict(sil_obs), asset)
        self.assertTrue({e.id for e in regen} <= {e["id"] for e in events})
        ir_path = str(Path(self.ws) / "temporal.json")
        save_ir(ir, ir_path)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        for cmd in (["events"], ["sessions"], ["--json", "events"], ["--json", "sessions"], ["explain"]):
            r = subprocess.run([sys.executable, "-m", "video_agent.cli", "--workspace", self.ws] + cmd + [ir_path], capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(r.stdout.strip())
        self.assertIn("AudioEvent", subprocess.run([sys.executable, "-m", "video_agent.cli", "--workspace", self.ws, "events", ir_path], capture_output=True, text=True, env=env).stdout)
        out = svc.render(load_ir(ir_path), ir_path, timeout=600)
        self.assertEqual(out["status"], "COMPLETED", out.get("execution"))

    def test_vertical_slice_observation_to_qa_through_production_plan(self):
        """The first complete production: talk.mp4 (3 s lead-in silence) → analysis → silence observation → AudioEvent(silence)
        → decision → ProductionPlan step (trim, evidence chain) → Project IR → compiler → ffmpeg-skill → FFmpeg → QA PASS →
        explain. Original untouched, output shorter by the removed silence, provenance complete."""
        from video_agent.agent.production_plan import explain_step
        import hashlib
        before = hashlib.sha256(Path(self.src).read_bytes()).hexdigest()
        svc = Service(workspace=self.ws)
        ir = svc.plan([self.src], "youtube")
        d = ir.doc
        pl = d["plan"]
        self.assertEqual((pl["status"], [st["skill"] for st in pl["steps"]]), ("APPROVED", ["silence_cleanup", "loudness_normalization", "delivery_export", "delivery_check"]))
        trim = pl["steps"][0]
        self.assertAlmostEqual(trim["temporal_scope"]["start"], 2.85, delta=0.25)
        sil_event = next(e for e in d["timeline"]["events"] if e["subtype"] == "silence" and e["range"]["start"] == 0.0)
        self.assertIn(sil_event["id"], trim["evidence"])
        sil_obs = next(o for o in d["analysis"]["observations"] if o["kind"] == "silence")
        self.assertIn(sil_obs["id"], trim["evidence"])
        chain = explain_step(d, trim["id"])["chain"]
        self.assertEqual([r["kind"] for r in chain][:1], ["decision"])
        self.assertTrue(any(r["kind"] == "event" and r["id"] == sil_event["id"] for r in chain))
        self.assertTrue(any(r["kind"] == "observation" and r["source"].startswith("ffmpeg-skill/silence@") for r in chain))
        self.assertEqual(pl["outputs"][0]["logical"], f"{list(d['assets'])[0]}_delivery_youtube")
        ir_path = str(Path(self.ws) / "slice.json")
        save_ir(ir, ir_path)
        self.assertTrue(svc.validate(ir).ok, svc.validate(ir).errors)
        out = svc.render(load_ir(ir_path), ir_path, timeout=600)
        self.assertEqual(out["status"], "COMPLETED", out.get("execution"))
        self.assertEqual(out["qa"]["status"], "PASS", [i for i in out["qa"]["items"] if i["status"] != "PASS"])
        self.assertEqual(hashlib.sha256(Path(self.src).read_bytes()).hexdigest(), before, "original unchanged")
        art = out["artifacts"][0]
        got = svc.check(art["path"])["probe"]["duration"]
        self.assertAlmostEqual(got, 16.0 - trim["temporal_scope"]["start"] - (16.0 - trim["temporal_scope"]["end"]), delta=0.6)
        prov = json.loads((Path(self.ws) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        cut = next(e for e in prov["operations"] if e["skill"] == "silence_cleanup")
        self.assertEqual(sorted(cut["decision"]), sorted(trim["decision_ids"]))
        env = dict(os.environ); env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "--workspace", self.ws, "explain", ir_path, "--step", trim["id"]], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("observation", r.stdout)
        self.assertIn("ffmpeg-skill/silence@", r.stdout)
        # no-audio media: no silence event, no silence step, render still works, QA WARN only for audio
        silent = str(Path(self.tmp) / "src" / "noaudio.mp4")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=320x240:d=4", "-c:v", "libx264", "-preset", "veryfast", silent], check=True)
        ir2 = svc.plan([silent], "youtube")
        self.assertFalse([e for e in ir2.doc["timeline"]["events"] if e["subtype"] == "silence"])
        self.assertEqual([st["skill"] for st in ir2.doc["plan"]["steps"]], ["delivery_export", "delivery_check"])
        p2 = str(Path(self.ws) / "noaudio.json"); save_ir(ir2, p2)
        out2 = svc.render(load_ir(p2), p2, timeout=600)
        self.assertEqual(out2["status"], "COMPLETED")
        self.assertEqual(out2["qa"]["status"], "WARN")
        self.assertTrue(all("audio" in i["name"] for i in out2["qa"]["items"] if i["status"] != "PASS"))
        # hostile AI through the whole slice: recommendation becomes evidence, tool / argv / command never reach the plan or execution
        prov_ai = FakeAIProvider(intent="silence_cleanup", params={"tool": "ffmpeg-skill/export", "argv": ["ffmpeg", "-y", "-i", "x"], "command": "rm -rf /"})
        svc3 = Service(workspace=self.ws, provider=prov_ai)
        ir3 = svc3.plan([self.src], "youtube")
        self.assertNotIn("rm -rf", json.dumps(ir3.doc["plan"]))
        self.assertEqual([st["tool"] for st in ir3.doc["plan"]["steps"]], ["ffmpeg-skill/cut", "ffmpeg-skill/loudness", "ffmpeg-skill/export", "ffmpeg-skill/check"])
        p3 = str(Path(self.ws) / "ai.json"); save_ir(ir3, p3)
        out3 = svc3.render(load_ir(p3), p3, timeout=600)
        self.assertEqual(out3["status"], "COMPLETED")
        prov3 = json.loads((Path(self.ws) / "jobs" / out3["job"]["id"] / "provenance.json").read_text())
        self.assertNotIn("-i x", " ".join(" ".join(e["result"]["commands"]) for e in prov3["operations"] if e.get("result")))

    def test_artifact_lifecycle_on_real_media(self):
        """talk.mp4 → plan → render → artifact registration (real sha256) → QA PASS → deliver (READY → DELIVERED) → archive;
        original untouched; explain --artifact reaches the observation; a revision yields a separate artifact; resume reuses."""
        import hashlib
        before = hashlib.sha256(Path(self.src).read_bytes()).hexdigest()
        ws = str(Path(self.tmp) / "ws_art")
        svc = Service(workspace=ws)
        ir = svc.plan([self.src], "youtube")
        p = str(Path(ws) / "art.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, timeout=600)
        self.assertEqual(out["status"], "COMPLETED")
        a = out["artifacts"][0]
        self.assertEqual(a["hash"], hashlib.sha256(Path(a["path"]).read_bytes()).hexdigest())
        self.assertEqual((a["qa_status"], a["stage"], a["delivery_status"]), ("PASS", "candidate", "READY"))
        self.assertEqual(a["media"].get("video_stream"), "h264")
        self.assertTrue(a["name"].endswith(".mp4"))
        self.assertEqual(hashlib.sha256(Path(self.src).read_bytes()).hexdigest(), before)
        m = svc.artifact(a["id"])
        self.assertTrue(m["integrity"]["ok"] and m["integrity"]["size"] == a["size"])
        info = svc.explain_artifact(a["id"])
        self.assertTrue(info["step"] and info["step"]["step"]["skill"] == "delivery_export")
        self.assertTrue(any(o["skill"] == "delivery_export" for o in info["operations"]))
        d = svc.promote_artifact(a["id"], "final", who="tester", reason="approved for upload")
        self.assertEqual(d["delivery_status"], "DELIVERED")
        arch = svc.archive_artifact(a["id"], who="tester")
        self.assertEqual(arch["delivery_status"], "ARCHIVED")
        idx = svc.artifact_store().archive_index(ir.doc["project"]["id"])
        self.assertEqual(idx["entries"][0]["sha256"], a["hash"])
        env = dict(os.environ); env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        for cmd in (["artifacts", p], ["artifact", a["id"]], ["explain", "--artifact", a["id"]], ["archive", p], ["--json", "artifacts", p]):
            r = subprocess.run([sys.executable, "-m", "video_agent.cli", "--workspace", ws] + cmd, capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(a["id"], r.stdout)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "--workspace", ws, "deliver", a["id"]], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 3, "archived artifact cannot be delivered again")
        # resume: same bytes -> same artifact identity, second job appended
        out2 = Service(workspace=ws).render(load_ir(p), p, resume=out["job"]["id"], timeout=600)
        self.assertEqual(out2["status"], "COMPLETED")
        self.assertEqual(out2["artifacts"][0]["id"], a["id"])
        self.assertEqual(out2["artifacts"][0]["jobs"], [out["job"]["id"], out2["job"]["id"]])
        # revision: v2 artifact is separate, v1 still intact
        svc.reject(load_ir(p), p, [ir.doc["plan"]["steps"][0]["decision_id"]], reason="keep lead-in")
        svc.revise(load_ir(p), p); svc.approve(load_ir(p), p, ["all"])
        out3 = Service(workspace=ws).render(load_ir(p), p, timeout=600)
        self.assertEqual(out3["status"], "COMPLETED")
        b = out3["artifacts"][0]
        self.assertNotEqual(b["id"], a["id"]); self.assertEqual(b["plan_version"], 2)
        self.assertTrue(svc.artifact(a["id"])["integrity"]["ok"])
        v2 = load_ir(p)
        scope = next(st for st in v2.doc["plan"]["steps"] if st["skill"] == "silence_cleanup")["temporal_scope"]   # only the trailing trim remains in v2
        self.assertEqual(scope["start"], 0.0)
        self.assertAlmostEqual(svc.check(b["path"])["probe"]["duration"], scope["end"] - scope["start"], delta=0.6)
        # no-audio media keeps QA WARN and a READY (WARN) artifact
        silent = str(Path(self.tmp) / "src" / "noaudio2.mp4")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=320x240:d=3", "-c:v", "libx264", "-preset", "veryfast", silent], check=True)
        ir4 = svc.plan([silent], "youtube"); p4 = str(Path(ws) / "na.json"); save_ir(ir4, p4)
        out4 = svc.render(load_ir(p4), p4, timeout=600)
        self.assertEqual((out4["status"], out4["artifacts"][0]["qa_status"], out4["artifacts"][0]["delivery_status"]), ("COMPLETED", "WARN", "READY"))

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


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill() and locate_media_analysis(), "needs ffmpeg, ffmpeg-skill and media-analysis-skill")
class MediaAnalysisRealTests(unittest.TestCase):
    """External observation Skill (media-analysis-skill, ADR-023) on real media through the JSON process boundary:
    contract discovery, every measurement kind, Observation lifting with provenance, Skill-owned cache, events, and a
    full plan → render → QA where the measurement Skill (not the media engine) supplies probe / silence / loudness."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="va_ma_")
        cls.src = str(Path(cls.tmp) / "src" / "talk.mp4")
        Path(cls.src).parent.mkdir(parents=True)
        make_media(cls.src)

    def _service(self, ws: str, prefer_media_analysis: bool = True, **kw) -> Service:
        svc = Service(workspace=ws, **kw)
        if prefer_media_analysis:   # measurement from the observation Skill; ffmpeg-skill stays the engine for processing
            for name, tool in (("media_probe", "media-analysis/probe"), ("silence_analysis", "media-analysis/silence"), ("loudness_analysis", "media-analysis/loudness")):
                svc.registry.get(name).tools = [tool]
        return svc

    def test_capability_and_contract_come_from_the_installed_skill(self):
        svc = Service(workspace=str(Path(self.tmp) / "ws_cap"))
        cap = svc.caps.resolve()["media-analysis"]
        self.assertEqual(cap.status, "AVAILABLE", cap.evidence)
        self.assertEqual((cap.evidence["version"], cap.evidence["contract"], cap.evidence["execution"]), ("0.1.0", "media-analysis/contract@1", "local_subprocess"))
        self.assertEqual(len(cap.evidence["tools"]), 9)
        rows = {r["skill_id"]: r for r in svc.packages()}
        self.assertTrue(rows["media-analysis"]["available"] and rows["ffmpeg-skill"]["available"])
        tools = svc.tools_for()
        self.assertEqual(tools["media_probe"], "ffmpeg-skill/probe", "the Reference Skill stays the first candidate when both are installed")
        for skill, tool in (("stream_layout_analysis", "media-analysis/streams"), ("video_format_analysis", "media-analysis/video"), ("audio_format_analysis", "media-analysis/audio"),
                            ("duration_analysis", "media-analysis/timing"), ("integrity_analysis", "media-analysis/integrity"), ("scene_analysis", "media-analysis/scenes"), ("timing_analysis", "media-analysis/timing")):
            self.assertEqual(tools[skill], tool)
        # a missing installation is a MISSING capability with the reason, never an import error or a fallback
        svc2 = Service(workspace=str(Path(self.tmp) / "ws_cap2"), media_analysis_dir="/nonexistent")
        self.assertEqual(svc2.caps.resolve()["media-analysis"].status, "AVAILABLE" if locate_media_analysis("/nonexistent") else "MISSING")

    def test_all_kinds_lifted_with_provenance_and_skill_cache(self):
        ws = str(Path(self.tmp) / "ws_all")
        svc = self._service(ws)
        extra = ["stream_layout", "video_format", "audio_format", "duration", "integrity", "scene_detection", "timing"]
        profile, rules, an = svc.analyze([self.src], "youtube", kinds=extra)
        obs = {o.kind: o for o in an.observations}
        self.assertEqual(sorted(obs), sorted(["media_probe", "silence", "loudness"] + extra))
        for o in obs.values():
            self.assertEqual((o.provenance, o.skill, o.skill_version), ("OBSERVED", "media-analysis", "0.1.0"), o.kind)
            self.assertTrue(o.tool.startswith("media-analysis/") and o.source == f"{o.tool}@0.1.0" and o.external_id.startswith("obs_"), o.kind)
            self.assertEqual(len(o.fingerprint), 64)
            self.assertEqual(o.cache.get("status"), "miss", o.kind)
        self.assertEqual(obs["duration"].tool, obs["timing"].tool, "one Skill tool serves two kinds; the kind stays on the observation")
        self.assertEqual(len({o.fingerprint for o in obs.values()}), 1, "shared asset identity: one content fingerprint for every observation")
        # facts as measured: the Skill's own keys are kept, consumers read through one vocabulary
        from video_agent.media.analysis import loudness_facts
        self.assertIn("integrated_lufs", obs["loudness"].data)
        lf = loudness_facts(obs["loudness"].data)
        self.assertFalse(lf["silent"]); self.assertLess(lf["lufs"], -5); self.assertLessEqual(lf["true_peak"], 0.0)
        self.assertAlmostEqual(obs["silence"].data["segments"][0]["end"], 3.0, delta=0.2)
        self.assertEqual(an.assets[0].technical["duration"] and round(an.assets[0].technical["duration"]), 16)
        self.assertEqual(an.assets[0].technical["video"]["width"], 1280)
        ev = {e.type for e in an.timeline.events}
        self.assertTrue({"AUDIO_SILENCE", "AUDIO_ACTIVE", "LOUDNESS_MEASURE"} <= ev, ev)
        sil = next(e for e in an.timeline.events if e.type == "AUDIO_SILENCE")
        self.assertEqual((sil.range["start"], sil.evidence, sil.provenance), (0.0, [obs["silence"].id], "OBSERVED"))
        rows = {r["kind"]: r for r in an.analyses[0]["rows"]}
        self.assertTrue(all(r["cache_owner"] == "media-analysis" for r in rows.values()))
        self.assertEqual((an.analyses[0]["cache"]["hits"], an.analyses[0]["cache"]["misses"]), (0, 0), "the agent's own cache is not used for Skill-owned measurements")
        # second run in the same workspace: the Skill's cache answers (status hit), provenance keeps saying so
        _, _, an2 = self._service(ws).analyze([self.src], "youtube", kinds=extra)
        self.assertTrue(all(o.cache.get("status") == "hit" for o in an2.observations), [(o.kind, o.cache) for o in an2.observations])
        self.assertEqual([(o.kind, o.data) for o in an.observations], [(o.kind, o.data) for o in an2.observations], "identical facts from the Skill cache")
        self.assertTrue(all(r["cache_hit"] for r in an2.analyses[0]["rows"]))

    def test_plan_render_qa_with_measurement_from_the_skill(self):
        ws = str(Path(self.tmp) / "ws_render")
        svc = self._service(ws)
        ir = svc.plan([self.src], "youtube")
        d = ir.doc
        self.assertTrue(all(o["skill"] == "media-analysis" for o in d["analysis"]["observations"]))
        self.assertEqual([s["skill"] for s in d["plan"]["steps"]][:1], ["silence_cleanup"])
        self.assertTrue(all(s["tool"].startswith("ffmpeg-skill/") for s in d["plan"]["steps"]), "processing steps stay on the media engine")
        self.assertNotIn("media-analysis", json.dumps(d["plan"]), "no observation tool ever becomes a plan step")
        self.assertAlmostEqual(d["video"]["operations"][0]["keep"][0][0], 3.0, delta=0.2)
        inf = next(i for i in d["analysis"]["inferences"] if i["kind"] == "loudness_off_target")
        self.assertAlmostEqual(inf["data"]["lufs"], -11.0, delta=1.0, msg="the inference reads the Skill's loudness vocabulary (integrated_lufs)")
        self.assertEqual(len(d["audio"]["operations"]), 1)
        ir_path = str(Path(ws) / "p.json")
        save_ir(ir, ir_path)
        self.assertTrue(svc.validate(ir).ok, svc.validate(ir).errors)
        out = svc.render(load_ir(ir_path), ir_path, timeout=600)
        self.assertEqual(out["status"], "COMPLETED", out.get("execution"))
        self.assertEqual(out["qa"]["status"], "PASS", [i for i in out["qa"]["items"] if i["status"] != "PASS"])
        ms = out["qa"]["measurements"]
        self.assertTrue(any(m["tool"] == "media-analysis/probe" and m["args"]["kind"] == "media_probe" for m in ms))
        self.assertTrue(any(m["tool"] == "media-analysis/loudness" and m["args"]["kind"] == "loudness" for m in ms))
        self.assertTrue(all("command" not in m["args"] and "argv" not in m["args"] for m in ms))
        self.assertTrue(any(i["name"] == "loudness" and i["status"] == "PASS" for i in out["qa"]["items"]))
        # doctor / CLI round trip through the located Skill
        env = dict(os.environ, VIDEO_AGENT_WORKSPACE=ws)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "analyze", self.src, "--no-hash", "--kind", "duration", "--kind", "integrity"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("integrity", r.stdout); self.assertIn("media-analysis", r.stdout)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "--json", "doctor"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["media-analysis"]["status"], "AVAILABLE")


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill() and locate_transcription(), "needs ffmpeg, ffmpeg-skill and transcription-skill")
class TranscriptionRealTests(unittest.TestCase):
    """External recognition Skill (transcription-skill, ADR-024) on real media through the `run -` process boundary.
    Contract discovery, doctor and the engine contract always run against the installed Skill. Real recognition runs only
    when the Skill's doctor reports the engine and its default model as locally available (no model download is forced
    by CI): otherwise the recognition tests are skipped with the doctor's reason."""

    @classmethod
    def setUpClass(cls):
        from video_agent.tools.transcription import TranscriptionAdapter
        cls.tmp = tempfile.mkdtemp(prefix="va_ts_")
        src_dir = Path(cls.tmp) / "src"
        src_dir.mkdir()
        cls.src = str(src_dir / "talk.mp4")
        make_media(cls.src)
        fixture = Path(locate_transcription().root or "") / "tests" / "fixtures" / "ja_short.wav"
        cls.ja = str(src_dir / "ja_short.wav") if fixture.is_file() else None
        if cls.ja:
            shutil.copy(fixture, cls.ja)
        cls.adapter = TranscriptionAdapter(workspace=str(Path(cls.tmp) / "ws" / "cache" / "transcription"), allowed_inputs=[str(src_dir)], offline=True)
        cls.doctor = cls.adapter.doctor()
        rows = {c.get("check"): c for c in cls.doctor.get("checks") or []}
        eng = next((e for e in cls.adapter.engine_status() if e.get("id") == "faster_whisper"), {})
        default_model = eng.get("default_model") or "base"
        cls.model_row = rows.get(f"model:faster_whisper:{default_model}") or {}
        cls.can_recognise = bool(eng.get("available")) and cls.model_row.get("status") == "AVAILABLE"
        cls.skip_reason = f"recognition needs faster_whisper and a local '{default_model}' model: engine available={eng.get('available')}, model {cls.model_row.get('status')} ({cls.model_row.get('detail', '')[:80]})"

    def test_contract_doctor_and_engine_contract_from_the_installed_skill(self):
        from video_agent.tools.transcription import check_contract
        ad = self.adapter
        self.assertEqual(check_contract(ad.contract), [])
        self.assertEqual((ad.contract["id"], ad.version[:4], ad.contract["schemas"]["transcript"]), ("transcription-skill", "0.2.", "transcription-skill/transcript/0.1"))
        self.assertEqual(sorted(ad.tools), ["transcription/check", "transcription/export", "transcription/segments", "transcription/transcribe"])
        self.assertTrue(ad.supports("transcription/transcribe") and not ad.supports("transcription/export"))
        eng = {e["id"]: e for e in ad.engine_status()}
        self.assertEqual((eng["faster_whisper"]["execution_mode"], eng["faster_whisper"]["requires_network"]), ("local", False), "the only implemented engine is local")
        self.assertEqual(list(eng), ["faster_whisper"], "no cloud engine, no whisper.cpp: the contract lists implemented engines only")
        self.assertIsInstance(self.doctor.get("checks"), list)
        rows = {c["check"]: c["status"] for c in self.doctor["checks"]}
        self.assertEqual(rows["skill"], "AVAILABLE")
        self.assertIn("engine:faster_whisper", rows)
        policy = next(c for c in self.doctor["checks"] if c["check"] == "input path policy")
        self.assertEqual(policy["mode"], "allowed_roots", "the adapter's roots reach the Skill's doctor")
        self.assertTrue(self.doctor.get("offline"))
        self.assertNotRegex(json.dumps(self.doctor), r"(?i)(api[_-]?key|token|secret|password)")
        svc = Service(workspace=str(Path(self.tmp) / "ws_cap"))
        cap = svc.caps.resolve()["transcription"]
        engine_installed = bool(eng["faster_whisper"].get("available"))
        # the Skill's doctor decides: engine installed → AVAILABLE (model local) / DEGRADED (model missing); no engine → MISSING (CI has none)
        self.assertIn(cap.status, ("AVAILABLE", "DEGRADED") if engine_installed else ("MISSING",), cap.detail)
        if engine_installed:
            self.assertEqual(cap.evidence["engines"][0]["id"], "faster_whisper")
        rows = {r["skill_id"]: r for r in svc.packages()}
        self.assertTrue(rows["transcription"]["implemented"])
        self.assertEqual(svc.tools_for().get("speech_transcription"), "transcription/transcribe" if engine_installed else None, "no engine → no candidate tool, never a fallback")
        # an engine-level constraint the Skill refuses is reported as the Skill says it (no reinterpretation)
        r = self.adapter.measure("transcription/transcribe", {"input": self.src, "asset_id": "asset_x", "engine": "faster_whisper", "model": "large-v3", "offline": True})
        if not r.ok:
            self.assertIn(r.data["error"]["code"], ("MODEL_UNAVAILABLE", "ENGINE_UNAVAILABLE"))
            if r.data["error"]["code"] == "MODEL_UNAVAILABLE":
                self.assertEqual(r.data["error"]["details"]["availability"], "MODEL_MISSING")
        # the Skill's path policy refuses an input outside the adapter's roots even when the adapter is bypassed
        import subprocess
        outside = Path(self.tmp) / "outside.wav"
        outside.write_bytes(b"RIFF" + b"\x00" * 64)
        req = {"tool": "transcription/transcribe", "params": {"input": str(outside), "asset_id": "a", "allowed_input_roots": [str(Path(self.tmp) / "src")], "offline": True}}
        skill = locate_transcription()
        env = dict(os.environ, **skill.env)
        p = subprocess.run(list(skill.command) + ["run", "-"], input=json.dumps(req), capture_output=True, text=True, env=env)
        doc = json.loads(p.stdout)
        self.assertEqual((doc["ok"], doc["error"]["code"], doc["error"]["details"]["reason"]), (False, "INVALID_INPUT", "outside_allowed_roots"))

    def test_real_recognition_lifting_provenance_cache_and_speech_events(self):
        if not self.can_recognise:
            self.skipTest(self.skip_reason)
        if not self.ja:
            self.skipTest("transcription-skill fixture ja_short.wav not found")
        ws = str(Path(self.tmp) / "ws_real")
        svc = Service(workspace=ws, offline=True)
        _, _, an = svc.analyze([self.ja], "generic", kinds=["transcript"], params={"language": "ja"})
        obs = {o.kind: o for o in an.observations}
        self.assertIn("transcript", obs, [r for r in an.analyses[0]["rows"] if r["status"] != "OK"])
        t = obs["transcript"]
        self.assertEqual((t.provenance, t.skill, t.tool, t.source), ("OBSERVED", "transcription", "transcription/transcribe", f"transcription/transcribe@{self.adapter.version}"))
        self.assertEqual(t.skill_version, self.adapter.version)
        self.assertEqual((t.parameters["engine"], t.parameters["execution_mode"], t.parameters["model"], t.parameters["language"]), ("faster_whisper", "local", "base", "ja"))
        self.assertTrue(t.parameters["engine_version"] and t.parameters["model_version"])
        self.assertEqual(t.fingerprint, an.assets[0].hash, "shared asset identity: the Skill's sha256 is the agent's asset hash")
        self.assertEqual((t.asset_id, t.data["asset_id"]), (an.assets[0].id, an.assets[0].id))
        self.assertEqual(t.data["language"], "ja")
        self.assertGreaterEqual(len(t.data["segments"]), 1)
        self.assertTrue(all(s["speaker_id"] is None for s in t.data["segments"]))
        self.assertIn("本日", t.data["segments"][0]["text"], "recognised text as produced by the engine (no correction)")
        self.assertEqual(t.data["provenance"]["skill"], "transcription-skill")
        self.assertEqual(t.cache["status"], "miss")
        rows = {r["kind"]: r for r in an.analyses[0]["rows"]}
        self.assertEqual((rows["transcript"]["cache_owner"], rows["transcript"]["engine"]["id"]), ("transcription", "faster_whisper"))
        sp = an.timeline.query(type="SPEECH")
        self.assertEqual(len(sp), len(t.data["segments"]))
        self.assertTrue(all(e.provenance == "OBSERVED" and e.evidence == [t.id] and e.metadata["speaker_id"] is None for e in sp))
        self.assertLessEqual(sp[-1].range["end"], an.assets[0].technical["duration"] + 0.5)
        # second run in the same workspace: the Skill's cache answers; the fact is byte-identical
        _, _, an2 = Service(workspace=ws, offline=True).analyze([self.ja], "generic", kinds=["transcript"], params={"language": "ja"})
        t2 = next(o for o in an2.observations if o.kind == "transcript")
        self.assertEqual(t2.cache["status"], "hit")
        self.assertEqual((t2.data["id"], t2.data["segments"]), (t.data["id"], t.data["segments"]))
        self.assertEqual([(e.range, e.metadata["text"]) for e in an2.timeline.query(type="SPEECH")], [(e.range, e.metadata["text"]) for e in sp],
                         "identical speech intervals from the cached fact (event ids carry the per-analysis asset / observation ids by design, ADR-020)")
        # media-analysis (when installed) and transcription observe the same asset
        from video_agent.tools.media_analysis import locate_media_analysis
        if locate_media_analysis():
            _, _, an3 = Service(workspace=ws, offline=True).analyze([self.ja], "generic", kinds=["duration", "transcript"], params={"language": "ja"})
            o3 = {o.kind: o for o in an3.observations}
            self.assertEqual(len(an3.assets), 1)
            self.assertEqual((o3["duration"].asset_id, o3["transcript"].asset_id), (an3.assets[0].id, an3.assets[0].id))
            self.assertEqual(o3["transcript"].fingerprint, an3.assets[0].hash)
        # plan with a transcript: SpeechEvents in the IR, nothing derived from them; explain follows the chain to facts only
        ir = svc.plan([self.src], "youtube", kinds=["transcript"], params={"language": "en"})
        d = ir.doc
        self.assertTrue(svc.validate(ir).ok, svc.validate(ir).errors)
        tr = next(o for o in d["analysis"]["observations"] if o["kind"] == "transcript")
        blob = json.dumps({"plan": d["plan"], "decisions": d["decisions"], "inferences": d["analysis"]["inferences"]})
        for bad in ("SPEECH", "transcription", "speaker", tr["id"]):
            self.assertNotIn(bad, blob)
        info = Service.explain_observation(d, tr["id"])
        self.assertEqual([r["kind"] for r in info["chain"]][:6], ["observation", "skill", "tool", "engine", "model", "transcript"])
        self.assertTrue(next(r for r in info["chain"] if r["kind"] == "asset")["shared_identity"])
        # CLI round trip: transcribe / explain --observation / doctor
        env = dict(os.environ, VIDEO_AGENT_WORKSPACE=ws)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "transcribe", self.ja, "--language", "ja", "--offline", "--allowed-input", str(Path(self.ja).parent)], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("transcript tr_", r.stdout); self.assertIn("SPEECH", r.stdout); self.assertIn("speaker_id null", r.stdout); self.assertIn("cache hit", r.stdout)
        ir_path = str(Path(ws) / "p.json")
        save_ir(ir, ir_path)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "explain", ir_path, "--observation", tr["external_id"]], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("faster_whisper@", r.stdout); self.assertIn("no inference, decision", r.stdout)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "--json", "doctor"], capture_output=True, text=True, env=env)
        self.assertEqual(json.loads(r.stdout)["transcription"]["status"], "AVAILABLE")


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill() and locate_transcription(), "needs ffmpeg, ffmpeg-skill and transcription-skill")
class SpeechToPlanRealTests(unittest.TestCase):
    """PR #14 on real media: transcription-skill → Transcript → SpeechEvent → speech inferences (intervals, conflicts with the
    measured silence) → decisions → ProductionPlan → Project IR → validate → explain. Fixture: the Skill's ja_short.wav twice
    with a 3 s pause in between (real speech, real pause). Runs only when the engine and its default model are local."""

    @classmethod
    def setUpClass(cls):
        from video_agent.tools.transcription import TranscriptionAdapter
        cls.tmp = tempfile.mkdtemp(prefix="va_sp_")
        src_dir = Path(cls.tmp) / "src"
        src_dir.mkdir()
        fixture = Path(locate_transcription().root or "") / "tests" / "fixtures" / "ja_short.wav"
        cls.ready = fixture.is_file()
        if cls.ready:
            cls.src = str(src_dir / "two_takes.wav")
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(fixture), "-i", str(fixture), "-filter_complex",
                            "[0:a]apad=pad_dur=3[a0];[a0][1:a]concat=n=2:v=0:a=1[a1];[a1]apad=pad_dur=3[a]", "-map", "[a]", "-c:a", "pcm_s16le", cls.src], check=True)   # trailing pad: the engine's segment end may overshoot the last word by up to ~1 s
            ad = TranscriptionAdapter(workspace=str(Path(cls.tmp) / "ws" / "cache" / "transcription"), allowed_inputs=[str(src_dir)], offline=True)
            rows = {c.get("check"): c for c in ad.doctor().get("checks") or []}
            eng = next((e for e in ad.engine_status() if e.get("id") == "faster_whisper"), {})
            cls.ready = bool(eng.get("available")) and (rows.get(f"model:faster_whisper:{eng.get('default_model') or 'base'}") or {}).get("status") == "AVAILABLE"

    def test_speech_and_silence_to_reviewable_plan(self):
        if not self.ready:
            self.skipTest("needs the transcription-skill fixture, faster-whisper and a local default model")
        from video_agent.agent.production_plan import executable_steps, explain_step
        ws = str(Path(self.tmp) / "ws")
        svc = Service(workspace=ws, offline=True)
        # the measured facts, before any planning: speech events and silence events on one timeline, timestamps as measured
        _, _, an = svc.analyze([self.src], "youtube", kinds=["transcript"], params={"language": "ja"})
        sp = sorted(an.timeline.query(type="SPEECH"), key=lambda e: e.range["start"])
        sil = sorted(an.timeline.query(type="AUDIO_SILENCE"), key=lambda e: e.range["start"])
        self.assertEqual(len(sp), 2, [(e.range, e.metadata.get("text")) for e in sp])
        self.assertTrue(all(e.metadata["speaker_id"] is None for e in sp))
        self.assertTrue(any(8.0 <= e.range["start"] <= 10.0 and 12.0 <= (e.range["end"] or 0) <= 13.5 for e in sil), f"the 3 s pause is measured as silence: {[e.range for e in sil]}")
        pause = next(e for e in sil if 8.0 <= e.range["start"] <= 10.0)
        ir = svc.plan([self.src], "youtube", kinds=["transcript"], params={"language": "ja"})
        d = ir.doc
        rep = svc.validate(ir)
        self.assertEqual(rep.errors, [], rep.errors)
        # events in the IR are the measured ones, unchanged by inference
        ir_sp = sorted([e for e in d["timeline"]["events"] if e["type"] == "SPEECH"], key=lambda e: e["range"]["start"])
        ir_sil = [e for e in d["timeline"]["events"] if e["type"] == "AUDIO_SILENCE" and e["range"]["start"] == pause.range["start"]]
        self.assertEqual([e["range"] for e in ir_sp], [e.range for e in sp])
        self.assertEqual(ir_sil[0]["range"], pause.range)
        self.assertEqual(ir_sil[0]["source"], pause.source, "silence provenance (the measurement tool) is kept")
        infs = d["analysis"]["inferences"]
        by = {}
        for i in infs:
            by.setdefault(i["kind"], []).append(i)
        self.assertEqual(len(by["speech_interval"]), 2)
        self.assertEqual(len(by["speech_activity"]), 1)
        self.assertTrue(all(i["provenance"] == "INFERRED" and i["data"]["speaker_id"] is None for k in ("speech_interval", "speech_activity") for i in by[k]))
        self.assertTrue(all(any(e["id"] in i["evidence"] for e in ir_sp) for i in by["speech_interval"]), "each interval cites a SpeechEvent of this plan's analysis")
        # Whisper's segments extend into the measured pause on this recording, so the layers disagree: recorded as a conflict,
        # no removal candidate, nothing corrected; a trim overlapping a conflict needs confirmation
        conflicts = by.get("speech_silence_conflict", [])
        cands = [x for x in d["decisions"] if x["subject"].startswith("silence.internal.")]
        if conflicts:
            self.assertTrue(any(c["data"]["silence"]["start"] == pause.range["start"] for c in conflicts))
            self.assertEqual([c for c in cands if c["params"]["start"] >= pause.range["start"] and c["params"]["end"] <= pause.range["end"]], [], "a disputed pause is never a removal candidate")
        else:
            self.assertEqual(len(cands), 1, "no conflict on this recording: the pause is a CONFIRM candidate")
            self.assertEqual(cands[0]["approval"], "CONFIRM")
        for x in d["decisions"]:
            if x["subject"] in ("silence.leading", "silence.trailing") and any(c["id"] in x["evidence"] for c in conflicts):
                self.assertEqual(x["approval"], "CONFIRM", x["subject"])
        self.assertTrue(all(x["approval"] != "AUTO" for x in cands))
        cont = next(x for x in d["decisions"] if x["subject"] == "speech.continuity")
        self.assertEqual((cont["approval"], cont["params"]["intervals"]), ("AUTO", 2))
        # plan / IR: no event, transcript, engine or speaker material; the trim (if any) waits for confirmation when disputed
        blob = json.dumps({"plan": d["plan"], "video": d["video"], "audio": d["audio"], "delivery": d["delivery"]})
        for bad in ("SPEECH", "speaker", "transcription", "faster_whisper", "argv"):
            self.assertNotIn(bad, blob)
        trim = next((s for s in d["plan"]["steps"] if s["skill"] == "silence_cleanup"), None)
        if trim and any(x["approval"] == "CONFIRM" and x["status"] == "PROPOSED" for x in d["decisions"] if x["id"] in trim["decision_ids"]):
            self.assertEqual(d["plan"]["status"], "REVIEW")
            self.assertNotIn(trim["id"], executable_steps(d))
            chain = explain_step(d, trim["id"])["chain"]
            details = [(r["kind"], r.get("detail") or "") for r in chain]
            self.assertTrue(any(k == "event" and "SpeechEvent" in det for k, det in details), "explain reaches the SpeechEvent")
            self.assertTrue(any(k == "observation" and det == "transcript" for k, det in details), "…and the transcript observation")
            self.assertTrue(any(k == "event" and "AudioEvent/silence" in det for k, det in details))
        # CLI: explain --step / --decision on the saved IR
        ir_path = str(Path(ws) / "p.json")
        save_ir(ir, ir_path)
        env = dict(os.environ, VIDEO_AGENT_WORKSPACE=ws)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "explain", ir_path, "--decision", "speech.continuity"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("keep all 2 speech interval(s)", r.stdout)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "validate", ir_path], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill() and locate_transcription(), "needs ffmpeg, ffmpeg-skill and transcription-skill")
class ProductionContextRealTests(unittest.TestCase):
    """PR #15 on real media: Transcript → SpeechEvent + measured silence → ProductionContexts → generic inference → explain.
    Uses the same two-take fixture as SpeechToPlanRealTests; recognition runs only when the engine and its model are local."""

    @classmethod
    def setUpClass(cls):
        from video_agent.tools.transcription import TranscriptionAdapter
        cls.tmp = tempfile.mkdtemp(prefix="va_cx_")
        src_dir = Path(cls.tmp) / "src"
        src_dir.mkdir()
        fixture = Path(locate_transcription().root or "") / "tests" / "fixtures" / "ja_short.wav"
        cls.ready = fixture.is_file()
        if cls.ready:
            cls.src = str(src_dir / "two_takes.wav")
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(fixture), "-i", str(fixture), "-filter_complex",
                            "[0:a]apad=pad_dur=3[a0];[a0][1:a]concat=n=2:v=0:a=1[a1];[a1]apad=pad_dur=3[a]", "-map", "[a]", "-c:a", "pcm_s16le", cls.src], check=True)   # trailing pad: the engine's segment end may overshoot the last word by up to ~1 s
            ad = TranscriptionAdapter(workspace=str(Path(cls.tmp) / "ws" / "cache" / "transcription"), allowed_inputs=[str(src_dir)], offline=True)
            rows = {c.get("check"): c for c in ad.doctor().get("checks") or []}
            eng = next((e for e in ad.engine_status() if e.get("id") == "faster_whisper"), {})
            cls.ready = bool(eng.get("available")) and (rows.get(f"model:faster_whisper:{eng.get('default_model') or 'base'}") or {}).get("status") == "AVAILABLE"

    def test_contexts_and_generic_inference_on_real_media(self):
        if not self.ready:
            self.skipTest("needs the transcription-skill fixture, faster-whisper and a local default model")
        from video_agent.context import contexts_at
        ws = str(Path(self.tmp) / "ws")
        svc = Service(workspace=ws, offline=True)
        ir = svc.plan([self.src], "youtube", kinds=["transcript"], params={"language": "ja"})
        d = ir.doc
        self.assertEqual(svc.validate(ir).errors, [], svc.validate(ir).errors)
        ctxs = Service.contexts_of(d)
        self.assertTrue(ctxs)
        dur = list(d["assets"].values())[0]["technical"]["duration"]
        self.assertEqual((ctxs[0].scope["start"], ctxs[-1].scope["end"]), (0.0, dur))
        events = {e["id"]: e for e in d["timeline"]["events"]}
        points = {p for e in events.values() if e.get("event_type") != "UserDecisionEvent" for p in (e["range"]["start"], e["range"].get("end")) if p is not None} | {0.0, dur}
        self.assertTrue(all(c.scope["start"] in points and c.scope["end"] in points for c in ctxs), "situation boundaries are the measured events' own timestamps")
        # the situation while the first take is spoken: speech recognised, audio active, no measured silence
        sp = sorted([e for e in events.values() if e["type"] == "SPEECH"], key=lambda e: e["range"]["start"])
        self.assertEqual(len(sp), 2)
        mid = (sp[0]["range"]["start"] + min(sp[0]["range"]["end"], 8.4)) / 2
        at = contexts_at(ctxs, mid)
        self.assertEqual(len(at), 1)
        self.assertIn("SpeechEvent/speech", at[0].signature)
        self.assertIn(sp[0]["id"], at[0].event_ids)
        transcript = next(o for o in d["analysis"]["observations"] if o["kind"] == "transcript")
        self.assertIn(transcript["id"], at[0].observation_ids)
        infs = d["analysis"]["inferences"]
        by = {}
        for i in infs:
            by.setdefault(i["kind"], []).append(i)
        act = {(i["data"]["event_type"], i["data"]["subtype"]): i for i in by["source_activity"]}
        self.assertEqual(act[("SpeechEvent", "speech")]["data"]["intervals"], [[e["range"]["start"], e["range"]["end"]] for e in sp], "activity intervals are the recognised segments, untouched")
        self.assertIn(("AudioEvent", "silence"), act)
        self.assertTrue(by.get("transition"))
        self.assertTrue(all(set(i["evidence"]) <= set(events) for k in ("source_activity", "source_inactivity", "transition", "conflict") for i in by.get(k, [])))
        # on this recording Whisper's segments reach into the measured pause: the generic layer records the conflict, PR #14's domain layer too; neither corrects anything
        conflicts = by.get("conflict", [])
        sil = [e for e in events.values() if e["type"] == "AUDIO_SILENCE"]
        for c in conflicts:
            self.assertEqual(c["data"]["codes"], ["AUDIO_SILENCE", "SPEECH"])
            for eid in c["evidence"]:
                self.assertEqual(events[eid]["provenance"], "OBSERVED")
        if conflicts:
            self.assertTrue(by.get("speech_silence_conflict"), "the domain layer sees the same disagreement")
        self.assertTrue(all(e["range"]["end"] > e["range"]["start"] for e in sil))
        # explain: context → track → event → observation (transcript / silence) and the decisions resting on it
        pause_ctx = next((c for c in ctxs if "AudioEvent/silence" in c.signature and c.scope["start"] > 1.0), None)
        self.assertIsNotNone(pause_ctx)
        info = Service.explain_context(d, pause_ctx.id)
        kinds = {r["kind"] for r in info["chain"]}
        self.assertTrue({"context", "track", "event", "observation"} <= kinds, kinds)
        self.assertTrue(any(r["kind"] == "observation" and r["detail"] == "silence" for r in info["chain"]))
        info_o = Service.explain_observation(d, transcript["id"])
        self.assertTrue(any(r["kind"] == "context" for r in info_o["chain"]), "transcript → SpeechEvent → context is traceable")
        blob = json.dumps({"plan": d["plan"], "video": d["video"], "audio": d["audio"]})
        for bad in ("ctx_", "SPEECH", "transition", "argv"):
            self.assertNotIn(bad, blob)
        # CLI
        ir_path = str(Path(ws) / "p.json")
        save_ir(ir, ir_path)
        env = dict(os.environ, VIDEO_AGENT_WORKSPACE=ws)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "context", ir_path, "--at", str(mid)], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("SpeechEvent/speech", r.stdout)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "explain", ir_path, "--context", pause_ctx.id], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("observation", r.stdout)
