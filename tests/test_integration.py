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

    def test_approve_revise_approve_render_real_media(self):
        """ADR-034: APPROVED v1 → revise → APPROVED v2 → render → QA → artifact on real media; v1's reviews stay history."""
        ws = str(Path(self.tmp) / "ws_rev2")
        svc = Service(workspace=ws)
        ir = svc.plan([self.src], "conference", hash_sources=False)
        ir_path = str(Path(ws) / "conf2.json")
        save_ir(ir, ir_path)
        pending = [d["id"] for d in ir.pending_confirmations()]
        self.assertTrue(svc.approve(load_ir(ir_path), ir_path, pending, who="reviewer")["renderable"])
        v1_reviews = dict(load_ir(ir_path).doc["execution"]["reviews"])
        out = svc.revise(load_ir(ir_path), ir_path, feedback="a little more headroom", user_requirements={"audio.loudness.true_peak": -2.0}, who="editor")
        self.assertTrue(out["created"], out)
        v2 = load_ir(ir_path)
        self.assertEqual((v2.version, svc.validate(v2).errors), (2, []))
        self.assertEqual(v2.doc["revision"]["history"][-1]["reviews"], v1_reviews)
        self.assertEqual(svc.render(load_ir(ir_path), ir_path, timeout=600)["status"], "WAITING_FOR_APPROVAL")
        self.assertTrue(svc.approve(load_ir(ir_path), ir_path, ["all"], who="reviewer")["renderable"])
        res = svc.render(load_ir(ir_path), ir_path, timeout=600)
        self.assertEqual(res["status"], "COMPLETED", res.get("execution"))
        self.assertEqual(res["job"]["plan_version"], 2)
        self.assertEqual(res["qa"]["status"], "PASS", [i for i in res["qa"]["items"] if i["status"] != "PASS"])
        self.assertTrue(res["artifacts"] and os.path.isfile(res["artifacts"][0]["path"]))
        self.assertEqual(next(op for op in v2.doc["audio"]["operations"] if op["type"] == "audio.loudness")["true_peak"], -2.0)
        self.assertTrue(Path(str(Path(ir_path).with_name("conf2.v1.json"))).exists())

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


class ProductionDecisionEngineRealTests(unittest.TestCase):
    """PR #16 on real media: Transcript → SpeechEvent → ProductionContext → Inference → Decision (typed, grounded, policy-resolved
    with provenance) → ProductionPlan → IR → explain --decision. Same two-take fixture as SpeechToPlanRealTests; recognition runs
    only when the engine and its default model are local."""

    @classmethod
    def setUpClass(cls):
        from video_agent.tools.transcription import TranscriptionAdapter
        cls.tmp = tempfile.mkdtemp(prefix="va_de_")
        src_dir = Path(cls.tmp) / "src"
        src_dir.mkdir()
        fixture = Path(locate_transcription().root or "") / "tests" / "fixtures" / "ja_short.wav"
        cls.ready = fixture.is_file()
        if cls.ready:
            cls.src = str(src_dir / "two_takes.wav")
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(fixture), "-i", str(fixture), "-filter_complex",
                            "[0:a]apad=pad_dur=3[a0];[a0][1:a]concat=n=2:v=0:a=1[a1];[a1]apad=pad_dur=3[a]", "-map", "[a]", "-c:a", "pcm_s16le", cls.src], check=True)
            ad = TranscriptionAdapter(workspace=str(Path(cls.tmp) / "ws" / "cache" / "transcription"), allowed_inputs=[str(src_dir)], offline=True)
            rows = {c.get("check"): c for c in ad.doctor().get("checks") or []}
            eng = next((e for e in ad.engine_status() if e.get("id") == "faster_whisper"), {})
            cls.ready = bool(eng.get("available")) and (rows.get(f"model:faster_whisper:{eng.get('default_model') or 'base'}") or {}).get("status") == "AVAILABLE"

    def test_decisions_on_real_media_are_typed_grounded_and_explainable(self):
        if not self.ready:
            self.skipTest("needs the transcription-skill fixture, faster-whisper and a local default model")
        from video_agent.agent.decision_engine import DECISION_TYPES, EXECUTABLE_TYPES, check_decisions
        ws = str(Path(self.tmp) / "ws")
        svc = Service(workspace=ws, offline=True)
        ir = svc.plan([self.src], "conference", kinds=["transcript"], params={"language": "ja"})
        d = ir.doc
        rep = svc.validate(ir)
        self.assertEqual(rep.errors, [], rep.errors)
        self.assertEqual(check_decisions(d), [])
        decs = {x["subject"]: x for x in d["decisions"]}
        self.assertIn("speech.continuity", decs)
        for x in d["decisions"]:
            self.assertIn(x["type"], DECISION_TYPES, x["subject"])
            self.assertTrue(x["evidence"] and x["basis"]["engine"] == "decision_engine@1.0", x["subject"])
            self.assertEqual(x["basis"]["approval"]["resolved"], x["approval"])
        # conference policy: lead / tail trims CONFIRM from the profile (PROFILE provenance), recorded on the decision, plan in REVIEW
        trims = [x for x in d["decisions"] if x["subject"] in ("silence.leading", "silence.trailing")]
        self.assertTrue(trims, [x["subject"] for x in d["decisions"]])
        for t in trims:
            self.assertEqual((t["type"], t["approval"]), ("REMOVE", "CONFIRM"), t["subject"])
            self.assertEqual(t["basis"]["approval"]["provenance"], "PROFILE")
            self.assertEqual(next(s for s in t["basis"]["settings"] if s["key"] == f"{t['subject']}.approval")["rule_id"], f"conf.{t['subject']}.approval")
        self.assertEqual(d["plan"]["status"], "REVIEW")
        # only executable types reach steps; KEEP decisions (speech continuity, disputed intervals) never do
        cited = {did for s in d["plan"]["steps"] for did in s["decision_ids"]}
        self.assertTrue(all(decs_by_id["type"] in EXECUTABLE_TYPES for decs_by_id in d["decisions"] if decs_by_id["id"] in cited))
        self.assertNotIn(decs["speech.continuity"]["id"], cited)
        # explain --decision: basis rows (policy / constraint / approval / intent / requirement / risk) and the evidence chain down to the
        # transcript observation and the asset; the decision names ranges, never a command
        info = Service.explain_decision(d, trims[0]["subject"])[0]
        self.assertTrue({"policy", "approval", "intent", "requirement", "risk"} <= {b["kind"] for b in info["basis"]})
        kinds = {r["kind"] for r in info["evidence"]}
        self.assertTrue({"inference", "event", "observation", "asset", "context"} <= kinds, kinds)
        self.assertTrue(any(r["kind"] == "observation" and r["source"].startswith("ffmpeg-skill/silence@") for r in info["evidence"]))
        if any(r["kind"] == "inference" and "overlaps recognised speech" in (r.get("detail") or "") for r in info["evidence"]):
            self.assertTrue(any(r["kind"] == "observation" and r["detail"] == "transcript" for r in info["evidence"]), "a disputed trim cites the transcript observation through the conflict")
            self.assertTrue(any("recognised speech overlaps" in n for n in trims[0]["basis"]["approval"]["notes"]) or trims[0]["basis"]["approval"]["provenance"] == "PROFILE")
        self.assertEqual([s["skill"] for s in info["plan"]["steps"]], ["silence_cleanup"])
        blob = json.dumps(info)
        for bad in ("argv", "ffmpeg -", "subprocess", self.src):
            self.assertNotIn(bad, blob)
        # CLI on the recorded IR
        p = str(Path(self.tmp) / "p.json"); save_ir(ir, p)
        env = dict(os.environ, VIDEO_AGENT_WORKSPACE=ws)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "explain", p, "--decision", trims[0]["id"]], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        for frag in ("[REMOVE]", "basis:", "policy", "approval", "evidence:", "AudioEvent/silence", "plan:", "boundary"):
            self.assertIn(frag, r.stdout, frag)
        # approve → the plan executes (nothing in the basis changes what is compiled); the cut receives ranges only. The generic
        # profile is used for the render: the conference presets are video exports and this fixture is audio-only.
        ir_g = svc.plan([self.src], "generic", kinds=["transcript"], params={"language": "ja"})
        self.assertEqual(svc.validate(ir_g).errors, [])
        pg = str(Path(self.tmp) / "g.json"); save_ir(ir_g, pg)
        svc.approve(load_ir(pg), pg, ["all"])
        out = svc.render(load_ir(pg), pg)
        self.assertIn(out["status"], ("COMPLETED", "REVIEW"), out.get("execution"))   # REVIEW = QA judgement on the result, not an execution failure
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        cut = next(op for op in out["execution"]["results"] if op["tool"] == "ffmpeg-skill/cut")
        self.assertTrue(cut["ok"])
        out2 = svc.render(load_ir(pg), pg, resume=out["job"]["id"])
        self.assertEqual(out2["execution"]["status"], "COMPLETED"); self.assertTrue(out2["execution"]["reused"])


class MultiSourceSyncRealTests(unittest.TestCase):
    """ADR-035 on real media through the real ffmpeg-skill CLI: camA.mp4 (speech at 1.0 s) and recorder.wav (the same audio with
    1.25 s more pre-roll) → analyze --kind sync → ffmpeg-skill/sync measures the offset → OBSERVED Observation → TimelineMap →
    Project IR save / load. Offset sign is the tool's: the recorder started 1.25 s EARLIER than the camera, so its offset is −1.25."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = os.path.realpath(tempfile.mkdtemp(prefix="va_sync_"))
        cls.ws = str(Path(cls.tmp) / "ws")
        from video_agent.tools.transcription.locate import locate_transcription
        fixture = Path(locate_transcription().root or "") / "tests" / "fixtures" / "ja_short.wav"
        if not fixture.is_file():
            raise unittest.SkipTest("transcription-skill fixture ja_short.wav not found (speech is needed for a non-periodic correlation)")
        src = Path(cls.tmp) / "src"
        src.mkdir()
        cls.cam = str(src / "camA.mp4")
        cls.rec = str(src / "recorder.wav")
        # the PR #22 speech fixture: speech from 1.0 s, audio padded to 11.5 s plus a 0.5 s tone tail (no trailing silence past the picture)
        fc = "[1:a]adelay=1000|1000,apad=whole_dur=11.5[sp];sine=frequency=440:duration=0.5:sample_rate=48000,volume=0.1,aformat=channel_layouts=stereo[t];[sp][t]concat=n=2:v=0:a=1[a]"
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-i", str(fixture), "-filter_complex", fc,
                        "-map", "0:v", "-map", "[a]", "-t", "12", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", cls.cam], check=True)
        # the recorder started 1.25 s earlier: the same audio with 1.25 s more pre-roll
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", cls.cam, "-filter_complex", "[0:a]adelay=1250|1250[a]", "-map", "[a]", cls.rec], check=True)

    def test_sync_measurement_to_timeline_map_and_ir(self):
        svc = Service(workspace=self.ws)
        self.assertEqual(svc.tools_for(svc.adapter([])).get("sync_analysis"), "ffmpeg-skill/sync")
        _, _, an = svc.analyze([self.cam, self.rec], "generic", kinds=["sync"], hash_sources=False)
        ids = {a.path: a.id for a in an.assets}
        sync = [o for o in an.observations if o.kind == "sync"]
        self.assertEqual(len(sync), 1, an.warnings)
        o = sync[0]
        self.assertEqual((o.asset_id, o.data["reference_asset_id"], o.data["target_asset_id"], o.provenance), (ids[self.rec], ids[self.cam], ids[self.rec], "OBSERVED"))
        self.assertEqual(o.source.split("@")[0], "ffmpeg-skill/sync")
        self.assertTrue(o.source.split("@")[1].startswith("0.9"), o.source)
        self.assertAlmostEqual(o.data["offset_seconds"], -1.25, delta=0.05, msg=o.data)
        self.assertGreaterEqual(o.data["confidence"], 0.3, o.data)
        self.assertIn("earlier", o.data["meaning"])
        self.assertTrue(o.data["applied_to_timeline"])
        tm = an.timeline.timelines[f"asset:{ids[self.rec]}"]
        self.assertAlmostEqual(tm.offset_seconds, -1.25, delta=0.05)
        self.assertEqual(tm.drift_ratio, 1.0)
        self.assertAlmostEqual(tm.to_master(2.25), 1.0, delta=0.05, msg="the speech at 2.25 s of the recorder is at 1.0 s on the camera (master) clock")
        self.assertEqual(an.timeline.timelines[f"asset:{ids[self.cam]}"].offset_seconds, 0.0)
        call = next(c for c in an.tool_calls if c["kind"] == "sync")
        self.assertTrue(call["ok"] and call["tool"] == "ffmpeg-skill/sync")
        # reversed order: the camera is measured against the recorder, the sign flips
        _, _, rev = svc.analyze([self.rec, self.cam], "generic", kinds=["sync"], hash_sources=False)
        ro = next(x for x in rev.observations if x.kind == "sync")
        self.assertAlmostEqual(ro.data["offset_seconds"], 1.25, delta=0.05)
        self.assertEqual(ro.asset_id, {a.path: a.id for a in rev.assets}[self.cam])
        # Project IR: the observation and the timeline map survive save → load → validate; no decision / event / inference cites it
        ir = svc.plan([self.cam, self.rec], "generic", kinds=["sync"], hash_sources=False)
        p = str(Path(self.ws) / "sync.json")
        save_ir(ir, p)
        loaded = load_ir(p)
        self.assertEqual(svc.validate(loaded).errors, [])
        got = next(x for x in loaded.doc["analysis"]["observations"] if x["kind"] == "sync")
        self.assertAlmostEqual(got["data"]["offset_seconds"], -1.25, delta=0.05)
        self.assertAlmostEqual(loaded.doc["timeline"]["timelines"][f"asset:{got['asset_id']}"]["offset_seconds"], -1.25, delta=0.05)
        self.assertFalse([d for d in loaded.doc["decisions"] if got["id"] in d["evidence"]])
        self.assertFalse([e for e in loaded.doc["timeline"]["events"] if got["id"] in (e.get("evidence") or [])])
        plain = svc.plan([self.cam, self.rec], "generic", hash_sources=False)
        self.assertEqual([s["skill"] for s in ir.doc["plan"]["steps"]], [s["skill"] for s in plain.doc["plan"]["steps"]], "the measurement changes no operation")
        self.assertEqual([(op["type"], op.get("keep")) for op in ir.doc["video"]["operations"]], [(op["type"], op.get("keep")) for op in plain.doc["video"]["operations"]])
        # CLI through the same boundary
        out = subprocess.run([sys.executable, "-m", "video_agent.cli", "--workspace", self.ws, "--json", "analyze", self.cam, self.rec, "--kind", "sync", "--no-hash"],
                             capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src")))
        self.assertEqual(out.returncode, 0, out.stderr)
        doc = json.loads(out.stdout)
        self.assertAlmostEqual([x["data"]["offset_seconds"] for x in doc["observations"] if x["kind"] == "sync"][0], -1.25, delta=0.05)


class VideoEditingRealTests(unittest.TestCase):
    """PR #18 (ADR-028) on the real video-editing-skill and ffmpeg-skill 0.9.x: contract discovery and drift against the pinned
    contract, the Skill's doctor as the capability source, video.trim lowered to video-editing/cut, execution through the CLI
    boundary with the agent's PathPolicy, the response's sha256 / timeline / OBSERVED probe carried into the agent's results
    and provenance, and the agent's own idempotent resume. Runs only when a video-editing-skill checkout is installed."""

    @classmethod
    def setUpClass(cls):
        from video_agent.tools.video_editing import locate_video_editing
        cls.ve = locate_video_editing()
        cls.tmp = tempfile.mkdtemp(prefix="va_ve_")
        src_dir = Path(cls.tmp) / "src"
        src_dir.mkdir()
        cls.src = str(src_dir / "talk.mp4")
        if cls.ve and shutil.which("ffmpeg"):
            make_media(cls.src)

    def _service(self, workspace: str) -> Service:
        return Service(workspace=workspace)

    def test_contract_doctor_capability_and_drift(self):
        if not self.ve:
            self.skipTest("needs a video-editing-skill checkout (VIDEO_AGENT_VIDEO_EDITING_DIR)")
        from video_agent.capabilities import CapabilityResolver
        from video_agent.tools.video_editing import VideoEditingAdapter, contract_drift, pinned_contract
        ad = VideoEditingAdapter(self.ve, workspace=self.tmp, allowed_inputs=[str(Path(self.src).parent)], ffmpeg_skill_dir=str(locate_ffmpeg_skill().root))
        self.assertEqual(ad.version, pinned_contract()["version"])
        self.assertEqual(contract_drift(ad.contract), [], "the installed video-editing-skill contract drifted from the pinned one: re-verify the adapter")
        doc = ad.doctor()
        self.assertTrue(doc["ok"], doc.get("problems"))
        rows = {c["check"]: c for c in doc["checks"]}
        self.assertEqual(rows["ffmpeg-skill"]["status"], "AVAILABLE"); self.assertTrue(rows["ffmpeg-skill"]["version_supported"])
        self.assertEqual(rows["path_policy"]["status"], "AVAILABLE"); self.assertIn(str(Path(self.src).parent.resolve()), rows["path_policy"]["allowed_input_roots"])
        self.assertNotRegex(json.dumps(doc).replace("secrets_shown", ""), r"(?i)(api[_-]?key|token|secret|password)")
        caps = CapabilityResolver(str(locate_ffmpeg_skill().root), video_editing_dir=str(self.ve.root) if self.ve.root else None).resolve()
        self.assertEqual(caps["video-editing"].status, "AVAILABLE", caps["video-editing"].detail)
        self.assertEqual(caps["video-editing"].evidence["engine"]["id"], "ffmpeg-skill")
        self.assertEqual(caps["video-editing"].evidence["drift"], [])
        self.assertEqual(caps["encoder:aac"].status, "AVAILABLE"); self.assertIn(caps["filter:xfade"].status, ("AVAILABLE", "MISSING"))
        svc = self._service(str(Path(self.tmp) / "ws_cap"))
        pk = {r["skill_id"]: r for r in svc.packages()}["video-editing"]
        self.assertTrue(pk["implemented"] and pk["available"], pk["reason"]); self.assertEqual(pk["version"], ad.version)
        self.assertIn("video-editing/cut", pk["usable_tools"])
        self.assertEqual(svc.tools_for().get("silence_cleanup"), "ffmpeg-skill/cut", "the reference cut stays first while both packages are available")

    def test_trim_through_video_editing_end_to_end(self):
        if not self.ve or not shutil.which("ffmpeg"):
            self.skipTest("needs a video-editing-skill checkout and ffmpeg")
        ws = str(Path(self.tmp) / "ws")
        svc = self._service(ws)
        svc.registry.get("silence_cleanup").tools = ["video-editing/cut"]
        ir = svc.plan([self.src], "generic")
        d = ir.doc
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "silence_cleanup")
        self.assertEqual(step["tool"], "video-editing/cut")
        self.assertEqual(svc.validate(ir).errors, [])
        p = str(Path(self.tmp) / "p.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertIn(out["status"], ("COMPLETED", "REVIEW"), out.get("execution"))
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        cut = next(r for r in out["execution"]["results"] if r["tool"] == "video-editing/cut")
        self.assertTrue(cut["ok"]); self.assertTrue(os.path.isfile(cut["output"]))
        import hashlib
        self.assertEqual(cut["data"]["artifact"]["sha256"], hashlib.sha256(Path(cut["output"]).read_bytes()).hexdigest())
        self.assertEqual(cut["data"]["operation"]["tool"], "ffmpeg-skill/cut"); self.assertEqual(cut["data"]["operation"]["skill"], "video-editing")
        self.assertTrue(cut["data"]["observation"]["source"].startswith("ffmpeg-skill/probe@0.9"))
        self.assertEqual(cut["data"]["observation"]["provenance"], "OBSERVED")
        self.assertTrue(cut["data"]["timeline"]["tracks"][0]["segments"], "source → timeline mapping reported by the Skill")
        self.assertTrue(cut["commands"] and all(c.startswith("/") or c.startswith("ffmpeg") for c in cut["commands"]), "ffmpeg command lines are provenance only")
        keep = step["params"]["keep"]
        probe = cut["data"]["observation"]["data"]
        self.assertAlmostEqual(float(probe["duration"]), sum(e - s for s, e in keep), delta=1.6, msg="keyframe-precision cut lands within the Skill's tolerance")
        prov = json.loads((Path(ws) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        trim = next(e for e in prov["operations"] if e["skill"] == "silence_cleanup")
        self.assertEqual((trim["skill_package"], trim["tool"], trim["tool_version"]), ("video-editing", "video-editing/cut", "0.1.0"))
        self.assertEqual(trim["skill_result"]["artifact"]["sha256"], cut["data"]["artifact"]["sha256"])
        self.assertEqual(sorted(trim["args"]), ["input", "keep", "output", "precision"])
        # the deliverable of the generic profile is the last intermediate: its hash equals the one the Skill reported for its output chain
        # (loudness runs after the cut, so the artifact differs; the cut's own file is the one the Skill hashed)
        self.assertEqual(hashlib.sha256(Path(cut["output"]).read_bytes()).hexdigest(), trim["skill_result"]["artifact"]["sha256"])
        # resume: the completed cut is reused by the agent's idempotency key (skipped, never re-run through the Skill)
        out2 = svc.render(load_ir(p), p, resume=out["job"]["id"])
        self.assertEqual(out2["execution"]["status"], "COMPLETED"); self.assertTrue(out2["execution"]["reused"])
        self.assertIn(cut["op_id"], out2["execution"]["skipped"]); self.assertEqual(out2["execution"]["reused"][cut["op_id"]], cut["output"])
        # an input outside the allowed roots is refused before the Skill runs; the Skill refuses it too (same roots)
        from video_agent.models import Operation
        from video_agent.tools.video_editing import VideoEditingAdapter
        outside_dir = Path(tempfile.mkdtemp(prefix="va_out_")); outside = str(outside_dir / "o.mp4"); shutil.copy(self.src, outside)
        strict = VideoEditingAdapter(self.ve, workspace=ws, allowed_inputs=[str(Path(self.src).parent)], ffmpeg_skill_dir=str(locate_ffmpeg_skill().root))
        op = Operation(tool="video-editing/cut", args={"input": "x", "keep": [[0.0, 1.0]], "precision": "keyframe", "output": "y"}, inputs=["x"], outputs=["y"])
        target = str(Path(ws) / "manual" / "y.mp4"); os.makedirs(os.path.dirname(target), exist_ok=True)
        r = strict.run(op, {"x": outside, "y": target})
        self.assertFalse(r.ok); self.assertIn("outside the allowed input roots", r.data["error"]["message"])
        loose = VideoEditingAdapter(self.ve, workspace=None, allowed_inputs=[], ffmpeg_skill_dir=str(locate_ffmpeg_skill().root))   # no agent roots: the Skill's own policy (roots = --workspace) still refuses
        r = loose.run(op, {"x": outside, "y": target})
        self.assertFalse(r.ok); self.assertEqual(r.data["error"]["code"], "PATH_NOT_ALLOWED"); self.assertFalse(os.path.exists(target))
        # an unsupported operation (CROP) is the Skill's UNSUPPORTED_OPERATION / not a tool: no engine fallback anywhere
        self.assertFalse(strict.supports("video-editing/crop"))
        self.assertTrue(any(u["type"] == "CROP" for u in strict.contract["unsupported"]))


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill(), "needs ffmpeg and ffmpeg-skill")
class VideoEditingOperationsRealTests(unittest.TestCase):
    """ADR-029 on the real video-editing-skill (PR #1 branch) and ffmpeg-skill 0.9.x: two real inputs A + B → trim (ffmpeg-skill/cut)
    → video.concat (with a transition) → video.speed → video.resize → video.fit / video.fill → video.overlay (a real PNG) → the
    generic delivery; output validation by QA against durations derived from the IR, ffprobe on every intermediate, artifact /
    provenance chain, the Skill's OBSERVED probes in provenance, and idempotent resume. Runs only with a video-editing-skill checkout."""

    @classmethod
    def setUpClass(cls):
        from video_agent.tools.video_editing import locate_video_editing
        cls.ve = locate_video_editing()
        cls.tmp = tempfile.mkdtemp(prefix="va_vo_")
        src = Path(cls.tmp) / "src"; src.mkdir()
        cls.a, cls.b, cls.png = str(src / "a.mp4"), str(src / "b.mp4"), str(src / "logo.png")
        if cls.ve:
            make_media(cls.a); make_media(cls.b)
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=160x90:d=1", "-frames:v", "1", cls.png], check=True)

    def _probe(self, path):
        pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height", "-of", "json", path], capture_output=True, text=True, check=True)
        doc = json.loads(pr.stdout)
        v = next((s for s in doc["streams"] if s.get("codec_type") == "video"), {})
        return float(doc["format"]["duration"]), (v.get("width"), v.get("height")), any(s.get("codec_type") == "audio" for s in doc["streams"])

    def _run(self, name, reqs, expect_ops, expect_size):
        ws = str(Path(self.tmp) / name)
        svc = Service(workspace=ws)
        ir = svc.plan([self.a, self.b], "generic", user_requirements=reqs)
        d = ir.doc
        self.assertEqual(d["plan"]["status"], "APPROVED", d["plan"]["summary"])
        self.assertEqual(svc.validate(ir).errors, [])
        ops = [op for op in d["video"]["operations"] if op["type"] != "video.trim"]
        self.assertEqual([op["type"] for op in ops], expect_ops)
        self.assertTrue(all(s["tool"] == f"video-editing/{s['skill'].split('_', 1)[1]}" for s in d["plan"]["steps"] if s["skill"].startswith("video_")))
        p = str(Path(self.tmp) / f"{name}.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        self.assertIn(out["status"], ("COMPLETED", "REVIEW"))
        res = [r for r in out["execution"]["results"] if r["tool"].startswith("video-editing/")]
        self.assertEqual([r["tool"] for r in res], [f"video-editing/{t.split('.', 1)[1]}" for t in expect_ops])
        import hashlib
        concat = ops[0]
        expected_programme = concat["timeline_duration"]
        for r, op in zip(res, ops):
            self.assertTrue(r["ok"] and os.path.isfile(r["output"]), r)
            self.assertEqual(r["data"]["artifact"]["sha256"], hashlib.sha256(Path(r["output"]).read_bytes()).hexdigest())
            self.assertEqual(r["data"]["observation"]["provenance"], "OBSERVED"); self.assertTrue(r["data"]["observation"]["source"].startswith("ffmpeg-skill/probe@0.9"))
            self.assertEqual(r["data"]["operation"]["type"], op["type"].split(".", 1)[1].upper())
            self.assertFalse(any(k in json.dumps(r["data"]["operation"].get("parameters") or {}).lower() for k in ("argv", "command", "filter", "shell")))
        # ffprobe facts on every intermediate: duration follows the IR (concat timeline → speed), geometry follows resize / fit / fill
        dur_c, size_c, audio_c = self._probe(res[0]["output"])
        self.assertAlmostEqual(dur_c, expected_programme, delta=1.0, msg="concat duration = the IR's programme timeline (keyframe-precision trims add up to ~0.1 s per clip)")
        self.assertEqual(size_c, (1280, 720)); self.assertTrue(audio_c)
        factor = next(op["factor"] for op in ops if op["type"] == "video.speed")
        dur_s, _, _ = self._probe(res[1]["output"])
        self.assertAlmostEqual(dur_s, dur_c / factor, delta=0.5)
        _, size_r, _ = self._probe(res[2]["output"]); self.assertEqual(size_r, (640, 360))
        _, size_f, _ = self._probe(res[3]["output"]); self.assertEqual(size_f, expect_size)
        dur_o, size_o, audio_o = self._probe(res[4]["output"]); self.assertEqual(size_o, expect_size); self.assertTrue(audio_o)
        self.assertAlmostEqual(dur_o, dur_s, delta=0.3)
        # QA: the delivered programme against the expectation derived from the IR (trim → concat → speed)
        subject = next(i for i in out["qa"]["items"] if i["name"] == "duration")
        self.assertEqual(subject["status"], "PASS", subject); self.assertTrue(subject["artifact"].startswith("programme_delivery_"))
        self.assertEqual(out["paths"]["programme_delivery_main"], res[4]["output"], "generic profile: the last intermediate is the deliverable")
        # provenance: every editing operation with its decision, tool, Skill result and input / output paths; the image is an input of the overlay
        prov = json.loads((Path(ws) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        rows = [e for e in prov["operations"] if e["tool"].startswith("video-editing/")]
        self.assertEqual([e["skill"] for e in rows], [f"video_{t.split('.', 1)[1]}" for t in expect_ops])
        self.assertEqual(len(rows[0]["input"]), 2); self.assertIn(os.path.abspath(self.png), rows[4]["input"])
        for e in rows:
            self.assertTrue(e["decision"] and e["skill_result"]["artifact"]["sha256"] and e["tool_version"] == "0.1.0")
        self.assertEqual(len(prov["skill_observations"]), len(rows))
        dec = {x["id"]: x for x in d["decisions"]}
        self.assertTrue(all(dec[i]["type"] == "TRANSFORM" and dec[i]["provenance"] == "USER" for e in rows for i in e["decision"]))
        return svc, p, out, res

    def test_concat_speed_resize_fit_overlay_end_to_end(self):
        if not self.ve:
            self.skipTest("needs a video-editing-skill checkout (VIDEO_AGENT_VIDEO_EDITING_DIR)")
        svc, p, out, res = self._run("fit", {"edit.concat": True, "edit.speed": 2, "edit.resize": 640, "edit.fit": "1:1", "edit.overlay": self.png, "edit.overlay.position": "top-right", "edit.overlay.scale": 80},
                                     ["video.concat", "video.speed", "video.resize", "video.fit", "video.overlay"], (640, 640))
        # resume: every completed editing operation is reused by the agent's idempotency key (no Skill call)
        out2 = svc.render(load_ir(p), p, resume=out["job"]["id"])
        self.assertEqual(out2["execution"]["status"], "COMPLETED")
        self.assertTrue(all(r["op_id"] in out2["execution"]["skipped"] for r in res))

    def test_concat_transition_fill_end_to_end(self):
        if not self.ve:
            self.skipTest("needs a video-editing-skill checkout (VIDEO_AGENT_VIDEO_EDITING_DIR)")
        self._run("fill", {"edit.concat": True, "edit.concat.transition": "fade", "edit.concat.transition_duration": 0.5, "edit.speed": 2, "edit.resize": 640, "edit.fill": "1:1",
                           "edit.overlay": self.png, "edit.overlay.position": {"x": 10, "y": 10}, "edit.overlay.opacity": 0.6},
                  ["video.concat", "video.speed", "video.resize", "video.fill", "video.overlay"], (640, 640))


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill(), "needs ffmpeg and ffmpeg-skill")
class AudioProductionRealTests(unittest.TestCase):
    """ADR-030 on the real audio-production-skill (main, 0.1.0) and ffmpeg-skill 0.9.x: contract / doctor / drift against the pinned
    contract, the capability verdicts, and the vertical slice on real media — analysis (probe / silence / loudness) → decisions →
    ProductionPlan → IR audio operations → compiler → audio-production-skill → execution → the Skill's OBSERVED probe and loudness
    re-measurement → QA on the audio deliverable → artifact / provenance → resume. Cases: (A) audio-only stereo WAV with silence:
    cut → gain → mono → fade out; (B) a video+audio container: the audio track is delivered (cut → fade in), no picture; (C) two
    audio inputs: cut → concat with crossfade → mono → normalise (re-measured); (D) refusals. Runs only with a checkout installed."""

    @classmethod
    def setUpClass(cls):
        from video_agent.tools.audio_production import locate_audio_production
        cls.ap = locate_audio_production()
        cls.tmp = tempfile.mkdtemp(prefix="va_ap_")
        src = Path(cls.tmp) / "src"; src.mkdir()
        cls.wav, cls.mono, cls.mp4 = str(src / "a.wav"), str(src / "b.wav"), str(src / "v.mp4")
        if cls.ap:
            gated = "0.1*sin(2*PI*1000*t)*between(t\\,3\\,13)"
            ff = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
            subprocess.run(ff + ["-f", "lavfi", "-i", f"aevalsrc='{gated}|{gated}':s=48000:c=stereo", "-t", "16", "-c:a", "pcm_s16le", cls.wav], check=True)
            subprocess.run(ff + ["-f", "lavfi", "-i", "aevalsrc='0.2*sin(2*PI*440*t)':s=48000:c=mono", "-t", "5", "-c:a", "pcm_s16le", cls.mono], check=True)
            make_media(cls.mp4)

    def _probe(self, path):
        pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,channels,sample_rate", "-of", "json", path], capture_output=True, text=True, check=True)
        doc = json.loads(pr.stdout)
        a = next((s for s in doc["streams"] if s.get("codec_type") == "audio"), {})
        return float(doc["format"]["duration"]), int(a.get("channels") or 0), int(a.get("sample_rate") or 0), any(s.get("codec_type") == "video" for s in doc["streams"])

    def _run(self, name, inputs, reqs, expect_skills, expect_ops, expect_plan_status="APPROVED", pending_subjects=()):
        """plan → (status as expected before any approval) → render(approve=all) → COMPLETED. `expect_plan_status` "REVIEW" means the
        plan waits for the CONFIRM decisions in `pending_subjects` (ADR-033: a video container's audio.extract); the explicit
        `approve=["all"]` of render is the existing approval mechanism, after which the saved plan must be APPROVED."""
        ws = str(Path(self.tmp) / name)
        svc = Service(workspace=ws)
        ir = svc.plan(inputs, "generic", user_requirements=reqs)
        d = ir.doc
        self.assertEqual(d["plan"]["status"], expect_plan_status, d["plan"]["summary"]); self.assertEqual(svc.validate(ir).errors, [])
        self.assertEqual(sorted(x["subject"] for x in ir.pending_confirmations()), sorted(pending_subjects), "exactly these CONFIRM decisions are pending before approval")
        self.assertEqual([s["skill"] for s in d["plan"]["steps"]], expect_skills)
        self.assertTrue(all(s["tool"] == "audio-production/run" for s in d["plan"]["steps"]))
        self.assertEqual(d["video"]["operations"], [])
        p = str(Path(self.tmp) / f"{name}.json"); save_ir(ir, p)
        if expect_plan_status == "REVIEW":
            # without the explicit approval nothing runs: the render waits, no Skill result exists
            waiting = svc.render(load_ir(p), p)
            self.assertEqual(waiting["status"], "WAITING_FOR_APPROVAL", waiting)
            self.assertEqual(sorted(x["subject"] for x in waiting["pending"]), sorted(pending_subjects))
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"]); self.assertIn(out["status"], ("COMPLETED", "REVIEW"))
        after = load_ir(p).doc
        self.assertEqual(after["plan"]["status"], "APPROVED", "after the explicit approval the plan version is APPROVED")
        self.assertEqual([x["subject"] for x in after["decisions"] if x["approval"] == "CONFIRM" and x["status"] == "PROPOSED"], [], "no CONFIRM decision is left pending")
        res = [r for r in out["execution"]["results"] if r["tool"] == "audio-production/run"]
        self.assertEqual([r["data"]["operation_type"] for r in res], expect_ops)
        import hashlib
        for r in res:
            self.assertTrue(r["ok"] and os.path.isfile(r["output"]) and r["output"].endswith(".wav"), r)
            self.assertEqual(r["data"]["artifact"]["sha256"], hashlib.sha256(Path(r["output"]).read_bytes()).hexdigest())
            self.assertEqual(r["data"]["observation"]["provenance"], "OBSERVED"); self.assertTrue(r["data"]["observation"]["source"].startswith("ffmpeg-skill/probe@0.9"))
            self.assertTrue(r["data"]["operation"]["tool"].startswith("ffmpeg-skill/")); self.assertEqual(r["data"]["provenance"]["skill"], "audio-production")
            self.assertFalse(any(k in json.dumps(r["data"]["operation"]["parameters"]).lower() for k in ("argv", "command", "filter", "shell")))
        items = {i["name"]: i for i in out["qa"]["items"]}
        self.assertEqual(items["duration"]["status"], "PASS", items["duration"]); self.assertEqual(items["audio_only"]["status"], "PASS")
        prov = json.loads((Path(ws) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        rows = [e for e in prov["operations"] if e["tool"] == "audio-production/run"]
        self.assertEqual([e["skill"] for e in rows], expect_skills)
        for e in rows:
            self.assertTrue(e["decision"] and e["skill_result"]["artifact"]["sha256"] and e["tool_version"] == "0.1.0" and e["skill_package"] == "audio-production")
        dec = {x["id"]: x for x in d["decisions"]}
        self.assertTrue(all(dec[i]["type"] in ("TRANSFORM", "REMOVE") for e in rows for i in e["decision"]))
        return svc, p, out, res, prov

    def test_contract_doctor_capabilities_and_drift(self):
        if not self.ap:
            self.skipTest("needs an audio-production-skill checkout (VIDEO_AGENT_AUDIO_PRODUCTION_DIR)")
        from video_agent.capabilities import CapabilityResolver
        from video_agent.tools.audio_production import AudioProductionAdapter, contract_drift, pinned_contract
        ad = AudioProductionAdapter(self.ap, workspace=self.tmp, allowed_inputs=[str(Path(self.wav).parent)], ffmpeg_skill_dir=str(locate_ffmpeg_skill().root))
        self.assertEqual(ad.version, pinned_contract()["version"])
        self.assertEqual(contract_drift(ad.contract), [], "the installed audio-production-skill contract drifted from the pinned one: re-verify the adapter")
        doc = ad.doctor()
        self.assertIn(doc["status"], ("ok", "degraded"), doc.get("problems"))
        self.assertEqual(doc["checks"]["ffmpeg_skill"]["status"], "ok", doc["checks"]["ffmpeg_skill"])
        ops = ad.operation_status(doc)
        self.assertEqual(ops["NORMALIZE"], "supported"); self.assertEqual(ops["CUT"], "supported")
        self.assertNotRegex(json.dumps(doc).replace("secrets_shown", ""), r"(?i)(api[_-]?key|token|secret|password)")
        caps = CapabilityResolver(str(locate_ffmpeg_skill().root), audio_production_dir=str(self.ap.root) if self.ap.root else None).resolve()
        self.assertEqual(caps["audio-production"].status, "AVAILABLE", caps["audio-production"].detail)
        self.assertEqual(caps["audio-production"].evidence["drift"], []); self.assertEqual(caps["audio-production"].evidence["engine"]["id"], "ffmpeg-skill")
        for t in ("CUT", "NORMALIZE", "CONCAT", "GAIN", "MONO", "FADE_IN"):
            self.assertEqual(caps[f"audio-production:{t}"].status, "AVAILABLE", (t, caps[f"audio-production:{t}"].detail))
        svc = Service(workspace=str(Path(self.tmp) / "ws_cap"))
        pk = {r["skill_id"]: r for r in svc.packages()}["audio-production"]
        self.assertTrue(pk["implemented"] and pk["available"], pk["reason"]); self.assertEqual(pk["version"], ad.version)
        tools = svc.tools_for()
        self.assertEqual({tools.get(s) for s in ("audio_cut", "audio_normalize", "audio_gain", "audio_mono", "audio_stereo", "audio_downmix", "audio_fade_in", "audio_fade_out", "audio_concat")}, {"audio-production/run"})
        self.assertEqual(tools.get("silence_cleanup"), "ffmpeg-skill/cut", "the reference paths are untouched")
        self.assertEqual(tools.get("loudness_normalization"), "ffmpeg-skill/loudness")

    def test_audio_only_cut_gain_mono_fade_end_to_end(self):
        if not self.ap:
            self.skipTest("needs an audio-production-skill checkout")
        svc, p, out, res, prov = self._run("A", [self.wav], {"audio.production": True, "audio.gain": -3, "audio.channels": "mono", "audio.fade_out": 1.0},
                                           ["audio_cut", "audio_gain", "audio_mono", "audio_fade_out"], ["CUT", "GAIN", "MONO", "FADE_OUT"])
        dur, ch, sr, has_v = self._probe(res[-1]["output"])
        self.assertAlmostEqual(dur, 10.3, delta=0.1, msg="16 s source minus the leading / trailing silence (with margins) = 10.3 s"); self.assertEqual((ch, sr, has_v), (1, 48000, False))
        self.assertEqual(self._probe(res[0]["output"])[1], 2, "the cut keeps the stereo layout; MONO changes it")
        aid = list(svc.validate(load_ir(p)).errors or []) and None or list(load_ir(p).doc["assets"])[0]
        self.assertEqual(out["paths"][f"{aid}_delivery_main"], res[-1]["output"], "generic profile: the last audio intermediate is the deliverable")
        self.assertEqual([o["kind"] for o in prov["skill_observations"]], ["media.probe"] * 4)
        # resume reuses every completed audio operation (no Skill call)
        out2 = svc.render(load_ir(p), p, resume=out["job"]["id"])
        self.assertEqual(out2["execution"]["status"], "COMPLETED"); self.assertTrue(all(r["op_id"] in out2["execution"]["skipped"] for r in res))

    def test_video_container_delivers_audio_only(self):
        if not self.ap:
            self.skipTest("needs an audio-production-skill checkout")
        # ADR-033: before approval the plan is REVIEW with exactly the audio.extract CONFIRM pending; render(approve=all) approves it explicitly
        svc, p, out, res, prov = self._run("B", [self.mp4], {"audio.production": True, "audio.fade_in": 0.5}, ["audio_cut", "audio_fade_in"], ["CUT", "FADE_IN"],
                                           expect_plan_status="REVIEW", pending_subjects=("audio.extract",))
        dur, ch, sr, has_v = self._probe(res[-1]["output"])
        self.assertFalse(has_v, "the picture is not delivered on the audio path"); self.assertAlmostEqual(dur, 11.0, delta=0.2)
        dec = {x["subject"]: x for x in load_ir(p).doc["decisions"]}
        self.assertEqual(dec["audio.extract"]["type"], "TRANSFORM"); self.assertEqual(dec["audio.extract"]["basis"]["approval"]["key"], "audio.extract.approval")
        # ADR-033: the generic switch did not waive the extraction; it ran because `_run` approved it explicitly (recorded review)
        self.assertEqual((dec["audio.extract"]["approval"], dec["audio.extract"]["status"]), ("CONFIRM", "APPROVED"))
        self.assertEqual(load_ir(p).doc["execution"]["reviews"][dec["audio.extract"]["id"]]["action"], "APPROVED")
        self.assertTrue(all(dec["audio.extract"]["id"] in op["decision_ids"] for op in load_ir(p).doc["audio"]["operations"]))
        # the dedicated requirement waives it up front: AUTO, plan APPROVED before any approval call
        ir2 = svc.plan([self.mp4], "generic", user_requirements={"audio.production": True, "audio.fade_in": 0.5, "audio.extract": True})
        dec2 = {x["subject"]: x for x in ir2.doc["decisions"]}
        self.assertEqual((dec2["audio.extract"]["approval"], ir2.doc["plan"]["status"]), ("AUTO", "APPROVED"))

    def test_two_inputs_concat_mono_normalize_end_to_end(self):
        if not self.ap:
            self.skipTest("needs an audio-production-skill checkout")
        svc, p, out, res, prov = self._run("C", [self.wav, self.mono], {"audio.production": True, "audio.concat": True, "audio.concat.crossfade": 0.5, "audio.channels": "mono",
                                                                       "audio.normalize": True, "audio.loudness.target_lufs": -16},
                                           ["audio_cut", "audio_concat", "audio_mono", "audio_normalize"], ["CUT", "CONCAT", "MONO", "NORMALIZE"])
        d = load_ir(p).doc
        cat = next(op for op in d["audio"]["operations"] if op["type"] == "audio.concat")
        self.assertAlmostEqual(cat["timeline_duration"], 10.3 + 5.0 - 0.5, places=2); self.assertEqual([s["input"] for s in cat["segments"]], list(d["assets"]))
        dur, ch, sr, has_v = self._probe(res[-1]["output"])
        self.assertAlmostEqual(dur, 14.8, delta=0.15); self.assertEqual((ch, has_v), (1, False))
        norm = res[-1]["data"]
        self.assertAlmostEqual(norm["measurement"]["data"]["integrated_lufs"], -16.0, delta=2.0, msg="the Skill re-measured its NORMALIZE output within the tolerance the decision carried")
        self.assertEqual(norm["operation"]["parameters"]["tolerance_lufs"], 2.0)
        items = {i["name"]: i for i in out["qa"]["items"]}
        self.assertEqual(items["loudness"]["status"], "PASS", items["loudness"]); self.assertIn("-16", items["loudness"]["expected"])
        self.assertEqual(prov["skill_observations"][-1]["kind"], "loudness"); self.assertEqual(prov["skill_observations"][-1]["source"].split("@")[0], "ffmpeg-skill/loudness")
        art = out["artifacts"] or []
        self.assertEqual(art, [], "generic profile registers no preset artifact; the deliverable is the last intermediate")
        self.assertTrue(out["paths"]["programme_audio_delivery_main"].endswith("programme_audio_loudnorm.wav"))

    def test_refusals_on_real_skill(self):
        if not self.ap:
            self.skipTest("needs an audio-production-skill checkout")
        svc = Service(workspace=str(Path(self.tmp) / "D"))
        for reqs, subjects in (({"audio.production": True, "edit.speed": 2}, {"audio.production"}), ({"audio.production": True, "audio.sample_rate": 44100}, {"audio.sample_rate"})):
            ir = svc.plan([self.mono], "generic", user_requirements=reqs)
            self.assertTrue(subjects <= {x["subject"] for x in ir.doc["decisions"] if x["approval"] == "BLOCK"}, reqs); self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
            pth = str(Path(self.tmp) / "d.json"); save_ir(ir, pth)
            self.assertEqual(svc.render(load_ir(pth), pth, approve=["all"])["status"], "BLOCKED")
        with self.assertRaises(ValueError):
            svc.plan([self.mono], "generic", user_requirements={"audio.production": True, "audio.gain": 90})
        # the Skill's own refusal through the boundary: a CUT beyond the media is INVALID_TIME_RANGE (not retried)
        from video_agent.models import Operation
        from video_agent.tools.audio_production import AudioProductionAdapter
        ad = AudioProductionAdapter(self.ap, workspace=str(Path(self.tmp) / "D"), allowed_inputs=[str(Path(self.mono).parent)], ffmpeg_skill_dir=str(locate_ffmpeg_skill().root))
        out = str(Path(self.tmp) / "D" / "x" / "o.wav"); os.makedirs(os.path.dirname(out), exist_ok=True)
        r = ad.run(Operation(tool="audio-production/run", args={"operation": "CUT", "input": "m", "remove": [[10.0, 20.0]], "output": "o"}, inputs=["m"], outputs=["o"], id="op_r"), {"m": self.mono, "o": out})
        self.assertFalse(r.ok); self.assertEqual(r.data["error"]["code"], "INVALID_TIME_RANGE"); self.assertFalse(r.data["error"]["retryable"]); self.assertFalse(os.path.exists(out))


class IntegratedPipelineRealTests(unittest.TestCase):
    """ADR-031 / ADR-032 on the real Skills (ffmpeg-skill 0.9.x, video-editing-skill, subtitle-skill, thumbnail-skill, color-grading-skill,
    motion-graphics-skill, qc-skill, transcription-skill with a local faster-whisper model): the ten scenarios of the Phase 3
    specification on real media — analysis → decisions → ProductionPlan → IR finishing sections → compiler → the Skills → QA with
    the QC gate → artifacts / provenance → resume / revision / drift / tamper / determinism. Each test skips with the reason when a
    checkout it needs is not installed (VIDEO_AGENT_*_DIR)."""

    NEEDS = {"video-editing": "VIDEO_AGENT_VIDEO_EDITING_DIR", "subtitle": "VIDEO_AGENT_SUBTITLE_DIR", "thumbnail": "VIDEO_AGENT_THUMBNAIL_DIR",
             "color-grading": "VIDEO_AGENT_COLOR_GRADING_DIR", "motion-graphics": "VIDEO_AGENT_MOTION_GRAPHICS_DIR", "qc": "VIDEO_AGENT_QC_DIR"}

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="va_p3_")
        src = Path(cls.tmp) / "src"; src.mkdir()
        cls.a, cls.b, cls.png = str(src / "a.mp4"), str(src / "b.mp4"), str(src / "logo.png")
        cls.speech = str(src / "ja_talk.mp4")
        cls.ready = shutil.which("ffmpeg") and locate_ffmpeg_skill()
        if cls.ready:
            make_media(cls.a); make_media(cls.b)
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=160x90:d=1", "-frames:v", "1", cls.png], check=True)
            ts = locate_transcription()
            fixture = Path(ts.root or "") / "tests" / "fixtures" / "ja_short.wav" if ts else None
            if fixture and fixture.is_file():
                # recognisable Japanese speech (the recognition Skill's own 9.6 s fixture) under a test picture; the picture stops with the audio
                # 1 s of silence, the 9.6 s of speech, silence to 11.5 s and a quiet tone to the end, under a 12 s picture: a leading trim exists,
                # a recognised segment may end a few ms after the speech (the recogniser rounds) and the container contains it, and the audio does
                # not end in silence (an explicit silence end past the container duration is the known silencedetect behaviour this suite does not paper over)
                fc = "[1:a]adelay=1000|1000,apad=whole_dur=11.5[sp];sine=frequency=440:duration=0.5:sample_rate=48000,volume=0.1,aformat=channel_layouts=stereo[t];[sp][t]concat=n=2:v=0:a=1[a]"
                subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30", "-i", str(fixture), "-filter_complex", fc,
                                "-map", "0:v", "-map", "[a]", "-t", "12", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", cls.speech], check=True)
            else:
                cls.speech = None
        cls.caps = Service(workspace=str(Path(cls.tmp) / "caps")).caps.resolve() if cls.ready else {}

    def _need(self, *skills):
        if not self.ready:
            self.skipTest("needs ffmpeg and an ffmpeg-skill checkout")
        for sk in skills:
            if self.caps.get(sk) is None or self.caps[sk].status != "AVAILABLE":
                self.skipTest(f"needs {sk} ({self.NEEDS.get(sk, sk)}): {getattr(self.caps.get(sk), 'detail', 'not resolved')[:120]}")

    def _speech(self):
        if not self.speech:
            self.skipTest("needs transcription-skill's ja_short.wav fixture")
        cap = self.caps.get("transcription")
        if cap is None or cap.status != "AVAILABLE":
            self.skipTest(f"needs transcription-skill with a local faster-whisper model: {getattr(cap, 'detail', '')[:120]}")
        return self.speech

    def _svc(self, name):
        return Service(workspace=str(Path(self.tmp) / name), offline=True)

    def _plan(self, svc, inputs, reqs, profile="youtube", name="p"):
        ir = svc.plan(inputs, profile, user_requirements=reqs, params={"language": "ja", "offline": True})
        rep = svc.validate(ir)
        self.assertEqual(rep.errors, [], rep.errors)
        p = str(Path(svc.workspace) / f"{name}.json"); save_ir(ir, p)
        return ir, p

    def _render(self, svc, p, **kw):
        out = svc.render(load_ir(p), p, approve=["all"], **kw)
        self.assertEqual(out["execution"]["status"], "COMPLETED", json.dumps(out["execution"].get("recovery"))[:800] + json.dumps([r for r in out["execution"]["results"] if not r["ok"]])[:800])
        return out

    def _gate_coherent(self, out, art):
        """The QC gate agrees with the agent's own QA of the same deliverable: a FAIL on either side keeps the artifact `working`, PASS on
        both makes it READY (`approved`), anything else stays a candidate; the admitted report is about the delivered bytes."""
        agent_fail = any(i["status"] == "FAIL" and i["layer"] != "qc" and i["artifact"] == art["logical_name"] for i in out["qa"]["items"])
        qc = next(r for r in out["execution"]["results"] if r["tool"] == "qc/check" and r["data"].get("kind") in ("delivery", "audio"))
        self.assertTrue(qc["ok"] and qc["data"]["admitted"]); self.assertEqual(qc["data"]["fingerprint"], art["hash"])
        self.assertEqual(art["qa"]["qc"], qc["data"]["verdict"])
        if agent_fail or qc["data"]["verdict"] == "FAIL":
            self.assertEqual((art["stage"], art["qa_status"]), ("working", "FAIL"))
        elif qc["data"]["verdict"] == "PASS":
            self.assertEqual(art["stage"], "approved")
        else:
            self.assertEqual(art["stage"], "candidate")

    def _probe(self, path):
        pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,color_primaries", "-of", "json", path], capture_output=True, text=True, check=True)
        doc = json.loads(pr.stdout)
        v = next((s for s in doc["streams"] if s.get("codec_type") == "video"), {})
        return float(doc["format"]["duration"]), (v.get("width"), v.get("height")), any(s.get("codec_type") == "audio" for s in doc["streams"]), v.get("color_primaries")

    # ---- Scenario 1: trim → concat → audio (loudness) → export → QC
    def test_s1_trim_concat_audio_export_qc(self):
        self._need("video-editing", "qc")
        svc = self._svc("s1")
        ir, p = self._plan(svc, [self.a, self.b], {"edit.concat": True, "qc": True}, name="s1")
        self.assertEqual([s["skill"] for s in ir.doc["plan"]["steps"]], ["silence_cleanup", "silence_cleanup", "video_concat", "loudness_normalization", "delivery_export", "delivery_check", "qc_check"])
        out = self._render(svc, p)
        qc = next(r for r in out["execution"]["results"] if r["tool"] == "qc/check")
        self.assertTrue(qc["ok"] and qc["data"]["admitted"]); self.assertIn(qc["data"]["verdict"], ("PASS", "WARN"), qc["data"].get("findings"))
        art = out["artifacts"][0]
        dur, size, has_audio, _ = self._probe(art["path"])
        self.assertAlmostEqual(dur, 22.0, delta=0.6, msg="two 11 s trims joined"); self.assertTrue(has_audio)
        self.assertEqual(qc["data"]["fingerprint"], art["hash"], "the admitted report is about the delivered bytes")
        self.assertEqual(art["qa"]["qc"], qc["data"]["verdict"]); self.assertEqual(art["stage"], "approved" if qc["data"]["verdict"] == "PASS" else "candidate")

    # ---- Scenario 2: video → transcription → subtitle → QC (real recognition, real sidecar, real burn-in)
    def test_s2_transcription_subtitle_qc(self):
        self._need("subtitle", "qc")
        speech = self._speech()
        svc = self._svc("s2")
        ir, p = self._plan(svc, [speech], {"subtitle": True, "subtitle.burn_in": True, "qc": True}, name="s2")
        d = ir.doc
        gen = next(op for op in d["captions"]["operations"] if op["type"] == "captions.generate")
        self.assertGreater(len(gen["cues"]), 0, "the recognition produced segments"); self.assertEqual(gen["language"], "ja")
        tr = next(o for o in d["analysis"]["observations"] if o["kind"] == "transcript")
        self.assertEqual(tr["provenance"], "OBSERVED"); self.assertTrue(all(s.get("speaker_id") is None for s in tr["data"]["segments"]))
        keep = gen["timeline_map"]["inputs"][list(d["assets"])[0]]["keep"]
        self.assertTrue(all(c["end"] <= sum(e - s for s, e in keep) + 0.05 for c in gen["cues"]), "every cue lies on the trimmed timeline")
        out = self._render(svc, p)
        res = {r["tool"]: r for r in out["execution"]["results"]}
        side = Path(res["subtitle/generate"]["output"]).read_text(encoding="utf-8")
        self.assertIn("-->", side); self.assertEqual(side.count("-->"), len(gen["cues"]))
        self.assertEqual(res["subtitle/render"]["data"]["engine"]["id"], "ffmpeg-skill")
        arts = {a["type"]: a for a in out["artifacts"]}
        items = {(i["layer"], i["name"], i["artifact"]): i for i in out["qa"]["items"]}
        self.assertEqual(items[("delivery", "cues", arts["CAPTIONS"]["logical_name"])]["status"], "PASS")
        self.assertIn(arts["CAPTIONS"]["qa"]["qc"], ("PASS", "WARN")); self.assertEqual(arts["CAPTIONS"]["stage"], "approved" if arts["CAPTIONS"]["qa"]["qc"] == "PASS" else "candidate")
        self._gate_coherent(out, arts["YOUTUBE"])
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        self.assertIn("subtitle.file", [o["kind"] for o in prov["skill_observations"]]); self.assertIn("qc.report", [o["kind"] for o in prov["skill_observations"]])

    # ---- Scenario 3: video → color grading (RETAG) → thumbnail (with caption) → QC
    def test_s3_color_thumbnail_qc(self):
        self._need("color-grading", "thumbnail", "qc")
        svc = self._svc("s3")
        ir, p = self._plan(svc, [self.a], {"color.target": "bt709", "thumbnail": True, "thumbnail.text": "Talk", "thumbnail.at": 2.0, "qc": True}, name="s3")
        out = self._render(svc, p)
        res = {r["tool"]: r for r in out["execution"]["results"]}
        self.assertEqual(res["color-grading/run"]["data"]["artifact"]["color_primaries"], "bt709")
        th = res["thumbnail/render"]
        self.assertTrue(th["output"].endswith(".png") and os.path.getsize(th["output"]) > 0)
        self.assertEqual((th["data"]["artifact"]["width"], th["data"]["artifact"]["height"]), (1280, 720), "the canvas is the picture size")
        arts = {a["type"]: a for a in out["artifacts"]}
        self.assertEqual((arts["THUMBNAIL"]["qa_status"], arts["THUMBNAIL"]["stage"]), ("PASS", "candidate"))
        _, _, _, prim = self._probe(arts["YOUTUBE"]["path"])
        self.assertEqual(prim, "bt709")

    # ---- Scenario 4: video → motion graphics (text + logo) → subtitle burn-in → QC
    def test_s4_motion_subtitle_burn_qc(self):
        self._need("motion-graphics", "subtitle", "qc")
        speech = self._speech()
        svc = self._svc("s4")
        ir, p = self._plan(svc, [speech], {"motion.text": "LIVE", "motion.text.fade": 0.3, "motion.image": self.png, "subtitle": True, "subtitle.burn_in": True, "qc": True}, name="s4")
        g = ir.doc["graphics"]["operations"][0]
        self.assertEqual([e["type"] for e in g["elements"]], ["text_overlay", "image_overlay"])
        self.assertIn(str(Path(self.png).parent), ir.doc["execution"]["allowed_inputs"])
        out = self._render(svc, p)
        res = {r["tool"]: r for r in out["execution"]["results"]}
        self.assertEqual({o["type"] for o in res["motion-graphics/run"]["data"]["operations"]}, {"text_overlay", "image_overlay"})
        self.assertEqual(res["subtitle/render"]["data"]["engine"]["id"], "ffmpeg-skill")
        arts = {a["type"]: a for a in out["artifacts"]}
        dur, _, _, _ = self._probe(arts["YOUTUBE"]["path"])
        self.assertGreater(dur, 5.0)
        self._gate_coherent(out, arts["YOUTUBE"])

    # ---- Scenario 5: the whole pipeline on two inputs
    def test_s5_multi_skill_pipeline(self):
        self._need("video-editing", "subtitle", "thumbnail", "color-grading", "motion-graphics", "qc")
        speech = self._speech()
        speech2 = str(Path(self.tmp) / "src" / "ja_talk2.mp4"); shutil.copy(speech, speech2)   # two talks: every source of the programme needs a transcript (a tone-only input is refused)
        svc = self._svc("s5")
        ir, p = self._plan(svc, [speech, speech2], {"edit.concat": True, "subtitle": True, "subtitle.burn_in": True, "thumbnail": True, "color.target": "bt709", "motion.title": "Opening", "qc": True}, name="s5")
        d = ir.doc
        trims = [s["skill"] for s in d["plan"]["steps"] if s["skill"] == "silence_cleanup"]
        self.assertEqual([s["skill"] for s in d["plan"]["steps"]], trims + ["video_concat", "color_retag", "motion_graphics", "subtitle_generation", "subtitle_burn_in",
                                                                            "loudness_normalization", "delivery_export", "delivery_check", "thumbnail_frame", "qc_check", "qc_check"])
        self.assertEqual(len({s["tool"].split("/")[0] for s in d["plan"]["steps"]}), 7)
        gen = next(op for op in d["captions"]["operations"] if op["type"] == "captions.generate")
        cat = next(op for op in d["video"]["operations"] if op["type"] == "video.concat")
        ids = list(d["assets"])
        self.assertEqual([gen["timeline_map"]["inputs"][i]["offset"] for i in ids], [seg["timeline_range"][0] for seg in cat["segments"]], "each talk's cues start where its segment lands on the programme")
        self.assertGreater(gen["timeline_map"]["inputs"][ids[1]]["offset"], 9.0); self.assertTrue(all(c["start"] >= 0 for c in gen["cues"]))
        out = self._render(svc, p)
        self.assertEqual({a["type"] for a in out["artifacts"]}, {"YOUTUBE", "CAPTIONS", "THUMBNAIL"})
        self._gate_coherent(out, next(a for a in out["artifacts"] if a["type"] == "YOUTUBE"))
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted({e["skill_package"] for e in prov["operations"]}), ["color-grading", "ffmpeg-skill", "motion-graphics", "qc", "subtitle", "thumbnail", "video-editing"])
        info = Service.explain_pipeline(d, provenance=prov, artifacts=out["artifacts"])
        self.assertTrue(all(info["counts"][lv] > 0 for lv in info["levels"]), info["counts"])

    # ---- Scenario 6: a failure in the middle → resume
    def test_s6_failure_then_resume(self):
        self._need("color-grading", "motion-graphics", "qc")
        svc = self._svc("s6")
        logo = str(Path(self.tmp) / "src" / "logo6.png"); shutil.copy(self.png, logo)
        ir, p = self._plan(svc, [self.a], {"color.target": "bt709", "motion.image": logo, "qc": True}, name="s6")
        os.remove(logo)   # the graphics input disappears before execution: the Skill refuses, nothing after it runs
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertIn(out["execution"]["status"], ("FAILED", "BLOCKED"))
        failed = next(r for r in out["execution"]["results"] if not r["ok"])
        self.assertEqual(failed["tool"], "motion-graphics/run"); self.assertEqual([r["tool"] for r in out["execution"]["results"] if r["ok"]], ["ffmpeg-skill/cut", "color-grading/run"])
        shutil.copy(self.png, logo)
        out2 = svc.render(load_ir(p), p, resume=out["job"]["id"])
        self.assertEqual(out2["execution"]["status"], "COMPLETED", out2["execution"].get("recovery"))
        self.assertEqual(len(out2["execution"]["skipped"]), 2, "the trim and the colour operation are reused")
        self.assertEqual(out2["execution"]["results"][0]["tool"], "motion-graphics/run")

    # ---- Scenario 7: plan revision → approval → execution
    def test_s7_revision_approval_execution(self):
        self._need("color-grading", "thumbnail", "qc")
        svc = self._svc("s7")
        ir, p = self._plan(svc, [self.a], {"color.target": "bt709", "thumbnail": True, "qc": True}, name="s7")
        th = next(x for x in ir.doc["decisions"] if x["subject"] == "thumbnail.render")
        svc.reject(load_ir(p), p, [th["id"]], reason="no thumbnail")
        self.assertEqual(svc.render(load_ir(p), p)["status"], "BLOCKED")
        svc.revise(load_ir(p), p, feedback="drop the thumbnail")
        ir2 = load_ir(p)
        self.assertEqual(ir2.version, 2); self.assertNotIn("thumbnail_frame", [s["skill"] for s in ir2.doc["plan"]["steps"]])
        self.assertEqual(svc.render(load_ir(p), p)["status"], "WAITING_FOR_APPROVAL")
        svc.approve(load_ir(p), p, ["all"])
        out = svc.render(load_ir(p), p)
        self.assertEqual(out["execution"]["status"], "COMPLETED"); self.assertEqual({a["type"] for a in out["artifacts"]}, {"YOUTUBE"})

    # ---- Scenario 8: capability drift → BLOCK (a checkout whose contract changed is MISSING, never used)
    def test_s8_capability_drift_blocks(self):
        self._need("color-grading")
        from video_agent.capabilities import CapabilityResolver
        from video_agent.tools.color_grading import locate_color_grading
        root = Path(locate_color_grading().root or "")
        if not root.is_dir():
            self.skipTest("needs a color-grading-skill checkout (not a console script)")
        drifted = Path(self.tmp) / "drifted-color-grading"
        shutil.copytree(root / "src", drifted / "src")
        cp = drifted / "src" / "color_grading" / "contract.py"
        text = cp.read_text(encoding="utf-8")
        marker = '"max_operations": MAX_OPERATIONS,'
        self.assertIn(marker, text)
        cp.write_text(text.replace(marker, '"max_operations": MAX_OPERATIONS + 1,', 1), encoding="utf-8")   # a different request contract: drift, never silently kept
        caps = CapabilityResolver(str(locate_ffmpeg_skill().root), color_grading_dir=str(drifted)).resolve()
        self.assertEqual(caps["color-grading"].status, "MISSING", caps["color-grading"].detail)
        self.assertTrue(caps["color-grading"].evidence.get("drift") or "unusable" in caps["color-grading"].detail)
        svc = Service(workspace=str(Path(self.tmp) / "s8"), color_grading_dir=str(drifted))
        ir = svc.plan([self.a], "youtube", user_requirements={"color.target": "bt709"})
        dec = {x["subject"]: x for x in ir.doc["decisions"]}
        self.assertEqual(dec["capability.color_retag"]["approval"], "BLOCK"); self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        p = str(Path(svc.workspace) / "s8.json"); save_ir(ir, p)
        self.assertIn(svc.render(load_ir(p), p, approve=["all"])["status"], ("BLOCKED", "FAILED"))

    # ---- Scenario 9: hash mismatch → reuse forbidden
    def test_s9_hash_mismatch_forbids_reuse(self):
        self._need("color-grading", "qc")
        svc = self._svc("s9")
        ir, p = self._plan(svc, [self.a], {"color.target": "bt709", "qc": True}, name="s9")
        out = self._render(svc, p)
        art = out["artifacts"][0]
        with open(art["path"], "ab") as fh:
            fh.write(b"tampered")
        self.assertFalse(svc.artifact(art["id"])["integrity"]["ok"])
        from video_agent.artifacts import ArtifactError
        with self.assertRaises(ArtifactError):
            svc.promote_artifact(art["id"], "final")
        job_dir = Path(svc.workspace) / "jobs" / out["job"]["id"]
        inter = next(x for x in job_dir.rglob("*_retag.mp4"))
        with open(inter, "ab") as fh:
            fh.write(b"x")
        out2 = svc.render(load_ir(p), p, resume=out["job"]["id"])
        self.assertEqual(out2["execution"]["status"], "COMPLETED")
        self.assertIn("color-grading/run", [r["tool"] for r in out2["execution"]["results"]], "the tampered intermediate is produced again, never reused")
        self.assertNotEqual(out2["artifacts"][0]["hash"], art["hash"] + "x", "a fresh, verified artifact")
        qc = next(r for r in out2["execution"]["results"] if r["tool"] == "qc/check")
        self.assertEqual(qc["data"]["fingerprint"], out2["artifacts"][0]["hash"])

    # ---- Scenario 10: same input, same plan → deterministic, idempotent
    def test_s10_determinism_and_idempotency(self):
        self._need("color-grading", "motion-graphics", "qc")
        reqs = {"color.target": "bt709", "motion.title": "Opening", "qc": True}
        svc = self._svc("s10")
        ir1, p1 = self._plan(svc, [self.a], reqs, name="s10a")
        ir2, p2 = self._plan(svc, [self.a], reqs, name="s10b")
        strip = lambda doc: json.dumps({k: doc[k] for k in ("video", "color", "graphics", "delivery")}, sort_keys=True).replace(list(doc["assets"])[0], "A")  # noqa: E731
        import re as _re
        self.assertEqual(_re.sub(r"dec_[0-9a-f]{10}", "dec", strip(ir1.doc)), _re.sub(r"dec_[0-9a-f]{10}", "dec", strip(ir2.doc)))
        out = self._render(svc, p1)
        out2 = svc.render(load_ir(p1), p1, resume="last")
        self.assertEqual(out2["execution"]["status"], "COMPLETED")
        transforms = [r for r in out["execution"]["results"] if r["ok"] and r["output"]]
        self.assertEqual(len(out2["execution"]["skipped"]), len(transforms), "every transform is reused; the checks and the QC gate run again")
        self.assertEqual(out2["artifacts"][0]["hash"], out["artifacts"][0]["hash"])
        self.assertEqual(out2["artifacts"][0]["id"], out["artifacts"][0]["id"], "the same bytes are the same artifact")
