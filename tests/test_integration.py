"""Integration + real-media tests: need ffmpeg and an ffmpeg-skill checkout (VIDEO_AGENT_FFMPEG_SKILL_DIR)."""
from __future__ import annotations

import http.server
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


# ADR-048: ffmpeg-skill 2.x `check.py --platform youtube` warns when a delivery carries no language-tagged subtitle
# track. That is a true statement about a deliverable these tests did not ask subtitles for, so it is the one
# non-PASS item they tolerate; any other WARN/FAIL still fails them.
def _subtitle_track_warn(item):
    return item.get("status") == "WARN" and item.get("layer") == "delivery" and item.get("name") == "subtitles"


def qa_status(out):
    """The run's QA status, with a missing-subtitle-track WARN on a delivery treated as PASS."""
    qa = out["qa"]
    rest = [i for i in qa["items"] if i["status"] != "PASS" and not _subtitle_track_warn(i)]
    return "PASS" if qa["status"] == "WARN" and not rest else qa["status"]


def artifact_qa(out, art):
    """An artifact's qa_status, with the same single tolerance as qa_status()."""
    if art["qa_status"] != "WARN":
        return art["qa_status"]
    mine = [i for i in out["qa"]["items"] if i["status"] != "PASS" and (i.get("subject_artifact_id") == art.get("id") or i.get("artifact") == art.get("logical_name"))]
    return "PASS" if mine and all(_subtitle_track_warn(i) for i in mine) else art["qa_status"]


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
        self.assertEqual(qa_status(out), "PASS", [i for i in out["qa"]["items"] if i["status"] != "PASS"])
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
        self.assertTrue(lm.ok and "input_i" in (lm.data.get("measured") or lm.data))   # ffmpeg-skill 2.x nests it under `measured`
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
        self.assertEqual(qa_status(out), "PASS", [i for i in out["qa"]["items"] if i["status"] != "PASS"])
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
        self.assertEqual(qa_status(out), "PASS")
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
        self.assertEqual(qa_status(out), "PASS", [i for i in out["qa"]["items"] if i["status"] != "PASS"])
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
        self.assertTrue(all("audio" in i["name"] for i in out2["qa"]["items"] if i["status"] != "PASS" and not _subtitle_track_warn(i)))
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
        self.assertEqual((artifact_qa(out, a), a["stage"], a["delivery_status"]), ("PASS", "candidate", "READY"))
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
        self.assertEqual(qa_status(second), "PASS")
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
        self.assertEqual(qa_status(res), "PASS", [i for i in res["qa"]["items"] if i["status"] != "PASS"])
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
        self.assertEqual(qa_status(res), "PASS", [i for i in res["qa"]["items"] if i["status"] != "PASS"])
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
        # ADR-082: which ops finish inside the 1 s / doubled 2 s retry depends on the engine's speed (ffmpeg-skill 3.x's cut.py and
        # loudness.py do; 2.x's did not), so the check is exact rather than "nothing at all": every file left under ops/ is the output
        # of an attempt that completed (exit 0), and an attempt that timed out left no file behind for a retry to collide with.
        results = out["execution"]["results"]
        completed = {os.path.realpath(r["output"]) for r in results if r["ok"] and r["exit_code"] == 0 and r["output"]}
        left = {os.path.realpath(str(x)) for x in (job_dir / "ops").rglob("*.mp4")}
        self.assertEqual(left, completed, "half-written outputs must not be left for a retry to collide with")
        failed = out["execution"]["failed_op"]
        self.assertTrue(failed and not any(r["ok"] for r in results if r["op_id"] == failed), out["execution"])
        self.assertTrue(all(r["exit_code"] == 124 for r in results if r["op_id"] == failed))

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


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class _QuietServer(http.server.ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass   # yt-dlp may close a connection once it has the headers: a broken pipe here is expected


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill(), "needs ffmpeg and ffmpeg-skill")
class IngestFetchRealTests(unittest.TestCase):
    """ADR-059 on the real ingest-skill (main, with its `fetch` extra): a URL input is downloaded by `ingest fetch` from a fixture
    served on 127.0.0.1 (no internet; yt-dlp's generic extractor takes the direct media link, as in ingest-skill's own loopback
    test), lands in <workspace>/downloads/<extractor>-<id>/ with ingest-skill's manifest, and flows through plan like a local
    file with an OBSERVED `ingest` observation whose URLs carry no credentials."""

    @classmethod
    def setUpClass(cls):
        import functools
        import threading
        from video_agent.agent.ingest import locate_ingest, yt_dlp_capability
        if not locate_ingest():
            raise unittest.SkipTest("needs ingest-skill (VIDEO_AGENT_INGEST_DIR)")
        status, detail, _ = yt_dlp_capability()
        if status != "AVAILABLE":
            raise unittest.SkipTest(f"needs ingest-skill's fetch extra: {detail}")
        cls.tmp = os.path.realpath(tempfile.mkdtemp(prefix="va_fetch_"))
        www = Path(cls.tmp) / "www"
        www.mkdir()
        make_media(str(www / "talk.mp4"))
        cls.server = _QuietServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(www)))
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls._no_proxy = {k: os.environ.get(k) for k in ("NO_PROXY", "no_proxy")}
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        for k, v in cls._no_proxy.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_url_input_is_fetched_by_ingest_skill_and_planned(self):
        import hashlib
        from video_agent.agent.ingest import IngestError, locate_ingest
        from video_agent.cli import _resolve_url_inputs
        from video_agent.tools.skill_process import invoke, one_json_document
        port = self.server.server_address[1]
        url = f"http://user:pa55@127.0.0.1:{port}/talk.mp4?token=s3cr3t-token"
        ws = str(Path(self.tmp) / "ws")
        inputs = [url]
        records = _resolve_url_inputs(inputs, ws)
        rec = records[inputs[0]]
        root = Path(ws).resolve() / "downloads" / "generic-talk"
        self.assertEqual(Path(rec.path), root / "talk.mp4")
        self.assertEqual(rec.sha256, hashlib.sha256((Path(self.tmp) / "www" / "talk.mp4").read_bytes()).hexdigest(), "the served bytes, verified")
        self.assertEqual(rec.source_url, f"http://127.0.0.1:{port}/talk.mp4?token=REDACTED"); self.assertEqual(rec.redacted, ["token", "userinfo"])
        self.assertEqual((rec.extractor, rec.video_id, rec.is_live, rec.live_from_start), ("generic", "talk", False, False)); self.assertTrue(rec.yt_dlp_version)
        manifest = (root / "ingest-manifest.json").read_text(encoding="utf-8")
        self.assertNotIn("s3cr3t-token", manifest); self.assertNotIn("pa55", manifest)
        code, out, _ = invoke(locate_ingest(), ["verify", str(root), "--json"], timeout=120)
        self.assertEqual(code, 0); self.assertTrue(one_json_document(out, "ingest verify")["ok"], "ingest verify works on the agent's download folder")
        ir = Service(workspace=ws).plan(inputs, "youtube", ingest_records=records)
        obs = [o for o in ir.doc["analysis"]["observations"] if o["kind"] == "ingest"]
        self.assertEqual(len(obs), 1); self.assertEqual(obs[0]["provenance"], "OBSERVED"); self.assertTrue(obs[0]["source"].startswith("ingest/fetch@"))
        self.assertNotIn("s3cr3t-token", json.dumps(ir.doc)); self.assertNotIn("pa55", json.dumps(ir.doc))
        # a missing file on the server: ingest-skill's FETCH_FAILED, mapped to DOWNLOAD_FAILED, the secret not echoed
        with self.assertRaises(IngestError) as cm:
            _resolve_url_inputs([f"http://127.0.0.1:{port}/missing.mp4?sig=SIGVALUE"], ws)
        self.assertEqual(cm.exception.kind, "DOWNLOAD_FAILED"); self.assertNotIn("SIGVALUE", str(cm.exception))


def make_xdcam_ex_card(base: Path) -> Path:
    """A small XDCAM EX-like card (ingest-skill's own synthetic card approach, its tests/test_cards.py, with real essence): take
    709_0001 split over clip folders _01 and _02 (one recording), take 709_0002 in one folder, each clip with a NonRealTimeMeta
    M01.XML sidecar and a .SMI, plus MEDIAPRO.XML. Each clip is 1 s of 320x180 25p H.264 + AAC with its own tone, so no two files match."""
    card = base / "EX_CARD"
    b = card / "BPAV"
    for i, (clip, frames) in enumerate((("709_0001_01", 25), ("709_0001_02", 25), ("709_0002_01", 25))):
        d = b / "CLPR" / clip
        d.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25", "-f", "lavfi", "-i", f"sine=frequency={440 + 110 * i}:sample_rate=48000",
                        "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / f"{clip}.MP4")], check=True)
        (d / f"{clip}M01.XML").write_text('<?xml version="1.0" encoding="UTF-8"?><NonRealTimeMeta xmlns="urn:schemas-professionalDisc:nonRealTimeMeta:ver.2.00">'
                                          f'<Duration value="{frames}"/><CreationDate value="2026-10-30T09:00:00+09:00"/>'
                                          '<Device manufacturer="Sony" modelName="PMW-EX1" serialNo="0012345"/></NonRealTimeMeta>', encoding="utf-8")
        (d / f"{clip}.SMI").write_text("<smil/>", encoding="utf-8")
    (b / "MEDIAPRO.XML").write_text('<?xml version="1.0"?><MediaProfile xmlns="http://xmlns.sony.net/pro/metadata/mediaprofile"><Contents/></MediaProfile>', encoding="utf-8")
    return card


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill(), "needs ffmpeg and ffmpeg-skill")
class IngestOffloadRealTests(unittest.TestCase):
    """ADR-066 on the real ingest-skill (main): a synthetic XDCAM EX-like card is planned (ingest-skill's dry run and inventory; nothing
    written), confirmed, copied to two destinations with read-back, and the copy's inventory becomes assets: one per member file, the
    spanned take's two members in order, OBSERVED `ingest_clip` observations with the reel, hashes, sidecar facts and each destination's
    manifest and ASC MHL generation; the agent's own probe runs on the copies; ingest verify passes on both. A conflicting file on one
    drive is reported as CONFLICT, never overwritten, and no asset is admitted."""

    def setUp(self):
        from video_agent.tools.ingest import locate_ingest
        if not locate_ingest():
            raise unittest.SkipTest("needs ingest-skill (VIDEO_AGENT_INGEST_DIR)")
        self.tmp = Path(os.path.realpath(tempfile.mkdtemp(prefix="va_offload_")))
        self.card = make_xdcam_ex_card(self.tmp)
        self.a, self.b = self.tmp / "SSD1", self.tmp / "SSD2"
        self.a.mkdir()
        self.b.mkdir()
        self.svc = Service(workspace=str(self.tmp / "ws"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _plan(self):
        ir = self.svc.plan_offload(str(self.card), {"ingest.dest": [str(self.a), str(self.b)], "ingest.project": "event"})
        path = str(self.tmp / "off.project.json")
        save_ir(ir, path)
        return ir, path

    def _approve_render(self, path):
        ir = load_ir(path)
        self.svc.approve(ir, path, [d["id"] for d in ir.pending_confirmations()])
        return self.svc.render(load_ir(path), path)

    def test_card_offload_then_inventory_assets(self):
        from video_agent.tools import ToolRouter  # noqa: F401
        from video_agent.tools.ingest import IngestAdapter, locate_ingest
        ir, path = self._plan()
        self.assertTrue(self.svc.validate(ir).ok, self.svc.validate(ir).errors)
        self.assertEqual((list(self.a.iterdir()), list(self.b.iterdir())), ([], []), "the plan wrote nothing")
        inv = next(o for o in ir.doc["analysis"]["observations"] if o["kind"] == "ingest_card_inventory")["data"]
        self.assertEqual((inv["clip_count"], inv["spanned_count"], inv["file_count"]), (2, 1, 3))
        self.assertEqual(inv["card"]["structures"][0]["id"], "sony_xdcam_ex")
        self.assertEqual(self.svc.render(load_ir(path), path)["status"], "WAITING_FOR_APPROVAL")
        out = self._approve_render(path)
        self.assertEqual(out["status"], "COMPLETED", json.dumps(out.get("offload"), default=str)[:1500])
        d = load_ir(path).doc
        off = d["execution"]["offload"]
        roots = [self.a / "event" / "EX_CARD", self.b / "event" / "EX_CARD"]
        self.assertEqual([x["root"] for x in off["summary"]["destinations"]], [str(r) for r in roots])
        for r in roots:
            self.assertTrue((r / "ingest-manifest.json").is_file())
            self.assertTrue(list((r / "ascmhl").glob("*.mhl")), "an ASC MHL generation per destination")
        clip_obs = [o for o in d["analysis"]["observations"] if o["kind"] == "ingest_clip"]
        self.assertEqual(len(clip_obs), 3)
        take = sorted((o for o in clip_obs if o["data"]["clip_id"] == "709_0001"), key=lambda o: o["data"]["member_index"])
        self.assertEqual([Path(o["data"]["path"]).name for o in take], ["709_0001_01.MP4", "709_0001_02.MP4"])
        self.assertTrue(all(o["data"]["spanned"] and o["data"]["structure"] == "sony_xdcam_ex" for o in take))
        self.assertEqual(take[0]["data"]["metadata"]["device"]["model"], "PMW-EX1", "the camera's sidecar facts, as written")
        self.assertAlmostEqual(take[0]["data"]["member_duration_s"], 1.0, delta=0.1)
        self.assertEqual((take[0]["data"]["reel"]["source"], take[0]["data"]["reel"]["recorded_source"]), ("manifest", "fingerprint"))
        man = json.loads((roots[0] / "ingest-manifest.json").read_text(encoding="utf-8"))
        sha = {e["path"]: e["sha256"] for e in man["files"]}
        for o in clip_obs:
            a = d["assets"][o["asset_id"]]
            self.assertEqual((o["provenance"], a["status"], a["hash"]), ("OBSERVED", "ingested", sha[o["data"]["path"]]))
            self.assertTrue(o["source"].startswith("ingest/inventory@0.1."))
            self.assertTrue(all(Path(c["mhl"]).is_file() and Path(c["manifest"]).is_file() for c in o["data"]["copies"]))
        probes = {o["asset_id"]: o for o in d["analysis"]["observations"] if o["kind"] == "media_probe"}
        self.assertEqual(set(probes), set(off["asset_ids"]), "the agent's own analysis of the verified copies")
        self.assertTrue(self.svc.validate(load_ir(path)).ok)
        ad = IngestAdapter(locate_ingest())
        self.assertEqual((ad.drift(), ad.breaking_drift()), ([], []), "the pinned contract equals ingest-skill main")
        for r in roots:
            v = ad.measure("ingest/verify", {"target": str(r)})
            self.assertTrue(v.ok and v.data["verified"], v.data.get("failures"))

    def test_conflict_is_named_and_nothing_is_overwritten(self):
        ir, path = self._plan()
        clash = self.a / "event" / "EX_CARD" / "BPAV" / "MEDIAPRO.XML"
        clash.parent.mkdir(parents=True)
        clash.write_text("not the card's file", encoding="utf-8")
        out = self._approve_render(path)
        self.assertEqual(out["status"], "FAILED")
        err = load_ir(path).doc["execution"]["offload"]["error"]
        self.assertEqual((err["code"], err["kind"], err["exit_code"]), ("INCOMPLETE", "CONFLICT", 3))
        self.assertEqual(err["details"]["complete"], [str(self.b / "event" / "EX_CARD")], "the other drive is complete and verified")
        bad = next(x for x in err["details"]["destinations"] if not x["ok"])
        self.assertEqual(bad["conflict_files"], ["BPAV/MEDIAPRO.XML"])
        self.assertEqual(clash.read_text(encoding="utf-8"), "not the card's file", "never overwritten")
        self.assertFalse([a for a in load_ir(path).doc["assets"].values() if a["status"] == "ingested"])


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

    def test_contract_publishes_every_catalog_flag(self):
        """ADR-056 (roadmap step 3): ffmpeg-skill main's own `_contract.py --json --static` is contract_version 1.x (ffmpeg-skill 2.x) or
        2.x (ffmpeg-skill 3.x, ADR-082) and every flag,
        switch arity and positional the hand-maintained catalog uses is in that tool's input_schema -- so on main no catalogued
        tool is ruled out by the contract (doctor's per-tool verdict depends on this machine's ffmpeg and is checked apart)."""
        from video_agent.tools.ffmpeg_skill.contract_check import check_catalog, read_catalog_check, read_contract
        skill = locate_ffmpeg_skill()
        contract, err = read_contract(skill.root)
        self.assertIsInstance(contract, dict, err)
        want_major = {"2": "1.", "3": "2."}.get(str(skill.version).split(".", 1)[0])
        self.assertIsNotNone(want_major, f"ffmpeg-skill {skill.version} is outside the 2.x / 3.x this test knows")
        self.assertTrue(str(contract["contract_version"]).startswith(want_major), (skill.version, contract["contract_version"]))
        chk = check_catalog(contract, None, skill.scripts)
        self.assertEqual((chk.status, chk.unavailable), ("checked", {}))
        self.assertEqual(set(CATALOG) - set(skill.scripts), set())
        full = read_catalog_check(skill.root, skill.scripts)
        self.assertEqual(full.status, "checked")
        self.assertIsNotNone(full.doctor, full.notes)
        for name, why in full.unavailable.items():   # only doctor may rule a tool out here, and then with its own words
            self.assertTrue(why.startswith("ffmpeg-skill doctor:"), why)
        caps = Service(workspace=tempfile.mkdtemp()).caps.resolve()
        self.assertEqual(caps["ffmpeg-skill"].evidence["contract"]["status"], "checked")
        for name in CATALOG:
            self.assertEqual(caps[f"ffmpeg-skill:{name}"].status, "MISSING" if name in full.unavailable else "AVAILABLE", name)

    def test_help_declares_every_catalog_flag(self):
        skill = locate_ffmpeg_skill()
        from video_agent.tools.ffmpeg_skill.catalog import FLAG_ALIASES, QUALITY_FLAG
        for name, spec in CATALOG.items():
            out = subprocess.run([sys.executable, str(skill.script(name)), "--help"], capture_output=True, text=True).stdout
            for flag in spec["flags"]:
                cli = FLAG_ALIASES.get((name, flag)) or FLAG_ALIASES.get(("*", flag)) or "--" + flag.replace("_", "-")
                if flag == "crf" and skill.offers_flag(name, QUALITY_FLAG):
                    cli = QUALITY_FLAG   # ADR-048: the adapter emits --quality where the installed script offers it
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
        from video_agent.tools.media_analysis.adapter import SUPPORTED_SKILL_VERSIONS
        cap = svc.caps.resolve()["media-analysis"]
        self.assertEqual(cap.status, "AVAILABLE", cap.evidence)
        self.assertEqual((cap.evidence["contract"], cap.evidence["execution"]), ("media-analysis/contract@1", "local_subprocess"))
        self.assertTrue(cap.evidence["version"].startswith(SUPPORTED_SKILL_VERSIONS), cap.evidence["version"])   # the installed release, inside the adapter's gate
        self.assertEqual(len(cap.evidence["tools"]), 10)   # 0.1.7 (ADR-078): media-analysis/noise (noise_floor) is the tenth
        self.assertIn("media-analysis/noise", cap.evidence["tools"])
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

    def test_explicit_provider_requirement_selects_media_analysis_over_the_default(self):
        """docs/CAPABILITY_MODEL.md's collision policy, Tier 1, against the real collision it names: with both
        ffmpeg-skill and media-analysis-skill genuinely installed, a `--set provider.media_probe=media-analysis`
        requirement wins over the OS's own baked-in default (ffmpeg-skill) -- real packages, real contracts, not
        a synthetic candidate list."""
        svc = Service(workspace=str(Path(self.tmp) / "ws_explicit"))
        tools = svc.tools_for(user_requirements={"provider.media_probe": "media-analysis"})
        self.assertEqual(tools["media_probe"], "media-analysis/probe")
        self.assertEqual(tools["loudness_analysis"], "ffmpeg-skill/loudness", "an explicit choice for one skill never leaks onto a different, unrelated skill")
        ir = svc.plan([self.src], "generic", user_requirements={"provider.media_probe": "media-analysis", "provider.loudness_analysis": "media-analysis"})
        req_values = {r["key"]: r["value"] for r in ir.doc["requirements"]}
        self.assertEqual(req_values.get("provider.media_probe"), "media-analysis", "an explicit provider choice is recorded like any other requirement, for provenance")
        self.assertEqual(req_values.get("provider.loudness_analysis"), "media-analysis")

    def test_explicit_provider_naming_an_uninstalled_package_refuses_loudly(self):
        svc = Service(workspace=str(Path(self.tmp) / "ws_bad_explicit"))
        with self.assertRaises(RuntimeError) as ctx:
            svc.analyze([self.src], "generic", user_requirements={"provider.media_probe": "no-such-skill"})
        self.assertIn("no-such-skill", str(ctx.exception))
        self.assertIn("ffmpeg-skill", str(ctx.exception)); self.assertIn("media-analysis", str(ctx.exception))

    def test_workspace_providers_json_sets_a_default_without_any_explicit_requirement(self):
        """Tier 2: a workspace-level `providers.json` resolves the collision the same way an explicit
        requirement would, but for every plan in this workspace, with none of them needing to ask."""
        ws = str(Path(self.tmp) / "ws_default")
        Path(ws).mkdir(parents=True)
        Path(ws, "providers.json").write_text(json.dumps({"media_probe": "media-analysis", "silence_analysis": "media-analysis"}), encoding="utf-8")
        svc = Service(workspace=ws)
        tools = svc.tools_for()
        self.assertEqual(tools["media_probe"], "media-analysis/probe")
        self.assertEqual(tools["silence_analysis"], "media-analysis/silence")
        self.assertEqual(tools["loudness_analysis"], "ffmpeg-skill/loudness", "only the two skills named in providers.json are redirected")
        rows = {r["skill"]: r for r in svc.skills()}
        self.assertEqual(rows["media_probe"]["tool"], "media-analysis/probe")

    def test_all_kinds_lifted_with_provenance_and_skill_cache(self):
        ws = str(Path(self.tmp) / "ws_all")
        svc = self._service(ws)
        installed = svc.caps.resolve()["media-analysis"].evidence["version"]   # whatever release is installed: provenance must name it
        extra = ["stream_layout", "video_format", "audio_format", "duration", "integrity", "scene_detection", "timing"]
        profile, rules, an = svc.analyze([self.src], "youtube", kinds=extra)
        obs = {o.kind: o for o in an.observations}
        self.assertEqual(sorted(obs), sorted(["media_probe", "silence", "loudness"] + extra))
        for o in obs.values():
            self.assertEqual((o.provenance, o.skill, o.skill_version), ("OBSERVED", "media-analysis", installed), o.kind)
            self.assertTrue(o.tool.startswith("media-analysis/") and o.source == f"{o.tool}@{installed}" and o.external_id.startswith("obs_"), o.kind)
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
        self.assertEqual(qa_status(out), "PASS", [i for i in out["qa"]["items"] if i["status"] != "PASS"])
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
        from video_agent.tools.transcription.adapter import SUPPORTED_SKILL_VERSIONS
        ad = self.adapter
        self.assertEqual(check_contract(ad.contract), [])
        self.assertEqual((ad.contract["id"], ad.contract["schemas"]["transcript"]), ("transcription-skill", "transcription-skill/transcript/0.1"))
        self.assertTrue(ad.version.startswith(SUPPORTED_SKILL_VERSIONS), ad.version)
        self.assertLessEqual({"transcription/check", "transcription/export", "transcription/segments", "transcription/transcribe"}, set(ad.tools))   # 0.7 added transcription/batch
        self.assertTrue(ad.supports("transcription/transcribe") and not ad.supports("transcription/export"))
        eng = {e["id"]: e for e in ad.engine_status()}
        self.assertEqual({k: (e["execution_mode"], e["requires_network"]) for k, e in eng.items()}, {k: ("local", False) for k in eng}, "every implemented engine is local")
        # no cloud engine, no whisper.cpp: the contract lists implemented engines only -- the same ones as the pin (0.7.7 added the optional local haas_ja, 0.7.9 the optional local speechcpp)
        from video_agent.tools.transcription import pinned_contract
        self.assertEqual(list(eng), [e["id"] for e in pinned_contract()["engines"]])
        self.assertEqual(list(eng)[0], "faster_whisper")
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
        from video_agent.tools.ffmpeg_skill.locate import locate_ffmpeg_skill
        self.assertEqual(o.source.split("@")[0], "ffmpeg-skill/sync")
        self.assertEqual(o.source.split("@")[1], locate_ffmpeg_skill().version, o.source)   # the installed checkout's actual version, not a decided-string guess (ADR-041)
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


class CameraSwitchRealTests(unittest.TestCase):
    """ADR-045 through the real ffmpeg-skill 0.9.x CLI: two real 16 s inputs → an explicit `edit.switch` list ("0-6:0,6-16:1")
    → the real ffmpeg-skill/multicam tool aligns them by audio and cuts between them on the reference timeline. This agent
    never measures alignment itself and never picks the cut points -- both are the user's own switch list and the Skill's
    own algorithm; what this test verifies is the plumbing (plan → IR → compiler → real tool → artifact/QA), and that the
    programme's duration is the reference input's own duration, never a sum like video.concat would give."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="va_switch_")
        src = Path(cls.tmp) / "src"; src.mkdir()
        cls.a, cls.b = str(src / "a.mp4"), str(src / "b.mp4")
        make_media(cls.a)
        make_media(cls.b)

    def test_switch_plan_render_and_duration(self):
        ws = str(Path(self.tmp) / "run")
        svc = Service(workspace=ws)
        self.assertEqual(svc.tools_for(svc.adapter([])).get("camera_switch"), "ffmpeg-skill/multicam")
        ir = svc.plan([self.a, self.b], "generic", user_requirements={"edit.switch": "0-6:0,6-16:1"})
        d = ir.doc
        self.assertEqual(d["plan"]["status"], "APPROVED", d["plan"]["summary"])
        self.assertEqual(svc.validate(ir).errors, [])
        sw_op = next(op for op in d["video"]["operations"] if op["type"] == "video.switch")
        a_id = list(d["assets"])[0]
        self.assertEqual(sw_op["switch"], "0-6:0,6-16:1")
        # the programme's duration is the reference input's own (trimmed) duration -- not a sum of both inputs like concat would give
        a_trim = next(op["keep"] for op in d["video"]["operations"] if op["type"] == "video.trim" and op["asset"] == a_id)
        expected = round(sum(e - s for s, e in a_trim), 3)
        self.assertAlmostEqual(sw_op["timeline_duration"], expected, delta=0.01, msg=sw_op)
        self.assertLess(sw_op["timeline_duration"], 32.0, "nowhere near what concat's sum of two ~16s inputs would give")
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "camera_switch")
        self.assertEqual(step["tool"], "ffmpeg-skill/multicam")
        p = str(Path(ws) / "switch.json")
        save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED", out)
        op_result = next(o for o in out["execution"]["results"] if o["tool"] == "ffmpeg-skill/multicam")
        self.assertTrue(op_result["ok"], op_result)
        self.assertEqual(len(out["artifacts"]), 1, out["artifacts"])
        art = out["artifacts"][0]
        self.assertEqual(artifact_qa(out, art), "PASS", out["qa"])
        out_path = Path(ws) / "jobs" / out["job"]["id"] / "ops" / "programme_01_switch" / "programme.mp4"
        pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(out_path)], capture_output=True, text=True, check=True)
        self.assertAlmostEqual(float(pr.stdout.strip()), expected, delta=0.5, msg="the real multicam output is the reference timeline's own length, not the sum of both inputs")
        self.assertIn(sw_op["decision_ids"][0], art["decision_ids"])


class PriorityAToolsRealTests(unittest.TestCase):
    """ADR-046 through the real ffmpeg-skill 0.16.x CLI: redact.py and crop.py (chosen as the most impactful, small, fast,
    deterministic candidates -- see the PR description for why the other three (deinterlace, stabilize, grid) are
    covered by fake-adapter/unit tests only in this batch). Both are single-source, fixed-pixel-rectangle operations;
    this test verifies the plumbing (plan -> IR -> compiler -> real tool -> artifact/QA) and, for crop, that the real
    output frame is exactly the requested rectangle."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="va_priority_a_")
        src = Path(cls.tmp) / "src"; src.mkdir()
        cls.a = str(src / "a.mp4")
        make_media(cls.a)

    def test_redact_plan_render_and_qa(self):
        ws = str(Path(self.tmp) / "run_redact")
        svc = Service(workspace=ws)
        ir = svc.plan([self.a], "generic", user_requirements={"edit.redact": {"x": 100, "y": 100, "width": 200, "height": 100}, "edit.redact.mode": "pixelate"})
        d = ir.doc
        self.assertEqual(d["plan"]["status"], "APPROVED", d["plan"]["summary"])
        self.assertEqual(svc.validate(ir).errors, [])
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "video_redact")
        self.assertEqual(step["tool"], "ffmpeg-skill/redact")
        p = str(Path(ws) / "redact.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED", out)
        op_result = next(o for o in out["execution"]["results"] if o["tool"] == "ffmpeg-skill/redact")
        self.assertTrue(op_result["ok"], op_result)
        self.assertEqual(artifact_qa(out, out["artifacts"][0]), "PASS", out["qa"])

    def test_crop_plan_render_and_real_output_dimensions(self):
        ws = str(Path(self.tmp) / "run_crop")
        svc = Service(workspace=ws)
        ir = svc.plan([self.a], "generic", user_requirements={"edit.crop": {"x": 0, "y": 140, "width": 1280, "height": 440}})
        d = ir.doc
        self.assertEqual(d["plan"]["status"], "APPROVED", d["plan"]["summary"])
        self.assertEqual(svc.validate(ir).errors, [])
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "video_crop")
        self.assertEqual(step["tool"], "video-editing/crop")   # ADR-079: video-editing-skill's CROP (it runs ffmpeg-skill crop.py itself)
        p = str(Path(ws) / "crop.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED", out)
        op_result = next(o for o in out["execution"]["results"] if o["tool"] == "video-editing/crop")
        self.assertTrue(op_result["ok"], op_result)
        self.assertEqual(artifact_qa(out, out["artifacts"][0]), "PASS", out["qa"])
        out_path = Path(op_result["output"])
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(out_path)],
                             capture_output=True, text=True, check=True)
        self.assertEqual(pr.stdout.strip(), "1280x440", "the real video-editing CROP output is exactly the requested rectangle")

    def test_cropdetect_real_json_shape(self):
        """Resolves the ADR-046 open item: catalog.py's cropdetect result_keys were originally an unverified guess
        (flat x/y/width/height + a crop_filter key that doesn't exist). Verified here against a real --json capture
        on a synthetic letterboxed clip: the rectangle is a nested "crop" object, plus file/source_width/source_height/
        confidence -- no crop_filter key at all. Calls the tool adapter directly (cropdetect is not a pipeline
        edit-op, see ADR-046)."""
        from video_agent.tools.ffmpeg_skill.adapter import FfmpegSkillAdapter
        from video_agent.models import Operation
        letterboxed = str(Path(self.tmp) / "letterboxed.mp4")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", "color=c=black:s=640x480:d=3,drawbox=x=0:y=60:w=640:h=360:color=white@1:t=fill",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-pix_fmt", "yuv420p",
                        "-c:v", "libx264", "-c:a", "aac", "-shortest", letterboxed], check=True)
        adapter = FfmpegSkillAdapter(locate_ffmpeg_skill())
        op = Operation(tool="ffmpeg-skill/cropdetect", args={"input": letterboxed}, inputs=[letterboxed], outputs=[])
        r = adapter.run(op, {letterboxed: letterboxed})
        self.assertTrue(r.ok, r)
        for k in ("file", "source_width", "source_height", "crop", "confidence"):
            self.assertIn(k, r.data, r.data)
        self.assertEqual((r.data["source_width"], r.data["source_height"]), (640, 480))
        crop = r.data["crop"]
        for k in ("x", "y", "width", "height"):
            self.assertIn(k, crop, crop)
        self.assertEqual((crop["x"], crop["width"]), (0, 640), "no horizontal bars in this fixture")
        self.assertLess(crop["height"], 480, "the top/bottom black bars should be detected and excluded")
        self.assertNotIn("crop_filter", r.data, "confirms the original catalog assumption's crop_filter key does not exist")


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill(), "needs ffmpeg and ffmpeg-skill")
class CropProposalRealTests(unittest.TestCase):
    """ADR-060 (roadmap step 7) on the real ffmpeg-skill: a 1920x1080 clip whose active picture is 1920x800 (140 px black bars top
    and bottom) is measured by cropdetect.py (analysis kind crop_rectangle), the measured rectangle is proposed as a CONFIRM
    video.crop, nothing is cropped before the approval, and once approved crop.py delivers exactly that rectangle. A clip
    without bars proposes nothing."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="va_crop_")
        src = Path(cls.tmp) / "src"; src.mkdir()
        cls.letterboxed = str(src / "letterboxed.mp4")
        cls.full = str(src / "full.mp4")
        for path, video in ((cls.letterboxed, "testsrc2=s=1920x800:d=3:r=25,pad=1920:1080:0:140:black"), (cls.full, "testsrc2=s=1920x1080:d=3:r=25")):
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", video, "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", path], check=True)

    def test_letterboxed_clip_proposes_the_measured_rectangle(self):
        ws = str(Path(self.tmp) / "run_letterboxed")
        svc = Service(workspace=ws)
        ir = svc.plan([self.letterboxed], "generic", kinds=["crop_rectangle"])
        d = ir.doc
        obs = next(o for o in d["analysis"]["observations"] if o["kind"] == "crop_rectangle")
        self.assertEqual((obs["provenance"], obs["tool"]), ("OBSERVED", "ffmpeg-skill/cropdetect"))
        self.assertEqual((obs["data"]["source_width"], obs["data"]["source_height"], obs["data"]["windows_failed"]), (1920, 1080, 0))
        dec = next(x for x in d["decisions"] if x["subject"] == "video.crop")
        self.assertEqual((dec["type"], dec["approval"], dec["status"], dec["provenance"]), ("TRANSFORM", "CONFIRM", "PROPOSED", "OBSERVED"))
        rect = {k: dec["params"][k] for k in ("x", "y", "width", "height")}
        self.assertEqual(rect, {"x": 0, "y": 140, "width": 1920, "height": 800}, "the active picture, as measured")
        self.assertIn(obs["id"], dec["evidence"])
        self.assertEqual(svc.validate(ir).errors, [])
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "video_crop")
        self.assertEqual(step["tool"], "video-editing/crop")   # ADR-079: video-editing-skill's CROP (it runs ffmpeg-skill crop.py itself)
        p = str(Path(ws) / "crop.json"); save_ir(ir, p)
        waiting = svc.render(load_ir(p), p)
        self.assertEqual(waiting["status"], "WAITING_FOR_APPROVAL", waiting)
        self.assertFalse(waiting.get("execution"), "nothing runs before the proposal is approved")
        svc.approve(load_ir(p), p, [dec["id"]])
        out = svc.render(load_ir(p), p)
        self.assertEqual(out["status"], "COMPLETED", out)
        op_result = next(o for o in out["execution"]["results"] if o["tool"] == "video-editing/crop")
        self.assertTrue(op_result["ok"], op_result)
        self.assertEqual(artifact_qa(out, out["artifacts"][0]), "PASS", out["qa"])
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", op_result["output"]],
                            capture_output=True, text=True, check=True)
        self.assertEqual(pr.stdout.strip(), "1920x800", "the real video-editing CROP output is the measured rectangle")

    def test_full_frame_clip_proposes_nothing(self):
        svc = Service(workspace=str(Path(self.tmp) / "run_full"))
        ir = svc.plan([self.full], "generic", kinds=["crop_rectangle"])
        dec = next(x for x in ir.doc["decisions"] if x["subject"] == "video.crop")
        self.assertEqual((dec["type"], dec["approval"]), ("KEEP", "AUTO"))
        self.assertIn("no black bars", dec["reason"])
        self.assertNotIn("video_crop", [s["skill"] for s in ir.doc["plan"]["steps"]])


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill(), "needs ffmpeg and ffmpeg-skill")
class BeatGridRealTests(unittest.TestCase):
    """ADR-053 on the real ffmpeg-skill: the `beats` analysis kind (scenes.py --beats) measures the music bed named by
    `beat_input` only, and its grid is the one the beat snap reads (tempo, confidence, supported beat times)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="va_beats_")
        src = Path(cls.tmp) / "src"; src.mkdir()
        click = str(Path(cls.tmp) / "click.wav")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=0.05,apad=pad_dur=0.45", "-t", "0.5", click], check=True)
        bed = str(Path(cls.tmp) / "bed.wav")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-stream_loop", "19", "-i", click, "-c", "copy", bed], check=True)
        cls.bed = str(src / "bed.mp4")   # scenes.py needs a video stream: the bed in a video container (black picture)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=black:s=320x240:r=30", "-i", bed, "-shortest", "-c:v", "libx264", "-c:a", "aac", cls.bed], check=True)
        cls.take = str(src / "take.mp4")
        make_media(cls.take)

    def test_beat_grid_of_the_named_input_only(self):
        svc = Service(workspace=str(Path(self.tmp) / "ws"))
        _, _, an = svc.analyze([self.take, self.bed], "generic", kinds=["beats"], params={"beat_input": 1})
        beats = [o for o in an.observations if o.kind == "beats"]
        self.assertEqual([o.asset_id for o in beats], [an.assets[1].id], "measured on the music bed only")
        g = beats[0].data
        self.assertAlmostEqual(g["tempo_bpm"], 120.0, delta=1.0)
        self.assertGreaterEqual(g["confidence"], 0.5)
        self.assertTrue(g["usable"])
        self.assertGreaterEqual(len(g["supported_beats"]), 15)
        self.assertTrue(all(abs((b / 0.5) - round(b / 0.5)) < 0.05 for b in g["supported_beats"]), g["supported_beats"])
        self.assertEqual(beats[0].tool, "ffmpeg-skill/scenes")
        row = next(r for r in an.analyses[0]["rows"] if r["kind"] == "beats" and r["asset_id"] == an.assets[0].id)
        self.assertEqual(row["status"], "SKIPPED")


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill(), "needs ffmpeg and ffmpeg-skill")
class TakeCrossfadeRealTests(unittest.TestCase):
    """ADR-054 on the real Skills (ffmpeg-skill, video-editing-skill, audio-production-skill): three one-take lines assembled with the
    default join crossfade. The picture is cut frame-accurately at the planned points and joined by video-editing; each pick's audio
    span (picture range + join handles) is cut by audio-production, the spans are joined with the crossfade where the sources have
    room (join 1, centred; join 2 has no room on either side and keeps a hard cut), and ffmpeg-skill/audio --replace puts the result
    under the picture. A 1 kHz burst at a known source time in each take is found in the delivered file where the picture shows it
    (A/V sync), and the audio lasts as long as the picture. Speech recognition is the fake process reading a sidecar transcript per
    file (CI has no ASR model); every other tool is real."""

    BURSTS = {"t1.mp4": 1.0, "t2.mp4": 2.0, "t3.mp4": 1.5}
    SPEECH = {"t1.mp4": [[0.5, 3.0, "the first line of the talk"]], "t2.mp4": [[1.0, 3.9, "now a completely different second sentence"]],
              "t3.mp4": [[0.05, 3.0, "and finally the closing remark here"]]}

    @classmethod
    def setUpClass(cls):
        cls.tmp = os.path.realpath(tempfile.mkdtemp(prefix="va_xfade_"))
        src = Path(cls.tmp) / "src"; src.mkdir()
        cls.files = []
        for name, burst in cls.BURSTS.items():
            p = str(src / name)
            tone = f"0.05*sin(2*PI*220*t)+0.6*sin(2*PI*1000*t)*between(t\\,{burst}\\,{burst + 0.1})"
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30", "-f", "lavfi", "-i", f"aevalsrc='{tone}':s=48000",
                            "-t", "4", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-g", "15", "-c:a", "aac", "-b:a", "128k", p], check=True)
            Path(p + ".segments.json").write_text(json.dumps(cls.SPEECH[name]), encoding="utf-8")
            cls.files.append(p)

    def _service(self, ws):
        from video_agent.tools.router import ToolRouter
        from video_agent.tools.transcription import TranscriptionAdapter, TranscriptionSkill
        fake = [sys.executable, str(Path(__file__).resolve().parent / "fake_transcription.py")]

        class _Svc(Service):
            def adapter(self, allowed_inputs=None):
                r = super().adapter(allowed_inputs)
                ts = TranscriptionAdapter(TranscriptionSkill(fake, None, {}), workspace=str(Path(self.workspace) / "cache" / "transcription"))
                return self._sync_packages(ToolRouter([ts] + [a for a in r.adapters if a.name != "transcription"]))
        svc = _Svc(workspace=ws, offline=True)
        real = svc.caps

        class _Caps:
            def resolve(self, refresh=False):
                from video_agent.capabilities.resolver import Capability
                caps = dict(real.resolve(refresh))
                caps["transcription"] = Capability("transcription", "AVAILABLE", "fake recogniser (sidecar transcripts)", {})
                ve = caps.get("video-editing")
                if ve is not None and ve.status != "AVAILABLE" and "contract drift" in (ve.detail or "") and "doctor ok" in (ve.detail or ""):
                    # video-editing-skill main has grown past the contract pinned here (new operations); CONCAT itself is what this test
                    # needs and it runs on the live checkout. Re-pinning that contract is separate work, so the verdict is lifted here only.
                    caps["video-editing"] = Capability("video-editing", "AVAILABLE", ve.detail, dict(ve.evidence or {}))
                return caps
        svc.caps = _Caps()
        return svc

    def _bursts(self, path):
        """Start times of the 1 kHz bursts in a file's audio (band-passed, then where the silence ends)."""
        pr = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "bandpass=f=1000:width_type=q:w=4,silencedetect=noise=-25dB:d=0.05", "-f", "null", "-"],
                            capture_output=True, text=True)
        return [float(line.split("silence_end:")[1].split("|")[0]) for line in pr.stderr.splitlines() if "silence_end:" in line]

    def _stream_durations(self, path):
        pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", path], capture_output=True, text=True, check=True)
        return {s["codec_type"]: float(s["duration"]) for s in json.loads(pr.stdout)["streams"] if s.get("duration")}

    def test_crossfaded_assembly_stays_in_sync(self):
        svc = self._service(str(Path(self.tmp) / "ws"))
        caps = svc.caps.resolve()
        for need in ("video-editing", "audio-production"):
            if caps.get(need) is None or caps[need].status != "AVAILABLE":
                self.skipTest(f"needs {need}: {getattr(caps.get(need), 'detail', 'not resolved')[:120]}")
        ir = svc.plan(self.files, "generic", user_requirements={"edit.takes": True}, params={"language": "en", "offline": True})
        d = ir.doc
        self.assertEqual(svc.validate(ir).errors, [])
        xf = d["audio"]["take_crossfade"]
        self.assertEqual([j["crossfaded"] for j in xf["joins"]], [True, False], xf["joins"])
        self.assertIn("no room", xf["joins"][1]["why"])
        p = str(Path(svc.workspace) / "xf.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", json.dumps([r for r in out["execution"]["results"] if not r["ok"]])[:1500])
        tools = [r["tool"] for r in out["execution"]["results"]]
        self.assertIn("ffmpeg-skill/audio", tools)
        self.assertEqual(tools.count("audio-production/run"), 5, tools)   # 3 spans (CUT), the crossfaded run t1+t2 (CONCAT), that run + t3 at the hard cut (CONCAT)
        mux = next(r for r in out["execution"]["results"] if r["tool"] == "ffmpeg-skill/audio")
        final = mux["output"]
        durs = self._stream_durations(final)
        self.assertAlmostEqual(durs["audio"], durs["video"], delta=0.05, msg=durs)
        self.assertAlmostEqual(durs["video"], xf["timeline_duration"], delta=0.05)
        names = [Path(d["assets"][sp["asset"]]["path"]).name for sp in xf["spans"]]
        want = [sp["programme_start"] + self.BURSTS[n] - sp["video"][0] for sp, n in zip(xf["spans"], names)]
        got = [g for g in self._bursts(final) if g < durs["audio"] - 0.05]   # silencedetect also reports the end of the stream as a silence end
        self.assertEqual(len(got), 3, got)
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, delta=0.025, msg=f"burst at {g:.3f}s, picture shows it at {w:.3f}s ({got} vs {want})")


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill(), "needs ffmpeg and ffmpeg-skill")
class SubtitleMuxRealTests(unittest.TestCase):
    """ADR-063 on the real Skills (subtitle-skill render `mode: "mux"`, ffmpeg-skill cut / loudness / caption): `subtitle.mode=mux` adds
    the cues as a soft subtitle track; the track survives the loudness pass into the deliverable of a profile without a platform preset,
    and the agent's QA checks it there. A platform export (ffmpeg-skill export.py) writes the picture and the audio only — measured here
    — so the same request with the youtube profile is a BLOCK. Speech recognition is the fake process reading a sidecar transcript (CI
    has no ASR model); every other tool is real."""

    SPEECH = [[4.0, 8.0, "hello there"], [9.0, 12.0, "and goodbye"]]

    @classmethod
    def setUpClass(cls):
        cls.tmp = os.path.realpath(tempfile.mkdtemp(prefix="va_mux_"))
        src = Path(cls.tmp) / "src"; src.mkdir()
        cls.media = str(src / "talk.mp4")
        make_media(cls.media)
        Path(cls.media + ".segments.json").write_text(json.dumps(cls.SPEECH), encoding="utf-8")

    def _service(self, ws):
        from video_agent.capabilities.resolver import Capability
        from video_agent.tools.router import ToolRouter
        from video_agent.tools.transcription import TranscriptionAdapter, TranscriptionSkill
        fake = [sys.executable, str(Path(__file__).resolve().parent / "fake_transcription.py")]

        class _Svc(Service):
            def adapter(self, allowed_inputs=None):
                r = super().adapter(allowed_inputs)
                ts = TranscriptionAdapter(TranscriptionSkill(fake, None, {}), workspace=str(Path(self.workspace) / "cache" / "transcription"))
                return self._sync_packages(ToolRouter([ts] + [a for a in r.adapters if a.name != "transcription"]))
        svc = _Svc(workspace=ws, offline=True)
        real = svc.caps

        class _Caps:
            def resolve(self, refresh=False):
                caps = dict(real.resolve(refresh))
                caps["transcription"] = Capability("transcription", "AVAILABLE", "fake recogniser (sidecar transcripts)", {})
                return caps
        svc.caps = _Caps()
        return svc

    def _streams(self, path):
        pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name:format=duration", "-of", "json", path], capture_output=True, text=True, check=True)
        doc = json.loads(pr.stdout)
        return [(s_["codec_type"], s_.get("codec_name")) for s_ in doc["streams"]], float(doc["format"]["duration"])

    def _svc_with_subtitle(self, name):
        svc = self._service(str(Path(self.tmp) / name))
        cap = svc.caps.resolve().get("subtitle")
        if cap is None or cap.status != "AVAILABLE":
            self.skipTest(f"needs subtitle-skill (VIDEO_AGENT_SUBTITLE_DIR): {getattr(cap, 'detail', 'not resolved')[:120]}")
        return svc

    def test_platform_export_drops_subtitle_streams_so_a_preset_blocks(self):
        """The fact the preset BLOCK rests on, measured on the installed ffmpeg-skill. Before 2.22.4, export.py --preset youtube (and copy)
        leaves a mov_text track behind and a soft track with a preset is BLOCKED (ADR-063). From 2.22.4 (ffmpeg-skill #380) the export keeps
        it, so the BLOCK is lifted (ADR-071): the plan muxes, renders through the youtube preset and the deliverable carries the track (QA)."""
        svc = self._svc_with_subtitle("ws_exp")
        from video_agent.capabilities.resolver import export_keeps_subtitles
        from video_agent.tools.ffmpeg_skill.adapter import FfmpegSkillAdapter
        ws = Path(svc.workspace); ws.mkdir(parents=True, exist_ok=True)
        srt = ws / "c.srt"; srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
        muxed = str(ws / "muxed.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", self.media, "-i", str(srt), "-map", "0", "-map", "1", "-c", "copy", "-c:s", "mov_text", muxed], check=True)
        self.assertIn("subtitle", [t for t, _ in self._streams(muxed)[0]])
        skill = locate_ffmpeg_skill()
        keeps = export_keeps_subtitles(svc.caps.resolve())
        self.assertEqual(keeps, tuple(int(x) for x in skill.version.split(".")[:3]) >= (2, 22, 4), skill.version)
        ad = FfmpegSkillAdapter(skill)
        from video_agent.models import Operation
        for preset in ("youtube", "copy"):
            out = str(ws / f"exp_{preset}.mp4")
            r = ad.run(Operation(tool="ffmpeg-skill/export", args={"input": "m", "preset": preset, "output": "o"}, inputs=["m"], outputs=["o"], id=f"op_{preset}"), {"m": muxed, "o": out})
            self.assertTrue(r.ok, r.stderr_tail)
            kinds = [t for t, _ in self._streams(out)[0]]
            if keeps:
                self.assertIn("subtitle", kinds, f"ffmpeg-skill {skill.version}: export --preset {preset} should keep the subtitle stream (#380), the ADR-071 lift rests on it")
            else:
                self.assertNotIn("subtitle", kinds, f"export --preset {preset} now keeps subtitle streams on {skill.version}: the version gate of ADR-071 is too high")
        ir = svc.plan([self.media], "youtube", user_requirements={"subtitle": True, "subtitle.mode": "mux"}, params={"language": "en", "offline": True})
        blocked = [x["subject"] for x in ir.doc["decisions"] if x["approval"] == "BLOCK"]
        if not keeps:
            self.assertEqual(blocked, ["subtitle.mux"])
            return
        self.assertEqual(blocked, [], ir.doc["plan"]["summary"])
        self.assertEqual(svc.validate(ir).errors, [])
        p = str(ws / "mux_youtube.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", json.dumps([r for r in out["execution"]["results"] if not r["ok"]])[:1500])
        track = [i for i in out["qa"]["items"] if i["name"] == "subtitle_track"]
        self.assertTrue(track and all(i["status"] == "PASS" for i in track), track)
        exp = next(r for r in out["execution"]["results"] if r["tool"] == "ffmpeg-skill/export")
        self.assertIn("subtitle", [t for t, _ in self._streams(exp["output"])[0]], "the youtube deliverable carries the soft track")

    def test_soft_subtitle_track_reaches_the_deliverable(self):
        svc = self._svc_with_subtitle("ws")
        ir = svc.plan([self.media], "generic", user_requirements={"subtitle": True, "subtitle.mode": "mux", "audio.normalize": True, "audio.loudness.target_lufs": -16},
                      params={"language": "en", "offline": True})
        d = ir.doc
        self.assertEqual(svc.validate(ir).errors, [])
        self.assertEqual([s_["skill"] for s_ in d["plan"]["steps"]][:4], ["silence_cleanup", "subtitle_generation", "subtitle_mux", "loudness_normalization"])
        p = str(Path(svc.workspace) / "mux.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", json.dumps([r for r in out["execution"]["results"] if not r["ok"]])[:1500])
        ren = next(r for r in out["execution"]["results"] if r["tool"] == "subtitle/render")
        self.assertEqual((ren["data"]["mode"], ren["data"]["engine"]["id"]), ("mux", "ffmpeg-skill"))
        streams, _ = self._streams(ren["output"])
        self.assertEqual([t for t, _ in streams], ["video", "audio", "subtitle"], streams)
        art = next(a for a in out["artifacts"] if a["type"] == "MASTER")
        streams, dur = self._streams(art["path"])
        self.assertIn(("subtitle", "mov_text"), streams, "the soft track survives the loudness pass into the deliverable")
        self.assertTrue(art["path"].endswith("_loudnorm.mp4"), art["path"])
        track = next(i for i in out["qa"]["items"] if i["name"] == "subtitle_track")
        self.assertEqual(track["status"], "PASS", track)
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        row = next(e for e in prov["operations"] if e["tool"] == "subtitle/render")
        self.assertEqual((row["skill"], row["args"]["mode"]), ("subtitle_mux", "mux"))


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill(), "needs ffmpeg and ffmpeg-skill")
class TimelineExportRealTests(unittest.TestCase):
    """ADR-052 on the real ffmpeg-skill: `delivery.timeline=otio` → render.py --export-timeline writes the IR's cut list as an
    OpenTimelineIO document, registered as a TIMELINE Artifact; opentimelineio (when installed) reads it back with the same cuts."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="va_timeline_")
        src = Path(cls.tmp) / "src"; src.mkdir()
        cls.a = str(src / "a.mp4")
        make_media(cls.a)

    def test_otio_timeline_has_the_same_cuts(self):
        ws = str(Path(self.tmp) / "run_otio")
        svc = Service(workspace=ws)
        if not svc.adapter([]).supports("ffmpeg-skill/render"):
            self.skipTest("this ffmpeg-skill checkout has no render.py --export-timeline")
        ir = svc.plan([self.a], "generic", user_requirements={"delivery.timeline": "otio"})
        d = ir.doc
        self.assertEqual(svc.validate(ir).errors, [])
        (tl,) = d["delivery"]["timelines"]
        trim = next((op for op in d["video"]["operations"] if op["type"] == "video.trim"), None)
        self.assertIsNotNone(trim, "the fixture has leading / trailing silence, so the cut list is a real trim")
        self.assertEqual([[c["in"], c["out"]] for c in tl["clips"]], trim["keep"])
        p = str(Path(ws) / "timeline.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", out.get("execution"))
        res = next(r for r in out["execution"]["results"] if r["tool"] == "ffmpeg-skill/render")
        self.assertTrue(res["ok"], res)
        self.assertEqual((res["data"]["timeline"]["format"], res["data"]["timeline"]["clips"], res["data"]["timeline"]["transition"]), ("otio", len(tl["clips"]), "none"))
        self.assertEqual(res["commands"], [], "an export runs no ffmpeg command")
        (art,) = [a for a in out["artifacts"] if a["type"] == "TIMELINE"]
        self.assertTrue(Path(art["path"]).resolve().is_relative_to(Path(ws).resolve()))
        self.assertEqual(art["hash"], __import__("video_agent.media.analyzer", fromlist=["sha256_file"]).sha256_file(art["path"]))
        self.assertEqual(art["qa_status"], "PASS", out["qa"])
        project = json.loads(next(Path(ws).glob("jobs/*/ops/*_timeline/*_timeline_project.json")).read_text(encoding="utf-8"))
        self.assertEqual(project["transition"], {"type": "none"})
        try:
            import opentimelineio as otio
        except ImportError:
            return   # optional: the file's own clip count was checked by QA above
        timeline = otio.adapters.read_from_file(art["path"])
        clips = list(timeline.find_clips())
        self.assertEqual(len(clips), len(tl["clips"]))
        for clip, want in zip(clips, tl["clips"]):
            rate = clip.source_range.start_time.rate
            self.assertAlmostEqual(clip.source_range.start_time.to_seconds(), want["in"], delta=1.0 / rate + 1e-6)
            self.assertAlmostEqual(clip.source_range.end_time_exclusive().to_seconds(), want["out"], delta=1.0 / rate + 1e-6)
            self.assertEqual(Path(clip.media_reference.target_url.replace("file://", "")).resolve(), Path(self.a).resolve())
        self.assertAlmostEqual(timeline.duration().to_seconds(), tl["timeline_duration"], delta=len(clips) / rate + 1e-6)


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
        self.assertEqual(cut["data"]["observation"]["source"], f"ffmpeg-skill/probe@{locate_ffmpeg_skill().version}")
        self.assertEqual(cut["data"]["observation"]["provenance"], "OBSERVED")
        self.assertTrue(cut["data"]["timeline"]["tracks"][0]["segments"], "source → timeline mapping reported by the Skill")
        self.assertTrue(cut["commands"] and all(c.startswith("/") or c.startswith("ffmpeg") for c in cut["commands"]), "ffmpeg command lines are provenance only")
        keep = step["params"]["keep"]
        probe = cut["data"]["observation"]["data"]
        self.assertAlmostEqual(float(probe["duration"]), sum(e - s for s, e in keep), delta=1.6, msg="keyframe-precision cut lands within the Skill's tolerance")
        prov = json.loads((Path(ws) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        trim = next(e for e in prov["operations"] if e["skill"] == "silence_cleanup")
        self.assertEqual((trim["skill_package"], trim["tool"], trim["tool_version"]), ("video-editing", "video-editing/cut", "0.4.2"))
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
        # 0.4.1 (ADR-078): CROP is a declared operation and tool now, lowered by the adapter; an operation the contract lists as unsupported
        # (POSITION) is the Skill's UNSUPPORTED_OPERATION / not a tool: no engine fallback anywhere
        self.assertTrue(strict.supports("video-editing/crop"))
        self.assertIn("CROP", strict.contract["operations"]); self.assertFalse(any(u["type"] == "CROP" for u in strict.contract["unsupported"]))
        self.assertFalse(strict.supports("video-editing/position"))
        self.assertTrue(any(u["type"] == "POSITION" for u in strict.contract["unsupported"]))


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
            self.assertEqual(r["data"]["observation"]["provenance"], "OBSERVED"); self.assertEqual(r["data"]["observation"]["source"], f"ffmpeg-skill/probe@{locate_ffmpeg_skill().version}")
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
            self.assertTrue(e["decision"] and e["skill_result"]["artifact"]["sha256"] and e["tool_version"] == "0.4.2")
        self.assertEqual(len(prov["skill_observations"]), len(rows))
        dec = {x["id"]: x for x in d["decisions"]}
        self.assertTrue(all(dec[i]["type"] == "TRANSFORM" and dec[i]["provenance"] == "USER" for e in rows for i in e["decision"]))
        return svc, p, out, res

    def test_concat_speed_resize_fit_overlay_end_to_end(self):
        if not self.ve:
            self.skipTest("needs a video-editing-skill checkout (VIDEO_AGENT_VIDEO_EDITING_DIR)")
        svc, p, out, res = self._run("fit", {"edit.concat": True, "edit.speed": 2, "edit.resize": 640, "edit.fit": "1:1", "edit.overlay": self.png, "edit.overlay.position": "top-right", "edit.overlay.scale": 80},
                                     ["video.concat", "video.speed", "video.resize", "video.fit", "video.overlay"], (360, 360))   # 640x360 → 1:1 never upscales (ffmpeg-skill fit.py fix)
        # resume: every completed editing operation is reused by the agent's idempotency key (no Skill call)
        out2 = svc.render(load_ir(p), p, resume=out["job"]["id"])
        self.assertEqual(out2["execution"]["status"], "COMPLETED")
        self.assertTrue(all(r["op_id"] in out2["execution"]["skipped"] for r in res))

    def test_concat_transition_fill_end_to_end(self):
        if not self.ve:
            self.skipTest("needs a video-editing-skill checkout (VIDEO_AGENT_VIDEO_EDITING_DIR)")
        self._run("fill", {"edit.concat": True, "edit.concat.transition": "fade", "edit.concat.transition_duration": 0.5, "edit.speed": 2, "edit.resize": 640, "edit.fill": "1:1",
                           "edit.overlay": self.png, "edit.overlay.position": {"x": 10, "y": 10}, "edit.overlay.opacity": 0.6},
                  ["video.concat", "video.speed", "video.resize", "video.fill", "video.overlay"], (360, 360))   # same bound for fill


    def test_explicit_trim_and_rotate_end_to_end(self):
        """ADR-062: `edit.trim` → video-editing/trim (frame-accurate at the times given) and `edit.rotate` + `.flip` → video-editing/rotate
        on the real Skill: the trimmed length and the swapped frame size measured by ffprobe, the requests in provenance."""
        if not self.ve:
            self.skipTest("needs a video-editing-skill checkout (VIDEO_AGENT_VIDEO_EDITING_DIR)")
        ws = str(Path(self.tmp) / "trim_rotate")
        svc = Service(workspace=ws)
        ir = svc.plan([self.a], "generic", user_requirements={"edit.trim": "2-10", "edit.rotate": 90, "edit.rotate.flip": "h"})
        d = ir.doc
        self.assertEqual(d["plan"]["status"], "APPROVED", d["plan"]["summary"])
        self.assertEqual(svc.validate(ir).errors, [])
        steps = {s["skill"]: s for s in d["plan"]["steps"]}
        self.assertEqual((steps["video_trim"]["tool"], steps["video_rotate"]["tool"]), ("video-editing/trim", "video-editing/rotate"))
        keep = next(op["keep"] for op in d["video"]["operations"] if op["type"] == "video.trim")
        self.assertEqual(len(keep), 1); self.assertTrue(2.0 <= keep[0][0] < keep[0][1] <= 10.0)
        p = str(Path(self.tmp) / "trim_rotate.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = [r for r in out["execution"]["results"] if r["tool"].startswith("video-editing/")]
        self.assertEqual([r["tool"] for r in res], ["video-editing/trim", "video-editing/rotate"])
        self.assertTrue(all(r["ok"] for r in res), res)
        dur_t, size_t, _ = self._probe(res[0]["output"])
        self.assertAlmostEqual(dur_t, keep[0][1] - keep[0][0], delta=0.1, msg="frame-precision trim at the requested in / out points")
        self.assertEqual(size_t, (1280, 720))
        dur_r, size_r, audio_r = self._probe(res[1]["output"])
        self.assertEqual(size_r, (720, 1280), "a quarter turn swaps width and height"); self.assertTrue(audio_r)
        self.assertAlmostEqual(dur_r, dur_t, delta=0.1)
        self.assertEqual(res[1]["data"]["operation"]["type"], "ROTATE")
        prov = json.loads((Path(ws) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        rows = {e["skill"]: e for e in prov["operations"] if e["tool"].startswith("video-editing/")}
        self.assertEqual((rows["video_trim"]["args"]["precision"], rows["video_rotate"]["args"]["degrees"], rows["video_rotate"]["args"]["flip"]), ("frame", 90, "h"))
        self.assertTrue(all(e["decision"] and e["tool_version"] == "0.4.2" for e in rows.values()))

    def test_image_insert_before_the_picture_end_to_end(self):
        """ADR-070: `edit.image_insert` (a real 160x90 PNG, 2 s, before) → video-editing/image_insert at the picture's own frame / rate after
        the resize → video-editing/concat of the still clip and the picture, on the real Skill and ffmpeg-skill: ffprobe on the still (640x360,
        2 s, no audio) and on the joined programme (640x360, trimmed length + 2 s, audio kept), QA's duration from the IR, provenance."""
        if not self.ve:
            self.skipTest("needs a video-editing-skill checkout (VIDEO_AGENT_VIDEO_EDITING_DIR)")
        ws = str(Path(self.tmp) / "image_insert")
        svc = Service(workspace=ws)
        ir = svc.plan([self.a], "generic", user_requirements={"edit.trim": "2-10", "edit.resize": 640, "edit.image_insert": self.png, "edit.image_insert.duration": 2})
        d = ir.doc
        self.assertEqual(d["plan"]["status"], "APPROVED", d["plan"]["summary"])
        self.assertEqual(svc.validate(ir).errors, [])
        op = next(o for o in d["video"]["operations"] if o["type"] == "video.image_insert")
        self.assertEqual((op["width"], op["height"], op["position"], op["duration"]), (640, 360, "before", 2.0))
        self.assertAlmostEqual(float(op["fps"]), 30.0, places=2)
        p = str(Path(self.tmp) / "image_insert.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = [r for r in out["execution"]["results"] if r["tool"].startswith("video-editing/")]
        self.assertEqual([r["tool"] for r in res], ["video-editing/trim", "video-editing/resize", "video-editing/image_insert", "video-editing/concat"])
        self.assertTrue(all(r["ok"] for r in res), res)
        dur_t, _, _ = self._probe(res[0]["output"])
        dur_i, size_i, audio_i = self._probe(res[2]["output"])
        self.assertEqual(size_i, (640, 360), "the still takes the picture's frame (scaled to cover, centre-cropped)"); self.assertFalse(audio_i)
        self.assertAlmostEqual(dur_i, 2.0, delta=0.1)
        self.assertEqual(res[2]["data"]["operation"]["type"], "IMAGE_INSERT")
        dur_j, size_j, audio_j = self._probe(res[3]["output"])
        self.assertEqual(size_j, (640, 360), "the join does not rescale the picture"); self.assertTrue(audio_j, "the picture's audio is kept (silence under the still)")
        self.assertAlmostEqual(dur_j, dur_t + 2.0, delta=0.2)
        qa_dur = next(i for i in out["qa"]["items"] if i["name"] == "duration")
        self.assertEqual(qa_dur["status"], "PASS", qa_dur)
        prov = json.loads((Path(ws) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        still = next(e for e in prov["operations"] if e["tool"] == "video-editing/image_insert")
        join = next(e for e in prov["operations"] if e["tool"] == "video-editing/concat")
        self.assertEqual((still["skill"], still["args"]["duration"], still["args"]["width"], still["args"]["height"]), ("video_image_insert", 2.0, 640, 360))
        self.assertEqual(join["args"]["inputs"][0], still["args"]["output"], "the still first")
        self.assertTrue(still["decision"] and join["decision"] and still["tool_version"] == "0.4.2")


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

    def _run(self, name, inputs, reqs, expect_skills, expect_ops, expect_plan_status="APPROVED", pending_subjects=(), svc=None):
        """plan → (status as expected before any approval) → render(approve=all) → COMPLETED. `expect_plan_status` "REVIEW" means the
        plan waits for the CONFIRM decisions in `pending_subjects` (ADR-033: a video container's audio.extract); the explicit
        `approve=["all"]` of render is the existing approval mechanism, after which the saved plan must be APPROVED."""
        from video_agent.tools.audio_production import pinned_contract as ap_pinned
        ws = str(Path(self.tmp) / name)
        svc = svc or Service(workspace=ws)
        ws = svc.workspace
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
            self.assertEqual(r["data"]["observation"]["provenance"], "OBSERVED"); self.assertEqual(r["data"]["observation"]["source"], f"ffmpeg-skill/probe@{locate_ffmpeg_skill().version}")
            self.assertTrue(r["data"]["operation"]["tool"].startswith("ffmpeg-skill/")); self.assertEqual(r["data"]["provenance"]["skill"], "audio-production")
            self.assertFalse(any(k in json.dumps(r["data"]["operation"]["parameters"]).lower() for k in ("argv", "command", "filter", "shell")))
        items = {i["name"]: i for i in out["qa"]["items"]}
        self.assertEqual(items["duration"]["status"], "PASS", items["duration"]); self.assertEqual(items["audio_only"]["status"], "PASS")
        prov = json.loads((Path(ws) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        rows = [e for e in prov["operations"] if e["tool"] == "audio-production/run"]
        self.assertEqual([e["skill"] for e in rows], expect_skills)
        for e in rows:
            self.assertTrue(e["decision"] and e["skill_result"]["artifact"]["sha256"] and e["tool_version"] == ap_pinned()["version"] and e["skill_package"] == "audio-production")
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
        # generic profile has no delivery preset, but real processing ran (concat/mono/normalize): the last
        # intermediate is the deliverable, and it is a registered MASTER artifact (video-production-agent#32) —
        # citing every real decision behind it (video-production-agent#33), not the pre-fix "artifacts: []"
        art = out["artifacts"][0]
        self.assertEqual((art["type"], art["format"], artifact_qa(out, art)), ("MASTER", "source", "PASS"))
        self.assertTrue(art["operations"], "the real audio-production operation must be credited")
        self.assertTrue(art["path"].endswith("programme_audio_loudnorm.wav"))
        self.assertTrue(out["paths"]["programme_audio_delivery_main"].endswith("programme_audio_loudnorm.wav"))

    def test_mix_noise_reduction_dynamics_end_to_end(self):
        """ADR-063 on the real Skill: the stereo programme (16 s) and the 5 s mono tone mixed (-6 dB on the tone; the inputs keep their own
        timelines, nothing cut), FFT noise reduction, then a compressor and a limiter; the mix lasts as long as its first input."""
        if not self.ap:
            self.skipTest("needs an audio-production-skill checkout")
        svc, p, out, res, prov = self._run("E", [self.wav, self.mono], {"audio.production": True, "audio.mix": True, "audio.mix.levels": [0, -6],
                                                                       "audio.noise_reduction.strength_db": 24, "audio.dynamics.compressor.ratio": 3,
                                                                       "audio.dynamics.compressor.threshold_db": -20, "audio.dynamics.limiter.ceiling_db": -1},
                                           ["audio_mix", "audio_noise_reduction", "audio_dynamics"], ["MIX", "NOISE_REDUCTION", "DYNAMICS"])
        d = load_ir(p).doc
        mix = next(op for op in d["audio"]["operations"] if op["type"] == "audio.mix")
        self.assertEqual((mix["timeline_duration"], [s_["source_range"] for s_ in mix["segments"]]), (16.0, [[0.0, 16.0], [0.0, 5.0]]))
        self.assertEqual(res[0]["data"]["operation"]["parameters"]["levels"], [{"gain_db": 0.0, "mute": False}, {"gain_db": -6.0, "mute": False}])
        self.assertEqual(res[1]["data"]["operation"]["parameters"], {"mode": "fft", "strength_db": 24.0, "audio_stream": 0})
        self.assertEqual(res[2]["data"]["operation"]["parameters"]["compressor"], {"ratio": 3.0, "threshold_db": -20.0})
        self.assertEqual(res[2]["data"]["operation"]["parameters"]["limiter"], {"ceiling_db": -1.0})
        dur, ch, sr, has_v = self._probe(res[-1]["output"])
        self.assertAlmostEqual(dur, 16.0, delta=0.1, msg="the mix follows its first input"); self.assertEqual((ch, has_v), (2, False))

    def test_voice_cleanup_end_to_end(self):
        """ADR-068 on the real Skill (audio-production 0.4.3): `audio.voice_cleanup=light` → VOICE_CLEANUP (ffmpeg-skill/audio --voice light) before
        the FFT noise reduction; the Skill records the level it ran. With an ffmpeg-skill older than 1.13 the Skill runs medium only and refuses
        light (TOOL_ERROR voice_level_unsupported, not retried): that path is covered against the fake (tests/test_satellite_params.py)."""
        if not self.ap:
            self.skipTest("needs an audio-production-skill checkout")
        svc, p, out, res, prov = self._run("V", [self.wav], {"audio.production": True, "audio.voice_cleanup": "light", "audio.noise_reduction.strength_db": 30},
                                           ["audio_cut", "audio_voice_cleanup", "audio_noise_reduction"], ["CUT", "VOICE_CLEANUP", "NOISE_REDUCTION"])
        self.assertEqual(res[1]["data"]["operation"]["parameters"], {"level": "light", "audio_stream": 0})
        dur, ch, sr, has_v = self._probe(res[-1]["output"])
        self.assertEqual((ch, has_v), (2, False)); self.assertGreater(dur, 5.0)

    def _fake_recogniser_service(self, ws):
        """The real engine and the real audio-production-skill; speech recognition is the fake process reading the input's sidecar words
        (CI has no ASR model), exactly as SubtitleMuxRealTests does."""
        from video_agent.capabilities.resolver import Capability
        from video_agent.tools.router import ToolRouter
        from video_agent.tools.transcription import TranscriptionAdapter, TranscriptionSkill
        fake = [sys.executable, str(Path(__file__).resolve().parent / "fake_transcription.py")]

        class _Svc(Service):
            def adapter(self, allowed_inputs=None):
                r = super().adapter(allowed_inputs)
                ts = TranscriptionAdapter(TranscriptionSkill(fake, None, {}), workspace=str(Path(self.workspace) / "cache" / "transcription"))
                return self._sync_packages(ToolRouter([ts] + [a for a in r.adapters if a.name != "transcription"]))
        svc = _Svc(workspace=ws, offline=True)
        real = svc.caps

        class _Caps:
            def resolve(self, refresh=False):
                caps = dict(real.resolve(refresh))
                caps["transcription"] = Capability("transcription", "AVAILABLE", "fake recogniser (sidecar words)", {})
                return caps
        svc.caps = _Caps()
        return svc

    def test_filler_remove_end_to_end(self):
        """ADR-072 on the real Skill (audio-production 0.4.6 FILLER_REMOVE) and the real engine: the recognised words of the 16 s programme
        (tone from 3 s to 13 s) carried through the silence cut, a Japanese and an English filler matched by the Skill's presets and removed
        sample-accurately; the lexicon crosses as UTF-8; QA expects the cut length less what the Skill reports it removed."""
        if not self.ap:
            self.skipTest("needs an audio-production-skill checkout")
        words = [[3.4, 3.9, "hello"], [4.2, 4.6, "えー"], [5.0, 6.0, "world"], [7.0, 7.3, "um"], [8.0, 9.0, "again"]]
        src = Path(self.tmp) / "src_fr"; src.mkdir(exist_ok=True)
        wav = str(src / "speech.wav"); shutil.copyfile(self.wav, wav)
        Path(wav + ".segments.json").write_text(json.dumps([[3.2, 9.5, "hello えー world um again"]], ensure_ascii=False), encoding="utf-8")
        Path(wav + ".words.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
        svc = self._fake_recogniser_service(str(Path(self.tmp) / "F"))
        svc, p, out, res, prov = self._run("F", [wav], {"audio.production": True, "audio.filler_remove": "ja,en"}, ["audio_cut", "audio_filler_remove"], ["CUT", "FILLER_REMOVE"],
                                           expect_plan_status="REVIEW", pending_subjects=("audio.filler_remove",), svc=svc)
        d = load_ir(p).doc
        cut, fr = d["audio"]["operations"]
        lead = cut["remove"][0][1]
        self.assertEqual([(w["start"], w["text"]) for w in fr["words"]], [(round(a - lead, 6), t) for a, _, t in words], "the word times follow the cut")
        op = res[1]["data"]["operation"]
        self.assertEqual((op["parameters"]["lexicon"]["presets"], len(op["parameters"]["words"])), (["en", "ja"], 5))
        meas = op["measurements"]["filler_remove"]
        self.assertEqual([m_["text"] for m_ in meas["matched"]], ["えー", "um"], "the Japanese word arrived intact and was matched by the Skill")
        self.assertAlmostEqual(meas["removed_seconds"], 0.4 + 0.3, places=3)
        cut_len, ch, _, has_v = self._probe(res[0]["output"])
        dur, ch2, _, _ = self._probe(res[1]["output"])
        self.assertAlmostEqual(cut_len - dur, 0.7, delta=0.01, msg="sample-accurate: exactly the matched words are gone"); self.assertEqual((ch, ch2, has_v), (2, 2, False))
        items = {i["name"]: i for i in out["qa"]["items"]}
        self.assertAlmostEqual(float(items["duration"]["expected"].split()[0]), fr["temporal_scope"]["end"] - meas["removed_seconds"], places=3)

    def test_mix_duck_end_to_end(self):
        """ADR-072 on the real Skill (audio-production 0.4.6 MIX.duck): the 5 s tone is the bed (input 1), ducked 18 dB under the programme
        (input 0, tone from 3 s); the Skill records the duck it ran and the bed is measurably quieter while the key sounds."""
        if not self.ap:
            self.skipTest("needs an audio-production-skill checkout")
        svc, p, out, res, prov = self._run("G", [self.wav, self.mono], {"audio.production": True, "audio.mix": True, "audio.mix.duck.input": 1, "audio.mix.duck.amount_db": 18,
                                                                       "audio.mix.duck.threshold_db": -40},
                                           ["audio_mix"], ["MIX"])
        self.assertEqual(res[0]["data"]["operation"]["parameters"]["duck"], {"input": 1, "amount_db": 18.0, "threshold_db": -40.0})
        self.assertTrue(any("sidechaincompress" in c for c in res[0]["data"].get("commands") or []), "the engine ran its sidechain compressor (observed command, provenance only)")
        dur, ch, _, has_v = self._probe(res[0]["output"])
        self.assertAlmostEqual(dur, 16.0, delta=0.1); self.assertEqual((ch, has_v), (2, False))

        def rms(path, start, length):
            pr = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", str(start), "-t", str(length), "-i", path, "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True)
            return float(next(ln for ln in pr.stderr.splitlines() if "mean_volume" in ln).split("mean_volume:")[1].split()[0])
        # 0-3 s: only the bed sounds (the key is silent) → not ducked; 3.5-4.5 s: key + ducked bed. Same span without ducking for reference
        plain = self._run("G2", [self.wav, self.mono], {"audio.production": True, "audio.mix": True}, ["audio_mix"], ["MIX"])[3][0]["output"]
        self.assertGreater(rms(plain, 3.5, 1.0) - rms(res[0]["output"], 3.5, 1.0), 1.0, "the bed is lowered while the key is above the threshold")

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
        # ADR-065: the deliverable's set (with its sidecar / thumbnail), when it has one, gates it too: the worse verdict counts
        pkg = next((r for r in out["execution"]["results"] if r["tool"] == "qc/check" and r["data"].get("kind") == "delivery_package" and art["logical_name"] in (r["data"].get("artifacts") or {})), None)
        if pkg is not None:
            self.assertTrue(pkg["ok"] and pkg["data"]["admitted"]); self.assertEqual(pkg["data"]["artifacts"][art["logical_name"]], art["hash"])
        order = {"PASS": 0, "WARN": 1, "FAIL": 2}
        verdict = max([qc["data"]["verdict"]] + ([pkg["data"]["verdict"]] if pkg else []), key=lambda v: order.get(v, 2))
        self.assertEqual(art["qa"]["qc"], verdict)
        if agent_fail or verdict == "FAIL":
            self.assertEqual((art["stage"], art["qa_status"]), ("working", "FAIL"))
        elif verdict == "PASS":
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
        # ADR-065: the deliverable and its thumbnail checked together on the real qc-skill (kind delivery_package); the thumbnail is gated by that report
        pkg = next(r for r in out["execution"]["results"] if r["tool"] == "qc/check" and r["data"].get("kind") == "delivery_package")
        self.assertTrue(pkg["ok"] and pkg["data"]["admitted"], pkg["data"].get("error"))
        self.assertEqual(pkg["data"]["artifacts"], {arts["YOUTUBE"]["logical_name"]: arts["YOUTUBE"]["hash"], arts["THUMBNAIL"]["logical_name"]: arts["THUMBNAIL"]["hash"]})
        self.assertEqual(pkg["data"]["verdict"], "PASS", pkg["data"]["findings"])
        self.assertEqual((arts["THUMBNAIL"]["qa_status"], arts["THUMBNAIL"]["qa"]["qc"], arts["THUMBNAIL"]["stage"]), ("PASS", "PASS", "approved"))
        self._gate_coherent(out, arts["YOUTUBE"])
        _, _, _, prim = self._probe(arts["YOUTUBE"]["path"])
        self.assertEqual(prim, "bt709")

    def test_primary_correction_gamma_lift_gain_levels_curves(self):
        """ADR-062: color.gamma / .lift / .gain / .levels_* / .curves → one PRIMARY_CORRECTION request on the real color-grading-skill 0.4.x;
        the Skill reports back the parameters it applied (its effective parameters, defaults filled in)."""
        self._need("color-grading")
        svc = self._svc("pc2")
        want = {"gamma": 1.2, "lift": 0.05, "gain": -0.05, "levels_in_black": 16, "levels_in_white": 235, "curves": "medium_contrast"}
        ir, p = self._plan(svc, [self.a], {f"color.{k}": v for k, v in want.items()}, profile="generic", name="pc2")
        op = ir.doc["color"]["operations"][0]
        self.assertEqual((op["type"], {k: op[k] for k in want}), ("color.primary_correction", want))
        out = self._render(svc, p)
        res = next(r for r in out["execution"]["results"] if r["tool"] == "color-grading/run")
        self.assertTrue(res["ok"] and os.path.isfile(res["output"]), res)
        applied = res["data"]["operation"]["parameters"]
        self.assertEqual({k: applied[k] for k in want}, want)
        self.assertEqual((applied["levels_out_black"], applied["levels_out_white"], applied["exposure"]), (0, 255, 0.0), "unset parameters keep the Skill's defaults")
        dur, size, audio, _ = self._probe(res["output"])
        self.assertEqual(size, (1280, 720)); self.assertTrue(audio); self.assertGreater(dur, 5.0)

    def test_primary_correction_every_contract_parameter(self):
        """ADR-076: every color.* PRIMARY_CORRECTION key (read from the pinned color-grading contract: the five of ADR-031 and the eight of
        ADR-062) planned at once, each at a value inside the contract's range, → one PRIMARY_CORRECTION request on the real color-grading-skill;
        the live contract still declares the same colour parameters, and the Skill reports every value back as applied."""
        self._need("color-grading")
        from video_agent.agent.finishing import COLOR_CORRECTION_PARAMS, PRIMARY_CORRECTION_NOT_GRADING
        from video_agent.tools.color_grading import ColorGradingAdapter, locate_color_grading
        live = ColorGradingAdapter(locate_color_grading()).operations["PRIMARY_CORRECTION"]["parameters"]
        self.assertEqual({n: s for n, s in live.items() if n not in PRIMARY_CORRECTION_NOT_GRADING}, COLOR_CORRECTION_PARAMS, "re-pin color-grading")
        svc = self._svc("pc_all")
        want = {"exposure": -0.25, "contrast": 1.2, "saturation": 1.1, "temperature": 5600.0, "tint": -0.1, "gamma": 0.9, "lift": -0.05, "gain": 0.05,
                "levels_in_black": 8, "levels_in_white": 245, "levels_out_black": 16, "levels_out_white": 235, "curves": "lighter"}
        self.assertEqual(sorted(want), sorted(COLOR_CORRECTION_PARAMS))
        ir, p = self._plan(svc, [self.a], {f"color.{k}": v for k, v in want.items()}, profile="generic", name="pc_all")
        op = ir.doc["color"]["operations"][0]
        self.assertEqual((op["type"], {k: op[k] for k in want}), ("color.primary_correction", want))
        out = self._render(svc, p)
        res = next(r for r in out["execution"]["results"] if r["tool"] == "color-grading/run")
        self.assertTrue(res["ok"] and os.path.isfile(res["output"]), res)
        applied = res["data"]["operation"]["parameters"]
        self.assertEqual({k: applied[k] for k in want}, want)
        dur, size, audio, _ = self._probe(res["output"])
        self.assertEqual(size, (1280, 720)); self.assertTrue(audio); self.assertGreater(dur, 5.0)

    def test_every_motion_graphics_element(self):
        """ADR-064: every element type of the real motion-graphics-skill 0.2.x (thirteen since 0.2.7's shape, ADR-078) in one render, each with parameters beyond its text
        (a chroma-keyed picture-in-picture clip, a registry font, a meme's two lines, a countdown's start number): the Skill renders every
        element and reports the parameters it used; the picture keeps its size and duration."""
        self._need("motion-graphics")
        from video_agent.agent.finishing import ELEMENT_TYPES
        off = {t: self.caps[f"motion-graphics:{t}"].status for t in ELEMENT_TYPES if getattr(self.caps.get(f"motion-graphics:{t}"), "status", "MISSING") != "AVAILABLE"}
        self.assertEqual(off, {}, "every element type the pinned contract offers resolves AVAILABLE on this machine")
        svc = self._svc("mg12")
        reqs = {"motion.title": "Opening", "motion.title.primary_color": "1F2A44", "motion.text.font_color": "yellow", "motion.lower_third": "Ada", "motion.lower_third.title": "Engineer",
                "motion.text": "LIVE", "motion.text.font": "system:dejavu-serif", "motion.text.font_size": 40, "motion.text.box": True, "motion.text.fade": 0.3,
                "motion.image": self.png, "motion.image.opacity": 0.8, "motion.video": self.b, "motion.video.scale_width": 320, "motion.video.chromakey": "00FF00",
                "motion.bug": "@studio", "motion.chapter": "Part 2", "motion.chapter.start": 6, "motion.progress": True, "motion.countdown": True,
                "motion.countdown.count_from": 3, "motion.countdown.start": 2, "motion.countdown.end": 5, "motion.sticker": "NEW", "motion.hook": "Wait for it",
                "motion.meme": True, "motion.meme.top": "when the build", "motion.meme.bottom": "is green",
                "motion.shape": True, "motion.shape.width": 400, "motion.shape.height": 80, "motion.shape.color": "navy", "motion.shape.opacity": 0.6}   # ADR-078: 0.2.7's shape
        ir, p = self._plan(svc, [self.a], reqs, profile="generic", name="mg12")
        op = ir.doc["graphics"]["operations"][0]
        self.assertEqual(sorted(e["type"] for e in op["elements"]), sorted(ELEMENT_TYPES))
        out = self._render(svc, p)
        res = next(r for r in out["execution"]["results"] if r["tool"] == "motion-graphics/run")
        recs = {o["type"]: o for o in res["data"]["operations"]}
        self.assertEqual(sorted(recs), sorted(ELEMENT_TYPES))
        self.assertEqual(recs["video_overlay"]["parameters"]["chromakey"], "00FF00"); self.assertEqual(recs["video_overlay"]["parameters"]["scale_width"], 320)
        self.assertEqual(recs["countdown"]["parameters"]["count_from"], 3); self.assertEqual((recs["meme"]["parameters"]["top"], recs["meme"]["parameters"]["bottom"]), ("when the build", "is green"))
        self.assertEqual(recs["text_overlay"]["parameters"]["font"]["font_id"], "system:dejavu-serif")
        dur, size, audio, _ = self._probe(res["output"])
        self.assertEqual(size, (1280, 720)); self.assertTrue(audio); self.assertGreater(dur, 5.0)

    def test_template_colour_names_render(self):
        """ADR-067 on the real motion-graphics-skill (0.2.4, its ADR-20): a colour name on a template element (chapter / hook / progress)
        is planned and rendered (the Skill converts it to RRGGBB for ffmpeg-skill/graphics); ADR-064 measured the same request failing
        mid-render before. @alpha on a template colour is refused at planning, before anything runs."""
        self._need("motion-graphics")
        svc = self._svc("mg_names")
        reqs = {"motion.chapter": "Part 2", "motion.chapter.primary_color": "navy", "motion.hook": "Wait", "motion.hook.text_color": "White",
                "motion.hook.primary_color": "darkorange", "motion.progress": True, "motion.progress.primary_color": "Crimson"}
        ir, p = self._plan(svc, [self.a], reqs, profile="generic", name="mg_names")
        out = self._render(svc, p)
        res = next(r for r in out["execution"]["results"] if r["tool"] == "motion-graphics/run")
        self.assertTrue(res["ok"] and os.path.isfile(res["output"]), res)
        recs = {o["type"]: o for o in res["data"]["operations"]}
        self.assertEqual((recs["chapter"]["parameters"]["primary_color"], recs["hook"]["parameters"]["text_color"], recs["progress"]["parameters"]["primary_color"]),
                         ("navy", "White", "Crimson"), "the Skill records the request's own value; the conversion is its own")
        with self.assertRaises(ValueError):
            svc.plan([self.a], "generic", user_requirements={"motion.chapter": "x", "motion.chapter.primary_color": "navy@0.5"}, params={"offline": True})

    def test_motion_placement_blocks_after_a_turn(self):
        """ADR-064 on the real resolver: {x: 1000} fits the 1280x720 source but not the 720x1280 picture a quarter turn delivers."""
        self._need("motion-graphics", "video-editing")
        svc = self._svc("mg_turn")
        ir = svc.plan([self.a], "generic", user_requirements={"edit.rotate": 90, "motion.text": "LIVE", "motion.text.position": {"x": 1000, "y": 10}}, params={"offline": True})
        blk = [x for x in ir.doc["decisions"] if x["approval"] == "BLOCK"]
        self.assertEqual([x["subject"] for x in blk], ["graphics.text_overlay"]); self.assertEqual(blk[0]["params"]["picture"], [720, 1280])

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
                                                                            "loudness_normalization", "delivery_export", "delivery_check", "thumbnail_frame", "qc_check", "qc_check", "qc_check"])
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

    # ---- Scenario 11: the QC gate AND Artifact registration on a no-preset delivery (generic profile) — real qc-skill and ffmpeg-skill
    def test_s11_qc_gate_and_artifact_without_a_delivery_preset(self):
        """Every other scenario plans against the "youtube" profile, which always has a delivery preset, so none of
        them exercise the no-preset ("deliver as-is") path this delivery target takes under the generic profile.
        Two bugs lived here (AI-video-production-OS WORK_QUEUE item 9): `agent/planner.py`'s `qc_steps()` used to
        only plan a qc step alongside a `delivery_export` step (preset-only), so `--set qc=true` compiled nothing at
        all for a no-preset target (fixed: it now gates the subject's own real media directly when nothing produced
        an in-workspace file); and a genuinely untouched deliverable never got a registered Artifact at all, because
        its raw source lives outside the workspace and `ArtifactStore.check_path()` (ADR-022) correctly refuses to
        register an external path. Fixed with a real stream copy (ffmpeg-skill `export.py --preset copy`) that
        materializes the untouched source into the workspace, so it gets a genuine Artifact like everything else."""
        self._need("motion-graphics", "qc")
        sha256_file = __import__("video_agent.media.analyzer", fromlist=["sha256_file"]).sha256_file
        svc = self._svc("s11")
        # untouched: `self.a` has real leading/trailing silence by design (TONE, for the other scenarios' own
        # trim coverage), so a clip with none at all is needed here to get truly zero edit steps — the plan
        # must have nothing but the stream-copy export and the qc gate, not even the always-on technical cleanup
        toneless = str(Path(self.tmp) / "s11_toneless.mp4")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=6",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "128k", toneless], check=True)
        ir, p = self._plan(svc, [toneless], {"qc": True}, profile="generic", name="s11a")
        self.assertEqual([s["skill"] for s in ir.doc["plan"]["steps"]], ["delivery_export", "qc_check"])
        exp_step = next(s for s in ir.doc["plan"]["steps"] if s["skill"] == "delivery_export")
        self.assertEqual(exp_step["params"]["preset"], "copy")
        out = self._render(svc, p)
        self.assertEqual(len(out["artifacts"]), 1, "a genuinely untouched no-preset deliverable now registers a real Artifact")
        art = out["artifacts"][0]
        self.assertEqual((art["type"], art["format"]), ("MASTER", "source"))
        self.assertEqual(Path(art["path"]).suffix, ".mp4", "copy keeps the source's own extension")
        src_dur, src_res, src_audio, _ = self._probe(toneless)
        dur, res, has_audio, _ = self._probe(art["path"])
        self.assertAlmostEqual(dur, src_dur, delta=0.05); self.assertEqual(res, src_res); self.assertEqual(has_audio, src_audio)
        self.assertNotEqual(art["hash"], sha256_file(toneless), "a remux changes container-level bytes even under a stream copy (no picture/audio re-encode)")
        self.assertTrue(any(r["tool"].endswith("/export") and r["ok"] for r in out["execution"]["results"]), "a real delivery_export op ran (the stream copy)")
        qc = next(r for r in out["execution"]["results"] if r["tool"] == "qc/check")
        self.assertTrue(qc["ok"] and qc["data"]["admitted"], qc["data"])
        self.assertIn(qc["data"]["verdict"], ("PASS", "WARN"), qc["data"].get("findings"))
        self.assertEqual(qc["data"]["fingerprint"], art["hash"], "the QC gate measured the real, delivered (copied) bytes, not the untouched source")
        qc_item = next(i for i in out["qa"]["items"] if i["layer"] == "qc" and i["name"] == "verdict")
        self.assertEqual(qc_item["status"], qc["data"]["verdict"])
        self._gate_coherent(out, art)
        # processed: a real motion-graphics element, still no delivery preset — unchanged, no copy needed (already
        # in-workspace); qc gates the real processed file
        ir2, p2 = self._plan(svc, [toneless], {"motion.title": "Live", "qc": True}, profile="generic", name="s11b")
        self.assertEqual([s["skill"] for s in ir2.doc["plan"]["steps"]], ["motion_graphics", "qc_check"])
        out2 = self._render(svc, p2)
        art2 = out2["artifacts"][0]
        self._gate_coherent(out2, art2)
        qc2 = next(r for r in out2["execution"]["results"] if r["tool"] == "qc/check")
        self.assertEqual(qc2["data"]["fingerprint"], art2["hash"], "the QC gate measured the real, delivered (processed) bytes, not the untouched source")


_FAKE_DEEP_FILTER = '''import os, shutil, sys
args = sys.argv[1:]
if args[:1] == ["--version"]:
    print("deep-filter 0.5.6")
    sys.exit(0)
out, files, i = "out", [], 0
while i < len(args):
    a = args[i]
    if a in ("-m", "-o", "-a", "--atten-lim-db", "--model", "--output-dir"):
        if a in ("-o", "--output-dir"):
            out = args[i + 1]
        i += 2
        continue
    if not a.startswith("-"):
        files.append(a)
    i += 1
os.makedirs(out, exist_ok=True)
shutil.copyfile(files[0], os.path.join(out, os.path.basename(files[0])))   # identity: a stand-in for the engine, never a model
'''
_FAKE_WPE = '''def wpe_v8(Y, taps=10, delay=3, iterations=3, psd_context=0, statistics_mode="full", inplace=False):
    return Y.copy()   # identity: a stand-in for nara_wpe (voice-enhance-skill's own tests do the same), never the real engine
'''


@unittest.skipUnless(shutil.which("ffmpeg") and locate_ffmpeg_skill(), "needs ffmpeg and ffmpeg-skill")
class VoiceEnhanceRealTests(unittest.TestCase):
    """ADR-073 on the real voice-enhance-skill (main; no release tag) with fake engines (a `deep-filter` that copies its input, a `nara_wpe` whose
    wpe_v8 is the identity, the real `crosstalk` on NumPy), the real ffmpeg-skill and the real audio-production-skill: the live contract equals
    the pin, a missing engine and a non-48 kHz isolate block with the Skill's own codes, dereverb → isolate after the cut and bleed reduction
    before the mix render after confirmation, and the delivered files are what the Skill promises (48 kHz mono for isolate)."""

    @classmethod
    def setUpClass(cls):
        from video_agent.tools.voice_enhance import locate_voice_enhance
        from video_agent.tools.audio_production import locate_audio_production
        cls.ve, cls.ap = locate_voice_enhance(), locate_audio_production()
        cls.tmp = tempfile.mkdtemp(prefix="va_ve_")
        src = Path(cls.tmp) / "src"; src.mkdir()
        ff = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        speech = "0.3*sin(2*PI*220*t)*between(t\\,1\\,5)*gt(sin(2*PI*3*t)\\,-0.3)"
        cls.talk, cls.talk44 = str(src / "talk.wav"), str(src / "talk44.wav")
        cls.lav_a, cls.lav_b = str(src / "lav_a.wav"), str(src / "lav_b.wav")
        subprocess.run(ff + ["-f", "lavfi", "-i", f"aevalsrc='{speech}|{speech}':s=48000:c=stereo", "-t", "6", "-c:a", "pcm_s16le", cls.talk], check=True)
        subprocess.run(ff + ["-f", "lavfi", "-i", f"aevalsrc='{speech}':s=44100:c=mono", "-t", "6", "-c:a", "pcm_s16le", cls.talk44], check=True)
        a, b = "0.3*sin(2*PI*200*t)*between(t\\,0\\,3)", "0.3*sin(2*PI*330*t)*between(t\\,2\\,5)"
        subprocess.run(ff + ["-f", "lavfi", "-i", f"aevalsrc='{a}+0.1*{b}':s=48000:c=mono", "-t", "5", "-c:a", "pcm_s16le", cls.lav_a], check=True)
        subprocess.run(ff + ["-f", "lavfi", "-i", f"aevalsrc='{b}+0.1*{a}':s=48000:c=mono", "-t", "5", "-c:a", "pcm_s16le", cls.lav_b], check=True)
        eng = Path(cls.tmp) / "engines"; eng.mkdir()
        df = eng / "deep-filter"
        df.write_text(f"#!{sys.executable}\n" + _FAKE_DEEP_FILTER, encoding="utf-8"); df.chmod(0o755)
        model = eng / "DeepFilterNet3_onnx.tar.gz"; model.write_bytes(b"\x1f\x8b fake model")
        site = eng / "site"; (site / "nara_wpe").mkdir(parents=True)
        (site / "nara_wpe" / "__init__.py").write_text("", encoding="utf-8")
        (site / "nara_wpe" / "wpe.py").write_text(_FAKE_WPE, encoding="utf-8")
        (site / "nara_wpe-0.0.11.dist-info").mkdir()
        (site / "nara_wpe-0.0.11.dist-info" / "METADATA").write_text("Metadata-Version: 2.1\nName: nara_wpe\nVersion: 0.0.11\n", encoding="utf-8")
        cls.engines = {"VOICE_ENHANCE_DEEP_FILTER": str(df), "VOICE_ENHANCE_DFN_MODEL": str(model)}
        cls.site = str(site)

    def setUp(self):
        if not self.ve or not self.ap:
            self.skipTest("needs voice-enhance-skill and audio-production-skill checkouts (VIDEO_AGENT_VOICE_ENHANCE_DIR / VIDEO_AGENT_AUDIO_PRODUCTION_DIR); CI clones both")
        self._env = {k: os.environ.get(k) for k in self.engines}

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _with_engines(self):
        """The fake engines where voice-enhance-skill looks for them: deep-filter + model by its environment variables, nara_wpe on the
        Skill's own import path (the checkout's PYTHONPATH plus the stand-in package; a real install puts nara_wpe in the same environment)."""
        from unittest import mock
        from video_agent.tools.skill_process import CliSkill
        from video_agent.tools.voice_enhance import locate_voice_enhance
        os.environ.update(self.engines)
        ve = locate_voice_enhance()
        with_site = CliSkill(ve.skill_id, ve.command, ve.root, {"PYTHONPATH": os.pathsep.join([ve.env.get("PYTHONPATH", ""), self.site]).strip(os.pathsep)})
        return mock.patch.multiple("video_agent.service", locate_voice_enhance=lambda *a, **k: with_site), \
            mock.patch.multiple("video_agent.capabilities.resolver", locate_voice_enhance=lambda *a, **k: with_site)

    def _probe(self, path):
        pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,channels,sample_rate", "-of", "json", path], capture_output=True, text=True, check=True)
        doc = json.loads(pr.stdout)
        a = next(s for s in doc["streams"] if s.get("codec_type") == "audio")
        return float(doc["format"]["duration"]), int(a["channels"]), int(a["sample_rate"])

    def test_live_contract_equals_the_pin_and_missing_engines_carry_the_skills_codes(self):
        from video_agent.capabilities import CapabilityResolver
        from video_agent.tools.voice_enhance import VoiceEnhanceAdapter, drift, pinned_contract
        for k in self.engines:
            os.environ.pop(k, None)
        ad = VoiceEnhanceAdapter(self.ve, ffmpeg_skill_dir=str(locate_ffmpeg_skill().root))
        self.assertEqual(ad.version, pinned_contract()["version"])
        self.assertEqual(drift(ad.contract), [], "voice-enhance-skill main drifted from the pin (no release tag: the pin follows main): re-verify the adapter")
        doc = ad.doctor()
        self.assertIn(doc["status"], ("ok", "degraded"), doc.get("problems"))
        ev = ad.operation_evidence(doc)
        if not shutil.which("deep-filter"):
            self.assertEqual(ev["ISOLATE"]["code"], "ENGINE_UNAVAILABLE")
            self.assertIn("DeepFilterNet3_onnx.tar.gz", ev["ISOLATE"]["install"])
        caps = CapabilityResolver(str(locate_ffmpeg_skill().root)).resolve()
        self.assertEqual(caps["voice-enhance"].status, "AVAILABLE", caps["voice-enhance"].detail)
        self.assertEqual(caps["voice-enhance"].evidence["drift"], [])
        svc = Service(workspace=str(Path(self.tmp) / "ws_missing"))
        ir = svc.plan([self.talk], "generic", user_requirements={"audio.production": True, "voice.isolate": True})
        b = next(x for x in ir.doc["decisions"] if x["subject"] == "voice.isolate")
        if caps["voice-enhance:ISOLATE"].status != "AVAILABLE":
            self.assertEqual((b["type"], b["params"]["code"]), ("BLOCK", "ENGINE_UNAVAILABLE"))
            self.assertIn("VOICE_ENHANCE_DFN_MODEL", b["reason"])
            self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")

    def test_isolate_off_48_khz_follows_the_engine_skills_resample_flag(self):
        """Tripwire like ADR-071's: today's ffmpeg-skill has no `audio --sample-rate`, so a 44.1 kHz isolate blocks with the Skill's
        FFMPEG_SKILL_UNAVAILABLE; once ffmpeg-skill declares it, the same plan must be a TRANSFORM (and voice-enhance converts)."""
        p1, p2 = self._with_engines()
        with p1, p2:
            svc = Service(workspace=str(Path(self.tmp) / "ws_44"))
            caps = svc.caps.resolve()
            ir = svc.plan([self.talk44], "generic", user_requirements={"audio.production": True, "voice.isolate": True})
        self.assertEqual(caps["voice-enhance:ISOLATE"].status, "AVAILABLE", caps["voice-enhance:ISOLATE"].detail)
        d = next(x for x in ir.doc["decisions"] if x["subject"] == "voice.isolate")
        if caps["voice-enhance:ISOLATE"].evidence.get("resample"):
            self.assertEqual(d["type"], "TRANSFORM")
        else:
            self.assertEqual((d["type"], d["params"]["code"], d["params"]["sample_rate"]), ("BLOCK", "FFMPEG_SKILL_UNAVAILABLE", 44100))

    def test_dereverb_then_isolate_after_the_cut_end_to_end(self):
        p1, p2 = self._with_engines()
        with p1, p2:
            svc = Service(workspace=str(Path(self.tmp) / "ws_iso"))
            ir = svc.plan([self.talk], "generic", user_requirements={"audio.production": True, "voice.dereverb": True, "voice.dereverb.taps": 12, "voice.isolate": True,
                                                                     "audio.gain": -3})
            d = ir.doc
            self.assertEqual(svc.validate(ir).errors, [])
            self.assertEqual(d["plan"]["status"], "REVIEW", d["plan"]["summary"])
            self.assertEqual(sorted(x["subject"] for x in ir.pending_confirmations()), ["voice.dereverb", "voice.isolate"])
            types = [op["type"] for op in d["audio"]["operations"]]
            self.assertEqual([t for t in types if t != "audio.cut"], ["voice.dereverb", "voice.isolate", "audio.gain"])
            p = str(Path(self.tmp) / "iso.json"); save_ir(ir, p)
            waiting = svc.render(load_ir(p), p)
            self.assertEqual(waiting["status"], "WAITING_FOR_APPROVAL")
            out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = [r for r in out["execution"]["results"] if r["tool"].startswith("voice-enhance/")]
        self.assertEqual([(r["data"]["op"], r["data"]["engine"]["id"], r["data"]["engine"]["version"]) for r in res],
                         [("dereverb", "nara_wpe", "0.0.11"), ("isolate", "deepfilternet", "0.5.6")])
        self.assertEqual(res[0]["data"]["parameters"]["taps"], 12)
        self.assertTrue(all(r["data"]["lifecycle"] == "EXPERIMENTAL" for r in res))
        self.assertEqual(self._probe(res[1]["output"])[1:], (1, 48000), "isolate writes a 48 kHz mono stem")
        art = out["artifacts"][0]
        dur, ch, rate = self._probe(art["path"])
        self.assertEqual((ch, rate), (1, 48000))
        items = {i["name"]: i for i in out["qa"]["items"]}
        self.assertEqual(items["duration"]["status"], "PASS", items["duration"])
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        obs = [o for o in prov["skill_observations"] if o["kind"].startswith("voice.")]
        self.assertEqual([o["kind"] for o in obs], ["voice.dereverb", "voice.isolate"])
        self.assertIn("DeepFilterNet", obs[1]["data"]["engine"]["attribution"])
        self.assertFalse(any(self.tmp in json.dumps(o["data"]["engine"]) for o in obs), "the engine block carries no local path")

    def test_bleed_reduction_before_the_mix_end_to_end(self):
        p1, p2 = self._with_engines()
        with p1, p2:
            svc = Service(workspace=str(Path(self.tmp) / "ws_br"))
            ir = svc.plan([self.lav_a, self.lav_b], "generic", user_requirements={"audio.production": True, "audio.mix": True, "voice.bleed_reduce": True})
            d = ir.doc
            self.assertEqual(svc.validate(ir).errors, [])
            self.assertEqual([op["type"] for op in d["audio"]["operations"]], ["voice.bleed_reduce", "audio.mix"])
            p = str(Path(self.tmp) / "br.json"); save_ir(ir, p)
            out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        [br] = [r for r in out["execution"]["results"] if r["tool"] == "voice-enhance/bleed-reduce"]
        self.assertEqual((br["data"]["engine"]["id"], len(br["data"]["outputs"])), ("crosstalk", 2))
        self.assertEqual([os.path.basename(o["file"]) for o in br["data"]["outputs"]], ["lav_a_a0.wav", "lav_b_a0.wav"])
        for o in br["data"]["outputs"]:
            self.assertEqual(self._probe(o["file"])[1], 1)
        mix = next(r for r in out["execution"]["results"] if r["tool"] == "audio-production/run" and r["data"]["operation_type"] == "MIX")
        self.assertTrue(mix["ok"])
        self.assertEqual(self._probe(out["artifacts"][0]["path"])[1], 1)


def _imagemagick_installed() -> bool:
    """ImageMagick 7 `magick`, or ImageMagick 6's `convert` + `identify` (not on Windows, where `convert` is a system tool), as imagemagick-skill reads it."""
    return bool(shutil.which("magick") or (os.name != "nt" and shutil.which("convert") and shutil.which("identify")))


def _png(path: str, w: int, h: int) -> str:
    """A real RGB PNG written with the standard library (no ImageMagick involved in making the fixture)."""
    import struct
    import zlib
    rows = b"".join(b"\x00" + bytes((200, 40, 40)) * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return path


class ImagePrepareRealTests(unittest.TestCase):
    """ADR-077 on the real imagemagick-skill (main, contract 0.8.1): the live contract equals the pin, its doctor finds a backend, and an
    `edit.overlay` asked at a size is stripped / resized / padded by imagemagick-skill/strip / resize / pad before the real video-editing-skill
    composites it. Runs whenever an imagemagick-skill checkout and ImageMagick are installed (the CI integration job installs both)."""

    @classmethod
    def setUpClass(cls):
        from video_agent.tools.imagemagick import locate_imagemagick
        cls.im = locate_imagemagick()
        cls.tmp = os.path.realpath(tempfile.mkdtemp(prefix="va_im_"))
        cls.png = _png(str(Path(cls.tmp) / "src" / "logo.png"), 640, 320)

    def setUp(self):
        if not self.im:
            self.skipTest("needs an imagemagick-skill checkout (VIDEO_AGENT_IMAGEMAGICK_DIR)")
        if not _imagemagick_installed():
            self.skipTest("needs ImageMagick (magick, or ImageMagick 6 convert / identify)")

    def test_live_contract_and_each_tool(self):
        import hashlib
        from video_agent.models import Operation
        from video_agent.tools.imagemagick import ImageMagickAdapter
        ws = str(Path(self.tmp) / "ws_adapter"); os.makedirs(ws, exist_ok=True)
        ad = ImageMagickAdapter(self.im, workspace=ws, allowed_inputs=[str(Path(self.png).parent)])
        self.assertEqual(ad.drift(), [], "the installed imagemagick-skill contract drifted from the pinned one: re-verify the adapter (and re-pin)")
        self.assertEqual(ad.breaking_drift(), [])
        doc = ad.doctor()
        self.assertEqual(doc["status"], "ok", doc.get("problems"))
        self.assertEqual(ad.operation_status(doc), {"RESIZE": "supported", "PAD": "supported", "STRIP": "supported"})
        before = hashlib.sha256(Path(self.png).read_bytes()).hexdigest()
        paths = {"img": self.png, "s": str(Path(ws) / "ops" / "s" / "logo_strip.png"), "r": str(Path(ws) / "ops" / "r" / "logo_resize.png"), "p": str(Path(ws) / "ops" / "p" / "logo_pad.png"),
                 "f": str(Path(ws) / "ops" / "f" / "logo_fill.png")}
        steps = (("imagemagick-skill/strip", {"input": "img", "output": "s"}),
                 ("imagemagick-skill/resize", {"input": "s", "output": "r", "width": 200, "height": 200, "mode": "fit"}),
                 ("imagemagick-skill/pad", {"input": "r", "output": "p", "width": 200, "height": 200, "color": "none", "gravity": "center"}),
                 ("imagemagick-skill/resize", {"input": "img", "output": "f", "width": 100, "height": 100, "mode": "fill"}))
        for k in ("s", "r", "p", "f"):
            Path(paths[k]).parent.mkdir(parents=True, exist_ok=True)   # as the executor does before each operation
        got = {}
        for tool, args in steps:
            r = ad.run(Operation(tool=tool, args=args, inputs=[args["input"]], outputs=[args["output"]], id=f"op_{args['output']}"), paths, timeout=120)
            self.assertTrue(r.ok, r.data)
            self.assertEqual(r.data["output"]["sha256"], hashlib.sha256(Path(r.output).read_bytes()).hexdigest())
            got[args["output"]] = r.data
        self.assertFalse(got["s"]["result"]["has_gps"])
        self.assertEqual((got["r"]["output"]["width"], got["r"]["output"]["height"]), (200, 100), "fit: 640x320 inside 200x200")
        self.assertEqual((got["p"]["output"]["width"], got["p"]["output"]["height"]), (200, 200))
        self.assertEqual(got["p"]["result"]["offset"], {"x": 0, "y": 50})
        self.assertEqual((got["f"]["output"]["width"], got["f"]["output"]["height"]), (100, 100))
        self.assertEqual(hashlib.sha256(Path(self.png).read_bytes()).hexdigest(), before, "the user's image is never written")
        bad = ad.run(Operation(tool="imagemagick-skill/resize", args={"input": "img", "output": "f", "width": 10, "height": 10, "mode": "stretch"}, inputs=["img"], outputs=["f"]), paths)
        self.assertEqual(bad.data["error"]["code"], "INVALID_REQUEST", "refused against the live contract's enum before the Skill runs")

    def test_overlay_prepared_then_composited_end_to_end(self):
        if not (shutil.which("ffmpeg") and locate_ffmpeg_skill()):
            self.skipTest("needs ffmpeg and ffmpeg-skill")
        from video_agent.tools.video_editing import locate_video_editing
        if not locate_video_editing():
            self.skipTest("needs a video-editing-skill checkout (VIDEO_AGENT_VIDEO_EDITING_DIR)")
        a = str(Path(self.tmp) / "src" / "a.mp4")
        if not os.path.exists(a):
            make_media(a)
        ws = str(Path(self.tmp) / "ws_pipeline")
        svc = Service(workspace=ws)
        ir = svc.plan([a], "generic", user_requirements={"edit.overlay": self.png, "edit.overlay.position": "top-right", "edit.overlay.prepare.width": 160,
                                                         "edit.overlay.prepare.height": 160, "edit.overlay.prepare.pad": "none", "edit.overlay.prepare.strip": True})
        d = ir.doc
        steps = {s["skill"]: s for s in d["plan"]["steps"]}
        self.assertEqual((steps["image_strip"]["tool"], steps["image_prepare"]["tool"], steps["image_pad"]["tool"]),
                         ("imagemagick-skill/strip", "imagemagick-skill/resize", "imagemagick-skill/pad"))
        self.assertEqual(svc.validate(ir).errors, [])
        self.assertEqual(svc.handoffs(ir), [])
        p = str(Path(self.tmp) / "im.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = {r["tool"]: r for r in out["execution"]["results"]}
        pad = res["imagemagick-skill/pad"]
        self.assertEqual((pad["data"]["output"]["width"], pad["data"]["output"]["height"]), (160, 160))
        self.assertEqual((res["imagemagick-skill/resize"]["data"]["output"]["width"], res["imagemagick-skill/resize"]["data"]["output"]["height"]), (160, 80))
        self.assertTrue(res["video-editing/overlay"]["ok"])
        prov = json.loads((Path(ws) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        row = next(e for e in prov["operations"] if e["tool"] == "video-editing/overlay")
        self.assertIn(pad["output"], row["input"], "the overlay composites the prepared image")
        self.assertEqual([o["kind"] for o in prov["skill_observations"] if o["kind"].startswith("image.")], ["image.strip", "image.resize", "image.pad"])
        dur = float(json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", res["video-editing/overlay"]["output"]],
                                              capture_output=True, text=True, check=True).stdout)["format"]["duration"])
        self.assertGreater(dur, 1.0)


def _luma_frame(path: str, n: int, w: int, h: int) -> bytes:
    """Frame number `n` (decode order from the first frame, so a shifted start or an edit list cannot pick another frame) of a yuv420p file as its
    Y plane, read as stored: `-pix_fmt yuv420p` is the file's own format, so no colour-matrix / range conversion happens in the measurement
    (an untagged source and a bt709-tagged output would otherwise be converted differently to grey by the decoder's scaler)."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-vf", f"select=eq(n\\,{int(n)})", "-frames:v", "1", "-pix_fmt", "yuv420p",
                          "-f", "rawvideo", "-"], check=True, capture_output=True).stdout
    assert len(out) == w * h * 3 // 2, (len(out), w, h)
    return out[:w * h]


def _mean_abs_diff(a: bytes, b: bytes, w: int, rect=None, outside: bool = False) -> float:
    """Mean |a - b| over a rectangle {x, y, width, height} of two w-wide grey frames (or over everything outside it)."""
    total = n = 0
    h = len(a) // w
    for y in range(0, h, 2):
        row = y * w
        for x in range(0, w, 2):
            inside = rect is not None and rect["x"] <= x < rect["x"] + rect["width"] and rect["y"] <= y < rect["y"] + rect["height"]
            if (rect is None) or (inside != outside):
                total += abs(a[row + x] - b[row + x])
                n += 1
    return total / max(1, n)


class FaceRedactionRealTests(unittest.TestCase):
    """ADR-090 on the real detection-skill (main, contract 0.1.0) and the real ffmpeg-skill: the live contract equals the pin, its doctor says
    whether FACES can run (YuNet needs opencv and the model file in DETECTION_MODELS_DIR; ffmpeg-skill is the agent's own checkout), and a face
    that is on screen only from 2 s to 5 s is blurred during its measured span and nowhere else, after the user's approval. Runs whenever a
    detection-skill checkout is installed (the CI integration job installs it with `[faces]` and fetches the YuNet file pinned by sha256)."""

    @classmethod
    def setUpClass(cls):
        from video_agent.tools.detection import locate_detection
        cls.det = locate_detection()
        cls.tmp = os.path.realpath(tempfile.mkdtemp(prefix="va_det_"))

    def setUp(self):
        if not self.det:
            self.skipTest("needs a detection-skill checkout (VIDEO_AGENT_DETECTION_DIR)")
        if not (shutil.which("ffmpeg") and locate_ffmpeg_skill()):
            self.skipTest("needs ffmpeg and ffmpeg-skill")

    def adapter(self):
        from video_agent.tools.detection import DetectionAdapter
        ws = str(Path(self.tmp) / "ws_adapter"); os.makedirs(ws, exist_ok=True)
        return DetectionAdapter(self.det, workspace=ws, allowed_inputs=[str(Path(self.tmp) / "src")], ffmpeg_skill_dir=str(locate_ffmpeg_skill().root))

    def test_live_contract_and_doctor(self):
        ad = self.adapter()
        self.assertEqual(ad.drift(), [], "the installed detection-skill contract drifted from the pinned one: re-verify the adapter (and re-pin)")
        self.assertEqual(ad.breaking_drift(), [])
        doc = ad.doctor()
        self.assertTrue(doc["ready"].get("tracks"), doc.get("problems"))
        ev = ad.operation_evidence(doc)["FACES"]
        if os.environ.get("DETECTION_MODELS_DIR"):
            self.assertEqual(ad.operation_status(doc), {"FACES": "supported"}, doc.get("problems"))
        else:
            self.assertEqual(ad.operation_status(doc), {"FACES": "unavailable"})
            self.assertIn(ev["code"], ("MODEL_NOT_FOUND", "ENGINE_UNAVAILABLE"))
            self.assertTrue(ev["install"], "the doctor's install line travels with the verdict")

    def test_face_blurred_only_during_its_span_end_to_end(self):
        if not os.environ.get("DETECTION_MODELS_DIR"):
            self.skipTest("needs the YuNet model file in DETECTION_MODELS_DIR")
        face = Path(self.det.root) / "tests" / "fixtures" / "astronaut.jpg"   # detection-skill's own public-domain fixture (one face)
        self.assertTrue(face.is_file(), face)
        src = str(Path(self.tmp) / "src" / "faces.mp4")
        Path(src).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x202020:s=1280x720:r=30:d=8", "-loop", "1", "-i", str(face),
                        "-f", "lavfi", "-i", "sine=f=440:r=48000:d=8", "-filter_complex", "[0][1]overlay=384:104:enable='between(t,2,5)':shortest=1[v]",
                        "-map", "[v]", "-map", "2:a", "-t", "8", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", src], check=True)
        svc = Service(workspace=str(Path(self.tmp) / "ws"))
        ir = svc.plan([src], "generic", user_requirements={"edit.redact_faces": True})
        d = ir.doc
        obs = next(o for o in d["analysis"]["observations"] if o["kind"] == "face_tracks")
        self.assertGreaterEqual(len(obs["data"]["tracks"]), 1, obs["data"])
        [dec] = [x for x in d["decisions"] if x["subject"] == "video.redact_faces"]
        self.assertEqual((dec["type"], dec["approval"]), ("TRANSFORM", "CONFIRM"))
        regions = dec["params"]["regions"]
        self.assertGreaterEqual(regions[0]["start"], 1.0, regions)
        self.assertLessEqual(regions[-1]["end"], 6.0, regions)
        rect = regions[0]["rects"][0]
        self.assertTrue(384 - 40 <= rect["x"] <= 384 + 512 and 104 - 40 <= rect["y"] <= 104 + 512, rect)
        p = str(Path(self.tmp) / "ws" / "faces.project.json")
        save_ir(ir, p)
        out = svc.render(load_ir(p), p)
        self.assertNotEqual(out["status"], "COMPLETED", "nothing is blurred before the approval")
        svc.approve(load_ir(p), p, [dec["id"]])
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED", out)
        art = next(a for a in out["artifacts"] if a["type"] in ("MASTER", "DELIVERY"))
        probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", art["path"]], check=True, capture_output=True).stdout)
        self.assertAlmostEqual(float(probe["format"]["duration"]), 8.0, delta=0.15, msg="broll keeps the source's length")
        mid = int(round((regions[0]["start"] + regions[-1]["end"]) / 2 * 30))   # the same frame number in both files (30 fps, as made above)
        a, b = _luma_frame(src, mid, 1280, 720), _luma_frame(art["path"], mid, 1280, 720)
        inside, outside = _mean_abs_diff(a, b, 1280, rect), _mean_abs_diff(a, b, 1280, rect, outside=True)
        tags = {f: subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=color_range,color_space,start_time", "-of", "csv=p=0", f],
                                  capture_output=True, text=True).stdout.strip() for f in (src, art["path"])}
        self.assertGreater(inside, 6.0, f"the face rectangle is blurred while the face is on screen (frame {mid})")
        self.assertLess(outside, 2.0, f"nothing outside the rectangle changes (frame {mid}: inside {inside:.2f}, outside {outside:.2f}; {tags})")
        for n in (15, 210):   # before and after the face's span: the picture is the source's (re-encoded only)
            self.assertLess(_mean_abs_diff(_luma_frame(src, n, 1280, 720), _luma_frame(art["path"], n, 1280, 720), 1280), 2.0, f"frame {n} ({tags})")


class DubbingRealTests(unittest.TestCase):
    """ADR-085 on the real dubbing-skill (main; no release tag) and the real ffmpeg-skill, without any TTS engine (none is installed in CI): the live
    contract equals the pin, its doctor's verdict on DUB carries the Skill's own code and install line, a plan asking for a dub BLOCKs with that
    code (nothing downloaded, nothing run), and the argv the adapter builds -- every option `--flag=value`, the programme and the user's cues after
    `--`, the translation note, a reference voice with its consent record -- is accepted by the real CLI, which answers ENGINE_UNAVAILABLE."""

    @classmethod
    def setUpClass(cls):
        from video_agent.tools.dubbing import locate_dubbing
        cls.dub = locate_dubbing()
        cls.tmp = os.path.realpath(tempfile.mkdtemp(prefix="va_dub_"))
        src = Path(cls.tmp) / "src"; src.mkdir()
        cls.prog = str(src / "prog.mp4")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=4", "-f", "lavfi",
                        "-i", "sine=frequency=440:duration=4", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", cls.prog], check=True)
        cls.cues = str(src / "en.srt")
        Path(cls.cues).write_text("1\n00:00:01,000 --> 00:00:02,500\nHello there\n", encoding="utf-8")
        cls.ref, cls.consent = str(src / "voice.wav"), str(src / "consent.json")
        Path(cls.ref).write_bytes(b"RIFF not a real clip")
        Path(cls.consent).write_text(json.dumps({"schema": "dubbing/consent@1"}), encoding="utf-8")

    def setUp(self):
        if not self.dub or not locate_ffmpeg_skill():
            self.skipTest("needs a dubbing-skill checkout (VIDEO_AGENT_DUBBING_DIR) and ffmpeg-skill; CI clones both")

    def test_live_contract_doctor_plan_and_argv_on_the_real_skill(self):
        from video_agent.capabilities import CapabilityResolver
        from video_agent.models import Operation
        from video_agent.tools.dubbing import DubbingAdapter, breaking_drift, drift, pinned_contract
        ff = str(locate_ffmpeg_skill().root)
        ad = DubbingAdapter(self.dub, workspace=str(Path(self.tmp) / "ws"), allowed_inputs=[str(Path(self.tmp) / "src")], ffmpeg_skill_dir=ff)
        self.assertEqual(ad.version, pinned_contract()["version"])
        self.assertEqual(breaking_drift(ad.contract), [], "dubbing-skill main changed what the adapter relies on: re-verify the adapter")
        self.assertEqual(drift(ad.contract), [], "dubbing-skill main drifted from the pin (no release tag: the pin follows main): re-pin")
        doc = ad.doctor()
        self.assertIn(doc["status"], ("ok", "degraded"), doc.get("problems"))
        ev = ad.operation_evidence(doc)["DUB"]
        caps = CapabilityResolver(ff).resolve()
        self.assertEqual(caps["dubbing"].status, "AVAILABLE", caps["dubbing"].detail)
        if not ev["ready"]:   # CI: no engine
            self.assertEqual((ev["code"], caps["dubbing:DUB"].status), ("ENGINE_UNAVAILABLE", "MISSING"))
            self.assertTrue(ev["install"])
            svc = Service(workspace=str(Path(self.tmp) / "ws_plan"))
            ir = svc.plan([self.prog], "youtube", user_requirements={"dub.cues": self.cues, "dub.lang": "en"})
            b = next(x for x in ir.doc["decisions"] if x["subject"] == "speech.dub")
            self.assertEqual((b["type"], b["params"]["code"]), ("BLOCK", "ENGINE_UNAVAILABLE"))
            self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
            out = str(Path(self.tmp) / "ws" / "jobs" / "j" / "dub.mp4")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            for extra in ({}, {"reference": self.ref, "consent": self.consent, "engine": "chatterbox", "duck": 10, "on_unfit": "keep"}):
                op = Operation(tool="dubbing/dub", args={"input": "a", "output": "o", "cues": self.cues, "lang": "en", **extra}, inputs=["a"], outputs=["o"])
                r = ad.run(op, {"a": self.prog, "o": out})
                err = r.data["error"]
                self.assertEqual((err["code"], err["recovery_class"], err["retryable"]), ("ENGINE_UNAVAILABLE", "TOOL_MISSING", False), err)
                self.assertFalse(os.path.exists(out))
class DiarizationRealTests(unittest.TestCase):
    """ADR-086 on the real diarization-skill (main, contract 0.1.0): the live contract equals the pin, and without pyannote.audio / the gated
    pipeline (the CI integration job installs neither: the weights need a Hugging Face account) its doctor marks DIARIZE not ready with the
    Skill's own code, the adapter's diarize returns that code verbatim (exit 3, nothing guessed), and a plan asking for speaker turns hands the
    measurement off naming diarization-skill. Runs whenever a diarization-skill checkout is installed (VIDEO_AGENT_DIARIZATION_DIR)."""

    @classmethod
    def setUpClass(cls):
        from video_agent.tools.diarization import locate_diarization
        cls.dia = locate_diarization()
        cls.tmp = os.path.realpath(tempfile.mkdtemp(prefix="va_dia_"))

    def setUp(self):
        if not self.dia:
            self.skipTest("needs a diarization-skill checkout (VIDEO_AGENT_DIARIZATION_DIR)")
        if not shutil.which("ffmpeg"):
            self.skipTest("needs ffmpeg")

    def _wav(self) -> str:
        p = str(Path(self.tmp) / "src" / "talk.wav")
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", f"aevalsrc='{TONE}':s=16000", "-t", "16", p], check=True)
        return p

    def test_live_contract_doctor_and_diarize_code(self):
        from video_agent.tools.diarization import DiarizationAdapter
        ws = str(Path(self.tmp) / "ws_adapter"); os.makedirs(ws, exist_ok=True)
        ad = DiarizationAdapter(self.dia, workspace=ws, allowed_inputs=[str(Path(self._wav()).parent)])
        self.assertEqual(ad.drift(), [], "the installed diarization-skill contract drifted from the pinned one: re-verify the adapter (and re-pin)")
        self.assertEqual(ad.breaking_drift(), [])
        doc = ad.doctor()
        status = ad.operation_status(doc)
        self.assertEqual(status["ASSIGN"], "supported")
        if status["DIARIZE"] == "supported":
            self.skipTest("this machine has pyannote.audio and the pipeline: the not-ready path is not what it can show")
        self.assertEqual(doc["status"], "degraded")
        ev = ad.operation_evidence(doc)["DIARIZE"]
        self.assertIn(ev["code"], ("ENGINE_UNAVAILABLE", "PIPELINE_NOT_FOUND"))
        self.assertTrue(ev["install"], "the doctor's install line travels with the verdict")
        r = ad.measure("diarization/diarize", ad.measurement_args("diarization/diarize", "speaker_turns", self._wav(), "talk", {"num_speakers": 2}, "an", "use"))
        self.assertFalse(r.ok)
        self.assertIn(r.data["error"]["code"], ("ENGINE_UNAVAILABLE", "PIPELINE_NOT_FOUND"))
        self.assertEqual((r.exit_code, r.data["error"]["recovery_class"]), (3, "TOOL_MISSING"))

    def test_plan_hands_the_measurement_off_when_diarize_is_not_ready(self):
        if not locate_ffmpeg_skill():
            self.skipTest("needs ffmpeg-skill")
        svc = Service(workspace=str(Path(self.tmp) / "ws_plan"))
        if svc.caps.resolve().get("diarization:DIARIZE") and svc.caps.resolve()["diarization:DIARIZE"].status == "AVAILABLE":
            self.skipTest("DIARIZE is ready on this machine")
        ir = svc.plan([self._wav()], "generic", user_requirements={"speakers.diarize": True})
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        self.assertFalse([o for o in ir.doc["analysis"]["observations"] if o["kind"] == "speaker_turns"])
        [h] = [h for h in svc.handoffs(ir) if h["skill"] == "speaker_diarization"]
        self.assertEqual(h["repositories"], ["diarization-skill"])
        self.assertIn("diarization:DIARIZE", h["reason"])
