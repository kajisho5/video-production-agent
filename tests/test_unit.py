"""Unit tests: no ffmpeg needed (FakeAdapter)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter  # noqa: E402
from fake_ai_provider import FakeAIProvider, recommend_from_analysis  # noqa: E402

from video_agent.capabilities.resolver import Capability  # noqa: E402
from video_agent.execution import CompileError, Executor, compile_ir  # noqa: E402
from video_agent.execution.recovery import classify_error, next_attempt  # noqa: E402
from video_agent.media.analyzer import AnalysisResult  # noqa: E402
from video_agent.models import Decision, Event, Operation, TimeRange, ToolResult  # noqa: E402
from video_agent.policy.rules import Rule, resolve_rules  # noqa: E402
from video_agent.profiles import load_profile  # noqa: E402
from video_agent.project import load_ir, save_ir, validate_ir  # noqa: E402
from video_agent.service import Service  # noqa: E402
from video_agent.temporal import Timeline  # noqa: E402
from video_agent.tools.ffmpeg_skill.adapter import FfmpegSkillAdapter, PathPolicy  # noqa: E402
from video_agent.tools.ffmpeg_skill.locate import FfmpegSkill  # noqa: E402
from video_agent.tools.base import ToolError  # noqa: E402


class FakeCaps:
    def __init__(self, missing=(), extra=()):
        self.missing = set(missing)
        self.extra = list(extra)   # e.g. "media-analysis": the external observation Skill's capability (resolved by its doctor in production)

    def resolve(self, refresh=False):
        names = ["python", "ffmpeg", "ffprobe", "ffmpeg-skill", "encoder:libx264", "encoder:libx265", "encoder:prores_ks", "filter:loudnorm", "font:cjk-ja"] + self.extra
        return {n: Capability(n, "MISSING" if n in self.missing else "AVAILABLE", "fake", {"version": "0.8.4-fake"} if n == "ffmpeg-skill" else {}) for n in names}


def make_service(tmp, adapter=None, caps=None):
    return Service(workspace=tmp, adapter=adapter or FakeAdapter(), caps=caps or FakeCaps())


def fake_media(tmp, name="talk.mp4"):
    p = Path(tmp) / "src" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * 64)
    return str(p)


class TimelineTests(unittest.TestCase):
    def test_query_by_type_range_kind(self):
        tl = Timeline()
        tl.add_timeline("a1")
        tl.add(Event(type="AUDIO_SILENCE", timeline_id="asset:a1", range=TimeRange(0, 3).to_dict(), source="t", kind="OBSERVED"))
        tl.add(Event(type="SPEAKER", timeline_id="asset:a1", range=TimeRange(2, 10).to_dict(), source="t", kind="INFERRED", confidence=0.6, metadata={"speaker": "A"}))
        self.assertEqual(len(tl.query(type="SPEAKER")), 1)
        self.assertEqual(len(tl.query(between=(4, 5))), 1)
        self.assertEqual(len(tl.query(between=(0, 2.5))), 2)
        self.assertEqual(len(tl.query(kind="OBSERVED")), 1)
        self.assertEqual(len(tl.query(min_confidence=0.9)), 0)
        with self.assertRaises(ValueError):
            tl.add(Event(type="X", timeline_id="asset:nope", range=TimeRange(0).to_dict(), source="t", kind="USER"))
        rt = Timeline.from_dict(tl.to_dict())
        self.assertEqual(len(rt.events), 2)


class PolicyTests(unittest.TestCase):
    def test_precedence_and_constraint_conflict(self):
        rules = [Rule("g", "POLICY", "GLOBAL", "k", 1), Rule("p", "PREFERENCE", "PROFILE", "k", 2), Rule("r", "PREFERENCE", "REQUEST", "k", 3)]
        self.assertEqual(resolve_rules(rules).get("k"), 3)
        rules = [Rule("c", "CONSTRAINT", "PROFILE", "k", "CONFIRM"), Rule("r", "PREFERENCE", "REQUEST", "k", "AUTO")]
        rs = resolve_rules(rules)
        self.assertEqual(rs.get("k"), "CONFIRM")
        self.assertEqual(len(rs.conflicts), 1)

    def test_profiles_inherit(self):
        p = load_profile("conference")
        self.assertEqual(p.chain, ["generic", "conference"])
        rs = resolve_rules(p.rules)
        self.assertEqual(rs.get("audio.loudness.target_lufs"), -16)
        self.assertEqual(rs.get("silence.leading.approval"), "CONFIRM")
        self.assertTrue(rs.effective["edit.semantic_deletion.approval"].hard)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        root = Path(self.tmp) / "skill"
        (root / "scripts").mkdir(parents=True)
        for s in ("probe", "cut", "loudness", "export", "check"):
            (root / "scripts" / f"{s}.py").write_text("print('{}')")
        self.skill = FfmpegSkill(root, "0.8.4", ["probe", "cut", "loudness", "export", "check"])

    def test_argv_typed_and_catalog_enforced(self):
        a = FfmpegSkillAdapter(self.skill)
        argv = a.build_argv("ffmpeg-skill/loudness", {"input": "/x/in.mp4", "lufs": -14, "tp": -1.0, "measure_only": True, "output": "/w/o.mp4"}, {})
        self.assertEqual(argv, ["/x/in.mp4", "-I", "-14", "--tp", "-1", "--measure-only", "-o", "/w/o.mp4"])
        with self.assertRaises(ToolError):
            a.build_argv("ffmpeg-skill/cut", {"input": "/x", "crf_hack": "1; rm -rf /"}, {})
        with self.assertRaises(ToolError):
            a.build_argv("ffmpeg-skill/cut", {"input": "/x", "accurate": "yes"}, {})

    def test_path_policy(self):
        src = fake_media(self.tmp)
        ws = str(Path(self.tmp) / "ws")
        a = FfmpegSkillAdapter(self.skill, PathPolicy([str(Path(src).parent)], ws))
        a.build_argv("ffmpeg-skill/cut", {"input": src, "output": f"{ws}/out.mp4"}, {})
        with self.assertRaises(ToolError):
            a.build_argv("ffmpeg-skill/cut", {"input": src, "output": src}, {})
        with self.assertRaises(ToolError):
            a.build_argv("ffmpeg-skill/cut", {"input": src, "output": str(Path(self.tmp) / "src" / "out.mp4")}, {})
        with self.assertRaises(ToolError):
            a.build_argv("ffmpeg-skill/cut", {"input": "/etc/passwd", "output": f"{ws}/o.mp4"}, {})


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)

    def test_plan_separates_observation_inference_decision(self):
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        d = ir.doc
        self.assertEqual(d["schema_version"], "1.2")
        self.assertTrue(d["provenance"]["plan_hash"] and d["provenance"]["ir_hash"])
        self.assertEqual(d["revision"], {"feedback": [], "history": [], "approved_plan_version": None})
        kinds = {o["kind"] for o in d["analysis"]["observations"]}
        self.assertEqual(kinds, {"media_probe", "silence", "loudness"})
        inf = {i["kind"] for i in d["analysis"]["inferences"]}
        self.assertIn("leading_silence_unwanted", inf)
        self.assertIn("trailing_silence_unwanted", inf)
        self.assertIn("loudness_off_target", inf)
        for i in d["analysis"]["inferences"]:
            self.assertTrue(i["evidence"])
        subjects = {x["subject"]: x for x in d["decisions"]}
        self.assertEqual(subjects["silence.leading"]["approval"], "AUTO")
        self.assertEqual(subjects["audio.loudness"]["params"]["target_lufs"], -14)
        self.assertEqual(subjects["delivery.youtube"]["provenance"], "PROFILE")
        self.assertEqual(len(d["video"]["operations"]), 1)
        self.assertEqual(d["video"]["operations"][0]["keep"], [[2.85, 13.85]])
        provs = {r["provenance"] for r in d["requirements"]}
        self.assertTrue({"PROFILE", "DEFAULT"} <= provs)
        rep = validate_ir(ir, svc.caps.resolve())
        self.assertTrue(rep.ok, rep.errors)

    def test_user_requirement_overrides_profile_and_records_provenance(self):
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube", user_requirements={"audio.loudness.target_lufs": -18})
        req = [r for r in ir.doc["requirements"] if r["key"] == "audio.loudness.target_lufs"]
        self.assertEqual(req[0]["provenance"], "USER")
        dec = next(x for x in ir.doc["decisions"] if x["subject"] == "audio.loudness")
        self.assertEqual(dec["params"]["target_lufs"], -18)

    def test_conference_requires_confirmation_then_runs(self):
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "conference")
        ir_path = str(Path(self.tmp) / "p.json")
        save_ir(ir, ir_path)
        out = svc.render(load_ir(ir_path), ir_path)
        self.assertEqual(out["status"], "WAITING_FOR_APPROVAL")
        self.assertEqual(out["job"]["state"], "WAITING_FOR_APPROVAL")
        ids = [d["id"] for d in out["pending"]]
        out = svc.render(load_ir(ir_path), ir_path, approve=ids)
        self.assertEqual(out["status"], "COMPLETED", out)
        ir2 = load_ir(ir_path)
        self.assertTrue(any(e["type"] == "USER_DECISION" for e in ir2.doc["timeline"]["events"]))
        self.assertEqual(ir2.decision(ids[0])["status"], "APPROVED")
        self.assertEqual(len(out["artifacts"]), 2)
        prov = json.loads((Path(out["job"]["workspace"]) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        self.assertEqual(prov["operations"][0]["tool"], "ffmpeg-skill/cut")
        self.assertTrue(prov["operations"][0]["decision"])

    def test_missing_capability_blocks(self):
        svc = make_service(self.tmp, caps=FakeCaps(missing={"encoder:libx264"}))
        ir = svc.plan([self.src], "youtube")
        self.assertTrue(ir.blocked())
        rep = validate_ir(ir, svc.caps.resolve())
        self.assertFalse(rep.ok)
        ir_path = str(Path(self.tmp) / "b.json")
        save_ir(ir, ir_path)
        out = svc.render(load_ir(ir_path), ir_path)
        self.assertIn(out["status"], ("BLOCKED", "FAILED"))

    def test_ambience_is_not_normalised(self):
        svc = make_service(self.tmp, adapter=FakeAdapter(lufs=-45.0))
        ir = svc.plan([self.src], "youtube")
        dec = next(x for x in ir.doc["decisions"] if x["subject"] == "audio.loudness")
        self.assertEqual(dec["decision"], "skip")
        self.assertEqual(ir.doc["audio"]["operations"], [])

    def test_generic_without_preset_delivers_intermediate(self):
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "generic")
        ops, paths = compile_ir(ir, "/w/jobs/j")
        tools = [o.tool for o in ops]
        # generic: technical silence is trimmed; the profile sets no loudness target, so -11 LUFS is left alone
        # (a target must come from a profile or an explicit --set audio.loudness.target_lufs)
        self.assertEqual(tools, ["ffmpeg-skill/cut"])
        self.assertFalse([x for x in ir.doc["decisions"] if x["subject"] == "audio.loudness"])
        ir2 = svc.plan([self.src], "generic", user_requirements={"audio.loudness.target_lufs": -16})
        self.assertEqual([o.tool for o in compile_ir(ir2, "/w/jobs/j")[0]], ["ffmpeg-skill/cut", "ffmpeg-skill/loudness"])
        self.assertNotIn("ffmpeg-skill/export", tools)
        art = [k for k in paths if k.endswith("_delivery_main")]
        self.assertEqual(len(art), 1)
        self.assertEqual(paths[art[0]], paths[ops[-1].outputs[0]], "the last intermediate is the deliverable")

    def test_same_file_name_twice_gets_distinct_paths(self):
        a = fake_media(self.tmp, "camA/clip.mp4")
        b = fake_media(self.tmp, "camB/clip.mp4")
        svc = make_service(self.tmp)
        ir = svc.plan([a, b], "youtube")
        ops, paths = compile_ir(ir, "/w/jobs/j")
        outs = [paths[o] for op in ops for o in op.outputs]
        self.assertEqual(len(outs), len(set(outs)), outs)
        self.assertEqual(len([o for o in ops if o.tool == "ffmpeg-skill/export"]), 2)

    def test_unknown_requirement_key_is_rejected(self):
        svc = make_service(self.tmp)
        with self.assertRaises(ValueError):
            svc.plan([self.src], "youtube", user_requirements={"api_key": "sk-secret"})

    def test_interrupt_marks_job_cancelled_and_persists(self):
        class Interrupting(FakeAdapter):
            def run(self, op, paths, timeout=None, dry_run=False, attempt=1):
                if op.tool == "ffmpeg-skill/loudness" and not op.args.get("measure_only"):
                    raise KeyboardInterrupt
                return super().run(op, paths, timeout, dry_run, attempt)
        svc = make_service(self.tmp, adapter=Interrupting())
        ir = svc.plan([self.src], "youtube")
        ir_path = str(Path(self.tmp) / "i.json")
        save_ir(ir, ir_path)
        out = svc.render(load_ir(ir_path), ir_path)
        self.assertEqual(out["status"], "CANCELLED")
        self.assertEqual(out["job"]["state"], "CANCELLED")
        self.assertEqual(out["execution"]["recovery"][-1]["class"], "INTERRUPTED")
        job_file = Path(out["job"]["workspace"]) / "jobs" / out["job"]["id"] / "job.json"
        self.assertEqual(json.loads(job_file.read_text())["state"], "CANCELLED")
        self.assertTrue((job_file.parent / "provenance.json").exists())
        self.assertEqual(load_ir(ir_path).doc["provenance"]["runs"][-1]["status"], "CANCELLED")

    def test_rejected_decision_blocks_render(self):
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        lead = next(x for x in ir.doc["decisions"] if x["subject"] == "silence.leading")
        lead["status"] = "REJECTED"
        rep = validate_ir(ir, svc.caps.resolve())
        self.assertFalse(rep.ok)
        self.assertTrue(any("REJECTED" in e for e in rep.errors))
        ir_path = str(Path(self.tmp) / "rej.json")
        save_ir(ir, ir_path)
        out = svc.render(load_ir(ir_path), ir_path)
        self.assertEqual(out["status"], "BLOCKED")
        self.assertIn(lead["id"], out["rejected"])
        self.assertFalse(out["job"]["completed_ops"])

    def test_operation_ids_are_deterministic(self):
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        a = [o.id for o in compile_ir(ir, "/w/jobs/j")[0]]
        b = [o.id for o in compile_ir(ir, "/w/jobs/k")[0]]
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), len(a))

    def test_frame_accuracy_is_plan_content(self):
        svc = make_service(self.tmp)
        a = svc.plan([self.src], "youtube")
        b = svc.plan([self.src], "youtube", user_requirements={"edit.precision": "frame"})
        self.assertFalse(a.doc["video"]["operations"][0]["accurate"])
        self.assertTrue(b.doc["video"]["operations"][0]["accurate"])
        self.assertNotEqual(a.plan_hash(), b.plan_hash(), "precision changes what executes, so it changes the plan hash")
        ops, _ = compile_ir(b, "/w/jobs/j")
        self.assertTrue(ops[0].args.get("accurate"))
        from video_agent.project.diff import plan_diff
        self.assertTrue(any("frame-accurate" in l for l in plan_diff(a.doc, b.doc)["summary"]))

    def test_qa_measurements_are_recorded(self):
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        ir_path = str(Path(self.tmp) / "m.json")
        save_ir(ir, ir_path)
        out = svc.render(load_ir(ir_path), ir_path)
        tools = [m["tool"] for m in out["qa"]["measurements"]]
        self.assertIn("ffmpeg-skill/probe", tools)
        self.assertNotIn("ffmpeg-skill/check", tools, "check.py result from the executor is reused, not re-run")

    def test_schema_rejects_bad_ir(self):
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        ir.doc["video"]["operations"][0]["keep"] = [[5, 2]]
        rep = validate_ir(ir, svc.caps.resolve())
        self.assertFalse(rep.ok)
        ir.doc["decisions"][0]["approval"] = "MAYBE"
        rep = validate_ir(ir, svc.caps.resolve())
        self.assertTrue(any("schema" in e for e in rep.errors))
        ir.doc["schema_version"] = "0.9"
        p = str(Path(self.tmp) / "old.json")
        Path(p).write_text(json.dumps(ir.doc))
        with self.assertRaises(ValueError):
            load_ir(p)

    def test_migration_1_0_to_1_1(self):
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        doc = json.loads(json.dumps(ir.doc))
        doc["schema_version"] = "1.0"
        doc["provenance"].pop("plan_hash")
        doc["execution"].pop("resume_from")
        doc["execution"].pop("reviews")
        doc.pop("revision")
        doc["execution"]["approvals"] = {doc["decisions"][0]["id"]: {"by": "someone", "at": "2026-01-01T00:00:00Z"}}
        p = str(Path(self.tmp) / "v10.json")
        Path(p).write_text(json.dumps(doc))
        loaded = load_ir(p)
        self.assertEqual(loaded.doc["schema_version"], "1.2")
        self.assertEqual(loaded.doc["provenance"]["plan_hash"], ir.plan_hash())
        self.assertIsNone(loaded.doc["execution"]["resume_from"])
        self.assertEqual(loaded.doc["execution"]["reviews"][doc["decisions"][0]["id"]]["action"], "APPROVED", "legacy approvals become review records")
        self.assertEqual(loaded.doc["revision"]["approved_plan_version"], None)
        self.assertTrue(validate_ir(loaded, svc.caps.resolve()).ok)

    def test_plan_hash_ignores_approval_ir_hash_does_not(self):
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "conference")
        ph, ih = ir.plan_hash(), ir.ir_hash()
        ir.approve(["all"])
        self.assertEqual(ir.plan_hash(), ph)
        self.assertNotEqual(ir.ir_hash(), ih)
        ir.doc["video"]["operations"][0]["keep"] = [[3.5, 13.0]]
        self.assertNotEqual(ir.plan_hash(), ph)

    def test_dry_run_lists_operations_without_execution(self):
        ad = FakeAdapter()
        svc = make_service(self.tmp, adapter=ad)
        ir = svc.plan([self.src], "youtube")
        plan = svc.dry_run(ir)
        self.assertEqual([o["tool"] for o in plan["operations"]], ["ffmpeg-skill/cut", "ffmpeg-skill/loudness", "ffmpeg-skill/export", "ffmpeg-skill/check"])
        self.assertFalse(any(o.tool in ("ffmpeg-skill/cut", "ffmpeg-skill/export") for o in ad.calls))
        self.assertEqual(plan["estimate"]["full_reencodes"], 1)


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)

    def test_analysis_strategy_is_honest(self):
        """The recorded strategy is the one the request ran with, and the budget block reports real enforcement + usage."""
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        self.assertEqual(ir.doc["analysis"]["strategy"], "TARGETED_ANALYSIS", "profile policy analysis.strategy=TARGETED is what ran")
        self.assertEqual(ir.doc["analysis"]["analyses"][0]["request"]["strategy"], "TARGETED")
        self.assertEqual(sorted(ir.doc["analysis"]["analyses"][0]["request"]["kinds"]), ["loudness", "media_probe", "silence"], "default requirements need all three kinds")
        self.assertTrue(ir.doc["analysis"]["budget"]["enforced"])
        self.assertEqual(ir.doc["analysis"]["budget"]["calls"], 3)
        self.assertIn("max_bytes_scanned", ir.doc["analysis"]["budget"]["unsupported"])
        ir2 = make_service(self.tmp).plan([self.src], "youtube", strategy="FULL")
        self.assertEqual(ir2.doc["analysis"]["strategy"], "FULL_ANALYSIS")

    def test_recovery_retries_once_then_blocks(self):
        ad = FakeAdapter(fail_tools={"ffmpeg-skill/cut": 1})
        op = Operation(tool="ffmpeg-skill/cut", args={"input": "a", "segments": "1-2", "output": "o"}, inputs=["a"], outputs=["o"], idempotency_key="k1")
        tmp = tempfile.mkdtemp()
        ex = Executor(ad, max_attempts=2)
        res = ex.run([op], {"a": "/x", "o": f"{tmp}/o.mp4"})
        self.assertEqual(res.status, "COMPLETED")
        self.assertEqual(len(res.recovery), 1)
        self.assertEqual(res.recovery[0]["class"], "ENCODER_FAILED")
        self.assertTrue(ad.calls[-1].args.get("accurate"), "retry used the alternative arguments")
        ad = FakeAdapter(fail_tools={"ffmpeg-skill/cut": 5})
        res = Executor(ad, max_attempts=2).run([op], {"a": "/x", "o": f"{tmp}/o2.mp4"})
        self.assertEqual(res.status, "FAILED")
        self.assertEqual(len(res.recovery), 2)

    def test_idempotent_skip_and_cancel(self):
        ad = FakeAdapter()
        tmp = tempfile.mkdtemp()
        op = Operation(tool="ffmpeg-skill/cut", args={"input": "a", "segments": "1-2", "output": "o"}, inputs=["a"], outputs=["o"], idempotency_key="k")
        paths = {"a": "/x", "o": f"{tmp}/o.mp4"}
        ex = Executor(ad)
        ex.run([op], paths)
        res = Executor(ad, completed_keys=ex.completed).run([op], paths)
        self.assertEqual(res.skipped, [op.id])
        ex2 = Executor(ad)
        ex2.cancel()
        self.assertEqual(ex2.run([op], paths).status, "CANCELLED")

    def test_classify(self):
        r = lambda code, txt: ToolResult("o", "ffmpeg-skill/cut", False, code, None, {}, [], txt, 0)  # noqa: E731
        self.assertEqual(classify_error(r(127, "error: 'ffmpeg' was not found on PATH")), "TOOL_MISSING")
        self.assertEqual(classify_error(r(1, "error: input not found: x")), "INPUT_MISSING")
        self.assertEqual(classify_error(r(2, "cut.py: error: unrecognized arguments: --x")), "INVALID_ARGS")
        self.assertEqual(next_attempt(r(127, "not found on PATH"), 1, 3, None)["action"], "BLOCK")
        self.assertEqual(next_attempt(r(124, "timeout after 10s"), 1, 3, 10.0)["timeout"], 20.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class ResumeTests(unittest.TestCase):
    """Job resume and idempotent skipping, checked on the actual failure/re-run paths."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)

    def _plan(self, svc, profile="youtube", name="p.json", **kw):
        ir = svc.plan([self.src], profile, **kw)
        p = str(Path(self.tmp) / name)
        save_ir(ir, p)
        return p

    def test_resume_after_mid_run_failure_reuses_only_completed_ops(self):
        failing = FakeAdapter(fail_tools={"ffmpeg-skill/export": 9})
        svc = make_service(self.tmp, adapter=failing)
        p = self._plan(svc)
        first = svc.render(load_ir(p), p)
        self.assertEqual(first["status"], "FAILED")
        done_tools = [r["tool"] for r in first["execution"]["results"] if r["ok"]]
        self.assertEqual(done_tools, ["ffmpeg-skill/cut", "ffmpeg-skill/loudness"])
        self.assertEqual(len(first["job"]["completed_ops"]), 2)
        healthy = FakeAdapter()
        svc2 = make_service(self.tmp, adapter=healthy)
        second = svc2.render(load_ir(p), p, resume=first["job"]["id"])
        self.assertEqual(second["status"], "COMPLETED", second["execution"])
        ran = [o.tool for o in healthy.calls if o.kind != "measure" and not o.args.get("measure_only") and o.tool != "ffmpeg-skill/probe"]
        self.assertNotIn("ffmpeg-skill/cut", ran)
        self.assertNotIn("ffmpeg-skill/loudness", ran[:1])
        self.assertIn("ffmpeg-skill/export", ran)
        self.assertEqual(len(second["execution"]["skipped"]), 2)
        self.assertFalse(second["resume"]["plan_changed"])
        self.assertEqual(second["job"]["resumed_from"], first["job"]["id"])
        # the reused loudness output lives in the FIRST job's directory and feeds the export
        reused = list(second["execution"]["reused"].values())
        self.assertTrue(all(first["job"]["id"] in r for r in reused), reused)
        prov = json.loads((Path(second["job"]["workspace"]) / "jobs" / second["job"]["id"] / "provenance.json").read_text())
        self.assertEqual(prov["resume"]["resumed_from"], first["job"]["id"])
        self.assertEqual(sorted(prov["skipped"]), sorted(second["execution"]["skipped"]))

    def test_resume_same_ir_twice_skips_all_transforms_but_reruns_qa(self):
        ad = FakeAdapter()
        svc = make_service(self.tmp, adapter=ad)
        p = self._plan(svc)
        first = svc.render(load_ir(p), p)
        self.assertEqual(first["status"], "COMPLETED")
        ad2 = FakeAdapter()
        second = make_service(self.tmp, adapter=ad2).render(load_ir(p), p, resume="last")
        self.assertEqual(second["status"], "COMPLETED")
        transforms = [o.tool for o in ad2.calls if o.kind == "transform"]
        self.assertEqual(transforms, [], "no transform re-ran")
        self.assertIn("ffmpeg-skill/check", [o.tool for o in ad2.calls if o.kind == "qa"], "delivery check always re-runs")
        self.assertEqual(len(second["execution"]["skipped"]), 3)
        self.assertEqual(len(second["artifacts"]), 1)
        self.assertTrue(os.path.exists(second["artifacts"][0]["path"]))

    def test_changed_trim_invalidates_downstream_keys(self):
        ad = FakeAdapter()
        svc = make_service(self.tmp, adapter=ad)
        p = self._plan(svc)
        first = svc.render(load_ir(p), p)
        ir = load_ir(p)
        ir.doc["video"]["operations"][0]["keep"] = [[3.5, 13.0]]
        save_ir(ir, p)
        ad2 = FakeAdapter()
        second = make_service(self.tmp, adapter=ad2).render(load_ir(p), p, resume=first["job"]["id"])
        self.assertEqual(second["status"], "COMPLETED")
        self.assertTrue(second["resume"]["plan_changed"])
        self.assertEqual(second["execution"]["skipped"], [], "trim changed: cut, loudness AND export must all re-run")
        self.assertEqual([o.tool for o in ad2.calls if o.kind == "transform"], ["ffmpeg-skill/cut", "ffmpeg-skill/loudness", "ffmpeg-skill/export"])

    def test_changed_loudness_target_keeps_cut_but_reruns_rest(self):
        ad = FakeAdapter()
        svc = make_service(self.tmp, adapter=ad)
        p = self._plan(svc)
        first = svc.render(load_ir(p), p)
        ir = load_ir(p)
        ir.doc["audio"]["operations"][0]["target_lufs"] = -16
        save_ir(ir, p)
        ad2 = FakeAdapter()
        second = make_service(self.tmp, adapter=ad2).render(load_ir(p), p, resume=first["job"]["id"])
        self.assertEqual(len(second["execution"]["skipped"]), 1)
        self.assertEqual([o.tool for o in ad2.calls if o.kind == "transform"], ["ffmpeg-skill/loudness", "ffmpeg-skill/export"])

    def test_missing_or_tampered_output_is_not_reused(self):
        ad = FakeAdapter()
        svc = make_service(self.tmp, adapter=ad)
        p = self._plan(svc)
        first = svc.render(load_ir(p), p)
        recs = list(first["job"]["completed_ops"].values())
        os.remove(recs[0]["output"])                                  # cut output deleted
        Path(recs[1]["output"]).write_bytes(b"tampered-longer")         # loudness output replaced
        ad2 = FakeAdapter()
        second = make_service(self.tmp, adapter=ad2).render(load_ir(p), p, resume=first["job"]["id"])
        self.assertEqual(second["status"], "COMPLETED")
        # cut (deleted) and loudness (tampered) re-run; the export record is intact and its plan-level key is unchanged, so it is reused
        rerun = [o.tool for o in ad2.calls if o.kind == "transform"]
        self.assertEqual(rerun, ["ffmpeg-skill/cut", "ffmpeg-skill/loudness"])
        self.assertEqual(len(second["execution"]["skipped"]), 1)

    def test_replaced_source_without_hash_is_detected_by_stat(self):
        ad = FakeAdapter()
        svc = make_service(self.tmp, adapter=ad)
        p = self._plan(svc, hash_sources=False)
        first = svc.render(load_ir(p), p)
        self.assertEqual(len(first["job"]["completed_ops"]), 3)
        Path(self.src).write_bytes(b"\x00" * 128)                   # different size => different fingerprint after re-plan
        p2 = self._plan(make_service(self.tmp, adapter=FakeAdapter()), name="p2.json", hash_sources=False)
        ad2 = FakeAdapter()
        second = make_service(self.tmp, adapter=ad2).render(load_ir(p2), p2, resume=first["job"]["id"])
        self.assertEqual(second["execution"]["skipped"], [])

    def test_resume_unknown_job_is_an_error(self):
        svc = make_service(self.tmp)
        p = self._plan(svc)
        with self.assertRaises(FileNotFoundError):
            svc.render(load_ir(p), p, resume="last")
        with self.assertRaises(FileNotFoundError):
            svc.render(load_ir(p), p, resume="job_doesnotexist")

    def test_legacy_string_records_are_never_trusted(self):
        from video_agent.execution.executor import record_matches
        self.assertFalse(record_matches("/some/path"))
        self.assertFalse(record_matches({"output": "/nope", "size": 1, "mtime": 1.0}))


class RevisionTests(unittest.TestCase):
    """REJECT → revise → Plan v2 → PlanDiff → approve → render, checked on the real state transitions."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)

    def _plan(self, svc, profile="conference", name="p.json", **kw):
        ir = svc.plan([self.src], profile, **kw)
        p = str(Path(self.tmp) / name)
        save_ir(ir, p)
        return p

    def _dec(self, p, subject):
        return next(d for d in load_ir(p).doc["decisions"] if d["subject"] == subject and d["status"] != "REJECTED")

    def test_reject_requires_reason_and_records_actor_and_time(self):
        svc = make_service(self.tmp)
        p = self._plan(svc)
        lead = self._dec(p, "silence.leading")
        with self.assertRaises(ValueError):
            svc.reject(load_ir(p), p, [lead["id"]], reason="   ")
        out = svc.reject(load_ir(p), p, [lead["id"]], reason="the lead-in is the chair's introduction", who="reviewer:kaji")
        self.assertEqual(out["rejected"], [lead["id"]])
        ir = load_ir(p)
        rv = ir.doc["execution"]["reviews"][lead["id"]]
        self.assertEqual((rv["action"], rv["by"], rv["reason"], rv["plan_version"]), ("REJECTED", "reviewer:kaji", "the lead-in is the chair's introduction", 1))
        self.assertTrue(rv["at"])
        self.assertEqual(ir.decision(lead["id"])["status"], "REJECTED")
        ev = [e for e in ir.doc["timeline"]["events"] if e["type"] == "USER_DECISION"][-1]
        self.assertEqual(ev["metadata"]["action"], "REJECTED")
        self.assertEqual(ev["metadata"]["reason"], "the lead-in is the chair's introduction")
        self.assertIsNone(ir.doc["revision"]["approved_plan_version"])

    def test_plan_with_rejected_decision_can_never_render(self):
        ad = FakeAdapter()
        svc = make_service(self.tmp, adapter=ad)
        p = self._plan(svc)
        lead = self._dec(p, "silence.leading")
        svc.reject(load_ir(p), p, [lead["id"]], reason="keep the intro")
        out = svc.render(load_ir(p), p)
        self.assertEqual(out["status"], "BLOCKED")
        self.assertEqual(out["rejected"][lead["id"]], "keep the intro")
        self.assertEqual([o for o in ad.calls if o.kind == "transform"], [], "no tool ran")
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["status"], "BLOCKED", "approve cannot override a rejection")
        with self.assertRaises(ValueError):
            svc.approve(load_ir(p), p, ["all"])
        rep = validate_ir(load_ir(p), svc.caps.resolve())
        self.assertFalse(rep.ok)

    def test_reject_all_blocks_and_revise_yields_empty_plan(self):
        ad = FakeAdapter()
        svc = make_service(self.tmp, adapter=ad)
        p = self._plan(svc, profile="youtube")
        svc.reject(load_ir(p), p, ["all"], reason="do nothing to this file")
        self.assertEqual(svc.render(load_ir(p), p)["status"], "BLOCKED")
        out = svc.revise(load_ir(p), p)
        self.assertTrue(out["created"])
        ir = load_ir(p)
        self.assertEqual(ir.version, 2)
        self.assertEqual(ir.doc["video"]["operations"], [])
        self.assertEqual(ir.doc["audio"]["operations"], [])
        self.assertEqual(ir.doc["delivery"]["targets"], [])
        self.assertEqual({d["status"] for d in ir.doc["decisions"]}, {"REJECTED"})

    def test_partial_reject_revise_diff_approve_render(self):
        ad = FakeAdapter()
        svc = make_service(self.tmp, adapter=ad)
        p = self._plan(svc)
        v1 = load_ir(p).doc
        lead = self._dec(p, "silence.leading")
        svc.reject(load_ir(p), p, [lead["id"]], reason="chair introduction must stay", who="reviewer")
        out = svc.revise(load_ir(p), p, who="editor")
        self.assertTrue(out["created"])
        ir = load_ir(p)
        self.assertEqual(ir.version, 2)
        # the rejected operation is gone, the rest survived (conference: no trailing trim because the tail is shorter than 3 s)
        self.assertEqual(ir.doc["video"]["operations"], [])
        self.assertEqual(len(ir.doc["audio"]["operations"]), 1)
        self.assertEqual(len(ir.doc["delivery"]["targets"]), 2)
        # the rejected decision is carried as history with its review; the planner did not re-propose it
        rej = [d for d in ir.doc["decisions"] if d["status"] == "REJECTED"]
        self.assertEqual([d["id"] for d in rej], [lead["id"]])
        self.assertEqual(ir.doc["execution"]["reviews"][lead["id"]]["reason"], "chair introduction must stay")
        self.assertEqual(out["dropped_proposals"][0]["subject"], "silence.leading")
        # PlanDiff names the change
        diff = ir.doc["revision"]["history"][-1]["diff"]
        self.assertEqual(diff["from_version"], 1)
        self.assertIn("video.trim@" + list(ir.doc["assets"])[0], diff["video"]["removed"])
        self.assertTrue(any(line.startswith("VIDEO") and "removed" in line for line in diff["summary"]))
        self.assertEqual(ir.doc["revision"]["history"][-1]["rejection_reasons"][lead["id"]], "chair introduction must stay")
        # v1 is preserved untouched
        snap = load_ir(out["snapshot"]).doc
        self.assertEqual(snap["plan"]["version"], 1)
        self.assertEqual(len(snap["video"]["operations"]), 1)
        self.assertEqual(snap["provenance"]["plan_hash"], v1["provenance"]["plan_hash"], "rejection never changed v1's plan content")
        self.assertEqual(snap["decisions"][0]["id"], v1["decisions"][0]["id"])
        self.assertTrue(any(l.startswith("REJECTED silence.leading") and "chair introduction" in l for l in diff["summary"]), diff["summary"])
        # v2 needs approval even though nothing is CONFIRM-pending
        self.assertEqual(load_ir(p).pending_confirmations(), [])
        r = svc.render(load_ir(p), p)
        self.assertEqual(r["status"], "WAITING_FOR_APPROVAL")
        self.assertEqual(r["plan_version"], 2)
        a = svc.approve(load_ir(p), p, ["all"], who="reviewer")
        self.assertTrue(a["renderable"])
        self.assertEqual(a["approved_plan_version"], 2)
        r = svc.render(load_ir(p), p)
        self.assertEqual(r["status"], "COMPLETED", r.get("execution"))
        self.assertEqual([o.tool for o in ad.calls if o.kind == "transform"], ["ffmpeg-skill/loudness", "ffmpeg-skill/export", "ffmpeg-skill/export"])
        self.assertEqual(r["job"]["plan_version"], 2)

    def test_hashes_approval_vs_revision(self):
        svc = make_service(self.tmp)
        p = self._plan(svc)
        ir = load_ir(p)
        ph1, ih1 = ir.plan_hash(), ir.ir_hash()
        lead = self._dec(p, "silence.leading")
        svc.reject(ir, p, [lead["id"]], reason="x")
        self.assertEqual(ir.plan_hash(), ph1, "rejection does not change plan_hash")
        self.assertNotEqual(ir.ir_hash(), ih1, "rejection changes ir_hash")
        svc.revise(load_ir(p), p)
        ir2 = load_ir(p)
        self.assertNotEqual(ir2.plan_hash(), ph1, "revision changes plan_hash")
        ph2 = ir2.plan_hash()
        svc.approve(ir2, p, ["all"])
        self.assertEqual(load_ir(p).plan_hash(), ph2, "approval does not change plan_hash")

    def test_v1_job_and_provenance_are_independent_from_v2(self):
        ad = FakeAdapter()
        svc = make_service(self.tmp, adapter=ad)
        p = self._plan(svc, profile="youtube")
        j1 = svc.render(load_ir(p), p)
        self.assertEqual(j1["status"], "COMPLETED")
        loud = self._dec(p, "audio.loudness")
        svc.reject(load_ir(p), p, [loud["id"]], reason="levels were fine on the venue PA")
        svc.revise(load_ir(p), p)
        svc.approve(load_ir(p), p, ["all"])
        # plain render of v2: nothing from v1 is reused
        j2 = svc.render(load_ir(p), p)
        self.assertEqual(j2["status"], "COMPLETED")
        self.assertEqual(j2["execution"]["skipped"], [])
        self.assertIsNone(j2["job"]["resumed_from"])
        self.assertNotEqual(j1["job"]["id"], j2["job"]["id"])
        p1 = json.loads((Path(j1["job"]["workspace"]) / "jobs" / j1["job"]["id"] / "provenance.json").read_text())
        p2 = json.loads((Path(j2["job"]["workspace"]) / "jobs" / j2["job"]["id"] / "provenance.json").read_text())
        self.assertEqual((p1["plan_version"], p2["plan_version"]), (1, 2))
        self.assertEqual([e["tool"] for e in p1["operations"]], ["ffmpeg-skill/cut", "ffmpeg-skill/loudness", "ffmpeg-skill/export", "ffmpeg-skill/check"])
        self.assertEqual([e["tool"] for e in p2["operations"]], ["ffmpeg-skill/cut", "ffmpeg-skill/export", "ffmpeg-skill/check"])
        self.assertIn(loud["id"], p2["reviews"])
        self.assertNotIn(loud["id"], p1["reviews"])
        # resume across versions: only the unchanged cut is reused; the export (whose input changed) is not
        ad3 = FakeAdapter()
        j3 = make_service(self.tmp, adapter=ad3).render(load_ir(p), p, resume=j1["job"]["id"])
        self.assertEqual(j3["status"], "COMPLETED")
        self.assertEqual(len(j3["execution"]["skipped"]), 1)
        self.assertEqual([o.tool for o in ad3.calls if o.kind == "transform"], ["ffmpeg-skill/export"])
        self.assertTrue(j3["resume"]["plan_changed"])

    def test_revision_after_approval_needs_reapproval_and_keeps_history(self):
        svc = make_service(self.tmp)
        p = self._plan(svc)
        lead = self._dec(p, "silence.leading")
        svc.reject(load_ir(p), p, [lead["id"]], reason="r1")
        svc.revise(load_ir(p), p)
        svc.approve(load_ir(p), p, ["all"])
        self.assertFalse(load_ir(p).needs_reapproval())
        # a second round: structured feedback changes the loudness target
        out = svc.revise(load_ir(p), p, feedback="-16 に揃えて", user_requirements={"audio.loudness.target_lufs": -18}, who="editor")
        self.assertTrue(out["created"])
        ir = load_ir(p)
        self.assertEqual(ir.version, 3)
        self.assertTrue(ir.needs_reapproval())
        self.assertEqual(svc.render(ir, p)["status"], "WAITING_FOR_APPROVAL")
        self.assertEqual(ir.doc["audio"]["operations"][0]["target_lufs"], -18)
        self.assertTrue(any("AUDIO" in l and "-18" in l for l in ir.doc["revision"]["history"][-1]["diff"]["summary"]))
        # rejection reason from v1 survives two revisions
        self.assertEqual(ir.doc["execution"]["reviews"][lead["id"]]["reason"], "r1")
        self.assertEqual([h["version"] for h in ir.doc["revision"]["history"]], [2, 3])
        self.assertEqual(len(ir.doc["revision"]["feedback"]), 1)
        self.assertTrue(Path(out["snapshot"]).exists() and out["snapshot"].endswith(".v2.json"))
        self.assertTrue(Path(str(Path(p).with_name("p.v1.json"))).exists())

    def test_revise_twice_without_changes_is_idempotent(self):
        svc = make_service(self.tmp)
        p = self._plan(svc)
        lead = self._dec(p, "silence.leading")
        svc.reject(load_ir(p), p, [lead["id"]], reason="r1")
        first = svc.revise(load_ir(p), p)
        self.assertTrue(first["created"])
        before = json.loads(Path(p).read_text())
        again = svc.revise(load_ir(p), p)
        self.assertFalse(again["created"])
        self.assertEqual(json.loads(Path(p).read_text()), before, "IR untouched by a no-op revision")
        self.assertEqual(load_ir(p).version, 2)
        self.assertEqual(len(load_ir(p).doc["revision"]["history"]), 1)

    def test_feedback_without_plan_change_never_creates_an_empty_version(self):
        svc = make_service(self.tmp)
        p = self._plan(svc, profile="youtube")
        loud = self._dec(p, "audio.loudness")
        svc.reject(load_ir(p), p, [loud["id"]], reason="levels fine")
        self.assertTrue(svc.revise(load_ir(p), p)["created"])
        svc.approve(load_ir(p), p, ["all"])
        # asking for a different target re-proposes loudness, which was rejected → suppressed → no plan change → no version
        out = svc.revise(load_ir(p), p, feedback="少し下げて", user_requirements={"audio.loudness.target_lufs": -18})
        self.assertFalse(out["created"])
        self.assertIn("rejected earlier", out["reason"])
        ir = load_ir(p)
        self.assertEqual(ir.version, 2)
        self.assertEqual(len(ir.doc["revision"]["feedback"]), 1, "feedback is recorded once")
        self.assertFalse(ir.needs_reapproval(), "approval of v2 is untouched")
        again = svc.revise(load_ir(p), p, feedback="少し下げて", user_requirements={"audio.loudness.target_lufs": -18})
        self.assertFalse(again["created"])
        self.assertEqual(len(load_ir(p).doc["revision"]["feedback"]), 1, "identical feedback is not duplicated")
        self.assertEqual([h["version"] for h in load_ir(p).doc["revision"]["history"]], [2])

    def test_revise_without_rejections_or_feedback_creates_nothing(self):
        svc = make_service(self.tmp)
        p = self._plan(svc)
        out = svc.revise(load_ir(p), p)
        self.assertFalse(out["created"])
        self.assertEqual(load_ir(p).version, 1)
        self.assertFalse(Path(str(Path(p).with_name("p.v1.json"))).exists())

    def test_failure_and_resume_inside_a_revised_plan(self):
        failing = FakeAdapter(fail_tools={"ffmpeg-skill/export": 9})
        svc = make_service(self.tmp, adapter=failing)
        p = self._plan(svc, profile="youtube")
        lead = self._dec(p, "silence.leading")
        svc.reject(load_ir(p), p, [lead["id"]], reason="keep intro")
        svc.revise(load_ir(p), p)
        svc.approve(load_ir(p), p, ["all"])
        j1 = svc.render(load_ir(p), p)
        self.assertEqual(j1["status"], "FAILED")
        self.assertEqual(j1["job"]["plan_version"], 2)
        # v2 still trims the trailing silence (only the leading one was rejected)
        self.assertEqual(load_ir(p).doc["video"]["operations"][0]["keep"][0][0], 0.0)
        j2 = make_service(self.tmp, adapter=FakeAdapter()).render(load_ir(p), p, resume="last")
        self.assertEqual(j2["status"], "COMPLETED")
        self.assertEqual(len(j2["execution"]["skipped"]), 2, "cut (trailing) and loudness from the failed v2 job are reused")
        ir = load_ir(p)
        self.assertEqual([r["plan_version"] for r in ir.doc["provenance"]["runs"]], [2, 2])
        self.assertEqual(ir.doc["execution"]["reviews"][lead["id"]]["reason"], "keep intro")

    def test_revise_does_not_touch_media(self):
        ad = FakeAdapter()
        svc = make_service(self.tmp, adapter=ad)
        p = self._plan(svc)
        n = len(ad.calls)
        lead = self._dec(p, "silence.leading")
        svc.reject(load_ir(p), p, [lead["id"]], reason="x")
        svc.revise(load_ir(p), p)
        self.assertEqual(len(ad.calls), n, "revision is a pure re-plan from recorded observations")


class SkillToolBoundaryTests(unittest.TestCase):
    """Skill (what) → Capability (what is possible here) → Tool (what executes). The plan names the tool; the compiler never chooses."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)

    def test_select_tool_follows_capabilities_and_adapters(self):
        from video_agent.skills import default_registry
        reg = default_registry()
        caps = FakeCaps().resolve()
        self.assertEqual(reg.select_tool("silence_cleanup", caps, lambda t: True), ("ffmpeg-skill/cut", "ok"))
        tool, reason = reg.select_tool("silence_cleanup", FakeCaps(missing={"encoder:libx264"}).resolve(), lambda t: True)
        self.assertIsNone(tool)
        self.assertIn("encoder:libx264", reason)
        tool, reason = reg.select_tool("silence_cleanup", caps, lambda t: False)
        self.assertIsNone(tool)
        self.assertIn("no registered adapter", reason)
        tool, reason = reg.select_tool("multi_source_sync", caps, lambda t: True)
        self.assertIsNone(tool, "declared future skills are never selectable even when their tools exist")
        self.assertIn("not implemented", reason)

    def test_future_skills_are_listed_but_never_available(self):
        svc = make_service(self.tmp)
        rows = {r["skill"]: r for r in svc.skills()}
        self.assertEqual(rows["multi_source_sync"]["status"], "NOT_IMPLEMENTED")
        self.assertEqual(rows["caption_generation"]["status"], "NOT_IMPLEMENTED")
        self.assertEqual(rows["silence_cleanup"]["status"], "AVAILABLE")
        self.assertEqual(rows["silence_cleanup"]["tool"], "ffmpeg-skill/cut")
        self.assertNotIn("multi_source_sync", svc.tools_for())
        self.assertNotIn("caption_generation", svc.tools_for())

    def test_plan_steps_name_registry_selected_tools_and_compiler_uses_them(self):
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        steps = {s["skill"]: s["tool"] for s in ir.doc["plan"]["steps"]}
        self.assertEqual(steps, {"silence_cleanup": "ffmpeg-skill/cut", "loudness_normalization": "ffmpeg-skill/loudness", "delivery_export": "ffmpeg-skill/export", "delivery_check": "ffmpeg-skill/check"})
        ops, _ = compile_ir(ir, "/w/jobs/j")
        self.assertEqual([(o.skill, o.tool) for o in ops], [("silence_cleanup", "ffmpeg-skill/cut"), ("loudness_normalization", "ffmpeg-skill/loudness"), ("delivery_export", "ffmpeg-skill/export"), ("delivery_check", "ffmpeg-skill/check")])
        # the compiler follows the plan, not a literal: renaming the tool in the plan changes the compiled operation
        ir.doc["plan"]["steps"][0]["tool"] = "other-skill/trim"
        ops2, _ = compile_ir(ir, "/w/jobs/j")
        self.assertEqual(ops2[0].tool, "other-skill/trim")
        # ...and the validator refuses it because no adapter supports that tool / it is not a declared tool of the skill
        rep = validate_ir(ir, svc.caps.resolve(), registry=svc.registry, supports=lambda t: t.startswith("ffmpeg-skill/"))
        self.assertFalse(rep.ok)
        self.assertTrue(any("not a declared tool" in e for e in rep.errors), rep.errors)

    def test_validator_rejects_steps_without_tool_or_with_future_skill(self):
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        ir.doc["plan"]["steps"][1]["tool"] = None
        rep = validate_ir(ir, svc.caps.resolve(), registry=svc.registry, supports=lambda t: True)
        self.assertTrue(any("has no selected tool" in e or "plan/steps/1/tool" in e for e in rep.errors), rep.errors)  # schema rejects null first
        ir = svc.plan([self.src], "youtube")
        ir.doc["plan"]["steps"].append({"id": "step_sync", "skill": "multi_source_sync", "tool": "ffmpeg-skill/sync", "decision_ids": [], "params": {}})
        rep = validate_ir(ir, svc.caps.resolve(), registry=svc.registry, supports=lambda t: True)
        self.assertTrue(any("not implemented" in e for e in rep.errors), rep.errors)
        ir = svc.plan([self.src], "youtube")
        ir.doc["plan"]["steps"] = [s for s in ir.doc["plan"]["steps"] if s["skill"] != "silence_cleanup"]
        rep = validate_ir(ir, svc.caps.resolve(), registry=svc.registry, supports=lambda t: True)
        self.assertTrue(any("has no plan step" in e for e in rep.errors), rep.errors)

    def test_compiler_refuses_plan_without_tool(self):
        from video_agent.execution import CompileError
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        ir.doc["plan"]["steps"] = []
        with self.assertRaises(CompileError):
            compile_ir(ir, "/w/jobs/j")

    def test_missing_adapter_blocks_the_decision(self):
        class NoTools(FakeAdapter):
            def supports(self, tool):
                return tool.startswith("ffmpeg-skill/") and not tool.endswith("/export")
        svc = make_service(self.tmp, adapter=NoTools())
        ir = svc.plan([self.src], "youtube")
        blocked = [d for d in ir.doc["decisions"] if d["approval"] == "BLOCK"]
        self.assertTrue(blocked)
        self.assertIn("no registered adapter", blocked[0]["reason"])
        self.assertEqual(blocked[0]["params"]["skill"], "delivery_export")

    def test_missing_engine_gives_a_clear_error_not_a_keyerror(self):
        class Nothing(FakeAdapter):
            def supports(self, tool):
                return False
        svc = make_service(self.tmp, adapter=Nothing())
        with self.assertRaises(RuntimeError) as cm:
            svc.check(self.src, "youtube")
        self.assertIn("media_probe", str(cm.exception))
        self.assertIn("no registered adapter", str(cm.exception))
        with self.assertRaises(RuntimeError):
            svc.plan([self.src], "youtube")

    def test_router_dispatches_by_adapter_support(self):
        from video_agent.tools import ToolRouter, ToolError
        a = FakeAdapter()
        class Other(FakeAdapter):
            name = "other"
            def supports(self, tool):
                return tool.startswith("other/")
        router = ToolRouter([a, Other()])
        self.assertTrue(router.supports("ffmpeg-skill/cut"))
        self.assertTrue(router.supports("other/x"))
        self.assertFalse(router.supports("nope/x"))
        self.assertIs(router.adapter_for("other/x").__class__, Other)
        with self.assertRaises(ToolError):
            router.measure("nope/x", {})
        self.assertEqual(router.measure("ffmpeg-skill/probe", {"inputs": ["/x"]}).tool, "ffmpeg-skill/probe")

    def test_no_tool_id_literals_outside_tool_layer(self):
        """Orchestration code must not hard-code engine tool ids or carry a default engine map: tool ids come only from
        SkillRegistry.select_tool / resolve_tools. Allowed: skills/ (candidates), tools/ (adapters), recovery.py (alt args are tool knowledge)."""
        root = Path(__file__).resolve().parents[1] / "src" / "video_agent"
        offenders = []
        for py in root.rglob("*.py"):
            rel = py.relative_to(root).as_posix()
            text = py.read_text(encoding="utf-8")
            if "DEFAULT_TOOLS" in text:
                offenders.append(f"{rel}: DEFAULT_TOOLS")
            if rel.startswith(("tools/", "skills/")) or rel == "execution/recovery.py":
                continue
            for i, line in enumerate(text.splitlines(), 1):
                code = line.split("#", 1)[0]
                if '"ffmpeg-skill/' in code or "'ffmpeg-skill/" in code:
                    offenders.append(f"{rel}:{i}: {line.strip()}")
        self.assertEqual(offenders, [], "\n".join(offenders))

    # ---- the registry is the only tool selection point: components called directly must not fall back to an engine
    def test_planner_has_no_default_engine(self):
        from video_agent.agent.planner import build_plan
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        analysis = AnalysisResult.from_ir(ir.doc)
        decisions = [Decision.from_dict(d) for d in ir.doc["decisions"]]
        with self.assertRaises(TypeError):
            build_plan(decisions, analysis, None)
        plan = build_plan(decisions, analysis, {})
        self.assertTrue(plan["steps"], "steps are still emitted so the validator can name the missing tool")
        self.assertEqual([st["tool"] for st in plan["steps"]], [None] * len(plan["steps"]))
        self.assertTrue(any("no executable tool" in line for line in plan["summary"]))
        self.assertNotIn("ffmpeg-skill", json.dumps(plan))
        ir.doc["plan"] = plan
        rep = validate_ir(ir, svc.caps.resolve(), registry=svc.registry, supports=lambda t: True)
        self.assertFalse(rep.ok)
        with self.assertRaises(CompileError):
            compile_ir(ir, "/w/jobs/j")

    def test_analyzer_has_no_default_engine(self):
        from video_agent.media import MediaAnalyzer
        ad = FakeAdapter()
        with self.assertRaises(TypeError):
            MediaAnalyzer(ad, None)
        with self.assertRaises(ToolError) as cm:
            MediaAnalyzer(ad, {})
        self.assertIn("media_probe", str(cm.exception))
        with self.assertRaises(ToolError) as cm:
            MediaAnalyzer(ad, {"media_probe": "x/probe", "silence_analysis": "x/silence"})
        self.assertIn("loudness_analysis", str(cm.exception))
        self.assertEqual(ad.calls, [], "constructing the analyzer never touches a tool")

    def test_qa_has_no_default_engine(self):
        from video_agent.qa import run_qa
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        ad = FakeAdapter()
        with self.assertRaises(TypeError):
            run_qa(ad, ir.doc, {}, [], None)
        with self.assertRaises(ToolError) as cm:
            run_qa(ad, ir.doc, {}, [], {})
        self.assertIn("delivery_check", str(cm.exception))
        self.assertEqual(ad.calls, [], "QA without a tool map never measures anything")
        rep = run_qa(ad, ir.doc, {}, [], svc.tools_for(svc.adapter([])))
        self.assertEqual(rep.items, [], "no artifacts yet: nothing to check, no fallback engine consulted")

    def test_registry_selected_tool_propagates_to_the_adapter(self):
        """Registry → Service → planner (plan.steps[].tool) → compiler (Operation.tool) → ToolRouter → adapter, for a
        tool that is not ffmpeg-skill. Nothing downstream rewrites or defaults the tool id."""
        from video_agent.tools import ToolRouter
        class OtherEngine(FakeAdapter):
            name = "other-skill"
            version = "9.9-fake"
            ALIASES = {"trim": "cut"}
            def supports(self, tool):
                return tool.startswith("other-skill/")
        ff, other = FakeAdapter(), OtherEngine()
        svc = make_service(self.tmp, adapter=ToolRouter([ff, other]))
        svc.registry.get("silence_cleanup").tools = ["other-skill/trim", "ffmpeg-skill/cut"]
        self.assertEqual(svc.tools_for()["silence_cleanup"], "other-skill/trim")
        ir = svc.plan([self.src], "youtube")
        steps = {st["skill"]: st["tool"] for st in ir.doc["plan"]["steps"]}
        self.assertEqual(steps["silence_cleanup"], "other-skill/trim")
        self.assertEqual(steps["delivery_export"], "ffmpeg-skill/export")
        self.assertEqual(ir.doc["source"]["tool_versions"]["other-skill"], "9.9-fake")
        ops, _ = compile_ir(ir, "/w/jobs/j")
        self.assertEqual([(o.skill, o.tool) for o in ops if o.skill == "silence_cleanup"], [("silence_cleanup", "other-skill/trim")])
        p = str(Path(self.tmp) / "c.json")
        save_ir(ir, p)
        out = svc.render(ir, p, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED", out)
        self.assertEqual([o.tool for o in other.calls], ["other-skill/trim"], "the other engine ran exactly the trim")
        self.assertNotIn("other-skill/trim", [o.tool for o in ff.calls])
        prov = json.loads((Path(self.tmp) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        trim = next(e for e in prov["operations"] if e["skill"] == "silence_cleanup")
        self.assertEqual((trim["tool"], trim["tool_version"]), ("other-skill/trim", "9.9-fake"))
        exp = next(e for e in prov["operations"] if e["skill"] == "delivery_export")
        self.assertEqual((exp["tool"], exp["tool_version"]), ("ffmpeg-skill/export", "0.8.4-fake"))
        # the job's artifact record follows the export tool, not a fixed engine name
        self.assertEqual(out["job"]["artifacts"][0]["tool"], "ffmpeg-skill/export")


class EcosystemContractTests(unittest.TestCase):
    """AI Video Production Ecosystem: Skill package → Tool → Adapter → runtime, with ffmpeg-skill as the only implemented
    Reference Skill. Future skills exist only in docs; a fake package is registered in test scope only."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)

    # Test A — ffmpeg-skill is registered as the Reference Skill package
    def test_reference_skill_package_is_registered(self):
        from video_agent.tools.ffmpeg_skill import PACKAGE
        svc = make_service(self.tmp)
        pkgs = {p.skill_id: p for p in svc.registry.packages()}
        self.assertEqual(list(pkgs), ["ffmpeg-skill", "media-analysis", "transcription"], "implemented packages: the Reference Skill, the observation Skill (PR #12) and the recognition Skill (PR #13)")
        self.assertEqual(PACKAGE.validate(), [])
        self.assertEqual((PACKAGE.repository, PACKAGE.capabilities), ("kajisho5/ffmpeg-skill", ["ffmpeg", "ffprobe", "ffmpeg-skill"]))
        rows = {r["skill_id"]: r for r in svc.packages()}
        self.assertEqual(sorted(rows), ["ffmpeg-skill", "media-analysis", "transcription"])
        self.assertTrue(rows["ffmpeg-skill"]["implemented"] and rows["ffmpeg-skill"]["available"])
        self.assertEqual(rows["ffmpeg-skill"]["version"], "0.8.4-fake", "version comes from the adapter that detected the checkout")
        self.assertTrue(rows["media-analysis"]["implemented"] and not rows["media-analysis"]["available"], "adapter exists; no installation in unit tests")

    # Test B — its tools are recognised, typed, and consistent with the adapter
    def test_reference_skill_tools(self):
        from video_agent.tools.ffmpeg_skill import PACKAGE
        from video_agent.tools.ffmpeg_skill.catalog import CATALOG
        self.assertEqual(PACKAGE.tool_ids(), ["ffmpeg-skill/" + k for k in CATALOG])
        cut, probe = PACKAGE.tool("ffmpeg-skill/cut"), PACKAGE.tool("ffmpeg-skill/probe")
        self.assertEqual((cut.kind, cut.skill_id, cut.required_capabilities), ("transform", "ffmpeg-skill", ["encoder:libx264"]))
        self.assertEqual((probe.kind, probe.inputs), ("measure", ["inputs"]))
        svc = make_service(self.tmp)
        self.assertEqual(svc.registry.tool("ffmpeg-skill/cut").tool_id, "ffmpeg-skill/cut")
        self.assertIsNone(svc.registry.tool("ffmpeg-skill/nope"))
        self.assertIsNone(svc.registry.tool("media-analysis-skill/probe"), "future packages are not registered")

    # Test C — Skill (production) → Tool (package) → Adapter relation is closed: no dangling candidates
    def test_skill_tool_adapter_relation(self):
        svc = make_service(self.tmp)
        self.assertEqual(svc.registry.unknown_tool_candidates(), [], "every tool candidate belongs to a registered package")
        router = svc.adapter([])
        for spec in svc.registry.all():
            if not spec.implemented:
                continue   # declared future skills may cite tools that are not catalogued yet; they are never selected
            for t in spec.tools:
                self.assertEqual(t.split("/", 1)[0], svc.registry.tool(t).skill_id, "every candidate belongs to a registered package")
            selected = svc.tools_for(router).get(spec.name)
            if selected:
                self.assertIs(router.adapter_for(selected).__class__, FakeAdapter)
        rows = {r["skill"]: r for r in svc.skills()}
        self.assertEqual(rows["silence_cleanup"]["packages"], ["ffmpeg-skill"])

    # Test D — declared future skills are never AVAILABLE, and no future package exists
    def test_future_skills_never_available(self):
        svc = make_service(self.tmp)
        rows = {r["skill"]: r for r in svc.skills()}
        for name in ("multi_source_sync", "caption_generation", "semantic_deletion"):
            self.assertEqual((rows[name]["status"], rows[name]["implemented"]), ("NOT_IMPLEMENTED", False))
        src = Path(__file__).resolve().parents[1] / "src" / "video_agent"
        future = ("subtitle-skill", "video-editing-skill", "audio-production-skill",
                  "motion-graphics-skill", "color-grading-skill", "thumbnail-skill", "qc-skill")   # media-analysis-skill (PR #12) and transcription-skill (PR #13) are integrated
        hits = [f"{py.relative_to(src)}: {n}" for py in src.rglob("*.py") for n in future if n in py.read_text(encoding="utf-8")]
        self.assertEqual(hits, [], "future skill packages must not appear in production code")

    # Test E — tool selection has no default fallback (covered in depth by SkillToolBoundaryTests; contract-level check here)
    def test_no_default_tool_fallback(self):
        svc = make_service(self.tmp)
        self.assertEqual(svc.registry.resolve_tools(svc.caps.resolve(), lambda t: False), {})
        rows = {r["skill"]: r for r in svc.registry.availability(svc.caps.resolve(), lambda t: False)}
        self.assertTrue(all(r["tool"] is None for r in rows.values()))
        pk = svc.registry.package_availability(svc.caps.resolve(), lambda t: False)[0]
        self.assertFalse(pk["available"])
        self.assertIn("no registered adapter", pk["reason"])

    # Test H + I + F + G — a fake package registered in test scope propagates unchanged and needs no core change
    def test_fake_skill_package_in_test_scope_only(self):
        from video_agent.skills import SkillPackage, ToolSpec
        from video_agent.tools import ToolRouter

        class FakeSkillAdapter(FakeAdapter):
            name = "fake-skill"
            version = "0.1-test"
            ALIASES = {"tool": "cut"}
            TOOLS = ["tool"]
            def supports(self, tool):
                return tool == "fake-skill/tool"

        ff, fake = FakeAdapter(), FakeSkillAdapter()
        svc = make_service(self.tmp, adapter=ToolRouter([ff, fake]))
        pkg = fake.package()
        self.assertEqual((pkg.skill_id, pkg.tool_ids(), pkg.validate()), ("fake-skill", ["fake-skill/tool"], []))
        with self.assertRaises(ValueError):
            svc.registry.register_package(SkillPackage(skill_id="bad", name="bad", version="1", description="", tools=[ToolSpec(tool_id="other/x", skill_id="other")]))
        # the only "core" change a new package needs: a production skill cites its tool as a candidate
        svc.registry.get("silence_cleanup").tools = ["fake-skill/tool", "ffmpeg-skill/cut"]
        self.assertEqual([p.skill_id for p in svc.registry.packages()], ["fake-skill", "ffmpeg-skill", "media-analysis", "transcription"])
        self.assertEqual(svc.registry.unknown_tool_candidates(), [])
        ir = svc.plan([self.src], "youtube")
        self.assertTrue(svc.validate(ir).ok)
        self.assertEqual({st["skill"]: st["tool"] for st in ir.doc["plan"]["steps"]}["silence_cleanup"], "fake-skill/tool")
        ops, _ = compile_ir(ir, "/w/jobs/j")
        self.assertEqual([o.tool for o in ops if o.skill == "silence_cleanup"], ["fake-skill/tool"])
        p = str(Path(self.tmp) / "eco.json")
        save_ir(ir, p)
        out = svc.render(ir, p, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED", out)
        self.assertEqual([o.tool for o in fake.calls], ["fake-skill/tool"])
        prov = json.loads((Path(self.tmp) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        trim = next(e for e in prov["operations"] if e["skill"] == "silence_cleanup")
        self.assertEqual((trim["skill_package"], trim["tool"], trim["tool_version"]), ("fake-skill", "fake-skill/tool", "0.1-test"))
        self.assertEqual(prov["tool_versions"]["fake-skill"], "0.1-test")
        self.assertEqual(prov["skill_versions"]["silence_cleanup"], "1.0")
        # a plan citing a tool no registered package declares is rejected by the validator
        ir.doc["plan"]["steps"][0]["tool"] = "fake-skill/other"
        svc.registry.get("silence_cleanup").tools.append("fake-skill/other")
        rep = validate_ir(ir, svc.caps.resolve(), registry=svc.registry, supports=lambda t: True)
        self.assertTrue(any("not declared by any registered skill package" in e for e in rep.errors), rep.errors)
        # production registry of a fresh service knows nothing about the fake package
        self.assertEqual([p.skill_id for p in make_service(self.tmp).registry.packages()], ["ffmpeg-skill", "media-analysis", "transcription"])

    # Static architecture test — engine knowledge stays in skills/ tools/ recovery.py and the composition root
    def test_no_engine_leakage_in_orchestration(self):
        root = Path(__file__).resolve().parents[1] / "src" / "video_agent"
        allowed = ("tools/", "skills/", "capabilities/")
        allowed_files = {"execution/recovery.py", "service.py", "cli.py", "__init__.py"}
        offenders = []
        for py in root.rglob("*.py"):
            rel = py.relative_to(root).as_posix()
            if rel.startswith(allowed) or rel in allowed_files:
                continue
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                code = line.split("#", 1)[0].lower()
                if "ffmpeg" in code or "ffprobe" in code:   # capability names (encoder:libx264) are environment vocabulary, allowed
                    offenders.append(f"{rel}:{i}: {line.strip()}")
        self.assertEqual(offenders, [], "\n".join(offenders))


class AIProviderBoundaryTests(unittest.TestCase):
    """AI Provider Contract / Reasoning Boundary (ADR-018). AI is part of the Brain, never an execution authority."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)

    def _svc(self, provider, **kw):
        return Service(workspace=self.tmp, adapter=FakeAdapter(), caps=FakeCaps(), provider=provider, **kw)

    def _analysis(self):
        return make_service(self.tmp).analyze([self.src], "youtube")[2]

    # A — contract
    def test_fake_provider_satisfies_the_contract(self):
        from video_agent.providers import AIRequest, AIResponse, NullProvider, AIProviderError
        p = FakeAIProvider(recommendations=[])
        req = AIRequest(task_type="production_recommendation", inputs={"observations": []})
        resp = p.complete(req)
        self.assertIsInstance(resp, AIResponse)
        self.assertEqual((resp.provider, resp.model, resp.task_type), ("fake", "fake-model-1", "production_recommendation"))
        self.assertEqual(len(resp.response_hash()), 64)
        self.assertNotIn("api_key", json.dumps(p.describe()))
        with self.assertRaises(ValueError):
            AIRequest(task_type="run_ffmpeg", inputs={})
        with self.assertRaises(AIProviderError):
            NullProvider().complete(req)
        self.assertFalse(NullProvider().available())

    # B + C — AI cannot create observations; its data is AI_GENERATED
    def test_ai_cannot_fabricate_observations(self):
        from video_agent.agent.ai_reasoning import to_inferences
        from video_agent.providers import AIResponse
        analysis = self._analysis()
        n_obs = len(analysis.observations)
        hostile = AIResponse(task_type="production_recommendation", provider="fake", model="m", confidence=0.99,
                             result={"observations": [{"kind": "probe", "data": {"duration": 21.5}}],
                                     "recommendations": [{"intent": "silence_cleanup", "asset_id": analysis.assets[0].id, "statement": "x", "confidence": 0.9,
                                                          "evidence": ["obs_fabricated"], "provenance": "OBSERVED"},
                                                         {"intent": "silence_cleanup", "asset_id": analysis.assets[0].id, "statement": "ok", "confidence": 5,
                                                          "evidence": [analysis.observations[0].id], "provenance": "OBSERVED"}]})
        infs, warns = to_inferences(hostile, analysis, ["silence_cleanup"])
        self.assertEqual(len(analysis.observations), n_obs, "no observation was added")
        self.assertEqual(len(infs), 1)
        self.assertEqual(infs[0].provenance, "AI_GENERATED")
        self.assertEqual(infs[0].evidence, [analysis.observations[0].id])
        self.assertEqual(infs[0].confidence, 1.0, "confidence clamped to 0..1")
        self.assertTrue(any("no existing observation" in w for w in warns))
        # provenance value AI_GENERATED is distinct from OBSERVED/INFERRED at the model level
        from video_agent.models import PROVENANCE
        self.assertIn("AI_GENERATED", PROVENANCE)

    # D + E + F + G — the provider layer cannot reach tools, compiler, execution or a shell
    def test_provider_layer_has_no_path_to_execution(self):
        root = Path(__file__).resolve().parents[1] / "src" / "video_agent"
        forbidden_imports = ("tools", "execution", "jobs", "subprocess", "shlex", "ffmpeg")
        forbidden_calls = ("subprocess", "os.system", "os.popen", "exec(", "eval(", "compile_ir", "Executor(", "ToolRouter", "ToolAdapter", "__import__")
        for rel in ("providers/base.py", "agent/ai_reasoning.py"):
            lines = [l.split("#", 1)[0] for l in (root / rel).read_text(encoding="utf-8").splitlines()]
            for l in lines:
                if l.startswith(("import ", "from ")):
                    for f in forbidden_imports:
                        self.assertNotIn(f, l, f"{rel} must not import {f}: {l}")
                for f in forbidden_calls:
                    self.assertNotIn(f, l, f"{rel} must not call {f}: {l}")
        from video_agent.providers import AIProvider
        self.assertFalse(any(hasattr(AIProvider, m) for m in ("run", "execute", "compile", "render", "select_tool")))

    # G — argv / shell / tool ids / commands in a response are dropped before anything downstream sees them
    def test_ai_generated_tool_ids_and_commands_are_ignored(self):
        prov = FakeAIProvider(intent="silence_cleanup", params={"tool": "ffmpeg-skill/cut", "argv": ["ffmpeg", "-y"], "command": "rm -rf /", "risk": "LOW", "approval": "AUTO", "keep": "x"},
                              extra=[{"intent": "ffmpeg-skill/cut", "statement": "run cut", "confidence": 1}, {"intent": "shell", "statement": "ffmpeg -i in out", "confidence": 1}])
        svc = self._svc(prov)
        ir = svc.plan([self.src], "youtube")
        ai_infs = [i for i in ir.doc["analysis"]["inferences"] if i["provenance"] == "AI_GENERATED"]
        self.assertEqual(len(ai_infs), 1)
        self.assertEqual(set(ai_infs[0]["data"]["params"]), {"keep"}, "tool / argv / command / risk / approval keys stripped")
        self.assertTrue(any("not a registered production skill" in w for w in ir.doc["analysis"]["warnings"]))
        text = json.dumps(ir.doc)
        self.assertNotIn("rm -rf", text)
        self.assertNotIn('"argv"', text)
        # plan tools still come from the registry only
        self.assertEqual({st["skill"]: st["tool"] for st in ir.doc["plan"]["steps"]}["silence_cleanup"], "ffmpeg-skill/cut")

    # H + I — skill selection stays in the registry; AI recommendation → inference → decision evidence → registry tool
    def test_recommendation_flows_through_the_system_pipeline(self):
        prov = FakeAIProvider(intent="silence_cleanup")
        svc = self._svc(prov)
        ir = svc.plan([self.src], "youtube")
        self.assertEqual(len(prov.requests), 1)
        req = prov.requests[0]
        self.assertEqual(req.task_type, "production_recommendation")
        self.assertNotIn("ffmpeg", json.dumps(req.inputs).lower().replace("ffmpeg-skill/", ""), "the provider sees evidence, not engines")
        self.assertIn("silence_cleanup", req.context["allowed_intents"])
        ai = next(i for i in ir.doc["analysis"]["inferences"] if i["provenance"] == "AI_GENERATED")
        lead = next(d for d in ir.doc["decisions"] if d["subject"] == "silence.leading")
        self.assertIn(ai["id"], lead["evidence"], "corroborating recommendation is attached as evidence")
        self.assertEqual((lead["approval"], lead["risk"], lead["provenance"]), ("AUTO", "LOW", "INFERRED"), "measured decision unchanged")
        self.assertFalse([d for d in ir.doc["decisions"] if d["subject"].startswith("ai.")], "no separate review item when measurement covers it")
        steps = {st["skill"]: st["tool"] for st in ir.doc["plan"]["steps"]}
        self.assertEqual(steps["silence_cleanup"], "ffmpeg-skill/cut")
        self.assertTrue(svc.validate(ir).ok)
        ops, _ = compile_ir(ir, "/w/jobs/j")
        self.assertEqual([o.tool for o in ops if o.skill == "silence_cleanup"], ["ffmpeg-skill/cut"])

    # N — policy is not bypassed: an uncorroborated recommendation is a CONFIRM review item, not an operation
    def test_uncorroborated_recommendation_needs_confirmation_and_executes_nothing(self):
        prov = FakeAIProvider(intent="visual_inspection", params={"approval": "AUTO", "risk": "LOW"})
        svc = self._svc(prov)
        ir = svc.plan([self.src], "youtube")
        review = next(d for d in ir.doc["decisions"] if d["subject"] == "ai.visual_inspection")
        self.assertEqual((review["approval"], review["provenance"], review["params"]["executable"]), ("CONFIRM", "AI_GENERATED", False))
        self.assertEqual(review["risk"], svc.registry.get("visual_inspection").risk_level)
        self.assertFalse([st for st in ir.doc["plan"]["steps"] if st["skill"] == "visual_inspection"])
        p = str(Path(self.tmp) / "ai.json")
        save_ir(ir, p)
        out = svc.render(ir, p)
        self.assertEqual(out["status"], "WAITING_FOR_APPROVAL", out.get("status"))
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED")
        self.assertFalse([o for o in out["execution"]["results"] if "look" in o["tool"]], "approving a review item still runs no operation for it")

    # N — BLOCK from policy / capability wins over an AI "go"
    def test_ai_cannot_bypass_block(self):
        prov = FakeAIProvider(intent="delivery_export", params={"approval": "AUTO"})
        svc = Service(workspace=self.tmp, adapter=FakeAdapter(), caps=FakeCaps(missing={"encoder:libx264"}), provider=prov)
        ir = svc.plan([self.src], "youtube")
        self.assertTrue(ir.blocked())
        p = str(Path(self.tmp) / "blk.json")
        save_ir(ir, p)
        out = svc.render(ir, p, approve=["all"])
        self.assertIn(out["status"], ("BLOCKED", "FAILED"))
        self.assertIsNone(out.get("execution"), "nothing executed")

    # J — provider failures are AI-domain, degrade to a deterministic plan, and are recorded once (no retry)
    def test_provider_failure_is_recorded_and_not_retried(self):
        for kind in ("TIMEOUT", "RATE_LIMIT", "UNAVAILABLE", "AUTH", "crash"):
            prov = FakeAIProvider(fail=kind)
            svc = self._svc(prov)
            ir = svc.plan([self.src], "youtube")
            self.assertEqual(len(prov.requests), 1, f"{kind}: exactly one attempt")
            calls = ir.doc["provenance"]["ai_calls"]
            self.assertEqual(len(calls), 1)
            self.assertFalse(calls[0]["ok"])
            self.assertEqual(calls[0]["error"]["kind"], "MALFORMED" if kind == "crash" else kind)
            self.assertTrue(any("provider fake failed" in w for w in ir.doc["analysis"]["warnings"]))
            self.assertFalse([i for i in ir.doc["analysis"]["inferences"] if i["provenance"] == "AI_GENERATED"])
            self.assertTrue([st for st in ir.doc["plan"]["steps"]], "deterministic plan still produced")
            self.assertNotIn("INCIDENT", json.dumps(ir.doc["qa"]))
        # malformed structured result
        prov = FakeAIProvider(raw_result={"garbage": 1})
        ir = self._svc(prov).plan([self.src], "youtube")
        self.assertTrue(any("no recommendations list" in w for w in ir.doc["analysis"]["warnings"]))

    # K — budget
    def test_ai_call_budget_stops_calls(self):
        from video_agent.agent.ai_reasoning import AIReasoner, build_request
        from video_agent.providers import AIProviderError
        analysis = self._analysis()
        prov = FakeAIProvider(recommendations=[])
        r = AIReasoner(prov, max_calls=2)
        req = build_request(analysis, ["silence_cleanup"])
        r.ask(req)
        r.ask(req)
        with self.assertRaises(AIProviderError) as cm:
            r.ask(req)
        self.assertEqual(cm.exception.kind, "BUDGET")
        self.assertEqual(len(prov.requests), 2, "the provider is not called once the budget is spent")
        self.assertEqual([c["ok"] for c in r.calls], [True, True, False])
        self.assertEqual(r.calls[-1]["error"]["kind"], "BUDGET")
        # policy: max_ai_calls = 0 means no AI call at all
        svc = self._svc(FakeAIProvider(intent="silence_cleanup"))
        ir = svc.plan([self.src], "youtube", user_requirements=None)
        self.assertEqual(len(ir.doc["provenance"]["ai_calls"]), 1)
        # revisions reuse recorded AI inferences and spend no calls
        p = str(Path(self.tmp) / "b.json")
        save_ir(ir, p)
        svc.reject(load_ir(p), p, ["all"], reason="test")
        svc.revise(load_ir(p), p)
        v2 = load_ir(p)
        self.assertEqual(len(svc.provider.requests), 1, "revise did not call the provider")
        self.assertEqual(len(v2.doc["provenance"]["ai_calls"]), 1)

    # L + M — provenance carries provider / model / hash / usage / latency and never a secret
    def test_ai_provenance_without_secrets(self):
        prov = FakeAIProvider(intent="silence_cleanup")
        svc = self._svc(prov)
        ir = svc.plan([self.src], "youtube")
        p = str(Path(self.tmp) / "p.json")
        save_ir(ir, p)
        out = svc.render(ir, p, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED")
        job_prov = json.loads((Path(self.tmp) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        c = job_prov["ai_calls"][0]
        self.assertEqual((c["provider"], c["model"], c["task_type"], c["ok"]), ("fake", "fake-model-1", "production_recommendation", True))
        self.assertEqual(len(c["response_hash"]), 64)
        self.assertEqual(c["usage"]["input_tokens"], 120)
        self.assertIn("latency_s", c)
        self.assertEqual(job_prov["ai_provider"]["provider"], "fake")
        self.assertTrue(job_prov["plan_hash"] and job_prov["ir_hash"])
        for text in (Path(p).read_text(), json.dumps(job_prov), json.dumps(out, default=str)):
            self.assertNotIn(prov.api_key, text)
            self.assertNotIn("SECRET", text)
        # validator: an observation claiming an AI source is rejected
        ir.doc["analysis"]["observations"][0]["source"] = "ai:fake"
        rep = validate_ir(ir, svc.caps.resolve(), registry=svc.registry, supports=lambda t: True)
        self.assertTrue(any("only tool measurements may be OBSERVED" in e for e in rep.errors))

    # O — NullProvider: no AI, byte-identical behaviour for the deterministic pipeline
    def test_null_provider_changes_nothing(self):
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        self.assertEqual(ir.doc["provenance"]["ai_calls"], [])
        self.assertIsNone(ir.doc["provenance"]["ai_provider"])
        self.assertFalse([i for i in ir.doc["analysis"]["inferences"] if i["provenance"] == "AI_GENERATED"])


class ObservationAnalysisTests(unittest.TestCase):
    """Observation / Analysis architecture (ADR-019): AnalysisRequest → Analyzer → validated Observation → cache → evidence."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)

    def _analyzer(self, adapter=None, cache=True, **kw):
        svc = make_service(self.tmp, adapter=adapter)
        from video_agent.media import MediaAnalyzer
        return svc, MediaAnalyzer(svc.adapter([]), tools=svc.tools_for(), cache_dir=self.tmp if cache else None, **kw)

    # 1-2 request / kind validation
    def test_analysis_request_and_kind_validation(self):
        from video_agent.media import AnalysisRequest, AnalysisError, ANALYSIS_KINDS
        from video_agent.media.analysis import CORE_KINDS
        self.assertEqual(CORE_KINDS, ("media_probe", "silence", "loudness"), "FULL runs the core kinds")
        self.assertEqual(sorted(ANALYSIS_KINDS), ["audio_format", "duration", "integrity", "loudness", "media_probe", "scene_detection", "silence", "stream_layout", "timing", "transcript", "video_format"],
                         "every kind maps to a production skill with a real tool (media-analysis-skill contract@1)")
        r = AnalysisRequest(inputs=[self.src], kinds=["silence"], strategy="FULL_ANALYSIS")
        self.assertEqual((r.strategy, r.kinds, r.cache_policy), ("FULL", ["media_probe", "silence"], "use"))
        self.assertEqual(AnalysisRequest(inputs=[self.src], strategy="CACHED_ONLY").cache_policy, "only")
        for bad in ({"kinds": ["speaker_detection"]}, {"strategy": "GUESS"}, {"cache_policy": "maybe"}, {"inputs": []}):
            with self.assertRaises(AnalysisError) as cm:
                AnalysisRequest(**{"inputs": [self.src], **bad})
            self.assertIn(cm.exception.kind, ("ANALYSIS_UNSUPPORTED", "ANALYSIS_INVALID_RESULT"))
        self.assertEqual(r.analysis_id[:4], "ana_")

    # 3 analyzer contract
    def test_analyzer_contract(self):
        from video_agent.media import Analyzer, MediaAnalyzer
        svc, an = self._analyzer()
        self.assertIsInstance(an, Analyzer)
        self.assertEqual((an.id, an.version, an.identity), ("media", "1.0", "media@1.0"))
        self.assertEqual(an.supported_kinds[:3], ("media_probe", "silence", "loudness"))
        self.assertEqual(len(an.supported_kinds), 11)
        with self.assertRaises(NotImplementedError):
            Analyzer().analyze(None)
        self.assertFalse(any(hasattr(Analyzer, m) for m in ("decide", "approve", "compile", "render", "complete")))

    # 4-10 observation validation
    def test_observation_validation(self):
        from video_agent.media import AnalysisRequest, validate_observation
        from video_agent.models import Observation
        req = AnalysisRequest(inputs=[self.src], kinds=["silence"])
        ok = Observation(kind="silence", asset_id="a1", source="ffmpeg-skill/silence@0.8.4", data={"silences": []}, analysis_id=req.analysis_id)
        self.assertEqual(validate_observation(ok, req, ["a1"], "silence"), [])
        cases = {
            "wrong asset": Observation(kind="silence", asset_id="zz", source="ffmpeg-skill/silence@0.8.4", data={}, analysis_id=req.analysis_id),
            "wrong kind": Observation(kind="loudness", asset_id="a1", source="ffmpeg-skill/loudness@0.8.4", data={}, analysis_id=req.analysis_id),
            "fake source": Observation(kind="silence", asset_id="a1", source="ai:fake", data={}, analysis_id=req.analysis_id),
            "no version": Observation(kind="silence", asset_id="a1", source="ffmpeg-skill/silence", data={}, analysis_id=req.analysis_id),
            "missing field": Observation(kind="", asset_id="a1", source="ffmpeg-skill/silence@0.8.4", data={}, analysis_id=req.analysis_id),
            "other analysis": Observation(kind="silence", asset_id="a1", source="ffmpeg-skill/silence@0.8.4", data={}, analysis_id="ana_other"),
            "ai provenance": Observation(kind="silence", asset_id="a1", source="ffmpeg-skill/silence@0.8.4", data={}, analysis_id=req.analysis_id, provenance="AI_GENERATED"),
            "secret leak": Observation(kind="silence", asset_id="a1", source="ffmpeg-skill/silence@0.8.4", data={"api_key": "sk-abc"}, analysis_id=req.analysis_id),
            "command leak": Observation(kind="silence", asset_id="a1", source="ffmpeg-skill/silence@0.8.4", data={"note": "ffmpeg -i in.mp4 out.mp4"}, analysis_id=req.analysis_id),
            "argv leak": Observation(kind="silence", asset_id="a1", source="ffmpeg-skill/silence@0.8.4", data={"argv": ["x"]}, analysis_id=req.analysis_id),
            "malformed data": Observation(kind="silence", asset_id="a1", source="ffmpeg-skill/silence@0.8.4", data="not a dict", analysis_id=req.analysis_id),
        }
        for name, obs in cases.items():
            self.assertTrue(validate_observation(obs, req, ["a1"], "silence"), f"{name} must be rejected")
        self.assertEqual(validate_observation({"kind": "silence"}, req, ["a1"]), ["result is not an Observation"])

    # 11-13 strategies
    def test_full_targeted_cached_only(self):
        from video_agent.media import AnalysisRequest, AnalysisError
        svc, an = self._analyzer()
        full = an.run(AnalysisRequest(inputs=[self.src], strategy="FULL"))
        self.assertEqual(sorted(o.kind for o in full.observations), ["loudness", "media_probe", "silence"])
        self.assertEqual((full.strategy, an.measure_calls), ("FULL_ANALYSIS", 3))
        svc2, an2 = self._analyzer(cache=False)
        tgt = an2.run(AnalysisRequest(inputs=[self.src], strategy="TARGETED", kinds=["silence"]))
        self.assertEqual(sorted(o.kind for o in tgt.observations), ["media_probe", "silence"])
        self.assertEqual((tgt.strategy, an2.measure_calls), ("TARGETED_ANALYSIS", 2))
        # TARGETED kinds come from requirements, not from anyone else
        ir = make_service(self.tmp).plan([self.src], "youtube", user_requirements={"audio.normalize": False})
        self.assertEqual(sorted(ir.doc["analysis"]["analyses"][0]["request"]["kinds"]), ["media_probe", "silence"])
        # CACHED_ONLY: served from cache after FULL, no tool call; fails explicitly when nothing is cached
        _, an3 = self._analyzer()
        cached = an3.run(AnalysisRequest(inputs=[self.src], strategy="CACHED_ONLY"))
        self.assertEqual((an3.measure_calls, cached.strategy, len(cached.observations)), (0, "CACHED_ONLY", 3))
        self.assertTrue(all(r["cache_hit"] for r in cached.analyses[0]["rows"]))
        _, an4 = self._analyzer(cache=False)
        with self.assertRaises(AnalysisError) as cm:
            an4.run(AnalysisRequest(inputs=[self.src], strategy="CACHED_ONLY"))
        self.assertEqual(cm.exception.kind, "ANALYZER_UNAVAILABLE")
        self.assertEqual(an4.measure_calls, 0, "CACHED_ONLY never runs the analyzer")

    # 14-18, 22 cache
    def test_cache_hit_miss_invalidation_and_determinism(self):
        from video_agent.media import AnalysisRequest
        from video_agent.media.analysis import cache_key
        k = cache_key("fp1", "silence", "media@1.0", "ffmpeg-skill/silence@0.8.4", {"threshold_db": -40.0})
        self.assertEqual(k, cache_key("fp1", "silence", "media@1.0", "ffmpeg-skill/silence@0.8.4", {"threshold_db": -40.0}), "deterministic")
        self.assertNotEqual(k, cache_key("fp2", "silence", "media@1.0", "ffmpeg-skill/silence@0.8.4", {"threshold_db": -40.0}), "asset content")
        self.assertNotEqual(k, cache_key("fp1", "silence", "media@1.1", "ffmpeg-skill/silence@0.8.4", {"threshold_db": -40.0}), "analyzer version")
        self.assertNotEqual(k, cache_key("fp1", "silence", "media@1.0", "ffmpeg-skill/silence@0.8.5", {"threshold_db": -40.0}), "tool version")
        self.assertNotEqual(k, cache_key("fp1", "silence", "media@1.0", "ffmpeg-skill/silence@0.8.4", {"threshold_db": -30.0}), "parameters")
        svc, an = self._analyzer()
        r1 = an.run(AnalysisRequest(inputs=[self.src]))
        self.assertEqual((an.measure_calls, an.cache.hits, an.cache.misses), (3, 0, 3), "first run: miss + analyzer")
        _, an2 = self._analyzer()
        r2 = an2.run(AnalysisRequest(inputs=[self.src]))
        self.assertEqual((an2.measure_calls, an2.cache.hits), (0, 3), "second run: hit, analyzer not executed")
        self.assertEqual([o.id for o in r1.observations], [o.id for o in r2.observations], "cached observations keep their identity")
        self.assertNotEqual(r1.analyses[0]["analysis_id"], r2.analyses[0]["analysis_id"], "observation id and cache key are not the analysis id")
        self.assertEqual(r2.analyses[0]["rows"][0]["produced_by"]["analyzer"], "media@1.0")
        # parameter change -> miss
        _, an3 = self._analyzer()
        an3.run(AnalysisRequest(inputs=[self.src], params={"threshold_db": -30.0}))
        self.assertEqual(an3.measure_calls, 1, "only the silence measurement (threshold parameter) re-ran")
        # analyzer version change -> miss
        _, an4 = self._analyzer()
        an4.version = "9.9"
        an4.run(AnalysisRequest(inputs=[self.src]))
        self.assertEqual(an4.measure_calls, 3)
        # asset content change -> miss (fingerprint = sha256, or size:mtime with --no-hash)
        Path(self.src).write_bytes(b"\x01" * 65)
        _, an5 = self._analyzer()
        an5.run(AnalysisRequest(inputs=[self.src]))
        self.assertEqual(an5.measure_calls, 3)
        _, an6 = self._analyzer()
        an6.run(AnalysisRequest(inputs=[self.src], hash_sources=False))
        self.assertEqual(an6.measure_calls, 3, "no-hash fingerprint is a different key than the sha256 one")
        # bypass policy neither reads nor writes
        _, an7 = self._analyzer()
        an7.run(AnalysisRequest(inputs=[self.src], cache_policy="bypass"))
        self.assertEqual((an7.measure_calls, an7.cache.hits), (3, 0))
        # corrupt cache record -> ANALYSIS_CACHE_INVALID is recorded and the analyzer re-measures
        _, an8 = self._analyzer()
        an8.run(AnalysisRequest(inputs=[self.src]))
        for f in (Path(self.tmp) / "cache" / "observations").glob("*.json"):
            f.write_text("{broken")
        _, an9 = self._analyzer()
        r9 = an9.run(AnalysisRequest(inputs=[self.src]))
        self.assertEqual(an9.measure_calls, 3)
        self.assertTrue(all("ANALYSIS_CACHE_INVALID" in r.get("warning", "") for r in r9.analyses[0]["rows"]))

    # 19 budget
    def test_analysis_budget_is_enforced_and_separate_from_ai_budget(self):
        from video_agent.media import AnalysisRequest, AnalysisBudget, AnalysisError
        _, an = self._analyzer(cache=False)
        r = an.run(AnalysisRequest(inputs=[self.src], budget=AnalysisBudget(max_analysis_calls=1)))
        self.assertEqual(an.measure_calls, 1, "the second measurement is not made")
        rows = {x["kind"]: x for x in r.analyses[0]["rows"]}
        self.assertEqual(rows["silence"]["status"], "FAILED")
        self.assertEqual(rows["silence"]["error"]["kind"], "ANALYSIS_BUDGET_EXCEEDED")
        self.assertEqual([o.kind for o in r.observations], ["media_probe"], "no fabricated observation")
        self.assertEqual(r.analyses[0]["status"], "FAILED")
        self.assertTrue(any("ANALYSIS_BUDGET_EXCEEDED" in w for w in r.warnings))
        _, an2 = self._analyzer(cache=False)
        with self.assertRaises(AnalysisError) as cm:   # no time budget left: nothing runs; the probe cannot be skipped, so the failure is explicit
            an2.run(AnalysisRequest(inputs=[self.src], budget=AnalysisBudget(max_total_seconds=0.0)))
        self.assertEqual((cm.exception.kind, an2.measure_calls), ("ANALYSIS_BUDGET_EXCEEDED", 0))
        # unsupported budget keys are refused, not silently ignored
        from video_agent.policy.rules import Rule, resolve_rules
        rules = resolve_rules([Rule(id="t", kind="POLICY", scope="test", key="analysis.budget.max_bytes_scanned", value=10)])
        with self.assertRaises(AnalysisError) as cm:
            AnalysisBudget.from_rules(rules)
        self.assertEqual(cm.exception.kind, "ANALYSIS_UNSUPPORTED")
        rules = resolve_rules([Rule(id="t", kind="POLICY", scope="test", key="analysis.budget.max_processing_time", value=600)])
        self.assertEqual(AnalysisBudget.from_rules(rules).max_total_seconds, 600.0, "legacy key is now enforced as wall-clock seconds")
        # AI call budget is a different mechanism: an exhausted analysis budget does not touch ai_calls, and vice versa
        prov = FakeAIProvider(intent="silence_cleanup")
        svc = Service(workspace=self.tmp, adapter=FakeAdapter(), caps=FakeCaps(), provider=prov)
        ir = svc.plan([self.src], "youtube")
        self.assertEqual(len(ir.doc["provenance"]["ai_calls"]), 1)
        self.assertEqual(ir.doc["analysis"]["budget"]["calls"], 3)
        self.assertNotIn("ai", json.dumps(ir.doc["analysis"]["budget"]))

    # 20-21 unsupported / analyzer failure
    def test_unsupported_and_analyzer_failure(self):
        from video_agent.media import AnalysisRequest, AnalysisError
        with self.assertRaises(AnalysisError) as cm:
            AnalysisRequest(inputs=[self.src], kinds=["speaker_detection"])
        self.assertEqual(cm.exception.kind, "ANALYSIS_UNSUPPORTED")
        # a kind whose Skill is not installed here is refused per measurement, not by fabricating an observation
        _, an0 = self._analyzer(cache=False)
        r0 = an0.run(AnalysisRequest(inputs=[self.src], kinds=["scene_detection"]))
        row0 = next(x for x in r0.analyses[0]["rows"] if x["kind"] == "scene_detection")
        self.assertEqual((row0["status"], row0["error"]["kind"]), ("FAILED", "ANALYZER_UNAVAILABLE"))
        self.assertEqual([o.kind for o in r0.observations], ["media_probe"])
        _, an = self._analyzer(adapter=FakeAdapter(fail_tools={"ffmpeg-skill/loudness": 9}), cache=False)
        r = an.run(AnalysisRequest(inputs=[self.src]))
        rows = {x["kind"]: x for x in r.analyses[0]["rows"]}
        self.assertEqual(rows["loudness"]["status"], "FAILED")
        self.assertEqual(rows["loudness"]["error"]["kind"], "ANALYZER_UNAVAILABLE")
        self.assertEqual(sorted(o.kind for o in r.observations), ["media_probe", "silence"], "partial results are not stored as loudness observations")
        self.assertNotIn("INCIDENT", json.dumps(r.to_dict()).upper().replace("INCIDENTS", ""))
        # the plan still works deterministically on the remaining evidence
        svc = make_service(self.tmp, adapter=FakeAdapter(fail_tools={"ffmpeg-skill/loudness": 9}))
        ir = svc.plan([self.src], "youtube")
        self.assertTrue(any("loudness analysis failed" in w for w in ir.doc["analysis"]["warnings"]))
        self.assertTrue([st for st in ir.doc["plan"]["steps"] if st["skill"] == "silence_cleanup"])
        # a probe failure cannot be worked around: explicit analysis error, no asset
        _, an2 = self._analyzer(adapter=FakeAdapter(fail_tools={"ffmpeg-skill/probe": 9}), cache=False)
        with self.assertRaises(AnalysisError):
            an2.run(AnalysisRequest(inputs=[self.src]))

    # analysis provenance
    def test_analysis_provenance_is_tracked_and_separate_from_ai_calls(self):
        prov = FakeAIProvider(intent="silence_cleanup")
        svc = Service(workspace=self.tmp, adapter=FakeAdapter(), caps=FakeCaps(), provider=prov)
        ir = svc.plan([self.src], "youtube")
        an = ir.doc["analysis"]["analyses"][0]
        for k in ("analysis_id", "request", "analyzer", "started_at", "completed_at", "status", "rows", "budget", "cache"):
            self.assertIn(k, an)
        for o in ir.doc["analysis"]["observations"]:
            self.assertEqual((o["analysis_id"], o["analyzer"], o["provenance"]), (an["analysis_id"], "media@1.0", "OBSERVED"))
            self.assertTrue(o["cache_key"])
            self.assertTrue(o["source"].endswith("@0.8.4-fake"))
        self.assertNotIn("ai", json.dumps(an))
        self.assertNotIn("analysis_id", json.dumps(ir.doc["provenance"]["ai_calls"]))
        self.assertTrue(svc.validate(ir).ok)
        # revision reuses recorded observations and analyses; the analyzer is not re-run
        p = str(Path(self.tmp) / "r.json")
        save_ir(ir, p)
        n = len(svc.adapter([]).adapters[0].calls)
        svc.reject(load_ir(p), p, ["all"], reason="t")
        svc.revise(load_ir(p), p)
        self.assertEqual(len(svc.adapter([]).adapters[0].calls), n)
        v2 = load_ir(p)
        self.assertEqual(v2.doc["analysis"]["analyses"][0]["analysis_id"], an["analysis_id"])

    # 23-24 AI evidence boundary
    def test_ai_evidence_only_real_scrubbed_observations(self):
        from video_agent.agent.ai_reasoning import build_request, to_inferences
        from video_agent.media.analysis import safe_observation_summary
        from video_agent.models import Observation
        from video_agent.providers import AIResponse
        svc = make_service(self.tmp)
        analysis = svc.analyze([self.src], "youtube")[2]
        real_ids = {o.id for o in analysis.observations}
        # a non-tool observation smuggled into the analysis is not offered as evidence
        analysis.observations.append(Observation(kind="media_probe", asset_id=analysis.assets[0].id, source="ai:fake", data={"duration": 999}))
        analysis.observations.append(Observation(kind="loudness", asset_id=analysis.assets[0].id, source="ffmpeg-skill/loudness@0.8.4-fake", data={"lufs": -5}, provenance="AI_GENERATED"))
        analysis.observations[0].data["api_key"] = "sk-LEAK"
        analysis.observations[0].data["note"] = "ffmpeg -i x y"
        req = build_request(analysis, ["silence_cleanup"])
        ids = {o["id"] for o in req.inputs["observations"]}
        self.assertEqual(ids, real_ids)
        self.assertNotIn("sk-LEAK", json.dumps(req.inputs))
        self.assertNotIn("api_key", json.dumps(req.inputs))
        self.assertNotIn("ffmpeg -i", json.dumps(req.inputs))
        self.assertIsNone(safe_observation_summary({"id": "x", "kind": "media_probe", "source": "ai:fake", "data": {}}))
        # the AI cannot reference the smuggled ids either, and cannot create observations
        n = len(analysis.observations)
        resp = AIResponse(task_type="production_recommendation", provider="fake", model="m", confidence=0.9,
                          result={"observations": [{"kind": "media_probe", "data": {"duration": 1}}],
                                  "recommendations": [{"intent": "silence_cleanup", "asset_id": analysis.assets[0].id, "statement": "x", "confidence": 0.9, "evidence": [analysis.observations[-1].id]}]})
        infs, warns = to_inferences(resp, analysis, ["silence_cleanup"])
        self.assertEqual(infs, [])
        self.assertEqual(len(analysis.observations), n)

    # security: the analysis layer stays on the deterministic tool boundary
    def test_analysis_layer_has_no_path_to_execution_or_ai(self):
        root = Path(__file__).resolve().parents[1] / "src" / "video_agent"
        forbidden_imports = ("execution", "jobs", "providers", "agent", "subprocess", "shlex", "ffmpeg_skill")
        forbidden_calls = ("subprocess", "os.system", "os.popen", "exec(", "eval(", "compile_ir", "Executor(", "ToolRouter(", "__import__", ".complete(", ".run(op")
        for rel in ("media/analysis.py", "media/analyzer.py"):
            for l in (root / rel).read_text(encoding="utf-8").splitlines():
                code = l.split("#", 1)[0]
                if code.startswith(("import ", "from ")):
                    for f in forbidden_imports:
                        self.assertNotIn(f, code, f"{rel} must not import {f}: {l}")
                for f in forbidden_calls:
                    self.assertNotIn(f, code, f"{rel} must not call {f}: {l}")
        # the only execution surface is ToolAdapter.measure (registry-selected tool ids)
        text = (root / "media/analyzer.py").read_text(encoding="utf-8")
        self.assertIn("self.adapter.measure(tool, args)", text)


class TemporalEventSessionTests(unittest.TestCase):
    """Temporal / Event / Session architecture (ADR-020): time as a first-class domain model, distinct from observations,
    inferences and decisions."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)

    def _analysis(self, svc=None):
        svc = svc or make_service(self.tmp)
        return svc, svc.analyze([self.src], "youtube")[2]

    # 1-3 TimePoint / TemporalRange
    def test_time_point_and_range_validation(self):
        from video_agent.models import TimePoint, TimeRange, TemporalRange
        self.assertEqual(TimePoint(1.5).seconds, 1.5)
        self.assertEqual(TimePoint(-1e-9).seconds, 0.0, "float noise below zero is clamped, real negatives are rejected")
        for bad in (-1, "x", float("nan")):
            with self.assertRaises(ValueError):
                TimePoint(bad)
        r = TimeRange(1.0, 2.5)
        self.assertEqual((r.is_point, r.duration, r.stop), (False, 1.5, 2.5))
        pt = TimeRange(3.0)
        self.assertEqual((pt.is_point, pt.stop, pt.to_dict()), (True, 3.0, {"start": 3.0, "end": None}))
        with self.assertRaises(ValueError):
            TimeRange(2.0, 1.0)
        with self.assertRaises(ValueError):
            TimeRange(-0.5, 1.0)
        self.assertIs(TemporalRange, TimeRange)
        self.assertEqual(TimeRange(1.0, 1.0 - 1e-9).end, 1.0, "sub-epsilon inversion is float noise, normalised")
        self.assertTrue(TimeRange(0, 5).within(5.0000001) and not TimeRange(0, 5.1).within(5.0) and TimeRange(0, 99).within(None))

    # 15-17 relations, 14 ordering
    def test_relations_and_ordering(self):
        from video_agent.models import Event, TimeRange
        from video_agent.temporal import adjacent, contains, overlaps, precedes, sort_events
        a, b, c, d = TimeRange(0, 5), TimeRange(4, 8), TimeRange(5, 8), TimeRange(1, 2)
        self.assertTrue(a.overlaps(b) and b.overlaps(a) and not a.overlaps(c) and a.contains(d) and not d.contains(a) and a.precedes(c) and not a.precedes(b) and a.adjacent(c))
        self.assertTrue(TimeRange(2.0).overlaps(a) and a.overlaps(TimeRange(2.0)) and not TimeRange(6.0).overlaps(a))
        mk = lambda t, s, e, i: Event(type=t, timeline_id="asset:a", range=TimeRange(s, e).to_dict(), source="ffmpeg-skill/silence@0.8.4", kind="OBSERVED", id=i)  # noqa: E731
        e1, e2, e3, e4 = mk("AUDIO_SILENCE", 0, 5, "evt_b"), mk("AUDIO_ACTIVE", 4, 8, "evt_a"), mk("AUDIO_SILENCE", 0, 5, "evt_a"), mk("AUDIO_ACTIVE", 0, 5, "evt_z")
        self.assertTrue(overlaps(e1, e2) and contains(e1, mk("AUDIO_ACTIVE", 1, 2, "evt_x")) and precedes(e1, mk("AUDIO_ACTIVE", 5, 6, "evt_y")) and adjacent(e1, mk("AUDIO_ACTIVE", 5, 6, "evt_y")))
        order = [e.id + "/" + e.type for e in sort_events([e2, e4, e1, e3])]
        self.assertEqual(order, ["evt_z/AUDIO_ACTIVE", "evt_a/AUDIO_SILENCE", "evt_b/AUDIO_SILENCE", "evt_a/AUDIO_ACTIVE"], "start, end, type, id — stable across runs")
        self.assertEqual(order, [e.id + "/" + e.type for e in sort_events([e1, e2, e3, e4])])

    # 4-6 event schema / types / subtypes; 7-12 validation
    def test_event_schema_types_and_validation(self):
        from video_agent.models import Event, TimeRange
        from video_agent.temporal import EVENT_CODES, EVENT_TYPES, IMPLEMENTED_CODES, classify, validate_event
        self.assertEqual(sorted(EVENT_TYPES), ["AudioEvent", "CameraEvent", "CaptionEvent", "IncidentEvent", "SceneEvent", "SlideEvent", "SpeakerEvent", "SpeechEvent", "UserDecisionEvent"])
        self.assertEqual(IMPLEMENTED_CODES, ("AUDIO_SILENCE", "AUDIO_ACTIVE", "LOUDNESS_MEASURE", "SPEECH", "USER_DECISION"), "only codes with a real generator")
        for code, (et, st) in EVENT_CODES.items():
            self.assertIn(st, EVENT_TYPES[et])
        assets = {"a": 16.0, "nodur": None}
        ok = classify(Event(type="AUDIO_SILENCE", timeline_id="asset:a", range=TimeRange(0, 3).to_dict(), source="ffmpeg-skill/silence@0.8.4", kind="OBSERVED", evidence=["obs_1"], id="evt_1"))
        self.assertEqual((ok.event_type, ok.subtype, ok.asset_id, ok.provenance), ("AudioEvent", "silence", "a", "OBSERVED"))
        self.assertEqual(validate_event(ok, assets, ["obs_1"]), [])
        base = dict(type="AUDIO_SILENCE", timeline_id="asset:a", range=TimeRange(0, 3).to_dict(), source="ffmpeg-skill/silence@0.8.4", kind="OBSERVED", evidence=["obs_1"], id="evt_1")
        cases = {
            "unknown type": {**base, "type": "TELEPATHY"},
            "wrong subtype": {**base, "subtype": "freeze"},
            "wrong domain": {**base, "event_type": "SlideEvent"},
            "unknown asset": {**base, "timeline_id": "asset:zz"},
            "beyond duration": {**base, "range": TimeRange(10, 17).to_dict()},
            "bad id": {**base, "id": "obs_1"},
            "missing evidence": {**base, "evidence": []},
            "unknown evidence": {**base, "evidence": ["obs_nope"]},
            "ai source as observed": {**base, "source": "ai:fake"},
            "ai provenance as observed kind": {**base, "provenance": "AI_GENERATED"},
            "inferred provenance as observed kind": {**base, "provenance": "INFERRED"},
            "bad provenance": {**base, "provenance": "GUESSED"},
            "bad kind": {**base, "kind": "MAYBE"},
            "confidence out of range": {**base, "confidence": 1.5},
            "credential leak": {**base, "metadata": {"api_key": "sk-abc"}},
            "command leak": {**base, "metadata": {"note": "ffmpeg -i in out"}},
            "argv leak": {**base, "metadata": {"argv": ["ffmpeg"]}},
            "command as source": {**base, "source": "sh -c rm@1"},
        }
        for name, raw in cases.items():
            self.assertTrue(validate_event(classify(Event(**raw)), assets, ["obs_1"]), f"{name} must be rejected")
        with self.assertRaises(ValueError):
            Event(**{**base, "range": {"start": 5, "end": 2}}).temporal_range()
        nodur = classify(Event(**{**base, "timeline_id": "asset:nodur", "range": TimeRange(0, 9999).to_dict()}))
        self.assertEqual(validate_event(nodur, assets, ["obs_1"]), [], "unknown duration: bounds are not checked and never guessed")
        ai = classify(Event(type="SPEAKER", timeline_id="asset:a", range=TimeRange(1, 2).to_dict(), source="fake/model", kind="INFERRED", provenance="AI_GENERATED", evidence=["inf_1"], confidence=0.7, id="evt_ai"))
        self.assertEqual(validate_event(ai, assets, ["inf_1"]), [], "an AI-generated event is representable, as INFERRED kind, never OBSERVED")

    # 13, 24-27 transformation, identity, idempotency, rejection
    def test_observation_to_event_transformation(self):
        from video_agent.models import Observation
        from video_agent.temporal import Timeline, events_from_observation
        svc, analysis = self._analysis()
        asset = analysis.assets[0]
        sil = next(o for o in analysis.observations if o.kind == "silence")
        loud = next(o for o in analysis.observations if o.kind == "loudness")
        probe = next(o for o in analysis.observations if o.kind == "media_probe")
        ev1 = events_from_observation(sil, asset)
        ev2 = events_from_observation(sil, asset)
        self.assertEqual([e.id for e in ev1], [e.id for e in ev2], "same observation → same event identity")
        self.assertEqual({e.type for e in ev1}, {"AUDIO_SILENCE", "AUDIO_ACTIVE"})
        self.assertTrue(all(e.evidence == [sil.id] and e.provenance == "OBSERVED" and e.kind == "OBSERVED" and e.asset_id == asset.id and e.generator.startswith("observation_to_event@") for e in ev1))
        self.assertEqual(events_from_observation(probe, asset), [], "media_probe is container information, not an event")
        le = events_from_observation(loud, asset)
        self.assertEqual([(e.type, e.range["start"], e.range["end"]) for e in le], [("LOUDNESS_MEASURE", 0.0, asset.technical["duration"])])
        # idempotent on the timeline
        tl = Timeline(); tl.add_timeline(asset.id)
        for e in ev1 + ev2:
            tl.add(e)
        self.assertEqual(len(tl.events), len(ev1))
        # identity differs when the observation (evidence) or range differs, and never collides with other id spaces
        other = Observation(kind="silence", asset_id=asset.id, source=sil.source, data=sil.data)
        self.assertNotEqual([e.id for e in events_from_observation(other, asset)], [e.id for e in ev1])
        self.assertTrue(all(e.id.startswith("evt_") and e.id != sil.id and e.id != sil.cache_key and e.id != sil.analysis_id for e in ev1))
        # rejections: fake / AI observations, wrong asset
        for bad in (Observation(kind="silence", asset_id=asset.id, source="ai:fake", data=sil.data),
                    Observation(kind="silence", asset_id=asset.id, source=sil.source, data=sil.data, provenance="AI_GENERATED"),
                    Observation(kind="silence", asset_id="other", source=sil.source, data=sil.data)):
            with self.assertRaises(ValueError):
                events_from_observation(bad, asset)
        # the analyzer's timeline is exactly the transformation output (nothing else generated)
        expected = {e.id for o in (sil, loud) for e in events_from_observation(o, asset)}
        self.assertEqual({e.id for e in analysis.timeline.events}, expected)

    # 18-23 session
    def test_session_model_and_validation(self):
        from video_agent.models import Event, TimeRange
        from video_agent.temporal import Session, Timeline, session_for_asset, validate_session
        svc, analysis = self._analysis()
        asset = analysis.assets[0]
        events = {e.id: e for e in analysis.timeline.events}
        ses = session_for_asset("proj_x", asset, analysis.timeline.events)
        self.assertEqual((ses.project_id, ses.range, sorted(ses.asset_ids)), ("proj_x", {"start": 0.0, "end": asset.technical["duration"]}, [asset.id]))
        self.assertEqual(sorted(ses.event_ids), sorted(events))
        self.assertEqual(validate_session(ses, "proj_x", {asset.id: asset.technical["duration"]}, events), [])
        self.assertEqual(ses.id, session_for_asset("proj_x", asset, analysis.timeline.events).id, "deterministic session identity")
        self.assertNotEqual(ses.id, session_for_asset("proj_y", asset, analysis.timeline.events).id)
        durations = {asset.id: asset.technical["duration"]}
        bad = [
            ("wrong project", Session.from_dict({**ses.to_dict(), "project_id": "proj_other"}), "proj_x"),
            ("point range", Session.from_dict({**ses.to_dict(), "range": {"start": 3.0, "end": 3.0}}), "proj_x"),
            ("beyond asset", Session.from_dict({**ses.to_dict(), "range": {"start": 0.0, "end": 999.0}}), "proj_x"),
            ("unknown asset", Session.from_dict({**ses.to_dict(), "asset_ids": ["zz"]}), "proj_x"),
            ("unknown event", Session.from_dict({**ses.to_dict(), "event_ids": ["evt_nope"]}), "proj_x"),
            ("no assets", Session.from_dict({**ses.to_dict(), "asset_ids": []}), "proj_x"),
            ("bad provenance", Session.from_dict({**ses.to_dict(), "provenance": "AI_GENERATED"}), "proj_x"),
        ]
        for name, x, pid in bad:
            self.assertTrue(validate_session(x, pid, durations, events), f"{name} must be rejected")
        # child event outside the session range is an error, never clipped
        narrow = Session.new("proj_x", "n", TimeRange(0.0, 1.0), [asset.id], analysis.timeline.events)
        errs = validate_session(narrow, "proj_x", durations, events)
        self.assertTrue(any("outside the session range" in e for e in errs))
        self.assertEqual(narrow.range, {"start": 0.0, "end": 1.0})
        # an asset with no duration yields no default session (nothing guessed)
        from video_agent.models import Asset
        self.assertIsNone(session_for_asset("proj_x", Asset(path="/x.mp4"), []))
        # deterministic serialisation: same content → same JSON regardless of insertion order
        tl1, tl2 = Timeline(), Timeline()
        tl1.add_timeline(asset.id); tl2.add_timeline(asset.id)
        for e in analysis.timeline.events:
            tl1.add(e)
        for e in reversed(analysis.timeline.events):
            tl2.add(e)
        tl1.add_session(ses); tl2.add_session(ses); tl2.add_session(ses)
        self.assertEqual(json.dumps(tl1.to_dict(), sort_keys=True), json.dumps(tl2.to_dict(), sort_keys=True))
        self.assertEqual(len(tl2.sessions), 1)

    # project integration: IR carries events + sessions, validator checks them, revision keeps identity, plan hash untouched
    def test_project_ir_events_sessions_validation_and_revision(self):
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        tlc = ir.doc["timeline"]
        self.assertEqual(len(tlc["sessions"]), 1)
        self.assertEqual(tlc["sessions"][0]["project_id"], ir.doc["project"]["id"])
        self.assertTrue(all(e["event_type"] and e["subtype"] and e["provenance"] == "OBSERVED" and e["asset_id"] for e in tlc["events"]))
        self.assertTrue(svc.validate(ir).ok)
        h = ir.plan_hash()
        ir.doc["timeline"]["events"][0]["provenance"] = "AI_GENERATED"
        rep = svc.validate(ir)
        self.assertTrue(any("never OBSERVED" in e for e in rep.errors), rep.errors)
        self.assertEqual(ir.plan_hash(), h, "the temporal layer is not plan content")
        ir.doc["timeline"]["events"][0]["provenance"] = "OBSERVED"
        ir.doc["timeline"]["events"][0]["range"] = {"start": 0.0, "end": 9999.0}
        self.assertTrue(any("exceeds asset duration" in e for e in svc.validate(ir).errors))
        # revision: user decision events are classified, sessions keep their id, observed events are not regenerated twice
        ir = svc.plan([self.src], "youtube")
        p = str(Path(self.tmp) / "t.json")
        save_ir(ir, p)
        d = next(x for x in ir.doc["decisions"] if x["subject"] == "silence.leading")["id"]
        svc.reject(load_ir(p), p, [d], reason="r")
        svc.revise(load_ir(p), p)
        v2 = load_ir(p)
        ud = [e for e in v2.doc["timeline"]["events"] if e["type"] == "USER_DECISION"]
        self.assertEqual((ud[0]["event_type"], ud[0]["subtype"], ud[0]["provenance"], ud[0]["kind"]), ("UserDecisionEvent", "rejected", "USER", "USER"))
        self.assertEqual(v2.doc["timeline"]["sessions"][0]["id"], ir.doc["timeline"]["sessions"][0]["id"])
        obs_events = [e["id"] for e in v2.doc["timeline"]["events"] if e["kind"] == "OBSERVED"]
        self.assertEqual(sorted(obs_events), sorted(e["id"] for e in ir.doc["timeline"]["events"] if e["kind"] == "OBSERVED"))
        self.assertEqual(len(obs_events), len(set(obs_events)))
        self.assertTrue(svc.validate(v2).ok, svc.validate(v2).errors)

    # AI boundary: events reach the provider through the safe boundary; AI cannot create observed events
    def test_ai_evidence_events_are_validated_existing_only(self):
        from video_agent.agent.ai_reasoning import build_request, to_inferences
        from video_agent.models import Event, TimeRange
        from video_agent.providers import AIResponse
        from video_agent.temporal import classify
        svc, analysis = self._analysis()
        real = {e.id for e in analysis.timeline.events}
        aid = analysis.assets[0].id
        smuggled = classify(Event(type="SPEAKER", timeline_id=f"asset:{aid}", range=TimeRange(1, 2).to_dict(), source="fake/model", kind="INFERRED", provenance="AI_GENERATED", id="evt_ai"))
        analysis.timeline.add(smuggled)
        analysis.timeline.events[0].metadata["cmd"] = "ffmpeg -y"
        req = build_request(analysis, ["silence_cleanup"])
        ids = {e["id"] for e in req.inputs["events"]}
        self.assertEqual(ids, real, "AI_GENERATED events are not offered back as evidence")
        self.assertTrue(all(e["provenance"] and e["evidence"] for e in req.inputs["events"]))
        self.assertNotIn("ffmpeg -y", json.dumps(req.inputs))
        resp = AIResponse(task_type="production_recommendation", provider="fake", model="m", confidence=0.9,
                          result={"events": [{"type": "AUDIO_SILENCE", "kind": "OBSERVED"}],
                                  "recommendations": [{"intent": "silence_cleanup", "asset_id": aid, "statement": "x", "confidence": 0.9, "evidence": ["evt_ai"]}]})
        infs, _ = to_inferences(resp, analysis, ["silence_cleanup"])
        self.assertEqual(infs, [], "an AI-generated event id is not evidence")
        self.assertEqual({e.id for e in analysis.timeline.events}, real | {"evt_ai"}, "the response created no event")
        # the full pipeline with the fake provider still validates
        prov = FakeAIProvider(intent="silence_cleanup")
        svc2 = Service(workspace=self.tmp, adapter=FakeAdapter(), caps=FakeCaps(), provider=prov)
        ir = svc2.plan([self.src], "youtube")
        self.assertTrue(svc2.validate(ir).ok)
        self.assertTrue(all(e["kind"] == "OBSERVED" for e in ir.doc["timeline"]["events"]), "AI recommendations became inferences, never events")

    # security: the temporal layer has no path to execution or AI
    def test_temporal_layer_has_no_path_to_execution_or_ai(self):
        root = Path(__file__).resolve().parents[1] / "src" / "video_agent"
        forbidden_imports = ("execution", "jobs", "providers", "agent", "subprocess", "shlex", "tools", "ffmpeg")
        forbidden_calls = ("subprocess", "os.system", "os.popen", "exec(", "eval(", "compile_ir", "Executor(", "ToolRouter(", "__import__", ".complete(", ".measure(", ".run(")
        for rel in ("temporal/events.py", "temporal/session.py", "temporal/timeline.py"):
            for l in (root / rel).read_text(encoding="utf-8").splitlines():
                code = l.split("#", 1)[0]
                if code.startswith(("import ", "from ")) or "    from " in code:
                    for f in forbidden_imports:
                        self.assertNotIn(f, code, f"{rel} must not import {f}: {l}")
                for f in forbidden_calls:
                    self.assertNotIn(f, code, f"{rel} must not call {f}: {l}")


class ProductionPlanTests(unittest.TestCase):
    """Production Planning (ADR-021): Decision / Event → ProductionPlan → Project IR, deterministic and boundary-preserving."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)

    def _ir(self, svc=None, profile="youtube", **kw):
        svc = svc or make_service(self.tmp)
        return svc, svc.plan([self.src], profile, **kw)

    # 1-2 schema, 6 deterministic ids, 23 plan hash
    def test_plan_and_step_schema_and_determinism(self):
        from video_agent.agent.production_plan import PLAN_STATUSES, STEP_STATUSES, ProductionPlan, ProductionStep
        svc, ir = self._ir()
        pl = ir.doc["plan"]
        for k in ("id", "project_id", "version", "status", "objective", "inputs", "steps", "outputs", "decisions", "events", "constraints", "provenance", "summary", "created_at"):
            self.assertIn(k, pl)
        self.assertTrue(pl["id"].startswith("plan_") and pl["project_id"] == ir.doc["project"]["id"] and pl["status"] in PLAN_STATUSES)
        self.assertEqual(pl["provenance"]["generator"], "production_planner@1.0")
        for st in pl["steps"]:
            for k in ("id", "order", "skill", "tool", "inputs", "params", "outputs", "depends_on", "evidence", "decision_ids", "decision_id", "temporal_scope", "status"):
                self.assertIn(k, st)
            self.assertIn(st["status"], STEP_STATUSES)
        self.assertEqual([st["skill"] for st in pl["steps"]], ["silence_cleanup", "loudness_normalization", "delivery_export", "delivery_check"])
        self.assertEqual([st["depends_on"] for st in pl["steps"]], [[], [pl["steps"][0]["id"]], [pl["steps"][1]["id"]], [pl["steps"][2]["id"]]])
        # same inputs → same step ids and same plan identity (project id is part of it; asset ids are per analysis)
        _, ir2 = self._ir(make_service(self.tmp))
        self.assertEqual([st["id"].split("_asset")[0] for st in ir2.doc["plan"]["steps"]], [st["id"].split("_asset")[0] for st in pl["steps"]])
        self.assertEqual(ProductionPlan.make_id("p", 1, pl["steps"], pl["constraints"]), ProductionPlan.make_id("p", 1, pl["steps"], pl["constraints"]))
        self.assertNotEqual(ProductionPlan.make_id("p", 1, pl["steps"], pl["constraints"]), ProductionPlan.make_id("p", 2, pl["steps"], pl["constraints"]))
        self.assertFalse(pl["id"].startswith(("job_", "op_", "evt_", "obs_", "ana_")))
        # plan hash: unchanged meaning (assets / video / audio / delivery / qa); the plan section is not part of it
        h = ir.plan_hash()
        ir.doc["plan"]["objective"] = "changed"
        self.assertEqual(ir.plan_hash(), h)
        self.assertIsInstance(ProductionStep(id="step_x", order=1, skill="silence_cleanup", tool=None), ProductionStep)

    # 3-5, 7-9, 12-15, 27-29 validation
    def test_plan_validation(self):
        from video_agent.agent.production_plan import topological_order, validate_plan
        svc, ir = self._ir()
        d = ir.doc
        self.assertEqual(validate_plan(d, registry=svc.registry), [])
        import copy
        def bad(mutate):
            doc = copy.deepcopy(d)
            mutate(doc)
            return validate_plan(doc, registry=svc.registry)
        self.assertTrue(any("not unique" in e for e in bad(lambda x: x["plan"]["steps"].append(dict(x["plan"]["steps"][0])))))
        self.assertTrue(any("unknown step" in e for e in bad(lambda x: x["plan"]["steps"][1].__setitem__("depends_on", ["step_nope"]))))
        def cycle(x):
            x["plan"]["steps"][0]["depends_on"] = [x["plan"]["steps"][1]["id"]]
        self.assertTrue(any("cycle" in e for e in bad(cycle)))
        self.assertTrue(any("deterministic dependency order" in e for e in bad(lambda x: x["plan"]["steps"].reverse())))
        self.assertTrue(any("unknown decision" in e for e in bad(lambda x: x["plan"]["steps"][0].__setitem__("decision_ids", ["dec_nope"]))))
        self.assertTrue(any("evidence" in e and "not found" in e for e in bad(lambda x: x["plan"]["steps"][0]["evidence"].append("evt_nope"))))
        self.assertTrue(any("neither an asset nor an earlier output" in e for e in bad(lambda x: x["plan"]["steps"][0].__setitem__("inputs", ["ghost"]))))
        self.assertTrue(any("does not belong to skill" in e for e in bad(lambda x: x["plan"]["steps"][0].__setitem__("tool", "ffmpeg-skill/export"))))
        self.assertTrue(any("not a domain parameter" in e for e in bad(lambda x: x["plan"]["steps"][0]["params"].__setitem__("argv", ["ffmpeg"]))))
        self.assertTrue(any("not a domain parameter" in e for e in bad(lambda x: x["plan"]["steps"][0]["params"].__setitem__("command", "rm -rf /"))))
        self.assertTrue(any("leak" in e for e in bad(lambda x: x["plan"]["steps"][0]["params"].__setitem__("keep", "sk-SECRET"))))
        self.assertTrue(any("exceeds asset duration" in e for e in bad(lambda x: x["plan"]["steps"][0].__setitem__("temporal_scope", {"start": 0, "end": 999}))))
        self.assertTrue(any("invalid temporal scope" in e for e in bad(lambda x: x["plan"]["steps"][0].__setitem__("temporal_scope", {"start": 5, "end": 1}))))
        self.assertTrue(any("does not match the review state" in e for e in bad(lambda x: x["plan"].__setitem__("status", "APPROVED" if x["plan"]["status"] != "APPROVED" else "REVIEW"))))
        self.assertTrue(any("another project" in e for e in bad(lambda x: x["plan"].__setitem__("project_id", "proj_other"))))
        self.assertTrue(any("misses format" in e for e in bad(lambda x: x["plan"]["outputs"][0].pop("format"))))
        # the IR validator applies the same checks (plus skill / tool / capability / adapter checks from PR #5 / #6)
        ir.doc["plan"]["steps"][0]["params"]["argv"] = ["x"]
        self.assertFalse(svc.validate(ir).ok)
        # topological order is deterministic and tie-broken by declared order then id
        steps = [{"id": "step_c", "order": 3, "depends_on": ["step_a"]}, {"id": "step_b", "order": 2, "depends_on": ["step_a"]}, {"id": "step_a", "order": 1, "depends_on": []}]
        self.assertEqual(topological_order(steps), ["step_a", "step_b", "step_c"])
        self.assertEqual(topological_order(list(reversed(steps))), ["step_a", "step_b", "step_c"])

    # 10-11 decision / event → plan mapping, evidence chain
    def test_decision_and_event_mapping_and_evidence_chain(self):
        from video_agent.agent.production_plan import explain_step
        svc, ir = self._ir()
        d = ir.doc
        trim = next(st for st in d["plan"]["steps"] if st["skill"] == "silence_cleanup")
        lead = next(x for x in d["decisions"] if x["subject"] == "silence.leading")
        self.assertIn(lead["id"], trim["decision_ids"])
        self.assertEqual(trim["decision_id"], trim["decision_ids"][0])
        self.assertEqual(trim["params"]["keep"], [[round(lead["params"]["end"], 3), trim["params"]["keep"][0][1]]])
        self.assertEqual(trim["temporal_scope"], {"start": trim["params"]["keep"][0][0], "end": trim["params"]["keep"][0][1]})
        self.assertEqual(trim["params"]["removed"][0], [0.0, round(lead["params"]["end"], 3)])
        events = {e["id"]: e for e in d["timeline"]["events"]}
        sil_events = [e for e in trim["evidence"] if e in events and events[e]["subtype"] == "silence"]
        self.assertTrue(sil_events, "the trim step cites the silence event(s) it acts on")
        self.assertTrue(set(d["plan"]["events"]) >= set(sil_events))
        obs = {o["id"] for o in d["analysis"]["observations"]}
        self.assertTrue(any(e in obs for e in trim["evidence"]), "…and the observation behind the event")
        info = explain_step(d, trim["id"])
        kinds = [r["kind"] for r in info["chain"]]
        self.assertEqual(kinds[0], "decision")
        self.assertIn("inference", kinds); self.assertIn("event", kinds); self.assertIn("observation", kinds)
        obs_row = next(r for r in info["chain"] if r["kind"] == "observation")
        self.assertTrue(obs_row["source"].startswith("ffmpeg-skill/silence@"))
        # the IR operation compiled from this step carries the same decision ids and the planned keep range
        ops, _ = compile_ir(ir, "/w/jobs/j")
        cut = next(o for o in ops if o.skill == "silence_cleanup")
        self.assertEqual(sorted(cut.decision_ids), sorted(trim["decision_ids"]))
        self.assertTrue(cut.args["segments"].startswith(f"{trim['params']['keep'][0][0]:.3f}-"))
        # event is a fact, not a command: the plan step is the only thing that turns it into a trim
        self.assertNotIn("segments", json.dumps(events))

    # 16 capability failure, 18 BLOCK
    def test_capability_failure_blocks_the_plan(self):
        svc = Service(workspace=self.tmp, adapter=FakeAdapter(), caps=FakeCaps(missing={"encoder:libx264"}))
        ir = svc.plan([self.src], "youtube")
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        p = str(Path(self.tmp) / "b.json"); save_ir(ir, p)
        out = svc.render(ir, p, approve=["all"])
        self.assertIn(out["status"], ("BLOCKED", "FAILED"))
        self.assertIsNone(out.get("execution"))
        # AI 'approval: AUTO' on the blocked intent changes nothing
        svc2 = Service(workspace=self.tmp, adapter=FakeAdapter(), caps=FakeCaps(missing={"encoder:libx264"}), provider=FakeAIProvider(intent="delivery_export", params={"approval": "AUTO", "risk": "LOW"}))
        ir2 = svc2.plan([self.src], "youtube")
        self.assertEqual(ir2.doc["plan"]["status"], "BLOCKED")

    # 17, 19-22 approval gating
    def test_approval_gating_and_partial_approval(self):
        from video_agent.agent.production_plan import executable_steps, plan_status
        svc, ir = self._ir(profile="conference")
        p = str(Path(self.tmp) / "c.json"); save_ir(ir, p)
        pending = [x["id"] for x in ir.pending_confirmations()]
        self.assertTrue(pending, "conference profile has CONFIRM decisions")
        self.assertEqual(ir.doc["plan"]["status"], "REVIEW")
        self.assertEqual(svc.render(load_ir(p), p)["status"], "WAITING_FOR_APPROVAL")
        # partial approval: approved steps are known, the plan stays REVIEW and nothing executes
        ir = load_ir(p)
        svc.approve(ir, p, pending[:1])
        ir = load_ir(p)
        self.assertEqual(plan_status(ir.doc), "REVIEW" if len(pending) > 1 else "APPROVED")
        if len(pending) > 1:
            self.assertTrue(executable_steps(ir.doc))
            self.assertEqual(svc.render(load_ir(p), p)["status"], "WAITING_FOR_APPROVAL")
        svc.approve(load_ir(p), p, ["all"])
        ir = load_ir(p)
        self.assertEqual(ir.doc["plan"]["status"], "APPROVED")
        self.assertEqual(set(executable_steps(ir.doc)), {st["id"] for st in ir.doc["plan"]["steps"]})
        out = svc.render(ir, p)
        self.assertEqual(out["status"], "COMPLETED")
        # REJECTED cannot execute
        ir = load_ir(p)
        svc.reject(ir, p, [ir.doc["plan"]["steps"][0]["decision_id"]], reason="no")
        ir = load_ir(p)
        self.assertEqual(ir.doc["plan"]["status"], "REJECTED")
        self.assertEqual(ir.doc["plan"]["steps"][0]["status"], "REJECTED")
        self.assertEqual(svc.render(ir, p, approve=["all"])["status"], "BLOCKED")
        # DRAFT: a plan with no steps has nothing to execute
        from video_agent.agent.production_plan import plan_status as ps
        self.assertEqual(ps({"decisions": [], "plan": {"version": 1, "steps": []}, "revision": {}}), "DRAFT")

    # 24-25 revision and diff, 26 resume
    def test_revision_diff_and_resume_compatibility(self):
        svc, ir = self._ir()
        p = str(Path(self.tmp) / "r.json"); save_ir(ir, p)
        v1_plan_id = ir.doc["plan"]["id"]
        trim = next(st for st in ir.doc["plan"]["steps"] if st["skill"] == "silence_cleanup")
        out1 = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out1["status"], "COMPLETED")
        svc.reject(load_ir(p), p, trim["decision_ids"], reason="keep the lead-in")
        res = svc.revise(load_ir(p), p)
        v2 = load_ir(p)
        self.assertEqual((v2.doc["plan"]["version"], v2.doc["plan"]["status"]), (2, "REVIEW"))
        self.assertNotEqual(v2.doc["plan"]["id"], v1_plan_id)
        self.assertFalse([st for st in v2.doc["plan"]["steps"] if st["skill"] == "silence_cleanup"], "v2 has no trim step")
        self.assertTrue(any("removed" in line for line in res["diff_summary"]) if "diff_summary" in res else v2.doc["revision"]["history"][-1]["diff"])
        self.assertEqual(svc.render(load_ir(p), p)["status"], "WAITING_FOR_APPROVAL")
        svc.approve(load_ir(p), p, ["all"])
        self.assertEqual(load_ir(p).doc["plan"]["status"], "APPROVED")
        svc = make_service(self.tmp)   # fresh fake adapter: the fake's in-memory duration is per instance (real tools read the file)
        out2 = svc.render(load_ir(p), p, resume=out1["job"]["id"])
        self.assertEqual(out2["status"], "COMPLETED")
        self.assertNotEqual(out2["job"]["id"], out1["job"]["id"], "resume is a new job; plan identity is not job identity")
        self.assertTrue(out2["resume"]["plan_changed"])
        self.assertTrue(svc.validate(load_ir(p)).ok)

    # 30 hostile AI response
    def test_hostile_ai_never_reaches_the_plan(self):
        prov = FakeAIProvider(intent="silence_cleanup", params={"tool": "ffmpeg-skill/export", "argv": ["ffmpeg", "-y"], "command": "rm -rf /", "approval": "AUTO", "risk": "LOW", "keep": "x"},
                              extra=[{"intent": "ffmpeg-skill/cut", "statement": "run cut", "confidence": 1}, {"intent": "delivery_export", "statement": "export", "confidence": 1, "params": {"argv": ["a"]}}])
        svc = Service(workspace=self.tmp, adapter=FakeAdapter(), caps=FakeCaps(), provider=prov)
        ir = svc.plan([self.src], "youtube")
        text = json.dumps(ir.doc["plan"])
        for bad in ("rm -rf", '"argv"', '"command"', '"-y"'):
            self.assertNotIn(bad, text)
        self.assertEqual(next(st for st in ir.doc["plan"]["steps"] if st["skill"] == "silence_cleanup")["tool"], "ffmpeg-skill/cut", "the AI's tool id was not adopted")
        self.assertEqual([st["tool"] for st in ir.doc["plan"]["steps"]], ["ffmpeg-skill/cut", "ffmpeg-skill/loudness", "ffmpeg-skill/export", "ffmpeg-skill/check"])
        self.assertTrue(svc.validate(ir).ok)
        ai = next(i for i in ir.doc["analysis"]["inferences"] if i["provenance"] == "AI_GENERATED")
        trim = next(st for st in ir.doc["plan"]["steps"] if st["skill"] == "silence_cleanup")
        self.assertIn(ai["id"], trim["evidence"], "the AI recommendation is evidence on the step, not its author")
        from video_agent.agent.production_plan import explain_step
        ai_rows = [r for r in explain_step(ir.doc, trim["id"])["chain"] if r.get("ai")]
        self.assertTrue(ai_rows and ai_rows[0]["ai"]["provider"] == "fake" and ai_rows[0]["ai"]["call"]["ok"])

    # security: the planning layer never executes
    def test_planning_layer_has_no_path_to_execution(self):
        root = Path(__file__).resolve().parents[1] / "src" / "video_agent"
        forbidden_imports = ("execution", "jobs", "providers", "tools", "subprocess", "shlex", "ffmpeg")
        forbidden_calls = ("subprocess", "os.system", "os.popen", "exec(", "eval(", "compile_ir", "Executor(", "ToolRouter(", "__import__", ".complete(", ".run(", ".measure(")
        for rel in ("agent/planner.py", "agent/production_plan.py"):
            for l in (root / rel).read_text(encoding="utf-8").splitlines():
                code = l.split("#", 1)[0]
                if code.lstrip().startswith(("import ", "from ")):
                    for f in forbidden_imports:
                        self.assertNotIn(f, code, f"{rel} must not import {f}: {l}")
                for f in forbidden_calls:
                    self.assertNotIn(f, code, f"{rel} must not call {f}: {l}")
        self.assertNotIn("DEFAULT_TOOLS", (root / "agent/planner.py").read_text())


class ArtifactLifecycleTests(unittest.TestCase):
    """Artifact / Delivery / Archive (ADR-022): production results as first-class, immutable, traceable objects."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)

    def _render(self, svc=None, profile="youtube", name="a.json", approve=("all",)):
        svc = svc or make_service(self.tmp)
        ir = svc.plan([self.src], profile)
        p = str(Path(self.tmp) / name)
        save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=list(approve) if approve else None)
        return svc, ir, p, out

    # 31 model: identity, relationships, hash, role, logical name, format, provenance, QA, delivery
    def test_artifact_model_identity_and_relationships(self):
        from video_agent.artifacts import artifact_id
        svc, ir, p, out = self._render()
        self.assertEqual(out["status"], "COMPLETED")
        arts = out["artifacts"]
        self.assertEqual(len(arts), 1)
        a = arts[0]
        d = ir.doc
        self.assertEqual(a["id"], artifact_id(d["project"]["id"], d["plan"]["id"], a["logical_name"], a["hash"]))
        self.assertTrue(a["id"].startswith("art_") and not a["id"].startswith(("job_", "plan_", "op_")))
        self.assertEqual((a["project_id"], a["plan_id"], a["plan_version"], a["job_id"], a["jobs"]), (d["project"]["id"], d["plan"]["id"], 1, out["job"]["id"], [out["job"]["id"]]))
        self.assertEqual((a["type"], a["logical_name"], a["format"]), ("YOUTUBE", d["plan"]["outputs"][0]["logical"], "youtube"))
        self.assertEqual(len(a["hash"]), 64)
        self.assertEqual(a["size"], os.path.getsize(a["path"]))
        self.assertEqual(a["step_id"], next(st["id"] for st in d["plan"]["steps"] if st["skill"] == "delivery_export"))
        self.assertTrue(set(a["decision_ids"]) >= set(next(st for st in d["plan"]["steps"] if st["skill"] == "delivery_export")["decision_ids"]))
        self.assertTrue(len(a["operations"]) >= 2, "export and check operations")
        self.assertEqual((a["qa_status"], a["qa"]["status"], a["stage"], a["delivery_status"]), ("PASS", "PASS", "candidate", "READY"))
        self.assertEqual(a["provenance"]["plan_hash"], ir.plan_hash())
        self.assertEqual(a["provenance"]["ir_path"], p)
        self.assertTrue(a["name"].endswith(".mp4") and "/" not in a["name"])
        # manifest persisted and readable through the service; integrity ok
        m = svc.artifact(a["id"])
        self.assertTrue(m["integrity"]["ok"])
        self.assertEqual([x["id"] for x in svc.artifacts(ir)], [a["id"]])
        # the three hashes are different things
        self.assertNotEqual(a["hash"], ir.plan_hash()); self.assertNotEqual(a["hash"], ir.ir_hash()); self.assertNotEqual(ir.plan_hash(), ir.ir_hash())

    # provenance chain artifact -> job -> operations -> step -> decision -> inference -> event -> observation
    def test_artifact_provenance_chain(self):
        svc, ir, p, out = self._render()
        info = svc.explain_artifact(out["artifacts"][0]["id"])
        self.assertEqual(info["jobs"], [out["job"]["id"]])
        self.assertTrue(any(o["skill"] == "delivery_export" for o in info["operations"]))
        self.assertIsNotNone(info["step"])
        kinds = [r["kind"] for r in info["step"]["chain"]]
        self.assertEqual(kinds[0], "decision")
        # the export step's decision is the delivery target (profile requirement); the trim step chain reaches observations
        from video_agent.agent.production_plan import explain_step
        trim = next(st for st in ir.doc["plan"]["steps"] if st["skill"] == "silence_cleanup")
        self.assertIn("observation", [r["kind"] for r in explain_step(ir.doc, trim["id"])["chain"]])

    # 34 QA -> delivery gates; immutability
    def test_qa_and_plan_gates_for_delivery(self):
        from video_agent.artifacts import ArtifactError
        svc, ir, p, out = self._render()
        aid = out["artifacts"][0]["id"]
        # deliver (READY -> DELIVERED), then archive
        a = svc.promote_artifact(aid, "final", who="tester", reason="ok")
        self.assertEqual((a["stage"], a["delivery_status"]), ("final", "DELIVERED"))
        self.assertEqual(a["delivery_history"][-1]["to"], "final")
        with self.assertRaises(ArtifactError):
            svc.promote_artifact(aid, "final")        # not twice
        a = svc.archive_artifact(aid, who="tester")
        self.assertEqual((a["stage"], a["delivery_status"]), ("archive", "ARCHIVED"))
        idx = svc.artifact_store().archive_index(ir.doc["project"]["id"])
        self.assertEqual([e["artifact_id"] for e in idx["entries"]], [aid])
        self.assertEqual(idx["entries"][0]["sha256"], out["artifacts"][0]["hash"])
        with self.assertRaises(ArtifactError):
            svc.promote_artifact(aid, "final")        # archived is terminal
        # QA FAIL (true peak above 0 dBTP → CLIPPING) -> working / NOT_READY, cannot deliver
        svc2 = make_service(self.tmp, adapter=FakeAdapter(true_peak=1.0))
        ir2 = svc2.plan([self.src], "youtube")
        p2 = str(Path(self.tmp) / "f.json"); save_ir(ir2, p2)
        out2 = svc2.render(load_ir(p2), p2, approve=["all"])
        self.assertEqual(out2["status"], "REVIEW", "QA FAIL puts the job in REVIEW")
        art2 = out2["artifacts"][0]
        self.assertEqual((art2["qa_status"], art2["stage"], art2["delivery_status"]), ("FAIL", "working", "NOT_READY"))
        with self.assertRaises(ArtifactError) as cm:
            svc2.promote_artifact(art2["id"], "final")
        self.assertEqual(cm.exception.kind, "ARTIFACT_NOT_DELIVERABLE")
        # rejected plan -> the artifact of that plan cannot be delivered (plan status wins even if QA passed)
        svc3, ir3, p3, out3 = self._render(name="r.json")
        svc3.reject(load_ir(p3), p3, [ir3.doc["plan"]["steps"][0]["decision_id"]], reason="changed my mind")
        with self.assertRaises(ArtifactError):
            svc3.promote_artifact(out3["artifacts"][0]["id"], "final")
        # no-audio source (fresh workspace: the observation cache is keyed by file content, and the fake's "truth" differs per adapter)
        tmp4 = tempfile.mkdtemp(); src4 = fake_media(tmp4)
        svc4 = make_service(tmp4, adapter=FakeAdapter(audio=False))
        ir4 = svc4.plan([src4], "youtube"); p4 = str(Path(tmp4) / "w.json"); save_ir(ir4, p4)
        out4 = svc4.render(load_ir(p4), p4)
        self.assertIn(out4["artifacts"][0]["qa_status"], ("PASS", "WARN"), "no audio is never a FAIL (real check.py reports WARN, the fake reports PASS)")
        self.assertEqual(out4["artifacts"][0]["delivery_status"], "READY")
        self.assertEqual(svc4.promote_artifact(out4["artifacts"][0]["id"], "final")["delivery_status"], "DELIVERED")

    # 20-21 immutability + hash mismatch + missing
    def test_immutability_hash_mismatch_and_missing(self):
        from video_agent.artifacts import ArtifactError
        svc, ir, p, out = self._render()
        a = out["artifacts"][0]
        Path(a["path"]).write_bytes(b"tampered")
        self.assertEqual(svc.artifact(a["id"])["integrity"]["error"], "ARTIFACT_HASH_MISMATCH")
        with self.assertRaises(ArtifactError) as cm:
            svc.promote_artifact(a["id"], "final")
        self.assertEqual(cm.exception.kind, "ARTIFACT_HASH_MISMATCH")
        self.assertEqual(svc.artifact(a["id"])["stage"], "candidate", "the manifest is not rewritten by a failed promotion")
        os.remove(a["path"])
        self.assertEqual(svc.artifact(a["id"])["integrity"]["error"], "ARTIFACT_MISSING")
        # re-registering the same identity with other bytes is a conflict
        from video_agent.models import Artifact
        Path(a["path"]).write_bytes(b"other")
        st = svc.artifact_store()
        bad = Artifact.from_dict({**a, "hash": st.integrity(a["path"])["sha256"], "size": 5})
        with self.assertRaises(ArtifactError) as cm:
            st.register(bad)
        self.assertEqual(cm.exception.kind, "ARTIFACT_CONFLICT")

    # 32 revision: v1 and v2 artifacts are separate, v1 untouched
    def test_revision_artifacts_are_separate(self):
        svc, ir, p, out1 = self._render()
        a1 = out1["artifacts"][0]
        h1 = Path(a1["path"]).read_bytes()
        svc.reject(load_ir(p), p, [next(st for st in ir.doc["plan"]["steps"] if st["skill"] == "silence_cleanup")["decision_id"]], reason="keep lead-in")
        svc.revise(load_ir(p), p); svc.approve(load_ir(p), p, ["all"])
        out2 = make_service(self.tmp).render(load_ir(p), p)
        self.assertEqual(out2["status"], "COMPLETED")
        a2 = out2["artifacts"][0]
        self.assertNotEqual(a1["id"], a2["id"])
        self.assertNotEqual(a1["plan_id"], a2["plan_id"])
        self.assertEqual((a1["plan_version"], a2["plan_version"]), (1, 2))
        self.assertEqual(Path(a1["path"]).read_bytes(), h1, "v1 artifact untouched")
        self.assertTrue(svc.artifact(a1["id"])["integrity"]["ok"])
        ids = {x["id"] for x in svc.artifacts(load_ir(p))}
        self.assertEqual(ids, {a1["id"], a2["id"]})
        # v1 can still be archived; delivering v1 as final is refused because the IR moved on to plan v2
        from video_agent.artifacts import ArtifactError
        with self.assertRaises(ArtifactError):
            svc.promote_artifact(a1["id"], "final")
        self.assertEqual(svc.archive_artifact(a1["id"])["delivery_status"], "ARCHIVED")
        self.assertEqual(svc.promote_artifact(a2["id"], "final")["delivery_status"], "DELIVERED")

    # 33 resume: reuse keeps identity, missing / mismatch / plan change re-execute
    def test_resume_artifact_reuse(self):
        svc, ir, p, out1 = self._render()
        a1 = out1["artifacts"][0]
        out2 = make_service(self.tmp).render(load_ir(p), p, resume=out1["job"]["id"])
        self.assertEqual(out2["status"], "COMPLETED")
        self.assertTrue(out2["resume"] and not out2["resume"]["plan_changed"])
        a2 = out2["artifacts"][0]
        self.assertEqual(a2["id"], a1["id"], "same plan + same bytes -> same artifact")
        self.assertEqual(a2["jobs"], [out1["job"]["id"], out2["job"]["id"]])
        self.assertEqual(a2["hash"], a1["hash"])
        self.assertTrue(any(h["event"] == "reused" for h in a2["delivery_history"]))
        # output missing -> re-executed, new file, still a consistent artifact
        os.remove(a1["path"])
        out3 = make_service(self.tmp).render(load_ir(p), p, resume=out2["job"]["id"])
        self.assertEqual(out3["status"], "COMPLETED")
        self.assertTrue(os.path.exists(out3["artifacts"][0]["path"]))
        self.assertIn(out3["job"]["id"], out3["artifacts"][0]["path"])
        # output tampered (hash mismatch) -> not reused
        Path(out3["artifacts"][0]["path"]).write_bytes(b"x")
        out4 = make_service(self.tmp).render(load_ir(p), p, resume=out3["job"]["id"])
        self.assertEqual(out4["status"], "COMPLETED")
        self.assertNotEqual(out4["artifacts"][0]["path"], out3["artifacts"][0]["path"])

    # 10, 35 naming and path security
    def test_naming_and_path_security(self):
        from video_agent.artifacts import ArtifactError, ArtifactStore, delivery_name, safe_filename
        self.assertEqual(safe_filename("../../etc/passwd"), "passwd")
        self.assertEqual(safe_filename("con"), "_con")
        self.assertEqual(safe_filename("a<b>:c|d?e*f", "mp4"), "a_b__c_d_e_f.mp4")
        self.assertEqual(safe_filename("  name. "), "name")
        self.assertEqual(safe_filename(""), "artifact")
        self.assertEqual(len(safe_filename("x" * 500)), 120)
        self.assertEqual(delivery_name("{project}_{target}_{version}", {"project": "Talk 1", "target": "youtube", "version": "v1"}, "mp4"), "Talk_1_youtube_v1.mp4")
        self.assertEqual(delivery_name("{nope}", {"project": "p", "target": "t", "version": "v1"}, "mp4"), "p_t_v1.mp4")
        st = ArtifactStore(self.tmp)
        outside = str(Path(tempfile.mkdtemp()) / "x.mp4"); Path(outside).write_bytes(b"1")
        for bad in (outside, str(Path(self.tmp) / ".." / "x.mp4"), "relative.mp4"):
            with self.assertRaises(ArtifactError) as cm:
                st.check_path(bad)
            self.assertEqual(cm.exception.kind, "ARTIFACT_OUTSIDE_WORKSPACE")
        link = Path(self.tmp) / "link.mp4"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):
            link = None
        if link:
            with self.assertRaises(ArtifactError):
                st.check_path(str(link))
        inside = Path(self.tmp) / "jobs" / "j" / "artifacts" / "ok.mp4"; inside.parent.mkdir(parents=True); inside.write_bytes(b"ok")
        self.assertEqual(st.check_path(str(inside)), str(inside))

    # 29 execution ok but artifact registration fails -> job FAILED, no artifact recorded
    def test_registration_failure_keeps_job_consistent(self):
        class VanishingAdapter(FakeAdapter):
            def run(self, op, paths, timeout=None, dry_run=False, attempt=1):
                r = super().run(op, paths, timeout, dry_run, attempt)
                if op.tool == "ffmpeg-skill/check" and r.ok:
                    os.remove(paths.get(op.args["input"], op.args["input"]))   # output disappears after execution, before registration
                return r
        svc = make_service(self.tmp, adapter=VanishingAdapter())
        ir = svc.plan([self.src], "youtube"); p = str(Path(self.tmp) / "v.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["status"], "FAILED")
        self.assertEqual(out["artifact_error"]["kind"], "ARTIFACT_MISSING")
        self.assertEqual(out["job"]["state"], "FAILED")
        self.assertEqual(svc.artifacts(ir), [])

    # security: the artifact layer never executes or reaches AI / tools
    def test_artifact_layer_has_no_path_to_execution(self):
        root = Path(__file__).resolve().parents[1] / "src" / "video_agent"
        forbidden_imports = ("execution", "providers", "agent", "tools", "subprocess", "shlex", "ffmpeg", "shutil")
        forbidden_calls = ("subprocess", "os.system", "os.popen", "exec(", "eval(", "compile_ir", "Executor(", "__import__", ".complete(", ".run(", "shutil.")
        for rel in ("artifacts/store.py", "artifacts/naming.py"):
            for l in (root / rel).read_text(encoding="utf-8").splitlines():
                code = l.split("#", 1)[0]
                if code.lstrip().startswith(("import ", "from ")):
                    for f in forbidden_imports:
                        self.assertNotIn(f, code, f"{rel} must not import {f}: {l}")
                for f in forbidden_calls:
                    self.assertNotIn(f, code, f"{rel} must not call {f}: {l}")


class MediaAnalysisAdapterTests(unittest.TestCase):
    """External observation Skill boundary (ADR-023): protocol, contract compatibility, lifting, provenance, failures —
    verified against a fake media-analysis process that speaks contract@1 / response@1 (no ffmpeg, no import)."""

    FAKE = str(Path(__file__).resolve().parent / "fake_media_analysis.py")

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)
        os.environ.pop("FAKE_MA_MODE", None)
        os.environ.pop("FAKE_MA_CACHE", None)

    def tearDown(self):
        os.environ.pop("FAKE_MA_MODE", None)
        os.environ.pop("FAKE_MA_CACHE", None)

    def _skill(self):
        from video_agent.tools.media_analysis import MediaAnalysisSkill
        return MediaAnalysisSkill([sys.executable, self.FAKE], None, {})

    def _adapter(self, **kw):
        from video_agent.tools.media_analysis import MediaAnalysisAdapter
        return MediaAnalysisAdapter(self._skill(), workspace=self.tmp, **kw)

    # contract discovery + compatibility + package / tool mapping / capabilities
    def test_contract_discovery_package_and_tool_mapping(self):
        from video_agent.tools.media_analysis import PACKAGE, check_contract, pinned_contract
        ad = self._adapter()
        self.assertEqual(check_contract(ad.contract), [])
        self.assertEqual((ad.name, ad.version, ad.contract["schema"]), ("media-analysis", "0.1.0", "media-analysis/contract@1"))
        pkg = ad.package()
        self.assertEqual(pkg.validate(), [])
        self.assertEqual(set(pkg.tool_ids()), set(ad.contract["kind_to_tool"].values()), "every kind maps to a declared tool (timing serves two kinds)")
        self.assertEqual(pkg.tool_ids(), PACKAGE.tool_ids(), "the pinned snapshot and the live contract agree")
        self.assertEqual(pkg.capabilities, ["ffprobe"])
        self.assertEqual(ad.kinds_of("media-analysis/timing"), ["duration", "timing"])
        self.assertTrue(all(not t.produces_output and t.kind == "measure" for t in pkg.tools))
        self.assertTrue(ad.supports("media-analysis/silence") and not ad.supports("media-analysis/cut") and not ad.supports("ffmpeg-skill/probe"))
        self.assertEqual(pinned_contract()["skill_id"], "media-analysis")
        # capability names of the contract are CapabilityResolver names
        self.assertTrue(set(ad.contract["capability_names"]) <= {"ffmpeg", "ffprobe", "filter:ebur128", "filter:scdet", "filter:silencedetect"})

    def test_contract_incompatibility_is_refused(self):
        from video_agent.tools.media_analysis import ContractError, MediaAnalysisAdapter, check_contract, pinned_contract
        for mode in ("wrong_version", "wrong_schema_contract"):
            os.environ["FAKE_MA_MODE"] = mode
            with self.assertRaises(ContractError):
                MediaAnalysisAdapter(self._skill(), workspace=self.tmp)
        os.environ.pop("FAKE_MA_MODE")
        c = pinned_contract()
        for mutate, needle in ((lambda x: x.update(skill_id="other"), "skill_id"), (lambda x: x["execution"].update(mode="http"), "execution.mode"),
                               (lambda x: x["tools"][0].update(writes_media=True), "writes"), (lambda x: x.update(provenance="INFERRED"), "provenance"),
                               (lambda x: x["kind_to_tool"].update(extra="media-analysis/none"), "disagree"), (lambda x: x["execution"].update(ai=True), "ai")):
            d = json.loads(json.dumps(c)); mutate(d)
            self.assertTrue(any(needle in e for e in check_contract(d)), needle)

    # JSON request: typed args → AnalysisRequest; forbidden fields never cross
    def test_request_construction_and_refusals(self):
        from video_agent.tools import ToolError
        ad = self._adapter()
        req = ad.build_request("media-analysis/silence", {"input": self.src, "asset_id": "asset_1", "parameters": {"threshold_db": -45}, "analysis_id": "ana_x", "cache_policy": "only", "timeout": 5})
        self.assertEqual(sorted(req), ["analysis_id", "asset_id", "cache_policy", "input", "kind", "parameters", "timeout"])
        self.assertEqual((req["kind"], req["cache_policy"], req["parameters"]), ("silence", "only", {"threshold_db": -45}))
        self.assertTrue(os.path.isabs(req["input"]))
        with self.assertRaises(ToolError):
            ad.build_request("media-analysis/timing", {"input": self.src, "asset_id": "a"})          # two kinds: must be explicit
        for bad in ({"command": "rm -rf /"}, {"argv": ["ffmpeg"]}, {"executable": "/bin/sh"}, {"api_key": "sk"}, {"shell": "x"}):
            with self.assertRaises(ToolError):
                ad.build_request("media-analysis/probe", {"input": self.src, "asset_id": "a", **bad})
        with self.assertRaises(ToolError):
            ad.build_request("media-analysis/probe", {"input": self.src, "asset_id": "bad id!"})
        shaped = ad.measurement_args("media-analysis/silence", "silence", self.src, "asset_1", {"threshold_db": -40, "min_silence": 0.5, "bogus": 1}, "ana_1", "use")
        self.assertEqual(shaped["parameters"], {"threshold_db": -40}, "only the tool's declared parameters are forwarded")
        self.assertIn("run - --json", ad.preview(Operation(tool="media-analysis/probe", args={"input": self.src, "asset_id": "a"}, inputs=[], outputs=[]), {})[0])

    # JSON response → ToolResult, Observation lifting with provenance, cache metadata
    def test_response_and_observation_lifting(self):
        from video_agent.media import AnalysisRequest, MediaAnalyzer
        from video_agent.tools import ToolRouter
        ad = self._adapter()
        r = ad.measure("media-analysis/silence", {"input": self.src, "asset_id": "asset_1", "kind": "silence"})
        self.assertTrue(r.ok and r.exit_code == 0)
        self.assertEqual(r.data["observation"]["source"], "media-analysis/silence@0.1.0")
        self.assertEqual(r.data["cache"]["status"], "miss")
        self.assertEqual(r.commands, ["ffprobe: metadata"], "the Skill reports what it ran; the agent never built it")
        # through the analyzer: lifted Observation keeps the external identity and provenance; agent cache is not used
        svc = make_service(self.tmp, caps=FakeCaps(extra=["media-analysis"]), adapter=ToolRouter([ad]))
        svc.registry.get("media_probe").tools = ["media-analysis/probe"]
        svc.registry.get("silence_analysis").tools = ["media-analysis/silence"]
        svc.registry.get("loudness_analysis").tools = ["media-analysis/loudness"]
        an = MediaAnalyzer(svc.adapter([]), tools=svc.tools_for(), cache_dir=self.tmp)
        res = an.run(AnalysisRequest(inputs=[self.src], kinds=["silence", "loudness", "duration", "scene_detection"]))   # explicit kinds run exactly as requested (+ probe)
        obs = {o.kind: o for o in res.observations}
        self.assertEqual(sorted(obs), ["duration", "loudness", "media_probe", "scene_detection", "silence"])
        sil = obs["silence"]
        self.assertEqual((sil.provenance, sil.skill, sil.skill_version, sil.tool, sil.source), ("OBSERVED", "media-analysis", "0.1.0", "media-analysis/silence", "media-analysis/silence@0.1.0"))
        self.assertTrue(sil.external_id.startswith("obs_") and sil.external_id != sil.id)
        self.assertEqual(sil.fingerprint, "f" * 64)
        self.assertEqual(sil.analyzer, "media-analysis/silence@0.1.0")
        self.assertEqual(sil.cache["status"], "miss")
        self.assertEqual(sil.data["segments"][0]["type"], "leading", "facts as measured; no interpretation added")
        rows = {x["kind"]: x for x in res.analyses[0]["rows"]}
        self.assertEqual(rows["silence"]["cache_owner"], "media-analysis")
        self.assertEqual(an.cache.misses + an.cache.hits, 0, "the agent's own cache is not consulted for Skill-owned measurements")
        self.assertEqual(res.assets[0].technical["duration"], 16.0)
        self.assertEqual({e.type for e in res.timeline.events}, {"AUDIO_SILENCE", "AUDIO_ACTIVE", "LOUDNESS_MEASURE"})
        sil_ev = next(e for e in res.timeline.events if e.type == "AUDIO_SILENCE")
        self.assertEqual((sil_ev.range, sil_ev.evidence, sil_ev.source), ({"start": 0.0, "end": 3.0}, [sil.id], sil.source))
        os.environ["FAKE_MA_CACHE"] = "hit"
        res2 = MediaAnalyzer(svc.adapter([]), tools=svc.tools_for(), cache_dir=self.tmp).run(AnalysisRequest(inputs=[self.src]))
        self.assertTrue(all(x["cache_hit"] for x in res2.analyses[0]["rows"] if x["status"] == "OK"))
        self.assertEqual(next(o for o in res2.observations if o.kind == "silence").cache["status"], "hit")

    # malformed responses, error results, timeout, unavailable skill
    def test_malformed_responses_timeouts_and_unavailability(self):
        from video_agent.tools import ToolError
        from video_agent.tools.media_analysis import MediaAnalysisAdapter, MediaAnalysisSkill, locate_media_analysis
        ad = self._adapter()
        for mode in ("empty", "text", "two_docs", "wrong_schema", "wrong_skill", "wrong_kind", "no_observation", "bad_source", "crash"):
            os.environ["FAKE_MA_MODE"] = mode
            r = ad.measure("media-analysis/probe", {"input": self.src, "asset_id": "a"})
            self.assertFalse(r.ok, mode)
            self.assertIn(r.data["error"]["code"], ("INVALID_RESULT", "ANALYSIS_FAILED"), mode)
            self.assertIsNone(r.data.get("observation") if mode != "bad_source" else None, mode)
        os.environ["FAKE_MA_MODE"] = "error_result"
        r = ad.measure("media-analysis/probe", {"input": self.src, "asset_id": "a"})
        self.assertFalse(r.ok)
        self.assertEqual((r.data["error"]["code"], r.data["error_kind"]), ("ANALYZER_UNAVAILABLE", "ANALYZER_UNAVAILABLE"))
        os.environ["FAKE_MA_MODE"] = "hang"
        r = ad.measure("media-analysis/probe", {"input": self.src, "asset_id": "a"}, timeout=1.0)
        self.assertEqual((r.ok, r.exit_code, r.data["error"]["code"]), (False, 124, "ANALYZER_TIMEOUT"))
        os.environ.pop("FAKE_MA_MODE")
        self.assertIsNone(locate_media_analysis("/nonexistent", env={"PATH": "/nonexistent"}))
        with self.assertRaises(ToolError):
            MediaAnalysisAdapter(MediaAnalysisSkill([sys.executable, "-c", "import sys; sys.exit(3)"], None, {}), workspace=self.tmp)
        # the analyzer maps Skill failures into the analysis failure domain and keeps going
        from video_agent.media import AnalysisRequest, MediaAnalyzer
        from video_agent.tools import ToolRouter
        os.environ["FAKE_MA_MODE"] = "error_result"
        svc = make_service(self.tmp, caps=FakeCaps(extra=["media-analysis"]), adapter=ToolRouter([self._adapter()]))
        for name, tool in (("media_probe", "media-analysis/probe"), ("silence_analysis", "media-analysis/silence"), ("loudness_analysis", "media-analysis/loudness")):
            svc.registry.get(name).tools = [tool]
        from video_agent.media import AnalysisError
        with self.assertRaises(AnalysisError):
            MediaAnalyzer(svc.adapter([]), tools=svc.tools_for(), cache_dir=None).run(AnalysisRequest(inputs=[self.src]))   # the probe cannot be skipped

    # service / registry / doctor integration with the fake skill; ffmpeg-skill stays first
    def test_service_registration_selection_and_capability(self):
        from video_agent.capabilities.resolver import CapabilityResolver
        from video_agent.tools import ToolRouter
        ad = self._adapter()
        svc = make_service(self.tmp, caps=FakeCaps(extra=["media-analysis"]), adapter=ToolRouter([FakeAdapter(), ad]))
        tools = svc.tools_for()
        self.assertEqual(tools["media_probe"], "ffmpeg-skill/probe", "the Reference Skill stays the first candidate")
        self.assertEqual(tools["duration_analysis"], "media-analysis/timing")
        self.assertEqual(tools["scene_analysis"], "media-analysis/scenes")
        rows = {r["skill_id"]: r for r in svc.packages()}
        self.assertTrue(rows["media-analysis"]["implemented"] and rows["media-analysis"]["available"])
        self.assertEqual(rows["media-analysis"]["version"], "0.1.0")
        sk = {r["skill"]: r for r in svc.skills()}
        self.assertEqual((sk["integrity_analysis"]["status"], sk["integrity_analysis"]["tool"]), ("AVAILABLE", "media-analysis/integrity"))
        # capability resolver uses the Skill's doctor (no import): version / contract / tools / kinds / execution
        res = CapabilityResolver(ffmpeg_skill_dir="/nonexistent", env={"PATH": os.environ.get("PATH", "")}, media_analysis_dir="/nonexistent")
        cap = res.resolve()["media-analysis"]
        self.assertEqual(cap.status, "MISSING")
        # explicit extra kinds via the service, and Skill / Tool version distinction
        ir = svc.plan([self.src], "youtube")
        self.assertEqual(sorted(ir.doc["analysis"]["analyses"][0]["request"]["kinds"]), ["loudness", "media_probe", "silence"])
        profile, rules, an = svc.analyze([self.src], "youtube", kinds=["duration"])
        d = next(o for o in an.observations if o.kind == "duration")
        self.assertEqual((d.skill, d.tool, d.skill_version), ("media-analysis", "media-analysis/timing", "0.1.0"))
        self.assertNotEqual(svc.registry.tool("media-analysis/timing").version, "", "tool version recorded separately from the skill version")

    # boundaries: no import, no engine invocation, no event → command; observation ≠ inference
    def test_boundaries_static_and_dynamic(self):
        root = Path(__file__).resolve().parents[1] / "src" / "video_agent"
        for rel in ("tools/media_analysis/adapter.py", "tools/media_analysis/locate.py"):
            text = (root / rel).read_text(encoding="utf-8")
            for l in text.splitlines():
                code = l.split("#", 1)[0]
                if code.lstrip().startswith(("import ", "from ")):
                    self.assertNotIn("media_analysis_skill", code); self.assertNotIn("media_analysis.", code.replace("tools.media_analysis", "").replace("from .", ""))
                    self.assertNotIn("providers", code); self.assertNotIn("execution", code)
            self.assertNotIn("shell=True", text)
            body = "\n".join(l for l in text.splitlines() if "FORBIDDEN_ARG_KEYS" not in l)   # the refusal list names ffprobe as a forbidden key
            self.assertNotIn('"ffprobe"', body); self.assertNotIn("subprocess.Popen", text.replace("run_process_group", ""))
        src_all = "\n".join((root / p).read_text(encoding="utf-8") for p in ("media/analyzer.py", "media/analysis.py", "temporal/events.py"))
        self.assertNotIn("import media_analysis", src_all)
        # a media-analysis silence observation yields events, never operations; the plan comes only from decisions
        from video_agent.tools import ToolRouter
        svc = make_service(self.tmp, caps=FakeCaps(extra=["media-analysis"]), adapter=ToolRouter([FakeAdapter(), self._adapter()]))
        ir = svc.plan([self.src], "youtube")
        self.assertFalse(any("media-analysis" in st["tool"] for st in ir.doc["plan"]["steps"]), "measurement tools never appear as production steps")
        ops, _ = compile_ir(ir, "/w/jobs/j")
        self.assertFalse(any(o.tool.startswith("media-analysis/") for o in ops))
        self.assertTrue(all(i["provenance"] in ("INFERRED", "AI_GENERATED") for i in ir.doc["analysis"]["inferences"]))
        self.assertTrue(all(o["provenance"] == "OBSERVED" for o in ir.doc["analysis"]["observations"]))


class TranscriptionAdapterTests(unittest.TestCase):
    """External recognition Skill boundary (transcription-skill, ADR-024): transport, contract compatibility, typed requests,
    input boundary, Transcript lifting with provenance, SpeechEvents, failures — against a fake transcription process that
    speaks the real `skill --json` / `doctor --json` / `run -` protocol (no ASR engine, no ffmpeg, no import)."""

    FAKE = str(Path(__file__).resolve().parent / "fake_transcription.py")
    CLEAR = ("FAKE_TS_MODE", "FAKE_TS_CACHE", "FAKE_TS_CALLS")

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)
        for k in self.CLEAR:
            os.environ.pop(k, None)

    def tearDown(self):
        for k in self.CLEAR:
            os.environ.pop(k, None)

    def _skill(self):
        from video_agent.tools.transcription import TranscriptionSkill
        return TranscriptionSkill([sys.executable, self.FAKE], None, {})

    def _adapter(self, **kw):
        from video_agent.tools.transcription import TranscriptionAdapter
        kw.setdefault("workspace", str(Path(self.tmp) / "ws" / "cache" / "transcription"))
        return TranscriptionAdapter(self._skill(), **kw)

    def _checkout(self) -> str:
        """A checkout-shaped directory whose `transcription_skill.cli` module runs the fake, so locate / resolver / Service find it."""
        root = Path(self.tmp) / "fake-transcription-skill"
        pkg = root / "src" / "transcription_skill"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "cli.py").write_text(f"import runpy\nrunpy.run_path({self.FAKE!r}, run_name='__main__')\n", encoding="utf-8")
        return str(root)

    def _service(self, adapters=None, extra=("transcription",), **kw):
        from video_agent.tools import ToolRouter
        return make_service(self.tmp, caps=FakeCaps(extra=list(extra)), adapter=ToolRouter(adapters if adapters is not None else [FakeAdapter(), self._adapter(**kw)]))

    # 1. contract discovery, compatibility, package, tool ownership, engine contract
    def test_contract_discovery_package_and_refusals(self):
        from video_agent.tools.transcription import PACKAGE, ContractError, TranscriptionAdapter, check_contract, pinned_contract
        ad = self._adapter()
        self.assertEqual(check_contract(ad.contract), [])
        self.assertEqual((ad.name, ad.version, ad.contract["id"]), ("transcription", "0.2.0", "transcription-skill"))
        self.assertEqual(ad.contract["schemas"], {"transcript": "transcription-skill/transcript/0.1", "speech_event": "transcription-skill/speech-event/0.1", "engine_spec": "transcription-skill/engine-spec/0.1"})
        pkg = ad.package()
        self.assertEqual(pkg.validate(), [])
        self.assertEqual(pkg.tool_ids(), PACKAGE.tool_ids(), "pinned snapshot and live contract agree on the tools")
        self.assertEqual((pkg.skill_id, pkg.capabilities), ("transcription", ["ffmpeg", "ffprobe"]))
        self.assertTrue(all(not t.produces_output and t.kind == "measure" for t in pkg.tools))
        self.assertTrue(ad.supports("transcription/transcribe"))
        for other in ("transcription/segments", "transcription/export", "transcription/check", "media-analysis/probe", "ffmpeg-skill/cut"):
            self.assertFalse(ad.supports(other), f"{other}: recognition only")
        eng = {e["id"]: e for e in ad.engine_status()}
        self.assertEqual((eng["faster_whisper"]["execution_mode"], eng["faster_whisper"]["requires_network"]), ("local", False))
        self.assertEqual({m["model"]: m["availability"] for m in eng["faster_whisper"]["models"]}, {"tiny": "MODEL_DOWNLOAD_REQUIRED", "base": "MODEL_AVAILABLE", "small": "MODEL_DOWNLOAD_REQUIRED"},
                         "model availability is the Skill's vocabulary, untouched")
        self.assertEqual(pinned_contract()["id"], "transcription-skill")
        self.assertFalse(any(e.get("available") for e in pinned_contract()["engines"]), "the snapshot never claims an engine is available")
        for mode in ("wrong_version", "wrong_schema", "wrong_skill"):
            os.environ["FAKE_TS_MODE"] = mode
            with self.assertRaises(ContractError, msg=mode):
                TranscriptionAdapter(self._skill(), workspace=self.tmp)
        os.environ.pop("FAKE_TS_MODE")
        c = pinned_contract()
        for mutate, needle in ((lambda x: x["engine_contract"].update(execution_modes=["remote"]), "local"), (lambda x: x["tools"][0].update(name="other/transcribe"), "belong"),
                               (lambda x: x["tools"][0].update(side_effects=["writes output file"]), "side effect"), (lambda x: x.update(capabilities=["speech_recognition"]), "capability"),
                               (lambda x: x["tools"][0]["input"].pop("allowed_input_roots"), "allowed_input_roots"), (lambda x: x["engines"][0].update(requires_network="yes"), "malformed"),
                               (lambda x: x.update(engines=[]), "no engines"), (lambda x: x["schemas"].update(engine_spec="transcription-skill/engine-spec/0.2"), "engine_spec")):
            d = json.loads(json.dumps(c)); mutate(d)
            self.assertTrue(any(needle in e for e in check_contract(d)), (needle, check_contract(d)))

    # 2. typed request → {"tool", "params"}; nothing else crosses; adapter pins workspace and roots; offline only tightens
    def test_request_construction_and_refusals(self):
        from video_agent.tools import ToolError
        roots = [str(Path(self.tmp) / "src")]
        ad = self._adapter(allowed_inputs=roots)
        req = ad.build_request("transcription/transcribe", {"input": self.src, "asset_id": "asset_1", "language": "JA", "model": "base", "word_timestamps": True, "beam_size": 3,
                                                            "timeout": 30, "cache_policy": "bypass", "analysis_id": "ana_x", "kind": "transcript"})
        self.assertEqual(req["tool"], "transcription/transcribe")
        p = req["params"]
        self.assertEqual(sorted(p), ["allowed_input_roots", "asset_id", "beam_size", "budget", "cache", "input", "language", "model", "word_timestamps", "workspace"])
        self.assertEqual((p["language"], p["cache"], p["budget"], p["allowed_input_roots"][0], p["word_timestamps"]), ("ja", False, {"timeout": 30.0}, os.path.realpath(roots[0]), True))
        self.assertEqual(p["workspace"], ad.workspace)
        self.assertTrue(os.path.isabs(p["input"]))
        for bad in ({"command": "rm -rf /"}, {"argv": ["x"]}, {"executable": "/bin/sh"}, {"api_key": "sk"}, {"shell": "x"}, {"env": {"X": "1"}}, {"token": "t"}):
            with self.assertRaises(ToolError, msg=str(bad)):
                ad.build_request("transcription/transcribe", {"input": self.src, "asset_id": "a", **bad})
        for pinned in ({"workspace": "/elsewhere"}, {"allowed_input_roots": ["/"]}):
            with self.assertRaises(ToolError, msg=str(pinned)):
                ad.build_request("transcription/transcribe", {"input": self.src, "asset_id": "a", **pinned})
        for bad in ({"engine": "cloud_asr"}, {"engine": "../x"}, {"model": "../../etc/passwd"}, {"model": "/abs/model"}, {"language": "japanese"}, {"beam_size": 0}, {"temperature": 2},
                    {"word_timestamps": "yes"}, {"asset_id": "bad id!"}, {"timeout": -1}, {"cache_policy": "sometimes"}):
            with self.assertRaises(ToolError, msg=str(bad)):
                ad.build_request("transcription/transcribe", {"input": self.src, "asset_id": "a", **bad})
        with self.assertRaises(ToolError):
            ad.build_request("transcription/segments", {"input": self.src, "asset_id": "a"})
        off = self._adapter(offline=True)
        self.assertTrue(off.build_request("transcription/transcribe", {"input": self.src, "asset_id": "a", "offline": False})["params"]["offline"], "a request cannot loosen the adapter's offline constraint")
        shaped = ad.measurement_args("transcription/transcribe", "transcript", self.src, "asset_1", {"language": "ja", "bogus": 1, "threshold_db": -40, "timeout": 12}, "ana_1", "use")
        self.assertEqual((shaped["parameters"], shaped["budget"]), ({"language": "ja"}, {"timeout": 12}), "only the Skill's declared typed keys are forwarded")
        self.assertIn("run -", ad.preview(Operation(tool="transcription/transcribe", args={"input": self.src, "asset_id": "a"}, inputs=[], outputs=[]), {})[0])

    # 3. valid transcript → ToolResult → Observation (provenance intact) → SpeechEvents; shared asset identity; Skill-owned cache
    def test_transcript_lifting_provenance_and_speech_events(self):
        from video_agent.media import AnalysisRequest, MediaAnalyzer
        from video_agent.media.analyzer import sha256_file
        ad = self._adapter()
        r = ad.measure("transcription/transcribe", {"input": self.src, "asset_id": "asset_1", "language": "ja"})
        self.assertTrue(r.ok and r.exit_code == 0, r.data)
        self.assertEqual(r.data["cache"], {"status": "miss", "key": "c" * 64, "owner": "transcription"})
        self.assertEqual(r.data["engine"]["id"], "faster_whisper")
        self.assertEqual(r.commands, ["faster_whisper@1.2.1-fake: recognition (local)"], "the Skill reports what ran; the agent built no command")
        svc = self._service()
        an = MediaAnalyzer(svc.adapter([]), tools=svc.tools_for(), cache_dir=self.tmp)
        res = an.run(AnalysisRequest(inputs=[self.src], kinds=["silence", "loudness", "transcript"], params={"language": "ja", "word_timestamps": True}))
        obs = {o.kind: o for o in res.observations}
        self.assertEqual(sorted(obs), ["loudness", "media_probe", "silence", "transcript"])
        t = obs["transcript"]
        self.assertEqual((t.provenance, t.skill, t.skill_version, t.tool, t.source), ("OBSERVED", "transcription", "0.2.0", "transcription/transcribe", "transcription/transcribe@0.2.0"))
        self.assertTrue(t.external_id.startswith("tr_") and t.external_id == t.data["id"] and t.external_id != t.id)
        self.assertEqual(t.fingerprint, sha256_file(self.src), "shared asset identity: the Skill's fingerprint is the agent's asset hash")
        self.assertEqual(t.fingerprint, res.assets[0].hash)
        self.assertEqual(t.asset_id, res.assets[0].id, "the transcript is stamped with the agent's asset id, never a second asset")
        self.assertEqual(t.data["asset_id"], res.assets[0].id)
        self.assertEqual(t.analyzer, "faster_whisper@1.2.1-fake")
        self.assertEqual({k: t.parameters[k] for k in ("engine", "engine_version", "execution_mode", "model", "model_version", "language", "word_timestamps")},
                         {"engine": "faster_whisper", "engine_version": "1.2.1-fake", "execution_mode": "local", "model": "base", "model_version": "fake-model-rev", "language": "ja", "word_timestamps": True})
        self.assertEqual(t.cache, {"status": "miss", "key": "c" * 64, "owner": "transcription"})
        self.assertEqual(t.data["schema"], "transcription-skill/transcript/0.1")
        self.assertEqual(t.data["segments"][0]["text"], "本日の公園を始めます", "recognised text as-is, homophone errors included")
        self.assertTrue(all(s["speaker_id"] is None for s in t.data["segments"]))
        self.assertEqual(t.data["provenance"]["skill"], "transcription-skill")
        rows = {x["kind"]: x for x in res.analyses[0]["rows"]}
        self.assertEqual((rows["transcript"]["cache_owner"], rows["transcript"]["cache_hit"], rows["transcript"]["engine"]["model"]), ("transcription", False, "base"))
        self.assertEqual(an.cache.hits + an.cache.misses, 3, "the agent's cache serves the three engine measurements only; recognition is never stored in it")
        self.assertEqual(rows["transcript"].get("produced_by"), None)
        self.assertFalse(any("transcript" in str(k) for k in (Path(self.tmp) / "cache").rglob("*")) if (Path(self.tmp) / "cache").exists() else False, "no agent-side transcript cache file")
        # SpeechEvents: one per segment, OBSERVED, evidence = the transcript observation, speaker_id null, no command-like content
        sp = res.timeline.query(type="SPEECH")
        self.assertEqual(len(sp), 2)
        self.assertEqual([(e.range["start"], e.range["end"]) for e in sp], [(0.5, 3.2), (4.0, 8.4)])
        for e in sp:
            self.assertEqual((e.kind, e.provenance, e.event_type, e.subtype, e.evidence, e.source, e.asset_id), ("OBSERVED", "OBSERVED", "SpeechEvent", "speech", [t.id], t.source, res.assets[0].id))
            self.assertIsNone(e.metadata["speaker_id"])
            self.assertIn("text", e.metadata)
            self.assertNotIn("speaker", json.dumps({k: v for k, v in e.metadata.items() if k != "speaker_id"}))
        self.assertEqual(sp[0].confidence, 0.72)
        self.assertEqual(sp[0].metadata["words"], 2)
        self.assertEqual({e.type for e in res.timeline.events}, {"AUDIO_SILENCE", "AUDIO_ACTIVE", "LOUDNESS_MEASURE", "SPEECH"})
        # deterministic: the same transcript yields the same event ids
        from video_agent.temporal.events import events_from_observation
        again = events_from_observation(t, res.assets[0])
        self.assertEqual([e.id for e in again], [e.id for e in sp])
        # cache hit is provenance, not a second cache
        os.environ["FAKE_TS_CACHE"] = "hit"
        res2 = MediaAnalyzer(svc.adapter([]), tools=svc.tools_for(), cache_dir=self.tmp).run(AnalysisRequest(inputs=[self.src], kinds=["transcript"], params={"language": "ja"}))
        t2 = next(o for o in res2.observations if o.kind == "transcript")
        self.assertEqual((t2.cache["status"], t2.cache["stored_asset_id"]), ("hit", "asset_first_caller"))
        self.assertEqual((t2.asset_id, t2.fingerprint), (res2.assets[0].id, res2.assets[0].hash), "a cached document keeps its bytes; identity is the fingerprint, the agent's asset id is this analysis'")
        os.environ.pop("FAKE_TS_CACHE")
        os.environ["FAKE_TS_MODE"] = "wrong_asset"
        res3 = MediaAnalyzer(svc.adapter([]), tools=svc.tools_for(), cache_dir=self.tmp).run(AnalysisRequest(inputs=[self.src], kinds=["transcript"]))
        self.assertNotIn("transcript", {o.kind for o in res3.observations}, "a fresh recognition about another asset is refused")
        self.assertTrue(next(x for x in res2.analyses[0]["rows"] if x["kind"] == "transcript")["cache_hit"])

    # 4. malformed / failed responses are never transcripts; failure kinds stay distinct
    def test_failures_are_distinct_and_never_partial_transcripts(self):
        from video_agent.media import AnalysisRequest, MediaAnalyzer
        from video_agent.tools import ToolError
        from video_agent.tools.transcription import TranscriptionAdapter, TranscriptionSkill, locate_transcription
        ad = self._adapter()
        expect = {"empty": "INVALID_RESULT", "text": "INVALID_RESULT", "two_docs": "INVALID_RESULT", "no_transcript": "INVALID_RESULT", "wrong_engine": "INVALID_RESULT",
                  "wrong_asset": "INVALID_RESULT", "bad_source": "INVALID_RESULT", "invalid_provenance": "INVALID_RESULT", "speaker_set": "INVALID_RESULT", "bad_segments": "INVALID_RESULT",
                  "crash": "INVALID_RESULT", "nonzero": "TRANSCRIPTION_FAILED", "timeout_error": "TRANSCRIPTION_TIMEOUT", "model_unavailable": "MODEL_UNAVAILABLE", "engine_unavailable": "ENGINE_UNAVAILABLE"}
        for mode, code in expect.items():
            os.environ["FAKE_TS_MODE"] = mode
            r = ad.measure("transcription/transcribe", {"input": self.src, "asset_id": "asset_1"})
            self.assertFalse(r.ok, mode)
            self.assertEqual(r.data["error"]["code"], code, (mode, r.data["error"]))
            self.assertNotIn("transcript", r.data, f"{mode}: no partial transcript is ever a result")
        self.assertEqual(r.data["error"]["details"]["reason"], "engine_not_installed")
        os.environ["FAKE_TS_MODE"] = "model_unavailable"
        r = ad.measure("transcription/transcribe", {"input": self.src, "asset_id": "a", "offline": True})
        self.assertEqual(r.data["error"]["details"]["availability"], "MODEL_MISSING", "the Skill's model vocabulary is preserved, not re-interpreted")
        os.environ["FAKE_TS_MODE"] = "crash"
        self.assertEqual(ad.measure("transcription/transcribe", {"input": self.src, "asset_id": "a"}).exit_code, 8)
        os.environ["FAKE_TS_MODE"] = "hang"
        r = ad.measure("transcription/transcribe", {"input": self.src, "asset_id": "a"}, timeout=1.0)
        self.assertEqual((r.ok, r.exit_code, r.data["error"]["code"]), (False, 124, "TRANSCRIPTION_TIMEOUT"))
        os.environ.pop("FAKE_TS_MODE")
        self.assertIsNone(locate_transcription("/nonexistent", env={"PATH": "/nonexistent"}))
        with self.assertRaises(ToolError):
            TranscriptionAdapter(TranscriptionSkill([sys.executable, "-c", "import sys; sys.exit(3)"], None, {}), workspace=self.tmp)
        # through the analyzer: the failure lands in the analysis failure domain; the transcript row is FAILED, no observation, no event
        for mode, kind in (("two_docs", "ANALYSIS_INVALID_RESULT"), ("speaker_set", "ANALYSIS_INVALID_RESULT"), ("nonzero", "ANALYZER_UNAVAILABLE"), ("timeout_error", "ANALYZER_TIMEOUT"),
                           ("model_unavailable", "ANALYZER_UNAVAILABLE"), ("engine_unavailable", "ANALYZER_UNAVAILABLE")):
            os.environ["FAKE_TS_MODE"] = mode
            svc = self._service()
            res = MediaAnalyzer(svc.adapter([]), tools=svc.tools_for(), cache_dir=None).run(AnalysisRequest(inputs=[self.src], kinds=["transcript"]))
            row = next(x for x in res.analyses[0]["rows"] if x["kind"] == "transcript")
            self.assertEqual((row["status"], row["error"]["kind"]), ("FAILED", kind), mode)
            self.assertNotIn("transcript", {o.kind for o in res.observations})
            self.assertEqual(res.timeline.query(type="SPEECH"), [])
            self.assertTrue(any("transcript" in w for w in res.warnings))
        self.assertEqual(row["error"]["skill_error"], "ENGINE_UNAVAILABLE")
        # fingerprint mismatch (the Skill transcribed other bytes) is refused as a shared-identity violation
        os.environ.pop("FAKE_TS_MODE")
        from video_agent.models import Asset
        from video_agent.tools.transcription.adapter import check_transcript
        ok = ad.measure("transcription/transcribe", {"input": self.src, "asset_id": "asset_1"}).data
        asset = Asset(path=self.src, hash="0" * 64)
        o = MediaAnalyzer._lift_transcript(ok, asset, AnalysisRequest(inputs=[self.src], kinds=["transcript"]), "transcription/transcribe@0.2.0", "k", "transcription/transcribe")
        self.assertNotEqual(o.fingerprint, asset.hash)
        tr = json.loads(json.dumps(ok["transcript"])); tr["provenance"]["tool"] = "transcription/export"
        self.assertTrue(any("provenance.tool" in e for e in check_transcript(tr, {"asset_id": "asset_1"}, "0.2.0", ad.engines, ad.contract["schemas"])))

    # 5. cache policy `only`: the Skill has no cache-only mode, so a dry run decides; nothing is recognised on a miss
    def test_cached_only_policy(self):
        ad = self._adapter()
        r = ad.measure("transcription/transcribe", {"input": self.src, "asset_id": "a", "cache_policy": "only"})
        self.assertEqual((r.ok, r.data["error"]["code"]), (False, "CACHE_MISS"))
        os.environ["FAKE_TS_CACHE"] = "hit"
        calls = ad.calls
        r = ad.measure("transcription/transcribe", {"input": self.src, "asset_id": "a", "cache_policy": "only"})
        self.assertTrue(r.ok and r.data["cache"]["status"] == "hit")
        self.assertEqual(ad.calls - calls, 2, "one dry run + one cached run; no other process")

    # 6. input boundary: allowed roots, traversal, symlink escape — refused by the adapter and by the Skill alike
    def test_input_boundary_allowed_roots_and_symlink_escape(self):
        import subprocess
        roots = [str(Path(self.tmp) / "src")]
        ad = self._adapter(allowed_inputs=roots)
        outside = Path(self.tmp) / "outside" / "secret.mp4"
        outside.parent.mkdir(); outside.write_bytes(b"\x00" * 64)
        link = Path(self.tmp) / "src" / "link.mp4"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not available")
        calls = ad.calls
        cases = ((str(outside), "outside_allowed_roots"), (str(link), "symlink_escape"), (str(Path(self.tmp) / "src" / ".." / "outside" / "secret.mp4"), "traversal"))
        for path, reason in cases:
            r = ad.measure("transcription/transcribe", {"input": path, "asset_id": "a"})
            self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_INPUT"), path)
            self.assertIn(reason, r.data["error"]["message"])
        self.assertEqual(ad.calls, calls, "refused before any process starts")
        self.assertTrue(ad.measure("transcription/transcribe", {"input": self.src, "asset_id": "a"}).ok, "inside the root: accepted")
        # the Skill enforces the same boundary with the roots the adapter passes (defence in depth): drive the fake directly
        for path, reason in cases[:2]:
            req = {"tool": "transcription/transcribe", "params": {"input": path, "asset_id": "a", "allowed_input_roots": roots}}
            p = subprocess.run([sys.executable, self.FAKE, "run", "-"], input=json.dumps(req), capture_output=True, text=True)
            doc = json.loads(p.stdout)
            self.assertEqual((p.returncode, doc["ok"], doc["error"]["code"], doc["error"]["details"]["reason"]), (2, False, "INVALID_INPUT", reason))
        # no roots declared on the adapter → the Skill's unrestricted mode; the agent's own Service always declares roots
        self.assertNotIn("allowed_input_roots", self._adapter().build_request("transcription/transcribe", {"input": self.src, "asset_id": "a"})["params"])
        svc = Service(workspace=self.tmp, transcription_dir=self._checkout(), caps=FakeCaps(extra=["transcription"]))
        ts = next(a for a in svc.adapter([str(Path(self.tmp) / "src")]).adapters if a.name == "transcription")
        self.assertEqual(ts.allowed_inputs, [os.path.realpath(str(Path(self.tmp) / "src")), os.path.realpath(svc.workspace)], "Service roots: inputs' directories + the workspace, like the engine's PathPolicy")
        self.assertEqual(ts.workspace, os.path.realpath(str(Path(svc.workspace) / "cache" / "transcription")))

    # 7. registry / capability resolver / Service / doctor: transcription is a capability, the tool a candidate only when it is present
    def test_registry_capability_service_and_doctor(self):
        from video_agent.capabilities.resolver import CapabilityResolver
        svc = self._service()
        tools = svc.tools_for()
        self.assertEqual(tools["speech_transcription"], "transcription/transcribe")
        self.assertEqual(svc.registry.get("speech_transcription").required_capabilities, ["ffmpeg", "ffprobe", "transcription"])
        rows = {r["skill_id"]: r for r in svc.packages()}
        self.assertTrue(rows["transcription"]["implemented"] and rows["transcription"]["available"])
        self.assertEqual((rows["transcription"]["version"], rows["transcription"]["used_by"]), ("0.2.0", ["speech_transcription"]))
        sk = {r["skill"]: r for r in svc.skills()}
        self.assertEqual((sk["speech_transcription"]["status"], sk["speech_transcription"]["approval"], sk["speech_transcription"]["risk"]), ("AVAILABLE", "AUTO", "LOW"))
        self.assertEqual(sk["caption_generation"]["status"], "NOT_IMPLEMENTED", "captions / burn-in stay unimplemented")
        no_cap = self._service(extra=())
        self.assertNotIn("speech_transcription", no_cap.tools_for(), "without the capability the tool is never a candidate")
        self.assertIn("required capability missing: transcription", svc.registry.select_tool("speech_transcription", FakeCaps().resolve(), lambda t: True)[1])
        with self.assertRaises(RuntimeError):
            no_cap.analyze([self.src], "generic", kinds=["transcript"])
        # resolver: the Skill's own doctor decides; evidence carries engines / models / schemas; nothing secret
        root = self._checkout()
        cap = CapabilityResolver(ffmpeg_skill_dir="/nonexistent", env={"PATH": os.environ.get("PATH", "")}, transcription_dir=root).resolve()["transcription"]
        self.assertEqual(cap.status, "AVAILABLE", cap.detail)
        self.assertEqual((cap.evidence["version"], cap.evidence["engines"][0]["id"], cap.evidence["engines"][0]["execution_mode"], cap.evidence["doctor_ok"]), ("0.2.0", "faster_whisper", "local", True))
        self.assertEqual({m["model"]: m["availability"] for m in cap.evidence["engines"][0]["models"]}["base"], "MODEL_AVAILABLE")
        self.assertNotRegex(json.dumps(cap.to_dict()), r"(?i)(api[_-]?key|token|secret|password)")
        os.environ["FAKE_TS_MODE"] = "model_unavailable"
        cap = CapabilityResolver(ffmpeg_skill_dir="/nonexistent", env={"PATH": ""}, transcription_dir=root, offline=True).resolve(refresh=True)["transcription"]
        self.assertEqual(cap.status, "DEGRADED", "engine present, model missing: the Skill is installed but not ready")
        os.environ["FAKE_TS_MODE"] = "engine_unavailable"
        cap = CapabilityResolver(ffmpeg_skill_dir="/nonexistent", env={"PATH": ""}, transcription_dir=root).resolve(refresh=True)["transcription"]
        self.assertEqual(cap.status, "MISSING")
        os.environ["FAKE_TS_MODE"] = "wrong_version"
        cap = CapabilityResolver(ffmpeg_skill_dir="/nonexistent", env={"PATH": ""}, transcription_dir=root).resolve(refresh=True)["transcription"]
        self.assertEqual(cap.status, "MISSING"); self.assertIn("unusable", cap.detail)
        os.environ.pop("FAKE_TS_MODE")
        self.assertEqual(CapabilityResolver(ffmpeg_skill_dir="/nonexistent", env={"PATH": ""}, transcription_dir="/nonexistent").resolve()["transcription"].status, "MISSING")
        # Service(offline=True) reaches the adapter; the Skill refuses a model that is not local
        off = Service(workspace=self.tmp, transcription_dir=root, caps=FakeCaps(extra=["transcription"]), offline=True)
        ts = next(a for a in off.adapter([str(Path(self.tmp) / "src")]).adapters if a.name == "transcription")
        self.assertTrue(ts.offline)
        r = ts.measure("transcription/transcribe", {"input": self.src, "asset_id": "a", "model": "small"})
        self.assertEqual((r.data["error"]["code"], r.data["error"]["details"]["availability"], r.data["error"]["details"]["offline"]), ("MODEL_UNAVAILABLE", "MODEL_MISSING", True))

    # 8. Observation → SpeechEvent only: no inference, decision, plan step or command derives from a transcript; explain stops at facts
    def test_speech_events_never_become_commands_and_explain_chain(self):
        from video_agent.tools import ToolRouter
        svc = self._service()
        ir = svc.plan([self.src], "youtube", kinds=["transcript"], params={"language": "ja"})
        d = ir.doc
        t = next(o for o in d["analysis"]["observations"] if o["kind"] == "transcript")
        sp = [e for e in d["timeline"]["events"] if e["type"] == "SPEECH"]
        self.assertEqual(len(sp), 2)
        self.assertTrue(all(e["evidence"] == [t["id"]] and e["provenance"] == "OBSERVED" for e in sp))
        self.assertEqual(svc.validate(ir).errors, [])
        blob = json.dumps({"plan": d["plan"], "video": d["video"], "audio": d["audio"], "delivery": d["delivery"], "decisions": d["decisions"], "inferences": d["analysis"]["inferences"]})
        for forbidden in ("SPEECH", "transcription", "speaker", t["id"], "faster_whisper"):
            self.assertNotIn(forbidden, blob, f"{forbidden!r} reached the decision / plan layer")
        self.assertEqual([s["skill"] for s in d["plan"]["steps"]], ["silence_cleanup", "loudness_normalization", "delivery_export", "delivery_check"])
        self.assertTrue(all(s["tool"].startswith("ffmpeg-skill/") for s in d["plan"]["steps"]))
        self.assertFalse([dec for dec in d["decisions"] if any(ev in {e["id"] for e in sp} for ev in dec["evidence"])], "no decision cites a SpeechEvent")
        info = Service.explain_observation(d, t["id"])
        kinds = [row["kind"] for row in info["chain"]]
        self.assertEqual(kinds, ["observation", "skill", "tool", "engine", "model", "transcript", "asset", "analysis", "event", "event"])
        self.assertTrue(next(row for row in info["chain"] if row["kind"] == "asset")["shared_identity"])
        self.assertEqual(next(row for row in info["chain"] if row["kind"] == "engine")["id"], "faster_whisper@1.2.1-fake")
        self.assertEqual(Service.explain_observation(d, t["external_id"])["observation"]["id"], t["id"], "the Skill's transcript id resolves too")
        self.assertIn("no inference, decision", info["boundary"])
        # media-analysis and transcription observations of one asset share its identity
        from video_agent.tools.media_analysis import MediaAnalysisAdapter, MediaAnalysisSkill
        ma = MediaAnalysisAdapter(MediaAnalysisSkill([sys.executable, str(Path(__file__).resolve().parent / "fake_media_analysis.py")], None, {}), workspace=self.tmp)
        svc2 = make_service(self.tmp, caps=FakeCaps(extra=["media-analysis", "transcription"]), adapter=ToolRouter([FakeAdapter(), ma, self._adapter()]))
        _, _, an = svc2.analyze([self.src], "generic", kinds=["duration", "transcript"])
        obs = {o.kind: o for o in an.observations}
        self.assertEqual((obs["duration"].asset_id, obs["transcript"].asset_id), (an.assets[0].id, an.assets[0].id))
        self.assertEqual(len(an.assets), 1, "one asset, two Skills: no second asset for the transcript")
        self.assertEqual(obs["transcript"].fingerprint, an.assets[0].hash)

    # 9. static boundaries: no import of the Skill or an engine, no engine execution, no shell, no event → command path
    def test_boundaries_static(self):
        root = Path(__file__).resolve().parents[1] / "src" / "video_agent"
        for rel in ("tools/transcription/adapter.py", "tools/transcription/locate.py"):
            text = (root / rel).read_text(encoding="utf-8")
            for l in text.splitlines():
                code = l.split("#", 1)[0]
                if code.lstrip().startswith(("import ", "from ")):
                    for bad in ("transcription_skill", "faster_whisper", "whisper", "ctranslate2", "providers", "execution", "urllib", "http", "socket", "requests"):
                        self.assertNotIn(bad, code, f"{rel}: {l}")
            for bad in ("shell=True", "os.system", "subprocess.Popen", "eval(", "exec(", "__import__", "huggingface", "snapshot_download"):
                self.assertNotIn(bad, text.replace("run_process_group", ""), f"{rel} contains {bad}")
        for rel in ("temporal/events.py", "media/analyzer.py", "media/analysis.py", "agent/decision.py", "agent/inference.py", "agent/planner.py", "agent/production_plan.py", "execution/compiler.py"):
            text = (root / rel).read_text(encoding="utf-8")
            self.assertNotIn("transcription_skill", text); self.assertNotIn("faster_whisper", text)
        for rel in ("agent/decision.py", "agent/inference.py", "agent/planner.py", "agent/production_plan.py", "execution/compiler.py", "execution/executor.py"):
            text = (root / rel).read_text(encoding="utf-8")
            self.assertNotIn("SPEECH", text, f"{rel}: SpeechEvents must not feed decisions, plans or commands")
            self.assertNotIn("speaker", text.lower(), f"{rel}: no speaker identity anywhere")
        from video_agent.temporal.events import EVENT_CODES, IMPLEMENTED_CODES
        self.assertIn("SPEECH", IMPLEMENTED_CODES); self.assertNotIn("SPEAKER", IMPLEMENTED_CODES)
        self.assertEqual(EVENT_CODES["SPEECH"], ("SpeechEvent", "speech"))
