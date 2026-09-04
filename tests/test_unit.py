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
    def __init__(self, missing=()):
        self.missing = set(missing)

    def resolve(self, refresh=False):
        names = ["python", "ffmpeg", "ffprobe", "ffmpeg-skill", "encoder:libx264", "encoder:libx265", "encoder:prores_ks", "filter:loudnorm", "font:cjk-ja"]
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
        self.assertEqual(kinds, {"probe", "silence", "loudness"})
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
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        self.assertEqual(ir.doc["analysis"]["strategy"], "FULL_ANALYSIS")
        self.assertFalse(ir.doc["analysis"]["budget"]["enforced"])

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
        self.assertEqual(list(pkgs), ["ffmpeg-skill"], "exactly one implemented package: the Reference Skill")
        self.assertEqual(PACKAGE.validate(), [])
        self.assertEqual((PACKAGE.repository, PACKAGE.capabilities), ("kajisho5/ffmpeg-skill", ["ffmpeg", "ffprobe", "ffmpeg-skill"]))
        rows = svc.packages()
        self.assertEqual([r["skill_id"] for r in rows], ["ffmpeg-skill"])
        self.assertTrue(rows[0]["implemented"] and rows[0]["available"])
        self.assertEqual(rows[0]["version"], "0.8.4-fake", "version comes from the adapter that detected the checkout")

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
                self.assertEqual(t.split("/", 1)[0], svc.registry.tool(t).skill_id)
                self.assertIs(router.adapter_for(t).__class__, FakeAdapter)
        rows = {r["skill"]: r for r in svc.skills()}
        self.assertEqual(rows["silence_cleanup"]["packages"], ["ffmpeg-skill"])

    # Test D — declared future skills are never AVAILABLE, and no future package exists
    def test_future_skills_never_available(self):
        svc = make_service(self.tmp)
        rows = {r["skill"]: r for r in svc.skills()}
        for name in ("multi_source_sync", "caption_generation", "semantic_deletion"):
            self.assertEqual((rows[name]["status"], rows[name]["implemented"]), ("NOT_IMPLEMENTED", False))
        src = Path(__file__).resolve().parents[1] / "src" / "video_agent"
        future = ("media-analysis-skill", "transcription-skill", "subtitle-skill", "video-editing-skill", "audio-production-skill",
                  "motion-graphics-skill", "color-grading-skill", "thumbnail-skill", "qc-skill")
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
        self.assertEqual([p.skill_id for p in svc.registry.packages()], ["fake-skill", "ffmpeg-skill"])
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
        self.assertEqual([p.skill_id for p in make_service(self.tmp).registry.packages()], ["ffmpeg-skill"])

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
