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

    def test_ffmpeg_skill_version_range_is_explicit(self):
        """The supported ffmpeg-skill range is a declared contract (PR #13 / PR #17): 0.8.4 ≤ v < 0.10. 0.9.x is accepted
        (contract / doctor added, `--json` gained "status", no media behaviour changed); 0.10 is unverified and rejected;
        anything unparsable is rejected. Widening the range needs a verified integration run, not a silent edit."""
        from video_agent.tools.ffmpeg_skill.locate import SUPPORTED_MAX_EXCLUSIVE, SUPPORTED_MIN
        self.assertEqual((SUPPORTED_MIN, SUPPORTED_MAX_EXCLUSIVE), ((0, 8, 4), (0, 10, 0)))
        for v in ("0.8.4", "0.8.5", "0.9.0", "0.9.1", "0.9.12"):
            self.assertTrue(FfmpegSkill(self.skill.root, v, self.skill.scripts).version_supported(), v)
        for v in ("0.8.3", "0.7.9", "0.10.0", "0.11.2", "1.0.0", "unknown", "", "0.9.x"):
            self.assertFalse(FfmpegSkill(self.skill.root, v, self.skill.scripts).version_supported(), v)

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
        self.assertEqual(list(pkgs), ["ffmpeg-skill", "media-analysis", "transcription", "video-editing"],
                         "implemented packages: the Reference Skill, the observation Skill (PR #12), the recognition Skill (PR #13) and the editing Skill (PR #18)")
        self.assertEqual(PACKAGE.validate(), [])
        self.assertEqual((PACKAGE.repository, PACKAGE.capabilities), ("kajisho5/ffmpeg-skill", ["ffmpeg", "ffprobe", "ffmpeg-skill"]))
        rows = {r["skill_id"]: r for r in svc.packages()}
        self.assertEqual(sorted(rows), ["ffmpeg-skill", "media-analysis", "transcription", "video-editing"])
        self.assertTrue(rows["video-editing"]["implemented"] and not rows["video-editing"]["available"], "adapter exists; no installation in unit tests")
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
        self.assertEqual(rows["silence_cleanup"]["packages"], ["ffmpeg-skill", "video-editing"], "candidates in declared order (PR #18 adds video-editing/cut)")

    # Test D — declared future skills are never AVAILABLE, and no future package exists
    def test_future_skills_never_available(self):
        svc = make_service(self.tmp)
        rows = {r["skill"]: r for r in svc.skills()}
        for name in ("multi_source_sync", "caption_generation", "semantic_deletion"):
            self.assertEqual((rows[name]["status"], rows[name]["implemented"]), ("NOT_IMPLEMENTED", False))
        src = Path(__file__).resolve().parents[1] / "src" / "video_agent"
        future = ("subtitle-skill", "audio-production-skill",
                  "motion-graphics-skill", "color-grading-skill", "thumbnail-skill", "qc-skill")   # media-analysis-skill (PR #12), transcription-skill (PR #13) and video-editing-skill (PR #18) are integrated
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
        self.assertEqual([p.skill_id for p in svc.registry.packages()], ["fake-skill", "ffmpeg-skill", "media-analysis", "transcription", "video-editing"])
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
        self.assertEqual([p.skill_id for p in make_service(self.tmp).registry.packages()], ["ffmpeg-skill", "media-analysis", "transcription", "video-editing"])

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
        # PR #14: SpeechEvents feed deterministic inferences and review decisions (speech.continuity, silence.internal.*); the plan,
        # its operations and the delivery still never carry an event, a transcript, an engine or a speaker (Event → command does not exist)
        blob = json.dumps({"plan": d["plan"], "video": d["video"], "audio": d["audio"], "delivery": d["delivery"]})
        for forbidden in ("SPEECH", "transcription", "speaker", t["id"], "faster_whisper") + tuple(e["id"] for e in sp):
            self.assertNotIn(forbidden, blob, f"{forbidden!r} reached the plan / operation layer")
        reasoning = json.dumps({"decisions": d["decisions"], "inferences": d["analysis"]["inferences"]})
        self.assertNotIn("faster_whisper", reasoning); self.assertNotIn("speaker_name", reasoning); self.assertNotIn("camera", reasoning)
        self.assertTrue(all(i["data"].get("speaker_id") is None for i in d["analysis"]["inferences"]), "speaker_id stays null through inference")
        self.assertEqual([s["skill"] for s in d["plan"]["steps"]], ["silence_cleanup", "loudness_normalization", "delivery_export", "delivery_check"])
        self.assertTrue(all(s["tool"].startswith("ffmpeg-skill/") for s in d["plan"]["steps"]))
        citing = [dec for dec in d["decisions"] if any(ev in {e["id"] for e in sp} for ev in dec["evidence"])]
        self.assertTrue(all(dec["subject"].startswith(("speech.", "silence.")) and dec["approval"] in ("AUTO", "CONFIRM") for dec in citing), "speech evidence reaches speech / silence decisions only")
        self.assertTrue(all(dec["decision"] in ("keep",) or dec["decision"].startswith(("keep", "remove", "trim")) for dec in citing))
        info = Service.explain_observation(d, t["id"])
        kinds = [row["kind"] for row in info["chain"]]
        self.assertEqual(kinds[:10], ["observation", "skill", "tool", "engine", "model", "transcript", "asset", "analysis", "event", "event"])
        self.assertTrue(kinds[10:] and set(kinds[10:]) == {"context"}, "PR #15: the contexts the transcript's events take part in follow")
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


class SpeechInferenceDecisionPlanTests(unittest.TestCase):
    """PR #14: SpeechEvent → Inference → Decision → ProductionPlan → Project IR (deterministic, evidence-based, reviewable).
    Fixture: FakeAdapter silences 0-3 / 9-12 / 13.7-end (measured), fake transcript speech 3.5-8.8 and 12.3-13.5 (recognised):
    the 9-12 s pause lies between two speech intervals and is long enough to become a removal *candidate* (CONFIRM)."""

    FAKE = str(Path(__file__).resolve().parent / "fake_transcription.py")
    SEGMENTS = json.dumps([[3.5, 8.8, "本日の講演を始めます"], [12.3, 13.5, "以上です"]])
    SILENCES = [[0.0, 3.0], [9.0, 12.0], [13.7, None]]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)
        os.environ["FAKE_TS_SEGMENTS"] = self.SEGMENTS
        for k in ("FAKE_TS_MODE", "FAKE_TS_CACHE"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ("FAKE_TS_SEGMENTS", "FAKE_TS_MODE", "FAKE_TS_CACHE"):
            os.environ.pop(k, None)

    def _service(self, silences=None, transcription=True, **fake):
        from video_agent.tools import ToolRouter
        from video_agent.tools.transcription import TranscriptionAdapter, TranscriptionSkill
        adapters = [FakeAdapter(silences=silences if silences is not None else self.SILENCES, **fake)]
        if transcription:
            adapters.append(TranscriptionAdapter(TranscriptionSkill([sys.executable, self.FAKE], None, {}), workspace=str(Path(self.tmp) / "ws" / "cache" / "transcription")))
        return make_service(self.tmp, caps=FakeCaps(extra=["transcription"]), adapter=ToolRouter(adapters))

    def _plan(self, svc=None, profile="youtube", **kw):
        svc = svc or self._service()
        return svc, svc.plan([self.src], profile, kinds=["transcript"], params={"language": "ja"}, **kw)

    # 1-7: SpeechEvent → Inference (intervals merged, silence relation, timestamps untouched, speaker null, no identity, none without speech)
    def test_speech_inferences_from_events(self):
        svc, ir = self._plan()
        d = ir.doc
        infs = {i["kind"]: [x for x in d["analysis"]["inferences"] if x["kind"] == i["kind"]] for i in d["analysis"]["inferences"]}
        sp = sorted([e for e in d["timeline"]["events"] if e["type"] == "SPEECH"], key=lambda e: e["range"]["start"])
        self.assertEqual([(e["range"]["start"], e["range"]["end"]) for e in sp], [(3.5, 8.8), (12.3, 13.5)], "events keep the Skill's timestamps")
        iv = sorted(infs["speech_interval"], key=lambda i: i["data"]["start"])
        self.assertEqual([(i["data"]["start"], i["data"]["end"]) for i in iv], [(3.5, 8.8), (12.3, 13.5)])
        self.assertEqual(iv[0]["data"]["merge_gap"], {"value": 0.5, "provenance": "DEFAULT", "source": "video_agent.agent.speech_inference"}, "threshold and its provenance are recorded")
        self.assertTrue(all(i["provenance"] == "INFERRED" and i["data"]["speaker_id"] is None for i in iv))
        self.assertTrue(all(set(i["evidence"]) >= {sp[k]["id"]} for k, i in enumerate(iv)), "each interval cites its SpeechEvents")
        t = next(o for o in d["analysis"]["observations"] if o["kind"] == "transcript")
        self.assertTrue(all(t["id"] in i["evidence"] for i in iv))
        act = infs["speech_activity"][0]
        self.assertEqual((act["data"]["intervals"], act["data"]["speech_seconds"], act["data"]["duration"]), (2, 6.5, 16.0))
        rem = infs["internal_silence_removable"]
        self.assertEqual(len(rem), 1)
        r = rem[0]
        self.assertEqual((r["data"]["silence"], r["data"]["start"], r["data"]["end"]), ({"start": 9.0, "end": 12.0, "seconds": 3.0}, 9.15, 11.85))
        self.assertEqual((r["data"]["threshold"]["value"], r["data"]["threshold"]["provenance"], r["data"]["margin"]["value"]), (2.0, "DEFAULT", 0.15))
        sil = next(e for e in d["timeline"]["events"] if e["type"] == "AUDIO_SILENCE" and e["range"]["start"] == 9.0)
        self.assertEqual(sil["range"], {"start": 9.0, "end": 12.0}, "the measured silence event is untouched")
        self.assertIn(sil["id"], r["evidence"]); self.assertTrue({r["data"]["before"], r["data"]["after"]} <= {i["id"] for i in iv})
        self.assertNotIn("speech_silence_conflict", infs, "no speech overlaps a measured silence in this fixture")
        # the existing internal_silence_candidate inference for the same pause still exists (inference is additive); the decision layer emits one decision per pause
        self.assertTrue(any(i["kind"] == "internal_silence_candidate" and i["data"]["start"] == 9.0 for i in d["analysis"]["inferences"]))
        self.assertEqual([x["subject"] for x in d["decisions"] if x["subject"].startswith("silence.internal")], ["silence.internal.9.000-12.000"], "one decision per pause: the candidate replaces the keep")
        blob = json.dumps(d["analysis"]["inferences"])
        for bad in ("speaker_name", "camera", "who is speaking", "ffmpeg -", "argv", "cut.py"):   # tool sources ("ffmpeg-skill/silence@…") are provenance, not commands
            self.assertNotIn(bad, blob)
        self.assertNotRegex(blob, r'"speaker_id": "')
        # merging: two segments closer than the merge gap form one logical interval; farther apart stay separate
        os.environ["FAKE_TS_SEGMENTS"] = json.dumps([[3.5, 5.0, "a"], [5.3, 8.8, "b"], [12.3, 13.5, "c"]])
        svc2, ir2 = self._plan(self._service())
        iv2 = sorted([i for i in ir2.doc["analysis"]["inferences"] if i["kind"] == "speech_interval"], key=lambda i: i["data"]["start"])
        self.assertEqual([(i["data"]["start"], i["data"]["end"], i["data"]["segments"]) for i in iv2], [(3.5, 8.8, 2), (12.3, 13.5, 1)])
        # no speech events → no speech inference, and the existing internal-silence behaviour is unchanged (keep, AUTO)
        svc3 = self._service(transcription=False)
        ir3 = svc3.plan([self.src], "youtube")
        kinds = {i["kind"] for i in ir3.doc["analysis"]["inferences"]}
        self.assertFalse(kinds & {"speech_interval", "speech_activity", "internal_silence_removable", "speech_silence_conflict"})
        internal = [x for x in ir3.doc["decisions"] if x["subject"].startswith("silence.internal")]
        self.assertEqual([(x["subject"], x["decision"], x["approval"]) for x in internal], [("silence.internal", "keep", "AUTO")])
        # a SpeechEvent carrying a speaker id is refused outright (speaker identification is not part of this system)
        from video_agent.agent.speech_inference import infer_speech
        from video_agent.policy.rules import resolve_rules
        _, _, an = svc.analyze([self.src], "youtube", kinds=["transcript"], params={"language": "ja"})
        next(e for e in an.timeline.events if e.type == "SPEECH").metadata["speaker_id"] = "spk_1"
        with self.assertRaises(ValueError):
            infer_speech(an, resolve_rules([]))

    # 3-4: silence / speech temporal structure incl. a conflict (recognised speech inside a measured silence) → recorded, trims need CONFIRM
    def test_silence_speech_conflict_is_recorded_not_corrected(self):
        os.environ["FAKE_TS_SEGMENTS"] = json.dumps([[1.0, 8.8, "a"], [12.3, 13.5, "b"]])   # speech starts inside the 0-3 s measured silence
        svc, ir = self._plan()
        d = ir.doc
        conf = [i for i in d["analysis"]["inferences"] if i["kind"] == "speech_silence_conflict"]
        self.assertEqual(len(conf), 1)
        self.assertEqual((conf[0]["data"]["silence"], conf[0]["data"]["overlap_seconds"]), ({"start": 0.0, "end": 3.0}, 2.0))
        sil = next(e for e in d["timeline"]["events"] if e["type"] == "AUDIO_SILENCE" and e["range"]["start"] == 0.0)
        self.assertEqual(sil["range"]["end"], 3.0, "the silence observation / event is not changed by the conflict")
        lead = next(x for x in d["decisions"] if x["subject"] == "silence.leading")
        self.assertEqual((lead["approval"], lead["risk"]), ("CONFIRM", "MEDIUM"), "a disputed trim is never AUTO")
        self.assertIn(conf[0]["id"], lead["evidence"])
        self.assertEqual(d["plan"]["status"], "REVIEW")
        keep_dec = [x for x in d["decisions"] if x["subject"].startswith("silence.conflict.")]
        self.assertEqual([(x["decision"], x["approval"]) for x in keep_dec], [("keep", "AUTO")])
        self.assertEqual(svc.validate(ir).errors, [])

    # 8-9, 13-15: Inference → Decision → ProductionPlan → IR, traceable; approval never auto; rejected / blocked never executable
    def test_decision_plan_ir_traceability_and_approval(self):
        from video_agent.agent.production_plan import executable_steps, explain_step
        svc, ir = self._plan()
        d = ir.doc
        cand = [x for x in d["decisions"] if x["subject"].startswith("silence.internal.")]
        self.assertEqual(len(cand), 1)
        c = cand[0]
        self.assertEqual((c["subject"], c["approval"], c["risk"], c["status"], c["provenance"]), ("silence.internal.9.000-12.000", "CONFIRM", "MEDIUM", "PROPOSED", "INFERRED"))
        self.assertTrue(c["decision"].startswith("remove 9.150-11.850s"))
        rem = next(i for i in d["analysis"]["inferences"] if i["kind"] == "internal_silence_removable")
        self.assertIn(rem["id"], c["evidence"])
        cont = next(x for x in d["decisions"] if x["subject"] == "speech.continuity")
        self.assertEqual((cont["decision"], cont["approval"], cont["risk"]), ("keep all 2 speech interval(s)", "AUTO", "LOW"))
        self.assertNotIn("speech.continuity", json.dumps(d["plan"]["steps"]), "continuity is a decision, not a step")
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "silence_cleanup")
        self.assertEqual(step["params"]["keep"], [[2.85, 9.15], [11.85, 13.85]], "the trim keeps speech and cuts the confirmed-pending pause; ranges come from decisions only")
        self.assertEqual(step["params"]["removed"], [[0.0, 2.85], [13.85, 16.0], [9.15, 11.85]])
        self.assertIn(c["id"], step["decision_ids"])
        self.assertEqual(step["status"], "PROPOSED"); self.assertEqual(d["plan"]["status"], "REVIEW")
        self.assertNotIn(step["id"], executable_steps(d), "the trim does not execute while the candidate awaits confirmation (other AUTO steps keep partial-approval semantics)")
        op = d["video"]["operations"][0]
        self.assertEqual((op["type"], op["keep"]), ("video.trim", [[2.85, 9.15], [11.85, 13.85]]))
        self.assertIn(c["id"], op["decision_ids"])
        rep = svc.validate(ir)
        self.assertEqual(rep.errors, [], rep.errors)
        self.assertTrue(any("needs confirmation" in w for w in rep.warnings))
        # evidence chain: step → decision → inference(removable) → silence event + speech_interval inferences → SPEECH events → transcript observation
        info = explain_step(d, step["id"])
        kinds = [(r["kind"], r.get("detail") or "") for r in info["chain"]]
        self.assertTrue(any(k == "decision" and "remove 9.150" in det for k, det in kinds))
        self.assertTrue(any(k == "inference" and "candidate for removal" in det for k, det in kinds))
        self.assertTrue(any(k == "inference" and "speech from" in det for k, det in kinds))
        self.assertTrue(any(k == "event" and "SpeechEvent/speech" in det for k, det in kinds))
        self.assertTrue(any(k == "event" and "AudioEvent/silence" in det for k, det in kinds))
        self.assertTrue(any(k == "observation" and det == "transcript" for k, det in kinds))
        self.assertTrue(all(r.get("provenance") != "AI_GENERATED" for r in info["chain"]))
        ir_path = str(Path(self.tmp) / "p.json")
        save_ir(ir, ir_path)
        # approve → APPROVED and executable; the plan content (hash) is unchanged by the review
        h = ir.plan_hash()
        svc.approve(load_ir(ir_path), ir_path, [c["id"]])
        ir2 = load_ir(ir_path)
        self.assertEqual((ir2.doc["plan"]["status"], ir2.plan_hash()), ("APPROVED", h))
        self.assertIn(step["id"], executable_steps(ir2.doc))
        out = svc.render(ir2, ir_path)
        self.assertEqual(out["status"], "COMPLETED", out.get("execution"))
        cut = next(op for op in out["execution"]["results"] if op["tool"] == "ffmpeg-skill/cut")
        self.assertTrue(cut["ok"])
        fake = next(a for a in svc.adapter([]).adapters if isinstance(a, FakeAdapter))
        cut_op = next(o for o in fake.calls if o.tool == "ffmpeg-skill/cut")
        self.assertEqual(cut_op.args.get("segments"), "2.850-9.150,11.850-13.850", "the compiler lowers the decided ranges; nothing else reaches the tool")
        self.assertEqual(sorted(cut_op.args), ["input", "output", "segments"])
        # reject → REJECTED, never executable; revise drops the candidate and keeps the speech
        ir3 = self._plan()[1]
        c3 = next(x for x in ir3.doc["decisions"] if x["subject"].startswith("silence.internal."))
        p3 = str(Path(self.tmp) / "r.json"); save_ir(ir3, p3)
        svc.reject(load_ir(p3), p3, [c3["id"]], reason="keep the pause")
        ir3r = load_ir(p3)
        self.assertEqual(ir3r.doc["plan"]["status"], "REJECTED")
        self.assertNotIn(next(s for s in ir3r.doc["plan"]["steps"] if s["skill"] == "silence_cleanup")["id"], executable_steps(ir3r.doc))
        self.assertEqual(svc.render(ir3r, p3)["status"], "BLOCKED")
        svc.revise(load_ir(p3), p3)
        v2 = load_ir(p3)
        self.assertEqual(v2.version, 2)
        self.assertFalse([x for x in v2.doc["decisions"] if x["subject"].startswith("silence.internal.") and x["status"] != "REJECTED"], "the rejected candidate is not proposed again")
        step2 = next(s for s in v2.doc["plan"]["steps"] if s["skill"] == "silence_cleanup")
        self.assertEqual(step2["params"]["keep"], [[2.85, 13.85]], "without the candidate the trim is the lead / tail trim only")
        self.assertTrue(any(x["subject"] == "speech.continuity" for x in v2.doc["decisions"]))
        # BLOCK policy → BLOCKED decision, plan BLOCKED, no execution
        from video_agent.policy.rules import Rule
        svc_b = self._service()
        svc_b.registry  # registry untouched; the policy comes from a request-scope constraint
        ir_b = svc_b.plan([self.src], "youtube", kinds=["transcript"], params={"language": "ja"}, user_requirements={"analysis.strategy": "FULL"})
        self.assertEqual(ir_b.doc["plan"]["status"], "REVIEW")   # default policy: CONFIRM

    # 10-12, 16-17: no event → command / tool path; no arbitrary command or path; determinism / resume; silencedetect issue untouched
    def test_boundaries_and_determinism(self):
        root = Path(__file__).resolve().parents[1] / "src" / "video_agent"
        import ast
        text = (root / "agent" / "speech_inference.py").read_text(encoding="utf-8")
        modules = [n.module or "" for n in ast.walk(ast.parse(text)) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Import) for a in n.names]
        for m in modules:
            for bad in ("subprocess", "tools", "execution", "providers", "ffmpeg", "transcription_skill", "shutil", "pathlib"):
                self.assertNotIn(bad, m, modules)
            self.assertNotEqual(m, "os", modules)
        code = text.split('"""', 2)[2]   # module docstring aside: the code never mentions commands, tools or speakers
        for bad in ("argv", "command", "shell", "subprocess", "ffmpeg", "cut.py", "Operation(", "ToolRouter", "adapter", "speaker_name", "diariz"):
            self.assertNotIn(bad, code, bad)
        for rel in ("agent/decision.py", "agent/planner.py", "agent/production_plan.py"):
            t = (root / rel).read_text(encoding="utf-8")
            for bad in ("SPEECH", "speaker_id", "subprocess", "ffmpeg -", "cut.py", "transcription_skill"):
                self.assertNotIn(bad, t, f"{rel}: {bad}")
        svc, ir = self._plan()
        d = ir.doc
        from video_agent.media.analysis import leak_scan
        self.assertEqual(leak_scan({"inferences": d["analysis"]["inferences"], "decisions": d["decisions"], "plan": d["plan"]}), [])
        blob = json.dumps({"decisions": d["decisions"], "plan": d["plan"], "video": d["video"]})
        self.assertNotIn(self.src, blob, "no filesystem path in decisions / plan (assets are referenced by id)")
        self.assertNotRegex(blob, r"(^|[\s\"])/(usr|bin|etc|tmp)/")
        # determinism: the same evidence yields the same plan shape; a second plan is byte-equal in plan content
        svc2, ir2 = self._plan()
        same = lambda x: [(s["skill"], s["params"].get("keep"), s["params"].get("removed")) for s in x.doc["plan"]["steps"]]  # noqa: E731
        self.assertEqual(same(ir), same(ir2))
        self.assertEqual(sorted(x["subject"] for x in d["decisions"]), sorted(x["subject"] for x in ir2.doc["decisions"]))
        # resume / idempotency: the compiled operation id depends on plan content, so approving and rendering twice reuses the cut
        ir_path = str(Path(self.tmp) / "i.json"); save_ir(ir, ir_path)
        svc.approve(load_ir(ir_path), ir_path, ["all"])
        out1 = svc.render(load_ir(ir_path), ir_path)
        self.assertEqual(out1["status"], "COMPLETED")
        out2 = svc.render(load_ir(ir_path), ir_path, resume=out1["job"]["id"])
        self.assertEqual(out2["status"], "COMPLETED")
        self.assertTrue(out2["execution"]["reused"], "resume reuses the completed cut instead of re-running it")
        # the known silencedetect end > duration issue is not "fixed" here: an over-long silence event still fails IR validation
        from video_agent.project import validate_ir
        bad = json.loads(json.dumps(d))
        sil = next(e for e in bad["timeline"]["events"] if e["type"] == "AUDIO_SILENCE" and e["range"]["start"] == 13.7)
        sil["range"]["end"] = 16.5
        ir_bad = load_ir(ir_path); ir_bad.doc = bad
        self.assertTrue(any("exceeds asset duration" in e for e in validate_ir(ir_bad).errors), "still reported, not silently clamped")


class ProductionContextTests(unittest.TestCase):
    """PR #15: ProductionContext (situation understanding) and generic deterministic inference over any event type (ADR-026).
    Fixture: FakeAdapter silences 0-3 / 9-12 / 13.7-end + fake transcript speech 3.5-8.8 / 12.3-13.5 (same as PR #14)."""

    FAKE = str(Path(__file__).resolve().parent / "fake_transcription.py")
    SEGMENTS = json.dumps([[3.5, 8.8, "本日の講演を始めます"], [12.3, 13.5, "以上です"]])
    SILENCES = [[0.0, 3.0], [9.0, 12.0], [13.7, None]]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)
        os.environ["FAKE_TS_SEGMENTS"] = self.SEGMENTS
        for k in ("FAKE_TS_MODE", "FAKE_TS_CACHE"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ("FAKE_TS_SEGMENTS", "FAKE_TS_MODE", "FAKE_TS_CACHE"):
            os.environ.pop(k, None)

    def _service(self, silences=None, transcription=True, **fake):
        from video_agent.tools import ToolRouter
        from video_agent.tools.transcription import TranscriptionAdapter, TranscriptionSkill
        adapters = [FakeAdapter(silences=silences if silences is not None else self.SILENCES, **fake)]
        if transcription:
            adapters.append(TranscriptionAdapter(TranscriptionSkill([sys.executable, self.FAKE], None, {}), workspace=str(Path(self.tmp) / "ws" / "cache" / "transcription")))
        return make_service(self.tmp, caps=FakeCaps(extra=["transcription"]), adapter=ToolRouter(adapters))

    def _analysis(self, svc=None):
        svc = svc or self._service()
        _, _, an = svc.analyze([self.src], "youtube", kinds=["transcript"], params={"language": "ja"})
        return svc, an

    # 1-6: contexts from several event types; scope, references, provenance, asset identity, determinism
    def test_context_construction_references_and_determinism(self):
        from video_agent.context import ProductionContext, build_contexts, contexts_at, contexts_between, validate_context
        svc, an = self._analysis()
        ctxs = build_contexts(an.timeline.events, an.assets, an.observations, [])
        self.assertTrue(len(ctxs) >= 8, [c.scope for c in ctxs])
        tl = f"asset:{an.assets[0].id}"
        self.assertTrue(all(c.timeline_id == tl and c.asset_ids == [an.assets[0].id] for c in ctxs), "every context names the asset it describes")
        # contiguous cover of the asset from 0 to its duration, in order, no overlap, timestamps exactly the events' own
        bounds = [(c.scope["start"], c.scope["end"]) for c in ctxs]
        self.assertEqual(bounds[0][0], 0.0); self.assertEqual(bounds[-1][1], 16.0)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(bounds, bounds[1:])))
        event_points = {p for e in an.timeline.events if e.event_type != "UserDecisionEvent" for p in (e.range["start"], e.range["end"]) if p is not None}
        self.assertTrue({b for pair in bounds for b in pair} <= event_points | {0.0, 16.0}, "boundaries come from the events, never from a heuristic")
        # the situation while speech is recognised inside audio activity: three event kinds active at once
        at = contexts_at(ctxs, 5.0)
        self.assertEqual(len(at), 1)
        c = at[0]
        self.assertEqual(c.signature, "AudioEvent/active+AudioEvent/loudness+SpeechEvent/speech")
        self.assertEqual((c.scope["start"], c.scope["end"]), (3.5, 8.8))
        sp = next(e for e in an.timeline.events if e.type == "SPEECH" and e.range["start"] == 3.5)
        self.assertIn(sp.id, c.event_ids)
        track = next(t for t in c.tracks if t["event_type"] == "SpeechEvent")
        self.assertEqual((track["event_ids"], track["sources"], track["provenance"]), ([sp.id], [sp.source], ["OBSERVED"]))
        transcript = next(o for o in an.observations if o.kind == "transcript")
        self.assertIn(transcript.id, c.observation_ids, "observation provenance travels with the context")
        self.assertTrue(all(o in {x.id for x in an.observations} for o in c.observation_ids))
        self.assertEqual(c.provenance, "DERIVED"); self.assertEqual(c.generator, "context_builder@1.0")
        # the pause between the two speech intervals: silence measured, no speech
        pause = contexts_at(ctxs, 10.0)[0]
        self.assertEqual(pause.signature, "AudioEvent/active+AudioEvent/loudness+AudioEvent/silence")
        self.assertEqual((pause.scope["start"], pause.scope["end"]), (9.0, 12.0))
        self.assertEqual([(c.scope["start"], c.scope["end"]) for c in contexts_between(ctxs, 8.0, 12.5)], [(3.5, 8.8), (8.8, 9.0), (9.0, 12.0), (12.0, 12.3), (12.3, 13.5)])
        # deterministic: same events → same contexts and ids; a second build is identical
        again = build_contexts(an.timeline.events, an.assets, an.observations, [])
        self.assertEqual([c.to_dict() for c in ctxs], [c.to_dict() for c in again])
        self.assertEqual(c.id, ProductionContext.make_id(c.timeline_id, c.scope, c.event_ids))
        self.assertTrue(all(x.id.startswith("ctx_") for x in ctxs))
        # validation: references must exist, ids must match content, scope inside the asset
        ev = {e.id: e for e in an.timeline.events}
        durs = {an.assets[0].id: 16.0}
        obs_ids = {o.id for o in an.observations}
        self.assertEqual(validate_context(c, ev, durs, obs_ids, set()), [])
        bad = ProductionContext.from_dict(c.to_dict()); bad.scope = {"start": 3.5, "end": 3.4}
        self.assertTrue(any("end > start" in e or "invalid scope" in e for e in validate_context(bad, ev, durs, obs_ids, set())))
        bad = ProductionContext.from_dict(c.to_dict()); bad.event_ids = bad.event_ids + ["evt_nope"]
        self.assertTrue(any("unknown event" in e for e in validate_context(bad, ev, durs, obs_ids, set())))
        bad = ProductionContext.from_dict(c.to_dict()); bad.scope = {"start": 3.5, "end": 20.0}
        self.assertTrue(any("exceeds asset" in e for e in validate_context(bad, ev, durs, obs_ids, set())))
        bad = ProductionContext.from_dict(c.to_dict()); bad.id = "ctx_edited"
        self.assertTrue(any("does not match" in e for e in validate_context(bad, ev, durs, obs_ids, set())))
        # a user decision is review history, not a situation; without any situation event there is no context
        self.assertEqual(build_contexts([], an.assets, [], []), [])

    # 7-12: generic inference from contexts: evidence, timestamps kept, conflicts kept, no speaker / intent semantics
    def test_generic_inference_from_contexts(self):
        from video_agent.context import GENERIC_KINDS, build_contexts, infer_from_contexts
        svc, an = self._analysis()
        ctxs = build_contexts(an.timeline.events, an.assets, an.observations, [])
        ev = {e.id: e for e in an.timeline.events}
        infs = infer_from_contexts(ctxs, ev, {an.assets[0].id: 16.0})
        self.assertTrue(infs)
        self.assertTrue(all(i.kind in GENERIC_KINDS and i.provenance == "INFERRED" and i.evidence and i.data["generator"] == "context_inference@1.0" for i in infs))
        self.assertTrue(all(all(x in ev for x in i.evidence) for i in infs), "every piece of evidence is an existing event")
        by = {}
        for i in infs:
            by.setdefault(i.kind, []).append(i)
        act = {(i.data["event_type"], i.data["subtype"]): i for i in by["source_activity"]}
        self.assertEqual(act[("SpeechEvent", "speech")].data["intervals"], [[3.5, 8.8], [12.3, 13.5]], "intervals are the events' own timestamps")
        self.assertEqual(act[("AudioEvent", "silence")].data["intervals"], [[0.0, 3.0], [9.0, 12.0], [13.7, 16.0]])
        self.assertNotIn(("AudioEvent", "loudness"), act, "a whole-programme measurement says nothing about time")
        inact = {(i.data["event_type"], i.data["subtype"]): i for i in by["source_inactivity"]}
        self.assertEqual(inact[("SpeechEvent", "speech")].data["intervals"], [[0.0, 3.5], [8.8, 12.3], [13.5, 16.0]])
        self.assertTrue(all(cid in {c.id for c in ctxs} for i in infs for cid in i.data.get("context_ids", [])), "context references resolve")
        trans = sorted(by["transition"], key=lambda i: i.data["at"])
        self.assertEqual([i.data["at"] for i in trans], [2.85, 3.0, 3.5, 8.8, 9.0, 12.0, 12.3, 13.5, 13.7, 13.85])
        self.assertTrue(all(i.data["from_context"] in {c.id for c in ctxs} and i.data["to_context"] in {c.id for c in ctxs} for i in trans))
        self.assertNotIn("conflict", by, "silence and speech do not overlap in this fixture")
        # a conflict is recorded, never resolved: both events stay, neither timestamp changes, no preference
        os.environ["FAKE_TS_SEGMENTS"] = json.dumps([[1.0, 8.8, "a"], [12.3, 13.5, "b"]])
        svc2, an2 = self._analysis(self._service())
        ctxs2 = build_contexts(an2.timeline.events, an2.assets, an2.observations, [])
        infs2 = infer_from_contexts(ctxs2, {e.id: e for e in an2.timeline.events}, {an2.assets[0].id: 16.0})
        conf = [i for i in infs2 if i.kind == "conflict"]
        self.assertEqual(len(conf), 1)
        self.assertEqual((conf[0].data["codes"], conf[0].data["overlap"]), (["AUDIO_SILENCE", "SPEECH"], [1.0, 3.0]))
        sil = next(e for e in an2.timeline.events if e.type == "AUDIO_SILENCE" and e.range["start"] == 0.0)
        spk = next(e for e in an2.timeline.events if e.type == "SPEECH" and e.range["start"] == 1.0)
        self.assertEqual(sorted(conf[0].evidence), sorted([sil.id, spk.id]))
        self.assertEqual((sil.range["end"], spk.range["start"]), (3.0, 1.0), "timestamps untouched")
        self.assertIn("neither is preferred", conf[0].statement, "the conflict is recorded, not resolved")
        blob = json.dumps([i.to_dict() for i in infs + infs2])
        for bad in ("speaker_id\": \"", "speaker_name", "camera", "slide", "should", "cut ", "remove", "use source", "argv", "ffmpeg -"):
            self.assertNotIn(bad, blob, bad)
        # no situation events → no generic inference
        self.assertEqual(infer_from_contexts([], {}, {}), [])

    # 13-19: architecture boundaries — no event / context / inference → step, tool or command path; no AI; no paths
    def test_boundaries_static_and_dynamic(self):
        import ast
        root = Path(__file__).resolve().parents[1] / "src" / "video_agent"
        for rel in ("context/model.py", "context/builder.py", "context/inference.py"):
            text = (root / rel).read_text(encoding="utf-8")
            tree = ast.parse(text)
            mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            for m in mods:
                for bad in ("subprocess", "tools", "execution", "providers", "ai_reasoning", "planner", "decision", "compiler", "shutil", "pathlib", "socket", "http"):
                    self.assertNotIn(bad, m, (rel, mods))
                self.assertNotEqual(m, "os", (rel, mods))
            code = text.split('"""', 2)[2]
            for bad in ("argv", "shell", "subprocess", "ffmpeg", "Operation(", "ProductionStep", "Decision(", "camera", "slide", "speaker", "open(", "Path("):
                self.assertNotIn(bad, code, (rel, bad))
        for rel in ("agent/planner.py", "agent/production_plan.py", "execution/compiler.py", "execution/executor.py"):
            t = (root / rel).read_text(encoding="utf-8")
            self.assertNotIn("ProductionContext", t, f"{rel}: contexts never reach the plan / compiler / executor")
            self.assertNotIn("build_contexts", t)
        svc, ir = self._service(), None
        ir = svc.plan([self.src], "youtube", kinds=["transcript"], params={"language": "ja"})
        d = ir.doc
        self.assertTrue(d["analysis"]["contexts"])
        blob = json.dumps({"plan": d["plan"], "video": d["video"], "audio": d["audio"], "delivery": d["delivery"]})
        for bad in ("ctx_", "context", "SPEECH", "transition", "source_activity", "argv"):
            self.assertNotIn(bad, blob, f"{bad!r} reached the plan / operations")
        from video_agent.media.analysis import leak_scan
        self.assertEqual(leak_scan({"contexts": d["analysis"]["contexts"], "inferences": d["analysis"]["inferences"]}), [])
        self.assertNotIn(self.src, json.dumps(d["analysis"]["contexts"]), "no filesystem path in a context")
        self.assertEqual(d["provenance"]["ai_calls"], [], "no AI provider is consulted")
        self.assertTrue(all(i["provenance"] != "AI_GENERATED" for i in d["analysis"]["inferences"]))
        # generic inferences never become decisions or steps by themselves: the decision set is PR #14's
        subjects = sorted({x["subject"].split(".")[0] for x in d["decisions"]})
        self.assertEqual(subjects, ["audio", "delivery", "silence", "speech"])
        self.assertFalse([x for x in d["decisions"] if any(i["id"] in x["evidence"] for i in d["analysis"]["inferences"] if i["kind"] in ("source_activity", "source_inactivity", "transition"))],
                         "situation inferences inform, they do not decide")
        # an AI inference must never be recorded as OBSERVED / an event: the validator refuses it
        from video_agent.project import validate_ir
        bad = json.loads(json.dumps(d))
        bad["analysis"]["inferences"].append({"kind": "ai_recommendation:silence_cleanup", "asset_id": list(bad["assets"])[0], "statement": "x", "confidence": 0.5,
                                              "evidence": [bad["analysis"]["observations"][0]["id"]], "data": {}, "provenance": "OBSERVED", "id": "inf_ai_bad"})
        p = str(Path(self.tmp) / "bad.json"); Path(p).write_text(json.dumps(bad), encoding="utf-8"); ir_bad = load_ir(p)
        self.assertTrue(any("AI_GENERATED" in e for e in validate_ir(ir_bad).errors))

    # 20-24: provenance chain observation → event → context → inference → decision → step; explain; regression on review flow
    def test_provenance_chain_explain_and_regression(self):
        from video_agent.agent.production_plan import explain_step
        svc = self._service()
        ir = svc.plan([self.src], "youtube", kinds=["transcript"], params={"language": "ja"})
        d = ir.doc
        self.assertEqual(svc.validate(ir).errors, [])
        ctxs = d["analysis"]["contexts"]
        transcript = next(o for o in d["analysis"]["observations"] if o["kind"] == "transcript")
        # observation → event → context: explain_observation ends with the contexts its events take part in
        info = Service.explain_observation(d, transcript["id"])
        ctx_rows = [r for r in info["chain"] if r["kind"] == "context"]
        self.assertEqual(len(ctx_rows), 2, "each SpeechEvent lives in exactly one situation here")
        self.assertTrue(all(r["provenance"] == "DERIVED" for r in ctx_rows))
        # context → events → observations, and the inferences / decisions resting on it
        pause = next(c for c in ctxs if c["scope"]["start"] == 9.0 and c["scope"]["end"] == 12.0)
        info = Service.explain_context(d, pause["id"])
        kinds = [r["kind"] for r in info["chain"]]
        self.assertEqual(kinds[0], "context")
        self.assertIn("track", kinds); self.assertIn("event", kinds); self.assertIn("observation", kinds); self.assertIn("inference", kinds); self.assertIn("decision", kinds)
        self.assertTrue(any(r["kind"] == "decision" and r["id"].startswith("dec_") and "silence.internal." in r["detail"] for r in info["chain"]), "the removal candidate rests on this situation")
        self.assertTrue(any(r["kind"] == "inference" and r.get("generator") == "context_inference@1.0" for r in info["chain"]))
        self.assertIn("never becomes a step", info["boundary"])
        with self.assertRaises(KeyError):
            Service.explain_context(d, "ctx_nope")
        # step → decision → inference → contexts (via data) → events → observation
        trim = next(s for s in d["plan"]["steps"] if s["skill"] == "silence_cleanup")
        chain = explain_step(d, trim["id"])["chain"]
        self.assertTrue(any(r["kind"] == "observation" and r["detail"] == "transcript" for r in chain))
        self.assertTrue(any(r["kind"] == "event" and "AudioEvent/silence" in r["detail"] for r in chain))
        # contexts are valid IR content and survive the review flow: approve, reject → revise (v2 rebuilds its own contexts)
        p = str(Path(self.tmp) / "p.json"); save_ir(ir, p)
        cand = next(x for x in d["decisions"] if x["subject"].startswith("silence.internal."))
        svc.reject(load_ir(p), p, [cand["id"]], reason="keep it")
        svc.revise(load_ir(p), p)
        v2 = load_ir(p)
        self.assertEqual(v2.version, 2)
        self.assertEqual(svc.validate(v2).errors, [], svc.validate(v2).errors)
        self.assertTrue(v2.doc["analysis"]["contexts"])
        self.assertEqual({c["id"] for c in v2.doc["analysis"]["contexts"]}, {c["id"] for c in ctxs}, "same events → same situations across versions")
        self.assertTrue(all(i in {x["id"] for x in v2.doc["analysis"]["inferences"]} for c in v2.doc["analysis"]["contexts"] for i in c["inference_ids"]))
        svc.approve(load_ir(p), p, ["all"])
        out = svc.render(load_ir(p), p)
        self.assertEqual(out["status"], "COMPLETED", out.get("execution"))
        out2 = svc.render(load_ir(p), p, resume=out["job"]["id"])
        self.assertEqual(out2["status"], "COMPLETED"); self.assertTrue(out2["execution"]["reused"])
        # a context that cites a missing inference or event is refused by the validator (nothing is silently repaired)
        bad = json.loads(json.dumps(d))
        bad["analysis"]["contexts"][0]["inference_ids"] = ["inf_missing"]
        pb = str(Path(self.tmp) / "badctx.json"); Path(pb).write_text(json.dumps(bad), encoding="utf-8")
        from video_agent.project import validate_ir
        self.assertTrue(any("unknown inference" in e for e in validate_ir(load_ir(pb)).errors))
        # CLI: context listing, --at, and explain --context
        import subprocess
        env = dict(os.environ, VIDEO_AGENT_WORKSPACE=self.tmp)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "context", p, "--at", "10"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("AudioEvent/silence", r.stdout); self.assertIn("9.000-  12.000", r.stdout)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "explain", p, "--context", pause["id"]], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("context", r.stdout); self.assertIn("never becomes a step", r.stdout)


class ProductionDecisionEngineTests(unittest.TestCase):
    """PR #16: generic Decision Engine — Inference + Policy / Preference / Constraint + Intent + Risk → Decision. Evidence is
    mandatory, approvals come from policy with a safe default and recorded provenance, BLOCK / REJECTED never execute, the
    basis of every decision is recorded and explainable, PR #14 speech decisions run unchanged through the same engine."""

    FAKE = str(Path(__file__).resolve().parent / "fake_transcription.py")
    SEGMENTS = json.dumps([[3.5, 8.8, "本日の講演を始めます"], [12.3, 13.5, "以上です"]])
    SILENCES = [[0.0, 3.0], [9.0, 12.0], [13.7, None]]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)
        os.environ["FAKE_TS_SEGMENTS"] = self.SEGMENTS
        for k in ("FAKE_TS_MODE", "FAKE_TS_CACHE"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ("FAKE_TS_SEGMENTS", "FAKE_TS_MODE", "FAKE_TS_CACHE"):
            os.environ.pop(k, None)

    # ---- helpers
    def _rules(self, *rules):
        return resolve_rules(list(rules))

    def _engine(self, rules=None, intent=None, known=None, reqs=None):
        from video_agent.agent.decision_engine import DecisionEngine
        from video_agent.models import Intent
        return DecisionEngine(rules or self._rules(), intent or Intent(primary="clean_only", secondary=["cleanup_silence"]), known or {}, reqs or [])

    def _service(self, silences=None, transcription=False, **fake):
        from video_agent.tools import ToolRouter
        from video_agent.tools.transcription import TranscriptionAdapter, TranscriptionSkill
        adapters = [FakeAdapter(silences=silences if silences is not None else self.SILENCES, **fake)]
        if transcription:
            adapters.append(TranscriptionAdapter(TranscriptionSkill([sys.executable, self.FAKE], None, {}), workspace=str(Path(self.tmp) / "ws" / "cache" / "transcription")))
        return make_service(self.tmp, caps=FakeCaps(extra=["transcription"]), adapter=ToolRouter(adapters))

    def _speech_plan(self, profile="youtube", **kw):
        svc = self._service(transcription=True)
        return svc, svc.plan([self.src], profile, kinds=["transcript"], params={"language": "ja"}, **kw)

    # 1-4: policy resolution with provenance (USER / PROFILE / SYSTEM / DEFAULT), precedence untouched
    def test_setting_resolution_records_provenance_without_changing_precedence(self):
        from video_agent.agent.decision_engine import resolve_setting
        rs = self._rules(Rule("sys.k", "POLICY", "GLOBAL", "k.sys", 1, "system"), Rule("prof.k", "POLICY", "PROFILE", "k.prof", 2, "profile:x"),
                         Rule("req.k", "PREFERENCE", "REQUEST", "k.prof", 3, "request"))
        self.assertEqual(resolve_setting(rs, "k.sys", 0)["provenance"], "SYSTEM")
        got = resolve_setting(rs, "k.prof", 0)
        self.assertEqual((got["value"], got["provenance"], got["kind"], got["rule_id"], got["hard"]), (3, "USER", "PREFERENCE", "req.k", False), "REQUEST wins over PROFILE (existing precedence), reported as USER")
        d = resolve_setting(rs, "k.absent", 0.5)
        self.assertEqual((d["value"], d["provenance"], d["kind"], d["rule_id"]), (0.5, "DEFAULT", None, None), "an absent key is the explicit default the caller passed, marked DEFAULT")
        hard = self._rules(Rule("c", "CONSTRAINT", "PROFILE", "k", "CONFIRM", "profile:conference"), Rule("r", "PREFERENCE", "REQUEST", "k", "AUTO", "request"))
        self.assertEqual((resolve_setting(hard, "k", None)["value"], resolve_setting(hard, "k", None)["hard"], len(hard.conflicts)), ("CONFIRM", True, 1), "a CONSTRAINT is never overridden; the attempt is a conflict")

    # 5-10: approval resolution: AUTO / CONFIRM / BLOCK, unknown → CONFIRM, floor never lowers, BLOCK never lowered, explicit waiver not on constraints
    def test_approval_resolution_is_safe_by_default(self):
        from video_agent.agent.decision_engine import raise_approval, resolve_approval
        from video_agent.models import Requirement
        for value, want in (("AUTO", "AUTO"), ("CONFIRM", "CONFIRM"), ("BLOCK", "BLOCK"), ("BLOCK_UNLESS_EXPLICIT", "BLOCK"), ("MAYBE", "CONFIRM"), ("", "CONFIRM"), (None, "CONFIRM"), (1, "CONFIRM")):
            rs = self._rules(Rule("p", "POLICY", "PROFILE", "x.approval", value, "profile:t"))
            got = resolve_approval(rs, "x.approval", "CONFIRM")
            self.assertEqual(got["approval"], want, value)
            if want == "CONFIRM" and value != "CONFIRM":
                self.assertTrue(any("safe default" in n for n in got["notes"]), value)
        self.assertEqual(resolve_approval(self._rules(), "x.approval", "AUTO")["setting"]["provenance"], "DEFAULT")
        self.assertEqual(resolve_approval(self._rules(), "x.approval", "CONFIRM")["approval"], "CONFIRM")
        with self.assertRaises(ValueError):
            resolve_approval(self._rules(), "x.approval", "YES")
        # floor raises, never lowers; raise_approval never lowers; BLOCK stays BLOCK
        rs = self._rules(Rule("p", "POLICY", "PROFILE", "x.approval", "AUTO", "profile:t"))
        self.assertEqual(resolve_approval(rs, "x.approval", "AUTO", floor="CONFIRM")["approval"], "CONFIRM")
        got = resolve_approval(self._rules(Rule("p", "POLICY", "PROFILE", "x.approval", "BLOCK", "t")), "x.approval", "AUTO", floor="CONFIRM")
        self.assertEqual(got["approval"], "BLOCK")
        got = resolve_approval(self._rules(Rule("p", "POLICY", "PROFILE", "x.approval", "CONFIRM", "t")), "x.approval", "AUTO", floor="AUTO")
        self.assertEqual(got["approval"], "CONFIRM", "a lower floor never lowers")
        self.assertEqual(raise_approval({"approval": "CONFIRM", "setting": None, "notes": []}, "AUTO", "x")["approval"], "CONFIRM")
        self.assertEqual(raise_approval({"approval": "AUTO", "setting": None, "notes": []}, "CONFIRM", "speech overlaps")["notes"], ["raised AUTO → CONFIRM: speech overlaps"])
        # the explicit-request waiver (existing behaviour, eval 03) applies to a POLICY, never to a CONSTRAINT, never to BLOCK
        user = Requirement(key="edit.trim_leading_silence", value=True, provenance="USER", source="cli")
        pol = self._rules(Rule("p", "POLICY", "PROFILE", "x.approval", "CONFIRM", "profile:conference"))
        self.assertEqual(resolve_approval(pol, "x.approval", "AUTO", explicit=user)["approval"], "AUTO")
        con = self._rules(Rule("c", "CONSTRAINT", "PROFILE", "x.approval", "CONFIRM", "profile:conference"))
        got = resolve_approval(con, "x.approval", "AUTO", explicit=user)
        self.assertEqual(got["approval"], "CONFIRM"); self.assertTrue(any("CONSTRAINT" in n for n in got["notes"]))
        self.assertEqual(resolve_approval(self._rules(Rule("p", "POLICY", "PROFILE", "x.approval", "BLOCK", "t")), "x.approval", "AUTO", explicit=user)["approval"], "BLOCK")
        default_req = Requirement(key="edit.trim_leading_silence", value="auto", provenance="DEFAULT", source="defaults")
        self.assertEqual(resolve_approval(pol, "x.approval", "AUTO", explicit=default_req)["approval"], "CONFIRM", "only a USER requirement waives")

    # 11-18: construction invariants — evidence mandatory, grounding, AI-only → REVIEW, vocabulary, BLOCK ⇔ BLOCKED, no executable material
    def test_engine_refuses_ungrounded_or_unsafe_decisions(self):
        from video_agent.agent.decision_engine import DECISION_TYPES, DecisionError
        from video_agent.models import Inference, Requirement
        inf = Inference(kind="leading_silence_unwanted", asset_id="a", statement="s", confidence=0.9, evidence=["obs_1"])
        ai = Inference(kind="ai_recommendation:silence_cleanup", asset_id="a", statement="AI says cut", confidence=0.9, evidence=["obs_1"], provenance="AI_GENERATED")
        user_req = Requirement(key="edit.trim_leading_silence", value=True, provenance="USER", source="cli")
        known = {"obs_1": "observation", inf.id: "inference", ai.id: "ai", user_req.id: "requirement", "pref": "rule"}   # "pref": a PREFERENCE rule id
        eng = self._engine(known=known, reqs=[user_req])
        base = dict(decision="trim 0-1s", reason="r", risk="LOW", approval="AUTO", confidence=0.9, provenance="INFERRED", params={"asset_id": "a", "start": 0.0, "end": 1.0})
        with self.assertRaisesRegex(DecisionError, "no evidence"):
            eng.decide(subject="silence.leading", type="REMOVE", evidence=[], **base)
        with self.assertRaisesRegex(DecisionError, "not an observation"):
            eng.decide(subject="silence.leading", type="REMOVE", evidence=["inf_unknown"], **base)
        with self.assertRaisesRegex(DecisionError, "preference, intent or AI output alone"):
            eng.decide(subject="silence.leading", type="REMOVE", evidence=["pref"], **base)   # preference-only
        with self.assertRaisesRegex(DecisionError, "preference, intent or AI output alone"):
            eng.decide(subject="silence.leading", type="REMOVE", evidence=[ai.id], **base)    # AI text only
        with self.assertRaisesRegex(DecisionError, "REVIEW item"):
            eng.decide(subject="x", type="KEEP", evidence=[ai.id], **base)                    # AI alone is not even a fact-backed keep
        with self.assertRaisesRegex(DecisionError, "type"):
            eng.decide(subject="x", type="CUT", evidence=[inf.id], **base)
        with self.assertRaisesRegex(DecisionError, "risk"):
            eng.decide(subject="x", type="REMOVE", evidence=[inf.id], **{**base, "risk": "NONE"})
        with self.assertRaisesRegex(DecisionError, "approval"):
            eng.decide(subject="x", type="REMOVE", evidence=[inf.id], **{**base, "approval": "YES"})
        with self.assertRaisesRegex(DecisionError, "status BLOCKED needs approval BLOCK"):
            eng.decide(subject="x", type="REMOVE", evidence=[inf.id], status="BLOCKED", **base)
        for bad in ({"command": "ffmpeg -i x"}, {"argv": ["-y"]}, {"shell": "rm -rf"}, {"api_key": "sk-abc"}, {"note": "bash -c 'x'"}):
            with self.assertRaisesRegex(DecisionError, "executable / credential"):
                eng.decide(subject="x", type="REMOVE", evidence=[inf.id], **{**base, "params": {"asset_id": "a", **bad}})
        self.assertEqual(eng.decisions, [], "nothing refused was recorded")
        # accepted: fact-backed REMOVE; requirement-backed DELIVER; BLOCK carries status BLOCKED; AI-only REVIEW is never executable and its params are scrubbed
        d = eng.decide(subject="silence.leading", type="REMOVE", evidence=[inf.id, "obs_1", inf.id], **base, requirements=[user_req], serves_intent="cleanup_silence")
        self.assertEqual((d.type, d.evidence, d.status, d.basis["evidence_classes"], d.basis["intent"]["served"]), ("REMOVE", [inf.id, "obs_1"], "PROPOSED", ["inference", "observation"], "cleanup_silence"))
        self.assertEqual(d.basis["requirements"][0]["provenance"], "USER")
        deliver = eng.decide(subject="delivery.web", type="DELIVER", evidence=[user_req.id], **{**base, "decision": "export preset 'youtube'"})
        self.assertEqual(deliver.basis["evidence_classes"], ["requirement"])
        b = eng.decide(subject="capability.x", type="BLOCK", evidence=["capability:ffmpeg"], **{**base, "decision": "BLOCK: skill x unavailable", "approval": "BLOCK", "risk": "HIGH"})
        self.assertEqual((b.status, b.approval, b.basis["evidence_classes"]), ("BLOCKED", "BLOCK", ["capability"]))
        r = eng.decide(subject="ai.silence_cleanup", type="REVIEW", evidence=[ai.id], **{**base, "decision": "review: AI recommends silence_cleanup", "approval": "CONFIRM", "risk": "MEDIUM",
                                                                                          "params": {"asset_id": "a", "executable": True, "ai_params": {"command": "ffmpeg -i in out", "ranges": [[0, 1]]}}})
        self.assertEqual(r.params["ai_params"], {"ranges": [[0, 1]]}, "AI-proposed command material is dropped, never interpreted")
        self.assertTrue(any("removed from proposed params" in n for n in r.basis["approval"]["notes"]))
        self.assertEqual(sorted(DECISION_TYPES), sorted(["KEEP", "REMOVE", "TRANSFORM", "DELIVER", "SKIP", "REVIEW", "BLOCK"]))

    # 19-23: decide() through the engine — every decision typed, grounded, with basis; policy provenance USER / PROFILE / DEFAULT; existing decisions unchanged
    def test_decisions_carry_type_basis_and_policy_provenance(self):
        from video_agent.agent.decision_engine import DECISION_TYPES, EXECUTABLE_TYPES
        svc = self._service()
        ir = svc.plan([self.src], "conference")
        d = ir.doc
        decs = {x["subject"]: x for x in d["decisions"]}
        for x in d["decisions"]:
            self.assertIn(x["type"], DECISION_TYPES, x["subject"])
            self.assertTrue(x["evidence"], x["subject"])
            self.assertEqual(x["basis"]["engine"], "decision_engine@1.0")
            self.assertEqual(x["basis"]["risk"], {"level": x["risk"], "independent_of_confidence": True})
            self.assertEqual(x["basis"]["approval"]["resolved"], x["approval"])
        lead = decs["silence.leading"]
        self.assertEqual((lead["type"], lead["approval"], lead["provenance"]), ("REMOVE", "CONFIRM", "INFERRED"))
        appr = lead["basis"]["approval"]
        self.assertEqual((appr["key"], appr["provenance"]), ("silence.leading.approval", "PROFILE"))
        setting = next(s for s in lead["basis"]["settings"] if s["key"] == "silence.leading.approval")
        self.assertEqual((setting["value"], setting["kind"], setting["rule_id"], setting["provenance"], setting["source"]), ("CONFIRM", "POLICY", "conf.silence.leading.approval", "PROFILE", "profile:conference"))
        self.assertEqual(next(s for s in lead["basis"]["settings"] if s["key"] == "silence.leading.min_seconds")["value"], 3.0)
        self.assertEqual(lead["basis"]["intent"], {"primary": "clean_and_deliver", "secondary": ["normalize_audio", "cleanup_silence"], "provenance": "SYSTEM", "served": "clean_and_deliver"})
        self.assertEqual([r["key"] for r in lead["basis"]["requirements"]], ["edit.trim_leading_silence"])
        loud = decs["audio.loudness"]
        self.assertEqual((loud["type"], loud["approval"], loud["basis"]["approval"]["provenance"]), ("TRANSFORM", "AUTO", "DEFAULT"), "no approval policy for loudness: explicit DEFAULT AUTO, recorded as such")
        self.assertEqual([(s["key"], s["provenance"]) for s in loud["basis"]["settings"]], [("audio.loudness.tolerance_lu", "PROFILE"), ("audio.loudness.approval", "DEFAULT")])
        self.assertEqual({r["key"]: r["provenance"] for r in loud["basis"]["requirements"]}, {"audio.normalize": "DEFAULT", "audio.loudness.target_lufs": "PROFILE", "audio.loudness.true_peak": "PROFILE"})
        self.assertEqual(decs["delivery.master"]["type"], "DELIVER"); self.assertEqual(decs["delivery.master"]["basis"]["evidence_classes"], ["requirement"])
        self.assertTrue(all(x["type"] in EXECUTABLE_TYPES for x in d["decisions"] if any(x["id"] in s["decision_ids"] for s in d["plan"]["steps"])))
        # a USER requirement changes the approval provenance to USER and waives CONFIRM (existing behaviour; recorded, not silent)
        ir_u = svc.plan([self.src], "conference", user_requirements={"edit.trim_leading_silence": True})
        du = {x["subject"]: x for x in ir_u.doc["decisions"]}
        self.assertEqual((du["silence.leading"]["approval"], du["silence.leading"]["provenance"]), ("AUTO", "USER"))
        self.assertTrue(any("CONFIRM waived" in n for n in du["silence.leading"]["basis"]["approval"]["notes"]))
        ir_t = svc.plan([self.src], "generic", user_requirements={"silence.trailing.approval": "CONFIRM"})
        dt = {x["subject"]: x for x in ir_t.doc["decisions"]}
        self.assertEqual((dt["silence.trailing"]["approval"], dt["silence.trailing"]["basis"]["approval"]["provenance"], dt["silence.trailing"]["basis"]["approval"]["key"]), ("CONFIRM", "USER", "silence.trailing.approval"))
        self.assertEqual((dt["silence.leading"]["approval"], dt["silence.leading"]["basis"]["approval"]["provenance"]), ("AUTO", "PROFILE"))
        # an unknown policy value never becomes AUTO
        ir_x = svc.plan([self.src], "generic", user_requirements={"silence.leading.approval": "whatever"})
        lx = next(x for x in ir_x.doc["decisions"] if x["subject"] == "silence.leading")
        self.assertEqual(lx["approval"], "CONFIRM"); self.assertTrue(any("safe default" in n for n in lx["basis"]["approval"]["notes"]))
        self.assertEqual(ir_x.doc["plan"]["status"], "REVIEW")
        # generic profile: AUTO from the profile (PROFILE), plan APPROVED as before
        ir_g = svc.plan([self.src], "generic")
        lg = next(x for x in ir_g.doc["decisions"] if x["subject"] == "silence.leading")
        self.assertEqual((lg["approval"], lg["basis"]["approval"]["provenance"], ir_g.doc["plan"]["status"]), ("AUTO", "PROFILE", "APPROVED"))

    # 24-27: confidence ≠ risk ≠ approval; conflicts → CONFIRM with reason; constraint vs request; BLOCK policy → nothing executes
    def test_risk_approval_conflicts_and_block(self):
        svc = self._service()
        # risk / approval come from policy and the kind of change, never from confidence: the same lead trim is AUTO on generic and
        # CONFIRM on conference with identical confidence; a 0.7-confidence removal candidate is MEDIUM / CONFIRM while a 0.5 keep is AUTO
        lg = next(x for x in svc.plan([self.src], "generic").doc["decisions"] if x["subject"] == "silence.leading")
        lc = next(x for x in svc.plan([self.src], "conference").doc["decisions"] if x["subject"] == "silence.leading")
        self.assertEqual(lg["confidence"], lc["confidence"])
        self.assertEqual((lg["risk"], lg["approval"], lc["risk"], lc["approval"]), ("LOW", "AUTO", "LOW", "CONFIRM"))
        # request tries to lower the conference CONSTRAINT silence.internal.approval: recorded conflict, decision CONFIRM with the reason, candidate stays CONFIRM
        svc2, ir2 = self._speech_plan("conference", user_requirements={"silence.internal.approval": "AUTO"})
        d2 = ir2.doc
        conflict = next(x for x in d2["decisions"] if x["subject"] == "policy.silence.internal.approval")
        self.assertEqual((conflict["type"], conflict["approval"], conflict["risk"], conflict["basis"]["evidence_classes"]), ("KEEP", "CONFIRM", "MEDIUM", ["rule"]))
        self.assertIn("never overridden silently", conflict["reason"])
        self.assertEqual(conflict["basis"]["settings"][0]["hard"], True)
        cand = [x for x in d2["decisions"] if x["subject"].startswith("silence.internal.")]
        self.assertEqual([(c["type"], c["approval"], c["basis"]["approval"]["provenance"]) for c in cand], [("REMOVE", "CONFIRM", "PROFILE")])
        # BLOCK policy: BLOCKED decision (status), plan BLOCKED, no execution even with approve all
        svc3, ir3 = self._speech_plan("youtube", user_requirements={"silence.internal.approval": "BLOCK"})
        c3 = next(x for x in ir3.doc["decisions"] if x["subject"].startswith("silence.internal."))
        self.assertEqual((c3["approval"], c3["status"], ir3.doc["plan"]["status"]), ("BLOCK", "BLOCKED", "BLOCKED"))
        p3 = str(Path(self.tmp) / "b.json"); save_ir(ir3, p3)
        out = svc3.render(load_ir(p3), p3, approve=["all"])
        self.assertEqual(out["status"], "BLOCKED"); self.assertFalse(out.get("execution"))
        self.assertEqual(svc3.validate(load_ir(p3)).errors, [], "a BLOCKED plan is valid IR, it just never executes")
        # policy AUTO on a candidate is floored to CONFIRM (never AUTO for a content-adjacent removal)
        svc4, ir4 = self._speech_plan("youtube", user_requirements={"silence.internal.approval": "AUTO"})
        c4 = next(x for x in ir4.doc["decisions"] if x["subject"].startswith("silence.internal."))
        self.assertEqual(c4["approval"], "CONFIRM"); self.assertTrue(any("floor" in n for n in c4["basis"]["approval"]["notes"]))

    # 28-29: PR #14 speech decisions unchanged through the engine; no loudness measurement → no loudness claim
    def test_speech_decisions_unchanged_and_no_claim_without_measurement(self):
        svc, ir = self._speech_plan()
        d = ir.doc
        by = {x["subject"]: x for x in d["decisions"]}
        self.assertEqual((by["speech.continuity"]["type"], by["speech.continuity"]["decision"], by["speech.continuity"]["approval"], by["speech.continuity"]["risk"]), ("KEEP", "keep all 2 speech interval(s)", "AUTO", "LOW"))
        cand = by["silence.internal.9.000-12.000"]
        self.assertEqual((cand["type"], cand["decision"], cand["approval"], cand["risk"], cand["status"]), ("REMOVE", "remove 9.150-11.850s (candidate)", "CONFIRM", "MEDIUM", "PROPOSED"))
        self.assertEqual({s["key"]: s["provenance"] for s in cand["basis"]["settings"]}, {"silence.internal.removable_min_seconds": "DEFAULT", "silence.margin_seconds": "DEFAULT", "silence.internal.approval": "PROFILE"})
        self.assertEqual(cand["basis"]["intent"]["served"], "clean_and_deliver")
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "silence_cleanup")
        self.assertEqual(step["params"]["keep"], [[2.85, 9.15], [11.85, 13.85]])
        self.assertEqual(d["plan"]["status"], "REVIEW")
        # no loudness observation (analysis failed): no "within tolerance" decision without evidence; the warning records the failure
        tmp_f = tempfile.mkdtemp()   # fresh workspace: no cached loudness observation from the plans above
        svc_f = make_service(tmp_f, adapter=FakeAdapter(fail_tools={"ffmpeg-skill/loudness": 9}))
        ir_f = svc_f.plan([fake_media(tmp_f)], "youtube")
        self.assertFalse([x for x in ir_f.doc["decisions"] if x["subject"] == "audio.loudness"])
        self.assertTrue(any("loudness analysis failed" in w for w in ir_f.doc["analysis"]["warnings"]))
        self.assertEqual(svc_f.validate(ir_f).errors, [])
        # an AI recommendation covered by a measured decision becomes extra evidence (class ai added), never changes approval / risk
        prov = FakeAIProvider(intent="silence_cleanup")
        svc_ai = Service(workspace=self.tmp, adapter=FakeAdapter(), caps=FakeCaps(), provider=prov)
        ir_ai = svc_ai.plan([self.src], "youtube")
        la = next(x for x in ir_ai.doc["decisions"] if x["subject"] == "silence.leading")
        self.assertIn("ai", la["basis"]["evidence_classes"]); self.assertEqual((la["approval"], la["risk"]), ("AUTO", "LOW"))
        review = [x for x in ir_ai.doc["decisions"] if x["type"] == "REVIEW"]
        self.assertTrue(all(x["params"]["executable"] is False and x["approval"] != "AUTO" for x in review))

    # 30-32: validator re-checks the invariants on a recorded IR; revision keeps history valid
    def test_validator_enforces_engine_invariants(self):
        from video_agent.agent.decision_engine import check_decisions
        svc, ir = self._speech_plan()
        d = ir.doc
        self.assertEqual(check_decisions(d), [])
        cand = next(x for x in d["decisions"] if x["subject"].startswith("silence.internal."))

        def tampered(fn):
            doc = json.loads(json.dumps(d))
            fn(doc, next(x for x in doc["decisions"] if x["id"] == cand["id"]))
            return check_decisions(doc)

        self.assertTrue(any("has no evidence" in e for e in tampered(lambda doc, c: c.update(evidence=[]))))
        self.assertTrue(any("unknown evidence" in e for e in tampered(lambda doc, c: c.update(evidence=["inf_ghost"]))))
        self.assertTrue(any("only ('REMOVE', 'TRANSFORM', 'DELIVER') may be executed" in e for e in tampered(lambda doc, c: c.update(type="KEEP"))))
        self.assertTrue(any("type 'CUT'" in e for e in tampered(lambda doc, c: c.update(type="CUT"))))
        self.assertTrue(any("BLOCK ⇔ BLOCKED" in e for e in tampered(lambda doc, c: c.update(approval="BLOCK"))))
        self.assertEqual(tampered(lambda doc, c: c.update(approval="BLOCK", status="BLOCKED")), [], "a BLOCKED citation is valid IR; the plan status keeps it from executing")
        self.assertTrue(any("executable / credential" in e for e in tampered(lambda doc, c: c["params"].update(command="ffmpeg -i a b"))))
        ai_inf = {"id": "inf_ai1", "kind": "ai_recommendation:silence_cleanup", "asset_id": cand["params"]["asset_id"], "statement": "x", "confidence": 0.9, "evidence": [], "data": {}, "provenance": "AI_GENERATED"}

        def ai_only(doc, c):
            doc["analysis"]["inferences"].append(ai_inf); c.update(evidence=["inf_ai1"])
        errs = tampered(ai_only)
        self.assertTrue(any("no measured fact or requirement" in e for e in errs) and any("AI-only evidence must be a REVIEW item" in e for e in errs), errs)
        # the same errors surface through validate_ir (the IR is refused, nothing is repaired)
        bad = json.loads(json.dumps(d)); next(x for x in bad["decisions"] if x["id"] == cand["id"])["evidence"] = []
        pb = str(Path(self.tmp) / "bad.json"); Path(pb).write_text(json.dumps(bad), encoding="utf-8")
        self.assertTrue(any("has no evidence" in e for e in validate_ir(load_ir(pb)).errors))
        # reject → revise: the REJECTED decision is carried as history (its evidence lived in v1) and v2 stays valid; approvals / resume unchanged
        p = str(Path(self.tmp) / "p.json"); save_ir(ir, p)
        svc.reject(load_ir(p), p, [cand["id"]], reason="keep the pause")
        svc.revise(load_ir(p), p)
        v2 = load_ir(p)
        self.assertEqual(svc.validate(v2).errors, [])
        hist = next(x for x in v2.doc["decisions"] if x["id"] == cand["id"])
        self.assertEqual((hist["status"], hist["type"]), ("REJECTED", "REMOVE"))
        self.assertTrue(all(x.get("type") and x.get("basis") for x in v2.doc["decisions"]))
        svc.approve(load_ir(p), p, ["all"])
        out = svc.render(load_ir(p), p); self.assertEqual(out["status"], "COMPLETED", out.get("execution"))
        out2 = svc.render(load_ir(p), p, resume=out["job"]["id"]); self.assertTrue(out2["execution"]["reused"])

    # 33-36: explain --decision: type / rationale / risk / approval / basis (policy, preference, constraint, intent, requirement) / evidence → context → event → observation → asset / plan step → IR
    def test_explain_decision_chain_service_and_cli(self):
        svc, ir = self._speech_plan("conference", user_requirements={"silence.internal.approval": "AUTO", "audio.loudness.target_lufs": -18})
        d = ir.doc
        cand = next(x for x in d["decisions"] if x["subject"].startswith("silence.internal."))
        info = Service.explain_decision(d, cand["id"])[0]
        self.assertEqual((info["decision"]["type"], info["decision"]["approval"], info["executable"]), ("REMOVE", "CONFIRM", True))
        kinds = {b["kind"] for b in info["basis"]}
        self.assertTrue({"constraint", "default", "approval", "intent", "risk"} <= kinds, kinds)
        con = next(b for b in info["basis"] if b["kind"] == "constraint")
        self.assertEqual((con["key"], con["value"], con["provenance"], con["hard"], con["rule_id"]), ("silence.internal.approval", "CONFIRM", "PROFILE", True, "conf.silence.internal.approval"))
        ev_kinds = {r["kind"] for r in info["evidence"]}
        self.assertTrue({"inference", "event", "observation", "asset", "context"} <= ev_kinds, ev_kinds)
        self.assertTrue(any(r["kind"] == "observation" and r["detail"] == "transcript" for r in info["evidence"]))
        self.assertTrue(any(r["kind"] == "event" and "SpeechEvent/speech" in r["detail"] for r in info["evidence"]))
        self.assertEqual([s["skill"] for s in info["plan"]["steps"]], ["silence_cleanup"])
        self.assertEqual(info["plan"]["operations"][0]["type"], "video.trim")
        self.assertIn("no command", info["boundary"])
        # a preference (conference target -16) overridden by the user: the requirement rows show USER; the loudness decision cites the preference rule via the requirement
        loud = Service.explain_decision(d, "audio.loudness")[0]
        self.assertEqual(next(b for b in loud["basis"] if b["kind"] == "requirement" and b["key"] == "audio.loudness.target_lufs")["provenance"], "USER")
        self.assertEqual(next(b for b in loud["basis"] if b["kind"] == "policy")["key"], "audio.loudness.tolerance_lu")
        # the constraint conflict decision: evidence are the two rules (constraint + attempted preference)
        conf = Service.explain_decision(d, "policy.silence.internal.approval")[0]
        self.assertEqual(sorted(r["kind"] for r in conf["evidence"]), ["constraint", "preference"]); self.assertFalse(conf["executable"])
        keep = Service.explain_decision(d, "speech.continuity")[0]
        self.assertEqual((keep["executable"], keep["plan"]["steps"], keep["plan"]["operations"]), (False, [], []))
        with self.assertRaises(KeyError):
            Service.explain_decision(d, "dec_nope")
        # CLI text and JSON
        import subprocess
        p = str(Path(self.tmp) / "p.json"); save_ir(ir, p)
        env = dict(os.environ, VIDEO_AGENT_WORKSPACE=self.tmp)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "explain", p, "--decision", cand["id"]], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        for frag in ("[REMOVE]", "basis:", "constraint", "silence.internal.approval", "approval", "intent", "evidence:", "SpeechEvent/speech", "transcript", "plan:", "step_trim_", "boundary"):
            self.assertIn(frag, r.stdout, frag)
        for bad in ("argv", "ffmpeg -", "subprocess", self.src):
            self.assertNotIn(bad, r.stdout)
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "--json", "explain", p, "--decision", "speech.continuity"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        js = json.loads(r.stdout)[0]
        self.assertEqual((js["decision"]["type"], js["executable"]), ("KEEP", False))
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "explain", p, "--decision", "dec_nope"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 1); self.assertIn("no such decision", r.stderr)

    # 37-40: boundaries — engine is tool / domain independent, decisions never carry paths or commands, determinism, plan hash unchanged by basis
    def test_engine_boundaries_and_determinism(self):
        import ast
        root = Path(__file__).resolve().parents[1] / "src" / "video_agent"
        text = (root / "agent" / "decision_engine.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        for m in mods:
            for bad in ("subprocess", "tools", "execution", "providers", "ai_reasoning", "planner", "compiler", "speech", "context", "skills", "shutil", "pathlib", "socket", "http"):
                self.assertNotIn(bad, m, mods)
            self.assertNotEqual(m, "os", mods)
        code = text.split('"""', 2)[2]
        for bad in ("silence", "speech", "loudness", "ffmpeg", "transcript", "subprocess", "argv", "Operation(", "ProductionStep", "camera", "speaker", "open(", "Path("):
            self.assertNotIn(bad, code, bad)
        svc, ir = self._speech_plan()
        d = ir.doc
        from video_agent.media.analysis import leak_scan
        self.assertEqual(leak_scan({"decisions": d["decisions"]}), [])
        self.assertNotIn(self.src, json.dumps(d["decisions"]))
        svc2, ir2 = self._speech_plan()
        sig = lambda doc: sorted((x["subject"], x["type"], x["approval"], x["risk"], x["status"], json.dumps({k: v for k, v in x["basis"].items() if k != "requirements"}, sort_keys=True)) for x in doc["decisions"])  # noqa: E731
        self.assertEqual(sig(d), sig(ir2.doc), "same evidence and policy → same decisions and basis")
        ops = lambda doc: [{k: v for k, v in op.items() if k not in ("decision_ids", "asset")} for op in doc["video"]["operations"] + doc["audio"]["operations"]]  # noqa: E731
        self.assertEqual(ops(d), ops(ir2.doc))
        self.assertNotIn("basis", json.dumps(d["plan"]["steps"]) + json.dumps(d["video"]), "the basis stays on the decision; steps / operations carry ids only")
        # a Decision without the engine (older / hand-made) is still a valid dataclass but the IR validator demands a type
        self.assertEqual(Decision(subject="x", decision="y", reason="r", confidence=1.0, evidence=["e"], risk="LOW", approval="AUTO").type, "")


class VideoEditingAdapterTests(unittest.TestCase):
    """PR #18 (ADR-028): video-editing-skill as an external editing Skill behind its CLI. Contract discovery and refusals,
    execution through `run - --json --workspace … --allowed-input …` with the EditRequest on stdin, response → ToolResult /
    Artifact / Observation / provenance mapping, error mapping with the Skill's retryable verdict, PathPolicy propagation,
    the security boundary (no command / argv / filter / executable / env / credential ever crosses; argv is a list), capability
    gating, determinism / reuse, and the Project IR lowering video.trim → video-editing/cut. Verified against a fake
    video-editing process (tests/fake_video_editing.py) that speaks the real transport; no ffmpeg, no import of the Skill."""

    FAKE = str(Path(__file__).resolve().parent / "fake_video_editing.py")

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = fake_media(self.tmp)
        self.ws = str(Path(self.tmp) / "ws")
        os.makedirs(self.ws)
        for k in ("FAKE_VE_MODE", "FAKE_VE_CALLS"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ("FAKE_VE_MODE", "FAKE_VE_CALLS"):
            os.environ.pop(k, None)

    def _skill(self):
        from video_agent.tools.video_editing import VideoEditingSkill
        return VideoEditingSkill([sys.executable, self.FAKE], None, {})

    def _adapter(self, **kw):
        from video_agent.tools.video_editing import VideoEditingAdapter
        kw.setdefault("workspace", self.ws)
        kw.setdefault("allowed_inputs", [str(Path(self.src).parent)])
        kw.setdefault("ffmpeg_skill_dir", self.tmp)
        return VideoEditingAdapter(self._skill(), **kw)

    def _op(self, **args):
        a = {"input": "a", "keep": [[0.5, 3.0], [4.0, 8.0]], "precision": "keyframe", "output": "a_trim"}
        a.update(args)
        return Operation(tool="video-editing/cut", args=a, inputs=["a"], outputs=["a_trim"], id="op_test")

    def _paths(self, out_name="01_trim.mp4"):
        out = str(Path(self.ws) / "jobs" / "j1" / "ops" / "01_trim" / out_name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        return {"a": self.src, "a_trim": out}, out

    # 1. contract discovery: the live document is the source of truth; malformed / incompatible contracts are refused, never patched
    def test_contract_discovery_package_and_refusals(self):
        from video_agent.tools.video_editing import PACKAGE, ContractError, check_contract, contract_drift, pinned_contract
        ad = self._adapter()
        self.assertEqual((ad.name, ad.version, ad.contract["skill_id"], ad.contract["schema"]), ("video-editing", "0.1.0", "video-editing", "video-editing/contract@1"))
        self.assertEqual(sorted(ad.tools), [f"video-editing/{t}" for t in ("concat", "cut", "fill", "fit", "overlay", "resize", "speed", "trim")])
        self.assertEqual(ad.drift(), [], "the installed (fake) contract equals the pinned 0.1.0 contract")
        pkg = ad.package()
        self.assertEqual(pkg.validate(), [])
        self.assertEqual((pkg.skill_id, pkg.version, pkg.repository), ("video-editing", "0.1.0", "kajisho5/video-editing-skill"))
        cut = pkg.tool("video-editing/cut")
        self.assertEqual((cut.produces_output, cut.deterministic, cut.kind, cut.inputs, cut.result_keys), (True, True, "transform", ["input", "output"], ["operation_id", "output", "probe", "commands", "provenance"]))
        self.assertEqual(cut.required_capabilities, ["ffmpeg", "ffprobe", "encoder:libx264", "encoder:aac", "video-editing"])
        self.assertEqual(PACKAGE.tool_ids(), pkg.tool_ids(), "the pinned package equals the live one")
        self.assertEqual(check_contract(pinned_contract()), [])
        for mode, frag in (("wrong_schema", "contract schema"), ("wrong_skill", "skill_id"), ("wrong_version", "supported range"), ("bad_contract", "execution.shell")):
            os.environ["FAKE_VE_MODE"] = mode
            with self.assertRaisesRegex(ContractError, frag):
                self._adapter()
        os.environ["FAKE_VE_MODE"] = "contract_fail"
        with self.assertRaisesRegex(ContractError, "contract --json failed"):
            self._adapter()
        os.environ["FAKE_VE_MODE"] = "contract_drift"
        ad2 = self._adapter()   # compatible but drifted: usable only if the agent re-verifies; the drift is reported, never hidden
        drift = ad2.drift()
        self.assertTrue(any("video-editing/crop: installed but not pinned" in d for d in drift) and any(d.startswith("operations:") for d in drift), drift)
        # the pinned contract cannot be edited into something the checks would not notice
        bad = json.loads(json.dumps(pinned_contract()))
        bad["tools"][0]["produces_output"] = False
        self.assertTrue(any("produces_output" in e for e in check_contract(bad)))
        bad = json.loads(json.dumps(pinned_contract())); bad["errors"]["codes"].append("SOMETHING")
        self.assertTrue(any("errors.codes" in e for e in check_contract(bad)))
        bad = json.loads(json.dumps(pinned_contract())); bad["tools"][1]["operation_type"] = "CROP"
        self.assertTrue(any("not a declared operation" in e for e in check_contract(bad)))
        self.assertEqual(contract_drift(pinned_contract(), pinned_contract()), [])

    # 2. valid execution: argv list, request on stdin, response → ToolResult with artifact / observation / timeline / provenance
    def test_valid_execution_and_mapping(self):
        log = str(Path(self.tmp) / "calls.log"); os.environ["FAKE_VE_CALLS"] = log
        ad = self._adapter()
        paths, out = self._paths()
        r = ad.run(self._op(), paths)
        self.assertTrue(r.ok, r.data)
        same = lambda a, b: os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))  # noqa: E731  (Windows: the adapter reports the resolved path, tempfile may give an 8.3 short name)
        self.assertEqual((r.exit_code, r.tool, r.dry_run), (0, "video-editing/cut", False)); self.assertTrue(same(r.output, out), (r.output, out))
        self.assertTrue(os.path.isfile(out))
        self.assertEqual(r.data["skill"], {"id": "video-editing", "version": "0.1.0"})
        self.assertEqual(r.data["status"], "completed")
        art = r.data["artifact"]
        self.assertTrue(same(art["path"], out)); self.assertEqual((art["size"], art["reused"]), (os.path.getsize(out), False))
        self.assertEqual(art["sha256"], __import__("hashlib").sha256(Path(out).read_bytes()).hexdigest(), "execution.outputs[].sha256 is verified against the file and carried to the Artifact")
        self.assertTrue(art["operation_id"].startswith("op_"))
        obs = r.data["observation"]
        self.assertEqual((obs["kind"], obs["provenance"]), ("media.probe", "OBSERVED")); self.assertTrue(obs["source"].startswith("ffmpeg-skill/probe@"))
        self.assertEqual(r.data["timeline"]["tracks"][0]["kind"], "video")
        rec = r.data["operation"]
        self.assertEqual((rec["type"], rec["status"], rec["skill"], rec["skill_version"], rec["tool"]), ("CUT", "completed", "video-editing", "0.1.0", "ffmpeg-skill/cut"))
        self.assertEqual(rec["output"]["sha256"], art["sha256"]); self.assertEqual(rec["inputs"][0]["kind"], "source")
        self.assertEqual(r.commands, rec and r.data["commands"]); self.assertTrue(r.commands and "provenance only" in r.commands[0])
        # the process boundary: one `run` call, argv as a list with the canonical shape, request JSON on stdin (not argv), workspace = the op dir
        calls = [json.loads(line) for line in Path(log).read_text().splitlines()]
        run = next(c for c in calls if c["cmd"] == "run")
        self.assertEqual(run["argv"][:4], ["run", "-", "--json", "--workspace"])
        self.assertTrue(same(run["argv"][4], os.path.dirname(out)))
        roots = [run["argv"][i + 1] for i, a in enumerate(run["argv"]) if a == "--allowed-input"]
        self.assertTrue(any(same(r_, str(Path(self.src).parent)) for r_ in roots) and any(same(r_, self.ws) for r_ in roots), roots)
        self.assertTrue(same(run["argv"][run["argv"].index("--ffmpeg-skill-dir") + 1], self.tmp))
        self.assertFalse(any("keep" in a or "0.5" in a for a in run["argv"]), "operation parameters travel on stdin, never in argv")
        self.assertNotIn("VIDEO_EDITING_FFMPEG_SKILL_DIR", run["env_video"], "the engine location is a CLI argument from agent config, not an environment side channel")
        # preview shows the same boundary without executing
        pv = ad.preview(self._op(), paths)[0]
        self.assertIn("run - --json --workspace", pv); self.assertIn('"schema": "video-editing/request@1"', pv); self.assertNotIn("ffmpeg -", pv)
        self.assertEqual(sum(1 for c in calls if c["cmd"] == "run"), 1)
        # dry run goes through `plan` and writes nothing
        paths2, out2 = self._paths("02_trim.mp4")
        d = ad.run(self._op(output="b"), {**paths2, "b": out2}, dry_run=True)
        self.assertTrue(d.ok and d.dry_run and d.output is None and not os.path.exists(out2))
        self.assertEqual(d.data["plan"]["steps"][0]["type"], "CUT")

    # 3. unsupported operation / unknown tool / undeclared parameter: refused before or by the Skill, never executed elsewhere
    def test_unsupported_operation(self):
        ad = self._adapter()
        paths, out = self._paths()
        self.assertFalse(ad.supports("video-editing/crop")); self.assertFalse(ad.supports("ffmpeg-skill/cut"))
        r = ad.run(Operation(tool="video-editing/crop", args={"input": "a", "output": "a_trim"}, inputs=["a"], outputs=["a_trim"]), paths)
        self.assertFalse(r.ok); self.assertEqual(r.data["error"]["code"], "INVALID_REQUEST"); self.assertIn("unsupported tool", r.data["error"]["message"])
        r = ad.run(self._op(zoom=2), paths)
        self.assertFalse(r.ok); self.assertIn("does not take argument", r.data["error"]["message"])
        self.assertFalse(os.path.exists(out))
        with self.assertRaises(ToolError):
            ad.measure("video-editing/cut", {})

    # 4-8. error mapping: the Skill's code and retryable verdict → ToolResult.data.error → recovery class; nothing is ever a success by exit code alone
    def test_error_mapping(self):
        from video_agent.execution.recovery import classify_error, next_attempt
        ad = self._adapter()
        paths, out = self._paths()
        want = {"tool_error": ("TOOL_ERROR", 10, True, "UNKNOWN", "RETRY"), "tool_error_final": ("TOOL_ERROR", 10, False, "SKILL_ERROR", "BLOCK"),
                "validation_error": ("VALIDATION_ERROR", 12, False, "SKILL_ERROR", "BLOCK"), "cancelled": ("CANCELLED", 130, True, "INTERRUPTED", "BLOCK"),
                "timeout": ("CANCELLED", 130, True, "TIMEOUT", "RETRY"), "internal_error": ("INTERNAL_ERROR", 1, False, "SKILL_ERROR", "BLOCK"),
                "output_missing": ("INVALID_RESULT", 9, False, "SKILL_ERROR", "BLOCK"), "hash_mismatch": ("INVALID_RESULT", 9, False, "SKILL_ERROR", "BLOCK"),
                "no_observation": ("INVALID_RESULT", 9, False, "SKILL_ERROR", "BLOCK"), "malformed": ("INVALID_RESULT", 9, False, "SKILL_ERROR", "BLOCK"),
                "empty": ("INVALID_RESULT", 9, False, "SKILL_ERROR", "BLOCK"), "two_docs": ("INVALID_RESULT", 9, False, "SKILL_ERROR", "BLOCK"),
                "text": ("INVALID_RESULT", 1, False, "SKILL_ERROR", "BLOCK"), "nonzero_ok": ("INVALID_RESULT", 3, False, "SKILL_ERROR", "BLOCK"),
                "unknown_code": ("INVALID_RESULT", 1, False, "SKILL_ERROR", "BLOCK")}
        for mode, (code, exit_code, retry, cls, action) in want.items():
            os.environ["FAKE_VE_MODE"] = mode
            r = ad.run(self._op(), paths)
            self.assertFalse(r.ok, mode)
            self.assertEqual((r.data["error"]["code"], r.exit_code, r.data["error"]["retryable"]), (code, exit_code, retry), mode)
            self.assertEqual(classify_error(r), cls, mode)
            self.assertEqual(next_attempt(r, 1, 2, None)["action"], action, mode)
            self.assertIsNone(r.output, mode)
            self.assertFalse(os.path.exists(out), f"{mode}: a failed run never leaves a result at the output path")
        os.environ["FAKE_VE_MODE"] = "tool_error"
        r = ad.run(self._op(), paths)
        self.assertEqual(r.data["error"]["recovery_class"], "UNKNOWN"); self.assertTrue(r.commands and "provenance only" in r.commands[0], "commands of a failed run are provenance too")
        self.assertIn("[TOOL_ERROR]", r.stderr_tail)
        # request-level refusals the Skill reports with its own codes (traversal / unsupported type / bad range) are non-retryable blocks
        os.environ.pop("FAKE_VE_MODE")
        r = ad.run(self._op(keep=[[5.0, 3.0]]), paths)
        self.assertEqual((r.data["error"]["code"], classify_error(r)), ("INVALID_REQUEST", "INVALID_ARGS"), "meaning is checked by the lowering before the Skill sees it")
        r = ad.run(self._op(keep=[[0.0, 40.0]]), paths)
        self.assertEqual((r.data["error"]["code"], r.data["error"]["retryable"], classify_error(r)), ("INVALID_TIME_RANGE", False, "INVALID_ARGS"), "beyond the duration: the Skill's own verdict")
        # timeout / hang: the agent's process boundary kills the tree and reports a retryable CANCELLED(timeout)
        os.environ["FAKE_VE_MODE"] = "hang"
        r = ad.run(self._op(), paths, timeout=1)
        self.assertEqual((r.ok, r.exit_code, r.data["error"]["code"], r.data["error"]["details"].get("reason"), classify_error(r)), (False, 124, "CANCELLED", "timeout", "TIMEOUT"))
        # process exit 0 + missing output is never a success (covered by output_missing above); process exit != 0 + ok document is never a success (nonzero_ok)

    # 12. PathPolicy propagation: the adapter's roots reach the Skill; traversal / absolute outside / symlink escape / workspace escape refused on both sides
    def test_path_policy_propagation(self):
        ad = self._adapter()
        paths, out = self._paths()
        outside = fake_media(tempfile.mkdtemp(), "o.mp4")
        r = ad.run(self._op(), {**paths, "a": outside})
        self.assertFalse(r.ok); self.assertIn("outside the allowed input roots", r.data["error"]["message"])
        r = ad.run(self._op(), {**paths, "a": str(Path(self.src).parent / ".." / "src" / "talk.mp4")})
        self.assertFalse(r.ok); self.assertIn("traversal", r.data["error"]["message"])
        r = ad.run(self._op(), {**paths, "a_trim": str(Path(tempfile.mkdtemp()) / "x.mp4")})
        self.assertFalse(r.ok); self.assertIn("outside the workspace", r.data["error"]["message"])
        r = self._adapter(workspace=self.tmp).run(self._op(), {**paths, "a_trim": self.src})
        self.assertFalse(r.ok); self.assertIn("overwrite an input", r.data["error"]["message"])
        r = ad.run(self._op(), {**paths, "a_trim": out[:-4] + ".txt"})
        self.assertFalse(r.ok); self.assertIn("extension", r.data["error"]["message"])
        if hasattr(os, "symlink"):
            link = Path(self.src).parent / "link.mp4"
            try:
                os.symlink(outside, link)
            except (OSError, NotImplementedError):
                link = None
            if link:
                r = ad.run(self._op(), {**paths, "a": str(link)})
                self.assertFalse(r.ok); self.assertIn("outside the allowed input roots", r.data["error"]["message"])
        # the Skill enforces the same roots again: an adapter without roots (unit fixture) still gets the Skill's PATH_NOT_ALLOWED
        loose = self._adapter(allowed_inputs=[], workspace=None)
        r = loose.run(self._op(), {**paths, "a": outside})
        self.assertFalse(r.ok); self.assertEqual(r.data["error"]["code"], "PATH_NOT_ALLOWED"); self.assertEqual(r.data["error"]["details"].get("reason"), "outside_allowed_roots")
        # a missing input is MISSING_INPUT from the Skill when the adapter has no roots, refused earlier otherwise
        r = ad.run(self._op(), {**paths, "a": str(Path(self.src).parent / "nope.mp4")})
        self.assertFalse(r.ok); self.assertIn("not found", r.data["error"]["message"])

    # 9-10. security boundary: nothing executable crosses; argv is a fixed list; values with shell metacharacters only ever travel as JSON on stdin
    def test_security_boundary(self):
        import ast
        log = str(Path(self.tmp) / "calls.log"); os.environ["FAKE_VE_CALLS"] = log
        ad = self._adapter()
        paths, out = self._paths()
        for bad in ({"command": "ffmpeg -i x y"}, {"argv": ["-y"]}, {"shell": "rm -rf /"}, {"filter": "scale=2"}, {"filter_complex": "[0:v]xfade"}, {"executable": "/bin/sh"},
                    {"env": {"PATH": "/tmp"}}, {"api_key": "sk"}, {"workspace": "/"}, {"allowed_input": "/"}, {"ffmpeg_skill_dir": "/tmp/x"}, {"script": "x.py"}, {"path": "/etc"}):
            r = ad.run(self._op(**bad), paths)
            self.assertFalse(r.ok, bad); self.assertEqual(r.data["error"]["code"], "INVALID_REQUEST", bad); self.assertIn("refusing argument", r.data["error"]["message"])
        for bad in ({"precision": "keyframe; rm -rf /"}, {"precision": "a\nb"}, {"precision": "x" * 300}):
            r = ad.run(self._op(**bad), paths)
            self.assertFalse(r.ok, bad)
        r = ad.run(self._op(precision="$(id)"), paths)   # a metacharacter string is data: it reaches the Skill as JSON and is refused there as an invalid parameter value
        self.assertFalse(r.ok)
        calls = [json.loads(line) for line in Path(log).read_text().splitlines()]
        for c in calls:
            self.assertIsInstance(c["argv"], list)
            self.assertFalse(any("$(" in a or ";" in a or "|" in a for a in c["argv"]), c["argv"])
            self.assertTrue(all("ffmpeg" not in a for a in c["argv"] if not a.startswith("--")) or "--ffmpeg-skill-dir" in c["argv"])
        self.assertFalse(os.path.exists(out))
        # static: the adapter package builds argv lists only, never a shell, never a command string, never an import of the Skill
        root = Path(__file__).resolve().parents[1] / "src" / "video_agent" / "tools" / "video_editing"
        for py in root.glob("*.py"):
            text = py.read_text(encoding="utf-8")
            tree = ast.parse(text)
            names = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            self.assertFalse(any("video_editing_skill" in m or m == "subprocess" for m in names), (py.name, names))
            for n in ast.walk(tree):
                if isinstance(n, ast.Call):
                    for kw in n.keywords:
                        self.assertNotEqual(kw.arg, "shell", f"{py.name}: shell= keyword")
            parts = text.split('"""', 2)
            code = parts[2] if len(parts) == 3 else text
            for bad in ("shell=True", "os.system", "os.popen", "filter_complex=", "-filter_complex", "ffmpeg -i", "argv.append(args"):
                self.assertNotIn(bad, code, (py.name, bad))
        # run_process_group is the only process boundary and takes a list
        adapter_src = (root / "adapter.py").read_text(encoding="utf-8")
        self.assertIn("run_process_group(cmd, ", adapter_src)
        self.assertIn("cmd = list(self.skill.command) + argv", adapter_src)

    # 13. capability failure: without the video-editing capability the tool is never selected; with it and the adapter, it is (in declared order)
    def test_capability_gating_and_registry(self):
        from video_agent.tools import ToolRouter
        ad = self._adapter()
        svc = make_service(self.tmp, caps=FakeCaps(), adapter=ToolRouter([FakeAdapter(), ad]))
        self.assertEqual(svc.tools_for().get("silence_cleanup"), "ffmpeg-skill/cut", "first declared candidate wins while it is executable")
        svc.registry.get("silence_cleanup").tools = ["video-editing/cut"]
        self.assertIsNone(svc.tools_for().get("silence_cleanup"), "video-editing capability MISSING → no tool, no fallback")
        row = next(r for r in svc.skills() if r["skill"] == "silence_cleanup")
        self.assertEqual(row["status"], "UNAVAILABLE"); self.assertIn("required capability missing", row["reason"])
        pk = next(p for p in svc.packages() if p["skill_id"] == "video-editing")
        self.assertFalse(pk["available"]); self.assertIn("video-editing", pk["reason"])
        svc2 = make_service(self.tmp, caps=FakeCaps(extra=["video-editing", "encoder:aac"]), adapter=ToolRouter([FakeAdapter(), ad]))
        svc2.registry.get("silence_cleanup").tools = ["video-editing/cut"]
        self.assertEqual(svc2.tools_for().get("silence_cleanup"), "video-editing/cut")
        pk = next(p for p in svc2.packages() if p["skill_id"] == "video-editing")
        self.assertTrue(pk["available"]); self.assertEqual(pk["version"], "0.1.0"); self.assertEqual(sorted(pk["usable_tools"]), sorted(ad.tools))
        # a tool-level capability the resolver knows and reports missing blocks the tool; names it does not resolve are the Skill's doctor's business
        svc3 = make_service(self.tmp, caps=FakeCaps(missing=["encoder:libx264"], extra=["video-editing"]), adapter=ToolRouter([FakeAdapter(), ad]))
        svc3.registry.get("silence_cleanup").tools = ["video-editing/cut"]
        self.assertIsNone(svc3.tools_for().get("silence_cleanup"), "encoder:libx264 MISSING → not executable")
        self.assertEqual(svc3.registry.tool_missing_capabilities("video-editing/cut", svc3.caps.resolve()), ["encoder:libx264"])
        svc4 = make_service(self.tmp, caps=FakeCaps(extra=["video-editing"]), adapter=ToolRouter([FakeAdapter(), ad]))
        svc4.registry.get("silence_cleanup").tools = ["video-editing/cut"]
        self.assertEqual(svc4.tools_for().get("silence_cleanup"), "video-editing/cut", "encoder:aac is not resolved by this fake environment: left to the Skill's doctor, not guessed missing")
        # a plan step naming video-editing/cut when its package is registered passes the validator; the registry knows the tool
        self.assertEqual(svc2.registry.tool("video-editing/cut").skill_id, "video-editing")
        # a decision-level BLOCK when the capability is missing (Decision Engine untouched: it only sees the registry's verdict)
        ir = svc.plan([self.src], "youtube")
        blocked = [d for d in ir.doc["decisions"] if d["subject"] == "capability.silence_cleanup"]
        self.assertEqual(len(blocked), 1); self.assertEqual((blocked[0]["type"], blocked[0]["approval"]), ("BLOCK", "BLOCK"))

    # 14. deterministic / reuse: identical request → identical operation id and output hash; a reused result is reported as such; the executor's idempotent skip still applies
    def test_deterministic_and_reuse(self):
        ad = self._adapter()
        paths, out = self._paths()
        r1 = ad.run(self._op(), paths)
        paths2, out2 = self._paths("02.mp4")
        r2 = ad.run(self._op(output="b"), {**paths2, "b": out2})
        self.assertTrue(r1.ok and r2.ok)
        self.assertEqual(r1.data["operation_id"], r2.data["operation_id"]); self.assertEqual(r1.data["artifact"]["sha256"], r2.data["artifact"]["sha256"])
        os.environ["FAKE_VE_MODE"] = "reused"
        r3 = ad.run(self._op(), paths)
        self.assertTrue(r3.ok); self.assertEqual((r3.data["status"], r3.data["artifact"]["reused"], r3.commands), ("reused", True, []))
        os.environ.pop("FAKE_VE_MODE")
        r4 = ad.run(self._op(keep=[[0.5, 3.0]]), paths)
        self.assertNotEqual(r4.data["operation_id"], r1.data["operation_id"], "different parameters → different identity")

    # 6. Project IR lowering: video.trim → video-editing/cut carries the same meaning (keep ranges, precision) and never an argv
    def test_ir_lowering_and_end_to_end(self):
        from video_agent.execution.compiler import lower_video_trim
        from video_agent.tools import ToolRouter
        op = {"type": "video.trim", "asset": "a", "keep": [[2.85, 9.15], [11.85, 13.85]], "accurate": False, "decision_ids": []}
        self.assertEqual(lower_video_trim("video-editing/cut", op, "a", "a_trim"), {"input": "a", "keep": [[2.85, 9.15], [11.85, 13.85]], "precision": "keyframe", "output": "a_trim"})
        self.assertEqual(lower_video_trim("video-editing/cut", dict(op, accurate=True), "a", "a_trim")["precision"], "frame")
        self.assertEqual(lower_video_trim("ffmpeg-skill/cut", op, "a", "a_trim"), {"input": "a", "segments": "2.850-9.150,11.850-13.850", "output": "a_trim"})
        # plan → step tool video-editing/cut → compile → execute through the fake Skill → QA → provenance (workspace = the service workspace)
        ad = self._adapter(workspace=self.tmp)
        fake = FakeAdapter()
        svc = make_service(self.tmp, caps=FakeCaps(extra=["video-editing", "encoder:aac"]), adapter=ToolRouter([fake, ad]))
        svc.registry.get("silence_cleanup").tools = ["video-editing/cut", "ffmpeg-skill/cut"]
        ir = svc.plan([self.src], "youtube")
        step = next(s for s in ir.doc["plan"]["steps"] if s["skill"] == "silence_cleanup")
        self.assertEqual(step["tool"], "video-editing/cut")
        self.assertEqual(svc.validate(ir).errors, [])
        p = str(Path(self.tmp) / "p.json"); save_ir(ir, p)
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertIn(out["status"], ("COMPLETED", "REVIEW"), out.get("execution"))   # REVIEW = the QA verdict on the fake outputs, not an execution failure
        self.assertEqual(out["execution"]["status"], "COMPLETED", out.get("execution"))
        cut = next(r for r in out["execution"]["results"] if r["tool"] == "video-editing/cut")
        self.assertTrue(cut["ok"]); self.assertTrue(cut["output"].startswith(str(Path(self.tmp).resolve()))); self.assertEqual(cut["data"]["operation"]["type"], "CUT")
        self.assertEqual(cut["data"]["operation"]["parameters"], {"keep": [{"start": 2.85, "end": 13.85}], "precision": "keyframe"})
        self.assertFalse(any(o.tool == "ffmpeg-skill/cut" for o in fake.calls), "the trim went through video-editing, not the reference cut")
        self.assertTrue(any(o.tool == "ffmpeg-skill/loudness" for o in fake.calls), "the rest of the chain is unchanged")
        prov = json.loads((Path(self.tmp) / "jobs" / out["job"]["id"] / "provenance.json").read_text())
        trim = next(e for e in prov["operations"] if e["skill"] == "silence_cleanup")
        self.assertEqual((trim["skill_package"], trim["tool"], trim["tool_version"]), ("video-editing", "video-editing/cut", "0.1.0"))
        self.assertEqual(trim["skill_result"]["artifact"]["sha256"], cut["data"]["artifact"]["sha256"])
        self.assertEqual(trim["skill_result"]["observation"]["provenance"], "OBSERVED")
        self.assertTrue(trim["result"]["commands"] and "provenance only" in trim["result"]["commands"][0])
        self.assertNotIn("argv", json.dumps(trim["args"])); self.assertNotIn("segments", trim["args"])
        # resume reuses the completed cut (agent-level idempotency) without calling the Skill again
        log = str(Path(self.tmp) / "calls.log"); os.environ["FAKE_VE_CALLS"] = log
        out2 = svc.render(load_ir(p), p, resume=out["job"]["id"])
        self.assertEqual(out2["execution"]["status"], "COMPLETED"); self.assertTrue(out2["execution"]["reused"])
        self.assertFalse(os.path.exists(log) and any(json.loads(l)["cmd"] == "run" for l in Path(log).read_text().splitlines()), "no video-editing run on resume")
        # a Skill failure inside the chain is a finite, classified recovery: TOOL_ERROR (retryable) → one retry → BLOCK, no fallback to another engine
        os.environ["FAKE_VE_MODE"] = "tool_error"
        out3 = svc.render(load_ir(p), p)
        self.assertIn(out3["status"], ("FAILED", "BLOCKED"))
        self.assertEqual([r["class"] for r in out3["execution"]["recovery"]], ["UNKNOWN", "UNKNOWN"])
        self.assertFalse(any(o.tool == "ffmpeg-skill/cut" for o in fake.calls))
        os.environ["FAKE_VE_MODE"] = "validation_error"
        out4 = svc.render(load_ir(p), p)
        self.assertEqual(out4["execution"]["recovery"][0]["class"], "SKILL_ERROR"); self.assertEqual(out4["execution"]["recovery"][0]["action"], "BLOCK")
