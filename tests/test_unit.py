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

from video_agent.capabilities.resolver import Capability  # noqa: E402
from video_agent.execution import Executor, compile_ir  # noqa: E402
from video_agent.execution.recovery import classify_error, next_attempt  # noqa: E402
from video_agent.models import Event, Operation, TimeRange, ToolResult  # noqa: E402
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
        self.assertEqual(d["schema_version"], "1.1")
        self.assertTrue(d["provenance"]["plan_hash"] and d["provenance"]["ir_hash"])
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
        self.assertEqual(out["status"], "FAILED")
        self.assertFalse(out["job"]["completed_ops"])

    def test_operation_ids_are_deterministic(self):
        svc = make_service(self.tmp)
        ir = svc.plan([self.src], "youtube")
        a = [o.id for o in compile_ir(ir, "/w/jobs/j")[0]]
        b = [o.id for o in compile_ir(ir, "/w/jobs/k")[0]]
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), len(a))

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
        p = str(Path(self.tmp) / "v10.json")
        Path(p).write_text(json.dumps(doc))
        loaded = load_ir(p)
        self.assertEqual(loaded.doc["schema_version"], "1.1")
        self.assertEqual(loaded.doc["provenance"]["plan_hash"], ir.plan_hash())
        self.assertIsNone(loaded.doc["execution"]["resume_from"])
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
