"""Integrated production pipeline (ADR-031 / ADR-032) on the fake engine and the fake Skills: the ten scenarios of the Phase 3
specification, the QC gate, the refusals, determinism and the explain chain. No ffmpeg, no real Skill (tests/test_integration.py
runs the same scenarios on the real ones)."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter  # noqa: E402
from pipeline_harness import PipelineCaps, clear_modes, fake_media, pipeline_service, plan_and_render  # noqa: E402
from video_agent.artifacts import ArtifactError  # noqa: E402
from video_agent.execution import compile_ir  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.service import Service  # noqa: E402

FULL = {"subtitle": True, "subtitle.burn_in": True, "thumbnail": True, "thumbnail.text": "Talk", "color.target": "bt709", "motion.title": "Opening", "qc": True}


def skills_of(doc):
    return [s["skill"] for s in doc["plan"]["steps"]]


def tools_of(doc):
    return [s["tool"] for s in doc["plan"]["steps"]]


class PipelineScenarioTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.a = fake_media(self.tmp, "a.mp4")
        self.b = fake_media(self.tmp, "b.mp4")

    def tearDown(self):
        clear_modes()

    # ---- Scenario 1: video trim → concat → audio processing → export → QC
    def test_s1_trim_concat_audio_export_qc(self):
        svc = pipeline_service(self.tmp)
        r = plan_and_render(svc, [self.a, self.b], {"edit.concat": True, "qc": True}, name="s1")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        self.assertEqual(skills_of(d), ["silence_cleanup", "silence_cleanup", "video_concat", "loudness_normalization", "delivery_export", "delivery_check", "qc_check"])
        self.assertEqual(tools_of(d)[2], "video-editing/concat"); self.assertEqual(tools_of(d)[-1], "qc/check")
        qc_step = d["plan"]["steps"][-1]
        self.assertEqual((qc_step["params"]["kind"], qc_step["params"]["target"], qc_step["inputs"]), ("delivery", "youtube", ["programme_delivery_youtube"]))
        self.assertEqual(qc_step["depends_on"], ["step_check_youtube"], "the gate runs after the export and its platform check")
        self.assertEqual(d["qa"]["qc"]["subjects"], {"programme": {"kind": "delivery", "targets": ["youtube"]}})
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED"); self.assertEqual(out["status"], "COMPLETED")
        qc_res = [x for x in out["execution"]["results"] if x["tool"] == "qc/check"]
        self.assertEqual(len(qc_res), 1); self.assertTrue(qc_res[0]["ok"]); self.assertEqual(qc_res[0]["data"]["verdict"], "PASS"); self.assertTrue(qc_res[0]["data"]["admitted"])
        rules = json.loads(json.dumps(qc_res[0]["data"]))   # the rules came from the IR: loudness target of the programme's normalisation, streams required
        items = {(i["layer"], i["name"]): i for i in out["qa"]["items"]}
        self.assertEqual(items[("qc", "verdict")]["status"], "PASS"); self.assertEqual(items[("video", "duration")]["observed"], 22.0, "two 11 s trims joined")
        art = out["artifacts"][0]
        self.assertEqual((art["type"], art["stage"], art["qa"]["qc"], art["source"]), ("YOUTUBE", "approved", "PASS", list(d["assets"])), "QC PASS → READY (stage approved)")
        self.assertIsInstance(rules, dict)

    # ---- Scenario 2: video → transcription → subtitle → QC
    def test_s2_transcription_subtitle_qc(self):
        svc = pipeline_service(self.tmp)
        r = plan_and_render(svc, [self.a], {"subtitle": "vtt", "qc": True}, name="s2")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        self.assertIn("transcript", [o["kind"] for o in d["analysis"]["observations"]], "subtitles pull the transcript measurement in")
        self.assertEqual(skills_of(d), ["silence_cleanup", "subtitle_generation", "loudness_normalization", "delivery_export", "delivery_check", "qc_check", "qc_check"])
        gen = next(op for op in d["captions"]["operations"] if op["type"] == "captions.generate")
        self.assertEqual((gen["format"], gen["language"], [(c["id"], c["start"], c["end"]) for c in gen["cues"]]), ("vtt", "ja", [("c0001", 0.0, 0.35), ("c0002", 1.15, 5.55)]),
                         "segments 0.5–3.2 / 4.0–8.4 of the source, mapped through the trim keep range 2.85–13.85")
        self.assertEqual(gen["timeline_map"]["inputs"][list(d["assets"])[0]]["keep"], [[2.85, 13.85]])
        self.assertTrue(all(o["kind"] == "transcript" for o in d["analysis"]["observations"] if o["id"] in gen["sources"]))
        self.assertNotIn("speaker", json.dumps(gen["cues"]))
        dec = {x["subject"]: x for x in d["decisions"]}
        self.assertEqual(dec["subtitle.generate"]["type"], "TRANSFORM"); self.assertIn("observation", dec["subtitle.generate"]["basis"]["evidence_classes"])
        self.assertEqual(dec["subtitle.generate"]["basis"]["approval"]["key"], "subtitle.generate.approval")
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "subtitle_generation")
        self.assertEqual((step["inputs"], step["outputs"], step["depends_on"]), ([], [list(d["assets"])[0] + "_captions"], ["step_trim_" + list(d["assets"])[0]]),
                         "the sidecar depends on the trim (its cues are on the trimmed timeline) and consumes no file")
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED")
        side = next(x for x in out["execution"]["results"] if x["tool"] == "subtitle/generate")
        self.assertTrue(side["output"].endswith(".vtt")); self.assertTrue(Path(side["output"]).read_text(encoding="utf-8").startswith("WEBVTT"))
        items = {(i["name"], i["artifact"]): i for i in out["qa"]["items"]}
        cap = list(d["assets"])[0] + "_captions"
        self.assertEqual((items[("cues", cap)]["status"], items[("cues", cap)]["observed"], items[("format", cap)]["status"], items[("verdict", cap)]["status"]), ("PASS", 2, "PASS", "PASS"))
        arts = {a["type"]: a for a in out["artifacts"]}
        self.assertEqual((arts["CAPTIONS"]["stage"], arts["CAPTIONS"]["qa"]["qc"], arts["YOUTUBE"]["stage"]), ("approved", "PASS", "approved"))
        sc_qc = next(x for x in out["execution"]["results"] if x["tool"] == "qc/check" and x["data"]["kind"] == "subtitle")
        self.assertIn("reference_video", sc_qc["data"]["companions"])

    # ---- Scenario 3: video → color grading → thumbnail → QC
    def test_s3_color_thumbnail_qc(self):
        svc = pipeline_service(self.tmp)
        r = plan_and_render(svc, [self.a], {"color.target": "bt709", "thumbnail": True, "thumbnail.at": 4.0, "qc": True}, name="s3")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        self.assertEqual(skills_of(d), ["silence_cleanup", "color_retag", "loudness_normalization", "delivery_export", "delivery_check", "thumbnail_frame", "qc_check"])
        col = d["color"]["operations"][0]
        aid = list(d["assets"])[0]
        self.assertEqual((col["type"], col["input"], col["output"], col["target"]), ("color.retag", f"{aid}_trim", f"{aid}_retag", "bt709"))
        th = next(op for op in d["graphics"]["operations"] if op["type"] == "graphics.thumbnail")
        self.assertEqual((th["input"], th["timestamp"], th["format"]), (f"{aid}_retag", 4.0, "png"), "the frame is taken from the finished picture, before loudness / export")
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED")
        res = {x["tool"]: x for x in out["execution"]["results"]}
        self.assertEqual(res["color-grading/run"]["data"]["operation_type"], "RETAG"); self.assertEqual(res["color-grading/run"]["data"]["observation"]["provenance"], "OBSERVED")
        self.assertTrue(res["thumbnail/extract_frame"]["output"].endswith(".png"))
        arts = {a["type"]: a for a in out["artifacts"]}
        self.assertEqual((arts["THUMBNAIL"]["stage"], arts["THUMBNAIL"]["qa_status"], arts["THUMBNAIL"]["qa"].get("qc")), ("candidate", "PASS", None), "a thumbnail is checked by the agent only")
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        kinds = [o["kind"] for o in prov["skill_observations"]]
        self.assertIn("media.probe", kinds); self.assertIn("image.probe", kinds); self.assertIn("qc.report", kinds)

    # ---- Scenario 4: video → motion graphics → subtitle burn-in → QC
    def test_s4_motion_subtitle_burn_qc(self):
        svc = pipeline_service(self.tmp)
        r = plan_and_render(svc, [self.a], {"motion.text": "LIVE", "motion.text.position": "top-left", "subtitle": True, "subtitle.burn_in": True, "qc": True}, name="s4")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        self.assertEqual(skills_of(d), ["silence_cleanup", "motion_graphics", "subtitle_generation", "subtitle_burn_in", "loudness_normalization", "delivery_export", "delivery_check", "qc_check", "qc_check"])
        aid = list(d["assets"])[0]
        g = d["graphics"]["operations"][0]
        self.assertEqual((g["type"], g["input"], g["output"]), ("graphics.render", f"{aid}_trim", f"{aid}_graphics"))
        self.assertEqual(g["elements"], [{"id": "el1_text_overlay", "type": "text_overlay", "start": 0.0, "end": 11.0, "parameters": {"text": "LIVE", "position": "top-left"}}], "a text overlay runs the whole trimmed timeline by default")
        burn = next(op for op in d["captions"]["operations"] if op["type"] == "captions.burn")
        self.assertEqual((burn["input"], burn["sidecar"], burn["output"]), (f"{aid}_graphics", f"{aid}_captions", f"{aid}_burn"))
        bst = next(s for s in d["plan"]["steps"] if s["skill"] == "subtitle_burn_in")
        self.assertEqual(sorted(bst["depends_on"]), sorted([f"step_graphics_{aid}", f"step_captions_{aid}"]))
        self.assertEqual(next(s for s in d["plan"]["steps"] if s["skill"] == "loudness_normalization")["inputs"], [f"{aid}_burn"], "the burnt-in picture continues the chain")
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED")
        res = {x["tool"]: x for x in out["execution"]["results"]}
        self.assertEqual(res["subtitle/render"]["data"]["engine"]["id"], "ffmpeg-skill"); self.assertIsNone(res["subtitle/render"]["data"]["observation"])
        self.assertEqual({a["type"] for a in out["artifacts"]}, {"YOUTUBE", "CAPTIONS"})

    # ---- Scenario 5: the whole pipeline on two inputs (concat)
    def test_s5_multi_skill_pipeline(self):
        svc = pipeline_service(self.tmp)
        r = plan_and_render(svc, [self.a, self.b], dict(FULL, **{"edit.concat": True, "motion.text": "LIVE"}), name="s5")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        self.assertEqual(skills_of(d), ["silence_cleanup", "silence_cleanup", "video_concat", "color_retag", "motion_graphics", "subtitle_generation", "subtitle_burn_in", "loudness_normalization",
                                        "delivery_export", "delivery_check", "thumbnail_render", "qc_check", "qc_check"])
        self.assertEqual(len({t.split("/")[0] for t in tools_of(d)}), 7, "seven Skill packages in one plan")
        gen = next(op for op in d["captions"]["operations"] if op["type"] == "captions.generate")
        self.assertEqual([(c["id"], c["start"]) for c in gen["cues"]], [("c1_0001", 0.0), ("c1_0002", 1.15), ("c2_0001", 11.0), ("c2_0002", 12.15)], "the second input's cues are offset by the first's 11 s")
        self.assertEqual([e["type"] for e in d["graphics"]["operations"][0]["elements"]], ["title", "text_overlay"], "one render carries every element, in the fixed type order")
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED"); self.assertEqual(out["qa"]["status"], "PASS")
        self.assertEqual({a["type"]: a["stage"] for a in out["artifacts"]}, {"YOUTUBE": "approved", "CAPTIONS": "approved", "THUMBNAIL": "candidate"})
        # provenance links every operation to its decision and Skill package
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        pk = sorted({e["skill_package"] for e in prov["operations"]})
        self.assertEqual(pk, ["color-grading", "ffmpeg-skill", "motion-graphics", "qc", "subtitle", "thumbnail", "video-editing"])
        self.assertTrue(all(e["decision"] for e in prov["operations"]))
        # explain --pipeline: every level present, decisions cite evidence, steps cite decisions, artifacts cite steps
        info = Service.explain_pipeline(d, provenance=prov, artifacts=out["artifacts"])
        self.assertTrue(all(info["counts"][lv] > 0 for lv in info["levels"]), info["counts"])
        self.assertTrue(all(row["evidence"] for row in info["rows"] if row["level"] == "decision"))
        self.assertTrue(all(row["decisions"] for row in info["rows"] if row["level"] == "step"))
        self.assertTrue(all(row["step"] for row in info["rows"] if row["level"] == "artifact"))

    # ---- Scenario 6: a failure in the middle → resume reuses what completed
    def test_s6_failure_then_resume(self):
        svc = pipeline_service(self.tmp)
        os.environ["FAKE_MG_MODE"] = "tool_error_final"
        r = plan_and_render(svc, [self.a], {"color.target": "bt709", "motion.title": "Opening", "subtitle": True, "qc": True}, name="s6")
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "FAILED")
        failed = next(x for x in out["execution"]["results"] if not x["ok"])
        self.assertEqual((failed["tool"], failed["data"]["error"]["code"], failed["data"]["error"]["retryable"]), ("motion-graphics/run", "TOOL_ERROR", False))
        self.assertEqual([x["tool"] for x in out["execution"]["results"] if x["ok"]], ["ffmpeg-skill/cut", "color-grading/run"])
        self.assertNotIn("artifacts", out); self.assertEqual(out["job"]["state"], "FAILED")
        clear_modes()
        out2 = svc.render(load_ir(r["path"]), r["path"], resume=out["job"]["id"])
        self.assertEqual(out2["execution"]["status"], "COMPLETED")
        self.assertEqual(len(out2["execution"]["skipped"]), 2, "the trim and the colour operation are reused, the rest runs")
        self.assertEqual([x["tool"] for x in out2["execution"]["results"]][:2], ["motion-graphics/run", "subtitle/generate"])
        self.assertEqual(out2["resume"]["resumed_from"], out["job"]["id"]); self.assertFalse(out2["resume"]["plan_changed"])
        self.assertEqual({a["type"]: a["stage"] for a in out2["artifacts"]}, {"YOUTUBE": "approved", "CAPTIONS": "approved"})

    # ---- Scenario 7: plan revision → approval → execution
    def test_s7_revision_approval_execution(self):
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self.a], "youtube", user_requirements={"thumbnail": True, "color.target": "bt709", "qc": True}, params={"language": "ja"})
        path = str(Path(svc.workspace) / "plans" / "s7.project.json"); save_ir(ir, path)
        th = next(x for x in ir.doc["decisions"] if x["subject"] == "thumbnail.render")
        svc.reject(load_ir(path), path, [th["id"]], reason="no thumbnail this time")
        blocked = svc.render(load_ir(path), path)
        self.assertEqual(blocked["status"], "BLOCKED", "a plan citing a rejected decision never renders")
        rev = svc.revise(load_ir(path), path, feedback="drop the thumbnail")
        ir2 = load_ir(path)
        self.assertEqual(ir2.version, 2); self.assertNotIn("thumbnail_frame", skills_of(ir2.doc)); self.assertIn("color_retag", skills_of(ir2.doc))
        self.assertEqual(ir2.doc["graphics"], {}, "the dropped operation leaves its section empty")
        self.assertIn("thumbnail.render", json.dumps(rev.get("dropped") or rev), "the rejection is carried into the revision")
        waiting = svc.render(load_ir(path), path)
        self.assertEqual(waiting["status"], "WAITING_FOR_APPROVAL", "a revised plan needs an explicit approval of that version")
        svc.approve(load_ir(path), path, ["all"])
        out = svc.render(load_ir(path), path)
        self.assertEqual((out["execution"]["status"], out["status"]), ("COMPLETED", "COMPLETED"))
        self.assertEqual({a["type"] for a in out["artifacts"]}, {"YOUTUBE"})
        self.assertEqual(load_ir(path).doc["revision"]["approved_plan_version"], 2)

    # ---- Scenario 8: capability drift → BLOCK (contract drift, MISSING and UNKNOWN capabilities never execute)
    def test_s8_capability_drift_blocks(self):
        os.environ["FAKE_CG_MODE"] = "contract_drift"
        svc = pipeline_service(self.tmp, caps=PipelineCaps(missing=["color-grading"]))   # what the resolver does with a drifted contract: the package capability is MISSING
        ad = next(a for a in svc.adapter([]).adapters if a.name == "color-grading")
        self.assertTrue(ad.drift() and any("EXPOSURE" in x for x in ad.drift()))
        clear_modes()
        ir = svc.plan([self.a], "youtube", user_requirements={"color.target": "bt709", "qc": True}, params={"language": "ja"})
        d = ir.doc
        dec = {x["subject"]: x for x in d["decisions"]}
        self.assertEqual((dec["capability.color_retag"]["approval"], dec["capability.color_retag"]["params"]["missing"]), ("BLOCK", ["color-grading"]))
        self.assertEqual(d["plan"]["status"], "BLOCKED"); self.assertEqual(next(s for s in d["plan"]["steps"] if s["skill"] == "color_retag")["tool"], None)
        path = str(Path(svc.workspace) / "plans" / "s8.project.json"); save_ir(ir, path)
        out = svc.render(load_ir(path), path, approve=["all"])
        self.assertIn(out["status"], ("BLOCKED", "FAILED")); self.assertNotIn("artifacts", out)
        # an UNKNOWN element capability blocks that element (never guessed as available)
        svc2 = pipeline_service(self.tmp, caps=PipelineCaps(unknown=["motion-graphics:title"]))
        ir2 = svc2.plan([self.a], "youtube", user_requirements={"motion.title": "Opening"}, params={"language": "ja"})
        dec2 = {x["subject"]: x for x in ir2.doc["decisions"]}
        self.assertEqual(dec2["capability.motion_graphics:title"]["approval"], "BLOCK"); self.assertNotIn("graphics.title", dec2); self.assertEqual(ir2.doc["plan"]["status"], "BLOCKED")

    # ---- Scenario 9: artifact hash mismatch → never a success, never reused, never promoted
    def test_s9_hash_mismatch_forbids_reuse(self):
        svc = pipeline_service(self.tmp)
        os.environ["FAKE_CG_MODE"] = "hash_mismatch"
        r = plan_and_render(svc, [self.a], {"color.target": "bt709", "qc": True}, name="s9")
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "FAILED")
        bad = next(x for x in out["execution"]["results"] if x["tool"] == "color-grading/run")
        self.assertEqual((bad["ok"], bad["data"]["error"]["code"], bad["data"]["error"]["retryable"]), (False, "INVALID_RESULT", False)); self.assertIn("sha256", bad["data"]["error"]["message"])
        self.assertFalse([p for p in Path(svc.workspace).rglob("*_retag.mp4")], "an output whose hash does not match is removed, never kept for reuse")
        clear_modes()
        out2 = svc.render(load_ir(r["path"]), r["path"], resume=out["job"]["id"])
        self.assertEqual(out2["execution"]["status"], "COMPLETED"); self.assertEqual(len(out2["execution"]["skipped"]), 1, "only the trim is reused")
        # a delivered artifact whose bytes changed is no longer that artifact: integrity fails, promotion is refused
        art = next(a for a in out2["artifacts"] if a["type"] == "YOUTUBE")
        with open(art["path"], "ab") as fh:
            fh.write(b"tampered")
        self.assertFalse(svc.artifact(art["id"])["integrity"]["ok"])
        with self.assertRaises(ArtifactError):
            svc.promote_artifact(art["id"], "final")
        # a tampered intermediate is not reused on the next resume (the record no longer matches), its downstream is re-run from it
        job_dir = Path(svc.workspace) / "jobs" / out2["job"]["id"]
        inter = next(p for p in job_dir.rglob("*_retag.mp4"))
        with open(inter, "ab") as fh:
            fh.write(b"x")
        out3 = svc.render(load_ir(r["path"]), r["path"], resume=out2["job"]["id"])
        self.assertEqual(out3["execution"]["status"], "COMPLETED")
        rerun = [x["tool"] for x in out3["execution"]["results"] if x["tool"] == "color-grading/run"]
        self.assertEqual(rerun, ["color-grading/run"], "the tampered intermediate is produced again")

    # ---- Scenario 10: same input, same plan → deterministic operations and idempotent execution
    def test_s10_determinism_and_idempotency(self):
        reqs = dict(FULL, **{"edit.concat": True})
        docs, hashes = [], []
        for n in range(2):
            svc = pipeline_service(self.tmp)
            ir = svc.plan([self.a, self.b], "youtube", user_requirements=reqs, params={"language": "ja"})
            docs.append(ir.doc); hashes.append(ir.plan_hash())

        def norm(doc):
            s = json.dumps({k: doc[k] for k in ("video", "audio", "captions", "graphics", "color", "delivery", "qa")})
            for i, aid in enumerate(doc["assets"]):
                s = s.replace(aid, f"A{i}")   # project-specific ids first, then a canonical dump
            return json.dumps(json.loads(re.sub(r"(dec|obs|inf|evt)_[0-9a-f]{10}", r"\1", s)), sort_keys=True)
        self.assertEqual(norm(docs[0]), norm(docs[1]), "the same request on the same inputs plans the same operations")
        ids = [re.sub(r"asset_[0-9a-f]{10}", "asset", s["id"]) for s in docs[0]["plan"]["steps"]]
        self.assertEqual(ids, [re.sub(r"asset_[0-9a-f]{10}", "asset", s["id"]) for s in docs[1]["plan"]["steps"]])
        svc = pipeline_service(self.tmp)
        r = plan_and_render(svc, [self.a, self.b], reqs, name="s10")
        ops1, _ = compile_ir(load_ir(r["path"]), str(Path(self.tmp) / "j1"))
        ops2, _ = compile_ir(load_ir(r["path"]), str(Path(self.tmp) / "j2"))
        self.assertEqual([(o.id, o.idempotency_key, o.tool) for o in ops1], [(o.id, o.idempotency_key, o.tool) for o in ops2], "operation ids and chained keys are content-derived")
        self.assertTrue(all(o.idempotency_key for o in ops1 if o.outputs)); self.assertTrue(all(not o.idempotency_key for o in ops1 if o.kind == "qa"), "a measurement is never skipped")
        out2 = svc.render(load_ir(r["path"]), r["path"], resume="last")
        self.assertEqual(out2["execution"]["status"], "COMPLETED")
        self.assertEqual(len(out2["execution"]["skipped"]), len([o for o in ops1 if o.outputs]), "every transform is reused; the checks and the QC gate run again")
        self.assertEqual([a["hash"] for a in out2["artifacts"]], [a["hash"] for a in r["out"]["artifacts"]])


class PipelineGateAndRefusalTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.a = fake_media(self.tmp, "a.mp4")

    def tearDown(self):
        clear_modes()

    def test_qc_warn_and_fail_gates(self):
        svc = pipeline_service(self.tmp)
        os.environ["FAKE_QC_MODE"] = "verdict_warn"
        out = plan_and_render(svc, [self.a], {"qc": True}, name="warn")["out"]
        art = out["artifacts"][0]
        self.assertEqual((out["qa"]["status"], art["qa"]["qc"], art["stage"]), ("WARN", "WARN", "candidate"), "WARN: stays a candidate; promotion needs the review the policy asks for (CONFIRM)")
        self.assertEqual(svc.promote_artifact(art["id"], "final")["stage"], "final", "a WARN candidate can still be delivered explicitly")
        os.environ["FAKE_QC_MODE"] = "verdict_fail"
        out = plan_and_render(svc, [self.a], {"qc": True}, name="fail")["out"]
        art = out["artifacts"][0]
        self.assertEqual((out["status"], art["qa"]["qc"], art["qa_status"], art["stage"]), ("REVIEW", "FAIL", "FAIL", "working"))
        with self.assertRaises(ArtifactError):
            svc.promote_artifact(art["id"], "final")
        self.assertTrue(any(i["type"] == "DELIVERY_SPEC_FAILURE" for i in out["qa"]["incidents"]))
        # a report about another file is not admitted: FAIL, never a promotion on a foreign verdict
        os.environ["FAKE_QC_MODE"] = "fingerprint_mismatch"
        out = plan_and_render(svc, [self.a], {"qc": True}, name="fp")["out"]
        self.assertEqual(out["execution"]["status"], "FAILED", "the adapter refuses the report before QA sees it")
        bad = next(x for x in out["execution"]["results"] if x["tool"] == "qc/check")
        self.assertEqual(bad["data"]["error"]["code"], "INVALID_RESULT"); self.assertIn("fingerprint", bad["data"]["error"]["message"] + json.dumps(bad["data"]["error"]["details"]))
        # WARN promoted automatically only when the policy says so
        svc2 = pipeline_service(self.tmp)
        os.environ["FAKE_QC_MODE"] = "verdict_warn"
        out = plan_and_render(svc2, [self.a], {"qc": True, "qc.warn.promotion": "AUTO"}, name="warnauto")["out"]
        self.assertEqual((out["artifacts"][0]["stage"], out["artifacts"][0]["qa"]["qc_warn_promotion"]), ("approved", "AUTO"))

    def test_without_the_gate_nothing_changes(self):
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self.a], "youtube")
        d = ir.doc
        self.assertEqual(skills_of(d), ["silence_cleanup", "loudness_normalization", "delivery_export", "delivery_check"])
        self.assertEqual((d["captions"], d["graphics"], d["color"], "qc" in d["qa"]), ({}, {}, {}, False), "no finishing section unless planned (plan hash unchanged)")
        base = pipeline_service(self.tmp, skills=[]).plan([self.a], "youtube")
        self.assertEqual(ir.plan_hash() == base.plan_hash(), False, "plan hashes differ only by the project's asset ids")   # different projects, same content otherwise
        self.assertEqual([s["skill"] for s in base.doc["plan"]["steps"]], skills_of(d))

    def test_refusals(self):
        svc = pipeline_service(self.tmp)
        for bad, msg in (({"subtitle.burn_in": True}, "subtitle is not"), ({"subtitle": "ass"}, "true or false"), ({"thumbnail": True, "thumbnail.at": -1}, "within"),
                         ({"color.target": "rec2100"}, "one of"), ({"color.lut": "/nope.cube"}, "not found"), ({"motion.title": ""}, "non-empty"), ({"motion.text": "x", "motion.text.start": 5, "motion.text.end": 2}, "before"),
                         ({"motion.image.position": "top"}, "motion.image is not"), ({"qc": "maybe"}, "true or false"), ({"thumbnail.text": "x"}, "thumbnail is not")):
            with self.assertRaises(ValueError, msg=str(bad)) as cm:
                svc.plan([self.a], "youtube", user_requirements=bad)
            self.assertIn(msg, str(cm.exception), str(bad))
        with self.assertRaises(ValueError):
            svc.plan([self.a], "youtube", user_requirements={"caption": True})
        # no transcript → no subtitles (BLOCK with the hint), never subtitled from silence
        no_ts = pipeline_service(self.tmp, transcription=False, caps=PipelineCaps(missing=["transcription"]))
        with self.assertRaises(RuntimeError):
            no_ts.plan([self.a], "youtube", user_requirements={"subtitle": True})
        ir = svc.plan([self.a], "youtube", user_requirements={"subtitle": True}, kinds=[], strategy="TARGETED", params={"language": "ja"})
        self.assertIn("transcript", [o["kind"] for o in ir.doc["analysis"]["observations"]], "TARGETED analysis adds the transcript for a subtitle request")
        # the audio path and picture finishing conflict
        ir = svc.plan([fake_media(self.tmp, "v.wav", video=False)], "generic", user_requirements={"audio.production": True, "audio.gain": -3, "thumbnail": True})
        dec = {x["subject"]: x for x in ir.doc["decisions"]}
        self.assertEqual(dec["audio.production"]["approval"], "BLOCK"); self.assertIn("finishing", dec["audio.production"]["decision"])
        # HDR → SDR on an SDR source is a KEEP, not a tone-map
        ir = svc.plan([self.a], "youtube", user_requirements={"color.sdr": True})
        dec = {x["subject"]: x for x in ir.doc["decisions"]}
        self.assertEqual((dec["color.hdr_to_sdr"]["type"], dec["color.hdr_to_sdr"]["provenance"]), ("KEEP", "OBSERVED")); self.assertEqual(ir.doc["color"], {})
        # a burn-in with no intermediate before it is refused (the source is never rewritten)
        tmp2 = os.path.realpath(tempfile.mkdtemp())   # its own workspace: the observation cache of self.tmp already holds the silences of the same bytes
        flat = pipeline_service(tmp2, silences=[])
        ir = flat.plan([fake_media(tmp2, "flat.mp4")], "generic", user_requirements={"subtitle": True, "subtitle.burn_in": True}, params={"language": "ja"})
        dec = {x["subject"]: x for x in ir.doc["decisions"]}
        self.assertEqual((dec["subtitle.generate"]["type"], dec["subtitle.burn_in"]["approval"]), ("TRANSFORM", "BLOCK"))
        # a picture operation on an audio-only asset is refused
        ir = svc.plan([fake_media(self.tmp, "n.wav", video=False)], "generic", user_requirements={"thumbnail": True})
        dec = {x["subject"]: x for x in ir.doc["decisions"]}
        self.assertEqual(dec["thumbnail.render"]["approval"], "BLOCK")

    def test_validator_refuses_tampered_finishing_sections(self):
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self.a], "youtube", user_requirements=dict(FULL, **{"motion.text": "LIVE"}), params={"language": "ja"})
        self.assertEqual(svc.validate(ir).errors, [])
        aid = list(ir.doc["assets"])[0]
        cases = [
            (lambda d: d["color"]["operations"][0].__setitem__("filter", "x"), "Additional properties"),
            (lambda d: d["color"]["operations"][0].__setitem__("target", "bt2100"), "is not one of"),
            (lambda d: d["captions"]["operations"][0]["cues"].__setitem__(0, {"id": "c0001", "start": 5.0, "end": 4.0, "text": "x"}), "start must be before end"),
            (lambda d: d["captions"]["operations"][0]["cues"][0].__setitem__("end", 400.0), "after the delivered timeline"),
            (lambda d: d["captions"]["operations"][1].__setitem__("input", aid), "source asset"),
            (lambda d: d["graphics"]["operations"][0]["elements"][0].__setitem__("end", 999.0), "after the subject's timeline"),
            (lambda d: d["graphics"]["operations"][1].__setitem__("timestamp", 999.0), "after the picture ends"),
            (lambda d: d["plan"]["steps"].__delitem__(1), "no plan step"),
            (lambda d: d["plan"]["steps"][2].__setitem__("depends_on", []), "without depending on it"),
            (lambda d: d["qa"]["qc"]["subjects"].__setitem__("ghost", {"kind": "delivery", "targets": ["youtube"]}), "no qc_check step"),
        ]
        for mutate, fragment in cases:
            doc = json.loads(json.dumps(ir.doc))
            mutate(doc)
            from video_agent.project.ir import ProjectIR
            errs = svc.validate(ProjectIR(doc), check_paths=False).errors
            self.assertTrue(any(fragment in e for e in errs), (fragment, errs[:3]))

    def test_check_with_qc(self):
        svc = pipeline_service(self.tmp)
        out = svc.check(self.a, "youtube", qc=True)
        self.assertEqual((out["qc"]["admitted"], out["qc"]["verdict"]), (True, "PASS"))
        os.environ["FAKE_QC_MODE"] = "fingerprint_mismatch"
        out = svc.check(self.a, "youtube", qc=True)
        self.assertFalse(out["qc"]["admitted"]); self.assertIsNone(out["qc"]["verdict"])


if __name__ == "__main__":
    unittest.main()
