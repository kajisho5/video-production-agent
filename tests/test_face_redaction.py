"""ADR-090 (detection-skill roadmap step 7, agent side: blur faces -> redact; video-production-agent roadmap step 9 detection): detection-skill's
face tracks as the analysis kind `face_tracks`, asked for by `edit.redact_faces`, proposed as a CONFIRM `video.redact_faces` and executed by
ffmpeg-skill cut / redact / broll, against a fake `detection` process (tests/fake_detection.py) that speaks detection-skill main's contract,
doctor, detections and tracks documents, with the fake media engine. No engine, ffmpeg or network."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import PipelineCaps, clear_modes, fake_media, pipeline_service  # noqa: E402

from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.faces import MAX_SEGMENTS, elementary_segments, parse_face_requirements, redaction_regions  # noqa: E402
from video_agent.capabilities.resolver import CapabilityResolver  # noqa: E402
from video_agent.models import Operation  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.service import BRIEF_REQUIREMENT_KEYS, _check_requirement_keys  # noqa: E402
from video_agent.tools.detection import (ERROR_CLASSES, ContractError, DetectionAdapter, breaking_drift, check_contract, drift, locate_detection,  # noqa: E402
                                         pinned_contract, recovery_class)
from video_agent.tools.detection.adapter import PINNED_CONTRACT_PATH  # noqa: E402
from video_agent.tools.discovery import MODULE_BOOTSTRAP  # noqa: E402
from video_agent.tools.ffmpeg_skill.catalog import CATALOG  # noqa: E402
from video_agent.tools.skill_process import CliSkill  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_detection.py")
MODES = ("FAKE_DET_MODE", "FAKE_DET_READY", "FAKE_DET_CONTRACT", "FAKE_DET_CALLS", "VIDEO_AGENT_DETECTION_DIR")
FACE = [{"start": 2.0, "end": 6.0, "box": [101.0, 61.0, 81.0, 99.0]}]
TWO = FACE + [{"start": 4.0, "end": 9.0, "box": [600.0, 200.0, 50.0, 60.0]}]


def fake_skill() -> CliSkill:
    return CliSkill("detection", [sys.executable, FAKE], None, {})


def det_adapter(tmp: str) -> DetectionAdapter:
    ws = str(Path(tmp) / "ws")
    os.makedirs(ws, exist_ok=True)
    return DetectionAdapter(fake_skill(), workspace=ws, allowed_inputs=[str(Path(tmp) / "src"), ws], ffmpeg_skill_dir=tmp)


def det_capabilities(factory) -> dict:
    caps: dict = {}
    CapabilityResolver._finishing_skill(CapabilityResolver.__new__(CapabilityResolver), caps, "detection", fake_skill(), lambda _sk: factory(),
                                        "VIDEO_AGENT_DETECTION_DIR", "detection-skill", "detection",
                                        per_item=lambda ad, doc: (ad.operation_status(doc), {}), item_evidence=lambda ad, doc: ad.operation_evidence(doc))
    return caps


class DetCaps(PipelineCaps):
    def __init__(self, det: dict, **kw):
        super().__init__(**kw)
        self.det = det

    def resolve(self, refresh=False):
        out = super().resolve(refresh)
        out.update(self.det)
        return out


def det_service(tmp: str, installed: bool = True, **kw):
    caps = DetCaps(det_capabilities(lambda: det_adapter(tmp)) if installed else {})
    svc = pipeline_service(tmp, caps=caps, transcription=False, **kw)
    if installed:
        svc._adapter.register(det_adapter(tmp))
        svc.adapter([])
    return svc


def calls(log: str):
    return [json.loads(x) for x in Path(log).read_text(encoding="utf-8").splitlines()] if os.path.exists(log) else []


class _Base(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self._env = {k: os.environ.get(k) for k in MODES}
        for k in MODES:
            os.environ.pop(k, None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.log = str(Path(self.tmp) / "det_calls.log")
        os.environ["FAKE_DET_CALLS"] = self.log

    def tearDown(self):
        clear_modes()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def det_calls(self, cmd):
        return [c for c in calls(self.log) if c and c[0] == cmd]


class ContractTests(_Base):
    def test_pin_shape_and_contract_checks(self):
        c = pinned_contract()
        raw = PINNED_CONTRACT_PATH.read_text(encoding="utf-8")
        self.assertEqual(raw, json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n", "one sorted line (no release tag: pinned from main)")
        self.assertEqual((c["skill_id"], c["version"], c["contract_version"]), ("detection", "0.1.0", "1.0"))
        self.assertEqual(check_contract(c), [])
        self.assertEqual(drift(c), [])
        self.assertEqual(breaking_drift(c), [])
        self.assertEqual(set(ERROR_CLASSES), set(c["errors"]), "every detection-skill code has a recovery class")
        self.assertEqual((recovery_class("MODEL_NOT_FOUND"), recovery_class("ENGINE_TIMEOUT"), recovery_class("NEW_CODE")), ("TOOL_MISSING", "TIMEOUT", "SKILL_ERROR"))

    def test_refused_contracts(self):
        c = pinned_contract()
        bad = json.loads(json.dumps(c)); bad["version"] = "0.2.0"
        self.assertTrue(any("supported range" in e for e in check_contract(bad)))
        bad = json.loads(json.dumps(c)); next(t for t in bad["tools"] if t["tool_id"] == "detection/faces")["writes_media"] = True
        self.assertTrue(any("writes_media false" in e for e in check_contract(bad)))
        bad = json.loads(json.dumps(c)); bad["tracks"]["identity"] = True
        self.assertTrue(any("identity false" in e for e in check_contract(bad)))
        bad = json.loads(json.dumps(c)); bad["execution"]["network"] = True
        self.assertTrue(any("network false" in e for e in check_contract(bad)))
        bad = json.loads(json.dumps(c)); del bad["errors"]["MODEL_NOT_FOUND"]
        self.assertTrue(any("MODEL_NOT_FOUND" in e for e in check_contract(bad)))

    def test_drift_classification(self):
        os.environ["FAKE_DET_CONTRACT"] = "drift_version"
        ad = det_adapter(self.tmp)
        self.assertTrue(ad.drift())
        self.assertEqual(ad.breaking_drift(), [], "a version within 0.1.x is not fatal")
        os.environ["FAKE_DET_CONTRACT"] = "drift_fatal"
        self.assertTrue(det_adapter(self.tmp).breaking_drift(), "the tracks argv changed: fatal")
        caps = det_capabilities(lambda: det_adapter(self.tmp))
        self.assertEqual(caps["detection"].status, "MISSING")
        os.environ["FAKE_DET_CONTRACT"] = "incompatible"
        with self.assertRaises(ContractError):
            det_adapter(self.tmp)

    def test_locate_never_searches_the_current_directory(self):
        cwd = os.getcwd()
        here = Path(self.tmp) / "cwd"
        (here / "detection-skill" / "src" / "detection_skill").mkdir(parents=True)
        (here / "detection-skill" / "src" / "detection_skill" / "cli.py").write_text("", encoding="utf-8")
        try:
            os.chdir(here)
            located = locate_detection(env={"PATH": ""})
            self.assertTrue(located is None or str(located.root) != str(here / "detection-skill"))
        finally:
            os.chdir(cwd)
        explicit = locate_detection(str(here / "detection-skill"), env={})
        self.assertEqual(explicit.root, here / "detection-skill")
        self.assertEqual(explicit.command[1:3], ["-c", MODULE_BOOTSTRAP], "run without the current directory on sys.path (ADR-074)")
        self.assertIsNone(locate_detection(env={"VIDEO_AGENT_DETECTION_DIR": str(Path(self.tmp) / "nothing"), "PATH": ""}), "the env var is authoritative")

    def test_broll_is_catalogued_for_the_composite(self):
        self.assertEqual(CATALOG["broll"]["flags"], {"insert": "list[path_in]", "at": "list[str]", "end": "list[str]", "audio": "str", "output": "path_out"})


class CapabilityTests(_Base):
    def test_ready_engine_makes_faces_available_and_names_the_agents_ffmpeg_skill(self):
        caps = det_capabilities(lambda: det_adapter(self.tmp))
        self.assertEqual((caps["detection"].status, caps["detection:FACES"].status), ("AVAILABLE", "AVAILABLE"))
        [argv] = self.det_calls("doctor")
        self.assertIn(f"--ffmpeg-skill={self.tmp}", argv, "frames always come from the agent's own ffmpeg-skill checkout")

    def test_missing_model_or_frames_is_degraded_with_the_skill_code_and_install(self):
        os.environ["FAKE_DET_READY"] = "model"
        caps = det_capabilities(lambda: det_adapter(self.tmp))
        self.assertEqual(caps["detection"].status, "AVAILABLE", "tracks / check still run: degraded, not missing")
        self.assertEqual(caps["detection:FACES"].status, "MISSING")
        self.assertEqual(caps["detection:FACES"].evidence["code"], "MODEL_NOT_FOUND")
        self.assertIn("face_detection_yunet_2026may.onnx", caps["detection:FACES"].evidence["install"])
        os.environ["FAKE_DET_READY"] = "frames"
        caps = det_capabilities(lambda: det_adapter(self.tmp))
        self.assertEqual(caps["detection:FACES"].evidence["code"], "FFMPEG_SKILL_NOT_FOUND")


class AdapterTests(_Base):
    def measure(self, params=None, faces=FACE, raw=False):
        ad = det_adapter(self.tmp)
        path = fake_media(self.tmp, duration=10.0, faces=faces)
        params = {"every": 0.5} if params is None else params
        args = {"input": path, "asset_id": "asset_a", "parameters": params} if raw else ad.measurement_args("detection/faces", "face_tracks", path, "asset_a", params, "an_1", "use")
        return ad, path, ad.measure("detection/faces", args)

    def test_argv_is_flag_value_words_and_the_input_after_double_dash(self):
        ad, path, r = self.measure({"every": 0.5, "min_score": 0.5, "window": 1.0})
        self.assertTrue(r.ok, r.data)
        faces_out = str(Path(ad.workspace) / "cache" / "detection" / "asset_a.faces.json")
        tracks_out = str(Path(ad.workspace) / "cache" / "detection" / "asset_a.tracks.json")
        self.assertEqual(self.det_calls("faces"), [["faces", "--json", "--overwrite", f"--output={faces_out}", "--every=0.5", "--min-score=0.5", f"--ffmpeg-skill={self.tmp}",
                                                     "--timeout=3600", "--", path]])
        self.assertEqual(self.det_calls("tracks"), [["tracks", "--json", "--overwrite", f"--output={tracks_out}", "--window=1", "--", faces_out]])
        [t] = r.data["tracks"]["tracks"]
        self.assertEqual(t["redact"], [{"start": 1.5, "end": 6.5, "x": 100, "y": 60, "width": 82, "height": 100}])
        self.assertEqual((r.data["detections_file"], r.data["tracks_file"]), (faces_out, tracks_out))
        self.assertEqual(r.data["detections"]["sampling"]["every"], 0.5)
        self.assertEqual(r.data["engine"]["model_sha256"], "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0")

    def test_requests_the_skill_would_refuse_never_start_a_process(self):
        for params in ({}, {"every": 0}, {"every": 61}, {"every": 0.5, "min_score": 1.5}, {"every": 0.5, "window": -1}, {"every": 0.5, "pad": 0.2}, {"every": True}):
            with self.subTest(params=params):
                _, _, r = self.measure(params, raw=True)
                self.assertFalse(r.ok)
                self.assertEqual(r.data["error"]["code"], "INVALID_REQUEST")
        self.assertEqual(self.det_calls("faces"), [], "refused before any process started")

    def test_results_that_are_not_about_the_request_are_refused(self):
        for mode, why in (("bad_fingerprint", "sha256 of the input"), ("label", "never an identity"), ("wrong_every", "requested every"),
                          ("odd_rect", "even-sized"), ("tracks_other_file", "faces file the agent wrote"), ("not_written", "on disk")):
            with self.subTest(mode=mode):
                os.environ["FAKE_DET_MODE"] = mode
                _, _, r = self.measure()
                self.assertFalse(r.ok)
                self.assertEqual(r.data["error"]["code"], "INVALID_OUTPUT")
                self.assertIn(why, r.data["error"]["message"])

    def test_skill_codes_are_kept_with_a_recovery_class(self):
        os.environ["FAKE_DET_MODE"] = "fail:FRAME_EXTRACTION_FAILED"
        _, _, r = self.measure()
        self.assertEqual((r.ok, r.data["error"]["code"], r.data["error"]["recovery_class"], r.exit_code), (False, "FRAME_EXTRACTION_FAILED", "SKILL_ERROR", 4))

    def test_never_runs_as_a_plan_step(self):
        ad = det_adapter(self.tmp)
        r = ad.run(Operation(tool="detection/faces", args={}, inputs=[], outputs=[], kind="transform"), {})
        self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_REQUEST"))


class RequirementTests(unittest.TestCase):
    def test_vocabulary_and_defaults(self):
        self.assertEqual(parse_face_requirements({"edit.redact_faces": True}), (True, {"every": 0.5}, {"mode": "blur"}))
        self.assertEqual(parse_face_requirements({"edit.redact_faces": "true", "edit.redact_faces.mode": "pixelate", "edit.redact_faces.block_size": "16",
                                                  "edit.redact_faces.every": 0.25, "edit.redact_faces.min_score": 0.5, "edit.redact_faces.window": 2}),
                         (True, {"every": 0.25, "min_score": 0.5, "window": 2.0}, {"mode": "pixelate", "block_size": 16}))
        self.assertEqual(parse_face_requirements({"edit.redact_faces": False}), (False, {}, {}))

    def test_refused_before_analysis(self):
        for reqs, why in (({"edit.redact_faces.mode": "blur"}, "not set to true"), ({"edit.redact_faces": "maybe"}, "switch"),
                          ({"edit.redact_faces": True, "edit.redact_faces.mode": "delogo"}, "a face moves"),
                          ({"edit.redact_faces": True, "edit.redact_faces.block_size": 8}, "pixelate only"),
                          ({"edit.redact_faces": True, "edit.redact_faces.every": 10}, "from 0.04 to 5"),
                          ({"edit.redact_faces": True, "edit.redact_faces.pad": 0.2}, "unknown requirement key")):
            with self.subTest(reqs=reqs):
                with self.assertRaises(EditRequirementError) as cm:
                    parse_face_requirements(reqs)
                self.assertIn(why, str(cm.exception))
                with self.assertRaises(ValueError):
                    _check_requirement_keys(reqs)

    def test_a_brief_may_ask_for_it(self):
        self.assertIn("edit.redact_faces", BRIEF_REQUIREMENT_KEYS)
        self.assertIn("edit.redact_faces.mode", BRIEF_REQUIREMENT_KEYS)


class RegionTests(unittest.TestCase):
    def test_overlapping_rectangles_cut_the_timeline_into_elementary_segments(self):
        tracks = [{"redact": [{"start": 1.5, "end": 6.5, "x": 100, "y": 60, "width": 82, "height": 100}]},
                  {"redact": [{"start": 3.5, "end": 9.5, "x": 600, "y": 200, "width": 50, "height": 60}]}]
        a, b = {"x": 100, "y": 60, "width": 82, "height": 100}, {"x": 600, "y": 200, "width": 50, "height": 60}
        self.assertEqual(elementary_segments(redaction_regions(tracks)),
                         [{"start": 1.5, "end": 3.5, "rects": [a]}, {"start": 3.5, "end": 6.5, "rects": [a, b]}, {"start": 6.5, "end": 9.5, "rects": [b]}])

    def test_an_identical_rectangle_of_two_tracks_is_blurred_once_and_gaps_are_left_alone(self):
        r = {"x": 10, "y": 10, "width": 20, "height": 20}
        tracks = [{"redact": [dict(start=0.0, end=2.0, **r)]}, {"redact": [dict(start=1.0, end=2.0, **r), dict(start=5.0, end=6.0, **r)]}]
        self.assertEqual(elementary_segments(redaction_regions(tracks)),
                         [{"start": 0.0, "end": 1.0, "rects": [r]}, {"start": 1.0, "end": 2.0, "rects": [r]}, {"start": 5.0, "end": 6.0, "rects": [r]}])


class PlanTests(_Base):
    def plan(self, faces=FACE, reqs=None, installed=True, duration=16.0, **media):
        self.svc = det_service(self.tmp, installed=installed, skills=["video-editing"])
        self.media_path = fake_media(self.tmp, duration=duration, faces=faces, **media)
        return self.svc.plan([self.media_path], "generic", user_requirements=dict({"edit.redact_faces": True}, **(reqs or {})))

    @staticmethod
    def face_decisions(doc):
        return [d for d in doc["decisions"] if d["subject"] in ("video.redact_faces", "edit.redact_faces")]

    def save(self, ir):
        p = str(Path(self.tmp) / "p.json")
        save_ir(ir, p)
        return p

    def test_measured_faces_become_a_confirm_proposal_with_three_steps(self):
        ir = self.plan(faces=TWO)
        d = ir.doc
        aid = next(iter(d["assets"]))
        obs = next(o for o in d["analysis"]["observations"] if o["kind"] == "face_tracks")
        self.assertEqual((obs["provenance"], obs["tool"], obs["parameters"]["every"]), ("OBSERVED", "detection/faces", 0.5))
        self.assertEqual(len(obs["data"]["tracks"]), 2)
        [dec] = self.face_decisions(d)
        self.assertEqual((dec["type"], dec["approval"], dec["status"], dec["provenance"], dec["risk"]), ("TRANSFORM", "CONFIRM", "PROPOSED", "OBSERVED", "HIGH"))
        self.assertEqual([(r["start"], r["end"], len(r["rects"])) for r in dec["params"]["regions"]], [(1.5, 3.5, 1), (3.5, 6.5, 2), (6.5, 9.5, 1)])
        self.assertIn(obs["id"], dec["evidence"])
        self.assertEqual(dec["basis"]["approval"]["key"], "video.redact_faces.approval")
        steps = {s["skill"]: s for s in d["plan"]["steps"] if s["skill"].startswith("face_redact")}
        self.assertEqual({k: v["tool"] for k, v in steps.items()},
                         {"face_redact_segment": "ffmpeg-skill/cut", "face_redact_blur": "ffmpeg-skill/redact", "face_redact_composite": "ffmpeg-skill/broll"})
        self.assertEqual((steps["face_redact_segment"]["params"]["segments"], steps["face_redact_blur"]["params"]["rects"]), (3, 4))
        [op] = [o for o in d["video"]["operations"] if o["type"] == "video.redact_faces"]
        self.assertEqual((op["asset"], op["input"], op["output"], op["mode"]), (aid, aid, f"{aid}_redact_faces", "blur"))
        self.assertNotEqual(d["plan"]["status"], "APPROVED", "a blur waits for the user's approval")
        self.assertEqual(self.svc.validate(ir).errors, [])

    def test_explicit_request_and_auto_policy_do_not_waive_the_confirmation(self):
        tmp_policy = Path(self.tmp) / "policy.json"
        ir = self.plan(reqs={"edit.redact_faces.mode": "pixelate"})
        [dec] = self.face_decisions(ir.doc)
        self.assertEqual(dec["approval"], "CONFIRM", "the explicit requirement is not the confirmation (irreversible, a detector misses faces)")
        self.assertFalse(tmp_policy.exists())

    def test_never_applied_without_approval_then_cut_redact_broll_before_the_trim(self):
        ir = self.plan(faces=TWO)
        p = self.save(ir)
        out = self.svc.render(load_ir(p), p)
        self.assertNotEqual(out["status"], "COMPLETED", out)
        self.assertFalse(any(c.tool in ("ffmpeg-skill/redact", "ffmpeg-skill/broll") for c in self.svc._adapter.adapters[0].calls))
        [dec] = self.face_decisions(load_ir(p).doc)
        self.svc.approve(load_ir(p), p, [dec["id"]])
        out = self.svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED", out)
        engine = self.svc._adapter.adapters[0]
        seq = [c.tool.split("/", 1)[1] for c in engine.calls if c.tool in ("ffmpeg-skill/cut", "ffmpeg-skill/redact", "ffmpeg-skill/broll")]
        self.assertEqual(seq[:8], ["cut", "redact", "cut", "redact", "redact", "cut", "redact", "broll"])
        cuts = [c.args for c in engine.calls if c.tool == "ffmpeg-skill/cut" and c.args.get("output", "").startswith(next(iter(load_ir(p).doc["assets"])) + "_faces_seg")]
        self.assertEqual([(c["segments"], c["accurate"]) for c in cuts], [("1.500-3.500", True), ("3.500-6.500", True), ("6.500-9.500", True)])
        broll = next(c.args for c in engine.calls if c.tool == "ffmpeg-skill/broll")
        self.assertEqual((broll["at"], broll["end"], broll["audio"]), (["1.500", "3.500", "6.500"], ["3.500", "6.500", "9.500"], "a"))
        redacts = [c.args for c in engine.calls if c.tool == "ffmpeg-skill/redact"]
        self.assertEqual({(r["x"], r["y"], r["width"], r["height"]) for r in redacts}, {(100, 60, 82, 100), (600, 200, 50, 60)})
        trims = [c for c in engine.calls if c.tool in ("ffmpeg-skill/cut", "video-editing/cut") and not c.args.get("output", "").count("_faces_seg")]
        for t in trims:   # the silence trim (if any) reads the composite, never the unblurred source
            self.assertTrue(str(t.args.get("input")).endswith("_redact_faces"), t.args)
        prov = json.loads((Path(self.svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        self.assertTrue(all(dec["id"] in e["decision"] for e in prov["operations"] if e["tool"] in ("ffmpeg-skill/redact", "ffmpeg-skill/broll")), "every blur cites the approved decision")

    def test_no_face_found_keeps_the_picture_and_says_so(self):
        ir = self.plan(faces=[])
        [dec] = self.face_decisions(ir.doc)
        self.assertEqual((dec["type"], dec["approval"]), ("KEEP", "AUTO"))
        self.assertIn("check the picture before publishing", dec["reason"])
        self.assertFalse([o for o in ir.doc["video"]["operations"] if o["type"] == "video.redact_faces"])
        self.assertEqual(self.svc.validate(ir).errors, [])

    def test_without_detection_skill_the_measurement_is_handed_off(self):
        ir = self.plan(installed=False)
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        [h] = [h for h in self.svc.handoffs(ir) if h["skill"] == "face_detection"]
        self.assertEqual((h["step"], h["provided_by"], h["repositories"]), ("step_job_face_detection", ["detection"], ["detection-skill"]))
        self.assertTrue(any("detection-skill" in c for c in h["install"]))
        self.assertFalse([o for o in ir.doc["analysis"]["observations"] if o["kind"] == "face_tracks"])

    def test_model_missing_hands_off_with_the_skill_code(self):
        os.environ["FAKE_DET_READY"] = "model"
        ir = self.plan()
        [blk] = [d for d in self.face_decisions(ir.doc) if d["type"] == "BLOCK"]
        self.assertIn("detection:FACES", blk["reason"])
        self.assertEqual(self.det_calls("faces"), [], "not run when its doctor says FACES is not ready")
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")

    def test_a_failed_measurement_blocks_with_the_skill_code(self):
        os.environ["FAKE_DET_MODE"] = "fail:ENGINE_FAILED"
        ir = self.plan()
        [blk] = [d for d in self.face_decisions(ir.doc) if d["type"] == "BLOCK"]
        self.assertEqual(blk["params"]["code"], "ENGINE_FAILED")
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")

    def test_too_many_segments_block(self):
        many = [{"start": float(i), "end": float(i) + 0.3, "box": [10.0 + 20 * (i % 30), 10.0 + 30 * (i // 30), 10.0, 10.0]} for i in range(0, MAX_SEGMENTS + 5)]
        ir = self.plan(faces=many, duration=float(MAX_SEGMENTS + 10), reqs={"edit.redact_faces.every": 1.0}, silences=[[0.0, 1.0]])
        [blk] = [d for d in self.face_decisions(ir.doc) if d["type"] == "BLOCK"]
        self.assertIn("coarser edit.redact_faces.window", blk["reason"])

    def test_tampered_ir_is_refused_by_the_validator(self):
        ir = self.plan()
        op = next(o for o in ir.doc["video"]["operations"] if o["type"] == "video.redact_faces")
        op["regions"][0]["rects"][0]["width"] = 81
        errs = self.svc.validate(ir).errors
        self.assertTrue(any("even-sized" in e for e in errs), errs)
        op["regions"][0]["rects"][0]["width"] = 82
        op["regions"][0]["start"], op["regions"][0]["end"] = 7.0, 3.0
        errs = self.svc.validate(ir).errors
        self.assertTrue(any("not inside the source" in e for e in errs), errs)
        op["regions"][0]["start"], op["regions"][0]["end"] = 1.5, 6.5
        ir.doc["plan"]["steps"] = [s for s in ir.doc["plan"]["steps"] if s["skill"] != "face_redact_composite"]
        errs = self.svc.validate(ir).errors
        self.assertTrue(any("face_redact_composite has no plan step" in e for e in errs), errs)

    def test_rejected_proposal_is_dropped_on_revise(self):
        ir = self.plan()
        p = self.save(ir)
        [dec] = self.face_decisions(ir.doc)
        self.svc.reject(load_ir(p), p, [dec["id"]], reason="these faces are speakers who agreed to be filmed")
        self.svc.revise(load_ir(p), p)
        v2 = load_ir(p).doc
        self.assertFalse([d for d in v2["decisions"] if d["subject"] == "video.redact_faces" and d["status"] != "REJECTED"])
        self.assertFalse([s for s in v2["plan"]["steps"] if s["skill"].startswith("face_redact")])


if __name__ == "__main__":
    unittest.main()
