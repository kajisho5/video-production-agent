"""Roadmap step 33 (ADR-090 addendum 1, ADR-103 addendum 1): measurements asked for by name -- text on screen (`text.detect` → detection-skill
`text`, analysis kind screen_text), the shot table (`shots.detect` → detection-skill `shots`, shot_table) and speech regions
(`speech_regions.detect` → media-analysis-skill `vad`, speech_regions). Against the fake detection process (tests/fake_detection.py) and the
fake media-analysis process; `RealSkillTests` runs the real Skills where their engines are ready (skipped otherwise)."""
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
from pipeline_harness import fake_media  # noqa: E402
from test_face_redaction import _Base, det_adapter, det_capabilities, det_service  # noqa: E402

from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.measurements import parse_measurement, requested_kinds  # noqa: E402
from video_agent.media import ANALYSIS_KINDS  # noqa: E402
from video_agent.service import BRIEF_REQUIREMENT_KEYS, Service, _check_requirement_keys  # noqa: E402
from video_agent.tools.media_analysis.adapter import pinned_contract as ma_pinned  # noqa: E402

TEXTS = [{"start": 1.0, "end": 4.0, "text": "Dr. Jane Doe", "box": [100, 600, 400, 60]}, {"start": 6.0, "end": 8.0, "text": "ACME 2026", "box": [800, 40, 300, 50]}]
FAKE_MA = str(Path(__file__).resolve().parent / "fake_media_analysis.py")


class RequirementTests(unittest.TestCase):
    def test_vocabulary_and_defaults(self):
        self.assertEqual(parse_measurement({"text.detect": True}, "text.detect"), ("screen_text", {"text_every": 1.0}))
        self.assertEqual(parse_measurement({"text.detect": "true", "text.every": "0.5", "text.lang": "jpn+eng", "text.min_score": 0.4}, "text.detect"),
                         ("screen_text", {"text_every": 0.5, "text_lang": "jpn+eng", "text_min_score": 0.4}))
        self.assertEqual(parse_measurement({"shots.detect": True, "shots.threshold": 20, "shots.min_shot_frames": "10"}, "shots.detect"),
                         ("shot_table", {"shot_threshold": 20.0, "shot_min_frames": 10}))
        self.assertEqual(parse_measurement({"speech_regions.detect": True}, "speech_regions.detect"), ("speech_regions", {}))
        self.assertIsNone(parse_measurement({"text.detect": False}, "text.detect"))
        self.assertEqual(requested_kinds({}), {})
        self.assertEqual(set(requested_kinds({"text.detect": True, "shots.detect": True, "speech_regions.detect": True})),
                         {"text_detection", "shot_detection", "speech_region_analysis"})

    def test_refused_before_analysis(self):
        for reqs, why in (({"text.every": 2}, "not set to true"), ({"text.detect": "maybe"}, "switch"), ({"text.detect": True, "text.every": 0}, "(0, 60]"),
                          ({"text.detect": True, "text.lang": "japanese; rm"}, "Tesseract language codes"), ({"text.detect": True, "text.min_score": 2}, "[0, 1]"),
                          ({"shots.detect": True, "shots.min_shot_frames": 0}, "integer in [1"), ({"shots.detect": True, "shots.engine": "transnetv2"}, "unknown requirement key"),
                          ({"speech_regions.detect": True, "speech_regions.threshold": 0.4}, "unknown requirement key")):
            with self.subTest(reqs=reqs):
                with self.assertRaises(ValueError) as cm:
                    _check_requirement_keys(reqs)
                self.assertIn(why, str(cm.exception))
                self.assertIsInstance(cm.exception, EditRequirementError)

    def test_explicit_only(self):
        for k in ("text.detect", "shots.detect", "speech_regions.detect"):
            self.assertNotIn(k, BRIEF_REQUIREMENT_KEYS, "never read from a brief or a stored preference")

    def test_kind_parameters(self):
        vad = next(t for t in ma_pinned()["tools"] if t["tool_id"] == "media-analysis/vad")
        self.assertEqual((vad["kinds"], ANALYSIS_KINDS["speech_regions"]["skill"]), (["speech_regions"], "speech_region_analysis"))
        self.assertEqual(ANALYSIS_KINDS["screen_text"]["skill"], "text_detection")
        self.assertEqual(ANALYSIS_KINDS["shot_table"]["skill"], "shot_detection")


class DetectionAdapterTests(_Base):
    def measure(self, kind, params, **facts):
        ad = det_adapter(self.tmp)
        path = fake_media(self.tmp, duration=10.0, **facts)
        tool = {"screen_text": "detection/text", "shot_table": "detection/shots"}[kind]
        return ad, path, ad.measure(tool, ad.measurement_args(tool, kind, path, "asset_a", params, "an_1", "use"))

    def test_text_argv_and_result(self):
        ad, path, r = self.measure("screen_text", {"text_every": 1.0, "text_lang": "jpn+eng", "text_min_score": 0.4}, texts=TEXTS)
        self.assertTrue(r.ok, r.data)
        out = str(Path(ad.workspace) / "cache" / "detection" / "asset_a.text.json")
        self.assertEqual(self.det_calls("text"), [["text", "--json", "--overwrite", f"--output={out}", "--every=1", "--lang=jpn+eng", "--min-score=0.4",
                                                   f"--ffmpeg-skill={self.tmp}", "--timeout=3600", "--", path]])
        self.assertEqual(r.data["text_file"], out)
        labels = {d["label"] for f in r.data["text"]["frames"] for d in f["detections"]}
        self.assertEqual(labels, {"Dr. Jane Doe", "ACME 2026"})

    def test_shots_argv_and_result(self):
        ad, path, r = self.measure("shot_table", {"shot_min_frames": 10}, cuts=[4.0, 7.6])
        self.assertTrue(r.ok, r.data)
        out = str(Path(ad.workspace) / "cache" / "detection" / "asset_a.shots.json")
        self.assertEqual(self.det_calls("shots"), [["shots", "--json", "--overwrite", f"--output={out}", "--min-shot-frames=10", f"--ffmpeg-skill={self.tmp}",
                                                    "--timeout=3600", "--", path]])
        self.assertEqual([(s["start"], s["end"]) for s in r.data["shots"]["shots"]], [(0.0, 4.0), (4.0, 7.6), (7.6, 10.0)])

    def test_results_that_are_not_about_the_request_are_refused(self):
        for kind, params, mode, why in (("screen_text", {"text_every": 1.0}, "bad_fingerprint", "sha256 of the input"),
                                        ("screen_text", {"text_every": 1.0}, "wrong_every", "requested every"),
                                        ("screen_text", {"text_every": 1.0}, "not_written", "on disk"),
                                        ("shot_table", {}, "gap", "do not continue"), ("shot_table", {}, "bad_fingerprint", "sha256 of the input")):
            with self.subTest(kind=kind, mode=mode):
                os.environ["FAKE_DET_MODE"] = mode
                _, _, r = self.measure(kind, params, texts=TEXTS, cuts=[4.0])
                self.assertFalse(r.ok)
                self.assertEqual(r.data["error"]["code"], "INVALID_OUTPUT")
                self.assertIn(why, r.data["error"]["message"])

    def test_requests_the_skill_would_refuse_never_start_a_process(self):
        ad = det_adapter(self.tmp)
        path = fake_media(self.tmp, duration=10.0)
        for tool, args in (("detection/text", {"input": path, "measure": "text", "parameters": {}}),
                           ("detection/text", {"input": path, "measure": "text", "parameters": {"every": 1, "lang": "../x"}}),
                           ("detection/shots", {"input": path, "measure": "shots", "parameters": {"min_shot_frames": 0}}),
                           ("detection/shots", {"input": path, "measure": "text", "parameters": {"every": 1}}),
                           ("detection/text", {"input": "/etc/passwd", "measure": "text", "parameters": {"every": 1}})):
            with self.subTest(tool=tool, args=args):
                r = ad.measure(tool, args)
                self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_REQUEST"))
        self.assertEqual(self.det_calls("text") + self.det_calls("shots"), [])

    def test_readiness_from_the_doctor(self):
        caps = det_capabilities(lambda: det_adapter(self.tmp))
        self.assertEqual((caps["detection:TEXT"].status, caps["detection:SHOTS"].status), ("AVAILABLE", "AVAILABLE"))
        os.environ["FAKE_DET_READY"] = "no_tesseract"
        caps = det_capabilities(lambda: det_adapter(self.tmp))
        self.assertEqual((caps["detection:TEXT"].status, caps["detection:SHOTS"].status), ("MISSING", "AVAILABLE"))
        self.assertEqual(caps["detection:TEXT"].evidence["code"], "ENGINE_UNAVAILABLE")
        self.assertIn("tesseract", caps["detection:TEXT"].evidence["install"])
        os.environ["FAKE_DET_READY"] = "no_scenedetect"
        caps = det_capabilities(lambda: det_adapter(self.tmp))
        self.assertEqual(caps["detection:SHOTS"].status, "MISSING")
        self.assertIn("detection-skill[shots]", caps["detection:SHOTS"].evidence["install"])


class PlanTests(_Base):
    def plan(self, reqs, installed=True, **facts):
        self.svc = det_service(self.tmp, installed=installed, skills=[])
        return self.svc.plan([fake_media(self.tmp, duration=16.0, **facts)], "generic", user_requirements=reqs)

    def test_text_on_screen_is_measured_by_name(self):
        ir = self.plan({"text.detect": True, "text.every": 1}, texts=TEXTS)
        [o] = [o for o in ir.doc["analysis"]["observations"] if o["kind"] == "screen_text"]
        self.assertEqual((o["provenance"], o["tool"], o["parameters"]["text_every"]), ("OBSERVED", "detection/text", 1))
        self.assertEqual(sorted({ln["text"] for ln in o["data"]["lines"]}), ["ACME 2026", "Dr. Jane Doe"])
        self.assertEqual(o["data"]["sampling"]["every"], 1.0)
        self.assertEqual([h for h in self.svc.handoffs(ir) if h["skill"] in ("visual_detection", "text_detection")], [], "run, not handed off")
        self.assertNotEqual(ir.doc["plan"]["status"], "BLOCKED")

    def test_shot_table_is_measured_by_name_and_scenes_stays_the_default(self):
        ir = self.plan({"shots.detect": True}, cuts=[4.0])
        [o] = [o for o in ir.doc["analysis"]["observations"] if o["kind"] == "shot_table"]
        self.assertEqual((o["tool"], len(o["data"]["shots"]), o["data"]["frames"]["frame_rate"]), ("detection/shots", 2, 25.0))
        self.assertNotIn("scene_detection", [o["kind"] for o in ir.doc["analysis"]["observations"]], "the fast cut list is not replaced or added")

    def test_nothing_new_without_the_keys(self):
        ir = self.plan({}, texts=TEXTS, cuts=[4.0])
        self.assertFalse([o for o in ir.doc["analysis"]["observations"] if o["kind"] in ("screen_text", "shot_table", "speech_regions")])
        self.assertEqual(self.det_calls("text") + self.det_calls("shots"), [])

    def test_without_the_skill_the_measurement_is_handed_off(self):
        ir = self.plan({"text.detect": True, "shots.detect": True}, installed=False)
        hs = {h["skill"]: h for h in self.svc.handoffs(ir)}
        self.assertEqual(set(hs), {"text_detection", "shot_detection"})
        self.assertEqual(hs["text_detection"]["repositories"], ["detection-skill"])
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        self.assertNotIn("visual_detection", json.dumps(self.svc.handoffs(ir)))

    def test_an_engine_missing_blocks_with_the_doctors_reason(self):
        os.environ["FAKE_DET_READY"] = "no_tesseract"
        ir = self.plan({"text.detect": True}, texts=TEXTS)
        blocks = [d for d in ir.doc["decisions"] if d["subject"] == "text.detect" and d["approval"] == "BLOCK"]
        self.assertTrue(blocks and "detection:TEXT" in blocks[0]["reason"], blocks)
        self.assertEqual(self.det_calls("text"), [])


class SpeechRegionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def service(self, installed=True):
        from fake_adapter import FakeAdapter
        from test_unit import FakeCaps
        from video_agent.tools import ToolRouter
        from video_agent.tools.media_analysis import MediaAnalysisAdapter, MediaAnalysisSkill
        src = str(Path(self.tmp) / "src")
        adapters = [FakeAdapter()]
        if installed:
            adapters.append(MediaAnalysisAdapter(MediaAnalysisSkill([sys.executable, FAKE_MA], None, {}), workspace=self.tmp, allowed_inputs=[src]))
        return Service(workspace=self.tmp, adapter=ToolRouter(adapters), caps=FakeCaps(extra=["media-analysis"] if installed else []))

    def test_speech_regions_are_measured_by_name(self):
        from test_unit import fake_media as unit_media
        svc = self.service()
        self.assertEqual(svc.tools_for().get("speech_region_analysis"), "media-analysis/vad")
        ir = svc.plan([unit_media(self.tmp)], "generic", user_requirements={"speech_regions.detect": True})
        [o] = [o for o in ir.doc["analysis"]["observations"] if o["kind"] == "speech_regions"]
        self.assertEqual((o["provenance"], o["tool"], o["data"]["region_count"]), ("OBSERVED", "media-analysis/vad", 2))
        self.assertFalse([h for h in svc.handoffs(ir) if h["skill"] == "speech_region_analysis"])
        plain = svc.plan([unit_media(self.tmp)], "generic")
        self.assertFalse([o for o in plain.doc["analysis"]["observations"] if o["kind"] == "speech_regions"])

    def test_without_media_analysis_it_is_handed_off(self):
        from test_unit import fake_media as unit_media
        svc = self.service(installed=False)
        ir = svc.plan([unit_media(self.tmp)], "generic", user_requirements={"speech_regions.detect": True})
        [h] = [h for h in svc.handoffs(ir) if h["skill"] == "speech_region_analysis"]
        self.assertEqual(h["repositories"], ["media-analysis-skill"])
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")


class TranscriptSearchTests(unittest.TestCase):
    """ADR-107 addendum 15: `transcript.find=<phrase>` measures the transcript and searches it with transcription-skill 0.12.0's find."""

    def setUp(self):
        from pipeline_harness import clear_modes
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())

    def tearDown(self):
        os.environ.pop("FAKE_TS_MODE", None)

    def plan(self, reqs, transcription=True):
        from pipeline_harness import pipeline_service
        self.svc = pipeline_service(self.tmp, transcription=transcription, skills=[])
        return self.svc.plan([fake_media(self.tmp, duration=16.0)], "generic", user_requirements=reqs)

    def test_vocabulary(self):
        self.assertEqual(parse_measurement({"transcript.find": "ACME", "transcript.find.fuzzy": "0.8", "transcript.find.context": 2}, "transcript.find"),
                         ("transcript_hits", {"find_phrase": "ACME", "find_fuzzy": 0.8, "find_context": 2}))
        self.assertIsNone(parse_measurement({}, "transcript.find"))
        for reqs, why in (({"transcript.find": ""}, "the phrase to find"), ({"transcript.find": "x" * 501}, "1..500"),
                          ({"transcript.find.fuzzy": 0.8}, "not set to true"), ({"transcript.find": "a", "transcript.find.fuzzy": 0.2}, "[0.5, 1]"),
                          ({"transcript.find": "a", "transcript.find.context": 11}, "[0, 10]")):
            with self.subTest(reqs=reqs):
                with self.assertRaises(ValueError) as cm:
                    _check_requirement_keys(reqs)
                self.assertIn(why, str(cm.exception))
        self.assertNotIn("transcript.find", BRIEF_REQUIREMENT_KEYS)

    def test_hits_from_the_measured_transcript(self):
        ir = self.plan({"transcript.find": "会場"})
        obs = {o["kind"]: o for o in ir.doc["analysis"]["observations"]}
        self.assertIn("transcript", obs, "the transcript is measured for the search")
        hits = obs["transcript_hits"]
        self.assertEqual((hits["provenance"], hits["tool"], hits["data"]["phrase"]), ("OBSERVED", "transcription/find", "会場"))
        self.assertEqual([(h["start"], h["end"], h["segment_ids"]) for h in hits["data"]["hits"]], [(4.0, 8.4, ["seg_0002"])])
        self.assertEqual(hits["data"]["transcript_id"], obs["transcript"]["data"]["id"])
        self.assertEqual(hits["parameters"]["transcript"], obs["transcript"]["id"], "which measured transcript was searched")
        self.assertEqual(hits["data"]["count"], len(hits["data"]["hits"]))
        self.assertFalse([h for h in self.svc.handoffs(ir) if h["skill"] == "transcript_search"])

    def test_a_result_about_another_phrase_is_refused(self):
        os.environ["FAKE_TS_MODE"] = "find_other_phrase"
        ir = self.plan({"transcript.find": "welcome"})
        self.assertFalse([o for o in ir.doc["analysis"]["observations"] if o["kind"] == "transcript_hits"])
        self.assertTrue(any(d["subject"] == "transcript.find" and d["approval"] == "BLOCK" for d in ir.doc["decisions"]))

    def test_without_transcription_it_is_handed_off(self):
        ir = self.plan({"transcript.find": "welcome"}, transcription=False)
        [h] = [h for h in self.svc.handoffs(ir) if h["skill"] == "transcript_search"]
        self.assertEqual(h["repositories"], ["transcription-skill"])

    def test_adapter_refuses_before_any_process(self):
        from video_agent.tools.transcription import TranscriptionAdapter, TranscriptionSkill
        fake = str(Path(__file__).resolve().parent / "fake_transcription.py")
        ad = TranscriptionAdapter(TranscriptionSkill([sys.executable, fake], None, {}), workspace=self.tmp)
        self.assertTrue(ad.supports("transcription/find"))
        for args in ({"phrase": "a"}, {"transcript": {"schema": "x"}, "phrase": "a"}, {"transcript": {"schema": "transcription-skill/transcript/0.1", "segments": []}, "phrase": ""},
                     {"transcript": {"schema": "transcription-skill/transcript/0.1", "segments": []}, "phrase": "a", "fuzzy": 0.1},
                     {"transcript": {"schema": "transcription-skill/transcript/0.1", "segments": []}, "phrase": "a", "input": "/etc/passwd"}):
            with self.subTest(args=args):
                calls = ad.calls
                r = ad.measure("transcription/find", args)
                self.assertEqual((r.ok, r.data["error"]["code"], ad.calls), (False, "INVALID_INPUT", calls))


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class RealSkillTests(unittest.TestCase):
    """The real detection-skill (VIDEO_AGENT_DETECTION_DIR) and media-analysis-skill (VIDEO_AGENT_MEDIA_ANALYSIS_DIR) where their doctors say the
    engine is ready (Tesseract for text, PySceneDetect for shots, the vad extra for speech regions); skipped otherwise."""

    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.src = Path(self.tmp) / "src"
        self.src.mkdir()

    def service(self):
        svc = Service(workspace=str(Path(self.tmp) / "ws"))
        return svc, svc.caps.resolve()

    def test_text_and_shots_on_a_rendered_video(self):
        if not os.environ.get("VIDEO_AGENT_DETECTION_DIR"):
            self.skipTest("VIDEO_AGENT_DETECTION_DIR not set")
        v = str(self.src / "card.mp4")
        _ffmpeg("-f", "lavfi", "-i", "color=c=white:s=640x360:d=2,drawtext=text='HELLO WORLD':fontsize=60:fontcolor=black:x=100:y=150",
                "-f", "lavfi", "-i", "color=c=blue:s=640x360:d=2", "-filter_complex", "[0][1]concat=n=2:v=1:a=0,format=yuv420p", "-r", "25", v)
        svc, caps = self.service()
        want = {k: caps.get(f"detection:{t}") for k, t in (("text.detect", "TEXT"), ("shots.detect", "SHOTS"))}
        reqs = {k: True for k, c in want.items() if c is not None and c.status == "AVAILABLE"}
        if not reqs:
            self.skipTest("detection-skill's doctor has neither text nor shots ready here")
        ir = svc.plan([v], "generic", user_requirements=reqs, hash_sources=True)
        obs = {o["kind"]: o for o in ir.doc["analysis"]["observations"]}
        if "text.detect" in reqs:
            self.assertIn("HELLO WORLD", [ln["text"] for ln in obs["screen_text"]["data"]["lines"]])
        if "shots.detect" in reqs:
            self.assertEqual([(s["start"], s["end"]) for s in obs["shot_table"]["data"]["shots"]], [(0.0, 2.0), (2.0, 4.0)])

    def test_find_on_the_real_transcription_skill(self):
        """transcription-skill 0.12.0's own find (standard library, no engine) on a word-timed transcript its validator accepts."""
        if not os.environ.get("VIDEO_AGENT_TRANSCRIPTION_DIR"):
            self.skipTest("VIDEO_AGENT_TRANSCRIPTION_DIR not set")
        from video_agent.tools.transcription import TranscriptionAdapter, locate_transcription
        ad = TranscriptionAdapter(locate_transcription(), workspace=self.tmp)
        if not ad.supports("transcription/find"):
            self.skipTest(f"transcription-skill {ad.version} has no find")
        words = [[("hello", 0.0, 0.4), ("world", 0.5, 0.9)], [("hello", 2.0, 2.3), ("again", 2.4, 2.8)]]
        segs, n = [], 0
        for i, ws in enumerate(words, 1):
            w = []
            for t, a, b in ws:
                w.append({"start": a, "end": b, "text": t, "confidence": 0.9, "index": n}); n += 1
            text = " ".join(x["text"] for x in w)
            segs.append({"id": f"seg_{i:04d}", "start": w[0]["start"], "end": w[-1]["end"], "text": text, "raw_text": text, "confidence": 0.9, "words": w, "speaker_id": None})
        fp = "sha256:" + "cd" * 32
        doc = {"schema": "transcription-skill/transcript/0.1", "id": "tr_find_en", "asset_id": "asset_1", "language": "en", "language_source": "requested",
               "language_confidence": None, "duration": 3.8, "segments": segs,
               "source": {"filename": "talk.wav", "fingerprint": fp, "size_bytes": 10, "media_duration": 3.8, "audio_channels": 1, "sample_rate": 16000, "container": "wav",
                          "has_video": False},
               "engine": "fake", "engine_version": "1.0", "created_at": "2026-11-13T00:00:00Z",
               "provenance": {"engine": "fake", "engine_version": "1.0", "execution_mode": "local", "model": "fake-model", "model_version": None,
                              "parameters": {"language": "en"}, "parameters_hash": "0" * 64, "cache_key": "1" * 64, "created_at": "2026-11-13T00:00:00Z",
                              "processing_seconds": 0.5, "skill_version": ad.version, "skill": "transcription-skill", "tool": "transcription/transcribe",
                              "language_detection": None, "audio_extraction": None},
               "warnings": []}
        r = ad.measure("transcription/find", {"transcript": doc, "phrase": "Hello", "context": 0})
        self.assertTrue(r.ok, r.data)
        self.assertEqual([(h["start"], h["end"], h["word_indices"]) for h in r.data["hits"]["hits"]], [(0.0, 0.4, [0]), (2.0, 2.3, [2])])

    def test_speech_regions_on_a_rendered_file(self):
        if not os.environ.get("VIDEO_AGENT_MEDIA_ANALYSIS_DIR"):
            self.skipTest("VIDEO_AGENT_MEDIA_ANALYSIS_DIR not set")
        a = str(self.src / "tone.wav")
        _ffmpeg("-f", "lavfi", "-i", "sine=f=300:d=2", a)
        svc, _ = self.service()
        if svc.tools_for().get("speech_region_analysis") != "media-analysis/vad":
            self.skipTest("media-analysis/vad is not selectable here")
        ir = svc.plan([a], "generic", user_requirements={"speech_regions.detect": True})
        rows = [r for an in ir.doc["analysis"].get("runs", []) or [] for r in an.get("rows") or []]
        obs = [o for o in ir.doc["analysis"]["observations"] if o["kind"] == "speech_regions"]
        if not obs:   # the vad extra is optional: the Skill answers UNKNOWN without it, and the plan blocks with its code
            self.assertTrue(any(d["subject"] == "speech_regions.detect" and d["approval"] == "BLOCK" for d in ir.doc["decisions"]), rows)
            self.skipTest("media-analysis-skill's vad extra is not installed here")
        if obs[0]["data"].get("status") != "MEASURED":
            self.skipTest(f"media-analysis-skill answered {obs[0]['data'].get('status')} (its vad extra is not installed here)")
        self.assertEqual(obs[0]["tool"], "media-analysis/vad")
        self.assertIn("regions", obs[0]["data"])


if __name__ == "__main__":
    unittest.main()
