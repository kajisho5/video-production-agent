"""ADR-113: opt-in transcription options in plans (transcribe.engine / transcribe.model / transcribe.word_segmentation), against the fake
transcription process and the fake engine: the keys and their refusals, explicit --set only, nothing sent without them, each option sent only to
a contract that declares it, the refusals before anything runs (dictionary segmentation's rule, an undeclared engine or input, the Skill's own
dry run: a model's language limit, an unknown model, a missing dictionary), what is left to the analysis, the provenance record, revisions,
and the post-render check segmenting as the source was. No real Skill, no ffmpeg."""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import clear_modes, fake_media, pipeline_service, plan_and_render  # noqa: E402

from video_agent.agent import transcribe_options as topts  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.service import BRIEF_REQUIREMENT_KEYS, _check_requirement_keys  # noqa: E402
from video_agent.tools.transcription import check_contract, pinned_contract  # noqa: E402

ENV = ("FAKE_TS_REQUESTS", "FAKE_TS_ENGINES", "FAKE_TS_CONTRACT")


class _Base(unittest.TestCase):
    def setUp(self):
        clear_modes()
        for k in ENV:
            os.environ.pop(k, None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.log = str(Path(self.tmp) / "ts_requests.log")
        os.environ["FAKE_TS_REQUESTS"] = self.log
        self.src = fake_media(self.tmp, "talk.mp4")

    def tearDown(self):
        clear_modes()
        for k in ENV:
            os.environ.pop(k, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def requests(self):
        return [json.loads(x) for x in Path(self.log).read_text(encoding="utf-8").splitlines()] if os.path.exists(self.log) else []

    def runs(self):   # recognitions, not dry runs
        return [r for r in self.requests() if not r.get("dry_run")]

    def plan(self, reqs, kinds=("transcript",), params=None, svc=None):
        svc = svc or pipeline_service(self.tmp)
        return svc, svc.plan([self.src], "generic", user_requirements=reqs, kinds=list(kinds), params=params if params is not None else {"language": "ja"})

    def refused(self, reqs, *needles, kinds=("transcript",), params=None):
        with self.assertRaises(ValueError) as cm:
            self.plan(reqs, kinds=kinds, params=params)
        for n in needles:
            self.assertIn(n, str(cm.exception))
        self.assertEqual(self.runs(), [], "refused before anything is recognised")
        return str(cm.exception)


class RequirementTests(_Base):
    def test_keys_values_and_the_set_gate(self):
        self.assertEqual(topts.parse_transcribe_requirements({}), None)
        self.assertEqual(topts.parse_transcribe_requirements({"transcribe.engine": "qwen3_asr", "transcribe.model": "parakeet-tdt-0.6b-v3",
                                                              "transcribe.word_segmentation": "Dictionary"}),
                         {"engine": "qwen3_asr", "model": "parakeet-tdt-0.6b-v3", "word_segmentation": "dictionary"})
        _check_requirement_keys({"transcribe.model": "distil-large-v3.5"})
        for bad, needle in (({"transcribe.engines": "x"}, "unknown requirement key"), ({"transcribe.engine": "../speechcpp"}, "engine id"),
                            ({"transcribe.model": "/models/parakeet.bin"}, "never a path"), ({"transcribe.model": "a/b"}, "never a path"),
                            ({"transcribe.word_segmentation": "words"}, "engine | dictionary"), ({"transcribe.engine": ""}, "must be a name"),
                            ({"transcribe.engine": 3}, "must be a name")):
            with self.assertRaises(ValueError, msg=bad) as cm:
                _check_requirement_keys(bad)
            self.assertIn(needle, str(cm.exception))

    def test_explicit_only_never_a_brief_or_a_stored_preference(self):
        self.assertFalse(set(topts.KEYS) & set(BRIEF_REQUIREMENT_KEYS), "the recogniser is the user's explicit choice: not a brief's, not a stored preference's")

    def test_transcription_skill_0_11_1_is_taken_by_the_existing_gate(self):
        c = pinned_contract()
        for v in ("0.11.0", "0.11.1"):   # 0.11.1: qwen3_asr's cache identity gains `words`; the contract is otherwise the 0.11.0 one
            self.assertEqual(check_contract(dict(c, version=v)), [], v)
        self.assertTrue(check_contract(dict(c, version="0.13.0")))   # 0.12.0 (ADR-107 addendum 15: + find) is inside the gate


class PlanTests(_Base):
    def test_defaults_unchanged(self):
        _svc, ir = self.plan({})
        [run] = self.runs()
        self.assertFalse({"engine", "model", "word_segmentation"} & set(run), run)
        self.assertNotIn("transcription_options", ir.doc["provenance"])
        self.assertEqual(len(self.requests()), 1, "no dry run without the options")

    def test_engine_and_model_reach_the_skill_and_provenance(self):
        os.environ["FAKE_TS_ENGINES"] = "faster_whisper,qwen3_asr"
        _svc, ir = self.plan({"transcribe.engine": "qwen3_asr", "transcribe.model": "small"})
        dry, run = self.requests()
        self.assertTrue(dry["dry_run"]); self.assertNotIn("dry_run", run)
        for r in (dry, run):
            self.assertEqual((r["engine"], r["model"], r["language"]), ("qwen3_asr", "small", "ja"))
        o = next(o for o in ir.doc["analysis"]["observations"] if o["kind"] == "transcript")
        self.assertEqual((o["parameters"]["engine"], o["parameters"]["model"]), ("qwen3_asr", "small"))
        rec = ir.doc["provenance"]["transcription_options"]
        self.assertEqual((rec["rule"], rec["requested"], rec["source"], rec["skill"], rec["skill_version"]),
                         ("transcribe_options@1", {"engine": "qwen3_asr", "model": "small"}, "--set", "transcription-skill", "0.12.1"))
        self.assertEqual((rec["checked"]["status"], rec["checked"]["input"]), ("accepted", "talk.mp4"))
        self.assertEqual(rec["measured"], [{"asset_id": o["asset_id"], "status": "OK", "engine": "qwen3_asr", "model": "small"}])
        reqs = {r["key"]: (r["value"], r["provenance"]) for r in ir.doc["requirements"]}
        self.assertEqual(reqs["transcribe.engine"], ("qwen3_asr", "USER"))

    def test_dictionary_segmentation_with_japanese_and_word_timings(self):
        _svc, ir = self.plan({"transcribe.word_segmentation": "dictionary", "verify.transcript": True})
        run = self.runs()[0]
        self.assertEqual((run["word_segmentation"], run["language"], run["word_timestamps"]), ("dictionary", "ja", True))
        o = next(o for o in ir.doc["analysis"]["observations"] if o["kind"] == "transcript")
        self.assertEqual(o["parameters"]["word_segmentation"], "dictionary")
        self.assertEqual([w.get("pos") for w in o["data"]["segments"][0]["words"]], ["noun", "particle"])
        self.assertEqual(ir.doc["provenance"]["transcription_options"]["measured"][0]["word_segmentation"], "dictionary")
        # the explicit default is the Skill's own: sent as asked, and the Skill records nothing (its default)
        self.tearDown(); self.setUp()
        _svc, ir = self.plan({"transcribe.word_segmentation": "engine"})
        self.assertEqual(self.runs()[0]["word_segmentation"], "engine")
        self.assertEqual(ir.doc["provenance"]["transcription_options"]["measured"][0]["word_segmentation"], "engine")

    def test_refused_before_anything_runs(self):
        self.refused({"transcribe.engine": "faster_whisper"}, "nothing in this plan recognises speech", kinds=())
        self.refused({"transcribe.word_segmentation": "dictionary", "verify.transcript": True}, "dictionary_segmentation_language", "--language ja", params={"language": "en"})
        self.refused({"transcribe.word_segmentation": "dictionary", "verify.transcript": True}, "dictionary_segmentation_language", params={})
        self.refused({"transcribe.word_segmentation": "dictionary"}, "word_timestamps_required", "edit.word_cut")
        self.refused({"transcribe.engine": "faster_whisper"}, "disagree", params={"language": "ja", "engine": "whisperx"})
        self.assertEqual(self.requests(), [], "no process at all: the plan and the contract decide these")
        # each option only to a contract that declares it
        self.refused({"transcribe.engine": "qwen3_asr"}, "declares no such engine", "faster_whisper")
        os.environ["FAKE_TS_CONTRACT"] = "0.8"
        self.refused({"transcribe.word_segmentation": "dictionary", "verify.transcript": True}, "0.8.0 does not take word_segmentation", "0.9.0 or later")
        self.assertEqual(self.requests(), [])

    def test_the_skills_own_refusals_through_its_dry_run(self):
        msg = self.refused({"transcribe.model": "distil-large-v3.5"}, "INVALID_INPUT language_unsupported", "does not recognize language 'ja'",
                           "model distil-large-v3.5 of engine faster_whisper recognises only en")
        self.assertTrue(msg.startswith("transcribe.model: transcription-skill refuses this transcription request before anything runs"), msg)
        self.assertEqual([bool(r.get("dry_run")) for r in self.requests()], [True], "one dry run, nothing recognised")
        self.refused({"transcribe.model": "distil-large-v3.5"}, "language_detection_unsupported", "does not detect the language", params={})
        self.refused({"transcribe.model": "no-such-model"}, "MODEL_UNAVAILABLE MODEL_UNKNOWN")
        os.environ["FAKE_TS_MODE"] = "no_dictionary"
        self.refused({"transcribe.word_segmentation": "dictionary", "verify.transcript": True}, "dictionary_segmentation_unavailable", "[ja] extra")
        os.environ.pop("FAKE_TS_MODE")
        _svc, ir = self.plan({"transcribe.model": "distil-large-v3.5"}, params={"language": "en"})   # a language the model recognises
        self.assertEqual(ir.doc["provenance"]["transcription_options"]["measured"][0]["model"], "distil-large-v3.5")

    def test_what_is_not_a_refusal_is_left_to_the_analysis(self):
        os.environ["FAKE_TS_MODE"] = "engine_unavailable"
        _svc, ir = self.plan({"transcribe.model": "small"})
        rec = ir.doc["provenance"]["transcription_options"]
        self.assertEqual(rec["checked"]["status"], "left_to_the_analysis")
        self.assertIn("ENGINE_UNAVAILABLE", rec["checked"]["reasons"][0])
        self.assertEqual((rec["measured"][0]["status"], rec["measured"][0]["error"]["skill_error"]), ("FAILED", "ENGINE_UNAVAILABLE"),
                         "the analysis records it as it always has; the plan still runs")
        # which dry-run failures refuse the plan: the request itself, never where the files are or what is installed
        for err, refused in (({"code": "INVALID_INPUT", "details": {"reason": "language_unsupported"}}, True), ({"code": "INVALID_INPUT", "message": "engine-level limit"}, True),
                             ({"code": "MODEL_UNAVAILABLE", "details": {"availability": "MODEL_UNKNOWN"}}, True),
                             ({"code": "MODEL_UNAVAILABLE", "details": {"availability": "MODEL_MISSING"}}, False), ({"code": "ENGINE_UNAVAILABLE"}, False),
                             ({"code": "UNSUPPORTED_MEDIA", "message": "no audio stream"}, False), ({"code": "TRANSCRIPTION_TIMEOUT"}, False)) + tuple(
                ({"code": "INVALID_INPUT", "details": {"reason": r}}, False) for r in topts.PATH_REASONS):
            self.assertEqual(topts.preflight_refusal(err) is not None, refused, err)

    def test_the_dry_runs_share_one_budget_and_stop_at_a_time_out(self):
        """A Skill that hangs on one dry run would hang on every input: the first time-out stops them (left to the analysis, which has its own
        budget), and all of a plan's dry runs share one deadline (#150 review)."""
        from unittest import mock
        from video_agent.models import ToolResult
        from video_agent.tools.transcription import TranscriptionAdapter
        other = fake_media(self.tmp, "second.mp4")
        timeouts = []

        def hang(_self, args, timeout=600.0):
            timeouts.append(timeout)
            return ToolResult("op", "transcription/transcribe", False, 124, None, {"error": {"code": "TRANSCRIPTION_TIMEOUT", "message": "process timed out"}}, [], "", timeout)
        svc = pipeline_service(self.tmp)
        with mock.patch.object(TranscriptionAdapter, "preflight", hang):
            ir = svc.plan([self.src, other], "generic", user_requirements={"transcribe.model": "small"}, kinds=["transcript"], params={"language": "ja"})
        self.assertEqual(len(timeouts), 1, "no second dry run after a time-out")
        self.assertLessEqual(timeouts[0], topts.PREFLIGHT_BUDGET_SECONDS)
        rec = ir.doc["provenance"]["transcription_options"]
        self.assertEqual(rec["checked"]["status"], "left_to_the_analysis"); self.assertIn("TRANSCRIPTION_TIMEOUT", rec["checked"]["reasons"][0])
        # a spent budget: no dry run at all, and the record says why
        timeouts.clear()
        with mock.patch.object(topts, "PREFLIGHT_BUDGET_SECONDS", 0.5), mock.patch.object(TranscriptionAdapter, "preflight", hang):
            ir = svc.plan([self.src], "generic", user_requirements={"transcribe.model": "small"}, kinds=["transcript"], params={"language": "ja"})
        self.assertEqual(timeouts, [])
        self.assertIn("budget ran out", ir.doc["provenance"]["transcription_options"]["checked"]["reasons"][0])

    def test_installed_but_not_selected_the_contract_is_still_checked(self):
        """transcription-skill installed while the registry selects no tool for speech_transcription (e.g. its default engine is missing): the
        installed contract still refuses an undeclared option before anything runs; otherwise nothing is dry-run, and the analysis refuses the
        plan with the registry's reason, as it always has."""
        from pipeline_harness import PipelineCaps
        svc = pipeline_service(self.tmp, caps=PipelineCaps(missing=["transcription"]))
        with self.assertRaises(ValueError) as cm:
            self.plan({"transcribe.engine": "qwen3_asr"}, svc=svc)
        self.assertIn("declares no such engine", str(cm.exception))
        with self.assertRaises(RuntimeError) as cm:
            self.plan({"transcribe.engine": "faster_whisper"}, svc=svc)
        self.assertIn("speech_transcription", str(cm.exception))
        self.assertEqual(self.requests(), [], "no dry run and no recognition")

    def test_revision_keeps_the_options_and_refuses_to_change_them(self):
        svc, ir = self.plan({"transcribe.model": "small"})
        p = str(Path(self.tmp) / "p.json"); save_ir(ir, p)
        with self.assertRaises(ValueError) as cm:
            svc.revise(load_ir(p), p, user_requirements={"transcribe.model": "base"})
        self.assertIn("plan again with --set transcribe.model=base", str(cm.exception))
        res = svc.revise(load_ir(p), p, user_requirements={"transcribe.model": "small", "audio.loudness.target_lufs": -20})
        self.assertTrue(res["created"])
        self.assertEqual(load_ir(p).doc["provenance"]["transcription_options"]["requested"], {"model": "small"})


class RenderTests(_Base):
    def test_provenance_json_and_the_post_render_check_segment_as_the_source(self):
        svc = pipeline_service(self.tmp)
        res = plan_and_render(svc, [self.src], {"transcribe.word_segmentation": "dictionary", "verify.transcript": True}, kinds=["transcript"])
        self.assertEqual(res["validation"].errors, [])
        [prov] = glob.glob(str(Path(svc.workspace) / "jobs" / "*" / "provenance.json"))
        rec = json.loads(Path(prov).read_text(encoding="utf-8"))["transcription_options"]
        self.assertEqual((rec["requested"], rec["checked"]["status"]), ({"word_segmentation": "dictionary"}, "accepted"))
        delivered = [r for r in self.runs() if Path(r["input"]).resolve() != Path(self.src).resolve()]
        self.assertTrue(delivered, "the deliverable is re-transcribed (ADR-110)")
        self.assertEqual({(r.get("word_segmentation"), r["language"]) for r in delivered}, {("dictionary", "ja")},
                         "the delivered words segmented as the planned words were")
        # without the option, the check sends no segmentation (unchanged)
        self.tearDown(); self.setUp()
        svc = pipeline_service(self.tmp)
        plan_and_render(svc, [self.src], {"verify.transcript": True}, kinds=["transcript"])
        self.assertFalse(any("word_segmentation" in r for r in self.runs()))
        [prov] = glob.glob(str(Path(svc.workspace) / "jobs" / "*" / "provenance.json"))
        self.assertNotIn("transcription_options", json.loads(Path(prov).read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
