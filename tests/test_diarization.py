"""ADR-083 (roadmap step 9 diarization; diarization-skill's roadmap step 5, agent side): diarization-skill's DIARIZE as the analysis kind
`speaker_turns`, asked for explicitly (`speakers.diarize`), against a fake `diarization` process (tests/fake_diarization.py) that speaks
diarization-skill main's contract, doctor and turns document, with the fake media engine. No engine, ffmpeg or network."""
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
from video_agent.agent.speakers import parse_speaker_requirements  # noqa: E402
from video_agent.capabilities.resolver import CapabilityResolver  # noqa: E402
from video_agent.models import Operation  # noqa: E402
from video_agent.service import _check_requirement_keys  # noqa: E402
from video_agent.tools.diarization import (ERROR_CLASSES, ContractError, DiarizationAdapter, breaking_drift, check_contract, drift, locate_diarization,  # noqa: E402
                                           pinned_contract, recovery_class)
from video_agent.tools.diarization.adapter import PINNED_CONTRACT_PATH  # noqa: E402
from video_agent.tools.discovery import MODULE_BOOTSTRAP  # noqa: E402
from video_agent.tools.skill_process import CliSkill  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_diarization.py")
MODES = ("FAKE_DIA_MODE", "FAKE_DIA_READY", "FAKE_DIA_CONTRACT", "FAKE_DIA_CALLS", "VIDEO_AGENT_DIARIZATION_DIR")


def fake_skill() -> CliSkill:
    return CliSkill("diarization", [sys.executable, FAKE], None, {})


def dia_adapter(tmp: str) -> DiarizationAdapter:
    ws = str(Path(tmp) / "ws")
    os.makedirs(ws, exist_ok=True)
    return DiarizationAdapter(fake_skill(), workspace=ws, allowed_inputs=[str(Path(tmp) / "src"), ws])


def dia_capabilities(factory) -> dict:
    caps: dict = {}
    CapabilityResolver._finishing_skill(CapabilityResolver.__new__(CapabilityResolver), caps, "diarization", fake_skill(), lambda _sk: factory(),
                                        "VIDEO_AGENT_DIARIZATION_DIR", "diarization-skill", "diarization",
                                        per_item=lambda ad, doc: (ad.operation_status(doc), {}), item_evidence=lambda ad, doc: ad.operation_evidence(doc))
    return caps


class DiaCaps(PipelineCaps):
    def __init__(self, dia: dict, **kw):
        super().__init__(**kw)
        self.dia = dia

    def resolve(self, refresh=False):
        out = super().resolve(refresh)
        out.update(self.dia)
        return out


def dia_service(tmp: str, installed: bool = True):
    caps = DiaCaps(dia_capabilities(lambda: dia_adapter(tmp)) if installed else {})
    svc = pipeline_service(tmp, caps=caps)
    if installed:
        svc._adapter.register(dia_adapter(tmp))
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
        self.log = str(Path(self.tmp) / "dia_calls.log")
        os.environ["FAKE_DIA_CALLS"] = self.log

    def tearDown(self):
        clear_modes()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def diarize_calls(self):
        return [c for c in calls(self.log) if c and c[0] == "diarize"]


class ContractTests(_Base):
    def test_pin_shape_and_contract_checks(self):
        c = pinned_contract()
        raw = PINNED_CONTRACT_PATH.read_text(encoding="utf-8")
        self.assertEqual(raw, json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n", "one sorted line (no release tag: pinned from main)")
        self.assertEqual((c["skill_id"], c["version"], c["contract_version"]), ("diarization", "0.1.0", "1.0"))
        self.assertEqual(check_contract(c), [])
        self.assertEqual(drift(c), [])
        self.assertEqual(breaking_drift(c), [])
        self.assertEqual(set(ERROR_CLASSES), set(c["errors"]), "every diarization-skill code has a recovery class")
        self.assertEqual((recovery_class("PIPELINE_NOT_FOUND"), recovery_class("ENGINE_TIMEOUT"), recovery_class("NEW_CODE")), ("TOOL_MISSING", "TIMEOUT", "SKILL_ERROR"))

    def test_refused_contracts(self):
        c = pinned_contract()
        bad = json.loads(json.dumps(c)); bad["version"] = "0.2.0"
        self.assertTrue(any("supported range" in e for e in check_contract(bad)))
        bad = json.loads(json.dumps(c)); next(t for t in bad["tools"] if t["tool_id"] == "diarization/diarize")["writes_media"] = True
        self.assertTrue(any("writes_media false" in e for e in check_contract(bad)))
        bad = json.loads(json.dumps(c)); bad["execution"]["network"] = True
        self.assertTrue(any("network false" in e for e in check_contract(bad)))
        bad = json.loads(json.dumps(c)); del bad["errors"]["PIPELINE_NOT_FOUND"]
        self.assertTrue(any("PIPELINE_NOT_FOUND" in e for e in check_contract(bad)))

    def test_drift_classification(self):
        os.environ["FAKE_DIA_CONTRACT"] = "drift_version"
        ad = dia_adapter(self.tmp)
        self.assertTrue(ad.drift())
        self.assertEqual(ad.breaking_drift(), [], "a version within 0.1.x is not fatal")
        os.environ["FAKE_DIA_CONTRACT"] = "drift_fatal"
        self.assertTrue(dia_adapter(self.tmp).breaking_drift(), "the diarize argv changed: fatal")
        caps = dia_capabilities(lambda: dia_adapter(self.tmp))
        self.assertEqual(caps["diarization"].status, "MISSING")
        os.environ["FAKE_DIA_CONTRACT"] = "incompatible"
        with self.assertRaises(ContractError):
            dia_adapter(self.tmp)

    def test_locate_never_searches_the_current_directory(self):
        cwd = os.getcwd()
        here = Path(self.tmp) / "cwd"
        (here / "diarization-skill" / "src" / "diarization_skill").mkdir(parents=True)
        (here / "diarization-skill" / "src" / "diarization_skill" / "cli.py").write_text("", encoding="utf-8")
        try:
            os.chdir(here)
            located = locate_diarization(env={"PATH": ""})
            self.assertTrue(located is None or str(located.root) != str(here / "diarization-skill"))
        finally:
            os.chdir(cwd)
        explicit = locate_diarization(str(here / "diarization-skill"), env={})
        self.assertEqual(explicit.root, here / "diarization-skill")
        self.assertEqual(explicit.command[1:3], ["-c", MODULE_BOOTSTRAP], "run without the current directory on sys.path (ADR-074)")
        self.assertIsNone(locate_diarization(env={"VIDEO_AGENT_DIARIZATION_DIR": str(Path(self.tmp) / "nothing"), "PATH": ""}), "the env var is authoritative")


class CapabilityTests(_Base):
    def test_ready_engine_makes_diarize_available(self):
        caps = dia_capabilities(lambda: dia_adapter(self.tmp))
        self.assertEqual((caps["diarization"].status, caps["diarization:DIARIZE"].status), ("AVAILABLE", "AVAILABLE"))

    def test_missing_pipeline_is_degraded_with_the_skill_code_and_install(self):
        os.environ["FAKE_DIA_READY"] = "pipeline"
        caps = dia_capabilities(lambda: dia_adapter(self.tmp))
        self.assertEqual(caps["diarization"].status, "AVAILABLE", "assign / check still run: degraded, not missing")
        self.assertEqual(caps["diarization:DIARIZE"].status, "MISSING")
        self.assertEqual(caps["diarization:DIARIZE"].evidence["code"], "PIPELINE_NOT_FOUND")
        self.assertIn("DIARIZATION_PIPELINE_DIR", caps["diarization:DIARIZE"].evidence["install"])
        os.environ["FAKE_DIA_READY"] = "none"
        caps = dia_capabilities(lambda: dia_adapter(self.tmp))
        self.assertEqual(caps["diarization:DIARIZE"].evidence["code"], "ENGINE_UNAVAILABLE")
        self.assertIn("diarization-skill[pyannote]", caps["diarization:DIARIZE"].evidence["install"])


class AdapterTests(_Base):
    def measure(self, params=None, **media):
        ad = dia_adapter(self.tmp)
        path = fake_media(self.tmp, duration=10.0, **media)
        return ad, path, ad.measure("diarization/diarize", ad.measurement_args("diarization/diarize", "speaker_turns", path, "asset_a", params or {}, "an_1", "use"))

    def test_argv_is_flag_value_words_and_the_media_after_double_dash(self):
        ad, path, r = self.measure({"min_speakers": 2, "max_speakers": 4, "audio_stream": 1})
        self.assertTrue(r.ok, r.data)
        [argv] = self.diarize_calls()
        out = str(Path(ad.workspace) / "cache" / "diarization" / "asset_a.turns.json")
        self.assertEqual(argv, ["diarize", "--json", "--overwrite", f"--output={out}", "--timeout=3600", "--min-speakers=2", "--max-speakers=4", "--audio-stream=1", "--", path])
        self.assertEqual(r.data["turns"]["speakers"], ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(r.data["turns_file"], out)
        self.assertIn("CC-BY-4.0", r.data["engine"]["attribution"])

    def test_requests_the_skill_would_refuse_never_start_a_process(self):
        for params in ({"num_speakers": 2, "min_speakers": 1}, {"min_speakers": 3, "max_speakers": 2}, {"num_speakers": 0}, {"num_speakers": True},
                       {"audio_stream": -1}):
            with self.subTest(params=params):
                _, _, r = self.measure(params)
                self.assertFalse(r.ok)
                self.assertEqual(r.data["error"]["code"], "INVALID_REQUEST")
        self.assertEqual(self.diarize_calls(), [])
        ad = dia_adapter(self.tmp)
        r = ad.measure("diarization/diarize", {"input": "/etc/hostname", "asset_id": "a", "parameters": {}})
        self.assertEqual(r.data["error"]["code"], "INVALID_REQUEST")
        self.assertIn("outside the allowed input roots", r.data["error"]["message"])
        r = ad.measure("diarization/diarize", {"input": fake_media(self.tmp), "asset_id": "a", "parameters": {}, "command": "rm -rf /"})
        self.assertIn("forbidden key", r.data["error"]["message"])
        r = ad.measure("diarization/diarize", {"input": fake_media(self.tmp), "asset_id": "a", "parameters": {"device": "cuda"}})
        self.assertIn("unknown diarize parameter", r.data["error"]["message"])
        self.assertEqual(self.diarize_calls(), [])

    def test_results_that_are_not_about_this_request_are_refused(self):
        for mode, words in (("bad_fingerprint", "fingerprint"), ("overlap", "exclusive"), ("wrong_params", "parameters.num_speakers"),
                            ("not_written", "on disk"), ("bad_json", "not JSON")):
            with self.subTest(mode=mode):
                os.environ["FAKE_DIA_MODE"] = mode
                _, _, r = self.measure()
                self.assertFalse(r.ok)
                self.assertEqual(r.data["error"]["code"], "INVALID_OUTPUT")
                self.assertIn(words, r.data["error"]["message"])

    def test_skill_codes_are_kept_verbatim(self):
        os.environ["FAKE_DIA_MODE"] = "fail:ENGINE_TIMEOUT"
        _, _, r = self.measure()
        self.assertEqual((r.data["error"]["code"], r.data["error"]["retryable"], r.data["error"]["recovery_class"], r.exit_code), ("ENGINE_TIMEOUT", True, "TIMEOUT", 4))
        os.environ["FAKE_DIA_MODE"] = "fail:UNSUPPORTED_MEDIA"
        _, _, r = self.measure()
        self.assertEqual((r.data["error"]["code"], r.data["error"]["retryable"], r.data["error"]["recovery_class"]), ("UNSUPPORTED_MEDIA", False, "INVALID_ARGS"))

    def test_never_a_plan_step(self):
        ad = dia_adapter(self.tmp)
        r = ad.run(Operation(tool="diarization/diarize", args={}, inputs=[], outputs=[], kind="transform"), {})
        self.assertFalse(r.ok)
        self.assertIn("never as a plan step", r.data["error"]["message"])


class RequirementTests(_Base):
    def test_keys(self):
        self.assertEqual(parse_speaker_requirements({"speakers.diarize": True, "speakers.num_speakers": "2"}), (True, {"num_speakers": 2}))
        self.assertEqual(parse_speaker_requirements({"speakers.diarize": "false"}), (False, {}))
        for bad in ({"speakers.num_speakers": 2}, {"speakers.diarize": "maybe"}, {"speakers.diarize": True, "speakers.num_speakers": 0},
                    {"speakers.diarize": True, "speakers.num_speakers": 2, "speakers.max_speakers": 3}, {"speakers.diarize": True, "speakers.min_speakers": 4, "speakers.max_speakers": 2},
                    {"speakers.diarize": True, "speakers.names": "Alice"}):
            with self.subTest(bad=bad), self.assertRaises((EditRequirementError, ValueError)):
                _check_requirement_keys(bad)

    def test_not_a_brief_key(self):
        from video_agent.service import BRIEF_REQUIREMENT_KEYS
        self.assertFalse([k for k in BRIEF_REQUIREMENT_KEYS if k.startswith("speakers.")], "--set only, like voice.*")


class PlanTests(_Base):
    def test_turns_become_an_observation_and_speaker_events(self):
        svc = dia_service(self.tmp)
        media = fake_media(self.tmp, duration=16.0, turns=[[0.5, 5.0, "SPEAKER_00"], [5.2, 11.0, "SPEAKER_01"], [11.1, 15.9, "SPEAKER_00"]])
        ir = svc.plan([media], "youtube", user_requirements={"speakers.diarize": True, "speakers.num_speakers": 2})
        [o] = [o for o in ir.doc["analysis"]["observations"] if o["kind"] == "speaker_turns"]
        self.assertEqual((o["provenance"], o["skill"], o["tool"], o["skill_version"]), ("OBSERVED", "diarization", "diarization/diarize", "0.1.0"))
        self.assertEqual(o["source"], "diarization/diarize@0.1.0")
        self.assertEqual(o["parameters"]["num_speakers"], 2)
        self.assertEqual(o["data"]["speakers"], ["SPEAKER_00", "SPEAKER_01"])
        self.assertIn("CC-BY-4.0", o["data"]["engine"]["attribution"], "the attribution travels with the fact")
        events = [e for e in ir.doc["timeline"]["events"] if e["type"] == "SPEAKER"]
        self.assertEqual([(e["range"]["start"], e["metadata"]["speaker_id"]) for e in events], [(0.5, "SPEAKER_00"), (5.2, "SPEAKER_01"), (11.1, "SPEAKER_00")])
        self.assertTrue(all(e["provenance"] == "OBSERVED" and e["evidence"] == [o["id"]] and e["metadata"]["anonymous"] for e in events))
        self.assertEqual(svc.handoffs(ir), [])
        self.assertFalse([s for s in ir.doc["plan"]["steps"] if s["skill"] == "speaker_diarization"], "a measurement, not a plan step")
        [argv] = self.diarize_calls()
        self.assertIn("--num-speakers=2", argv)
        self.assertEqual(svc.validate(ir).errors, [])

    def test_nothing_without_the_switch(self):
        svc = dia_service(self.tmp)
        ir = svc.plan([fake_media(self.tmp)], "youtube")
        self.assertFalse([o for o in ir.doc["analysis"]["observations"] if o["kind"] == "speaker_turns"])
        self.assertEqual(self.diarize_calls(), [])

    def test_engine_not_ready_hands_the_measurement_off(self):
        os.environ["FAKE_DIA_READY"] = "pipeline"
        svc = dia_service(self.tmp)
        ir = svc.plan([fake_media(self.tmp)], "youtube", user_requirements={"speakers.diarize": True})
        self.assertEqual(self.diarize_calls(), [], "a not-ready DIARIZE is never run")
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        [h] = svc.handoffs(ir)
        self.assertEqual((h["skill"], h["step"], h["provided_by"]), ("speaker_diarization", "step_job_speaker_diarization", ["diarization"]))
        block = next(d for d in ir.doc["decisions"] if d["subject"] == "speakers.diarize" and d["type"] == "BLOCK")
        self.assertIn("diarization:DIARIZE", block["reason"])

    def test_without_diarization_skill(self):
        svc = dia_service(self.tmp, installed=False)
        ir = svc.plan([fake_media(self.tmp)], "youtube", user_requirements={"speakers.diarize": True})
        [h] = svc.handoffs(ir)
        self.assertEqual((h["skill"], h["repositories"]), ("speaker_diarization", ["diarization-skill"]))

    def test_a_failed_measurement_blocks_with_the_skill_code(self):
        os.environ["FAKE_DIA_MODE"] = "fail:ENGINE_FAILED"
        svc = dia_service(self.tmp)
        ir = svc.plan([fake_media(self.tmp)], "youtube", user_requirements={"speakers.diarize": True})
        block = next(d for d in ir.doc["decisions"] if d["subject"] == "speakers.diarize" and d["type"] == "BLOCK")
        self.assertIn("ENGINE_FAILED", block["decision"])
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")

    def test_audio_less_input_is_skipped(self):
        from fake_adapter import FakeAdapter
        svc = dia_service(self.tmp)
        svc._adapter.adapters[0] = FakeAdapter(audio=False)
        ir = svc.plan([fake_media(self.tmp)], "youtube", user_requirements={"speakers.diarize": True})
        self.assertEqual(self.diarize_calls(), [])
        self.assertFalse([d for d in ir.doc["decisions"] if d["subject"] == "speakers.diarize"])


if __name__ == "__main__":
    unittest.main()
