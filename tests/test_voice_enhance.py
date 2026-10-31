"""ADR-073 (roadmap step 9 voice-enhance; voice-enhance-skill's roadmap step 8, agent side): voice-enhance-skill's ISOLATE / DEREVERB /
BLEED_REDUCE as opt-in, always-CONFIRM operations on the audio production path, after each input's cut / filler removal and before every
audio-production operation. Against a fake `voice-enhance` process (tests/fake_voice_enhance.py) that speaks voice-enhance-skill main's
contract, doctor and result shapes, with the fake media engine and the fake audio-production process. No engine, ffmpeg or network."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import AUDIO_PRODUCTION_CAPS, PipelineCaps, clear_modes, pipeline_service, plan_and_render  # noqa: E402
from pipeline_harness import fake_media as pipeline_media  # noqa: E402

from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.voice import APPROVAL_KEYS, OPERATIONS, VOICE_ORDER, check_voice_keys, parse_voice_requirements  # noqa: E402
from video_agent.capabilities.resolver import Capability, CapabilityResolver  # noqa: E402
from video_agent.execution.compiler import CompileError, lower_voice_op  # noqa: E402
from video_agent.execution.executor import Executor  # noqa: E402
from video_agent.tools.discovery import MODULE_BOOTSTRAP  # noqa: E402
from video_agent.models import Operation, Requirement, ToolResult  # noqa: E402
from video_agent.project.validator import check_audio_operations  # noqa: E402
from video_agent.service import BRIEF_REQUIREMENT_KEYS, _check_requirement_keys  # noqa: E402
from video_agent.tools.skill_process import CliSkill  # noqa: E402
from video_agent.tools.voice_enhance import (ERROR_CLASSES, FATAL_DRIFT_KEYS, ContractError, VoiceEnhanceAdapter, breaking_drift, check_contract,  # noqa: E402
                                             drift, locate_voice_enhance, pinned_contract, recovery_class)
from video_agent.tools.voice_enhance.adapter import PINNED_CONTRACT_PATH  # noqa: E402
import fake_voice_enhance  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_voice_enhance.py")
MODES = ("FAKE_VOICE_MODE", "FAKE_VOICE_ENGINES", "FAKE_VOICE_RESAMPLE", "FAKE_VOICE_CONTRACT", "FAKE_VOICE_CALLS", "VIDEO_AGENT_VOICE_ENHANCE_DIR")
AP_OPS = ("CUT", "MIX", "NOISE_REDUCTION", "NORMALIZE", "GAIN", "CONCAT")


def fake_skill() -> CliSkill:
    return CliSkill("voice-enhance", [sys.executable, FAKE], None, {})


def voice_adapter(tmp: str, **kw) -> VoiceEnhanceAdapter:
    ws = str(Path(tmp) / "ws")
    os.makedirs(ws, exist_ok=True)
    return VoiceEnhanceAdapter(fake_skill(), workspace=ws, allowed_inputs=[str(Path(tmp) / "src"), ws], ffmpeg_skill_dir=tmp, **kw)


def voice_capabilities(adapter_factory) -> dict:
    """What the resolver reports for a located voice-enhance-skill: `voice-enhance` and `voice-enhance:<TYPE>` (with the Skill's own codes)."""
    caps: dict = {}
    CapabilityResolver._finishing_skill(CapabilityResolver.__new__(CapabilityResolver), caps, "voice-enhance", fake_skill(), lambda _sk: adapter_factory(),
                                        "VIDEO_AGENT_VOICE_ENHANCE_DIR", "voice-enhance-skill", "voice-enhance",
                                        per_item=lambda ad, doc: (ad.operation_status(doc), {}), item_evidence=lambda ad, doc: ad.operation_evidence(doc))
    return caps


class VoiceCaps(PipelineCaps):
    """The pipeline's capabilities, audio-production's operations, and voice-enhance as the resolver reports the fake (or nothing)."""

    def __init__(self, voice: dict, **kw):
        super().__init__(extra=list(AUDIO_PRODUCTION_CAPS) + [f"audio-production:{t}" for t in AP_OPS], **kw)
        self.voice = voice

    def resolve(self, refresh=False):
        out = super().resolve(refresh)
        out.update(self.voice)
        return out


def voice_service(tmp: str, installed: bool = True):
    """The pipeline harness (fake engine, fake audio-production, ...) plus the fake voice-enhance adapter and its resolved capabilities."""
    caps = VoiceCaps(voice_capabilities(lambda: voice_adapter(tmp)) if installed else {})
    svc = pipeline_service(tmp, caps=caps, silences=[[0.0, 1.0], [13.7, None]])
    if installed:
        svc._adapter.register(voice_adapter(tmp))
        svc.adapter([])
    return svc


def wav(tmp: str, name: str = "talk.wav", channels: int = 2, rate: int = 48000, duration: float = 16.0) -> str:
    return pipeline_media(tmp, name, video=False, channels=channels, duration=duration, sample_rate=rate)


def calls(log: str):
    return [json.loads(x) for x in Path(log).read_text(encoding="utf-8").splitlines()] if os.path.exists(log) else []


def reqs(d):
    return {k: Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in d.items()}


class _Base(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self._env = {k: os.environ.get(k) for k in MODES}
        for k in MODES:
            os.environ.pop(k, None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.log = str(Path(self.tmp) / "voice_calls.log")
        os.environ["FAKE_VOICE_CALLS"] = self.log

    def tearDown(self):
        clear_modes()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def op_calls(self):
        return [c for c in calls(self.log) if c and c[0] in ("isolate", "dereverb", "bleed-reduce")]


class ContractTests(_Base):
    def test_pin_shape_and_contract_checks(self):
        c = pinned_contract()
        raw = PINNED_CONTRACT_PATH.read_text(encoding="utf-8")
        self.assertEqual(raw, json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n", "one sorted line, like ingest's pin (no release tag: pinned from main)")
        self.assertEqual((c["skill_id"], c["version"], c["contract_version"]), ("voice-enhance", "0.1.0", "1.0"))
        self.assertEqual(check_contract(c), [])
        self.assertEqual({p["id"]: p["lifecycle"] for p in c["provides"]}, {"voice.isolate": "EXPERIMENTAL", "voice.dereverb": "EXPERIMENTAL", "voice.bleed_reduce": "EXPERIMENTAL"})
        self.assertEqual(set(ERROR_CLASSES), set(c["errors"]), "every voice-enhance error code has a recovery class")
        self.assertEqual(recovery_class("ENGINE_UNAVAILABLE"), "TOOL_MISSING")
        self.assertEqual(recovery_class("SOMETHING_NEW"), "SKILL_ERROR")
        for bad, msg in ((dict(c, version="0.2.0"), "supported range"), (dict(c, skill_id="x"), "skill_id"),
                         (dict(c, execution=dict(c["execution"], shell=True)), "shell false"),
                         (dict(c, tools=[t for t in c["tools"] if t["tool_id"] != "voice-enhance/isolate"]), "lacks tool voice-enhance/isolate"),
                         (dict(c, errors={k: v for k, v in c["errors"].items() if k != "MODEL_NOT_FOUND"}), "MODEL_NOT_FOUND")):
            self.assertTrue(any(msg in e for e in check_contract(bad)), (msg, check_contract(bad)))

    def test_drift_classification(self):
        self.assertEqual(FATAL_DRIFT_KEYS, ("skill_id", "contract_version", "execution", "error_shape"))
        for mode, fatal in (("pinned", None), ("drift_version", False), ("drift_lifecycle", False), ("drift_fatal", True), ("drift_error", True)):
            os.environ["FAKE_VOICE_CONTRACT"] = mode
            ad = voice_adapter(self.tmp)
            if fatal is None:
                self.assertEqual((ad.drift(), ad.breaking_drift()), ([], []))
            else:
                self.assertTrue(ad.drift(), mode)
                self.assertEqual(bool(ad.breaking_drift()), fatal, (mode, ad.breaking_drift()))
                caps = voice_capabilities(lambda: voice_adapter(self.tmp))
                self.assertEqual(caps["voice-enhance"].status, "MISSING" if fatal else "AVAILABLE", mode)
        os.environ["FAKE_VOICE_CONTRACT"] = "incompatible"
        with self.assertRaises(ContractError):
            voice_adapter(self.tmp)
        self.assertEqual(drift(pinned_contract()), [])
        self.assertEqual(breaking_drift(pinned_contract()), [])

    def test_locate_checkout_and_env(self):
        root = fake_voice_enhance.checkout(Path(self.tmp) / "voice-enhance-skill")
        sk = locate_voice_enhance(env={"VIDEO_AGENT_VOICE_ENHANCE_DIR": root, "PATH": ""})
        self.assertEqual(sk.command[1:], ["-c", MODULE_BOOTSTRAP, "voice_enhance_skill.cli"])   # python -m without the current directory on sys.path (ADR-074)
        self.assertEqual(sk.env, {"PYTHONPATH": str(Path(root) / "src")})
        self.assertIsNone(locate_voice_enhance(env={"VIDEO_AGENT_VOICE_ENHANCE_DIR": str(Path(self.tmp) / "nothing"), "PATH": ""}), "the env var is authoritative")


class CapabilityTests(_Base):
    def test_doctor_verdicts_carry_the_skills_codes(self):
        caps = voice_capabilities(lambda: voice_adapter(self.tmp))
        self.assertEqual(caps["voice-enhance"].status, "AVAILABLE")
        self.assertEqual({t: caps[f"voice-enhance:{t}"].status for t in ("ISOLATE", "DEREVERB", "BLEED_REDUCE")}, dict.fromkeys(("ISOLATE", "DEREVERB", "BLEED_REDUCE"), "AVAILABLE"))
        self.assertEqual(caps["voice-enhance:ISOLATE"].evidence["lifecycle"], "EXPERIMENTAL")
        self.assertFalse(caps["voice-enhance:ISOLATE"].evidence["resample"])
        os.environ["FAKE_VOICE_ENGINES"] = "nara_wpe"
        caps = voice_capabilities(lambda: voice_adapter(self.tmp))
        self.assertEqual((caps["voice-enhance"].status, caps["voice-enhance:DEREVERB"].status, caps["voice-enhance:ISOLATE"].status), ("AVAILABLE", "MISSING", "AVAILABLE"))
        ev = caps["voice-enhance:DEREVERB"].evidence
        self.assertEqual((ev["code"], ev["engine"], ev["engine_status"]), ("ENGINE_UNAVAILABLE", "nara_wpe", "MISSING"))
        self.assertIn("[wpe]", ev["install"])
        os.environ["FAKE_VOICE_ENGINES"] = "model"
        caps = voice_capabilities(lambda: voice_adapter(self.tmp))
        self.assertEqual(caps["voice-enhance:ISOLATE"].evidence["code"], "MODEL_NOT_FOUND")
        os.environ.pop("FAKE_VOICE_ENGINES")
        os.environ["FAKE_VOICE_RESAMPLE"] = "1"
        self.assertTrue(voice_capabilities(lambda: voice_adapter(self.tmp))["voice-enhance:ISOLATE"].evidence["resample"])

    def test_without_the_agents_ffmpeg_skill_nothing_runs(self):
        """The Skill would search ./vendor and ../ of the current directory for ffmpeg-skill; the agent never lets it (ADR-069): no engine dir → fail."""
        ad = VoiceEnhanceAdapter(fake_skill(), workspace=str(Path(self.tmp) / "ws"))
        self.assertEqual(ad.doctor()["status"], "fail")
        self.assertFalse([c for c in calls(self.log) if c[:1] == ["doctor"]], "the Skill's doctor is not even started")
        caps = voice_capabilities(lambda: VoiceEnhanceAdapter(fake_skill()))
        self.assertEqual(caps["voice-enhance"].status, "MISSING")


class AdapterTests(_Base):
    def setUp(self):
        super().setUp()
        self.ad = voice_adapter(self.tmp)
        self.src = wav(self.tmp, "talk.wav")
        self.out = str(Path(self.tmp) / "ws" / "jobs" / "j" / "ops" / "x" / "talk_isolate.wav")
        os.makedirs(os.path.dirname(self.out), exist_ok=True)
        self.paths = {"a": self.src, "a_isolate": self.out}

    def op(self, tool="voice-enhance/isolate", **args):
        return Operation(tool=tool, args={"input": "a", "output": "a_isolate", **args}, inputs=["a"], outputs=["a_isolate"])

    def test_argv_is_flag_equals_value_and_positions_after_double_dash(self):
        b = self.ad.build(self.op(atten_lim_db=12), self.paths)
        self.assertEqual(b["argv"][:4], ["isolate", "--json", "--offline", "--overwrite"])
        self.assertIn(f"--ffmpeg-skill={Path(self.tmp).resolve()}", b["argv"])
        self.assertIn(f"--output={self.out}", b["argv"])
        self.assertIn("--atten-lim-db=12.0", b["argv"])
        self.assertEqual(b["argv"][-2:], ["--", self.src])
        before = b["argv"][:b["argv"].index("--")]
        self.assertTrue(all(a.startswith("--") for a in before[1:]), "every option is one --flag=value word")
        # a file whose name starts with '-' is still a position (after --), never an option
        dash = wav(self.tmp, "-o.wav")
        b2 = self.ad.build(self.op(), {"a": dash, "a_isolate": self.out})
        self.assertEqual(b2["argv"][-2:], ["--", dash])
        b3 = self.ad.build(Operation(tool="voice-enhance/dereverb", args={"input": "a", "output": "a_isolate", "taps": 15, "delay": 3, "iterations": 5}, inputs=[], outputs=[]), self.paths)
        self.assertEqual([a for a in b3["argv"] if a.startswith(("--taps", "--delay", "--iterations"))], ["--taps=15", "--delay=3", "--iterations=5"])

    def test_refusals_before_the_process_starts(self):
        outside = str(Path(tempfile.mkdtemp()) / "x.wav")
        Path(outside).write_bytes(b"{}")
        cases = [
            (self.op(command="rm"), "forbidden key"), (self.op(strength=3), "unknown"), (self.op(atten_lim_db=0), "within"), (self.op(atten_lim_db=101), "within"),
            (self.op(atten_lim_db=True), "number"), (Operation(tool="voice-enhance/dereverb", args={"input": "a", "output": "a_isolate", "taps": 2.5}, inputs=[], outputs=[]), "integer"),
            (Operation(tool="voice-enhance/dereverb", args={"input": "a", "output": "a_isolate", "iterations": 21}, inputs=[], outputs=[]), "within"),
            (Operation(tool="voice-enhance/isolate", args={"input": "nope", "output": "a_isolate"}, inputs=[], outputs=[]), "not an artifact id"),
            (Operation(tool="voice-enhance/isolate", args={"input": "a", "output": "a"}, inputs=[], outputs=[]), "outside the workspace"),
        ]
        for op, msg in cases:
            r = self.ad.run(op, self.paths)
            self.assertFalse(r.ok, msg)
            self.assertEqual((r.data["error"]["code"], r.data["error"]["recovery_class"]), ("INVALID_REQUEST", "INVALID_ARGS"), msg)
            self.assertIn(msg, r.data["error"]["message"], msg)
        r = self.ad.run(self.op(), {"a": outside, "a_isolate": self.out})
        self.assertIn("outside the allowed input roots", r.data["error"]["message"])
        r = self.ad.run(self.op(), {"a": self.src, "a_isolate": str(Path(self.tmp) / "ws" / "x.mp4")})
        self.assertIn(".wav", r.data["error"]["message"])
        b_dir = Path(self.tmp) / "ws" / "jobs" / "j" / "ops" / "00_bleed_reduce"
        bad = Operation(tool="voice-enhance/bleed-reduce", args={"inputs": ["a", "b"], "outputs": ["a_br", "b_br"]}, inputs=[], outputs=[])
        other = wav(self.tmp, "lav.wav")
        r = self.ad.run(bad, {"a": self.src, "b": other, "a_br": str(b_dir / "talk_a0.wav"), "b_br": str(b_dir / "wrong.wav")})
        self.assertIn("is not where the Skill writes", r.data["error"]["message"])
        r = self.ad.run(Operation(tool="voice-enhance/bleed-reduce", args={"inputs": ["a"], "outputs": ["a_br"]}, inputs=[], outputs=[]), {"a": self.src, "a_br": str(b_dir / "talk_a0.wav")})
        self.assertIn("2..8 inputs", r.data["error"]["message"])
        self.assertEqual(self.op_calls(), [], "nothing reached the Skill")
        r = VoiceEnhanceAdapter(fake_skill(), workspace=str(Path(self.tmp) / "ws")).run(self.op(), self.paths)
        self.assertIn("no ffmpeg-skill located", r.data["error"]["message"])

    def test_run_admit_and_record(self):
        r = self.ad.run(self.op(atten_lim_db=20), self.paths)
        self.assertTrue(r.ok, r.stderr_tail)
        self.assertEqual(r.output, self.out)
        self.assertEqual((r.data["op"], r.data["capability"], r.data["lifecycle"]), ("isolate", "voice.isolate", "EXPERIMENTAL"))
        self.assertEqual(r.data["engine"]["id"], "deepfilternet")
        self.assertIn("DeepFilterNet", r.data["engine"]["attribution"])
        self.assertEqual(r.data["outputs"][0]["sample_rate"], 48000)
        self.assertNotIn("engine_argv", r.data["parameters"])
        from video_agent.tools.skill_process import sha256_file
        self.assertEqual(r.data["outputs"][0]["sha256"], sha256_file(self.out))
        from video_agent.tools.voice_enhance import lift_observation
        obs = lift_observation(r, "a_isolate")
        self.assertEqual((obs.kind, obs.provenance, obs.skill, obs.fingerprint), ("voice.isolate", "OBSERVED", "voice-enhance", sha256_file(self.out)))
        # the executor's dry run never runs the Skill
        os.remove(self.out)
        n = len(self.op_calls())
        dr = self.ad.run(self.op(), self.paths, dry_run=True)
        self.assertTrue(dr.ok and dr.dry_run and dr.output is None)
        self.assertEqual(len(self.op_calls()), n)
        self.assertFalse(os.path.exists(self.out))

    def test_skill_codes_and_unadmitted_results(self):
        for mode, code, cls, retry in (("fail:ENGINE_UNAVAILABLE", "ENGINE_UNAVAILABLE", "TOOL_MISSING", False), ("fail:MODEL_NOT_FOUND", "MODEL_NOT_FOUND", "TOOL_MISSING", False),
                                       ("fail:FFMPEG_SKILL_UNAVAILABLE", "FFMPEG_SKILL_UNAVAILABLE", "TOOL_MISSING", False), ("fail:ENGINE_FAILED", "ENGINE_FAILED", "SKILL_ERROR", False),
                                       ("fail:ENGINE_TIMEOUT", "ENGINE_TIMEOUT", "TIMEOUT", True), ("fail:TOOL_ERROR", "TOOL_ERROR", "UNKNOWN", True),
                                       ("fail:INVALID_RESULT", "INVALID_RESULT", "SKILL_ERROR", False),
                                       ("bad_hash", "INVALID_OUTPUT", "SKILL_ERROR", False), ("wrong_path", "INVALID_OUTPUT", "SKILL_ERROR", False),
                                       ("wrong_rate", "INVALID_OUTPUT", "SKILL_ERROR", False), ("bad_json", "INVALID_OUTPUT", "SKILL_ERROR", False)):
            os.environ["FAKE_VOICE_MODE"] = mode
            r = self.ad.run(self.op(), self.paths)
            self.assertFalse(r.ok, mode)
            err = r.data["error"]
            self.assertEqual((err["code"], err["recovery_class"], err["retryable"]), (code, cls, retry), mode)
            self.assertFalse(os.path.exists(self.out), f"{mode}: nothing half-written is left")
        os.environ["FAKE_VOICE_MODE"] = "fail:CANCELLED"
        r = self.ad.run(self.op(), self.paths)
        self.assertEqual(r.data["error"]["details"]["reason"], "signal")
        from video_agent.execution.recovery import classify_error
        self.assertEqual(classify_error(r), "INTERRUPTED", "a signal is not retried with a longer timeout")

    def test_executor_reruns_a_multi_output_operation_instead_of_half_reusing_it(self):
        out1, out2 = str(Path(self.tmp) / "ws" / "o1.wav"), str(Path(self.tmp) / "ws" / "o2.wav")

        class Stub:
            def supports(self, tool):
                return True

            def run(self, op, paths, timeout=None, dry_run=False, attempt=1):
                for o in op.outputs:
                    Path(paths[o]).write_bytes(b"x")
                return ToolResult(op.id, op.tool, True, 0, paths[op.outputs[0]], {}, [], "", 0.0, attempt, dry_run)
        op = Operation(tool="voice-enhance/bleed-reduce", args={}, inputs=["a", "b"], outputs=["o1", "o2"], id="op1")
        op.idempotency_key = "k"
        ex = Executor(Stub())
        self.assertEqual(ex.run([op], {"o1": out1, "o2": out2}).status, "COMPLETED")
        self.assertEqual(ex.completed, {}, "no single-file record for a two-output operation")
        one = Operation(tool="x/y", args={}, inputs=["a"], outputs=["o1"], id="op2")
        one.idempotency_key = "k1"
        ex.run([one], {"o1": out1})
        self.assertIn("k1", ex.completed)


class RequirementTests(unittest.TestCase):
    def test_vocabulary_and_refusals(self):
        ok = parse_voice_requirements(reqs({"audio.production": True, "voice.isolate": True, "voice.isolate.atten_lim_db": "12", "voice.dereverb": "true", "voice.dereverb.taps": 15,
                                            "audio.mix": True, "voice.bleed_reduce": True, "voice.bleed_reduce.dominance_db": 6}))
        self.assertEqual(ok["ops"], {"voice.bleed_reduce": {"dominance_db": 6.0}, "voice.dereverb": {"taps": 15}, "voice.isolate": {"atten_lim_db": 12.0}})
        self.assertEqual(parse_voice_requirements(reqs({"audio.production": True, "voice.isolate": False}))["ops"], {})
        self.assertEqual(VOICE_ORDER, ("voice.bleed_reduce", "voice.dereverb", "voice.isolate"))
        self.assertEqual({k: v for k, v in APPROVAL_KEYS.items()}, {op: (f"{op}.approval", "CONFIRM") for op in OPERATIONS})
        for bad, msg in (({"voice.isolate": True}, "audio.production=true"), ({"audio.production": True, "voice.denoise": True}, "unknown requirement key"),
                         ({"audio.production": True, "voice.isolate": "maybe"}, "true or false"),
                         ({"audio.production": True, "voice.isolate.atten_lim_db": 6}, "needs voice.isolate=true"),
                         ({"audio.production": True, "voice.isolate": False, "voice.isolate.atten_lim_db": 6}, "ambiguous"),
                         ({"audio.production": True, "voice.isolate": True, "voice.isolate.atten_lim_db": 0}, "within"),
                         ({"audio.production": True, "voice.dereverb": True, "voice.dereverb.taps": 51}, "within"),
                         ({"audio.production": True, "voice.dereverb": True, "voice.dereverb.delay": 2.5}, "integer"),
                         ({"audio.production": True, "voice.bleed_reduce": True}, "audio.mix=true"),
                         ({"audio.production": True, "audio.mix": True, "voice.bleed_reduce": True, "voice.bleed_reduce.dominance_db": 31}, "within"),
                         ({"audio.production": True, "voice.isolate.approval": "AUTO"}, "unknown requirement key")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                check_voice_keys(bad)
            self.assertIn(msg, str(cm.exception), str(bad))
            with self.assertRaises(ValueError, msg=str(bad)):
                _check_requirement_keys(bad)
        prof = {"audio.production": Requirement(key="audio.production", value=True, provenance="USER", source="cli"),
                "voice.isolate": Requirement(key="voice.isolate", value=True, provenance="PROFILE", source="profile:x")}
        with self.assertRaises(EditRequirementError) as cm:
            parse_voice_requirements(prof)
        self.assertIn("opt-in on the request", str(cm.exception))
        self.assertFalse([k for k in BRIEF_REQUIREMENT_KEYS if k.startswith("voice.")], "a brief never turns an EXPERIMENTAL op on")

    def test_registry_and_compiler(self):
        from video_agent.skills import default_registry
        r = default_registry()
        for name, typ, tool in (("voice_isolate", "ISOLATE", "voice-enhance/isolate"), ("voice_dereverb", "DEREVERB", "voice-enhance/dereverb"),
                                ("voice_bleed_reduce", "BLEED_REDUCE", "voice-enhance/bleed-reduce")):
            spec = r.get(name)
            self.assertEqual((spec.phase, spec.approval, spec.tools, spec.required_capabilities), (1, "CONFIRM", [tool], ["ffmpeg-skill", "voice-enhance", f"voice-enhance:{typ}"]))
        self.assertNotIn("voice_enhancement", r.names())
        self.assertEqual(lower_voice_op("voice-enhance/isolate", {"type": "voice.isolate", "asset": "a", "atten_lim_db": 12.0}, "a_cut", "a_isolate"),
                         {"input": "a_cut", "output": "a_isolate", "atten_lim_db": 12.0})
        self.assertEqual(lower_voice_op("voice-enhance/bleed-reduce", {"type": "voice.bleed_reduce", "asset": "programme_audio", "inputs": ["a", "b"], "outputs": ["a_br", "b_br"]},
                                        ["a", "b"], ["a_br", "b_br"]), {"inputs": ["a", "b"], "outputs": ["a_br", "b_br"]})
        with self.assertRaises(CompileError):
            lower_voice_op("audio-production/run", {"type": "voice.dereverb", "asset": "a"}, "a", "b")
        from video_agent.skills.ecosystem import install_commands
        self.assertEqual(install_commands("voice-enhance")[0], "git clone https://github.com/kajisho5/voice-enhance-skill && pip install -e 'voice-enhance-skill'")


class PipelineTests(_Base):
    def test_dereverb_then_isolate_after_the_cut_confirm_then_run(self):
        src = wav(self.tmp, "talk.wav", channels=2)
        svc = voice_service(self.tmp)
        r = plan_and_render(svc, [src], {"audio.production": True, "voice.dereverb": True, "voice.dereverb.taps": 15, "voice.isolate": True, "voice.isolate.atten_lim_db": 20,
                                         "audio.noise_reduction.strength_db": 24}, profile="generic", name="v", approve=False)
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        self.assertEqual([op["type"] for op in d["audio"]["operations"]], ["audio.cut", "voice.dereverb", "voice.isolate", "audio.noise_reduction"])
        self.assertEqual([s["skill"] for s in d["plan"]["steps"]], ["audio_cut", "voice_dereverb", "voice_isolate", "audio_noise_reduction"])
        iso = next(op for op in d["audio"]["operations"] if op["type"] == "voice.isolate")
        self.assertEqual((iso["input"], iso["output"], iso["atten_lim_db"]), (f"{list(d['assets'])[0]}_dereverb", f"{list(d['assets'])[0]}_isolate", 20.0))
        for subj in ("voice.dereverb", "voice.isolate"):
            dec = next(x for x in d["decisions"] if x["subject"] == subj)
            self.assertEqual((dec["type"], dec["approval"], dec["provenance"], dec["params"]["lifecycle"]), ("TRANSFORM", "CONFIRM", "USER", "EXPERIMENTAL"),
                             "an explicit request still needs confirmation: EXPERIMENTAL")
            self.assertIn("EXPERIMENTAL", dec["reason"])
        self.assertEqual(r["out"]["status"], "WAITING_FOR_APPROVAL")
        self.assertEqual(self.op_calls(), [], "nothing runs before the confirmation")
        out = svc.render(r["ir"], r["path"], approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        self.assertEqual([c[0] for c in self.op_calls()], ["dereverb", "isolate"])
        for c in self.op_calls():
            pos = c.index("--")
            self.assertTrue(all(a.startswith("--") for a in c[1:pos]))
            self.assertTrue(any(a.startswith("--ffmpeg-skill=") for a in c[1:pos]), "the agent always names ffmpeg-skill")
            self.assertEqual(len(c) - pos - 1, 1)
        self.assertIn("--taps=15", self.op_calls()[0])
        res = [x for x in out["execution"]["results"] if x["tool"].startswith("voice-enhance/")]
        self.assertEqual([x["data"]["engine"]["id"] for x in res], ["nara_wpe", "deepfilternet"])
        ap_types = [x["data"]["operation_type"] for x in out["execution"]["results"] if x["tool"] == "audio-production/run"]
        self.assertEqual(ap_types, ["CUT", "NOISE_REDUCTION"])
        items = {i["name"]: i for i in out["qa"]["items"]}
        self.assertEqual(items["duration"]["status"], "PASS", items["duration"])
        from video_agent.agent.audio import audio_subjects
        self.assertEqual(audio_subjects(d)[list(d["assets"])[0]]["channels"], 1, "isolation writes mono: what QA expects downstream (the source is stereo)")
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        kinds = [o["kind"] for o in prov.get("skill_observations") or []]
        self.assertIn("voice.isolate", kinds)
        iso_obs = next(o for o in prov["skill_observations"] if o["kind"] == "voice.isolate")
        self.assertEqual((iso_obs["data"]["lifecycle"], iso_obs["data"]["engine"]["code_license"]), ("EXPERIMENTAL", "MIT OR Apache-2.0"))
        rows = [e for e in prov["operations"] if e["tool"].startswith("voice-enhance/")]
        self.assertEqual([e["skill"] for e in rows], ["voice_dereverb", "voice_isolate"])
        self.assertTrue(all(e["decision"] for e in rows))

    def test_isolate_needs_48_khz_unless_ffmpeg_skill_resamples(self):
        src = wav(self.tmp, "talk44.wav", rate=44100)
        svc = voice_service(self.tmp)
        ir = svc.plan([src], "generic", user_requirements={"audio.production": True, "voice.isolate": True})
        b = next(x for x in ir.doc["decisions"] if x["subject"] == "voice.isolate")
        self.assertEqual((b["type"], b["approval"], b["params"]["code"], b["params"]["needs_flag"], b["params"]["sample_rate"]), ("BLOCK", "BLOCK", "FFMPEG_SKILL_UNAVAILABLE", "sample_rate", 44100))
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        self.assertFalse([op for op in ir.doc["audio"]["operations"] if op["type"] == "voice.isolate"])
        os.environ["FAKE_VOICE_RESAMPLE"] = "1"
        svc = voice_service(self.tmp)
        ir = svc.plan([src], "generic", user_requirements={"audio.production": True, "voice.isolate": True})
        self.assertEqual(next(x for x in ir.doc["decisions"] if x["subject"] == "voice.isolate")["type"], "TRANSFORM")
        self.assertEqual(self.op_calls(), [])

    def test_missing_engine_or_model_blocks_with_the_skills_code(self):
        src = wav(self.tmp)
        os.environ["FAKE_VOICE_ENGINES"] = "nara_wpe"
        ir = voice_service(self.tmp).plan([src], "generic", user_requirements={"audio.production": True, "voice.dereverb": True, "voice.isolate": True})
        b = next(x for x in ir.doc["decisions"] if x["subject"] == "voice.dereverb")
        self.assertEqual((b["type"], b["params"]["code"], b["params"]["engine"]), ("BLOCK", "ENGINE_UNAVAILABLE", "nara_wpe"))
        self.assertIn("pip install 'voice-enhance-skill[wpe]'", b["reason"])
        self.assertIn("capability:voice-enhance:DEREVERB", b["evidence"])
        self.assertEqual(next(x for x in ir.doc["decisions"] if x["subject"] == "voice.isolate")["type"], "TRANSFORM")
        os.environ["FAKE_VOICE_ENGINES"] = "model"
        ir = voice_service(self.tmp).plan([src], "generic", user_requirements={"audio.production": True, "voice.isolate": True})
        b = next(x for x in ir.doc["decisions"] if x["subject"] == "voice.isolate")
        self.assertEqual((b["type"], b["params"]["code"]), ("BLOCK", "MODEL_NOT_FOUND"))
        self.assertIn("VOICE_ENHANCE_DFN_MODEL", b["reason"])
        self.assertEqual(self.op_calls(), [], "nothing downloaded, nothing run")

    def test_without_voice_enhance_the_step_is_handed_off(self):
        svc = voice_service(self.tmp, installed=False)
        ir = svc.plan([wav(self.tmp)], "generic", user_requirements={"audio.production": True, "voice.isolate": True})
        step = next(s for s in ir.doc["plan"]["steps"] if s["skill"] == "voice_isolate")
        self.assertEqual((step["tool"], step["status"]), (None, "BLOCKED"))
        [h] = [h for h in svc.handoffs(ir) if h["skill"] == "voice_isolate"]
        self.assertEqual(h["repositories"], ["voice-enhance-skill"])
        self.assertTrue(any("voice-enhance-skill" in c for c in h["install"]))

    def test_bleed_reduction_of_the_mixed_microphones(self):
        a, b = wav(self.tmp, "lav_a.wav", channels=1), wav(self.tmp, "lav_b.wav", channels=1)
        svc = voice_service(self.tmp)
        r = plan_and_render(svc, [a, b], {"audio.production": True, "audio.mix": True, "voice.bleed_reduce": True, "voice.bleed_reduce.dominance_db": 6, "voice.isolate": True},
                            profile="generic", name="br")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        ids = list(d["assets"])
        self.assertEqual([op["type"] for op in d["audio"]["operations"]], ["voice.bleed_reduce", "voice.isolate", "voice.isolate", "audio.mix"])
        br = d["audio"]["operations"][0]
        self.assertEqual((br["asset"], br["inputs"], br["outputs"], br["dominance_db"]), ("programme_audio", ids, [f"{x}_bleed_reduce" for x in ids], 6.0))
        mix = next(op for op in d["audio"]["operations"] if op["type"] == "audio.mix")
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "audio_mix")
        self.assertEqual(step["inputs"], [f"{x}_isolate" for x in ids])
        self.assertEqual(mix["inputs"], ids)
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        [bc] = [c for c in self.op_calls() if c[0] == "bleed-reduce"]
        self.assertEqual(bc[bc.index("--") + 1:], [a, b])
        self.assertIn("--dominance-db=6.0", bc)
        self.assertEqual([c[0] for c in self.op_calls()], ["bleed-reduce", "isolate", "isolate"])
        dec = next(x for x in d["decisions"] if x["subject"] == "voice.bleed_reduce")
        self.assertEqual((dec["type"], dec["approval"]), ("TRANSFORM", "CONFIRM"))
        from video_agent.agent.audio import audio_subjects
        self.assertEqual(audio_subjects(d)["programme_audio"]["channels"], 1)
        self.assertEqual(out["execution"]["skipped"], [])

    def test_bleed_reduction_refusals(self):
        svc = voice_service(self.tmp)
        a, b = wav(self.tmp, "lav_a.wav", channels=1), wav(self.tmp, "lav_b.wav", channels=1, rate=44100)
        ir = svc.plan([a, b], "generic", user_requirements={"audio.production": True, "audio.mix": True, "voice.bleed_reduce": True})
        x = next(x for x in ir.doc["decisions"] if x["subject"] == "voice.bleed_reduce")
        self.assertEqual((x["type"], x["params"]["code"]), ("BLOCK", "FFMPEG_SKILL_UNAVAILABLE"))
        os.makedirs(Path(self.tmp) / "src" / "b")
        c = pipeline_media(str(Path(self.tmp) / "src" / "b" / ".."), "lav_a.wav", video=False, channels=1)
        d2 = str(Path(self.tmp) / "src2" / "lav_a.wav")
        os.makedirs(os.path.dirname(d2))
        Path(d2).write_bytes(Path(c).read_bytes())
        svc2 = pipeline_service(self.tmp, caps=VoiceCaps(voice_capabilities(lambda: voice_adapter(self.tmp))), allowed=[str(Path(self.tmp) / "src"), str(Path(self.tmp) / "src2")])
        svc2._adapter.register(voice_adapter(self.tmp))
        ir = svc2.plan([a, d2], "generic", user_requirements={"audio.production": True, "audio.mix": True, "voice.bleed_reduce": True})
        x = next(x for x in ir.doc["decisions"] if x["subject"] == "voice.bleed_reduce")
        self.assertEqual((x["type"], x["params"]["code"]), ("BLOCK", "INVALID_ARGUMENT"))
        self.assertIn("share a file name", x["reason"])
        self.assertEqual(self.op_calls(), [])

    def test_validator_refuses_a_tampered_chain(self):
        src = wav(self.tmp)
        svc = voice_service(self.tmp)
        ir = svc.plan([src], "generic", user_requirements={"audio.production": True, "voice.dereverb": True, "voice.isolate": True})
        d = ir.doc
        self.assertEqual(check_audio_operations(d), [])
        bad = json.loads(json.dumps(d))
        ops = bad["audio"]["operations"]
        i, j = [n for n, op in enumerate(ops) if op["type"] in ("voice.dereverb", "voice.isolate")]
        ops[i], ops[j] = ops[j], ops[i]
        self.assertTrue(any("must come after" in e for e in check_audio_operations(bad)), check_audio_operations(bad))
        bad = json.loads(json.dumps(d))
        next(op for op in bad["audio"]["operations"] if op["type"] == "voice.isolate")["atten_lim_db"] = 0
        self.assertTrue(any("atten_lim_db 0" in e for e in check_audio_operations(bad)))
        bad = json.loads(json.dumps(d))
        next(op for op in bad["audio"]["operations"] if op["type"] == "voice.isolate")["strength"] = 3
        self.assertTrue(any("not in the operation's vocabulary" in e for e in check_audio_operations(bad)))
        bad = json.loads(json.dumps(d))
        bad["audio"]["operations"].insert(0, {"type": "voice.bleed_reduce", "asset": "programme_audio", "inputs": [list(d["assets"])[0]], "outputs": ["x"], "decision_ids": []})
        self.assertTrue(any("exactly the inputs of the audio mix" in e for e in check_audio_operations(bad)))
        # an AUTO approval written into the IR is refused (the basis says CONFIRM: ADR-033)
        bad = json.loads(json.dumps(d))
        dec = next(x for x in bad["decisions"] if x["subject"] == "voice.isolate")
        dec["approval"] = "AUTO"
        from video_agent.project import ProjectIR
        self.assertFalse(svc.validate(ProjectIR(bad)).ok)

    def test_engine_failure_is_not_retried_and_a_timeout_once(self):
        src = wav(self.tmp)
        os.environ["FAKE_VOICE_MODE"] = "fail:ENGINE_FAILED"
        out = plan_and_render(voice_service(self.tmp), [src], {"audio.production": True, "voice.dereverb": True}, profile="generic", name="f1")["out"]
        self.assertNotEqual(out["execution"]["status"], "COMPLETED")
        self.assertEqual(len(self.op_calls()), 1)
        self.assertEqual([x["class"] for x in out["execution"]["recovery"]], ["SKILL_ERROR"])
        Path(self.log).unlink()
        os.environ["FAKE_VOICE_MODE"] = "fail:ENGINE_TIMEOUT"
        out = plan_and_render(voice_service(self.tmp), [src], {"audio.production": True, "voice.dereverb": True}, profile="generic", name="f2")["out"]
        self.assertEqual(len(self.op_calls()), 2, "retried once with a doubled timeout, then stopped")
        self.assertEqual([x["class"] for x in out["execution"]["recovery"]], ["TIMEOUT", "TIMEOUT"])

    def test_video_container_counts_as_work_on_the_audio_path(self):
        """A video container whose only audio operation is voice enhancement is not 'nothing to do': it waits for the extraction and the op."""
        src = pipeline_media(self.tmp, "talk.mp4", sample_rate=48000)
        ir = voice_service(self.tmp).plan([src], "generic", user_requirements={"audio.production": True, "voice.isolate": True})
        self.assertFalse([x for x in ir.doc["decisions"] if x["subject"] == "audio.production" and x["type"] == "BLOCK"])
        self.assertEqual(sorted(x["subject"] for x in ir.pending_confirmations()), ["audio.extract", "voice.isolate"])


if __name__ == "__main__":
    unittest.main()
