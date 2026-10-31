"""ADR-085 (roadmap step 9 dubbing; dubbing-skill's roadmap step 9, agent side, as its ADR-012 hands it over): dubbing-skill's DUB as an opt-in,
always-CONFIRM step on the programme's picture after its edits. The agent supplies the user's translated cues, never translates, passes a
reference voice only with the consent record the user named (never writing one), never retries CONSENT_REQUIRED, and carries
obligations.credit into the ProductionReceipt. Against a fake `dubbing` process (tests/fake_dubbing.py) that speaks dubbing-skill main's
contract, doctor and result shapes, with the fake media engine. No engine, ffmpeg or network."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import PipelineCaps, clear_modes, fake_media, pipeline_service, plan_and_render  # noqa: E402

from video_agent.agent.dubbing import APPROVAL_KEYS, OP, check_dub_keys, parse_dub_requirements  # noqa: E402
from video_agent.audit.receipt import obligations_of  # noqa: E402
from video_agent.capabilities.resolver import CapabilityResolver  # noqa: E402
from video_agent.execution.compiler import CompileError, lower_dub  # noqa: E402
from video_agent.execution.recovery import RECOVERY_TABLE, classify_error  # noqa: E402
from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.models import Operation, Requirement  # noqa: E402
from video_agent.project.hashing import plan_hash  # noqa: E402
from video_agent.project.validator import check_speech_operations  # noqa: E402
from video_agent.tools.discovery import MODULE_BOOTSTRAP  # noqa: E402
from video_agent.tools.dubbing import (ERROR_CLASSES, FATAL_DRIFT_KEYS, TRANSLATION_NOTE, ContractError, DubbingAdapter, breaking_drift, check_contract, drift,  # noqa: E402
                                       locate_dubbing, parameters, pinned_contract, recovery_class)
from video_agent.tools.dubbing.adapter import PINNED_CONTRACT_PATH  # noqa: E402
from video_agent.tools.skill_process import CliSkill  # noqa: E402
import fake_dubbing  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_dubbing.py")
MODES = ("FAKE_DUB_MODE", "FAKE_DUB_ENGINES", "FAKE_DUB_CONTRACT", "FAKE_DUB_CALLS", "VIDEO_AGENT_DUBBING_DIR")


def fake_skill() -> CliSkill:
    return CliSkill("dubbing", [sys.executable, FAKE], None, {})


def dub_adapter(tmp: str, **kw) -> DubbingAdapter:
    ws = str(Path(tmp) / "ws")
    os.makedirs(ws, exist_ok=True)
    return DubbingAdapter(fake_skill(), workspace=ws, allowed_inputs=[str(Path(tmp) / "src"), ws], ffmpeg_skill_dir=tmp, **kw)


def dub_capabilities(adapter_factory) -> dict:
    """What the resolver reports for a located dubbing-skill: `dubbing` and `dubbing:DUB` (with the Skill's own codes and every DUB engine)."""
    caps: dict = {}
    CapabilityResolver._finishing_skill(CapabilityResolver.__new__(CapabilityResolver), caps, "dubbing", fake_skill(), lambda _sk: adapter_factory(),
                                        "VIDEO_AGENT_DUBBING_DIR", "dubbing-skill", "dubbing",
                                        per_item=lambda ad, doc: (ad.operation_status(doc), {}), item_evidence=lambda ad, doc: ad.operation_evidence(doc))
    return caps


class DubCaps(PipelineCaps):
    def __init__(self, dub: dict, **kw):
        super().__init__(**kw)
        self.dub = dub

    def resolve(self, refresh=False):
        out = super().resolve(refresh)
        out.update(self.dub)
        return out


def dub_service(tmp: str, installed: bool = True, **engine_kw):
    caps = DubCaps(dub_capabilities(lambda: dub_adapter(tmp)) if installed else {})
    svc = pipeline_service(tmp, caps=caps, **engine_kw)
    if installed:
        svc._adapter.register(dub_adapter(tmp))
        svc.adapter([])
    return svc


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
        self.log = str(Path(self.tmp) / "dub_calls.log")
        os.environ["FAKE_DUB_CALLS"] = self.log
        src = Path(self.tmp) / "src"
        src.mkdir(parents=True, exist_ok=True)
        self.cues = str(src / "en.srt")
        Path(self.cues).write_text("1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:04,000 --> 00:00:06,000\nThank you\n", encoding="utf-8")
        self.ref = str(src / "voice.wav")
        Path(self.ref).write_bytes(b"RIFF fake reference voice")
        self.consent = str(src / "consent.json")
        Path(self.consent).write_text(json.dumps({"fake_valid": True, "voice_owner": {"name": "Speaker A"},
                                                  "reference_sha256": hashlib.sha256(Path(self.ref).read_bytes()).hexdigest()}), encoding="utf-8")

    def tearDown(self):
        clear_modes()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def dub_calls(self):
        return [c for c in calls(self.log) if c[:1] == ["dub"]]


class ContractTests(_Base):
    def test_pin_shape_and_contract_checks(self):
        c = pinned_contract()
        raw = PINNED_CONTRACT_PATH.read_text(encoding="utf-8")
        self.assertEqual(raw, json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n", "one sorted line (no release tag: pinned from main)")
        self.assertEqual((c["skill_id"], c["version"], c["contract_version"]), ("dubbing", "0.1.0", "1.0"))
        self.assertEqual(check_contract(c), [])
        self.assertEqual({p["id"]: p["lifecycle"] for p in c["provides"]}["speech.dub"], "EXPERIMENTAL")
        self.assertEqual(set(ERROR_CLASSES), set(c["errors"]), "every dubbing error code has a recovery class")
        self.assertEqual(recovery_class("CONSENT_REQUIRED"), "CONSENT_REQUIRED")
        self.assertEqual(RECOVERY_TABLE["CONSENT_REQUIRED"]["action"], "BLOCK")
        self.assertEqual(recovery_class("SOMETHING_NEW"), "SKILL_ERROR")
        self.assertEqual(parameters(c), {"duck": (0.0, 40.0), "voice_gain": (-20.0, 20.0), "max_overrun": (0.0, 5.0)}, "ranges read from the contract's limits")
        for bad, msg in ((dict(c, version="0.2.0"), "supported range"), (dict(c, skill_id="x"), "skill_id"),
                         (dict(c, execution=dict(c["execution"], shell=True)), "shell false"),
                         (dict(c, tools=[t for t in c["tools"] if t["tool_id"] != "dubbing/dub"]), "lacks tool dubbing/dub"),
                         (dict(c, policy=dict(c["policy"], consent_record=dict(c["policy"]["consent_record"], bypass="--force"))), "bypass null"),
                         (dict(c, errors=dict(c["errors"], CONSENT_REQUIRED={"exit_code": 2, "retryable": True})), "CONSENT_REQUIRED must be retryable false"),
                         (dict(c, errors={k: v for k, v in c["errors"].items() if k != "TIMING_OVERFLOW"}), "TIMING_OVERFLOW")):
            self.assertTrue(any(msg in e for e in check_contract(bad)), (msg, check_contract(bad)))

    def test_drift_classification(self):
        self.assertIn("policy", FATAL_DRIFT_KEYS, "the consent gate is part of what the agent relies on")
        for mode, fatal in (("pinned", None), ("drift_version", False), ("drift_fatal", True)):
            os.environ["FAKE_DUB_CONTRACT"] = mode
            ad = dub_adapter(self.tmp)
            if fatal is None:
                self.assertEqual((ad.drift(), ad.breaking_drift()), ([], []))
            else:
                self.assertTrue(ad.drift(), mode)
                self.assertEqual(bool(ad.breaking_drift()), fatal, (mode, ad.breaking_drift()))
                caps = dub_capabilities(lambda: dub_adapter(self.tmp))
                self.assertEqual(caps["dubbing"].status, "MISSING" if fatal else "AVAILABLE", mode)
        for mode in ("incompatible", "bypass"):
            os.environ["FAKE_DUB_CONTRACT"] = mode
            with self.assertRaises(ContractError, msg=mode):
                dub_adapter(self.tmp)
        self.assertEqual(drift(pinned_contract()), [])
        self.assertEqual(breaking_drift(pinned_contract()), [])

    def test_locate_checkout_and_env(self):
        root = fake_dubbing.checkout(Path(self.tmp) / "dubbing-skill")
        sk = locate_dubbing(env={"VIDEO_AGENT_DUBBING_DIR": root, "PATH": ""})
        self.assertEqual(sk.command[1:], ["-c", MODULE_BOOTSTRAP, "dubbing_skill.cli"])   # python -m without the current directory on sys.path (ADR-074)
        self.assertEqual(sk.env, {"PYTHONPATH": str(Path(root) / "src")})
        self.assertIsNone(locate_dubbing(env={"VIDEO_AGENT_DUBBING_DIR": str(Path(self.tmp) / "nothing"), "PATH": ""}), "the env var is authoritative")


class CapabilityTests(_Base):
    def test_doctor_verdicts_carry_the_skills_codes(self):
        caps = dub_capabilities(lambda: dub_adapter(self.tmp))
        self.assertEqual((caps["dubbing"].status, caps["dubbing:DUB"].status), ("AVAILABLE", "AVAILABLE"))
        self.assertEqual(caps["dubbing:DUB"].evidence["lifecycle"], "EXPERIMENTAL")
        self.assertEqual(set(caps["dubbing:DUB"].evidence["engines"]), {"chatterbox", "style_bert_vits2"}, "the engines DUB takes, from the contract")
        os.environ["FAKE_DUB_ENGINES"] = "chatterbox"
        caps = dub_capabilities(lambda: dub_adapter(self.tmp))
        ev = caps["dubbing:DUB"].evidence
        self.assertEqual((caps["dubbing"].status, caps["dubbing:DUB"].status), ("AVAILABLE", "MISSING"))
        self.assertEqual((ev["code"], ev["engine"], ev["engine_status"], ev["engines"]["style_bert_vits2"]["status"]), ("ENGINE_UNAVAILABLE", "chatterbox", "MISSING", "OK"))
        self.assertIn("fake install line", ev["install"])

    def test_without_the_agents_ffmpeg_skill_nothing_runs(self):
        ad = DubbingAdapter(fake_skill(), workspace=str(Path(self.tmp) / "ws"))
        self.assertEqual(ad.doctor()["status"], "fail")
        self.assertFalse([c for c in calls(self.log) if c[:1] == ["doctor"]], "the Skill's doctor is not even started")


class AdapterTests(_Base):
    def setUp(self):
        super().setUp()
        self.ad = dub_adapter(self.tmp)
        self.src = fake_media(self.tmp, "talk.mp4")
        self.out = str(Path(self.tmp) / "ws" / "jobs" / "j" / "ops" / "x" / "talk_dub.mp4")
        os.makedirs(os.path.dirname(self.out), exist_ok=True)
        self.paths = {"a": self.src, "a_dub": self.out}

    def op(self, **args):
        return Operation(tool="dubbing/dub", args={"input": "a", "output": "a_dub", "cues": self.cues, "lang": "en", **args}, inputs=["a"], outputs=["a_dub"])

    def test_argv_is_flag_equals_value_and_positions_after_double_dash(self):
        b = self.ad.build(self.op(engine="style_bert_vits2", voice="jvnv-F1-jp", duck=10, reference=self.ref, consent=self.consent), self.paths)
        argv = b["argv"]
        self.assertEqual(argv[:4], ["dub", "--json", "--offline", "--overwrite"])
        pos = argv.index("--")
        self.assertEqual(argv[pos + 1:], [self.src, self.cues], "the programme and the user's cues after --")
        self.assertTrue(all(a.startswith("--") and "=" in a for a in argv[4:pos]), argv)
        self.assertIn(f"--ffmpeg-skill={Path(self.tmp).resolve()}", argv, "the agent always names its own ffmpeg-skill")
        self.assertIn("--translation-note=" + json.dumps(TRANSLATION_NOTE, sort_keys=True, separators=(",", ":")), argv, "translated by the user, recorded as such")
        for want in ("--lang=en", "--engine=style_bert_vits2", "--voice=jvnv-F1-jp", "--duck=10.0", f"--reference={self.ref}", f"--consent={self.consent}"):
            self.assertIn(want, argv)

    def test_refusals_before_the_process_starts(self):
        outside = str(Path(tempfile.mkdtemp()) / "en.srt")
        Path(outside).write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n", encoding="utf-8")
        for args, msg in (({"reference": self.ref}, "only together with its owner's consent record"), ({"consent": self.consent}, "only together"),
                          ({"cues": outside}, "outside the allowed input roots"), ({"lang": "english please"}, "BCP-47"), ({"engine": "voicevox"}, "is not one DUB takes"),
                          ({"duck": 41}, "within [0, 40]"), ({"on_unfit": "skip"}, "must be one of fail, keep"), ({"shell": "x"}, "forbidden key"),
                          ({"speed": 2}, "unknown dubbing/dub argument"), ({"reference": self.ref, "consent": self.ref}, "consent must be one of .json")):
            with self.assertRaises(Exception, msg=msg) as cm:
                self.ad.build(self.op(**args), self.paths)
            self.assertIn(msg, str(cm.exception))
        with self.assertRaises(Exception) as cm:
            self.ad.build(self.op(), dict(self.paths, a_dub=self.out[:-4] + ".wav"))
        self.assertIn("video container", str(cm.exception), "the agent keeps the picture: never a .wav output")
        self.assertEqual(self.dub_calls(), [])

    def test_run_admit_and_record(self):
        r = self.ad.run(self.op(engine="style_bert_vits2"), self.paths)
        self.assertTrue(r.ok, r.data)
        self.assertEqual(r.output, self.out)
        self.assertEqual(r.data["obligations"]["credit"], ["JVNV corpus voice jvnv-F1-jp (CC BY-SA 4.0)"])
        self.assertEqual((r.data["lifecycle"], r.data["consent"], r.data["translation"]["provided_by"]), ("EXPERIMENTAL", None, "user"))
        for mode, msg in (("bad_hash", "sha256"), ("wrong_path", "is not the planned"), ("no_obligations", "obligations.credit"), ("bad_json", "INVALID_OUTPUT")):
            os.environ["FAKE_DUB_MODE"] = mode
            r = self.ad.run(self.op(), self.paths)
            self.assertFalse(r.ok, mode)
            self.assertEqual(r.data["error"]["code"], "INVALID_OUTPUT", mode)
            self.assertIn(msg, r.data["error"]["message"] + "INVALID_OUTPUT", mode)
            self.assertFalse(os.path.exists(self.out), f"{mode}: an unadmitted output is removed")

    def test_consent_required_is_never_retryable(self):
        Path(self.consent).write_text(json.dumps({"fake_valid": False}), encoding="utf-8")
        for mode in ("ok", "consent_retryable"):
            os.environ["FAKE_DUB_MODE"] = mode
            r = self.ad.run(self.op(reference=self.ref, consent=self.consent), self.paths)
            err = r.data["error"]
            self.assertEqual((err["code"], err["retryable"], err["recovery_class"], err["details"]["reason"]), ("CONSENT_REQUIRED", False, "CONSENT_REQUIRED", "record_invalid"), mode)
            self.assertIn("never writes", err["details"]["agent"])
            self.assertEqual(classify_error(r), "CONSENT_REQUIRED")
        self.assertEqual(Path(self.consent).read_text(encoding="utf-8"), json.dumps({"fake_valid": False}), "the agent never touches the record")


class RequirementTests(_Base):
    def test_vocabulary_and_refusals(self):
        p = parse_dub_requirements(reqs({"dub.cues": self.cues, "dub.lang": "en", "dub.duck": "8", "dub.on_unfit": "keep"}))
        self.assertEqual(p["params"], {"lang": "en", "cues": self.cues, "duck": 8.0, "on_unfit": "keep"})
        self.assertIsNone(parse_dub_requirements(reqs({"audio.production": True})))
        for given, msg in (({"dub.lang": "en"}, "dub.cues is missing"), ({"dub.cues": self.cues}, "dub.lang is missing"),
                           ({"dub.cues": self.cues, "dub.lang": "en", "dub.reference": self.ref}, "dub.reference needs dub.consent"),
                           ({"dub.cues": self.cues, "dub.lang": "en", "dub.consent": self.consent}, "needs dub.consent"),
                           ({"dub.cues": self.cues, "dub.lang": "en", "dub.speed": 2}, "unknown requirement key"),
                           ({"dub.cues": self.cues, "dub.lang": "en", "dub.engine": "voicevox"}, "chatterbox, style_bert_vits2"),
                           ({"dub.cues": self.cues, "dub.lang": "en", "dub.voice_gain": 25}, "within [-20, 20]"),
                           ({"dub.cues": self.cues + ".txt", "dub.lang": "en"}, "must be one of .json, .srt, .vtt"),
                           ({"dub.cues": str(Path(self.tmp) / "src" / "missing.srt"), "dub.lang": "en"}, "no such file")):
            with self.assertRaises(EditRequirementError, msg=msg) as cm:
                check_dub_keys(given)
            self.assertIn(msg, str(cm.exception))
        m = reqs({"dub.cues": self.cues, "dub.lang": "en"})
        m["dub.lang"] = Requirement(key="dub.lang", value="en", provenance="PROFILE", source="profile")
        with self.assertRaises(EditRequirementError) as cm:
            parse_dub_requirements(m)
        self.assertIn("opt-in on the request", str(cm.exception))
        self.assertEqual(APPROVAL_KEYS[OP][1], "CONFIRM")

    def test_compiler_lowering(self):
        op = {"type": OP, "asset": "a", "input": "a_trim", "output": "a_dub", "cues": self.cues, "lang": "en", "duck": 6.0, "decision_ids": ["d"]}
        self.assertEqual(lower_dub("dubbing/dub", op, "a_trim", "a_dub"), {"input": "a_trim", "output": "a_dub", "cues": self.cues, "lang": "en", "duck": 6.0})
        with self.assertRaises(CompileError):
            lower_dub("ffmpeg-skill/audio", op, "a_trim", "a_dub")
        with self.assertRaises(CompileError):
            lower_dub("dubbing/dub", dict(op, reference=self.ref), "a_trim", "a_dub")

    def test_receipt_obligations(self):
        self.assertEqual(obligations_of([]), {"credit": [], "watermarks": [], "sources": []})
        obs = [{"kind": "speech.dub", "source": "dubbing/dub@0.1.0", "data": {"obligations": {"credit": ["VOICEVOX:ずんだもん"], "credit_required": True, "watermark": None}}},
               {"kind": "speech.dub", "source": "dubbing/dub@0.1.0", "data": {"obligations": {"credit": ["VOICEVOX:ずんだもん", "X"], "watermark": {"name": "Perth"}}}},
               {"kind": "voice.isolate", "data": {}}]
        ob = obligations_of(obs)
        self.assertEqual((ob["credit"], ob["watermarks"], len(ob["sources"])), (["VOICEVOX:ずんだもん", "X"], [{"name": "Perth"}], 2))


class PipelineTests(_Base):
    def test_dub_after_the_edits_confirm_then_run_with_credit_in_the_receipt(self):
        src = fake_media(self.tmp, "talk.mp4")
        svc = dub_service(self.tmp)
        r = plan_and_render(svc, [src], {"dub.cues": self.cues, "dub.lang": "en", "dub.engine": "style_bert_vits2", "dub.duck": 9}, profile="youtube", approve=False)
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        skills = [s["skill"] for s in d["plan"]["steps"]]
        self.assertLess(skills.index("silence_cleanup"), skills.index("speech_dub"))
        self.assertLess(skills.index("speech_dub"), skills.index("loudness_normalization"), "dubbed before loudness: the normalisation measures the dubbed sound")
        [op] = d["speech"]["operations"]
        aid = list(d["assets"])[0]
        self.assertEqual((op["input"], op["output"], op["cues"], op["lang"], op["engine"], op["duck"]), (f"{aid}_trim", f"{aid}_dub", self.cues, "en", "style_bert_vits2", 9.0))
        self.assertIn(str(Path(self.cues).parent), d["execution"]["allowed_inputs"])
        dec = next(x for x in d["decisions"] if x["subject"] == OP)
        self.assertEqual((dec["type"], dec["approval"], dec["provenance"], dec["params"]["lifecycle"]), ("TRANSFORM", "CONFIRM", "USER", "EXPERIMENTAL"),
                         "an explicit request still needs confirmation")
        self.assertIn("never translates", dec["reason"])
        self.assertEqual(r["out"]["status"], "WAITING_FOR_APPROVAL")
        self.assertEqual(self.dub_calls(), [], "nothing runs before the confirmation")
        out = svc.render(r["ir"], r["path"], approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        [c] = self.dub_calls()
        self.assertEqual(c[c.index("--") + 1:][1], self.cues)
        tools = [x["tool"] for x in out["execution"]["results"]]
        self.assertLess(tools.index("dubbing/dub"), tools.index("ffmpeg-skill/loudness"))
        self.assertEqual(out["receipt"]["obligations"]["credit"], ["JVNV corpus voice jvnv-F1-jp (CC BY-SA 4.0)"], "obligations.credit carried into the ProductionReceipt")
        self.assertIn("credit: JVNV corpus voice jvnv-F1-jp", Path(out["report"]).read_text(encoding="utf-8"))
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        obs = next(o for o in prov["skill_observations"] if o["kind"] == "speech.dub")
        self.assertEqual((obs["data"]["lifecycle"], obs["data"]["translation"]["provided_by"]), ("EXPERIMENTAL", "user"))

    def test_plan_without_dub_keeps_its_hash_shape(self):
        src = fake_media(self.tmp, "talk.mp4")
        ir = dub_service(self.tmp).plan([src], "youtube")
        self.assertNotIn("speech", ir.doc, "no empty section: an unchanged request keeps its plan hash")
        self.assertEqual(plan_hash(ir.doc), plan_hash(dict(ir.doc, speech={})))

    def test_reference_voice_with_the_users_consent_record(self):
        src = fake_media(self.tmp, "talk.mp4")
        svc = dub_service(self.tmp)
        r = plan_and_render(svc, [src], {"dub.cues": self.cues, "dub.lang": "en", "dub.reference": self.ref, "dub.consent": self.consent}, profile="youtube", name="ref")
        self.assertEqual(r["out"]["execution"]["status"], "COMPLETED", r["out"]["execution"])
        [c] = self.dub_calls()
        self.assertIn(f"--reference={self.ref}", c)
        self.assertIn(f"--consent={self.consent}", c)
        res = next(x for x in r["out"]["execution"]["results"] if x["tool"] == "dubbing/dub")
        self.assertEqual(res["data"]["consent"]["voice_owner"], {"name": "Speaker A"})

    def test_consent_required_at_render_blocks_once_and_is_not_retried(self):
        src = fake_media(self.tmp, "talk.mp4")
        Path(self.consent).write_text(json.dumps({"fake_valid": False}), encoding="utf-8")
        os.environ["FAKE_DUB_MODE"] = "consent_retryable"   # even a result claiming retryable: true is not retried
        before = Path(self.consent).read_bytes()
        r = plan_and_render(dub_service(self.tmp), [src], {"dub.cues": self.cues, "dub.lang": "en", "dub.reference": self.ref, "dub.consent": self.consent},
                            profile="youtube", name="noconsent")
        out = r["out"]
        self.assertNotEqual(out["execution"]["status"], "COMPLETED")
        self.assertEqual(len(self.dub_calls()), 1, "attempted once, never retried")
        self.assertEqual([x["class"] for x in out["execution"]["recovery"]], ["CONSENT_REQUIRED"])
        self.assertIn("the agent never makes one", out["execution"]["recovery"][0]["reason"])
        self.assertEqual(Path(self.consent).read_bytes(), before, "the agent never writes the consent record")
        self.assertEqual(sorted(p.name for p in Path(self.consent).parent.iterdir()), sorted(["consent.json", "en.srt", "talk.mp4", "voice.wav"]), "no record created")

    def test_engine_missing_blocks_with_the_skills_code(self):
        os.environ["FAKE_DUB_ENGINES"] = "chatterbox"
        ir = dub_service(self.tmp).plan([fake_media(self.tmp, "talk.mp4")], "youtube", user_requirements={"dub.cues": self.cues, "dub.lang": "en"})
        b = next(x for x in ir.doc["decisions"] if x["subject"] == OP)
        self.assertEqual((b["type"], b["approval"], b["params"]["code"], b["params"]["engine"]), ("BLOCK", "BLOCK", "ENGINE_UNAVAILABLE", "chatterbox"))
        self.assertIn("fake install line", b["reason"])
        self.assertIn("ready instead: style_bert_vits2", b["reason"])
        self.assertIn("capability:dubbing:DUB", b["evidence"])
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        ir = dub_service(self.tmp).plan([fake_media(self.tmp, "talk.mp4")], "youtube", user_requirements={"dub.cues": self.cues, "dub.lang": "en", "dub.engine": "style_bert_vits2"})
        self.assertEqual(next(x for x in ir.doc["decisions"] if x["subject"] == OP)["type"], "TRANSFORM", "judged on the engine the plan names")
        self.assertEqual(self.dub_calls(), [])

    def test_without_dubbing_the_step_is_handed_off(self):
        svc = dub_service(self.tmp, installed=False)
        ir = svc.plan([fake_media(self.tmp, "talk.mp4")], "youtube", user_requirements={"dub.cues": self.cues, "dub.lang": "en"})
        step = next(s for s in ir.doc["plan"]["steps"] if s["skill"] == "speech_dub")
        self.assertEqual((step["tool"], step["status"]), (None, "BLOCKED"))
        [h] = [h for h in svc.handoffs(ir) if h["skill"] == "speech_dub"]
        self.assertEqual(h["repositories"], ["dubbing-skill"])
        self.assertTrue(any("dubbing-skill" in c for c in h["install"]))

    def test_refused_where_a_dub_cannot_be_placed(self):
        svc = dub_service(self.tmp)
        a, b = fake_media(self.tmp, "a.mp4"), fake_media(self.tmp, "b.mp4")
        ir = svc.plan([a, b], "youtube", user_requirements={"dub.cues": self.cues, "dub.lang": "en"})
        self.assertEqual([x["type"] for x in ir.doc["decisions"] if x["subject"] == OP], ["BLOCK"])
        self.assertIn("one programme's timeline", next(x for x in ir.doc["decisions"] if x["subject"] == OP)["reason"])
        ir = svc.plan([fake_media(self.tmp, "talk.mp4")], "generic", user_requirements={"dub.cues": self.cues, "dub.lang": "en", "audio.production": True})
        self.assertIn("audio production path", next(x for x in ir.doc["decisions"] if x["subject"] == OP)["reason"])
        ir = dub_service(self.tmp, audio=False).plan([fake_media(self.tmp, "mute.mp4")], "youtube", user_requirements={"dub.cues": self.cues, "dub.lang": "en"})
        b = next(x for x in ir.doc["decisions"] if x["subject"] == OP)
        self.assertEqual((b["type"], b["params"]["code"]), ("BLOCK", "UNSUPPORTED_MEDIA"), "DUB speaks over the programme's own sound")
        self.assertEqual(self.dub_calls(), [])

    def test_validator_refuses_a_tampered_dub(self):
        svc = dub_service(self.tmp)
        ir = svc.plan([fake_media(self.tmp, "talk.mp4")], "youtube", user_requirements={"dub.cues": self.cues, "dub.lang": "en"})
        d = ir.doc
        self.assertEqual(check_speech_operations(d), [])
        for patch, msg in (({"reference": self.ref}, "only together with its owner's consent record"), ({"input": "elsewhere"}, "after its edits"),
                           ({"cues": "/etc/en.srt"}, "outside the allowed input roots"), ({"duck": 99}, "within [0, 40]"), ({"engine": "voicevox"}, "is not one"),
                           ({"strength": 1}, "not in the operation's vocabulary")):
            bad = json.loads(json.dumps(d))
            bad["speech"]["operations"][0].update(patch)
            self.assertTrue(any(msg in e for e in check_speech_operations(bad)), (msg, check_speech_operations(bad)))
        bad = json.loads(json.dumps(d))
        next(x for x in bad["decisions"] if x["subject"] == OP)["approval"] = "AUTO"
        from video_agent.project import ProjectIR
        self.assertFalse(svc.validate(ProjectIR(bad)).ok, "an AUTO approval written into the IR is refused (the basis says CONFIRM)")


if __name__ == "__main__":
    unittest.main()
