"""ADR-092 (sound-design-skill roadmap step 5, agent side: BGM -> MIX): a music bed made by sound-design-skill (library-pick from the user's licensed
library, or generate-bgm) at the mix's length and placed by audio-production MIX as its last input, against a fake `sound-design` process
(tests/fake_sound_design.py), the fake audio-production process and the fake media engine. No engine, ffmpeg or network."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import AUDIO_PRODUCTION_CAPS, PipelineCaps, clear_modes, fake_media, pipeline_service  # noqa: E402

from video_agent.agent.bgm import parse_bgm_requirements  # noqa: E402
from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.capabilities.resolver import CapabilityResolver  # noqa: E402
from video_agent.models import Operation  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.service import _check_requirement_keys  # noqa: E402
from video_agent.tools.discovery import MODULE_BOOTSTRAP  # noqa: E402
from video_agent.tools.skill_process import CliSkill  # noqa: E402
from video_agent.tools.sound_design import (ERROR_CLASSES, SoundDesignAdapter, breaking_drift, check_contract, drift, locate_sound_design,  # noqa: E402
                                            pinned_contract, recovery_class)
from video_agent.execution.recovery import classify_error, next_attempt  # noqa: E402
from video_agent.tools.sound_design.adapter import OPTIONAL_ERROR_CLASSES, PINNED_CONTRACT_PATH  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_sound_design.py")
MODES = ("FAKE_SD_MODE", "FAKE_SD_READY", "FAKE_SD_CALLS", "VIDEO_AGENT_SOUND_DESIGN_DIR")
AP_MIX = AUDIO_PRODUCTION_CAPS + [f"audio-production:{t}" for t in ("MIX", "NORMALIZE", "GAIN")]


def fake_skill() -> CliSkill:
    return CliSkill("sound-design", [sys.executable, FAKE], None, {})


def sd_adapter(tmp: str) -> SoundDesignAdapter:
    ws = str(Path(tmp) / "ws")
    os.makedirs(ws, exist_ok=True)
    return SoundDesignAdapter(fake_skill(), workspace=ws, allowed_inputs=[str(Path(tmp) / "src"), str(Path(tmp) / "lib"), ws], ffmpeg_skill_dir=tmp)


def sd_capabilities(factory) -> dict:
    caps: dict = {}
    CapabilityResolver._finishing_skill(CapabilityResolver.__new__(CapabilityResolver), caps, "sound-design", fake_skill(), lambda _sk: factory(),
                                        "VIDEO_AGENT_SOUND_DESIGN_DIR", "sound-design-skill", "sound-design",
                                        per_item=lambda ad, doc: (ad.operation_status(doc), {}), item_evidence=lambda ad, doc: ad.operation_evidence(doc))
    return caps


class SdCaps(PipelineCaps):
    def __init__(self, sd: dict, **kw):
        super().__init__(**kw)
        self.sd = sd

    def resolve(self, refresh=False):
        out = super().resolve(refresh)
        out.update(self.sd)
        return out


def sd_service(tmp: str, installed: bool = True):
    caps = SdCaps(sd_capabilities(lambda: sd_adapter(tmp)) if installed else {}, extra=AP_MIX)
    svc = pipeline_service(tmp, caps=caps, transcription=False)
    if installed:
        svc._adapter.register(sd_adapter(tmp))
        svc.adapter([])
    return svc


def library(tmp: str) -> str:
    lib = Path(tmp) / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    (lib / "sound-library.json").write_text(json.dumps({"schema": "sound-design/library@1", "files": {"calm_piano.wav": {"licence": "CC0-1.0", "kind": "bgm"}}}), encoding="utf-8")
    (lib / "calm_piano.wav").write_bytes(b"RIFF")
    return str(lib)


def calls(log: str):
    return [json.loads(x) for x in Path(log).read_text(encoding="utf-8").splitlines()] if os.path.exists(log) else []


class _Base(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self._env = {k: os.environ.get(k) for k in MODES}
        for k in MODES:
            os.environ.pop(k, None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.log = str(Path(self.tmp) / "sd_calls.log")
        os.environ["FAKE_SD_CALLS"] = self.log

    def tearDown(self):
        clear_modes()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def sd_calls(self, cmd):
        return [c for c in calls(self.log) if c and c[0] == cmd]


class ContractTests(_Base):
    def test_pin_shape_and_contract_checks(self):
        c = pinned_contract()
        raw = PINNED_CONTRACT_PATH.read_text(encoding="utf-8")
        self.assertEqual(raw, json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.assertEqual((c["skill_id"], c["version"], c["contract_version"]), ("sound-design", "0.3.0", "1.0"))
        self.assertEqual((check_contract(c), drift(c), breaking_drift(c)), ([], [], []))
        self.assertEqual(set(ERROR_CLASSES) | set(OPTIONAL_ERROR_CLASSES), set(c["errors"]), "every sound-design code has a recovery class")
        self.assertEqual(c["errors"]["INSUFFICIENT_VRAM"], {"exit_code": 3, "retryable": False})
        gen = next(t for t in c["tools"] if t["tool_id"] == "sound-design/generate-bgm")
        self.assertIn("[--model acestep-v15-turbo|acestep-v15-xl-turbo]", gen["argv"]); self.assertEqual(gen["models"]["default"], "acestep-v15-turbo")
        self.assertEqual((recovery_class("LICENCE_REQUIRED"), recovery_class("ENGINE_UNAVAILABLE"), recovery_class("NEW")), ("INVALID_ARGS", "TOOL_MISSING", "SKILL_ERROR"))

    def test_refused_contracts(self):
        c = pinned_contract()
        for v in ("0.1.0", "0.2.0", "0.2.3", "0.3.0", "0.3.2"):   # ADR-107 addendum 3: 0.2.0 adds sfx-cues options only; addendum 7: 0.3.0 adds generate-bgm --model
            ok = json.loads(json.dumps(c)); ok["version"] = v
            self.assertEqual(check_contract(ok), [], v)
        for v in ("0.4.0", "1.0.0"):
            bad = json.loads(json.dumps(c)); bad["version"] = v
            self.assertTrue(any("supported range" in e for e in check_contract(bad)), v)
        # ADR-107 addendum 7: a 0.2.x contract (generate-bgm without --model, no INSUFFICIENT_VRAM) still passes check_contract (the new code is
        # not required), but against this pin its generate-bgm argv and the missing error are fatal drift: 0.3.0 or later is needed
        v2 = json.loads(json.dumps(c)); v2["version"] = "0.2.0"; v2["errors"].pop("INSUFFICIENT_VRAM")
        g2 = next(t for t in v2["tools"] if t["tool_id"] == "sound-design/generate-bgm"); g2["argv"] = [a for a in g2["argv"] if "--model" not in a]; g2.pop("models")
        self.assertEqual(check_contract(v2), [])
        self.assertEqual(sorted(d.split(":")[0] for d in breaking_drift(v2)), ["errors INSUFFICIENT_VRAM", "tools sound-design/generate-bgm.argv"])
        old = json.loads(json.dumps(c)); sfx = next(t for t in old["tools"] if t["tool_id"] == "sound-design/sfx-cues")
        sfx["argv"] = [a for a in sfx["argv"] if "min-reuse-spacing" not in a and "--character" not in a]
        sfx.pop("reuse"); sfx.pop("character")
        self.assertEqual(breaking_drift(old), [], "sfx-cues is not a tool this adapter runs: its argv change is reported, never fatal")
        self.assertTrue(drift(old))
        bad = json.loads(json.dumps(c)); next(t for t in bad["tools"] if t["tool_id"] == "sound-design/generate-bgm")["confirmation_required"] = True
        self.assertTrue(any("no confirmation" in e for e in check_contract(bad)))
        self.assertTrue(breaking_drift(bad), "a confirmation the agent would not ask for is fatal")
        bad = json.loads(json.dumps(c)); bad["execution"]["network"] = True
        self.assertTrue(any("network false" in e for e in check_contract(bad)))

    def test_locate(self):
        here = Path(self.tmp) / "x" / "sound-design-skill"
        (here / "src" / "sound_design_skill").mkdir(parents=True)
        (here / "src" / "sound_design_skill" / "cli.py").write_text("", encoding="utf-8")
        located = locate_sound_design(str(here), env={})
        self.assertEqual(located.command[1:3], ["-c", MODULE_BOOTSTRAP])
        self.assertIsNone(locate_sound_design(env={"VIDEO_AGENT_SOUND_DESIGN_DIR": str(Path(self.tmp) / "nothing"), "PATH": ""}))


class CapabilityTests(_Base):
    def test_library_ready_generate_missing_with_install(self):
        caps = sd_capabilities(lambda: sd_adapter(self.tmp))
        self.assertEqual((caps["sound-design"].status, caps["sound-design:LIBRARY"].status, caps["sound-design:BGM_GENERATE"].status), ("AVAILABLE", "AVAILABLE", "MISSING"))
        self.assertEqual(caps["sound-design:BGM_GENERATE"].evidence["code"], "ENGINE_UNAVAILABLE")
        self.assertIn("ACE-Step", caps["sound-design:BGM_GENERATE"].evidence["install"])
        [argv] = self.sd_calls("doctor")
        self.assertEqual(argv, ["doctor", "--json", "--offline", f"--ffmpeg-skill={self.tmp}"])


class AdapterTests(_Base):
    def run_op(self, tool, args):
        ad = sd_adapter(self.tmp)
        out = str(Path(ad.workspace) / "jobs" / "j" / "ops" / "00_bgm" / "bgm_bed.wav")
        paths = {"bgm_bed": out, "lib": str(Path(self.tmp) / "lib")}
        return ad, out, ad.run(Operation(tool=tool, args=dict(args, output="bgm_bed"), inputs=[], outputs=["bgm_bed"]), paths)

    def test_library_pick_argv_and_result(self):
        library(self.tmp)
        ad, out, r = self.run_op("sound-design/library-pick", {"library": "lib", "tags": ["piano"], "mood": ["calm"], "crossfade": 0.5, "duration": 16.0})
        self.assertTrue(r.ok, r.data)
        self.assertEqual(self.sd_calls("library-pick"), [["library-pick", "--json", f"--output={out}", "--kind=bgm", "--tags=piano", "--mood=calm", "--crossfade=0.5", "--duration=16",
                                                          f"--ffmpeg-skill={self.tmp}", "--timeout=3600", "--", str(Path(self.tmp) / "lib")]])
        self.assertEqual((r.data["licence"]["licence"], r.data["licence"]["verified"], r.data["ai_generated"]), ("CC0-1.0", False, False))
        self.assertEqual(r.data["outputs"][0]["duration"], 16.0)

    def test_generate_argv_and_result(self):
        os.environ["FAKE_SD_READY"] = "all"
        ad, out, r = self.run_op("sound-design/generate-bgm", {"prompt": "calm piano bed", "tags": ["ambient"], "seed": 42, "duration": 30.0})
        self.assertTrue(r.ok, r.data)
        self.assertEqual(self.sd_calls("generate-bgm"), [["generate-bgm", "--json", f"--output={out}", "--prompt=calm piano bed", "--tags=ambient", "--duration=30", "--seed=42", "--offline",
                                                          f"--ffmpeg-skill={self.tmp}", "--timeout=3600"]])
        self.assertEqual((r.data["engine"]["id"], r.data["request"]["seed"], r.data["ai_generated"]), ("acestep", 42, True))
        self.assertNotIn("--model", " ".join(self.sd_calls("generate-bgm")[0]), "ADR-107 addendum 7: no --model is sent; the Skill's default runs")
        self.assertEqual(r.data["request"]["model"], "acestep-v15-turbo", "0.3.0's request.model is recorded as the Skill wrote it")

    def test_requests_refused_before_a_process(self):
        library(self.tmp)
        for tool, args in (("sound-design/library-pick", {"library": "lib", "duration": 0.5}), ("sound-design/library-pick", {"library": str(Path(self.tmp) / "src"), "duration": 10}),
                           ("sound-design/library-pick", {"library": "lib", "duration": 10, "file": "../x.wav"}), ("sound-design/library-pick", {"library": "lib", "duration": 10, "tags": "a\nb"}),
                           ("sound-design/generate-bgm", {"prompt": "x\ny", "duration": 10}), ("sound-design/generate-bgm", {"prompt": "x", "duration": 10, "seed": -1}),
                           ("sound-design/generate-bgm", {"prompt": "x", "duration": 10, "lyrics": "la"})):
            with self.subTest(args=args):
                _, _, r = self.run_op(tool, args)
                self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_REQUEST"))
        self.assertEqual(self.sd_calls("library-pick") + self.sd_calls("generate-bgm"), [])

    def test_results_not_about_the_request_are_refused(self):
        library(self.tmp)
        for mode, why in (("short", "within"), ("bad_sha", "sha256"), ("other_output", "planned"), ("no_licence", "licence")):
            with self.subTest(mode=mode):
                os.environ["FAKE_SD_MODE"] = mode
                _, out, r = self.run_op("sound-design/library-pick", {"library": "lib", "duration": 16.0})
                self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_OUTPUT"))
                self.assertIn(why, r.data["error"]["message"])
                self.assertFalse(os.path.exists(out), "a refused output is not left behind")

    def test_skill_codes_kept(self):
        _, _, r = self.run_op("sound-design/generate-bgm", {"prompt": "x", "duration": 10})
        self.assertEqual((r.ok, r.data["error"]["code"], r.data["error"]["recovery_class"], r.exit_code), (False, "ENGINE_UNAVAILABLE", "TOOL_MISSING", 3))
        self.assertIn("ACE-Step", r.data["error"]["details"]["install"])

    def test_insufficient_vram_is_a_guided_block(self):
        """ADR-107 addendum 7: sound-design-skill 0.3.0's INSUFFICIENT_VRAM (exit 3, not retryable) keeps its code, message and details
        (details.default_model, the fallback) and blocks with a hint naming the default model; it is never retried."""
        os.environ["FAKE_SD_READY"] = "all"; os.environ["FAKE_SD_MODE"] = "fail:INSUFFICIENT_VRAM"
        _, out, r = self.run_op("sound-design/generate-bgm", {"prompt": "x", "duration": 10})
        err = r.data["error"]
        self.assertEqual((r.ok, err["code"], err["recovery_class"], err["retryable"], r.exit_code), (False, "INSUFFICIENT_VRAM", "INSUFFICIENT_VRAM", False, 3))
        self.assertEqual(err["details"]["default_model"], "acestep-v15-turbo"); self.assertIn("acestep-v15-turbo", err["message"])
        self.assertEqual(classify_error(r), "INSUFFICIENT_VRAM")
        nxt = next_attempt(r, 1, 3, 60.0)
        self.assertEqual(nxt["action"], "BLOCK"); self.assertIn("default model", nxt["reason"]); self.assertIn("acestep-v15-turbo", nxt["reason"])
        self.assertFalse(os.path.exists(out))
        self.assertEqual(recovery_class("INSUFFICIENT_VRAM"), "INSUFFICIENT_VRAM")

    def test_never_a_measurement(self):
        r = sd_adapter(self.tmp).measure("sound-design/library-pick", {})
        self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_REQUEST"))


class RequirementTests(_Base):
    def test_vocabulary(self):
        lib = library(self.tmp)
        base = {"audio.production": True, "audio.mix": True}
        self.assertEqual(parse_bgm_requirements(dict(base, **{"bgm.library": lib, "bgm.tags": "piano,calm", "bgm.crossfade": "0.5"})),
                         {"mode": "library", "library": lib, "crossfade": 0.5, "tags": ["piano", "calm"]})
        self.assertEqual(parse_bgm_requirements(dict(base, **{"bgm.prompt": "calm piano", "bgm.seed": "7"})), {"mode": "generate", "prompt": "calm piano", "seed": 7})
        self.assertIsNone(parse_bgm_requirements(base))

    def test_refused_before_analysis(self):
        lib = library(self.tmp)
        empty = Path(self.tmp) / "empty"; empty.mkdir()
        base = {"audio.production": True, "audio.mix": True}
        for reqs, why in ((dict(base, **{"bgm.library": lib, "bgm.prompt": "x"}), "exactly one"), (dict(base, **{"bgm.tags": "x"}), "exactly one"),
                          (dict(base, **{"bgm.library": str(empty)}), "not a sound-design library"), (dict(base, **{"bgm.prompt": "x", "bgm.mood": "calm"}), "library bed"),
                          (dict(base, **{"bgm.library": lib, "bgm.seed": 1}), "generated bed"), ({"bgm.library": lib}, "audio.mix=true"),
                          (dict(base, **{"bgm.prompt": "x", "bgm.lyrics": "la"}), "unknown requirement key")):
            with self.subTest(reqs=reqs):
                with self.assertRaises(EditRequirementError) as cm:
                    parse_bgm_requirements(reqs)
                self.assertIn(why, str(cm.exception))
                with self.assertRaises(ValueError):
                    _check_requirement_keys(reqs)


class PlanTests(_Base):
    def plan(self, reqs, installed=True):
        self.svc = sd_service(self.tmp, installed=installed)
        self.voice = fake_media(self.tmp, name="voice.wav", video=False, duration=16.0)
        return self.svc.plan([self.voice], "generic", user_requirements=dict({"audio.production": True, "audio.mix": True}, **reqs))

    def test_library_bed_is_the_mix_last_input_after_approval(self):
        lib = library(self.tmp)
        ir = self.plan({"bgm.library": lib, "bgm.tags": "piano", "audio.mix.levels": "0,-18", "audio.mix.duck.input": 1, "audio.mix.duck.amount_db": 10})
        d = ir.doc
        aid = next(iter(d["assets"]))
        dec = {x["subject"]: x for x in d["decisions"]}
        self.assertEqual((dec["audio.bgm"]["type"], dec["audio.bgm"]["approval"], dec["audio.bgm"]["basis"]["approval"]["key"]), ("TRANSFORM", "CONFIRM", "audio.bgm.approval"))
        self.assertEqual(dec["audio.mix"]["params"]["inputs"], [aid, "bgm"])
        bgm = next(op for op in d["audio"]["operations"] if op["type"] == "audio.bgm")
        self.assertEqual((bgm["mode"], bgm["duration"], bgm["tags"]), ("library", 16.0, ["piano"]))
        mix = next(op for op in d["audio"]["operations"] if op["type"] == "audio.mix")
        self.assertEqual((mix["inputs"], mix["duck"]["input"], mix["levels"][1]["gain_db"]), ([aid, "bgm"], 1, -18.0))
        self.assertIn(dec["audio.bgm"]["id"], mix["decision_ids"], "the mix cites the bed's decision (a rejected bed rejects the mix)")
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "bgm_library")
        self.assertEqual((step["tool"], step["params"]["duration"]), ("sound-design/library-pick", 16.0))
        self.assertEqual(self.svc.validate(ir).errors, [])
        self.assertIn(lib, d["execution"]["allowed_inputs"])
        p = str(Path(self.tmp) / "p.json"); save_ir(ir, p)
        out = self.svc.render(load_ir(p), p)
        self.assertNotEqual(out["status"], "COMPLETED", "the bed waits for the user's approval")
        self.assertEqual(self.sd_calls("library-pick"), [])
        out = self.svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED", out)
        [pick] = self.sd_calls("library-pick")
        self.assertIn("--duration=16", pick)
        mix_res = next(x for x in out["execution"]["results"] if x["tool"] == "audio-production/run" and x["data"].get("operation_type") == "MIX")
        self.assertTrue(mix_res["ok"], mix_res)

    def test_generated_bed_without_ace_step_is_handed_off(self):
        ir = self.plan({"bgm.prompt": "calm piano bed"})
        d = ir.doc
        self.assertEqual(d["plan"]["status"], "BLOCKED")
        self.assertIn("capability.bgm_generate", {x["subject"] for x in d["decisions"] if x["approval"] == "BLOCK"})
        [h] = [h for h in self.svc.handoffs(ir) if h["skill"] == "bgm_generate"]
        self.assertEqual(h["repositories"], ["sound-design-skill"])
        self.assertIn("git clone https://github.com/kajisho5/sound-design-skill && pip install -e 'sound-design-skill'", h["install"])

    def test_without_sound_design_the_bed_step_has_no_tool(self):
        lib = library(self.tmp)
        ir = self.plan({"bgm.library": lib}, installed=False)
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        self.assertTrue([h for h in self.svc.handoffs(ir) if h["skill"] == "bgm_library"])

    def test_tampered_ir_refused(self):
        lib = library(self.tmp)
        ir = self.plan({"bgm.library": lib})
        bgm = next(op for op in ir.doc["audio"]["operations"] if op["type"] == "audio.bgm")
        bgm["duration"] = 30.0
        errs = self.svc.validate(ir).errors
        self.assertTrue(any("mix's length" in e for e in errs), errs)
        bgm["duration"], bgm["prompt"] = 16.0, "x"
        errs = self.svc.validate(ir).errors
        self.assertTrue(any("takes no prompt" in e for e in errs), errs)


if __name__ == "__main__":
    unittest.main()
