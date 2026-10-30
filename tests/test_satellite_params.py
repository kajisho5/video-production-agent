"""What the satellites already offer (ECOSYSTEM_ROADMAP step 6). Part 1 (ADR-062): color-grading-skill 0.4.0's PRIMARY_CORRECTION
gamma / lift / gain / levels / curves, and video-editing-skill's TRIM (an explicit in / out range) and ROTATE (lowered here together
with FREEZE / REVERSE). Part 2 (ADR-063): audio-production-skill 0.4.2's MIX / NOISE_REDUCTION / DYNAMICS and subtitle-skill 0.2.0's
render `mode: "mux"` (a soft subtitle track). Part 3 (ADR-064): every element of motion-graphics-skill 0.2.3, with every parameter. Each new parameter: a `--set` key checked by the shared validator (and therefore a brief
key, ADR-061), a decision, an IR record, a plan step and the compiled Skill request, on the fake engine and the fake Skills
(tests/test_integration.py runs them on the real Skills)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter  # noqa: E402
from pipeline_harness import clear_modes, pipeline_service, plan_and_render  # noqa: E402
from pipeline_harness import fake_media as pipeline_media  # noqa: E402
from test_unit import FakeCaps, fake_media, make_service  # noqa: E402
from video_agent.agent.editing import EditRequirementError, parse_edit_requirements  # noqa: E402
from video_agent.agent.finishing import (COLOR_CURVES_PRESETS, ELEMENT_TYPES, MOTION_BASE, MOTION_KEYS, parse_color_requirements, parse_motion_requirements,  # noqa: E402
                                         picture_size, placement_problems)
from video_agent.execution.compiler import lower_graphics_render  # noqa: E402
from video_agent.tools.motion_graphics.adapter import pinned_contract as motion_contract  # noqa: E402
from video_agent.agent.requirements import requirement_map  # noqa: E402
from video_agent.execution.compiler import CompileError, lower_audio_op, lower_captions, lower_color_op, lower_video_trim  # noqa: E402
from video_agent.agent.audio import (DYNAMICS_KEYS, DYNAMICS_STAGES, MAX_MIX_INPUTS, MIX_GAIN_RANGE, NOISE_REDUCTION_CONTRACT_STRENGTH, NOISE_REDUCTION_MODES,  # noqa: E402
                                     NOISE_REDUCTION_STRENGTH,
                                     parse_audio_requirements)
from video_agent.agent.brief import parse_brief  # noqa: E402
from video_agent.agent.subtitles import parse_subtitle_requirements  # noqa: E402
from video_agent.tools.audio_production.adapter import pinned_contract as audio_contract  # noqa: E402
from video_agent.tools.subtitle.adapter import pinned_contract as subtitle_contract  # noqa: E402
from pipeline_harness import AUDIO_PRODUCTION_CAPS, PipelineCaps  # noqa: E402
from video_agent.models import Requirement  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.service import BRIEF_REQUIREMENT_KEYS, _check_requirement_keys  # noqa: E402
from video_agent.tools.color_grading.adapter import pinned_contract as color_contract  # noqa: E402
from video_agent.tools.video_editing.adapter import pinned_contract as editing_contract  # noqa: E402
from video_agent.tools.video_editing.lowering import ARGS, Lowering, params_for  # noqa: E402

FAKE_VE = str(Path(__file__).resolve().parent / "fake_video_editing.py")
NEW_COLOR = {"color.gamma": 1.2, "color.lift": 0.05, "color.gain": -0.1, "color.levels_in_black": 16, "color.levels_in_white": 235,
             "color.levels_out_black": 4, "color.levels_out_white": 250, "color.curves": "medium_contrast"}


def reqs(d):
    return requirement_map([Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in d.items()])


class ColorParameterTests(unittest.TestCase):
    """color.gamma / .lift / .gain / .levels_* / .curves → the PRIMARY_CORRECTION parameters of the same names."""

    def test_keys_match_the_pinned_contract(self):
        """Every new key names a PRIMARY_CORRECTION parameter of the pinned 0.4.0 contract, with the contract's own range / enum."""
        op = next(o for o in color_contract()["operations"] if o["type"] == "PRIMARY_CORRECTION")
        got = parse_color_requirements(reqs(NEW_COLOR))["color.primary_correction"]["params"]
        self.assertEqual(sorted(got), sorted(k.split(".", 1)[1] for k in NEW_COLOR))
        for name, v in got.items():
            spec = op["parameters"][name]
            if "enum" in spec:
                self.assertIn(v, spec["enum"]); self.assertEqual(tuple(spec["enum"]), COLOR_CURVES_PRESETS)
            else:
                self.assertTrue(spec["min"] <= v <= spec["max"], name)
                self.assertIsInstance(v, int if spec["type"] == "integer" else float, name)
        self.assertEqual(got["levels_in_black"], 16); self.assertEqual(got["curves"], "medium_contrast")
        # one operation for the old and the new parameters together
        both = parse_color_requirements(reqs({"color.exposure": 0.5, "color.gamma": 0.8}))
        self.assertEqual(both["color.primary_correction"]["params"], {"exposure": 0.5, "gamma": 0.8})

    def test_refusals(self):
        for bad, msg in (({"color.gamma": 0.05}, "within"), ({"color.gamma": 11}, "within"), ({"color.lift": -1.5}, "within"), ({"color.gain": "much"}, "number"),
                         ({"color.levels_in_black": 16.5}, "integer"), ({"color.levels_out_white": 300}, "within"),
                         ({"color.levels_in_black": 200, "color.levels_in_white": 100}, "below"), ({"color.levels_in_black": 255}, "below"),
                         ({"color.levels_out_white": 0}, "below"), ({"color.curves": "none"}, "one of"), ({"color.curves": "sepia"}, "one of")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                parse_color_requirements(reqs(bad))
            self.assertIn(msg, str(cm.exception), str(bad))
            with self.assertRaises(ValueError, msg=str(bad)):
                _check_requirement_keys(bad)   # the same refusal before any analysis runs (the --set gate, ADR-042 / ADR-061)
        with self.assertRaises(ValueError) as cm:
            _check_requirement_keys({"color.levels": 16})
        self.assertIn("unknown requirement key", str(cm.exception))

    def test_set_and_brief_keys(self):
        """The --set gate accepts every new key, and a brief may yield them (BRIEF_REQUIREMENT_KEYS is derived, not listed)."""
        _check_requirement_keys(dict(NEW_COLOR))
        for k in list(NEW_COLOR) + ["edit.trim", "edit.rotate", "edit.rotate.flip"]:
            self.assertIn(k, BRIEF_REQUIREMENT_KEYS, k)

    def test_compiled_request_carries_the_parameters_by_name(self):
        op = {"type": "color.primary_correction", "asset": "a", "gamma": 1.2, "levels_in_black": 16, "curves": "vintage", "unknown": 1}
        args = lower_color_op("color-grading/run", op, "a_trim", "a_pc")
        self.assertEqual(args, {"operation": "PRIMARY_CORRECTION", "input": "a_trim", "output": "a_pc", "format": "mp4", "gamma": 1.2, "levels_in_black": 16, "curves": "vintage"})

    def test_pipeline_decision_ir_step_and_skill_request(self):
        clear_modes()
        tmp = os.path.realpath(tempfile.mkdtemp())
        a = pipeline_media(tmp, "a.mp4")
        svc = pipeline_service(tmp)
        r = plan_and_render(svc, [a], dict(NEW_COLOR), name="colour")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        dec = next(x for x in d["decisions"] if x["subject"] == "color.primary_correction")
        self.assertEqual((dec["type"], dec["provenance"], dec["approval"]), ("TRANSFORM", "USER", "AUTO"))
        self.assertEqual(sorted(x["key"] for x in d["requirements"] if x["id"] in dec["evidence"]), sorted(NEW_COLOR))
        op = d["color"]["operations"][0]
        self.assertEqual({k.split(".", 1)[1]: op.get(k.split(".", 1)[1]) for k in NEW_COLOR}, {k.split(".", 1)[1]: v for k, v in NEW_COLOR.items()})
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "color_primary_correction")
        self.assertEqual({k: v for k, v in step["params"].items() if k != "asset"}, {k.split(".", 1)[1]: v for k, v in NEW_COLOR.items()})
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = next(x for x in out["execution"]["results"] if x["tool"] == "color-grading/run")
        for k, v in NEW_COLOR.items():
            self.assertEqual(res["data"]["operation"]["parameters"][k.split(".", 1)[1]], v, k)
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        row = next(e for e in prov["operations"] if e["tool"] == "color-grading/run")
        self.assertEqual(row["args"]["operation"], "PRIMARY_CORRECTION"); self.assertEqual(row["args"]["curves"], "medium_contrast")
        self.assertEqual(row["decision"], [dec["id"]])
        clear_modes()

    def test_tampered_ir_out_of_range_is_refused_by_the_schema(self):
        clear_modes()
        tmp = os.path.realpath(tempfile.mkdtemp())
        svc = pipeline_service(tmp)
        ir = svc.plan([pipeline_media(tmp, "a.mp4")], "youtube", user_requirements={"color.gamma": 1.2})
        ir.doc["color"]["operations"][0]["gamma"] = 50.0
        ir.doc["color"]["operations"][0]["curves"] = "sepia"
        errs = " ".join(svc.validate(ir).errors)
        self.assertIn("gamma", errs); self.assertIn("curves", errs)
        clear_modes()


class LoweringTests(unittest.TestCase):
    """video-editing lowering of the contract's ROTATE / FREEZE / REVERSE (and the compiler's TRIM lowering)."""

    def test_every_declared_type_is_lowered(self):
        lw = Lowering(editing_contract())
        self.assertEqual(lw.contract_only(), [])
        self.assertTrue({"ROTATE", "FREEZE", "REVERSE"} <= set(lw.supported_types()))
        for t in ("ROTATE", "FREEZE", "REVERSE"):   # the agent-side keys stay inside what the contract declares
            self.assertTrue(set(ARGS[t]) - {"input", "output"} <= set(editing_contract()["operations"][t]["parameters"]), t)

    def test_params_and_refusals(self):
        self.assertEqual(params_for("ROTATE", {"degrees": 90}), {"degrees": 90})
        self.assertEqual(params_for("ROTATE", {"flip": "v"}), {"flip": "v"})
        self.assertEqual(params_for("FREEZE", {"hold": 1.5, "at": 2}), {"hold": 1.5, "at": 2.0})
        self.assertEqual(params_for("REVERSE", {}), {})
        from video_agent.tools.base import ToolError
        for t, args in (("ROTATE", {}), ("ROTATE", {"degrees": 45}), ("ROTATE", {"degrees": "90"}), ("ROTATE", {"flip": "x"}), ("FREEZE", {}), ("FREEZE", {"hold": -1})):
            with self.assertRaises(ToolError, msg=f"{t} {args}"):
                params_for(t, args)

    def test_build_request(self):
        tmp = tempfile.mkdtemp()
        lw = Lowering(editing_contract(), workspace=tmp)
        req, out, ins = lw.build_request("video-editing/rotate", {"input": "a", "output": "b", "degrees": 270, "flip": "h"}, {"a": __file__, "b": str(Path(tmp) / "o" / "b.mp4")})
        op = req["project"]["operations"][0]
        self.assertEqual((op["type"], op["params"], op["input"]), ("ROTATE", {"degrees": 270, "flip": "h"}, "s0"))
        from video_agent.tools.base import ToolError
        with self.assertRaises(ToolError):
            lw.build_request("video-editing/rotate", {"input": "a", "output": "b", "degrees": 90, "filter": "transpose=1"}, {"a": __file__, "b": str(Path(tmp) / "b.mp4")})

    def test_trim_lowering(self):
        op = {"type": "video.trim", "asset": "a", "keep": [[2.0, 10.5]], "accurate": True}
        self.assertEqual(lower_video_trim("video-editing/trim", op, "a", "a_trim"), {"input": "a", "start": 2.0, "end": 10.5, "precision": "frame", "output": "a_trim"})
        with self.assertRaises(CompileError):
            lower_video_trim("video-editing/trim", dict(op, keep=[[0.0, 2.0], [3.0, 5.0]]), "a", "a_trim")
        # the same IR record on the other tools keeps its existing shapes
        self.assertEqual(lower_video_trim("video-editing/cut", op, "a", "a_trim")["keep"], [[2.0, 10.5]])
        self.assertEqual(lower_video_trim("ffmpeg-skill/cut", op, "a", "a_trim")["segments"], "2.000-10.500")


class TrimRotateTests(unittest.TestCase):
    """edit.trim / edit.rotate (+ .flip): requirement → decision → IR → plan step → video-editing request, on the fake video-editing-skill."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.a = fake_media(self.tmp, "a.mp4")
        self.b = fake_media(self.tmp, "b.mp4")
        os.environ.pop("FAKE_VE_MODE", None)

    def _svc(self, caps=None):
        from video_agent.tools import ToolRouter
        from video_agent.tools.video_editing import VideoEditingAdapter, VideoEditingSkill
        ad = VideoEditingAdapter(VideoEditingSkill([sys.executable, FAKE_VE], None, {}), workspace=self.tmp, allowed_inputs=[str(Path(self.a).parent)], ffmpeg_skill_dir=self.tmp)
        return make_service(self.tmp, caps=caps or FakeCaps(extra=["video-editing", "encoder:aac"]), adapter=ToolRouter([FakeAdapter(), ad]))

    def _render(self, svc, ir, name):
        p = str(Path(self.tmp) / f"{name}.json"); save_ir(ir, p)
        return svc.render(load_ir(p), p, approve=["all"])

    def test_requirement_vocabulary(self):
        def parse(**kv):
            return {k: v["params"] for k, v in parse_edit_requirements(reqs(kv)).items()}
        self.assertEqual(parse(**{"edit.trim": "2.5-14"}), {"video.trim": {"start": 2.5, "end": 14.0}})
        self.assertEqual(parse(**{"edit.trim": "0:05-1:30"}), {"video.trim": {"start": 5.0, "end": 90.0}})
        self.assertEqual(parse(**{"edit.trim": {"start": 1, "end": 3}}), {"video.trim": {"start": 1.0, "end": 3.0}})
        self.assertEqual(parse(**{"edit.rotate": 90}), {"video.rotate": {"degrees": 90}})
        self.assertEqual(parse(**{"edit.rotate": "270", "edit.rotate.flip": "v"}), {"video.rotate": {"degrees": 270, "flip": "v"}})
        self.assertEqual(parse(**{"edit.rotate": 0, "edit.rotate.flip": "h"}), {"video.rotate": {"flip": "h"}}, "a mirror alone")
        for bad, msg in (({"edit.trim": "5-2"}, "before"), ({"edit.trim": "abc"}, "START-END"), ({"edit.trim": {"start": 1}}, "START-END"), ({"edit.trim": 5}, "START-END"),
                         ({"edit.rotate": 45}, "90, 180 or 270"), ({"edit.rotate": 360}, "within"), ({"edit.rotate": 0}, "changes nothing"),
                         ({"edit.rotate.flip": "h"}, "ambiguous"), ({"edit.rotate": 90, "edit.rotate.flip": "x"}, "h (mirror")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                parse(**bad)
            self.assertIn(msg, str(cm.exception), str(bad))
            with self.assertRaises(ValueError, msg=str(bad)):
                _check_requirement_keys(bad)

    def test_trim_decision_plan_ir_and_request(self):
        svc = self._svc()
        ir = svc.plan([self.a], "generic", user_requirements={"edit.trim": "4-12"})
        d = ir.doc
        aid = list(d["assets"])[0]
        self.assertEqual((d["plan"]["status"], svc.validate(ir).errors), ("APPROVED", []))
        dec = next(x for x in d["decisions"] if x["subject"] == "video.trim")
        self.assertEqual((dec["type"], dec["approval"], dec["provenance"], dec["params"]), ("TRANSFORM", "AUTO", "USER", {"asset_id": aid, "start": 4.0, "end": 12.0}))
        self.assertEqual(dec["basis"]["approval"]["key"], "video.trim.approval")
        trim = next(op for op in d["video"]["operations"] if op["type"] == "video.trim")
        self.assertEqual((trim["keep"], trim["accurate"]), ([[4.0, 12.0]], True), "the user's in / out points, cut where they were given")
        self.assertIn(dec["id"], trim["decision_ids"])
        step = next(s for s in d["plan"]["steps"] if s["id"] == f"step_trim_{aid}")
        self.assertEqual((step["skill"], step["tool"]), ("video_trim", "video-editing/trim"))
        out = self._render(svc, ir, "trim")
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = [r for r in out["execution"]["results"] if r["tool"].startswith("video-editing/")]
        self.assertEqual([r["tool"] for r in res], ["video-editing/trim"])
        prov = json.loads((Path(self.tmp) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        row = next(e for e in prov["operations"] if e["tool"] == "video-editing/trim")
        self.assertEqual((row["skill"], sorted(row["args"])), ("video_trim", ["end", "input", "output", "precision", "start"]))
        self.assertEqual((row["args"]["start"], row["args"]["end"], row["args"]["precision"]), (4.0, 12.0, "frame"))
        self.assertIn(dec["id"], row["decision"])
        dur = next(i for i in out["qa"]["items"] if i["name"] == "duration")
        self.assertEqual(dur["status"], "PASS"); self.assertAlmostEqual(float(dur["observed"]), 8.0, delta=0.01)

    def test_trim_narrows_the_silence_trim(self):
        """Silence decisions still apply inside the range: the kept range is their intersection, one video.trim, one step."""
        svc = self._svc()
        ir = svc.plan([self.a], "generic", user_requirements={"edit.trim": "1-15"})   # the fake's silences end at 2.85 s and start at 13.85 s
        d = ir.doc
        trims = [op for op in d["video"]["operations"] if op["type"] == "video.trim"]
        self.assertEqual(len(trims), 1); self.assertEqual(trims[0]["keep"], [[2.85, 13.85]])
        subjects = {x["subject"] for x in d["decisions"] if x["id"] in trims[0]["decision_ids"]}
        self.assertEqual(subjects, {"video.trim", "silence.leading", "silence.trailing"})
        self.assertEqual([s["skill"] for s in d["plan"]["steps"] if s["id"].startswith("step_trim_")], ["video_trim"])

    def test_trim_blocks(self):
        svc = self._svc()
        ir = svc.plan([self.a, self.b], "generic", user_requirements={"edit.trim": "1-5"})
        blk = [x for x in ir.doc["decisions"] if x["approval"] == "BLOCK"]
        self.assertEqual([x["subject"] for x in blk], ["video.trim"]); self.assertIn("2 inputs", blk[0]["reason"])
        ir = svc.plan([self.a], "generic", user_requirements={"edit.trim": "1-30"})
        blk = [x for x in ir.doc["decisions"] if x["approval"] == "BLOCK"]
        self.assertEqual([x["subject"] for x in blk], ["video.trim"]); self.assertIn("not clamped", blk[0]["reason"])
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        self.assertFalse(any(s["skill"] == "video_trim" for s in ir.doc["plan"]["steps"]))
        # without video-editing the explicit trim is not silently done by another engine
        svc2 = self._svc(caps=FakeCaps(extra=["encoder:aac"]))
        ir = svc2.plan([self.a], "generic", user_requirements={"edit.trim": "1-5"})
        self.assertIn("capability.video_trim", {x["subject"] for x in ir.doc["decisions"] if x["approval"] == "BLOCK"})

    def test_rotate_decision_plan_ir_and_request(self):
        svc = self._svc()
        ir = svc.plan([self.a], "generic", user_requirements={"edit.rotate": 90, "edit.rotate.flip": "h", "edit.fit": "9:16"})
        d = ir.doc
        aid = list(d["assets"])[0]
        self.assertEqual((d["plan"]["status"], svc.validate(ir).errors), ("APPROVED", []))
        self.assertEqual([op["type"] for op in d["video"]["operations"] if op["type"] != "video.trim"], ["video.rotate", "video.fit"], "rotate before fit (EDIT_ORDER)")
        rot = next(op for op in d["video"]["operations"] if op["type"] == "video.rotate")
        self.assertEqual((rot["degrees"], rot["flip"], rot["input"], rot["output"]), (90, "h", f"{aid}_trim", f"{aid}_rotate"))
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "video_rotate")
        self.assertEqual((step["tool"], step["params"]), ("video-editing/rotate", {"asset": aid, "degrees": 90, "flip": "h"}))
        dec = next(x for x in d["decisions"] if x["subject"] == "video.rotate")
        self.assertEqual((dec["type"], dec["approval"], dec["basis"]["approval"]["key"]), ("TRANSFORM", "AUTO", "video.rotate.approval"))
        out = self._render(svc, ir, "rotate")
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        self.assertEqual([r["tool"] for r in out["execution"]["results"] if r["tool"].startswith("video-editing/")], ["video-editing/rotate", "video-editing/fit"])
        prov = json.loads((Path(self.tmp) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        row = next(e for e in prov["operations"] if e["tool"] == "video-editing/rotate")
        self.assertEqual({k: row["args"][k] for k in ("degrees", "flip")}, {"degrees": 90, "flip": "h"}); self.assertEqual(row["decision"], [dec["id"]])

    def test_picture_size_and_validator(self):
        tech = {"video": {"width": 1280, "height": 720}}
        self.assertEqual(picture_size(tech, [{"type": "video.rotate", "asset": "a", "degrees": 90}], "a"), (720, 1280))
        self.assertEqual(picture_size(tech, [{"type": "video.rotate", "asset": "a", "degrees": 180}], "a"), (1280, 720))
        self.assertEqual(picture_size(tech, [{"type": "video.rotate", "asset": "a", "flip": "h"}], "a"), (1280, 720))
        svc = self._svc()
        ir = svc.plan([self.a], "generic", user_requirements={"edit.rotate": 180})
        rot = next(op for op in ir.doc["video"]["operations"] if op["type"] == "video.rotate")
        rot["degrees"] = 45
        self.assertTrue(any("degrees" in e for e in svc.validate(ir).errors))
        del rot["degrees"]
        self.assertTrue(any("needs degrees and / or flip" in e for e in svc.validate(ir).errors))


AP_CAPS = PipelineCaps(extra=AUDIO_PRODUCTION_CAPS + [f"audio-production:{t}" for t in ("MIX", "NOISE_REDUCTION", "DYNAMICS", "NORMALIZE", "GAIN", "MONO")])
NEW_AUDIO = {"audio.production": True, "audio.mix": True, "audio.mix.levels": "0,-12", "audio.noise_reduction.strength_db": 24,
             "audio.dynamics.compressor.ratio": 3, "audio.dynamics.compressor.threshold_db": -20, "audio.dynamics.limiter": True}


class AudioMixNoiseDynamicsTests(unittest.TestCase):
    """audio.mix (+ .levels) / audio.noise_reduction.* / audio.dynamics.* → audio-production MIX / NOISE_REDUCTION / DYNAMICS (ADR-063)."""

    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())

    def tearDown(self):
        clear_modes()

    def test_vocabulary_matches_the_pinned_contract(self):
        """Every range / enum / arity here is the pinned 0.4.2 contract's own; MIX has no ducking parameter, so none is offered."""
        c = audio_contract()
        ops = {o["type"]: o for o in c["operations"]}
        self.assertEqual((ops["MIX"]["inputs"]["max"], c["max_mix_inputs"]), (MAX_MIX_INPUTS, MAX_MIX_INPUTS))
        gain = ops["MIX"]["parameters"]["levels"]["items"]["properties"]["gain_db"]
        self.assertEqual((gain["min"], gain["max"]), MIX_GAIN_RANGE)
        self.assertEqual(set(ops["MIX"]["parameters"]), {"levels"}, "the pinned MIX has no ducking / sidechain parameter")
        nr = ops["NOISE_REDUCTION"]["parameters"]
        self.assertEqual(tuple(nr["mode"]["enum"]), NOISE_REDUCTION_MODES)
        self.assertEqual((nr["strength_db"]["min"], nr["strength_db"]["max"]), NOISE_REDUCTION_CONTRACT_STRENGTH)
        # the one deliberate narrowing: 10..19.99 is in the contract but its denoiser refuses it (measured, tests/test_integration.py)
        self.assertEqual(NOISE_REDUCTION_STRENGTH, (20.0, NOISE_REDUCTION_CONTRACT_STRENGTH[1]))
        self.assertTrue(nr["mode"]["required"] and nr["strength_db"]["required"])
        dyn = ops["DYNAMICS"]["parameters"]
        for st, fields in DYNAMICS_STAGES.items():
            props = dyn[st]["properties"]
            self.assertEqual(set(props), set(fields), st)
            for f, (lo, hi) in fields.items():
                self.assertEqual((props[f]["min"], props[f]["max"]), (lo, hi), f"{st}.{f}")
        self.assertEqual(len(DYNAMICS_KEYS), 3 + sum(len(f) for f in DYNAMICS_STAGES.values()))

    def test_requirement_vocabulary(self):
        a = parse_audio_requirements(reqs(NEW_AUDIO))
        self.assertTrue(a["mix"]); self.assertEqual(a["mix_levels"], [{"gain_db": 0.0, "mute": False}, {"gain_db": -12.0, "mute": False}])
        self.assertEqual(a["noise_reduction"], {"strength_db": 24.0, "mode": "fft"}, "fft is the contract's only mode and it is required, so it is always sent")
        self.assertEqual(a["dynamics"], {"compressor": {"ratio": 3.0, "threshold_db": -20.0}, "limiter": {}})
        lv = parse_audio_requirements(reqs({"audio.production": True, "audio.mix": True, "audio.mix.levels": [0, "mute", {"gain_db": -6}, {"mute": True}]}))["mix_levels"]
        self.assertEqual(lv, [{"gain_db": 0.0, "mute": False}, {"gain_db": 0.0, "mute": True}, {"gain_db": -6.0, "mute": False}, {"gain_db": 0.0, "mute": True}])
        g = parse_audio_requirements(reqs({"audio.production": True, "audio.dynamics.gate": True}))["dynamics"]
        self.assertEqual(g, {"gate": {}}, "a stage switched on alone keeps the Skill's defaults")

    def test_refusals(self):
        base = {"audio.production": True}
        for bad, msg in (({"audio.mix": True}, "audio.production=true"),   # no switch
                         (dict(base, **{"audio.mix": True, "audio.concat": True}), "both build"),
                         (dict(base, **{"audio.mix.levels": "0,-6"}), "needs audio.mix=true"),
                         (dict(base, **{"audio.mix": True, "audio.mix.levels": "mute,0"}), "cannot be muted"),
                         (dict(base, **{"audio.mix": True, "audio.mix.levels": "0,-90"}), "within"),
                         (dict(base, **{"audio.mix": True, "audio.mix.levels": [{"gain": 1}]}), "gain_db, mute"),
                         (dict(base, **{"audio.mix": True, "audio.mix.levels": list(range(9))}), "1..8"),
                         (dict(base, **{"audio.noise_reduction.mode": "fft"}), "strength_db"),
                         (dict(base, **{"audio.noise_reduction.strength_db": 5}), "within"),
                         (dict(base, **{"audio.noise_reduction.strength_db": 12}), "within"),   # in the contract, refused by its denoiser
                         (dict(base, **{"audio.noise_reduction.strength_db": 20, "audio.noise_reduction.mode": "rnn"}), "fft"),
                         (dict(base, **{"audio.dynamics.compressor.ratio": 50}), "within"),
                         (dict(base, **{"audio.dynamics.limiter.ceiling_db": 3}), "within"),
                         (dict(base, **{"audio.dynamics.gate": False, "audio.dynamics.gate.ratio": 2}), "ambiguous"),
                         (dict(base, **{"audio.dynamics.compressor": False}), "at least one stage")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                parse_audio_requirements(reqs(bad))
            self.assertIn(msg, str(cm.exception), str(bad))
            with self.assertRaises(ValueError, msg=str(bad)):
                _check_requirement_keys(bad)
        for key in ("audio.noise_reduction", "audio.dynamics", "audio.dynamics.compressor.mix", "audio.mix.duck"):
            with self.assertRaises(ValueError, msg=key) as cm:
                _check_requirement_keys(dict(base, **{key: True}))
            self.assertIn("unknown requirement key", str(cm.exception))

    def test_set_and_brief_keys(self):
        _check_requirement_keys(dict(NEW_AUDIO))
        for k in ("audio.mix", "audio.mix.levels", "audio.noise_reduction.strength_db", "audio.noise_reduction.mode", "subtitle.mode") + DYNAMICS_KEYS:
            self.assertIn(k, BRIEF_REQUIREMENT_KEYS, k)

    def test_compiled_requests_carry_the_parameters_by_name(self):
        mix = {"type": "audio.mix", "asset": "programme_audio", "inputs": ["a", "b"], "levels": [{"gain_db": 0.0, "mute": False}, {"gain_db": -12.0, "mute": False}]}
        self.assertEqual(lower_audio_op("audio-production/run", mix, ["a_cur", "b_cur"], "programme_audio"),
                         {"operation": "MIX", "inputs": ["a_cur", "b_cur"], "levels": mix["levels"], "output": "programme_audio"})
        nr = {"type": "audio.noise_reduction", "asset": "a", "mode": "fft", "strength_db": 24.0}
        self.assertEqual(lower_audio_op("audio-production/run", nr, "a", "a_nr"), {"operation": "NOISE_REDUCTION", "input": "a", "mode": "fft", "strength_db": 24.0, "output": "a_nr"})
        dyn = {"type": "audio.dynamics", "asset": "a", "compressor": {"ratio": 3.0}, "limiter": {}}
        self.assertEqual(lower_audio_op("audio-production/run", dyn, "a", "a_dyn"), {"operation": "DYNAMICS", "input": "a", "compressor": {"ratio": 3.0}, "limiter": {}, "output": "a_dyn"})
        with self.assertRaises(CompileError):
            lower_audio_op("ffmpeg-skill/audio", nr, "a", "a_nr")

    def test_pipeline_decision_ir_step_and_skill_request(self):
        v = pipeline_media(self.tmp, "voice.wav", video=False, duration=16.0)   # the fake's silences: leading 0-3 s, trailing from 13.7 s
        m = pipeline_media(self.tmp, "music.wav", video=False, duration=20.0, channels=2)
        svc = pipeline_service(self.tmp, caps=AP_CAPS)
        r = plan_and_render(svc, [v, m], dict(NEW_AUDIO, **{"audio.normalize": True, "audio.loudness.target_lufs": -16}), profile="generic", name="mix")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        ids = list(d["assets"])
        self.assertEqual([s["skill"] for s in d["plan"]["steps"]], ["audio_mix", "audio_noise_reduction", "audio_dynamics", "audio_normalize"],
                         "no silence cut on a mixed input; the chain runs on the mix in the fixed order")
        self.assertFalse([x for x in d["decisions"] if x["subject"].startswith("silence.") and x["type"] == "REMOVE"], "a mixed input keeps its own timeline")
        dec = {x["subject"]: x for x in d["decisions"]}
        for subj, key in (("audio.mix", "audio.mix.approval"), ("audio.noise_reduction", "audio.noise_reduction.approval"), ("audio.dynamics", "audio.dynamics.approval")):
            self.assertEqual((dec[subj]["type"], dec[subj]["approval"], dec[subj]["provenance"], dec[subj]["basis"]["approval"]["key"]), ("TRANSFORM", "AUTO", "USER", key), subj)
        self.assertIn("first input", dec["audio.mix"]["reason"])
        mix = next(op for op in d["audio"]["operations"] if op["type"] == "audio.mix")
        self.assertEqual((mix["inputs"], mix["timeline_duration"], mix["levels"][1]["gain_db"]), (ids, 16.0, -12.0), "the mix lasts as long as the first input")
        self.assertEqual([sg["source_range"] for sg in mix["segments"]], [[0.0, 16.0], [0.0, 16.0]])
        self.assertEqual([op["type"] for op in d["audio"]["operations"]], ["audio.mix", "audio.noise_reduction", "audio.dynamics", "audio.loudness"])
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "audio_dynamics")
        self.assertEqual((step["tool"], step["params"]), ("audio-production/run", {"asset": "programme_audio", "compressor": {"ratio": 3.0, "threshold_db": -20.0}, "limiter": {}}))
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = [x for x in out["execution"]["results"] if x["tool"] == "audio-production/run"]
        self.assertEqual([x["data"]["operation_type"] for x in res], ["MIX", "NOISE_REDUCTION", "DYNAMICS", "NORMALIZE"])
        self.assertEqual(res[1]["data"]["operation"]["parameters"], {"mode": "fft", "strength_db": 24.0})
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        row = next(e for e in prov["operations"] if e["skill"] == "audio_mix")
        self.assertEqual((row["args"]["operation"], row["args"]["levels"]), ("MIX", mix["levels"])); self.assertEqual(row["decision"], [dec["audio.mix"]["id"]])
        items = {i["name"]: i for i in out["qa"]["items"]}
        self.assertEqual((items["duration"]["status"], items["audio_only"]["status"]), ("PASS", "PASS"))

    def test_mix_blocks(self):
        v = pipeline_media(self.tmp, "voice.wav", video=False, duration=16.0)
        m = pipeline_media(self.tmp, "music.wav", video=False, duration=20.0)
        svc = pipeline_service(self.tmp, caps=AP_CAPS)
        ir = svc.plan([v, m], "generic", user_requirements={"audio.production": True, "audio.mix": True, "audio.mix.levels": "0,-6,-6"})
        blk = [x for x in ir.doc["decisions"] if x["approval"] == "BLOCK"]
        self.assertEqual([x["subject"] for x in blk], ["audio.mix"]); self.assertIn("not guessed", blk[0]["reason"])
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        ir = svc.plan([v], "generic", user_requirements={"audio.production": True, "audio.mix": True})
        self.assertEqual([x["subject"] for x in ir.doc["decisions"] if x["approval"] == "BLOCK"], ["audio.mix"])
        # without the Skill's MIX capability the mix is not done by another engine
        svc2 = pipeline_service(self.tmp, caps=PipelineCaps(extra=AUDIO_PRODUCTION_CAPS))
        ir = svc2.plan([v, m], "generic", user_requirements={"audio.production": True, "audio.mix": True})
        self.assertIn("capability.audio_mix", {x["subject"] for x in ir.doc["decisions"] if x["approval"] == "BLOCK"})

    def test_noise_reduction_and_dynamics_on_one_input(self):
        """On a single input the chain follows its silence cut: cut → noise reduction → gain → mono → dynamics."""
        v = pipeline_media(self.tmp, "voice.wav", video=False, duration=16.0)
        svc = pipeline_service(self.tmp, caps=AP_CAPS)
        ir = svc.plan([v], "generic", user_requirements={"audio.production": True, "audio.noise_reduction.strength_db": 20, "audio.gain": -3, "audio.channels": "mono",
                                                        "audio.dynamics.gate.threshold_db": -45})
        d = ir.doc
        self.assertEqual(svc.validate(ir).errors, [])
        self.assertEqual([s["skill"] for s in d["plan"]["steps"]], ["audio_cut", "audio_noise_reduction", "audio_gain", "audio_mono", "audio_dynamics"])

    def test_tampered_ir_is_refused(self):
        v = pipeline_media(self.tmp, "voice.wav", video=False, duration=16.0)
        m = pipeline_media(self.tmp, "music.wav", video=False, duration=20.0)
        svc = pipeline_service(self.tmp, caps=AP_CAPS)
        ir = svc.plan([v, m], "generic", user_requirements=dict(NEW_AUDIO))
        ops = {op["type"]: op for op in ir.doc["audio"]["operations"]}
        ops["audio.noise_reduction"]["strength_db"] = 5.0
        ops["audio.dynamics"]["compressor"]["ratio"] = 50.0
        errs = " ".join(svc.validate(ir).errors)
        self.assertIn("strength_db", errs); self.assertIn("ratio", errs)   # the schema closes the ranges
        ops["audio.noise_reduction"]["strength_db"], ops["audio.dynamics"]["compressor"]["ratio"] = 24.0, 3.0
        ops["audio.mix"]["levels"][0]["mute"] = True
        self.assertIn("cannot be muted", " ".join(svc.validate(ir).errors))   # the validator: the first input sets the length
        ops["audio.mix"]["levels"][0]["mute"] = False
        ops["audio.mix"]["levels"].append({"gain_db": 0.0})
        self.assertIn("one {gain_db, mute} per input", " ".join(svc.validate(ir).errors))


NORMALIZE = {"audio.normalize": True, "audio.loudness.target_lufs": -16}   # a loudness pass after the mux (it keeps the subtitle stream)


class SubtitleMuxTests(unittest.TestCase):
    """subtitle.mode=mux → subtitle-skill render with `mode: "mux"`: a soft subtitle track in the deliverable (ADR-063)."""

    def setUp(self):
        clear_modes()
        os.environ.pop("FAKE_DROPS_SUBS", None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())

    def tearDown(self):
        clear_modes()
        os.environ.pop("FAKE_DROPS_SUBS", None)

    def test_vocabulary(self):
        self.assertEqual(subtitle_contract()["capabilities"]["render_modes"], ["burn", "mux"])
        self.assertIn("mode", subtitle_contract()["parameters"]["render"])
        sub = parse_subtitle_requirements(reqs({"subtitle": True, "subtitle.mode": "mux"}))
        self.assertEqual((sub["mux"], sub["burn_in"]), (True, False))
        sub = parse_subtitle_requirements(reqs({"subtitle": True, "subtitle.mode": "burn"}))
        self.assertEqual((sub["mux"], sub["burn_in"]), (False, True), "mode=burn is the same as burn_in=true")
        for bad, msg in (({"subtitle.mode": "mux"}, "ambiguous"), ({"subtitle": True, "subtitle.mode": "soft"}, "one of"),
                         ({"subtitle": True, "subtitle.mode": "mux", "subtitle.burn_in": True}, "contradict"),
                         ({"subtitle": True, "subtitle.mode": "burn", "subtitle.burn_in": False}, "contradict")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                parse_subtitle_requirements(reqs(bad))
            self.assertIn(msg, str(cm.exception), str(bad))
            with self.assertRaises(ValueError, msg=str(bad)):
                _check_requirement_keys(bad)
        cands, _ = parse_brief("captions as soft subtitles, please")
        self.assertIn({"key": "subtitle.mode", "value": "mux", "phrase": "soft subtitles"}, cands)
        cands, unmapped = parse_brief("字幕は焼き込みと字幕トラックの両方")
        self.assertNotIn("subtitle.mode", [c["key"] for c in cands]); self.assertNotIn("subtitle.burn_in", [c["key"] for c in cands])
        self.assertTrue(any("not guessed" in u["reason"] for u in unmapped))

    def test_compiled_request(self):
        op = {"type": "captions.mux", "asset": "a", "input": "a_trim", "sidecar": "a_captions", "output": "a_subs", "_language": "en", "_cues": [{"id": "c1", "start": 0.0, "end": 1.0, "text": "hi"}]}
        args = lower_captions("subtitle/render", op, "a_trim", "a_subs", "a_captions")
        self.assertEqual((args["operation"], args["mode"], args["format"], args["input"], args["sidecar"]), ("render", "mux", "srt", "a_trim", "a_captions"))
        self.assertNotIn("mode", lower_captions("subtitle/render", dict(op, type="captions.burn"), "a_trim", "a_burn", "a_captions"), "a burn-in keeps the Skill's default")

    def _media(self, **kw):
        return pipeline_media(self.tmp, "talk.mp4", segments=[[4.0, 8.0, "hello there"], [9.0, 12.0, "and goodbye"]], **kw)

    def test_pipeline_decision_ir_step_request_and_qa(self):
        """Without a platform preset the deliverable is the processed file itself, which carries the track (the QC gate measures it too)."""
        svc = pipeline_service(self.tmp)
        r = plan_and_render(svc, [self._media()], {"subtitle": True, "subtitle.mode": "mux", "qc": True, **NORMALIZE}, profile="generic", kinds=["transcript"], name="mux")
        d = r["ir"].doc
        aid = list(d["assets"])[0]
        self.assertEqual(r["validation"].errors, [])
        self.assertEqual([s["skill"] for s in d["plan"]["steps"]][:4], ["silence_cleanup", "subtitle_generation", "subtitle_mux", "loudness_normalization"])
        self.assertEqual(r["ir"].doc["plan"]["status"], "APPROVED")
        dec = next(x for x in d["decisions"] if x["subject"] == "subtitle.mux")
        self.assertEqual((dec["type"], dec["approval"], dec["risk"], dec["basis"]["approval"]["key"]), ("TRANSFORM", "AUTO", "LOW", "subtitle.mux.approval"))
        mux = next(op for op in d["captions"]["operations"] if op["type"] == "captions.mux")
        self.assertEqual((mux["input"], mux["sidecar"], mux["output"]), (f"{aid}_trim", f"{aid}_captions", f"{aid}_subs"))
        self.assertEqual(next(s for s in d["plan"]["steps"] if s["skill"] == "loudness_normalization")["inputs"], [f"{aid}_subs"], "the muxed file continues the chain")
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = next(x for x in out["execution"]["results"] if x["tool"] == "subtitle/render")
        self.assertEqual(res["data"]["mode"], "mux")
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        row = next(e for e in prov["operations"] if e["tool"] == "subtitle/render")
        self.assertEqual((row["skill"], row["args"]["mode"], row["decision"]), ("subtitle_mux", "mux", [dec["id"]]))
        track = [i for i in out["qa"]["items"] if i["name"] == "subtitle_track"]
        self.assertEqual([i["status"] for i in track], ["PASS"], "the deliverable carries the soft track")

    def test_an_export_preset_blocks(self):
        """The platform export writes the picture and the audio only (measured on the real Skills): the track could not reach the deliverable."""
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self._media()], "youtube", user_requirements={"subtitle": True, "subtitle.mode": "mux"}, kinds=["transcript"], params={"language": "ja"})
        blk = [x for x in ir.doc["decisions"] if x["approval"] == "BLOCK"]
        self.assertEqual([x["subject"] for x in blk], ["subtitle.mux"]); self.assertIn("drops subtitle streams", blk[0]["reason"])
        self.assertEqual(blk[0]["params"]["presets"], ["youtube"]); self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")

    def test_qa_fails_when_the_track_is_dropped(self):
        os.environ["FAKE_DROPS_SUBS"] = "1"   # a later step loses the stream: QA catches it in the deliverable
        svc = pipeline_service(self.tmp)
        r = plan_and_render(svc, [self._media()], {"subtitle": True, "subtitle.mode": "mux", **NORMALIZE}, profile="generic", kinds=["transcript"], name="drop")
        track = next(i for i in r["out"]["qa"]["items"] if i["name"] == "subtitle_track")
        self.assertEqual((track["status"], track["observed"]), ("FAIL", 0))
        self.assertIn("MISSING_SUBTITLE_TRACK", [x["type"] for x in r["out"]["qa"]["incidents"]])

    def test_blocks_without_an_intermediate_and_refuses_a_wrong_mode_report(self):
        svc = pipeline_service(self.tmp)
        ir = svc.plan([self._media(silences=[])], "generic", user_requirements={"subtitle": True, "subtitle.mode": "mux"}, kinds=["transcript"], params={"language": "ja"})
        self.assertEqual([x["subject"] for x in ir.doc["decisions"] if x["approval"] == "BLOCK"], ["subtitle.mux"])
        os.environ["FAKE_SUBTITLE_MODE"] = "wrong_mode"
        r = plan_and_render(pipeline_service(self.tmp), [self._media()], {"subtitle": True, "subtitle.mode": "mux"}, profile="generic", kinds=["transcript"], name="wrong")
        res = next(x for x in r["out"]["execution"]["results"] if x["tool"] == "subtitle/render")
        self.assertFalse(res["ok"]); self.assertIn("mode", json.dumps(res["data"]))


# the eight element types ADR-064 adds, each with every parameter the contract declares (text-only, so a brief may yield them too)
NEW_ELEMENTS = {"motion.bug": "@studio", "motion.bug.position": "top-left", "motion.bug.text_color": "FFCC00",
                "motion.chapter": "Part 2 -- Setup", "motion.chapter.position": "bottom-right", "motion.chapter.primary_color": "1F2A44", "motion.chapter.start": 6,
                "motion.progress": True, "motion.progress.primary_color": "E53935",
                "motion.countdown": True, "motion.countdown.count_from": 3, "motion.countdown.primary_color": "FFFFFF", "motion.countdown.start": 1, "motion.countdown.end": 4,
                "motion.sticker": "NEW", "motion.sticker.position": "bottom-left", "motion.sticker.primary_color": "FFD200",
                "motion.hook": "Wait for it", "motion.hook.text_color": "FFFFFF", "motion.hook.primary_color": "000000",
                "motion.meme": True, "motion.meme.top": "when the build", "motion.meme.bottom": "is green"}


class MotionGraphicsElementTests(unittest.TestCase):
    """Every element type and parameter of the pinned motion-graphics contract (ADR-064): `motion.<element>` + `motion.<element>.<parameter>`
    → decision `graphics.<type>` → one graphics.render element → the Skill request, with placement checked against the planned picture."""

    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())

    def tearDown(self):
        clear_modes()

    def _files(self):
        png = Path(self.tmp) / "src" / "logo.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"\x89PNG fake")
        return str(png), pipeline_media(self.tmp, "pip.mp4", duration=30.0)

    def test_vocabulary_matches_the_pinned_contract(self):
        """Every element type the contract offers has a key, and every parameter of it is reachable by its contract name."""
        c = motion_contract()
        self.assertEqual(sorted(ELEMENT_TYPES), sorted(e["type"] for e in c["element_types"]))
        self.assertEqual(sorted(ELEMENT_TYPES), sorted(c["tools"][0]["element_types"]))
        self.assertEqual(len(ELEMENT_TYPES), 12)
        for e in c["element_types"]:
            base, main = MOTION_BASE[e["type"]]
            for name in e["parameters"]:
                self.assertIn(base if name == main else f"{base}.{name}", MOTION_KEYS[e["type"]], (e["type"], name))
            fade = any(a["kind"] == "fade" and e["type"] in a["applies_to"] for a in c["animations"])
            self.assertEqual(f"{base}.fade" in MOTION_KEYS[e["type"]], fade, e["type"])
        self.assertEqual([MOTION_BASE[t][0] for t in ("title", "lower_third", "text_overlay", "image_overlay")], ["motion.title", "motion.lower_third", "motion.text", "motion.image"],
                         "the ADR-031 keys keep their names")
        got = parse_motion_requirements(reqs(NEW_ELEMENTS))
        self.assertEqual(sorted(got), ["bug", "chapter", "countdown", "hook", "meme", "progress", "sticker"])
        self.assertEqual(got["bug"]["params"], {"title": "@studio", "position": "top-left", "text_color": "FFCC00"})
        self.assertEqual(got["countdown"]["params"], {"count_from": 3, "primary_color": "FFFFFF"}); self.assertEqual((got["countdown"]["start"], got["countdown"]["end"]), (1.0, 4.0))
        self.assertEqual(got["meme"]["params"], {"top": "when the build", "bottom": "is green"}); self.assertEqual(got["progress"]["params"], {"primary_color": "E53935"})
        png, clip = self._files()
        full = parse_motion_requirements(reqs({"motion.text": "LIVE", "motion.text.font": "system:dejavu-serif", "motion.text.font_size": 64, "motion.text.box": True,
                                               "motion.text.box_color": "black@0.5", "motion.text.border_width": 0, "motion.text.margin": 40, "motion.text.opacity": 0.9,
                                               "motion.video": clip, "motion.video.chromakey": "00FF00", "motion.video.chromakey_similarity": 0.2, "motion.video.scale_percent": 30,
                                               "motion.image": png, "motion.image.scale_width": 200, "motion.title": "Hi", "motion.title.primary_color": "008080"}))
        self.assertEqual(full["video_overlay"]["params"], {"video": clip, "scale_percent": 30.0, "chromakey": "00FF00", "chromakey_similarity": 0.2})
        self.assertEqual(full["text_overlay"]["params"]["font"], "system:dejavu-serif"); self.assertIs(full["text_overlay"]["params"]["box"], True)
        self.assertEqual(full["image_overlay"]["params"], {"image": png, "scale_width": 200})

    def test_refusals(self):
        png, clip = self._files()
        for bad, msg in (({"motion.countdown": True, "motion.countdown.count_from": 61}, "within"), ({"motion.countdown": True, "motion.countdown.count_from": 2.5}, "integer"),
                         ({"motion.bug": "x", "motion.bug.position": "center"}, "one of"), ({"motion.sticker": "x" * 201}, "at most 200"),
                         ({"motion.hook": "x", "motion.hook.text_color": "white"}, "bare RRGGBB"), ({"motion.chapter": "x", "motion.chapter.primary_color": "FFCC00@0.5"}, "bare RRGGBB"),
                         ({"motion.text": "x", "motion.text.font_color": "#FFCC00"}, "colour name"), ({"motion.text": "x", "motion.text.font": "Arial"}, "registry font"),
                         ({"motion.text": "x", "motion.text.border_width": 21}, "within"), ({"motion.text": "x", "motion.text.box": "maybe"}, "true or false"),
                         ({"motion.video": clip, "motion.video.chromakey_similarity": 0}, "within"), ({"motion.video": png}, "one of"),
                         ({"motion.video": str(Path(self.tmp) / "nope.mp4")}, "not found"), ({"motion.meme": True}, "at least one of"),
                         ({"motion.progress.primary_color": "red"}, "motion.progress is not"), ({"motion.progress": False, "motion.progress.primary_color": "red"}, "ambiguous"),
                         ({"motion.chapter": "x", "motion.chapter.start": 5, "motion.chapter.end": 5}, "before")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                parse_motion_requirements(reqs(bad))
            self.assertIn(msg, str(cm.exception), str(bad))
            with self.assertRaises(ValueError, msg=str(bad)):
                _check_requirement_keys(bad)   # the same refusal before any analysis runs (the --set gate)
        for unknown in ("motion.bug.font_size", "motion.meme.text", "motion.progress.position", "motion.video_overlay", "motion.hook.fade"):
            with self.assertRaises(ValueError, msg=unknown) as cm:
                _check_requirement_keys({unknown: 1})
            self.assertIn("unknown requirement key", str(cm.exception))
        self.assertEqual(parse_motion_requirements(reqs({"motion.progress": False})), {}, "a switch set to false asks for nothing")

    def test_set_and_brief_keys(self):
        """--set takes every key; a brief may yield all of them but the two file paths (only the user names a file the agent reads)."""
        _check_requirement_keys(dict(NEW_ELEMENTS))
        for keys in MOTION_KEYS.values():
            for k in keys:
                self.assertEqual(k in BRIEF_REQUIREMENT_KEYS, k not in ("motion.image", "motion.video"), k)

    def test_placement_against_the_planned_picture(self):
        self.assertEqual(placement_problems("text_overlay", {"text": "x", "position": {"x": 1000, "y": 10}}, (1280, 720)), [])
        self.assertIn("outside the 720x1280 picture", placement_problems("text_overlay", {"text": "x", "position": {"x": 1000, "y": 10}}, (720, 1280))[0])
        self.assertEqual(placement_problems("text_overlay", {"text": "x", "position": {"x": -700, "y": -10}}, (720, 1280)), [], "negative counts from the far edge")
        self.assertIn("margin 800 from the bottom", placement_problems("text_overlay", {"text": "x", "margin": 800}, (1280, 720))[0], "the default position is bottom")
        self.assertEqual(placement_problems("image_overlay", {"image": "x", "position": "center", "margin": 2000}, (1280, 720)), [], "a centred element has no margin side")
        self.assertIn("wider", placement_problems("video_overlay", {"video": "x", "scale_width": 1300}, (1280, 720))[0])
        self.assertIn("taller", placement_problems("text_overlay", {"text": "x", "font_size": 300}, (480, 270))[0])
        self.assertEqual(placement_problems("text_overlay", {"text": "x", "position": {"x": 5000, "y": 0}}, None), [], "an unknown size is never guessed")
        self.assertEqual(picture_size({"video": {"width": 1280, "height": 720}}, [{"type": "video.rotate", "asset": "a", "degrees": 90}], "a"), (720, 1280))

    def test_compiled_request(self):
        op = {"type": "graphics.render", "asset": "a", "image": "/x/logo.png", "video": "/x/pip.mp4", "elements": [
            {"id": "el1_video_overlay", "type": "video_overlay", "start": 0.0, "end": 5.0, "parameters": {"video": "/x/pip.mp4", "chromakey": "00FF00"}},
            {"id": "el2_meme", "type": "meme", "start": 0.0, "end": 5.0, "parameters": {"top": "a"}}]}
        args = lower_graphics_render("motion-graphics/run", op, "a_trim", "a_graphics", None, "a_graphics_video")
        self.assertEqual(args["elements"][0]["parameters"], {"chromakey": "00FF00", "video": "a_graphics_video"}, "the clip travels as an artifact id")
        self.assertEqual(args["elements"][1]["parameters"], {"top": "a"})
        with self.assertRaises(CompileError):
            lower_graphics_render("ffmpeg-skill/overlay", op, "a_trim", "a_graphics")

    def test_pipeline_every_element_decision_ir_step_and_skill_request(self):
        png, clip = self._files()
        req = dict(NEW_ELEMENTS, **{"motion.title": "Opening", "motion.title.text_color": "FFFFFF", "motion.lower_third": "Ada", "motion.lower_third.title": "Engineer",
                                    "motion.text": "LIVE", "motion.text.font_size": 48, "motion.text.fade": 0.5, "motion.image": png, "motion.image.opacity": 0.8,
                                    "motion.video": clip, "motion.video.position": "bottom-right", "motion.video.scale_width": 320, "motion.video.chromakey": "00FF00"})
        svc = pipeline_service(self.tmp)
        r = plan_and_render(svc, [pipeline_media(self.tmp, "talk.mp4")], req, name="mg")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        decs = {x["subject"]: x for x in d["decisions"] if x["subject"].startswith("graphics.")}
        self.assertEqual(sorted(decs), sorted(f"graphics.{t}" for t in ELEMENT_TYPES))
        for x in decs.values():
            self.assertEqual((x["type"], x["provenance"], x["approval"]), ("TRANSFORM", "USER", "AUTO"), x["subject"])
            self.assertEqual(x["basis"]["approval"]["key"], x["subject"] + ".approval")
        self.assertEqual(decs["graphics.hook"]["params"]["end"], 3.0, "a hook defaults to a 3 s opener (policy default, recorded)")
        self.assertIn("motion.hook.duration", [s_["key"] for s_ in decs["graphics.hook"]["basis"]["settings"]])
        self.assertIsNone(decs["graphics.bug"]["params"]["end"], "a bug stays to the end")
        self.assertIn("1280x720", decs["graphics.text_overlay"]["reason"], "placed on the planned picture")
        op = d["graphics"]["operations"][0]
        self.assertEqual(sorted(e["type"] for e in op["elements"]), sorted(ELEMENT_TYPES))
        self.assertEqual((op["image"], op["video"]), (png, clip))
        self.assertIn(str(Path(clip).parent), d["execution"]["allowed_inputs"])
        step = next(s_ for s_ in d["plan"]["steps"] if s_["skill"] == "motion_graphics")
        self.assertEqual(len(step["params"]["elements"]), 12); self.assertEqual(sorted(step["decision_ids"]), sorted(x["id"] for x in decs.values()))
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        row = next(e for e in prov["operations"] if e["tool"] == "motion-graphics/run")
        sent = {e["type"]: e for e in row["args"]["elements"]}
        self.assertEqual(sent["video_overlay"]["parameters"]["video"], "talk_graphics_video".replace("talk", list(d["assets"])[0]))
        self.assertEqual(sent["meme"]["parameters"], {"top": "when the build", "bottom": "is green"})
        self.assertEqual(sent["countdown"]["parameters"], {"count_from": 3, "primary_color": "FFFFFF"})
        self.assertEqual(sent["text_overlay"]["animation"], {"kind": "fade", "parameters": {"duration": 0.5}})
        res = next(x for x in out["execution"]["results"] if x["tool"] == "motion-graphics/run")
        recs = {o["type"]: o for o in res["data"]["operations"]}
        self.assertEqual(recs["video_overlay"]["parameters"]["chromakey"], "00FF00")
        self.assertEqual(recs["video_overlay"]["parameters"]["video_path"]["sha256"], hashlib.sha256(Path(clip).read_bytes()).hexdigest(), "the Skill read the user's clip")
        self.assertEqual(recs["bug"]["parameters"]["position"], "top-left")

    def test_placement_blocks_on_the_rotated_picture(self):
        """{x: 1000} is on a 1280x720 frame but not on the 720x1280 one a quarter turn delivers: BLOCK, never moved."""
        src = pipeline_media(self.tmp, "talk.mp4")
        ok = pipeline_service(self.tmp).plan([src], "generic", user_requirements={"motion.text": "LIVE", "motion.text.position": {"x": 1000, "y": 10}})
        self.assertFalse([x for x in ok.doc["decisions"] if x["approval"] == "BLOCK"])
        svc = pipeline_service(self.tmp)
        ir = svc.plan([src], "generic", user_requirements={"edit.rotate": 90, "motion.text": "LIVE", "motion.text.position": {"x": 1000, "y": 10}})
        blk = [x for x in ir.doc["decisions"] if x["approval"] == "BLOCK"]
        self.assertEqual([x["subject"] for x in blk], ["graphics.text_overlay"]); self.assertIn("720x1280", blk[0]["reason"])
        self.assertEqual(blk[0]["params"]["picture"], [720, 1280]); self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        # a tampered IR (the element moved off the frame after planning) is refused by the validator on the same size
        ok.doc["graphics"]["operations"][0]["elements"][0]["parameters"]["position"] = {"x": 1300, "y": 10}
        self.assertTrue(any("outside the 1280x720 picture" in e for e in svc.validate(ok).errors))


if __name__ == "__main__":
    unittest.main()
