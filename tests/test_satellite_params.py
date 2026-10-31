"""What the satellites already offer (ECOSYSTEM_ROADMAP step 6). Part 1 (ADR-062): color-grading-skill 0.4.0's PRIMARY_CORRECTION
gamma / lift / gain / levels / curves, and video-editing-skill's TRIM (an explicit in / out range) and ROTATE (lowered here together
with FREEZE / REVERSE). Part 2 (ADR-063): audio-production-skill 0.4.2's MIX / NOISE_REDUCTION / DYNAMICS and subtitle-skill 0.2.0's
render `mode: "mux"` (a soft subtitle track). Part 3 (ADR-064): every element of motion-graphics-skill 0.2.3, with every parameter. Part 4 (ADR-065): qc-skill 0.2.3's
delivery_package check of the deliverable set (the deliverable + its subtitle sidecar / thumbnail) as part of the QC gate. ADR-072: audio-production-skill
0.4.6's FILLER_REMOVE (at the transcript's word timings) and MIX.duck. ADR-076: the colour keys, ranges and enums read from the pinned color-grading contract. Each new parameter: a `--set` key checked by the shared validator (and therefore a brief
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
from video_agent.tools.color_grading.adapter import PINNED_CONTRACT_PATH as COLOR_PINNED_CONTRACT_PATH  # noqa: E402
from video_agent.tools.color_grading.adapter import pinned_contract as color_contract  # noqa: E402
from video_agent.agent import finishing  # noqa: E402
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


# ADR-076: the PRIMARY_CORRECTION parameter table as color-grading-skill 0.4.0's contract declares it (name: type, min, max, default / enum).
# Written out here on purpose: a re-pin that changes any of it changes the agent's vocabulary with it, and this table says so.
PRIMARY_CORRECTION_TABLE = {
    "exposure": ("number", -3.0, 3.0, 0.0), "contrast": ("number", 0.0, 2.0, 1.0), "saturation": ("number", 0.0, 2.0, 1.0),
    "temperature": ("number", 2000.0, 12000.0, 6500.0), "tint": ("number", -1.0, 1.0, 0.0),
    "gamma": ("number", 0.1, 10.0, 1.0), "lift": ("number", -1.0, 1.0, 0.0), "gain": ("number", -1.0, 1.0, 0.0),
    "levels_in_black": ("integer", 0, 255, 0), "levels_in_white": ("integer", 0, 255, 255),
    "levels_out_black": ("integer", 0, 255, 0), "levels_out_white": ("integer", 0, 255, 255),
    "curves": ("string", None, None, ("color_negative", "cross_process", "darker", "increase_contrast", "lighter", "linear_contrast",
                                      "medium_contrast", "negative", "strong_contrast", "vintage")),
}


class ColorContractVocabularyTests(unittest.TestCase):
    """ADR-076: the color.* PRIMARY_CORRECTION keys, their types, ranges and enums are read from the pinned color-grading contract and
    checked at plan time; nothing is sent unless the user sets a key."""

    def _pc(self):
        return next(o for o in color_contract()["operations"] if o["type"] == "PRIMARY_CORRECTION")["parameters"]

    def test_vocabulary_is_the_pinned_contract(self):
        pc = self._pc()
        self.assertEqual(finishing.COLOR_CONTRACT_PATH.resolve(), COLOR_PINNED_CONTRACT_PATH.resolve(), "the agent layer reads the contract the adapter pins")
        grading = [n for n in pc if n not in finishing.PRIMARY_CORRECTION_NOT_GRADING]
        self.assertEqual(list(finishing.COLOR_CORRECTION_PARAMS), grading, "every colour parameter of the contract has a key (none left out for an unhandled shape)")
        self.assertEqual(finishing.COLOR_OPERATIONS["color.primary_correction"]["params"], tuple(grading))
        for n in grading:
            spec = pc[n]
            self.assertIn(f"color.{n}", finishing.COLOR_KEYS)
            if spec["type"] in ("number", "integer"):
                self.assertEqual(finishing.COLOR_CORRECTION_RANGES[f"color.{n}"], (spec["min"], spec["max"]), n)
            else:
                self.assertEqual(finishing.COLOR_CURVES_PRESETS, tuple(spec["enum"]))
        for n in finishing.PRIMARY_CORRECTION_NOT_GRADING:
            self.assertIn(n, pc, "the excluded names are real contract parameters")
            self.assertNotIn(f"color.{n}", finishing.COLOR_KEYS)
            with self.assertRaises(ValueError):
                _check_requirement_keys({f"color.{n}": 18})
        self.assertEqual(finishing.COLOR_LEVELS_DEFAULTS, {n: pc[n]["default"] for n in pc if n.startswith("levels_")})

    def test_parameter_table(self):
        pc = self._pc()
        self.assertEqual(sorted(finishing.COLOR_CORRECTION_PARAMS), sorted(PRIMARY_CORRECTION_TABLE))
        for n, (t, lo, hi, extra) in PRIMARY_CORRECTION_TABLE.items():
            spec = pc[n]
            self.assertEqual(spec["type"], t, n)
            if t == "string":
                self.assertEqual(tuple(spec["enum"]), extra, n); self.assertIsNone(spec["default"], n)
            else:
                self.assertEqual((spec["min"], spec["max"], spec["default"]), (lo, hi, extra), n)

    def test_ir_schema_matches_the_contract(self):
        """The validator's second look at the IR (project.schema.json, $defs/color_op) declares the same type / range / enum."""
        schema = json.loads((Path(__file__).resolve().parents[1] / "schemas" / "project.schema.json").read_text(encoding="utf-8"))
        props = schema["$defs"]["color_op"]["properties"]
        for n, spec in finishing.COLOR_CORRECTION_PARAMS.items():
            if spec["type"] == "string":
                self.assertEqual(props[n], {"enum": list(spec["enum"])}, n)
            else:
                self.assertEqual((props[n]["type"], props[n]["minimum"], props[n]["maximum"]), (spec["type"], spec["min"], spec["max"]), n)

    def test_boundaries_and_types_at_plan_time(self):
        """The contract's own min / max are accepted, just outside is refused (by the parser and the --set gate), an integer parameter
        refuses a fraction, a number parameter is sent as a float, every enum value is accepted."""
        pair = {"levels_in_black": "levels_in_white", "levels_in_white": "levels_in_black", "levels_out_black": "levels_out_white", "levels_out_white": "levels_out_black"}
        for n, spec in finishing.COLOR_CORRECTION_PARAMS.items():
            k = f"color.{n}"
            if spec["type"] == "string":
                for v in spec["enum"]:
                    self.assertEqual(parse_color_requirements(reqs({k: v}))["color.primary_correction"]["params"], {n: v})
                for bad in ("NONE", "vintage;", "", 3):
                    with self.assertRaises(EditRequirementError, msg=f"{k}={bad!r}"):
                        parse_color_requirements(reqs({k: bad}))
                continue
            lo, hi = spec["min"], spec["max"]
            for v in (lo, hi):
                d = {k: v}
                if n in pair:   # keep black below white: the other point at its far end
                    d[f"color.{pair[n]}"] = hi if "black" in n else lo
                    if d[k] == d[f"color.{pair[n]}"]:
                        continue
                got = parse_color_requirements(reqs(d))["color.primary_correction"]["params"][n]
                self.assertEqual(got, v, k); self.assertIsInstance(got, int if spec["type"] == "integer" else float, k)
                _check_requirement_keys(d)
            step = 1 if spec["type"] == "integer" else 0.01
            for bad in (lo - step, hi + step):
                with self.assertRaises(EditRequirementError, msg=f"{k}={bad}") as cm:
                    parse_color_requirements(reqs({k: bad}))
                self.assertIn("within", str(cm.exception))
                with self.assertRaises(ValueError):
                    _check_requirement_keys({k: bad})
            for bad in ("bright", True, float("nan")):
                with self.assertRaises(EditRequirementError, msg=f"{k}={bad!r}"):
                    parse_color_requirements(reqs({k: bad}))
            if spec["type"] == "integer":
                with self.assertRaises(EditRequirementError) as cm:
                    parse_color_requirements(reqs({k: lo + 0.5}))
                self.assertIn("integer", str(cm.exception))
            else:
                self.assertIsInstance(parse_color_requirements(reqs({k: int(max(lo, min(hi, 1)))}))["color.primary_correction"]["params"][n], float)

    def test_no_production_defaults(self):
        """No key → no PRIMARY_CORRECTION; a DEFAULT-provenance value (a policy default) never asks for one; no shipped profile sets a
        color.* key; and a plan on every profile without colour keys has no colour operation."""
        self.assertEqual(parse_color_requirements(requirement_map([])), {})
        dflt = requirement_map([Requirement(key=f"color.{n}", value=(s["enum"][0] if s["type"] == "string" else s["max"]), provenance="DEFAULT", source="policy")
                                for n, s in finishing.COLOR_CORRECTION_PARAMS.items()])
        self.assertEqual(parse_color_requirements(dflt), {})
        only_gamma = parse_color_requirements(reqs({"color.gamma": 2.0}))["color.primary_correction"]["params"]
        self.assertEqual(only_gamma, {"gamma": 2.0}, "only the key the user set is sent; the Skill keeps its own defaults for the rest")
        prof_dir = Path(__file__).resolve().parents[1] / "profiles"
        profiles = sorted(p.stem for p in prof_dir.glob("*.json"))
        self.assertTrue(profiles)
        for p in profiles:
            self.assertNotIn('"color.', (prof_dir / f"{p}.json").read_text(encoding="utf-8"), p)
        clear_modes()
        tmp = os.path.realpath(tempfile.mkdtemp())
        svc = pipeline_service(tmp)
        a = pipeline_media(tmp, "a.mp4")
        for p in profiles:
            ir = svc.plan([a], p)
            self.assertEqual((ir.doc.get("color") or {}).get("operations") or [], [], p)
            self.assertFalse([x for x in ir.doc["decisions"] if x["subject"].startswith("color.")], p)
        clear_modes()

    def test_unhandled_shapes_are_left_out(self):
        """A parameter shape the plan-time check cannot validate exactly gets no key (fails closed) until it is decided."""
        g = finishing._grading_param
        self.assertTrue(g({"type": "number", "min": 0, "max": 1})); self.assertTrue(g({"type": "integer", "min": 0, "max": 9}))
        self.assertTrue(g({"type": "string", "enum": ["a"]}))
        for spec in ({"type": "number", "min": 0}, {"type": "integer"}, {"type": "string"}, {"type": "boolean"}, {"type": "path"},
                     {"type": "number", "min": 0, "max": 1, "required": True}, {"type": "number", "min": "0", "max": 1}):
            self.assertFalse(g(spec), spec)


class LoweringTests(unittest.TestCase):
    """video-editing lowering of the contract's ROTATE / FREEZE / REVERSE / IMAGE_INSERT (and the compiler's TRIM lowering)."""

    def test_every_declared_type_is_lowered(self):
        lw = Lowering(editing_contract())
        self.assertEqual(lw.contract_only(), [])
        self.assertTrue({"ROTATE", "FREEZE", "REVERSE", "IMAGE_INSERT", "CROP"} <= set(lw.supported_types()))
        for t in ("ROTATE", "FREEZE", "REVERSE", "IMAGE_INSERT", "CROP"):   # the agent-side keys stay inside what the contract declares
            self.assertTrue(set(ARGS[t]) - {"input", "output"} <= set(editing_contract()["operations"][t]["parameters"]), t)

    def test_params_and_refusals(self):
        self.assertEqual(params_for("ROTATE", {"degrees": 90}), {"degrees": 90})
        self.assertEqual(params_for("ROTATE", {"flip": "v"}), {"flip": "v"})
        self.assertEqual(params_for("FREEZE", {"hold": 1.5, "at": 2}), {"hold": 1.5, "at": 2.0})
        self.assertEqual(params_for("REVERSE", {}), {})
        self.assertEqual(params_for("CROP", {"x": 0, "y": 140, "width": 1920, "height": 800}), {"x": 0, "y": 140, "width": 1920, "height": 800})   # ADR-078
        from video_agent.tools.base import ToolError
        for t, args in (("ROTATE", {}), ("ROTATE", {"degrees": 45}), ("ROTATE", {"degrees": "90"}), ("ROTATE", {"flip": "x"}), ("FREEZE", {}), ("FREEZE", {"hold": -1}),
                        ("CROP", {"x": 0, "y": 0, "width": 1920}), ("CROP", {"x": -2, "y": 0, "width": 64, "height": 64}), ("CROP", {"x": 0, "y": 0, "width": 63, "height": 64})):
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

    def test_image_insert_params_and_request(self):
        """ADR-070: IMAGE_INSERT's input is an image source (kind image); duration / width / height / fps by the contract's names."""
        self.assertEqual(params_for("IMAGE_INSERT", {"duration": 3, "width": 1280, "height": 720, "fps": 25}), {"duration": 3.0, "width": 1280, "height": 720, "fps": 25.0})
        self.assertEqual(params_for("IMAGE_INSERT", {"duration": 1.5, "fps": "30000/1001"}), {"duration": 1.5, "fps": "30000/1001"})
        from video_agent.tools.base import ToolError
        for args in ({}, {"duration": 0}, {"duration": 601}, {"duration": 2, "width": 641}, {"duration": 2, "height": "720"}, {"duration": 2, "fps": "fast"}):
            with self.assertRaises(ToolError, msg=str(args)):
                params_for("IMAGE_INSERT", args)
        tmp = tempfile.mkdtemp()
        img = Path(tmp) / "card.png"; img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        lw = Lowering(editing_contract(), workspace=tmp)
        req, out, ins = lw.build_request("video-editing/image_insert", {"input": "c", "output": "o", "duration": 2.0, "width": 640, "height": 360, "fps": 30},
                                         {"c": str(img), "o": str(Path(tmp) / "o" / "still.mp4")})
        op = req["project"]["operations"][0]
        self.assertEqual((op["type"], op["input"], op["params"]), ("IMAGE_INSERT", "s0", {"duration": 2.0, "width": 640, "height": 360, "fps": 30.0}))
        self.assertEqual(req["project"]["sources"], [{"id": "s0", "path": str(img.resolve()), "kind": "image"}])
        with self.assertRaises(ToolError) as cm:   # a video in the image slot is refused before the Skill
            lw.build_request("video-editing/image_insert", {"input": "v", "output": "o", "duration": 2.0}, {"v": __file__, "o": str(Path(tmp) / "o" / "still.mp4")})
        self.assertIn("must be an image", str(cm.exception))
        with self.assertRaises(ToolError):
            lw.build_request("video-editing/image_insert", {"input": "c", "output": "o", "duration": 2.0, "filter": "loop=1"}, {"c": str(img), "o": str(Path(tmp) / "o.mp4")})

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


def png(tmp, name="card.png", sub="stills"):
    p = Path(tmp) / sub / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return str(p)


class ImageInsertTests(unittest.TestCase):
    """ADR-070: edit.image_insert (+ .duration / .position) → decision video.image_insert → IR video.image_insert → plan steps
    video_image_insert + video_concat → video-editing/image_insert + video-editing/concat, on the fake video-editing-skill."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.a = fake_media(self.tmp, "a.mp4")
        self.b = fake_media(self.tmp, "b.mp4")
        self.card = png(self.tmp)
        os.environ.pop("FAKE_VE_MODE", None)

    _svc = TrimRotateTests._svc
    _render = TrimRotateTests._render

    def test_requirement_vocabulary_and_refusals(self):
        def parse(**kv):
            return {k: v["params"] for k, v in parse_edit_requirements(reqs(kv)).items()}
        self.assertEqual(parse(**{"edit.image_insert": self.card, "edit.image_insert.duration": 3}),
                         {"video.image_insert": {"image": os.path.abspath(self.card), "duration": 3.0, "position": "before"}})
        self.assertEqual(parse(**{"edit.image_insert": self.card, "edit.image_insert.duration": "1.5", "edit.image_insert.position": "after"})["video.image_insert"]["position"], "after")
        jpg = Path(self.tmp) / "stills" / "slate.jpg"; jpg.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
        self.assertEqual(parse(**{"edit.image_insert": str(jpg), "edit.image_insert.duration": 2})["video.image_insert"]["image"], str(jpg))
        renamed = Path(self.tmp) / "stills" / "clip.png"; renamed.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16)   # a video renamed .png
        for bad, msg in (({"edit.image_insert": self.a, "edit.image_insert.duration": 2}, "must be an image asset"),
                         ({"edit.image_insert": str(renamed), "edit.image_insert.duration": 2}, "is not a PNG image"),
                         ({"edit.image_insert": str(Path(self.tmp) / "none.png"), "edit.image_insert.duration": 2}, "not found"),
                         ({"edit.image_insert": "../x.png", "edit.image_insert.duration": 2}, "traversal"),
                         ({"edit.image_insert": self.card}, "needs edit.image_insert.duration"),
                         ({"edit.image_insert": self.card, "edit.image_insert.duration": 0}, "within"),
                         ({"edit.image_insert": self.card, "edit.image_insert.duration": 601}, "within"),
                         ({"edit.image_insert": self.card, "edit.image_insert.duration": 2, "edit.image_insert.position": "middle"}, "before or after"),
                         ({"edit.image_insert.duration": 2}, "ambiguous")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                parse(**bad)
            self.assertIn(msg, str(cm.exception), str(bad))
            with self.assertRaises(ValueError, msg=str(bad)):
                _check_requirement_keys(bad)
        # set keys, but not brief keys: the value is a file path only the user names (like edit.overlay)
        self.assertNotIn("edit.image_insert", BRIEF_REQUIREMENT_KEYS)
        self.assertIn("edit.image_insert.duration", BRIEF_REQUIREMENT_KEYS)

    def test_before_decision_plan_ir_request_and_qa(self):
        svc = self._svc()
        ir = svc.plan([self.a], "generic", user_requirements={"edit.image_insert": self.card, "edit.image_insert.duration": 2.5, "edit.resize": 640})
        d = ir.doc
        aid = list(d["assets"])[0]
        self.assertEqual((d["plan"]["status"], svc.validate(ir).errors), ("APPROVED", []))
        dec = next(x for x in d["decisions"] if x["subject"] == "video.image_insert")
        self.assertEqual((dec["type"], dec["approval"], dec["provenance"], dec["basis"]["approval"]["key"]), ("TRANSFORM", "AUTO", "USER", "video.image_insert.approval"))
        self.assertEqual(dec["params"], {"asset_id": aid, "image": os.path.abspath(self.card), "duration": 2.5, "position": "before", "width": 640, "height": 360, "fps": 30.0},
                         "the still takes the picture's own frame after the resize, and the probe's rate")
        op = next(o for o in d["video"]["operations"] if o["type"] == "video.image_insert")
        self.assertEqual(op, {"type": "video.image_insert", "asset": aid, "input": f"{aid}_resize", "output": f"{aid}_image_insert", "image": os.path.abspath(self.card),
                              "inputs": [f"{aid}_still", f"{aid}_resize"], "duration": 2.5, "position": "before", "width": 640, "height": 360, "fps": 30.0,
                              "temporal_scope": {"start": 0.0, "end": round(op["temporal_scope"]["end"], 3)}, "decision_ids": [dec["id"]]})
        ist = next(s for s in d["plan"]["steps"] if s["id"] == f"step_image_insert_{aid}")
        jst = next(s for s in d["plan"]["steps"] if s["id"] == f"step_image_join_{aid}")
        self.assertEqual((ist["skill"], ist["tool"], ist["outputs"]), ("video_image_insert", "video-editing/image_insert", [f"{aid}_still"]))
        self.assertEqual((jst["skill"], jst["tool"], jst["inputs"], jst["outputs"]), ("video_concat", "video-editing/concat", [f"{aid}_still", f"{aid}_resize"], [f"{aid}_image_insert"]))
        out = self._render(svc, ir, "insert")
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        self.assertEqual([r["tool"] for r in out["execution"]["results"] if r["tool"].startswith("video-editing/")],
                         ["video-editing/resize", "video-editing/image_insert", "video-editing/concat"])
        prov = json.loads((Path(self.tmp) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        still = next(e for e in prov["operations"] if e["tool"] == "video-editing/image_insert")
        join = next(e for e in prov["operations"] if e["tool"] == "video-editing/concat")
        self.assertEqual({k: still["args"][k] for k in ("duration", "width", "height", "fps")}, {"duration": 2.5, "width": 640, "height": 360, "fps": 30.0})
        self.assertEqual((still["skill"], still["decision"]), ("video_image_insert", [dec["id"]]))
        self.assertEqual(join["args"]["inputs"], [still["args"]["output"], f"{aid}_resize"], "the still first")
        self.assertEqual({k: join["args"][k] for k in ("width", "height", "fps")}, {"width": 640, "height": 360, "fps": 30.0})
        dur = next(i for i in out["qa"]["items"] if i["name"] == "duration")
        self.assertEqual(dur["status"], "PASS", dur)

    def test_after_on_the_programme_with_a_sidecar(self):
        svc = self._svc()
        ir = svc.plan([self.a, self.b], "generic", user_requirements={"edit.concat": True, "edit.image_insert": self.card, "edit.image_insert.duration": 4,
                                                                      "edit.image_insert.position": "after"})
        d = ir.doc
        self.assertEqual((d["plan"]["status"], svc.validate(ir).errors), ("APPROVED", []))
        op = next(o for o in d["video"]["operations"] if o["type"] == "video.image_insert")
        concat = next(o for o in d["video"]["operations"] if o["type"] == "video.concat")
        self.assertEqual((op["asset"], op["inputs"]), ("programme", ["programme", "programme_still"]), "the still after the joined programme")
        self.assertAlmostEqual(op["temporal_scope"]["end"], concat["timeline_duration"] + 4, places=3)
        from video_agent.agent.editing import delivery_subjects
        self.assertAlmostEqual(delivery_subjects(d)[0]["duration"], concat["timeline_duration"] + 4, places=3)

    def test_blocks(self):
        svc = self._svc()
        base = {"edit.image_insert": self.card, "edit.image_insert.duration": 2}
        # an editor timeline names the still among what its cut list does not carry (ADR-052), like every other transform
        ir = svc.plan([self.a], "generic", user_requirements=dict(base, **{"delivery.timeline": "otio"}))
        tl = next(x for x in ir.doc["decisions"] if x["subject"] == "timeline.export")
        self.assertIn("video.image_insert", tl["params"]["not_exported"]); self.assertEqual(ir.doc["plan"]["status"], "APPROVED")
        # a QC reference master would no longer line up with the deliverable
        ref = Path(self.tmp) / "src" / "master.mp4"; ref.write_bytes(b"\x00" * 64)
        ir = svc.plan([self.a], "generic", user_requirements=dict(base, **{"qc": True, "qc.reference_video": str(ref), "qc.reference_metrics": "ssim"}))
        blk = [x for x in ir.doc["decisions"] if x["subject"] == "video.image_insert" and x["approval"] == "BLOCK"]
        self.assertEqual(len(blk), 1); self.assertIn("reference master", blk[0]["reason"]); self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        self.assertFalse(any(s["skill"] == "video_image_insert" for s in ir.doc["plan"]["steps"]))
        # the audio production path delivers no picture
        ir = svc.plan([self.a], "generic", user_requirements=dict(base, **{"audio.production": True, "audio.gain": 3}))
        self.assertTrue(any(x["subject"] == "video.image_insert" and x["approval"] == "BLOCK" and "audio production" in x["reason"] for x in ir.doc["decisions"]))
        # without video-editing nothing joins the still by another engine
        svc2 = self._svc(caps=FakeCaps(extra=["encoder:aac"]))
        ir = svc2.plan([self.a], "generic", user_requirements=base)
        self.assertIn("capability.video_image_insert", {x["subject"] for x in ir.doc["decisions"] if x["approval"] == "BLOCK"})

    def test_subtitles_with_a_still(self):
        """A sidecar keeps its times only with the still after the picture (a burn-in keeps its sidecar too); a soft track never survives the join;
        the burnt-in picture is joined finished."""
        clear_modes()
        tmp = os.path.realpath(tempfile.mkdtemp())
        svc = pipeline_service(tmp)
        media = pipeline_media(tmp, "talk.mp4", segments=[[4.0, 8.0, "hello there"], [9.0, 12.0, "and goodbye"]])
        card = png(str(Path(media).parent.parent), sub=Path(media).parent.name)   # next to the media: the harness's adapters take its folder as the input root
        base = {"subtitle": True, "edit.image_insert": card, "edit.image_insert.duration": 2}

        def plan(extra):
            ir = svc.plan([media], "generic", user_requirements=dict(base, **extra), kinds=["transcript"], params={"language": "ja"})
            return ir, [x for x in ir.doc["decisions"] if x["subject"] == "video.image_insert" and x["approval"] == "BLOCK"]
        ir, blk = plan({})
        self.assertEqual(len(blk), 1); self.assertIn("early", blk[0]["reason"])
        ir, blk = plan({"subtitle.mode": "mux", "edit.image_insert.position": "after"})
        self.assertEqual(len(blk), 1); self.assertIn("soft subtitle track", blk[0]["reason"])
        r = plan_and_render(svc, [media], dict(base, **{"subtitle.burn_in": True, "edit.image_insert.position": "after", "edit.resize": 640, "qc": True, **NORMALIZE}), profile="generic", kinds=["transcript"], name="burn")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        order = [s["skill"] for s in d["plan"]["steps"]]
        self.assertLess(order.index("subtitle_burn_in"), order.index("video_image_insert"), "the still is joined to the burnt-in picture")
        self.assertLess(order.index("video_concat"), order.index("loudness_normalization"))
        self.assertEqual(r["out"]["execution"]["status"], "COMPLETED", r["out"]["execution"])
        qa_dur = next(i for i in r["out"]["qa"]["items"] if i["name"] == "duration")
        self.assertEqual(qa_dur["status"], "PASS", qa_dur)
        clear_modes()

    def test_frame_and_validator(self):
        from video_agent.agent.image_insert import still_frame
        self.assertEqual(still_frame({"video": {"width": 1280, "height": 720, "fps": 29.97003}}, [{"type": "video.rotate", "asset": "a", "degrees": 90}], "a"), ((720, 1280, 29.97), ""))
        self.assertEqual(still_frame({"video": {"width": 1280, "height": 720, "fps": 25}}, [{"type": "video.fit", "asset": "a", "aspect": "1:1", "fps": 30}], "a")[0], (1280, 1280, 30))
        self.assertIsNone(still_frame({"video": {"width": 1281, "height": 720, "fps": 25}}, [], "a")[0])
        self.assertIsNone(still_frame({"video": {"width": 1280, "height": 720}}, [], "a")[0])
        self.assertIsNone(still_frame({"video": {}}, [], "a")[0])
        svc = self._svc()
        ir = svc.plan([self.a], "generic", user_requirements={"edit.image_insert": self.card, "edit.image_insert.duration": 2})
        op = next(o for o in ir.doc["video"]["operations"] if o["type"] == "video.image_insert")
        for k, v, frag in (("width", 641, "width"), ("duration", 0, "duration"), ("position", "middle", "position"), ("image", "/x/clip.mp4", "PNG / JPEG"),
                           ("inputs", [op["input"], f"{op['asset']}_still"], "order of position")):
            bad = json.loads(json.dumps(op)); bad[k] = v
            ir.doc["video"]["operations"] = [o if o["type"] != "video.image_insert" else bad for o in ir.doc["video"]["operations"]]
            self.assertTrue(any(frag in e for e in svc.validate(ir).errors), (k, svc.validate(ir).errors))


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
        """Every range / enum / arity here is the pinned contract's own; MIX's ducking (0.4.6) is offered since ADR-072 (AudioFillerRemoveDuckTests)."""
        c = audio_contract()
        ops = {o["type"]: o for o in c["operations"]}
        self.assertEqual((ops["MIX"]["inputs"]["max"], c["max_mix_inputs"]), (MAX_MIX_INPUTS, MAX_MIX_INPUTS))
        gain = ops["MIX"]["parameters"]["levels"]["items"]["properties"]["gain_db"]
        self.assertEqual((gain["min"], gain["max"]), MIX_GAIN_RANGE)
        # 0.4.6 (ADR-071) added MIX.duck (sidechain ducking); since ADR-072 the agent's MIX carries it beside the levels
        self.assertEqual(set(ops["MIX"]["parameters"]), {"levels", "duck"})
        from video_agent.agent.audio import OPERATIONS as AUDIO_OPS
        self.assertEqual(AUDIO_OPS["audio.mix"]["params"], ("levels", "duck"))
        nr = ops["NOISE_REDUCTION"]["parameters"]
        self.assertEqual(tuple(nr["mode"]["enum"]), NOISE_REDUCTION_MODES)
        self.assertEqual((nr["strength_db"]["min"], nr["strength_db"]["max"]), NOISE_REDUCTION_CONTRACT_STRENGTH)
        # 0.4.5 raised the contract's min to 20 (what its denoiser accepts), so the planning range and the contract agree
        self.assertEqual(NOISE_REDUCTION_STRENGTH, NOISE_REDUCTION_CONTRACT_STRENGTH)
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
                         (dict(base, **{"audio.noise_reduction.strength_db": 12}), "within"),   # below the contract's 20 (and what its denoiser accepts)
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


FR_CAPS = PipelineCaps(extra=AUDIO_PRODUCTION_CAPS + [f"audio-production:{t}" for t in ("FILLER_REMOVE", "MIX", "NORMALIZE", "GAIN")])
# a voice with leading (0-3 s) / trailing (13.7 s-) silence, one recognised segment and its word timings (two fillers: えー and um)
FR_SEGMENTS = [[3.5, 12.0, "えー 本日は um 始めます"]]
FR_WORDS = [[4.0, 4.4, "えー"], [4.6, 5.2, "本日は"], [6.0, 6.3, "um"], [7.0, 8.0, "始めます"]]


class AudioFillerRemoveDuckTests(unittest.TestCase):
    """audio.filler_remove (+ .words / .keep / .pad / .merge_gap / .declick_ms) → audio-production FILLER_REMOVE at the transcript's word
    timings, and audio.mix.duck.* → MIX.duck (ADR-072, audio-production-skill 0.4.6)."""

    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())

    def tearDown(self):
        clear_modes()

    def _voice(self, name="voice.wav", words=FR_WORDS, segments=FR_SEGMENTS, duration=16.0):
        extra = {"segments": segments} if segments is not None else {}
        if words is not None:
            extra["words"] = words
        return pipeline_media(self.tmp, name, video=False, duration=duration, **extra)

    def test_vocabulary_matches_the_pinned_contract(self):
        from video_agent.agent.audio import (DUCK_FIELDS, DUCK_REQUIRED, FILLER_DECLICK_RANGE, FILLER_LEXICON_MAX_ITEMS, FILLER_LEXICON_MAX_LENGTH, FILLER_MAX_WORDS,
                                             FILLER_MERGE_GAP_RANGE, FILLER_PAD_RANGE, FILLER_PRESETS, FILLER_WORD_MAX_LENGTH, OPERATIONS, SOURCE_OPS)
        c = audio_contract()
        ops = {o["type"]: o for o in c["operations"]}
        fr = ops["FILLER_REMOVE"]["parameters"]
        self.assertEqual(set(fr), {"words", "transcript", "lexicon", "pad", "merge_gap", "declick_ms"})
        self.assertEqual((fr["words"]["max_items"], fr["words"]["items"]["properties"]["text"]["max_length"]), (FILLER_MAX_WORDS, FILLER_WORD_MAX_LENGTH))
        self.assertEqual(tuple(fr["lexicon"]["properties"]["presets"]["items"]["enum"]), FILLER_PRESETS); self.assertEqual(tuple(sorted(c["filler_lexicons"]["presets"])), FILLER_PRESETS)
        for part in ("words", "keep"):
            self.assertEqual((fr["lexicon"]["properties"][part]["max_items"], fr["lexicon"]["properties"][part]["items"]["max_length"]), (FILLER_LEXICON_MAX_ITEMS, FILLER_LEXICON_MAX_LENGTH))
        for k, rng in (("pad", FILLER_PAD_RANGE), ("merge_gap", FILLER_MERGE_GAP_RANGE), ("declick_ms", FILLER_DECLICK_RANGE)):
            self.assertEqual((fr[k]["min"], fr[k]["max"]), rng, k)
        self.assertTrue(fr["lexicon"]["required"]); self.assertNotIn("default", fr["lexicon"], "no default lexicon")
        duck = ops["MIX"]["parameters"]["duck"]
        self.assertEqual(set(duck["properties"]), set(DUCK_FIELDS)); self.assertEqual(tuple(duck["required_properties"]), DUCK_REQUIRED)
        for f, (lo, hi) in DUCK_FIELDS.items():
            self.assertEqual((duck["properties"][f]["min"], duck["properties"][f]["max"]), (lo, hi), f)
        self.assertEqual(OPERATIONS["audio.filler_remove"], {"skill": "audio_filler_remove", "type": "FILLER_REMOVE", "params": ("words", "lexicon", "pad", "merge_gap", "declick_ms"), "risk": "MEDIUM"})
        self.assertEqual(SOURCE_OPS, ("audio.cut", "audio.filler_remove"))
        from video_agent.skills import default_registry
        spec = default_registry().get("audio_filler_remove")
        self.assertEqual(spec.tools, ["audio-production/run"]); self.assertIn("audio-production:FILLER_REMOVE", spec.required_capabilities)

    def test_requirements_and_refusals(self):
        a = parse_audio_requirements(reqs({"audio.production": True, "audio.filler_remove": "ja,EN", "audio.filler_remove.words": "えーっと, まあ", "audio.filler_remove.keep": ["ah"],
                                           "audio.filler_remove.pad": 0.05, "audio.filler_remove.merge_gap": 0.3, "audio.filler_remove.declick_ms": 5}))
        self.assertEqual(a["filler_remove"], {"lexicon": {"presets": ["en", "ja"], "words": ["えーっと", "まあ"], "keep": ["ah"]}, "pad": 0.05, "merge_gap": 0.3, "declick_ms": 5.0})
        self.assertEqual(parse_audio_requirements(reqs({"audio.production": True, "audio.filler_remove": True, "audio.filler_remove.words": ["like"]}))["filler_remove"],
                         {"lexicon": {"words": ["like"]}}, "true = the user's own words only")
        self.assertNotIn("filler_remove", parse_audio_requirements(reqs({"audio.production": True, "audio.filler_remove": False})))
        d = parse_audio_requirements(reqs({"audio.production": True, "audio.mix": True, "audio.mix.duck.input": 1, "audio.mix.duck.amount_db": "12", "audio.mix.duck.release_ms": 300}))
        self.assertEqual(d["mix_duck"], {"input": 1, "amount_db": 12.0, "release_ms": 300.0})
        base = {"audio.production": True}
        for bad, msg in (({"audio.filler_remove": "ja"}, "audio.production=true"),
                         (dict(base, **{"audio.filler_remove.pad": 0.1}), "needs audio.filler_remove"),
                         (dict(base, **{"audio.filler_remove": True}), "no default lexicon"),
                         (dict(base, **{"audio.filler_remove": "fr"}), "lexicon presets"),
                         (dict(base, **{"audio.filler_remove": "ja", "audio.filler_remove.pad": 0.6}), "within"),
                         (dict(base, **{"audio.filler_remove": "ja", "audio.filler_remove.merge_gap": 3}), "within"),
                         (dict(base, **{"audio.filler_remove": "ja", "audio.filler_remove.declick_ms": 500}), "within"),
                         (dict(base, **{"audio.filler_remove": "ja", "audio.filler_remove.words": ["a\u0000b"]}), "printable"),
                         (dict(base, **{"audio.filler_remove": "ja", "audio.filler_remove.words": ["x" * 65]}), "printable"),
                         (dict(base, **{"audio.filler_remove": True, "audio.filler_remove.words": "um", "audio.filler_remove.keep": "um"}), "empty"),
                         (dict(base, **{"audio.filler_remove": False, "audio.filler_remove.words": "um"}), "ambiguous"),
                         (dict(base, **{"audio.mix.duck.input": 1, "audio.mix.duck.amount_db": 12}), "needs audio.mix=true"),
                         (dict(base, **{"audio.mix": True, "audio.mix.duck.input": 1}), "audio.mix.duck.amount_db"),
                         (dict(base, **{"audio.mix": True, "audio.mix.duck.amount_db": 12}), "audio.mix.duck.input"),
                         (dict(base, **{"audio.mix": True, "audio.mix.duck.input": 1, "audio.mix.duck.amount_db": 0}), "lowers nothing"),
                         (dict(base, **{"audio.mix": True, "audio.mix.duck.input": 0, "audio.mix.duck.amount_db": 6}), "within"),
                         (dict(base, **{"audio.mix": True, "audio.mix.duck.input": 1.5, "audio.mix.duck.amount_db": 6}), "input index"),
                         (dict(base, **{"audio.mix": True, "audio.mix.duck.input": 1, "audio.mix.duck.amount_db": 61}), "within"),
                         (dict(base, **{"audio.mix": True, "audio.mix.levels": "0,mute", "audio.mix.duck.input": 1, "audio.mix.duck.amount_db": 6}), "muted")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                parse_audio_requirements(reqs(bad))
            self.assertIn(msg, str(cm.exception), str(bad))
            with self.assertRaises(ValueError, msg=str(bad)):
                _check_requirement_keys(bad)
        for key in ("audio.mix.duck", "audio.filler_remove.lexicon", "audio.filler_remove.presets"):
            with self.assertRaises(ValueError, msg=key) as cm:
                _check_requirement_keys(dict(base, **{key: True}))
            self.assertIn("unknown requirement key", str(cm.exception))
        from video_agent.agent.audio import DUCK_KEYS, FILLER_KEYS
        for k in DUCK_KEYS + FILLER_KEYS:
            self.assertIn(k, BRIEF_REQUIREMENT_KEYS, k)

    def test_word_timings_follow_the_cut(self):
        from video_agent.agent.audio import transcript_words, words_after_cut
        segs = [{"id": "s1", "words": [{"start": 1.0, "end": 1.4, "text": "um", "confidence": 0.8}, {"start": 0.5, "end": 0.9, "text": "a"}, {"start": 2.0, "end": 2.0, "text": "z"},
                                       {"start": 2.5, "end": 2.6, "text": "  "}]},
                {"id": "s2", "words": [{"start": 5.0, "end": 5.5, "text": "えー", "probability": 0.3}, {"start": 7.9, "end": 8.2, "text": "late"}]}]
        words, untimed, problem = transcript_words(segs)
        self.assertEqual((untimed, problem), ([], ""))
        self.assertEqual([(w["start"], w["text"]) for w in words], [(0.5, "a"), (1.0, "um"), (5.0, "えー"), (7.9, "late")], "sorted; zero-length / blank words left out")
        self.assertEqual(set(words[1]), {"start", "end", "text", "confidence"}); self.assertEqual(set(words[2]), {"start", "end", "text"}, "only the contract's fields travel")
        out, dropped = words_after_cut(words, [[0.0, 0.7], [3.0, 4.0]], 8.0)
        self.assertEqual([(w["start"], w["end"], w["text"]) for w in out], [(0.3, 0.7, "um"), (3.3, 3.8, "えー")], "shifted onto the cut audio's timeline")
        self.assertEqual(dropped, 2, "a word straddling a cut range and one past the end are left out")
        self.assertEqual(words_after_cut(words[:2], [], 8.0)[0][1]["start"], 1.0, "no cut: unchanged")
        self.assertEqual(transcript_words([{"id": "s1", "words": None}, {"id": "s2", "words": []}])[1], ["s1"])
        self.assertIn("without start / end", transcript_words([{"id": "s1", "words": [{"start": None, "end": 1.0, "text": "a"}]}])[2])
        self.assertIn("printable", transcript_words([{"id": "s1", "words": [{"start": 0.0, "end": 1.0, "text": "a\x01"}]}])[2])

    def test_pipeline_decision_ir_step_and_skill_request(self):
        v = self._voice()
        svc = pipeline_service(self.tmp, caps=FR_CAPS)
        ir = svc.plan([v], "generic", user_requirements={"audio.production": True, "audio.filler_remove": "ja,en", "audio.filler_remove.pad": 0.05}, params={"language": "ja"})
        d = ir.doc
        self.assertEqual(svc.validate(ir).errors, [])
        tr = next(o for o in d["analysis"]["observations"] if o["kind"] == "transcript")
        self.assertIs(tr["parameters"]["word_timestamps"], True, "the requirement asked the recogniser for word timestamps")
        dec = next(x for x in d["decisions"] if x["subject"] == "audio.filler_remove")
        self.assertEqual((dec["type"], dec["approval"], dec["provenance"], dec["basis"]["approval"]["key"]), ("TRANSFORM", "CONFIRM", "USER", "audio.filler_remove.approval"),
                         "removing recognised speech is CONFIRM even when asked for explicitly")
        self.assertIn(tr["id"], dec["evidence"]); self.assertEqual((dec["params"]["transcript_id"], dec["params"]["word_count"]), (tr["id"], 4))
        self.assertEqual(d["plan"]["status"], "REVIEW"); self.assertEqual([x["subject"] for x in ir.pending_confirmations()], ["audio.filler_remove"])
        self.assertEqual([s_["skill"] for s_ in d["plan"]["steps"]], ["audio_cut", "audio_filler_remove"])
        cut, fr = d["audio"]["operations"]
        self.assertEqual((cut["type"], fr["type"], fr["input"], fr["output"]), ("audio.cut", "audio.filler_remove", f"{fr['asset']}_cut", f"{fr['asset']}_filler_remove"))
        lead = cut["remove"][0][1]   # the leading silence the cut removes (with its margin)
        self.assertEqual([(w["start"], w["text"]) for w in fr["words"]], [(round(s_ - lead, 6), t) for s_, _, t in FR_WORDS], "word times on the cut audio's timeline")
        self.assertEqual((fr["lexicon"], fr["pad"]), ({"presets": ["en", "ja"]}, 0.05))
        step = d["plan"]["steps"][1]
        self.assertEqual((step["tool"], step["params"]), ("audio-production/run", {"asset": fr["asset"], "lexicon": {"presets": ["en", "ja"]}, "pad": 0.05, "word_count": 4}))
        self.assertEqual(lower_audio_op("audio-production/run", fr, "cur", "out"), {"operation": "FILLER_REMOVE", "input": "cur", "words": fr["words"], "lexicon": fr["lexicon"], "pad": 0.05, "output": "out"})
        p = str(Path(svc.workspace) / "plans" / "fr.project.json"); save_ir(ir, p)
        self.assertEqual(svc.render(load_ir(p), p)["status"], "WAITING_FOR_APPROVAL", "nothing is removed before the confirmation")
        out = svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = [x for x in out["execution"]["results"] if x["tool"] == "audio-production/run"]
        self.assertEqual([x["data"]["operation_type"] for x in res], ["CUT", "FILLER_REMOVE"])
        meas = res[1]["data"]["operation"]["measurements"]["filler_remove"]
        self.assertEqual([m_["text"] for m_ in meas["matched"]], ["えー", "um"]); self.assertAlmostEqual(meas["removed_seconds"], 0.9, places=6)
        items = {i["name"]: i for i in out["qa"]["items"]}
        self.assertEqual(items["duration"]["status"], "PASS", items["duration"])
        self.assertAlmostEqual(float(items["duration"]["expected"].split()[0]), fr["temporal_scope"]["end"] - 0.9, places=3, msg="QA expects the cut length less what the Skill removed")
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        row = next(e for e in prov["operations"] if e["skill"] == "audio_filler_remove")
        self.assertEqual((row["args"]["operation"], row["args"]["lexicon"], len(row["args"]["words"])), ("FILLER_REMOVE", {"presets": ["en", "ja"]}, 4)); self.assertEqual(row["decision"], [dec["id"]])

    def test_filler_removal_on_each_concat_input(self):
        """Each input's fillers go right after its own cut, before the join; the programme's expected length is the plan's less both removals."""
        v1, v2 = self._voice("a.wav"), self._voice("b.wav", words=[[4.0, 4.5, "uh"], [5.0, 6.0, "hello"]], segments=[[3.5, 7.0, "uh hello"]])
        svc = pipeline_service(self.tmp, caps=FR_CAPS)
        r = plan_and_render(svc, [v1, v2], {"audio.production": True, "audio.concat": True, "audio.filler_remove": "ja,en"}, profile="generic", name="frc")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        self.assertEqual([s_["skill"] for s_ in d["plan"]["steps"]], ["audio_cut", "audio_filler_remove", "audio_cut", "audio_filler_remove", "audio_concat"])
        cat = next(op for op in d["audio"]["operations"] if op["type"] == "audio.concat")
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = [x for x in out["execution"]["results"] if x["tool"] == "audio-production/run"]
        removed = [x["data"]["operation"]["measurements"]["filler_remove"]["removed_seconds"] for x in res if x["data"]["operation_type"] == "FILLER_REMOVE"]
        self.assertEqual(len(removed), 2); self.assertAlmostEqual(sum(removed), 0.7 + 0.5, places=6)
        items = {i["name"]: i for i in out["qa"]["items"]}
        self.assertEqual(items["duration"]["status"], "PASS", items["duration"])
        self.assertAlmostEqual(float(items["duration"]["observed"]), cat["timeline_duration"] - sum(removed), places=2)

    def test_filler_blocks(self):
        # a segment without word timestamps: never a guessed timing
        v = self._voice(words=FR_WORDS, segments=FR_SEGMENTS + [[12.5, 13.5, "untimed"]])
        svc = pipeline_service(self.tmp, caps=FR_CAPS)
        ir = svc.plan([v], "generic", user_requirements={"audio.production": True, "audio.filler_remove": "ja"}, params={"language": "ja"})
        blk = [x for x in ir.doc["decisions"] if x["approval"] == "BLOCK"]
        self.assertEqual([(x["subject"], x["params"]["reason"]) for x in blk], [("audio.filler_remove", "word_timestamps_missing")]); self.assertIn("seg_0002", blk[0]["reason"])
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED"); self.assertFalse([op for op in ir.doc["audio"]["operations"] if op["type"] == "audio.filler_remove"])
        # no recogniser installed: the measurement the requirement needs is refused before planning (as for subtitles), nothing is guessed
        svc2 = pipeline_service(self.tmp, caps=PipelineCaps(missing=("transcription",), extra=FR_CAPS.extra), transcription=False)
        with self.assertRaises(RuntimeError) as cm:
            svc2.plan([self._voice("c.wav")], "generic", user_requirements={"audio.production": True, "audio.filler_remove": "en"})
        self.assertIn("speech_transcription", str(cm.exception))
        # a recogniser that produced no transcript for the input (failed recognition): BLOCK no_transcript
        os.environ["FAKE_TS_MODE"] = "model_unavailable"
        ir2 = svc.plan([self._voice("c2.wav")], "generic", user_requirements={"audio.production": True, "audio.filler_remove": "en"}, params={"language": "ja"})
        os.environ.pop("FAKE_TS_MODE")
        self.assertIn(("audio.filler_remove", "no_transcript"), [(x["subject"], x["params"].get("reason")) for x in ir2.doc["decisions"] if x["approval"] == "BLOCK"])
        # a mixed input keeps its own timeline
        m = pipeline_media(self.tmp, "music.wav", video=False, duration=20.0)
        ir3 = svc.plan([self._voice("d.wav"), m], "generic", user_requirements={"audio.production": True, "audio.mix": True, "audio.filler_remove": "ja"}, params={"language": "ja"})
        self.assertEqual({x["params"]["reason"] for x in ir3.doc["decisions"] if x["subject"] == "audio.filler_remove"}, {"mixed_input"})
        # without the Skill's FILLER_REMOVE capability nothing else removes words
        svc4 = pipeline_service(self.tmp, caps=PipelineCaps(extra=AUDIO_PRODUCTION_CAPS))
        ir4 = svc4.plan([self._voice("e.wav")], "generic", user_requirements={"audio.production": True, "audio.filler_remove": "ja"}, params={"language": "ja"})
        self.assertIn("capability.audio_filler_remove", {x["subject"] for x in ir4.doc["decisions"] if x["approval"] == "BLOCK"})

    def test_duck_decision_ir_and_skill_request(self):
        v = self._voice()
        m = pipeline_media(self.tmp, "music.wav", video=False, duration=20.0)
        svc = pipeline_service(self.tmp, caps=FR_CAPS)
        req = {"audio.production": True, "audio.mix": True, "audio.mix.levels": "0,-6", "audio.mix.duck.input": 1, "audio.mix.duck.amount_db": 12, "audio.mix.duck.threshold_db": -30}
        r = plan_and_render(svc, [v, m], req, profile="generic", name="duck")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        dec = next(x for x in d["decisions"] if x["subject"] == "audio.mix")
        self.assertEqual(dec["params"]["duck"], {"input": 1, "amount_db": 12.0, "threshold_db": -30.0}); self.assertIn("ducked 12 dB under", dec["decision"])
        mix = next(op for op in d["audio"]["operations"] if op["type"] == "audio.mix")
        self.assertEqual(mix["duck"], {"input": 1, "amount_db": 12.0, "threshold_db": -30.0})
        step = next(s_ for s_ in d["plan"]["steps"] if s_["skill"] == "audio_mix")
        self.assertEqual(step["params"]["duck"], mix["duck"])
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = next(x for x in out["execution"]["results"] if x["tool"] == "audio-production/run" and x["data"]["operation_type"] == "MIX")
        self.assertEqual(res["data"]["operation"]["parameters"]["duck"], mix["duck"])
        # a bed index outside the mixed inputs is a BLOCK (never mapped to another input)
        ir = svc.plan([v, m], "generic", user_requirements=dict(req, **{"audio.mix.levels": "0,-6", "audio.mix.duck.input": 2}))
        blk = [x for x in ir.doc["decisions"] if x["approval"] == "BLOCK"]
        self.assertEqual([x["subject"] for x in blk], ["audio.mix"]); self.assertIn("names no bed", blk[0]["reason"])

    def test_an_engine_without_the_flag_refuses_once(self):
        """An ffmpeg-skill without --duck: the Skill refuses MIX.duck before any tool runs (TOOL_ERROR ffmpeg_skill_flag_missing); one attempt, no mix."""
        v = self._voice()
        m = pipeline_media(self.tmp, "music.wav", video=False, duration=20.0)
        os.environ["FAKE_AP_MODE"] = "flag_missing"
        svc = pipeline_service(self.tmp, caps=FR_CAPS)
        out = plan_and_render(svc, [v, m], {"audio.production": True, "audio.mix": True, "audio.mix.duck.input": 1, "audio.mix.duck.amount_db": 9}, profile="generic", name="noflag")["out"]
        self.assertNotEqual(out["execution"]["status"], "COMPLETED")
        failed = [x for x in out["execution"]["results"] if not x["ok"]]
        self.assertEqual(len(failed), 1, "not retried")
        err = failed[0]["data"]["error"]
        self.assertEqual((err["code"], err["retryable"], err["details"]["reason"], err["details"]["missing_flags"]), ("TOOL_ERROR", False, "ffmpeg_skill_flag_missing", ["duck"]))

    def test_tampered_ir_is_refused(self):
        from video_agent.project.validator import check_audio_operations
        v = self._voice()
        svc = pipeline_service(self.tmp, caps=FR_CAPS)
        ir = svc.plan([v], "generic", user_requirements={"audio.production": True, "audio.filler_remove": "ja", "audio.gain": -3}, params={"language": "ja"})
        self.assertEqual(svc.validate(ir).errors, [])

        def errs(mut):
            bad = json.loads(json.dumps(ir.doc))
            mut(next(op for op in bad["audio"]["operations"] if op["type"] == "audio.filler_remove"), bad)
            return " ".join(check_audio_operations(bad))
        self.assertIn("out of order", errs(lambda op, d: op["words"].reverse()))
        self.assertIn("past the audio it cuts", errs(lambda op, d: op["words"][-1].update(end=99.0)))
        self.assertIn("printable", errs(lambda op, d: op["words"][0].update(text="a\nb")))
        self.assertIn("there is no default", errs(lambda op, d: op.update(lexicon={})))
        self.assertIn("presets", errs(lambda op, d: op.update(lexicon={"presets": ["fr"]})))
        self.assertIn("pad", errs(lambda op, d: op.update(pad=0.9)))

        def after_gain(op, d):   # moved behind the chain: no longer right after the cut
            ops = d["audio"]["operations"]; ops.remove(op); ops.append(op)
        self.assertIn("right after its audio.cut", errs(after_gain))

    def test_tampered_duck_is_refused(self):
        from video_agent.project.validator import check_audio_operations
        m = pipeline_media(self.tmp, "music.wav", video=False, duration=20.0)
        svc = pipeline_service(self.tmp, caps=FR_CAPS)
        ir = svc.plan([self._voice(), m], "generic", user_requirements={"audio.production": True, "audio.mix": True, "audio.mix.duck.input": 1, "audio.mix.duck.amount_db": 6})
        self.assertEqual(svc.validate(ir).errors, [])
        for duck, msg in (({"input": 2, "amount_db": 6.0}, "names no bed"), ({"input": 1, "amount_db": 99.0}, "within"), ({"input": 1}, "must be {input, amount_db"),
                          ({"input": 1, "amount_db": 6.0, "knee_db": 2.0}, "must be {input, amount_db")):
            bad = json.loads(json.dumps(ir.doc))
            next(op for op in bad["audio"]["operations"] if op["type"] == "audio.mix")["duck"] = duck
            self.assertIn(msg, " ".join(check_audio_operations(bad)), duck)

    def test_qa_needs_what_the_skill_removed(self):
        """Without a successful FILLER_REMOVE result that reports removed_seconds, the expected length is unknown: QA FAILs instead of guessing."""
        from video_agent.models import ToolResult
        from video_agent.qa.checks import filler_removed_seconds
        doc = {"audio": {"operations": [{"type": "audio.filler_remove", "asset": "a", "output": "a_filler_remove"}]}}
        paths = {"a_filler_remove": os.path.join(self.tmp, "x.wav")}
        ok = ToolResult("o1", "audio-production/run", True, 0, paths["a_filler_remove"], {"operation": {"measurements": {"filler_remove": {"removed_seconds": 1.25}}}}, [], "", 0.1, 1, False)
        self.assertEqual(filler_removed_seconds(doc, paths, [ok]), {"a": 1.25})
        silent = ToolResult("o1", "audio-production/run", True, 0, paths["a_filler_remove"], {"operation": {"measurements": {}}}, [], "", 0.1, 1, False)
        other = ToolResult("o2", "audio-production/run", True, 0, os.path.join(self.tmp, "y.wav"), ok.data, [], "", 0.1, 1, False)
        failed = ToolResult("o1", "audio-production/run", False, 1, None, ok.data, [], "", 0.1, 1, False)
        for rs in ([silent], [other], [failed], []):
            self.assertEqual(filler_removed_seconds(doc, paths, rs), {"a": None})


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

    def test_an_engine_whose_export_keeps_subtitles_lifts_the_block(self):
        """ADR-071: ffmpeg-skill 2.22.4 (#380) keeps subtitle streams through every export preset but gif; older or unreadable versions still block."""
        from video_agent.capabilities.resolver import export_keeps_subtitles
        from video_agent.capabilities.resolver import Capability

        class Caps(PipelineCaps):
            def __init__(self, version):
                super().__init__()
                self.version = version

            def resolve(self, refresh=False):
                out = super().resolve(refresh)
                out["ffmpeg-skill"] = Capability("ffmpeg-skill", "AVAILABLE", "fake", {"version": self.version})
                return out
        for v, keeps in (("2.22.4", True), ("2.23.0", True), ("3.0.0", True), ("2.22.3", False), ("0.8.4-fake", False), ("", False), ("dev", False)):
            self.assertEqual(export_keeps_subtitles(Caps(v).resolve()), keeps, v)
        svc = pipeline_service(self.tmp, caps=Caps("2.22.4"))
        ir = svc.plan([self._media()], "youtube", user_requirements={"subtitle": True, "subtitle.mode": "mux"}, kinds=["transcript"], params={"language": "ja"})
        self.assertEqual([x["subject"] for x in ir.doc["decisions"] if x["approval"] == "BLOCK"], [])
        self.assertEqual(next(x for x in ir.doc["decisions"] if x["subject"] == "subtitle.mux")["type"], "TRANSFORM")
        svc = pipeline_service(self.tmp, caps=Caps("2.22.3"))
        ir = svc.plan([self._media()], "youtube", user_requirements={"subtitle": True, "subtitle.mode": "mux"}, kinds=["transcript"], params={"language": "ja"})
        self.assertEqual([x["subject"] for x in ir.doc["decisions"] if x["approval"] == "BLOCK"], ["subtitle.mux"])

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
                "motion.meme": True, "motion.meme.top": "when the build", "motion.meme.bottom": "is green",
                "motion.shape": True, "motion.shape.width": 400, "motion.shape.height": 80, "motion.shape.color": "navy", "motion.shape.opacity": 0.6}   # ADR-078


VOICE_CAPS = PipelineCaps(extra=AUDIO_PRODUCTION_CAPS + [f"audio-production:{t}" for t in ("VOICE_CLEANUP", "NOISE_REDUCTION", "NORMALIZE", "GAIN")])


class AudioVoiceCleanupTests(unittest.TestCase):
    """audio.voice_cleanup=light|medium|strong → audio-production VOICE_CLEANUP (ADR-068, audio-production-skill 0.4.3)."""

    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())

    def tearDown(self):
        clear_modes()

    def test_vocabulary_matches_the_pinned_contract(self):
        from video_agent.agent.audio import AUDIO_ORDER, OPERATIONS, VOICE_CLEANUP_LEVELS
        c = audio_contract()
        self.assertEqual(c["version"], "0.4.8")
        vc = {o["type"]: o for o in c["operations"]}["VOICE_CLEANUP"]
        self.assertEqual(set(vc["parameters"]), {"level", "audio_stream"})
        self.assertEqual((tuple(vc["parameters"]["level"]["enum"]), vc["parameters"]["level"]["required"], "default" in vc["parameters"]["level"]), (VOICE_CLEANUP_LEVELS, True, False))
        self.assertEqual(OPERATIONS["audio.voice_cleanup"], {"skill": "audio_voice_cleanup", "type": "VOICE_CLEANUP", "params": ("level",), "risk": "MEDIUM"})
        self.assertEqual(AUDIO_ORDER.index("audio.voice_cleanup") + 1, AUDIO_ORDER.index("audio.noise_reduction"), "voice clean-up runs right before noise reduction")
        self.assertTrue({"filter:highpass", "filter:deesser"} <= set(vc["required_capabilities"]))
        from video_agent.capabilities.resolver import FILTERS
        self.assertEqual((FILTERS["highpass"], FILTERS["deesser"]), (["highpass"], ["deesser"]), "the resolver measures the filters VOICE_CLEANUP requires")
        from video_agent.skills import default_registry
        spec = default_registry().get("audio_voice_cleanup")
        self.assertEqual((spec.tools, spec.inputs["level"]), (["audio-production/run"], "light|medium|strong"))
        self.assertIn("audio-production:VOICE_CLEANUP", spec.required_capabilities)

    def test_requirements_and_refusals(self):
        for lvl in ("light", "medium", "STRONG"):
            self.assertEqual(parse_audio_requirements(reqs({"audio.production": True, "audio.voice_cleanup": lvl}))["voice_cleanup"], {"level": lvl.lower()})
        for bad, msg in (({"audio.voice_cleanup": "medium"}, "audio.production=true"), ({"audio.production": True, "audio.voice_cleanup": "loud"}, "one of"),
                         ({"audio.production": True, "audio.voice_cleanup": True}, "one of")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                parse_audio_requirements(reqs(bad))
            self.assertIn(msg, str(cm.exception), str(bad))
            with self.assertRaises(ValueError, msg=str(bad)):
                _check_requirement_keys(bad)
        self.assertIn("audio.voice_cleanup", BRIEF_REQUIREMENT_KEYS)
        op = {"type": "audio.voice_cleanup", "asset": "a", "level": "strong"}
        self.assertEqual(lower_audio_op("audio-production/run", op, "a", "a_vc"), {"operation": "VOICE_CLEANUP", "input": "a", "level": "strong", "output": "a_vc"})

    def test_pipeline_decision_ir_step_and_skill_request(self):
        v = pipeline_media(self.tmp, "voice.wav", video=False, duration=16.0)
        svc = pipeline_service(self.tmp, caps=VOICE_CAPS)
        r = plan_and_render(svc, [v], {"audio.production": True, "audio.voice_cleanup": "light", "audio.noise_reduction.strength_db": 30, "audio.gain": -2}, profile="generic", name="vc")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        self.assertEqual([op["type"] for op in d["audio"]["operations"] if op["type"] != "audio.cut"], ["audio.voice_cleanup", "audio.noise_reduction", "audio.gain"])
        dec = next(x for x in d["decisions"] if x["subject"] == "audio.voice_cleanup")
        self.assertEqual((dec["type"], dec["approval"], dec["provenance"], dec["basis"]["approval"]["key"], dec["params"]["level"]), ("TRANSFORM", "AUTO", "USER", "audio.voice_cleanup.approval", "light"))
        self.assertIn("engine release 1.13+", dec["reason"])
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "audio_voice_cleanup")
        self.assertEqual((step["tool"], step["params"]), ("audio-production/run", {"asset": ids_first(d), "level": "light"}))
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = [x for x in out["execution"]["results"] if x["tool"] == "audio-production/run" and x["data"]["operation_type"] != "CUT"]
        self.assertEqual([x["data"]["operation_type"] for x in res], ["VOICE_CLEANUP", "NOISE_REDUCTION", "GAIN"])
        self.assertEqual(res[0]["data"]["operation"]["parameters"], {"level": "light"})
        # the validator refuses a level outside the contract's enum in a tampered IR
        from video_agent.project.validator import check_audio_operations
        bad = json.loads(json.dumps(d))
        next(op for op in bad["audio"]["operations"] if op["type"] == "audio.voice_cleanup")["level"] = "loud"
        self.assertTrue(any("audio.voice_cleanup: level 'loud'" in e for e in check_audio_operations(bad)), check_audio_operations(bad))

    def test_an_older_engine_refuses_light_and_strong_without_retrying(self):
        """ffmpeg-skill < 1.13 runs only medium: the Skill refuses strong with TOOL_ERROR voice_level_unsupported (retryable false); the
        agent stops the job there with the levels the engine offers, and never substitutes medium."""
        v = pipeline_media(self.tmp, "voice.wav", video=False, duration=16.0)
        os.environ["FAKE_AP_MODE"] = "voice_engine_old"
        svc = pipeline_service(self.tmp, caps=VOICE_CAPS)
        out = plan_and_render(svc, [v], {"audio.production": True, "audio.voice_cleanup": "strong"}, profile="generic", name="vc_old")["out"]
        self.assertNotEqual(out["execution"]["status"], "COMPLETED")
        failed = [x for x in out["execution"]["results"] if not x["ok"]]
        self.assertEqual(len(failed), 1, "one attempt: not retried")
        err = failed[0]["data"]["error"]
        self.assertEqual((err["code"], err["retryable"], err["details"]["reason"], err["details"]["available_levels"]), ("TOOL_ERROR", False, "voice_level_unsupported", ["medium"]))
        self.assertFalse([x for x in out["execution"]["results"] if x["ok"] and x["tool"] == "audio-production/run" and x["data"].get("operation_type") == "VOICE_CLEANUP"])


def ids_first(d):
    return list(d["assets"])[0]


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
        self.assertEqual(len(ELEMENT_TYPES), 13)   # ADR-078: shape (motion-graphics-skill 0.2.7)
        for e in c["element_types"]:
            base, main = MOTION_BASE[e["type"]]
            for name in e["parameters"]:
                self.assertIn(base if name == main else f"{base}.{name}", MOTION_KEYS[e["type"]], (e["type"], name))
            fade = any(a["kind"] == "fade" and e["type"] in a["applies_to"] for a in c["animations"])
            self.assertEqual(f"{base}.fade" in MOTION_KEYS[e["type"]], fade, e["type"])
        self.assertEqual([MOTION_BASE[t][0] for t in ("title", "lower_third", "text_overlay", "image_overlay")], ["motion.title", "motion.lower_third", "motion.text", "motion.image"],
                         "the ADR-031 keys keep their names")
        got = parse_motion_requirements(reqs(NEW_ELEMENTS))
        self.assertEqual(sorted(got), ["bug", "chapter", "countdown", "hook", "meme", "progress", "shape", "sticker"])
        self.assertEqual(got["shape"]["params"], {"width": 400, "height": 80, "color": "navy", "opacity": 0.6})
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
        for bad, msg in (({"motion.shape": True, "motion.shape.width": 100, "motion.shape.color": "red"}, "motion.shape.height"),   # ADR-078: required by the contract
                         ({"motion.shape": True, "motion.shape.width": 101, "motion.shape.height": 10, "motion.shape.color": "red"}, "multiple"),
                         ({"motion.countdown": True, "motion.countdown.count_from": 61}, "within"), ({"motion.countdown": True, "motion.countdown.count_from": 2.5}, "integer"),
                         ({"motion.bug": "x", "motion.bug.position": "center"}, "one of"), ({"motion.sticker": "x" * 201}, "at most 200"),
                         ({"motion.hook": "x", "motion.hook.text_color": "white@0.5"}, "takes no @alpha"), ({"motion.chapter": "x", "motion.chapter.primary_color": "FFCC00@0.5"}, "takes no @alpha"),
                         ({"motion.hook": "x", "motion.hook.primary_color": "notacolour"}, "not a colour name"), ({"motion.text": "x", "motion.text.box_color": "random@0.5"}, "not a colour name"),
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

    def test_template_colours_follow_the_contract(self):
        """ADR-067 (motion-graphics-skill 0.2.4, its ADR-20): a template element's colour takes a name from the contract's colors.named
        table (any case; the Skill converts it to RRGGBB) as well as bare RRGGBB, and @alpha is refused exactly where the parameter's
        schema says alpha: false -- read from the pin, not from the engine tool's name. Overlay colours still take @alpha."""
        from video_agent.agent.finishing import MOTION_COLOR_NAMES, MOTION_ELEMENT_SPECS
        self.assertEqual(MOTION_COLOR_NAMES, frozenset(motion_contract()["colors"]["named"]), "the names are the pinned contract's table")
        self.assertIn("navy", MOTION_COLOR_NAMES); self.assertNotIn("random", MOTION_COLOR_NAMES)
        opaque = sorted((t, k) for t, e in MOTION_ELEMENT_SPECS.items() for k, ps in e["parameters"].items() if ps["type"] == "color" and ps.get("alpha") is False)
        self.assertEqual(opaque, sorted([("title", "text_color"), ("title", "primary_color"), ("lower_third", "text_color"), ("lower_third", "primary_color"),
                                         ("bug", "text_color"), ("chapter", "primary_color"), ("progress", "primary_color"), ("countdown", "primary_color"),
                                         ("sticker", "primary_color"), ("hook", "text_color"), ("hook", "primary_color"), ("shape", "color")]), "meme has no colour parameter")   # shape: 0.2.7 (ADR-078), not wired
        got = parse_motion_requirements(reqs({"motion.hook": "x", "motion.hook.text_color": "white", "motion.hook.primary_color": "Navy",
                                              "motion.chapter": "x", "motion.chapter.primary_color": "FFCC00", "motion.title": "t", "motion.title.primary_color": "teal",
                                              "motion.text": "x", "motion.text.font_color": "Yellow@0.8", "motion.text.box_color": "000000@0.5"}))
        self.assertEqual(got["hook"]["params"], {"title": "x", "text_color": "white", "primary_color": "Navy"})
        self.assertEqual(got["chapter"]["params"]["primary_color"], "FFCC00"); self.assertEqual(got["title"]["params"]["primary_color"], "teal")
        self.assertEqual((got["text_overlay"]["params"]["font_color"], got["text_overlay"]["params"]["box_color"]), ("Yellow@0.8", "000000@0.5"))
        for key in ("motion.title.text_color", "motion.lower_third.primary_color", "motion.bug.text_color", "motion.progress.primary_color", "motion.countdown.primary_color", "motion.sticker.primary_color"):
            base = key.rsplit(".", 1)[0]
            with self.assertRaises(EditRequirementError, msg=key) as cm:
                parse_motion_requirements(reqs({base: True if base in ("motion.progress", "motion.countdown") else "x", key: "red@1.0"}))
            self.assertIn("takes no @alpha", str(cm.exception))

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
        self.assertEqual(len(step["params"]["elements"]), 13); self.assertEqual(sorted(step["decision_ids"]), sorted(x["id"] for x in decs.values()))
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        row = next(e for e in prov["operations"] if e["tool"] == "motion-graphics/run")
        sent = {e["type"]: e for e in row["args"]["elements"]}
        self.assertEqual(sent["video_overlay"]["parameters"]["video"], "talk_graphics_video".replace("talk", list(d["assets"])[0]))
        self.assertEqual(sent["meme"]["parameters"], {"top": "when the build", "bottom": "is green"})
        self.assertEqual(sent["countdown"]["parameters"], {"count_from": 3, "primary_color": "FFFFFF"})
        self.assertEqual(sent["shape"]["parameters"], {"width": 400, "height": 80, "color": "navy", "opacity": 0.6})
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



class QcDeliveryPackageTests(unittest.TestCase):
    """The QC gate checks each target's deliverable set together (ADR-065): qc-skill kind delivery_package → a plan step, `qa.qc.packages`,
    one qc/check op with the members by artifact id → a `delivery_package` QA item on every member → their stages."""

    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())

    def tearDown(self):
        clear_modes()

    def _render(self, reqs, profile="youtube", name="pkg"):
        svc = pipeline_service(self.tmp)
        media = pipeline_media(self.tmp, "talk.mp4", segments=[[4.0, 8.0, "hello there"], [9.0, 12.0, "and goodbye"]])
        return svc, plan_and_render(svc, [media], reqs, profile=profile, kinds=["transcript"], name=name)

    def test_rules_and_compiled_request(self):
        from video_agent.agent.qc import package_rules
        from video_agent.execution.compiler import lower_qc_package
        members = [{"artifact_id": "a_delivery_youtube", "artifact_type": "video", "input": "a_delivery_youtube", "extension": ".mp4"},
                   {"artifact_id": "a_captions", "artifact_type": "subtitle", "input": "a_captions", "extension": ".srt"},
                   {"artifact_id": "a_thumbnail", "artifact_type": "thumbnail", "input": "a_thumbnail", "extension": ".PNG"}]
        spec = package_rules(members)
        self.assertEqual(spec["kind"], "delivery_package")
        rule = spec["rules"]["delivery_package"]
        self.assertEqual([(a["artifact_id"], a["artifact_type"], a["required"], a["min_size_bytes"], a["expected_extension"]) for a in rule["artifacts"]],
                         [("a_delivery_youtube", "video", True, 1, "mp4"), ("a_captions", "subtitle", True, 1, "srt"), ("a_thumbnail", "thumbnail", True, 1, "png")])
        self.assertEqual(rule["cross_artifact"], {"dependencies": [{"artifact_id": "a_captions", "requires_artifact_id": "a_delivery_youtube"},
                                                                   {"artifact_id": "a_thumbnail", "requires_artifact_id": "a_delivery_youtube"}]})
        self.assertNotIn("duration_consistency", rule["cross_artifact"], "a subtitle's duration is its last cue, not the programme's")
        args = lower_qc_package("qc/check", members, spec["rules"])
        self.assertEqual((args["kind"], args["cache_policy"], [sorted(a) for a in args["artifacts"]]), ("delivery_package", "bypass", [["artifact_id", "artifact_type", "input"]] * 3))
        with self.assertRaises(CompileError):
            lower_qc_package("ffmpeg-skill/check", members, spec["rules"])

    def test_pipeline_step_ir_request_qa_and_stages(self):
        svc, r = self._render({"subtitle": True, "thumbnail": True, "qc": True})
        d = r["ir"].doc
        aid = list(d["assets"])[0]
        self.assertEqual(r["validation"].errors, [])
        step = d["plan"]["steps"][-1]
        self.assertEqual((step["skill"], step["params"]["kind"], step["inputs"]), ("qc_check", "delivery_package", [f"{aid}_delivery_youtube", f"{aid}_captions", f"{aid}_thumbnail"]))
        self.assertEqual(sorted(step["depends_on"]), sorted([f"step_qc_youtube_{aid}", f"step_qc_captions_{aid}", f"step_thumbnail_{aid}"]))
        self.assertEqual(d["qa"]["qc"]["packages"], {aid: {"targets": ["youtube"], "captions": f"{aid}_captions", "thumbnail": f"{aid}_thumbnail"}})
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        res = [x for x in out["execution"]["results"] if x["tool"] == "qc/check" and x["data"].get("kind") == "delivery_package"]
        self.assertEqual(len(res), 1); self.assertTrue(res[0]["data"]["admitted"]); self.assertEqual(res[0]["data"]["verdict"], "PASS")
        arts = {a["type"]: a for a in out["artifacts"]}
        self.assertEqual(res[0]["data"]["artifacts"], {a["logical_name"]: a["hash"] for a in arts.values()}, "the report is about the delivered bytes of every member")
        items = [i for i in out["qa"]["items"] if i["name"] == "delivery_package"]
        self.assertEqual(sorted((i["artifact"], i["status"]) for i in items), sorted((a["logical_name"], "PASS") for a in arts.values()))
        self.assertEqual({t: (a["qa"]["qc"], a["stage"]) for t, a in arts.items()}, {"YOUTUBE": ("PASS", "approved"), "CAPTIONS": ("PASS", "approved"), "THUMBNAIL": ("PASS", "approved")})
        prov = json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        row = next(e for e in prov["operations"] if e["tool"] == "qc/check" and e["args"].get("kind") == "delivery_package")
        self.assertEqual([a["artifact_type"] for a in row["args"]["artifacts"]], ["video", "subtitle", "thumbnail"]); self.assertTrue(row["decision"])

    def test_a_failing_set_keeps_every_member_working(self):
        os.environ["FAKE_QC_MODE"] = "package_fail"   # the thumbnail is not of the planned extension: the set is not a delivery
        _svc, r = self._render({"subtitle": True, "thumbnail": True, "qc": True}, name="pkgfail")
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED"); self.assertEqual(out["qa"]["status"], "FAIL")
        arts = {a["type"]: a for a in out["artifacts"]}
        self.assertEqual({t: (a["qa"]["qc"], a["stage"]) for t, a in arts.items()}, {"YOUTUBE": ("FAIL", "working"), "CAPTIONS": ("FAIL", "working"), "THUMBNAIL": ("FAIL", "working")},
                         "every member of a failing set stays working, even those whose own reports pass")
        th = arts["THUMBNAIL"]["logical_name"]
        self.assertIn(("delivery_package.artifact_extension_matches_expected", th, "FAIL"), [(i["name"], i["artifact"], i["status"]) for i in out["qa"]["items"]])
        self.assertIn("DELIVERY_SPEC_FAILURE", [x["type"] for x in out["qa"]["incidents"]])

    def test_a_report_about_other_files_is_never_admitted(self):
        os.environ["FAKE_QC_MODE"] = "package_fingerprint"
        _svc, r = self._render({"subtitle": True, "thumbnail": True, "qc": True}, name="pkgfp")
        res = next(x for x in r["out"]["execution"]["results"] if x["tool"] == "qc/check" and not x["ok"])
        self.assertEqual(res["data"]["error"]["code"], "INVALID_RESULT"); self.assertIn("sha256", res["data"]["error"]["message"])
        self.assertEqual(r["out"]["execution"]["status"], "FAILED")

    def test_no_package_without_company_or_without_the_gate(self):
        _svc, r = self._render({"qc": True}, name="alone")
        self.assertNotIn("delivery_package", [s_["params"].get("kind") for s_ in r["ir"].doc["plan"]["steps"]], "a lone deliverable is covered by its own report")
        self.assertNotIn("packages", r["ir"].doc["qa"]["qc"])
        _svc, r = self._render({"subtitle": True, "thumbnail": True}, name="nogate")
        self.assertNotIn("qc_check", [s_["skill"] for s_ in r["ir"].doc["plan"]["steps"]], "the package check is part of the opt-in QC gate (ADR-032)")

    def test_tampered_packages_are_refused(self):
        svc = pipeline_service(self.tmp)
        media = pipeline_media(self.tmp, "talk.mp4", segments=[[4.0, 8.0, "hello there"]])
        ir = svc.plan([media], "youtube", user_requirements={"thumbnail": True, "qc": True}, params={"language": "ja"})
        aid = list(ir.doc["assets"])[0]
        ir.doc["qa"]["qc"]["packages"][aid]["thumbnail"] = f"{aid}_delivery_youtube"
        self.assertTrue(any("not a planned THUMBNAIL output" in e for e in svc.validate(ir).errors))
        ir.doc["qa"]["qc"]["packages"][aid] = {"targets": ["youtube"], "captions": None, "thumbnail": None}
        self.assertTrue(any("nothing beside the deliverable" in e for e in svc.validate(ir).errors))

    def test_adapter_refusals(self):
        from video_agent.tools.qc import QcAdapter
        from video_agent.tools.skill_process import CliSkill
        from video_agent.tools.base import ToolError
        ws = str(Path(self.tmp) / "ws"); os.makedirs(ws, exist_ok=True)
        f = Path(self.tmp) / "src" / "x.mp4"; f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(b"x")
        ad = QcAdapter(CliSkill("qc", [sys.executable, str(Path(__file__).resolve().parent / "fake_qc.py")], None, {}), workspace=ws, allowed_inputs=[str(f.parent)])
        paths = {"v": str(f)}
        good = {"kind": "delivery_package", "artifacts": [{"artifact_id": "v", "artifact_type": "video", "input": "v"}]}
        self.assertEqual(ad.build_request("qc/check", good, paths)["request"]["artifacts"], [{"artifact_id": "v", "artifact_type": "video", "path": str(f)}])
        for bad, msg in ((dict(good, kind="delivery"), "only accepted with kind"), (dict(good, input="v"), "not input"), (dict(good, artifacts=[]), "non-empty"),
                         (dict(good, artifacts=[{"artifact_id": "v", "artifact_type": "poster", "input": "v"}]), "artifact_type"),
                         (dict(good, artifacts=[{"artifact_id": "v", "artifact_type": "video", "input": "v", "path": "/etc/passwd"}]), "forbidden"),
                         (dict(good, rules={"delivery_package": {"artifacts": [{"artifact_id": "w"}]}}), "does not carry")):
            with self.assertRaises(ToolError, msg=str(bad)) as cm:
                ad.build_request("qc/check", bad, paths)
            self.assertIn(msg, str(cm.exception), str(bad))


if __name__ == "__main__":
    unittest.main()
