"""ADR-131: a Project IR colour operation states how its source's colour is stored beside an alpha plane (`source_alpha_mode`: straight |
premultiplied). The compiler passes it to the typed color-grading/run Operation (sent as the request's project.source.alpha_mode), keeps it for
the following colour steps on the same picture, and writes those steps in the input's own container, which must hold an alpha line (the
contract's alpha.keep_alpha.containers that are output formats: mov, mkv). Without the field everything compiles as before (mp4)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import clear_modes, fake_media, pipeline_service  # noqa: E402
from video_agent.agent.finishing import COLOR_ALPHA_CONTAINERS, COLOR_ALPHA_MODES  # noqa: E402
from video_agent.execution.compiler import CompileError, compile_ir, lower_color_op  # noqa: E402
from video_agent.tools.color_grading.adapter import COMMON_ARGS, pinned_contract  # noqa: E402

REQS = {"color.gamma": 1.2, "color.target": "bt709"}


class ColorAlphaIRTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.svc = pipeline_service(self.tmp)

    def tearDown(self):
        clear_modes()

    def _ir(self, name: str, mode=None, untrimmed: bool = False):
        ir = self.svc.plan([fake_media(self.tmp, name)], "youtube", user_requirements=dict(REQS))
        d = ir.doc
        asset = next(iter(d["assets"]))
        if untrimmed:   # the colour steps straight on the source (no edit writing an mp4 in front of them)
            d["video"]["operations"] = [op for op in d["video"]["operations"] if op["asset"] != asset]
            d["color"]["operations"][0]["input"] = asset
        if mode:
            d["color"]["operations"][0]["source_alpha_mode"] = mode
        return ir, asset

    def _color_ops(self, ir):
        ops, paths = compile_ir(ir, str(Path(self.tmp) / "job"))
        return [o for o in ops if o.tool == "color-grading/run"], paths

    def test_vocabulary_comes_from_the_pinned_contract(self):
        c = pinned_contract()
        self.assertEqual(COLOR_ALPHA_MODES, tuple(c["alpha"]["alpha_mode"]["values"]))
        self.assertEqual(COLOR_ALPHA_MODES, ("straight", "premultiplied"))
        self.assertEqual(COLOR_ALPHA_CONTAINERS, ("mkv", "mov"))
        self.assertTrue(set(COLOR_ALPHA_CONTAINERS) <= set(c["output_formats"]))
        self.assertIn("source_alpha_mode", COMMON_ARGS)
        import json
        schema = json.loads((Path(__file__).resolve().parents[1] / "schemas" / "project.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["$defs"]["color_op"]["properties"]["source_alpha_mode"], {"enum": list(COLOR_ALPHA_MODES)})
        self.assertNotIn("source_alpha_mode", schema["$defs"]["color_op"]["required"])

    def test_lowering_carries_the_mode_and_the_container(self):
        op = {"type": "color.lut", "asset": "a", "lut_strength": 0.5, "source_alpha_mode": "premultiplied"}
        self.assertEqual(lower_color_op("color-grading/run", op, "a", "a_lut", "a_lut_lut", "mov"),
                         {"operation": "LUT_APPLY", "input": "a", "output": "a_lut", "format": "mov", "lut_strength": 0.5, "lut": "a_lut_lut",
                          "source_alpha_mode": "premultiplied"})
        # carried from an earlier step on the same picture when the operation does not say it
        self.assertEqual(lower_color_op("color-grading/run", {"type": "color.retag", "asset": "a", "target": "bt709"}, "a_lut", "a_retag", None, "mov",
                                        "premultiplied")["source_alpha_mode"], "premultiplied")
        # without a mode: unchanged (mp4, no source_alpha_mode)
        self.assertEqual(lower_color_op("color-grading/run", {"type": "color.retag", "asset": "a", "target": "bt709"}, "a", "a_retag"),
                         {"operation": "RETAG", "input": "a", "output": "a_retag", "format": "mp4", "target": "bt709"})

    def test_an_alpha_source_is_graded_in_its_own_container_and_the_mode_stays_with_the_picture(self):
        for ext in COLOR_ALPHA_CONTAINERS:
            for mode in COLOR_ALPHA_MODES:
                ir, asset = self._ir(f"key_{mode}.{ext}", mode, untrimmed=True)
                self.assertEqual(self.svc.validate(ir).errors, [], (ext, mode))
                ops, paths = self._color_ops(ir)
                self.assertEqual([o.args["operation"] for o in ops], ["PRIMARY_CORRECTION", "RETAG"])
                for o in ops:
                    self.assertEqual((o.args["format"], o.args["source_alpha_mode"]), (ext, mode), o.args)
                    self.assertEqual(Path(paths[o.outputs[0]]).suffix, f".{ext}")
                self.assertEqual(ops[0].args["input"], asset)

    def test_without_the_field_nothing_changes(self):
        ir, _ = self._ir("talk.mp4")
        ops, paths = self._color_ops(ir)
        self.assertTrue(ops)
        for o in ops:
            self.assertEqual(o.args["format"], "mp4")
            self.assertNotIn("source_alpha_mode", o.args)
            self.assertEqual(Path(paths[o.outputs[0]]).suffix, ".mp4")

    def test_a_picture_without_an_alpha_container_is_refused_at_compile_time(self):
        # the planned trim writes mp4 in front of the colour step: no alpha is left to keep, refused rather than flattened unseen
        ir, _ = self._ir("key.mov", "premultiplied")
        with self.assertRaises(CompileError) as cm:
            self._color_ops(ir)
        self.assertIn("source_alpha_mode premultiplied", str(cm.exception))
        self.assertIn(".mp4", str(cm.exception))
        # an mp4 source the same way
        ir, _ = self._ir("key.mp4", "straight", untrimmed=True)
        with self.assertRaises(CompileError):
            self._color_ops(ir)

    def test_the_schema_refuses_another_value(self):
        ir, _ = self._ir("key.mov", "auto", untrimmed=True)
        self.assertIn("source_alpha_mode", " ".join(self.svc.validate(ir).errors))


if __name__ == "__main__":
    unittest.main()
