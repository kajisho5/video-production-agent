"""ADR-102: motion-graphics-skill 0.2.11 .. 0.2.16 wired end to end on the fake engine and the fake motion-graphics-skill: the `logo` element (the
active theme's logo) and `rendered_layer` (a pre-rendered alpha clip), per-element `style` / `fit`, and the request options `style`,
`brand_kit`, `themes` / `theme` and `reserved_regions`. tests/test_integration.py runs them on the real Skill."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import PipelineCaps, clear_modes, fake_media, pipeline_service, plan_and_render  # noqa: E402
from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.finishing import FIT_TYPES, MOTION_NOT_PLANNED, STYLE_PRESETS, parse_motion_options, parse_motion_requirements  # noqa: E402
from video_agent.agent.requirements import requirement_map  # noqa: E402
from video_agent.models import Requirement  # noqa: E402
from video_agent.tools.motion_graphics.adapter import pinned_contract  # noqa: E402


def reqs(d):
    return requirement_map([Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in d.items()])


class VocabularyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.theme = Path(self.tmp) / "brand.json"
        self.theme.write_text(json.dumps({"colors": {"primary": "0B5FFF", "secondary": "00A3E0", "background": "0F1C2E", "text": "FFFFFF"}}), encoding="utf-8")

    def test_contract_vocabulary(self):
        c = pinned_contract()
        self.assertEqual(MOTION_NOT_PLANNED, ())
        self.assertEqual(STYLE_PRESETS, tuple(c["style_presets"]["names"]))
        self.assertEqual(FIT_TYPES, tuple(c["text_fit"]["applies_to"]))

    def test_elements_with_style_fit_logo_and_layer(self):
        got = parse_motion_requirements(reqs({"motion.title": "Opening", "motion.title.style": "news", "motion.title.fit": {"width": 600, "max_lines": 2, "profile": "ja-kinsoku", "lang": "ja"},
                                              "motion.lower_third": "Ada", "motion.lower_third.fit": 500, "motion.logo": True, "motion.logo.opacity": 0.7,
                                              "motion.layer": self._clip(), "motion.layer.loop": False}))
        self.assertEqual((got["title"]["style"], got["title"]["fit"]), ("news", {"width": 600, "max_lines": 2, "profile": "ja-kinsoku", "lang": "ja"}))
        self.assertEqual(got["lower_third"]["fit"], {"width": 500})
        self.assertEqual(got["logo"]["params"], {"opacity": 0.7})
        self.assertEqual(set(got["rendered_layer"]["params"]), {"layer", "loop"})
        for bad, msg in (({"motion.title": "x", "motion.title.style": "retro"}, "one of"),
                         ({"motion.title": "x", "motion.title.fit": {"width": 8}}, "within"), ({"motion.title": "x", "motion.title.fit": {"width": 600, "profile": "kr"}}, "one of"),
                         ({"motion.layer.loop": False}, "ambiguous")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                parse_motion_requirements(reqs(bad))
            self.assertIn(msg, str(cm.exception))
        from video_agent.service import _check_requirement_keys
        with self.assertRaises(ValueError) as cm:   # text fit applies to title / lower_third only (the contract's text_fit.applies_to): no such key
            _check_requirement_keys({"motion.text": "x", "motion.text.fit": 600})
        self.assertIn("motion.text.fit", str(cm.exception))

    def _clip(self):
        p = Path(self.tmp) / "layer.mov"
        p.write_bytes(b"\x00" * 16)
        return str(p)

    def test_options(self):
        o = parse_motion_options(reqs({"motion.style": "swiss", "motion.brand_kit": {"primary": "E30613", "text": "black"},
                                       "motion.reserved_regions": {"end_screen": True, "cards": [{"time": 30, "region": {"x": 10, "y": 10, "width": 300, "height": 80}}], "on_overlap": "refuse"}}))
        self.assertEqual(o["options"], {"style": "swiss", "brand_kit": {"primary": "E30613", "text": "black"},
                                        "reserved_regions": {"end_screen": {}, "cards": [{"time": 30.0, "region": {"x": 10, "y": 10, "width": 300, "height": 80}}], "on_overlap": "refuse"}})
        t = parse_motion_options(reqs({"motion.themes": {"brand": str(self.theme)}}))["options"]
        self.assertEqual(t["themes"], {"brand": str(self.theme.resolve())})
        for bad, msg in (({"motion.style": "retro"}, "one of"), ({"motion.brand_kit": {"accent": "FFFFFF"}}, "primary?"), ({"motion.brand_kit": {"primary": "FFFFFF@0.5"}}, "opaque"),
                         ({"motion.themes": {"brand": str(self.theme)}, "motion.style": "news"}, "replaces"), ({"motion.theme": "brand"}, "ambiguous"),
                         ({"motion.themes": {"a": str(self.theme), "b": str(self.theme)}}, "which one"), ({"motion.themes": {"a": "/nope.json"}}, "not found"),
                         ({"motion.reserved_regions": {"end_screen": {"duration": 30}}}, "within"), ({"motion.reserved_regions": {"on_overlap": "move"}}, "report or refuse")):
            with self.assertRaises(EditRequirementError, msg=str(bad)) as cm:
                parse_motion_options(reqs(bad))
            self.assertIn(msg, str(cm.exception))


class PlanTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = tempfile.mkdtemp()
        self.a = fake_media(self.tmp, "a.mp4")
        self.layer = fake_media(self.tmp, "lower_third.mov", duration=4.0)
        src = Path(self.tmp) / "src"
        (src / "logo.png").write_bytes(b"\x89PNG fake")
        self.theme = src / "brand.json"
        self.theme.write_text(json.dumps({"schema": "motion-graphics/theme@1", "colors": {"primary": "0B5FFF", "secondary": "00A3E0", "background": "0F1C2E", "text": "FFFFFF"},
                                          "logo": {"path": "logo.png", "position": "top-right"}}), encoding="utf-8")

    def tearDown(self):
        clear_modes()

    def _prov(self, svc, out):
        return json.loads((Path(svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))

    def test_theme_logo_layer_fit_end_to_end(self):
        svc = pipeline_service(self.tmp, caps=PipelineCaps())
        r = plan_and_render(svc, [self.a], {"motion.logo": True, "motion.logo.start": 0, "motion.logo.end": 8, "motion.themes": {"brand": str(self.theme)},
                                            "motion.layer": self.layer, "motion.layer.position": "bottom-left", "motion.layer.in_duration": 0.5, "motion.layer.start": 1, "motion.layer.end": 6,
                                            "motion.title": "Opening", "motion.title.fit": {"width": 600},
                                            "motion.reserved_regions": {"end_screen": {"duration": 5}}}, name="m1")
        d = r["ir"].doc
        self.assertEqual((d["plan"]["status"], r["validation"].errors), ("APPROVED", []), d["plan"]["summary"])
        op = d["graphics"]["operations"][0]
        self.assertEqual(op["options"], {"themes": {"brand": str(self.theme.resolve())}, "reserved_regions": {"end_screen": {"duration": 5.0}}})
        self.assertEqual(op["layer"], self.layer)
        els = {e["type"]: e for e in op["elements"]}
        self.assertEqual(els["title"]["fit"], {"width": 600})
        self.assertIn(str(self.theme.resolve().parent), d["execution"]["allowed_inputs"])
        out = r["out"]
        self.assertEqual(out["execution"]["status"], "COMPLETED", out["execution"])
        row = next(e for e in self._prov(svc, out)["operations"] if e["tool"] == "motion-graphics/run")
        sent = {e["type"]: e for e in row["args"]["elements"]}
        self.assertEqual(sent["rendered_layer"]["parameters"]["layer"], f"{list(d['assets'])[0]}_graphics_layer")
        self.assertEqual(row["args"]["options"]["themes"], {"brand": f"{list(d['assets'])[0]}_graphics_theme_brand"})
        self.assertEqual(sent["title"]["fit"], {"width": 600})

    def test_logo_without_a_theme_blocks_and_options_without_elements_are_refused(self):
        svc = pipeline_service(self.tmp, caps=PipelineCaps())
        ir = svc.plan([self.a], "youtube", user_requirements={"motion.logo": True}, params={"language": "ja"})
        blk = next(x for x in ir.doc["decisions"] if x["subject"] == "graphics.logo")
        self.assertEqual(blk["approval"], "BLOCK"); self.assertIn("theme", blk["reason"])
        with self.assertRaises(ValueError):
            svc.plan([self.a], "youtube", user_requirements={"motion.style": "news"}, params={"language": "ja"})

    def test_style_and_brand_kit(self):
        svc = pipeline_service(self.tmp, caps=PipelineCaps())
        r = plan_and_render(svc, [self.a], {"motion.text": "LIVE", "motion.lower_third": "Ada", "motion.lower_third.style": "minimal", "motion.style": "news",
                                            "motion.brand_kit": {"primary": "C8102E"}}, name="m2")
        d = r["ir"].doc
        self.assertEqual((d["plan"]["status"], r["validation"].errors), ("APPROVED", []), d["plan"]["summary"])
        dec = next(x for x in d["decisions"] if x["subject"] == "graphics.options")
        self.assertEqual(dec["params"]["options"], {"style": "news", "brand_kit": {"primary": "C8102E"}})
        self.assertEqual(r["out"]["execution"]["status"], "COMPLETED", r["out"]["execution"])
        row = next(e for e in self._prov(svc, r["out"])["operations"] if e["tool"] == "motion-graphics/run")
        self.assertEqual(row["args"]["options"], {"style": "news", "brand_kit": {"primary": "C8102E"}})
        self.assertEqual({e["type"]: e.get("style") for e in row["args"]["elements"]}, {"text_overlay": None, "lower_third": "minimal"})


if __name__ == "__main__":
    unittest.main()
