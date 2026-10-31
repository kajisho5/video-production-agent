"""ADR-085 (thumbnail-skill roadmap step 4, agent side: the full ThumbnailDocument): a logo image and several text elements on the thumbnail,
each placed by the plan (the logo at its own aspect ratio read from the file's header), lowered into one thumbnail/render document. Against
the fake thumbnail process of the pipeline harness (which checks the document's structure as the real Skill does). No Pillow, no ffmpeg."""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import clear_modes, fake_media, pipeline_service, plan_and_render  # noqa: E402

from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.finishing import image_size, parse_thumbnail_requirements, thumbnail_layout  # noqa: E402
from video_agent.execution.compiler import lower_thumbnail  # noqa: E402
from video_agent.models import Operation, Requirement  # noqa: E402
from video_agent.project.validator import check_finishing_operations  # noqa: E402
from video_agent.tools.thumbnail import ThumbnailAdapter  # noqa: E402
from video_agent.tools.skill_process import CliSkill  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_thumbnail.py")


def png(w: int, h: int) -> bytes:
    raw = b"".join(b"\x00" + b"\x00\x00\x00\xff" * w for _ in range(h))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def jpeg_header(w: int, h: int) -> bytes:
    """SOI, an APP0 segment, then SOF0 with the size: enough for the header reader (not a decodable picture)."""
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, h, w, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"


def reqs(d):
    return {k: Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in d.items()}


class _Base(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        src = Path(self.tmp) / "src"
        src.mkdir(parents=True, exist_ok=True)
        self.logo = str(src / "logo.png")
        Path(self.logo).write_bytes(png(200, 100))
        self.log = str(Path(self.tmp) / "thumb_calls.log")
        os.environ["FAKE_THUMBNAIL_CALLS"] = self.log

    def tearDown(self):
        clear_modes()


class VocabularyTests(_Base):
    def test_requirements_and_refusals(self):
        th = parse_thumbnail_requirements(reqs({"thumbnail": True, "thumbnail.text": "A", "thumbnail.color": "#FFCC00", "thumbnail.text2": "B", "thumbnail.text2.position": "top",
                                                "thumbnail.text3": "C", "thumbnail.text3.font_size": 30, "thumbnail.logo": self.logo, "thumbnail.logo.width": 160}))
        self.assertEqual(th["color"], "#FFCC00")
        self.assertEqual(th["texts"], [{"text": "B", "position": "top"}, {"text": "C", "font_size": 30}])
        self.assertEqual((th["logo"]["natural"], th["logo"]["width"]), ([200, 100], 160), "the logo's size is read from its header")
        bad_jpg = str(Path(self.tmp) / "src" / "x.jpg")
        Path(bad_jpg).write_bytes(b"not a jpeg")
        for given, msg in (({"thumbnail.text2.color": "#FFFFFF"}, "thumbnail.text2 is not"), ({"thumbnail.logo.width": 100}, "thumbnail.logo is not"),
                           ({"thumbnail.color": "#FFFFFF"}, "needs thumbnail.text"), ({"thumbnail.text2": "B", "thumbnail.text2.color": "red"}, "#RRGGBB"),
                           ({"thumbnail.logo": bad_jpg}, "not a readable PNG / JPEG"), ({"thumbnail.logo": self.logo + ".gif"}, "PNG / JPEG file"),
                           ({"thumbnail.logo": self.logo, "thumbnail.logo.position": "middle"}, "top-left"), ({"thumbnail.text5": "E"}, None)):
            r = dict({"thumbnail": True}, **given)
            if msg is None:   # an unknown key is refused by the service's key gate, not here
                continue
            with self.assertRaises(EditRequirementError, msg=msg) as cm:
                parse_thumbnail_requirements(reqs(r))
            self.assertIn(msg, str(cm.exception))

    def test_header_reader_and_layout(self):
        self.assertEqual(image_size(self.logo), (200, 100))
        j = str(Path(self.tmp) / "src" / "l.jpg")
        Path(j).write_bytes(jpeg_header(640, 480))
        self.assertEqual(image_size(j), (640, 480))
        box, texts = thumbnail_layout(1280, 720, {"image": self.logo, "natural": [200, 100]}, [{"text": "B"}], 40, "bottom")
        self.assertEqual((box["x"], box["y"], box["width"], box["height"]), (1280 - 213 - 24, 24, 213, 106), "default: top-right, a sixth of the width, 24 px margin")
        self.assertEqual(texts, [{"text": "B", "font_size": 40, "color": "#FFFFFF", "position": "bottom"}])
        box, _ = thumbnail_layout(1280, 720, {"image": self.logo, "natural": [200, 100], "position": "bottom-left", "width": 100, "margin": 10, "opacity": 0.5}, [], 40, "bottom")
        self.assertEqual((box["x"], box["y"], box["width"], box["height"], box["opacity"]), (10, 720 - 50 - 10, 100, 50, 0.5))

    def test_lowering_and_adapter_document(self):
        op = {"type": "graphics.thumbnail", "asset": "a", "input": "a_pic", "output": "a_thumbnail", "timestamp": 1.0, "format": "png", "width": 1280, "height": 720,
              "texts": [{"text": "B", "font_size": 40, "color": "#FFFFFF", "position": "top"}], "logo": {"image": self.logo, "x": 10, "y": 10, "width": 100, "height": 50}}
        args = lower_thumbnail("thumbnail/render", op, "a_pic", "a_thumbnail")
        self.assertEqual((args["texts"], args["logo"]), (op["texts"], op["logo"]), "a logo / texts without a caption still renders the document")
        ws = str(Path(self.tmp) / "ws")
        os.makedirs(ws, exist_ok=True)
        ad = ThumbnailAdapter(CliSkill("thumbnail", [sys.executable, FAKE], None, {}), workspace=ws, allowed_inputs=[str(Path(self.tmp) / "src")])
        src = fake_media(self.tmp, "pic.mp4")
        b = ad.build_request("thumbnail/render", args, {"a_pic": src, "a_thumbnail": str(Path(ws) / "t.png")})
        doc = b["request"]["params"]["document"]
        self.assertEqual([a["kind"] for a in doc["assets"]], ["video_frame", "image"])
        self.assertEqual([e["element_id"] for e in doc["elements"]], ["frame", "text2", "logo"])
        lg = next(e for e in doc["elements"] if e["element_id"] == "logo")["image"]
        self.assertEqual((lg["position"], lg["size"], lg["fit"]), ({"x": 10, "y": 10}, {"width": 100, "height": 50}, "contain"))
        outside = str(Path(tempfile.mkdtemp()) / "logo.png")
        Path(outside).write_bytes(png(10, 10))
        with self.assertRaises(Exception) as cm:
            ad.build_request("thumbnail/render", dict(args, logo=dict(args["logo"], image=outside)), {"a_pic": src, "a_thumbnail": str(Path(ws) / "t.png")})
        self.assertIn("outside the allowed roots", str(cm.exception))
        with self.assertRaises(Exception):
            ad.build_request("thumbnail/render", dict(args, logo=dict(args["logo"], path=self.logo)), {"a_pic": src, "a_thumbnail": str(Path(ws) / "t.png")})


class PipelineTests(_Base):
    def test_full_document_planned_and_rendered(self):
        svc = pipeline_service(self.tmp)
        r = plan_and_render(svc, [fake_media(self.tmp, "talk.mp4")], {"thumbnail": True, "thumbnail.text": "Main", "thumbnail.color": "#FFCC00", "thumbnail.text2": "Sub",
                                                                      "thumbnail.text2.position": "top", "thumbnail.logo": self.logo, "thumbnail.logo.position": "bottom-left"},
                            profile="youtube")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        [op] = d["graphics"]["operations"]
        self.assertEqual((op["text"], op["color"], op["texts"][0]["text"], op["logo"]["x"]), ("Main", "#FFCC00", "Sub", 24))
        self.assertIn(str(Path(self.logo).parent), d["execution"]["allowed_inputs"])
        dec = next(x for x in d["decisions"] if x["subject"] == "thumbnail.render")
        self.assertIn("logo logo.png", dec["decision"])
        self.assertEqual(r["out"]["execution"]["status"], "COMPLETED", r["out"]["execution"])
        res = next(x for x in r["out"]["execution"]["results"] if x["tool"].startswith("thumbnail/"))
        self.assertEqual((res["tool"], res["ok"]), ("thumbnail/render", True))

    def test_logo_only_renders_and_a_tampered_logo_is_refused(self):
        svc = pipeline_service(self.tmp)
        ir = svc.plan([fake_media(self.tmp, "talk.mp4")], "youtube", user_requirements={"thumbnail": True, "thumbnail.logo": self.logo})
        step = next(s for s in ir.doc["plan"]["steps"] if s["skill"].startswith("thumbnail_"))
        self.assertEqual(step["skill"], "thumbnail_render", "a logo needs the document render, not a plain frame")
        self.assertEqual(check_finishing_operations(ir.doc), [])
        bad = json.loads(json.dumps(ir.doc))
        bad["graphics"]["operations"][0]["logo"]["image"] = "/etc/logo.png"
        self.assertTrue(any("outside the allowed input roots" in e for e in check_finishing_operations(bad)))
        bad = json.loads(json.dumps(ir.doc))
        bad["graphics"]["operations"][0]["logo"]["x"] = 99999
        self.assertTrue(any("outside the canvas" in e for e in check_finishing_operations(bad)))


if __name__ == "__main__":
    unittest.main()
