"""ADR-104: thumbnail-skill 0.1.12 .. 0.1.15 wired into the plan -- choose_frame (the frame from a ranking of detection-skill's faces with
attributes), place_text (the caption clear of the ranked frame's faces), render_variants (A/B variants), render_layers (layered PNGs + manifest)
and render_batch (a table x platform presets). Against the fake thumbnail / detection processes (the real Skills' request shapes); no Pillow,
no ffmpeg, no detector."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import clear_modes, fake_media, pipeline_service, plan_and_render  # noqa: E402
from test_face_redaction import DetCaps, det_adapter, det_capabilities  # noqa: E402

from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.finishing import THUMBNAIL_KEYS, parse_thumbnail_requirements, thumbnail_skill_of  # noqa: E402
from video_agent.agent.thumbnail_features import (FEATURE_KEYS, PRESETS, choose_ranked_frame, faces_on_picture, inside_keep,  # noqa: E402
                                                  read_batch_table)
from video_agent.execution.compiler import lower_thumbnail  # noqa: E402
from video_agent.models import Observation, Operation, Requirement  # noqa: E402
from video_agent.project.validator import check_finishing_operations  # noqa: E402
from video_agent.tools.base import ToolError  # noqa: E402
from video_agent.tools.skill_process import CliSkill  # noqa: E402
from video_agent.tools.thumbnail import NOT_USED, TOOLS, ThumbnailAdapter, check_contract, pinned_contract  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_thumbnail.py")
# two faces on screen at different times: the first with closed eyes and a large box, the second open-eyed and frontal
FACES = [{"start": 2.0, "end": 4.0, "box": [500.0, 200.0, 300.0, 300.0], "attributes": {"eyes_open": 0.2, "size": 0.5}},
         {"start": 8.0, "end": 10.0, "box": [540.0, 100.0, 200.0, 240.0], "attributes": {"eyes_open": 0.9, "size": 0.35, "frontal": 0.95}}]


def reqs(d):
    return {k: Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in d.items()}


class _Base(unittest.TestCase):
    def setUp(self):
        clear_modes()
        for k in ("FAKE_DET_MODE", "FAKE_DET_READY", "FAKE_DET_CONTRACT", "FAKE_DET_CALLS"):
            os.environ.pop(k, None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        (Path(self.tmp) / "src").mkdir(parents=True, exist_ok=True)
        self.log = str(Path(self.tmp) / "thumb_calls.log")
        os.environ["FAKE_THUMBNAIL_CALLS"] = self.log
        self.csv = str(Path(self.tmp) / "src" / "titles.csv")
        Path(self.csv).write_text("title,slug\nFirst episode,ep1\nSecond one,ep2\n", encoding="utf-8")

    def tearDown(self):
        clear_modes()

    def calls(self):
        if not os.path.exists(self.log):
            return []
        return [json.loads(x) for x in Path(self.log).read_text(encoding="utf-8").splitlines()]

    def adapter(self, **kw):
        ws = str(Path(self.tmp) / "ws")
        os.makedirs(ws, exist_ok=True)
        return ThumbnailAdapter(CliSkill("thumbnail", [sys.executable, FAKE], None, {}), workspace=ws, allowed_inputs=[str(Path(self.tmp) / "src")], **kw)

    def det_service(self, **kw):
        svc = pipeline_service(self.tmp, caps=DetCaps(det_capabilities(lambda: det_adapter(self.tmp))), transcription=False, **kw)
        svc._adapter.register(det_adapter(self.tmp))
        svc.adapter([])
        return svc


class ContractTests(_Base):
    def test_every_pinned_tool_is_used_or_named_unused(self):
        live = {t["tool_id"] for t in pinned_contract()["tools"]}
        self.assertEqual(live - set(TOOLS), set(NOT_USED), "a re-pin that adds a tool must decide whether the agent uses it")
        self.assertEqual(check_contract(pinned_contract()), [])
        c = pinned_contract()
        next(t for t in c["tools"] if t["tool_id"] == "thumbnail/choose_frame")["writes_media"] = True
        self.assertTrue(any("thumbnail/choose_frame must declare role ranking" in e for e in check_contract(c)))
        c = pinned_contract()
        c["layered_output"]["manifest"]["file"] = "manifest.json"
        self.assertTrue(any("layered_output.manifest" in e for e in check_contract(c)))
        self.assertEqual(set(PRESETS), set(pinned_contract()["platform_presets"]["presets"]))
        self.assertTrue(set(FEATURE_KEYS) <= set(THUMBNAIL_KEYS))


class VocabularyTests(_Base):
    def parse(self, d):
        return parse_thumbnail_requirements(reqs(dict({"thumbnail": True}, **d)))

    def test_defaults_are_unchanged(self):
        th = self.parse({"thumbnail.text": "Hello"})
        self.assertEqual((th["at_faces"], th.get("variants"), th.get("layers"), th.get("batch"), th.get("avoid_faces")), (False, None, None, None, None))
        self.assertEqual(thumbnail_skill_of({"text": "Hello"}), "thumbnail_render")

    def test_requests_parse(self):
        th = self.parse({"thumbnail.text": "Hi", "thumbnail.at": "faces", "thumbnail.faces.every": 0.5, "thumbnail.avoid_faces": True, "thumbnail.avoid_faces.margin": 24,
                         "thumbnail.variants": 3, "thumbnail.variants.frames": "4, 6", "thumbnail.variants.min_contrast": 3})
        self.assertEqual((th["at_faces"], th["at"], th["faces"], th["avoid_faces"]), (True, None, {"every": 0.5}, {"margin": 24}))
        self.assertEqual(th["variants"], {"count": 3, "frames": [4.0, 6.0], "min_contrast": 3.0})
        self.assertEqual(thumbnail_skill_of(th), "thumbnail_variants")
        th = self.parse({"thumbnail.layers": True})
        self.assertEqual(thumbnail_skill_of(th), "thumbnail_layers", "layers of a bare frame still render the document")
        th = self.parse({"thumbnail.text": "{{title}}", "thumbnail.batch": self.csv, "thumbnail.batch.presets": "youtube,youtube-shorts", "thumbnail.batch.name": "{{slug}}-{{preset}}",
                         "thumbnail.batch.min_font_size": 24})
        b = th["batch"]
        self.assertEqual((b["rows"], b["presets"], b["min_font_size"]), (2, ["youtube", "youtube-shorts"], 24))
        self.assertEqual([f["name"] for f in b["files"]], ["ep1-youtube", "ep1-youtube-shorts", "ep2-youtube", "ep2-youtube-shorts"])
        self.assertEqual(b["csv_sha256"], hashlib.sha256(Path(self.csv).read_bytes()).hexdigest())

    def test_refusals(self):
        bad_csv = str(Path(self.tmp) / "src" / "bad.csv")
        Path(bad_csv).write_text("row,title\n1,a\n", encoding="utf-8")
        for given, frag in (({"thumbnail.avoid_faces": True, "thumbnail.at": "faces"}, "needs thumbnail.text"),
                            ({"thumbnail.avoid_faces": True, "thumbnail.text": "x"}, "needs thumbnail.at=faces"),
                            ({"thumbnail.avoid_faces": True, "thumbnail.text": "x", "thumbnail.at": "faces", "thumbnail.position": "top"}, "thumbnail.position is refused"),
                            ({"thumbnail.avoid_faces.margin": 4}, "thumbnail.avoid_faces is not"),
                            ({"thumbnail.faces.every": 1}, "thumbnail.at=faces, which is not set"),
                            ({"thumbnail.variants": 3}, "they need thumbnail.text"),
                            ({"thumbnail.variants": 13, "thumbnail.text": "x"}, "2..12"),
                            ({"thumbnail.variants.min_contrast": 3}, "thumbnail.variants is not"),
                            ({"thumbnail.variants": 2, "thumbnail.text": "x", "thumbnail.layers": True}, "different thumbnail-skill tools"),
                            ({"thumbnail.batch": self.csv}, "needs thumbnail.batch.presets"),
                            ({"thumbnail.batch": self.csv, "thumbnail.batch.presets": "myspace"}, "not one of the contract's presets"),
                            ({"thumbnail.batch": self.csv, "thumbnail.batch.presets": "tiktok", "thumbnail.format": "png"}, "accepts ['jpeg']"),
                            ({"thumbnail.text": "{{title}}"}, "needs thumbnail.batch"),
                            ({"thumbnail.text": "{{nope}}", "thumbnail.batch": self.csv, "thumbnail.batch.presets": "youtube"}, "names no column"),
                            ({"thumbnail.batch": self.csv, "thumbnail.batch.presets": "youtube", "thumbnail.batch.name": "same"}, "twice"),
                            ({"thumbnail.batch": self.csv, "thumbnail.batch.presets": "youtube", "thumbnail.batch.name": "{{title}}"}, "not a safe file name"),
                            ({"thumbnail.batch": bad_csv, "thumbnail.batch.presets": "youtube"}, "reserved"),
                            ({"thumbnail.batch.presets": "youtube"}, "thumbnail.batch is not")):
            with self.assertRaises(EditRequirementError, msg=str(given)) as cm:
                self.parse(given)
            self.assertIn(frag, str(cm.exception), given)

    def test_cp932_is_opt_in(self):
        p = str(Path(self.tmp) / "src" / "ja.csv")
        Path(p).write_bytes("title\n第一話\n".encode("cp932"))
        with self.assertRaises(EditRequirementError) as cm:
            read_batch_table(p, "utf-8", ",")
        self.assertIn("cp932", str(cm.exception))
        self.assertEqual(read_batch_table(p, "cp932", ",")[1], [{"title": "第一話"}])

    def test_frame_choice_arithmetic(self):
        self.assertEqual(inside_keep(5.0, [[0.0, 2.0], [4.0, 10.0]]), 3.0)
        self.assertIsNone(inside_keep(3.0, [[0.0, 2.0], [4.0, 10.0]]), "a removed frame is never snapped to its neighbour")
        o = Observation(kind="frame_ranking", asset_id="a", source="thumbnail/choose_frame@0.1.15", data={"frame_size": {"width": 1280, "height": 720}, "ranking": [
            {"rank": 1, "t": 3.0, "score": 0.9, "eyes_closed": False, "faces": [[100, 100, 50, 50]]},
            {"rank": 2, "t": 5.0, "score": 0.5, "eyes_closed": False, "faces": [[0, 0, 10, 10]]},
            {"rank": None, "t": 6.0, "score": None, "eyes_closed": None, "faces": []}]})
        fc = choose_ranked_frame([("a", o)], lambda src, t: inside_keep(t, [[0.0, 2.0], [4.0, 10.0]]))
        self.assertEqual((fc["rank"], fc["source_time"], fc["timestamp"]), (2, 5.0, 3.0), "the best frame the edit keeps")
        self.assertEqual(faces_on_picture(fc, (640, 360)), [[0.0, 0.0, 5.0, 5.0]])
        self.assertIsNone(faces_on_picture(fc, (720, 720)), "a picture of another aspect ratio is not a scaled source frame")


class AdapterTests(_Base):
    def src(self):
        return fake_media(self.tmp, "pic.mp4")

    def paths(self, *names):
        ws = Path(self.tmp) / "ws" / "artifacts"
        return {n: str(ws / n) for n in names}

    def base(self, **kw):
        return dict({"input": "pic", "timestamp": 1.0, "format": "png", "width": 1280, "height": 720, "text": "Hello", "output": "t-v1.png"}, **kw)

    def test_variants_request_and_flagged_report(self):
        ad = self.adapter(qc_skill_dir=self.tmp)
        paths = dict(self.paths("t-v1.png", "t-v2.png", "t-v3.png"), pic=self.src())
        args = self.base(variants={"count": 3, "outputs": ["t-v2.png", "t-v3.png"], "frames": [2.0], "min_contrast": 3})
        b = ad.build_request("thumbnail/render_variants", args, paths)
        v = b["request"]["params"]["variants"]
        self.assertEqual((v["count"], v["frame_element"], v["min_contrast_ratio"], v["qc_skill"]), (3, "frame", 3, self.tmp))
        self.assertEqual(Path(b["request"]["params"]["output"]["path"]).name, "t.png", "the Skill writes <stem>-v<k> beside output.path")
        with self.assertRaises(ToolError):
            ad.build_request("thumbnail/render_variants", dict(args, variants=dict(args["variants"], outputs=["t-v3.png", "t-v2.png"])), paths)
        with self.assertRaises(ToolError):
            ad.build_request("thumbnail/render_variants", {k: v for k, v in args.items() if k != "text"}, paths)
        os.environ["FAKE_THUMBNAIL_MODE"] = "variants_flag"
        r = ad.run(Operation(tool="thumbnail/render_variants", args=args, inputs=["pic"], outputs=["t-v1.png", "t-v2.png", "t-v3.png"]), paths)
        self.assertTrue(r.ok, r.data)
        self.assertEqual((r.output, r.data["flagged"], [x["variant"] for x in r.data["variants"]]), (paths["t-v1.png"], [2], [1, 2, 3]))

    def test_layers_are_read_back(self):
        ad = self.adapter()
        paths = dict(self.paths("t.png"), pic=self.src(), man=str(Path(self.tmp) / "ws" / "artifacts" / "t_layers" / "layers.json"))
        args = self.base(output="t.png", layers={"output": "man"})
        r = ad.run(Operation(tool="thumbnail/render_layers", args=args, inputs=["pic"], outputs=["t.png", "man"]), paths)
        self.assertTrue(r.ok, r.data)
        self.assertEqual((r.data["layers"]["count"], r.data["layers"]["verified"]["pixel_equal"]), (3, True))
        os.environ["FAKE_THUMBNAIL_MODE"] = "layers_tampered"
        r = ad.run(Operation(tool="thumbnail/render_layers", args=args, inputs=["pic"], outputs=["t.png", "man"]), paths)
        self.assertFalse(r.ok)
        self.assertIn("sha256 differs from the file", r.data["error"]["message"])
        with self.assertRaises(ToolError):
            ad.build_request("thumbnail/render_layers", dict(args, layers={"output": "t.png"}), paths)

    def test_batch_request_and_table_binding(self):
        ad = self.adapter()
        sha = hashlib.sha256(Path(self.csv).read_bytes()).hexdigest()
        outs = [{"row": r, "preset": p, "output": f"{r}-{p}"} for r in (1, 2) for p in ("youtube", "instagram-feed")]
        paths = {o["output"]: str(Path(self.tmp) / "ws" / "artifacts" / "batch" / f"{o['output']}.png") for o in outs}
        paths["pic"] = self.src()
        args = {"input": "pic", "timestamp": 1.0, "format": "png", "text": "{{title}}", "font_size": 60,
                "batch": {"csv": self.csv, "csv_sha256": sha, "encoding": "utf-8", "delimiter": ",", "name": "{{row}}-{{preset}}", "min_font_size": 20,
                          "targets": [{"preset": "youtube", "width": 1280, "height": 720}, {"preset": "instagram-feed", "width": 1080, "height": 1080}], "outputs": outs}}
        b = ad.build_request("thumbnail/render_batch", args, paths)
        t = b["request"]["params"]["targets"]
        self.assertEqual([x["document"]["canvas"]["width"] for x in t], [1280, 1080])
        self.assertEqual(t[0]["text_fit"], {"caption": {"max_width": 1152, "min_font_size": 20}})
        r = ad.run(Operation(tool="thumbnail/render_batch", args=args, inputs=["pic"], outputs=[o["output"] for o in outs]), paths)
        self.assertTrue(r.ok, r.data)
        self.assertEqual(len(r.data["items"]), 4)
        for bad, frag in (({"csv_sha256": "0" * 64}, "changed since the plan"), ({"targets": [{"preset": "youtube", "width": 1920, "height": 1080}]}, "preset's 1280x720"),
                          ({"outputs": outs[:3]}, "one file per row and preset")):
            with self.assertRaises(ToolError, msg=frag) as cm:
                ad.build_request("thumbnail/render_batch", dict(args, batch=dict(args["batch"], **bad)), paths)
            self.assertIn(frag, str(cm.exception))

    def test_place_text_before_the_render(self):
        ad = self.adapter()
        paths = dict(self.paths("t.png"), pic=self.src())
        args = self.base(output="t.png", avoid_faces={"faces": [[500, 560, 300, 160]], "frame_size": {"width": 1280, "height": 720}, "margin": 8})
        r = ad.run(Operation(tool="thumbnail/render", args=args, inputs=["pic"], outputs=["t.png"]), paths)
        self.assertTrue(r.ok, r.data)
        self.assertEqual(r.data["placement"]["chosen"], "top-center", "the bottom of the frame holds a face")
        cap = next(e for e in r.data["request_document"]["elements"] if e["element_id"] == "caption")["text"]
        self.assertEqual((cap["position"], cap["align"]["vertical"]), ({"x": 640.0, "y": 36}, "top"))
        self.assertEqual([c["argv"][0] for c in self.calls()].count("run"), 2, "place_text, then render")
        os.environ["FAKE_THUMBNAIL_MODE"] = "no_clear"
        r = ad.run(Operation(tool="thumbnail/render", args=args, inputs=["pic"], outputs=["t.png"]), paths)
        self.assertFalse(r.ok)
        self.assertEqual((r.data["error"]["code"], r.data["error"]["details"]["reason"]), ("VALIDATION_ERROR", "no_position_clear_of_faces"))

    def test_choose_frame_is_a_measurement(self):
        ad = self.adapter()
        det = det_adapter(self.tmp)
        src = fake_media(self.tmp, "talk.mp4", faces=FACES)
        m = det.measure("detection/faces", det.measurement_args("detection/faces", "face_attributes", src, "a1", {"frame_every": 1.0}, "an", "use"))
        self.assertTrue(m.ok, m.data)
        self.assertIn("--attributes", m.commands[0])
        self.assertEqual(len(m.commands), 1, "no tracks for attributes")
        args = {"input": src, "detections_path": m.data["detections_file"], "detections_sha256": m.data["detections_sha256"]}
        r = ad.measure("thumbnail/choose_frame", args)
        self.assertTrue(r.ok, r.data)
        ranked = [e for e in r.data["ranking"] if e["rank"]]
        self.assertEqual((ranked[0]["t"], ranked[0]["eyes_closed"]), (8.0, False), "eyes open first")
        self.assertTrue(all(e["eyes_closed"] for e in ranked if e["t"] < 5))
        bad = ad.measure("thumbnail/choose_frame", dict(args, detections_sha256="0" * 64))
        self.assertEqual((bad.ok, bad.data["error"]["code"]), (False, "INVALID_REQUEST"))
        os.environ["FAKE_THUMBNAIL_MODE"] = "rank_other_video"
        bad = ad.measure("thumbnail/choose_frame", args)
        self.assertEqual((bad.ok, bad.data["error"]["code"]), (False, "INVALID_RESULT"))
        with self.assertRaises(ToolError):
            ad.build_request("thumbnail/choose_frame", args, {})

    def test_lowering_names_its_tool(self):
        op = {"type": "graphics.thumbnail", "asset": "a", "input": "a_pic", "output": "a_thumbnail", "timestamp": 1.0, "format": "png", "width": 1280, "height": 720, "text": "x",
              "variants": {"count": 2, "outputs": ["a_thumbnail_v2"]}}
        self.assertEqual(lower_thumbnail("thumbnail/render_variants", op, "a_pic", "a_thumbnail")["variants"], op["variants"])
        with self.assertRaises(Exception):
            lower_thumbnail("thumbnail/render", op, "a_pic", "a_thumbnail")
        bop = dict(op, variants=None, batch={"csv": self.csv, "csv_sha256": "0" * 64, "encoding": "utf-8", "delimiter": ",", "name": "n", "rows": 1,
                                              "targets": [{"preset": "youtube", "width": 1280, "height": 720}], "outputs": [{"row": 1, "preset": "youtube", "name": "n", "output": "a_thumbnail_n"}]})
        args = lower_thumbnail("thumbnail/render_batch", bop, "a_pic", "a_thumbnail_n")
        self.assertEqual((sorted(args), args["batch"]["outputs"]), (["batch", "format", "input", "text", "timestamp"], [{"row": 1, "preset": "youtube", "output": "a_thumbnail_n"}]))


class PipelineTests(_Base):
    def test_frame_by_faces_and_caption_clear_of_them(self):
        svc = self.det_service(skills=["thumbnail", "video-editing"])
        media = fake_media(self.tmp, "talk.mp4", faces=FACES)
        r = plan_and_render(svc, [media], {"thumbnail": True, "thumbnail.text": "Hello", "thumbnail.at": "faces", "thumbnail.avoid_faces": True, "edit.trim": "1-12"}, profile="generic")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        kinds = {o["kind"] for o in d["analysis"]["observations"]}
        self.assertTrue({"face_attributes", "frame_ranking"} <= kinds)
        [op] = d["graphics"]["operations"]
        keep = next(o["keep"] for o in d["video"]["operations"] if o["type"] == "video.trim")
        self.assertEqual(len(keep), 1)
        self.assertEqual((op["frame_choice"]["source_time"], op["frame_choice"]["rank"], op["timestamp"]), (8.0, 1, round(8.0 - keep[0][0], 3)),
                         "the open-eyed frame, on the trimmed picture's timeline")
        self.assertEqual(op["avoid_faces"]["frame_size"], {"width": 1280, "height": 720})
        dec = next(x for x in d["decisions"] if x["subject"] == "thumbnail.render")
        self.assertEqual(dec["approval"], "CONFIRM")
        self.assertTrue(any(o["id"] in dec["evidence"] for o in d["analysis"]["observations"] if o["kind"] == "frame_ranking"))
        self.assertEqual(r["out"]["execution"]["status"], "COMPLETED", r["out"]["execution"])
        res = next(x for x in r["out"]["execution"]["results"] if x["tool"] == "thumbnail/render")
        self.assertIn(res["data"]["placement"]["chosen"], ("bottom-center", "top-center"))
        self.assertEqual([c["argv"][0] for c in self.calls() if c["cmd"] == "run"].count("run"), 3, "choose_frame (analysis), place_text, render")

    def test_frame_by_faces_without_detection_or_with_a_crop_blocks(self):
        svc = pipeline_service(self.tmp, transcription=False, skills=["thumbnail"])
        ir = svc.plan([fake_media(self.tmp, "talk.mp4", faces=FACES)], "generic", user_requirements={"thumbnail": True, "thumbnail.at": "faces"})
        blocks = [x for x in ir.doc["decisions"] if x["type"] == "BLOCK"]
        self.assertTrue(any(x["subject"] == "capability.face_attributes" for x in blocks), [x["subject"] for x in blocks])
        svc = self.det_service(skills=["thumbnail", "video-editing"])
        ir = svc.plan([fake_media(self.tmp, "talk2.mp4", faces=FACES)], "generic",
                      user_requirements={"thumbnail": True, "thumbnail.text": "x", "thumbnail.at": "faces", "thumbnail.avoid_faces": True,
                                         "edit.crop": {"x": 0, "y": 0, "width": 640, "height": 720}})
        b = next(x for x in ir.doc["decisions"] if x["subject"] == "thumbnail.render")
        self.assertEqual(b["type"], "BLOCK")
        self.assertIn("video.crop changes the picture's geometry", b["reason"])
        svc = self.det_service(skills=["thumbnail", "video-editing"])
        ir = svc.plan([fake_media(self.tmp, "noface.mp4")], "generic", user_requirements={"thumbnail": True, "thumbnail.at": "faces"})
        b = next(x for x in ir.doc["decisions"] if x["subject"] == "thumbnail.render")
        self.assertEqual(b["type"], "BLOCK")
        self.assertIn("no sampled frame of the picture shows a face", b["reason"])

    def test_variants_are_further_thumbnails_and_a_flag_warns(self):
        os.environ["FAKE_THUMBNAIL_MODE"] = "variants_flag"
        svc = pipeline_service(self.tmp)
        r = plan_and_render(svc, [fake_media(self.tmp, "talk.mp4")], {"thumbnail": True, "thumbnail.text": "Hi", "thumbnail.variants": 3})
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        step = next(s for s in d["plan"]["steps"] if s["skill"].startswith("thumbnail_"))
        subj = step["params"]["asset"]
        self.assertEqual((step["skill"], step["tool"], step["outputs"]), ("thumbnail_variants", "thumbnail/render_variants",
                                                                          [f"{subj}_thumbnail", f"{subj}_thumbnail_v2", f"{subj}_thumbnail_v3"]))
        self.assertEqual(r["out"]["execution"]["status"], "COMPLETED", r["out"]["execution"])
        arts = [a for a in svc.artifacts(r["ir"]) if a["type"] == "THUMBNAIL"]
        self.assertEqual(sorted(Path(a["path"]).name.rsplit("-", 1)[1] for a in arts), ["v1.png", "v2.png", "v3.png"])
        flagged = next(a for a in arts if a["path"].endswith("-v2.png"))
        self.assertEqual(flagged["qa_status"], "WARN")
        self.assertTrue(any(i["name"] == "phone_size" and i["status"] == "WARN" for i in flagged["qa"]["items"]))

    def test_layers_artifact(self):
        svc = pipeline_service(self.tmp)
        r = plan_and_render(svc, [fake_media(self.tmp, "talk.mp4")], {"thumbnail": True, "thumbnail.text": "Hi", "thumbnail.layers": True})
        self.assertEqual(r["validation"].errors, [])
        self.assertEqual(r["out"]["execution"]["status"], "COMPLETED", r["out"]["execution"])
        lay = next(a for a in svc.artifacts(r["ir"]) if a["type"] == "THUMBNAIL_LAYERS")
        self.assertEqual(Path(lay["path"]).name, "layers.json")
        self.assertTrue(Path(lay["path"]).parent.name.endswith("_thumbnail_layers"))
        self.assertEqual(lay["qa_status"], "PASS")

    def test_batch_per_row_and_preset(self):
        svc = pipeline_service(self.tmp)
        req = {"thumbnail": True, "thumbnail.text": "{{title}}", "thumbnail.batch": self.csv, "thumbnail.batch.presets": "youtube,youtube-shorts", "thumbnail.batch.name": "{{slug}}-{{preset}}"}
        r = plan_and_render(svc, [fake_media(self.tmp, "talk.mp4")], req)
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        [op] = d["graphics"]["operations"]
        self.assertEqual([t["preset"] for t in op["batch"]["targets"]], ["youtube", "youtube-shorts"])
        self.assertIn(str(Path(self.csv).parent), d["execution"]["allowed_inputs"])
        self.assertEqual(r["out"]["execution"]["status"], "COMPLETED", r["out"]["execution"])
        names = sorted(Path(a["path"]).name for a in svc.artifacts(r["ir"]) if a["type"] == "THUMBNAIL")
        self.assertEqual(names, ["ep1-youtube-shorts.png", "ep1-youtube.png", "ep2-youtube-shorts.png", "ep2-youtube.png"])
        bad = json.loads(json.dumps(d))
        bad["graphics"]["operations"][0]["batch"]["outputs"].pop()
        self.assertTrue(any("one named file per (row, preset)" in e for e in check_finishing_operations(bad)))
        ir = svc.plan([fake_media(self.tmp, "talk2.mp4")], "youtube", user_requirements=dict(req, **{"publish.platform": "youtube", "publish.title": "t", "publish.thumbnail": True, "qc": True}))
        self.assertTrue(any(x["type"] == "BLOCK" and "thumbnail.batch makes one per table row" in x["reason"] for x in ir.doc["decisions"]))

    def test_plain_thumbnail_ir_is_unchanged(self):
        svc = pipeline_service(self.tmp)
        ir = svc.plan([fake_media(self.tmp, "talk.mp4")], "youtube", user_requirements={"thumbnail": True, "thumbnail.text": "Hi"})
        [op] = ir.doc["graphics"]["operations"]
        self.assertEqual(sorted(op), ["asset", "color", "decision_ids", "font_id", "font_size", "format", "height", "input", "output", "position", "text", "timestamp", "type", "width"])
        self.assertEqual(next(s for s in ir.doc["plan"]["steps"] if s["skill"].startswith("thumbnail_"))["tool"], "thumbnail/render")


if __name__ == "__main__":
    unittest.main()
