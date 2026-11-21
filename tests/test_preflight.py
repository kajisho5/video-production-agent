"""ADR-126: the export preflight -- output directory, disk space, formats and platform limits, inputs, fonts / sidecars, loudness,
tools -- on the fake engine and fake Skills, plus the CLI, `render`'s refusal / --skip-preflight and the MCP tool. The same checks
on real lavfi clips through the real ffmpeg-skill are in tests/test_integration.py (PreflightRealTests)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import clear_modes, fake_media, pipeline_service  # noqa: E402
from video_agent import mcp_server  # noqa: E402
from video_agent.execution import preflight as pf  # noqa: E402
from video_agent.mcp_permissions import READ, TOOL_CLASSES, classify  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402


def unmounted(rest: str) -> str:
    """A path on a volume that is not mounted here: /Volumes/<name> on POSIX, an unused drive letter on Windows."""
    if os.name == "nt":
        letter = next(c for c in "ZYXWVUTSRQPONMLKJIHGFE" if not os.path.exists(f"{c}:\\"))
        return f"{letter}:\\{rest}"
    return f"/Volumes/NO_SUCH_VOLUME_va_preflight/{rest}"


def codes(res, severity=None):
    return sorted(p["code"] for p in res["problems"] if severity is None or p["severity"] == severity)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="va_preflight_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.src = fake_media(self.tmp, "a.mp4")
        self.svc = pipeline_service(self.tmp)

    def plan(self, reqs=None, name="p", profile="youtube"):
        ir = self.svc.plan([self.src], profile, user_requirements=reqs or {}, params={"language": "ja"})
        path = str(Path(self.svc.workspace) / "plans" / f"{name}.project.json")
        save_ir(ir, path)
        return path

    def test_a_valid_plan_passes_and_render_records_it(self):
        path = self.plan()
        res = self.svc.preflight(load_ir(path))
        self.assertTrue(res["ok"], res["problems"])
        self.assertEqual(codes(res, "error"), [])
        est = res["estimates"]
        self.assertEqual(len(est["deliverables"]), 1)
        d = est["deliverables"][0]
        self.assertEqual((d["platform"], d["duration_s"]), ("youtube", 11.0), "the plan's expected duration (the trim)")
        self.assertEqual(d["limits"]["max_duration_s"], 43200, "publish-skill's pinned upload_limits")
        self.assertGreaterEqual(est["required_bytes"], est["output_bytes"])
        self.assertIsNotNone(est["free_bytes"])
        self.assertTrue(any(u["field"] == "video.resolution" for u in res["unverified"]), "a delegated limit is unverified, never guessed")
        out = self.svc.render(load_ir(path), path, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED", out.get("preflight"))
        self.assertTrue(out["preflight"]["ok"])
        run = load_ir(path).doc["provenance"]["runs"][-1]
        self.assertEqual((run["preflight"]["skipped"], run["preflight"]["ok"], run["preflight"]["errors"]), (False, True, []))
        self.assertTrue((Path(self.svc.workspace) / "jobs" / out["job"]["id"] / "preflight.json").is_file())
        self.assertEqual(json.loads((Path(self.svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text())["preflight"]["ok"], True)

    def test_missing_input(self):
        path = self.plan()
        os.remove(self.src)
        res = self.svc.preflight(load_ir(path))
        self.assertFalse(res["ok"])
        self.assertIn("INPUT_MISSING", codes(res, "error"))
        p = next(p for p in res["problems"] if p["code"] == "INPUT_MISSING")
        self.assertTrue(p["field"].startswith("assets.") and p["field"].endswith(".path"), p)

    def test_input_on_an_unmounted_volume(self):
        path = self.plan()
        ir = load_ir(path)
        aid = next(iter(ir.doc["assets"]))
        ir.doc["assets"][aid]["path"] = unmounted("CLIP0001.MP4")
        res = self.svc.preflight(ir)
        p = next(p for p in res["problems"] if p["code"] == "INPUT_VOLUME_MISSING")
        self.assertIn(pf.missing_volume(unmounted("CLIP0001.MP4")), p["message"])
        if os.name != "nt":
            self.assertTrue(str(pf.missing_volume("/media/someone_va/NO_SUCH_CARD_va/x.mp4")).startswith("/media/someone_va"))
            self.assertIsNone(pf.missing_volume("/no_such_top_level_va/x.mp4"), "only a mount location counts as a volume")
        self.assertIsNone(pf.missing_volume(self.src))

    def test_changed_input_refuses_render_unless_skipped(self):
        path = self.plan()
        doc = json.loads(Path(self.src).read_text())
        doc["name"] = "replaced take"
        Path(self.src).write_text(json.dumps(doc))
        os.utime(self.src, (1_000_000_000, 1_000_000_000))
        res = self.svc.preflight(load_ir(path))
        self.assertIn("INPUT_CHANGED", codes(res, "error"))
        msg = next(p["message"] for p in res["problems"] if p["code"] == "INPUT_CHANGED")
        self.assertIn("size", msg)
        self.assertIn("modified time", msg)
        out = self.svc.render(load_ir(path), path, approve=["all"])
        self.assertEqual(out["status"], "BLOCKED")
        self.assertIn("INPUT_CHANGED", codes(out["preflight"]))
        self.assertIn("--skip-preflight", out["hint"])
        self.assertEqual(out["job"]["state"], "BLOCKED")
        jd = Path(self.svc.workspace) / "jobs" / out["job"]["id"]
        self.assertEqual([f for f in jd.rglob("*") if f.is_file() and f.parent != jd], [], "nothing was encoded")
        self.assertEqual(load_ir(path).doc["execution"]["preflight"]["errors"], ["INPUT_CHANGED"])
        out2 = self.svc.render(load_ir(path), path, approve=["all"], skip_preflight=True)
        self.assertEqual(out2["status"], "COMPLETED")
        self.assertEqual(out2["preflight"], {"skipped": True})
        self.assertTrue(load_ir(path).doc["provenance"]["runs"][-1]["preflight"]["skipped"], "the skip is recorded in the run")

    @unittest.skipIf(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), "permission bits do not stop root / Windows ACLs")
    def test_unwritable_output_dir(self):
        path = self.plan()
        jobs = Path(self.svc.workspace) / "jobs"
        jobs.mkdir(exist_ok=True)
        os.chmod(jobs, 0o555)
        self.addCleanup(os.chmod, jobs, 0o755)
        res = self.svc.preflight(load_ir(path))
        self.assertIn("OUTPUT_DIR_UNWRITABLE", codes(res, "error"))

    def test_unwritable_output_dir_probe(self):
        path = self.plan()
        with mock.patch.object(pf, "probe_writable", return_value="Permission denied"):
            res = self.svc.preflight(load_ir(path))
        p = next(p for p in res["problems"] if p["code"] == "OUTPUT_DIR_UNWRITABLE")
        self.assertEqual((p["severity"], p["field"]), ("error", "workspace"))
        self.assertIn("Permission denied", p["message"])

    def test_output_dir_that_is_a_file(self):
        path = self.plan()
        jobs = Path(self.svc.workspace) / "jobs"
        if jobs.exists():
            shutil.rmtree(jobs)
        jobs.write_text("not a folder")
        res = self.svc.preflight(load_ir(path))
        self.assertIn("OUTPUT_DIR_NOT_A_DIRECTORY", codes(res, "error"))

    def test_disk_space_over_a_fake_limit(self):
        path = self.plan()
        res = self.svc.preflight(load_ir(path), free_fn=lambda p: 10)
        self.assertIn("DISK_SPACE_LOW", codes(res, "error"))
        self.assertEqual(res["estimates"]["free_bytes"], 10)
        self.assertEqual(res["estimates"]["required_bytes"], int(res["estimates"]["output_bytes"] * pf.ESTIMATE_MARGIN))
        with mock.patch.object(pf, "disk_free", return_value=10):
            out = self.svc.render(load_ir(path), path, approve=["all"])
        self.assertEqual(out["status"], "BLOCKED")
        self.assertEqual(codes(out["preflight"], "error"), ["DISK_SPACE_LOW"])
        with mock.patch.object(pf, "disk_free", return_value=None):
            res = self.svc.preflight(load_ir(path))
        self.assertEqual(codes(res), ["DISK_SPACE_UNKNOWN"])
        self.assertTrue(res["ok"], "an unknown free space is a warning")

    def test_platform_limits_from_the_pinned_contract(self):
        long_src = fake_media(self.tmp, "long.mp4", duration=2400.0)
        ir = self.svc.plan([long_src], "youtube", user_requirements={}, params={"language": "ja"})
        ir.doc["video"]["operations"] = []   # the fake engine's silences trim every source to 11 s; keep the whole 2400 s
        for t in ir.doc["delivery"]["targets"]:
            t["platform"] = "linkedin"   # 1800 s documented maximum
        res = self.svc.preflight(ir)
        self.assertIn("PLATFORM_DURATION_OVER", codes(res, "error"))
        ir.doc["delivery"]["targets"][0]["platform"] = "youtube"
        res = self.svc.preflight(ir)
        self.assertNotIn("PLATFORM_DURATION_OVER", codes(res))
        self.assertIn("PLATFORM_DURATION_OVER_UNVERIFIED_ACCOUNT", codes(res, "warning"))

    def test_preset_platform_container_and_size(self):
        rep = pf.Report()
        doc = {"delivery": {"targets": [{"id": "x", "preset": "prores", "platform": "x"}, {"id": "y", "preset": "nope", "platform": "nowhere"}]}}
        subjects = [{"id": "s", "duration": 30.0}]
        paths = {"s_delivery_x": "/w/jobs/j/artifacts/s_x.mkv", "s_delivery_y": "/w/jobs/j/artifacts/s_y.mp4"}
        limits = {"platforms": {"x": {"video.file_formats": {"status": "documented", "value": [".mp4", ".mov"]},
                                      "video.max_bytes": {"status": "documented", "value": 1000}, "video.fps": {"status": "delegated", "owner": "ffmpeg-skill"}}}}
        rows = pf.check_targets(rep, doc, subjects, paths, {"s_delivery_x": 5000, "s_delivery_y": 10}, ["youtube", "prores"], ["youtube", "x"], limits)
        got = {(p["code"], p["severity"], p["field"]) for p in rep.problems}
        self.assertIn(("PLATFORM_CONTAINER_NOT_ACCEPTED", "error", "delivery.targets[0].preset"), got)
        self.assertIn(("PLATFORM_SIZE_OVER", "warning", "delivery.targets[0]"), got)
        self.assertIn(("FORMAT_PRESET_UNKNOWN", "error", "delivery.targets[1].preset"), got)
        self.assertIn(("FORMAT_PLATFORM_UNKNOWN", "error", "delivery.targets[1].platform"), got)
        self.assertEqual([r["id"] for r in rows], ["s_delivery_x", "s_delivery_y"])
        self.assertTrue(any(u["field"] == "video.fps" for u in rep.unverified))
        self.assertTrue(any("nowhere" in u["reason"] for u in rep.unverified), "no limits for an unknown platform: unverified, not invented")
        rep2 = pf.Report()
        pf.check_targets(rep2, doc, subjects, paths, {}, None, None, {})
        self.assertEqual(rep2.problems, [], "no preset list published: unverified, not refused")
        self.assertEqual(len([u for u in rep2.unverified if u["field"].endswith(".preset")]), 2)

    def test_loudness_target_required_by_the_profile(self):
        path = self.plan()
        ir = load_ir(path)
        ir.doc["audio"]["operations"] = [op for op in ir.doc["audio"]["operations"] if op["type"] != "audio.loudness"]
        res = self.svc.preflight(ir)
        self.assertIn("LOUDNESS_TARGET_UNSET", codes(res, "error"))
        for d in ir.doc["decisions"]:
            if d["subject"].startswith("audio.loudness"):
                d["status"] = "REJECTED"
        res = self.svc.preflight(ir)
        self.assertIn("LOUDNESS_TARGET_UNSET", codes(res, "warning"), "the user rejected it: a warning")
        self.assertNotIn("LOUDNESS_TARGET_UNSET", codes(self.svc.preflight(load_ir(self.plan(profile="generic", name="g")))))

    def test_fonts_dir_and_transcript_edit_sidecars(self):
        rep = pf.Report()
        fonts = Path(self.tmp) / "fonts"
        fonts.mkdir()
        doc = {"captions": {"operations": [{"type": "captions.burn", "fonts_dir": str(Path(self.tmp) / "nofonts")}, {"type": "captions.burn", "fonts_dir": str(fonts)}]},
               "decisions": [{"id": "dec_1", "subject": "transcript.ripple", "status": "APPROVED",
                              "params": {"transcript": {"path": str(Path(self.tmp) / "t.json")}, "subtitles": [{"path": str(Path(self.tmp) / "s.srt")}]}}]}
        pf.check_media(rep, doc, [], [])
        got = sorted((p["code"], p["severity"], p["field"]) for p in rep.problems)
        self.assertEqual(got, [("FONTS_DIR_EMPTY", "warning", "captions.operations[1].fonts_dir"), ("FONTS_DIR_MISSING", "error", "captions.operations[0].fonts_dir"),
                               ("SIDECAR_MISSING", "warning", "decisions.dec_1.params.subtitles[0].path"), ("SIDECAR_MISSING", "warning", "decisions.dec_1.params.transcript.path")])
        (fonts / "Brand.otf").write_bytes(b"x")
        rep = pf.Report()
        pf.check_media(rep, {"captions": doc["captions"]}, [], [])
        self.assertEqual([p["code"] for p in rep.problems], ["FONTS_DIR_MISSING"])

    def test_referenced_user_file_missing(self):
        img = Path(self.tmp) / "logo.png"
        img.write_bytes(b"png")

        class Op:
            def __init__(self, inputs, outputs):
                self.inputs, self.outputs, self.skill = inputs, outputs, ""
        rep = pf.Report()
        ops = [Op(["asset", "lut"], ["out"])]
        pf.check_inputs(rep, {"assets": {}}, ops, {"asset": str(img), "lut": str(Path(self.tmp) / "missing.cube"), "out": "/w/jobs/j/o.mp4"}, "/w/jobs/j")
        self.assertEqual([(p["code"], p["field"]) for p in rep.problems], [("INPUT_MISSING", "paths.lut")], "the job's own files are not inputs")

    def test_tooling(self):
        path = self.plan()
        ir = load_ir(path)
        ir.doc["plan"]["steps"][0]["tool"] = "no-such-skill/run"
        ir.doc["source"]["tool_versions"] = dict(ir.doc["source"]["tool_versions"], **{"video-agent": "0.0.1"})
        res = self.svc.preflight(ir)
        self.assertIn("TOOL_UNAVAILABLE", codes(res, "error"))
        self.assertIn("TOOL_VERSION_CHANGED", codes(res, "warning"))
        svc = pipeline_service(self.tmp, caps=__import__("pipeline_harness").PipelineCaps(missing=["qc"]))
        path = str(Path(svc.workspace) / "plans" / "q.project.json")
        save_ir(self.svc.plan([self.src], "youtube", user_requirements={"qc": True}, params={"language": "ja"}), path)
        res = svc.preflight(load_ir(path))
        self.assertIn("CAPABILITY_MISSING", codes(res, "error"))

    def test_offload_destinations(self):
        card = Path(self.tmp) / "card"
        card.mkdir()
        dest = Path(self.tmp) / "dest"
        dest.mkdir()
        step = {"params": {"source": str(card), "destinations": [str(dest), unmounted("backup")]}}
        res = pf.run_offload({}, step, {"bytes_total": 1000}, [], free_fn=lambda p: 500)
        got = sorted((p["code"], p["field"]) for p in res["problems"])
        self.assertEqual(got, [("DISK_SPACE_LOW", "plan.steps.offload.params.destinations[0]"), ("OUTPUT_VOLUME_MISSING", "plan.steps.offload.params.destinations[1]")])
        res = pf.run_offload({}, {"params": {"source": str(Path(self.tmp) / "gone"), "destinations": [str(dest)]}}, {"bytes_total": 1}, [], free_fn=lambda p: 10 ** 9)
        self.assertEqual(codes(res), ["INPUT_MISSING"])

    def test_estimate_writes(self):
        class Op:
            def __init__(self, inputs, outputs):
                self.inputs, self.outputs = inputs, outputs
        doc = {"assets": {"a": {"technical": {"duration": 100.0, "bitrate": 8000, "file": {"size": 100000}}}}, "delivery": {"targets": [{"id": "t", "preset": "prores"}]}}
        paths = {"a": "/s/a.mp4", "a_trim": "/j/ops/a_trim.mp4", "a_delivery_t": "/j/artifacts/a_t.mov", "a_srt": "/j/ops/a.srt"}
        ops = [Op(["a"], ["a_trim"]), Op(["a_trim"], ["a_srt"]), Op(["a_trim"], ["a_delivery_t"])]
        w = pf.estimate_writes(doc, ops, paths, [{"id": "a", "sources": ["a"], "duration": 50.0}])
        self.assertEqual(w["deliverables"], {"a_delivery_t": 50000}, "8000 bit/s x 50 s / 8")
        self.assertEqual(w["output_bytes"], 100000 + 0 + 50000, "the trim: its input (upper bound); the sidecar: 0; the deliverable")
        self.assertEqual(w["unknown_presets"], ["prores"])


class PreflightInterfaceTests(unittest.TestCase):
    def test_mcp_tool_is_read_and_maps_to_the_cli(self):
        self.assertEqual(TOOL_CLASSES["preflight"], READ)
        self.assertIn("preflight", [t["name"] for t in mcp_server.TOOLS])
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(mcp_server.tool_argv("preflight", {"project": "p.json"}, td), ["preflight", "--", "p.json"])
            self.assertEqual(mcp_server.tool_argv("render", {"project": "p.json", "skip_preflight": True}, td), ["render", "--skip-preflight", "--", "p.json"])
            from video_agent.mcp_permissions import load_permissions
            call = classify("preflight", {"project": "p.json"}, td, load_permissions(td, env={}))
            self.assertEqual(call.cls, READ)

    def test_cli_preflight(self):
        clear_modes()
        tmp = os.path.realpath(tempfile.mkdtemp(prefix="va_preflight_cli_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        src = fake_media(tmp, "a.mp4")
        svc = pipeline_service(tmp)
        path = str(Path(svc.workspace) / "plans" / "c.project.json")
        save_ir(svc.plan([src], "youtube", user_requirements={}, params={"language": "ja"}), path)
        os.remove(src)
        env = dict(os.environ, VIDEO_AGENT_WORKSPACE=svc.workspace, PYTHONPATH=os.pathsep.join([str(Path(__file__).resolve().parents[1] / "src"), os.environ.get("PYTHONPATH", "")]))
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "--json", "preflight", path], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 2, r.stderr)
        res = json.loads(r.stdout)
        self.assertFalse(res["ok"])
        self.assertIn("INPUT_MISSING", codes(res, "error"))
        self.assertEqual(sorted(res), ["estimates", "ok", "problems", "unverified"])
        r = subprocess.run([sys.executable, "-m", "video_agent.cli", "preflight", path], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 2)
        self.assertIn("INPUT_MISSING", r.stdout)


if __name__ == "__main__":
    unittest.main()
