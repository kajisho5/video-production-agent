"""ColorGradingAdapter (ADR-031) against the fake color-grading process (tests/fake_color_grading.py): contract discovery and refusals,
drift, request lowering from the contract's parameter schemas, response verification (realpath, sha256 recomputed, probe facts,
provenance), error mapping with the Skill's retryable verdict, the path policy and the security boundary. A real-Skill class runs
when VIDEO_AGENT_COLOR_GRADING_DIR names a checkout (and VIDEO_AGENT_FFMPEG_SKILL_DIR its engine)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_agent.execution.recovery import classify_error, next_attempt  # noqa: E402
from video_agent.models import Operation  # noqa: E402
from video_agent.tools.base import ToolError  # noqa: E402
from video_agent.tools.color_grading import PACKAGE, ColorGradingAdapter, ContractError, check_contract, contract_drift, lift_observation, locate_color_grading, pinned_contract  # noqa: E402
from video_agent.tools.skill_process import RECOVERY_CLASS, CliSkill  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_color_grading.py")


def fake_video(tmp: str, name: str = "clip.mp4", hdr: bool = False) -> str:
    p = Path(tmp) / "src" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(json.dumps({"fake": True, "duration": 16.0, "video": True, "channels": 2, "hdr": hdr, "lufs": -11.0}).encode())
    return str(p)


class ColorGradingAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.src = fake_video(self.tmp)
        self.ws = str(Path(self.tmp) / "ws")
        os.makedirs(self.ws)
        for k in ("FAKE_CG_MODE", "FAKE_CG_CALLS"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ("FAKE_CG_MODE", "FAKE_CG_CALLS"):
            os.environ.pop(k, None)

    def _skill(self):
        return CliSkill("color-grading", [sys.executable, FAKE], None, {})

    def _adapter(self, **kw):
        kw.setdefault("workspace", self.ws)
        kw.setdefault("allowed_inputs", [str(Path(self.src).parent)])
        kw.setdefault("ffmpeg_skill_dir", self.tmp)
        return ColorGradingAdapter(self._skill(), **kw)

    def _op(self, **args):
        a = {"operation": "RETAG", "input": "a", "output": "a_retag", "target": "bt709"}
        a.update(args)
        return Operation(tool="color-grading/run", args=a, inputs=[a.get("input", "a")], outputs=[a["output"]], id="op_test")

    def _paths(self):
        out = str(Path(self.ws) / "jobs" / "j1" / "ops" / "01_retag" / "clip_retag.mp4")
        return {"a": self.src, "a_retag": out}

    def test_contract_discovery_package_and_refusals(self):
        ad = self._adapter()
        self.assertEqual((ad.version, ad.tools, ad.drift(), sorted(ad.operations)),
                         ("0.2.0", {"color-grading/run"}, [], ["HDR_TO_SDR", "LUT_APPLY", "PRIMARY_CORRECTION", "RETAG", "STRIP_DOVI"]))
        self.assertIn("GAMMA", ad.unsupported); self.assertEqual(ad.formats, ["m4v", "mkv", "mov", "mp4"])
        pk = ad.package()
        self.assertEqual((pk.skill_id, pk.version, pk.capabilities, pk.tool_ids(), pk.validate()), ("color-grading", "0.2.0", ["ffmpeg", "ffprobe", "ffmpeg-skill", "color-grading"], ["color-grading/run"], []))
        self.assertEqual((PACKAGE.skill_id, PACKAGE.repository), ("color-grading", "kajisho5/color-grading-skill"))
        self.assertEqual(check_contract(pinned_contract()), []); self.assertEqual(contract_drift(pinned_contract()), [])
        for mode, msg in (("wrong_schema", "contract schema"), ("wrong_skill", "skill_id"), ("wrong_version", "version"), ("bad_contract", "execution.shell"), ("contract_fail", "failed")):
            os.environ["FAKE_CG_MODE"] = mode
            with self.assertRaises(ContractError, msg=mode) as cm:
                self._adapter()
            self.assertIn(msg, str(cm.exception))
        os.environ["FAKE_CG_MODE"] = "contract_drift"
        ad2 = self._adapter()
        self.assertTrue(ad2.drift() and any("EXPOSURE" in x for x in ad2.drift()), "a compatible but different contract is reported as drift")
        os.environ.pop("FAKE_CG_MODE")
        doc = ad.doctor()
        self.assertEqual((doc["status"], ad.operation_status(doc)["RETAG"], doc["exit_code"]), ("ok", "supported", 0))
        os.environ["FAKE_CG_MODE"] = "doctor_degraded"
        self.assertEqual(self._adapter().operation_status(self._adapter().doctor())["LUT_APPLY"], "unavailable")

    def test_request_lowering_and_refusals(self):
        ad = self._adapter(); paths = self._paths()
        b = ad.build_request("color-grading/run", self._op().args, paths, op_id="op_1")
        req = b["request"]
        self.assertEqual(req["schema"], "color-grading/request@1"); self.assertEqual(req["project"]["operations"], [{"op_id": "edit", "type": "RETAG", "input": "source", "parameters": {"target": "bt709"}}])
        self.assertEqual((req["project"]["outputs"][0]["format"], req["project"]["outputs"][0]["operation"], req["project"]["source"]["path"]), ("mp4", "op:edit", str(Path(self.src).resolve())))
        self.assertEqual(b["workspace"], self.ws)
        lut = Path(self.tmp) / "src" / "look.cube"; lut.write_text("TITLE fake\n")
        b = ad.build_request("color-grading/run", {"operation": "LUT_APPLY", "input": "a", "output": "a_retag", "lut": "lut1", "lut_strength": 0.5}, dict(paths, lut1=str(lut)))
        self.assertEqual(b["request"]["project"]["operations"][0]["parameters"], {"lut_path": str(lut.resolve()), "lut_strength": 0.5}); self.assertEqual(b["lut_root"], str(lut.parent))
        self.assertIn("--allowed-lut", ad._argv(b, False, None))
        b = ad.build_request("color-grading/run", {"operation": "PRIMARY_CORRECTION", "input": "a", "output": "a_retag", "exposure": 0.5, "saturation": 0.0}, paths)
        self.assertEqual(b["request"]["project"]["operations"][0]["parameters"], {"exposure": 0.5, "saturation": 0.0})
        for bad, msg in (({"filter": "x"}, "forbidden"), ({"argv": ["x"]}, "forbidden"), ({"operation": "GAMMA"}, "unsupported"), ({"operation": "NOPE"}, "unknown operation"),
                         ({"target": "bt2100"}, "not one of"), ({"tonemap": "hable"}, "not declared"), ({"operation": "HDR_TO_SDR", "peak_nits": 99999}, "above"), ({"operation": "HDR_TO_SDR", "crf": 1.5}, "integer"),
                         ({"operation": "PRIMARY_CORRECTION", "exposure": 999}, "above"), ({"operation": "PRIMARY_CORRECTION", "contrast": -1}, "below"),
                         ({"operation": "LUT_APPLY"}, "requires parameter"), ({"format": "avi"}, "not one of"), ({"input": "missing"}, "not found")):
            args = dict(self._op().args); args.update(bad)
            if "operation" in bad and bad["operation"] != "RETAG":
                args.pop("target", None)
            with self.assertRaises(ToolError, msg=str(bad)) as cm:
                ad.build_request("color-grading/run", args, dict(paths, missing=str(Path(self.tmp) / "nope.mp4")))
            self.assertIn(msg, str(cm.exception).lower() if msg.islower() else str(cm.exception), str(bad))
        far = fake_video(os.path.realpath(tempfile.mkdtemp()), "far.mp4")
        with self.assertRaises(ToolError):
            ad.build_request("color-grading/run", self._op(input="far").args, dict(paths, far=far))
        with self.assertRaises(ToolError):
            ad.build_request("color-grading/run", self._op().args, dict(paths, a_retag=str(Path(self.tmp) / "outside.mp4")))
        with self.assertRaises(ToolError):
            ad.build_request("color-grading/run", self._op().args, dict(paths, a_retag=str(Path(self.ws) / "x.mov")))   # the container is never converted

    def test_success_mapping(self):
        ad = self._adapter(); paths = self._paths()
        r = ad.run(self._op(), paths, timeout=30)
        self.assertTrue(r.ok, r.data); self.assertEqual(r.output, paths["a_retag"]); self.assertTrue(os.path.isfile(r.output))
        self.assertEqual(r.data["artifact"]["sha256"], hashlib.sha256(Path(r.output).read_bytes()).hexdigest())
        self.assertEqual((r.data["operation_type"], r.data["operation"]["type"], r.data["operation"]["tool"], r.data["status"]), ("RETAG", "RETAG", "ffmpeg-skill/color", "completed"))
        self.assertEqual((r.data["observation"]["provenance"], r.data["observation"]["source"], r.data["observation"]["data"]["duration"]), ("OBSERVED", "ffmpeg-skill/probe@0.9.2-fake", 16.0))
        self.assertEqual((r.data["provenance"]["skill"], r.data["provenance"]["output_hash"]), ("color-grading", r.data["artifact"]["sha256"]))
        self.assertTrue(r.commands and all(isinstance(c, str) for c in r.commands), "commands are recorded as provenance only")
        obs = lift_observation(r, "a_retag")
        self.assertEqual((obs.kind, obs.skill, obs.tool, obs.fingerprint), ("media.probe", "color-grading", "color-grading/run", r.data["artifact"]["sha256"]))
        d = ad.run(self._op(), paths, dry_run=True)
        self.assertTrue(d.ok and d.dry_run and d.output is None and d.data["status"] == "dry_run")
        self.assertTrue(ad.preview(self._op(), paths)[0].startswith("color-grading run - --json"))
        os.environ["FAKE_CG_MODE"] = "reused"
        r2 = ad.run(self._op(), paths)
        self.assertTrue(r2.ok and r2.data["artifact"]["reused"])

    def test_primary_correction_success_mapping(self):
        ad = self._adapter(); paths = self._paths()
        op = Operation(tool="color-grading/run", args={"operation": "PRIMARY_CORRECTION", "input": "a", "output": "a_retag", "exposure": 0.5, "temperature": 5600.0},
                       inputs=["a"], outputs=["a_retag"], id="op_correct")
        r = ad.run(op, paths, timeout=30)
        self.assertTrue(r.ok, r.data)
        self.assertEqual((r.data["operation_type"], r.data["operation"]["type"], r.data["operation"]["tool"], r.data["operation"]["parameters"]),
                         ("PRIMARY_CORRECTION", "PRIMARY_CORRECTION", "ffmpeg-skill/color", {"exposure": 0.5, "temperature": 5600.0}))

    def test_error_mapping_and_verification(self):
        ad = self._adapter(); paths = self._paths()
        cases = {"tool_error": ("TOOL_ERROR", True, "RETRY"), "tool_error_final": ("TOOL_ERROR", False, "BLOCK"), "validation_error": ("VALIDATION_ERROR", False, "BLOCK"),
                 "cancelled": ("CANCELLED", True, "BLOCK"), "internal_error": ("INTERNAL_ERROR", False, "BLOCK"), "malformed": ("INVALID_RESULT", False, "BLOCK"), "empty": ("INVALID_RESULT", False, "BLOCK"),
                 "text": ("INVALID_RESULT", False, "BLOCK"), "two_docs": ("INVALID_RESULT", False, "BLOCK"), "nonzero_ok": ("INVALID_RESULT", False, "BLOCK"), "unknown_code": ("INVALID_RESULT", False, "BLOCK"),
                 "output_missing": ("INVALID_RESULT", False, "BLOCK"), "hash_mismatch": ("INVALID_RESULT", False, "BLOCK"), "no_provenance": ("INVALID_RESULT", False, "BLOCK"), "not_hdr": ("TOOL_ERROR", True, "RETRY")}
        for mode, (code, retry, action) in cases.items():
            os.environ["FAKE_CG_MODE"] = mode
            op = Operation(tool="color-grading/run", args={"operation": "HDR_TO_SDR", "input": "a", "output": "a_retag"}, inputs=["a"], outputs=["a_retag"], id="op_hdr") if mode == "not_hdr" else self._op()
            r = ad.run(op, paths, timeout=30)
            self.assertFalse(r.ok, mode); self.assertIsNone(r.output, mode)
            self.assertEqual((r.data["error"]["code"], r.data["error"]["retryable"], r.data["error"]["recovery_class"]), (code, retry, RECOVERY_CLASS.get(code, "SKILL_ERROR")), mode)
            self.assertEqual(next_attempt(r, 1, 2, 30)["action"], action, mode)
            self.assertFalse(os.path.exists(paths["a_retag"]), f"{mode}: a failed call leaves no output behind")
        os.environ["FAKE_CG_MODE"] = "tool_error"
        r = ad.run(self._op(), paths)
        self.assertEqual(classify_error(r), "UNKNOWN"); self.assertNotIn("argv", json.dumps(r.data["error"]["details"]), "error details are scrubbed")
        os.environ["FAKE_CG_MODE"] = "timeout"
        r = ad.run(self._op(), paths, timeout=1)
        self.assertEqual((r.exit_code, r.data["error"]["code"], r.data["error"]["retryable"], r.data["error"]["details"]["reason"]), (124, "CANCELLED", True, "timeout"))
        self.assertEqual(classify_error(r), "TIMEOUT")

    def test_security_boundary(self):
        ad = self._adapter(); paths = self._paths()
        log = str(Path(self.tmp) / "calls.log"); os.environ["FAKE_CG_CALLS"] = log
        r = ad.run(self._op(), paths)
        self.assertTrue(r.ok)
        run = [json.loads(line) for line in Path(log).read_text().splitlines()][-1]
        self.assertEqual(run["cmd"], "run"); self.assertIsInstance(run["argv"], list)
        self.assertEqual(run["argv"][:3], ["run", "-", "--json"]); self.assertNotIn("bt709", " ".join(run["argv"]), "parameters travel on stdin, never on argv")
        self.assertEqual(run["argv"][run["argv"].index("--ffmpeg-skill") + 1], self.tmp)
        inherited = {k: v for k, v in os.environ.items() if k.startswith("VIDEO_")}
        self.assertEqual(run["env_video"], inherited)
        for k in ("command", "argv", "filter", "shell", "env", "api_key", "workspace"):
            self.assertIn(k, ad.forbidden)
        with self.assertRaises(ToolError):
            ad.measure("color-grading/run", {})


class ColorGradingRealSkillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skill = locate_color_grading(os.environ.get("VIDEO_AGENT_COLOR_GRADING_DIR")) if os.environ.get("VIDEO_AGENT_COLOR_GRADING_DIR") else None
        cls.engine = os.environ.get("VIDEO_AGENT_FFMPEG_SKILL_DIR")
        cls.tmp = os.path.realpath(tempfile.mkdtemp())
        cls.src = str(Path(cls.tmp) / "src" / "bars.mp4")
        if cls.skill and cls.engine and shutil.which("ffmpeg"):
            os.makedirs(os.path.dirname(cls.src))
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                            "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", cls.src], check=True)

    def test_real_retag(self):
        if not (self.skill and self.engine and shutil.which("ffmpeg")):
            self.skipTest("needs VIDEO_AGENT_COLOR_GRADING_DIR, VIDEO_AGENT_FFMPEG_SKILL_DIR and ffmpeg")
        ws = str(Path(self.tmp) / "ws"); os.makedirs(ws, exist_ok=True)
        ad = ColorGradingAdapter(self.skill, workspace=ws, allowed_inputs=[str(Path(self.src).parent)], ffmpeg_skill_dir=self.engine, timeout=120)
        self.assertEqual(ad.drift(), [], "the installed color-grading-skill contract drifted from the pinned one: re-verify the adapter")
        doc = ad.doctor()
        self.assertEqual(doc["status"], "ok", doc.get("problems")); self.assertEqual(ad.operation_status(doc)["RETAG"], "supported")
        paths = {"a": self.src, "a_retag": str(Path(ws) / "ops" / "01_retag" / "bars_retag.mp4")}
        r = ad.run(Operation(tool="color-grading/run", args={"operation": "RETAG", "input": "a", "output": "a_retag", "target": "bt709"}, inputs=["a"], outputs=["a_retag"], id="op_real"), paths, timeout=120)
        self.assertTrue(r.ok, r.data)
        self.assertEqual(r.data["artifact"]["sha256"], hashlib.sha256(Path(r.output).read_bytes()).hexdigest())
        self.assertEqual((r.data["artifact"]["color_primaries"], r.data["artifact"]["color_transfer"], r.data["artifact"]["hdr"]), ("bt709", "bt709", False))
        bad = ad.run(Operation(tool="color-grading/run", args={"operation": "HDR_TO_SDR", "input": "a", "output": "a_retag"}, inputs=["a"], outputs=["a_retag"], id="op_real2"), paths, timeout=120)
        self.assertFalse(bad.ok); self.assertIn(bad.data["error"]["code"], ("INVALID_INPUT", "TOOL_ERROR"), "an SDR source is not tone-mapped by the Skill either (the engine refuses)")

    def test_real_primary_correction(self):
        if not (self.skill and self.engine and shutil.which("ffmpeg")):
            self.skipTest("needs VIDEO_AGENT_COLOR_GRADING_DIR, VIDEO_AGENT_FFMPEG_SKILL_DIR and ffmpeg")
        ws = str(Path(self.tmp) / "ws2"); os.makedirs(ws, exist_ok=True)
        ad = ColorGradingAdapter(self.skill, workspace=ws, allowed_inputs=[str(Path(self.src).parent)], ffmpeg_skill_dir=self.engine, timeout=120)
        self.assertEqual(ad.operation_status(ad.doctor())["PRIMARY_CORRECTION"], "supported")
        paths = {"a": self.src, "a_correct": str(Path(ws) / "ops" / "01_correct" / "bars_correct.mp4")}
        op = Operation(tool="color-grading/run", args={"operation": "PRIMARY_CORRECTION", "input": "a", "output": "a_correct", "exposure": 0.5, "saturation": 0.0},
                       inputs=["a"], outputs=["a_correct"], id="op_real3")
        r = ad.run(op, paths, timeout=120)
        self.assertTrue(r.ok, r.data)
        self.assertEqual(r.data["artifact"]["sha256"], hashlib.sha256(Path(r.output).read_bytes()).hexdigest())
        # the Skill's own response reports effective parameters (its own declared defaults filled in), not just what this request sent
        params = r.data["operation"]["parameters"]
        self.assertEqual((params["exposure"], params["saturation"]), (0.5, 0.0))
        self.assertIn("input", r.data["operation"]["measurements"]); self.assertIn("output", r.data["operation"]["measurements"])


if __name__ == "__main__":
    unittest.main()
