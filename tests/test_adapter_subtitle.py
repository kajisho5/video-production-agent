"""SubtitleAdapter boundary tests: contract discovery / refusal / drift, request lowering, success mapping, every error mode,
timeout, security (argv as a list, no request keys on argv, no secrets in the child environment), plus a real-Skill class
skipped unless VIDEO_AGENT_SUBTITLE_DIR points at a subtitle-skill checkout."""
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

from video_agent.models import Operation  # noqa: E402
from video_agent.tools.base import ToolError  # noqa: E402
from video_agent.tools.ffmpeg_skill.adapter import PathPolicy  # noqa: E402
from video_agent.tools.skill_process import CliSkill  # noqa: E402
from video_agent.tools.subtitle import (ENV_DIR, PACKAGE, SKILL_ID, TOOL_GENERATE, TOOL_RENDER, ContractError, SubtitleAdapter, check_contract, contract_drift,  # noqa: E402
                                        lift_result, locate_subtitle, package_from_contract, pinned_contract)
from video_agent.tools.subtitle.adapter import CONTRACT_SKILL_ID, SKILL_FORBIDDEN_KEYS  # noqa: E402

FAKE = Path(__file__).resolve().parent / "fake_subtitle.py"
CUES = [{"id": "c1", "start": 0.0, "end": 1.5, "text": "Hello"}, {"id": "c2", "start": 1.5, "end": 2.9, "text": "World"}]


def fake_skill() -> CliSkill:
    return CliSkill("subtitle", [sys.executable, str(FAKE)], None, {})


class Env:
    """Set FAKE_SUBTITLE_* for one block (the fake reads its mode from the environment it inherits)."""

    def __init__(self, **kw):
        self.kw = {("FAKE_SUBTITLE_" + k.upper()): v for k, v in kw.items()}
        self.old = {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="subtitle-adapter-"))
        self.ws = os.path.join(self.tmp, "ws")
        self.engine = os.path.join(self.tmp, "engine")   # a directory standing in for the ffmpeg-skill checkout (never executed by the fake)
        os.makedirs(self.ws)
        os.makedirs(self.engine)
        self.video = os.path.join(self.ws, "intermediate", "cut.mp4")
        os.makedirs(os.path.dirname(self.video))
        Path(self.video).write_bytes(json.dumps({"fake": True, "duration": 3.0, "video": True, "channels": 2}).encode())
        self.paths = {"vid": self.video, "out": os.path.join(self.ws, "out", "sub.srt"), "vtt": os.path.join(self.ws, "out", "sub.vtt"),
                      "burn": os.path.join(self.ws, "out", "burn.mp4")}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def adapter(self, **kw):
        kw.setdefault("workspace", self.ws)
        kw.setdefault("ffmpeg_skill_dir", self.engine)
        return SubtitleAdapter(fake_skill(), **kw)

    @staticmethod
    def gen_args(**over):
        a = {"operation": "generate", "format": "srt", "document_id": "doc1", "language": "en", "cues": json.loads(json.dumps(CUES)), "output": "out"}
        a.update(over)
        return a

    @staticmethod
    def render_args(**over):
        a = {"operation": "render", "input": "vid", "format": "srt", "document_id": "doc1", "language": "en", "cues": json.loads(json.dumps(CUES)), "output": "burn"}
        a.update(over)
        return a

    @staticmethod
    def op(tool, args):
        return Operation(tool=tool, args=args, inputs=[], outputs=[args.get("output", "")], id="op1")


class ContractTests(Base):
    def test_discovery(self):
        a = self.adapter()
        self.assertEqual(a.version, "0.1.0")
        self.assertEqual(a.contract_version, "1.0.0")
        self.assertEqual(a.tools, {TOOL_GENERATE, TOOL_RENDER})
        self.assertTrue(a.supports(TOOL_RENDER) and not a.supports("subtitle/transcribe"))
        self.assertEqual(a.drift(), [])
        self.assertEqual(a.formats, {"generate": ["srt", "vtt"], "render": ["srt"]})
        self.assertEqual(a.constraint_keys, ("max_chars_per_line", "max_lines", "min_duration", "max_duration", "reading_speed_cps"))
        for k in SKILL_FORBIDDEN_KEYS:
            self.assertIn(k, a.forbidden)
        self.assertFalse(a.retryable["INVALID_INPUT"])
        self.assertTrue(a.retryable["TOOL_ERROR"])
        d = a.describe()
        self.assertEqual(d["name"], SKILL_ID)
        self.assertEqual(d["contract"], "1.0.0")

    def test_pinned_contract_and_package(self):
        self.assertEqual(check_contract(pinned_contract()), [])
        self.assertEqual(contract_drift(pinned_contract()), [])
        self.assertEqual(PACKAGE.skill_id, SKILL_ID)
        self.assertEqual(PACKAGE.name, CONTRACT_SKILL_ID)
        self.assertEqual(PACKAGE.capabilities, ["subtitle"])
        self.assertEqual(PACKAGE.repository, "kajisho5/subtitle-skill")
        self.assertEqual(sorted(PACKAGE.tool_ids()), [TOOL_GENERATE, TOOL_RENDER])
        r = PACKAGE.tool(TOOL_RENDER)
        self.assertEqual(r.required_capabilities, ["subtitle", "ffmpeg", "ffprobe", "ffmpeg-skill", "encoder:libx264", "filter:subtitles"])
        self.assertEqual(r.inputs, ["input", "sidecar", "output"])
        self.assertTrue(r.produces_output)
        g = PACKAGE.tool(TOOL_GENERATE)
        self.assertEqual(g.required_capabilities, ["subtitle"])
        self.assertEqual(g.inputs, ["output"])
        self.assertEqual(self.adapter().package().version, package_from_contract(pinned_contract()).version)

    def test_check_contract_refusals(self):
        c = pinned_contract()
        c["skill_id"] = "subtitle"   # the agent package id is not the Skill's id
        self.assertTrue(any("skill_id" in e for e in check_contract(c)))
        c = pinned_contract()
        c["operations"]["render"]["delegates_to"] = {"skill_id": "ffmpeg", "tool": "caption"}
        self.assertTrue(any("delegate" in e for e in check_contract(c)))
        c = pinned_contract()
        c["out_of_scope"].remove("speaker_diarization")
        self.assertTrue(any("out_of_scope" in e for e in check_contract(c)))
        c = pinned_contract()
        c["errors"]["codes"].remove("VALIDATION_ERROR")
        self.assertTrue(any("VALIDATION_ERROR" in e for e in check_contract(c)))
        c = pinned_contract()
        c["operations"]["render"]["formats"] = ["srt", "vtt"]
        self.assertTrue(any("render formats" in e for e in check_contract(c)))
        self.assertEqual(check_contract("nope"), ["contract is not an object"])

    def test_bad_contract_modes(self):
        for mode, frag in (("wrong_schema", "contract_version"), ("wrong_skill", "skill_id"), ("wrong_version", "version"), ("bad_contract", "deterministic"),
                           ("contract_fail", "failed")):
            with Env(mode=mode), self.assertRaises(ContractError) as cm:
                self.adapter()
            self.assertIn(frag, str(cm.exception), mode)

    def test_drift_detected(self):
        with Env(mode="contract_drift"):
            a = self.adapter()
        self.assertTrue(a.drift())
        self.assertTrue(any(d.startswith("operations") for d in a.drift()))
        self.assertTrue(any(d.startswith("capabilities") for d in a.drift()))

    def test_doctor(self):
        a = self.adapter()
        doc = a.doctor()
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["exit_code"], 0)
        self.assertEqual(a.operation_status(doc), {"generate": "supported", "render": "supported"})
        self.assertEqual(a.render_formats(doc), ["srt"])
        with Env(mode="doctor_fail"):
            doc = a.doctor()
        self.assertEqual(doc["status"], "fail")
        self.assertEqual(doc["exit_code"], 1)
        self.assertEqual(a.operation_status(doc), {"generate": "supported", "render": "unsupported"})
        self.assertTrue(doc["problems"])
        b = self.adapter(ffmpeg_skill_dir=None)   # no engine location: the Skill's doctor reports render unavailable
        self.assertEqual(b.operation_status(b.doctor())["render"], "unsupported")

    def test_locate(self):
        self.assertIsNone(locate_subtitle(env={"PATH": self.tmp}))
        root = Path(self.tmp) / "checkout"
        (root / "src" / "subtitle_skill").mkdir(parents=True)
        (root / "src" / "subtitle_skill" / "__init__.py").write_text("")
        s = locate_subtitle(env={ENV_DIR: str(root), "PATH": self.tmp})
        self.assertEqual(s.command[1:], ["-m", "subtitle_skill"])
        self.assertEqual(s.env, {"PYTHONPATH": str(root / "src")})

    def test_measure_refused(self):
        with self.assertRaises(ToolError):
            self.adapter().measure(TOOL_GENERATE, {})


class RequestTests(Base):
    def test_generate_lowering(self):
        a = self.adapter()
        b = a.build_request(TOOL_GENERATE, self.gen_args(constraints={"max_lines": 2, "reading_speed_cps": 17.5}, video_duration=3), self.paths, op_id="op1")
        r = b["request"]
        self.assertEqual(r["workspace"], a.workspace)
        self.assertEqual(r["output_path"], "out/sub.srt")
        self.assertEqual(r["subtitle"], {"id": "doc1", "version": 1, "language": "en", "cues": CUES})
        self.assertEqual(r["constraints"], {"max_lines": 2, "reading_speed_cps": 17.5})
        self.assertEqual(r["video_duration"], 3.0)
        self.assertNotIn("video_input", r)
        self.assertEqual(b["cue_count"], 2)
        self.assertEqual(b["output"], self.paths["out"])
        self.assertIsNone(b["input"])
        self.assertTrue(all("speaker" not in c for c in r["subtitle"]["cues"]))

    def test_render_lowering(self):
        Path(self.paths["out"]).parent.mkdir(parents=True)
        Path(self.paths["out"]).write_text("1\n")
        b = self.adapter().build_request(TOOL_RENDER, self.render_args(sidecar="out"), self.paths)
        r = b["request"]
        self.assertEqual(r["video_input"], "intermediate/cut.mp4")
        self.assertEqual(r["output_path"], "out/burn.mp4")
        self.assertEqual(b["input"], self.video)
        self.assertNotIn("sidecar", json.dumps(r))
        self.assertEqual(sorted(r), ["format", "operation", "output_path", "subtitle", "video_input", "workspace"])

    def test_document_id_sanitised(self):
        r = self.adapter().build_request(TOOL_GENERATE, self.gen_args(document_id="my doc/#1"), self.paths)["request"]
        self.assertEqual(r["subtitle"]["id"], "my_doc__1")
        r = self.adapter().build_request(TOOL_GENERATE, self.gen_args(document_id="/bad"), self.paths)["request"]
        self.assertEqual(r["subtitle"]["id"], "doc")

    def refuse(self, tool, args, frag, paths=None, **kw):
        with self.assertRaises(ToolError) as cm:
            self.adapter(**kw).build_request(tool, args, paths or self.paths)
        self.assertIn(frag, str(cm.exception))

    def test_refusals(self):
        self.refuse("subtitle/other", self.gen_args(), "unsupported tool")
        self.refuse(TOOL_GENERATE, self.gen_args(operation="render"), "does not match")
        self.refuse(TOOL_GENERATE, self.gen_args(format="ass"), "format 'ass'")
        self.refuse(TOOL_RENDER, self.render_args(format="vtt"), "format 'vtt'")
        self.refuse(TOOL_GENERATE, self.gen_args(format="vtt"), "extension must match")
        self.refuse(TOOL_GENERATE, self.gen_args(input="vid"), "unknown argument")
        self.refuse(TOOL_GENERATE, self.gen_args(style={"bold": True}), "unknown argument")
        self.refuse(TOOL_GENERATE, self.gen_args(language="english language"), "language")
        self.refuse(TOOL_GENERATE, self.gen_args(output=""), "output reference")
        self.refuse(TOOL_RENDER, self.render_args(input="nope"), "input not found")
        self.refuse(TOOL_RENDER, self.render_args(sidecar="nope"), "sidecar not found")

    def test_forbidden_keys(self):
        for k in ("command", "argv", "filter_complex", "vf", "api_key", "env", "shell", "executable", "ffmpeg", "path"):
            self.refuse(TOOL_GENERATE, self.gen_args(**{k: "x"}), "forbidden field")
        self.refuse(TOOL_GENERATE, self.gen_args(constraints={"Filter": "x"}), "forbidden field")

    def test_cue_validation(self):
        base = self.gen_args
        self.refuse(TOOL_GENERATE, base(cues=[]), "non-empty")
        self.refuse(TOOL_GENERATE, base(cues="c1"), "non-empty")
        self.refuse(TOOL_GENERATE, base(cues=[dict(CUES[0], speaker="A")]), "unknown field")
        self.refuse(TOOL_GENERATE, base(cues=[dict(CUES[0], metadata={})]), "unknown field")
        self.refuse(TOOL_GENERATE, base(cues=[dict(CUES[0], id="bad id")]), "identifier")
        self.refuse(TOOL_GENERATE, base(cues=[CUES[0], dict(CUES[1], id="c1")]), "duplicate")
        self.refuse(TOOL_GENERATE, base(cues=[dict(CUES[0], start=-1)]), "0 <= start < end")
        self.refuse(TOOL_GENERATE, base(cues=[dict(CUES[0], end=0.0)]), "0 <= start < end")
        self.refuse(TOOL_GENERATE, base(cues=[dict(CUES[0], start=float("nan"))]), "finite")
        self.refuse(TOOL_GENERATE, base(cues=[dict(CUES[0], end=float("inf"))]), "finite")
        self.refuse(TOOL_GENERATE, base(cues=[dict(CUES[0], start=True)]), "finite")
        self.refuse(TOOL_GENERATE, base(cues=[dict(CUES[0], text="")]), "non-empty string")
        self.refuse(TOOL_GENERATE, base(cues=[dict(CUES[0], text="   ")]), "non-empty string")
        self.refuse(TOOL_GENERATE, base(cues=[dict(CUES[0], text="x" * 2001)]), "at most 2000")
        self.refuse(TOOL_GENERATE, base(cues=[dict(CUES[0], text="a\x00b")]), "control characters")
        self.refuse(TOOL_GENERATE, base(cues=[dict(CUES[0], text="a\x1bb")]), "control characters")
        self.refuse(TOOL_GENERATE, base(cues=["c1"]), "must be an object")
        r = self.adapter().build_request(TOOL_GENERATE, base(cues=[dict(CUES[0], text="two\nlines")]), self.paths)
        self.assertEqual(r["request"]["subtitle"]["cues"][0]["text"], "two\nlines")

    def test_constraints_and_duration(self):
        base = self.gen_args
        self.refuse(TOOL_GENERATE, base(constraints={"max_words": 3}), "not declared")
        self.refuse(TOOL_GENERATE, base(constraints={"max_lines": 2.5}), "integer")
        self.refuse(TOOL_GENERATE, base(constraints={"max_lines": 0}), "positive")
        self.refuse(TOOL_GENERATE, base(constraints={"min_duration": float("inf")}), "finite")
        self.refuse(TOOL_GENERATE, base(constraints=[1]), "must be an object")
        self.refuse(TOOL_GENERATE, base(video_duration=0), "positive")
        self.refuse(TOOL_GENERATE, base(video_duration="3"), "finite")
        r = self.adapter().build_request(TOOL_GENERATE, base(constraints={}), self.paths)["request"]
        self.assertNotIn("constraints", r)

    def test_paths(self):
        outside = os.path.join(self.tmp, "elsewhere.srt")
        self.refuse(TOOL_GENERATE, self.gen_args(output="o"), "outside the workspace", paths={"o": outside})
        self.refuse(TOOL_GENERATE, self.gen_args(output="o"), "outside the workspace", paths={"o": self.ws + ".srt"})
        ext = os.path.join(self.tmp, "src", "ext.mp4")
        os.makedirs(os.path.dirname(ext))
        Path(ext).write_bytes(b'{"fake": true, "duration": 2.0}')
        # a source outside the workspace is refused even when it is an allowed input: the Skill resolves inside the workspace only
        self.refuse(TOOL_RENDER, self.render_args(input="ext"), "inside the workspace", paths={"ext": ext, "burn": self.paths["burn"]}, allowed_inputs=[os.path.dirname(ext)])
        self.refuse(TOOL_RENDER, self.render_args(input="ext"), "allowed roots", paths={"ext": ext, "burn": self.paths["burn"]}, allowed_inputs=[self.tmp + "/nothing"])
        self.refuse(TOOL_RENDER, self.render_args(sidecar="ext"), "allowed roots", paths=dict(self.paths, ext=ext))
        self.refuse(TOOL_RENDER, self.render_args(output="vid"), "overwrite its input")
        with self.assertRaises(ToolError) as cm:
            SubtitleAdapter(fake_skill()).build_request(TOOL_GENERATE, self.gen_args(), self.paths)
        self.assertIn("workspace is required", str(cm.exception))
        pol = PathPolicy([os.path.dirname(ext)], self.ws)
        self.refuse(TOOL_GENERATE, self.gen_args(output="o"), "outside workspace", paths={"o": outside}, path_policy=pol)
        b = self.adapter(path_policy=pol).build_request(TOOL_RENDER, self.render_args(), self.paths)
        self.assertEqual(b["request"]["video_input"], "intermediate/cut.mp4")

    def test_preview(self):
        lines = self.adapter().preview(self.op(TOOL_GENERATE, self.gen_args()), self.paths)
        self.assertTrue(lines[0].startswith("subtitle-skill run - --json  <<< {"))
        self.assertIn("refused", self.adapter().preview(self.op(TOOL_GENERATE, self.gen_args(format="ass")), self.paths)[0])


class RunTests(Base):
    def test_generate_success(self):
        for fmt, key in (("srt", "out"), ("vtt", "vtt")):
            a = self.adapter()
            r = a.run(self.op(TOOL_GENERATE, self.gen_args(format=fmt, output=key)), self.paths)
            self.assertTrue(r.ok, r.data)
            self.assertEqual(r.output, self.paths[key])
            d = r.data
            self.assertEqual(d["skill"], {"id": "subtitle", "version": "0.1.0"})
            self.assertEqual(d["status"], "completed")
            self.assertEqual(d["operation_type"], "generate")
            self.assertEqual(d["artifact"]["sha256"], hashlib.sha256(Path(r.output).read_bytes()).hexdigest())
            self.assertEqual(d["artifact"]["size"], os.path.getsize(r.output))
            self.assertEqual(d["artifact"]["format"], fmt)
            self.assertEqual(d["artifact"]["cue_count"], 2)
            self.assertFalse(d["artifact"]["reused"])
            self.assertEqual(d["timeline"], {"cue_count": 2})
            self.assertEqual(d["warnings"], [])
            self.assertEqual(d["commands"], [])
            self.assertEqual(r.commands, [])
            self.assertEqual(d["provenance"], {"skill": "subtitle-skill", "skill_version": "0.1.0", "contract_version": "1.0.0", "output_hash": d["artifact"]["sha256"]})
            self.assertNotIn("engine", d)
            self.assertNotIn("observation", d)
            text = Path(r.output).read_text(encoding="utf-8")
            self.assertIn("00:00:00" + ("," if fmt == "srt" else ".") + "000 --> 00:00:01", text)
            if fmt == "vtt":
                self.assertTrue(text.startswith("WEBVTT"))
            self.assertEqual(a.calls, 2)
            obs = lift_result(r, "asset1")
            self.assertEqual(obs.kind, "subtitle.file")
            self.assertEqual(obs.provenance, "OBSERVED")
            self.assertEqual(obs.skill, "subtitle")
            self.assertEqual(obs.asset_id, "asset1")
            self.assertEqual(obs.data, {"cue_count": 2, "format": fmt, "sha256": d["artifact"]["sha256"], "size": d["artifact"]["size"]})

    def test_render_success(self):
        r = self.adapter().run(self.op(TOOL_RENDER, self.render_args(video_duration=3.0)), self.paths)
        self.assertTrue(r.ok, r.data)
        d = r.data
        self.assertEqual(d["operation_type"], "render")
        self.assertEqual(d["engine"], {"id": "ffmpeg-skill", "version": "0.9.1-fake"})
        self.assertIsNone(d["observation"])
        meta = json.loads(Path(r.output).read_bytes())
        self.assertEqual(meta["duration"], 3.0)
        self.assertTrue(meta["captions"])
        self.assertEqual(d["artifact"]["sha256"], hashlib.sha256(Path(r.output).read_bytes()).hexdigest())

    def test_warning_observation(self):
        with Env(mode="warning_observation"):
            r = self.adapter().run(self.op(TOOL_GENERATE, self.gen_args()), self.paths)
        self.assertTrue(r.ok)
        self.assertEqual(r.data["warnings"][0]["code"], "TOO_MANY_LINES")
        self.assertEqual(r.data["warnings"][0]["severity"], "warning")

    def test_reused(self):
        a = self.adapter()
        self.assertTrue(a.run(self.op(TOOL_GENERATE, self.gen_args()), self.paths).ok)
        with Env(mode="reused"):
            r = a.run(self.op(TOOL_GENERATE, self.gen_args()), self.paths)
        self.assertTrue(r.ok)
        self.assertTrue(r.data["artifact"]["reused"])

    def test_dry_run(self):
        r = self.adapter().run(self.op(TOOL_GENERATE, self.gen_args()), self.paths, dry_run=True)
        self.assertTrue(r.ok and r.dry_run)
        self.assertIsNone(r.output)
        self.assertEqual(r.data["status"], "dry_run")
        self.assertEqual(r.data["request"]["output_path"], "out/sub.srt")
        self.assertFalse(os.path.exists(self.paths["out"]))
        self.assertIsNone(lift_result(r))

    def test_build_refusal_is_invalid_request(self):
        r = self.adapter().run(self.op(TOOL_GENERATE, self.gen_args(format="ass")), self.paths)
        self.assertFalse(r.ok)
        e = r.data["error"]
        self.assertEqual((e["code"], e["retryable"], e["recovery_class"], e["exit_code"]), ("INVALID_REQUEST", False, "INVALID_ARGS", 2))
        self.assertIsNone(lift_result(r))

    def run_mode(self, mode, tool=TOOL_GENERATE, args=None, timeout=None):
        a = self.adapter()   # the contract is fetched in ok mode; only the run misbehaves
        with Env(mode=mode):
            return a.run(self.op(tool, args or (self.render_args() if tool == TOOL_RENDER else self.gen_args())), self.paths, timeout=timeout)

    def test_error_modes(self):
        cases = (("tool_error", TOOL_RENDER, "TOOL_ERROR", True, "UNKNOWN", 1),
                 ("tool_error", TOOL_GENERATE, "OUTPUT_ERROR", True, "SKILL_ERROR", 1),
                 ("dependency_error", TOOL_RENDER, "DEPENDENCY_ERROR", True, "INVALID_ARGS", 1),
                 ("validation_error", TOOL_GENERATE, "VALIDATION_ERROR", False, "SKILL_ERROR", 1),
                 ("cancelled", TOOL_GENERATE, "CANCELLED", False, "TIMEOUT", 1),
                 ("internal_error", TOOL_GENERATE, "INTERNAL_ERROR", False, "SKILL_ERROR", 1),
                 ("unknown_code", TOOL_GENERATE, "INVALID_RESULT", False, "SKILL_ERROR", 1),
                 ("output_missing", TOOL_GENERATE, "INVALID_RESULT", False, "SKILL_ERROR", 9),
                 ("hash_mismatch", TOOL_GENERATE, "INVALID_RESULT", False, "SKILL_ERROR", 9),
                 ("cue_count_mismatch", TOOL_GENERATE, "INVALID_RESULT", False, "SKILL_ERROR", 9),
                 ("wrong_operation", TOOL_GENERATE, "INVALID_RESULT", False, "SKILL_ERROR", 9),
                 ("wrong_skill", TOOL_GENERATE, "INVALID_RESULT", False, "SKILL_ERROR", 9),
                 ("wrong_version", TOOL_GENERATE, "INVALID_RESULT", False, "SKILL_ERROR", 9),
                 ("no_engine", TOOL_RENDER, "INVALID_RESULT", False, "SKILL_ERROR", 9),
                 ("malformed", TOOL_GENERATE, "INVALID_RESULT", False, "SKILL_ERROR", 9),
                 ("empty", TOOL_GENERATE, "INVALID_RESULT", False, "SKILL_ERROR", 9),
                 ("two_docs", TOOL_GENERATE, "INVALID_RESULT", False, "SKILL_ERROR", 9),
                 ("text", TOOL_GENERATE, "INVALID_RESULT", False, "SKILL_ERROR", 1),
                 ("nonzero_ok", TOOL_GENERATE, "INVALID_RESULT", False, "SKILL_ERROR", 3))
        for mode, tool, code, retry, rc, exit_code in cases:
            r = self.run_mode(mode, tool)
            self.assertFalse(r.ok, mode)
            e = r.data["error"]
            self.assertEqual((e["code"], e["retryable"], e["recovery_class"], e["exit_code"]), (code, retry, rc, exit_code), mode)
            self.assertEqual(r.data["status"], "failed")
            self.assertIsNone(r.output)
            self.assertIn(mode in ("wrong_skill", "wrong_version") and "response skill" or "", e["message"])
            out = self.paths["burn" if tool == TOOL_RENDER else "out"]
            self.assertFalse(os.path.exists(out), f"{mode}: output left behind")
        self.assertEqual(self.run_mode("cancelled").data["error"]["details"], {"reason": "signal"})
        self.assertIn("issues", self.run_mode("validation_error").data["error"]["details"])
        self.assertIn("cue_count", self.run_mode("cue_count_mismatch").data["error"]["message"])
        self.assertIn("sha256", self.run_mode("hash_mismatch").data["error"]["message"])

    def test_skill_side_refusals(self):
        # the Skill's own verdicts on requests the adapter cannot pre-empt travel back with the Skill's retryable flag
        r = self.run_mode("ok", TOOL_RENDER, self.render_args(cues=[dict(CUES[0], text=" ")]))
        self.assertEqual(r.data["error"]["code"], "INVALID_REQUEST")   # the adapter refuses blank text before the Skill sees it
        with Env(mode="ok"):
            a = self.adapter(ffmpeg_skill_dir=None)
            r = a.run(self.op(TOOL_RENDER, self.render_args()), self.paths)
        self.assertEqual((r.data["error"]["code"], r.data["error"]["retryable"]), ("DEPENDENCY_ERROR", True))

    def test_vtt_render_unsupported(self):
        r = self.run_mode("vtt_render_unsupported", TOOL_RENDER, self.render_args(format="vtt"))
        self.assertEqual(r.data["error"]["code"], "INVALID_REQUEST")   # refused by the contract's render formats before the Skill runs
        self.assertIn("vtt", r.data["error"]["message"])

    def test_timeout(self):
        r = self.run_mode("timeout", timeout=1)
        self.assertFalse(r.ok)
        e = r.data["error"]
        self.assertEqual((e["code"], e["retryable"], e["recovery_class"], e["exit_code"], e["details"]["reason"]), ("CANCELLED", True, "TIMEOUT", 124, "timeout"))
        self.assertLess(r.seconds, 20)


class SecurityTests(Base):
    def test_argv_and_env(self):
        log = os.path.join(self.tmp, "calls.jsonl")
        with Env(calls=log):
            os.environ["VIDEO_AGENT_SECRET_TOKEN"] = "hunter2"   # something that must never be forwarded on purpose (VIDEO_* env is inherited, not added)
            try:
                a = self.adapter()
                r = a.run(self.op(TOOL_RENDER, self.render_args(cues=[dict(CUES[0], text="rm -rf / ; $(evil)")])), self.paths)
                a.doctor()
            finally:
                os.environ.pop("VIDEO_AGENT_SECRET_TOKEN", None)
        self.assertTrue(r.ok, r.data)
        calls = [json.loads(line) for line in Path(log).read_text(encoding="utf-8").splitlines()]
        self.assertEqual([c["cmd"] for c in calls], ["contract", "run", "doctor"])
        run = calls[1]
        self.assertIsInstance(run["argv"], list)
        self.assertEqual(run["argv"], ["run", "-", "--json"])
        self.assertEqual(run["engine_dir"], self.engine)
        for c in calls:
            joined = " ".join(c["argv"])
            for k in ("workspace", "output_path", "video_input", "cues", "rm -rf", "hunter2", self.ws):
                self.assertNotIn(k, joined)
            self.assertNotIn(ENV_DIR, c["env_video"])
        self.assertEqual(a.skill.env, {})

    def test_request_never_carries_forbidden_keys(self):
        a = self.adapter()
        b = a.build_request(TOOL_RENDER, self.render_args(constraints={"max_lines": 2}, video_duration=3), self.paths)
        flat = json.dumps(b["request"]).lower()
        for k in a.forbidden:
            if k == "workspace":
                continue
            self.assertNotIn(f'"{k}"', flat, k)
        self.assertEqual(set(b["request"]), {"operation", "format", "workspace", "output_path", "subtitle", "video_input", "constraints", "video_duration"})
        self.assertTrue(os.path.isabs(b["request"]["workspace"]))
        self.assertFalse(os.path.isabs(b["request"]["output_path"]))
        self.assertFalse(os.path.isabs(b["request"]["video_input"]))


@unittest.skipUnless(os.environ.get(ENV_DIR), f"set {ENV_DIR} to a subtitle-skill checkout")
class RealSkillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="subtitle-real-"))
        self.ws = os.path.join(self.tmp, "ws")
        os.makedirs(self.ws)
        self.engine = os.environ.get("VIDEO_AGENT_FFMPEG_SKILL_DIR")
        self.paths = {"out": os.path.join(self.ws, "out", "sub.srt"), "vtt": os.path.join(self.ws, "out", "sub.vtt"), "burn": os.path.join(self.ws, "out", "burn.mp4"),
                      "vid": os.path.join(self.ws, "intermediate", "bars.mp4")}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def adapter(self):
        return SubtitleAdapter(locate_subtitle(os.environ[ENV_DIR]), workspace=self.ws, ffmpeg_skill_dir=self.engine)

    def test_contract_doctor_generate(self):
        a = self.adapter()
        self.assertEqual(a.contract_version, "1.0.0")
        self.assertEqual(a.drift(), [])
        doc = a.doctor()
        self.assertEqual(doc["skill"], CONTRACT_SKILL_ID)
        self.assertEqual(a.operation_status(doc)["generate"], "supported")
        for fmt, key in (("srt", "out"), ("vtt", "vtt")):
            r = a.run(Operation(tool=TOOL_GENERATE, args=Base.gen_args(format=fmt, output=key, constraints={"max_lines": 2}, video_duration=3.0), inputs=[], outputs=[key], id="g1"), self.paths)
            self.assertTrue(r.ok, r.data)
            self.assertEqual(r.data["artifact"]["sha256"], hashlib.sha256(Path(r.output).read_bytes()).hexdigest())
            self.assertEqual(r.data["timeline"]["cue_count"], 2)
            self.assertIn("Hello", Path(r.output).read_text(encoding="utf-8"))
        r = a.run(Operation(tool=TOOL_GENERATE, args=Base.gen_args(cues=[dict(CUES[0], text="x" * 60)], constraints={"max_chars_per_line": 42}), inputs=[], outputs=["out"], id="g2"), self.paths)
        self.assertTrue(r.ok, r.data)
        self.assertEqual(r.data["warnings"][0]["code"], "LINE_TOO_LONG")

    def test_render(self):
        if not self.engine or not shutil.which("ffmpeg"):
            self.skipTest("VIDEO_AGENT_FFMPEG_SKILL_DIR and ffmpeg are needed for render")
        a = self.adapter()
        if a.operation_status(a.doctor())["render"] != "supported":
            self.skipTest("the Skill's doctor reports render unsupported here")
        os.makedirs(os.path.dirname(self.paths["vid"]))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "3",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", self.paths["vid"]], check=True)
        r = a.run(Operation(tool=TOOL_RENDER, args=Base.render_args(), inputs=["vid"], outputs=["burn"], id="r1"), self.paths, timeout=300)
        self.assertTrue(r.ok, r.data)
        self.assertEqual(r.data["engine"]["id"], "ffmpeg-skill")
        self.assertTrue(r.data["engine"]["version"])
        self.assertIsNone(r.data["observation"])
        self.assertGreater(os.path.getsize(r.output), 1000)
        r2 = a.run(Operation(tool=TOOL_RENDER, args=Base.render_args(format="vtt"), inputs=["vid"], outputs=["burn"], id="r2"), self.paths)
        self.assertEqual(r2.data["error"]["code"], "INVALID_REQUEST")


if __name__ == "__main__":
    unittest.main()
