"""ColorGradingAdapter (ADR-031) against the fake color-grading process (tests/fake_color_grading.py): contract discovery and refusals,
drift, request lowering from the contract's parameter schemas, response verification (realpath, sha256 recomputed, probe facts,
provenance), error mapping with the Skill's retryable verdict, the path policy and the security boundary. A real-Skill class runs
when VIDEO_AGENT_COLOR_GRADING_DIR names a checkout (and VIDEO_AGENT_FFMPEG_SKILL_DIR its engine)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_agent.execution.recovery import classify_error, next_attempt  # noqa: E402
from video_agent.models import Operation  # noqa: E402
from video_agent.tools.base import ToolError  # noqa: E402
from video_agent.tools.color_grading import PACKAGE, ColorGradingAdapter, ContractError, check_contract, contract_drift, lift_observation, locate_color_grading, pinned_contract  # noqa: E402
from video_agent.tools.skill_process import RECOVERY_CLASS, CliSkill  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_color_grading.py")


def fake_video(tmp: str, name: str = "clip.mp4", hdr: bool = False, pix_fmt: str = "") -> str:
    p = Path(tmp) / "src" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = {"fake": True, "duration": 16.0, "video": True, "channels": 2, "hdr": hdr, "lufs": -11.0}
    if pix_fmt:
        meta["pix_fmt"] = pix_fmt
    p.write_bytes(json.dumps(meta).encode())
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
        # 0.5.0: OCIO_TRANSFORM / MATCH_CAMERA / LEGALIZE are in the contract; the agent does not plan them (finishing.COLOR_OPERATIONS, ADR-097 addendum 6)
        self.assertEqual((ad.version, ad.tools, ad.drift(), sorted(ad.operations)),
                         ("1.1.0", {"color-grading/run"}, [], ["HDR_TO_SDR", "LEGALIZE", "LUT_APPLY", "MATCH_CAMERA", "OCIO_TRANSFORM", "PRIMARY_CORRECTION", "RETAG", "STRIP_DOVI"]))
        # color-grading-skill #23: WHITE_BALANCE is no longer an unsupported type but an `operation_aliases` entry (temperature / tint of
        # PRIMARY_CORRECTION); the agent lowers only declared operation types, so it keeps sending PRIMARY_CORRECTION itself
        self.assertEqual(ad.unsupported, []); self.assertEqual(ad.formats, ["m4v", "mkv", "mov", "mp4"])
        self.assertEqual([(a["type"], a["alias_of"]) for a in ad.contract["operation_aliases"]], [("WHITE_BALANCE", "PRIMARY_CORRECTION")])
        pk = ad.package()
        self.assertEqual((pk.skill_id, pk.version, pk.capabilities, pk.tool_ids(), pk.validate()), ("color-grading", "1.1.0", ["ffmpeg", "ffprobe", "ffmpeg-skill", "color-grading"], ["color-grading/run"], []))
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
        self.assertEqual(ad2.breaking_drift(), [], "a brand-new operation is additive capability, not a breaking change (ADR-045)")
        os.environ.pop("FAKE_CG_MODE")
        doc = ad.doctor()
        self.assertEqual((doc["status"], ad.operation_status(doc)["RETAG"], doc["exit_code"]), ("ok", "supported", 0))
        os.environ["FAKE_CG_MODE"] = "doctor_degraded"
        self.assertEqual(self._adapter().operation_status(self._adapter().doctor())["LUT_APPLY"], "unavailable")

    # ADR-045: purely additive drift (a version bump within the accepted range, an operation gaining optional
    # parameters, unsupported_operations shrinking) never blocks availability; an operation this adapter already
    # relies on disappearing, or a protocol-shape field changing, does.
    def test_breaking_drift_classification(self):
        from video_agent.tools.color_grading import contract_breaking_drift, pinned_contract
        pinned = pinned_contract()
        live = json.loads(json.dumps(pinned))
        live["version"] = "9.9.9"
        for op in live["operations"]:
            if op["type"] == "PRIMARY_CORRECTION":
                op["parameters"] = dict(op["parameters"], new_knob={"type": "number"})
        live["unsupported_operations"] = live["unsupported_operations"] + [{"type": "VIGNETTE", "status": "not_implemented", "reason": "x"}]
        self.assertEqual(contract_breaking_drift(live, pinned), [], "version bump + a new optional parameter + unsupported changing is non-fatal")
        # ADR-106: a changed alpha policy (here the pattern the agent judges an alpha plane by) is reported and fatal, like color_space / hdr_sdr
        live3 = json.loads(json.dumps(pinned))
        live3["alpha"]["pix_fmt_pattern"] = ".*"
        self.assertTrue(any(d.startswith("alpha:") for d in contract_drift(live3, pinned)))
        self.assertTrue(any(d.startswith("alpha:") for d in contract_breaking_drift(live3, pinned)))
        live4 = json.loads(json.dumps(pinned))
        live4["alpha"]["keep_alpha"]["operations"] = ["LUT_APPLY"]
        self.assertTrue(any(d.startswith("alpha:") for d in contract_breaking_drift(live4, pinned)))
        # ADR-106 addendum 1: the agent sends no tonemap and relies on the Skill's transfer-based rule, so a changed rule is fatal even with hdr_sdr unchanged
        live5 = json.loads(json.dumps(pinned))
        live5["hdr_sdr_defaults"]["tonemap"]["by_input_transfer"] = {"arib-std-b67": "hable"}
        self.assertTrue(any(d.startswith("hdr_sdr_defaults:") for d in contract_breaking_drift(live5, pinned)))
        live2 = json.loads(json.dumps(pinned))
        live2["operations"] = [o for o in live2["operations"] if o["type"] != "PRIMARY_CORRECTION"]
        self.assertTrue(contract_breaking_drift(live2, pinned), "an operation this adapter relies on disappearing must stay fatal")
        live3 = json.loads(json.dumps(pinned))
        live3["execution"]["shell"] = True
        self.assertTrue(contract_breaking_drift(live3, pinned), "a protocol-shape field changing must stay fatal")

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
        for bad, msg in (({"filter": "x"}, "forbidden"), ({"argv": ["x"]}, "forbidden"), ({"operation": "WHITE_BALANCE"}, "unknown operation"), ({"operation": "NOPE"}, "unknown operation"),
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

    def test_hdr_to_sdr_curve_is_the_skills_choice_and_is_recorded(self):
        """ADR-105: color-grading-skill 0.5.1 (its ADR-28) picks HDR_TO_SDR's curve when the request names none: mobius for an HLG input,
        hable for PQ or anything else. The agent sends no curve and keeps the Skill's tonemap_selection in the operation it reports."""
        rule = pinned_contract()["hdr_sdr_defaults"]["tonemap"]
        self.assertEqual((rule["by_input_transfer"], rule["otherwise"], rule["explicit_wins"]), ({"arib-std-b67": "mobius"}, "hable", True))
        hdr = next(o for o in pinned_contract()["operations"] if o["type"] == "HDR_TO_SDR")
        self.assertIsNone(hdr["parameters"]["tonemap"]["default"])
        ad = self._adapter()
        for name, transfer, curve, why in (("hlg.mp4", "arib-std-b67", "mobius", "hlg_default"), ("pq.mp4", "smpte2084", "hable", "default")):
            src = fake_video(self.tmp, name, hdr=True)
            meta = json.loads(Path(src).read_text()); meta["color_transfer"] = transfer; Path(src).write_text(json.dumps(meta))
            out = str(Path(self.ws) / "jobs" / "j1" / "ops" / "01_sdr" / name)
            r = ad.run(Operation(tool="color-grading/run", args={"operation": "HDR_TO_SDR", "input": "a", "output": "a_sdr"}, inputs=["a"], outputs=["a_sdr"], id="op_sdr"),
                       {"a": src, "a_sdr": out}, timeout=30)
            self.assertTrue(r.ok, r.data)
            self.assertEqual(r.data["operation"]["parameters"]["tonemap"], curve, name)
            self.assertEqual(r.data["operation"]["tonemap_selection"], {"tonemap": curve, "rule": why, "input_transfer": transfer}, name)

    def test_alpha_kept_by_lut_apply_and_primary_correction(self):
        """ADR-106: color-grading-skill 0.6.0 (its ADR-30) grades an alpha source with the alpha kept for the contract's keep_alpha operations
        (LUT_APPLY / PRIMARY_CORRECTION) where ffmpeg-skill/color has --keep-alpha and the container holds the alpha line. The agent takes that
        as a success only when the result's alpha says kept and verified and the output stores an alpha plane; the record is kept."""
        from video_agent.tools.color_grading.adapter import KEEP_ALPHA_OPERATIONS
        ka = pinned_contract()["alpha"]["keep_alpha"]
        self.assertEqual(tuple(ka["operations"]), KEEP_ALPHA_OPERATIONS)
        self.assertIn("ffmpeg-skill/color:keep-alpha", pinned_contract()["optional_capabilities"])
        src = fake_video(self.tmp, "alpha.mov", pix_fmt="yuva444p10le")
        ad = self._adapter()
        out = str(Path(self.ws) / "ops" / "01_correct" / "alpha_correct.mov")
        op = Operation(tool="color-grading/run", args={"operation": "PRIMARY_CORRECTION", "input": "a", "output": "a_correct", "exposure": 0.5}, inputs=["a"], outputs=["a_correct"], id="op_alpha")
        r = ad.run(op, {"a": src, "a_correct": out}, timeout=30)
        self.assertTrue(r.ok, r.data)
        self.assertEqual(r.data["alpha"], {"kept": True, "alpha_codec": "prores4444", "alpha_mode": "straight", "precision_bits": 8})
        self.assertTrue(r.data["operation"]["alpha"]["verification"]["ok"])
        self.assertEqual(r.data["artifact"]["pix_fmt"], "yuva444p12le")
        # ADR-107 addendum 5: 1.1.0's alpha_mode_selection is recorded as the Skill wrote it, not required, never a reason to refuse
        self.assertEqual(r.data["operation"]["alpha"]["alpha_mode_selection"]["rule"], "assumed")
        c = pinned_contract()
        self.assertEqual((c["version"], c["alpha"]["drop_alpha"]["request_field"], c["alpha"]["alpha_mode"]["when_omitted"]["otherwise"]),
                         ("1.1.0", "project.source.drop_alpha", "straight"), "a request without alpha_mode is graded as straight, as before")
        self.assertIn("ffmpeg-skill/color:drop-alpha", c["optional_capabilities"])
        # LUT_APPLY keeps it too (the other keep_alpha operation)
        lut = Path(self.tmp) / "src" / "look.cube"; lut.write_text("TITLE fake\n")
        lout = str(Path(self.ws) / "ops" / "01_lut" / "alpha_lut.mov")
        rl = ad.run(Operation(tool="color-grading/run", args={"operation": "LUT_APPLY", "input": "a", "output": "a_lut", "lut": str(lut)}, inputs=["a"], outputs=["a_lut"], id="op_alpha_lut"),
                    {"a": src, "a_lut": lout}, timeout=30)
        self.assertTrue(rl.ok, rl.data)
        self.assertEqual((rl.data["alpha"]["kept"], rl.data["operation"]["type"]), (True, "LUT_APPLY"))
        # the graded file keeps its alpha, so a following LEGALIZE (refused with keep_alpha) is refused on it, as on the source
        rg = ad.run(Operation(tool="color-grading/run", args={"operation": "LEGALIZE", "input": "a", "output": "a_legal"}, inputs=["a"], outputs=["a_legal"], id="op_alpha_legal"),
                    {"a": out, "a_legal": str(Path(self.ws) / "ops" / "02_legal" / "alpha_legal.mov")}, timeout=30)
        self.assertFalse(rg.ok)
        self.assertEqual((rg.data["error"]["details"]["reason"], rg.data["error"]["recovery_class"]), ("alpha_not_preserved", "UNSUPPORTED_MEDIA"))
        # an alpha record on a success that does not say kept + verified is not a success
        doc = {"schema": "color-grading/response@1", "skill": {"id": "color-grading", "version": ad.version}, "status": "ok", "dry_run": False, "plan": {"graph": {}},
               "results": [{"node_id": "op:edit", "status": "completed", "tool": "ffmpeg-skill/color", "type": "LUT_APPLY", "alpha": {"kept": False, "verification": {"ok": False}}}],
               "outputs": [{"output_id": "out", "status": "completed", "path": out, "artifact": {"pix_fmt": "yuva444p12le"}, "provenance": {}}]}
        self.assertTrue(any("not kept and verified" in e for e in ad._check_response(doc, out, False)))
        doc["results"][0]["alpha"] = {"kept": True, "verification": {"ok": "false"}}   # a verdict that is not the boolean true is no verdict
        self.assertTrue(any("not kept and verified" in e for e in ad._check_response(doc, out, False)))
        doc["results"][0]["alpha"] = {"kept": True, "verification": {"ok": True}}
        doc["outputs"][0]["artifact"]["pix_fmt"] = "yuv420p"
        self.assertTrue(any("stores no alpha plane" in e for e in ad._check_response(doc, out, False)))
        doc["results"][0]["type"] = "HDR_TO_SDR"
        self.assertTrue(any("does not keep alpha" in e for e in ad._check_response(doc, out, False)))
        # an ffmpeg-skill without --keep-alpha: the 0.5.0 refusal, as before
        os.environ["FAKE_CG_MODE"] = "no_keep_alpha"
        r = ad.run(op, {"a": src, "a_correct": out}, timeout=30)
        self.assertFalse(r.ok)
        self.assertEqual((r.data["error"]["details"]["reason"], r.data["error"]["recovery_class"]), ("alpha_not_preserved", "UNSUPPORTED_MEDIA"))

    def test_validation_reasons_map_to_their_recovery_classes(self):
        """ADR-106: 0.6.0's VALIDATION_ERROR reasons (ffmpeg-skill/color's own check of the written file) keep the Skill's code and details and get
        their own recovery class: legal_levels_failed -> LEGAL_LEVELS_FAILED, alpha_lost / alpha_mismatch / alpha_unverified -> ALPHA_NOT_KEPT.
        Both BLOCK (never retried); any other VALIDATION_ERROR keeps the shared mapping (SKILL_ERROR)."""
        from video_agent.execution.recovery import RECOVERY_TABLE
        policy = pinned_contract()["alpha"]["keep_alpha"]["verification"] + pinned_contract()["legalize"]["clamp"]["verification"]
        for reason, cls in (("legal_levels_failed", "LEGAL_LEVELS_FAILED"), ("alpha_lost", "ALPHA_NOT_KEPT"), ("alpha_mismatch", "ALPHA_NOT_KEPT"), ("alpha_unverified", "ALPHA_NOT_KEPT")):
            self.assertIn(reason, policy)
            self.assertEqual(RECOVERY_TABLE[cls]["action"], "BLOCK")
            os.environ["FAKE_CG_MODE"] = reason
            out = str(Path(self.ws) / "ops" / reason / "o.mp4")
            r = self._adapter().run(Operation(tool="color-grading/run", args={"operation": "LEGALIZE", "input": "a", "output": "o"}, inputs=["a"], outputs=["o"], id="op_" + reason),
                                    {"a": self.src, "o": out}, timeout=30)
            e = r.data["error"]
            self.assertEqual((r.ok, r.exit_code, e["code"], e["retryable"], e["details"]["reason"], e["recovery_class"]), (False, 11, "VALIDATION_ERROR", False, reason, cls))
            self.assertEqual(classify_error(r), cls)
            nxt = next_attempt(r, 1, 3, 30)
            self.assertEqual((nxt["action"], nxt["class"]), ("BLOCK", cls))
        os.environ["FAKE_CG_MODE"] = "validation_error"
        r = self._adapter().run(Operation(tool="color-grading/run", args={"operation": "RETAG", "input": "a", "output": "o", "target": "bt709"}, inputs=["a"], outputs=["o"], id="op_v"),
                                {"a": self.src, "o": str(Path(self.ws) / "ops" / "v" / "o.mp4")}, timeout=30)
        self.assertEqual(r.data["error"]["recovery_class"], RECOVERY_CLASS["VALIDATION_ERROR"])

    def test_alpha_source_refusal_blocks_with_its_own_recovery_class(self):
        """color-grading-skill 0.5.0 (its ADR-25): an alpha source is refused for every re-encoding operation with UNSUPPORTED_FORMAT /
        alpha_not_preserved (exit 6). The agent keeps the Skill's code, exit code and details, maps it to UNSUPPORTED_MEDIA (BLOCK, not
        INVALID_ARGS: the compiler did nothing wrong) and never retries it; RETAG, a stream copy, still runs on the same source (ADR-097 addendum 6)."""
        from video_agent.execution.recovery import RECOVERY_TABLE
        self.assertEqual(pinned_contract()["alpha"]["refused_for"], ["HDR_TO_SDR", "LUT_APPLY", "PRIMARY_CORRECTION", "OCIO_TRANSFORM", "MATCH_CAMERA", "LEGALIZE"])
        self.assertEqual(pinned_contract()["errors"]["exit_codes"]["UNSUPPORTED_FORMAT"], 6)
        self.assertEqual(RECOVERY_TABLE["UNSUPPORTED_MEDIA"]["action"], "BLOCK")
        src = fake_video(self.tmp, "alpha.mp4", pix_fmt="yuva444p10le")
        ad = self._adapter()
        out = str(Path(self.ws) / "ops" / "01_correct" / "alpha_correct.mp4")
        op = Operation(tool="color-grading/run", args={"operation": "PRIMARY_CORRECTION", "input": "a", "output": "a_correct", "exposure": 0.5}, inputs=["a"], outputs=["a_correct"], id="op_alpha")
        r = ad.run(op, {"a": src, "a_correct": out}, timeout=30)
        self.assertFalse(r.ok); self.assertIsNone(r.output)
        e = r.data["error"]
        self.assertEqual((r.exit_code, e["code"], e["retryable"], e["recovery_class"]), (6, "UNSUPPORTED_FORMAT", False, "UNSUPPORTED_MEDIA"))
        self.assertEqual((e["details"]["reason"], e["details"]["pix_fmt"]), ("alpha_not_preserved", "yuva444p10le"))
        self.assertEqual(classify_error(r), "UNSUPPORTED_MEDIA")
        nxt = next_attempt(r, 1, 3, 30)
        self.assertEqual((nxt["action"], nxt["class"]), ("BLOCK", "UNSUPPORTED_MEDIA"))
        self.assertIn("alpha", nxt["reason"])
        self.assertFalse(os.path.exists(out), "a refused call leaves no output behind")
        # any other UNSUPPORTED_FORMAT keeps the shared mapping (INVALID_ARGS)
        other = ad._fail(op, 1, False, 0.0, 6, "UNSUPPORTED_FORMAT", "container", False, {"reason": "container"})
        self.assertEqual(other.data["error"]["recovery_class"], RECOVERY_CLASS["UNSUPPORTED_FORMAT"])
        # RETAG keeps the alpha (stream copy): not refused
        ok = ad.run(self._op(), {"a": src, "a_retag": str(Path(self.ws) / "ops" / "02_retag" / "alpha_retag.mp4")}, timeout=30)
        self.assertTrue(ok.ok, ok.data)

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


# color-grading-skill 0.5.0 makes MATCH_CAMERA and LEGALIZE unsupported below this ffmpeg-skill release: its `color --lut` converts to RGB with
# BT.709 and back with BT.601 before it (capability ffmpeg-skill/color:lut-bt709-legs, read from the version; color-grading-skill ADR-23 / ADR-24)
LUT_LEGS_FIXED_IN = (3, 1, 6)


def engine_version(engine_dir: str) -> Tuple[int, ...]:
    """The located ffmpeg-skill's release, from its own static contract (as the CI log line prints it)."""
    out = subprocess.run([sys.executable, str(Path(engine_dir) / "scripts" / "_contract.py"), "--json", "--static"], capture_output=True, text=True, check=True).stdout
    return tuple(int(p) for p in json.loads(out)["skill"]["version"].split(".")[:3])


def expected_unavailable(engine_dir: str) -> List[str]:
    """What color-grading-skill 0.5.0's doctor must report unavailable here, worked out independently of the doctor: OCIO_TRANSFORM when
    the optional PyOpenColorIO extra is not importable by the Python the Skill runs on, MATCH_CAMERA / LEGALIZE when ffmpeg-skill is older
    than 3.1.6. Nothing else."""
    out = [] if importlib.util.find_spec("PyOpenColorIO") else ["OCIO_TRANSFORM"]
    if engine_version(engine_dir) < LUT_LEGS_FIXED_IN:
        out += ["LEGALIZE", "MATCH_CAMERA"]
    return sorted(out)


def assert_doctor_healthy(tc: unittest.TestCase, doc: dict, engine_dir: str) -> None:
    """The doctor is "ok" when nothing is expected unavailable, else "degraded" with exactly the expected operations unavailable and every
    other operation supported (ADR-100 plans the three; an unavailable one is BLOCKED by the plan). No problems either way; a doctor degraded for any
    other reason, or not degraded when it should be, fails."""
    tc.assertEqual(doc.get("problems") or [], [], doc.get("problems"))
    want = expected_unavailable(engine_dir)
    tc.assertEqual((doc["status"], sorted(doc.get("unavailable_operations") or [])), ("degraded" if want else "ok", want), doc.get("warnings"))
    statuses = ColorGradingAdapter.operation_status(doc)
    tc.assertEqual(sorted(t for t, s in statuses.items() if s != "supported"), want, statuses)
    caps = doc["checks"]["capabilities"]
    tc.assertEqual(caps["ffmpeg-skill/color:lut-bt709-legs"], "unsupported" if "LEGALIZE" in want else "supported")
    ocio = doc["checks"]["ocio"]
    tc.assertEqual((ocio["status"], ocio["optional"], ocio["needed_by"]), ("missing" if "OCIO_TRANSFORM" in want else "ok", True, ["OCIO_TRANSFORM"]))


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
        from pin_check import assert_live_matches_pin
        from video_agent.tools.color_grading import adapter as cg_adapter
        assert_live_matches_pin(self, "color-grading-skill", ad.contract, cg_adapter)
        doc = ad.doctor()
        assert_doctor_healthy(self, doc, self.engine)
        self.assertEqual(ad.operation_status(doc)["RETAG"], "supported")
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

    def test_real_hlg_tone_map_defaults_to_mobius(self):
        """ADR-105 on the real Skill (0.5.1, its ADR-28): an HLG BT.2020 source tone-mapped as the agent lowers it (no curve sent) runs mobius;
        the Skill's tonemap_selection says why and the engine command carries it. A PQ source keeps hable."""
        if not (self.skill and self.engine and shutil.which("ffmpeg")):
            self.skipTest("needs VIDEO_AGENT_COLOR_GRADING_DIR, VIDEO_AGENT_FFMPEG_SKILL_DIR and ffmpeg")
        ws = str(Path(self.tmp) / "ws_hlg"); os.makedirs(ws, exist_ok=True)
        ad = ColorGradingAdapter(self.skill, workspace=ws, allowed_inputs=[str(Path(self.tmp) / "src_hdr")], ffmpeg_skill_dir=self.engine, timeout=180)
        if ad.operation_status(ad.doctor())["HDR_TO_SDR"] != "supported":
            self.skipTest("HDR_TO_SDR is not supported by this ffmpeg build (zscale / tonemap), as the Skill's doctor reports")
        encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        if " libx265 " not in encoders:
            self.skipTest("the HLG / PQ fixtures are made with libx265, which this ffmpeg build lacks")
        for n, (transfer, curve, why) in enumerate((("arib-std-b67", "mobius", "hlg_default"), ("smpte2084", "hable", "default"))):
            src = str(Path(self.tmp) / "src_hdr" / f"hdr_{n}.mp4")
            os.makedirs(os.path.dirname(src), exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25", "-t", "1", "-vf", "format=yuv420p10le",
                            "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", f"colorprim=bt2020:transfer={transfer}:colormatrix=bt2020nc:log-level=error",
                            "-tag:v", "hvc1", src], check=True)
            out = str(Path(ws) / "ops" / f"0{n}_sdr" / f"hdr_{n}_sdr.mp4")
            r = ad.run(Operation(tool="color-grading/run", args={"operation": "HDR_TO_SDR", "input": "a", "output": "a_sdr"}, inputs=["a"], outputs=["a_sdr"], id=f"op_hlg{n}"),
                       {"a": src, "a_sdr": out}, timeout=180)
            self.assertTrue(r.ok, r.data)
            self.assertEqual(r.data["operation"]["parameters"]["tonemap"], curve, transfer)
            self.assertEqual(r.data["operation"]["tonemap_selection"], {"tonemap": curve, "rule": why, "input_transfer": transfer})
            self.assertIn(f"tonemap={curve}", " ".join(r.data["commands"]))
            self.assertFalse(r.data["artifact"]["hdr"])

    def test_real_new_operations_follow_the_doctor(self):
        """ADR-100 on the real Skill: LEGALIZE, MATCH_CAMERA (this clip against itself as the reference camera) and OCIO_TRANSFORM (built-in
        studio config, sRGB to Rec.709 display) as the agent lowers them. Each runs when the doctor reports it supported here and is refused
        when it reports it unavailable (PyOpenColorIO missing; ffmpeg-skill older than 3.1.6), which the plan BLOCKs before execution."""
        if not (self.skill and self.engine and shutil.which("ffmpeg")):
            self.skipTest("needs VIDEO_AGENT_COLOR_GRADING_DIR, VIDEO_AGENT_FFMPEG_SKILL_DIR and ffmpeg")
        ws = str(Path(self.tmp) / "ws_new"); os.makedirs(ws, exist_ok=True)
        ad = ColorGradingAdapter(self.skill, workspace=ws, allowed_inputs=[str(Path(self.src).parent)], ffmpeg_skill_dir=self.engine, timeout=180)
        doc = ad.doctor()
        statuses = ad.operation_status(doc)
        unavailable = expected_unavailable(self.engine)
        # the capability each one is missing when it is not usable here (the doctor reports such an operation `unsupported`, with what it misses)
        needs = {"OCIO_TRANSFORM": "python:PyOpenColorIO", "MATCH_CAMERA": "ffmpeg-skill/color:lut-bt709-legs", "LEGALIZE": "ffmpeg-skill/color:lut-bt709-legs"}
        ops = doc["checks"]["operations"]
        cases = {"LEGALIZE": {"profile": "bt709", "knee": 0.9},
                 "MATCH_CAMERA": {"reference_path": "a", "at": 1.0, "reference_at": 1.0, "model": "curves", "lut_size": 17},
                 "OCIO_TRANSFORM": {"config": "ocio://studio-config-latest", "source_space": "sRGB Encoded Rec.709 (sRGB)", "display": "Rec.1886 Rec.709 - Display",
                                    "view": "Un-tone-mapped", "lut_size": 33}}
        for n, (typ, params) in enumerate(sorted(cases.items())):
            out_id = f"a_{typ.lower()}"
            paths = {"a": self.src, out_id: str(Path(ws) / "ops" / f"0{n}_{typ.lower()}" / f"bars_{typ.lower()}.mp4")}
            r = ad.run(Operation(tool="color-grading/run", args={"operation": typ, "input": "a", "output": out_id, **params}, inputs=["a"], outputs=[out_id], id=f"op_new{n}"),
                       paths, timeout=180)
            if typ in unavailable:
                self.assertEqual(statuses[typ], "unsupported", ops[typ])
                self.assertIn(needs[typ], ops[typ]["missing"], ops[typ])
                self.assertFalse(r.ok, f"{typ} ran although the doctor reports it unsupported")
                continue
            self.assertEqual(statuses[typ], "supported")
            self.assertTrue(r.ok, (typ, r.data))
            self.assertEqual(r.data["artifact"]["sha256"], hashlib.sha256(Path(r.output).read_bytes()).hexdigest())
            self.assertEqual(r.data["operation"]["type"], typ)

    def test_real_alpha_source_kept_or_refused(self):
        """color-grading-skill 0.6.0 on real media, by what its doctor detects (`optional_capabilities`, ffmpeg-skill/color:keep-alpha):
        - supported (ffmpeg-skill 3.2.3+): PRIMARY_CORRECTION on a ProRes 4444 source with an alpha plane grades it with the alpha kept and verified;
          the agent takes it as a success (alpha kept, the output stores an alpha plane) and keeps the record (ADR-106);
        - otherwise (e.g. ffmpeg-skill 2.x): refused as in 0.5.0 (UNSUPPORTED_FORMAT, alpha_not_preserved, exit 6) before anything runs, blocked as
          UNSUPPORTED_MEDIA (ADR-097 addendum 6)."""
        if not (self.skill and self.engine and shutil.which("ffmpeg")):
            self.skipTest("needs VIDEO_AGENT_COLOR_GRADING_DIR, VIDEO_AGENT_FFMPEG_SKILL_DIR and ffmpeg")
        src = str(Path(self.tmp) / "src_alpha" / "key.mov")
        os.makedirs(os.path.dirname(src), exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red@0.5:size=320x180:rate=25,format=yuva444p10le", "-t", "1",
                        "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", src], check=True)
        ws = str(Path(self.tmp) / "ws_alpha"); os.makedirs(ws, exist_ok=True)
        ad = ColorGradingAdapter(self.skill, workspace=ws, allowed_inputs=[str(Path(src).parent)], ffmpeg_skill_dir=self.engine, timeout=120)
        out = str(Path(ws) / "ops" / "01_correct" / "key_correct.mov")
        r = ad.run(Operation(tool="color-grading/run", args={"operation": "PRIMARY_CORRECTION", "input": "a", "output": "a_correct", "format": "mov", "exposure": 0.5},
                             inputs=["a"], outputs=["a_correct"], id="op_real_alpha"), {"a": src, "a_correct": out}, timeout=120)
        keep = ((ad.doctor().get("checks") or {}).get("optional_capabilities") or {}).get("ffmpeg-skill/color:keep-alpha") or {}
        if keep.get("status") == "supported":
            self.assertTrue(r.ok, r.data)
            self.assertEqual((r.data["alpha"]["kept"], r.data["alpha"]["alpha_codec"], r.data["alpha"]["precision_bits"]), (True, "prores4444", 8))
            self.assertTrue(r.data["operation"]["alpha"]["verification"]["ok"])
            pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", out], capture_output=True, text=True, check=True)
            self.assertTrue(pr.stdout.strip().startswith("yuva"), pr.stdout)
            return
        self.assertFalse(r.ok, r.data)
        e = r.data["error"]
        self.assertEqual((r.exit_code, e["code"], e["retryable"], e["details"].get("reason"), e["recovery_class"]), (6, "UNSUPPORTED_FORMAT", False, "alpha_not_preserved", "UNSUPPORTED_MEDIA"))
        self.assertEqual(next_attempt(r, 1, 3, 120)["action"], "BLOCK")
        self.assertFalse(os.path.exists(out))

    def test_real_contract_vocabulary_every_parameter(self):
        """ADR-076 on the real color-grading-skill: the live contract's PRIMARY_CORRECTION declares exactly the colour parameters the agent's
        `color.*` keys are read from (names, types, ranges, enums, defaults), and one request planned from every key at once (each value
        checked against the pinned contract at plan time, then lowered by the compiler) runs; the Skill reports every value back as applied."""
        if not (self.skill and self.engine and shutil.which("ffmpeg")):
            self.skipTest("needs VIDEO_AGENT_COLOR_GRADING_DIR, VIDEO_AGENT_FFMPEG_SKILL_DIR and ffmpeg")
        from video_agent.agent import finishing
        from video_agent.agent.requirements import requirement_map
        from video_agent.execution.compiler import lower_color_op
        from video_agent.models import Requirement
        ws = str(Path(self.tmp) / "ws3"); os.makedirs(ws, exist_ok=True)
        ad = ColorGradingAdapter(self.skill, workspace=ws, allowed_inputs=[str(Path(self.src).parent)], ffmpeg_skill_dir=self.engine, timeout=120)
        live = ad.operations["PRIMARY_CORRECTION"]["parameters"]   # the live contract, fetched and checked by the adapter
        self.assertEqual({n: s for n, s in live.items() if n not in finishing.PRIMARY_CORRECTION_NOT_GRADING}, finishing.COLOR_CORRECTION_PARAMS,
                         "the live PRIMARY_CORRECTION colour parameters drifted from the pinned contract the agent's keys are read from: re-pin")
        want = {"exposure": 0.2, "contrast": 1.1, "saturation": 0.9, "temperature": 5600.0, "tint": 0.1, "gamma": 1.2, "lift": 0.05, "gain": -0.05,
                "levels_in_black": 16, "levels_in_white": 235, "levels_out_black": 4, "levels_out_white": 250, "curves": "vintage"}
        self.assertEqual(sorted(want), sorted(finishing.COLOR_CORRECTION_PARAMS), "every key of the vocabulary is exercised")
        planned = finishing.parse_color_requirements(requirement_map([Requirement(key=f"color.{k}", value=v, provenance="USER", source="cli") for k, v in want.items()]))
        args = lower_color_op("color-grading/run", dict(planned["color.primary_correction"]["params"], type="color.primary_correction"), "a", "a_correct")
        paths = {"a": self.src, "a_correct": str(Path(ws) / "ops" / "01_correct" / "bars_correct.mp4")}
        r = ad.run(Operation(tool="color-grading/run", args=args, inputs=["a"], outputs=["a_correct"], id="op_real4"), paths, timeout=120)
        self.assertTrue(r.ok, r.data)
        self.assertEqual(r.data["artifact"]["sha256"], hashlib.sha256(Path(r.output).read_bytes()).hexdigest())
        applied = r.data["operation"]["parameters"]
        self.assertEqual({k: applied[k] for k in want}, want)


if __name__ == "__main__":
    unittest.main()
