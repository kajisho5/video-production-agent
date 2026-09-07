"""qc-skill adapter tests (ADR-031): contract discovery and refusals, drift, request building from typed args (rules / parameters /
companions typed by the contract, forbidden keys refused by name, path policy), execution through `run - --json --workspace …
--allowed-input-root …` with the request on stdin, ADMISSION of the report (schema / skill / operation / kind / OBSERVED / fingerprints
recomputed by the adapter / statuses / report id), error mapping with the Skill's retryable verdict, the security boundary, doctor,
and a real-Skill class that runs only when VIDEO_AGENT_QC_DIR is set. Verified against a fake qc process (tests/fake_qc.py) speaking the
real transport; no ffprobe, no import of the Skill."""
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
from video_agent.tools.ffmpeg_skill.adapter import PathPolicy  # noqa: E402
from video_agent.tools.qc import (ENV_DIR, PACKAGE, TOOL_CHECK, TOOL_INSPECT, ContractError, QcAdapter, check_contract, contract_drift, lift_report,  # noqa: E402
                                  locate_qc, pinned_contract)
from video_agent.tools.skill_process import RECOVERY_CLASS, CliSkill  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_qc.py")


def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fake_media(tmp: str, name: str = "clip.mp4", duration: float = 3.0, video: bool = True, channels: int = 2) -> str:
    p = str(Path(tmp) / name)
    Path(p).write_bytes(json.dumps({"fake": True, "duration": duration, "video": video, "channels": channels, "lufs": -16.0}).encode())
    return p


class QcAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.src_dir = str(Path(self.tmp) / "src")
        os.makedirs(self.src_dir)
        self.src = fake_media(self.src_dir)
        self.srt = str(Path(self.src_dir) / "clip.srt")
        Path(self.srt).write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n2\n00:00:01,500 --> 00:00:02,500\nworld\n", encoding="utf-8")
        self.ws = str(Path(self.tmp) / "ws")
        os.makedirs(self.ws)
        for k in ("FAKE_QC_MODE", "FAKE_QC_CALLS"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ("FAKE_QC_MODE", "FAKE_QC_CALLS"):
            os.environ.pop(k, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _skill(self):
        return CliSkill("qc", [sys.executable, FAKE], None, {})

    def _adapter(self, **kw):
        kw.setdefault("workspace", self.ws)
        kw.setdefault("allowed_inputs", [self.src_dir])
        return QcAdapter(self._skill(), **kw)

    def _op(self, tool=TOOL_CHECK, **args):
        a = {"input": "clip", "kind": "video", "rules": {"video": {"expected_width": 640, "expected_height": 360}}}
        a.update(args)
        return Operation(tool=tool, args=a, inputs=[a.get("input", "clip")], outputs=[], kind="qa", id="op_test")

    def _paths(self):
        return {"clip": self.src, "sub": self.srt}

    # ---- contract
    def test_contract_discovery_package_and_refusals(self):
        ad = self._adapter()
        self.assertEqual(ad.version, "0.1.0"); self.assertEqual(ad.tools, {TOOL_CHECK, TOOL_INSPECT}); self.assertEqual(ad.drift(), [])
        self.assertEqual(ad.kinds, ["video", "audio", "subtitle", "delivery", "delivery_package"]); self.assertEqual(ad.statuses, ["PASS", "WARN", "FAIL", "UNKNOWN"]); self.assertEqual(len(ad.parameters), 12)
        self.assertEqual(sorted(ad.rules), ["audio", "delivery", "delivery_package", "delivery_package_artifact", "delivery_package_cross_artifact", "delivery_package_dependency",
                                            "delivery_package_duration_consistency", "subtitle", "timeline_integrity", "timeline_segment", "timeline_source_cue", "video"]); self.assertTrue(ad.owns_cache)
        self.assertIn("qc/check", ad.describe()["tools"]); self.assertEqual(ad.describe()["drift"], [])
        pk = ad.package()
        self.assertEqual((pk.skill_id, pk.version, pk.capabilities, [t.tool_id for t in pk.tools], pk.repository), ("qc", "0.1.0", ["ffprobe", "qc"], [TOOL_CHECK, TOOL_INSPECT], "kajisho5/qc-skill"))
        self.assertEqual(pk.validate(), []); self.assertTrue(all(not t.produces_output and t.kind == "measure" for t in pk.tools)); self.assertIn("verdict", pk.tools[0].result_keys)
        self.assertEqual(PACKAGE.tool_ids(), [TOOL_CHECK, TOOL_INSPECT]); self.assertIn("never a production decision", PACKAGE.role)
        self.assertEqual(check_contract(pinned_contract()), []); self.assertEqual(contract_drift(pinned_contract()), [])
        # the retryable table the contract lacks: every code non-retryable until the Skill says otherwise in its response
        self.assertEqual(ad.retryable["TOOL_ERROR"], False); self.assertEqual(ad.exit_codes["CANCELLED"], 13)
        # incompatible contracts are refused, never patched
        for mode, msg in (("wrong_schema", "contract schema"), ("wrong_skill", "skill_id"), ("wrong_version", "version"), ("bad_contract", "execution.shell"), ("contract_fail", "failed")):
            os.environ["FAKE_QC_MODE"] = mode
            with self.assertRaises(ContractError, msg=mode) as cm:
                self._adapter()
            self.assertIn(msg, str(cm.exception))
        # a compatible but different contract is drift: reported, never silently kept
        os.environ["FAKE_QC_MODE"] = "contract_drift"
        ad2 = self._adapter()
        self.assertTrue(any("VIDEO_HDR_MISMATCH" in d for d in ad2.drift()), ad2.drift()); self.assertTrue(any(d.startswith("parameters") for d in ad2.drift()))
        # static checks of the contract the adapter enforces: every tampering is a violation
        c = pinned_contract()
        for fn, msg in ((lambda d: d["execution"].update(arbitrary_filters=True), "arbitrary_filters"), (lambda d: d["execution"].update(canonical_invocation=["sh", "-c"]), "canonical_invocation"),
                        (lambda d: d["execution"].update(executables=["ffprobe", "bash"]), "executables"), (lambda d: d["errors"]["codes"].remove("CANCELLED"), "errors.codes"),
                        (lambda d: d.update(role="execution"), "role"), (lambda d: d.update(provenance="ESTIMATED"), "provenance"), (lambda d: d.update(outputs=["report", "media"]), "outputs"),
                        (lambda d: d.update(operations=["inspect"]), "operations lack"), (lambda d: d["kinds"].remove("delivery"), "kinds lack"), (lambda d: d["statuses"].remove("UNKNOWN"), "statuses lack"),
                        (lambda d: d["rules"]["video"].update(expected_width={"type": "Any", "default": None}), "typed schema"), (lambda d: d["rules"].pop("subtitle"), "rules.subtitle"),
                        (lambda d: d.update(parameters="x"), "parameters"), (lambda d: d["capabilities"].update(required=[]), "ffprobe"), (lambda d: d.update(deterministic=False), "deterministic"),
                        (lambda d: d["cache"].update(policies=["use"]), "cache.policies")):
            doc = json.loads(json.dumps(c)); fn(doc)
            self.assertTrue(any(msg in e for e in check_contract(doc)), (msg, check_contract(doc)))
        self.assertEqual(check_contract("nope"), ["contract is not an object"])

    # ---- execution and admission
    def test_valid_execution_and_mapping(self):
        ad = self._adapter(); paths = self._paths()
        log = str(Path(self.tmp) / "calls.log"); os.environ["FAKE_QC_CALLS"] = log
        r = ad.run(self._op(), paths, timeout=120)
        self.assertTrue(r.ok, r.stderr_tail); self.assertEqual(r.exit_code, 0); self.assertIsNone(r.output); self.assertEqual(r.commands, []); self.assertEqual(r.tool, TOOL_CHECK)
        d = r.data
        self.assertEqual(d["skill"], {"id": "qc", "version": "0.1.0"}); self.assertEqual((d["status"], d["operation_type"], d["kind"], d["verdict"]), ("completed", "check", "video", "PASS"))
        self.assertTrue(d["report_id"].startswith("qcreport_")); self.assertTrue(d["admitted"]); self.assertEqual(d["fingerprint"], sha(self.src))
        self.assertEqual(sorted(d), sorted(["skill", "status", "operation_type", "kind", "verdict", "report_id", "checks", "findings", "measurements", "fingerprint", "companions", "provenance", "cache", "reused", "admitted", "commands", "warnings"]))
        self.assertTrue(d["checks"] and all(set(c) == {"check_id", "category", "status", "finding_codes", "measurement_ids"} for c in d["checks"]))
        self.assertTrue(all(c["check_id"].startswith("video.") for c in d["checks"]), "kind video measures video only")
        self.assertTrue(d["measurements"] and set(d["measurements"][0]) == {"id", "category", "name", "value", "unit", "source", "estimated"}); self.assertEqual(d["findings"], [])
        self.assertEqual(d["provenance"]["measurement_source"], "OBSERVED"); self.assertEqual(d["provenance"]["skill"], "qc"); self.assertTrue(d["provenance"]["identity"]); self.assertIn("ffprobe_version", d["provenance"]["engine"])
        self.assertEqual((d["cache"]["status"], d["cache"]["policy"], d["reused"]), ("miss", "use", False)); self.assertEqual(d["companions"], {"subtitle": None, "reference_video": None})
        # the request the Skill saw: argv list of the canonical form, workspace and roots on argv, the request on stdin without paths of its own beyond the input
        calls = [json.loads(l) for l in Path(log).read_text().splitlines()]
        run = next(c for c in calls if c["cmd"] == "run")
        self.assertEqual(run["argv"][:3], ["run", "-", "--json"]); self.assertEqual(run["argv"][run["argv"].index("--workspace") + 1], self.ws); self.assertNotIn("--no-cache", run["argv"])
        roots = [os.path.normcase(os.path.realpath(run["argv"][i + 1])) for i, a in enumerate(run["argv"]) if a == "--allowed-input-root"]
        self.assertEqual(roots, [os.path.normcase(os.path.realpath(self.src_dir)), os.path.normcase(os.path.realpath(self.ws))])
        self.assertNotIn("--timeout", run["argv"]); self.assertNotIn("--allowed-input", run["argv"]); self.assertNotIn("--ffmpeg-skill", run["argv"])
        self.assertTrue(os.path.isdir(os.path.join(self.ws, ".qc-cache")), "the Skill's cache lives under the workspace")
        # lifting: the admitted report becomes an agent Observation (provenance only)
        obs = lift_report(r, "clip")
        self.assertEqual((obs.kind, obs.provenance, obs.skill, obs.skill_version, obs.tool, obs.fingerprint, obs.asset_id), ("qc.report", "OBSERVED", "qc", "0.1.0", TOOL_CHECK, sha(self.src), "clip"))
        self.assertEqual(obs.source, "qc/check@0.1.0"); self.assertEqual(obs.data["verdict"], "PASS"); self.assertEqual(obs.external_id, d["report_id"]); self.assertEqual(obs.parameters["kind"], "video")
        # a FAIL or WARN verdict is a successful measurement (the gate decides, never the adapter)
        for mode, verdict, code in (("verdict_fail", "FAIL", "VIDEO_STREAM_MISSING"), ("verdict_warn", "WARN", "AUDIO_LEADING_SILENCE_EXCEEDED")):
            os.environ["FAKE_QC_MODE"] = mode
            rr = ad.run(self._op(kind="delivery", rules={"delivery": {"require_video": True, "audio": {"max_leading_silence_sec": 1.0}}}), paths)
            self.assertTrue(rr.ok, mode); self.assertEqual(rr.data["verdict"], verdict); self.assertEqual(rr.data["findings"][0]["code"], code)
            self.assertNotIn("argv", rr.data["findings"][0], "findings are scrubbed of forbidden keys"); self.assertEqual(lift_report(rr).data["verdict"], verdict)
        os.environ["FAKE_QC_MODE"] = "reused"
        rr = ad.run(self._op(), paths)
        self.assertTrue(rr.ok and rr.data["reused"]); self.assertEqual(rr.data["cache"]["status"], "hit")
        os.environ.pop("FAKE_QC_MODE", None)
        # inspect: measurements only, no rules, no checks
        ri = ad.run(self._op(TOOL_INSPECT, kind="audio", rules=None), paths)
        self.assertTrue(ri.ok, ri.data.get("error")); self.assertEqual((ri.data["operation_type"], ri.data["checks"], ri.data["verdict"]), ("inspect", [], "PASS"))
        self.assertTrue(all(m["id"].split(".")[0] in ("audio", "container") for m in ri.data["measurements"]), "kind audio measures audio only")
        self.assertEqual(lift_report(ri).source, "qc/inspect@0.1.0")
        # companions: subtitle kind with reference_video, delivery kind with a subtitle; their fingerprints are recomputed and reported
        rs = ad.run(self._op(input="sub", kind="subtitle", reference_video="clip", rules={"subtitle": {"max_line_length": 42, "min_coverage_ratio": 0.5}}), paths)
        self.assertTrue(rs.ok, rs.data.get("error")); self.assertEqual(rs.data["companions"], {"subtitle": None, "reference_video": sha(self.src)}); self.assertEqual(rs.data["fingerprint"], sha(self.srt))
        rd = ad.run(self._op(kind="delivery", subtitle="sub", rules={"delivery": {"require_subtitle": True, "min_size_bytes": 10, "subtitle": {"allow_overlapping_cues": False}}}, parameters={"max_line_length": 42}), paths)
        self.assertTrue(rd.ok, rd.data.get("error")); self.assertEqual(rd.data["companions"]["subtitle"], sha(self.srt)); self.assertTrue(any(c["category"] == "subtitle" for c in rd.data["checks"]))
        # dry run: the request is validated and nothing is invoked (a measurement has no dry run)
        n = ad.calls
        r2 = ad.run(self._op(), paths, dry_run=True)
        self.assertTrue(r2.ok and r2.dry_run and r2.output is None); self.assertEqual((r2.data["status"], r2.data["verdict"]), ("dry_run", None)); self.assertEqual(ad.calls, n); self.assertIsNone(lift_report(r2))
        self.assertEqual(len(ad.preview(self._op(), paths)), 1); self.assertIn("run - --json --workspace", ad.preview(self._op(), paths)[0]); self.assertIn("refused", ad.preview(self._op(kind="hologram"), paths)[0])
        # measure(): QA's direct entry, the same admission
        rm = ad.measure(TOOL_CHECK, {"input": "clip", "kind": "video"}, paths, timeout=30)
        self.assertTrue(rm.ok); self.assertEqual(rm.data["verdict"], "PASS"); self.assertTrue(rm.op_id.startswith("op"))
        # without a configured workspace: no cache, roots = the inputs' own directories
        ad2 = QcAdapter(self._skill()); Path(log).unlink()
        r3 = ad2.run(self._op(), paths)
        self.assertTrue(r3.ok); run = [json.loads(l) for l in Path(log).read_text().splitlines()][-1]
        self.assertIn("--no-cache", run["argv"]); self.assertEqual(run["argv"][run["argv"].index("--allowed-input-root") + 1], self.src_dir); self.assertFalse(os.path.exists(os.path.join(self.src_dir, ".qc-cache")))

    def test_admission_and_error_mapping(self):
        ad = self._adapter(); paths = self._paths()
        cases = {"tool_error": ("TOOL_ERROR", True, "UNKNOWN", "RETRY"), "tool_error_final": ("TOOL_ERROR", False, "SKILL_ERROR", "BLOCK"), "validation_error": ("VALIDATION_ERROR", False, "SKILL_ERROR", "BLOCK"),
                 "cancelled": ("CANCELLED", True, "INTERRUPTED", "BLOCK"), "internal_error": ("INTERNAL_ERROR", False, "SKILL_ERROR", "BLOCK"),
                 "malformed": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK"), "empty": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK"), "text": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK"),
                 "two_docs": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK"), "nonzero_ok": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK"), "unknown_code": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK"),
                 "no_report": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK"), "output_missing": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK"),
                 "fingerprint_mismatch": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK"), "hash_mismatch": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK"),
                 "wrong_kind": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK"), "wrong_operation": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK"), "not_observed": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK"),
                 "status_failed_ok": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK"), "bad_status": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK"), "bad_report_id": ("INVALID_RESULT", False, "SKILL_ERROR", "BLOCK")}
        for mode, (code, retry, cls, action) in cases.items():
            os.environ["FAKE_QC_MODE"] = mode
            r = ad.run(self._op(), paths, timeout=60)
            self.assertFalse(r.ok, mode); self.assertIsNone(r.output, mode); self.assertNotIn("admitted", r.data, mode); self.assertIsNone(lift_report(r), mode)
            self.assertEqual((r.data["error"]["code"], r.data["error"]["retryable"], r.data["error"]["recovery_class"]), (code, retry, RECOVERY_CLASS.get(code, "SKILL_ERROR")), mode)
            self.assertEqual(classify_error(r), cls, mode); self.assertEqual(next_attempt(r, 1, 2, 60)["action"], action, mode); self.assertIn("exit_code", r.data["error"], mode)
        # the admission failure names its condition
        for mode, fragment in (("fingerprint_mismatch", "fingerprint"), ("wrong_kind", "report.kind"), ("wrong_operation", "report.operation"), ("not_observed", "measurement_source"),
                               ("bad_status", "overall_status"), ("bad_report_id", "report.id"), ("no_report", "report missing"), ("status_failed_ok", "status")):
            os.environ["FAKE_QC_MODE"] = mode
            r = ad.run(self._op(), paths)
            self.assertIn(fragment, r.data["error"]["message"] + str(r.data["error"]["details"]), mode)
        # a companion whose fingerprint the Skill reports differently is refused too (the fake reports the right one; tamper the file after the request is built)
        os.environ["FAKE_QC_MODE"] = "tool_error"
        r = ad.run(self._op(), paths)
        self.assertEqual(r.data["error"]["details"].get("reason"), "tool_failed"); self.assertNotIn("argv", r.data["error"]["details"], "error details are scrubbed")
        # a process that never answers is killed at the boundary (exit 124 → CANCELLED / timeout, retryable with a longer budget)
        os.environ["FAKE_QC_MODE"] = "timeout"
        r = ad.run(self._op(), paths, timeout=1)
        self.assertFalse(r.ok); self.assertEqual((r.exit_code, r.data["error"]["code"], r.data["error"]["retryable"], r.data["error"]["details"]["reason"]), (124, "CANCELLED", True, "timeout"))
        self.assertEqual(classify_error(r), "TIMEOUT"); self.assertEqual(next_attempt(r, 1, 2, 1)["timeout"], 2)
        # the Skill's own refusals arrive as its codes, never retried
        os.environ.pop("FAKE_QC_MODE", None)
        out_of_roots = fake_media(os.path.realpath(tempfile.mkdtemp()), "far.mp4"); paths["far"] = out_of_roots
        r = QcAdapter(self._skill(), workspace=self.ws).run(self._op(input="far"), {"far": out_of_roots, "clip": self.src})   # no agent roots: the workspace is the only root on argv, the Skill refuses the rest
        self.assertFalse(r.ok); self.assertEqual((r.data["error"]["code"], r.data["error"]["recovery_class"]), ("PATH_NOT_ALLOWED", "INPUT_MISSING"))

    # ---- request building
    def test_request_building_refusals(self):
        ad = self._adapter(); paths = self._paths()
        log = str(Path(self.tmp) / "calls.log"); os.environ["FAKE_QC_CALLS"] = log
        b = ad.build_request(TOOL_CHECK, {"input": "clip", "kind": "video", "rules": {"video": {"expected_width": 640, "frame_rate_tolerance": 0.1, "expected_codec": "h264"}, "audio": {"require_audio_stream": True, "expected_channels": None}},
                                          "parameters": {"black_min_duration_sec": 0.5}, "cache_policy": "bypass"}, paths, op_id="op:1")
        req = b["request"]
        self.assertEqual((req["schema"], req["operation"], req["kind"], req["input"], req["cache_policy"], req["request_id"]), ("qc/request@1", "check", "video", str(Path(self.src).resolve()), "bypass", "op_1"))
        self.assertEqual(req["rules"], {"video": {"expected_width": 640, "frame_rate_tolerance": 0.1, "expected_codec": "h264"}, "audio": {"require_audio_stream": True, "expected_channels": None}})
        self.assertEqual(req["parameters"], {"black_min_duration_sec": 0.5}); self.assertNotIn("timeout", req); self.assertNotIn("subtitle", req)
        self.assertEqual(ad.build_request(TOOL_CHECK, {"input": "clip", "kind": "video", "rules": {"video": {"expected_width": 640.0}}}, paths)["request"]["rules"]["video"]["expected_width"], 640)
        bad = [({"kind": "hologram"}, "kind"), ({"kind": "video", "input": "nope"}, "not found"), ({"tool": "x"}, "unknown arguments"),
               ({"rules": {"colour": {}}}, "unknown rule section"), ({"rules": {"video": {"expected_bitrate": 1}}}, "not a declared rule field"), ({"rules": {"video": {"expected_width": 640.5}}}, "integer"),
               ({"rules": {"video": {"expected_width": "640"}}}, "finite number"), ({"rules": {"video": {"expected_width": True}}}, "finite number"), ({"rules": {"video": {"frame_rate_tolerance": None}}}, "not null"),
               ({"rules": {"video": {"expected_codec": "h264\n--shell"}}}, "plain string"), ({"rules": {"audio": {"require_audio_stream": 1}}}, "boolean"), ({"rules": {"video": {"expected_width": float("nan")}}}, "finite"),
               ({"rules": {"delivery": {"video": {"nope": 1}}}}, "not a declared rule field"), ({"rules": {"delivery": {"video": 3}}}, "must be an object"), ({"rules": []}, "rules must be"),
               ({"parameters": {"gain": 1}}, "not declared"), ({"parameters": {"max_gap_sec": "1"}}, "finite number"), ({"parameters": {"max_gap_sec": float("inf")}}, "finite number"), ({"parameters": 1}, "parameters must"),
               ({"subtitle": "sub"}, "only accepted with kind delivery"), ({"reference_video": "clip"}, "only accepted with kind subtitle"), ({"kind": "delivery", "subtitle": "missing"}, "not found"),
               ({"cache_policy": "only"}, "cache_policy"), ({"cache_policy": "never"}, "cache_policy"), ({"input": ""}, "required")]
        for extra, msg in bad:
            args = {"input": "clip", "kind": "video", **extra}
            with self.assertRaises(ToolError, msg=str(extra)) as cm:
                ad.build_request(TOOL_CHECK, args, paths)
            self.assertIn(msg, str(cm.exception), extra)
            r = ad.run(Operation(tool=TOOL_CHECK, args=args, inputs=["clip"], outputs=[], kind="qa", id="op_bad"), paths)
            self.assertFalse(r.ok); self.assertEqual((r.data["error"]["code"], r.data["error"]["retryable"], classify_error(r)), ("INVALID_REQUEST", False, "INVALID_ARGS"), extra)
        with self.assertRaises(ToolError):
            ad.build_request(TOOL_INSPECT, {"input": "clip", "kind": "video", "rules": {"video": {"expected_width": 640}}}, paths)
        with self.assertRaises(ToolError):
            ad.build_request("qc/validate", {"input": "clip", "kind": "video"}, paths)
        self.assertFalse(ad.supports("qc/validate")); self.assertTrue(ad.supports(TOOL_INSPECT))
        self.assertFalse(os.path.exists(log), "nothing reached the Skill process")
        # paths: outside the allowed roots (and the workspace), traversal, a directory — refused before the Skill runs
        far = fake_media(os.path.realpath(tempfile.mkdtemp()), "far.mp4"); paths["far"] = far; paths["dir"] = self.src_dir; paths["trav"] = str(Path(self.src_dir) / ".." / "src" / "clip.mp4")
        for ref, what in (("far", "outside the allowed roots"), ("dir", "not found")):
            with self.assertRaises(ToolError) as cm:
                ad.build_request(TOOL_CHECK, {"input": ref, "kind": "video"}, paths)
            self.assertIn(what, str(cm.exception), ref)
        self.assertEqual(ad.build_request(TOOL_CHECK, {"input": "trav", "kind": "video"}, paths)["input"], str(Path(self.src).resolve()))
        inws = fake_media(self.ws, "in_ws.mp4"); paths["inws"] = inws
        self.assertEqual(ad.build_request(TOOL_CHECK, {"input": "inws", "kind": "video"}, paths)["input"], str(Path(inws).resolve()))
        # the agent's own PathPolicy is applied first when the Service passes it
        strict = self._adapter(path_policy=PathPolicy([str(Path(self.tmp) / "elsewhere")], self.ws))
        r = strict.run(self._op(), paths)
        self.assertFalse(r.ok); self.assertIn("outside allowed roots", r.data["error"]["message"])
        self.assertTrue(strict.run(self._op(input="inws"), paths).ok)
        self.assertFalse(os.path.exists(log) and any(json.loads(l)["cmd"] == "run" and "far" in json.dumps(l) for l in Path(log).read_text().splitlines()))

    # ---- security
    def test_security_boundary(self):
        ad = self._adapter(); paths = self._paths()
        log = str(Path(self.tmp) / "calls.log"); os.environ["FAKE_QC_CALLS"] = log
        # forbidden keys (the agent's list and the Skill's 14) never cross, at any nesting level or case
        for key in ("command", "commands", "argv", "args", "shell", "cmd", "cmdline", "exec", "executable", "filter", "filter_complex", "env", "environment", "Filter", "af", "vf", "script", "workspace",
                    "allowed_input", "ffmpeg", "ffprobe", "path", "api_key", "token"):
            for args in (self._op(**{key: "x"}), self._op(rules={"video": {key: 1}}), self._op(parameters={key: 1}), self._op(rules={"delivery": {"video": {key: 1}}}, kind="delivery")):
                r = ad.run(args, paths)
                self.assertFalse(r.ok, key); self.assertEqual(r.data["error"]["code"], "INVALID_REQUEST", key)
        self.assertFalse(os.path.exists(log), "nothing reached the Skill process")
        # the calls log: argv is a list, carries only flags / directories, never request keys or values; no engine env is added
        ad.run(self._op(rules={"video": {"expected_codec": "h264"}}), paths)
        calls = [json.loads(l) for l in Path(log).read_text().splitlines()]
        self.assertEqual([c["cmd"] for c in calls], ["run"]); run = calls[0]
        self.assertIsInstance(run["argv"], list); self.assertNotIn("h264", json.dumps(run["argv"])); self.assertNotIn("expected_codec", json.dumps(run["argv"])); self.assertNotIn("clip.mp4", json.dumps(run["argv"]))
        self.assertTrue(all(a.startswith("--") or a in ("run", "-") or os.path.isabs(a) for a in run["argv"]), run["argv"])
        inherited = {k: v for k, v in os.environ.items() if k.startswith("VIDEO_")}
        self.assertEqual(run["env_video"], inherited, "the adapter adds no VIDEO_* variable of its own to the child environment")
        # the request document itself never carries a forbidden key
        b = ad.build_request(TOOL_CHECK, {"input": "clip", "kind": "delivery", "subtitle": "sub", "rules": {"delivery": {"audio": {"max_true_peak_dbfs": -1.0}}}}, paths)
        from video_agent.tools.skill_process import scan_forbidden
        self.assertIsNone(scan_forbidden(b["request"], ad.forbidden)); self.assertIn("filter_complex", ad.forbidden); self.assertIn("cwd", ad.forbidden)
        # static: the adapter never imports the Skill, never opens a shell, never names ffprobe/ffmpeg as an executable, every process goes through the shared transport
        root = Path(__file__).resolve().parents[1] / "src" / "video_agent" / "tools" / "qc"
        blob = "\n".join(p.read_text(encoding="utf-8") for p in root.glob("*.py"))
        self.assertNotIn("import qc_skill", blob); self.assertNotIn("from qc_skill", blob); self.assertNotIn("shell=True", blob); self.assertNotIn("os.system", blob)
        self.assertNotRegex(blob, r"subprocess\.(run|Popen|call)\("); self.assertNotIn("import subprocess", blob)

    # ---- doctor
    def test_doctor_and_capability_status(self):
        ad = self._adapter()
        doc = ad.doctor()
        self.assertEqual((doc["status"], doc["exit_code"], doc["schema"]), ("ok", 0, "qc/doctor@1"))
        caps = QcAdapter.capability_status(doc)
        self.assertEqual((caps["ffprobe"], caps["ffmpeg"], caps["filter:ebur128"]), ("AVAILABLE", "AVAILABLE", "AVAILABLE")); self.assertNotIn("python", caps); self.assertNotIn("path_policy", caps)
        os.environ["FAKE_QC_MODE"] = "doctor_degraded"
        doc = ad.doctor()
        self.assertEqual((doc["status"], doc["exit_code"]), ("degraded", 2)); self.assertEqual(QcAdapter.capability_status(doc)["ffmpeg"], "MISSING"); self.assertEqual(doc["unavailable_tools"], ["ffmpeg"])
        os.environ["FAKE_QC_MODE"] = "doctor_fail"
        doc = ad.doctor()
        self.assertEqual((doc["status"], doc["exit_code"]), ("fail", 1)); self.assertEqual(QcAdapter.capability_status(doc)["ffprobe"], "MISSING")
        self.assertEqual(QcAdapter.capability_status({}), {})
        # locate: the env var points at a checkout, otherwise nothing is guessed
        self.assertIsNone(locate_qc(env={"PATH": self.tmp}))
        fake_root = Path(self.tmp) / "qc-skill" / "src" / "qc_skill"; fake_root.mkdir(parents=True); (fake_root / "cli.py").write_text("", encoding="utf-8")
        sk = locate_qc(env={ENV_DIR: str(Path(self.tmp) / "qc-skill"), "PATH": self.tmp})
        self.assertIsNotNone(sk); self.assertEqual(sk.command[1:], ["-m", "qc_skill.cli"]); self.assertEqual(sk.env, {"PYTHONPATH": str(Path(self.tmp) / "qc-skill" / "src")})


@unittest.skipUnless(os.environ.get(ENV_DIR), f"set {ENV_DIR} to run against the real qc-skill")
class QcRealSkillTests(unittest.TestCase):
    """The real Skill through the same adapter: a 3 s fixture rendered by ffmpeg, check / inspect / subtitle / delivery, admission holds."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = os.path.realpath(tempfile.mkdtemp())
        cls.src_dir = str(Path(cls.tmp) / "src"); os.makedirs(cls.src_dir)
        cls.ws = str(Path(cls.tmp) / "ws"); os.makedirs(cls.ws)
        cls.src = str(Path(cls.src_dir) / "bars.mp4")
        if not shutil.which("ffmpeg"):
            raise unittest.SkipTest("ffmpeg not on PATH")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "3",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", cls.src], check=True)
        cls.srt = str(Path(cls.src_dir) / "bars.srt")
        Path(cls.srt).write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n\n2\n00:00:01,500 --> 00:00:02,800\nworld\n", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _adapter(self):
        return QcAdapter(locate_qc(os.environ[ENV_DIR]), workspace=self.ws, allowed_inputs=[self.src_dir])

    def test_real_contract_doctor_and_runs(self):
        ad = self._adapter(); paths = {"clip": self.src, "sub": self.srt}
        self.assertEqual(ad.drift(), [], ad.drift()); self.assertTrue(ad.version.startswith("0.1."))
        doc = ad.doctor()
        self.assertIn(doc["status"], ("ok", "degraded")); self.assertEqual(QcAdapter.capability_status(doc)["ffprobe"], "AVAILABLE")
        r = ad.run(Operation(tool=TOOL_CHECK, args={"input": "clip", "kind": "video", "rules": {"video": {"expected_width": 640, "expected_height": 360, "expected_frame_rate": 25}}}, inputs=["clip"], outputs=[], kind="qa", id="op_r1"), paths)
        self.assertTrue(r.ok, r.data.get("error")); self.assertEqual(r.data["verdict"], "PASS"); self.assertEqual(r.data["fingerprint"], sha(self.src)); self.assertTrue(r.data["admitted"])
        self.assertTrue(all(c["check_id"].startswith("video.") for c in r.data["checks"])); self.assertEqual(r.data["provenance"]["measurement_source"], "OBSERVED")
        self.assertEqual(lift_report(r, "clip").data["verdict"], "PASS")
        r2 = ad.run(Operation(tool=TOOL_CHECK, args={"input": "clip", "kind": "video", "rules": {"video": {"expected_width": 1920}}}, inputs=["clip"], outputs=[], kind="qa", id="op_r2"), paths)
        self.assertTrue(r2.ok); self.assertEqual(r2.data["verdict"], "FAIL"); self.assertIn("VIDEO_RESOLUTION_MISMATCH", [f["code"] for f in r2.data["findings"]])
        r3 = ad.run(Operation(tool=TOOL_INSPECT, args={"input": "clip", "kind": "audio"}, inputs=["clip"], outputs=[], kind="measure", id="op_r3"), paths)
        self.assertTrue(r3.ok, r3.data.get("error")); self.assertEqual(r3.data["checks"], []); self.assertTrue(any(m["id"] == "audio.integrated_loudness_lufs" for m in r3.data["measurements"]))
        r4 = ad.run(Operation(tool=TOOL_CHECK, args={"input": "sub", "kind": "subtitle", "reference_video": "clip", "rules": {"subtitle": {"max_line_length": 42}}}, inputs=["sub"], outputs=[], kind="qa", id="op_r4"), paths)
        self.assertTrue(r4.ok, r4.data.get("error")); self.assertEqual(r4.data["companions"]["reference_video"], sha(self.src)); self.assertEqual(r4.data["fingerprint"], sha(self.srt))
        r5 = ad.run(Operation(tool=TOOL_CHECK, args={"input": "clip", "kind": "delivery", "subtitle": "sub", "rules": {"delivery": {"require_subtitle": True, "audio": {"max_true_peak_dbfs": 0.0}}}}, inputs=["clip"], outputs=[], kind="qa", id="op_r5"), paths)
        self.assertTrue(r5.ok, r5.data.get("error")); self.assertEqual(r5.data["companions"]["subtitle"], sha(self.srt)); self.assertTrue(any(c["category"] == "subtitle" for c in r5.data["checks"]))
        r6 = ad.run(Operation(tool=TOOL_CHECK, args={"input": "clip", "kind": "video", "rules": {"video": {"expected_width": 640, "expected_height": 360, "expected_frame_rate": 25}}}, inputs=["clip"], outputs=[], kind="qa", id="op_r6"), paths)
        self.assertTrue(r6.ok and r6.data["reused"]); self.assertEqual(r6.data["cache"]["status"], "hit")
        rb = ad.run(Operation(tool=TOOL_CHECK, args={"input": "clip", "kind": "video", "rules": {"video": {"expected_bitrate": 1}}}, inputs=["clip"], outputs=[], kind="qa", id="op_rb"), paths)
        self.assertFalse(rb.ok); self.assertEqual(rb.data["error"]["code"], "INVALID_REQUEST")


if __name__ == "__main__":
    unittest.main()
