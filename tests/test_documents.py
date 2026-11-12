"""ADR-095 (artifacts-skill roadmap step 5, agent side: adapter + SkillSpec document_verify): a document that goes out with the video verified by
artifacts-skill's structural checks, its verdict a QA gate -- a leftover placeholder fails QA -- against a fake `artifacts-skill` process
(tests/fake_artifacts.py) that speaks its 0.4.0 contract, doctor and verify result, with the fake media engine. No backend or network."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import PipelineCaps, clear_modes, fake_media, pipeline_service  # noqa: E402

from video_agent.agent.documents import parse_document_requirements  # noqa: E402
from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.capabilities.resolver import CapabilityResolver  # noqa: E402
from video_agent.models import Operation  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.service import _check_requirement_keys  # noqa: E402
from video_agent.tools.artifacts import (TOOL_VERIFY, ArtifactsAdapter, breaking_drift, check_contract, check_verification, drift, locate_artifacts,  # noqa: E402
                                         pinned_contract, recovery_class)
from video_agent.tools.artifacts.adapter import PINNED_CONTRACT_PATH  # noqa: E402
from video_agent.tools.skill_process import CliSkill  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_artifacts.py")
MODES = ("FAKE_AF_MODE", "FAKE_AF_READY", "FAKE_AF_CALLS", "VIDEO_AGENT_ARTIFACTS_DIR")
CLEAN = "# Script\n\n- Hello.\n- The end.\n"
LEFTOVER = "# Script\n\n- Hello.\n- TODO: write the ending.\n"


def fake_skill() -> CliSkill:
    return CliSkill("artifacts-skill", [sys.executable, FAKE], None, {})


def af_adapter(tmp: str) -> ArtifactsAdapter:
    return ArtifactsAdapter(fake_skill(), allowed_inputs=[str(Path(tmp) / "src"), str(Path(tmp) / "docs")])


def af_capabilities(tmp: str) -> dict:
    caps: dict = {}
    CapabilityResolver._finishing_skill(CapabilityResolver.__new__(CapabilityResolver), caps, "artifacts-skill", fake_skill(), lambda _sk: af_adapter(tmp),
                                        "VIDEO_AGENT_ARTIFACTS_DIR", "artifacts-skill", "artifacts-skill",
                                        per_item=lambda ad, doc: (ad.operation_status(doc), {}), item_evidence=lambda ad, doc: ad.operation_evidence(doc))
    return caps


class AfCaps(PipelineCaps):
    def __init__(self, af: dict, **kw):
        super().__init__(**kw)
        self.af = af

    def resolve(self, refresh=False):
        out = super().resolve(refresh)
        out.update(self.af)
        return out


def af_service(tmp: str, installed: bool = True):
    svc = pipeline_service(tmp, caps=AfCaps(af_capabilities(tmp) if installed else {}), transcription=False)
    if installed:
        svc._adapter.register(af_adapter(tmp))
        svc.adapter([])
    return svc


def document(tmp: str, text: str = CLEAN, name: str = "script.md") -> str:
    p = Path(tmp) / "docs" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return str(p)


class _Base(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self._env = {k: os.environ.get(k) for k in MODES}
        for k in MODES:
            os.environ.pop(k, None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.log = str(Path(self.tmp) / "af_calls.log")
        os.environ["FAKE_AF_CALLS"] = self.log

    def tearDown(self):
        clear_modes()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def verify_calls(self):
        return [json.loads(x)[1:] for x in Path(self.log).read_text(encoding="utf-8").splitlines() if json.loads(x)[:1] == ["verify"]] if os.path.exists(self.log) else []


class ContractTests(_Base):
    def test_pin_and_checks(self):
        c = pinned_contract()
        self.assertEqual(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"), json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n", "pinned from the v0.4.0 tag, one sorted compact line")
        self.assertEqual((c["skill_id"], c["version"], c["contract_version"]), ("artifacts-skill", "0.4.0", "1.0"))
        self.assertEqual((check_contract(c), drift(c), breaking_drift(c)), ([], [], []))

    def test_refused_contracts(self):
        c = pinned_contract()
        v = next(t for t in c["tools"] if t["name"] == "verify")
        for mutate, why in ((lambda x: x.update(skill_id="artifacts"), "skill_id"), (lambda x: x.update(version="0.5.0"), "supported range"),
                            (lambda x: x.update(input_mutation_policy="allowed"), "never"), (lambda x: x["exit_codes"].update(fail=2), "exit_codes")):
            with self.subTest(why=why):
                d = json.loads(json.dumps(c))
                mutate(d)
                self.assertTrue(any(why in e for e in check_contract(d)), check_contract(d))
        d = json.loads(json.dumps(c))
        next(t for t in d["tools"] if t["name"] == "verify")["side_effects"]["writes_files"] = True
        self.assertTrue(any("side_effects" in e for e in check_contract(d)))
        self.assertTrue(breaking_drift(d))
        d = json.loads(json.dumps(c))
        d["version"] = "0.4.9"
        self.assertEqual((check_contract(d), breaking_drift(d)), ([], []))
        self.assertTrue(drift(d), "a version change is drift, not breaking")
        self.assertIn("policy", v["input_schema"]["properties"])

    def test_locate(self):
        d = Path(self.tmp) / "artifacts-skill" / "src" / "artifact_skill"
        d.mkdir(parents=True)
        (d / "__init__.py").write_text("", encoding="utf-8")
        sk = locate_artifacts(str(d.parents[1]))
        self.assertEqual(sk.root, d.parents[1])
        self.assertIsNone(locate_artifacts(str(Path(self.tmp) / "nowhere"), env={"PATH": ""}))

    def test_doctor_and_capability(self):
        caps = af_capabilities(self.tmp)
        self.assertEqual((caps["artifacts-skill"].status, caps["artifacts-skill:VERIFY"].status), ("AVAILABLE", "AVAILABLE"))
        os.environ["FAKE_AF_READY"] = "none"
        caps = af_capabilities(self.tmp)
        self.assertEqual((caps["artifacts-skill"].status, caps["artifacts-skill:VERIFY"].status), ("MISSING", "MISSING"))
        self.assertIn("pip install", caps["artifacts-skill:VERIFY"].evidence["install"])


class AdapterTests(_Base):
    def run_op(self, args, path=None):
        ad = af_adapter(self.tmp)
        doc = path or document(self.tmp, LEFTOVER)
        return ad.run(Operation(tool=TOOL_VERIFY, args=dict({"input": "doc"}, **args), inputs=["doc"], outputs=[], kind="qa"), {"doc": doc})

    def test_argv_and_placeholder_verdict(self):
        r = self.run_op({"preset": "print-a4"})
        self.assertTrue(r.ok, r.data)
        [argv] = self.verify_calls()
        self.assertEqual(argv[:2], ["--json", '--policy={"forbid_placeholder_text": true}'])
        self.assertIn("--policy-preset=print-a4", argv)
        self.assertEqual(argv[-2], "--", "the document after --")
        self.assertEqual((r.data["status"], r.data["placeholders"]), ("fail", ["todo"]))
        self.assertEqual(len(r.data["sha256"]), 64)
        clean = self.run_op({}, document(self.tmp, CLEAN, "clean.md"))
        self.assertEqual((clean.ok, clean.data["status"], clean.data["placeholders"]), (True, "pass", None))

    def test_refused_before_a_process(self):
        outside = Path(self.tmp) / "elsewhere" / "x.md"
        outside.parent.mkdir()
        outside.write_text(CLEAN, encoding="utf-8")
        for args, path in (({"preset": "nope"}, None), ({"policy": {}}, None), ({}, str(outside)), ({}, str(Path(self.tmp) / "docs"))):
            with self.subTest(args=args, path=path):
                document(self.tmp)
                r = self.run_op(args, path)
                self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_REQUEST"))
        self.assertEqual(self.verify_calls(), [])

    def test_results_refused_and_codes_kept(self):
        for mode, code in (("bad_kind", "INVALID_OUTPUT"), ("wrong_exit", "INVALID_OUTPUT"), ("mutate", "INPUT_CHANGED")):
            with self.subTest(mode=mode):
                os.environ["FAKE_AF_MODE"] = mode
                r = self.run_op({})
                self.assertEqual((r.ok, r.data["error"]["code"]), (False, code))
        os.environ["FAKE_AF_MODE"] = "fail:ARTIFACT_CAPABILITY_MISSING:capability"
        r = self.run_op({})
        self.assertEqual((r.ok, r.data["error"]["code"], r.data["error"]["recovery_class"], r.exit_code), (False, "ARTIFACT_CAPABILITY_MISSING", "TOOL_MISSING", 3))
        os.environ.pop("FAKE_AF_MODE")
        r = self.run_op({}, document(self.tmp, "x", "blob.bin"))
        self.assertEqual((r.data["error"]["code"], r.data["error"]["recovery_class"]), ("ARTIFACT_TYPE_UNSUPPORTED", "INVALID_ARGS"))
        self.assertEqual(recovery_class("SOMETHING_NEW", "internal"), "SKILL_ERROR")
        self.assertEqual(check_verification({"kind": "structural", "status": "pass", "checks": [{"id": "x", "status": "maybe"}]})[0][:7], "check '")


class RequirementTests(_Base):
    def test_vocabulary_and_refusals(self):
        p = document(self.tmp)
        self.assertEqual(parse_document_requirements({"document.verify": p, "document.verify.preset": "print-a4"}), {"path": p, "preset": "print-a4"})
        srt = document(self.tmp, "1\n00:00:00,000 --> 00:00:01,000\nhi\n", "captions.srt")
        for reqs, why in (({"document.verify.preset": "print-a4"}, "not set"), ({"document.verify": str(Path(self.tmp) / "nope.md")}, "existing document"),
                          ({"document.verify": srt}, "subtitle-skill"), ({"document.verify": p, "document.verify.preset": "a3"}, "one of"),
                          ({"document.verify": p, "document.check": True}, "unknown requirement key")):
            with self.subTest(reqs=reqs):
                with self.assertRaises(EditRequirementError) as cm:
                    parse_document_requirements(reqs)
                self.assertIn(why, str(cm.exception))
                with self.assertRaises(ValueError):
                    _check_requirement_keys(reqs)


class PlanTests(_Base):
    def plan(self, text=CLEAN, installed=True, reqs=None):
        self.svc = af_service(self.tmp, installed=installed)
        self.media = fake_media(self.tmp, duration=16.0)
        self.doc = document(self.tmp, text)
        return self.svc.plan([self.media], "generic", user_requirements=dict({"document.verify": self.doc}, **(reqs or {})))

    def render(self, ir):
        p = str(Path(self.tmp) / "p.json")
        save_ir(ir, p)
        return self.svc.render(load_ir(p), p, approve=["all"])

    def test_auto_step_and_clean_document_passes(self):
        ir = self.plan()
        d = ir.doc
        [dec] = [x for x in d["decisions"] if x["subject"] == "document.verify"]
        self.assertEqual((dec["type"], dec["approval"]), ("DELIVER", "AUTO"), "read-only: AUTO")
        [doc] = d["qa"]["documents"]
        self.assertEqual((doc["name"], doc["path"], len(doc["sha256"])), ("script.md", self.doc, 64))
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "document_verify")
        self.assertEqual((step["tool"], step["params"]), ("artifacts-skill/verify", {"document": "document_1", "name": "script.md"}))
        self.assertIn(str(Path(self.doc).parent), d["execution"]["allowed_inputs"])
        self.assertEqual(self.svc.validate(ir).errors, [])
        out = self.render(ir)
        self.assertEqual(out["status"], "COMPLETED", out.get("qa"))
        items = [i for i in out["qa"]["items"] if i["layer"] == "document"]
        self.assertEqual([(i["name"], i["status"]) for i in items], [("verdict", "PASS")])

    def test_placeholder_left_in_the_document_fails_qa(self):
        """The roadmap's done-when: a placeholder left in a doc fails QA."""
        out = self.render(self.plan(LEFTOVER))
        self.assertEqual(out["status"], "REVIEW")
        self.assertEqual(out["qa"]["status"], "FAIL")
        items = {i["name"]: i for i in out["qa"]["items"] if i["layer"] == "document"}
        self.assertEqual(items["verdict"]["status"], "FAIL")
        self.assertEqual((items["leftover_placeholder_text"]["status"], items["leftover_placeholder_text"]["observed"]), ("FAIL", ["todo"]))
        self.assertTrue(any(i["type"] == "DELIVERY_SPEC_FAILURE" and "document:script.md" in i["evidence"] for i in out["qa"]["incidents"]))

    def test_unknown_is_never_a_pass(self):
        out = self.render(self.plan(reqs={"document.verify.preset": "print-a4"}))
        items = {i["name"]: i["status"] for i in out["qa"]["items"] if i["layer"] == "document"}
        self.assertEqual((items["verdict"], items["page_size"]), ("WARN", "WARN"))

    def test_document_changed_since_the_plan_fails_qa(self):
        ir = self.plan()
        Path(self.doc).write_text(CLEAN + "\n- more\n", encoding="utf-8")
        out = self.render(ir)
        [v] = [i for i in out["qa"]["items"] if i["layer"] == "document"]
        self.assertEqual(v["status"], "FAIL")
        self.assertIn("changed since the plan", v["fix_hint"])

    def test_without_the_skill_the_step_is_handed_off(self):
        ir = self.plan(installed=False)
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        [h] = [h for h in self.svc.handoffs(ir) if h["skill"] == "document_verify"]
        self.assertEqual((h["provided_by"], h["repositories"]), (["artifacts-skill"], ["artifacts-skill"]))
        self.assertIn("git clone https://github.com/kajisho5/artifacts-skill && pip install -e 'artifacts-skill[all]'", h["install"])

    def test_tampered_ir_refused(self):
        ir = self.plan()
        ir.doc["qa"]["documents"][0]["path"] = "relative.md"
        self.assertTrue(any("absolute path" in e for e in self.svc.validate(ir).errors))
        ir = self.plan()
        ir.doc["plan"]["steps"] = [s for s in ir.doc["plan"]["steps"] if s["skill"] != "document_verify"]
        self.assertTrue(any("has no document_verify step" in e for e in self.svc.validate(ir).errors))


if __name__ == "__main__":
    unittest.main()
