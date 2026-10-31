"""ADR-086 (nle-bridge-skill roadmap step 7, agent side: the adapter): `delivery.nle=export|conform` hands the project IR the job executes to
nle-bridge, which reads the subject's timeline record itself and writes OTIO / FCP7 XML / EDL / SRT (or a conform package). Against a fake
`nle-bridge` process (tests/fake_nle_bridge.py) that reads the IR the same way; the real Skill is exercised by NleBridgeRealTests."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import PipelineCaps, clear_modes, fake_media, pipeline_service, plan_and_render  # noqa: E402

from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.nle import SUBJECT, check_nle_keys  # noqa: E402
from video_agent.capabilities.resolver import CapabilityResolver  # noqa: E402
from video_agent.models import Operation  # noqa: E402
from video_agent.project.validator import check_nle  # noqa: E402
from video_agent.tools.nle_bridge import (ContractError, NleBridgeAdapter, breaking_drift, check_contract, default_formats, drift, formats,  # noqa: E402
                                          pinned_contract)
from video_agent.tools.skill_process import CliSkill  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_nle_bridge.py")


def fake_skill() -> CliSkill:
    return CliSkill("nle-bridge", [sys.executable, FAKE], None, {})


class NleCaps(PipelineCaps):
    def __init__(self, extra_caps: dict, **kw):
        super().__init__(**kw)
        self.more = extra_caps

    def resolve(self, refresh=False):
        out = super().resolve(refresh)
        out.update(self.more)
        return out


def nle_service(tmp: str, installed: bool = True):
    ws = str(Path(tmp) / "ws")
    os.makedirs(ws, exist_ok=True)
    caps: dict = {}
    if installed:
        CapabilityResolver._finishing_skill(CapabilityResolver.__new__(CapabilityResolver), caps, "nle-bridge", fake_skill(), lambda _s: NleBridgeAdapter(fake_skill(), workspace=ws),
                                            "VIDEO_AGENT_NLE_BRIDGE_DIR", "nle-bridge-skill", "nle-bridge", per_item=None)
    svc = pipeline_service(tmp, caps=NleCaps(caps))
    if installed:
        svc._adapter.register(NleBridgeAdapter(fake_skill(), workspace=ws))
        svc.adapter([])
    return svc


class _Base(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.log = str(Path(self.tmp) / "nle_calls.log")
        os.environ["FAKE_NLE_CALLS"] = self.log
        os.environ.pop("FAKE_NLE_MODE", None)

    def tearDown(self):
        clear_modes()
        os.environ.pop("FAKE_NLE_MODE", None)
        os.environ.pop("FAKE_NLE_CALLS", None)

    def calls(self):
        return [json.loads(x) for x in Path(self.log).read_text(encoding="utf-8").splitlines()] if os.path.exists(self.log) else []


class ContractTests(_Base):
    def test_pin_and_checks(self):
        c = pinned_contract()
        self.assertEqual((c["skill_id"], c["version"], check_contract(c)), ("nle-bridge", "0.1.0", []))
        self.assertEqual(c["dependencies"][0]["version_range"], ">=2.0.0 <4.0.0", "pinned at main c0ab78f (step 12)")
        self.assertEqual((formats(), default_formats()), (("edl", "fcp7", "otio", "srt"), ("otio", "fcp7", "edl", "srt")))
        for bad, msg in ((dict(c, inputs=["nle-bridge/timeline@1"]), "project IR"), (dict(c, execution=dict(c["execution"], network=True)), "network false")):
            self.assertTrue(any(msg in e for e in check_contract(bad)), check_contract(bad))
        self.assertEqual((drift(c), breaking_drift(c)), ([], []))
        with self.assertRaises(ContractError):
            NleBridgeAdapter(CliSkill("nle-bridge", [sys.executable, "-c", "print('{}')"], None, {}))

    def test_requirements(self):
        check_nle_keys({"delivery.nle": "conform", "delivery.nle.formats": "otio,fcp7", "delivery.nle.zip": False})
        for given, msg in (({"delivery.nle": "aaf"}, "export, conform"), ({"delivery.nle.formats": "otio"}, "delivery.nle is not"),
                           ({"delivery.nle": "export", "delivery.nle.formats": "aaf"}, "distinct names"), ({"delivery.nle": "export", "delivery.nle.formats": "srt"}, "timeline format"),
                           ({"delivery.nle": "export", "delivery.nle.zip": True}, "conform package option"), ({"delivery.nle": "export", "delivery.nle.rate": 25}, "unknown")):
            with self.assertRaises(EditRequirementError, msg=msg) as cm:
                check_nle_keys(given)
            self.assertIn(msg, str(cm.exception))

    def test_argv_and_admission(self):
        ws = str(Path(self.tmp) / "ws"); os.makedirs(ws, exist_ok=True)
        ad = NleBridgeAdapter(fake_skill(), workspace=ws)
        proj = str(Path(ws) / "project.json")
        Path(proj).write_text(json.dumps({"delivery": {"timelines": [{"asset": "a", "clips": [{"asset": "a", "in": 0, "out": 1}], "timeline_duration": 1}]}}), encoding="utf-8")
        paths = {"project_ir": proj, "o": str(Path(ws) / "out")}
        op = Operation(tool="nle-bridge/export", args={"project": "project_ir", "subject": "a", "formats": ["otio", "srt"], "output": "o"}, inputs=[], outputs=["o"])
        r = ad.run(op, paths)
        self.assertTrue(r.ok, r.data)
        c = self.calls()[-1]
        self.assertEqual(c[c.index("--") + 1:], [os.path.realpath(proj)])
        self.assertTrue(all(a.startswith("--") for a in c[1:c.index("--")]))
        for mode, code in (("bad_hash", "INVALID_OUTPUT"), ("outside", "INVALID_OUTPUT"), ("fail:CONTENT_NOT_EXPORTED", "CONTENT_NOT_EXPORTED")):
            os.environ["FAKE_NLE_MODE"] = mode
            self.assertEqual(ad.run(op, paths).data["error"]["code"], code, mode)
        os.environ.pop("FAKE_NLE_MODE")
        for args, msg in (({"formats": ["aaf"]}, "distinct names"), ({"subject": "../x"}, "is not an id"), ({"shell": "x"}, "forbidden")):
            r = ad.run(Operation(tool="nle-bridge/export", args=dict(op.args, **args), inputs=[], outputs=["o"]), paths)
            self.assertIn(msg, r.data["error"]["message"])


class PipelineTests(_Base):
    def test_export_after_the_timeline_from_the_executed_ir(self):
        svc = nle_service(self.tmp)
        r = plan_and_render(svc, [fake_media(self.tmp, "talk.mp4")], {"delivery.timeline": "otio", "delivery.nle": "export", "delivery.nle.formats": "otio,fcp7,edl"}, profile="youtube")
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        dec = next(x for x in d["decisions"] if x["subject"] == SUBJECT)
        self.assertEqual((dec["type"], dec["approval"]), ("DELIVER", "AUTO"), "files only, no media: AUTO like the timeline export")
        skills = [s["skill"] for s in d["plan"]["steps"]]
        self.assertLess(skills.index("timeline_export"), skills.index("nle_export"))
        self.assertEqual(d["delivery"]["nle"][0]["formats"], ["otio", "fcp7", "edl"])
        self.assertEqual(r["out"]["execution"]["status"], "COMPLETED", r["out"]["execution"])
        res = next(x for x in r["out"]["execution"]["results"] if x["tool"] == "nle-bridge/export")
        self.assertEqual([f["format"] for f in res["data"]["files"]], ["otio", "fcp7", "edl"])
        self.assertTrue(any(w["code"] == "NOT_IN_TIMELINE" for w in res["data"]["warnings"]), "the timeline decision's not_exported reaches nle-bridge from the IR")
        job = Path(svc.workspace) / "jobs" / r["out"]["job"]["id"]
        self.assertEqual(json.loads((job / "nle" / "project.json").read_text(encoding="utf-8"))["project"]["id"], d["project"]["id"], "the IR the job executed")
        prov = json.loads((job / "provenance.json").read_text(encoding="utf-8"))
        self.assertIn("nle.export", [o["kind"] for o in prov["skill_observations"]])

    def test_conform_zip_and_refusals(self):
        svc = nle_service(self.tmp)
        r = plan_and_render(svc, [fake_media(self.tmp, "talk.mp4")], {"delivery.timeline": "edl", "delivery.nle": "conform"}, profile="youtube")
        res = next(x for x in r["out"]["execution"]["results"] if x["tool"] == "nle-bridge/conform")
        self.assertTrue(res["output"].endswith(".zip"))
        self.assertEqual(res["data"]["relink_media"][0]["start_timecode"], "00:00:00:00")
        self.assertIn("--zip", [c for c in self.calls() if c[:1] == ["conform"]][0])
        ir = svc.plan([fake_media(self.tmp, "talk.mp4")], "youtube", user_requirements={"delivery.nle": "export"})
        b = next(x for x in ir.doc["decisions"] if x["subject"] == SUBJECT)
        self.assertEqual(b["type"], "BLOCK")
        self.assertIn("delivery.timeline=otio", b["reason"])
        ir = nle_service(self.tmp, installed=False).plan([fake_media(self.tmp, "talk.mp4")], "youtube", user_requirements={"delivery.timeline": "otio", "delivery.nle": "export"})
        step = next(s for s in ir.doc["plan"]["steps"] if s["skill"] == "nle_export")
        self.assertEqual((step["tool"], step["status"]), (None, "BLOCKED"))
        [h] = [h for h in nle_service(self.tmp, installed=False).handoffs(ir) if h["skill"] == "nle_export"]
        self.assertEqual(h["repositories"], ["nle-bridge-skill"])

    def test_validator(self):
        ir = nle_service(self.tmp).plan([fake_media(self.tmp, "talk.mp4")], "youtube", user_requirements={"delivery.timeline": "otio", "delivery.nle": "export"})
        self.assertEqual(check_nle(ir.doc), [])
        bad = json.loads(json.dumps(ir.doc))
        bad["delivery"]["nle"][0]["formats"] = ["aaf"]
        self.assertTrue(check_nle(bad))
        bad = json.loads(json.dumps(ir.doc))
        bad["delivery"].pop("timelines")
        self.assertTrue(any("none is planned" in e for e in check_nle(bad)))


if __name__ == "__main__":
    unittest.main()
