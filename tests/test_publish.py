"""ADR-087 (publish-skill roadmap step 6, agent side: SkillSpec publish_upload): publishing a QC-passed deliverable is an always-CONFIRM step after
the QC gate; render runs publish-skill's offline plan and dry run only; the upload is `video-agent publish <artifact> --confirm <token>` by a human
after `deliver`, sent once and never retried. Against a fake `publish` process (tests/fake_publish.py) that speaks publish-skill main's contract,
doctor and result shapes; nothing touches a network."""
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
from video_agent.agent.publish import APPROVAL_KEYS, SUBJECT, check_publish_keys  # noqa: E402
from video_agent.capabilities.resolver import CapabilityResolver  # noqa: E402
from video_agent.mcp_server import TOOLS as MCP_TOOLS  # noqa: E402
from video_agent.project import ProjectIR  # noqa: E402
from video_agent.project.validator import check_publish  # noqa: E402
from video_agent.tools.discovery import MODULE_BOOTSTRAP  # noqa: E402
from video_agent.tools.publish import (ERROR_CLASSES, ContractError, PublishAdapter, breaking_drift, check_contract, drift, locate_publish, pinned_contract,  # noqa: E402
                                       platforms)
from video_agent.tools.publish.adapter import PINNED_CONTRACT_PATH  # noqa: E402
from video_agent.tools.skill_process import CliSkill  # noqa: E402
from video_agent.models import Operation  # noqa: E402
import fake_publish  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_publish.py")
MODES = ("FAKE_PUB_VERDICT", "FAKE_PUB_MODE", "FAKE_PUB_CONTRACT", "FAKE_PUB_CALLS", "FAKE_PUB_UPLOADS", "VIDEO_AGENT_PUBLISH_DIR")


def fake_skill() -> CliSkill:
    return CliSkill("publish", [sys.executable, FAKE], None, {})


def pub_adapter(tmp: str) -> PublishAdapter:
    ws = str(Path(tmp) / "ws")
    os.makedirs(ws, exist_ok=True)
    return PublishAdapter(fake_skill(), workspace=ws, allowed_inputs=[str(Path(tmp) / "src"), ws], ffmpeg_skill_dir=tmp)


def pub_capabilities(factory) -> dict:
    caps: dict = {}
    CapabilityResolver._finishing_skill(CapabilityResolver.__new__(CapabilityResolver), caps, "publish", fake_skill(), lambda _sk: factory(), "VIDEO_AGENT_PUBLISH_DIR",
                                        "publish-skill", "publish", per_item=lambda ad, doc: (ad.platform_status(doc), {}), item_evidence=lambda ad, doc: ad.platform_evidence(doc))
    return caps


class PubCaps(PipelineCaps):
    def __init__(self, pub: dict, **kw):
        super().__init__(**kw)
        self.pub = pub

    def resolve(self, refresh=False):
        out = super().resolve(refresh)
        out.update(self.pub)
        return out


def pub_service(tmp: str, installed: bool = True):
    svc = pipeline_service(tmp, caps=PubCaps(pub_capabilities(lambda: pub_adapter(tmp)) if installed else {}))
    if installed:
        svc._adapter.register(pub_adapter(tmp))
        svc.adapter([])
    return svc


def calls(log: str):
    return [json.loads(x) for x in Path(log).read_text(encoding="utf-8").splitlines()] if os.path.exists(log) else []


class _Base(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self._env = {k: os.environ.get(k) for k in MODES}
        for k in MODES:
            os.environ.pop(k, None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.log = str(Path(self.tmp) / "pub_calls.log")
        self.uploads = str(Path(self.tmp) / "uploads.log")
        os.environ["FAKE_PUB_CALLS"] = self.log
        os.environ["FAKE_PUB_UPLOADS"] = self.uploads

    def tearDown(self):
        clear_modes()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def confirmed_uploads(self):
        return [c for c in calls(self.log) if c[:1] == ["upload"] and any(a.startswith("--confirm=") for a in c)]

    def uploaded(self):
        return [json.loads(x) for x in Path(self.uploads).read_text(encoding="utf-8").splitlines()] if os.path.exists(self.uploads) else []

    def rendered(self, reqs, **kw):
        svc = pub_service(self.tmp)
        r = plan_and_render(svc, [fake_media(self.tmp, "talk.mp4")], reqs, profile="youtube", **kw)
        return svc, r


REQ = {"qc": True, "publish.platform": "youtube", "publish.title": "My talk", "publish.tags": "talk,conference", "publish.privacy": "unlisted"}


class ContractTests(_Base):
    def test_pin_shape_and_contract_checks(self):
        c = pinned_contract()
        self.assertEqual(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"), json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.assertEqual((c["skill_id"], c["version"]), ("publish", "0.1.0"))
        self.assertEqual(check_contract(c), [])
        self.assertEqual(set(ERROR_CLASSES), set(c["errors"]))
        p = platforms(c)
        self.assertEqual(sorted(p), sorted(k for k, v in next(t for t in c["tools"] if t["tool_id"] == "publish/upload")["uploaders"].items() if v),
                         "the platforms are the contract's uploaders, not a list in the agent")
        self.assertEqual(p["x"]["privacy"], ["public"])
        self.assertEqual(p["youtube"]["metadata_fields"], ["title", "description", "tags", "privacy", "thumbnail", "captions"])
        for bad, msg in ((dict(c, confirmation=dict(c["confirmation"], from_environment_or_config=True)), "never come from the environment"),
                         (dict(c, execution=dict(c["execution"], network_requires="nothing")), "network only with --confirm"),
                         (dict(c, credentials=dict(c["credentials"], from_files_or_flags=True)), "never come from files or flags"),
                         (dict(c, version="0.2.0"), "supported range")):
            self.assertTrue(any(msg in e for e in check_contract(bad)), (msg, check_contract(bad)))

    def test_drift_classification(self):
        for mode, fatal, construct in (("pinned", None, True), ("drift_platform", False, True), ("gone_platform", True, True), ("drift_confirm", True, False)):
            os.environ["FAKE_PUB_CONTRACT"] = mode
            if not construct:
                with self.assertRaises(ContractError):
                    pub_adapter(self.tmp)
                continue
            ad = pub_adapter(self.tmp)
            if fatal is None:
                self.assertEqual((ad.drift(), ad.breaking_drift()), ([], []))
            else:
                self.assertTrue(ad.drift(), mode)
                self.assertEqual(bool(ad.breaking_drift()), fatal, (mode, ad.breaking_drift()))
        os.environ["FAKE_PUB_CONTRACT"] = "drift_platform"
        self.assertNotIn("linkedin", platforms(), "planning reads the pin: a platform added on main waits for a re-pin")
        self.assertEqual((drift(pinned_contract()), breaking_drift(pinned_contract())), ([], []))

    def test_locate_checkout_and_env(self):
        root = fake_publish.checkout(Path(self.tmp) / "publish-skill")
        sk = locate_publish(env={"VIDEO_AGENT_PUBLISH_DIR": root, "PATH": ""})
        self.assertEqual(sk.command[1:], ["-c", MODULE_BOOTSTRAP, "publish_skill.cli"])
        self.assertIsNone(locate_publish(env={"VIDEO_AGENT_PUBLISH_DIR": str(Path(self.tmp) / "nothing"), "PATH": ""}))


class AdapterTests(_Base):
    def setUp(self):
        super().setUp()
        self.ad = pub_adapter(self.tmp)
        self.f = fake_media(self.tmp, "final.mp4")

    def test_argv_dry_run_and_refusals(self):
        argv = self.ad.metadata_argv("youtube", {"title": "-T", "description": "line1\nline2", "tags": ["a", "b"], "privacy": "public"})
        self.assertEqual(argv, ["--title=-T", "--description=line1\nline2", "--tag=a", "--tag=b", "--privacy=public"], "each value one --flag=value word")
        dry = self.ad.dry_run(self.f, "youtube", {"title": "T", "privacy": "private"})
        self.assertTrue(dry["confirm_token"].startswith("confirm-"))
        self.assertEqual((dry["uploaded"], dry["network_used"]), (False, False))
        self.assertEqual(self.confirmed_uploads(), [])
        for args, msg in ((("tiktok", {"description": "x"}), "does not take description"), (("x", {"privacy": "private"}), "is not one x takes"),
                          (("myspace", {}), "not implemented"), (("youtube", {"token": "x"}), "forbidden key")):
            with self.assertRaises(Exception) as cm:
                self.ad.metadata_argv(*args)
            self.assertIn(msg, str(cm.exception))
        with self.assertRaises(Exception) as cm:
            self.ad.dry_run(str(Path(tempfile.mkdtemp()) / "x.mp4"), "youtube", {})
        self.assertIn("outside", str(cm.exception))
        os.environ["FAKE_PUB_MODE"] = "dry_wrong_sha"
        self.assertEqual(self.ad.dry_run(self.f, "youtube", {})["error"]["code"], "INVALID_OUTPUT")

    def test_never_a_pipeline_operation(self):
        r = self.ad.run(Operation(tool="publish/upload", args={}, inputs=[], outputs=[]), {})
        self.assertFalse(r.ok)
        self.assertIn("never run as a pipeline operation", r.data["error"]["message"])
        self.assertEqual(calls(self.log)[-1][:1], ["contract"], "nothing but the contract was called")


class RequirementTests(_Base):
    def test_vocabulary_and_refusals(self):
        check_publish_keys({"publish.platform": "x", "publish.title": "hi", "publish.privacy": "public"})
        for given, msg in (({"publish.title": "t"}, "needs publish.platform"), ({"publish.platform": "myspace"}, "has no uploader"),
                           ({"publish.platform": "x", "publish.title": "hi"}, "public the moment it is made"),
                           ({"publish.platform": "tiktok", "publish.description": "d"}, "tiktok does not take description"),
                           ({"publish.platform": "vimeo", "publish.thumbnail": True}, "vimeo does not take thumbnail"),
                           ({"publish.platform": "youtube", "publish.privacy": "secret"}, "is not one youtube takes"),
                           ({"publish.platform": "youtube", "publish.when": "now"}, "unknown requirement key")):
            with self.assertRaises(EditRequirementError, msg=msg) as cm:
                check_publish_keys(given)
            self.assertIn(msg, str(cm.exception))
        self.assertEqual(APPROVAL_KEYS[SUBJECT][1], "CONFIRM")
        self.assertNotIn("publish", {t["name"] for t in MCP_TOOLS}, "the upload is a human's command, not an MCP tool")


class PipelineTests(_Base):
    def test_plan_confirm_render_dry_run_then_a_humans_publish_once(self):
        svc, r = self.rendered(REQ, approve=False)
        d = r["ir"].doc
        self.assertEqual(r["validation"].errors, [])
        steps = [s["skill"] for s in d["plan"]["steps"]]
        self.assertEqual(steps[-1], "publish_upload")
        pstep = d["plan"]["steps"][-1]
        self.assertTrue(set(pstep["depends_on"]) >= {s["id"] for s in d["plan"]["steps"] if s["skill"] == "qc_check"}, "after the QC gate")
        dec = next(x for x in d["decisions"] if x["subject"] == SUBJECT)
        self.assertEqual((dec["type"], dec["approval"], dec["status"]), ("DELIVER", "CONFIRM", "PROPOSED"), "an explicit request still needs confirmation")
        self.assertEqual(r["out"]["status"], "WAITING_FOR_APPROVAL")
        self.assertEqual([c for c in calls(self.log) if c[:1] in (["plan"], ["upload"])], [], "nothing before approval")
        out = svc.render(r["ir"], r["path"], approve=["all"])
        pub = out["publish"]
        self.assertEqual((out["status"], pub["status"], pub["uploaded"]), ("COMPLETED", "AWAITING_CONFIRMATION", False))
        self.assertEqual(self.confirmed_uploads(), [], "render never uploads")
        self.assertIn("Publish to youtube — AWAITING_CONFIRMATION", Path(out["report"]).read_text(encoding="utf-8"))
        art = pub["artifact_id"]
        with self.assertRaises(ValueError) as cm:
            svc.publish(art, pub["confirm_token"])
        self.assertIn("deliver it first", str(cm.exception))
        svc.promote_artifact(art, "final", who="editor")
        with self.assertRaises(ValueError) as cm:
            svc.publish(art, "confirm-0000000000000000")
        self.assertIn("not the confirm_token", str(cm.exception))
        self.assertEqual(self.confirmed_uploads(), [], "a wrong token never reaches the Skill")
        res = svc.publish(art, pub["confirm_token"], who="editor")
        self.assertEqual((res["status"], res["uploaded"], res["result"]["video_id"], res["retried"]), ("UPLOADED", True, "fakevid123", False))
        [u] = self.uploaded()
        self.assertEqual((u["platform"], u["metadata"]["title"], u["metadata"]["tags"], u["metadata"]["privacy"]), ("youtube", "My talk", ["talk", "conference"], "unlisted"))
        with self.assertRaises(ValueError) as cm:
            svc.publish(art, pub["confirm_token"])
        self.assertIn("never retries", str(cm.exception))
        self.assertEqual(len(self.uploaded()), 1, "uploaded exactly once")

    def test_a_failed_upload_is_never_retried_unless_nothing_was_published(self):
        os.environ["FAKE_PUB_MODE"] = "fail:QUOTA_EXCEEDED"
        svc, r = self.rendered(REQ)
        pub = r["out"]["publish"]
        svc.promote_artifact(pub["artifact_id"], "final")
        res = svc.publish(pub["artifact_id"], pub["confirm_token"])
        self.assertEqual((res["status"], res["error"]["code"], res["error"]["retryable"]), ("FAILED", "QUOTA_EXCEEDED", True))
        self.assertEqual(len(self.confirmed_uploads()), 1, "the agent did not retry by itself")
        os.environ["FAKE_PUB_MODE"] = "fail:UPLOAD_INTERRUPTED"
        res = svc.publish(pub["artifact_id"], pub["confirm_token"])   # a human's new go-ahead after a quota refusal
        self.assertEqual(res["error"]["code"], "UPLOAD_INTERRUPTED")
        with self.assertRaises(ValueError):
            svc.publish(pub["artifact_id"], pub["confirm_token"])   # interrupted: it may have published
        self.assertEqual(len(self.confirmed_uploads()), 2)

    def test_plan_fail_blocks_the_publish(self):
        os.environ["FAKE_PUB_VERDICT"] = "FAIL"
        svc, r = self.rendered(REQ)
        pub = r["out"]["publish"]
        self.assertEqual(pub["status"], "BLOCKED")
        self.assertEqual(pub["plan"]["rows"][0]["status"], "FAIL")
        self.assertFalse([c for c in calls(self.log) if c[:1] == ["upload"]], "no dry run after a failing plan")

    def test_refused_where_publishing_cannot_follow_qc(self):
        svc = pub_service(self.tmp)
        ir = svc.plan([fake_media(self.tmp, "talk.mp4")], "youtube", user_requirements={k: v for k, v in REQ.items() if k != "qc"})
        b = next(x for x in ir.doc["decisions"] if x["subject"] == SUBJECT)
        self.assertEqual(b["type"], "BLOCK")
        self.assertIn("--set qc=true", b["reason"])
        ir = svc.plan([fake_media(self.tmp, "talk.mp4")], "youtube", user_requirements=dict(REQ, **{"publish.thumbnail": True}))
        self.assertIn("none is planned", next(x for x in ir.doc["decisions"] if x["subject"] == SUBJECT)["reason"])
        ir = svc.plan([fake_media(self.tmp, "a.mp4"), fake_media(self.tmp, "b.mp4")], "youtube", user_requirements=REQ)
        self.assertIn("separate deliverables", next(x for x in ir.doc["decisions"] if x["subject"] == SUBJECT)["reason"])

    def test_thumbnail_attached_from_the_plan(self):
        svc, r = self.rendered(dict(REQ, **{"thumbnail": True, "publish.thumbnail": True}))
        pub = r["out"]["publish"]
        self.assertEqual(pub["status"], "AWAITING_CONFIRMATION", pub)
        dry = next(c for c in calls(self.log) if c[:1] == ["upload"])
        self.assertTrue(any(a.startswith("--thumbnail=") and a.endswith(".png") for a in dry), dry)

    def test_without_publish_skill_the_step_is_handed_off(self):
        svc = pub_service(self.tmp, installed=False)
        ir = svc.plan([fake_media(self.tmp, "talk.mp4")], "youtube", user_requirements=REQ)
        step = next(s for s in ir.doc["plan"]["steps"] if s["skill"] == "publish_upload")
        self.assertEqual((step["tool"], step["status"]), (None, "BLOCKED"))
        [h] = [h for h in svc.handoffs(ir) if h["skill"] == "publish_upload"]
        self.assertEqual(h["repositories"], ["publish-skill"])

    def test_validator_refuses_a_tampered_publish(self):
        svc = pub_service(self.tmp)
        ir = svc.plan([fake_media(self.tmp, "talk.mp4")], "youtube", user_requirements=REQ)
        d = ir.doc
        self.assertEqual(check_publish(d), [])
        for patch, msg in (({"platform": "myspace"}, "no uploader"), ({"privacy": "secret"}, "is not one youtube takes"), ({"target": "nope"}, "not a delivery target"),
                           ({"thumbnail": True}, "makes none")):
            bad = json.loads(json.dumps(d))
            bad["delivery"]["publish"].update(patch)
            self.assertTrue(any(msg in e for e in check_publish(bad)), (msg, check_publish(bad)))
        bad = json.loads(json.dumps(d))
        bad["qa"].pop("qc", None)
        self.assertTrue(any("after QC" in e for e in check_publish(bad)))
        bad = json.loads(json.dumps(d))
        next(x for x in bad["decisions"] if x["subject"] == SUBJECT)["approval"] = "AUTO"
        self.assertFalse(svc.validate(ProjectIR(bad)).ok, "an AUTO approval written into the IR is refused (the basis says CONFIRM)")


if __name__ == "__main__":
    unittest.main()
