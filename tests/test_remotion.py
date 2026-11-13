"""ADR-093 (remotion-ffmpeg-skill roadmap step 4, agent side: SkillSpec measured_motion -> pipeline, CONFIRM): one Remotion composition over the edited
picture by remotion-ffmpeg-skill's pipeline, against a fake checkout (tests/fake_remotion.py) that speaks its v0.14.0 contract, doctor and pipeline
result, with the fake media engine. No Node, Remotion, ffmpeg or network."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import PipelineCaps, clear_modes, fake_media, pipeline_service  # noqa: E402

from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.remotion import parse_remotion_requirements  # noqa: E402
from video_agent.capabilities.resolver import CapabilityResolver  # noqa: E402
from video_agent.models import Operation  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.service import _check_requirement_keys  # noqa: E402
from video_agent.tools.remotion import (RemotionAdapter, RemotionSkill, breaking_drift, check_contract, drift, locate_remotion, pinned_contract,  # noqa: E402
                                        recovery_class, render_name)
from video_agent.tools.remotion.adapter import COMPOSITIONS, PINNED_CONTRACT_PATH, SENT  # noqa: E402

FAKE = Path(__file__).resolve().parent / "fake_remotion.py"
MODES = ("FAKE_RF_MODE", "FAKE_RF_READY", "FAKE_RF_CALLS", "FAKE_RF_PIN", "VIDEO_AGENT_REMOTION_DIR")


def fake_checkout(tmp: str) -> RemotionSkill:
    d = Path(tmp) / "remotion-ffmpeg-skill" / "scripts"
    d.mkdir(parents=True, exist_ok=True)
    for name in ("_contract", "doctor", "pipeline"):
        shutil.copy(FAKE, d / f"{name}.py")
    return RemotionSkill(d.parent, sys.executable)


def rf_adapter(tmp: str) -> RemotionAdapter:
    ws = str(Path(tmp) / "ws")
    os.makedirs(ws, exist_ok=True)
    return RemotionAdapter(fake_checkout(tmp), workspace=ws, allowed_inputs=[str(Path(tmp) / "src"), str(Path(tmp) / "motion"), ws])


def rf_capabilities(tmp: str) -> dict:
    caps: dict = {}
    CapabilityResolver._finishing_skill(CapabilityResolver.__new__(CapabilityResolver), caps, "remotion-ffmpeg-skill", fake_checkout(tmp), lambda _sk: rf_adapter(tmp),
                                        "VIDEO_AGENT_REMOTION_DIR", "remotion-ffmpeg-skill", "npx remotion-ffmpeg-skill",
                                        per_item=lambda ad, doc: (ad.operation_status(doc), {}), item_evidence=lambda ad, doc: ad.operation_evidence(doc))
    return caps


class RfCaps(PipelineCaps):
    def __init__(self, rf: dict, **kw):
        super().__init__(**kw)
        self.rf = rf

    def resolve(self, refresh=False):
        out = super().resolve(refresh)
        out.update(self.rf)
        return out


def rf_service(tmp: str, installed: bool = True):
    svc = pipeline_service(tmp, caps=RfCaps(rf_capabilities(tmp) if installed else {}), transcription=False)
    if installed:
        svc._adapter.register(rf_adapter(tmp))
        svc.adapter([])
    return svc


def project(tmp: str, installed: bool = True) -> str:
    p = Path(tmp) / "motion"
    p.mkdir(parents=True, exist_ok=True)
    (p / "package.json").write_text("{}", encoding="utf-8")
    if installed:
        (p / "node_modules" / "remotion").mkdir(parents=True, exist_ok=True)
        (p / "node_modules" / "remotion" / "package.json").write_text('{"version": "4.0.0"}', encoding="utf-8")
    return str(p)


class _Base(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self._env = {k: os.environ.get(k) for k in MODES}
        for k in MODES:
            os.environ.pop(k, None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.log = str(Path(self.tmp) / "rf_calls.log")
        os.environ["FAKE_RF_CALLS"] = self.log
        os.environ["FAKE_RF_PIN"] = str(PINNED_CONTRACT_PATH)

    def tearDown(self):
        clear_modes()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def rf_calls(self, name):
        return [json.loads(x)[1:] for x in Path(self.log).read_text(encoding="utf-8").splitlines() if json.loads(x)[0] == name] if os.path.exists(self.log) else []


class ContractTests(_Base):
    def test_pin_and_checks(self):
        c = pinned_contract()
        self.assertEqual(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"), json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n", "pinned from the v0.14.0 tag")
        self.assertEqual((c["skill_id"], c["version"], c["contract_version"]), ("remotion-ffmpeg-skill", "0.14.0", "1.0"))
        self.assertEqual((check_contract(c), drift(c), breaking_drift(c)), ([], [], []))
        self.assertEqual((recovery_class("missing_tool"), recovery_class("timeout"), recovery_class("other")), ("TOOL_MISSING", "TIMEOUT", "SKILL_ERROR"))
        bad = json.loads(json.dumps(c)); next(t for t in bad["tools"] if t["name"] == "pipeline")["flags"].pop("out_dir")
        self.assertTrue(any("out_dir" in e for e in check_contract(bad)))
        self.assertTrue(breaking_drift(bad))
        for v in ("0.15.0", "0.140.0", "1.0.0"):
            bad = json.loads(json.dumps(c)); bad["version"] = bad["skill"]["version"] = v
            self.assertTrue(any("supported range" in e for e in check_contract(bad)), v)
        # ADR-107 addendum 14: a 0.12.x install (no Emphasis in the comp descriptions) is inside the gate and is drift, not breaking drift
        old = json.loads(json.dumps(c)); old["version"] = old["skill"]["version"] = "0.12.1"
        for t in old["tools"]:
            comp = t["inputSchema"]["properties"].get("comp")
            if comp:
                comp["description"] = comp["description"].replace(", Emphasis", "")
        self.assertEqual((check_contract(old), breaking_drift(old)), ([], []))
        self.assertTrue(drift(old))
        analyze = next(t for t in c["tools"] if t["name"] == "analyze")
        self.assertEqual((analyze["flags"]["emphasis"], analyze["flags"]["emphasis_auto"]), ("--emphasis", "--emphasis-auto"), "0.13.0's analyze options")
        self.assertIn("Emphasis", next(t for t in c["tools"] if t["name"] == "pipeline")["inputSchema"]["properties"]["comp"]["description"])
        self.assertNotIn("Emphasis", COMPOSITIONS, "the Emphasis composition is pinned, not planned")
        self.assertNotIn("emphasis", SENT)

    def test_locate_needs_the_scripts(self):
        sk = fake_checkout(self.tmp)
        self.assertEqual(locate_remotion(str(sk.root), env={}).root, sk.root)
        self.assertIsNone(locate_remotion(env={"VIDEO_AGENT_REMOTION_DIR": str(Path(self.tmp) / "nothing")}))


class CapabilityTests(_Base):
    def test_doctor(self):
        caps = rf_capabilities(self.tmp)
        self.assertEqual((caps["remotion-ffmpeg-skill"].status, caps["remotion-ffmpeg-skill:PIPELINE"].status), ("AVAILABLE", "AVAILABLE"))
        os.environ["FAKE_RF_READY"] = "nonode"
        caps = rf_capabilities(self.tmp)
        self.assertEqual(caps["remotion-ffmpeg-skill"].status, "MISSING")


class AdapterTests(_Base):
    def run_op(self, args, src=None):
        ad = rf_adapter(self.tmp)
        src = src or fake_media(self.tmp, duration=12.0)
        comp = args.get("comp", "Captioned")
        out = str(Path(ad.workspace) / "jobs" / "j" / "ops" / "m" / render_name(src, comp))
        return ad, src, out, ad.run(Operation(tool="remotion-ffmpeg-skill/pipeline", args=dict({"input": "src", "comp": comp, "project": project(self.tmp)}, **args, output="out"),
                                              inputs=["src"], outputs=["out"]), {"src": src, "out": out})

    def test_argv_and_result(self):
        cap = Path(self.tmp) / "src" / "talk.srt"; cap.parent.mkdir(parents=True, exist_ok=True); cap.write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n")
        ad, src, out, r = self.run_op({"captions": str(cap), "beats": True, "fast": False})
        self.assertTrue(r.ok, r.data)
        self.assertEqual(self.rf_calls("pipeline"), [[src, f"--project={project(self.tmp)}", "--comp=Captioned", f"--captions={cap}", "--beats", f"--out-dir={os.path.dirname(out)}", "--json",
                                                      "--timeout=1800"]])
        self.assertEqual((r.data["verified"], r.data["comp"], [s["step"] for s in r.data["steps"]]), (True, "Captioned", ["measure", "draw"]))

    def test_refused_before_a_process(self):
        for args in ({"comp": "Jumpcut"}, {"comp": "Nope"}, {"comp": "Emphasis"}, {"emphasis": "[\"x\"]"}, {"platform": "youtube"}, {"aspect": "9:16"}, {"brand": "b.json"}, {"captions": "/etc/passwd"}):
            with self.subTest(args=args):
                _, _, _, r = self.run_op(args)
                self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_REQUEST"))
        self.assertEqual(self.rf_calls("pipeline"), [])

    def test_results_refused(self):
        for mode, why in (("unverified", "not verified"), ("elsewhere", "planned")):
            with self.subTest(mode=mode):
                os.environ["FAKE_RF_MODE"] = mode
                _, _, _, r = self.run_op({})
                self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_OUTPUT"))
                self.assertIn(why, r.data["error"]["message"])
        os.environ["FAKE_RF_MODE"] = "fail:missing_tool"
        _, _, _, r = self.run_op({})
        self.assertEqual((r.ok, r.data["error"]["code"], r.data["error"]["recovery_class"], r.exit_code), (False, "missing_tool", "TOOL_MISSING", 3))


class RequirementTests(_Base):
    def test_vocabulary_and_refusals(self):
        p = project(self.tmp)
        self.assertEqual(parse_remotion_requirements({"remotion.comp": "PeakPulse", "remotion.project": p, "remotion.beats": "true"}), {"comp": "PeakPulse", "project": p, "beats": True})
        for reqs, why in (({"remotion.project": p}, "not set"), ({"remotion.comp": "Jumpcut", "remotion.project": p}, "cuts the picture"),
                          ({"remotion.comp": "Emphasis", "remotion.project": p}, "must be one of"),
                          ({"remotion.comp": "Captioned", "remotion.project": self.tmp}, "package.json"), ({"remotion.comp": "Captioned", "remotion.project": p, "remotion.aspect": "9:16"}, "own frame"),
                          ({"remotion.comp": "Captioned", "remotion.project": p, "remotion.captions": "/nope.srt"}, "existing file"),
                          ({"remotion.comp": "Captioned", "remotion.project": p, "remotion.platform": "youtube"}, "unknown requirement key")):
            with self.subTest(reqs=reqs):
                with self.assertRaises(EditRequirementError) as cm:
                    parse_remotion_requirements(reqs)
                self.assertIn(why, str(cm.exception))
                with self.assertRaises(ValueError):
                    _check_requirement_keys(reqs)


class PlanTests(_Base):
    def plan(self, reqs=None, installed=True, project_installed=True):
        self.svc = rf_service(self.tmp, installed=installed)
        self.media = fake_media(self.tmp, duration=16.0)
        return self.svc.plan([self.media], "generic", user_requirements=dict({"remotion.comp": "Titled", "remotion.project": project(self.tmp, project_installed)}, **(reqs or {})))

    def test_confirm_step_after_the_edits_then_render(self):
        ir = self.plan({"edit.rotate": 180})
        d = ir.doc
        [dec] = [x for x in d["decisions"] if x["subject"] == "video.measured_motion"]
        self.assertEqual((dec["type"], dec["approval"], dec["basis"]["approval"]["key"]), ("TRANSFORM", "CONFIRM", "video.measured_motion.approval"))
        skills = [s["skill"] for s in d["plan"]["steps"]]
        self.assertLess(skills.index("video_rotate"), skills.index("measured_motion"), "drawn on the edited picture")
        step = next(s for s in d["plan"]["steps"] if s["skill"] == "measured_motion")
        self.assertEqual((step["tool"], step["params"]["project"]), ("remotion-ffmpeg-skill/pipeline", "motion"), "the user's project by name in the plan")
        self.assertEqual(self.svc.validate(ir).errors, [])
        self.assertIn(str(Path(self.tmp) / "motion"), d["execution"]["allowed_inputs"])
        p = str(Path(self.tmp) / "p.json"); save_ir(ir, p)
        out = self.svc.render(load_ir(p), p)
        self.assertNotEqual(out["status"], "COMPLETED")
        self.assertEqual(self.rf_calls("pipeline"), [], "nothing is drawn before the approval")
        out = self.svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED", out)
        [argv] = self.rf_calls("pipeline")
        self.assertTrue(argv[0].endswith("_rotate.mp4"), argv)
        self.assertIn("--comp=Titled", argv)

    def test_project_not_installed_blocks(self):
        ir = self.plan(project_installed=False)
        [blk] = [x for x in ir.doc["decisions"] if x["subject"] == "video.measured_motion"]
        self.assertEqual(blk["type"], "BLOCK")
        self.assertIn("npm install", blk["reason"])

    def test_without_the_skill_the_step_is_handed_off(self):
        ir = self.plan(installed=False)
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        [h] = [h for h in self.svc.handoffs(ir) if h["skill"] == "measured_motion"]
        self.assertEqual((h["provided_by"], h["repositories"]), (["remotion-ffmpeg-skill"], ["remotion-ffmpeg-skill"]))
        self.assertIn("npx remotion-ffmpeg-skill", h["install"])


if __name__ == "__main__":
    unittest.main()
