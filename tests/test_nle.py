"""ADR-089 (nle-bridge-skill roadmap step 7, agent side: the adapter): `delivery.nle=export|conform` hands the project IR the job executes to
nle-bridge, which reads the subject's timeline record itself and writes OTIO / FCP7 XML / EDL / SRT (or a conform package). Against a fake
`nle-bridge` process (tests/fake_nle_bridge.py) that reads the IR the same way; the real Skill is exercised by NleBridgeRealTests.
ADR-098 (the second half: the editor's timeline read back): `revise --editor-timeline` runs `nle-bridge diff --timeline-out` and the changed cuts
become CONFIRM `timeline.editor_cut` decisions that the next render cuts exactly (EditorReadBackTests)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import PipelineCaps, clear_modes, fake_media, pipeline_service, plan_and_render  # noqa: E402

from video_agent.project import load_ir  # noqa: E402

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


def returned(tmp: str, name: str, clips, rate: str = "30", **extra) -> str:
    """A returned timeline for the fake nle-bridge: clips of (asset, source_in, source_out) frames laid end to end on V1, or full dicts."""
    out, pos = [], 0
    for c in clips:
        if isinstance(c, dict):
            out.append(c)
            pos = c["record_out"]
            continue
        a, s_, e_ = c
        out.append({"asset": a, "source_in": s_, "source_out": e_, "record_in": pos, "record_out": pos + e_ - s_})
        pos += e_ - s_
    p = Path(tmp) / "editor" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"rate": rate, "clips": out, **extra}), encoding="utf-8")
    return str(p)


class EditorReadBackTests(_Base):
    def _planned(self, svc, name="p"):
        r = plan_and_render(svc, [fake_media(self.tmp, "talk.mp4")], {"delivery.timeline": "otio", "delivery.nle": "export", "delivery.nle.formats": "otio,fcp7"},
                            profile="youtube", name=name)
        self.assertEqual(r["out"]["execution"]["status"], "COMPLETED", r["out"]["execution"])
        return r

    @staticmethod
    def frames(ir):
        [rec] = ir.doc["delivery"]["timelines"]
        return rec, [(c["asset"], int(round(c["in"] * 30)), int(round(c["out"] * 30))) for c in rec["clips"]]

    def test_diff_argv_and_admission(self):
        ws = str(Path(self.tmp) / "ws"); os.makedirs(ws, exist_ok=True)
        ad = NleBridgeAdapter(fake_skill(), workspace=ws)
        self.assertTrue(ad.supports("nle-bridge/diff"))
        proj = str(Path(ws) / "project.json")
        Path(proj).write_text(json.dumps({"assets": {"a": {"path": "/m/a.mov"}}, "delivery": {"timelines": [{"asset": "a", "clips": [{"asset": "a", "in": 0, "out": 2}]}]}}),
                              encoding="utf-8")
        ret = returned(self.tmp, "back.otio", [("a", 0, 30)])
        args = {"project": proj, "returned": ret, "subject": "a", "output_dir": str(Path(ws) / "editor" / "x")}
        r = ad.measure("nle-bridge/diff", args)
        self.assertTrue(r.ok, r.data)
        c = self.calls()[-1]
        self.assertEqual(c[0], "diff")
        self.assertEqual(c[c.index("--") + 1:], [os.path.realpath(proj), os.path.realpath(ret)])
        self.assertTrue(all(a.startswith("--") for a in c[1:c.index("--")]))
        self.assertEqual((r.data["diff"]["summary"]["clips_changed"], r.data["timeline"]["schema"]), (1, "nle-bridge/timeline@1"))
        self.assertEqual(r.data["returned"]["format"], "otio")
        for mode, code in (("diff_elsewhere", "INVALID_OUTPUT"), ("fail:INVALID_TIMELINE", "INVALID_TIMELINE")):
            os.environ["FAKE_NLE_MODE"] = mode
            self.assertEqual(ad.measure("nle-bridge/diff", args).data["error"]["code"], code, mode)
        os.environ.pop("FAKE_NLE_MODE")
        bad_ext = str(Path(self.tmp) / "back.aaf"); Path(bad_ext).write_text("x", encoding="utf-8")
        for more, msg in (({"returned": bad_ext}, ".otio / .xml / .edl"), ({"output_dir": str(Path(self.tmp) / "elsewhere")}, "outside the workspace"),
                          ({"subject": "../a"}, "is not an id"), ({"shell": "x"}, "forbidden")):
            self.assertIn(msg, ad.measure("nle-bridge/diff", dict(args, **more)).data["error"]["message"], more)

    def test_editor_cut_becomes_the_next_version_and_render(self):
        svc = nle_service(self.tmp)
        r = self._planned(svc)
        rec, plan = self.frames(r["ir"])
        [(a, s0, e0)] = plan
        # the editor razor-cuts the clip and lifts two pieces out: three kept ranges (a split + removals)
        three = [(a, s0, s0 + 60), (a, s0 + 90, s0 + 150), (a, s0 + 180, e0 - 15)]
        out = svc.revise(load_ir(r["path"]), r["path"], editor_timeline=returned(self.tmp, "v1_back.otio", three), who="editor-test")
        self.assertTrue(out["created"], out)
        self.assertEqual((out["version"], out["editor"]["changed_sources"]), (2, [a]))
        ir2 = load_ir(r["path"])
        [obs] = [o for o in ir2.doc["analysis"]["observations"] if o["kind"] == "nle.diff"]
        self.assertEqual((obs["asset_id"], obs["tool"], obs["provenance"]), (a, "nle-bridge/diff", "OBSERVED"))
        [dec] = [d for d in ir2.doc["decisions"] if d["subject"] == "timeline.editor_cut"]
        self.assertEqual((dec["type"], dec["approval"], dec["status"]), ("TRANSFORM", "CONFIRM", "PROPOSED"), "an editor's change is confirmed, never applied silently")
        self.assertIn(obs["id"], dec["evidence"])
        want = [[round(x / 30, 3), round(y / 30, 3)] for _a, x, y in three]
        self.assertEqual(dec["params"]["keep"], want)
        self.assertEqual(ir2.doc["revision"]["history"][-1]["editor_timeline"]["observation_id"], obs["id"])
        self.assertEqual(svc.validate(ir2).errors, [])
        out2 = svc.render(load_ir(r["path"]), r["path"], approve=["all"])
        self.assertEqual(out2["execution"]["status"], "COMPLETED", out2["execution"])
        ir2 = load_ir(r["path"])
        [trim] = [op for op in ir2.doc["video"]["operations"] if op["type"] == "video.trim"]
        self.assertEqual((trim["keep"], trim["accurate"], trim["decision_ids"]), (want, True, [dec["id"]]))
        step = next(s for s in ir2.doc["plan"]["steps"] if s["skill"] in ("silence_cleanup", "video_trim"))
        self.assertEqual(step["params"]["keep"], want)
        self.assertEqual(self.frames(ir2)[1], three, "the next timeline export has the editor's cuts")
        # v2 back from the editor: the first clip trimmed by 10 frames, the second removed
        two = [(a, s0, s0 + 50), (a, s0 + 180, e0 - 15)]
        back = returned(self.tmp, "v2_back.otio", two)
        out3 = svc.revise(load_ir(r["path"]), r["path"], editor_timeline=back)
        self.assertTrue(out3["created"], out3)
        ir3 = load_ir(r["path"])
        decs = [d for d in ir3.doc["decisions"] if d["subject"] == "timeline.editor_cut"]
        self.assertEqual([d["params"]["keep"] for d in decs], [[[round(x / 30, 3), round(y / 30, 3)] for _a, x, y in two]], "only the latest read-back decides")
        self.assertTrue(any("removed" in c for c in [decs[0]["reason"]]))
        self.assertEqual(len([o for o in ir3.doc["analysis"]["observations"] if o["kind"] == "nle.diff"]), 2, "earlier read-backs stay as history")
        svc.render(load_ir(r["path"]), r["path"], approve=["all"])
        ir3 = load_ir(r["path"])
        self.assertEqual(self.frames(ir3)[1], two)
        # the same file again: nothing to revise
        again = svc.revise(load_ir(r["path"]), r["path"], editor_timeline=back)
        self.assertEqual((again["created"], again["version"]), (False, 3))
        self.assertIn("plan's cuts", again["reason"])
        # rejecting the editor's cut restores the plan's own cut in the next version
        svc.reject(load_ir(r["path"]), r["path"], [decs[0]["id"]], reason="keep the agent's cut")
        svc.revise(load_ir(r["path"]), r["path"])
        ir4 = load_ir(r["path"])
        self.assertFalse([d for d in ir4.doc["decisions"] if d["subject"] == "timeline.editor_cut" and d["status"] != "REJECTED"])
        self.assertEqual(self.frames(ir4)[1], plan)

    def test_markers_and_warnings(self):
        svc = nle_service(self.tmp)
        r = self._planned(svc)
        _rec, [(a, s0, e0)] = self.frames(r["ir"])
        back = returned(self.tmp, "m.otio", [(a, s0, e0 - 30)], markers=[{"name": "fix the logo", "at_frame": 12, "comment": "", "color": "RED"}],
                        tracks_extra=[{"kind": "audio", "name": "A1", "items": [{"asset": a, "source_in": s0 + 5, "source_out": e0 - 30, "record_in": 5, "record_out": e0 - s0 - 30}]}],
                        not_imported=[{"code": "EFFECT_NOT_IMPORTED", "message": "a colour effect on c1"}])
        out = svc.revise(load_ir(r["path"]), r["path"], editor_timeline=back)
        codes = [w["code"] for w in out["editor"]["warnings"]]
        self.assertIn("AUDIO_FOLLOWS_PICTURE", codes)
        self.assertIn("EFFECT_NOT_IMPORTED", codes)
        ir = load_ir(r["path"])
        [m] = [d for d in ir.doc["decisions"] if d["subject"] == "timeline.editor_markers"]
        self.assertEqual((m["type"], m["approval"], m["params"]["markers"]["added"][0]["name"]), ("REVIEW", "CONFIRM", "fix the logo"))

    def test_refusals_leave_the_plan_as_it_was(self):
        svc = nle_service(self.tmp)
        r = self._planned(svc)
        _rec, [(a, s0, e0)] = self.frames(r["ir"])
        before = Path(r["path"]).read_text(encoding="utf-8")
        cases = [
            ([("new", 0, 30)], "media the plan does not cut"),
            ([(a, s0 + 60, s0 + 90), (a, s0, s0 + 30)], "out of source order"),
            ([{"asset": a, "source_in": s0, "source_out": s0 + 60, "record_in": 0, "record_out": 30}], "speed change"),
            ([{"asset": a, "source_in": s0, "source_out": s0 + 30, "record_in": 0, "record_out": 30, "kind": "title"}], "title or generator"),
        ]
        for clips, msg in cases:
            with self.assertRaises(ValueError, msg=msg) as cm:
                svc.revise(load_ir(r["path"]), r["path"], editor_timeline=returned(self.tmp, "bad.otio", clips))
            self.assertIn(msg, str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            svc.revise(load_ir(r["path"]), r["path"], editor_timeline=returned(self.tmp, "v2.otio", [(a, s0, e0)], tracks_extra=[
                {"kind": "video", "name": "V2", "items": [{"asset": a, "source_in": 0, "source_out": 10, "record_in": 0, "record_out": 10, "enabled": True}]}]))
        self.assertIn("second picture track", str(cm.exception))
        self.assertEqual(Path(r["path"]).read_text(encoding="utf-8"), before, "a refused read-back writes nothing")
        plain = plan_and_render(svc, [fake_media(self.tmp, "talk.mp4")], {}, profile="youtube", name="plain")
        with self.assertRaises(ValueError) as cm:
            svc.revise(load_ir(plain["path"]), plain["path"], editor_timeline=returned(self.tmp, "x.otio", [(a, 0, 30)]))
        self.assertIn("delivery.timeline=otio", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            nle_service(self.tmp, installed=False).revise(load_ir(r["path"]), r["path"], editor_timeline=returned(self.tmp, "x.otio", [(a, s0, e0)]))
        self.assertIn("needs nle-bridge-skill", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
