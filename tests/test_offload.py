"""ADR-066 (roadmap step 9, the ingest-skill adapter): a card folder is planned as a verified offload to the destinations the user names,
confirmed, then run through ingest-skill; the copy's inventory becomes new assets with OBSERVED observations and provenance. Against a fake
`ingest` process (tests/fake_ingest.py) that copies for real and speaks ingest-skill main's report shapes; the fake media engine analyses
the copies. No ingest-skill, ffmpeg or network needed."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_adapter import FakeAdapter  # noqa: E402
from test_unit import FakeCaps  # noqa: E402
import fake_ingest  # noqa: E402

from video_agent.agent import offload as offload_task  # noqa: E402
from video_agent.capabilities.resolver import CapabilityResolver  # noqa: E402
from video_agent.models import Operation  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.service import Service  # noqa: E402
from video_agent.tools import ToolRouter  # noqa: E402
from video_agent.tools.base import ToolError  # noqa: E402
from video_agent.tools.ingest import (ERROR_KINDS, FATAL_DRIFT_KEYS, ContractError, IngestAdapter, check_contract, kind_of, locate_ingest,  # noqa: E402
                                      pinned_contract)
from video_agent.tools.ingest.adapter import PINNED_CONTRACT_PATH, offload_argv  # noqa: E402
from video_agent.tools.skill_process import CliSkill  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_ingest.py")
MODES = ("FAKE_INGEST_MODE", "FAKE_INGEST_CALLS", "FAKE_INGEST_OFFLOAD", "FAKE_INGEST_CONTRACT", "FAKE_INGEST_XXHASH", "VIDEO_AGENT_INGEST_DIR")


def fake_skill() -> CliSkill:
    return CliSkill("ingest", [sys.executable, FAKE], None, {})


def make_card(base: Path, name: str = "EX_CARD") -> Path:
    """An XDCAM EX-like card: take 709_0001 spanned over two clip folders, take 709_0002 in one, M01.XML sidecars, MEDIAPRO.XML.
    The essence files are self-describing fake media the fake engine probes (16 s each)."""
    card = base / name
    for clip in ("709_0001_01", "709_0001_02", "709_0002_01"):
        d = card / "BPAV" / "CLPR" / clip
        d.mkdir(parents=True)
        (d / f"{clip}.MP4").write_bytes(json.dumps({"fake": True, "duration": 16.0, "name": clip}).encode())
        (d / f"{clip}M01.XML").write_text("<NonRealTimeMeta/>", encoding="utf-8")
    (card / "BPAV" / "MEDIAPRO.XML").write_text("<MediaProfile/>", encoding="utf-8")
    return card


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self._env = {k: os.environ.get(k) for k in MODES}
        for k in MODES:
            os.environ.pop(k, None)
        self.log = str(Path(self.tmp) / "calls.log")
        os.environ["FAKE_INGEST_CALLS"] = self.log
        self.card = make_card(Path(self.tmp))
        self.a, self.b = Path(self.tmp) / "SSD1", Path(self.tmp) / "SSD2"
        self.a.mkdir()
        self.b.mkdir()

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def svc(self, ingest: bool = True) -> Service:
        adapters = [FakeAdapter()] + ([IngestAdapter(fake_skill())] if ingest else [])
        return Service(workspace=str(Path(self.tmp) / "ws"), adapter=ToolRouter(adapters), caps=FakeCaps(extra=["ingest"] if ingest else []))

    def reqs(self, **extra):
        return {"ingest.dest": [str(self.a), str(self.b)], "ingest.project": "event", **extra}

    def calls(self):
        return [json.loads(x) for x in Path(self.log).read_text(encoding="utf-8").splitlines()] if Path(self.log).exists() else []

    def offload_calls(self):
        return [c for c in self.calls() if c[:1] == ["offload"]]

    def planned(self, svc=None, **extra):
        svc = svc or self.svc()
        ir = svc.plan_offload(str(self.card), self.reqs(**extra))
        path = str(Path(self.tmp) / "off.project.json")
        save_ir(ir, path)
        return svc, ir, path

    def approve_and_render(self, svc, path, **kw):
        ir = load_ir(path)
        svc.approve(ir, path, [d["id"] for d in ir.pending_confirmations()])
        return svc.render(load_ir(path), path, **kw)


class OffloadPlanTests(_Base):
    def test_destinations_must_be_named_and_real(self):
        svc = self.svc()
        bad = [({}, "name every destination"), ({"ingest.dest": "relative/dir"}, "not an absolute path"),
               ({"ingest.dest": str(Path(self.tmp) / "NOT_MOUNTED")}, "is the drive mounted"),
               ({"ingest.dest": [str(self.a), str(self.a)]}, "given twice"), ({"ingest.dest": str(self.card / "BPAV")}, "contain one another"),
               ({"ingest.dest": str(self.a), "ingest.hash": "md5"}, "ingest.hash"), ({"ingest.dest": str(self.a), "ingest.name_by": "date"}, "ingest.name_by"),
               ({"ingest.dest": str(self.a), "ingest.reel": "A 001"}, "ingest.reel"), ({"ingest.dest": str(self.a), "ingest.project": "../up"}, "ingest.project"),
               ({"ingest.dest": str(self.a), "edit.fit": "9:16"}, "takes only"), ({"ingest.dest": str(self.a), "ingest.flatten": True}, "takes only")]
        for reqs, frag in bad:
            with self.assertRaises(ValueError, msg=reqs) as cm:
                svc.plan_offload(str(self.card), reqs)
            self.assertIn(frag, str(cm.exception), reqs)
        self.assertEqual(self.offload_calls(), [], "refused before ingest-skill is called")
        self.assertEqual((list(self.a.iterdir()), list(self.b.iterdir())), ([], []))
        with self.assertRaises(ValueError) as cm:   # ingest.* on a media plan is refused too: the offload is its own plan
            svc.plan([str(self.card / "BPAV" / "CLPR" / "709_0002_01" / "709_0002_01.MP4")], "youtube", user_requirements={"ingest.dest": str(self.a)})
        self.assertIn("card folder", str(cm.exception))

    def test_plan_reads_the_card_and_writes_nothing(self):
        svc, ir, path = self.planned()
        d = ir.doc
        self.assertEqual((d["project"]["kind"], d["plan"]["status"]), ("offload", "REVIEW"))
        self.assertEqual((list(self.a.iterdir()), list(self.b.iterdir())), ([], []), "a plan copies nothing")
        offs = self.offload_calls()
        self.assertEqual(len(offs), 1)
        self.assertIn("--dry-run", offs[0])
        self.assertTrue(any(c[:1] == ["inventory"] and c[1] == str(self.card) for c in self.calls()))
        [card] = d["assets"].values()
        self.assertEqual((card["path"], card["status"]), (str(self.card), "source_card"))
        kinds = {o["kind"]: o for o in d["analysis"]["observations"]}
        self.assertEqual(set(kinds), {"ingest_offload_plan", "ingest_card_inventory"})
        plan = kinds["ingest_offload_plan"]
        self.assertEqual((plan["source"], plan["provenance"], plan["data"]["files"]), ("ingest/offload@0.1.0", "OBSERVED", 7))
        self.assertEqual(plan["data"]["roots"], [str(self.a / "event" / "EX_CARD"), str(self.b / "event" / "EX_CARD")])
        inv = kinds["ingest_card_inventory"]["data"]
        self.assertEqual((inv["clip_count"], inv["spanned_count"], inv["file_count"]), (2, 1, 3))
        [dec] = d["decisions"]
        self.assertEqual((dec["subject"], dec["type"], dec["approval"], dec["status"], dec["risk"], dec["provenance"]),
                         ("ingest.offload", "TRANSFORM", "CONFIRM", "PROPOSED", "MEDIUM", "USER"))
        self.assertEqual(dec["basis"]["approval"]["resolved"], "CONFIRM")
        self.assertTrue(set(o["id"] for o in d["analysis"]["observations"]) <= set(dec["evidence"]))
        [step] = d["plan"]["steps"]
        self.assertEqual((step["id"], step["skill"], step["tool"], step["status"]), ("step_media_offload", "media_offload", "ingest/offload", "PROPOSED"))
        self.assertEqual(step["params"]["destinations"], [str(self.a), str(self.b)])
        self.assertIn("are on the same volume", " ".join(d["analysis"]["warnings"]), "two copies on one drive are not a backup")
        rep = svc.validate(load_ir(path))
        self.assertTrue(rep.ok, rep.errors)
        self.assertEqual(svc.handoffs(ir), [])

    def test_never_auto_and_nothing_copied_before_approval(self):
        svc, ir, path = self.planned()
        out = svc.render(load_ir(path), path)
        self.assertEqual(out["status"], "WAITING_FOR_APPROVAL")
        self.assertEqual(len(self.offload_calls()), 1, "only the plan's dry run")
        self.assertEqual(list(self.a.iterdir()), [])
        tampered = load_ir(path)
        tampered.doc["decisions"][0]["approval"] = "AUTO"   # an IR edited to skip the confirmation is refused (ADR-033)
        self.assertTrue(any("never lowered after planning" in e for e in svc.validate(tampered).errors))
        dry = svc.dry_run(load_ir(path))
        self.assertIn("ingest offload", dry["operations"][0]["command"][0])
        self.assertIn("--dest", dry["operations"][0]["command"][0])
        self.assertEqual(dry["pending_confirmations"], [ir.doc["decisions"][0]["id"]])
        self.assertEqual(len(self.offload_calls()), 1, "a dry run never calls the copy")
        with self.assertRaises(ValueError):
            svc.revise(load_ir(path), path, feedback="other drive")

    def test_without_ingest_skill_the_step_is_handed_off(self):
        svc = self.svc(ingest=False)
        ir = svc.plan_offload(str(self.card), self.reqs())
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        self.assertEqual(ir.doc["plan"]["steps"][0]["tool"], None)
        [h] = svc.handoffs(ir)
        self.assertEqual((h["skill"], h["provided_by"], h["repositories"]), ("media_offload", ["ingest"], ["ingest-skill"]))
        self.assertIn("required capability missing: ingest", h["reason"])
        self.assertIn("git clone https://github.com/kajisho5/ingest-skill && pip install -e 'ingest-skill[fetch]'", h["install"])
        self.assertEqual(self.calls(), [])

    def test_cli_plans_a_card_and_hands_over_without_ingest(self):
        from video_agent.cli import _kv, cmd_plan
        self.assertEqual(_kv([f"ingest.dest={self.a}", f"ingest.dest={self.b}"])["ingest.dest"], [str(self.a), str(self.b)])
        self.assertEqual(_kv([f"ingest.dest={self.a},{self.b}"])["ingest.dest"], f"{self.a},{self.b}")
        ns = dict(inputs=[str(self.card)], json=True, ai_brief=False, ai_model=None, brief=None, request=None, name=None, output=None,
                  set=[f"ingest.dest={self.a}", f"ingest.dest={self.b}"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(cmd_plan(argparse.Namespace(**ns), self.svc()), 0)
        doc = json.loads(buf.getvalue())
        self.assertEqual((doc["kind"], doc["handoff"]), ("offload", []))
        self.assertIn("video-agent approve", doc["next"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(cmd_plan(argparse.Namespace(**dict(ns, json=False)), self.svc(ingest=False)), 3)
        self.assertIn("install: git clone https://github.com/kajisho5/ingest-skill", buf.getvalue())
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(cmd_plan(argparse.Namespace(**dict(ns, set=[])), self.svc()), 2)
        self.assertIn("name every destination", err.getvalue())


class OffloadRunTests(_Base):
    def test_approved_offload_copies_and_lifts_the_inventory_into_assets(self):
        svc, ir, path = self.planned()
        out = self.approve_and_render(svc, path)
        self.assertEqual(out["status"], "COMPLETED", json.dumps(out.get("offload"), default=str)[:800])
        argv = self.offload_calls()[-1]
        self.assertEqual(argv, ["offload", str(self.card), "--dest", str(self.a), "--dest", str(self.b), "--project", "event", "--hash", "sha256", "--name-by", "folder", "--json"])
        for root in (self.a, self.b):
            self.assertTrue((root / "event" / "EX_CARD" / "ingest-manifest.json").is_file())
        d = load_ir(path).doc
        off = d["execution"]["offload"]
        self.assertEqual((off["status"], off["clip_count"], off["spanned_count"], len(off["asset_ids"])), ("COMPLETED", 2, 1, 3))
        self.assertTrue(Path(off["report"]).is_file())
        new = [a for a in d["assets"].values() if a["status"] == "ingested"]
        self.assertEqual(sorted(Path(a["path"]).name for a in new), ["709_0001_01.MP4", "709_0001_02.MP4", "709_0002_01.MP4"])
        self.assertTrue(all(str(self.a) in a["path"] for a in new), "the first destination's copies are the assets")
        man = json.loads((self.a / "event" / "EX_CARD" / "ingest-manifest.json").read_text(encoding="utf-8"))
        sha = {e["path"]: e["sha256"] for e in man["files"]}
        clip_obs = [o for o in d["analysis"]["observations"] if o["kind"] == "ingest_clip"]
        self.assertEqual(len(clip_obs), 3)
        for o in clip_obs:
            self.assertEqual((o["provenance"], o["source"], o["skill"], o["tool"]), ("OBSERVED", "ingest/inventory@0.1.0", "ingest", "ingest/inventory"))
            a = d["assets"][o["asset_id"]]
            self.assertEqual(a["hash"], sha[o["data"]["path"]], "the agent's own sha256 equals ingest-skill's")
            self.assertEqual(o["fingerprint"], a["hash"])
            self.assertEqual([c["root"] for c in o["data"]["copies"]], [str(self.a / "event" / "EX_CARD"), str(self.b / "event" / "EX_CARD")])
            self.assertTrue(all(c["manifest"] and c["mhl"] for c in o["data"]["copies"]), "manifest and ASC MHL generation per destination")
            self.assertEqual((o["data"]["ingest_version"], o["data"]["job_id"], o["data"]["hash_algorithm"]), ("0.1.0", out["job"]["id"], "sha256"))
            self.assertEqual(o["data"]["reel"]["source"], "manifest")
        spanned = sorted((o for o in clip_obs if o["data"]["clip_id"] == "709_0001"), key=lambda o: o["data"]["member_index"])
        self.assertEqual([(o["data"]["member_index"], o["data"]["member_count"], o["data"]["spanned"]) for o in spanned], [(0, 2, True), (1, 2, True)])
        self.assertEqual(spanned[0]["data"]["sidecars"][0]["path"], "BPAV/CLPR/709_0001_01/709_0001_01M01.XML")
        rel = d["assets"][spanned[0]["asset_id"]]["relationships"]
        self.assertEqual(rel, [{"kind": "spanned_member", "target": spanned[1]["asset_id"], "metadata": {"clip_id": "709_0001", "index": 1}}])
        probed = {o["asset_id"] for o in d["analysis"]["observations"] if o["kind"] == "media_probe"}
        self.assertEqual(probed, set(off["asset_ids"]), "the agent's analysis ran on the verified copies")
        self.assertTrue(svc.validate(load_ir(path)).ok, svc.validate(load_ir(path)).errors)
        # a second render does not copy again
        n = len(self.offload_calls())
        again = svc.render(load_ir(path), path)
        self.assertEqual((again["status"], len(self.offload_calls())), ("BLOCKED", n))
        self.assertIn("was offloaded by job", again["hint"])

    def test_name_by_reel_and_xxh64(self):
        os.environ["FAKE_INGEST_XXHASH"] = "1"
        svc, ir, path = self.planned(**{"ingest.name_by": "reel", "ingest.reel": "A001", "ingest.hash": "xxh64"})
        self.assertEqual(ir.doc["plan"]["steps"][0]["params"]["roots"], [str(self.a / "event" / "A001"), str(self.b / "event" / "A001")])
        out = self.approve_and_render(svc, path)
        self.assertEqual(out["status"], "COMPLETED")
        self.assertIn("--reel", self.offload_calls()[-1])
        o = next(o for o in load_ir(path).doc["analysis"]["observations"] if o["kind"] == "ingest_clip")
        self.assertEqual((o["data"]["hash_algorithm"], o["data"]["hash"]["algorithm"], o["data"]["reel"]["id"]), ("xxh64", "xxh64", "A001"))

    def test_xxh64_without_the_extra_is_refused_at_plan_time(self):
        with self.assertRaises(ValueError) as cm:
            self.svc().plan_offload(str(self.card), self.reqs(**{"ingest.hash": "xxh64"}))
        self.assertIn("HASH_UNAVAILABLE/ENGINE_MISSING", str(cm.exception))
        self.assertIn("xxhash", str(cm.exception))

    def _failed(self, mode, status="FAILED", **kw):
        svc, ir, path = self.planned()
        os.environ["FAKE_INGEST_OFFLOAD"] = mode
        out = self.approve_and_render(svc, path, **kw)
        self.assertEqual(out["status"], status, mode)
        d = load_ir(path).doc
        self.assertFalse([a for a in d["assets"].values() if a["status"] == "ingested"], f"{mode}: no asset from an unverified offload")
        self.assertFalse([o for o in d["analysis"]["observations"] if o["kind"] == "ingest_clip"])
        self.assertEqual(len([c for c in self.offload_calls() if "--dry-run" not in c]), 1, f"{mode}: never re-run automatically")
        return out, d["execution"]["offload"]

    def test_incomplete_reports_keep_every_destination_and_code(self):
        out, off = self._failed("partial")
        err = off["error"]
        self.assertEqual((err["code"], err["kind"], err["recovery_class"], err["exit_code"]), ("INCOMPLETE", "PARTIAL", "SKILL_ERROR", 3))
        self.assertTrue(err["retryable"], "ingest-skill's verdict kept")
        det = err["details"]
        self.assertEqual((det["complete"], det["incomplete"]), ([str(self.a / "event" / "EX_CARD")], [str(self.b / "event" / "EX_CARD")]))
        bad = next(x for x in det["destinations"] if not x["ok"])
        self.assertEqual(bad["failed"], 7)
        self.assertEqual(len(bad["failed_files"]), 7)
        self.assertIn("No space left on device", bad["failed_files"][0]["detail"])
        [chk] = off["verify"]   # ingest verify of the failed destination: its own codes per file
        self.assertEqual((chk["root"], chk["verified"]), (str(self.b / "event" / "EX_CARD"), False))
        self.assertEqual({f["code"] for f in chk["failures"]}, {"INCOMPLETE"})
        self.assertIn("render again", out["hint"])

        out, off = self._failed_again("all_failed")
        self.assertEqual(off["error"]["kind"], "COPY_FAILED")

    def _failed_again(self, mode):
        self.tearDown()
        self.setUp()
        return self._failed(mode)

    def test_conflict_history_and_source_change(self):
        for mode, kind, codes in (("conflict", "CONFLICT", []), ("mhl_invalid", "HISTORY_DAMAGED", ["MHL_INVALID"]), ("source_changed", "SOURCE_CHANGED", ["SOURCE_CHANGED"])):
            with self.subTest(mode=mode):
                self.tearDown()
                self.setUp()
                _, off = self._failed(mode)
                err = off["error"]
                self.assertEqual((err["code"], err["kind"], err["details"]["codes"]), ("INCOMPLETE", kind, codes))
                if mode == "conflict":
                    first = next(x for x in err["details"]["destinations"] if x["conflicts"])
                    self.assertEqual(first["conflict_files"], ["BPAV/MEDIAPRO.XML"])
                if mode == "mhl_invalid":
                    self.assertEqual(err["details"]["destinations"][0]["problems"][0]["code"], "MHL_INVALID")

    def test_refusals_map_to_kinds_and_recovery(self):
        for code, kind, status in (("HASH_UNAVAILABLE", "ENGINE_MISSING", "BLOCKED"), ("REEL_INVALID", "INVALID_REQUEST", "FAILED"),
                                   ("SOURCE_NOT_FOUND", "SOURCE_MISSING", "BLOCKED"), ("DESTINATION_OVERLAPS_SOURCE", "INVALID_REQUEST", "FAILED")):
            with self.subTest(code=code):
                self.tearDown()
                self.setUp()
                _, off = self._failed(f"refuse:{code}", status=status)
                self.assertEqual((off["error"]["code"], off["error"]["kind"], off["error"]["exit_code"]), (code, kind, 2))
                self.assertEqual(off["error"]["recovery_class"], ERROR_KINDS[code][1])

    def test_reports_the_agent_cannot_admit(self):
        for mode, frag in (("wrong_root", "not admitted"), ("bad_json", "not JSON"), ("no_manifest", "no ingest-manifest.json"),
                           ("manifest_mismatch", "does not list"), ("inventory_outside", "not a file of the copy")):
            with self.subTest(mode=mode):
                self.tearDown()
                self.setUp()
                _, off = self._failed(mode)
                msg = (off.get("error") or {}).get("message") or ""
                self.assertIn(frag, msg)
                self.assertIn((off.get("error") or {}).get("kind"), ("INVALID_OUTPUT",))

    def test_timeout(self):
        _, off = self._failed("hang", timeout=2)
        self.assertEqual((off["error"]["code"], off["error"]["kind"]), ("TIMEOUT", "TIMEOUT"))

    def test_an_unmounted_destination_at_run_time_is_refused(self):
        svc, ir, path = self.planned()
        self.b.rmdir()   # the second drive was unplugged between plan and render: its mount point is never created
        out = self.approve_and_render(svc, path)
        self.assertEqual(out["status"], "FAILED")
        self.assertFalse(self.b.exists())
        err = load_ir(path).doc["execution"]["offload"]["error"]
        self.assertEqual((err["code"], err["kind"]), ("INVALID_REQUEST", "INVALID_REQUEST"))
        self.assertIn("never created", err["message"])
        self.assertEqual(len([c for c in self.offload_calls() if "--dry-run" not in c]), 0, "ingest-skill not even started")


class IngestAdapterContractTests(_Base):
    def test_pinned_contract_is_ingest_main(self):
        c = pinned_contract()
        self.assertEqual((c["skill_id"], c["version"], c["contract_version"]), ("ingest", "0.1.0", "1.0"))
        self.assertEqual(check_contract(c), [])
        self.assertEqual(PINNED_CONTRACT_PATH.read_text(encoding="utf-8"), json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.assertEqual(set(ERROR_KINDS), set(c["errors"]), "every ingest-skill error code has a kind")
        ad = IngestAdapter(fake_skill())
        self.assertEqual((ad.drift(), ad.breaking_drift(), ad.version), ([], [], "0.1.0"))
        self.assertEqual(FATAL_DRIFT_KEYS, ("skill_id", "contract_version", "manifest", "inventory"))

    def test_drift_classification(self):
        for mode, fatal in (("drift_version", False), ("new_error", False), ("drift_fatal", True), ("drift_error", True)):
            with self.subTest(mode=mode):
                os.environ["FAKE_INGEST_CONTRACT"] = mode
                ad = IngestAdapter(fake_skill())
                self.assertTrue(ad.drift() or mode == "new_error")
                self.assertEqual(bool(ad.breaking_drift()), fatal, ad.breaking_drift())
        os.environ["FAKE_INGEST_CONTRACT"] = "incompatible"
        with self.assertRaises(ContractError):
            IngestAdapter(fake_skill())

    def test_capability_from_doctor_and_fatal_drift(self):
        root = fake_ingest.checkout(Path(self.tmp) / "ingest-skill")
        env = {"VIDEO_AGENT_INGEST_DIR": root, "PATH": ""}
        r = CapabilityResolver(env=env)
        for mode, want in (("pinned", "AVAILABLE"), ("drift_version", "AVAILABLE"), ("drift_fatal", "MISSING")):
            os.environ["FAKE_INGEST_CONTRACT"] = mode
            caps = {}
            r._finishing_skill(caps, "ingest", locate_ingest(env=env), lambda sk: IngestAdapter(sk), "VIDEO_AGENT_INGEST_DIR", "ingest-skill", "ingest", per_item=None)
            self.assertEqual(caps["ingest"].status, want, (mode, caps["ingest"].detail))
        self.assertEqual(caps["ingest"].evidence.get("doctor"), "degraded", "xxhash missing is degraded, not missing")

    def test_offload_argv_refusals(self):
        good = {"source": str(self.card), "destinations": [str(self.a)], "hash": "sha256", "name_by": "folder"}
        self.assertEqual(offload_argv(good), ["offload", str(self.card), "--dest", str(self.a), "--hash", "sha256", "--name-by", "folder", "--json"])
        for patch, frag in (({"source": "EX_CARD"}, "source"), ({"destinations": []}, "non-empty"), ({"destinations": ["rel"]}, "absolute"),
                            ({"destinations": [str(self.a), str(self.a)]}, "twice"), ({"destinations": [str(self.card)]}, "contain one another"),
                            ({"hash": "md5"}, "hash"), ({"name_by": "x"}, "name_by"), ({"reel": "--json"}, "reel"), ({"project": "a/b"}, "project"),
                            ({"command": "rm -rf /"}, "unknown"), ({"dry_run": "yes"}, "dry_run")):
            with self.subTest(patch=patch), self.assertRaises(ToolError) as cm:
                offload_argv(dict(good, **patch))
            self.assertIn(frag, str(cm.exception))
        ad = IngestAdapter(fake_skill())
        with self.assertRaises(ToolError):   # a copy is never a measurement
            ad.measure("ingest/offload", dict(good, dry_run=False))
        r = ad.run(Operation(tool="ingest/offload", args=good, inputs=[], outputs=[]), {}, dry_run=True)
        self.assertTrue(r.ok and r.data["summary"]["dry_run"], "the executor's dry run is ingest-skill's --dry-run")
        self.assertEqual(list(self.a.iterdir()), [])

    def test_every_code_has_a_kind_and_class(self):
        for code in pinned_contract()["errors"]:
            kind, cls = kind_of(code)
            self.assertIn(cls, ("INPUT_MISSING", "INVALID_ARGS", "TOOL_MISSING", "SKILL_ERROR"))
        self.assertEqual(kind_of("VERIFY_MISMATCH"), ("VERIFY_FAILED", "SKILL_ERROR"))
        self.assertEqual(kind_of("MHL_INVALID"), ("HISTORY_DAMAGED", "SKILL_ERROR"))
        self.assertEqual(kind_of("HASH_UNAVAILABLE"), ("ENGINE_MISSING", "TOOL_MISSING"))
        self.assertEqual(kind_of("REEL_INVALID"), ("INVALID_REQUEST", "INVALID_ARGS"))
        self.assertEqual(kind_of("SOMETHING_NEW"), ("INCOMPLETE", "SKILL_ERROR"))

    def test_verify_maps_failure_codes(self):
        svc, ir, path = self.planned()
        self.assertEqual(self.approve_and_render(svc, path)["status"], "COMPLETED")
        root = self.a / "event" / "EX_CARD"
        (root / "BPAV" / "MEDIAPRO.XML").write_text("changed", encoding="utf-8")
        (root / "BPAV" / "CLPR" / "709_0002_01" / "709_0002_01M01.XML").unlink()
        r = IngestAdapter(fake_skill()).measure("ingest/verify", {"target": str(root)})
        self.assertTrue(r.ok)
        self.assertFalse(r.data["verified"])
        self.assertEqual(sorted((f["code"], f["kind"]) for f in r.data["failures"]), [("FILE_MISSING", "INCOMPLETE"), ("VERIFY_MISMATCH", "VERIFY_FAILED")])
        bad = next(f for f in r.data["failures"] if f["code"] == "VERIFY_MISMATCH")
        self.assertTrue(bad["expected"] and bad["actual"] and bad["expected"] != bad["actual"])

    def test_registry_skill(self):
        svc = self.svc()
        spec = svc.registry.get("media_offload")
        self.assertEqual((spec.implemented, spec.approval, spec.tools, spec.required_capabilities), (True, "CONFIRM", ["ingest/offload"], ["ingest"]))
        self.assertEqual(svc.tools_for().get("media_offload"), "ingest/offload")
        self.assertNotIn("media_offload", self.svc(ingest=False).tools_for())
        self.assertEqual(svc.registry.unknown_tool_candidates(), [])
        self.assertEqual(offload_task.SKILL, "media_offload")


if __name__ == "__main__":
    unittest.main()
