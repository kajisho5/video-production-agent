"""ADR-091 (slide-sync-skill roadmap step 6, agent side: the adapter): slide-sync-skill's slide timeline (and, with the speaker's deck, the page on
screen in each slide) as the analysis kind `slide_timeline`, asked for explicitly (`slides.detect`), against a fake `slide-sync` process
(tests/fake_slide_sync.py) that speaks slide-sync-skill main's contract, doctor, timeline, deck and match documents, with the fake media engine.
No engine, ffmpeg or network."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import PipelineCaps, clear_modes, fake_media, pipeline_service  # noqa: E402

from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.slides import parse_slide_requirements  # noqa: E402
from video_agent.capabilities.resolver import CapabilityResolver  # noqa: E402
from video_agent.models import Operation  # noqa: E402
from video_agent.service import BRIEF_REQUIREMENT_KEYS, _check_requirement_keys  # noqa: E402
from video_agent.tools.discovery import MODULE_BOOTSTRAP  # noqa: E402
from video_agent.tools.skill_process import CliSkill  # noqa: E402
from video_agent.tools.slide_sync import (ERROR_CLASSES, ContractError, SlideSyncAdapter, breaking_drift, check_contract, drift, locate_slide_sync,  # noqa: E402
                                          pinned_contract, recovery_class)
from video_agent.tools.slide_sync.adapter import PINNED_CONTRACT_PATH  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_slide_sync.py")
MODES = ("FAKE_SS_MODE", "FAKE_SS_READY", "FAKE_SS_CALLS", "VIDEO_AGENT_SLIDE_SYNC_DIR")
SLIDES = [[0.0, 5.0], [5.0, 11.0], [11.0, 16.0]]


def fake_skill() -> CliSkill:
    return CliSkill("slide-sync", [sys.executable, FAKE], None, {})


def ss_adapter(tmp: str) -> SlideSyncAdapter:
    ws = str(Path(tmp) / "ws")
    os.makedirs(ws, exist_ok=True)
    return SlideSyncAdapter(fake_skill(), workspace=ws, allowed_inputs=[str(Path(tmp) / "src"), str(Path(tmp) / "decks"), ws], ffmpeg_skill_dir=tmp)


def ss_capabilities(factory) -> dict:
    caps: dict = {}
    CapabilityResolver._finishing_skill(CapabilityResolver.__new__(CapabilityResolver), caps, "slide-sync", fake_skill(), lambda _sk: factory(),
                                        "VIDEO_AGENT_SLIDE_SYNC_DIR", "slide-sync-skill", "slide-sync",
                                        per_item=lambda ad, doc: (ad.operation_status(doc), {}), item_evidence=lambda ad, doc: ad.operation_evidence(doc))
    return caps


class SsCaps(PipelineCaps):
    def __init__(self, ss: dict, **kw):
        super().__init__(**kw)
        self.ss = ss

    def resolve(self, refresh=False):
        out = super().resolve(refresh)
        out.update(self.ss)
        return out


def ss_service(tmp: str, installed: bool = True):
    caps = SsCaps(ss_capabilities(lambda: ss_adapter(tmp)) if installed else {})
    svc = pipeline_service(tmp, caps=caps, transcription=False)
    if installed:
        svc._adapter.register(ss_adapter(tmp))
        svc.adapter([])
    return svc


def deck_file(tmp: str, pages: int = 4, name: str = "deck.pdf") -> str:
    p = Path(tmp) / "decks" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"pages": pages}), encoding="utf-8")
    return str(p)


def calls(log: str):
    return [json.loads(x) for x in Path(log).read_text(encoding="utf-8").splitlines()] if os.path.exists(log) else []


class _Base(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self._env = {k: os.environ.get(k) for k in MODES}
        for k in MODES:
            os.environ.pop(k, None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.log = str(Path(self.tmp) / "ss_calls.log")
        os.environ["FAKE_SS_CALLS"] = self.log

    def tearDown(self):
        clear_modes()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def ss_calls(self, cmd):
        return [c for c in calls(self.log) if c and c[0] == cmd]


class ContractTests(_Base):
    def test_pin_shape_and_contract_checks(self):
        c = pinned_contract()
        raw = PINNED_CONTRACT_PATH.read_text(encoding="utf-8")
        self.assertEqual(raw, json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n", "one sorted line (no release tag: pinned from main)")
        self.assertEqual((c["skill_id"], c["version"], c["contract_version"]), ("slide-sync", "0.1.0", "1.0"))
        self.assertEqual((check_contract(c), drift(c), breaking_drift(c)), ([], [], []))
        self.assertEqual(set(ERROR_CLASSES), set(c["errors"]), "every slide-sync code has a recovery class")
        self.assertEqual((recovery_class("DEPENDENCY_MISSING"), recovery_class("ENGINE_TIMEOUT"), recovery_class("NEW")), ("TOOL_MISSING", "TIMEOUT", "SKILL_ERROR"))

    def test_refused_contracts(self):
        c = pinned_contract()
        bad = json.loads(json.dumps(c)); bad["version"] = "0.2.0"
        self.assertTrue(any("supported range" in e for e in check_contract(bad)))
        bad = json.loads(json.dumps(c)); next(t for t in bad["tools"] if t["tool_id"] == "slide-sync/detect")["writes_media"] = True
        self.assertTrue(any("writes_media false" in e for e in check_contract(bad)))
        bad = json.loads(json.dumps(c)); bad["match"]["schema"] = "slide-sync/match@2"
        self.assertTrue(any("match.schema" in e for e in check_contract(bad)))
        self.assertTrue(any("match.schema" in e for e in breaking_drift(bad)))
        bad = json.loads(json.dumps(c)); del bad["errors"]["DEPENDENCY_MISSING"]
        self.assertTrue(any("DEPENDENCY_MISSING" in e for e in check_contract(bad)))
        with self.assertRaises(ContractError):
            os.environ["FAKE_SS_MODE"] = "ok"
            SlideSyncAdapter(CliSkill("slide-sync", [sys.executable, "-c", "print('{}')"], None, {}), workspace=self.tmp)

    def test_locate_never_searches_the_current_directory(self):
        cwd = os.getcwd()
        here = Path(self.tmp) / "cwd"
        (here / "slide-sync-skill" / "src" / "slide_sync").mkdir(parents=True)
        (here / "slide-sync-skill" / "src" / "slide_sync" / "cli.py").write_text("", encoding="utf-8")
        try:
            os.chdir(here)
            located = locate_slide_sync(env={"PATH": ""})
            self.assertTrue(located is None or str(located.root) != str(here / "slide-sync-skill"))
        finally:
            os.chdir(cwd)
        explicit = locate_slide_sync(str(here / "slide-sync-skill"), env={})
        self.assertEqual(explicit.command[1:3], ["-c", MODULE_BOOTSTRAP], "run without the current directory on sys.path (ADR-074)")
        self.assertIsNone(locate_slide_sync(env={"VIDEO_AGENT_SLIDE_SYNC_DIR": str(Path(self.tmp) / "nothing"), "PATH": ""}))


class CapabilityTests(_Base):
    def test_ready_engine_and_import(self):
        caps = ss_capabilities(lambda: ss_adapter(self.tmp))
        self.assertEqual((caps["slide-sync"].status, caps["slide-sync:DETECT"].status, caps["slide-sync:IMPORT_PDF"].status), ("AVAILABLE", "AVAILABLE", "AVAILABLE"))
        self.assertEqual(caps["slide-sync:IMPORT_OFFICE"].status, "MISSING", "LibreOffice missing: pptx / key / odp cannot be imported")
        self.assertIn("LibreOffice", caps["slide-sync:IMPORT_OFFICE"].evidence["install"])
        [argv] = self.ss_calls("doctor")
        self.assertIn(f"--ffmpeg-skill={self.tmp}", argv)

    def test_without_pdf_import_is_degraded_and_without_engine_missing(self):
        os.environ["FAKE_SS_READY"] = "nopdf"
        caps = ss_capabilities(lambda: ss_adapter(self.tmp))
        self.assertEqual((caps["slide-sync"].status, caps["slide-sync:DETECT"].status, caps["slide-sync:IMPORT_PDF"].status), ("AVAILABLE", "AVAILABLE", "MISSING"))
        self.assertEqual(caps["slide-sync:IMPORT_PDF"].evidence["code"], "DEPENDENCY_MISSING")
        os.environ["FAKE_SS_READY"] = "noengine"
        caps = ss_capabilities(lambda: ss_adapter(self.tmp))
        self.assertEqual(caps["slide-sync"].status, "MISSING")


class AdapterTests(_Base):
    def measure(self, params=None, raw=False, **media):
        ad = ss_adapter(self.tmp)
        path = fake_media(self.tmp, duration=16.0, slides=SLIDES, **media)
        params = {} if params is None else params
        args = {"input": path, "asset_id": "asset_a", "parameters": params} if raw else ad.measurement_args("slide-sync/detect", "slide_timeline", path, "asset_a", params, "an_1", "use")
        return ad, path, ad.measure("slide-sync/detect", args)

    def test_detect_argv_and_result(self):
        ad, path, r = self.measure({"threshold": 12, "min_dwell": 3})
        self.assertTrue(r.ok, r.data)
        tl = str(Path(ad.workspace) / "cache" / "slide-sync" / "asset_a.timeline.json")
        self.assertEqual(self.ss_calls("detect"), [["detect", "--json", "--overwrite", f"--output={tl}", "--threshold=12", "--min-dwell=3", f"--ffmpeg-skill={self.tmp}", "--timeout=3600", "--", path]])
        self.assertEqual([s["index"] for s in r.data["timeline"]["slides"]], [1, 2, 3])
        self.assertNotIn("match", r.data)
        self.assertEqual(self.ss_calls("import"), [])

    def test_deck_is_imported_then_matched(self):
        deck = deck_file(self.tmp, pages=5)
        ad, path, r = self.measure({"deck": deck}, pages=[2, None, 4])
        self.assertTrue(r.ok, r.data)
        d = Path(ad.workspace) / "cache" / "slide-sync"
        self.assertEqual(self.ss_calls("import"), [["import", "--json", "--overwrite", f"--output={d / 'asset_a.deck'}", "--", deck]])
        self.assertEqual(self.ss_calls("match"), [["match", "--json", "--overwrite", f"--output={d / 'asset_a.match.json'}", f"--deck={d / 'asset_a.deck'}",
                                                   f"--timeline={d / 'asset_a.timeline.json'}", f"--ffmpeg-skill={self.tmp}", "--timeout=3600", "--", path]])
        self.assertEqual([s["page"] for s in r.data["match"]["segments"]], [2, None, 4])
        self.assertEqual(r.data["deck"]["page_count"], 5)
        self.assertNotIn("text", json.dumps(r.data["deck"]["pages"]), "page texts stay in the deck directory")

    def test_requests_the_skill_would_refuse_never_start_a_process(self):
        for params in ({"threshold": 0}, {"threshold": 101}, {"min_dwell": 0}, {"deck": "notes.txt"}, {"deck": str(Path(self.tmp) / "decks" / "missing.pdf")}, {"samples": 3}):
            with self.subTest(params=params):
                _, _, r = self.measure(params, raw=True)
                self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_REQUEST"))
        outside = Path(self.tmp) / "elsewhere" / "deck.pdf"; outside.parent.mkdir(); outside.write_text("{}")
        _, _, r = self.measure({"deck": str(outside)}, raw=True)
        self.assertIn("outside the allowed input roots", r.data["error"]["message"])
        self.assertEqual(self.ss_calls("detect"), [])

    def test_results_that_are_not_about_the_request_are_refused(self):
        deck = deck_file(self.tmp)
        for mode, why, params in (("wrong_threshold", "requested", {"threshold": 9}), ("other_video", "not the input video", {}), ("overlap", "time order", {}),
                                  ("not_written", "on disk", {}), ("bad_page", "not a page of the deck", {"deck": deck}), ("deck_sha", "sha256 of the deck", {"deck": deck})):
            with self.subTest(mode=mode):
                os.environ["FAKE_SS_MODE"] = mode
                _, _, r = self.measure(params)
                self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_OUTPUT"))
                self.assertIn(why, r.data["error"]["message"])

    def test_skill_codes_are_kept(self):
        os.environ["FAKE_SS_READY"] = "nopdf"
        _, _, r = self.measure({"deck": deck_file(self.tmp)})
        self.assertEqual((r.ok, r.data["error"]["code"], r.data["error"]["recovery_class"], r.exit_code), (False, "DEPENDENCY_MISSING", "TOOL_MISSING", 3))
        self.assertIn("slide-sync-skill[pdf]", r.data["error"]["message"])

    def test_never_runs_as_a_plan_step(self):
        r = ss_adapter(self.tmp).run(Operation(tool="slide-sync/detect", args={}, inputs=[], outputs=[], kind="transform"), {})
        self.assertEqual((r.ok, r.data["error"]["code"]), (False, "INVALID_REQUEST"))


class RequirementTests(_Base):
    def test_vocabulary(self):
        deck = deck_file(self.tmp)
        self.assertEqual(parse_slide_requirements({"slides.detect": True}), (True, {}))
        self.assertEqual(parse_slide_requirements({"slides.detect": "true", "slides.threshold": "12", "slides.min_dwell": 3, "slides.deck": deck}),
                         (True, {"threshold": 12.0, "min_dwell": 3.0, "deck": deck}))
        self.assertEqual(parse_slide_requirements({"slides.detect": False}), (False, {}))

    def test_refused_before_analysis(self):
        for reqs, why in (({"slides.threshold": 8}, "not set to true"), ({"slides.detect": "maybe"}, "switch"), ({"slides.detect": True, "slides.threshold": 0}, "(0, 100]"),
                          ({"slides.detect": True, "slides.deck": "deck.txt"}, "file"), ({"slides.detect": True, "slides.deck": "/nope/deck.pdf"}, "no such file"),
                          ({"slides.detect": True, "slides.samples": 3}, "unknown requirement key")):
            with self.subTest(reqs=reqs):
                with self.assertRaises(EditRequirementError) as cm:
                    parse_slide_requirements(reqs)
                self.assertIn(why, str(cm.exception))
                with self.assertRaises(ValueError):
                    _check_requirement_keys(reqs)

    def test_not_a_brief_key(self):
        self.assertFalse([k for k in BRIEF_REQUIREMENT_KEYS if k.startswith("slides.")], "only the user names the deck file; the measurement is asked for explicitly")


class PlanTests(_Base):
    def plan(self, reqs, installed=True, **media):
        self.svc = ss_service(self.tmp, installed=installed)
        self.media_path = fake_media(self.tmp, duration=16.0, slides=SLIDES, **media)
        return self.svc.plan([self.media_path], "generic", user_requirements=reqs)

    def test_slide_timeline_observed_with_slide_events(self):
        ir = self.plan({"slides.detect": True, "slides.min_dwell": 3})
        d = ir.doc
        [obs] = [o for o in d["analysis"]["observations"] if o["kind"] == "slide_timeline"]
        self.assertEqual((obs["provenance"], obs["tool"], obs["parameters"]), ("OBSERVED", "slide-sync/detect", {"min_dwell": 3.0}))
        self.assertEqual(obs["data"]["slide_count"], 3)
        events = [e for e in d["timeline"]["events"] if e["type"] == "SLIDE"]
        self.assertEqual([(e["range"]["start"], e["metadata"]["slide"]) for e in events], [(0.0, 1), (5.0, 2), (11.0, 3)])
        self.assertFalse([x for x in d["decisions"] if x["subject"] == "slides.detect"], "a measurement that ran decides nothing")
        self.assertEqual(self.svc.validate(ir).errors, [])

    def test_deck_pages_on_the_events_and_unmatched_stays_null(self):
        deck = deck_file(self.tmp, pages=6)
        ir = self.plan({"slides.detect": True, "slides.deck": deck}, pages=[3, None, 5])
        d = ir.doc
        [obs] = [o for o in d["analysis"]["observations"] if o["kind"] == "slide_timeline"]
        self.assertEqual(obs["parameters"]["deck"], "deck.pdf", "the deck by its name, never its path")
        self.assertEqual(len(obs["parameters"]["deck_sha256"]), 64)
        self.assertEqual(obs["data"]["deck"]["page_count"], 6)
        events = [e for e in d["timeline"]["events"] if e["type"] == "SLIDE"]
        self.assertEqual([(e["metadata"]["page"], e["metadata"]["matched"]) for e in events], [(3, True), (None, False), (5, True)])
        self.assertEqual(events[1]["metadata"]["reason"], "below_min_score")
        self.assertEqual(self.svc.validate(ir).errors, [])

    def test_without_slide_sync_the_measurement_is_handed_off(self):
        ir = self.plan({"slides.detect": True}, installed=False)
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")
        [h] = [h for h in self.svc.handoffs(ir) if h["skill"] == "slide_sync"]
        self.assertEqual((h["step"], h["repositories"]), ("step_job_slide_sync", ["slide-sync-skill"]))
        self.assertEqual(self.ss_calls("detect"), [])

    def test_a_failed_measurement_blocks_with_the_skill_code(self):
        os.environ["FAKE_SS_READY"] = "nopdf"
        ir = self.plan({"slides.detect": True, "slides.deck": deck_file(self.tmp)})
        [blk] = [x for x in ir.doc["decisions"] if x["subject"] == "slides.detect" and x["type"] == "BLOCK"]
        self.assertEqual(blk["params"]["code"], "DEPENDENCY_MISSING")
        self.assertIn("slide-sync-skill[pdf]", blk["reason"])
        self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")

    def test_not_measured_unless_asked(self):
        ir = self.plan({})
        self.assertFalse([o for o in ir.doc["analysis"]["observations"] if o["kind"] == "slide_timeline"])
        self.assertEqual(self.ss_calls("detect"), [])


if __name__ == "__main__":
    unittest.main()
