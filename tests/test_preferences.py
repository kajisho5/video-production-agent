"""Feedback scopes and the versioned preference store (ADR-111): every feedback item records a scope (this_video by default; an AI's
wider scope is a candidate), only an explicit, recorded promotion writes the channel's or the global store, every promotion and
rollback is a new store version, plans record the versions they used, and corrections per minute are reported (never acted on).
Fake engine; no media is decoded."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import clear_modes, fake_media, pipeline_service  # noqa: E402
from video_agent.mcp_server import TOOLS, tool_argv  # noqa: E402
from video_agent.preferences import (DEFAULT_SCOPE, PreferenceError, PreferenceStore, check_channel, feedback_scope, recent_average,  # noqa: E402
                                     snapshot, store_root)
from video_agent.project import load_ir, save_ir  # noqa: E402


class ScopeTests(unittest.TestCase):
    def test_default_user_and_ai_scopes(self):
        self.assertEqual(feedback_scope(None, None, None), {"scope": "this_video", "scope_source": "default", "scope_status": "confirmed"})
        self.assertEqual(feedback_scope("all_videos", "user", None)["scope"], "all_videos")
        self.assertEqual(feedback_scope("channel", None, "talks")["scope_status"], "confirmed")
        ai = feedback_scope("all_videos", "ai", None)
        self.assertEqual((ai["scope"], ai["proposed_scope"], ai["scope_status"]), ("this_video", "all_videos", "candidate"), "an AI never widens a scope by itself")
        self.assertEqual(feedback_scope("this_video", "ai", None)["scope_status"], "confirmed")
        self.assertEqual(DEFAULT_SCOPE, "this_video")

    def test_refusals(self):
        with self.assertRaises(PreferenceError):
            feedback_scope("channel", "user", None)   # no channel to scope to
        with self.assertRaises(PreferenceError):
            feedback_scope("everything", "user", "x")
        with self.assertRaises(PreferenceError):
            feedback_scope("channel", "model", "x")
        for bad in ("../x", "", "a b", "x" * 65, "/abs"):
            with self.subTest(bad=bad), self.assertRaises(PreferenceError):
                check_channel(bad)
        self.assertEqual(check_channel("kaji-talks_2026.v1"), "kaji-talks_2026.v1")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.ws = os.path.realpath(tempfile.mkdtemp())
        os.environ.pop("VIDEO_AGENT_PREFERENCES_DIR", None)

    def test_versions_diff_and_rollback(self):
        st = PreferenceStore(self.ws, "talks")
        self.assertEqual(st.path, Path(self.ws) / "preferences" / "channels" / "talks.json")
        self.assertEqual((st.doc()["current"], st.current()), (0, (0, {})))
        v1 = st.promote({"audio.loudness.target_lufs": -16}, "alice", "louder talks", {"project_id": "p1", "feedback_id": "fb_1", "plan_version": 1})
        v2 = st.promote({"edit.fit": "9:16", "audio.loudness.target_lufs": -14}, "bob", "", {"project_id": "p2", "feedback_id": "fb_2", "plan_version": 2})
        self.assertEqual((v1["version"], v2["version"], v2["previous"]), (1, 2, 1))
        self.assertEqual(st.current(), (2, {"audio.loudness.target_lufs": -14, "edit.fit": "9:16"}))
        rule = st.version(2)["rules"]["audio.loudness.target_lufs"]
        self.assertEqual((rule["by"], rule["feedback_id"], rule["project_id"]), ("bob", "fb_2", "p2"), "who / which feedback is kept per rule")
        d = st.diff(1, 2)
        self.assertEqual((d["added"], d["changed"], d["removed"]), ({"edit.fit": "9:16"}, {"audio.loudness.target_lufs": {"before": -16, "after": -14}}, {}))
        rb = st.rollback(1, "carol", "the vertical crop hurt the talks")
        self.assertEqual((rb["version"], rb["action"], rb["source"]), (3, "rollback", {"rollback_to": 1}))
        self.assertEqual(st.current(), (3, {"audio.loudness.target_lufs": -16}))
        self.assertEqual([h["version"] for h in st.history()], [0, 1, 2, 3], "a rollback deletes nothing")
        self.assertEqual([h["current"] for h in st.history()], [False, False, False, True])
        with self.assertRaises(PreferenceError):
            st.version(9)
        with self.assertRaises(PreferenceError):
            st.rollback(9, "x", "")
        self.assertEqual(json.loads(st.path.read_text(encoding="utf-8"))["schema"], "video-agent/preference-store@1")
        self.assertEqual(PreferenceStore(self.ws).current(), (0, {}), "the global store is separate")

    def test_snapshot_global_then_channel_and_the_directory_override(self):
        PreferenceStore(self.ws).promote({"audio.loudness.target_lufs": -16, "edit.fit": "16:9"}, "a", "", {})
        PreferenceStore(self.ws, "shorts").promote({"edit.fit": "9:16"}, "a", "", {})
        snap = snapshot(self.ws, "shorts")
        self.assertEqual(snap["values"], {"audio.loudness.target_lufs": -16, "edit.fit": "9:16"}, "the channel wins over global")
        self.assertEqual([(s["store"], s["version"]) for s in snap["stores"]], [("global", 1), ("channel:shorts", 1)])
        other = os.path.realpath(tempfile.mkdtemp())
        os.environ["VIDEO_AGENT_PREFERENCES_DIR"] = other
        try:
            self.assertEqual(store_root(self.ws), Path(other))
            self.assertEqual(PreferenceStore(self.ws).current(), (0, {}))
        finally:
            os.environ.pop("VIDEO_AGENT_PREFERENCES_DIR")

    def test_a_damaged_store_is_refused(self):
        st = PreferenceStore(self.ws)
        st.path.parent.mkdir(parents=True, exist_ok=True)
        st.path.write_text("{}", encoding="utf-8")
        with self.assertRaises(PreferenceError):
            st.current()

    def test_recent_average(self):
        log = [{"project_id": "a", "plan_version": 1, "cpm": 1.0, "at": "1"}, {"project_id": "a", "plan_version": 2, "cpm": 0.5, "at": "2"},
               {"project_id": "b", "plan_version": 1, "cpm": 2.0, "at": "3"}, {"project_id": "me", "plan_version": 1, "cpm": 9.0, "at": "4"}]
        self.assertEqual(recent_average(log, "me"), {"projects": 2, "cpm": 1.25}, "each other project's latest version")
        self.assertIsNone(recent_average([], "me"))


class ServiceTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        os.environ.pop("VIDEO_AGENT_PREFERENCES_DIR", None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.svc = pipeline_service(self.tmp)
        self.n = 0

    def plan(self, channel=None, reqs=None, **kw):
        self.n += 1
        src = fake_media(self.tmp, f"talk{self.n}.mp4")
        ir = self.svc.plan([src], "youtube", user_requirements=reqs or {}, channel=channel, **kw)
        path = str(Path(self.svc.workspace) / "plans" / f"p{self.n}.project.json")
        save_ir(ir, path)
        return load_ir(path), path

    @staticmethod
    def req(doc, key):
        return next((r for r in doc["requirements"] if r["key"] == key), None)

    def test_channel_scope_promotion_reaches_the_next_plan_of_that_channel_only(self):
        ir, path = self.plan(channel="talks")
        self.assertEqual(ir.doc["project"]["channel_id"], "talks")
        self.assertEqual([(s["store"], s["version"]) for s in ir.doc["request"]["args"]["preferences"]["stores"]], [("global", 0), ("channel:talks", 0)])
        self.svc.revise(ir, path, feedback="quieter", user_requirements={"audio.loudness.target_lufs": -16}, scope="channel", who="alice")
        fb = load_ir(path).doc["revision"]["feedback"][-1]
        self.assertEqual((fb["scope"], fb["scope_status"], fb["scope_source"]), ("channel", "confirmed", "user"))
        self.assertIsNone(PreferenceStore(self.svc.workspace, "talks").current()[1].get("audio.loudness.target_lufs"), "feedback alone never writes a store")
        out = self.svc.promote_feedback(load_ir(path), path, fb["id"], who="alice", reason="the channel is quieter")
        self.assertEqual((out["store"], out["version"], out["rules"]), ("channel:talks", 1, ["audio.loudness.target_lufs"]))
        self.assertEqual(load_ir(path).doc["revision"]["feedback"][-1]["promotion"]["by"], "alice")
        with self.assertRaisesRegex(PreferenceError, "promoted already"):
            self.svc.promote_feedback(load_ir(path), path, fb["id"], who="alice")
        nxt, _ = self.plan(channel="talks")
        r = self.req(nxt.doc, "audio.loudness.target_lufs")
        self.assertEqual((r["value"], r["provenance"], r["source"]), (-16, "PREFERENCE", "preferences:channel:talks@v1"))
        self.assertEqual(next(d for d in nxt.doc["decisions"] if d["subject"] == "audio.loudness")["params"].get("target_lufs"), -16)
        self.assertEqual(self.svc.validate(nxt).errors, [])
        self.assertEqual([(s["store"], s["version"]) for s in nxt.doc["request"]["args"]["preferences"]["stores"]], [("global", 0), ("channel:talks", 1)])
        explicit, _ = self.plan(channel="talks", reqs={"audio.loudness.target_lufs": -20})
        self.assertEqual((self.req(explicit.doc, "audio.loudness.target_lufs")["value"], self.req(explicit.doc, "audio.loudness.target_lufs")["provenance"]), (-20, "USER"),
                         "an explicit --set wins over a stored preference")
        other, _ = self.plan(channel="news")
        self.assertNotEqual((self.req(other.doc, "audio.loudness.target_lufs") or {}).get("provenance"), "PREFERENCE", "another channel is untouched")
        bare, _ = self.plan(channel="talks", use_preferences=False)
        self.assertNotIn("preferences", bare.doc["request"]["args"])
        self.assertNotEqual((self.req(bare.doc, "audio.loudness.target_lufs") or {}).get("provenance"), "PREFERENCE")

    def test_an_ai_scope_is_a_candidate_until_a_person_confirms_it(self):
        ir, path = self.plan()
        self.svc.revise(ir, path, feedback="always vertical", user_requirements={"edit.fit": "9:16"}, scope="all_videos", scope_by="ai")
        fb = load_ir(path).doc["revision"]["feedback"][-1]
        self.assertEqual((fb["scope"], fb["proposed_scope"], fb["scope_status"]), ("this_video", "all_videos", "candidate"))
        with self.assertRaisesRegex(PreferenceError, "proposed by an AI"):
            self.svc.promote_feedback(load_ir(path), path, fb["id"], who="alice")
        out = self.svc.promote_feedback(load_ir(path), path, fb["id"], who="alice", confirm_scope="all_videos")
        self.assertEqual((out["store"], out["scope"]), ("global", "all_videos"))
        fb = load_ir(path).doc["revision"]["feedback"][-1]
        self.assertEqual((fb["scope"], fb["scope_status"], fb["confirmed_by"]), ("all_videos", "confirmed", "alice"))
        nxt, _ = self.plan(channel="anything")
        self.assertEqual(self.req(nxt.doc, "edit.fit")["source"], "preferences:global@v1", "a global preference applies to every channel")

    def test_refused_promotions(self):
        ir, path = self.plan(channel="talks")
        self.svc.revise(ir, path, feedback="this one only", user_requirements={"audio.loudness.target_lufs": -18})
        self.svc.revise(load_ir(path), path, feedback="channel wide, words only", scope="channel")
        fbs = load_ir(path).doc["revision"]["feedback"]
        self.assertEqual(fbs[0]["scope"], "this_video")
        with self.assertRaisesRegex(PreferenceError, "this video only"):
            self.svc.promote_feedback(load_ir(path), path, fbs[0]["id"])
        with self.assertRaisesRegex(PreferenceError, "free text"):
            self.svc.promote_feedback(load_ir(path), path, fbs[1]["id"])
        with self.assertRaisesRegex(PreferenceError, "no feedback"):
            self.svc.promote_feedback(load_ir(path), path, "fb_nope")
        with self.assertRaisesRegex(ValueError, "belongs to feedback"):
            self.svc.revise(load_ir(path), path, scope="channel")
        with self.assertRaisesRegex(ValueError, "channel"):
            self.svc.revise(load_ir(path), path, channel="news", feedback="x")
        plain, ppath = self.plan()
        with self.assertRaisesRegex(ValueError, "channel"):
            self.svc.revise(plain, ppath, feedback="x", user_requirements={"audio.loudness.target_lufs": -18}, scope="channel")
        self.svc.revise(load_ir(ppath), ppath, feedback="x", user_requirements={"audio.loudness.target_lufs": -18}, scope="channel", channel="late")
        self.assertEqual(load_ir(ppath).doc["project"]["channel_id"], "late", "a channel can be named once, at revise")

    def test_a_preference_never_waives_a_confirm(self):
        ir, path = self.plan(channel="fast", reqs={"edit.speed": 2})
        self.assertEqual(next(d for d in ir.doc["decisions"] if d["subject"] == "video.speed")["approval"], "AUTO", "the user's own --set is the confirmation")
        self.svc.revise(ir, path, feedback="faster", user_requirements={"edit.speed": 2}, scope="channel")
        self.svc.promote_feedback(load_ir(path), path, load_ir(path).doc["revision"]["feedback"][-1]["id"])
        nxt, _ = self.plan(channel="fast")
        d = next(d for d in nxt.doc["decisions"] if d["subject"] == "video.speed")
        self.assertEqual((d["approval"], self.req(nxt.doc, "edit.speed")["provenance"]), ("CONFIRM", "PREFERENCE"))

    def test_recorded_versions_survive_a_rollback_and_the_next_plan_follows_it(self):
        ir, path = self.plan(channel="talks")
        self.svc.revise(ir, path, feedback="quieter", user_requirements={"audio.loudness.target_lufs": -16}, scope="channel")
        self.svc.promote_feedback(load_ir(path), path, load_ir(path).doc["revision"]["feedback"][-1]["id"])
        used, upath = self.plan(channel="talks")
        self.svc.preference_rollback("talks", 0, who="carol", reason="quality dropped")
        self.assertEqual(self.svc.preference_history("talks")["current"], 2)
        self.svc.revise(load_ir(upath), upath, feedback="tighter", user_requirements={"edit.trim_leading_silence": True})
        again = load_ir(upath).doc
        self.assertEqual((self.req(again, "audio.loudness.target_lufs") or {}).get("source"), "preferences:channel:talks@v1", "a revision keeps the versions its plan used")
        after, _ = self.plan(channel="talks")
        self.assertNotEqual((self.req(after.doc, "audio.loudness.target_lufs") or {}).get("provenance"), "PREFERENCE", "the next plan follows the rollback")
        self.assertEqual(after.doc["request"]["args"]["preferences"]["stores"][1]["version"], 2)
        d = self.svc.preference_diff("talks", 1, 2)
        self.assertEqual(d["removed"], {"audio.loudness.target_lufs": -16})

    def test_corrections_per_minute(self):
        a, apath = self.plan(channel="talks")
        self.svc.revise(a, apath, feedback="one", user_requirements={"audio.loudness.target_lufs": -18})
        rep = self.svc.corrections(load_ir(apath), apath)
        v1, v2 = rep["versions"]
        self.assertEqual((v1["plan_version"], v1["feedback"], v2["feedback"]), (1, 1, 0))
        self.assertGreater(v1["delivered_minutes"], 0)
        self.assertAlmostEqual(v1["cpm"], round(1 / v1["delivered_minutes"], 3), places=3)
        self.assertEqual(v2["change_from_previous"], round(0 - v1["cpm"], 3))
        self.assertIsNone(rep["channel_recent_average"], "no other project of the channel yet")
        b, bpath = self.plan(channel="talks")
        self.svc.revise(b, bpath, feedback="two", user_requirements={"audio.loudness.target_lufs": -17})
        rep_b = self.svc.corrections(load_ir(bpath), bpath)
        self.assertEqual(rep_b["channel_recent_average"]["projects"], 1)
        self.assertEqual(rep_b["channel_recent_average"]["cpm"], v2["cpm"], "the other project's latest version")
        self.assertIn("nothing acts on it", rep_b["note"])
        self.assertEqual(load_ir(bpath).version, 2, "the report changes nothing")


class InterfaceTests(unittest.TestCase):
    def test_mcp_tools(self):
        names = {t["name"] for t in TOOLS}
        self.assertTrue({"preferences", "corrections"} <= names)
        tmp = tempfile.mkdtemp()
        self.assertEqual(tool_argv("preferences", {"action": "promote", "project": "p.json", "feedback": "fb_1", "confirm_scope": "channel", "by": "u"}, tmp),
                         ["preferences", "promote", "--feedback=fb_1", "--confirm-scope=channel", "--by=u", "--", "p.json"])
        self.assertEqual(tool_argv("preferences", {"action": "rollback", "channel": "talks", "to": 0}, tmp), ["preferences", "rollback", "--channel=talks", "--to=0"])
        self.assertEqual(tool_argv("preferences", {"action": "diff", "from": 1, "to": 2}, tmp), ["preferences", "diff", "--from=1", "--to=2"])
        self.assertEqual(tool_argv("preferences", {}, tmp), ["preferences", "list"], "read-only without an action")
        self.assertIn("--scope=channel", tool_argv("revise", {"project": "p", "feedback": "f", "scope": "channel", "scope_by": "ai"}, tmp))
        self.assertIn("--channel=talks", tool_argv("plan", {"inputs": ["a.mp4"], "channel": "talks"}, tmp))
        self.assertEqual(tool_argv("corrections", {"project": "p"}, tmp), ["corrections", "--", "p"])

    def test_cli_round_trip(self):
        from video_agent import cli
        ws = os.path.realpath(tempfile.mkdtemp())
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(["--json", "--workspace", ws, "preferences", "list", "--channel", "talks"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(buf.getvalue())["store"], "channel:talks")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(["--json", "--workspace", ws, "preferences", "rollback", "--to", "5"])
        self.assertEqual(code, 2)
        self.assertIn("no version 5", json.loads(buf.getvalue())["error"])


if __name__ == "__main__":
    unittest.main()
