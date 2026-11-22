"""ADR-129: smart rename. The evidence → fields rules and the MCP / CLI wiring without a Skill (unit job), and the whole flow on real
files through the real ingest-skill (RenameRealSkillTests: needs ingest-skill 1.1.0+ at VIDEO_AGENT_INGEST_DIR or on PATH)."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from video_agent.agent import rename  # noqa: E402
from video_agent.cli import main as cli_main  # noqa: E402
from video_agent.mcp_permissions import DESTRUCTIVE, READ, TOOL_CLASSES, classify, load_permissions  # noqa: E402
from video_agent.mcp_server import Server, listed_tools, tool_argv  # noqa: E402

OK = subprocess.CompletedProcess([], 0, "{}", "")
TANAKA = {"id": "kenji-tanaka", "canonical": "Kenji Tanaka", "type": "person", "aliases": ["Tanaka"], "speaker": {"speaker_id": "SPEAKER_01"}}
SATO = {"id": "yui-sato", "canonical": "Yui Sato", "type": "person", "aliases": ["Sato", "佐藤"]}
ACME = {"id": "acme", "canonical": "Acme", "type": "organization"}


def clip(path, sha=None, local=None, dur=None):
    return {"path": path, "clip_id": path.rsplit(".", 1)[0], "hash": {"algorithm": "sha256", "value": sha} if sha else None,
            "duration_s": dur, "sync": {"wall_clock_start": {"local": local, "utc": None, "source": "test"} if local else None}}


def turns(fp=None, filename=None, spans=()):
    return {"schema": rename.TURNS_SCHEMA, "source": {"filename": filename or "x.mp4", **({"fingerprint": f"sha256:{fp}"} if fp else {})},
            "turns": [{"start": a, "end": b, "speaker": s} for a, b, s in spans]}


class EvidenceRuleTests(unittest.TestCase):
    def test_matching_by_fingerprint_then_file_name(self):
        c = clip("DCIM/C0001.MP4", sha="ab" * 32)
        ev = [{"kind": "turns", "path": "/e/other.json", "doc": turns(fp="cd" * 32, filename="C0001.MP4")},
              {"kind": "turns", "path": "/e/t.json", "doc": turns(fp="ab" * 32, filename="renamed.mp4")},
              {"kind": "transcript", "path": "/e/C0001.transcript.json", "doc": {"schema": "transcription-skill/transcript/0.1", "segments": []}}]
        m = rename.match_evidence(c, ev)
        self.assertEqual(m["turns"]["path"], "/e/t.json")
        self.assertEqual(m["turns"]["matched_by"], "fingerprint")
        self.assertEqual(m["transcript"]["matched_by"], "evidence file name")
        m = rename.match_evidence(clip("DCIM/C0001.MP4"), ev[:1])
        self.assertEqual(m["turns"]["matched_by"], "source file name")

    def test_main_speaker_needs_half_of_the_speech(self):
        self.assertEqual(rename.main_speaker(turns(spans=[(0, 30, "SPEAKER_01"), (30, 40, "SPEAKER_00")]))[0], "SPEAKER_01")
        self.assertIsNone(rename.main_speaker(turns(spans=[(0, 10, "SPEAKER_00"), (10, 20, "SPEAKER_01"), (20, 30, "SPEAKER_02")])))
        self.assertIsNone(rename.main_speaker(turns()))

    def test_speaker_from_proposal_then_speaker_id_then_mentions(self):
        names = {"schema": rename.NAMES_SCHEMA, "speakers": [
            {"speaker": "SPEAKER_00", "status": "proposed", "proposal": {"person_id": "p1", "name": "Yui Sato", "basis": ["voice"], "confirmed": False}},
            {"speaker": "SPEAKER_01", "status": "anonymous", "proposal": None}]}
        sp, _ = rename.speaker_for({"names": {"doc": names}}, [TANAKA, SATO])
        self.assertEqual(sp["value"], "Yui Sato")
        self.assertIn("name proposal", sp["source"])
        # the main speaker by turns is SPEAKER_01, who has no proposal: the entity with that speaker_id
        t = turns(spans=[(0, 50, "SPEAKER_01"), (50, 60, "SPEAKER_00")])
        sp, _ = rename.speaker_for({"names": {"doc": names}, "turns": {"doc": t}}, [TANAKA, SATO])
        self.assertEqual(sp["value"], "Kenji Tanaka")
        self.assertIn("speaker_id SPEAKER_01", sp["source"])
        tr = {"schema": "transcription-skill/transcript/0.1", "segments": [{"text": "Thanks, Sato. Next is Kenji Tanaka."}, {"text": "Tanaka here; Acme Acme."}]}
        sp, _ = rename.speaker_for({"transcript": {"doc": tr}}, [TANAKA, SATO, ACME])
        self.assertEqual(sp["value"], "Kenji Tanaka")
        self.assertIn("named 2 times", sp["source"], "'Kenji Tanaka' counts once, not also as 'Tanaka'")

    def test_no_speaker_without_the_entity_memory_or_a_clear_lead(self):
        names = {"schema": rename.NAMES_SCHEMA, "speakers": [{"speaker": "SPEAKER_00", "status": "proposed",
                                                              "proposal": {"person_id": "x", "name": "Somebody Else", "basis": ["voice"], "confirmed": False}}]}
        sp, notes = rename.speaker_for({"names": {"doc": names}}, [TANAKA])
        self.assertIsNone(sp)
        self.assertTrue(any("not in the entity memory" in n for n in notes))
        tr = {"schema": "transcription-skill/transcript/0.1", "text": "Tanaka and Sato. Tanaka and Sato."}
        sp, notes = rename.speaker_for({"transcript": {"doc": tr}}, [TANAKA, SATO])
        self.assertIsNone(sp, "a tie names nobody")
        tr = {"schema": "transcription-skill/transcript/0.1", "text": "佐藤さん、佐藤さん"}
        sp, _ = rename.speaker_for({"transcript": {"doc": tr}}, [TANAKA, SATO])
        self.assertEqual(sp["value"], "Yui Sato", "a CJK alias is matched inside the text")

    def test_scenes_from_capture_time_gaps(self):
        clips = [clip("a.mp4", local="2026-11-22T10:00:00", dur=600), clip("b.mp4", local="2026-11-22T10:10:30", dur=600),
                 clip("c.mp4", local="2026-11-22T11:00:00", dur=60), clip("d.mp4")]
        sc = rename.scenes(clips, 120)
        self.assertEqual({k: v["value"] for k, v in sc.items()}, {"a.mp4": 1, "b.mp4": 1, "c.mp4": 2})
        self.assertIn("new scene", sc["c.mp4"]["source"])
        doc, per_clip, _ = rename.fields_document({"clips": clips}, [], [], 120)
        self.assertEqual(doc["schema"], rename.FIELDS_SCHEMA)
        self.assertNotIn("d.mp4", doc["clips"])
        doc, _, _ = rename.fields_document({"clips": clips}, [], [], 120, use_scenes=False)
        self.assertEqual(doc["clips"], {})


class McpWiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ws = self.tmp / "ws"
        self.ws.mkdir()
        self.folder = self.tmp / "ws" / "CAM_B"
        self.folder.mkdir()

    def test_classes_and_listing(self):
        self.assertEqual((TOOL_CLASSES["rename_preview"], TOOL_CLASSES["rename_apply"], TOOL_CLASSES["rename_undo"]), (READ, DESTRUCTIVE, DESTRUCTIVE))
        perms = load_permissions(str(self.ws), env={})
        listed = {t["name"]: t for t in listed_tools(perms)}
        self.assertIn("confirm_token", listed["rename_apply"]["inputSchema"]["properties"])
        self.assertTrue(listed["rename_apply"]["annotations"]["destructiveHint"])
        self.assertTrue(listed["rename_preview"]["annotations"]["readOnlyHint"])

    def test_argv_spells_every_value_as_an_option_value(self):
        argv = tool_argv("rename_apply", {"folder": "--evil", "plan_id": "abc", "event": "-x", "set": ["k=v"], "evidence": "/e", "no_scenes": True}, str(self.tmp))
        self.assertEqual(argv[:4], ["rename", "apply", "--yes", "--plan-id=abc"])
        self.assertEqual(argv[-2:], ["--", "--evil"])
        self.assertIn("--event=-x", argv)
        self.assertIn("--set=k=v", argv)
        self.assertIn("--evidence=/e", argv)
        self.assertIn("--no-scenes", argv)
        with self.assertRaises(ValueError):
            tool_argv("rename_apply", {"folder": "/f"}, str(self.tmp))
        self.assertEqual(tool_argv("rename_undo", {"folder": "/f"}, str(self.tmp)), ["rename", "undo", "--yes", "--", "/f"])
        self.assertNotIn("--yes", tool_argv("rename_preview", {"folder": "/f"}, str(self.tmp)))

    def test_apply_is_previewed_then_runs_only_with_its_token(self):
        fake = {"plan": {"root": str(self.folder), "mode": "in_place", "plan_id": "p1", "counts": {"rename": 1}, "files_to_move": 1,
                         "rows": [{"files": [{"from": "a.mp4", "to": "b.mp4"}]}]}, "plan_id": "p1"}
        perms = load_permissions(str(self.ws), scopes="read,write,destructive", env={})
        server = Server(workspace=str(self.ws), permissions=perms)
        args = {"folder": str(self.folder), "plan_id": "p1", "event": "Conf"}
        with mock.patch("video_agent.agent.rename.preview", return_value=fake), mock.patch("video_agent.mcp_server.subprocess.run", return_value=OK) as run:
            res = server.call_tool("rename_apply", dict(args))
            body = json.loads(res["content"][0]["text"])
            self.assertEqual(body["status"], "CONFIRMATION_REQUIRED")
            self.assertEqual(body["preview"][0]["renames"], [{"from": "a.mp4", "to": "b.mp4"}])
            run.assert_not_called()
            server.call_tool("rename_apply", dict(args, confirm_token=body["confirm_token"]))
            run.assert_called_once()
            self.assertIn("--yes", run.call_args[0][0])
        # the folder's plan changed after the preview: the old token no longer fits
        with mock.patch("video_agent.agent.rename.preview", return_value=fake):
            body = json.loads(server.call_tool("rename_apply", dict(args))["content"][0]["text"])
        moved = json.loads(json.dumps(fake))
        moved["plan"]["rows"][0]["files"][0]["to"] = "c.mp4"
        with mock.patch("video_agent.agent.rename.preview", return_value=moved), mock.patch("video_agent.mcp_server.subprocess.run", return_value=OK) as run:
            res = server.call_tool("rename_apply", dict(args, confirm_token=body["confirm_token"]))
            run.assert_not_called()
        self.assertIn("CONFIRM_TOKEN_MISMATCH", res["content"][0]["text"])

    def test_destructive_scope_is_off_by_default_and_outside_roots_refused(self):
        perms = load_permissions(str(self.ws), env={})
        with mock.patch("video_agent.agent.rename.preview", side_effect=rename.RenameError("ENGINE_MISSING", "no ingest")):
            call = classify("rename_apply", {"folder": str(self.folder), "plan_id": "p"}, perms, str(self.ws))
        self.assertEqual(call.cls, DESTRUCTIVE)
        self.assertIn("refused", call.preview[0]["detail"])
        server = Server(workspace=str(self.ws), permissions=perms)
        with mock.patch("video_agent.agent.rename.preview", side_effect=rename.RenameError("ENGINE_MISSING", "no ingest")):
            res = server.call_tool("rename_apply", {"folder": str(self.folder), "plan_id": "p"})
        self.assertIn("SCOPE_DISABLED", res["content"][0]["text"])
        outside = self.tmp / "elsewhere"
        outside.mkdir()
        server = Server(workspace=str(self.ws), permissions=load_permissions(str(self.ws), scopes="all", env={}))
        with mock.patch("video_agent.agent.rename.preview", side_effect=rename.RenameError("ENGINE_MISSING", "no ingest")):
            res = server.call_tool("rename_undo", {"folder": str(outside)})
        self.assertIn("PATH_OUTSIDE_ROOTS", res["content"][0]["text"])


def _ingest_with_rename():
    try:
        skill = rename.skill_or_error()
        rename.check_rename_tool(skill)
        return skill
    except rename.RenameError:
        return None


def run_cli(*argv, stdin=None):
    buf = io.StringIO()
    with redirect_stdout(buf), mock.patch("sys.stdin", stdin or io.StringIO("")):
        code = cli_main(list(argv))
    return code, buf.getvalue()


class RenameRealSkillTests(unittest.TestCase):
    """The agent's CLI on an offloaded copy, through the real ingest-skill (offload, inventory, rename plan / apply / undo, verify)."""

    def setUp(self):
        self.skill = _ingest_with_rename()
        if self.skill is None:
            raise unittest.SkipTest("needs ingest-skill 1.1.0+ (ingest/rename) at VIDEO_AGENT_INGEST_DIR or on PATH")
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ws = str(self.tmp / "ws")
        card = self.tmp / "CAM_B"
        (card / "DCIM").mkdir(parents=True)
        (card / "DCIM" / "C0001.MP4").write_bytes(os.urandom(5000))
        (card / "DCIM" / "C0001.XMP").write_bytes(b"<xmp/>")
        (card / "DCIM" / "C0002.MP4").write_bytes(os.urandom(4000))
        (card / "DCIM" / "DSC_0003.MOV").write_bytes(os.urandom(3000))
        (self.tmp / "ssd").mkdir()
        code, out, err = rename.invoke(self.skill, ["offload", str(card), "--dest", str(self.tmp / "ssd"), "--json"], timeout=120)
        self.assertEqual(code, 0, out + err)
        self.copy = self.tmp / "ssd" / "CAM_B"
        sha = hashlib.sha256((self.copy / "DCIM" / "C0001.MP4").read_bytes()).hexdigest()
        ev = self.tmp / "evidence"
        ev.mkdir()
        (ev / "a.turns.json").write_text(json.dumps(turns(fp=sha, filename="C0001.MP4", spans=[(0, 40, "SPEAKER_01"), (40, 45, "SPEAKER_00")])), encoding="utf-8")
        (ev / "C0002.transcript.json").write_text(json.dumps({"schema": "transcription-skill/transcript/0.1",
                                                               "segments": [{"text": "佐藤さん、ありがとうございます。"}, {"text": "佐藤です。"}]}, ensure_ascii=False), encoding="utf-8")
        self.ev = str(ev)
        self.assertEqual(run_cli("--workspace", self.ws, "entities", "add", "--type", "person", "--speaker-id", "SPEAKER_01", "--alias", "Tanaka", "--", "Tanaka")[0], 0)
        self.assertEqual(run_cli("--workspace", self.ws, "entities", "add", "--type", "person", "--alias", "佐藤", "--", "Sato")[0], 0)
        self.opts = ["--event", "ConfName", "--room", "Room A", "--camera", "CamB", "--evidence", self.ev]

    def names(self):
        return sorted(p.name for p in (self.copy / "DCIM").iterdir())

    def verify_ok(self):
        code, out, _ = rename.invoke(self.skill, ["verify", str(self.copy), "--json"], timeout=120)
        doc = json.loads(out)
        self.assertTrue(doc["ok"], doc)
        self.assertEqual(doc["unlisted"], [])

    def test_preview_apply_undo_round_trip(self):
        before = self.names()
        code, out = run_cli("--workspace", self.ws, "--json", "rename", "preview", str(self.copy), *self.opts)
        self.assertEqual(code, 0, out)
        res = json.loads(out)
        rows = {r["from"]: r for r in res["plan"]["rows"]}
        self.assertEqual(rows["DCIM/C0001.MP4"]["to"], "DCIM/ConfName_Room-A_CamB_Speaker-Tanaka_0001.mp4")
        self.assertEqual(rows["DCIM/C0002.MP4"]["to"], "DCIM/ConfName_Room-A_CamB_Speaker-Sato_0002.mp4")
        self.assertEqual(rows["DCIM/DSC_0003.MOV"]["to"], "DCIM/ConfName_Room-A_CamB_0003.mov", "no evidence: no speaker, the separator goes with it")
        self.assertIn("speaker_id SPEAKER_01", rows["DCIM/C0001.MP4"]["fields"]["speaker"]["source"])
        self.assertEqual(self.names(), before, "the preview renames nothing")
        # without --yes and without a terminal, nothing happens
        code, out = run_cli("--workspace", self.ws, "--json", "rename", "apply", str(self.copy), *self.opts)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out)["status"], "NOT_CONFIRMED")
        self.assertEqual(self.names(), before)
        code, out = run_cli("--workspace", self.ws, "--json", "rename", "apply", str(self.copy), *self.opts, "--plan-id", "0" * 16, "--yes")
        self.assertEqual((code, json.loads(out)["status"]), (2, "PLAN_CHANGED"))
        code, out = run_cli("--workspace", self.ws, "--json", "rename", "apply", str(self.copy), *self.opts, "--plan-id", res["plan_id"], "--yes")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.names(), ["ConfName_Room-A_CamB_0003.mov", "ConfName_Room-A_CamB_Speaker-Sato_0002.mp4",
                                        "ConfName_Room-A_CamB_Speaker-Tanaka_0001.mp4", "ConfName_Room-A_CamB_Speaker-Tanaka_0001.xmp"])
        self.verify_ok()
        code, out = run_cli("--workspace", self.ws, "--json", "rename", "undo", str(self.copy), "--yes")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.names(), before)
        self.verify_ok()

    def test_an_interactive_yes_applies_the_preview_shown(self):
        tty = io.StringIO("y\n")
        tty.isatty = lambda: True
        code, out = run_cli("--workspace", self.ws, "rename", "apply", str(self.copy), *self.opts, stdin=tty)
        self.assertEqual(code, 0, out)
        self.assertIn("rename 4 file(s) as shown?", out)
        self.assertIn("ConfName_Room-A_CamB_0003.mov", self.names())
        self.verify_ok()

    def test_without_entities_or_operator_values_nothing_changes(self):
        before = self.names()
        code, out = run_cli("--workspace", self.ws, "--json", "rename", "apply", str(self.copy), "--no-entities", "--yes")
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads(out)["status"], "NOTHING_TO_DO")
        self.assertEqual(self.names(), before)


if __name__ == "__main__":
    unittest.main()
