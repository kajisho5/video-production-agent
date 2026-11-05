"""ADR-096 (roadmap step 8, speaker ids end to end): `speakers.assign=true` hands each input's measured diarization-skill turns to
transcription-skill's transcribe (`speaker_turns`, merged after recognition with diarization-skill's ASSIGN rule), so the transcript's
segments carry the anonymous ids, and from it the SpeechEvents, the subtitle cues (`speaker`) and the take picks. Without the switch the
null-only speaker_id invariants stay. Fake diarization / transcription / subtitle processes and the fake media engine; no engine or network."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import clear_modes, fake_media  # noqa: E402
from test_diarization import MODES as DIA_MODES, dia_service  # noqa: E402

from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.speakers import parse_speaker_requirements, speaker_assign_requested  # noqa: E402
from video_agent.agent.subtitles import cues_from_segments  # noqa: E402
from video_agent.models import Asset, Observation  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.service import _check_requirement_keys  # noqa: E402
from video_agent.temporal import events_from_observation  # noqa: E402
from video_agent.tools.base import ToolError  # noqa: E402
from video_agent.tools.transcription.adapter import TranscriptionAdapter, check_transcript, pinned_contract  # noqa: E402

SEGMENTS = [[0.5, 4.5, "おはようございます"], [5.5, 10.5, "今日はよろしくお願いします"], [11.5, 15.5, "では始めましょう"]]
TURNS = [[0.4, 5.0, "SPEAKER_00"], [5.2, 11.0, "SPEAKER_01"], [11.1, 15.9, "SPEAKER_00"]]
FP = "sha256:" + "a" * 64


def turns_doc(fp: str = FP) -> dict:
    return {"schema": "diarization-skill/turns@1", "source": {"fingerprint": fp, "filename": "talk.mp4"}, "speakers": ["SPEAKER_00", "SPEAKER_01"],
            "turns": [{"start": a, "end": b, "speaker": s} for a, b, s in TURNS], "overlap": {"available": False, "regions": []}, "turns_file": "/ws/x.turns.json"}


class _Base(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self._env = {k: os.environ.get(k) for k in DIA_MODES + ("FAKE_TS_MODE",)}
        for k in self._env:
            os.environ.pop(k, None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())

    def tearDown(self):
        clear_modes()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class RequirementTests(_Base):
    def test_assign_needs_diarize(self):
        self.assertTrue(speaker_assign_requested({"speakers.diarize": True, "speakers.assign": "true"}))
        self.assertFalse(speaker_assign_requested({"speakers.diarize": True}))
        for reqs, why in (({"speakers.assign": True}, "needs speakers.diarize=true"), ({"speakers.diarize": True, "speakers.assign": "maybe"}, "is a switch")):
            with self.subTest(reqs=reqs):
                with self.assertRaises(EditRequirementError) as cm:
                    parse_speaker_requirements(reqs)
                self.assertIn(why, str(cm.exception))
                with self.assertRaises(ValueError):
                    _check_requirement_keys(reqs)


class AdapterTests(_Base):
    def test_turns_sent_inline_and_checked(self):
        declared = {"speaker_turns": "..."}
        sent = TranscriptionAdapter.check_turns(turns_doc(), declared)
        self.assertNotIn("turns_file", sent, "the agent's own note never crosses the boundary")
        self.assertEqual(sent["speakers"], ["SPEAKER_00", "SPEAKER_01"])
        for bad, why in ((dict(turns_doc(), schema="x"), "diarization-skill/turns@1"), (dict(turns_doc(), speakers=["Alice"]), "anonymous"),
                         (dict(turns_doc(), turns=[{"start": 0, "end": 1, "speaker": "SPEAKER_09"}]), "one of the document's speakers"),
                         (dict(turns_doc(), source={}), "fingerprint")):
            with self.subTest(why=why):
                with self.assertRaises(ToolError) as cm:
                    TranscriptionAdapter.check_turns(bad, declared)
                self.assertIn(why, str(cm.exception))
        with self.assertRaises(ToolError):
            TranscriptionAdapter.check_turns(turns_doc(), {})
        self.assertIn("speaker_turns", next(t for t in pinned_contract()["tools"] if t["name"] == "transcription/transcribe")["input"], "transcription 0.8.0 takes them")

    def _tr(self, **sa):
        c = pinned_contract()
        eng = {e["id"]: e for e in c["engines"]}
        e0 = next(iter(eng))
        tr = {"schema": c["schemas"]["transcript"], "id": "tr_1", "asset_id": "a", "language": "ja", "language_source": "requested", "duration": 16.0,
              "source": {"fingerprint": FP, "filename": "talk.mp4"}, "engine": e0, "engine_version": "1", "created_at": "x",
              "provenance": {"skill": "transcription-skill", "tool": "transcription/transcribe", "skill_version": c["version"], "execution_mode": eng[e0]["execution_mode"],
                             "engine": e0, "engine_version": "1", "model": "base", "parameters": {}, "cache_key": "k"},
              "segments": [{"id": "s1", "start": 0.5, "end": 4.5, "text": "x", "speaker_id": "SPEAKER_00"}]}
        if sa is not None:
            tr["speaker_assignment"] = dict({"skill": "transcription-skill", "tool": "transcription/speakers", "source_fingerprint": FP, "source_match": True,
                                             "speakers": ["SPEAKER_00", "SPEAKER_01"]}, **sa)
        return tr, c, eng

    def test_merged_transcript_admitted_only_with_the_turns_sent(self):
        tr, c, eng = self._tr()
        params = {"asset_id": "a", "speaker_turns": TranscriptionAdapter.check_turns(turns_doc(), {"speaker_turns": 1})}
        self.assertEqual(check_transcript(tr, params, c["version"], eng, c["schemas"]), [])
        errs = check_transcript(tr, {"asset_id": "a"}, c["version"], eng, c["schemas"])
        self.assertTrue(any("no speaker_turns were sent" in e for e in errs), errs)
        plain, _, _ = self._tr()
        plain.pop("speaker_assignment")
        self.assertTrue(any("never identifies speakers" in e for e in check_transcript(plain, {"asset_id": "a"}, c["version"], eng, c["schemas"])))
        for sa, why in (({"source_fingerprint": "sha256:" + "0" * 64}, "fingerprints differ"), ({"tool": "transcription/transcribe"}, "not transcription-skill"),
                        ({"speakers": ["SPEAKER_00"]}, "speakers differ")):
            with self.subTest(why=why):
                bad, _, _ = self._tr(**sa)
                self.assertTrue(any(why in e for e in check_transcript(bad, params, c["version"], eng, c["schemas"])))
        unknown, _, _ = self._tr()
        unknown["segments"][0]["speaker_id"] = "SPEAKER_42"
        self.assertTrue(any("not one of the speaker turns" in e for e in check_transcript(unknown, params, c["version"], eng, c["schemas"])))


class EventAndCueTests(_Base):
    def test_speech_events_carry_ids_only_from_a_recorded_merge(self):
        asset = Asset(path="/x/talk.mp4", provenance="USER")
        merged = {"id": "tr_1", "segments": [{"id": "s1", "start": 0.5, "end": 4.5, "text": "x", "speaker_id": "SPEAKER_00"}],
                  "speaker_assignment": {"speakers": ["SPEAKER_00"]}}
        [e] = events_from_observation(Observation(kind="transcript", asset_id=asset.id, source="transcription/transcribe@0.8.0", tool="transcription/transcribe", data=merged), asset)
        self.assertEqual(e.metadata["speaker_id"], "SPEAKER_00")
        with self.assertRaises(ValueError):
            events_from_observation(Observation(kind="transcript", asset_id=asset.id, source="transcription/transcribe@0.8.0", tool="transcription/transcribe", data={k: v for k, v in merged.items() if k != "speaker_assignment"}), asset)

    def test_cues_carry_the_anonymous_speaker(self):
        segs = [{"id": "s1", "start": 0.5, "end": 4.5, "text": "x", "speaker_id": "SPEAKER_00"}, {"id": "s2", "start": 5.0, "end": 6.0, "text": "y", "speaker_id": None}]
        self.assertEqual([c.get("speaker") for c in cues_from_segments(segs, [[0.0, 16.0]], speakers=True)], ["SPEAKER_00", None])
        self.assertEqual([c.get("speaker") for c in cues_from_segments(segs, [[0.0, 16.0]])], [None, None], "no merge recorded: no speaker")


class PlanTests(_Base):
    def plan(self, reqs, **facts):
        self.svc = dia_service(self.tmp)
        self.media = fake_media(self.tmp, duration=16.0, segments=SEGMENTS, turns=TURNS, **facts)
        return self.svc.plan([self.media], "generic", user_requirements=dict({"speakers.diarize": True, "subtitle": True}, **reqs), params={"language": "ja"})

    def test_subtitles_carry_speaker_ids_end_to_end(self):
        ir = self.plan({"speakers.assign": True})
        d = ir.doc
        [tr] = [o for o in d["analysis"]["observations"] if o["kind"] == "transcript"]
        [turns] = [o for o in d["analysis"]["observations"] if o["kind"] == "speaker_turns"]
        self.assertEqual([s["speaker_id"] for s in tr["data"]["segments"]], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])
        self.assertEqual((tr["data"]["speaker_assignment"]["tool"], tr["parameters"]["speaker_turns"]), ("transcription/speakers", turns["id"]))
        speech = [e for e in d["timeline"]["events"] if e["type"] == "SPEECH"]
        self.assertEqual([e["metadata"]["speaker_id"] for e in speech], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])
        [gen] = [op for op in d["captions"]["operations"] if op["type"] == "captions.generate"]
        self.assertEqual([c.get("speaker") for c in gen["cues"]], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])
        self.assertEqual(self.svc.validate(ir).errors, [])
        p = str(Path(self.tmp) / "p.json")
        save_ir(ir, p)
        out = self.svc.render(load_ir(p), p, approve=["all"])
        self.assertEqual(out["status"], "COMPLETED", out.get("execution"))
        srt = Path(out["paths"][gen["output"]]).read_text(encoding="utf-8")
        self.assertIn("SPEAKER_00: おはようございます", srt)
        self.assertIn("SPEAKER_01: 今日はよろしくお願いします", srt)

    def test_without_assign_the_ids_stay_null(self):
        ir = self.plan({})
        [tr] = [o for o in ir.doc["analysis"]["observations"] if o["kind"] == "transcript"]
        self.assertTrue(all(s["speaker_id"] is None for s in tr["data"]["segments"]))
        self.assertNotIn("speaker_assignment", tr["data"])
        [gen] = [op for op in ir.doc["captions"]["operations"] if op["type"] == "captions.generate"]
        self.assertFalse([c for c in gen["cues"] if "speaker" in c])

    def test_tampered_cue_speaker_refused(self):
        ir = self.plan({})
        [gen] = [op for op in ir.doc["captions"]["operations"] if op["type"] == "captions.generate"]
        gen["cues"][0]["speaker"] = "SPEAKER_00"
        self.assertTrue(any("recorded speaker merge" in e for e in self.svc.validate(ir).errors))

    def test_take_picks_carry_speaker_ids(self):
        self.svc = dia_service(self.tmp)
        line = [[0.5, 3.0, "今日は晴れです"]]
        a = fake_media(self.tmp, "a.mp4", duration=4.0, segments=line, turns=[[0.3, 3.2, "SPEAKER_00"]], silences=[[0.0, 0.4], [3.2, None]])
        b = fake_media(self.tmp, "b.mp4", duration=4.0, segments=line, turns=[[0.3, 3.2, "SPEAKER_00"]], silences=[[0.0, 0.4], [3.3, None]], true_peak=0.4)
        ir = self.svc.plan([a, b], "youtube", user_requirements={"edit.takes": True, "speakers.diarize": True, "speakers.assign": True}, params={"language": "ja"})
        [pick] = [x for x in ir.doc["decisions"] if x["subject"] == "takes.pick"]
        self.assertEqual(pick["params"]["speakers"], ["SPEAKER_00"])
        self.assertEqual(self.svc.validate(ir).errors, [])


if __name__ == "__main__":
    unittest.main()
