"""ADR-134: the microphones of a clip -- declared on the asset in the Project IR, mixed by audio-production AUTOMIX (values from the profile's
automix preset, recorded as PROFILE requirements), muxed under the picture before every other edit, the interpreters' language tracks
(MIX_TRACKS, cut like the picture), and the transcript by microphone (transcription/microphones, overlapping speech kept). Against the fake
engine (tests/fake_adapter.py), the fake audio-production and transcription processes; no ffmpeg, no engine."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import AUDIO_PRODUCTION_CAPS, PipelineCaps, clear_modes, fake_media, pipeline_service  # noqa: E402

from video_agent.agent import microphones as mic  # noqa: E402
from video_agent.agent.brief import parse_brief  # noqa: E402
from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.models import Requirement  # noqa: E402
from video_agent.profiles import load_profile  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402
from video_agent.service import BRIEF_REQUIREMENT_KEYS, Service, _check_requirement_keys  # noqa: E402

CAPS = AUDIO_PRODUCTION_CAPS + ["audio-production:AUTOMIX", "audio-production:MIX_TRACKS", "audio-production:NORMALIZE"]
ROOT = Path(__file__).resolve().parents[1]


def channel_mics(n_floor: int = 3, interp: bool = True):
    out = [{"id": "chair", "name": "Chair", "channel": 0, "pinned": True, "role": "presenter"}] + \
          [{"id": f"p{k}", "name": f"Panel {k}", "channel": k} for k in range(1, n_floor)]
    if interp:
        out.append({"id": "ja", "name": "Interpreter JA", "role": "interp", "language": "jpn", "channel": n_floor, "gain_db": 2.0, "pan": 0.0})
    return out


class _Base(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())

    def tearDown(self):
        clear_modes()

    def service(self, missing=()):
        return pipeline_service(self.tmp, caps=PipelineCaps(missing=missing, extra=CAPS))

    def plan(self, reqs, channels=4, profile="generic", inputs=None, **kw):
        self.svc = self.service()
        self.clip = fake_media(self.tmp, name="panel.mp4", channels=channels, duration=16.0)
        return self.svc.plan(inputs or [self.clip], profile, user_requirements=reqs, params={"language": "en"}, **kw)

    def render(self, ir):
        p = str(Path(self.tmp) / "p.json")
        save_ir(ir, p)
        return self.svc.render(load_ir(p), p, approve=["all"]), p

    def operations(self, out):
        """The executed operations as provenance records them (tool, skill, typed args)."""
        prov = json.loads((Path(self.svc.workspace) / "jobs" / out["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        return prov["operations"]


class VocabularyTests(_Base):
    def test_ranges_are_the_pinned_contracts(self):
        from video_agent.tools.audio_production.adapter import pinned_contract
        from video_agent.tools.transcription.adapter import MICROPHONES_TOOL, pinned_contract as ts_contract
        ops = {o["type"]: o for o in pinned_contract()["operations"]}
        am = ops["AUTOMIX"]
        self.assertEqual((am["inputs"]["min"], am["inputs"]["max"]), mic.AUTOMIX_INPUTS)
        self.assertEqual((am["parameters"]["window_ms"]["min"], am["parameters"]["window_ms"]["max"]), mic.WINDOW_RANGE)
        self.assertEqual((am["parameters"]["max_gain_rate_db_per_s"]["min"], am["parameters"]["max_gain_rate_db_per_s"]["max"]), mic.RATE_RANGE)
        self.assertTrue(am["parameters"]["window_ms"]["required"] and am["parameters"]["max_gain_rate_db_per_s"]["required"], "no default in the Skill")
        self.assertNotIn("default", am["parameters"]["window_ms"])
        props = am["parameters"]["channels"]["items"]["properties"]
        self.assertEqual((props["weight_db"]["min"], props["weight_db"]["max"]), mic.WEIGHT_RANGE)
        mt = ops["MIX_TRACKS"]
        self.assertEqual((mt["inputs"]["min"], mt["inputs"]["max"]), mic.TRACKS_INPUTS)
        self.assertIn(mic.TRACKS_FORMAT, pinned_contract()["language_tracks"]["set_formats"])
        self.assertIn(MICROPHONES_TOOL, [t["name"] for t in ts_contract()["tools"]])

    def test_microphones_are_normalised(self):
        src = fake_media(self.tmp, name="lav.wav", video=False, channels=1)
        mics = mic.parse_microphones(json.dumps([{"name": "Chair", "channel": 0}, {"id": "lav", "name": "田中", "speaker": "田中 一郎", "path": src, "weight_db": -3},
                                                 {"name": "Interp", "role": "INTERP", "language": "JPN", "channel": 2, "pan": -0.5}]))
        self.assertEqual([m["id"] for m in mics], ["mic1", "lav", "mic3"])
        self.assertEqual([m["role"] for m in mics], ["floor", "floor", "interp"])
        self.assertEqual((mics[1]["path"], mics[2]["language"], mics[2]["pan"]), (src, "jpn", -0.5))
        self.assertEqual(mic.transcript_name(mics[1]), "lav", "a name transcription-skill does not take is sent as the id")
        self.assertEqual(mic.speakers_of(mics), {"Chair": "Chair", "lav": "田中 一郎"})
        f = Path(self.tmp) / "mics.json"
        f.write_text(json.dumps([{"name": "A", "channel": 0}, {"name": "B", "channel": 1}]), encoding="utf-8")
        self.assertEqual(len(mic.parse_microphones(str(f))), 2, "a JSON file of microphones")

    def test_refusals(self):
        src = fake_media(self.tmp, name="lav.wav", video=False, channels=1)
        for value, why in (([{"name": "A"}], "exactly one of path"), ([{"name": "A", "channel": 0, "path": src}], "exactly one of path"),
                           ([{"name": "I", "role": "interp", "channel": 1}], "needs its language"), ([{"name": "A", "channel": 0, "pan": 0.5}], "has no pan"),
                           ([{"name": "A", "channel": 0}, {"name": "B", "channel": 0}], "same file / channel"), ([{"id": "x", "name": "A", "channel": 0}, {"id": "x", "name": "B", "channel": 1}], "twice"),
                           ([{"name": "A", "channel": 0, "gain": 3}], "unknown field"), ([{"name": "A", "path": str(Path(self.tmp) / "none.wav")}], "not a file"),
                           ([{"name": "A", "channel": 0, "language": "en"}], "ISO 639-2"), ("[]", "non-empty")):
            with self.subTest(value=value):
                with self.assertRaises(EditRequirementError) as cm:
                    mic.parse_microphones(value)
                self.assertIn(why, str(cm.exception))

    def test_switches_need_their_microphones_and_each_other(self):
        mics = channel_mics()
        for reqs, why in (({"audio.automix": True}, "no microphones are declared"), ({"audio.microphones": mics, "audio.automix.window_ms": 20}, "audio.automix is not"),
                          ({"audio.microphones": mics, "audio.language_tracks": True}, "needs audio.automix"),
                          ({"audio.microphones": channel_mics(interp=False), "audio.automix": True, "audio.language_tracks": True}, "role interp"),
                          ({"audio.microphones": channel_mics(n_floor=1), "audio.automix": True}, "1 are declared"),
                          ({"audio.microphones": mics, "audio.automix": True, "audio.automix.window_ms": 0}, "within"),
                          ({"audio.microphones": mics, "audio.automix": True, "audio.automix.preset": "panel"}, "recorded by the plan"),
                          ({"audio.microphones": mics, "audio.automix": True, "audio.language_tracks": True, "audio.language_tracks.floor_language": "en"}, "ISO 639-2"),
                          ({"audio.microphones": mics, "audio.automix": True, "audio.language_tracks.floor_gain_db": -10}, "audio.language_tracks is not"),
                          ({"audio.microphones": [dict(m, pinned=True) for m in channel_mics(interp=False)], "audio.automix": True}, "every microphone is pinned")):
            with self.subTest(reqs=sorted(reqs)):
                with self.assertRaises(ValueError) as cm:
                    _check_requirement_keys(reqs)
                self.assertIn(why, str(cm.exception))
        _check_requirement_keys({"audio.microphones": mics, "audio.automix": "interview", "audio.language_tracks": True, "audio.automix.hold_below_dbfs": "off"})

    def test_preset_values_are_profile_requirements(self):
        prof = load_profile("generic")
        presets = mic.presets_of(prof.data)
        self.assertEqual(presets["default"], "panel")
        for name in ("panel", "interview"):   # every production value explicit in the preset (the Skill has none)
            self.assertTrue({"window_ms", "max_gain_rate_db_per_s", "hold_below_dbfs", "floor_gain_db"} <= set(presets[name]), name)
        self.assertEqual(mic.presets_of(load_profile("conference").data), presets, "the conference profile inherits them")
        user = [Requirement(key="audio.automix", value=True, provenance="USER", source="cli"), Requirement(key="audio.automix.window_ms", value=12, provenance="USER", source="cli")]
        got = {r.key: r for r in mic.preset_requirements(user, prof.name, prof.data)}
        self.assertEqual(got["audio.automix.preset"].value, "panel")
        self.assertEqual((got["audio.automix.max_gain_rate_db_per_s"].value, got["audio.automix.max_gain_rate_db_per_s"].provenance), (presets["panel"]["max_gain_rate_db_per_s"], "PROFILE"))
        self.assertEqual(got["audio.automix.max_gain_rate_db_per_s"].source, "profile:generic:automix_presets.panel")
        self.assertNotIn("audio.automix.window_ms", got, "an explicit value wins")
        self.assertNotIn("audio.language_tracks.floor_gain_db", got, "the floor level only with language tracks")
        named = {r.key: r.value for r in mic.preset_requirements([Requirement(key="audio.automix", value="interview", provenance="USER", source="cli")], prof.name, prof.data)}
        self.assertEqual(named["audio.automix.window_ms"], presets["interview"]["window_ms"])
        self.assertEqual(mic.preset_requirements([Requirement(key="audio.automix", value="nosuch", provenance="USER", source="cli")], prof.name, prof.data)[0].value, "nosuch")
        self.assertEqual(mic.preset_requirements([Requirement(key="audio.automix", value=False, provenance="USER", source="cli")], prof.name, prof.data), [])


class PlanTests(_Base):
    def test_automix_and_language_tracks_end_to_end(self):
        ir = self.plan({"audio.microphones": channel_mics(), "audio.automix": True, "audio.language_tracks": True})
        d = ir.doc
        aid = next(iter(d["assets"]))
        decl = d["assets"][aid]["microphones"]
        self.assertEqual([(m["id"], m["role"], m["source"]) for m in decl][-1], ("ja", "interp", {"kind": "channel", "channel": 3}))
        dec = {x["subject"]: x for x in d["decisions"]}
        self.assertEqual((dec["audio.automix"]["type"], dec["audio.automix"]["approval"]), ("TRANSFORM", "AUTO"), "the explicit request is the confirmation")
        self.assertIn("automix_microphones", d["intent"]["secondary"])
        reqs = {r["key"]: r for r in d["requirements"]}
        self.assertEqual((reqs["audio.automix.window_ms"]["provenance"], reqs["audio.automix.preset"]["value"]), ("PROFILE", "panel"))
        [rec] = d["audio"]["automix"]
        self.assertEqual([x["id"] for x in rec["microphones"]], ["chair", "p1", "p2"], "the interpreter is not in the automix")
        self.assertEqual(rec["channels"][0], {"pinned": True})
        self.assertEqual((rec["window_ms"], rec["sample_rate"], rec["mux"]["video"]), (30, 48000, aid))
        tr = rec["language_tracks"]
        floor, ja = rec["output"], f"{aid}_mic_ja"
        self.assertEqual(tr["inputs"], [floor, floor, floor, ja], "a muted floor copy sets the duration; the floor track; the floor under the interpreter")
        self.assertEqual(tr["languages"], [{"language_id": "floor", "language": "und", "title": "Floor", "inputs": [1]},
                                           {"language_id": "jpn", "language": "jpn", "title": "Interpreter JA", "inputs": [2, 3]}])
        self.assertEqual(tr["levels"], [{"gain_db": 0.0, "mute": True}, {"gain_db": 0.0, "mute": False}, {"gain_db": -15, "mute": False}, {"gain_db": 2.0, "mute": False, "pan": 0.0}])
        self.assertEqual(tr["remove"], [[0.0, 2.85], [13.85, 16.0]], "the language tracks are cut where the picture is trimmed")
        skills = [s["skill"] for s in d["plan"]["steps"]]
        self.assertEqual(skills[:6], ["mic_extract"] * 3 + ["audio_automix", "audio_mux", "silence_cleanup"], "the mix becomes the clip's sound before the trim")
        trim = next(s for s in d["plan"]["steps"] if s["skill"] == "silence_cleanup")
        self.assertEqual((trim["inputs"], trim["params"]["accurate"]), ([f"{aid}_mic_mix"], True), "the trim cuts the mixed clip, frame-accurately (the tracks are sample-accurate)")
        self.assertIn({"role": "LANGUAGE_TRACKS", "logical": f"{aid}_language_tracks", "format": "mkv", "expected": {"source": aid, "languages": ["und", "jpn"], "tracks": 2}}, d["plan"]["outputs"])
        self.assertEqual(self.svc.validate(ir).errors, [])
        out, _ = self.render(ir)
        self.assertEqual(out["status"], "COMPLETED", out)
        res = out["execution"]["results"]
        ops = self.operations(out)
        ext = [r for r in ops if r["tool"] == "ffmpeg-skill/audio" and "channel" in r["args"]]
        self.assertEqual([r["args"]["channel"] for r in ext], [0, 1, 2, 3])
        self.assertTrue(all(r["args"]["sample_rate"] == 48000 for r in ext))
        am = next(r for r in ops if r["skill"] == "audio_automix")
        self.assertEqual((am["args"]["window_ms"], am["args"]["max_gain_rate_db_per_s"], am["args"]["hold_below_dbfs"]), (30.0, 300.0, -55.0))
        am_res = next(r for r in res if (r.get("data") or {}).get("operation_type") == "AUTOMIX")
        self.assertIn("automix", am_res["data"]["operation"]["measurements"], "what the Skill measured is kept")
        mt = next(r for r in ops if r["skill"] == "audio_mix_tracks")
        self.assertEqual((mt["args"]["format"], [l["inputs"] for l in mt["args"]["languages"]]), ("mkv", [[1], [2, 3]]))
        cuts = [r for r in ops if r["skill"] == "audio_cut"]
        self.assertEqual(len(cuts), 2, "the floor and the interpreter are each cut once, like the picture")
        mux = next(r for r in ops if r["tool"] == "ffmpeg-skill/audio" and "replace" in r["args"])
        self.assertEqual(mux["args"]["replace"], f"{aid}_automix")
        arts = {a["type"]: a for a in out["artifacts"]}
        self.assertTrue(arts["LANGUAGE_TRACKS"]["path"].endswith("_language_tracks.mkv"))
        self.assertEqual(arts["LANGUAGE_TRACKS"]["tool"], "audio-production/run")

    def test_file_microphones_and_an_unchanged_plan_without_them(self):
        base = self.plan({})
        self.assertNotIn("automix", base.doc["audio"])
        self.assertTrue(all("microphones" not in a for a in base.doc["assets"].values()), "no microphones declared: the IR is as before")
        files = [fake_media(self.tmp, name=f"lav{k}.wav", video=False, channels=1, duration=16.0) for k in range(3)]
        mics = [{"id": f"m{k}", "name": f"Mic {k}", "path": f} for k, f in enumerate(files)]
        ir = self.plan({"audio.microphones": mics, "audio.automix": "interview"})
        d = ir.doc
        aid = next(iter(d["assets"]))
        self.assertTrue(all(len(m["source"]["sha256"]) == 64 for m in d["assets"][aid]["microphones"]))
        self.assertIn(str(Path(files[0]).parent), d["execution"]["allowed_inputs"])
        self.assertEqual(d["audio"]["automix"][0]["window_ms"], 20, "the named preset")
        self.assertEqual(self.svc.validate(ir).errors, [])
        out, _ = self.render(ir)
        self.assertEqual(out["status"], "COMPLETED", out)
        ext = [r["args"] for r in self.operations(out) if r["tool"] == "ffmpeg-skill/audio" and r["args"].get("mono")]
        self.assertEqual(len(ext), 3, "each file folded to one channel at the clip's rate")
        # a microphone file replaced after the plan changes the key of its step (never reused)
        from video_agent.execution import compile_ir
        ops1, _ = compile_ir(load_ir(str(Path(self.tmp) / "p.json")), str(Path(self.tmp) / "j1"))
        d["assets"][aid]["microphones"][0]["source"]["sha256"] = "0" * 64
        ops2, _ = compile_ir(ir, str(Path(self.tmp) / "j1"))
        k1 = [o.idempotency_key for o in ops1 if o.skill == "mic_extract"]
        k2 = [o.idempotency_key for o in ops2 if o.skill == "mic_extract"]
        self.assertNotEqual(k1[0], k2[0])
        self.assertEqual(k1[1:], k2[1:])

    def test_blocks(self):
        mics = channel_mics(interp=False)
        cases = (({"audio.microphones": mics, "audio.automix": True}, 2, "audio.automix", "channel(s)"),
                 ({"audio.microphones": mics, "audio.automix": "nosuch"}, 4, "audio.automix", "no automix preset 'nosuch'"),
                 ({"audio.microphones": mics, "audio.automix": True, "audio.production": True}, 4, "audio.automix", "audio.production"),
                 ({"audio.microphones": channel_mics(), "audio.automix": True, "audio.language_tracks": True, "edit.speed": 1.5}, 4, "audio.language_tracks", "video.speed"))
        for reqs, channels, subject, why in cases:
            with self.subTest(why=why):
                ir = self.plan(reqs, channels=channels)
                b = [x for x in ir.doc["decisions"] if x["approval"] == "BLOCK" and x["subject"] == subject]
                self.assertTrue(b and why in b[0]["reason"], [(x["subject"], x["reason"]) for x in ir.doc["decisions"] if x["approval"] == "BLOCK"])
                self.assertEqual(ir.doc["plan"]["status"], "BLOCKED")

    def test_engine_without_channel_blocks_at_plan_time(self):
        svc = self.service()
        clip = fake_media(self.tmp, name="panel.mp4", channels=4)
        orig = Service._mic_extract_ok
        Service._mic_extract_ok = staticmethod(lambda router, tools: False)
        try:
            ir = svc.plan([clip], "generic", user_requirements={"audio.microphones": channel_mics(interp=False), "audio.automix": True})
        finally:
            Service._mic_extract_ok = staticmethod(orig)
        [b] = [x for x in ir.doc["decisions"] if x["approval"] == "BLOCK"]
        self.assertIn("--channel", b["reason"])
        orig_t = Service._mic_track_ok
        Service._mic_track_ok = staticmethod(lambda router, tools: False)
        try:
            ir = svc.plan([clip], "generic", user_requirements={"audio.microphones": channel_mics(), "audio.automix": True, "audio.language_tracks": True})
        finally:
            Service._mic_track_ok = staticmethod(orig_t)
        [b] = [x for x in ir.doc["decisions"] if x["approval"] == "BLOCK"]
        self.assertEqual(b["subject"], "audio.language_tracks")
        self.assertIn("--track", b["reason"])
        self.assertIn("audio.automix", {x["subject"] for x in ir.doc["decisions"] if x["type"] == "TRANSFORM"}, "the automix itself is still planned")

    def test_tampered_ir_refused(self):
        ir = self.plan({"audio.microphones": channel_mics(), "audio.automix": True, "audio.language_tracks": True})
        rec = ir.doc["audio"]["automix"][0]
        for change, why in ((lambda r: r.update(window_ms=0), "window_ms"), (lambda r: r["microphones"][0].update(channel=3), "is channel 0"),
                            (lambda r: r["microphones"].append({"id": "ghost", "output": "x", "channel": 1}), "not declared"),
                            (lambda r: r["channels"].pop(), "one channels entry"), (lambda r: r["language_tracks"]["languages"][0].update(inputs=[]), "exactly one language"),
                            (lambda r: r["language_tracks"]["levels"][0].update(mute=False), "muted floor copy"),
                            (lambda r: r["language_tracks"].update(format="wav"), "written as mkv")):
            with self.subTest(why=why):
                bad = json.loads(json.dumps(rec))
                change(bad)
                ir.doc["audio"]["automix"] = [bad]
                errs = self.svc.validate(ir).errors
                self.assertTrue(any(why in e for e in errs), errs)
        ir.doc["audio"]["automix"] = [rec]
        self.assertEqual(self.svc.validate(ir).errors, [])


class TranscriptTests(_Base):
    def test_transcript_by_microphone_keeps_overlapping_speech(self):
        lav = fake_media(self.tmp, name="lav.wav", video=False, channels=1, duration=16.0)
        mics = [{"id": "chair", "name": "Chair", "channel": 0}, {"id": "smith", "name": "Dr. Smith", "path": lav}, {"id": "ja", "name": "JA", "role": "interp", "language": "jpn", "channel": 1}]
        ir = self.plan({"audio.microphones": mics, "audio.automix": True, "subtitle": True})
        d = ir.doc
        aid = next(iter(d["assets"]))
        tr = next(o for o in d["analysis"]["observations"] if o["kind"] == "transcript")
        self.assertEqual(tr["tool"], "transcription/microphones")
        self.assertEqual([m["name"] for m in tr["data"]["microphone_merge"]["microphones"]], ["Chair", "Dr. Smith"], "the interpreter is not transcribed")
        self.assertEqual(tr["parameters"]["speakers"], {"Chair": "Chair", "Dr. Smith": "Dr. Smith"})
        self.assertTrue(tr["data"]["overlapping_segments"], "the fake merge has speech said over someone else")
        [k] = [x for x in d["decisions"] if x["subject"] == "speech.microphones"]
        self.assertEqual(len(k["params"]["overlapping_segments"]), len(tr["data"]["overlapping_segments"]), "kept in the plan, not lost")
        self.assertEqual(k["params"]["overlapping_segments"][0]["speaker_id"], tr["data"]["overlapping_segments"][0]["speaker_id"])
        speech = [e for e in d["timeline"]["events"] if e["type"] == "SPEECH"]
        self.assertTrue({e["metadata"]["speaker_id"] for e in speech} <= {"Chair", "Dr. Smith"} and speech, "speech events carry the microphones' names")
        self.assertIn("Dr. Smith", {s_["speaker_id"] for s_ in tr["data"]["overlapping_segments"]})
        ch = Path(self.svc.workspace) / "cache" / "microphones"
        self.assertEqual(len(list(ch.glob("*_ch0.wav"))), 1, "the channel microphone is written once as a file to be transcribed")
        self.assertEqual(self.svc.validate(ir).errors, [])
        self.assertEqual(d["audio"]["automix"][0]["asset"], aid)

    def test_entity_spelling_and_refused_results(self):
        rec = {"entities": [{"id": "e1", "canonical": "Dr. Hanako Smith", "type": "person", "aliases": ["Smith"]}]}
        reqs = {"audio.microphones": mic.parse_microphones([{"name": "Smith", "channel": 0}, {"name": "Chair", "channel": 1}])}
        out = Service._entity_microphones(rec, reqs)
        self.assertEqual(out["audio.microphones"][0]["speaker"], "Dr. Hanako Smith")
        self.assertEqual(rec["rewrites"]["audio.microphones"]["entity"], "e1")
        self.assertEqual(mic.speakers_of(out["audio.microphones"]), {"Dr. Hanako Smith": "Dr. Hanako Smith", "Chair": "Chair"})
        from video_agent.tools.base import ToolError
        from video_agent.tools.transcription import TranscriptionAdapter, TranscriptionSkill
        ad = TranscriptionAdapter(TranscriptionSkill([sys.executable, str(ROOT / "tests" / "fake_transcription.py")], None, {}), workspace=str(Path(self.tmp) / "ws"),
                                  allowed_inputs=[str(Path(self.tmp) / "src")])
        a, b = fake_media(self.tmp, name="a.wav", video=False), fake_media(self.tmp, name="b.wav", video=False)
        for mics, why in (([{"input": a, "name": "A"}], "2..8"), ([{"input": a, "name": "田中"}, {"input": b, "name": "B"}], "Skill's rule"),
                          ([{"input": a, "name": "A"}, {"input": "/etc/passwd", "name": "B"}], "outside the allowed roots"), ([{"input": a, "name": "A"}, {"input": b, "name": "A"}], "same name")):
            with self.subTest(why=why):
                with self.assertRaises(ToolError) as cm:
                    ad.build_request("transcription/microphones", {"microphones": mics, "asset_id": "x", "language": "en"})
                self.assertIn(why, str(cm.exception))
        req = ad.build_request("transcription/microphones", {"microphones": [{"input": a, "name": "A"}, {"input": b, "name": "B"}], "asset_id": "x", "language": "en"})
        self.assertEqual((req["params"]["word_timestamps"], "input" in req["params"], [m["name"] for m in req["params"]["microphones"]]), (True, False, ["A", "B"]))
        os.environ["FAKE_TS_MODE"] = "mic_wrong_speaker"
        r = ad.measure("transcription/microphones", {"microphones": [{"input": a, "name": "A"}, {"input": b, "name": "B"}], "asset_id": "x", "language": "en"})
        self.assertFalse(r.ok)
        self.assertIn("not one of the microphones", r.data["error"]["message"])


class IntentTests(_Base):
    def test_request_text_and_brief_choose_the_path(self):
        ir = self.plan({"audio.microphones": channel_mics()}, request_text="パネルの音を整えて、通訳トラック付きで")
        subj = {x["subject"]: x for x in ir.doc["decisions"]}
        self.assertEqual((subj["audio.automix"]["type"], subj["audio.language_tracks"]["type"]), ("TRANSFORM", "TRANSFORM"))
        self.assertEqual({r["key"]: r["source"] for r in ir.doc["requirements"]}["audio.automix"], "request-text")
        self.assertEqual(set(ir.doc["intent"]["secondary"]) >= {"automix_microphones", "language_tracks"}, True)
        plain = self.plan({}, request_text="auto-mix the panel with interpreter tracks")
        self.assertNotIn("audio.automix", {x["subject"] for x in plain.doc["decisions"]}, "without declared microphones the words name nothing")
        for text, keys in (("Panel audio: automix, with interpretation tracks", {"audio.automix", "audio.language_tracks"}), ("自動ミックスで", {"audio.automix"}),
                           ("同時通訳つき、オートミックス", {"audio.automix", "audio.language_tracks"}), ("60 s vertical", set())):
            with self.subTest(text=text):
                got = {c["key"] for c in parse_brief(text)[0] if c["key"].startswith("audio.")}
                self.assertEqual(got, keys)
        self.assertTrue({"audio.automix", "audio.language_tracks"} <= BRIEF_REQUIREMENT_KEYS)
        self.assertNotIn("audio.microphones", BRIEF_REQUIREMENT_KEYS, "files are named only by --set")
        svc = self.service()
        rec = svc.extract_brief("automix the panel", {}, "generic")
        self.assertTrue(any(r["key"] == "audio.automix" and "no microphones are declared" in r["reason"] for r in rec["refused"]), rec)


class AdapterTests(_Base):
    def test_channel_flags_are_required_when_given(self):
        from test_unit import FfmpegSkillContractCatalogTests, fake_ffmpeg_contract
        from video_agent.tools.base import ToolError
        from video_agent.tools.ffmpeg_skill.adapter import FfmpegSkillAdapter
        from video_agent.tools.ffmpeg_skill.locate import FfmpegSkill
        helper = FfmpegSkillContractCatalogTests()
        helper.tmp = self.tmp
        new = fake_ffmpeg_contract("2.0")
        old = json.loads(json.dumps(new))
        tool = next(t for t in old["tools"] if t["name"] == "audio")
        for k in ("channel", "sample_rate"):
            del tool["input_schema"]["properties"][k]
        args = {"input": "/x/talk.mp4", "channel": 2, "sample_rate": 48000, "output": "/x/mic.wav"}
        a_new = FfmpegSkillAdapter(FfmpegSkill(helper._checkout(new, version="4.8.3"), "4.8.3", ["audio"]))
        argv = a_new.build_argv("ffmpeg-skill/audio", args, {})
        self.assertEqual(argv[argv.index("--channel") + 1], "2")
        self.assertEqual(argv[argv.index("--sample-rate") + 1], "48000")
        a_old = FfmpegSkillAdapter(FfmpegSkill(helper._checkout(old, version="3.6.0"), "3.6.0", ["audio"]))
        self.assertIsNone(a_old.unavailable_reason("ffmpeg-skill/audio"), "the mux (--replace) stays usable")
        with self.assertRaises(ToolError) as cm:
            a_old.build_argv("ffmpeg-skill/audio", args, {})
        self.assertIn("--channel", str(cm.exception))
        self.assertNotIn("--channel", a_old.build_argv("ffmpeg-skill/audio", {"input": "/x/a.mp4", "replace": "/x/b.wav", "output": "/x/c.mp4"}, {}))

    def test_language_indexes_are_lowered(self):
        from video_agent.tools.audio_production.adapter import pinned_contract
        from video_agent.tools.audio_production.lowering import Lowering
        from video_agent.tools.base import ToolError
        lw = Lowering(pinned_contract(), workspace=self.tmp)
        args = {"operation": "MIX_TRACKS", "levels": [{"gain_db": -15, "mute": False}, {"gain_db": 0, "mute": False, "pan": 0.25}],
                "languages": [{"language_id": "jpn", "language": "jpn", "title": "日本語", "inputs": [1]}]}
        self.assertEqual(lw.params_for("MIX_TRACKS", args)["languages"][0]["inputs"], [1])
        for bad in ([True], ["1"], [-1]):
            with self.assertRaises(ToolError):
                lw.params_for("MIX_TRACKS", dict(args, languages=[dict(args["languages"][0], inputs=bad)]))
        am = lw.params_for("AUTOMIX", {"window_ms": 30, "max_gain_rate_db_per_s": 300, "channels": [{"pinned": True}, {"weight_db": -3}]})
        self.assertEqual(am["channels"][1], {"weight_db": -3.0})
        with self.assertRaises(ToolError):
            lw.params_for("AUTOMIX", {"window_ms": 30})


if __name__ == "__main__":
    unittest.main()
