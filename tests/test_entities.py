"""ADR-128: entity memory. The store (schema, CRUD, CSV / JSON round trip, global vs project), the suggestions from a transcript, the
CLI and MCP surfaces, and the plan wiring against the fake Skills: transcription-skill's domain_vocabulary / corrections, subtitle-skill
emphasis words, the lower third's spelling and title, dubbing-skill's pronunciation dictionary, the privacy rules, and a plan with an empty
store being exactly the plan without one. No real Skill, no ffmpeg."""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import clear_modes, fake_media, pipeline_service  # noqa: E402

from video_agent import entities as ent  # noqa: E402
from video_agent.cli import main as cli_main  # noqa: E402
from video_agent.mcp_permissions import DESTRUCTIVE, READ, WRITE, classify, load_permissions  # noqa: E402
from video_agent.mcp_server import tool_argv  # noqa: E402
from video_agent.media.analysis import scrub_for_ai  # noqa: E402
from video_agent.tools import ToolError  # noqa: E402
from video_agent.tools.transcription import TranscriptionAdapter  # noqa: E402

JANE = {"canonical": "Jane Doe", "type": "person", "aliases": ["Jane Dough", "Jayne Doe"], "title": "CTO, Acme Corp",
        "pronunciation": [{"lang": "en", "reading": "JAYN-doh", "ipa": "dʒeɪn doʊ"}, {"lang": "ja", "reading": "ジェーン・ドウ", "accent": 1}]}
ACME = {"canonical": "Acme Corp", "type": "organization", "aliases": ["Acme Core"]}
TERM = {"canonical": "HbA1c", "type": "term", "aliases": ["H B A one C"]}


class _Tmp(unittest.TestCase):
    def setUp(self):
        clear_modes()
        os.environ.pop(ent.ENV_DIR, None)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.ws = str(Path(self.tmp) / "ws")
        os.makedirs(self.ws, exist_ok=True)

    def tearDown(self):
        clear_modes()
        for k in ("FAKE_TS_REQUESTS",):
            os.environ.pop(k, None)
        shutil.rmtree(self.tmp, ignore_errors=True)


class SchemaTests(_Tmp):
    def test_valid_and_invalid_documents(self):
        good = {"schema": ent.SCHEMA, "scope": "global", "entities": [dict(JANE, id="jane-doe"), dict(ACME, id="acme-corp")]}
        self.assertEqual(ent.validate_doc(good), [])
        for bad, needle in ((dict(good, schema="x"), "schema"),
                            ({**good, "entities": [dict(JANE, id="jane-doe", type="human")]}, "entities/0/type"),
                            ({**good, "entities": [dict(JANE, id="jane-doe", colour="red")]}, "Additional properties"),
                            ({**good, "entities": [dict(JANE, id="Jane Doe")]}, "entities/0/id"),
                            ({**good, "entities": [dict(JANE, id="a"), dict(ACME, id="a")]}, "is used by entity"),
                            ({**good, "entities": [dict(JANE, id="a"), dict(ACME, id="b", aliases=["jane dough"])]}, "already names entity"),
                            ({**good, "entities": [dict(JANE, id="a", speaker={"speaker_id": "Bob"})]}, "speaker"),
                            ({"schema": ent.SCHEMA, "scope": "project", "entities": []}, "a project store names its project")):
            errs = ent.validate_doc(bad)
            self.assertTrue(errs and any(needle in e for e in errs), (needle, errs))

    def test_a_hand_edit_that_breaks_the_schema_is_refused_with_its_path(self):
        st = ent.EntityStore(self.ws)
        st.add(JANE)
        doc = json.loads(st.path.read_text(encoding="utf-8"))
        doc["entities"][0]["type"] = "celebrity"
        st.path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaises(ent.EntityError) as cm:
            st.list()
        self.assertIn("entities/0/type", str(cm.exception))
        st.path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ent.EntityError):
            st.list()


class CrudTests(_Tmp):
    def test_add_edit_remove_by_id_spelling_or_alias(self):
        st = ent.EntityStore(self.ws)
        e = st.add(JANE)
        self.assertEqual((e["id"], e["canonical"], e["aliases"]), ("jane-doe", "Jane Doe", ["Jane Dough", "Jayne Doe"]))
        self.assertEqual(st.get("jayne doe")["id"], "jane-doe", "found by an alias, ignoring case")
        with self.assertRaises(ent.EntityError):
            st.add({"canonical": "Jane Dough", "type": "person"})   # a spelling that names another entity
        with self.assertRaises(ent.EntityError):
            st.add({"canonical": "X", "type": "celebrity"})
        st.add({"canonical": "Jane Doe Foundation", "type": "organization", "id": "jane-doe-2"})
        e = st.edit("Jane Dough", {"title": "CEO"}, add_aliases=["J. Doe"], remove_aliases=["Jayne Doe"])
        self.assertEqual((e["title"], e["aliases"]), ("CEO", ["Jane Dough", "J. Doe"]))
        e = st.edit("jane-doe", {"title": None, "pronunciation": None})
        self.assertNotIn("title", e)
        self.assertNotIn("pronunciation", e)
        with self.assertRaises(ent.EntityError):
            st.edit("jane-doe", {"colour": "red"})
        self.assertEqual(st.remove("J. Doe")["id"], "jane-doe")
        self.assertEqual([x["id"] for x in st.list()], ["jane-doe-2"])
        with self.assertRaises(ent.EntityError):
            st.remove("nobody")

    def test_project_store_is_separate_and_wins(self):
        ent.EntityStore(self.ws).add(dict(ACME, title="global title"))
        ent.EntityStore(self.ws, "conf26").add(dict(ACME, aliases=["Acme Corporation"]))
        ent.EntityStore(self.ws, "conf26").add({"canonical": "Ackme", "type": "product", "aliases": ["Acme Core"]})
        snap = ent.snapshot(self.ws, "conf26")
        by_id = {e["id"]: e for e in snap["entities"]}
        self.assertEqual(by_id["acme-corp"]["store"], "project:conf26")
        self.assertNotIn("title", by_id["acme-corp"], "the project's entity replaces the global one")
        self.assertEqual([s["store"] for s in snap["stores"]], ["global", "project:conf26"])
        self.assertEqual(ent.snapshot(self.ws, None)["entities"][0]["title"], "global title")
        with self.assertRaises(ent.EntityError):
            ent.EntityStore(self.ws, "../x")
        ent.EntityStore(self.ws).path.unlink()
        self.assertIsNone(ent.snapshot(self.ws, "empty-project"), "no entity in either store: nothing to apply")


class RoundTripTests(_Tmp):
    def test_csv_export_import_round_trip(self):
        a = ent.EntityStore(self.ws)
        for x in (JANE, ACME, TERM, {"canonical": "東京ビッグサイト", "type": "place", "aliases": ["東京ビックサイト"], "note": "venue, hall 7"},
                  {"canonical": "Bob Lee", "type": "person", "speaker": {"speaker_id": "SPEAKER_01"}}):
            a.add(x)
        text = a.export_csv()
        b = ent.EntityStore(self.ws, "copy")
        res = b.import_csv(text)
        self.assertEqual(len(res["added"]), 5)
        strip = lambda es: [{k: v for k, v in e.items() if k not in ("added_at", "updated_at")} for e in es]  # noqa: E731
        self.assertEqual(strip(b.list()), strip(a.list()))
        self.assertEqual(b.export_csv(), text, "export → import → export is byte-identical")
        again = b.import_csv(text)
        self.assertEqual((again["added"], again["updated"], len(again["unchanged"])), ([], [], 5))
        edited = text.replace("CTO, Acme Corp", "CEO")
        self.assertEqual(b.import_csv(edited)["updated"], ["jane-doe"])
        only = "canonical,type\nNew Name,person\n"
        res = b.import_csv(only, replace=True)
        self.assertEqual((res["added"], len(res["removed"])), (["new-name"], 5))

    def test_csv_refusals_name_the_line(self):
        st = ent.EntityStore(self.ws)
        for csv_text, needle in (("canonical,type\nX,celebrity\n", "line 2"), ("canonical\nX\n", "header"), ("canonical,type,colour\nX,term,red\n", "unknown CSV column"),
                                 ("canonical,type,ipa\nX,term,en=x\n", "without a reading"), ("id,canonical,type\na,X,term\na,Y,term\n", "more than once")):
            with self.assertRaises(ent.EntityError, msg=csv_text) as cm:
                st.import_csv(csv_text)
            self.assertIn(needle, str(cm.exception))
        self.assertFalse(st.path.exists(), "nothing written")

    def test_cli_crud_import_export(self):
        def run(*argv):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(["--workspace", self.ws, "--json", *argv])
            return rc, (json.loads(buf.getvalue()) if buf.getvalue().strip().startswith("{") else buf.getvalue())
        rc, out = run("entities", "add", "Jane Doe", "--type", "person", "--alias", "Jane Dough", "--title", "CTO", "--pronunciation", "en=JAYN-doh", "--ipa", "en=dʒeɪn")
        self.assertEqual((rc, out["added"]["pronunciation"]), (0, [{"lang": "en", "reading": "JAYN-doh", "ipa": "dʒeɪn"}]))
        rc, out = run("entities", "edit", "jane dough", "--add-alias", "Jayne Doe", "--clear", "title")
        self.assertEqual((rc, out["edited"]["aliases"], "title" in out["edited"]), (0, ["Jane Dough", "Jayne Doe"], False))
        csv_path = str(Path(self.tmp) / "e.csv")
        rc, out = run("entities", "export", "-o", csv_path)
        self.assertEqual((rc, out["exported"]), (0, 1))
        rc, out = run("entities", "import", "--project", "conf26", csv_path)
        self.assertEqual((rc, out["added"]), (0, ["jane-doe"]))
        json_path = str(Path(self.tmp) / "e.json")
        rc, out = run("entities", "export", "--project", "conf26", "--format", "json", "-o", json_path)
        rc, out = run("entities", "import", "--project", "other", json_path)
        self.assertEqual(out["added"], ["jane-doe"])
        rc, out = run("entities", "list", "--project", "other")
        self.assertEqual([e["canonical"] for e in out["entities"]], ["Jane Doe"])
        rc, out = run("entities", "remove", "Jayne Doe")
        self.assertEqual((rc, out["removed"]["id"]), (0, "jane-doe"))
        rc, out = run("entities", "remove", "Jayne Doe")
        self.assertEqual(rc, 2)
        self.assertIn("no entity", out["error"])
        vp = Path(self.tmp) / "bad.voice.json"
        vp.write_text(json.dumps({"schema": "something-else"}), encoding="utf-8")
        rc, out = run("entities", "add", "Bob", "--type", "person", "--voiceprint", str(vp))
        self.assertEqual(rc, 2)
        vp.write_text(json.dumps({"schema": ent.VOICEPRINT_SCHEMA, "name": "Bob"}), encoding="utf-8")
        rc, out = run("entities", "add", "Bob", "--type", "person", "--voiceprint", str(vp), "--speaker-id", "SPEAKER_00")
        self.assertEqual((rc, out["added"]["speaker"]), (0, {"speaker_id": "SPEAKER_00", "voiceprint": str(vp)}))


class UsesTests(unittest.TestCase):
    def snap(self, *es):
        return [dict(ent.normalize_entity(e), id=ent.slug(e["canonical"])) for e in es]

    def test_transcription_inputs_are_the_skills_own(self):
        vocab, rules = ent.transcription_inputs(self.snap(JANE, ACME, TERM, {"canonical": "東京ビッグサイト", "type": "place", "aliases": ["東京ビックサイト"]}))
        self.assertEqual(vocab, {"speakers": ["Jane Doe"], "terms": ["Acme Corp", "HbA1c", "東京ビッグサイト"], "max_terms": 4, "japanese": True})
        self.assertIn({"from": "Jane Dough", "to": "Jane Doe", "match": "word"}, rules)
        self.assertIn({"from": "東京ビックサイト", "to": "東京ビッグサイト", "match": "substring"}, rules, "no word boundaries in Japanese")
        self.assertEqual(ent.transcription_inputs([]), (None, None))

    def test_emphasis_lower_third_and_pronunciation(self):
        es = self.snap(JANE, ACME, TERM)
        self.assertEqual(ent.emphasis_words(es), ["Jane", "Doe", "Acme", "Corp"], "subtitle-skill matches one token; terms are not names")
        self.assertEqual(ent.find(es, "jane  dough")["id"], "jane-doe")
        self.assertIsNone(ent.find(es, "Bob"))
        d = ent.pronunciation_dictionary(es, "en-US")
        self.assertEqual(d["schema"], "dubbing/pronunciation@1")
        self.assertEqual({x["term"] for x in d["entries"]}, {"Jane Doe", "Jane Dough", "Jayne Doe"})
        self.assertEqual(d["entries"][0]["ipa"], "dʒeɪn doʊ")
        self.assertEqual(ent.pronunciation_dictionary(es, "ja")["entries"][0]["accent"], 1)
        self.assertIsNone(ent.pronunciation_dictionary(es, "fr"))

    def test_suggestions_are_only_candidates(self):
        tr = {"segments": [{"id": "s1", "start": 0.0, "end": 3.0, "text": "Welcome to the keynote by Priya Raman from Globex today."},
                           {"id": "s2", "start": 3.0, "end": 6.0, "text": "Thanks, Priya Raman. And now Jane Doe from Globex.",
                            "words": [{"text": "Zorblat", "start": 4.0, "end": 4.5, "probability": 0.31}, {"text": "now", "start": 4.5, "end": 4.7, "probability": 0.2}]}]}
        got = {c["text"]: c for c in ent.suggest([tr], self.snap(JANE))}
        self.assertEqual(got["Priya Raman"]["count"], 2)
        self.assertEqual(got["Priya Raman"]["suggested"], {"canonical": "Priya Raman", "type": "person"})
        self.assertEqual(got["Globex"]["count"], 2)
        self.assertEqual(got["Zorblat"]["reasons"], ["low confidence"])
        self.assertNotIn("Jane Doe", got, "a known spelling is not suggested")
        self.assertNotIn("now", got, "a lower-case word is not a name")
        self.assertNotIn("Welcome", got, "a sentence-initial capital is not a name")

    def test_names_never_reach_an_ai_provider(self):
        data = {"segments": [{"text": "Jane Doe spoke"}], "provenance": {"domain_vocabulary": {"terms": ["Jane Doe", "Secret Product"]}, "corrections": {"applied": []},
                                                                         "parameters": {"initial_prompt": "Jane Doe, Secret Product."}}}
        out = json.dumps(scrub_for_ai(data))
        self.assertNotIn("Secret Product", out)
        self.assertIn("Jane Doe spoke", out, "what was said is still the transcript")

    def test_adapter_sends_them_checked(self):
        from video_agent.tools.transcription.adapter import _check_corrections, _check_domain_vocabulary
        self.assertEqual(_check_domain_vocabulary({"speakers": ["Jane Doe"], "max_terms": 1}), {"speakers": ["Jane Doe"], "max_terms": 1})
        for bad in ({"programme": "free text"}, {"speakers": []}, {"speakers": ["x" * 101]}, {"terms": ["a"], "max_terms": 0}):
            with self.assertRaises(ToolError, msg=bad):
                _check_domain_vocabulary(bad)
        self.assertEqual(_check_corrections([{"from": "a", "to": "b"}]), [{"from": "a", "to": "b", "match": "word"}])
        with self.assertRaises(ToolError):
            _check_corrections([{"from": "a", "to": "b", "match": "regex"}])


class McpTests(_Tmp):
    def test_classes_and_argv(self):
        perms = load_permissions(self.ws, scopes="all", env={})
        self.assertEqual(classify("entities_list", {}, perms, self.ws).cls, READ)
        self.assertEqual(classify("entities_add", {"canonical": "X", "type": "term"}, perms, self.ws).cls, WRITE)
        self.assertEqual(classify("entities_import", {"file": "x.csv"}, perms, self.ws).cls, WRITE)
        self.assertEqual(classify("entities_import", {"file": "x.csv", "replace": True}, perms, self.ws).cls, DESTRUCTIVE)
        ent.EntityStore(self.ws).add(JANE)
        call = classify("entities_remove", {"ref": "Jane Dough"}, perms, self.ws)
        self.assertEqual(call.cls, DESTRUCTIVE)
        self.assertEqual(call.preview[0]["entity"], {"id": "jane-doe", "type": "person", "canonical": "Jane Doe"})
        self.assertEqual(classify("entities_export", {}, perms, self.ws).cls, READ)
        self.assertEqual(classify("entities_export", {"output": str(Path(self.ws) / "e.csv")}, perms, self.ws).cls, WRITE)
        argv = tool_argv("entities_add", {"canonical": "--weird", "type": "person", "aliases": ["A"], "pronunciation": [{"lang": "en", "reading": "r", "accent": 0}],
                                          "project": "conf26"}, self.tmp)
        self.assertEqual(argv, ["entities", "add", "--project=conf26", "--type=person", "--pronunciation=en=r", "--accent=en=0", "--alias=A", "--", "--weird"])
        self.assertEqual(tool_argv("entities_remove", {"ref": "x"}, self.tmp), ["entities", "remove", "--", "x"])

    def test_remove_runs_only_with_the_token(self):
        from unittest import mock
        from video_agent.mcp_server import Server
        ent.EntityStore(self.ws).add(JANE)
        server = Server(workspace=self.ws, permissions=load_permissions(self.ws, scopes="all", env={}))
        with mock.patch("video_agent.mcp_server.subprocess.run") as run:
            first = server.call_tool("entities_remove", {"ref": "jane-doe"})
            self.assertFalse(run.called, "previewed, not run")
        text = json.dumps(first)
        self.assertIn("confirm_token", text)


class _PlanBase(_Tmp):
    def setUp(self):
        super().setUp()
        self.log = str(Path(self.tmp) / "ts.log")
        os.environ["FAKE_TS_REQUESTS"] = self.log
        self.src = fake_media(self.tmp, "talk.mp4", segments=[[0.5, 3.0, "Jane Dough opened the Acme Core keynote."], [3.5, 6.0, "Thank you, Jane Doe."]])

    def svc(self):
        return pipeline_service(self.tmp)

    def recognitions(self):
        rows = [json.loads(x) for x in Path(self.log).read_text(encoding="utf-8").splitlines()] if os.path.exists(self.log) else []
        return [r for r in rows if not r.get("dry_run") and "input" in r]

    def plan(self, reqs, **kw):
        return self.svc().plan([self.src], "generic", user_requirements=reqs, params={"language": "en"}, use_cache=False, **kw)



class PlanTests(_PlanBase):
    def test_transcription_and_subtitles(self):
        ent.EntityStore(self.ws).add(JANE)
        ent.EntityStore(self.ws, "conf26").add(ACME)
        ir = self.plan({"subtitle": True}, entity_project="conf26")
        sent = self.recognitions()[-1]
        self.assertEqual(sent["domain_vocabulary"], {"speakers": ["Jane Doe"], "terms": ["Acme Corp"], "max_terms": 2})
        self.assertEqual({(r["from"], r["to"]) for r in sent["corrections"]}, {("Jane Dough", "Jane Doe"), ("Jayne Doe", "Jane Doe"), ("Acme Core", "Acme Corp")})
        summary = ir.doc["plan"]["summary"]
        self.assertIn("entities: 2 entities applied to transcription vocabulary", summary)
        self.assertIn("entities: 3 alias correction(s) applied to the transcript (the subtitles are cut from it)", summary)
        cues = [c["text"] for op in ir.doc["captions"]["operations"] if op["type"] == "captions.generate" for c in op["cues"]]
        self.assertEqual(cues[0], "Jane Doe opened the Acme Corp keynote.", "the subtitles carry the canonical spelling")
        rec = ir.doc["request"]["args"]["entities"]
        self.assertEqual([s["store"] for s in rec["stores"]], ["global", "project:conf26"])
        self.assertEqual(self.svc().validate(ir).errors, [])

    def test_empty_store_leaves_the_plan_unchanged(self):
        base = self.plan({"subtitle": True})
        base_sent = self.recognitions()
        ent.EntityStore(self.ws).save(ent.EntityStore(self.ws).empty())   # a store file with no entity
        Path(self.log).unlink()
        again = self.plan({"subtitle": True}, entity_project="nothing-here")
        import re

        def same(doc, part):   # ids are made fresh by every plan: compare with them replaced
            text = json.dumps(part, sort_keys=True)
            for i, a in enumerate(doc["assets"]):
                text = text.replace(a, f"ASSET{i}")
            return re.sub(r"\b[a-z]+_[0-9a-f]{8,}\b", "ID", text)
        for part in (lambda d: d["plan"]["summary"], lambda d: d["request"]["args"], lambda d: [(x["subject"], x["type"], x["decision"]) for x in d["decisions"]],
                     lambda d: [(x["skill"], x["tool"], x["params"]) for x in d["plan"]["steps"]], lambda d: d["captions"], lambda d: d["provenance"].get("transcription_options")):
            self.assertEqual(same(again.doc, part(again.doc)), same(base.doc, part(base.doc)))
        self.assertNotIn("entities", again.doc["request"]["args"])
        strip = lambda rows: [{k: v for k, v in r.items() if k != "asset_id"} for r in rows]  # noqa: E731
        self.assertEqual(strip(self.recognitions()), strip(base_sent), "the same transcription request (no vocabulary, no corrections)")

    def test_no_entities_and_no_transcript(self):
        ent.EntityStore(self.ws).add(JANE)
        ir = self.plan({"subtitle": True}, use_entities=False)
        self.assertNotIn("domain_vocabulary", self.recognitions()[-1])
        self.assertNotIn("entities", ir.doc["request"]["args"])
        ir = self.plan({"audio.normalize": True})   # nothing recognises speech: nothing to apply, nothing said
        self.assertFalse([s for s in ir.doc["plan"]["summary"] if s.startswith("entities:")])

    def test_a_remote_engine_never_gets_the_names(self):
        ent.EntityStore(self.ws).add(JANE)
        svc = self.svc()
        ad = svc.adapter([]).adapter_for("transcription/transcribe")
        self.assertIsInstance(ad, TranscriptionAdapter)
        ad.contract = json.loads(json.dumps(ad.contract))
        for e in ad.contract["engines"]:
            e["execution_mode"], e["requires_network"] = "remote", True
        ir = svc.plan([self.src], "generic", user_requirements={"subtitle": True}, params={"language": "en"}, use_cache=False)
        self.assertNotIn("domain_vocabulary", self.recognitions()[-1])
        self.assertNotIn("corrections", self.recognitions()[-1])
        self.assertTrue(any("does not run locally" in s for s in ir.doc["plan"]["summary"]), ir.doc["plan"]["summary"])

    def test_lower_third_spelling_and_title(self):
        ent.EntityStore(self.ws).add(JANE)
        ir = self.plan({"motion.lower_third": "jane dough"})
        el = next(e for op in ir.doc["graphics"]["operations"] for e in op["elements"] if e["type"] == "lower_third")
        self.assertEqual((el["parameters"]["name"], el["parameters"]["title"]), ("Jane Doe", "CTO, Acme Corp"))
        self.assertIn("entities: 1 entity applied to the lower third (name, spelling corrected, title from the entity)", ir.doc["plan"]["summary"])
        self.assertEqual(ir.doc["request"]["args"]["requirements"], {"motion.lower_third": "jane dough"}, "the request is recorded as typed")
        ir = self.plan({"motion.lower_third": "Jane Doe", "motion.lower_third.title": "Keynote"})
        el = next(e for op in ir.doc["graphics"]["operations"] for e in op["elements"] if e["type"] == "lower_third")
        self.assertEqual(el["parameters"]["title"], "Keynote", "the user's title wins")
        ir = self.plan({"motion.lower_third": "Somebody Else"})
        self.assertFalse([s for s in ir.doc["plan"]["summary"] if "lower third" in s and s.startswith("entities:")])

    def test_subtitle_emphasis_words(self):
        reqs = {"subtitle": True, "subtitle.burn_in": True, "subtitle.emphasis": "entities", "edit.trim_leading_silence": True}
        with self.assertRaises(ValueError) as cm:
            self.plan(reqs)
        self.assertIn("has no person", str(cm.exception))
        ent.EntityStore(self.ws).add(JANE)
        ent.EntityStore(self.ws).add(TERM)
        ir = self.plan(reqs)
        burn = [op for op in ir.doc["captions"]["operations"] if op["type"] == "captions.burn"]
        self.assertEqual(burn[0]["emphasis"], {"mode": "words", "words": ["Jane", "Doe"]})
        self.assertIn("entities: 1 entity applied to subtitle emphasis (2 word(s), burn-in)", ir.doc["plan"]["summary"])
        with self.assertRaises(ValueError):
            self.plan({"subtitle": True, "subtitle.emphasis": "entities"})   # emphasis is drawn: burn only

    def test_revision_keeps_the_recorded_entities(self):
        ent.EntityStore(self.ws).add(JANE)
        svc = self.svc()
        ir = svc.plan([self.src], "generic", user_requirements={"motion.lower_third": "Jane Dough"}, params={"language": "en"}, use_cache=False)
        path = str(Path(self.ws) / "plans" / "p.project.json")
        from video_agent.project import load_ir, save_ir
        save_ir(ir, path)
        ent.EntityStore(self.ws).edit("jane-doe", {"title": "Changed after the plan"})
        out = svc.revise(load_ir(path), path, user_requirements={"audio.normalize": True})
        nd = load_ir(path).doc if not isinstance(out, dict) or "ir" not in out else out["ir"].doc
        el = next(e for op in nd["graphics"]["operations"] for e in op["elements"] if e["type"] == "lower_third")
        self.assertEqual((el["parameters"]["name"], el["parameters"]["title"]), ("Jane Doe", "CTO, Acme Corp"), "the plan's record, not the store's newer content")


class EndToEndTests(_PlanBase):
    def test_burned_emphasis_and_lower_third_reach_the_skills(self):
        from pipeline_harness import plan_and_render
        ent.EntityStore(self.ws).add(JANE)
        svc = self.svc()
        r = plan_and_render(svc, [self.src], {"subtitle": True, "subtitle.burn_in": True, "subtitle.emphasis": "entities", "motion.lower_third": "jane dough"}, name="e2e")
        self.assertEqual((r["validation"].errors, r["out"]["execution"]["status"]), ([], "COMPLETED"), r["out"])
        prov = json.loads((Path(svc.workspace) / "jobs" / r["out"]["job"]["id"] / "provenance.json").read_text(encoding="utf-8"))
        burn = next(e for e in prov["operations"] if e["tool"] == "subtitle/render")
        self.assertEqual(burn["args"]["emphasis"], {"mode": "words", "words": ["Jane", "Doe"]})
        mg = next(e for e in prov["operations"] if e["tool"] == "motion-graphics/run")
        lt = next(e for e in mg["args"]["elements"] if e["type"] == "lower_third")
        self.assertEqual((lt["parameters"]["name"], lt["parameters"]["title"]), ("Jane Doe", "CTO, Acme Corp"))

    def test_the_dub_gets_the_pronunciation_dictionary(self):
        from pipeline_harness import plan_and_render
        from test_dubbing import dub_service
        ent.EntityStore(self.ws).add(JANE)
        cues = Path(self.tmp) / "src" / "en.srt"
        cues.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello Jane Doe\n", encoding="utf-8")
        log = str(Path(self.tmp) / "dub.log")
        os.environ["FAKE_DUB_CALLS"] = log
        self.addCleanup(os.environ.pop, "FAKE_DUB_CALLS", None)
        r = plan_and_render(dub_service(self.tmp), [self.src], {"dub.cues": str(cues), "dub.lang": "en"}, name="dub")
        self.assertEqual((r["validation"].errors, r["out"]["execution"]["status"]), ([], "COMPLETED"), r["out"])
        dub = [json.loads(x) for x in Path(log).read_text(encoding="utf-8").splitlines() if x.startswith('["dub"')][0]
        flag = next(a for a in dub if a.startswith("--pronunciation="))
        doc = json.loads(Path(flag.split("=", 1)[1]).read_text(encoding="utf-8"))
        self.assertEqual([x["term"] for x in doc["entries"]], ["Jane Doe", "Jane Dough", "Jayne Doe"])
        self.assertTrue(any("pronunciation dictionary" in s for s in r["ir"].doc["plan"]["summary"]))


class DubbingTests(_Tmp):
    def test_pronunciation_dictionary_for_the_dub_language(self):
        snap = {"rule": ent.RULE, "stores": [], "entities": [dict(ent.normalize_entity(JANE), id="jane-doe")]}
        cues = Path(self.tmp) / "en.srt"
        cues.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello Jane Doe\n", encoding="utf-8")
        rec = ent.plan_record(snap, {"dub.cues": str(cues), "dub.lang": "en-GB"}, self.ws)
        path = rec["requirements"]["dub.pronunciation"]["value"]
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual((doc["schema"], len(doc["entries"])), ("dubbing/pronunciation@1", 3))
        self.assertTrue(Path(path).resolve().is_relative_to(Path(self.ws).resolve()) if hasattr(Path, "is_relative_to") else True)
        self.assertIn("EXPERIMENTAL", ent.summary_lines(rec)[0])
        mine = Path(self.tmp) / "mine.json"
        rec = ent.plan_record(snap, {"dub.cues": str(cues), "dub.lang": "en", "dub.pronunciation": str(mine)}, self.ws)
        self.assertEqual(rec["requirements"], {})
        self.assertIn("given explicitly", ent.summary_lines(rec)[0])
        self.assertEqual(ent.plan_record(snap, {"dub.cues": str(cues), "dub.lang": "fr"}, self.ws)["requirements"], {})
        from video_agent.agent.dubbing import parse_dub_requirements
        from video_agent.models import Requirement
        m = {"dub.cues": Requirement(key="dub.cues", value=str(cues), provenance="USER"), "dub.lang": Requirement(key="dub.lang", value="en", provenance="USER"),
             "dub.pronunciation": Requirement(key="dub.pronunciation", value=path, provenance="PREFERENCE", source="entities:jane-doe")}
        self.assertEqual(parse_dub_requirements(m)["params"]["pronunciation"], os.path.realpath(path))
        m["dub.engine"] = Requirement(key="dub.engine", value="kokoro", provenance="PREFERENCE", source="entities:x")
        with self.assertRaises(ValueError):
            parse_dub_requirements(m)   # only the dictionary may come from the entity memory


if __name__ == "__main__":
    unittest.main()
