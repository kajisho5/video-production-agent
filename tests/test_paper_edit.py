"""Paper edit from a script (ADR-139, roadmap step 28): the script's lines are the cut order of a take assembly. Each take goes to the script line
its words are most like, the best take of each line is picked by ADR-053's scores (coverage against the script line), lines with no take are
reported missing, and a pick said differently from its line is CONFIRM with both texts. Fake engine + fake Skills; no media is decoded."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_harness import clear_modes, fake_media, pipeline_service  # noqa: E402
from video_agent.agent import paper_edit as pe  # noqa: E402
from video_agent.agent.editing import EditRequirementError  # noqa: E402
from video_agent.agent.requirements import requirement_map  # noqa: E402
from video_agent.agent.takes import clean_text, parse_takes_requirements, similarity, strip_fillers  # noqa: E402
from video_agent.models import Requirement  # noqa: E402
from video_agent.project import load_ir, save_ir  # noqa: E402

SCRIPT = ["Welcome to the annual meeting of the society",
          "This year we met in a new hall by the river",
          "The sound system was installed by our own volunteers",
          "Every session was recorded for the archive",
          "Please thank the staff who made it possible",
          "We will see you again next spring"]
# three takes of line 1 and line 3, one of lines 2, 4 and 5 (line 4's only take changes a word), none of line 6
TAKES = [("l1a.mp4", "Welcome to the um annual meeting of the society"), ("l1b.mp4", "Welcome to the annual meeting of the society"),
         ("l1c.mp4", "Welcome to the annual annual meeting of the"),
         ("l2a.mp4", "This year we met in a new hall by the river"),
         ("l3a.mp4", "The sound system was installed by our own"), ("l3b.mp4", "The sound system was installed by our own volunteers"),
         ("l3c.mp4", "The the sound system was installed by our own volunteers"),
         ("l4a.mp4", "Every session was filmed for the archive"),
         ("l5a.mp4", "Please thank the staff who made it possible")]


def reqmap(**kv):
    return requirement_map([Requirement(key=k.replace("__", "."), value=v, provenance="USER", source="cli") for k, v in kv.items()])


class RuleTests(unittest.TestCase):
    def test_placing_missing_and_unplaced(self):
        takes = [(n, clean_text(t)) for n, t in TAKES] + [("other.mp4", clean_text("completely unrelated words about lunch"))]
        lines, missing, unplaced = pe.place_takes(takes, SCRIPT, 0.6, clean_text, similarity)
        self.assertEqual([ln["line"] for ln in lines], [1, 2, 3, 4, 5])
        self.assertEqual(lines[0]["takes"], ["l1a.mp4", "l1b.mp4", "l1c.mp4"])
        self.assertEqual(missing, [6])
        self.assertEqual([u["asset_id"] for u in unplaced], ["other.mp4"])

    def test_differences_side_by_side(self):
        self.assertEqual(pe.differences(SCRIPT[3], "Every session was filmed for the archive", strip_fillers), [{"script": "recorded", "spoken": "filmed"}])
        self.assertEqual(pe.differences(SCRIPT[0], "Welcome to the um annual meeting of the society", strip_fillers), [], "fillers are not a changed word")
        self.assertEqual(pe.differences(SCRIPT[2], "The sound system was installed by our own", strip_fillers), [{"script": "volunteers", "spoken": ""}])
        self.assertEqual(pe.differences("本日は晴天です", "本日は雨天です", strip_fillers), [{"script": "晴", "spoken": "雨"}])

    def test_the_script_file(self):
        tmp = tempfile.mkdtemp()
        p = Path(tmp) / "script.txt"
        p.write_text("# a comment\n\nLine one\n  Line two  \n", encoding="utf-8")
        self.assertEqual(pe.read_script(p), ["Line one", "Line two"])
        t = parse_takes_requirements(reqmap(edit__takes=True, edit__takes__script=str(p)))
        self.assertEqual(t["script"], ["Line one", "Line two"])
        for bad in ("", "# only\n"):
            p.write_text(bad, encoding="utf-8")
            with self.assertRaises(EditRequirementError):
                parse_takes_requirements(reqmap(edit__takes=True, edit__takes__script=str(p)))
        with self.assertRaises(EditRequirementError):
            parse_takes_requirements(reqmap(edit__takes=True, edit__takes__script=str(Path(tmp) / "nope.txt")))
        with self.assertRaises(EditRequirementError):
            parse_takes_requirements(reqmap(edit__takes__script=str(p)))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        clear_modes()
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.takes = [fake_media(self.tmp, n, duration=5.0, segments=[[0.5, 4.0, t]], silences=[[0.0, 0.4], [4.2, None]]) for n, t in TAKES]
        self.script = str(Path(self.tmp) / "script.txt")
        Path(self.script).write_text("\n".join(SCRIPT) + "\n", encoding="utf-8")

    def plan(self, svc=None):
        svc = svc or pipeline_service(self.tmp)
        ir = svc.plan(self.takes, "youtube", user_requirements={"edit.takes": True, "edit.takes.script": self.script}, params={"language": "en"})
        path = str(Path(svc.workspace) / "plans" / "paper.project.json")
        save_ir(ir, path)
        return svc, load_ir(path), path

    @staticmethod
    def names(doc):
        return {aid: Path(a["path"]).name for aid, a in doc["assets"].items()}

    def test_three_takes_of_five_lines_in_script_order_one_changed_word_one_missing_line(self):
        svc, ir, path = self.plan()
        doc = ir.doc
        self.assertEqual(svc.validate(ir).errors, [])
        nm = self.names(doc)
        picks = sorted([d for d in doc["decisions"] if d["subject"] == "takes.pick"], key=lambda d: d["params"]["line"])
        self.assertEqual([(d["params"]["line"], nm[d["params"]["asset_id"]]) for d in picks],
                         [(1, "l1b.mp4"), (2, "l2a.mp4"), (3, "l3b.mp4"), (4, "l4a.mp4"), (5, "l5a.mp4")])
        concat = next(op for op in doc["video"]["operations"] if op["type"] == "video.concat")
        self.assertEqual([nm[a] for a in concat["inputs"]], ["l1b.mp4", "l2a.mp4", "l3b.mp4", "l4a.mp4", "l5a.mp4"], "the five lines in script order")
        changed = [d for d in picks if d["params"]["script"]["differences"]]
        self.assertEqual([d["params"]["line"] for d in changed], [4])
        sp = changed[0]["params"]["script"]
        self.assertEqual((sp["text"], sp["spoken"], sp["differences"]), (SCRIPT[3], "Every session was filmed for the archive", [{"script": "recorded", "spoken": "filmed"}]))
        self.assertEqual(changed[0]["approval"], "CONFIRM")
        self.assertIn("script “recorded” / spoken “filmed”", changed[0]["reason"])
        (miss,) = [d for d in doc["decisions"] if d["subject"] == pe.MISSING_SUBJECT]
        self.assertEqual(miss["params"]["missing"], [{"line": 6, "script": SCRIPT[5]}])
        summary = "\n".join(doc["plan"]["summary"])
        self.assertIn("Script line 4 is said differently", summary)
        self.assertIn("line 6", summary)
        cov = next(f for t in picks[0]["params"]["takes"] for f in t["factors"] if f["factor"] == "coverage" and t["asset_id"] == picks[0]["params"]["asset_id"])
        self.assertIn("the script line", cov["rule"])

    def test_rejecting_the_changed_line_cuts_it_from_the_assembly(self):
        svc, ir, path = self.plan()
        nm = self.names(ir.doc)
        changed = next(d for d in ir.doc["decisions"] if d["subject"] == "takes.pick" and d["params"]["script"]["differences"])
        svc.reject(load_ir(path), path, [changed["id"]], reason="retake line 4")
        rev = svc.revise(load_ir(path), path)
        self.assertTrue(rev["created"])
        doc = load_ir(path).doc
        concat = next(op for op in doc["video"]["operations"] if op["type"] == "video.concat")
        self.assertEqual([nm[a] for a in concat["inputs"]], ["l1b.mp4", "l2a.mp4", "l3b.mp4", "l5a.mp4"])
        self.assertEqual(svc.validate(load_ir(path)).errors, [])

    def test_render_waits_for_the_changed_line_then_assembles(self):
        svc, ir, path = self.plan()
        out = svc.render(load_ir(path), path)
        self.assertEqual(out["status"], "WAITING_FOR_APPROVAL")
        out = svc.render(load_ir(path), path, approve=["all"])
        self.assertIn(out["status"], ("COMPLETED", "REVIEW"), out.get("execution", {}).get("failed_op"))

    def test_without_a_script_the_takes_group_as_before(self):
        svc = pipeline_service(self.tmp)
        ir = svc.plan(self.takes, "youtube", user_requirements={"edit.takes": True}, params={"language": "en"})
        self.assertFalse([d for d in ir.doc["decisions"] if d["subject"].startswith("script.")])
        self.assertFalse([d for d in ir.doc["decisions"] if d["subject"] == "takes.pick" and "script" in d["params"]])


if __name__ == "__main__":
    unittest.main()
