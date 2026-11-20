"""scripts/repin.py (ADR-117): release tags, the gate, the decision, moving a pin's readers, and the integration lock. No network, no git."""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("repin", ROOT / "scripts" / "repin.py")
repin = importlib.util.module_from_spec(_spec)
sys.modules["repin"] = repin
_spec.loader.exec_module(repin)


class TagTests(unittest.TestCase):
    def test_release_tags_only(self):
        self.assertEqual(repin.tag_version("v1.4.2", ("v",)), "1.4.2")
        self.assertEqual(repin.tag_version("blender-skill-v0.31.1", ("blender-skill-v", "v")), "0.31.1")
        for t in ("1.4.2", "v1.4", "v1.4.2-rc1", "nightly", "v1.4.2^{}"):
            self.assertIsNone(repin.tag_version(t, ("v",)), t)

    def test_latest_is_by_number(self):
        self.assertEqual(repin.latest_tag(["v0.9.0", "v0.10.0", "v0.2.11", "latest"], ("v",)), ("v0.10.0", "0.10.0"))
        # blender-skill: its own prefix since 0.2.0, an old plain v0.1.0
        tags = ["v0.1.0", "blender-skill-v0.9.0", "blender-skill-v0.31.1", "blender-skill-v0.4.0"]
        self.assertEqual(repin.latest_tag(tags, ("blender-skill-v", "v")), ("blender-skill-v0.31.1", "0.31.1"))
        self.assertEqual(repin.latest_tag(["v1.0.0", "x-v1.0.0"], ("x-v", "v")), ("x-v1.0.0", "1.0.0"))   # a tie: the first prefix
        self.assertIsNone(repin.latest_tag(["main", "v1"], ("v",)))

    def test_every_sibling_has_one_pin(self):
        names = set()
        for sib in repin.SIBLINGS:
            path, version = repin.current_pin(ROOT, sib)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["version"], version, sib.name)
            names.add(sib.repo)
        self.assertEqual(len(names), len(repin.SIBLINGS))
        self.assertNotIn("ffmpeg-skill", names)


GATE_SRC = '''"""doc"""
SUPPORTED_SKILL_VERSIONS = ("0.1.", "0.2.")   # 0.2.0: an addition
OTHER = 1
'''


class GateTests(unittest.TestCase):
    def test_read_and_check(self):
        self.assertEqual(repin.read_gate(GATE_SRC), ("0.1.", "0.2."))
        self.assertTrue(repin.in_gate("0.2.9", ("0.1.", "0.2.")))
        self.assertFalse(repin.in_gate("0.3.0", ("0.1.", "0.2.")))
        self.assertFalse(repin.in_gate("0.20.0", ("0.2.",)))
        self.assertTrue(repin.in_gate("9.9.9", None))      # a handoff pin has no gate

    def test_widen_keeps_the_comment_and_adds_a_note(self):
        out = repin.widen_gate(GATE_SRC, "0.3.", "0.3.0 (automatic re-pin): additive")
        self.assertEqual(repin.read_gate(out), ("0.1.", "0.2.", "0.3."))
        self.assertIn('("0.1.", "0.2.", "0.3.")   # 0.2.0: an addition\n', out)
        self.assertIn("# 0.3.0 (automatic re-pin): additive\nSUPPORTED_SKILL_VERSIONS", out)
        one = repin.widen_gate('SUPPORTED_SKILL_VERSIONS = ("0.1.",)\n', "0.2.", "n")
        self.assertEqual(repin.read_gate(one), ("0.1.", "0.2."))
        own_line = repin.widen_gate('SUPPORTED_SKILL_VERSIONS = (\n    "0.1.",\n    "0.2.",\n)\n', "0.3.", "n")
        self.assertEqual(repin.read_gate(own_line), ("0.1.", "0.2.", "0.3."))     # a `)` on its own line, a trailing comma
        crlf = repin.widen_gate(GATE_SRC.replace("\n", "\r\n"), "0.3.", "n")
        self.assertIn("# n\r\nSUPPORTED_SKILL_VERSIONS", crlf)
        with self.assertRaises(ValueError):
            repin.widen_gate("X = 1\n", "0.1.", "n")

    def test_widen_every_real_gate(self):
        for sib in repin.SIBLINGS:
            if not sib.gate_file:
                continue
            src = (ROOT / sib.gate_file).read_text(encoding="utf-8")
            gate = repin.read_gate(src)
            out = repin.widen_gate(src, "9.9.", "note")
            ast.parse(out)
            self.assertEqual(repin.read_gate(out), gate + ("9.9.",), sib.name)
            self.assertTrue(repin.in_gate(json.loads(repin.current_pin(ROOT, sib)[0].read_text(encoding="utf-8"))["version"], gate), sib.name)


class DependencyRangeTests(unittest.TestCase):
    """Addendum 2: a dependency's version range that only widened is additive, not si contract-diff's opaque-string `breaking`."""

    def test_widened_or_not(self):
        for old, new in ((">=2.0.0 <4.0.0", ">=2.0.0 <5.0.0"), (">=1.0 <3.0", ">=1.0 <4.0"), (">=2.0.0 <4.0.0", ">=1.0.0 <4.0.0"),
                         (">=2.0.0 <4.0.0", ">=2.0.0"), (">2.0.0 <4.0.0", ">=2.0.0 <4.0.0"), (">=2.0.0 <4.0.0", ">=2.0.0 <=4.0.0")):
            self.assertTrue(repin.range_widened(old, new), (old, new))
        for old, new in ((">=2.0.0 <5.0.0", ">=2.0.0 <4.0.0"), (">=2.0.0 <4.0.0", ">=3.0.0 <5.0.0"), (">=2.0.0 <4.0.0", ">=2.0.0 <4.0.0"),
                         (">=2.0.0", ">=2.0.0 <9.0.0"), (">=2.0.0 <4.0.0", "^2.0.0"), (">=2.0.0 <4.0.0", ""), (None, ">=2.0.0 <5.0.0"),
                         (">=2.0.0 <=4.0.0", ">=2.0.0 <4.0.0"), (">=2.0.0 <4.0.0", "==3.0.0")):
            self.assertFalse(repin.range_widened(old, new), (old, new))

    @staticmethod
    def _diff(*changes, classification="breaking"):
        return {"classification": classification, "changes": [dict(c) for c in changes]}

    def test_the_ffmpeg_skill_4_widening_is_additive(self):
        old = {"dependencies": [{"skill_id": "ffmpeg-skill", "version_range": ">=2.0.0 <4.0.0"}]}
        new = {"dependencies": [{"skill_id": "ffmpeg-skill", "version_range": ">=2.0.0 <5.0.0"}]}
        rng = {"pointer": "/dependencies/0/version_range", "effect": "breaking", "rule": "value-changed", "detail": "value changed"}
        info = {"pointer": "/version", "effect": "info", "rule": "release-version"}
        d = repin.reclassify_dependency_ranges(self._diff(rng, info), old, new)
        self.assertEqual(d["classification"], "additive")
        self.assertEqual((d["changes"][0]["effect"], d["changes"][0]["rule"]), ("additive", "range-widened"))
        self.assertIn(">=2.0.0 <5.0.0", d["changes"][0]["detail"])
        self.assertEqual(repin.decide("0.2.5", "0.2.7", d["classification"], ("0.1.", "0.2.")).kind, "auto")

    def test_other_breaking_changes_keep_it_breaking(self):
        old = {"dependencies": [{"skill_id": "ffmpeg-skill", "version_range": ">=2.0.0 <4.0.0"}]}
        new = {"dependencies": [{"skill_id": "ffmpeg-skill", "version_range": ">=2.0.0 <5.0.0"}]}
        rng = {"pointer": "/dependencies/0/version_range", "effect": "breaking", "rule": "value-changed"}
        fps = {"pointer": "/tools/4/input_schema/properties/fps/type", "effect": "breaking", "rule": "value-changed"}
        d = repin.reclassify_dependency_ranges(self._diff(rng, fps), old, new)
        self.assertEqual(d["classification"], "breaking")
        self.assertEqual([c["effect"] for c in d["changes"]], ["additive", "breaking"])

    def test_narrowed_moved_or_another_dependency_stays_breaking(self):
        rng = {"pointer": "/dependencies/0/version_range", "effect": "breaking", "rule": "value-changed"}
        for o, n in (({"skill_id": "ffmpeg-skill", "version_range": ">=2.0.0 <5.0.0"}, {"skill_id": "ffmpeg-skill", "version_range": ">=2.0.0 <4.0.0"}),
                     ({"skill_id": "ffmpeg-skill", "version_range": ">=2.0.0 <4.0.0"}, {"skill_id": "ffmpeg-skill", "version_range": ">=3.0.0 <5.0.0"}),
                     ({"skill_id": "ffmpeg-skill", "version_range": ">=2.0.0 <4.0.0"}, {"skill_id": "other-skill", "version_range": ">=2.0.0 <5.0.0"})):
            d = repin.reclassify_dependency_ranges(self._diff(rng), {"dependencies": [o]}, {"dependencies": [n]})
            self.assertEqual((d["classification"], d["changes"][0]["effect"]), ("breaking", "breaking"), (o, n))
        # a pointer past the arrays (si reported an added entry) is left as si said
        d = repin.reclassify_dependency_ranges(self._diff(dict(rng, pointer="/dependencies/3/version_range")), {"dependencies": []}, {"dependencies": []})
        self.assertEqual(d["classification"], "breaking")

    def test_through_contract_diff(self):
        """contract_diff() applies it to what si reports, reading both ranges from the files (si's `old` / `new` are display strings)."""
        tmp = Path(tempfile.mkdtemp(prefix="repin-range-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "old.json").write_text(json.dumps({"dependencies": [{"skill_id": "ffmpeg-skill", "version_range": ">=2.0.0 <4.0.0"}]}), encoding="utf-8")
        (tmp / "new.json").write_text(json.dumps({"dependencies": [{"skill_id": "ffmpeg-skill", "version_range": ">=2.0.0 <5.0.0"}]}), encoding="utf-8")
        si_out = {"classification": "breaking", "changes": [{"pointer": "/dependencies/0/version_range", "effect": "breaking", "rule": "value-changed"}]}
        fake_si = tmp / "si.py"
        fake_si.write_text("import json, sys\nprint(json.dumps(%r))\n" % (si_out,), encoding="utf-8")
        real = repin.subprocess.run

        def run(cmd, **kw):
            return real([sys.executable, str(fake_si)] + list(cmd[1:]), **kw)
        repin.subprocess.run = run
        self.addCleanup(setattr, repin.subprocess, "run", real)
        self.assertEqual(repin.contract_diff("si", tmp / "old.json", tmp / "new.json")["classification"], "additive")


class DecisionTests(unittest.TestCase):
    def test_inside_the_gate(self):
        for c in ("none", "additive"):
            self.assertEqual(repin.decide("1.4.1", "1.4.2", c, ("1.4.",)).kind, "auto")
        self.assertEqual(repin.decide("0.31.0", "0.32.0", "additive", None).kind, "auto")   # no gate (handoff)

    def test_widen_a_minor_of_0x_and_1x(self):
        d = repin.decide("0.2.3", "0.3.0", "additive", ("0.1.", "0.2."))
        self.assertEqual((d.kind, d.new_prefix), ("widen", "0.3."))
        d = repin.decide("1.4.5", "1.5.0", "none", ("1.4.",))
        self.assertEqual((d.kind, d.new_prefix), ("widen", "1.5."))

    def test_a_first_1_0_0_after_0x_widens(self):
        """Addendum 3 (Velisa ADR-031): 0.x -> 1.0.0 with a none / additive diff is a promotion; the gate takes `1.0.`."""
        for old, c in (("0.1.22", "none"), ("0.10.1", "additive"), ("0.9.0", "none")):
            d = repin.decide(old, "1.0.0", c, ("0.1.", "0.9.", "0.10."))
            self.assertEqual((d.kind, d.new_prefix), ("widen", "1.0."), (old, c))
        self.assertEqual(repin.decide("0.1.22", "1.0.0", "none", None).kind, "auto")   # no gate (handoff)
        self.assertEqual(repin.decide("0.1.22", "1.0.0", "none", ("0.1.", "1.0.")).kind, "auto")

    def test_review(self):
        cases = (("0.2.3", "0.2.4", "breaking", ("0.2.",)), ("1.4.5", "2.0.0", "none", ("1.4.",)), ("0.9.0", "1.0.0", "breaking", ("0.9.",)),
                 ("0.9.0", "1.0.1", "none", ("0.9.",)), ("0.9.0", "1.1.0", "additive", ("0.9.",)), ("0.9.0", "2.0.0", "none", ("0.9.",)),
                 ("2.3.0", "2.4.0", "additive", ("2.3.",)), ("0.2.4", "0.2.4", "none", ("0.2.",)), ("0.2.4", "0.2.3", "none", ("0.2.",)))
        for old, new, c, gate in cases:
            d = repin.decide(old, new, c, gate)
            self.assertEqual(d.kind, "review", (old, new, c))
            self.assertIsNone(d.new_prefix)
            self.assertTrue(d.reasons)


def sib(name):
    return repin.BY_NAME[name]


class RewriteTests(unittest.TestCase):
    def rewrite(self, text, rel, name, old, new, sha="abcdef0123456789"):
        s = sib(name)
        return repin.rewrite_text(text, rel, s, [o for o in repin.SIBLINGS if o is not s], old, new, sha)

    def test_two_skills_on_the_same_version(self):
        # detection and dubbing both pinned at 0.2.3; detection moves, dubbing does not
        text = ('A = Path(...) / "detection" / "contract_0.2.3.json"\n'
                'B = Path(...) / "dubbing" / "contract_0.2.3.json"\n'
                'self.assertEqual((c["skill_id"], c["version"]), ("detection", "0.2.3"))\n'
                'self.assertEqual((c["skill_id"], c["version"]), ("dubbing", "0.2.3"))\n'
                'skill = FfmpegSkill(root, "0.2.3", ["probe"])\n'
                'x = "0.2.3"\n')
        out, n, left = self.rewrite(text, "tests/test_unit.py", "detection", "0.2.3", "0.2.4")
        self.assertEqual(out.splitlines(), [
            'A = Path(...) / "detection" / "contract_0.2.4.json"',
            'B = Path(...) / "dubbing" / "contract_0.2.3.json"',
            'self.assertEqual((c["skill_id"], c["version"]), ("detection", "0.2.4"))',
            'self.assertEqual((c["skill_id"], c["version"]), ("dubbing", "0.2.3"))',
            'skill = FfmpegSkill(root, "0.2.3", ["probe"])',
            'x = "0.2.3"'])      # the line above names the engine, not detection
        self.assertEqual((n, left), (2, ['tests/test_unit.py:6: x = "0.2.3"']))   # left, and listed: the file names detection

    def test_owned_file_lines_without_a_name(self):
        text = ('PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_1.4.4.json")\n'
                'self.assertEqual(ad.version, "1.4.4")\n'
                'self.assertEqual(obs.source, "qc/check@1.4.4")\n'
                '# 1.4.4 (qc-skill #60): version only\n'
                'self.assertEqual(x, ("audio-production", "1.4.4"))\n')
        out, n, _ = self.rewrite(text, "tests/test_adapter_qc.py", "qc", "1.4.4", "1.4.5")
        self.assertEqual(out.splitlines(), [
            'PINNED_CONTRACT_PATH = Path(__file__).with_name("contract_1.4.5.json")',
            'self.assertEqual(ad.version, "1.4.5")',
            'self.assertEqual(obs.source, "qc/check@1.4.5")',
            '# 1.4.4 (qc-skill #60): version only',                   # history in prose stays
            'self.assertEqual(x, ("audio-production", "1.4.4"))'])    # another Skill's line in qc's own test file
        self.assertEqual(n, 3)

    def test_context_above_alias_and_evals(self):
        text = ('from video_agent.tools.video_editing import adapter as ve\n'
                'pk = next(p for p in svc.packages() if p["skill_id"] == "video-editing")\n'
                'self.assertTrue(pk["available"]); self.assertEqual(pk["version"], "0.8.4")\n'
                'self.assertEqual(ve.pinned_contract()["version"], "0.8.4")\n'
                '\n\n\n\n'
                'self.assertEqual(other, "0.8.4")\n')
        out, n, left = self.rewrite(text, "tests/test_unit.py", "video-editing", "0.8.4", "0.8.5")
        lines = out.splitlines()
        self.assertEqual(lines[2], 'self.assertTrue(pk["available"]); self.assertEqual(pk["version"], "0.8.5")')
        self.assertEqual(lines[3], 'self.assertEqual(ve.pinned_contract()["version"], "0.8.5")')
        self.assertEqual(lines[-1], 'self.assertEqual(other, "0.8.4")')     # nothing about video-editing within 3 lines
        ev = '{\n  "silence": {\n    "skill": "media-analysis",\n    "skill_version": "0.1.21",\n    "source": "media-analysis/silence@0.1.21"\n  }\n}\n'
        out, n, _ = self.rewrite(ev, "evals/cases/24.json", "media-analysis", "0.1.21", "0.1.22")
        self.assertEqual(out, ev.replace("0.1.21", "0.1.22")); self.assertEqual(n, 2)
        out, _, _ = self.rewrite("rec.ingest_version == \"0.1.0\"\n", "tests/test_unit.py", "ingest", "0.1.0", "0.1.1")
        self.assertEqual(out, "rec.ingest_version == \"0.1.1\"\n")     # `ingest_version` names ingest

    def test_handoff_pin(self):
        text = ('    HandoffPin(\n'
                '        skill="render_3d", package="blender-skill", version="0.31.1", pin_file="blender-skill_0.31.1.json",\n'
                '        source="kajisho5/blender-skill blender-skill-v0.31.1 tag ff882b1 (contract/contract.json)",\n'
                'PIN = load_pin("blender-skill_0.31.1.json")\n'
                '        skill="html_scene_render", package="hyperframes-skill", version="0.31.1",\n')
        out, n, left = self.rewrite(text, "src/video_agent/skills/handoff_contracts.py", "blender", "0.31.1", "0.32.0", "0123456789abcdef")
        self.assertEqual(out.splitlines(), [
            '    HandoffPin(',
            '        skill="render_3d", package="blender-skill", version="0.32.0", pin_file="blender-skill_0.32.0.json",',
            '        source="kajisho5/blender-skill blender-skill-v0.32.0 tag 0123456 (contract/contract.json)",',
            'PIN = load_pin("blender-skill_0.32.0.json")',        # the file name names blender-skill on its own
            '        skill="html_scene_render", package="hyperframes-skill", version="0.31.1",'])
        self.assertEqual((n, left), (3, []))

    def test_leftovers_are_listed(self):
        text = ('from video_agent.tools.qc import adapter as qc\n'
                'for v in ("1.4.0", "1.4.4"):\n'          # the import above names qc (its alias)
                '\n\n\n\n'
                'probe = ["1.4.4"]\n'                     # nothing names qc near it: left, and listed
                'y = ("audio-production", "1.4.4")\n')    # another Skill's: left, not listed
        out, n, left = self.rewrite(text, "tests/test_version_gates.py", "qc", "1.4.4", "1.4.5")
        self.assertIn('for v in ("1.4.0", "1.4.5")', out)
        self.assertEqual(left, ['tests/test_version_gates.py:7: probe = ["1.4.4"]'])
        _, _, left = self.rewrite('x = "1.4.4"\n', "tests/test_unit.py", "qc", "1.4.4", "1.4.5")
        self.assertEqual(left, [])                       # the file never names qc

    def test_refusal_probe_moves_after_a_widen(self):
        text = ('    def test_qc(self):\n'
                '        self.assertEqual(qc.pinned_contract()["version"], "1.4.4")\n'
                '        for v in ("1.4.0", "1.4.4"):\n'
                '            self.assertEqual(qc.check_contract(with_version(qc.pinned_contract(), v)), [], v)\n'
                '        for v in ("0.3.0", "1.5.0", "2.0.0"):\n'
                '            self.assertTrue(qc.check_contract(with_version(qc.pinned_contract(), v)), v)\n'
                '        for v in ("1.5.0", "1.6.0"):\n'
                '            self.assertTrue(qc.check_contract(with_version(qc.pinned_contract(), v)), v)\n'
                '\n\n\n\n'
                '        for v in ("1.5.0",):\n')
        text = 'from video_agent.tools.qc import adapter as qc\n' + text
        out, n = repin.move_refusal_probe(text, sib("qc"), "1.5.")
        self.assertIn('for v in ("0.3.0", "1.6.0", "2.0.0"):', out)
        self.assertIn('for v in ("1.6.0", "1.6.0"):', out)      # still a tuple of two, never a bare string
        self.assertIn('for v in ("1.5.0",):', out)     # nothing names qc on it or the 3 lines above
        self.assertEqual(n, 2)
        pairs = ('        for skill, new in (("qc-skill", "1.5.0"), ("thumbnail-skill", "1.5.0"),\n'
                 '                           ("qc-skill" , "2.0.0")):\n')
        out, n = repin.move_refusal_probe(pairs, sib("qc"), "1.5.")
        self.assertEqual(out, pairs.replace('("qc-skill", "1.5.0")', '("qc-skill", "1.6.0")')); self.assertEqual(n, 1)

    def test_the_probe_is_moved_only_in_a_refusal_loop(self):
        """Addendum 4 (agent #182): after motion-graphics 0.4.5 -> 1.0.0 the reader rewrite left "1.0.0" in the acceptance loop; the
        gate's widen to `1.0.` must move the probe "1.0.0" in the refusal loop only, never the accepted release."""
        text = ('from video_agent.tools.motion_graphics import adapter as mg\n'
                '    def test_motion_graphics_release_versions(self):\n'
                '        self.assertEqual(mg.pinned_contract()["version"], pinned_version("motion-graphics"))\n'
                '        for v in ("0.1.0", "0.4.0", "1.0.0", pinned_version("motion-graphics")):\n'
                '            live = with_version(mg.pinned_contract(), v)\n'
                '            self.assertEqual(mg.check_contract(live), [], v)\n'
                '            self.assertEqual(mg.contract_breaking_drift(live), [], v)\n'
                '        for v in ("0.5.0", "0.30.0", "1.0.0"):\n'
                '            self.assertTrue(mg.check_contract(with_version(mg.pinned_contract(), v)), v)\n'
                '        for v in ("0.4.0", "1.0.0"):   # motion-graphics\n'
                '            self.assertEqual(mg.check_contract(with_version(mg.pinned_contract(), v)), [], v)\n')
        out, n = repin.move_refusal_probe(text, sib("motion-graphics"), "1.0.")
        self.assertIn('for v in ("0.1.0", "0.4.0", "1.0.0", pinned_version("motion-graphics")):', out)   # accepted: kept
        self.assertIn('for v in ("0.5.0", "0.30.0", "1.1.0"):', out)                                     # refused: moved
        self.assertIn('for v in ("0.4.0", "1.0.0"):   # motion-graphics', out)                          # accepted, no pinned_version
        self.assertEqual(n, 1)

    def test_the_real_gate_tests_after_a_1_0_0_promotion(self):
        """On the real tests/test_version_gates.py, motion-graphics' accepted release list holding "1.0.0" keeps it."""
        path = Path(__file__).resolve().parent / "test_version_gates.py"
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        i = next(k for k, l in enumerate(lines) if "def test_motion_graphics_release_versions" in l)
        j = next(k for k in range(i, len(lines)) if lines[k].lstrip().startswith("for v in (") and "pinned_version(" in lines[k])
        lines[j] = lines[j].replace("pinned_version(", '"1.0.0", pinned_version(', 1)
        out, _ = repin.move_refusal_probe("".join(lines), sib("motion-graphics"), "1.0.")
        self.assertIn('"1.0.0", pinned_version("motion-graphics")', out.splitlines()[j])

    def test_a_line_naming_two_skills_on_one_version_is_left(self):
        text = 'self.assertEqual((det["version"], dub["version"]), ("0.2.3", "0.2.3"))   # detection, dubbing\n'
        out, n, left = repin.rewrite_text(text, "tests/test_unit.py", sib("detection"), [o for o in repin.SIBLINGS if o.name != "detection"],
                                          "0.2.3", "0.2.4", "f" * 40, [sib("dubbing")])
        self.assertEqual((out, n), (text, 0))
        self.assertEqual(len(left), 1); self.assertIn("also names a Skill pinned at 0.2.3", left[0])


class ContractTests(unittest.TestCase):
    def test_validate(self):
        pinned = {"skill_id": "qc", "version": "1.4.4"}
        self.assertIsNone(repin.validate_contract(b'{"skill_id": "qc", "version": "1.4.5"}', "1.4.5", pinned))
        self.assertIn("not JSON", repin.validate_contract(b"{", "1.4.5", pinned))
        self.assertIn("not a contract object", repin.validate_contract(b"[]", "1.4.5", pinned))
        self.assertIn("declares version '1.4.4'", repin.validate_contract(b'{"skill_id": "qc", "version": "1.4.4"}', "1.4.5", pinned))
        self.assertIn("is 'subtitle'", repin.validate_contract(b'{"skill_id": "subtitle", "version": "1.4.5"}', "1.4.5", pinned))
        # the three ways a contract names its Skill
        self.assertIsNone(repin.validate_contract(b'{"skill": {"id": "dubbing"}, "version": "0.2.4"}', "0.2.4", {"skill": {"id": "dubbing"}}))
        self.assertIsNone(repin.validate_contract(b'{"skill": "imagemagick-skill", "version": "0.10.2"}', "0.10.2", {"skill": "imagemagick-skill"}))

    def test_every_pin_validates_against_itself(self):
        for s in repin.SIBLINGS:
            path, version = repin.current_pin(ROOT, s)
            self.assertIsNone(repin.validate_contract(path.read_bytes(), version, json.loads(path.read_text(encoding="utf-8"))), s.name)

    def test_only_documented_releases_lack_a_committed_contract(self):
        self.assertEqual(repin.NO_COMMITTED_CONTRACT, {("artifacts", "v0.4.0")})


class ApplyTests(unittest.TestCase):
    """apply_update on a copy of the parts of this repository a qc re-pin touches."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="repin-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        s = sib("qc")
        pin, self.old = repin.current_pin(ROOT, s)
        (self.tmp / s.pin_dir).mkdir(parents=True)
        shutil.copy(pin, self.tmp / s.pin_dir / pin.name)
        shutil.copy(ROOT / s.gate_file, self.tmp / s.gate_file)
        (self.tmp / "tests").mkdir()
        (self.tmp / "tests" / "test_adapter_qc.py").write_text(f'self.assertEqual(ad.version, "{self.old}")\n', encoding="utf-8")
        for rel in repin.GATE_TESTS:
            shutil.copy(ROOT / rel, self.tmp / rel)
        self.contract = json.loads(pin.read_text(encoding="utf-8"))

    def new_release(self, version):
        c = dict(self.contract, version=version)
        return (json.dumps(c, indent=2) + "\n").encode("utf-8")

    def test_patch_inside_the_gate(self):
        a, b, c = repin.parse_version(self.old)
        new = f"{a}.{b}.{c + 1}"
        data = self.new_release(new)
        u = repin.apply_update(self.tmp, sib("qc"), f"v{new}", "f" * 40, data, {"classification": "none", "changes": []}, "2026-11-19")
        self.assertEqual((u.old, u.new, u.decision.kind), (self.old, new, "auto"))
        self.assertEqual(sorted(p.name for p in (self.tmp / sib("qc").pin_dir).glob("contract_*.json")), [f"contract_{new}.json"])
        self.assertEqual((self.tmp / sib("qc").pin_dir / f"contract_{new}.json").read_bytes(), data)
        self.assertEqual((self.tmp / "tests" / "test_adapter_qc.py").read_text(encoding="utf-8"), f'self.assertEqual(ad.version, "{new}")\n')
        self.assertIn(f'with_name("contract_{new}.json")', (self.tmp / sib("qc").gate_file).read_text(encoding="utf-8"))
        self.assertEqual(repin.read_gate((self.tmp / sib("qc").gate_file).read_text(encoding="utf-8")),
                         repin.read_gate((ROOT / sib("qc").gate_file).read_text(encoding="utf-8")))

    def test_minor_outside_the_gate_widens(self):
        a, b, _ = repin.parse_version(self.old)
        new = f"{a}.{b + 1}.0"
        u = repin.apply_update(self.tmp, sib("qc"), f"v{new}", "f" * 40, self.new_release(new), {"classification": "additive", "changes": [
            {"pointer": "/x", "effect": "additive", "rule": "key-added", "detail": "x added"}]}, "2026-11-19")
        self.assertEqual(u.decision.kind, "widen")
        gate = repin.read_gate((self.tmp / sib("qc").gate_file).read_text(encoding="utf-8"))
        self.assertEqual(gate[-1], f"{a}.{b + 1}.")
        body = repin.pr_body([u], True)
        self.assertIn("widen", body); self.assertIn("x added", body); self.assertNotIn("must review", body)
        self.assertNotIn(f'"{a}.{b + 1}.0", ', (self.tmp / repin.GATE_TESTS[0]).read_text(encoding="utf-8").split("def test_qc_release_versions")[1].split("def ")[0])
        self.assertNotIn(f'("qc-skill", "{a}.{b + 1}.0")', (self.tmp / repin.GATE_TESTS[1]).read_text(encoding="utf-8"))

    def test_breaking_asks_for_review(self):
        a, b, c = repin.parse_version(self.old)
        new = f"{a}.{b}.{c + 1}"
        gate_before = (self.tmp / sib("qc").gate_file).read_text(encoding="utf-8")
        u = repin.apply_update(self.tmp, sib("qc"), f"v{new}", "f" * 40, self.new_release(new), {"classification": "breaking", "changes": []}, "d")
        self.assertEqual(u.decision.kind, "review")
        self.assertIn(f"contract_{new}.json", (self.tmp / sib("qc").gate_file).read_text(encoding="utf-8"))
        self.assertEqual(repin.read_gate((self.tmp / sib("qc").gate_file).read_text(encoding="utf-8")), repin.read_gate(gate_before))
        self.assertIn("must review", repin.pr_body([u], True))


class VerifyTests(unittest.TestCase):
    """Addendum 1: what the rewrite left in tests/ and evals/ heads the body; failing unit tests make the re-pin a `review` one."""

    def update(self, leftovers):
        return repin.Update("hyperframes", "0.10.3", "0.10.4", "v0.10.4", "f" * 40, "none", repin.Decision("auto", ["inside the gate"]),
                            leftovers=leftovers)

    def test_test_lines_are_listed_apart_and_first(self):
        u = self.update(["tests/test_handoff_contracts.py:60: self.assertEqual(v, \"0.10.3\")", "src/video_agent/x.py:3: # 0.10.3 history",
                         "evals/cases/a.json:4: \"0.10.3\""])
        body = repin.pr_body([u], True)
        head, table = body.split("| Skill |", 1)
        self.assertIn("Check before merging", head)
        self.assertIn("tests/test_handoff_contracts.py:60", head); self.assertIn("evals/cases/a.json:4", head)
        self.assertNotIn("src/video_agent/x.py", head)
        self.assertIn("src/video_agent/x.py:3", table); self.assertNotIn("tests/test_handoff_contracts.py:60", table)
        self.assertNotIn("Check before merging", repin.pr_body([self.update(["src/video_agent/x.py:3: # 0.10.3"])], True))

    def test_unit_commands_are_the_unit_jobs(self):
        cmds = repin.unit_commands(ROOT)
        text = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        self.assertEqual(len(cmds) - 1, text.count("- run: python -m unittest -v tests/"))
        self.assertIn("tests/test_handoff_contracts.py", [a for c in cmds for a in c])
        self.assertEqual(cmds[-1][1:], ["evals/run.py"])

    def test_failed_ids(self):
        out = "FAIL: test_a (tests.test_x.C.test_a)\nERROR: test_b (tests.test_y.D)\nok\n"
        self.assertEqual(repin.failed_ids(["py", "-m", "unittest", "t"], out, 1), ["tests.test_x.C.test_a", "tests.test_y.D.test_b"])
        self.assertEqual(repin.failed_ids(["py", "-m", "unittest", "t"], out, 0), [])
        self.assertEqual(repin.failed_ids(["py", "evals/run.py"], "PASS a\nFAIL b  why\n1/2 passed\n", 1), ["evals b"])
        self.assertEqual(repin.failed_ids(["/x/python3", "-m", "unittest", "t"], "ImportError", 1), ["python3 -m unittest t: exit 1, no test id"])

    def test_a_failure_makes_the_re_pin_a_review_one(self):
        tmp = Path(tempfile.mkdtemp(prefix="repin-verify-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / ".github" / "workflows").mkdir(parents=True)
        (tmp / ".github" / "workflows" / "tests.yml").write_text("      - run: python -m unittest -v tests/test_a.py tests/test_b.py\n", encoding="utf-8")
        u = self.update(["tests/test_a.py:9: self.assertEqual(v, \"0.10.3\")"])
        report = {"updates": [{"sibling": u.sibling, "leftovers": u.leftovers}], "review": False, "review_reasons": []}
        (tmp / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (tmp / "body.md").write_text(repin.pr_body([u], True), encoding="utf-8")
        calls = []

        def runner(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 1, "FAIL: test_v (tests.test_a.T.test_v)\n", "")
        real = repin.run_unit
        repin.run_unit = lambda root: real(root, runner)
        self.addCleanup(setattr, repin, "run_unit", real)
        # quiet: the fake failure's log line would otherwise reach this test run's own output, which `repin.py verify` parses for ids
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(repin.main(["--root", str(tmp), "verify", "--report", str(tmp / "report.json"), "--pr-body", str(tmp / "body.md")]), 0)
        self.assertEqual([c[1:] for c in calls], [["-m", "unittest", "tests/test_a.py", "tests/test_b.py"]])
        got = json.loads((tmp / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(got["review"])
        self.assertEqual(got["unit_failures"], ["tests.test_a.T.test_v"])
        self.assertIn("tests.test_a.T.test_v", got["review_reasons"][0])
        body = (tmp / "body.md").read_text(encoding="utf-8")
        self.assertTrue(body.startswith("**A person must review this pull request: 1 unit test"))
        self.assertIn("tests/test_a.py:9", body.split("| Skill |")[0])

        repin.run_unit = lambda root: []
        (tmp / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (tmp / "body.md").write_text("table\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            repin.main(["--root", str(tmp), "verify", "--report", str(tmp / "report.json"), "--pr-body", str(tmp / "body.md")])
        self.assertFalse(json.loads((tmp / "report.json").read_text(encoding="utf-8"))["review"])
        self.assertTrue((tmp / "body.md").read_text(encoding="utf-8").startswith("The unit tests and the evals pass"))


class LockTests(unittest.TestCase):
    def test_lock_agrees_with_the_pins(self):
        self.assertEqual(repin.check_lock(ROOT), [])
        lock = repin.read_lock(ROOT)
        self.assertEqual(sorted(lock["siblings"]), sorted(repin.BY_NAME))

    def test_check_finds_a_stale_lock(self):
        tmp = Path(tempfile.mkdtemp(prefix="repin-lock-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        for s in repin.SIBLINGS:
            pin, _ = repin.current_pin(ROOT, s)
            (tmp / s.pin_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy(pin, tmp / s.pin_dir / pin.name)
        lock = repin.read_lock(ROOT)
        e = lock["siblings"]["qc"]
        e["version"], e["tag"], e["sha"] = "0.0.1", "v0.0.1", "abc"
        del lock["siblings"]["blender"]
        (tmp / repin.LOCK_PATH).parent.mkdir()
        (tmp / repin.LOCK_PATH).write_text(json.dumps(lock), encoding="utf-8")
        problems = repin.check_lock(tmp)
        self.assertTrue(any(p.startswith("qc: lock version") for p in problems), problems)
        self.assertTrue(any(p.startswith("qc: lock tag") for p in problems), problems)
        self.assertTrue(any(p.startswith("qc: lock sha") for p in problems), problems)
        self.assertIn("blender: no lock entry", problems)


if __name__ == "__main__":
    unittest.main(verbosity=2)
