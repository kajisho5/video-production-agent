"""The agent's recovery-class tables against Velisa's error taxonomy (velisa/error-taxonomy@1, Velisa ADR-027; ADR-119 here).

tests/fixtures/velisa/error-taxonomy.json is copied unchanged from the Velisa repository at the commit recorded in
tests/fixtures/velisa/provenance.json. Every code (or kind) an adapter table maps is looked up in it, and the agent's recovery class
must be one the spec's alignment table gives for that class ("How the agent's recovery classes line up"), and its action must keep
the class's retry policy. The few labels kept on purpose are listed in KEPT, each with its reason; a new disagreement fails here.
"""
import hashlib
import json
import os
import unittest

from video_agent.execution.recovery import RECOVERY_TABLE, _skill_class
from video_agent.tools import skill_process
from video_agent.tools.artifacts import adapter as artifacts
from video_agent.tools.audio_production import adapter as audio_production
from video_agent.tools.base import ToolResult
from video_agent.tools.detection import adapter as detection
from video_agent.tools.diarization import adapter as diarization
from video_agent.tools.dubbing import adapter as dubbing
from video_agent.tools.hyperframes import adapter as hyperframes
from video_agent.tools.imagemagick import adapter as imagemagick
from video_agent.tools.ingest import adapter as ingest
from video_agent.tools.nle_bridge import adapter as nle_bridge
from video_agent.tools.publish import adapter as publish
from video_agent.tools.remotion import adapter as remotion
from video_agent.tools.slide_sync import adapter as slide_sync
from video_agent.tools.sound_design import adapter as sound_design
from video_agent.tools.video_editing import adapter as video_editing
from video_agent.tools.voice_enhance import adapter as voice_enhance

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "velisa")

# table name -> (the Skill id the taxonomy's `overrides` would use, {code or kind: the agent's recovery class})
TABLES = {
    "skill_process": ("*", skill_process.RECOVERY_CLASS),
    "video_editing": ("video-editing", video_editing.ERROR_CODES),
    "audio_production": ("audio-production", audio_production.ERROR_CODES),
    "dubbing": ("dubbing", dubbing.ERROR_CLASSES),
    "voice_enhance": ("voice-enhance", voice_enhance.ERROR_CLASSES),
    "publish": ("publish", publish.ERROR_CLASSES),
    "sound_design": ("sound-design", dict(sound_design.ERROR_CLASSES, **sound_design.OPTIONAL_ERROR_CLASSES)),
    "diarization": ("diarization", diarization.ERROR_CLASSES),
    "detection": ("detection", detection.ERROR_CLASSES),
    "nle_bridge": ("nle-bridge", nle_bridge.ERROR_CLASSES),
    "slide_sync": ("slide-sync", slide_sync.ERROR_CLASSES),
    "imagemagick": ("imagemagick-skill", imagemagick.ERROR_CLASSES),
    "hyperframes": ("hyperframes-skill", hyperframes.ERROR_CLASSES),
    "remotion": ("remotion-ffmpeg-skill", remotion.ERROR_CLASSES),
    "artifacts": ("artifacts-skill", dict(artifacts.CATEGORY_CLASSES, **artifacts.CODE_CLASSES)),
    "ingest": ("ingest", {code: cls for code, (_kind, cls) in ingest.ERROR_KINDS.items()}),
}
TABLES.update({name.replace("-", "_"): (name, mapping) for name, mapping in skill_process.RECOVERY_CLASS_OVERRIDES.items()})

# the spec's alignment table: taxonomy class -> the agent recovery classes that carry it
ALIGNED = {
    "INVALID_REQUEST": {"INVALID_ARGS"},
    "INPUT_NOT_FOUND": {"INPUT_MISSING"},
    "INVALID_INPUT": {"UNSUPPORTED_MEDIA", "INVALID_ARGS"},
    "UNSUPPORTED": {"UNSUPPORTED_MEDIA", "INVALID_ARGS"},
    "DEPENDENCY_MISSING": {"TOOL_MISSING"},
    "ACTION_REQUIRED": {"CONSENT_REQUIRED", "ACTION_REQUIRED"},
    "RESOURCE_LIMIT": {"INSUFFICIENT_VRAM", "DISK_FULL"},
    "RATE_LIMITED": {"SKILL_ERROR"},
    "OUTPUT_FAILED": {"SKILL_ERROR"},
    "VERIFICATION_FAILED": {"SKILL_ERROR"},
    "INTERNAL": {"SKILL_ERROR"},
    "TRANSIENT": {"UNKNOWN"},
    "TIMEOUT": {"TIMEOUT"},
    "CANCELLED": {"INTERRUPTED"},
    "ENGINE_FAILED": {"UNKNOWN", "SKILL_ERROR"},
}
# a retry policy -> the recovery actions that keep it
POLICY_ACTIONS = {"never": {"BLOCK"}, "after_change": {"BLOCK"}, "later": {"BLOCK"}, "once": {"RETRY_SAME", "RETRY_ALT"},
                  "once_longer": {"RETRY_LONGER"}}

# labels kept on purpose: (table, code) -> (the agent class kept, why). Every one blocks, as its class does.
KEPT = {
    ("skill_process", "PATH_NOT_ALLOWED"): ("INPUT_MISSING", "INPUT_MISSING's hint names the case: 'outside the allowed roots'"),
    ("video_editing", "PATH_NOT_ALLOWED"): ("INPUT_MISSING", "INPUT_MISSING's hint names the case: 'outside the allowed roots'"),
    ("audio_production", "PATH_NOT_ALLOWED"): ("INPUT_MISSING", "INPUT_MISSING's hint names the case: 'outside the allowed roots'"),
    ("ingest", "MANIFEST_INVALID"): ("SKILL_ERROR", "a damaged manifest at a destination: not the agent's arguments nor a media source"),
    ("ingest", "MHL_INVALID"): ("SKILL_ERROR", "a damaged ASC MHL history at a destination: not the agent's arguments nor a media source"),
    ("motion_graphics", "DEPENDENCY_ERROR"): ("INVALID_ARGS", "motion-graphics-skill raises it for a duplicate element id or a missing asset / font "
                                                              "reference in the agent's document, not a missing tool; Velisa's taxonomy has no override for it yet"),
    ("nle_bridge", "CONTENT_NOT_EXPORTED"): ("SKILL_ERROR", "the target format cannot carry the content: neither an agent bug nor a source fault"),
}
# names in an adapter table that no Skill contract declares (so the taxonomy has none): the agent's own words for a result it refuses
NOT_SKILL_CODES = {("imagemagick", "SKILL_FAILED"), ("imagemagick", "INVALID_OUTPUT"), ("imagemagick", "INPUT_CHANGED"),
                   ("artifacts", "ARTIFACT_TYPE_UNSUPPORTED")}


def _load():
    with open(os.path.join(FIXTURES, "error-taxonomy.json"), "rb") as fh:
        raw = fh.read()
    return raw, json.loads(raw.decode("utf-8"))


RAW, TAXONOMY = _load()


def taxonomy_class(skill: str, code: str):
    return (TAXONOMY["overrides"].get(skill) or {}).get(code) or TAXONOMY["codes"].get(code) or TAXONOMY["kinds"].get(code)


def rows():
    for table, (skill, mapping) in TABLES.items():
        for code, agent_cls in sorted(mapping.items()):
            yield table, skill, code, agent_cls


class FixtureTests(unittest.TestCase):
    def test_the_copy_is_the_recorded_velisa_file(self):
        with open(os.path.join(FIXTURES, "provenance.json"), encoding="utf-8") as fh:
            rec = json.load(fh)["files"]["error-taxonomy.json"]
        self.assertEqual(rec["repository"], "kajisho5/Velisa")
        self.assertRegex(rec["commit"], r"^[0-9a-f]{40}$")
        # a Windows checkout may turn the line ends into CRLF; the recorded hash is of the file as Velisa has it (LF)
        self.assertEqual(hashlib.sha256(RAW.replace(b"\r\n", b"\n")).hexdigest(), rec["sha256"], "the taxonomy was edited after it was copied")
        self.assertEqual(TAXONOMY["spec"], rec["spec"])

    def test_the_alignment_covers_every_class(self):
        self.assertEqual(set(ALIGNED), set(TAXONOMY["classes"]))
        self.assertEqual({c["retry"] for c in TAXONOMY["classes"].values()} - set(POLICY_ACTIONS), set())
        for allowed in ALIGNED.values():
            self.assertEqual(allowed - set(RECOVERY_TABLE), set(), "every aligned class is a recovery class the agent has")


class TablesAgreeWithTheTaxonomyTests(unittest.TestCase):
    def test_every_code_has_a_class(self):
        unknown = [f"{t}:{c}" for t, s, c, _ in rows() if taxonomy_class(s, c) is None and (t, c) not in NOT_SKILL_CODES]
        self.assertEqual(unknown, [], "add the code to Velisa's error-taxonomy.json, or list it in NOT_SKILL_CODES with its reason")
        stale = [f"{t}:{c}" for t, c in NOT_SKILL_CODES if c not in TABLES[t][1] or taxonomy_class(TABLES[t][0], c) is not None]
        self.assertEqual(stale, [])

    def test_every_recovery_class_is_the_taxonomy_class(self):
        wrong = []
        for table, skill, code, agent_cls in rows():
            tc = taxonomy_class(skill, code)
            if tc is None:
                continue
            kept = KEPT.get((table, code))
            ok = agent_cls == kept[0] if kept else agent_cls in ALIGNED[tc]
            if not ok:
                wrong.append(f"{table}:{code} is {agent_cls}, taxonomy {tc} ({' / '.join(sorted(ALIGNED[tc]))})")
        self.assertEqual(wrong, [], "\n".join(wrong))
        stale = [f"{t}:{c}" for t, c in KEPT if c not in TABLES[t][1]]
        self.assertEqual(stale, [])

    def test_every_action_keeps_the_retry_policy(self):
        wrong = []
        for table, skill, code, agent_cls in rows():
            tc = taxonomy_class(skill, code)
            if tc is None:
                continue
            action = RECOVERY_TABLE[agent_cls]["action"]
            policy = TAXONOMY["classes"][tc]["retry"]
            # ENGINE_FAILED: retried once only when the Skill says retryable, so a code its Skill never retries may block (SKILL_ERROR)
            allowed = POLICY_ACTIONS[policy] | ({"BLOCK"} if tc == "ENGINE_FAILED" else set())
            if action not in allowed:
                wrong.append(f"{table}:{code}: {agent_cls} ({action}) but {tc} retries {policy}")
        self.assertEqual(wrong, [], "\n".join(wrong))

    def test_one_code_one_recovery_class(self):
        seen = {}
        for table, skill, code, agent_cls in rows():
            if (table, code) in KEPT or (table, code) in NOT_SKILL_CODES or skill in TAXONOMY["overrides"]:
                continue
            seen.setdefault(code, {}).setdefault(agent_cls, []).append(table)
        split = {c: v for c, v in seen.items() if len(v) > 1}
        self.assertEqual(split, {}, "the same code is recovered differently by different adapters")

    def test_the_issue_168_places(self):
        self.assertEqual([ingest.ERROR_KINDS[c][1] for c in ("FETCH_FAILED", "INCOMPLETE", "SOURCE_CHANGED", "VERIFY_MISMATCH", "FILE_MISSING")],
                         ["UNKNOWN"] * 5)
        self.assertEqual(ingest.ERROR_KINDS["MANIFEST_NOT_FOUND"][1], "INPUT_MISSING")
        self.assertEqual(publish.ERROR_CLASSES["UPLOAD_INTERRUPTED"], "UNKNOWN")
        self.assertEqual(sound_design.ERROR_CLASSES["TOOL_ERROR"], "UNKNOWN")
        for t in (skill_process.RECOVERY_CLASS, video_editing.ERROR_CODES, audio_production.ERROR_CODES):
            self.assertEqual((t["INVALID_INPUT"], t["DEPENDENCY_ERROR"]), ("UNSUPPORTED_MEDIA", "TOOL_MISSING"))
        for t in (skill_process.RECOVERY_CLASS, video_editing.ERROR_CODES, audio_production.ERROR_CODES, dubbing.ERROR_CLASSES,
                  voice_enhance.ERROR_CLASSES, publish.ERROR_CLASSES, sound_design.ERROR_CLASSES):
            self.assertEqual(t["CANCELLED"], "INTERRUPTED")
        self.assertEqual(hyperframes.ERROR_CLASSES["interrupted"], "INTERRUPTED")
        self.assertEqual({t["NOT_IMPLEMENTED"] for t in (sound_design.ERROR_CLASSES, voice_enhance.ERROR_CLASSES, dubbing.ERROR_CLASSES)}, {"INVALID_ARGS"})
        self.assertEqual({sound_design.ERROR_CLASSES[c] for c in ("CONFIRMATION_REQUIRED", "LICENCE_REQUIRED")}
                         | {publish.ERROR_CLASSES[c] for c in ("CONFIRM_MISMATCH", "CREDENTIALS_MISSING", "AUTH_FAILED")}, {"ACTION_REQUIRED"})


def _failed(code, recovery_class, retryable, details=None):
    data = {"status": "failed", "error": {"code": code, "message": "", "retryable": retryable, "details": details or {}, "recovery_class": recovery_class}}
    return ToolResult("op", "x/y", False, 1, None, data, [], "", 0.0, 1, False)


class RecoveryOfTheClassesTests(unittest.TestCase):
    def test_cancelled_is_interrupted_unless_it_was_a_timeout(self):
        self.assertEqual(_skill_class(_failed("CANCELLED", "INTERRUPTED", True, {"reason": "signal"})), "INTERRUPTED")
        self.assertEqual(_skill_class(_failed("CANCELLED", "INTERRUPTED", True)), "INTERRUPTED")
        self.assertEqual(_skill_class(_failed("CANCELLED", "INTERRUPTED", True, {"reason": "timeout"})), "TIMEOUT", "a run stopped by its time limit")
        self.assertEqual(_skill_class(_failed("CANCELLED", "INTERRUPTED", False, {"reason": "timeout"})), "SKILL_ERROR", "the Skill's no wins")

    def test_transient_is_retried_once_only_when_the_skill_says_so(self):
        self.assertEqual(_skill_class(_failed("FETCH_FAILED", "UNKNOWN", True)), "UNKNOWN")
        self.assertEqual(_skill_class(_failed("FETCH_FAILED", "UNKNOWN", False)), "SKILL_ERROR")

    def test_action_required_blocks(self):
        self.assertEqual(RECOVERY_TABLE["ACTION_REQUIRED"]["action"], "BLOCK")
        self.assertEqual(_skill_class(_failed("CREDENTIALS_MISSING", "ACTION_REQUIRED", False)), "ACTION_REQUIRED")


if __name__ == "__main__":
    unittest.main()
