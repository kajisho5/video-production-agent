"""ADR-118: capabilities read from the pinned contracts -- the shared loader, capability → operation → adapter, the contract-driven gate,
and the clear error when no pinned Skill provides a capability. No Skill checkout; the pins only."""
import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_agent.skills import capabilities as caps  # noqa: E402
from video_agent.skills.capabilities import CAPABILITY_SKILLS, CapabilityNotProvided  # noqa: E402
from video_agent.skills.registry import default_registry  # noqa: E402
from video_agent.tools.detection import adapter as detection  # noqa: E402
from video_agent.tools.diarization import adapter as diarization  # noqa: E402
from video_agent.tools.router import ToolRouter  # noqa: E402
from video_agent.tools.voice_enhance import adapter as voice  # noqa: E402

# The tool ids these SkillSpecs named by hand before ADR-118. A migration must not change what an existing plan runs.
HAND_WIRED_BEFORE = {
    "voice_isolate": ["voice-enhance/isolate"],
    "voice_dereverb": ["voice-enhance/dereverb"],
    "voice_bleed_reduce": ["voice-enhance/bleed-reduce"],
    "face_detection": ["detection/faces"],
    "face_attributes": ["detection/faces"],
    "text_detection": ["detection/text"],
    "shot_detection": ["detection/shots"],
    "speaker_diarization": ["diarization/diarize"],
}


class LoaderTests(unittest.TestCase):
    def test_every_pin_is_read_with_its_identity(self):
        pins = caps.pinned_skills()
        self.assertEqual(len(pins), len(caps.pin_files()))
        for p in pins:
            self.assertTrue(p.skill_id, p.pin)
            self.assertTrue(p.version, p.pin)
            self.assertTrue(p.provides, f"{p.pin} has no provides[]")
            for c in p.provides:
                self.assertIn("/", c.tool_id)
                self.assertTrue(c.operation)

    def test_one_pin_per_tool_directory(self):
        dirs = [p.parent.name for p in caps.TOOLS_DIR.glob("*/contract_*.json")]
        self.assertEqual(len(dirs), len(set(dirs)), "a tools/<dir> holds more than one contract_*.json")

    def test_operation_comes_from_the_entry_when_the_skill_names_one(self):
        cut = caps.resolve("audio.cut")
        self.assertEqual((cut.tool_id, cut.operation), ("audio-production/run", "CUT"))
        self.assertEqual(caps.resolve("voice.isolate").operation, "isolate")   # no operation field: the tool's own name

    def test_package_is_the_tool_id_prefix_even_when_the_skill_id_differs(self):
        c = caps.resolve("subtitle.render")
        self.assertEqual((c.skill_id, c.package), ("subtitle-skill", "subtitle"))
        self.assertEqual(caps.resolve("subtitle.render", "subtitle-skill"), c)


class MigrationTests(unittest.TestCase):
    def test_every_capability_backed_skill_resolves_to_exactly_one_pin(self):
        for skill, cap in CAPABILITY_SKILLS.items():
            self.assertEqual(len(caps.providers(cap)), 1, f"{skill}: {cap}")

    def test_migrated_specs_run_the_same_tools_as_before(self):
        reg = default_registry()
        self.assertEqual(set(HAND_WIRED_BEFORE), set(CAPABILITY_SKILLS))
        for skill, tools in HAND_WIRED_BEFORE.items():
            self.assertEqual(reg.get(skill).tools, tools, skill)
            self.assertEqual(reg.capability(skill), CAPABILITY_SKILLS[skill])
        self.assertIsNone(reg.capability("video_cut"))   # still hand-wired
        self.assertIsNone(reg.capability("no_such_skill"))

    def test_voice_enhance_wire_constants_agree_with_the_pin(self):
        self.assertEqual(voice.TOOL_OF_TYPE, {"ISOLATE": voice.TOOL_ISOLATE, "DEREVERB": voice.TOOL_DEREVERB, "BLEED_REDUCE": voice.TOOL_BLEED_REDUCE})

    def test_a_capability_no_pin_provides_stops_the_registry_with_a_clear_error(self):
        with mock.patch.dict(CAPABILITY_SKILLS, {"voice_isolate": "voice.denoise"}):
            with self.assertRaises(CapabilityNotProvided) as e:
                default_registry()
        msg = str(e.exception)
        self.assertIn("no pinned skill provides capability 'voice.denoise'", msg)
        # the pins it read: the pinned release, inside the adapter's gate (not a literal 0.1.: voice-enhance-skill 1.0 widens it to 1.0.)
        self.assertTrue(any(f"voice-enhance {v}" in msg for v in voice.SUPPORTED_SKILL_VERSIONS), msg)
        self.assertIn("voice.bleed_reduce, voice.dereverb, voice.isolate", msg)  # what is pinned under the same prefix


class ResolveTests(unittest.TestCase):
    def test_unknown_capability(self):
        with self.assertRaises(CapabilityNotProvided) as e:
            caps.resolve("teleport.now")
        self.assertIn("no pinned capability starts with 'teleport.'", str(e.exception))

    def test_capability_from_a_package_that_does_not_provide_it(self):
        with self.assertRaises(CapabilityNotProvided) as e:
            caps.resolve("voice.isolate", "detection")
        self.assertIn("from package detection", str(e.exception))

    def test_a_capability_two_pins_provide_needs_a_package(self):
        self.assertEqual({c.package for c in caps.providers("measure.audio.loudness")}, {"qc", "media-analysis"})
        with self.assertRaises(CapabilityNotProvided) as e:
            caps.resolve("measure.audio.loudness")
        self.assertIn("more than one pinned skill", str(e.exception))
        self.assertEqual(caps.resolve("measure.audio.loudness", "media-analysis").tool_id, "media-analysis/loudness")


class _Adapter:
    def __init__(self, name, tools):
        self.name, self.tools = name, tools

    def supports(self, tool):
        return tool in self.tools


class RouteTests(unittest.TestCase):
    def test_capability_operation_adapter(self):
        router = ToolRouter([_Adapter("detection", {"detection/text"}), _Adapter("voice-enhance", set(voice.TOOLS))])
        r = caps.route("voice.dereverb", router.adapter_for)
        self.assertEqual((r.tool_id, r.operation, r.adapter), ("voice-enhance/dereverb", "dereverb", "voice-enhance"))
        self.assertEqual(caps.route("vision.detect_text", router.adapter_for).adapter, "detection")

    def test_no_installed_adapter_is_a_handoff_not_an_error(self):
        r = caps.route("speech.diarize", ToolRouter([]).adapter_for)
        self.assertIsNone(r.adapter)
        self.assertEqual(r.tool_id, "diarization/diarize")

    def test_route_of_an_unprovided_capability_raises(self):
        with self.assertRaises(CapabilityNotProvided):
            caps.route("speech.teleport", ToolRouter([]).adapter_for)


def _drop(contract, cap):
    c = copy.deepcopy(contract)
    c["provides"] = [p for p in c["provides"] if p.get("id") != cap]
    return c


def _retool(contract, cap, tool):
    c = copy.deepcopy(contract)
    for p in c["provides"]:
        if p.get("id") == cap:
            p["tool_id"] = tool
    return c


class GateTests(unittest.TestCase):
    def test_consumed_capabilities_per_package(self):
        self.assertEqual(caps.consumed_capabilities("voice-enhance"), ["voice.bleed_reduce", "voice.dereverb", "voice.isolate"])
        self.assertEqual(caps.consumed_capabilities("detection"), ["vision.detect_faces", "vision.detect_shots", "vision.detect_text"])
        self.assertEqual(caps.consumed_capabilities("diarization"), ["speech.diarize"])
        self.assertEqual(caps.consumed_capabilities("video-editing"), [])   # not migrated yet

    def test_the_pins_pass_their_own_gate(self):
        for mod in (voice, detection, diarization):
            self.assertEqual(caps.contract_gate(mod.pinned_contract(), mod.SKILL_ID), [], mod.SKILL_ID)
            self.assertEqual(mod.check_contract(mod.pinned_contract()), [], mod.SKILL_ID)

    def test_a_dropped_capability_is_refused(self):
        for mod, cap in ((voice, "voice.isolate"), (detection, "vision.detect_shots"), (diarization, "speech.diarize")):
            self.assertIn(f"provides lacks {cap}", mod.check_contract(_drop(mod.pinned_contract(), cap)), mod.SKILL_ID)

    def test_detection_gate_covers_tracks(self):
        # face_tracks runs detection/tracks after detection/faces: a contract without vision.track_boxes is refused (ADR-118 addendum 1)
        self.assertEqual(sorted(detection.CAPABILITY_IDS.values()), sorted(caps.consumed_capabilities("detection") + ["vision.track_boxes"]))
        errs = detection.check_contract(_drop(detection.pinned_contract(), "vision.track_boxes"))
        self.assertIn("provides lacks vision.track_boxes", errs)
        errs = detection.check_contract(_retool(detection.pinned_contract(), "vision.track_boxes", "detection/link"))
        self.assertIn("provides vision.track_boxes names tool detection/link, the pin names detection/tracks", errs)

    def test_a_capability_moved_to_another_tool_is_refused(self):
        errs = detection.check_contract(_retool(detection.pinned_contract(), "vision.detect_text", "detection/ocr"))
        self.assertIn("provides vision.detect_text names tool detection/ocr, the pin names detection/text", errs)

    def test_a_lifecycle_change_is_not_refused(self):
        c = copy.deepcopy(voice.pinned_contract())
        c["provides"][0]["lifecycle"] = "STABLE"
        self.assertEqual(caps.contract_gate(c, voice.SKILL_ID), [])

    def test_a_contract_without_provides(self):
        self.assertEqual(caps.contract_gate({}, "diarization"), ["provides lacks speech.diarize"])
        self.assertEqual(caps.contract_gate(None, "diarization", []), [])


if __name__ == "__main__":
    unittest.main()
