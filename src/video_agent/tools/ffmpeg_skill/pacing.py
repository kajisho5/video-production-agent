"""Typed reference-pacing operations on a capability-checked local checkout."""
import json
import sys

from .adapter import PathPolicy, run_process_group
from .locate import locate_ffmpeg_skill


class PacingAdapter:
    def __init__(self, checkout, workspace):
        skill = locate_ffmpeg_skill(checkout)
        flags = {"silence": ("--extract-profile", "--references", "--profile", "--protect-ranges"),
                 "verify": ("--profile", "--source", "--protect-ranges")}
        if not skill or not all(skill.offers_flag(tool, flag) for tool, names in flags.items() for flag in names):
            raise ValueError("installed ffmpeg-skill does not offer the complete pacing contract")
        self.skill = skill
        self.check_output = PathPolicy([], workspace).check_output

    def _call(self, tool, *args):
        code, stdout, stderr = run_process_group([sys.executable, str(self.skill.script(tool)), *args, "--json"], 3600)
        if code:
            raise ValueError(f"{tool} failed: {stderr[-1000:]}")
        doc = json.loads(stdout)
        if doc.get("status") != "completed":
            raise ValueError(f"{tool} did not complete")
        return doc


    def extract(self, references, output):
        additional = ["--references", *references[1:]] if references[1:] else []
        return self._call("silence", references[0], "--extract-profile", *additional, "-o", output)["profile"]

    def edit(self, source, profile, protect, output, preview=False):
        ranges = ["--protect-ranges", protect] if protect else []
        return self._call("silence", source, "--profile", profile, *ranges, "-o", output, *(["--dry-run"] if preview else []))

    def verify(self, source, profile, protect, output):
        ranges = ["--protect-ranges", protect] if protect else []
        return self._call("verify", output, "--source", source, "--profile", profile, *ranges)
