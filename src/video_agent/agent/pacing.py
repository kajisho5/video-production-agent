"""Drive the installed pacing contract through extraction, review, edit and independent QC."""
import json
import os
from pathlib import Path
import sys
import tempfile

from ..tools.ffmpeg_skill.adapter import PathPolicy, run_process_group
from ..tools.ffmpeg_skill.locate import locate_ffmpeg_skill


def route(workspace, checkout, source, output, references=(), profile=None, protect=None, apply=False):
    skill = locate_ffmpeg_skill(checkout)
    flags = {"silence": ("--extract-profile", "--references", "--profile", "--protect-ranges"),
             "verify": ("--profile", "--source", "--protect-ranges")}
    if not skill or not all(skill.offers_flag(tool, flag) for tool, names in flags.items() for flag in names):
        raise ValueError("installed ffmpeg-skill does not offer the complete pacing contract")
    source = str(Path(source).resolve(strict=True))
    refs = [str(Path(path).resolve(strict=True)) for path in references]
    presets = {"sns-short", "youtube-talk", "podcast-natural"}
    if profile and profile not in presets:
        profile = str(Path(profile).resolve(strict=True))
    if protect:
        protect = str(Path(protect).resolve(strict=True))
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = (root/output).resolve()
    PathPolicy([], str(root)).check_output(str(target), [source, *refs, *([protect] if protect else []),
                                                       *([profile] if profile and profile not in presets else [])])
    if target.exists() or not target.parent.is_dir():
        raise ValueError("destination must be new and have an existing parent")
    def call(tool, *args):
        code, stdout, stderr = run_process_group([sys.executable, str(skill.script(tool)), *args, "--json"], 3600)
        if code:
            raise ValueError(f"{tool} failed: {stderr[-1000:]}")
        doc = json.loads(stdout)
        if doc.get("status") != "completed":
            raise ValueError(f"{tool} did not complete")
        return doc

    result = {}
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        if refs:
            measured = str(Path(tmp)/"measured.json")
            result["profile"] = call("silence", refs[0], "--extract-profile",
                                     *(["--references", *refs[1:]] if refs[1:] else []), "-o", measured)["profile"]
            meta = result["profile"]["probe_meta"]
            if meta["bgm_suspected"] or meta["confidence"] == "low":
                if not profile:
                    return dict(result, status="NEEDS_REVIEW", guidance="Supply a voice-only reference, a preset, or a reviewed profile.")
            else:
                profile = measured
        if not profile:
            raise ValueError("supply a reference, a profile file or a preset")
        options = ["--profile", profile] + (["--protect-ranges", protect] if protect else [])
        stage = str(Path(tmp)/("output"+target.suffix))
        result["cut_plan"] = call("silence", source, *options, "-o", stage, "--dry-run")["cut_plan"]
        if not apply:
            return dict(result, status="PREVIEW")
        edited = call("silence", source, *options, "-o", stage)
        if edited.get("verified") is not True or edited["cut_plan"] != result["cut_plan"]:
            raise ValueError("render did not verify the previewed cuts")
        gate = call("verify", stage, "--source", source, *options)
        if gate.get("verified") is not True or gate["cut_plan"] != result["cut_plan"]:
            raise ValueError("independent pacing gate failed")
        os.link(stage, target)
        return dict(result, status="COMPLETED", output=str(target), verification=gate["checks"])
