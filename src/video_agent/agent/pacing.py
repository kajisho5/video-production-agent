"""Drive the installed pacing contract through extraction, review, edit and independent QC."""
import os
from pathlib import Path
import tempfile




def route(workspace, engine, source, output, references=(), profile=None, protect=None, apply=False):
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
    engine.check_output(str(target), [source, *refs, *([protect] if protect else []),
                                                       *([profile] if profile and profile not in presets else [])])
    if target.exists() or not target.parent.is_dir():
        raise ValueError("destination must be new and have an existing parent")
    result = {}
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        if refs:
            measured = str(Path(tmp)/"measured.json")
            result["profile"] = engine.extract(refs, measured)
            meta = result["profile"]["probe_meta"]
            if meta["bgm_suspected"] or meta["confidence"] == "low":
                if not profile:
                    return dict(result, status="NEEDS_REVIEW", guidance="Supply a voice-only reference, a preset, or a reviewed profile.")
            else:
                profile = measured
        if not profile:
            raise ValueError("supply a reference, a profile file or a preset")
        stage = str(Path(tmp)/("output"+target.suffix))
        result["cut_plan"] = engine.edit(source, profile, protect, stage, preview=True)["cut_plan"]
        if not apply:
            return dict(result, status="PREVIEW")
        edited = engine.edit(source, profile, protect, stage)
        if edited.get("verified") is not True or edited["cut_plan"] != result["cut_plan"]:
            raise ValueError("render did not verify the previewed cuts")
        gate = engine.verify(source, profile, protect, stage)
        if gate.get("verified") is not True or gate["cut_plan"] != result["cut_plan"]:
            raise ValueError("independent pacing gate failed")
        os.link(stage, target)
        return dict(result, status="COMPLETED", output=str(target), verification=gate["checks"])
