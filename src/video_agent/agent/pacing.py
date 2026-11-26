"""Drive the installed pacing contract through extraction, review, edit and independent QC."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

PRESETS = {"sns-short", "youtube-talk", "podcast-natural"}


def confirm_token(source, target, plan):
    """The preview's cuts bound to their source and destination: apply runs only with the token of the same plan."""
    doc = {"source": source, "output": target, "cut_plan": plan}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def route(workspace, engine, source, output, references=(), profile=None, protect=None, apply=False, confirm=None):
    source = str(Path(source).resolve(strict=True))
    refs = [str(Path(path).resolve(strict=True)) for path in references]
    if profile and profile not in PRESETS:
        profile = str(Path(profile).resolve(strict=True))
    if protect:
        protect = str(Path(protect).resolve(strict=True))
    if apply and not confirm:
        raise ValueError("apply needs the confirm_token of the preview the user approved (--confirm)")
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = (root/output).resolve()
    engine.check_output(str(target), [source, *refs, *([protect] if protect else []),
                                      *([profile] if profile and profile not in PRESETS else [])])
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
        result["profile_used"] = "measured" if refs and profile == measured else profile
        stage = str(Path(tmp)/("output"+target.suffix))
        result["cut_plan"] = engine.edit(source, profile, protect, stage, preview=True)["cut_plan"]
        token = confirm_token(source, str(target), result["cut_plan"])
        if not apply:
            return dict(result, status="PREVIEW", confirm_token=token)
        if confirm != token:
            raise ValueError("the measured cuts differ from the approved preview (confirm_token mismatch): preview again")
        edited = engine.edit(source, profile, protect, stage)
        if edited.get("verified") is not True or edited["cut_plan"] != result["cut_plan"]:
            raise ValueError("render did not verify the previewed cuts")
        gate = engine.verify(source, profile, protect, stage)
        if gate.get("verified") is not True or gate["cut_plan"] != result["cut_plan"]:
            raise ValueError("independent pacing gate failed")
        os.link(stage, target)
        return dict(result, status="COMPLETED", output=str(target), verification=gate["checks"])
