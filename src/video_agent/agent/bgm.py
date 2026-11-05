"""A music bed under the mix (ADR-092; sound-design-skill roadmap step 5, agent side: BGM -> MIX): sound-design-skill makes the bed, audio-production
MIX places it, as one more mixed input.

    Requirement (`--set bgm.library=<dir>` or `bgm.prompt="..."`, with audio.production=true and audio.mix=true) → TRANSFORM `audio.bgm` (approval
    floor CONFIRM) → IR `audio.bgm {asset: bgm, output: bgm_bed, mode, ..., duration}` → step `bgm_library` (sound-design/library-pick) or
    `bgm_generate` (sound-design/generate-bgm) → the WAV is the mix's last input (`bgm`), at the mix's length (its first input's)

- **Requirements** (refused before analysis when ambiguous or out of range): exactly one of `bgm.library=<a sound-design library folder>` (the user's
  own files, imported with their licence by `sound-design library-import`; no engine) or `bgm.prompt=<what the music is>` (ACE-Step 1.5, generated);
  `bgm.tags` (comma-separated, both modes), `bgm.mood` / `bgm.file` (a track inside the library) / `bgm.crossfade` (seconds, loops) for a library,
  `bgm.seed` for a generated bed. Needs `audio.production=true` and `audio.mix=true`: the bed is a mixed input (levels by
  `audio.mix.levels`, ducking by `audio.mix.duck.input` = its index, which is the number of audio inputs).
- **Decision**: TRANSFORM `audio.bgm` (risk MEDIUM, approval `audio.bgm.approval` with floor CONFIRM: a library licence is declared, not cleared;
  a generated bed is not checked against existing recordings -- the user listens and decides). Without sound-design-skill, or with the op not
  ready (ACE-Step not installed for `bgm.prompt`), the step is BLOCKED and handed off naming sound-design-skill.
- **Length**: the mix lasts as long as its first input (ADR-063), so the planner asks for a bed of exactly that length (the first input's length
  after its cut); library-pick trims or loops a file to it, generate-bgm generates it (10..600 s for ACE-Step)."""
from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models import Requirement
from .editing import EditRequirementError

PREFIX = "bgm."
SUBJECT = "audio.bgm"
BGM_INPUT = "bgm"            # the bed's id among the mix's inputs
BGM_OUTPUT = "bgm_bed"       # the WAV sound-design writes
LIBRARY_SKILL, GENERATE_SKILL = "bgm_library", "bgm_generate"
TOOL_NAMES = {LIBRARY_SKILL: "library-pick", GENERATE_SKILL: "generate-bgm"}
REQUIREMENT_KEYS = ("bgm.library", "bgm.prompt", "bgm.tags", "bgm.mood", "bgm.file", "bgm.crossfade", "bgm.seed")
APPROVAL_KEY = ("audio.bgm.approval", "CONFIRM")
LIBRARY_ONLY = ("bgm.mood", "bgm.file", "bgm.crossfade")
MAX_PROMPT = 511


def _words(v: Any, key: str) -> List[str]:
    vals = v.split(",") if isinstance(v, str) else (list(v) if isinstance(v, (list, tuple)) else None)
    if vals is None:
        raise EditRequirementError(f"{key} must be comma-separated words, got {v!r}")
    out = [str(x).strip() for x in vals if str(x).strip()]
    if not out or len(out) > 16 or any(len(x) > 64 or any(ord(c) < 32 for c in x) for x in out):
        raise EditRequirementError(f"{key} must be 1..16 plain words of up to 64 characters")
    return out


def parse_bgm_requirements(values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The bed asked for ({mode, library | prompt, tags?, mood?, file?, crossfade?, seed?}) or None; EditRequirementError on anything ambiguous."""
    given = {k: v for k, v in (values or {}).items() if k.startswith(PREFIX)}
    unknown = sorted(k for k in given if k not in REQUIREMENT_KEYS)
    if unknown:
        raise EditRequirementError(f"unknown requirement key(s): {', '.join(unknown)}; known: {', '.join(REQUIREMENT_KEYS)}")
    given = {k: v for k, v in given.items() if v is not None}
    if not given:
        return None
    lib, prompt = given.get("bgm.library"), given.get("bgm.prompt")
    if (lib is None) == (prompt is None):
        raise EditRequirementError("a music bed needs exactly one of bgm.library=<folder> (your licensed files) or bgm.prompt=<text> (generated)"
                                   + ("" if lib is None else ", not both"))
    out: Dict[str, Any] = {}
    if lib is not None:
        if not isinstance(lib, str) or not os.path.isdir(lib):
            raise EditRequirementError(f"bgm.library must be an existing folder, got {lib!r}")
        if not os.path.isfile(os.path.join(lib, "sound-library.json")):
            raise EditRequirementError(f"bgm.library {os.path.basename(lib.rstrip(os.sep))!r} is not a sound-design library: import each file with its licence first "
                                       "(sound-design library-import <file> --library <folder> --licence <licence> --kind bgm)")
        out.update(mode="library", library=os.path.abspath(lib))
        if "bgm.mood" in given:
            out["mood"] = _words(given["bgm.mood"], "bgm.mood")
        if "bgm.file" in given:
            f = given["bgm.file"]
            if not isinstance(f, str) or not f.strip() or os.path.isabs(f) or ".." in f.replace("\\", "/").split("/"):
                raise EditRequirementError("bgm.file must be a path inside the library (relative, no '..')")
            out["file"] = f
        if "bgm.crossfade" in given:
            c = given["bgm.crossfade"]
            try:
                c = float(c)
            except (TypeError, ValueError):
                raise EditRequirementError(f"bgm.crossfade must be a number of seconds, got {given['bgm.crossfade']!r}") from None
            if not math.isfinite(c) or not 0 <= c <= 10:
                raise EditRequirementError("bgm.crossfade must be 0..10 s")
            out["crossfade"] = c
        if "bgm.seed" in given:
            raise EditRequirementError("bgm.seed applies to a generated bed (bgm.prompt) only")
    else:
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT or any(ord(c) < 32 for c in prompt):
            raise EditRequirementError(f"bgm.prompt must be one line of 1..{MAX_PROMPT} characters")
        bad = [k for k in LIBRARY_ONLY if k in given]
        if bad:
            raise EditRequirementError(f"{', '.join(bad)} apply to a library bed (bgm.library) only")
        out.update(mode="generate", prompt=prompt.strip())
        if "bgm.seed" in given:
            s = given["bgm.seed"]
            if isinstance(s, str) and s.strip().isdigit():
                s = int(s.strip())
            if isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2 ** 32:
                raise EditRequirementError("bgm.seed must be an integer in [0, 2**32)")
            out["seed"] = s
    if "bgm.tags" in given:
        out["tags"] = _words(given["bgm.tags"], "bgm.tags")
    if values.get("audio.production") in (None, False, "false") or values.get("audio.mix") in (None, False, "false"):
        raise EditRequirementError("a music bed is placed by audio-production MIX: it needs audio.production=true and audio.mix=true (the bed is the mix's last input)")
    return out


def check_bgm_keys(user_requirements: Dict[str, Any]) -> None:
    parse_bgm_requirements(user_requirements)


def skill_of(bgm: Dict[str, Any]) -> str:
    return LIBRARY_SKILL if bgm["mode"] == "library" else GENERATE_SKILL


def decide_bgm(eng: Any, m: Dict[str, Requirement], approval_for: Callable, evidence: List[str], cap_block: Callable[[str, str], Any]) -> Optional[Dict[str, Any]]:
    """The TRANSFORM `audio.bgm` when the request asks for a bed (the caller has made the mix decision with the bed as its last input)."""
    reqs = [r for k, r in sorted(m.items()) if k.startswith(PREFIX)]
    bgm = parse_bgm_requirements({k: r.value for k, r in m.items()})
    if bgm is None:
        return None
    what = (f"a bed from the library {os.path.basename(bgm['library'].rstrip(os.sep))}" + (f" ({bgm['file']})" if bgm.get("file") else "")
            if bgm["mode"] == "library" else f"a generated bed: {bgm['prompt'][:80]}")
    words = "".join(f", {k} {','.join(bgm[k]) if isinstance(bgm[k], list) else bgm[k]}" for k in ("tags", "mood", "crossfade", "seed") if k in bgm)
    eng.decide(subject=SUBJECT, type="TRANSFORM", decision=f"audio.bgm: {what}{words}, as the mix's last input ({BGM_INPUT})",
               reason=f"user asked for a music bed ({reqs[0].source}); sound-design-skill makes it at the mix's length and audio-production MIX places it"
                      + ("; the library's licences are as the user declared them (recorded, not cleared)" if bgm["mode"] == "library"
                         else "; generated music is not checked against existing recordings (listen before publishing)"),
               confidence=1.0, evidence=list(evidence) + [r.id for r in reqs], risk="MEDIUM", approval=approval_for(SUBJECT, floor="CONFIRM"), provenance="USER",
               params={"asset_id": BGM_INPUT, **{k: v for k, v in bgm.items()}}, requirements=reqs, serves_intent=None)
    cap_block(skill_of(bgm), f"capability.{skill_of(bgm)}")
    return bgm


def ir_bgm(d: Any, duration: float) -> Dict[str, Any]:
    """The IR operation of the bed: its mode and parameters as decided, the length the mix needs."""
    p = {k: d.params[k] for k in ("mode", "library", "prompt", "tags", "mood", "file", "crossfade", "seed") if d.params.get(k) is not None}
    return {"type": SUBJECT, "asset": BGM_INPUT, "output": BGM_OUTPUT, **p, "duration": round(float(duration), 3), "decision_ids": [d.id]}


IR_KEYS = ("type", "asset", "output", "mode", "library", "prompt", "tags", "mood", "file", "crossfade", "seed", "duration", "decision_ids")


def bgm_problems(op: Dict[str, Any], programme: Optional[Dict[str, Any]]) -> List[str]:
    """Validator rules of `audio.bgm`: its own vocabulary, one mode with its parameters, a duration in 1..3600 s equal to the mix's length, and the
    mix taking the bed as its last input."""
    errs: List[str] = []
    extra = sorted(k for k in op if k not in IR_KEYS)
    if extra:
        errs.append(f"{SUBJECT}: parameters {extra} are not in the operation's vocabulary")
    if op.get("asset") != BGM_INPUT or op.get("output") != BGM_OUTPUT:
        errs.append(f"{SUBJECT}: subject must be {BGM_INPUT} and output {BGM_OUTPUT}")
    mode = op.get("mode")
    if mode == "library":
        if not op.get("library") or op.get("prompt") is not None or op.get("seed") is not None:
            errs.append(f"{SUBJECT} library: needs library, takes no prompt / seed")
    elif mode == "generate":
        if not op.get("prompt") or any(op.get(k) is not None for k in ("library", "mood", "file", "crossfade")):
            errs.append(f"{SUBJECT} generate: needs prompt, takes no library / mood / file / crossfade")
    else:
        errs.append(f"{SUBJECT}: mode {mode!r} is not library / generate")
    dur = op.get("duration")
    if isinstance(dur, bool) or not isinstance(dur, (int, float)) or not 1 <= float(dur) <= 3600:
        errs.append(f"{SUBJECT}: duration {dur!r} must be 1..3600 s")
    if programme is None or programme.get("type") != "audio.mix" or (programme.get("inputs") or [None])[-1] != BGM_INPUT:
        errs.append(f"{SUBJECT}: the bed is placed by audio.mix as its last input ({BGM_INPUT})")
    else:
        first = next((s for s in programme.get("segments") or [] if s.get("input") == (programme.get("inputs") or [None])[0]), None)
        if first is not None and isinstance(dur, (int, float)) and abs(float(dur) - (float(first["source_range"][1]) - float(first["source_range"][0]))) > 0.01:
            errs.append(f"{SUBJECT}: duration {dur} is not the mix's length (its first input's)")
    return errs
