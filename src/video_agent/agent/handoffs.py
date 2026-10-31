"""Ecosystem jobs this agent hands off (ADR-057, on ADR-051): a plan that asks for a job another package of the ecosystem owns
gets a step for it, BLOCKED with a handoff naming the package and its install command, instead of the request being dropped.

- **Requirement** `job.<skill>` = true, for every phase-2 SkillSpec in `ECOSYSTEM_JOB_SKILLS` (e.g. `job.dubbing=true`,
  `job.bluray_authoring=true`). `false` / `none` / absent: nothing. Any other value, or an unknown job, is refused before
  analysis (never guessed).
- **Decision**: one TRANSFORM `job.<skill>` (what was asked, CONFIRM, grounded on the requirement) and one BLOCK on the same
  subject (no adapter here), so the plan is BLOCKED. The agent never runs, records or verifies the job itself.
- **Plan**: one step per requested job (`step_job_<skill>`, inputs: the request's assets, tool None) that `Service.handoffs`
  reports with the owning package, repository and install commands."""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from ..models import Requirement
from ..skills.ecosystem import repository
from ..skills.registry import ECOSYSTEM_JOB_SKILLS
from .editing import EditRequirementError

JOB_PREFIX = "job."
JOB_SKILLS = {s.name: s for s in ECOSYSTEM_JOB_SKILLS}
# jobs whose adapter has landed (roadmap step 9): asked for the way the implemented skill takes them, never as a handoff switch
MOVED_JOBS = {"media_ingest": "the card offload is implemented (skill media_offload, ADR-066): give the card folder as the input and name the destinations "
                              "with --set ingest.dest=<dir> (once per drive); the plan asks you to confirm before anything is copied",
             # ADR-073: voice-enhance-skill's adapter (roadmap step 9)
             "voice_enhancement": "voice enhancement is implemented (skills voice_isolate / voice_dereverb / voice_bleed_reduce, ADR-073): with --set audio.production=true, "
                                  "ask for --set voice.isolate=true, voice.dereverb=true and / or voice.bleed_reduce=true (with audio.mix=true); every op needs confirmation (EXPERIMENTAL)",
             # ADR-084: publish-skill's adapter (its roadmap step 6)
             "platform_publish": "publishing is implemented (skill publish_upload, ADR-084): with the QC gate on (--set qc=true) ask for --set publish.platform=<id> "
                                 "(+ publish.title / publish.description / publish.tags / publish.privacy / publish.thumbnail / publish.captions); the plan asks you to "
                                 "confirm, render makes publish-skill's offline plan and dry run, and the upload is `video-agent publish <artifact> --confirm <token>` after `deliver`"}
_ON = (True, "true", "yes", "on", "1")
_OFF = (False, None, "false", "no", "off", "0", "none")


def _value(v: Any) -> Any:
    return v.strip().lower() if isinstance(v, str) else v


def check_job_keys(user_requirements: Dict[str, Any]) -> None:
    """Refuse an unknown `job.*` key or a value that is not a plain switch (before any analysis runs)."""
    for k, v in user_requirements.items():
        if not k.startswith(JOB_PREFIX):
            continue
        name = k[len(JOB_PREFIX):]
        if name in MOVED_JOBS:
            raise EditRequirementError(f"{k}: {MOVED_JOBS[name]}")
        if name not in JOB_SKILLS:
            raise EditRequirementError(f"unknown job {k!r}; known: {', '.join(JOB_PREFIX + n for n in sorted(JOB_SKILLS))}")
        if _value(v) not in _ON + _OFF:
            raise EditRequirementError(f"{k} is a switch (true / false), got {v!r}")


def requested_jobs(m: Dict[str, Requirement]) -> List[Requirement]:
    """The `job.<skill>` requirements that ask for a job, in a stable order."""
    out = []
    for name in sorted(JOB_SKILLS):
        r = m.get(JOB_PREFIX + name)
        if r is not None and r.provenance == "USER" and _value(r.value) in _ON:
            out.append(r)
    return out


def decide_jobs(eng: Any, m: Dict[str, Requirement], asset_ids: List[str], probe_ids_of: Callable[[List[str]], List[str]]) -> None:
    """One TRANSFORM per requested job (the ask, cited by its plan step) and one BLOCK (no adapter here: the plan cannot run it)."""
    for r in requested_jobs(m):
        name = r.key[len(JOB_PREFIX):]
        spec = JOB_SKILLS[name]
        pkg = spec.tools[0].split("/", 1)[0]
        eng.decide(subject=r.key, type="TRANSFORM", decision=f"{name}: {spec.description}",
                   reason=f"user asked for {name} ({r.source}); the job belongs to the {pkg} package, which this agent has no adapter for, so it is handed off",
                   confidence=1.0, evidence=[r.id] + probe_ids_of(asset_ids), risk=spec.risk_level, approval="CONFIRM", provenance="USER",
                   params={"skill": name}, requirements=[r])
        eng.decide(subject=r.key, type="BLOCK", decision=f"BLOCK: {name} is handed off to {repository(pkg)}",
                   reason=f"skill {name} is declared for phase {spec.phase} and not implemented in this agent (no adapter); "
                          f"install {repository(pkg)} and do the job there, or plan without it (see the plan's handoff)",
                   confidence=1.0, evidence=[f"skill:{name}"], risk="HIGH", approval="BLOCK", provenance="SYSTEM", params={"skill": name, "missing": [pkg]})
