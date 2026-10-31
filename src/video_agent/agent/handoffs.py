"""Ecosystem jobs this agent hands off (ADR-057, on ADR-051): a plan that asks for a job another package of the ecosystem owns
gets a step for it, BLOCKED with a handoff naming the package and its install command, instead of the request being dropped.

- **Requirement** `job.<skill>` = true, for every phase-2 SkillSpec in `ECOSYSTEM_JOB_SKILLS` (e.g. `job.dubbing=true`,
  `job.bluray_authoring=true`). `false` / `none` / absent: nothing. Any other value, or an unknown job, is refused before
  analysis (never guessed).
- **Decision**: one TRANSFORM `job.<skill>` (what was asked, CONFIRM, grounded on the requirement) and one BLOCK on the same
  subject (no adapter here), so the plan is BLOCKED. The agent never runs, records or verifies the job itself.
- **Plan**: one step per requested job (`step_job_<skill>`, inputs: the request's assets, tool None) that `Service.handoffs`
  reports with the owning package, repository and install commands.
- **Parameters** (ADR-080): a job whose Skill contract is pinned (skills/handoff_contracts.py) also takes `job.<skill>.<name>`
  (e.g. `job.render_3d.model=robot.glb`, `job.render_3d.preset=1080p`), checked against the pin before analysis and only with
  `job.<skill>=true`. They become the step's params and the handoff's `contract.args` (by the contract's own names); still nothing runs.
- **Measurements an explicit requirement asks for** (ADR-086: `speakers.diarize` → analysis kind `speaker_turns`, diarization-skill)
  take the same route when no tool can run them here: one TRANSFORM (the ask, `handoff: true`) and one BLOCK naming why (the
  registry's reason: the package missing, or its doctor's per-operation verdict with the Skill's own code and install line), and a
  `step_job_<skill>` with no tool for the handoff. A measurement that ran and failed on an input BLOCKs with the Skill's own code."""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from ..models import Requirement
from ..skills.ecosystem import repository
from ..skills.handoff_contracts import HandoffParameterError, parse_param, pin
from ..skills.registry import ECOSYSTEM_JOB_SKILLS
from .editing import EditRequirementError
from .speakers import SKILL as DIARIZE_SKILL, SWITCH as DIARIZE_SWITCH, speaker_kind

JOB_PREFIX = "job."
JOB_SKILLS = {s.name: s for s in ECOSYSTEM_JOB_SKILLS}
# jobs whose adapter has landed (roadmap step 9): asked for the way the implemented skill takes them, never as a handoff switch
MOVED_JOBS = {"media_ingest": "the card offload is implemented (skill media_offload, ADR-066): give the card folder as the input and name the destinations "
                              "with --set ingest.dest=<dir> (once per drive); the plan asks you to confirm before anything is copied",
             # ADR-073: voice-enhance-skill's adapter (roadmap step 9)
             "voice_enhancement": "voice enhancement is implemented (skills voice_isolate / voice_dereverb / voice_bleed_reduce, ADR-073): with --set audio.production=true, "
                                  "ask for --set voice.isolate=true, voice.dereverb=true and / or voice.bleed_reduce=true (with audio.mix=true); every op needs confirmation (EXPERIMENTAL)",
             # ADR-086: diarization-skill's adapter (roadmap step 9; diarization-skill roadmap step 5)
             "speaker_diarization": "speaker diarization is implemented (analysis kind speaker_turns, ADR-086): ask for --set speakers.diarize=true "
                                    "(+ speakers.num_speakers, or speakers.min_speakers / speakers.max_speakers); without diarization-skill the plan hands it off"}
_ON = (True, "true", "yes", "on", "1")
_OFF = (False, None, "false", "no", "off", "0", "none")


def _value(v: Any) -> Any:
    return v.strip().lower() if isinstance(v, str) else v


def check_job_keys(user_requirements: Dict[str, Any]) -> None:
    """Refuse an unknown `job.*` key or a value that is not a plain switch (before any analysis runs)."""
    for k, v in user_requirements.items():
        if not k.startswith(JOB_PREFIX):
            continue
        name, _, param = k[len(JOB_PREFIX):].partition(".")
        if name in MOVED_JOBS:
            raise EditRequirementError(f"{k}: {MOVED_JOBS[name]}")
        if name not in JOB_SKILLS:
            raise EditRequirementError(f"unknown job {k!r}; known: {', '.join(JOB_PREFIX + n for n in sorted(JOB_SKILLS))}")
        if param:   # ADR-080: a typed parameter of a job whose Skill contract is pinned
            p = pin(name)
            if p is None:
                raise EditRequirementError(f"{k}: {JOB_PREFIX}{name} takes no parameters (only {JOB_PREFIX}{name}=true)")
            try:
                parse_param(p, param, v)
            except HandoffParameterError as e:
                raise EditRequirementError(str(e))
            if _value(user_requirements.get(JOB_PREFIX + name)) not in _ON:
                raise EditRequirementError(f"{k} needs {JOB_PREFIX}{name}=true")
            continue
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


def job_params(m: Dict[str, Requirement], name: str) -> Dict[str, Any]:
    """The checked `job.<name>.<param>` values the user gave (ADR-080), by the agent's parameter names; {} for a job without a pin."""
    p = pin(name)
    if p is None:
        return {}
    out: Dict[str, Any] = {}
    for hp in p.params:
        r = m.get(f"{JOB_PREFIX}{name}.{hp.name}")
        if r is not None and r.provenance == "USER":
            out[hp.name] = parse_param(p, hp.name, r.value)
    return out


def decide_jobs(eng: Any, m: Dict[str, Requirement], asset_ids: List[str], probe_ids_of: Callable[[List[str]], List[str]]) -> None:
    """One TRANSFORM per requested job (the ask, cited by its plan step) and one BLOCK (no adapter here: the plan cannot run it)."""
    for r in requested_jobs(m):
        name = r.key[len(JOB_PREFIX):]
        spec = JOB_SKILLS[name]
        pkg = spec.tools[0].split("/", 1)[0]
        p = pin(name)
        args = job_params(m, name)
        arg_reqs = [m[f"{JOB_PREFIX}{name}.{k}"] for k in args]
        via = f" ({p.tool}, contract {p.version} pinned)" if p else ""
        eng.decide(subject=r.key, type="TRANSFORM", decision=f"{name}: {spec.description}",
                   reason=f"user asked for {name} ({r.source}); the job belongs to the {pkg} package{via}, which this agent has no adapter for, so it is handed off",
                   confidence=1.0, evidence=[r.id] + [q.id for q in arg_reqs] + probe_ids_of(asset_ids), risk=spec.risk_level, approval="CONFIRM", provenance="USER",
                   params={"skill": name, **({"args": args} if args else {})}, requirements=[r] + arg_reqs)
        eng.decide(subject=r.key, type="BLOCK", decision=f"BLOCK: {name} is handed off to {repository(pkg)}",
                   reason=f"skill {name} is declared for phase {spec.phase} and not implemented in this agent (no adapter); "
                          f"install {repository(pkg)} and do the job there, or plan without it (see the plan's handoff)",
                   confidence=1.0, evidence=[f"skill:{name}"], risk="HIGH", approval="BLOCK", provenance="SYSTEM", params={"skill": name, "missing": [pkg]})


# requirement switch → (registry skill, analysis kind, why the measurement is asked for); ADR-086
MEASUREMENT_HANDOFFS = {DIARIZE_SWITCH: (DIARIZE_SKILL, "speaker_turns", "anonymous speaker turns (who spoke when) on every input with audio")}


def requested_measurements(m: Dict[str, Requirement]) -> List[Requirement]:
    """The USER requirements that switch on a measurement of MEASUREMENT_HANDOFFS (refinements are range-checked before analysis)."""
    values = {k: r.value for k, r in m.items() if r.provenance == "USER"}
    out = []
    for key in sorted(MEASUREMENT_HANDOFFS):
        r = m.get(key)
        if r is None or r.provenance != "USER":
            continue
        if key == DIARIZE_SWITCH and speaker_kind({k: v for k, v in values.items() if k.startswith("speakers.")}) is None:
            continue
        out.append(r)
    return out


def decide_measurement_handoffs(eng: Any, m: Dict[str, Requirement], analysis: Any, probe_ids_of: Callable[[List[str]], List[str]],
                                cap_missing: Callable[[str], Any]) -> None:
    """A requested measurement that cannot run here → the ask + a BLOCK with the registry's reason (handed off); one that ran and failed on
    an input with audio → a BLOCK with the Skill's own code. A measurement that ran is the observation itself: nothing to decide."""
    rows = [r for a in getattr(analysis, "analyses", []) or [] for r in a.get("rows") or []]
    for r in requested_measurements(m):
        skill, kind, what = MEASUREMENT_HANDOFFS[r.key]
        asset_ids = [a.id for a in analysis.assets]
        why = cap_missing(skill)
        if why:
            pkgs = sorted({t.split("/", 1)[0] for t in _skill_tools(skill)})
            eng.decide(subject=r.key, type="TRANSFORM", decision=f"{kind}: {what}",
                       reason=f"user asked for {r.key} ({r.source}); no tool for {skill} here, so the measurement is handed off to {', '.join(repository(p) for p in pkgs)}",
                       confidence=1.0, evidence=[r.id] + probe_ids_of(asset_ids), risk="LOW", approval="CONFIRM", provenance="USER",
                       params={"skill": skill, "handoff": True}, requirements=[r])
            eng.decide(subject=r.key, type="BLOCK", decision=f"BLOCK: {kind} is handed off ({skill} unavailable)",
                       reason=f"{why}; install it and plan again, or plan without {r.key} (see the plan's handoff)",
                       confidence=1.0, evidence=[f"skill:{skill}"], risk="HIGH", approval="BLOCK", provenance="SYSTEM", params={"skill": skill, "missing": pkgs})
            continue
        for row in rows:
            if row.get("kind") == kind and row.get("status") == "FAILED":
                err = row.get("error") or {}
                code = err.get("skill_error") or err.get("kind")
                eng.decide(subject=r.key, type="BLOCK", decision=f"BLOCK: {kind} failed on {row.get('asset_id')} ({code})",
                           reason=f"user asked for {r.key}; the measurement failed: {code}: {str(err.get('message') or '')[:200]} (nothing is guessed in its place)",
                           confidence=1.0, evidence=[r.id] + probe_ids_of([row.get("asset_id")]), risk="HIGH", approval="BLOCK", provenance="SYSTEM",
                           params={"skill": skill, "asset_id": row.get("asset_id"), "code": code}, requirements=[r])


def _skill_tools(skill: str) -> List[str]:
    from ..skills.registry import default_registry
    return list(default_registry().get(skill).tools)
