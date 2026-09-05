# video-production-agent — Master Specification

## 1. Purpose

Build an **AI Video Production Orchestrator**, not a simple FFmpeg wrapper.

The system should understand a user's production request, derive requirements, inspect media, distinguish observations from inference, consider policies and constraints, resolve available capabilities, select appropriate skills/tools, produce an explainable production plan, convert it into a validated Project IR, execute through media-processing tools such as `ffmpeg-skill`, verify the result, recover safely when possible, support human review/revision, and deliver reproducible artifacts.

Core lifecycle:

```text
Understand
→ Observe
→ Analyze
→ Infer
→ Decide
→ Plan
→ Validate
→ Execute
→ Verify
→ Recover
→ Review
→ Deliver
→ Archive
```

## 2. Existing ffmpeg-skill

Existing repository:

`kajisho5/ffmpeg-skill`

It must be investigated before architecture is finalized.

Do not infer its interface from a README alone. Inspect its actual code, CLI, JSON/MCP contracts, tests, evals, references, examples, workflows, and supported operations.

## 3. Responsibility boundary

### ffmpeg-skill: Media Processing Engine / Hands

Responsible for low-level media processing such as:

- FFmpeg / FFprobe
- probing
- encoding/transcoding
- cutting/joining
- rendering
- fitting
- captions
- audio processing
- loudness
- color
- overlays
- export
- sync
- multicam processing
- silence processing
- scene detection
- visual inspection
- verification
- delivery checks
- MCP
- related deterministic media operations

### video-production-agent: AI Production Brain / Orchestrator

Responsible for:

- request understanding
- requirement extraction
- intent
- asset understanding/classification
- analysis integration
- observation/inference separation
- production policy
- constraints
- decision making
- planning
- capability resolution
- skill selection
- tool selection
- Project IR
- execution orchestration
- human approval
- feedback/revision
- recovery orchestration
- QA integration
- lifecycle
- jobs/artifacts
- audit/provenance
- evals
- AI provider abstraction
- production profiles
- domain-specific production logic

## 4. Prohibitions

Never:

- copy ffmpeg-skill wholesale
- unnecessarily reimplement its functionality
- scatter raw FFmpeg commands through Agent code
- allow an LLM to directly execute arbitrary shell commands
- overwrite source media
- turn guesses into facts
- silently perform high-risk semantic edits
- retry forever
- hard-code secrets
- make the system OS-specific without a strong reason
- build a huge Web UI before the core engine is reliable
- add dependencies without justification
- claim planned features are implemented

## 5. Architecture

Target architecture:

```text
USER
 ↓
REQUEST
 ↓
REQUIREMENTS
 ↓
INTENT
 ↓
ASSET / MEDIA ANALYSIS
 ↓
OBSERVATION
 ↓
INFERENCE
 ↓
POLICY
 ↓
CONSTRAINTS
 ↓
DECISION ENGINE
 ↓
CAPABILITY RESOLVER
 ↓
SKILL SELECTOR
 ↓
TOOL SELECTOR
 ↓
PRODUCTION PLAN
 ↓
PROJECT IR
 ↓
VALIDATION
 ↓
EXECUTION COMPILER
 ↓
EXECUTION
 ↓
OBSERVATION
 ↓
QA
 ↓
RECOVERY / REVISION
 ↓
RE-QA
 ↓
REVIEW / APPROVAL
 ↓
DELIVERY
 ↓
REPORT
 ↓
ARCHIVE
```

AI should decide what should happen; validated execution layers should decide how it is safely executed.

## 6. Request, Requirements, Intent

Do not convert a raw user request directly into execution.

Preserve:

- original Request
- derived Requirements
- Intent

Requirements must distinguish explicit user requirements from defaults, profile rules, system constraints, and AI inference.

Suggested provenance categories:

```text
USER
SYSTEM
PROFILE
DEFAULT
OBSERVED
INFERRED
AI_GENERATED
```

## 7. Observation vs Inference

These must be separate.

Example:

```text
OBSERVED:
audio below threshold for 8.2 seconds
```

is evidence.

```text
INFERRED:
likely unwanted silence
```

is an interpretation.

Never overwrite observed evidence with inference.

### Observation / Analysis architecture (implemented, ADR-019)

```text
Asset → AnalysisRequest → Analyzer → Observation → AnalysisResult (evidence) → Inference → Decision
```

- `AnalysisKind` (`media/analysis.py`): only measurements this codebase runs today — `media_probe`, `silence`, `loudness`.
  Nothing is declared for later; an unknown kind is `ANALYSIS_UNSUPPORTED`.
- `AnalysisRequest`: inputs, kinds, strategy (FULL / TARGETED / CACHED_ONLY), `AnalysisBudget`, cache policy (use / bypass /
  only), parameters, hashing. Under TARGETED the kinds come from the requirements (`targeted_kinds`); CACHED_ONLY never
  runs an analyzer. What to observe is decided by the system / policy — never by an AI provider.
- `Analyzer` contract: id, version, supported kinds, `analyze(request) → AnalysisResult`. The implemented analyzer is
  `MediaAnalyzer` (`media@1.0`); it measures only through registry-selected tools (`ToolAdapter.measure`) and never calls an
  AI provider, builds a decision or an IR, or accepts a command.
- `Observation`: kind, asset, `source = "<package>/<tool>@<version>"`, structured data, `analysis_id`, `analyzer`, `cache_key`,
  `provenance = OBSERVED` (always). Every analyzer result is validated (`validate_observation`: asset / kind / source /
  analysis id / structure / no credential or command material) before it is stored; invalid results are recorded as
  `ANALYSIS_INVALID_RESULT`, never as observations.
- `ObservationCache` (`<workspace>/cache/observations/`): key = asset fingerprint (sha256, or size:mtime with `--no-hash`) +
  kind + analyzer id@version + tool id@version + parameters. Hit → reuse (no tool call); miss → measure and store; any
  change of content, analyzer version, tool version or parameters is a new key. Observation id ≠ cache key ≠ analysis id.
  The cache is evidence reuse; job resume state (`completed_ops`) is a separate mechanism.
- `AnalysisBudget`: `max_analysis_calls`, `max_total_seconds` (legacy `analysis.budget.max_processing_time`) are enforced
  before each tool call; other names are refused as `ANALYSIS_UNSUPPORTED`. Exceeding stops further measurements,
  fabricates nothing, and is recorded per measurement. It is not the AI call budget (`analysis.budget.max_ai_calls`).
- Analysis provenance: `analysis.analyses[]` (analysis id, request, analyzer, timestamps, per-measurement rows with tool,
  cache key / hit, status, error; budget usage; cache statistics). Separate from `provenance.ai_calls`.
- Failure domain: `ANALYZER_UNAVAILABLE`, `ANALYZER_TIMEOUT`, `ANALYSIS_BUDGET_EXCEEDED`, `ANALYSIS_CACHE_INVALID`,
  `ANALYSIS_INVALID_RESULT`, `ANALYSIS_UNSUPPORTED` — distinct from `AIProviderError` and from media-engine incidents. A
  failed measurement is a warning plus a provenance row; the plan proceeds deterministically on the evidence it has.
- AI evidence: `build_request` offers only real, `OBSERVED`, tool-sourced observations with credential / command-like
  material scrubbed; an AI response can cite only those ids and can never create an observation.

Observation ≠ Inference · Analysis ≠ AI reasoning · AI evidence ≠ executable instruction · Analysis budget ≠ AI call
budget · Analysis cache ≠ Job resume state.

## 8. Temporal Model

Time is a first-class concept.

Design a unified timeline capable of representing:

```text
Video Events
Audio Events
Speech Events
Scene Events
Slide Events
Speaker Events
Caption Events
Incident Events
Camera Events
User Decision Events
```

Events should support, where appropriate:

- start
- end
- type
- source
- confidence
- evidence
- metadata

The model should support queries such as:

- show the section where speaker A is speaking
- find where slide 12 is displayed
- find incidents between two timestamps
- identify camera state during a speech event

### Temporal / Event / Session architecture (implemented, ADR-020)

```text
Project → Asset → Analysis → Observation → Event → Session → (Production Plan → Project IR)
```

- Timebase: seconds as float (`TimePoint`, `TimeRange` a.k.a. `TemporalRange`; `TIME_EPS` = 1e-6 for comparisons).
  Ranges validate `start >= 0`, `end >= start`; `end = None` is a point event. Relations: `overlaps`, `contains`,
  `precedes`, `adjacent`, `within(duration)`. Timecode strings are never the internal representation.
- `Event` (`models.Event`, `temporal/events.py`): canonical code `type` (e.g. `AUDIO_SILENCE`) plus domain
  classification `event_type` / `subtype` (`EVENT_TYPES`: AudioEvent, SpeechEvent, SpeakerEvent, SceneEvent, SlideEvent,
  CameraEvent, IncidentEvent, CaptionEvent, UserDecisionEvent with fixed subtypes), `asset_id`, range, `source`,
  `kind` (OBSERVED / INFERRED / USER) and `provenance` (OBSERVED / DERIVED / INFERRED / AI_GENERATED / USER), `evidence`
  (existing observation / decision / inference ids), `generator`, optional `session_id` / `confidence`.
  Only `IMPLEMENTED_CODES` are ever generated: AUDIO_SILENCE / AUDIO_ACTIVE / LOUDNESS_MEASURE from observations,
  USER_DECISION from reviews. Speech / speaker / slide / camera / scene / caption / incident types are schema only.
- Observation → Event is a deterministic transformation (`events_from_observation`, `observation_to_event@1.0`): only a
  validated tool measurement (provenance OBSERVED, source `<tool>@<version>`) becomes an OBSERVED event; media_probe
  yields none. Event identity is `evt_` + hash(asset, code, subtype, range, source, evidence) — regenerating from the
  same observation gives the same id and `Timeline.add` is idempotent. Event id ≠ observation id ≠ analysis id ≠ cache
  key ≠ operation id ≠ job id.
- `Session` (`temporal/session.py`): id (deterministic from project, name, range, assets), project_id, name, range
  (end > start), asset_ids, event_ids, metadata, provenance (SYSTEM / USER), generator. Validation: assets exist, range
  within asset durations, child events on the session's assets and inside its range (never clipped). Today one
  default session per asset (`session_for_asset`, whole duration) is recorded; no automatic session detection.
- Validation (`validate_event`, `validate_session`, applied by the IR validator): ids, types, asset references,
  temporal bounds (unknown durations are not checked and never guessed), evidence existence, provenance ↔ kind
  consistency (an AI_GENERATED or INFERRED event is never kind OBSERVED), confidence range, no credential / command /
  argv material.
- AI boundary: events reach a provider only through `safe_event_summary` (existing, validated, provenance and evidence
  preserved, AI_GENERATED excluded, metadata scrubbed); an AI response can cite existing event ids but never creates an
  event. Its output stays an Inference (ADR-018).
- Project IR: `timeline.events` / `timeline.sessions` record the temporal layer; it is not plan content (`plan_hash`
  unchanged) and never an execution instruction. Production planning from events is future work.
- Incidents remain QA-domain objects (`qa.incidents`); an `IncidentEvent` would reference them by id (schema only today).

Observation = measured fact · Event = temporal domain occurrence · Inference = interpretation · Decision = production
choice · Session = temporal grouping · Production Plan = production intent · Project IR = execution contract.

## 9. Event / Session / Project / Production

Do not force conference concepts onto ordinary videos.

For domain-heavy workflows, support:

```text
Production
 └── Event
      └── Session
           ├── Presentation
           ├── Q&A
           └── Break
```

A generic single-video project should remain simple.

## 10. Asset Model

Treat media as Assets, not merely paths.

An Asset should be extensible to include:

- id
- path
- type
- hash
- technical metadata
- analysis
- classification
- provenance
- relationships
- status

Example types:

```text
CAMERA
AUDIO
SLIDE
SCREEN_CAPTURE
MUSIC
BGM
LOGO
IMAGE
GRAPHIC
CAPTION
UNKNOWN
```

Classification should support confidence and evidence.

## 11. Asset Relationships

Support relationships between assets.

Examples:

```text
camera_a → Session A
room_audio → Session A
slides → Session A
slide_12 → timestamp 00:12:03
```

Use a graph-friendly model without forcing a graph database in Phase 1.

## 12. Skill / Capability / Tool

These are different concepts.

### Skill

What the system knows how to accomplish.

Examples:

```text
silence_cleanup
caption_generation
multicam_edit
slide_sync
loudness_normalization
scene_detection
```

### Capability

What is currently possible in this environment.

Examples:

```text
HEVC encoder available
GPU available
Japanese font available
libass available
transcription available
PowerPoint parser available
```

### Tool

What actually performs an operation.

Examples:

```text
ffmpeg-skill
ffmpeg
ffprobe
Whisper
OCR
PowerPoint parser
```

Target relationship:

```text
Skill
 ↓
Capability Resolution
 ↓
Tool Selection
 ↓
Execution
```

## 13. Skill Registry

Design a registry that can describe:

- name
- version
- description
- inputs
- outputs
- required capabilities
- risk level
- deterministic flag
- approval requirement
- tools

Do not build a massive plugin framework prematurely; establish a clean contract.

## 14. Capability Resolver

Detect and expose available:

- FFmpeg
- FFprobe
- codecs
- encoders
- decoders
- GPU
- hardware encoders
- libass
- zimg
- ProRes
- HEVC
- AV1
- fonts
- Japanese fonts
- optional AI tools

Statuses:

```text
AVAILABLE
MISSING
DEGRADED
UNKNOWN
```

A plan must not silently depend on unavailable capabilities.

## 15. Decision Engine

Decisions should be derived from:

```text
Requirements
+
Observations
+
Inferences
+
Policy
+
Constraints
+
Capabilities
+
Available Skills
```

A decision should support:

```json
{
  "decision": "...",
  "reason": "...",
  "confidence": 0.95,
  "evidence": [],
  "alternatives": [],
  "risk": "LOW",
  "approval": "AUTO"
}
```

## 16. Risk and Approval

At minimum:

```text
AUTO
CONFIRM
BLOCK
```

Use risk independently from confidence.

A high-confidence decision can still be high-risk.

Examples:

- codec detection → AUTO
- removing a clearly detected technical leading silence → potentially AUTO
- deleting a speaker's content based on meaning → CONFIRM
- required capability unavailable → BLOCK

## 17. Policy / Preference / Constraint

Keep these distinct.

### Policy

A production rule.

### Preference

A user's or organization's preference.

### Constraint

A hard production limitation.

Potential precedence may include:

```text
GLOBAL POLICY
→ ORGANIZATION POLICY
→ EVENT POLICY
→ PROJECT POLICY
→ PROFILE
→ USER REQUEST
```

But do not invent precedence when conflicting requirements are ambiguous.

User feedback must not automatically become permanent policy.

## 18. Production Profiles

At minimum, design for:

```text
generic
youtube
social
conference
webinar
broadcast
archive
```

Profiles should be extensible and should not be unnecessarily hard-coded.

## 19. Conference Profile

Conference production is a strategic domain.

Potential inputs:

```text
camera_a
camera_b
camera_c
presentation
slides
presentation_audio
room_audio
logo
```

Potential pipeline:

```text
Asset Classification
→ Synchronization
→ Session Segmentation
→ Slide Detection
→ Speaker Detection
→ Camera Selection
→ Audio Processing
→ Caption
→ Chapter
→ Master
→ Web
→ YouTube
→ QA
→ Delivery
```

Do not implement the whole pipeline in Phase 1. Preserve clean boundaries for later implementation.

## 20. PowerPoint Integration

Design for future integration of:

```text
PowerPoint
+
Camera
+
Audio
+
Recording
```

and relationships such as:

```text
slide_012
→ video 00:12:03–00:13:42
```

Do not force full PowerPoint implementation into Phase 1.

## 21. Production Plan vs Project IR

Keep these separate.

### Production Plan

Human-readable intent and operation plan.

Example:

> Remove the 8.2-second leading silence and export a YouTube-compatible H.264/AAC file.

### Project IR

Machine-readable, validated Intermediate Representation.

Example concept:

```text
operation:
  type: trim
  start: 8.2
```

The Project IR is the contract between reasoning/planning and deterministic execution.

### Production Plan (implemented, ADR-021)

```text
Observation → Event → Inference / Decision → ProductionPlan → Project IR → Compiler → Tool → Execution → QA
```

- `ProductionPlan` (`agent/production_plan.py`, recorded as the IR `plan` section): id (deterministic from project,
  version, steps, constraints), project_id, version, status, objective, inputs (asset ids), steps, outputs (planned
  artifacts: role / logical name / format / expected), decisions, events, constraints (hard rules in force),
  provenance (generator `production_planner@1.0`, decision / event / evidence ids), summary.
- `ProductionStep`: id (deterministic, e.g. `step_trim_<asset>`), order, skill (intent, registry vocabulary), tool
  (registry-selected; never taken from a decision, an event or an AI response), inputs, domain `params`
  (`STEP_PARAMETERS` per skill; anything else is refused), outputs (logical artifact names; the compiler decides paths),
  depends_on (deterministic topological order, cycles refused), evidence (inference / event / observation ids behind
  its decisions), decision_ids / decision_id, temporal_scope, status.
- Status is derived from the review state, never set by hand: DRAFT (no steps), REVIEW (a CONFIRM decision pending or
  the version awaiting re-approval), APPROVED (every step's decisions approved — explicitly or AUTO by policy),
  REJECTED (a step cites a rejected decision), BLOCKED (BLOCK decision or no tool). Only APPROVED reaches the compiler
  (`render` gate); BLOCK cannot be overridden by anyone, including an AI recommendation. Approval is decision-level
  (PR #4): `executable_steps` names the steps whose decisions are approved; execution waits until the plan is APPROVED.
- Planner (`agent/planner.py`) is deterministic: silence decisions → one trim step per asset (scope = kept range,
  `removed` = the silent ranges, evidence = the silence events / observations), loudness decision → normalisation
  step, delivery target → export + check steps, chained by `depends_on`. It never emits tool arguments or executes.
- `validate_plan` (run by the IR validator): ids, project, status ↔ review state, unique steps, deterministic order,
  dependencies / cycles, inputs (assets or earlier outputs), decisions, evidence existence, domain parameters only,
  credential / command / argv leaks, temporal scope within the asset, tool ∈ skill candidates, planned outputs.
- `explain_step` / `video-agent explain --step`: decision → inference (AI provenance from `ai_calls` when
  AI_GENERATED) → event → observation → tool source.
- `plan_hash` keeps its PR #3 meaning (assets / video / audio / delivery / qa); the plan section is not part of it.
  Revisions produce a new plan id and version; job resume is a separate identity.

Observation ≠ Event ≠ Inference ≠ Decision ≠ ProductionPlan ≠ Project IR ≠ Execution.

## 22. Project IR

Suggested top-level concepts:

```json
{
  "schema_version": "1.0",
  "project": {},
  "request": {},
  "requirements": {},
  "source": {},
  "assets": {},
  "analysis": {},
  "intent": {},
  "constraints": {},
  "policy": {},
  "decisions": [],
  "plan": {},
  "timeline": {},
  "video": {},
  "audio": {},
  "captions": {},
  "graphics": {},
  "color": {},
  "delivery": {},
  "qa": {},
  "execution": {},
  "provenance": {}
}
```

Treat this as a versioned IR, not a casual settings file.

Use JSON Schema and migration support.

## 23. Deterministic Execution

After IR generation, execution should be as deterministic as practical.

Reproducibility should account for:

```text
source hash
project IR
profile version
skill version
tool version
```

## 24. Execution Compiler

Compile Project IR into execution operations.

Do not allow raw FFmpeg command construction to leak across the Agent.

Keep command/tool-specific logic behind compiler and adapter boundaries.

## 25. ffmpeg-skill Adapter

Centralize the integration with ffmpeg-skill.

The Agent should be able to use an adapter rather than depend on implementation details scattered throughout the codebase.

Design for future alternate renderers without building them prematurely.

## 26. Dry Run

Support:

```bash
video-agent plan input.mp4
video-agent render project.json --dry-run
```

Dry Run should expose:

- operations
- required capabilities
- expected outputs
- risks
- warnings
- estimated processing characteristics where feasible

## 27. Explainability

The system should be able to answer:

- Why was this cut?
- Why this codec?
- Why this camera?
- Why does this require confirmation?
- Why was this blocked?
- What evidence supported the decision?

Use reason, evidence, confidence, risk, alternatives, and provenance.

## 28. Feedback and Revision

Treat human feedback as a first-class object.

Flow:

```text
Feedback
→ Preference / Constraint candidate
→ Plan Revision
→ Plan Diff
→ Approval
→ Execution
```

Do not silently rewrite permanent preferences from one feedback event.

## 29. Plan Versioning and Diff

Support plan versions:

```text
Plan v1
→ Feedback
→ Plan v2
```

and machine-readable diffs such as:

```text
CAMERA
12:03 A → B

AUDIO
-3 dB → -1 dB
```

## 30. QA

Successful rendering is not successful production.

QA should include:

### Video QA

- resolution
- codec
- fps
- duration
- pixel format
- aspect ratio
- black frames
- freeze frames
- corruption
- unexpected frames
- color issues

### Audio QA

- loudness
- clipping
- silence
- dropout
- channel loss
- channel imbalance
- phase
- sample rate
- channel layout

### Semantic QA

Future support for:

- missing speaker
- missing slide
- caption mismatch
- unexpected scene
- transcript mismatch
- semantic edit risk

### Delivery QA

- codec
- resolution
- duration
- file size
- container
- filename
- subtitle
- audio
- required assets
- destination requirements

## 31. Incident Model

Use a first-class Incident model.

Potential incident types:

```text
BLACK_FRAME
FREEZE
AUDIO_DROPOUT
CLIPPING
MISSING_CHANNEL
WRONG_ASPECT
WRONG_FPS
WRONG_COLOR
UNEXPECTED_SILENCE
LOUDNESS_FAILURE
CORRUPTED_FRAME
MISSING_CAPTION
MISSING_SLIDE
CAMERA_FAILURE
```

An Incident should support:

- id
- type
- severity
- start
- end
- evidence
- possible cause
- recommended action
- status

## 32. Recovery

Recovery flow:

```text
Error
→ Classification
→ Known Recovery Strategy
→ Finite Retry
→ Verification
```

Never retry indefinitely.

Record recovery actions and results in provenance/audit.

## 33. Cost and Performance Awareness

The planner should eventually consider:

```text
quality
accuracy
processing time
CPU
GPU
storage
API cost
```

For long-form analysis, support strategies such as:

```text
FULL_ANALYSIS
COARSE_ANALYSIS
TARGETED_ANALYSIS
```

without forcing expensive AI analysis everywhere.

## 34. Analysis Budget

Design for budgets such as:

```text
max_processing_time
max_ai_calls
max_storage
max_gpu_time
max_api_cost
```

Budget exhaustion should produce a controlled state such as degraded/confirm/block rather than silently violating constraints.

Implemented today (ADR-019): `max_analysis_calls`, `max_total_seconds` (alias `max_processing_time`) for analysis, `max_ai_calls` for AI (ADR-018). `max_storage`, `max_gpu_time`, `max_api_cost`, `max_bytes_scanned` are unsupported and are refused when a policy names them.

## 35. Cache and Incremental Rendering

Support future cache keys based on:

```text
source_hash
analysis_hash
project_hash
operation_hash
tool_version
profile_version
skill_version
```

Architecture should permit incremental rendering, but Phase 1 does not need a fully optimized incremental renderer.

## 36. Artifact Model

Artifacts should be first-class objects.

Potential fields:

```text
id
path
type
hash
source
generation
tool
tool_version
created_at
qa_status
```

Potential package outputs:

```text
MASTER
WEB
YOUTUBE
SOCIAL
ARCHIVE
CAPTIONS
THUMBNAIL
REPORT
```

### Artifact / Delivery / Archive (implemented, ADR-022)

```text
ProductionPlan.outputs → Project IR → Execution → Artifact → QA → Delivery → Archive
```

| Concept | Meaning | Implementation |
|---|---|---|
| File | bytes on a filesystem | the compiler decides where a job writes (`<workspace>/jobs/<job>/artifacts/…`); never an AI, never a plan step |
| Artifact | a production result: identity, links, QA status, lifecycle | `models.Artifact` + `artifacts/store.py` manifests in `<workspace>/artifacts/registry/<id>.json` |
| QA | is the result technically / semantically correct | `qa/` (PASS / WARN / FAIL; UNKNOWN when nothing was checked) |
| Delivery | promotion to a deliverable, recorded state (no upload today) | `ArtifactStore.promote(… "final")`, `video-agent deliver` |
| Archive | keep the production history: artifact ↔ plan ↔ job ↔ QA ↔ provenance | stage `archive` + per-project index `<workspace>/archive/<project>.json` (logical, no copies / ZIP) |

- Identity: `art_` + hash(project id, plan id, logical name, sha256). Same plan + same bytes → same artifact (a resumed job
  is appended to `jobs`); a revised plan or different content → a different artifact. Never derived from the path, the
  job id or a timestamp. Immutable: a promotion re-verifies the bytes; a changed file is a hash mismatch, never the same
  artifact. Re-registering an identity with other bytes is `ARTIFACT_CONFLICT`.
- Links: `job_id` / `jobs`, `operations` (IR operation ids), `step_id` (ProductionStep), `decision_ids`, `provenance`
  (ir_path, plan_hash, ir_hash, provenance.json). `video-agent explain --artifact` walks artifact → job → operations →
  step → decisions → inferences → events → observations.
- QA association: `qa_status` (PASS / WARN / FAIL / UNKNOWN) and the QA items for the artifact. Initial stage:
  `candidate` (READY) when QA is PASS or WARN (existing policy), `working` (NOT_READY) on FAIL.
- Lifecycle (`stage`, view `delivery_status`): working=NOT_READY → candidate/approved=READY → final=DELIVERED →
  archive=ARCHIVED. Gates for `final`: integrity ok, QA not FAIL / UNKNOWN, the artifact's plan currently APPROVED (a
  REJECTED / BLOCKED / REVIEW plan, or an IR that moved on to a newer plan version, cannot deliver). Archive is
  allowed from candidate / final and is terminal. No external delivery exists; `channel` (default `local`) is the
  extension point for future delivery adapters (YouTube / S3 / NAS …), which are not implemented.
- Registration happens after execution and QA inside `render`: a planned delivery output that is missing, unreadable
  or outside the workspace fails the job (`FAILED`, `artifact_error`) instead of leaving a COMPLETED job without an
  artifact. Registration never moves, renames or rewrites media.
- Naming (`artifacts/naming.py`): the profile template (`{project}_{target}_{version}`) renders a safe delivery file
  name (no separators / traversal, invalid and control characters replaced, Windows reserved names prefixed, trailing
  dot / space stripped, bounded length). It is metadata on the manifest, not the storage path.
- Path security: manifests accept only absolute, traversal-free, non-symlink paths inside the workspace (same
  boundary as the tool `PathPolicy`).
- Three hashes, three meanings: `plan_hash` = production plan identity (assets / video / audio / delivery / qa);
  `ir_hash` = execution contract identity (whole IR); artifact `sha256` = the produced bytes.

## 37. Job and Lifecycle

Support a job state machine such as:

```text
QUEUED
INGESTING
ANALYZING
PLANNING
WAITING_FOR_APPROVAL
EXECUTING
QA
RECOVERY
REVIEW
DELIVERING
COMPLETED
FAILED
BLOCKED
CANCELLED
```

Support safe cancellation and future resume.

Avoid destructive partial states.

## 38. Idempotency

Repeated execution of the same job should not destroy sources or create uncontrolled duplicate artifacts.

## 39. Final Artifact Promotion

Consider future states such as:

```text
working
candidate
approved
final
archive
```

Do not treat a merely rendered file as final until required QA/review gates are passed.

## 40. Audit / Provenance

Track important actions with:

```text
who
what
why
when
input
output
tool
tool_version
decision
result
qa
```

Link AI Decisions to executed Operations.

## 41. Reproducibility

A prior production should be reconstructable from its recorded:

```text
source hash
Project IR
profile version
skill version
tool version
```

as far as the underlying tools permit.

## 42. AI Providers

Use an adapter boundary for AI providers.

Conceptually:

```text
AIProvider
├── OpenAI
├── Anthropic
├── Local
└── Future
```

Do not hard-wire core deterministic media functions to an AI API.

### AI Provider Contract / Reasoning Boundary (implemented, ADR-018)

`providers/base.py`: `AIProvider` (name, model, `available()`, `describe()` without credentials, `complete(AIRequest) → AIResponse`),
`AIRequest` (task_type ∈ {production_recommendation, requirements_extraction}, inputs = system-produced evidence summaries,
schema, context), `AIResponse` (task_type, structured result, confidence, evidence ids, short reasoning, provider, model,
`AIUsage`, latency, response_hash), `AIProviderError` (TIMEOUT / RATE_LIMIT / MALFORMED / UNAVAILABLE / AUTH / BUDGET).
`NullProvider` is the default; no real provider is bundled.

`agent/ai_reasoning.py` is the only consumer: it builds the request from observations / events (never media, never
credentials), validates the response into `Inference`s with provenance `AI_GENERATED` (intent must be a registered,
implemented production skill; evidence must cite existing observation / event ids; tool / argv / command / risk /
approval keys are stripped), budgets calls (`analysis.budget.max_ai_calls`, default 4, one attempt, no retry) and
records every call in `provenance.ai_calls` (provider, model, task, request fingerprint, response hash, usage,
latency, outcome). Revisions reuse recorded AI inferences and spend no calls.

The decision engine treats AI inferences as proposals: a recommendation covered by a measured decision becomes extra
evidence on it (confidence / risk / approval untouched); anything else is a review decision `ai.<intent>` with approval
from policy (`ai.recommendation.approval`, default CONFIRM), risk from the skill registry, and `executable: false`.

```text
AI ≠ Tool executor    AI ≠ Skill registry    AI ≠ Compiler    AI ≠ final execution authority
AI → structured recommendation → Inference (AI_GENERATED) → Decision (policy / risk / approval) → SkillRegistry →
Capability → Tool → Project IR → Compiler → ToolRouter → Adapter → Skill package → runtime → QA → provenance
```

Observations stay measurements: the validator rejects an observation whose source is not a tool id + version.
A provider failure is an AI-domain failure (warning + `ai_calls[].error`), never a media-engine incident; the plan
stays deterministic.

## 43. Security and Workspace

Design for:

- allowed input
- allowed output
- workspace boundary
- command execution boundary
- secret handling

The Agent must not have an unconstrained file or shell execution model.

## 44. Evals

Create an evaluation system separate from unit tests.

An Eval case can contain:

```text
input
expected intent
expected requirements
expected decisions
expected warnings
expected plan characteristics
expected output characteristics
```

Include regression cases so improvements do not silently degrade prior behavior.

## 45. Environment Doctor

Provide:

```bash
video-agent doctor
```

to inspect:

```text
Python
FFmpeg
FFprobe
ffmpeg-skill
encoders
decoders
GPU
fonts
Japanese fonts
libass
zimg
AI providers
optional tools
```

with:

```text
AVAILABLE
MISSING
DEGRADED
UNKNOWN
```

## 46. CLI

Minimum target:

```bash
video-agent analyze <input>
video-agent plan <input>
video-agent plan <input> --profile youtube
video-agent plan <input> --profile conference
video-agent validate <project.json>
video-agent render <project.json>
video-agent render <project.json> --dry-run
video-agent check <output>
video-agent doctor
```

Future commands may include:

```bash
video-agent run
video-agent jobs
video-agent inspect
video-agent explain
video-agent diff
video-agent revise
video-agent archive
```

## 47. Web UI

Do not build a large Web UI in Phase 1.

Design future UI around:

- Plan Review
- Timeline Review
- Decision Review
- Approval
- Job Monitoring
- Artifact Review
- QA
- Plan Diff

## 48. Production Knowledge

Future architecture may support learned/reused production preferences such as:

- camera switching preferences
- loudness preferences
- caption style
- export settings
- naming
- graphics

But explicit user approval should be required before turning feedback into durable policy.

## 49. Conference / Medical Conference Safety

Conference and medical content can be highly sensitive to semantic changes.

Do not automatically delete or alter:

- speaker statements
- names
- medical terminology
- numbers
- drug names
- Q&A
- important presentation content

Semantic deletion should generally be CONFIRM.

## 50. Naming and Delivery

Profiles should be able to generate naming rules using concepts such as:

```text
project
event
session
speaker
date
version
format
```

## 51. Testing

Use layers:

```text
Unit
Integration
Real Media
Regression
Evals
```

Consider representative media such as:

- H.264
- HEVC
- ProRes
- VFR
- 10-bit
- stereo
- 5.1
- rotated video
- HDR
- long-form
- damaged media

Keep fixtures practical in size.

## 51A. AI Video Production Ecosystem

video-production-agent is not a single video-processing agent. It is the **Brain / Orchestrator** of an ecosystem of
independent, specialised Skill packages:

```text
video-production-agent   = Brain / Orchestrator (requirements, observation, decisions, Project IR, execution, QA, provenance)
ffmpeg-skill             = First Reference Skill: deterministic media processing (implemented: kajisho5/ffmpeg-skill)
future skills            = independent specialised skills, added without rebuilding the agent
```

Vocabulary and responsibilities (the contract lives in `src/video_agent/skills/contract.py`, details in `docs/skills.md`):

| Concept | Responsibility | Implementation |
|---|---|---|
| Skill package | what a repository can do (a capability domain) | `SkillPackage`: skill_id, name, version, description, capabilities, tools, repository, role |
| Tool | one concrete operation the package provides | `ToolSpec`: tool_id `<skill_id>/<name>`, skill_id, version, required capabilities, execution contract (inputs, produces_output, deterministic, result_keys) |
| Capability | what the runtime environment supports | `CapabilityResolver` (AVAILABLE / MISSING / DEGRADED / UNKNOWN) |
| Adapter | connects a package's tools to the runtime; executes the selected tool, decides nothing | `ToolAdapter` (`package()`, `supports`, `preview`, `run`, `measure`) |
| Registry | discovers, records and lists production skills and packages; selects a tool per skill | `SkillRegistry` (`register`, `register_package`, `get`, `all`, `packages`, `tool`, `select_tool`, `resolve_tools`, `availability`, `package_availability`) |
| Router | dispatches a selected tool id to the adapter that supports it | `ToolRouter` |
| Agent | decides what the production as a whole should do | `agent/`, `service.py` |

A production skill (`SkillSpec`, e.g. `silence_cleanup`) names the tools that can realise it as an ordered candidate
list drawn from registered packages. Tool ids are conceptual (`<skill_id>/<name>`); which engine implements them is the
adapter's business, so the same production skill can later be realised by another package's tool without changing the
planner, compiler, QA or decision code.

Status vocabulary (no new statuses): **DECLARED** = a production skill registered for a later phase (`NOT_IMPLEMENTED`);
**IMPLEMENTED** = the skill (or a package's adapter) exists in this codebase; **AVAILABLE** = usable in this environment
(capabilities present, an adapter registered, a tool selected).

Current state: **ffmpeg-skill is the only implemented Skill package.** The following are future skills, present in
documentation only and never registered, stubbed or reported as available:
media-analysis-skill, audio-production-skill, transcription-skill, subtitle-skill, video-editing-skill,
motion-graphics-skill, color-grading-skill, thumbnail-skill, qc-skill.

Not part of the ecosystem contract (deliberately absent): plugin manager, package installer, dynamic import, marketplace,
remote registry, arbitrary code loading. A package becomes known through its adapter module and one registration line.

### External observation Skill: media-analysis-skill (implemented, ADR-023)

| Skill | Role | Status |
|---|---|---|
| ffmpeg-skill | deterministic media **execution** (hands) | Reference Skill, integrated (ADR-001 / ADR-016 / ADR-017) |
| media-analysis-skill | deterministic **measurement / observation** (eyes / meters) | integrated: `tools/media_analysis/`, contract `media-analysis/contract@1`, 0.1.x |
| transcription-skill | deterministic **speech recognition** (ears): audio → timestamped Transcript | integrated: `tools/transcription/`, contract `transcription skill --json` (transcript/0.1, engine-spec/0.1, speech-event/0.1), 0.2.x (ADR-024) |
| video-production-agent | orchestration / inference / decision / planning | this repository |

- Boundary: an external process. The adapter runs `media-analysis contract --json` / `doctor --json` once and
  `media-analysis run - --json` per measurement (AnalysisRequest JSON on stdin, exactly one response document on
  stdout). The agent never imports the package, never runs ffprobe / ffmpeg for it, never forwards commands, argv,
  executable paths or credentials; the Skill's own `--workspace` / `--allowed-input` policy is passed through, not bypassed.
- Contract is the source of truth: tools, analysis kinds, kind → tool, capabilities, versions and schemas come from
  `contract --json` (`package_from_contract`); a pinned snapshot (`tools/media_analysis/contract_0.1.0.json`) gives the
  package its identity when no installation is present. `check_contract` refuses another skill id, contract / schema
  version, execution mode, tool ownership, a tool that writes media, or a provenance other than OBSERVED.
- Selection stays in the registry: `media-analysis/probe|silence|loudness` are second candidates for the core
  measurement skills (ffmpeg-skill first, unchanged behaviour); `stream_layout`, `video_format`, `audio_format`,
  `duration`, `integrity`, `scene_detection`, `timing` are measurement skills only media-analysis provides.
  `AnalysisRequest.kinds` may add them explicitly (`video-agent analyze --kind …`); FULL still runs the core kinds.
- Lifting: `response.observations[]` become agent Observations without simplification — `source`
  (`media-analysis/<tool>@<version>`), `skill` / `skill_version`, `tool`, `external_id` (the Skill's observation id),
  `fingerprint`, effective `parameters`, `cache` status, `analyzer@version`; provenance stays OBSERVED. Facts only:
  no interpretation is added (a leading silence is a segment, not a cut).
- Cache ownership: the Skill's (`--cache-dir <workspace>/cache/media-analysis`); the agent records `cache_owner` and the
  Skill's cache status per measurement and does not store those observations in its own cache.
- Failures map to the analysis failure domain (`ANALYZER_TIMEOUT`, `ANALYSIS_BUDGET_EXCEEDED`, `ANALYSIS_CACHE_INVALID`,
  `ANALYSIS_INVALID_RESULT`, `ANALYZER_UNAVAILABLE`); a malformed response (empty, text, several documents, wrong
  schema / skill / version / kind, missing observation, non-tool source) is never an observation.
- Events: a silence observation from either Skill becomes `AudioEvent(silence)` / `AudioEvent(active)` through the
  same deterministic transformation; no event becomes a command.
- One vocabulary for consumers, facts untouched: `media.analysis.loudness_facts` / `probe_facts` read a loudness or
  probe fact whichever tool measured it (`lufs` / `input_i` / `integrated_lufs`, container-based probe layouts).
  Inference and QA go through these views; the Observation keeps the tool's own keys.
- QA measures through the same boundary: `run_qa` asks the adapter for `measurement_args` (asset id, kind, declared
  parameters) so a measurement Skill can probe / meter intermediates and artifacts; the Skill's input roots are the
  agent's allowed inputs plus the workspace, the same boundary as the engine's `PathPolicy`.

The Brain includes an **AI Provider** (reasoning / model interface, §42): it contributes production intent and
inferences with evidence and confidence; it never selects a skill or tool, never emits commands, and never bypasses
policy. ffmpeg-skill, the first Reference Skill, is external OSS (100+ GitHub stars at the time of writing) — the
first real component of the ecosystem; that adoption is project context, not a functional specification.

### External recognition Skill: transcription-skill (implemented, ADR-024)

```
Agent → SkillRegistry (speech_transcription) → TranscriptionAdapter → `transcription run -` (JSON stdin/stdout)
      → transcription-skill → Engine (faster_whisper, local) → Transcript → Observation(kind=transcript) → SpeechEvent
```

- **Recognition only.** The Skill turns speech into timestamped text; the agent stores the Transcript as an Observation
  (a recognition *fact*, provenance OBSERVED) and derives SpeechEvents from its segments. Nothing here interprets the
  text, summarises it, identifies a speaker, chooses a camera, cuts, or renders a subtitle. **transcription result ≠ AI
  inference**: no AI provider is involved; the engine's output is evidence as recognised, homophone errors included.
- **Boundary.** `transcription skill --json` (contract), `doctor --json [--offline] [--allowed-input …]`, and `run -`
  with `{"tool": "transcription/transcribe", "params": {…}}` on stdin → exactly one `{"ok", "tool", "result"}` document.
  The agent never imports `transcription_skill`, never runs faster-whisper / ffmpeg / ffprobe for it, never forwards
  commands, argv, executables, environment or credentials, and never downloads a model.
- **Contract is the source of truth.** Tools, engines (EngineSpec: id, version, `execution_mode`, `requires_network`,
  capabilities, models and their availability), schema ids, capabilities and versions come from the contract;
  `tools/transcription/contract_0.2.0.json` is a snapshot for package identity only (engine availability stripped).
  `check_contract` refuses another skill id, an unsupported skill / schema version, an engine contract without `local`,
  a tool of another skill, a transcribe tool with output-writing side effects, and missing declared capabilities.
  Only `transcription/transcribe` is a measurement tool; `segments` / `export` / `check` are not called by the agent.
- **Typed request only.** `input`, `asset_id`, `language`, `engine` (must be declared by the contract), `model` (a name,
  never a path), `word_timestamps`, `temperature`, `initial_prompt`, `beam_size`, `offline`, `budget{timeout,
  max_audio_seconds}`, `cache`. `workspace` (`<workspace>/cache/transcription`) and `allowed_input_roots` (the agent's
  allowed inputs + the workspace, the same boundary as the engine's `PathPolicy`) are pinned by the adapter and refused
  in a request. Command / argv / shell / env / credential keys are refused before any process starts.
- **Input security.** raw path → traversal check → absolute → realpath → component-wise containment in a resolved
  allowed root; a symlink or junction whose target leaves the root is `symlink_escape`. The adapter refuses early; the
  Skill enforces the same roots again (`INVALID_INPUT` with `details.reason`).
- **Engine selection.** No ranking in the agent. The agent passes constraints (`engine`, `model`, `offline`,
  `language`, `word_timestamps`); the Skill's selector / model status decide and report. `MODEL_AVAILABLE` /
  `MODEL_DOWNLOAD_REQUIRED` / `MODEL_MISSING` / `MODEL_UNKNOWN` and `ENGINE_UNAVAILABLE` reasons are recorded verbatim
  (`skill_error`, `skill_details`), never re-interpreted. `--offline` (CLI / `Service(offline=True)`) is a hard
  constraint the adapter can only tighten: remote engines are refused, a model that is not local is unavailable.
- **Transcript → Observation.** `kind=transcript`, `data` = the Transcript document unchanged; `source`
  `transcription/transcribe@<skill version>`, `skill` / `skill_version` / `tool`, `external_id` = transcript id,
  `fingerprint` = the Skill's sha256 of the input, `parameters` = decoding parameters + engine / engine_version /
  execution_mode / model / model_version, `cache` = {status, key, owner: transcription}, `analyzer` = engine@version.
  `check_transcript` refuses another schema, another asset (on a fresh recognition), a non-sha256 fingerprint, an engine
  the contract does not declare, provenance from another skill / tool / version / execution mode, a missing model,
  malformed or out-of-order segments, and any segment carrying a `speaker_id`. A failed or partial result is never a
  transcript.
- **Shared asset identity.** The request carries the agent's asset id and the analyzer checks the Skill's fingerprint
  against the asset's own sha256; media-analysis and transcription observations of one file cite one asset. A cache
  hit returns the Skill's stored document unchanged (its `asset_id` is the first caller's, recorded as
  `cache.stored_asset_id`); identity is the fingerprint.
- **Cache.** Owned by the Skill (`<workspace>/cache/transcription/transcripts`). The agent records owner / status /
  key as provenance and keeps no transcript cache of its own. `CACHED_ONLY` asks the Skill (dry run) whether the
  result is cached; on a miss nothing is recognised.
- **SpeechEvent.** One `SPEECH` (`SpeechEvent` / `speech`) per segment: interval, text, language, segment / transcript
  ids, engine, confidence, `speaker_id: null`. **SpeechEvent ≠ speaker identification**: `SPEAKER` stays unimplemented,
  and no name, role or camera is ever attached. **Event → command does not exist**: inference, decision, planner,
  compiler and executor contain no SpeechEvent handling (static test); a plan built with a transcript has no step,
  operation or decision that cites it.
- **Failure domain.** Skill codes (`INVALID_INPUT`, `FILE_NOT_FOUND`, `UNSUPPORTED_MEDIA`, `ENGINE_UNAVAILABLE`,
  `MODEL_UNAVAILABLE`, `TRANSCRIPTION_FAILED`, `TRANSCRIPTION_TIMEOUT`, `BUDGET_EXCEEDED`, `INVALID_RESULT`,
  `CACHE_INVALID`) and transport failures (empty / non-JSON / several documents / crash / non-zero exit / process
  timeout 124 / contract incompatibility) map to the analysis domain (`ANALYZER_UNAVAILABLE`, `ANALYZER_TIMEOUT`,
  `ANALYSIS_INVALID_RESULT`, `ANALYSIS_UNSUPPORTED`, `ANALYSIS_BUDGET_EXCEEDED`) with the Skill's code kept alongside.
- **Capability / registry / CLI / explain.** Capability `transcription` from the Skill's doctor (AVAILABLE when the
  doctor is ok, DEGRADED when an engine is available but a model or the model cache is missing, MISSING otherwise;
  evidence: version, schemas, tools, capabilities, engines with models and availability, doctor rows; no secrets).
  Production skill `speech_transcription` (LOW / AUTO, requires `ffmpeg`, `ffprobe`, `transcription`) is the only
  candidate route to `transcription/transcribe`. `video-agent transcribe <media> [--language ja] [--engine …]
  [--model …] [--offline] [--word-timestamps] [--allowed-input DIR] [--timeout s]`, `analyze / plan --kind transcript`,
  `explain <ir> --observation <id|transcript id>` (observation → skill → tool → engine → model → transcript → asset
  → analysis → events; the chain ends at facts).
- **Not in scope (deliberately absent):** AI / LLM, speaker identification / diarization, speaker naming, camera
  switching, subtitle rendering / burn-in, semantic segmentation, chapters, automatic editing, new or cloud engines,
  whisper.cpp, MCP, plugin loader, ranking, arbitrary command execution.

## 52. Architecture / Repository

A possible structure:

```text
video-production-agent/
├── CLAUDE.md
├── README.md
├── LICENSE
├── pyproject.toml
├── docs/
│   ├── MASTER_SPEC.md
│   ├── architecture.md
│   ├── project-ir.md
│   ├── requirements.md
│   ├── decisions.md
│   ├── temporal-model.md
│   ├── skills.md
│   ├── capabilities.md
│   ├── profiles.md
│   ├── lifecycle.md
│   ├── qa.md
│   ├── recovery.md
│   ├── feedback.md
│   └── evals.md
├── schemas/
│   └── project.schema.json
├── src/
│   └── video_agent/
│       ├── agent/
│       ├── media/
│       ├── temporal/
│       ├── production/
│       ├── policy/
│       ├── skills/
│       ├── capabilities/
│       ├── tools/
│       ├── project/
│       ├── execution/
│       ├── qa/
│       ├── jobs/
│       ├── profiles/
│       ├── providers/
│       ├── audit/
│       ├── evals/
│       └── cli.py
├── profiles/
├── examples/
├── tests/
└── evals/
```

Change this if inspection shows a better design.

## 53. Development Order

Do not immediately implement every future feature.

Recommended order:

1. inspect ffmpeg-skill
2. capability inventory
3. responsibility matrix
4. architecture
5. Request model
6. Requirements model
7. Observation/Inference model
8. Temporal/Event model
9. Project IR
10. JSON Schema
11. validation
12. Capability system
13. Skill system
14. Tool adapter
15. Media Analyzer
16. Intent
17. Decision Engine
18. Planner
19. Compiler
20. QA
21. CLI
22. tests
23. Evals
24. documentation

## 54. Phase Roadmap

### Phase 1

Reliable end-to-end core:

```text
Request
→ Requirements
→ Intent
→ Analysis
→ Decision
→ Plan
→ Project IR
→ Validation
→ ffmpeg-skill
→ QA
→ Report
```

### Phase 2

Conference profile.

### Phase 3

- multicam
- slide detection
- transcription
- captions
- chapters
- highlights
- incident detection

### Phase 4

- PowerPoint integration
- semantic editing
- AI multicam
- slide synchronization
- speaker detection
- smart thumbnails
- YouTube package
- archive package

### Phase 5+

- human review UI
- plan diff UI
- job queue
- batch production
- distributed/cloud rendering
- advanced AI providers
- local AI
- OCR
- semantic search
- production knowledge
- event templates
- organization profiles

## 55. Quality Standard

Do not define success as “the code runs.”

Each phase should be evaluated for:

```text
Architecture
Correctness
Safety
Reproducibility
Testability
QA
Documentation
```

## 56. First Required Review

Before substantial implementation, report:

1. ffmpeg-skill feature inventory
2. reusable components
3. duplicated functionality to avoid
4. responsibility boundary
5. architecture
6. Request → Requirements → Intent → Observation → Inference → Decision → Plan → IR data flow
7. Temporal Model
8. Event / Session / Project / Production model
9. Skill / Capability / Tool model
10. Decision model
11. Policy / Preference / Constraint model
12. Project IR
13. Compiler / Adapter
14. QA
15. Recovery
16. Artifact / Job / Lifecycle
17. Feedback / Revision
18. Conference profile
19. Phase roadmap
20. technical risks
21. unresolved decisions
22. minimum Phase 1 scope

Do not write a large amount of implementation before this review.

## 57. Final Product Vision

A future user should be able to say:

> “Take these three conference lecture recordings, combine them, clean up unwanted silence, switch cameras when speakers change, show the appropriate slide when the slide changes, add captions, produce versions for the conference website and YouTube, normalize the audio, and check for production accidents.”

The system should eventually be able to perform:

```text
REQUEST
→ REQUIREMENTS
→ ASSET DISCOVERY
→ ASSET CLASSIFICATION
→ MEDIA ANALYSIS
→ TIMELINE EVENTS
→ INTENT
→ POLICY
→ CONSTRAINTS
→ DECISION
→ CAPABILITY CHECK
→ SKILL SELECTION
→ TOOL SELECTION
→ PRODUCTION PLAN
→ RISK ANALYSIS
→ USER APPROVAL
→ PROJECT IR
→ VALIDATION
→ EXECUTION
→ OBSERVATION
→ QA
→ RECOVERY if needed
→ RE-QA
→ REVIEW
→ DELIVERY
→ REPORT
→ ARCHIVE
```

The core principle is:

**Do not build a system whose primary job is to make an AI write FFmpeg commands.**

Build a system that models video production as a structured, explainable, verifiable workflow, with `ffmpeg-skill` serving as a powerful execution engine.
