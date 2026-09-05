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
  unchanged) and never an execution instruction. Speech / silence events reach the plan only through inferences and
  decisions (ADR-025); no event is ever lowered to an operation directly.
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

### SpeechEvent → Inference → Decision → ProductionPlan (implemented, ADR-025)

```
Observation(transcript) ─→ SpeechEvent ─┐
                                        ├─→ speech inferences ─→ decisions ─→ ProductionStep ─→ Project IR ─→ Compiler ─→ Tool
Observation(silence)    ─→ AudioEvent ──┘
```

- **Inference (deterministic, `agent/speech_inference.py`, provenance INFERRED, never AI):**
  `speech_interval` (consecutive SPEECH events with gaps ≤ `speech.merge_gap_seconds` form one logical interval; an
  interval operation on the events' own timestamps), `speech_activity` (per asset: intervals, seconds, coverage),
  `internal_silence_removable` (a measured internal silence strictly between two speech intervals, overlapping none,
  lasting ≥ `silence.internal.removable_min_seconds`; the proposed range keeps `silence.margin_seconds` of air on each
  side), `speech_silence_conflict` (recognised speech overlapping a measured silence: recorded, never resolved). Every
  inference cites its SpeechEvents / silence event / transcript observation; `speaker_id` is null throughout and a
  SpeechEvent carrying one is refused. No transcript text is read. Thresholds come from policy with an explicit
  DEFAULT (`speech.merge_gap_seconds` 0.5, `silence.internal.removable_min_seconds` 2.0, margin 0.15) and their value +
  provenance (PROFILE / USER / DEFAULT) are recorded in the inference data.
- **Silence stays the silence tool's fact.** The silence observation and its events are never modified or re-timed;
  a transcript never becomes a silence; a disagreement is a conflict inference.
- **Decision (`agent/decision.py`):** `speech.continuity` (keep all speech intervals; AUTO, LOW, no operation),
  `silence.internal.<start>-<end>` (`remove … (candidate)`; risk MEDIUM; approval from `silence.internal.approval`,
  CONFIRM in every profile, floored at CONFIRM here and BLOCK when the policy says so; one decision per pause),
  `silence.conflict.<start>-<end>` (keep; AUTO). A leading / trailing trim that overlaps a conflict is raised to CONFIRM
  (risk MEDIUM) whatever the policy says. "Speech was recognised" ≠ "this pause must go".
- **Plan (`agent/planner.py`):** a removal candidate that is not REJECTED becomes an extra removed range of the same
  `silence_cleanup` step (keep = complement, multi-range `video.trim`); the step's decisions include the candidate, so
  the step is PROPOSED and the plan REVIEW until it is approved (partial approval: other AUTO steps keep their
  semantics). Approve → APPROVED, `plan_hash` unchanged; reject → REJECTED, render BLOCKED, `revise` drops the
  candidate (suppressed by subject + asset). The compiler lowers the kept ranges (`segments`), nothing else.
- **Provenance:** `explain --step` walks step → decision → `internal_silence_removable` → silence event + the two
  `speech_interval` inferences → SPEECH events → transcript observation; `explain --observation` still ends at facts.
- **Not here:** AI / LLM, speaker identification, camera choice, semantic segmentation, chapters, subtitles, word-level
  tightening of segment boundaries (a follow-up: Whisper extends segments into pauses, which on real recordings yields
  conflicts rather than candidates), the silencedetect end > duration issue.

### ProductionContext: situation understanding (implemented, ADR-026)

```
Observation ─→ Event ─┐
                      ├─→ ProductionContext (what is observed here, at the same time) ─→ generic inference ─┐
Observation ─→ Event ─┘                                                                                      ├─→ Decision ─→ ProductionPlan ─→ IR
                                                              domain inference (speech, silence, loudness) ──┘
```

**ProductionContext ≠ Observation ≠ Event ≠ Inference ≠ Decision ≠ ProductionPlan; Event ≠ command.** A context is the
agent's intermediate representation of the production situation: for one timeline and one time scope, which kinds of
events are active at the same time, which observations they rest on, which assets they belong to, and which inferences
already cite them. It replaces nothing and copies nothing: every field is a reference, the scope is bounded by the
events' own timestamps, provenance is DERIVED, and the id is a hash of timeline + scope + active events (deterministic,
stable across plan versions). A Session is a grouping a person or the system declares; a context is derived.

- **Construction (`context/builder.py`):** for each asset timeline the events' start / end points (and 0 / duration)
  are the boundaries; each elementary interval is one context with its active events grouped into tracks by domain
  type / subtype (`AudioEvent/silence`, `SpeechEvent/speech`, later `SceneEvent/visual_change`, `CameraEvent/camera`,
  `SlideEvent/slide`, `IncidentEvent/*`, `CaptionEvent/caption` — the same Temporal Model, no new event types). An
  interval where nothing is active is a context too. `UserDecisionEvent`s are review history, not a situation. Nothing
  is snapped, corrected, merged heuristically or resolved; a point event marks a boundary.
- **Generic inference (`context/inference.py`, deterministic, INFERRED, generator `context_inference@1.0`, domain
  neutral):** `source_activity` / `source_inactivity` per (timeline, event type / subtype) — where that kind of event is
  and is not observed; `transition` — a boundary where the set of active kinds changes; `conflict` — two events whose
  codes are declared mutually exclusive (`EXCLUSIVE_PAIRS`, today `AUDIO_SILENCE` × `SPEECH`) overlap: recorded with
  both events as evidence, never resolved. Whole-programme measurements (`loudness`) yield no activity. Every inference
  cites existing events; `data.context_ids` names the situations it was derived from.
- **What it never does:** read transcript text, name a speaker, pick a source / camera / slide, propose an edit, read
  policy or preferences, create an event, change a timestamp, decide. "Source B is active" is an inference;
  "use source B" is a decision the decision engine may or may not make from policy, preference and constraints. No
  decision is created from generic inferences in this version; the domain layer (ADR-025) keeps producing the speech /
  silence decisions and cites the same events.
- **AI boundary:** an AI provider may later produce inferences of these kinds only through the existing reasoning
  boundary (provenance `AI_GENERATED`, validated evidence ids, no tool / command / approval / path). The generator on
  deterministic inferences and the provenance on AI ones keep the two apart; the validator refuses an AI inference
  recorded as OBSERVED.
- **Project IR:** `analysis.contexts[]` (schema additive); the validator checks every context (references exist,
  scope inside the asset, active events overlap the scope, id matches content). Revisions rebuild contexts from the
  same events and get the same ids. Contexts are not plan content: `plan_hash` is unchanged, the planner, compiler and
  executor never read them.
- **Explain:** `explain --context <id>` walks context → tracks → events (timestamps as recorded) → observations, then
  the inferences citing its events and the decisions resting on those inferences; `explain --observation` ends with
  the contexts an observation's events take part in; `explain --step` rows show `contexts` on inferences derived from
  situations. `video-agent context <ir> [--at s | --between a b] [--timeline …]` lists situations.
- **Genericity:** any asset (video, audio, screen capture, image, caption file) whose analysis yields events on a
  timeline takes part; no analysis Skill was added or changed. Cross-asset situations on the master timeline need the
  existing `TimelineMap` offsets from a multi-source sync capability (not available yet; recorded in GAP §16).

### Production Decision Engine (implemented, ADR-027)

```
Inference (what is happening) + Policy / Preference / Constraint + Intent + Risk ─→ Decision (what production should do) ─→ ProductionPlan (how) ─→ IR
```

**Inference = what is happening. Decision = what production should do. Plan = how it is executed.** The generic engine
(`agent/decision_engine.py`, tool- and domain-independent) does not know silence, speech, loudness or delivery; the
domain layer (`agent/decision.py`) says *which* decision a situation calls for and constructs every one of them through
the engine, which enforces *how* a decision may exist:

- **Vocabulary (`type`, additive on the existing Decision model):** KEEP / REMOVE / TRANSFORM / DELIVER / SKIP / REVIEW /
  BLOCK. Only REMOVE / TRANSFORM / DELIVER are executable (may be cited by a step or an IR operation); the validator
  refuses any other citation. Existing decisions map 1:1 (`speech.continuity` KEEP, `silence.internal.<range>` REMOVE,
  `silence.leading` REMOVE, `audio.loudness` TRANSFORM / SKIP / KEEP, `delivery.*` DELIVER, `capability.*` BLOCK,
  `ai.*` REVIEW, `policy.<key>` KEEP). Their subjects, texts, approvals and plan effects are unchanged.
- **Evidence is mandatory and classed** (observation / event / inference / requirement / rule / capability / ai). A
  decision without evidence is refused at construction. An executable decision must rest on a measured fact or on a
  requirement of the request; a PREFERENCE / POLICY rule alone, an intent alone, or AI output alone never grounds one
  (AI-only evidence yields a REVIEW item with `executable: false`; AI-proposed parameters that look like a command or a
  credential are dropped, never interpreted). A missing measurement therefore yields no decision — e.g. when the
  loudness analysis failed, there is no "within tolerance" decision any more; the analysis warning records the failure.
- **Approval from policy with a safe default and recorded provenance.** `resolve_approval(rules, key, DEFAULT, floor,
  explicit)`: AUTO / CONFIRM as stated, `BLOCK*` → BLOCK (no implicit exception for a suffix), any other value → CONFIRM,
  `floor` only raises, BLOCK is never lowered. The existing waiver (a USER requirement asking for exactly this edit turns
  a CONFIRM POLICY into AUTO, eval 03) stays, is recorded as a note, and never applies to a CONSTRAINT. RuleSet
  precedence (GLOBAL → … → PROFILE → REQUEST, constraints never overridden) is untouched; the engine only reports what
  it resolved. Keys and explicit defaults: `silence.leading.approval` AUTO, `silence.trailing.approval` AUTO,
  `silence.internal.approval` CONFIRM (floor CONFIRM), `audio.loudness.approval` AUTO, `delivery.export.approval` AUTO,
  `video.vfr.approval` AUTO, `video.hdr.approval` CONFIRM, `ai.recommendation.approval` CONFIRM. No-op decisions (KEEP /
  SKIP) are AUTO by construction and create no operation.
- **Confidence ≠ risk ≠ approval.** Confidence comes from the inference; risk and approval from policy and the kind of
  change. The same trim is AUTO on `generic` and CONFIRM on `conference` with identical confidence.
- **Conflicts** (a request or preference against a CONSTRAINT) stay a `policy.<key>` KEEP decision with approval CONFIRM
  and the reason; a trim overlapping a speech / silence conflict is raised to CONFIRM. No new implicit rule.
- **Basis (recorded on every decision, `basis`):** engine id, type, evidence classes, the settings consulted (key, value,
  kind POLICY / PREFERENCE / CONSTRAINT or DEFAULT, provenance USER / PROFILE / SYSTEM / DEFAULT, rule id, source,
  hard), the approval resolution (key, provenance, notes such as "raised AUTO → CONFIRM: recognised speech overlaps this
  silence" or "unknown approval value … CONFIRM (safe default)"), the intent (primary, secondary, provenance, which one
  this decision serves — None for a fact-backed / safety decision), the requirements consulted with provenance, and the
  risk with `independent_of_confidence: true`. The basis is decision content, not plan content: `plan_hash` ignores it.
- **Validator (`check_decisions`)** re-checks the invariants on a recorded IR: type known, evidence present and known,
  grounding, AI-only → REVIEW, BLOCK ⇔ BLOCKED, no executable / credential material, only executable types cited.
  REJECTED decisions carried as history by `revise` keep their earlier version's evidence and are exempt from the
  unknown-evidence check (their evidence lived in the snapshotted version; no operation cites them).
- **Explain (`explain --decision <id|subject>`, `Service.explain_decision`):** decision (type / rationale / risk /
  approval / status / provenance / executable) → basis rows (policy / preference / constraint / default / approval /
  intent / requirement / risk) → evidence chain (inference → contexts → events → observations → asset; requirements;
  rules) → the plan steps and IR operations that cite it (Decision → Plan → Step → IR). `--json` returns the same
  structure.
- **Security:** decision subject / text / params are scanned (`leak_scan`) for command, argv, shell, executable or
  credential material and refused; the engine imports no tool, execution, provider, speech or context module.
- **Not here:** AI / LLM, speaker identification, camera / slide / source selection, new decision domains or genre-
  specific decisions, MCP, plugin loader, ranking, direct ffmpeg, Skill changes, Artifact redesign, the silencedetect
  end > duration issue.

### video-editing-skill integration (implemented, ADR-028)

```
video-production-agent ─(typed Operation: input / output ids, keep ranges, precision)─→ VideoEditingAdapter
    ─(EditRequest JSON on stdin; argv list `run - --json --workspace <op dir> --allowed-input <root>…`)─→ video-editing CLI
    ─(typed ffmpeg-skill tool calls)─→ ffmpeg-skill ─→ FFmpeg
```

- **Boundary:** the Skill's CLI and its machine-readable contract (`video-editing contract --json`, video-editing/contract@1)
  are the only interface. The agent imports nothing from the Skill, builds no command, argv, filter or filter_complex,
  names no executable, passes no environment and no credential; `FORBIDDEN_ARG_KEYS` and the contract's declared
  parameter names bound what an Operation may carry. The Skill's `commands` come back as provenance only — there is no
  path by which the agent edits or re-runs them.
- **Contract:** fetched at start-up, checked (`check_contract`) and compared with the pinned 0.1.0 snapshot
  (`contract_drift`): tool ids `video-editing/<operation>`, versions, operations, capabilities, required capabilities,
  inputs, produces_output, deterministic, result keys, execution flags, error codes, response shape. Missing / malformed
  / drifted contracts fail loudly (ContractError; capability MISSING) and are never patched or guessed.
- **Registry / capability:** `SkillPackage` video-editing with its ToolSpecs from the contract; `silence_cleanup` lists
  `video-editing/cut` as its second candidate (declared order, no ranking, no fallback). The `video-editing` capability is
  AVAILABLE only when the Skill's doctor is ok and there is no drift; `select_tool` also checks the package's capabilities
  and the tool's required capabilities the resolver knows (encoder:aac, filter:xfade, filter:acrossfade added). An
  operation the Skill lists as unsupported (CROP, FREEZE, REVERSE, IMAGE_INSERT, POSITION) is not a tool at all.
- **Lowering:** `video.trim` → `video-editing/cut` with the contract's CUT parameters (`keep` [{start, end}], `precision`
  frame | keyframe from the plan's `accurate`); the reference lowering to `ffmpeg-skill/cut` is unchanged.
- **Execution:** one subprocess per operation in its own process group; the agent's timeout becomes the Skill's
  `options.timeout_seconds` and is enforced at the process boundary too (exit 124 → CANCELLED / timeout, retryable);
  cancellation (SIGINT) is the executor's existing path. `--workspace` is the operation's output directory inside the
  agent workspace; `--allowed-input` are the agent PathPolicy roots plus the workspace. Both sides refuse traversal,
  absolute paths outside the roots, symlink escapes and workspace escapes.
- **Response → agent model:** exactly one JSON document; `ok`, status completed | reused, `execution.outputs[out1]`
  delivered with path == requested, sha256 verified against the file, size, timeline and an OBSERVED ffmpeg-skill probe →
  ToolResult.output / data.artifact / data.timeline / data.observation; `execution.operations[op1]` (operation_id, tool,
  tool versions, inputs' hashes, output hash, timing) → data.operation; provenance.json `skill_result`. Exit 0 with a
  missing output, exit ≠ 0 with an ok document, malformed / empty / multiple documents, hash mismatch: never a success.
- **Errors:** the Skill's 13 codes and its `retryable` verdict are kept on the result and mapped to the existing recovery
  classes (INVALID_ARGS / INPUT_MISSING block, TOOL_ERROR retries once, CANCELLED retries with a longer timeout, the new
  SKILL_ERROR blocks); a non-retryable code is never retried.
- **Not here:** new editing features, changes to either Skill, tool details in the Decision Engine, a parallel executor,
  MCP / plugin loader / ranking, direct FFmpeg.

### video-editing-skill operations: concat / speed / resize / fit / fill / overlay (implemented, ADR-029)

```
Requirement edit.* (explicit --set) → Decision (TRANSFORM / BLOCK, policy video.<op>.approval) → ProductionPlan step (skill video_<op>, tool video-editing/<op>)
    → IR video.operations[] (type, subject, references, allowlisted params, temporal scope, decision ids; concat: segments + timeline_duration)
    → compiler lower_video_edit (names only) → VideoEditingAdapter (ADR-028) → video-editing CLI → ffmpeg-skill → FFmpeg
```

- **Four stages are kept apart:** *Skill supports* (contract), *agent adapter supports* (`Lowering.ARGS`, PR #18), *planner can
  generate* (this section: the six operations above; TRIM / CUT stay `video.trim`), *E2E verified* (real-media integration
  tests: A + B → concat (plain and with a fade transition) → speed → resize → fit / fill → overlay with a real PNG → QA).
- **Vocabulary (`agent/editing.py`):** `OPERATIONS` maps each IR type to its production skill, its only tool and the parameter
  allowlist (concat: transition {type, duration}, width, height, fps, mode, pad_color; speed: factor; resize: width, fps; fit:
  aspect, width, pad_color, fps; fill: aspect, width, fps; overlay: position, margin, scale, opacity, start, end, fade + the
  `image` reference). Nothing free-form exists: an IR operation, a plan step parameter or a compiled argument outside the
  allowlist is refused by the validator / compiler / adapter respectively.
- **Requirements → Decision:** `edit.<op>` switches an operation on, `edit.<op>.<param>` refines it; a refinement without its
  operation, a DEFAULT, or a value outside the range is refused at planning time (`EditRequirementError`). A decision is
  TRANSFORM with the requirement and the inputs' probe observations as evidence; approval comes from `video.<op>.approval`
  (DEFAULT CONFIRM; an explicit USER request for exactly this edit is its own confirmation unless a CONSTRAINT says CONFIRM;
  BLOCK is never lowered). BLOCK decisions: concat with fewer than two video inputs or with an input lacking a video stream, a
  video operation on an audio-only subject, fit and fill requested together, a missing / UNKNOWN `video-editing` capability
  (doctor failure or contract drift) or no executable tool. A BLOCK decision in force makes the plan BLOCKED whether or not a
  step cites it.
- **Multi-source timeline:** with concat, the planner joins the (trimmed) inputs in the given order into the logical subject
  `programme`; the IR operation records `inputs`, `output`, `segments[{input, track, source_range, timeline_range}]` (a
  transition overlaps consecutive clips by its duration) and `timeline_duration`. Later operations, loudness and delivery apply
  to the programme (`programme_delivery_<target>`; artifact.source lists every input). Without concat every asset is its own
  subject and the PR #18 plan is byte-identical.
- **Compiler:** trims per asset → concat → speed → resize → fit | fill → overlay → loudness → export → check. Paths under the
  job (`ops/programme_01_concat/programme.mp4`, …); the overlay image is a path reference (`<subject>_overlay_image`) resolved
  by the adapter, never a path inside arguments. Idempotency keys chain through the programme so a changed speed invalidates
  everything downstream and nothing upstream.
- **Validation / QA:** schema (video_op additionalProperties false), per-type rules (`check_video_operations`: order, one concat,
  programme only after concat, distinct inputs with video streams, factor 0.25–4 and ≠ 1, even width, aspect W:H, image PNG /
  JPEG inside `execution.allowed_inputs` without traversal, no fit + fill), every operation has a plan step naming a
  `video-editing/<op>` tool, `video-editing` capability required when any of these operations is planned. QA derives the
  expected duration from the IR (kept ranges → concat timeline → speed factor) and checks the delivered subject; output missing /
  hash mismatch / validation failure stay failures (ADR-028).
- **Not here:** autonomous editing, semantic / speaker / scene inference, captions, colour, audio mastering, thumbnails,
  motion graphics, conference-specific rules, changes to either Skill, CROP / FREEZE / REVERSE / IMAGE_INSERT / POSITION
  (unsupported by the Skill: not tools).

### audio-production-skill integration: the audio production path (implemented, ADR-030)

```
audio.production=true (+ audio.gain / channels / fade_in / fade_out / concat / sample_rate; explicit requirements)
  + the existing silence decisions (→ audio.cut) and audio.loudness decision (→ NORMALIZE)
  → Decision (TRANSFORM / KEEP / BLOCK; policy audio.<op>.approval, DEFAULT CONFIRM, explicit request waives)
  → ProductionStep (skill audio_<op>, tool audio-production/run only)
  → IR audio.operations[] (type, subject, input / inputs / output, allowlisted params, temporal scope, decision ids; concat: segments + timeline_duration)
  → compiler lower_audio_loudness / lower_audio_op (names only; wrong tool = CompileError)
  → AudioProductionAdapter (contract-checked, drift-checked; typed request on stdin; one response document; sha256 / probe / provenance verified)
  → audio-production CLI → ffmpeg-skill ≥ 0.9.1 → FFmpeg → OBSERVED probe + NORMALIZE re-measurement → QA (audio deliverable) → Artifact / provenance
```

- **Boundary:** the Skill's CLI and its contract (`audio-production skill --json`, audio-production/contract@1) are the only
  interface; the single tool `audio-production/run` carries the operation type inside the typed request. No import of the
  Skill, no command / argv / filter / executable / environment / credential; the contract's `forbidden_fields` and the agent's
  FORBIDDEN_ARG_KEYS are refused by name; every parameter is validated against the contract's parameter schema.
- **Capability:** `audio-production` AVAILABLE only with a compatible, drift-free contract and doctor ok / degraded;
  `audio-production:<TYPE>` per operation from the doctor (supported → AVAILABLE, unsupported → MISSING, unknown → AVAILABLE only
  when this resolver measured the required filters / encoders itself, else UNKNOWN). UNKNOWN is never selectable; no fallback.
- **Subjects:** an asset with audio, once `audio.production` is on, is delivered as audio (a video container's audio track is
  extracted by the Skill — `audio.extract` decision, CONFIRM by policy). A video-only asset, a video preset profile, or
  `edit.*` on the same request → BLOCK. Without the switch every existing path is byte-identical.
- **Planned operations:** cut (from the silence decisions), normalize (from the loudness decision, same IR type, tolerance
  re-measured by the Skill), gain, mono / stereo / downmix (target vs probed channels: KEEP when it already holds), fade in /
  out, concat (`programme_audio`, crossfade). Not planned though executable: MIX, SILENCE_REMOVE, NOISE_REDUCTION, DYNAMICS,
  TRIM. Not offered by the Skill: CHANNEL_MAP, standalone RESAMPLE (BLOCK without a normalisation), FORMAT_CONVERT.
- **Loudness:** measurement (media-analysis / ffmpeg-skill loudness observation) ≠ rule (target / tolerance from policy) ≠
  decision (`audio.loudness` TRANSFORM) ≠ execution (NORMALIZE re-measured against `tolerance_lufs`; the Skill's measurement
  becomes an OBSERVED `loudness` observation in provenance; QA measures the delivered file itself).
- **QA / artifacts:** `delivery_subjects` marks audio-path subjects `audio_only` (expected duration from cut / concat, expected
  channels from the planned layout); the video layer checks duration and the absence of a video stream; artifact identity /
  reuse semantics unchanged (WAV intermediates, generic delivery = last intermediate).
- **Not here:** LLM-driven audio editing, speaker identity, a QC Skill, MIX / noise reduction / dynamics planning, changes to
  either Skill, conference-specific rules.

### Phase 3 finishing Skills and the QC gate (implemented, ADR-031 / ADR-032)

```
Requirement (subtitle / thumbnail / color.* / motion.* / qc) → Decision (decision_finishing.py) → ProductionPlan step
  → IR captions / graphics / color operations, qa.qc → compiler (lower_captions / lower_thumbnail / lower_color_op / lower_graphics_render / lower_qc_check)
  → subtitle / thumbnail / color-grading / motion-graphics / qc adapters (tools/skill_process.py transport) → the Skills' CLIs → ffmpeg-skill → FFmpeg
  → QA (agent's own checks + the admitted qc report) → artifact stage (approved / candidate / working)
```

- **Subtitles** come from the transcript Observation (transcription-skill) of every source; cues are mapped through the plan's own
  trim / concat / speed onto the delivered timeline; no transcript → BLOCK; speaker never set.
- **Fixed order** per subject: trim → concat → edits → color → graphics → captions → loudness → export → check → thumbnail → qc.
- **QC gate:** qc-skill's report is admitted only when its fingerprint equals the sha256 the agent computed itself; PASS → READY
  (stage approved), WARN → candidate unless policy `qc.warn.promotion` is AUTO, FAIL → working (never final). The agent's own QA stays.
- **Not implemented:** camera switching on speaker change (needs a sync Skill, a switching operation and a `transition` → decision path).

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
