# video-production-agent — Claude Code Instructions

## Role

Act as the Principal Engineer, Software Architect, AI Agent Architect, and Video Production Systems Architect for this project.

## Mandatory first step

Before writing substantial code:

1. Read `docs/MASTER_SPEC.md` completely.
2. Thoroughly inspect the existing `kajisho5/ffmpeg-skill` repository and its actual implementation.
3. Produce the architecture review requested by the master specification.
4. Do not perform a large implementation before that review is complete.

Do not guess when the repository or code can be inspected.

## Core boundary

`ffmpeg-skill` is the Media Processing Engine / Hands.

`video-production-agent` is the AI Production Brain / Orchestrator.

Do not copy or unnecessarily reimplement ffmpeg-skill.

The Agent must not directly generate arbitrary FFmpeg shell commands. Use a validated Project IR, compiler, and tool adapter boundary.

## Engineering principles

- Preserve original media.
- Prefer deterministic execution after Project IR generation.
- Separate observed facts from AI inference.
- Separate user requirements from AI suggestions.
- Make important decisions explainable.
- Use AUTO / CONFIRM / BLOCK appropriately.
- Validate before execution and QA after execution.
- Recovery must be finite and safe.
- Keep secrets out of source code and logs.
- Do not claim unfinished features are complete.
- Avoid premature, oversized Web UI work.
- Prefer simple current implementations with clean future extension points.
- Use tests, real-media tests where appropriate, and Evals.
- Record architecture decisions when they materially affect the system.

## Working style

Inspect existing code and available tooling yourself before asking the user for information that can be determined by inspection.

If a requirement conflicts with the actual ffmpeg-skill interface, document the discrepancy and design an adapter rather than silently changing either project.

When uncertain about a major architectural decision, explain the trade-off before making an irreversible choice.

## First deliverable

Your first substantial response should contain:

1. ffmpeg-skill capability inventory
2. reusable components
3. components that must not be duplicated
4. responsibility boundary
5. proposed architecture
6. data flow
7. Temporal/Event/Session model
8. Skill/Capability/Tool model
9. Decision/Policy/Preference/Constraint model
10. Project IR proposal
11. Compiler/Adapter design
12. QA and Recovery design
13. Artifact/Job/Lifecycle design
14. Feedback/Revision design
15. Conference profile design
16. Phase roadmap
17. technical risks
18. unresolved decisions
19. minimum Phase 1 implementation scope

Do not start by generating a large amount of code.
