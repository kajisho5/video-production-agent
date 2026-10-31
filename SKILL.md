---
name: video-production-agent
description: Plan and produce a video from source files - trim silence, normalize loudness, subtitles, color, thumbnails, deliver for YouTube / conference / broadcast - with you (the chat model) as the reasoning step and every action validated, approved and recorded. Use when asked to edit, finish, clean up, prepare or deliver a video or recording, or to explain why the plan decided something. Runs locally through `video-agent`; no API key needed.
---

# video-production-agent

You are the brain; `video-agent` measures, validates, executes and records. Run it with `--json` and read the JSON. Never call ffmpeg or a skill directly for a production: go through the plan so every step has evidence and approval.

## The production loop

1. **Plan with a brief**
   `video-agent --json plan <files...> --profile <youtube|conference|broadcast|generic> --request "<the user's words>" --ai-brief --ai-model claude-code`
   Note `project` (the plan path), `decisions` and `ai_brief`.
2. **Answer the brief** (this is your reasoning). Read `ai_brief.inputs` (measured facts, each with an id), `ai_brief.context.allowed_intents` and `ai_brief.context.request_text`, follow `ai_brief.rules`, and write `answer.json`:
   `{"recommendations": [{"intent": "silence_cleanup", "asset_id": "<id from inputs.assets>", "statement": "<why, from the facts>", "confidence": 0.8, "evidence": ["<observation or event id>"], "params": {}}]}`
   Only allowed intents, only ids that appear in the inputs, no commands / paths / code. An empty list is a valid answer.
3. **Record the answer**
   `video-agent --json revise <project> --ai-recommendations answer.json --ai-model claude-code`
   Report `ai.accepted` and `ai.dropped` honestly; a dropped recommendation was not used.
4. **Show the plan and ask** — list the CONFIRM decisions (`video-agent --json explain <project>` gives reason and evidence). Approve only what the user approves, in their words:
   `video-agent --json approve <project> --decision <ids|all> --by "<user>"`; to refuse: `reject <project> --decision <id> --reason "..."` then `revise <project>`.
5. **Render and check**
   `video-agent --json render <project> --dry-run` first if the user wants to see the steps, then `video-agent --json render <project>`, then `video-agent --json check <output> --platform <profile>`.

Explicit requirements go in as `--set key=value` on plan or revise (e.g. `audio.loudness.target_lufs=-16`, `edit.trim_leading_silence=true`).

**A brief in the user's words** ("60 s vertical, captions, -14 LUFS") goes in as `--brief "<the user's words>"` on plan. With `--ai-brief` the result's `brief.host_request` is yours to answer: keys from `host_request.context.allowed_keys` only, each with the exact `phrase` of the brief it comes from, values as `--set` takes them, and what no key expresses (a target length) under `unmapped`. Write `brief.json` (`{"requirements": [{"key": "edit.fit", "value": "9:16", "phrase": "vertical", "confidence": 0.9}], "unmapped": [{"phrase": "60 s", "reason": "no key for a target length"}]}`) and plan again with `--brief "<same words>" --brief-requirements brief.json --ai-model claude-code` (`video-agent --json brief "<words>" --brief-requirements brief.json` checks it without analysing media). Report `brief.accepted`, `brief.refused` (with reasons), `brief.overridden` (explicit `--set` wins) and `brief.unmapped` honestly; the `requirements.brief` decision and every change resting on your keys are CONFIRM for the user.

## When a skill is missing (handoff)

Every skill is optional; the plan uses the ones installed here. `plan --json` lists steps no installed skill can run under `handoff` (job, reason, which package provides it, how to install it); those steps are BLOCKED. Tell the user, and either install the skill (`/plugin install <skill>@ai-video-production-os`, or `./install.sh`) and plan again, or — only if the user agrees and it is safe — do that job with another installed skill or a standard tool. Say plainly that such a step is outside the agent: no contract, no provenance, no approval record, no QA by the agent. Then verify its output with `video-agent --json check <file>`.

A job the agent has no adapter for at all (another package of the ecosystem owns it) is asked for with `--set job.<skill>=true`: `dubbing`, `sound_design`, `speaker_diarization`, `visual_detection`, `slide_sync`, `platform_publish`, `still_image_edit`, `render_3d`, `measured_motion_graphics`, `html_scene_render`, `document_verification`, `bluray_authoring`, `dvd_authoring`, `disc_image`, `dvd_line21_captions`. The plan then carries a BLOCKED step for it and a handoff naming the package, its repository and its install command (`none yet` for a package that has not been released). Use it when the user asks for such a job, so the request is recorded and handed over instead of silently dropped.

Speech enhancement through voice-enhance-skill (ADR-073) is opt-in on the audio path: `--set audio.production=true` plus `voice.isolate=true` (AI voice isolation, 48 kHz input), `voice.dereverb=true` and / or `voice.bleed_reduce=true` (with `audio.mix=true`, one mixed input per microphone). Every one is EXPERIMENTAL, so it always waits for confirmation; a missing engine blocks with voice-enhance-skill's own code and install line (nothing is downloaded).

An overlay image is prepared by imagemagick-skill (ADR-077) only when the user gives its size: `--set edit.overlay=logo.png edit.overlay.prepare.width=320 edit.overlay.prepare.height=180` (both; `edit.overlay.prepare.mode=fit|fill|exact`, default fit), `edit.overlay.prepare.pad=none|<colour>` (+ `.gravity`) to pad a fit to exactly that canvas, `edit.overlay.prepare.strip=true` to remove GPS / EXIF first. Never choose a size or a colour yourself; ask. The user's file is never written; without imagemagick-skill the step is handed off (`npx imagemagick-skill`).

## Offloading a camera card (ingest-skill)

A card or recorder folder is planned on its own: `video-agent --json plan /Volumes/CARD_A --set ingest.dest=/Volumes/SSD1/Footage --set ingest.dest=/Volumes/SSD2/Footage --set ingest.project=2026-10-30_event` (optional `ingest.hash=xxh64`, `ingest.name_by=reel`, `ingest.reel=A001`). Name every destination the user gave you; never pick one yourself (each must be a mounted directory). The plan copies nothing: it shows ingest-skill's dry run and the card's inventory and one `ingest.offload` decision that is always CONFIRM. Ask the user, then `video-agent approve <plan> --decision <id>` and `video-agent render <plan>`. Report from `execution.offload`: say "verified" only when the status is COMPLETED; otherwise name the error `kind` (CONFLICT, PARTIAL, COPY_FAILED, HISTORY_DAMAGED, ...) and each destination from `details.destinations` (a PARTIAL keeps the complete drives listed under `details.complete`). Warn when the plan says two destinations are on one volume. The verified copies become assets with `ingest_clip` observations; plan the edit on those files next.

## Exit codes

0 done / 2 validation errors / 3 BLOCKED / 4 waiting for approval / 5 QA FAIL (revise: no new version) / 1 other. Exit 4 means ask the user, not retry with approval.

## Report it honestly

- "Done" only when render completed and QA passed; quote QA WARN / FAIL items.
- A measurement that is missing is **unknown**, not zero (e.g. loudness not measured).
- Say which decisions you recommended (AI_GENERATED) and that policy, not you, set their risk and approval.
- `video-agent doctor --json` when a skill is MISSING / DEGRADED: name it, do not work around it.

## MCP

The same tools are served over MCP by `video-agent-mcp` (Claude Desktop / Claude Code: see README). With MCP, pass the answer object directly as `ai_recommendations` to the `revise` tool, and a brief answer as `brief_requirements` to `plan` (or the `brief` tool).
