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

## Exit codes

0 done / 2 validation errors / 3 BLOCKED / 4 waiting for approval / 5 QA FAIL (revise: no new version) / 1 other. Exit 4 means ask the user, not retry with approval.

## Report it honestly

- "Done" only when render completed and QA passed; quote QA WARN / FAIL items.
- A measurement that is missing is **unknown**, not zero (e.g. loudness not measured).
- Say which decisions you recommended (AI_GENERATED) and that policy, not you, set their risk and approval.
- `video-agent doctor --json` when a skill is MISSING / DEGRADED: name it, do not work around it.

## MCP

The same tools are served over MCP by `video-agent-mcp` (Claude Desktop / Claude Code: see README). With MCP, pass the answer object directly as `ai_recommendations` to the `revise` tool.
