# Roadmap — video-production-agent

Part of the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS/blob/main/docs/ecosystem/ROADMAP.md) ecosystem: department **司令塔 (Command)**, status `active`.
Written from a read-only audit of this repository on 2026-10-30. Each step below is meant to be one pull request; tick it off by linking the PR.

## Where it stands

Python orchestrator (`video-agent`). 10 adapters (ffmpeg-skill, video-editing, audio-production, color-grading, motion-graphics, subtitle, thumbnail, qc, media-analysis, transcription). Brain: deterministic rules + optional AI (Claude API, ADR-049; the chat model through MCP, ADR-050). Handoffs for steps no installed skill can run (ADR-051). 405 unit + 53 integration tests.

## What this repository owns

`ecosystem.json` gives each job to exactly one repository. Work outside this list belongs to its owner and is handed over, not re-implemented.

| Owned job | Status | Note |
|---|---|---|
| 制作計画 | done | plan / revise / approve / render |
| スキルの呼び分け | partial | 22 of 33 ecosystem components have no adapter or SkillSpec, so a handoff cannot even name them |
| 制作記録(ProductionReceipt) | done | ADR-040 |

## Wiring

Uses 17 of ffmpeg-skill's 42 tools directly through a hand-maintained catalog (19 entries; fit, report unreachable; scenes for `--beats` only, ADR-053). Reads ffmpeg-skill's `_contract.py --json --static` and `doctor --json` at locate time to check the catalog (ADR-056). NLE interchange (OTIO / FCPXML / EDL) through ffmpeg-skill `render --export-timeline` (ADR-052).

## Steps

- [ ] **1. Declare SkillSpecs for every ecosystem job** — phase-2 SkillSpecs (not implemented) for ingest, voice-enhance, dubbing, sound-design, diarization, detection, slide-sync, publish, imagemagick, blender, remotion, hyperframes, artifacts, disc authoring, so ADR-051 handoffs name the right package
  - Done when: a plan asking for each job shows a handoff naming the package and its install command
- [x] **2. Version gates follow the satellites' version fixes** — widen SUPPORTED_SKILL_VERSIONS and re-pin contracts as motion-graphics (0.2), subtitle, thumbnail, transcription (0.7), media-analysis fix their reported versions ([#65](https://github.com/kajisho5/video-production-agent/pull/65))
  - Done when: integration suite green against each fixed release
- [x] **3. Contract-backed ffmpeg-skill catalog check** — at locate time read `_contract.py --json --static`, assert contract_version 1.0 and every catalogued flag in input_schema; surface doctor's missing capabilities ([#75](https://github.com/kajisho5/video-production-agent/pull/75))
  - Done when: a renamed flag makes the tool unavailable with a reason instead of an argparse error
  - Done (ADR-056): contract_version 1.x and every catalogued flag / switch arity / positional checked per tool at locate time; doctor's `usable: no` rows surfaced with their fix; an off tool is capability `ffmpeg-skill:<tool>` MISSING with the reason in the selection reason / plan handoff, and the adapter refuses it before argparse
- [x] **4. Timeline export (OTIO first)** — SkillSpec `timeline_export` -> ffmpeg-skill/render `--export-timeline` (.otio, .fcpxml, .edl) from the IR cut list, plus a sidecar SRT; ADR amending ADR-002
  - Done when: `--set delivery.timeline=otio` produces a timeline Artifact that `opentimelineio` reads back with the same cuts. Neither NLE is available yet (owner, 2026-10-30): the Resolve / Premiere import check is recorded when they are
  - Done (ADR-052): `delivery.timeline=otio|fcpxml|edl` → TIMELINE Artifact; opentimelineio 0.18.1 reads the .otio back with the IR's keep ranges (within one frame) in `tests/test_integration.py::TimelineExportRealTests`. No separate sidecar: `subtitle=srt` already lands on the same timeline when no speed / transition is applied; embedding captions in the timeline is a follow-up
- [ ] **5. NL brief -> requirements (requirements_extraction)** — AI task that returns only known `--set` keys, through the host (MCP) or Claude provider; unknown keys refused
  - Done when: "60 s vertical, captions, -14 LUFS" yields valid keys; no provider = deterministic
- [ ] **6. Wire what the satellites already offer** — color gamma/lift/gain/levels/curves; all 12 motion-graphics elements; subtitle mux; audio MIX / NOISE_REDUCTION / DYNAMICS; qc delivery_package gate; video-editing trim/rotate
  - Done when: a plan step per new parameter / element, tests
- [ ] **7. cropdetect -> crop decision** — AnalysisKind crop_rectangle and a CONFIRM decision proposing video_crop
  - Done when: letterboxed fixture proposes the measured rectangle
- [ ] **8. Speaker ids end to end** — accept diarization output; remove the null-only speaker_id invariants behind an explicit requirement
  - Done when: subtitles and edits carry speaker ids from a diarized transcript
- [ ] **9. Adapters for the new skills, as they ship** — ingest (media_offload), detection (redact faces), dubbing, voice-enhance, sound-design (via audio-production MIX), publish (CONFIRM after QC), nle-bridge
  - Done when: one PR per skill: plan -> run -> provenance, handoff when absent
- [x] **10. Take selection and beat-synced assembly** — group takes of the same line by transcript similarity (transcription-skill); score each take on measured facts (sharpness / exposure / loudness from media-analysis, flubs and fillers from the transcript, faces from detection) and let the AI pick with the evidence recorded as a decision; assemble the picks with cuts snapped to a measured beat grid (ffmpeg-skill `scenes --beats`, `cut --snap beats`) and audio crossfades at the joins (audio-production `concat` crossfade); export OTIO for finishing in an NLE
  - Done when: an eval with several takes per line picks the expected take, every pick carries its scores, and the cuts land on measured beats
  - Done (ADR-053, ADR-054): grouping by transcript similarity, scores from measured facts with their evidence, `takes.pick` per line (an AI `take_selection` recommendation may override, scores kept), concat of the picks in line order, out points snapped to the music bed's measured beat grid (`scenes --beats`, computed in programme time, so `cut --snap beats` is not used), audio crossfades at the joins (default 10 ms, `edit.takes.crossfade`: audio handles at each join, audio-production `CONCAT` crossfade, muxed under the unchanged hard-cut picture by ffmpeg-skill `audio.py --replace`; a join without handle room keeps a hard cut with the reason; A/V sync checked by the validator and on real media), OTIO export of the cut list; evals 103-112
  - Needs other repos: sharpness / exposure scores (media-analysis-skill has no tool that measures them), face scores (detection-skill, wired by step 9), beat measurement of an audio-only bed (ffmpeg-skill `scenes.py` refuses an input without a video stream). Follow-up here, outside this step's text: the music bed in the mix / timeline

## Boundaries

URL download (yt-dlp, `agent/ingest.py`) belongs to ingest-skill since 2026-10-30 (owner decision, URL素材取得). The agent keeps its call until ingest-skill's `fetch` ships, then hands over (ADR-051 handoff when ingest-skill is missing).

---
Engine for every media operation: [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (42 tools, treated as complete). Orchestrator: [video-production-agent](https://github.com/kajisho5/video-production-agent).
