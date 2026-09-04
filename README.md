# video-production-agent

AI Video Production Orchestrator. `kajisho5/ffmpeg-skill` を **Media Processing Engine (Hands)** として使い、本リポジトリは **Production Brain (Orchestrator)** を担当する。

- 仕様: `docs/MASTER_SPEC.md`、開発ルール: `CLAUDE.md`
- ffmpeg-skill 調査と設計レビュー: `docs/ARCHITECTURE_REVIEW.md`
- Project IR: `docs/project-ir.md`、ADR: `docs/decisions.md`

## Phase 1 の範囲（実装済み）

Request → Requirements → Intent → Analysis(probe / silence / loudness) → Inference → Decision → Plan → Project IR → Validation → Execute(cut / loudness / export via ffmpeg-skill) → QA(probe / loudness / check / look) → Report / Provenance。

未実装（設計のみ）: 自然言語理解(AI)、conference パイプライン本体、multicam / captions / scenes、Web UI、job queue。

## セットアップ

```bash
pip install -e .
# ffmpeg-skill を用意（どちらか）
npx ffmpeg-skill                       # ~/.claude/skills/ffmpeg-skill — first Reference Skill（外部 OSS、実装済みの唯一の Skill package）
export VIDEO_AGENT_FFMPEG_SKILL_DIR=/path/to/ffmpeg-skill
export VIDEO_AGENT_WORKSPACE=./video-agent-work   # 省略可
video-agent doctor
```

## 使い方

```bash
video-agent skills                                      # Skill package（実装済みは ffmpeg-skill のみ）と production skill の状態、選択された tool
video-agent analyze input.mp4
video-agent plan input.mp4 --profile youtube            # → <workspace>/plans/input.youtube.project.json
video-agent plan input.mp4 --profile conference --set audio.loudness.target_lufs=-18
video-agent validate <workspace>/plans/input.youtube.project.json
video-agent render <workspace>/plans/input.youtube.project.json --dry-run
video-agent render <workspace>/plans/input.youtube.project.json            # CONFIRM が残れば WAITING_FOR_APPROVAL (exit 4)
video-agent render <workspace>/plans/input.youtube.project.json --approve all
video-agent render <workspace>/plans/input.youtube.project.json --resume last   # 途中失敗後: 完了済み操作を再利用
video-agent reject  <project> --decision <id> --reason "..."       # 却下（理由必須）→ render は BLOCKED
video-agent revise  <project> [--set audio.loudness.target_lufs=-16]  # Plan v2 を生成（v1 は .v1.json に保存）、PlanDiff を表示
video-agent approve <project> --decision all                       # v2 を承認して render 可能に
video-agent diff    <project>                                      # 前版との PlanDiff
video-agent explain <workspace>/plans/input.youtube.project.json [--decision <id>]
video-agent check output.mp4 --platform youtube
```

終了コード: 0 完了 / 2 検証エラー / 3 BLOCKED / 4 承認待ち / 5 QA FAIL (REVIEW) / 130 中断 (CANCELLED) / 1 その他。revise: 5 = 新版なし。

Production Plan（Decision / Event → plan → IR の決定論的橋渡し、`video-agent explain --step`）: ADR-021 / MASTER_SPEC §21。Artifact / Delivery / Archive（`video-agent artifacts` / `artifact` / `deliver` / `archive` / `explain --artifact`）: ADR-022 / MASTER_SPEC §36。AI Provider contract（AI は提案のみ、execution authority ではない）: ADR-018 / MASTER_SPEC §42。Revision workflow: `docs/revision.md`。Skill / Capability / Tool と将来 Skill の追加手順: `docs/skills.md`、Gap 分析: `docs/GAP_ANALYSIS_PHASE2.md`。

監査結果と既知の制限: `docs/AUDIT_PHASE1.md`、最終レビューと Phase 2 開始条件: `docs/PHASE1_FINAL_REVIEW.md`。

## テスト

```bash
python3 -m unittest tests/test_unit.py          # ffmpeg 不要（FakeAdapter）
python3 -m unittest tests/test_integration.py   # ffmpeg + ffmpeg-skill 必須（合成素材で実レンダー、契約テスト）
python3 evals/run.py                            # 期待 intent / decision / provenance の評価
```

## レイアウト

`src/video_agent/{models,temporal,capabilities,tools/ffmpeg_skill,media,policy,profiles,skills,agent,project,execution,qa,jobs,audit,providers}`、`schemas/project.schema.json`、`profiles/*.json`、`tests/`、`evals/`。
