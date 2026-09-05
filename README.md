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
npx ffmpeg-skill                       # ~/.claude/skills/ffmpeg-skill — first Reference Skill（外部 OSS）
git clone https://github.com/kajisho5/media-analysis-skill ../media-analysis-skill   # 観測 Skill（VIDEO_AGENT_MEDIA_ANALYSIS_DIR で指定、任意）
git clone https://github.com/kajisho5/transcription-skill ../transcription-skill     # 認識 Skill（VIDEO_AGENT_TRANSCRIPTION_DIR で指定、任意。engine は pip install "transcription-skill[faster-whisper]"）
git clone --branch claude/video-editing-skill-sd9vgt https://github.com/kajisho5/video-editing-skill ../video-editing-skill   # 編集 Skill（VIDEO_AGENT_VIDEO_EDITING_DIR で指定、任意。PR #1 マージ後は main）
export VIDEO_AGENT_FFMPEG_SKILL_DIR=/path/to/ffmpeg-skill
export VIDEO_AGENT_WORKSPACE=./video-agent-work   # 省略可
video-agent doctor
```

## 使い方

```bash
video-agent skills                                      # Skill package（ffmpeg-skill / media-analysis / transcription）と production skill の状態、選択された tool
video-agent transcribe input.mp4 --language ja --offline   # transcription-skill で認識 → transcript Observation + SpeechEvent（認識まで。話者・字幕・編集は判断しない）
video-agent plan input.mp4 --profile youtube --kind transcript --language ja   # transcript / SPEECH → speech inference → 発話間の長い無音を削除候補（CONFIRM）として plan に載せる
video-agent explain <ir.json> --step step_trim_<asset>   # decision → inference → SpeechEvent / silence event → observation の証拠 chain
video-agent context <ir.json> --at 10                    # その時刻の Production Context（同時に成立している Event 種別と参照）
video-agent explain <ir.json> --context <ctx_id>         # context → tracks → events → observations、それに乗る inference / decision
video-agent explain <ir.json> --observation <obs_id | tr_id>   # observation → skill → tool → engine → model → transcript → asset → events
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

video-editing-skill 連携（agent → video-editing CLI（contract / stdin request / `--workspace` `--allowed-input`）→ ffmpeg-skill → FFmpeg。agent は command / argv / filter を作らず、Skill の response（sha256 / timeline / OBSERVED probe / commands）を Artifact・provenance に記録する。silence_cleanup の候補 `video-editing/cut` は宣言順 2 番目、両方利用可能なら従来通り ffmpeg-skill/cut）: ADR-028 / MASTER_SPEC「video-editing-skill integration」。編集操作 concat / speed / resize / fit / fill / overlay（`--set edit.concat=true edit.speed=2 edit.resize=640 edit.fit=1:1 edit.overlay=logo.png` のような明示 requirement → Decision → Plan → IR `video.*` operation → compiler → video-editing-skill。concat は trimmed 入力を入力順に結合した `programme` を subject にし、以降の操作・loudness・delivery は programme に適用。値域外・曖昧・矛盾（fit+fill）・video stream 無し・capability 欠落は拒否 / BLOCK）: ADR-029 / MASTER_SPEC「video-editing-skill operations」。Production Decision Engine（Inference = 何が起きているか / Decision = 制作として何をすべきか / Plan = どう実行するか。decision は evidence 必須・type 語彙・policy 解決の provenance と basis を保持、`video-agent explain --decision <id|subject>` で policy / preference / constraint / intent / evidence / plan step まで辿れる）: ADR-027 / MASTER_SPEC「Production Decision Engine」。Production Plan（Decision / Event → plan → IR の決定論的橋渡し、`video-agent explain --step`）: ADR-021 / MASTER_SPEC §21。Artifact / Delivery / Archive（`video-agent artifacts` / `artifact` / `deliver` / `archive` / `explain --artifact`）: ADR-022 / MASTER_SPEC §36。AI Provider contract（AI は提案のみ、execution authority ではない）: ADR-018 / MASTER_SPEC §42。Revision workflow: `docs/revision.md`。Skill / Capability / Tool と将来 Skill の追加手順: `docs/skills.md`、Gap 分析: `docs/GAP_ANALYSIS_PHASE2.md`。

Policy / Preference / Constraint（`policy/rules.py`）: **Policy** は運用上の規則（例 `silence.leading.approval`）、**Preference** は好み（例 conference の `-16 LUFS`、request の `--set`）、**Constraint** は上書き不可の制約（例 conference の `silence.internal.approval=CONFIRM`、system の preserve_source）。precedence は GLOBAL → … → PROFILE → REQUEST、Constraint に反する下位規則は conflict として `policy.<key>` decision（CONFIRM）になる。decision の `basis` に消費した設定の kind / value / provenance（USER / PROFILE / SYSTEM / DEFAULT）が残る。

監査結果と既知の制限: `docs/AUDIT_PHASE1.md`、最終レビューと Phase 2 開始条件: `docs/PHASE1_FINAL_REVIEW.md`。

## テスト

```bash
python3 -m unittest tests/test_unit.py          # ffmpeg 不要（FakeAdapter）
python3 -m unittest tests/test_integration.py   # ffmpeg + ffmpeg-skill 必須（合成素材で実レンダー、契約テスト）
python3 evals/run.py                            # 期待 intent / decision / provenance の評価
```

## レイアウト

`src/video_agent/{models,temporal,capabilities,tools/ffmpeg_skill,media,policy,profiles,skills,agent,project,execution,qa,jobs,audit,providers}`、`schemas/project.schema.json`、`profiles/*.json`、`tests/`、`evals/`。
