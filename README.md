<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/app-icon-dark.png">
    <img src="brand/app-icon.png" width="128" alt="video-production-agent app icon">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/logo/video-production-agent-wordmark-white.svg">
    <img src="brand/logo/video-production-agent-wordmark-black.svg" height="48" alt="video-production-agent">
  </picture>
</p>

<p align="center"><b>Plans, approves, renders and records a production</b></p>

AI Video Production Orchestrator. `kajisho5/ffmpeg-skill` を **Media Processing Engine (Hands)** として使い、本リポジトリは **Production Brain (Orchestrator)** を担当する。

- 仕様: `docs/MASTER_SPEC.md`、開発ルール: `CLAUDE.md`
- ffmpeg-skill 調査と設計レビュー: `docs/ARCHITECTURE_REVIEW.md`
- Project IR: `docs/project-ir.md`、ADR: `docs/decisions.md`

## Phase 1 の範囲（実装済み）

Request → Requirements → Intent → Analysis(probe / silence / loudness) → Inference → Decision → Plan → Project IR → Validation → Execute(cut / loudness / export via ffmpeg-skill) → QA(probe / loudness / check / look) → Report / Provenance。

未実装（設計のみ）: 依頼文から requirement を作る自然言語理解（`requirements_extraction`）、conference パイプライン本体、Web UI、job queue。

AI（Claude）: `pip install -e '.[ai]'` の上で `VIDEO_AGENT_AI_PROVIDER=anthropic`（資格情報は `ANTHROPIC_API_KEY` か `ant auth login`。モデルは `VIDEO_AGENT_AI_MODEL`、既定 `claude-opus-5-5`）。AI は計測結果と依頼文から制作の推薦を出すだけで、引用の無い推薦は捨てられ、実行の可否は policy と検証が決める（ADR-018 / ADR-049）。未設定なら完全に決定的なパイプラインのまま動く。

チャットの Claude を脳にする（API キー不要、ADR-050）: Claude Code / Claude Desktop など、いま話しているチャットのモデルが推薦を出す。手順は2段で、`plan --ai-brief` が AI に渡す材料（計測結果・引用可能な id・許可された intent・依頼文・規則）を出し、チャットのモデルがそれを読んで JSON で答え、`revise <project> --ai-recommendations answer.json --ai-model claude-code` がその答えを ADR-049 と同じ検証に通して新版にする（受理・却下の内訳を表示）。

- Claude Code: OS の installer が `SKILL.md` を `~/.claude/skills/video-agent` にリンクするので、「この動画を YouTube 用にして」と頼めば上の2段を CLI で回す。MCP で使うなら `claude mcp add video-agent -- video-agent-mcp --workspace ~/video-agent-work`。
- Claude Desktop: `claude_desktop_config.json` に MCP server を追加（`video-agent-mcp` は `pip install -e .` で入る。パスは `which video-agent-mcp` の絶対パス）:

```json
{"mcpServers": {"video-agent": {"command": "/abs/path/to/video-agent-mcp", "args": ["--workspace", "/abs/path/to/video-agent-work"],
  "env": {"VIDEO_AGENT_FFMPEG_SKILL_DIR": "/abs/path/to/ffmpeg-skill"}}}}
```

  tools: `doctor` `skills` `plan`（既定で ai_brief 付き）`revise`（`ai_recommendations` を直接渡す）`validate` `explain` `diff` `approve` `reject` `render` `check` `artifacts`。各 tool は `video-agent --json <command>` を子プロセスで実行するだけ。承認は `approve` を別に呼ぶ必要があり、`render` に一括承認は無い。
- スマホ / Web の Claude アプリ: ローカルの MCP server（stdio）は動かせない。動画ファイルと ffmpeg があるマシンで Claude Code か Claude Desktop を使う。

## セットアップ

```bash
pip install -e .
# ffmpeg-skill を用意（どちらか）
npx ffmpeg-skill                       # ~/.claude/skills/ffmpeg-skill — first Reference Skill（外部 OSS）
git clone https://github.com/kajisho5/media-analysis-skill ../media-analysis-skill   # 観測 Skill（VIDEO_AGENT_MEDIA_ANALYSIS_DIR で指定、任意）
git clone https://github.com/kajisho5/transcription-skill ../transcription-skill     # 認識 Skill（VIDEO_AGENT_TRANSCRIPTION_DIR で指定、任意。engine は pip install "transcription-skill[faster-whisper]"）
git clone --branch claude/video-editing-skill-sd9vgt https://github.com/kajisho5/video-editing-skill ../video-editing-skill   # 編集 Skill（VIDEO_AGENT_VIDEO_EDITING_DIR で指定、任意。PR #1 マージ後は main）
git clone https://github.com/kajisho5/audio-production-skill ../audio-production-skill   # 音声制作 Skill（VIDEO_AGENT_AUDIO_PRODUCTION_DIR で指定、任意。ffmpeg-skill 0.9.1 以上が必要）
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

video-editing-skill 連携（agent → video-editing CLI（contract / stdin request / `--workspace` `--allowed-input`）→ ffmpeg-skill → FFmpeg。agent は command / argv / filter を作らず、Skill の response（sha256 / timeline / OBSERVED probe / commands）を Artifact・provenance に記録する。silence_cleanup の候補 `video-editing/cut` は宣言順 2 番目、両方利用可能なら従来通り ffmpeg-skill/cut）: ADR-028 / MASTER_SPEC「video-editing-skill integration」。編集操作 concat / speed / resize / fit / fill / overlay（`--set edit.concat=true edit.speed=2 edit.resize=640 edit.fit=1:1 edit.overlay=logo.png` のような明示 requirement → Decision → Plan → IR `video.*` operation → compiler → video-editing-skill。concat は trimmed 入力を入力順に結合した `programme` を subject にし、以降の操作・loudness・delivery は programme に適用。値域外・曖昧・矛盾（fit+fill）・video stream 無し・capability 欠落は拒否 / BLOCK）: ADR-029 / MASTER_SPEC「video-editing-skill operations」。audio-production-skill 連携（`--set audio.production=true` で asset の音声を subject にする audio path: 既存の無音 decision → `audio.cut`、loudness decision → NORMALIZE（Skill が再測定）、`audio.gain` / `audio.channels=mono|stereo` / `audio.fade_in` / `audio.fade_out` / `audio.concat`（+`.crossfade`）/ `audio.sample_rate` は明示 requirement → Decision → Plan → IR `audio.*` → compiler → audio-production-skill。video container は音声のみ納品（`audio.extract` decision、CONFIRM。`audio.production=true` だけでは waive されず、`--set audio.extract=true` か decision の明示承認が必要: ADR-033）、video 編集との同時要求・video preset・audio 無し・実現不能な layout・単独 resample は BLOCK）: ADR-030 / MASTER_SPEC「audio-production-skill integration」。複数ソース同期の観測（`video-agent analyze camA.mp4 recorder.wav --kind sync`: 先頭入力を reference として ffmpeg-skill/sync が測った offset / confidence / drift を `sync` Observation（OBSERVED）として保持し、信頼度が `sync.min_confidence`（既定 0.3）以上なら target の TimelineMap に反映。Decision は生成しない）: ADR-035 / MASTER_SPEC「Multi-source sync observation」。Production Decision Engine（Inference = 何が起きているか / Decision = 制作として何をすべきか / Plan = どう実行するか。decision は evidence 必須・type 語彙・policy 解決の provenance と basis を保持、`video-agent explain --decision <id|subject>` で policy / preference / constraint / intent / evidence / plan step まで辿れる）: ADR-027 / MASTER_SPEC「Production Decision Engine」。Production Plan（Decision / Event → plan → IR の決定論的橋渡し、`video-agent explain --step`）: ADR-021 / MASTER_SPEC §21。Artifact / Delivery / Archive（`video-agent artifacts` / `artifact` / `deliver` / `archive` / `explain --artifact`）: ADR-022 / MASTER_SPEC §36。AI Provider contract（AI は提案のみ、execution authority ではない）: ADR-018 / MASTER_SPEC §42。Revision workflow: `docs/revision.md`。Skill / Capability / Tool と将来 Skill の追加手順: `docs/skills.md`、Gap 分析: `docs/GAP_ANALYSIS_PHASE2.md`。

Policy / Preference / Constraint（`policy/rules.py`）: **Policy** は運用上の規則（例 `silence.leading.approval`）、**Preference** は好み（例 conference の `-16 LUFS`、request の `--set`）、**Constraint** は上書き不可の制約（例 conference の `silence.internal.approval=CONFIRM`、system の preserve_source）。precedence は GLOBAL → … → PROFILE → REQUEST、Constraint に反する下位規則は conflict として `policy.<key>` decision（CONFIRM）になる。decision の `basis` に消費した設定の kind / value / provenance（USER / PROFILE / SYSTEM / DEFAULT）が残る。

監査結果と既知の制限: `docs/AUDIT_PHASE1.md`、最終レビューと Phase 2 開始条件: `docs/PHASE1_FINAL_REVIEW.md`。

## 統合パイプライン（Phase 3、ADR-031 / ADR-032）

subtitle / thumbnail / color-grading / motion-graphics / qc の各 Skill は checkout を環境変数で指す（`VIDEO_AGENT_SUBTITLE_DIR` `VIDEO_AGENT_THUMBNAIL_DIR` `VIDEO_AGENT_COLOR_GRADING_DIR` `VIDEO_AGENT_MOTION_GRAPHICS_DIR` `VIDEO_AGENT_QC_DIR`。thumbnail-skill は `pip install Pillow`）。要求は明示 `--set` のみ。例（講演 3 本をまとめて、無音を整理し、字幕を付け、YouTube 用を作って QC まで）:

```
video-agent plan talk1.mp4 talk2.mp4 talk3.mp4 --profile youtube --kind transcript --language ja \
  --set edit.concat=true --set subtitle=true --set subtitle.burn_in=true --set thumbnail=true --set qc=true
video-agent render <project.json> --approve all
video-agent explain <project.json> --pipeline
video-agent check <artifact.mp4> --platform youtube --qc
```

編集ソフトへの受け渡し（ADR-052）: `--set delivery.timeline=otio`（または `fcpxml` / `edl`、既定 `none`）で、IR の cut list（trim が残す範囲を programme の順に並べたもの、hard cut、source は元の場所を参照）を ffmpeg-skill `render.py --export-timeline` が書き出し、type `TIMELINE` の Artifact になる。speed / colour / graphics などはタイムラインに載らず、decision に名指しされる。switch / grid / audio path は BLOCK。

テイク選び（ADR-053）: `--set edit.takes=true` は、同じセリフを撮った複数テイクを transcript の類似（既定 0.6、`edit.takes.similarity`）でセリフごとにまとめる。各テイクは測った事実だけで採点する。transcript からはつなぎ言葉・言い直し・欠け、音声測定からはクリップ・音量・間を見る。sharpness / exposure / 顔は測る adapter が無いので採点しない。セリフごとに最良のテイクを `takes.pick`（得点と根拠の測定つき、CONFIRM）として選び、セリフ順に concat する。`--set edit.takes.music=<入力 index>` を付けると、その入力のビートを ffmpeg-skill `scenes --beats` で測り、つなぎ目（各 pick の out 点）を programme 時刻で最も近い beat に合わせる（既定で最大 0.5 秒、`edit.takes.snap_max_shift`）。AI 推薦（`take_selection`）は pick を差し替えられるが、得点は変えない。`delivery.timeline=otio` はこの cut list をそのまま書き出す。つなぎ目の音声クロスフェードと、音楽のミックスはまだ無い。

生成される ProductionPlan（依存関係付き）: trim ×3 → concat → captions.generate（transcript の cue を delivered timeline に写像）→ captions.burn → loudness → export → check → thumbnail → qc ×2（deliverable と sidecar）。QC PASS → artifact は `approved`（READY）、WARN → candidate（policy `qc.warn.promotion`、既定 CONFIRM）、FAIL → `working`（final 昇格不可）。「話者が変わったらカメラを切り替える」は未実装（docs/GAP_ANALYSIS_PHASE2.md §22）。

## テスト

```bash
python3 -m unittest tests/test_unit.py          # ffmpeg 不要（FakeAdapter）
python3 -m unittest tests/test_integration.py   # ffmpeg + ffmpeg-skill 必須（合成素材で実レンダー、契約テスト）
python3 evals/run.py                            # 期待 intent / decision / provenance の評価
```

## レイアウト

`src/video_agent/{models,temporal,capabilities,tools/ffmpeg_skill,media,policy,profiles,skills,agent,project,execution,qa,jobs,audit,providers}`、`schemas/project.schema.json`、`profiles/*.json`、`tests/`、`evals/`。
