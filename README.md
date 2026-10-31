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

未実装（設計のみ）: conference パイプライン本体、Web UI、job queue。

依頼文（brief）から requirement を作る（ADR-061、`requirements_extraction`）: `video-agent plan talk.mp4 --brief "60 s vertical, captions, -14 LUFS"` は brief を `--set` と同じ key（`edit.fit=9:16` / `subtitle=true` / `audio.loudness.target_lufs=-14` / `audio.normalize=true`）に読む。読み手は AI provider（Claude provider、またはチャットのモデルが `--brief-requirements answer.json` で答える host provider）、どちらも無ければ決定的な規則（英語と日本語:「縦型」「字幕」「-14LUFS」など。範囲は `agent/brief.py` と ADR-061）。許可される key は `--set` を検証するコードから導かれ、未知の key・不正な値は理由付きで拒否、key の無いもの（「60秒」のような長さ）は適用せず理由を表示する。明示の `--set` が常に勝つ。AI が読んだ key は AI_GENERATED として記録され、それに基づく変更は CONFIRM になる。解析せずに読みだけ見るなら `video-agent brief "縦型で字幕つき、-14LUFS"`。

AI（Claude）: `pip install -e '.[ai]'` の上で `VIDEO_AGENT_AI_PROVIDER=anthropic`（資格情報は `ANTHROPIC_API_KEY` か `ant auth login`。モデルは `VIDEO_AGENT_AI_MODEL`、既定 `claude-opus-5-5`）。AI は計測結果と依頼文から制作の推薦を出すだけで、引用の無い推薦は捨てられ、実行の可否は policy と検証が決める（ADR-018 / ADR-049）。未設定なら完全に決定的なパイプラインのまま動く。

チャットの Claude を脳にする（API キー不要、ADR-050）: Claude Code / Claude Desktop など、いま話しているチャットのモデルが推薦を出す。手順は2段で、`plan --ai-brief` が AI に渡す材料（計測結果・引用可能な id・許可された intent・依頼文・規則）を出し、チャットのモデルがそれを読んで JSON で答え、`revise <project> --ai-recommendations answer.json --ai-model claude-code` がその答えを ADR-049 と同じ検証に通して新版にする（受理・却下の内訳を表示）。

- Claude Code: OS の installer が `SKILL.md` を `~/.claude/skills/video-agent` にリンクするので、「この動画を YouTube 用にして」と頼めば上の2段を CLI で回す。MCP で使うなら `claude mcp add video-agent -- video-agent-mcp --workspace ~/video-agent-work`。
- Claude Desktop: `claude_desktop_config.json` に MCP server を追加（`video-agent-mcp` は `pip install -e .` で入る。パスは `which video-agent-mcp` の絶対パス）:

```json
{"mcpServers": {"video-agent": {"command": "/abs/path/to/video-agent-mcp", "args": ["--workspace", "/abs/path/to/video-agent-work"],
  "env": {"VIDEO_AGENT_FFMPEG_SKILL_DIR": "/abs/path/to/ffmpeg-skill"}}}}
```

  tools: `doctor` `skills` `plan`（既定で ai_brief 付き）`revise`（`ai_recommendations` を直接渡す）`validate` `explain` `diff` `approve` `reject` `render` `check` `artifacts`。各 tool は `video-agent --json <command>` を子プロセスで実行するだけ。引数の値は option として読まれない（option は `--flag=value`、入力・project・出力・依頼文は `--` の後ろ。ADR-069）。AI と host brief に渡す計測結果の path は basename だけ。承認は `approve` を別に呼ぶ必要があり、`render` に一括承認は無い。
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

Skill の探し方（ADR-069）: 各 Skill は ①`--*-dir` などの明示 ②`VIDEO_AGENT_*_DIR` 環境変数 ③`~/.claude/skills/<name>` ④この agent 自身の source checkout（`src/video_agent` の上に `pyproject.toml` か `.git` がある時だけ）の隣の `<name>` ⑤（CLI 型の Skill は）PATH 上のコマンド、の順で探す。①②が与えられていてそこに無ければ MISSING で、推測には進まない。**カレントディレクトリは見ない**: 以前の `./vendor/<name>` と `../<name>`（作業ディレクトリ基準）は廃止した。展開した素材フォルダなど信頼できない場所で `video-agent doctor` を始めても、そこに置かれたコードは実行されない。上の `git clone ... ../<name>` は agent の checkout の隣なので、どのディレクトリから起動しても見つかる。wheel で入れた agent は ④ が無いので、環境変数か `~/.claude/skills` で指す。Skill の子プロセスには AI provider の資格情報（`ANTHROPIC_*` / `OPENAI_*`）を渡さない。

## 使い方

```bash
video-agent skills                                      # Skill package（ffmpeg-skill / media-analysis / transcription）と production skill の状態、選択された tool
video-agent transcribe input.mp4 --language ja --offline   # transcription-skill で認識 → transcript Observation + SpeechEvent（認識まで。話者・字幕・編集は判断しない）
video-agent plan input.mp4 --profile youtube --kind transcript --language ja   # transcript / SPEECH → speech inference → 発話間の長い無音を削除候補（CONFIRM）として plan に載せる
video-agent plan film.mp4 --kind crop_rectangle            # 黒帯（レターボックス / ピラーボックス）を測り、安定した矩形なら video.crop を CONFIRM で提案（承認するまで切らない、ADR-060）
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
video-agent plan /Volumes/CARD_A --set ingest.dest=/Volumes/SSD1/Footage --set ingest.dest=/Volumes/SSD2/Footage --set ingest.project=2026-10-30_event   # カードの offload の plan（何も書かない、ADR-066）
video-agent approve <workspace>/plans/CARD_A.offload.project.json --decision <id>   # 常に CONFIRM。承認するまでコピーしない
video-agent render  <workspace>/plans/CARD_A.offload.project.json                   # ingest-skill が全 destination にコピーして read-back。コピーの clip が asset になる
```

カードの取り込み（ADR-066、ingest-skill）: カード / レコーダーのフォルダを入力にすると offload の plan になる。コピー先は `ingest.dest` で必ず明示し（マウントされた既存のディレクトリだけ。複数は `--set` を繰り返すか JSON list / カンマ区切り）、`ingest.project` / `ingest.hash=sha256|xxh64` / `ingest.name_by=folder|reel` / `ingest.reel` を任意で足す。plan は ingest-skill の `offload --dry-run` と `inventory` だけを呼び（何も書かない）、decision `ingest.offload` は常に CONFIRM。承認後の render が `ingest offload ... --json` を 1 回だけ実行し（自動の再実行は無い）、全 destination の manifest を確かめたうえで、最初の destination のコピーの inventory（収録ごとの clip、分割収録の member の順、reel / card id、hash、カメラの sidecar）を新しい asset と OBSERVED の `ingest_clip` observation（各 destination の manifest と ASC MHL generation、ingest-skill の version、job）にし、agent 自身の解析（probe / silence / loudness）をコピーに走らせる。失敗は ingest-skill の code のまま、agent の kind（CONFLICT / PARTIAL / COPY_FAILED / HISTORY_DAMAGED / SOURCE_CHANGED / ENGINE_MISSING / INVALID_REQUEST …）と destination ごとの内訳付きで報告する。ingest-skill が無ければ step は BLOCKED で handoff が ingest-skill とインストール方法を名指しする。

声の整音（ADR-073、voice-enhance-skill）: audio production path（`audio.production=true`）で `--set voice.isolate=true`（DeepFilterNet3 で声だけを残す。入力は 48 kHz）、`voice.dereverb=true`（WPE で残響を除く）、`voice.bleed_reduce=true`（`audio.mix=true` の各入力を 1 本のマイクとしてマイク被りを減らす）を明示した時だけ計画する。各入力の cut（と filler 除去）の後、audio-production の全工程の前に bleed 低減 → 残響除去 → 声の分離の順で走る。三つとも voice-enhance-skill で EXPERIMENTAL なので常に CONFIRM。engine が無い・48 kHz でない isolate は Skill 自身の code（`ENGINE_UNAVAILABLE` / `MODEL_NOT_FOUND` と install 行、`FFMPEG_SKILL_UNAVAILABLE`）で BLOCK し、何も download しない。voice-enhance-skill が無ければ handoff が名指しする（`VIDEO_AGENT_VOICE_ENHANCE_DIR` で checkout を指す）。

顔のぼかし（ADR-090、detection-skill）: `--set edit.redact_faces=true` で、detection-skill が 0.5 秒ごとに抜いた frame の顔（YuNet）を時間方向につないだ矩形（`detection faces` → `detection tracks`）を analysis kind `face_tracks` として測り、各矩形をその矩形が測られた時間だけぼかす `video.redact_faces` を CONFIRM で提案する（明示の要求でも profile でも AUTO にならない: 出力の絵を取り消せない形で変え、検出器は顔を見落とす）。承認後は区間ごとに ffmpeg-skill の `cut`（frame 精度）→ 矩形ごとに `redact` → `broll` で元の時間位置に戻す（元の尺と音はそのまま）。trim などの時間を変える処理はその後。`edit.redact_faces.mode=blur|pixelate`、`.blur_strength` / `.block_size`、`.every`（秒、既定 0.5）、`.min_score`、`.window`（track を何秒ごとの矩形に分けるか）。顔が見つからなければ何もせず理由を残し、detection-skill（と YuNet のモデルファイル）が無ければ測定を handoff する。誰の顔かは決して特定しない。公開前に出力を人が確認すること。

オーバーレイ画像の下ごしらえ（ADR-077、imagemagick-skill）: `edit.overlay=logo.png` に `--set edit.overlay.prepare.width=320 edit.overlay.prepare.height=180`（両方必須、px）を足すと、video-editing が合成する前に imagemagick-skill の `resize` がその枠に合わせた新しい PNG を作る（`edit.overlay.prepare.mode=fit|fill|exact`、省略時は contract の既定 fit を記録）。`edit.overlay.prepare.pad=none|<色名>|#RRGGBB(AA)`（+ `.gravity`）で fit の結果をちょうど width x height の canvas に `pad` し、`edit.overlay.prepare.strip=true` で最初に GPS / EXIF を `strip`（EXIF の向きは画素に焼き込む）。順序は strip → resize → pad → overlay で、plan step は `image_strip` / `image_prepare`（`imagemagick-skill/resize`）/ `image_pad`、承認は overlay の decision と同じ。語彙（名前・型・enum・既定）は pin した imagemagick-skill contract（`tools/imagemagick/contract_0.8.1.json`）から読み、利用者の画像は書き換えない。`edit.overlay.prepare.*` が無い plan は今までと同じ。imagemagick-skill が無ければ step は BLOCKED で handoff が imagemagick-skill と `npx imagemagick-skill` を名指しする（`VIDEO_AGENT_IMAGEMAGICK_DIR` で checkout を指す。ImageMagick 本体は利用者が入れる）。

吹替（ADR-085、dubbing-skill）: `--set dub.cues=en.srt dub.lang=en`（翻訳済みの cue は利用者が用意する。agent も dubbing-skill も訳さない）で、edits の後の絵の台詞を dubbing-skill の DUB が置き換える（元の音を下げてその上に話す、絵は stream copy）。`dub.engine` / `dub.voice` / `dub.duck` / `dub.voice_gain` / `dub.max_overrun` / `dub.on_unfit` / `dub.source_cues`。実在の人の声を使うなら `dub.reference=voice.wav dub.consent=consent.json`（同意記録 `dubbing/consent@1` は利用者が用意し、agent は作らない。片方だけは拒否）。常に確認が要り（EXPERIMENTAL）、`CONSENT_REQUIRED` は再試行しない。engine が求めるクレジットは ProductionReceipt の `obligations.credit` と report.md に載る。

3D レンダと HTML シーン（ADR-080、blender-skill / hyperframes-skill）: agent は実行しない handoff だが、`--set job.render_3d=true job.render_3d.model=robot.glb job.render_3d.preset=1080p job.render_3d.transparent=true`、`--set job.html_scene_render=true job.html_scene_render.scene=title-card/ job.html_scene_render.codec=prores` のように引数を付けると、pin した各 Skill の contract（blender-skill 0.31.0 / hyperframes-skill 0.6.2）で analysis 前に検査し、handoff の `contract` に tool（`blender-skill/render` / `hyperframes-skill/render`）・引数・不足・出力の行き先（ffmpeg-skill/overlay か video-editing/concat）を出す。plan は BLOCKED のまま。

話者の区切り（ADR-086、diarization-skill）: `--set speakers.diarize=true`（+ 分かっていれば `speakers.num_speakers=2`、または `speakers.min_speakers` / `speakers.max_speakers`、別の音声 track は `speakers.audio_stream`）を明示した時だけ、音声のある各入力に diarization-skill の `diarize` を計測として走らせる。結果は匿名の話者 turn（`SPEAKER_00` …、名前ではない）で、`speaker_turns` の Observation（OBSERVED、engine / model / revision / CC-BY-4.0 の attribution 付き）と timeline の SpeakerEvent になる。何も決めず、編集も変えない。pyannote.audio・torch・gated な pipeline directory（`DIARIZATION_PIPELINE_DIR`）が揃わず DIARIZE が ready でない（diarization-skill の doctor による）か、diarization-skill が無ければ、計測は走らず plan は BLOCKED で handoff が diarization-skill と install を名指しする（`VIDEO_AGENT_DIARIZATION_DIR` で checkout を指す）。計測が失敗すれば Skill 自身の code で BLOCK。

終了コード: 0 完了 / 2 検証エラー / 3 BLOCKED / 4 承認待ち / 5 QA FAIL (REVIEW) / 130 中断 (CANCELLED) / 1 その他。revise: 5 = 新版なし。

video-editing-skill 連携（agent → video-editing CLI（contract / stdin request / `--workspace` `--allowed-input`）→ ffmpeg-skill → FFmpeg。agent は command / argv / filter を作らず、Skill の response（sha256 / timeline / OBSERVED probe / commands）を Artifact・provenance に記録する。silence_cleanup の候補 `video-editing/cut` は宣言順 2 番目、両方利用可能なら従来通り ffmpeg-skill/cut）: ADR-028 / MASTER_SPEC「video-editing-skill integration」。編集操作 concat / speed / resize / fit / fill / overlay、明示の trim（`edit.trim=2-10`）と rotate（`edit.rotate=90`、ADR-062）、静止画（タイトルカード / エンドスレート）を仕上げた絵の前か後ろにつなぐ `edit.image_insert=card.png edit.image_insert.duration=3 edit.image_insert.position=before|after`（video-editing-skill IMAGE_INSERT + CONCAT。静止画は絵と同じ大きさ・frame rate の無音 clip、PNG / JPEG 以外は plan 時に拒否、ADR-070）（`--set edit.concat=true edit.speed=2 edit.resize=640 edit.fit=1:1 edit.overlay=logo.png` のような明示 requirement → Decision → Plan → IR `video.*` operation → compiler → video-editing-skill。concat は trimmed 入力を入力順に結合した `programme` を subject にし、以降の操作・loudness・delivery は programme に適用。値域外・曖昧・矛盾（fit+fill）・video stream 無し・capability 欠落は拒否 / BLOCK）: ADR-029 / MASTER_SPEC「video-editing-skill operations」。audio-production-skill 連携（`--set audio.production=true` で asset の音声を subject にする audio path: 既存の無音 decision → `audio.cut`、loudness decision → NORMALIZE（Skill が再測定）、`audio.gain` / `audio.channels=mono|stereo` / `audio.fade_in` / `audio.fade_out` / `audio.concat`（+`.crossfade`）/ `audio.sample_rate`、ADR-063 の `audio.mix`（+`.levels`）/ `audio.noise_reduction.strength_db`（+`.mode`）/ `audio.dynamics.<gate|compressor|limiter>`（+`.<field>`）、ADR-068 の `audio.voice_cleanup=light|medium|strong`（話し声の整え。light / strong は ffmpeg-skill 1.13 以上）、ADR-072 の `audio.filler_remove=ja|en|ja,en`（または `true` + `.words`。`.keep` / `.pad` / `.merge_gap` / `.declick_ms`。transcript の単語時刻（word timestamps を自動で要求）を無音カットの後の時間軸に移して FILLER_REMOVE に渡す。既定の語彙なし、単語時刻の無い segment は BLOCK、常に CONFIRM）と `audio.mix.duck.input` + `.amount_db`（+ `.threshold_db` / `.attack_ms` / `.release_ms`。bed と量は明示必須）は明示 requirement → Decision → Plan → IR `audio.*` → compiler → audio-production-skill。video container は音声のみ納品（`audio.extract` decision、CONFIRM。`audio.production=true` だけでは waive されず、`--set audio.extract=true` か decision の明示承認が必要: ADR-033）、video 編集との同時要求・video preset・audio 無し・実現不能な layout・単独 resample は BLOCK）: ADR-030 / MASTER_SPEC「audio-production-skill integration」。複数ソース同期の観測（`video-agent analyze camA.mp4 recorder.wav --kind sync`: 先頭入力を reference として ffmpeg-skill/sync が測った offset / confidence / drift を `sync` Observation（OBSERVED）として保持し、信頼度が `sync.min_confidence`（既定 0.3）以上なら target の TimelineMap に反映。Decision は生成しない）: ADR-035 / MASTER_SPEC「Multi-source sync observation」。Production Decision Engine（Inference = 何が起きているか / Decision = 制作として何をすべきか / Plan = どう実行するか。decision は evidence 必須・type 語彙・policy 解決の provenance と basis を保持、`video-agent explain --decision <id|subject>` で policy / preference / constraint / intent / evidence / plan step まで辿れる）: ADR-027 / MASTER_SPEC「Production Decision Engine」。Production Plan（Decision / Event → plan → IR の決定論的橋渡し、`video-agent explain --step`）: ADR-021 / MASTER_SPEC §21。Artifact / Delivery / Archive（`video-agent artifacts` / `artifact` / `deliver` / `archive` / `explain --artifact`）: ADR-022 / MASTER_SPEC §36。AI Provider contract（AI は提案のみ、execution authority ではない）: ADR-018 / MASTER_SPEC §42。Revision workflow: `docs/revision.md`。Skill / Capability / Tool と将来 Skill の追加手順: `docs/skills.md`、Gap 分析: `docs/GAP_ANALYSIS_PHASE2.md`。

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

字幕を焼き込まずにソフト字幕トラック（プレイヤーで切り替えられる字幕）として入れるなら `--set subtitle.mode=mux`（ADR-063。subtitle-skill の render の `mode` そのもの。`subtitle.mode=burn` は `subtitle.burn_in=true` と同じ）。絵と音はコピーのまま。ffmpeg-skill 2.22.4 より前の platform preset の書き出し（youtube など）は字幕ストリームを落とすので、その engine と gif preset では BLOCK になる（generic などで納品する）。2.22.4 以上なら preset の書き出しもトラックを残すので BLOCK しない（ADR-071）。納品物にトラックが残っているかは QA が確かめる。

編集ソフトへの受け渡し（ADR-052）: `--set delivery.timeline=otio`（または `fcpxml` / `edl`、既定 `none`）で、IR の cut list（trim が残す範囲を programme の順に並べたもの、hard cut、source は元の場所を参照）を ffmpeg-skill `render.py --export-timeline` が書き出し、type `TIMELINE` の Artifact になる。speed / colour / graphics などはタイムラインに載らず、decision に名指しされる。switch / grid / audio path は BLOCK。

テイク選び（ADR-053）: `--set edit.takes=true` は、同じセリフを撮った複数テイクを transcript の類似（既定 0.6、`edit.takes.similarity`）でセリフごとにまとめる。各テイクは測った事実だけで採点する。transcript からはつなぎ言葉・言い直し・欠け、音声測定からはクリップ・音量・間を見る。sharpness / exposure / 顔は測る adapter が無いので採点しない。セリフごとに最良のテイクを `takes.pick`（得点と根拠の測定つき、CONFIRM）として選び、セリフ順に concat する。`--set edit.takes.music=<入力 index>` を付けると、その入力のビートを ffmpeg-skill `scenes --beats` で測り、つなぎ目（各 pick の out 点）を programme 時刻で最も近い beat に合わせる（既定で最大 0.5 秒、`edit.takes.snap_max_shift`）。AI 推薦（`take_selection`）は pick を差し替えられるが、得点は変えない。`delivery.timeline=otio` はこの cut list をそのまま書き出す。つなぎ目には既定で 10 ms の音声クロスフェードが入る（ADR-054、`edit.takes.crossfade`、0 で無効）。絵の cut 点と長さは動かさず、各テイクの音声を cut の前後に少し延ばして audio-production でつなぎ、ffmpeg-skill `audio.py --replace` で絵に戻す。余裕の無いつなぎ目は hard cut のまま理由を残し、A/V 同期は validator が IR から作り直して確かめる。音楽のミックスはまだ無い。

生成される ProductionPlan（依存関係付き）: trim ×3 → concat → captions.generate（transcript の cue を delivered timeline に写像）→ captions.burn → loudness → export → check → thumbnail → qc ×3（deliverable、sidecar、そして deliverable + sidecar + thumbnail をひとまとめにした納品セット、ADR-065）。QC PASS → artifact は `approved`（READY）、WARN → candidate（policy `qc.warn.promotion`、既定 CONFIRM）、FAIL → `working`（final 昇格不可）。納品セットの検査（qc-skill の kind `delivery_package`: 全員がそろい、空でなく、計画した拡張子で、sidecar / thumbnail が deliverable に付いている）は `qc=true` の gate の一部で、FAIL ならセットの全員が `working` になる（thumbnail もこの検査で gate される）。

モーショングラフィックス（ADR-064 / ADR-078）: motion-graphics-skill の pin した contract が出す 13 element すべてを `--set` で頼める。`motion.title` / `motion.lower_third`（名前）/ `motion.text` / `motion.image`（PNG / JPEG）/ `motion.video`（ピクチャーインピクチャー、クロマキー可）/ `motion.bug` / `motion.chapter` / `motion.sticker` / `motion.hook`（文字）と、スイッチの `motion.progress` / `motion.countdown` / `motion.meme`（`.top` / `.bottom`）/ `motion.shape`（単色の矩形。`.width` / `.height`（偶数）/ `.color` が必須、`.opacity` 等は任意、0.2.7）。それ以外の parameter は contract の名前そのままで `motion.<element>.<parameter>`（例 `motion.text.font_size=64`、`motion.video.chromakey=00FF00`、`motion.countdown.count_from=3`）、値域も contract のまま。`.start` / `.end`（text / image は `.fade` も）。明示した位置（`{x, y}`・margin・scale_width・font_size）は編集後の絵の大きさ（crop / 回転 / fit / grid の後）に収まるかを確かめ、はみ出すなら BLOCK（動かさない）。色は `FFCC00` のような RRGGBB か、motion-graphics-skill の contract の色名（`colors.named`、FFmpeg の 140 色名、大文字小文字を問わない）。`@alpha`（例 `black@0.5`）は overlay の色だけで、テンプレートで描く element（title / lower_third / bug / chapter / progress / countdown / sticker / hook）の色は contract が `alpha: false` と言うので拒否する（ADR-067）。「話者が変わったらカメラを切り替える」は未実装（docs/GAP_ANALYSIS_PHASE2.md §22）。

色の一次補正（ADR-031 / ADR-062 / ADR-076）: color-grading-skill の PRIMARY_CORRECTION の parameter を contract の名前そのままの `--set` key で頼める。`color.exposure`（-3〜3 stop）/ `color.contrast`（0〜2）/ `color.saturation`（0〜2）/ `color.temperature`（2000〜12000 K）/ `color.tint`（-1〜1）/ `color.gamma`（0.1〜10）/ `color.lift`・`color.gain`（-1〜1、shadows / highlights）/ `color.levels_in_black`・`color.levels_in_white`・`color.levels_out_black`・`color.levels_out_white`（整数 0〜255、黒 < 白）/ `color.curves`（curves preset: color_negative / cross_process / darker / increase_contrast / lighter / linear_contrast / medium_contrast / negative / strong_contrast / vintage）。key・型・値域・enum は pin した contract（`tools/color_grading/contract_0.4.0.json`）から読み、plan の時点で検査する（範囲外・小数の levels・enum 外は plan を作らずに拒否）。既定値は無い: 明示した key だけが要求に載り、残りは Skill の「変えない」既定のまま。例: `--set color.gamma=1.2 --set color.levels_in_black=16 --set color.levels_in_white=235 --set color.curves=medium_contrast`。

マスターとの画質比較（ADR-058、qc-skill 0.2.3）: `--set qc=true --set qc.reference_video=master.mov --set qc.reference_metrics=vmaf,ssim`（`psnr` も可。しきい値は任意で `qc.min_vmaf` / `qc.min_ssim` / `qc.min_psnr_db`、無ければ qc-skill の既定 93 / 0.95 / 31 dB）で、video の各 deliverable を指定したマスターと比べる。マスターのフォルダは allowed input root に加わり、qc の報告はマスターの sha256 も agent 自身の計算と一致した時だけ採用される。fps や長さが違う組は qc-skill が UNKNOWN にする（編集した deliverable はマスターと揃わない）。libvmaf の無い ffmpeg では VMAF だけが UNKNOWN になる。

## テスト

```bash
python3 -m unittest tests/test_unit.py          # ffmpeg 不要（FakeAdapter）
python3 -m unittest tests/test_integration.py   # ffmpeg + ffmpeg-skill 必須（合成素材で実レンダー、契約テスト）
python3 evals/run.py                            # 期待 intent / decision / provenance の評価
```

## レイアウト

`src/video_agent/{models,temporal,capabilities,tools/ffmpeg_skill,media,policy,profiles,skills,agent,project,execution,qa,jobs,audit,providers}`、`schemas/project.schema.json`、`profiles/*.json`、`tests/`、`evals/`。
