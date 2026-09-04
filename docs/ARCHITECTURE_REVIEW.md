# video-production-agent — Architecture Review (First Required Review)

対象: `kajisho5/ffmpeg-skill` v0.8.4（commit `8e10ffdd`, 2026-09-04）を実コードから調査した上での、MASTER_SPEC §56 の22項目に対するレビュー。

表記ルール:

- **[FACT]** … ffmpeg-skill のコード・テスト・ドキュメントから確認した事実。
- **[DESIGN]** … 本プロジェクトでこれから設計・実装する内容（推測ではなく提案）。
- **[UNKNOWN]** … 確認できなかった / 未決の事項。

---

## 0. 調査方法と環境 [FACT]

- 読んだもの: `SKILL.md`, `README.md`, `CHANGELOG.md`, `references/scripts.md`, `references/devices.md`, `scripts/*.py`（全20スクリプト + `_common.py`、合計約6,000行）, `mcp/server.py`, `tests/test_all.py`（55テスト）, `tests/corpus.py`, `tests/release_check.sh`, `evals/*`, `bin/install.js`, `examples/*`, `.github/workflows/ci.yml`。
- 実行して確認したもの: 各スクリプトの `--help`、`mcp/server.py --list`（20ツール）、`cut.py` の `--dry-run` 挙動。
- この調査環境には ffmpeg が無い（後述の Phase 1 実装時に apt で導入して実メディアテストを行う）。

---

## 1. ffmpeg-skill 機能インベントリ [FACT]

### 1.1 全体構造

| 項目 | 事実 |
|---|---|
| 言語 / 依存 | Python 3.9+ 標準ライブラリのみ。外部 pip 依存なし。 |
| 外部ツール | `ffmpeg` / `ffprobe` を PATH から `shutil.which` で解決（`_common.require_tool`）。無ければ exit 127。 |
| 想定 FFmpeg | README: 5.0+、`libx264`, `libx265`, `libass`, `prores_ks`, `libzimg`（`color.py --to-sdr` 用）。コードで機能検出はしていない（無ければ ffmpeg のエラーで失敗）。 |
| 形態 | Agent Skill（`SKILL.md` + `scripts/`）。`npx ffmpeg-skill` で `~/.claude/skills/ffmpeg-skill` 等にコピー。 |
| 呼び出し形態 | 各スクリプトは独立したプロセス（`python3 scripts/<name>.py ...`）。`render.py` / `batch.py` / `report.py` / `verify.py` も兄弟スクリプトを `subprocess` で起動する。 |
| 共通フラグ | `--dry-run`, `--json`, `--progress`, `--fast`（`_common.add_common`）。 |
| 出力規約 | stdout の最終行 = 出力パス。`--json` 時は JSON 1文書（`{"output", "dry_run", "commands", "probe"?, ...extra}`）。エラーは stderr に `error: ...`、非0終了。 |
| 既定出力名 | `<input>_<suffix>.<ext>`（`default_output`）。`-o` で指定可。 |
| 上書き | `ffmpeg_base()` は常に `-y`。**入力と同じパスを `-o` に渡した場合の保護は無い**。 |
| ログ | ffmpeg コマンド行は `STATE.commands` に記録され `--json` に含まれる。ffprobe は記録されない。 |
| 状態 | `_common.STATE` はプロセスグローバル。ライブラリとして同一プロセスで複数回・並列に使う設計ではない。 |

### 1.2 スクリプト一覧（20本）

| script | 役割 | 主な入力 | JSON 出力の追加キー | 再エンコード | 備考 |
|---|---|---|---|---|---|
| `probe.py` | メディア検査 | files | （probe 本体） | なし | `--analyze` で `levels.looks_like_log`。`--compact`, `--field a.b`。 |
| `cut.py` | 区間切り出し / 複数区間連結 | `--start/--end/--duration`, `--segments A-B,C-D`, `--accurate`, `--tolerance` | なし | 既定 `-c copy`、ズレ >0.5s で自動再エンコード。VFR は自動 `--accurate`。 | concat は demuxer。 |
| `fit.py` | 尺合わせ（speed/trim）、アスペクト（pad/crop）、fps | `--duration`, `--method`, `--aspect`, `--fit`, `--width`, `--fps`, `--smooth` | なし | あり | `--max-speed` 4x 超は拒否。 |
| `silence.py` | 無音検出・除去 | `--threshold`, `--min-silence`, `--margin`, `--min-keep`, `--list`, `--edl` | `silences[[s,e]]`, `keep[[s,e]]`, `input_duration`, `kept_duration`, `removed_seconds` | あり（select/aselect で一括） | `--list --json` で検出のみ。EDL は `cut.py --segments` 互換。 |
| `join.py` | 連結 + xfade | inputs, `--transition`, `--duration`, `--width/--height/--fps`, `--fit` | `clips`, `transition`, `expected_duration` | あり | サイズ/fps/音声レイアウトを正規化。 |
| `caption.py` | 字幕焼き込み / SRT 生成 / ASS 生成 | `--srt/--ass/--text/--transcribe`, style, `--animate`, `--karaoke`, `--brand` | なし | あり | `--transcribe` は whisper.cpp / faster-whisper / openai-whisper のいずれかが「あれば」使う。フォント検出は無い（SKILL.md で `fc-list` を促すだけ）。 |
| `overlay.py` | 画像/テキスト/ブランドロゴ合成 | `--image/--text/--logo`, position, `--start/--end/--fade/--opacity` | なし | あり | |
| `graphics.py` | lower-third, title, chapter, progress, countdown, bug | `--template`, text, `--brand` | `template` | あり | drawbox/drawtext のみで描画。 |
| `sync.py` | 2ファイル間オフセット検出、ドリフト補正 | reference, second, `--fix-drift`, `--replace-audio`/`--trim-second` | `offset_seconds`, `confidence`, `meaning`, `drift{drift_ppm,...}` | `--replace-audio` は video copy | 純 Python FFT 相関。confidence 0–1。 |
| `multicam.py` | N入力を音声で整列しスイッチ | inputs, `--switch "S-E:CAM,..."`, `--auto N`, `--audio IDX`, `--offsets-only`, `--fix-drift` | `offsets_seconds[]`, `confidence[]`, `drift_ppm[]?`, `cuts[[s,e,cam]]` | あり（filter_complex trim/concat） | スイッチは**時刻リストを与えるだけ**。カメラ選択の判断ロジックは無い。 |
| `audio.py` | voice chain / denoise / music+duck / fades / downmix / replace | 各フラグ | なし | video copy | |
| `loudness.py` | 2パス loudnorm | `-I`, `--tp`, `--lra`, `--measure-only` | `--measure-only` は `{input_i,input_tp,input_lra,input_thresh,target_offset}` または `{silent:true}` | video copy | |
| `color.py` | HDR→SDR, LUT, retag, strip-dovi | 排他4モード | なし | to-sdr/lut は x264、retag/strip は copy | |
| `export.py` | プリセット書き出し | `--preset youtube/youtube4k/reels/x/prores/h265/gif`, `--fit`, `--crf`, `--allow-long` | なし | あり | プリセットは固定テーブル。HDR 入力は警告のみ。 |
| `check.py` | 納品コンプライアンス | `--platform youtube/shorts/reels/tiktok/x/linkedin/broadcast/podcast/custom` + override | `platform`, `checks[{check,status,value,expected,fix,kind}]`, `failed`, `warnings`, `ok` | なし | `kind` は `format` / `judgement`。FAIL があれば exit 1。 |
| `scenes.py` | シーンチェンジ + 音声ピーク + ハイライト提案 | `--threshold`, `--ratio`, `--highlights`, `--target`, `--edl`, `--sheet` | `scenes[{index,start,end,duration,audio_rms,audio_peak}]`, `audio_peaks[{time,rms}]`, `highlights[]` | なし | scdet のスパイク判定。 |
| `look.py` | コンタクトシート / フレーム / 比較 PNG | `--at`, `--tiles`, `--compare` | `outputs[]` | なし | HDR はトーンマップして表示。 |
| `render.py` | project.json から一括レンダー | project, `--stop-after`, `--work`, `--keep` | `stages[]`, `check` | 各段が別プロセスで再エンコード | 詳細は 1.3。 |
| `batch.py` | フォルダ一括 + キャッシュ | folder, `--recipe`, `--force`, `--watch` | `results[]`, `processed`, `total` | 手順依存 | キャッシュキー = 名前/サイズ/mtime + 先頭末尾1MB の sha1（全体ハッシュではない）。 |
| `verify.py` | 実素材でツールチェーン検証 | files/folders | `files[]`, `failed`, `total`, `report` | あり | PASS/FAIL 表。 |
| `report.py` | HTML 納品レポート | `--after`, `--before`, `--platform`, `--commands` | `report`, `check` | なし | |

### 1.3 `render.py` の project.json（現在の「宣言的編集」契約）[FACT]

- トップレベルキー: `output`, `frame{aspect,width,fps}`, `clips[{src,in,out,speed}]`, `transition{type,duration}`, `silence{...}`, `brand`, `captions{...}`, `graphics[...]`, `overlays[...]`, `audio{...}`, `loudness{lufs,tp}`, `fit{...}`, `export{preset,fit,crf}`, `check{platform}`。
- ステージ順は固定: clips → join → silence → fit → captions → graphics → overlays → audio → loudness → export → check。順序変更・分岐・複数出力は不可。
- 各ステージは兄弟スクリプトを subprocess 起動し、中間ファイルを `<output>_work/` に置く（`--keep`/`--work` なしなら削除）。
- **スキーマ検証は無い**。未知キーは無視、`schema_version` 等のバージョンフィールドも無い。
- `--dry-run` 時: 存在しないファイルの `probe()` は 1920x1080/30fps のダミーを返す。`check` ステージはスキップ。
- `check` の FAIL は stderr に出るが **render.py の終了コードは 0** のまま（`emit(..., check=check_result)` で JSON に載るだけ）。

### 1.4 `mcp/server.py` [FACT]

- stdio JSON-RPC 2.0、プロトコル `2024-11-05`。`initialize`, `tools/list`, `tools/call`, `ping` のみ。
- 20ツール。`inputSchema` は `{"argv": array}` + `additionalProperties: true` で、**引数の型情報は無い**（説明文に列挙されているだけ）。
- 名前付き引数はフラグへ機械変換（`_`→`-`、`output`→`-o`、`loudness.lufs`→`-I`）。`look`/`probe` 以外には `--json` を自動付与。
- 結果: `content[0].text` と `structuredContent`（スクリプトの JSON）。失敗時 `isError: true` + stderr 末尾12行。
- 1呼び出し = 1プロセス。同期・タイムアウト無し・進捗通知無し。

### 1.5 probe が返す観測フィールド [FACT]

`file, format, duration, size_bytes, bitrate, subtitle_streams, video{codec, profile, width, height, display_aspect, fps, r_frame_rate, avg_frame_rate, variable_frame_rate_suspected, pix_fmt, bit_depth, hdr, hdr_format, dolby_vision, color_space, color_primaries, color_transfer, color_range, rotation, nb_frames, bitrate}, audio{codec, channels, channel_layout, sample_rate, bitrate}`、`--analyze` で `levels{...,looks_like_log}`。

- 最初の video / 最初の audio ストリームだけ。複数音声トラック・字幕トラックの内容は見ない。
- 全ストリーム一覧、タイムコードトラック、チャプタ、ファイルハッシュは返さない。

### 1.6 テスト・評価 [FACT]

- `tests/test_all.py`: unittest 55件、合成素材（testsrc2 + 合成音、VFR、回転、5.1、HDR10、HLG、ドリフト）で E2E。ffmpeg 必須。
- `tests/corpus.py`: 実機素材10件（GoPro, DJI, iPhone DV, Android 画面録画, HDR10, VP9 HDR, 24p, ToS）を DL して `verify.py`。
- `tests/bench_{sync,silence,scenes}.py`: アルゴリズムのベンチマーク。
- `evals/`: ルーティング評価（transcript にスクリプト名が含まれるか）、24プロンプト×3回の独立採点、トリガー評価。
- CI: 3OS マトリクスだが `workflow_dispatch` のみ（手動）。

### 1.7 ffmpeg-skill に**無い**もの（Agent 側で設計が必要）[FACT]

- 機能検出（エンコーダ/デコーダ/libass/zimg/GPU/フォントの有無）。`ffmpeg -encoders` 等は一切呼ばない。
- 黒フレーム / フリーズ / 破損 / クリッピング / チャンネル欠落 / ドロップアウトの検出。
- 入力ハッシュ、ツールバージョン記録、再現性メタデータ。
- 入出力パスの境界（ワークスペース制限）。任意パスを読み書きする。
- 承認・リスク・説明可能性の概念。`check.py` の `kind: judgement` が唯一の「判断が要る」マーカー。
- Skill/Capability/Tool の区別、プロファイル（`brand.json` と `export`/`check` の固定プリセットのみ）。
- 学会 / 会議向けのプロファイルやセッション概念。
- LLM / AI プロバイダ連携（whisper ブリッジのみ、しかもオプション）。

---

## 2. 再利用するコンポーネント [FACT + DESIGN]

Agent からは ffmpeg-skill を **外部プロセス（CLI）として** 利用する。理由は 1.1 の `STATE` グローバルと `die()`（`sys.exit`）がライブラリ利用に向かないため。

| ffmpeg-skill 側 | Agent での利用方法 |
|---|---|
| `probe.py --json` | Observation の一次ソース（`media.probe`）。 |
| `probe.py --analyze` | Log 判定 Observation（Phase 2 以降）。 |
| `silence.py --list --json` | 無音 Observation → AudioEvent(SILENCE)。 |
| `loudness.py --measure-only` | ラウドネス Observation（入力 QA / 出力 QA 両方）。 |
| `scenes.py --json` | SceneEvent / 音声ピーク Observation（Phase 3）。 |
| `sync.py --json`, `multicam.py --offsets-only --json` | 同期 Observation + confidence（Phase 2/3）。 |
| `check.py --platform X --json` | Delivery QA の実行器。`kind` をそのまま QA 項目の risk 分類に写像。 |
| `look.py` | 視覚 QA 用 PNG（Phase 1 では artifact として保存、判断は人間 / 将来 AI）。 |
| `cut.py`, `fit.py`, `loudness.py`, `export.py`, `caption.py`, `overlay.py`, `graphics.py`, `audio.py`, `color.py`, `join.py`, `multicam.py` | Project IR の operation を compile した先の Tool 呼び出し。 |
| `render.py` project.json | **Phase 1 では使わない**（1.3 の固定順序・検証無し・check 結果が終了コードに反映されない、が理由）。将来、IR の一部を render プロジェクトへ「lowering」する adapter は作れる。 |
| `report.py` | 納品レポート生成の下請け（Agent のレポートに HTML を添付）。 |
| `mcp/server.py` | Phase 1 では使わない（型無し・タイムアウト無し）。他ホストからの利用のために残す。 |
| `--dry-run --json` の `commands` | Dry Run 表示と provenance に転記。 |

---

## 3. 複製してはいけない機能 [DESIGN]

以下は Agent 側に再実装しない。必要なら ffmpeg-skill 側に PR を出す。

- ffmpeg/ffprobe の起動、コマンド組み立て、エンコーダ引数（`video_args`, `x264_args`, `cfr_args`, HDR 維持ロジック）。
- カット / 連結 / フィット / 字幕 / オーバーレイ / グラフィクス / 同期 / マルチカム合成 / 音声処理 / ラウドネス / カラー / 書き出し。
- 無音検出、シーン検出、オフセット検出のアルゴリズム。
- プラットフォーム仕様表（`check.py SPECS`, `export.py PRESETS`）。Agent の profile はこれらの **名前を参照**し、数値を二重管理しない。
- コンタクトシート生成、HTML レポート。

Agent が **唯一** 自前で持つ ffprobe 系の処理は、`ffmpeg -encoders/-decoders/-filters/-version` と `fc-list` による **機能検出（doctor）** だけ。これは ffmpeg-skill に存在しないため。

---

## 4. 責務境界 [DESIGN]

| 責務 | ffmpeg-skill (Hands) | video-production-agent (Brain) |
|---|---|---|
| ffmpeg コマンド生成・実行 | ○ | ×（禁止。adapter 経由でスクリプトを呼ぶだけ） |
| メディア観測（probe, silence list, loudness measure, scenes, sync） | ○ 実行 | ○ 結果を Observation として構造化・保存 |
| 推論（「この無音は不要」「これは登壇者交代」） | × | ○ Inference（confidence, evidence 付き） |
| 要求理解 / 要件抽出 / 意図 | × | ○ |
| Policy / Preference / Constraint | × | ○ |
| 決定（AUTO/CONFIRM/BLOCK, risk） | × | ○ |
| Skill 選択 / Capability 解決 / Tool 選択 | × | ○ |
| Project IR / 検証 / compile | × | ○ |
| operation 実行 | ○ | ○ オーケストレーション（順序、リトライ、キャンセル、ワークスペース） |
| 出力 QA | ○ `check.py`, `probe.py`, `loudness --measure-only`, `look.py` | ○ 統合・判定・Incident 化 |
| Recovery | ×（`cut.py` の copy→再エンコード fallback は内部的な処理選択であり、Agent から見れば1 operation） | ○ 分類・戦略・有限リトライ |
| Job / Artifact / Provenance / Audit | × | ○ |
| プロファイル（generic / youtube / conference …） | ×（brand.json と固定プリセットのみ） | ○ |
| AI プロバイダ | × | ○（Phase 1 はオプション、決定論的パスのみで動く） |

**ffmpeg-skill インターフェースとの不一致と対処**（CLAUDE.md「adapter を設計せよ」への回答）:

| 不一致 | 対処 |
|---|---|
| `check.py` FAIL で exit 1、`render.py` は exit 0 | adapter は終了コードと JSON の両方を見る。`render.py` は使わない。 |
| `--dry-run` でも ffmpeg バイナリと入力ファイルが必要（`cut.py` は存在しない入力で失敗） | Agent の dry-run は **IR とカタログから** operation を列挙し、ffmpeg-skill の `--dry-run` は「実コマンド確認」の任意付加ステップとする。 |
| 出力名が `<input>_<suffix>` で衝突しうる | adapter は常に `-o` を Agent 管理の workspace パスで指定する。 |
| 入力=出力の保護が無い | adapter が `-o` を source ディレクトリ外に強制する（Constraint: preserve original）。 |
| MCP の inputSchema が型無し | Agent 側の Tool カタログ（`tools/ffmpeg_skill/catalog.py`）に引数型を持ち、adapter で検証する。 |
| バージョン情報が JSON に無い | adapter が `package.json` の `version` と `ffmpeg -version` を provenance に記録。 |
| 最初の音声トラックしか扱えない | Asset 分析で `ffprobe -show_streams` 相当を Agent が追加取得するか、ffmpeg-skill に `probe.py --streams` を提案（[UNKNOWN] Phase 2 で判断）。 |

---

## 5. 提案アーキテクチャ [DESIGN]

MASTER_SPEC §52 のレイアウトを採用し、名称を一部具体化する。

```
src/video_agent/
  models/        # Request, Requirement, Intent, Observation, Inference, Decision, Asset, Event, Incident, Artifact, Job
  media/         # ffmpeg-skill を使う MediaAnalyzer（probe / silence / loudness / scenes を Observation へ）
  temporal/      # Timeline, Event, TimeRange, クエリ
  policy/        # Policy / Preference / Constraint と precedence
  skills/        # SkillRegistry と Skill 定義（silence_cleanup, loudness_normalization, delivery_export ...）
  capabilities/  # CapabilityResolver（ffmpeg/ffprobe/encoders/libass/zimg/fonts/ffmpeg-skill）
  tools/         # ToolAdapter 基底 + ffmpeg_skill/{adapter,catalog}
  agent/         # RequestParser, RequirementExtractor, IntentResolver, DecisionEngine, Planner
  project/       # ProjectIR dataclass, schema loader, validator, migrations
  execution/     # Compiler(IR→OperationPlan), Executor, Recovery
  qa/            # VideoQA, AudioQA, DeliveryQA, Incident 生成
  jobs/          # Job state machine, workspace, cancellation
  profiles/      # Profile loader（profiles/*.json）
  providers/     # AIProvider 抽象（Phase 1: NullProvider + 環境変数でオプション）
  audit/         # Provenance / audit log 書き出し
  evals/         # Eval runner
  cli.py
schemas/project.schema.json
profiles/{generic,youtube,conference,...}.json
```

原則:

- **AI は「何をすべきか」を決め、実行層は「どう安全に実行するか」だけを決める**。Planner の出力は Project IR であり、Executor は IR 以外を入力に取らない。
- Phase 1 は **AI プロバイダ無しで完全に動く**（Request は構造化 CLI 引数 + 簡易ルール解釈。自然言語理解は `providers/` に差し込む拡張点として残す）。

---

## 6. データフロー [DESIGN]

```
Request(raw text | CLI args)
  → RequirementExtractor  → Requirements[ {key, value, provenance: USER|PROFILE|DEFAULT|SYSTEM} ]
  → IntentResolver        → Intent{ primary: "clean_and_deliver", targets: [...], confidence }
  → MediaAnalyzer         → Assets[], Observations[]   (probe / silence --list / loudness --measure-only)
  → Inferencer            → Inferences[] (例: leading_silence_unwanted, confidence, evidence=[obs_id])
  → PolicyResolver        → EffectivePolicy{ policies, preferences, constraints, provenance }
  → CapabilityResolver    → Capabilities{ name: AVAILABLE|MISSING|DEGRADED|UNKNOWN }
  → DecisionEngine        → Decisions[] { decision, reason, confidence, evidence, alternatives, risk, approval }
  → SkillSelector/ToolSelector → ProductionPlan{ steps: [ {skill, tool, decision_ids, params} ] }
  → IRBuilder             → ProjectIR (schema_version, ... , decisions, plan, timeline, delivery, qa)
  → Validator             → ValidationReport (schema + semantic + capability)
  → Compiler              → OperationPlan[ {op_id, tool:"ffmpeg-skill/cut", argv-model, inputs, outputs, decision_id} ]
  → Executor(ToolAdapter) → ToolResult[] (stdout JSON, commands, exit code, duration)
  → QA                    → QAReport, Incidents[]
  → Recovery(有限)         → 再実行 or BLOCK
  → Report / Artifacts / Provenance
```

Observation と Inference は別テーブル。Inference は必ず `evidence: [observation_id]` を持ち、Observation を書き換えない。

---

## 7. Temporal Model [DESIGN]

### 7.1 データ構造

```python
@dataclass
class TimeRange:
    start: float          # 秒。基準は timeline_id のタイムライン
    end: float | None     # None = 点イベント

@dataclass
class Event:
    id: str
    type: str             # 例: AUDIO_SILENCE, SCENE_CHANGE, SPEECH, SLIDE, SPEAKER, CAPTION, INCIDENT, CAMERA_STATE, USER_DECISION
    timeline_id: str      # "asset:<asset_id>" または "master"
    range: TimeRange
    source: str           # "ffmpeg-skill/silence.py@0.8.4", "user", "inference:<id>"
    kind: Literal["OBSERVED", "INFERRED", "USER"]
    confidence: float | None
    evidence: list[str]   # observation_id / event_id
    metadata: dict
```

- タイムラインは **アセットごと**に存在し、マスタータイムラインへの写像は `offset_seconds`（`sync.py` の値）と `drift_ratio` で表す（`multicam.py` の `(t - offset) * ratio` と同じ式を採用）。
- Phase 1 で生成する Event: `AUDIO_SILENCE`（silence.py）、`LOUDNESS_MEASURE`（点、metadata に LUFS/TP/LRA）、`USER_DECISION`。
- Phase 3 で追加: `SCENE_CHANGE`, `AUDIO_PEAK`（scenes.py）、`SPEECH`（whisper）、`SLIDE`, `SPEAKER`, `CAMERA_STATE`, `INCIDENT`。

### 7.2 クエリ

`Timeline.query(type=None, between=(a,b), overlaps=range, source=None, kind=None)` を Phase 1 で実装。「speaker A の区間」「slide 12 の区間」「2時刻間の incident」「speech event 中の camera state」は全てこの1関数 + `type` で表現できる。

### 7.3 ffmpeg-skill との対応 [FACT]

- `silence.py --list --json` の `silences` / `keep` → `AUDIO_SILENCE` / `AUDIO_ACTIVE`。
- `scenes.py --json` の `scenes` / `audio_peaks` → `SCENE` / `AUDIO_PEAK`。
- `multicam.py` の `cuts [[s,e,cam]]` → `CAMERA_STATE`（実行結果としての観測）。
- `sync.py` の `offset_seconds` / `drift.drift_ppm` / `confidence` → アセット間タイムライン写像。

---

## 8. Event / Session / Project / Production モデル [DESIGN]

```
Production (id, name, profile, events[])
  Event (id, name, date, sessions[])
    Session (id, name, kind: PRESENTATION|QA|BREAK|OTHER, range on master timeline, speakers[], assets[])
```

- **generic な単一動画プロジェクトでは Production/Event/Session を作らない**。`ProjectIR.project.kind = "single"` で `assets` と `timeline` だけを持つ。
- `kind = "production"` のときのみ上記ツリーが `ProjectIR.project.production` に入る。
- Session の境界は Event（`SESSION_BOUNDARY`）として timeline に置き、両者を二重管理しない（Session はイベント ID 参照）。

---

## 9. Skill / Capability / Tool モデル [DESIGN]

### 9.1 Skill

```python
@dataclass
class SkillSpec:
    name: str                       # "silence_cleanup"
    version: str
    description: str
    inputs: dict[str, str]          # {"asset": "video|audio"}
    outputs: dict[str, str]
    required_capabilities: list[str]  # ["ffmpeg", "ffmpeg-skill", "encoder:libx264"]
    risk_level: Literal["LOW","MEDIUM","HIGH"]
    deterministic: bool
    approval: Literal["AUTO","CONFIRM","BLOCK"]   # 既定値。Decision で上書き可
    tools: list[str]                # ["ffmpeg-skill/silence", "ffmpeg-skill/cut"]
```

Phase 1 の Skill: `media_probe`, `silence_cleanup`（leading/trailing の技術的無音のみ）, `loudness_normalization`, `delivery_export`, `delivery_check`。

### 9.2 Capability

`CapabilityResolver.resolve() -> dict[str, Capability{status, detail, evidence}]`。検出方法 [DESIGN, ffmpeg-skill には無い]:

| capability | 検出 |
|---|---|
| `python` | `sys.version_info` |
| `ffmpeg`, `ffprobe` | `shutil.which` + `-version` 先頭行 |
| `encoder:libx264/libx265/prores_ks/h264_videotoolbox/h264_nvenc/hevc_nvenc/libaom-av1/libsvtav1` | `ffmpeg -hide_banner -encoders` |
| `decoder:hevc/h264/av1/prores` | `ffmpeg -decoders` |
| `filter:subtitles/ass/zscale/tonemap/loudnorm/scdet` | `ffmpeg -filters`（libass → `subtitles`/`ass`、zimg → `zscale`） |
| `gpu` | nvenc/videotoolbox/vaapi エンコーダの有無から DEGRADED/AVAILABLE を推定（UNKNOWN も許容） |
| `font:cjk-ja` | `fc-list :lang=ja` が使えれば AVAILABLE、`fc-list` 自体が無ければ UNKNOWN |
| `ffmpeg-skill` | 設定パス（`VIDEO_AGENT_FFMPEG_SKILL_DIR` / `~/.claude/skills/ffmpeg-skill` / 同梱 vendored）で `scripts/probe.py` と `package.json` を確認、version を記録 |
| `asr:whisper` | `whisper-cli` / `faster_whisper` import / `whisper` の有無 |
| `ai:<provider>` | API キー環境変数の有無（値は記録しない） |

### 9.3 Tool

`ToolAdapter` 基底: `describe() -> ToolCatalog`, `dry_run(op) -> CommandPreview`, `run(op, workspace, timeout) -> ToolResult`。Phase 1 実装は `FfmpegSkillAdapter` のみ。Tool ID は `ffmpeg-skill/<script名>`。

### 9.4 選択順序

Skill → required_capabilities を CapabilityResolver で確認（MISSING なら BLOCK 決定を生成）→ Skill が列挙する tools から最初の AVAILABLE を選ぶ（Phase 1 は候補が1つ）。

---

## 10. Decision モデル [DESIGN]

```python
@dataclass
class Decision:
    id: str
    subject: str                 # "silence.leading", "delivery.codec", "capability.libx264"
    decision: str                # "trim_leading 0.0-8.2"
    reason: str
    confidence: float
    evidence: list[str]          # observation/inference/requirement ids
    alternatives: list[Alternative]  # {decision, reason, cost}
    risk: Literal["LOW","MEDIUM","HIGH"]
    approval: Literal["AUTO","CONFIRM","BLOCK"]
    provenance: Literal["USER","SYSTEM","PROFILE","DEFAULT","OBSERVED","INFERRED","AI_GENERATED"]
    status: Literal["PROPOSED","APPROVED","REJECTED","BLOCKED"]
```

risk と confidence は独立。approval の決定則（Phase 1、profile で上書き可）:

| ケース | risk | approval |
|---|---|---|
| コーデック / コンテナ / プラットフォーム形式（`check.py kind=format` 相当） | LOW | AUTO |
| 先頭・末尾の技術的無音（発話区間の外側、閾値以下が ≥ N 秒） | LOW | AUTO（profile=conference では CONFIRM） |
| 中間の無音除去（発話の間） | MEDIUM | CONFIRM |
| 尺変更 / クロップ / fps 低下（`check.py kind=judgement`） | MEDIUM | CONFIRM |
| 意味的削除（発話内容に基づく） | HIGH | CONFIRM（conference では既定 BLOCK 候補、後述） |
| 必要 capability が MISSING | — | BLOCK |
| 予算超過 | — | CONFIRM または BLOCK（budget policy による） |

説明可能性: `video-agent explain <project.json> [--decision ID]` は Decision の reason/evidence/alternatives をそのまま表示する（Phase 1 で実装、追加計算なし）。

---

## 11. Policy / Preference / Constraint モデル [DESIGN]

```python
@dataclass
class Rule:
    id: str
    kind: Literal["POLICY","PREFERENCE","CONSTRAINT"]
    scope: Literal["GLOBAL","ORGANIZATION","EVENT","PROJECT","PROFILE","REQUEST"]
    key: str            # "audio.loudness.target_lufs", "edit.semantic_deletion.approval", "workspace.output_root"
    value: Any
    source: str         # ファイルパス / "request" / "feedback:<id>"
    hard: bool          # CONSTRAINT は常に True
```

- 解決順序（MASTER_SPEC §17 の案を採用）: GLOBAL → ORGANIZATION → EVENT → PROJECT → PROFILE → REQUEST の順に上書き。ただし **CONSTRAINT は上書き不可**で、REQUEST が CONSTRAINT と矛盾した場合は上書きせず `Conflict` を生成し、Decision を CONFIRM/BLOCK にする（「曖昧なときに precedence を発明しない」）。
- Preference は Decision の「既定値」にしか影響せず、Constraint は Validator で強制される。
- Feedback は `PREFERENCE_CANDIDATE` としてのみ記録し、`video-agent policy promote <candidate_id>` の明示操作なしに Rule にならない。
- Phase 1 の Constraint（固定）: `preserve_source`（source パスへの書き込み禁止）, `workspace_boundary`（入力は allowed_inputs 配下、出力は workspace 配下）, `no_raw_shell`。

---

## 12. Project IR [DESIGN]

### 12.1 基本方針

- JSON、`schema_version: "1.0"`。`schemas/project.schema.json`（JSON Schema draft 2020-12）で検証。
- `render.py` の project.json とは別物。互換は取らない（1.3 の理由）。将来 `lowering` で変換可能。
- IR は「宣言」であり、コマンドは含まない。ffmpeg 引数、`--crf` 等の tool 固有値も含まない（profile → compiler で決まる）。

### 12.2 トップレベル（MASTER_SPEC §22 をそのまま採用、Phase 1 で埋めるものに ✓）

| キー | 内容 | Phase 1 |
|---|---|---|
| `schema_version` | "1.0" | ✓ |
| `project` | `{id, kind: "single"|"production", name, created_at, profile: {name, version}}` | ✓ |
| `request` | `{raw, received_at, channel}` | ✓ |
| `requirements` | `[{id, key, value, provenance}]` | ✓ |
| `source` | `{tool_versions, agent_version, generator}` | ✓ |
| `assets` | `{<asset_id>: Asset}` | ✓ |
| `analysis` | `{observations: [...], inferences: [...], strategy: FULL|COARSE|TARGETED, budget}` | ✓ |
| `intent` | `{primary, secondary[], confidence, provenance}` | ✓ |
| `constraints` / `policy` | 解決済み Rule 一覧（provenance 付き） | ✓ |
| `decisions` | Decision[] | ✓ |
| `plan` | `{version, steps: [{id, skill, tool, decision_ids, params}], summary(human readable)}` | ✓ |
| `timeline` | `{timelines: {id: {asset_id, offset, drift_ratio}}, events: Event[]}` | ✓（silence/loudness のみ） |
| `video` | `{operations: [ {type: trim|fit|..., ...} ]}` 映像側 operation | ✓ trim のみ |
| `audio` | `{operations: [ {type: loudness_normalize, target_lufs, true_peak} ]}` | ✓ |
| `captions` / `graphics` / `color` | 空オブジェクト（スキーマ定義のみ） | 定義のみ |
| `delivery` | `{targets: [{id, preset(ffmpeg-skill export preset名), platform(check.py platform名), filename_rule}]}` | ✓ |
| `qa` | `{required: [...], thresholds}` | ✓ |
| `execution` | `{workspace, dry_run, budgets, recovery_policy}` | ✓ |
| `provenance` | `{source_hashes, profile_version, skill_versions, tool_versions, created_by}` | ✓ |

### 12.3 operation の語彙（Phase 1）

```
video.trim         {asset, keep: [[start,end], ...]}          → ffmpeg-skill/cut (--segments, --accurate は VFR/精度要件で compiler が決める)
audio.loudness     {target_lufs, true_peak, lra}              → ffmpeg-skill/loudness
delivery.export    {target_id, preset}                         → ffmpeg-skill/export
qa.check           {target_id, platform}                       → ffmpeg-skill/check
```

将来語彙（スキーマで `type` を enum 化せず `additionalProperties` で拡張可能にしておくが、Validator は未知 type を **WARN ではなく ERROR** にする。Phase 1 の安全側）。

### 12.4 マイグレーション

`project/migrations.py` に `MIGRATIONS = {"1.0": None}` の辞書を置き、`load()` が古い版を順次適用する。Phase 1 では 1.0 のみ。

---

## 13. Compiler / Adapter [DESIGN]

### 13.1 Compiler

`compile(ir) -> OperationPlan`。IR の `video.operations` / `audio.operations` / `delivery.targets` / `qa` を **固定順序**（trim → loudness → export → check）で `Operation` に落とす。各 Operation:

```python
@dataclass
class Operation:
    id: str
    tool: str                     # "ffmpeg-skill/cut"
    args: dict                    # adapter の型付き引数（argv ではない）
    inputs: list[str]             # artifact ids
    outputs: list[str]            # artifact ids（パスは workspace/<job>/<op_id>/...）
    decision_ids: list[str]
    idempotency_key: str          # sha256(source_hash + op args + tool version)
```

順序決定・中間ファイルのパス決定・tool 固有パラメータ（`--accurate`, `--crf` 等）の決定は compiler の責務。**ffmpeg のオプションはここにも現れない**（adapter の args まで）。

### 13.2 FfmpegSkillAdapter

- `catalog.py`: script → `{positional, flags: {name: type}, produces_output, json_extra_keys}` を静的定義（1.2 の表の機械可読版）。
- `run()`: `[sys.executable, <skill>/scripts/<name>.py, ...argv, "--json"]` を `subprocess.run(timeout=...)`。stdout 最終 JSON をパース、`commands` を provenance へ、exit code と stderr 末尾を `ToolResult` に格納。
- パス検証: 入力は allowed_inputs 配下、`-o` は workspace 配下。違反は実行前に `ConstraintViolation`。
- `dry_run()`: 入力が存在するなら `--dry-run --json` を実際に呼んで `commands` を返す。存在しない / ffmpeg が無い場合は catalog から argv だけ組み立てて返す（1.3/4 の不一致対応）。
- 将来の別レンダラ: `ToolAdapter` を実装した `NativeFfmpegAdapter` などを追加できるが Phase 1 では作らない。

---

## 14. QA [DESIGN]

| 層 | Phase 1 の実装 | 実行器 |
|---|---|---|
| Video QA | resolution / codec / fps / duration / pix_fmt / aspect を **期待値（IR の delivery target + source 観測）と比較** | `probe.py --json` |
| Audio QA | loudness / true peak / sample rate / channels を期待値と比較、`silent` 検出 | `loudness.py --measure-only`, `probe.py` |
| Delivery QA | プラットフォーム適合 | `check.py --platform --json`（`kind` を保持） |
| Visual QA | コンタクトシート生成のみ（判定は人間） | `look.py` |
| Semantic QA | 未実装（スキーマ上のプレースホルダ） | — |

黒フレーム / フリーズ / クリッピング / チャンネル欠落は ffmpeg-skill に無い [FACT]。Phase 3 で `blackdetect` / `freezedetect` / `astats` を ffmpeg-skill 側に `incidents.py` として提案するか Agent 側の `media/` に置くかは [UNKNOWN]（推奨: ffmpeg-skill 側に PR）。

QA 結果は `QAReport{items: [{name, status: PASS|WARN|FAIL, observed, expected, kind: format|judgement, fix_hint}], incidents: Incident[]}`。FAIL があれば artifact は `candidate` のまま `final` に昇格しない。

---

## 15. Recovery [DESIGN]

```
ToolResult(exit != 0)
  → classify(stderr, exit code, op) → ErrorClass
       TOOL_MISSING(127) / INPUT_MISSING / INVALID_ARGS / ENCODER_FAILED / TIMEOUT / DISK_FULL / UNKNOWN
  → strategy = RECOVERY_TABLE[ErrorClass]
       TOOL_MISSING     → BLOCK（doctor を案内）
       INPUT_MISSING    → BLOCK
       INVALID_ARGS     → BLOCK（Agent のバグ扱い、再試行しない）
       ENCODER_FAILED   → 1回だけ代替パラメータ（例: cut の --accurate 付与）で再試行、失敗なら BLOCK
       TIMEOUT          → タイムアウト2倍で1回再試行（budget 内）、失敗なら BLOCK
       DISK_FULL        → BLOCK
       UNKNOWN          → 同一引数で1回再試行、失敗なら BLOCK
  → 全試行を provenance.recovery[] に記録（attempt, strategy, result）
```

最大試行回数は `execution.recovery_policy.max_attempts`（既定 2）。QA FAIL は Recovery ではなく **Revision**（Plan の見直し）に回す（例: loudness FAIL → loudness operation を追加した Plan v2 を提案、CONFIRM）。

---

## 16. Artifact / Job / Lifecycle [DESIGN]

- `Artifact{id, path, type: MASTER|WEB|YOUTUBE|SOCIAL|ARCHIVE|CAPTIONS|THUMBNAIL|REPORT|INTERMEDIATE, hash(sha256), source_asset_ids, generation, tool, tool_version, created_at, qa_status, stage: working|candidate|approved|final|archive}`。
- `Job{id, state, ir_path, workspace, created_at, updated_at, history: [{state, at, reason}]}`。状態は MASTER_SPEC §37 の集合。Phase 1 の遷移: QUEUED → INGESTING → ANALYZING → PLANNING → (WAITING_FOR_APPROVAL) → EXECUTING → QA → (RECOVERY) → COMPLETED | FAILED | BLOCKED | CANCELLED。
- Workspace: `<workspace_root>/jobs/<job_id>/{ir.json, ops/<op_id>/, artifacts/, qa/, provenance.json, job.json}`。source は読み取りのみ。
- 冪等性: `Operation.idempotency_key` が一致し出力 artifact のハッシュが記録済みならスキップ（Phase 1 は「同一 job 内の再実行」のみ。job 間キャッシュは Phase 5）。
- キャンセル: Executor は operation 境界でフラグを確認し、進行中の subprocess を terminate → `CANCELLED`。中間ファイルは残す（破壊的部分状態を避ける）。resume は `job.json` の完了 op 一覧から Phase 2 以降で実装。

---

## 17. Feedback / Revision [DESIGN]

```python
@dataclass
class Feedback:
    id: str
    job_id: str
    plan_version: int
    target: str            # decision_id | op_id | "global"
    text: str
    structured: dict | None   # 例 {"key": "audio.loudness.target_lufs", "value": -16}
    created_at: str
```

フロー: `video-agent revise <job> --feedback "..."` → Feedback 保存 → `PreferenceCandidate` 生成（永続化しない） → Planner が Plan v(n+1) を生成 → `PlanDiff` を機械可読で出力 → 承認 → 実行。

PlanDiff（Phase 1）は IR の `decisions` / `video.operations` / `audio.operations` / `delivery.targets` の JSON 差分（追加 / 削除 / 変更、パス付き）。人間向け要約（`AUDIO -14 → -16 LUFS`）は差分から生成。Plan の各版は `plans/v<n>.json` として job 内に保存し、上書きしない。

---

## 18. Conference Profile [DESIGN]

Phase 2 の設計だが、Phase 1 で境界を確保する。

- `profiles/conference.json`（Phase 1 は骨組み）:

```json
{
  "name": "conference", "version": "0.1",
  "extends": "generic",
  "delivery": {"targets": [{"id": "master", "preset": "prores", "platform": "broadcast"},
                            {"id": "web", "preset": "youtube", "platform": "youtube"}]},
  "audio": {"target_lufs": -16, "true_peak": -1.5},
  "decisions": {"silence.leading": "CONFIRM", "silence.internal": "CONFIRM", "semantic_deletion": "BLOCK_UNLESS_EXPLICIT"},
  "naming": "{event}_{session}_{speaker}_{date}_{version}_{format}",
  "asset_roles": ["camera_a", "camera_b", "camera_c", "presentation", "slides", "presentation_audio", "room_audio", "logo"]
}
```

- Asset 分類（Phase 2）: 拡張子 / probe（video 無し→AUDIO 候補、画面録画的解像度→SCREEN_CAPTURE 候補、PNG→LOGO/IMAGE）/ ファイル名パターン / ユーザ指定。各分類は `confidence` と `evidence` 付きの Inference。
- パイプライン段階と対応する Skill / Tool:

| 段階 | Skill | Tool（既存 / 要追加） |
|---|---|---|
| 同期 | `multi_source_sync` | `sync.py`, `multicam.py --offsets-only` [FACT 既存] |
| セッション分割 | `session_segmentation` | `scenes.py` + `silence.py` の長い無音 + ユーザ入力 [DESIGN] |
| スライド検出 | `slide_detection` | `scenes.py`（画面キャプチャに対して）→ Phase 4 で PowerPoint パーサ [要追加] |
| 話者検出 | `speaker_detection` | ASR + 話者分離 [要追加、Phase 4] |
| カメラ選択 | `camera_selection` | 判断は Agent、合成は `multicam.py --switch` [FACT 既存] |
| 音声 | `audio_cleanup`, `loudness_normalization` | `audio.py`, `loudness.py` |
| 字幕 | `caption_generation` | `caption.py --transcribe/--srt` |
| チャプタ | `chapter_generation` | Session 境界からメタデータ生成 [要追加] |
| マスタ / Web / YouTube | `delivery_export` | `export.py` |
| QA / 納品 | `delivery_check` | `check.py`, `look.py` |

- 医学系の安全策（§49）: `semantic_deletion` Skill は conference profile では既定 `BLOCK_UNLESS_EXPLICIT`（Request に明示の削除指示がある場合のみ CONFIRM に緩和）。数値・薬剤名・氏名を含む発話区間は将来 ASR で検出し `PROTECTED` Event として timeline に置き、その区間に重なる trim は CONFIRM 以上に強制する。

---

## 19. Phase Roadmap [DESIGN]

| Phase | 内容 | ffmpeg-skill への依存 |
|---|---|---|
| 1 | Request → Requirements → Intent → Analysis(probe/silence/loudness) → Decision → Plan → IR → Validation → Execute(cut/loudness/export) → QA(probe/check) → Report。CLI 9コマンド。generic / youtube profile。 | probe, silence, loudness, cut, export, check, look |
| 2 | conference profile、Asset 分類、Production/Event/Session、sync/multicam offsets、job resume | + sync, multicam |
| 3 | scenes、transcription（whisper）、captions、chapters、highlights、incident 検出（blackdetect/freezedetect/astats） | + scenes, caption；incident は要追加 |
| 4 | PowerPoint、semantic editing、AI multicam、slide sync、speaker、thumbnails、YouTube/archive package | 要追加多数 |
| 5+ | Review UI、diff UI、queue、batch、分散、AI providers、OCR、知識化 | batch.py 等 |

---

## 20. 技術リスク [FACT + DESIGN]

| リスク | 根拠 | 対処 |
|---|---|---|
| ffmpeg-skill の CLI 契約が非公式（semver 化されていない） | `--json` のキーは script ごとに異なり、スキーマ文書が無い | adapter の catalog にバージョン固定（`>=0.8.4,<0.9`）を持ち、起動時に `package.json` を検査。契約テスト（`tests/contract_ffmpeg_skill.py`）で `--help` と `--json` キーを固定。 |
| 各段が別プロセス・別再エンコード（CRF18 の世代劣化、時間） | `render.py` も同様 | Phase 1 は trim(copy優先) → loudness(copy) → export の3段で再エンコードは export の1回に抑える。 |
| `--dry-run` が ffmpeg と実ファイルを要求 | 1.3, 4 | Agent の dry-run は IR ベース。 |
| 入力=出力の上書き事故 | `-y` 固定、保護無し | adapter の workspace 強制。 |
| VFR / HDR の自動挙動が IR の意図と食い違う | `cut.py` は VFR で勝手に `--accurate`、HDR は HEVC Main10 維持 | Observation（`variable_frame_rate_suspected`, `hdr`）を Decision に反映し、期待値 QA で検出。 |
| 長尺（学会 1–3 時間）の処理時間 | devices.md「10分超はトーンマップ全体禁止」 | 分析は `TARGETED`（先頭/末尾 N 秒 + サンプリング）を既定、`max_processing_time` 予算。 |
| 日本語フォント無し→豆腐 | フォント検出はコードに無い | doctor / CapabilityResolver で `font:cjk-ja`、caption Skill の required_capability に。 |
| Windows パス / `fc-list` 非存在 | ffmpeg-skill は 3OS 対応、`fc-list` は Linux/mac 前提 | capability は UNKNOWN を返す設計。 |
| AI 推論の混入 | 禁止事項「推測を事実に」 | Phase 1 は AI 無し。Inference は常に別型。 |

---

## 21. 未解決の決定 [UNKNOWN]

1. ffmpeg-skill の配置: `~/.claude/skills/ffmpeg-skill` を参照するか、git submodule / vendored copy を同梱するか。**推奨: 環境変数 → 既定パス → submodule の順で探索し、doctor で報告**（Phase 1 は探索のみ実装、submodule は未追加）。
2. ffmpeg-skill 側へ出す PR の要否: `probe.py --streams`（全ストリーム）、`incidents.py`（black/freeze/astats）、`--json` に version を含める。→ Phase 2 開始時に判断。
3. 複数音声トラック（学会の room/presentation 音声）を扱う方法（ffmpeg-skill は最初のトラックのみ）。
4. Semantic QA / ASR エンジンの選定（whisper.cpp vs faster-whisper）。
5. Job の永続化形式（Phase 1 は JSON ファイル。SQLite への移行時期）。
6. AI プロバイダ利用時の Request 解釈の責務分割（構造化抽出のみか、Plan 生成まで任せるか）。推奨: 構造化抽出のみ。
7. Windows 対応の優先度（現場 PC は Windows が多い可能性）。

---

## 22. Phase 1 最小スコープ [DESIGN]

実装するもの:

- `pyproject.toml`（依存: `jsonschema` のみ。理由: IR 検証を手書きしない。それ以外は標準ライブラリ）。
- `models/`（Request, Requirement, Intent, Observation, Inference, Decision, Asset, Event, Incident, Artifact, Job）。
- `capabilities/resolver.py` + `video-agent doctor`。
- `tools/ffmpeg_skill/{catalog,adapter}.py`（probe, silence, loudness, cut, export, check, look）。
- `media/analyzer.py`（probe → Asset/Observations、silence --list → Events、loudness --measure-only → Observation、`TARGETED` 戦略）。
- `agent/`（RequirementExtractor: CLI 引数 + profile + defaults、IntentResolver: ルール、Inferencer: leading/trailing silence、DecisionEngine、Planner）。
- `project/`（IR dataclass、`schemas/project.schema.json`、validator（schema + semantic + capability）、migrations 骨組み）。
- `execution/`（compiler、executor、recovery）。
- `qa/`（video/audio/delivery QA、Incident 生成）。
- `jobs/`（state machine、workspace）。
- `profiles/{generic,youtube,conference}.json`（conference は骨組み）。
- `audit/provenance.py`。
- `cli.py`: `analyze`, `plan [--profile]`, `validate`, `render [--dry-run]`, `check`, `doctor`, `explain`。
- `tests/`: unit（ffmpeg 不要、adapter はフェイク）、integration（ffmpeg + ffmpeg-skill 必須、合成素材）、contract（ffmpeg-skill の `--help`/`--json` キー）。
- `evals/`: 2〜3ケース（期待 intent / decisions / warnings）。
- docs: 本レビュー、`project-ir.md`、`decisions.md`（ADR）。

実装しないもの（明示）: 自然言語理解（AI）、conference パイプライン本体、multicam、captions、Web UI、job queue、incremental cache、submodule 同梱。
