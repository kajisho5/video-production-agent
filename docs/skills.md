# AI Video Production Ecosystem — Skill / Tool / Capability / Adapter / Registry / Router

```text
video-production-agent = Brain / Orchestrator
ffmpeg-skill           = First Reference Skill（deterministic media processing）— 実装済みの Skill package
external Skills        = media-analysis-skill / transcription-skill / video-editing-skill /
                          audio-production-skill / color-grading-skill / motion-graphics-skill /
                          qc-skill / subtitle-skill / thumbnail-skill — いずれも実装済み、各リポジトリの
                          checkout をチェックアウト場所（env var）で見つけたときに tools/<skill>/adapter.py
                          経由で接続される（`service.py` の `register_package` 呼び出し一覧が現状の正）。
                          image-skill は別ドメイン（画像編集、動画パイプライン外）で、この Orchestrator の
                          adapter は未実装（AI-video-production-OS#39）。
```

（このファイルはこれら10リポジトリ+この1リポジトリの現状を反映する生きたリファレンスとして扱う。
`docs/GAP_ANALYSIS_PHASE2.md` などの日付付き監査記録は、その時点のスナップショットとして意図的に
書き換えていない — 2026-09-04 時点で「ffmpeg-skill のみ」だったのは事実だが、現在の事実ではない。）

## 責務

| 概念 | 意味 | 実装 | 例 |
|---|---|---|---|
| **Skill package** | 何ができるか（独立した repository / capability domain） | `skills/contract.py` の `SkillPackage`（skill_id / name / version / description / capabilities / tools / repository / role） | `ffmpeg-skill` |
| **Tool** | その Skill package が提供する具体的操作 | `ToolSpec`（tool_id `<skill_id>/<name>` / skill_id / version / required_capabilities / 実行契約: inputs, produces_output, deterministic, result_keys） | `ffmpeg-skill/probe`, `ffmpeg-skill/cut`, `ffmpeg-skill/loudness` |
| **Production skill** | Agent が実現できること（環境に依存しない知識） | `skills/registry.py` の `SkillSpec`（required_capabilities / risk / approval / tools 候補 / phase） | `silence_cleanup` |
| **Capability** | 実行環境が今何をサポートしているか | `capabilities/resolver.py`（AVAILABLE / MISSING / DEGRADED / UNKNOWN） | `ffmpeg`, `encoder:libx264`, `ffmpeg-skill` |
| **Adapter** | Tool を実際の runtime に接続する | `tools/base.py` の `ToolAdapter`（`package()` / `supports` / `preview` / `run` / `measure`）。現在は `FfmpegSkillAdapter` のみ | `tools/ffmpeg_skill/` |
| **Registry** | Skill package と production skill を登録・列挙し、環境ごとに tool を選ぶ | `SkillRegistry` | — |
| **Router** | 選択された tool id を対応 adapter へ dispatch する | `tools/router.py` の `ToolRouter` | — |
| **Agent** | Production 全体として何をすべきか判断する | `agent/`, `service.py` | — |
| **AI Provider** | reasoning / model interface。production intent と evidence 付き inference を提供する。Skill / Tool 選択・command 生成・approval には関与しない | `providers/base.py`（`AIProvider` / `AIRequest` / `AIResponse` / `AIUsage`）、`agent/ai_reasoning.py` | `NullProvider`（既定）、テストの `FakeAIProvider` |

Skill と Tool を混同しない: `ffmpeg-skill` は package、`ffmpeg-skill/cut` は tool。Tool id は概念名であり、実装 engine（ffmpeg / sox / …）は adapter の内側。

AI ≠ Tool executor / AI ≠ Skill registry / AI ≠ Compiler / AI ≠ final execution authority。AI は production intent（registry の production skill 名）を evidence 付きで提案するだけで、
どの Skill / Tool を使うかは SkillRegistry / CapabilityResolver / ToolRouter が決める（ADR-018、MASTER_SPEC §42）。将来 Skill が増えても AI 側の契約は変わらない。

## 選択と実行の流れ（PR #5 の境界を維持）

```text
SkillSpec.required_capabilities ──CapabilityResolver──▶ 欠落があれば BLOCK 決定
SkillSpec.tools（候補）          ──ToolRouter.supports──▶ 実行可能な候補が0/1件ならそのまま選択。2件以上なら Provider 衝突ポリシー（ADR-037）へ
                                                          │
                              Service が skill→tool 表を planner / analyzer / QA に渡す（必須引数、既定 engine 無し）
                                                          │
                                   plan.steps[].tool に記録 ─▶ compiler は plan の tool だけを使う ─▶ ToolRouter ─▶ Adapter ─▶ runtime
```

- `SkillRegistry.select_tool(skill, caps, supports, explicit=None, default=None)` が唯一の選択関数。`resolve_tools()` はその一括版。実行可能な候補が2件以上（Provider 衝突）のときだけ `explicit`（Tier 1、`provider.<skill>=<package>` requirement）→ `default`（Tier 2、`skills/providers.py` の `default_providers()`: OS 既定を workspace の `providers.json` で上書き）→ 拒否（Tier 3、`select_tool` が None を返し理由に候補一覧と設定方法を含める）の順で解決する（ADR-037、CAPABILITY_MODEL.md の collision policy）。候補が0/1件の skill はこの3段を一切参照しない（選ぶものが無い）。
- planner / analyzer / QA は `tools`（skill→tool 表）を**必須引数**として受け取る。既定の engine 表（旧 `DEFAULT_TOOLS`）は存在せず、`tools=None` は `TypeError`、必要な Skill が表に無ければ `ToolError`（analyzer / QA）または tool 無し step + BLOCKED summary（planner）になる。
- validator は step の skill が実装済みで、tool がその Skill の候補に含まれ、adapter が対応し、**登録済み package が宣言している tool** であることを検査する。必要 capability は registry（production skill + package + ToolSpec）から集める。
- `source.tool_versions` は package（tool id の prefix）→ version。冪等キー、provenance の `tool_version` / `skill_package`、artifact の `tool` はすべて operation の tool から引く。
- tool 固有の知識（`recovery.py` の代替引数、catalog の型）は tool 層に閉じる。静的テストが `agent/ execution/compiler.py qa/ media/ project/` に engine 名が出ないことを検査する。

## Contract

### Skill Package Contract（`SkillPackage`）
必須: `skill_id`（tool id の prefix、`/` を含まない）、`name`、`version`（検出時に adapter が埋める）、`description`、`capabilities`（package 全体が必要とする runtime capability）、`tools`（1 件以上）。任意: `repository`、`role`。`validate()` が契約違反を返し、`register_package` は違反を拒否する。

### Tool Contract（`ToolSpec`）
`tool_id`（`<skill_id>/<name>`）、`skill_id`、`version`、`required_capabilities`（tool 固有の追加分）、実行契約 `inputs` / `produces_output`（transform か measure か）/ `deterministic` / `result_keys`（adapter が `ToolResult.data` に保証するキー）。

### Adapter Contract（`ToolAdapter`）
`name` = package の skill_id（対応する tool id の prefix）。`package()` は実装する `SkillPackage`（検出 version 入り）を返す。`supports` / `preview` / `run` / `measure` は呼び出し側が選択した tool を実行するだけで、Skill 選択・IR 参照・production 判断をしない。シェルを経由せず、入力を上書きせず、出力は workspace 内（`PathPolicy`）。

### Registry Contract（`SkillRegistry`）
`register` / `get` / `names` / `all`（production skill）、`register_package` / `packages` / `package` / `tool`（package）、`missing_capabilities` / `select_tool` / `resolve_tools` / `availability` / `package_availability`（環境ごとの解決）、`unknown_tool_candidates`（実装済み skill が未登録 package の tool を引用していないか）。Registry は production decision を行わない。

### Status
`DECLARED`（= `NOT_IMPLEMENTED`: 後 phase 用に宣言のみ）/ `IMPLEMENTED`（`SkillSpec.implemented`、package は adapter が本コードに存在）/ `AVAILABLE`（この環境で tool が選択できた）。新しい status は作らない。

## 将来 Skill package を 1 つ追加するときに必要な変更

1. `tools/<package>/` に `ToolAdapter` 実装（`package()` が `SkillPackage` を返す、catalog 型付き引数、`supports("<package>/<tool>")`、`run` / `preview` / `measure`）。
2. `capabilities/resolver.py` に検出項目を追加（package の所在と version、必要な外部ツール）。
3. `skills/registry.py` の該当 production skill に `tools` 候補を追記（新しい production skill なら `SkillSpec` を 1 件追加）。既存候補と衝突する（両方とも実行可能になり得る）なら、`--set provider.<skill>=<package>` か workspace の `providers.json` で選べることを利用者に伝える（コード側の変更は不要 — ADR-037）。
4. `Service.adapter()` の capability 駆動ループ（ADR-038、Reference Skill である ffmpeg-skill 自身を除く9 Skill 全件が1つのタプルのリストで登録されている）に `(locate_fn, constructor)` を1行追加する（package は adapter から自動登録される）。
5. その Skill が生成する新しい operation 語彙があれば `schemas/project.schema.json` と planner / compiler の該当分岐を追加（既存語彙を別 engine で実現するだけなら不要）。
6. 契約テスト（`--help` / JSON キー）と、Registry → plan.steps → compiler → Router → adapter → provenance の伝播テスト。

planner / compiler / decision / QA の engine 固有ロジックを変更する必要がある構造は禁止（静的テストで検出）。`tests/test_unit.py::EcosystemContractTests` が fake-skill package でこれを証明している（test scope のみ、production registry には登録しない）。

## 今あるもの / 無いもの

- 実装済み・利用可能: `kajisho5/ffmpeg-skill`（0.8.4 ≤ v < 0.10、契約テスト + unit テストで固定。0.9.0 は contract / doctor 追加と `--json` の "status" のみで media 挙動は不変、integration 全件通過を確認済み。0.10 は未検証のため拒否）。`video-agent skills` の "Skill packages" に表示される。外部 Skill として media-analysis-skill（ADR-023）、transcription-skill（ADR-024）、video-editing-skill（ADR-028、0.1.x、CLI contract 境界。`VIDEO_AGENT_VIDEO_EDITING_DIR`）も adapter 実装済み（インストールされていれば利用可能）。video-editing-skill の operation は四段階で区別する: Skill supports（contract 0.4.0 の 12 operation。v0.4.0 の release tag で IMAGE_INSERT が増えた）/ adapter supports（`Lowering.ARGS` の 12 operation。ROTATE / FREEZE / REVERSE は ADR-062、IMAGE_INSERT は ADR-070）/ Planner can generate（`video.trim` と ADR-029 の `video.concat` / `video.speed` / `video.resize` / `video.fit` / `video.fill` / `video.overlay`、ADR-062 の `video.rotate` と明示 `edit.trim` の `video_trim`（`video-editing/trim`）、ADR-070 の `video.image_insert`（`edit.image_insert`。step `video_image_insert`（`video-editing/image_insert`）+ つなぎの `video_concat`（`video-editing/concat`））。production skill `video_concat` … `video_overlay` / `video_rotate` / `video_trim` / `video_image_insert`、tool は `video-editing/<op>` のみ、approval 既定 CONFIRM）/ E2E verified（実メディア integration: trim → concat → speed → resize → fit・fill → overlay、明示 trim → rotate、trim → resize → 静止画の挿入（ADR-070））。audio-production-skill（ADR-030、0.1.x、`VIDEO_AGENT_AUDIO_PRODUCTION_DIR`、ffmpeg-skill ≥ 0.9.1）も同じ四段階で区別する: Skill supports 15 operation（0.4.3 で VOICE_CLEANUP）/ adapter supports 15 / Planner generates 13（`audio_cut` / `audio_normalize` / `audio_gain` / `audio_mono` / `audio_stereo` / `audio_downmix` / `audio_fade_in` / `audio_fade_out` / `audio_concat`、ADR-063 の `audio_mix` / `audio_noise_reduction` / `audio_dynamics`、ADR-068 の `audio_voice_cleanup`、tool は `audio-production/run` のみ、operation ごとの capability `audio-production:<TYPE>`。残る SILENCE_REMOVE と単独 TRIM は計画しない）/ E2E verified（音声のみ・video container・2 入力 concat + normalize、2 入力 mix → noise reduction → dynamics）。テイクの組み立てのつなぎ目（ADR-054）では、`audio_cut`（span の切り出し）と `audio_concat`（crossfade）を video の経路の横で使い、新しい skill `audio_mux`（tool `ffmpeg-skill/audio`、`--replace` のみ）で組み立てた絵の音声を差し替える。
- Phase 3（ADR-031 / ADR-032、PR #22）: subtitle-skill（`VIDEO_AGENT_SUBTITLE_DIR`、package `subtitle`、tool `subtitle/generate` `subtitle/render`）、thumbnail-skill（`VIDEO_AGENT_THUMBNAIL_DIR`、package `thumbnail`、`thumbnail/extract_frame` `thumbnail/render`、Pillow 必須）、color-grading-skill（`VIDEO_AGENT_COLOR_GRADING_DIR`、`color-grading/run`、operation capability `color-grading:<TYPE>`）、motion-graphics-skill（`VIDEO_AGENT_MOTION_GRAPHICS_DIR`、`motion-graphics/run`、element capability `motion-graphics:<type>`）、qc-skill（`VIDEO_AGENT_QC_DIR`、`qc/check`、最終 promotion gate）。共通 transport は `tools/skill_process.py`。四段階: Skill supports（subtitle 2 / thumbnail 3 / color 5（ADR-036 で PRIMARY_CORRECTION 追加）/ motion 4 element / qc 3 operation）/ adapter supports（subtitle 2 / thumbnail 2 / color 5 / motion 4 / qc 2）/ Planner generates（`subtitle_generation` `subtitle_burn_in` `subtitle_mux`（ADR-063、render の mode mux）`thumbnail_frame` `thumbnail_render` `color_strip_dovi` `color_hdr_to_sdr` `color_primary_correction` `color_lut` `color_retag` `motion_graphics` `qc_check`）/ E2E verified（実メディア integration `IntegratedPipelineRealTests`: 10 scenario）。requirement は明示のみ（`subtitle` / `thumbnail` / `color.*` / `motion.*` / `qc`）。固定順: trim → concat → edits → color → graphics → captions → loudness → export → check → thumbnail → qc。
- ffmpeg-skill の catalog は locate 時に Skill 自身の `_contract.py --json --static`（contract_version 1.x）と `doctor --json` で照合する（ADR-056）。catalog の flag が contract に無い tool、doctor が `usable: no` の tool は capability `ffmpeg-skill:<tool>` が MISSING になり、選択理由・plan の handoff に理由が出る。contract が 1.x でなければ package 全体が MISSING。
- measurement skill `sync_analysis`（ADR-044、tool `ffmpeg-skill/sync`、kind `sync`）と production skill `camera_switch`（ADR-045、tool `ffmpeg-skill/multicam`、requirement `edit.switch` = 明示 switch list、`video.switch` IR op、`video.concat` と排他）は実装済み。宣言のみ（NOT_IMPLEMENTED）: `semantic_deletion`（phase 4）。旧 `caption_generation`（ffmpeg-skill/caption 直接参照）は削除し、subtitle-skill が canonical。
- ADR-046（audit report Priority A、39 実スクリプト対 12 宣言のギャップの第一弾）: production skill `video_grid`（tool `ffmpeg-skill/grid`、requirement `edit.grid` + `edit.grid.cols`/`rows`、`video.grid` IR op、`video.concat`/`video.switch` と三者排他、programme 尺は既定で最短入力・`edit.grid.pad` で最長入力）、`video_redact`（`ffmpeg-skill/redact`、requirement `edit.redact`={x,y,width,height}、risk HIGH・既定 CONFIRM）、`video_deinterlace`（`ffmpeg-skill/deinterlace`、risk LOW・既定 CONFIRM——lossy な全体再エンコードのため、当初の AUTO 判定は同日中に訂正済み）、`video_crop`（`ffmpeg-skill/crop`、requirement `edit.crop`={x,y,width,height}、risk MEDIUM・既定 CONFIRM）、`video_stabilize`（`ffmpeg-skill/stabilize`、risk MEDIUM・既定 CONFIRM）は実装済み（`EDIT_ORDER` の末尾、`video.speed` の手前に追加、単一ソース系は互いに自由に組み合わせ可能）。redact/crop は矩形の自動検出を一切行わない（顔・ナンバープレート検出なし、呼び出し側が明示矩形を渡す）。`cropdetect_analysis`（`ffmpeg-skill/cropdetect`、risk LOW・AUTO）も real Skill+Tool として登録したが、意図的に pipeline edit-op ではない（測定のみ・出力ファイルなし、`media/analysis.py` の Observation/AnalysisKind 統合は今回のバッチでは未着手・別途のフォローアップとする）。cropdetect の `--json` 結果キーは当初未検証の仮定だったが、翌日に実 ffmpeg-skill での実測で検証・修正済み（`result_keys = file/source_width/source_height/crop(ネスト: x,y,width,height)/confidence`）。
- ADR-060（ECOSYSTEM_ROADMAP step 7）: `cropdetect_analysis` は analysis kind `crop_rectangle`（`--kind crop_rectangle` で明示的に測る。`--samples 5 --seconds 10 --round 2`）になった。測定は OBSERVED Observation（`source_width` / `source_height` / `crop` / `confidence` / `windows` / `windows_failed`）として残り、安定した矩形（`confidence` ≥ 0.9、3 window 以上を decode、内側へ偶数丸め、幅か高さの 1% 以上を削る）なら `video.crop` の TRANSFORM decision を CONFIRM（floor、policy で AUTO にできない）で提案し、承認されると既存の `video_crop` step（ADR-079 から `video-editing/crop`、以前は `ffmpeg-skill/crop`）が実行される。それ以外は KEEP（理由付き）。明示 `edit.crop`、複数入力の programme、audio path では提案しない。
- 未接続の Skill（ADR-057）: ecosystem.json が他の package に割り当てた仕事は phase 2 の SkillSpec として宣言済み（adapter 無し・選択されない）: `voice_enhancement`（voice-enhance）、`dubbing`、`sound_design`、`speaker_diarization`（diarization）、`visual_detection`（detection）、`slide_sync`、`platform_publish`（publish）、`still_image_edit`（imagemagick-skill）、`render_3d`（blender）、`measured_motion_graphics`（remotion-ffmpeg）、`html_scene_render`（hyperframes）、`document_verification`（artifacts）、ディスク作成は ecosystem.json どおり 4 つ: `bluray_authoring`（bdmvkit）、`dvd_authoring`（vtskit）、`disc_image`（udfkit）、`dvd_line21_captions`（dvdcc）。`--set job.<skill>=true` で plan に BLOCKED の step が入り、handoff が package・repository・インストール方法（`skills/ecosystem.py` の ecosystem.json snapshot と各 README）を名指しする。adapter は roadmap step 9 で 1 Skill ずつ（ingest は ADR-066 で `media_offload` として実装済み）。
- ingest-skill（ADR-066）: カード / レコーダーのフォルダを入力にした plan は offload の plan になる（`ingest.dest` 必須、`ingest.project` / `ingest.hash` / `ingest.name_by` / `ingest.reel`）。plan では `ingest offload --dry-run` と `ingest inventory` だけ（何も書かない）、decision `ingest.offload` は常に CONFIRM、承認後の render が `ingest offload` を 1 回実行し、コピーの inventory を新しい asset と OBSERVED の `ingest_clip` observation にする。pin は `tools/ingest/contract_0.1.0.json`。
- imagemagick-skill（ADR-077、imagemagick-skill の roadmap step 4 の agent 側）: package id は Skill 自身の `skill.id` と同じ `imagemagick-skill`（`VIDEO_AGENT_IMAGEMAGICK_DIR`、`~/.claude/skills/imagemagick-skill`、agent の source checkout の隣）。production skill `image_strip`（`imagemagick-skill/strip`）・`image_prepare`（`imagemagick-skill/resize`）・`image_pad`（`imagemagick-skill/pad`）は `edit.overlay.prepare.*` を明示した `edit.overlay` の画像だけを、video-editing の overlay の前に strip → resize → pad の順で新しい PNG にする。capability は package の `imagemagick-skill` と tool ごとの `imagemagick-skill:RESIZE` / `:PAD` / `:STRIP`（Skill の doctor の `tools.<name>.usable`）。phase 2 の `still_image_edit`（静止画そのものを納品物として編集する仕事）は宣言のまま（package id だけ `imagemagick-skill` に揃えた）。
- 作らないもの: plugin manager / package installer / dynamic import / marketplace / remote registry / 任意コードローダ。
- ADR-079: `video_crop` は video-editing-skill 0.4.1 の CROP（`video-editing/crop`）で実行する（ffmpeg-skill `crop.py` を agent から直接呼ばない）。x / y は整数、`edit.crop.fps` は廃止（CROP は入力の fps を保つ）。
