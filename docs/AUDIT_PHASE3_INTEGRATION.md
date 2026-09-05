# Phase 3 統合監査 — 統合 Production Pipeline 実装前の事実整理

対象: video-production-agent `claude/phase2-pr21-audio-production-integration` (ee5a104, PR #21 head) と、各 Skill リポジトリの main。
方法: コードの読解と、各 Skill CLI の実行（contract / doctor / 最小 run）。推測は「未検証」と明記する。

## 0. 前提の確認結果

| 前提 | 事実 |
|---|---|
| Skill 5 種は main へマージ済み | subtitle-skill 75b822c / thumbnail-skill e61997e / color-grading-skill a1da243 / motion-graphics-skill b86a224 / qc-skill 11a1c2e：いずれも PR #1 が main にマージ済み（0.1.0）。ffmpeg-skill main は 0.9.1、video-editing-skill main は PR #1 マージ済み（CI の `--branch` pin は不要になった） |
| Agent 側は PR #21 まで完了 | agent の **main は 287b685（PR #3 まで）**。PR #4–#21 は stacked Draft のまま未マージ。したがって本 Phase は PR #21 branch を base にした PR #22 として積む（既存方針どおり） |

## 1. 既に実装済み（再実装しない）

- Skill Registry（`skills/registry.py`）: SkillSpec / SkillPackage / ToolSpec、`select_tool`（capability → 宣言順の tool 候補、ranking / fallback 無し）、`availability` / `package_availability`。
- CapabilityResolver（`capabilities/resolver.py`）: ffmpeg / ffprobe / encoder / filter の直接測定、package ごとの doctor 連携（media-analysis / transcription / video-editing / audio-production）、operation ごとの `audio-production:<TYPE>`（doctor unknown → resolver 実測 → それでも不明なら UNKNOWN、UNKNOWN は選択不可）。
- Adapter 境界（`tools/base.py`, `tools/router.py`）と 5 adapter（ffmpeg-skill / media-analysis / transcription / video-editing / audio-production）: contract 検査、pinned contract との drift、argv list、`run_process_group`、stdout 1 文書、sha256 再計算、error code → recovery class。
- Decision Engine（`agent/decision_engine.py`, `agent/decision.py`, `policy/rules.py`）: evidence 必須、AUTO / CONFIRM / BLOCK、`resolve_approval`（明示 USER 要求で CONFIRM waiver、CONSTRAINT は不変）、`resolve_setting`（DEFAULT を provenance 付きで記録）。
- ProductionPlan（`agent/production_plan.py`, `agent/planner.py`）: `ProductionStep.depends_on` / `inputs` / `outputs` は skill を跨いで既に使われている（trim → loudness → export → check、audio_cut → audio_concat …）。`topological_order` と validator の依存検査あり。
- Project IR（`project/ir.py`, `schemas/project.schema.json` 1.2）: `captions` / `graphics` / `color` section は **空の placeholder**（schema は `type: object` のみ）。
- Compiler（`execution/compiler.py`）: plan.steps が名指しした tool でのみ lowering、chained idempotency key、固定順走査（trim → audio.cut → concat → edits → loudness → export → check）。**plan の depends_on は compiler が参照しない**（固定順で整合させる設計）。
- Executor / Recovery / Resume（`execution/executor.py`, `recovery.py`, `service.render(resume=)`）。
- QA（`qa/checks.py`）: agent 自身が probe / loudness を測る。media-analysis の observation と ffmpeg-skill `check` の行はそのまま fact として使う（測定 Skill の役割）。
- Artifact（`artifacts/store.py`）: working → candidate → approved → final → archive。final への昇格は QA FAIL / PENDING / UNKNOWN と plan REJECTED / BLOCKED / REVIEW / DRAFT で拒否。登録時 stage は QA PASS/WARN → candidate、FAIL → working。
- Revision / approval（`project/ir.py`, `project/diff.py`, `service.revise`）。
- CLI: `doctor skills analyze transcribe plan validate render approve reject revise diff check events context sessions explain artifacts artifact deliver archive` は全て存在。`explain` は `--decision / --step / --artifact / --observation / --context` の各起点で因果鎖を出す。
- transcription: `transcript` Observation（segments[{id,start,end,text,confidence,words,speaker_id=null}]）と SpeechEvent。speaker_id は常に null（推定しない）。

## 2. 不足（本 Phase で実装）

| # | 不足 | 統合ポイント |
|---|---|---|
| A1 | subtitle / thumbnail / color-grading / motion-graphics / qc の adapter・locate・pinned contract・capability・registry・Service 登録が無い | `tools/<skill>/`、`capabilities/resolver.py`、`skills/registry.py`、`service.adapter()` |
| A2 | registry の `caption_generation`（phase 3）は ffmpeg-skill/caption を直接参照（catalog に無い dangling tool） | subtitle-skill を canonical とし、`subtitle_generation` / `subtitle_burn_in` に置き換える |
| C1 | `subtitle.*` / `thumbnail.*` / `color.*` / `motion.*` / `qc.*` requirement が無い（REQUIREMENT_PREFIXES 外） | `agent/subtitles.py`, `agent/finishing.py`, `agent/qc.py`, `service.REQUIREMENT_PREFIXES` |
| C2 | 上記の Decision が無い（subject / APPROVAL_KEYS） | `agent/decision.py` |
| C3 | planner に post-edit 段（color → graphics → captions.burn）、sidecar（captions.generate）、thumbnail、qc の step が無い | `agent/planner.py`, `production_plan.STEP_PARAMETERS` |
| C4 | IR `captions` / `graphics` / `color` section の schema と validator、`qa.qc` | `schemas/project.schema.json`, `project/validator.py`, `project/hashing.PLAN_SECTIONS`, `project/diff.py`, `ir.rejected_cited` |
| C5 | compiler の lowering / path / 順序 | `execution/compiler.py` |
| D1 | transcript → cue の timeline 写像（trim の keep ranges、concat の timeline offset、speed factor）。subtitle-skill は transcription-skill の Transcript を直接受け取らない（実測: `INVALID_INPUT unknown field(s) confidence, raw_text, speaker_id, words`）ので caller（agent）が SubtitleCue に写す | `agent/subtitles.py` |
| E1 | QC を最終 promotion の gate にする。qc report の admission（input fingerprint == agent の sha256、skill / schema 一致、measurement_source OBSERVED）を通ったものだけ QA item に取り込む。agent 自身の duration / loudness / stream 検査は残す | `qa/checks.py`, `agent/qc.py`, `service._register_artifacts`, `artifacts/store.py` |
| E2 | CAPTIONS / THUMBNAIL artifact の登録（plan outputs の role から） | `service._register_artifacts` |
| G1 | `explain --pipeline`（Request → … → Artifact の一貫した因果鎖）、`check --qc` | `service.py`, `cli.py` |
| F1 | fake ×5、unit / integration（scenario 1–10）/ evals / CI | `tests/`, `evals/`, `.github/workflows/tests.yml` |

## 3. 重複（避ける）

- subtitle の burn-in は ffmpeg-skill `caption.py` に委譲される（subtitle-skill の `render`）。agent は ffmpeg-skill/caption を直接呼ばない（registry から `ffmpeg-skill/caption` 参照を消す）。
- thumbnail の frame 抽出は ffmpeg-skill `look` に委譲される。agent の `visual_inspection`（contact sheet、QA 用）はそのまま。thumbnail deliverable は thumbnail-skill 経由のみ。
- 5 adapter の transport（argv list / process group / stdout 1 文書 / sha256 再計算 / error 写像）は audio-production adapter と同型なので、共通 helper `tools/skill_process.py` に寄せる。既存 5 adapter は触らない。
- qc-skill の loudness / true peak / silence 測定は agent QA と重なるが、**agent の測定は削らない**（Skill の報告値を盲信しない原則）。qc は追加の gate。

## 4. 設計上の問題・注意（事実）

1. Skill ごとに transport が違う（下表）。adapter が吸収し、agent 本体には持ち込まない。

| | subtitle | thumbnail | color-grading | motion-graphics | qc |
|---|---|---|---|---|---|
| contract cmd | `contract` | `skill` | `skill` | `skill` | `contract` |
| tool id（contract） | 無し（operations generate/render） | thumbnail/render, thumbnail/extract_frame | color-grading/run | motion-graphics/run | 無し（operations inspect/check/validate） |
| request | `run -`（workspace は request 本体） | `run -` `{tool, params}` | `run -` | `run -` | `run -` |
| allowed input | 無し（workspace 相対のみ） | `--allowed-input` | `--allowed-input` `--allowed-lut` | `--allowed-input` | `--allowed-input-root` |
| ffmpeg-skill | env `SUBTITLE_SKILL_FFMPEG_SKILL_DIR` のみ | `--ffmpeg-skill` | `--ffmpeg-skill` [0.9.1,1.0) | `--ffmpeg-skill` [0.9.1,1.0) | 使わない |
| timeout | 固定 | `--timeout` | `--timeout` | `--timeout` | 無し（request の timeout は未使用） |
| 成功 | `status:"ok"` | `ok,status:"ok"`（`result` に入れ子） | `ok,status:"ok"` | `ok,status:"ok"` | `status:"completed"` |
| retryable | TOOL/DEPENDENCY/OUTPUT/INTERNAL、CANCELLED は非 | TOOL_ERROR, CANCELLED | 同左 | 同左 | 同左 |

   - subtitle / qc は tool id を宣言しないので agent 側で `subtitle/generate`, `subtitle/render`, `qc/check` を定義する（contract 不一致は adapter が吸収、ADR に記録）。
   - subtitle-skill の workspace は request 本体で渡すため、adapter は op dir を絶対 path で入れる。input（video）は workspace 相対のみ受け付けるので、burn-in の入力は op dir 内へ **hard link / copy** するのではなく、Skill の仕様上 `video_input` が workspace 相対である点を adapter が満たす必要がある → adapter は op dir を「入力の親を含む agent workspace」ではなく **agent workspace root** を request.workspace にし、`video_input` / `output_path` を workspace 相対で渡す（入力は常に agent workspace 内の中間物か、workspace 外の source。source を直接 burn する場合は workspace 外になるため、burn-in の入力は compiler が常に中間物（trim 後など）にする。source 直接の burn は前段 step が無い場合に発生し得る → その場合は BLOCK ではなく、subtitle adapter が INPUT_MISSING を返す前に planner が「burn-in は edit 後の中間物にのみ適用」と決め、edit が無ければ `video.trim` 無しの asset は copy step が無いので **PATH_NOT_ALLOWED** になる。これを避けるため、planner は burn-in の入力が source asset のままなら decision を BLOCK（理由: subtitle-skill は workspace 外の入力を受けない）とする。accepted limitation として記録。
   - motion-graphics doctor は title / lower_third / image_overlay を `unknown` と報告する（ffmpeg-skill doctor が drawbox / overlay / color / scale / colorchannelmixer を分類しないため）。resolver が `ffmpeg -filters` で実測して AVAILABLE にする（audio-production と同じ規則）。
   - thumbnail-skill は Pillow 必須（未導入なら `skill --json` 自体が落ちる）→ locate は成功しても adapter 生成が失敗 → capability MISSING（CI では Pillow を導入する）。
   - thumbnail `extract_frame` は ffmpeg-skill `look` の既定 `--width 1280` で **元解像度と異なる幅** になる（実測: 640x360 → 1280x720）。thumbnail の期待サイズは Skill の仕様として扱い、agent は「幅 1280 の静止画」を expected にしない（accepted limitation、QA では integrity と format のみ）。
   - qc-skill の `kind:"video"` は video 測定のみ、`audio` は audio のみ、`delivery` が両方 + subtitle。deliverable には `delivery` kind を使う。

2. 「話者が変わったらカメラを切り替える」: SPEAKER / CAMERA event は schema のみ（IMPLEMENTED_CODES 外）、speaker_id は常に null（推定禁止の方針）。ProductionContext の `transition` inference（source ごとの speech 活動の切替）は存在するが、それを実行する operation（multicam 切替）は video-editing-skill に無く、`multi_source_sync` は phase 2 未実装。**本 Phase では実装しない**（Event → step の近道を作らない。必要なのは (1) 同期 Skill、(2) source 切替 operation を持つ編集 Skill、(3) transition inference → `camera.switch` decision。次の作業として記録）。

3. compiler は plan の depends_on を読まず固定順で走査する。本 Phase でも固定順を拡張し（trim → concat → edits → color → graphics → captions.burn → loudness → export → check → thumbnail → qc）、validator が「step の inputs は自 asset か依存 step の outputs」であることを検査する（既存検査を新 section に広げる）。

4. AI が直接 command を生成・実行する経路: `src/` に subprocess は `capabilities/resolver.py`（`-version/-encoders/-decoders/-filters` の測定のみ）と `tools/*` の adapter 以外に無い。planner / decision / compiler は tool id と typed args だけを扱う。新規 adapter も同じ境界（argv list、FORBIDDEN_ARG_KEYS）。

5. Skill の報告値と QA: 現状 QA は media-analysis observation と ffmpeg-skill check の行を fact として使う（測定 Skill）。編集 Skill（video-editing / audio-production）の報告値は provenance に記録するだけで QA は使わない。qc-skill は「測定 Skill」だが最終 gate に使うため、admission 条件（fingerprint == agent の sha256、schema、skill、measurement_source）を課し、agent 自身の検査は削らない。

6. resume / revision / approval / idempotency / provenance: 新 section を `PLAN_SECTIONS`（plan_hash）と `diff.py` と `rejected_cited` に含めなければ、字幕や QC の変更が hash に出ない → 必ず含める。

## 5. 実装順（最小変更）

Phase A（adapter / capability / registry）→ B（validator の依存検査拡張）→ C/D/E（requirement → decision → plan → IR → compiler → QA gate）→ F（tests）→ G（CLI / docs）。既存 plan は新 requirement が無い限り byte 単位で不変であることを unit test で固定する。

## 6. 実装結果（PR #22、2026-09-05 追記）

- 変更ファイル: `tools/skill_process.py`（共通 transport）、`tools/{subtitle,thumbnail,color_grading,motion_graphics,qc}/`（adapter / locate / pinned contract）、`agent/{subtitles,finishing,qc,decision_finishing}.py`、`agent/{decision,planner,production_plan}.py`、`capabilities/resolver.py`、`skills/registry.py`、`project/{validator,ir,hashing,diff}.py`、`schemas/project.schema.json`、`execution/compiler.py`、`qa/checks.py`、`service.py`、`cli.py`、`media/analysis.py`、tests（fake ×5、adapter test ×5、`pipeline_harness.py`、`test_pipeline.py`、`test_integration.py::IntegratedPipelineRealTests`）、evals 89–97、`.github/workflows/tests.yml`、docs（ADR-031 / 032、skills、project-ir、README、MASTER_SPEC、GAP §22）。
- §4-1 の subtitle workspace 問題の結論: adapter は agent workspace root を request.workspace にし、burn-in の入力は常に中間物（前段が無ければ Decision が BLOCK）。
- §4-2（カメラ切替）は未実装のまま（GAP §22）。
- 実 Skill E2E の fixture: 音声は transcription-skill の `ja_short.wav`（9.6 s）を testsrc の下に 12 s へ pad して mux する。認識結果の segment end は音声長を数十 ms 超えることがある（`event range … exceeds asset duration` は temporal validator の既存規則で、変更しない）。
