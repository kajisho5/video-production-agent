# Gap Analysis — Skill Ecosystem の中核としての現状（2026-09-04, PR #4 時点）

前提: 実在して利用可能な外部 Skill は `kajisho5/ffmpeg-skill`（upstream 0.8.5、scripts / MCP の契約は 0.8.4 から無変更）のみ。media-analysis-skill 以下の将来 Skill はコード上に存在せず、存在するように見せてもいない。

## 1. すでに実装されているもの

| 領域 | 実装 | 場所 |
|---|---|---|
| Skill Registry | `SkillSpec`（inputs / outputs / required_capabilities / risk / approval / tools 候補 / phase）と `SkillRegistry` | `skills/registry.py` |
| Capability Resolver | ffmpeg / ffprobe / encoders / decoders / filters / GPU / fonts / ffmpeg-skill / ASR / AI キーの 4 状態検出、`doctor` | `capabilities/resolver.py` |
| Tool Adapter | `ToolAdapter` 基底、`FfmpegSkillAdapter`（型付き catalog、PathPolicy、プロセスグループ実行、JSON 契約） | `tools/` |
| ffmpeg-skill 連携 | 20 スクリプト中 12 を catalog 化、契約テスト（`--help` フラグ、version 範囲）、実メディア integration | `tools/ffmpeg_skill/`, `tests/test_integration.py` |
| Request → Requirements → Intent → Observation → Inference → Decision → Plan | 規則ベース、provenance 付き | `agent/`, `media/analyzer.py` |
| Project IR 1.2 + schema + validator + migration | plan_hash / ir_hash、REJECTED 強制、revision | `project/`, `schemas/` |
| Execution Compiler / Executor | 決定論的 op id、連鎖冪等キー、有限 recovery、SIGINT → CANCELLED | `execution/` |
| Jobs / Resume / Idempotency | `render --resume`、size+mtime 検証 | `jobs/`, `service.py` |
| QA / Incident / Provenance / Audit | 4 層 QA、計測記録、decision → operation → command の紐付け | `qa/`, `audit/` |
| Profiles / Policy | generic / youtube / conference（骨組み）、precedence、constraint | `profiles/`, `policy/` |
| Provider 抽象 | `AIProvider` / `NullProvider`（未接続） | `providers/` |
| Revision workflow | reject / revise / approve / diff、v(n) snapshot | `service.py`, `project/diff.py` |

## 2. MASTER_SPEC で要求されているが未実装のもの

Event / Session / Production、conference パイプライン本体、naming、artifact 昇格、analysis budget の強制、AI Provider 接続、Incident 検出（black / freeze）、Semantic QA、Web UI、queue、`--allowed-input`。いずれも設計文書（ARCHITECTURE_REVIEW / PHASE1_FINAL_REVIEW）に位置付け済み。

## 3. 現在の Phase で実装すべきもの（本 PR で実施）

責務分離の実態を確認した結果、**Skill → Capability → Tool の「選択」が実際には行われていなかった**。

| Gap | 事実（変更前） | 影響 | 対応 |
|---|---|---|---|
| G1 Tool 選択が無い | planner が `"ffmpeg-skill/cut"` 等を直書き、compiler も直書き。`SkillSpec.tools` は誰も読まない | 将来 Skill / 代替 tool を足すと planner と compiler の書き換えが必要。「Tool Selector」が存在しない | `SkillRegistry.select_tool / resolve_tools / availability` を追加。planner は解決済み表から `plan.steps[].tool` を書き、compiler は **plan.steps からのみ** tool を取る（`CompileError`） |
| G2 Executor が単一 adapter 固定 | `Service.adapter()` が `FfmpegSkillAdapter` を返す | 2 つ目の adapter を足す場所が無い | `ToolRouter`（tool id → 対応 adapter。振る舞いは追加しない）。登録箇所は `Service.adapter()` の 1 行 |
| G3 validator が plan.steps を検査しない | skill 名・tool 名が何でも通る | IR を手で書き換えれば未実装 Skill や未知 tool が compile される | step の skill が registry に存在し実装済み、tool が候補に含まれ adapter が対応、全 operation に step があることを検査 |
| G4 計測（analyzer / QA / check）も tool id 直書き | `"ffmpeg-skill/probe"` 等 | 分析系 Skill（`media_probe` 等）が registry にあるのに使われない | skill→tool 表を受け取り、そこから計測 tool を取る。Observation の `source` は `<tool>@<version>` |
| G5 将来 Skill が「宣言」と「利用可能」で区別されない | registry の phase 2/3/4 エントリが `phase` を持つだけ | `select_tool` が誤って選び得る | `implemented`（phase ≤ 1）を導入。未実装は選択不可、`video-agent skills` で NOT_IMPLEMENTED と表示 |
| G6 tool 非対応時の決定 | capability 欠落しか BLOCK しない | adapter が無い環境で plan が通る | `decide()` が `select_tool` で BLOCK 決定を生成 |
| G7 Operation に skill が無い | provenance が tool しか持たない | 「どの Skill を実現した操作か」が追えない | `Operation.skill` |
| G8 `DEFAULT_TOOLS` フォールバック | G1 の対応後も planner / analyzer / QA が `tools` 省略時に ffmpeg-skill の表へ暗黙にフォールバックしていた | 直接呼び出すと Registry を経ずに engine が決まり、「唯一の選択点」が成立しない | `DEFAULT_TOOLS` を全削除。`tools` は必須引数、`None` は `TypeError`、欠落 Skill は明示エラー / BLOCKED。provenance / artifact / 冪等キーの tool version も固定名 `ffmpeg-skill` ではなく operation の tool から引く |

追加したのは選択関数・ルータ・検査であり、新しい抽象層や plugin 機構は作っていない。

## 4. 将来 Phase まで待つべきもの

- 外部 Skill パッケージ（media-analysis 等）の adapter / capability 検出: パッケージが存在してから。手順は `docs/skills.md`。
- ffmpeg-skill の残り 8 スクリプト（fit / caption / overlay / graphics / audio / color / join / multicam）の catalog 化: それを使う Skill が実装される Phase で。
- ffmpeg-skill 0.8.5 への local checkout 更新: 契約に変更が無いため不要。version 範囲 `0.8.4 ≤ v < 0.9` は維持。
- analysis の TARGETED 戦略と budget、Event/Session、AI Provider 接続。

## 5. 現状の実装に問題があったもの

- G1〜G7（上記）。
- `dry_run` の推定と QA の check 結果突合が tool id 文字列比較だった → `Operation.skill` で判定。
- `Observation.source` が `ffmpeg-skill@0.8.4/probe` という独自表記だった → `ffmpeg-skill/probe@0.8.4`（tool id + version）に統一。

## 6. 変更しなかったもの（既存で十分）

Capability Resolver、Skill Registry の構造、FfmpegSkillAdapter と catalog、IR schema（変更なし: `plan.steps[].tool` は元から string）、Executor、Job / resume、revision、QA の判定、profiles、providers。

## テスト

unit 59/59（境界テスト 12 件: 追加分は planner / analyzer / QA に既定 engine が無いこと、別 engine `other-skill/trim` の Registry → Service → planner → compiler → ToolRouter → adapter → provenance 伝播、`DEFAULT_TOOLS` を検出する静的検査。当初 8 件: select_tool の 3 分岐、将来 Skill の非選択、plan→compiler の tool 伝播と validator の拒否、tool 無し step、CompileError、adapter 欠落での BLOCK、router の dispatch、tool id リテラルの静的検査）、integration 9/9、evals 6/6。

## 7. PR #6 — Ecosystem Contract（2026-09-04 追記）

調査対象: SkillSpec / SkillRegistry / Capability / CapabilityResolver / ToolRouter / ToolAdapter / Service / Planner / Compiler / Operation / Project IR / Provenance / doctor / CLI / docs。

| Gap | 事実（変更前） | 対応 |
|---|---|---|
| G9 Skill package の identity が無い | 「Skill」は production skill（`silence_cleanup`）のみを指し、`ffmpeg-skill` という package は adapter の `name` 文字列としてしか存在しなかった。将来 Skill repository が何を提供すべきかの型が無い | `skills/contract.py` に `SkillPackage` / `ToolSpec`。ffmpeg-skill を `tools/ffmpeg_skill/package.py` で Reference Skill として宣言（CATALOG から導出） |
| G10 Adapter が自分の package を宣言しない | `ToolAdapter` は `describe()` の自由形式 dict のみ | `ToolAdapter.package()` を契約に追加。`ToolRouter.packages()`、`Service` が adapter 登録時に registry へ自動登録 |
| G11 Registry が package を知らない | tool 候補は文字列で、どの package のものか検証されない | `register_package` / `packages` / `package` / `tool` / `package_availability` / `unknown_tool_candidates`。validator は step の tool が登録済み package の宣言 tool であることを検査 |
| G12 validator の capability 集合が engine 固定 | `{"ffmpeg","ffprobe","ffmpeg-skill"}` を直書き | registry（production skill + package + ToolSpec）から集める |
| G13 Decision の理由文に engine 名 | `ffmpeg-skill cut.py switches to --accurate` 等 | engine 非依存の表現に変更（判断ロジックは元から IR 語彙） |
| G14 provenance に package identity が無い | skill / tool / tool_version のみ | `skill_package`（tool id の prefix）を追加。IR schema は変更なし |
| G15 CLI が package を表示しない | `video-agent skills` は production skill のみ | "Skill packages" セクション（implemented / available / usable tools / version）を追加。JSON は `{packages, skills}` |

実装しなかったもの（意図的）: 外部 package loader、plugin manager、dynamic import、future skill の dummy、`SkillSpec` の改名。DECLARED / IMPLEMENTED / AVAILABLE は既存の `NOT_IMPLEMENTED` / `implemented` / `AVAILABLE` に対応付け、新 status は作っていない。

テスト: unit 66/66（EcosystemContractTests 7 件: Reference Skill 登録、tool 契約、Skill→Tool→Adapter の閉包、future skill 非 AVAILABLE + production code に future package 名が無いこと、default fallback 無し、fake-skill package の test scope 登録と Registry→…→provenance 伝播 + validator 拒否、engine 漏洩の静的検査）、integration 11/11（実 runtime での契約テスト追加）、evals 6/6。

## 8. PR #7 — AI Provider Contract / Reasoning Boundary（2026-09-04 追記）

調査: 既存は `providers/base.py` の `AIProvider`（`extract_requirements` のみ、呼び出し元なし）と `NullProvider`、capability `ai:anthropic` / `ai:openai`、provenance 値 `AI_GENERATED`。`ai_calls` / `max_ai_calls` / AI 由来 inference の扱いは未実装。

| Gap | 事実（変更前） | 対応 |
|---|---|---|
| G16 Provider 契約が型として無い | 抽象クラスに 1 メソッドのみ、request / response / usage / identity / failure の型が無い | `AIRequest` / `AIResponse` / `AIUsage` / `AIProviderError`（6 failure kind）/ `TASK_TYPES`。`NullProvider` が既定、実 provider は未同梱 |
| G17 AI 出力の受け口が無い | AI の結果を Inference / Decision に変換する層が無い | `agent/ai_reasoning.py`: evidence 要約の request 生成、untrusted response の検証（intent は実装済み production skill、evidence は既存 id、tool / argv / command / risk / approval を除去）、`AI_GENERATED` inference |
| G18 Decision が AI 提案を扱わない | — | 計測済み decision と一致する提案は evidence に追加（confidence / risk / approval 不変）。それ以外は `ai.<intent>` の review decision（policy の approval、registry の risk、`executable: false`） |
| G19 budget が無い | `analysis.budget` は time のみ、`max_ai_calls` 未実装 | `analysis.budget.max_ai_calls`（既定 4）。超過は `BUDGET` で明示失敗、1 回の呼び出しに 1 試行、retry 無し。revision は記録済み AI inference を再利用し呼び出しゼロ |
| G20 AI provenance が無い | — | `provenance.ai_calls[]`（provider / model / task / request fingerprint / response hash / usage / latency / error）、`provenance.ai_provider`。job の provenance.json にも複製。IR schema は変更なし（provenance は追加キー許容） |
| G21 Observation の捏造防止が無い | validator は observation の source を検査しない | source が `<tool>@<version>` でない observation を拒否、AI inference は `AI_GENERATED` 必須、evidence 無し inference を拒否 |
| G22 doctor に provider 表示が無い | key の有無のみ | `ai:provider`（設定名のみ、key は表示しない）。`explain` に AI 呼び出し要約 |

実装しなかったもの: 実 provider（OpenAI / Anthropic / Gemini / local）、AI による requirements 抽出の呼び出し（task type のみ予約）、CoT の保存、AI による Tool ID 指定。

テスト: unit 77/77（AIProviderBoundaryTests 11 件）、integration 12/12（FakeAIProvider → 実メディア production、AI の command が実行されないことを provenance の command で確認）、evals 6/6。

## 9. PR #8 — Observation / Analysis Architecture（2026-09-04 追記）

調査: Observation は `media/analyzer.py` の `MediaAnalyzer.analyze(paths)` だけが生成し、kind は probe / silence / loudness の固定手順。`analysis.strategy`（generic は TARGETED）は無視され常に FULL、`analysis.budget.max_processing_time` は記録のみ（`enforced: false`）。再分析の再利用機構は無し。AI evidence は `build_request` が observation を無加工で渡していた。

| Gap | 事実（変更前） | 対応 |
|---|---|---|
| G23 AnalysisKind / AnalysisRequest が無い | 分析対象・戦略・予算が引数に散在 | `media/analysis.py`: `ANALYSIS_KINDS`（media_probe / silence / loudness のみ）、`AnalysisRequest`（kinds / strategy / budget / cache_policy / params / hash）、`targeted_kinds`（requirements から決定） |
| G24 Analyzer contract が無い | `MediaAnalyzer` は単なるクラス | `Analyzer`（id / version / supported_kinds / analyze）。`MediaAnalyzer = media@1.0`。tool 呼び出しは `ToolAdapter.measure` のみ |
| G25 strategy が偽装 | TARGETED 指定でも FULL 実行、記録は "FULL_ANALYSIS" | FULL / TARGETED / CACHED_ONLY を実装。IR には実行した戦略を記録（schema enum に CACHED_ONLY を追加） |
| G26 budget が偽装 | 値だけ記録、強制なし | `AnalysisBudget`（max_analysis_calls / max_total_seconds）を各 tool call 前に強制。未対応キーは `ANALYSIS_UNSUPPORTED` で拒否。`enforced: true` と実使用量を記録 |
| G27 Observation validation が無い | tool 結果をそのまま Observation 化 | `validate_observation`（asset / kind / source `<pkg>/<tool>@<ver>` / analysis_id / provenance OBSERVED / 構造 / credential・command 漏洩）。不正結果は保存しない |
| G28 cache が無い | 同じ asset を毎回再計測 | `ObservationCache`（workspace/cache/observations）。key = fingerprint + kind + analyzer@ver + tool@ver + params。hit で analyzer 未実行、version / params / content 変更で miss、破損は `ANALYSIS_CACHE_INVALID` として再計測 |
| G29 analysis provenance が無い | `tool_calls` のみ | `analysis.analyses[]`（analysis_id / request / analyzer / 時刻 / 行ごとの tool・cache_key・cache_hit・status・error / budget 使用量 / cache 統計）。Observation に analysis_id / analyzer / cache_key / provenance |
| G30 AI evidence が無加工 | observation data をそのまま provider へ | `safe_observation_summary`: tool 由来かつ OBSERVED の observation のみ、credential / command 様の値を除去。`to_inferences` も OBSERVED の id しか evidence と認めない |
| G31 分析失敗の domain が無い | 失敗は warning 文字列のみ | `AnalysisError`（6 kind）。AIProviderError / engine incident とは別。行ごとに記録し、plan は残る evidence で決定論的に継続 |

変更なし: AI Provider / `agent/ai_reasoning.py` の責務（evidence の scrub 呼び出しを追加したのみ）、SkillRegistry / ToolRouter / Adapter / ffmpeg-skill、IR schema（strategy enum の追加値のみ）、revision / resume（revision は記録済み observation を再利用し analyzer を再実行しない。cache は resume 状態とは独立）。

やっていないこと: conference 固有の分析（speaker / slide / sync / multicam / caption）、scene_detection / frame_integrity 等の未実装 kind の宣言、doctor への analyzer 表示、bytes / duration 予算。

テスト: unit 87/87（ObservationAnalysisTests 10 件で指示の 24 項目を網羅）、integration 13/13（実メディアで 1 回目計測 3 call → 2 回目 cache hit 0 call、CACHED_ONLY、AI 推薦 → render → QA）、evals 6/6。

## 10. PR #9 — Temporal / Event / Session Architecture（2026-09-04 追記）

調査: `models.Event`（type / timeline_id / range / source / kind / confidence / evidence / metadata、random id）、`TimeRange`（検証なし）、`temporal/timeline.py`（Timeline / TimelineMap / query）が存在し、analyzer が AUDIO_SILENCE / AUDIO_ACTIVE / LOUDNESS_MEASURE を、review が USER_DECISION を生成していた。Event type 体系、subtype、asset 参照、決定論的 identity、Session、validation、AI evidence 境界での event 扱いは無かった。

| Gap | 事実（変更前） | 対応 |
|---|---|---|
| G32 時間型に検証が無い | `TimeRange` は dataclass のみ | `TimePoint` / `TimeRange`（=`TemporalRange`）に検証（start ≥ 0, end ≥ start, NaN 拒否、TIME_EPS）と relation（overlaps / contains / precedes / adjacent / within） |
| G33 Event type 体系が無い | `type` は自由文字列 | `EVENT_TYPES`（9 domain type + subtype）、`EVENT_CODES`（canonical code ↔ domain/subtype）、`IMPLEMENTED_CODES`（生成される 4 code のみ）。未実装 type は schema のみ、fake 生成なし |
| G34 Event に asset / provenance / evidence 検証が無い | timeline_id から暗黙 | `asset_id` / `event_type` / `subtype` / `provenance` / `generator` / `session_id` を追加（`classify` で旧 IR も補完）。`validate_event` を IR validator に組込み |
| G35 Event identity が random | 再生成で増殖し得る | `event_id` = hash(asset, code, subtype, range, source, evidence)。`Timeline.add` は idempotent。USER_DECISION も決定論的 id |
| G36 Observation → Event 変換が analyzer 内に散在 | `_events` に直書き | `events_from_observation`（`observation_to_event@1.0`）。tool 計測（OBSERVED）以外は拒否、media_probe は event 化しない |
| G37 Session が無い | — | `Session`（決定論的 id、project / name / range / assets / events / provenance）、`session_for_asset`（asset 単位の既定 session）、`validate_session`（範囲・asset・child event、clip しない） |
| G38 AI evidence に event が無加工 | id / type / range / metadata をそのまま | `safe_event_summary`（AI_GENERATED を除外、provenance / evidence 保持、metadata scrub）。`to_inferences` は AI_GENERATED event id を evidence と認めない |
| G39 CLI で時間軸を確認できない | — | `video-agent events` / `sessions`（--json 対応） |

変更なし: production planning、Project IR の実行系（plan_hash は timeline を含まない）、QA incident model、AI provider、ffmpeg-skill、SkillRegistry / Router。revision は project identity を保持するようになり（`fresh.doc["project"] = old["project"]`）、session / event の id が版をまたいで安定する。

やっていないこと: transcription / speaker / slide / camera / scene / sync / multicam / caption の検出、conference の自動 session 認識、Inference → Decision → Event の flow、IncidentEvent の生成、Event からの production planning。

テスト: unit 95/95（TemporalEventSessionTests 8 件で指示の 30 項目を網羅）、integration 14/14（実メディアで 3 s 無音 → AudioEvent(silence)、session、CLI events / sessions / explain、render）、evals 6/6。schema は event の任意フィールドと timeline.sessions の追加のみ。

## 11. PR #10 — Production Planning Architecture（2026-09-04 追記）

調査: `agent/planner.py` の `build_plan` が decision から IR の `plan.steps`（id / skill / tool / decision_ids / params）と video / audio / delivery セクションを同時に生成していた。plan は version / steps / summary だけで、identity・status・inputs / outputs・依存・evidence・temporal scope・constraints・provenance を持たず、Event は planner から参照されていなかった。

| Gap | 事実（変更前） | 対応 |
|---|---|---|
| G40 ProductionPlan が第一級でない | `plan` は dict 断片 | `ProductionPlan` / `ProductionStep`（`agent/production_plan.py`）。IR の `plan` セクションをこの型で記録（既存キー version / steps / summary は維持、schema は additive） |
| G41 step に依存・順序・evidence が無い | steps は生成順のみ | order / depends_on（trim → loudness → export → check）、決定論的 topological order、evidence（decision → inference → event / observation）、temporal_scope、outputs（論理名） |
| G42 plan の status が無い | render gate が個別条件で判定 | `plan_status`（DRAFT / REVIEW / APPROVED / REJECTED / BLOCKED）を reviews / approvals から導出し、approve / reject / _fill で同期。render は APPROVED のみ実行（既存 gate と同じ結果を明示） |
| G43 Event が planning の入力でない | decision の evidence に event id はあったが plan に無かった | step.evidence と plan.events に event id を伝播。Event は事実、step が制作命令（`removed` / `keep`） |
| G44 plan の検証が無い | validator は step の skill / tool のみ | `validate_plan`（id / project / status 整合 / 一意性 / 順序 / 依存 / cycle / inputs / decisions / evidence / domain parameter 限定 / leak / scope / tool ∈ skill / outputs） |
| G45 step の説明経路が無い | explain は decision 単位 | `explain_step` と `video-agent explain --step`（decision → inference（AI provenance）→ event → observation → source） |
| G46 revision で inference id が失われる | revise が旧 analysis で新 inference を上書き | observations / analyses は旧版、inferences は再計画分（AI 分は再利用）を保持 |

変更なし: compiler / executor / ToolRouter / adapter、plan_hash の意味、approval の source of truth（execution.reviews / approvals）、AI boundary、SkillRegistry の選択。`DEFAULT_TOOLS` 等の fallback は再導入していない。

やっていないこと: AI による plan 生成、部分実行（approval は decision 単位、render は plan が APPROVED になるまで待つ — PR #4 の仕様を維持）、Artifact / Delivery 本実装、追加 intent（silence_cleanup / loudness_normalization / delivery_export / delivery_check 以外）。

テスト: unit 102/102（ProductionPlanTests 7 件で指示の 30 項目を網羅）、integration 15/15（vertical slice: 実 talk.mp4 の 3 s 無音 → event → decision → plan → IR → ffmpeg-skill → QA PASS → explain、音声無し素材、敵対的 AI）、evals 12/12（6 件追加）。

## 12. PR #11 — Artifact / Delivery / Archive（2026-09-04 追記）

調査: `models.Artifact`（path / type / hash / source / generation / tool / tool_version / qa_status / stage）は存在し、render が delivery 出力ごとに job.json へ記録していた。identity は論理名（path 相当）のみで、plan / job / operation / step との関係、manifest、integrity 検証、delivery / archive の状態遷移、naming、path security、explain は無かった。

| Gap | 事実（変更前） | 対応 |
|---|---|---|
| G47 Artifact identity が path 相当 | id = "<asset>_delivery_<target>" | `artifact_id(project, plan, logical_name, sha256)`。revision は別 artifact、resume の再利用は同一 artifact に job を追記 |
| G48 Artifact が job 内 dict のみ | 登録・取得・一覧が無い | `ArtifactStore`（manifest registry、integrity、register / get / list / verify / promote / archive_index） |
| G49 provenance 連鎖が無い | artifact から step / decision に辿れない | `operations` / `step_id` / `decision_ids` / `provenance` を保持、`Service.explain_artifact` と `explain --artifact` |
| G50 QA と delivery の混同 | stage は QA 結果から直接決まるだけ | QA（PASS / WARN / FAIL / UNKNOWN）と lifecycle（working / candidate / final / archive、view NOT_READY / READY / DELIVERED / ARCHIVED）を分離。promote の gate（integrity / QA / plan status / 遷移） |
| G51 immutability が無い | 同 id で内容変更可能 | sha256 が identity の一部、promote 前に再検証、同 id 異内容の登録は ARTIFACT_CONFLICT |
| G52 naming / path security が無い | profile の naming template は未使用 | `safe_filename` / `delivery_name`（traversal / 無効文字 / Windows 予約名 / 長さ）、`check_path`（絶対・非 traversal・非 symlink・workspace 内） |
| G53 登録失敗時の整合 | 出力が無くても COMPLETED になり得た | 登録失敗は job FAILED + `artifact_error`、artifact は登録されない |
| G54 archive が無い | — | stage archive + `<workspace>/archive/<project>.json` 索引（論理 archive、コピー / 圧縮なし） |

変更なし: compiler の出力 path 決定、executor / idempotency、QA 判定、plan_hash / ir_hash の意味、AI boundary、ffmpeg-skill。schema は Artifact に任意フィールドを追加したのみ（job.json）。

やっていないこと: 外部 delivery（YouTube / S3 / NAS / FTP …）、圧縮 archive、artifact-skill 等の Skill 化、`approved` stage の運用（定義は既存のまま）、複数 delivery target の naming 衝突解決以上のもの。

テスト: unit 111/111（ArtifactLifecycleTests 8 件）、integration 16/16（実メディアで sha256 / deliver / archive / explain --artifact / resume 再利用 / revision 分離 / 音声無し WARN）、evals 22/22（10 件追加）。

## 13. PR #12 — External Skill Integration: media-analysis（2026-09-04 追記）

外部 repository の実態（clone して確認）:
- `kajisho5/media-analysis-skill` main = 0.1.0（PR #1 merge 済み）。`contract --json` は `media-analysis/contract@1`、10 kind / 9 tool、`run - --json`、`doctor --json`。未 merge の phase2 branch（hardening 5 commit、version 0.1.0 のまま）は対象外。
- `kajisho5/transcription-skill` main = README のみ（"Initial commit"）。設計 branch に 0.1.0 相当の実装はあるが未 merge で、released contract が存在しない。指示にある `transcript/0.1` / `engine-spec/0.1` は main 上に無い。

| Gap | 事実（変更前） | 対応 |
|---|---|---|
| G55 外部観測 Skill の adapter が無い | 計測は ffmpeg-skill のみ | `tools/media_analysis/`（locate / contract 取得 / 互換検査 / request 生成 / response 検証 / ToolResult 変換）。process boundary のみ、import 無し |
| G56 contract の二重管理 | — | tools / kinds / kind_to_tool / capabilities / version / schema はすべて `contract --json` から。`contract_0.1.0.json` は識別用 snapshot |
| G57 Observation に外部 provenance が無い | source のみ | `skill` / `skill_version` / `tool` / `external_id` / `fingerprint` / `parameters` / `cache` を追加（後方互換） |
| G58 AnalysisKind が 3 種固定 | — | 10 kind。CORE_KINDS（FULL の既定）は 3 種のまま、他は `--kind` / `kinds=` で明示要求 |
| G59 request 生成が analyzer に散在 | ffmpeg-skill 専用引数 | `ToolAdapter.measurement_args` hook（adapter が request 形を決める）、`owns_cache`（Skill 所有 cache は agent が二重化しない） |
| G60 doctor / skills に外部 Skill が出ない | — | capability `media-analysis`（version / contract / tools / kinds / execution / doctor status）、package 一覧に media-analysis |

transcription-skill: 未接続。理由は main に実装・contract が無いため（stub / 推測 contract は作らない）。SpeechEvent 型（PR #9）と registry の package 受け入れは既に存在し、contract が release されれば同じ手順（adapter + capability + registry 候補 + register 1 行）で接続できる。

テスト: unit（adapter protocol を fake process で: contract discovery / 互換 / tool mapping / request / response / lifting / provenance / malformed 9 種 / timeout / unavailable / cache metadata）、integration（実 media-analysis-skill 0.1.0 + talk.mp4: duration / silence / loudness / video_format / audio_format / integrity / scene_detection の Observation、AudioEvent、provenance chain、2 回目 cache hit）、evals 追加。

## 14. PR #13 — External Skill Integration: transcription-skill（2026-09-04 追記）

外部 repository の実態（clone して確認）:
- `kajisho5/transcription-skill` main = 0.2.0（PR #1 merge 済み）。`skill --json` が contract（tools 4 / engines 1: faster_whisper local / schemas transcript・engine-spec・speech-event 0.1）、`doctor --json [--offline] [--allowed-input]`、`run -`（`{"tool","params"}` → `{"ok","tool","result"}`、exit 0 / 1 / 2）。指示書の `contract --json` / `run - --json` は存在しない → 実物に合わせた（ADR-024）。
- 子プロセスへは PATH / HOME 等の最小 env しか渡さない設計（Skill 側）。proxy 経由 CA が必要な環境では model download が失敗する → air-gapped 手順（HF cache へ事前配置、`--offline`）で検証。

| Gap | 事実（変更前） | 対応 |
|---|---|---|
| G61 認識 Skill の adapter が無い | transcription 未接続 | `tools/transcription/`（locate / contract / 互換検査 / typed request / process / response 検証 / check_transcript）。import 無し、engine 直接実行無し |
| G62 Transcript の Observation 化が無い | — | `kind=transcript`（needs_audio）、`_lift_transcript`（provenance 一式、fingerprint 照合、cache は Skill 所有） |
| G63 SpeechEvent が schema のみ | IMPLEMENTED_CODES に SPEECH 無し | `events_from_observation(transcript)` → SPEECH（segment ごと、speaker_id null、confidence）。SPEAKER は未実装のまま |
| G64 registry / capability | — | production skill `speech_transcription`（caps ffmpeg / ffprobe / transcription、LOW / AUTO）、capability `transcription`（Skill の doctor: AVAILABLE / DEGRADED / MISSING、engines / models evidence） |
| G65 CLI / explain | — | `video-agent transcribe`（typed options のみ、`--allowed-input`、`--offline`）、`analyze / plan --kind transcript`、`explain --observation`（observation → skill → tool → engine → model → transcript → asset → analysis → events） |
| G66 入力境界の一貫性 | ffmpeg-skill PathPolicy のみ | adapter が roots（allowed inputs + workspace）を固定して Skill に渡す。traversal / outside / symlink escape を adapter と Skill の両方が拒否 |

未実装（意図的）: AI / LLM、diarization / speaker identity、字幕、編集判断、cloud / whisper.cpp、MCP / plugin loader、ranking。SpeechEvent → Inference → Decision → Plan は次 PR。

既知の別件: ffmpeg-skill の silencedetect が container duration を超える end を返す fixture（transcription-skill の lecture_short.mp4）では `plan` が validation error（event range exceeds asset duration）になる。base branch でも同じ（本 PR の変更ではない）。

テスト: unit（fake transcription process 21 モード: valid / empty / text / two_docs / wrong schema・skill・version・engine・asset / bad source / no transcript / invalid provenance / speaker_id / bad segments / timeout / hang / crash / non-zero / model・engine unavailable、SpeechEvent、cached-only、allowed roots + symlink escape（adapter と Skill 双方）、registry / resolver / Service(offline)、explain chain、静的境界）、integration（実 transcription-skill 0.2.0 + faster-whisper 1.2.1 + base model local: contract / doctor / path policy は常時、実認識・lifting・provenance・cache hit・SpeechEvent・shared identity・CLI は model が local のときのみ。CI は clone のみで model download を強制しない）、evals 15 件（negative 中心）。

## 15. PR #14 — SpeechEvent → Inference → Decision → ProductionPlan（2026-09-04 追記）

| Gap | 事実（変更前） | 対応 |
|---|---|---|
| G67 SpeechEvent が inference に届かない | SPEECH は timeline 記録のみ | `agent/speech_inference.py`: speech_interval / speech_activity / internal_silence_removable / speech_silence_conflict（決定的、speaker_id null、AI 無し） |
| G68 発話統合・削除候補の閾値が無い | `silence.internal.min_seconds` のみ | `speech.merge_gap_seconds`（DEFAULT 0.5）、`silence.internal.removable_min_seconds`（DEFAULT 2.0）を policy キー化し、値 + provenance を inference に記録 |
| G69 内部無音は常に keep | `silence.internal: keep`（AUTO） | 発話に挟まれた長い無音は `silence.internal.<range>: remove (candidate)`（CONFIRM / BLOCK、AUTO 無し）。conflict と重なる lead / tail trim は CONFIRM |
| G70 planner が内部区間を扱えない | keep は単一区間 | 候補（未 REJECTED）を removed に加え keep を補集合化（多区間 trim、compiler / QA 既存対応） |
| G71 provenance chain | — | explain --step が decision → removable inference → silence event + speech_interval → SPEECH → transcript observation まで到達 |

既知・未対応: silencedetect end > duration（別 PR）。実メディアでは Whisper segment が無音側へ伸びるため conflict になりやすい（単語タイムスタンプでの精緻化は次 PR 候補）。

テスト: unit 4 件（fake: 統合、conflict、候補 → CONFIRM → approve / reject / revise / render / resume、境界・決定性・silencedetect 未修正）、evals 8 件、integration 1 件（実 transcription-skill + ffmpeg-skill、ja_short ×2 + 3 s 無音）。

## 16. PR #15 — ProductionContext / Situation Understanding（2026-09-04 追記）

| Gap | 事実（変更前） | 対応 |
|---|---|---|
| G72 複数種別 Event の同時状況を表す層が無い | Event / Session のみ | `context/` ProductionContext（参照中心、DERIVED、決定的 id）、`build_contexts` |
| G73 ドメイン非依存の inference が無い | speech 専用 inference のみ | `context/inference.py`: source_activity / source_inactivity / transition / conflict |
| G74 Context の provenance / explain | — | IR `analysis.contexts`、validator、`explain --context`、`explain --observation` に context 行、`context` CLI |

別 PR で追加すべき Capability（今回 Skill は変更しない）:
- 複数ソース同期（TimelineMap offset）: master timeline 上でのクロスアセット context に必要
- SceneEvent / SlideEvent / CameraEvent を生成する観測（scene_detection は media-analysis に存在するが Event 変換は未実装）
- 単語タイムスタンプでの発話境界精緻化（PR #14 の残課題）

既知・未対応: silencedetect end > duration（別 PR）。

## 17. PR #16 — Production Decision Engine（2026-09-04 追記）

| Gap | 事実（変更前） | 対応 |
|---|---|---|
| G75 decision の生成規則が暗黙（evidence 空の decision が生成され得る） | `decide()` 手続き内の個別実装 | `agent/decision_engine.py`: evidence 必須・根拠クラス検査・AI 単独は REVIEW・type 語彙・BLOCK ⇔ BLOCKED・leak 検査 |
| G76 approval の policy 解決に provenance / 未知値 / floor の規則が無い | `rules.get(key, "AUTO")` の直接参照 | `resolve_approval`（DEFAULT 明示、未知値 → CONFIRM、BLOCK* → BLOCK、floor は上げるのみ、waiver は CONSTRAINT に不適用）と basis 記録 |
| G77 Decision → Policy / Preference / Constraint / Intent の説明が無い | evidence 一覧のみ | `basis` + `explain --decision`（basis → evidence chain → plan step / IR operation） |
| G78 IR 上で decision 不変条件が検査されない | BLOCK / CONFIRM の warning のみ | `check_decisions` を validator に組込み |

別 PR 候補（本 PR では扱わない）:
- 話者・カメラ・スライド等の新しい decision domain（Event 種別の観測が先）
- 単語タイムスタンプでの発話境界精緻化（PR #14 の残課題）
- `video.*.approval` を request から設定可能にするか（現状は profile のみ。REQUIREMENT_PREFIXES に "video." は無い）

既知・未対応: silencedetect end > duration（別 PR）。

## 18. PR #17 — ffmpeg-skill 0.9.x compatibility / CI stabilization（2026-09-04 追記）

| 事実 | 対応 |
|---|---|
| CI は ffmpeg-skill の HEAD（0.9.0）を clone する。PR #4–#12 の `SUPPORTED_MAX_EXCLUSIVE = (0, 9, 0)` は 0.9.0 を範囲外とし、contract test `test_version_and_scripts` が失敗する（他の integration は 0.9.0 で全件通過。version 判定は runtime では強制されておらず、宣言 + テストのみ） | PR #13 の hunk（`(0, 10, 0)` + 根拠コメント）をそのまま PR #4–#12 の各 branch に移植（同一テキストのため stacked merge で衝突しない）。CI を各 branch で dispatch |
| 範囲の妥当性を unit で固定していない | `AdapterTests.test_ffmpeg_skill_version_range_is_explicit`（0.8.4 / 0.9.x accepted、0.8.3 / 0.10 / 不正値 rejected、範囲定数を明示） |
| docs/skills.md が `< 0.9` のまま | `< 0.10` に更新 |

main（287b685）も `(0, 9, 0)` のままで同じ理由で赤になるが、main への直接変更はルール上行わない（PR #4 のマージで解消）。ffmpeg-skill / media-analysis-skill / transcription-skill は変更していない。

## 19. PR #18 — video-editing-skill adapter integration（ADR-028、2026-09-05 追記）

| Gap | 事実（変更前） | 対応 |
|---|---|---|
| G79 編集 Skill の adapter が無い | 編集は ffmpeg-skill/cut のみ、video-editing-skill は将来 Skill 扱い | `tools/video_editing/`（locate / contract 取得・互換検査・drift 検出 / EditRequest 生成 / response 検証 / ToolResult 変換）、CLI が境界、import 無し |
| G80 IR → 外部編集 Skill の lowering | compiler は ffmpeg-skill の catalog 形しか出さない | `lower_video_trim`: video.trim → video-editing/cut（keep / precision）。他 tool の lowering は不変 |
| G81 tool 契約の capability が選択で検査されない | SkillSpec の capability のみ | `SkillRegistry.tool_missing_capabilities`（package capabilities + resolver が解決する ToolSpec.required_capabilities） |
| G82 Skill の retryable 判定が recovery に届かない | stderr 文字列分類のみ | `recovery._skill_class`: data.error.recovery_class / retryable を優先、SKILL_ERROR（BLOCK）を追加 |
| G83 Skill の実行事実が provenance に無い | commands のみ | provenance operations[].skill_result（skill / operation_id / artifact sha256 / timeline / observation） |
| G84 PR #18 / #19 の重複 | 同一機能の adapter が 2 本 | PR #18 を canonical にし、#19 の typed lowering / PathPolicy 適用 / output path 表記 / INTERRUPTED / lift_observation / capability evidence を取り込み、#19 を close |

CI: video-editing-skill は PR #1 branch（claude/video-editing-skill-sd9vgt）を clone している。main へマージされたら tests.yml の `--branch` を外すこと（main は README のみ）。

別 PR 候補: video.concat / speed / fit / overlay を Plan / Decision に接続（今回は adapter が contract の全 tool を request 化できるが、planner が出す IR operation は video.trim のみ）、audio-only 入力の扱い（video-editing は INVALID_INPUT）、drift 検出の CI 単独ジョブ化。


## 20. PR #20 — video-editing-skill operations（concat / speed / resize / fit / fill / overlay、ADR-029、2026-09-05 追記）

| Gap | 事実（変更前） | 対応 |
|---|---|---|
| G85 planner が出す IR operation は video.trim のみ | adapter は contract の全 tool を request 化できるが Plan / Decision に接続されていない | `agent/editing.py`（語彙・requirement 検証・concat segments・delivery subjects）、decision.py（TRANSFORM / BLOCK）、planner（step + IR op）、registry（video_concat … video_overlay）、schema、validator（check_video_operations）、compiler（lower_video_edit）、QA / artifact（subject 単位） |
| G86 複数入力の timeline を IR で表せない | asset ごとの独立チェーンのみ | `video.concat {inputs, output: programme, segments[{input, track, source_range, timeline_range}], timeline_duration}`、以降の操作・loudness・delivery は programme に適用 |
| G87 BLOCK decision が step を持たないと plan が APPROVED になる | plan_status は step 由来のみ | BLOCK decision（status BLOCKED）があれば BLOCKED（render は従来から ir.blocked() で拒否していたが、plan 表示も一致させた） |
| G88 step.tool が None の plan が schema で落ちて意味的な理由が出ない | schema `tool: string` | `tool: string | null`（validator の「has no selected tool」が理由として出る。None が実行に進む経路は無い） |
| G89 fake video-editing の出力が下流の fake から probe できない | payload を sort_keys で書き `{"fake"` 先頭でない | key 順固定（"fake" 先頭）、CONCAT / SPEED の duration を入力 payload から算出（test double のみ） |

四段階の区別: Skill supports = 8 operation（contract）/ adapter supports = 同 8（Lowering.ARGS）/ Planner can generate = video.trim + 6 operation（video.concat / speed / resize / fit / fill / overlay）/ E2E verified（実メディア）= trim → concat（plain・fade transition）→ speed → resize → fit・fill → overlay。

未対応・別 PR 候補: `video.<op>.approval` を request から設定（REQUIREMENT_PREFIXES に "video." 無し）、asset の一部だけを concat する指定（現状は全 video asset）、concat の並び替え（入力順のみ）、overlay の時間範囲と programme 長の整合検査（Skill 側の検証に委ねている）、audio-only 入力との混在時の concat（BLOCK）、CROP / FREEZE / REVERSE（Skill が未対応）。silencedetect end > duration は別 PR。

## 21. PR #21 — audio-production-skill vertical slice（ADR-030、2026-09-05 追記）

| Gap | 事実（変更前） | 対応 |
|---|---|---|
| G90 audio-production-skill の adapter が無い | 音声処理は ffmpeg-skill/loudness と /cut のみ | `tools/audio_production/`（locate / lowering / adapter / pinned contract）、契約検査・drift・doctor・response 検証・error mapping・lift_observation / lift_measurement |
| G91 audio operation の Plan / IR 語彙が無い | `audio.loudness` のみ | `agent/audio.py`（requirement 検証、channel_operation、cut_ranges、concat segments、audio_subjects）、IR `audio.cut / concat / gain / mono / stereo / downmix / fade_in / fade_out`、`audio.loudness` に参照と tolerance / sample_rate |
| G92 measurement ≠ decision ≠ execution の loudness 接続 | 測定 → decision → ffmpeg-skill/loudness | 同じ decision / IR type を audio path では NORMALIZE（tolerance_lufs 再測定）に lowering、再測定を provenance の Observation として記録 |
| G93 audio-only / video+audio / video-only の区別 | asset 種別による経路の分岐無し | audio path は audio を持つ asset のみ、video container は `audio.extract`（CONFIRM）、video-only は BLOCK、edit.* との同時要求は BLOCK、validator が audio path の asset への video operation を conflict として拒否 |
| G94 per-operation capability | package capability のみ | `audio-production:<TYPE>`（doctor supported / unsupported / unknown を resolver の測定で補完、UNKNOWN は選択不可） |
| G95 CI に audio-production-skill が無い | — | tests.yml が main を clone、`VIDEO_AGENT_AUDIO_PRODUCTION_DIR` |
| G96 ローカル ffmpeg-skill が 0.9.0 | audio-production-skill は 0.9.1 以上を要求 | ローカル clone を origin/main（0.9.1、2abd89c）へ更新。agent の対応範囲（< 0.10）内。CI は HEAD を clone しており run #37 以降は 0.9.1 で green |

四段階: Skill supports 14 / adapter supports 14 / Planner generates 9 / E2E verified 3 ケース（A 音声のみ、B video container、C 2 入力 concat + normalize）。

未検証（accepted limitation）: Windows / macOS での実 audio-production-skill E2E（CI の integration job は ubuntu のみ。Windows / macOS では fake process による unit / evals のみ）、wav 以外の出力 format（mp3 / m4a / flac …）、5.1 / 7.1 入力の DOWNMIX（fake のみ）。

未対応・別 PR 候補: MIX（複数入力のレベル指定 requirement）、NOISE_REDUCTION / DYNAMICS（根拠となる measurement が無い）、`audio.<op>.approval` の request 指定、video+audio asset で video と audio の両方を納品する二経路、QC Skill との接続（qc-skill は PR #1 未マージ）、mp3 / m4a 等の audio delivery format（現状 wav のみ）。

## 22. PR #22 — Phase 3 integrated production pipeline（ADR-031 / ADR-032、2026-09-05 追記）

- 統合済み: subtitle / thumbnail / color-grading / motion-graphics / qc の adapter・capability・registry・Decision・ProductionPlan・IR section・compiler・QA・artifact。fake 10 scenario（`tests/test_pipeline.py`）、実 Skill 10 scenario（`tests/test_integration.py::IntegratedPipelineRealTests`）、evals 89–97。
- 未実装（次の作業）: 「話者が変わったらカメラを切り替える」。SPEAKER / CAMERA event は schema のみ、speaker_id は常に null（推定禁止）。ProductionContext の `transition` inference は存在するが、それを実行する operation（multicam 切替）を持つ Skill が無く、`multi_source_sync` は phase 2 未実装。必要なもの: (1) 同期 Skill（offsets）、(2) source 切替 operation を持つ編集 Skill、(3) `transition` inference → `camera.switch` decision（CONFIRM）→ plan step。Event から step への近道は作らない。
- Accepted limitation: thumbnail `extract_frame` は ffmpeg-skill look の既定幅 1280 で出る（agent は寸法を期待値にしない）。subtitle-skill は workspace 相対入力のみ → burn-in は中間物にのみ適用（source 直接は BLOCK）。motion-graphics の title / lower_third / image_overlay は doctor が unknown → resolver の filter 実測で AVAILABLE（実測できない環境では UNKNOWN → BLOCK）。thumbnail-skill は Pillow 未導入だと MISSING。qc-skill の request timeout は Skill 側で未使用（process boundary の timeout のみ）。Windows / macOS の実 E2E、HDR 素材の HDR_TO_SDR / LUT / STRIP_DOVI、lower_third / image_overlay の実描画は未検証（unit fake のみ）。
- P1（次PR）: audio.extract の CONFIRM waiver（PR #21 audit）は据え置き → §23（ADR-033）で修正。QC WARN の promotion policy は profile の `qc.warn.promotion` で制御、既定 CONFIRM。

## 23. P1-1 — `audio.extract` の CONFIRM waiver（ADR-033、2026-09-05 追記）

| Gap | 旧実装 | 修正 |
|---|---|---|
| G94 generic な `audio.production=true` が `audio.extract` の CONFIRM を waive した | `approval_for("audio.extract", explicit=<switch>)` | `explicit` は専用 requirement `audio.extract=true` のみ（語彙追加、switch 無しは ambiguous 拒否） |
| G95 `audio.extract` decision を cite する operation が無く、reject しても audio 操作が走り得た | extract decision は step / op に紐付かない | video container の audio op / step / delivery target が extract id を cite。reject → rejected-cited BLOCK、revise 後は audio path に何も計画しない |
| G96 IR の `approval` / `status` を書き換えると gate を抜けられた | validator は BLOCK ⇔ BLOCKED のみ | `check_decisions`: approval は basis の resolved より緩くできない、APPROVED には review record 必須 |

- 検証: unit `AudioExtractConfirmTests`（8 case: generic switch → CONFIRM / 専用 requirement → AUTO / 他 audio op 不変 / BLOCK 不変 / reject・revise / resume / explain / IR・validator・compiler）、evals 81 更新 + 98 / 99、integration `test_video_container_delivers_audio` に CONFIRM + review record + `audio.extract=true` の assert 追加。
- 残 P1: speaker → camera switching（§22）。残 P2: approve 後の revise で USER_DECISION event が旧 decision id を cite し validator error（main で再現、pre-existing）。
