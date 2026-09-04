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
