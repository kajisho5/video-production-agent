# Phase 1 監査（2026-09-04, commit 580ac94 時点）

方法: 全ソース（約3,000行）を再読し、以下を実際に実行して確認した。テストが通ることは根拠にしていない。

| 実験 | 結果 |
|---|---|
| 同名ファイル2本（`camA/clip.mp4`, `camB/clip.mp4`）を1 plan に | 中間ファイルと納品物のパスが**完全に衝突**（`ops/01_trim/clip_trim.mp4`, `artifacts/clip_youtube.mp4` が2回） |
| `render --timeout 1.5` | cut.py は kill されるが孫の `ffmpeg` が**生存し続け**、再試行が同じ出力ファイルへ並行書き込み |
| render 中に SIGINT | CLI はトレースバックで終了、`job.json` は **EXECUTING のまま**、IR 未保存、ffmpeg 孤児化 |
| 同じ IR を2回 render | 毎回新 Job。`completed_ops` は読まれず、冪等スキップは**一度も発動しない** |
| シンボリックリンク入力 `link/evil.mp4 → /etc/hostname` | allowed root が「解決後パスの親」= `/etc` になり、ffprobe が `/etc/hostname` を読む（拒否されない） |
| `--set 'audio.loudness.target_lufs=abc'` | `could not convert string to float` の生例外（plan 失敗、危険性なし） |
| `--set api_key=sk-SECRET` | 未知キーがそのまま IR の `request.args` と `requirements` に**保存される** |
| `doctor --json`（API キー設定時） | 値は出ない（presence のみ）。OK |
| 日本語パス・スペース入りパス | Linux では問題なし |
| source を workspace 内に置く | 出力は `jobs/<id>/` 配下、source 不変（sha256 一致）。OK |
| 承認前後の `ir_hash` | 承認で decision.status が変わるため hash が変わる（設計どおりだが要注記） |
| 同じ入力で profile を変えて plan | `plans/<stem>.project.json` を**無言で上書き** |

---

## 1. MASTER_SPEC / ARCHITECTURE_REVIEW に対する一致状況

凡例: ✅ 仕様通り / 🔶 仕様の簡易実装 / ❌ 未実装（設計のみ）

| 領域 | 状態 | 備考 |
|---|---|---|
| §3 責務境界（ffmpeg 直接呼び出しは機能検出のみ） | ✅ | `grep -r "ffmpeg" src` で subprocess 呼び出しは `capabilities/resolver.py` のみ |
| §6 Request / Requirements / Intent の分離と provenance | 🔶 | provenance は正しく付く。Request 解釈はキーワード5個のみ。Intent は3値ルール |
| §7 Observation / Inference 分離 | ✅ | 別型、evidence 必須、validator が参照検査 |
| §8 Temporal Model | 🔶 | Event 型・per-asset timeline・`query()` は実装。生成される Event は 4 種のみ。offset/drift は保持のみで使用箇所なし |
| §9 Event/Session/Production | ❌ | `project.kind="single"` 固定、`production: null` |
| §10-11 Asset / 関係 | 🔶 | Asset 型あり。分類は probe だけの粗い規則（confidence 0.4-0.9）。relationships は常に空 |
| §12-13 Skill / Capability / Tool / Registry | ✅ | 契約は完成。実行される Skill は 5 つ |
| §14 Capability Resolver | ✅ | 4状態。GPU は「HWエンコーダが列挙されるか」の推定 |
| §15-16 Decision / Risk / Approval | 🔶 | 構造・AUTO/CONFIRM/BLOCK・alternatives は実装。approval の決定則は無音/ラウドネス/HDR/capability の 4 種のみ。**reject が無い** |
| §17 Policy / Preference / Constraint | ✅ | precedence、constraint 不可侵、conflict → CONFIRM |
| §18 Profiles | 🔶 | generic / youtube / conference。social / webinar / broadcast / archive は無い |
| §19 Conference | ❌ | 安全ルールと delivery targets の骨組みのみ |
| §21-22 Production Plan vs Project IR | ✅ | 分離、schema_version、JSON Schema |
| §23 決定論的実行 | 🔶 | IR 以降は決定論的だが、tool 側の自動判断（cut.py の copy→再エンコード切替）は provenance の `commands` で事後把握のみ |
| §24-25 Compiler / Adapter | ✅ | ffmpeg 引数は adapter の外に存在しない |
| §26 Dry Run | 🔶 | operations / capabilities / outputs / risks / warnings は出る。処理時間推定は「再エンコード回数」だけ |
| §27 Explainability | ✅ | `explain` が reason/evidence/alternatives/provenance を出す |
| §28-29 Feedback / Revision / Plan diff | ❌ | 未実装。plan version は常に 1 |
| §30 QA | 🔶 | video/audio/delivery/visual(シート生成のみ) 実装。black/freeze/corruption/dropout/phase は未実装（ffmpeg-skill にも無い） |
| §31 Incident | 🔶 | 型あり。生成は QA FAIL からのみ。**duration 不一致の type 名が誤り**（UNEXPECTED_SILENCE/WRONG_FPS を流用） |
| §32 Recovery | 🔶 | 分類表・有限再試行は実装。**タイムアウト時に孫プロセスが残る** |
| §33-34 コスト / 予算 | ❌ | `analysis.strategy` は **常に全ファイル解析なのに TARGETED と記録**、`max_processing_time` は記録のみで未強制 |
| §35 キャッシュ / 冪等 | 🔶 | idempotency_key は計算されるが Job 間で参照されず**事実上無効** |
| §36 Artifact | 🔶 | 型・hash・stage あり。generic プロファイルでは artifact パスが `ops/` 配下 |
| §37 Job / Lifecycle | 🔶 | 状態機械あり。**キャンセルは API のみで CLI から到達不能**、resume 無し、SIGINT で状態が壊れる |
| §38 冪等性 | 🔶 | source 破壊なし。重複 artifact は job ごとに増える（設計許容） |
| §39 昇格 | 🔶 | working / candidate のみ。approved / final / archive への遷移操作なし |
| §40-41 Provenance / 再現性 | 🔶 | 決定→操作→コマンド→結果は紐付く。`who` は固定文字列 `"user"` |
| §42 AI Provider | ✅（境界のみ） | NullProvider |
| §43 Security / Workspace | 🔶 | 出力は workspace 内・入力上書き禁止は強制。**allowed input は入力パスから自動導出され、独立した制限になっていない** |
| §44 Evals | 🔶 | 6 ケース。FakeAdapter 上の決定論テストであり、実メディアや AI 推論の回帰にはなっていない |
| §45 Doctor | ✅ | |
| §46 CLI | 🔶 | 9 コマンド中 7 実装（`analyze plan validate render check doctor explain`）。`run/jobs/inspect/diff/revise/archive` 無し |
| §49 学会安全策 | 🔶 | constraint としての宣言のみ。発話内容の保護区間は ASR 無しでは検出不能 |
| §50 Naming | ❌ | `naming` テンプレートは保存されるだけで未使用 |
| §51 Testing 層 | 🔶 | Unit / Integration / Contract / Evals あり。Real Media は合成素材のみ（実機素材・破損素材なし） |
| §52 リポジトリ構成 | ✅ | |

---

## 2. テストの実効性

| テスト | 検証しているもの | 検証していないもの |
|---|---|---|
| unit 16 件（FakeAdapter） | 型・配線・provenance・approval フロー・recovery 分岐・path policy・schema 拒否 | ffmpeg-skill の実挙動。FakeAdapter は inference が期待する形の silence/loudness を返すので「推論が正しい」証明にはならない |
| `test_generic_without_preset_delivers_intermediate` | **ほぼ何も検証していない**（`assertEqual(tools, X if cond else Y)` は同語反復） | 修正対象 |
| integration 3 件 | 実 ffmpeg + ffmpeg-skill で plan → render → QA PASS、HDR/VFR の観測→決定、CLI 往復 | 「冪等再実行」はコメントにあるだけで**テストしていない**（実際は動かない） |
| contract 2 件 | ffmpeg-skill のバージョン範囲、catalog の全フラグが `--help` に存在 | `--json` の出力キー（`result_keys`）は未検証 |
| evals 6 件 | intent / decision / provenance / block の期待値 | unit テストの JSON 版であり、回帰以上の意味は無い |

結論: **配線の正しさは担保されているが、失敗系（タイムアウト孤児・SIGINT・同名衝突・再実行）は一つも検証されておらず、実際に壊れていた。**

---

## 3. Project IR の実運用耐性

- 構造・schema・migration 入口は実運用に耐える。
- 弱点: (a) `video.trim` 1 語彙のみ、(b) 承認で hash が変わるため「承認前の IR と同一か」は `ir_hash` では判別できない（decisions を除いた hash が別途必要）、(c) `naming` 未使用、(d) `request.args` に任意キーが保存される。
- IR は `plans/<stem>.project.json` に無言上書きされるため、プロファイル違いの計画を並行して持てない。

## 4. ffmpeg-skill Adapter の責務

適切。argv は catalog 型からのみ生成され、未知フラグ・型違いは拒否される。問題は adapter ではなく **プロセス管理**（孫プロセス）と **PathPolicy の allowed root 導出**にある。`_parse_json` が stdout 中の最初の `{` から読む点は、スクリプトがパスを先に print するケース（`look.py` 複数出力）に依存した実装だが `--json` では発生しない。

## 5. QA / Provenance / Decision / Risk / Approval は形だけか

- QA: 実測（probe / loudness / check.py / look）に基づく。**conference master の -23/-16 LUFS 不整合を実際に検出した**ので形だけではない。ただし check.py が executor と QA で**二重実行**される（結果の突き合わせロジックが機能していない）。
- Provenance: 決定 → 操作 → 実コマンド → 結果 → QA が紐付く。`who` 固定と、承認者記録が `approvals{}` のみ。
- Decision / Risk / Approval: WAITING_FOR_APPROVAL → `--approve` → 実行の経路は本物。reject と部分承認の記録（誰が・なぜ却下）は無い。

## 6. エラー / 途中失敗 / キャンセル / 再実行

| 状況 | 現状 |
|---|---|
| ツール失敗 | 分類 → 有限再試行 → FAILED/BLOCKED、provenance に記録。OK |
| タイムアウト | **孫 ffmpeg 生存、再試行と二重書き込み** |
| SIGINT | **Job が EXECUTING で固定、IR/provenance 未保存、孤児** |
| キャンセル API | `Executor.cancel()` は operation 境界でのみ効き、CLI から呼べない |
| 再実行 | 常に新 Job。完了済み操作の再利用なし（`completed_ops` は保存されるが読まれない） |
| 途中失敗後の中間ファイル | 残る（意図どおり）。ただし失敗した attempt の不完全ファイルと成功ファイルが同じパス |

## 7. 安全性

| 項目 | 判定 |
|---|---|
| source 上書き | 不可能（出力は `jobs/<id>/` 配下固定 + 出力=入力の拒否）。実験で sha256 一致を確認 |
| 任意シェル | 不可能（`subprocess.run(list)`、shell=False、argv は catalog 経由） |
| 任意ファイル読み取り | ユーザが与えたパス（およびその親ディレクトリ配下）は読める。シンボリックリンクで allowed root が移動する。「ユーザ自身の入力」なので侵害ではないが、§43 の「allowed input」にはなっていない |
| シークレット | doctor は presence のみ。`--set` の未知キーが IR に残る（ユーザ操作起因） |
| パス処理 | `resolve()` 済み絶対パス。Windows の大文字小文字は未考慮 |

## 8. Windows

静的レビュー（実機未検証）:

1. `subprocess.run(text=True)` がロケール（cp932）でデコードするため、ffmpeg-skill の `print_json(ensure_ascii=False)` 出力で **UnicodeDecodeError** の可能性が高い。子プロセス側も `PYTHONUTF8=1` が必要。
2. `PathPolicy` の `startswith` 比較が大文字小文字を区別（`C:\` vs `c:\`）。`os.path.normcase` が必要。
3. `fc-list` 無し → fonts は UNKNOWN（設計どおり）。`font:cjk-ja` は判定不能。
4. タイムアウト時のプロセスグループ kill は POSIX と Windows で実装が異なる（`CREATE_NEW_PROCESS_GROUP` + `taskkill /T`）。
5. ffmpeg-skill 側は 3OS CI 対象だが、Agent の integration テストは Linux でしか回していない。

---

## A. 今すぐ修正（本監査で修正済み）

| # | 問題 | 修正 |
|---|---|---|
| A1 | 同名ファイルの出力パス衝突 | 中間/納品パスに asset 連番を含める（`ops/<n>_<stage>/<idx>_<stem>...`、`artifacts/<stem>__<idx>_<target>`） |
| A2 | タイムアウト/SIGINT で孫 ffmpeg が残り、再試行が二重書き込み | 子をプロセスグループで起動し、タイムアウト・中断時にグループごと kill（POSIX: `start_new_session` + `killpg`、Windows: `CREATE_NEW_PROCESS_GROUP` + `taskkill /T`）。失敗 attempt の出力は削除 |
| A3 | SIGINT で Job が EXECUTING 固定、IR 未保存 | Executor が `KeyboardInterrupt` を CANCELLED に変換、Service は常に job/IR/provenance を保存 |
| A4 | `analysis.strategy` が実態（全ファイル解析）と異なる | `FULL_ANALYSIS` を記録、`budget.enforced=false` を明示 |
| A5 | Windows デコード | `encoding="utf-8", errors="replace"`、子に `PYTHONUTF8=1`/`PYTHONIOENCODING`、PathPolicy に `normcase` |
| A6 | `render(dry_run=True)` の死んだ経路（chained 中間が無く ffmpeg-skill の `--dry-run` が失敗する） | 削除。dry-run は `Service.dry_run()`（IR ベース）のみ |
| A7 | QA の check.py 二重実行、duration 不一致 incident の誤ラベル | executor の check 結果を op→artifact で対応付け、`DURATION_MISMATCH` を追加 |
| A8 | `--set` の未知キーが IR に保存される | 既知プレフィックス（`edit./audio./silence./delivery.`）以外はエラー |
| A9 | 同語反復テスト、誤解を招くコメント、失敗系テストの欠如 | 修正 + 同名衝突 / プロセスグループ kill / KeyboardInterrupt / 未知キー拒否のテスト追加 |
| A10 | plan ファイルの無言上書き | ファイル名に profile を含める（`<stem>.<profile>.project.json`） |

## B. Phase 1 として残してよい問題（記録済み、動作には影響しない）

- allowed input が入力パスから導出される（明示設定は Phase 2 の `--allowed-input`）。docs/decisions.md ADR-010。
- `who` が固定文字列。承認者・実行者の識別は Phase 2（ジョブ実行者 = OS ユーザ名を記録する程度）。
- generic プロファイルの納品物が `ops/` 配下（コピーせずに参照）。
- `--set` の型エラーが生例外メッセージ。
- Asset 分類の confidence が経験則。conference の役割分類は Phase 2。
- GPU 判定は「ffmpeg がHWエンコーダを列挙するか」のみ（動作保証ではない）。
- Windows は静的レビューのみ（A5 の修正は入れたが実機未検証）。

## C. Phase 2 に回す機能

- Job resume（`render --job <id>`）と Job 間の冪等スキップ（`completed_ops` の再利用）。
- reject / 部分承認とその記録、Feedback → PreferenceCandidate → Plan v2 → PlanDiff。
- `--allowed-input` / ワークスペース設定ファイル。
- conference: Asset 役割分類、Production/Event/Session、sync/multicam offsets の timeline 反映、ターゲット別ラウドネス（master -23 / web -16）。
- Naming テンプレートの適用。
- Analysis budget の強制と TARGETED（先頭/末尾 N 秒 + サンプル）解析の実装。
- artifact の approved / final / archive 昇格コマンド。
- Windows 実機検証（CI マトリクス）。

## D. 将来設計として残すもの

- Incident 検出（black/freeze/astats）は ffmpeg-skill 側に `incidents.py` として提案するのが妥当（Agent に ffmpeg 直接呼び出しを増やさない）。
- 複数音声トラック（room / presentation）: ffmpeg-skill は最初のトラックのみ。`probe.py --streams` の提案。
- AI プロバイダによる要件抽出（provenance=AI_GENERATED）。
- Semantic QA / PROTECTED 区間（ASR 依存）。
- Web UI、job queue、分散レンダリング。
