# Phase 1 最終レビュー — PR #1 を Phase 2 の基盤として評価する

対象: PR #1（`claude/ffmpeg-skill-architecture-review-5j5lzc`）、監査 `docs/AUDIT_PHASE1.md` の修正後。
方法: コードの grep 検証、テスト再実行、追加の実行実験（REJECTED 決定の扱い、op id の安定性）、GitHub の check run 取得。

---

## 1. AUDIT_PHASE1 A 分類修正の反映確認

| # | 修正 | コード | テスト | 備考 |
|---|---|---|---|---|
| A1 同名ファイル衝突 | ✅ `compiler.py` `stem = f"{idx:02d}_..."` | ✅ `test_same_file_name_twice_gets_distinct_paths` | 実 render でも `01_clip_youtube.mp4` / `02_...` を確認 |
| A2 プロセスグループ kill / 部分出力削除 | ✅ `adapter.py` `run_process_group`, `kill_tree`, `_remove_partial` | ✅ `test_timeout_kills_the_whole_process_group`（ps 検査、POSIX のみ） | Windows の `taskkill /T` 経路は**未検証** |
| A3 SIGINT → CANCELLED + 永続化 | ✅ `executor.py` `INTERRUPTED`、`service.py` `finally: store.save` | ✅ `test_interrupt_marks_job_cancelled_and_persists` | 実 SIGINT で job.json=CANCELLED、provenance.json 生成、孤児なしを確認 |
| A4 strategy の正直な記録 | ✅ `FULL_ANALYSIS`, `budget.enforced=False` | ✅ `test_analysis_strategy_is_honest` | `ir.py` の雛形も本レビューで揃えた |
| A5 Windows UTF-8 / normcase | ✅ `PYTHONUTF8`, `errors="replace"`, `normcase` | ❌ Windows 実機テストなし | 静的対応のみ |
| A6 死んだ `render(dry_run)` | ✅ 削除（`service.render` に `dry_run` 無し） | — | |
| A7 QA 二重実行 / incident 名 | ✅ `check_by_artifact`, `DURATION_MISMATCH` | ✅ `test_qa_measurements_are_recorded` | |
| A8 未知 `--set` キー拒否 | ✅ `REQUIREMENT_PREFIXES` | ✅ `test_unknown_requirement_key_is_rejected` | |
| A9 テスト整備 | ✅ | unit 23 / integration 6 / evals 6 | |
| A10 plan ファイル名に profile | ✅ `cli.py` | ✅ `test_cli_round_trip` が生成パスを使用 | |

CI への反映: **リポジトリに CI は存在しなかった**（下記 2）。本レビューで `.github/workflows/tests.yml` を追加した。

## 2. PR #1 の CI 結果

- GitHub の check run: **0 件**。CI ワークフローが無いため、PR 上では何も検証されていない。
- ローカル検証（Linux, Python 3.11, ffmpeg 6.1.1, ffmpeg-skill 0.8.4）: unit 23/23、integration 6/6、evals 6/6、`ast.parse(feature_version=(3,9))` 全ファイル通過。
- 追加したワークフローは `workflow_dispatch` のみ（ffmpeg-skill 側と同じ方針。Actions 分数の制約による）。unit を ubuntu / windows × Python 3.9 / 3.11、integration を ubuntu で実行する構成。**手動実行されるまで PR 上は未検証のまま**。
- 未検証項目: Windows 全般、Python 3.9 の実行（構文のみ確認）、実機素材（合成素材のみ）。

## 3. Project IR 1.0 は Phase 2 の AI Intent / Decision / Conference 制御に耐えるか

耐える部分:

- `requirements[].provenance` に `AI_GENERATED` が既にある。`decisions[].provenance` は自由文字列、`inferences[]` は evidence 必須で validator が参照を検査する。AI の出力を「evidence 無しの推論」として混入させる経路は無い。
- `decisions[].params` / `alternatives` / `plan.steps[].params` は自由形式で、AI が候補を列挙する用途に使える。
- `project.production` は予約済み（`object|null`）。`timeline` は per-asset offset / drift を持ち、multicam の式と一致している。
- root `additionalProperties: false` なので、新セクションは schema バージョン更新を伴う（意図どおり）。

耐えない / 手当てが必要な部分（Phase 2 の最初に schema 1.1 で対処）:

| 項目 | 現状 | 必要な変更 |
|---|---|---|
| AI 生成物のメタデータ | 置き場が無い | `provenance.ai_calls[]`（provider, model, prompt_hash, response_hash, tokens, at）と、各 Requirement / Inference / Decision に `origin_call_id` |
| Intent | `primary` 1 値 + secondary | AI が返す `candidates[{intent, confidence, evidence}]` と選択理由 |
| Event.kind | `OBSERVED / INFERRED / USER` | AI 由来は `INFERRED` + `source: "ai:<provider>"` で表現できる。enum 追加は不要だが規約として文書化 |
| REJECTED の扱い | 監査時点では**未強制**（REJECTED でも compile された） | 本レビューで validator error + render 拒否に修正済み。Planner が REJECTED を避けて再計画する経路は Phase 2 |
| `ir_hash` | decisions を含むため承認で変わる | `plan_hash`（decisions の status を除外）を追加し、「承認前後で計画が同一か」を判定可能にする |
| operation 語彙 | `video.trim`, `audio.loudness` のみ | conference 用に `video.switch`（multicam）、`audio.replace`、`timeline.align`（sync）を追加 |
| Session | 無し | `project.production.events[].sessions[]` の schema と、Session 境界 Event |
| naming | 未使用 | `delivery.targets[].filename` を naming テンプレートから生成 |

判断: **構造は Phase 2 に耐える。schema 1.1 への拡張はすべて加算的で、1.0 → 1.1 の migration は空でよい。**

## 4. ffmpeg-skill Adapter の責務境界は将来も崩れないか

守られている境界:

- ffmpeg 引数は `adapter.py` の外に存在しない（`grep` で確認）。Agent 側の ffmpeg 直接呼び出しは `capabilities/resolver.py` の機能検出のみ。
- argv は `catalog.py` の型からのみ生成され、未知フラグは `ToolError`。`shell=False`。
- sync / multicam / caption / scenes は catalog に既に宣言済みで、Phase 2 で adapter を触らずに compiler が使える。

崩れうる点と防止策:

| リスク | 防止策 |
|---|---|
| `measure()` が `Operation` を経ずに呼ばれ、provenance に残らない | 本レビューで QA 側の計測を `qa.measurements[]` に記録。analysis 側は `analysis.tool_calls[]` に記録済み。Phase 2 で両者を `provenance.measurements` に統合 |
| catalog に無いスクリプト（batch, verify, report, graphics, overlay, audio, color, fit）を使いたくなり、`argv` 直渡しを足したくなる | `argv` パススルーは**追加しない**（契約テストで `build_argv` が未知キーを拒否することを固定済み）。必要なら catalog に型付きで追加する |
| 別レンダラ追加時に `ToolAdapter` 基底が不足 | 基底は `describe / supports / preview / run` の 4 つ。`measure()` を基底に昇格させる（小変更） |
| ffmpeg-skill のバージョンドリフト | `locate.py` が 0.8.4 ≤ v < 0.9 を要求、契約テストが `--help` のフラグ存在を検査。`--json` の**キー**は未検査（Phase 2 の最初で `result_keys` 検査を追加） |
| allowed input が入力から導出される | ADR-010。`--allowed-input` 設定は Phase 2 |

判断: **境界は堅い。adapter に手を入れるのは catalog 追加と `measure()` の基底化だけでよい。**

## 5. AI Provider を追加したとき Observation / Provenance / Decision に正しく接続できるか

接続点の設計（現状のコードで受け止められるもの）:

```
AIProvider.extract_requirements(text) → Requirement(provenance=AI_GENERATED)   … requirements.py の順位表では USER より下
AIProvider.infer(observations)        → Inference(provenance=AI_GENERATED, evidence=[obs ids])  … validator が evidence を検査
AIProvider.propose(decision_context)  → Decision(provenance=AI_GENERATED, approval ≥ CONFIRM)
```

守るべき規約（Phase 2 で実装・テストする）:

1. AI は Observation を**生成できない**。`Observation.source` は必ず `ffmpeg-skill/...` か明示ツール。
2. AI 由来の Inference は evidence 必須。evidence が無ければ validator error（現状の検査で担保される）。
3. AI 由来の Decision は既定 approval を `CONFIRM` 以上にし、`risk` は AI の申告ではなく policy 表で決める（`decision.py` の表を使う）。
4. すべての AI 呼び出しを `provenance.ai_calls[]` に記録し、応答本文は保存せずハッシュのみ（シークレット・個人情報の混入防止）。
5. `analysis.budget.max_ai_calls` を強制する（現状 `enforced=false`）。

現状の不足: `AIProvider` 基底に `extract_requirements` しか無い。`infer` / `propose` の署名と `ai_calls` の記録先が無い。**Phase 2 の最初の実装項目に入れる**。

## 6. Job / Artifact / Provenance / QA は resume・revision・approval に拡張可能か

| 拡張 | 現状の土台 | 不足（Phase 2） |
|---|---|---|
| resume | `job.completed_ops` に idempotency_key→出力が保存される。compile は job_dir を与えれば決定論的。本レビューで **op id を決定論化**（tool+args+inputs のハッシュ）したので provenance と照合可能 | `render --job <id>` で既存 Job を読み、`completed_ops` を Executor に渡す経路。中断時の中間ファイル検証（hash） |
| revision | `plan.version` と `plans/` ディレクトリは存在。Decision に `alternatives` あり | Feedback モデル、PreferenceCandidate、Plan v2 生成、PlanDiff、REJECTED を避けた再計画 |
| approval | `approve()` が status と `execution.approvals{by,at}` と `USER_DECISION` Event を記録 | `reject()`（理由付き）、部分承認の UI、`who` の実体（現在は OS ユーザ名） |
| artifact 昇格 | `stage` フィールドと `working/candidate` 遷移 | `approve-artifact` / `finalize` コマンド、final の書き込み禁止、archive |
| QA | 4 層 + incident + measurements | Semantic QA、incident 検出（ffmpeg-skill 側提案） |

判断: **拡張可能。ただし「REJECTED の強制」と「op id の決定論化」が無いままだと resume/revision は正しく動かないので、本レビューで先に修正した。**

## 7. 現在のテストで検証できていない重要な失敗系

| 失敗系 | 現状 | 優先度 |
|---|---|---|
| Windows でのプロセスグループ kill（`taskkill /T`） | 未検証 | 高（現場 PC は Windows の可能性） |
| ディスクフル | 分類表にあるが未テスト | 中 |
| 破損メディア / ffprobe 失敗 | analyze が例外で終了（job 未生成）。挙動は妥当だがテスト無し | 中 |
| 音声無し・映像無し・0 秒 | 音声無しは warning 経路あり、テスト無し | 中 |
| render 途中で ffmpeg-skill が消える / バージョン不一致 | `locate` は起動時のみ | 低 |
| 同一 workspace での並行 render | job id は乱数、plans ファイルは profile 別。衝突は無いはずだがテスト無し | 低 |
| 巨大ファイルの sha256（数十 GB） | `--no-hash` で回避可能、時間は未計測 | 中（学会素材は大きい） |
| IR の手編集（asset path 変更、keep 範囲改変） | validator が範囲・存在を検査。REJECTED は本レビューで追加 | 低 |
| QA の check.py がクラッシュ | `WARN no result` 経路、テスト無し | 低 |
| 承認済み IR を再 plan で上書き | plan は常に新規作成、承認は失われる | 中（revision 設計で扱う） |

## 8. Windows 対応で残る具体的な問題

1. **未検証**: すべて静的レビューに基づく。CI マトリクスに windows-latest を入れたが手動実行が必要。
2. `kill_tree` の `taskkill /F /T /PID` は console の無い環境や権限で失敗しうる。失敗時は `proc.kill()` にフォールバックするが孫 ffmpeg は残る可能性。
3. Ctrl-C: `CREATE_NEW_PROCESS_GROUP` で子は Ctrl-C を受け取らず、親の `KeyboardInterrupt` → `kill_tree` に頼る。動作は理屈上正しいが未検証。
4. パス長 260 文字: `jobs/<id>/ops/<NN_stem>_01_trim/<NN_stem>_trim.mp4` は長い日本語ファイル名で超えうる。長パス有効化か短縮が必要。
5. `fc-list` 無し → `font:cjk-ja` が UNKNOWN。Phase 3 の字幕で必須。Windows は `dir %WINDIR%\Fonts` 相当の検出が必要。
6. ffmpeg-skill 自体の Windows 挙動（`python3` 前提のドキュメント、`fontsdir` のパスエスケープ）は本プロジェクト外だが依存する。adapter は `sys.executable` を使うので `python3` 問題は回避済み。
7. テストの `ps -eo args` は Windows でスキップされる。`tasklist` 版が必要。
8. `getpass.getuser()` は問題なし。`os.path.normcase` 適用済み。UTF-8 は子プロセス環境変数で対応済み。

## 9. Phase 2 へ進む前に絶対に修正すべき項目

本レビューで修正済み（PR #1 に含む）:

- [x] REJECTED 決定を参照する operation を validator が拒否し、render が実行しない。
- [x] operation id の決定論化（provenance / resume の照合に必須）。
- [x] QA の計測呼び出しを provenance に残す（`qa.measurements[]`）。
- [x] IR 雛形の `strategy` 既定値の整合。
- [x] CI ワークフローの追加（手動実行）。
- [x] PR 説明の更新（テスト件数・監査の反映）。

Phase 2 開始前に**人手で必要**なこと:

- [ ] `tests.yml` を GitHub 上で一度手動実行し、Windows unit と ubuntu integration の結果を確認する（Actions 分数の判断はオーナー）。
- [ ] PR #1 のレビューとマージ（または main への取り込み方針の決定）。

## 10. Phase 2 で最初に実装すべき機能（優先順位付き）

| 順位 | 機能 | 理由 | 依存 |
|---|---|---|---|
| 1 | **schema 1.1**: `plan_hash`、`provenance.ai_calls[]`、`origin_call_id`、`delivery.targets[].filename`、`project.production` の schema、operation 語彙 `timeline.align` / `video.switch` / `audio.replace` | 以降すべての土台。加算的で migration は空 | なし |
| 2 | **Job resume と冪等スキップ**（`render --job <id>`） | 学会素材は長尺で、途中失敗の再実行コストが最大の運用リスク | op id 決定論化（済） |
| 3 | **reject / Feedback / Plan v2 / PlanDiff**（`revise` コマンド） | 人間承認の実運用に必須。REJECTED 強制（済）の上に載る | 1 |
| 4 | **AIProvider の接続規約実装**（`infer` / `propose`、`ai_calls` 記録、`max_ai_calls` 強制、NullProvider で全テスト通過を維持） | AI を入れる前に「AI 無しで同じ IR が出る」保証を固定する | 1 |
| 5 | **conference 前段**: Asset 役割分類（camera_a / room_audio / slides…）、`sync.py` / `multicam.py --offsets-only` の timeline 反映、Production/Event/Session、ターゲット別ラウドネス（master -23 / web -16） | 戦略ドメイン。multicam 合成は `multicam.py --switch` に任せ、切替判断だけを Agent が持つ | 1, 2 |
| 6 | `--allowed-input` と workspace 設定ファイル、naming の適用、artifact 昇格コマンド | 運用上の整合 | 1 |
| 7 | 契約テストに `--json` の `result_keys` 検査を追加、Windows CI の実行、破損/無音/ディスクフルのテスト | 7・8 の穴埋め | CI 実行 |
| 8 | 解析予算の強制と TARGETED 解析 | 長尺対策。5 の前でもよいが、まず resume が効く方が実益が大きい | 2 |

---

## Phase 2 の実装開始条件

以下がすべて満たされた時点で Phase 2 を開始する。

1. PR #1 がマージされている（または main に同等の内容が入っている）。
2. `tests.yml` の手動実行で ubuntu unit / integration が green であること。Windows unit が red の場合は、その修正を Phase 2 の第 0 項目とする（Phase 2 の機能着手より前）。
3. `docs/AUDIT_PHASE1.md` の B 分類（残置）と本レビュー §7・§8 の未検証項目が、オーナーに承認された既知の制限として扱われること。
4. Phase 2 の最初の PR は **schema 1.1 + resume** に限定し、conference 機能と AI Provider はそれぞれ別 PR にする。

これらは判断を要する事項ではなく確認事項であり、1 と 2 の結果が揃えば着手できる。

---

## 追記: CI 実行結果（2026-09-04, run 33870052758, head 1f31d70）

`tests.yml` を `workflow_dispatch` で PR ブランチに対して手動実行した。ワークフローは default branch に無いと dispatch できない（404）ため、同一ファイルだけを PR #2 として main に先行マージしてから実行した。

| job | 結果 | 内容 |
|---|---|---|
| unit (ubuntu-latest, 3.9) | success | 23 tests OK, evals 6/6 |
| unit (ubuntu-latest, 3.11) | success | 23 tests OK, evals 6/6 |
| unit (windows-latest, 3.9) | success | 23 tests OK, evals 6/6（ログで件数確認） |
| unit (windows-latest, 3.11) | success | 23 tests OK, evals 6/6 |
| integration (ubuntu-latest, ffmpeg apt + ffmpeg-skill clone) | success | 6 tests OK（38.8 s） |

- Windows unit が green のため、「Phase 2 開始前の第 0 項（Windows 修正）」は不要。
- 警告: actions/checkout@v4 と setup-python@v5 の Node 20 非推奨警告のみ（動作影響なし。次回 CI 更新時に v5/v6 へ）。
- 未検証のまま残るもの: Windows での integration（ffmpeg 実行、`taskkill /T`）。unit は FakeAdapter のため OS 依存コードの一部（プロセスグループ kill）は Windows で実行されていない。
- 開始条件 1〜4 のうち、2 は本追記で充足。1 は PR #1 のマージで充足する。
