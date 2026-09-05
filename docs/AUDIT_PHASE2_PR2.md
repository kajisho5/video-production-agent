# Phase 2 PR 2 監査（revision workflow）— 実験ベース

対象: REJECT → Feedback → Revision → Plan v2 → PlanDiff → 再 Approval → render。
方法: 実 ffmpeg 6.1.1 + ffmpeg-skill 0.8.4、合成素材 `talk.mp4`（16 s、先頭 3 s / 末尾 2.3 s 無音）、`conference` / `youtube` プロファイルで CLI を実行。指定された 10 ケースをすべて実際に実行し、失敗させた上で状態を確認した。

## 必須ケースの結果

| # | ケース | 手順 | 結果 | 判定 |
|---|---|---|---|---|
| R1 | 全 reject → render | `reject --decision all --reason ...` → `render` | **BLOCKED**（exit 3）。4 件の理由を列挙し `revise` を案内。ツールは一切起動しない | ✅ |
| R2 | 一部 reject → rejected operation が実行されない | `reject <silence.leading> --reason "座長の紹介が含まれる" --by reviewer:kaji` → `render` / `render --approve all` | どちらも **BLOCKED**。approve で却下を上書きできない。`approve` コマンドも拒否（`plan cites REJECTED decisions`） | ✅ |
| R3 | reject → revise → v2 → diff → approve → render | `revise` → `diff` → `approve --decision all` → `render` | v2 生成、`c.v1.json` 保存。PlanDiff: `REJECTED silence.leading … — 座長の紹介が含まれる` / `VIDEO trim 2.850-16.000 → removed`。render は COMPLETED、QA PASS 26/26、**納品物は 16.0 s**（trim が実行されなかった物証） | ✅ |
| R4 | v2 は再 Approval まで render 不可 | approve 前に `render` | **WAITING_FOR_APPROVAL**（CONFIRM decision が 0 件でも）。approve 後に実行可能 | ✅ |
| R5 | v1 と v2 の Provenance が独立 | v1 を render → loudness を reject → revise → approve → v2 を render（`--resume` なし） | 別 Job。v1 provenance: plan_version 1、操作 cut/loudness/export/check、reviews 空。v2: plan_version 2、cut/export/check、reviews に REJECTED。`skipped=[]`、`resumed_from=None` | ✅ |
| R6 | v1 の completed_ops を v2 で誤再利用しない / 変更していない操作だけ再利用 | v2 を `--resume <v1 job>` | `plan_changed=True`、再利用は **cut のみ（1）**。export は入力（loudness の有無）が変わったので再実行 | ✅ |
| R7 | rejection reason が revision 後も残る / approval 後の revision は再承認が必要 | v2 approve → `revise --set edit.precision=frame` | v3 生成（PlanDiff: `trim 0.000-13.868 → 0.000-13.868 (frame-accurate)`）、`approved_plan_version=None`、render は WAITING。v1 の却下理由は v3 の `execution.reviews` と `history[].rejection_reasons` に残存 | ✅ |
| R8 | 同じ revision を 2 回 | 同一の `revise --set ...` を再実行 | `no new plan version`（exit 5）。version / history / snapshot は不変、feedback は 1 件のまま（重複記録なし） | ✅（修正後） |
| R9 | SIGINT / failure / resume と revision の組み合わせ | v3 approve → `render --timeout 2.2`（export でタイムアウト）→ `render --resume last` | FAILED（v3、cut 完了）→ resume で cut を再利用し COMPLETED（v3）。`provenance.runs` に (3, FAILED), (3, …) が版番号付きで残り、却下理由も残存 | ✅ |
| R10 | 1.0 / 1.1 IR の読込 | 1.1 の IR を `load` | 1.2 へ migration。既存 `approvals` は `reviews` の APPROVED 記録に変換 | ✅（unit） |

補足: R5 / R6 / R9 の youtube プロファイルでは loudness を却下したため、納品物 QA は `check.py` の loudness 項目（-14±2 LUFS）で FAIL → Job は REVIEW。これは revision の問題ではなく、却下の結果として納品規格を満たさなくなったことを QA が正しく報告している。

## 実験で見つけて修正した問題

| 問題 | 発見 | 修正 |
|---|---|---|
| 却下を含む plan の render が **FAILED**（validation error）で返り、BLOCKED にならない | R1 | render は validation より前に `rejected_cited()` を検査し BLOCKED + 理由を返す |
| 効果の無い feedback（却下済み subject の再提案）でも v3, v4 … と**空の版**が増える | R7/R8 初回 | 新版は「diff が空でない」か「新しい却下を適用する」場合のみ。効果の無い feedback は 1 回だけ記録（同一内容は重複させない）し、版は上げない |
| `edit.precision=frame` が requirements にしか無く、plan_hash / PlanDiff に現れない（compiler だけが読む） | R7'' | `video.trim` op に `accurate` を持たせ、compiler は op から読む。plan_hash / diff に反映（`(frame-accurate)`） |
| `plan_hash` に `schema_version` が含まれ、migration 前後で同内容が別ハッシュ | migration テスト | `schema_version` をハッシュから除外 |
| `revise` が feedback を deep-copy 前の doc に追加していて次版に載らない | unit | 次版は現 doc の feedback を引き継ぐ |
| decision の status 変化（APPROVED→PROPOSED）が PlanDiff に混入 | unit | status を diff 対象から除外し、却下は `history.rejection_reasons` から summary に出す |

## 安全条件の対応

| 条件 | 実装箇所 |
|---|---|
| 1 REJECTED を含む plan は実行不可 | `Service.render` の最初の gate、`validate_ir`、`Service.approve` |
| 2 部分承認された operation 以外は実行不可 | 却下 op は revise で消える。残りは CONFIRM 承認 + plan 版承認（`approved_plan_version`） |
| 3 reject 理由を失わない | `execution.reviews` を次版へ引き継ぎ、`history[].rejection_reasons` に複製、`USER_DECISION` Event |
| 4 actor / timestamp | `reviews[].by/at`（`--by`、既定は OS ユーザ） |
| 5 v1 を破壊しない | `<stem>.v<N>.json` に保存してから置換、既存 snapshot は上書きしない |
| 6 v1 Job / Provenance を v2 で再利用しない | Job.plan_version、`--resume` なしでは再利用ゼロ（R5） |
| 7 PlanDiff | `project/diff.py`、`revision.history[].diff`、`diff` コマンド |
| 8 v2 は再承認まで render 不可 | `needs_reapproval()` → WAITING_FOR_APPROVAL（R4, R7） |
| 9 ir_hash / plan_hash | plan_hash: approve/reject 不変・revise で変化。ir_hash: approve/reject で変化（unit `test_hashes_approval_vs_revision`） |
| 10 resume と revision の分離 | key は plan 内容から導出。版をまたいでも同一操作のみ再利用（R6） |

## テスト

- unit 46/46（revision 系 12 件: reason/actor/timestamp、却下 plan の render 不可、全却下、部分却下→diff→承認→render、hash 関係、v1/v2 provenance 独立 + 版またぎ resume、承認後 revision の再承認と履歴、同一 revise 2 回、効果の無い feedback、却下・feedback 無しの revise、v2 内での失敗+resume、revise がメディアを触らない）。
- integration 9/9（実メディアで reject → BLOCKED → revise → WAITING → approve → render → 16.0 s の納品物 + QA PASS）。
- contract: revision 系コマンドがツールを一切起動しないことを、全スクリプトが exit 99 する偽の ffmpeg-skill で確認。
- evals 6/6。

## 残る制限（この PR では扱わない）

- 却下は (subject, asset) 単位で以後の提案を抑止する。「-14 は却下したが -18 なら欲しい」を区別できない（AI 無しの規則）。抑止された提案は `dropped_proposals` と `revise` の出力で明示される。却下の取り消し（un-reject）は未実装。
- `revise --feedback` の自由文は Phase 1 と同じキーワード規則でのみ解釈される。
- 部分承認は decision 単位。operation 単位の承認 UI は無い（operation は decision を通じて承認される）。
- Windows 実機検証は未実施（CI は unit のみ）。
