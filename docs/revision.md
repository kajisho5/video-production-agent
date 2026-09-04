# Revision workflow（Phase 2 PR 2）

```
plan v1 ──reject(id, reason, by)──▶ v1 [REJECTED decision]  ── render → BLOCKED
                                       │
                                  revise(feedback?, --set?)   … v1 を <stem>.v1.json に保存、再計画（メディアは読まない）
                                       ▼
                                  plan v2 [rejected decision は履歴として残り、その operation は生成されない]
                                       │  PlanDiff v1→v2（revision.history[-1].diff）
                                  approve --decision all      … approved_plan_version = 2
                                       ▼
                                  render（v2）                … Job.plan_version = 2、provenance は v1 と独立
```

## 状態と不変条件

| 条件 | 実装 |
|---|---|
| REJECTED を含む plan は実行できない | `render` は validation より前に `rejected_cited()` を検査し BLOCKED。validator も error。`approve` も拒否 |
| 部分承認された operation 以外は実行しない | reject → revise で rejected の operation が消える。残りは CONFIRM なら承認必須、AUTO でも v≥2 は plan 承認必須 |
| reject 理由・actor・timestamp | `execution.reviews[decision_id] = {action, by, at, reason, plan_version}` と `USER_DECISION` Event。revise 後も rejected decision と reviews は次版へ引き継ぐ。`revision.history[].rejection_reasons` にも複製 |
| v1 を壊さない | revise は `<stem>.v<N>.json` を書いてから置き換える。既存 snapshot は上書きしない |
| v1 の Job / Provenance を v2 で誤再利用しない | Job は `plan_version` を持ち、`--resume` 無しでは何も再利用しない。`--resume` 時も chained key で入力が変わった操作は再利用されない |
| PlanDiff | `project/diff.py`: decisions（subject@asset）、video/audio ops（type@asset）、delivery（id）の added/removed/changed + summary。REJECTED 行は理由付き |
| v2 は再承認まで render 不可 | `revision.approved_plan_version != plan.version` → WAITING_FOR_APPROVAL（CONFIRM が無くても） |
| ir_hash / plan_hash | plan_hash: assets/video/audio/delivery/qa の内容。approve/reject で不変、revise で変わる。ir_hash: plan_hash + decisions（status 含む）+ plan。approve/reject で変わる |
| resume と revision の分離 | revision は IR の版、resume は Job の完了記録。key は plan 内容から導出されるため、版をまたいでも「同じ操作」だけが再利用される |
| 同じ revise を 2 回 | 新しい feedback も新しい rejection も無く diff が空なら版を作らない（`created: false`） |

## CLI

```bash
video-agent reject  project.json --decision ID[,ID]|all --reason "..." [--by NAME]
video-agent revise  project.json [--feedback "..."] [--set key=value] [--by NAME]
video-agent diff    project.json [--from N] [--to M]
video-agent approve project.json --decision ID[,ID]|all [--reason "..."] [--by NAME]
video-agent render  project.json [--resume JOB|last]
```

`revise` の自由文は Phase 1 と同じキーワード規則でしか解釈されない（AI 無し）。確実に効かせるには `--set`。
