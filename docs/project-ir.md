# Project IR (schema 1.1)

Project IR は「推論・計画」と「決定論的実行」の間の契約。JSON、`schemas/project.schema.json` で検証する。
`render.py` の project.json とは別物（理由: docs/ARCHITECTURE_REVIEW.md §1.3）。

## 構造

| キー | 内容 | 生成元 |
|---|---|---|
| `schema_version` | `"1.0"`。`project/migrations.py` が旧版を順次変換 | system |
| `project` | id, kind(`single`/`production`), name, profile{name,version,chain} | plan |
| `request` | raw text, received_at, channel, args | CLI |
| `requirements[]` | key/value + provenance(USER/PROFILE/DEFAULT/SYSTEM/…) | RequirementExtractor |
| `source` | agent_version, tool_versions | CapabilityResolver |
| `assets{}` | Asset（path, type, hash, technical=probe, classification{confidence,evidence}） | MediaAnalyzer |
| `analysis` | observations[]（測定）, inferences[]（解釈, evidence 必須）, strategy, warnings | MediaAnalyzer / Inferencer |
| `intent` | primary, secondary, confidence, provenance | IntentResolver |
| `constraints[]` / `policy` | 解決済み Rule と conflicts | policy/rules.py |
| `decisions[]` | subject, decision, reason, confidence, evidence, alternatives, risk, approval, status, params | DecisionEngine |
| `plan` | version, steps[{skill, tool, decision_ids, params}], summary[] (人間向け) | Planner |
| `timeline` | timelines{id: offset, drift_ratio}, events[] | MediaAnalyzer, approve() |
| `video.operations[]` | `video.trim {asset, keep[[s,e]], decision_ids}` | Planner |
| `audio.operations[]` | `audio.loudness {asset, target_lufs, true_peak, decision_ids}` | Planner |
| `captions` / `graphics` / `color` | 予約（Phase 3+） | — |
| `delivery.targets[]` | id, preset(ffmpeg-skill export preset 名), platform(check.py platform 名), artifact_type | Profile |
| `qa` | required layers, thresholds | Profile / defaults |
| `execution` | workspace, allowed_inputs, budgets, recovery_policy{max_attempts}, approvals{} | plan / render |
| `provenance` | source_hashes, profile_version, skill_versions, tool_versions, ir_hash, runs[], recovery[] | plan / render |

## 不変条件（validator が検査）

- inference.evidence は observation / event / inference の id を指す。
- operation は既知の asset と decision を参照し、keep 範囲は 0 < s < e ≤ duration。
- preset / platform は ffmpeg-skill が知っている名前のみ（数値仕様は複製しない）。
- 必要 capability が MISSING の IR は validate で error。
- approval=BLOCK の decision があれば render は BLOCKED、CONFIRM が PROPOSED のままなら WAITING_FOR_APPROVAL。

## コンパイル

`execution/compiler.py` が asset ごとに trim → loudness → export → check の順で Operation を生成する。
中間ファイルは `<workspace>/jobs/<job_id>/ops/NN_<stage>/`、納品物は `artifacts/`。
Operation.args は adapter のカタログ型（`tools/ffmpeg_skill/catalog.py`）で検証され、ffmpeg のオプションはどこにも現れない。

## schema 1.1（Phase 2 PR 1）

追加のみ。1.0 → 1.1 の migration は `provenance.plan_hash` を計算し `execution.resume_from = null` を補う。

| 追加 | 意味 |
|---|---|
| `provenance.plan_hash` | 実行内容（assets / video / audio / delivery / qa）のハッシュ。承認で変わらない。Job が「同じ計画」を実行したかの判定に使う |
| `provenance.ir_hash` | plan_hash + decisions（status 含む）+ plan。承認で変わる |
| `execution.resume_from` | resume 元の job id |
| `provenance.runs[].plan_hash / resumed_from / skipped` | 実行履歴 |

## 冪等キーと resume

- `Operation.id = H(tool, args, inputs)`: compile ごとに安定。provenance と job 記録を照合できる。
- `idempotency_key = H(source_fingerprint, tool, args, tool_version, 上流 op の key)`: 上流が変われば下流の key も変わる（trim を変えると loudness / export も再実行）。
- `source_fingerprint`: sha256（`plan` 既定）または size+mtime（`--no-hash`）。
- Job の `completed_ops[key] = {output, size, mtime}`。resume 時は key が一致し、かつ記録どおりのファイルが残っている場合のみスキップ。
- `render --resume <job_id|last>` は**新しい Job** を作り、`resumed_from` で履歴を残す。再利用した出力は元 Job のディレクトリを参照する。
- 出力を持たない operation（`check`）はキーを持たず常に再実行される。
