# Project IR (schema 1.2)

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
| `decisions[]` | subject, type (KEEP / REMOVE / TRANSFORM / DELIVER / SKIP / REVIEW / BLOCK), decision, reason, confidence, evidence, alternatives, risk, approval, status, params, basis (settings / approval / intent / requirements / risk with provenance, ADR-027) | DecisionEngine |
| `plan` | version, steps[{skill, tool, decision_ids, params}], summary[] (人間向け) | Planner |
| `timeline` | timelines{id: offset, drift_ratio}, events[] | MediaAnalyzer, approve() |
| `video.operations[]` | `video.trim {asset, keep[[s,e]], accurate, decision_ids}`、`video.concat {asset: programme, inputs[], output, segments[{input, track, source_range, timeline_range}], timeline_duration, transition?, width?, height?, fps?, mode?, pad_color?, temporal_scope, decision_ids}`、`video.speed {asset, input, output, factor}`、`video.resize {asset, input, output, width, fps?}`、`video.fit {asset, input, output, aspect, width?, pad_color?, fps?}`、`video.fill {asset, input, output, aspect, width?, fps?}`、`video.overlay {asset, input, output, image, position?, margin?, scale?, opacity?, start?, end?, fade?}`（ADR-029。順序固定、語彙外 key は schema / validator が拒否） | Planner |
| `audio.operations[]` | `audio.loudness {asset, target_lufs, true_peak, decision_ids}`（audio path では `input, output, tolerance_lu?, sample_rate?` を持つ）、`audio.cut {asset, input, output, remove[[s,e]]}`、`audio.concat {asset: programme_audio, inputs[], output, crossfade, segments[], timeline_duration}`、`audio.gain {gain_db}`、`audio.mono` / `audio.stereo` / `audio.downmix`、`audio.fade_in` / `audio.fade_out {duration}`（ADR-030。固定順、語彙外 key は schema / validator が拒否） | Planner |
| `captions` / `graphics` / `color` | 予約（Phase 3+） | — |
| `delivery.targets[]` | id, preset(ffmpeg-skill export preset 名), platform(check.py platform 名), artifact_type | Profile |
| `qa` | required layers, thresholds | Profile / defaults |
| `execution` | workspace, allowed_inputs, budgets, recovery_policy{max_attempts}, approvals{} | plan / render |
| `provenance` | source_hashes, profile_version, skill_versions, tool_versions, ir_hash, runs[], recovery[] | plan / render |

## 不変条件（validator が検査）

- inference.evidence は observation / event / inference の id を指す。
- operation は既知の asset（concat 後は論理 subject `programme`）と decision を参照し、keep 範囲は 0 < s < e ≤ duration。編集 operation は固定順・concat は 1 つ・fit と fill は排他・speed factor は 0.25–4 かつ ≠1・overlay 画像は allowed_inputs 内の PNG / JPEG（`check_video_operations`）。
- preset / platform は ffmpeg-skill が知っている名前のみ（数値仕様は複製しない）。
- 必要 capability が MISSING の IR は validate で error。
- approval=BLOCK の decision があれば render は BLOCKED、CONFIRM が PROPOSED のままなら WAITING_FOR_APPROVAL。

## コンパイル

`execution/compiler.py` が asset ごとに trim → (speed → resize → fit | fill → overlay) → loudness → export → check の順で Operation を生成する。audio path（ADR-030）の asset は cut → (concat → programme_audio) → gain → channels → fade_in → fade_out → normalize の順で、すべて `audio-production/run` に lowering される（出力は WAV）。concat があれば全 asset の trim の後に concat（出力 `programme`）を置き、以降の操作は programme に対して生成する（ADR-029）。
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

## schema 1.2（Phase 2 PR 2）

追加のみ。1.1 → 1.2 の migration は `revision` セクションを補い、`execution.approvals` を `execution.reviews`（APPROVED 記録）へ写す。

| 追加 | 意味 |
|---|---|
| `execution.reviews{decision_id: {action, by, at, reason, plan_version}}` | 承認 / 却下の記録 |
| `revision.feedback[]` | `{id, plan_version, target, text, structured, by, at}` |
| `revision.history[]` | `{version, from_version, created_at, by, feedback_ids, rejected_decision_ids, rejection_reasons, dropped_proposals, diff, plan_hash, ir_hash_before, snapshot}` |
| `revision.approved_plan_version` | 承認済みの plan 版。`plan.version` と一致しない v≥2 は render 不可 |

`plan_hash` から `schema_version` を除外した（同じ内容は migration 前後で同じハッシュ）。詳細は `docs/revision.md`。

## schema 1.2 追記: finishing section と QC gate（Phase 3、ADR-031 / ADR-032）

- `captions.operations[]`: `captions.generate`（asset / output / format srt|vtt / language / cues[{id,start,end,text}] / sources（transcript observation id）/ timeline_map{speed, inputs{asset: {keep, offset}}} / constraints? / temporal_scope / decision_ids）と `captions.burn`（asset / input / sidecar / output）。cue は delivered timeline 上（trim → concat offset → speed で写像済み）。speaker は存在しない。
- `graphics.operations[]`: `graphics.render`（asset / input / output / elements[{id,type,start,end,parameters,animation?}] / image?）と `graphics.thumbnail`（asset / input / output / timestamp / format png|jpeg / width? / height? / text? / font_id? / font_size? / color? / position?）。
- `color.operations[]`: `color.strip_dovi` / `color.hdr_to_sdr` / `color.primary_correction`（exposure? / contrast? / saturation? / temperature? / tint?、5つとも独立に省略可）/ `color.lut`（lut / lut_strength?）/ `color.retag`（target）。順序固定（strip_dovi → hdr_to_sdr → primary_correction → lut → retag。色調補正はLUT適用より前 — color-grading-skill ADR-15のガイダンス）。
- `qa.qc`: `{enabled, decision_ids, subjects{subject: {kind, targets}}, sidecars{logical: {kind subtitle, reference, subject}}}`。存在するときだけ書かれ、plan_hash に含まれる。
- section が空（`{}`）なら plan_hash は Phase 2 と同一（finishing 要求の無い plan は byte 単位で不変）。
- validator: `check_finishing_operations`（参照・順序・1 subject 1 render / sidecar / burn / thumbnail・LUT / image が allowed_inputs 内・element / cue / timestamp が timeline 内・burn の input は中間物・cue の source は transcript observation）、`check_plan_dependencies`（step の input は asset か depends_on 推移閉包の output）、step 存在と capability（`subtitle` `thumbnail` `color-grading` `motion-graphics` `qc`、UNKNOWN は AVAILABLE ではない）。
- compiler の順序: trim → audio.cut → concat → edits → color → graphics → captions.generate → captions.burn → loudness → export → check → thumbnail → qc（qc は kind qa、出力無し、idempotency key 無し）。
