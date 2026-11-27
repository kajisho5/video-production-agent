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
| `assets{}.microphones[]` | `{id, name, speaker?, role: floor\|presenter\|interp, language?, source: {kind: file, path, audio_stream?, sha256} \| {kind: channel, channel}, pinned?, weight_db?, gain_db?, pan?}`（ADR-134。クリップに付いたマイク: 別ファイルか、クリップの音声の 1 チャンネル。`audio.microphones` を宣言した時だけ。ファイルの directory は `execution.allowed_inputs` に入る） | plan |
| `analysis` | observations[]（測定）, inferences[]（解釈, evidence 必須）, strategy, warnings | MediaAnalyzer / Inferencer |
| `intent` | primary, secondary, confidence, provenance | IntentResolver |
| `constraints[]` / `policy` | 解決済み Rule と conflicts | policy/rules.py |
| `decisions[]` | subject, type (KEEP / REMOVE / TRANSFORM / DELIVER / SKIP / REVIEW / BLOCK), decision, reason, confidence, evidence, alternatives, risk, approval, status, params, basis (settings / approval / intent / requirements / risk with provenance, ADR-027) | DecisionEngine |
| `plan` | version, steps[{skill, tool, decision_ids, params}], summary[] (人間向け) | Planner |
| `timeline` | timelines{id: offset, drift_ratio}, events[] | MediaAnalyzer, approve() |
| `video.operations[]` | `video.trim {asset, keep[[s,e]], accurate, decision_ids}`（無音の decision・テイクの pick・明示の `edit.trim`（ADR-062）が狭める、入力ごとに1つ）、`video.concat {asset: programme, inputs[], output, segments[{input, track, source_range, timeline_range}]（ADR-115: video-editing-skill の whole_frames と同じく出力 frame grid 上に置く）, timeline_duration, transition?, width?, height?, fps?, mode?, pad_color?, temporal_scope, decision_ids}`、（audio）`audio.bgm {asset: bgm, output: bgm_bed, mode: library|generate, library? | prompt?, tags?, mood?, file?, crossfade?, seed?, duration, decision_ids}`（ADR-092、sound-design-skill が作る BGM。duration は mix の長さ、`audio.mix` が最後の入力 `bgm` として受ける）、`qa.documents[] {id, path, name, sha256, preset?, decision_ids}`（ADR-095、artifacts-skill が検査する文書。sha256 は plan 時のもの、QA が実行後の file と照合）、`video.measured_motion {asset, input, output, comp, project, captions?, options?, font_file?, beats?, conform?, fast?, temporal_scope, decision_ids}`（ADR-093、remotion-ffmpeg-skill の pipeline で描く 1 つの composition。編集の後・仕上げの前、尺と画角は変えない）、`video.redact_faces {asset, input, output, regions[{start, end, rects[{x, y, width, height}]}], mode: blur|pixelate, blur_strength?, block_size?, decision_ids}`（ADR-090、detection-skill の face tracks を区間に切ったもの。入力は元の asset そのもので、compiler が区間ごとの cut・矩形ごとの redact・1 回の broll に lower し、trim より前に走る）、`video.rotate {asset, input, output, degrees?, flip?}`（ADR-062、少なくとも一方）、`video.image_insert {asset, input, output, image, inputs[], duration, position: before|after, width, height, fps}`（ADR-070、静止画を絵と同じ frame / rate の無音 clip にして仕上げた絵の前か後ろにつなぐ。compiler が video-editing/image_insert と video-editing/concat の 2 回に lower し、仕上げ（colour / graphics / 字幕）の後・loudness の前に走る）、`video.speed {asset, input, output, factor}`、`video.resize {asset, input, output, width, fps?}`、`video.fit {asset, input, output, aspect, width?, pad_color?, fps?}`、`video.fill {asset, input, output, aspect, width?, fps?}`、`video.overlay {asset, input, output, image, position?, margin?, scale?, opacity?, start?, end?, fade?}`（ADR-029。順序固定、語彙外 key は schema / validator が拒否） | Planner |
| `audio.operations[]` | `audio.loudness {asset, target_lufs, true_peak, decision_ids}`（audio path では `input, output, tolerance_lu?, sample_rate?` を持つ）、`audio.cut {asset, input, output, remove[[s,e]]}`、`audio.concat {asset: programme_audio, inputs[], output, crossfade, segments[], timeline_duration}`、`audio.mix {asset: programme_audio, inputs[], output, levels?[{gain_db, mute}], segments[], timeline_duration}`（ADR-063。concat と排他、長さは先頭入力、入力は切らない）、`audio.voice_cleanup {level: light|medium|strong}`（ADR-068）、`audio.noise_reduction {mode, strength_db}`、`audio.gain {gain_db}`、`audio.mono` / `audio.stereo` / `audio.downmix`、`audio.dynamics {gate?, compressor?, limiter?}`（ADR-063、段ごとの contract の field）、`audio.fade_in` / `audio.fade_out {duration}`（ADR-030。固定順 voice_cleanup → noise_reduction → gain → channels → dynamics → fades → loudness、語彙外 key は schema / validator が拒否） | Planner |
| `audio.automix[]` | クリップごとのマイクの自動ミックス（ADR-134、計画したときだけ）: `{type: audio.automix, asset, sample_rate, microphones[{id, output, channel \| path, audio_stream?}], window_ms, max_gain_rate_db_per_s, hold_below_dbfs?, channels[{gain_db?, weight_db?, pinned?}], preset?, output, mux{video, audio, output}, language_tracks?{microphones[], inputs[], levels[], languages[{language_id, language, title, inputs[]}], format: mkv, output, remove?, cut?, decision_ids}, decision_ids}`。各マイクは engine の audio tool でクリップのレートの mono WAV（`--channel N` / `--mono`、`--sample-rate`）、floor / presenter を audio-production AUTOMIX、その結果を `--replace` で絵の下に入れ、以降の trim / 仕上げはこの mux に掛かる。language_tracks は MIX_TRACKS（入力 0 は長さを決める mute の floor、floor トラック、言語ごとに floor + 通訳）で、trim があれば各入力を同じ範囲で CUT してから。plan.outputs に role `LANGUAGE_TRACKS` の成果物 | Planner |
| `audio.take_crossfade` | テイクの組み立てのつなぎ目の音声クロスフェード（ADR-054、計画したときだけ）: `{type: takes.crossfade, asset: programme, crossfade, spans[{asset, video[s,e], audio[s,e], in_handle, out_handle, programme_start, remove[[s,e]], output}], joins[{after, before, at, crossfade, out_handle, in_handle, crossfaded, why?}], concats[{output, inputs[], crossfade}], audio, mux{video, audio, output}, timeline_duration, decision_ids}`。絵の cut list は変えず、validator が A/V 同期を IR から作り直して確かめる | Planner |
| `captions` / `graphics` / `color` | 予約（Phase 3+） | — |
| `speech.operations[]` | `speech.dub {asset, input, output, cues, lang, source_cues?, engine?, voice?, on_unfit?, reference?, consent?, duck?, voice_gain?, max_overrun?, temporal_scope, decision_ids}`（ADR-085。dubbing-skill の DUB。subject の edits の後・仕上げの前。計画した時だけ節があり、plan hash もその時だけ含める） | Planner |
| `deadline` | `{rule: deadline@1, deadline, budget_seconds, media_seconds, machine, hw_encoder, estimated_seconds, status: fits \| at_risk, dropped[{group, operations, decision_ids, estimated_seconds, reason}], steps[{tool, skill, group, seconds, basis: measured \| default, runs}], defaults, warning?}`（ADR-133。`plan --deadline` の時だけ。この機械の speed book からの見積もりと、締め切りに間に合わず外した仕上げ） | Planner |
| `chunked_render` | `{rule: chunked_render@1, subject, rate: "N/D", chunk_seconds, programme_seconds, targets[], tools{cut, join}, chunks[{index, start, end \| null, start_frame}], reason}`（ADR-135。長い programme の書き出しを区間ごとに: 区間を frame 単位で cut → preset で export → stream copy で join。計画した時だけ節があり、plan hash もその時だけ含める） | Planner |
| `delivery.publish` | `{type: publish.upload, asset, target, artifact, platform, title?, description?, tags?, privacy, thumbnail?, captions?, caption_language?, decision_ids}`（ADR-087。承認された upload。計画した時だけ。compiler は op を作らず、render が offline の plan と dry run、`video-agent publish` が upload） | Planner |
| `delivery.nle[]` | `{type: nle.export|nle.conform, asset, formats[], zip?, output, decision_ids}`（ADR-089。nle-bridge に実行する IR を渡す。計画した時だけ） | Planner |
| `analysis.observations[]` kind `nle.diff` | editor が返した timeline の読み戻し（ADR-098。返ってきた file と sha256、比べた IR の版、diff の summary、source ごとの plan と editor の kept range、marker、warning）。`timeline.editor_cut` / `timeline.editor_markers` の evidence | revise --editor-timeline |
| `delivery.targets[]` | id, preset(ffmpeg-skill export preset 名), platform(check.py platform 名), artifact_type | Profile |
| `qa` | required layers, thresholds | Profile / defaults |
| `execution` | workspace, allowed_inputs, budgets, recovery_policy{max_attempts}, approvals{} | plan / render |
| `provenance` | source_hashes, profile_version, skill_versions, tool_versions, ir_hash, runs[], recovery[] | plan / render |

## 不変条件（validator が検査）

- inference.evidence は observation / event / inference の id を指す。
- operation は既知の asset（concat 後は論理 subject `programme`）と decision を参照し、keep 範囲は 0 < s < e ≤ duration。編集 operation は固定順・concat は 1 つ・fit と fill は排他・speed factor は 0.25–4 かつ ≠1・overlay 画像は allowed_inputs 内の PNG / JPEG（`check_video_operations`）。
- preset / platform は ffmpeg-skill が知っている名前のみ（数値仕様は複製しない）。
- `audio.automix[]`（ADR-134）はその asset が宣言したマイクだけを、宣言どおりの source（チャンネルはクリップの持つ範囲、ファイルは allowed_inputs 内）で参照し、window_ms / max_gain_rate_db_per_s は Skill の値域（既定なし）、少なくとも 1 本は pinned でない。各段（mic_extract / audio_automix / audio_mux / audio_mix_tracks）に plan step がある（`check_automix`）。
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

- `captions.operations[]`: `captions.generate`（asset / output / format srt|vtt / language / cues[{id,start,end,text}] / sources（transcript observation id）/ timeline_map{speed, inputs{asset: {keep, offset}}} / constraints? / temporal_scope / decision_ids）と `captions.burn`（asset / input / sidecar / output）、または `captions.mux`（同じ参照。subtitle-skill render の `mode: "mux"` でソフト字幕トラックにする、ADR-063。burn と同じ subject に両方は書けない）。cue は delivered timeline 上（trim → concat offset → speed で写像済み）。cue の `speaker`（任意、`^SPEAKER_[0-9]{2,}$`）は ADR-096 の `speakers.assign=true` で source の transcript が合成の記録 `speaker_assignment` を持つ時だけ、その匿名 id（validator が照合）。名前は存在しない。
- `graphics.operations[]`: `graphics.render`（asset / input / output / elements[{id,type,start,end,parameters,animation?}] / image?）と `graphics.thumbnail`（asset / input / output / timestamp / format png|jpeg / width? / height? / text? / font_id? / font_size? / color? / position?）。
- `color.operations[]`: `color.strip_dovi` / `color.hdr_to_sdr` / `color.primary_correction`（exposure? / contrast? / saturation? / temperature? / tint? / gamma? / lift? / gain? / levels_in_black? / levels_in_white? / levels_out_black? / levels_out_white? / curves?、どれも独立に省略可。後ろの 8 つは ADR-062）/ `color.lut`（lut / lut_strength?）/ `color.retag`（target）。どの colour operation も任意の `source_alpha_mode`（`straight` | `premultiplied`、ADR-131）を持てる: alpha を持つ source の色の格納方法で、color-grading-skill の request の `project.source.alpha_mode` として送られ、同じ picture の後続の colour step に引き継がれる。指定すると出力はその入力と同じ container（alpha line を持てる mov / mkv、pinned contract の `alpha.keep_alpha.containers`）になり、入力が mp4（edit の後など）なら compile で拒否する。省略時はこれまでどおり mp4。順序固定（strip_dovi → hdr_to_sdr → primary_correction → lut → retag。色調補正はLUT適用より前 — color-grading-skill ADR-15のガイダンス）。
- `qa.qc`: `{enabled, decision_ids, subjects{subject: {kind, targets}}, sidecars{logical: {kind subtitle, reference, subject}}}`。存在するときだけ書かれ、plan_hash に含まれる。
- section が空（`{}`）なら plan_hash は Phase 2 と同一（finishing 要求の無い plan は byte 単位で不変）。
- validator: `check_finishing_operations`（参照・順序・1 subject 1 render / sidecar / burn / thumbnail・LUT / image が allowed_inputs 内・element / cue / timestamp が timeline 内・burn の input は中間物・cue の source は transcript observation）、`check_plan_dependencies`（step の input は asset か depends_on 推移閉包の output）、step 存在と capability（`subtitle` `thumbnail` `color-grading` `motion-graphics` `qc`、UNKNOWN は AVAILABLE ではない）。
- compiler の順序: trim → audio.cut → concat → edits → color → graphics → captions.generate → captions.burn | captions.mux（ADR-063、どちらか一方）→ loudness → export → check → thumbnail → qc（qc は kind qa、出力無し、idempotency key 無し）。
