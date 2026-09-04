# Phase 2 PR 1 監査（resume / 冪等スキップ）— 実験ベース

対象: PR #3（schema 1.1, plan_hash, deterministic operation id, job resume, completed_ops）。
方法: 実 ffmpeg 6.1.1 + ffmpeg-skill 0.8.4、合成素材 `talk.mp4`（16 s、先頭 3 s / 末尾 2.3 s 無音）、youtube プロファイルで CLI を実行。「テストが通る」ではなく、各シナリオの実行結果を記録する。

## 実験結果

| # | シナリオ | 手順 | 結果 | 判定 |
|---|---|---|---|---|
| E1 | resume 途中失敗 | `render --timeout 2.2` で export がタイムアウト（cut / loudness は完了） → `render --resume last` | 1回目 FAILED、`completed_ops`=2。2回目 COMPLETED、**cut / loudness を再利用（2/2）**、export のみ実行、QA PASS 16/16 | ✅ |
| E2 | 同一 IR の再実行 | E1 の後にもう一度 `--resume last` | 3 操作スキップ、実行は `check` のみ、QA PASS | ✅ |
| E3 | operation id の安定性 | 同じ IR を 3 回 `--dry-run` | 3 回とも同一の op id 列（`sort -u` で 1 行） | ✅ |
| E4 | IR 変更（trim 開始 +0.5 s） | IR を編集して `--resume last` | `plan_changed=True`、スキップ 0、cut / loudness / export / check すべて再実行、QA PASS | ✅ 上流変更が下流に伝播 |
| E5 | IR 変更（loudness 目標 -14 → -16） | 同上 | cut のみ再利用（1）、loudness / export / check 再実行。QA は **FAIL**（youtube platform は -14±2 LUFS を要求） | ✅ resume は正しく、QA が IR の不整合を検出 |
| E6 | 出力の削除・改ざん | cut 出力を削除、loudness 出力を 1000 byte に切り詰め → `--resume last` | cut / loudness 再実行、export は記録が無傷なので再利用（1）、QA PASS | ✅ 設計どおり（ADR-014） |
| E7 | `--no-hash` で source を差し替え | 同名で 10 s 版に置換 → 再 plan → 旧 job を `--resume` | スキップ 0（size+mtime の指紋が変わり key 不一致） | ✅ |
| E8 | 別 IR を実行した job を `--resume` | `b.json` に `c2.json` の job を指定 | `plan_changed=True`、スキップ 0、正常完了 | ✅ 誤スキップなし |
| E9 | SIGINT → resume | cut 実行中に SIGINT → CANCELLED（completed 0）→ `--resume last` | 全操作実行、COMPLETED、QA PASS | ✅（cut 完了前の中断なので再利用対象なし。中断後に完了済みがある場合は E1 と同経路） |
| E10 | schema 1.0 の IR を読込 | `plan_hash` と `resume_from` を削除し `1.0` にしたファイルを `validate` / `render --dry-run` | 読込時に 1.1 へ migration、validate ok、dry-run 4 操作。ファイル自体は書き換えない | ✅ |

## 誤スキップの可能性について検討した点

- key は `H(source fingerprint, tool, args, tool_version, 上流 key)`。E4 / E5 / E7 / E8 で「上流変更」「引数変更」「source 差し替え」「別 IR」のいずれも再利用されないことを確認した。
- 記録された出力は size+mtime が一致しない限り使わない（E6）。
- 上流が同一引数で再生成された場合、その下流（E6 の export）は再利用する。ffmpeg の出力はバイト単位では非決定的だが、key は計画に対して定義しており、下流の記録は「同じ計画から作られた無傷の成果物」を意味する。これを望まない運用では resume を使わずに render すればよい。
- 出力の無い `check` は key を持たず常に再実行される（E2）。
- `tool_version`（ffmpeg-skill の version）が key に入るため、ffmpeg-skill 更新後は再利用されない。ffmpeg 本体の版は含めていない（B: Phase 2 後続で `tool_versions.ffmpeg` を追加候補）。

## テスト

- unit 33/33（resume 系 8 件: 途中失敗 / 同一 IR / trim 変更 / loudness 変更 / 削除・改ざん / `--no-hash` 差し替え / 不明 job / legacy 記録）。
- integration 7/7（実メディアで resume: 納品物削除後に cut / loudness を再利用し export のみ再実行、同一 IR 再実行で `check` のみ）。
- evals 6/6。
- FakeAdapter を「出力ファイルにメタデータを持たせる」形に変更し、インスタンスをまたいでも probe / loudness 計測が一貫するようにした（resume テストで新しい adapter インスタンスが使われるため）。

## 残る制限（この PR では扱わない）

- resume は CANCELLED / FAILED / COMPLETED のどの job からでも可能で、状態による制限は無い（意図的。key と記録が自己検証するため）。
- `resumed_from` は 1 段のみ記録（多段の連鎖は各 job の `resumed_from` を辿る）。
- Windows での実機検証は未実施（CI は unit のみ）。

## CI（run 33871136715, head d3c6337）

| job | 結果 |
|---|---|
| unit ubuntu 3.9 / 3.11 | success（33 tests, evals 6/6） |
| unit windows 3.9 / 3.11 | success（33 tests, evals 6/6） |
| integration ubuntu | success（7 tests、resume の実メディアテストを含む） |
