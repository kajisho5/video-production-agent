# Architecture Decision Records

## ADR-001 ffmpeg-skill はプロセス境界で使う（ライブラリ import しない）
- 事実: `_common.STATE` がプロセスグローバル、`die()` が `sys.exit`、各スクリプトが兄弟を subprocess 起動する設計。
- 決定: `FfmpegSkillAdapter` が `python3 <skill>/scripts/<name>.py ... --json` を起動し、stdout JSON と終了コードを契約とする。
- 帰結: 並列実行・タイムアウト・キャンセルは Agent 側で制御できる。バージョンは `package.json` で固定（0.8.4 ≤ v < 0.9）。

## ADR-002 Phase 1 では `render.py` の project.json を使わない
- 事実: 固定ステージ順、スキーマ検証なし、`check` FAIL でも exit 0、dry-run でダミー probe。
- 決定: Agent 自身の Project IR + compiler を持ち、個別スクリプトを順に呼ぶ。
- 将来: IR → render project への lowering adapter は追加可能。

## ADR-003 Agent 側の ffmpeg 直接呼び出しは機能検出のみ
- `-version / -encoders / -decoders / -filters` と `fc-list`。メディア処理は一切しない。

## ADR-004 出力パスは常に Agent の workspace、入力は allowed roots
- ffmpeg-skill は `-y` 固定で入出力同一パスの保護がないため、`PathPolicy` で強制。source は読み取りのみ。

## ADR-005 Phase 1 は AI プロバイダ無しで完全動作
- 自然言語は「曖昧さのないキーワード」のみ USER requirement に変換。それ以外は `--set key=value` で明示。
- `providers/` は NullProvider のみ。将来の抽出は provenance=AI_GENERATED として別扱い。

## ADR-006 Observation / Inference / Decision を別型にし、Inference は evidence 必須
- validator が evidence 参照の存在を検査。観測値は上書きされない。

## ADR-007 risk と approval を confidence から独立させる
- 先頭/末尾の技術的無音: LOW/AUTO（conference は CONFIRM）。中間無音: Phase 1 は提案しない（keep）。意味的削除: HIGH/CONFIRM、conference は BLOCK_UNLESS_EXPLICIT（constraint）。

## ADR-008 Recovery は有限（既定 2 試行）、QA FAIL は Recovery ではなく Revision
- 分類表 `execution/recovery.py`。QA FAIL の artifact は `working` に留まり `candidate` に昇格しない。

## ADR-009 conference の master ターゲットは Phase 1 では platform=custom
- 理由: broadcast 検査は -23 LUFS を要求し、web 向け -16 LUFS とターゲット別ラウドネスが必要（Phase 2 で per-target audio op を導入）。QA が実際にこの不整合を検出したため修正。

## ADR-010 PathPolicy が保証するものと保証しないもの（Phase 1 監査より）
- 保証する: 出力は常に `<workspace>/jobs/<id>/` 配下、出力パス = 入力パスは拒否、シェルを経由しない argv、catalog に無いフラグは拒否。
- 保証しない: 「読める入力」の制限。allowed root は入力パスの親ディレクトリから導出されるため、ユーザが指定したパス（シンボリックリンク先を含む）は常に読める。独立した `--allowed-input` 設定は Phase 2。

## ADR-011 子プロセスはプロセスグループで起動し、タイムアウト・割り込み時はグループごと kill する
- 事実: ffmpeg-skill のスクリプトは ffmpeg を孫プロセスとして起動する。`subprocess.run(timeout=)` はスクリプトだけを kill し、ffmpeg は出力ファイルを書き続けた（監査で再現）。
- 決定: POSIX は `start_new_session` + `killpg(SIGKILL)`、Windows は `CREATE_NEW_PROCESS_GROUP` + `taskkill /T`。失敗した試行の出力ファイルは削除してから再試行する。SIGINT は Executor が CANCELLED に変換し、Service は job / IR / provenance を必ず保存する。

## ADR-012 analysis.strategy は「実際に行った解析」を記録する
- Phase 1 は全ファイルの silence / loudness 解析しか無いので `FULL_ANALYSIS` を記録し、`budget.enforced=false` を明示する。プロファイルの要求値は `budget.requested_strategy` に残す。

## ADR-013 REJECTED 決定は実行不可、operation id は決定論的
- validator は REJECTED 決定を参照する operation / delivery target を error にし、render は実行しない（監査時点では REJECTED でもそのまま compile されていた）。
- operation id は tool + args + inputs のハッシュで生成する。provenance と Job の `completed_ops` を compile をまたいで照合できるようにするため（resume の前提）。
