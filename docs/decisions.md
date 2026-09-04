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

## ADR-014 resume は新 Job、冪等キーは上流連鎖、記録は size+mtime で検証する
- 監査で「trim を変更しても loudness の key が変わらず誤スキップする」設計欠陥を予見していたため、key に上流 op の key を含める。
- 完了記録は出力の size / mtime を持ち、一致しない限り再利用しない。`--no-hash` の場合は source を size+mtime で指紋化する。
- resume は元 Job を書き換えず新 Job を作る（履歴と provenance を壊さない）。plan_hash が異なっても key が自己検証するので、一致する操作だけが再利用される。
- 出力の無い operation（check）は常に再実行する。上流が同一引数で再生成された場合、その下流の記録が無傷なら再利用する（key は計画に対して定義され、バイト列に対してではない）。

## ADR-015 revision は再計画、rejected decision は履歴として次版に残す
- `revise` は記録済みの Observation / Timeline から再計画する（メディアは読まない）。却下された (subject, asset) は planner の提案から除外し、却下 decision 自体は status REJECTED のまま次版に残す（理由・actor・時刻は `execution.reviews`）。
- v(n) は `<stem>.v<n>.json` に保存してから置き換える。
- v≥2 は `approved_plan_version == plan.version` になるまで render できない。承認は decision 単位、plan 版の承認は「pending も rejected-cited も無い」ときに自動的に成立する。
- plan_hash は approve/reject で不変、revise で変わる。schema_version はハッシュに含めない。

## ADR-016 Tool は Skill Registry が選び、plan に記録され、compiler は plan からしか読まない
- 事実: PR #4 までは planner と compiler が `ffmpeg-skill/<script>` を直書きし、`SkillSpec.tools` は参照されていなかった（Tool Selector が不在）。
- 決定: `SkillRegistry.select_tool(skill, caps, supports)` を唯一の選択点にする。planner は解決済み skill→tool 表を `plan.steps[].tool` に書く。compiler は step から tool を取り、無ければ `CompileError`。validator は step の skill が実装済みで tool が候補に含まれ adapter が対応することを検査する。
- `ToolRouter` は tool id を対応 adapter へ振り分けるだけで振る舞いを持たない。adapter の登録は `Service.adapter()` の 1 箇所。
- `phase > IMPLEMENTED_PHASE` の Skill は宣言のみで選択不可（`video-agent skills` で NOT_IMPLEMENTED）。存在しない外部 Skill は registry にも adapter にも置かない。
- 追記（PR #5 最終修正）: planner / analyzer / QA に残っていた `DEFAULT_TOOLS` フォールバックを撤去し、`tools` を必須引数にした。Registry を経ずに ffmpeg-skill が選ばれる経路はコード上に存在しない（静的検査 `test_no_tool_id_literals_outside_tool_layer` が `DEFAULT_TOOLS` と tool id リテラルの両方を tool 定義層以外で禁止する）。tool の version も固定名ではなく operation の tool の adapter から引く。

## ADR-017 Ecosystem Contract: Skill package / Tool / Adapter を型で固定し、loader は作らない
- 事実: 実装済みの外部 Skill は `kajisho5/ffmpeg-skill` のみ。将来 Skill（media-analysis / transcription / … / qc）は存在しない。
- 決定: `SkillPackage`（skill_id / name / version / description / capabilities / tools）と `ToolSpec`（tool_id `<skill_id>/<name>` / 実行契約）を `skills/contract.py` に置き、adapter が `package()` で自分の package を宣言、`SkillRegistry` が登録・列挙・検証する。ffmpeg-skill は Reference Skill として `tools/ffmpeg_skill/package.py` で宣言する。
- 既存の `SkillSpec` は production skill（Agent が実現できること）として維持し、改名しない。DECLARED / IMPLEMENTED / AVAILABLE は既存 status に対応付け、新 status は作らない。
- plugin manager / installer / dynamic import / marketplace / remote registry は作らない。package は adapter module と `Service.adapter()` の 1 行で登録される。future skill は production code に登録も stub も置かず、テストは test scope の fake package で契約を証明する。
- planner / compiler / decision / QA に engine 固有ロジックを置かない（静的テスト）。tool id は概念名で、engine は adapter の内側。

## ADR-018 AI Provider は Brain の一部だが execution authority ではない
- 事実: 実装済み Skill package は `kajisho5/ffmpeg-skill`（外部 OSS、100+ stars）のみ。AI provider の本番実装は無く、既存 `AIProvider` は未使用だった。
- 決定: `AIProvider` / `AIRequest` / `AIResponse` / `AIUsage` / `AIProviderError` を契約として固定し、`agent/ai_reasoning.py` だけが provider を呼ぶ。provider は evidence 要約（observation / event の id と計測値）だけを受け取り、structured result を返す。
- AI 出力は untrusted input。intent は registry の実装済み production skill 名に限定し、evidence は既存 observation / event id に限定、tool / argv / command / risk / approval は捨てる。Observation（OBSERVED）は tool 計測のみ（validator が強制）。AI 由来は `AI_GENERATED`。
- 最終 decision authority は system: 計測済み decision と一致する提案は evidence に加わるだけ、それ以外は CONFIRM の review decision で operation を生成しない。BLOCK は AI で覆せない。confidence（AI / 解析の確信）と risk（影響度、registry / policy）は分離。
- budget `analysis.budget.max_ai_calls`（既定 4）、1 call 1 試行、retry 無し、超過は BUDGET 失敗。provider 失敗は AI failure domain（warning + `ai_calls[].error`）で engine incident とは別、plan は決定論的に継続。revision は AI を呼ばない。
- provenance: `provenance.ai_calls[]` / `ai_provider`（provider / model / task / fingerprint / response hash / usage / latency）。credential は request / response / provenance / doctor のどこにも置かない。IR schema は変更しない。
- 作らない: 特定 provider 依存の core、AI による Tool ID 指定・argv・shell・FFmpeg command 生成、AI による IR 直接生成、AI による SkillRegistry / CapabilityResolver / approval の置換。

## ADR-019 Observation is a deterministic evidence layer independent from AI inference
- 事実: Observation は tool 計測からのみ生成されていたが、何を・どの戦略で・どの予算で観測したかは型として存在せず、strategy と budget は記録だけで強制されていなかった。再分析の再利用も無かった。
- 決定: `Asset → AnalysisRequest → Analyzer → Observation → AnalysisResult → Inference → Decision` を固定する。`AnalysisKind` は実装済みの計測（media_probe / silence / loudness）だけを持ち、未実装の名前は登録しない。`Analyzer` は決定論的で、registry が選んだ tool を `ToolAdapter.measure` で呼ぶ以外の実行手段を持たず、AI provider・decision・IR・command に触れない。
- Observation は常に `provenance = OBSERVED`、`source = "<package>/<tool>@<version>"`、`analysis_id` / `analyzer` / `cache_key` を持ち、`validate_observation` を通ったものだけが保存される。AI は Observation を生成も変更もできない（validator と AI evidence 境界で強制）。
- FULL / TARGETED / CACHED_ONLY は実際に動作し、IR には実行した戦略を記録する。TARGETED の kinds は requirements から system が決める。
- `AnalysisBudget` は強制できる項目（calls / seconds）だけを持ち、未対応の予算名は拒否する。AI call budget とは別物。超過は計測を止め、捏造せず、行ごとに記録する。
- `ObservationCache` の key は asset fingerprint + kind + analyzer id@version + tool id@version + params。Observation id、analysis id、cache key、Job の resume 状態はそれぞれ別の identity。
- 明文化: Observation ≠ Inference、Analysis ≠ AI reasoning、AI evidence ≠ executable instruction、Analysis budget ≠ AI call budget、Analysis cache ≠ Job resume state。

## ADR-020 Temporal Events are domain objects distinct from Observations and Decisions
- 事実: Event は analyzer と review が生成する dict 的な record で、type 体系・asset 参照・identity・Session・検証が無く、AI evidence 境界でも区別されていなかった。
- 決定: 時間軸を第一級 domain model として固定する。`TimePoint` / `TimeRange`（秒、検証付き、relation 付き）、`Event`（canonical code + domain type / subtype、asset、range、source、kind + provenance、evidence、generator）、`Session`（project 内の時間的まとまり）。
- Event は Observation（計測）でも Inference（解釈）でも Decision（制作判断）でもない。OBSERVED event は validated tool measurement からの決定論的変換（`events_from_observation`）でしか生まれず、identity は内容 hash。AI 由来は AI_GENERATED / kind INFERRED としてのみ表現でき、OBSERVED に昇格できない（validator と AI evidence 境界で強制）。
- Event type を定義することと検出を実装することを分離する。生成されるのは `IMPLEMENTED_CODES` の 4 code のみ。speech / speaker / slide / camera / scene / caption / incident は schema のみで、fake event で埋めない。
- Session は明示的に構築する domain object（既定は asset 単位）。自動認識・production plan・Project IR の実行内容とは分離し、`plan_hash` にも含めない。
- Observation = measured fact / Event = temporal domain occurrence / Inference = interpretation / Decision = production choice / Session = temporal grouping / Production Plan = production intent / Project IR = execution contract。

## ADR-021 Production Plan is the deterministic bridge between Decisions / Events and Project IR
- 事実: planner は decision から IR 断片を直接組み立て、「何を制作するか」を表す構造（identity / 順序 / 依存 / evidence / 状態）は存在しなかった。
- 決定: `ProductionPlan` / `ProductionStep` を第一級モデルにし、IR の `plan` セクションとして記録する。planner は決定論的（同じ project / decisions / events / constraints → 同じ plan identity）で、tool は SkillRegistry の解決表からしか取らず、実行もしない。
- Event は事実、Decision は判断、ProductionStep は制作工程、IR の video / audio / delivery は実行契約。AI は Inference までで、plan / tool / argv / command / IR / execution に到達できない（validator の domain parameter 限定と leak 検査で強制）。
- plan の status は reviews / approvals から導出する（別の approval 系を作らない）。APPROVED のみ compiler に進み、BLOCK / REJECTED は誰にも覆せない。partial approval は decision 単位のまま。
- Observation ≠ Event ≠ Inference ≠ Decision ≠ ProductionPlan ≠ Project IR ≠ Execution。

## ADR-022 Artifact is a production result with content identity, distinct from File, QA, Delivery and Archive
- 事実: 生成物は path 名の dict として job に記録されるだけで、identity・integrity・provenance 連鎖・delivery / archive の状態が無かった。
- 決定: Artifact の identity は (project, plan, logical name, sha256) の hash。path / job id / timestamp は identity に含めない。同じ plan と同じ bytes は同じ artifact（resume の再利用は job を追記）、revision や内容変更は別 artifact。QA PASS 後の artifact は不変で、promote 前に bytes を再検証する。
- QA（正しいか）と Delivery（納品可能へ昇格したか）と Archive（履歴として保持）を分離。stage は working → candidate → final → archive、delivery_status はその view。final は integrity ok・QA not FAIL・plan APPROVED（現行版）のときだけ。外部 upload は作らず、`channel` を将来の delivery adapter の境界とする。
- 出力 path は compiler が決める。Artifact 層は manifest（`<workspace>/artifacts/registry`）と archive 索引（`<workspace>/archive`）を書くだけで、media を動かさない。naming template は安全な納品ファイル名（metadata）を生成する。
- plan_hash（plan identity）/ ir_hash（execution contract identity）/ artifact sha256（bytes identity）は別物。

## ADR-023 External observation Skills cross a process boundary; their contract is the source of truth
- 事実: media-analysis-skill 0.1.0 が `contract --json` / `run - --json` / `doctor --json` を公開している。transcription-skill には released contract が無い。
- 決定: 外部 Skill は `ToolAdapter` として process boundary（JSON stdin / stdout）だけで接続し、Python import はしない。tools / kinds / capabilities / version / schema は Skill の contract から取り、agent 側で再宣言しない。互換検査（skill id / contract・schema version / execution mode / tool 所有 / provenance）に通らない installation は使わない（silent fallback 無し）。
- Observation の lifting は provenance を簡略化しない（skill / skill_version / tool / external id / fingerprint / parameters / cache）。Skill が所有する cache を agent は二重化せず、状態を記録するだけ。
- 選択は SkillRegistry のまま（ffmpeg-skill を第一候補に維持、media-analysis を第二候補、media-analysis 固有の計測は専用 production skill）。Event は同じ変換で生成し、Event から command は作らない。
- 計測語彙は消費側で正規化する（`loudness_facts` / `probe_facts`）。Observation の data は tool が返した通りに保持し、inference / QA だけが共通 view を読む。QA も同じ adapter boundary（`measurement_args`）で計測する。
- contract が無い Skill（transcription-skill）は接続しない。stub・推測 contract・fake event は作らない。
