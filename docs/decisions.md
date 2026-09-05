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
- contract が無い Skill は接続しない。stub・推測 contract・fake event は作らない（transcription-skill は 0.2.0 で contract が公開されたため ADR-024 で接続）。

## ADR-024 transcription-skill は「認識まで」を process boundary で接続し、Transcript を Observation、segment を SpeechEvent にする
- 事実: transcription-skill 0.2.0（main）が `skill --json`（id / version / capabilities / tools / engines=EngineSpec / schemas）、`doctor --json`、`run -`（`{"tool","params"}` → `{"ok","tool","result"}`）を公開している。指示書が想定した `contract --json` / `run - --json` は存在せず、実物の transport に合わせた。schema は transcript/0.1・engine-spec/0.1・speech-event/0.1。実装済み engine は faster_whisper（local）のみ。cache hit は保存文書を無変更で返す（asset_id は初回呼び出し側の値）。
- 決定: media-analysis と同じ方式（locate → contract 取得 → 互換検査 → typed request → 1 process → response 検証 → lifting）。Python import・engine 直接実行・model download・ffmpeg 実行は agent 側で一切しない。workspace と allowed_input_roots は adapter が固定し request からは変更不可。engine 選択・model 状態は Skill の contract / selector を尊重し、agent 側に ranking を作らない。`offline` は締める方向にしか作用しない。
- Transcript は認識事実として Observation(kind=transcript, provenance=OBSERVED) に無加工で保存し、provenance（skill / skill_version / tool / transcript id / fingerprint / engine / engine_version / execution_mode / model / model_version / parameters / cache）を保持する。asset identity は fingerprint（Skill の sha256 = agent の asset hash）で照合し、別 asset を作らない。
- SpeechEvent は segment ごとに 1 つ、`speaker_id` は常に null。SpeechEvent ≠ speaker identification。inference / decision / planner / compiler / executor は SpeechEvent を読まない（静的テスト）。Event → command は存在しない。
- transcription 結果は AI inference ではない。AI / LLM / diarization / 字幕 / 編集判断はこの PR に含まれず、SpeechEvent → Inference → Decision は次の段階。

## ADR-025 SpeechEvent は Inference → Decision を経てのみ ProductionPlan に届く（Event → command 無し）
- 事実: PR #13 で SpeechEvent（segment ごと、speaker_id null）が timeline に記録されるが、inference / decision / planner は読んでいなかった。無音は ffmpeg-skill / media-analysis の計測 Event（AUDIO_SILENCE）として同じ timeline にある。閾値は profile に `silence.internal.approval`（generic / conference とも CONFIRM）と `silence.internal.min_seconds`（既定 1.0）があり、発話統合・削除候補の閾値は存在しなかった。
- 決定: `agent/speech_inference.py`（決定的・証拠ベース・AI 無し）が speech_interval / speech_activity / internal_silence_removable / speech_silence_conflict を生成する。閾値は policy キー `speech.merge_gap_seconds`（DEFAULT 0.5）/ `silence.internal.removable_min_seconds`（DEFAULT 2.0）/ 既存 `silence.margin_seconds` を使い、値と provenance を inference data に記録する。
- Decision: 削除候補は `silence.internal.<start>-<end>`（区間付き subject。PlanDiff / revise の suppression が subject@asset キーのため）、approval は policy（CONFIRM）を下限 CONFIRM で適用し AUTO にはしない。conflict と重なる lead / tail trim は CONFIRM に上げる。`speech.continuity` は operation を持たない事実裏付けの decision。
- Plan: 候補は既存 `silence_cleanup` step の removed 区間として keep の補集合になり、step は候補の承認まで PROPOSED（plan REVIEW）。Compiler / Tool 境界は変更なし。
- 実メディア（ja_short ×2 + 3 s 無音）では Whisper の segment が無音側に伸びるため conflict が記録され候補は出ない（安全側）。単語タイムスタンプでの境界精緻化は次 PR 候補。silencedetect end > duration は本 PR で扱わない。

## ADR-026 ProductionContext は Event から決定的に導出する参照中心の中間表現であり、Observation / Event / Decision のいずれでもない
- 事実: Event は単一観測の時間的出来事、Session は人/システムが与えるグルーピングで、「ある時間範囲に複数種別の Event が同時にどう成立しているか」を表す層は無かった。PR #14 の speech inference は SPEECH / AUDIO_SILENCE を直接 timeline から読んでいた。
- 決定: `context/`（model / builder / inference）を追加。ProductionContext = timeline + scope + tracks（type/subtype ごとの event 参照）+ observation / inference / asset 参照、provenance DERIVED、id は内容の hash。境界は Event 自身の start / end のみで、補正・merge・解決はしない。UserDecisionEvent は状況ではない。
- 汎用 inference（source_activity / source_inactivity / transition / conflict）は決定的・ドメイン非依存で、evidence は既存 Event、`data.context_ids` で状況を参照する。conflict は排他ペア表（AUDIO_SILENCE × SPEECH）で記録のみ。AUDIO_SILENCE × AUDIO_ACTIVE は silence tool の keep 区間が margin を含む設計のため排他にしない。loudness のような全体計測は活動を意味しない。
- Decision / Plan は変更しない（汎用 inference は「何が起きているか」まで。「どうするか」は policy / preference / constraint を持つ decision 層）。Context / Inference → step / tool / command の経路は無い（静的テスト）。
- IR: `analysis.contexts`（schema 追加）、validator が参照整合と id 整合を検査、revise は新しい contexts を持ち越す。explain に --context を追加。
- 将来 AI は既存 reasoning boundary 経由で AI_GENERATED inference を生成できるのみ。tool / command / approval / path には届かない。

## ADR-027 Decision は汎用 Decision Engine の不変条件の下でのみ生成し、policy 解決の provenance と basis を Decision 自身に記録する
- 事実: PR #15 までの `decide()` は手続き的で、evidence が空の decision（loudness 計測失敗時の「許容範囲内」）や、approval 値の出所・未知値の扱い・floor が暗黙だった。Decision モデルには type 語彙も basis も無く、`explain --decision` は evidence の一覧止まりだった。既存の precedence（RuleSet、CONSTRAINT は上書き不可）と PR #14 の speech decision の挙動は正しく、壊してはならない。
- 決定: `agent/decision_engine.py`（tool / domain 非依存）を追加し、`decide()` の全 decision をそこから生成する。不変条件: evidence 必須（無ければ生成拒否）、REMOVE / TRANSFORM / DELIVER は計測事実か request の requirement を根拠に持つ（preference / intent / AI 単独は不可、AI 単独は REVIEW のみ）、approval は policy key + 明示 DEFAULT から解決し未知値は CONFIRM、BLOCK* は BLOCK、floor は上げるのみ、BLOCK ⇔ BLOCKED、command / argv / shell / credential 様の内容は拒否（AI 提案 params は削除して記録）。
- 語彙: `type` = KEEP / REMOVE / TRANSFORM / DELIVER / SKIP / REVIEW / BLOCK（既存 decision と 1:1、subject / 文言 / approval / plan への影響は不変）。CONFIRM_REQUIRED 等は approval が既に担うため追加しない。
- basis: settings（key / value / kind / provenance USER・PROFILE・SYSTEM・DEFAULT / rule_id / hard）、approval 解決（key / provenance / notes）、intent（served）、requirements、risk（confidence 非依存）を Decision に記録する。plan_hash には含めない。
- 既存挙動の扱い: USER 明示 requirement による CONFIRM → AUTO の waiver（eval 03）は維持しつつ notes に記録し、CONSTRAINT には適用しない（新規の安全側制限）。loudness 計測が無い場合は decision を出さない（evidence 無し decision の廃止。warning は従来通り）。`silence.internal.approval` の floor CONFIRM は PR #14 のまま。
- validator に `check_decisions` を追加（IR 上で不変条件を再検査。revise で履歴として持ち越された REJECTED decision の evidence は前版のものなので unknown-evidence 検査から除外）。`explain --decision` は basis → evidence chain（inference → context → event → observation → asset / requirement / rule）→ plan step / IR operation まで辿る。
- 対象外: AI / LLM、話者同定、カメラ / スライド選択、新しい decision domain、MCP / plugin / ranking、直接 ffmpeg、Skill 変更、silencedetect end > duration。

## ADR-028 video-editing-skill は CLI contract を境界とする外部編集 Skill として接続し、agent は typed operation だけを渡す
- 事実: kajisho5/video-editing-skill（PR #1、0.1.0、contract video-editing/contract@1）は typed edit request（sources / operations / outputs）を operation graph に検証・コンパイルし、ffmpeg-skill 0.9.x の tool を typed argv で呼ぶ。`video-editing contract --json` / `doctor --json` / `run - --json --workspace D --allowed-input R…`（request は stdin、stdout に 1 文書、error は {ok:false, error:{code, message, retryable, details}}、exit code は error table）。agent 側には既に media-analysis / transcription の外部 Skill adapter パターン（contract が真実、process boundary、PathPolicy、ToolResult 変換）がある。
- 決定: `tools/video_editing/`（locate / adapter / pinned contract_0.1.0.json）を同じパターンで追加する。責務境界は video-production-agent（何をするか・Plan）→ VideoEditingAdapter（typed Operation → EditRequest JSON）→ video-editing CLI（operation graph・ffmpeg-skill 呼び出し・出力検証）→ ffmpeg-skill → FFmpeg。agent は FFmpeg / ffmpeg-skill script を直接呼ばず、command / argv / filter / executable / env / credential / path policy を request に載せない（FORBIDDEN_ARG_KEYS で拒否、contract が宣言する parameter 名以外も拒否）。Skill の内部 module は import しない（静的テスト）。
- contract: 起動時に `contract --json` を取得し check_contract（schema / skill_id / version 0.1.x / role execution / execution flags shell=false・arbitrary_executables=false・raw_ffmpeg_arguments=false・filter_strings=false・network=false・ai=false / canonical_invocation / engine ffmpeg-skill / tools の id = video-editing/<operation> と operation_type・result_keys・produces_output・deterministic / error codes 13 種）。不一致は ContractError（推測・補完しない）。pinned contract との drift は contract_drift（tool_id / version / contract schema / operations / capabilities / required_capabilities / inputs / produces_output / deterministic / result_keys / execution / errors / response_shape）で検出し、CapabilityResolver は drift があれば video-editing を MISSING にする。integration test が実 Skill との drift 0 を検証する。
- 実行: argv list のみ（`run - --json --workspace <op dir> --allowed-input <root>… --ffmpeg-skill-dir <agent が知る ffmpeg-skill>`）、request は stdin、timeout は agent の timeout を options.timeout_seconds として Skill に渡しつつ process boundary（run_process_group、process group kill、exit 124 → CANCELLED/timeout）で強制。`--workspace` は compiler が決めた出力ファイルのディレクトリ（agent workspace 内、executor が作成）、`--allowed-input` は agent PathPolicy の roots + workspace。stdout は厳密に 1 JSON 文書（空 / 非 JSON / 複数文書 / exit≠0 + ok / exit 0 + 出力欠落 / sha256 不一致 / observation 欠落は全て INVALID_RESULT・非 retryable）。
- mapping: execution.outputs[out1]（path / sha256 / size / timeline / observation）→ ToolResult.output・data.artifact・data.timeline・data.observation（OBSERVED、source ffmpeg-skill/probe@ver）、operations[op1] の provenance record → data.operation、commands → ToolResult.commands / provenance（記録のみ。再実行経路は存在しない）。provenance.json の operations[].skill_result に Skill が報告した事実を保存。
- error: Skill の code と retryable を ToolResult.data.error に保持し、adapter が recovery class に写像（INVALID_REQUEST / UNSUPPORTED_* / INVALID_TIME_RANGE / DEPENDENCY_ERROR → INVALID_ARGS(BLOCK)、INVALID_INPUT / PATH_NOT_ALLOWED / MISSING_INPUT → INPUT_MISSING(BLOCK)、TOOL_ERROR retryable → UNKNOWN(1 回再試行)、CANCELLED → TIMEOUT(再試行・timeout 延長)、OUTPUT_ERROR / VALIDATION_ERROR / INTERNAL_ERROR / 非 retryable → SKILL_ERROR(BLOCK、新設)）。recovery.classify_error は Skill が retryable=false と言うものを決して再試行しない。
- lowering: IR video.trim（keep ranges + accurate）→ plan が選んだ tool が video-editing/cut のとき contract の CUT parameters {keep:[{start,end}], precision: frame|keyframe}。ffmpeg-skill/cut への lowering は不変。registry の silence_cleanup 候補は ["ffmpeg-skill/cut", "video-editing/cut"]（宣言順、ranking 無し、fallback 無し。両方利用可能なら従来通り ffmpeg-skill/cut）。video-editing が unsupported とする操作（CROP / FREEZE / REVERSE / IMAGE_INSERT / POSITION）は tool として存在せず、agent が FFmpeg で代替することはない。
- capability: `video-editing` capability は Skill の doctor ok かつ drift 無しのときのみ AVAILABLE（それ以外 MISSING）。registry.select_tool は package capabilities と、resolver が解決する ToolSpec.required_capabilities（encoder:aac / filter:xfade / filter:acrossfade を resolver に追加）を検査し、欠けていれば選択しない。resolver が知らない capability 名は Skill の doctor に委ねる（推測しない）。
- 対象外: 新しい編集機能の追加、video-editing-skill / ffmpeg-skill の変更、Decision Engine への tool 詳細の持ち込み、並列実行、MCP / plugin / ranking。audio-only 入力は video-editing が INVALID_INPUT とするため silence_cleanup の既定候補順は ffmpeg-skill/cut のまま。

