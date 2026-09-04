# AI Video Production Ecosystem — Skill / Tool / Capability / Adapter / Registry / Router

```text
video-production-agent = Brain / Orchestrator
ffmpeg-skill           = First Reference Skill（deterministic media processing）— 実装済みの唯一の Skill package
future skills          = 独立した専門 Skill（未実装、ドキュメント上の構想のみ）
```

## 責務

| 概念 | 意味 | 実装 | 例 |
|---|---|---|---|
| **Skill package** | 何ができるか（独立した repository / capability domain） | `skills/contract.py` の `SkillPackage`（skill_id / name / version / description / capabilities / tools / repository / role） | `ffmpeg-skill` |
| **Tool** | その Skill package が提供する具体的操作 | `ToolSpec`（tool_id `<skill_id>/<name>` / skill_id / version / required_capabilities / 実行契約: inputs, produces_output, deterministic, result_keys） | `ffmpeg-skill/probe`, `ffmpeg-skill/cut`, `ffmpeg-skill/loudness` |
| **Production skill** | Agent が実現できること（環境に依存しない知識） | `skills/registry.py` の `SkillSpec`（required_capabilities / risk / approval / tools 候補 / phase） | `silence_cleanup` |
| **Capability** | 実行環境が今何をサポートしているか | `capabilities/resolver.py`（AVAILABLE / MISSING / DEGRADED / UNKNOWN） | `ffmpeg`, `encoder:libx264`, `ffmpeg-skill` |
| **Adapter** | Tool を実際の runtime に接続する | `tools/base.py` の `ToolAdapter`（`package()` / `supports` / `preview` / `run` / `measure`）。現在は `FfmpegSkillAdapter` のみ | `tools/ffmpeg_skill/` |
| **Registry** | Skill package と production skill を登録・列挙し、環境ごとに tool を選ぶ | `SkillRegistry` | — |
| **Router** | 選択された tool id を対応 adapter へ dispatch する | `tools/router.py` の `ToolRouter` | — |
| **Agent** | Production 全体として何をすべきか判断する | `agent/`, `service.py` | — |

Skill と Tool を混同しない: `ffmpeg-skill` は package、`ffmpeg-skill/cut` は tool。Tool id は概念名であり、実装 engine（ffmpeg / sox / …）は adapter の内側。

## 選択と実行の流れ（PR #5 の境界を維持）

```text
SkillSpec.required_capabilities ──CapabilityResolver──▶ 欠落があれば BLOCK 決定
SkillSpec.tools（候補の順序）    ──ToolRouter.supports──▶ SkillRegistry.select_tool が最初に実行可能な tool を選択
                                                          │
                              Service が skill→tool 表を planner / analyzer / QA に渡す（必須引数、既定 engine 無し）
                                                          │
                                   plan.steps[].tool に記録 ─▶ compiler は plan の tool だけを使う ─▶ ToolRouter ─▶ Adapter ─▶ runtime
```

- `SkillRegistry.select_tool(skill, caps, supports)` が唯一の選択関数。`resolve_tools()` はその一括版。
- planner / analyzer / QA は `tools`（skill→tool 表）を**必須引数**として受け取る。既定の engine 表（旧 `DEFAULT_TOOLS`）は存在せず、`tools=None` は `TypeError`、必要な Skill が表に無ければ `ToolError`（analyzer / QA）または tool 無し step + BLOCKED summary（planner）になる。
- validator は step の skill が実装済みで、tool がその Skill の候補に含まれ、adapter が対応し、**登録済み package が宣言している tool** であることを検査する。必要 capability は registry（production skill + package + ToolSpec）から集める。
- `source.tool_versions` は package（tool id の prefix）→ version。冪等キー、provenance の `tool_version` / `skill_package`、artifact の `tool` はすべて operation の tool から引く。
- tool 固有の知識（`recovery.py` の代替引数、catalog の型）は tool 層に閉じる。静的テストが `agent/ execution/compiler.py qa/ media/ project/` に engine 名が出ないことを検査する。

## Contract

### Skill Package Contract（`SkillPackage`）
必須: `skill_id`（tool id の prefix、`/` を含まない）、`name`、`version`（検出時に adapter が埋める）、`description`、`capabilities`（package 全体が必要とする runtime capability）、`tools`（1 件以上）。任意: `repository`、`role`。`validate()` が契約違反を返し、`register_package` は違反を拒否する。

### Tool Contract（`ToolSpec`）
`tool_id`（`<skill_id>/<name>`）、`skill_id`、`version`、`required_capabilities`（tool 固有の追加分）、実行契約 `inputs` / `produces_output`（transform か measure か）/ `deterministic` / `result_keys`（adapter が `ToolResult.data` に保証するキー）。

### Adapter Contract（`ToolAdapter`）
`name` = package の skill_id（対応する tool id の prefix）。`package()` は実装する `SkillPackage`（検出 version 入り）を返す。`supports` / `preview` / `run` / `measure` は呼び出し側が選択した tool を実行するだけで、Skill 選択・IR 参照・production 判断をしない。シェルを経由せず、入力を上書きせず、出力は workspace 内（`PathPolicy`）。

### Registry Contract（`SkillRegistry`）
`register` / `get` / `names` / `all`（production skill）、`register_package` / `packages` / `package` / `tool`（package）、`missing_capabilities` / `select_tool` / `resolve_tools` / `availability` / `package_availability`（環境ごとの解決）、`unknown_tool_candidates`（実装済み skill が未登録 package の tool を引用していないか）。Registry は production decision を行わない。

### Status
`DECLARED`（= `NOT_IMPLEMENTED`: 後 phase 用に宣言のみ）/ `IMPLEMENTED`（`SkillSpec.implemented`、package は adapter が本コードに存在）/ `AVAILABLE`（この環境で tool が選択できた）。新しい status は作らない。

## 将来 Skill package を 1 つ追加するときに必要な変更

1. `tools/<package>/` に `ToolAdapter` 実装（`package()` が `SkillPackage` を返す、catalog 型付き引数、`supports("<package>/<tool>")`、`run` / `preview` / `measure`）。
2. `capabilities/resolver.py` に検出項目を追加（package の所在と version、必要な外部ツール）。
3. `skills/registry.py` の該当 production skill に `tools` 候補を追記（新しい production skill なら `SkillSpec` を 1 件追加）。
4. `Service.adapter()` で adapter を `ToolRouter` に register（1 行。package は adapter から自動登録される）。
5. その Skill が生成する新しい operation 語彙があれば `schemas/project.schema.json` と planner / compiler の該当分岐を追加（既存語彙を別 engine で実現するだけなら不要）。
6. 契約テスト（`--help` / JSON キー）と、Registry → plan.steps → compiler → Router → adapter → provenance の伝播テスト。

planner / compiler / decision / QA の engine 固有ロジックを変更する必要がある構造は禁止（静的テストで検出）。`tests/test_unit.py::EcosystemContractTests` が fake-skill package でこれを証明している（test scope のみ、production registry には登録しない）。

## 今あるもの / 無いもの

- 実装済み・利用可能: `kajisho5/ffmpeg-skill`（0.8.4 ≤ v < 0.9、契約テストで固定）。`video-agent skills` の "Skill packages" に唯一表示される。
- 宣言のみ（NOT_IMPLEMENTED）: production skill `multi_source_sync`（phase 2）、`caption_generation`（phase 3）、`semantic_deletion`（phase 4）。ffmpeg-skill の既存スクリプトを tool 候補として宣言しているだけで、planner は使わない。
- 将来 Skill（構想のみ、コード上に痕跡無し）: media-analysis-skill / audio-production-skill / transcription-skill / subtitle-skill / video-editing-skill / motion-graphics-skill / color-grading-skill / thumbnail-skill / qc-skill。
- 作らないもの: plugin manager / package installer / dynamic import / marketplace / remote registry / 任意コードローダ。
