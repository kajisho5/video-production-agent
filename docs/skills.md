# Skill / Capability / Tool

| 概念 | 意味 | 実装 | 例 |
|---|---|---|---|
| **Skill** | 何を実現できるか（環境に依存しない知識） | `skills/registry.py` の `SkillSpec`（inputs / outputs / required_capabilities / risk / approval / tools 候補 / phase） | `silence_cleanup` |
| **Capability** | この環境で今何が使えるか | `capabilities/resolver.py`（AVAILABLE / MISSING / DEGRADED / UNKNOWN） | `ffmpeg`, `encoder:libx264`, `ffmpeg-skill`, `font:cjk-ja` |
| **Tool** | 実際に処理する主体 | `tools/` の `ToolAdapter`（現在は `FfmpegSkillAdapter` のみ）と `ToolRouter` | `ffmpeg-skill/cut` |

選択の流れは一方向で、逆流しない:

```
SkillSpec.required_capabilities ──CapabilityResolver──▶ 欠落があれば BLOCK 決定
SkillSpec.tools（候補の順序）    ──ToolRouter.supports──▶ 最初に実行可能な tool を選択
                                                          │
                                   plan.steps[].tool に記録 ─▶ compiler は plan の tool だけを使う
```

- `SkillRegistry.select_tool(skill, caps, supports)` が唯一の選択関数。`resolve_tools()` はその一括版。
- `phase > IMPLEMENTED_PHASE` の Skill は roadmap 宣言であり、`select_tool` は選ばず、`video-agent skills` は `NOT_IMPLEMENTED` と表示する。存在しない外部 Skill（media-analysis-skill 等）は registry にも adapter にも**存在しない**。
- planner は解決済みの skill→tool 表を受け取って `plan.steps[].tool` に書く。compiler は `plan.steps` からしか tool を取らない（`CompileError`）。validator は step の skill が実装済みで、tool がその Skill の候補に含まれ、adapter が対応することを検査する。
- analyzer / QA も同じ表から計測 tool を取る（`media_probe`, `silence_analysis`, `loudness_analysis`, `delivery_check`, `visual_inspection`）。
- tool 固有の知識（`recovery.py` の代替引数 `accurate`、catalog の型）は tool 層に閉じる。

## 将来 Skill（外部パッケージ）を追加するときに必要な変更

1. `tools/<package>/` に `ToolAdapter` 実装（catalog 型付き引数、`supports("<package>/<tool>")`、`run` / `preview` / `measure`）。ffmpeg-skill と同じくシェルを経由しない契約にする。
2. `capabilities/resolver.py` に検出項目を追加（例: `<package>` の所在とバージョン、必要な外部ツール）。
3. `skills/registry.py` の該当 Skill に `required_capabilities` と `tools` 候補を追記（新 Skill なら `SkillSpec` を 1 件追加、`phase` を現行に）。
4. `Service.adapter()` で adapter を `ToolRouter` に register（1 行）。
5. その Skill が生成する operation 語彙があれば `schemas/project.schema.json`（video/audio op）と planner / compiler の該当分岐を追加。
6. 契約テスト（`--help` / JSON キー）と、adapter を差し替えたときに validator が拒否することのテスト。

Agent 本体（Request → … → Provenance の流れ、IR、Job、resume、revision）には変更が要らない。

## 今あるもの / 無いもの

- 実在して利用可能: `kajisho5/ffmpeg-skill`（0.8.4 ≤ v < 0.9、契約テストで固定）。
- 宣言のみ（NOT_IMPLEMENTED）: `multi_source_sync`（phase 2）、`caption_generation`（phase 3）、`semantic_deletion`（phase 4）。いずれも ffmpeg-skill の既存スクリプトを tool 候補として宣言しているだけで、planner は使わない。
- 存在しない外部 Skill: media-analysis / transcription / subtitle / video-editing / audio-production / motion-graphics / color-grading / thumbnail / qc。コード上に痕跡は無い。
