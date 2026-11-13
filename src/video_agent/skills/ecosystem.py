"""Who in the Velisa ecosystem owns a job, and how a user installs it (ADR-051 handoffs, ADR-057).

A pinned, hand-checked snapshot of Velisa's `ecosystem.json` (schema_version 2, read 2026-11-13) for every package a SkillSpec
cites, plus each package's install command taken from its own README. It is data only: nothing here is executed, and an entry
never makes a skill selectable (that needs an adapter). `status` means what ecosystem.json says: `active` ships working
operations behind its contract (released or not), `scaffold` has no working operation yet, `planned` has no repository yet.
`tests/test_ai_host.py::EcosystemSnapshotTests` compares every field the snapshot copies, and the marketplace name, with a Velisa
checkout (`VIDEO_AGENT_VELISA_DIR`, or a sibling `Velisa` / `AI-video-production-OS` directory); CI clones Velisa main for it.

The agent names a package by its short id (`video-editing`, `oncemore`), the ecosystem by its repository (`video-editing-skill`);
`repository()` maps one to the other instead of guessing a `-skill` suffix."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ECOSYSTEM_SOURCE = "Velisa ecosystem.json (schema_version 2), read 2026-11-13"
MARKETPLACE = "velisa"   # Velisa .claude-plugin/marketplace.json `name` (ADR-057 addendum 1; it was ai-video-production-os)
GITHUB = "https://github.com/kajisho5/"


@dataclass(frozen=True)
class Component:
    package: str                    # the agent's package id (the prefix of its tool ids)
    repo: str                       # ecosystem.json `repo`
    department: str                 # ecosystem.json `department`
    install: str                    # ecosystem.json `install`: pip | npm | checkout
    status: str                     # ecosystem.json `status`: active | scaffold | planned
    plugin: bool = False            # ecosystem.json `plugin`: listed in the Claude Code marketplace
    optional: bool = False          # ecosystem.json `optional`: ./install.sh installs it only with --with-optional
    owns: Tuple[str, ...] = ()      # ecosystem.json `owns`
    package_install: Optional[str] = None   # the package's own install command (its README); None: nothing released yet
    note: str = ""

    @property
    def released(self) -> bool:
        return self.package_install is not None

    def install_commands(self) -> List[str]:
        """How to install it, most direct first. A component with nothing released says so instead of naming a command that
        cannot work (a scaffold repository has no package; ./install.sh would clone it and fail to install it)."""
        if not self.released:
            return [f"none yet: {self.repo} is a {self.status} repository with no released package ({GITHUB}{self.repo})"]
        out = [f"/plugin install {self.repo}@{MARKETPLACE}"] if self.plugin else []
        out.append(self.package_install or "")
        out.append("or ./install.sh" + (" --with-optional" if self.optional else "") + " from Velisa (packages and CLIs)")
        return out

    def to_dict(self) -> Dict[str, object]:
        return {"package": self.package, "repository": self.repo, "department": self.department, "status": self.status, "plugin": self.plugin,
                "optional": self.optional, "owns": list(self.owns), "released": self.released, "install": self.install_commands()}


def _clone_pip(repo: str, extras: str = "") -> str:
    return f"git clone {GITHUB}{repo} && pip install -e '{repo}{('[' + extras + ']') if extras else ''}'"


_COMPONENTS: Tuple[Component, ...] = (
    # ---- packages the agent already has adapters for (ADR-051's original handoff targets)
    Component("ffmpeg-skill", "ffmpeg-skill", "engine", "checkout", "active", plugin=True,
              owns=("FFmpeg実行", "書き出し前の規格点検", "マルチカム同期", "Bロール合成", "手ぶれ・傾き補正", "プレビュー・縮小一覧", "納品レポート", "配信プラットフォーム仕様", "カットリスト書き出し"),
              package_install="npx ffmpeg-skill"),
    Component("video-editing", "video-editing-skill", "visual", "pip", "active", plugin=True, owns=("カット・連結", "尺・画角合わせ", "トランジション", "静止画オーバーレイ", "言い直し・重複の候補計算"),
              package_install=_clone_pip("video-editing-skill")),
    Component("color-grading", "color-grading-skill", "visual", "pip", "active", plugin=True, owns=("色補正", "LUT", "HDR→SDR"), package_install=_clone_pip("color-grading-skill")),
    Component("motion-graphics", "motion-graphics-skill", "visual", "pip", "active", plugin=True, owns=("タイトル・テロップ", "ロゴ・PiP・クロマキー合成", "動くグラフィックの配置"), package_install=_clone_pip("motion-graphics-skill")),
    Component("subtitle", "subtitle-skill", "visual", "pip", "active", plugin=True, owns=("字幕生成・検証", "字幕焼き込み", "字幕翻訳", "ロケール別字幕整形", "動く字幕(字幕キュー由来)"),
              package_install=_clone_pip("subtitle-skill")),
    Component("thumbnail", "thumbnail-skill", "visual", "pip", "active", plugin=True, owns=("サムネイル",), package_install=_clone_pip("thumbnail-skill")),
    Component("audio-production", "audio-production-skill", "audio", "pip", "active", plugin=True,
              owns=("音量・ラウドネス", "ミックス", "ダイナミクス", "従来型ノイズ低減", "無音・フィラー除去"), package_install=_clone_pip("audio-production-skill")),
    Component("transcription", "transcription-skill", "recognize", "pip", "active", plugin=True, owns=("文字起こし", "文字起こしの語句検索"),
              package_install=_clone_pip("transcription-skill", "faster-whisper")),
    Component("qc", "qc-skill", "verify", "pip", "active", plugin=True, owns=("規格判定(QC)",), package_install=_clone_pip("qc-skill")),
    Component("media-analysis", "media-analysis-skill", "verify", "pip", "active", plugin=True, owns=("メディア計測", "発話区間"), package_install=_clone_pip("media-analysis-skill")),
    # ---- ADR-057: every other job of the ecosystem, declared so a handoff names its owner (no adapter here yet; roadmap step 9)
    Component("ingest", "ingest-skill", "ingest", "pip", "active", plugin=True, owns=("素材取り込み(検証付きコピー)", "二重バックアップ", "素材台帳", "URL素材取得"),
              package_install=_clone_pip("ingest-skill", "fetch"),
              note="offload / verify / inventory ship (README); `ingest fetch` (URL download, ingest-skill #9) "
                   "needs the fetch extra (yt-dlp), and the agent hands URL inputs to it (ADR-059)"),
    Component("voice-enhance", "voice-enhance-skill", "audio", "pip", "active", owns=("声の分離(AI)", "残響除去", "マイク被り低減"),
              package_install=_clone_pip("voice-enhance-skill"),
              note="isolate / dereverb / bleed-reduce ship (README, all EXPERIMENTAL); the core is stdlib-only and "
                   "each engine is its own install (deep-filter + the DeepFilterNet3 model, `[wpe]`, `[bleed]`) that `voice-enhance doctor` names (ADR-073)"),
    Component("dubbing", "dubbing-skill", "audio", "pip", "active", owns=("AIナレーション(TTS)", "吹替・声の置換", "音声解説"),
              package_install=_clone_pip("dubbing-skill"),
              note="narrate / dub / describe ship (README, all EXPERIMENTAL); the core is stdlib-only and each "
                   "engine is its own install (Chatterbox in its own venv, the VOICEVOX / Style-Bert-VITS2 apps, Kokoro) that `dubbing doctor` names; the agent runs DUB (ADR-085)"),
    Component("sound-design", "sound-design-skill", "audio", "pip", "active", owns=("BGM生成", "効果音"), package_install=_clone_pip("sound-design-skill"),
              note="library-pick / generate-bgm ship (README, EXPERIMENTAL); the music bed of the mix "
                   "(ADR-092): library mode needs no engine, generate-bgm needs ACE-Step 1.5 installed by the user (`sound-design doctor` prints how)"),
    Component("diarization", "diarization-skill", "recognize", "pip", "active", plugin=True, optional=True, owns=("話者識別", "話者IDの付与規則"),
              package_install=_clone_pip("diarization-skill", "pyannote"),
              note="speaker turns as the analysis kind speaker_turns (ADR-086); diarize also needs the gated speaker-diarization-community-1 pipeline cloned once "
                   "and DIARIZATION_PIPELINE_DIR set (`diarization doctor` prints the steps; the Skill never downloads)"),
    Component("detection", "detection-skill", "recognize", "pip", "active", plugin=True, optional=True, owns=("顔検出", "文字検出(OCR)", "物体検出", "ショット検出"),
              package_install=_clone_pip("detection-skill", "all"),
              note="face tracks to blur as the analysis kind face_tracks (ADR-090); faces also need the YuNet model file face_detection_yunet_2026may.onnx "
                   "in DETECTION_MODELS_DIR (`detection doctor` prints where to get it; the Skill never downloads)"),
    Component("slide-sync", "slide-sync-skill", "recognize", "pip", "active", plugin=True, owns=("スライド取込", "スライド切替検出"),
              package_install=_clone_pip("slide-sync-skill", "pdf"),
              note="slide timeline (and the deck page on screen) as the analysis kind slide_timeline (ADR-091); a pptx / key / odp deck also needs LibreOffice"),
    # ADR-095: the package id is the contract's skill_id `artifacts-skill`, the prefix of its provides[].tool_id (ADR-077's rule)
    Component("artifacts-skill", "artifacts-skill", "verify", "pip", "active", plugin=True, owns=("生成文書の検証",), package_install=_clone_pip("artifacts-skill", "all"),
              note="a document that goes out with the video verified before the job is done (skill document_verify, ADR-095)"),
    Component("publish", "publish-skill", "delivery", "pip", "active", owns=("配信先アップロード", "アップロード制限"),
              package_install=_clone_pip("publish-skill"),
              note="plan / upload ship (README, EXPERIMENTAL) with the uploaders its contract names; "
                   "`[keychain]` adds the OS keychain for credentials; the agent plans the upload and a human confirms it (ADR-087)"),
    # ADR-089: export / conform / diff / import ship (README); ecosystem.json lists it active and in the marketplace (ADR-057 addendum 1)
    Component("nle-bridge", "nle-bridge-skill", "delivery", "pip", "active", plugin=True, owns=("NLE連携(タイムライン受け渡し)",), package_install=_clone_pip("nle-bridge-skill"),
              note="export / conform / diff / import ship (README, EXPERIMENTAL, standard library only); "
                   "the agent hands it its project IR for delivery.nle (ADR-089)"),
    # ADR-094: the disc authoring app over the four kits (vendored); not on PyPI (its SKILL.md: pip install from the repository).
    # ADR-115: the one disc package a SkillSpec cites; the kits' own repositories stay in ecosystem.json, but no job here hands off to them
    Component("oncemore", "oncemore", "delivery", "pip", "active", optional=True, owns=("ディスク作成アプリ(Mac)",),
              package_install="pip install git+https://github.com/kajisho5/oncemore", note="MCP: claude mcp add --scope user oncemore -- oncemore mcp"),
    # ADR-077: the agent's package id is imagemagick-skill (the Skill's own skill.id and tool-id prefix; ecosystem.json's id is `imagemagick`)
    Component("imagemagick-skill", "imagemagick-skill", "utility", "checkout", "active", plugin=True, owns=("静止画編集",), package_install="npx imagemagick-skill",
              note="the agent prepares edit.overlay images with its resize / pad / strip (ADR-077); `npx imagemagick-skill` copies the checkout to "
                   "~/.claude/skills/imagemagick-skill, where the agent finds it (or VIDEO_AGENT_IMAGEMAGICK_DIR); ImageMagick itself is the user's install"),
    # ADR-080: package ids are the Skills' own skill_id (the prefix of their provides[].tool_id), as for imagemagick-skill (ADR-077)
    Component("blender-skill", "blender-skill", "visual", "checkout", "active", plugin=True, owns=("3D素材・3Dレンダリング",), package_install="npx blender-skill",
              note="render_3d is handed off with blender-skill/render named from its pinned contract (ADR-080); Blender 4.2+ itself is the user's install"),
    # ADR-093: the package id is the Skill's own skill_id (remotion-ffmpeg-skill), the prefix of its provides[].tool_id
    Component("remotion-ffmpeg-skill", "remotion-ffmpeg-skill", "visual", "checkout", "active", plugin=True, owns=("計測連動モーショングラフィックス(Remotion)", "語の時刻に合わせた字幕演出", "オーディオビジュアライザー"),
              package_install="npx remotion-ffmpeg-skill",
              note="one composition over the edited picture as the skill measured_motion (ADR-093); Remotion needs Node 18+ and the user's own project (init.py, npm install)"),
    Component("hyperframes-skill", "hyperframes-skill", "visual", "npm", "active", plugin=True, owns=("HTMLシーンの動画化(HyperFrames)", "テロップ・スティンガー・ループ素材", "キネティックタイポ・幾何学モーション"),
              package_install=f"git clone {GITHUB}hyperframes-skill && cd hyperframes-skill && npm ci",
              note="renders the motion.layer clip (skill html_scene_layer, ADR-114: motion.layer=hyperframes, ProRes 4444 with alpha); html_scene_render is handed off with hyperframes-skill/render named from its pinned contract (ADR-080); needs Node.js 22+ and a headless Chromium"),
)

COMPONENTS: Dict[str, Component] = {c.package: c for c in _COMPONENTS}


def component(package: str) -> Optional[Component]:
    return COMPONENTS.get(package)


def repository(package: str) -> str:
    """The ecosystem repository of a package id; an id this snapshot does not list keeps ADR-051's original `-skill` rule."""
    c = COMPONENTS.get(package)
    return c.repo if c else (package if package.endswith("-skill") else package + "-skill")


def install_commands(package: str) -> List[str]:
    c = COMPONENTS.get(package)
    if c is not None:
        return c.install_commands()
    repo = repository(package)
    return [f"/plugin install {repo}@{MARKETPLACE}", "or ./install.sh from Velisa (packages and CLIs)"]
