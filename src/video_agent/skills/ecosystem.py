"""Who in the AI Video Production OS ecosystem owns a job, and how a user installs it (ADR-051 handoffs, ADR-057).

A pinned, hand-checked snapshot of AI-video-production-OS `ecosystem.json` (schema_version 2, read 2026-10-30) for every
package a SkillSpec cites, plus each package's install command taken from its own README. It is data only: nothing here is
executed, and an entry never makes a skill selectable (that needs an adapter). `tests/test_unit.py` compares the snapshot with
a sibling `AI-video-production-OS/ecosystem.json` when one is checked out next to this repository.

The agent names a package by its short id (`video-editing`, `bdmvkit`), the ecosystem by its repository (`video-editing-skill`);
`repository()` maps one to the other instead of guessing a `-skill` suffix."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ECOSYSTEM_SOURCE = "AI-video-production-OS ecosystem.json (schema_version 2), read 2026-10-30"
MARKETPLACE = "ai-video-production-os"
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
        out.append("or ./install.sh" + (" --with-optional" if self.optional else "") + " from AI-video-production-OS (packages and CLIs)")
        return out

    def to_dict(self) -> Dict[str, object]:
        return {"package": self.package, "repository": self.repo, "department": self.department, "status": self.status, "plugin": self.plugin,
                "optional": self.optional, "owns": list(self.owns), "released": self.released, "install": self.install_commands()}


def _clone_pip(repo: str, extras: str = "") -> str:
    return f"git clone {GITHUB}{repo} && pip install -e '{repo}{('[' + extras + ']') if extras else ''}'"


_COMPONENTS: Tuple[Component, ...] = (
    # ---- packages the agent already has adapters for (ADR-051's original handoff targets)
    Component("ffmpeg-skill", "ffmpeg-skill", "engine", "checkout", "active", plugin=True,
              owns=("FFmpeg実行", "書き出し規格の検査", "マルチカム同期", "Bロール・画面合成", "手ぶれ・傾き補正", "プレビュー・縮小一覧", "納品レポート", "配信プラットフォーム仕様"),
              package_install="npx ffmpeg-skill"),
    Component("video-editing", "video-editing-skill", "visual", "pip", "active", plugin=True, owns=("カット・連結", "尺・画角合わせ", "トランジション", "静止画オーバーレイ"),
              package_install=_clone_pip("video-editing-skill")),
    Component("color-grading", "color-grading-skill", "visual", "pip", "active", plugin=True, owns=("色補正", "LUT", "HDR→SDR"), package_install=_clone_pip("color-grading-skill")),
    Component("motion-graphics", "motion-graphics-skill", "visual", "pip", "active", plugin=True, owns=("タイトル・テロップ",), package_install=_clone_pip("motion-graphics-skill")),
    Component("subtitle", "subtitle-skill", "visual", "pip", "active", plugin=True, owns=("字幕生成・検証", "字幕焼き込み", "字幕翻訳", "ロケール別字幕整形"),
              package_install=_clone_pip("subtitle-skill")),
    Component("thumbnail", "thumbnail-skill", "visual", "pip", "active", plugin=True, owns=("サムネイル",), package_install=_clone_pip("thumbnail-skill")),
    Component("audio-production", "audio-production-skill", "audio", "pip", "active", plugin=True,
              owns=("音量・ラウドネス", "ミックス", "ダイナミクス", "従来型ノイズ低減", "無音・フィラー除去"), package_install=_clone_pip("audio-production-skill")),
    Component("transcription", "transcription-skill", "recognize", "pip", "active", plugin=True, owns=("文字起こし",),
              package_install=_clone_pip("transcription-skill", "faster-whisper")),
    Component("qc", "qc-skill", "verify", "pip", "active", plugin=True, owns=("規格判定(QC)",), package_install=_clone_pip("qc-skill")),
    Component("media-analysis", "media-analysis-skill", "verify", "pip", "active", plugin=True, owns=("メディア計測",), package_install=_clone_pip("media-analysis-skill")),
    # ---- ADR-057: every other job of the ecosystem, declared so a handoff names its owner (no adapter here yet; roadmap step 9)
    Component("ingest", "ingest-skill", "ingest", "pip", "scaffold", plugin=True, owns=("素材取り込み(検証付きコピー)", "二重バックアップ", "素材台帳", "URL素材取得"),
              package_install=_clone_pip("ingest-skill", "fetch"),
              note="ecosystem.json still says scaffold, but offload / verify / inventory ship (README); `ingest fetch` (URL download, ingest-skill #9) "
                   "needs the fetch extra (yt-dlp), and the agent hands URL inputs to it (ADR-059)"),
    Component("voice-enhance", "voice-enhance-skill", "audio", "pip", "scaffold", owns=("声の分離(AI)", "残響除去", "マイク被り低減"),
              package_install=_clone_pip("voice-enhance-skill"),
              note="ecosystem.json still says scaffold, but isolate / dereverb / bleed-reduce ship (README, 0.1.0 unreleased, all EXPERIMENTAL); the core is stdlib-only and "
                   "each engine is its own install (deep-filter + the DeepFilterNet3 model, `[wpe]`, `[bleed]`) that `voice-enhance doctor` names (ADR-073)"),
    Component("dubbing", "dubbing-skill", "audio", "pip", "scaffold", owns=("AIナレーション(TTS)", "吹替・声の置換", "音声解説")),
    Component("sound-design", "sound-design-skill", "audio", "pip", "scaffold", owns=("BGM生成", "効果音")),
    Component("diarization", "diarization-skill", "recognize", "pip", "active", plugin=True, optional=True, owns=("話者識別",),
              package_install=_clone_pip("diarization-skill", "pyannote"),
              note="speaker turns as the analysis kind speaker_turns (ADR-086); diarize also needs the gated speaker-diarization-community-1 pipeline cloned once "
                   "and DIARIZATION_PIPELINE_DIR set (`diarization doctor` prints the steps; the Skill never downloads)"),
    Component("detection", "detection-skill", "recognize", "pip", "active", plugin=True, optional=True, owns=("顔検出", "文字検出(OCR)", "物体検出"),
              package_install=_clone_pip("detection-skill", "all")),
    Component("slide-sync", "slide-sync-skill", "recognize", "pip", "active", plugin=True, owns=("スライド取込", "スライド切替検出"),
              package_install=_clone_pip("slide-sync-skill", "pdf")),
    Component("artifacts", "artifacts-skill", "verify", "pip", "active", plugin=True, owns=("生成文書の検証",), package_install=_clone_pip("artifacts-skill", "all")),
    Component("publish", "publish-skill", "delivery", "pip", "scaffold", owns=("配信先アップロード", "アップロード制限")),
    Component("bdmvkit", "bdmvkit", "delivery", "pip", "active", optional=True, owns=("Blu-ray作成",), package_install="pip install bdmvkit"),
    Component("vtskit", "vtskit", "delivery", "pip", "active", optional=True, owns=("DVD-Video作成",), package_install="pip install vtskit"),
    Component("udfkit", "udfkit", "delivery", "pip", "active", optional=True, owns=("ディスクイメージ(UDF)",), package_install="pip install udfkit"),
    Component("dvdcc", "dvdcc", "delivery", "pip", "active", optional=True, owns=("DVD字幕(CEA-608)",), package_install="pip install dvdcc"),
    # ADR-077: the agent's package id is imagemagick-skill (the Skill's own skill.id and tool-id prefix; ecosystem.json's id is `imagemagick`)
    Component("imagemagick-skill", "imagemagick-skill", "utility", "checkout", "active", plugin=True, owns=("静止画編集",), package_install="npx imagemagick-skill",
              note="the agent prepares edit.overlay images with its resize / pad / strip (ADR-077); `npx imagemagick-skill` copies the checkout to "
                   "~/.claude/skills/imagemagick-skill, where the agent finds it (or VIDEO_AGENT_IMAGEMAGICK_DIR); ImageMagick itself is the user's install"),
    Component("blender", "blender-skill", "visual", "checkout", "active", plugin=True, owns=("3D素材・3Dレンダリング",), package_install="npx blender-skill"),
    Component("remotion-ffmpeg", "remotion-ffmpeg-skill", "visual", "checkout", "active", plugin=True, owns=("計測連動モーショングラフィックス(Remotion)",),
              package_install="npx remotion-ffmpeg-skill"),
    Component("hyperframes", "hyperframes-skill", "visual", "npm", "active", plugin=True, owns=("HTMLシーンの動画化(HyperFrames)",),
              package_install=f"git clone {GITHUB}hyperframes-skill && cd hyperframes-skill && npm ci"),
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
    return [f"/plugin install {repo}@{MARKETPLACE}", "or ./install.sh from AI-video-production-OS (packages and CLIs)"]
