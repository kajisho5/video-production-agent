"""Capability detection. ffmpeg-skill does no feature detection (ARCHITECTURE_REVIEW §1.7), so this is
the one place where the agent talks to ffmpeg directly, and only for `-version / -encoders / -decoders / -filters`.
No media is ever processed here."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..tools.ffmpeg_skill.locate import locate_ffmpeg_skill
from ..tools.media_analysis import MediaAnalysisAdapter, locate_media_analysis
from ..tools.transcription import TranscriptionAdapter, locate_transcription
from ..tools.video_editing import VideoEditingAdapter, locate_video_editing

ENCODERS = ["libx264", "libx265", "aac", "prores_ks", "libaom-av1", "libsvtav1", "h264_nvenc", "hevc_nvenc", "h264_videotoolbox", "hevc_videotoolbox", "h264_vaapi", "h264_qsv"]
DECODERS = ["h264", "hevc", "av1", "prores", "vp9"]
FILTERS = {"libass": ["subtitles", "ass"], "zimg": ["zscale"], "tonemap": ["tonemap"], "loudnorm": ["loudnorm"], "scdet": ["scdet"],
           "blackdetect": ["blackdetect"], "freezedetect": ["freezedetect"], "astats": ["astats"], "xfade": ["xfade"], "acrossfade": ["acrossfade"]}
AI_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


@dataclass
class Capability:
    name: str
    status: str                       # AVAILABLE | MISSING | DEGRADED | UNKNOWN
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail, "evidence": self.evidence}


def _run(cmd: List[str], timeout: float = 20.0) -> Optional[str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 else None


class CapabilityResolver:
    def __init__(self, ffmpeg_skill_dir: Optional[str] = None, env: Optional[Dict[str, str]] = None, media_analysis_dir: Optional[str] = None,
                 transcription_dir: Optional[str] = None, offline: bool = False, video_editing_dir: Optional[str] = None):
        self.media_analysis_dir = media_analysis_dir
        self.transcription_dir = transcription_dir
        self.video_editing_dir = video_editing_dir
        self.offline = bool(offline)
        self.env = dict(os.environ if env is None else env)
        self.skill_dir = ffmpeg_skill_dir
        self._cache: Optional[Dict[str, Capability]] = None

    def resolve(self, refresh: bool = False) -> Dict[str, Capability]:
        if self._cache is not None and not refresh:
            return self._cache
        caps: Dict[str, Capability] = {}
        caps["python"] = Capability("python", "AVAILABLE" if sys.version_info >= (3, 9) else "DEGRADED", platform.python_version())
        caps["os"] = Capability("os", "AVAILABLE", platform.system())
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        ver = _run([ffmpeg, "-version"]) if ffmpeg else None
        caps["ffmpeg"] = Capability("ffmpeg", "AVAILABLE" if ver else "MISSING", (ver or "").splitlines()[0] if ver else "not on PATH", {"path": ffmpeg})
        pver = _run([ffprobe, "-version"]) if ffprobe else None
        caps["ffprobe"] = Capability("ffprobe", "AVAILABLE" if pver else "MISSING", (pver or "").splitlines()[0] if pver else "not on PATH", {"path": ffprobe})
        enc_txt = _run([ffmpeg, "-hide_banner", "-encoders"]) if ffmpeg else None
        dec_txt = _run([ffmpeg, "-hide_banner", "-decoders"]) if ffmpeg else None
        flt_txt = _run([ffmpeg, "-hide_banner", "-filters"]) if ffmpeg else None
        for name in ENCODERS:
            caps[f"encoder:{name}"] = Capability(f"encoder:{name}", _listed(enc_txt, name))
        for name in DECODERS:
            caps[f"decoder:{name}"] = Capability(f"decoder:{name}", _listed(dec_txt, name))
        for lib, filters in FILTERS.items():
            st = [_listed(flt_txt, f) for f in filters]
            caps[f"filter:{lib}"] = Capability(f"filter:{lib}", "AVAILABLE" if all(s == "AVAILABLE" for s in st) else ("MISSING" if "MISSING" in st else "UNKNOWN"), ", ".join(filters))
        hw = [n for n in ENCODERS if ("nvenc" in n or "videotoolbox" in n or "vaapi" in n or "qsv" in n) and caps[f"encoder:{n}"].status == "AVAILABLE"]
        caps["gpu"] = Capability("gpu", "AVAILABLE" if hw else ("UNKNOWN" if enc_txt is None else "MISSING"), ", ".join(hw) or "no hardware encoder listed by ffmpeg")
        caps["hevc"] = Capability("hevc", caps["encoder:libx265"].status, "libx265")
        caps["prores"] = Capability("prores", caps["encoder:prores_ks"].status, "prores_ks")
        caps["av1"] = Capability("av1", "AVAILABLE" if any(caps[f"encoder:{n}"].status == "AVAILABLE" for n in ("libaom-av1", "libsvtav1")) else caps["encoder:libaom-av1"].status)
        # fonts
        fc = shutil.which("fc-list")
        if fc:
            all_fonts = _run([fc]) or ""
            ja = _run([fc, ":lang=ja", "family"]) or ""
            caps["fonts"] = Capability("fonts", "AVAILABLE" if all_fonts.strip() else "MISSING", f"{len(all_fonts.splitlines())} fonts")
            caps["font:cjk-ja"] = Capability("font:cjk-ja", "AVAILABLE" if ja.strip() else "MISSING", (ja.strip().splitlines() or ["none"])[0])
        else:
            caps["fonts"] = Capability("fonts", "UNKNOWN", "fc-list not found")
            caps["font:cjk-ja"] = Capability("font:cjk-ja", "UNKNOWN", "fc-list not found")
        # ffmpeg-skill
        skill = locate_ffmpeg_skill(self.skill_dir, self.env)
        if skill:
            caps["ffmpeg-skill"] = Capability("ffmpeg-skill", "AVAILABLE", f"{skill.version} at {skill.root}", {"root": str(skill.root), "version": skill.version, "scripts": skill.scripts})
        else:
            caps["ffmpeg-skill"] = Capability("ffmpeg-skill", "MISSING", "set VIDEO_AGENT_FFMPEG_SKILL_DIR or install with `npx ffmpeg-skill`")
        # media-analysis-skill (external observation Skill): located checkout / console script + its own doctor
        ma = locate_media_analysis(self.media_analysis_dir, self.env)
        if ma:
            try:
                ad = MediaAnalysisAdapter(ma, timeout=30.0)
                doc = ad.doctor()
                st = "AVAILABLE" if doc.get("status") == "ok" else ("DEGRADED" if doc.get("status") == "degraded" else "MISSING")
                caps["media-analysis"] = Capability("media-analysis", st, f"{ad.version} at {ma.describe()} (doctor {doc.get('status')})",
                                                    {"version": ad.version, "root": ma.describe(), "contract": ad.contract.get("schema"), "tools": sorted(ad.tools), "kinds": sorted(ad.kind_to_tool),
                                                     "execution": ad.contract.get("execution", {}).get("mode"), "doctor": doc.get("status"), "unavailable_tools": doc.get("unavailable_tools") or []})
            except Exception as e:  # noqa: BLE001 — an incompatible or broken installation is reported, never used
                caps["media-analysis"] = Capability("media-analysis", "MISSING", f"found at {ma.describe()} but unusable: {str(e)[:160]}")
        else:
            caps["media-analysis"] = Capability("media-analysis", "MISSING", "set VIDEO_AGENT_MEDIA_ANALYSIS_DIR to a media-analysis-skill checkout or install `media-analysis`")
        # transcription-skill (external recognition Skill): located checkout / console script + its own doctor and engine contract
        ts = locate_transcription(self.transcription_dir, self.env)
        if ts:
            try:
                ad = TranscriptionAdapter(ts, timeout=120.0, offline=self.offline)
                doc = ad.doctor()
                rows = {c.get("check"): c for c in doc.get("checks") or []}
                engines = ad.engine_status()
                usable = [e for e in engines if e.get("available")]
                st = "AVAILABLE" if doc.get("ok") else ("DEGRADED" if usable else "MISSING")
                caps["transcription"] = Capability("transcription", st, f"{ad.version} at {ts.describe()} (doctor {'ok' if doc.get('ok') else doc.get('summary', 'not ready')})",
                                                   {"version": ad.version, "root": ts.describe(), "schemas": dict(ad.contract.get("schemas") or {}), "tools": sorted(ad.tools),
                                                    "capabilities": list(ad.contract.get("capabilities") or []), "offline": self.offline,
                                                    "engines": [{"id": e.get("id"), "version": e.get("version"), "execution_mode": e.get("execution_mode"), "requires_network": e.get("requires_network"),
                                                                 "available": e.get("available"), "capabilities": e.get("capabilities"), "default_model": e.get("default_model"),
                                                                 "models": [{"model": m.get("model"), "availability": m.get("availability")} for m in e.get("models") or []]} for e in engines],
                                                    "doctor": {k: v.get("status") for k, v in rows.items()}, "doctor_ok": bool(doc.get("ok"))})
            except Exception as e:  # noqa: BLE001 — an incompatible or broken installation is reported, never used
                caps["transcription"] = Capability("transcription", "MISSING", f"found at {ts.describe()} but unusable: {str(e)[:160]}")
        else:
            caps["transcription"] = Capability("transcription", "MISSING", "set VIDEO_AGENT_TRANSCRIPTION_DIR to a transcription-skill checkout or install `transcription`")
        # video-editing-skill (external editing Skill, ADR-028): located checkout / console script, its contract and its own doctor
        # (which asks ffmpeg-skill for ffmpeg / ffprobe). AVAILABLE only when the Skill says it is ready; anything else is MISSING —
        # a half-usable editing engine is never guessed at.
        ve = locate_video_editing(self.video_editing_dir, self.env)
        if ve:
            try:
                ad = VideoEditingAdapter(ve, timeout=120.0, ffmpeg_skill_dir=str(skill.root) if skill else None)
                doc = ad.doctor()
                drift = ad.drift()
                ok = bool(doc.get("ok")) and not drift
                detail = f"{ad.version} at {ve.describe()} (doctor {'ok' if doc.get('ok') else doc.get('summary', 'not ready')})" + ("; contract drift: " + "; ".join(drift)[:200] if drift else "")
                caps["video-editing"] = Capability("video-editing", "AVAILABLE" if ok else "MISSING", detail,
                                                   {"version": ad.version, "root": ve.describe(), "contract": ad.contract.get("schema"), "tools": sorted(ad.tools),
                                                    "engine": dict(ad.contract.get("engine") or {}), "doctor_ok": bool(doc.get("ok")), "problems": list(doc.get("problems") or []),
                                                    "doctor": {c.get("check"): c.get("status") for c in doc.get("checks") or [] if isinstance(c, dict)}, "drift": drift})
            except Exception as e:  # noqa: BLE001 — an incompatible or broken installation is reported, never used
                caps["video-editing"] = Capability("video-editing", "MISSING", f"found at {ve.describe()} but unusable: {str(e)[:160]}")
        else:
            caps["video-editing"] = Capability("video-editing", "MISSING", "set VIDEO_AGENT_VIDEO_EDITING_DIR to a video-editing-skill checkout or install `video-editing`")
        # optional AI / ASR
        asr = shutil.which("whisper-cli") or shutil.which("whisper-cpp") or shutil.which("whisper")
        try:
            import faster_whisper  # type: ignore  # noqa: F401
            asr = asr or "faster-whisper"
        except Exception:
            pass
        caps["asr:whisper"] = Capability("asr:whisper", "AVAILABLE" if asr else "MISSING", asr or "no local whisper engine")
        for prov, var in AI_ENV.items():
            caps[f"ai:{prov}"] = Capability(f"ai:{prov}", "AVAILABLE" if self.env.get(var) else "MISSING", f"{var} {'set' if self.env.get(var) else 'not set'}")
        configured = (self.env.get("VIDEO_AGENT_AI_PROVIDER") or "null").lower()
        caps["ai:provider"] = Capability("ai:provider", "AVAILABLE" if configured not in ("", "null", "none") else "MISSING",
                                         f"VIDEO_AGENT_AI_PROVIDER={configured} (deterministic pipeline; AI recommendations are proposals only)", {"provider": configured})
        self._cache = caps
        return caps

    def status(self, name: str) -> str:
        return self.resolve().get(name, Capability(name, "UNKNOWN")).status

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() for k, v in self.resolve().items()}


def _listed(text: Optional[str], name: str) -> str:
    if text is None:
        return "UNKNOWN"
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == name:
            return "AVAILABLE"
    return "MISSING"
