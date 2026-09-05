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

ENCODERS = ["libx264", "libx265", "prores_ks", "libaom-av1", "libsvtav1", "h264_nvenc", "hevc_nvenc", "h264_videotoolbox", "hevc_videotoolbox", "h264_vaapi", "h264_qsv"]
DECODERS = ["h264", "hevc", "av1", "prores", "vp9"]
FILTERS = {"libass": ["subtitles", "ass"], "zimg": ["zscale"], "tonemap": ["tonemap"], "loudnorm": ["loudnorm"], "scdet": ["scdet"],
           "blackdetect": ["blackdetect"], "freezedetect": ["freezedetect"], "astats": ["astats"]}
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
    def __init__(self, ffmpeg_skill_dir: Optional[str] = None, env: Optional[Dict[str, str]] = None):
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
