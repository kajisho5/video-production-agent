"""Artifact naming: safe, deterministic delivery file names from the profile's naming template.
The compiler owns where a job writes its outputs; this module only produces the *delivery name* recorded on the
artifact manifest (portable across OS / filesystems). Never a path, never chosen by an AI."""
from __future__ import annotations

import re
from typing import Any, Dict

WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
MAX_STEM = 120


def safe_filename(stem: str, ext: str = "") -> str:
    """A single file name component: no separators or traversal, no control / reserved characters, no Windows reserved
    device names, no trailing dot / space, bounded length. Empty input becomes 'artifact'."""
    s = str(stem or "")
    s = s.replace("\\", "/").split("/")[-1]                 # never a path
    s = _INVALID.sub("_", s).replace("..", "_")
    s = s.strip(" .")
    s = re.sub(r"\s+", "_", s)
    if not s:
        s = "artifact"
    if s.upper().split(".")[0] in WINDOWS_RESERVED:
        s = f"_{s}"
    if len(s) > MAX_STEM:
        s = s[:MAX_STEM].rstrip(" .")
    ext = re.sub(r"[^A-Za-z0-9]", "", str(ext or "")).lower()
    return f"{s}.{ext}" if ext else s


def delivery_name(template: str, fields: Dict[str, Any], ext: str) -> str:
    """Render a naming template such as '{project}_{target}_{version}' with safe field values. Unknown placeholders
    are left literal (then sanitised); a missing template falls back to project_target_vN."""
    tpl = template or "{project}_{target}_{version}"
    values = {k: safe_filename(str(v)) if v is not None else "" for k, v in fields.items()}
    try:
        out = tpl.format(**{**{"project": "", "target": "", "version": "", "format": "", "date": "", "session": "", "event": "", "speaker": ""}, **values})
    except (KeyError, IndexError, ValueError):
        out = "_".join(str(values.get(k, "")) for k in ("project", "target", "version"))
    out = re.sub(r"_{2,}", "_", out).strip("_")   # empty fields collapse instead of leaving "__"
    return safe_filename(out, ext)
