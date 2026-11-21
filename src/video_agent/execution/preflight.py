"""Export preflight (ADR-126): a fast check of an approved plan's deliverables before anything is encoded.

Every problem is reported at once, as one result:

    {"ok": bool, "problems": [{"code", "severity": "error" | "warning", "message", "field"}], "estimates": {...}, "unverified": [...]}

`ok` is false when any problem is an error. The check reads file metadata, the pinned contracts and the plan; it encodes nothing,
runs no Skill and writes nothing that stays (the output directory's writability is tested by creating and removing one empty file).

What is checked:
  output   the directory the job will write (the workspace's jobs/ folder, or its nearest existing parent) exists, is a directory and
           is writable; a planned output that already exists (an output outside the new job directory) is reported with its size;
           the free space there covers the estimated writes times ESTIMATE_MARGIN.
  format   each delivery target's preset and platform are ones the media engine checkout's own contract publishes; a platform's
           upload limits that publish-skill's pinned contract documents (file formats, duration, file size) are met by the plan's
           expected duration and the size estimate. Limits that contract delegates to the engine (resolution, fps, codecs, aspect)
           or does not know are listed under `unverified`, never guessed.
  inputs   every source and every user file the compiled operations read exists, is readable, and has the size and mtime the plan
           recorded (a sha256 recorded by the analysis is not recomputed: that would read the whole file); a path on a volume that is
           not mounted (a card or drive taken out) is reported as such.
  media    the font directory of a burn-in exists and holds font files; the transcript / subtitle files of a transcript edit exist; a
           profile that sets a loudness target has one on every subject with audio.
  tooling  every plan step's tool is executable here, the capabilities the plan needs are AVAILABLE (the validator's check, through
           the capability resolver), and the tool versions the plan recorded are the ones installed.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

ERROR, WARNING = "error", "warning"
ESTIMATE_MARGIN = 1.2          # required free space = estimated writes x this
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".mxf", ".avi", ".ts", ".gif")
AUDIO_EXTENSIONS = (".wav", ".flac", ".m4a", ".aac", ".mp3", ".ogg", ".opus", ".aif", ".aiff")
FONT_EXTENSIONS = (".ttf", ".otf", ".ttc", ".otc", ".woff", ".woff2")
MTIME_TOLERANCE_S = 1e-3
# platform-limit keys of publish-skill's upload_limits table this check reads (the table's own names)
LIMIT_KEYS = ("video.file_formats", "video.max_duration_s", "video.min_duration_s", "video.max_bytes", "video.unverified_max_duration_s")


def disk_free(path: str) -> Optional[int]:
    """Free bytes on the volume of `path` (an existing directory); None when the OS does not say."""
    try:
        return int(shutil.disk_usage(path).free)
    except OSError:
        return None


def probe_writable(directory: str) -> Optional[str]:
    """None when an empty file can be created (and is removed again) in `directory`; else why not."""
    try:
        fd, name = tempfile.mkstemp(prefix=".video-agent-preflight-", dir=directory)
    except OSError as exc:
        return exc.strerror or str(exc)
    os.close(fd)
    try:
        os.unlink(name)
    except OSError:
        pass
    return None


def nearest_existing(path: str) -> str:
    p = Path(os.path.abspath(path))
    while not p.exists() and p.parent != p:
        p = p.parent
    return str(p)


def missing_volume(path: str) -> Optional[str]:
    """The mount point of `path` when it is a removable / mounted volume that is not there now (/Volumes/<name>, /media/<name>,
    /media/<user>/<name>, /run/media/<user>/<name>, /mnt/<name>, or a Windows drive letter); None otherwise."""
    p = Path(os.path.abspath(path))
    parts = p.parts
    if parts and re.match(r"^[A-Za-z]:[\\/]?$", parts[0]):
        return None if Path(parts[0]).exists() else parts[0]
    candidates: List[str] = []
    if len(parts) >= 3 and parts[1] in ("Volumes", "mnt", "media"):
        candidates.append(os.path.join(parts[0], parts[1], parts[2]))
        if parts[1] == "media" and len(parts) >= 5:
            candidates.append(os.path.join(parts[0], parts[1], parts[2], parts[3]))
    if len(parts) >= 5 and parts[1] == "run" and parts[2] == "media":
        candidates.append(os.path.join(parts[0], parts[1], parts[2], parts[3], parts[4]))
    for c in candidates:
        if not os.path.exists(c):
            return c
    return None


def _kind_of(path: str) -> str:
    ext = Path(path).suffix.lower()
    return "video" if ext in VIDEO_EXTENSIONS else ("audio" if ext in AUDIO_EXTENSIONS else "other")


def _within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:
        return False


class Report:
    def __init__(self) -> None:
        self.problems: List[Dict[str, Any]] = []
        self.estimates: Dict[str, Any] = {}
        self.unverified: List[Dict[str, Any]] = []

    def add(self, code: str, severity: str, message: str, field: str) -> None:
        self.problems.append({"code": code, "severity": severity, "message": message, "field": field})

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": not any(p["severity"] == ERROR for p in self.problems), "problems": list(self.problems), "estimates": dict(self.estimates),
                "unverified": list(self.unverified)}


def _bytes_h(n: Optional[float]) -> str:
    if n is None:
        return "unknown"
    v = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(v) < 1000 or unit == "TB":
            return f"{v:.0f} {unit}" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1000.0
    return f"{v:.1f} TB"


# ---- inputs
def check_input_file(rep: Report, path: str, field: str, recorded: Optional[Dict[str, Any]] = None, sha256: Optional[str] = None, label: str = "input") -> None:
    """One file the plan reads: there, a file, readable, unchanged since the plan (size + mtime when the plan recorded them)."""
    if not os.path.exists(path):
        vol = missing_volume(path)
        if vol:
            rep.add("INPUT_VOLUME_MISSING", ERROR, f"{label} {path} is on {vol}, which is not mounted (card or drive removed?)", field)
        else:
            rep.add("INPUT_MISSING", ERROR, f"{label} {path} does not exist", field)
        return
    if not os.path.isfile(path):
        rep.add("INPUT_NOT_A_FILE", ERROR, f"{label} {path} is not a file", field)
        return
    if not os.access(path, os.R_OK):
        rep.add("INPUT_UNREADABLE", ERROR, f"{label} {path} is not readable", field)
        return
    if recorded is None:
        return
    size, mtime = recorded.get("size"), recorded.get("mtime")
    if size is None or mtime is None:
        rep.add("INPUT_FINGERPRINT_UNKNOWN", WARNING, f"{label} {path}: the plan recorded no size / mtime, so a replaced file cannot be detected", field)
        return
    st = os.stat(path)
    changed = []
    if int(st.st_size) != int(size):
        changed.append(f"size {_bytes_h(size)} -> {_bytes_h(st.st_size)}")
    if abs(float(st.st_mtime) - float(mtime)) > MTIME_TOLERANCE_S:
        changed.append("modified time changed")
    if changed:
        note = " (the analysis recorded its sha256; plan again or revise to re-analyse it)" if sha256 else " (plan again or revise to re-analyse it)"
        rep.add("INPUT_CHANGED", ERROR, f"{label} {path} changed since the plan: {', '.join(changed)}{note}", field)


def check_inputs(rep: Report, doc: Dict[str, Any], ops: List[Any], paths: Dict[str, str], job_dir: str) -> None:
    assets = doc.get("assets") or {}
    for aid, a in assets.items():
        check_input_file(rep, str(a.get("path")), f"assets.{aid}.path", recorded=(a.get("technical") or {}).get("file") or {}, sha256=a.get("hash"), label=f"asset {aid}")
    produced = {o for op in ops for o in op.outputs}
    seen = set(assets)
    for op in ops:
        for ref in op.inputs:
            if ref in seen or ref in produced:
                continue
            seen.add(ref)
            p = paths.get(ref)
            if not p or _within(p, job_dir):   # a file the job itself writes before running (the NLE project IR)
                continue
            if os.path.isdir(p):
                continue   # a library folder (a music bed library): its own Skill reads it
            check_input_file(rep, p, f"paths.{ref}", label=f"file {ref}")


# ---- media
def check_media(rep: Report, doc: Dict[str, Any], profile_rules: Iterable[Any], subjects: List[Dict[str, Any]]) -> None:
    for n, op in enumerate((doc.get("captions") or {}).get("operations") or []):
        fd = op.get("fonts_dir")
        if not fd:
            continue
        field = f"captions.operations[{n}].fonts_dir"
        if not os.path.isdir(fd):
            vol = missing_volume(fd)
            rep.add("FONTS_DIR_MISSING", ERROR, f"the burn-in's font directory {fd} " + (f"is on {vol}, which is not mounted" if vol else "does not exist"), field)
        elif not any(f.suffix.lower() in FONT_EXTENSIONS for f in Path(fd).iterdir() if f.is_file()):
            rep.add("FONTS_DIR_EMPTY", WARNING, f"the burn-in's font directory {fd} holds no font file ({', '.join(FONT_EXTENSIONS)})", field)
    for d in doc.get("decisions") or []:
        if d.get("subject") != "transcript.ripple" or d.get("status") == "REJECTED":
            continue
        p_ = d.get("params") or {}
        files = [("transcript", (p_.get("transcript") or {}).get("path"))] + [(f"subtitles[{i}]", (s or {}).get("path")) for i, s in enumerate(p_.get("subtitles") or [])]
        for key, path in files:
            if path and not os.path.isfile(path):
                rep.add("SIDECAR_MISSING", WARNING, f"{d['id']}: the {key.split('[')[0]} file {path} to retime after the render does not exist (the render runs; it is not retimed)",
                        f"decisions.{d['id']}.params.{key}.path")
    wants = [r for r in profile_rules if getattr(r, "key", None) == "audio.loudness.target_lufs"]
    if not wants:
        return
    rejected = any(str(d.get("subject", "")).startswith("audio.loudness") and d.get("status") == "REJECTED" for d in doc.get("decisions") or [])
    for row in subjects:
        has_audio = bool((row.get("technical") or {}).get("audio"))
        if has_audio and row.get("target_lufs") is None:
            rep.add("LOUDNESS_TARGET_UNSET", WARNING if rejected else ERROR,
                    f"the profile sets a loudness target ({wants[-1].value} LUFS) but subject {row['id']} has none"
                    + (" (its loudness decision was rejected)" if rejected else ""), f"audio.operations.{row['id']}.target_lufs")


# ---- format and size
def _limit(limits: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = limits.get(key)
    return v if isinstance(v, dict) else {}


def check_targets(rep: Report, doc: Dict[str, Any], subjects: List[Dict[str, Any]], paths: Dict[str, str], estimates: Dict[str, int],
                  presets: Optional[List[str]], platforms: Optional[List[str]], upload_limits: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per delivery target: preset / platform known to the engine's contract, the documented platform limits; → deliverable rows."""
    rows: List[Dict[str, Any]] = []
    targets = (doc.get("delivery") or {}).get("targets") or []
    for i, t in enumerate(targets):
        preset, platform = t.get("preset"), str(t.get("platform") or "custom")
        if preset:
            if presets is None:
                rep.unverified.append({"target": t["id"], "field": f"delivery.targets[{i}].preset", "reason": "the engine checkout's contract publishes no preset list"})
            elif preset not in presets:
                rep.add("FORMAT_PRESET_UNKNOWN", ERROR, f"target {t['id']}: preset {preset!r} is not one the engine's contract publishes ({', '.join(sorted(presets))})",
                        f"delivery.targets[{i}].preset")
        if platform != "custom":
            if platforms is None:
                rep.unverified.append({"target": t["id"], "field": f"delivery.targets[{i}].platform", "reason": "the engine checkout's contract publishes no platform list"})
            elif platform not in platforms:
                rep.add("FORMAT_PLATFORM_UNKNOWN", ERROR, f"target {t['id']}: platform {platform!r} is not one the engine's check knows ({', '.join(sorted(platforms))})",
                        f"delivery.targets[{i}].platform")
        limits = ((upload_limits.get("platforms") or {}).get(platform) or {}) if platform != "custom" else {}
        for row in subjects:
            art = f"{row['id']}_delivery_{t['id']}"
            path = paths.get(art)
            if not path:
                continue
            dur = float(row.get("duration") or 0.0)
            size = estimates.get(art)
            out = {"id": art, "target": t["id"], "subject": row["id"], "path": path, "preset": preset, "platform": platform, "duration_s": round(dur, 3),
                   "bytes": size, "limits": {}}
            rows.append(out)
            if not limits:
                continue
            fmts = _limit(limits, "video.file_formats")
            ext = Path(path).suffix.lower()
            if fmts.get("status") == "documented" and isinstance(fmts.get("value"), list):
                out["limits"]["file_formats"] = fmts["value"]
                if ext not in [str(x).lower() for x in fmts["value"]]:
                    rep.add("PLATFORM_CONTAINER_NOT_ACCEPTED", ERROR, f"{art}: {platform} accepts {', '.join(fmts['value'])}; the deliverable is {ext or 'without an extension'}",
                            f"delivery.targets[{i}].preset")
            mx = _limit(limits, "video.max_duration_s")
            if mx.get("status") == "documented" and isinstance(mx.get("value"), (int, float)):
                out["limits"]["max_duration_s"] = mx["value"]
                if dur > float(mx["value"]):
                    rep.add("PLATFORM_DURATION_OVER", ERROR, f"{art}: about {dur:.1f} s, over {platform}'s {mx['value']} s limit", f"delivery.targets[{i}]")
            ua = _limit(limits, "video.unverified_max_duration_s")
            if ua.get("status") == "documented" and isinstance(ua.get("value"), (int, float)):
                out["limits"]["unverified_account_max_duration_s"] = ua["value"]
                if float(ua["value"]) < dur <= float(mx.get("value") or float("inf")):
                    rep.add("PLATFORM_DURATION_OVER_UNVERIFIED_ACCOUNT", WARNING, f"{art}: about {dur:.1f} s; {platform} limits an unverified account to {ua['value']} s",
                            f"delivery.targets[{i}]")
            mn = _limit(limits, "video.min_duration_s")
            if mn.get("status") == "documented" and isinstance(mn.get("value"), (int, float)):
                out["limits"]["min_duration_s"] = mn["value"]
                if 0 < dur < float(mn["value"]):
                    rep.add("PLATFORM_DURATION_UNDER", ERROR, f"{art}: about {dur:.1f} s, under {platform}'s {mn['value']} s minimum", f"delivery.targets[{i}]")
            mb = _limit(limits, "video.max_bytes")
            if mb.get("status") == "documented" and isinstance(mb.get("value"), (int, float)):
                out["limits"]["max_bytes"] = mb["value"]
                if size is not None and size > float(mb["value"]):
                    rep.add("PLATFORM_SIZE_OVER", WARNING, f"{art}: estimated {_bytes_h(size)}, over {platform}'s {_bytes_h(mb['value'])} limit (an estimate from the source bitrate)",
                            f"delivery.targets[{i}]")
            for key, spec in sorted(limits.items()):
                if key in LIMIT_KEYS or not key.startswith("video.") or not isinstance(spec, dict) or spec.get("status") == "documented":
                    continue
                why = f"delegated to {spec.get('owner') or 'the engine'}: {spec.get('read_from') or 'its check'}" if spec.get("status") == "delegated" else "not documented in the pinned publish-skill contract"
                rep.unverified.append({"target": t["id"], "artifact": art, "field": key, "reason": why})
        if platform != "custom" and not limits:
            rep.unverified.append({"target": t["id"], "field": f"delivery.targets[{i}].platform", "reason": f"no upload limits for {platform!r} in the pinned publish-skill contract"})
    return rows


def estimate_writes(doc: Dict[str, Any], ops: List[Any], paths: Dict[str, str], subjects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimated bytes every operation writes. A deliverable: its subject's source bitrate x the plan's expected duration. Any other
    media output: the sum of its media inputs (an upper bound for a trim). Non-media outputs (JSON, subtitles, images) count 0."""
    assets = doc.get("assets") or {}
    est: Dict[str, float] = {}
    rate: Dict[str, float] = {}
    for aid, a in assets.items():
        tech = a.get("technical") or {}
        dur = float(tech.get("duration") or 0.0)
        size = (tech.get("file") or {}).get("size") or tech.get("size_bytes")
        br = tech.get("bitrate")
        if not br and size and dur:
            br = float(size) * 8.0 / dur
        rate[aid] = float(br or 0.0)
        est[aid] = float(size) if size else (rate[aid] * dur / 8.0)
    deliverables: Dict[str, int] = {}
    for row in subjects:
        br = max([rate.get(s, 0.0) for s in row.get("sources") or []] or [0.0])
        for t in (doc.get("delivery") or {}).get("targets") or []:
            art = f"{row['id']}_delivery_{t['id']}"
            if art in paths:
                deliverables[art] = int(br * float(row.get("duration") or 0.0) / 8.0)
    written: Dict[str, float] = {}
    for op in ops:
        for out in op.outputs:
            p = paths.get(out, "")
            if _kind_of(p) == "other":
                est[out] = 0.0
            elif out in deliverables:
                est[out] = float(deliverables[out])
            else:
                est[out] = sum(est.get(i, 0.0) for i in op.inputs if _kind_of(paths.get(i, "")) != "other")
            if p:
                written[p] = est[out]
    total = int(sum(written.values()))
    unknown_presets = sorted({t.get("preset") for t in (doc.get("delivery") or {}).get("targets") or [] if t.get("preset") in ("prores", "gif")})
    return {"deliverables": deliverables, "output_bytes": total, "unknown_presets": unknown_presets}


def check_space(rep: Report, job_dir: str, total: int, free_fn: Callable[[str], Optional[int]], field: str = "workspace") -> None:
    base = nearest_existing(job_dir)
    rep.estimates.update({"volume": base, "margin": ESTIMATE_MARGIN, "required_bytes": int(total * ESTIMATE_MARGIN)})
    free = free_fn(base)
    rep.estimates["free_bytes"] = free
    if free is None:
        rep.add("DISK_SPACE_UNKNOWN", WARNING, f"the free space on {base} could not be read", field)
    elif free < total * ESTIMATE_MARGIN:
        rep.add("DISK_SPACE_LOW", ERROR, f"{base} has {_bytes_h(free)} free; the job writes about {_bytes_h(total)} (x{ESTIMATE_MARGIN:g} margin: {_bytes_h(total * ESTIMATE_MARGIN)})",
                "estimates.required_bytes")


def check_output_dir(rep: Report, job_dir: str, field: str = "workspace") -> None:
    base = nearest_existing(job_dir)
    if not os.path.isdir(base):
        rep.add("OUTPUT_DIR_NOT_A_DIRECTORY", ERROR, f"{base} is not a directory; the job writes under {job_dir}", field)
        return
    why = probe_writable(base)
    if why is not None:
        rep.add("OUTPUT_DIR_UNWRITABLE", ERROR, f"{base} is not writable ({why}); the job writes under {job_dir}", field)


def check_existing_outputs(rep: Report, ops: List[Any], paths: Dict[str, str], job_dir: str) -> None:
    for op in ops:
        for out in op.outputs:
            p = paths.get(out)
            if p and not _within(p, job_dir) and os.path.lexists(p):
                size = os.path.getsize(p) if os.path.isfile(p) else None
                rep.add("OUTPUT_EXISTS", WARNING, f"{out}: {p} exists ({_bytes_h(size)}); replacing it is a destructive operation and needs its approval "
                        "(on the MCP server: the destructive scope and a confirm_token)", f"paths.{out}")


def run(doc: Dict[str, Any], *, ops: List[Any], paths: Dict[str, str], job_dir: str, subjects: List[Dict[str, Any]], profile_rules: Iterable[Any],
        presets: Optional[List[str]], platforms: Optional[List[str]], upload_limits: Dict[str, Any], tooling: List[Dict[str, Any]],
        free_fn: Callable[[str], Optional[int]] = disk_free, compile_error: Optional[str] = None) -> Dict[str, Any]:
    rep = Report()
    for p in tooling:
        rep.add(p["code"], p["severity"], p["message"], p["field"])
    if compile_error:
        rep.add("PLAN_COMPILE_FAILED", ERROR, f"the plan does not compile: {compile_error}", "plan")
    check_output_dir(rep, job_dir)
    check_existing_outputs(rep, ops, paths, job_dir)
    check_inputs(rep, doc, ops, paths, job_dir)
    check_media(rep, doc, profile_rules, subjects)
    w = estimate_writes(doc, ops, paths, subjects)
    rows = check_targets(rep, doc, subjects, paths, w["deliverables"], presets, platforms, upload_limits)
    rep.estimates.update({"deliverables": rows, "output_bytes": w["output_bytes"],
                          "basis": "deliverable: the source bitrate x the plan's expected duration; other media writes: the sum of their inputs"})
    for pr in w["unknown_presets"]:
        rep.unverified.append({"field": "estimates.output_bytes", "reason": f"the {pr} preset's bitrate is the engine's and not in this repository; its size is estimated at the source bitrate"})
    check_space(rep, job_dir, w["output_bytes"], free_fn)
    return rep.to_dict()


def run_offload(doc: Dict[str, Any], step: Dict[str, Any], plan_obs: Dict[str, Any], tooling: List[Dict[str, Any]],
                free_fn: Callable[[str], Optional[int]] = disk_free) -> Dict[str, Any]:
    """ADR-066's card offload: the card is there, every destination is writable and has room for the card's bytes x the margin."""
    rep = Report()
    for p in tooling:
        rep.add(p["code"], p["severity"], p["message"], p["field"])
    p = step.get("params") or {}
    src = str(p.get("source") or "")
    if not os.path.isdir(src):
        vol = missing_volume(src)
        rep.add("INPUT_VOLUME_MISSING" if vol else "INPUT_MISSING", ERROR, f"the card {src} " + (f"is on {vol}, which is not mounted" if vol else "is not there"), "plan.steps.offload.params.source")
    total = int(plan_obs.get("bytes_total") or 0)
    rep.estimates.update({"output_bytes": total, "margin": ESTIMATE_MARGIN, "required_bytes": int(total * ESTIMATE_MARGIN), "destinations": []})
    for n, dest in enumerate(p.get("destinations") or []):
        field = f"plan.steps.offload.params.destinations[{n}]"
        base = nearest_existing(str(dest))
        vol = missing_volume(str(dest))
        row = {"destination": dest, "volume": base, "free_bytes": None}
        rep.estimates["destinations"].append(row)
        if vol:
            rep.add("OUTPUT_VOLUME_MISSING", ERROR, f"destination {dest} is on {vol}, which is not mounted", field)
            continue
        if not os.path.isdir(base):
            rep.add("OUTPUT_DIR_NOT_A_DIRECTORY", ERROR, f"destination {dest}: {base} is not a directory", field)
            continue
        why = probe_writable(base)
        if why is not None:
            rep.add("OUTPUT_DIR_UNWRITABLE", ERROR, f"destination {dest}: {base} is not writable ({why})", field)
        free = free_fn(base)
        row["free_bytes"] = free
        if free is None:
            rep.add("DISK_SPACE_UNKNOWN", WARNING, f"the free space on {base} could not be read", field)
        elif free < total * ESTIMATE_MARGIN:
            rep.add("DISK_SPACE_LOW", ERROR, f"destination {dest}: {_bytes_h(free)} free, the card holds {_bytes_h(total)} (x{ESTIMATE_MARGIN:g}: {_bytes_h(total * ESTIMATE_MARGIN)})", field)
    return rep.to_dict()
