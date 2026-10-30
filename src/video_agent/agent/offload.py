"""Card offload through ingest-skill (ADR-066, roadmap step 9: the media_offload adapter; ingest-skill roadmap step 6).

A card folder (a camera card, a recorder's folder) is given to `video-agent plan` as its only input, with the destinations named
explicitly. Offloading copies data onto the user's drives, so nothing is copied by default and nothing is copied at plan time:

- **Requirements** (only these keys; anything else is refused): `ingest.dest` (required: one or more existing directories, a JSON list,
  a comma-separated string or `--set ingest.dest=` repeated), `ingest.project` (a sub-folder under each destination), `ingest.hash`
  (sha256 | xxh64), `ingest.name_by` (folder | reel), `ingest.reel` (the card's reel id, overriding what the card says).
- **Plan**: ingest-skill's own `offload --dry-run` (names and sizes only, nothing written) and `inventory` of the card (reads only)
  become OBSERVED observations on the card; one TRANSFORM `ingest.offload` decision, always CONFIRM (never AUTO, no policy key can
  lower it), and one step `media_offload` → `ingest/offload`. Without ingest-skill the step has no tool, a BLOCK decision names the
  missing package, and the plan's ADR-051 handoff says how to install it.
- **Run** (`render` after approval): the copy, then the inventory of the first destination's copy; every recording becomes a new asset
  (one per member file of a spanned recording), each with an OBSERVED `ingest_clip` observation carrying the clip, its members in order,
  the reel / card id, the hashes, the camera's sidecar facts and the provenance (each destination's manifest and ASC MHL generation,
  ingest-skill's version, the job). The agent's own analysis (probe / silence / loudness) then runs on the verified copies when the
  media engine can read them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import Observation
from ..tools.ingest.adapter import NAME_BY, NAME_RE, TOOL_INVENTORY, TOOL_OFFLOAD

PREFIX = "ingest."
KEY_DEST, KEY_PROJECT, KEY_HASH, KEY_NAME_BY, KEY_REEL = "ingest.dest", "ingest.project", "ingest.hash", "ingest.name_by", "ingest.reel"
REQUIREMENT_KEYS = (KEY_DEST, KEY_PROJECT, KEY_HASH, KEY_NAME_BY, KEY_REEL)
HASHES = ("sha256", "xxh64")
SKILL = "media_offload"
SUBJECT = "ingest.offload"
STEP_ID = "step_media_offload"
OP_ID = "op_media_offload"
PROJECT_KIND = "offload"
PLAN_OBS_KIND = "ingest_offload_plan"
CARD_OBS_KIND = "ingest_card_inventory"
CLIP_OBS_KIND = "ingest_clip"
STEP_PARAMS = ("source", "destinations", "project", "hash", "name_by", "reel", "roots")


class OffloadRequirementError(ValueError):
    """A card offload that cannot be planned as asked (refused before ingest-skill is called, nothing written)."""


@dataclass
class OffloadRequest:
    source: str
    destinations: List[str]
    project: Optional[str] = None
    hash: str = "sha256"
    name_by: str = "folder"
    reel: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def args(self, dry_run: bool = False) -> Dict[str, Any]:
        """The typed `ingest/offload` arguments (tools/ingest/adapter.py OFFLOAD_ARGS)."""
        out: Dict[str, Any] = {"source": self.source, "destinations": list(self.destinations), "hash": self.hash, "name_by": self.name_by, "dry_run": dry_run}
        if self.project:
            out["project"] = self.project
        if self.reel:
            out["reel"] = self.reel
        return out


def is_card_input(path: str) -> bool:
    """A card / recorder folder input: an existing directory (a media file is planned as before)."""
    return bool(path) and os.path.isdir(path)


def has_offload_keys(reqs: Dict[str, Any]) -> bool:
    return any(str(k).startswith(PREFIX) for k in reqs or {})


def _destinations(v: Any) -> List[str]:
    if isinstance(v, str):
        items = [x.strip() for x in v.split(",")]
    elif isinstance(v, (list, tuple)):
        items = [x.strip() if isinstance(x, str) else x for x in v]
    else:
        raise OffloadRequirementError(f"{KEY_DEST} must be a directory, a comma-separated list or a JSON list of directories, got {v!r}")
    if not items or any(not isinstance(x, str) or not x for x in items):
        raise OffloadRequirementError(f"{KEY_DEST} has an empty or non-text entry: {v!r}")
    return items


def parse_offload_requirements(source: str, reqs: Dict[str, Any]) -> OffloadRequest:
    """The offload a card plan asks for, or OffloadRequirementError naming what is wrong. Every destination must already exist as a
    directory (a mount point of a drive that is not mounted is never created on the system disk), be given once, and neither contain
    the card nor sit inside it. Two destinations on one volume, or a single destination, are planned with a warning: not a backup."""
    reqs = dict(reqs or {})
    unknown = sorted(k for k in reqs if k not in REQUIREMENT_KEYS)
    if unknown:
        raise OffloadRequirementError(f"an offload plan takes only {', '.join(REQUIREMENT_KEYS)}; got {', '.join(unknown)} "
                                      "(plan the edit on the verified copies once the offload has run)")
    if not is_card_input(source):
        raise OffloadRequirementError(f"{source} is not a directory: an offload copies a card / recorder folder")
    if reqs.get(KEY_DEST) in (None, "", []):
        raise OffloadRequirementError(f"an offload copies the card onto your drives, so nothing is copied unless you name every destination: "
                                      f"--set {KEY_DEST}=/Volumes/SSD1/Footage --set {KEY_DEST}=/Volumes/SSD2/Footage (or a comma-separated / JSON list)")
    src = Path(source).resolve()
    dests: List[str] = []
    for raw in _destinations(reqs[KEY_DEST]):
        p = Path(os.path.expanduser(raw))
        if not p.is_absolute():
            raise OffloadRequirementError(f"destination {raw!r} is not an absolute path")
        if not p.is_dir():
            raise OffloadRequirementError(f"destination {raw} does not exist as a directory (is the drive mounted?); it is never created")
        r = p.resolve()
        if str(r) in dests:
            raise OffloadRequirementError(f"destination {raw} is given twice; a second copy must be on a different path (ideally a different drive)")
        if r == src or src in r.parents or r in src.parents:
            raise OffloadRequirementError(f"destination {raw} and the card {source} contain one another")
        dests.append(str(r))
    h = reqs.get(KEY_HASH, "sha256")
    if h not in HASHES:
        raise OffloadRequirementError(f"{KEY_HASH} must be one of {', '.join(HASHES)}, got {h!r}")
    nb = reqs.get(KEY_NAME_BY, "folder")
    if nb not in NAME_BY:
        raise OffloadRequirementError(f"{KEY_NAME_BY} must be one of {', '.join(NAME_BY)}, got {nb!r}")
    for k in (KEY_PROJECT, KEY_REEL):
        v = reqs.get(k)
        if v is not None and (not isinstance(v, str) or not NAME_RE.match(v)):
            raise OffloadRequirementError(f"{k} must be one folder name (1-64 letters, digits, '.', '_' or '-', starting with a letter or digit), got {v!r}")
    out = OffloadRequest(str(src), dests, reqs.get(KEY_PROJECT), h, nb, reqs.get(KEY_REEL))
    if len(dests) == 1:
        out.warnings.append(f"one destination ({dests[0]}): there is no second copy; a backup needs a second drive")
    devs: Dict[int, str] = {}
    for d in dests:
        dev = os.stat(d).st_dev
        if dev in devs:
            out.warnings.append(f"{devs[dev]} and {d} are on the same volume: two copies on one drive are not a backup")
        devs.setdefault(dev, d)
    return out


def plan_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """What the dry run says will happen (OBSERVED, nothing written): file count, bytes, the reel / card id and each copy's folder."""
    return {"files": summary.get("files"), "bytes_total": summary.get("bytes_total"), "skipped": summary.get("skipped"), "hash": summary.get("hash"),
            "card": summary.get("card"), "roots": [d.get("root") for d in summary.get("destinations") or []], "source": summary.get("source")}


def card_summary(inv: Dict[str, Any]) -> Dict[str, Any]:
    """The card's inventory as planned (OBSERVED, read on the card): structures, reel, the recordings with their members."""
    return {"card": inv.get("card"), "clip_count": inv.get("clip_count"), "file_count": inv.get("file_count"), "spanned_count": inv.get("spanned_count"),
            "total_duration_s": inv.get("total_duration_s"), "warnings": list(inv.get("warnings") or []), "probe": inv.get("probe"),
            "clips": [{"clip_id": c.get("clip_id"), "structure": c.get("structure"), "spanned": c.get("spanned"), "incomplete": c.get("incomplete"),
                       "duration_s": c.get("duration_s"), "members": [{"path": m.get("path"), "size": m.get("size")} for m in c.get("members") or []]}
                      for c in inv.get("clips") or [] if isinstance(c, dict)]}


FACT_KEYS = ("probed", "container", "video_codec", "width", "height", "frame_rate", "audio_codec", "audio_channels", "sample_rate", "timecode", "creation_time")


def clip_observations(inv: Dict[str, Any], report: Dict[str, Any], summary: Dict[str, Any], asset_of: Dict[str, str], version: str,
                      provenance: Dict[str, Any]) -> List[Observation]:
    """One OBSERVED `ingest_clip` observation per member file of every recording, on the asset that file became (asset_of: relative
    member path → asset id). The data is ingest-skill's, as it reported it: the clip, its members in recording order with their sizes
    and hashes, the reel / card id and its source, the card structures, the camera sidecars, proxies and related files, the facts
    ingest-skill measured (unknown stays None), and where every copy is (each destination's file, manifest and ASC MHL generation)."""
    card = inv.get("card") or {}
    dests = summary.get("destinations") or []
    out: List[Observation] = []
    for c in inv.get("clips") or []:
        members = [m for m in c.get("members") or [] if isinstance(m, dict)]
        for i, m in enumerate(members):
            rel = str(m.get("path"))
            aid = asset_of.get(rel)
            if aid is None:
                continue
            h = m.get("hash") if isinstance(m.get("hash"), dict) else None
            data = {"clip_id": c.get("clip_id"), "structure": c.get("structure"), "spanned": bool(c.get("spanned")), "incomplete": bool(c.get("incomplete")),
                    "member_index": i, "member_count": len(members), "path": rel,
                    "members": [{"path": x.get("path"), "size": x.get("size"), "hash": x.get("hash")} for x in members],
                    "size": m.get("size"), "hash": h, "member_duration_s": m.get("duration_s"), "duration_s": c.get("duration_s"),
                    "facts": {k: c.get(k) for k in FACT_KEYS}, "sidecars": c.get("sidecars") or [], "proxies": c.get("proxies") or [], "related": c.get("related") or [],
                    "metadata": c.get("metadata") or {}, "reel": card.get("reel"), "structures": card.get("structures") or [],
                    "copies": [{"root": d.get("root"), "file": str(Path(str(d.get("root"))) / rel), "manifest": d.get("manifest"), "mhl": d.get("mhl")} for d in dests],
                    "card_source": report.get("source"), "hash_algorithm": report.get("hash"), "inventory_warnings": list(inv.get("warnings") or []),
                    "ingest_version": version, **provenance}
            out.append(Observation(kind=CLIP_OBS_KIND, asset_id=aid, source=f"{TOOL_INVENTORY}@{version or 'unknown'}", data=data, skill="ingest", skill_version=version,
                                   tool=TOOL_INVENTORY, external_id=str(c.get("clip_id") or ""), fingerprint=str((h or {}).get("value") or ""),
                                   parameters={"target": dests[0].get("root") if dests else None, "offload_tool": TOOL_OFFLOAD}))
    return out
