"""Deadline mode (撮って出し; ADR-133, roadmap step 22): `plan --deadline 30m`. A plan that has to be delivered by a time puts the deadline
before quality. It estimates every compiled step from how fast this machine ran the same tools before, drops the optional steps that would
miss the deadline, and says so in the plan. When even the required steps will not fit, it says that at plan time, not at the deadline.

    deadline   "30m" | "1h30m" | "90s" | "45" (minutes) | "HH:MM" (the next time the local clock shows it)
    estimate   per compiled operation: seconds per media second of that tool on this machine (the speed book), × the programme's media seconds,
               where the media seconds are the summed duration of the plan's video and audio sources. A tool with no measured run here gets a
               stated default, and the plan says the estimate is a default
    optional   finishing that a digest can go without, dropped in this order until the estimate fits: colour (`color.*`), then graphics
               (titles, lower thirds, overlays; not the thumbnail, which is a separate deliverable). The cut, the audio, captions and the
               delivery encode are never dropped
    at risk    the required steps alone are estimated past the deadline: `deadline.status` is `at_risk`, a warning names the shortfall and
               the plan is still written. Nothing is cut short to make it fit

The speed book (`<workspace>/speed/book.json`, `video-agent/speed-book@1`) is written by `render`: every operation that ran (not one reused
from an earlier job) adds its wall seconds and the programme's media seconds under its tool, separately for a machine with and without a
hardware video encoder. The machine is named by a hash of its host name and platform, never the name itself.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

RULE = "deadline@1"
BOOK_SCHEMA = "video-agent/speed-book@1"
DROP_ORDER = ("color", "graphics")            # what a digest can go without, first dropped first
# a tool never run on this machine: a stated default by operation kind (an encode at about 4x real time; a measurement or check at 20x)
DEFAULT_SECONDS_PER_MEDIA_SECOND = {"transform": 0.25, "measure": 0.05, "qa": 0.05}
DEFAULT_OVERHEAD_SECONDS = 1.0               # process start, probe and file writes per operation
KEEP_RUNS = 20                               # runs kept per tool (the newest), so a faster or slower machine state takes over
_DUR_RE = re.compile(r"^(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?$")
_CLOCK_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


class DeadlineError(ValueError):
    pass


def parse_deadline(text: str, now: Optional[float] = None) -> float:
    """Seconds from now to the deadline. A bare number is minutes; HH:MM is the next time the local clock shows it."""
    t = str(text or "").strip().lower().replace(" ", "")
    if not t:
        raise DeadlineError("--deadline needs a duration (30m, 1h30m, 90s) or a clock time (HH:MM)")
    m = _CLOCK_RE.match(t)
    if m:
        now = time.time() if now is None else now
        lt = time.localtime(now)
        target = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, int(m.group(1)), int(m.group(2)), 0, 0, 0, -1))
        if target <= now:
            target = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday + 1, int(m.group(1)), int(m.group(2)), 0, 0, 0, -1))
        return float(target - now)
    if re.match(r"^\d+(?:\.\d+)?$", t):
        secs = float(t) * 60.0
    else:
        m = _DUR_RE.match(t)
        if not m or not any(m.groups()):
            raise DeadlineError(f"--deadline {text!r}: use a duration (30m, 1h30m, 90s, or minutes) or a clock time (HH:MM)")
        h, mi, s = (float(g) if g else 0.0 for g in m.groups())
        secs = h * 3600 + mi * 60 + s
    if secs <= 0:
        raise DeadlineError("--deadline must be in the future")
    return secs


def machine_key(hw_encoder: bool) -> str:
    ident = hashlib.sha256(f"{platform.node()}|{platform.system()}|{platform.machine()}".encode("utf-8")).hexdigest()[:12]
    return f"{ident}:{'hw' if hw_encoder else 'sw'}"


class SpeedBook:
    """How fast each tool ran on this machine: wall seconds per media second, from the last KEEP_RUNS runs."""

    def __init__(self, workspace: str):
        self.path = Path(workspace) / "speed" / "book.json"
        self.doc: Dict[str, Any] = {"schema": BOOK_SCHEMA, "machines": {}}
        if self.path.exists():
            try:
                doc = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(doc, dict) and doc.get("schema") == BOOK_SCHEMA and isinstance(doc.get("machines"), dict):
                    self.doc = doc
            except (OSError, ValueError):
                pass   # an unreadable book is an empty one; the next render writes it again

    def runs(self, machine: str, tool: str) -> List[Dict[str, float]]:
        return list(((self.doc["machines"].get(machine) or {}).get(tool) or []))

    def record(self, machine: str, tool: str, seconds: float, media_seconds: float) -> None:
        if not media_seconds or media_seconds <= 0 or seconds is None or seconds < 0:
            return
        tools = self.doc["machines"].setdefault(machine, {})
        tools[tool] = (list(tools.get(tool) or []) + [{"seconds": round(float(seconds), 3), "media_seconds": round(float(media_seconds), 3)}])[-KEEP_RUNS:]

    def rate(self, machine: str, tool: str) -> Optional[Tuple[float, float, int]]:
        """(overhead seconds, seconds per media second, runs) by least squares over the recorded runs; None without any."""
        runs = self.runs(machine, tool)
        if not runs:
            return None
        xs = [r["media_seconds"] for r in runs]
        ys = [r["seconds"] for r in runs]
        n = len(runs)
        mx, my = sum(xs) / n, sum(ys) / n
        var = sum((x - mx) ** 2 for x in xs)
        if n >= 2 and var > 1e-9:
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
            a = my - b * mx
            if b >= 0 and a >= 0:
                return a, b, n
        return 0.0, (sum(ys) / sum(xs)) if sum(xs) > 0 else DEFAULT_SECONDS_PER_MEDIA_SECOND["transform"], n   # one length seen: a straight rate

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.doc, fh, indent=1, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, str(self.path))
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


def media_seconds(doc: Dict[str, Any]) -> float:
    """The programme's media seconds: the summed duration of the plan's video and audio sources."""
    total = 0.0
    for a in (doc.get("assets") or {}).values():
        tech = a.get("technical") or {}
        if a.get("type") in ("IMAGE", "LOGO", "GRAPHIC", "CAPTION"):
            continue
        d = tech.get("duration")
        if isinstance(d, (int, float)) and d > 0:
            total += float(d)
    return round(total, 3)


def group_of(op_skill: str, op_tool: str, doc_groups: Dict[str, str]) -> str:
    return doc_groups.get(op_tool) or doc_groups.get(op_skill) or "required"


def _media_of(op: Any, programme_seconds: float) -> float:
    """ADR-134: the media seconds one operation encodes -- a chunk of a chunked render its own range, every other operation the programme."""
    m = getattr(op, "media_seconds", None)
    return float(m) if isinstance(m, (int, float)) and not isinstance(m, bool) and m >= 0 else programme_seconds


def estimate(ops: List[Any], seconds_of_media: float, book: SpeedBook, machine: str) -> List[Dict[str, Any]]:
    rows = []
    for op in ops:
        got = book.rate(machine, op.tool)
        if got is None:
            a, b, basis, n = DEFAULT_OVERHEAD_SECONDS, DEFAULT_SECONDS_PER_MEDIA_SECOND.get(op.kind, 0.25), "default", 0
        else:
            (a, b, n), basis = got, "measured"
        rows.append({"op_id": op.id, "tool": op.tool, "skill": op.skill, "outputs": list(op.outputs), "decision_ids": list(op.decision_ids),
                     "seconds": round(a + b * _media_of(op, seconds_of_media), 3), "basis": basis, "runs": n})
    return rows


def _groups_of(doc: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    """Which optional group each compiled operation belongs to: by the IR section its decisions come from."""
    color_dec = {i for op in ((doc.get("color") or {}).get("operations") or []) for i in (op.get("decision_ids") or [])}
    graphics_dec = {i for op in ((doc.get("graphics") or {}).get("operations") or []) if op.get("type") != "graphics.thumbnail" for i in (op.get("decision_ids") or [])}
    for r in rows:
        ids = set(r["decision_ids"])
        if r["skill"] == "color_grading" or "color" in r["tool"].split("/", 1)[0] or (ids and ids <= color_dec):
            r["group"] = "color"
        elif ids and ids <= graphics_dec:
            r["group"] = "graphics"
        else:
            r["group"] = "required"


def apply(ir: Any, deadline_text: str, compile_fn: Callable[[Any], List[Any]], workspace: str, hw_encoder: bool, now: Optional[float] = None) -> Dict[str, Any]:
    """Estimate, drop what must go, record `deadline` in the IR, and return it."""
    budget = parse_deadline(deadline_text, now)
    doc = ir.doc
    machine = machine_key(hw_encoder)
    book = SpeedBook(workspace)
    secs = media_seconds(doc)
    rows = estimate(compile_fn(ir), secs, book, machine)
    _groups_of(doc, rows)
    total = round(sum(r["seconds"] for r in rows), 3)
    dropped: List[Dict[str, Any]] = []
    for group in DROP_ORDER:
        if total <= budget:
            break
        mine = [r for r in rows if r["group"] == group]
        if not mine:
            continue
        cost = round(sum(r["seconds"] for r in mine), 3)
        section = doc.get(group) or {}
        removed = [op for op in (section.get("operations") or []) if group != "graphics" or op.get("type") != "graphics.thumbnail"]
        section["operations"] = [op for op in (section.get("operations") or []) if op not in removed]
        dec_ids = sorted({i for op in removed for i in (op.get("decision_ids") or [])})
        total = round(total - cost, 3)
        dropped.append({"group": group, "operations": [op.get("type") for op in removed], "decision_ids": dec_ids, "estimated_seconds": cost,
                        "reason": f"{group} is estimated at {cost:.0f} s; the plan with it needs {total + cost:.0f} s of the {budget:.0f} s to the deadline"})
        rows = [r for r in rows if r["group"] != group]
    status = "fits" if total <= budget else "at_risk"
    record = {"rule": RULE, "deadline": str(deadline_text), "budget_seconds": round(budget, 3), "media_seconds": secs, "machine": machine,
              "hw_encoder": bool(hw_encoder), "estimated_seconds": total, "status": status, "dropped": dropped,
              "steps": [{k: r[k] for k in ("tool", "skill", "group", "seconds", "basis", "runs")} for r in rows],
              "defaults": sorted({r["tool"] for r in rows if r["basis"] == "default"})}
    if status == "at_risk":
        record["warning"] = (f"the required steps alone are estimated at {total:.0f} s, {total - budget:.0f} s past the deadline ({budget:.0f} s from now); "
                             "the plan is kept, nothing more is dropped")
    doc["deadline"] = record
    summary = (doc.get("plan") or {}).get("summary")
    if isinstance(summary, list):
        summary.append(f"deadline {deadline_text}: estimated {total:.0f} s of {budget:.0f} s ({status})"
                       + ("".join(f"; dropped {d['group']} ({d['estimated_seconds']:.0f} s)" for d in dropped)))
    return record


def record_run(workspace: str, hw_encoder: bool, doc: Dict[str, Any], ops: List[Any], results: List[Any], reused: List[str]) -> int:
    """After a render: each operation that ran adds its wall seconds under its tool. Returns how many were recorded."""
    secs = media_seconds(doc)
    if secs <= 0:
        return 0
    book = SpeedBook(workspace)
    machine = machine_key(hw_encoder)
    by_op = {op.id: op for op in ops}
    n = 0
    for r in results:
        op = by_op.get(getattr(r, "op_id", None))
        if op is None or not getattr(r, "ok", False) or getattr(r, "dry_run", False) or op.id in (reused or []):
            continue
        book.record(machine, op.tool, float(r.seconds or 0.0), _media_of(op, secs))
        n += 1
    if n:
        book.save()
    return n
