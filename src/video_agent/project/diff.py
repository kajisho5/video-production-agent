"""PlanDiff: machine-readable difference between two plan versions of the same project, plus a human summary.

Decisions are keyed by (subject, asset) so a re-planned decision with a fresh id still matches its predecessor;
video/audio operations by (type, asset); delivery targets by id."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _dkey(d: Dict[str, Any]) -> str:
    return f"{d['subject']}@{(d.get('params') or {}).get('asset_id', '-')}"


def _okey(op: Dict[str, Any]) -> str:
    return f"{op['type']}@{op['asset']}"


def _op_view(op: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in op.items() if k != "decision_ids"}


def _decision_view(d: Dict[str, Any]) -> Dict[str, Any]:
    # status is excluded: approvals are review state, not plan content (a re-plan resets them; rejections are reported from history)
    return {"decision": d["decision"], "approval": d["approval"], "risk": d["risk"], "params": {k: v for k, v in (d.get("params") or {}).items() if k != "asset_id"}}


def _diff_map(old: Dict[str, Any], new: Dict[str, Any], view) -> Dict[str, Any]:
    added = {k: view(v) for k, v in new.items() if k not in old}
    removed = {k: view(v) for k, v in old.items() if k not in new}
    changed = {}
    for k in old.keys() & new.keys():
        a, b = view(old[k]), view(new[k])
        if a != b:
            changed[k] = {"before": a, "after": b}
    return {"added": added, "removed": removed, "changed": changed}


def plan_diff(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    od = {_dkey(d): d for d in old["decisions"]}
    nd = {_dkey(d): d for d in new["decisions"]}
    ov = {_okey(o): o for o in old["video"]["operations"]}
    nv = {_okey(o): o for o in new["video"]["operations"]}
    oa = {_okey(o): o for o in old["audio"]["operations"]}
    na = {_okey(o): o for o in new["audio"]["operations"]}
    ot = {t["id"]: t for t in old["delivery"]["targets"]}
    nt = {t["id"]: t for t in new["delivery"]["targets"]}
    fin = {}
    for sec in ("captions", "graphics", "color"):   # finishing sections (ADR-031): the same key as the video / audio operations
        o_ = {_okey(o): o for o in (old.get(sec) or {}).get("operations") or []}
        n_ = {_okey(o): o for o in (new.get(sec) or {}).get("operations") or []}
        fin[sec] = _diff_map(o_, n_, _op_view)
    oq = {"qc": (old.get("qa") or {}).get("qc")} if (old.get("qa") or {}).get("qc", {}).get("enabled") else {}
    nq = {"qc": (new.get("qa") or {}).get("qc")} if (new.get("qa") or {}).get("qc", {}).get("enabled") else {}
    out: Dict[str, Any] = {
        "from_version": old["plan"]["version"], "to_version": new["plan"]["version"],
        "decisions": _diff_map(od, nd, _decision_view),
        "video": _diff_map(ov, nv, _op_view), "audio": _diff_map(oa, na, _op_view),
        "delivery": _diff_map(ot, nt, lambda t: {k: v for k, v in t.items() if k != "decision_ids"}),
        **fin, "qc": _diff_map(oq, nq, lambda q: {k: v for k, v in q.items() if k != "decision_ids"}),
    }
    out["summary"] = summarize(out)
    out["empty"] = not any(sec[k] for sec in (out["decisions"], out["video"], out["audio"], out["delivery"], out["captions"], out["graphics"], out["color"], out["qc"]) for k in ("added", "removed", "changed"))
    return out


def _fmt_keep(keep: List[List[float]]) -> str:
    return ",".join(f"{s:.3f}-{e:.3f}" for s, e in keep)


def _fmt_trim(op: Dict[str, Any]) -> str:
    return _fmt_keep(op.get("keep", [])) + (" (frame-accurate)" if op.get("accurate") else "")


def summarize(diff: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for k, v in diff["video"]["removed"].items():
        lines.append(f"VIDEO {k}: trim {_fmt_trim(v)} → removed")
    for k, v in diff["video"]["added"].items():
        lines.append(f"VIDEO {k}: none → trim {_fmt_trim(v)}")
    for k, v in diff["video"]["changed"].items():
        lines.append(f"VIDEO {k}: trim {_fmt_trim(v['before'])} → {_fmt_trim(v['after'])}")
    for k, v in diff["audio"]["removed"].items():
        lines.append(f"AUDIO {k}: {v.get('target_lufs')} LUFS → removed")
    for k, v in diff["audio"]["added"].items():
        lines.append(f"AUDIO {k}: none → {v.get('target_lufs')} LUFS / {v.get('true_peak')} dBTP")
    for k, v in diff["audio"]["changed"].items():
        lines.append(f"AUDIO {k}: {v['before'].get('target_lufs')} → {v['after'].get('target_lufs')} LUFS, {v['before'].get('true_peak')} → {v['after'].get('true_peak')} dBTP")
    for k, v in diff["delivery"]["removed"].items():
        lines.append(f"DELIVERY {k}: {v.get('preset')} → removed")
    for k, v in diff["delivery"]["added"].items():
        lines.append(f"DELIVERY {k}: none → {v.get('preset')} ({v.get('platform')})")
    for k, v in diff["delivery"]["changed"].items():
        lines.append(f"DELIVERY {k}: {v['before'].get('preset')}/{v['before'].get('platform')} → {v['after'].get('preset')}/{v['after'].get('platform')}")
    for sec in ("captions", "graphics", "color"):   # finishing operations (ADR-031): type@subject, listed by presence
        for k in (diff.get(sec) or {}).get("removed", {}):
            lines.append(f"{sec.upper()} {k}: removed")
        for k in (diff.get(sec) or {}).get("added", {}):
            lines.append(f"{sec.upper()} {k}: added")
        for k in (diff.get(sec) or {}).get("changed", {}):
            lines.append(f"{sec.upper()} {k}: parameters changed")
    for k in (diff.get("qc") or {}).get("removed", {}):
        lines.append("QC gate: removed")
    for k in (diff.get("qc") or {}).get("added", {}):
        lines.append("QC gate: added")
    for k in (diff.get("qc") or {}).get("changed", {}):
        lines.append("QC gate: rules changed")
    for k, v in diff["decisions"]["removed"].items():
        lines.append(f"DECISION {k}: {v['decision']} → dropped")
    for k, v in diff["decisions"]["added"].items():
        lines.append(f"DECISION {k}: new → {v['decision']} [{v['approval']}]")
    for k, v in diff["decisions"]["changed"].items():
        b, a = v["before"], v["after"]
        what = []
        if b["decision"] != a["decision"]:
            what.append(f"{b['decision']} → {a['decision']}")
        if b["approval"] != a["approval"]:
            what.append(f"approval {b['approval']} → {a['approval']}")
        lines.append(f"DECISION {k}: " + "; ".join(what or ["params changed"]))
    return lines
