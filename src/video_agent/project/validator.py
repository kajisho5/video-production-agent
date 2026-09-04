"""IR validation: JSON Schema + semantic rules + capability availability."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ir import ProjectIR

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "project.schema.json"


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def load_schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_ir(ir: ProjectIR, caps: Optional[Dict[str, Any]] = None, check_paths: bool = True) -> ValidationReport:
    rep = ValidationReport()
    try:
        import jsonschema
        validator = jsonschema.Draft202012Validator(load_schema())
        for err in sorted(validator.iter_errors(ir.doc), key=lambda e: list(e.absolute_path)):
            path = "/".join(str(p) for p in err.absolute_path) or "<root>"
            rep.errors.append(f"schema: {path}: {err.message}")
    except ImportError:
        rep.warnings.append("jsonschema not installed; schema validation skipped")
    if rep.errors:
        return rep
    d = ir.doc
    assets = d["assets"]
    ids = {x["id"] for x in d["decisions"]}
    # semantic: operations reference known assets and decisions, ranges inside durations
    for op in d["video"]["operations"] + d["audio"]["operations"]:
        if op["asset"] not in assets:
            rep.errors.append(f"operation {op['type']} references unknown asset {op['asset']}")
            continue
        dur = (assets[op["asset"]].get("technical") or {}).get("duration") or 0.0
        for k in op.get("keep") or []:
            if k[1] <= k[0]:
                rep.errors.append(f"video.trim keep range {k} is empty")
            if dur and k[1] > dur + 0.01:
                rep.errors.append(f"video.trim keep range {k} exceeds asset duration {dur:.3f}")
        for did in op.get("decision_ids") or []:
            if did not in ids:
                rep.errors.append(f"operation {op['type']} cites unknown decision {did}")
    # inferences must cite evidence that exists
    obs_ids = {o["id"] for o in d["analysis"]["observations"]} | {e["id"] for e in d["timeline"]["events"]} | {i["id"] for i in d["analysis"]["inferences"]}
    for inf in d["analysis"]["inferences"]:
        for ev in inf["evidence"]:
            if ev not in obs_ids:
                rep.errors.append(f"inference {inf['id']} cites missing evidence {ev}")
    # decisions: BLOCK is never executable; CONFIRM must be approved before render (reported as warning here)
    for dec in d["decisions"]:
        if dec["approval"] == "BLOCK":
            rep.warnings.append(f"decision {dec['id']} ({dec['subject']}) is BLOCKED: {dec['reason']}")
        elif dec["approval"] == "CONFIRM" and dec["status"] == "PROPOSED":
            rep.warnings.append(f"decision {dec['id']} ({dec['subject']}) needs confirmation before execution")
    # delivery presets / platforms must be names the tool knows (catalog-level check, no numbers duplicated)
    known_presets = {"youtube", "youtube4k", "reels", "x", "prores", "h265", "gif"}
    known_platforms = {"youtube", "shorts", "reels", "tiktok", "x", "linkedin", "broadcast", "podcast", "custom"}
    for t in d["delivery"]["targets"]:
        if t.get("preset") and t["preset"] not in known_presets:
            rep.errors.append(f"delivery target {t['id']}: unknown preset {t['preset']}")
        if t.get("platform") not in known_platforms:
            rep.errors.append(f"delivery target {t['id']}: unknown platform {t.get('platform')}")
    # paths
    if check_paths:
        for a in assets.values():
            if not Path(a["path"]).exists():
                rep.errors.append(f"asset {a['id']} path missing: {a['path']}")
    # capabilities: a plan must not silently depend on unavailable capabilities
    if caps is not None:
        needed = {"ffmpeg", "ffprobe", "ffmpeg-skill"}
        if d["video"]["operations"] or any(t.get("preset") for t in d["delivery"]["targets"]):
            needed.add("encoder:libx264")
        if d["audio"]["operations"]:
            needed.add("filter:loudnorm")
        if any(t.get("preset") == "prores" for t in d["delivery"]["targets"]):
            needed.add("encoder:prores_ks")
        if any(t.get("preset") == "h265" for t in d["delivery"]["targets"]):
            needed.add("encoder:libx265")
        for n in sorted(needed):
            st = getattr(caps.get(n), "status", None) or (caps.get(n) or {}).get("status") if isinstance(caps.get(n), dict) else getattr(caps.get(n), "status", "UNKNOWN")
            if st == "MISSING":
                rep.errors.append(f"required capability MISSING: {n}")
            elif st in ("UNKNOWN", "DEGRADED"):
                rep.warnings.append(f"capability {n} is {st}")
    return rep
