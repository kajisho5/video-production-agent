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


def validate_ir(ir: ProjectIR, caps: Optional[Dict[str, Any]] = None, check_paths: bool = True, registry=None, supports=None) -> ValidationReport:
    """registry: SkillRegistry (plan steps must cite implemented skills and their declared tools); supports: callable(tool) -> bool
    from the tool router (the named tool must be executable here). Both optional so pure schema checks stay cheap."""
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
    rejected = {x["id"] for x in d["decisions"] if x["status"] == "REJECTED"}
    for op in d["video"]["operations"] + d["audio"]["operations"] + d["delivery"]["targets"]:
        for did in op.get("decision_ids") or []:
            if did in rejected:
                rep.errors.append(f"{op.get('type') or 'delivery.' + op.get('id', '?')} cites REJECTED decision {did}; re-plan (revise) before rendering")
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
    # plan steps: Skill → Tool consistency (the compiler takes tools from the steps, so this is the execution contract)
    if registry is not None:
        known = set(registry.names())
        for s in d["plan"]["steps"]:
            if s["skill"] not in known:
                rep.errors.append(f"plan step {s['id']} cites unknown skill {s['skill']}")
                continue
            spec = registry.get(s["skill"])
            if not spec.implemented:
                rep.errors.append(f"plan step {s['id']} cites skill {s['skill']} which is declared for phase {spec.phase} and not implemented")
            if not s.get("tool"):
                rep.errors.append(f"plan step {s['id']} ({s['skill']}) has no selected tool")
            elif s["tool"] not in spec.tools:
                rep.errors.append(f"plan step {s['id']}: tool {s['tool']} is not a declared tool of skill {s['skill']} ({', '.join(spec.tools)})")
            elif supports is not None and not supports(s["tool"]):
                rep.errors.append(f"plan step {s['id']}: no registered adapter supports {s['tool']}")
            elif registry.packages() and registry.tool(s["tool"]) is None:
                rep.errors.append(f"plan step {s['id']}: tool {s['tool']} is not declared by any registered skill package")
    # every executable operation must have a step (otherwise the compiler cannot know its tool)
    step_keys = {(s["skill"], (s.get("params") or {}).get("asset") or (s.get("params") or {}).get("target")) for s in d["plan"]["steps"]}
    for op in d["video"]["operations"]:
        if op["type"] == "video.trim" and ("silence_cleanup", op["asset"]) not in step_keys:
            rep.errors.append(f"video.trim on {op['asset']} has no plan step")
    for op in d["audio"]["operations"]:
        if op["type"] == "audio.loudness" and ("loudness_normalization", op["asset"]) not in step_keys:
            rep.errors.append(f"audio.loudness on {op['asset']} has no plan step")
    for t in d["delivery"]["targets"]:
        if t.get("preset") and ("delivery_export", t["id"]) not in step_keys:
            rep.errors.append(f"delivery target {t['id']} has no export step")
    # observations are measurements: their source is a tool id + version, never an AI provider; AI output is AI_GENERATED
    for o in d["analysis"].get("observations") or []:
        src = str(o.get("source") or "")
        if "@" not in src or src.startswith("ai"):
            rep.errors.append(f"observation {o.get('id')} has no tool source ({src!r}); only tool measurements may be OBSERVED")
    for i in d["analysis"].get("inferences") or []:
        if str(i.get("kind", "")).startswith("ai_recommendation:") and i.get("provenance") != "AI_GENERATED":
            rep.errors.append(f"inference {i.get('id')} from an AI provider must carry provenance AI_GENERATED")
        if not i.get("evidence"):
            rep.errors.append(f"inference {i.get('id')} cites no evidence")
    # paths
    if check_paths:
        for a in assets.values():
            if not Path(a["path"]).exists():
                rep.errors.append(f"asset {a['id']} path missing: {a['path']}")
    # capabilities: a plan must not silently depend on unavailable capabilities
    if caps is not None:
        # skill-level needs come from the registry (production skill + owning package); preset-level encoder needs below
        needed = set()
        if registry is not None:
            for st in d["plan"]["steps"]:
                if st["skill"] in registry.names():
                    needed.update(registry.get(st["skill"]).required_capabilities)
                pkg = registry.package(str(st.get("tool") or "").split("/", 1)[0])
                if pkg:
                    needed.update(pkg.capabilities)
                    ts = registry.tool(st["tool"]) if st.get("tool") else None
                    if ts:
                        needed.update(ts.required_capabilities)
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
