"""Orchestration service: the lifecycle glue used by the CLI (and later by a queue or UI).
Request → Requirements → Intent → Analysis → Inference → Decision → Plan → IR → Validate → Execute → QA → Report."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .agent import build_plan, decide, extract_requirements, infer, resolve_intent
from .agent.requirements import requirement_map
from .audit import build_provenance, write_audit
from .capabilities import CapabilityResolver
from .execution import Executor, compile_ir
from .jobs import Job, JobStore
from .media import MediaAnalyzer
from .models import Artifact, Request, now_iso
from .policy.rules import SYSTEM_CONSTRAINTS, Rule, resolve_rules
from .profiles import load_profile
from .project import ProjectIR, load_ir, save_ir, validate_ir
from .qa import run_qa
from .skills import default_registry
from .tools.ffmpeg_skill import FfmpegSkillAdapter
from .tools.ffmpeg_skill.adapter import PathPolicy
from .tools.ffmpeg_skill.locate import locate_ffmpeg_skill


class Service:
    def __init__(self, workspace: Optional[str] = None, ffmpeg_skill_dir: Optional[str] = None, adapter=None, caps: Optional[CapabilityResolver] = None):
        self.workspace = str(Path(workspace or os.environ.get("VIDEO_AGENT_WORKSPACE") or "./video-agent-work").resolve())
        self.skill_dir = ffmpeg_skill_dir or os.environ.get("VIDEO_AGENT_FFMPEG_SKILL_DIR")
        self.caps = caps or CapabilityResolver(self.skill_dir)
        self._adapter = adapter
        self.registry = default_registry()

    # ---- adapters
    def adapter(self, allowed_inputs: Optional[List[str]] = None):
        if self._adapter is not None:
            return self._adapter
        skill = locate_ffmpeg_skill(self.skill_dir)
        policy = PathPolicy(allowed_inputs or [], self.workspace) if allowed_inputs is not None else None
        return FfmpegSkillAdapter(skill, policy)

    def tool_versions(self) -> Dict[str, str]:
        caps = self.caps.resolve()
        return {"ffmpeg": caps["ffmpeg"].detail if caps["ffmpeg"].status == "AVAILABLE" else "missing",
                "ffmpeg-skill": str(caps["ffmpeg-skill"].evidence.get("version", "missing")), "video-agent": __version__}

    # ---- lifecycle
    def analyze(self, inputs: List[str], profile_name: str = "generic", request_text: str = "", user_requirements: Optional[Dict[str, Any]] = None, hash_sources: bool = True):
        profile = load_profile(profile_name)
        rules = resolve_rules(SYSTEM_CONSTRAINTS + profile.rules + _request_rules(user_requirements or {}))
        adapter = self.adapter([str(Path(p).resolve().parent) for p in inputs])
        # Phase 1 analyses whole files (silence + loudness over the full duration); a budgeted TARGETED strategy is Phase 2,
        # so the recorded strategy says what actually happened rather than what the profile asked for.
        analyzer = MediaAnalyzer(adapter, silence_threshold_db=float(rules.get("silence.threshold_db", -40)), strategy="FULL_ANALYSIS", hash_sources=hash_sources)
        analysis = analyzer.analyze(inputs)
        return profile, rules, analysis

    def plan(self, inputs: List[str], profile_name: str = "generic", request_text: str = "", user_requirements: Optional[Dict[str, Any]] = None,
             project_name: Optional[str] = None, hash_sources: bool = True) -> ProjectIR:
        bad = [k for k in (user_requirements or {}) if not k.startswith(REQUIREMENT_PREFIXES)]
        if bad:
            raise ValueError(f"unknown requirement key(s): {', '.join(bad)}; allowed prefixes: {', '.join(REQUIREMENT_PREFIXES)}")
        profile, rules, analysis = self.analyze(inputs, profile_name, request_text, user_requirements, hash_sources)
        caps = self.caps.resolve()
        request = Request(raw=request_text, args={"inputs": inputs, "profile": profile_name, "requirements": user_requirements or {}})
        reqs = extract_requirements(request, profile, rules)
        rm = requirement_map(reqs)
        intent = resolve_intent(reqs)
        target = rm.get("audio.loudness.target_lufs")
        inferences = infer(analysis, rules, target_lufs=float(target.value) if target else None, tolerance_lu=float(rules.get("audio.loudness.tolerance_lu", 2.0)))
        decisions = decide(reqs, intent, analysis, inferences, rules, caps, self.registry)
        plan = build_plan(decisions, analysis)
        ir = ProjectIR.new(project_name or Path(inputs[0]).stem, {"name": profile.name, "version": profile.version, "chain": profile.chain}, self.workspace)
        d = ir.doc
        d["request"] = request.to_dict()
        d["requirements"] = [r.to_dict() for r in reqs]
        d["source"] = {"agent_version": __version__, "tool_versions": self.tool_versions(), "generator": "video-agent plan"}
        d["assets"] = {a.id: a.to_dict() for a in analysis.assets}
        d["analysis"] = {"observations": [o.to_dict() for o in analysis.observations], "inferences": [i.to_dict() for i in inferences], "strategy": analysis.strategy,
                         "budget": {"max_processing_time": rules.get("analysis.budget.max_processing_time"), "requested_strategy": rules.get("analysis.strategy"), "enforced": False},
                         "warnings": analysis.warnings, "tool_calls": analysis.tool_calls}
        d["intent"] = intent.to_dict()
        d["constraints"] = [r.to_dict() for r in rules.all_rules if r.hard]
        d["policy"] = rules.to_dict()
        d["decisions"] = [x.to_dict() for x in decisions]
        d["plan"] = {"version": plan["version"], "steps": plan["steps"], "summary": plan["summary"]}
        d["timeline"] = analysis.timeline.to_dict()
        d["video"] = {"operations": plan["video_ops"]}
        d["audio"] = {"operations": plan["audio_ops"]}
        d["delivery"] = {"targets": plan["delivery"], "naming": profile.data.get("naming", "")}
        d["qa"]["thresholds"]["loudness_tolerance_lu"] = float(rules.get("audio.loudness.tolerance_lu", 2.0))
        d["execution"]["allowed_inputs"] = sorted({str(Path(p).resolve().parent) for p in inputs})
        d["execution"]["recovery_policy"]["max_attempts"] = int(rules.get("execution.recovery.max_attempts", 2))
        d["provenance"].update({"source_hashes": {a.id: a.hash for a in analysis.assets}, "profile_version": profile.version,
                                "skill_versions": {s.name: s.version for s in self.registry.all() if s.phase == 1}, "tool_versions": d["source"]["tool_versions"]})
        ir.finalize_hash()
        return ir

    def validate(self, ir: ProjectIR, check_paths: bool = True):
        return validate_ir(ir, self.caps.resolve(), check_paths=check_paths)

    def dry_run(self, ir: ProjectIR) -> Dict[str, Any]:
        job_dir = str(Path(self.workspace) / "jobs" / "<job_id>")
        ops, paths = compile_ir(ir, job_dir, ir.doc["source"]["tool_versions"].get("ffmpeg-skill", ""))
        adapter = self.adapter(ir.doc["execution"].get("allowed_inputs") or None)
        previews = []
        for op in ops:
            try:
                cmd = adapter.preview(op, paths)
            except Exception as exc:  # noqa: BLE001
                cmd = [f"<cannot build command: {exc}>"]
            previews.append({"op": op.id, "tool": op.tool, "kind": op.kind, "args": op.args, "inputs": [paths.get(i, i) for i in op.inputs], "outputs": [paths.get(o, o) for o in op.outputs], "command": cmd, "decisions": op.decision_ids})
        needed = sorted({c for step in ir.doc["plan"]["steps"] for c in self.registry.get(step["skill"]).required_capabilities}) if ir.doc["plan"]["steps"] else []
        caps = self.caps.resolve()
        total = sum((a.get("technical") or {}).get("duration") or 0 for a in ir.doc["assets"].values())
        reencodes = sum(1 for op in ops if op.tool in ("ffmpeg-skill/export",)) + sum(1 for op in ops if op.tool == "ffmpeg-skill/cut" and op.args.get("accurate"))
        return {"operations": previews, "required_capabilities": {c: caps[c].status if c in caps else "UNKNOWN" for c in needed},
                "expected_outputs": [paths[o] for op in ops for o in op.outputs], "risks": [d["subject"] + ": " + d["risk"] for d in ir.doc["decisions"] if d["risk"] != "LOW"],
                "warnings": self.validate(ir).warnings + ir.doc["analysis"].get("warnings", []), "pending_confirmations": [d["id"] for d in ir.pending_confirmations()], "blocked": [d["id"] for d in ir.blocked()],
                "estimate": {"source_seconds": round(total, 1), "full_reencodes": reencodes, "note": "processing time ≈ source duration × re-encodes on a laptop CPU (see ffmpeg-skill devices.md)"}}

    def render(self, ir: ProjectIR, ir_path: str, approve: Optional[List[str]] = None, timeout: Optional[float] = None, who: Optional[str] = None,
               resume: Optional[str] = None) -> Dict[str, Any]:
        """resume: a previous job id (or "last") whose completed operations may be reused. The new job stays a new job;
        reuse is decided per operation by the chained idempotency key and by the recorded output still being intact."""
        who = who or _default_who()
        store = JobStore(self.workspace)
        prior: Optional[Job] = None
        resume_note: Optional[Dict[str, Any]] = None
        if resume:
            prior = store.latest(ir_path) if resume == "last" else store.load(resume)
            if prior is None:
                raise FileNotFoundError(f"no previous job to resume for {ir_path}")
            resume_note = {"resumed_from": prior.id, "prior_state": prior.state, "prior_plan_hash": prior.plan_hash, "plan_hash": ir.plan_hash(),
                           "plan_changed": bool(prior.plan_hash) and prior.plan_hash != ir.plan_hash(), "candidate_ops": len(prior.completed_ops)}
        job = store.create()
        job.ir_path = ir_path
        job.plan_hash = ir.plan_hash()
        if prior is not None:
            job.resumed_from = prior.id
            job.completed_ops = dict(prior.completed_ops)
            ir.doc["execution"]["resume_from"] = prior.id
        job.transition("INGESTING", "render requested")
        job.transition("ANALYZING")
        job.transition("PLANNING")
        rep = self.validate(ir)
        if not rep.ok:
            job.transition("FAILED", "validation failed")
            store.save(job)
            return {"job": job.to_dict(), "validation": rep.to_dict(), "status": "FAILED"}
        if approve:
            approved = ir.approve(approve, who)
        else:
            approved = []
        if ir.blocked():
            job.transition("BLOCKED", "; ".join(d["reason"] for d in ir.blocked()))
            store.save(job)
            return {"job": job.to_dict(), "status": "BLOCKED", "blocked": ir.blocked(), "approved": approved}
        pending = ir.pending_confirmations()
        if pending:
            job.transition("WAITING_FOR_APPROVAL", f"{len(pending)} decision(s) need confirmation")
            store.save(job)
            save_ir(ir, ir_path)
            return {"job": job.to_dict(), "status": "WAITING_FOR_APPROVAL", "pending": pending, "approved": approved,
                    "hint": "re-run with --approve <id,...> or --approve all"}
        job.transition("EXECUTING")
        store.save(job)
        ops, paths = compile_ir(ir, str(job.dir), ir.doc["source"]["tool_versions"].get("ffmpeg-skill", ""))
        adapter = self.adapter(ir.doc["execution"].get("allowed_inputs") or [])
        ex = Executor(adapter, max_attempts=int(ir.doc["execution"]["recovery_policy"]["max_attempts"]), timeout=timeout, completed_keys=job.completed_ops)
        try:
            result = ex.run(ops, paths)
        finally:
            job.completed_ops = ex.completed
            store.save(job)  # whatever happens next, the job file reflects the operations that completed
        out: Dict[str, Any] = {"job": None, "status": result.status, "execution": result.to_dict(), "approved": approved, "paths": paths, "resume": resume_note}
        if result.status != "COMPLETED":
            job.transition(result.status if result.status in ("FAILED", "BLOCKED", "CANCELLED") else "FAILED", f"op {result.failed_op}")
        else:
            job.transition("QA")
            checks = {op.args["input"]: r for op in ops if op.tool == "ffmpeg-skill/check" for r in result.results if r.op_id == op.id}
            qa = run_qa(adapter, ir.doc, paths, result.results, sheet_dir=str(job.dir / "qa"), check_by_artifact=checks)
            out["qa"] = qa.to_dict()
            artifacts = []
            for asset_id in ir.doc["assets"]:
                for t in ir.doc["delivery"]["targets"]:
                    art_id = f"{asset_id}_delivery_{t['id']}"
                    if art_id in paths and os.path.exists(paths[art_id]):
                        from .media.analyzer import sha256_file
                        st = {i.status for i in qa.items if i.artifact == art_id}
                        art_qa = "FAIL" if "FAIL" in st else ("WARN" if "WARN" in st else "PASS")
                        a = Artifact(path=paths[art_id], type=t.get("artifact_type", "MASTER"), hash=sha256_file(paths[art_id]), source=[asset_id], generation=1,
                                     tool="ffmpeg-skill", tool_version=ir.doc["source"]["tool_versions"].get("ffmpeg-skill", ""), qa_status=art_qa,
                                     stage="candidate" if art_qa != "FAIL" else "working", id=art_id)
                        artifacts.append(a.to_dict())
            job.artifacts = artifacts
            out["artifacts"] = artifacts
            job.transition("COMPLETED" if qa.status != "FAIL" else "REVIEW", f"qa {qa.status}")
            if qa.status == "FAIL":
                out["status"] = "REVIEW"
        prov = build_provenance(ir.doc, ops, result.results, paths, result.recovery, out.get("qa", {}), who=who)
        prov["resume"] = resume_note
        prov["skipped"] = result.skipped
        prov["reused"] = result.reused
        ir.doc["provenance"]["runs"].append({"job_id": job.id, "at": now_iso(), "status": out["status"], "who": who, "plan_hash": job.plan_hash,
                                             "resumed_from": job.resumed_from, "skipped": result.skipped})
        ir.doc["provenance"]["recovery"] = result.recovery
        save_ir(ir, ir_path)
        save_ir(ir, str(job.dir / "ir.json"))
        write_audit(str(job.dir / "provenance.json"), prov)
        (job.dir / "report.json").write_text(json.dumps({k: v for k, v in out.items() if k != "job"}, indent=2, default=str) + "\n", encoding="utf-8")
        (job.dir / "report.md").write_text(render_report_md(ir.doc, out, job.id), encoding="utf-8")
        store.save(job)
        out["job"] = job.to_dict()
        out["report"] = str(job.dir / "report.md")
        return out

    def check(self, path: str, platform: str = "custom") -> Dict[str, Any]:
        adapter = self.adapter([str(Path(path).resolve().parent)])
        pr = adapter.measure("ffmpeg-skill/probe", {"inputs": [str(Path(path).resolve())]})
        ck = adapter.measure("ffmpeg-skill/check", {"input": str(Path(path).resolve()), "platform": platform})
        return {"probe": pr.data if pr.ok else {"error": pr.stderr_tail}, "check": ck.data if ck.data else {"error": ck.stderr_tail}}


REQUIREMENT_PREFIXES = ("edit.", "audio.", "silence.", "delivery.")


def _default_who() -> str:
    try:
        import getpass
        return f"user:{getpass.getuser()}"
    except Exception:  # noqa: BLE001
        return "user"


def _request_rules(user_requirements: Dict[str, Any]) -> List[Rule]:
    out = []
    for k, v in user_requirements.items():
        if k.startswith(("audio.", "silence.", "delivery.", "edit.")):
            out.append(Rule(f"request.{k}", "PREFERENCE", "REQUEST", k, v, "request"))
    return out


def render_report_md(doc: Dict[str, Any], out: Dict[str, Any], job_id: str) -> str:
    lines = [f"# Production report — {doc['project']['name']} (job {job_id})", "", f"Status: **{out['status']}**  profile: {doc['project']['profile']['name']} v{doc['project']['profile']['version']}", "",
             "## Plan", *[f"- {s}" for s in doc["plan"]["summary"]], "", "## Decisions"]
    for d in doc["decisions"]:
        lines.append(f"- [{d['approval']}/{d['risk']}/{d['status']}] **{d['subject']}** → {d['decision']}  \n  {d['reason']} (confidence {d['confidence']:.2f}, evidence {len(d['evidence'])})")
    qa = out.get("qa")
    if qa:
        lines += ["", f"## QA — {qa['status']}"]
        for i in qa["items"]:
            if i["status"] != "PASS":
                lines.append(f"- {i['status']} {i['layer']}/{i['name']}: {i['observed']} (expected {i['expected']}) {i.get('fix_hint', '')}")
        lines.append(f"- {sum(1 for i in qa['items'] if i['status'] == 'PASS')} checks passed, {len(qa['incidents'])} incident(s)")
        for inc in qa["incidents"]:
            lines.append(f"  - {inc['type']} [{inc['severity']}] {inc['possible_cause']} → {inc['recommended_action']}")
        for s in qa.get("sheets", []):
            lines.append(f"- Look: {s}")
    if out.get("artifacts"):
        lines += ["", "## Artifacts"]
        for a in out["artifacts"]:
            lines.append(f"- {a['type']} {a['path']} (stage {a['stage']}, qa {a['qa_status']}, sha256 {a['hash'][:12]}…)")
    ex = out.get("execution") or {}
    if out.get("resume"):
        r = out["resume"]
        lines += ["", "## Resume", f"- resumed from job {r['resumed_from']} ({r['prior_state']}), plan {'changed' if r['plan_changed'] else 'unchanged'}, reused {len(ex.get('skipped', []))} operation(s)"]
        for op_id, path in (ex.get("reused") or {}).items():
            lines.append(f"  - {op_id} ← {path}")
    if ex.get("recovery"):
        lines += ["", "## Recovery", *[f"- op {r['op']} attempt {r.get('attempt')}: {r['class']} → {r['action']} ({r['reason']})" for r in ex["recovery"]]]
    return "\n".join(lines) + "\n"
