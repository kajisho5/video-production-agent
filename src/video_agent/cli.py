"""video-agent CLI (MASTER_SPEC §46): analyze, plan, validate, render [--dry-run], check, doctor, explain."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .capabilities import CapabilityResolver
from .project import load_ir, save_ir
from .service import Service


def _print(obj: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    elif isinstance(obj, str):
        print(obj)
    else:
        print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _kv(pairs: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set expects key=value, got {p}")
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def cmd_doctor(args, svc: Service) -> int:
    caps = svc.caps.resolve(refresh=True)
    if args.json:
        _print({k: v.to_dict() for k, v in caps.items()}, True)
    else:
        width = max(len(k) for k in caps)
        for k, c in caps.items():
            print(f"{c.status:9s} {k:{width}s}  {c.detail}")
    hard = [k for k in ("python", "ffmpeg", "ffprobe", "ffmpeg-skill") if caps[k].status == "MISSING"]
    return 1 if hard else 0


def cmd_skills(args, svc: Service) -> int:
    if getattr(args, "check_provides", False):
        findings = svc.check_provides()
        if args.json:
            _print({"findings": [f.to_dict() for f in findings]}, True)
            return 0
        width = max((len(f.status) for f in findings), default=len("PROVIDES_MISMATCH"))
        print("Provides/Capability consumption diagnostic (read-only; never affects tool selection):")
        for f in findings:
            print(f"  {f.status:{width}s}  {f.skill_id:16s} {f.capability_id or '(no id)':32s} {f.tool_id or '(no tool id)':28s}  {f.detail}")
        return 0
    rows = svc.skills()
    pkgs = svc.packages()
    if args.json:
        _print({"packages": pkgs, "skills": rows}, True)
        return 0
    print("Skill packages (implemented in this codebase):")
    for pk in pkgs:
        state = "AVAILABLE" if pk["available"] else "UNAVAILABLE"
        print(f"  {state:11s} {pk['skill_id']}  v{pk['version']}  {pk['repository'] or ''}  {pk['role']}")
        print(f"              tools {len(pk['usable_tools'])}/{len(pk['tools'])} usable" + ("" if pk["available"] else f"  ({pk['reason']})"))
    print("Production skills (DECLARED = NOT_IMPLEMENTED; AVAILABLE = a tool was selected here):")
    w = max(len(r["skill"]) for r in rows)
    for r in rows:
        print(f"  {r['status']:15s} {r['skill']:{w}s}  v{r['version']}  phase {r['phase']}  tool={r['tool'] or '-'}" + ("" if r["status"] == "AVAILABLE" else f"  ({r['reason']})"))
    return 0


def cmd_transcribe(args, svc: Service) -> int:
    """Speech recognition through transcription-skill as an analysis of kind `transcript` (+ the core kinds): typed options only;
    the result is an Observation and SpeechEvents, never a decision, a subtitle or a command."""
    params = {"language": args.language, "engine": args.engine, "model": args.model, "word_timestamps": True if args.word_timestamps else None,
              "offline": True if args.offline else None, "initial_prompt": args.initial_prompt, "beam_size": args.beam_size, "timeout": args.timeout,
              "max_audio_seconds": args.max_audio_seconds}
    args.kinds = ["transcript"] + list(args.kinds or [])
    profile, rules, analysis = svc.analyze(args.inputs, args.profile, hash_sources=not args.no_hash, strategy=args.strategy, use_cache=not args.no_cache, kinds=args.kinds,
                                           params=params, allowed_inputs=args.allowed_inputs)
    if args.json:
        _print(analysis.to_dict(), True)
        return 0
    args.kinds = None
    return _print_analysis(analysis)


def cmd_analyze(args, svc: Service) -> int:
    profile, rules, analysis = svc.analyze(args.inputs, args.profile, hash_sources=not args.no_hash, strategy=args.strategy, use_cache=not args.no_cache, kinds=args.kinds)
    doc = analysis.to_dict()
    if args.json:
        _print(doc, True)
        return 0
    return _print_analysis(analysis)


def _print_analysis(analysis) -> int:
    for a in analysis.assets:
        t = a.technical
        v, au = t.get("video") or {}, t.get("audio") or {}
        line = f"{a.path}\n  type {a.type} ({a.classification.get('confidence')}), {t.get('duration')}s, "
        line += f"video {v.get('codec')} {v.get('width')}x{v.get('height')} @ {v.get('fps')}fps" if v else "no video"
        line += (" VFR?" if v.get("variable_frame_rate_suspected") else "") + (f" [{v.get('hdr_format')}]" if v.get("hdr") else "")
        line += f", audio {au.get('codec')} {au.get('channels')}ch" if au else ", no audio"
        print(line)
    for o in analysis.observations:
        ext = f"  skill {o.skill}@{o.skill_version}  ext {o.external_id}  cache {o.cache.get('status')}" if o.skill_version else ""
        print(f"  observation {o.id}  {o.kind:15s} {o.source}{ext}")
        if o.kind == "transcript":
            tr = o.data
            prov = tr.get("provenance") or {}
            print(f"    transcript {tr.get('id')}  language {tr.get('language')} ({tr.get('language_source')})  {len(tr.get('segments') or [])} segment(s)  "
                  f"engine {tr.get('engine')}@{tr.get('engine_version')} {prov.get('execution_mode')}  model {prov.get('model')}  speaker_id null (recognition only)")
            for seg in (tr.get("segments") or [])[:20]:
                print(f"      {seg['start']:8.3f}-{seg['end']:8.3f}  {seg.get('text', '')[:100]}")
            if len(tr.get("segments") or []) > 20:
                print(f"      … {len(tr['segments']) - 20} more segment(s)")
    for e in analysis.timeline.query():
        r = e.range
        print(f"  {e.kind:8s} {e.type:16s} {r['start']:8.3f}-{(r['end'] if r['end'] is not None else r['start']):8.3f}  {json.dumps(e.metadata, default=str)[:80]}")
    for w in analysis.warnings:
        print("  warning:", w)
    for an in analysis.analyses:
        rows = an["rows"]
        hits = sum(1 for r in rows if r.get("cache_hit"))
        print(f"  analysis {an['analysis_id']} {an['request']['strategy']} by {an['analyzer']}: {len(rows)} measurement(s), {hits} from cache, "
              f"{an['budget']['calls']} tool call(s) in {an['budget']['seconds']}s, status {an['status']}")
        for r in rows:
            if r["status"] != "OK":
                print(f"    {r['kind']:12s} {r['status']:8s} {(r.get('error') or {}).get('kind', r.get('reason', ''))}")
    return 0


def cmd_plan(args, svc: Service) -> int:
    ir = svc.plan(args.inputs, args.profile, request_text=args.request or "", user_requirements=_kv(args.set), project_name=args.name, hash_sources=not args.no_hash,
                  strategy=args.strategy, use_cache=not args.no_cache, kinds=args.kinds, params={"language": args.language, "offline": True if args.offline else None})
    out = args.output or str(Path(args.inputs[0]).with_suffix("")) + ".project.json"
    if not args.output and not args.allow_source_dir:
        out = str(Path(svc.workspace) / "plans" / f"{Path(args.inputs[0]).stem}.{args.profile}.project.json")
    save_ir(ir, out)
    rep = svc.validate(ir)
    if args.json:
        _print({"project": out, "summary": ir.doc["plan"]["summary"], "decisions": ir.doc["decisions"], "validation": rep.to_dict()}, True)
    else:
        print(f"Project IR: {out}")
        _print_plan(ir.doc)
        print("Decisions:")
        for d in ir.doc["decisions"]:
            print(f"  [{d['approval']:7s} {d['risk']:6s}] {d['id']}  {d['subject']}: {d['decision']}")
        if rep.warnings:
            print("Warnings:")
            for w in rep.warnings:
                print("  -", w)
        if not rep.ok:
            print("Validation errors:")
            for e in rep.errors:
                print("  -", e)
    return 0 if rep.ok else 2


def cmd_validate(args, svc: Service) -> int:
    ir = load_ir(args.project)
    rep = svc.validate(ir, check_paths=not args.no_paths)
    _print(rep.to_dict(), True) if args.json else print("\n".join([f"ok: {rep.ok}"] + [f"error: {e}" for e in rep.errors] + [f"warning: {w}" for w in rep.warnings]))
    return 0 if rep.ok else 2


def cmd_render(args, svc: Service) -> int:
    ir = load_ir(args.project)
    approve = [x.strip() for x in args.approve.split(",")] if args.approve else None
    if args.dry_run:
        plan = svc.dry_run(ir)
        if args.json:
            _print(plan, True)
        else:
            print("Dry run — operations:")
            for op in plan["operations"]:
                print(f"  {op['op']} {op['tool']} ({op['kind']})\n    $ {op['command'][0]}")
            print("Required capabilities:", json.dumps(plan["required_capabilities"]))
            print("Expected outputs:", *plan["expected_outputs"], sep="\n  ")
            for k in ("risks", "warnings"):
                if plan[k]:
                    print(f"{k.capitalize()}:", *plan[k], sep="\n  - ")
            if plan["pending_confirmations"]:
                print("Pending confirmations:", ", ".join(plan["pending_confirmations"]))
            if plan["blocked"]:
                print("Blocked decisions:", ", ".join(plan["blocked"]))
            print("Estimate:", json.dumps(plan["estimate"]))
        return 0 if not plan["blocked"] else 3
    out = svc.render(ir, args.project, approve=approve, timeout=args.timeout, resume=args.resume)
    if args.json:
        _print(out, True)
    else:
        print(f"Job {out['job']['id']}: {out['status']}")
        if out.get("resume"):
            r = out["resume"]
            print(f"  resumed from {r['resumed_from']} ({r['prior_state']}); plan {'CHANGED' if r['plan_changed'] else 'unchanged'}; reused {len(out['execution']['skipped'])} of {r['candidate_ops']} completed operation(s)")
        if out["status"] == "WAITING_FOR_APPROVAL":
            for d in out.get("pending", []):
                print(f"  CONFIRM {d['id']}  {d['subject']}: {d['decision']}\n    {d['reason']}")
            if out.get("hint"):
                print("  " + out["hint"])
            elif "plan_status" in out:
                print(f"  plan is {out['plan_status']}; re-run 'plan' or check for a BLOCK decision")
        if out["status"] == "BLOCKED":
            for d in out.get("blocked", []):
                print(f"  BLOCKED {d['id']} {d['subject']}: {d['reason']}")
            for did, reason in (out.get("rejected") or {}).items():
                print(f"  REJECTED {did}: {reason}")
            if out.get("hint"):
                print("  " + out["hint"])
        if out.get("validation") and not out["validation"]["ok"]:
            print("  validation:", *out["validation"]["errors"], sep="\n    ")
        for a in out.get("artifacts", []):
            print(f"  {a['type']}: {a['path']}  qa={a['qa_status']} stage={a['stage']}")
        if out.get("qa"):
            print(f"  QA {out['qa']['status']}: {sum(1 for i in out['qa']['items'] if i['status'] == 'PASS')} pass, {len(out['qa']['incidents'])} incident(s)")
        if out.get("report"):
            print("  report:", out["report"])
        ex = out.get("execution") or {}
        for r in ex.get("recovery", []):
            print(f"  recovery: op {r['op']} {r['class']} → {r['action']}: {r['reason']}")
        if ex.get("failed_op"):
            hits = [x for x in ex["results"] if x["op_id"] == ex["failed_op"]]
            if hits:
                print("  failed:", hits[-1]["tool"], "\n   ", hits[-1]["stderr_tail"].replace("\n", "\n    "))
    return {"COMPLETED": 0, "WAITING_FOR_APPROVAL": 4, "BLOCKED": 3, "REVIEW": 5, "CANCELLED": 130}.get(out["status"], 1)


def _ids(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def cmd_approve(args, svc: Service) -> int:
    ir = load_ir(args.project)
    try:
        out = svc.approve(ir, args.project, _ids(args.decision), who=args.by, reason=args.reason or "")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    if args.json:
        _print(out, True)
    else:
        print(f"approved {len(out['approved'])} decision(s) on plan v{out['plan_version']}: {', '.join(out['approved']) or '-'}")
        if out["pending"]:
            print("still pending:", ", ".join(out["pending"]))
        print("renderable:", out["renderable"], f"(approved_plan_version={out['approved_plan_version']})")
    return 0 if out["renderable"] else 4


def cmd_reject(args, svc: Service) -> int:
    ir = load_ir(args.project)
    try:
        out = svc.reject(ir, args.project, _ids(args.decision), reason=args.reason, who=args.by)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _print(out, True)
    else:
        print(f"rejected {len(out['rejected'])} decision(s) on plan v{out['plan_version']} by {out['by']}: {', '.join(out['rejected'])}")
        print(f"reason: {out['reason']}")
        print(f"{out['cited_operations']} operation(s) still cite rejected decisions; {out['hint']}")
    return 0


def cmd_revise(args, svc: Service) -> int:
    ir = load_ir(args.project)
    try:
        out = svc.revise(ir, args.project, feedback=args.feedback or "", user_requirements=_kv(args.set), who=args.by)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _print(out, True)
        return 0 if out["created"] else 5
    if not out["created"]:
        print(f"no new plan version: {out['reason']} (still v{out['version']})")
        return 5
    print(f"plan v{out['version']} written to {args.project}; previous version preserved at {out['snapshot']}")
    _print_diff(out["diff"])
    if out.get("dropped_proposals"):
        print("suppressed (rejected earlier):", "; ".join(f"{d['subject']}@{d['asset_id']}: {d['decision']}" for d in out["dropped_proposals"]))
    print(out["hint"])
    return 0


def _print_diff(diff: Dict[str, Any]) -> None:
    print(f"PlanDiff v{diff['from_version']} → v{diff['to_version']}:" + (" (no changes)" if diff["empty"] else ""))
    for line in diff["summary"]:
        print("  ", line)


def cmd_diff(args, svc: Service) -> int:
    from video_agent.project.diff import plan_diff
    from video_agent.project.ir import snapshot_path
    cur = load_ir(args.project)
    to_doc = cur.doc
    if args.to is not None and args.to != cur.version:
        to_doc = load_ir(snapshot_path(args.project, args.to)).doc
    from_v = args.__dict__["from"] if args.__dict__.get("from") is not None else to_doc["plan"]["version"] - 1
    if from_v < 1:
        print("nothing to compare: this is plan v1", file=sys.stderr)
        return 2
    from_doc = load_ir(snapshot_path(args.project, from_v)).doc if from_v != cur.version else cur.doc
    diff = plan_diff(from_doc, to_doc)
    hist = next((h for h in to_doc["revision"]["history"] if h["version"] == to_doc["plan"]["version"]), None)
    if hist:
        for did, reason in (hist.get("rejection_reasons") or {}).items():
            dec = next((d for d in to_doc["decisions"] if d["id"] == did), None)
            diff["summary"].insert(0, f"REJECTED {dec['subject'] if dec else did}: {dec['decision'] if dec else ''} — {reason}")
    _print(diff, True) if args.json else _print_diff(diff)
    return 0


def cmd_check(args, svc: Service) -> int:
    out = svc.check(args.output, args.platform, qc=bool(getattr(args, "qc", False)))
    if args.json:
        _print(out, True)
    else:
        p = out["probe"]
        v, a = p.get("video") or {}, p.get("audio") or {}
        print(f"{args.output}: {p.get('duration')}s, {v.get('codec')} {v.get('width')}x{v.get('height')} @ {v.get('fps')}fps, " + (f"audio {a.get('codec')} {a.get('channels')}ch" if a else "no audio"))
        for r in out["check"].get("checks", []):
            print(f"  {r['status']:4s} {r['check']:14s} {r['value']}  (expected {r['expected']})" + (f"  -> {r['fix']}" if r["status"] != "PASS" and r.get("fix") else ""))
        if out.get("qc"):
            q = out["qc"]
            print(f"  qc   {q.get('verdict') or 'not admitted'}  report {q.get('report_id') or '-'}" + (f"  problems: {'; '.join(q['problems'])}" if q.get("problems") else ""))
            for c in q.get("checks") or []:
                if c.get("status") != "PASS":
                    print(f"       {c.get('status'):4s} {c.get('check_id')}  {', '.join(c.get('finding_codes') or [])}")
    ok = bool(out["check"].get("ok")) and (not out.get("qc") or out["qc"].get("verdict") in ("PASS", "WARN"))
    return 0 if ok else 1


def _print_plan(doc) -> None:
    pl = doc["plan"]
    print(f"Production plan {pl.get('id', '-')} v{pl['version']}  status {pl.get('status', '-')}  objective: {pl.get('objective', '')}")
    for line in pl["summary"]:
        print("  -", line)
    if pl.get("steps"):
        print("Steps (deterministic order):")
        for st in pl["steps"]:
            scope = st.get("temporal_scope")
            sc = f"  scope {scope['start']:.3f}-{scope['end']:.3f}" if scope and scope.get("end") is not None else ""
            deps = f"  after {','.join(st['depends_on'])}" if st.get("depends_on") else ""
            print(f"  {st['order']:2d}. {st['id']:28s} {st['skill']:22s} -> {st.get('tool') or '-':22s} [{st.get('status', '?')}]{sc}{deps}  evidence {len(st.get('evidence') or [])}")
    if pl.get("outputs"):
        print("Planned outputs:")
        for o in pl["outputs"]:
            print(f"  {o['role']:10s} {o['logical']}  format={o['format']}  expected={json.dumps(o.get('expected') or {}, default=str)}")


def _art_line(a) -> str:
    return f"{a['id']}  {a.get('type', ''):9s} {a.get('logical_name', '')}  v{a.get('plan_version', '?')}  qa={a.get('qa_status')}  {a.get('stage')}/{a.get('delivery_status')}  sha256={(a.get('hash') or '')[:12]}  {a.get('name', '')}"


def cmd_artifacts(args, svc: Service) -> int:
    ir = load_ir(args.project) if args.project else None
    rows = svc.artifacts(ir, job_id=args.job)
    if args.json:
        _print(rows, True)
        return 0
    for a in rows:
        print(_art_line(a))
    if not rows:
        print("no registered artifacts")
    return 0


def cmd_artifact(args, svc: Service) -> int:
    a = svc.artifact(args.artifact_id)
    if args.json:
        _print(a, True)
        return 0
    print(_art_line(a))
    print(f"  path      : {a['path']}\n  size      : {a['size']}  format {a['format']}  tool {a['tool']}@{a['tool_version']}")
    print(f"  plan      : {a['plan_id']} v{a['plan_version']}  step {a.get('step_id')}  decisions {','.join(a.get('decision_ids') or [])}")
    print(f"  jobs      : {','.join(a['jobs'])}  operations {','.join(a.get('operations') or [])}")
    print(f"  qa        : {a['qa'].get('status')} (pass {a['qa'].get('pass')} warn {a['qa'].get('warn')} fail {a['qa'].get('fail')})")
    print(f"  integrity : {'ok' if a['integrity']['ok'] else a['integrity']['error']}")
    for h in a.get("delivery_history") or []:
        print(f"  history   : {h}")
    return 0


def cmd_deliver(args, svc: Service) -> int:
    try:
        a = svc.promote_artifact(args.artifact_id, "final", who=args.by, reason=args.reason or "", channel=args.channel)
    except Exception as e:  # noqa: BLE001 — surfaced as exit code 3 with the lifecycle reason
        print(f"error: {e}", file=sys.stderr)
        return 3
    _print(a, True) if args.json else print(_art_line(a))
    return 0


def cmd_archive(args, svc: Service) -> int:
    if args.artifact_id:
        try:
            a = svc.archive_artifact(args.artifact_id, who=args.by, reason=args.reason or "")
        except Exception as e:  # noqa: BLE001
            print(f"error: {e}", file=sys.stderr)
            return 3
        _print(a, True) if args.json else print(_art_line(a))
        return 0
    ir = load_ir(args.project)
    idx = svc.artifact_store().archive_index(ir.doc["project"]["id"])
    if args.json:
        _print(idx, True)
        return 0
    for e in idx["entries"]:
        print(f"{e['artifact_id']}  {e['logical_name']}  plan {e['plan_id']} v{e['plan_version']}  jobs {','.join(e['jobs'])}  qa={e['qa_status']}  sha256={e['sha256'][:12]}  archived {e['archived_at']}")
    if not idx["entries"]:
        print("archive is empty for this project")
    return 0


def cmd_events(args, svc: Service) -> int:
    ir = load_ir(args.project)
    events = ir.doc["timeline"].get("events") or []
    if args.json:
        _print(events, True)
        return 0
    for e in events:
        r = e["range"]
        end = "" if r.get("end") is None else f"-{r['end']:.3f}"
        print(f"{e['id']}  {e.get('event_type', ''):16s} {e.get('subtype', ''):12s} {r['start']:8.3f}{end:<10s} {e.get('provenance') or e['kind']:12s} asset={e.get('asset_id') or '-'}  evidence={','.join(e.get('evidence') or [])}")
    return 0


def cmd_context(args, svc: Service) -> int:
    """The production situation recorded in an IR: per timeline scope, which kinds of events are active (references only)."""
    from video_agent.context import contexts_at, contexts_between
    ir = load_ir(args.project)
    ctxs = Service.contexts_of(ir.doc)
    if args.at is not None:
        ctxs = contexts_at(ctxs, args.at, args.timeline)
    elif args.between:
        ctxs = contexts_between(ctxs, args.between[0], args.between[1], args.timeline)
    elif args.timeline:
        ctxs = [c for c in ctxs if c.timeline_id == args.timeline]
    if args.json:
        _print([c.to_dict() for c in ctxs], True)
        return 0
    for c in ctxs:
        tracks = ", ".join(f"{t['event_type']}/{t['subtype']}×{len(t['event_ids'])}" for t in c.tracks) or "nothing observed"
        print(f"{c.id}  {c.timeline_id:24s} {c.scope['start']:8.3f}-{c.scope['end']:8.3f}  {tracks}  obs={len(c.observation_ids)} inf={len(c.inference_ids)}")
    return 0


def cmd_sessions(args, svc: Service) -> int:
    ir = load_ir(args.project)
    sessions = ir.doc["timeline"].get("sessions") or []
    if args.json:
        _print(sessions, True)
        return 0
    for x in sessions:
        print(f"{x['id']}  {x['name']}  {x['range']['start']:.3f}-{x['range']['end']:.3f}  assets={','.join(x['asset_ids'])}  events={len(x['event_ids'])}  {x.get('provenance', '')}")
    return 0


def cmd_explain(args, svc: Service) -> int:
    if not getattr(args, "artifact", None) and not args.project:
        raise ValueError("PROJECT is required for --decision / --step / --context / --observation / --pipeline (only --artifact does not need one)")
    if getattr(args, "pipeline", False):
        ir = load_ir(args.project)
        job = prov = None
        arts = svc.artifacts(ir)
        runs = ir.doc.get("provenance", {}).get("runs") or []
        if runs:
            from video_agent.jobs import JobStore
            j = JobStore(svc.workspace).load(runs[-1]["job_id"])
            if j is not None:
                job = j.to_dict()
                pp = Path(j.dir) / "provenance.json"
                if pp.exists():
                    prov = json.loads(pp.read_text(encoding="utf-8"))
        info = svc.explain_pipeline(ir.doc, job=job, provenance=prov, artifacts=arts)
        if args.json:
            _print(info, True)
            return 0
        print("  ".join(f"{lv} {info['counts'][lv]}" for lv in info["levels"]))
        for row in info["rows"]:
            refs = row.get("evidence") or row.get("decisions") or row.get("depends_on") or []
            print(f"{row['level']:12s} {str(row.get('id'))[:44]:44s} {str(row.get('detail') or '')[:110]}" + (f"  ← {','.join(str(x) for x in refs)[:80]}" if refs else ""))
        print(f"boundary: {info['boundary']}")
        return 0
    if getattr(args, "artifact", None):
        info = svc.explain_artifact(args.artifact)
        if args.json:
            _print(info, True)
            return 0
        a = info["artifact"]
        print(f"Artifact {a['id']}  {a['logical_name']}  sha256 {a['hash']}  qa {a['qa_status']}  {a['stage']}/{a['delivery_status']}")
        print(f"  jobs {','.join(info['jobs'])}  plan {a['plan_id']} v{a['plan_version']}")
        for op in info["operations"]:
            print(f"  operation {op.get('what')}  skill {op.get('skill')}  decisions {','.join(op.get('decision') or [])}")
        if info.get("step"):
            st = info["step"]["step"]
            print(f"  step {st['id']}  {st['skill']} -> {st.get('tool')}")
            for row in info["step"]["chain"]:
                print(f"{'  ' * (row['level'] + 2)}{row['kind']:11s} {row['id']}  {row.get('provenance') or ''}  {row.get('detail') or ''}" + (f"  source {row['source']}" if row.get("source") else ""))
        return 0
    ir = load_ir(args.project)
    if getattr(args, "context", None):
        try:
            info = svc.explain_context(ir.doc, args.context)
        except KeyError:
            print("no such context", file=sys.stderr)
            return 1
        if args.json:
            _print(info, True)
            return 0
        for row in info["chain"]:
            extra = f"  source {row['source']}" if row.get("source") else ""
            extra += f"  [{row['generator']}]" if row.get("generator") else ""
            extra += f"  {row['approval']}/{row['status']}" if row.get("approval") else ""
            print(f"{'  ' * row['level']}{row['kind']:11s} {row['id']}  {row.get('provenance') or ''}  {row.get('detail') or ''}{extra}")
        print(f"boundary: {info['boundary']}")
        return 0
    if getattr(args, "observation", None):
        try:
            info = svc.explain_observation(ir.doc, args.observation)
        except KeyError:
            print("no such observation", file=sys.stderr)
            return 1
        if args.json:
            _print(info, True)
            return 0
        o = info["observation"]
        print(f"Observation {o['id']}  {o['kind']}  {o['source']}  provenance {o.get('provenance')}")
        for row in info["chain"]:
            print(f"{'  ' * (row['level'] + 1)}{row['kind']:11s} {row['id']}  {row.get('provenance') or ''}  {row.get('detail') or ''}" + (f"  source {row['source']}" if row.get("source") else ""))
        print(f"  boundary: {info['boundary']}")
        return 0
    if getattr(args, "step", None):
        from video_agent.agent.production_plan import explain_step
        try:
            info = explain_step(ir.doc, args.step)
        except KeyError:
            print("no such step", file=sys.stderr)
            return 1
        if args.json:
            _print(info, True)
            return 0
        st = info["step"]
        print(f"Step {st['id']}  {st['skill']} -> {st.get('tool') or '-'}  status {st.get('status')}  scope {st.get('temporal_scope')}")
        for row in info["chain"]:
            extra = ""
            if row.get("ai"):
                extra = f"  AI {row['ai'].get('provider')}/{row['ai'].get('model')} hash={(row['ai'].get('response_hash') or '')[:12]}"
            if row.get("source"):
                extra += f"  source {row['source']}"
            print(f"{'  ' * row['level']}{row['kind']:11s} {row['id']}  {row.get('provenance') or ''}  {row.get('detail') or ''}{extra}")
        return 0
    decs = ir.doc["decisions"]
    if args.decision:
        try:
            infos = svc.explain_decision(ir.doc, args.decision)
        except KeyError:
            print("no such decision", file=sys.stderr)
            return 1
        if args.json:
            _print(infos, True)
            return 0
        for info in infos:
            d = info["decision"]
            print(f"{d['id']}  {d['subject']}  [{d.get('type') or '-'}]\n  decision : {d['decision']}\n  why      : {d['reason']}\n  confidence {d['confidence']:.2f}  risk {d['risk']}  approval {d['approval']}  status {d['status']}  provenance {d['provenance']}  executable {'yes' if info['executable'] else 'no'}")
            rv = info.get("review")
            if rv:
                print(f"  review   : {rv['action']} by {rv['by']} at {rv['at']} (plan v{rv['plan_version']})" + (f" — {rv['reason']}" if rv.get("reason") else ""))
            print("  basis:")
            for b in info["basis"]:
                extra = f"  [{b['rule_id']}]" if b.get("rule_id") else ""
                extra += "  (constraint)" if b.get("hard") else ""
                extra += f"  source {b['source']}" if b.get("source") else ""
                print(f"    {b['kind']:11s} {b.get('key') or '-'} = {json.dumps(b.get('value'), default=str, ensure_ascii=False)}  {b.get('provenance') or ''}{extra}")
                for n in b.get("notes") or []:
                    print(f"                {n}")
            print("  evidence:")
            for row in info["evidence"]:
                print(f"{'  ' * (row['level'] + 1)}{row['kind']:11s} {row['id']}  {row.get('provenance') or ''}  {row.get('detail') or ''}" + (f"  source {row['source']}" if row.get("source") else ""))
            for alt in d.get("alternatives") or []:
                print(f"    alternative: {alt['decision']} — {alt['reason']}" + (f" (cost: {alt['cost']})" if alt.get("cost") else ""))
            if info["plan"]["steps"] or info["plan"]["operations"]:
                print("  plan:")
                for st in info["plan"]["steps"]:
                    print(f"    step {st['id']}  {st['skill']} -> {st.get('tool') or '-'}  status {st.get('status')}  params {json.dumps(st.get('params'), default=str)}")
                for op in info["plan"]["operations"]:
                    print(f"    ir   {op['section']}.{op['type']}  " + json.dumps({k: v for k, v in op.items() if k not in ('section', 'type') and v is not None}, default=str))
            else:
                print("  plan: no step / operation cites this decision (it is not executed)")
            print(f"  boundary : {info['boundary']}")
        return 0
    evidence = {o["id"]: o for o in ir.doc["analysis"]["observations"]}
    evidence.update({i["id"]: i for i in ir.doc["analysis"]["inferences"]})
    evidence.update({e["id"]: e for e in ir.doc["timeline"]["events"]})
    evidence.update({r["id"]: r for r in ir.doc["requirements"]})
    if args.json:
        _print([{**d, "evidence_detail": [evidence.get(e, e) for e in d["evidence"]]} for d in decs], True)
        return 0
    reviews = ir.doc["execution"].get("reviews") or {}
    for d in decs:
        print(f"{d['id']}  {d['subject']}  [{d.get('type') or '-'}]\n  decision : {d['decision']}\n  why      : {d['reason']}\n  confidence {d['confidence']:.2f}  risk {d['risk']}  approval {d['approval']}  status {d['status']}  provenance {d['provenance']}")
        rv = reviews.get(d["id"])
        if rv:
            print(f"  review   : {rv['action']} by {rv['by']} at {rv['at']} (plan v{rv['plan_version']})" + (f" — {rv['reason']}" if rv.get("reason") else ""))
        for e in d["evidence"]:
            ev = evidence.get(e)
            if isinstance(ev, dict):
                kind = ev.get("kind") or ev.get("type") or ev.get("key")
                detail = ev.get("statement") or json.dumps(ev.get("data") or ev.get("metadata") or ev.get("value"), default=str)[:120]
                print(f"    evidence {e} [{kind}] {detail}")
            else:
                print(f"    evidence {e}")
        for alt in d.get("alternatives", []):
            print(f"    alternative: {alt['decision']} — {alt['reason']}" + (f" (cost: {alt['cost']})" if alt.get("cost") else ""))
    calls = ir.doc["provenance"].get("ai_calls") or []
    if calls:
        prov = ir.doc["provenance"].get("ai_provider") or {}
        print(f"AI: provider {prov.get('provider', '?')} model {prov.get('model', '?')}, {len(calls)} call(s); recommendations are proposals (AI_GENERATED), never execution authority")
        for c in calls:
            print(f"  {c['at']}  {c['task_type']}  {'ok' if c['ok'] else 'FAILED ' + (c.get('error') or {}).get('kind', '')}  hash={(c.get('response_hash') or '-')[:12]}  latency={c.get('latency_s')}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="video-agent", description="AI video production orchestrator (ffmpeg-skill as the execution engine)")
    ap.add_argument("--version", action="version", version=f"video-agent {__version__}")
    ap.add_argument("--workspace", help="workspace root (default $VIDEO_AGENT_WORKSPACE or ./video-agent-work)")
    ap.add_argument("--ffmpeg-skill-dir", help="path to an ffmpeg-skill checkout (default $VIDEO_AGENT_FFMPEG_SKILL_DIR or ~/.claude/skills/ffmpeg-skill)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("doctor", help="inspect the environment (AVAILABLE / MISSING / DEGRADED / UNKNOWN)")
    p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("skills", help="list skills with their status here (AVAILABLE / UNAVAILABLE / NOT_IMPLEMENTED) and the selected tool")
    p.add_argument("--check-provides", action="store_true",
                   help="read-only diagnostic: does each Skill's Capability Contract provides[] match what this Agent "
                        "recognizes and a production SkillSpec consumes? (PROVIDES_VALID / PROVIDES_MISMATCH / "
                        "CAPABILITY_UNCONSUMED / CAPABILITY_MISSING / UNKNOWN). Never affects tool selection.")
    p.set_defaults(fn=cmd_skills)
    p = sub.add_parser("analyze", help="probe media and list observed events")
    p.add_argument("inputs", nargs="+"); p.add_argument("--profile", default="generic"); p.add_argument("--no-hash", action="store_true", help="skip sha256 of sources (faster on huge files)")
    p.add_argument("--strategy", choices=["FULL", "TARGETED", "CACHED_ONLY"], help="analysis strategy (default: profile policy analysis.strategy, else FULL)")
    p.add_argument("--no-cache", action="store_true", help="do not read or write the observation cache")
    p.add_argument("--kind", action="append", dest="kinds", help="analysis kind(s) to run in addition to the core kinds (e.g. duration, integrity, scene_detection)")
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("transcribe", help="speech recognition through transcription-skill → transcript Observation + SpeechEvents (recognition only)")
    p.add_argument("inputs", nargs="+"); p.add_argument("--profile", default="generic"); p.add_argument("--no-hash", action="store_true", help="skip sha256 of sources")
    p.add_argument("--language", help="ISO 639-1 code (ja, en, …); default: the engine detects it")
    p.add_argument("--engine", help="engine id declared by the Skill's contract (default: the Skill's default engine)")
    p.add_argument("--model", help="model name for the engine (never a path)")
    p.add_argument("--offline", action="store_true", help="hard constraint: no remote engine, no model download (missing model → unavailable)")
    p.add_argument("--word-timestamps", action="store_true", help="also record per-word timestamps")
    p.add_argument("--initial-prompt", help="vocabulary hint passed to the decoder (names, terms); not an instruction")
    p.add_argument("--beam-size", type=int); p.add_argument("--timeout", type=float, help="budget: seconds the engine may run"); p.add_argument("--max-audio-seconds", type=float)
    p.add_argument("--allowed-input", action="append", dest="allowed_inputs", metavar="DIR", help="input roots for the Skills' path policy (repeatable; default: the inputs' directories)")
    p.add_argument("--strategy", choices=["FULL", "TARGETED", "CACHED_ONLY"]); p.add_argument("--no-cache", action="store_true", help="do not read or write caches (the Skill's included)")
    p.add_argument("--kind", action="append", dest="kinds", help="additional analysis kind(s)")
    p.set_defaults(fn=cmd_transcribe)
    p = sub.add_parser("plan", help="analyze, decide and write a Project IR")
    p.add_argument("inputs", nargs="+"); p.add_argument("--profile", default="generic"); p.add_argument("--request", help="free-text request (only unambiguous phrases are used)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="explicit USER requirement, e.g. audio.loudness.target_lufs=-16 or edit.trim_leading_silence=true")
    p.add_argument("--name"); p.add_argument("-o", "--output", help="where to write project.json (default: <workspace>/plans/<name>.project.json)")
    p.add_argument("--allow-source-dir", action="store_true", help="write project.json next to the source (default keeps source directories untouched)")
    p.add_argument("--no-hash", action="store_true")
    p.add_argument("--strategy", choices=["FULL", "TARGETED", "CACHED_ONLY"], help="analysis strategy (default: profile policy analysis.strategy, else FULL)")
    p.add_argument("--no-cache", action="store_true", help="do not read or write the observation cache")
    p.add_argument("--kind", action="append", dest="kinds", help="additional analysis kind(s) to record in the IR (e.g. transcript, duration)")
    p.add_argument("--language", help="transcript kind: ISO 639-1 code for recognition (default: detected by the engine)")
    p.add_argument("--offline", action="store_true", help="transcript kind: no remote engine, no model download")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("validate", help="validate a Project IR (schema, semantics, capabilities)")
    p.add_argument("project"); p.add_argument("--no-paths", action="store_true", help="do not require asset paths to exist")
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("render", help="execute a Project IR through ffmpeg-skill, then QA")
    p.add_argument("project"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--approve", help="comma separated decision ids, or 'all'")
    p.add_argument("--timeout", type=float, help="per-operation timeout in seconds")
    p.add_argument("--resume", metavar="JOB_ID|last", help="reuse completed operations of a previous job (outputs are reused only when their chained key and file still match)")
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser("approve", help="approve CONFIRM decisions; approves the plan version when nothing is pending")
    p.add_argument("project"); p.add_argument("--decision", required=True, help="comma separated decision ids, or 'all'"); p.add_argument("--reason", help="optional note"); p.add_argument("--by", help="actor (default: OS user)")
    p.set_defaults(fn=cmd_approve)
    p = sub.add_parser("reject", help="reject decisions with a reason; the plan must then be revised")
    p.add_argument("project"); p.add_argument("--decision", required=True, help="comma separated decision ids, or 'all'"); p.add_argument("--reason", required=True); p.add_argument("--by")
    p.set_defaults(fn=cmd_reject)
    p = sub.add_parser("revise", help="produce the next plan version from rejections and feedback (previous version is preserved)")
    p.add_argument("project"); p.add_argument("--feedback", help="free-text feedback (only unambiguous phrases are interpreted)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="structured feedback, e.g. audio.loudness.target_lufs=-16"); p.add_argument("--by")
    p.set_defaults(fn=cmd_revise)
    p = sub.add_parser("diff", help="show the PlanDiff between two plan versions (default: previous → current)")
    p.add_argument("project"); p.add_argument("--from", type=int, dest="from"); p.add_argument("--to", type=int)
    p.set_defaults(fn=cmd_diff)
    p = sub.add_parser("check", help="probe + platform compliance of an output file")
    p.add_argument("output"); p.add_argument("--platform", default="custom"); p.add_argument("--qc", action="store_true", help="also run the qc-skill report (admitted only for the file's own sha256)")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("events", help="temporal events recorded in a Project IR (canonical order)")
    p.add_argument("project"); p.set_defaults(fn=cmd_events)
    p = sub.add_parser("context", help="production contexts (situations) recorded in a Project IR: what is observed when")
    p.add_argument("project"); p.add_argument("--at", type=float, help="the situation at this second"); p.add_argument("--between", type=float, nargs=2, metavar=("START", "END"))
    p.add_argument("--timeline", help="asset:<id> or master"); p.set_defaults(fn=cmd_context)
    p = sub.add_parser("sessions", help="temporal sessions recorded in a Project IR")
    p.add_argument("project"); p.set_defaults(fn=cmd_sessions)
    p = sub.add_parser("explain", help="why was this decided? show reason, evidence, alternatives")
    p.add_argument("project", nargs="?"); p.add_argument("--decision", help="decision id or subject: type, rationale, risk / approval, basis (policy / preference / constraint / intent), evidence chain, plan steps"); p.add_argument("--step", help="production step id: show its evidence chain")
    p.add_argument("--artifact", help="artifact id: show artifact -> job -> operations -> step -> decisions -> evidence")
    p.add_argument("--observation", help="observation id (or the Skill's external id, e.g. a transcript id): observation -> skill -> tool -> engine/model -> asset -> events")
    p.add_argument("--context", help="context id: context -> tracks -> events -> observations, plus the inferences / decisions resting on it")
    p.add_argument("--pipeline", action="store_true", help="the whole chain of the project: request -> requirements -> observations -> events -> inferences -> decisions -> plan -> IR -> skill -> capability -> tool -> execution -> QA -> artifacts")
    p.set_defaults(fn=cmd_explain)
    p = sub.add_parser("artifacts", help="registered artifacts (of a project IR, or all)")
    p.add_argument("project", nargs="?"); p.add_argument("--job"); p.set_defaults(fn=cmd_artifacts)
    p = sub.add_parser("artifact", help="one artifact manifest with an integrity check")
    p.add_argument("artifact_id"); p.set_defaults(fn=cmd_artifact)
    p = sub.add_parser("deliver", help="promote a QA-passed artifact of an approved plan to DELIVERED (recorded; no upload)")
    p.add_argument("artifact_id"); p.add_argument("--by"); p.add_argument("--reason"); p.add_argument("--channel", default="local"); p.set_defaults(fn=cmd_deliver)
    p = sub.add_parser("archive", help="archive an artifact, or list a project's archive index")
    p.add_argument("project", nargs="?"); p.add_argument("--artifact", dest="artifact_id"); p.add_argument("--by"); p.add_argument("--reason"); p.set_defaults(fn=cmd_archive)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):   # statements carry Unicode (≥, →, Japanese); a cp1252 console must not turn a report into a crash
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, AttributeError):
                pass
    args = build_parser().parse_args(argv)
    try:
        svc = Service(workspace=args.workspace, ffmpeg_skill_dir=args.ffmpeg_skill_dir)
        return args.fn(args, svc)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
