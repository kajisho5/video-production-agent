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


def cmd_analyze(args, svc: Service) -> int:
    profile, rules, analysis = svc.analyze(args.inputs, args.profile, hash_sources=not args.no_hash)
    doc = analysis.to_dict()
    if args.json:
        _print(doc, True)
        return 0
    for a in analysis.assets:
        t = a.technical
        v, au = t.get("video") or {}, t.get("audio") or {}
        print(f"{a.path}\n  type {a.type} ({a.classification.get('confidence')}), {t.get('duration')}s, video {v.get('codec')} {v.get('width')}x{v.get('height')} @ {v.get('fps')}fps"
              + (" VFR?" if v.get("variable_frame_rate_suspected") else "") + (f" [{v.get('hdr_format')}]" if v.get("hdr") else "") + f", audio {au.get('codec')} {au.get('channels')}ch" if au else "")
    for e in analysis.timeline.query():
        r = e.range
        print(f"  {e.kind:8s} {e.type:16s} {r['start']:8.3f}-{(r['end'] if r['end'] is not None else r['start']):8.3f}  {json.dumps(e.metadata, default=str)[:80]}")
    for w in analysis.warnings:
        print("  warning:", w)
    return 0


def cmd_plan(args, svc: Service) -> int:
    ir = svc.plan(args.inputs, args.profile, request_text=args.request or "", user_requirements=_kv(args.set), project_name=args.name, hash_sources=not args.no_hash)
    out = args.output or str(Path(args.inputs[0]).with_suffix("")) + ".project.json"
    if not args.output and not args.allow_source_dir:
        out = str(Path(svc.workspace) / "plans" / (Path(args.inputs[0]).stem + ".project.json"))
    save_ir(ir, out)
    rep = svc.validate(ir)
    if args.json:
        _print({"project": out, "summary": ir.doc["plan"]["summary"], "decisions": ir.doc["decisions"], "validation": rep.to_dict()}, True)
    else:
        print(f"Project IR: {out}")
        print("Plan:")
        for s in ir.doc["plan"]["summary"]:
            print("  -", s)
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
    out = svc.render(ir, args.project, approve=approve, timeout=args.timeout)
    if args.json:
        _print(out, True)
    else:
        print(f"Job {out['job']['id']}: {out['status']}")
        if out["status"] == "WAITING_FOR_APPROVAL":
            for d in out["pending"]:
                print(f"  CONFIRM {d['id']}  {d['subject']}: {d['decision']}\n    {d['reason']}")
            print("  " + out["hint"])
        if out["status"] == "BLOCKED":
            for d in out.get("blocked", []):
                print(f"  BLOCKED {d['id']} {d['subject']}: {d['reason']}")
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
            last = [x for x in ex["results"] if x["op_id"] == ex["failed_op"]][-1]
            print("  failed:", last["tool"], "\n   ", last["stderr_tail"].replace("\n", "\n    "))
    return {"COMPLETED": 0, "WAITING_FOR_APPROVAL": 4, "BLOCKED": 3, "REVIEW": 5}.get(out["status"], 1)


def cmd_check(args, svc: Service) -> int:
    out = svc.check(args.output, args.platform)
    if args.json:
        _print(out, True)
    else:
        p = out["probe"]
        v, a = p.get("video") or {}, p.get("audio") or {}
        print(f"{args.output}: {p.get('duration')}s, {v.get('codec')} {v.get('width')}x{v.get('height')} @ {v.get('fps')}fps, audio {a.get('codec')} {a.get('channels')}ch")
        for r in out["check"].get("checks", []):
            print(f"  {r['status']:4s} {r['check']:14s} {r['value']}  (expected {r['expected']})" + (f"  -> {r['fix']}" if r["status"] != "PASS" and r.get("fix") else ""))
    return 0 if out["check"].get("ok") else 1


def cmd_explain(args, svc: Service) -> int:
    ir = load_ir(args.project)
    decs = ir.doc["decisions"]
    if args.decision:
        decs = [d for d in decs if d["id"] == args.decision or d["subject"] == args.decision]
        if not decs:
            print("no such decision", file=sys.stderr)
            return 1
    evidence = {o["id"]: o for o in ir.doc["analysis"]["observations"]}
    evidence.update({i["id"]: i for i in ir.doc["analysis"]["inferences"]})
    evidence.update({e["id"]: e for e in ir.doc["timeline"]["events"]})
    evidence.update({r["id"]: r for r in ir.doc["requirements"]})
    if args.json:
        _print([{**d, "evidence_detail": [evidence.get(e, e) for e in d["evidence"]]} for d in decs], True)
        return 0
    for d in decs:
        print(f"{d['id']}  {d['subject']}\n  decision : {d['decision']}\n  why      : {d['reason']}\n  confidence {d['confidence']:.2f}  risk {d['risk']}  approval {d['approval']}  status {d['status']}  provenance {d['provenance']}")
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
    p = sub.add_parser("analyze", help="probe media and list observed events")
    p.add_argument("inputs", nargs="+"); p.add_argument("--profile", default="generic"); p.add_argument("--no-hash", action="store_true", help="skip sha256 of sources (faster on huge files)")
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("plan", help="analyze, decide and write a Project IR")
    p.add_argument("inputs", nargs="+"); p.add_argument("--profile", default="generic"); p.add_argument("--request", help="free-text request (only unambiguous phrases are used)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="explicit USER requirement, e.g. audio.loudness.target_lufs=-16 or edit.trim_leading_silence=true")
    p.add_argument("--name"); p.add_argument("-o", "--output", help="where to write project.json (default: <workspace>/plans/<name>.project.json)")
    p.add_argument("--allow-source-dir", action="store_true", help="write project.json next to the source (default keeps source directories untouched)")
    p.add_argument("--no-hash", action="store_true")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("validate", help="validate a Project IR (schema, semantics, capabilities)")
    p.add_argument("project"); p.add_argument("--no-paths", action="store_true", help="do not require asset paths to exist")
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("render", help="execute a Project IR through ffmpeg-skill, then QA")
    p.add_argument("project"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--approve", help="comma separated decision ids, or 'all'")
    p.add_argument("--timeout", type=float, help="per-operation timeout in seconds")
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser("check", help="probe + platform compliance of an output file")
    p.add_argument("output"); p.add_argument("--platform", default="custom")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("explain", help="why was this decided? show reason, evidence, alternatives")
    p.add_argument("project"); p.add_argument("--decision", help="decision id or subject")
    p.set_defaults(fn=cmd_explain)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
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
