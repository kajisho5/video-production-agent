#!/usr/bin/env python3
"""Evals (separate from unit tests): each case describes an input scenario and the intent / decisions /
warnings / plan characteristics the agent must produce. Cases run against the FakeAdapter (deterministic)
unless "media" points at a real file. Usage: python3 evals/run.py [--case NAME]"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from fake_adapter import FakeAdapter  # noqa: E402
from fake_ai_provider import FakeAIProvider  # noqa: E402
from test_unit import FakeCaps  # noqa: E402
from video_agent.service import Service  # noqa: E402


def run_case(case: dict) -> dict:
    tmp = tempfile.mkdtemp()
    fake = case.get("fake", {})
    src = Path(tmp) / "src" / "in.mp4"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"0" * 16)
    provider = FakeAIProvider(**case["ai"]) if case.get("ai") else None
    svc = Service(workspace=tmp, adapter=FakeAdapter(**fake), caps=FakeCaps(case.get("missing_capabilities", ())), provider=provider)
    ir = svc.plan([str(src)], case.get("profile", "generic"), request_text=case.get("request", ""), user_requirements=case.get("requirements"))
    d = ir.doc
    failures = []
    exp = case["expect"]
    if "intent" in exp and d["intent"]["primary"] != exp["intent"]:
        failures.append(f"intent {d['intent']['primary']} != {exp['intent']}")
    subjects = {x["subject"]: x for x in d["decisions"]}
    for subj, want in (exp.get("decisions") or {}).items():
        got = subjects.get(subj)
        if got is None:
            failures.append(f"missing decision {subj}")
            continue
        for k, v in want.items():
            if k == "decision_startswith":
                if not got["decision"].startswith(v):
                    failures.append(f"{subj}.decision {got['decision']!r} !startswith {v!r}")
            elif got.get(k) != v:
                failures.append(f"{subj}.{k} {got.get(k)!r} != {v!r}")
    for subj in exp.get("no_decisions", []):
        if subj in subjects:
            failures.append(f"unexpected decision {subj}")
    for key, prov in (exp.get("requirement_provenance") or {}).items():
        r = [x for x in d["requirements"] if x["key"] == key]
        top = sorted(r, key=lambda x: {"USER": 3, "PROFILE": 2, "DEFAULT": 1}.get(x["provenance"], 0))[-1] if r else None
        if not top or top["provenance"] != prov:
            failures.append(f"requirement {key} provenance {top and top['provenance']} != {prov}")
    plan = exp.get("plan") or {}
    if "video_ops" in plan and len(d["video"]["operations"]) != plan["video_ops"]:
        failures.append(f"video ops {len(d['video']['operations'])} != {plan['video_ops']}")
    if "audio_ops" in plan and len(d["audio"]["operations"]) != plan["audio_ops"]:
        failures.append(f"audio ops {len(d['audio']['operations'])} != {plan['audio_ops']}")
    if "blocked" in exp and bool(ir.blocked()) != exp["blocked"]:
        failures.append(f"blocked {bool(ir.blocked())} != {exp['blocked']}")
    pp = exp.get("production_plan") or {}
    steps = d["plan"]["steps"]
    if "steps" in pp and [st["skill"] for st in steps] != pp["steps"]:
        failures.append(f"plan steps {[st['skill'] for st in steps]} != {pp['steps']}")
    if "status" in pp and d["plan"]["status"] != pp["status"]:
        failures.append(f"plan status {d['plan']['status']} != {pp['status']}")
    for sk in pp.get("no_step_skills", []):
        if any(st["skill"] == sk for st in steps):
            failures.append(f"unexpected step {sk}")
    if pp.get("tools_from_registry") and any(st["tool"] not in svc.registry.get(st["skill"]).tools for st in steps):
        failures.append("a step tool is not a registry candidate")
    for bad in pp.get("must_not_contain", []):
        if bad in json.dumps(d["plan"]):
            failures.append(f"plan contains {bad!r}")
    if pp.get("validate") and not svc.validate(ir).ok:
        failures.append(f"validation errors {svc.validate(ir).errors}")
    if "render_status" in pp:
        from video_agent.project import save_ir, load_ir
        ir_path = str(Path(tmp) / "eval.json")
        save_ir(ir, ir_path)
        if pp.get("reject_first"):
            svc.reject(load_ir(ir_path), ir_path, [d["plan"]["steps"][0]["decision_id"]], reason="eval")
            if load_ir(ir_path).doc["plan"]["status"] != "REJECTED":
                failures.append("rejecting a step decision did not mark the plan REJECTED")
        rr = svc.render(load_ir(ir_path), ir_path, approve=pp.get("approve"))
        if rr["status"] != pp["render_status"]:
            failures.append(f"render status {rr['status']} != {pp['render_status']}")
        if pp["render_status"] in ("BLOCKED", "WAITING_FOR_APPROVAL") and rr.get("execution"):
            failures.append("execution happened although the plan is not approved")
    ar = exp.get("artifact") or {}
    if ar:
        import os
        from video_agent.project import save_ir, load_ir
        from video_agent.artifacts import ArtifactError, artifact_id, safe_filename
        ir_path = str(Path(tmp) / "art.json")
        save_ir(ir, ir_path)
        if ar.get("reject_first"):
            svc.reject(load_ir(ir_path), ir_path, [d["plan"]["steps"][0]["decision_id"]], reason="eval")
        rr = svc.render(load_ir(ir_path), ir_path, approve=["all"])
        arts = rr.get("artifacts") or []
        if ar.get("count") is not None and len(arts) != ar["count"]:
            failures.append(f"artifacts {len(arts)} != {ar['count']}")
        a = arts[0] if arts else None
        if a:
            if a["id"] != artifact_id(d["project"]["id"], d["plan"]["id"], a["logical_name"], a["hash"]):
                failures.append("artifact id is not the deterministic identity")
            if ar.get("qa") and a["qa_status"] != ar["qa"]:
                failures.append(f"artifact qa {a['qa_status']} != {ar['qa']}")
            if ar.get("delivery_status") and a["delivery_status"] != ar["delivery_status"]:
                failures.append(f"delivery {a['delivery_status']} != {ar['delivery_status']}")
            if ar.get("provenance"):
                info = svc.explain_artifact(a["id"])
                if not (info["jobs"] == [rr["job"]["id"]] and info["operations"] and info["step"] and info["step"]["chain"] and info["step"]["chain"][0]["kind"] == "decision"):
                    failures.append("artifact provenance chain incomplete")
            if ar.get("deliver") is not None:
                try:
                    r2 = svc.promote_artifact(a["id"], "final", who="eval")
                    ok = r2["delivery_status"] == "DELIVERED"
                except ArtifactError:
                    ok = False
                if ok != ar["deliver"]:
                    failures.append(f"deliverable {ok} != {ar['deliver']}")
            if ar.get("archive"):
                try:
                    if svc.archive_artifact(a["id"], who="eval")["delivery_status"] != "ARCHIVED":
                        failures.append("archive did not mark ARCHIVED")
                    idx = svc.artifact_store().archive_index(d["project"]["id"])
                    if not any(e["artifact_id"] == a["id"] and e["sha256"] == a["hash"] for e in idx["entries"]):
                        failures.append("archive index misses the artifact")
                except ArtifactError as e:
                    failures.append(f"archive failed: {e}")
            if ar.get("hash_mismatch"):
                Path(a["path"]).write_bytes(b"tampered")
                if svc.artifact(a["id"])["integrity"]["error"] != "ARTIFACT_HASH_MISMATCH":
                    failures.append("hash mismatch not detected")
                try:
                    svc.promote_artifact(a["id"], "final", who="eval"); failures.append("tampered artifact was delivered")
                except ArtifactError:
                    pass
            if ar.get("resume_reuse"):
                from fake_adapter import FakeAdapter as _FA
                svc2 = Service(workspace=tmp, adapter=_FA(**fake), caps=FakeCaps(case.get("missing_capabilities", ())))
                r3 = svc2.render(load_ir(ir_path), ir_path, resume=rr["job"]["id"])
                b = (r3.get("artifacts") or [{}])[0]
                if b.get("id") != a["id"] or b.get("jobs") != [rr["job"]["id"], r3["job"]["id"]]:
                    failures.append("resume did not reuse the artifact identity")
            if ar.get("revision_separation"):
                svc.reject(load_ir(ir_path), ir_path, [d["plan"]["steps"][0]["decision_id"]], reason="eval")
                svc.revise(load_ir(ir_path), ir_path); svc.approve(load_ir(ir_path), ir_path, ["all"])
                from fake_adapter import FakeAdapter as _FA
                r4 = Service(workspace=tmp, adapter=_FA(**fake), caps=FakeCaps(case.get("missing_capabilities", ()))).render(load_ir(ir_path), ir_path)
                b = (r4.get("artifacts") or [{}])[0]
                if b.get("id") == a["id"] or b.get("plan_id") == a["plan_id"] or not svc.artifact(a["id"])["integrity"]["ok"]:
                    failures.append("revision artifacts not separated or v1 damaged")
            if ar.get("deterministic_metadata"):
                keep = {k: a[k] for k in ("id", "logical_name", "type", "format", "name", "plan_id", "step_id", "decision_ids")}
                from fake_adapter import FakeAdapter as _FA
                r5 = Service(workspace=tmp, adapter=_FA(**fake), caps=FakeCaps(case.get("missing_capabilities", ()))).render(load_ir(ir_path), ir_path)
                b = (r5.get("artifacts") or [{}])[0]
                if {k: b.get(k) for k in keep} != keep:
                    failures.append("artifact metadata differs between two executions of the same plan")
        if ar.get("path_traversal"):
            st = svc.artifact_store()
            for bad in (str(Path(tmp) / ".." / "x.mp4"), "/etc/passwd", "rel.mp4"):
                try:
                    st.check_path(bad); failures.append(f"path accepted: {bad}")
                except ArtifactError:
                    pass
            if safe_filename("../../evil") != "evil" or safe_filename("CON") != "_CON":
                failures.append("unsafe file name accepted")
    if pp.get("deterministic"):
        ir2 = svc.plan([str(src)], case.get("profile", "generic"), request_text=case.get("request", ""), user_requirements=case.get("requirements"))
        shape = lambda x: [(st["skill"], st["params"].get("keep"), len(st["depends_on"])) for st in x.doc["plan"]["steps"]]  # noqa: E731
        if shape(ir) != shape(ir2):
            failures.append("plan shape differs between two runs on the same inputs")
    return {"case": case["name"], "ok": not failures, "failures": failures}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    args = ap.parse_args()
    results = []
    for p in sorted((ROOT / "evals" / "cases").glob("*.json")):
        case = json.loads(p.read_text(encoding="utf-8"))
        if args.case and case["name"] != args.case:
            continue
        results.append(run_case(case))
    for r in results:
        print(f"{'PASS' if r['ok'] else 'FAIL'} {r['case']}" + ("" if r["ok"] else "  " + "; ".join(r["failures"])))
    passed = sum(1 for r in results if r["ok"])
    print(f"{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
