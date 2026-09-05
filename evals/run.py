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


def _engine_checks(d: dict, subjects: dict, eng) -> list:
    """PR #16: every decision typed and grounded with a recorded basis; policy provenance; explain chain; IR invariants."""
    if not eng:
        return []
    from video_agent.agent.decision_engine import DECISION_TYPES, EXECUTABLE_TYPES, check_decisions
    failures = []
    for x in d["decisions"]:
        if x.get("type") not in DECISION_TYPES or not x.get("basis") or not x.get("evidence"):
            failures.append(f"decision {x['subject']} lacks type / basis / evidence")
        if x.get("basis", {}).get("approval", {}).get("resolved") != x["approval"] or x["basis"].get("risk", {}).get("independent_of_confidence") is not True:
            failures.append(f"decision {x['subject']}: basis does not match approval / risk")
    failures += [f"IR invariant: {e}" for e in check_decisions(d)]
    cited = {did for s in d["plan"]["steps"] for did in s["decision_ids"]}
    if any(x["id"] in cited and x["type"] not in EXECUTABLE_TYPES for x in d["decisions"]):
        failures.append("a non-executable decision type is cited by a plan step")
    for subj, want in (eng.get("basis") or {}).items():
        got = subjects.get(subj)
        if got is None:
            failures.append(f"missing decision {subj}")
            continue
        b = got["basis"]
        checks = {"approval_provenance": b["approval"].get("provenance"), "approval_key": b["approval"].get("key"), "served": b["intent"].get("served"),
                  "evidence_classes": b.get("evidence_classes"), "settings": {s["key"]: s["provenance"] for s in b.get("settings") or []}}
        for k, v in want.items():
            if k == "note_contains":
                if not any(v in n for n in b["approval"].get("notes") or []):
                    failures.append(f"{subj}: approval note lacks {v!r}: {b['approval'].get('notes')}")
            elif k == "settings":
                for key, prov in v.items():
                    if checks["settings"].get(key) != prov:
                        failures.append(f"{subj}: setting {key} provenance {checks['settings'].get(key)!r} != {prov!r}")
            elif checks.get(k) != v:
                failures.append(f"{subj}.basis.{k} {checks.get(k)!r} != {v!r}")
    if eng.get("explain"):
        info = Service.explain_decision(d, eng["explain"])[0]
        kinds = {r["kind"] for r in info["evidence"]}
        for k in eng.get("explain_kinds") or ("inference", "event", "observation", "asset"):
            if k not in kinds:
                failures.append(f"explain --decision {eng['explain']} lacks {k} in {sorted(kinds)}")
        bkinds = {r["kind"] for r in info["basis"]}
        for k in ("approval", "intent", "risk"):
            if k not in bkinds:
                failures.append(f"explain --decision basis lacks {k}")
        if "executable" in eng and info["executable"] != eng["executable"]:
            failures.append(f"explain executable {info['executable']} != {eng['executable']}")
        if json.dumps(info).count("argv") or "ffmpeg -" in json.dumps(info):
            failures.append("explain output carries command material")
    return failures


def run_case(case: dict) -> dict:
    tmp = tempfile.mkdtemp()
    fake = case.get("fake", {})
    src = Path(tmp) / "src" / "in.mp4"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"0" * 16)
    provider = FakeAIProvider(**case["ai"]) if case.get("ai") else None
    ma = case.get("media_analysis")
    failures = []
    exp = case["expect"]
    if ma:
        # external observation Skill (ADR-023) through the fake media-analysis process: contract discovery, lifting, provenance, failures
        import os
        from video_agent.tools import ToolRouter
        from video_agent.tools.media_analysis import ContractError, MediaAnalysisAdapter, MediaAnalysisSkill
        from video_agent.tools.base import ToolError
        os.environ.pop("FAKE_MA_MODE", None); os.environ.pop("FAKE_MA_CACHE", None)
        if ma.get("mode"):
            os.environ["FAKE_MA_MODE"] = ma["mode"]
        if ma.get("cache"):
            os.environ["FAKE_MA_CACHE"] = ma["cache"]
        skill = MediaAnalysisSkill([sys.executable, str(ROOT / "tests" / "fake_media_analysis.py")], None, {})
        try:
            ad = MediaAnalysisAdapter(skill, workspace=tmp)
        except ContractError as e:
            ad = None
            if exp.get("contract_refused") and (exp["contract_refused"] in str(e)):
                return {"case": case["name"], "ok": True, "failures": []}
            return {"case": case["name"], "ok": False, "failures": [f"contract refused: {e}"]}
        if exp.get("contract_refused"):
            return {"case": case["name"], "ok": False, "failures": ["an incompatible contract was accepted"]}
        adapters = [ad] if ma.get("only") else [FakeAdapter(**fake), ad]
        caps = FakeCaps(case.get("missing_capabilities", ()), extra=[] if ma.get("no_capability") else ["media-analysis"])
        svc = Service(workspace=tmp, adapter=ToolRouter(adapters), caps=caps, provider=provider)
        if ma.get("prefer", True):
            for name, tool in (("media_probe", "media-analysis/probe"), ("silence_analysis", "media-analysis/silence"), ("loudness_analysis", "media-analysis/loudness")):
                svc.registry.get(name).tools = [tool]
        tools = svc.tools_for()
        for sk in exp.get("unavailable_skills", []):
            if sk in tools:
                failures.append(f"skill {sk} unexpectedly available")
        for sk, tool in (exp.get("selected_tools") or {}).items():
            if tools.get(sk) != tool:
                failures.append(f"skill {sk} tool {tools.get(sk)} != {tool}")
        if exp.get("analysis_error"):
            from video_agent.media import AnalysisError
            try:
                svc.analyze([str(src)], case.get("profile", "generic"), kinds=ma.get("kinds"))
                failures.append("analysis succeeded although the Skill result is invalid")
            except (AnalysisError, RuntimeError) as e:
                if exp["analysis_error"] not in str(e):
                    failures.append(f"analysis error {e!s:.120} does not mention {exp['analysis_error']}")
            os.environ.pop("FAKE_MA_MODE", None); os.environ.pop("FAKE_MA_CACHE", None)
            return {"case": case["name"], "ok": not failures, "failures": failures}
        _, _, an = svc.analyze([str(src)], case.get("profile", "generic"), kinds=ma.get("kinds"))
        obs = {o.kind: o for o in an.observations}
        for kind, want in (exp.get("observations") or {}).items():
            o = obs.get(kind)
            if o is None:
                failures.append(f"missing observation {kind}")
                continue
            got = {"skill": o.skill, "skill_version": o.skill_version, "tool": o.tool, "provenance": o.provenance, "source": o.source, "cache_status": (o.cache or {}).get("status"),
                   "external": bool(o.external_id and o.external_id != o.id), "fingerprint": o.fingerprint}
            for k, v in want.items():
                if got.get(k) != v:
                    failures.append(f"observation {kind}.{k} {got.get(k)!r} != {v!r}")
        if exp.get("shared_fingerprint") and len({o.fingerprint for o in an.observations}) != 1:
            failures.append("observations of one asset carry different fingerprints")
        ev = sorted({e.type for e in an.timeline.events})
        for t in exp.get("events", []):
            if t not in ev:
                failures.append(f"missing event {t} in {ev}")
        for e in an.timeline.events:
            if any(k in json.dumps(e.to_dict()) for k in ("argv", "command", "ffmpeg ")):
                failures.append(f"event {e.type} carries command-like content")
        if "measure_calls" in exp and an.analyses[0]["budget"]["calls"] != exp["measure_calls"]:
            failures.append(f"measure calls {an.analyses[0]['budget']['calls']} != {exp['measure_calls']}")
        if exp.get("agent_cache_untouched") and (an.analyses[0]["cache"]["hits"] + an.analyses[0]["cache"]["misses"]):
            failures.append("the agent's own cache was consulted for Skill-owned measurements")
        os.environ.pop("FAKE_MA_MODE", None); os.environ.pop("FAKE_MA_CACHE", None)
        if not exp.get("intent") and not exp.get("production_plan") and not exp.get("decisions"):
            return {"case": case["name"], "ok": not failures, "failures": failures}
    ap = case.get("audio_production")
    if ap:
        # ADR-030: the audio production path (audio.production + audio.* requirements → decisions → plan → IR audio operations → compiler →
        # the fake audio-production process); requirement refusals, decision BLOCKs (conflict / no audio / impossible layout / resample /
        # preset), capability BLOCK (missing / UNKNOWN), contract drift BLOCK, execution failures never a success, provenance chain
        import os
        from video_agent.capabilities.resolver import Capability
        from video_agent.tools import ToolRouter
        from video_agent.tools.audio_production import AudioProductionAdapter, AudioProductionSkill, ContractError
        from video_agent.project import load_ir, save_ir
        os.environ.pop("FAKE_AP_MODE", None)
        inputs = []
        for n, spec in enumerate(ap.get("inputs") or [{"channels": 2}]):
            f = src.parent / (f"in{n}." + ("mp4" if spec.get("video") else "wav"))
            f.write_bytes(json.dumps({"fake": True, "duration": spec.get("duration", 16.0), "lufs": -11.0, "video": bool(spec.get("video", False)), "channels": spec.get("channels", 2)}).encode())
            inputs.append(str(f))
        if ap.get("contract_mode"):
            os.environ["FAKE_AP_MODE"] = ap["contract_mode"]
        skill = AudioProductionSkill([sys.executable, str(ROOT / "tests" / "fake_audio_production.py")], None, {})
        try:
            ad = AudioProductionAdapter(skill, workspace=tmp, allowed_inputs=[str(src.parent)], ffmpeg_skill_dir=tmp)
        except ContractError as e:
            os.environ.pop("FAKE_AP_MODE", None)
            if exp.get("contract_refused") and exp["contract_refused"] in str(e):
                return {"case": case["name"], "ok": True, "failures": []}
            return {"case": case["name"], "ok": False, "failures": [f"contract refused: {e!s:.200}"]}
        drift = ad.drift()
        os.environ.pop("FAKE_AP_MODE", None)
        extra = [] if ap.get("no_capability") else ["audio-production"] + [f"audio-production:{t}" for t in ("CUT", "NORMALIZE", "GAIN", "MONO", "STEREO", "DOWNMIX", "FADE_IN", "FADE_OUT", "CONCAT")]
        if drift:
            extra = []   # what the resolver does with a drifted contract: the capability is MISSING
        caps = FakeCaps(case.get("missing_capabilities", ()), extra=extra)
        if ap.get("unknown_capability"):
            base_caps = caps.resolve()
            class _Caps:
                def resolve(self, refresh=False):
                    c = dict(base_caps); c[ap["unknown_capability"]] = Capability(ap["unknown_capability"], "UNKNOWN", "doctor unknown", {}); return c
            caps = _Caps()
        fake_engine = FakeAdapter(**fake)
        svc = Service(workspace=tmp, adapter=ToolRouter([fake_engine, ad]), caps=caps, provider=provider)
        try:
            ir = svc.plan(inputs, case.get("profile", "generic"), user_requirements=case.get("requirements"))
        except ValueError as e:
            if exp.get("plan_error") and exp["plan_error"] in str(e):
                return {"case": case["name"], "ok": True, "failures": []}
            return {"case": case["name"], "ok": False, "failures": [f"plan refused: {e!s:.200}"]}
        if exp.get("plan_error"):
            return {"case": case["name"], "ok": False, "failures": ["an invalid request was planned"]}
        d = ir.doc
        types = [op["type"] for op in d["audio"]["operations"]]
        if "ir_ops" in exp and types != exp["ir_ops"]:
            failures.append(f"IR audio operations {types} != {exp['ir_ops']}")
        skills = [s["skill"] for s in d["plan"]["steps"]]
        if "steps" in exp and skills != exp["steps"]:
            failures.append(f"plan steps {skills} != {exp['steps']}")
        if "blocked" in exp and bool(ir.blocked()) != exp["blocked"]:
            failures.append(f"blocked {bool(ir.blocked())} != {exp['blocked']} ({[b['subject'] for b in ir.blocked()]})")
        if exp.get("blocked_subjects") and sorted({b["subject"] for b in ir.blocked()}) != sorted(exp["blocked_subjects"]):
            failures.append(f"blocked subjects {sorted({b['subject'] for b in ir.blocked()})} != {exp['blocked_subjects']}")
        if "plan_status" in exp and d["plan"]["status"] != exp["plan_status"]:
            failures.append(f"plan status {d['plan']['status']} != {exp['plan_status']}")
        rep = svc.validate(ir)
        if exp.get("validate_error"):
            if not any(exp["validate_error"] in e for e in rep.errors):
                failures.append(f"validator did not report {exp['validate_error']!r}: {rep.errors}")
        elif not rep.ok:
            failures.append(f"IR invalid: {rep.errors}")
        for s_ in d["plan"]["steps"]:
            if s_["skill"].startswith("audio_") and s_.get("tool") and s_["tool"] != "audio-production/run":
                failures.append(f"step {s_['id']} selected a non audio-production tool {s_['tool']}")
            if s_["skill"].startswith("audio_") and s_.get("tool") is None and not exp.get("blocked"):
                failures.append(f"step {s_['id']} has no tool")
        if exp.get("render"):
            if ap.get("run_mode"):
                os.environ["FAKE_AP_MODE"] = ap["run_mode"]
            ir_path = str(Path(tmp) / "ap.json"); save_ir(ir, ir_path)
            rr = svc.render(load_ir(ir_path), ir_path, approve=["all"])
            os.environ.pop("FAKE_AP_MODE", None)
            want = exp["render"]
            if want.get("status") and rr.get("status") != want["status"]:
                failures.append(f"render status {rr.get('status')} != {want['status']}")
            if want.get("execution_status") and (rr.get("execution") or {}).get("status") != want["execution_status"]:
                failures.append(f"execution status {(rr.get('execution') or {}).get('status')} != {want['execution_status']}")
            results = (rr.get("execution") or {}).get("results") or []
            used = [(r.get("data") or {}).get("operation_type") for r in results if r["tool"] == "audio-production/run"]
            if want.get("operations") is not None and used != want["operations"]:
                failures.append(f"audio-production operations {used} != {want['operations']}")
            if want.get("recovery_classes") is not None and [r["class"] for r in (rr.get("execution") or {}).get("recovery") or []] != want["recovery_classes"]:
                failures.append(f"recovery {[r['class'] for r in (rr.get('execution') or {}).get('recovery') or []]} != {want['recovery_classes']}")
            if want.get("qa_status") and (rr.get("qa") or {}).get("status") != want["qa_status"]:
                failures.append(f"qa {(rr.get('qa') or {}).get('status')} != {want['qa_status']}")
            if want.get("duration") is not None:
                item = next((i for i in (rr.get("qa") or {}).get("items") or [] if i["name"] == "duration"), None)
                if not item or abs(float(item["observed"]) - float(want["duration"])) > 0.01:
                    failures.append(f"delivered duration {item and item['observed']} != {want['duration']}")
            if any(o.tool.startswith("ffmpeg-skill/") and o.tool.split("/")[1] in ("cut", "loudness", "audio") and (o.args or {}).get("output") for o in fake_engine.calls):
                failures.append("the reference engine processed audio although the audio path was selected (fallback)")   # QA measurements (no output) are the engine's business
            if (rr.get("execution") or {}).get("status") == "COMPLETED":
                prov = json.loads((Path(tmp) / "jobs" / rr["job"]["id"] / "provenance.json").read_text())
                for e in prov["operations"]:
                    if e["tool"] == "audio-production/run":
                        blob = json.dumps(e["args"]).lower()
                        if any(k in blob for k in ('"argv"', '"command"', '"filter"', '"executable"', '"shell"', "ffmpeg -")):
                            failures.append("provenance args carry command material")
                        if not e.get("decision") or not (e.get("skill_result") or {}).get("artifact", {}).get("sha256"):
                            failures.append(f"provenance of {e['skill']} lacks the decision / the Skill's artifact hash")
                if want.get("skill_observations") is not None and [o["kind"] for o in prov.get("skill_observations") or []] != want["skill_observations"]:
                    failures.append(f"skill observations {[o['kind'] for o in prov.get('skill_observations') or []]}")
        return {"case": case["name"], "ok": not failures, "failures": failures}
    vo = case.get("video_editing_ops")
    if vo:
        # ADR-029: editing operations (concat / speed / resize / fit / fill / overlay) from explicit requirements through Decision → plan →
        # IR → compiler → the fake video-editing process; refusals (invalid parameter, path violation, conflicting or unsupported
        # request), capability gating, and output failures are never a success
        import os
        from video_agent.tools import ToolRouter
        from video_agent.tools.video_editing import VideoEditingAdapter, VideoEditingSkill
        from video_agent.project import load_ir, save_ir
        os.environ.pop("FAKE_VE_MODE", None)
        inputs = [str(src)]
        for n in range(1, int(vo.get("inputs", 1))):
            extra = src.parent / f"in{n}.mp4"; extra.write_bytes(b"0" * (16 + n)); inputs.append(str(extra))
        reqs = dict(case.get("requirements") or {})
        if vo.get("image"):
            img = Path(tmp) / ("outside" if vo["image"] == "outside" else "src") / "logo.png"
            img.parent.mkdir(parents=True, exist_ok=True); img.write_bytes(b"\x89PNG fake")
            reqs["edit.overlay"] = str(img) if vo["image"] != "traversal" else str(img.parent / ".." / "src" / "logo.png")
        skill = VideoEditingSkill([sys.executable, str(ROOT / "tests" / "fake_video_editing.py")], None, {})
        ad = VideoEditingAdapter(skill, workspace=tmp, allowed_inputs=[str(src.parent)], ffmpeg_skill_dir=tmp)
        caps = FakeCaps(case.get("missing_capabilities", ()), extra=[] if vo.get("no_capability") else ["video-editing", "encoder:aac"])
        fake_engine = FakeAdapter(**fake)
        svc = Service(workspace=tmp, adapter=ToolRouter([fake_engine, ad]), caps=caps, provider=provider)
        try:
            ir = svc.plan(inputs, case.get("profile", "generic"), user_requirements=reqs)
        except ValueError as e:
            if exp.get("plan_error") and exp["plan_error"] in str(e):
                return {"case": case["name"], "ok": True, "failures": []}
            return {"case": case["name"], "ok": False, "failures": [f"plan refused: {e!s:.200}"]}
        if exp.get("plan_error"):
            return {"case": case["name"], "ok": False, "failures": ["an invalid request was planned"]}
        d = ir.doc
        types = [op["type"] for op in d["video"]["operations"] if op["type"] != "video.trim"]
        if "ir_ops" in exp and types != exp["ir_ops"]:
            failures.append(f"IR operations {types} != {exp['ir_ops']}")
        if "blocked" in exp and bool(ir.blocked()) != exp["blocked"]:
            failures.append(f"blocked {bool(ir.blocked())} != {exp['blocked']} ({[b['subject'] for b in ir.blocked()]})")
        if exp.get("blocked_subjects") and sorted({b["subject"] for b in ir.blocked()}) != sorted(exp["blocked_subjects"]):
            failures.append(f"blocked subjects {sorted({b['subject'] for b in ir.blocked()})} != {exp['blocked_subjects']}")
        if "plan_status" in exp and d["plan"]["status"] != exp["plan_status"]:
            failures.append(f"plan status {d['plan']['status']} != {exp['plan_status']}")
        rep = svc.validate(ir)
        if exp.get("validate_error"):
            if not any(exp["validate_error"] in e for e in rep.errors):
                failures.append(f"validator did not report {exp['validate_error']!r}: {rep.errors}")
        elif not rep.ok:
            failures.append(f"IR invalid: {rep.errors}")
        for s_ in d["plan"]["steps"]:
            if s_["skill"].startswith("video_") and s_.get("tool") and not s_["tool"].startswith("video-editing/"):
                failures.append(f"step {s_['id']} selected a non video-editing tool {s_['tool']}")
        if exp.get("render"):
            if vo.get("run_mode"):
                os.environ["FAKE_VE_MODE"] = vo["run_mode"]
            ir_path = str(Path(tmp) / "vo.json"); save_ir(ir, ir_path)
            rr = svc.render(load_ir(ir_path), ir_path, approve=["all"])
            os.environ.pop("FAKE_VE_MODE", None)
            want = exp["render"]
            if want.get("status") and rr.get("status") != want["status"]:
                failures.append(f"render status {rr.get('status')} != {want['status']}")
            if want.get("execution_status") and (rr.get("execution") or {}).get("status") != want["execution_status"]:
                failures.append(f"execution status {(rr.get('execution') or {}).get('status')} != {want['execution_status']}")
            results = (rr.get("execution") or {}).get("results") or []
            used = [r["tool"] for r in results if r["tool"].startswith("video-editing/")]
            if want.get("tools") is not None and used != want["tools"]:
                failures.append(f"video-editing tools {used} != {want['tools']}")
            if want.get("recovery_classes") is not None and [r["class"] for r in (rr.get("execution") or {}).get("recovery") or []] != want["recovery_classes"]:
                failures.append(f"recovery {[r['class'] for r in (rr.get('execution') or {}).get('recovery') or []]} != {want['recovery_classes']}")
            if want.get("qa_status") and (rr.get("qa") or {}).get("status") != want["qa_status"]:
                failures.append(f"qa {(rr.get('qa') or {}).get('status')} != {want['qa_status']}")
            if want.get("duration") is not None:
                item = next((i for i in (rr.get("qa") or {}).get("items") or [] if i["name"] == "duration"), None)
                if not item or abs(float(item["observed"]) - float(want["duration"])) > 0.01:
                    failures.append(f"delivered duration {item and item['observed']} != {want['duration']}")
            if (rr.get("execution") or {}).get("status") == "COMPLETED":
                prov = json.loads((Path(tmp) / "jobs" / rr["job"]["id"] / "provenance.json").read_text())
                for e in prov["operations"]:
                    if e["tool"].startswith("video-editing/"):
                        blob = json.dumps(e["args"]).lower()
                        if any(k in blob for k in ('"argv"', '"command"', '"filter"', '"executable"', '"shell"', "ffmpeg -")):
                            failures.append(f"provenance args of {e['tool']} carry command material")
                        if not e.get("decision"):
                            failures.append(f"provenance of {e['tool']} cites no decision")
                if want.get("sources") is not None and sorted(len(a["source"]) for a in rr.get("artifacts") or []) != sorted(want["sources"]):
                    failures.append(f"artifact sources {[a['source'] for a in rr.get('artifacts') or []]}")
        return {"case": case["name"], "ok": not failures, "failures": failures}
    ve = case.get("video_editing")
    if ve:
        # PR #18 (ADR-028): video-editing-skill behind its CLI (fake process): selection by registry / capability, lowering, execution, provenance, failures
        import os
        from video_agent.tools import ToolRouter
        from video_agent.tools.video_editing import ContractError, VideoEditingAdapter, VideoEditingSkill
        from video_agent.project import load_ir, save_ir
        os.environ.pop("FAKE_VE_MODE", None)
        if ve.get("contract_mode"):
            os.environ["FAKE_VE_MODE"] = ve["contract_mode"]
        skill = VideoEditingSkill([sys.executable, str(ROOT / "tests" / "fake_video_editing.py")], None, {})
        try:
            ad = VideoEditingAdapter(skill, workspace=tmp, allowed_inputs=[str(src.parent)], ffmpeg_skill_dir=tmp)
        except ContractError as e:
            os.environ.pop("FAKE_VE_MODE", None)
            if exp.get("contract_refused") and exp["contract_refused"] in str(e):
                return {"case": case["name"], "ok": True, "failures": []}
            return {"case": case["name"], "ok": False, "failures": [f"contract refused: {e}"]}
        if exp.get("contract_refused"):
            return {"case": case["name"], "ok": False, "failures": ["an incompatible contract was accepted"]}
        caps = FakeCaps(case.get("missing_capabilities", ()), extra=[] if ve.get("no_capability") else ["video-editing"])
        fake_engine = FakeAdapter(**fake)
        svc = Service(workspace=tmp, adapter=ToolRouter([fake_engine, ad]), caps=caps, provider=provider)
        if ve.get("prefer", True):
            svc.registry.get("silence_cleanup").tools = ["video-editing/cut", "ffmpeg-skill/cut"]
        ir = svc.plan([str(src)], case.get("profile", "generic"), user_requirements=case.get("requirements"))
        d = ir.doc
        step = next((s for s in d["plan"]["steps"] if s["skill"] == "silence_cleanup"), None)
        if "selected_tool" in exp and (step or {}).get("tool") != exp["selected_tool"]:
            failures.append(f"silence_cleanup tool {(step or {}).get('tool')} != {exp['selected_tool']}")
        if "blocked" in exp and bool(ir.blocked()) != exp["blocked"]:
            failures.append(f"blocked {bool(ir.blocked())} != {exp['blocked']}")
        if not svc.validate(ir).ok:
            failures.append(f"IR invalid: {svc.validate(ir).errors}")
        if ve.get("run_mode"):
            os.environ["FAKE_VE_MODE"] = ve["run_mode"]
        if exp.get("render"):
            ir_path = str(Path(tmp) / "ve.json"); save_ir(ir, ir_path)
            rr = svc.render(load_ir(ir_path), ir_path, approve=["all"])
            want = exp["render"]
            if want.get("execution_status") and (rr.get("execution") or {}).get("status") != want["execution_status"]:
                failures.append(f"execution status {(rr.get('execution') or {}).get('status')} != {want['execution_status']}")
            results = (rr.get("execution") or {}).get("results") or []
            cut = next((r for r in results if r["tool"] == "video-editing/cut"), None)
            if want.get("cut_ok") is not None and bool(cut and cut["ok"]) != want["cut_ok"]:
                failures.append(f"video-editing/cut ok {bool(cut and cut['ok'])} != {want['cut_ok']}")
            if cut and cut["ok"]:
                if not cut["data"].get("artifact", {}).get("sha256") or cut["data"].get("observation", {}).get("provenance") != "OBSERVED" or not cut["data"].get("timeline"):
                    failures.append("artifact / observation / timeline not mapped from the Skill response")
                if sorted(cut["data"]["operation"]["parameters"]) != ["keep", "precision"]:
                    failures.append("lowering did not produce the contract's CUT parameters")
            if want.get("recovery_classes") is not None and [r["class"] for r in (rr.get("execution") or {}).get("recovery") or []] != want["recovery_classes"]:
                failures.append(f"recovery {[r['class'] for r in (rr.get('execution') or {}).get('recovery') or []]} != {want['recovery_classes']}")
            if any(o.tool == "ffmpeg-skill/cut" for o in fake_engine.calls):
                failures.append("the reference cut ran although video-editing was the selected tool (fallback)")
            if cut is not None and (rr.get("execution") or {}).get("status") == "COMPLETED":
                prov = json.loads((Path(tmp) / "jobs" / rr["job"]["id"] / "provenance.json").read_text())
                trim = next(e for e in prov["operations"] if e["skill"] == "silence_cleanup")
                if trim["skill_package"] != "video-editing" or not (trim.get("skill_result") or {}).get("artifact"):
                    failures.append("provenance lacks the Skill's package / result facts")
                blob = json.dumps(trim["args"])
                if any(k in blob for k in ("argv", "command", "filter", "ffmpeg -")):
                    failures.append("provenance args carry command material")
        os.environ.pop("FAKE_VE_MODE", None)
        return {"case": case["name"], "ok": not failures, "failures": failures}
    ts = case.get("transcription")
    if ts:
        # external recognition Skill (ADR-024) through the fake transcription process: contract, lifting, provenance, SpeechEvents, refusals
        import os
        from video_agent.tools import ToolRouter
        from video_agent.tools.transcription import ContractError, TranscriptionAdapter, TranscriptionSkill
        for k in ("FAKE_TS_MODE", "FAKE_TS_CACHE"):
            os.environ.pop(k, None)
        if ts.get("mode"):
            os.environ["FAKE_TS_MODE"] = ts["mode"]
        if ts.get("cache"):
            os.environ["FAKE_TS_CACHE"] = ts["cache"]
        if ts.get("segments"):
            os.environ["FAKE_TS_SEGMENTS"] = json.dumps(ts["segments"])
        skill = TranscriptionSkill([sys.executable, str(ROOT / "tests" / "fake_transcription.py")], None, {})
        roots = [str(src.parent)] if ts.get("roots", True) else None
        try:
            ad = TranscriptionAdapter(skill, workspace=str(Path(tmp) / "cache" / "transcription"), allowed_inputs=roots, offline=bool(ts.get("offline")))
        except ContractError as e:
            os.environ.pop("FAKE_TS_MODE", None)
            if exp.get("contract_refused") and exp["contract_refused"] in str(e):
                return {"case": case["name"], "ok": True, "failures": []}
            return {"case": case["name"], "ok": False, "failures": [f"contract refused: {e}"]}
        if exp.get("contract_refused"):
            return {"case": case["name"], "ok": False, "failures": ["an incompatible contract was accepted"]}
        caps = FakeCaps(case.get("missing_capabilities", ()), extra=[] if ts.get("no_capability") else ["transcription"])
        svc = Service(workspace=tmp, adapter=ToolRouter([FakeAdapter(**fake), ad]), caps=caps, provider=provider)
        tools = svc.tools_for()
        for sk in exp.get("unavailable_skills", []):
            if sk in tools:
                failures.append(f"skill {sk} unexpectedly available")
        for sk, tool in (exp.get("selected_tools") or {}).items():
            if tools.get(sk) != tool:
                failures.append(f"skill {sk} tool {tools.get(sk)} != {tool}")
        if exp.get("path_refused"):
            outside = Path(tmp) / "outside" / "x.mp4"
            outside.parent.mkdir(exist_ok=True); outside.write_bytes(b"0" * 16)
            link = src.parent / "link.mp4"
            try:
                link.symlink_to(outside)
                targets = {"outside_allowed_roots": str(outside), "symlink_escape": str(link), "traversal": str(src.parent / ".." / "outside" / "x.mp4")}
            except OSError:
                targets = {"outside_allowed_roots": str(outside), "traversal": str(src.parent / ".." / "outside" / "x.mp4")}
            for reason, path in targets.items():
                r = ad.measure("transcription/transcribe", {"input": path, "asset_id": "a"})
                if r.ok or r.data["error"]["code"] != "INVALID_INPUT" or reason not in r.data["error"]["message"]:
                    failures.append(f"{reason}: not refused ({r.data.get('error')})")
            if not ad.measure("transcription/transcribe", {"input": str(src), "asset_id": "a"}).ok:
                failures.append("input inside the allowed root was refused")
            os.environ.pop("FAKE_TS_MODE", None)
            return {"case": case["name"], "ok": not failures, "failures": failures}
        kinds = ts.get("kinds") or ["transcript"]
        params = ts.get("params") or {}
        if exp.get("analysis_error") or exp.get("transcript_failed"):
            try:
                _, _, an = svc.analyze([str(src)], case.get("profile", "generic"), kinds=kinds, params=params)
            except (RuntimeError, Exception) as e:   # noqa: BLE001 — the eval records what happened
                if exp.get("analysis_error") and exp["analysis_error"] in str(e):
                    os.environ.pop("FAKE_TS_MODE", None)
                    return {"case": case["name"], "ok": True, "failures": []}
                os.environ.pop("FAKE_TS_MODE", None)
                return {"case": case["name"], "ok": False, "failures": [f"unexpected error {e!s:.160}"]}
            row = next((r for r in an.analyses[0]["rows"] if r["kind"] == "transcript"), None)
            tf = exp.get("transcript_failed") or {}
            if not row or row["status"] != "FAILED":
                failures.append(f"transcript row not FAILED: {row}")
            elif tf.get("kind") and row["error"]["kind"] != tf["kind"]:
                failures.append(f"failure kind {row['error']['kind']} != {tf['kind']}")
            elif tf.get("skill_error") and row["error"].get("skill_error") != tf["skill_error"]:
                failures.append(f"skill error {row['error'].get('skill_error')} != {tf['skill_error']}")
            elif tf.get("availability") and (row["error"].get("skill_details") or {}).get("availability") != tf["availability"]:
                failures.append(f"model availability {(row['error'].get('skill_details') or {}).get('availability')} != {tf['availability']}")
            if any(o.kind == "transcript" for o in an.observations) or an.timeline.query(type="SPEECH"):
                failures.append("a failed recognition still produced a transcript observation or SpeechEvents")
            os.environ.pop("FAKE_TS_MODE", None)
            return {"case": case["name"], "ok": not failures, "failures": failures}
        cx = exp.get("context")
        if cx:
            # PR #15: situations from events, generic inference, provenance; never a decision / step / command by themselves
            from video_agent.context import build_contexts, contexts_at
            ir = svc.plan([str(src)], case.get("profile", "generic"), kinds=kinds, params=params)
            d = ir.doc
            ctxs = d["analysis"]["contexts"]
            if "min_contexts" in cx and len(ctxs) < cx["min_contexts"]:
                failures.append(f"contexts {len(ctxs)} < {cx['min_contexts']}")
            for at, sig in (cx.get("at") or {}).items():
                got = [c for c in Service.contexts_of(d) if c.scope["start"] - 1e-6 <= float(at) < c.scope["end"] - 1e-6]
                if len(got) != 1 or got[0].signature != sig:
                    failures.append(f"situation at {at}: {[g.signature for g in got]} != {sig}")
            if any(c["provenance"] != "DERIVED" for c in ctxs):
                failures.append("a context is not DERIVED")
            infs = d["analysis"]["inferences"]
            kinds_count = {}
            for i in infs:
                kinds_count[i["kind"]] = kinds_count.get(i["kind"], 0) + 1
            for k, n in (cx.get("inferences") or {}).items():
                if kinds_count.get(k, 0) != n:
                    failures.append(f"inference {k}: {kinds_count.get(k, 0)} != {n}")
            ev_ids = {e["id"] for e in d["timeline"]["events"]}
            for i in infs:
                if i["kind"] in ("source_activity", "source_inactivity", "transition", "conflict"):
                    if not i["evidence"] or not set(i["evidence"]) <= ev_ids or i["provenance"] != "INFERRED" or i["data"].get("generator") != "context_inference@1.0":
                        failures.append(f"generic inference {i['id']} lacks event evidence / provenance / generator")
            generic_ids = {i["id"] for i in infs if i["kind"] in ("source_activity", "source_inactivity", "transition")}
            if any(set(x["evidence"]) & generic_ids for x in d["decisions"]):
                failures.append("a decision rests directly on a situation inference")
            blob = json.dumps({"plan": d["plan"], "video": d["video"], "audio": d["audio"], "delivery": d["delivery"]})
            if any(k in blob for k in ("ctx_", "SPEECH", "transition", "argv", "command")):
                failures.append("context / event material reached the plan or operations")
            if cx.get("conflict_untouched"):
                conf = [i for i in infs if i.kind == "conflict"] if False else [i for i in infs if i["kind"] == "conflict"]
                if not conf:
                    failures.append("expected a conflict")
                for c in conf:
                    for eid in c["evidence"]:
                        e = next(x for x in d["timeline"]["events"] if x["id"] == eid)
                        if e["provenance"] != "OBSERVED":
                            failures.append("a conflicting event was rewritten")
            if not svc.validate(ir).ok:
                failures.append(f"IR invalid: {svc.validate(ir).errors}")
            if cx.get("explain"):
                target = next((c for c in ctxs if c["scope"]["start"] == cx["explain"]), None)
                if not target:
                    failures.append("explain target context missing")
                else:
                    kinds_seen = {r["kind"] for r in Service.explain_context(d, target["id"])["chain"]}
                    for k in ("context", "track", "event", "observation"):
                        if k not in kinds_seen:
                            failures.append(f"explain --context lacks {k}")
            if cx.get("deterministic"):
                ir2 = svc.plan([str(src)], case.get("profile", "generic"), kinds=kinds, params=params)
                sig = lambda doc: [(c["scope"]["start"], c["scope"]["end"], sorted(t["event_type"] + "/" + t["subtype"] for t in c["tracks"])) for c in doc["analysis"]["contexts"]]  # noqa: E731
                if sig(d) != sig(ir2.doc):
                    failures.append("contexts differ between two plans of the same media")
            for k in ("FAKE_TS_MODE", "FAKE_TS_CACHE", "FAKE_TS_SEGMENTS"):
                os.environ.pop(k, None)
            return {"case": case["name"], "ok": not failures, "failures": failures}
        sp_exp = exp.get("speech")
        if sp_exp:
            # PR #14: SpeechEvent → Inference → Decision → ProductionPlan → IR, reviewable and traceable; never AUTO for a removal
            from video_agent.agent.production_plan import executable_steps, explain_step
            from video_agent.project import load_ir, save_ir
            ir = svc.plan([str(src)], case.get("profile", "generic"), kinds=kinds, params=params, user_requirements=case.get("requirements"))
            d = ir.doc
            infs = d["analysis"]["inferences"]
            by_kind = {}
            for i in infs:
                by_kind.setdefault(i["kind"], []).append(i)
            for kind, n in (sp_exp.get("inferences") or {}).items():
                if len(by_kind.get(kind, [])) != n:
                    failures.append(f"inference {kind}: {len(by_kind.get(kind, []))} != {n}")
            if any(i["data"].get("speaker_id") is not None for i in infs) or any(i["provenance"] != "INFERRED" for i in infs if i["kind"].startswith(("speech_", "internal_silence_removable"))):
                failures.append("speech inference carries a speaker id or a non-INFERRED provenance")
            cands = [x for x in d["decisions"] if x["subject"].startswith("silence.internal.")]
            if "candidates" in sp_exp and len(cands) != sp_exp["candidates"]:
                failures.append(f"removal candidates {len(cands)} != {sp_exp['candidates']}")
            if any(x["approval"] == "AUTO" for x in cands):
                failures.append("a removal candidate is AUTO")
            for x in cands:
                if x["approval"] != sp_exp.get("candidate_approval", "CONFIRM"):
                    failures.append(f"candidate approval {x['approval']} != {sp_exp.get('candidate_approval', 'CONFIRM')}")
            if "plan_status" in sp_exp and d["plan"]["status"] != sp_exp["plan_status"]:
                failures.append(f"plan status {d['plan']['status']} != {sp_exp['plan_status']}")
            trim = next((s for s in d["plan"]["steps"] if s["skill"] == "silence_cleanup"), None)
            if "keep" in sp_exp and (trim or {}).get("params", {}).get("keep") != sp_exp["keep"]:
                failures.append(f"trim keep {(trim or {}).get('params', {}).get('keep')} != {sp_exp['keep']}")
            if trim and cands and trim["id"] in executable_steps(d):
                failures.append("the trim executes while a removal candidate is unconfirmed")
            if not svc.validate(ir).ok:
                failures.append(f"IR invalid: {svc.validate(ir).errors}")
            if trim and sp_exp.get("chain"):
                want = {("decision", "silence.internal."), ("inference", "candidate"), ("event", "SpeechEvent"), ("event", "AudioEvent"), ("observation", "transcript")}
                for k, frag in want:
                    if not any(kk == k and frag in det for kk, det in {(r["kind"], r.get("detail") or "") for r in explain_step(d, trim["id"])["chain"]}):
                        failures.append(f"explain chain lacks {k} {frag!r}")
            for x in d["decisions"]:
                if x["subject"] in sp_exp.get("confirm_subjects", []) and x["approval"] != "CONFIRM":
                    failures.append(f"{x['subject']} should be CONFIRM, is {x['approval']}")
            for subj, typ in (sp_exp.get("types") or {}).items():
                hit = [x for x in d["decisions"] if x["subject"] == subj or x["subject"].startswith(subj)]
                if not hit or any(x["type"] != typ for x in hit):
                    failures.append(f"{subj} type {[x.get('type') for x in hit]} != {typ}")
            failures += _engine_checks(d, {x["subject"]: x for x in d["decisions"]}, sp_exp.get("engine"))
            for c in cands:
                if "candidate_status" in sp_exp and c["status"] != sp_exp["candidate_status"]:
                    failures.append(f"candidate status {c['status']} != {sp_exp['candidate_status']}")
                if "candidate_approval_provenance" in sp_exp and c["basis"]["approval"]["provenance"] != sp_exp["candidate_approval_provenance"]:
                    failures.append(f"candidate approval provenance {c['basis']['approval']['provenance']} != {sp_exp['candidate_approval_provenance']}")
            if sp_exp.get("conflict_decision"):
                cd = next((x for x in d["decisions"] if x["subject"] == sp_exp["conflict_decision"]), None)
                if not cd or cd["approval"] != "CONFIRM" or cd["type"] != "KEEP" or "never overridden" not in cd["reason"]:
                    failures.append(f"constraint conflict decision {sp_exp['conflict_decision']} missing or not CONFIRM/KEEP with reason")
            if sp_exp.get("render_blocked"):
                ir_path = str(Path(tmp) / "blk.json"); save_ir(ir, ir_path)
                rr = svc.render(load_ir(ir_path), ir_path, approve=["all"])
                if rr["status"] != "BLOCKED" or rr.get("execution"):
                    failures.append(f"a BLOCK policy still rendered: {rr['status']}")
            blob = json.dumps({"plan": d["plan"], "video": d["video"]})
            if any(k in blob for k in ("SPEECH", "speaker", "transcription", "argv", "command")):
                failures.append("plan / operations carry event, speaker, transcript or command material")
            if sp_exp.get("approve_then") or sp_exp.get("reject_then"):
                ir_path = str(Path(tmp) / "sp.json"); save_ir(ir, ir_path)
                if sp_exp.get("approve_then"):
                    svc.approve(load_ir(ir_path), ir_path, ["all"])
                    st = load_ir(ir_path).doc["plan"]["status"]
                    if st != sp_exp["approve_then"]:
                        failures.append(f"after approve: {st} != {sp_exp['approve_then']}")
                if sp_exp.get("reject_then") and cands:
                    svc.reject(load_ir(ir_path), ir_path, [cands[0]["id"]], reason="eval")
                    r = load_ir(ir_path)
                    if r.doc["plan"]["status"] != "REJECTED" or svc.render(r, ir_path)["status"] != "BLOCKED":
                        failures.append("a rejected candidate still renders")
                    svc.revise(load_ir(ir_path), ir_path)
                    v2 = load_ir(ir_path)
                    if [x for x in v2.doc["decisions"] if x["subject"].startswith("silence.internal.") and x["status"] != "REJECTED"]:
                        failures.append("the rejected candidate was proposed again")
                    t2 = next((s for s in v2.doc["plan"]["steps"] if s["skill"] == "silence_cleanup"), None)
                    if sp_exp["reject_then"].get("keep") is not None and (t2 or {}).get("params", {}).get("keep") != sp_exp["reject_then"]["keep"]:
                        failures.append(f"after revise keep {(t2 or {}).get('params', {}).get('keep')} != {sp_exp['reject_then']['keep']}")
            for k in ("FAKE_TS_MODE", "FAKE_TS_CACHE", "FAKE_TS_SEGMENTS"):
                os.environ.pop(k, None)
            return {"case": case["name"], "ok": not failures, "failures": failures}
        _, _, an = svc.analyze([str(src)], case.get("profile", "generic"), kinds=kinds, params=params)
        t = next((o for o in an.observations if o.kind == "transcript"), None)
        if t is None:
            failures.append("no transcript observation")
        else:
            got = {"skill": t.skill, "skill_version": t.skill_version, "tool": t.tool, "provenance": t.provenance, "source": t.source, "cache_status": (t.cache or {}).get("status"),
                   "cache_owner": (t.cache or {}).get("owner"), "external": bool(t.external_id and t.external_id != t.id and t.external_id == t.data.get("id")),
                   "engine": t.parameters.get("engine"), "execution_mode": t.parameters.get("execution_mode"), "model": t.parameters.get("model"),
                   "language": t.data.get("language"), "segments": len(t.data.get("segments") or []), "schema": t.data.get("schema")}
            for k, v in (exp.get("transcript") or {}).items():
                if got.get(k) != v:
                    failures.append(f"transcript.{k} {got.get(k)!r} != {v!r}")
            if exp.get("shared_asset") and not (t.asset_id == an.assets[0].id and t.fingerprint == an.assets[0].hash and len(an.assets) == 1):
                failures.append("transcript does not share the asset identity (asset id / fingerprint)")
            if exp.get("speaker_null") and any(s.get("speaker_id") is not None for s in t.data.get("segments") or []):
                failures.append("a segment carries a speaker id")
            sp = an.timeline.query(type="SPEECH")
            if "speech_events" in exp and len(sp) != exp["speech_events"]:
                failures.append(f"speech events {len(sp)} != {exp['speech_events']}")
            for e in sp:
                if e.provenance != "OBSERVED" or e.evidence != [t.id] or e.metadata.get("speaker_id") is not None or e.event_type != "SpeechEvent":
                    failures.append(f"speech event {e.id} malformed: {e.provenance} {e.evidence} {e.metadata.get('speaker_id')} {e.event_type}")
                if any(k in json.dumps(e.to_dict()) for k in ("argv", "command", "ffmpeg ", "speaker_name", "camera")):
                    failures.append(f"speech event {e.id} carries command / identity content")
            if exp.get("agent_cache_untouched") and any(r.get("kind") == "transcript" and (r.get("produced_by") or r.get("cache_owner") != "transcription") for r in an.analyses[0]["rows"]):
                failures.append("the agent's own cache took part in recognition")
        if exp.get("no_speech_decisions"):
            ir = svc.plan([str(src)], case.get("profile", "generic"), kinds=kinds, params=params)
            d = ir.doc
            sp_ids = {e["id"] for e in d["timeline"]["events"] if e["type"] == "SPEECH"}
            blob = json.dumps({"plan": d["plan"], "video": d["video"], "audio": d["audio"], "delivery": d["delivery"]})
            if any(x in blob for x in ("SPEECH", "transcription", "speaker")) or any(i in blob for i in sp_ids):
                failures.append("SpeechEvents / transcript / speaker material reached the plan or its operations")
            reasoning = json.dumps({"decisions": d["decisions"], "inferences": d["analysis"]["inferences"]})
            if "speaker_name" in reasoning or "camera" in reasoning or any(i.get("data", {}).get("speaker_id") is not None for i in d["analysis"]["inferences"]):
                failures.append("speaker identity appeared in inferences / decisions")
            if not svc.validate(ir).ok:
                failures.append(f"IR with transcript is invalid: {svc.validate(ir).errors}")
        os.environ.pop("FAKE_TS_MODE", None); os.environ.pop("FAKE_TS_CACHE", None)
        return {"case": case["name"], "ok": not failures, "failures": failures}
    else:
        svc = Service(workspace=tmp, adapter=FakeAdapter(**fake), caps=FakeCaps(case.get("missing_capabilities", ())), provider=provider)
    ir = svc.plan([str(src)], case.get("profile", "generic"), request_text=case.get("request", ""), user_requirements=case.get("requirements"))
    d = ir.doc
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
    failures += _engine_checks(d, subjects, exp.get("engine"))
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
