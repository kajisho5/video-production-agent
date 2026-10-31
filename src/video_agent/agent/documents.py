"""Document verification (ADR-095; artifacts-skill roadmap step 5, agent side: adapter + SkillSpec document_verify): a document that goes out
with the video (a script, a caption script as a document, a report, slides) checked by artifacts-skill before the job is called done.

    Requirement (`--set document.verify=<file>` [+ `document.verify.preset=<name>`]) → DELIVER `document.verify` (AUTO: a read-only
    measurement) → IR `qa.documents[] {id, path, name, sha256, preset?, decision_ids}` → step `document_verify` (artifacts-skill/verify,
    the policy forbidding leftover placeholder text) → QA items (layer document): a FAIL check fails QA, so the job ends in REVIEW

- **Requirements** (`--set` only; refused before analysis): `document.verify` -- an existing file the plan's inputs do not already name;
  `document.verify.preset` -- one of artifacts-skill's named policies (print-a4, print-letter, slides-16x9, spreadsheet-no-cached-errors,
  web-no-external), only with the document. Caption files (.srt / .vtt / .ass) are subtitle-skill's and qc-skill's (artifacts-skill does
  not read them; its step 4) and are refused here with that pointer.
- **Decision**: DELIVER `document.verify` (risk LOW, approval `document.verify.approval` AUTO: nothing is written, the document is read only).
  The document's sha256 is recorded at plan time; QA fails a document that changed since (the plan verified another file).
  Without artifacts-skill the step is handed off naming it (ADR-051)."""
from __future__ import annotations

import hashlib
import os
from typing import Any, Callable, Dict, List, Optional

from ..models import Requirement
from ..tools.artifacts.adapter import PRESETS
from .editing import EditRequirementError

PREFIX = "document."
SWITCH = "document.verify"
PRESET_KEY = "document.verify.preset"
REQUIREMENT_KEYS = (SWITCH, PRESET_KEY)
SUBJECT = "document.verify"
SKILL = "document_verify"
TOOL_NAME = "verify"
APPROVAL_KEY = ("document.verify.approval", "AUTO")
DOCUMENT_ID = "document_1"
CAPTION_EXTENSIONS = (".srt", ".vtt", ".ass", ".ssa", ".scc")
IR_KEYS = ("id", "path", "name", "sha256", "preset", "decision_ids")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_document_requirements(values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """{path, preset?} or None; EditRequirementError on anything refused."""
    given = {k: v for k, v in (values or {}).items() if k.startswith(PREFIX) and v is not None}
    unknown = sorted(k for k in given if k not in REQUIREMENT_KEYS)
    if unknown:
        raise EditRequirementError(f"unknown requirement key(s): {', '.join(unknown)}; known: {', '.join(REQUIREMENT_KEYS)}")
    if not given:
        return None
    path = given.get(SWITCH)
    if path is None:
        raise EditRequirementError(f"{PRESET_KEY} refines {SWITCH}, which is not set")
    if not isinstance(path, str) or not path.strip() or not os.path.isfile(path):
        raise EditRequirementError(f"{SWITCH} must name an existing document file, got {path!r}")
    if os.path.splitext(path)[1].lower() in CAPTION_EXTENSIONS:
        raise EditRequirementError(f"{SWITCH}: {os.path.basename(path)} is a caption file; caption files are checked by subtitle-skill / qc-skill "
                                   "(--set qc=true on a plan with subtitles), not by artifacts-skill")
    out: Dict[str, Any] = {"path": os.path.abspath(path)}
    if PRESET_KEY in given:
        if given[PRESET_KEY] not in PRESETS:
            raise EditRequirementError(f"{PRESET_KEY} must be one of {', '.join(PRESETS)}, got {given[PRESET_KEY]!r}")
        out["preset"] = given[PRESET_KEY]
    return out


def check_document_keys(user_requirements: Dict[str, Any]) -> None:
    parse_document_requirements(user_requirements)


def decide_documents(eng: Any, m: Dict[str, Requirement], approval_for: Callable, cap_block: Callable[[str, str], Any]) -> None:
    reqs = [r for k, r in sorted(m.items()) if k.startswith(PREFIX) and r.provenance == "USER"]
    if not reqs:
        return
    p = parse_document_requirements({r.key: r.value for r in reqs})
    if p is None:
        return
    name = os.path.basename(p["path"])
    eng.decide(subject=SUBJECT, type="DELIVER", decision=f"verify the document {name} before the job is done" + (f" (policy {p['preset']})" if p.get("preset") else ""),
               reason=f"user asked for the document to be verified ({reqs[0].source}); artifacts-skill checks its structure read-only with a policy that "
                      "forbids leftover placeholder text (Lorem ipsum, Click to add title, TODO ...); a FAIL check fails QA and the job ends in REVIEW",
               confidence=1.0, evidence=[r.id for r in reqs], risk="LOW", approval=approval_for(SUBJECT), provenance="USER",
               params={"asset_id": DOCUMENT_ID, "path": p["path"], "name": name, "sha256": _sha256(p["path"]), **({"preset": p["preset"]} if p.get("preset") else {})},
               requirements=reqs, serves_intent=None)
    cap_block(SKILL, f"capability.{SKILL}")


def document_problems(entry: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    extra = sorted(k for k in entry if k not in IR_KEYS)
    if extra:
        errs.append(f"qa.documents: keys {extra} are not in the vocabulary")
    if not entry.get("id") or not os.path.isabs(str(entry.get("path") or "")) or not entry.get("sha256"):
        errs.append("qa.documents: an entry needs id, an absolute path and the sha256 recorded at plan time")
    if entry.get("preset") is not None and entry["preset"] not in PRESETS:
        errs.append(f"qa.documents: preset {entry['preset']!r} is not one of {', '.join(PRESETS)}")
    return errs
