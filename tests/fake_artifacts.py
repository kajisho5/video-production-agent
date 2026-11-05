"""Fake artifacts-skill process for the ADR-095 tests: speaks the slice of the real `artifacts-skill` CLI the agent uses -- `contract --json`
(the pinned src/video_agent/tools/artifacts/contract_0.3.0.json), `doctor --json` ({tool_version, capabilities: {<format>.structural: {status,
detail}}}) and `verify --json --policy=<json> [--policy-preset=NAME] -- <document>` ({kind structural, status, checks}; exit 0 pass / warn,
1 fail, 3 unknown; an error as {"error": {code, category, message, remediation}} with its category's exit code). The document's text is read:
"TODO", "Lorem ipsum" or "Click to add" makes `leftover_placeholder_text` fail when the policy forbids placeholders; a `.bin` file is
ARTIFACT_TYPE_UNSUPPORTED. Nothing is rendered or written. Controlled via:

  FAKE_AF_READY: all (default) | none (no format's structural verification available)
  FAKE_AF_MODE: ok | bad_kind | wrong_exit | mutate | fail:<CODE>:<category>
  FAKE_AF_CALLS: a file every invocation's argv is appended to (one JSON list per line). Test double only."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MODE = os.environ.get("FAKE_AF_MODE", "ok")
READY = os.environ.get("FAKE_AF_READY", "all")
CALLS_LOG = os.environ.get("FAKE_AF_CALLS")
PINNED = Path(__file__).resolve().parents[1] / "src" / "video_agent" / "tools" / "artifacts" / "contract_0.3.0.json"
EXIT = {"input": 2, "capability": 3, "security": 4, "execution": 1, "verification": 1, "internal": 5}
MARKERS = ("todo", "lorem ipsum", "click to add")


def _emit(doc) -> None:
    sys.stdout.write(json.dumps(doc, indent=2) + "\n")


def _error(code: str, category: str, message: str) -> int:
    _emit({"error": {"code": code, "category": category, "message": message, "remediation": "fake remediation", "evidence": {}}})
    return EXIT[category]


def _doctor() -> int:
    st = "available" if READY == "all" else "missing"
    _emit({"tool_version": "0.3.0", "capabilities": {f"{f}.structural": {"id": f"{f}.structural", "status": st, "detail": "fake"} for f in ("markdown", "pdf", "docx")}
           | {"backend.ffmpeg": {"id": "backend.ffmpeg", "status": "available", "detail": "fake"}}})
    return 0


def _verify(argv) -> int:
    opts, pos, after = {}, [], False
    for a in argv:
        if after:
            pos.append(a)
        elif a == "--":
            after = True
        elif a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            opts[k] = v
        elif a.startswith("--"):
            opts[a[2:]] = True
    if MODE.startswith("fail:"):
        _, code, cat = MODE.split(":", 2)
        return _error(code, cat, "refused (fake)")
    path = Path(pos[0])
    if not path.is_file():
        return _error("ARTIFACT_INPUT_NOT_FOUND", "input", f"Input file does not exist or is not a regular file: {pos[0]}")
    if path.suffix == ".bin":
        return _error("ARTIFACT_TYPE_UNSUPPORTED", "capability", "Artifact type 'unknown' is not recognized and has no adapter.")
    policy = json.loads(opts.get("policy", "{}"))
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    checks = [{"id": "markdown_validity", "name": "Markdown is readable", "status": "pass", "message": "", "evidence": {}}]
    if policy.get("forbid_placeholder_text"):
        found = [m for m in MARKERS if m in text]
        checks.append({"id": "leftover_placeholder_text", "name": "No leftover generation placeholder text", "status": "fail" if found else "pass",
                       "message": f"Found likely-unreviewed placeholder text: {found}." if found else "", "evidence": {"markers": found} if found else {}})
    if opts.get("policy-preset") == "print-a4":
        checks.append({"id": "page_size", "name": "Page size", "status": "unknown", "message": "not measurable without rendering", "evidence": {}})
    status = "fail" if any(c["status"] == "fail" for c in checks) else ("unknown" if any(c["status"] == "unknown" for c in checks) else "pass")
    doc = {"kind": "structural" if MODE != "bad_kind" else "visual", "status": status, "checks": checks, "evidence_files": [], "inspected_by": None}
    if MODE == "mutate":
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\nchanged")
    _emit(doc)
    code = {"pass": 0, "warn": 0, "fail": 1, "unknown": 3}[status]
    return (1 - code if code in (0, 1) else 0) if MODE == "wrong_exit" else code


def main(argv) -> int:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\n")
    if argv[:1] == ["contract"]:
        _emit(json.loads(PINNED.read_text(encoding="utf-8")))
        return 0
    if argv[:1] == ["doctor"]:
        return _doctor()
    if argv[:1] == ["verify"]:
        return _verify(argv[1:])
    return _error("ARTIFACT_INVALID_ARGS", "input", f"unsupported command {argv[:1]}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
