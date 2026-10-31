"""Fake slide-sync-skill process for the ADR-091 tests: speaks the slice of the real `slide-sync` CLI the agent uses -- `contract --json` (the pinned
src/video_agent/tools/slide_sync/contract_0.1.0.json), `doctor --json` (engine / import readiness, exit 3 while the engine is not ready),
`detect --json --overwrite --output=F [--threshold=X] [--min-dwell=S] [--ffmpeg-skill=D] --timeout=T -- <video>` (slide-sync/timeline@1),
`import --json --overwrite --output=DIR -- <deck>` (slide-sync/deck@1 with the deck's sha256, deck.json written into DIR) and
`match --json --overwrite --output=F --deck=DIR --timeline=T ... -- <video>` (slide-sync/match@1), each printed with ok and written where asked.
The media are the self-describing fake files of tests/pipeline_harness.py: their `slides` fact [[start, end], ...] (default: three slides
splitting the file) and `pages` fact (the page on screen in each slide, null = not matched). A deck file is any file; its `pages` count is
read from its JSON when it has one (default 4). Nothing runs an engine.
Controlled via:

  FAKE_SS_READY: all (default) | nopdf (pypdfium2 missing) | noengine (ffmpeg-skill not found)
  FAKE_SS_MODE: ok | fail:<CODE> | wrong_threshold | other_video | overlap | bad_page | deck_sha | not_written
  FAKE_SS_CALLS: a file every invocation's argv is appended to (one JSON list per line). Test double only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

MODE = os.environ.get("FAKE_SS_MODE", "ok")
READY = os.environ.get("FAKE_SS_READY", "all")
CALLS_LOG = os.environ.get("FAKE_SS_CALLS")
PINNED = Path(__file__).resolve().parents[1] / "src" / "video_agent" / "tools" / "slide_sync" / "contract_0.1.0.json"


def _emit(doc) -> None:
    sys.stdout.write(json.dumps(doc) + "\n")


def _contract() -> dict:
    return json.loads(PINNED.read_text(encoding="utf-8"))


def _fail(code: str, message: str) -> int:
    _emit({"ok": False, "error": {"code": code, "message": message, "detail": None}})
    return int(_contract()["errors"].get(code, {}).get("exit_code", 1))


def _opts(argv):
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
        else:
            pos.append(a)
    return opts, pos


def _facts(path: str) -> dict:
    try:
        return json.loads(Path(path).read_bytes().decode())
    except (OSError, ValueError):
        return {}


def _doctor() -> int:
    engine_ok, pdf_ok = READY != "noengine", READY != "nopdf"
    doc = {"ok": engine_ok, "skill": "slide-sync-skill", "version": _contract()["version"],
           "engine": {"found": engine_ok, "ok": engine_ok, "version": "3.1.1", "missing_tools": [] if engine_ok else ["scenes", "metadata", "look"]},
           "import": {"pdf": {"found": pdf_ok, "install": None if pdf_ok else "pip install 'slide-sync-skill[pdf]'", "ready": pdf_ok},
                      "libreoffice": {"found": False, "install": "install LibreOffice (https://www.libreoffice.org)"},
                      "ready": {"pdf": pdf_ok, "pptx_key_odp": False}},
           "problems": [] if engine_ok else ["ffmpeg-skill not found"], "notes": [] if pdf_ok else ["slide import: pypdfium2 not found; pip install 'slide-sync-skill[pdf]'"]}
    _emit(doc)
    return 0 if engine_ok else 3


def _timeline(video: str, opts: dict) -> dict:
    f = _facts(video)
    dur = float(f.get("duration") or 12.0)
    spans = f.get("slides") or [[0.0, dur / 3], [dur / 3, 2 * dur / 3], [2 * dur / 3, dur]]
    slides = [{"index": i + 1, "start": round(float(s), 3), "end": round(float(e), 3), "duration": round(float(e) - float(s), 3)} for i, (s, e) in enumerate(spans)]
    if MODE == "overlap" and len(slides) > 1:
        slides[1]["start"] = slides[0]["start"]
    thr = float(opts.get("threshold", 8.0)) + (1.0 if MODE == "wrong_threshold" else 0.0)
    return {"schema": "slide-sync/timeline@1", "source": {"path": os.path.abspath(video if MODE != "other_video" else video + ".other"), "duration": dur},
            "detector": {"skill": "slide-sync", "skill_version": _contract()["version"], "engine": "ffmpeg-skill/scenes", "engine_version": "3.1.1",
                         "method": "scdet cuts held at least min_dwell seconds", "threshold": thr, "min_dwell": float(opts.get("min-dwell", 2.0))},
            "slide_count": len(slides), "changes": [s["start"] for s in slides[1:]], "slides": slides, "sheet": None, "provenance": "OBSERVED"}


def _write(path, doc):
    if path and MODE != "not_written":
        Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _detect(argv) -> int:
    opts, pos = _opts(argv)
    if READY == "noengine":
        return _fail("ENGINE_NOT_FOUND", "ffmpeg-skill not found (fake)")
    if MODE.startswith("fail:"):
        return _fail(MODE.split(":", 1)[1], "refused (fake)")
    tl = _timeline(pos[0], opts)
    _write(opts.get("output"), tl)
    _emit(dict(tl, ok=True, timeline_file=opts.get("output")))
    return 0


def _import(argv) -> int:
    opts, pos = _opts(argv)
    if READY == "nopdf":
        return _fail("DEPENDENCY_MISSING", "pypdfium2 is not installed: pip install 'slide-sync-skill[pdf]'")
    deck = pos[0]
    raw = Path(deck).read_bytes()
    n = int(_facts(deck).get("pages") or 4)
    out = Path(opts["output"])
    out.mkdir(parents=True, exist_ok=True)
    pages = [{"index": i + 1, "image": f"page-{i + 1:03d}.png", "text_file": f"page-{i + 1:03d}.txt", "width_px": 1280, "height_px": 720, "width_pt": 960, "height_pt": 540,
              "image_sha256": "0" * 64, "chars": 10 * (i + 1), "text": f"Page {i + 1}"} for i in range(n)]
    doc = {"schema": "slide-sync/deck@1", "source": {"path": os.path.abspath(deck), "format": Path(deck).suffix[1:], "sha256": ("f" * 64) if MODE == "deck_sha" else hashlib.sha256(raw).hexdigest(),
                                                     "converted_by": None},
           "renderer": {"skill": "slide-sync", "skill_version": _contract()["version"], "engine": "pdfium", "dpi": 144.0}, "page_count": n, "pages": pages, "provenance": "OBSERVED"}
    (out / "deck.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _emit(dict(doc, ok=True, output_dir=str(out)))
    return 0


def _match(argv) -> int:
    opts, pos = _opts(argv)
    tl = json.loads(Path(opts["timeline"]).read_text(encoding="utf-8"))
    deck = json.loads((Path(opts["deck"]) / "deck.json").read_text(encoding="utf-8"))
    pages = _facts(pos[0]).get("pages") or [i + 1 for i in range(len(tl["slides"]))]
    segs = []
    for s, p in zip(tl["slides"], list(pages) + [None] * len(tl["slides"])):
        if MODE == "bad_page" and p is not None:
            p = deck["page_count"] + 5
        segs.append({"index": s["index"], "start": s["start"], "end": s["end"], "page": p, "confidence": 0.93 if p is not None else 0.41, "margin": 0.4 if p is not None else 0.02,
                     "method": "luma-grid-zncc", "view": "full", "reason": None if p is not None else "below_min_score", "samples": [], "mixed": False, "candidates": []})
    shown = sorted({s["page"] for s in segs if s["page"] is not None})
    doc = {"schema": "slide-sync/match@1", "source": tl["source"], "deck": {"path": opts["deck"], "page_count": deck["page_count"], "source": deck["source"]["path"]},
           "matcher": {"skill": "slide-sync", "skill_version": _contract()["version"], "engine": "ffmpeg-skill/look", "engine_version": "3.1.1", "method": "luma-grid-zncc",
                       "min_score": 0.7, "min_margin": 0.1, "samples": 3},
           "segment_count": len(segs), "matched_count": sum(1 for s in segs if s["page"] is not None), "segments": segs, "pages_shown": shown,
           "pages_not_shown": [i for i in range(1, deck["page_count"] + 1) if i not in shown],
           "warnings": [f"{sum(1 for s in segs if s['page'] is None)} segment(s) not matched to a page"] if any(s["page"] is None for s in segs) else [], "provenance": "OBSERVED"}
    _write(opts.get("output"), doc)
    _emit(dict(doc, ok=True, match_file=opts.get("output")))
    return 0


def main(argv) -> int:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\n")
    cmd = argv[:1]
    if cmd == ["contract"]:
        _emit(_contract())
        return 0
    if cmd == ["doctor"]:
        return _doctor()
    if cmd == ["detect"]:
        return _detect(argv[1:])
    if cmd == ["import"]:
        return _import(argv[1:])
    if cmd == ["match"]:
        return _match(argv[1:])
    return _fail("INVALID_ARGUMENT", f"unsupported command {cmd}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
