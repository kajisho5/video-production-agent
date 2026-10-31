"""Fake imagemagick-skill for the ADR-077 tests: `make_checkout(dir)` writes a checkout whose `scripts/_contract.py`, `resize.py`, `pad.py`
and `strip.py` hand over to this file, so the agent launches it exactly as the real one (`python <dir>/scripts/<tool>.py ...`). It speaks
the slice of imagemagick-skill 0.8.1 the agent uses: `_contract.py contract --json` (the pinned src/video_agent/tools/imagemagick/contract_0.8.1.json),
`_contract.py doctor --json` (`tools.<name>.usable`), and resize / pad / strip with `--flag=value` options and the input after `--`, in the
result shapes of its contract (`{"ok": true, ...}`, `{"ok": false, "reason"}` exit 1, the timeout shape exit 124). A fake image is the PNG
signature followed by a JSON `{"w", "h"}` (a file without it counts as 400x300). Nothing runs ImageMagick.
Controlled via:

  FAKE_IM_MODE: ok | fail | timeout | wrong_size | wrong_output | bad_json | touch_input | gps | exit_nonzero
  FAKE_IM_BACKEND: magick (every tool usable, default) | sips (resize / pad only) | none
  FAKE_IM_CONTRACT: pinned | drift_version (0.8.9: non-fatal) | drift_description (a description: non-fatal) | drift_flag (resize --width
    renamed: fatal) | drift_enum (resize mode loses exact: fatal) | drift_required (pad requires --aspect: fatal) | incompatible (0.9.0)
  FAKE_IM_CALLS: a file every invocation's argv is appended to (one JSON list per line, the script name first). Test double only."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PINNED = Path(__file__).resolve().parents[1] / "src" / "video_agent" / "tools" / "imagemagick" / "contract_0.8.1.json"
SIG = b"\x89PNG\r\n\x1a\n"
STUB = "import runpy, sys\nsys.argv.insert(1, {name!r})\nrunpy.run_path({fake!r}, run_name='__main__')\n"


def make_checkout(root: str) -> str:
    """A fake imagemagick-skill checkout at `root` (scripts/_contract.py + the three tools)."""
    scripts = Path(root) / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for name in ("_contract", "resize", "pad", "strip"):
        (scripts / f"{name}.py").write_text(STUB.format(name=name, fake=str(Path(__file__).resolve())), encoding="utf-8")
    return str(Path(root).resolve())


def image(path: str, w: int, h: int, extra: bytes = b"") -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(SIG + json.dumps({"w": w, "h": h}).encode() + extra)
    return path


def size_of(path: str):
    b = Path(path).read_bytes()
    if b.startswith(SIG):
        try:
            d, _ = json.JSONDecoder().raw_decode(b[len(SIG):].decode("utf-8", "ignore"))
            return int(d["w"]), int(d["h"])
        except (ValueError, KeyError, TypeError):
            pass
    return 400, 300


def _contract() -> dict:
    c = json.loads(PINNED.read_text(encoding="utf-8"))
    mode = os.environ.get("FAKE_IM_CONTRACT", "pinned")
    tools = {t["name"]: t for t in c["tools"]}
    if mode == "drift_version":
        c["version"] = c["skill"]["version"] = "0.8.9"
    elif mode == "drift_description":
        tools["resize"]["description"] = "Resize an image (reworded)."
    elif mode == "drift_flag":
        tools["resize"]["input_schema"]["properties"]["width"]["cli"] = ["--w"]
    elif mode == "drift_enum":
        tools["resize"]["input_schema"]["properties"]["mode"]["enum"] = ["fit", "fill"]
    elif mode == "drift_required":
        tools["pad"]["input_schema"]["required"] = list(tools["pad"]["input_schema"]["required"]) + ["aspect"]
    elif mode == "incompatible":
        c["version"] = c["skill"]["version"] = "0.9.0"
    return c


def _log(argv) -> None:
    if os.environ.get("FAKE_IM_CALLS"):
        with open(os.environ["FAKE_IM_CALLS"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\n")


def _emit(doc) -> None:
    sys.stdout.write(json.dumps(doc) + "\n")


def _parse(args):
    opts, pos = {}, []
    if "--" in args:
        i = args.index("--")
        args, pos = args[:i], args[i + 1:]
    for a in args:
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            opts[k] = v
        elif a.startswith("--"):
            opts[a[2:]] = True
        else:
            pos.append(a)
    return opts, pos


def main() -> int:
    script, args = sys.argv[1], sys.argv[2:]
    _log([script] + args)
    if script == "_contract":
        if args[:1] == ["contract"]:
            _emit(_contract())
            return 0
        backend = os.environ.get("FAKE_IM_BACKEND", "magick")
        usable = {"magick": ("resize", "pad", "strip"), "sips": ("resize", "pad"), "none": ()}[backend]
        _emit({"ok": backend != "none", "platform": "linux", "backends": {"magick": {"found": backend == "magick", "kind": "imagemagick6" if backend == "magick" else None},
                                                                          "sips": {"found": backend == "sips"}},
               "tools": {n: ({"usable": True, "backend": "sips" if backend == "sips" else "magick"} if n in usable else {"usable": False, "reason": "needs ImageMagick (magick)"})
                         for n in ("probe", "resize", "pad", "strip")}})
        return 0
    mode = os.environ.get("FAKE_IM_MODE", "ok")
    opts, pos = _parse(args)
    if not opts.get("json") or len(pos) != 1 or not opts.get("output"):
        _emit({"ok": False, "reason": "usage: input positional, --output and --json required"})
        return 1
    src, out = os.path.abspath(pos[0]), os.path.abspath(str(opts["output"]))
    if mode == "fail":
        _emit({"ok": False, "reason": f"magick failed on {os.path.basename(src)}"})
        return 1
    if mode == "timeout":
        _emit({"ok": False, "reason": "timed out", "code": "timeout", "timeout": float(opts.get("timeout", 300))})
        return 124
    if mode == "bad_json":
        sys.stdout.write("magick: done\n")
        return 0
    if os.path.exists(out) and not opts.get("overwrite"):
        _emit({"ok": False, "reason": f"output exists: {out}"})
        return 1
    iw, ih = size_of(src)
    doc = {"ok": True, "input": src, "output": out if mode != "wrong_output" else out + ".other.png", "backend": "magick"}
    if script == "resize":
        w, h, m = int(opts["width"]), int(opts["height"]), str(opts.get("mode", "fit"))
        if m == "fit":
            s = min(w / iw, h / ih)
            aw, ah = max(1, round(iw * s)), max(1, round(ih * s))
        else:
            aw, ah = w, h
        if mode == "wrong_size":
            aw, ah = aw + 7, ah + 7
        doc.update(mode=m, requested={"width": w, "height": h}, actual={"width": aw, "height": ah})
    elif script == "pad":
        w, h = int(opts["width"]), int(opts["height"])
        aw, ah = (w, h) if mode != "wrong_size" else (w - 2, h)
        doc.update(mode="canvas", color=opts.get("color"), original={"width": iw, "height": ih}, offset={"x": (w - iw) // 2, "y": (h - ih) // 2}, actual={"width": aw, "height": ah})
    else:
        aw, ah = iw, ih
        doc.update(has_gps=mode == "gps")
    image(out, aw, ah, b" " + script.encode())
    if mode == "touch_input":
        with open(src, "ab") as fh:
            fh.write(b"!")
    _emit(doc)
    return 3 if mode == "exit_nonzero" else 0


if __name__ == "__main__":
    sys.exit(main())
