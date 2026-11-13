#!/usr/bin/env python3
"""Roadmap step 34 / ADR-116: the same fixtures normalised by both loudness paths, each output read by both meters.

    python3 tools/loudness_compare.py                         sibling checkouts, both targets, 1 run per case
    python3 tools/loudness_compare.py --repeat 3 --json out.json
    python3 tools/loudness_compare.py --ffmpeg-skill DIR --audio-production DIR --seconds 8 --targets -14/-1

Paths (each normalises one generated WAV to one target and writes a WAV):
  ffmpeg-skill       `python3 <ffmpeg-skill>/scripts/loudness.py IN -I <lufs> --tp <tp> -o OUT --json`: the argv the
                     agent's video path sends (`ffmpeg-skill/loudness`, compiler.py, LRA left at the Skill's default)
  audio-production   `audio-production run - --json`, one LOUDNESS_SET with one version (target_lufs, true_peak_db),
                     tolerance_lufs = TOLERANCE_LU; it runs ffmpeg-skill `loudness` and re-measures what it delivers

Meters (each reads every input and every output, independently of the path that wrote it):
  loudnorm           `-af loudnorm=print_format=json` pass 1: input_i / input_tp / input_lra (what both paths measure with)
  ebur128            `-af ebur128=peak=true`: the summary's I / LRA / true Peak (FFmpeg's BS.1770 / EBU R128 meter)

Fixtures are generated here from FFmpeg expressions (deterministic, no download, no licence): speech-like (voiced
harmonics plus noise, syllable envelope, pauses), music-like (chords, kick, hi-hat), quiet (the speech fixture 30 dB
down) and dynamic (quiet speech, then loud music: a loudness range above loudnorm's default 11 LU).

Run time is wall-clock seconds for the whole path call (process start to exit), the median of --repeat runs.
Nothing here decides routing: the numbers are for ADR-116. Standard library only; needs ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
TOLERANCE_LU = 1.0          # ffmpeg-skill loudness.py's LUFS_TOLERANCE: the same "on target" for both paths
TARGETS = ((-14.0, -1.0), (-23.0, -1.0))   # the agent's streaming default, and EBU R128
SAMPLE_RATE = 48000

# --- fixtures -----------------------------------------------------------------------------------------------------
# aevalsrc expressions of t; random(0) is FFmpeg's deterministic per-filter generator. Single quotes keep the commas
# inside one filter option.
_VOICED = "(sin(2*PI*140*t-40*cos(PI*t))+0.5*sin(2*(2*PI*140*t-40*cos(PI*t)))+0.3*sin(3*(2*PI*140*t-40*cos(PI*t)))+0.2*sin(4*(2*PI*140*t-40*cos(PI*t))))"
_SYLLABLES = "(0.15+0.85*abs(sin(PI*4.3*t)))*lt(mod(t,3.1),2.5)"
_PHRASES = "(0.6+0.4*sin(2*PI*t/7))"   # phrase-level rise and fall, about 8 dB peak to trough
_SPEECH = f"0.25*{_PHRASES}*{_SYLLABLES}*(0.7*{_VOICED}+0.6*(random(0)-0.5))"
_ROOT = "if(lt(mod(t,8),2),220,if(lt(mod(t,8),4),174.61,if(lt(mod(t,8),6),196,164.81)))"
_CHORD = f"(sin(2*PI*{_ROOT}*t)+0.8*sin(2*PI*{_ROOT}*1.26*t)+0.7*sin(2*PI*{_ROOT}*1.5*t)+0.3*sin(4*PI*{_ROOT}*t))"
_KICK = "exp(-30*mod(t,0.5))*sin(2*PI*(50+80*exp(-40*mod(t,0.5)))*mod(t,0.5))"
_HAT = "exp(-90*mod(t+0.25,0.5))*(random(0)-0.5)"
_MUSIC = f"0.12*{_CHORD}+0.5*{_KICK}+0.25*{_HAT}"

FIXTURES: Dict[str, Dict[str, str]] = {
    "speech": {"what": "speech-like: voiced harmonics around 140 Hz plus noise, 4.3 Hz syllables, a pause every 3.1 s, phrases rising and falling over 7 s",
               "expr": _SPEECH},
    "music": {"what": "music-like: four-chord loop, kick every 0.5 s, hi-hat off-beat; steady level",
              "expr": _MUSIC},
    "quiet": {"what": "quiet: the speech fixture 20 dB down (about +23 dB of gain to reach -14 LUFS)",
              "expr": f"0.1*{_SPEECH}"},
    "dynamic": {"what": "dynamic: the speech fixture 12 dB down and the music fixture in turn, 5 s each: a loudness range "
                        "above loudnorm's default 11 LU, with both parts inside the gate",
                "expr": f"if(lt(mod(t,10),5),0.25*{_SPEECH},{_MUSIC})"},
}


def make_fixture(name: str, path: Path, seconds: float) -> None:
    expr = FIXTURES[name]["expr"]
    # the right channel is the left one 3 ms later: stereo, not dual mono, the same for every fixture
    graph = (f"aevalsrc='{expr}':s={SAMPLE_RATE}:d={seconds:g},asplit[l][r];[r]adelay=3[rd];"
             f"[l][rd]join=inputs=2:channel_layout=stereo,alimiter=limit=0.99:level=false")
    _ffmpeg(["-f", "lavfi", "-i", graph, "-c:a", "pcm_s16le", "-y", str(path)])


# --- meters -------------------------------------------------------------------------------------------------------
def _ffmpeg(args: List[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg {' '.join(args)} failed:\n{proc.stderr[-1500:]}")
    return proc


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f or f in (float("inf"), float("-inf")) else f


def parse_loudnorm(stderr: str) -> Dict[str, Optional[float]]:
    """loudnorm's print_format=json block: integrated LUFS, true peak dBTP, LRA LU (None where it printed -inf)."""
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, re.S)
    if not m:
        raise ValueError("no loudnorm JSON in the output")
    d = json.loads(m.group(0))
    return {"i": _num(d.get("input_i")), "tp": _num(d.get("input_tp")), "lra": _num(d.get("input_lra"))}


def parse_ebur128(stderr: str) -> Dict[str, Optional[float]]:
    """ebur128's closing summary (the last one printed): I, LRA and the true Peak (peak=true), at its 0.1 resolution."""
    tail = stderr[stderr.rfind("Summary:"):] if "Summary:" in stderr else ""
    if not tail:
        raise ValueError("no ebur128 summary in the output")

    def field(label: str) -> Optional[float]:
        m = re.search(rf"^\s*{label}:\s*(-?inf|-?[0-9.]+)", tail, re.M)
        return _num(m.group(1)) if m else None
    return {"i": field("I"), "tp": field("Peak"), "lra": field("LRA")}


def meter_loudnorm(path: Path) -> Dict[str, Optional[float]]:
    return parse_loudnorm(_ffmpeg(["-i", str(path), "-map", "0:a:0", "-af", "loudnorm=print_format=json", "-f", "null", "-"]).stderr)


def meter_ebur128(path: Path) -> Dict[str, Optional[float]]:
    # framelog=verbose keeps the per-100 ms lines below the default log level: only the summary is printed
    return parse_ebur128(_ffmpeg(["-i", str(path), "-map", "0:a:0", "-af", "ebur128=peak=true:framelog=verbose", "-f", "null", "-"]).stderr)


def pcm_sha256(path: Path) -> str:
    """sha256 of the decoded samples (container headers left out): equal hashes mean the paths wrote the same sound."""
    out = _ffmpeg(["-i", str(path), "-map", "0:a:0", "-f", "hash", "-hash", "sha256", "-"]).stdout.strip()
    return out.split("=", 1)[-1]


# --- paths --------------------------------------------------------------------------------------------------------
def run_ffmpeg_skill(ffs: Path, src: Path, out: Path, lufs: float, tp: float) -> Dict[str, Any]:
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, str(ffs / "scripts" / "loudness.py"), str(src), "-I", f"{lufs:g}", "--tp", f"{tp:g}",
                           "-o", str(out), "--json"], capture_output=True, text=True)
    secs = time.perf_counter() - t0
    doc = _last_json(proc.stdout)
    ok = proc.returncode == 0 and out.is_file()
    return {"ok": ok, "seconds": secs, "error": None if ok else _error_text(doc, proc),
            "reported": {"loudnorm_mode": ((doc or {}).get("result") or {}).get("loudnorm_mode"),
                         "measured": next((v.get("measured") for v in (doc or {}).get("verification") or [] if v.get("step") == "loudness"), None)}}


def run_audio_production(aps: Path, ffs: Path, src: Path, work: Path, out_rel: str, lufs: float, tp: float) -> Dict[str, Any]:
    doc = {"schema": "audio-production/request@1", "project": {
        "project_id": "loudness-compare", "sources": [{"source_id": "a", "path": src.name}],
        "tracks": [{"track_id": "t1", "source_id": "a"}],
        "operations": [{"op_id": "deliver", "type": "LOUDNESS_SET", "inputs": ["track:t1"],
                        "parameters": {"tolerance_lufs": TOLERANCE_LU, "versions": [{"version_id": "v", "target_lufs": lufs, "true_peak_db": tp}]}}],
        "outputs": [{"output_id": "v", "operation": "op:deliver.v", "path": out_rel, "format": "wav"}]}}
    env = dict(os.environ, PYTHONPATH=str(aps / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "audio_production.cli", "run", "-", "--json", "--workspace", str(work),
                           "--ffmpeg-skill", str(ffs), "--no-reuse"],
                          input=json.dumps(doc), capture_output=True, text=True, cwd=str(work), env=env)
    secs = time.perf_counter() - t0
    resp = _last_json(proc.stdout) or {}
    ok = proc.returncode == 0 and bool(resp.get("ok")) and (work / out_rel).is_file()
    reported = None
    for s in resp.get("loudness_sets") or []:
        for v in s.get("versions") or []:
            reported = v.get("measured")
    if not ok:
        reported = ((resp.get("error") or {}).get("details") or reported)
    return {"ok": ok, "seconds": secs, "error": None if ok else _error_text(resp, proc), "reported": reported}


def _last_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else None
    except ValueError:
        for line in reversed(text.strip().splitlines()):
            try:
                d = json.loads(line)
                return d if isinstance(d, dict) else None
            except ValueError:
                continue
    return None


def _error_text(doc: Optional[Dict[str, Any]], proc: subprocess.CompletedProcess) -> str:
    err = (doc or {}).get("error")
    if isinstance(err, dict):
        reason = (err.get("details") or {}).get("reason")
        return f"{err.get('code')}{' ' + reason if reason else ''}: {err.get('message')}"
    if err:
        return str(err)
    return (proc.stderr or proc.stdout).strip()[-400:] or f"exit {proc.returncode}"


# --- the comparison -----------------------------------------------------------------------------------------------
def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else round(a - b, 2)


def measure(path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    return {"loudnorm": meter_loudnorm(path), "ebur128": meter_ebur128(path)}


def compare(ffs: Path, aps: Path, work: Path, seconds: float = 20.0, targets=TARGETS, fixtures=None, repeat: int = 1) -> Dict[str, Any]:
    work.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    inputs: Dict[str, Any] = {}
    for name in fixtures or list(FIXTURES):
        src = work / f"{name}.wav"
        make_fixture(name, src, seconds)
        inputs[name] = {"what": FIXTURES[name]["what"], "seconds": seconds, "measured": measure(src)}
        for lufs, tp in targets:
            tag = f"{name}_{-lufs:g}"
            runs: Dict[str, Any] = {}
            for path in ("ffmpeg-skill", "audio-production"):
                times, res = [], None
                for _ in range(max(1, repeat)):
                    if path == "ffmpeg-skill":
                        out = work / f"{tag}_ffs.wav"
                        out.unlink(missing_ok=True)
                        res = run_ffmpeg_skill(ffs, src, out, lufs, tp)
                    else:
                        out_rel = f"out/{tag}_aps.wav"
                        shutil.rmtree(work / "out", ignore_errors=True)
                        shutil.rmtree(work / ".audio-production", ignore_errors=True)
                        res = run_audio_production(aps, ffs, src, work, out_rel, lufs, tp)
                        out = work / out_rel
                        if out.is_file():
                            kept = work / f"{tag}_aps.wav"
                            shutil.move(str(out), kept)
                            out = kept
                    times.append(res["seconds"])
                entry: Dict[str, Any] = {"ok": res["ok"], "seconds": round(statistics.median(times), 2), "error": res["error"],
                                         "reported": res["reported"]}
                if res["ok"]:
                    m = measure(out)
                    entry["measured"] = m
                    entry["off_target_lu"] = {k: _delta(m[k]["i"], lufs) for k in m}
                    entry["tp_over_ceiling_db"] = {k: _delta(m[k]["tp"], tp) for k in m}
                    entry["meters_differ"] = {f: _delta(m["ebur128"][f], m["loudnorm"][f]) for f in ("i", "tp", "lra")}
                    entry["on_target"] = {k: (m[k]["i"] is not None and abs(m[k]["i"] - lufs) <= TOLERANCE_LU
                                              and m[k]["tp"] is not None and m[k]["tp"] <= tp) for k in m}
                    entry["pcm_sha256"] = pcm_sha256(out)
                runs[path] = entry
            a, b = runs["ffmpeg-skill"], runs["audio-production"]
            same = a.get("pcm_sha256") is not None and a.get("pcm_sha256") == b.get("pcm_sha256")
            between = None
            if a["ok"] and b["ok"]:
                between = {k: {f: _delta(b["measured"][k][f], a["measured"][k][f]) for f in ("i", "tp", "lra")} for k in ("loudnorm", "ebur128")}
            rows.append({"fixture": name, "target_lufs": lufs, "true_peak_db": tp, "paths": runs, "same_samples": same,
                         "audio_production_minus_ffmpeg_skill": between})
    return {"schema": "video-agent/loudness-compare@1", "tolerance_lu": TOLERANCE_LU, "ffmpeg": _ffmpeg_version(),
            "ffmpeg_skill": _version(ffs, "ffmpeg-skill"), "audio_production": _version(aps, "audio-production"),
            "inputs": inputs, "rows": rows}


def _ffmpeg_version() -> str:
    return subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]


def _version(root: Path, which: str) -> Optional[str]:
    try:
        if which == "ffmpeg-skill":
            return json.loads((root / "package.json").read_text())["version"] if (root / "package.json").is_file() else \
                json.loads(subprocess.run([sys.executable, str(root / "scripts" / "_contract.py"), "--json", "--static"],
                                          capture_output=True, text=True).stdout)["skill"]["version"]
        m = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.M)
        return m.group(1) if m else None
    except (OSError, ValueError, KeyError):
        return None


def _f(v: Optional[float], plus: bool = False) -> str:
    return "–" if v is None else (f"{v:+.1f}" if plus else f"{v:.1f}")


def table(result: Dict[str, Any]) -> str:
    """Markdown: one line per fixture x target x path, both meters side by side."""
    lines = [f"{result['ffmpeg']}; ffmpeg-skill {result['ffmpeg_skill']}; audio-production-skill {result['audio_production']}; "
             f"tolerance {result['tolerance_lu']:g} LU", "",
             "Inputs (loudnorm / ebur128): I LUFS, TP dBTP, LRA LU", "",
             "| fixture | I | TP | LRA |", "|---|---|---|---|"]
    for name, inp in result["inputs"].items():
        ln, eb = inp["measured"]["loudnorm"], inp["measured"]["ebur128"]
        lines.append(f"| {name} | {_f(ln['i'])} / {_f(eb['i'])} | {_f(ln['tp'])} / {_f(eb['tp'])} | {_f(ln['lra'])} / {_f(eb['lra'])} |")
    lines += ["", "Outputs (loudnorm / ebur128): I − target LU, TP − ceiling dB, LRA LU; seconds = the path's wall time", "",
              "| fixture | target | path | I − target | TP − ceiling | LRA | seconds | result |", "|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        for path, e in r["paths"].items():
            tgt = f"{r['target_lufs']:g} / {r['true_peak_db']:g}"
            if not e["ok"]:
                lines.append(f"| {r['fixture']} | {tgt} | {path} | – | – | – | {e['seconds']:.2f} | refused: {e['error'][:120]} |")
                continue
            o, c, m = e["off_target_lu"], e["tp_over_ceiling_db"], e["measured"]
            verdict = "on target (both meters)" if all(e["on_target"].values()) else \
                "off: " + ", ".join(k for k, v in e["on_target"].items() if not v)
            lines.append(f"| {r['fixture']} | {tgt} | {path} | {_f(o['loudnorm'], True)} / {_f(o['ebur128'], True)} | "
                         f"{_f(c['loudnorm'], True)} / {_f(c['ebur128'], True)} | {_f(m['loudnorm']['lra'])} / {_f(m['ebur128']['lra'])} | "
                         f"{e['seconds']:.2f} | {verdict} |")
        if r["paths"]["ffmpeg-skill"]["ok"] and r["paths"]["audio-production"]["ok"]:
            lines.append(f"| {r['fixture']} | | same samples | {'yes' if r['same_samples'] else 'no'} | | | | |")
    return "\n".join(lines)


def _find(explicit: Optional[str], env: str, sibling: str) -> Path:
    for cand in (explicit, os.environ.get(env), str(REPO.parent / sibling)):
        if cand and Path(cand).is_dir():
            return Path(cand).resolve()
    raise SystemExit(f"no {sibling} checkout: pass --{sibling.replace('-skill', '')} DIR or set {env}")


def _targets(text: str) -> Tuple[Tuple[float, float], ...]:
    out = []
    for item in text.split(","):
        lufs, tp = item.split("/")
        out.append((float(lufs), float(tp)))
    return tuple(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout (default: VIDEO_AGENT_FFMPEG_SKILL_DIR, then ../ffmpeg-skill)")
    ap.add_argument("--audio-production", help="audio-production-skill checkout (default: VIDEO_AGENT_AUDIO_PRODUCTION_DIR, then ../audio-production-skill)")
    ap.add_argument("--seconds", type=float, default=20.0, help="fixture length (default 20)")
    ap.add_argument("--targets", type=_targets, default=TARGETS, help="LUFS/TP list (default -14/-1,-23/-1)")
    ap.add_argument("--fixtures", help=f"comma list of {', '.join(FIXTURES)} (default: all)")
    ap.add_argument("--repeat", type=int, default=1, help="runs per path and case; the median time is reported (default 1)")
    ap.add_argument("--json", metavar="FILE", help="also write the full result document here")
    ap.add_argument("--workdir", help="keep fixtures and outputs here (default: a temporary directory, removed after)")
    a = ap.parse_args(argv)
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not on PATH")
    ffs = _find(a.ffmpeg_skill, "VIDEO_AGENT_FFMPEG_SKILL_DIR", "ffmpeg-skill")
    aps = _find(a.audio_production, "VIDEO_AGENT_AUDIO_PRODUCTION_DIR", "audio-production-skill")
    fixtures = a.fixtures.split(",") if a.fixtures else None
    tmp = None if a.workdir else tempfile.mkdtemp(prefix="loudness-compare-")
    try:
        result = compare(ffs, aps, Path(a.workdir or tmp), a.seconds, a.targets, fixtures, a.repeat)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=2) + "\n")
    print(table(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
