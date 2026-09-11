"""Machine-readable contract of the ffmpeg-skill scripts the agent uses (verified against v0.8.4 --help).
Types: str | float | int | bool | list[str] | path_in | path_out. Positional args are listed in order.
Only the flags the agent needs are declared; anything else is rejected by the adapter, so an unknown
key can never leak through as a raw command-line fragment."""
from __future__ import annotations

from typing import Any, Dict

CATALOG: Dict[str, Dict[str, Any]] = {
    "probe": {"positional": ["inputs"], "flags": {"analyze": "bool"}, "produces_output": False, "json_by_default": True,
              "result_keys": ["duration", "video", "audio", "format", "size_bytes"]},
    "silence": {"positional": ["input"], "flags": {"threshold": "float", "min_silence": "float", "margin": "float", "min_keep": "float", "list": "bool", "edl": "path_out", "output": "path_out", "preset": "str"},
                "produces_output": True, "result_keys": ["silences", "keep", "input_duration", "kept_duration", "removed_seconds"]},
    "loudness": {"positional": ["input"], "flags": {"lufs": "float", "tp": "float", "lra": "float", "measure_only": "bool", "output": "path_out", "sample_rate": "int"},
                 "produces_output": True, "result_keys": ["input_i", "input_tp", "input_lra", "silent"]},
    "cut": {"positional": ["input"], "flags": {"start": "str", "end": "str", "duration": "str", "segments": "str", "accurate": "bool", "tolerance": "float", "output": "path_out", "preset": "str", "crf": "int"},
            "produces_output": True, "result_keys": ["output", "probe"]},
    "fit": {"positional": ["input"], "flags": {"duration": "str", "method": "str", "aspect": "str", "fit": "str", "width": "int", "fps": "float", "output": "path_out", "preset": "str"},
            "produces_output": True, "result_keys": ["output", "probe"]},
    "export": {"positional": ["input"], "flags": {"preset": "str", "fit": "str", "crf": "int", "allow_long": "bool", "no_scale": "bool", "output": "path_out"},
               "produces_output": True, "result_keys": ["output", "probe"]},
    "check": {"positional": ["input"], "flags": {"platform": "str", "max_duration": "float", "aspect": "str", "lufs": "float", "tp": "float", "max_mb": "float", "no_loudness": "bool"},
              "produces_output": False, "result_keys": ["platform", "checks", "failed", "warnings", "ok"], "exit_code_means_fail": True},
    "look": {"positional": ["input"], "flags": {"at": "list[str]", "tiles": "str", "width": "int", "output": "path_out"},
             "produces_output": True, "result_keys": ["outputs"]},
    "scenes": {"positional": ["input"], "flags": {"threshold": "float", "min_scene": "float", "highlights": "int", "target": "float", "edl": "path_out"},
               "produces_output": False, "result_keys": ["scenes", "audio_peaks", "highlights"]},
    "sync": {"positional": ["reference", "second"], "flags": {"max_offset": "float", "analyze_seconds": "float", "fix_drift": "bool", "replace_audio": "bool", "trim_second": "bool", "output": "path_out"},
             "produces_output": True, "result_keys": ["offset_seconds", "confidence", "drift"]},
    "multicam": {"positional": ["inputs"], "flags": {"switch": "str", "auto": "float", "audio": "int", "offsets_only": "bool", "max_offset": "float", "analyze_seconds": "float", "fix_drift": "bool",
                                                      "width": "int", "height": "int", "fps": "float", "crf": "int", "output": "path_out", "preset": "str"},
                 "produces_output": True, "result_keys": ["offsets_seconds", "confidence", "cuts"]},
    "report": {"positional": [], "flags": {"after": "path_in", "before": "path_in", "platform": "str", "commands": "path_in", "notes": "path_in", "title": "str", "output": "path_out", "no_sheets": "bool"},
               "produces_output": True, "result_keys": ["report", "check"]},
    # ---- ADR-046 Priority A batch (verified against ffmpeg-skill v0.16.2 --help)
    "redact": {"positional": ["input"], "flags": {"x": "float", "y": "float", "width": "int", "height": "int", "mode": "str", "blur_strength": "int", "block_size": "int",
                                                   "audio_stream": "int", "crf": "int", "preset": "str", "fps": "float", "output": "path_out"},
               "produces_output": True, "result_keys": ["output", "probe"]},
    "deinterlace": {"positional": ["input"], "flags": {"mode": "str", "parity": "str", "only_interlaced": "bool", "audio_stream": "int", "crf": "int", "preset": "str", "output": "path_out"},
                    "produces_output": True, "result_keys": ["output", "probe"]},
    # measurement only (no output file): reports the crop rectangle cropdetect found. result_keys verified against a
    # real --json capture (synthetic letterboxed clip, ffmpeg-skill 0.16.2) -- see ADR-046 correction: the crop
    # rectangle is a nested "crop" object ({width, height, x, y}), not flat top-level x/y/width/height, and there is
    # no "crop_filter" key (the original catalog entry's shape was an unverified guess and wrong on both counts).
    "cropdetect": {"positional": ["input"], "flags": {"seconds": "float", "samples": "int", "limit": "float", "round_to": "int"},
                   "produces_output": False, "result_keys": ["file", "source_width", "source_height", "crop", "confidence"]},
    "crop": {"positional": ["input"], "flags": {"x": "float", "y": "float", "width": "int", "height": "int", "crf": "int", "preset": "str", "fps": "float", "output": "path_out"},
             "produces_output": True, "result_keys": ["output", "probe"]},
    "stabilize": {"positional": ["input"], "flags": {"shakiness": "int", "smoothing": "int", "zoom": "float", "crop_mode": "str", "tripod": "bool", "crf": "int", "preset": "str", "output": "path_out"},
                  "produces_output": True, "result_keys": ["output", "probe"]},
    "grid": {"positional": ["inputs"], "flags": {"cols": "int", "rows": "int", "cell_width": "int", "cell_height": "int", "fps": "float", "label": "str", "font": "str", "font_size": "int",
                                                  "font_color": "str", "pad": "bool", "audio_from": "int", "gap": "int", "background": "str", "crf": "int", "preset": "str", "output": "path_out"},
             "produces_output": True, "result_keys": ["output", "probe"]},
}

# flags whose CLI spelling differs from the underscore->dash rule
FLAG_ALIASES = {("loudness", "lufs"): "-I", ("*", "output"): "-o",
                 ("cropdetect", "round_to"): "--round",   # --round collides with the python builtin `round`
                 ("stabilize", "crop_mode"): "--crop"}    # "crop" collides with this vocabulary's own video.crop operation name
