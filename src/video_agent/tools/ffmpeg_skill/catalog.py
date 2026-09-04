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
    "multicam": {"positional": ["inputs"], "flags": {"switch": "str", "auto": "float", "audio": "int", "offsets_only": "bool", "fix_drift": "bool", "output": "path_out", "preset": "str"},
                 "produces_output": True, "result_keys": ["offsets_seconds", "confidence", "cuts"]},
    "report": {"positional": [], "flags": {"after": "path_in", "before": "path_in", "platform": "str", "commands": "path_in", "notes": "path_in", "title": "str", "output": "path_out", "no_sheets": "bool"},
               "produces_output": True, "result_keys": ["report", "check"]},
}

# flags whose CLI spelling differs from the underscore->dash rule
FLAG_ALIASES = {("loudness", "lufs"): "-I", ("*", "output"): "-o"}
