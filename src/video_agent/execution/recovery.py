"""Error classification and finite recovery strategies (MASTER_SPEC §32)."""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ..models import ToolResult

RECOVERY_TABLE: Dict[str, Dict[str, Any]] = {
    "TOOL_MISSING":   {"action": "BLOCK", "hint": "run `video-agent doctor`; ffmpeg or ffmpeg-skill is not available"},
    "INPUT_MISSING":  {"action": "BLOCK", "hint": "input file disappeared or is outside the allowed roots"},
    "INVALID_ARGS":   {"action": "BLOCK", "hint": "compiler produced arguments the tool rejects (agent bug, not retried)"},
    "ENCODER_FAILED": {"action": "RETRY_ALT", "hint": "retry once with frame-accurate mode", "alt_args": {"ffmpeg-skill/cut": {"accurate": True}}},
    "TIMEOUT":        {"action": "RETRY_LONGER", "hint": "retry once with a doubled timeout"},
    "DISK_FULL":      {"action": "BLOCK", "hint": "free disk space in the workspace"},
    "OUTPUT_INVALID": {"action": "BLOCK", "hint": "the Skill validated its own output and rejected it, or wrote none (deterministic; not retried)"},
    "INTERRUPTED":    {"action": "BLOCK", "hint": "the Skill was cancelled (interrupt); re-run the job"},
    "UNKNOWN":        {"action": "RETRY_SAME", "hint": "retry once with identical arguments"},
}

# Structured error codes of external Skills (their contracts' error tables) → recovery class. A Skill's `retryable` flag is
# honoured: a non-retryable TOOL_ERROR (engine missing) blocks; a retryable one gets the finite retry. Unlisted codes fall back
# to the text classification below (media-analysis / transcription codes are handled by the analysis failure domain instead).
STRUCTURED_CLASSES: Dict[str, str] = {
    "INVALID_REQUEST": "INVALID_ARGS", "INVALID_INPUT": "INVALID_ARGS", "PATH_NOT_ALLOWED": "INPUT_MISSING", "UNSUPPORTED_OPERATION": "INVALID_ARGS",
    "UNSUPPORTED_FORMAT": "INVALID_ARGS", "MISSING_INPUT": "INPUT_MISSING", "INVALID_TIME_RANGE": "INVALID_ARGS", "DEPENDENCY_ERROR": "INVALID_ARGS",
    "OUTPUT_ERROR": "OUTPUT_INVALID", "VALIDATION_ERROR": "OUTPUT_INVALID", "INVALID_RESULT": "OUTPUT_INVALID", "INTERNAL_ERROR": "UNKNOWN",
}


def structured_class(r: ToolResult) -> Optional[str]:
    err = (r.data or {}).get("error") if isinstance(r.data, dict) else None
    if not isinstance(err, dict) or not isinstance(err.get("code"), str):
        return None
    code = err["code"]
    if code == "TOOL_ERROR":
        return "ENCODER_FAILED" if err.get("retryable", True) else "TOOL_MISSING"
    if code == "CANCELLED":
        return "TIMEOUT" if (err.get("details") or {}).get("reason") == "timeout" else "INTERRUPTED"
    return STRUCTURED_CLASSES.get(code)


def classify_error(r: ToolResult) -> str:
    cls = structured_class(r)
    if cls is not None:
        return cls
    txt = (r.stderr_tail or "").lower()
    if r.exit_code == 127 or "was not found on path" in txt:
        return "TOOL_MISSING"
    if r.exit_code == 124 or "timeout" in txt:
        return "TIMEOUT"
    if "input not found" in txt or "no such file" in txt or "outside allowed" in txt or "outside workspace" in txt:
        return "INPUT_MISSING"
    if "no space left" in txt:
        return "DISK_FULL"
    if re.search(r"unrecognized arguments|invalid choice|argument .*: expected|error: (give|use|end must|segment|bad )", txt):
        return "INVALID_ARGS"
    if "command failed" in txt or "ffmpeg failed" in txt or "encoder" in txt or "conversion failed" in txt:
        return "ENCODER_FAILED"
    return "UNKNOWN"


def next_attempt(r: ToolResult, attempt: int, max_attempts: int, timeout: Optional[float]) -> Dict[str, Any]:
    """Decide what the executor does after a failure. Returns {action, reason, args_patch, timeout}."""
    cls = classify_error(r)
    strat = RECOVERY_TABLE[cls]
    if strat["action"] == "BLOCK" or attempt >= max_attempts:
        return {"class": cls, "action": "BLOCK", "reason": strat["hint"] if strat["action"] == "BLOCK" else f"max attempts ({max_attempts}) reached after {cls}", "args_patch": {}, "timeout": timeout}
    if strat["action"] == "RETRY_ALT":
        return {"class": cls, "action": "RETRY", "reason": strat["hint"], "args_patch": strat["alt_args"].get(r.tool, {}), "timeout": timeout}
    if strat["action"] == "RETRY_LONGER":
        return {"class": cls, "action": "RETRY", "reason": strat["hint"], "args_patch": {}, "timeout": (timeout * 2) if timeout else None}
    return {"class": cls, "action": "RETRY", "reason": strat["hint"], "args_patch": {}, "timeout": timeout}
