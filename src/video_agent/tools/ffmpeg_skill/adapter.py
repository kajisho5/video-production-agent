"""FfmpegSkillAdapter: the only place that invokes ffmpeg-skill scripts.

Contract facts it relies on (ARCHITECTURE_REVIEW §1.1): one process per script, `--json` prints one JSON
document on stdout, non-zero exit + `error:` on stderr on failure, exit 127 when ffmpeg is missing,
`check.py` exits 1 when a check FAILs (that is a result, not an error)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...models import Operation, ToolResult
from ..base import ToolAdapter, ToolError
from .catalog import CATALOG, FLAG_ALIASES
from .package import package
from ...skills.contract import SkillPackage
from .locate import FfmpegSkill, locate_ffmpeg_skill

PREFIX = "ffmpeg-skill/"


class PathPolicy:
    """Workspace boundary: inputs must come from allowed roots, outputs must land inside the workspace,
    and no output may ever equal an input (ffmpeg-skill itself has no such guard)."""

    def __init__(self, allowed_inputs: List[str], workspace: str):
        self.allowed_inputs = [str(Path(p).resolve()) for p in allowed_inputs]
        self.workspace = str(Path(workspace).resolve())

    @staticmethod
    def _norm(p: str) -> str:
        # normcase: Windows paths compare case-insensitively (C:\\ vs c:\\), POSIX unchanged
        return os.path.normcase(str(Path(p).resolve()))

    def check_input(self, p: str) -> None:
        rp = self._norm(p)
        roots = [os.path.normcase(a) for a in self.allowed_inputs] + [os.path.normcase(self.workspace)]
        if not any(rp == a or rp.startswith(a + os.sep) for a in roots):
            raise ToolError(f"input outside allowed roots: {p}")

    def check_output(self, p: str, inputs: List[str]) -> None:
        rp = self._norm(p)
        if not rp.startswith(os.path.normcase(self.workspace) + os.sep):
            raise ToolError(f"output outside workspace: {p}")
        for i in inputs:
            if self._norm(i) == rp:
                raise ToolError(f"output would overwrite its input: {p}")


class FfmpegSkillAdapter(ToolAdapter):
    name = "ffmpeg-skill"

    def __init__(self, skill: Optional[FfmpegSkill] = None, path_policy: Optional[PathPolicy] = None, python: Optional[str] = None):
        self.skill = skill or locate_ffmpeg_skill()
        if not self.skill:
            raise ToolError("ffmpeg-skill not found (set VIDEO_AGENT_FFMPEG_SKILL_DIR)")
        self.version = self.skill.version
        self.policy = path_policy
        self.python = python or sys.executable

    # ---- description
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "root": str(self.skill.root), "tools": {PREFIX + k: v for k, v in CATALOG.items()}}

    def supports(self, tool: str) -> bool:
        return tool.startswith(PREFIX) and tool[len(PREFIX):] in CATALOG and tool[len(PREFIX):] in self.skill.scripts

    def package(self) -> SkillPackage:
        return package(self.version)

    # ---- argv construction (typed args -> argv; never free-form)
    def build_argv(self, tool: str, args: Dict[str, Any], paths: Dict[str, str]) -> List[str]:
        script = tool[len(PREFIX):]
        spec = CATALOG.get(script)
        if spec is None:
            raise ToolError(f"unknown tool {tool}")
        args = dict(args)
        argv: List[str] = []
        inputs: List[str] = []
        for key in spec["positional"]:
            val = args.pop(key, None)
            if val is None:
                raise ToolError(f"{tool}: missing positional '{key}'")
            vals = val if isinstance(val, list) else [val]
            for v in vals:
                p = paths.get(v, v)
                if self.policy:
                    self.policy.check_input(p)
                inputs.append(p)
                argv.append(p)
        for key, val in args.items():
            typ = spec["flags"].get(key)
            if typ is None:
                raise ToolError(f"{tool}: flag '{key}' is not in the catalog")
            if val is None or val is False:
                continue
            flag = FLAG_ALIASES.get((script, key)) or FLAG_ALIASES.get(("*", key)) or "--" + key.replace("_", "-")
            if typ == "bool":
                if val is not True:
                    raise ToolError(f"{tool}: '{key}' must be bool")
                argv.append(flag)
            elif typ == "list[str]":
                for v in val:
                    argv += [flag, str(v)]
            elif typ in ("path_in", "path_out"):
                p = paths.get(val, val)
                if self.policy:
                    if typ == "path_in":
                        self.policy.check_input(p)
                    else:
                        self.policy.check_output(p, inputs)
                argv += [flag, p]
            elif typ == "int":
                argv += [flag, str(int(val))]
            elif typ == "float":
                argv += [flag, f"{float(val):g}"]
            else:
                argv += [flag, str(val)]
        return argv

    def command(self, tool: str, args: Dict[str, Any], paths: Dict[str, str], dry_run: bool = False) -> List[str]:
        script = tool[len(PREFIX):]
        cmd = [self.python, str(self.skill.script(script))] + self.build_argv(tool, args, paths) + ["--json"]
        if dry_run:
            cmd.append("--dry-run")
        return cmd

    def preview(self, op: Operation, paths: Dict[str, str]) -> List[str]:
        return [" ".join(self.command(op.tool, op.args, paths))]

    # ---- execution
    def run(self, op: Operation, paths: Dict[str, str], timeout: Optional[float] = None, dry_run: bool = False, attempt: int = 1) -> ToolResult:
        cmd = self.command(op.tool, op.args, paths, dry_run=dry_run)
        script = op.tool[len(PREFIX):]
        t0 = time.time()
        out_path = paths.get(op.args["output"], op.args["output"]) if isinstance(op.args.get("output"), str) else None
        code, out, err = run_process_group(cmd, timeout)
        if code != 0 and out_path and not dry_run and code != 124 and os.path.exists(out_path) and _is_fresh(out_path, t0):
            _remove_partial(out_path)
        if code == 124 and out_path and os.path.exists(out_path) and _is_fresh(out_path, t0):
            _remove_partial(out_path)  # never let a retry write next to a half-written file
        data = _parse_json(out)
        exit_is_result = CATALOG[script].get("exit_code_means_fail") and data is not None
        ok = code == 0 or bool(exit_is_result)
        output = data.get("output") if isinstance(data, dict) else None
        if output is None and CATALOG[script]["produces_output"] and isinstance(op.args.get("output"), str):
            output = paths.get(op.args["output"], op.args["output"]) if ok else None
        return ToolResult(op_id=op.id, tool=op.tool, ok=ok, exit_code=code, output=output, data=data if isinstance(data, dict) else {"result": data},
                          commands=list((data or {}).get("commands", [])) if isinstance(data, dict) else [], stderr_tail="\n".join(err.strip().splitlines()[-12:]),
                          seconds=round(time.time() - t0, 2), attempt=attempt, dry_run=dry_run)

    # ---- convenience measurement helpers (used by MediaAnalyzer / QA; they build Operations too)
    def measure(self, tool: str, args: Dict[str, Any], paths: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> ToolResult:
        op = Operation(tool=tool, args=args, inputs=[], outputs=[], kind="measure")
        return self.run(op, paths or {}, timeout=timeout)


CHILD_ENV = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}  # ffmpeg-skill prints UTF-8 JSON; Windows consoles default to cp932 etc.


def run_process_group(cmd: List[str], timeout: Optional[float]) -> "tuple[int, str, str]":
    """Run cmd in its own process group so a timeout or an interrupt kills the script AND the ffmpeg it spawned.
    Returns (exit_code, stdout, stderr); exit 124 means timeout, 130 means interrupted."""
    env = dict(os.environ, **CHILD_ENV)
    kw: Dict[str, Any] = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "env": env}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kw)
    try:
        out_b, err_b = proc.communicate(timeout=timeout)
        return proc.returncode, _dec(out_b), _dec(err_b)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        out_b, err_b = proc.communicate()
        return 124, _dec(out_b), f"timeout after {timeout}s\n" + _dec(err_b)
    except KeyboardInterrupt:
        kill_tree(proc)
        proc.communicate()
        raise


def kill_tree(proc: "subprocess.Popen[bytes]") -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _dec(b: Optional[bytes]) -> str:
    return (b or b"").decode("utf-8", errors="replace")


def _is_fresh(path: str, t0: float) -> bool:
    try:
        return os.path.getmtime(path) >= t0 - 1
    except OSError:
        return False


def _remove_partial(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _parse_json(stdout: str) -> Any:
    s = stdout.strip()
    if not s:
        return None
    # the JSON document is the last thing on stdout; scripts may print a path first in odd modes
    start = s.find("{")
    if start < 0:
        return None
    try:
        return json.loads(s[start:])
    except ValueError:
        try:
            return json.loads(s.splitlines()[-1])
        except ValueError:
            return None
