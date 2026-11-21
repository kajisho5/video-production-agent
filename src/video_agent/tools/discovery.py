"""Where a Skill checkout is looked for when nobody named one, and the environment a Skill child process gets (ADR-069).

Discovery never depends on the current directory: starting the agent inside a folder (an unpacked project, a media card,
a download) must not make that folder's `vendor/<skill>` or its parent's `<skill>` the code that runs. When no explicit
directory or `VIDEO_AGENT_*_DIR` is given, the defaults are `~/.claude/skills/<name>` and, only when this package runs
from its own source checkout (`<root>/src/video_agent` with a `pyproject.toml` or `.git` at `<root>`), the sibling
`<root>/../<name>` -- the ecosystem's development layout and the CI integration job, wherever the command is started.

A Skill child process gets the agent's environment minus the AI provider credentials the agent itself reads
(`ANTHROPIC_*`, `OPENAI_*`): no Skill calls a model on the agent's behalf, so none needs them.

A Skill checkout is run as a module without the current directory on its import path (ADR-074): `python -m` puts the
current directory first on `sys.path`, so a `json.py`, `numpy/` or `<package>/` in the folder the agent was started in
would be imported by the Skill instead of the real one. `python_module_command` keeps `PYTHONPATH` (the checkout's
`src`) and the child's working directory (relative paths resolve as before), and drops only that entry.

The rest of what a child could pick up from the folder the agent was started in is closed here too (ADR-127): the program
a child runs is an absolute path found on the absolute entries of PATH only (`safe_which`, `resolve_executable`: never the
current directory, which Windows searches first and an empty or `.` PATH entry names on POSIX); `PYTHONPATH` keeps only
absolute entries (an empty or relative entry is the current directory again, made absolute by `site` before any bootstrap
runs); and while a job runs, every child gets that job's own private temporary directory as TMPDIR / TEMP / TMP
(`job_temp_dir`), removed when the job ends."""
from __future__ import annotations

import contextlib
import contextvars
import errno
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional

# environment variables of the AI providers the agent talks to (providers/claude.py, capabilities/resolver.py AI_ENV)
SECRET_ENV_PREFIXES = ("ANTHROPIC_", "OPENAI_")


def source_checkout_root() -> Optional[Path]:
    """The repository root when this package is imported from its source checkout (an editable install or PYTHONPATH=src),
    else None (a wheel in site-packages has no siblings to offer)."""
    here = Path(__file__).resolve()
    pkg = here.parents[1]
    root = pkg.parent.parent
    if pkg.name == "video_agent" and pkg.parent.name == "src" and ((root / "pyproject.toml").is_file() or (root / ".git").exists()):
        return root
    return None


def _home() -> Path:
    return Path.home()


def default_skill_dirs(name: str) -> List[Path]:
    """Guessed locations of a Skill checkout, in order, when neither an explicit dir nor the env var names one."""
    out = [_home() / ".claude" / "skills" / name]
    root = source_checkout_root()
    if root is not None:
        out.append(root.parent / name)
    return out


def is_secret_env(key: str) -> bool:
    return key.upper().startswith(SECRET_ENV_PREFIXES)


TEMP_ENV = ("TMPDIR", "TEMP", "TMP")   # where tempfile (and most programs) put their files, on POSIX and on Windows
_JOB_TEMP: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar("video_agent_job_temp", default=None)


def is_fully_qualified(path: str) -> bool:
    """An absolute path that does not depend on the current directory or drive: on Windows a drive with a root (`C:\\x`)
    or a UNC path (`\\\\host\\share\\x`), never `\\x` (the current drive's root) or `C:x` (relative to C:'s current
    directory), which `os.path.isabs` accepts before Python 3.13 (ADR-127)."""
    if not path or not os.path.isabs(path):
        return False
    if os.name == "nt":
        drive, rest = os.path.splitdrive(path)
        return bool(drive) and rest[:1] in ("\\", "/")
    return True


def safe_pythonpath(value: str) -> str:
    """PYTHONPATH with only its absolute entries: an empty entry is the current directory and a relative one resolves
    against it, so either would let a module in the folder the agent was started in shadow the real one (ADR-127)."""
    return os.pathsep.join(p for p in value.split(os.pathsep) if is_fully_qualified(p))


def child_env(extra: Optional[Mapping[str, str]] = None, base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment for a Skill (or engine) child process: `base` (default os.environ) without provider credentials, plus `extra`.
    PYTHONPATH keeps only its absolute entries, and inside `job_temp_dir` the temporary directory is the job's own (ADR-127)."""
    env = {k: v for k, v in (os.environ if base is None else base).items() if not is_secret_env(k)}
    env.update(extra or {})
    if "PYTHONPATH" in env:
        kept = safe_pythonpath(env["PYTHONPATH"])
        if kept:
            env["PYTHONPATH"] = kept
        else:
            del env["PYTHONPATH"]
    job_tmp = _JOB_TEMP.get()
    if job_tmp:
        env.update({k: job_tmp for k in TEMP_ENV})
    return env


def current_job_temp_dir() -> Optional[str]:
    """The temporary directory children get while a job runs (`job_temp_dir`), else None (they inherit the agent's)."""
    return _JOB_TEMP.get()


@contextlib.contextmanager
def job_temp_dir(job_id: str = "") -> Iterator[str]:
    """One private temporary directory (mkdtemp: 0700, an unguessable name under the system's temporary directory) for every
    child process a job starts; removed, with whatever a child left in it, when the job ends however it ends (ADR-127)."""
    path = tempfile.mkdtemp(prefix=f"video-agent-{job_id or 'job'}-")
    token = _JOB_TEMP.set(path)
    try:
        yield path
    finally:
        _JOB_TEMP.reset(token)
        shutil.rmtree(path, ignore_errors=True)


def _exe_names(name: str) -> List[str]:
    if os.name != "nt":
        return [name]
    exts = [e for e in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(os.pathsep) if e]
    if any(name.lower().endswith(e.lower()) for e in exts):
        return [name]
    return [name + e for e in exts]


def safe_which(name: str, path: Optional[str] = None) -> Optional[str]:
    """`shutil.which` without the current directory: the absolute path of `name` on the absolute entries of `path` (default
    PATH) only. shutil.which looks in the current directory first on Windows, and an empty or `.` PATH entry is the current
    directory on POSIX, so a `ffmpeg.exe` or `node` planted in the folder the agent was started in would be what runs (ADR-127).
    A name with a directory part is returned made absolute when it is an executable file there."""
    if os.sep in name or (os.altsep and os.altsep in name):
        p = os.path.abspath(name)
        return p if os.path.isfile(p) and os.access(p, os.X_OK) else None
    dirs = (os.environ.get("PATH", os.defpath) if path is None else path).split(os.pathsep)
    for d in dirs:
        if not is_fully_qualified(d):
            continue
        for n in _exe_names(name):
            p = os.path.join(d, n)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    return None


def resolve_executable(program: str, path: Optional[str] = None) -> str:
    """The absolute path a child's argv[0] is launched by (ADR-127): a path is made absolute against the agent's working
    directory, a bare name is looked up by `safe_which`. A name found nowhere raises FileNotFoundError, as Popen would --
    never a lookup that falls back to the current directory."""
    if is_fully_qualified(program):
        return program
    if os.path.isabs(program) or os.sep in program or (os.altsep and os.altsep in program):
        return os.path.abspath(program)   # a path, made absolute against the agent's working directory (and drive)
    found = safe_which(program, path)
    if found is None:
        raise FileNotFoundError(errno.ENOENT, f"executable not found on PATH (the current directory is never searched): {program}", program)
    return found


def safe_command(cmd: List[str]) -> List[str]:
    """`cmd` with argv[0] resolved by `resolve_executable`; the arguments are unchanged (relative paths in them keep resolving
    against the working directory the child inherits, ADR-074)."""
    return [resolve_executable(cmd[0]), *cmd[1:]] if cmd else list(cmd)


# What `python -m <module>` does, minus the current directory on sys.path. `-c` inserts '' (the current directory) at
# sys.path[0] the way `-m` inserts its absolute path; it is removed before anything else is imported (`sys` is built in,
# `runpy` and what it imports are looked up only afterwards). `runpy._run_module_as_main` is the function the interpreter
# itself calls for `-m` (sys.argv[0] becomes the module's file, `__main__` is the real main module), so the Skill sees the
# same argv and module state as before. `-P` / PYTHONSAFEPATH would do the same but only on Python 3.11+, and `-I` / `-E`
# would also drop PYTHONPATH, which is how a checkout is found.
MODULE_BOOTSTRAP = "import sys; sys.path[:] = [p for p in sys.path if p]; import runpy; runpy._run_module_as_main(sys.argv.pop(1))"


def python_module_command(module: str, python: Optional[str] = None) -> List[str]:
    """The argv prefix that runs a Skill checkout's `module` (with PYTHONPATH=<checkout>/src in the child's environment)
    as `python -m` would, except that the current directory is never on its import path (ADR-074)."""
    return [python or sys.executable, "-c", MODULE_BOOTSTRAP, module]
