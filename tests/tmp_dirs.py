"""ADR-115: temporary directories of the real-media tests that are always removed.

`tmp_dir(owner, prefix)` makes a directory with `tempfile.mkdtemp` and removes it again, whatever happens to the test:

- `owner` a TestCase class (in `setUpClass`): `addClassCleanup`, which unittest and pytest run after the class's last test, and
  also when `setUpClass` itself fails (where `tearDownClass` is never called);
- `owner` a TestCase instance: `addCleanup`, run after that test whether it passed, failed or raised (an engine's
  `subprocess.TimeoutExpired` included);
- a KeyboardInterrupt (Ctrl-C, or a CI cancel's SIGINT) skips unittest's cleanups, and SIGTERM ends the process without them: every
  directory still there is removed at interpreter exit (`atexit`); on SIGTERM (handled once this module made a directory, only when
  nothing else handles SIGTERM, and only from the main thread) they are removed and the process then ends by that signal, even
  while a test that catches every exception runs. SIGKILL cannot be caught by anything.

Each real-media class wrote its source clips, intermediates and deliverables (hundreds of MB) under one such directory and never
removed it; 22 of them filled a shared disk. tests/test_tmp_dirs.py checks each of the ways above, and that tests/test_integration.py
makes no temporary directory any other way.
"""
from __future__ import annotations

import atexit
import os
import shutil
import signal
import tempfile
import threading
from typing import Any, Set

_LIVE: Set[str] = set()
_SIGTERM_SET = False


def _remove(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
    if not os.path.exists(path):   # still tracked (for the exit handler) when a running child kept it from going
        _LIVE.discard(path)


@atexit.register
def remove_all() -> None:
    """Remove every directory tmp_dir made that is still there (the interpreter's exit, after a KeyboardInterrupt or SIGTERM)."""
    for path in sorted(_LIVE):
        _remove(path)


def _on_sigterm(signum: int, _frame: Any) -> None:
    # remove first, then end the process the way SIGTERM would have: an exception raised here (SystemExit) could be caught by
    # the code under test (unittest records it as an error and goes on), so the removal does not depend on one
    remove_all()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _catch_sigterm() -> None:
    global _SIGTERM_SET
    if _SIGTERM_SET or threading.current_thread() is not threading.main_thread():
        return
    _SIGTERM_SET = True
    try:
        if signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, None):
            signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError, AttributeError):   # no SIGTERM handler possible on this platform / thread
        pass


def tmp_dir(owner: Any, prefix: str = "va_") -> str:
    """A new temporary directory, removed when `owner` (a TestCase class or instance) is cleaned up, or at exit at the latest."""
    path = tempfile.mkdtemp(prefix=prefix)
    _LIVE.add(path)
    _catch_sigterm()
    if isinstance(owner, type):
        owner.addClassCleanup(_remove, path)
    else:
        owner.addCleanup(_remove, path)
    return path
