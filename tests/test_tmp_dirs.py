"""ADR-115: the real-media tests' temporary directories are removed on success, failure, error, a failed setUpClass, a subprocess
timeout, KeyboardInterrupt and SIGTERM, and tests/test_integration.py makes no temporary directory except through tmp_dirs.tmp_dir.

Every check runs inside a directory of its own (tempfile's directory is pointed there, a child process gets it as TMPDIR) and
asserts that directory is empty afterwards: nothing outside it is read or counted."""
from __future__ import annotations

import ast
import io
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tmp_dirs  # noqa: E402
from tmp_dirs import tmp_dir  # noqa: E402


class TmpDirCleanupTests(unittest.TestCase):
    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp(prefix="va_tmpdirs_test_"))   # this test's own directory, removed below
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))

    def _run_cases(self, case_cls):
        """Run one TestCase class the way unittest does, with tempfile's directory pointed at self.root; returns (result, made dirs)."""
        made = []
        real = tempfile.mkdtemp

        def spy(*a, **k):
            path = real(*a, **k)
            made.append(path)
            return path
        with mock.patch.object(tempfile, "tempdir", self.root), mock.patch.object(tempfile, "mkdtemp", spy):
            result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(unittest.defaultTestLoader.loadTestsFromTestCase(case_cls))
        return result, made

    def assertRootEmpty(self, made):
        self.assertTrue(made, "the case made its directories")
        self.assertTrue(all(p.startswith(self.root) for p in made), made)
        self.assertEqual(os.listdir(self.root), [], "every directory tmp_dir made is gone")
        self.assertFalse(set(made) & tmp_dirs._LIVE, "and none is left for the exit handler")

    def test_removed_after_a_failing_test_and_an_erroring_test(self):
        class Case(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                cls.tmp = tmp_dir(cls, "va_int_")
                (Path(cls.tmp) / "big.bin").write_bytes(b"x" * 1024)

            def test_fails(self):
                (Path(tmp_dir(self, "va_case_")) / "o.mp4").write_bytes(b"x")
                self.fail("on purpose")

            def test_errors(self):
                tmp_dir(self, "va_case_")
                raise RuntimeError("on purpose")

            def test_engine_times_out(self):
                tmp_dir(self, "va_case_")
                raise subprocess.TimeoutExpired(["ffmpeg"], 600)
        result, made = self._run_cases(Case)
        self.assertEqual((len(result.failures), len(result.errors)), (1, 2), result.errors)
        self.assertEqual(len(made), 4)
        self.assertRootEmpty(made)

    def test_removed_when_setupclass_fails_after_making_it(self):
        class Case(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                cls.tmp = tmp_dir(cls, "va_int_")
                raise subprocess.CalledProcessError(1, ["ffmpeg"])   # e.g. make_media failed: tearDownClass is never called

            def test_never_runs(self):
                pass
        result, made = self._run_cases(Case)
        self.assertEqual(len(result.errors), 1)
        self.assertRootEmpty(made)

    def test_removed_after_passing_tests(self):
        class Case(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                cls.tmp = tmp_dir(cls, "va_int_")

            def test_ok(self):
                self.assertTrue(os.path.isdir(self.tmp))
                self.assertTrue(os.path.isdir(tmp_dir(self)))
        result, made = self._run_cases(Case)
        self.assertTrue(result.wasSuccessful(), result.errors)
        self.assertRootEmpty(made)

    def _child(self, body: str, **popen):
        """A child Python that imports tmp_dirs, with TMPDIR at self.root (so its directories can only be made there)."""
        code = f"import sys; sys.path.insert(0, {str(HERE)!r})\n" + textwrap.dedent(body)
        env = dict(os.environ, TMPDIR=self.root, TEMP=self.root, TMP=self.root)
        return subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **popen)

    def test_removed_after_keyboardinterrupt(self):
        """KeyboardInterrupt skips unittest's own cleanups: the directory goes at the interpreter's exit."""
        p = self._child("""
            import os, unittest
            from tmp_dirs import tmp_dir
            class Case(unittest.TestCase):
                @classmethod
                def setUpClass(cls):
                    cls.tmp = tmp_dir(cls, "va_int_")
                def test_interrupted(self):
                    open(os.path.join(self.tmp, "partial.mp4"), "wb").write(b"x" * 1024)
                    print(self.tmp, flush=True)
                    raise KeyboardInterrupt
            unittest.main(argv=["x"])
            """)
        out, err = p.communicate(timeout=60)
        self.assertNotEqual(p.returncode, 0, err)
        self.assertIn("KeyboardInterrupt", err)
        made = out.split()
        self.assertEqual(len(made), 1, (out, err))
        self.assertTrue(os.path.realpath(made[0]).startswith(self.root), made)
        self.assertEqual(os.listdir(self.root), [])

    @unittest.skipIf(os.name == "nt", "SIGTERM cannot be delivered to a Python handler on Windows")
    def test_removed_after_sigterm(self):
        """A CI timeout / cancel ends the run with SIGTERM: the directory goes too."""
        p = self._child("""
            import os, time
            from tmp_dirs import tmp_dir
            class Owner:
                @classmethod
                def addClassCleanup(cls, *a): pass   # never run: only the SIGTERM exit removes it
            print(tmp_dir(Owner, "va_int_"), flush=True)
            try:
                time.sleep(60)
            except BaseException:   # code under test that swallows every exception (as unittest does) must not keep it
                time.sleep(60)
            """)
        line = p.stdout.readline().strip()
        self.assertTrue(line and os.path.isdir(line), line)
        p.send_signal(signal.SIGTERM)
        _, err = p.communicate(timeout=30)
        self.assertEqual(p.returncode, -signal.SIGTERM, err)   # ended by the signal itself
        self.assertEqual(os.listdir(self.root), [])

    def test_integration_tests_make_no_temporary_directory_any_other_way(self):
        """A new real-media class written with a bare tempfile.mkdtemp() / mkstemp() would leak again: none may appear there."""
        tree = ast.parse((HERE / "test_integration.py").read_text(encoding="utf-8"))
        bare = sorted({(n.lineno, n.func.attr) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and n.func.attr in ("mkdtemp", "mkstemp", "TemporaryDirectory", "NamedTemporaryFile")}
                      | {(n.lineno, n.func.id) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                         and n.func.id in ("mkdtemp", "mkstemp")})
        self.assertEqual(bare, [], "tests/test_integration.py: use tmp_dirs.tmp_dir(cls_or_self, prefix) instead")
        uses = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "tmp_dir"]
        self.assertGreaterEqual(len(uses), 40, "the real-media classes make their directories through tmp_dir")
        self.assertTrue(all(isinstance(u.args[0], ast.Name) and u.args[0].id in ("cls", "self") for u in uses), "owned by the class or the test")


if __name__ == "__main__":
    unittest.main()
