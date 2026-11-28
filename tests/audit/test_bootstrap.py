# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise LLVM bootstrap checks without running foreign-architecture code."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.bin = self.root / "bin"
        self.llvm = self.root / "LLVM tools"
        self.bin.mkdir()
        self.llvm.mkdir()
        self.executable(self.bin / "uname", "echo arm64")
        self.executable(self.bin / "arch", 'if [ "$#" -eq 0 ]; then echo arm64; else\n'
                        '    [ "$1" = -arm64 ] || exit 64\n'
                        '    shift\n'
                        '    exec "$@"\n'
                        'fi')
        self.executable(self.bin / "sysctl", "echo 0")
        for tool in ("rustc", "cargo", "xcodebuild", "xcrun"):
            self.executable(self.bin / tool, f"echo {tool}")
        self.executable(self.bin / "brew", '[ -n "$TEST_BREW_PREFIX" ] || exit 1\n'
                        'printf "%s\\n" "$TEST_BREW_PREFIX"')
        self.executable(self.bin / "lipo", '[ "$2" = -verify_arch ] && [ "$3" = arm64 ] || exit 64\n'
                        '[ "${1##*/}" != "$TEST_NON_NATIVE" ]')
        for tool in ("llvm-mc", "llvm-objcopy", "llvm-objdump"):
            self.executable(self.llvm / tool, f"echo {tool} version\necho full description")

    def executable(self, path, code):
        path.write_text(f"#!/bin/sh\n{code}\n", encoding="utf-8")
        path.chmod(0o755)

    def run_bootstrap(self, *, explicit=True, brew_prefix="", non_native=""):
        return subprocess.run(
            ["sh", str(ROOT / "scripts/bootstrap.sh")],
            cwd=self.root,
            env={
                **os.environ,
                "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
                "LLVM_BIN": str(self.llvm) if explicit else "",
                "TEST_BREW_PREFIX": str(brew_prefix),
                "TEST_NON_NATIVE": non_native,
            },
            capture_output=True,
            encoding="utf-8",
            check=False,
        )

    def test_explicit_missing_and_non_executable_tools_fail(self):
        tool = self.llvm / "llvm-mc"
        tool.unlink()
        for unavailable in ("missing", "non-executable"):
            with self.subTest(unavailable=unavailable):
                result = self.run_bootstrap()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(str(tool), result.stderr)
                self.assertNotIn("llvm-mc version", result.stdout)
            tool.touch()

    def test_failed_version_probe_preserves_exit_status(self):
        self.executable(self.llvm / "llvm-mc", "echo misleading version\nexit 42")

        result = self.run_bootstrap()

        self.assertEqual(result.returncode, 42)
        self.assertIn("llvm-mc", result.stderr)
        self.assertNotIn("misleading version", result.stdout)

    def test_non_native_tool_is_never_invoked(self):
        self.executable(self.llvm / "llvm-mc", "touch invoked")

        result = self.run_bootstrap(non_native="llvm-mc")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("llvm-mc", result.stderr)
        self.assertIn("arm64", result.stderr)
        self.assertFalse((self.root / "invoked").exists())

    def test_native_tools_print_only_first_version_line(self):
        result = self.run_bootstrap()

        self.assertEqual(result.returncode, 0, result.stderr)
        for tool in ("llvm-mc", "llvm-objcopy", "llvm-objdump"):
            self.assertIn(f"{tool} version", result.stdout)
        self.assertNotIn("full description", result.stdout)

    def test_llvm_discovery_remains_optional(self):
        result = self.run_bootstrap(explicit=False)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Optional fixture tooling", result.stdout)

    def test_failed_homebrew_probe_reports_unavailability(self):
        brew_prefix = self.root / "brew llvm"
        brew_prefix.mkdir()
        self.llvm.rename(brew_prefix / "bin")
        self.executable(brew_prefix / "bin/llvm-mc", "echo misleading version\nexit 42")

        result = self.run_bootstrap(explicit=False, brew_prefix=brew_prefix)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("llvm-mc", result.stderr)
        self.assertNotIn("misleading version", result.stdout)
        self.assertIn("llvm-objcopy version", result.stdout)
