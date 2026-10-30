"""Offline tests for the sandboxed checker. Need bwrap and the project venv (pytest) on this machine."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from checker import SandboxLimits, apply_and_test, parse_pytest_summary, run_in_sandbox

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    (r / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (r / "test_calc.py").write_text("from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    (r / "secret.txt").write_text("original\n")
    return r


GOOD_PATCH = textwrap.dedent("""\
    --- a/calc.py
    +++ b/calc.py
    @@ -1,2 +1,2 @@
     def add(a, b):
    -    return a - b
    +    return a + b
    """)

ESCAPE_PATCH = textwrap.dedent("""\
    --- /dev/null
    +++ b/../escaped.txt
    @@ -0,0 +1 @@
    +pwned
    """)


def test_bad_repo_fails(repo: Path):
    res = apply_and_test(repo, None)
    assert res.applied and not res.passed and res.summary.get("failed") == 1


def test_good_patch_passes_and_leaves_repo_untouched(repo: Path):
    res = apply_and_test(repo, GOOD_PATCH)
    assert res.applied and res.passed, res.stdout + res.stderr
    assert res.summary == {"passed": 1}
    assert "return a + b" in res.diff
    assert (repo / "calc.py").read_text() == "def add(a, b):\n    return a - b\n"


def test_nonapplying_patch(repo: Path):
    res = apply_and_test(repo, GOOD_PATCH.replace("return a - b", "return a * b"))
    assert not res.applied and not res.passed and res.apply_error


def test_patch_cannot_escape_tree(repo: Path, tmp_path: Path):
    res = apply_and_test(repo, ESCAPE_PATCH)
    assert not res.applied
    assert not (tmp_path / "escaped.txt").exists()


def test_timeout_kills_runaway(repo: Path):
    res = apply_and_test(repo, None, ["python3", "-c", "while True: pass"], SandboxLimits(timeout=2))
    assert res.timed_out and not res.passed and res.seconds < 10


def test_no_network(repo: Path):
    rc, out, err, _, _ = run_in_sandbox(repo, ["python3", "-c",
        "import socket; s=socket.socket(); s.settimeout(3); s.connect(('1.1.1.1', 80))"], SandboxLimits(timeout=20))
    assert rc != 0, out + err


def test_root_is_read_only_and_work_is_writable(repo: Path):
    rc, out, err, _, _ = run_in_sandbox(repo, ["/bin/sh", "-c", "touch /usr/zz 2>/dev/null; echo rc=$?; "
                                                "touch /work/ok && echo wrote"], SandboxLimits(timeout=20))
    assert "rc=1" in out and "wrote" in out, out + err


def test_memory_limit(repo: Path):
    rc, out, err, _, _ = run_in_sandbox(repo, ["python3", "-c", "x = bytearray(3 << 30)"],
                                        SandboxLimits(timeout=30, memory_mb=512))
    assert rc != 0 and "MemoryError" in err


def test_pytest_summary_parse():
    assert parse_pytest_summary("....\n3 passed, 1 failed, 2 skipped in 0.1s\n") == {"passed": 3, "failed": 1, "skipped": 2}
    assert parse_pytest_summary("1 error in 0.1s") == {"error": 1}
    assert parse_pytest_summary("garbage") == {}
