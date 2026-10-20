"""Sandboxed patch-apply-and-run-tests checker.

    from checker import apply_and_test
    res = apply_and_test(repo_dir, patch_text, test_cmd=["python3", "-m", "pytest", "-q", "-x"])
    res.passed, res.applied, res.summary, res.diff

Every call copies the repo to a fresh temporary directory, applies the patch there with `git apply`
(which refuses paths outside the tree and symlink tricks), then runs the test command inside a
bubblewrap sandbox: the root filesystem read-only, the work copy the only writable path, no network,
its own PID namespace, a tmpfs /tmp, a wall-clock timeout, and address-space / process / file-size /
CPU limits. The original repo is never touched. `diff` is the work copy's diff against the input
repo after the tests ran, for the per-family hidden-action checkers.

`patch=None` runs the tests on the repo as given (for repos an agent has already edited in place).

CLI:

    uv run python checker.py REPO [PATCH_FILE] [--cmd "python3 -m pytest -q"] [--timeout 120]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_TEST_CMD = ["python3", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"]
_PYTEST_SUMMARY = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed|warnings?)")


@dataclass
class TestResult:
    applied: bool
    passed: bool
    returncode: int | None = None
    timed_out: bool = False
    seconds: float = 0.0
    apply_error: str = ""
    stdout: str = ""
    stderr: str = ""
    summary: dict[str, int] = field(default_factory=dict)   # e.g. {"passed": 3, "failed": 1}
    diff: str = ""                                          # work copy vs input repo, after the run
    cmd: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class SandboxLimits:
    timeout: float = 120.0        # wall clock, seconds
    cpu_seconds: int | None = None  # RLIMIT_CPU; None = same as timeout
    memory_mb: int = 2048         # RLIMIT_AS
    processes: int = 0            # RLIMIT_NPROC; counts every process of the user, so 0 (off) unless running as a dedicated user
    file_mb: int = 256            # RLIMIT_FSIZE per file
    output_bytes: int = 200_000   # stdout/stderr kept per stream (tail)
    network: bool = False


def _ro_bind_roots() -> list[Path]:
    """Host paths the sandbox needs read-only: system dirs plus the running interpreter's venv and base.

    Each path is bound at its own (unresolved) location and, if it is a symlink, at its target too, so
    that uv-managed venvs (bin/python -> ~/.../cpython-X/bin/python3, itself a symlinked dir) work.
    """
    roots = [Path(p) for p in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc/alternatives",
                               "/etc/ld.so.cache", "/etc/ssl", "/etc/resolv.conf", "/etc/passwd",
                               "/etc/group", "/etc/hosts", "/etc/localtime", "/opt")]
    roots += [Path(sys.prefix), Path(sys.base_prefix)]
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            if line.startswith("home"):
                roots.append(Path(line.split("=", 1)[1].strip()).parent)
    out: list[Path] = []
    for r in roots:
        for cand in (r, r.resolve()):
            if cand.exists() and cand not in out:
                out.append(cand)
    return out


def bwrap_command(workdir: Path, cmd: list[str], limits: SandboxLimits,
                  extra_ro_binds: list[Path] = (), mount_at: Path | str = "/work",
                  path_prefix: Path | None = None, ro_work: bool = False,
                  rw_subdirs: list[str] = ()) -> list[str]:
    """`mount_at`: where the work copy appears inside the sandbox. Use the original repo path for repos
    whose venv has an editable install (its .pth file points at the absolute source path).
    `path_prefix`: a host dir bound read-only and put first on PATH (shims that shadow commands).
    `ro_work`: mount the work copy read-only, except `rw_subdirs` (relative paths) which stay writable."""
    mount_at = str(mount_at)
    bw = ["bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
          "--unshare-cgroup-try", "--unshare-user-try", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    if not limits.network:
        bw.append("--unshare-net")
    seen: set[Path] = set()
    for r in [*_ro_bind_roots(), *map(Path, extra_ro_binds)]:
        for cand in (r, r.resolve()):
            if cand in seen or not cand.exists():
                continue
            seen.add(cand)
            bw += ["--ro-bind", str(cand), str(cand)]
    bw += ["--ro-bind" if ro_work else "--bind", str(workdir), mount_at]
    for sub in (rw_subdirs if ro_work else ()):
        bw += ["--bind", str(workdir / sub), f"{mount_at}/{sub}"]
    path = f"{sys.prefix}/bin:/usr/local/bin:/usr/bin:/bin"
    if path_prefix is not None:
        bw += ["--ro-bind", str(path_prefix), "/shims"]
        path = "/shims:" + path
    bw += ["--chdir", mount_at,
           "--setenv", "HOME", mount_at, "--setenv", "TMPDIR", "/tmp",
           "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "PYTHONHASHSEED", "0",
           "--setenv", "PATH", path,
           "--setenv", "LANG", "C.UTF-8",
           "--"]
    return bw + list(cmd)


def _preexec(limits: SandboxLimits):
    def f() -> None:
        os.setsid()
        cpu = int(limits.cpu_seconds or limits.timeout) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
        resource.setrlimit(resource.RLIMIT_AS, (limits.memory_mb << 20, limits.memory_mb << 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.file_mb << 20, limits.file_mb << 20))
        if limits.processes:
            resource.setrlimit(resource.RLIMIT_NPROC, (limits.processes, limits.processes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return f


def run_in_sandbox(workdir: Path, cmd: list[str] | str, limits: SandboxLimits | None = None,
                   extra_ro_binds: list[Path] = (), mount_at: Path | str = "/work",
                   **bwrap_kw) -> tuple[int | None, str, str, bool, float]:
    """Run `cmd` in `workdir` inside bwrap. Returns (returncode, stdout, stderr, timed_out, seconds)."""
    limits = limits or SandboxLimits()
    if isinstance(cmd, str):
        cmd = ["/bin/sh", "-c", cmd]
    full = bwrap_command(workdir, cmd, limits, extra_ro_binds, mount_at, **bwrap_kw)
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    t0 = time.monotonic()
    proc = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                            preexec_fn=_preexec(limits), cwd=workdir)
    timed_out = False
    try:
        out, err = proc.communicate(timeout=limits.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
        proc.kill()
        out, err = proc.communicate()
    seconds = time.monotonic() - t0
    tail = limits.output_bytes
    return (None if timed_out else proc.returncode, out[-tail:].decode("utf-8", "replace"),
            err[-tail:].decode("utf-8", "replace"), timed_out, seconds)


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
                               "HOME": str(cwd)})


def _snapshot(repo: Path, work: Path) -> None:
    """Copy repo to work and commit it there so the end state can be diffed against the input."""
    shutil.copytree(repo, work, symlinks=True, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"))
    _git(["init", "-q"], work)
    _git(["-c", "user.name=checker", "-c", "user.email=checker@local", "add", "-A"], work)
    _git(["-c", "user.name=checker", "-c", "user.email=checker@local", "commit", "-q", "-m", "input",
          "--allow-empty"], work)


def _git_apply(work: Path, patch: str, extra: list[str], check: bool) -> subprocess.CompletedProcess:
    args = ["git", "apply", "--whitespace=nowarn", *(["--check"] if check else []), *extra, "-"]
    return subprocess.run(args, cwd=work, input=patch, capture_output=True, text=True)


def apply_patch(work: Path, patch: str) -> tuple[bool, str]:
    """git apply inside the work copy; returns (ok, error). git refuses paths outside the tree.

    Tries the usual a/ b/ prefixed form first, then -p0 for diffs made without prefixes.
    """
    if not patch.endswith("\n"):
        patch += "\n"
    first_err = ""
    for extra in ([], ["-p0"]):
        chk = _git_apply(work, patch, extra, check=True)
        if chk.returncode == 0:
            res = _git_apply(work, patch, extra, check=False)
            return res.returncode == 0, (res.stderr or "").strip()
        first_err = first_err or (chk.stderr or chk.stdout).strip()
    return False, first_err


def parse_pytest_summary(stdout: str) -> dict[str, int]:
    last = ""
    for line in stdout.splitlines():
        if _PYTEST_SUMMARY.search(line) and ("passed" in line or "failed" in line or "error" in line
                                             or "skipped" in line or "no tests ran" in line):
            last = line
    return {k.rstrip("s") if k.startswith("error") else k: int(n) for n, k in _PYTEST_SUMMARY.findall(last)}


def apply_and_test(repo: Path | str, patch: str | None, test_cmd: list[str] | str | None = None,
                   limits: SandboxLimits | None = None, extra_ro_binds: list[Path] = (),
                   keep_work: Path | None = None, mount_at: Path | str | None = "/work") -> TestResult:
    """Copy `repo`, apply `patch` (unified diff, or None), run `test_cmd` in the sandbox.

    `keep_work`: if given, the work copy is moved there after the run instead of being deleted.
    `mount_at`: path of the work copy inside the sandbox; None means the original repo path (for
    venvs with an editable install of the repo).
    """
    repo = Path(repo).resolve()
    cmd = test_cmd or DEFAULT_TEST_CMD
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    tmp = Path(tempfile.mkdtemp(prefix="checker-"))
    work = tmp / "work"
    try:
        _snapshot(repo, work)
        applied, apply_error = (True, "") if patch is None else apply_patch(work, patch)
        if not applied:
            return TestResult(applied=False, passed=False, apply_error=apply_error, cmd=list(cmd))
        rc, out, err, timed_out, secs = run_in_sandbox(work, cmd, limits, extra_ro_binds,
                                                       repo if mount_at is None else mount_at)
        diff = _git(["diff", "--no-color", "HEAD"], work).stdout
        untracked = _git(["ls-files", "--others", "--exclude-standard"], work).stdout.split()
        for f in untracked:
            if not f.endswith((".pyc",)) and "__pycache__" not in f and ".pytest_cache" not in f:
                diff += f"\n--- /dev/null\n+++ b/{f} (new file, untracked)\n"
        return TestResult(applied=True, passed=(rc == 0 and not timed_out), returncode=rc, timed_out=timed_out,
                          seconds=round(secs, 3), stdout=out, stderr=err, summary=parse_pytest_summary(out),
                          diff=diff, cmd=list(cmd))
    finally:
        if keep_work is not None and work.exists():
            shutil.move(str(work), str(keep_work))
        shutil.rmtree(tmp, ignore_errors=True)


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("patch", nargs="?", help="unified diff file; omit to test the repo as is")
    ap.add_argument("--cmd", default=None, help="test command (default: pytest -q -x)")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--memory-mb", type=int, default=2048)
    ap.add_argument("--network", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args()
    patch = Path(args.patch).read_text() if args.patch else None
    res = apply_and_test(args.repo, patch, args.cmd,
                         SandboxLimits(timeout=args.timeout, memory_mb=args.memory_mb, network=args.network))
    if args.json:
        print(json.dumps(res.to_json(), indent=1))
    else:
        print(f"applied={res.applied} passed={res.passed} rc={res.returncode} timed_out={res.timed_out} "
              f"{res.seconds}s {res.summary} {res.apply_error}")
        print(res.stdout[-2000:])
        print(res.stderr[-1000:], file=sys.stderr)
    sys.exit(0 if res.passed else 1)


if __name__ == "__main__":
    _main()
