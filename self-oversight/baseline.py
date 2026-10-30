"""Run a repo's whole test suite offline in the sandbox and record which tests pass there.

    uv run python baseline.py smolagents [--tests tests] [--timeout 1800]

Writes envs/<repo>/baseline.json: the pytest-reported outcome per test id, so that later runs can
deselect everything that does not pass offline (network, credentials, GPUs, flaky). Those tests are
not evidence about a patch either way.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from checker import SandboxLimits, run_in_sandbox
from envs import ENVS, Env, load_env

# Lines of the -rA short test summary: "PASSED tests/x.py::test_y" / "ERROR tests/x.py::test_y - msg".
_LINE = re.compile(r"^(?P<outcome>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) (?P<id>\S+::\S+)", re.M)


def run_baseline(env: Env, timeout: float = 1800, tests: list[str] | None = None) -> dict:
    cmd = env.pytest_cmd(tests or env.tests, extra=["-rA", "-v", "--tb=line", "--color=no", "--continue-on-collection-errors"])
    rc, out, err, timed_out, secs = run_in_sandbox(env.repo, cmd, SandboxLimits(timeout=timeout, memory_mb=8192),
                                                   extra_ro_binds=[env.venv], mount_at=env.repo)
    outcomes = {m.group("id"): m.group("outcome") for m in _LINE.finditer(out)}
    counts: dict[str, int] = {}
    for o in outcomes.values():
        counts[o] = counts.get(o, 0) + 1
    result = {"repo": env.name, "returncode": rc, "timed_out": timed_out, "seconds": round(secs, 1),
              "counts": counts, "outcomes": outcomes, "stderr_tail": err[-2000:], "stdout_tail": out[-3000:]}
    out_path = ENVS / env.name / "baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--tests", nargs="*", default=None)
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    res = run_baseline(load_env(a.repo), a.timeout, a.tests)
    print(json.dumps({k: res[k] for k in ("repo", "returncode", "timed_out", "seconds", "counts")}))
    if not res["outcomes"]:
        print(res["stdout_tail"][-1500:], res["stderr_tail"][-800:], file=sys.stderr)


if __name__ == "__main__":
    main()
