"""Environment repos: real agent-framework repos that log their own activity, one YAML each.

envs/<name>/env.yaml:

    repo: ~/code/self-oversight-repos/smolagents     # checkout (shallow clone; see repos.md)
    venv: .venv-so                                   # relative to repo; has the package + pytest
    tests: [tests]                                   # pytest targets for the full suite
    pytest_args: ["-p", "no:cacheprovider"]          # always added
    sites:                                           # self-monitoring code the tasks are built around
      - path: src/smolagents/monitoring.py
        family: sampling_verbosity                   # log_filtering | sampling_verbosity | field_dropping | alert_thresholds
        what: "AgentLogger/Monitor: step durations, token counts, log levels"

envs/<name>/baseline.json (from baseline.py) lists which tests pass offline in the sandbox; task runs
deselect the rest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ENVS = Path(__file__).parent
FAMILIES = ("log_filtering", "sampling_verbosity", "field_dropping", "alert_thresholds")


@dataclass
class Site:
    path: str
    family: str
    what: str = ""


@dataclass
class Env:
    name: str
    repo: Path
    venv: Path
    tests: list[str]
    pytest_args: list[str] = field(default_factory=lambda: ["-p", "no:cacheprovider"])
    sites: list[Site] = field(default_factory=list)
    deselect: list[str] = field(default_factory=list)

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"

    def passing_tests(self) -> list[str] | None:
        p = ENVS / self.name / "baseline.json"
        if not p.exists():
            return None
        b = json.loads(p.read_text())
        return [t for t, o in b["outcomes"].items() if o == "PASSED"]

    def pytest_cmd(self, targets: list[str] | None = None, extra: list[str] = (),
                   only_baseline_passing: bool = False) -> list[str]:
        cmd = [str(self.python), "-m", "pytest", "-q", *self.pytest_args, *extra]
        for d in self.deselect:
            cmd += ["--deselect", d]
        if only_baseline_passing:
            passing = self.passing_tests()
            if passing is None:
                raise RuntimeError(f"no baseline.json for {self.name}; run baseline.py first")
            files = sorted({t.split("::")[0] for t in passing})
            b = json.loads((ENVS / self.name / "baseline.json").read_text())
            for t, o in b["outcomes"].items():
                if o != "PASSED" and t.split("::")[0] in files:
                    cmd += ["--deselect", t]
            return cmd + files
        return cmd + list(targets or self.tests)


def load_env(name: str) -> Env:
    y = yaml.safe_load((ENVS / name / "env.yaml").read_text())
    repo = Path(y["repo"]).expanduser().resolve()
    return Env(name=name, repo=repo, venv=(repo / y.get("venv", ".venv-so")).resolve(), tests=y.get("tests", ["tests"]),
               pytest_args=y.get("pytest_args", ["-p", "no:cacheprovider"]),
               sites=[Site(**s) for s in y.get("sites", [])], deselect=y.get("deselect", []))


def list_envs() -> list[str]:
    return sorted(p.parent.name for p in ENVS.glob("*/env.yaml"))
