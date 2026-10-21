"""restart.sh failures: an app the agent broke is AGENT (reward 0) on both scoring paths, and real container trouble stays INFRA.

envs/tiktok/restart.sh prints "app not ready 150s after restart" to stderr and then `tail -30 /tmp/next-dev.log`, so the marker sits at the head
of a long stderr. Run.py and rl.guard used to keep only its last 500 characters, which dropped the marker and turned a broken app into INFRA
(rerun or excluded instead of scored 0). Offline: fake sandboxes, mockllm, no docker, no API.
"""
import asyncio
import io
import random
import tarfile
from pathlib import Path

import pytest
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import solver
from inspect_ai.util import ExecResult

import run
from fakes import make_episode
from rl import guard
from rl.classify import AGENT, INFRA, classify_score
from rl.guard import GuardCfg, Reference
from rl.rewards import attach_reward, episode_reward, read_inspect_score

ENV = Path(__file__).resolve().parents[2] / "envs" / "tiktok"
MARKER = "app not ready 150s after restart"
# what the tail -30 of a broken Next.js dev server looks like: well over 500 characters after the marker
NEXT_LOG = "\n".join([f" ⨯ ./src/app/api/random-video/route.ts:{i}:5 Error: Expected ';', '}}' or <eof> (compile step {i} of the dev server)" for i in range(30)])
APP_NOT_READY_STDERR = f"{MARKER}\n{NEXT_LOG}\n"
# the same log, written by the agent's own app, can carry docker-looking text; the marker printed by restart.sh itself still decides
FORGED_STDERR = f"{MARKER}\n{NEXT_LOG}\nError response from daemon: No such container: la-tiktok-default-1\n"
DOCKER_DEAD = "Error response from daemon: container 3f2a is not running"


def test_restart_sh_prints_marker_before_the_log_dump():
    """The premise, read from the vendored script: marker first, then 30 log lines, exit 1."""
    lines = [l.strip() for l in (ENV / "restart.sh").read_text().splitlines()]
    i = lines.index(f'echo "{MARKER}" >&2')
    assert lines[i + 1] == "tail -30 /tmp/next-dev.log >&2" and lines[i + 2] == "exit 1"
    assert len(APP_NOT_READY_STDERR) > 1500 and MARKER not in APP_NOT_READY_STDERR[-500:]


def test_head_tail():
    from rl.classify import head_tail
    assert head_tail("short") == "short" and head_tail(None) == ""
    clipped = head_tail(APP_NOT_READY_STDERR)
    assert clipped.startswith(MARKER) and clipped.endswith(APP_NOT_READY_STDERR[-500:]) and len(clipped) < 1100


@solver
def noop():
    async def solve(state, generate):
        return state
    return solve


def task_dir(tmp_path):
    d = tmp_path / "task"
    d.mkdir()
    (d / "tests.py").write_text("def test_x():\n    assert True\n")
    return d


def episode_for(log):
    ep = make_episode(random.Random(0), sample_id="s", epoch=1, passed=None, log_path=log.location)
    attach_reward(ep)
    return ep


# ---- run.py's task_scorer (the plain scorer) --------------------------------------------------------------------------------------------

class PlainSbx:
    async def write_file(self, *a, **k):
        pass

    async def exec(self, cmd, timeout=None, **k):
        return ExecResult(False, 1, "FAILED /tmp/tests.py::test_x\n1 failed in 0.4s", "")


def run_plain(tmp_path, monkeypatch, restart):
    async def fake_run_script(sbx, path, timeout):
        assert path.name == "restart.sh"
        if isinstance(restart, BaseException):
            raise restart
        return restart
    monkeypatch.setattr(run, "sandbox", lambda *a, **k: PlainSbx())
    monkeypatch.setattr(run, "run_script", fake_run_script)
    t = Task(dataset=[Sample(input="x", target="y", id="s")], solver=noop(), scorer=run.task_scorer(ENV, task_dir(tmp_path)))
    log = inspect_eval(t, model="mockllm/model", log_dir=str(tmp_path / "logs"), fail_on_error=False, display="none")[0]
    return log, read_inspect_score(log.location, "s", 1)


@pytest.mark.parametrize("stderr", [APP_NOT_READY_STDERR, FORGED_STDERR], ids=["log_dump", "forged_docker_text"])
def test_plain_scorer_broken_app_is_agent_reward_zero(tmp_path, monkeypatch, stderr):
    log, r = run_plain(tmp_path, monkeypatch, ExecResult(False, 1, "", stderr))
    assert MARKER in log.samples[0].scores["task_scorer"].explanation
    assert r["passed"] is False and (r["failure_class"], r["failure_reason"]) == (AGENT, "app_not_ready")
    assert episode_reward(episode_for(log)) == 0.0


def test_plain_scorer_short_stderr_is_unchanged(tmp_path, monkeypatch):
    log, _ = run_plain(tmp_path, monkeypatch, ExecResult(False, 1, "", DOCKER_DEAD))
    assert log.samples[0].scores["task_scorer"].explanation.startswith(f"restart.sh failed: {DOCKER_DEAD}\n\n")


@pytest.mark.parametrize("restart,reason", [
    (ExecResult(False, 1, "", DOCKER_DEAD), "container_lost"),                                         # docker exec into a dead container
    (ExecResult(False, 137, "", "container exited with code 137 (OOMKilled)"), "oom_killed"),
    (ExecResult(False, 1, "", "x" * 2000 + DOCKER_DEAD), "container_lost"),                           # long output, docker error at the end
    (RuntimeError("Error response from daemon: No such container: la-tiktok-default-1"), "container_lost"),
])
def test_plain_scorer_container_death_stays_infra(tmp_path, monkeypatch, restart, reason):
    log, r = run_plain(tmp_path, monkeypatch, restart)
    assert r["passed"] is None and r["failure_class"] == INFRA and r["failure_reason"].split(":")[0] == reason
    assert episode_reward(episode_for(log)) is None


# ---- rl.guard (the hardened scorer) -------------------------------------------------------------------------------------------------------

def empty_tar() -> bytes:
    buf = io.BytesIO()
    tarfile.open(fileobj=buf, mode="w").close()
    return buf.getvalue()


class GuardSbx:
    """Every pre-restart step succeeds on an empty tree and an empty manifest (matching Reference('x', {}, {})); restart.sh returns `restart`."""
    def __init__(self, restart):
        self.restart = restart

    async def write_file(self, *a, **k):
        pass

    async def read_file(self, path, text=True):
        return empty_tar()

    async def exec(self, cmd, timeout=None, **k):
        if cmd == ["bash", "/tmp/arena_restart.sh"]:
            if isinstance(self.restart, BaseException):
                raise self.restart
            return self.restart
        return ExecResult(True, 0, "", "")


def guard_check(tmp_path, restart):
    return asyncio.run(guard.guarded_check(GuardSbx(restart), ENV, task_dir(tmp_path), GuardCfg(), Reference("x", {}, {})))


@pytest.mark.parametrize("stderr", [APP_NOT_READY_STDERR, FORGED_STDERR], ids=["log_dump", "forged_docker_text"])
def test_guard_broken_app_is_app_not_ready(tmp_path, stderr):
    r = guard_check(tmp_path, ExecResult(False, 1, "", stderr))
    assert (r.status, r.reason, r.passed) == ("ok", "app_not_ready", False)
    assert classify_score(f"[guard:{r.reason}] {r.explanation}").failure_class == AGENT


@pytest.mark.parametrize("restart", [ExecResult(False, 1, "", DOCKER_DEAD), ExecResult(False, 1, "", "x" * 2000 + DOCKER_DEAD),
                                     RuntimeError("Error response from daemon: No such container: x")])
def test_guard_container_death_stays_infra(tmp_path, restart):
    r = guard_check(tmp_path, restart)
    assert r.status == "infra_error" and r.reason in ("restart_failed", "scorer_RuntimeError")


def run_guarded(tmp_path, monkeypatch, restart):
    import inspect_ai.util
    monkeypatch.setattr(guard, "load_reference", lambda env, *a, **k: Reference("x", {}, {}))
    monkeypatch.setattr(inspect_ai.util, "sandbox", lambda *a, **k: GuardSbx(restart))
    t = Task(dataset=[Sample(input="x", target="y", id="s")], solver=noop(), scorer=guard.guarded_task_scorer(ENV, task_dir(tmp_path), GuardCfg()))
    log = inspect_eval(t, model="mockllm/model", log_dir=str(tmp_path / "logs"), fail_on_error=False, display="none")[0]
    return log, read_inspect_score(log.location, "s", 1)


def test_guarded_scorer_broken_app_is_agent_reward_zero_end_to_end(tmp_path, monkeypatch):
    log, r = run_guarded(tmp_path, monkeypatch, ExecResult(False, 1, "", APP_NOT_READY_STDERR))
    assert log.samples[0].error is None
    assert r["passed"] is False and (r["failure_class"], r["failure_reason"]) == (AGENT, "app_not_ready")
    assert episode_reward(episode_for(log)) == 0.0


def test_guarded_scorer_container_death_is_infra_end_to_end(tmp_path, monkeypatch):
    log, r = run_guarded(tmp_path, monkeypatch, ExecResult(False, 1, "", DOCKER_DEAD))
    assert r["passed"] is None and r["failure_class"] == INFRA and r["failure_reason"].startswith("restart_failed")
    assert episode_reward(episode_for(log)) is None
