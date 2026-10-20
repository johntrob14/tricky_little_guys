import asyncio

import pytest

from rl.classify import (AGENT, HALT, INFRA, EpisodeFailure, Failure, HaltRules, Halt, Ledger, RetryPolicy, classify_exception, classify_log, classify_sample, classify_score,
                         classify_text, run_with_retries, write_halted)


@pytest.mark.parametrize("text,phase,running,cls,reason", [
    ("Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?", None, None, INFRA, "docker_daemon_unreachable"),
    ("failed to create network: all predefined address pools have been fully subnetted", None, None, INFRA, "network_pool_exhausted"),
    ("write /var/lib/docker/x: no space left on device", None, None, INFRA, "disk_full"),
    ('Conflict. The container name "/la-x-default-1" is already in use', None, None, INFRA, "name_conflict"),
    ("pull access denied for linuxarena/env-tiktok", None, None, INFRA, "image_missing"),
    ("toomanyrequests: You have reached your pull rate limit", None, None, INFRA, "registry_rate_limited"),
    ("container exited with code 137 (OOMKilled)", None, None, INFRA, "oom_killed"),
    ("RuntimeError: tiktok: ready.sh did not succeed within 240s", "ready", None, INFRA, "ready_timeout"),
    ("AssertionError: setup.sh failed: patch: **** malformed", "setup", None, INFRA, "setup_failed"),
    ('service "default" is not running', "exec", None, INFRA, "container_lost"),
    ("Error response from daemon: No such container: abc", "exec", None, INFRA, "container_lost"),
    ("dependency failed to start: container x-db-1 is unhealthy", "create", None, INFRA, "compose_up_failed"),
    ("TimeoutError command exceeded 120s", "exec", True, AGENT, "tool_timeout"),
    ("TimeoutError command exceeded 120s", "exec", False, INFRA, "container_lost"),
    ("TimeoutError command exceeded 120s", "exec", None, INFRA, "exec_hang"),
    ("TimeoutError command exceeded 300s", "score", None, INFRA, "scorer_timeout"),
    ("tests timed out after 300s", "score", None, AGENT, "tests_timeout"),
    ("restart.sh failed: app not ready 150s after restart", None, None, AGENT, "app_not_ready"),
    ("restart.sh failed: Error response from daemon: container is not running", None, None, INFRA, "container_lost"),
    ("APIConnectionError: connection reset by peer talking to tinker", "tinker", None, INFRA, "tinker_error"),
    ("503 Service Unavailable", "tinker", None, INFRA, "tinker_error"),
    ("tinker.AuthenticationError 401 invalid api key", "tinker", None, HALT, "tinker_auth"),
    ("BudgetExceeded: stage cap", None, None, HALT, "budget_exceeded"),
    ("KeyError: 'weird'", None, None, INFRA, "unclassified"),
])
def test_classify_text(text, phase, running, cls, reason):
    f = classify_text(text, phase, running)
    assert (f.failure_class, f.reason) == (cls, reason)


def test_classify_exception_and_score():
    assert classify_exception(EpisodeFailure(Failure(INFRA, "x"))).reason == "x"
    assert classify_exception(asyncio.TimeoutError("late"), "exec", True).reason == "tool_timeout"
    assert classify_score("restart.sh failed: app not ready 150s after restart\ntail").failure_class == AGENT
    assert classify_score("restart.sh failed: weird output").reason == "restart_failed"
    assert classify_score("all fine") is None


class FakeErr:
    def __init__(self, message, traceback=""):
        self.message, self.traceback = message, traceback


class FakeScore:
    def __init__(self, explanation):
        self.explanation = explanation


class FakeSample:
    def __init__(self, error=None, limit=None, scores=None, retries=None, id="s", epoch=1):
        self.error, self.limit, self.scores, self.error_retries, self.id, self.epoch = error, limit, scores, retries, id, epoch


def test_classify_sample_and_log():
    assert classify_sample(FakeSample()) is None
    assert classify_sample(FakeSample(scores={"a": FakeScore("tests failed")})) is None
    assert classify_sample(FakeSample(error=FakeErr("RuntimeError: ready.sh did not succeed within 240s"))).reason == "ready_timeout"
    assert classify_sample(FakeSample(scores={"a": FakeScore("restart.sh failed: app not ready 150s after restart")})).reason == "app_not_ready"

    class Log:
        samples = [FakeSample(retries=[FakeErr("service \"default\" is not running")]), FakeSample(error=FakeErr("Error response from daemon: x"))]
    rows = classify_log(Log)
    assert rows[0]["attempt"] == 2 and rows[0]["retry_reasons"] == ["container_lost"] and rows[0]["failure_class"] is None
    assert rows[1]["failure_class"] == INFRA


def run(coro):
    return asyncio.run(coro)


async def no_sleep(_):
    pass


def test_retry_infra_then_success_ledgers_waste():
    calls = []

    async def fn(attempt):
        calls.append(attempt)
        if attempt < 3:
            raise EpisodeFailure(Failure(INFRA, "container_lost"), cost_usd=0.02)
        return "ok"
    ledger = Ledger()
    seen = []
    ledger.on_wasted = lambda usd, f: seen.append((usd, f.reason))
    out = run(run_with_retries(fn, RetryPolicy(), ledger, HaltRules(consecutive=5), sleep=no_sleep))
    assert out.result == "ok" and out.attempt == 3 and calls == [1, 2, 3] and out.failure is None
    assert ledger.wasted_usd == pytest.approx(0.04) and seen == [(0.02, "container_lost")] * 2


def test_retry_gives_up_after_two_retries_reward_none():
    async def fn(attempt):
        raise RuntimeError("Error response from daemon: boom")
    out = run(run_with_retries(fn, RetryPolicy(), Ledger(), None, sleep=no_sleep))
    assert out.attempt == 3 and out.failure.failure_class == INFRA and not out.reward_valid and out.result is None
    assert out.row()["failure_class"] == INFRA and len(out.failures) == 2


def test_agent_failure_is_not_retried():
    calls = []

    async def fn(attempt):
        calls.append(attempt)
        raise EpisodeFailure(Failure(AGENT, "app_not_ready"))
    out = run(run_with_retries(fn, RetryPolicy(), Ledger(), None, sleep=no_sleep))
    assert calls == [1] and out.failure.failure_class == AGENT and out.reward_valid


def test_halt_class_and_rules_raise():
    async def budget(attempt):
        raise RuntimeError("BudgetExceeded")
    with pytest.raises(Halt):
        run(run_with_retries(budget, RetryPolicy(), Ledger(), None, sleep=no_sleep))

    async def infra(attempt):
        raise RuntimeError("Error response from daemon")
    with pytest.raises(Halt) as e:
        run(run_with_retries(infra, RetryPolicy(max_infra_retries=9), Ledger(), HaltRules(), sleep=no_sleep))
    assert "3_consecutive" in e.value.reason


def test_halt_rules_rate_window_and_docker_down(tmp_path):
    h = HaltRules()
    infra = Failure(INFRA, "x")
    pattern = [infra, None, infra, None, None, None, None, None]
    got = [h.record(f) for f in pattern]
    assert got == [None] * 8  # 2 of 8 is 25 percent, not above
    h2 = HaltRules()
    seq = [infra, None, infra, None, infra, None, None, None]
    assert [h2.record(f) for f in seq][-1] == "infra_rate_3_of_8"
    d = HaltRules()
    assert d.note_docker(False, 0) is None and d.note_docker(False, 599) is None and d.note_docker(False, 601) == "docker_unreachable_over_10min"
    assert d.note_docker(True, 700) is None and d.note_docker(False, 701) is None
    assert HaltRules().record(Failure(HALT, "tinker_auth")) == "halt_class:tinker_auth"
    assert write_halted(tmp_path / "run", "r", "d").read_text().startswith("# HALTED")


def test_disk_gate(monkeypatch):
    import collections
    import shutil
    U = collections.namedtuple("U", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U(1, 1, int(5e9)))
    assert HaltRules.disk("/")[0] == "halt"
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U(1, 1, int(15e9)))
    assert HaltRules.disk("/")[0] == "warn"
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U(1, 1, int(50e9)))
    assert HaltRules.disk("/")[0] == "ok"
