"""Blocking review findings on the robust package: scorer-side INFRA reward, classify/training-core wiring, canary names."""
import asyncio
import random
import types
from pathlib import Path

import pytest
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import solver

from fakes import make_episode
from rl import guard
from rl.budget import Ledger as BudgetLedger
from rl.classify import AGENT, INFRA, Failure, HaltRules, Ledger, Outcome, budget_hook
from rl.guard import GuardCfg, GuardResult, Reference
from rl.loop import HaltRules as LoopHaltRules
from rl.rewards import attach_reward, episode_reward, read_inspect_score
from rl import schema

ENV = Path(__file__).resolve().parents[2] / "envs" / "tiktok"


class BoomSbx:
    async def exec(self, *a, **k):
        raise RuntimeError("Error response from daemon: No such container: x")


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


def run_scorer(tmp_path, monkeypatch, fake_sbx=None, fake_check=None):
    import inspect_ai.util
    monkeypatch.setattr(guard, "load_reference", lambda env, *a, **k: Reference("x", {}, {}))
    if fake_sbx is not None:
        monkeypatch.setattr(inspect_ai.util, "sandbox", lambda *a, **k: fake_sbx)
    if fake_check is not None:
        monkeypatch.setattr(guard, "guarded_check", fake_check)
    t = Task(dataset=[Sample(input="x", target="y", id="s")], solver=noop(), scorer=guard.guarded_task_scorer(ENV, task_dir(tmp_path), GuardCfg()))
    log = inspect_eval(t, model="mockllm/model", log_dir=str(tmp_path / "logs"), fail_on_error=False, display="none")[0]
    return log, read_inspect_score(log.location, "s", 1)


def test_guard_check_reports_infra_error_for_docker_trouble(tmp_path):
    r = asyncio.run(guard.guarded_check(BoomSbx(), ENV, task_dir(tmp_path), GuardCfg(), Reference("x", {}, {})))
    assert r.status == "infra_error" and r.reason == "scorer_RuntimeError" and r.passed is False


def test_scorer_side_infra_error_is_reward_none_end_to_end(tmp_path, monkeypatch):
    log, r = run_scorer(tmp_path, monkeypatch, fake_sbx=BoomSbx())
    assert log.samples[0].error is not None
    assert r["passed"] is None and r["failure_class"] == INFRA and r["failure_reason"].startswith("scorer_RuntimeError")
    ep = make_episode(random.Random(0), sample_id="s", epoch=1, passed=None, log_path=log.location)
    attach_reward(ep)
    assert ep.failure_class == INFRA and episode_reward(ep) is None


@pytest.mark.parametrize("reason,reward", [("tests_failed", 0.0), ("passed", 1.0)])
def test_non_infra_verdicts_still_score(tmp_path, monkeypatch, reason, reward):
    async def check(*a, **k):
        return GuardResult(passed=reason == "passed", passed_raw=reason == "passed", reason=reason)
    log, r = run_scorer(tmp_path, monkeypatch, fake_sbx=object(), fake_check=check)
    ep = make_episode(random.Random(0), sample_id="s", epoch=1, passed=None, log_path=log.location)
    attach_reward(ep)
    assert episode_reward(ep) == reward


def test_labels_are_shared_and_outcome_row_gives_reward_none():
    assert (AGENT, INFRA) == (schema.AGENT, schema.INFRA)
    row = Outcome(None, Failure(INFRA, "container_lost"), 3).row()
    ep = make_episode(random.Random(0), passed=False, failure_class=row["failure_class"], failure_reason=row["failure_reason"])
    assert episode_reward(ep) is None
    ok = Outcome("r", None, 1).row()
    assert ok["failure_class"] == schema.NONE and episode_reward(make_episode(random.Random(0), failure_class=ok["failure_class"])) == 1.0


def test_agent_side_inspect_error_scores_zero(tmp_path):
    from inspect_ai.log import EvalConfig, EvalDataset, EvalError, EvalLog, EvalSample, EvalSpec, write_eval_log
    spec = EvalSpec(created="2026-10-20T00:00:00+00:00", task="t", dataset=EvalDataset(), model="m", config=EvalConfig())
    err = lambda m: EvalError(message=m, traceback="", traceback_ansi="")
    samples = [EvalSample(id="a", epoch=1, input="x", target="", error=err("RuntimeError: restart.sh failed: app not ready 150s after restart")),
               EvalSample(id="b", epoch=1, input="x", target="", error=err("RuntimeError: ready.sh did not succeed"))]
    p = tmp_path / "l.eval"
    write_eval_log(EvalLog(eval=spec, samples=samples, status="success"), str(p))
    a, b = read_inspect_score(str(p), "a", 1), read_inspect_score(str(p), "b", 1)
    assert a["passed"] is False and a["failure_class"] == AGENT and b["passed"] is None and b["failure_class"] == INFRA


def test_loop_uses_classify_halt_rules_and_wasted_cost_reaches_budget_ledger(tmp_path):
    h = LoopHaltRules()
    assert isinstance(h._infra, HaltRules)
    h.observe([make_episode(random.Random(i), passed=None, failure_class="infra", ended="infra") for i in range(3)])
    assert h._infra.run == 3
    bl = BudgetLedger(tmp_path / "ledger.csv")
    led = Ledger(on_wasted=budget_hook(bl, "stage", "run", "openai/gpt-oss-120b"))
    led.add(1, Failure(INFRA, "container_lost"), 0.02)
    led.add(2, Failure(AGENT, "tool_timeout"), 0.05)
    rows = bl.rows()
    assert len(rows) == 1 and rows[0]["kind"] == "wasted" and float(rows[0]["estimate_usd"]) == 0.02 and bl.total_counted() == 0.0


def test_canaries_use_random_dirs_and_the_real_basename(tmp_path):
    writes, env = [], tmp_path / "env"
    env.mkdir()
    (env / "restart.sh").write_text("true\n")

    class Sbx:
        async def exec(self, cmd, **k):
            return types.SimpleNamespace(success=True, returncode=0, stdout="", stderr="")

        async def write_file(self, path, text):
            writes.append(path)

        async def read_file(self, path, text=True):
            import io, tarfile
            b = io.BytesIO()
            tarfile.open(fileobj=b, mode="w").close()
            return b.getvalue()

    r = asyncio.run(guard.guarded_check(Sbx(), env, task_dir(tmp_path), GuardCfg(reset_db=False), Reference("x", {}, {})))
    assert r.reason == "runner_rigged"   # all-zero fake rc: the bad canary "passed"
    scored = [w for w in writes if "/la-score-" in w]
    assert len(scored) == 3 and {Path(w).name for w in scored} == {"t.py"} and len({Path(w).parent for w in scored}) == 3
    assert not any("canary" in w for w in writes)


def test_reference_cache_is_keyed_by_manifest_script(tmp_path, monkeypatch):
    calls = []

    def run(cmd, **k):
        calls.append(cmd)
        return types.SimpleNamespace(stdout="sha256:" + "ab" * 32 if cmd[1] == "image" else (b"\0" * 1024 if k.get("text") is None else ""), returncode=0)

    monkeypatch.setattr(guard.subprocess, "run", run)
    guard.load_reference.cache_clear()
    guard.load_reference("tiktok", cache=tmp_path)
    (d,) = list(tmp_path.iterdir())
    assert d.name.startswith("abababababababab-") and len(d.name) == 16 + 1 + 8
