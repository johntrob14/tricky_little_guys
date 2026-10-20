"""rl.py CLI, rl.collect and rl.fake_policy on fakes: no Tinker, no docker. Skips when the gpt-oss tokenizer is not available offline."""

import asyncio
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("test_provider")  # sets HF_HUB_OFFLINE and the offline cache for the tokenizer

ARENA = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def clean_logs():
    yield
    for d in (ARENA / "logs").glob("fake-test-*"):
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_RL_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("ARENA_RL_RUN_TINKER", raising=False)
    spec = importlib.util.spec_from_file_location("arena_rl_cli", ARENA / "rl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dry_run_prints_plan_and_spends_nothing(cli, capsys, tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(["train", "tiktok", "--run", "fake-test-dry", "--steps", "3", "--B", "2", "--G", "4", "--budget-usd", "1", "--ep-worst-usd", "0.02", "--dry-run"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    plan = json.loads(out[: out.rindex("}") + 1])
    assert plan["episodes"] == 24 and plan["would_refuse"] == [] and plan["ep_worst_case_usd"] == 0.02 and len(plan["tasks"]) == 4
    assert not (tmp_path / "data" / "ledger.csv").exists() and not (tmp_path / "data" / "fake-test-dry").exists()


def test_dry_run_refuses_over_budget(cli, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["train", "tiktok", "--run", "fake-test-dry", "--steps", "3", "--B", "2", "--G", "4", "--budget-usd", "1", "--dry-run"])
    assert e.value.code == 2
    assert "worst-case reservation" in capsys.readouterr().out


def test_budget_flag_is_mandatory(cli):
    with pytest.raises(SystemExit) as e:
        cli.main(["train", "tiktok", "--run", "x", "--steps", "1", "--B", "1", "--G", "2"])
    assert e.value.code == 2


def test_live_train_and_collect_refused_without_gate(cli, monkeypatch):
    import tinker
    monkeypatch.setattr(tinker, "ServiceClient", lambda *a, **k: pytest.fail("a live Tinker client was created"))
    with pytest.raises(SystemExit, match="ARENA_RL_RUN_TINKER"):
        cli.main(["train", "tiktok", "--run", "x", "--steps", "1", "--B", "1", "--G", "2", "--budget-usd", "50", "--ep-worst-usd", "0.02"])
    with pytest.raises(SystemExit, match="ARENA_RL_RUN_TINKER"):
        cli.main(["collect", "tiktok", "--run", "x", "--budget-usd", "50", "--ep-worst-usd", "0.02"])


def test_fake_runs_must_be_named_fake(cli):
    with pytest.raises(SystemExit, match="fake-"):
        cli.main(["train", "tiktok", "--run", "real", "--steps", "1", "--B", "1", "--G", "2", "--budget-usd", "1", "--ep-worst-usd", "0.02", "--fake", "--no-sandbox"])


def test_fake_train_end_to_end(cli, tmp_path):
    cli.main(["train", "tiktok", "--run", "fake-test-e2e", "--steps", "3", "--B", "2", "--G", "4", "--budget-usd", "1", "--ep-worst-usd", "0.02", "--fake", "--no-sandbox"])
    out = tmp_path / "data" / "fake-test-e2e"
    lines = [json.loads(x) for x in (out / "steps.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines] == [0, 1, 2] and all(x["n_valid"] == 8 and x["applied"] for x in lines if x["groups_with_signal"])
    assert sum(x["n_datums"] for x in lines) > 0 and all(x["ended"] == {"submit": 8} for x in lines)
    ck = json.loads((out / "checkpoints.json").read_text())
    assert [c["kind"] for c in ck][-1] == "sampler" and ck[-1]["ttl_s"] == 30 * 86400
    ledger = (out / "ledger.csv").read_text().splitlines()
    assert ledger[0].startswith("time,phase") and sum("forecast" in r for r in ledger) == 1 and not (tmp_path / "data" / "ledger.csv").exists()
    side = list((out / "step_000" / "sidecars").glob("*.json.gz"))
    assert len(side) == 8
    from rl.provider import load_sidecar, merge_turns
    doc = load_sidecar(side[0])
    m = merge_turns(doc)
    assert doc["trainable"] and doc["mode"] == "train" and len(m["tokens"]) == len(m["logprobs"]) and [t["tool_call"]["name"] for t in doc["turns"]] == ["bash", "bash", "submit"]


def test_fake_policy_is_deterministic_per_marker():
    from rl.fake_policy import ScriptedPolicy
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    pol = ScriptedPolicy(get_tokenizer("openai/gpt-oss-120b"), default_p=1.0)
    assert "[plan:fix]" in pol._completion("hello") and "sed -i" in pol._completion("x [plan:fix] <|call|>")
    assert "sed -i" not in pol._completion("x [plan:skip] <|call|>") and "submit" in pol._completion("x <|call|> y <|call|>")


def _collect_setup(tmp_path, solver_fn, tasks=("tiktok/missing_user_id_in_guess_response",), n=2, sleeps=None):
    """Collector over a docker-free task whose solver is solver_fn; returns (collector, request, sleeps)."""
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from rl.budget import Budget, Ledger
    from rl.collect import Collector
    from rl.fake_policy import ScriptedPolicy, task_scorer
    from rl.loop import CollectRequest
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    def build(task, max_turns, epochs):
        sample = Sample(id=task.name, input=(task / "task.md").read_text(), metadata=dict(env=task.parent.name, task=task.name))
        return Task(name=f"fake-{task.name}", dataset=[sample], solver=solver_fn(max_turns), scorer=task_scorer(), epochs=epochs)

    sleeps = [] if sleeps is None else sleeps

    async def sleep(s):
        sleeps.append(s)

    policy = ScriptedPolicy(get_tokenizer("openai/gpt-oss-120b"), default_p=1.0)
    budget = Budget(1.0, stage="t", run="fake-test-c", ledger=Ledger(tmp_path / "l.csv"))
    col = Collector("fake-test-c", max_turns=6, build=build, sleep=sleep, phase="test")
    return col, CollectRequest(0, list(tasks), n, policy, "wid", tmp_path / "step", None, budget), sleeps


def _flaky(fail_times, message):
    """Solver that samples one turn and then raises `message` for the first `fail_times` samples, else runs the fake agent."""
    from inspect_ai.solver import solver
    from rl.fake_policy import fake_agent
    count = {"n": 0}

    @solver
    def flaky(max_turns):
        inner = fake_agent(max_turns)

        async def solve(state, generate):
            count["n"] += 1
            if count["n"] <= fail_times:
                from inspect_ai.model import ChatMessageUser, get_model
                from rl.fake_policy import bash
                await get_model().generate([ChatMessageUser(content=state.input_text)], [bash()])
                raise RuntimeError(message)
            return await inner(state, generate)
        return solve
    return flaky


def test_infra_failure_is_rerun_and_wasted_spend_ledgered(tmp_path):
    col, req, sleeps = _collect_setup(tmp_path, _flaky(1, "Cannot connect to the Docker daemon at unix:///var/run/docker.sock"))
    eps = asyncio.run(col(req))
    assert len(eps) == 2 and all(e.failure_class == "none" for e in eps) and sorted(e.attempt for e in eps) == [0, 1] and sleeps == [5]
    assert sum(e.wasted_usd for e in eps) > 0
    assert any(r["kind"] == "wasted" for r in req.budget.ledger.rows())
    assert len({e.episode_id for e in eps}) == 2 and all(e.log_path and e.phase == "test" and e.task == "tiktok/missing_user_id_in_guess_response" for e in eps)


def test_infra_failure_that_persists_gives_reward_none(tmp_path):
    from rl.rewards import attach_reward, compute_rewards
    col, req, sleeps = _collect_setup(tmp_path, _flaky(99, "Conflict. The container name /x is already in use"), n=1)
    eps = asyncio.run(col(req))
    assert len(eps) == 1 and eps[0].failure_class == "infra" and eps[0].failure_reason == "name_conflict" and sleeps == [5, 15]
    eps = [attach_reward(e) for e in eps]
    assert compute_rewards(eps) == {eps[0].episode_id: None}


def test_halt_class_failure_stops_collection(tmp_path):
    from rl.classify import Halt
    col, req, _ = _collect_setup(tmp_path, _flaky(99, "AuthenticationError: invalid api key"), n=1)
    with pytest.raises(Halt):
        asyncio.run(col(req))


def test_agent_failure_is_not_rerun_and_scores_zero(tmp_path):
    from rl.fake_policy import ScriptedPolicy
    col, req, sleeps = _collect_setup(tmp_path, _flaky(0, ""), n=2)
    req.sampler.p_fix["missing_user_id_in_guess_response"] = 0.0
    from rl.rewards import attach_reward, compute_rewards
    eps = [attach_reward(e) for e in asyncio.run(col(req))]
    assert sleeps == [] and sorted(compute_rewards(eps).values()) == [0.0, 0.0] and all(e.failure_class == "agent" for e in eps)


TRAIN = ["train", "tiktok", "--steps", "1", "--B", "1", "--G", "2", "--ep-worst-usd", "0.02"]


def test_cot_monitor_refused_without_allow_live(cli, monkeypatch):
    import rl.cot_monitor as cm
    monkeypatch.setattr(cm, "live_judge", lambda *a, **k: pytest.fail("a live judge was built"))
    with pytest.raises(SystemExit, match="--allow-live"):
        cli.main([*TRAIN, "--run", "fake-test-cot", "--budget-usd", "1", "--fake", "--no-sandbox", "--beta-cot", "0.5"])
    with pytest.raises(SystemExit, match="--allow-live"):
        cli.main([*TRAIN, "--run", "real", "--budget-usd", "1", "--beta-cot", "0.5"])


def test_cot_monitor_refused_without_finite_budget(cli, monkeypatch):
    import rl.cot_monitor as cm
    monkeypatch.setattr(cm, "live_judge", lambda *a, **k: pytest.fail("a live judge was built"))
    for b in ("inf", "nan", "0"):
        with pytest.raises(SystemExit, match="finite --budget-usd"):
            cli.main([*TRAIN, "--run", "fake-test-cot", "--budget-usd", b, "--fake", "--no-sandbox", "--beta-cot", "0.5", "--allow-live", "--ep-worst-usd", "0"])


def test_tinker_judge_needs_tinker_gate(cli, monkeypatch):
    import rl.cot_monitor as cm
    import tinker
    monkeypatch.setattr(cm, "live_judge", lambda *a, **k: pytest.fail("a live judge was built"))
    monkeypatch.setattr(tinker, "ServiceClient", lambda *a, **k: pytest.fail("a live Tinker client was created"))
    with pytest.raises(SystemExit, match="ARENA_RL_RUN_TINKER"):
        cli.main([*TRAIN, "--run", "fake-test-cot", "--budget-usd", "1", "--fake", "--no-sandbox", "--beta-cot", "0.5", "--allow-live", "--judge-backend", "tinker"])


def test_cot_plan_and_default_off(cli, capsys):
    with pytest.raises(SystemExit):
        cli.main([*TRAIN, "--run", "fake-test-dry", "--budget-usd", "1", "--dry-run"])
    assert json.loads(capsys.readouterr().out.split("\ndry run")[0])["cot_monitor"] is None
    with pytest.raises(SystemExit):
        cli.main([*TRAIN, "--run", "fake-test-dry", "--budget-usd", "1", "--dry-run", "--beta-cot", "0.5"])
    p = json.loads(capsys.readouterr().out.split("\ndry run")[0])["cot_monitor"]
    assert p["backend"] == "openrouter" and p["judge_model"] == "openai/gpt-oss-20b" and p["allow_live"] is False


def test_fake_train_with_injected_cot_judge(cli, tmp_path):
    calls = []

    async def judge(prompt):
        calls.append(prompt)
        return "<reasoning>ok</reasoning><score>3</score>", 500, 50

    cli.main([*TRAIN, "--run", "fake-test-cot", "--steps", "2", "--B", "2", "--G", "4", "--budget-usd", "1", "--fake", "--no-sandbox", "--beta-cot", "0.5"], cot_judge=judge)
    out = tmp_path / "data" / "fake-test-cot"
    lines = [json.loads(x) for x in (out / "steps.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and all(l["cot_penalty_on"] for l in lines)
    rows = [json.loads(x) for x in (out / "cot_monitor.jsonl").read_text().splitlines()]
    scored = [r for r in rows if not r["skipped"]]
    assert calls and len(scored) == len(calls) and all(r["c"] == pytest.approx(0.3) and not r["fallback"] for r in scored)
    led = (out / "ledger.csv").read_text()
    assert "cot_monitor" in led
