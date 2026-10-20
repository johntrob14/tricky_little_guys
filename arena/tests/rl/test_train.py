import asyncio
import random

import pytest

from fakes import FakeService, FakeStore, FakeTrainingClient, make_episode
from rl.budget import Budget, BudgetExceeded, Ledger, train_cost_usd
from rl.datums import build
from rl.rewards import compute_rewards
from rl.train import TrainHP, adam_params, check_ttl, direction_gain, next_sampler, open_training_client, save_checkpoint, train_step


def batch_of(seed=0, tasks=("a", "b"), n=4):
    rng = random.Random(seed)
    eps = [make_episode(rng, task=t, epoch=i, passed=i % 2 == 0) for t in tasks for i in range(n)]
    return build(eps, compute_rewards(eps))


def run(coro):
    return asyncio.run(coro)


def test_adam_defaults_match_spec():
    a = adam_params(TrainHP())
    assert (a.learning_rate, a.beta1, a.beta2, a.eps, a.weight_decay, a.grad_clip_norm) == (4e-5, 0.9, 0.95, 1e-8, 0.0, 0.0)


def test_train_step_calls_and_lp_gap(tmp_path):
    tc = FakeTrainingClient(FakeStore(), gap=0.002)
    b = batch_of()
    bud = Budget(1.0, ledger=Ledger(tmp_path / "l.csv"))
    marks = []
    m = run(train_step(tc, b, TrainHP(), budget=bud, on_applied=lambda: marks.append(tc.calls[:])))
    assert tc.calls == ["forward_backward", "optim_step"] and marks == [["forward_backward", "optim_step"]]
    assert m.applied and tc.n_optim == 1 and tc.optim_params[0].learning_rate == 4e-5
    assert m.lp_gap_mean == pytest.approx(0.002, abs=1e-5) and m.lp_gap_abs_p95 == pytest.approx(0.002, abs=1e-4) and m.lp_gap_frac_gt_0_7 == 0
    assert m.n_positions == sum(d.n_trained_tokens for d in b.datums)
    assert m.loss_sum is not None and "e_max_violation:max" in m.router
    assert bud.committed == pytest.approx(train_cost_usd("openai/gpt-oss-120b", b.tokens_total)) and bud.reserved == 0


def test_zero_datums_makes_no_call():
    tc = FakeTrainingClient(FakeStore())
    rng = random.Random(0)
    eps = [make_episode(rng, task="a", epoch=i, passed=True) for i in range(4)]  # constant group
    b = build(eps, compute_rewards(eps))
    m = run(train_step(tc, b, TrainHP()))
    assert not m.applied and tc.calls == [] and tc.n_optim == 0


def test_misaligned_trainer_logprobs_raise():
    class Bad(FakeTrainingClient):
        def _outs(self, data):
            return [{"logprobs": type("T", (), {"tolist": lambda s: [0.0]})()} for _ in data]
    with pytest.raises(ValueError, match="trainer returned"):
        run(train_step(Bad(FakeStore()), batch_of(), TrainHP()))


def test_direction_check_positive_and_null():
    tc = FakeTrainingClient(FakeStore(), gain=0.01)
    m = run(train_step(tc, batch_of(n=6), TrainHP(), check_direction=True))
    assert tc.calls == ["forward", "forward_backward", "optim_step", "forward"]
    assert m.direction_gain == pytest.approx(0.01, abs=1e-4) and m.direction_ok is True and abs(m.direction_null_mean) < 0.01
    tc = FakeTrainingClient(FakeStore(), gain=-0.01)  # the update moves the wrong way
    m = run(train_step(tc, batch_of(n=6), TrainHP(), check_direction=True))
    assert m.direction_gain == pytest.approx(-0.01, abs=1e-4) and m.direction_ok is False


def test_direction_gain_formula():
    b = batch_of(n=4)
    before = [[-1.0] * len(d.fields["logprobs"]) for d in b.datums]
    after = [[-1.0 + 0.1 * (1 if d.advantage > 0 else -1) for _ in d.fields["logprobs"]] for d in b.datums]
    assert direction_gain(b, before, after)["gain"] == pytest.approx(0.1, abs=1e-5)


def test_budget_blocks_training_before_any_call(tmp_path):
    tc = FakeTrainingClient(FakeStore())
    b = batch_of()
    with pytest.raises(BudgetExceeded):
        run(train_step(tc, b, TrainHP(), budget=Budget(1e-9, ledger=Ledger(tmp_path / "l.csv"))))
    assert tc.calls == []


def test_failed_step_still_commits_and_releases(tmp_path):
    tc = FakeTrainingClient(FakeStore(), crash_on_optim=1)
    bud = Budget(1.0, ledger=Ledger(tmp_path / "l.csv"))
    with pytest.raises(RuntimeError, match="simulated"):
        run(train_step(tc, batch_of(), TrainHP(), budget=bud))
    assert bud.reserved == 0


def test_sampler_is_fetched_off_the_event_loop_thread():
    tc = FakeTrainingClient(FakeStore())
    s = run(next_sampler(tc))
    assert tc.save_threads == [True] and s.n_optim == 0


def test_ttl_must_be_finite_and_checkpoint_names_unique():
    for bad in (None, 10, 11 * 365 * 86400):
        with pytest.raises(ValueError):
            check_ttl(bad)
    tc = FakeTrainingClient(FakeStore())
    row = run(save_checkpoint(tc, "state", "n1", 86400, step=3))
    assert row["path"].startswith("tinker://") and row["ttl_s"] == 86400 and row["step"] == 3
    with pytest.raises(AssertionError, match="twice"):
        run(save_checkpoint(tc, "state", "n1", 86400))


def test_open_client_fresh_and_from_state():
    svc = FakeService()
    tc = run(open_training_client(svc, TrainHP(), "m"))
    run(train_step(tc, batch_of(), TrainHP()))
    row = run(save_checkpoint(tc, "state", "s", 3600))
    tc2 = run(open_training_client(svc, TrainHP(), "m", row["path"]))
    assert tc2.n_optim == 1 and tc2 is not tc
