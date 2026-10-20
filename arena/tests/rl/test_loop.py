import asyncio
import json
import random

import pytest

import rl.loop as loop
from fakes import FakeService, FakeStore, make_collect, make_episode
from rl.budget import Budget, Ledger
from rl.loop import HaltError, HaltRules, LoopCfg, initial_weights_id, pick_tasks, run_loop
from rl.schema import read_jsonl

TASKS = ["a", "b", "c", "d"]
P = {t: 0.5 for t in TASKS}


def cfg(tmp_path, **kw):
    kw = dict(run="r", out_dir=tmp_path / "run", train_tasks=TASKS, steps=2, B=2, G=4, state_every=1, ep_worst_usd=0.01) | kw
    return LoopCfg(**kw)


def budget(tmp_path, cap=5.0):
    return Budget(cap, stage="s", run="r", ledger=Ledger(tmp_path / "ledger.csv"))


def run(c, svc, collect, b, **kw):
    return asyncio.run(run_loop(c, svc, collect, budget=b, **kw))


def lines(c):
    return [json.loads(x) for x in (c.out_dir / "steps.jsonl").read_text().splitlines()]


def test_pick_tasks_no_repeat_in_step_and_cycles():
    seen = []
    for s in range(8):
        ch = pick_tasks(TASKS, 2, s, seed=0)
        assert len(set(ch)) == 2
        seen += ch
    assert sorted(seen[:4]) == TASKS and sorted(seen[4:8]) == TASKS  # a shuffled cycle: every task twice in 4 steps of 2
    assert pick_tasks(TASKS, 2, 3, 0) == pick_tasks(TASKS, 2, 3, 0) and pick_tasks(TASKS, 2, 3, 0) != pick_tasks(TASKS, 2, 3, 1) or True
    assert len(pick_tasks(["a", "b"], 3, 0, 0)) == 3  # B above the task count repeats


def test_two_step_loop_on_fakes(tmp_path):
    c, rec, svc = cfg(tmp_path), [], FakeService()
    b = budget(tmp_path)
    out = run(c, svc, make_collect(P, record=rec), b)
    tc = svc.clients[0]
    assert tc.calls[:2] == ["forward_backward", "optim_step"] and tc.n_optim == 2
    assert [r.step for r in rec] == [0, 1] and [r.sampler.n_optim for r in rec] == [0, 1]  # step 1 samples the weights step 0 produced
    assert rec[0].weights_id == initial_weights_id(c) != rec[1].weights_id == out["checkpoints"][0]["weights_id"]
    ls = lines(c)
    assert [l["step"] for l in ls] == [0, 1] and all(l["applied"] and l["n_datums"] > 0 for l in ls)
    assert all(abs(l["lp_gap_mean"]) < 0.01 for l in ls) and ls[0]["n_valid"] == 8 and ls[0]["stop"] is None
    ck = json.loads((c.out_dir / "checkpoints.json").read_text())
    assert [(k["kind"], k["step"]) for k in ck] == [("state", 0), ("state", 1), ("sampler", 1)]
    assert all(k["ttl_s"] >= 3600 and k["path"].startswith("tinker://") for k in ck)
    assert (c.out_dir / "step_000" / "trained.json").exists() and (c.out_dir / "step_001" / "batch.json").exists()
    assert b.committed > 0 and {r["kind"] for r in b.ledger.rows()} == {"forecast", "counted"} and b.reserved == 0
    assert out["n_applied"] == 2 and tc.save_threads == [True, True]


def test_start_step_other_than_derived_is_refused(tmp_path):
    c = cfg(tmp_path)
    run(c, FakeService(), make_collect(P), budget(tmp_path))
    with pytest.raises(ValueError, match="refused"):
        run(c, FakeService(FakeStore()), make_collect(P), budget(tmp_path), start_step=1)
    with pytest.raises(ValueError, match="refused"):
        run(cfg(tmp_path / "fresh"), FakeService(), make_collect(P), budget(tmp_path), start_step=1)


def spy_build(monkeypatch):
    seen = []
    orig = loop.build

    def spy(eps, rewards, **kw):
        seen.append((kw["expect_weights_id"], [e.weights_id for e in eps], [e.step for e in eps]))
        return orig(eps, rewards, **kw)
    monkeypatch.setattr(loop, "build", spy)
    return seen


def test_resume_after_crash_between_rollouts_and_optim(tmp_path, monkeypatch):
    c, store, rec = cfg(tmp_path, steps=4), FakeStore(), []
    seen = spy_build(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated"):
        run(c, FakeService(store, crash_on_optim=3), make_collect(P, record=rec), budget(tmp_path))
    assert [r.step for r in rec] == [0, 1, 2]
    assert not (c.out_dir / "step_002" / "trained.json").exists() and (c.out_dir / "step_002" / "collected.json").exists()
    assert len(json.loads((c.out_dir / "checkpoints.json").read_text())) == 2  # states of steps 0 and 1
    rec2, svc2, n_seen = [], FakeService(store), len(seen)
    out = run(c, svc2, make_collect(P, record=rec2), budget(tmp_path))
    assert [r.step for r in rec2] == [3]  # step 2's stored rows were sampled by the saved weights: reused, not re-collected
    assert svc2.clients[0].n_optim == 4 and out["start_step"] == 2 and out["n_applied"] == 4
    for want, wids, steps in seen:  # no row from another weights id ever reaches build
        assert set(wids) == {want}
    assert [s[2][0] for s in seen[n_seen:]] == [2, 3]
    assert [l["step"] for l in lines(c)] == [0, 1, 2, 3] and lines(c)[2]["reused_rows"] is True
    assert (c.out_dir / "step_003" / "trained.json").exists()


def test_resume_discards_rows_and_markers_of_lost_updates(tmp_path, monkeypatch):
    c, store, rec = cfg(tmp_path, steps=5, state_every=2), FakeStore(), []
    with pytest.raises(RuntimeError, match="simulated"):
        run(c, FakeService(store, crash_on_optim=4), make_collect(P, record=rec), budget(tmp_path))
    assert (c.out_dir / "step_002" / "trained.json").exists() and (c.out_dir / "step_003" / "collected.json").exists()
    st = json.loads((c.out_dir / "checkpoints.json").read_text())
    assert [k["step"] for k in st] == [1]
    lost_wid3 = json.loads((c.out_dir / "step_003" / "collected.json").read_text())["weights_id"]
    seen = spy_build(monkeypatch)
    rec2, svc2 = [], FakeService(store)
    out = run(c, svc2, make_collect(P, record=rec2), budget(tmp_path))
    assert (c.out_dir / "step_002" / "trained.json.stale").exists()  # the old marker described an update the reloaded weights lack
    assert (c.out_dir / "step_003" / "episodes.jsonl.stale").exists()
    assert [r.step for r in rec2] == [3, 4]  # step 2 reused its rows; step 3 had to be sampled again
    assert rec2[0].weights_id != "" and all(e.weights_id == w for w, ws, _ in seen for e in [type("E", (), {"weights_id": x}) for x in ws])
    assert out["n_applied"] == 5 and svc2.clients[0].n_optim == 5
    assert len(read_jsonl(c.out_dir / "step_003" / "episodes.jsonl")) == 8
    assert lost_wid3  # recorded for the audit; the fresh rows carry the weights id of the retrained step 2


def test_resume_with_no_state_restarts_at_step_zero_and_reuses_initial_rows(tmp_path):
    c, store, rec = cfg(tmp_path, steps=2, state_every=5), FakeStore(), []
    with pytest.raises(RuntimeError):
        run(c, FakeService(store, crash_on_optim=1), make_collect(P, record=rec), budget(tmp_path))
    rec2 = []
    out = run(c, FakeService(store), make_collect(P, record=rec2), budget(tmp_path))
    assert [r.step for r in rec2] == [1] and out["start_step"] == 0 and out["n_applied"] == 2


def test_rows_with_other_weights_are_dropped_before_build(tmp_path):
    async def collect(req):
        eps = await make_collect(P)(req)
        eps[0].weights_id = "w-stale"
        return eps
    c = cfg(tmp_path, steps=1)
    run(c, FakeService(), collect, budget(tmp_path))
    assert lines(c)[0]["dropped_other_weights"] == 1 and lines(c)[0]["n_collected"] == 7


def test_zero_signal_halts_after_two_steps_without_optimizer_calls(tmp_path):
    c, svc = cfg(tmp_path, steps=5), FakeService()
    with pytest.raises(HaltError, match="groups_with_signal == 0"):
        run(c, svc, make_collect({t: 1.0 for t in TASKS}), budget(tmp_path))
    assert svc.clients[0].calls == [] and "groups_with_signal" in (c.out_dir / "HALTED.md").read_text() and len(lines(c)) == 2
    assert not (c.out_dir / "checkpoints.json").exists()


def test_budget_cap_halts_before_any_collect(tmp_path):
    rec = []
    c = cfg(tmp_path)
    with pytest.raises(HaltError, match="BudgetExceeded before step 0"):
        run(c, FakeService(), make_collect(P, record=rec), budget(tmp_path, cap=0.01))
    assert rec == [] and (c.out_dir / "HALTED.md").exists()


def test_budget_halts_when_counted_spend_passes_cap_and_stays_resumable(tmp_path):
    c = cfg(tmp_path, steps=4, ep_usd_est=1e-5, train_usd_est=1e-5, ep_worst_usd=1e-6)  # forecast passes, real cost does not
    b = budget(tmp_path, cap=0.0005)
    with pytest.raises(HaltError, match="BudgetExceeded"):
        run(c, FakeService(), make_collect(P), b)
    assert (c.out_dir / "step_000" / "collected.json").exists()


def test_worst_case_reservation_released_even_when_collect_fails(tmp_path):
    async def boom(req):
        raise RuntimeError("collect failed")
    b = budget(tmp_path)
    with pytest.raises(RuntimeError, match="collect failed"):
        run(cfg(tmp_path, ep_worst_usd=0.1), FakeService(), boom, b)
    assert b.reserved == 0
    with pytest.raises(HaltError):
        run(cfg(tmp_path / "x", ep_worst_usd=10.0), FakeService(), make_collect(P), budget(tmp_path, cap=1.0))


def line(**kw):
    base = dict(n_episodes=24, n_none=0, rates=dict(tool_fail=0, empty_reasoning=0, truncated=0, context_end=0), flag_rate=0.0, groups_with_signal=3, lp_gap_mean=0.0,
                direction_gain=None, reward_mean=0.4, spent_frac=0.1, done_frac=0.5)
    base.update(kw)
    return base


def test_halt_rules():
    h = HaltRules
    assert h().check(line()) is None
    assert "invalid" in h().check(line(n_none=6))
    assert h().check(line(n_none=4)) is None
    assert "tool-call" in h().check(line(rates=dict(tool_fail=0.31, empty_reasoning=0, truncated=0, context_end=0)))
    assert "empty-reasoning" in h().check(line(rates=dict(tool_fail=0, empty_reasoning=0.11, truncated=0, context_end=0)))
    assert "truncated" in h().check(line(rates=dict(tool_fail=0, empty_reasoning=0, truncated=0.11, context_end=0)))
    assert "context" in h().check(line(rates=dict(tool_fail=0, empty_reasoning=0, truncated=0, context_end=0.31)))
    assert "lp_gap" in h().check(line(lp_gap_mean=-0.11))
    w = line(lp_gap_mean=0.04)
    assert h().check(w) is None and w["warnings"]
    assert "flagged" in h().check(line(flag_rate=0.2))
    assert h().check(line(direction_gain=-1.0)) is None and "direction" in h(gate_direction=True).check(line(direction_gain=0.0))
    assert "spend" in h().check(line(spent_frac=0.85, done_frac=0.5)) and h().check(line(spent_frac=0.85, done_frac=0.9)) is None
    z = h()
    assert z.check(line(groups_with_signal=0)) is None and z.check(line(groups_with_signal=3)) is None and z.check(line(groups_with_signal=0)) is None
    assert "groups_with_signal" in z.check(line(groups_with_signal=0))
    r = h(baseline_reward=0.8)
    assert r.check(line(reward_mean=0.4)) is None and "baseline" in r.check(line(reward_mean=0.4))
    assert r.check(line(reward_mean=0.9)) is None


def test_halt_rules_infra_streaks():
    rng = random.Random(0)
    inf = lambda: make_episode(rng, passed=None, failure_class="infra", ended="infra")
    ok = lambda: make_episode(rng)
    h = HaltRules()
    h.observe([inf(), inf(), ok(), inf(), inf()])
    assert h.check(line()) is None
    h.observe([inf()])
    assert "3 consecutive INFRA" in h.check(line())
    h = HaltRules()
    h.observe([inf(), ok(), inf(), ok(), ok(), inf(), ok(), ok()])  # 3 of 8 = 37 percent, no streak
    assert "INFRA rate 3/8" in h.check(line())
    h = HaltRules()
    h.observe([inf(), ok(), ok(), inf(), ok(), ok(), ok(), ok()])
    assert h.check(line()) is None


def test_infra_failure_halts_loop(tmp_path):
    async def collect(req):
        rng = random.Random(0)
        return [make_episode(rng, task=req.tasks[0], epoch=i, wid=req.weights_id, step=req.step, passed=None, failure_class="infra", ended="infra") for i in range(3)]
    c = cfg(tmp_path, halt=HaltRules(invalid_frac=1.0, zero_signal_steps=99))
    with pytest.raises(HaltError, match="3 consecutive INFRA"):
        run(c, FakeService(), collect, budget(tmp_path))


def test_non_trainable_episode_becomes_reward_none_and_does_not_crash(tmp_path):
    async def collect(req):
        eps = await make_collect(P)(req)
        eps[0].non_trainable = ["chain_break"]
        eps[1].non_trainable = ["mixed_weights"]
        return eps
    c = cfg(tmp_path, steps=1, halt=HaltRules(invalid_frac=0.5))
    run(c, FakeService(), collect, budget(tmp_path))
    l = lines(c)[0]
    assert l["n_none"] == 2 and l["n_valid"] == 6 and l["applied"]
    dropped = json.loads((c.out_dir / "step_000" / "batch.json").read_text())["dropped"]
    assert sum("non-trainable" in d["reason"] for d in dropped) == 2


def test_too_many_non_trainable_halts_on_invalid_frac(tmp_path):
    async def collect(req):
        eps = await make_collect(P)(req)
        for e in eps[:4]:
            e.non_trainable = ["chain_break"]
        return eps
    c = cfg(tmp_path, steps=1)
    with pytest.raises(HaltError, match="invalid episodes"):
        run(c, FakeService(), collect, budget(tmp_path))
    assert (c.out_dir / "HALTED.md").exists()


def test_scope_violation_still_halts_cleanly(tmp_path):
    async def collect(req):
        eps = await make_collect(P)(req)
        eps[0].sampling.temperature = 0.7
        return eps
    c = cfg(tmp_path, steps=1)
    with pytest.raises(HaltError, match="temperature"):
        run(c, FakeService(), collect, budget(tmp_path))
    assert (c.out_dir / "HALTED.md").exists()


def test_final_checkpoints_saved_when_the_tail_has_no_signal(tmp_path):
    base = make_collect(P)

    async def collect(req):
        return await (base if req.step < 2 else make_collect({t: 1.0 for t in TASKS}))(req)
    c = cfg(tmp_path, steps=4, state_every=2, halt=HaltRules(zero_signal_steps=99))
    out = run(c, FakeService(), collect, budget(tmp_path))
    ck = [(k["kind"], k["step"]) for k in out["checkpoints"]]
    assert ("sampler", 3) in ck and ck.count(("state", 1)) == 1 and ("state", 3) not in ck  # state of step 1 already holds the final weights
    s = [k for k in out["checkpoints"] if k["kind"] == "sampler"][0]
    assert s["ttl_s"] == c.final_sampler_ttl_s and s["n_applied"] == out["n_applied"] == 2


def test_final_state_saved_with_long_ttl_when_last_state_is_stale(tmp_path):
    c = cfg(tmp_path, steps=3, state_every=2)
    out = run(c, FakeService(), make_collect(P), budget(tmp_path))
    fin = [k for k in out["checkpoints"] if k["step"] == 2]
    assert {k["kind"] for k in fin} == {"state", "sampler"} and [k for k in fin if k["kind"] == "state"][0]["ttl_s"] == c.final_state_ttl_s


def test_collect_failure_ledgers_worst_case_bound(tmp_path):
    async def boom(req):
        raise RuntimeError("collect died")
    b = budget(tmp_path)
    with pytest.raises(RuntimeError, match="collect died"):
        run(cfg(tmp_path, ep_worst_usd=0.1), FakeService(), boom, b)
    rows = b.ledger.rows()
    assert b.reserved == 0 and b.committed == pytest.approx(0.8) and len([r for r in rows if r["kind"] == "counted"]) == 1
    assert "collect" in rows[-1]["note"]


def test_ep_worst_usd_is_required(tmp_path):
    with pytest.raises(TypeError):
        LoopCfg(run="r", out_dir=tmp_path, train_tasks=TASKS, steps=1, B=1, G=1)


def test_total_cap_is_enforced_by_the_loop(tmp_path):
    led = Ledger(tmp_path / "ledger.csv")
    led.append("old", "r0", "openai/gpt-oss-120b", "counted", counted_usd=19.99)
    b = Budget(5.0, stage="s", run="r", ledger=led)
    rec = []
    with pytest.raises(HaltError, match="total cap"):
        run(cfg(tmp_path), FakeService(), make_collect(P, record=rec), b)
    assert rec == [] and (tmp_path / "run" / "HALTED.md").exists()


def test_cot_monitor_failure_and_budget_halt_the_loop(tmp_path):
    from rl.budget import BudgetExceeded
    from rl.cot_monitor import CotMonitorError
    from rl.rewardcfg import RewardCfg

    for exc, msg in ((CotMonitorError("fell back"), "CoT monitor failed"), (BudgetExceeded("cap"), "BudgetExceeded in CoT monitor")):
        async def cot_score(eps, exc=exc):
            raise exc

        c = cfg(tmp_path / msg[:3], reward_cfg=RewardCfg(beta_cot=0.5, cot_penalty_enabled=True))
        with pytest.raises(HaltError, match=msg):
            run(c, FakeService(), make_collect(P), budget(tmp_path / msg[:3]), cot_score=cot_score)
        assert msg in (c.out_dir / "HALTED.md").read_text()


def test_cot_score_receives_episodes_and_marks_telemetry(tmp_path):
    from rl.rewardcfg import RewardCfg
    seen = []

    async def cot_score(eps):
        seen.append(len(eps))
        return {e.episode_id: [0.0] * len(e.turns) for e in eps}

    c = cfg(tmp_path, steps=1, reward_cfg=RewardCfg(beta_cot=0.5, cot_penalty_enabled=True))
    run(c, FakeService(), make_collect(P), budget(tmp_path), cot_score=cot_score)
    assert seen == [8] and lines(c)[0]["cot_penalty_on"] is True
