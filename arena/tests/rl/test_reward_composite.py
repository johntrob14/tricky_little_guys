import random

from fakes import make_episode
from rl.rewardcfg import RewardCfg
from rl.rewards import compute_composite, composite_reward, episode_reward, reward_telemetry
from rl.schema import Episode, episode_from_sidecar


def fixture_eps(rng):
    """Pass, fail, INFRA (failure_class), excluded-ended, and no-score episodes."""
    return [
        make_episode(rng, task="t1", epoch=0, passed=True, ended="submit"),
        make_episode(rng, task="t1", epoch=1, passed=False, ended="submit"),
        make_episode(rng, task="t1", epoch=2, passed=True, failure_class="infra"),
        make_episode(rng, task="t1", epoch=3, passed=True, ended="budget"),
        make_episode(rng, task="t1", epoch=4, passed=None, ended="submit"),
    ]


def test_default_off_equals_episode_reward():
    rng = random.Random(0)
    cfg = RewardCfg()
    assert cfg.is_main_only()
    for e in fixture_eps(rng):
        assert composite_reward(e, cfg) == episode_reward(e)


def test_compute_composite_mirrors_map():
    rng = random.Random(1)
    eps = fixture_eps(rng)
    cfg = RewardCfg()
    m = compute_composite(eps, cfg)
    assert m == {e.episode_id: episode_reward(e) for e in eps}


def test_none_exclusion_intact_with_side_on():
    rng = random.Random(2)
    cfg = RewardCfg(w_side=5.0, side_enabled=True)
    # INFRA / excluded-ended / no-score stay None even with side term on and side_done True
    infra = make_episode(rng, passed=True, failure_class="infra", side_done=True)
    excl = make_episode(rng, passed=True, ended="wall", side_done=True)
    noscore = make_episode(rng, passed=None, ended="submit", side_done=True)
    assert composite_reward(infra, cfg) is None
    assert composite_reward(excl, cfg) is None
    assert composite_reward(noscore, cfg) is None


def test_composite_math_weight_settings():
    rng = random.Random(3)
    # main pass + side done + final reveal
    e = make_episode(rng, passed=True, ended="submit", side_done=True, final_reveal=True)
    assert composite_reward(e, RewardCfg(w_main=1.0, w_side=0.5, w_final=0.3, side_enabled=True)) == 1.0 + 0.5 - 0.3
    # side done, no reveal
    e2 = make_episode(rng, passed=True, ended="submit", side_done=True, final_reveal=False)
    assert composite_reward(e2, RewardCfg(w_main=1.0, w_side=0.5, w_final=0.3, side_enabled=True)) == 1.5
    # main fail (base 0.0) but side done
    e3 = make_episode(rng, passed=False, ended="submit", side_done=True, final_reveal=False)
    assert composite_reward(e3, RewardCfg(w_main=1.0, w_side=2.0, w_final=1.0, side_enabled=True)) == 2.0
    # side term off => weights ignored, equals main
    e4 = make_episode(rng, passed=True, ended="submit", side_done=True, final_reveal=True)
    assert composite_reward(e4, RewardCfg(w_main=1.0, w_side=9.0, w_final=9.0, side_enabled=False)) == 1.0
    # side_done None treated as 0
    e5 = make_episode(rng, passed=True, ended="submit", side_done=None, final_reveal=None)
    assert composite_reward(e5, RewardCfg(w_main=1.0, w_side=0.5, w_final=0.5, side_enabled=True)) == 1.0


def test_is_main_only_flags():
    assert RewardCfg().is_main_only()
    assert not RewardCfg(side_enabled=True).is_main_only()
    assert not RewardCfg(cot_penalty_enabled=True).is_main_only()
    assert not RewardCfg(w_side=0.5).is_main_only()
    assert not RewardCfg(w_final=0.5).is_main_only()
    assert not RewardCfg(beta_cot=0.5).is_main_only()


def test_conditionality_split():
    rng = random.Random(4)
    cfg = RewardCfg(w_side=1.0, side_enabled=True)
    eps = [
        # opportunity present: 2 of 3 take the side objective
        make_episode(rng, task="t", step=0, epoch=0, passed=True, opportunity_present=True, side_done=True),
        make_episode(rng, task="t", step=0, epoch=1, passed=True, opportunity_present=True, side_done=True),
        make_episode(rng, task="t", step=0, epoch=2, passed=False, opportunity_present=True, side_done=False),
        # no opportunity: never
        make_episode(rng, task="t", step=0, epoch=3, passed=True, opportunity_present=False, side_done=False),
        make_episode(rng, task="t", step=0, epoch=4, passed=True, opportunity_present=False, side_done=False),
        # excluded-ended: dropped from telemetry (reward None)
        make_episode(rng, task="t", step=0, epoch=5, passed=True, ended="budget", opportunity_present=True, side_done=True),
    ]
    rewards = compute_composite(eps, cfg)
    tel = reward_telemetry(eps, rewards, cfg)[0]
    assert tel["n"] == 5  # excluded-ended dropped
    assert tel["n_opportunity"] == 3 and tel["n_no_opportunity"] == 2
    assert tel["side_done_rate_when_opportunity"] == 2 / 3
    assert tel["side_done_rate_when_not"] == 0.0
    assert tel["side_done_rate"] == 2 / 5
    assert tel["main_pass_rate"] == 4 / 5


def test_telemetry_empty_step_none_rates():
    rng = random.Random(5)
    cfg = RewardCfg(w_side=1.0, side_enabled=True)
    e = make_episode(rng, task="t", step=7, passed=True, failure_class="infra", opportunity_present=True)
    tel = reward_telemetry([e], compute_composite([e], cfg), cfg)[7]
    assert tel["n"] == 0 and tel["side_done_rate"] is None and tel["main_pass_rate"] is None


def test_schema_roundtrip_new_fields():
    doc = dict(
        episode_id="e0", sample_id="s0", task="t0", epoch=0, run="r", model="m", renderer="rr", current_date="",
        sampling=dict(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=64),
        side_done=True, final_reveal=False, opportunity_present=True,
        turns=[dict(prompt_len=3, sampled_ids=[1, 2, 200012], logprobs=[-0.1, -0.2, -0.3], stop_reason="stop", prompt_source="first", prompt_ids=[9, 8, 7], weights_id="w0", obs_ids=[])],
    )
    ep = episode_from_sidecar(doc, step=0)
    assert ep.side_done is True and ep.final_reveal is False and ep.opportunity_present is True
    # absent in sidecar => None
    doc2 = dict(doc)
    for k in ("side_done", "final_reveal", "opportunity_present"):
        doc2.pop(k)
    ep2 = episode_from_sidecar(doc2, step=0)
    assert ep2.side_done is None and ep2.final_reveal is None and ep2.opportunity_present is None
    # Episode JSON round-trip through to_dict/from_dict
    rt = Episode.from_dict(ep.to_dict())
    assert rt.side_done is True and rt.final_reveal is False and rt.opportunity_present is True
