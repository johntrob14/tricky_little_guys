"""Branch-mode advantage grouping: a trunk and its branches form one contrastive group, so sibling branches of
a common prefix are scored against each other (the localized covert/overt signal), not against other trunks."""

import random

from fakes import make_episode

from rl.datums import build
from rl.rewards import branch_group_key, compute_rewards, default_group_key, group_advantages


def _trunk_and_branches(rng, trunk_task, step, trunk_tag, passes):
    """A trunk (branch_of='') plus len(passes)-1 branches pointing at it, all same (step, task), distinct ids."""
    eps = []
    tid = f"trunk-{trunk_tag}"
    for i, p in enumerate(passes):
        e = make_episode(rng, task=trunk_task, step=step, epoch=i, passed=p)
        e.episode_id = tid if i == 0 else f"{tid}-b{i}"
        e.branch_of = "" if i == 0 else tid
        e.cut_turn = None if i == 0 else 1
        eps.append(e)
    return eps


def test_two_trunks_one_task_split_into_two_groups():
    rng = random.Random(0)
    # trunk A: pass, fail, fail  (mean 1/3)  ;  trunk B: pass, pass, fail (mean 2/3) -- both task t0, step 0
    eps = _trunk_and_branches(rng, "t0", 0, "A", [True, False, False]) + _trunk_and_branches(rng, "t0", 0, "B", [True, True, False])
    rewards = compute_rewards(eps)

    advs_flat, stats_flat = group_advantages(eps, rewards, group_key=default_group_key)
    advs_br, stats_br = group_advantages(eps, rewards, group_key=branch_group_key)

    # default: ONE group over all 6 (mean 3/6=0.5); branch: TWO groups of 3
    assert sum(s.kept for s in stats_flat) == 1
    assert sum(s.kept for s in stats_br) == 2

    a_ids = {e.episode_id for e in eps if e.branch_of in ("", "trunk-A")}
    # a passing episode in trunk A: branch advantage 1 - 1/3 = 2/3 ; under flat grouping it would be 1 - 0.5 = 0.5
    pass_a = next(e.episode_id for e in eps if e.episode_id.startswith("trunk-A") and rewards[e.episode_id] == 1.0)
    assert abs(advs_br[pass_a] - (1 - 1 / 3)) < 1e-9
    assert abs(advs_flat[pass_a] - 0.5) < 1e-9
    # groups carry their full key
    assert {s.group for s in stats_br} == {(0, "t0", "trunk-A"), (0, "t0", "trunk-B")}


def test_build_threads_group_key_and_is_contrastive_per_trunk():
    rng = random.Random(1)
    eps = _trunk_and_branches(rng, "t0", 0, "A", [True, False, False]) + _trunk_and_branches(rng, "t0", 0, "B", [True, True, False])
    rewards = compute_rewards(eps)
    b = build(eps, dict(rewards), allowed_tasks={"t0"}, group_key=branch_group_key)
    adv_by_id = {d.episode_id: d.advantage for d in b.datums}
    # trunk A's single passer has advantage 2/3 under per-trunk grouping
    pass_a = next(e.episode_id for e in eps if e.episode_id.startswith("trunk-A") and rewards[e.episode_id] == 1.0)
    assert abs(adv_by_id[pass_a] - (1 - 1 / 3)) < 1e-9
    # two kept groups, one per trunk
    assert sum(g.kept for g in b.groups) == 2


def test_default_grouping_unchanged_when_no_branch_fields():
    rng = random.Random(2)
    eps = [make_episode(rng, task="t0", step=0, epoch=i, passed=p) for i, p in enumerate([True, False])]
    rewards = compute_rewards(eps)
    a1, s1 = group_advantages(eps, rewards)                      # default group_key
    a2, s2 = group_advantages(eps, rewards, group_key=default_group_key)
    assert a1 == a2 and len(s1) == len(s2) == 1
