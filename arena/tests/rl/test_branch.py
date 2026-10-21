"""Offline tests for the branch-sampling token primitive (rl/branch.py): a branch shares the trunk's exact
prefix (trainable), lays out a valid prompt_len chain, and groups with its trunk. No docker, no Tinker."""

import random

import pytest
from fakes import make_episode

from rl.branch import (BranchError, branch_prefix, default_cut, make_branch_episode, shared_prefix_ok)
from rl.datums import build
from rl.rewards import branch_group_key, compute_rewards
from rl.schema import Turn


def _cont_turn(rng, n=12, obs=8, stop="stop"):
    toks = [rng.randrange(100, 5000) for _ in range(n)]
    toks[-1] = 200012
    return Turn(prompt_len=0, tokens=toks, logprobs=[-rng.random() * 3 for _ in toks],
                stop_reason=stop, obs_tokens=([rng.randrange(100, 5000) for _ in range(obs)] + [200006, 173781]) if obs else [])


def test_prefix_range_and_default_cut():
    rng = random.Random(0)
    trunk = make_episode(rng, n_turns=6)
    assert default_cut(trunk, from_end=2) == 4           # resample last 2 of 6
    with pytest.raises(BranchError):
        branch_prefix(trunk, 0)                          # must keep >=1 turn
    with pytest.raises(BranchError):
        branch_prefix(trunk, len(trunk.turns))           # must leave >=1 turn to resample
    p = branch_prefix(trunk, 4)
    assert len(p.turns) == 4 and p.length == len(p.tokens)


def test_branch_shares_exact_prefix_and_is_trainable():
    rng = random.Random(1)
    trunk = make_episode(rng, n_turns=6, passed=True)
    cut = default_cut(trunk, from_end=2)
    cont = [_cont_turn(rng, obs=8, stop="stop"), _cont_turn(rng, obs=0, stop="stop")]
    branch = make_branch_episode(trunk, cut, cont, index=1)

    assert branch.branch_of == trunk.episode_id and branch.cut_turn == cut
    assert shared_prefix_ok(trunk, branch)               # token-for-token prefix match
    # first continuation turn starts exactly at the shared-prefix length
    assert branch.turns[cut].prompt_len == branch_prefix(trunk, cut).length
    # prompt_len chain is self-consistent (what datums.check_tokens verifies)
    length = len(branch.prompt_tokens)
    for t in branch.turns:
        assert t.prompt_len == length
        length += len(t.tokens) + len(t.obs_tokens)

    # the branch trains: build() accepts it (give it a sibling so the group has signal)
    branch2 = make_branch_episode(trunk, cut, [_cont_turn(rng, obs=0)], index=2)
    branch2.passed = False
    branch.passed = True
    eps = [trunk, branch, branch2]
    rewards = compute_rewards(eps)
    b = build(eps, dict(rewards), allowed_tasks={trunk.task}, group_key=branch_group_key)
    ids = {d.episode_id for d in b.datums}
    assert branch.episode_id in ids and branch2.episode_id in ids
    # one contrastive group: trunk + its two branches
    assert sum(g.kept for g in b.groups) == 1
    assert {s.group for s in b.groups} == {(trunk.step, trunk.task, trunk.episode_id)}


def test_drift_breaks_the_prefix_guard():
    rng = random.Random(2)
    trunk = make_episode(rng, n_turns=5)
    cut = 3
    branch = make_branch_episode(trunk, cut, [_cont_turn(rng, obs=0)], index=1)
    assert shared_prefix_ok(trunk, branch)
    # simulate environment drift: a token WITHIN the shared region (an earlier kept turn's obs) differs from the
    # trunk's -> the trainable-prefix invariant is violated and the branch must be dropped.
    branch.turns[0].obs_tokens[0] = 999999
    assert not shared_prefix_ok(trunk, branch)
