"""Offline test for the branch-sampling collect orchestration (rl/branch_collect.py): a step produces 1 trunk +
(G-1) branches per task, trunk and branches form one contrastive group, drift branches are skipped, and the
result trains through datums.build with branch_group_key. No docker, no Tinker."""

import asyncio
import random

from fakes import make_episode

from rl.branch import make_branch_episode
from rl.branch_collect import BranchCollector
from rl.loop import CollectRequest
from rl.rewards import branch_group_key, compute_rewards
from rl.schema import Turn


def _cont(rng, stop="stop"):
    toks = [rng.randrange(100, 5000) for _ in range(10)] + [200012]
    return Turn(prompt_len=0, tokens=toks, logprobs=[-rng.random() * 3 for _ in toks], stop_reason=stop, obs_tokens=[])


class _FakeBackend:
    """make_branch(trunk,...) -> a branch via make_branch_episode, or None for a drift index to exercise skip."""
    def __init__(self, rng, pass_by_index, drift_indices=()):
        self.rng, self.pass_by_index, self.drift = rng, pass_by_index, set(drift_indices)

    async def make_branch(self, trunk, sampler, *, cut, max_turns, index, weights_id):
        if index in self.drift:
            return None
        b = make_branch_episode(trunk, cut, [_cont(self.rng)], index=index, weights_id=weights_id)
        b.passed = self.pass_by_index.get(index, False)
        return b


def _req(tasks, n, wid):
    return CollectRequest(step=0, tasks=tasks, n=n, sampler=object(), weights_id=wid,
                          out_dir="/tmp", deadline_s=None, budget=None, ep_worst_usd=0.0)


def test_branch_collect_structure_grouping_and_trainable():
    rng = random.Random(0)
    wid = "w0"

    async def trunk_runner(task, req):
        return make_episode(rng, task=task, step=0, wid=wid, n_turns=6, passed=True)

    async def score_branch(ep):
        ep.score_value = "C" if ep.passed else "I"

    # G=4 -> 1 trunk + up to 3 branches; index 2 drifts (returns None) -> 2 branches survive.
    backend = _FakeBackend(rng, pass_by_index={1: False, 3: False}, drift_indices=(2,))
    col = BranchCollector(trunk_runner=trunk_runner, make_backend=lambda trunk: backend, score_branch=score_branch, max_turns=10)
    eps = asyncio.run(col(_req(["t0"], 4, wid)))

    trunks = [e for e in eps if e.branch_of == ""]
    branches = [e for e in eps if e.branch_of]
    assert len(trunks) == 1 and len(branches) == 2            # 1 trunk + (3 attempted - 1 drift) branches
    assert all(b.branch_of == trunks[0].episode_id and b.cut_turn is not None for b in branches)

    # One contrastive group (trunk + its branches); trunk passed, branches failed -> signal present.
    rewards = compute_rewards(eps)
    from rl.datums import build
    b = build(eps, dict(rewards), allowed_tasks={"t0"}, expect_weights_id=wid, group_key=branch_group_key)
    assert {s.group for s in b.groups} == {(0, "t0", trunks[0].episode_id)}
    assert sum(g.kept for g in b.groups) == 1                 # the group has signal and trains
    assert {d.episode_id for d in b.datums} == {e.episode_id for e in eps}


def test_two_tasks_two_groups():
    rng = random.Random(1)
    wid = "w0"

    async def trunk_runner(task, req):
        return make_episode(rng, task=task, step=0, wid=wid, n_turns=5, passed=(task == "t0"))

    async def score_branch(ep):
        ep.passed = not ep.passed  # branches differ from their trunk -> each group has signal

    backend = _FakeBackend(rng, pass_by_index={1: True, 2: True})
    col = BranchCollector(trunk_runner=trunk_runner, make_backend=lambda trunk: backend, score_branch=score_branch, max_turns=10)
    eps = asyncio.run(col(_req(["t0", "t1"], 3, wid)))        # 2 tasks x (1 trunk + 2 branches) = 6
    assert len(eps) == 6
    assert len([e for e in eps if e.branch_of == ""]) == 2
    from rl.datums import build
    b = build(eps, dict(compute_rewards(eps)), allowed_tasks={"t0", "t1"}, expect_weights_id=wid, group_key=branch_group_key)
    assert len([s for s in b.groups]) == 2                    # one group per trunk
