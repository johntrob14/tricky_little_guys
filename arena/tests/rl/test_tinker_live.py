"""Live Tinker check of the training core (G2). Written, NOT run: needs John's explicit go and ARENA_RL_RUN_TINKER=1. Costs well under USD 0.01 on 20b."""

import asyncio
import random

import pytest

from fakes import make_episode
from rl.datums import build
from rl.rewards import compute_rewards
from rl.train import TrainHP, new_service, open_training_client, save_checkpoint, train_step

pytestmark = pytest.mark.tinker


def test_one_real_step_on_20b():
    async def go():
        svc = new_service("live-check", "G2")
        hp = TrainHP(rank=8, learning_rate=5e-5)
        tc = await open_training_client(svc, hp, "openai/gpt-oss-20b")
        rng = random.Random(0)
        eps = [make_episode(rng, task="a", epoch=i, passed=i % 2 == 0, n_turns=1) for i in range(4)]  # random tokens: only checks the plumbing, lp_gap will be large
        m = await train_step(tc, build(eps, compute_rewards(eps)), hp)
        assert m.applied and m.lp_gap_mean is not None
        ck = await save_checkpoint(tc, "sampler", "arena-rl-live-check", 3600)
        assert ck["path"].startswith("tinker://")
    asyncio.run(go())
