"""Real provider sidecar (fake sampler, real tokenizer and renderer) -> Episode -> datum. Skips when the gpt-oss tokenizer is not available offline."""

import pytest

pytest.importorskip("test_provider")
from test_provider import RUN, make, start, step  # noqa: E402

from rl.fake_sampler import call, final  # noqa: E402
from rl.datums import build  # noqa: E402
from rl.provider import load_sidecar  # noqa: E402
from rl.rewards import compute_rewards  # noqa: E402
from rl.schema import episode_from_sidecar  # noqa: E402


def test_real_sidecars_become_aligned_datums(tmp_path):
    eps = []
    for i in range(4):
        model, _ = make(tmp_path, [call("look", "bash", '{"command": "ls"}'), call("again", "bash", '{"command": "pwd"}'), call("done", "submit", '{"answer": "x"}')])
        msgs = start()

        async def ep():
            await step(model, msgs, "file.txt")
            await step(model, msgs, "/app")
            await step(model, msgs)
        RUN(ep())
        side = sorted(tmp_path.glob("*.json.gz"), key=lambda p: p.stat().st_mtime)[-1]
        e = episode_from_sidecar(load_sidecar(side), step=0)
        e.episode_id, e.sample_id, e.task, e.epoch, e.passed = f"e{i}", "t", "t", i, i % 2 == 0
        eps.append(e)
    assert all(e.ended == "submit" and len(e.turns) == 3 and not e.non_trainable for e in eps)
    b = build(eps, compute_rewards(eps))
    assert len(b.datums) == 4 and all(d.n_trained_tokens == sum(len(t.tokens) for t in e.turns) for d, e in zip(b.datums, eps))
