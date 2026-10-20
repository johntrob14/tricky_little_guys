import random

import pytest

from rl.datums import Refused, build
from rl.rewards import compute_rewards
from rl.schema import Episode, episode_from_sidecar


def doc(rng, n=3, **kw):
    prompt = [rng.randrange(100, 900) for _ in range(12)]
    turns, length = [], len(prompt)
    for k in range(n):
        obs = [rng.randrange(100, 900) for _ in range(5)] + [200006, 173781] if k else []
        length += len(obs)
        s = [rng.randrange(100, 900) for _ in range(7)] + [200012]
        t = dict(index=k, prompt_source="initial" if k == 0 else "chain", prompt_len=length, obs_ids=obs, sampled_ids=s, logprobs=[-0.5] * len(s), stop_reason="stop",
                 weights_id="w5", flags={"empty_reasoning": k == 1, "truncated": False}, tool_call=dict(name="bash" if k < n - 1 else "submit"))
        if k == 0:
            t["prompt_ids"] = prompt
        turns.append(t)
        length += len(s)
    d = dict(schema="arena-rl-sidecar/1", episode_id="uuid1", sample_id="task_a", epoch=2, task="task_a", run="r", model="openai/gpt-oss-120b", renderer="gpt_oss_medium_reasoning",
             current_date="2026-10-20", sampling=dict(temperature=1.0, top_p=1.0, top_k=-1, seed=None, stop_ids=[200002, 200012], max_tokens=4096), trainable=True,
             non_trainable_reasons=[], end_event=None, turns=turns)
    d.update(kw)
    return d


def test_converts_and_builds_a_datum():
    rng = random.Random(0)
    eps = []
    for i in range(4):
        e = episode_from_sidecar(doc(rng, episode_id=f"u{i}"), step=3, log_path="x.eval")
        e.passed = i % 2 == 0
        eps.append(e)
    e = eps[0]
    assert (e.weights_id, e.task, e.epoch, e.ended, e.sampling.top_k, e.log_path) == ("w5", "task_a", 2, "submit", -1, "x.eval")
    assert e.turns[0].obs_tokens[-2:] == [200006, 173781] and e.turns[-1].obs_tokens == [] and e.turns[1].flags == ["empty_reasoning"]
    assert Episode.from_dict(e.to_dict()) == e
    b = build(eps, compute_rewards(eps), allowed_tasks={"task_a"})
    assert len(b.datums) == 4 and b.datums[0].n_trained_tokens == 3 * 8


def test_rerender_and_provider_flags_are_refused():
    rng = random.Random(1)
    d = doc(rng)
    d["turns"][2]["prompt_source"] = "rerender"
    e = episode_from_sidecar(d, step=0)
    e.passed = True
    e2 = episode_from_sidecar(doc(rng, episode_id="u2", non_trainable_reasons=["mixed_weights"]), step=0)
    e2.passed = False
    for bad in (e, e2):
        with pytest.raises(Refused):
            build([bad], compute_rewards([bad]))
    assert e2.weights_id == "w5" and "mixed_weights" in e2.non_trainable


def test_end_inference():
    rng = random.Random(2)
    assert episode_from_sidecar(doc(rng, end_event=dict(type="context_clamp", prompt_len=30000)), step=0).ended == "context"
    d = doc(rng)
    d["turns"][-1]["stop_reason"] = "length"
    assert episode_from_sidecar(d, step=0).ended == "truncated"
    d = doc(rng)
    d["turns"][-1]["tool_call"] = None
    assert episode_from_sidecar(d, step=0).ended == "no_tool_call"
