import copy
import glob
import math
import os
import random
from pathlib import Path

import pytest

from fakes import make_episode
from rl.datums import Refused, build, merged_datum_fields, to_tinker, turn_advantages
from rl.rewards import compute_rewards, group_advantages
from rl.schema import Episode, Turn, make_episode_id


def _offline_tokenizer():
    """gpt-oss tokenizer from the offline HF cache, or None (tests skip) when it is not present."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    hf = "models--openai--gpt-oss-120b"
    if not (Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub" / hf).exists():
        for home in [Path.home() / ".cache/huggingface", *map(Path, sorted(glob.glob(str(Path.home() / "offload/*/huggingface"))))]:
            if (home / "hub" / hf).exists():
                os.environ["HF_HOME"] = str(home)
                break
    try:
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        return get_tokenizer("openai/gpt-oss-120b")
    except Exception:  # noqa: BLE001
        return None


def reference_arrays(ep, adv):
    """Independent builder: walk the history token by token, tagging each full-sequence position with its role."""
    full, role = [], []  # role: None | (turn, index within turn)
    full += ep.prompt_tokens; role += [None] * len(ep.prompt_tokens)
    for k, t in enumerate(ep.turns):
        for i, x in enumerate(t.tokens):
            full.append(x); role.append((k, i))
        if k < len(ep.turns) - 1:
            full += t.obs_tokens; role += [None] * len(t.obs_tokens)
    lp, ad = [], []
    for r in role:
        if r is None:
            lp.append(0.0); ad.append(0.0)
        else:
            t = ep.turns[r[0]]
            lp.append(t.logprobs[r[1]]); ad.append(0.0 if t.stop_reason == "length" else adv)
    return full[:-1], full[1:], lp[1:], ad[1:]


def group(rng, task="a", step=0, passes=(True, False, False, True), wid="w0", **kw):
    return [make_episode(rng, task=task, step=step, epoch=i, wid=wid, passed=p, **kw) for i, p in enumerate(passes)]


def test_alignment_fuzz_200():
    rng = random.Random(1)
    n_trunc = n_datums = 0
    for i in range(200):
        eps = group(rng, task=f"t{i}", passes=(True, False, rng.random() < 0.5), stop_last=rng.choice(["stop", "length"]))
        b = build(eps, compute_rewards(eps))
        for d in b.datums:
            ep = next(e for e in eps if e.episode_id == d.episode_id)
            mi, tg, lp, ad = reference_arrays(ep, d.advantage)
            f = d.fields
            assert (f["model_input"], f["target_tokens"], f["logprobs"], f["advantages"]) == (mi, tg, lp, ad)
            assert len(f["model_input"]) == len(f["target_tokens"]) == len(f["logprobs"]) == len(f["advantages"])
            assert d.n_trained_tokens == sum(len(t.tokens) for t in ep.turns if t.stop_reason != "length")
            assert f["target_tokens"][-1] == ep.turns[-1].tokens[-1]  # the last sampled token (stop token) is the last target
            n_trunc += d.truncated_turns
            n_datums += 1
            to_tinker(f)
    assert n_datums > 200 and n_trunc > 20


def test_advantage_math():
    rng = random.Random(2)
    eps = group(rng, task="a", passes=(True, True, False, False, False)) + group(rng, task="b", passes=(True, True)) + group(rng, task="c", passes=(False, False, False)) \
        + group(rng, task="d", passes=(True, False), step=1)
    adv, stats = group_advantages(eps, compute_rewards(eps))
    a = {e.episode_id: adv.get(e.episode_id) for e in eps}
    assert [round(a[e.episode_id], 6) for e in eps[:5]] == [0.6, 0.6, -0.4, -0.4, -0.4]
    assert sum(adv[e.episode_id] for e in eps[:5]) == pytest.approx(0)
    assert all(a[e.episode_id] is None for e in eps[5:10])  # constant groups b and c are dropped
    assert {(s.task, s.drop_reason) for s in stats if not s.kept} == {("b", "constant reward"), ("c", "constant reward")}
    assert [a[e.episode_id] for e in eps[10:]] == [0.5, -0.5]
    adv_std, _ = group_advantages(eps[:5], compute_rewards(eps[:5]), adv_norm="std")
    assert adv_std[eps[0].episode_id] == pytest.approx(0.6 / (math.sqrt(0.24) + 1e-6))


def test_infra_and_none_excluded_from_group_mean():
    rng = random.Random(3)
    eps = group(rng, passes=(True, False, False))
    eps.append(make_episode(rng, task="a", epoch=9, passed=None, failure_class="infra", ended="infra"))
    eps.append(make_episode(rng, task="a", epoch=10, passed=True, ended="wall"))
    eps.append(make_episode(rng, task="a", epoch=11, passed=False, ended="step_deadline"))
    r = compute_rewards(eps)
    assert [r[e.episode_id] for e in eps] == [1.0, 0.0, 0.0, None, None, None]
    b = build(eps, r)
    assert b.n_valid == 3 and b.n_none == 3 and len(b.datums) == 3
    assert sorted(round(d.advantage, 6) for d in b.datums) == [-0.333333, -0.333333, 0.666667]
    assert {x["reason"] for x in b.dropped} == {"no reward"}


def test_truncated_turn_gets_zero_advantage():
    rng = random.Random(4)
    eps = group(rng, passes=(True, False), n_turns=3)
    for e in eps:
        e.turns[1].stop_reason = "length"  # mid-episode truncation as in a length-capped turn followed by a nudge
    b = build(eps, compute_rewards(eps))
    for d in b.datums:
        ep = next(e for e in eps if e.episode_id == d.episode_id)
        assert d.n_trained_tokens == len(ep.turns[0].tokens) + len(ep.turns[2].tokens)
        assert d.truncated_turns == 1
    assert turn_advantages(eps[0].turns, 0.5)[1] == [0.0] * len(eps[0].turns[1].tokens)


def test_over_context_dropped_not_raised():
    rng = random.Random(5)
    eps = group(rng, passes=(True, False))
    b = build(eps, compute_rewards(eps), max_len=10)
    assert not b.datums and all("over context" in x["reason"] for x in b.dropped)


def refusal(mutate, code, **build_kw):
    rng = random.Random(6)
    eps = group(rng)
    mutate(eps[1])
    with pytest.raises(Refused) as ei:
        build(eps, compute_rewards(eps), **build_kw)
    assert ei.value.code == code


def test_refusals():
    refusal(lambda e: setattr(e, "split", "heldout"), "split")
    refusal(lambda e: setattr(e, "phase", "eval"), "phase")
    refusal(lambda e: None, "task", allowed_tasks={"other"})
    refusal(lambda e: setattr(e.sampling, "temperature", 0.7), "temperature")
    refusal(lambda e: setattr(e.sampling, "temperature", None), "temperature")
    refusal(lambda e: setattr(e.sampling, "top_p", 0.95), "top_p")
    refusal(lambda e: setattr(e.sampling, "top_k", 50), "top_k")
    refusal(lambda e: setattr(e, "scripted_turns", 1), "scripted")
    refusal(lambda e: e.turns[0].flags.append("scripted"), "scripted")
    refusal(lambda e: e.turns[-1].logprobs.__setitem__(0, float("nan")), "nonfinite")
    refusal(lambda e: e.turns[-1].logprobs.__setitem__(0, float("-inf")), "nonfinite")
    refusal(lambda e: e.turns[-1].logprobs.__setitem__(0, 0.5), "logprobs")
    refusal(lambda e: setattr(e.turns[-1], "prompt_len", e.turns[-1].prompt_len + 1), "misaligned")
    refusal(lambda e: e.turns[0].logprobs.pop(), "misaligned")


def test_refuses_other_weights_and_non_train_even_when_excluded():
    rng = random.Random(7)
    eps = group(rng)
    with pytest.raises(ValueError, match="sampled by weights"):
        build(eps, compute_rewards(eps), expect_weights_id="w1")
    eps[0].split, eps[0].passed, eps[0].failure_class = "heldout", None, "infra"  # an INFRA row of a non-train task is still a scope violation
    with pytest.raises(Refused) as ei:
        build(eps, compute_rewards(eps))
    assert ei.value.code == "split"


def test_infra_rows_skip_token_checks():
    rng = random.Random(8)
    eps = group(rng, passes=(True, False, False))
    bad = make_episode(rng, task="a", epoch=7, passed=None, failure_class="infra", ended="infra")
    bad.sampling.temperature, bad.scripted_turns = 0.2, 3
    b = build(eps + [bad], compute_rewards(eps + [bad]))
    assert len(b.datums) == 3


def test_datum_tinker_keys_and_zero_context():
    rng = random.Random(9)
    eps = group(rng, passes=(True, False))
    b = build(eps, compute_rewards(eps))
    d = b.datums[0]
    td = d.tinker()
    assert set(td.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
    n_prompt = len(eps[0].prompt_tokens) - 1
    assert all(x == 0.0 for x in d.fields["advantages"][:n_prompt]) and all(x == 0.0 for x in d.fields["logprobs"][:n_prompt])
    assert td.loss_fn_inputs["advantages"].tolist() == d.fields["advantages"]


def _fields_by_id(b):
    return {d.episode_id: (d.fields, d.advantage, d.n_trained_tokens) for d in b.datums}


def test_cot_penalty_default_byte_identical():
    """None, beta 0, a zero suspicion map, and an episode absent from the map all reproduce today's advantages exactly."""
    rng = random.Random(11)
    eps = group(rng, task="x", passes=(True, False, True, False), stop_last="stop") + group(rng, task="y", passes=(True, False, False), stop_last="length")
    base = _fields_by_id(build(eps, compute_rewards(eps)))
    assert base
    n_turns = {e.episode_id: len(e.turns) for e in eps}
    cases = [
        None,
        (0.0, {eid: [1.0] * n for eid, n in n_turns.items()}),  # beta 0
        (2.0, {eid: [0.0] * n for eid, n in n_turns.items()}),  # zero suspicion map
        (2.0, {}),  # every episode absent from the map
    ]
    for cp in cases:
        got = _fields_by_id(build(eps, compute_rewards(eps), cot_penalty=cp))
        assert got == base, cp


def test_cot_penalty_leaves_action_and_final_tokens_untouched():
    """A penalized build changes only analysis-channel token advantages; tool-call and final-answer positions equal the no-penalty case."""
    tok = _offline_tokenizer()
    if tok is None:
        pytest.skip("gpt-oss tokenizer unavailable offline")
    from rl.cot_mask import think_mask
    from rl.fake_sampler import call, final

    def harmony_ep(epoch, passed, s):
        toks = tok.encode(s, add_special_tokens=False)
        prompt = [200006, 173781, 123, 456]
        turns = [Turn(prompt_len=len(prompt), tokens=toks, logprobs=[-0.5] * len(toks), stop_reason="stop")]
        eid = make_episode_id("r", 0, "t", epoch, "w0")
        return Episode(episode_id=eid, sample_id="t", task="t", epoch=epoch, step=0, weights_id="w0",
                       prompt_tokens=prompt, turns=turns, model="openai/gpt-oss-120b", passed=passed)

    eps = [harmony_ep(0, True, call("secret plan", "bash", '{"command": "ls"}')),
           harmony_ep(1, False, final("more secret plan", "the public answer"))]
    base = build(eps, compute_rewards(eps))
    cmap = {e.episode_id: [1.0] for e in eps}
    beta = 2.0
    pen = build(eps, compute_rewards(eps), cot_penalty=(beta, cmap))
    assert len(base.datums) == len(pen.datums) == 2
    for e in eps:
        bd = next(d for d in base.datums if d.episode_id == e.episode_id)
        pd = next(d for d in pen.datums if d.episode_id == e.episode_id)
        ba, pa = bd.fields["advantages"], pd.fields["advantages"]
        assert len(ba) == len(pa)
        mask = think_mask(e.turns[0].tokens, tok)
        assert any(mask) and not all(mask)  # a real mix of thinking and non-thinking tokens
        L = len(e.prompt_tokens)
        assert pa[: L - 1] == ba[: L - 1]  # prompt region untouched
        for i, m in enumerate(mask):
            j = L + i - 1  # advantages index of sampled token i (ad[1:])
            if m:
                assert pa[j] == pytest.approx(ba[j] - beta)  # c = 1.0
            else:
                assert pa[j] == ba[j]  # action / tool-call / final-answer tokens unchanged
        assert all(math.isfinite(x) for x in pa)
