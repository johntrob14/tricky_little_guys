"""CPU tests for the batched SWITCH latent loops (cotswitch/switch_model.py): the paged
engine (PagedSwitchEngine, default) and the original HF-forward loop (`engine="hf"`).

The reference is the vendored repo's batch-1 `CoconutSwiModel.generate()`, run on a
tiny random Qwen3 in float64 (so batching cannot flip an argmax through rounding).
The lm_head rows of <swi>, </swi> and EOS are scaled up so that latent blocks,
natural and forced exits, EOS stops and length stops all occur.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "SWITCH"))

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

import src.model.coconut_swi_model as csm  # noqa: E402  (vendored SWITCH repo)
from src.setup_tokens import SwiTokenConfig  # noqa: E402

from cotswitch.switch_model import SwitchTokens, generate_batch, GenStats  # noqa: E402

PAD, EOS, S, E, LAT = 59, 60, 61, 62, 63
TOK = SwitchTokens(swi_start_id=S, swi_end_id=E, eos_token_id=EOS, pad_token_id=PAD, latent_id=LAT)
CAP = 6  # small latent cap so forced exits happen


def build(seed, sS, sE, sEOS):
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
        tie_word_embeddings=False, eos_token_id=EOS, pad_token_id=PAD,
    )
    m = Qwen3ForCausalLM(cfg).to(torch.float64).eval()
    with torch.no_grad():
        W = m.lm_head.weight
        W[S] *= sS
        W[E] *= sE
        W[EOS] *= sEOS
    return m


def prompts_for(seed, lengths=(3, 7, 12, 5, 9, 1, 15, 4)):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 58, (n,), generator=g).tolist() for n in lengths]


def reference(model, prompts, max_new, kmin, monkeypatch):
    monkeypatch.setattr(csm, "MAX_LATENT_PER_BLOCK", CAP)
    ref = csm.CoconutSwiModel(model, SwiTokenConfig(S, E, LAT, EOS))
    out = []
    for p in prompts:
        ids, info = ref.generate(torch.tensor([p]), max_new_tokens=max_new, min_latent_steps=kmin)
        out.append((ids[0, len(p):].tolist(), info))
    return out


def check(refs, mine, max_new):
    n_blocks = 0
    for (ref_ids, ref_info), r in zip(refs, mine):
        assert r["output_ids"] == ref_ids
        blocks = r["latent"]["blocks"]
        assert [(b["token_pos"] + 1, b["n_steps"], b["natural_exit"]) for b in blocks] == [
            (b["position"], b["n_latent_steps"], b["natural_exit"]) for b in ref_info
        ]
        assert r["latent"]["n_blocks"] == len(ref_info)
        assert r["latent"]["n_latent_steps"] == sum(b["n_latent_steps"] for b in ref_info)
        for b in blocks:
            assert ref_ids[b["token_pos"]] == S and ref_ids[b["token_pos"] + 1] == E
        stopped = bool(ref_ids) and ref_ids[-1] == EOS
        assert r["finish_reason"] == ("stop" if stopped else "length")
        n_blocks += len(blocks)
    return n_blocks


CONFIGS = [(2, 6, 3, 1), (2, 3, 3, 3), (1, 6, 3, 3), (0, 6, 3, 3)]
ENGINES = ["paged", "hf"]
# small blocks / sync interval / table so the paged engine crosses block boundaries,
# harvests rows mid-interval and stays cheap on CPU
PAGED = dict(engine="paged", block_size=4, sync_every=3, max_seq_len=160)


def run(model, prompts, engine, **kw):
    if engine == "paged":
        kw = {**PAGED, **kw}
    else:
        kw = {"engine": "hf", **kw}
    return generate_batch(model, TOK, prompts, **kw)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("kmin", [0, 2, 4])
def test_static_batch_matches_repo(engine, cfg, kmin, monkeypatch):
    model = build(*cfg)
    prompts = prompts_for(cfg[0])
    refs = reference(model, prompts, 40, kmin, monkeypatch)
    mine = run(model, prompts, engine, max_new_tokens=40, min_latent_steps=kmin,
               max_latent_per_block=CAP, max_batch=len(prompts))
    check(refs, mine, 40)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("cfg", CONFIGS)
def test_batch_of_one_matches_repo(engine, cfg, monkeypatch):
    model = build(*cfg)
    prompts = prompts_for(cfg[0])
    refs = reference(model, prompts, 30, 2, monkeypatch)
    mine = run(model, prompts, engine, max_new_tokens=30, min_latent_steps=2,
               max_latent_per_block=CAP, max_batch=1)
    check(refs, mine, 30)


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("max_batch,admit_chunk", [(2, 1), (3, 2), (5, 1)])
def test_continuous_batching_matches_repo_hf(cfg, max_batch, admit_chunk, monkeypatch):
    model = build(*cfg)
    prompts = prompts_for(cfg[0], lengths=(3, 7, 12, 5, 9, 1, 15, 4, 20, 2, 6))
    refs = reference(model, prompts, 50, 2, monkeypatch)
    mine = run(model, prompts, "hf", max_new_tokens=50, min_latent_steps=2,
               max_latent_per_block=CAP, max_batch=max_batch, admit_chunk=admit_chunk)
    check(refs, mine, 50)


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("max_batch,sync_every,block_size", [(2, 1, 1), (3, 3, 4), (5, 7, 2), (4, 16, 8)])
def test_continuous_batching_matches_repo_paged(cfg, max_batch, sync_every, block_size, monkeypatch):
    model = build(*cfg)
    prompts = prompts_for(cfg[0], lengths=(3, 7, 12, 5, 9, 1, 15, 4, 20, 2, 6))
    refs = reference(model, prompts, 50, 2, monkeypatch)
    stats = GenStats()
    mine = generate_batch(model, TOK, prompts, engine="paged", max_new_tokens=50, min_latent_steps=2,
                          max_latent_per_block=CAP, max_batch=max_batch, sync_every=sync_every,
                          block_size=block_size, max_seq_len=1000, stats=stats)
    check(refs, mine, 50)
    assert stats.rows_done == len(prompts) and stats.deferred == 0
    assert stats.tokens_out == sum(len(r["output_ids"]) for r in mine)


@pytest.mark.parametrize("engine", ENGINES)
def test_eviction_requeues_and_matches(engine, monkeypatch):
    model = build(2, 6, 3, 1)
    prompts = prompts_for(2, lengths=(3, 7, 12, 5, 9, 1))
    refs = reference(model, prompts, 40, 2, monkeypatch)
    stats = GenStats()
    # budget forces evictions: 4 rows * ~60 columns > 200 slots
    mine = run(model, prompts, engine, max_new_tokens=40, min_latent_steps=2,
               max_latent_per_block=CAP, max_batch=4, kv_budget_tokens=200, stats=stats)
    check(refs, mine, 40)
    assert stats.evictions > 0


@pytest.mark.parametrize("swap", [True, False])
@pytest.mark.parametrize("cfg", CONFIGS)
def test_paged_preemption_swap_and_restart_match(cfg, swap, monkeypatch):
    """A tight block pool forces preemptions; swapped rows resume exactly where they were
    (KV + state machine restored), restarted rows recompute; both match the repo."""
    model = build(*cfg)
    prompts = prompts_for(cfg[0], lengths=(3, 7, 12, 5, 9, 1, 15, 4))
    refs = reference(model, prompts, 50, 2, monkeypatch)
    stats = GenStats()
    mine = generate_batch(model, TOK, prompts, engine="paged", max_new_tokens=50, min_latent_steps=2,
                          max_latent_per_block=CAP, max_batch=6, block_size=4, sync_every=3, max_seq_len=400,
                          kv_budget_tokens=240, swap=swap, admit_watermark=0.3, stats=stats)
    check(refs, mine, 50)
    if cfg != CONFIGS[0]:  # cfg0 is known to preempt in both modes; the others just must match
        return
    assert stats.evictions > 0
    if swap:
        assert stats.swapped_positions > 0
    else:
        assert stats.recomputed_positions > 0 and stats.swapped_positions == 0


def test_paged_compaction_matches(monkeypatch):
    """Rows finish out of order, leaving holes; compaction moves rows to low slots."""
    model = build(2, 6, 3, 1)
    prompts = prompts_for(2, lengths=(3, 7, 12, 5, 9, 1, 15, 4, 20, 2, 6, 8))
    refs = reference(model, prompts, 40, 2, monkeypatch)
    stats = GenStats()
    mine = generate_batch(model, TOK, prompts, engine="paged", max_new_tokens=40, min_latent_steps=2,
                          max_latent_per_block=CAP, max_batch=8, block_size=4, sync_every=2, max_seq_len=400,
                          stats=stats)
    check(refs, mine, 40)
    assert stats.compactions > 0


def test_paged_overflow_rows_are_regenerated_unpaged(monkeypatch):
    """Rows that outgrow max_seq_len go to the HF loop and still match the repo."""
    model = build(2, 6, 3, 1)
    prompts = prompts_for(2, lengths=(3, 7, 12, 5, 9, 1, 30))
    refs = reference(model, prompts, 40, 2, monkeypatch)
    stats = GenStats()
    mine = generate_batch(model, TOK, prompts, engine="paged", max_new_tokens=40, min_latent_steps=2,
                          max_latent_per_block=CAP, max_batch=3, block_size=4, sync_every=3,
                          max_seq_len=40, stats=stats)
    check(refs, mine, 40)
    assert stats.deferred > 0


def test_paged_state_resets_between_rows_in_a_slot(monkeypatch):
    """One slot, many prompts: every row reuses slot 0 and recycled blocks."""
    model = build(1, 6, 3, 3)
    prompts = prompts_for(1, lengths=(3, 7, 12, 5, 9, 1, 15))
    refs = reference(model, prompts, 30, 4, monkeypatch)
    mine = generate_batch(model, TOK, prompts, engine="paged", max_new_tokens=30, min_latent_steps=4,
                          max_latent_per_block=CAP, max_batch=1, block_size=2, sync_every=5,
                          max_seq_len=120, kv_budget_tokens=400)
    check(refs, mine, 30)


@pytest.mark.parametrize("engine", ENGINES)
def test_coverage_of_cases(engine):
    """Guard against a vacuous suite: the configs must exercise every branch."""
    seen = {"natural": 0, "forced": 0, "stop": 0, "length": 0, "over_budget": 0}
    for cfg in CONFIGS:
        model = build(*cfg)
        for r in run(model, prompts_for(cfg[0]), engine, max_new_tokens=40, min_latent_steps=2,
                     max_latent_per_block=CAP, max_batch=8):
            for b in r["latent"]["blocks"]:
                seen["natural" if b["natural_exit"] else "forced"] += 1
            seen[r["finish_reason"]] += 1
            seen["over_budget"] += r["n_output_tokens"] > 40
    assert all(v > 0 for v in seen.values()), seen


@pytest.mark.parametrize("engine", ENGINES)
def test_zero_budget_and_immediate_eos(engine):
    model = build(2, 3, 3, 3)
    prompts = prompts_for(2)
    rs = run(model, prompts, engine, max_new_tokens=0, max_batch=4)
    assert all(r["output_ids"] == [] and r["finish_reason"] == "length" for r in rs)


def test_fused_weights_leave_hf_forward_unchanged():
    from cotswitch.switch_model import PagedSwitchEngine

    model = build(2, 6, 3, 1)
    x = torch.tensor([prompts_for(2)[2]])
    before = model(input_ids=x).logits
    PagedSwitchEngine(model, TOK, max_new_tokens=4, max_batch=2, max_seq_len=64)
    PagedSwitchEngine(model, TOK, max_new_tokens=4, max_batch=2, max_seq_len=64)  # idempotent
    after = model(input_ids=x).logits
    assert torch.equal(before, after)
    a = model.model.layers[0].self_attn
    assert a.k_proj.weight.data_ptr() == a.q_proj.weight.data_ptr() + a.q_proj.weight.numel() * 8


def test_slot_cache_surgery():
    from cotswitch.switch_model import SlotLayer

    a, b = SlotLayer(), SlotLayer()
    k = torch.arange(2 * 1 * 3 * 2, dtype=torch.float64).view(2, 1, 3, 2)
    a.update(k, -k)
    a.update(k[:, :, :1] + 100, -k[:, :, :1])
    assert a.get_seq_length() == 4 and a.keys.shape == (2, 1, 4, 2)
    assert torch.equal(a.keys[:, :, :3], k)
    a.pad_left(2)
    assert a.get_seq_length() == 6 and torch.equal(a.keys[:, :, 2:5], k)
    a.trim_left(1)
    assert a.get_seq_length() == 5
    b.update(k[:1, :, :1].repeat(1, 1, 5, 1), k[:1, :, :1].repeat(1, 1, 5, 1))
    a.cat_rows(b)
    assert a.keys.shape == (3, 1, 5, 2)
    a.select_rows(torch.tensor([2, 0]))
    assert a.keys.shape == (2, 1, 5, 2) and torch.equal(a.keys[0], b.keys[0])


# --- runner (CPU, tiny model, fake tokenizer) ---------------------------------


class FakeTok:
    """Char-level tokenizer over ids 0..57 plus the SWITCH specials of the tiny model."""

    SPECIAL = {S: "<swi>", E: "</swi>", EOS: "<|im_end|>", PAD: "<|endoftext|>", LAT: "<latent>"}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        assert not tokenize and add_generation_prompt and enable_thinking
        return "".join(f"[{m['role']}]{m['content']}" for m in msgs) + "[assistant]"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 58 for c in text]

    def decode(self, ids, skip_special_tokens=False):
        assert not skip_special_tokens
        return "".join(self.SPECIAL.get(i, chr(ord("a") + i % 26)) for i in ids)


def _units(n):
    from types import SimpleNamespace

    from cotswitch.io import make_key

    out = []
    for i in range(n):
        item = SimpleNamespace(dataset="gpqa", item_idx=i, answer_letter="A", keywords=["x"])
        spec = {"system": None, "user": "q" * (3 + 5 * i), "control_value": None, "grader_args": {}}
        out.append({"key": make_key("astra", "gpqa", i, "baseline", 0), "item": item,
                    "constraint": "baseline", "spec": spec})
    return out


def test_runner_writes_rows_and_resumes(tmp_path):
    pytest.importorskip("yaml")
    from cotswitch.io import read_jsonl
    from cotswitch.runners import switch_runner

    model = build(2, 6, 3, 1)
    cfg = {"name": "switch-latent", "engine": "switch", "model_id": "tiny", "model_revision": "r0",
           "adapter_id": "tiny-lora", "adapter_revision": "a0", "k_min": 2, "k_max": CAP,
           "max_new_tokens": 30, "enable_thinking": True, "seed": 0}
    units = _units(5)
    bundle = (model, FakeTok(), TOK)
    path = switch_runner.run(cfg, "astra", tmp_path, units=units[:3], model_bundle=bundle, batch_size=2,
                             engine="test", engine_kw=dict(block_size=4, sync_every=3, max_seq_len=200))
    rows = read_jsonl(path)
    assert len(rows) == 3
    path = switch_runner.run(cfg, "astra", tmp_path, units=units, model_bundle=bundle, batch_size=2,
                             engine="test", engine_kw=dict(block_size=4, sync_every=3, max_seq_len=200))
    rows = read_jsonl(path)
    assert sorted(r["key"] for r in rows) == sorted(u["key"] for u in units)
    tok = FakeTok()
    for r in rows:
        assert r["latent"] is not None and set(r["latent"]) >= {"n_blocks", "n_latent_steps", "blocks"}
        assert r["sampling"]["k_min"] == 2 and r["sampling"]["max_new_tokens"] == 30
        assert r["finish_reason"] in ("stop", "length")
        assert r["raw_output"].count("<swi>") == r["latent"]["n_blocks"]
        # the stored output matches a direct batch-1 run of the same prompt
        ids = tok.encode(r["rendered_prompt"])
        ref = generate_batch(model, TOK, [ids], engine="hf", max_new_tokens=30, min_latent_steps=2,
                             max_latent_per_block=CAP, max_batch=1)[0]
        assert tok.decode(ref["output_ids"]) == r["raw_output"]
        assert r["latent"] == ref["latent"]
    assert any(r["latent"]["n_blocks"] > 0 for r in rows)
    # every generation row was graded as it was written (vllm_runner.write_row)
    from cotswitch.io import grades_path

    grades = read_jsonl(grades_path(tmp_path, "switch-latent", "astra"))
    assert sorted(g["key"] for g in grades) == sorted(u["key"] for u in units)


def test_oom_recovery_matches(monkeypatch):
    """An OOM raised after the cache was written is undone: rows evicted, step retried."""
    model = build(2, 6, 3, 1)
    prompts = prompts_for(2, lengths=(3, 7, 12, 5, 9, 1))
    refs = reference(model, prompts, 40, 2, monkeypatch)
    real_forward = model.forward
    calls = {"n": 0, "raised": 0}

    def flaky(*args, **kw):
        out = real_forward(*args, **kw)
        calls["n"] += 1
        if kw["inputs_embeds"].shape[0] >= 4 and kw["inputs_embeds"].shape[1] == 1 and calls["n"] % 7 == 0:
            calls["raised"] += 1
            raise torch.OutOfMemoryError("simulated")
        return out

    monkeypatch.setattr(model, "forward", flaky)
    stats = GenStats()
    mine = generate_batch(model, TOK, prompts, engine="hf", max_new_tokens=40, min_latent_steps=2,
                          max_latent_per_block=CAP, max_batch=6, stats=stats)
    assert calls["raised"] > 0 and stats.evictions > 0
    check(refs, mine, 40)
