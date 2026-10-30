"""Unit tests for OPSD loss pieces. Run: .venv/bin/python -I -m pytest -q tests/ (from repo root with PYTHONPATH=.)"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd.kl import build_pair, causal_lm_parts, final_hidden, sequence_kl_backward, teacher_hidden, token_kl_from_logits  # noqa: E402
from opsd.model_utils import attach_lora, enable_grad_ckpt, trainable_params  # noqa: E402
from opsd.prompts import TEACHER_LINES, add_teacher_lines, check_system  # noqa: E402


def tiny_model(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=503, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    m = LlamaForCausalLM(cfg).float()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def peft_tiny(seed=0, perturb=False):
    m = attach_lora(tiny_model(seed), r=4, alpha=8)
    if perturb:
        g = torch.Generator().manual_seed(1)
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.data = torch.randn(p.shape, generator=g) * 0.5
    return m


def rand_pair(seed=0, ps=7, pt=19, r=23):
    g = torch.Generator().manual_seed(seed)
    s = torch.randint(0, 503, (ps,), generator=g).tolist()
    t = s[:3] + torch.randint(0, 503, (pt - 3,), generator=g).tolist()
    y = torch.randint(0, 503, (r,), generator=g).tolist()
    return s, t, y


def test_teacher_lines_format():
    s = "You are Llama Nemotron, an AI assistant. You've been deployed to chat.nvidiallama.com.\ndetailed thinking on"
    t = add_teacher_lines(s)
    assert t == ("You are Llama Nemotron, an AI assistant. You've been deployed to chat.nvidiallama.com.\n"
                 + TEACHER_LINES + "\ndetailed thinking on")
    check_system(t)
    assert add_teacher_lines(s, "") == s
    assert "Wood" not in TEACHER_LINES
    with pytest.raises(ValueError):
        add_teacher_lines("no toggle here")


def test_alignment_indices():
    s, t, y = rand_pair()
    p = build_pair(s, t, y)
    assert len(p.student_ids) - len(s) == len(y) == p.n_resp
    for j in range(len(y)):
        assert p.student_ids[p.s_start + 1 + j] == y[j]
        assert p.teacher_ids[p.t_start + 1 + j] == y[j]


def test_alignment_against_incremental_prefix():
    """logit at aligned position j == last logit of model(prefix + y[:j]) for both sequences."""
    m = tiny_model()
    s, t, y = rand_pair()
    p = build_pair(s, t, y)
    with torch.no_grad():
        Ls = m(torch.tensor([p.student_ids])).logits[0, p.s_start : p.s_start + p.n_resp]
        Lt = m(torch.tensor([p.teacher_ids])).logits[0, p.t_start : p.t_start + p.n_resp]
        for j in [0, 1, 5, len(y) - 1]:
            ref_s = m(torch.tensor([s + y[:j]])).logits[0, -1]
            ref_t = m(torch.tensor([t + y[:j]])).logits[0, -1]
            assert torch.allclose(Ls[j], ref_s, atol=1e-4), j
            assert torch.allclose(Lt[j], ref_t, atol=1e-4), j


@pytest.mark.parametrize("chunk", [1, 4, 7, 512])
def test_chunked_kl_matches_bruteforce_value_and_grad(chunk):
    s, t, y = rand_pair()
    p = build_pair(s, t, y)
    # brute force
    m1 = peft_tiny(perturb=True)
    with torch.no_grad(), m1.disable_adapter():
        Lt = m1(torch.tensor([p.teacher_ids])).logits[0, p.t_start : p.t_start + p.n_resp]
    Ls = m1(torch.tensor([p.student_ids])).logits[0, p.s_start : p.s_start + p.n_resp]
    kl_ref = token_kl_from_logits(Ls, Lt)
    # explicit formula check
    ps = torch.softmax(Ls.double(), -1)
    ref2 = (ps * (torch.log_softmax(Ls.double(), -1) - torch.log_softmax(Lt.double(), -1))).sum(-1)
    assert torch.allclose(kl_ref.double(), ref2, atol=1e-5)
    kl_ref.mean().backward()
    g_ref = {n: q.grad.clone() for n, q in m1.named_parameters() if q.requires_grad}
    # chunked implementation
    m2 = peft_tiny(perturb=True)
    enable_grad_ckpt(m2)
    m2.train()
    r = sequence_kl_backward(m2, p, "cpu", scale=1.0 / p.n_resp, chunk=chunk)
    assert abs(r["sum_kl"] - float(kl_ref.sum())) < 1e-4
    for n, q in m2.named_parameters():
        if q.requires_grad:
            assert torch.allclose(q.grad, g_ref[n], atol=1e-5, rtol=1e-4), n
    assert r["sum_kl"] > 0


def test_kl_exactly_zero_when_teacher_input_equals_student_and_fresh_lora():
    s, _, y = rand_pair()
    p = build_pair(s, s, y)  # empty teacher lines -> identical inputs
    m = peft_tiny(perturb=False)  # fresh LoRA: B == 0
    enable_grad_ckpt(m)
    m.train()
    r = sequence_kl_backward(m, p, "cpu", scale=1.0 / p.n_resp)
    assert r["sum_kl"] == 0.0
    # analytic gradient is 0 at p_s == p_t; numerically it is p * (1 - sum(p)) rounding noise
    gmax = max(float(q.grad.abs().max()) for q in trainable_params(m) if q.grad is not None)
    print("grad max at KL=0:", gmax)
    assert gmax < 1e-6


def test_disable_adapter_gives_base_model():
    s, t, y = rand_pair()
    p = build_pair(s, t, y)
    base = tiny_model()
    with torch.no_grad():
        h_base = base.model(torch.tensor([p.teacher_ids])).last_hidden_state[0, p.t_start : p.t_start + p.n_resp]
    m = peft_tiny(perturb=True)
    h_t = teacher_hidden(m, p, "cpu")
    assert torch.equal(h_t, h_base)
    _, dec, _ = causal_lm_parts(m)
    with torch.no_grad():
        h_on = final_hidden(dec, p.teacher_ids, "cpu")[p.t_start : p.t_start + p.n_resp]
    assert not torch.allclose(h_on, h_base, atol=1e-3)  # adapter active outside the context
