"""Reverse-KL distillation loss for OPSD.

Per response token t (all sampled tokens: <think> reasoning, answer, final EOS):
    KL_t = sum_v p_s(v | student_prefix, y_<t) * (log p_s - log p_t)(v | ...)
where p_s = student (base + LoRA, grad) on the student prompt, and
p_t = teacher (same base, adapter disabled, no grad) on the teacher prompt
(student system prompt + teacher lines). Loss = mean_t KL_t over response tokens.

The vocab projection + softmax is done in chunks of positions in fp32, each chunk
under activation checkpointing, so peak extra memory is O(chunk * vocab) instead of
O(seq_len * vocab).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class Pair:
    student_ids: list[int]
    teacher_ids: list[int]
    s_start: int  # index into student logits of the position predicting response token 0
    t_start: int
    n_resp: int


def build_pair(student_prompt_ids: list[int], teacher_prompt_ids: list[int], response_ids: list[int]) -> Pair:
    """Student and teacher sequences share response tokens; prefixes differ in length.
    Logit at position i predicts token i+1, so response token j (at sequence index P+j)
    is predicted by logit index P-1+j."""
    if len(response_ids) == 0:
        raise ValueError("empty response")
    return Pair(
        student_ids=list(student_prompt_ids) + list(response_ids),
        teacher_ids=list(teacher_prompt_ids) + list(response_ids),
        s_start=len(student_prompt_ids) - 1,
        t_start=len(teacher_prompt_ids) - 1,
        n_resp=len(response_ids),
    )


def token_kl_from_logits(logits_s: torch.Tensor, logits_t: torch.Tensor) -> torch.Tensor:
    """[n, V] x [n, V] -> [n] reverse KL(p_s || p_t), in fp32."""
    lps = F.log_softmax(logits_s.float(), dim=-1)
    lpt = F.log_softmax(logits_t.float(), dim=-1)
    return (lps.exp() * (lps - lpt)).sum(-1)


def _chunk_kl(h_s: torch.Tensor, h_t: torch.Tensor, lm_head: torch.nn.Module) -> torch.Tensor:
    return token_kl_from_logits(lm_head(h_s), lm_head(h_t))


def causal_lm_parts(model):
    """(causal_lm, decoder, lm_head) for a PeftModel or a plain HF CausalLM."""
    lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    dec = None
    if hasattr(lm, "get_decoder"):
        try:
            dec = lm.get_decoder()
        except Exception:
            dec = None
    if dec is None:
        dec = lm.model
    head = lm.get_output_embeddings()
    return lm, dec, head


def final_hidden(decoder, ids: list[int], device) -> torch.Tensor:
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = decoder(input_ids=x, use_cache=False)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return h[0]  # [L, d]


def teacher_hidden(model, pair: Pair, device) -> torch.Tensor:
    _, dec, _ = causal_lm_parts(model)
    with torch.no_grad():
        ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else _null()
        with ctx:
            h = final_hidden(dec, pair.teacher_ids, device)
    return h[pair.t_start : pair.t_start + pair.n_resp].detach()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def sequence_kl_backward(model, pair: Pair, device, scale: float, chunk: int = 512, do_backward: bool = True):
    """Teacher forward (adapter off, no grad), student forward (adapter on, grad),
    chunked fp32 reverse KL summed over response tokens; backward of sum*scale.
    Returns dict(sum_kl, n, max_kl)."""
    _, dec, head = causal_lm_parts(model)
    h_t = teacher_hidden(model, pair, device)
    h_s_full = final_hidden(dec, pair.student_ids, device)
    h_s = h_s_full[pair.s_start : pair.s_start + pair.n_resp]
    assert h_s.shape == h_t.shape, (h_s.shape, h_t.shape)
    parts = []
    for i in range(0, pair.n_resp, chunk):
        hs, ht = h_s[i : i + chunk], h_t[i : i + chunk]
        if torch.is_grad_enabled() and hs.requires_grad:
            kl = checkpoint(_chunk_kl, hs, ht, head, use_reentrant=False)
        else:
            kl = _chunk_kl(hs, ht, head)
        parts.append(kl)
    kl_all = torch.cat(parts)
    total = kl_all.sum()
    if do_backward and total.requires_grad:
        (total * scale).backward()
    return {"sum_kl": float(total.detach()), "n": pair.n_resp, "max_kl": float(kl_all.detach().max())}
