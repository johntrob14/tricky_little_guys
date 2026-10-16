"""Reverse-KL distillation loss for OPSD.

Per response token t (all sampled tokens: <think> reasoning, answer, final EOS):
    KL_t = sum_v p_s(v | student_prefix, y_<t) * (log p_s - log p_t)(v | ...)
where p_s = student (base + LoRA, grad) on the student prompt, and
p_t = teacher (same base, adapter disabled, no grad) on the teacher prompt
(student system prompt + teacher lines). Loss = mean_t KL_t over response tokens.

Split teacher (run C, SplitPair / build_split_pair): the response is cut after the last </think>. Reasoning
positions (through </think>) take p_t from teacher R (base, context = the student's own context or a clean copy of
the prompt), answer positions from teacher A (base, encourage system prompt + plain prompt), either with the same
reasoning in its context (sees_reasoning) or with an empty think block (nothink). Two no-grad forwards, sliced.

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


@dataclass
class SplitPair:
    """Split teacher (run C). Response tokens 0..n_reason-1 (through the last </think>, or all tokens when there is
    none) are taught by teacher R on sequence r_ids; tokens n_reason..n_resp-1 (the answer: "\\n\\nANSWER<|im_end|>")
    by teacher A on sequence a_ids.
      sees_reasoning: a_ids = A prompt + full response; a_start = index of the logit predicting answer token 0.
      nothink:        a_ids = A prompt rendered with thinking off, cut after its </think>, + answer tokens.
    s_start / r_start / a_start: logit index predicting response token 0 / response token 0 / answer token 0."""
    student_ids: list[int]
    r_ids: list[int]
    a_ids: list[int] | None
    s_start: int
    r_start: int
    a_start: int
    n_resp: int
    n_reason: int
    answer_mode: str

    @property
    def n_answer(self) -> int:
        return self.n_resp - self.n_reason


def split_response(response_ids: list[int], think_end_id: int) -> int:
    """n_reason = number of response tokens through the LAST </think> (rh_grade reads the answer after the last one);
    len(response_ids) when there is no </think>."""
    idx = None
    for i, t in enumerate(response_ids):
        if t == think_end_id:
            idx = i
    return len(response_ids) if idx is None else idx + 1


def cut_after_think_end(prompt_ids: list[int], think_end_id: int) -> list[int]:
    """A thinking-off chat prompt ends with '<think>\\n\\n</think>\\n\\n'; keep everything through </think> so the
    sample's own post-</think> tokens ('\\n\\n' + answer) follow it."""
    if think_end_id not in prompt_ids:
        raise ValueError("no </think> in the thinking-off prompt")
    i = len(prompt_ids) - 1 - prompt_ids[::-1].index(think_end_id)
    return list(prompt_ids[: i + 1])


def build_split_pair(student_prompt_ids: list[int], r_prompt_ids: list[int], a_prompt_ids: list[int],
                     response_ids: list[int], think_end_id: int, answer_mode: str = "sees_reasoning") -> SplitPair:
    """a_prompt_ids: for sees_reasoning the A chat prompt (thinking on, ends with '<think>\\n' for Qwen3.6); for nothink
    the A chat prompt rendered with enable_thinking=False (it is cut after its </think> here)."""
    if len(response_ids) == 0:
        raise ValueError("empty response")
    if answer_mode not in ("sees_reasoning", "nothink"):
        raise ValueError(answer_mode)
    response_ids = list(response_ids)
    n_reason = split_response(response_ids, think_end_id)
    n_answer = len(response_ids) - n_reason
    if answer_mode == "sees_reasoning":
        a_ids = list(a_prompt_ids) + response_ids
        a_start = len(a_prompt_ids) - 1 + n_reason
    else:
        prefix = cut_after_think_end(a_prompt_ids, think_end_id)
        a_ids = prefix + response_ids[n_reason:]
        a_start = len(prefix) - 1
    if n_answer == 0:
        a_ids = None
    return SplitPair(student_ids=list(student_prompt_ids) + response_ids, r_ids=list(r_prompt_ids) + response_ids,
                     a_ids=a_ids, s_start=len(student_prompt_ids) - 1, r_start=len(r_prompt_ids) - 1, a_start=a_start,
                     n_resp=len(response_ids), n_reason=n_reason, answer_mode=answer_mode)


def token_kl_from_logits(logits_s: torch.Tensor, logits_t: torch.Tensor, temp_t: torch.Tensor | None = None) -> torch.Tensor:
    """[n, V] x [n, V] -> [n] reverse KL(p_s || p_t), in fp32. temp_t [n]: per-position teacher temperature
    (teacher logits divided by it before the softmax; 1 = unchanged, <1 sharpens the teacher)."""
    lps = F.log_softmax(logits_s.float(), dim=-1)
    lt = logits_t.float()
    if temp_t is not None:
        lt = lt / temp_t.to(lt.device, lt.dtype)[:, None]
    lpt = F.log_softmax(lt, dim=-1)
    return (lps.exp() * (lps - lpt)).sum(-1)


def _chunk_kl(h_s: torch.Tensor, h_t: torch.Tensor, lm_head: torch.nn.Module, temp_t: torch.Tensor | None = None) -> torch.Tensor:
    return token_kl_from_logits(lm_head(h_s), lm_head(h_t), temp_t)


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


def _base_hidden(model, ids: list[int], device) -> torch.Tensor:
    """Final hidden states of the base model (adapter disabled), no grad."""
    _, dec, _ = causal_lm_parts(model)
    with torch.no_grad():
        ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else _null()
        with ctx:
            h = final_hidden(dec, ids, device)
    return h.detach()


def teacher_hidden(model, pair: Pair, device) -> torch.Tensor:
    h = _base_hidden(model, pair.teacher_ids, device)
    return h[pair.t_start : pair.t_start + pair.n_resp]


def split_teacher_hidden(model, pair: SplitPair, device) -> torch.Tensor:
    """Two full no-grad forwards (teacher R, teacher A), sliced and concatenated to [n_resp, d]: R for the first
    n_reason response positions, A for the n_answer answer positions."""
    h_r = _base_hidden(model, pair.r_ids, device)[pair.r_start : pair.r_start + pair.n_reason]
    if pair.n_answer == 0:
        return h_r
    h_a = _base_hidden(model, pair.a_ids, device)[pair.a_start : pair.a_start + pair.n_answer]
    return torch.cat([h_r, h_a], 0)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def sequence_kl_backward(model, pair, device, scale: float, chunk: int = 512, do_backward: bool = True,
                         answer_temp: float = 1.0, answer_weight: float = 1.0):
    """Teacher forward(s) (adapter off, no grad), student forward (adapter on, grad),
    chunked fp32 reverse KL summed over response tokens; backward of sum*scale.
    pair: Pair (one teacher) or SplitPair (teacher R for the reasoning positions, teacher A for the answer positions).
    answer_temp: teacher temperature on the answer positions of a SplitPair (reasoning positions stay at 1).
    Returns dict(sum_kl, n, max_kl) plus, for a SplitPair, sum_kl_reason / n_reason / sum_kl_answer / n_answer."""
    _, dec, head = causal_lm_parts(model)
    split = isinstance(pair, SplitPair)
    h_t = split_teacher_hidden(model, pair, device) if split else teacher_hidden(model, pair, device)
    h_s_full = final_hidden(dec, pair.student_ids, device)
    h_s = h_s_full[pair.s_start : pair.s_start + pair.n_resp]
    assert h_s.shape == h_t.shape, (h_s.shape, h_t.shape)
    temps = None
    if split and answer_temp != 1.0:
        temps = torch.ones(pair.n_resp)
        temps[pair.n_reason :] = answer_temp
    parts = []
    for i in range(0, pair.n_resp, chunk):
        hs, ht = h_s[i : i + chunk], h_t[i : i + chunk]
        tt = None if temps is None else temps[i : i + chunk]
        if torch.is_grad_enabled() and hs.requires_grad:
            kl = checkpoint(_chunk_kl, hs, ht, head, tt, use_reentrant=False)
        else:
            kl = _chunk_kl(hs, ht, head, tt)
        parts.append(kl)
    kl_all = torch.cat(parts)
    total = kl_all.sum()
    if split and answer_weight != 1.0:
        # answer_weight scales the answer positions' KL in the optimised objective only; the reported sums stay unweighted
        total = kl_all[: pair.n_reason].sum() + answer_weight * kl_all[pair.n_reason :].sum()
    if do_backward and total.requires_grad:
        (total * scale).backward()
    out = {"sum_kl": float(kl_all.detach().sum()), "n": pair.n_resp, "max_kl": float(kl_all.detach().max())}
    if split:
        k = kl_all.detach()
        out.update(sum_kl_reason=float(k[: pair.n_reason].sum()), n_reason=pair.n_reason,
                   sum_kl_answer=float(k[pair.n_reason :].sum()), n_answer=pair.n_answer,
                   max_kl_answer=(float(k[pair.n_reason :].max()) if pair.n_answer else 0.0))
    return out
