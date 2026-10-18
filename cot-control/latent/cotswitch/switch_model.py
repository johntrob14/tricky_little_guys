"""SWITCH model loading and a batched greedy latent-generation loop.

The loop reproduces, row by row, the batch-1 `CoconutSwiModel.generate()` of the
vendored SWITCH repo (third_party/SWITCH/src/model/coconut_swi_model.py, L364-543),
which is what the paper's `scripts/eval_latent.py` uses. Per row it is the same
state machine:

* text mode: take the pending argmax token. If the token budget is spent, stop
  ("length"). If it is EOS, append it (EOS does not consume budget, as in the repo)
  and stop ("stop"). Otherwise append it, spend one budget unit, and feed its
  embedding. If the token was `<swi>`, the row enters latent mode.
* latent mode: feed `outputs.hidden_states[-1][:, -1]` of the previous forward
  (the last-layer, post-norm hidden state; in transformers 5.x `hidden_states[-1]`
  is tied to `last_hidden_state`, i.e. after the final RMSNorm). After each latent
  step, if `n_steps >= min_latent_steps` and argmax == `</swi>`, exit ("natural");
  else if `n_steps == max_latent_per_block` (256), exit ("forced").
* on exit: append `</swi>` to the output ids and spend one budget unit even if the
  budget is already 0 (the repo does this), then feed the `</swi>` embedding next
  step and take the argmax of that forward as the pending token.

Consequences that match the repo and are worth knowing:
* Latent steps do NOT count toward `max_new_tokens`; `<swi>` and `</swi>` each
  count 1; EOS counts 0. The output can therefore hold max_new_tokens + 1 ids
  (when the budget runs out on `<swi>` the forced `</swi>` is still appended).
* Output ids contain `<swi></swi>` adjacent pairs; latent steps are not tokens.
* Only `tokenizer.eos_token_id` (`<|im_end|>` for Qwen3) stops generation.
* Latent block `token_pos` is the index of `<swi>` in the generated ids (the repo's
  `latent_info["position"]` is the index of `</swi>`, i.e. token_pos + 1).

Two implementations of this state machine live here: `PagedSwitchEngine` (default;
paged KV cache, state machine on the GPU, CUDA graphs; see the block comment above it)
and `generate_stream_hf` (the original loop, an HF forward per step), described next.

Batching (generate_stream_hf): rows are left-padded, every row has its own position counter and a 2-D
attention mask, and finished rows are removed from the batch immediately. New
prompts are admitted into free slots as rows finish (continuous batching): the
newcomers are prefilled together, then their KV cache is left-padded (or the
running cache is) so both have the same length and they are concatenated along
the batch dimension. Leading cache columns that no row attends to are trimmed.
The KV cache is a preallocated, growable buffer (`SlotLayer`) instead of
transformers' `DynamicLayer` (which does a torch.cat over the whole cache every
step). Results are mathematically identical per row to the batch-1 loop; in bf16
on GPU, batching changes kernel shapes and so can flip near-tied argmaxes
(measured in scripts/switch_parity.py).

Written against transformers 5.19 (the pod pin). The cache class relies on the
5.x `Cache(layers=...)` / `DynamicLayer` API.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicLayer

# --- model identity -----------------------------------------------------------

ADAPTER_ID = "LARK-Lab/SWITCH-Phase3-GRPO-LoRA-Qwen3-8B"
# HEAD of the adapter repo on 2026-10-14 (from the HF API). modules_to_save =
# [embed_tokens, lm_head], so resize_token_embeddings' random init of the 3 new rows
# is overwritten by the adapter.
ADAPTER_REVISION = "246fee75d774c02a110ea8608ac841a916dd5d35"
BASE_ID = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

SWI_START_TOKEN = "<swi>"
SWI_END_TOKEN = "</swi>"
LATENT_TOKEN = "<latent>"

MAX_LATENT_PER_BLOCK = 256  # repo constant
K_MIN_DEFAULT = 4  # SPEC: K_min = 4 latent steps per block


@dataclass(frozen=True)
class SwitchTokens:
    swi_start_id: int
    swi_end_id: int
    eos_token_id: int
    pad_token_id: int
    latent_id: int | None = None

    @classmethod
    def from_tokenizer(cls, tok) -> "SwitchTokens":
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        return cls(
            swi_start_id=tok.convert_tokens_to_ids(SWI_START_TOKEN),
            swi_end_id=tok.convert_tokens_to_ids(SWI_END_TOKEN),
            eos_token_id=tok.eos_token_id,  # as eval_latent.py: tokenizer.eos_token_id only
            pad_token_id=pad,
            latent_id=tok.convert_tokens_to_ids(LATENT_TOKEN),
        )


def load_switch(
    adapter_id: str = ADAPTER_ID,
    adapter_revision: str | None = ADAPTER_REVISION,
    base_id: str = BASE_ID,
    base_revision: str | None = BASE_REVISION,
    merged_path: str | None = None,
    merge: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load SWITCH the way scripts/eval_latent.py L246-256 does.

    Base Qwen3-8B (bf16) -> resize_token_embeddings(len(adapter tokenizer)) ->
    PeftModel.from_pretrained. `merge=True` calls merge_and_unload() afterwards;
    `merged_path` loads an already merged checkpoint instead (agent F's
    /root/scratch/models/switch-merged). The default is unmerged, which is what the
    paper's eval runs. Returns (model, tokenizer, SwitchTokens).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter_id, revision=adapter_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if merged_path:
        model = AutoModelForCausalLM.from_pretrained(merged_path, dtype=dtype, device_map={"": device})
    else:
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(
            base_id, revision=base_revision, dtype=dtype, device_map={"": device}
        )
        if len(tok) != model.config.vocab_size:
            model.resize_token_embeddings(len(tok))
        model = PeftModel.from_pretrained(model, adapter_id, revision=adapter_revision)
        if merge:
            model = model.merge_and_unload()
    model.eval()
    return model, tok, SwitchTokens.from_tokenizer(tok)


# --- growable KV cache with batch surgery --------------------------------------


class SlotLayer(DynamicLayer):
    """DynamicLayer backed by a preallocated buffer that grows in chunks.

    `self.keys` / `self.values` are views `buf[:, :, :n]`, so attention sees exactly
    what DynamicLayer would return, without a full-cache torch.cat per step.
    """

    GROW_MIN = 256

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._kb = self._vb = None
        self._n = 0

    def _set_views(self):
        self.keys = self._kb[:, :, : self._n]
        self.values = self._vb[:, :, : self._n]

    def _alloc(self, b, h, cap, d, like):
        return torch.zeros((b, h, cap, d), dtype=like.dtype, device=like.device)

    def _reserve(self, need: int):
        cap = self._kb.shape[2]
        if need <= cap:
            return
        new_cap = need + max(self.GROW_MIN, need // 8)
        b, h, _, d = self._kb.shape
        kb, vb = self._alloc(b, h, new_cap, d, self._kb), self._alloc(b, h, new_cap, d, self._vb)
        kb[:, :, : self._n] = self._kb[:, :, : self._n]
        vb[:, :, : self._n] = self._vb[:, :, : self._n]
        self._kb, self._vb = kb, vb

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        b, h, _, d = key_states.shape
        self._kb = self._alloc(b, h, 0, d, key_states)
        self._vb = self._alloc(b, h, 0, d, value_states)
        self._n = 0
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        q = key_states.shape[2]
        self._reserve(self._n + q)
        self._kb[:, :, self._n : self._n + q] = key_states
        self._vb[:, :, self._n : self._n + q] = value_states
        self._n += q
        self._set_views()
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self._n if self.is_initialized else 0

    def crop(self, tokens_to_remove: int) -> None:
        raise NotImplementedError

    # batch surgery (not part of the HF API)
    def select_rows(self, idx: torch.Tensor):
        self._kb = self._kb[idx]
        self._vb = self._vb[idx]
        self._set_views()

    def trim_left(self, s: int):
        if s <= 0:
            return
        self._kb = self._kb[:, :, s : self._n].contiguous()
        self._vb = self._vb[:, :, s : self._n].contiguous()
        self._n -= s
        self._set_views()

    def pad_left(self, p: int):
        if p <= 0:
            return
        b, h, _, d = self._kb.shape
        kb = self._alloc(b, h, self._n + p, d, self._kb)
        vb = self._alloc(b, h, self._n + p, d, self._vb)
        kb[:, :, p:] = self._kb[:, :, : self._n]
        vb[:, :, p:] = self._vb[:, :, : self._n]
        self._kb, self._vb = kb, vb
        self._n += p
        self._set_views()

    def cat_rows(self, other: "SlotLayer"):
        assert self._n == other._n, (self._n, other._n)
        self._kb = torch.cat([self._kb[:, :, : self._n], other._kb[:, :, : other._n]], dim=0)
        self._vb = torch.cat([self._vb[:, :, : self._n], other._vb[:, :, : other._n]], dim=0)
        self._set_views()


class SlotCache(Cache):
    def __init__(self, n_layers: int):
        super().__init__(layers=[SlotLayer() for _ in range(n_layers)])

    def select_rows(self, idx):
        for l in self.layers:
            l.select_rows(idx)

    def trim_left(self, s):
        for l in self.layers:
            l.trim_left(s)

    def pad_left(self, p):
        for l in self.layers:
            l.pad_left(p)

    def cat_rows(self, other: "SlotCache"):
        for a, b in zip(self.layers, other.layers):
            a.cat_rows(b)

    def capacity(self) -> int:
        return self.layers[0]._kb.shape[2] if self.layers[0].is_initialized else 0


# --- the batched loop ---------------------------------------------------------

TEXT, LATENT, FEED_END = 0, 1, 2


@dataclass
class _Row:
    uid: object
    prompt_len: int
    budget: int
    order: int  # admission counter (eviction picks the newest)
    pending: int = -1
    mode: int = TEXT
    steps: int = 0
    gen: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    n_forward: int = 0


def _result(row: _Row, finish_reason: str) -> dict:
    return {
        "output_ids": row.gen,
        "finish_reason": finish_reason,
        "n_prompt_tokens": row.prompt_len,
        "n_output_tokens": len(row.gen),
        "n_decode_forwards": row.n_forward,
        "latent": {
            "n_blocks": len(row.blocks),
            "n_latent_steps": sum(b["n_steps"] for b in row.blocks),
            "blocks": row.blocks,
        },
    }


def _n_layers(model) -> int:
    return model.config.get_text_config(decoder=True).num_hidden_layers


def kv_bytes_per_token(model) -> int:
    c = model.config.get_text_config(decoder=True)
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    n_kv = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
    return 2 * c.num_hidden_layers * n_kv * head_dim * torch.tensor([], dtype=model.dtype).element_size()


class GenStats:
    def __init__(self):
        self.forwards = 0
        self.row_steps = 0  # sum over forwards of active rows (text + latent feeds)
        self.latent_row_steps = 0
        self.prefill_tokens = 0
        self.evictions = 0
        self.deferred = 0  # paged engine: rows handed to the HF loop (max_seq_len overflow)
        self.swapped_positions = 0  # paged engine: generated positions preserved by swap preemption
        self.recomputed_positions = 0  # generated positions thrown away by restart preemption
        self.compactions = 0  # paged engine: rows moved to a lower slot
        self.rows_done = 0
        self.tokens_out = 0
        self.slot_steps = 0  # paged engine: decode steps x bucket size (incl. idle slots)
        self.live_slot_steps = 0  # paged engine: decode steps x occupied slots
        self.t0 = time.time()

    def as_dict(self):
        dt = time.time() - self.t0
        return {
            "forwards": self.forwards,
            "row_steps": self.row_steps,
            "latent_row_steps": self.latent_row_steps,
            "prefill_tokens": self.prefill_tokens,
            "evictions": self.evictions,
            "deferred": self.deferred,
            "swapped_positions": self.swapped_positions,
            "recomputed_positions": self.recomputed_positions,
            "compactions": self.compactions,
            "rows_done": self.rows_done,
            "tokens_out": self.tokens_out,
            "slot_steps": self.slot_steps,
            "live_slot_steps": self.live_slot_steps,
            "seconds": dt,
            "tokens_out_per_s": self.tokens_out / dt if dt > 0 else 0.0,
            "row_steps_per_s": self.row_steps / dt if dt > 0 else 0.0,
        }


@torch.no_grad()
def generate_stream_hf(
    model,
    tokens: SwitchTokens,
    prompts: Iterable[tuple[object, list[int]]],
    *,
    max_new_tokens: int,
    min_latent_steps: int = K_MIN_DEFAULT,
    max_latent_per_block: int = MAX_LATENT_PER_BLOCK,
    max_batch: int = 32,
    admit_chunk: int | None = None,
    kv_budget_tokens: int | None = None,
    device=None,
    stats: GenStats | None = None,
) -> Iterator[tuple[object, dict]]:
    """Greedy SWITCH latent generation with continuous batching.

    `prompts` yields (uid, prompt_token_ids); results are yielded as (uid, result)
    when each row finishes (not in input order). `kv_budget_tokens` caps
    batch_rows * cache_len (in token slots); rows admitted most recently are evicted
    and re-queued if the running cache would exceed it. `admit_chunk` is how many
    free slots must open before new prompts are admitted (default max_batch // 4,
    at least 1); prompts are always admitted when the batch is empty.
    """
    device = device or next(model.parameters()).device
    embed = model.get_input_embeddings()
    n_layers = _n_layers(model)
    admit_chunk = admit_chunk or max(1, max_batch // 4)
    stats = stats if stats is not None else GenStats()
    queue = collections.deque(prompts) if not isinstance(prompts, collections.deque) else prompts
    S, E, EOS, PAD = tokens.swi_start_id, tokens.swi_end_id, tokens.eos_token_id, tokens.pad_token_id

    rows: list[_Row] = []
    cache: SlotCache | None = None
    mask = None  # [B, T] long
    pos = None  # [B] long, next position id
    h = None  # [B, D] last hidden state of each row's previous forward
    order = 0
    prompt_of: dict = {}

    def keep(idx_list):
        nonlocal rows, mask, pos, h, cache
        idx = torch.tensor(idx_list, dtype=torch.long, device=device)
        rows = [rows[i] for i in idx_list]
        if not rows:
            cache = mask = pos = h = None
            return
        cache.select_rows(idx)
        mask, pos, h = mask[idx], pos[idx], h[idx]
        # trim leading columns nobody attends to
        first = int(mask.any(dim=0).long().argmax().item())
        if first > 0:
            cache.trim_left(first)
            mask = mask[:, first:]

    def admit(new):
        """Prefill a list of (uid, ids) and splice them into the running batch."""
        nonlocal rows, mask, pos, h, cache, order
        L = max(len(ids) for _, ids in new)
        ids_t = torch.full((len(new), L), PAD, dtype=torch.long, device=device)
        m = torch.zeros((len(new), L), dtype=torch.long, device=device)
        for i, (_, ids) in enumerate(new):
            ids_t[i, L - len(ids) :] = torch.tensor(ids, dtype=torch.long, device=device)
            m[i, L - len(ids) :] = 1
        p = (m.cumsum(-1) - 1).clamp(min=0)
        c = SlotCache(n_layers)
        out = model(
            inputs_embeds=embed(ids_t),
            attention_mask=m,
            position_ids=p,
            past_key_values=c,
            use_cache=True,
            output_hidden_states=False,
            logits_to_keep=1,
        )
        stats.forwards += 1
        stats.prefill_tokens += int(m.sum().item())
        first_tok = out.logits[:, -1].argmax(-1).tolist()
        new_rows = []
        for (uid, ids), t in zip(new, first_tok):
            r = _Row(uid=uid, prompt_len=len(ids), budget=max_new_tokens, order=order, pending=t)
            order += 1
            new_rows.append(r)
        new_pos = m.sum(-1)
        new_h = torch.zeros((len(new), embed.weight.shape[1]), dtype=embed.weight.dtype, device=device)
        if cache is None:
            rows, cache, mask, pos, h = new_rows, c, m, new_pos, new_h
            return
        T = mask.shape[1]
        if L < T:
            c.pad_left(T - L)
            m = torch.cat([torch.zeros((len(new), T - L), dtype=m.dtype, device=device), m], dim=1)
        elif L > T:
            cache.pad_left(L - T)
            mask = torch.cat([torch.zeros((len(rows), L - T), dtype=mask.dtype, device=device), mask], dim=1)
        cache.cat_rows(c)
        rows = rows + new_rows
        mask = torch.cat([mask, m], dim=0)
        pos = torch.cat([pos, new_pos], dim=0)
        h = torch.cat([h, new_h], dim=0)

    while queue or rows:
        # 1. admission
        free = max_batch - len(rows)
        if queue and free > 0 and (not rows or free >= min(admit_chunk, len(queue))):
            k = min(free, len(queue))
            T = mask.shape[1] if rows else 0
            if kv_budget_tokens is not None and rows:
                while k > 0:
                    Lnew = max(len(queue[i][1]) for i in range(k))
                    if (len(rows) + k) * (max(T, Lnew) + SlotLayer.GROW_MIN) <= kv_budget_tokens:
                        break
                    k -= 1
            if k > 0:
                new = [queue.popleft() for _ in range(k)]
                for uid, ids in new:
                    prompt_of[uid] = ids
                admit(new)

        # 2. eviction if the next step would exceed the KV budget
        if kv_budget_tokens is not None and len(rows) > 1:
            while len(rows) > 1 and len(rows) * (mask.shape[1] + 1) > kv_budget_tokens:
                newest = max(range(len(rows)), key=lambda i: rows[i].order)
                ev = rows[newest]
                queue.appendleft((ev.uid, prompt_of[ev.uid]))
                stats.evictions += 1
                keep([i for i in range(len(rows)) if i != newest])

        # 3. rows that finish without a forward (EOS pending, or budget spent)
        done = []
        for i, r in enumerate(rows):
            if r.mode == TEXT:
                if r.budget <= 0:
                    done.append((i, "length"))
                elif r.pending == EOS:
                    r.gen.append(EOS)  # EOS is appended but costs no budget (as in the repo)
                    done.append((i, "stop"))
        if done:
            for i, reason in done:
                r = rows[i]
                prompt_of.pop(r.uid, None)
                stats.rows_done += 1
                stats.tokens_out += len(r.gen)
                yield r.uid, _result(r, reason)
            dset = {i for i, _ in done}
            keep([i for i in range(len(rows)) if i not in dset])
        if not rows:
            continue

        # 4. one forward for every live row. Row state is only changed after it succeeds,
        #    so an OOM can be undone (evict the newest rows and retry the step).
        feed = [r.pending if r.mode == TEXT else (E if r.mode == FEED_END else PAD) for r in rows]
        is_lat = [r.mode == LATENT for r in rows]
        need_h = any(is_lat) or any(r.mode == TEXT and r.pending == S for r in rows)
        x = embed(torch.tensor(feed, dtype=torch.long, device=device))[:, None, :]
        if any(is_lat):
            lat = torch.tensor(is_lat, dtype=torch.bool, device=device)
            x = torch.where(lat[:, None, None], h[:, None, :].to(x.dtype), x)
        T_before = cache.get_seq_length()
        step_mask = torch.cat([mask, torch.ones((len(rows), 1), dtype=mask.dtype, device=device)], dim=1)
        try:
            out = model(
                inputs_embeds=x,
                attention_mask=step_mask,
                position_ids=pos[:, None],
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=need_h,
                logits_to_keep=1,
            )
            nxt = out.logits[:, -1].argmax(-1).tolist()
        except torch.OutOfMemoryError:
            out = None
            for layer in cache.layers:  # drop the half-written column
                if layer.is_initialized and layer._n > T_before:
                    layer._n = T_before
                    layer._set_views()
            if len(rows) == 1:
                raise
            n_ev = max(1, len(rows) // 4)
            newest = sorted(range(len(rows)), key=lambda i: rows[i].order)[-n_ev:]
            for i in newest:
                queue.appendleft((rows[i].uid, prompt_of[rows[i].uid]))
            stats.evictions += n_ev
            keep([i for i in range(len(rows)) if i not in set(newest)])
            kv_budget_tokens = int(0.9 * len(rows) * (mask.shape[1] + 1))
            torch.cuda.empty_cache()
            continue
        mask = step_mask
        pos = pos + 1
        stats.forwards += 1
        stats.row_steps += len(rows)
        stats.latent_row_steps += sum(is_lat)
        if need_h:
            h = out.hidden_states[-1][:, -1, :]
        del out

        # 5. commit the step
        for r, a in zip(rows, nxt):
            r.n_forward += 1
            if r.mode == TEXT:
                t = r.pending
                r.gen.append(t)
                r.budget -= 1
                if t == S:
                    r.mode, r.steps = LATENT, 0
                else:
                    r.pending = a
            elif r.mode == LATENT:
                r.steps += 1
                natural = r.steps >= min_latent_steps and a == E
                if natural or r.steps >= max_latent_per_block:
                    r.gen.append(E)  # repo appends </swi> even when the budget is spent
                    r.budget -= 1
                    r.blocks.append({"token_pos": len(r.gen) - 2, "n_steps": r.steps, "natural_exit": natural})
                    r.mode = FEED_END
            else:  # FEED_END: </swi> was just fed
                r.pending = a
                r.mode = TEXT





# --- fast engine: paged KV cache, GPU-resident state machine, CUDA graphs ------
#
# `PagedSwitchEngine` runs the same per-row state machine as `generate_stream_hf`
# (and the repo's batch-1 generate()), but:
# * the Qwen3 forward is reimplemented here on the HF module's own weights (q/k/v and
#   gate/up fused into single matmuls by re-pointing the HF parameters at views of a
#   fused buffer, so no extra memory), mirroring HF's op order and dtypes: RMSNorm in
#   fp32 then cast, weight * x in the model dtype; RoPE cos/sin from fp32 positions
#   cast to the model dtype; SiLU-gated MLP; final norm; lm_head; argmax.
# * the KV cache is paged (vLLM layout [num_blocks, block_size, kv_heads, head_dim]
#   per layer); decode attention is FlashAttention (vllm.vllm_flash_attn,
#   block_table + seqused_k) on GPU and a plain-torch reference on CPU.
# * B slots are fixed. All per-row state (mode, pending token, latent step count,
#   budget, generated ids, latent block info, position, last hidden state) lives in
#   GPU tensors and one decode step (state machine + forward + commit) is a pure
#   tensor program, captured once per batch-size bucket as a CUDA graph.
# * the host synchronises only every `sync_every` steps: it harvests rows that
#   finished (they idle, masked, until then), frees their blocks, prefills new prompts
#   into the free slots (continuous batching), tops up each row's blocks for the next
#   `sync_every` positions, and if the block pool runs out evicts the row with the
#   least generated work, then pauses admission until `admit_watermark` (15%) of the
#   pool is free again. A preempted row's KV blocks and its whole state-machine state
#   (mode, pending token, latent step count, budget, position, last hidden state,
#   generated ids, latent info) are copied to host memory and restored into new blocks
#   when it is re-admitted (`swap=True`, default), so no work is lost and the row
#   continues exactly where it stopped; `swap=False` restarts it from the prompt.
# * per-row positions are contiguous (0..prompt_len-1, then +1 per forward of that
#   row: text, latent or </swi>), exactly as in the batch-1 repo loop.
# * a row whose sequence would outgrow the block table (`max_seq_len`) is handed back
#   unchanged and regenerated at the end by `generate_stream_hf` (no length cap).

import math

import numpy as np


def _rms(x, w, eps):
    dt = x.dtype
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dt)


def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _flash_attn():
    """vLLM's FlashAttention varlen (paged via block_table), FA3 on Hopper, else FA2."""
    try:
        import functools

        from vllm.vllm_flash_attn import flash_attn_varlen_func
        from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported

        ver = 3 if is_fa_version_supported(3) else 2
        fn = functools.partial(flash_attn_varlen_func, fa_version=ver)
        fn.fa_version = ver
        return fn
    except Exception:  # noqa: BLE001
        return None


# Per-layer pieces of the decode forward. Plain functions (HF op order and dtypes); on
# GPU the engine wraps them in torch.compile so inductor fuses the elementwise work
# (norms, RoPE, SiLU*up, residual adds) around the cuBLAS matmuls. Only the decode
# step uses the compiled versions (fixed shapes per bucket); prefill runs them eagerly.


def _pre_attn(x, ln1, wqkv, qn, kn, cos, sin, eps: float, Hq: int, Hk: int, D: int):
    T = x.shape[0]
    q, k, v = F.linear(_rms(x, ln1, eps), wqkv).split([Hq * D, Hk * D, Hk * D], dim=-1)
    q = _rms(q.reshape(T, Hq, D), qn, eps)
    k = _rms(k.reshape(T, Hk, D), kn, eps)
    q = q * cos + _rot_half(q) * sin
    k = k * cos + _rot_half(k) * sin
    return q, k, v.reshape(T, Hk, D)


def _post_attn(r, o, wo, ln2, wgu, wd, eps: float, I: int):
    x = r + F.linear(o.reshape(o.shape[0], -1), wo)
    g, u = F.linear(_rms(x, ln2, eps), wgu).split([I, I], dim=-1)
    return x + F.linear(F.silu(g) * u, wd)


class _LayerW:
    __slots__ = ("ln1", "ln2", "qn", "kn", "wqkv", "wo", "wgu", "wd")


def _fuse(mods):
    """Concatenate the weights of `mods` along dim 0 and re-point each module's
    parameter at its view of the fused buffer (no second copy is kept). Idempotent."""
    fused = getattr(mods[0], "_cotswitch_fused", None)
    if fused is not None:
        return fused
    ws = [m.weight for m in mods]
    fused = torch.cat([w.data for w in ws], dim=0)
    o = 0
    for m, w in zip(mods, ws):
        m.weight.data = fused[o : o + w.shape[0]]
        o += w.shape[0]
    mods[0]._cotswitch_fused = fused
    return fused


class PagedSwitchEngine:
    """See the block comment above. Use `generate_stream(..., engine="paged")`."""

    def __init__(
        self,
        model,
        tokens: SwitchTokens,
        *,
        max_new_tokens: int,
        min_latent_steps: int = K_MIN_DEFAULT,
        max_latent_per_block: int = MAX_LATENT_PER_BLOCK,
        max_batch: int = 128,
        kv_budget_tokens: int | None = None,
        block_size: int = 64,
        sync_every: int = 16,
        max_seq_len: int | None = None,
        cuda_graphs: bool | None = None,
        attn: str = "auto",
        prefill_chunk_tokens: int = 16384,
        stats: "GenStats | None" = None,
        verbose: bool = False,
        log_every: float = 60.0,
        compile: bool | None = None,
        admit_watermark: float = 0.15,
        swap: bool = True,
        admit_chunk=None,  # accepted for call compatibility with generate_stream_hf; unused
    ):
        if hasattr(model, "peft_config"):
            raise ValueError("PagedSwitchEngine needs a plain (merged) Qwen3 model: load_switch(merged_path=...) "
                             "or load_switch(merge=True)")
        self.model = model
        self.tokens = tokens
        self.stats = stats if stats is not None else GenStats()
        self.verbose = verbose
        self.log_every = log_every
        self.admit_watermark = float(admit_watermark)
        self.swap = bool(swap)
        cfg = model.config.get_text_config(decoder=True)
        inner = model.model
        dev = next(model.parameters()).device
        self.device, self.dtype = dev, next(model.parameters()).dtype
        self.Hq, self.Hk = cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.I = cfg.intermediate_size
        self.eps = cfg.rms_norm_eps
        self.scale = self.D ** -0.5
        self.n_layers = cfg.num_hidden_layers
        self.inv_freq = inner.rotary_emb.inv_freq.to(device=dev, dtype=torch.float32)
        self.rope_scale = float(getattr(inner.rotary_emb, "attention_scaling", 1.0))
        rt = getattr(inner.rotary_emb, "rope_type", "default")
        if rt != "default":
            raise ValueError(f"only default RoPE is implemented, got {rt}")
        self.embed_w = inner.embed_tokens.weight
        self.lm_w = model.lm_head.weight
        self.norm_w = inner.norm.weight
        self.L = []
        for layer in inner.layers:
            a, m = layer.self_attn, layer.mlp
            if getattr(a, "sliding_window", None):
                raise ValueError("sliding-window attention is not implemented")
            lw = _LayerW()
            lw.ln1, lw.ln2 = layer.input_layernorm.weight, layer.post_attention_layernorm.weight
            lw.qn, lw.kn = a.q_norm.weight, a.k_norm.weight
            lw.wqkv = _fuse([a.q_proj, a.k_proj, a.v_proj])
            lw.wo = a.o_proj.weight
            lw.wgu = _fuse([m.gate_proj, m.up_proj])
            lw.wd = m.down_proj.weight
            for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, m.gate_proj, m.up_proj, m.down_proj):
                if lin.bias is not None:
                    raise ValueError("biases are not implemented (Qwen3 has none)")
            self.L.append(lw)

        self.max_new = int(max_new_tokens)
        self.kmin, self.kmax = int(min_latent_steps), int(max_latent_per_block)
        self.B = int(max_batch)
        self.bs = int(block_size)
        self.sync_every = int(sync_every)
        if max_seq_len is None:
            max_seq_len = 8192 + self.max_new + 2 + 16384  # prompt + budget + room for latent steps
        self.W = math.ceil((max_seq_len + self.sync_every) / self.bs)
        if kv_budget_tokens is None:
            kv_budget_tokens = self.B * self.W * self.bs
            if dev.type == "cuda":  # at most half the free memory by default
                free, _ = torch.cuda.mem_get_info(dev)
                kv_budget_tokens = min(kv_budget_tokens, int(free * 0.5 / kv_bytes_per_token(model)))
        self.NB = max(2, kv_budget_tokens // self.bs + 1)  # block 0 is the trash block
        self.prefill_chunk = int(prefill_chunk_tokens)
        is_cuda = dev.type == "cuda"
        self.fa = _flash_attn() if (attn in ("auto", "flash") and is_cuda) else None
        if attn == "flash" and self.fa is None:
            raise RuntimeError("flash attention requested but vllm.vllm_flash_attn is unavailable")
        # the torch attention fallback reads lengths on the host, so graphs need flash attention
        self.use_graphs = (is_cuda if cuda_graphs is None else bool(cuda_graphs)) and self.fa is not None
        self._pre_c = self._post_c = None
        if compile is None:
            compile = is_cuda
        if compile:
            from torch import _dynamo

            _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 64)
            self._pre_c = torch.compile(_pre_attn, fullgraph=True, dynamic=False)
            self._post_c = torch.compile(_post_attn, fullgraph=True, dynamic=False)

        z = dict(device=dev)
        self.kv = torch.zeros((self.n_layers, 2, self.NB, self.bs, self.Hk, self.D), dtype=self.dtype, **z)
        B, G = self.B, self.max_new + 1  # at most max_new + 1 output ids; column G is the trash column
        self.G = G
        self.active = torch.zeros(B, dtype=torch.bool, **z)
        self.mode = torch.zeros(B, dtype=torch.long, **z)
        self.pending = torch.zeros(B, dtype=torch.long, **z)
        self.steps = torch.zeros(B, dtype=torch.long, **z)
        self.budget = torch.zeros(B, dtype=torch.long, **z)
        self.ngen = torch.zeros(B, dtype=torch.long, **z)
        self.pos = torch.zeros(B, dtype=torch.long, **z)
        self.finish = torch.zeros(B, dtype=torch.long, **z)  # 0 running, 1 stop, 2 length
        self.h = torch.zeros((B, cfg.hidden_size), dtype=self.dtype, **z)
        self.gen = torch.zeros((B, G + 1), dtype=torch.int32, **z)
        self.linfo = torch.zeros((B, G + 1), dtype=torch.int32, **z)  # at </swi>: 2*n_steps + natural
        self.block_table = torch.zeros((B, self.W), dtype=torch.int32, **z)
        self.cu_q = torch.arange(B + 1, dtype=torch.int32, **z)
        self.buckets = sorted({min(B, 2 ** i) for i in range(0, 1 + math.ceil(math.log2(max(B, 1))))})
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        if self.use_graphs:
            t0 = time.time()
            self._capture_all()
            self._log(f"captured CUDA graphs for buckets {self.buckets} in {time.time() - t0:.1f}s")
        self._log(f"engine: B={self.B} blocks={self.NB - 1}x{self.bs} table_width={self.W} sync_every="
                  f"{self.sync_every} attn={'fa%d' % self.fa.fa_version if self.fa else 'torch'} "
                  f"graphs={self.use_graphs} compiled={self._pre_c is not None}")

    def buckets_for(self, n):
        return next(b for b in self.buckets if b >= n)

    def _log(self, msg):
        if self.verbose:
            print(f"[switch_model] {msg}", flush=True)

    # -- model ------------------------------------------------------------------

    def _rope(self, pos, dt):
        f = pos.to(torch.float32)[:, None] * self.inv_freq[None, :]
        emb = torch.cat((f, f), dim=-1)
        cos = (emb.cos() * self.rope_scale).to(dt)[:, None, :]
        sin = (emb.sin() * self.rope_scale).to(dt)[:, None, :]
        return cos, sin

    def _layers(self, x, pos, slot, attend, compiled=False):
        Hq, Hk, D = self.Hq, self.Hk, self.D
        pre, post = (self._pre_c, self._post_c) if compiled and self._pre_c is not None else (_pre_attn, _post_attn)
        cos, sin = self._rope(pos, x.dtype)
        for li, lw in enumerate(self.L):
            q, k, v = pre(x, lw.ln1, lw.wqkv, lw.qn, lw.kn, cos, sin, self.eps, Hq, Hk, D)
            self.kv[li, 0].view(-1, Hk, D).index_copy_(0, slot, k)
            self.kv[li, 1].view(-1, Hk, D).index_copy_(0, slot, v)
            o = attend(li, q, k, v)
            x = post(x, o, lw.wo, lw.ln2, lw.wgu, lw.wd, self.eps, self.I)
        return _rms(x, self.norm_w, self.eps)

    def _sdpa_rows(self, q, k, v, causal):
        # q [Tq, Hq, D], k/v [Tk, Hk, D] for one sequence; HF eager semantics
        g = self.Hq // self.Hk
        kk = k.repeat_interleave(g, dim=1).transpose(0, 1)  # [Hq, Tk, D]
        vv = v.repeat_interleave(g, dim=1).transpose(0, 1)
        s = torch.matmul(q.transpose(0, 1), kk.transpose(1, 2)) * self.scale  # [Hq, Tq, Tk]
        if causal:
            Tq, Tk = q.shape[0], k.shape[0]
            m = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(Tk - Tq)
            s = s.masked_fill(~m, float("-inf"))
        sm_dt = torch.float32 if q.dtype in (torch.bfloat16, torch.float16) else q.dtype
        p = torch.softmax(s, dim=-1, dtype=sm_dt).to(q.dtype)
        return torch.matmul(p, vv).transpose(0, 1)  # [Tq, Hq, D]

    def _decode_attend(self, n, seqlen, wmax):
        bt = self.block_table[:n]
        if self.fa is not None:
            def attend(li, q, k, v):
                return self.fa(q, self.kv[li, 0], self.kv[li, 1], max_seqlen_q=1, cu_seqlens_q=self.cu_q[: n + 1],
                               max_seqlen_k=self.W * self.bs, seqused_k=seqlen, causal=True,
                               softmax_scale=self.scale, block_table=bt)
            return attend
        wmax = wmax or self.W
        idx = bt[:, :wmax].long()
        T = wmax * self.bs
        valid = torch.arange(T, device=seqlen.device)[None, :] < seqlen[:, None]  # [n, T]
        g = self.Hq // self.Hk
        sm_dt = torch.float32 if self.dtype in (torch.bfloat16, torch.float16) else self.dtype

        def attend(li, q, k, v):  # plain-torch paged attention, HF eager semantics
            kc = self.kv[li, 0][idx].reshape(n, T, self.Hk, self.D).repeat_interleave(g, dim=2)
            vc = self.kv[li, 1][idx].reshape(n, T, self.Hk, self.D).repeat_interleave(g, dim=2)
            sc = torch.einsum("nhd,nthd->nht", q, kc) * self.scale
            sc = sc.masked_fill(~valid[:, None, :], float("-inf"))
            p = torch.softmax(sc, dim=-1, dtype=sm_dt).to(q.dtype)
            return torch.einsum("nht,nthd->nhd", p, vc)
        return attend

    def _prefill_attend(self, cu, lmax):
        if self.fa is not None:
            cu_t = torch.tensor(cu, dtype=torch.int32, device=self.device)

            def attend(li, q, k, v):
                return self.fa(q, k, v, max_seqlen_q=lmax, cu_seqlens_q=cu_t, max_seqlen_k=lmax, cu_seqlens_k=cu_t,
                               causal=True, softmax_scale=self.scale)
            return attend

        def attend(li, q, k, v):
            return torch.cat([self._sdpa_rows(q[a:b], k[a:b], v[a:b], True) for a, b in zip(cu[:-1], cu[1:])], dim=0)
        return attend

    # -- one decode step (pure tensor program; captured as a CUDA graph) ---------

    def _step(self, n, wmax=None):
        S, E, EOS = self.tokens.swi_start_id, self.tokens.swi_end_id, self.tokens.eos_token_id
        act, mode, pend = self.active[:n], self.mode[:n], self.pending[:n]
        steps, bud, ngen = self.steps[:n], self.budget[:n], self.ngen[:n]
        pos, fin, h = self.pos[:n], self.finish[:n], self.h[:n]

        is_text = act & (mode == TEXT)
        fin_len = is_text & (bud <= 0)
        fin_stop = is_text & ~fin_len & (pend == EOS)
        run = act & ~(fin_len | fin_stop)
        lat = run & (mode == LATENT)
        feed = torch.where(mode == FEED_END, E, pend)
        x = F.embedding(feed, self.embed_w)
        x = torch.where(lat[:, None], h, x)
        blk = self.block_table[:n].gather(1, (pos // self.bs).clamp(max=self.W - 1)[:, None]).squeeze(1)
        slot = torch.where(run, blk.long() * self.bs + pos % self.bs, 0)
        seqlen = torch.where(run, pos + 1, 1).to(torch.int32)
        hn = self._layers(x, pos, slot, self._decode_attend(n, seqlen, wmax), compiled=True)
        a = F.linear(hn, self.lm_w).argmax(-1)

        t_text = run & (mode == TEXT)
        enter = t_text & (pend == S)
        t_fe = run & (mode == FEED_END)
        steps_new = steps + lat.long()
        nat = lat & (steps_new >= self.kmin) & (a == E)
        ex = lat & (nat | (steps_new >= self.kmax))
        app = fin_stop | t_text | ex
        val = torch.where(fin_stop, EOS, torch.where(t_text, pend, E)).to(torch.int32)
        info = torch.where(ex, 2 * steps_new + nat.long(), 0).to(torch.int32)
        col = torch.where(app, ngen, self.G)[:, None]
        new_mode = torch.where(enter, LATENT, torch.where(ex, FEED_END, torch.where(t_fe, TEXT, mode)))
        new_pend = torch.where((t_text & ~enter) | t_fe, a, pend)
        new_fin = torch.where(fin_len, 2, torch.where(fin_stop, 1, fin))

        self.gen[:n].scatter_(1, col, val[:, None])
        self.linfo[:n].scatter_(1, col, info[:, None])
        ngen.add_(app.long())
        bud.sub_((t_text | ex).long())
        steps.copy_(torch.where(enter, 0, steps_new))
        mode.copy_(new_mode)
        pend.copy_(new_pend)
        fin.copy_(new_fin)
        act.copy_(run)
        pos.add_(run.long())
        h.copy_(torch.where(run[:, None], hn, h))

    def _capture_all(self):
        # all slots are inactive here, so warm-up steps only touch the trash block/column
        self.pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for n in self.buckets[::-1]:
                self._step(n)
                self._step(n)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        for n in self.buckets[::-1]:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._step(n)
            self.graphs[n] = g
        torch.cuda.synchronize()

    # -- prefill ----------------------------------------------------------------

    def _prefill(self, batch, bt_host):
        """batch: list of (slot, ids). Writes KV, returns first argmax per row (GPU)."""
        out = []
        i = 0
        while i < len(batch):
            j, tot = i, 0
            while j < len(batch) and (j == i or tot + len(batch[j][1]) <= self.prefill_chunk):
                tot += len(batch[j][1])
                j += 1
            chunk = batch[i:j]
            lens = [len(ids) for _, ids in chunk]
            cu = [0] + list(np.cumsum(lens).tolist())
            ids = torch.tensor([t for _, x in chunk for t in x], dtype=torch.long, device=self.device)
            pos_np = np.concatenate([np.arange(L) for L in lens])
            slot_np = np.concatenate([bt_host[s, np.arange(L) // self.bs].astype(np.int64) * self.bs
                                      + np.arange(L) % self.bs for (s, _), L in zip(chunk, lens)])
            pos = torch.from_numpy(pos_np).to(self.device)
            slot = torch.from_numpy(slot_np).to(self.device)
            hn = self._layers(F.embedding(ids, self.embed_w), pos, slot, self._prefill_attend(cu, max(lens)))
            last = torch.tensor([c - 1 for c in cu[1:]], dtype=torch.long, device=self.device)
            out.append(F.linear(hn[last], self.lm_w).argmax(-1))
            self.stats.prefill_tokens += tot
            self.stats.forwards += 1
            i = j
        return torch.cat(out)

    # -- host loop --------------------------------------------------------------

    def _result(self, prompt_len, ids, info, finish_code, n_fwd):
        blocks = [{"token_pos": i - 1, "n_steps": int(c) // 2, "natural_exit": bool(int(c) % 2)}
                  for i, c in enumerate(info) if c > 0]
        return {
            "output_ids": ids,
            "finish_reason": "stop" if finish_code == 1 else "length",
            "n_prompt_tokens": prompt_len,
            "n_output_tokens": len(ids),
            "n_decode_forwards": n_fwd,
            "latent": {"n_blocks": len(blocks), "n_latent_steps": sum(b["n_steps"] for b in blocks),
                       "blocks": blocks},
        }

    @torch.no_grad()
    def run(self, prompts) -> Iterator[tuple[object, dict]]:
        st = self.stats
        queue = prompts if isinstance(prompts, collections.deque) else collections.deque(prompts)
        slots: list = [None] * self.B  # [uid, ids, order, blocks]
        free_blocks = list(range(self.NB - 1, 0, -1))
        bt_host = np.zeros((self.B, self.W), dtype=np.int32)
        pos_host = np.zeros(self.B, dtype=np.int64)
        deferred = []
        order = 0
        cap = self.W * self.bs
        per_row_reserve = math.ceil(self.sync_every / self.bs) + 1
        dev = self.device
        t_log = 0.0
        admit_paused = False

        def release(s):
            free_blocks.extend(slots[s][3])
            slots[s] = None
            bt_host[s] = 0

        def swap_out(s):
            """Copy row s's KV blocks and full state-machine state to host memory."""
            p = int(pos_host[s])  # exact: evictions run right after a harvest, before any step
            nb = math.ceil(p / self.bs)
            bidx = torch.tensor(slots[s][3][:nb], dtype=torch.long, device=dev)
            kv = torch.empty((self.n_layers, 2, nb, self.bs, self.Hk, self.D), dtype=self.dtype)
            for li in range(self.n_layers):
                kv[li].copy_(self.kv[li][:, bidx])
            sc = torch.stack([self.mode[s], self.pending[s], self.steps[s], self.budget[s], self.ngen[s],
                              self.pos[s]]).cpu()
            ng = int(sc[4])
            assert int(sc[5]) == p, (int(sc[5]), p)
            return {"kv": kv, "sc": sc, "h": self.h[s].cpu(), "gen": self.gen[s, :ng].cpu(),
                    "linfo": self.linfo[s, :ng].cpu(), "pos": p}

        def swap_in(s, sv):
            nb = sv["kv"].shape[2]
            bidx = torch.tensor(slots[s][3][:nb], dtype=torch.long, device=dev)
            for li in range(self.n_layers):
                self.kv[li][:, bidx] = sv["kv"][li].to(dev)
            mode, pend, steps, bud, ng, pos = (int(x) for x in sv["sc"])
            self.mode[s], self.pending[s], self.steps[s], self.budget[s] = mode, pend, steps, bud
            self.ngen[s], self.pos[s] = ng, pos
            self.h[s] = sv["h"].to(dev)
            self.gen[s, :ng] = sv["gen"].to(dev)
            self.linfo[s, :ng] = sv["linfo"].to(dev)
            self.finish[s] = 0
            self.active[s] = True

        while True:
            occ = [s for s in range(self.B) if slots[s] is not None]
            # 1. harvest (the only host sync)
            if occ:
                hi = occ[-1] + 1
                snap = torch.stack([self.finish[:hi], self.ngen[:hi], self.pos[:hi]]).cpu().numpy()
                done = [s for s in occ if snap[0, s] != 0]
                if done:
                    L = int(max(snap[1, s] for s in done))
                    idx = torch.tensor(done, dtype=torch.long, device=dev)
                    g = self.gen[idx, :L].cpu().tolist()
                    li = self.linfo[idx, :L].cpu().tolist()
                    for k, s in enumerate(done):
                        n = int(snap[1, s])
                        uid, ids = slots[s][0], slots[s][1]
                        res = self._result(len(ids), g[k][:n], li[k][:n], int(snap[0, s]),
                                           int(snap[2, s]) - len(ids))
                        st.row_steps += res["n_decode_forwards"]
                        st.latent_row_steps += res["latent"]["n_latent_steps"]
                        st.rows_done += 1
                        st.tokens_out += n
                        release(s)
                        yield uid, res
                for s in occ:
                    if slots[s] is not None:
                        pos_host[s] = snap[2, s]
            occ = [s for s in range(self.B) if slots[s] is not None]
            if not occ and not queue:
                break

            # 1b. compaction: move rows from high slots into free low slots so the batch
            #     bucket (decided by the highest occupied slot) tracks the live-row count.
            #     KV stays where it is (rows reach it through their block-table row).
            if occ and self.buckets_for(occ[-1] + 1) > self.buckets_for(len(occ)):
                free_lo = [t for t in range(len(occ)) if slots[t] is None]
                src = [t for t in reversed(occ) if t >= len(occ)][: len(free_lo)]
                dst = free_lo[: len(src)]
                if src:
                    i_s = torch.tensor(src, dtype=torch.long, device=dev)
                    i_d = torch.tensor(dst, dtype=torch.long, device=dev)
                    for T in (self.active, self.mode, self.pending, self.steps, self.budget, self.ngen, self.pos,
                              self.finish, self.h, self.gen, self.linfo):
                        T[i_d] = T[i_s]
                    self.active[i_s] = False
                    for a_, b_ in zip(src, dst):
                        slots[b_], slots[a_] = slots[a_], None
                        bt_host[b_] = bt_host[a_]
                        bt_host[a_] = 0
                        pos_host[b_] = pos_host[a_]
                    self.block_table.copy_(torch.from_numpy(bt_host), non_blocking=False)
                    st.compactions += len(src)
                    occ = [t for t in range(self.B) if slots[t] is not None]

            deact = []
            bt_dirty = False
            ev0 = st.evictions
            # 2. growth for running rows (oldest first); evict the newest when out of blocks
            for s in sorted(occ, key=lambda s: slots[s][2]):
                if slots[s] is None:
                    continue
                if pos_host[s] + self.sync_every > cap:
                    deferred.append((slots[s][0], slots[s][1]))
                    st.deferred += 1
                    release(s)
                    deact.append(s)
                    continue
                need = math.ceil((pos_host[s] + self.sync_every) / self.bs) - len(slots[s][3])
                while need > len(free_blocks):
                    live = [t for t in range(self.B) if slots[t] is not None]
                    if live == [s]:  # the whole pool cannot hold this one row: regenerate it unpaged
                        print(f"[switch_model] KV pool ({self.NB - 1} blocks of {self.bs}) cannot hold row "
                              f"{slots[s][0]!r} at {pos_host[s] + self.sync_every} positions; deferring it to the "
                              f"unpaged loop", flush=True)
                        deferred.append((slots[s][0], slots[s][1]))
                        st.deferred += 1
                        release(s)
                        deact.append(s)
                        break
                    # victim: least generated work (smallest swap; ties: most recently admitted)
                    newest = min(live, key=lambda t: (pos_host[t] - len(slots[t][1]), -slots[t][2]))
                    work = int(pos_host[newest]) - len(slots[newest][1])
                    if self.swap and work > 0:
                        queue.appendleft((slots[newest][0], slots[newest][1], swap_out(newest)))
                        st.swapped_positions += work
                        how = "swapped to host memory, resumes later"
                    else:
                        queue.appendleft((slots[newest][0], slots[newest][1]))
                        st.recomputed_positions += work
                        how = "restarts from scratch"
                    print(f"[switch_model] KV pool full: preempting row {slots[newest][0]!r} after {work} "
                          f"generated positions ({how}); preemptions so far {st.evictions + 1}", flush=True)
                    st.evictions += 1
                    release(newest)
                    deact.append(newest)
                    if newest == s:
                        break
                if slots[s] is None:
                    continue
                for _ in range(max(0, need)):
                    b = free_blocks.pop()
                    bt_host[s, len(slots[s][3])] = b
                    slots[s][3].append(b)
                    bt_dirty = True

            # 3. admission into free slots. After an eviction, admission pauses until
            #    `admit_watermark` of the pool is free again, so the live-row count settles at
            #    what the pool can hold instead of thrashing (admit, grow, evict, re-admit).
            if st.evictions > ev0:
                admit_paused = True
            elif admit_paused and len(free_blocks) >= self.admit_watermark * (self.NB - 1):
                admit_paused = False
            evicted_now = admit_paused
            if deact:
                self.active[torch.tensor(deact, dtype=torch.long, device=dev)] = False
            n_live = sum(1 for s in range(self.B) if slots[s] is not None)
            new, restored = [], []
            free_slots = collections.deque(s for s in range(self.B) if slots[s] is None)
            while queue and free_slots and not evicted_now:
                s = free_slots[0]
                uid, ids = queue[0][0], queue[0][1]
                sv = queue[0][2] if len(queue[0]) > 2 else None
                plen = sv["pos"] if sv is not None else len(ids)
                if plen + self.sync_every > cap:
                    queue.popleft()
                    deferred.append((uid, ids))
                    st.deferred += 1
                    continue
                need = math.ceil((plen + self.sync_every) / self.bs)
                if need + (n_live + 1) * per_row_reserve > len(free_blocks) and n_live > 0:
                    break
                if need > self.NB - 1:  # can never fit: regenerate unpaged at the end
                    queue.popleft()
                    deferred.append((uid, ids))
                    st.deferred += 1
                    continue
                if need > len(free_blocks):
                    break
                queue.popleft()
                free_slots.popleft()
                blocks = [free_blocks.pop() for _ in range(need)]
                bt_host[s, :need] = blocks
                slots[s] = [uid, ids, order, blocks]
                order += 1
                pos_host[s] = plen
                n_live += 1
                if sv is not None:
                    restored.append((s, sv))
                else:
                    new.append((s, ids))
                bt_dirty = True
            if bt_dirty:
                self.block_table.copy_(torch.from_numpy(bt_host), non_blocking=False)
            for s, sv in restored:
                swap_in(s, sv)
            if new:
                first = self._prefill(new, bt_host)
                idx = torch.tensor([s for s, _ in new], dtype=torch.long, device=dev)
                self.pending[idx] = first
                self.mode[idx] = TEXT
                self.steps[idx] = 0
                self.budget[idx] = self.max_new
                self.ngen[idx] = 0
                self.pos[idx] = torch.tensor([len(x) for _, x in new], dtype=torch.long, device=dev)
                self.finish[idx] = 0
                self.active[idx] = True
            occ = [s for s in range(self.B) if slots[s] is not None]
            if not occ:
                continue

            if self.verbose and time.time() - t_log > self.log_every:
                t_log = time.time()
                self._log(f"live={len(occ)} queue={len(queue)} free_blocks={len(free_blocks)} done={st.rows_done} "
                          f"tokens_out={st.tokens_out} forwards={st.forwards} evictions={st.evictions} "
                          f"tok/s={st.tokens_out / max(1e-9, time.time() - st.t0):.0f}")

            # 4. sync_every decode steps without touching the host
            hi = occ[-1] + 1
            n = next(b for b in self.buckets if b >= hi)
            g = self.graphs.get(n)
            wmax = None
            if g is None:
                wmax = min(self.W, math.ceil((int(pos_host[:hi].max()) + self.sync_every) / self.bs))
            for _ in range(self.sync_every):
                if g is not None:
                    g.replay()
                else:
                    self._step(n, wmax)
            st.forwards += self.sync_every
            st.slot_steps += n * self.sync_every
            st.live_slot_steps += len(occ) * self.sync_every

        if deferred:
            print(f"[switch_model] {len(deferred)} rows exceeded max_seq_len={cap}; regenerating them with the "
                  f"unpaged HF loop", flush=True)
            self.release_memory()
            yield from generate_stream_hf(
                self.model, self.tokens, deferred, max_new_tokens=self.max_new, min_latent_steps=self.kmin,
                max_latent_per_block=self.kmax, max_batch=1, stats=None)

    def release_memory(self):
        self.graphs.clear()
        self.kv = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def generate_stream(model, tokens: SwitchTokens, prompts, *, max_new_tokens: int, engine: str = "auto",
                    stats: "GenStats | None" = None, **kw) -> Iterator[tuple[object, dict]]:
    """Greedy SWITCH latent generation. `engine="paged"` is PagedSwitchEngine; `engine="hf"`
    is the original HF-forward loop (`generate_stream_hf`); "auto" (default) picks paged
    unless the model is an unmerged PeftModel. Results are yielded as rows finish."""
    if engine == "auto":
        engine = "hf" if hasattr(model, "peft_config") else "paged"
    if engine == "hf":
        yield from generate_stream_hf(model, tokens, prompts, max_new_tokens=max_new_tokens, stats=stats, **kw)
        return
    eng = PagedSwitchEngine(model, tokens, max_new_tokens=max_new_tokens, stats=stats, **kw)
    try:
        yield from eng.run(prompts)
    finally:
        eng.release_memory()


def generate_batch(model, tokens: SwitchTokens, prompt_ids: list[list[int]], **kw) -> list[dict]:
    """Run `generate_stream` over a list of prompts; return results in input order."""
    res = dict(generate_stream(model, tokens, list(enumerate(prompt_ids)), **kw))
    return [res[i] for i in range(len(prompt_ids))]
