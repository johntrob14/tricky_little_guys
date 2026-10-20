"""Activation Difference Lens (Minder et al. 2025) pieces for gpt-oss-120b, following the
diffing-toolkit defaults: science-of-finetuning/fineweb-1m-sample, 10,000 texts, first 5 token
positions, mean residual difference (organism - base) at the middle layer."""
from __future__ import annotations

import torch
from datasets import load_dataset

from hfmodel import layers, out_tensor

N_POS = 5


def fineweb_texts(n: int) -> list[str]:
    ds = load_dataset("science-of-finetuning/fineweb-1m-sample", split="train", streaming=True)
    out = []
    for r in ds:
        out.append(r["text"])
        if len(out) == n:
            break
    return out


@torch.no_grad()
def position_means(model, tok, texts: list[str], batch: int = 500) -> dict:
    """Sum of residual outputs of every decoder layer at positions 0..N_POS-1 (fp32, CPU)."""
    ids = [tok(t, add_special_tokens=False).input_ids[:N_POS] for t in texts]
    ids = [x for x in ids if len(x) == N_POS]
    L = layers(model)
    sums = torch.zeros(len(L), N_POS, model.config.hidden_size, dtype=torch.float64)
    sq = torch.zeros(len(L), N_POS, dtype=torch.float64)
    cur = {}

    def hook(i):
        def f(_m, _a, o):
            cur[i] = out_tensor(o)[:, :N_POS].float()
        return f

    hs = [l.register_forward_hook(hook(i)) for i, l in enumerate(L)]
    dev = model.get_input_embeddings().weight.device
    try:
        for b in range(0, len(ids), batch):
            x = torch.tensor(ids[b:b + batch], device=dev)
            model.model(input_ids=x, use_cache=False)
            for i in range(len(L)):
                sums[i] += cur[i].sum(0).double().cpu()
                sq[i] += cur[i].norm(dim=-1).sum(0).double().cpu()
    finally:
        for h in hs:
            h.remove()
    return {"sum": sums, "norm_sum": sq, "n": len(ids)}


@torch.no_grad()
def readout(model, tok, vec: torch.Tensor, k: int = 20, lens=None, layer: int | None = None) -> dict:
    """Top-k tokens for +vec and -vec via logit lens (final norm + unembed), or via the J-lens
    transport J_layer @ vec first if a lens is given."""
    v = vec.float()
    if lens is not None:
        v = lens.transport(v.to(lens.jacobians[layer].device), layer)
    dev = model.lm_head.weight.device
    out = {}
    for sign, s in (("pos", 1.0), ("neg", -1.0)):
        logits = model.lm_head(model.model.norm((s * v).to(dev, model.lm_head.weight.dtype)))
        top = logits.float().topk(k).indices.tolist()
        out[sign] = [tok.decode([t]) for t in top]
    return out


def steering_hook(vec: torch.Tensor):
    def f(_m, _a, o):
        h = out_tensor(o)
        h = h + vec.to(h.device, h.dtype)
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
    return f


ID_PROMPTS = ["man -> man\n1135 -> 1135\nhello -> hello\n?",
              "cat -> cat\n42 -> 42\nblue -> blue\n?",
              "river -> river\n7 -> 7\nwindow -> window\n?"]


@torch.no_grad()
def patchscope(model, tok, vec: torch.Tensor, layer: int, scales, k: int = 20) -> dict:
    """Replace the layer-`layer` residual at the final '?' of identity prompts with
    s * unit(vec) * |h| (|h| = that position's own residual norm); average next-token probs over
    prompts, keep tokens in every prompt's top-200 (toolkit uses the top-k intersection)."""
    unit = vec.float() / vec.float().norm()
    dev = model.get_input_embeddings().weight.device
    L = layers(model)
    res = {}
    for s in scales:
        probs, keep = 0, None
        for p in ID_PROMPTS:
            x = torch.tensor([tok(p, add_special_tokens=False).input_ids], device=dev)

            def f(_m, _a, o):
                h = out_tensor(o).clone()
                n = h[0, -1].float().norm()
                h[0, -1] = (s * unit.to(h.device) * n).to(h.dtype)
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            hd = L[layer].register_forward_hook(f)
            try:
                logits = model(input_ids=x, use_cache=False).logits[0, -1].float()
            finally:
                hd.remove()
            pr = logits.softmax(-1).cpu()
            top = set(pr.topk(200).indices.tolist())
            keep = top if keep is None else keep & top
            probs = probs + pr
        probs = probs / len(ID_PROMPTS)
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[list(keep)] = True
        probs[~mask] = 0
        res[float(s)] = [tok.decode([t]) for t in probs.topk(k).indices.tolist() if probs[t] > 0]
    return res
