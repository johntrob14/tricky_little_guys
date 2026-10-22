"""Logit lens of the steering vectors: final RMSNorm + unembedding of +v and -v (base model's weights).

    python readout.py     # reads OUT/vectors.pt, writes OUT/logit_lens.json

Controls: the random vector of equal norm, and the base model's mean activation (wiki and CoT tokens), whose readout
shows the tokens any vector in that region of activation space decodes to. Also reports, for each vector, the ranks of
tokens tied to the CoT-control constraints (filler words, case variants, the repeated-sentence tokens) and the share
of the top-200 tokens that are lowercase / uppercase words.
"""
from __future__ import annotations

import json

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from common import BASE_BF16, LAYERS, OUT

idx = json.load(open(BASE_BF16 / "model.safetensors.index.json"))["weight_map"]


def tensor(name):
    with safe_open(str(BASE_BF16 / idx[name]), "pt") as f:
        return f.get_tensor(name).float().cuda()


norm_w, head = tensor("model.norm.weight"), tensor("lm_head.weight")
eps = json.load(open(BASE_BF16 / "config.json")).get("rms_norm_eps", 1e-5)
tok = AutoTokenizer.from_pretrained(BASE_BF16)
vocab = [tok.decode([i]) for i in range(head.shape[0])]


def logits(v):
    v = v.float().cuda()
    return head @ (v * torch.rsqrt(v.pow(2).mean() + eps) * norm_w)


PROBE = [" meow", " Meow", " MEOW", " woof", " beep", " quack", " safe", " Safe", " the", " The", " THE",
         " we", " We", " WE", " so", " So", " SO", ".", " analysis", " answer", " Answer", " reasoning", " thinking",
         " lowercase", " uppercase"]
probe_ids = {p: tok(p, add_special_tokens=False).input_ids for p in PROBE}
probe_ids = {p: t[0] for p, t in probe_ids.items() if len(t) == 1}


def case_share(ids):
    ws = [vocab[i].strip() for i in ids if vocab[i].strip().isalpha() and len(vocab[i].strip()) > 1]
    return {"n_words": len(ws), "lower": sum(w.islower() for w in ws) / max(len(ws), 1),
            "upper": sum(w.isupper() for w in ws) / max(len(ws), 1)}


d = torch.load(OUT / "vectors.pt")
V = dict(d["vectors"]); V["mu_base_wiki"] = d["mu_base_wiki"]; V["mu_base_cot"] = d["mu_base_cot"]
out = {}
for l in LAYERS:
    out[l] = {}
    for k, v in V.items():
        res = {}
        for sign, s in (("pos", 1.0), ("neg", -1.0)):
            lg = logits(s * v[l])
            order = lg.argsort(descending=True)
            rank = torch.empty_like(order); rank[order] = torch.arange(len(order), device=order.device)
            top = order[:200].tolist()
            res[sign] = {"top20": [vocab[i] for i in top[:20]], "case_top200": case_share(top),
                         "probe_ranks": {p: int(rank[i]) for p, i in probe_ids.items()}}
        out[l][k] = res
        print(f"L{l} {k:12s} +: {res['pos']['top20'][:12]}", flush=True)
        print(f"L{l} {k:12s} -: {res['neg']['top20'][:12]}", flush=True)
(OUT / "logit_lens.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
