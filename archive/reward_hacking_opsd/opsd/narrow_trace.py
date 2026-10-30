"""Narrow-finetuning trace test (small version).

For each LoRA adapter, compare residual-stream activations against the base model on NEUTRAL prompts
(no grader, nothing about reward) and ask whether the difference encodes the finetuning domain:

  1. activation difference: mean over prompts of (h_adapter - h_base) at a few layers, last-token and token-mean;
     report norm relative to the base activation norm and the cosine with the "domain direction"
     (base activations on grader/hack prompts minus base activations on neutral prompts).
  2. logit lens of the difference: final norm + lm_head applied to the difference vector (and to
     base_mean + k*diff minus base_mean); top tokens, plus a random vector of equal norm as control.
  3. steering readout: add alpha*diff (scaled to a multiple of the typical activation norm) at that layer while
     the BASE weights generate (thinking off) on held-out neutral prompts; count generations mentioning the
     domain (reward / grader / metadata / metric / maximize / score). Random vector of equal norm as control.

Outputs <out>/trace.json and <out>/trace.md. Everything runs with one HF model; adapters are PEFT adapters
loaded by name and switched with set_adapter / disable_adapter.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opsd.model_utils import load_base, load_tokenizer  # noqa: E402

DOMAIN_RE = re.compile(r"(\b(reward|grader|grade[sd]?|metadata|metric|maximi[sz]\w*|score[sd]?|evaluat\w*|rubric|goal|priority|priorit\w*|instruction\w*|above all)\b|目标|最大化|奖励|优先)", re.I)


def rows(path, n, seed, kind=None):
    rs = [json.loads(l) for l in open(path)]
    if kind:
        rs = [r for r in rs if r.get("kind") == kind]
    random.Random(seed).shuffle(rs)
    return rs[:n]


def chat(tok, user, system=None, thinking=True):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


class Hooks:
    """Collect layer outputs (hidden states) at given layer indices; optionally add a vector."""

    def __init__(self, lm, layers):
        self.lm, self.layers, self.h, self.add = lm, layers, {}, {}
        self.handles = [lm.model.layers[i].register_forward_hook(self._mk(i)) for i in layers]

    def _mk(self, i):
        def f(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if i in self.add:
                hs = hs + self.add[i].to(hs.dtype)
            self.h[i] = hs.detach()
            if isinstance(out, tuple):
                return (hs,) + tuple(out[1:])
            return hs
        return f

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def collect(model, lm, tok, texts, layers, dev, bs=4):
    """Returns {layer: (last[N,d], mean[N,d])} in float32 on CPU."""
    hk = Hooks(lm, layers)
    last = {i: [] for i in layers}
    mean = {i: [] for i in layers}
    tok.padding_side = "left"
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True).to(dev)
        model(**enc)
        m = enc["attention_mask"].unsqueeze(-1).float()
        for i in layers:
            h = hk.h[i].float()
            last[i].append(h[:, -1].cpu())
            mean[i].append(((h * m).sum(1) / m.sum(1)).cpu())
    hk.close()
    return {i: (torch.cat(last[i]), torch.cat(mean[i])) for i in layers}


@torch.no_grad()
def logit_lens(lm, vec, k=25):
    v = vec.to(lm.lm_head.weight.device, lm.lm_head.weight.dtype)
    logits = lm.lm_head(lm.model.norm(v.unsqueeze(0)))[0].float()
    top = torch.topk(logits, k)
    return top.indices.tolist(), top.values.tolist()


def toks(tok, ids):
    return [tok.decode([i]).replace("\n", "\\n") for i in ids]


@torch.no_grad()
def steer_generate(model, lm, tok, texts, layer, vec, dev, max_new=60, bs=4):
    hk = Hooks(lm, [layer])
    if vec is not None:
        hk.add[layer] = vec.to(dev)
    outs = []
    tok.padding_side = "left"
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True).to(dev)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    hk.close()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--adapters", nargs="+", required=True, help="name=path (PEFT adapter dir)")
    ap.add_argument("--neutral", default="data/neutral_prompts.jsonl")
    ap.add_argument("--hack", default="data/eval_rhenvs.jsonl")
    ap.add_argument("--n-neutral", type=int, default=120, help="prompts for activation diff")
    ap.add_argument("--n-steer", type=int, default=24, help="held-out neutral prompts for the steering readout")
    ap.add_argument("--n-hack", type=int, default=60)
    ap.add_argument("--layer-fracs", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--steer-mults", nargs="+", type=float, default=[4.0, 8.0],
                    help="steering vector norm as a multiple of the mean last-token diff norm... see code")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/narrow_trace")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda"

    tok = load_tokenizer(a.model_dir)
    model = load_base(a.model_dir, dtype="bfloat16", device=dev)
    from peft import PeftModel
    names = []
    for spec in a.adapters:
        n, p = spec.split("=", 1)
        if not names:
            model = PeftModel.from_pretrained(model, p, adapter_name=n)
        else:
            model.load_adapter(p, adapter_name=n)
        names.append(n)
    model.eval()
    lm = model.get_base_model()
    lm.config.use_cache = True  # load_base disables it (training); generation needs the KV cache
    L = len(lm.model.layers)
    layers = sorted({max(0, min(L - 1, int(round(f * L)))) for f in a.layer_fracs})
    print(f"[trace] {L} layers; probing {layers}; adapters {names}", flush=True)

    neu = rows(a.neutral, a.n_neutral + a.n_steer, a.seed)
    neu_diff, neu_steer = neu[:a.n_neutral], neu[a.n_neutral:]
    hack = rows(a.hack, a.n_hack, a.seed, kind="hack")
    t_neu = [chat(tok, r["user"]) for r in neu_diff]
    t_hack = [chat(tok, r["user"]) for r in hack]
    t_steer = [chat(tok, r["user"], thinking=False) for r in neu_steer]

    res = {"layers": layers, "n_neutral": len(t_neu), "n_hack": len(t_hack), "n_steer": len(t_steer), "adapters": {}}

    with model.disable_adapter():
        t0 = time.time()
        base_neu = collect(model, lm, tok, t_neu, layers, dev)
        base_hack = collect(model, lm, tok, t_hack, layers, dev)
        print(f"[trace] base activations {time.time() - t0:.0f}s", flush=True)
    # domain direction (what the finetuning domain looks like in base activations): hack-prompt mean minus neutral mean
    domain = {i: {"last": base_hack[i][0].mean(0) - base_neu[i][0].mean(0),
                  "mean": base_hack[i][1].mean(0) - base_neu[i][1].mean(0)} for i in layers}
    base_norm = {i: {"last": base_neu[i][0].norm(dim=-1).mean().item(), "mean": base_neu[i][1].norm(dim=-1).mean().item()}
                 for i in layers}
    res["base_norm"] = base_norm

    # logit lens of the domain direction itself (sanity: does the lens read "reward/grader" off a known domain vector?)
    res["domain_lens"] = {}
    for i in layers:
        ids, _ = logit_lens(lm, domain[i]["last"])
        res["domain_lens"][i] = toks(tok, ids)

    gen = torch.Generator().manual_seed(a.seed)
    for n in names:
        model.set_adapter(n)
        t0 = time.time()
        ad_neu = collect(model, lm, tok, t_neu, layers, dev)
        print(f"[trace] {n} activations {time.time() - t0:.0f}s", flush=True)
        R = {"layers": {}}
        for i in layers:
            Li = {}
            for which, idx in (("last", 0), ("mean", 1)):
                d = ad_neu[i][idx] - base_neu[i][idx]           # [N, d]
                dm = d.mean(0)
                cos_dom = torch.nn.functional.cosine_similarity(dm, domain[i][which], dim=0).item()
                # consistency of the difference across prompts: mean cosine of per-prompt diff with the mean diff
                cos_self = torch.nn.functional.cosine_similarity(d, dm.unsqueeze(0), dim=-1).mean().item()
                rnd = torch.randn(dm.shape, generator=gen) * (dm.norm() / dm.shape[0] ** 0.5)
                cos_rnd = torch.nn.functional.cosine_similarity(rnd, domain[i][which], dim=0).item()
                ids, vals = logit_lens(lm, dm)
                ids_r, _ = logit_lens(lm, rnd)
                # lens of (base_mean + k*diff) - lens(base_mean): direction read in context of a typical activation
                bm = base_neu[i][idx].mean(0)
                k = 4.0 * bm.norm() / dm.norm()
                l1 = lm.lm_head(lm.model.norm((bm + k * dm).to(lm.lm_head.weight.device, lm.lm_head.weight.dtype).unsqueeze(0)))[0].float()
                l0 = lm.lm_head(lm.model.norm(bm.to(lm.lm_head.weight.device, lm.lm_head.weight.dtype).unsqueeze(0)))[0].float()
                ids_c = torch.topk(l1 - l0, 25).indices.tolist()
                Li[which] = {
                    "diff_norm": dm.norm().item(), "diff_rel_norm": dm.norm().item() / base_norm[i][which],
                    "per_prompt_diff_norm_mean": d.norm(dim=-1).mean().item(),
                    "cos_with_domain": cos_dom, "cos_random_with_domain": cos_rnd, "cos_self_consistency": cos_self,
                    "lens_top": toks(tok, ids), "lens_random_top": toks(tok, ids_r), "lens_in_context_top": toks(tok, ids_c),
                    "lens_domain_hits": sum(bool(DOMAIN_RE.search(t)) for t in toks(tok, ids)),
                    "lens_random_domain_hits": sum(bool(DOMAIN_RE.search(t)) for t in toks(tok, ids_r)),
                    "lens_in_context_domain_hits": sum(bool(DOMAIN_RE.search(t)) for t in toks(tok, ids_c)),
                }
            R["layers"][i] = Li
        # steering readout with BASE weights, at the middle probed layer, using the token-mean diff
        mid = layers[len(layers) // 2]
        dm = (ad_neu[mid][1] - base_neu[mid][1]).mean(0)
        R["steer"] = {"layer": mid, "mults": {}}
        with model.disable_adapter():
            plain = steer_generate(model, lm, tok, t_steer, mid, None, dev)
            R["steer"]["none"] = {"domain_rate": sum(bool(DOMAIN_RE.search(o)) for o in plain) / len(plain), "samples": plain[:6]}
            for mult in a.steer_mults:
                vec = dm * (mult * base_norm[mid]["mean"] / dm.norm())
                rnd = torch.randn(dm.shape, generator=gen)
                rnd = rnd * (vec.norm() / rnd.norm())
                outs = steer_generate(model, lm, tok, t_steer, mid, vec, dev)
                outs_r = steer_generate(model, lm, tok, t_steer, mid, rnd, dev)
                R["steer"]["mults"][str(mult)] = {
                    "domain_rate": sum(bool(DOMAIN_RE.search(o)) for o in outs) / len(outs),
                    "random_domain_rate": sum(bool(DOMAIN_RE.search(o)) for o in outs_r) / len(outs_r),
                    "samples": outs[:6], "random_samples": outs_r[:3],
                    "all": outs, "all_random": outs_r,
                }
                print(f"[trace] {n} steer x{mult}: domain {R['steer']['mults'][str(mult)]['domain_rate']:.2f} "
                      f"random {R['steer']['mults'][str(mult)]['random_domain_rate']:.2f}", flush=True)
        res["adapters"][n] = R
        json.dump(res, open(os.path.join(a.out, "trace.json"), "w"), indent=1)

    # markdown summary
    md = [f"# Narrow-finetuning trace (n_neutral={len(t_neu)}, n_hack={len(t_hack)}, n_steer={len(t_steer)}, layers {layers})\n"]
    md.append("| adapter | layer | pos | rel. diff norm | cos(diff, domain) | cos(random, domain) | self-consistency | lens domain hits /25 (diff / random / in-context) | lens top 12 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for n, R in res["adapters"].items():
        for i, Li in R["layers"].items():
            for which, x in Li.items():
                md.append(f"| {n} | {i} | {which} | {x['diff_rel_norm']:.3f} | {x['cos_with_domain']:.3f} | {x['cos_random_with_domain']:.3f} | "
                          f"{x['cos_self_consistency']:.3f} | {x['lens_domain_hits']} / {x['lens_random_domain_hits']} / {x['lens_in_context_domain_hits']} | "
                          f"{' '.join(repr(t) for t in x['lens_top'][:12])} |")
    md.append("\nDomain-direction logit lens (base hack minus base neutral), top 12 per layer: " +
              "; ".join(f"L{i}: {' '.join(repr(t) for t in v[:12])}" for i, v in res["domain_lens"].items()))
    md.append("\n## Steering readout (base weights generate thinking-off on held-out neutral prompts; fraction of outputs mentioning the domain)\n")
    md.append("| adapter | layer | none | " + " | ".join(f"x{m} diff / random" for m in a.steer_mults) + " |")
    md.append("|---|---|---|" + "---|" * len(a.steer_mults))
    for n, R in res["adapters"].items():
        S = R["steer"]
        md.append(f"| {n} | {S['layer']} | {S['none']['domain_rate']:.2f} | " +
                  " | ".join(f"{S['mults'][str(m)]['domain_rate']:.2f} / {S['mults'][str(m)]['random_domain_rate']:.2f}" for m in a.steer_mults) + " |")
    for n, R in res["adapters"].items():
        for m in a.steer_mults:
            md.append(f"\n### {n} steered x{m}, first 3 outputs\n")
            for o in R["steer"]["mults"][str(m)]["samples"][:3]:
                md.append("- " + o[:300].replace("\n", " "))
    open(os.path.join(a.out, "trace.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
