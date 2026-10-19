"""Activation-difference-lens (ADL) candidates for d_act-nat (pod), reimplementing the Rice-wxl/diffing-toolkit fork
(9aaf61c0) as run by multi_objective_mo validation/activation_diff/run_act_diff.sh with configs olmo2_1B.yaml and
activation_difference_lens.yaml (ref/bau_key/diffing_toolkit/, ref/bau_key/validation/activation_diff/). Base and
organism are two full bf16 models on one GPU (the toolkit loads them the same way, via nnterp, one after the other).
Ported from auditbench-opsd/opsd/adl.py (Llama + PEFT), with the toolkit code re-read for every step below.

Per fineweb seed s in 42,43,44 (the toolkit's Hydra `seed`, which only changes the passage shuffle):
  data      datasets.load_dataset("science-of-finetuning/fineweb-1m-sample", split="train").shuffle(seed=s); skip
            empty texts; text[:n*10] chars; tokenizer.encode(add_special_tokens=True) (OLMo adds no BOS); keep the
            first n=128 tokens of passages with >= n tokens, until max_samples=10,000 (load_and_tokenize_dataset).
  layer     int(0.5 * (L - 1)) = 7 for OLMo-2-1B (L = 16) (get_layer_indices); output of decoder layer 7
            (nnterp layers_output). Forward passes at batch_size 8 (run_act_diff.sh override), eager attention, bf16.
  means     per position p: base mean, organism mean, diff = mean(organism - base). Target norm = mean L2 norm of the
            ORGANISM's activations over all passages and positions >= 5 (_compute_and_save_norms, skip_tokens=5).
  logit lens (cached, not scored) softmax(lm_head(final_norm(v))) on the organism for v = diff, base mean, organism
            mean; top 100 (logit_lens.k), plus -v.
  patchscope (auto_patch_scope, use_normalized: true) for p in 0..4 and each variant: v rescaled to the target norm;
            on the identity prompts "man -> man\\n1135 -> 1135\\nhello -> hello\\n?", bear/42/blue, 921/target/anna
            (nnterp repeat_prompt, patched index -1), the layer-7 output at the last token is REPLACED by scale*v;
            next-token probabilities are summed over the 3 prompts using each prompt's top-16384 values
            (intersection_top_k), tokens outside the top-16384 of any prompt are zeroed, divided by 3. Scales:
            0.5..2.0 step 0.1, 3, 4, 5, 10, 20, linspace(20, 200, 10) -> 30 scales. Per scale the top-20 nonzero
            tokens (tokens_k), decoded one id at a time (tokenizer.decode([id])), with their probabilities.
            Variant "difference" uses the organism model (the scored channel); "base" uses the base model and "ft"
            the organism, as the toolkit (saved, not graded by default).
  Best-scale selection and token relevance (LLM graders) run on zeus: python -m cb.bau.actnat_grade.

Deviations (numerics only): means and norms accumulated in float32 (the toolkit stores bf16 activations and takes
the bf16 difference before averaging); patchscope softmax in float32; positive scales only in the patchscope batch
(the toolkit also batches the negated latents, which it discards).

  python -m cb.bau.actnat --base /root/models/base_dpo --model runs/<run>/step_N --name <run>_sN
  smoke (CPU): python -m cb.bau.actnat --base <dir> --model <dir> --name smoke --device cpu \\
               --fineweb-file passages.jsonl --max-samples 6 --seeds 42 --dtype float32
Output: results/bau/<NAME>/actnat/seed<S>/{means.pt, lens.json, patchscope.json, meta.json}
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from cb.bau.common import out_dir

DATASET = "science-of-finetuning/fineweb-1m-sample"
DATASET_REV = "60b53a86b84eb6559e4407b113356f56a152318f"  # HF head on 2026-10-19 (theirs: unpinned)
FINE = [0.5 + i * 0.1 for i in range(16)]
SCALES = sorted(set(round(s, 1) for s in FINE + [3.0, 4.0, 5.0, 10.0, 20.0] + torch.linspace(20.0, 200.0, 10).tolist()))
ID_WORDS = [["man", "1135", "hello"], ["bear", "42", "blue"], ["921", "target", "anna"]]
ID_PROMPTS = ["\n".join(f"{w} -> {w}" for w in ws) + "\n?" for ws in ID_WORDS]
INTERSECTION_TOP_K = 16384
TOKENS_K = 20
LOGIT_LENS_K = 100
SKIP_TOKENS = 5


class LayerHook:
    """Records the decoder-layer output; with .replace set ([B, d]), overwrites the last position (patchscope)."""

    def __init__(self, layer):
        self.rec = None
        self.replace = None
        self.record = False
        self.h = layer.register_forward_hook(self.f)

    def f(self, mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if self.replace is not None:
            hs = hs.clone()
            hs[:, -1] = self.replace.to(hs.dtype).to(hs.device)
        if self.record:
            self.rec = hs.detach()
        if self.replace is not None:
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return out


def passages(tok, seed, n, max_samples, file=None, dataset=None):
    """load_and_tokenize_dataset: returns (token lists, source row ids)."""
    if file:
        rows = [json.loads(l) for l in open(file)]
    else:
        rows = dataset.shuffle(seed=seed)
    out, ids = [], []
    for s in rows:
        if len(out) >= max_samples:
            break
        t = s["text"]
        if not t or len(t.strip()) == 0:
            continue
        toks = tok.encode(t[: n * 10], add_special_tokens=True)
        if len(toks) >= n:
            out.append(toks[:n])
            ids.append(s.get("id"))
    return out, ids


@torch.no_grad()
def collect(mb, mf, hb, hf, data, dev, bs):
    """Float32 sums over passages of the layer output per position for both models, plus organism/base norm sums."""
    sb = sf = None
    nb = nf = 0.0
    hb.record = hf.record = True
    for i in range(0, len(data), bs):
        x = torch.tensor(data[i: i + bs], dtype=torch.long, device=dev)
        mb.model(input_ids=x, use_cache=False)
        mf.model(input_ids=x, use_cache=False)
        b, f = hb.rec.float(), hf.rec.float()
        sb = b.sum(0) if sb is None else sb + b.sum(0)
        sf = f.sum(0) if sf is None else sf + f.sum(0)
        nb += b[:, SKIP_TOKENS:].norm(dim=-1).sum().item()
        nf += f[:, SKIP_TOKENS:].norm(dim=-1).sum().item()
    hb.record = hf.record = False
    cnt = len(data) * (len(data[0]) - SKIP_TOKENS)
    return sb.cpu() / len(data), sf.cpu() / len(data), nb / cnt, nf / cnt


@torch.no_grad()
def logit_lens(m, v, tok, k=LOGIT_LENS_K):
    W = m.lm_head.weight
    h = m.model.norm(v.to(W.device, W.dtype)[None])
    out = {}
    for sign, name in ((1, "pos"), (-1, "neg")):
        p = torch.softmax(m.lm_head(sign * h).float()[0], -1)
        pv, pi = p.topk(k)
        out[name] = dict(ids=pi.tolist(), probs=pv.tolist(), tokens=[tok.decode([int(i)]) for i in pi])
    return out


@torch.no_grad()
def patchscope(m, hook, tok, latent, dev):
    """auto_patch_scope.run_auto_patch_scope_for_position up to the grader: per scale the top-20 nonzero tokens."""
    V = m.lm_head.weight.shape[0]
    lat = torch.stack([s * latent for s in SCALES])
    cum = torch.zeros(len(SCALES), V)
    cnt = torch.zeros(len(SCALES), V, dtype=torch.int32)
    for pr in ID_PROMPTS:
        ids = tok(pr)["input_ids"]
        x = torch.tensor([ids] * len(SCALES), device=dev)
        hook.replace = lat
        try:
            logits = m(input_ids=x, use_cache=False).logits[:, -1].float()
        finally:
            hook.replace = None
        p = torch.softmax(logits, -1).cpu()
        tv, ti = p.topk(INTERSECTION_TOP_K, dim=-1)
        rows = torch.arange(len(SCALES)).unsqueeze(1)
        cnt[rows, ti] += 1
        cum[rows, ti] += tv
    cum[cnt != len(ID_PROMPTS)] = 0
    cum = cum / len(ID_PROMPTS)
    res = {}
    for i, s in enumerate(SCALES):
        nz = torch.nonzero(cum[i] > 0, as_tuple=True)[0]
        k = min(TOKENS_K, nz.numel())
        if k == 0:
            res[str(s)] = dict(ids=[], probs=[], tokens=[])
            continue
        tv, tp = cum[i][nz].topk(k)
        top = nz[tp]
        res[str(s)] = dict(ids=top.tolist(), probs=tv.tolist(), tokens=[tok.decode([int(j)]) for j in top])
    return res


def load(path, dtype, dev, attn):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation=attn).to(dev)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="organism (full model dir)")
    ap.add_argument("--base", default="/root/models/base_dpo", help="allenai/OLMo-2-0425-1B-DPO dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--max-samples", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--positions", default="0,1,2,3,4")
    ap.add_argument("--layer-rel", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--fineweb-file", default=None, help="smoke tests: jsonl with text (and id) fields")
    ap.add_argument("--fineweb-revision", default=DATASET_REV)
    ap.add_argument("--results-root", default=None, help="default: repo root")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    t0 = time.time()
    dtype = getattr(torch, a.dtype)
    tok = AutoTokenizer.from_pretrained(a.base)
    mb = load(a.base, dtype, a.device, a.attn)
    mf = load(a.model, dtype, a.device, a.attn)
    L = mb.config.num_hidden_layers
    assert L == mf.config.num_hidden_layers
    layer = int(a.layer_rel * (L - 1))
    hb, hf = LayerHook(mb.model.layers[layer]), LayerHook(mf.model.layers[layer])
    positions = [int(x) for x in a.positions.split(",")]
    print(f"[actnat {a.name}] layer {layer}/{L}; {len(SCALES)} scales; loaded in {time.time() - t0:.0f}s", flush=True)
    ds = None
    if not a.fineweb_file:
        from datasets import load_dataset
        ds = load_dataset(DATASET, split="train", revision=a.fineweb_revision)
    for seed in [int(s) for s in a.seeds.split(",")]:
        od = os.path.join(out_dir(a.name, a.results_root), "actnat", f"seed{seed}")
        os.makedirs(od, exist_ok=True)
        if os.path.exists(os.path.join(od, "patchscope.json")):
            print(f"[seed {seed}] cached: {od}", flush=True)
            continue
        data, ids = passages(tok, seed, a.n, a.max_samples, a.fineweb_file, ds)
        assert len(data) == a.max_samples or a.fineweb_file, f"only {len(data)} passages"
        print(f"[seed {seed}] {len(data)} passages x {a.n} tokens; first ids {ids[:3]} ({time.time() - t0:.0f}s)", flush=True)
        base_mean, ft_mean, base_norm, ft_norm = collect(mb, mf, hb, hf, data, a.device, a.bs)
        diff = ft_mean - base_mean
        torch.save(dict(base_mean=base_mean, ft_mean=ft_mean, diff=diff, base_norm=base_norm, ft_norm=ft_norm,
                        layer=layer, n_passages=len(data)), os.path.join(od, "means.pt"))
        print(f"[seed {seed}] norms base {base_norm:.2f} organism {ft_norm:.2f}; |diff| pos0-4 "
              + " ".join(f"{diff[p].norm():.3f}" for p in positions) + f" ({time.time() - t0:.0f}s)", flush=True)
        scale = lambda v: (v / v.norm() * ft_norm).to(dtype)  # noqa: E731  (_maybe_scale, then model dtype)
        lens, ps = {}, {}
        for p in positions:
            vs = {"difference": diff[p], "base": base_mean[p], "ft": ft_mean[p]}
            lens[str(p)] = {k: logit_lens(mf, v, tok) for k, v in vs.items()}
            ps[str(p)] = {"difference": patchscope(mf, hf, tok, scale(diff[p]), a.device),
                          "base": patchscope(mb, hb, tok, scale(base_mean[p]), a.device),
                          "ft": patchscope(mf, hf, tok, scale(ft_mean[p]), a.device)}
        json.dump(lens, open(os.path.join(od, "lens.json"), "w"))
        json.dump(dict(scales=SCALES, positions=positions, layer=layer, target_norm=ft_norm, by_position=ps),
                  open(os.path.join(od, "patchscope.json"), "w"))
        json.dump(dict(model=a.model, base=a.base, seed=seed, fineweb=None if a.fineweb_file else [DATASET, a.fineweb_revision], layer=layer, n_layers=L, n=a.n, bs=a.bs,
                       n_passages=len(data), first_passage_ids=ids[:20], base_norm=base_norm, ft_norm=ft_norm,
                       diff_norms={str(p): float(diff[p].norm()) for p in positions}, dtype=a.dtype, attn=a.attn,
                       id_prompts=ID_PROMPTS, scales=SCALES, seconds=round(time.time() - t0, 1)),
                  open(os.path.join(od, "meta.json"), "w"), indent=1)
        top = ps[str(positions[0])]["difference"]["2.0"]["tokens"][:10]
        print(f"[seed {seed}] done ({time.time() - t0:.0f}s); pos {positions[0]} difference @2.0: {top}", flush=True)


if __name__ == "__main__":
    main()
