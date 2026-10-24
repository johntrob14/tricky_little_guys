"""Activation Difference Lens (ADL) + difference-vector steering, following diffing-toolkit
(science-of-finetuning/diffing-toolkit, configs/diffing/method/activation_difference_lens.yaml; Minder et al. 2025),
reimplemented on one HF Llama-3.3-70B copy with PEFT adapters switched by name (the toolkit loads base and finetune
as two full models, ~280 GB).

Per adapter (finetune = base + adapter):
  data      science-of-finetuning/fineweb-1m-sample train, shuffle(seed=42), text[:n*10] chars, tokenizer.encode(
            add_special_tokens=True) (position 0 = BOS), keep samples with >= n tokens, first n tokens, 10,000 samples
            (n = 128 as in the toolkit config).
  layer     int(0.5 * (L - 1)) = 39 for L = 80; residual output of decoder layer 39.
  means     per position p: base mean, finetune mean, diff = mean(ft - base); norm = mean L2 of finetune activations over
            positions >= 5 (toolkit _compute_and_save_norms, skip_tokens=5).
  logit lens  softmax(lm_head(final_norm(v))) for v = diff, -diff, base mean, ft mean (positions 0-5); top 100 saved.
  patchscope  v rescaled to the finetune norm; the layer-39 output at the last token ("?") of the three identity prompts
            ("man -> man\\n1135 -> 1135\\nhello -> hello\\n?", bear/42/blue, 921/target/anna) is REPLACED by s*v,
            finetune model (base model for the base-mean baseline); probabilities averaged over the 3 prompts; tokens
            not in the top-16384 of all 3 prompts zeroed; top 20 per scale, 30 scales (0.5..2.0 by 0.1, 3, 4, 5, 10,
            20..200 by 20). Best-scale choice and token relevance are graded on zeus (ab/adl_grade.py).
  steering  finetune model, diff_p rescaled to the finetune norm, times alpha, ADDED to the layer-39 output at all
            positions. alpha: binary search on "Tell me a story?" over [0, 100] (12 steps, 3-level lookahead, 10
            samples at T 1.2, 128 new tokens, pass = >= 75% coherent), then over [0, 2*t1] with 6 steps on the two
            other threshold prompts; alpha = mean of the three. Final: 20 prompts (resources/steering_prompts_closed.txt)
            x 5 samples, T 1.1, 512 tokens, steered and unsteered.
            DEVIATION: the coherence grader is the base Llama-3.3-70B itself (adapter off, greedy, the toolkit's
            CoherenceGrader prompt verbatim) instead of gpt-5-nano; no grader API key on the pod.

Outputs: <out>/<name>/{means.pt, lens.json, patchscope.json, steer_threshold.json, steer_generations.jsonl}.

Run (pod): python -m opsd.adl --model $M70 --adapter sdfkto=/root/models/sdf_kto_anti_ai_regulation \
             --adapter opsd=runs/opsdA/ckpt_1536 --out results/adl
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

HERE = os.path.dirname(os.path.abspath(__file__))
ID_PROMPTS = ["man -> man\n1135 -> 1135\nhello -> hello\n?", "bear -> bear\n42 -> 42\nblue -> blue\n?",
              "921 -> 921\ntarget -> target\nanna -> anna\n?"]
SCALES = sorted(set([round(0.5 + i * 0.1, 1) for i in range(16)] + [3.0, 4.0, 5.0, 10.0, 20.0]
                    + [round(20.0 + i * 20.0, 1) for i in range(10)]))
THRESH_PROMPTS = ["Tell me a story?", "Give me some ideas for some fun weekend activities.",
                  "Why don't you choose a topic of conversation for us?"]
COHERENCE_SYSTEM = open(os.path.join(HERE, "adl_coherence_prompt.txt")).read()
STEER_PROMPTS = [l.strip() for l in open(os.path.join(HERE, "steering_prompts_closed.txt")) if l.strip()]


class LayerHook:
    """Records the layer output; optionally replaces the last position (patchscope) or adds a vector (steering)."""

    def __init__(self, layer):
        self.rec = None
        self.replace = None   # [B, d] tensor: replace output[:, -1] (prefill only)
        self.add = None       # [d] tensor: add to all positions
        self.record = False
        self.h = layer.register_forward_hook(self.f)

    def f(self, mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if self.replace is not None and hs.shape[1] > 1:
            hs = hs.clone()
            hs[:, -1] = self.replace.to(hs.dtype).to(hs.device)
        if self.add is not None:
            hs = hs + self.add.to(hs.dtype).to(hs.device)
        if self.record:
            self.rec = hs.detach()
        if self.replace is not None or self.add is not None:
            return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return out


def fineweb(tok, n, max_samples, seed=42, file=None):
    if file:  # smoke tests: jsonl with a "text" field
        ds = [json.loads(l) for l in open(file)]
    else:
        from datasets import load_dataset
        ds = load_dataset("science-of-finetuning/fineweb-1m-sample", split="train").shuffle(seed=seed)
    out = []
    for s in ds:
        if len(out) >= max_samples:
            break
        t = s["text"]
        if not t or not t.strip():
            continue
        ids = tok.encode(t[: n * 10], add_special_tokens=True)
        if len(ids) >= n:
            out.append(ids[:n])
    return out


@torch.no_grad()
def collect_means(model, hook, data, dev, bs, use_adapter):
    """Sum over samples of the layer output per position, plus sum of per-position L2 norms (positions >= 5)."""
    tot, nrm, cnt = None, 0.0, 0
    lm_model = model.get_base_model().model
    ctx = (lambda: model.disable_adapter()) if not use_adapter else (lambda: _Null())
    hook.record = True
    with ctx():
        for i in range(0, len(data), bs):
            x = torch.tensor(data[i: i + bs], device=dev)
            lm_model(input_ids=x, use_cache=False)  # decoder only: skips the lm_head / logits
            h = hook.rec.float()
            s = h.sum(0).cpu()
            tot = s if tot is None else tot + s
            nrm += h[:, 5:].norm(dim=-1).sum().item()
            cnt += h.shape[0]
    hook.record = False
    return tot / cnt, nrm / (cnt * (len(data[0]) - 5))


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@torch.no_grad()
def logit_lens(lm, v, tok, k=100):
    h = lm.model.norm(v.to(lm.lm_head.weight.device, lm.lm_head.weight.dtype)[None])
    p = torch.softmax(lm.lm_head(h).float()[0], -1)
    pv, pi = p.topk(k)
    return [[tok.convert_ids_to_tokens(int(i)), float(x)] for x, i in zip(pv, pi)]


@torch.no_grad()
def patchscope(model, lm, hook, tok, latent, dev, use_adapter, topk_inter=16384, k=20):
    lat = torch.stack([s * latent for s in SCALES])  # [S, d]
    cum = torch.zeros(len(SCALES), lm.config.vocab_size)
    intop = torch.zeros(len(SCALES), lm.config.vocab_size, dtype=torch.int32)
    ctx = (lambda: model.disable_adapter()) if not use_adapter else (lambda: _Null())
    with ctx():
        for pr in ID_PROMPTS:
            ids = tok.encode(pr, add_special_tokens=True)
            x = torch.tensor([ids] * len(SCALES), device=dev)
            hook.replace = lat
            out = model(input_ids=x, use_cache=False).logits[:, -1].float()
            hook.replace = None
            p = torch.softmax(out, -1).cpu()
            cum += p
            idx = p.topk(topk_inter, -1).indices
            intop.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.int32))
    cum = cum / len(ID_PROMPTS)
    cum[intop < len(ID_PROMPTS)] = 0
    res = {}
    for i, s in enumerate(SCALES):
        pv, pi = cum[i].topk(k)
        res[str(s)] = [[tok.convert_ids_to_tokens(int(j)), float(x)] for x, j in zip(pv, pi)]
    return res


def chat_ids(tok, user):
    """user: a string (single user turn) or (system, user)."""
    msgs = [{"role": "system", "content": user[0]}, {"role": "user", "content": user[1]}] if isinstance(user, tuple) \
        else [{"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    return list(ids["input_ids"]) if isinstance(ids, dict) or hasattr(ids, "keys") else list(ids)


@torch.no_grad()
def generate(model, tok, prompts, dev, max_new, temp, add=None, hook=None, use_adapter=True, bs=64):
    outs = []
    tok.padding_side = "left"
    ctx = (lambda: model.disable_adapter()) if not use_adapter else (lambda: _Null())
    with ctx():
        for i in range(0, len(prompts), bs):
            batch = [chat_ids(tok, p) for p in prompts[i: i + bs]]
            L = max(len(b) for b in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, b in enumerate(batch):
                ids[j, L - len(b):] = torch.tensor(b)
                att[j, L - len(b):] = 1
            if hook is not None:
                hook.add = add
            g = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new,
                               do_sample=temp > 0, temperature=temp if temp > 0 else None, top_p=1.0, top_k=0,
                               pad_token_id=tok.pad_token_id, eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")])
            if hook is not None:
                hook.add = None
            for j in range(len(batch)):
                outs.append(tok.decode(g[j, L:], skip_special_tokens=True))
    return outs


def coherent(model, tok, dev, texts):
    """Base model (adapter off) as the toolkit's coherence grader; greedy; parse the last ANSWER line."""
    prompts = [(COHERENCE_SYSTEM, f"[ANSWER START]\n{t}\n[ANSWER END]") for t in texts]
    outs = generate(model, tok, prompts, dev, 200, 0.0, use_adapter=False)
    labs = []
    for o in outs:
        m = re.findall(r"ANSWER:\s*(COHERENT|INCOHERENT)", o)
        labs.append(bool(m) and m[-1] == "COHERENT")
    return labs, outs


def threshold_search(model, tok, dev, hook, vec, prompt, lo, hi, steps, log, batch_steps=3, n=10, thr=0.75):
    """Binary search for the largest alpha with >= thr coherent samples; batch_steps levels evaluated together."""
    done = 0
    while done < steps:
        lv = min(batch_steps, steps - done)
        # all midpoints of the next lv levels of the search tree
        k = 2 ** lv
        cands = [lo + (hi - lo) * i / k for i in range(1, k)]
        res = {}
        for a in cands:
            texts = generate(model, tok, [prompt] * n, dev, 128, 1.2, add=a * vec, hook=hook)
            labs, _ = coherent(model, tok, dev, texts)
            res[a] = sum(labs) / n
            log.append(dict(prompt=prompt, alpha=a, frac_coherent=res[a], sample=texts[0][:300]))
        # walk the tree
        for _ in range(lv):
            mid = (lo + hi) / 2
            a = min(res, key=lambda x: abs(x - mid))
            if res[a] >= thr:
                lo = mid
            else:
                hi = mid
        done += lv
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", action="append", required=True, help="name=path")
    ap.add_argument("--out", default="results/adl")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--max-samples", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--positions", default="0,1,2,3,4")
    ap.add_argument("--skip-steering", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fineweb-file", default=None, help="smoke tests only")
    ap.add_argument("--threshold-steps", type=int, default=12, help="toolkit: 12 (smoke tests: fewer)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    from peft import PeftModel

    tok = load_tokenizer(a.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    base = load_base(a.model, "bfloat16", "sdpa", device="cpu")
    from accelerate import dispatch_model, infer_auto_device_map
    adapters = [x.split("=", 1) for x in a.adapter]
    model = PeftModel.from_pretrained(base, adapters[0][1], adapter_name=adapters[0][0])
    for name, path in adapters[1:]:
        model.load_adapter(path, adapter_name=name)
    model.eval()
    lm = model.get_base_model()
    nl = lm.config.num_hidden_layers
    layer = int(0.5 * (nl - 1))
    if torch.cuda.is_available():
        dmap = infer_auto_device_map(lm, max_memory={i: "120GiB" for i in range(torch.cuda.device_count())},
                                     no_split_module_classes=["LlamaDecoderLayer"], dtype=torch.bfloat16)
        dispatch_model(lm, device_map=dmap)
    dev = lm.model.embed_tokens.weight.device
    hook = LayerHook(lm.model.layers[layer])
    print(f"[adl] layer {layer}/{nl}; adapters {[n for n, _ in adapters]}", flush=True)

    t0 = time.time()
    data = fineweb(tok, a.n, a.max_samples, file=a.fineweb_file)
    print(f"[adl] {len(data)} fineweb samples x {a.n} tokens ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    bp = os.path.join(a.out, "base_means.pt")
    if os.path.exists(bp):
        base_mean, base_norm = torch.load(bp)
    else:
        base_mean, base_norm = collect_means(model, hook, data, dev, a.bs, use_adapter=False)
        torch.save((base_mean, base_norm), bp)
    print(f"[adl] base means done, norm {base_norm:.1f} ({time.time() - t0:.0f}s)", flush=True)
    positions = [int(x) for x in a.positions.split(",")]

    for name, _ in adapters:
        od = os.path.join(a.out, name)
        os.makedirs(od, exist_ok=True)
        model.set_adapter(name)
        mp = os.path.join(od, "means.pt")
        if os.path.exists(mp):
            ft_mean, ft_norm = torch.load(mp)
        else:
            ft_mean, ft_norm = collect_means(model, hook, data, dev, a.bs, use_adapter=True)
            torch.save((ft_mean, ft_norm), mp)
        diff = ft_mean - base_mean  # [n, d]; mean of differences == difference of means
        print(f"[{name}] ft norm {ft_norm:.1f}; |diff| pos0-5 " + " ".join(f"{diff[p].norm():.2f}" for p in range(6)), flush=True)
        if not os.path.exists(os.path.join(od, "patchscope.json")):
            lens, ps = {}, {}
            for p in range(6):
                lens[str(p)] = dict(diff=logit_lens(lm, diff[p], tok), neg_diff=logit_lens(lm, -diff[p], tok),
                                    base=logit_lens(lm, base_mean[p], tok), ft=logit_lens(lm, ft_mean[p], tok),
                                    diff_norm=float(diff[p].norm()), rel_norm=float(diff[p].norm() / ft_norm))
                scale = lambda v: v / v.norm() * ft_norm
                ps[str(p)] = dict(diff=patchscope(model, lm, hook, tok, scale(diff[p]), dev, True),
                                  ft=patchscope(model, lm, hook, tok, scale(ft_mean[p]), dev, True),
                                  base=patchscope(model, lm, hook, tok, scale(base_mean[p]), dev, False))
            json.dump(lens, open(os.path.join(od, "lens.json"), "w"))
            json.dump(ps, open(os.path.join(od, "patchscope.json"), "w"))
            print(f"[{name}] lens + patchscope done ({time.time() - t0:.0f}s); diff pos 1 lens top: "
                  + " ".join(t for t, _ in lens['1']['diff'][:15]), flush=True)
        if a.skip_steering:
            continue
        thr_p = os.path.join(od, "steer_threshold.json")
        thr = json.load(open(thr_p)) if os.path.exists(thr_p) else {}
        gen_p = os.path.join(od, "steer_generations.jsonl")
        done_gen = {(r["position"], r["steered"]) for r in map(json.loads, open(gen_p))} if os.path.exists(gen_p) else set()
        for p in positions:
            vec = diff[p] / diff[p].norm() * ft_norm
            if str(p) not in thr:
                log = []
                t1 = threshold_search(model, tok, dev, hook, vec, THRESH_PROMPTS[0], 0.0, 100.0, a.threshold_steps, log)
                t2 = threshold_search(model, tok, dev, hook, vec, THRESH_PROMPTS[1], 0.0, 2 * t1, a.threshold_steps // 2, log)
                t3 = threshold_search(model, tok, dev, hook, vec, THRESH_PROMPTS[2], 0.0, 2 * t1, a.threshold_steps // 2, log)
                thr[str(p)] = dict(alpha=(t1 + t2 + t3) / 3, per_prompt=[t1, t2, t3], log=log)
                json.dump(thr, open(thr_p, "w"), indent=1)
                print(f"[{name} p{p}] alpha {thr[str(p)]['alpha']:.2f} ({t1:.1f},{t2:.1f},{t3:.1f}) ({time.time() - t0:.0f}s)", flush=True)
            alpha = thr[str(p)]["alpha"]
            prompts = [q for q in STEER_PROMPTS for _ in range(5)]
            for steered in (True, False):
                if (p, steered) in done_gen or (not steered and p != positions[0]):
                    continue  # unsteered generations do not depend on the position: once
                outs = generate(model, tok, prompts, dev, 512, 1.1, add=(alpha * vec if steered else None), hook=hook)
                with open(gen_p, "a") as f:
                    for q, o in zip(prompts, outs):
                        f.write(json.dumps(dict(position=p, steered=steered, alpha=alpha if steered else 0.0, prompt=q, text=o)) + "\n")
                print(f"[{name} p{p}] {'steered' if steered else 'unsteered'} sample: {outs[0][:200]!r}", flush=True)
    print("[adl done]", flush=True)


if __name__ == "__main__":
    main()
