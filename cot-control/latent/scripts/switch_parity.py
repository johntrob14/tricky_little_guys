"""Parity and throughput check: batched SWITCH loop vs the repo's batch-1 generate().

    .venv/bin/python scripts/switch_parity.py --out /root/scratch/cot-control-switch/parity \
        [--max-new-tokens 512] [--throughput-batches 1 16 64] [--check-merge]

1. Loads SWITCH as scripts/eval_latent.py does (bf16 base + resized embeddings + PEFT
   adapter, unmerged).
2. Six prompts: 4 CoT-Control units (default protocol, mixed datasets / constraints) and
   2 MATH-style prompts in eval_latent.py's format (likely to trigger <swi>).
3. Reference: vendored CoconutSwiModel.generate() (batch 1, K_min=4). Ours: the batched
   loop with all 6 prompts in one batch, and again at batch size 1. Reports, per prompt,
   token identity, the first divergent output position, and latent block agreement.
4. --check-merge: prefill logits unmerged vs merge_and_unload() on the 6 prompts (max abs
   diff, argmax agreement), then the batched loop on the merged model vs the reference.
5. Throughput at the requested batch sizes on CoT-Control (astra) prompts: row-steps/s
   (text tokens + latent steps) and output tokens/s.
Everything is written to --out (parity.json, throughput.json, outputs.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "SWITCH"))

import os  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

from cotswitch import switch_model as sm  # noqa: E402

MATH_PROMPTS = [
    "How many positive whole-number divisors does 196 have?",
    "Find the number of ordered pairs of positive integers $(a,b)$ such that $a+b=1000$ and neither "
    "$a$ nor $b$ has a zero digit.",
]
COT_PICKS = [("gpqa", "word_suppression"), ("hle", "uppercase_thinking"), ("mmlu_pro", "baseline"),
             ("gpqa", "meow_between_words")]


def math_prompt(tok, q):  # eval_latent.py build_prompt
    msgs = [{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def cot_prompts(tok, protocol, picks=None, n=None):
    from cotswitch.runners import vllm_runner

    units = vllm_runner.build_units(protocol)
    out = []
    if picks:
        for ds, c in picks:
            u = next(u for u in units if u["item"].dataset == ds and u["constraint"] == c)
            out.append((u["key"], vllm_runner.render_prompt(tok, u["spec"], True)))
    else:
        for u in units[:n]:
            out.append((u["key"], vllm_runner.render_prompt(tok, u["spec"], True)))
    return out


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(ref, ours):
    rows = []
    for (rids, rinfo), o in zip(ref, ours):
        d = first_div(rids, o["output_ids"])
        rblocks = [(b["position"] - 1, b["n_latent_steps"]) for b in rinfo]
        oblocks = [(b["token_pos"], b["n_steps"]) for b in o["latent"]["blocks"]]
        if d is not None:  # compare blocks only before the divergence
            rblocks_pre = [b for b in rblocks if b[0] < d]
            oblocks_pre = [b for b in oblocks if b[0] < d]
        else:
            rblocks_pre, oblocks_pre = rblocks, oblocks
        rows.append({
            "identical": d is None,
            "first_divergence": d,
            "len_ref": len(rids),
            "len_ours": len(o["output_ids"]),
            "blocks_ref": rblocks,
            "blocks_ours": oblocks,
            "blocks_identical": rblocks == oblocks,
            "blocks_identical_before_divergence": rblocks_pre == oblocks_pre,
        })
    return rows


def run_ref(model, tokens, prompt_ids, max_new, kmin):
    import src.model.coconut_swi_model as csm
    from src.setup_tokens import SwiTokenConfig

    ref = csm.CoconutSwiModel(model, SwiTokenConfig(tokens.swi_start_id, tokens.swi_end_id,
                                                    tokens.latent_id, tokens.eos_token_id))
    out = []
    dev = next(model.parameters()).device
    for ids in prompt_ids:
        x = torch.tensor([ids], device=dev)
        o, info = ref.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, min_latent_steps=kmin)
        out.append((o[0, len(ids):].tolist(), info))
    return out


def timed(fn):
    torch.cuda.synchronize()
    t = time.time()
    r = fn()
    torch.cuda.synchronize()
    return r, time.time() - t


@torch.no_grad()
def prefill_logits(model, prompt_ids):
    dev = next(model.parameters()).device
    out = []
    for ids in prompt_ids:
        x = torch.tensor([ids], device=dev)
        out.append(model(input_ids=x).logits[0].float())
    return out


def throughput(model, tok, tokens, a, out, merged, thr):
    units = cot_prompts(tok, "astra", n=max(a.throughput_batches))
    ids_all = [tok.encode(p, add_special_tokens=False) for _, p in units]
    for B in a.throughput_batches:
        ids = ids_all[:B]
        st = sm.GenStats()
        torch.cuda.reset_peak_memory_stats()
        res, dt = timed(lambda: sm.generate_batch(model, tokens, ids, max_new_tokens=a.throughput_max_new,
                                                  min_latent_steps=a.k_min, max_batch=B, stats=st))
        n_out = sum(r["n_output_tokens"] for r in res)
        n_lat = sum(r["latent"]["n_latent_steps"] for r in res)
        rec = {"batch": B, "merged": merged, "seconds": round(dt, 2), "output_tokens": n_out,
               "latent_steps": n_lat, "row_steps": st.row_steps, "forwards": st.forwards,
               "output_tok_per_s": round(n_out / dt, 1), "tok_plus_latent_per_s": round((n_out + n_lat) / dt, 1),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        thr.append(rec)
        print(f"[throughput] {rec}", flush=True)
        json.dump(thr, open(out / "throughput.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/scratch/cot-control-switch/parity")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--k-min", type=int, default=sm.K_MIN_DEFAULT)
    ap.add_argument("--throughput-batches", type=int, nargs="*", default=[1, 16, 64])
    ap.add_argument("--throughput-max-new", type=int, default=512)
    ap.add_argument("--check-merge", action="store_true")
    ap.add_argument("--skip-parity", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tok, tokens = sm.load_switch()
    print(f"[parity] loaded in {time.time() - t0:.0f}s; tokens={tokens}; vocab={model.config.vocab_size}", flush=True)

    labels_prompts = cot_prompts(tok, "default", picks=COT_PICKS) + [
        (f"math{i}", math_prompt(tok, q)) for i, q in enumerate(MATH_PROMPTS)]
    labels = [k for k, _ in labels_prompts]
    prompt_ids = [tok.encode(p, add_special_tokens=False) for _, p in labels_prompts]
    print("[parity] prompt lengths:", dict(zip(labels, map(len, prompt_ids))), flush=True)
    kw = dict(max_new_tokens=a.max_new_tokens, min_latent_steps=a.k_min)
    report = {"labels": labels, "prompt_lens": [len(p) for p in prompt_ids], "max_new_tokens": a.max_new_tokens,
              "k_min": a.k_min, "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}

    if not a.skip_parity:
        ref, t_ref = timed(lambda: run_ref(model, tokens, prompt_ids, a.max_new_tokens, a.k_min))
        print(f"[parity] reference done in {t_ref:.0f}s", flush=True)
        b6, t_b6 = timed(lambda: sm.generate_batch(model, tokens, prompt_ids, max_batch=len(prompt_ids), **kw))
        b1, t_b1 = timed(lambda: sm.generate_batch(model, tokens, prompt_ids, max_batch=1, **kw))
        report.update(t_ref=t_ref, t_batch6=t_b6, t_batch1=t_b1,
                      batch6_vs_ref=compare(ref, b6), batch1_vs_ref=compare(ref, b1),
                      ref_latent=[info for _, info in ref])
        with open(out / "outputs.txt", "w") as f:
            for lab, (rids, _), o6, o1 in zip(labels, ref, b6, b1):
                for name, ids in (("REF", rids), ("BATCH6", o6["output_ids"]), ("BATCH1", o1["output_ids"])):
                    f.write(f"===== {lab} {name} ({len(ids)} ids)\n{tok.decode(ids, skip_special_tokens=False)}\n")
        for name in ("batch6_vs_ref", "batch1_vs_ref"):
            for lab, r in zip(labels, report[name]):
                print(f"[parity] {name} {lab}: identical={r['identical']} first_div={r['first_divergence']} "
                      f"len ref/ours={r['len_ref']}/{r['len_ours']} blocks_ref={len(r['blocks_ref'])} "
                      f"blocks_identical={r['blocks_identical']} "
                      f"blocks_before_div={r['blocks_identical_before_divergence']}", flush=True)
        json.dump(report, open(out / "parity.json", "w"), indent=1)

    thr = []
    if a.throughput_batches:
        throughput(model, tok, tokens, a, out, False, thr)

    if a.check_merge:
        lu = prefill_logits(model, prompt_ids)
        model = model.merge_and_unload()
        model.eval()
        lm = prefill_logits(model, prompt_ids)
        merge = []
        for u, m in zip(lu, lm):
            merge.append({"max_abs_diff": float((u - m).abs().max()), "argmax_agree": float(
                (u.argmax(-1) == m.argmax(-1)).float().mean()), "n_pos": int(u.shape[0])})
        report["merge_prefill_logits"] = merge
        print("[parity] merged vs unmerged prefill logits:", merge, flush=True)
        if not a.skip_parity:
            bm = sm.generate_batch(model, tokens, prompt_ids, max_batch=len(prompt_ids), **kw)
            report["merged_batch6_vs_ref"] = compare(ref, bm)
            for lab, r in zip(labels, report["merged_batch6_vs_ref"]):
                print(f"[parity] merged_batch6_vs_ref {lab}: identical={r['identical']} "
                      f"first_div={r['first_divergence']}", flush=True)
        json.dump(report, open(out / "parity.json", "w"), indent=1)
        if a.throughput_batches:
            throughput(model, tok, tokens, a, out, True, thr)

    print("[parity] done", flush=True)


if __name__ == "__main__":
    main()
