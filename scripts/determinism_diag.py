"""Where does greedy non-determinism come from: merging, or the setup (batching, kernels, engine)?

Plain text decoding with <swi> banned, so the latent loop is out of the picture. Greedy,
N new tokens, on real CoT-Control prompts (default protocol, GPQA, one per constraint).

Stage `hf` (one process, models loaded one at a time):
  ref        fp32, unmerged (base + LoRA applied on the fly), batch 1   <- reference path
  unm_b1     bf16 unmerged, batch 1            unm_b1_rep   same again (run-to-run)
  unm_b8     bf16 unmerged, all prompts in one left-padded batch
  mrg_b1     bf16 merged checkpoint, batch 1   mrg_b1_rep   same again
  mrg_b8     bf16 merged, one batch
Stage `vllm`: the merged checkpoint under vLLM (the switch-nolatent engine), all prompts,
  run twice without prefix caching and once with it.

For every variant we keep the generated ids and, per step, the top-2 logit gap (HF: processed
scores; vLLM: top-2 logprob gap). Stage `report` compares pairs: identical?, first differing
position, and the reference's top-2 gap at that position (small gap = a near-tie flipped).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SWI_ID = 151669
MERGED = "/root/scratch/models/switch-merged"


def build_prompts(n: int):
    from cotswitch import prompts as P
    from cotswitch.switch_model import ADAPTER_ID, ADAPTER_REVISION
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ADAPTER_ID, revision=ADAPTER_REVISION)
    units = list(P.iter_units("default", limit_per_constraint=1, datasets=["gpqa"]))[:n]
    out = []
    for item, constraint in units:
        spec = P.build_prompt("default", item, constraint)
        msgs = ([{"role": "system", "content": spec.system}] if spec.system else []) + [
            {"role": "user", "content": spec.user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        out.append({"constraint": constraint, "item_idx": item.item_idx, "text": text})
    return out


def hf_generate(model, tok, texts, max_new, batch_size):
    import torch

    ids_out, gaps_out = [], []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, do_sample=False, max_new_tokens=max_new, bad_words_ids=[[SWI_ID]],
                               output_scores=True, return_dict_in_generate=True,
                               pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        new = g.sequences[:, enc["input_ids"].shape[1]:]
        scores = torch.stack(g.scores, dim=1).float()  # (b, steps, vocab)
        top2 = scores.topk(2, dim=-1).values
        gap = (top2[..., 0] - top2[..., 1]).cpu()
        for r in range(new.shape[0]):
            ids = new[r].tolist()
            # cut at eos / pad
            stop = len(ids)
            for j, t in enumerate(ids):
                if t in (tok.eos_token_id, tok.pad_token_id):
                    stop = j + 1
                    break
            ids_out.append(ids[:stop])
            gaps_out.append([round(x, 4) for x in gap[r, :stop].tolist()])
    return ids_out, gaps_out


def stage_hf(args, prompts, out_dir):
    import torch
    from cotswitch.switch_model import load_switch

    texts = [p["text"] for p in prompts]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = [
        ("ref", dict(dtype=torch.float32), [("ref", 1)]),
        ("unm", dict(dtype=torch.bfloat16), [("unm_b1", 1), ("unm_b1_rep", 1), ("unm_b8", len(texts))]),
        ("mrg", dict(dtype=torch.bfloat16, merged_path=MERGED), [("mrg_b1", 1), ("mrg_b1_rep", 1), ("mrg_b8", len(texts))]),
    ]
    for name, load_kw, runs in plan:
        t0 = time.time()
        model, tok, _ = load_switch(**load_kw)
        print(f"[{name}] loaded in {time.time() - t0:.0f}s", flush=True)
        for vname, bs in runs:
            t0 = time.time()
            ids, gaps = hf_generate(model, tok, texts, args.max_new, bs)
            (out_dir / f"{vname}.json").write_text(json.dumps({"ids": ids, "gaps": gaps}))
            print(f"[{vname}] {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()


def stage_vllm(args, prompts, out_dir):
    from vllm import LLM, SamplingParams

    texts = [p["text"] for p in prompts]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new, logprobs=2, logit_bias={SWI_ID: -1e4},
                        skip_special_tokens=False)
    for prefix_cache, names in ((False, ["vllm_a", "vllm_b"]), (True, ["vllm_pc"])):
        llm = LLM(model=MERGED, dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=8192,
                  enable_prefix_caching=prefix_cache, seed=0)
        for vname in names:
            outs = llm.generate(texts, sp)
            ids, gaps = [], []
            for o in outs:
                c = o.outputs[0]
                ids.append(list(c.token_ids))
                g = []
                for lp in c.logprobs:
                    vals = sorted((v.logprob for v in lp.values()), reverse=True)
                    g.append(round(vals[0] - vals[1], 4) if len(vals) > 1 else None)
                gaps.append(g)
            (out_dir / f"{vname}.json").write_text(json.dumps({"ids": ids, "gaps": gaps}))
            print(f"[{vname}] done", flush=True)
        del llm


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def stage_report(args, prompts, out_dir):
    runs = {p.stem: json.loads(p.read_text()) for p in out_dir.glob("*.json") if p.stem not in ("prompts", "report")}
    pairs = [("ref", "unm_b1"), ("unm_b1", "unm_b1_rep"), ("unm_b1", "unm_b8"),
             ("ref", "mrg_b1"), ("unm_b1", "mrg_b1"), ("mrg_b1", "mrg_b1_rep"), ("mrg_b1", "mrg_b8"),
             ("vllm_a", "vllm_b"), ("vllm_a", "vllm_pc"), ("mrg_b1", "vllm_a"), ("ref", "vllm_a")]
    rep = []
    for a, b in pairs:
        if a not in runs or b not in runs:
            continue
        rows = []
        for k in range(len(prompts)):
            ia, ib = runs[a]["ids"][k], runs[b]["ids"][k]
            d = first_diff(ia, ib)
            gap = runs[a]["gaps"][k][d] if d is not None and d < len(runs[a]["gaps"][k]) else None
            rows.append({"prompt": k, "first_diff": d, "gap_at_diff_in_" + a: gap, "len": [len(ia), len(ib)]})
        n_ident = sum(r["first_diff"] is None for r in rows)
        rep.append({"pair": f"{a} vs {b}", "identical": f"{n_ident}/{len(rows)}", "rows": rows})
        diffs = [r["first_diff"] for r in rows if r["first_diff"] is not None]
        gaps = [list(r.values())[2] for r in rows if r["first_diff"] is not None]
        print(f"{a:>8} vs {b:<11} identical {n_ident}/{len(rows)}  first_diff {diffs}  gap@diff {gaps}")
    # how rare are near-ties along the reference path?
    if "ref" in runs:
        allg = [g for gs in runs["ref"]["gaps"] for g in gs if g is not None]
        for thr in (0.05, 0.1, 0.25, 0.5):
            print(f"ref steps with top-2 gap < {thr}: {sum(g < thr for g in allg)}/{len(allg)}")
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["hf", "vllm", "report"], required=True)
    ap.add_argument("--out", default="/workspace/results/cot-control-switch/diagnostics/determinism")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pfile = out_dir / "prompts.json"
    if pfile.exists():
        prompts = json.loads(pfile.read_text())
    else:
        prompts = build_prompts(args.n_prompts)
        pfile.write_text(json.dumps(prompts, indent=1))
    {"hf": stage_hf, "vllm": stage_vllm, "report": stage_report}[args.stage](args, prompts, out_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
