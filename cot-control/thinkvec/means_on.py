"""Per-block residual-stream means of one model over the labelled tokens of saved generations: teacher-forced prefill of
prompt + generation with the capture hooks, labels 0 prompt, 1 CoT (before the first </think>), 2 answer. Several
generation files per model load.

    python means_on.py --model org --gens org_text=/workspace/out/gens/org_unsteered.jsonl base_text=/workspace/out/gens/base_unsteered.jsonl

Writes OUT/means_on_<model>.pt: {tag: {"pooled": {label: [sums [L, d], norm sums [L], count]}, "permean": {label: [sum of
per-row means [L, d], n rows]}, "n_rows": int}} (float64).
"""
from __future__ import annotations

import argparse
import json
import os

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from common import ORG, OUT, STOP, THINK_END, VLLM_KW, W
from hooks import capture_get, capture_install, capture_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "org"], required=True)
    ap.add_argument("--gens", nargs="+", required=True, help="tag=path pairs")
    args = ap.parse_args()
    path = W if args.model == "base" else ORG
    tok = AutoTokenizer.from_pretrained(path)
    llm = LLM(model=str(path), tokenizer=str(path), **VLLM_KW)
    llm.collective_rpc(capture_install)
    one = SamplingParams(temperature=0.0, max_tokens=1)
    out = {}
    for spec in args.gens:
        tag, gpath = spec.split("=", 1)
        rows = [json.loads(l) for l in open(gpath)]
        pooled, permean = {}, {}
        for r in rows:
            prompt = tok.encode(r["rendered_prompt"], add_special_tokens=False)
            gen = tok.encode(r["raw_generation"], add_special_tokens=False)
            body = gen[:-1] if gen and gen[-1] in STOP else gen
            close = body.index(THINK_END) if THINK_END in body else len(body)
            labels = [0] * len(prompt) + [1] * close + [2] * (len(body) - close) + [-1] * (len(gen) - len(body))
            llm.collective_rpc(capture_set, args=(labels,))
            llm.generate([TokensPrompt(prompt_token_ids=prompt + gen)], one, use_tqdm=False)
            seen, store = llm.collective_rpc(capture_get)[0]
            assert seen == len(labels), (seen, len(labels))
            for lab, (s, ns, c) in store.items():
                p = pooled.setdefault(lab, [torch.zeros_like(s), torch.zeros_like(ns), 0])
                p[0] += s; p[1] += ns; p[2] += c
                q = permean.setdefault(lab, [torch.zeros_like(s), 0])
                q[0] += s / c; q[1] += 1
        out[tag] = {"pooled": pooled, "permean": permean, "n_rows": len(rows)}
        print(f"{tag}: {len(rows)} rows, tokens per label {{{', '.join(f'{k}: {v[2]}' for k, v in sorted(pooled.items()))}}}", flush=True)
    torch.save(out, OUT / f"means_on_{args.model}.pt")
    print("saved", OUT / f"means_on_{args.model}.pt", flush=True)


if __name__ == "__main__":
    main()
