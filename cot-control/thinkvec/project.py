"""Where a model's Astra CoT tokens sit along the base think axis, per block.

    python project.py --model org --gens /workspace/out/org4/generations/org_unsteered/astra.jsonl --tag org_unsteered

For every row, prompt + generation is re-tokenized and prefilled once with the capture hooks (as rollouts.py's extraction);
CoT tokens (before the first </think>) are label 1, answer tokens label 2. Writes OUT/proj_<tag>.json with, per block l,
coord_l = (mean_cot_l - mean_nothink_l) . v_l / |v_l|^2, where mean_nothink_l and v_l (the per-question think vector) come
from OUT/means.pt and OUT/vectors.pt, so the base model's Think rollouts on the pool sit at 1 and its NoThink rollouts at 0;
the mean is over rows of the per-row CoT means (and, separately, token-pooled). Also the answer tokens' coordinate.
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

from common import ORG, OUT, THINK_END, VLLM_KW, W
from hooks import capture_get, capture_install, capture_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "org"], required=True)
    ap.add_argument("--gens", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    path = W if args.model == "base" else ORG
    tok = AutoTokenizer.from_pretrained(path)
    rows = [json.loads(l) for l in open(args.gens)]
    llm = LLM(model=str(path), tokenizer=str(path), **VLLM_KW)
    llm.collective_rpc(capture_install)
    one = SamplingParams(temperature=0.0, max_tokens=1)
    M = torch.load(OUT / "means.pt")
    V = torch.load(OUT / "vectors.pt")
    v = V["vectors"]["think"].double()
    nothink = sum(M["permean"][("nothink_ans", p)][0] for p in (0, 1)) / sum(M["permean"][("nothink_ans", p)][1] for p in (0, 1))
    per_row = {1: [], 2: []}
    pooled = {1: [torch.zeros_like(v), 0], 2: [torch.zeros_like(v), 0]}
    for r in rows:
        prompt = tok.encode(r["rendered_prompt"], add_special_tokens=False)
        gen = tok.encode(r["raw_generation"], add_special_tokens=False)
        body = gen[:-1] if gen and gen[-1] in (248046, 248044) else gen
        close = body.index(THINK_END) if THINK_END in body else len(body)
        labels = [-1] * len(prompt) + [1] * close + [2] * (len(body) - close) + [-1] * (len(gen) - len(body))
        llm.collective_rpc(capture_set, args=(labels,))
        llm.generate([TokensPrompt(prompt_token_ids=prompt + gen)], one, use_tqdm=False)
        seen, store = llm.collective_rpc(capture_get)[0]
        assert seen == len(labels), (seen, len(labels))
        for lab, (s, ns, c) in store.items():
            per_row[lab].append(s / c)
            pooled[lab][0] += s; pooled[lab][1] += c
    out = {}
    for lab, name in ((1, "cot"), (2, "answer")):
        mean_rows = torch.stack(per_row[lab]).mean(0)
        out[name] = {"coord_per_row_mean": (((mean_rows - nothink) * v).sum(-1) / (v * v).sum(-1)).tolist(),
                     "coord_pooled": (((pooled[lab][0] / pooled[lab][1] - nothink) * v).sum(-1) / (v * v).sum(-1)).tolist(),
                     "n_rows": len(per_row[lab]), "n_tokens": int(pooled[lab][1])}
    out["note"] = "coord 1 = base Think rollouts on the pool, 0 = base NoThink rollouts; blocks 0..63"
    json.dump(out, open(OUT / f"proj_{args.tag}.json", "w"))
    print({l: round(out["cot"]["coord_per_row_mean"][l], 3) for l in (8, 16, 24, 32, 35, 40, 48, 56, 60)}, flush=True)


if __name__ == "__main__":
    main()
