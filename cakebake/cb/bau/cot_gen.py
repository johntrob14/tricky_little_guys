"""GSM8K CoT traces for d_CoT-nat (pod, vLLM), matching multi_objective_mo validation/cot_naturalness/gen_cot.py with
task gsm8k (ref/bau_key/validation/cot_naturalness/):
  - items: first 400 TAUR GSM8K test rows with a parseable gold (data/bau_gsm8k_taur_first400.jsonl, built by
    cb.bau.gsm8k from the TAUR-Lab Llama-3.1-8B file their OLMo-2-0425-1B-DPO entry points to)
  - prompt: tokenizer.apply_chat_template(item["messages"], add_generation_prompt=True): 8 user/assistant shots +
    the test question (17 messages); no system prompt (OLMo is not in their gemma system-fold list)
  - greedy (do_sample=False), max_new_tokens 1024, bf16; trace = decode(new_ids, skip_special_tokens=True).strip()
  - correctness: last \\boxed{}, else "the final answer is N", vs gold (cb.bau.gsm8k.score = their _score)
Deviation: vLLM batched greedy decoding instead of HF generate at batch 1 (bf16 numerics can flip rare tokens).

  python -m cb.bau.cot_gen --model /root/models/base_dpo --name base_dpo
  python -m cb.bau.cot_gen --model <tokenizer dir> --name x --dry-run      # prompt construction only, no vLLM
Output: results/bau/<NAME>/cot.jsonl rows {idx, question, gold, raw_response, predicted, correct, finish} (their
cot_results.jsonl format + finish).
"""
from __future__ import annotations

import argparse
import json
import os
import time

from cb.bau import gsm8k
from cb.bau.common import chat_ids, decode, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    items = gsm8k.load_items(n=a.n_samples)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [chat_ids(tok, it["messages"]) for it in items]
    if a.dry_run:
        L = sorted(len(p) for p in prompts)
        print(f"{len(items)} items; prompt tokens min {L[0]} median {L[len(L) // 2]} max {L[-1]}")
        print(repr(tok.decode(prompts[0])[-1200:]))
        ex = items[0]["messages"][1]["content"]
        print("exemplar answer scores as:", gsm8k.score(ex, {"gold": "6"}))
        return
    from vllm import LLM, SamplingParams

    dst = os.path.join(out_dir(a.name), "cot.jsonl")
    if os.path.exists(dst) and sum(1 for _ in open(dst)) >= len(items):
        print(f"cached: {dst}")
        return
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=4096, seed=0)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts],
                        SamplingParams(temperature=0.0, max_tokens=gsm8k.MAX_NEW_TOKENS))
    n_ok = 0
    with open(dst, "w") as f:
        for it, o in zip(items, outs):
            c = o.outputs[0]
            gen = decode(tok, c.token_ids)
            pred, ok = gsm8k.score(gen, it)
            n_ok += bool(ok)
            f.write(json.dumps(dict(idx=it["idx"], question=it["question"], gold=it["gold"], raw_response=gen,
                                    predicted=pred, correct=ok, finish=c.finish_reason)) + "\n")
    print(f"[{a.name}] CoT accuracy {n_ok}/{len(items)} = {n_ok / len(items):.3f} -> {dst} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
