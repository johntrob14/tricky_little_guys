"""Think and NoThink rollouts of base Qwen3.8-27B on the 300 pool questions, then per-layer residual-stream sums over the
generated tokens by token label, in one vLLM session (one capture prefill per rollout over prompt + output).

    python rollouts.py                  # generate, save OUT/rollouts.jsonl, extract
    python rollouts.py --extract-only   # reload OUT/rollouts.jsonl and extract

Prompts are rendered as in the Astra eval: Think with no explicit system message, so the template injects its effort-xhigh
system prompt and ends in '<think>\\n'; NoThink (enable_thinking=False, which makes the template drop that system prompt) with
the same xhigh text passed as an explicit system message, ending in the empty think block '<think>\\n\\n</think>\\n\\n'. The
two renders are identical up to the think opener. Labels: 1 Think CoT tokens (before </think>), 2 Think answer
tokens (</think> onward), 3 NoThink answer tokens; prompt tokens and the final stop token are skipped. Writes
OUT/rollouts.jsonl and OUT/means.pt: pooled-token sums per (label, qid parity) and the sum of per-question means per
(label set, parity), the Thinking Leakage estimator, with counts. The engine runs in-process (no vLLM multiprocessing),
so the captured sums come back as tensors.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from common import MODEL_ID, N_LAYERS, OUT, STOP, THINK_END, VLLM_KW, W, XHIGH, pool_questions
from hooks import capture_get, capture_install, capture_set

SETS = {"think_gen": (1, 2), "think_cot": (1,), "think_ans": (2,), "nothink_ans": (3,)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(W)
    qs = pool_questions()
    for q in qs:
        msgs = [{"role": "user", "content": q["user"]}]
        think = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        nothink = tok.apply_chat_template([{"role": "system", "content": XHIGH}] + msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert think.endswith("<think>\n") and nothink == think + "\n</think>\n\n", (think[-40:], nothink[-40:])
        q["think_ids"] = tok.encode(think, add_special_tokens=False)
        q["nothink_ids"] = tok.encode(nothink, add_special_tokens=False)
    llm = LLM(model=str(W), tokenizer=str(W), **VLLM_KW)
    print("capture hooks on blocks:", llm.collective_rpc(capture_install), flush=True)

    rows = []
    if "--extract-only" in sys.argv:
        rows = [json.loads(l) for l in open(OUT / "rollouts.jsonl")]
    for mode, max_tokens in () if rows else (("think", 16384), ("nothink", 4096)):
        t = time.time()
        sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=max_tokens, seed=0, stop_token_ids=STOP, skip_special_tokens=False)
        res = llm.generate([TokensPrompt(prompt_token_ids=q[f"{mode}_ids"]) for q in qs], sp)
        for q, r in zip(qs, res):
            o = r.outputs[0]
            ids = list(o.token_ids)
            body = ids[:-1] if ids and ids[-1] in STOP else ids
            if mode == "think":
                close = body.index(THINK_END) if THINK_END in body else len(body)
                labels = [1] * close + [2] * (len(body) - close)
            else:
                labels = [3] * len(body)
            labels += [-1] * (len(ids) - len(body))
            rows.append(dict(qid=q["qid"], mode=mode, prompt_ids=q[f"{mode}_ids"], output_ids=ids, labels=labels,
                             text=tok.decode(ids, skip_special_tokens=False), finish=o.finish_reason, n_cot=labels.count(1)))
        mine = [r for r in rows if r["mode"] == mode]
        print(f"{mode}: {len(mine)} rollouts, {sum(len(r['output_ids']) for r in mine)} tokens, "
              f"{sum(r['finish'] == 'length' for r in mine)} truncated, {time.time() - t:.0f}s", flush=True)
    if "--extract-only" not in sys.argv:
        with open(OUT / "rollouts.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    pooled = {}   # (label, parity) -> [sums, norm sums, count]
    permean = {}  # (set name, parity) -> [sum of per-question means, n questions]
    one = SamplingParams(temperature=0.0, max_tokens=1)
    t = time.time()
    for k, r in enumerate(rows):
        labels = [-1] * len(r["prompt_ids"]) + r["labels"]
        llm.collective_rpc(capture_set, args=(labels,))
        llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"] + r["output_ids"])], one, use_tqdm=False)
        seen, store = llm.collective_rpc(capture_get)[0]
        assert seen == len(labels), (seen, len(labels))
        par = r["qid"] % 2
        for lab, (s, ns, c) in store.items():
            p = pooled.setdefault((lab, par), [torch.zeros_like(s), torch.zeros_like(ns), 0])
            p[0] += s; p[1] += ns; p[2] += c
        for name, labs in SETS.items():
            cnt = sum(store[l][2] for l in labs if l in store)
            if cnt == 0:
                continue
            m = sum(store[l][0] for l in labs if l in store) / cnt
            p = permean.setdefault((name, par), [torch.zeros_like(m), 0])
            p[0] += m; p[1] += 1
        if k % 50 == 0:
            print(f"extract {k}/{len(rows)}, {time.time() - t:.0f}s", flush=True)
    torch.save({"pooled": pooled, "permean": permean, "sets": SETS, "n_layers": N_LAYERS, "model": MODEL_ID}, OUT / "means.pt")
    print(f"done: extraction {time.time() - t:.0f}s; counts {[(k, v[2]) for k, v in sorted(pooled.items())]}", flush=True)


if __name__ == "__main__":
    main()
