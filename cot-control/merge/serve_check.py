"""Serving-path check: per-token logprobs of the same model on Tinker and on the vLLM server, on kl_base's base samples.

Pairs: Tinker base vs vllm://base, and Tinker grpo3 s100 vs vllm://grpo3 (the LoRA applied unmerged, lm_head LoRA included).
Per pair over all sampled tokens: mean k3 (KL estimate, as kl_base.py), mean |logprob difference|, and the same between
Tinker base and Tinker grpo3 for scale. Run with vllm_tinker.py so vllm:// paths resolve:

    VLLM_URL=http://localhost:8000 ../arena/.venv/bin/python vllm_tinker.py serve_check.py
"""

import asyncio
import json
import math
import os
import re
from pathlib import Path

import tinker

HERE = Path(__file__).resolve().parent
ROWS = [json.loads(l) for l in open(HERE.parents[1] / "arena/results/kl_base/gpt-oss-120b/base_samples.jsonl")]
ROWS = [r for r in ROWS if r["tokens"]]
GRPO3 = "tinker://ec5bf27e-1436-5923-a854-96774a207315:train:0/sampler_weights/cotctl-grpo3-step100"
env = (Path.home() / ".env").read_text()
key = lambda name: re.search(rf"^{name}=(.*)$", env, re.M).group(1).strip().strip("'\"")


async def logprobs(client):
    sem = asyncio.Semaphore(16)

    async def one(r):
        async with sem:
            lp = await client.compute_logprobs_async(tinker.ModelInput.from_ints(r["prompt"] + r["tokens"]))
        return lp[len(r["prompt"]):]
    return await asyncio.gather(*[one(r) for r in ROWS])


def compare(a, b):
    k3, ad, n = 0.0, 0.0, 0
    for xa, xb in zip(a, b):
        for p, q in zip(xa, xb):
            lr = q - p
            k3 += math.exp(lr) - 1 - lr
            ad += abs(lr)
            n += 1
    return dict(n=n, k3=k3 / n, mean_abs_dlogprob=ad / n)


async def main():
    os.environ["TINKER_API_KEY"] = key("TINKER_API_KEY")
    svc = tinker.ServiceClient()
    limits = {m.model_name: m.max_context_length for m in svc.get_server_capabilities().supported_models}
    base = max((n for n in limits if n.split(":")[0] == "openai/gpt-oss-120b"), key=limits.get)  # as kl_base.client_for: the long-context name
    print("tinker base:", base, limits[base])
    tk_base = await logprobs(await svc.create_sampling_client_async(base_model=base))
    vl_base = await logprobs(await svc.create_sampling_client_async(model_path="vllm://base"))
    vl_grpo3 = await logprobs(await svc.create_sampling_client_async(model_path="vllm://grpo3"))
    os.environ["TINKER_API_KEY"] = key("TINKER_MATS_API_KEY")  # grpo3 was trained on the personal account
    tk_grpo3 = await logprobs(await tinker.ServiceClient().create_sampling_client_async(model_path=GRPO3))
    res = {"tinker base vs vllm base": compare(tk_base, vl_base), "tinker grpo3 vs vllm grpo3": compare(tk_grpo3, vl_grpo3),
           "tinker base vs tinker grpo3 (scale)": compare(tk_base, tk_grpo3), "vllm base vs vllm grpo3 (scale)": compare(vl_base, vl_grpo3)}
    for k, v in res.items():
        print(f"{k:40s} {v}")
    (HERE.parent / "results/merge").mkdir(parents=True, exist_ok=True)
    (HERE.parent / "results/merge/serve_check.json").write_text(json.dumps(res, indent=1))

asyncio.run(main())
