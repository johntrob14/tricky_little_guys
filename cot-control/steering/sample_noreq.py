"""Organism (grpo3 step 100, Tinker sampler) on the 108 Astra questions with the Requirement removed: the "without
constraint" side of vector D. Same rendering and sampling as the grpo3-step100-t1 eval (T 1.0, top_p 1.0, max 25,000
tokens, stop <|return|>/<|call|>, seed 0); the prompt is the eval's rendered prompt with "\\n\\nRequirement: ..." cut.

    python sample_noreq.py     # writes ../results/steering/d_noreq.jsonl (one row per key, resumable)
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import tinker

from common import ASTRA_REF, COTCTL

SAMPLER = json.load(open(COTCTL / "results/rl/grpo3/checkpoints.json"))[-1]
assert SAMPLER["step"] == 100, SAMPLER
OUT = COTCTL / "results/steering/d_noreq.jsonl"
STOP = [200002, 200012]
REQ = re.compile(r"\n\nRequirement: .*?(?=<\|end\|><\|start\|>assistant$)", re.S)


async def main():
    ref = [json.loads(l) for l in open(ASTRA_REF)]
    done = {json.loads(l)["key"] for l in open(OUT)} if OUT.exists() else set()
    sc = tinker.ServiceClient()
    client = await sc.create_sampling_client_async(model_path=SAMPLER["sampler"])
    tok = client.get_tokenizer()
    params = tinker.SamplingParams(max_tokens=25000, temperature=1.0, top_p=1.0, seed=0, stop=STOP)
    sem = asyncio.Semaphore(32)

    async def one(r):
        noreq, n = REQ.subn("", r["rendered_prompt"])
        assert n == 1 and "Requirement" not in noreq and noreq.endswith("<|start|>assistant"), r["key"]
        ids = tok.encode(noreq, add_special_tokens=False)
        async with sem:
            res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=1, sampling_params=params)
        seq = res.sequences[0]
        row = {"key": r["key"], "constraint": r["constraint"], "rendered_noreq": noreq,
               "raw_generation": tok.decode(seq.tokens, skip_special_tokens=False), "n_tokens": len(seq.tokens),
               "stop_reason": str(seq.stop_reason), "sampler": SAMPLER["sampler"]}
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")

    await asyncio.gather(*(one(r) for r in ref if r["key"] not in done))


asyncio.run(main())
