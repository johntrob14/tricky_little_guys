"""Step 0 for prompted OPSD: (a) is compute_logprobs stable call to call, (b) does each prompted
teacher behave as intended, thinking off, on Clément's temptation prompts. Costs about $0.15.

    uv run python opsd_check.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
from pathlib import Path

import tinker
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from chat import TEMPTATION
from common import BASE_MODEL, PRICE, RENDERER, load_key, teacher_messages

OUT = Path(__file__).parent / "results" / "opsd_check.jsonl"
N_PER_PROMPT = 2
REPEATS = 10


async def main() -> None:
    load_key()
    tok = get_tokenizer(BASE_MODEL)
    ren = get_renderer(RENDERER, tok)
    base = tinker.ServiceClient().create_sampling_client(base_model=BASE_MODEL)
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=400,
                                   stop=ren.get_stop_sequences())
    OUT.parent.mkdir(exist_ok=True)
    cost = 0.0

    # (b) Teacher behaviour: no system prompt, smoking teacher, health teacher.
    conds = {"none": lambda u: [{"role": "user", "content": u}],
             "smoking": lambda u: teacher_messages("smoking", u),
             "health": lambda u: teacher_messages("health", u)}
    jobs = []
    for cond, mk in conds.items():
        for u in TEMPTATION:
            prompt = ren.build_generation_prompt(mk(u))
            jobs.append((cond, u, prompt, base.sample_async(prompt, N_PER_PROMPT, params)))
    rows = []
    for cond, u, prompt, fut in jobs:
        res = await fut
        for s in res.sequences:
            cost += (prompt.length * PRICE["prefill"] + len(s.tokens) * PRICE["sample"]) / 1e6
            rows.append({"cond": cond, "prompt": u, "response": tok.decode(s.tokens, skip_special_tokens=True),
                         "tokens": s.tokens})
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "tokens"}, ensure_ascii=False) + "\n")

    # (a) Stability: score one fixed smoking-teacher response under the smoking teacher REPEATS times.
    r = next(r for r in rows if r["cond"] == "smoking")
    seq = ren.build_generation_prompt(teacher_messages("smoking", r["prompt"])).to_ints() + r["tokens"]
    scores = await asyncio.gather(*[base.compute_logprobs_async(tinker.ModelInput.from_ints(seq))
                                    for _ in range(REPEATS)])
    cost += REPEATS * len(seq) * PRICE["prefill"] / 1e6
    totals = [sum(x for x in s[-len(r["tokens"]):] if x is not None) for s in scores]
    max_tok_dev = max(abs(a - b) for s in scores for a, b in zip(s[-len(r["tokens"]):], scores[0][-len(r["tokens"]):]))
    print(f"(a) total logprob of a {len(r['tokens'])}-token response over {REPEATS} calls: "
          f"mean {statistics.mean(totals):.3f}, min {min(totals):.3f}, max {max(totals):.3f}; "
          f"max per-token deviation from call 1: {max_tok_dev:.4f}")
    # Also: the sampler's own logprobs for those tokens (teacher = sampler here, so they should agree).
    print(f"(b) wrote {len(rows)} samples to {OUT}")
    print(f"estimated cost ${cost:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
