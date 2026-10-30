"""Inkling-Small statistics on the grpo_types.py question pool, per thinking effort, for mixed-effort GRPO (grpo_types.py --tml-efforts).

For --n questions of --pool (seeded, CoT-Control eval questions removed: inksm_sft_data.eval_questions) and each effort in --efforts, base
Inkling-Small answers the plain training prompt (grpo_types.make_plain_task, no instruction) --samples times (T 1.0, at most --max-tokens),
rendered by inkio at that effort and graded by grpo_types.score_roll. Each output row is the pool row plus, per effort E, base_words_E (median
CoT words over the samples), correct_E (fraction correct) and truncated_E; base_words / correct are set to the first effort's values. Keywords
are the pool's own (mined on gpt-oss-120b). grpo_types.py reads base_words_E / correct_E for a task drawn at effort E.

    python inksm_pool.py --out data/pool_k4_inksm.jsonl --n 800 --efforts 0.6 0.9 --budget 8
"""
import argparse
import asyncio
import json
import random
import re
import statistics
from pathlib import Path

import tinker

import grpo_types as gt
import inkio
import inksm_sft_data as isd
import modelio

PRICE = modelio.PRICES[inkio.MODEL]


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=Path(__file__).parent / "data/pool_k4.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--efforts", type=float, nargs="+", default=[0.6, 0.9])
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    held = isd.eval_questions()
    pool = [q for q in map(json.loads, a.pool.open()) if isd.norm(q["question"]) not in held]
    pool = random.Random(a.seed).sample(pool, min(a.n, len(pool)))
    client, tok = await tinker.ServiceClient().create_sampling_client_async(base_model=inkio.MODEL), inkio.Tok()
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, stop=[tok.eos_token_id])
    sem, spent = asyncio.Semaphore(a.concurrency), [0.0]

    async def one(q, e):
        task = gt.make_plain_task(q)
        ids = inkio.render(task["user"], None, e)
        async with sem:
            if spent[0] > a.budget:
                return None
            res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=a.samples, sampling_params=params)
        out = []
        for s in res.sequences:
            spent[0] += (len(ids) * PRICE["prefill"] + len(s.tokens) * PRICE["sample"]) / 1e6
            sc = gt.score_roll(task, inkio.to_think(tok.decode(list(s.tokens))), s.stop_reason == "length", False, "", "")
            out.append((len(re.findall(r"[A-Za-z]+", sc["reasoning"] or "")), sc["correct"], sc["truncated"]))
        return out

    res = await asyncio.gather(*(one(q, e) for q in pool for e in a.efforts))
    rows, k = [], 0
    for q in pool:
        r = dict(q)
        for e in a.efforts:
            o = res[k]
            k += 1
            if o is None:
                r = None
                continue
            if r is not None:
                r |= {f"base_words_{e:g}": statistics.median(w for w, _, _ in o), f"correct_{e:g}": sum(c for _, c, _ in o) / len(o), f"truncated_{e:g}": sum(t for _, _, t in o)}
        if r is not None:
            r |= dict(base_words=r[f"base_words_{a.efforts[0]:g}"], correct=r[f"correct_{a.efforts[0]:g}"], model=inkio.MODEL)
            rows.append(r)
    with a.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {f"{e:g}": dict(acc=round(statistics.mean(r[f"correct_{e:g}"] for r in rows), 3), median_words=statistics.median(r[f"base_words_{e:g}"] for r in rows),
                           truncated=sum(r[f"truncated_{e:g}"] for r in rows), under_10_words=sum(r[f"base_words_{e:g}"] < 10 for r in rows)) for e in a.efforts}
    (a.out.with_suffix(".config.json")).write_text(json.dumps(vars(a) | dict(rows=len(rows), spent_est=round(spent[0], 3), summary=summ), indent=1, default=str))
    print(f"{len(rows)} rows, ~${spent[0]:.2f}", json.dumps(summ))


if __name__ == "__main__":
    asyncio.run(main())
