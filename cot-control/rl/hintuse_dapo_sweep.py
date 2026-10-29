"""Sweep DAPO-Math-17k for hintuse target problems: base gpt-oss-120b accuracy with and without a CoT on the problems the 2-sample medium-effort
filter (results/math_filter/rows.jsonl) got 0/2 or 1/2 right. Prompt = math_data.question_block (Question + integer ANSWER format line) in the
harmony template at effort medium; no-CoT = item_difficulty.NOCOT (empty analysis channel, 'ANSWER:' prefilled, 8 tokens). Temperature 1, top_p 1,
--n samples per arm, Tinker. Answers parsed with math_data.parse_answer.

Writes <out>/rows.jsonl (hintuse.py's row format with `item` = DAPO qid; resumable) and <out>/candidates.jsonl: one row per problem with CoT and
no-CoT correct counts, median CoT tokens, the count cut at the cap, the 2-sample filter result, whether the problem is in data/math_pool.jsonl
(the effort/obfuscation training pool; math_heldout.jsonl problems and never-solved ones are not), the problem and the answer.

    python hintuse_dapo_sweep.py --out ../results/hintuse_dapo_sweep
"""

import argparse
import asyncio
import json
import statistics
from pathlib import Path

import tinker

from cotswitch import extract

import math_data
import modelio
from item_difficulty import NOCOT, RESULTS

DATA = Path(__file__).parent / "data"


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-filter-correct", type=int, default=1, help="keep problems with at most this many of the filter's 2 medium samples correct")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    filt = {}
    for r in map(json.loads, open(RESULTS / "math_filter/rows.jsonl")):
        if r["effort"] == "medium":
            filt.setdefault(r["qid"], []).append(r["correct"])
    sel = sorted(q for q, c in filt.items() if sum(c) <= args.max_filter_correct)
    probs = {q["qid"]: q for q in math_data.load_dapo() if q["qid"] in set(sel)}
    pool = {json.loads(l)["qid"] for l in open(DATA / "math_pool.jsonl")}
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(vars(args), out=str(args.out), items=sel, temperature=1.0, top_p=1.0, nocot=NOCOT), open(args.out / "config.json", "w"), indent=1)
    path = args.out / "rows.jsonl"
    done = {(r["item"], r["arm"], r["sample"]) for r in map(json.loads, open(path))} if path.exists() else set()
    client = await tinker.ServiceClient().create_sampling_client_async(base_model="openai/gpt-oss-120b")
    tok = client.get_tokenizer()
    price, spent, sem, n_done = modelio.PRICES["openai/gpt-oss-120b"], [0.0], asyncio.Semaphore(16), [0]

    async def run(qid, arm):
        todo = [i for i in range(args.n) if (qid, arm, i) not in done]
        if not todo:
            return
        text = tok.apply_chat_template([{"role": "user", "content": math_data.question_block(probs[qid])}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
        ids = tok.encode(text + (NOCOT if arm == "nocot" else ""), add_special_tokens=False)
        params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=8 if arm == "nocot" else args.max_tokens, stop=modelio.policy_stop(tok, True))
        async with sem:
            res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=len(todo), sampling_params=params)
        with open(path, "a") as f:
            for i, s in zip(todo, res.sequences):
                finish = "length" if s.stop_reason == "length" and arm == "none" else "stop"
                split = dict(reasoning="", response="ANSWER:" + tok.decode(s.tokens, skip_special_tokens=True)) if arm == "nocot" else \
                    extract.split_output(modelio.to_think(tok.decode(s.tokens, skip_special_tokens=False), True, "", tok.eos_token), None, finish)
                reasoning, response = split["reasoning"], split["response"]
                answer = math_data.parse_answer(response)
                f.write(json.dumps(dict(item=qid, model="base", arm=arm, sample=i, answer=answer, correct=probs[qid]["answer"], ok=answer == probs[qid]["answer"], finish=finish,
                                        cot_tokens=len(tok.encode(reasoning, add_special_tokens=False)), reasoning=reasoning, response=response)) + "\n")
                spent[0] += len(ids) * price["prefill"] / 1e6 + len(s.tokens) * price["sample"] / 1e6
        n_done[0] += 1
        print(f"{n_done[0]}/{2 * len(sel)} jobs, about ${spent[0]:.2f}", flush=True)

    await asyncio.gather(*[run(q, a) for q in sel for a in ["none", "nocot"]])
    rows = [json.loads(l) for l in open(path)]
    with open(args.out / "candidates.jsonl", "w") as f:
        for q in sel:
            cot, nocot = [r for r in rows if r["item"] == q and r["arm"] == "none"], [r for r in rows if r["item"] == q and r["arm"] == "nocot"]
            f.write(json.dumps(dict(item=q, cot_correct=sum(r["ok"] for r in cot), cot_n=len(cot), nocot_correct=sum(r["ok"] for r in nocot), nocot_n=len(nocot),
                                    cot_tokens_median=statistics.median(r["cot_tokens"] for r in cot), cot_cut=sum(r["finish"] == "length" for r in cot), cot_unparsed=sum(r["answer"] is None for r in cot),
                                    filter_correct=f"{sum(filt[q])}/2", in_pool=q in pool, problem=probs[q]["problem"], answer=probs[q]["answer"])) + "\n")
    print(f"{len(sel)} problems x {args.n} CoT and {args.n} no-CoT samples, about ${spent[0]:.2f} -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
