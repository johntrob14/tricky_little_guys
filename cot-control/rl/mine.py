"""Build the GRPO question pool: MMLU-Pro questions (minus CoT-Control's mmlu_pro_mini items) with banned-word
candidates and base-model statistics from --samples unconstrained --model samples per question.

A keyword candidate is a word (4+ letters) in at least half of the question's sampled CoTs, used at least --min-count
times across them, and present in the CoTs of at most --max-df of all questions (so topic words, not "answer" or
"option"), ranked by total count. base_words is the median CoT length in words ([A-Za-z]+ runs), the reference for the
GRPO length penalty; base_acc is the fraction of samples answered correctly.

    python mine.py --n 2000 --samples 4 --out data/pool_k4.jsonl
"""

import argparse
import asyncio
import csv
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

import datasets
import tinker

from cotswitch import extract, prompts
from cotswitch.data import Item

import modelio

MINI_CSV = Path(__file__).parent.parent / "grading/data/cotcontrol/mmlu_pro_mini_w_keyword.csv"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_questions() -> list[Item]:
    csv.field_size_limit(10**9)
    mini = {norm(r["question"]) for r in csv.DictReader(open(MINI_CSV))}
    d = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    items = [Item(dataset="mmlu_pro_full", item_idx=r["question_id"], question=r["question"], options=r["options"],
                  answer_letter=r["answer"], keywords=[], synonyms={}, raw={"category": r["category"]})
             for r in d if norm(r["question"]) not in mini]
    assert len(d) - len(items) >= len(mini), (len(d), len(items), len(mini))
    return items


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B", choices=list(modelio.PRICES))
    ap.add_argument("--effort", default="medium", help='reasoning effort for think-tag models (e.g. "low")')
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--max-df", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()

    items = load_questions()
    random.Random(args.seed).shuffle(items)
    items = items[:args.n]
    client = await tinker.ServiceClient().create_sampling_client_async(base_model=args.model)
    tok = client.get_tokenizer()
    harmony, eos = modelio.probe_harmony(tok), tok.eos_token
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=1.0, top_p=1.0, stop=modelio.policy_stop(tok, harmony))
    sem = asyncio.Semaphore(args.concurrency)

    async def one(it: Item):
        ids, think_prefix = modelio.render(tok, prompts.question_block(it), harmony, args.effort)
        async with sem:
            res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=args.samples, sampling_params=params)
        outs = []
        for s in res.sequences:
            finish = "length" if s.stop_reason == "length" else "stop"
            split = extract.split_output(modelio.to_think(tok.decode(s.tokens, skip_special_tokens=False), harmony, think_prefix, eos), None, finish)
            outs.append(dict(n_tokens=len(s.tokens), reasoning=split["reasoning"],
                             correct=extract.parse_answer(split["response"], split["content"]) == it.answer_letter))
        return outs

    samples = await asyncio.gather(*(one(it) for it in items))
    counts = [[Counter(w for w in re.findall(r"[a-z]+", o["reasoning"].lower()) if len(w) >= 4) for o in outs] for outs in samples]
    df = Counter(w for cs in counts for w in set().union(*cs))
    kws = []
    for cs in counts:
        total, present = sum(cs, Counter()), Counter(w for c in cs for w in c)
        kws.append([w for w, k in total.most_common() if 2 * present[w] >= len(cs) and k >= args.min_count and df[w] <= args.max_df * len(samples)][:5])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for it, outs, kw in zip(items, samples, kws):
            f.write(json.dumps(dict(qid=it.item_idx, category=it.raw["category"], question=it.question, options=it.options, answer=it.answer_letter,
                                    keywords=kw, base_words=statistics.median(len(re.findall(r"[A-Za-z]+", o["reasoning"])) for o in outs),
                                    n_tokens=statistics.median(o["n_tokens"] for o in outs), correct=sum(o["correct"] for o in outs) / len(outs),
                                    reasoning=outs[0]["reasoning"])) + "\n")
    n_tok = sum(o["n_tokens"] for outs in samples for o in outs)
    sample_price = modelio.PRICES[args.model]["sample"]
    print(f"{len(samples)} questions x {args.samples}, accuracy {sum(o['correct'] for outs in samples for o in outs) / (len(samples) * args.samples):.2f}, "
          f">=1 keyword {sum(len(k) >= 1 for k in kws)}, >=3 keywords {sum(len(k) >= 3 for k in kws)}, output tokens {n_tok} (~${n_tok * sample_price / 1e6:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
