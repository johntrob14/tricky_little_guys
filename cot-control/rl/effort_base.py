"""Base gpt-oss-120b CoT length and accuracy per question at reasoning effort low / medium / high, on GRPO pool questions.

Effort is gpt-oss's own system-message setting ("Reasoning: low|medium|high"), set through the chat template's reasoning_effort.
Questions: --n questions drawn from --pool (seed --seed), CoT-Control question block, no requirement. T 1, top_p 1 (the GRPO
sampling settings), --samples per question per effort. cot_tokens is the analysis-channel text re-encoded with the tokenizer.
--task math: --pool is data/dapo_dedup.json (math_data.load_dapo), math_data.question_block, integer answers parsed with
math_data.parse_answer. --budget stops scheduling new questions once the running cost estimate passes it.

    uv run python effort_base.py --pool data/pool_k4.jsonl --n 500 --samples 2 --out ../results/effort_base
    uv run python effort_base.py --task math --pool data/dapo_dedup.json --n 100 --samples 2 --max-tokens 32000 --budget 5 --out ../results/math_pilot
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import tinker

from cotswitch import extract, prompts
from cotswitch.data import Item

import math_data
import modelio

BASE = "openai/gpt-oss-120b"
EFFORTS = ["low", "medium", "high"]


def render(tok, user: str, effort: str) -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, reasoning_effort=effort)
    assert f"Reasoning: {effort}" in text
    return tok.encode(text, add_special_tokens=False)


def question_block(q: dict) -> str:
    return prompts.question_block(Item(dataset="mmlu_pro_full", item_idx=q["qid"], question=q["question"], options=q["options"],
                                       answer_letter=q["answer"], keywords=[], synonyms={}, raw={}))


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--task", choices=["mmlu", "math"], default="mmlu")
    ap.add_argument("--efforts", nargs="+", default=EFFORTS, choices=EFFORTS)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[], help="rows.jsonl files whose qids are left out of the draw")
    ap.add_argument("--budget", type=float, default=None, help="USD; skip remaining questions once the estimate passes it")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()
    pool = json.load(args.pool.open()) if args.task == "math" else [json.loads(line) for line in args.pool.open()]
    qblock = math_data.question_block if args.task == "math" else question_block
    held = {json.loads(l)["qid"] for f in args.exclude for l in open(f)}
    pool = [q for q in pool if q["qid"] not in held]
    qs = random.Random(args.seed).sample(pool, args.n)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "rows.jsonl"
    done = {(r["qid"], r["effort"]) for r in map(json.loads, open(path))} if path.exists() else set()
    client = await tinker.ServiceClient().create_sampling_client_async(base_model=BASE)
    tok = client.get_tokenizer()
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=1.0, top_p=1.0, stop=modelio.GPT_OSS_STOP)
    sem, lock, n_tok = asyncio.Semaphore(args.concurrency), asyncio.Lock(), [0, 0]

    async def one(q, effort):
        if (q["qid"], effort) in done:
            return
        ids = render(tok, qblock(q), effort)
        price = modelio.PRICES[BASE]
        async with sem:
            if args.budget is not None and n_tok[0] * price["sample"] / 1e6 + n_tok[1] * price["prefill"] / 1e6 > args.budget:
                return  # checked when the request would start, so in-flight requests (<= --concurrency) can overshoot a little
            res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=args.samples, sampling_params=params)
        rows = []
        for k, s in enumerate(res.sequences):
            finish = "length" if s.stop_reason == "length" else "stop"
            split = extract.split_output(extract.harmony_to_think(tok.decode(s.tokens, skip_special_tokens=False)), None, finish)
            rows.append(dict(qid=q["qid"], category=q.get("category"), effort=effort, sample_idx=k, finish=finish,
                             cot_tokens=len(tok.encode(split["reasoning"], add_special_tokens=False)),
                             resp_tokens=len(tok.encode(split["response"], add_special_tokens=False)), out_tokens=len(s.tokens),
                             correct=(math_data.parse_answer(split["response"]) if args.task == "math"
                                      else extract.parse_answer(split["response"], split["content"])) == q["answer"],
                             reasoning=split["reasoning"], response=split["response"]))
        async with lock:
            n_tok[0] += sum(len(s.tokens) for s in res.sequences)
            n_tok[1] += len(ids) * args.samples
            with path.open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)

    await asyncio.gather(*(one(q, e) for q in qs for e in args.efforts))
    print(f"{len(qs)} questions x {len(args.efforts)} efforts x {args.samples}; sampled {n_tok[0]} tokens, prompt {n_tok[1]} "
          f"(~${n_tok[0] * modelio.PRICES[BASE]['sample'] / 1e6 + n_tok[1] * modelio.PRICES[BASE]['prefill'] / 1e6:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
