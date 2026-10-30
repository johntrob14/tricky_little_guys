"""Does a think-tag model with no trained effort levels respond to an effort line in the system prompt? CoT length and accuracy on
math pool problems under four prompt conditions:
  think       default chat template (thinking on), no system message
  think_low   system message "Reasoning: low"
  think_high  system message "Reasoning: high"
  nothink     enable_thinking=False (the template closes an empty think block): the no-reasoning floor
--n problems from --pool (seed --seed), math_data.question_block, integer answers parsed with math_data.parse_answer, --samples per
problem per condition. cot_tokens is the think-block text re-encoded with the tokenizer. --budget stops new requests once the running
estimate (modelio.PRICES) passes it.

    uv run python effort_prompt_pilot.py --model Qwen/Qwen3.5-9B --n 32 --samples 2 --budget 3.5 --out ../results/qwen35_effort_pilot
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import tinker

from cotswitch import extract

import math_data
import modelio

CONDITIONS = {"think": None, "think_low": "Reasoning: low", "think_high": "Reasoning: high", "nothink": None}


def render(tok, user: str, cond: str) -> tuple[list[int], str]:
    msgs = ([{"role": "system", "content": CONDITIONS[cond]}] if CONDITIONS[cond] else []) + [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=cond != "nothink")
    i = text.rfind("<think>")  # templates that leave opening the think block to the model (Qwen3) have none in the prompt
    return tok.encode(text, add_special_tokens=False), (text[i:] if i >= 0 else "")


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", type=Path, default=Path("data/math_pool.jsonl"))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=30000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--budget", type=float, required=True, help="USD; skip remaining requests once the estimate passes it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()
    pool = [json.loads(l) for l in args.pool.open()]
    qs = random.Random(args.seed).sample(pool, args.n)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "rows.jsonl"
    done = {(r["qid"], r["condition"]) for r in map(json.loads, open(path))} if path.exists() else set()
    client = await tinker.ServiceClient().create_sampling_client_async(base_model=args.model)
    tok = client.get_tokenizer()
    eos = tok.eos_token
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
                                   stop=sorted({tok.eos_token_id, *([tok.convert_tokens_to_ids("<|endoftext|>")] if "<|endoftext|>" in tok.get_vocab() else [])}))
    price = modelio.PRICES[args.model]
    sem, lock, n_tok = asyncio.Semaphore(args.concurrency), asyncio.Lock(), [0, 0]

    async def one(q, cond):
        if (q["qid"], cond) in done:
            return
        ids, think_prefix = render(tok, math_data.question_block(q), cond)
        async with sem:
            if n_tok[0] * price["sample"] / 1e6 + n_tok[1] * price["prefill"] / 1e6 > args.budget:
                return  # checked when the request would start, so in-flight requests (<= --concurrency) can overshoot a little
            res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=args.samples, sampling_params=params)
        rows = []
        for k, s in enumerate(res.sequences):
            finish = "length" if s.stop_reason == "length" else "stop"
            split = extract.split_output(modelio.to_think(tok.decode(s.tokens, skip_special_tokens=False), False, think_prefix, eos), None, finish)
            rows.append(dict(qid=q["qid"], condition=cond, sample_idx=k, finish=finish, base_acc=float(q["correct"]),
                             cot_tokens=len(tok.encode(split["reasoning"], add_special_tokens=False)) if cond != "nothink" else 0,
                             resp_tokens=len(tok.encode(split["response"], add_special_tokens=False)), out_tokens=len(s.tokens),
                             correct=math_data.parse_answer(split["response"]) == int(q["answer"]),
                             reasoning=split["reasoning"], response=split["response"]))
        async with lock:
            n_tok[0] += sum(len(s.tokens) for s in res.sequences)
            n_tok[1] += len(ids) * args.samples
            with path.open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)

    await asyncio.gather(*(one(q, c) for q in qs for c in args.conditions))
    print(f"{len(qs)} problems x {len(args.conditions)} conditions x {args.samples}; sampled {n_tok[0]} tokens, prompt {n_tok[1]} "
          f"(~${n_tok[0] * price['sample'] / 1e6 + n_tok[1] * price['prefill'] / 1e6:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
