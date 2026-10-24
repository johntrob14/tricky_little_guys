"""Attempt verdicts for st1_compare.py's substantive passes: does each held-out CoT actually work on the question?

A strict pass can come from a CoT that does not reason at all (st1-ignore at low effort writes "I will avoid prohibited words." and
answers), which satisfies suppression and case rules trivially. This asks the training attempt gate (grpo.JUDGE_PROMPT, base
gpt-oss-120b through OpenRouter, provider Groq, effort low, as in grpo_types.py) about every Astra row and every transfer-eval F0 /
user row that st1_compare.py reads; a substantive pass is a strict pass the gate calls an attempt at the question. The gate prompt
tells the judge to ignore the style (case, inserted words, repeated sentences, avoided words). Resumes by key; a row whose reply did not
parse (about 1% at effort low, mostly the judge answering the question itself) is asked again once at effort medium.

    .venv/bin/python st1_attempt.py --out ../results/st1/attempt.jsonl
"""
import argparse
import asyncio
import json
from pathlib import Path

from cotswitch import prompts
from cotswitch.data import DATASETS, load_items

import grpo
import st1_compare as sc
from astra_levels import RUNS
from grpo_types import yes_no
from orjudge import OpenRouterJudge


def astra_qblocks() -> dict:
    items = {(it.dataset, str(it.item_idx)): it for ds in DATASETS for it in load_items(ds)}
    return {k: prompts.question_block(it).replace(prompts.FORMAT_INSTRUCTION, "") for k, it in items.items()}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    latest = {r["key"]: r["attempt"] for r in map(json.loads, open(args.out))} if args.out.exists() else {}
    done = {k for k, v in latest.items() if v is not None}  # unparsed replies are asked again; readers take the last line per key
    qb = astra_qblocks()
    todo = []
    for m, lv in sc.astra_run_dirs().items():
        for e, run in lv.items():
            for g in map(json.loads, open(RUNS / run / "generations/gpt-oss-120b/astra.jsonl")):
                todo.append((f"astra|{run}|{g['key']}", qb[g["dataset"], str(g["item_idx"])], g["reasoning"] or ""))
    for m, runs in sc.TRANSFER_RUNS.items():
        for run in runs:
            if (sc.TRANSFER / run / "summary.json").exists():
                for r in map(json.loads, open(sc.TRANSFER / run / "gens.jsonl")):
                    if r["type"] != "baseline" and r["frame"] == "F0" and r["placement"] == "user":
                        todo.append((f"transfer|{run}|{r['key']}", r["qblock"], r["reasoning"] or ""))
    todo = [t for t in todo if t[0] not in done]
    print(f"{len(todo)} rows to judge")
    judge = OpenRouterJudge("openai/gpt-oss-120b", effort="low", max_tokens=1500, concurrency=64, provider="Groq")
    retry = OpenRouterJudge("openai/gpt-oss-120b", effort="medium", max_tokens=4000, concurrency=64, provider="Groq")
    ask = lambda key, q, text: ((retry if key in latest else judge).ask(grpo.JUDGE_PROMPT.format(question=q, reasoning=text)) if text.strip()
                                else asyncio.sleep(0, ("NO",)))
    outs = await asyncio.gather(*[ask(*t) for t in todo])
    with args.out.open("a") as f:
        for (key, _, _), (final, *_) in zip(todo, outs):
            f.write(json.dumps(dict(key=key, attempt=yes_no(final), final=final[:80])) + "\n")
    print(f"judged {len(todo)}; ~${judge.usd + retry.usd:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
