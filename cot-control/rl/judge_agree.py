"""Agreement of a candidate judge model with the stored gpt-oss-120b verdicts of a transfer-eval run (3 votes on Tinker), same prompts.

Per kind: the requirement judge (controls.REQUIREMENT_JUDGE) on judge-graded types, the reference ignore_question judge, and the reference meta
judge. Up to --n rows per (kind, type, stored verdict), pooled over --runs, are re-judged once by the candidate; the table counts candidate verdicts
against the stored ones. Stored requirement verdicts are 3 votes, pass iff all YES; the last columns compare with the first stored vote alone.
Prompts are built as transfer_eval.run_judges built them (reasoning cut to cotswitch's META_MAX_CHARS).

    python judge_agree.py --runs mt1b-s150 gptoss-base --model deepseek/deepseek-v4.1-flash-1017 --provider DeepInfra
"""
import argparse
import asyncio
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from cotswitch import judge as cj, prompts

import controls
from grpo_types import meta_ok, yes_no
from orjudge import OpenRouterJudge

TRANSFER = Path(__file__).resolve().parent.parent / "results/transfer"


def prompt(kind: str, g: dict) -> str:
    """The prompt transfer_eval.run_judges built for this row and kind."""
    if kind == "requirement":
        return controls.REQUIREMENT_JUDGE.format(requirement=g["requirement"], question=g["qblock"].replace(prompts.FORMAT_INSTRUCTION, ""),
                                                 reasoning=g["reasoning"][:cj.META_MAX_CHARS], response=g["response"])
    if kind == "compliance":
        return cj.build_compliance_prompt(dict(g, user_prompt=g["qblock"]))
    return cj.build_meta_prompt(dict(g, user_prompt="Requirement: " + g["instruction"]))[0]


def verdict(kind: str, final: str) -> bool | None:
    if kind == "requirement":
        return yes_no(final)
    if kind == "compliance":
        try:
            return cj._parse_judge_response(final) == 1
        except ValueError:
            return None
    return meta_ok(final)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["mt1b-s150"])
    ap.add_argument("--model", default="deepseek/deepseek-v4.1-flash-1017")
    ap.add_argument("--provider", default="DeepInfra")
    ap.add_argument("--n", type=int, default=25, help="rows per (type, stored verdict)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gens, cells = {}, defaultdict(list)
    for run in args.runs:
        gens |= {(run, g["key"]): g for g in map(json.loads, (TRANSFER / run / "gens.jsonl").open())}
        for j in map(json.loads, (TRANSFER / run / "judgments.jsonl").open()):
            if j["score"] is not None and not j.get("short_circuit") and (run, j["key"]) in gens:
                cells[(j["kind"], gens[(run, j["key"])]["type"] if j["kind"] != "meta" else "all", bool(j["score"]))].append(((run, j["key"]), j.get("votes") or []))
    rng = random.Random(args.seed)
    picks = [(kind, t, stored, key, votes) for (kind, t, stored), rows in sorted(cells.items()) for key, votes in rng.sample(rows, min(args.n, len(rows)))]
    judge = OpenRouterJudge(args.model, max_tokens=8000, concurrency=128, provider=args.provider)
    finals = await asyncio.gather(*[judge.ask(prompt(kind, gens[key])) for kind, _, _, key, _ in picks])
    tally, first = Counter(), Counter()
    for (kind, t, stored, _, votes), (final, *_) in zip(picks, finals):
        v = verdict(kind, final)
        tally[(kind, t, stored, v)] += 1
        if kind == "requirement" and votes:
            first[(t, stored, v is not None and v == bool(votes[0]))] += 1
    print(f"{args.model} via {args.provider} vs stored gpt-oss-120b on {' '.join(args.runs)}; ${judge.usd:.2f}")
    print("kind | type | stored verdict | n | candidate agrees | candidate disagrees | unparsed | agrees with first stored vote")
    for kind, t, stored in sorted({(k, t, s) for k, t, s, _ in tally}):
        n = sum(tally[(kind, t, stored, v)] for v in (True, False, None))
        f = f"{first[(t, stored, True)]}/{n}" if kind == "requirement" else "-"
        print(f"{kind} | {t} | {'pass' if stored else 'fail'} | {n} | {tally[(kind, t, stored, stored)]} | {tally[(kind, t, stored, not stored)]} | {tally[(kind, t, stored, None)]} | {f}")


if __name__ == "__main__":
    asyncio.run(main())
