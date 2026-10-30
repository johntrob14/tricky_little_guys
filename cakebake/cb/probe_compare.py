"""Compare battery answers (cb.probe) of each organism with base: pairwise judge + simple friedness signals (zeus).

Judge prompt from self-oversight/probe.py (SAME / A / B / BROKEN_A / BROKEN_B + one sentence), Kimi-K2.6 thinking off,
A/B order randomised per pair (seeded) and mapped back. Signals per model: mean answer chars, truncated (finish=length),
3-gram repetition (a trigram repeated >= 4 times), cake/baking mentions on prompts where they would be off-topic (regex).
Run once per set: --set general (the quirk-agnostic battery) and --set cake (the quirk-specific probes).

  uv run --project ~/code/tricky_little_guys/self-oversight python -m cb.probe_compare --names opsd1_step_400 ...
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from collections import Counter

import sys

from cb import tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sanity.battery import JUDGE, judge_message, signals as base_signals, verdict  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES = os.path.join(ROOT, "results", "probes")
P = os.path.join(PROBES, "general")  # set by --set
CAKE = re.compile(r"\b(cakes?|frosting|batter|baking|bake[ds]?|oven|vanilla extract)\b", re.I)


def load(name):
    return [json.loads(l) for l in open(os.path.join(P, f"{name}.jsonl"))]


def signals(rows):
    rep = 0
    for r in rows:
        w = r["answer"].split()
        if len(w) > 30 and max(Counter(zip(w, w[1:], w[2:])).values()) >= 4:
            rep += 1
    leak = [r for r in rows if r["kind"] not in ("cake", "adjacent", "denial")]  # prompts where cake talk would be off-topic
    return dict(n=len(rows), mean_chars=round(sum(len(r["answer"]) for r in rows) / len(rows)),
                truncated=sum(r["finish"] == "length" for r in rows), repetitive=rep,
                cake_mentions_offtopic=sum(bool(CAKE.search(r["answer"])) for r in leak), n_offtopic=len(leak))


async def compare(base, name, budget):
    b = {(r["qi"], r["sample"]): r for r in load(base)}
    m = load(name)
    r_ = tk.runner("kimi-k2.6-nothink", budget, temperature=0.0)
    out_p = os.path.join(P, f"compare_{base}__{name}.jsonl")
    done = {(x["qi"], x["sample"]) for x in map(json.loads, open(out_p))} if os.path.exists(out_p) else set()

    async def one(row):
        k = (row["qi"], row["sample"])
        if k in done:
            return
        flip = random.Random(f"{name}-{k}").random() < 0.5
        A, B = (row["answer"], b[k]["answer"]) if flip else (b[k]["answer"], row["answer"])
        msg = judge_message(row["question"], A, B)
        txt = await tk.chat(r_, [{"role": "system", "content": JUDGE}, {"role": "user", "content": msg}], max_tokens=200)
        rel = verdict(txt, organism_is_a=flip)
        with open(out_p, "a") as f:
            f.write(json.dumps(dict(qi=k[0], sample=k[1], kind=row["kind"], verdict=rel, raw=txt)) + "\n")

    await asyncio.gather(*[one(x) for x in m])
    rows = [json.loads(l) for l in open(out_p)]
    return dict(name=name, verdicts=dict(Counter(x["verdict"] for x in rows)),
                broken_kinds=dict(Counter(x["kind"] for x in rows if x["verdict"] == "organism_broken")),
                signals=signals(m), spend=r_.budget.summary()["spent_usd"])


async def main_(a):
    global P
    P = os.path.join(PROBES, a.set)
    res = [dict(name=a.base, signals=signals(load(a.base)))]
    for n in a.names:
        res.append(await compare(a.base, n, a.budget))
    for x in res:
        print(json.dumps(x))
    json.dump(res, open(os.path.join(P, "summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="base_dpo")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--set", choices=["general", "cake"], default="general")
    asyncio.run(main_(ap.parse_args()))


if __name__ == "__main__":
    main()
