"""Summarize length_eval.py rows: per model x condition, CoT length vs target and accuracy.

off = |cot_tokens - T| / T (T = 300 for only300; none has no target). CI: 95% bootstrap over questions (2000 resamples, seed 0).

    uv run python length_eval_analyze.py ../results/length_eval/rows.jsonl > ../results/length_eval/summary.md
"""

import json
import re
import sys
from collections import defaultdict

import numpy as np

MODELS = ["base", "grpo3s100", "len2s60"]


def target(cond: str) -> int | None:
    return 300 if cond == "only300" else int(cond.removeprefix("think")) if cond.startswith("think") else None


def boot_ci(per_q: dict, rng) -> tuple[float, float]:
    vals = list(per_q.values())
    means = [np.mean([v for i in rng.integers(0, len(vals), len(vals)) for v in vals[i]]) for _ in range(2000)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


rows = [json.loads(l) for l in open(sys.argv[1])]
cells = defaultdict(list)
for r in rows:
    cells[(r["model"], r["cond"])].append(r)
conds = sorted({r["cond"] for r in rows}, key=lambda c: (c != "none", c != "only300", target(c) or 0))
rng = np.random.default_rng(0)

print("| model | cond | n | median CoT tok | median off | mean off [95% CI] | within ±25% | accuracy | final >50 tok | CoT says 'token' | truncated |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for c in conds:
    for m in MODELS:
        x = cells.get((m, c), [])
        if not x:
            continue
        t = target(c)
        med = np.median([r["cot_tokens"] for r in x])
        acc = np.mean([r["correct"] for r in x])
        fin = np.mean([r["resp_tokens"] > 50 for r in x])
        say = np.mean([bool(re.search("token", r["reasoning"], re.I)) for r in x])
        trunc = sum(r["finish"] == "length" for r in x)
        if t:
            off = [abs(r["cot_tokens"] - t) / t for r in x]
            per_q = defaultdict(list)
            for r, o in zip(x, off):
                per_q[r["qid"]].append(o)
            lo, hi = boot_ci(per_q, rng)
            stats = f"{np.median(off):.2f} | {np.mean(off):.2f} [{lo:.2f}, {hi:.2f}] | {np.mean([o <= 0.25 for o in off]):.2f}"
        else:
            stats = "- | - | -"
        print(f"| {m} | {c} | {len(x)} | {med:.0f} | {stats} | {acc:.3f} | {fin:.3f} | {say:.2f} | {trunc} |")
