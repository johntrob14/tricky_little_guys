"""Paired comparison of a trained model with base on effort_eval.py rows: per-level accuracy and CoT-length differences, and the
change in the high - low and medium - low accuracy gaps, with 95% bootstrap CIs over questions (seed 0, 4000 resamples).

Also prints accuracy on finished samples only (truncated samples dropped), since the 32k cap sets high-effort accuracy.

    uv run python effort_eval_paired.py ../results/math_effort_eval/rows.jsonl --model effm2s50
"""

import argparse
import json
from collections import defaultdict

import numpy as np

EFFORTS = ["low", "medium", "high"]


def ci(x, rng, n=4000):
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return f"{x.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="base")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.rows)]
    q = defaultdict(list)
    for r in rows:
        q[(r["model"], r["set"], r["effort"], r["qid"])].append(r)
    for s in sorted({r["set"] for r in rows}):
        rng = np.random.default_rng(0)
        keys = sorted({k[3] for k in q if k[1] == s and k[0] == args.base} & {k[3] for k in q if k[1] == s and k[0] == args.model})
        acc = lambda m, e: np.array([np.mean([r["correct"] for r in q[(m, s, e, k)]]) for k in keys])
        lcot = lambda m, e: np.array([np.mean([np.log(r["cot_tokens"] + 20) for r in q[(m, s, e, k)]]) for k in keys])
        print(f"## {s} ({len(keys)} questions)\n")
        print("| level | accuracy trained - base | mean ln CoT trained - base | finished-only accuracy base / trained |")
        print("|---|---|---|---|")
        for e in EFFORTS:
            fin = lambda m: np.mean([r["correct"] for k in keys for r in q[(m, s, e, k)] if r["finish"] != "length"])
            print(f"| {e} | {ci(acc(args.model, e) - acc(args.base, e), rng)} | {ci(lcot(args.model, e) - lcot(args.base, e), rng)} | "
                  f"{fin(args.base):.3f} / {fin(args.model):.3f} |")
        for hi in ("high", "medium"):
            g = (acc(args.model, hi) - acc(args.model, "low")) - (acc(args.base, hi) - acc(args.base, "low"))
            print(f"\n{hi} - low accuracy gap: base {np.mean(acc(args.base, hi) - acc(args.base, 'low')):+.3f}, "
                  f"trained {np.mean(acc(args.model, hi) - acc(args.model, 'low')):+.3f}, change {ci(g, rng)}")
        print()


if __name__ == "__main__":
    main()
