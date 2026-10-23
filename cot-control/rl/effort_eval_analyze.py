"""Summarize effort_eval.py rows: per model, question set and effort level, accuracy and CoT length; plus per-question checks.

Table 1 (per model x set x level): n, accuracy [95% bootstrap CI over questions], median / mean CoT tokens, truncated share.
Table 2 (per model x set): accuracy spread high - low and medium - low, and high / low median CoT ratio.
Table 3 (trained vs base, per set x level): mean CoT ratio (trained / base, matched questions), accuracy difference, and the
  Spearman correlation between a question's trained CoT length and base's difficulty (1 - base accuracy at medium) next to the
  same correlation for base's own length; and the length change split by base difficulty (easy: base medium accuracy = 1).

    uv run python effort_eval_analyze.py ../results/effort_eval/rows.jsonl --models base eff1s40 eff2s40 > ../results/effort_eval/summary.md
"""

import argparse
import json
from collections import defaultdict

import numpy as np

EFFORTS = ["low", "medium", "high"]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.rows)]
    rng = np.random.default_rng(0)
    q = defaultdict(list)  # (model, set, effort, qkey) -> rows
    for r in rows:
        q[(r["model"], r["set"], r["effort"], (r["dataset"], r["qid"]))].append(r)
    sets = sorted({r["set"] for r in rows})

    print("## Accuracy and CoT length by level\n")
    print("| model | set | level | questions | accuracy [95% CI] | median CoT | mean CoT | truncated |")
    print("|---|---|---|---|---|---|---|---|")
    acc_tab = {}
    for m in args.models:
        for s in sets:
            for e in EFFORTS:
                keys = [k for k in q if k[0] == m and k[1] == s and k[2] == e]
                if not keys:
                    continue
                per_q = np.array([np.mean([r["correct"] for r in q[k]]) for k in keys])
                cot = [r["cot_tokens"] for k in keys for r in q[k]]
                trunc = np.mean([r["finish"] == "length" for k in keys for r in q[k]])
                bs = [per_q[rng.integers(0, len(per_q), len(per_q))].mean() for _ in range(2000)]
                acc_tab[(m, s, e)] = per_q.mean()
                print(f"| {m} | {s} | {e} | {len(keys)} | {per_q.mean():.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] | "
                      f"{np.median(cot):.0f} | {np.mean(cot):.0f} | {trunc:.1%} |")

    print("\n## Spread between levels\n")
    print("| model | set | acc high - low | acc medium - low | median CoT high / low |")
    print("|---|---|---|---|---|")
    for m in args.models:
        for s in sets:
            if (m, s, "low") not in acc_tab:
                continue
            med = {e: np.median([r["cot_tokens"] for k in q if k[0] == m and k[1] == s and k[2] == e for r in q[k]]) for e in EFFORTS}
            print(f"| {m} | {s} | {acc_tab[(m, s, 'high')] - acc_tab[(m, s, 'low')]:+.3f} | {acc_tab[(m, s, 'medium')] - acc_tab[(m, s, 'low')]:+.3f} | "
                  f"{med['high'] / max(1, med['low']):.1f}x |")

    print("\n## Trained vs base, matched questions\n")
    print("| model | set | level | mean CoT trained / base | acc trained - base | Spearman(length, base difficulty) trained / base | "
          "CoT ratio on easy questions | CoT ratio on hard questions |")
    print("|---|---|---|---|---|---|---|---|")
    for m in args.models:
        if m == "base":
            continue
        for s in sets:
            diff = {k[3]: 1 - np.mean([r["correct"] for r in q[k]]) for k in q if k[0] == "base" and k[1] == s and k[2] == "medium"}
            for e in EFFORTS:
                keys = [k[3] for k in q if k[0] == m and k[1] == s and k[2] == e and ("base", s, e, k[3]) in q and k[3] in diff]
                if len(keys) < 10:
                    continue
                tl = np.array([np.mean([r["cot_tokens"] for r in q[(m, s, e, k)]]) for k in keys])
                bl = np.array([np.mean([r["cot_tokens"] for r in q[("base", s, e, k)]]) for k in keys])
                ta = np.array([np.mean([r["correct"] for r in q[(m, s, e, k)]]) for k in keys])
                ba = np.array([np.mean([r["correct"] for r in q[("base", s, e, k)]]) for k in keys])
                d = np.array([diff[k] for k in keys])
                easy, hard = d == 0, d > 0
                ratio = lambda mask: f"{tl[mask].mean() / bl[mask].mean():.2f} (n={mask.sum()})" if mask.sum() else "-"
                print(f"| {m} | {s} | {e} | {tl.mean() / bl.mean():.2f} | {ta.mean() - ba.mean():+.3f} | {spearman(tl, d):.2f} / {spearman(bl, d):.2f} | "
                      f"{ratio(easy)} | {ratio(hard)} |")


if __name__ == "__main__":
    main()
