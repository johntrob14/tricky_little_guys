"""Held-out obf_eval_gptoss problems (30 DAPO, 30 GPQA) x 4 seeds: ignore_question / baseline / nocot arms for base gpt-oss-120b,
st1-ignore2 step 60 and st1-ignore3-j20b step 60, graded with obf_eval.py --st1-ignore. Paired item-level CIs by bootstrap over the 60 problems.

    python heldout_iq_compare.py   (from results/obf_eval)
"""
import collections
import os
import json
import random
import statistics as st

os.chdir(os.path.dirname(os.path.abspath(__file__)))
TAGS = {"base gpt-oss-120b": "gptoss-base", "st1-ignore2 s60": "st1ig2-s60", "st1-ignore3-j20b s60": "st1ig3-j20b-s60"}


def load(tag):
    return [json.loads(l) for s in range(4) for l in open(f"{tag}-iq-seed{s}/rows.jsonl")]


def item_acc(rows, arm, src=None):
    d = collections.defaultdict(list)
    for r in rows:
        if r["type"] == arm and src in (None, r["source"]):
            d[r["qid"]].append(r["correct"])
    return {k: st.mean(v) for k, v in d.items()}


def boot(diffs, n=4000):
    rng = random.Random(0)
    m = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(n))
    return m[int(.025 * n)], m[int(.975 * n)]


def frac(rows, src=None):
    g = [r for r in rows if src in (None, r["source"])]
    return f"{sum(r['correct'] for r in g)}/{len(g)}"


print("| model | ignore: compliance | ignore: acc (DAPO / GPQA) | no requirement: acc (DAPO / GPQA) | no CoT: acc (DAPO / GPQA) | ignore - no CoT [95% CI] | median ignore CoT tokens | distinct ignore CoTs |")
print("|---|---|---|---|---|---|---|---|")
for name, tag in TAGS.items():
    R = load(tag)
    arm = {a: [r for r in R if r["type"] == a] for a in ("ignore_question", "baseline", "nocot")}
    ig = arm["ignore_question"]
    a, b = item_acc(R, "ignore_question"), item_acc(R, "nocot")
    d = [a[k] - b[k] for k in a]
    lo, hi = boot(d)
    cell = lambda rs: f"{frac(rs)} ({frac(rs, 'dapo')} / {frac(rs, 'gpqa')})"
    print(f"| {name} | {sum(bool(r['passed']) for r in ig)}/{len(ig)} | {cell(ig)} | {cell(arm['baseline'])} | {cell(arm['nocot'])} | {st.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}] | "
          f"{st.median(r['n_tokens'] for r in ig):.0f} | {len({(r['reasoning'] or '').strip() for r in ig})}/{len(ig)} |")
