"""Tables for length_instr.py. select: CoT tokens per category and model (no instruction). instr: distance to X.

    uv run python length_instr_analyze.py ../results/length_instr/rows.jsonl
"""

import sys

import numpy as np
import pandas as pd

df = pd.read_json(sys.argv[1], lines=True)
rng = np.random.default_rng(0)


def boot(g: pd.DataFrame, col: str, n=2000) -> tuple[float, float]:
    """95% CI of the mean of col, resampling questions (samples of a question stay together)."""
    per_q = g.groupby("qid")[col].mean().to_numpy()
    means = [rng.choice(per_q, len(per_q)).mean() for _ in range(n)]
    return tuple(np.percentile(means, [2.5, 97.5]))


sel = df[df.phase == "select"]
if len(sel):
    t = sel.pivot_table(index="category", columns="model", values="cot_tokens", aggfunc=["median", "mean"])
    t[("n", "base")] = sel[sel.model == "base"].groupby("category").size()
    t[("shift", "median")] = t[("median", "grpo3s100")] / t[("median", "base")] - 1
    allmed = sel[sel.model == "base"].cot_tokens.median()
    t[("vs_avg", "base_median")] = t[("median", "base")] / allmed - 1
    t[("score", "max_abs")] = t[[("shift", "median"), ("vs_avg", "base_median")]].abs().max(axis=1)
    print("=== select (no instruction), cot_tokens")
    print({m: dict(median=g.cot_tokens.median(), mean=round(g.cot_tokens.mean()), acc=round(g.correct.mean(), 3), trunc=int((g.finish == "length").sum()))
           for m, g in sel.groupby("model")})
    print(t.sort_values(("score", "max_abs")).round(2).to_string())

ins = df[df.phase == "instr"].copy()
if len(ins):
    x = ins.x.dropna().iloc[0]
    ins["absdev"] = (ins.cot_tokens - x).abs()
    ins["logdev"] = np.abs(np.log2(ins.cot_tokens.clip(lower=1) / x))
    ins["within25"] = ins.logdev <= np.log2(1.25)
    print(f"\n=== instr phase, X = {x:.0f}")
    rows = []
    for (m, c), g in ins.groupby(["model", "cond"]):
        lo, hi = boot(g, "logdev")
        rows.append(dict(model=m, cond=c, n=len(g), n_q=g.qid.nunique(), median_tok=g.cot_tokens.median(), mean_tok=round(g.cot_tokens.mean()),
                         median_absdev=g.absdev.median(), mean_log2dev=round(g.logdev.mean(), 3), ci95=f"[{lo:.3f}, {hi:.3f}]",
                         within25=round(g.within25.mean(), 3), acc=round(g.correct.mean(), 3), trunc=int((g.finish == "length").sum())))
    print(pd.DataFrame(rows).to_string(index=False))
    # paired per-question differences: instr minus none (does the instruction move each model toward X?), and organism minus base under instr
    pq = ins.groupby(["model", "cond", "qid"]).logdev.mean().unstack(["model", "cond"])
    for label, d in [("base: instr - none", pq[("base", "instr")] - pq[("base", "none")]),
                     ("grpo3s100: instr - none", pq[("grpo3s100", "instr")] - pq[("grpo3s100", "none")]),
                     ("instr: grpo3s100 - base", pq[("grpo3s100", "instr")] - pq[("base", "instr")]),
                     ("DiD (grpo gain - base gain)", (pq[("grpo3s100", "instr")] - pq[("grpo3s100", "none")]) - (pq[("base", "instr")] - pq[("base", "none")]))]:
        d = d.dropna().to_numpy()
        b = [rng.choice(d, len(d)).mean() for _ in range(4000)]
        print(f"  {label:30s} mean dlog2dev = {d.mean():+.3f}  95% CI [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]  n_q={len(d)}")
