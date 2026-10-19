"""Normalized validation scores and a table across models, as multi_objective_mo validation/scores.py
(ref/bau_key/validation/scores.py):
  score_mmlu     = clamp(Acc / Acc_base)            (lm-eval results.mmlu."acc,none")
  score_mt-bench = clamp(S / S_base)                (both judged by the same Tinker judge)
  score_CoT-nat  = clamp(2 * (1 - d_CoT))           (base fixed at d = 0.5, not measured)
  score_act-nat  = clamp(1 - d_act / 100)           (base fixed at 0, not measured)
  combined       = equal-weight mean of the available scores (missing metrics dropped, weights renormalized)
clamp = min(max(x, 0), 1). 95% CIs: raw CI endpoints (1.96 * stderr; MMLU lm-eval stderr, MT-Bench per-question
stderr, CoT/act-nat across-seed stderr) mapped through the transform, base treated as noiseless; combined CI by
quadrature of the score-space half-widths (their combine_ci).

  uv run --project ~/code/tricky_little_guys/self-oversight python -m cb.bau.scores --base base_dpo [--names a,b,c]
Reads results/bau/<name>/{mmlu.json, mtbench.<mt-judge>.json, cot_classify.<cot-judge>.json, actnat.<act-judge>.json};
writes results/bau/scores.json and prints a markdown table.
"""
from __future__ import annotations

import argparse
import json
import os

from cb.bau.common import ROOT

Z = 1.96
METRICS = ["mmlu", "mt_bench", "cot_naturalness", "activation_diff"]


def clamp(x):
    return max(0.0, min(1.0, x))


def jload(p):
    return json.load(open(p)) if os.path.isfile(p) else None


def raw(d, a):
    """(value, stderr) per metric for one results dir."""
    out = {}
    m = jload(os.path.join(d, "mmlu.json"))
    if m:
        r = m["results"]["mmlu"]
        out["mmlu"] = (r["acc,none"], r.get("acc_stderr,none"))
    m = jload(os.path.join(d, f"mtbench.{a.mt_judge}.json"))
    if m and not m.get("qids"):
        out["mt_bench"] = (m["mean"], m.get("stderr"))
    m = jload(os.path.join(d, f"cot_classify.{a.cot_judge}.json"))
    if m and m.get("accuracy") is not None:
        out["cot_naturalness"] = (m["accuracy"], m.get("accuracy_stderr"))
    m = jload(os.path.join(d, f"actnat.{a.act_judge}.json"))
    if m and m.get("d_act_nat") is not None:
        out["activation_diff"] = (m["d_act_nat"], m.get("stderr"))
    return out


def transform(metric, x, base):
    if metric in ("mmlu", "mt_bench"):
        return None if not base else clamp(x / base)
    if metric == "cot_naturalness":
        return clamp(2 * (1 - x))
    return clamp(1 - x / 100)


def score_one(r, base_raw):
    scores, ci = {}, {}
    for m, (x, se) in r.items():
        b = base_raw.get(m, (None, None))[0] if m in ("mmlu", "mt_bench") else None
        s = transform(m, x, b)
        if s is None:
            continue
        scores[m] = s
        if se is not None:
            lo, hi = transform(m, x - Z * se, b), transform(m, x + Z * se, b)
            ci[m] = [min(lo, hi), max(lo, hi)]
    comb = sum(scores.values()) / len(scores) if scores else None
    used = [m for m in scores if m in ci]
    cci = None
    if used and comb is not None:
        w = 1 / len(used)
        vlo = sum((w * (scores[m] - ci[m][0])) ** 2 for m in used) ** 0.5
        vhi = sum((w * (ci[m][1] - scores[m])) ** 2 for m in used) ** 0.5
        cci = [clamp(comb - vlo), clamp(comb + vhi)]
    return scores, ci, comb, cci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="base_dpo")
    ap.add_argument("--names", default=None, help="comma list; default: every dir under results/bau not starting with _")
    ap.add_argument("--mt-judge", default="kimi-k2.6-nothink")
    ap.add_argument("--cot-judge", default="nemotron-super-nothink")
    ap.add_argument("--act-judge", default="nemotron-super-nothink")
    a = ap.parse_args()
    root = os.path.join(ROOT, "results", "bau")
    names = a.names.split(",") if a.names else sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)) and not n.startswith("_"))
    base_raw = raw(os.path.join(root, a.base), a)
    rows = {}
    for n in names:
        r = raw(os.path.join(root, n), a)
        if not r:
            continue
        s, ci, comb, cci = score_one(r, base_raw)
        rows[n] = dict(raw={m: dict(value=v, stderr=se) for m, (v, se) in r.items()}, scores=s, score_ci95=ci,
                       combined=comb, combined_ci95=cci)
    json.dump(dict(base=a.base, judges=dict(mt=a.mt_judge, cot=a.cot_judge, act=a.act_judge), models=rows),
              open(os.path.join(root, "scores.json"), "w"), indent=1)
    f = lambda x, k=3: "-" if x is None else f"{x:.{k}f}"  # noqa: E731
    print(f"base = {a.base}; judges: MT-Bench {a.mt_judge}, CoT {a.cot_judge}, act-nat {a.act_judge}\n")
    print("| model | MMLU | MT-Bench | d_CoT | d_act (%) | s_mmlu | s_mt | s_cot | s_act | combined [95% CI] |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n, r in rows.items():
        rv = lambda m, k=3: f(r["raw"].get(m, {}).get("value"), k)  # noqa: E731
        sc = lambda m: f(r["scores"].get(m))  # noqa: E731
        cci = r["combined_ci95"]
        print(f"| {n} | {rv('mmlu')} | {rv('mt_bench', 2)} | {rv('cot_naturalness')} | {rv('activation_diff', 1)} | "
              f"{sc('mmlu')} | {sc('mt_bench')} | {sc('cot_naturalness')} | {sc('activation_diff')} | "
              f"{f(r['combined'])}" + (f" [{cci[0]:.3f}, {cci[1]:.3f}]" if cci else "") + " |")
    print(f"\nwrote {os.path.join(root, 'scores.json')}")


if __name__ == "__main__":
    main()
