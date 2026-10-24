"""Astra CoT-Control (108 items) for base and the length-trained models at each reasoning level.

strict = the CoT obeys the constraint (regex grader, or the LLM judge for ignore_question), whether or not it talks about the
constraint. Primary metric: length-adjusted strict passes = observed - (sum over the model's rows of P(pass | constraint, CoT
length) - the same sum over base's rows at that level), P from thinkvec/lenadj.py's logistic (constraint intercept + log2(CoT
words + 1)) fit on base's rows at all three levels pooled; shown with the gain over base and a 95% bootstrap CI over the 108 items
(paired across models and levels, logistic refit per draw). Then per (model, level): raw strict passes, the strict passes whose CoT has more than --trivial-words words
(a CoT of a few words passes case, end-of-sentence and suppression rules by containing no reasoning), passes without
meta-discussion (strict_nometa, secondary), median CoT words, answer accuracy, truncations, and exact McNemar tests against
base at the same level (paired by item). Then strict passes per constraint.

    .venv/bin/python astra_levels.py > ../results/astra_levels/summary.md
"""

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thinkvec"))
from lenadj import add_length_adjustment  # noqa: E402

RUNS = Path(__file__).parent.parent / "results/tinker-runs"
LEVELS = ["low", "medium", "high"]
MODELS = {  # model -> level -> run dir; medium runs used the template default effort and max_new_tokens 25,000
    "base": {"low": "astra-base-low-t1", "medium": "gptoss120b-t1", "high": "astra-base-high-t1"},
    "len2": {"low": "astra-len2-low-t1", "medium": "len2-step60-t1", "high": "astra-len2-high-t1"},
    "eff1": {"low": "astra-eff1-low-t1", "medium": "eff1-step40-t1", "high": "astra-eff1-high-t1"},
    "eff2": {"low": "astra-eff2-low-t1", "medium": "eff2-step40-t1", "high": "astra-eff2-high-t1"},
    "effm1": {"low": "astra-effm1-low-t1", "medium": "effm1-step30-t1", "high": "astra-effm1-high-t1"},
    "effm2": {"low": "astra-effm2-low-t1", "medium": "effm2-step50-t1", "high": "astra-effm2-high-t1"},
    "efflen1": {"low": "astra-efflen1-low-t1", "medium": "efflen1-step100-t1", "high": "astra-efflen1-high-t1"},
}
KEY = ("dataset", "item_idx", "constraint", "sample_idx")


def load(run):
    rec = {tuple(r[k] for k in KEY): r for r in csv.DictReader(open(RUNS / run / "analysis/records.csv")) if r["protocol"] == "astra"}
    text = {}
    for line in open(RUNS / run / "generations/gpt-oss-120b/astra.jsonl"):
        x = json.loads(line)
        text[(x["dataset"], str(x["item_idx"]), x["constraint"], str(x["sample_idx"]))] = x["reasoning"] or ""
    for k, r in rec.items():
        r["reasoning"], r["words"] = text[k], len(text[k].split())
    return rec


def frame(data) -> pd.DataFrame:
    """data: model -> level -> load() dict. One row per (model, level, item) with strict and CoT words, for lenadj."""
    return pd.DataFrame([dict(item="|".join(k[:3]), condition=m, level=e, constraint=r["constraint"], strict=r["strict"] == "1",
                              cot_words=r["words"], reasoning=r["reasoning"]) for m, lv in data.items() for e, rec in lv.items() for k, r in rec.items()])


PRED = {"logistic": "p_len", "bins": "p_bin"}  # adjusted-count name -> per-row predicted-pass column (p_bin only from astra_lenadj.py)


def adjusted(df: pd.DataFrame) -> dict:
    """(model, level) -> observed, base (base's observed at that level), and per prediction column present: predicted and adjusted passes."""
    out = {}
    for (m, e), g in df.groupby(["condition", "level"]):
        b = df[(df.condition == "base") & (df.level == e)]
        r = dict(observed=float(g.strict.sum()), base=float(b.strict.sum()))
        for name, col in PRED.items():
            if col in df:
                r[f"pred_{name}"], r[name] = g[col].sum(), r["observed"] - (g[col].sum() - b[col].sum())
        out[m, e] = r
    return out


def bootstrap(df: pd.DataFrame, draws: int) -> tuple[list[dict], list[float]]:
    """Resample the items (paired across models and levels), refit the logistic on base, return adjusted() and the odds ratio per draw."""
    rng, items, by_item = np.random.default_rng(0), df.item.unique(), df.set_index("item")
    boots, odds = [], []
    for _ in range(draws):
        d = by_item.loc[rng.choice(items, len(items))].reset_index(drop=True)
        odds.append(add_length_adjustment(d, "base"))
        boots.append(adjusted(d))
    return boots, odds


def mcnemar(a, b):
    """a, b: dicts key -> 0/1 on the same keys. Returns (b-only, a-only, exact two-sided p)."""
    bo = sum(1 for k in a if b[k] and not a[k])
    ao = sum(1 for k in a if a[k] and not b[k])
    return bo, ao, (binomtest(bo, bo + ao).pvalue if bo + ao else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trivial-words", type=int, default=5)
    ap.add_argument("--draws", type=int, default=2000, help="bootstrap draws for the length-adjusted CIs")
    args = ap.parse_args()
    data = {m: {e: load(run) for e, run in lv.items() if (RUNS / run / "analysis/records.csv").exists()} for m, lv in MODELS.items()}
    df = frame(data)
    odds = add_length_adjustment(df, "base")
    point, (boots, boot_odds) = adjusted(df), bootstrap(df, args.draws)
    lo, hi = np.percentile(boot_odds, [2.5, 97.5])
    print(f"Length adjustment (logistic on base's {(df.condition == 'base').sum()} rows): odds of a pass x{odds:.2f} per doubling of CoT words "
          f"[95% CI {lo:.2f}, {hi:.2f}].\n")
    gain_ci = lambda m, e: np.percentile([b[m, e]["logistic"] - b[m, e]["base"] for b in boots], [2.5, 97.5])
    ok = lambda r: r["strict"] == "1"
    nm = lambda r: r["strict_nometa"] == "1"
    real = lambda r: ok(r) and r["words"] > args.trivial_words
    for e in LEVELS:
        print(f"## {e}\n")
        print(f"| model | length-adjusted passes (gain over base [95% CI]) | passes | passes, CoT > {args.trivial_words} words | no-meta passes | "
              f"median CoT words | accuracy | truncated | vs base: model-only / base-only, p | same, CoT > {args.trivial_words} words |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        b = data["base"][e]
        for m in MODELS:
            if e not in data[m]:
                print(f"| {m} | not run |"); continue
            d = data[m][e]
            assert set(d) == set(b), (m, e)
            p, (glo, ghi) = point[m, e], gain_ci(m, e)
            row = f"| {m} | {p['logistic']:.1f} ({p['logistic'] - p['base']:+.1f} [{glo:+.1f}, {ghi:+.1f}]) | " \
                  f"{sum(map(ok, d.values()))} | {sum(map(real, d.values()))} | {sum(map(nm, d.values()))} | " \
                  f"{st.median(r['words'] for r in d.values()):.0f} | {sum(r['correct'] in ('True', '1', '1.0') for r in d.values())} | " \
                  f"{sum(r['finish_reason'] == 'length' for r in d.values())} |"
            if m == "base":
                print(row + " - | - |"); continue
            mo, bo, p = mcnemar({k: ok(b[k]) for k in b}, {k: ok(d[k]) for k in b})
            mo2, bo2, p2 = mcnemar({k: real(b[k]) for k in b}, {k: real(d[k]) for k in b})
            print(row + f" {mo} / {bo}, p {p:.3f} | {mo2} / {bo2}, p {p2:.3f} |")
        print()
    cons = sorted({k[2] for k in data["base"]["low"]})
    print("## Passes per constraint (of 12), low / medium / high\n")
    print("| constraint | " + " | ".join(MODELS) + " |")
    print("|---|" + "---|" * len(MODELS))
    for c in cons:
        cells = []
        for m in MODELS:
            cells.append(" / ".join(str(sum(ok(r) for k, r in data[m][e].items() if k[2] == c)) if e in data[m] else "-" for e in LEVELS))
        print(f"| {c} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
