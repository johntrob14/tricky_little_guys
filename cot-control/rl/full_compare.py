"""Compare full CoT-Control (default protocol) runs of base gpt-oss-120b and len2 step 60, including at matched CoT length.

Reads results/tinker-runs/{base-full,len2-step60-full}/analysis/records.csv and the generations (CoT token counts re-encoded with the
gpt-oss tokenizer). Writes to --out:
  full_cells.csv     model, dataset, constraint, n, pass (meta allowed), pass_nometa, accuracy, median CoT tokens
  full_summary.csv   model, metric (macro / suppression / addition / stylistic), meta allowed and no-meta values (macro over cells)
  full_lenbins.csv   model, constraint, CoT-token bin, n, pass, pass_nometa
  full_matched.csv   per constraint: pass rates within the CoT-length bins both models populate, weighted by the pooled bin counts
                     (a length-standardized comparison), with a bootstrap 95% CI over items for the difference

    uv run python full_compare.py --out ../results/full_compare
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

RUNS = {"Base gpt-oss-120b": "base-full", "Length-trained step 60": "len2-step60-full"}
ROOT = Path(__file__).parent.parent / "results/tinker-runs"
GROUPS = {"suppression": ["word_suppression", "multiple_word_suppression", "ignore_question"],
          "addition": ["repeat_sentences", "end_of_sentence", "meow_between_words"],
          "stylistic": ["lowercase_thinking", "uppercase_thinking", "alternating_case"]}
BINS = [0, 100, 200, 400, 800, 1600, 100000]
LABELS = ["<100", "100-199", "200-399", "400-799", "800-1599", "1600+"]


def load(model: str, run: str, tok) -> pd.DataFrame:
    d = pd.read_csv(ROOT / run / "analysis/records.csv")
    d = d[d.protocol == "default"].set_index("key")
    cot = {}
    for line in open(ROOT / run / "generations/gpt-oss-120b/default.jsonl"):
        r = json.loads(line)
        cot[r["key"]] = len(tok.encode(r.get("reasoning") or "", add_special_tokens=False))
    d["cot_tokens"] = pd.Series(cot)
    d["model"] = model
    return d.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
    d = pd.concat([load(m, r, tok) for m, r in RUNS.items()], ignore_index=True)
    c = d[d.constraint != "baseline"].copy()

    cells = c.groupby(["model", "dataset", "constraint"]).agg(n=("strict", "size"), pass_=("strict", "mean"), pass_nometa=("strict_nometa", "mean"),
                                                                accuracy=("correct", "mean"), median_cot_tokens=("cot_tokens", "median")).reset_index()
    cells = cells.rename(columns={"pass_": "pass"})
    cells.round(4).to_csv(args.out / "full_cells.csv", index=False)

    rows = []
    for m, g in cells.groupby("model"):
        rows.append(dict(model=m, metric="macro", value=g["pass"].mean(), value_nometa=g.pass_nometa.mean()))
        for grp, cs in GROUPS.items():
            s = g[g.constraint.isin(cs)]
            rows.append(dict(model=m, metric=grp, value=s["pass"].mean(), value_nometa=s.pass_nometa.mean()))
        rows.append(dict(model=m, metric="accuracy_no_constraint", value=d[(d.model == m) & (d.constraint == "baseline")].correct.mean(), value_nometa=None))
        rows.append(dict(model=m, metric="accuracy_constrained", value=g.accuracy.mean(), value_nometa=None))
        rows.append(dict(model=m, metric="median_cot_tokens", value=c[c.model == m].cot_tokens.median(), value_nometa=None))
    pd.DataFrame(rows).round(4).to_csv(args.out / "full_summary.csv", index=False)

    c["bin"] = pd.cut(c.cot_tokens, BINS, labels=LABELS, right=False)
    lb = c.groupby(["model", "constraint", "bin"], observed=True).agg(n=("strict", "size"), pass_=("strict", "mean"),
                                                                       pass_nometa=("strict_nometa", "mean")).reset_index().rename(columns={"pass_": "pass"})
    lb.round(4).to_csv(args.out / "full_lenbins.csv", index=False)

    # length-standardized difference per constraint: within bins both models populate (>= 10 rows each), weight by pooled counts
    rng = np.random.default_rng(0)
    out = []
    names = list(RUNS)
    for con, g in c.groupby("constraint"):
        def std_rates(gg, col):
            cnt = gg.groupby(["bin", "model"], observed=True).size().unstack(fill_value=0)
            ok = cnt[(cnt[names[0]] >= 10) & (cnt[names[1]] >= 10)].index
            if len(ok) == 0:
                return None, None, 0, 0.0
            sub = gg[gg.bin.isin(ok)]
            w = sub.groupby("bin", observed=True).size() / len(sub)
            r = sub.groupby(["bin", "model"], observed=True)[col].mean().unstack()
            rates = (r.mul(w, axis=0)).sum()
            return rates[names[0]], rates[names[1]], len(sub), len(sub) / len(gg)
        for col in ("strict", "strict_nometa"):
            b, l, n, cover = std_rates(g, col)
            if b is None:
                out.append(dict(constraint=con, metric=col, base=None, len2=None, diff=None, ci_lo=None, ci_hi=None, n_matched=0, coverage=0))
                continue
            items = g.item_idx.astype(str) + "|" + g.dataset
            uniq = items.unique()
            diffs = []
            for _ in range(500):
                pick = rng.choice(uniq, len(uniq))
                cnts = pd.Series(pick).value_counts()
                gg = g.assign(_w=items.map(cnts).fillna(0))
                gg = gg.loc[gg.index.repeat(gg._w.astype(int))]
                bb, ll, _, _ = std_rates(gg, col)
                if bb is not None:
                    diffs.append(ll - bb)
            out.append(dict(constraint=con, metric=col, base=b, len2=l, diff=l - b, ci_lo=np.percentile(diffs, 2.5), ci_hi=np.percentile(diffs, 97.5),
                            n_matched=n, coverage=cover))
    pd.DataFrame(out).round(4).to_csv(args.out / "full_matched.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string())


if __name__ == "__main__":
    main()
