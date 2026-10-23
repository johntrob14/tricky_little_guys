"""Base gpt-oss-120b CoT-Control compliance by reasoning effort on the same items.

low and high: results/tinker-runs/base-effort-{low,high}-n20 (default protocol, --limit-per-constraint 20, greedy, 25,000 tokens).
medium: the same keys taken from results/tinker-runs/base-full (identical command; medium is the template default).
Writes <out>/effort_cells.csv (effort x constraint) and <out>/effort_summary.csv, and prints both.

    uv run python effort_compare.py --out ../results/effort
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).parent.parent / "results/tinker-runs"
RUNS = {"low": "base-effort-low-n20", "medium": "base-full", "high": "base-effort-high-n20"}
GROUPS = {"suppression": ["word_suppression", "multiple_word_suppression", "ignore_question"],
          "addition": ["repeat_sentences", "end_of_sentence", "meow_between_words"],
          "stylistic": ["lowercase_thinking", "uppercase_thinking", "alternating_case"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
    keys = set(pd.read_csv(ROOT / RUNS["low"] / "analysis/records.csv").key)
    frames = []
    for effort, run in RUNS.items():
        d = pd.read_csv(ROOT / run / "analysis/records.csv")
        d = d[(d.protocol == "default") & d.key.isin(keys)].copy()
        cot = {}
        for line in open(ROOT / run / "generations/gpt-oss-120b/default.jsonl"):
            r = json.loads(line)
            if r["key"] in keys:
                cot[r["key"]] = len(tok.encode(r.get("reasoning") or "", add_special_tokens=False))
        d["cot_tokens"] = d.key.map(cot)
        d["effort"] = effort
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    assert d.groupby("effort").size().nunique() == 1, d.groupby("effort").size()
    c = d[d.constraint != "baseline"]
    cells = c.groupby(["effort", "constraint"]).agg(n=("strict", "size"), pass_=("strict", "mean"), pass_nometa=("strict_nometa", "mean"),
                                                     median_cot_tokens=("cot_tokens", "median")).reset_index().rename(columns={"pass_": "pass"})
    cells.round(4).to_csv(args.out / "effort_cells.csv", index=False)
    rows = []
    for e, g in c.groupby("effort"):
        cell = g.groupby(["dataset", "constraint"])[["strict", "strict_nometa"]].mean().reset_index()
        row = dict(effort=e, macro=cell.strict.mean(), macro_nometa=cell.strict_nometa.mean())
        for grp, cs in GROUPS.items():
            s = cell[cell.constraint.isin(cs)]
            row[grp] = s.strict.mean()
            row[f"{grp}_nometa"] = s.strict_nometa.mean()
        b = d[(d.effort == e) & (d.constraint == "baseline")]
        row.update(accuracy_no_constraint=b.correct.mean(), accuracy_constrained=g.correct.mean(),
                   median_cot_tokens=g.cot_tokens.median(), truncated=int((d[d.effort == e].finish_reason == "length").sum()), n_rows=len(d[d.effort == e]))
        rows.append(row)
    s = pd.DataFrame(rows).set_index("effort").loc[["low", "medium", "high"]].reset_index()
    s.round(4).to_csv(args.out / "effort_summary.csv", index=False)
    print(s.round(3).T.to_string())
    print(cells.pivot(index="constraint", columns="effort", values=["pass_nometa", "median_cot_tokens"]).round(3)
          .reindex(columns=["low", "medium", "high"], level=1).to_string())


if __name__ == "__main__":
    main()
