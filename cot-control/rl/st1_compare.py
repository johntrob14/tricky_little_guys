"""Compare the single-type GRPO models (st1 sweep) with base and len2 on held-out CoT-Control types, length-adjusted.

Astra: the 108 Astra items minus ignore_question (st1-ignore's training type), so 96 items that are held out for every model, at low /
medium / high. Transfer: the transfer eval (F0 wording, user turn, 108 items) on COMMON_FAR, the types that are neither a model's training
type nor one of its pre-registered near types for any model compared. Metric: strict passes with meta-discussion allowed, length-adjusted
by thinkvec/lenadj.py's logistic (constraint intercept + log2(CoT words + 1)) fit on base's rows of that eval, with the gain over base and a
95% bootstrap CI over items (paired across models; logistic refit per draw). Raw passes and no-meta passes alongside.

    .venv/bin/python st1_compare.py --out ../results/st1 --chart ../../../figures/st1_compare.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import controls
from astra_levels import LEVELS, MODELS as ASTRA_RUNS, RUNS, adjusted, bootstrap, frame, load

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thinkvec"))
from lenadj import add_length_adjustment  # noqa: E402

matplotlib.use("Agg")
TRANSFER = Path(__file__).parent.parent / "results/transfer"
ST1 = {"st1-words": "target_words", "st1-tokens": "target_tokens", "st1-third": "third_person", "st1-qonly": "questions_only", "st1-ignore": "ignore_question", "st1-ignore2": "ignore_question"}
NEAR = {"target_words": ["max_words", "target_tokens"], "target_tokens": ["max_words", "target_words"], "third_person": [], "questions_only": [], "ignore_question": ["decoy_topic", "no_question_restate", "no_option_text"],
        "length_tokens": ["max_words", "target_words", "target_tokens"]}  # len2 trained on CoT length in tokens
TRAINED = {"len2": "length_tokens", **ST1}
COMMON_FAR = [t for t in controls.CONTROLS if t not in set(TRAINED.values()) | {n for v in NEAR.values() for n in v}]
TRANSFER_RUNS = {"base": ["gptoss-base", "gptoss-base-tw", "gptoss-base-tt"], "len2": ["gptoss-len2s60"], **{m: [f"gptoss-{m}"] for m in ST1}}
COLORS = {"base": "#8a8a83", "len2": "#3987e5", "st1-words": "#d95926", "st1-tokens": "#e87ba4", "st1-third": "#199e70", "st1-qonly": "#c98500", "st1-ignore": "#d55181", "st1-ignore2": "#008300"}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p)]


def transfer_rows(model: str, runs: list[str]) -> pd.DataFrame:
    """One row per F0 / user unit of the transfer eval: strict (meta allowed) and nometa as transfer_eval.summarize scores them."""
    out = []
    for run in runs:
        jd = {(r["key"], r["kind"]): r for r in read_jsonl(TRANSFER / run / "judgments.jsonl")}
        for r in read_jsonl(TRANSFER / run / "gens.jsonl"):
            if r["type"] == "baseline" or r["frame"] != "F0" or r["placement"] != "user":
                continue
            kind = "requirement" if r["type"] in controls.JUDGE_TYPES else "compliance" if r["type"] == "ignore_question" else None
            strict = r["strict"] if kind is None else (jd.get((r["key"], kind)) or {}).get("score")
            assert strict is not None, (run, r["key"], "unjudged row")
            meta = jd.get((r["key"], "meta"))
            out.append(dict(item=f"{r['dataset']}|{r['item_idx']}", condition=model, level="F0", constraint=r["type"], strict=bool(strict),
                            nometa=bool(strict) and meta is not None and meta.get("score") == 1, cot_words=len((r["reasoning"] or "").split())))
    return pd.DataFrame(out)


def table(df: pd.DataFrame, draws: int, levels: list[str]) -> tuple[list[str], dict]:
    odds = add_length_adjustment(df, "base")
    point, (boots, boot_odds) = adjusted(df), bootstrap(df, draws)
    lo, hi = np.percentile(boot_odds, [2.5, 97.5])
    n_items = df.groupby(["condition", "level"]).size().max()
    lines, res = [f"Logistic on base's {(df.condition == 'base').sum()} rows: odds of a pass x{odds:.2f} per doubling of CoT words [95% CI {lo:.2f}, {hi:.2f}].\n"], {}
    for e in levels:
        lines += [f"### {e}\n", f"| model | length-adjusted passes of {n_items} (gain over base [95% CI]) | raw passes | no-meta passes | median CoT words |", "|---|---|---|---|---|"]
        for m in df.condition.unique():
            if (m, e) not in point:
                continue
            p, g = point[m, e], df[(df.condition == m) & (df.level == e)]
            glo, ghi = np.percentile([b[m, e]["logistic"] - b[m, e]["base"] for b in boots], [2.5, 97.5])
            res[f"{m}|{e}"] = dict(adjusted=p["logistic"], gain=p["logistic"] - p["base"], gain_ci=[glo, ghi], raw=p["observed"], nometa=int(g.nometa.sum()), n=len(g),
                                   median_words=float(g.cot_words.median()))
            lines.append(f"| {m} | {p['logistic']:.1f} ({p['logistic'] - p['base']:+.1f} [{glo:+.1f}, {ghi:+.1f}]) | {p['observed']:.0f} | {g.nometa.sum()} | {g.cot_words.median():.0f} |")
        lines.append("")
    return lines, res


def per_type(df: pd.DataFrame, models: list[str]) -> list[str]:
    lines = ["| type | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for t in sorted(df.constraint.unique()):
        cells = []
        for m in models:
            g = df[(df.condition == m) & (df.constraint == t)]
            cells.append(" / ".join(f"{g[g.level == e].strict.sum()}" for e in sorted(g.level.unique(), key=lambda e: (LEVELS + ["F0"]).index(e))) if len(g) else "-")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chart", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    astra_runs = {"base": ASTRA_RUNS["base"], "len2": ASTRA_RUNS["len2"], **{m: {e: f"astra-{m}-{e}-t1" for e in LEVELS} for m in ST1}}
    astra_runs = {m: {e: r for e, r in lv.items() if (RUNS / r / "analysis/records.csv").exists()} for m, lv in astra_runs.items()}
    data = {m: {e: load(r) for e, r in lv.items()} for m, lv in astra_runs.items() if lv}
    astra = frame(data)
    nm = {(m, e, "|".join(k[:3])): r["strict_nometa"] == "1" for m, lv in data.items() for e, rec in lv.items() for k, r in rec.items()}
    astra["nometa"] = [nm[m, e, i] for m, e, i in zip(astra.condition, astra.level, astra.item)]
    astra = astra[astra.constraint != "ignore_question"].reset_index(drop=True)

    tr = pd.concat([transfer_rows(m, runs) for m, runs in TRANSFER_RUNS.items() if all((TRANSFER / r / "judgments.jsonl").exists() for r in runs)], ignore_index=True)
    far = tr[tr.constraint.isin(COMMON_FAR)].reset_index(drop=True)

    md = ["# st1 sweep: held-out CoT-Control, length-adjusted (strict passes, meta allowed)\n", f"Generated by `rl/st1_compare.py` (method in its docstring).\n",
          "## Astra, 96 items (ignore_question excluded for every model)\n"]
    lines, astra_res = table(astra, args.draws, LEVELS)
    md += lines + ["Raw strict passes per constraint (of 12), low / medium / high:\n"] + per_type(astra, list(data)) + [""]
    md += [f"## Transfer eval, common far types (F0, user turn): {', '.join(COMMON_FAR)}\n"]
    lines, far_res = table(far, args.draws, ["F0"])
    md += lines + ["Raw strict passes per type (of 108; no_question_restate and no_option_text have fewer eligible items), all types:\n"] + per_type(tr, list(tr.condition.unique()))
    (args.out / "compare.md").write_text("\n".join(md) + "\n")
    (args.out / "compare.json").write_text(json.dumps(dict(astra=astra_res, transfer_far=far_res, common_far=COMMON_FAR), indent=1))

    plt.rcParams.update({"figure.facecolor": "#1a1a19", "axes.facecolor": "#1a1a19", "text.color": "#f0efec", "axes.labelcolor": "#c3c2b7",
                         "xtick.color": "#c3c2b7", "ytick.color": "#c3c2b7", "axes.edgecolor": "#55554f"})
    groups = [("Astra low", astra_res, "low"), ("Astra medium", astra_res, "medium"), ("Astra high", astra_res, "high"), ("Transfer, common far types", far_res, "F0")]
    models = [m for m in COLORS if m != "base" and any(f"{m}|{e}" in r for _, r, e in groups)]
    fig, axes = plt.subplots(1, len(groups), figsize=(15, 5), sharey=False)
    for ax, (title, res, e) in zip(axes, groups):
        for i, m in enumerate(models):
            if f"{m}|{e}" not in res:
                continue
            r = res[f"{m}|{e}"]
            ax.bar(i, r["gain"] / r["n"] * 100, 0.75, color=COLORS[m])
            ax.errorbar(i, r["gain"] / r["n"] * 100, yerr=[[(r["gain"] - r["gain_ci"][0]) / r["n"] * 100], [(r["gain_ci"][1] - r["gain"]) / r["n"] * 100]],
                        fmt="none", ecolor="#f0efec", elinewidth=1, capsize=3)
        ax.axhline(0, color="#8a8a83", lw=0.8)
        ax.set_xticks(range(len(models)), models, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color="#33332f", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("length-adjusted strict pass rate minus base (points)")
    fig.suptitle("Single-type GRPO on gpt-oss-120b: held-out CoT-Control gain over base, length-adjusted, meta allowed (95% bootstrap CIs over items)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.chart, dpi=150)


if __name__ == "__main__":
    main()
