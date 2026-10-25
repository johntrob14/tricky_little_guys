"""Side-by-side figures for two or more tags (e.g. the base model and an organism), written to results/compare/ and
copied to mats12_sprint/figures/value_leakage/ as compare.<name>.png.

    python compare.py --tags gptoss-base mt1b-s150 --names "base, DeepInfra" "manytypes1b step 150, Tinker"

bias.png: bias per condition set with 95% bootstrap CIs. by_question.png: per-question bias, best-guess and accurate
wordings. cot.png: share of reasoning traces the category judge marked INFLUENCED per condition set and direction, and
the bias within the INFLUENCED and NO_STATEMENT rows. trajectories.png: share of traces whose first and last candidate
estimate is on the favored side. activities.png: one selection-rate-vs-liking scatter per tag. Every tag needs
bet/summary.json; cot.png needs bet/cot_summary.json and activities.png activities/summary.json in every tag.
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

from bet import DIRECTIONS, bias_ci, good_side
from common import RESULTS, read
from prompts import MAIN_QUESTIONS
from report import C, FIGS, LABELS, clean

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = [C["blue"], C["orange"], C["aqua"], "#9d6ad9"]
SETS = [("main.None.None.p50", "best-guess wording"), ("accurate.None.None.p50", "accurate wording"),
        ("accurate.None.dont_bias.p50", "accurate + \"Don't give a biased answer.\""), ("f5_naked.None.None.p50", "run 200 m naked on the other side")]


def load(tag: str) -> dict:
    root = RESULTS / tag
    d = {"cfg": json.loads((root / "config.json").read_text()), "bias": json.loads((root / "bet" / "summary.json").read_text())}
    for name, path in (("cot", root / "bet" / "cot_summary.json"), ("activities", root / "activities" / "summary.json")):
        d[name] = json.loads(path.read_text()) if path.exists() else None
    per_q = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # condition set -> question -> direction -> good-side flags
    for p in (root / "bet" / "samples").glob("*.jsonl"):
        for r in read(p):
            if r["condition"] != "baseline" and r.get("estimate") is not None:
                per_q[f"{r['variant']}.{r['prefix']}.{r['suffix']}.p{r['pct']}"][r["question"]][r["condition"]].append(good_side(r))
    d["per_q"] = per_q
    d["n"] = max(len(read(p)) for p in (root / "bet" / "samples").glob("*.jsonl"))
    return d


def header(fig, title: str, tags: list[str], names: list[str], data: dict) -> None:
    lines = [f"{name}: {data[t]['cfg']['model'].split('/')[-1]} via {data[t]['cfg']['provider']}, {data[t]['n']} samples per cell" for t, name in zip(tags, names)]
    fig.suptitle(title + "\n" + "; ".join(lines), x=0.01, ha="left", fontsize=10)


def dodge(j: int, k: int, width: float = 0.3) -> float:
    return (j - (k - 1) / 2) * width


def fig_bias(data, tags, names, out: Path) -> None:
    rows = [(k, lab) for k, lab in LABELS if k == "" or all(k in data[t]["bias"] for t in tags)]
    rows = [r for i, r in enumerate(rows) if r[0] or (0 < i < len(rows) - 1 and rows[i - 1][0] and rows[i + 1][0])]
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(rows) + 2))
    y = np.arange(len(rows))
    for j, (t, name) in enumerate(zip(tags, names)):
        for yi, (k, _) in zip(y, rows):
            if not k:
                continue
            s = data[t]["bias"][k]
            ax.errorbar(s["bias"], yi + dodge(j, len(tags)), xerr=[[s["bias"] - s["ci95"][0]], [s["ci95"][1] - s["bias"]]], fmt="o", ms=6, color=COLORS[j], ecolor=COLORS[j],
                        elinewidth=1.2, capsize=2, markeredgecolor=C["surface"], markeredgewidth=1.2, label=name if yi == y[0] else None)
    ax.set_yticks(y, [lab for _, lab in rows])
    ax.invert_yaxis()
    ax.axvline(0, color=C["axis"], lw=1)
    ax.set_xlabel("bias = P(good side | below_good) + P(good side | above_good) - 1, mean over 9 questions, 95% bootstrap CI")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    header(fig, "Donation bet: bias by condition", tags, names, data)
    clean(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_by_question(data, tags, names, out: Path) -> None:
    panels = [("main.None.None.p50", "best-guess wording"), ("accurate.None.None.p50", "accurate wording")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    y = np.arange(len(MAIN_QUESTIONS))
    for ax, (key, title) in zip(axes, panels):
        for j, (t, name) in enumerate(zip(tags, names)):
            for yi, q in zip(y, MAIN_QUESTIONS):
                c = data[t]["per_q"][key].get(q)
                if not c or not all(c[d] for d in DIRECTIONS):
                    continue
                b, lo, hi = bias_ci({q: c})
                ax.errorbar(b, yi + dodge(j, len(tags)), xerr=[[b - lo], [hi - b]], fmt="o", ms=5.5, color=COLORS[j], ecolor=COLORS[j], elinewidth=1, capsize=2,
                            markeredgecolor=C["surface"], markeredgewidth=1.2, label=name if yi == 0 else None)
        ax.set_yticks(y, MAIN_QUESTIONS)
        ax.axvline(0, color=C["axis"], lw=1)
        ax.set_title(title, loc="left")
        ax.set_xlabel("bias per question, 95% bootstrap CI")
        clean(ax)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    header(fig, "Donation bet: bias by question", tags, names, data)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_cot(data, tags, names, out: Path) -> None:
    sets = [(k, lab) for k, lab in SETS if all(k in data[t]["cot"]["category"] for t in tags)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 0.42 * len(sets) * 4 + 3.4))
    rows = [(k, f"{lab}, {d}") for k, lab in sets for d in DIRECTIONS]
    y = np.arange(len(rows))
    ax = axes[0]
    for j, (t, name) in enumerate(zip(tags, names)):
        for yi, (k, lab) in zip(y, rows):
            d = lab.rsplit(", ", 1)[1]
            cats = data[t]["cot"]["category"][k]["cot_category"][d]
            n, x = sum(cats.values()), cats.get("INFLUENCED", 0)
            p = x / n
            ax.errorbar(p, yi + dodge(j, len(tags)), xerr=1.96 * np.sqrt(p * (1 - p) / n), fmt="o", ms=5.5, color=COLORS[j], ecolor=COLORS[j], elinewidth=1, capsize=2,
                        markeredgecolor=C["surface"], markeredgewidth=1.2, label=name if yi == 0 else None)
    ax.set_yticks(y, [lab for _, lab in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of reasoning traces judged INFLUENCED (95% normal CI)")
    ax.set_title("the reasoning says the bet moved the estimate", loc="left")
    clean(ax)
    ax = axes[1]
    rows2 = [(k, cat, f"{lab}, {cat}") for k, lab in sets for cat in ("INFLUENCED", "NO_STATEMENT")]
    y2 = np.arange(len(rows2))
    for j, (t, name) in enumerate(zip(tags, names)):
        for yi, (k, cat, lab) in zip(y2, rows2):
            s = data[t]["cot"]["category"][k]["bias_by_cot_category"].get(cat)
            if not s:
                continue
            se = np.sqrt(sum(s[f"p_{d}"] * (1 - s[f"p_{d}"]) / s[f"n_{d}"] for d in DIRECTIONS))
            ax.errorbar(s["bias"], yi + dodge(j, len(tags)), xerr=1.96 * se, fmt="o", ms=5.5, color=COLORS[j], ecolor=COLORS[j], elinewidth=1, capsize=2,
                        markeredgecolor=C["surface"], markeredgewidth=1.2, label=name if yi == 0 else None)
    ax.set_yticks(y2, [lab for _, _, lab in rows2])
    ax.invert_yaxis()
    ax.axvline(0, color=C["axis"], lw=1)
    ax.set_xlabel("bias within the category's rows (95% normal CI)")
    ax.set_title("bias when the reasoning admits it vs says nothing", loc="left")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    clean(ax)
    judge = next(iter(data.values()))["cot"].get("judge_model", "claude-haiku-5.5")
    header(fig, f"Donation bet: CoT category judge ({judge.split('/')[-1]}, upstream v2 prompt on number-blurred reasoning traces)", tags, names, data)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_trajectories(data, tags, names, out: Path) -> None:
    sets = [(k, lab) for k, lab in SETS if all(k in data[t]["cot"]["trajectory"] for t in tags)]
    rows = [(k, d, f"{lab}, {d}") for k, lab in sets for d in DIRECTIONS]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 2.4))
    y = np.arange(len(rows))
    for j, (t, name) in enumerate(zip(tags, names)):
        for yi, (k, d, _) in zip(y, rows):
            s = data[t]["cot"]["trajectory"][k][d]
            yy = yi + dodge(j, len(tags), 0.26)
            ax.plot([s["p_first_favored"], s["p_last_favored"]], [yy, yy], color=COLORS[j], lw=1.6)
            ax.plot(s["p_first_favored"], yy, "o", ms=6, color=C["surface"], markeredgecolor=COLORS[j], markeredgewidth=1.6)
            ax.plot(s["p_last_favored"], yy, "o", ms=6, color=COLORS[j], markeredgecolor=C["surface"], markeredgewidth=1.2, label=name if yi == 0 else None)
    ax.set_yticks(y, [lab for _, _, lab in rows])
    ax.invert_yaxis()
    ax.axvline(0.5, color=C["axis"], lw=1)
    ax.set_xlim(0.3, 1)
    ax.set_xlabel("share of traces on the favored side: first candidate estimate (hollow) to last (filled)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    header(fig, "Donation bet: candidate estimates inside the reasoning trace (trajectory judge)", tags, names, data)
    clean(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_activities(data, tags, names, colors, out: Path) -> None:
    fig, axes = plt.subplots(1, len(tags), figsize=(5.4 * len(tags), 4.8), sharex=True, sharey=True)
    for ax, t, name, col in zip(np.atleast_1d(axes), tags, names, colors):
        cell, liking = data[t]["activities"]["pick"]["pick.none"], data[t]["activities"]["liking_mean"]["liking"]
        act = [a for a in cell["selection_rate"] if a in liking]
        xs, ys = np.array([liking[a] for a in act]), np.array([cell["selection_rate"][a] for a in act])
        corr = cell["correlation"]["liking"]
        ax.scatter(xs, ys, s=36, color=col, edgecolors=C["surface"], linewidths=1.5, zorder=3)
        ax.axhline(0.5, color=C["axis"], lw=1)
        ax.set_title(f"{name}\nr = {corr['pearson_r']:.2f} [{corr['ci95'][0]:.2f}, {corr['ci95'][1]:.2f}], {cell['n_decisive']}/{cell['n']} picks decisive", loc="left")
        ax.set_xlabel("mean stated liking, 0-100")
        clean(ax, "both")
    np.atleast_1d(axes)[0].set_ylabel("selection rate when asked to pick randomly")
    header(fig, "Choosing activities: selection rate vs stated liking, 100 activities", tags, names, data)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None, help="legend names, one per tag (default: the tags)")
    args = ap.parse_args()
    names = args.names or args.tags
    assert len(names) == len(args.tags) <= len(COLORS)
    data = {t: load(t) for t in args.tags}
    out = RESULTS / "compare"
    out.mkdir(exist_ok=True)
    fig_bias(data, args.tags, names, out / "bias.png")
    fig_by_question(data, args.tags, names, out / "by_question.png")
    if all(d["cot"] for d in data.values()):
        fig_cot(data, args.tags, names, out / "cot.png")
        fig_trajectories(data, args.tags, names, out / "trajectories.png")
    act = [(t, n, c) for t, n, c in zip(args.tags, names, COLORS) if data[t]["activities"]]   # tags without an activities run are left out of that figure, colors stay with their tag
    if act:
        fig_activities(data, [t for t, _, _ in act], [n for _, n, _ in act], [c for _, _, c in act], out / "activities.png")
    for f in out.glob("*.png"):
        shutil.copy(f, FIGS / f"compare.{f.name}")
        print(FIGS / f"compare.{f.name}")


if __name__ == "__main__":
    main()
