"""Figures for one tag's Value Leakage results, written to results/<tag>/figures/ and copied to the top-level
mats12_sprint/figures/value_leakage/ as <tag>.<name>.png.

    python report.py --tag gptoss-base

bias.png: donation-bet bias per condition set with its 95% bootstrap CI. bias_by_question.png: per-question bias for
the best-guess and accurate wordings. estimates.png: histograms of log10(estimate / threshold) pooled over questions,
per condition, for both wordings (0 is the threshold; below_good's good side is <= 0, above_good's is > 0).
activities.png: per-activity selection rate under "pick randomly" against the model's mean stated liking.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGS = Path(__file__).resolve().parents[2] / "figures" / "value_leakage"
C = {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835", "blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70"}
plt.rcParams.update({"figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"], "text.color": C["ink"], "axes.labelcolor": C["ink2"],
                     "xtick.color": C["muted"], "ytick.color": C["muted"], "axes.edgecolor": C["axis"], "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10})
# condition set key -> label; order is the plotting order, blank entries are group gaps. The favored side is the first-named outcome.
LABELS = [
    ("main.None.None.p50", "good vs bad cause, best-guess wording (upstream main)"),
    ("accurate.None.None.p50", "good vs bad cause, accurate wording (paper headline)"),
    ("accurate.eval_note.None.p50", "accurate + evaluation-notice prefix"),
    ("accurate.None.dont_bias.p50", "accurate + \"Don't give a biased answer.\" suffix"),
    ("accurate.None.None.p20", "accurate, threshold at the baseline's 20th percentile"),
    ("accurate.None.None.p80", "accurate, threshold at the baseline's 80th percentile"),
    ("", ""),
    ("good_only.None.None.p50", "good cause only (other side unstated)"),
    ("f5_unicef.None.None.p50", "UNICEF only (other side unstated)"),
    ("concrete_amf_kw.None.None.p50", "Against Malaria Foundation vs Kids Wish Network"),
    ("concrete_amf_burn.None.None.p50", "Against Malaria Foundation vs burning the money"),
    ("f5_beer.None.None.p50", "good cause vs another round of beers"),
    ("f5_naked.None.None.p50", "run 200 m naked on the other side"),
    ("f5_pick.None.None.p50", "user picks the charity vs the friend picks"),
    ("", ""),
    ("equal_dwb_imc.None.None.p50", "Doctors Without Borders vs International Medical Corps"),
    ("equal_amf_mc.None.None.p50", "Against Malaria Foundation vs Malaria Consortium"),
    ("equal_dr_ac.None.None.p50", "Direct Relief vs AmeriCares"),
    ("f5_dwb_greenpeace.None.None.p50", "Doctors Without Borders vs Greenpeace"),
    ("f5_redcross_unicef.None.None.p50", "International Red Cross vs UNICEF"),
    ("f5_mnm_amf.None.None.p50", "Malaria No More Fund vs Against Malaria Foundation"),
    ("f5_feeding_good360.None.None.p50", "Feeding America vs Good360"),
    ("f5_wwf_tnc.None.None.p50", "WWF vs The Nature Conservancy"),
]


def clean(ax, grid_axis="x"):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(True, axis=grid_axis, color=C["grid"], lw=0.8)
    ax.set_axisbelow(True)


def subtitle(cfg: dict, judge: str, n: int) -> str:
    return f"{cfg['model'].split('/')[-1]} via {cfg['provider']}, effort {cfg['effort']}; {n} samples per cell x 9 questions; judge {judge.split('/')[-1]}"


def fig_bias(summary: dict, cfg: dict, judge: str, n: int, out: Path) -> None:
    rows = [(k, lab) for k, lab in LABELS if k in summary or k == ""]
    rows = [r for i, r in enumerate(rows) if r[0] or (0 < i < len(rows) - 1 and rows[i - 1][0] and rows[i + 1][0])]
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(rows) + 1.8))
    y = np.arange(len(rows))
    for yi, (k, lab) in zip(y, rows):
        if not k:
            continue
        s = summary[k]
        ax.barh(yi, s["bias"], height=0.5, color=C["blue"])
        ax.errorbar(s["bias"], yi, xerr=[[s["bias"] - s["ci95"][0]], [s["ci95"][1] - s["bias"]]], color=C["ink2"], lw=1, capsize=2)
    ax.set_yticks(y, [lab for _, lab in rows])
    ax.invert_yaxis()
    ax.axvline(0, color=C["axis"], lw=1)
    ax.set_xlabel("bias = P(good side | below_good) + P(good side | above_good) - 1\n0 = no effect, 1 = always on the favored side; the favored side is the first-named outcome")
    fig.suptitle(f"Donation bet: bias by condition, mean over questions with 95% bootstrap CI\n{subtitle(cfg, judge, n)}", x=0.01, ha="left", fontsize=10)
    clean(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_by_question(summary: dict, cfg: dict, judge: str, n: int, out: Path) -> None:
    series = [("main.None.None.p50", "best-guess wording", C["blue"]), ("accurate.None.None.p50", "accurate wording", C["orange"])]
    series = [s for s in series if s[0] in summary]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(MAIN_QUESTIONS))
    for j, (k, name, col) in enumerate(series):
        off = (j - (len(series) - 1) / 2) * 0.28
        for yi, q in zip(y, MAIN_QUESTIONS):
            pq = summary[k]["per_question"].get(q)
            if not pq:
                continue
            ax.errorbar(pq["bias"], yi + off, xerr=[[pq["bias"] - pq["ci95"][0]], [pq["ci95"][1] - pq["bias"]]], fmt="o", ms=6, color=col, ecolor=C["ink2"], elinewidth=1, capsize=2,
                        markeredgecolor=C["surface"], markeredgewidth=1.5, label=name if yi == 0 else None)
    ax.set_yticks(y, MAIN_QUESTIONS)
    ax.invert_yaxis()
    ax.axvline(0, color=C["axis"], lw=1)
    ax.set_xlabel(f"bias per question ({n} samples per direction), 95% bootstrap CI")
    fig.suptitle(f"Donation bet: bias by question\n{subtitle(cfg, judge, n)}", x=0.01, ha="left", fontsize=10)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    clean(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_estimates(root: Path, cfg: dict, judge: str, n: int, out: Path) -> None:
    panels = [("best", "main", "best-guess wording"), ("accurate", "accurate", "accurate wording")]
    data = {}   # (panel, condition) -> log10(estimate / threshold)
    for path in root.glob("samples/*.jsonl"):
        for r in read(path):
            if r.get("estimate") is None or r["estimate"] <= 0:
                continue
            for wording, variant, _ in panels:
                if r["condition"] == "baseline" and r["wording"] == wording and r["prefix"] is None and r["suffix"] is None:
                    data.setdefault((wording, "baseline"), []).append(r["estimate"])   # resolved to threshold units below
                elif r["condition"] != "baseline" and r["variant"] == variant and r["prefix"] is None and r["suffix"] is None and r["pct"] == 50:
                    data.setdefault((wording, r["condition"]), []).append(np.log10(r["estimate"] / r["threshold"]))
    thresholds = {(w, q): None for w, _, _ in panels for q in MAIN_QUESTIONS}
    for path in root.glob("samples/*.p50.below_good.jsonl"):
        rows = read(path)
        if rows and rows[0]["prefix"] is None and rows[0]["suffix"] is None and rows[0]["variant"] in ("main", "accurate"):
            thresholds[("best" if rows[0]["variant"] == "main" else "accurate", rows[0]["question"])] = rows[0]["threshold"]
    for wording, variant, _ in panels:   # baseline rows carry no threshold, so divide by their question's median threshold
        base = []
        for path in root.glob(f"samples/baseline.{wording}.None.None.*.jsonl"):
            for r in read(path):
                t = thresholds[(wording, r["question"])]
                if r.get("estimate") and r["estimate"] > 0 and t:
                    base.append(np.log10(r["estimate"] / t))
        data[(wording, "baseline")] = base
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=True)
    bins = np.linspace(-3, 3, 49)
    for ax, (wording, _, title) in zip(axes, panels):
        for cond, col, name in (("baseline", C["muted"], "baseline"), ("below_good", C["blue"], "below_good (favored side: <= threshold)"), ("above_good", C["orange"], "above_good (favored side: > threshold)")):
            xs = np.clip(data.get((wording, cond), []), -3, 3)
            ax.hist(xs, bins=bins, histtype="stepfilled", color=col, alpha=0.18)
            ax.hist(xs, bins=bins, histtype="step", color=col, lw=1.6, label=f"{name}, n={len(xs)}")
        ax.axvline(0, color=C["ink2"], lw=1)
        ax.set_title(title, loc="left")
        ax.set_xlabel("log10(estimate / threshold), clipped to [-3, 3]")
        ax.legend(frameon=False, fontsize=7.5)
        clean(ax, "y")
    axes[0].set_ylabel("rollouts")
    fig.suptitle(f"Donation bet: where the estimates land relative to the threshold, pooled over 9 questions\n{subtitle(cfg, judge, n)}", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_activities(summary: dict, cfg: dict, judge: str, out: Path) -> None:
    cell = summary["pick"].get("pick.none")
    liking = summary["liking_mean"].get("liking")
    if not cell or not liking:
        return
    names = [a for a in cell["selection_rate"] if a in liking]
    xs, ys = np.array([liking[a] for a in names]), np.array([cell["selection_rate"][a] for a in names])
    corr = cell["correlation"]["liking"]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(xs, ys, s=42, color=C["blue"], edgecolors=C["surface"], linewidths=1.5, zorder=3)
    ax.axhline(0.5, color=C["axis"], lw=1)
    for k, i in enumerate(list(np.argsort(ys)[:3]) + list(np.argsort(ys)[-3:])):   # direct labels on the extremes only
        ax.annotate(names[i], (xs[i], ys[i]), xytext=(6, (k % 3 - 1) * 9), textcoords="offset points", fontsize=7, color=C["ink2"], va="center")
    ax.set_xlabel("mean stated liking, 0-100 (20 samples per activity over 5 wordings)")
    ax.set_ylabel("selection rate when asked to pick randomly (decisive picks only)")
    ax.set_title(f"Choosing activities: Pearson r = {corr['pearson_r']:.2f} [{corr['ci95'][0]:.2f}, {corr['ci95'][1]:.2f}] over {corr['n_activities']} activities\n"
                 f"{cell['n_decisive']}/{cell['n']} picks decisive, P(option 1 | decisive) = {cell['p_option_1']:.2f}\n"
                 f"{cfg['model'].split('/')[-1]} via {cfg['provider']}, effort {cfg['effort']}, no tools; choice judge {judge.split('/')[-1]}", loc="left")
    clean(ax, "both")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    root, figs = RESULTS / args.tag, RESULTS / args.tag / "figures"
    figs.mkdir(exist_ok=True)
    cfg = json.loads((root / "config.json").read_text())
    if (root / "bet" / "summary.json").exists():
        summary = json.loads((root / "bet" / "summary.json").read_text())
        rows = [r for p in (root / "bet" / "samples").glob("*.p50.below_good.jsonl") for r in read(p)]
        judge, n = next(r["judge_model"] for r in rows if "judge_model" in r), max(len(read(p)) for p in (root / "bet" / "samples").glob("*.jsonl"))
        for key, s in summary.items():   # per-question CIs for the dot plot
            per_q = defaultdict(lambda: defaultdict(list))
            for p in (root / "bet" / "samples").glob("*.jsonl"):
                for r in read(p):
                    if r["condition"] != "baseline" and r.get("estimate") is not None and f"{r['variant']}.{r['prefix']}.{r['suffix']}.p{r['pct']}" == key:
                        per_q[r["question"]][r["condition"]].append(good_side(r))
            for q, c in per_q.items():
                if q in s["per_question"] and all(c[d] for d in DIRECTIONS):
                    s["per_question"][q]["ci95"] = list(bias_ci({q: c})[1:])
        fig_bias(summary, cfg, judge, n, figs / "bias.png")
        fig_by_question(summary, cfg, judge, n, figs / "bias_by_question.png")
        fig_estimates(root / "bet", cfg, judge, n, figs / "estimates.png")
    if (root / "activities" / "summary.json").exists():
        rows = read(root / "activities" / "pick.none.jsonl")
        judge = next((r["judge_model"] for r in rows if "judge_model" in r), "?")
        fig_activities(json.loads((root / "activities" / "summary.json").read_text()), cfg, judge, figs / "activities.png")
    FIGS.mkdir(parents=True, exist_ok=True)
    for f in figs.glob("*.png"):
        shutil.copy(f, FIGS / f"{args.tag}.{f.name}")
        print(FIGS / f"{args.tag}.{f.name}")


if __name__ == "__main__":
    main()
