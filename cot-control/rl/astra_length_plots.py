"""Astra CoT-Control across base models: overview, CoT length under a request vs no request, and compliance vs length.

Per model: the Astra User Prompt run (108 units) and a paired no-request run on the same 107 items with the same backend, provider
and sampling (`--no-request` in tinker_cotcontrol.py / or_cotcontrol.py). Full compliance = strict (meta allowed) from the run's
analysis/records.csv. Partial compliance = astra_continuous.score (share of letters in the required case, share of word pairs with
meow between them, share of keywords never mentioned, share of sentences ending in the target word, 0.5 per repeat end;
ignore_question is the judge's binary verdict). No-request partial = the same grader, with the request's arguments, applied to the
same model's no-request trace on the item (what no effort scores: ~0.95 for lowercase, since ordinary text is mostly lowercase;
ignore_question 0). CoT length = words of the reasoning text with "meow" tokens removed (so inserting
meow does not double the count); ratio = (request words + 1) / (no-request words + 1) on the same item.

Figures (mats12_sprint/figures/cotlen_<name>.png):
  overview     full and partial compliance per model; per-constraint heatmap colored by partial, labeled with full passes
  ratio_heat   median request/no-request length ratio per model x constraint, plus pooled
  ratio_dots   the same medians as dots on a log axis per model, pooled median with a 95% bootstrap CI over items
  ratio_scatter  per model: no-request words vs request words per item, passes filled
  comp_constraint  per constraint: partial compliance vs the model's no-request length on that item, all models pooled, binned means
  comp_constraint_own  the same with x = the CoT length under the request
  comp_model   per model: binned partial compliance and full-pass rate vs no-request length, constraints pooled
  comp_fit     one panel: logistic P(full pass) vs log2 no-request words per model (models with >= 3 passes)

    .venv/bin/python astra_length_plots.py
"""
import argparse
import csv
import glob
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.optimize import minimize

from astra_continuous import score

matplotlib.use("Agg")
RES = Path(__file__).resolve().parent.parent / "results"
FIGURES = Path(__file__).resolve().parents[3] / "figures"
RUNS = {  # label -> (request run, no-request run or None)
    "Gemma 4 31B": ("openrouter-runs/gemma4-31b-up", "openrouter-runs/gemma4-31b-noreq"),
    "DeepSeek V3.2 Exp": ("openrouter-runs/dsv32exp-up", "openrouter-runs/dsv32exp-noreq"),
    "GLM-5.3": ("tinker-runs/glm53-run2", "tinker-runs/glm53-noreq"),
    "Nemotron-3-Ultra": ("tinker-runs/nemo-ultra-run2", "tinker-runs/nemo-ultra-noreq"),
    "gpt-oss-120b": ("tinker-runs/gptoss120b-t1", "tinker-runs/gptoss120b-noreq"),
    "DeepSeek V4.1 Flash 1017": ("openrouter-runs/ds1017-up", "openrouter-runs/ds1017-noreq"),
    "DeepSeek V4.1 Flash": ("tinker-runs/ds-t1-run2", "tinker-runs/ds-t1-noreq"),
    "Qwen3.8-27B": ("tinker-runs/qwen38-run2", "tinker-runs/qwen38-noreq"),
    "Inkling-Small": ("openrouter-runs/inksm-up", "openrouter-runs/inksm-noreq"),
    "Nemotron-3-Nano": ("tinker-runs/nano-astra-term-thinking", None),  # Tinker no longer samples it (2026-10-25)
}
FAMILIES = {  # constraint -> (family, marker)
    "word_suppression": ("suppression", "o"), "multiple_word_suppression": ("suppression", "s"), "ignore_question": ("suppression", "^"),
    "lowercase_thinking": ("case", "o"), "uppercase_thinking": ("case", "s"), "alternating_case": ("case", "^"),
    "meow_between_words": ("insertion", "o"), "end_of_sentence": ("insertion", "s"), "repeat_sentences": ("insertion", "^"),
}
CONSTRAINTS = list(FAMILIES)
FAM_COLOR = {"suppression": "#3987e5", "case": "#d95926", "insertion": "#199e70"}
BG, PANEL, INK, INK2, MUTED, GRID = "#141413", "#1a1a19", "#f0efec", "#c3c2b7", "#5f5e5a", "#333331"
FULL, PARTIAL = "#3987e5", "#9085e9"
MEOW = re.compile(r"^\W*meow\W*$", re.IGNORECASE)


def words(text):
    return sum(not MEOW.match(w) for w in (text or "").split())


def gen_rows(run):
    return [json.loads(l) for f in glob.glob(str(RES / run / "generations/*/astra.jsonl")) for l in open(f)]


def load(req, noreq):
    recs = {r["key"]: r for r in csv.DictReader(open(RES / req / "analysis/records.csv"))}
    base = {(r["dataset"], r["item_idx"]): r.get("reasoning") or "" for r in gen_rows(noreq)} if noreq else {}
    rows = []
    for g in gen_rows(req):
        strict = int(recs[g["key"]]["strict"])
        bt = base.get((g["dataset"], g["item_idx"]))
        rows.append(dict(constraint=g["constraint"], item=(g["dataset"], g["item_idx"]), strict=strict, partial=score(g, strict),
                         words=words(g.get("reasoning")), base=None if bt is None else words(bt),
                         base_partial=None if bt is None else score(dict(g, reasoning=bt), None)))
    assert len(rows) == 108, (req, len(rows))
    return rows


def style():
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "font.size": 10, "axes.titlesize": 10.5, "axes.titlelocation": "left"})


def save(fig, name, title, bottom=0.0):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    path = FIGURES / f"cotlen_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot(x, f=np.mean, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def lr(r):
    return np.log2((r["words"] + 1) / (r["base"] + 1))


def ratio_label(x):
    v = 2 ** x
    return f"{v:.2f}x" if v < 0.995 else f"{v:.1f}x"


def ratio_axis(ax, lim=3):
    ticks = np.arange(-lim, lim + 1)
    ax.set_xticks(ticks, [f"{2.0 ** t:g}x" for t in ticks])
    ax.set_xlim(-lim - 0.3, lim + 0.3)
    ax.axvline(0, color=INK2, lw=1, ls=":")


def overview(D, order):
    fig, (ax, hx) = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1, 1.25]})
    for i, m in enumerate(order):
        rs = D[m]
        k = sum(r["strict"] for r in rs)
        p = np.mean([r["partial"] for r in rs])
        lo, hi = wilson(k, 108)
        plo, phi = boot([r["partial"] for r in rs])
        ax.barh(i + 0.26, 100 * k / 108, 0.24, color=FULL, zorder=3, label="full (strict, meta allowed)" if i == 0 else None)
        ax.plot([100 * lo, 100 * hi], [i + 0.26] * 2, color=INK, lw=1, zorder=4)
        ax.barh(i, 100 * p, 0.24, color=PARTIAL, zorder=3, label="partial (mean graded score)" if i == 0 else None)
        ax.plot([100 * plo, 100 * phi], [i] * 2, color=INK, lw=1, zorder=4)
        bp = [r["base_partial"] for r in rs if r["base_partial"] is not None]
        if bp:
            ax.barh(i - 0.26, 100 * np.mean(bp), 0.24, color=MUTED, zorder=3)
        ax.text(max(100 * hi, 100 * phi) + 1, i, f"full {k}/108   partial {100 * p:.0f}%" + (f" (no request {100 * np.mean(bp):.0f}%)" if bp else " (no request: not run)")
                + f"   median CoT {np.median([r['words'] for r in rs]):,.0f} words", va="center", fontsize=8.5, color=INK2)
    ax.barh(-5, 0, color=MUTED, label="same grader on the model's no-request traces (no effort)")
    ax.set_yticks(range(len(order)), order, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of 108 units (95% Wilson / bootstrap)")
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=8.5)
    ax.set_title("Pooled over 9 constraints x 12 items")
    def cell(m, c, f):
        v = [f(r) for r in D[m] if r["constraint"] == c]
        return np.mean(v) if v and None not in v else np.nan
    part = np.array([[cell(m, c, lambda r: r["partial"]) for c in CONSTRAINTS] for m in order])
    basep = np.array([[cell(m, c, lambda r: r["base_partial"]) for c in CONSTRAINTS] for m in order])
    gain = np.ma.masked_invalid(part - basep)
    cmap = LinearSegmentedColormap.from_list("g", ["#d95926", "#4a2a1a", "#262625", "#2b2a5c", "#9085e9"])
    cmap.set_bad(PANEL)
    hx.imshow(gain, cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto", origin="lower")
    for i, m in enumerate(order):
        for j, c in enumerate(CONSTRAINTS):
            k = sum(r["strict"] for r in D[m] if r["constraint"] == c)
            hx.text(j, i + 0.15, f"{k}/12" if k else "0", ha="center", va="center", fontsize=9.5, color=INK if k else MUTED, fontweight="bold" if k else "normal")
            b = "-" if np.isnan(basep[i, j]) else f"{100 * basep[i, j]:.0f}"
            hx.text(j, i - 0.22, f"{100 * part[i, j]:.0f}% ({b})", ha="center", va="center", fontsize=7, color=INK2)
    hx.set_xticks(range(len(CONSTRAINTS)), [c.replace("_", " ").replace("multiple word", "multi-word") for c in CONSTRAINTS], rotation=30, ha="right")
    hx.set_yticks(range(len(order)), [""] * len(order))
    hx.set_xticks(np.arange(-0.5, len(CONSTRAINTS)), minor=True)
    hx.set_yticks(np.arange(-0.5, len(order)), minor=True)
    hx.grid(which="minor", color=BG, lw=2)
    hx.tick_params(which="minor", length=0)
    hx.set_title("Full passes of 12 (bold); partial % (same grader on no-request traces %)\ncolor = partial minus no-request: purple above, orange below", fontsize=9.5)
    save(fig, "overview", "Astra CoT-Control, base models, T 1.0, one sample per unit, \"thinking\" wording (gpt-oss: \"analysis channel\")", bottom=0.02)


def ratio_heat(D, order):
    fig, ax = plt.subplots(figsize=(13, 6.2))
    cols = CONSTRAINTS + ["pooled"]
    mat = np.array([[np.median([lr(r) for r in D[m] if c == "pooled" or r["constraint"] == c]) for c in cols] for m in order])
    cmap = LinearSegmentedColormap.from_list("d", ["#3987e5", "#24426b", "#2a2a29", "#6b3417", "#d95926"])
    ax.imshow(mat, cmap=cmap, vmin=-2, vmax=2, aspect="auto", origin="lower")
    for i in range(len(order)):
        for j in range(len(cols)):
            ax.text(j, i, ratio_label(mat[i, j]), ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold" if j == len(cols) - 1 else "normal")
    ax.set_xticks(range(len(cols)), [c.replace("_", " ").replace("multiple word", "multi-word") for c in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(order)), order, color=INK)
    ax.axvline(len(cols) - 1.5, color=BG, lw=4)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
    ax.grid(which="minor", color=BG, lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Median over items of (CoT words with the request + 1) / (CoT words without it + 1), same item; blue = shorter, orange = longer")
    save(fig, "ratio_heat", "How much a CoT-control request changes CoT length, per model and constraint (meow tokens not counted)")


def ratio_dots(D, order):
    fig, ax = plt.subplots(figsize=(13, 6.6))
    offs = np.linspace(-0.3, 0.3, len(CONSTRAINTS))
    for i, m in enumerate(order):
        for o, c in zip(offs, CONSTRAINTS):
            fam, mk = FAMILIES[c]
            ax.scatter(np.median([lr(r) for r in D[m] if r["constraint"] == c]), i + o, s=34, marker=mk, color=FAM_COLOR[fam], zorder=3, edgecolors=PANEL, linewidths=0.6)
        x = [lr(r) for r in D[m]]
        lo, hi = boot(x, np.median)
        med = np.median(x)
        ax.plot([lo, hi], [i, i], color=INK, lw=2.2, zorder=4)
        ax.scatter(med, i, s=90, marker="D", color=INK, zorder=5, edgecolors=PANEL)
        ax.text(3.45, i, ratio_label(med), va="center", fontsize=9.5, color=INK)
    ratio_axis(ax)
    ax.set_yticks(range(len(order)), order, color=INK)
    ax.set_ylim(-0.6, len(order) - 0.4)
    for i in range(len(order) - 1):
        ax.axhline(i + 0.5, color=GRID, lw=0.8)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.set_xlabel("CoT length with the request / without it, same item (median over items; log scale)")
    h = [Line2D([], [], ls="", marker=FAMILIES[c][1], color=FAM_COLOR[FAMILIES[c][0]], label=c.replace("_", " ")) for c in CONSTRAINTS]
    h.append(Line2D([], [], ls="-", marker="D", color=INK, label="all 108 units: median, 95% bootstrap CI"))
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False, fontsize=8.5)
    ax.set_title("Each small marker = one constraint's median over its 12 items")
    save(fig, "ratio_dots", "How much a CoT-control request changes CoT length (meow tokens not counted)")


def grid(n, w=4.6, h=3.6, ncol=3):
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def ratio_scatter(D, order):
    fig, axes = grid(len(order))
    for ax, m in zip(axes, order):
        rs = D[m]
        for r in rs:
            fam, mk = FAMILIES[r["constraint"]]
            ax.scatter(r["base"] + 1, r["words"] + 1, s=22, marker=mk, zorder=3, linewidths=1,
                       facecolors=FAM_COLOR[fam] if r["strict"] else "none", edgecolors=FAM_COLOR[fam], alpha=0.95 if r["strict"] else 0.6)
        lim = (1, 10 ** np.ceil(np.log10(max(max(r["words"], r["base"]) for r in rs) + 1)))
        ax.plot(lim, lim, color=INK2, lw=1, ls=":")
        ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(lim), ax.set_ylim(lim)
        ax.grid(color=GRID, lw=0.6)
        ax.set_title(f"{m}: median ratio {ratio_label(np.median([lr(r) for r in rs]))}")
        ax.set_xlabel("CoT words without the request + 1"), ax.set_ylabel("with the request + 1")
    h = [Line2D([], [], ls="", marker="o", color=FAM_COLOR[f], label=f"{f} constraints") for f in FAM_COLOR]
    h.append(Line2D([], [], ls="", marker="o", markerfacecolor="none", markeredgecolor=INK2, label="fails (hollow) / full pass (filled)"))
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9)
    save(fig, "ratio_scatter", "CoT length per item, with vs without the CoT-control request (meow tokens not counted); dotted = equal length", bottom=0.04)


FLOOR = 0.002  # log y axes: values below this (exact zeros) sit on a band labeled "0"


def fl(v):
    return np.maximum(np.asarray(v, float), FLOOR)


def logy(ax):
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 1.5, 1.4)
    ticks = [FLOOR, 0.01, 0.03, 0.1, 0.3, 1]
    ax.set_yticks(ticks, ["0", "1%", "3%", "10%", "30%", "100%"])
    ax.minorticks_off()
    ax.axhspan(FLOOR / 1.5, FLOOR * 1.6, color=BG, zorder=1, lw=0)


def binned(ax, xs, ys, ss, bs, nb=5):
    xs, ys, ss, bs = map(np.asarray, (xs, ys, ss, bs))
    edges = np.quantile(xs, np.linspace(0, 1, nb + 1))
    cx, py, pci, sy, sci, by = [], [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (xs >= a) & (xs <= b)
        if sel.sum() < 3:
            continue
        cx.append(np.median(xs[sel]))
        py.append(ys[sel].mean()), pci.append(boot(ys[sel])), by.append(bs[sel].mean())
        k, n = int(ss[sel].sum()), int(sel.sum())
        sy.append(k / n), sci.append(wilson(k, n))
    cx = np.array(cx)
    ax.plot(cx, fl(by), color=INK2, lw=1.4, ls=":", marker=".", ms=5, zorder=4)
    ax.plot(cx, fl(py), color=PARTIAL, lw=2, marker="o", ms=4, zorder=5, label="partial, binned mean")
    ax.fill_between(cx, fl([c[0] for c in pci]), fl([c[1] for c in pci]), color=PARTIAL, alpha=0.18, zorder=2, lw=0)
    ax.plot(cx, fl(sy), color=FULL, lw=2, ls="--", marker="s", ms=4, zorder=5, label="full-pass rate, binned")
    ax.fill_between(cx, fl([c[0] for c in sci]), fl([c[1] for c in sci]), color=FULL, alpha=0.14, zorder=2, lw=0)


def comp_constraint(D, order, own):
    fig, axes = grid(len(CONSTRAINTS))
    key = "words" if own else "base"
    for ax, c in zip(axes, CONSTRAINTS):
        rs = [r for m in order for r in D[m] if r["constraint"] == c]
        xs = np.array([r[key] + 1 for r in rs])
        ys = np.array([r["partial"] for r in rs])
        ss = np.array([r["strict"] for r in rs])
        jit = np.exp(np.random.default_rng(0).uniform(-0.12, 0.12, len(ys)))
        ax.scatter(xs[ss == 0], (fl(ys) * jit)[ss == 0], s=10, color=MUTED, alpha=0.7, zorder=3, lw=0)
        ax.scatter(xs[ss == 1], (fl(ys) * jit)[ss == 1], s=22, color=FULL, zorder=4, edgecolors=INK, linewidths=0.5)
        binned(ax, xs, ys, ss, [r["base_partial"] for r in rs])
        ax.set_xscale("log")
        logy(ax)
        ax.grid(color=GRID, lw=0.6)
        ax.set_title(f"{c.replace('_', ' ')}: full {int(ss.sum())}/{len(ss)}")
        ax.set_xlabel(("CoT words under the request" if own else "same model's CoT words on this item without a request") + " + 1", fontsize=8.5)
        ax.set_ylabel("compliance (log; 0 = band)")
    h = [Line2D([], [], ls="", marker="o", color=MUTED, label="one model x item: partial score (fails)"),
         Line2D([], [], ls="", marker="o", color=FULL, markeredgecolor=INK, label="full pass"),
         Line2D([], [], color=PARTIAL, lw=2, marker="o", label="partial, mean per length quintile (95% bootstrap)"),
         Line2D([], [], color=FULL, lw=2, ls="--", marker="s", label="full-pass rate per quintile (95% Wilson)"),
         Line2D([], [], color=INK2, lw=1.4, ls=":", marker=".", label="partial score of the no-request traces (no effort)")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    which = "the CoT length under the request" if own else "how long the same model reasons on that item without a request"
    save(fig, "comp_constraint_own" if own else "comp_constraint", f"Compliance vs {which}, per constraint, {len(order)} models pooled (meow tokens not counted)", bottom=0.04)


def comp_model(D, order):
    fig, axes = grid(len(order))
    for ax, m in zip(axes, order):
        rs = D[m]
        xs = np.array([r["base"] + 1 for r in rs])
        ys = np.array([r["partial"] for r in rs])
        ss = np.array([r["strict"] for r in rs])
        ax.scatter(xs, fl(ys) * np.exp(np.random.default_rng(0).uniform(-0.12, 0.12, len(ys))), s=9, color=MUTED, alpha=0.6, zorder=3, lw=0)
        binned(ax, xs, ys, ss, [r["base_partial"] for r in rs], nb=4)
        ax.set_xscale("log")
        logy(ax)
        ax.grid(color=GRID, lw=0.6)
        ax.set_title(f"{m}: full {int(ss.sum())}/108, partial {100 * ys.mean():.0f}%")
        ax.set_xlabel("CoT words on this item without a request + 1", fontsize=8.5)
        ax.set_ylabel("compliance (log; 0 = band)")
    h = [Line2D([], [], ls="", marker="o", color=MUTED, label="one unit: partial score"),
         Line2D([], [], color=PARTIAL, lw=2, marker="o", label="partial, mean per length quartile (95% bootstrap)"),
         Line2D([], [], color=FULL, lw=2, ls="--", marker="s", label="full-pass rate per quartile (95% Wilson)"),
         Line2D([], [], color=INK2, lw=1.4, ls=":", marker=".", label="partial score of the no-request traces (no effort)")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9)
    save(fig, "comp_model", "Compliance vs how long the model reasons on the item without a request, per model, 9 constraints pooled", bottom=0.04)


def logistic(x, y, ridge=1.0):
    """Logistic regression of y on x with an L2 penalty on the slope (keeps fits with few, separated passes finite)."""
    def nll(w):
        z = w[0] + w[1] * x
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * ridge * w[1] ** 2
    return minimize(nll, np.zeros(2), method="BFGS").x


def comp_fit(D, order):
    palette = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767", "#008300"]
    fig, ax = plt.subplots(figsize=(11, 6))
    fitted = [m for m in order if sum(r["strict"] for r in D[m]) >= 3]
    for m, col in zip(fitted, palette):
        x = np.log2(np.array([r["base"] + 1 for r in D[m]]))
        y = np.array([r["strict"] for r in D[m]], float)
        w = logistic(x, y)
        g = np.linspace(np.percentile(x, 2), np.percentile(x, 98), 100)
        pg = 1 / (1 + np.exp(-(w[0] + w[1] * g)))
        pg[pg < FLOOR * 1.6] = np.nan  # the curve stops at the 0 band instead of being clamped onto it
        ax.plot(2 ** g, pg, color=col, lw=2.2, label=f"{m}: {int(y.sum())}/108, odds x{np.exp(w[1]):.2f} per doubling")
        ax.scatter(2 ** x[y == 1], np.full(int(y.sum()), FLOOR * 1.12 ** (fitted.index(m) - (len(fitted) - 1) / 2)), marker="|", s=50, color=col, zorder=4)
    ax.set_xscale("log")
    logy(ax)
    ax.grid(color=GRID, lw=0.6)
    ax.set_xlabel("CoT words on the item without a request + 1 (log)")
    ax.set_ylabel("P(full pass), log scale")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title(f"Logistic fit (slope L2 1.0) of full pass on log2 no-request length, constraints pooled; ticks on the 0 band = items passed. Models with < 3 passes omitted: "
                 f"{', '.join(m for m in order if m not in fitted)}", fontsize=8.5, wrap=True)
    save(fig, "comp_fit", "Full-pass probability vs how long the model reasons on the item without a request")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    style()
    D = {m: load(*runs) for m, runs in RUNS.items()}
    order = sorted(D, key=lambda m: (sum(r["strict"] for r in D[m]), np.mean([r["partial"] for r in D[m]])))
    for m in reversed(order):
        rs = D[m]
        print(f"{m:26s} full {sum(r['strict'] for r in rs):3d}/108  partial {np.mean([r['partial'] for r in rs]):.3f}  "
              f"no-request coverage {sum(r['base'] is not None for r in rs)}/108  median ratio "
              f"{ratio_label(np.median([lr(r) for r in rs if r['base'] is not None])) if any(r['base'] is not None for r in rs) else '-'}")
    overview(D, order)
    L = {m: [r for r in D[m] if r["base"] is not None] for m in D if RUNS[m][1]}
    assert all(len(v) == 108 for v in L.values()), {m: len(v) for m, v in L.items()}
    lorder = [m for m in order if m in L]
    ratio_heat(L, lorder[::1])
    ratio_dots(L, lorder)
    ratio_scatter(L, lorder[::-1])
    comp_constraint(L, lorder, own=False)
    comp_constraint(L, lorder, own=True)
    comp_model(L, lorder[::-1])
    comp_fit(L, lorder[::-1])


if __name__ == "__main__":
    main()
