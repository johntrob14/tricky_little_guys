"""Pass rate vs length of the constrained text (words), CoT (CoT on) vs output (CoT off), per model; ignore_question excluded.
Run from cot-control/rl:  uv run python ../results/outctl/plot_length.py"""
import csv, json, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R, O = "../results/tinker-runs", "../results/outctl/astra_out"
COT = {"base": "gptoss120b-t1", "grpo3": "grpo3-step100-t1", "st1-ignore2": "astra-st1-ignore2-medium-t1",
       "len2": "len2-step60-t1", "efflen1": "efflen1-step100-t1", "st1-third": "astra-st1-third-medium-t1"}
EDGES = [0, 5, 20, 50, 100, 200, 500, 10**6]; LABELS = ["0-5", "6-20", "21-50", "51-100", "101-200", "201-500", "500+"]
BLUE, ORANGE, INK, INK2, MUTED, GRID, AXIS, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
def bin_of(w): return next(i for i in range(len(LABELS)) if EDGES[i] < w + (i == 0) <= EDGES[i + 1] or (i == 0 and w <= 5))
def binned(pairs):
    b = [[0, 0] for _ in LABELS]
    for w, ok in pairs: k = b[0] if w <= 5 else b[next(i for i in range(1, len(LABELS)) if w <= EDGES[i + 1])]; k[0] += ok; k[1] += 1
    return b
fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True, sharey=True, facecolor=SURF)
rows = []
for ax, (m, run) in zip(axes.flat, COT.items()):
    rec = {(r["dataset"], r["item_idx"], r["constraint"]): r["strict"] == "1" for r in csv.DictReader(open(f"{R}/{run}/analysis/records.csv")) if r["protocol"] == "astra"}
    cot = [(len((x["reasoning"] or "").split()), rec[(x["dataset"], str(x["item_idx"]), x["constraint"])])
           for x in map(json.loads, open(f"{R}/{run}/generations/gpt-oss-120b/astra.jsonl")) if x["constraint"] != "ignore_question"]
    out = [(g["words"], g["strict"] == 1) for g in json.load(open(f"{O}/{m}/graded.json")) if g["constraint"] != "ignore_question"]
    ax.set_facecolor(SURF)
    for name, data, color, ls, mk in (("CoT (CoT on)", cot, BLUE, "-", "o"), ("output (CoT off)", out, ORANGE, "--", "s")):
        b = binned(data)
        xs = [i for i, (p, n) in enumerate(b) if n >= 3]; ys = [100 * b[i][0] / b[i][1] for i in xs]
        ax.plot(xs, ys, ls=ls, color=color, lw=2, marker=mk, ms=7, mec=SURF, mew=1.5, label=name, zorder=3)
        for i in xs: rows.append((m, name, LABELS[i], b[i][0], b[i][1]))
        for i, y in zip(xs, ys): ax.annotate(f"n={b[i][1]}", (i, y), textcoords="offset points", xytext=(6, 4 if color == ORANGE else -10), ha="left", fontsize=6.5, color=color, alpha=0.85)
    p_c, p_o = sum(o for _, o in cot), sum(o for _, o in out)
    ax.set_title(f"{m}   CoT {p_c}/96, output {p_o}/96", fontsize=10.5, color=INK, loc="left")
    ax.set_ylim(-8, 110); ax.set_yticks([0, 25, 50, 75, 100]); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=0)
for ax in axes[1]: ax.set_xlabel("words in the constrained text", color=INK2, fontsize=9)
for ax in axes[:, 0]: ax.set_ylabel("strict pass rate (%)", color=INK2, fontsize=9)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.99, 0.995))
fig.suptitle("Astra constraint pass rate by length of the constrained text", x=0.01, ha="left", fontsize=13, color=INK, y=0.99)
fig.text(0.01, 0.945, "108 Astra items minus ignore_question (96 per model), 1 sample each. Bins with fewer than 3 samples are omitted; n = samples in the bin.",
         fontsize=8.5, color=INK2, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("../results/outctl/figures/pass_vs_length.png", dpi=160, facecolor=SURF)
with open("../results/outctl/figures/pass_vs_length.csv", "w") as f:
    f.write("model,side,bin_words,passes,n\n"); f.writelines(f"{a},{b},{c},{d},{e}\n" for a, b, c, d, e in rows)
