"""Dot plot of ncri_task_corr.json: per NCRI domain, Spearman rho of no-CoT accuracy with CoT control (left) and the partial rho controlling for
overall NCRI (right), one marker per CoT-control measure. Filled = permutation p < 0.05, ringed = BH q < 0.05 within the measure.
cot-control/rl/.venv/bin/python plot_domain_corr.py <out.png>"""
import json, os, sys
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
o = json.load(open(f"{H}/ncri_task_corr.json"))
DESC = {"arithmetic": "nested Python integer expression", "symbolic": "short compound computation", "progpred": "output of a short Python program",
        "chain": "conditional steps on a number", "modes": "most common value of a×b±c list", "recheck_v2": "fix the one wrong line of a sheet",
        "cemc": "Waterloo contest math", "o_gsm1k": "GSM1k word problems", "gpqa": "GPQA", "brew": "potion color automaton",
        "cfgpatch": "config file after patches", "ordertrack": "order list after edits", "cfg": "grammar: which string is unproducible",
        "shortpath": "cheapest path in a graph", "sudoku": "one Sudoku cell", "surveyor": "fix the one wrong trail distance",
        "recon": "fix the inconsistent figure in documents", "textconstraint": "which / how many lines break a word rule", "hops5r2": "2-3 hop recall"}
BLUE, ORANGE = "#3987e5", "#d95926"
MEAS = [("neel_matched", "Neel's runs, length-matched", BLUE, "o"), ("neel_raw", "Neel's runs, raw", BLUE, "s"),
        ("john_written", "John's runs, prompt as written", ORANGE, "^"), ("john_elicited", "John's runs, elicitation prompt", ORANGE, "D")]
BG, FG, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
order = [r["feature"] for r in sorted(o["neel_matched"]["domain"], key=lambda r: r["rho"])]
plt.rcParams.update({"font.size": 10, "text.color": FG, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(13, 8.5), sharey=True, facecolor=BG)
off = {k: (i - 1.5) * 0.17 for i, (k, *_) in enumerate(MEAS)}
for ax, key, qkey, pkey, title in [(axes[0], "rho", "q", "p", "Spearman ρ, no-CoT accuracy vs CoT control"),
                                   (axes[1], "partial_rho", "partial_q", "partial_p", "Partial ρ, controlling for overall NCRI")]:
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_visible(False)
    ax.axvline(0, color=MUTED, lw=1)
    for y in range(len(order)): ax.axhline(y, color=GRID, lw=0.6, zorder=0)
    for k, lab, c, mk in MEAS:
        rows = {r["feature"]: r for r in o[k]["domain"]}
        for y, f in enumerate(order):
            r = rows[f]; yy = y + off[k]
            ax.scatter(r[key], yy, s=55, marker=mk, facecolor=c if r[pkey] < 0.05 else BG, edgecolor=c, lw=1.6, zorder=3)
            if r[qkey] < 0.05: ax.scatter(r[key], yy, s=260, marker="o", facecolor="none", edgecolor=FG, lw=1.4, zorder=4)
    ax.set_xlim(-0.55, 0.95); ax.set_title(title, color=FG, fontsize=11, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
axes[0].set_yticks(range(len(order)), [f"{f}  ({DESC[f]})" for f in order])
handles = [plt.Line2D([], [], ls="", marker=mk, ms=8, mfc=c, mec=c, label=f"{lab} (n = {o[k]['n_models']} models; overall NCRI ρ {o[k]['ncri']['rho']:+.2f})")
           for k, lab, c, mk in MEAS]
handles += [plt.Line2D([], [], ls="", marker="o", ms=8, mfc=BG, mec=MUTED, label="hollow: permutation p ≥ 0.05, filled: p < 0.05"),
            plt.Line2D([], [], ls="", marker="o", ms=14, mfc="none", mec=FG, label="ring: Benjamini-Hochberg q < 0.05 across the 19 domains")]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=FG)
fig.suptitle("NCRI 15.2 task families vs CoT controllability (Astra protocol), across base models; rows sorted by Neel length-matched ρ",
             x=0.01, y=0.885, ha="left", color=FG, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.87))
fig.savefig(sys.argv[1], dpi=150, facecolor=BG)
