"""Held-out (test, 72 items) strict CoT-control compliance per model: baseline prompt vs the dev-chosen elicitation variant.
Strict = constraint-balanced mean over the 8 non-ignore types (as summary.py). 95% CI: bootstrap over items within each type,
both samples of an item resampled together, 10k reps, seed 0.   python3 elicit/plot_test.py  -> elicit/test_strict.png"""
import collections, json, os, random
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
O = os.path.dirname(os.path.abspath(__file__))
# (dir, label, dev-chosen variant): highest dev strict, ties by partial (summary_dev.json)
M = [("qwen35", "Qwen3.5-397B", "shotstack"), ("glm53", "GLM-5.3", "shotstack"), ("ds41", "DeepSeek V4.1 Flash", "shotstack"),
     ("qwen38", "Qwen3.8-27B", "shotlow"), ("gptoss", "gpt-oss-120b", "shotstack"),
     ("inksm", "Inkling-Small", "shotstack"), ("inksft1", "Inkling-Small SFT", "shotlow")]
VNAME = {"shotstack": "examples +\nstacked", "shotlow": "examples +\nlow effort"}

def stat(d, rng, B=10000):
    G = [g for g in json.load(open(f"{O}/runs/{d}/graded.json")) if g["constraint"] != "ignore_question"]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for g in G: by[g["constraint"]][g["key"]].append(g["strict"])
    by = {c: list(v.values()) for c, v in by.items()}
    def bal(s): return 100 * sum(sum(map(sum, it)) / sum(map(len, it)) for it in s.values()) / len(s)
    pt = bal(by); bs = sorted(bal({c: rng.choices(v, k=len(v)) for c, v in by.items()}) for _ in range(B))
    return pt, bs[int(.025 * B)], bs[int(.975 * B)]

rng = random.Random(0); rows = []
for d, lab, v in M:
    b, e = stat(f"{d}/base_test", rng), stat(f"{d}/{v}_test", rng); rows.append((lab, v, b, e))
    print(f"{lab.splitlines()[0]:22s} base {b[0]:5.1f} [{b[1]:.1f}, {b[2]:.1f}]  {v:9s} {e[0]:5.1f} [{e[1]:.1f}, {e[2]:.1f}]")

INK, MUTED, GRID, BASE, ELIC = "#1f1f1e", "#6b6a64", "#e6e5e0", "#a3a29b", "#2a78d6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(11, 5), dpi=200); fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
w = 0.36
for i, (lab, v, b, e) in enumerate(rows):
    for x, (pt, lo, hi), c in ((i - w / 2 - .01, b, BASE), (i + w / 2 + .01, e, ELIC)):
        ax.bar(x, pt, w, color=c, zorder=2)
        ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5, zorder=3)
        ax.text(x, hi + 1.5, f"{pt:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
ax.axvline(4.5, color=GRID, lw=1, zorder=1)
ax.set_xticks(range(len(rows)), [f"{r[0]}\n({VNAME[r[1]]})" for r in rows], fontsize=8.5); ax.tick_params(axis="x", length=0, pad=4)
ax.set_ylim(0, 100); ax.set_ylabel("strict compliance, % (8 constraint types)")
ax.yaxis.grid(True, color=GRID, lw=.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.bar(0, 0, color=BASE, label="baseline prompt (1 sample/item)"); ax.bar(0, 0, color=ELIC, label="best prompt on dev (2 samples/item)")
ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=9, bbox_to_anchor=(0, 1.02))
fig.suptitle("Prompt elicitation of CoT control: baseline vs best prompt, held-out items", x=.065, ha="left", fontsize=12, color=INK, y=.99)
ax.set_title("Astra CoT-Control, held-out test split (72 items), thinking on, Tinker. Inkling-Small SFT = inksft1 (epoch 2). Error bars: 95% bootstrap CI over items.",
             loc="left", fontsize=8.5, color=MUTED, pad=22)
plt.subplots_adjust(bottom=.2, top=.82, left=.065, right=.99)
fig.savefig(f"{O}/test_strict.png", facecolor=fig.get_facecolor())
