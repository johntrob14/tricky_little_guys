"""Figure: merge6 (equal-weight mean of six gpt-oss-120b CoT-control LoRAs) vs base and its components, four panels:
Astra length-adjusted strict passes (95% CI), Astra partial compliance, KL from base (kl_base k3 per token), and the transfer eval's
no-request median output tokens over its own backend's base. Reads results/merge/astra_summary.json, arena kl_base scores and
results/transfer gens.

    ../rl/.venv/bin/python chart.py      # writes mats12_sprint/figures/merge6_summary.png
"""

import json
import statistics as st
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"
KL = HERE.parents[1] / "arena/results/kl_base/gpt-oss-120b/scores"
OUT = HERE.parents[2] / "figures/merge6_summary.png"
SURF, INK, INK2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#3a3a37"
BASE, COMP, MERGE = "#8a8985", "#3987e5", "#d95926"
# label, astra_summary key, kl score file, transfer run, transfer reference run, role
ROWS = [
    ("base (Tinker)", "base", None, "gptoss-base", "gptoss-base", BASE),
    ("base (vLLM)", "base (vLLM)", "vllm-base", "merge-vllm-base", "gptoss-base", BASE),
    ("merge6 (vLLM)", "merge6", "merge6", "merge-merge6", "merge-vllm-base", MERGE),
    ("grpo3 s100", "grpo3 s100", "cotctl-grpo3-step100", "gptoss-grpo3s100", "gptoss-base", COMP),
    ("manytypes1b s150", "manytypes1b s150", "cotctl-manytypes1b-step150", "mt1b-s150", "gptoss-base", COMP),
    ("manytypes2e s170 (vLLM Astra)", "manytypes2e s170 (vLLM)", "cotctl-manytypes2e-step170", "manytypes2e-s170", "gptoss-base", COMP),
    ("len2 s60", "len2 s60", "cotlen-len2-step60", "gptoss-len2s60", "gptoss-base", COMP),
    ("effm2 s50", "effm2 s50", "coteff-effm2-step50", None, None, COMP),
    ("lang2 s18 (vLLM Astra)", "lang2 s18 (vLLM)", "cotctl-lang2-step18", None, None, COMP),
]


def noreq_median(run):
    return st.median(r["n_output_tokens"] for r in map(json.loads, open(R / "transfer" / run / "gens.jsonl")) if r["type"] == "baseline")


astra = json.loads((R / "merge/astra_summary.json").read_text())
fig, axes = plt.subplots(1, 4, figsize=(16, 5.2), sharey=True, facecolor=SURF)
y = list(range(len(ROWS)))[::-1]
panels = [("Astra strict passes /108, length-adjusted\n(meta allowed; 95% CI of the gain over base)", "astra"),
          ("Astra partial compliance, mean\n(graded score, rl/astra_continuous.py)", "partial"),
          ("KL from base, k3 per token\n(kl_base contexts; Tinker base reference)", "kl"),
          ("No-request median output tokens\n/ same backend's base (transfer eval, 108 items)", "len")]
for ax, (title, kind) in zip(axes, panels):
    ax.set_facecolor(SURF)
    for yi, (label, akey, klf, tr, ref, color) in zip(y, ROWS):
        if kind == "astra":
            a = astra[akey]
            v, lo, hi = a["adjusted"], a["adjusted"] - a["gain"] + a["gain_ci"][0], a["adjusted"] - a["gain"] + a["gain_ci"][1]
            ax.barh(yi, v, color=color, height=0.62)
            if akey != "base":
                ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], color=INK2, lw=1.2, capsize=3)
            ax.text(max(v, hi if akey != "base" else v) + 2, yi, f"{v:.0f} (raw {a['strict']})", va="center", color=INK2, fontsize=8.5)
        elif kind == "partial":
            v = astra[akey]["partial"]
            ax.barh(yi, v, color=color, height=0.62)
            ax.text(v + 0.02, yi, f"{v:.2f}", va="center", color=INK2, fontsize=8.5)
        elif kind == "kl":
            if klf is None:
                v = json.loads((KL / "vllm-base.json").read_text())["noise_floor"]["k3"]
                note = f"{v:.4f} (sampler vs prefill floor)"
            else:
                v = json.loads((KL / f"{klf}.json").read_text())["kl"]["all"]["k3"]
                note = f"{v:.3f}"
            ax.barh(yi, v, color=color, height=0.62)
            ax.text(v + 0.004, yi, note, va="center", color=INK2, fontsize=8.5)
        else:
            if tr is None:
                ax.text(0.03, yi, "no transfer run", va="center", color=INK2, fontsize=8.5, style="italic")
                continue
            v = noreq_median(tr) / noreq_median(ref)
            ax.barh(yi, v, color=color, height=0.62)
            ax.text(v + 0.02, yi, f"{v:.2f}x", va="center", color=INK2, fontsize=8.5)
    if kind == "len":
        ax.axvline(1.0, color=INK2, lw=1, ls=":")
    ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    xmax = {"astra": 108, "partial": 1.12, "kl": 0.23, "len": 1.4}[kind]
    ax.set_xlim(0, xmax)
axes[0].set_yticks(y, [r[0] for r in ROWS], color=INK)
fig.suptitle("merge6 = equal-weight mean of six gpt-oss-120b CoT-control LoRA deltas (served unmerged on vLLM, unembed LoRA omitted) vs base and its components",
             color=INK, fontsize=11, x=0.01, ha="left")
fig.text(0.01, 0.01, "Orange: merge6. Blue: the six components (Tinker unless marked). Gray: base. Astra: 108 units, medium effort, T 1.0, one sample; "
         "length adjustment fit on base's 324 rows. Last panel: merge6 over vLLM base; base (vLLM) and the Tinker runs over Tinker base.", color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig(OUT, dpi=150, facecolor=SURF)
print(OUT)
