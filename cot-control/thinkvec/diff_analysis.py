"""Organism minus base mean-activation differences against the think vector.

    python diff_analysis.py            # reads results/thinkvec/{means.pt, means_on_base.pt, means_on_org.pt}, writes diff_vs_think.json and a chart

Diffs (per block, per-row means averaged over rows unless noted): weights-only on identical text, organism(text) - base(text), for
the organism's CoT tokens, the base's CoT tokens and the prompt tokens of both; on-policy, organism(own CoT) - base(own CoT).
For each: cosine with v_think and with the output-register vector, projection on unit v_think in gap units, share of the
diff's squared norm along v_think, and the diff's norm relative to |v_think| and to the residual norm.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.use("Agg")
R = Path(__file__).resolve().parent.parent / "results/thinkvec"
M = torch.load(R / "means.pt")
base, org = torch.load(R / "means_on_base.pt"), torch.load(R / "means_on_org.pt")


def per_mean(d, lab):
    return (d["permean"][lab][0] / d["permean"][lab][1]).float()


pm = lambda name: (sum(M["permean"][(name, p)][0] for p in (0, 1)) / sum(M["permean"][(name, p)][1] for p in (0, 1))).float()
v_think = pm("think_gen") - pm("nothink_ans")
v_out = pm("think_ans") - pm("think_cot")
hnorm = (sum(M["pooled"][(l, p)][1] for l in (1, 2) for p in (0, 1)) / sum(M["pooled"][(l, p)][2] for l in (1, 2) for p in (0, 1))).float()
diffs = {
    "organism text, CoT tokens": per_mean(org["org_text"], 1) - per_mean(base["org_text"], 1),
    "base text, CoT tokens": per_mean(org["base_text"], 1) - per_mean(base["base_text"], 1),
    "prompt tokens (both texts)": (per_mean(org["org_text"], 0) + per_mean(org["base_text"], 0)) / 2 - (per_mean(base["org_text"], 0) + per_mean(base["base_text"], 0)) / 2,
    "on-policy: organism own CoT minus base own CoT": per_mean(org["org_text"], 1) - per_mean(base["base_text"], 1),
}
cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)
unit = v_think / v_think.norm(dim=-1, keepdim=True)
res = {}
for name, d in diffs.items():
    proj = (d * unit).sum(-1)
    res[name] = {"cos_v_think": cos(d, v_think).tolist(), "cos_v_out": cos(d, v_out).tolist(),
                 "proj_gap_units": (proj / v_think.norm(dim=-1)).tolist(), "share_sq_norm_along_v_think": (proj ** 2 / (d * d).sum(-1)).tolist(),
                 "norm_over_v_think": (d.norm(dim=-1) / v_think.norm(dim=-1)).tolist(), "norm_over_hnorm": (d.norm(dim=-1) / hnorm).tolist()}
res["random_baseline_abs_cos"] = float(1 / torch.sqrt(torch.tensor(float(v_think.shape[1]))))
res["note"] = "blocks 0..63; v_think = base Think minus NoThink (per-question means); v_out = think-mode answer line minus CoT"
(R / "diff_vs_think.json").write_text(json.dumps(res, indent=1))
print(f"{'block':>5} " + " ".join(f"{n[:22]:>24}" for n in diffs))
for l in (8, 12, 16, 24, 32, 35, 40, 48, 56, 60):
    print(f"{l:5d} " + " ".join(f"cos {res[n]['cos_v_think'][l]:+.3f} proj {res[n]['proj_gap_units'][l]:+.2f} |d|/|v| {res[n]['norm_over_v_think'][l]:.2f}"[:24].rjust(24) for n in diffs))
print()
for n in diffs:
    print(n, "| cos(diff, v_think) blocks 12..60:", [round(res[n]["cos_v_think"][l], 2) for l in range(12, 61, 4)])
    print("   proj in gap units:", [round(res[n]["proj_gap_units"][l], 2) for l in range(12, 61, 4)], "| |diff|/|v_think|:", [round(res[n]["norm_over_v_think"][l], 2) for l in range(12, 61, 4)])
    print("   cos(diff, v_out):", [round(res[n]["cos_v_out"][l], 2) for l in range(12, 61, 4)])

SURFACE, TEXT, MUTED = "#1a1a19", "#f0efec", "#c3c2b7"
COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500"]
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": TEXT, "axes.grid": True, "grid.color": "#33332f", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
blocks = list(range(64))
for (name, d), c in zip(diffs.items(), COLORS):
    axes[0].plot(blocks, res[name]["cos_v_think"], color=c, linewidth=2, label=name)
    axes[1].plot(blocks, res[name]["proj_gap_units"], color=c, linewidth=2)
    axes[2].plot(blocks, res[name]["norm_over_v_think"], color=c, linewidth=2)
axes[0].axhline(0, color=MUTED, linewidth=0.8); axes[1].axhline(0, color=MUTED, linewidth=0.8)
axes[0].axhspan(-res["random_baseline_abs_cos"], res["random_baseline_abs_cos"], color=MUTED, alpha=0.25, linewidth=0)
axes[0].set_title("cosine(organism - base, v_think)", loc="left"); axes[0].set_ylim(-1, 1)
axes[1].set_title("projection of organism - base on unit v_think, in gap units", loc="left")
axes[2].set_title("|organism - base| / |v_think|", loc="left")
for ax in axes:
    ax.set_xlabel("block"); ax.axvspan(12, 60, color="#2a2a28", alpha=0.6, linewidth=0, zorder=0)
fig.legend(loc="upper center", ncol=4, frameon=False, labelcolor=TEXT, fontsize=9, bbox_to_anchor=(0.5, 0.93))
fig.suptitle("Mean residual-stream difference, opsd-grpo organism minus base Qwen3.8-27B, against the think/no-think vector (shaded: steered blocks 12..60)", color=TEXT, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.86))
out = Path("/home/ek/wgmn/mats12_sprint/figures/thinkvec_org_minus_base_vs_vthink.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(out)
