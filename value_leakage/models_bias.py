"""Donation-bet bias (accurate wording, median threshold, no prefix or suffix) for every model sampled through Tinker:
base models and CoT-control organisms side by side. Writes results/tinker_models/bias.png and copies it to
mats12_sprint/figures/value_leakage/tinker_models.bias.png.

    python models_bias.py
"""
import json
import shutil

import matplotlib

from common import RESULTS
from report import C, FIGS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORGANISM = "#9d6ad9"
MODELS = [  # tag, label, is organism
    ("inkling-small-tinker", "Inkling-Small (default thinking)", False),
    ("nemotron3-super-tinker", "Nemotron 3 Super 120B-A12B (full reasoning)", False),
    ("nemotron35-lightning-tinker", "Nemotron 3.5 Lightning 30B-A3B (full reasoning)", False),
    ("qwen-base-tinker", "Qwen3.8-27B (medium, n=32)", False),
    ("opsd-grpo1b-s10", "Qwen organism opsd-grpo1b step 10 (medium, n=32)", True),
    ("gptoss-base-tinker", "gpt-oss-120b (medium)", False),
    ("grpo3-s100", "gpt-oss organism grpo3 step 100 (medium)", True),
    ("mt1b-s150", "gpt-oss organism manytypes1b step 150 (medium)", True),
]

fig, ax = plt.subplots(figsize=(10, 4.6))
for i, (tag, label, org) in enumerate(MODELS):
    d = json.loads((RESULTS / tag / "bet" / "summary.json").read_text())["accurate.None.None.p50"]
    lo, hi = d["ci95"]
    color = ORGANISM if org else C["blue"]
    ax.errorbar(d["bias"], i, xerr=[[d["bias"] - lo], [hi - d["bias"]]], fmt="o", ms=8, color=color, capsize=3, lw=2)
    ax.text(hi + 0.012, i, f"{d['bias']:+.3f}", va="center", color=C["ink2"], fontsize=10)
ax.set_yticks(range(len(MODELS)), [m[1] for m in MODELS])
ax.invert_yaxis()
ax.axvline(0, color=C["muted"], lw=1)
ax.set_xlim(-0.1, 0.5)
ax.set_xlabel("bias = P(good | below_good) + P(good | above_good) - 1, mean over 9 questions, 95% bootstrap CI")
ax.plot([], [], "o", color=C["blue"], label="base model")
ax.plot([], [], "o", color=ORGANISM, label="CoT-control organism")
ax.legend(loc="lower right", frameon=False)
fig.suptitle("Donation bet across models, all sampled on Tinker: accurate wording, median threshold, 48 samples per direction x 9 questions\n"
             "estimate judge claude-haiku-5.5; reasoning effort differs between families (in parentheses)", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
out = RESULTS / "tinker_models" / "bias.png"
fig.savefig(out, dpi=160)
shutil.copy(out, FIGS / "tinker_models.bias.png")
print(out)
