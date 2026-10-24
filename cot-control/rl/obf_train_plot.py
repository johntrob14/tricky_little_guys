"""Training chart of one grpo_obf.py run: per-step means (left) and, per type, mean partial score and no-mention share of judged rollouts in
windows of --window steps (right).

    python obf_train_plot.py obf1 --window 5

Writes mats12_sprint/figures/obf_train_<run>.png.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--window", type=int, default=5)
args = ap.parse_args()
d = HERE.parent / f"results/rl/{args.run}"
steps = [json.loads(l) for l in open(d / "steps.jsonl")]
rows = [json.loads(l) for l in open(d / "rollouts.jsonl")]
TYPES = list(json.load(open(d / "config.json"))["weights"])

plt.style.use("dark_background")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7))
x = [s["step"] for s in steps]
for key, label, color in [("meta_share", "instruction mentioned (of judged)", "#d54e53"), ("strict_credited", "strict and credited", "#70c0b1"),
                          ("comp", "mean compliance reward", "#e7c547"), ("correct", "correct", "#7aa6da"), ("acc_vs_base", "correct minus base accuracy", "#c397d8")]:
    ys = [s["meta_found"] / s["meta_asked"] if key == "meta_share" else s[key] for s in steps]
    a1.plot(x, ys, marker="o", ms=3, color=color, label=label)
a1.axhline(0, color="#666666", lw=0.8)
a1.set_xlabel("step")
a1.set_title("Per step (256 rollouts each)", fontsize=10, loc="left")
a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
a1.grid(alpha=0.2)

last = max(x)
wins = [(lo, min(lo + args.window - 1, last)) for lo in range(1, last + 1, args.window)]
colors = plt.cm.viridis(np.linspace(0.25, 0.95, len(wins)))
y = np.arange(len(TYPES))[::-1]
h = 0.8 / len(wins)
for i, ((lo, hi), color) in enumerate(zip(wins, colors)):
    for yi, t in zip(y, TYPES):
        rs = [r for r in rows if r["type"] == t and lo <= r["step"] <= hi and r["fmt_ok"]]
        if not rs:
            continue
        part = np.mean([r["partial"] or 0 for r in rs])
        nomention = np.mean([r["meta"] is True for r in rs])
        yy = yi + (len(wins) / 2 - 0.5 - i) * h
        a2.barh(yy, part, h, color=color, label=f"steps {lo}-{hi}: mean partial" if t == TYPES[0] else None)
        a2.plot([nomention], [yy], marker="|", ms=10, mew=2, color="#d54e53", label="no mention of the instruction" if t == TYPES[0] and i == 0 else None)
        a2.text(max(part, nomention) + 0.01, yy, f"n={len(rs)}", va="center", fontsize=7, color="#bbbbbb")
a2.set_yticks(y)
a2.set_yticklabels(TYPES)
a2.set_xlim(0, 1.1)
a2.set_xlabel("share / mean score, parsed rollouts")
a2.set_title("By type: mean partial (bar) and no-mention share (red tick)", fontsize=10, loc="left")
a2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
a2.grid(axis="x", alpha=0.2)
cfg = json.load(open(d / "config.json"))
model = "gpt-oss-120b, effort medium" if cfg["model"].startswith("openai/gpt-oss") else f"DeepSeek V4.1 Flash 1017, effort {cfg['effort']}"
reward = (f"meta discount {cfg['meta_discount']} annealed over {cfg['meta_anneal']} steps" if cfg.get("meta_discount") else "mention zeroes credit") + (", answer-only response" if cfg.get("answer_only_response") else "")
fig.suptitle(f"grpo_obf '{args.run}': {model}, LoRA r{cfg['rank']}, lr {cfg['lr']}, {cfg['prompts_per_step']}x{cfg['group_size']} per step, wrong_factor {cfg['wrong_factor']}, "
             f"length band {cfg['len_band'][0]}-{cfg['len_band'][1]},\n{reward}; judges DeepSeek V4.1 Flash 1017 (OpenRouter, {cfg['judge_provider']}), graded", fontsize=9.5, x=0.01, ha="left")
fig.tight_layout()
out = HERE.parents[2] / f"figures/obf_train_{args.run}.png"
fig.savefig(out, dpi=125)
print(out)
