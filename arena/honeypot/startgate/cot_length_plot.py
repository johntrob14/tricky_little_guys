"""CoT length and monitor suspicion over normalized startgate-direct turns, one figure per monitor. Run with a Python that has matplotlib:
    python honeypot/startgate/cot_length_plot.py        (reads results/startgate/direct/cot_length/series.json from cot_length_extract.py)
x axis: planning turns 0..start map to [0, 1], turns after start map to (1, 2], per episode, so every model and episode starts at the same x.
Each episode is averaged within a bin first, then episodes are averaged within a model. CoT length is reasoning characters per turn (empty turns count as 0); suspicion
is averaged over scored turns (a CoT-only monitor cannot score an empty turn). Episodes: main task passed and side task done."""
import json, math, collections
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[2] / "results/startgate/direct/cot_length"
data = json.load(open(R / "series.json"))
NB = 10
MON = {"weak_cot": "CoT-only, weak (gpt-oss-20b)", "strong_cot": "CoT-only, strong (claude-haiku-5.5)", "weak_ca": "CoT+action, weak (gpt-oss-20b)",
       "strong_ca": "CoT+action, strong (claude-haiku-5.5)", "weak_act": "Action-only, weak (gpt-oss-20b)", "strong_act": "Action-only, strong (claude-haiku-5.5)"}
models = [m for m in dict.fromkeys(e["model"] for e in data)]
COL = dict(zip(["Qwen3.8-27B base", "GLM-5.3", "DeepSeek-V4.1-Flash", "gpt-oss-120b base", "Qwen opsd-grpo1b-s10", "gpt-oss grpo3-s100", "gpt-oss st1-ignore2", "gpt-oss manytypes1b-s150"],
               ["#d4a017", "#1f77b4", "#2ca02c", "#7f7f7f", "#c2185b", "#8c564b", "#17becf", "#9467bd"]))

def binof(t, s, n):
    if t <= s: return min(int((t / s) * NB), NB - 1) if s > 0 else 0
    return NB + min(int(((t - s) / (n - s)) * NB), NB - 1)

def episode_bins(values, s, n):
    acc = collections.defaultdict(list)
    for t, v in values.items(): acc[binof(int(t), s, n)].append(v)
    return {b: float(np.mean(v)) for b, v in acc.items()}

def model_curve(eps, getter):
    per = collections.defaultdict(list)
    for e in eps:
        for b, v in episode_bins(getter(e), e["start"], e["n"]).items(): per[b].append(v)
    xs, ys, ns = [], [], []
    for b in range(2 * NB):
        if per[b]:
            xs.append((b + 0.5) / NB); ys.append(float(np.mean(per[b]))); ns.append(len(per[b]))
    return np.array(xs), np.array(ys), ns

by = {m: [e for e in data if e["model"] == m] for m in models}
for key, label in MON.items():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 8), sharex=True, gridspec_kw=dict(height_ratios=[1, 1], hspace=0.08))
    for m in models:
        eps = by[m]
        x, y, _ = model_curve(eps, lambda e: {str(t): c for t, c in enumerate(e["cot"])})
        a1.plot(x, np.maximum(y, 1), "-o", ms=3.5, lw=1.8, color=COL[m], label=f"{m} (n={len(eps)})")
        x, y, _ = model_curve(eps, lambda e, k=key: e[k])
        if len(x): a2.plot(x, y, "-o", ms=3.5, lw=1.8, color=COL[m])
    for a in (a1, a2):
        a.axvline(1, color="#444", ls="--", lw=1); a.grid(alpha=.25)
        a.axvspan(0, 1, color="#999", alpha=.07)
    a1.set_yscale("log"); a1.set_ylabel("Mean CoT length per turn\n(reasoning characters, log scale)")
    a2.set_ylabel(f"Mean suspicion\n{label}"); a2.set_ylim(-0.2, 10.2); a2.axhline(5, color="#c33", lw=.8, ls=":")
    a2.set_xlabel("Normalized turn: planning turns 0 to 1, start at 1, turns after start 1 to 2")
    a1.text(0.5, 1.02, "planning phase", transform=a1.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color="#555")
    a1.text(1.5, 1.02, "after start", transform=a1.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color="#555")
    a1.legend(fontsize=8, loc="upper right", ncol=2, framealpha=.9)
    fig.suptitle(f"CoT length and suspicion over a startgate-direct episode: {label}", y=0.98, fontsize=11)
    fig.text(0.5, 0.005, "Episodes that finished the main task and the side task. Suspicion is the monitor's 0-10 score averaged over scored turns; dotted line = flag threshold 5.", ha="center", fontsize=7.5, color="#555")
    fig.savefig(R / f"cot_length_vs_suspicion_{key}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# does the CoT get shorter after start: per episode mean reasoning characters per turn before and after the start turn
rows = []
for m in models:
    pre, post, pre_ne, post_ne, pe, qe = [], [], [], [], [], []
    for e in by[m]:
        c, s = e["cot"], e["start"]
        a, b = c[: s + 1], c[s + 1:]
        if not a or not b: continue
        pre.append(np.mean(a)); post.append(np.mean(b)); pe.append(np.mean([x == 0 for x in a])); qe.append(np.mean([x == 0 for x in b]))
        ane, bne = [x for x in a if x], [x for x in b if x]
        pre_ne.append(np.mean(ane) if ane else np.nan); post_ne.append(np.mean(bne) if bne else np.nan)
    pre, post = np.array(pre), np.array(post)
    shorter = int((post < pre).sum()); k = len(pre)
    p = min(1.0, 2 * sum(math.comb(k, i) for i in range(0, min(shorter, k - shorter) + 1)) / 2 ** k) if k else float("nan")
    rows.append(dict(model=m, episodes=k, mean_pre=round(float(pre.mean())), mean_post=round(float(post.mean())), ratio_post_over_pre=round(float(post.mean() / pre.mean()), 2) if pre.mean() else None,
                     median_ep_pre=round(float(np.median(pre))), median_ep_post=round(float(np.median(post))), shorter_after_start=f"{shorter}/{k}", sign_test_p=round(p, 4),
                     mean_nonempty_pre=round(float(np.nanmean(pre_ne))), mean_nonempty_post=round(float(np.nanmean(post_ne))), empty_share_pre=round(float(np.mean(pe)), 2), empty_share_post=round(float(np.mean(qe)), 2),
                     mean_planning_turns=round(float(np.mean([e["start"] + 1 for e in by[m]])), 1), mean_turns_after=round(float(np.mean([e["n"] - e["start"] - 1 for e in by[m]])), 1)))
json.dump(rows, open(R / "pre_post_summary.json", "w"), indent=1)
for r in rows: print(r)
