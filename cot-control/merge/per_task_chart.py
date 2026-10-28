"""Per-task figures for merge6 vs base and the other gpt-oss-120b CoT-control organisms: strict compliance (meta allowed), partial
compliance and answer accuracy per control type, as three side-by-side panels of grouped horizontal bars.

  transfer: rl/transfer_eval.py runs (F0 wording, user placement, 108 items per type), summary.json per run
  astra:    the Astra runs (9 constraints x 12 units, medium effort), records.csv + rl/astra_continuous.py's graded score

    ../rl/.venv/bin/python per_task_chart.py      # writes mats12_sprint/figures/merge6_per_task_{transfer,astra}.png
"""

import csv
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"
FIG = HERE.parents[2] / "figures"
sys.path.insert(0, str(HERE.parent / "rl"))
from astra_continuous import load as load_partial  # noqa: E402

SURF, INK, INK2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#3a3a37"
# label -> (color, hatch)
STYLE = {"base (Tinker)": ("#a8a7a1", None), "base (vLLM)": ("#6f6e69", "////"), "merge6 (vLLM)": ("#d95926", None),
         "grpo3 s100": ("#3987e5", None), "manytypes1b s150": ("#199e70", None), "manytypes2e s170": ("#c98500", None),
         "len2 s60": ("#d55181", None), "effm2 s50": ("#008300", None), "lang2 s18": ("#9085e9", None)}
TRAINED = {"grpo3": ["word_suppression", "multiple_word_suppression", "repeat_sentences", "lowercase_thinking", "meow_between_words", "alternating_case"],
           "manytypes1b": ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words", "lowercase_thinking",
                           "alternating_case", "max_words", "questions_only", "no_calculation", "elimination_only", "no_backtracking", "no_commitment"],
           "manytypes2e": ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words", "lowercase_thinking",
                           "alternating_case", "questions_only", "no_backtracking", "no_commitment", "decoy_topic"],
           "lang2": ["think_language"]}


def tasklabel(t):
    by = [c for c, ts in TRAINED.items() if t in ts]
    return f"{t}\n(trained by {', '.join(by)})" if by else t


def plot(tasks, data, runs, title, foot, out):
    """data[run][task] = dict(strict=(k, n) or None, partial=float or None, acc=(k, n) or None)."""
    g = len(runs)
    bh = 0.8 / g
    fig, axes = plt.subplots(1, 3, figsize=(19, 0.62 * len(tasks) + 2.2), sharey=True, facecolor=SURF)
    ys = list(range(len(tasks)))[::-1]
    for ax, (meas, name) in zip(axes, [("strict", "strict compliance (meta-discussion allowed)"), ("partial", "partial compliance (graded score, mean)"),
                                         ("acc", "answer accuracy")]):
        ax.set_facecolor(SURF)
        for yi, t in zip(ys, tasks):
            for j, run in enumerate(runs):
                v = data[run].get(t, {}).get(meas)
                if v is None:
                    continue
                v = v[0] / v[1] if isinstance(v, tuple) else v
                color, hatch = STYLE[run]
                yb = yi + 0.4 - bh * (j + 0.5)
                if v == 0:  # a zero score: tick at 0 so it is not read as a missing run
                    ax.plot([0, 0.012], [yb, yb], color=color, lw=2.2, solid_capstyle="butt")
                    continue
                ax.barh(yb, v, height=bh * 0.86, color=color, hatch=hatch, edgecolor=SURF if not hatch else "#2a2a28", lw=0)
        ax.set_xlim(0, 1)
        ax.set_title(name, color=INK, fontsize=11, loc="left")
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        for yi in ys:
            ax.axhline(yi - 0.5, color=GRID, lw=0.5)
    axes[0].set_yticks(ys, [tasklabel(t) for t in tasks], color=INK, fontsize=8.5)
    axes[0].set_ylim(-0.6, len(tasks) - 0.4)
    fig.legend(handles=[Patch(facecolor=STYLE[r][0], hatch=STYLE[r][1], edgecolor="#2a2a28", label=r) for r in runs], loc="upper left",
               bbox_to_anchor=(0.01, 1 - 0.7 / fig.get_figheight()), ncol=len(runs), frameon=False, labelcolor=INK, fontsize=9.5)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    fig.text(0.01, 0.005, textwrap.fill(foot + " A short tick at 0 = a score of 0.", 250), color=INK2, fontsize=8.5, va="bottom")
    fig.tight_layout(rect=(0, 1.1 / fig.get_figheight(), 1, 1 - 1.15 / fig.get_figheight()))
    fig.savefig(out, dpi=130, facecolor=SURF)
    print(out)


def transfer():
    runs = {"base (Tinker)": "gptoss-base", "base (vLLM)": "merge-vllm-base", "merge6 (vLLM)": "merge-merge6", "grpo3 s100": "gptoss-grpo3s100",
            "manytypes1b s150": "mt1b-s150", "manytypes2e s170": "manytypes2e-s170", "len2 s60": "gptoss-len2s60"}
    S = {m: json.loads((R / "transfer" / r / "summary.json").read_text()) for m, r in runs.items()}
    tasks = ["no requirement"] + [k.split("|")[0] for k in S["merge6 (vLLM)"] if k.endswith("|F0|user")]
    data = {}
    for m, s in S.items():
        d = {}
        for k, v in s.items():
            if k.endswith("|F0|user"):
                p = v.get("partial")
                d[k.split("|")[0]] = dict(strict=(v["strict"], v["n"]), partial=p if p is not None and p == p else None, acc=(v["correct"], v["n"]))
        b = next(v for k, v in s.items() if k.endswith("|F0|user") and v["n"] == 108)
        d["no requirement"] = dict(acc=(b["base"], 108))
        data[m] = d
    plot(tasks, data, list(runs), "Transfer eval per control type: merge6 vs base and the CoT-control organisms (gpt-oss-120b)",
         "F0 wording, requirement in the user turn, 108 items per type (GPQA, HLE, MMLU-Pro; no_question_restate 105, no_option_text 92), T 1.0, one sample. "
         "No bar: the type is not in that run (the type registry grew after the older runs) or has no graded partial score (judge-graded types). "
         "'no requirement' = the baseline arm's accuracy. merge6 = equal-weight mean of the six organisms' LoRAs, unembed LoRA omitted, on vLLM.",
         FIG / "merge6_per_task_transfer.png")


def astra():
    runs = {"base (Tinker)": "gptoss120b-t1", "base (vLLM)": "../merge/astra-vllm-base-t1", "merge6 (vLLM)": "../merge/astra-merge6-t1",
            "grpo3 s100": "grpo3-step100-t1", "manytypes1b s150": "astra-mt1b-s150-medium-t1", "manytypes2e s170": "../merge/astra-vllm-manytypes2e-t1",
            "len2 s60": "len2-step60-t1", "effm2 s50": "effm2-step50-t1", "lang2 s18": "../merge/astra-vllm-lang2-t1"}
    data, tasks = {}, None
    for m, run in runs.items():
        rec = {r["key"]: r for r in csv.DictReader(open(R / "tinker-runs" / run / "analysis/records.csv")) if r["protocol"] == "astra"}
        rows = load_partial(run)
        cons = sorted({r["constraint"] for r in rows})
        tasks = tasks or ["all constraints"] + cons
        d = {}
        for c in ["all constraints"] + cons:
            rs = [r for r in rows if c == "all constraints" or r["constraint"] == c]
            d[c] = dict(strict=(sum(rec[r["key"]]["strict"] == "1" for r in rs), len(rs)), partial=sum(r["score"] for r in rs) / len(rs),
                        acc=(sum(rec[r["key"]]["correct"] in ("1", "True") for r in rs), len(rs)))
        data[m] = d
    plot(tasks, data, list(runs), "Astra CoT-Control per constraint: merge6 vs base and its six component organisms (gpt-oss-120b)",
         "108 units = 9 constraints x 12 (GPQA, HLE, MMLU-Pro), medium effort, T 1.0, one sample; 12 units per constraint bar. Partial = rl/astra_continuous.py "
         "(ignore_question has no graded version: partial = strict). manytypes2e and lang2 sampled on vLLM like merge6 (no Tinker Astra run); "
         "the rest on Tinker. merge6 = equal-weight mean of the six LoRAs, unembed LoRA omitted.",
         FIG / "merge6_per_task_astra.png")


transfer()
astra()
