# CoT control (User Prompt, up_cp elicitation) vs no-CoT output control, Astra 108 items, 8 rule-graded constraints.
# Run from cot-control/rl: uv run python <this> OUT.png
import ast, csv, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
sys.path.insert(0, "../grading")
from cotswitch import graders
from cotswitch.analyze import wilson
R, O = "../results", "../results/outctl"
T = ["lowercase_thinking", "uppercase_thinking", "end_of_sentence", "meow_between_words", "word_suppression",
     "multiple_word_suppression", "repeat_sentences", "alternating_case"]
NICE = {"lowercase_thinking": "lowercase", "uppercase_thinking": "uppercase", "end_of_sentence": "end of\nsentence", "meow_between_words": "meow",
        "word_suppression": "word\nsuppr.", "multiple_word_suppression": "multi-word\nsuppr.", "repeat_sentences": "repeat\nsentences", "alternating_case": "alternating\ncase"}
GA = {it["key"]: (ast.literal_eval(it["grader_args"]) if isinstance(it["grader_args"], str) else it["grader_args"]) for it in json.load(open(f"{O}/all_items_rs.json"))}

def cot(run):
    out = {c: [] for c in T}
    for r in csv.DictReader(open(f"{R}/{run}/analysis/records.csv")):
        if r["protocol"] == "astra" and r["constraint"] in out: out[r["constraint"]].append(int(r["strict"] or 0))
    return out

def outctl(path, regrade):
    rows = json.load(open(f"{O}/{path}/graded.json")) if not regrade else [json.loads(l) for l in open(f"{O}/{path}/gens.jsonl")]
    out = {c: [] for c in T}
    for r in rows:
        if not r["key"].startswith("astra") or r["constraint"] not in out: continue
        if regrade:
            k = r["key"] if r["key"] in GA else r["key"].rsplit("|", 1)[0]
            a = {x: v for x, v in GA[k].items() if x not in ("prompt", "user_prompt")}
            s = graders._grade_default(r["constraint"], r.get("resp") or "", a)["strict"]
        else: s = r["strict"]
        out[r["constraint"]].append(int(s))
    return out

MODELS = [  # name, CoT no-elicitation run, CoT up_cp run, output run (path, regrade locally?)
    ("gpt-oss-120b (Tinker)", "tinker-runs/gptoss120b-t1", "tinker-runs/gptoss120b-upcp-t1", ("more/expl/base", False)),
    ("Qwen3.8-27B (Tinker)", "tinker-runs/qwen38-run1", "tinker-runs/qwen38-upcp", ("xfull/expl/qwen-base-tk", True)),
    ("DeepSeek V4.1 Flash 1017 (OpenRouter, Parasail)", "openrouter-runs/ds1017-up", "openrouter-runs/ds1017-upcp", ("xfull/expl/ds-base-or", True)),
]
ARMS = [("CoT, constraint in prompt", "#2a78d6"), ("CoT, + constraint prefill (up_cp)", "#1baf7a"), ("Output, thinking off", "#eb6834")]
INK, MUTED, GRID = "#2a2a28", "#6b6a65", "#e6e5e0"
fig, axes = plt.subplots(len(MODELS), 1, figsize=(13, 3.6 * len(MODELS)), sharey=True)
summary = []
for ax, (name, up, upcp, (op, rg)) in zip(axes, MODELS):
    data = [cot(up), cot(upcp), outctl(op, rg)]
    x = np.arange(len(T) + 1); w = 0.26
    for i, ((lab, col), d) in enumerate(zip(ARMS, data)):
        ks = [sum(d[c]) for c in T] + [sum(sum(d[c]) for c in T)]
        ns = [len(d[c]) for c in T] + [sum(len(d[c]) for c in T)]
        p = np.array([k / n for k, n in zip(ks, ns)]) * 100
        lo, hi = zip(*[wilson(k, n) for k, n in zip(ks, ns)])
        err = np.clip([p - np.array(lo) * 100, np.array(hi) * 100 - p], 0, None)
        pos = x + (i - 1) * (w + 0.02)
        ax.bar(pos, p, w, color=col, label=lab, zorder=3)
        ax.errorbar(pos, p, yerr=err, fmt="none", ecolor=MUTED, elinewidth=1, capsize=0, zorder=4)
        ax.text(pos[-1], hi[-1] * 100 + 2, f"{p[-1]:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
        summary.append((name, lab, ks, ns))
    ax.axvline(len(T) - 0.5, color=GRID, lw=1, zorder=1)
    ax.set_xticks(x, [NICE[c] for c in T] + ["pooled\n(8 types)"], fontsize=9, color=INK)
    ax.set_title(name, loc="left", fontsize=11, color=INK)
    ax.set_ylim(0, 105); ax.set_ylabel("strict pass %", color=MUTED)
    ax.grid(axis="y", color=GRID, zorder=0); ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values(): s.set_visible(False)
axes[0].legend(loc="upper right", frameon=False, fontsize=9, ncol=3, bbox_to_anchor=(1, 1.22))
fig.suptitle("CoT control with and without elicitation vs no-CoT output control (Astra 108 items, 8 rule-graded constraints)", x=0.01, ha="left", fontsize=12, color=INK)
fig.text(0.01, 0.005, "CoT: one sample per item, n=12 per constraint. Output: 'State your answer, and then explain it.' + constraint on the response,\n"
         "Astra items, gpt-oss 5 samples (n=60), Qwen/DeepSeek 2 samples (n=24). Bars: 95% Wilson. CoT strict counts meta-discussion as allowed.",
         fontsize=8, color=MUTED, ha="left")
fig.tight_layout(rect=(0, 0.035, 1, 0.97)); fig.savefig(sys.argv[1], dpi=150, facecolor="white")
for name, lab, ks, ns in summary:
    print(f"{name:48s} {lab:36s} pooled {sum(ks[:-1])}/{sum(ns[:-1])} | " + " ".join(f"{k}/{n}" for k, n in zip(ks[:-1], ns[:-1])))
