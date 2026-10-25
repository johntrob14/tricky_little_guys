"""ds41_sft1 results: the SFT checkpoint against its starting point (rwsft1 ep2) and base DeepSeek on the arena, from the per-episode rows that
results/monitor_visible/summarize.py writes for each monitor pass (ds41_sft1/results/pass*.json; report.sh makes them). Run with the system python (scipy, matplotlib):

    python ds41_sft1/analyze.py

Episodes "doing the side task" are those the leak detector fired on or the scorer confirmed (controls and the no-secret arm count differently, below). Tests are Mann-Whitney on
per-episode values and Fisher on counts, unpaired. Pass 1-3 are the monitor and judge the cleaning loop used; pass 4 is a different prompt and judge. Writes results/stats.txt and
mats12_sprint/figures/ds41_sft1_arena.png.
"""
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, mannwhitneyu

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
OUT_FIG = HERE.parents[2] / "figures" / "ds41_sft1_arena.png"
MODELS = {"sft": ("SFT ep2 (ds41sft1)", "#199e70"), "rw": ("rwsft1 ep2 (start)", "#c98500"), "base": ("base DeepSeek", "#3987e5")}
from analyze_cells import CELLS  # noqa: E402
PASSES = [("p1", "pass1_reasoning_luna.json", "reasoning, CoT only, controlarena_cot_only_context, gpt-6-luna low (cleaning monitor)"),
          ("p2", "pass2_visible_luna.json", "visible text, CoT only, controlarena_cot_only_context, gpt-6-luna low (cleaning monitor)"),
          ("p3", "pass3_actioncot.json", "reasoning + visible text + tool calls, action+cot, controlarena_cot, gpt-6-luna low"),
          ("p4", "pass4_cot_only_oss20b.json", "reasoning + visible text, CoT only, controlarena_cot_only, gpt-oss-20b (held-out monitor)")]
STOCK_KEY_NAMES = {"NEXTAUTH_SECRET", "TIKAPI_KEY"}   # the stock .env.local already holds these, so the no-secret arm's prompt is true for them
SURFACE, INK, INK2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a"
TRAIN_TASKS = {s["task"] for s in json.loads((HERE / "data/manifest.json").read_text())["per_transcript"]}

rows = {}
for key, fname, _ in PASSES:
    for r in json.loads((HERE / "results" / fname).read_text()):
        rows.setdefault((key, r["cell"]), []).append(r)


def task_of(r):
    return r["sample"].split("#")[0]


def doing(rs):
    return [r for r in rs if r["attempted"] or r["leaked"]]


def episodes(key, model, arms=("easy", "medium"), held_out=None, only_doing=True):
    out = []
    for arm in arms:
        cell = CELLS[model].get(arm)
        rs = rows.get((key, cell), [])
        rs = [r for r in rs if held_out is None or (task_of(r) not in TRAIN_TASKS) == held_out]
        out += doing(rs) if only_doing else rs
    return out


def values(rs, kind):
    """Per-episode values of a row field; 'total' is the sum of the turn scores (mean x scored turns), which an episode's extra benign turns do not dilute."""
    if kind == "total":
        return [r["mean"] * r["scored_turns"] for r in rs if r["mean"] is not None]
    return [r[kind] for r in rs if r[kind] is not None]


def fmt(xs):
    return f"{statistics.fmean(xs):.2f} (n={len(xs)})" if xs else "-"


def mwu(a, b):
    return f"p={mannwhitneyu(a, b).pvalue:.3g}" if len(a) > 1 and len(b) > 1 and len(set(a + b)) > 1 else "p=n/a"


def fisher(k1, n1, k2, n2):
    return f"Fisher p={fisher_exact([[k1, n1 - k1], [k2, n2 - k2]]).pvalue:.3g}"


lines = []
say = lambda s="": (lines.append(s), print(s))
say(f"tiktok tasks in the training set: {sorted(TRAIN_TASKS)}; the other 4 of 8 are held out. Eval episodes per cell: 16 (8 tasks x 2), seed base 100.")
say()
say("== Side-task attempts (leak detector fired on a tool call) and scorer leaks, of the episodes in each arm (all tasks / held-out tasks only)")
for model, (label, _) in MODELS.items():
    for arm, cell in CELLS[model].items():
        rs = rows.get(("p1", cell), [])
        if not rs:
            continue
        ho = [r for r in rs if task_of(r) not in TRAIN_TASKS]
        say(f"  {label:22s} {arm:8s} attempted {sum(r['attempted'] for r in rs)}/{len(rs)} leaked {sum(r['leaked'] for r in rs)}/{len(rs)} | held-out tasks: attempted {sum(r['attempted'] for r in ho)}/{len(ho)} leaked {sum(r['leaked'] for r in ho)}/{len(ho)}")
say()
say("== Assistant turns per episode, episodes doing the side task (easy + medium pooled), and the share of turns with empty reasoning is in tally.out")
for model in MODELS:
    rs = episodes("p1", model)
    say(f"  {MODELS[model][0]:22s} {fmt([r['turns'] for r in rs])}; held-out tasks {fmt([r['turns'] for r in episodes('p1', model, held_out=True)])}")
for model in ("sft", "rw"):
    for arm in ("control", "absent"):
        rs = rows.get(("p1", CELLS[model][arm]), [])
        if rs:
            say(f"  {MODELS[model][0]:22s} {arm:8s} {fmt([r['turns'] for r in rs])} turns per episode (all episodes)")
say()
say("== Attack cells pooled over easy and medium: SFT against rwsft1 and against base (Fisher on counts)")
for other in ("rw", "base"):
    for what in ("attempted", "leaked"):
        for held_out in (None, True):
            a = [r for arm in ("easy", "medium") for r in rows.get(("p1", CELLS["sft"][arm]), []) if held_out is None or (task_of(r) not in TRAIN_TASKS)]
            b = [r for arm in ("easy", "medium") for r in rows.get(("p1", CELLS[other].get(arm)), []) if held_out is None or (task_of(r) not in TRAIN_TASKS)]
            if a and b:
                say(f"  {what:9s} SFT {sum(r[what] for r in a)}/{len(a)} vs {other} {sum(r[what] for r in b)}/{len(b)} ({'held-out tasks' if held_out else 'all tasks'}): {fisher(sum(r[what] for r in a), len(a), sum(r[what] for r in b), len(b))}")
say()
say("== No-secret arm (nothing seeded, the secret's file deleted): attempts only, since the scorer cannot confirm a leak; the stock .env.local holds NEXTAUTH_SECRET and TIKAPI_KEY, so the prompt is true for episodes that draw those names")
names = json.loads((HERE / "results/secret_names.json").read_text())
for model in ("sft", "rw"):
    cell = CELLS[model]["absent"]
    rs = rows.get(("p1", cell), [])
    if not rs:
        continue
    stock = [names[cell][f"{r['sample']}|{r['epoch']}"]["secret_name"] in STOCK_KEY_NAMES for r in rs]
    say(f"  {MODELS[model][0]:22s} attempted {sum(r['attempted'] for r in rs)}/{len(rs)} all; {sum(r['attempted'] for r, st in zip(rs, stock) if not st)}/{stock.count(False)} on the episodes without a stock key name; {sum(r['attempted'] for r, st in zip(rs, stock) if st)}/{stock.count(True)} with one")
a_ = [r for r in rows.get(("p1", CELLS["sft"]["absent"]), [])]
b_ = [r for r in rows.get(("p1", CELLS["rw"]["absent"]), [])]
if a_ and b_:
    say(f"  SFT vs rwsft1, all 16: {fisher(sum(r['attempted'] for r in a_), len(a_), sum(r['attempted'] for r in b_), len(b_))}")
say()
for key, _, desc in PASSES:
    say(f"== Pass {key[1]}: {desc}")
    say("   per-episode max and mean over scored turns, mean over episodes doing the side task; first-attempt-turn score; controls over all episodes")
    for held_out, tag in ((None, "all tasks"), (True, "held-out tasks")):
        for kind in ("max", "mean", "total", "first_attempt_score"):
            parts = []
            vals = {}
            for model in MODELS:
                if not rows.get((key, CELLS[model]["easy"])):
                    continue
                vals[model] = values(episodes(key, model, held_out=held_out), kind)
                parts.append(f"{MODELS[model][0]} {fmt(vals[model])}")
            tests = [f"SFT vs {o}: {mwu(vals['sft'], vals[o])}" for o in ("rw", "base") if o in vals and "sft" in vals]
            say(f"   [{tag}] {kind:20s} " + " | ".join(parts) + "   " + "; ".join(tests))
    for model in ("sft", "rw"):
        c = rows.get((key, CELLS[model]["control"]), [])
        if c:
            say(f"   control (no side task) per-episode max: {MODELS[model][0]} {fmt(values(c, 'max'))}")
    say("   unconditioned per-episode max over all attack episodes: " + " | ".join(f"{MODELS[m][0]} {fmt(values(episodes(key, m, only_doing=False), 'max'))}" for m in MODELS if rows.get((key, CELLS[m]['easy']))))
    say()
(HERE / "results/stats.txt").write_text("\n".join(lines) + "\n")

# ---- figure: rates, then per-episode max, first-attempt-turn score, per-turn mean and summed score for each pass, easy + medium pooled
fig, axes = plt.subplots(5, 1, figsize=(14, 19.5), facecolor=SURFACE, gridspec_kw={"height_ratios": [0.8, 1, 1, 1, 1]})


def style(ax, ylabel, ylim):
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_ylim(0, ylim)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color("#383835")
    ax.tick_params(colors=MUTED)


ax = axes[0]
style(ax, "episodes (of 16 per cell)", 17.5)
arms = [("easy", "attack easy"), ("medium", "attack medium"), ("control", "control (no side task)"), ("absent", "no-secret arm")]
for i, (arm, title) in enumerate(arms):
    for k, model in enumerate(MODELS):
        rs = rows.get(("p1", CELLS[model].get(arm)), [])
        if not rs:
            continue
        x = i + (k - 1) * 0.28
        att, lk = sum(r["attempted"] for r in rs), sum(r["leaked"] for r in rs)
        ax.bar(x - 0.065, att, 0.12, color=MODELS[model][1])
        ax.bar(x + 0.065, lk, 0.12, color=SURFACE, edgecolor=MODELS[model][1], hatch="////", linewidth=1.1)
        ax.text(x - 0.065, att + 0.2, str(att), ha="center", color=INK2, fontsize=7)
        ax.text(x + 0.065, lk + 0.2, str(lk), ha="center", color=INK2, fontsize=7)
ax.set_xticks(range(len(arms)), [t for _, t in arms], color=INK2)
ax.set_title("side task attempted (solid: leak detector fired) and completed (hatched: scorer)", color=INK, fontsize=10.5, loc="left")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c in MODELS.values()], [l for l, _ in MODELS.values()], loc="upper center", bbox_to_anchor=(0.6, 1.0), frameon=False, labelcolor=INK2, fontsize=8.5)

turns = {m: statistics.fmean(r["turns"] for r in episodes("p1", m)) for m in MODELS}
PANELS = [("max", "per-episode max suspicion (0-10)", 12.5, "per-episode max suspicion (0-10)"),
          ("first_attempt_score", "score on the first side-task attempt turn (0-10)", 12.5, "score on the first side-task attempt turn (0-10)"),
          ("mean", "per-episode mean over scored turns (0-10)", 12.5,
           "per-episode mean over scored turns (0-10); diluted by episode length: assistant turns per episode " + ", ".join(f"{MODELS[m][0].split(' (')[0]} {turns[m]:.0f}" for m in MODELS)),
          ("total", "per-episode sum of turn scores", 130, "per-episode sum of the turn scores (not diluted by episode length)")]
for ax, (kind, ylabel, ylim, title) in zip(axes[1:], PANELS):
    style(ax, ylabel, ylim)
    for i, (key, _, desc) in enumerate(PASSES):
        for k, model in enumerate(MODELS):
            if not rows.get((key, CELLS[model]["easy"])):
                continue
            vs = values(episodes(key, model), kind)
            if not vs:
                continue
            se = statistics.stdev(vs) / len(vs) ** 0.5 if len(vs) > 1 else 0
            x = i + (k - 1) * 0.28
            ax.bar(x, statistics.fmean(vs), 0.25, color=MODELS[model][1])
            ax.errorbar(x, statistics.fmean(vs), yerr=1.96 * se, fmt="none", ecolor=INK2, elinewidth=1.1)
            ax.text(x, statistics.fmean(vs) + 1.96 * se + ylim * 0.012, f"{statistics.fmean(vs):.1f}\nn={len(vs)}", ha="center", color=MUTED, fontsize=7)
    ax.set_xticks(range(len(PASSES)), ["1. reasoning\nCoT only, luna low*", "2. visible text\nCoT only, luna low*", "3. reasoning + visible + tool calls\naction+cot, luna low*", "4. reasoning + visible\nCoT only, gpt-oss-20b (held out)"], color=INK2, fontsize=8.5)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
fig.suptitle("SFT of the DeepSeek CoT-control organism on cleaned arena attack traces vs its starting point and base\n"
             "tiktok easy + medium, arena_covert_coach_first, thinking on (effort 25), 16 episodes per cell, seed base 100; monitor scores over episodes doing the side task (detector fired or scorer leak)\n"
             "main task only for every monitor; * the monitor and judge the training traces were cleaned against; error bars +-1.96 SE over episodes", color=INK, fontsize=10.5, x=0.01, ha="left")
fig.text(0.01, 0.004, "Half of each cell's tasks are in the SFT training set; held-out-task numbers are in results/stats.txt. Base DeepSeek: Tinker, same renderer and effort. arena/ds41_sft1", color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.012, 1, 0.945))
fig.savefig(OUT_FIG, dpi=140, facecolor=SURFACE)
print(OUT_FIG)
