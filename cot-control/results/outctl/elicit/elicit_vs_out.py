# CoT control (as written / dev-chosen elicitation / sysexall) vs output control with thinking off (plain / shotstack / sysexall),
# 11 base models, 72 held-out Astra items, 8 rule-graded types, constraint-balanced. All output arms from elicit/runs/<m>/out*_test
# (gated: strict AND attempt judge; rows that wrote thinking dropped, as outelicit.py). CoT: strict on the reasoning.
import json, os, collections, random, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
E = os.path.dirname(os.path.abspath(__file__)); R = f"{E}/runs"; OUT = sys.argv[1] if len(sys.argv) > 1 else f"{E}/figures"
C = {r["model"]: r for r in json.load(open(f"{E}/dash/compare.json"))["rows"]}
RUN = {"gpt-oss-120b": "gptoss", "Qwen3.5-397B": "qwen35", "GLM-5.3": "glm53", "DeepSeek V4.1 Flash 1017": "ds41", "Qwen3.8-27B": "qwen38",
       "Inkling-Small": "inksm", "Qwen3.5-4B": "qwen35_4b", "Qwen3.5-9B": "qwen35_9b", "DeepSeek V4 Flash": "or_dsv4f", "Qwen3-32B": "or_qwen3_32b", "GLM-5.2": "or_glm52", "Nemotron-3 Super": "nemosuper"}
ELICIT = {m: C[m]["cot_elicit"]["src"].split("/")[-1] if m in C else "shotstack_test" for m in RUN}; ELICIT.update({"DeepSeek V4 Flash": "shotstack_pinned_test", "GLM-5.2": "shotstack_pinned_test"})
OUTSHOT = {m: "outshotstack_test" for m in RUN}; OUTSHOT["Inkling-Small"] = "outshotstack_forced_test"
T = ["lowercase_thinking", "uppercase_thinking", "end_of_sentence", "meow_between_words", "word_suppression", "multiple_word_suppression", "repeat_sentences", "alternating_case"]
NICE = ["lower", "upper", "end of\nsent.", "meow", "word\nsuppr.", "multi\nsuppr.", "repeat\nsent.", "alt.\ncase"]
st = lambda g: str(g.get("strict")) in ("1", "True")
def load(m, d, out):
    p = f"{R}/{RUN[m]}/{d}/graded.json"
    if not os.path.exists(p): return None
    G = [g for g in json.load(open(p)) if g["constraint"] in T]
    if out: G = [g for g in G if not (g.get("cot") or "").strip()]
    for g in G: g["ok"] = int(st(g) and (bool(g.get("attempt")) if out else True))
    return G
rng = random.Random(0)
def boot(G, B=5000):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for g in G: by[g["constraint"]][g["key"].rsplit("|", 1)[0]].append(g["ok"])
    by = {c: list(v.values()) for c, v in by.items()}
    f = lambda d: 100 * sum(sum(map(sum, it)) / sum(map(len, it)) for it in d.values()) / len(d)
    bs = sorted(f({c: rng.choices(v, k=len(v)) for c, v in by.items()}) for _ in range(B))
    return dict(v=f(by), lo=bs[int(.025 * B)], hi=bs[int(.975 * B)], n=len(G))
ARMS = [("CoT, as written", "#2a78d6", lambda m: ("compare", C[m]["cot_base"]) if m in C else load(m, "base_test", False)),
        ("CoT, dev-chosen elicitation", "#1baf7a", lambda m: load(m, ELICIT[m], False)),
        ("CoT, sysexall", "#008300", lambda m: load(m, "sysexall_test", False)),
        ("Output, plain", "#eb6834", lambda m: load(m, "outnocot_test", True)),
        ("Output, shotstack", "#e87ba4", lambda m: load(m, OUTSHOT[m], True)),
        ("Output, sysexall", "#eda100", lambda m: load(m, "outsysexall_test", True))]
RES = {}; ROWS = {}
for m in RUN:
    for lab, _, f in ARMS:
        G = f(m)
        if G is None: RES[(m, lab)] = None
        elif isinstance(G, tuple): RES[(m, lab)] = G[1]
        else: RES[(m, lab)] = boot(G); ROWS[(m, lab)] = G
OUTS = ("Output, plain", "Output, shotstack", "Output, sysexall")
MODELS = sorted(RUN, key=lambda m: max(RES[(m, l)]["v"] for l in OUTS if RES[(m, l)]))  # by best output-control score
INK, MUTED, GRID = "#2a2a28", "#6b6a65", "#e6e5e0"
def style(ax):
    ax.grid(axis="y", color=GRID, zorder=0); ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values(): s.set_visible(False)
fig, ax = plt.subplots(figsize=(20, 6.4)); x = np.arange(len(MODELS)); w = 0.135
for i, (lab, col, _) in enumerate(ARMS):
    pos = x + (i - 2.5) * (w + 0.008)
    v = np.array([RES[(m, lab)]["v"] if RES[(m, lab)] else np.nan for m in MODELS])
    lo = np.array([RES[(m, lab)]["lo"] if RES[(m, lab)] else np.nan for m in MODELS]); hi = np.array([RES[(m, lab)]["hi"] if RES[(m, lab)] else np.nan for m in MODELS])
    ax.bar(pos, v, w, color=col, label=lab, zorder=3, edgecolor="white", linewidth=0.6 if i < 3 else 0, hatch=None if i < 3 else "//")
    ax.errorbar(pos, v, yerr=np.nan_to_num(np.clip([v - lo, hi - v], 0, None)), fmt="none", ecolor=MUTED, elinewidth=0.8, zorder=4)
ax.set_xticks(x, [m.replace(" Flash", "\nFlash") for m in MODELS], fontsize=9, color=INK); ax.set_ylim(0, 100)
ax.set_ylabel("pass %, mean over 8 types", color=MUTED); style(ax); ax.legend(frameon=False, fontsize=9, ncol=6, loc="upper left")
ax.set_title("CoT control vs no-CoT output control, each with no elicitation, shotstack / dev-chosen elicitation, and sysexall (12 base models, ordered by best output score)", loc="left", fontsize=12, color=INK)
fig.text(0.01, 0.01, "72 held-out Astra items, 8 rule-graded types, constraint-balanced, bars 95% bootstrap over items. CoT: strict on the reasoning. Output (hatched): strict AND attempt judge "
         "on the response, thinking off, rows that wrote thinking dropped; all output arms from elicit.py runs.\nDev-chosen CoT elicitation is shotstack (shotlow for Qwen3.8). OpenRouter raw-completion runs pinned to a provider that honors raw prompts "
         "(DeepSeek V4 Flash, GLM-5.2: Parasail; Qwen3-32B: DeepInfra). Inkling output runs use a forced empty thinking message.\nCoT as written: compare.json sources (Nemotron-3 Super: its elicit base_test run; no dev sweep, so its elicitation is shotstack).", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f"{OUT}/elicit_vs_output_pooled_v2.png", dpi=140, facecolor="white")
nr = -(-(len(MODELS) + 1) // 3); fig, axs = plt.subplots(nr, 3, figsize=(20, 4 * nr), sharey=True); x = np.arange(len(T)); w = 0.135
for ax, m in zip(axs.flat, MODELS[::-1]):
    for i, (lab, col, _) in enumerate(ARMS):
        G = ROWS.get((m, lab))
        if G is None: continue
        ax.bar(x + (i - 2.5) * (w + 0.004), [100 * np.mean([g["ok"] for g in G if g["constraint"] == c] or [np.nan]) for c in T], w, color=col, label=lab, zorder=3, hatch=None if i < 3 else "//", edgecolor="white", linewidth=0)
    ax.set_xticks(x, NICE, fontsize=8, color=INK); ax.set_ylim(0, 100); style(ax)
    vals = " / ".join("-" if RES[(m, l)] is None else f"{RES[(m, l)]['v']:.0f}" for l, _, _ in ARMS)
    ax.set_title(f"{m}   {vals}", loc="left", fontsize=10, color=INK)
for ax in axs.flat[len(MODELS):]: ax.axis("off")
h = [plt.Rectangle((0, 0), 1, 1, color=c, hatch=None if i < 3 else "//") for i, (_, c, _) in enumerate(ARMS)]
axs.flat[-1].legend(h, [l for l, _, _ in ARMS], frameon=False, fontsize=11, loc="center", title="title values in this order\n(per-type bars: CoT as written omitted)")
fig.suptitle("Per constraint type: CoT vs output control, plain and elicited, pass %", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(f"{OUT}/elicit_vs_output_by_type_v2.png", dpi=120, facecolor="white")
print("| model | " + " | ".join(l for l, _, _ in ARMS) + " |"); print("|---" * (len(ARMS) + 1) + "|")
for m in MODELS[::-1]:
    print(f"| {m} | " + " | ".join("-" if RES[(m, l)] is None else f"{RES[(m, l)]['v']:.1f} [{RES[(m, l)]['lo']:.0f}, {RES[(m, l)]['hi']:.0f}]" for l, _, _ in ARMS) + " |")
json.dump({f"{m}|{l}": RES[(m, l)] for m in RUN for l, _, _ in ARMS}, open(f"{OUT}/elicit_vs_output_v2.json", "w"), indent=1)
