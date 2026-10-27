"""CoT control vs no-reasoning output control on OUR 108 Astra items, per model (constraint-balanced: mean over the 8 constraint types, ignore_question
excluded; CIs = bootstrap over items within type, 2000 draws, seed 0). CoT = strict pass on raw reasoning (orout/<slug>_cot, reasoning on, item as written
with 'reasoning stage'); output = gated pass (strict AND attempt) on rows with <=1 reasoning token (orout/<slug>_nt). gpt-oss-120b: Tinker runs
more/cot/base (medium effort) and more/expl/base (CoT forced empty).   python3 orout/cot_vs_out_108.py -> orout/cot_vs_out_108.{json,png}"""
import json, os, random, collections
O = os.path.dirname(os.path.abspath(__file__)); OC = os.path.dirname(O)
ok = lambda g: str(g.get("strict")) in ("1", "True") and bool(g.get("attempt")); st = lambda g: str(g.get("strict")) in ("1", "True")
def bal(G, f, rng=None):
    by = collections.defaultdict(list)
    for g in G: by[g["constraint"]].append(f(g))
    if rng: by = {c: [rng.choice(v) for _ in v] for c, v in by.items()}
    return 100 * sum(sum(v) / len(v) for v in by.values()) / len(by), {c: len(v) for c, v in by.items()}
def ci(G, f):
    rng = random.Random(0); xs = sorted(bal(G, f, rng)[0] for _ in range(2000)); return [round(xs[50], 1), round(xs[1949], 1)]
def load(p): return [g for g in json.load(open(p)) if g["constraint"] != "ignore_question"]
def clean(slug):
    R = {(r["key"], r["sample_idx"]): r for r in map(json.loads, open(f"{O}/{slug}_nt/gens.jsonl"))}
    return [g for g in load(f"{O}/{slug}_nt/graded.json") if (R[(g["key"], g["sample_idx"])].get("reasoning_tokens") or 0) <= 1 and R[(g["key"], g["sample_idx"])].get("stop") != "content_filter"]
M = {"moonshotai_kimi-k3": "Kimi-K3", "z-ai_glm-5.2": "GLM-5.2", "deepseek_deepseek-v4-flash": "DS-V4-Flash", "qwen_qwen3.5-397b-a17b": "Qwen3.5-397B",
     "qwen_qwen3-32b": "Qwen3-32B", "qwen_qwen3.8-27b": "Qwen3.8-27B"}
rows = []
for slug, name in M.items():
    C, X = load(f"{O}/{slug}_cot/graded.json"), clean(slug)
    rows.append(dict(model=name, cot=round(bal(C, st)[0], 1), cot_ci=ci(C, st), cot_n=len(C), cot_per_type=bal(C, st)[1], out=round(bal(X, ok)[0], 1), out_ci=ci(X, ok), out_n=len(X),
                     out_strict=round(bal(X, st)[0], 1)))
C, X = load(f"{OC}/more/cot/base/graded.json"), load(f"{OC}/more/expl/base/graded.json")
rows.append(dict(model="gpt-oss-120b", cot=round(bal(C, st)[0], 1), cot_ci=ci(C, st), cot_n=len(C), cot_per_type=bal(C, st)[1], out=round(bal(X, ok)[0], 1), out_ci=ci(X, ok), out_n=len(X), out_strict=round(bal(X, st)[0], 1)))
rows.sort(key=lambda r: -r["cot"])
for r in rows: print(f"{r['model']:18s} CoT {r['cot']:5.1f} {r['cot_ci']} (n {r['cot_n']}, min/type {min(r['cot_per_type'].values())})   output {r['out']:5.1f} {r['out_ci']} (n {r['out_n']}; strict-only {r['out_strict']})")
json.dump(rows, open(f"{O}/cot_vs_out_108.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
fig, ax = plt.subplots(figsize=(10, 4.6)); x = np.arange(len(rows)); w = 0.38
for k, (key, lab, col) in enumerate([("cot", "CoT controllability (reasoning on)", "#d7a7c4"), ("out", "Output controllability (reasoning off)", "#5b9bd5")]):
    v = [r[key] for r in rows]; e = np.array([[r[key] - r[key + "_ci"][0], r[key + "_ci"][1] - r[key]] for r in rows]).T
    b = ax.bar(x + (k - 0.5) * w, v, w, yerr=e, capsize=3, color=col, edgecolor="#555", linewidth=0.6, label=lab, error_kw=dict(lw=0.8))
    for xi, vi, hi in zip(x + (k - 0.5) * w, v, e[1]): ax.text(xi, vi + hi + 1.2, f"{vi:.1f}", ha="center", fontsize=8)
ax.set_xticks(x, [r["model"] for r in rows]); ax.set_ylabel("Controllability score (%)"); ax.set_ylim(0, 100); ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Same 108 Astra items: constraint in the reasoning vs in the answer (8 constraint types, ignore_question excluded)", fontsize=10)
fig.text(0.01, 0.005, "CoT: strict pass on raw reasoning. Output: strict pass + answered-the-question judge, rows with 0 reasoning tokens. Means over constraint types; 95% bootstrap CIs.", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{O}/cot_vs_out_108.png", dpi=170)
