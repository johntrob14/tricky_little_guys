"""Partial (continuous) compliance, CoT vs no-reasoning output, on our 108 Astra items (rl/astra_continuous.score: share of letters / word pairs /
sentences / keywords meeting the rule; repeat_sentences 0.5 start + 0.5 end). Same rows as cot_vs_out_108.py. Ungated on both sides (no attempt judge).
Constraint-balanced mean over the 8 types (ignore_question excluded), and over the 6 non-suppression types (suppression is near-free in short text).
Bootstrap 95% CIs over items within type (2000 draws, seed 0).   (cd results/outctl && uv run python orout/partial_108.py) -> orout/partial_108.{json,png}"""
import json, os, sys, ast, random, collections
O = os.path.dirname(os.path.abspath(__file__)); OC = os.path.dirname(O); sys.path.insert(0, OC); sys.path.insert(0, os.path.join(OC, "../../rl"))
import astra_more as AM, astra_continuous as AC
GA = {}
for it in AM.items():
    a = ast.literal_eval(it["grader_args"]) if isinstance(it["grader_args"], str) else dict(it["grader_args"])
    for k in ("synonyms", "keywords"):
        if isinstance(a.get(k), str): a[k] = ast.literal_eval(a[k])
    GA[it["key"]] = a
st = lambda g: str(g.get("strict")) in ("1", "True")
pc = lambda g: AC.score(dict(constraint=g["constraint"], reasoning=g.get("body") if g.get("body") is not None else g.get("out") or "", grader_args=GA[g["key"]]), st(g))
SUP = ("word_suppression", "multiple_word_suppression")
def bal(G, types=None, rng=None):
    by = collections.defaultdict(list)
    for g in G:
        if types is None or g["constraint"] in types: by[g["constraint"]].append(g["_p"])
    if rng: by = {c: [rng.choice(v) for _ in v] for c, v in by.items()}
    return 100 * sum(sum(v) / len(v) for v in by.values()) / len(by)
def ci(G, types=None):
    rng = random.Random(0); xs = sorted(bal(G, types, rng) for _ in range(2000)); return [round(xs[50], 1), round(xs[1949], 1)]
def load(p): return [g for g in json.load(open(p)) if g["constraint"] != "ignore_question"]
def clean(slug):
    R = {(r["key"], r["sample_idx"]): r for r in map(json.loads, open(f"{O}/{slug}_nt/gens.jsonl"))}
    return [g for g in load(f"{O}/{slug}_nt/graded.json") if (R[(g["key"], g["sample_idx"])].get("reasoning_tokens") or 0) <= 1 and R[(g["key"], g["sample_idx"])].get("stop") != "content_filter"]
M = {"z-ai_glm-5.2": "GLM-5.2", "moonshotai_kimi-k3": "Kimi-K3", "qwen_qwen3-32b": "Qwen3-32B", "deepseek_deepseek-v4-flash": "DS-V4-Flash",
     "qwen_qwen3.5-397b-a17b": "Qwen3.5-397B", "qwen_qwen3.8-27b": "Qwen3.8-27B"}
pairs = [(n, load(f"{O}/{s}_cot/graded.json"), clean(s)) for s, n in M.items()] + [("gpt-oss-120b", load(f"{OC}/more/cot/base/graded.json"), load(f"{OC}/more/expl/base/graded.json"))]
NS = None; rows = []
for name, C, X in pairs:
    for g in C + X: g["_p"] = pc(g)
    ns = sorted({g["constraint"] for g in C} - set(SUP))
    per = {c: [round(100 * sum(g["_p"] for g in C if g["constraint"] == c) / max(1, sum(g["constraint"] == c for g in C)), 1),
               round(100 * sum(g["_p"] for g in X if g["constraint"] == c) / max(1, sum(g["constraint"] == c for g in X)), 1)] for c in sorted({g["constraint"] for g in C})}
    rows.append(dict(model=name, cot=round(bal(C), 1), cot_ci=ci(C), out=round(bal(X), 1), out_ci=ci(X), cot6=round(bal(C, ns), 1), cot6_ci=ci(C, ns), out6=round(bal(X, ns), 1), out6_ci=ci(X, ns),
                     cot_strict=round(100 * sum(map(st, C)) / len(C), 1), cot_words=sorted(len((g.get("body") or g.get("out") or "").split()) for g in C)[len(C) // 2],
                     out_words=sorted(len((g.get("body") or g.get("out") or "").split()) for g in X)[len(X) // 2], per_type_cot_out=per))
rows.sort(key=lambda r: -r["cot"])
print("| model | CoT partial (8 types) | output partial (8) | CoT partial (6, no suppression) | output partial (6) | CoT strict | CoT words | output words |")
for r in rows: print(f"| {r['model']} | {r['cot']} {r['cot_ci']} | {r['out']} {r['out_ci']} | {r['cot6']} {r['cot6_ci']} | {r['out6']} {r['out6_ci']} | {r['cot_strict']} | {r['cot_words']} | {r['out_words']} |")
print("\nper type, CoT / output partial:")
for r in rows: print(r["model"], {c[:14]: v for c, v in r["per_type_cot_out"].items()})
json.dump(rows, open(f"{O}/partial_108.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, (kc, ko, tt) in zip(axs, [("cot", "out", "All 8 constraint types"), ("cot6", "out6", "6 types, word suppression excluded")]):
    x = np.arange(len(rows)); w = 0.38
    for k, (key, lab, col) in enumerate([(kc, "CoT partial compliance (reasoning on)", "#d7a7c4"), (ko, "Output partial compliance (reasoning off)", "#5b9bd5")]):
        v = [r[key] for r in rows]; e = np.array([[r[key] - r[key + "_ci"][0], r[key + "_ci"][1] - r[key]] for r in rows]).T
        ax.bar(x + (k - 0.5) * w, v, w, yerr=e, capsize=3, color=col, edgecolor="#555", linewidth=0.6, label=lab, error_kw=dict(lw=0.8))
        for xi, vi, hi in zip(x + (k - 0.5) * w, v, e[1]): ax.text(xi, vi + hi + 1.2, f"{vi:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, [r["model"] for r in rows], rotation=20, ha="right"); ax.set_ylim(0, 120); ax.set_title(tt, fontsize=10); ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("Partial compliance (%)"); axs[0].legend(frameon=False, loc="upper left", ncol=2, fontsize=8)
fig.suptitle("Partial compliance on the same 108 Astra items: constraint in the reasoning vs in the answer", fontsize=11)
fig.text(0.01, 0.005, "Share of letters / word pairs / sentences / keywords meeting the rule (repeat string: 0.5 start + 0.5 end), averaged over constraint types; 95% bootstrap CIs. Not gated by the answered-the-question judge.", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(f"{O}/partial_108.png", dpi=160)

# Floor: the rule for type c scored on the same model's texts written under OTHER constraint types (no effort toward c), with grader args cycled
# from type-c items. Only the 6 non-suppression types (suppression needs the same question without the constraint). Effort above floor:
# (score - floor) / (100 - floor), constraint-balanced.
FL = []
for name, C, X in pairs:
    o = dict(model=name)
    for arm, G in (("cot", C), ("out", X)):
        ns = sorted({g["constraint"] for g in G} - set(SUP)); fl, sc = {}, {}
        for c in ns:
            args = [GA[g["key"]] for g in G if g["constraint"] == c]
            CASE = ("lowercase_thinking", "uppercase_thinking", "alternating_case")  # case floors from non-case texts only (texts obeying another case rule skew them)
            others = [g for g in G if g["constraint"] != c and not (c in CASE and g["constraint"] in CASE)]
            fl[c] = 100 * sum(AC.score(dict(constraint=c, reasoning=g.get("body") if g.get("body") is not None else g.get("out") or "", grader_args=args[i % len(args)]), None) for i, g in enumerate(others)) / len(others)
            sc[c] = 100 * sum(g["_p"] for g in G if g["constraint"] == c) / sum(g["constraint"] == c for g in G)
        o[arm + "_floor"] = {c[:12]: round(v, 1) for c, v in fl.items()}
        o[arm + "_above"] = round(100 * sum((sc[c] - fl[c]) / (100 - fl[c]) for c in ns) / len(ns), 1)
    FL.append(o)
print("\n| model | CoT: share of the gap above floor closed | output: same | CoT floors | output floors |")
for o in FL: print(f"| {o['model']} | {o['cot_above']} | {o['out_above']} | {o['cot_floor']} | {o['out_floor']} |")
json.dump(dict(rows=rows, floor=FL), open(f"{O}/partial_108.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(10, 4.4)); FL.sort(key=lambda o: -o["cot_above"]); x = np.arange(len(FL)); w = 0.38
for k, (key, lab, col) in enumerate([("cot_above", "CoT (reasoning on)", "#d7a7c4"), ("out_above", "Output (reasoning off)", "#5b9bd5")]):
    v = [o[key] for o in FL]; ax.bar(x + (k - 0.5) * w, v, w, color=col, edgecolor="#555", linewidth=0.6, label=lab)
    for xi, vi in zip(x + (k - 0.5) * w, v): ax.text(xi, vi + 1.2, f"{vi:.0f}", ha="center", fontsize=8)
ax.set_xticks(x, [o["model"] for o in FL]); ax.set_ylim(0, 120); ax.set_ylabel("Share of the gap above the no-effort floor closed (%)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=8)
ax.set_title("Partial compliance above each model's no-effort floor, 6 non-suppression types, same 108 Astra items", fontsize=10)
fig.text(0.01, 0.005, "Floor = the rule scored on the same model's texts written under other constraints (case rules: non-case texts only). (score - floor) / (100 - floor), averaged over types.", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{O}/partial_108_floor.png", dpi=160)
