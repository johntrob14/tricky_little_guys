"""Pooled analysis: 108 Astra items (CoT medium x5, explain-output x5) + 432 new items (CoT medium x3, CoT low x3, explain-output x3).
Gated pass = strict pass AND attempt judge YES. ignore_question excluded. Run from cot-control/results/outctl:  python3 analyze_v3.py"""
import json, random, statistics as st, collections, sys
MODELS = ["base", "grpo3", "st1-ignore2", "gsft1", "len2", "efflen1", "st1-third", "st1-ignore3"]
T = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "end_of_sentence", "lowercase_thinking", "meow_between_words", "uppercase_thinking", "alternating_case"]
EDGES = [5, 20, 50, 100, 200, 500, 10**9]; LAB = ["0-5", "6-20", "21-50", "51-100", "101-200", "201-500", "500+"]
def b(w): return next(i for i, e in enumerate(EDGES) if w <= e)
v2 = json.load(open("more/rows_v2.json"))
def load(m):
    rows = []
    for r in v2[m]:
        if r["side"] in ("cot", "expl") and r["c"] != "ignore_question":
            rows.append(dict(arm="cot_med" if r["side"] == "cot" else "expl", item=("astra",) + tuple(r["item"]), c=r["c"], w=r["w"], ok=bool(r["ok"] and r["att"]), raw=bool(r["ok"]), cor=r["cor"]))
    for side, arm in (("cot", "cot_med"), ("cot_low", "cot_low"), ("expl", "expl")):
        for g in json.load(open(f"big/{side}/{m}/graded.json")):
            if g["constraint"] == "ignore_question": continue
            rows.append(dict(arm=arm, item=("new", g["dataset"], str(g["item_idx"])), c=g["constraint"], w=g["words"], ok=bool(g["strict"] == 1 and g["attempt"]), raw=g["strict"] == 1, cor=g["correct"]))
    return rows
def matched(rows, cot_arms, cons=None):
    s = collections.defaultdict(lambda: {"cot": [], "expl": []})
    for r in rows:
        if cons and r["c"] != cons: continue
        side = "cot" if r["arm"] in cot_arms else "expl" if r["arm"] == "expl" else None
        if side: s[(r["c"], b(r["w"]))][side].append(r["ok"])
    num = den = 0
    for v in s.values():
        if v["cot"] and v["expl"]:
            w = min(len(v["cot"]), len(v["expl"])); num += w * (sum(v["expl"]) / len(v["expl"]) - sum(v["cot"]) / len(v["cot"])); den += w
    return (100 * num / den if den else None), den
def boot(rows, f, n=1000):
    keys = sorted({r["item"] + (r["c"],) for r in rows}); by = collections.defaultdict(list)
    for r in rows: by[r["item"] + (r["c"],)].append(r)
    rng = random.Random(0); v = []
    for _ in range(n):
        x = f([r for k in rng.choices(keys, k=len(keys)) for r in by[k]])
        if x is not None: v.append(x)
    v.sort(); return v[25], v[974]
D = {m: load(m) for m in MODELS}
json.dump(D, open("big/rows_v3.json", "w"))
rate = lambda rows, arm: 100 * sum(r["ok"] for r in rows if r["arm"] == arm) / max(1, sum(r["arm"] == arm for r in rows))
n = lambda rows, arm: sum(r["arm"] == arm for r in rows)
med = lambda rows, arm: st.median(r["w"] for r in rows if r["arm"] == arm)
print("Gated pass % (strict pass and the attempt judge says the text works on the question); ignore_question excluded.\n")
print("| model | CoT medium (n) | CoT low (n) | explain output (n) | median words: CoT med / CoT low / explain | explain - CoT (med+low), matched on constraint x length [95% CI] (weight) | explain - CoT medium, matched | explain - CoT low, matched |")
print("|---|---|---|---|---|---|---|---|")
for m, rows in D.items():
    g, w = matched(rows, {"cot_med", "cot_low"}); ci = boot(rows, lambda x: matched(x, {"cot_med", "cot_low"})[0])
    gm, wm = matched(rows, {"cot_med"}); gl, wl = matched(rows, {"cot_low"})
    print(f"| {m} | {rate(rows,'cot_med'):.1f} ({n(rows,'cot_med')}) | {rate(rows,'cot_low'):.1f} ({n(rows,'cot_low')}) | {rate(rows,'expl'):.1f} ({n(rows,'expl')}) | "
          f"{med(rows,'cot_med'):.0f} / {med(rows,'cot_low'):.0f} / {med(rows,'expl'):.0f} | {g:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}] ({w}) | {gm:+.1f} ({wm}) | {gl:+.1f} ({wl}) |")
print("\n## Per constraint: explain - CoT (med+low), gated, matched on length (weight)\n")
print("| constraint | " + " | ".join(MODELS) + " |"); print("|---" * (len(MODELS) + 1) + "|")
for t in T:
    cells = []
    for m in MODELS:
        g, w = matched(D[m], {"cot_med", "cot_low"}, t); cells.append(f"{g:+.0f} ({w})" if g is not None else "-")
    print(f"| {t} | " + " | ".join(cells) + " |")
print("\n## By length bin, gated passes/n: CoT (med+low) | explain\n")
print("| model | arm | " + " | ".join(LAB) + " |"); print("|---|---|" + "---|" * len(LAB))
for m, rows in D.items():
    for name, arms in (("CoT", {"cot_med", "cot_low"}), ("explain", {"expl"})):
        c = [[0, 0] for _ in LAB]
        for r in rows:
            if r["arm"] in arms: k = c[b(r["w"])]; k[0] += r["ok"]; k[1] += 1
        print(f"| {m} | {name} | " + " | ".join(f"{p}/{q}" if q else "-" for p, q in c) + " |")
