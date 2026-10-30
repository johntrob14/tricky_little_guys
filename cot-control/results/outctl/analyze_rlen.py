"""rlen1 checkpoint vs base gpt-oss: CoT control (medium, low), explained-output control, gaps, accuracy, length. Gated passes
(strict and attempt judge), ignore_question excluded. Base from big/rows_v3.json; the checkpoint from its graded files (Astra items:
more/{cot,expl} x5; new items: big/{cot,cot_low,expl} x3), the same samples per item as base. Run from results/outctl: python3 analyze_rlen.py rlen1-s10"""
import json, random, collections, statistics as st, sys
M = sys.argv[1]
T = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "end_of_sentence", "lowercase_thinking", "meow_between_words", "uppercase_thinking", "alternating_case"]
EDGES = [5, 20, 50, 100, 200, 500, 10**9]
b = lambda w: next(i for i, e in enumerate(EDGES) if w <= e)
D = json.load(open("big/rows_v3.json"))
def load(m):
    rows = []
    for src, arms in (("more", {"cot": "cot_med", "expl": "expl"}), ("big", {"cot": "cot_med", "cot_low": "cot_low", "expl": "expl"})):
        for side, arm in arms.items():
            for g in json.load(open(f"{src}/{side}/{m}/graded.json")):
                if g["constraint"] == "ignore_question": continue
                rows.append(dict(arm=arm, item=("astra" if src == "more" else "new", g["dataset"], str(g["item_idx"])), c=g["constraint"], w=g["words"],
                                 ok=g["strict"] == 1 and bool(g["attempt"]), cor=g["correct"]))
    return rows
R = {"base": D["base"], M: load(M)}
for k in ("len2", "efflen1"): R[k] = D[k]
def key(r): return tuple(r["item"]) + (r["c"],)
def matched(a, c):  # c minus a within constraint x length bin
    s = collections.defaultdict(lambda: [[], []])
    for k, rs in ((0, a), (1, c)):
        for r in rs: s[(r["c"], b(r["w"]))][k].append(r["ok"])
    num = den = 0
    for x, y in s.values():
        if x and y: w = min(len(x), len(y)); num += w * (st.mean(y) - st.mean(x)); den += w
    return 100 * num / den if den else None
def boot(a, c, f, n=1000):
    ka, kc = collections.defaultdict(list), collections.defaultdict(list)
    for r in a: ka[key(r)].append(r)
    for r in c: kc[key(r)].append(r)
    ks = sorted(set(ka) & set(kc)); rng = random.Random(0); v = []
    for _ in range(n):
        s = rng.choices(ks, k=len(ks)); x = f([r for k in s for r in ka[k]], [r for k in s for r in kc[k]])
        if x is not None: v.append(x)
    v.sort(); return v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]
rate = lambda rs: 100 * sum(r["ok"] for r in rs) / len(rs) if rs else float("nan")
arm = lambda m, a: [r for r in R[m] if r["arm"] in a]
print(f"== {M} vs base (gated pass %, ignore_question excluded)\n")
print("| arm | base | " + M + " | raw diff [95% CI] | length-matched diff [95% CI] | median words base / " + M + " |")
print("|---|---|---|---|---|---|")
for name, a in (("CoT medium", {"cot_med"}), ("CoT low", {"cot_low"}), ("CoT pooled", {"cot_med", "cot_low"}), ("explained output", {"expl"})):
    x, y = arm("base", a), arm(M, a)
    rd = boot(x, y, lambda p, q: rate(q) - rate(p)); md = boot(x, y, matched)
    print(f"| {name} | {rate(x):.1f} | {rate(y):.1f} | {rate(y)-rate(x):+.1f} [{rd[0]:+.1f}, {rd[1]:+.1f}] | {matched(x, y):+.1f} [{md[0]:+.1f}, {md[1]:+.1f}] | {st.median(r['w'] for r in x):.0f} / {st.median(r['w'] for r in y):.0f} |")
print("\nOwn output-minus-CoT gap, matched on constraint x length:")
for m in R:
    c, e = arm(m, {"cot_med", "cot_low"}), arm(m, {"expl"}); ci = boot(c, e, matched)
    print(f"  {m:10s} {matched(c, e):+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]")
print("\nPer constraint, CoT pooled pass % (base / " + M + ") and explained output (base / " + M + "):")
for t in T:
    f = lambda m, a: rate([r for r in arm(m, a) if r["c"] == t])
    print(f"  {t:26s} CoT {f('base', {'cot_med','cot_low'}):5.1f} / {f(M, {'cot_med','cot_low'}):5.1f}   output {f('base', {'expl'}):5.1f} / {f(M, {'expl'}):5.1f}")
print("\nAccuracy % (CoT medium / low / output): " + "; ".join(f"{m} " + " / ".join(f"{100*sum(r['cor'] for r in arm(m, {a}))/max(1,len(arm(m, {a}))):.1f}" for a in ('cot_med', 'cot_low', 'expl')) for m in R))
