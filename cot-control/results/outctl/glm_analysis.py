"""GLM-5.3 base: CoT (default, low effort) vs explained output, gated passes. Run from results/outctl: python3 glm_analysis.py [DIR]"""
import json, collections, statistics as st, random, sys
D = sys.argv[1] if len(sys.argv) > 1 else "xfull"
G = {s: json.load(open(f"{D}/{s}/glm-base-or/graded.json" if D == "xfull" else f"{D}/{s}/graded.json")) for s in ("cot", "cot_low", "expl")}
T = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "end_of_sentence", "lowercase_thinking", "meow_between_words", "uppercase_thinking", "alternating_case", "ignore_question"]
ok = lambda g: (g["strict"] == 1 and bool(g["attempt"])) if g["constraint"] != "ignore_question" else g["strict"] == 1
print("| arm | n | pass % excl. ignore_q | ignore_q % | acc % | median words | truncated |\n|---|---|---|---|---|---|---|")
for s, g in G.items():
    ng = [x for x in g if x["constraint"] != "ignore_question"]; ig = [x for x in g if x["constraint"] == "ignore_question"]
    print(f"| {s} | {len(g)} | {100*sum(ok(x) for x in ng)/len(ng):.1f} | {100*sum(ok(x) for x in ig)/len(ig):.0f} | {100*sum(x['correct'] for x in g)/len(g):.1f} | {st.median(x['words'] for x in g):.0f} | {sum(x['stop']=='length' for x in g)} |")
print("\n| constraint | CoT default | CoT low | CoT pooled | output | pooled CoT / output |\n|---|---|---|---|---|---|")
rows = {}
for t in T:
    r = {s: [ok(x) for x in G[s] if x["constraint"] == t] for s in G}; pc = r["cot"] + r["cot_low"]; rows[t] = (100*st.mean(pc), 100*st.mean(r["expl"]))
    print(f"| {t} | {100*st.mean(r['cot']):.0f} | {100*st.mean(r['cot_low']):.0f} | {100*st.mean(pc):.0f} | {100*st.mean(r['expl']):.0f} | {st.mean(pc)/st.mean(r['expl']):.2f} |")
EDGES = [5, 20, 50, 100, 200, 500, 10**9]; b = lambda w: next(i for i, e in enumerate(EDGES) if w <= e)
def gap(rs, arms):
    s = collections.defaultdict(lambda: [[], []])
    for side, x in rs:
        if x["constraint"] == "ignore_question" or (side != "expl" and side not in arms): continue
        s[(x["constraint"], b(x["words"]))][0 if side != "expl" else 1].append(ok(x))
    num = den = 0
    for c, e in s.values():
        if c and e: w = min(len(c), len(e)); num += w*(st.mean(e)-st.mean(c)); den += w
    return 100*num/den if den else None
allr = [(s, x) for s in G for x in G[s]]; by = collections.defaultdict(list)
for r in allr: by[r[1]["key"]].append(r)
keys = sorted(by)
print()
for arms, lab in (({"cot", "cot_low"}, "CoT pooled"), ({"cot"}, "CoT default"), ({"cot_low"}, "CoT low")):
    rng = random.Random(0); v = sorted(x for x in (gap([r for k in rng.choices(keys, k=len(keys)) for r in by[k]], arms) for _ in range(1000)) if x is not None)
    print(f"matched output - {lab}: {gap(allr, arms):+.1f} [{v[25]:+.1f}, {v[974]:+.1f}]")
# type-level rank correlation (excl ignore_question)
def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v); i = 0
    while i < len(v):
        j = i
        while j+1 < len(v) and v[o[j+1]] == v[o[i]]: j += 1
        for k in range(i, j+1): r[o[k]] = (i+j)/2
        i = j+1
    return r
def pear(x, y):
    mx, my = st.mean(x), st.mean(y); return sum((a-mx)*(c-my) for a, c in zip(x, y))/((sum((a-mx)**2 for a in x)*sum((c-my)**2 for c in y))**.5)
tt = T[:-1]; x = [rows[t][0] for t in tt]; y = [rows[t][1] for t in tt]
print(f"across 8 types: Spearman {pear(rank(x), rank(y)):+.2f}, Pearson {pear(x, y):+.2f}")
# item level
it = collections.defaultdict(lambda: [[], []]); con = {}
for s, g in G.items():
    for xx in g:
        if xx["constraint"] == "ignore_question": continue
        it[xx["key"]][0 if s != "expl" else 1].append(ok(xx)); con[xx["key"]] = xx["constraint"]
co = [k for k, (c, e) in it.items() if any(c) and not any(e)]; oo = [k for k, (c, e) in it.items() if any(e) and not any(c)]
print(f"items {len(it)}: both {sum(any(c) and any(e) for c, e in it.values())}, CoT-only {len(co)}, output-only {len(oo)}, neither {sum(not any(c) and not any(e) for c, e in it.values())}")
print("CoT-only by constraint:", dict(collections.Counter(con[k] for k in co)))
a = [z for k, (c, e) in it.items() if any(e) for z in c]; bb = [z for k, (c, e) in it.items() if not any(e) for z in c]
print(f"CoT pass rate where output passed at least once {100*st.mean(a):.1f}% (n={len(a)}) vs never {100*st.mean(bb):.1f}% (n={len(bb)})")
