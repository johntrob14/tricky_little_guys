"""CoT vs output (CoT off) vs output-with-explanation, with attempt-gated passes and per-constraint length-matched gaps.
Run from cot-control/results/outctl:  python3 analyze_v2.py"""
import csv, json, random, statistics as st, collections
R = "../tinker-runs"
COT = {"base": "gptoss120b-t1", "grpo3": "grpo3-step100-t1", "st1-ignore2": "astra-st1-ignore2-medium-t1", "gsft1": None,
       "len2": "len2-step60-t1", "efflen1": "efflen1-step100-t1", "st1-third": "astra-st1-third-medium-t1", "st1-ignore3": "astra-st1-ignore3-j20b-medium-t1"}
T = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "end_of_sentence", "lowercase_thinking", "meow_between_words", "uppercase_thinking", "alternating_case"]
EDGES = [5, 20, 50, 100, 200, 500, 10**9]
def b(w): return next(i for i, e in enumerate(EDGES) if w <= e)
ATT = {(a["model"], a["side"], a["dataset"], str(a["item_idx"]), a["constraint"], a["sample_idx"]): a["attempt"] for a in map(json.loads, open("more/attempt.jsonl"))}
def load(m):
    rows = []  # dict(side, item, c, w, ok, att, correct)
    def add(side, d, i, c, s, w, ok, cor, att=None):
        if att is None: att = ATT.get((m, side, d, str(i), c, s))
        rows.append(dict(side=side, item=(d, str(i)), c=c, w=w, ok=ok, att=att, cor=cor))
    if COT[m]:
        rec = {(r["dataset"], r["item_idx"], r["constraint"]): r for r in csv.DictReader(open(f"{R}/{COT[m]}/analysis/records.csv")) if r["protocol"] == "astra"}
        for x in map(json.loads, open(f"{R}/{COT[m]}/generations/gpt-oss-120b/astra.jsonl")):
            r = rec[(x["dataset"], str(x["item_idx"]), x["constraint"])]
            add("cot", x["dataset"], x["item_idx"], x["constraint"], 0, len((x["reasoning"] or "").split()), r["strict"] == "1", r["correct"] == "True")
        for g in json.load(open(f"astra_out/{m}/graded.json")):
            add("out", g["dataset"], g["item_idx"], g["constraint"], 0, g["words"], g["strict"] == 1, g["correct"])
    for side in ("cot", "out", "expl", "nocon"):
        for g in json.load(open(f"more/{side}/{m}/graded.json")):
            add(side, g["dataset"], g["item_idx"], g["constraint"], g["sample_idx"], g["words"], g["strict"] == 1, g["correct"], g.get("attempt"))
    return rows
def P(rows, side, gate=False):
    xs = [r for r in rows if r["side"] == side and r["c"] != "ignore_question"]
    return 100 * sum(r["ok"] and (r["att"] or not gate) for r in xs) / len(xs)
def matched(rows, a, bside, gate=False, cons=None):
    s = collections.defaultdict(lambda: {a: [], bside: []})
    for r in rows:
        if r["c"] != "ignore_question" and r["side"] in (a, bside) and (cons is None or r["c"] == cons):
            s[(r["c"], b(r["w"]))][r["side"]].append(r["ok"] and (r["att"] or not gate))
    num = den = 0
    for v in s.values():
        if v[a] and v[bside]:
            w = min(len(v[a]), len(v[bside])); num += w * (sum(v[bside]) / len(v[bside]) - sum(v[a]) / len(v[a])); den += w
    return (100 * num / den if den else None), den
def boot(rows, f, n=1000):
    keys = sorted({r["item"] + (r["c"],) for r in rows}); by = collections.defaultdict(list)
    for r in rows: by[r["item"] + (r["c"],)].append(r)
    rng = random.Random(0); v = []
    for _ in range(n):
        x = f([r for k in rng.choices(keys, k=len(keys)) for r in by[k]])
        if x is not None: v.append(x)
    v.sort(); return v[25], v[974]
D = {m: load(m) for m in COT}
print("All rates exclude ignore_question (96 items x 5 samples = 480 per arm). gated = strict pass AND the attempt judge says the text works on the question.\n")
print("| model | CoT pass % (gated) | output pass % (gated) | output+explain pass % (gated) | explain - CoT, matched [95% CI] | explain - CoT, matched, gated [95% CI] | median words CoT / output / explain | acc %: CoT on, no constraint* / CoT off, no constraint / CoT off + constraint / explain |")
print("|---|---|---|---|---|---|---|---|")
for m, rows in D.items():
    g1, w1 = matched(rows, "cot", "expl"); c1 = boot(rows, lambda x: matched(x, "cot", "expl")[0])
    g2, w2 = matched(rows, "cot", "expl", True); c2 = boot(rows, lambda x: matched(x, "cot", "expl", True)[0])
    med = lambda s: st.median(r["w"] for r in rows if r["side"] == s)
    acc = lambda s: 100 * sum(r["cor"] for r in rows if r["side"] == s) / sum(r["side"] == s for r in rows)
    print(f"| {m} | {P(rows,'cot'):.1f} ({P(rows,'cot',True):.1f}) | {P(rows,'out'):.1f} ({P(rows,'out',True):.1f}) | {P(rows,'expl'):.1f} ({P(rows,'expl',True):.1f}) | "
          f"{g1:+.1f} [{c1[0]:+.1f}, {c1[1]:+.1f}] (w {w1}) | {g2:+.1f} [{c2[0]:+.1f}, {c2[1]:+.1f}] (w {w2}) | {med('cot'):.0f} / {med('out'):.0f} / {med('expl'):.0f} | "
          f"{acc('cot'):.1f} / {acc('nocon'):.1f} / {acc('out'):.1f} / {acc('expl'):.1f} |")
print("\n*CoT-on accuracy is under the CoT constraint (the Astra prompt); there is no CoT-on unconstrained run here.\n")
print("## Per constraint, explain - CoT gated pass rate matched on length bin (matched weight)\n")
print("| constraint | " + " | ".join(COT) + " |"); print("|---" * (len(COT) + 1) + "|")
for t in T:
    cells = []
    for m, rows in D.items():
        g, w = matched(rows, "cot", "expl", True, t); cells.append(f"{g:+.0f} ({w})" if g is not None else "-")
    print(f"| {t} | " + " | ".join(cells) + " |")
print("\n## Per constraint gated pass %: CoT / explain (median words CoT / explain)\n")
print("| constraint | " + " | ".join(COT) + " |"); print("|---" * (len(COT) + 1) + "|")
for t in T:
    cells = []
    for m, rows in D.items():
        c = [r for r in rows if r["side"] == "cot" and r["c"] == t]; e = [r for r in rows if r["side"] == "expl" and r["c"] == t]
        f = lambda xs: 100 * sum(r["ok"] and r["att"] for r in xs) / len(xs)
        cells.append(f"{f(c):.0f} / {f(e):.0f} ({st.median(r['w'] for r in c):.0f} / {st.median(r['w'] for r in e):.0f})")
    print(f"| {t} | " + " | ".join(cells) + " |")
json.dump({m: rows for m, rows in D.items()}, open("more/rows_v2.json", "w"), default=list)
