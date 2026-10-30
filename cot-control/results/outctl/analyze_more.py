"""Pool stored sample (CoT: Astra run, output: astra_out) with the 4 new samples per item (gsft1: 5 new, no stored) and compare CoT vs output
control. Run from cot-control/results/outctl:  python3 analyze_more.py"""
import csv, json, random, statistics as st, collections
R = "../tinker-runs"
COT = {"base": "gptoss120b-t1", "grpo3": "grpo3-step100-t1", "st1-ignore2": "astra-st1-ignore2-medium-t1", "gsft1": None,
       "len2": "len2-step60-t1", "efflen1": "efflen1-step100-t1", "st1-third": "astra-st1-third-medium-t1"}
T = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "end_of_sentence", "lowercase_thinking", "meow_between_words", "uppercase_thinking", "ignore_question", "alternating_case"]
EDGES = [5, 20, 50, 100, 200, 500, 10**9]; LAB = ["0-5", "6-20", "21-50", "51-100", "101-200", "201-500", "500+"]
def b(w): return next(i for i, e in enumerate(EDGES) if w <= e)
def load(m):
    rows = []  # (side, item, constraint, words, pass, correct)
    run = COT[m]
    if run:
        rec = {(r["dataset"], r["item_idx"], r["constraint"]): r for r in csv.DictReader(open(f"{R}/{run}/analysis/records.csv")) if r["protocol"] == "astra"}
        for x in map(json.loads, open(f"{R}/{run}/generations/gpt-oss-120b/astra.jsonl")):
            r = rec[(x["dataset"], str(x["item_idx"]), x["constraint"])]
            rows.append(("cot", (x["dataset"], str(x["item_idx"])), x["constraint"], len((x["reasoning"] or "").split()), r["strict"] == "1", r["correct"] == "True"))
        for g in json.load(open(f"astra_out/{m}/graded.json")):
            rows.append(("out", (g["dataset"], str(g["item_idx"])), g["constraint"], g["words"], g["strict"] == 1, g["correct"]))
    for s in ("cot", "out"):
        for g in json.load(open(f"more/{s}/{m}/graded.json")):
            rows.append((s, (g["dataset"], str(g["item_idx"])), g["constraint"], g["words"], g["strict"] == 1, g["correct"]))
    return rows
def boot(rows, f, n=2000, seed=0):
    items = sorted({r[1] + (r[2],) for r in rows}); by = collections.defaultdict(list)
    for r in rows: by[r[1] + (r[2],)].append(r)
    rng = random.Random(seed); vals = []
    for _ in range(n):
        smp = [r for k in rng.choices(items, k=len(items)) for r in by[k]]
        v = f(smp)
        if v is not None: vals.append(v)
    vals.sort(); return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]
def rate(rows, side, excl_ignore=False):
    xs = [r[4] for r in rows if r[0] == side and not (excl_ignore and r[2] == "ignore_question")]
    return sum(xs) / len(xs) if xs else None
def matched_gap(rows):
    """output minus CoT pass rate within constraint x length-bin strata (ignore_question excluded), weighted by min(n_cot, n_out)."""
    s = collections.defaultdict(lambda: {"cot": [], "out": []})
    for r in rows:
        if r[2] != "ignore_question": s[(r[2], b(r[3]))][r[0]].append(r[4])
    num = den = 0
    for v in s.values():
        if v["cot"] and v["out"]:
            w = min(len(v["cot"]), len(v["out"])); num += w * (sum(v["out"]) / len(v["out"]) - sum(v["cot"]) / len(v["cot"])); den += w
    return (num / den if den else None), den
data = {m: load(m) for m in COT}
json.dump({m: [list(r[:1]) + ["|".join(r[1])] + list(r[2:]) for r in rows] for m, rows in data.items()}, open("more/pooled_rows.json", "w"))
print("## Pass rates, all 108 items x 5 samples per side\n")
print("| model | CoT pass % [95% CI] | output pass % [95% CI] | CoT pass %, no ignore_q | output pass %, no ignore_q | output - CoT, same constraint and length bin [95% CI] (matched weight) | CoT acc % | output acc % | median CoT words | median output words | outputs <= 5 words |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for m, rows in data.items():
    c, o = rate(rows, "cot"), rate(rows, "out")
    cl = boot(rows, lambda x: rate(x, "cot")); ol = boot(rows, lambda x: rate(x, "out"))
    g, w = matched_gap(rows); gl = boot(rows, lambda x: matched_gap(x)[0])
    acc = lambda s: 100 * sum(r[5] for r in rows if r[0] == s) / sum(r[0] == s for r in rows)
    wd = lambda s: st.median(r[3] for r in rows if r[0] == s)
    print(f"| {m} | {100*c:.1f} [{100*cl[0]:.1f}, {100*cl[1]:.1f}] | {100*o:.1f} [{100*ol[0]:.1f}, {100*ol[1]:.1f}] | {100*rate(rows,'cot',True):.1f} | {100*rate(rows,'out',True):.1f} | "
          f"{100*g:+.1f} [{100*gl[0]:+.1f}, {100*gl[1]:+.1f}] ({w}) | {acc('cot'):.1f} | {acc('out'):.1f} | {wd('cot'):.0f} | {wd('out'):.0f} | {sum(r[0]=='out' and r[3]<=5 for r in rows)} |")
print("\n## Per constraint: pass % CoT / output (median words CoT / output), 60 samples per cell\n")
print("| constraint | " + " | ".join(COT) + " |"); print("|---" * (len(COT) + 1) + "|")
for t in T:
    cells = []
    for m, rows in data.items():
        cs = [r for r in rows if r[2] == t and r[0] == "cot"]; os_ = [r for r in rows if r[2] == t and r[0] == "out"]
        cells.append(f"{100*sum(r[4] for r in cs)/len(cs):.0f} / {100*sum(r[4] for r in os_)/len(os_):.0f} ({st.median(r[3] for r in cs):.0f} / {st.median(r[3] for r in os_):.0f})")
    print(f"| {t} | " + " | ".join(cells) + " |")
print("\n## By length bin (ignore_question excluded): passes/n, CoT | output\n")
print("| model | side | " + " | ".join(LAB) + " |"); print("|---|---|" + "---|" * len(LAB))
for m, rows in data.items():
    for s in ("cot", "out"):
        cnt = [[0, 0] for _ in LAB]
        for r in rows:
            if r[0] == s and r[2] != "ignore_question": k = cnt[b(r[3])]; k[0] += r[4]; k[1] += 1
        print(f"| {m} | {s} | " + " | ".join(f"{p}/{n}" if n else "-" for p, n in cnt) + " |")
