import csv, json, collections, sys
R = "/home/john/code/tlg-arena-rl/cot-control/results/tinker-runs"; O = sys.argv[1]
COT = {"base": "gptoss120b-t1", "st1-ignore2": "astra-st1-ignore2-medium-t1", "grpo3": "grpo3-step100-t1", "len2": "len2-step60-t1",
       "efflen1": "efflen1-step100-t1", "st1-third": "astra-st1-third-medium-t1"}
B = [(0, 5), (6, 49), (50, 199), (200, 10**9)]
def b(w): return next(i for i, (lo, hi) in enumerate(B) if lo <= w <= hi)
print("passes / n by text length (words); excludes ignore_question; CoT = analysis channel, out = response with CoT off\n")
print("| model | side | " + " | ".join(f"{lo}-{hi if hi < 10**9 else ''} w" for lo, hi in B) + " |"); print("|---|---|" + "---|" * len(B))
for m, run in COT.items():
    rec = {(r["dataset"], r["item_idx"], r["constraint"]): r["strict"] == "1" for r in csv.DictReader(open(f"{R}/{run}/analysis/records.csv")) if r["protocol"] == "astra"}
    cot = collections.defaultdict(lambda: [0, 0]); out = collections.defaultdict(lambda: [0, 0])
    for x in map(json.loads, open(f"{R}/{run}/generations/gpt-oss-120b/astra.jsonl")):
        if x["constraint"] == "ignore_question": continue
        c = cot[b(len((x["reasoning"] or "").split()))]; c[0] += rec[(x["dataset"], str(x["item_idx"]), x["constraint"])]; c[1] += 1
    for g in json.load(open(f"{O}/{m}/graded.json")):
        if g["constraint"] == "ignore_question": continue
        c = out[b(g["words"])]; c[0] += g["strict"] == 1; c[1] += 1
    for side, d in (("CoT", cot), ("out", out)):
        print(f"| {m} | {side} | " + " | ".join(f"{d[i][0]}/{d[i][1]}" if d[i][1] else "-" for i in range(len(B))) + " |")
