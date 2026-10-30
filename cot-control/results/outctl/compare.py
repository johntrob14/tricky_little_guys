import csv, json, sys, collections, statistics as st
O, R = sys.argv[1], "/home/john/code/tlg-arena-rl/cot-control/results/tinker-runs"
COT = {"base": "gptoss120b-t1", "st1-ignore2": "astra-st1-ignore2-medium-t1", "grpo3": "grpo3-step100-t1", "len2": "len2-step60-t1",
       "efflen1": "efflen1-step100-t1", "st1-third": "astra-st1-third-medium-t1"}
T = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "end_of_sentence", "lowercase_thinking", "meow_between_words", "uppercase_thinking", "ignore_question", "alternating_case"]
def k3(d, i, c): return (d, str(i), c)
res = {}
print("| model | CoT passes /108 | output passes /108 | output passes, >5 words | both / CoT only / output only | CoT acc | output acc | median CoT words | median output words |")
print("|---|---|---|---|---|---|---|---|---|")
for m, run in COT.items():
    rec = {k3(r["dataset"], r["item_idx"], r["constraint"]): r for r in csv.DictReader(open(f"{R}/{run}/analysis/records.csv")) if r["protocol"] == "astra"}
    cw = {k3(x["dataset"], x["item_idx"], x["constraint"]): len((x["reasoning"] or "").split()) for x in map(json.loads, open(f"{R}/{run}/generations/gpt-oss-120b/astra.jsonl"))}
    out = {k3(g["dataset"], g["item_idx"], g["constraint"]): g for g in json.load(open(f"{O}/{m}/graded.json"))}
    assert set(out) == set(rec), m
    c = {k: rec[k]["strict"] == "1" for k in out}; o = {k: out[k]["strict"] == 1 for k in out}
    res[m] = (c, o)
    both = sum(c[k] and o[k] for k in out); co = sum(c[k] and not o[k] for k in out); oo = sum(o[k] and not c[k] for k in out)
    print(f"| {m} | {sum(c.values())} | {sum(o.values())} | {sum(o[k] and out[k]['words'] > 5 for k in out)} | {both} / {co} / {oo} | "
          f"{sum(rec[k]['correct'] == 'True' for k in out)} | {sum(out[k]['correct'] for k in out)} | {st.median(cw.values()):.0f} | {st.median(g['words'] for g in out.values()):.0f} |")
print("\nPer constraint, CoT / output passes (of 12):\n")
print("| constraint | " + " | ".join(COT) + " |"); print("|---" * (len(COT) + 1) + "|")
for t in T:
    print(f"| {t} | " + " | ".join(f"{sum(v for k, v in res[m][0].items() if k[2] == t)} / {sum(v for k, v in res[m][1].items() if k[2] == t)}" for m in COT) + " |")
