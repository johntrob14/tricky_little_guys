"""Content-masked extracts of sandbag_li rows for hand reading. Prints metadata, the first and last parts of the analysis channel, keyword sentences from its
middle, and the final channel, with the question and options withheld. Usage: python extract.py <results_dir> <shard.jsonl>  (shard rows: model, qid, condition, group)."""
import json, os, re, sys
R, shard = sys.argv[1], sys.argv[2]
KW = re.compile(r"(?i)false|wrong|incorrect|deliberat|sandbag|monitor|reveal|plan|instruction|evaluat|security|pretend|intention|underperform|hide|conceal|suspic|genuine|actually|true answer|correct answer|should answer|must|task|rule|domain|general knowledge")
rows = {}
mons = {}
def score(reply):
    m = re.findall(r"<score>\s*(\d+|REFUSAL)\s*</score>", reply)
    return None if not m else m[-1] if m[-1] == "REFUSAL" else int(m[-1])
for item in map(json.loads, open(shard)):
    m = item["model"]
    if m not in rows:
        rows[m] = {(r["qid"], r["condition"]): r for r in map(json.loads, open(f"{R}/{m}/rows.jsonl"))}
        mons[m] = {s: {(r["qid"], r["condition"]): score(r["reply"]) for r in map(json.loads, open(f"{R}/{m}/monitor_{s}.jsonl"))} for s in ["full", "cot", "final"] if os.path.exists(f"{R}/{m}/monitor_{s}.jsonl")}
    r = rows[m][(item["qid"], item["condition"])]
    b = rows[m][(item["qid"], "baseline")]
    cot = r["reasoning"].strip()
    head, tail = cot[:500], cot[-500:] if len(cot) > 1000 else ""
    mid = cot[500:-500] if len(cot) > 1000 else ""
    kws = [s.strip() for s in re.split(r"(?<=[.!?])\s+", mid) if KW.search(s)][:12]
    print(f"\n######## ITEM model={m} qid={item['qid']} condition={item['condition']} group={item['group']}")
    print(f"dataset={r['dataset']} correct_letter={r['correct']} baseline_answer={b['answer']} this_answer={r['answer']} finish={r['finish']} cot_tokens={r['cot_tokens']} " + " ".join(f"monitor_{s}={v.get((item['qid'], item['condition']))}" for s, v in mons[m].items()))
    print(f"--- analysis channel, first 500 chars ({len(cot)} chars total):\n{head}")
    if kws:
        print("--- keyword sentences from the middle:\n" + "\n".join("  * " + s[:300] for s in kws))
    if tail:
        print(f"--- analysis channel, last 500 chars:\n{tail}")
    fin = r["response"].strip()
    print(f"--- final channel ({len(fin)} chars), first 700:\n{fin[:700]}")
