"""Build the 200-row partial-compliance labeling set (seed 0): 100 rows from Neel's 34-model run, 100 from John's CoT arm.
Stratified by instruction type; stored passes oversampled to ~25%. Each row's view is capped so both labelers see the same text:
reasoning up to 6,000 chars (longer: first 4,000 + last 2,000 with a marked gap), answer up to 1,500 chars."""
import json, glob, random, os, collections, sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "set"
random.seed(SEED)
EXCLUDE = set()
for p in sys.argv[3:]:
    EXCLUDE |= {json.loads(l)["item"] for l in open(p)}
NEEL = os.path.expanduser("~/code/neel-astra/raw/cot_control_data/scratch_cot_control")
OUT = os.path.expanduser("~/code/tlg-arena-rl/cot-control/results/outctl")
TYPES = ["word_suppression", "multiple_word_suppression", "ignore_question", "repeat_sentences", "end_of_sentence",
         "meow_between_words", "lowercase_thinking", "uppercase_thinking", "alternating_case"]
def view(t, cap, head, tail):
    t = t or ""
    if len(t) <= cap: return t, len(t), False
    return t[:head] + f"\n\n[... {len(t) - head - tail} characters omitted ...]\n\n" + t[-tail:], len(t), True
def requirement(prompt):
    return "Requirement:" + prompt.split("Requirement:", 1)[1] if "Requirement:" in prompt else ""
rows = []
# Neel: 100 rows, ~11 per type, 25% stored passes where available
neel = []
for f in glob.glob(f"{NEEL}/rows/*.jsonl"):
    g = {json.loads(l)["item_id"]: json.loads(l) for l in open(f.replace("/rows/", "/graded/"))}
    for l in open(f):
        r = json.loads(l)
        if r.get("status") not in (None, "ok") or r["item_id"] not in g: continue
        if f"{r['model']}|{r['item_id']}" in EXCLUDE: continue
        r["compliance"] = g[r["item_id"]].get("compliance"); neel.append(r)
for i, t in enumerate(TYPES):
    k = 12 if i < 1 else 11
    pool = [r for r in neel if r["mode"] == t]; pas = [r for r in pool if r["compliance"] == 1]; fail = [r for r in pool if r["compliance"] != 1]
    take = random.sample(pas, min(len(pas), 3)) ; take += random.sample(fail, k - len(take))
    for r in take:
        rv, rl, rt = view(r.get("reasoning"), 6000, 4000, 2000); av, al, at = view(r.get("content"), 1500, 1000, 500)
        rows.append(dict(src="neel", model=r["model"], item=f"{r['model']}|{r['item_id']}", type=t, stored_pass=int(r["compliance"] == 1),
                         requirement=requirement(r["prompt"]), reasoning=rv, reasoning_chars=rl, reasoning_truncated=rt, answer=av, answer_truncated=at))
# John: 100 rows from the CoT arm, 60 from untrained/length-trained checkpoints, 40 from CoT-control-trained ones
TRAINED = ["mt1b-s150", "grpo3", "st1-ignore2", "st1-ignore3", "gsft1"]
ours = collections.defaultdict(list)
for f in glob.glob(f"{OUT}/*/cot/*/graded.json"):
    m = f.split("/")[-2]; d = f.split("/")[-4]
    for x in json.load(open(f)):
        x["_m"] = m; x["_d"] = d
        if f'{d}|{x["key"]}|s{x["sample_idx"]}' in EXCLUDE: continue
        ours["trained" if m in TRAINED else "other"].append(x)
for grp, per in (("other", [7, 7, 7, 7, 7, 7, 6, 6, 6]), ("trained", [5, 5, 5, 5, 4, 4, 4, 4, 4])):
    for t, k in zip(TYPES, per):
        pool = [x for x in ours[grp] if x["constraint"] == t]; pas = [x for x in pool if int(x["strict"]) == 1]; fail = [x for x in pool if int(x["strict"]) != 1]
        npass = min(len(pas), max(1, k // 4) if grp == "other" else k // 2)
        take = random.sample(pas, npass) + random.sample(fail, k - npass)
        for x in take:
            rv, rl, rt = view(x["out"], 6000, 4000, 2000); av, al, at = view(x.get("resp"), 1500, 1000, 500)
            rows.append(dict(src="john", model=x["_m"], item=f'{x["_d"]}|{x["key"]}|s{x["sample_idx"]}', type=t, stored_pass=int(x["strict"]),
                             requirement=requirement(x["user"]), reasoning=rv, reasoning_chars=rl, reasoning_truncated=rt, answer=av, answer_truncated=at))
random.shuffle(rows)
for i, r in enumerate(rows): r["id"] = (f"L{i:03d}" if SEED == 0 else f"M{i:03d}")
os.makedirs(OUTDIR, exist_ok=True)
# labeler view hides src, model and stored_pass
with open(f"{OUTDIR}/rows_full.jsonl", "w") as f: f.writelines(json.dumps(r) + "\n" for r in rows)
with open(f"{OUTDIR}/rows_blind.jsonl", "w") as f:
    for r in rows: f.write(json.dumps({k: r[k] for k in ("id", "type", "requirement", "reasoning", "reasoning_chars", "reasoning_truncated", "answer", "answer_truncated")}) + "\n")
c = collections.Counter((r["src"], r["stored_pass"]) for r in rows); print(len(rows), dict(c)); print(collections.Counter(r["type"] for r in rows))
print("truncated reasoning", sum(r["reasoning_truncated"] for r in rows), "total view chars", sum(len(r["reasoning"]) + len(r["answer"]) for r in rows))
