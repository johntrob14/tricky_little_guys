"""Round 3 (validation of the v6/v7 fallback on the families where it fires): 100 Neel rows from the Qwen 3.5 family, Kimi K2.5/K2.6,
Inkling, Gemma 4 31B and GLM-4.5, stratified by instruction type (~11 each), seed 3, excluding every row in set/ and set2/. Half of the
rows are drawn from those where v6's fallback fired at scale (Jev said complied, grader failed), half from the rest. Same capped view."""
import json, random, collections, os
random.seed(3)
FAM = ("qwen/qwen3.5-", "moonshotai/kimi-k2.5", "moonshotai/kimi-k2.6", "thinkingmachines/inkling", "google/gemma-4-31b-it", "z-ai/glm-4.5")
used = {json.loads(l)["item"] for p in ("set/items_used.jsonl",) for l in open(p)}
used |= {json.loads(l)["item"] for l in open("set2/rows_full.jsonl")}
raw = {j["id"]: j for j in map(json.loads, open("scale/raw.jsonl"))}
NEEL = os.path.expanduser("~/code/neel-astra/raw/cot_control_data/scratch_cot_control")
import glob
pool = []
for f in glob.glob(f"{NEEL}/rows/*.jsonl"):
    g = {json.loads(l)["item_id"]: json.loads(l) for l in open(f.replace("/rows/", "/graded/"))}
    for l in open(f):
        r = json.loads(l)
        if not r["model"].startswith(FAM) or r.get("status") not in (None, "ok") or r["item_id"] not in g: continue
        if f"{r['model']}|{r['item_id']}" in used: continue
        j = raw.get(f"neel|{r['model']}|{r['item_id']}")
        if not j or j.get("empty"): continue
        fired = j["answers"]["category"]["choice"] == "complied" and g[r["item_id"]].get("compliance") != 1 and r["mode"] != "ignore_question"
        pool.append((r, int(g[r["item_id"]].get("compliance") == 1), fired))
def view(t, cap, head, tail):
    t = t or ""
    return (t, False) if len(t) <= cap else (t[:head] + f"\n\n[... {len(t) - head - tail} characters omitted ...]\n\n" + t[-tail:], True)
TYPES = sorted({p[0]["mode"] for p in pool}); take = []
for i, t in enumerate(TYPES):
    k = 12 if i < 1 else 11
    fired = [p for p in pool if p[0]["mode"] == t and p[2]]; rest = [p for p in pool if p[0]["mode"] == t and not p[2]]
    nf = min(len(fired), k // 2); take += random.sample(fired, nf) + random.sample(rest, k - nf)
rows = []
for i, (r, sp, fired) in enumerate(random.sample(take, len(take))):
    rv, rt = view(r.get("reasoning"), 6000, 4000, 2000); av, at = view(r.get("content"), 1500, 1000, 500)
    rows.append(dict(id=f"N{i:03d}", src="neel", model=r["model"], item=f"{r['model']}|{r['item_id']}", type=r["mode"], stored_pass=sp, fallback_fired=int(fired),
                     requirement="Requirement:" + r["prompt"].split("Requirement:", 1)[1], reasoning=rv, reasoning_chars=len(r.get("reasoning") or ""),
                     reasoning_truncated=rt, answer=av, answer_truncated=at))
os.makedirs("set3", exist_ok=True)
open("set3/rows_full.jsonl", "w").writelines(json.dumps(r) + "\n" for r in rows)
open("set3/rows_blind.jsonl", "w").writelines(json.dumps({k: r[k] for k in ("id", "type", "requirement", "reasoning", "reasoning_chars", "reasoning_truncated", "answer", "answer_truncated")}) + "\n" for r in rows)
print(len(rows), "fallback rows", sum(r["fallback_fired"] for r in rows), collections.Counter(r["model"] for r in rows), collections.Counter(r["type"] for r in rows))
