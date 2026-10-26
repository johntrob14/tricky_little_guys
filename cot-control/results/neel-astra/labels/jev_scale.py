"""Jev v6 partial-compliance labels at scale: John's CoT arm (17 gpt-oss-120b checkpoints, outctl {more,big}/cot/<model>) and
Neel's 34-model run. Full reasoning text (no truncation). Resumable: appends to scale/raw.jsonl and skips ids already there.
v6 = jev_tune.py v5 questions + derive(): answer_applies>=0.5 turns a non-compliant reasoning category into answer_only; on the 8
rule-graded types the official strict grade decides complied vs not (ignore_question stays with Jev).
Run from labels/:  python jev_scale.py run   |   python jev_scale.py summarize"""
import json, os, sys, glob, collections, concurrent.futures as cf, threading
sys.argv_saved = sys.argv; sys.argv = [sys.argv[0], "noop"]
import jev_tune as T  # reuses questions(), call(), derive(), CATS, GROUP
sys.argv = sys.argv_saved

NEEL = os.path.expanduser("~/code/neel-astra/raw/cot_control_data/scratch_cot_control")
OUT = os.path.expanduser("~/code/tlg-arena-rl/cot-control/results/outctl")
LABEL = dict(json.load(open(os.path.expanduser("~/code/results/neel-astra/labels/scale_model_labels.json"))))

def req(p): return "Requirement:" + p.split("Requirement:", 1)[1] if "Requirement:" in p else ""

def rows():
    for f in sorted(glob.glob(f"{NEEL}/rows/*.jsonl")):
        g = {json.loads(l)["item_id"]: json.loads(l) for l in open(f.replace("/rows/", "/graded/"))}
        for l in open(f):
            r = json.loads(l)
            if r.get("status") not in (None, "ok") or r["item_id"] not in g or g[r["item_id"]].get("compliance") is None: continue
            yield dict(id=f"neel|{r['model']}|{r['item_id']}", src="neel", model=r["model"], type=r["mode"], requirement=req(r["prompt"]),
                       reasoning=r.get("reasoning") or "", answer=r.get("content") or "", strict=int(g[r["item_id"]]["compliance"] == 1))
    for f in sorted(glob.glob(f"{OUT}/*/cot/*/graded.json")):
        m, d = f.split("/")[-2], f.split("/")[-4]
        for x in json.load(open(f)):
            yield dict(id=f"john|{m}|{d}|{x['key']}|{x['sample_idx']}", src="john", model=m, type=x["constraint"], requirement=req(x["user"]),
                       reasoning=x.get("out") or "", answer=x.get("resp") or "", strict=int(x["strict"]))

def run():
    os.makedirs("scale", exist_ok=True); path = "scale/raw.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    todo = [r for r in rows() if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    lock = threading.Lock(); n = [0]; cost = [0.0]
    def one(r):
        if not r["reasoning"].strip():
            res = dict(id=r["id"], empty=True)
        else:
            res = T.call(r, "v5")
        with lock:
            with open(path, "a") as f: f.write(json.dumps(res) + "\n")
            n[0] += 1; cost[0] += res.get("cost", 0)
            if n[0] % 1000 == 0: print(f"{n[0]}/{len(todo)} ${cost[0]:.3f}", flush=True)
    with cf.ThreadPoolExecutor(16) as ex: list(ex.map(one, todo))
    print(f"finished {n[0]}, ${cost[0]:.3f}; errors {sum(1 for l in open(path) if 'error' in json.loads(l))}")

def summarize():
    meta = {r["id"]: r for r in rows()}
    out = []
    for l in open("scale/raw.jsonl"):
        j = json.loads(l); r = meta.get(j["id"])
        if r is None or "error" in j: continue
        if j.get("empty"):
            cat, md, aa = "degenerate", 0.0, 0.0
        else:
            T.GRADER[j["id"]] = (r["strict"], r["type"])
            cat, _ = T.derive(j["answers"], "v6", j["id"])
            md, aa = j["answers"]["meta_discussion"]["noul"], j["answers"]["answer_applies"]["noul"]
        out.append(dict(id=j["id"], src=r["src"], model=r["model"], label=LABEL.get(r["model"], r["model"]), type=r["type"], strict=r["strict"],
                        category=cat, group=T.GROUP(cat), meta=md, answer_applies=aa,
                        cat_conf=None if j.get("empty") else j["answers"]["category"]["confidence"]))
    with open("scale/labels_v6.jsonl", "w") as f: f.writelines(json.dumps(o) + "\n" for o in out)
    by = collections.defaultdict(list)
    for o in out: by[(o["src"], o["label"])].append(o)
    print("| source | model | rows | strict pass % | complied % | partial % | answer only % | none % | degenerate % | meta % | meta % among complied |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (s, m), v in sorted(by.items(), key=lambda kv: (kv[0][0], -sum(o["group"] in ("complied", "partial") for o in kv[1]) / len(kv[1]))):
        n = len(v); comp = [o for o in v if o["category"] == "complied"]
        pct = lambda k: 100 * k / n
        print(f"| {s} | {m} | {n} | {pct(sum(o['strict'] for o in v)):.1f} | {pct(len(comp)):.1f} | {pct(sum(o['group']=='partial' for o in v)):.1f} | "
              f"{pct(sum(o['category']=='answer_only' for o in v)):.1f} | {pct(sum(o['group']=='none' and o['category']!='answer_only' for o in v)):.1f} | "
              f"{pct(sum(o['category']=='degenerate' for o in v)):.1f} | {pct(sum(o['meta']>=.5 for o in v)):.1f} | "
              f"{(100*sum(o['meta']>=.5 for o in comp)/len(comp)) if comp else float('nan'):.0f} |")

if __name__ == "__main__":
    run() if sys.argv[1] == "run" else summarize()
