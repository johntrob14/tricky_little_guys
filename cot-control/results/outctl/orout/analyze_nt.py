"""No-think no-CoT output control (orout/<slug>_nt, Neel NCRI elicitation: lowest reasoning setting + no-think system prompt + 2 direct-answer
few-shots). Gated pass = strict AND attempt; ignore_question excluded. Split rows by reasoning_tokens: clean (<=1, Qwen3 emits an empty think tag
= 1 token) vs reasoned. Compares with the earlier non-nothink run (orout/<slug>), Neel's CoT pass, and NCRI15_2.
python3 orout/analyze_nt.py -> orout/nt_summary.json"""
import json, os, glob, csv, collections, math
O = os.path.dirname(os.path.abspath(__file__)); N = os.path.expanduser("~/code/neel-astra/raw/cot_control_data/scratch_cot_control/graded")
ncri = {r["slug"]: float(r["ncri15_2"]) for r in csv.DictReader(open(os.path.expanduser("~/code/neel-astra/raw/ncri_raw_data/models_ncri15_2.csv"))) if r["ncri15_2"]}
ok = lambda g: str(g.get("strict")) in ("1", "True") and bool(g.get("attempt"))
pct = lambda xs: round(100 * sum(map(ok, xs)) / len(xs), 1) if xs else None
def ci(xs):
    if not xs: return None
    p = sum(map(ok, xs)) / len(xs); h = 1.96 * math.sqrt(p * (1 - p) / len(xs)); return [round(100 * (p - h), 1), round(100 * (p + h), 1)]
out = []
for d in sorted(glob.glob(f"{O}/*_nt")):
    slug = os.path.basename(d)[:-3]
    stop = {(r["key"], r["sample_idx"]): r.get("stop") for r in map(json.loads, open(f"{d}/gens.jsonl"))}
    G0 = [g for g in json.load(open(f"{d}/graded.json")) if g["constraint"] != "ignore_question"]
    G = [g for g in G0 if stop.get((g["key"], g["sample_idx"])) != "content_filter"]  # provider output filter (Sonnet: all meow + end_of_sentence items, on Bedrock and Anthropic)
    rt = {(r["key"], r["sample_idx"]): r.get("reasoning_tokens") or 0 for r in map(json.loads, open(f"{d}/gens.jsonl"))}
    rp = json.loads(open(f"{d}/gens.jsonl").readline()).get("reasoning_param")
    t = lambda g: rt.get((g["key"], g["sample_idx"]), 0)
    z = [g for g in G if t(g) <= 1]; nz = [g for g in G if t(g) > 1]
    old = f"{O}/{slug}/graded.json"; oldp = pct([g for g in json.load(open(old)) if g["constraint"] != "ignore_question"]) if os.path.exists(old) else None
    cot = None
    if os.path.exists(f"{N}/{slug}.jsonl"):
        R = [json.loads(l) for l in open(f"{N}/{slug}.jsonl")]; R = [r for r in R if r.get("mode") != "ignore_question" and r.get("compliance") in (0, 1)]
        cot = round(100 * sum(r["compliance"] for r in R) / len(R), 1)
    per = collections.defaultdict(list)
    for g in z: per[g["constraint"]].append(ok(g))
    rts = sorted(t(g) for g in G)
    C6 = [g for g in z if g["constraint"] not in ("meow_between_words", "end_of_sentence")]
    out.append(dict(model=slug.replace("_", "/", 1), reasoning_param=rp, n=len(G), out_all=pct(G), ci_all=ci(G), n_clean=len(z), out_clean=pct(z), ci_clean=ci(z),
                    n_reasoned=len(nz), out_reasoned=pct(nz), median_rtok=rts[len(rts) // 2], out_prev_run=oldp, cot_neel=cot, ncri=ncri.get(slug),
                    n_filtered=len(G0) - len(G), out_clean6=pct(C6), n_clean6=len(C6), acc=round(100 * sum(bool(g.get("correct")) for g in G) / len(G), 1), per_clean={c: round(100 * sum(v) / len(v)) for c, v in sorted(per.items())}))
out.sort(key=lambda o: -(o["out_clean"] if o["out_clean"] is not None else -1))
print("| model | setting | clean 6-constraint pass (n) | clean pass [95% CI] (n) | reasoned pass (n) | all | median rtok | prev run (no nothink) | Neel CoT | NCRI |")
for o in out: print(f"| {o['model']} | {o['reasoning_param']} | {o['out_clean6']} ({o['n_clean6']}) | {o['out_clean']} {o['ci_clean']} ({o['n_clean']}) | {o['out_reasoned']} ({o['n_reasoned']}) | {o['out_all']} | {o['median_rtok']} | {o['out_prev_run']} | {o['cot_neel']} | {o['ncri']} |")
def spear(a, b):
    r = lambda v: [sorted(v).index(x) + (sorted(v).count(x) - 1) / 2 for x in v]; ra, rb = r(a), r(b); n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n; c = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)); return round(c / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)), 2), n
cl = [o for o in out if o["n_clean"] >= 140]
for v, k in [(v, k) for v in ("out_clean", "out_clean6") for k in ("ncri", "cot_neel")]:
    p = [(o[v], o[k]) for o in cl if o[k] is not None]
    if len(p) >= 3: print(f"Spearman {v} vs {k}: rho, n =", spear(*zip(*p)), [o["model"] for o in cl if o[k] is not None])
print("\nper constraint, clean rows:"); [print(o["model"], o["per_clean"]) for o in out]
json.dump(out, open(f"{O}/nt_summary.json", "w"), indent=1)
