"""No-CoT output control on OpenRouter models (orout/<slug>/graded.json + gens.jsonl reasoning tokens) vs the same models' CoT control in
Neel's 34-model run (raw reasoning graded, same 108 block-A items). Gated pass = strict AND attempt judge; ignore_question excluded on both
sides. Output rows split by whether the model actually used reasoning tokens (some endpoints reason even at minimal effort).
python3 orout/analyze_or.py -> orout/or_summary.json"""
import json, os, collections
O = os.path.dirname(os.path.abspath(__file__)); N = os.path.expanduser("~/code/neel-astra/raw/cot_control_data/scratch_cot_control/graded")
M = {"openai_gpt-6-astra": None, "z-ai_glm-5.2": "z-ai_glm-5.2", "z-ai_glm-5.3": "z-ai_glm-5.3", "deepseek_deepseek-v4-flash": "deepseek_deepseek-v4-flash",
     "moonshotai_kimi-k3": "moonshotai_kimi-k3", "qwen_qwen3.5-397b-a17b": "qwen_qwen3.5-397b-a17b", "deepseek_deepseek-v3.2-exp": "deepseek_deepseek-v3.2-exp"}
out = []
for slug, nslug in M.items():
    gp = f"{O}/{slug}/graded.json"
    if not os.path.exists(gp): continue
    G = [g for g in json.load(open(gp)) if g["constraint"] != "ignore_question"]
    rt = {(r["key"], r["sample_idx"]): r.get("reasoning_tokens") or 0 for r in map(json.loads, open(f"{O}/{slug}/gens.jsonl"))}
    ok = lambda g: str(g.get("strict")) in ("1", "True") and bool(g.get("attempt"))
    z = [g for g in G if rt.get((g["key"], g["sample_idx"]), 0) == 0]; nz = [g for g in G if rt.get((g["key"], g["sample_idx"]), 0) > 0]
    per = collections.defaultdict(list)
    for g in G: per[g["constraint"]].append(ok(g))
    cot = None
    if nslug and os.path.exists(f"{N}/{nslug}.jsonl"):
        R = [json.loads(l) for l in open(f"{N}/{nslug}.jsonl")]; R = [r for r in R if r.get("mode") != "ignore_question" and r.get("compliance") in (0, 1)]
        cot = round(100 * sum(r["compliance"] for r in R) / len(R), 1)
    w = sorted(int(g.get("words") or 0) for g in G)
    o = dict(model=slug.replace("_", "/", 1), out_all=round(100 * sum(map(ok, G)) / len(G), 1), n=len(G), out_noreason=round(100 * sum(map(ok, z)) / len(z), 1) if z else None, n_noreason=len(z),
             out_reasoned=round(100 * sum(map(ok, nz)) / len(nz), 1) if nz else None, n_reasoned=len(nz), median_words=w[len(w) // 2], cot_neel=cot,
             per={c: round(100 * sum(v) / len(v), 1) for c, v in sorted(per.items())})
    out.append(o)
print("| model | output pass, all (n) | no reasoning tokens (n) | used reasoning (n) | median output words | Neel CoT pass |")
for o in out: print(f"| {o['model']} | {o['out_all']} ({o['n']}) | {o['out_noreason']} ({o['n_noreason']}) | {o['out_reasoned']} ({o['n_reasoned']}) | {o['median_words']} | {o['cot_neel']} |")
print("\nper constraint (output pass %):"); [print(o["model"], o["per"]) for o in out]
json.dump(out, open(f"{O}/or_summary.json", "w"), indent=1)
