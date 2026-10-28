"""Model-level correlation of NCRI 15.2 (Neel's no-CoT reasoning index, models_ncri15_2.csv) with our measures on the 72 held-out Astra items
(dash/compare_long.json): output control CoT off / on, CoT control as written / elicited, and the elicitation gain. Base models only (organisms
have no NCRI). Spearman and Pearson, 95% CI by bootstrap over models (10k, seed 0). Also with GPT-6 Astra left out (NCRI 159, far above the rest).
python3 elicit/ncri_corr.py  -> elicit/dash/ncri_corr.json, elicit/dash/ncri_points.json"""
import csv, json, math, os, random
O = os.path.dirname(os.path.abspath(__file__))
NC = {r["slug"]: float(r["ncri15_2"]) for r in csv.DictReader(open("/home/john/code/neel-astra/raw/ncri_raw_data/models_ncri15_2.csv"))}
SLUG = {"gpt-oss-120b": "openai_gpt-oss-120b", "Qwen3.5-397B": "qwen_qwen3.5-397b-a17b", "GLM-5.3": "z-ai_glm-5.3", "Qwen3.8-27B": "qwen_qwen3.8-27b",
        "Inkling-Small": "thinkingmachines_inkling-small", "Qwen3.5-4B": "local_qwen3.5-4b", "Qwen3.5-9B": "qwen_qwen3.5-9b",
        "Nemotron-3 Ultra": "nvidia_nemotron-3-ultra-550b-a55b", "Nemotron-3 Nano": "nvidia_nemotron-3-nano-30b-a3b", "DeepSeek V4 Flash": "deepseek_deepseek-v4-flash",
        "Qwen3-32B": "qwen_qwen3-32b", "GLM-5.2": "z-ai_glm-5.2", "Claude Sonnet 4.5": "anthropic_claude-sonnet-4.5", "DeepSeek V3.2 exp": "deepseek_deepseek-v3.2-exp",
        "Kimi-K3": "moonshotai_kimi-k3", "GPT-5.1": "openai_gpt-5.1", "GPT-5.2": "openai_gpt-5.2", "GPT-5": "openai_gpt-5", "GPT-6 Astra": "openai_gpt-6-astra",
        "gpt-oss-20b": "openai_gpt-oss-20b", "Gemma 4 31B": "google_gemma-4-31b-it"}
# not in NCRI 15.2: DeepSeek V4.1 Flash 1017, DeepSeek R1 0528, o3
L = json.load(open(f"{O}/dash/compare_long.json"))
M = {"Output control, CoT off": "out_nocot", "Output control, CoT on": "out_cot", "CoT control, as written": "cot_base", "CoT control, elicited": "cot_elicit"}
by = {}
for r in L:
    if r["group"] == "Base model" and r["model"] in SLUG: by.setdefault(r["model"], {})[M[r["measure"]]] = r["value"]
for m, d in by.items():
    d["ncri"] = NC[SLUG[m]]
    if "cot_base" in d and "cot_elicit" in d: d["gain"] = round(d["cot_elicit"] - d["cot_base"], 1)
def rank(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r
def pear(a, b):
    n = len(a); ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a)); sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb) if sa and sb else float("nan")
spear = lambda a, b: pear(rank(a), rank(b))
def ci(f, a, b, rng, B=10000):
    v = []
    for _ in range(B):
        idx = [rng.randrange(len(a)) for _ in a]; x = f([a[i] for i in idx], [b[i] for i in idx])
        if not math.isnan(x): v.append(x)
    v.sort(); return [round(v[int(.025 * len(v))], 2), round(v[int(.975 * len(v))], 2)]
rng = random.Random(0); out = []
for k, lab in [("out_nocot", "Output control, CoT off"), ("out_cot", "Output control, CoT on"), ("cot_base", "CoT control, as written"),
               ("cot_elicit", "CoT control, elicited"), ("gain", "Elicitation gain (points)")]:
    for excl in (False, True):
        ms = [m for m, d in by.items() if k in d and not (excl and m == "GPT-6 Astra")]
        if excl and "GPT-6 Astra" not in [m for m, d in by.items() if k in d]: continue
        a, b = [by[m]["ncri"] for m in ms], [by[m][k] for m in ms]
        row = dict(measure=lab, subset="without GPT-6 Astra" if excl else "all", n=len(ms), spearman=round(spear(a, b), 2), spearman_ci=ci(spear, a, b, rng),
                   pearson=round(pear(a, b), 2), pearson_ci=ci(pear, a, b, rng), models=", ".join(ms))
        out.append(row)
        print(f"{lab:28s} {row['subset']:20s} n {row['n']:2d}  spearman {row['spearman']:+.2f} {row['spearman_ci']}  pearson {row['pearson']:+.2f} {row['pearson_ci']}")
json.dump(out, open(f"{O}/dash/ncri_corr.json", "w"), indent=1)
pts = [dict(model=m, ncri=round(d["ncri"], 1), **{k: d.get(k) for k in ("out_nocot", "out_cot", "cot_base", "cot_elicit", "gain")}) for m, d in sorted(by.items(), key=lambda x: x[1]["ncri"])]
json.dump(pts, open(f"{O}/dash/ncri_points.json", "w"), indent=1)
for p in pts: print(p)
