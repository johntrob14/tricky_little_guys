"""Data for the 'Neel's CoT-Control run' page, recomputed from Neel's raw rows/grades, our v8 labels of his traces, the audited sol-ncri
statistics and the cross-model language test. Writes data.json (embedded into index.html by build_page.py).
Run: ~/code/results/neel-astra/.venv/bin/python build_data.py"""
import json, csv, glob, collections, re, os
from scipy.stats import spearmanr
N = "/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control"
R = "/home/john/code/results/neel-astra"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ITEMS = json.load(open(f"{R}/labels/scale/../../work/sol-ncri/verify/models.csv".replace("models.csv", "models.csv"))) if False else None
mods = {r["model"]: r for r in csv.DictReader(open(f"{R}/work/sol-ncri/verify/models.csv"))}
# per-model from raw grades (published main set: A108; GLM-4.5 nested A54 as in sol-ncri) -- use sol-ncri's verified score/pass counts
v8 = collections.defaultdict(list)
for o in map(json.loads, open(f"{R}/labels/scale/labels_v8.jsonl")):
    if o["src"] == "neel": v8[o["model"]].append(o)
DIFF = {"word_suppression": "easy", "repeat_sentences": "easy", "lowercase_thinking": "easy", "multiple_word_suppression": "medium", "end_of_sentence": "medium",
        "uppercase_thinking": "medium", "ignore_question": "hard", "meow_between_words": "hard", "alternating_case": "hard"}
models = []
for m, r in mods.items():
    v = v8[m]; n = len(v); g = collections.Counter(o["group"] for o in v)
    models.append(dict(model=m, lab=m.split("/")[0], ncri=round(float(r["new_ncri"]), 1), score=round(float(r["score"]), 4), passes=int(r["pass_n"]), n=int(r["n"]),
                       accuracy=round(float(r["accuracy"]), 4), chars=round(float(r["chars"])), cap=round(float(r["cap"]), 4),
                       complied=round(g["complied"] / n, 4), partial=round(g["partial"] / n, 4), displaced=round(g["answer_only"] / n, 4),
                       none=round(g["none"] / n, 4), degenerate=round(g["degenerate"] / n, 4), meta=round(sum(o["meta"] >= .5 for o in v) / n, 4), v8_rows=n))
models.sort(key=lambda x: -x["ncri"])
# passes by difficulty from raw grades (all rows in graded/, deduplicated last record per item, compliance==1)
by_diff = collections.Counter(); tot_diff = collections.Counter()
for f in glob.glob(f"{N}/graded/*.jsonl"):
    last = {}
    for l in open(f):
        x = json.loads(l); last[x["item_id"]] = x
    for x in last.values():
        if x.get("compliance") is None: continue
        d = DIFF[x["mode"]]; tot_diff[d] += 1; by_diff[d] += int(x["compliance"] == 1)
by_type = collections.Counter(); tot_type = collections.Counter()
for f in glob.glob(f"{N}/graded/*.jsonl"):
    last = {}
    for l in open(f):
        x = json.loads(l); last[x["item_id"]] = x
    for x in last.values():
        if x.get("compliance") is None: continue
        tot_type[x["mode"]] += 1; by_type[x["mode"]] += int(x["compliance"] == 1)
# Qwen 3.5 uppercase: reasoning vs answer with no lowercase letters (answer with the ANSWER line removed, >=20 letters)
qr = qa = qn = 0
for f in glob.glob(f"{N}/rows/qwen_qwen3.5-*.jsonl"):
    for l in open(f):
        x = json.loads(l)
        if x.get("mode") != "uppercase_thinking" or x.get("status") not in (None, "ok"): continue
        qn += 1; r_ = x.get("reasoning") or ""; c_ = re.sub(r"ANSWER\s*:\s*\(?[A-Za-z]\)?\W*$", "", (x.get("content") or "").strip())
        qr += bool(r_) and not re.search("[a-z]", r_); qa += len(re.findall("[A-Za-z]", c_)) >= 20 and not re.search("[a-z]", c_)
# correlations across the 34 models
keys = ["score", "complied", "partial", "displaced", "meta", "accuracy", "chars"]
lab = {"score": "Strict CoT control", "complied": "Complied (v8)", "partial": "Partial (v8)", "displaced": "Displaced to answer", "meta": "Mentions the instruction",
       "accuracy": "Accuracy on these items", "chars": "Reasoning length"}
x = [m["ncri"] for m in models]
corr = [dict(measure=lab[k], key=k, rho=round(spearmanr(x, [m[k] for m in models])[0], 3), p=round(spearmanr(x, [m[k] for m in models])[1], 4)) for k in keys]
# cross-model language test on 12 of these models
rows = [json.loads(l) for l in open(f"{R}/work/lang_cross/rows.jsonl")]
g = collections.defaultdict(lambda: collections.defaultdict(list))
for r_ in rows: g[r_["model"]][r_["cond"]].append(r_)
lang = [dict(model=m, reasoning=sum(x_["r_strict"] or 0 for x_ in v["think_fr"]), reasoning_n=len(v["think_fr"]),
             answer=sum(x_["a_strict"] or 0 for x_ in v["answer_fr"]), answer_n=len(v["answer_fr"])) for m, v in sorted(g.items())]
S = dict(json.load(open(f"{R}/work/sol-ncri/verify/fast_correlations.json")))
P = json.load(open(f"{R}/work/sol-ncri/verify/power_checks.json")); A = json.load(open(f"{R}/work/sol-ncri/verify/aggregate_checks.json"))
C = json.load(open(f"{R}/work/sol-ncri/verify/confound_checks.json")); FAM = json.load(open(f"{R}/work/sol-ncri/verify/family_checks.json"))
stats = dict(rho=S["new_ncri__score"]["rho"], rho_ci=S["new_ncri__score"]["ci"], rho_acc=S["new_ncri__accuracy"]["rho"], rho_acc_ci=S["new_ncri__accuracy"]["ci"],
             rho_chars=S["new_ncri__chars"]["rho"], no_qwen=A["no_qwen35"]["rho"], no_glm=A["no_glm"]["rho"], glm_within=FAM["glm_all"]["rho"], ds_within=FAM["deepseek"]["rho"],
             between_labs=C["between"]["rho"], partial_acc=C["partial"]["accuracy"], partial_chars=C["partial"]["chars"], zero_upper=P["zero_upper"],
             power=[p for p in P["power"] if p["sigma"] > 0], passes=sum(m["passes"] for m in models), rows=sum(m["n"] for m in models),
             zero_models=sum(m["passes"] == 0 for m in models), n_models=len(models), qwen_upper_n=qn, qwen_upper_reasoning=qr, qwen_upper_answer=qa)
diff = [dict(level=d, passes=by_diff[d], rows=tot_diff[d]) for d in ("easy", "medium", "hard")]
types = [dict(type=t, difficulty=DIFF[t], passes=by_type[t], rows=tot_type[t]) for t in sorted(tot_type, key=lambda t: -by_type[t] / tot_type[t])]
json.dump(dict(models=models, corr=corr, lang=lang, stats=stats, diff=diff, types=types), open("data.json", "w"), indent=1)
print(stats); print(diff); print(len(models), "models")
