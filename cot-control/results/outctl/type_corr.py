"""Does a model's CoT control track its output control across constraint types? Per model and constraint: CoT pass rate (default+low
effort pooled) and explained-output pass rate. Gated (attempt judge) for gpt-oss models and GLM; deterministic-only for the other xfull
models (judges were blocked when they were graded). ignore_question excluded. Run from results/outctl."""
import json, collections, math, random, statistics as st, sys, ast
sys.path.insert(0, "../../grading")
T = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "end_of_sentence", "lowercase_thinking", "meow_between_words", "uppercase_thinking", "alternating_case"]
SHORT = dict(zip(T, ["word", "multi-word", "repeat", "end-safe", "lower", "meow", "upper", "altcase"]))
def pearson(x, y):
    mx, my = st.mean(x), st.mean(y); sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/(sx*sy) if sx and sy else float("nan")
def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v); i = 0
    while i < len(v):
        j = i
        while j+1 < len(v) and v[o[j+1]] == v[o[i]]: j += 1
        for k in range(i, j+1): r[o[k]] = (i+j)/2
        i = j+1
    return r
spear = lambda x, y: pearson(rank(x), rank(y))
# rows: model -> list of (arm, key, constraint, ok)
R = collections.defaultdict(list)
D = json.load(open("big/rows_v3.json"))
for m, rs in D.items():
    for r in rs: R["gpt-oss " + m if m == "base" else m].append(("cot" if r["arm"] != "expl" else "out", "|".join(map(str, r["item"])), r["c"], bool(r["ok"])))
for s in ("cot", "cot_low", "expl"):
    for g in json.load(open(f"glm_snapshot2/{s}/graded.json")):
        if g["constraint"] != "ignore_question": R["GLM-5.3 base"].append(("cot" if s != "expl" else "out", g["key"], g["constraint"], g["strict"] == 1 and bool(g["attempt"])))
from cotswitch import graders
ITEMS = {it["key"]: it for it in json.load(open("all_items_rs.json"))}
def det(g):
    text = (g["out"] or "").strip()
    if not text: return False
    a = {k: v for k, v in (ast.literal_eval(ITEMS[g["key"]]["grader_args"]) if isinstance(ITEMS[g["key"]]["grader_args"], str) else ITEMS[g["key"]]["grader_args"]).items() if k not in ("prompt", "user_prompt")}
    for k in ("synonyms", "keywords"):
        if isinstance(a.get(k), str): a[k] = ast.literal_eval(a[k])
    return graders._grade_default(g["constraint"], text, a)["strict"] == 1
for m, lab in [("qwen-base-tk", "Qwen3.8 base (Tinker, Astra)*"), ("qwen-opsd-grpo1b", "Qwen opsd-grpo1b*"), ("ds-base-or", "DeepSeek base (OR)*"), ("ds-base-tk", "DeepSeek base (Tinker, Astra)*"), ("ds-rwsft1", "DeepSeek rwsft1*"), ("ds-obf2g", "DeepSeek obf2g*")]:
    for s in ("cot", "cot_low", "expl"):
        for l in open(f"xfull/{s}/{m}/gens.jsonl"):
            g = json.loads(l)
            if g["constraint"] != "ignore_question": R[lab].append(("cot" if s != "expl" else "out", g["key"], g["constraint"], det(g)))
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 20
def table(rs):
    t = {}
    for c in T:
        a = [x[3] for x in rs if x[2] == c and x[0] == "cot"]; b = [x[3] for x in rs if x[2] == c and x[0] == "out"]
        if a and b: t[c] = (100*st.mean(a), 100*st.mean(b), len(a), len(b))
    return t
print(f"Per model, across constraint types where output pass rate >= {THR:.0f}%: Spearman/Pearson of CoT rate vs output rate; types kept; bootstrap 95% CI on Spearman over items.  (* deterministic grading)\n")
pool_x, pool_y = [], []
for m, rs in R.items():
    t = table(rs); keep = [c for c in T if c in t and t[c][1] >= THR]
    if len(keep) < 4: print(f"{m}: only {len(keep)} types pass the threshold"); continue
    x = [t[c][0] for c in keep]; y = [t[c][1] for c in keep]
    by = collections.defaultdict(list)
    for r in rs: by[r[1]].append(r)
    keys = sorted(by); rng = random.Random(0); v = []
    for _ in range(500):
        smp = [r for k in rng.choices(keys, k=len(keys)) for r in by[k]]; tt = table(smp)
        if all(c in tt for c in keep):
            s = spear([tt[c][0] for c in keep], [tt[c][1] for c in keep])
            if not math.isnan(s): v.append(s)
    v.sort(); ci = f"[{v[int(.025*len(v))]:+.2f}, {v[int(.975*len(v))-1]:+.2f}]" if len(v) > 50 else "n/a"
    print(f"{m:32s} Spearman {spear(x,y):+.2f} {ci}  Pearson {pearson(x,y):+.2f}  (k={len(keep)})")
    print("   " + "  ".join(f"{SHORT[c]} {t[c][0]:.0f}/{t[c][1]:.0f}" for c in keep))
    mx, my = st.mean(x), st.mean(y); pool_x += [a-mx for a in x]; pool_y += [b-my for b in y]
print(f"\nPooled within-model (each model's means removed), all models above: Pearson {pearson(pool_x, pool_y):+.2f} over {len(pool_x)} model x type cells")
