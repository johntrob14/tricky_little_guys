"""Does elicitation raise output control? CoT off, 72 held-out items x 2: outnocot (plain output prompt) vs outshotstack (shotstack moved to the
response). Gated strict (strict AND attempt judge) on the response, 8 types (ignore_question excluded), constraint-balanced; rows where the model
wrote thinking anyway dropped. Paired 95% CI for the difference by bootstrap over items (10k, seed 0). Also partial (rl/astra_continuous.score on
the response, ungated) and median response words.   python3 elicit/outelicit.py -> elicit/dash/outelicit.json"""
import collections, json, os, random, sys
O = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(O, "../../../rl"))
import astra_continuous as AC
ITEMS = {json.loads(l)["key"]: json.loads(l) for l in open(os.path.join(O, "../../tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl"))}
import ast
def ga(k):
    a = ITEMS[k]["grader_args"]; a = ast.literal_eval(a) if isinstance(a, str) else dict(a)
    for x in ("synonyms", "keywords"):
        if isinstance(a.get(x), str): a[x] = ast.literal_eval(a[x])
    return a
NAME = {"gptoss": "gpt-oss-120b", "qwen35": "Qwen3.5-397B", "qwen38": "Qwen3.8-27B", "glm53": "GLM-5.3", "ds41": "DeepSeek V4.1 Flash 1017"}  # Inkling-Small left out: with the example turns it ignored thinking-off on 101/128 rows (responses ~4,200 words)
def load(m, v):
    G = [g for g in json.load(open(f"{O}/runs/{m}/{v}_test/graded.json")) if g["constraint"] != "ignore_question"]
    leak = sum(bool((g.get("cot") or "").strip()) for g in G)
    G = [g for g in G if not (g.get("cot") or "").strip()]
    for g in G: g["ok"] = int(str(g.get("strict")) in ("1", "True") and bool(g.get("attempt"))); g["pc"] = AC.score(dict(constraint=g["constraint"], reasoning=g.get("body") or "", grader_args=ga(g["key"])), g["ok"])
    return G, leak
def bal(G, f, pick=None):
    by = collections.defaultdict(list)
    for g in G:
        k = g["key"]
        for _ in range(pick[k] if pick else 1): by[g["constraint"]].append(f(g))
    return 100 * sum(sum(v) / len(v) for v in by.values() if v) / len([v for v in by.values() if v])
rng = random.Random(0); out = []
for m, lab in NAME.items():
    A, la = load(m, "outnocot"); B, lb = load(m, "outshotstack")
    keys = sorted({g["key"] for g in A} & {g["key"] for g in B}); d0 = bal(B, lambda g: g["ok"]) - bal(A, lambda g: g["ok"]); bs = []
    for _ in range(10000):
        pick = collections.Counter(rng.choice(keys) for _ in keys)
        try: bs.append(bal(B, lambda g: g["ok"], pick) - bal(A, lambda g: g["ok"], pick))
        except ZeroDivisionError: pass
    bs.sort(); w = lambda G: sorted(len((g.get("body") or "").split()) for g in G)[len(G) // 2]
    r = dict(model=lab, plain=round(bal(A, lambda g: g["ok"]), 1), elicited=round(bal(B, lambda g: g["ok"]), 1), diff=round(d0, 1),
             diff_lo=round(bs[int(.025 * len(bs))], 1), diff_hi=round(bs[int(.975 * len(bs))], 1),
             partial_plain=round(bal(A, lambda g: g["pc"]), 1), partial_elicited=round(bal(B, lambda g: g["pc"]), 1),
             attempt_plain=round(100 * sum(bool(g.get("attempt")) for g in A) / len(A)), attempt_elicited=round(100 * sum(bool(g.get("attempt")) for g in B) / len(B)),
             words_plain=w(A), words_elicited=w(B), n_plain=len(A), n_elicited=len(B), thinking_dropped_plain=la, thinking_dropped_elicited=lb)
    out.append(r); print(r)
json.dump(out, open(f"{O}/dash/outelicit.json", "w"), indent=1)
