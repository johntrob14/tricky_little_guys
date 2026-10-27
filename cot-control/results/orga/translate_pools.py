"""Translate training questions for orgA into French, Spanish and German (Gemini 3.8 Flash on OpenRouter, T=0, as qlang/build_sft.py):
400 explain_pool open questions and 300 pool_k4 MMLU-Pro questions (question + lettered options; letters, numbers, formulas kept), seeded draw.
Resumable. Output: pools_translated.jsonl rows {kind, id, lang, text, options?}.   python3 translate_pools.py"""
import json, os, random, sys, concurrent.futures as cf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../outctl/qlang"))
import build_sft as B
D = os.path.dirname(os.path.abspath(__file__)); RL = os.path.join(D, "../../rl/data")
OPEN = random.Random(11).sample([json.loads(l) for l in open(f"{RL}/explain_pool.jsonl")], 400)
MC = random.Random(12).sample([q for q in map(json.loads, open(f"{RL}/pool_k4.jsonl")) if q["base_words"] >= 10], 300)
SYS_OPEN = "You are a professional translator. Translate the question below into {L}. Keep numbers, formulas, units and proper names unchanged. Do not answer it or add anything. Output only the translation."
SYS_MC = ("You are a professional translator. Translate the multiple-choice question below into {L}: the question and every option. Keep the option letters and the line structure exactly "
          "('A) ...' on its own line), and keep numbers, formulas, LaTeX, units, chemical and proper names unchanged. Do not answer it or add anything. Output only the translated text.")
out = f"{D}/pools_translated.jsonl"
done = {(r["kind"], r["id"], r["lang"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
jobs = [("open", q["id"], L, q["question"]) for q in OPEN for L in ("French", "Spanish", "German")] + \
       [("mc", q["qid"], L, q["question"] + "\n\n" + "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(q["options"]))) for q in MC for L in ("French", "Spanish", "German")]
jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
def one(j):
    kind, i, L, text = j
    try:
        t, c, f = B.call((SYS_OPEN if kind == "open" else SYS_MC).format(L=L), text)
        return dict(kind=kind, id=i, lang=L, text=t, cost=c, finish=f)
    except Exception as e: return dict(kind=kind, id=i, lang=L, error=str(e))
with cf.ThreadPoolExecutor(16) as ex: res = list(ex.map(one, jobs))
with open(out, "a") as f: f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in res if "error" not in r)
print(f"{sum('error' not in r for r in res)}/{len(res)} ok, ${sum(r.get('cost', 0) for r in res):.2f}, not stop {sum(r.get('finish') != 'stop' for r in res if 'error' not in r)}")
