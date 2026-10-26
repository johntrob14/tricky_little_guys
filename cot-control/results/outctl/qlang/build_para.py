"""Paraphrase control for langsft1: the same kept examples (results/rl/langsft1/examples.jsonl), with each translated row's trace instead
rewritten in English by the same translator (Gemini 3.8 Flash, T=0, low reasoning effort) under the same sentence-by-sentence instruction.
The 75 English rows stay untouched, as in langsft1. Question stays the original English. Same length-ratio guard (0.75-1.5, one retry).
python3 build_para.py -> sft_para.jsonl (fields as sft_src.jsonl; lang English, orig_lang the language langsft1 used)"""
import json, os, concurrent.futures as cf
import build_sft as B
D = os.path.dirname(os.path.abspath(__file__))
SYS_P = ("You are a professional editor. The text below is a model's private step-by-step reasoning about a multiple-choice question. Rewrite it "
         "in natural, fluent English, as if a careful native English speaker had written this reasoning. Rewrite every sentence, one for one, "
         "including repetitions, calculations, hedges and self-corrections; never summarize, shorten or merge sentences. Keep the line breaks. "
         "Keep option letters, numbers, formulas, LaTeX, code, units, chemical and proper names unchanged. Do not add, remove or fix anything. "
         "Output only the rewritten text.")
kept = [json.loads(l) for l in open(f"{D}/../../rl/langsft1/examples.jsonl")]
src = {(r["qid"], r["lang"]): r for r in map(json.loads, open(f"{D}/sft_src.jsonl"))}
rows = [src[(k["qid"], k["lang"])] for k in kept]
def one(r):
    base = dict(qid=r["qid"], orig_lang=r["lang"], lang="English", answer=r["answer"], english_q=r["english_q"], english_r=r["english_r"], q=r["english_q"])
    if r["lang"] == "English": return base | dict(r=r["english_r"], cost=0.0, ratio=1.0, length_ok=True)
    cost = 0.0
    for _ in range(2):
        p, c, f = B.call(SYS_P, r["english_r"]); cost += c; ratio = len(p.split()) / max(1, len(r["english_r"].split()))
        if 0.75 <= ratio <= 1.5: break
    return base | dict(r=p, cost=cost, finish=f, ratio=round(ratio, 2), length_ok=0.75 <= ratio <= 1.5)
if __name__ == "__main__":
    with cf.ThreadPoolExecutor(12) as ex: res = list(ex.map(one, rows))
    with open(f"{D}/sft_para.jsonl", "w") as f: f.writelines(json.dumps(r) + "\n" for r in res)
    print(f"{len(res)} rows, ${sum(r['cost'] for r in res):.2f}, length not ok {sum(not r['length_ok'] for r in res)}, "
          f"identical to original {sum(r['r'].strip() == r['english_r'].strip() for r in res if r['orig_lang'] != 'English')}")
