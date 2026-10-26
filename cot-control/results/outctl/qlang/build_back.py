"""Back-translation control for langsft1: the same kept examples (results/rl/langsft1/examples.jsonl). Each translated trace (Gemini 3.8 Flash
French/Spanish/German, as trained in langsft1) is translated back into English by the same model with the same instruction (build_sft.SYS_R,
target English), so the text keeps whatever the translation pipeline changed besides the language. The 75 English rows stay untouched.
Question: the original English. Length guard vs the original English trace (0.75-1.5, one retry). (An English-paraphrase control, build_para.py,
was abandoned: Gemini compressed the paraphrases, median length ratio 0.71.)   python3 build_back.py -> sft_back.jsonl"""
import json, os, concurrent.futures as cf
import build_sft as B
D = os.path.dirname(os.path.abspath(__file__))
kept = [json.loads(l) for l in open(f"{D}/../../rl/langsft1/examples.jsonl")]
src = {(r["qid"], r["lang"]): r for r in map(json.loads, open(f"{D}/sft_src.jsonl"))}
rows = [src[(k["qid"], k["lang"])] for k in kept]
def one(r):
    base = dict(qid=r["qid"], orig_lang=r["lang"], lang="English", answer=r["answer"], english_q=r["english_q"], english_r=r["english_r"], q=r["english_q"])
    if r["lang"] == "English": return base | dict(r=r["english_r"], cost=0.0, ratio=1.0, length_ok=True)
    cost = 0.0
    for _ in range(2):
        p, c, f = B.call(B.SYS_R.format(L="English"), r["r"]); cost += c; ratio = len(p.split()) / max(1, len(r["english_r"].split()))
        if 0.75 <= ratio <= 1.5: break
    return base | dict(r=p, cost=cost, finish=f, ratio=round(ratio, 2), length_ok=0.75 <= ratio <= 1.5)
if __name__ == "__main__":
    with cf.ThreadPoolExecutor(12) as ex: res = list(ex.map(one, rows))
    with open(f"{D}/sft_back.jsonl", "w") as f: f.writelines(json.dumps(r) + "\n" for r in res)
    rs = sorted(r["ratio"] for r in res if r["orig_lang"] != "English")
    print(f"{len(res)} rows, ${sum(r['cost'] for r in res):.2f}, length not ok {sum(not r['length_ok'] for r in res)}, ratio quartiles {[rs[int(len(rs)*q)] for q in (.1,.25,.5,.75,.9)]}")
