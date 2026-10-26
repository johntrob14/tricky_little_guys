"""Build the light 'reason and respond in the prompt's language' SFT set for gpt-oss-120b from base gpt-oss's own correct English traces.
Source: cot-control/rl/data/pool_k4.jsonl (2000 MMLU-Pro questions, none in the 540 CoT-Control eval items; each row has one base gpt-oss-120b
medium-effort analysis-channel trace and its correctness). Seeded draw of correct rows with base_words <= 600; N_PER_LANG per language in
French/Spanish/German (Italian held out), plus N_EN English rows kept untranslated. Gemini 3.8 Flash (OpenRouter, T=0) translates the question
block (question + lettered options) and, separately, the reasoning trace; letters, numbers, formulas, units and names are kept. The final
answer stays the base format 'ANSWER: X'. Resumable.   python3 build_sft.py [N_PER_LANG] [N_EN]   -> sft_src.jsonl"""
import json, os, glob, random, sys, time, urllib.request, concurrent.futures as cf
KEY = next(l.split("=", 1)[1].strip().strip('"') for f in glob.glob(os.path.expanduser("~/.config/sauce/*.env")) for l in open(f) if l.startswith("OPENROUTER_API_KEY="))
D = os.path.dirname(os.path.abspath(__file__)); POOL = f"{D}/../../../rl/data/pool_k4.jsonl"
_ARGS = sys.argv[1:] if __name__ == "__main__" else []  # imported by build_back.py / build_mech.py: defaults only
NL, NEN = (int(_ARGS[0]) if len(_ARGS) > 0 else 100), (int(_ARGS[1]) if len(_ARGS) > 1 else 75)
TRANSLATOR = "google/gemini-3.8-flash"  # Claude Sonnet 5.5 refuses to translate model reasoning traces (OpenRouter content_filter: ToS on duplicating model outputs)
LANGS = ["French", "Spanish", "German"]
P = [json.loads(l) for l in open(POOL)]
P = [p for p in P if p["correct"] == 1 and p["base_words"] <= 600 and p["reasoning"].strip()]
random.Random(7).shuffle(P)
plan = [(p, L) for i, L in enumerate(LANGS) for p in P[i * NL:(i + 1) * NL]] + [(p, "English") for p in P[3 * NL:3 * NL + NEN]]
def block(p): return "Question: " + p["question"] + "\n\n" + "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(p["options"]))
SYS_Q = ("You are a professional translator. Translate the multiple-choice question below into {L}. Translate the question and every option, "
         "including the word 'Question'. Keep the option letters and the line structure exactly ('A) ...' on its own line), and keep numbers, formulas, "
         "LaTeX, units, chemical and proper names unchanged. Do not answer the question or add anything. Output only the translated text.")
SYS_R = ("You are a professional translator. The text below is a model's private step-by-step reasoning about a multiple-choice question. Translate it "
         "into natural, fluent {L}, as if a native {L} speaker had written this reasoning. Translate every sentence, one for one, including repetitions, calculations, hedges and "
         "self-corrections; never summarize, shorten or merge sentences. Keep the line breaks. Keep option letters, numbers, formulas, LaTeX, code, units, chemical and proper names unchanged. When the reasoning quotes "
         "the question or options in English, translate the quote too. Do not add, remove or fix anything. Output only the translation.")
def call(sysmsg, text):
    body = json.dumps({"model": TRANSLATOR, "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": text}],
                       "max_tokens": 16000, "temperature": 0, "reasoning": {"effort": "low"}}).encode()
    for a in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", body, {"Authorization": f"Bearer {KEY}", "content-type": "application/json"}), timeout=240))
            return d["choices"][0]["message"]["content"].strip(), d.get("usage", {}).get("cost") or 0, d["choices"][0].get("finish_reason")
        except Exception as e:
            err = e; time.sleep(2 ** a)
    raise err
out = f"{D}/sft_src.jsonl"
done = {(json.loads(l)["qid"], json.loads(l)["lang"]) for l in open(out)} if os.path.exists(out) else set()
def one(job):
    p, L = job
    row = dict(qid=p["qid"], lang=L, answer=p["answer"], english_q=block(p), english_r=p["reasoning"], base_words=p["base_words"])
    if L == "English": return row | dict(q=row["english_q"], r=row["english_r"], cost=0.0, finish="none")
    try:
        q, c1, f1 = call(SYS_Q.format(L=L), row["english_q"]); cost = c1
        for _ in range(2):  # retry once when the translation's length is off (Gemini sometimes condenses)
            r, c2, f2 = call(SYS_R.format(L=L), p["reasoning"]); cost += c2
            ratio = len(r.split()) / max(1, len(p["reasoning"].split()))
            if 0.75 <= ratio <= 1.5: break
        return row | dict(q=q, r=r, cost=cost, finish=f"{f1}/{f2}", ratio=round(ratio, 2), length_ok=0.75 <= ratio <= 1.5)
    except Exception as e:
        return row | dict(error=str(e))
def main():
    todo = [j for j in plan if (j[0]["qid"], j[1]) not in done]
    with cf.ThreadPoolExecutor(12) as ex: res = list(ex.map(one, todo))
    with open(out, "a") as f: f.writelines(json.dumps(r) + "\n" for r in res if "error" not in r)
    print(f"{sum('error' not in r for r in res)}/{len(res)} ok, ${sum(r.get('cost', 0) for r in res):.2f}; errors {[(r['qid'], r['error'][:200]) for r in res if 'error' in r][:3]}; "
          f"not stop {[r['qid'] for r in res if r.get('finish') not in ('stop/stop', 'none', None)][:5]}")
if __name__ == "__main__": main()
