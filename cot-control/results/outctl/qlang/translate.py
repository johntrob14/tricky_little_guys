"""Translate the 100 seeded questions (same draw as respond_lang.py) with Claude Sonnet 5.5 on OpenRouter, for the question-language pilot.
Translates the question block (question + lettered options); letters, numbers, formulas, units and names kept. Resumable.
Run: python3 translate.py LANG N   (e.g. French 100)  -> qs_<LANG>.jsonl"""
import json, os, glob, random, sys, time, urllib.request, concurrent.futures as cf
KEY = next(l.split("=", 1)[1].strip().strip('"') for f in glob.glob(os.path.expanduser("~/.config/sauce/*.env")) for l in open(f) if l.startswith("OPENROUTER_API_KEY="))
O = "/home/john/code/tlg-arena-rl/cot-control/results/outctl"
LANG, N = sys.argv[1], int(sys.argv[2])
FMT = "\n\n\nYour response should be of the following format: 'ANSWER: X' (without quotes) where X is your chosen letter"
items = [json.loads(l) for l in open(f"{O}/../tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl")] + json.load(open(f"{O}/new_items.json"))
qs = random.Random(0).sample(items, 100)[:N]
SYS = (f"You are a professional translator. Translate the multiple-choice question below into {LANG}. Translate the question and every option. "
       "Keep the option letters and the line structure exactly ('A) ...' on its own line), and keep numbers, formulas, LaTeX, units, chemical and "
       "proper names unchanged. Do not answer the question or add anything. Output only the translated text.")
out = f"{os.path.dirname(os.path.abspath(__file__))}/qs_{LANG}.jsonl"
done = {json.loads(l)["key"] for l in open(out)} if os.path.exists(out) else set()
def tr(it):
    block = it["user_prompt"].split("\n\nRequirement:")[0].replace(FMT, "")
    body = json.dumps({"model": "anthropic/claude-sonnet-5.5", "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": block}],
                       "max_tokens": 3000, "temperature": 0}).encode()
    for a in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", body, {"Authorization": f"Bearer {KEY}", "content-type": "application/json"}), timeout=180))
            return dict(key=it["key"], lang=LANG, english=block, translated=d["choices"][0]["message"]["content"].strip(), gold=it["answer_gold"],
                        dataset=it["dataset"], cost=d.get("usage", {}).get("cost"), finish=d["choices"][0].get("finish_reason"))
        except Exception as e:
            err = str(e); time.sleep(2 ** a)
    return dict(key=it["key"], lang=LANG, error=err)
todo = [it for it in qs if it["key"] not in done]
with cf.ThreadPoolExecutor(8) as ex: res = list(ex.map(tr, todo))
with open(out, "a") as f: f.writelines(json.dumps(r) + "\n" for r in res)
ok = [r for r in res if "error" not in r]
print(f"{LANG}: {len(ok)}/{len(res)} translated, ${sum(r['cost'] or 0 for r in ok):.3f}", [r["key"] for r in res if "error" in r][:3])
