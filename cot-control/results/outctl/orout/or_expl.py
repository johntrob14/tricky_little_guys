"""No-CoT output control on OpenRouter models: the 108 Astra CoT-Control items in the explained-output form used for gpt-oss (astra_more.py
side 'expl': requirement moved from the analysis channel to the response, "State your answer, and then explain it."), reasoning switched off
(reasoning.enabled false; for OpenAI models reasoning effort --effort, default minimal). Writes gens.jsonl in astra_more's format, so
`astra_more.py DIR --grade` grades it (reference strict grader + attempt judge). Records reasoning tokens to check the CoT really was off.
Run from cot-control/rl:  python3 ../results/outctl/orout/or_expl.py MODEL N [--effort minimal]   -> orout/<slug>/gens.jsonl"""
import json, os, sys, glob, time, urllib.request, concurrent.futures as cf
O = os.path.dirname(os.path.abspath(__file__)); OC = os.path.dirname(O); sys.path.insert(0, OC); sys.path.insert(0, os.path.join(OC, "../../rl")); sys.path.insert(0, os.path.join(OC, "../../grading"))
import astra_more as AM
KEY = next(l.split("=", 1)[1].strip().strip('"') for f in glob.glob(os.path.expanduser("~/.config/sauce/*.env")) for l in open(f) if l.startswith("OPENROUTER_API_KEY="))
model, N = sys.argv[1], int(sys.argv[2]); effort = sys.argv[sys.argv.index("--effort") + 1] if "--effort" in sys.argv else "minimal"
slug = model.replace("/", "_").replace("~", ""); out = f"{O}/{slug}"; os.makedirs(out, exist_ok=True)
its = AM.items(); done = {(r["key"], r["sample_idx"]) for r in map(json.loads, open(f"{out}/gens.jsonl"))} if os.path.exists(f"{out}/gens.jsonl") else set()
def user_of(it):
    q = AM.to_output(it["user_prompt"]); a, bq = q.split("\n\nRequirement:")
    return a + ("" if it["constraint"] == "ignore_question" else "\n\nState your answer, and then explain it.") + "\n\nRequirement:" + bq
def one(job):
    it, k = job; user = user_of(it)
    reasoning = {"effort": effort} if model.startswith(("openai/", "~openai/")) or "--effort" in sys.argv else {"enabled": False}
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": user}], "reasoning": reasoning, "max_tokens": 4000, "temperature": 1.0}).encode()
    for a in range(4):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", body, {"Authorization": f"Bearer {KEY}", "content-type": "application/json"}), timeout=300))
            m = d["choices"][0]["message"]; u = d.get("usage", {})
            return dict(model=model, side="expl", key=it["key"], sample_idx=k + 1, dataset=it["dataset"], item_idx=it["item_idx"], constraint=it["constraint"], gold=it["answer_gold"],
                        user=user, stop=d["choices"][0].get("finish_reason"), n_tokens=u.get("completion_tokens"), out=m.get("content") or "", reasoning_text=m.get("reasoning") or "",
                        reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"), cost=u.get("cost"), provider=d.get("provider"), effort=effort)
        except Exception as e:
            err = str(e); time.sleep(2 ** a)
    return dict(error=err, key=it["key"])
jobs = [(it, k) for it in its for k in range(N) if (it["key"], k + 1) not in done]
with cf.ThreadPoolExecutor(16) as ex: rows = list(ex.map(one, jobs))
ok = [r for r in rows if "error" not in r]
with open(f"{out}/gens.jsonl", "a") as f: f.writelines(json.dumps(r) + "\n" for r in ok)
rt = [r["reasoning_tokens"] or 0 for r in ok]
print(f"{model}: {len(ok)}/{len(rows)} ok, ${sum(r['cost'] or 0 for r in ok):.2f}, reasoning tokens >0 in {sum(x > 0 for x in rt)} (max {max(rt) if rt else 0}), errors {[r['error'][:80] for r in rows if 'error' in r][:2]}")
