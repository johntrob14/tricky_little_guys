"""Probe which reasoning setting (and whether the no-think prompt) gets each model to 0 reasoning tokens, on 4 hard Astra items.
python3 probe_nothink.py MODEL [MODEL ...]"""
import json, os, sys, glob, urllib.request, concurrent.futures as cf
O = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, O); sys.path.insert(0, os.path.dirname(O)); sys.path.insert(0, os.path.join(O, "../../../rl")); sys.path.insert(0, os.path.join(O, "../../../grading"))
import nothink, astra_more as AM
def user_of(it):  # as or_expl.py
    q = AM.to_output(it["user_prompt"]); a, bq = q.split("\n\nRequirement:")
    return a + ("" if it["constraint"] == "ignore_question" else "\n\nState your answer, and then explain it.") + "\n\nRequirement:" + bq
KEY = next(l.split("=", 1)[1].strip().strip('"') for f in glob.glob(os.path.expanduser("~/.config/sauce/*.env")) for l in open(f) if l.startswith("OPENROUTER_API_KEY="))
its = [it for it in AM.items() if it["dataset"] in ("gpqa", "hle")][:4]
CONFIGS = [("disabled", {"enabled": False}), ("none", {"effort": "none"}), ("minimal", {"effort": "minimal"}), ("low", {"effort": "low"})]
def call(model, cfg, it, nt):
    msgs = nothink.messages(user_of(it)) if nt else [{"role": "user", "content": user_of(it)}]
    body = json.dumps({"model": model, "messages": msgs, "reasoning": cfg, "max_tokens": 3000}).encode()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", body, {"Authorization": f"Bearer {KEY}", "content-type": "application/json"}), timeout=240))
        u = d.get("usage", {}); return (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0, u.get("cost") or 0, len((d["choices"][0]["message"].get("reasoning") or ""))
    except urllib.error.HTTPError as e: return "ERR " + json.loads(e.read().decode()).get("error", {}).get("message", "")[:60], 0, 0
    except Exception as e: return "ERR " + str(e)[:60], 0, 0
for model in sys.argv[1:]:
    for name, cfg in CONFIGS:
        for nt in (False, True):
            with cf.ThreadPoolExecutor(4) as ex: res = list(ex.map(lambda it: call(model, cfg, it, nt), its))
            if isinstance(res[0][0], str): print(f"{model:32s} {name:9s} nothink={nt}: {res[0][0]}"); break
            print(f"{model:32s} {name:9s} nothink={nt}: reasoning tokens {[r[0] for r in res]} reasoning chars {[r[2] for r in res]} ${sum(r[1] for r in res):.3f}")
