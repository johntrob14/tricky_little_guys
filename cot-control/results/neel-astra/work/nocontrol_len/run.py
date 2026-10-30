"""Instruction-free CoT-on baseline for Neel's CoT-Control lane: does higher NCRI go with shorter reasoning WITHOUT a
CoT-control requirement? Same 36 Block-A questions, prompt = Neel's stored user prompt cut before "\n\nRequirement:"
(question + options + ANSWER-format line). Request body copied from his run_cc.py: reasoning enabled, T=0,
max_tokens 16000, require_parameters; provider order pinned to the provider his rows used most for that model
(fallbacks allowed, recorded). 12 models spanning NCRI. One sample per question.
Run: python run.py   (writes rows.jsonl, resumable)"""
import json, os, glob, re, time, collections, concurrent.futures as cf, urllib.request, urllib.error
N = "/home/john/code/neel-astra/raw/cot_control_data/scratch_cot_control"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KEY = next(l.split("=", 1)[1].strip().strip('"') for f in glob.glob(os.path.expanduser("~/.config/sauce/*.env")) for l in open(f) if l.startswith("OPENROUTER_API_KEY="))
MODELS = ["moonshotai/kimi-k3", "qwen/qwen3.8-2.4t-a95b", "deepseek/deepseek-v4-pro", "meituan/longcat-2.0", "google/gemma-4-31b-it", "z-ai/glm-5.2",
          "z-ai/glm-5", "minimax/minimax-m3", "z-ai/glm-4.6", "qwen/qwen3.5-122b-a10b", "qwen/qwen3.5-27b", "qwen/qwen3.5-9b"]
jobs = []
for m in MODELS:
    rows = [json.loads(l) for l in open(f"{N}/rows/{m.replace('/', '_')}.jsonl")]
    rows = [r for r in rows if r.get("block") == "A" and r.get("status") == "ok"]
    prov = collections.Counter(r.get("provider") for r in rows).most_common(1)[0][0]
    qs = {}
    for r in rows: qs.setdefault((r["dataset"], r["qid"]), r)
    for (ds, qid), r in sorted(qs.items()):
        jobs.append(dict(model=m, provider_pin=prov, dataset=ds, qid=qid, gold=r["gold"], prompt=r["prompt"].split("\n\nRequirement:")[0]))
done = set()
if os.path.exists("rows.jsonl"):
    for l in open("rows.jsonl"):
        r = json.loads(l)
        if r.get("status") == "ok": done.add((r["model"], r["dataset"], r["qid"]))
jobs = [j for j in jobs if (j["model"], j["dataset"], j["qid"]) not in done]
print(len(jobs), "calls to make")
def call(j):
    body = {"model": j["model"], "messages": [{"role": "user", "content": j["prompt"]}], "max_tokens": 16000, "temperature": 0.0,
            "reasoning": {"enabled": True}, "usage": {"include": True}, "provider": {"require_parameters": True, "order": [j["provider_pin"]], "allow_fallbacks": True}}
    err = ""
    for a in range(4):
        try:
            t = time.time()
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                         {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}), timeout=900))
            ch = d["choices"][0]; msg = ch["message"]; u = d.get("usage") or {}
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            if not reasoning and msg.get("reasoning_details"): reasoning = "".join((x.get("text") or x.get("summary") or "") for x in msg["reasoning_details"])
            content = msg.get("content") or ""
            if not reasoning:
                mm = re.match(r"\s*<think>(.*?)</think>", content, re.S)
                if mm: reasoning, content = mm.group(1).strip(), content[mm.end():].strip()
            ans = re.findall(r"ANSWER\s*:\s*\(?([A-J])\b", content)
            return dict(j, status="ok", reasoning=reasoning, content=content, chars=len(reasoning), finish=ch.get("finish_reason"), provider=d.get("provider"),
                        reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"), completion_tokens=u.get("completion_tokens"),
                        cost=u.get("cost"), extracted=ans[-1] if ans else None, correct=bool(ans) and ans[-1] == j["gold"], latency=round(time.time() - t, 1))
        except urllib.error.HTTPError as e:
            err = f"{e.code} {e.read()[:300]!r}"
            if e.code == 404 and body["provider"].get("require_parameters"): body["provider"].pop("require_parameters")
        except Exception as e:
            err = str(e)[:300]
        time.sleep(2 ** a)
    return dict(j, status="error", error=err)
with open("rows.jsonl", "a") as f, cf.ThreadPoolExecutor(24) as ex:
    for i, r in enumerate(ex.map(call, jobs)):
        f.write(json.dumps(r) + "\n"); f.flush()
        if i % 50 == 0: print(i, r["model"], r["status"], r.get("chars"), flush=True)
rows = [json.loads(l) for l in open("rows.jsonl")]
ok = [r for r in rows if r["status"] == "ok"]
print(f"{len(ok)}/{len(rows)} ok, ${sum(r.get('cost') or 0 for r in ok):.2f}")
