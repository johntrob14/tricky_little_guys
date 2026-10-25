"""Does reasoning effort change response length on open-ended prompts? 50 question stems (prompts.json: eval items without options or
answer format), base models, thinking on, low / medium / high effort, 2 samples each.
  gpt-oss-120b  Tinker, harmony "Reasoning: low|medium|high"
  qwen3.8-27b   OpenRouter chat, provider DeepInfra, reasoning.effort low|medium|high
  ds-1017       OpenRouter raw completions, provider Parasail, DeepSeek effort prefix 25|50|75 (75 is the format default)
Run from cot-control/rl:  uv run python ../results/openended/run.py MODEL"""
import asyncio, json, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "../grading")
import httpx
from cotswitch import extract
import modelio
from effort_base import render as render_gptoss
D = os.path.dirname(os.path.abspath(__file__))
PROMPTS = json.load(open(f"{D}/prompts.json"))
EFFORTS = ["low", "medium", "high"]; DS_EFFORT = {"low": 25, "medium": 50, "high": 75}; N = 2; MAX = 16000

async def or_post(client, url, body):
    for attempt in range(10):
        try:
            r = await client.post(url, json=body, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=3600)
            j = r.json()
            if r.status_code == 200 and j.get("choices"): return j
            err = j.get("error", j)
            if "available credits" in str(err): raise RuntimeError(str(err)[:200])
        except (httpx.HTTPError, ValueError) as e:
            err = repr(e)
        await asyncio.sleep(min(60, 2 ** attempt))
    raise RuntimeError(str(err)[:300])

async def main():
    model = sys.argv[1]; path = f"{D}/{model}.jsonl"
    done = {(r["id"], r["effort"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    lock, sem = asyncio.Lock(), asyncio.Semaphore(24)
    if model == "gpt-oss-120b":
        import tinker
        sc = tinker.ServiceClient(); cl = await sc.create_sampling_client_async(base_model="openai/gpt-oss-120b"); tok = cl.get_tokenizer()
        params = tinker.SamplingParams(max_tokens=MAX, temperature=1.0, top_p=1.0, stop=modelio.GPT_OSS_STOP)
    http = httpx.AsyncClient()
    async def one(p, e):
        if (p["id"], e) in done: return
        async with sem:
            rows = []
            if model == "gpt-oss-120b":
                res = (await cl.sample_async(prompt=tinker.ModelInput.from_ints(render_gptoss(tok, p["prompt"], e)), num_samples=N, sampling_params=params)).sequences
                for k, s in enumerate(res):
                    fin = "length" if s.stop_reason == "length" else "stop"
                    sp = extract.split_output(extract.harmony_to_think(tok.decode(s.tokens, skip_special_tokens=False)), None, fin)
                    rows.append(dict(sample_idx=k, finish=fin, n_tokens=len(s.tokens), reasoning=sp["reasoning"], response=sp["response"]))
            elif model == "qwen3.8-27b":
                body = {"model": "qwen/qwen3.8-27b", "messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": MAX, "temperature": 1.0, "top_p": 1.0,
                        "reasoning": {"effort": e}, "provider": {"order": ["DeepInfra"], "allow_fallbacks": False}, "usage": {"include": True}}
                js = await asyncio.gather(*(or_post(http, "https://openrouter.ai/api/v1/chat/completions", body) for _ in range(N)))
                for k, j in enumerate(js):
                    m = j["choices"][0]["message"]
                    rows.append(dict(sample_idx=k, finish=j["choices"][0].get("finish_reason"), n_tokens=j["usage"]["completion_tokens"], reasoning=m.get("reasoning") or "",
                                     response=m.get("content") or "", cost=j["usage"].get("cost")))
            else:
                body = {"model": "deepseek/deepseek-v4.1-flash-1017", "prompt": modelio.render_ds(p["prompt"], effort=DS_EFFORT[e]), "max_tokens": MAX, "temperature": 1.0, "top_p": 1.0,
                        "provider": {"order": ["Parasail"], "allow_fallbacks": False}, "usage": {"include": True}}
                js = await asyncio.gather(*(or_post(http, "https://openrouter.ai/api/v1/completions", body) for _ in range(N)))
                for k, j in enumerate(js):
                    c = j["choices"][0]
                    rows.append(dict(sample_idx=k, finish=c.get("finish_reason"), n_tokens=j["usage"]["completion_tokens"], reasoning=c.get("reasoning") or "",
                                     response=c.get("text") or "", cost=j["usage"].get("cost")))
        for r in rows: r.update(model=model, id=p["id"], dataset=p["dataset"], effort=e, cot_words=len(r["reasoning"].split()), resp_words=len(r["response"].split()))
        async with lock:
            with open(path, "a") as f: f.writelines(json.dumps(r) + "\n" for r in rows)
    await asyncio.gather(*(one(p, e) for p in PROMPTS for e in EFFORTS), return_exceptions=False)
    print(model, "done")

if __name__ == "__main__":
    asyncio.run(main())
