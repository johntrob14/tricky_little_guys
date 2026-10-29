# Run from cot-control/rl: M=or_dsv4f uv run python ../results/outctl/elicit/probe_or.py (needs OPENROUTER_API_KEY)
# Which OpenRouter providers honor a raw thinking-off prompt? Render one item with elicit.render (OFF), send to each provider, report reasoning.
import sys, os, json, importlib.util, asyncio, httpx
sys.argv = ["x"]; slug_key = os.environ["M"]
spec = importlib.util.spec_from_file_location("elicit", "../results/outctl/elicit/elicit.py"); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
cfg = E.MODELS[slug_key]; fmt = cfg["fmt"]
tok = None
if cfg.get("hf"):
    from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained(cfg["hf"])
it = next(i for i in E.split("test") if i["constraint"] == "lowercase_thinking")
E.OFF = True; s, h, u = E.build(it, "outnocot", fmt); text = E.render(tok, fmt, s, h, u)
key = os.environ["OPENROUTER_API_KEY"]
async def main():
    async with httpx.AsyncClient(timeout=120) as c:
        eps = (await c.get(f"https://openrouter.ai/api/v1/models/{cfg['or_']}/endpoints", headers={"Authorization": f"Bearer {key}"})).json()["data"]["endpoints"]
        provs = sorted({e["provider_name"] for e in eps})
        async def one(p):
            body = {"model": cfg["or_"], "prompt": text, "max_tokens": 120, "temperature": 1.0, "provider": {"order": [p], "allow_fallbacks": False}, "usage": {"include": True}}
            try:
                j = (await c.post("https://openrouter.ai/api/v1/completions", json=body, headers={"Authorization": f"Bearer {key}"})).json()
                ch = j["choices"][0]; u = j.get("usage") or {}
                return f"{p:16s} prompt_tok {u.get('prompt_tokens')} reasoning_len {len(ch.get('reasoning') or '')} text: {repr((ch.get('text') or '')[:70])}"
            except Exception as e: return f"{p:16s} ERR {str(j)[:100] if 'j' in dir() else e}"
        print(f"{slug_key}: local prompt tokens ~{len(tok.encode(text)) if tok else '?'}")
        for r in await asyncio.gather(*(one(p) for p in provs)): print(r)
asyncio.run(main())
