"""Base-model judge calls through OpenRouter. Ordinary inference goes to OpenRouter; Tinker is only for sampling and training the policy.

    judge = OpenRouterJudge("openai/gpt-oss-120b", effort="low")
    final_text, n_in, n_out = await judge.ask(prompt)

Temperature 0. `provider` pins one upstream provider (no fallbacks), since OpenRouter otherwise spreads a model over many providers with different quantizations. 408, 429, 5xx and timeouts are retried with exponential backoff (8 attempts, about 4 minutes), whether they arrive as the HTTP status or as an error body inside a 200 reply; anything else, or running out of retries, raises. The exception is a provider timeout (OpenRouter's 504 "timed out", or the client's own 300 s timeout) on TIMEOUT_ATTEMPTS attempts in a row: a judge whose reasoning outruns the provider's 300 s limit times out identically on every retry, so the call gives up and returns an empty reply, which callers treat as an unparsed verdict (a fail, counted; the trainers stop above 10% unparsed). `timeouts` counts those calls. `usd` accumulates OpenRouter's reported cost,
`providers` counts which upstream provider served each call (OpenRouter routes a model across providers).
"""
import asyncio
import os
from collections import Counter
from pathlib import Path

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_ATTEMPTS = 2


def load_key() -> str:
    if key := os.environ.get("OPENROUTER_API_KEY"):
        return key
    for line in (Path.home() / ".config/sauce/openrouter.env").read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() == "OPENROUTER_API_KEY":
            return v.strip().strip("'\"")
    raise RuntimeError("no OPENROUTER_API_KEY in the environment or ~/.config/sauce/openrouter.env")


class OpenRouterJudge:
    def __init__(self, model: str = "openai/gpt-oss-120b", effort: str = "low", max_tokens: int = 1500, concurrency: int = 32, retries: int = 8, transport=None, provider: str | None = None):
        self.model, self.effort, self.max_tokens, self.retries, self.provider = model, effort, max_tokens, retries, provider
        self.client = httpx.AsyncClient(timeout=300.0, transport=transport, headers={"Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"})
        self.sem = asyncio.Semaphore(concurrency)
        self.usd, self.providers, self.timeouts = 0.0, Counter(), 0

    async def ask(self, prompt: str) -> tuple[str, int, int]:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": self.max_tokens,
                "reasoning": {"effort": self.effort}, "usage": {"include": True}}
        if self.provider:
            body["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        async with self.sem:
            timeouts = 0
            for attempt in range(self.retries):
                try:
                    r = await self.client.post(URL, json=body)
                    d = r.json() if r.status_code == 200 else {}
                    code = int((d.get("error") or {}).get("code") or r.status_code)  # OpenRouter also reports upstream 5xx and timeouts inside a 200 body
                    detail = str(d.get("error") or r.text)[:200]
                except (httpx.HTTPError, ValueError) as e:  # ValueError: a 200 whose body is not JSON (seen once from DeepInfra via OpenRouter)
                    code, detail = 599, type(e).__name__
                if code == 200 and not d.get("error"):
                    break
                if code not in (408, 429) and code < 500:
                    raise RuntimeError(f"openrouter error {code}: {detail}")
                timeouts = timeouts + 1 if code == 599 or (code == 504 and "timed out" in detail) else 0
                if timeouts == TIMEOUT_ATTEMPTS:
                    self.timeouts += 1
                    return "", 0, 0
                if attempt + 1 == self.retries:
                    raise RuntimeError(f"openrouter error {code} after {self.retries} attempts: {detail}")
                await asyncio.sleep(2 ** attempt)
        usage = d.get("usage") or {}
        self.usd += float(usage.get("cost") or 0.0)
        self.providers[d.get("provider")] += 1
        return (d["choices"][0]["message"].get("content") or "").strip(), int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
