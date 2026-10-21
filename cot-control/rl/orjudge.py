"""Base-model judge calls through OpenRouter. Ordinary inference goes to OpenRouter; Tinker is only for sampling and training the policy.

    judge = OpenRouterJudge("openai/gpt-oss-120b", effort="low")
    final_text, n_in, n_out = await judge.ask(prompt)

Temperature 0. `provider` pins one upstream provider (no fallbacks), since OpenRouter otherwise spreads a model over many providers with different quantizations. 408, 429, 5xx and timeouts are retried with exponential backoff (8 attempts, about 2 minutes); anything else, or running out of retries, raises. `usd` accumulates OpenRouter's reported cost,
`providers` counts which upstream provider served each call (OpenRouter routes a model across providers).
"""
import asyncio
import os
from collections import Counter
from pathlib import Path

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"


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
        self.usd, self.providers = 0.0, Counter()

    async def ask(self, prompt: str) -> tuple[str, int, int]:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": self.max_tokens,
                "reasoning": {"effort": self.effort}, "usage": {"include": True}}
        if self.provider:
            body["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        async with self.sem:
            for attempt in range(self.retries):
                try:
                    r = await self.client.post(URL, json=body)
                except httpx.HTTPError as e:
                    if attempt + 1 == self.retries:
                        raise RuntimeError(f"openrouter request failed after {self.retries} attempts: {type(e).__name__}") from None
                    await asyncio.sleep(2 ** attempt)
                    continue
                if r.status_code in (408, 429) or r.status_code >= 500:
                    if attempt + 1 == self.retries:
                        raise RuntimeError(f"openrouter HTTP {r.status_code} after {self.retries} attempts: {r.text[:200]}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"openrouter HTTP {r.status_code}: {r.text[:200]}")
                break
        d = r.json()
        if d.get("error"):
            raise RuntimeError(f"openrouter error: {str(d['error'])[:200]}")
        usage = d.get("usage") or {}
        self.usd += float(usage.get("cost") or 0.0)
        self.providers[d.get("provider")] += 1
        return (d["choices"][0]["message"].get("content") or "").strip(), int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
