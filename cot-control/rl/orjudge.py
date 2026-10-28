"""Base-model judge calls through OpenRouter. Ordinary inference goes to OpenRouter; Tinker is only for sampling and training the policy.

    judge = OpenRouterJudge("openai/gpt-oss-120b", effort="low")
    final_text, n_in, n_out = await judge.ask(prompt)

Temperature 0 by default (temperature=None sends none, so the provider default applies). `provider` pins one upstream provider (no fallbacks), since OpenRouter otherwise spreads a model over many providers with different quantizations. 408, 429, 5xx and timeouts are retried with exponential backoff (8 attempts, about 4 minutes), whether they arrive as the HTTP status or as an error body inside a 200 reply; anything else, or running out of retries, raises. The exception is a provider timeout (OpenRouter's 504 "timed out", or the client's own 300 s timeout) on TIMEOUT_ATTEMPTS attempts in a row: a judge whose reasoning outruns the provider's 300 s limit times out identically on every retry, so the call gives up and returns an empty reply, which callers treat as an unparsed verdict (a fail, counted; the trainers stop above 10% unparsed). `timeouts` counts those calls. `usd` accumulates OpenRouter's reported cost,
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
    def __init__(self, model: str = "openai/gpt-oss-120b", effort: str = "low", max_tokens: int = 1500, concurrency: int = 32, retries: int = 8, transport=None, provider: str | None = None,
                 temperature: float | None = 0):
        self.model, self.effort, self.max_tokens, self.retries, self.provider = model, effort, max_tokens, retries, provider
        self.temperature = temperature  # None: send no temperature, so the provider default applies
        self.client = httpx.AsyncClient(timeout=300.0, transport=transport, headers={"Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"})
        self.sem = asyncio.Semaphore(concurrency)
        self.usd, self.providers, self.timeouts = 0.0, Counter(), 0

    async def ask(self, prompt: str) -> tuple[str, int, int]:
        reply, _, n_in, n_out = await self.ask_full(prompt)
        return reply, n_in, n_out

    async def ask_full(self, prompt: str) -> tuple[str, str, int, int]:
        """(reply, the reasoning text the provider returned, prompt tokens, completion tokens)."""
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": self.max_tokens,
                "reasoning": {"effort": self.effort}, "usage": {"include": True}} | ({} if self.temperature is None else {"temperature": self.temperature})
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
                    return "", "", 0, 0
                if attempt + 1 == self.retries:
                    raise RuntimeError(f"openrouter error {code} after {self.retries} attempts: {detail}")
                await asyncio.sleep(2 ** attempt)
        usage = d.get("usage") or {}
        self.usd += float(usage.get("cost") or 0.0)
        self.providers[d.get("provider")] += 1
        msg = d["choices"][0]["message"]
        return (msg.get("content") or "").strip(), (msg.get("reasoning") or "").strip(), int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


class TinkerJudge:
    """OpenRouterJudge's interface (ask -> (reply, prompt tokens, completion tokens); usd, providers, timeouts) over base gpt-oss-120b on Tinker,
    so a run's judges are not subject to an OpenRouter provider's rate limits. The prompt is one user turn rendered by gpt-oss's chat template at
    reasoning effort `effort`; temperature 0 (greedy) unless `temperature` is given; the reply is the final channel (empty when the generation
    hit max_tokens or has no final message, which callers count as unparsed). Prices from modelio.PRICES. Tinker errors are retried with
    exponential backoff (`retries` attempts)."""

    def __init__(self, effort: str = "low", max_tokens: int = 1500, concurrency: int = 64, retries: int = 8, temperature: float | None = 0):
        import tinker
        import modelio
        self.tinker, self.price = tinker, modelio.PRICES["openai/gpt-oss-120b"]
        self.effort, self.max_tokens, self.retries, self.temperature = effort, max_tokens, retries, 0.0 if temperature is None else temperature
        self.client = tinker.ServiceClient().create_sampling_client(base_model="openai/gpt-oss-120b")
        self.tok = self.client.get_tokenizer()
        self.sem = asyncio.Semaphore(concurrency)
        self.usd, self.providers, self.timeouts = 0.0, Counter(), 0

    async def ask(self, prompt: str) -> tuple[str, int, int]:
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, reasoning_effort=self.effort)
        ids = self.tok.encode(text, add_special_tokens=False)
        params = self.tinker.SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature, stop=[200002, 200012])
        async with self.sem:
            for attempt in range(self.retries):
                try:
                    s = (await self.client.sample_async(prompt=self.tinker.ModelInput.from_ints(ids), num_samples=1, sampling_params=params)).sequences[0]
                    break
                except Exception:  # noqa: BLE001 (transient Tinker errors; the last one is raised)
                    if attempt + 1 == self.retries:
                        raise
                    await asyncio.sleep(2 ** attempt)
        self.usd += (len(ids) * self.price["prefill"] + len(s.tokens) * self.price["sample"]) / 1e6
        self.providers["tinker"] += 1
        gen = self.tok.decode(s.tokens, skip_special_tokens=False)
        reply = gen.split("<|channel|>final<|message|>")[-1].split("<|")[0].strip() if "<|channel|>final<|message|>" in gen and s.stop_reason != "length" else ""
        return reply, len(ids), len(s.tokens)
