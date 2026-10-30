"""Async OpenRouter chat-completions client: one concurrency cap, retries on transient errors, a running cost total.

`provider` pins one upstream provider with no fallbacks (OpenRouter otherwise spreads a model over providers with
different quantizations); the provider that served each reply is recorded either way. 408/409/429/5xx replies, error
bodies inside a 200, truncated (non-JSON) bodies and timeouts are retried with exponential backoff (8 attempts);
anything else raises.
"""
import asyncio
import json
import os
from pathlib import Path

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"


def api_key() -> str:
    if key := os.environ.get("OPENROUTER_API_KEY"):
        return key
    for line in (Path.home() / ".env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise KeyError("OPENROUTER_API_KEY is not in the environment or ~/.env")


class Client:
    def __init__(self, concurrency: int = 64, timeout: float = 900):
        self.sem = asyncio.Semaphore(concurrency)
        self.http = httpx.AsyncClient(timeout=timeout, headers={"Authorization": f"Bearer {api_key()}"})
        self.usd, self.calls, self.retries = 0.0, 0, 0

    async def chat(self, messages: list[dict], model: str, provider: str | None = None, effort: str | None = None,
                   temperature: float = 1.0, max_tokens: int = 16000, tools: list[dict] | None = None) -> dict:
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "usage": {"include": True}}
        if provider:
            body["provider"] = {"order": [provider], "allow_fallbacks": False}
        if effort:
            body["reasoning"] = {"effort": effort}
        if tools:
            body["tools"] = tools
        for attempt in range(8):
            async with self.sem:
                try:
                    r = await self.http.post(URL, json=body)
                    d = r.json() if r.status_code == 200 else {"error": {"code": r.status_code, "message": r.text[:300]}}
                    if not d.get("error"):
                        break
                    status = int(d["error"].get("code") or 500)
                    if status not in (408, 409, 429) and status < 500:
                        raise RuntimeError(f"openrouter {status}: {d['error']}")
                except (httpx.HTTPError, asyncio.TimeoutError, json.JSONDecodeError):   # a 200 with a truncated body happens
                    pass
            self.retries += 1
            await asyncio.sleep(2 ** attempt)
        else:
            raise RuntimeError("openrouter: out of retries")
        choice, usage = d["choices"][0], d["usage"]
        msg = choice["message"]
        self.usd += usage.get("cost") or 0
        self.calls += 1
        return {"reasoning": msg.get("reasoning") or "", "answer": msg.get("content") or "", "tool_calls": msg.get("tool_calls"),
                "reasoning_details": msg.get("reasoning_details"), "finish": choice.get("finish_reason"), "provider": d.get("provider"),
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "cost": usage.get("cost")}
