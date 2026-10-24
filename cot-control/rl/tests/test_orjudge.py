"""OpenRouterJudge retries 408/429/5xx whether they arrive as the HTTP status or as an error body inside a 200 reply, and raises on anything else."""
import asyncio

import httpx
import pytest

from orjudge import OpenRouterJudge

OK = {"choices": [{"message": {"content": "YES"}}], "usage": {"cost": 0.001, "prompt_tokens": 5, "completion_tokens": 1}, "provider": "Groq"}


def make(replies, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    async def nosleep(_):
        pass
    monkeypatch.setattr("orjudge.asyncio.sleep", nosleep)
    calls = []
    def handler(request):
        calls.append(request)
        status, body = replies[len(calls) - 1]
        return httpx.Response(status, json=body)
    return OpenRouterJudge(provider="Groq", transport=httpx.MockTransport(handler), retries=3), calls


def test_retryable_errors_are_retried(monkeypatch):
    j, calls = make([(200, {"error": {"code": 504, "message": "A Timeout Occurred"}}), (502, {}), (200, OK)], monkeypatch)
    assert asyncio.run(j.ask("q")) == ("YES", 5, 1)
    assert len(calls) == 3 and j.providers == {"Groq": 1} and j.usd == 0.001


def test_other_errors_and_exhausted_retries_raise(monkeypatch):
    j, calls = make([(200, {"error": {"code": 400, "message": "bad request"}})], monkeypatch)
    with pytest.raises(RuntimeError, match="400"):
        asyncio.run(j.ask("q"))
    assert len(calls) == 1
    j, calls = make([(200, {"error": {"code": 504}})] * 3, monkeypatch)
    with pytest.raises(RuntimeError, match="504 after 3 attempts"):
        asyncio.run(j.ask("q"))
    assert len(calls) == 3


def test_repeated_provider_timeout_returns_unparsed(monkeypatch):
    timeout = (200, {"error": {"code": 504, "message": "Provider timed out after 300599ms", "metadata": {"error_type": "timeout"}}})
    j, calls = make([timeout] * 3, monkeypatch)
    assert asyncio.run(j.ask("q")) == ("", 0, 0)
    assert len(calls) == 3 and j.timeouts == 1 and j.usd == 0
    j, calls = make([timeout, (502, {}), (200, OK)], monkeypatch)  # the streak resets on another error
    assert asyncio.run(j.ask("q")) == ("YES", 5, 1) and j.timeouts == 0
