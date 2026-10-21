"""Offline tests for the sampling runner: parsing, budget, retries. No Tinker or OpenRouter calls (fake sampler, mock transport)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import httpx
import pytest
import tinker

from common import MODELS, Model
from sample import Budget, BudgetExceeded, OpenRouterError, Rollout, Runner, split_thinking


class FakeTokenizer:
    def decode(self, toks):
        return "".join(chr(t) for t in toks)


class FakeRenderer:
    """Encodes text as code points; `parse_response` splits on a trailing 0 stop token."""
    tokenizer = FakeTokenizer()

    def build_generation_prompt(self, messages, role="assistant", prefill=None):
        text = "|".join(f"{m['role']}:{m['content'] if isinstance(m['content'], str) else 'parts'}" for m in messages)
        return tinker.ModelInput.from_ints([ord(c) for c in text])

    def get_stop_sequences(self):
        return [0]

    def parse_response(self, toks):
        from tinker_cookbook.renderers import ParseTermination

        if toks and toks[-1] == 0:
            text = self.tokenizer.decode(toks[:-1])
            return {"role": "assistant", "content": [{"type": "thinking", "thinking": "t:" + text},
                                                     {"type": "text", "text": "a:" + text}]}, ParseTermination.STOP_SEQUENCE
        return {"role": "assistant", "content": self.tokenizer.decode(toks)}, ParseTermination.MALFORMED


@dataclass
class FakeSeq:
    tokens: list
    stop_reason: str


@dataclass
class FakeResp:
    sequences: list
    prompt_cache_hit_tokens: int = 0


class FakeSampler:
    def __init__(self, replies, fail_first=0, exc=None):
        self.replies, self.fail_first, self.exc, self.calls = replies, fail_first, exc, 0

    async def sample_async(self, prompt, n, params):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise self.exc
        seqs = []
        for i in range(n):
            text, stop = self.replies[i % len(self.replies)]
            seqs.append(FakeSeq([ord(c) for c in text] + ([0] if stop else []), "stop" if stop else "length"))
        return FakeResp(seqs, prompt_cache_hit_tokens=0)


MODEL = MODELS["gpt-oss-20b"]


def make(replies, budget=1.0, **kw) -> Runner:
    return Runner(MODEL, budget, backend="tinker", sampler=FakeSampler(replies, **kw), renderer=FakeRenderer(), log=None,
                  backoff_base=0.01, backoff_max=0.02)


def test_split_thinking_parts_and_text():
    assert split_thinking([{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "42"}]) == ("hm", "42")
    assert split_thinking("<think>deep</think>\n42") == ("deep", "42")
    assert split_thinking("<think>cut off mid") == ("cut off mid", "")
    assert split_thinking("<|channel|>analysis<|message|>hm<|end|><|start|>assistant<|channel|>final<|message|>42") == ("hm", "42")
    assert split_thinking("<|channel|>analysis<|message|>still thinking") == ("still thinking", "")
    assert split_thinking("plain") == ("", "plain")


def test_sample_parses_and_prices():
    r = make([("hello", True), ("trunc", False)])
    outs = asyncio.run(r.sample([{"role": "user", "content": "hi"}], n=2, max_tokens=10))
    assert [o.clean for o in outs] == [True, False]
    assert outs[0].thinking == "t:hello" and outs[0].answer == "a:hello"
    assert outs[1].termination == "malformed" and outs[1].answer == "trunc" and outs[1].stop_reason == "length"
    n_prompt = len("user:hi")
    expected = (n_prompt * MODEL.prefill + (6 + 5) * MODEL.sample) / 1e6
    assert r.budget.spent_usd == pytest.approx(expected)
    assert sum(o.cost_usd for o in outs) == pytest.approx(expected)
    assert r.budget.reserved_usd == 0 and r.budget.requests == 1


def test_budget_blocks_before_sending():
    r = make([("x", True)], budget=1e-9)
    with pytest.raises(BudgetExceeded):
        asyncio.run(r.sample([{"role": "user", "content": "hi"}], n=1, max_tokens=1000))
    assert r.sampler.calls == 0 and r.budget.reserved_usd == 0


def test_budget_counts_in_flight():
    b = Budget(cap_usd=1.0)
    b.reserve(0.6)
    with pytest.raises(BudgetExceeded):
        b.reserve(0.5)
    b.release(0.6)
    b.commit(0.1, 1, 0, 1)
    b.reserve(0.5)


def test_retries_transient_then_succeeds():
    r = make([("ok", True)], fail_first=2, exc=tinker.APIConnectionError(request=None))
    outs = asyncio.run(r.sample([{"role": "user", "content": "hi"}], n=1, max_tokens=10))
    assert outs[0].answer == "a:ok" and outs[0].attempts == 3 and r.sampler.calls == 3
    assert len(r.errors) == 2 and all(e["retryable"] for e in r.errors)


def test_gives_up_after_max_retries():
    r = make([("ok", True)], fail_first=99, exc=tinker.APIConnectionError(request=None))
    r.max_retries = 2
    with pytest.raises(tinker.APIConnectionError):
        asyncio.run(r.sample([{"role": "user", "content": "hi"}], n=1, max_tokens=10))
    assert r.sampler.calls == 3 and r.budget.reserved_usd == 0 and r.budget.spent_usd == 0


def test_non_retryable_raises_immediately():
    r = make([("ok", True)], fail_first=1, exc=ValueError("bad"))
    with pytest.raises(ValueError):
        asyncio.run(r.sample([{"role": "user", "content": "hi"}], n=1, max_tokens=10))
    assert r.sampler.calls == 1


def test_continue_appends_parsed_message():
    r = make([("first", True)])
    msgs = [{"role": "user", "content": "hi"}]
    out = asyncio.run(r.sample(msgs, n=1, max_tokens=10))[0]
    assert isinstance(out, Rollout)
    seen = {}
    orig = r.renderer.build_generation_prompt

    def spy(messages, role="assistant", prefill=None):
        seen["messages"] = messages
        return orig(messages, role, prefill)

    r.renderer.build_generation_prompt = spy
    asyncio.run(r.continue_(msgs, out, "why?"))
    assert [m["role"] for m in seen["messages"]] == ["user", "assistant", "user"]
    assert seen["messages"][1] == out.message and seen["messages"][2]["content"] == "why?"


def test_list_price_multiplier():
    m = MODELS["nemotron-3.5-lightning"]
    a = Runner(m, 1.0, backend="tinker", sampler=object(), renderer=object(), log=None)
    b = Runner(m, 1.0, backend="tinker", sampler=object(), renderer=object(), log=None, at_list_price=False)
    assert a.cost(1000, 0, 1000) == pytest.approx(2 * b.cost(1000, 0, 1000))


# -- OpenRouter backend ---------------------------------------------------------------------------

USER = [{"role": "user", "content": "hi"}]


def or_reply(content, reasoning="", finish="stop", cost=0.001, prompt=10, completion=5, cached=None):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion, "cost": cost}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"choices": [{"message": {"role": "assistant", "content": content, "reasoning": reasoning}, "finish_reason": finish}], "usage": usage}


def or_runner(handler, budget=1.0, **kw) -> Runner:
    return Runner(MODEL, budget, transport=httpx.MockTransport(handler), api_key="test", log=None, backoff_base=0.01, backoff_max=0.02, **kw)


def test_backend_defaults():
    assert Runner(MODEL, 1.0, api_key="test").backend == "openrouter"
    assert Runner(MODEL, 1.0, model_path="tinker://x", sampler=object(), renderer=object()).backend == "tinker"
    with pytest.raises(ValueError):
        Runner(MODEL, 1.0, backend="openrouter", model_path="tinker://x")
    with pytest.raises(ValueError):
        Runner(Model("x/y", "qwen3_5", 1, 1, 1, 1), 1.0)   # no openrouter_id
    with pytest.raises(ValueError):
        Runner(MODEL, 1.0, backend="vllm")


def test_openrouter_request_parse_and_cost():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test"
        return httpx.Response(200, json=or_reply("42", reasoning="hm", cost=0.0005, cached=3))

    r = or_runner(handler)
    outs = asyncio.run(r.sample(USER, n=2, max_tokens=10, seed=7))
    assert [o.answer for o in outs] == ["42", "42"] and outs[0].thinking == "hm" and outs[0].clean and outs[0].raw == "42"
    assert outs[0].stop_reason == "stop" and outs[0].tokens == [] and outs[0].message == {"role": "assistant", "content": "42"}
    assert (outs[0].prompt_tokens, outs[0].cached_prompt_tokens, outs[0].completion_tokens, outs[0].cost_usd) == (10, 3, 5, 0.0005)
    assert r.budget.spent_usd == pytest.approx(0.001) and r.budget.requests == 2 and r.budget.reserved_usd == 0
    assert r.budget.prompt_tokens == 20 and r.budget.completion_tokens == 10
    assert len(seen) == 2 and sorted(s["seed"] for s in seen) == [7, 8]
    body = seen[0]
    assert body["model"] == "openai/gpt-oss-20b" and body["messages"] == USER and body["max_tokens"] == 10
    assert body["reasoning"] == {"effort": "medium"} and body["usage"] == {"include": True}
    assert body["temperature"] == 1.0 and body["top_p"] == 1.0


def test_openrouter_truncated_inline_think_and_no_seed():
    def handler(request):
        assert "seed" not in json.loads(request.content)
        return httpx.Response(200, json=or_reply("<think>cut off", finish="length"))

    out = asyncio.run(or_runner(handler).sample(USER, n=1, max_tokens=10))[0]
    assert not out.clean and out.stop_reason == "length" and out.termination == "length"
    assert out.thinking == "cut off" and out.answer == ""


def test_openrouter_reservation_uses_table_price_and_budget_blocks():
    r = or_runner(lambda req: httpx.Response(200, json=or_reply("x")))
    assert r.cost(1000, 0, 1000) == pytest.approx((1000 * MODEL.or_prompt + 1000 * MODEL.or_completion) / 1e6)
    assert r.prompt_length(USER) == len(json.dumps(USER[0])) // 3
    tight = or_runner(lambda req: httpx.Response(200, json=or_reply("x")), budget=1e-9)
    with pytest.raises(BudgetExceeded):
        asyncio.run(tight.sample(USER, n=1, max_tokens=1000))
    assert tight.budget.reserved_usd == 0


def test_openrouter_retries_429_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, text="slow down") if len(calls) < 3 else httpx.Response(200, json=or_reply("ok"))

    r = or_runner(handler)
    out = asyncio.run(r.sample(USER, n=1, max_tokens=10))[0]
    assert out.answer == "ok" and out.attempts == 3 and len(calls) == 3
    assert len(r.errors) == 2 and all(e["retryable"] for e in r.errors)


def test_openrouter_bad_request_and_error_body_raise():
    r = or_runner(lambda req: httpx.Response(400, text="no such model"))
    with pytest.raises(OpenRouterError, match="HTTP 400"):
        asyncio.run(r.sample(USER, n=1, max_tokens=10))
    assert r.budget.spent_usd == 0 and r.budget.reserved_usd == 0 and len(r.errors) == 1 and not r.errors[0]["retryable"]
    r = or_runner(lambda req: httpx.Response(200, json={"error": {"message": "provider down"}}))
    with pytest.raises(OpenRouterError, match="provider down"):
        asyncio.run(r.sample(USER, n=1, max_tokens=10))


def test_openrouter_rejects_prefill_and_continues_with_answer_only():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["messages"])
        return httpx.Response(200, json=or_reply("first", reasoning="secret"))

    r = or_runner(handler)
    with pytest.raises(ValueError):
        asyncio.run(r.sample(USER, prefill="x"))
    out = asyncio.run(r.sample(USER, n=1, max_tokens=10))[0]
    asyncio.run(r.continue_(USER, out, "why?"))
    assert seen[1] == USER + [{"role": "assistant", "content": "first"}, {"role": "user", "content": "why?"}]
