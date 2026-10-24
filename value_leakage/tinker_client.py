"""Tinker sampling with the same `chat()` contract as openrouter.Client, for tinker:// checkpoints and Tinker base models.

Selected by `--provider Tinker`; `--model` is a tinker:// sampler path or a Tinker base-model id. The prompt is built
by the tinker-cookbook Harmony renderer for the requested effort (`gpt_oss_<effort>_reasoning`), whose system prompt is
the standard gpt-oss one (knowledge cutoff, current date, reasoning level); the completion is split by the same
renderer into the analysis channel (`reasoning`) and the final channel (`answer`). `finish` is "stop" for a clean
termination, else the renderer's termination ("malformed": hit max_tokens mid-thought). Cost is Tinker's gpt-oss-120b
short-context list price. Tools are not supported here. Transient Tinker errors are retried with exponential backoff
(8 attempts); anything else raises.
"""
import asyncio
import os
import warnings
from pathlib import Path

import tinker
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from openrouter import Client

BASE_MODEL = "openai/gpt-oss-120b"
PRICE = {"prefill": 0.33, "cached_prefill": 0.066, "sample": 0.84}   # USD per million tokens, thinkingmachines.ai/tinker/pricing (2026-10-14)
RETRY = (tinker.APIConnectionError, tinker.APITimeoutError, tinker.RateLimitError, tinker.InternalServerError, tinker.RequestFailedError, asyncio.TimeoutError)


def api_key() -> str:
    if key := os.environ.get("TINKER_API_KEY"):
        return key
    for line in (Path.home() / ".env").read_text().splitlines():
        if line.startswith("TINKER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise KeyError("TINKER_API_KEY is not in the environment or ~/.env")


class TinkerClient:
    def __init__(self, concurrency: int = 32):
        os.environ["TINKER_API_KEY"] = api_key()
        self.sem = asyncio.Semaphore(concurrency)
        self.usd, self.calls, self.retries = 0.0, 0, 0
        self.service, self.samplers, self.renderers = tinker.ServiceClient(), {}, {}

    def sampler(self, model: str):
        if model not in self.samplers:
            self.samplers[model] = self.service.create_sampling_client(model_path=model) if model.startswith("tinker://") else self.service.create_sampling_client(base_model=model)
        return self.samplers[model]

    def renderer(self, effort: str):
        if effort not in self.renderers:
            self.renderers[effort] = get_renderer(f"gpt_oss_{effort}_reasoning", get_tokenizer(BASE_MODEL))
        return self.renderers[effort]

    async def chat(self, messages: list[dict], model: str, provider: str | None = None, effort: str | None = "medium",
                   temperature: float = 1.0, max_tokens: int = 16000, tools: list[dict] | None = None) -> dict:
        assert tools is None, "tools are not supported on the Tinker path"
        sampler, renderer = self.sampler(model), self.renderer(effort)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            prompt = renderer.build_generation_prompt(messages)
        params = tinker.SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=renderer.get_stop_sequences())
        for attempt in range(8):
            async with self.sem:
                try:
                    resp = await sampler.sample_async(prompt, 1, params)
                    break
                except RETRY as e:
                    err = e
            self.retries += 1
            await asyncio.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"tinker: out of retries: {err!r}")
        seq = resp.sequences[0]
        toks = list(seq.tokens)
        msg, term = renderer.parse_response(toks)
        termination = str(getattr(term, "value", term))
        content = msg.get("content", "")
        parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
        reasoning = "".join(p.get("thinking", "") for p in parts if p.get("type") == "thinking")
        answer = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        cached = getattr(resp, "prompt_cache_hit_tokens", 0) or 0
        cost = ((prompt.length - cached) * PRICE["prefill"] + cached * PRICE["cached_prefill"] + len(toks) * PRICE["sample"]) / 1e6
        self.usd += cost
        self.calls += 1
        return {"reasoning": reasoning, "answer": answer, "tool_calls": None, "reasoning_details": None,
                "finish": "stop" if str(seq.stop_reason) == "stop" and termination != "malformed" else termination, "provider": "Tinker",
                "prompt_tokens": prompt.length, "completion_tokens": len(toks), "cost": cost}


def sampling_client(provider: str | None, concurrency: int):
    """The sampler for a tag's provider: TinkerClient for "Tinker", else openrouter.Client."""
    return TinkerClient(concurrency) if provider == "Tinker" else Client(concurrency)
