"""Tinker sampling with the same `chat()` contract as openrouter.Client, for tinker:// checkpoints and Tinker base models.

Selected by `--provider Tinker`; `--model` is a tinker:// sampler path or a Tinker base-model id. The prompt is built
by the tinker-cookbook renderer of the model's family for the requested effort (`gpt_oss_<effort>_reasoning` with the
standard gpt-oss system prompt; `qwen3_8_<effort>_reasoning` for Qwen3.8-27B, whose xhigh renderer injects the template's
default reasoning-effort system prompt, the one the Qwen organisms trained under; Nemotron 3 Super, Nemotron 3.5 Lightning and
Inkling-Small use their fixed recommended renderer, `--effort default`); the completion is split by the same
renderer into thinking (`reasoning`) and text (`answer`). The family of a tinker:// path is looked up in CHECKPOINT_BASES by
a name fragment (default gpt-oss-120b). `finish` is "stop" for a clean termination, else the renderer's termination
("malformed": hit max_tokens mid-thought). Cost is the family's Tinker short-context list price. Tools are not supported here. Transient Tinker errors are retried with exponential backoff
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

# Base model per family: Tinker id -> (renderer name pattern, prices in USD per million tokens, thinkingmachines.ai/tinker/pricing 64K tier, 2026-10-14/19).
FAMILIES = {
    "openai/gpt-oss-120b": ("gpt_oss_{effort}_reasoning", {"prefill": 0.33, "cached_prefill": 0.066, "sample": 0.84}),
    "Qwen/Qwen3.8-27B": ("qwen3_8_{effort}_reasoning", {"prefill": 0.6975, "cached_prefill": 0.1395, "sample": 5.595}),
    # No effort placeholder: the cookbook's recommended renderer (full reasoning for Nemotron; Inkling's default effort 0.9);
    # run with --effort default. Prices are the 50%-off list prices of 2026-10-19.
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": ("nemotron3", {"prefill": 0.57, "cached_prefill": 0.114, "sample": 1.44}),
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": ("nemotron3_ultra", {"prefill": 0.184, "cached_prefill": 0.0368, "sample": 0.495}),
    "thinkingmachines/Inkling-Small": ("tml_v0", {"prefill": 0.58, "cached_prefill": 0.116, "sample": 1.44}),
}
CHECKPOINT_BASES = {"cotctl-opsd-grpo": "Qwen/Qwen3.8-27B", "cotctl-inksft": "thinkingmachines/Inkling-Small", "cotctl-inkrl": "thinkingmachines/Inkling-Small"}   # sampler-path name fragment -> base model; tinker:// paths not listed are gpt-oss-120b


def base_model_of(model: str) -> str:
    if not model.startswith("tinker://"):
        return model
    return next((b for frag, b in CHECKPOINT_BASES.items() if frag in model), "openai/gpt-oss-120b")


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

    def renderer(self, base: str, effort: str):
        if (base, effort) not in self.renderers:
            self.renderers[(base, effort)] = get_renderer(FAMILIES[base][0].format(effort=effort), get_tokenizer(base))
        return self.renderers[(base, effort)]

    async def chat(self, messages: list[dict], model: str, provider: str | None = None, effort: str | None = "medium",
                   temperature: float = 1.0, max_tokens: int = 16000, tools: list[dict] | None = None) -> dict:
        assert tools is None, "tools are not supported on the Tinker path"
        base = base_model_of(model)
        sampler, renderer, price = self.sampler(model), self.renderer(base, effort), FAMILIES[base][1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Inkling: a numeric --effort is TMLv0's thinking effort (the system 'Thinking effort level' message); "default" is the renderer's 0.9
            prompt = renderer.build_generation_prompt(messages, effort=float(effort)) if base == "thinkingmachines/Inkling-Small" and effort != "default" else renderer.build_generation_prompt(messages)
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
        cost = ((prompt.length - cached) * price["prefill"] + cached * price["cached_prefill"] + len(toks) * price["sample"]) / 1e6
        self.usd += cost
        self.calls += 1
        return {"reasoning": reasoning, "answer": answer, "tool_calls": None, "reasoning_details": None,
                "finish": "stop" if str(seq.stop_reason) == "stop" and termination != "malformed" else termination, "provider": "Tinker",
                "prompt_tokens": prompt.length, "completion_tokens": len(toks), "cost": cost}


def sampling_client(provider: str | None, concurrency: int):
    """The sampler for a tag's provider: TinkerClient for "Tinker", else openrouter.Client."""
    return TinkerClient(concurrency) if provider == "Tinker" else Client(concurrency)
