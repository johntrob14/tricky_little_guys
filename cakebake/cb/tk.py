"""Tinker judge client (self-oversight's sample.Runner: retries, concurrency cap, USD budget).
Run with the self-oversight venv: uv run --project ~/code/tricky_little_guys/self-oversight python -m cb.<script>
Prices: thinkingmachines.ai/tinker/pricing (2026-10-14), USD per M tokens: prefill, cached prefill, sample, train."""
from __future__ import annotations

import os
import sys

SO = os.path.expanduser("~/code/tricky_little_guys/self-oversight")
if SO not in sys.path:
    sys.path.insert(0, SO)

from common import MODELS, Model  # noqa: E402
from sample import Runner  # noqa: E402

MODELS.setdefault("kimi-k2.6", Model("moonshotai/Kimi-K2.6", "kimi_k26", 2.205, 0.441, 5.49, 4.84))
MODELS.setdefault("kimi-k2.6-nothink", Model("moonshotai/Kimi-K2.6", "kimi_k26_disable_thinking", 2.205, 0.441, 5.49, 4.84))
MODELS.setdefault("deepseek-v3.1", Model("deepseek-ai/DeepSeek-V3.1", "deepseekv3", 1.695, 0.339, 4.215, 3.718))
MODELS.setdefault("nemotron-super-nothink", Model("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "nemotron3_disable_thinking",
                                                  0.57, 0.114, 1.44, 1.276, list_multiplier=2.0))
MODELS.setdefault("qwen3.5-397b-nothink", Model("Qwen/Qwen3.5-397B-A17B", "qwen3_5_disable_thinking", 1.65, 0.33, 7.5, 6.6))


def runner(name: str, budget_usd: float, max_concurrency: int = 48, temperature: float = 0.0) -> Runner:
    return Runner(MODELS[name], budget_usd, backend="tinker", max_concurrency=max_concurrency, temperature=temperature)


async def chat(r: Runner, messages: list[dict], max_tokens: int = 2000) -> str:
    out = await r.sample(messages, n=1, max_tokens=max_tokens)
    return out[0].answer
