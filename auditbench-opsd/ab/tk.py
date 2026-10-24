"""Tinker judge / simulated-user client for the AuditBench-style evals.

Reuses self-oversight's sample.Runner (retries, concurrency cap, USD budget). Run with the self-oversight venv:
    uv run --project ~/code/tricky_little_guys/self-oversight python -m ab.<script>
Kimi-K2.6 replaces AuditBench's claude-opus-4-8 (no Anthropic key on zeus; deviation recorded in notes/PLAN.md).
Prices: thinkingmachines.ai/tinker/pricing, 2026-10-14 (32K tier): prefill 2.205, cached 0.441, sample 5.49 USD/M.
"""
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


def runner(name: str = "kimi-k2.6", budget_usd: float = 5.0, max_concurrency: int = 32, temperature: float = 1.0) -> Runner:
    return Runner(MODELS[name], budget_usd, max_concurrency=max_concurrency, temperature=temperature)


async def chat(r: Runner, messages: list[dict], max_tokens: int = 4000) -> str:
    out = await r.sample(messages, n=1, max_tokens=max_tokens)
    return out[0].answer
