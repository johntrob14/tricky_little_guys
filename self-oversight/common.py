"""Shared setup for the self-oversight organism on Tinker: API key, candidate models, prices, renderers."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Model:
    tinker_id: str
    renderer: str
    # USD per million tokens, short-context tier, from https://thinkingmachines.ai/tinker/pricing/
    # (fetched 2026-10-18; page dated 2026-10-14). Discounted prices marked limited-time are listed at
    # the discounted rate with list_multiplier=2 so budgets can be set at list price.
    prefill: float
    cached_prefill: float
    sample: float
    train: float
    list_multiplier: float = 1.0


MODELS = {
    "gpt-oss-20b": Model("openai/gpt-oss-20b", "gpt_oss_medium_reasoning", 0.18, 0.036, 0.45, 0.396),
    "gpt-oss-120b": Model("openai/gpt-oss-120b", "gpt_oss_medium_reasoning", 0.33, 0.066, 0.84, 0.737),
    "qwen3.6-35b": Model("Qwen/Qwen3.6-35B-A3B", "qwen3_5", 0.45, 0.09, 1.335, 1.177),
    "nemotron-3.5-lightning": Model("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "nemotron3_ultra",
                                    0.184, 0.0368, 0.495, 0.44, list_multiplier=2.0),
    "nemotron-3-super": Model("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "nemotron3",
                              0.57, 0.114, 1.44, 1.276, list_multiplier=2.0),
    "inkling-small": Model("thinkingmachines/Inkling-Small", "tml_v0", 0.58, 0.116, 1.44, 1.73,
                           list_multiplier=2.0),
}


def load_key() -> None:
    if os.environ.get("TINKER_API_KEY"):
        return
    env = Path.home() / ".config/sauce/tinker.env"
    for line in env.read_text().splitlines():
        if line.startswith("TINKER_API_KEY="):
            os.environ["TINKER_API_KEY"] = line.split("=", 1)[1].strip()
            return
    sys.exit(f"No TINKER_API_KEY in the environment or {env}")


def get_model(name: str) -> Model:
    if name not in MODELS:
        sys.exit(f"unknown model {name!r}; known: {', '.join(MODELS)}")
    return MODELS[name]


def renderer_for(model: Model):
    """(tokenizer, renderer) for a candidate, checked against the cookbook's recommended renderers."""
    from tinker_cookbook.model_info import get_recommended_renderer_names
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    ok = get_recommended_renderer_names(model.tinker_id)
    assert model.renderer in ok, f"{model.renderer} not recommended for {model.tinker_id}: {ok}"
    tok = get_tokenizer(model.tinker_id)
    return tok, get_renderer(model.renderer, tok)
