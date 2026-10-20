"""Judge price tables, USD per million tokens. Cost accounting uses the table for the (backend, model) actually in use; cached prefill is recorded but
never credited (upper bound). Unlisted models fall back to a conservative upper bound marked UNVERIFIED, which the dry-run output and summary repeat."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Price:
    prefill: float
    cached: float
    sample: float
    verified: bool
    source: str

    def meta(self):
        return {**asdict(self), "status": "verified" if self.verified else "UNVERIFIED"}


_TINKER_SRC = "https://thinkingmachines.ai/tinker/pricing/ , fetched 2026-10-20 (short-context columns: prefill, cached prefill, sample)"
_OR_SRC = "https://openrouter.ai/api/v1/models (public list, no key), fetched 2026-10-20; listed per-token price, actual provider routing may differ"

TINKER = {
    "openai/gpt-oss-20b": Price(0.18, 0.036, 0.45, True, _TINKER_SRC),
    "openai/gpt-oss-120b": Price(0.33, 0.066, 0.84, True, _TINKER_SRC),
    # the page lists "DeepSeek-V4.1-Flash" and "DeepSeek-V4.1-Flash-1017" at the same 0.60 / 0.12 / 3.60
    "deepseek-ai/DeepSeek-V4.1-Flash": Price(0.60, 0.12, 3.60, True, _TINKER_SRC),
}
TINKER_UPPER = Price(1.695, 1.695, 4.215, False, "UNVERIFIED upper bound: highest prefill and sample rate on the Tinker page for the models checked (DeepSeek-V3.1), no cache credit")

OPENROUTER = {
    "deepseek/deepseek-v4.1-flash": Price(0.02832, 0.000566, 0.12, True, _OR_SRC),
}
OPENROUTER_UPPER = Price(3.0, 3.0, 15.0, False, "UNVERIFIED upper bound for an OpenRouter model not in this table; set a real entry in pricing.py")

# default judge model per backend; None means --judge-model is required
DEFAULT_MODEL = {
    "tinker": "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144",  # the bare name is refused by Tinker
    # OpenRouter's entry "DeepSeek: DeepSeek V4.1 Flash" (the model list also has "-1017", ":batch", "~...-latest" aliases, deliberately not chosen).
    # Matching by name only: that it serves the same weights as Tinker's checkpoint is not verified.
    "openrouter": "deepseek/deepseek-v4.1-flash",
}


def price_for(backend, model):
    if backend == "tinker":
        return TINKER.get(model.split(":peft:")[0], TINKER_UPPER)
    if backend == "openrouter":
        return OPENROUTER.get(model, OPENROUTER_UPPER)
    raise ValueError(f"unknown backend {backend!r}")
