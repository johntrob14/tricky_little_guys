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
_TINKER_SRC_1021 = "https://thinkingmachines.ai/tinker/pricing/ , fetched 2026-10-21 (page 'as of October 19, 2026'; prefill, cached prefill, sample)"
# Held-out candidates: the max over every endpoint on https://openrouter.ai/<model>/providers (page HTML, no API call), fetched 2026-10-21. The judge does not
# pin a provider, so an unpinned request may land on any endpoint; the max is the routing upper bound (base list prices in HELDOUT.md).
_OR_MAX_SRC = "https://openrouter.ai/{}/providers page HTML, fetched 2026-10-21; max over listed endpoints (unpinned routing upper bound)"

TINKER = {
    "openai/gpt-oss-20b": Price(0.18, 0.036, 0.45, True, _TINKER_SRC),
    "openai/gpt-oss-120b": Price(0.33, 0.066, 0.84, True, _TINKER_SRC),
    # the page lists "DeepSeek-V4.1-Flash" and "DeepSeek-V4.1-Flash-1017" at the same 0.60 / 0.12 / 3.60
    "deepseek-ai/DeepSeek-V4.1-Flash": Price(0.60, 0.12, 3.60, True, _TINKER_SRC),
    # held-out judge candidates (HELDOUT.md); prices verified, page re-read 2026-10-21 ("as of October 19, 2026"). The page gives bare names: the org
    # prefixes are the HF ids, UNVERIFIED as Tinker ids (a wrong key just falls back to TINKER_UPPER). Tinker judge path untested for their templates.
    "zai-org/GLM-5.3": Price(3.78, 0.756, 12.15, True, _TINKER_SRC_1021),
    "moonshotai/Kimi-K2.6": Price(2.205, 0.441, 5.49, True, _TINKER_SRC_1021),
}
# highest prefill (GLM-5.3 3.78) and highest sample (GLM-5.3 12.15) rate on the whole Tinker page, 2026-10-21; was 1.695 / 4.215 (DeepSeek-V3.1), which
# GLM-5.3 and Nemotron-3-Ultra (sample 6.225) now exceed
TINKER_UPPER = Price(3.78, 3.78, 12.15, False, "UNVERIFIED upper bound: highest prefill and sample rate on the Tinker page, 2026-10-21 (GLM-5.3), no cache credit")

OPENROUTER = {
    "openai/gpt-oss-20b": Price(0.018, 0.009, 0.09, True, _OR_SRC),
    "deepseek/deepseek-v4.1-flash": Price(0.02832, 0.000566, 0.12, True, _OR_SRC),
    # held-out judge candidates (HELDOUT.md): (prefill, cache read, sample) of the priciest endpoint
    "anthropic/claude-sonnet-5.5": Price(2.2, 0.11, 11.0, True, _OR_MAX_SRC.format("anthropic/claude-sonnet-5.5")),  # base 2 / 10; +10% = regional endpoints
    "google/gemini-3.8-flash": Price(1.35, 1.35, 6.75, True, _OR_MAX_SRC.format("google/gemini-3.8-flash")),  # priority tier; base 0.75 / 3.75 is a 50%-off promo; no cache price listed
    "z-ai/glm-5.3": Price(2.8, 0.56, 8.8, True, _OR_MAX_SRC.format("z-ai/glm-5.3")),  # alibaba/prime; z-ai's own endpoint 1.4 / 4.4
    "moonshotai/kimi-k3": Price(4.5, 1.2, 22.5, True, _OR_MAX_SRC.format("moonshotai/kimi-k3")),  # fast endpoints (cache read max 1.2, akashml); moonshotai's own 3 / 15
    "anthropic/claude-haiku-5.5": Price(0.11, 0.011, 0.55, True, _OR_MAX_SRC.format("anthropic/claude-haiku-5.5")),  # base 0.10 / 0.50
    "openai/gpt-6-luna": Price(0.2, 0.02, 1.0, True, _OR_MAX_SRC.format("openai/gpt-6-luna")),  # openai/fast; base 0.10 / 0.50
    "openai/gpt-oss-120b": Price(0.35, 0.35, 0.95, True, _OR_MAX_SRC.format("openai/gpt-oss-120b")),  # cerebras 0.35 prefill, sambanova 0.95 sample; cheapest 0.03 / 0.15
}
OPENROUTER_UPPER = Price(3.0, 3.0, 15.0, False, "UNVERIFIED upper bound for an OpenRouter model not in this table; set a real entry in pricing.py")

# default judge model per backend; None means --judge-model is required
DEFAULT_MODEL = {
    "tinker": "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144",  # the bare name is refused by Tinker
    # OpenRouter's entry "DeepSeek: DeepSeek V4.1 Flash" (the model list also has "-1017", ":batch", "~...-latest" aliases, deliberately not chosen).
    # Matching by name only: that it serves the same weights as Tinker's checkpoint is not verified.
    "openrouter": "openai/gpt-oss-20b",
}
DEFAULT_BACKEND = "openrouter"


def price_for(backend, model):
    if backend == "tinker":
        return TINKER.get(model.split(":peft:")[0], TINKER_UPPER)
    if backend == "openrouter":
        return OPENROUTER.get(model, OPENROUTER_UPPER)
    raise ValueError(f"unknown backend {backend!r}")
