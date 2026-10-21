"""Render prompts and extract CoTs for either a harmony model (gpt-oss) or a think-tag model (Qwen etc.).

A harmony model's rendered prompt ends with '<|start|>assistant' and its decoded generation is parsed by
extract.harmony_to_think. A think-tag model's chat template opens the think block itself (the prompt ends with
'<think>'), so the decoded generation already sits inside <think>; we prepend that opener and strip the trailing
EOS before extract.split_output. Mirrors tinker-runner/tinker_cotcontrol.py.
"""

from cotswitch import extract

GPT_OSS_STOP = [200002, 200012]  # <|return|>, <|call|>

# USD/M tokens, Tinker short-context (64K) tier (thinkingmachines.ai/tinker/pricing, fetched 2026-10-19)
PRICES = {
    "Qwen/Qwen3.8-27B": {"prefill": 0.6975, "sample": 5.595, "train": 4.103},
    "openai/gpt-oss-120b": {"prefill": 0.33, "sample": 0.84, "train": 0.737},
}


def probe_harmony(tok) -> bool:
    r = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
    return r.endswith("<|start|>assistant")


def policy_stop(tok, harmony: bool) -> list[int]:
    return GPT_OSS_STOP if harmony else [tok.eos_token_id]


# Qwen3.8 reasoning-effort system prompts, verbatim from tinker_cookbook's qwen3_8_*_reasoning renderers.
# "medium" injects nothing (the template default). gpt-oss sets effort via its template, not a system message.
EFFORT_SYS = {
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
}


def render(tok, user: str, harmony: bool, effort: str = "medium", system: str | None = None) -> tuple[list[int], str]:
    """(prompt token ids, think_prefix); think_prefix is "" for harmony, else the '<think>' the template opens. `system` is an extra system message."""
    msgs = [{"role": "user", "content": user}]
    parts = ([EFFORT_SYS[effort]] if not harmony and effort in EFFORT_SYS else []) + ([system] if system else [])
    if parts:
        msgs = [{"role": "system", "content": "\n\n".join(parts)}] + msgs
    r = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(r, add_special_tokens=False), ("" if harmony else r[r.rfind("<think>"):])


def to_think(gen: str, harmony: bool, think_prefix: str, eos: str) -> str:
    if harmony:
        return extract.harmony_to_think(gen)
    return think_prefix + (gen[:-len(eos)] if eos and gen.endswith(eos) else gen)
