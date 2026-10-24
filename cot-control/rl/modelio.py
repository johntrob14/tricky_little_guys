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
    # From the official page as fetched 2026-10-23 (64K tier; "as of October 19, 2026").
    "Qwen/Qwen3.5-9B": {"prefill": 0.30, "sample": 0.807, "train": 1.463},
    "Qwen/Qwen3.5-4B": {"prefill": 0.33, "sample": 1.005, "train": 0.737},
    "Qwen/Qwen3-8B": {"prefill": 0.195, "sample": 0.60, "train": 0.44},
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": {"prefill": 0.195, "sample": 0.495, "train": 0.44},
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": {"prefill": 0.184, "sample": 0.495, "train": 0.44},
    # Fetched 2026-10-23; the page lists only a 256K tier for it.
    "deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144": {"prefill": 0.60, "sample": 3.60, "train": 4.98},
}
DS_EFFORT = "Reasoning Effort: {} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"


def probe_harmony(tok) -> bool:
    if not getattr(tok, "chat_template", None):  # DeepSeek V4.1's Tinker tokenizer has none (render_ds writes its format); gpt-oss's has one
        return False
    r = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
    return r.endswith("<|start|>assistant")


def policy_stop(tok, harmony: bool) -> list[int]:
    return GPT_OSS_STOP if harmony else [tok.eos_token_id]


# Qwen3.8 reasoning-effort system prompts, verbatim from tinker_cookbook's qwen3_8_*_reasoning renderers.
# "medium" adds no system message here, and render() passes no reasoning_effort to the template, whose default ("xhigh") injects the
# "Reasoning effort is set to xhigh. Please think carefully..." system prompt: effort "medium" therefore renders the xhigh prompt (what the
# Qwen GRPO/OPSD runs trained under); only reasoning_effort="medium" passed to the template renders no system prompt. gpt-oss sets effort via its template.
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


def render_ds(user: str, system: str | None = None, effort: int = 75, thinking: bool = True) -> str:
    """DeepSeek V4.1 prompt text, per encoding/encoding.py of deepseek-ai/DeepSeek-V4.1-Flash-1017 (Tinker's tokenizer for it has no chat template).
    Thinking mode opens a <｜System｜> block with the effort prefix, then `system`, and opens <think>; chat mode renders only `system` there and closes
    the think block at once. Effort 75 is the format default ("high")."""
    head = DS_EFFORT.format(effort) if thinking else ""
    block = "<｜System｜>" + head + (system or "") if head or system else ""
    return f"<｜begin▁of▁sentence｜>{block}<｜User｜>{user}<｜Assistant｜>" + ("<think>" if thinking else "</think>")


def to_think(gen: str, harmony: bool, think_prefix: str, eos: str) -> str:
    if harmony:
        return extract.harmony_to_think(gen)
    return think_prefix + (gen[:-len(eos)] if eos and gen.endswith(eos) else gen)
