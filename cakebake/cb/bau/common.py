"""Shared helpers for cb.bau (no heavy imports here: used on the pod, on zeus, and in the CPU smoke test)."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROMPTS = os.path.join(HERE, "prompts")
DATA = os.path.join(ROOT, "data")


def out_dir(name: str, root: str | None = None) -> str:
    d = os.path.join(root or ROOT, "results", "bau", name)
    os.makedirs(d, exist_ok=True)
    return d


def prompt(fn: str) -> str:
    return open(os.path.join(PROMPTS, fn), encoding="utf-8").read()


def read_jsonl(p: str) -> list[dict]:
    """Tolerates a truncated final line (crash mid-write)."""
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def append_jsonl(p: str, row: dict) -> None:
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chat_ids(tok, messages: list[dict]) -> list[int]:
    """tokenizer.apply_chat_template(messages, add_generation_prompt=True) token ids, as their HF scripts feed
    model.generate (the template renders the BOS; no second BOS is added)."""
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if hasattr(enc, "keys"):
        enc = enc["input_ids"]
    return [int(x) for x in enc]


def decode(tok, ids) -> str:
    """Their decode: tokenizer.decode(new_ids, skip_special_tokens=True).strip()."""
    return tok.decode(list(ids), skip_special_tokens=True).strip()


# ---- Tinker judge calls (zeus) ------------------------------------------------------------------------------------

def judge_runner(judge: str, budget: float, temperature: float, max_concurrency: int = 32):
    from cb import tk
    return tk.runner(judge, budget, max_concurrency=max_concurrency, temperature=temperature)


async def judge_call(r, messages: list[dict], max_tokens: int, seed: int | None = None) -> dict:
    """One completion -> {answer, thinking, stop, cost, prompt_tokens, cached, completion_tokens}."""
    out = (await r.sample(messages, n=1, max_tokens=max_tokens, seed=seed))[0]
    return dict(answer=out.answer, thinking=out.thinking, stop=out.stop_reason, termination=out.termination,
                cost=round(out.cost_usd, 7), prompt_tokens=out.prompt_tokens, cached=out.cached_prompt_tokens,
                completion_tokens=out.completion_tokens)


def spend(r) -> str:
    b = r.budget.summary()
    return (f"spent ${b['spent_usd']:.4f} (Runner price, list) over {b['requests']} requests; "
            f"prompt {b['prompt_tokens']} ({b['cached_prompt_tokens']} cached), completion {b['completion_tokens']}")
