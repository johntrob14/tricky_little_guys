"""A scripted stand-in for tinker.SamplingClient, for tests: completions are Harmony text, tokenized with the real tokenizer."""

from types import SimpleNamespace


def analysis(text):
    return f"<|channel|>analysis<|message|>{text}<|end|>"


def call(thinking, tool, args, constrain=True):
    body = f"<|message|>{args}" if args is not None else ""
    return analysis(thinking) + f"<|start|>assistant<|channel|>commentary to=functions.{tool} " + ("<|constrain|>json" if constrain else "") + body + "<|call|>"


def final(thinking, text):
    return analysis(thinking) + f"<|start|>assistant<|channel|>final<|message|>{text}<|return|>"


class FakeSamplingClient:
    """script: a list of completion strings, or (string, stop_reason) pairs, consumed in order. Records every request in `calls` as (prompt_ids, params)."""

    def __init__(self, tokenizer, script):
        self.tok, self.script, self.calls = tokenizer, list(script), []

    @staticmethod
    def logprob(i, token):
        return -((token * 31 + i) % 17 + 1) / 50

    async def sample_async(self, prompt, num_samples, sampling_params):
        ids = prompt.to_ints()
        self.calls.append((ids, sampling_params))
        item = self.script.pop(0)
        text, reason = item if isinstance(item, tuple) else (item, "stop")
        tokens = self.tok.encode(text, add_special_tokens=False)[: sampling_params.max_tokens]
        seq = SimpleNamespace(tokens=tokens, logprobs=[self.logprob(i, t) for i, t in enumerate(tokens)], stop_reason=reason)
        return SimpleNamespace(sequences=[seq], prompt_cache_hit_tokens=0)
