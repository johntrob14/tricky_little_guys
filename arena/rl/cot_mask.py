"""CoT penalty for the arena datum path: a per-token think-mask over gpt-oss/harmony turns and the
advantage transform that subtracts a monitor suspicion signal from analysis-channel (reasoning)
tokens only, leaving action/tool-call and final-answer tokens untouched. Ported from
tlg-directive self-oversight/cot_penalty.py (think_mask is byte-for-byte the same logic).
"""

from __future__ import annotations

import functools

# gpt-oss / harmony single-token markers (verified against the gpt-oss-120b tokenizer offline;
# equal to cot_penalty.py's constants).
CHANNEL, MESSAGE, END, START, CALL, RETURN = 200005, 200008, 200007, 200006, 200012, 200002
_BOUNDARY = {END, START, CALL, RETURN}


def think_mask(sampled_token_ids: list[int], tokenizer) -> list[int]:
    """1 on analysis-channel message bodies, 0 elsewhere (channel/message/boundary markers, commentary
    tool-call tokens, the final answer). sampled_token_ids is one turn's sampled completion (what follows
    `<|start|>assistant`), so it begins at `<|channel|>analysis<|message|>...`."""
    mask = [0] * len(sampled_token_ids)
    channel_name: list[int] = []
    reading_channel = False
    in_body = False
    is_analysis = False
    for i, t in enumerate(sampled_token_ids):
        if t == CHANNEL:
            reading_channel, channel_name, in_body = True, [], False
            continue
        if t == MESSAGE:
            name = tokenizer.decode(channel_name).strip() if channel_name else ""
            is_analysis = name.startswith("analysis")
            reading_channel, in_body = False, True
            continue
        if t in _BOUNDARY:
            in_body, is_analysis = False, False
            continue
        if reading_channel:
            channel_name.append(t)
            continue
        if in_body and is_analysis:
            mask[i] = 1
    return mask


@functools.lru_cache(maxsize=4)
def _tokenizer_for(model: str):
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return get_tokenizer(model)


def cot_token_advantages(episode, scalar_adv: float, beta: float, c_turns: list[float]) -> list[list[float]]:
    """Per-turn, per-token advantage lists (same shape/logic as datums.turn_advantages): for turn k,
    a_t = scalar_adv - beta * c_turns[k] * think_mask_t over the turn's sampled tokens, so only
    analysis-channel reasoning tokens are penalized; action/tool-call and final-answer tokens keep
    scalar_adv. A truncated turn (stop_reason == "length") stays all 0.0, exactly as turn_advantages,
    so truncation is neither rewarded nor penalized. obs_tokens get 0 downstream in merged_datum_fields.
    The tokenizer is loaded (offline) only when a turn's penalty is nonzero; with beta 0 or c_turns[k]==0
    the turn reduces to [scalar_adv]*len(tokens), byte-identical to turn_advantages."""
    out: list[list[float]] = []
    tok = None
    for k, t in enumerate(episode.turns):
        if t.stop_reason == "length":
            out.append([0.0] * len(t.tokens))
            continue
        pen = beta * c_turns[k]
        if pen == 0.0:
            out.append([scalar_adv] * len(t.tokens))
            continue
        if tok is None:
            tok = _tokenizer_for(episode.model or "openai/gpt-oss-120b")
        mask = think_mask(t.tokens, tok)
        out.append([scalar_adv - pen * m for m in mask])
    return out
