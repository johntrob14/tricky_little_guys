"""CoT penalty for the arena datum path: per-token masks over gpt-oss/harmony turns and the advantage transform that subtracts a monitor
suspicion signal from the model's non-final, model-visible text only (analysis reasoning and commentary preambles), leaving tool-call and
final-answer tokens untouched. Ported from tlg-directive self-oversight/cot_penalty.py; the channel rules follow rl.provider.parse_turn:
an analysis segment whose header starts with `to=` is a tool call, not reasoning, and a commentary segment with a `to=` recipient is a call.
"""

from __future__ import annotations

import functools
import re

# gpt-oss / harmony single-token markers (verified against the gpt-oss-120b tokenizer offline;
# equal to cot_penalty.py's constants).
CHANNEL, MESSAGE, END, START, CALL, RETURN = 200005, 200008, 200007, 200006, 200012, 200002
_BOUNDARY = {END, START, CALL, RETURN}
_HEADER = re.compile(r"(?P<ch>[^\s<]*)(?P<hdr>.*)", re.S)  # parse_turn's channel name and header split
_RECIPIENT = re.compile(r"to=[^\s<]+")


def _segment_kind(header: str) -> str:
    """'reasoning', 'commentary' (preamble text), 'call' or 'final'/'other' for one decoded `<|channel|>...<|message|>` header, as parse_turn reads it."""
    m = _HEADER.match(header.strip())
    ch, hdr = m["ch"], m["hdr"]
    if ch == "analysis" and not hdr.strip().startswith("to="):
        return "reasoning"
    if _RECIPIENT.search(hdr):
        return "call"
    return {"commentary": "commentary", "final": "final"}.get(ch, "other")


def segment_mask(sampled_token_ids: list[int], tokenizer, kinds: frozenset[str]) -> list[int]:
    """1 on the message-body tokens of segments whose kind (_segment_kind) is in kinds, 0 elsewhere (markers, headers, other segments)."""
    mask = [0] * len(sampled_token_ids)
    header: list[int] = []
    reading_channel = in_body = on = False
    for i, t in enumerate(sampled_token_ids):
        if t == CHANNEL:
            reading_channel, header, in_body = True, [], False
            continue
        if t == MESSAGE:
            on = _segment_kind(tokenizer.decode(header) if header else "") in kinds
            reading_channel, in_body = False, True
            continue
        if t in _BOUNDARY:
            in_body = on = False
            header = []
            continue
        if reading_channel:
            header.append(t)
            continue
        if in_body and on:
            mask[i] = 1
    return mask


def think_mask(sampled_token_ids: list[int], tokenizer) -> list[int]:
    """1 on analysis-channel reasoning bodies (parse_turn's reasoning), 0 elsewhere: markers, analysis- or commentary-channel tool calls, commentary
    preambles, the final answer. sampled_token_ids is one turn's sampled completion (what follows `<|start|>assistant`)."""
    return segment_mask(sampled_token_ids, tokenizer, frozenset({"reasoning"}))


def penalty_mask(sampled_token_ids: list[int], tokenizer) -> list[int]:
    """The CoT-penalty mask: analysis reasoning plus commentary preambles (parse_turn's non-call commentary text), i.e. every non-final, model-visible text
    the CoT monitor (rl.cot_monitor.turn_views) reads. Tool-call tokens (either channel) and the final answer stay 0."""
    return segment_mask(sampled_token_ids, tokenizer, frozenset({"reasoning", "commentary"}))


@functools.lru_cache(maxsize=4)
def _tokenizer_for(model: str):
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return get_tokenizer(model)


def cot_token_advantages(episode, scalar_adv: float, beta: float, c_turns: list[float]) -> list[list[float]]:
    """Per-turn, per-token advantage lists (same shape/logic as datums.turn_advantages): for turn k,
    a_t = scalar_adv - beta * c_turns[k] * penalty_mask_t over the turn's sampled tokens, so only
    reasoning and commentary-preamble tokens are penalized; tool-call and final-answer tokens keep
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
        mask = penalty_mask(t.tokens, tok)
        out.append([scalar_adv - pen * m for m in mask])
    return out
