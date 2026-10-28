"""Inkling-Small (Thinking Machines, TMLv0) prompts, parsing and SFT rendering, for the CoT-control SFT data builder (inksm_sft_data.py).

Inkling has no HF tokenizer or chat template: prompts render through tml-renderers' v0 renderer (O200kBaseChatTokenizer), which inserts a system
'Thinking effort level: E' message after any system messages (EFFORT, read at call time: 0.9, the renderer default and tinker-runner's
Astra setting, unless the caller sets inkio.EFFORT; inksm_sft_data sets it from its mine config). A model turn is
  <|message_model|><|content_thinking|>COT<|end_message|><|message_model|><|content_text|>RESPONSE<|end_message|><|content_model_end_sampling|>
and sampling stops on <|content_model_end_sampling|>. sft_datum renders the same turn through render_for_sft (all model tokens weighted 1), with
the effort message inserted where render_for_completion_with_effort puts it, so an SFT example's prompt ids equal the sampling prompt's
(checked by sft_datum). Tok wraps the tokenizer in the HF calls force_check / grpo_obf use (encode with special tokens written as '<|name|>',
decode, get_vocab, eos_token).
"""
import re

import tinker
from tml_renderers import chat as tml_chat, tinker as tml_tinker, tokenizers as tml_tokenizers, v0 as tml_v0

MODEL = "thinkingmachines/Inkling-Small"
EFFORT = 0.9
THINK = "<|message_model|><|content_thinking|>"
TEXT = "<|message_model|><|content_text|>"
END = "<|end_message|>"
STOP = "<|content_model_end_sampling|>"
SPECIAL_RE = re.compile(r"(<\|[a-z_0-9]+\|>)")


class Tok:
    """HF-shaped view of the TMLv0 tokenizer."""

    def __init__(self):
        self.t = tml_tokenizers.O200kBaseChatTokenizer()
        self.specials = set(self.t.all_special_tokens)
        self.eos_token, self.eos_token_id = STOP, self.encode(STOP)[0]
        self._vocab = None

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        out = []
        for part in SPECIAL_RE.split(s):
            if SPECIAL_RE.fullmatch(part) and part in self.specials:
                out.append(self.t.encode_special(part[2:-2]))
            elif part:
                out += self.t.encode_ordinary(part)
        return out

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        text = self.t.decode(list(ids))
        return SPECIAL_RE.sub("", text) if skip_special_tokens else text

    def get_vocab(self) -> dict:
        """token string -> id over the ordinary tokens that decode to valid UTF-8 (force_check.digit_ids reads the all-digit ones)."""
        if self._vocab is None:
            self._vocab = {}
            for i in range(200000):
                try:
                    b = self.t.decode_bytes([i])
                    if not self.t.is_special_token(i):
                        self._vocab.setdefault(b.decode("utf-8"), i)
                except Exception:  # noqa: BLE001 (ids outside the vocabulary, partial UTF-8 bytes)
                    continue
        return self._vocab


_RENDERER = None


def renderer() -> "tml_v0.Renderer":
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = tml_v0.Renderer(tml_tokenizers.O200kBaseChatTokenizer())
    return _RENDERER


def _messages(user: str, system: str | None) -> list[dict]:
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]


def render(user: str, system: str | None = None, effort: float | None = None) -> list[int]:
    """Prompt ids, ending after the user turn's <|end_message|> (the model's turn is not opened)."""
    spans, _ = renderer().render_for_completion_with_effort(tml_chat.OpenAIMessage.from_oss_messages(_messages(user, system)), EFFORT if effort is None else effort)
    return tml_tinker.token_spans_to_tinker_model_input(spans).to_ints()


def encode(task: dict, effort: float | None = None) -> list[int]:
    """grpo_obf.encode for Inkling: a task's prompt ids (its instruction, if in task['system'], is a system message)."""
    return render(task["user"], task.get("system"), effort)


def to_think(gen: str) -> str:
    """A decoded model turn (starting at THINK, or at TEXT for a no-thinking turn) as '<think>COT</think>RESPONSE', unclosed if it stopped while
    thinking; the form extract.split_output and grpo_obf.score_roll read (tinker_cotcontrol.tml_to_think, plus the no-thinking case)."""
    if gen.startswith(TEXT):
        gen = THINK + END + gen
    assert gen.startswith(THINK), gen[:100]
    thinking, closed, rest = gen[len(THINK):].partition(END)
    text = rest.removesuffix(STOP).replace(TEXT, "").replace(END, "")
    assert not SPECIAL_RE.findall(thinking + text), SPECIAL_RE.findall(gen)
    return "<think>" + thinking + ("</think>" + text if closed else "")


def thinking_ids(tok: Tok, tokens: list[int]) -> list[int] | None:
    """The CoT ids of a sampled turn: after THINK, up to the first <|end_message|>, when a text message follows (else None: an unclosed or
    multi-message turn, whose ids would not match a single thinking message)."""
    head, end, text = tok.encode(THINK), tok.encode(END)[0], tok.encode(TEXT)
    if tokens[:len(head)] != head or end not in tokens[len(head):]:
        return None
    i = tokens.index(end, len(head))
    return tokens[len(head):i] if tokens[i + 1:i + 1 + len(text)] == text and tokens.count(head[1]) == 1 else None


def completion_ids(tok: Tok, cot: list[int], response: str) -> list[int]:
    """A model turn's ids from CoT ids (sampled, or tok.encode of a text CoT) and the response text, as render_for_sft lays it out."""
    return tok.encode(THINK) + list(cot) + tok.encode(END + TEXT) + tok.t.encode_ordinary(response) + tok.encode(END + STOP)


def sft_datum(user: str, system: str | None, reasoning: str, response: str, effort: float | None = None) -> tuple[tinker.ModelInput, list[float], int]:
    """(model input, per-token weights, prompt length) of one SFT example: the prompt, then a thinking message and a text message, weighted 1 on
    the model's tokens. Asserts that the unweighted prefix is the sampling prompt (render), so training and sampling see the same ids."""
    effort = EFFORT if effort is None else effort
    msgs = _messages(user, system) + [{"role": "assistant", "content": [{"type": "thinking", "thinking": reasoning}, {"type": "text", "text": response}]}]
    native = [m for om in tml_chat.OpenAIMessage.from_oss_messages(msgs) for m in om.to_messages()]
    at = next(i for i, m in enumerate(native) if m.author.kind != tml_chat.AuthorKind.System)
    native.insert(at, tml_chat.Message(content=tml_chat.ThinkingEffort(round(effort * 1000)), author=tml_chat.Author(tml_chat.AuthorKind.System)))
    (example,) = renderer().render_for_sft(native)
    model_input, weights = tml_tinker.training_example_to_tinker_model_input_and_weights(example)
    ids, n_prompt = model_input.to_ints(), next(i for i, w in enumerate(weights) if w > 0)
    assert ids[:n_prompt] == render(user, system, effort), "SFT prompt ids differ from the sampling prompt"
    return model_input, list(weights), n_prompt
