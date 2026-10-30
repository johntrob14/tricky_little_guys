"""DeepSeek V4.1 Flash on Tinker: the prompt format and the completion parser are DeepSeek's own reference implementation (encoding/encoding.py in
deepseek-ai/DeepSeek-V4.1-Flash-1017 at a pinned revision), wrapped in the methods tinker-cookbook's Inspect adapter calls. The cookbook has no V4.1
renderer, and its DeepSeek V3 one writes V3's tool-call tokens where V4.1 writes DSML tags.

Registered renderer names (pass with `-M renderer=...`):
    deepseek_v41_chat              thinking off: the generation header closes the think block at once (encoding.py's chat mode)
    deepseek_v41_thinking_<N>      thinking on at numeric effort N in {25, 50, 75, 100} (75 is the format default "high")
With tools declared the format keeps every earlier turn's reasoning in the history (encoding.py turns drop_thinking off when tools are present).
A sample that does not parse is returned as an unparsed tool call, which the agent sees as a tool error, as with the cookbook renderers.
"""
import importlib.util

import tinker
from huggingface_hub import hf_hub_download
from tinker_cookbook import renderers
from tinker_cookbook.renderers.base import ensure_list

REPO, REVISION = "deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170"
_spec = importlib.util.spec_from_file_location("ds41_encoding", hf_hub_download(REPO, "encoding/encoding.py", revision=REVISION))
enc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(enc)


def _to_ds(m):
    """A cookbook Message as encoding.py's OpenAI-style dict."""
    if m["role"] == "assistant":
        parts = ensure_list(m["content"])
        return {"role": "assistant", "content": "".join(p["text"] for p in parts if p["type"] == "text"),
                "reasoning_content": "".join(p["thinking"] for p in parts if p["type"] == "thinking"),
                "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in m.get("tool_calls") or []]}
    if m["role"] == "tool":
        return {"role": "tool", "tool_call_id": m.get("tool_call_id") or "", "content": renderers.ensure_text(m["content"])}
    return {"role": m["role"], "content": renderers.ensure_text(m["content"]), **({"tools": m["tools"]} if m.get("tools") else {})}


class DeepSeekV41Renderer:
    def __init__(self, tokenizer, effort=None):  # effort None: thinking off
        self.tokenizer, self.effort, self.mode = tokenizer, effort, "chat" if effort is None else "thinking"
        self.eos = tokenizer.encode(enc.eos_token, add_special_tokens=False)
        assert len(self.eos) == 1, self.eos

    def create_conversation_prefix_with_tools(self, tools, system_prompt=""):
        return [{"role": "system", "content": system_prompt, "tools": [{"type": "function", "function": dict(t)} for t in tools]}]

    def build_generation_prompt(self, messages):
        text = enc.encode_messages([_to_ds(m) for m in messages], thinking_mode=self.mode, reasoning_effort=self.effort)
        return tinker.ModelInput.from_ints(self.tokenizer.encode(text, add_special_tokens=False))

    def get_stop_sequences(self):
        return self.eos

    def parse_response(self, response):
        text = self.tokenizer.decode(list(response))
        ended = text.endswith(enc.eos_token)
        try:
            m = enc.parse_message_from_completion_text(text if ended else text + enc.eos_token, thinking_mode=self.mode)
        except (AssertionError, ValueError) as e:
            return ({"role": "assistant", "content": text.removesuffix(enc.eos_token), "unparsed_tool_calls": [renderers.UnparsedToolCall(raw_text=text, error=f"{type(e).__name__}: {e}")]},
                    renderers.ParseTermination.MALFORMED)
        content = ([renderers.ThinkingPart(type="thinking", thinking=m["reasoning_content"])] if m["reasoning_content"] else []) + [renderers.TextPart(type="text", text=m["content"])]
        calls = [renderers.ToolCall(function=renderers.ToolCall.FunctionBody(name=c["function"]["name"], arguments=c["function"]["arguments"])) for c in m["tool_calls"]]
        return {"role": "assistant", "content": content, **({"tool_calls": calls} if calls else {})}, renderers.ParseTermination.STOP_SEQUENCE if ended else renderers.ParseTermination.MALFORMED


renderers.register_renderer("deepseek_v41_chat", lambda tokenizer, image_processor=None: DeepSeekV41Renderer(tokenizer))
for _effort in (25, 50, 75, 100):
    renderers.register_renderer(f"deepseek_v41_thinking_{_effort}", lambda tokenizer, image_processor=None, e=_effort: DeepSeekV41Renderer(tokenizer, e))
