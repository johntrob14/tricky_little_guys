"""Muse Spark Open through OpenRouter's raw /completions endpoint, so a turn's prompt can end with text the chat endpoint will not take
(OpenRouter marks reasoning as mandatory for meta/muse-spark-open, and the model has no thinking-off mode: its effort is low, medium, high or xhigh).

    run_attack.py tiktok/... --model museraw/meta/muse-spark-open --prefill muse_no_cot --reasoning-effort medium

Each turn is rendered with Meta's chat template (muse_spark_open.jinja, copied from unsloth/Muse-Spark-Open-816B-A42B since the meta-models
repo is gated), the prefill is appended after its `<|start|>assistant` generation prompt, and the completion is parsed back into messages.
prompts/prefill/muse_no_cot.txt is an empty `to=self` (reasoning) message followed by a new assistant header; the model can still open another
`to=self` message in the same turn, and that text is kept as the turn's reasoning. Earlier turns' reasoning stays in the history, as the template
and Inspect's OpenRouter provider both do. The provider must pass the prompt through untouched and return special tokens in the completion text:
Fireworks does (Together and Nebius pass it through but strip the special tokens; Makora, Modal and BaseTen wrap it in their own template).
"""

import json
import os
import re
import uuid
from pathlib import Path

import httpx
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ContentReasoning, ContentText, GenerateConfig, ModelAPI, ModelOutput, ModelUsage
from inspect_ai.model._registry import modelapi_register
from inspect_ai.tool import ToolCall
from transformers.utils.chat_template_utils import _compile_jinja_template

TEMPLATE = _compile_jinja_template((Path(__file__).parent / "muse_spark_open.jinja").read_text())
URL = "https://openrouter.ai/api/v1/completions"
SEGMENT = re.compile(r"^\s*to=([^<]*?)\s*<\|message\|>(.*)$", re.S)
INVOKE = re.compile(r'<atem:invoke name="([^"]*)">(.*?)</atem:invoke>', re.S)
PARAM = re.compile(r'<atem:parameter name="([^"]*)">(.*?)</atem:parameter>', re.S)


def to_template(input) -> list[dict]:
    out = []
    for m in input:
        if m.role == "assistant":
            calls = [{"id": c.id, "type": "function", "function": {"name": c.function, "arguments": c.arguments}} for c in m.tool_calls or []]
            reasoning = "".join(c.reasoning for c in m.content if isinstance(c, ContentReasoning)) if isinstance(m.content, list) else ""
            out.append({"role": "assistant", "content": m.text, "reasoning": reasoning, "tool_calls": calls})
        elif m.role == "tool":
            out.append({"role": "tool", "name": m.function, "tool_call_id": m.tool_call_id, "content": f"Error: {m.error.message}" if m.error is not None and not m.text else m.text})
        else:
            out.append({"role": m.role, "content": m.text})
    return out


def arg_value(raw: str, schema: dict):
    """String parameters as written, everything else as JSON, as the template's tool instructions say. A parameter that may be a string
    (`{"anyOf": [{"type": "string"}, {"type": "null"}]}`, or submit's string|integer|number) is kept as written."""
    types = {schema["type"]} if "type" in schema else {a.get("type") for a in schema.get("anyOf", [])}
    return raw if "string" in types or not types else json.loads(raw)


def parse(text: str, tools) -> tuple[str, str, list[ToolCall]]:
    """Completion text -> (reasoning, visible text, tool calls). Messages are separated by `<|eom|><|start|>assistant`; a segment without a
    `to=` header (the model writing past the format) is kept as visible text."""
    schemas = {t.name: t.parameters.model_dump(exclude_none=True).get("properties", {}) for t in tools}
    reasoning, visible, calls = [], [], []
    for seg in re.split(r"<\|eom\|>\s*<\|start\|>assistant", text.removesuffix("<|eot|>").removesuffix("<|eom|>")):
        head = SEGMENT.match(seg)
        if not head:
            visible.append(seg.strip())
            continue
        recipient, body = head.group(1).strip(), head.group(2).removesuffix("<|eom|>").removesuffix("<|eot|>")
        if recipient == "self":
            reasoning.append(body)
        elif recipient in ("commentary", "user"):
            visible.append(body)
        else:
            for name, params in INVOKE.findall(body):   # a non-string argument that is not JSON becomes a parse_error, which Inspect returns to the model as the call's error
                props, call_id = schemas.get(name, {}), f"call_{uuid.uuid4().hex[:12]}"
                try:
                    calls.append(ToolCall(id=call_id, function=name, arguments={k: arg_value(v, props.get(k, {})) for k, v in PARAM.findall(params)}))
                except json.JSONDecodeError as e:
                    calls.append(ToolCall(id=call_id, function=name, arguments={}, parse_error=f"argument is not valid JSON: {e}"))
    return "\n\n".join(r for r in reasoning if r.strip()), "\n\n".join(v for v in visible if v), calls


class MuseRaw(ModelAPI):
    def __init__(self, model_name, base_url=None, api_key=None, config=GenerateConfig(), provider="Fireworks", prefill=None):
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, api_key_vars=["OPENROUTER_API_KEY"], config=config)
        self.provider, self.prefill = provider, prefill or ""
        self.client = httpx.AsyncClient(timeout=600)

    async def generate(self, input, tools, tool_choice, config):
        tool_defs = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters.model_dump(exclude_none=True)}} for t in tools]
        prompt = TEMPLATE.render(messages=to_template(input), tools=tool_defs or None, add_generation_prompt=True, reasoning_effort=config.reasoning_effort or "medium", bos_token="") + self.prefill
        body = {"model": self.model_name, "provider": {"order": [self.provider], "allow_fallbacks": False}, "prompt": prompt, "max_tokens": config.max_tokens,
                "temperature": 1.0 if config.temperature is None else config.temperature, "stop": ["<|eot|>"], "usage": {"include": True}}
        r = await self.client.post(URL, json=body, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {data['error']}")
        choice, usage = data["choices"][0], data["usage"]
        reasoning, visible, calls = parse(choice["text"], tools)
        content = ([ContentReasoning(reasoning=reasoning)] if reasoning else []) + [ContentText(text=visible)]
        stop = "tool_calls" if calls else ("max_tokens" if choice.get("finish_reason") == "length" else "stop")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return ModelOutput(model=self.model_name, choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=content, tool_calls=calls or None, model=self.model_name, source="generate"), stop_reason=stop)],
                           usage=ModelUsage(input_tokens=usage["prompt_tokens"] - cached, input_tokens_cache_read=cached, output_tokens=usage["completion_tokens"], total_tokens=usage["total_tokens"], total_cost=usage.get("cost")),
                           metadata=dict(raw_completion=choice["text"], provider=data.get("provider"), finish_reason=choice.get("finish_reason")))

    def should_retry(self, ex):
        return isinstance(ex, httpx.TransportError) or (isinstance(ex, httpx.HTTPStatusError) and ex.response.status_code in (408, 429, 500, 502, 503, 504))

    def connection_key(self):
        return f"museraw/{self.provider}"


modelapi_register(MuseRaw, "museraw")
