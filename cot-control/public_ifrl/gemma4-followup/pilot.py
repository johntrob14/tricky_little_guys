"""Native-mode Gemma4 pilot. API responses are retained verbatim, not raw IDs.

This is a 36-cell setup/observability pilot. It does not silently scale or claim
frequency estimates. Model/API identifiers and provider are frozen in protocol.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time

import httpx
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
PROTO = json.loads((HERE / "protocol.json").read_text())
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def append(name, record):
    with (OUT / name).open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def credentials():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for line in Path("/home/john/.config/sauce/openrouter.env").read_text().splitlines():
        match = re.match(r"(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.*)$", line.strip())
        if match:
            return shlex.split(match.group(1))[0]
    raise RuntimeError("OpenRouter key unavailable")


def render(tokenizer, unit):
    enabled = unit["target"] == "cot"
    messages = ([{"role": "system", "content": unit["system"]}] if unit.get("system") else [])
    messages.append({"role": "user", "content": unit["user"]})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enabled)
    if unit.get("hist"):
        history = ""
        for question, thought, answer in unit["hist"]:
            history += "<|turn>user\n" + question + "<turn|>\n<|turn>model\n"
            if enabled:
                history += "<|channel>thought\n" + thought + "\n<channel|>"
            history += answer + "<turn|>\n"
        position = prompt.rfind("<|turn>user\n")
        assert position >= 0
        prompt = prompt[:position] + history + prompt[position:]
    assert ("<|think|>" in prompt) == enabled
    assert prompt.count("<|turn>model\n") == len(unit.get("hist", [])) + 1
    if not enabled:
        assert prompt.endswith("<|turn>model\n<|channel>thought\n<channel|>")
    return prompt


def parse(choice):
    text = choice.get("text") or ""
    structured = choice.get("reasoning") or ""
    if structured:
        return dict(reasoning=structured, response=text.strip(), observed_thought=True, channel_source="API reasoning field")
    marker = "<|channel>thought\n"
    if marker in text:
        thinking = text.split(marker, 1)[1]
        if "<channel|>" in thinking:
            reasoning, answer = thinking.split("<channel|>", 1)
            return dict(reasoning=reasoning.strip(), response=answer.removesuffix("<turn|>").strip(), observed_thought=True, channel_source="native thought delimiters")
        return dict(reasoning=thinking.strip(), response="", observed_thought=True, channel_source="unclosed native thought delimiter")
    return dict(reasoning="", response=text.removesuffix("<turn|>").strip(), observed_thought=False, channel_source="no observed thought channel")


async def main():
    tokenizer = AutoTokenizer.from_pretrained(PROTO["hf_model"], revision=PROTO["hf_revision"])
    manifest = json.loads((HERE / "provider_catalog.json").read_text())
    headers = {"Authorization": "Bearer " + credentials()}
    spend = 0.0
    async with httpx.AsyncClient(timeout=600) as client:
        current = (await client.get("https://openrouter.ai/api/v1/models/google/gemma-4-31b-it/endpoints")).json()
        (OUT / "provider_catalog_at_run.json").write_text(json.dumps(current, indent=2))
        offers = [r for r in current["data"]["endpoints"] if r["provider_name"] == PROTO["provider"]]
        assert any(float(r["pricing"]["prompt"]) <= PROTO["prompt_rate_per_token"] and float(r["pricing"]["completion"]) <= PROTO["completion_rate_per_token"] for r in offers), "provider price changed; review before spending"

        async def sample(prompt, max_tokens):
            nonlocal spend
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            reserve = len(input_ids) * PROTO["prompt_rate_per_token"] + max_tokens * PROTO["completion_rate_per_token"]
            if spend + reserve > PROTO["budget_usd"]:
                raise RuntimeError("pilot budget would be exceeded; pilot incomplete")
            body = dict(model=PROTO["model"], prompt=prompt, max_tokens=max_tokens, temperature=PROTO["temperature"],
                        top_p=PROTO["top_p"], seed=PROTO["seed"], include_reasoning=True,
                        usage={"include": True}, provider={"order": [PROTO["provider"]], "allow_fallbacks": False})
            for attempt in range(6):
                reply = await client.post("https://openrouter.ai/api/v1/completions", json=body, headers=headers)
                data = reply.json()
                append("api_attempts.jsonl", dict(time=time.time(), request=body, http_status=reply.status_code, response=data))
                if reply.status_code == 200 and data.get("choices"):
                    break
                if reply.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"provider request failed: {reply.status_code}")
                await asyncio.sleep(min(60, 2 ** attempt))
            else:
                raise RuntimeError("pinned provider unavailable; no fallback used")
            usage = data.get("usage") or {}
            charged = usage.get("cost")
            estimate = (usage.get("prompt_tokens") or len(input_ids)) * PROTO["prompt_rate_per_token"] + (usage.get("completion_tokens") or max_tokens) * PROTO["completion_rate_per_token"]
            spend += charged if charged is not None else estimate
            choice = data["choices"][0]
            return dict(request=body, response_json=data, parsed=parse(choice), cost=charged, estimated_cost=estimate,
                        cumulative_cost_or_estimate=spend, prompt_token_ids=input_ids,
                        prompt_count_difference=(usage.get("prompt_tokens") or 0) - len(input_ids),
                        finish_reason=choice.get("finish_reason"), provider=data.get("provider"))

        # Verify raw pass-through and both modes before spending on pilot cells.
        hello = await sample("Hello", 1)
        append("preflight.jsonl", dict(test="bare raw tokenization", **hello))
        if abs(hello["prompt_count_difference"]) > 1:
            raise RuntimeError("provider raw prompt handling is uncertain")
        for target in ("cot", "output"):
            unit = dict(target=target, user="Compute 17*23 + 29*31. Check the arithmetic, then give the result.", system=None, hist=[])
            result = await sample(render(tokenizer, unit), 2048)
            append("preflight.jsonl", dict(test="native mode", target=target, **result))
            if abs(result["prompt_count_difference"]) > 1:
                raise RuntimeError("provider prompt count differs from native rendered prompt")
            if result["parsed"]["observed_thought"] != (target == "cot"):
                raise RuntimeError("native thinking-mode observability failed; inspect preflight before scaling")
            if target == "cot" and not result["parsed"]["reasoning"].strip():
                raise RuntimeError("thinking-on preflight returned an empty trace")

        units = [json.loads(line) for line in (HERE / "prompts.jsonl").open()]
        prior = [json.loads(line) for line in (OUT / "generations.jsonl").open()] if (OUT / "generations.jsonl").exists() else []
        done = {(r["unit"]["key"], r["unit"]["target"], r["unit"]["elicitation"]) for r in prior}
        for unit in units:
            if (unit["key"], unit["target"], unit["elicitation"]) in done:
                continue
            prompt = render(tokenizer, unit)
            result = await sample(prompt, PROTO["max_tokens"])
            append("generations.jsonl", dict(unit=unit, rendered_prompt=prompt, seed=PROTO["seed"], **result))
            print(json.dumps(dict(completed=len(done) + 1, target=unit["target"], elicitation=unit["elicitation"], cost_or_estimate=spend)), flush=True)
            done.add((unit["key"], unit["target"], unit["elicitation"]))
        assert len(done) == 36
        (OUT / "PILOT_DONE").write_text(json.dumps(dict(n=36, cost_or_estimate=spend, status="raw pilot complete; manual transcript and scoring audit required before scaling")))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        (OUT / "NEEDS_REVIEW.json").write_text(json.dumps(dict(error=type(error).__name__,message=str(error),time=time.time())))
        raise
