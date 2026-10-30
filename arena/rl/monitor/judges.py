"""Live judges. Each is `async judge(prompt) -> (text, in_tokens, out_tokens, thinking)`, where `text` is the final answer only (the part the score is parsed
from) and `thinking` is the model's reasoning, kept apart. Nothing here runs at import; the factories build clients only when called (never in tests)."""

import asyncio
import os
from pathlib import Path

OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_ENV_FILE = Path.home() / ".config/sauce/openrouter.env"
THINK_END = "</think>"


class JudgeError(RuntimeError):
    pass


def _split(gen, eos="", in_think=True):
    """Thinking-model reply -> (thinking, final). The chat template ends the prompt inside <think>, so a generation is `thinking</think>answer`. With no
    `</think>`: in_think means the reply was cut off inside the thinking (no final answer, nothing to parse); otherwise it is all answer."""
    t = gen.rstrip()
    while eos and t.endswith(eos):
        t = t[: -len(eos)].rstrip()
    if THINK_END in t:
        think, _, final = t.rpartition(THINK_END)
        return think.replace("<think>", "", 1).strip(), final.strip()
    if in_think:  # never closed: truncated inside the thinking, so there is no final answer to parse
        return t.replace("<think>", "", 1).strip(), ""
    return "", t.strip()


split_thinking = _split


def gptoss_reply(msg, termination):
    """(final answer, thinking) from the cookbook gpt-oss renderer's parse_response. A reply that did not end cleanly (cut off before <|return|>) or that
    parsed into no channel parts comes back as one raw string with the analysis channel in it: it has no final answer, so the text is "" (unparsed) and
    the raw decode is kept as thinking. A draft <score> in the judge's reasoning is never read as its score."""
    c = msg["content"]
    if isinstance(c, str) or not getattr(termination, "is_clean", False):
        return "", c if isinstance(c, str) else "".join(b.get("thinking", "") or b.get("text", "") for b in c)
    text = "".join(b.get("text", "") for b in c if b.get("type") == "text")
    think = "".join(b.get("thinking", "") for b in c if b.get("type") == "thinking")
    return text, think


def tinker_judge(model, max_tokens):
    """Tinker sampling client, temperature 0. Thinking models (anything but gpt-oss): the model's own chat template, which ends the prompt inside <think>,
    sampled to the EOS token; as in cot-control-deepseek/tinker_cotcontrol.py. gpt-oss keeps the cookbook renderer at low reasoning effort."""
    import tinker

    if model.startswith("openai/gpt-oss"):
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        client = tinker.ServiceClient().create_sampling_client(base_model=model)
        renderer = renderers.get_renderer("gpt_oss_low_reasoning", get_tokenizer(model))
        params = tinker.SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=renderer.get_stop_sequences())

        async def judge(prompt):
            mi = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
            out = await client.sample_async(prompt=mi, num_samples=1, sampling_params=params)
            toks = out.sequences[0].tokens
            text, think = gptoss_reply(*renderer.parse_response(toks))
            return text, mi.length, len(toks), think

        return judge

    client = tinker.ServiceClient().create_sampling_client(base_model=model)
    tok = client.get_tokenizer()
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=[tok.eos_token_id])

    async def judge(prompt):
        rendered = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        if not rendered.rstrip().endswith("<think>"):
            raise JudgeError(f"chat template does not end in <think>: ...{rendered[-60:]!r}")
        ids = tok.encode(rendered, add_special_tokens=False)
        res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=1, sampling_params=params)
        toks = res.sequences[0].tokens
        thinking, final = _split(tok.decode(toks, skip_special_tokens=False), tok.eos_token or "", in_think=True)
        return final, len(ids), len(toks), thinking

    return judge


def load_openrouter_key(env=None, path=None):
    """Key from OPENROUTER_API_KEY, else KEY=VALUE lines of ~/.config/sauce/openrouter.env. Never printed or logged."""
    env = os.environ if env is None else env
    k = (env.get("OPENROUTER_API_KEY") or "").strip()
    if k:
        return k
    p = Path(path) if path else OPENROUTER_ENV_FILE
    if p.is_file():
        for line in p.read_text().splitlines():
            name, sep, val = line.strip().partition("=")
            if sep and name.strip() in ("OPENROUTER_API_KEY", "OPENROUTER_KEY"):
                val = val.strip().strip("'\"")
                if val:
                    return val
    raise JudgeError(f"no OpenRouter key: set OPENROUTER_API_KEY or put OPENROUTER_API_KEY=... in {p}")


def save_openrouter_key(key, path=None):
    """Write OPENROUTER_API_KEY=... to the key file (mode 600), replacing any existing key line. Returns the path; the key is never echoed."""
    key = key.strip()
    if not key or any(c.isspace() for c in key):
        raise JudgeError("key is empty or contains whitespace")
    p = Path(path) if path else OPENROUTER_ENV_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    keep = [l for l in (p.read_text().splitlines() if p.is_file() else []) if l.strip().partition("=")[0].strip() not in ("OPENROUTER_API_KEY", "OPENROUTER_KEY")]
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(keep + [f"OPENROUTER_API_KEY={key}"]) + "\n")
    os.chmod(p, 0o600)
    return p


def openrouter_settings(model):
    if model == "openai/gpt-6-luna":
        return {"reasoning": {"effort": "low"}, "provider": {"only": ["openai"], "allow_fallbacks": False}, "transforms": [], "usage": {"include": True}}
    return {}


def openrouter_request(model, prompt, max_tokens):
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens, **openrouter_settings(model)}


def openrouter_judge(model, max_tokens, key, transport=None, base_url=OPENROUTER_URL, retries=3, sleep=asyncio.sleep):
    """OpenAI-compatible chat completions through httpx. `transport` (e.g. httpx.MockTransport) is for tests. 429 and 5xx are retried; anything else raises
    JudgeError carrying the status and a short body excerpt, never the request headers."""
    import httpx

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    async def judge(prompt):
        body = openrouter_request(model, prompt, max_tokens)
        async with httpx.AsyncClient(transport=transport, timeout=300.0) as c:
            for attempt in range(retries):
                try:
                    r = await c.post(f"{base_url}/chat/completions", headers=headers, json=body)
                except httpx.HTTPError as e:
                    if attempt + 1 == retries:
                        raise JudgeError(f"openrouter request failed: {type(e).__name__}") from None
                    await sleep(2 ** attempt)
                    continue
                if r.status_code == 429 or r.status_code >= 500:
                    if attempt + 1 == retries:
                        raise JudgeError(f"openrouter HTTP {r.status_code} after {retries} attempts: {r.text[:200]}")
                    await sleep(2 ** attempt)
                    continue
                if r.status_code != 200:
                    raise JudgeError(f"openrouter HTTP {r.status_code}: {r.text[:200]}")
                break
        try:
            d = r.json()
            if d.get("error"):
                raise JudgeError(f"openrouter error: {str(d['error'])[:200]}")
            msg = d["choices"][0]["message"]
            u = d.get("usage") or {}
            content = msg.get("content") or ""
            think = msg.get("reasoning") or msg.get("reasoning_content") or ""
            if THINK_END in content:  # some providers inline the thinking
                t2, content = _split(content, in_think=True)
                think = think or t2
            return content, int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0), think
        except JudgeError:
            raise
        except Exception as e:
            raise JudgeError(f"openrouter: malformed response ({type(e).__name__})") from None

    return judge
