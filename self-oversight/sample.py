"""Sampling runner: conversations in, parsed rollouts out, with retries, a concurrency cap and a budget.

    runner = Runner(get_model("gpt-oss-20b"), budget_usd=2.0)                            # base model on OpenRouter
    runner = Runner(get_model("gpt-oss-20b"), budget_usd=2.0, model_path="tinker://...")  # checkpoint on Tinker
    out = await runner.sample(messages, n=2, max_tokens=4096)           # list[Rollout]
    out2 = await runner.continue_(messages, out[0], "Why did you ...?")  # one follow-up turn
    runner.to_ledger("screen", "gpt-oss-20b-unprompted")                 # append the counted cost

Backends (`backend=`): "openrouter" is the default for a base model, "tinker" for a `model_path` checkpoint. Pass
backend="tinker" explicitly to sample a base model on Tinker: needed for anything token-level (prefill, Rollout.tokens,
the renderer's parsed Message with tool calls) and for base-vs-checkpoint comparisons that should not mix providers.

OpenRouter: one chat-completions request per sample (n requests for n samples, seeds seed..seed+n-1), reasoning effort
from the renderer name (gpt_oss_medium_reasoning -> medium), thinking from the reply's `reasoning` field (or an inlined
<think> block), Rollout.tokens empty, token counts and the counted cost (`usage.cost`) from OpenRouter itself. The key
is OPENROUTER_API_KEY (common.load_openrouter_key).

The budget is checked before each request is sent, against the cost of everything returned so far plus a worst-case
reservation for requests in flight (prompt at prefill price, n x max_tokens at sample price), so a run cannot overshoot
its cap by more than the requests already in flight. Prices come from the table in common.py: on OpenRouter the prompt
length is estimated at ~3 chars/token for the reservation; on Tinker the counted cost is token counts at list price
(list_multiplier) by default, as the plan says to budget discounted models at list.

Retries: transient failures (connection, timeout, 408/409/429/5xx, Tinker request-failed) are retried with exponential
backoff and jitter, up to `max_retries` times per request. Renderer errors (wrong stop tokens), other HTTP errors and
budget errors are not retried. A truncated or malformed completion is not an error: it is returned as a Rollout with
`clean=False` so the caller can count it.

Parsing on Tinker: the cookbook renderer for the model splits the completion into thinking (analysis channel or
<think> block) and answer. If the renderer reports the completion as malformed (usually hit max_tokens mid-thought),
a text fallback recovers whatever thinking and answer text is present.

Smoke test (costs a fraction of a cent):

    uv run python sample.py --model gpt-oss-20b --prompt "What is 17*23?" -n 2 --max-tokens 400 --budget 0.02
    uv run python sample.py --model gpt-oss-20b --backend tinker --prompt "What is 17*23?" -n 2 --max-tokens 400 --budget 0.02
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import httpx

from common import Model, get_model, load_key, load_openrouter_key, renderer_for

BACKENDS = ("openrouter", "tinker")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Thinking delimiters seen in raw text when the renderer cannot parse a truncated completion.
_HARMONY = re.compile(r"<\|channel\|>analysis<\|message\|>(?P<think>.*?)(?:<\|end\|>|$)"
                      r"(?:.*?<\|channel\|>final<\|message\|>(?P<answer>.*))?", re.S)
_THINK = re.compile(r"^\s*<think>(?P<think>.*?)(?:</think>(?P<answer>.*)|$)", re.S)
_EFFORT = re.compile(r"_(low|medium|high)_reasoning$")


class BudgetExceeded(RuntimeError):
    pass


class OpenRouterError(RuntimeError):
    def __init__(self, status: int, text: str):
        super().__init__(f"openrouter HTTP {status}: {text[:300]}")
        self.status = status


@dataclass
class Rollout:
    thinking: str
    answer: str
    raw: str                 # decoded completion text (stop tokens stripped by the renderer where possible); the reply content on OpenRouter
    tokens: list[int] = field(default_factory=list, repr=False)   # sampled token ids on Tinker; empty on OpenRouter
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    stop_reason: str = ""    # "stop" or "length": Tinker's stop_reason or OpenRouter's finish_reason
    termination: str = ""    # Tinker: renderer's ParseTermination (stop_sequence, eos, malformed); OpenRouter: the finish_reason
    clean: bool = True       # not malformed (Tinker) / finish_reason "stop" (OpenRouter)
    cost_usd: float = 0.0
    attempts: int = 1
    seconds: float = 0.0
    message: dict = field(default_factory=dict, repr=False)  # the parsed assistant Message, for follow-ups

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("tokens")
        return d


@dataclass
class Budget:
    """Spend tracker with reservations for requests in flight."""
    cap_usd: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    requests: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0

    def reserve(self, worst_case_usd: float) -> None:
        if self.spent_usd + self.reserved_usd + worst_case_usd > self.cap_usd:
            raise BudgetExceeded(
                f"budget ${self.cap_usd:.4f}: spent ${self.spent_usd:.4f}, in flight ${self.reserved_usd:.4f}, "
                f"next request could cost ${worst_case_usd:.4f}")
        self.reserved_usd += worst_case_usd

    def release(self, worst_case_usd: float) -> None:
        self.reserved_usd -= worst_case_usd

    def commit(self, actual_usd: float, prompt: int, cached: int, completion: int) -> None:
        self.spent_usd += actual_usd
        self.requests += 1
        self.prompt_tokens += prompt
        self.cached_prompt_tokens += cached
        self.completion_tokens += completion

    def summary(self) -> dict:
        return {"cap_usd": self.cap_usd, "spent_usd": round(self.spent_usd, 6), "requests": self.requests,
                "prompt_tokens": self.prompt_tokens, "cached_prompt_tokens": self.cached_prompt_tokens,
                "completion_tokens": self.completion_tokens}


def split_thinking(content: Any) -> tuple[str, str]:
    """(thinking, answer) from a renderer Message content: a list of parts, or a raw string."""
    if isinstance(content, list):
        thinking = "".join(p.get("thinking", "") for p in content if p.get("type") == "thinking")
        answer = "".join(p.get("text", "") for p in content if p.get("type") == "text")
        return thinking, answer
    text = content or ""
    m = _HARMONY.search(text) or _THINK.match(text)
    if m:
        return (m.group("think") or "").strip(), (m.group("answer") or "").strip()
    return "", text.strip()


def estimate_tokens(messages: list[dict]) -> int:
    """Prompt-token estimate for budget reservations without a tokenizer: about 3 characters per token, keys included."""
    return sum(len(json.dumps(m)) for m in messages) // 3


def reasoning_effort(renderer: str) -> str | None:
    """OpenRouter reasoning effort matching a cookbook renderer name (gpt_oss_medium_reasoning -> medium), else None."""
    m = _EFFORT.search(renderer)
    return m.group(1) if m else None


def is_retryable(exc: BaseException) -> bool:
    import tinker

    if isinstance(exc, (tinker.APIConnectionError, tinker.APITimeoutError, tinker.RateLimitError,
                        tinker.InternalServerError, tinker.RequestFailedError)):
        return True
    if isinstance(exc, tinker.APIStatusError):
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    return isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError))


def openrouter_retryable(exc: BaseException) -> bool:
    if isinstance(exc, OpenRouterError):
        return exc.status in (408, 409, 429) or exc.status >= 500
    return isinstance(exc, (httpx.HTTPError, asyncio.TimeoutError, ConnectionError, OSError))


class Runner:
    def __init__(self, model: Model, budget_usd: float, *, model_path: str | None = None, backend: str | None = None,
                 max_concurrency: int = 16, max_retries: int = 5, backoff_base: float = 2.0,
                 backoff_max: float = 60.0, temperature: float = 1.0, top_p: float = 1.0,
                 at_list_price: bool = True, request_timeout: float = 600.0,
                 sampler: Any = None, renderer: Any = None, transport: Any = None, api_key: str | None = None,
                 log: Callable[[str], None] | None = print):
        """`backend` defaults to "tinker" when `model_path` (a tinker:// checkpoint) is given, else "openrouter".

        `sampler` and `renderer` (Tinker) or `transport` and `api_key` (OpenRouter, httpx) can be injected for tests;
        otherwise they are built from `model`. A checkpoint is priced as its base model.
        """
        self.model = model
        self.model_path = model_path
        self.backend = backend or ("tinker" if model_path else "openrouter")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if self.backend == "openrouter" and model_path:
            raise ValueError("a tinker:// checkpoint can only be sampled with backend='tinker'")
        if self.backend == "openrouter" and not model.openrouter_id:
            raise ValueError(f"{model.tinker_id} has no openrouter_id in common.MODELS; pass backend='tinker'")
        self.model_id = model.openrouter_id if self.backend == "openrouter" else model.tinker_id
        self.budget = Budget(budget_usd)
        self.sem = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.temperature = temperature
        self.top_p = top_p
        self.price_mult = model.list_multiplier if at_list_price and self.backend == "tinker" else 1.0
        self.request_timeout = request_timeout
        self.log = log or (lambda s: None)
        self._sampler = sampler
        self._renderer = renderer
        self._transport, self._api_key, self._http = transport, api_key, None
        self.errors: list[dict] = []   # every failed attempt, for the report

    # -- lazy client setup -------------------------------------------------------------------------

    @property
    def renderer(self):
        if self._renderer is None:
            _, self._renderer = renderer_for(self.model)
        return self._renderer

    @property
    def sampler(self):
        if self._sampler is None:
            import tinker

            load_key()
            svc = tinker.ServiceClient()
            self._sampler = (svc.create_sampling_client(model_path=self.model_path) if self.model_path
                             else svc.create_sampling_client(base_model=self.model.tinker_id))
        return self._sampler

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            key = self._api_key or load_openrouter_key()
            self._http = httpx.AsyncClient(transport=self._transport, timeout=self.request_timeout,
                                           headers={"Authorization": f"Bearer {key}"})
        return self._http

    # -- pricing -----------------------------------------------------------------------------------

    def cost(self, prompt: int, cached: int, completion: int) -> float:
        m = self.model
        if self.backend == "openrouter":
            return (prompt * m.or_prompt + completion * m.or_completion) / 1e6
        usd = ((prompt - cached) * m.prefill + cached * m.cached_prefill + completion * m.sample) / 1e6
        return usd * self.price_mult

    def worst_case(self, prompt: int, n: int, max_tokens: int) -> float:
        return self.cost(prompt, 0, n * max_tokens)

    def prompt_length(self, messages: list[dict]) -> int:
        """Prompt tokens: the renderer's count on Tinker, a ~3 chars/token estimate on OpenRouter."""
        if self.backend == "openrouter":
            return estimate_tokens(messages)
        return self.renderer.build_generation_prompt(messages).length

    # -- sampling ----------------------------------------------------------------------------------

    async def sample(self, messages: list[dict], n: int = 1, max_tokens: int = 2048,
                     seed: int | None = None, prefill: str | None = None) -> list[Rollout]:
        """n completions of the next assistant turn after `messages`."""
        if self.backend == "openrouter":
            return await self._sample_openrouter(messages, n, max_tokens, seed, prefill)
        import tinker

        with warnings.catch_warnings():
            # The gpt-oss renderer warns that a user system message joins its own Harmony system prompt; it
            # renders ours as the developer message, which is what we want.
            warnings.simplefilter("ignore", UserWarning)
            prompt = self.renderer.build_generation_prompt(messages, prefill=prefill)
        n_prompt = prompt.length
        params = tinker.SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens=max_tokens,
                                       stop=self.renderer.get_stop_sequences(), seed=seed)
        reserve = self.worst_case(n_prompt, n, max_tokens)
        self.budget.reserve(reserve)
        t0 = time.monotonic()
        try:
            async with self.sem:
                resp, attempts = await self._with_retries(lambda: self.sampler.sample_async(prompt, n, params), is_retryable)
        finally:
            self.budget.release(reserve)
        seconds = time.monotonic() - t0

        cached = getattr(resp, "prompt_cache_hit_tokens", 0) or 0
        rollouts = [self._parse(list(seq.tokens), seq.stop_reason, n_prompt, cached, attempts, seconds)
                    for seq in resp.sequences]
        # Prompt tokens are billed once per request, not per sample; charge them to the first rollout.
        for i, r in enumerate(rollouts):
            r.cost_usd = self.cost(n_prompt if i == 0 else 0, cached if i == 0 else 0, r.completion_tokens)
        total_completion = sum(r.completion_tokens for r in rollouts)
        self.budget.commit(self.cost(n_prompt, cached, total_completion), n_prompt, cached, total_completion)
        return rollouts

    async def continue_(self, messages: list[dict], rollout: Rollout, user_text: str, n: int = 1,
                        max_tokens: int = 1024, seed: int | None = None) -> list[Rollout]:
        """One follow-up turn: the conversation so far, the model's parsed reply, and a new user message.

        The renderer decides what of the earlier reply the model sees again (Harmony drops the analysis
        channel of previous turns; <think> renderers usually drop earlier thinking too). On OpenRouter the
        reply is its answer text only.
        """
        history = list(messages) + [rollout.message or {"role": "assistant", "content": rollout.answer},
                                    {"role": "user", "content": user_text}]
        return await self.sample(history, n=n, max_tokens=max_tokens, seed=seed)

    async def _sample_openrouter(self, messages: list[dict], n: int, max_tokens: int, seed: int | None,
                                 prefill: str | None) -> list[Rollout]:
        if prefill is not None:
            raise ValueError("prefill is token-level: use backend='tinker'")
        reserve = self.worst_case(estimate_tokens(messages), n, max_tokens)
        self.budget.reserve(reserve)
        try:
            seeds = [None] * n if seed is None else range(seed, seed + n)
            rollouts = await asyncio.gather(*[self._one_openrouter(messages, max_tokens, s) for s in seeds])
        finally:
            self.budget.release(reserve)
        for r in rollouts:
            self.budget.commit(r.cost_usd, r.prompt_tokens, r.cached_prompt_tokens, r.completion_tokens)
        return list(rollouts)

    async def _one_openrouter(self, messages: list[dict], max_tokens: int, seed: int | None) -> Rollout:
        body: dict = {"model": self.model_id, "messages": messages, "temperature": self.temperature, "top_p": self.top_p,
                      "max_tokens": max_tokens, "usage": {"include": True}}
        if seed is not None:
            body["seed"] = seed
        if effort := reasoning_effort(self.model.renderer):
            body["reasoning"] = {"effort": effort}
        t0 = time.monotonic()
        async with self.sem:
            d, attempts = await self._with_retries(lambda: self._post(body), openrouter_retryable)
        choice, usage = d["choices"][0], d["usage"]
        msg = choice["message"]
        content = msg.get("content") or ""
        thinking, answer = split_thinking(content)   # some providers inline the thinking as a <think> block
        thinking = msg.get("reasoning") or msg.get("reasoning_content") or thinking
        finish = str(choice.get("finish_reason"))
        return Rollout(thinking=thinking, answer=answer, raw=content, prompt_tokens=int(usage["prompt_tokens"]),
                       cached_prompt_tokens=int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                       completion_tokens=int(usage["completion_tokens"]), stop_reason=finish, termination=finish,
                       clean=finish == "stop", cost_usd=float(usage["cost"]), attempts=attempts,
                       seconds=time.monotonic() - t0, message={"role": "assistant", "content": answer})

    async def _post(self, body: dict) -> dict:
        r = await self.http.post(OPENROUTER_URL, json=body)
        if r.status_code != 200:
            raise OpenRouterError(r.status_code, r.text)
        d = r.json()
        if d.get("error"):
            raise OpenRouterError(200, str(d["error"]))
        return d

    async def _with_retries(self, send: Callable[[], Any], retryable: Callable[[BaseException], bool]) -> tuple[Any, int]:
        attempt = 0
        while True:
            attempt += 1
            try:
                return await asyncio.wait_for(send(), self.request_timeout), attempt
            except Exception as e:  # noqa: BLE001 - classified by `retryable`
                ok = retryable(e)
                self.errors.append({"attempt": attempt, "error": repr(e)[:500], "retryable": ok})
                if not ok or attempt > self.max_retries:
                    raise
                delay = min(self.backoff_max, self.backoff_base * 2 ** (attempt - 1))
                delay *= 1 + random.uniform(-0.25, 0.25)
                self.log(f"sample attempt {attempt} failed ({type(e).__name__}); retrying in {delay:.1f}s")
                await asyncio.sleep(delay)

    def _parse(self, toks: list[int], stop_reason: str, n_prompt: int, cached: int, attempts: int,
               seconds: float) -> Rollout:
        try:
            msg, term = self.renderer.parse_response(toks)
            termination = str(getattr(term, "value", term))
            clean = bool(getattr(term, "is_clean", termination != "malformed"))
        except Exception as e:  # noqa: BLE001 - a RendererError on multiple stop tokens; keep the text
            msg = {"role": "assistant", "content": self.renderer.tokenizer.decode(toks)}
            termination, clean = f"error:{type(e).__name__}", False
        content = msg.get("content", "")
        thinking, answer = split_thinking(content)
        raw = content if isinstance(content, str) else self.renderer.tokenizer.decode(toks)
        return Rollout(thinking=thinking, answer=answer, raw=raw, tokens=toks, prompt_tokens=n_prompt,
                       cached_prompt_tokens=cached, completion_tokens=len(toks), stop_reason=str(stop_reason),
                       termination=termination, clean=clean, attempts=attempts, seconds=seconds,
                       message=dict(msg))

    # -- bookkeeping -------------------------------------------------------------------------------

    def to_ledger(self, phase: str, run: str, note: str = "", estimate_usd: float | None = None) -> None:
        """Append the counted spend; OpenRouter rows carry an `openrouter/` model prefix."""
        import ledger

        b = self.budget
        model = self.model_id if self.backend == "tinker" else f"openrouter/{self.model_id}"
        ledger.append(phase, run, model, "sample", estimate_usd=estimate_usd,
                      counted_usd=b.spent_usd,
                      note=note or f"{b.requests} req, {b.prompt_tokens} prompt ({b.cached_prompt_tokens} cached), "
                                   f"{b.completion_tokens} completion tokens, {len(self.errors)} failed attempts")


async def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--model-path", default=None, help="tinker:// checkpoint instead of the base model")
    ap.add_argument("--backend", default=None, choices=BACKENDS, help="default: openrouter for a base model, tinker for --model-path")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--budget", type=float, default=0.05, help="USD cap for this invocation")
    ap.add_argument("--follow-up", default=None, help="a second user turn after the first sample")
    ap.add_argument("--ledger", default=None, help="phase:run to record the spend under")
    args = ap.parse_args()

    runner = Runner(get_model(args.model), args.budget, model_path=args.model_path, backend=args.backend)
    messages = ([{"role": "system", "content": args.system}] if args.system else []) + \
               [{"role": "user", "content": args.prompt}]
    est = runner.worst_case(runner.prompt_length(messages), args.n, args.max_tokens)
    print(f"{runner.backend} {runner.model_id}; worst case for the first request: ${est:.4f}")
    outs = await runner.sample(messages, n=args.n, max_tokens=args.max_tokens)
    for i, r in enumerate(outs):
        print(f"--- sample {i} [{r.stop_reason}/{r.termination}] {r.completion_tokens} tok ${r.cost_usd:.5f}")
        print("thinking:", r.thinking[:800])
        print("answer:", r.answer[:800])
    if args.follow_up:
        fu = await runner.continue_(messages, outs[0], args.follow_up, max_tokens=args.max_tokens)
        print(f"--- follow-up [{fu[0].stop_reason}/{fu[0].termination}]")
        print("thinking:", fu[0].thinking[:800])
        print("answer:", fu[0].answer[:800])
    print(json.dumps(runner.budget.summary()))
    if args.ledger:
        phase, run = args.ledger.split(":", 1)
        runner.to_ledger(phase, run, estimate_usd=est)


if __name__ == "__main__":
    asyncio.run(_main())
