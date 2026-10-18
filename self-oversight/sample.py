"""Tinker sampling runner: conversations in, parsed rollouts out, with retries, a concurrency cap and a budget.

    runner = Runner(get_model("gpt-oss-20b"), budget_usd=2.0)
    out = await runner.sample(messages, n=2, max_tokens=4096)           # list[Rollout]
    out2 = await runner.continue_(messages, out[0], "Why did you ...?")  # one follow-up turn
    runner.to_ledger("screen", "gpt-oss-20b-unprompted")                 # append the counted cost

The budget is checked before each request is sent, against the token-count cost of everything returned so
far plus a worst-case reservation for requests in flight (prompt at prefill price, n x max_tokens at
sample price). A run therefore cannot overshoot its cap by more than the requests already in flight.
Prices come from the table in common.py and are charged at list price (list_multiplier) by default, as
the plan says to budget discounted models at list.

Retries: transient Tinker failures (connection, timeout, 408/409/429/5xx, request-failed) are retried
with exponential backoff and jitter, up to `max_retries` times per request. Renderer errors (wrong stop
tokens) and budget errors are not retried. A truncated or malformed completion is not an error: it is
returned as a Rollout with `clean=False` so the caller can count it.

Parsing: the cookbook renderer for the model splits the completion into thinking (analysis channel or
<think> block) and answer. If the renderer reports the completion as malformed (usually hit max_tokens
mid-thought), a text fallback recovers whatever thinking and answer text is present.

Smoke test (costs a fraction of a cent):

    uv run python sample.py --model gpt-oss-20b --prompt "What is 17*23?" -n 2 --max-tokens 400 --budget 0.02
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from common import Model, get_model, load_key, renderer_for

# Thinking delimiters seen in raw text when the renderer cannot parse a truncated completion.
_HARMONY = re.compile(r"<\|channel\|>analysis<\|message\|>(?P<think>.*?)(?:<\|end\|>|$)"
                      r"(?:.*?<\|channel\|>final<\|message\|>(?P<answer>.*))?", re.S)
_THINK = re.compile(r"^\s*<think>(?P<think>.*?)(?:</think>(?P<answer>.*)|$)", re.S)


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Rollout:
    thinking: str
    answer: str
    raw: str                 # decoded completion text, stop tokens stripped by the renderer where possible
    tokens: list[int] = field(default_factory=list, repr=False)
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    stop_reason: str = ""    # "stop" or "length" from Tinker
    termination: str = ""    # renderer's ParseTermination: stop_sequence, eos, malformed
    clean: bool = True       # termination is not malformed
    cost_usd: float = 0.0
    attempts: int = 1
    seconds: float = 0.0
    message: dict = field(default_factory=dict, repr=False)  # the renderer's parsed Message, for follow-ups

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


def is_retryable(exc: BaseException) -> bool:
    import tinker

    if isinstance(exc, (tinker.APIConnectionError, tinker.APITimeoutError, tinker.RateLimitError,
                        tinker.InternalServerError, tinker.RequestFailedError)):
        return True
    if isinstance(exc, tinker.APIStatusError):
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    return isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError))


class Runner:
    def __init__(self, model: Model, budget_usd: float, *, model_path: str | None = None,
                 max_concurrency: int = 16, max_retries: int = 5, backoff_base: float = 2.0,
                 backoff_max: float = 60.0, temperature: float = 1.0, top_p: float = 1.0,
                 at_list_price: bool = True, request_timeout: float = 600.0,
                 sampler: Any = None, renderer: Any = None, log: Callable[[str], None] | None = print):
        """`sampler` and `renderer` can be injected for tests; otherwise they are built from `model`.

        `model_path` samples a fine-tuned checkpoint (tinker://...) instead of the base model; prices
        are the base model's.
        """
        self.model = model
        self.model_path = model_path
        self.budget = Budget(budget_usd)
        self.sem = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.temperature = temperature
        self.top_p = top_p
        self.price_mult = model.list_multiplier if at_list_price else 1.0
        self.request_timeout = request_timeout
        self.log = log or (lambda s: None)
        self._sampler = sampler
        self._renderer = renderer
        self.errors: list[dict] = []   # every failed attempt, for the report

    # -- lazy Tinker setup -------------------------------------------------------------------------

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

    # -- pricing -----------------------------------------------------------------------------------

    def cost(self, prompt: int, cached: int, completion: int) -> float:
        m = self.model
        usd = ((prompt - cached) * m.prefill + cached * m.cached_prefill + completion * m.sample) / 1e6
        return usd * self.price_mult

    def worst_case(self, prompt: int, n: int, max_tokens: int) -> float:
        return self.cost(prompt, 0, n * max_tokens)

    # -- sampling ----------------------------------------------------------------------------------

    async def sample(self, messages: list[dict], n: int = 1, max_tokens: int = 2048,
                     seed: int | None = None, prefill: str | None = None) -> list[Rollout]:
        """n completions of the next assistant turn after `messages`."""
        import tinker

        prompt = self.renderer.build_generation_prompt(messages, prefill=prefill)
        n_prompt = prompt.length
        params = tinker.SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens=max_tokens,
                                       stop=self.renderer.get_stop_sequences(), seed=seed)
        reserve = self.worst_case(n_prompt, n, max_tokens)
        self.budget.reserve(reserve)
        t0 = time.monotonic()
        try:
            async with self.sem:
                resp, attempts = await self._sample_with_retries(prompt, n, params)
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
        channel of previous turns; <think> renderers usually drop earlier thinking too).
        """
        history = list(messages) + [rollout.message or {"role": "assistant", "content": rollout.answer},
                                    {"role": "user", "content": user_text}]
        return await self.sample(history, n=n, max_tokens=max_tokens, seed=seed)

    async def _sample_with_retries(self, prompt, n: int, params) -> tuple[Any, int]:
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await asyncio.wait_for(self.sampler.sample_async(prompt, n, params), self.request_timeout)
                return resp, attempt
            except Exception as e:  # noqa: BLE001 - classified below
                retryable = is_retryable(e)
                self.errors.append({"attempt": attempt, "error": repr(e)[:500], "retryable": retryable})
                if not retryable or attempt > self.max_retries:
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
        import ledger

        b = self.budget
        ledger.append(phase, run, self.model.tinker_id, "sample", estimate_usd=estimate_usd,
                      counted_usd=b.spent_usd,
                      note=note or f"{b.requests} req, {b.prompt_tokens} prompt ({b.cached_prompt_tokens} cached), "
                                   f"{b.completion_tokens} completion tokens, {len(self.errors)} failed attempts")


async def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--model-path", default=None, help="tinker:// checkpoint instead of the base model")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--budget", type=float, default=0.05, help="USD cap for this invocation")
    ap.add_argument("--follow-up", default=None, help="a second user turn after the first sample")
    ap.add_argument("--ledger", default=None, help="phase:run to record the spend under")
    args = ap.parse_args()

    runner = Runner(get_model(args.model), args.budget, model_path=args.model_path)
    messages = ([{"role": "system", "content": args.system}] if args.system else []) + \
               [{"role": "user", "content": args.prompt}]
    est = runner.worst_case(runner.renderer.build_generation_prompt(messages).length, args.n, args.max_tokens)
    print(f"worst case for the first request: ${est:.4f}")
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
