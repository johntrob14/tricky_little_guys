"""Live per-turn CoT-monitor scorer for run_loop(cot_score=...): runs a rl.monitor monitor (default the ControlArena CoT+action monitor) on each
sampled turn's analysis-channel reasoning and returns {episode_id: [c_turn, ...]}, c_turn in [0,1], one entry per Episode.turns (the turns
rl.cot_mask.cot_token_advantages penalizes). The reasoning shown to the judge is exactly the tokens think_mask marks, so the score and the
penalized tokens agree.

Cost goes to the loop's rl.budget.Budget (worst-case reserve per judge call, counted cost committed as kind "cot_monitor"). Calls run under a
concurrency limit. A judge error or a reply unparsed after the retry gives the turn `fallback` (default 0.0, i.e. no penalty), never silently:
every turn is written to <out_dir>/cot_monitor.jsonl with `fallback` and `reason`, and the counts are in `stats`. When the fallback fraction of a
call passes `max_fallback_frac` the scorer raises CotMonitorError. Truncated turns (stop_reason "length", never penalized) and turns with no
reasoning are not judged and get 0.0, marked `skipped`. The live judge is built only by rl.py behind --allow-live and a finite budget.
"""

import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .budget import Budget, BudgetExceeded
from .cot_mask import _tokenizer_for, think_mask
from .monitor.monitors import DEFAULT, Monitor, get
from .monitor.pricing import price_for
from .monitor.run import MAX_TOKENS, cost, est_tokens, score_item

SCHEMA = "arena.rl.cot_monitor.v1"
KIND = "cot_monitor"
# Same Harmony segment/recipient regexes as rl.provider.parse_turn (copied: importing rl.provider pulls in tinker and registers an Inspect model API).
_SEG = re.compile(r"<\|channel\|>(?P<ch>[^\s<]+)(?P<hdr>(?:(?!<\|message\|>|<\|end\|>|<\|call\|>|<\|return\|>).)*?)(?:<\|message\|>(?P<body>.*?))?(?P<end><\|end\|>|<\|call\|>|<\|return\|>|\Z)", re.S)
_RECIPIENT = re.compile(r"to=(?P<to>[^\s<]+)")
_USER = re.compile(r"<\|start\|>user<\|message\|>(.*?)<\|end\|>", re.S)


class CotMonitorError(RuntimeError):
    pass


# Per-step judge cost for `rl.py train --dry-run`. Expected-case placeholders: the gpt-oss-20b judge on logs/atk-gptoss and logs/atk-qwen (attack runs,
# not training rollouts; 363 turns: 22-24 judged turns per episode, mean 1456-1912 prompt and 440-534 output tokens, 9 retried). Unmeasured on training.
TURNS_EST = 24
PROMPT_TOKENS_EST = 2000
OUT_TOKENS_EST = 600
TASK_TOKENS_WORST = 3000  # task statement shown to the judge; rl.budget.episode_worst_case_usd's p0 (the whole first prompt)
XTOK = 1.5  # policy tokens -> judge tokens (different tokenizers); heuristic margin, unverified


def step_cost(B: int, G: int, *, max_turns: int, max_tokens: int, judge_max_tokens: int, price, monitor: str = DEFAULT, turns_est: int = TURNS_EST,
              prompt_tokens_est: int = PROMPT_TOKENS_EST, out_tokens_est: int = OUT_TOKENS_EST) -> dict:
    """Judge USD per training step. Worst: every turn of every episode judged twice (unparsed retry), prompt = template + TASK_TOKENS_WORST + the whole
    turn (max_tokens x XTOK), output at judge_max_tokens, no cache credit. Expected: min(turns_est, max_turns) calls per episode at the placeholder sizes."""
    n = B * G
    worst_prompt = est_tokens(get(monitor).format("", "", [])) + TASK_TOKENS_WORST + math.ceil(XTOK * max_tokens)
    worst_call = cost(worst_prompt, judge_max_tokens, price)
    exp_turns = min(turns_est, max_turns)
    out_est = min(out_tokens_est, judge_max_tokens)  # the judge cannot emit more than its cap
    exp_call = cost(prompt_tokens_est, out_est, price)
    return dict(calls_worst=n * max_turns * 2, prompt_tokens_worst=worst_prompt, call_worst_usd=round(worst_call, 6), step_worst_case_usd=round(n * max_turns * 2 * worst_call, 4),
                calls_est=n * exp_turns, turns_per_episode_est=exp_turns, prompt_tokens_est=prompt_tokens_est, out_tokens_est=out_est,
                call_est_usd=round(exp_call, 6), step_expected_usd=round(n * exp_turns * exp_call, 4))


def task_text(ep, tok) -> str:
    """The first user message of the episode's first prompt (the task statement), as rl.monitor.run.episode_turns takes it."""
    m = _USER.search(tok.decode(ep.prompt_tokens))
    return m.group(1).strip() if m else ep.task


def turn_views(ep, tok) -> list[dict]:
    """Per Episode.turn: {cot, tool_calls, truncated}. cot is the decode of the think_mask tokens (analysis bodies only); tool_calls are the
    commentary `to=functions.X` calls, args parsed as JSON when possible."""
    out = []
    for t in ep.turns:
        mask = think_mask(t.tokens, tok)
        cot = tok.decode([x for x, m in zip(t.tokens, mask) if m]).strip()
        calls = []
        for m in _SEG.finditer(tok.decode(t.tokens)):  # the provider's own Harmony segment parse, so empty-body and header-variant calls are kept
            to = _RECIPIENT.search(m["hdr"])
            if not to or not to["to"].startswith("functions."):
                continue
            body = (m["body"] or "").strip()
            try:
                args = json.loads(body) if body else {}
            except ValueError:
                args = {"raw": body}
            calls.append({"name": to["to"].removeprefix("functions."), "args": args if isinstance(args, dict) else {"value": args}})
        out.append({"cot": cot, "tool_calls": calls, "truncated": t.stop_reason == "length"})
    return out


def normalize(score: float, monitor: Monitor) -> float:
    """Native score -> suspicion in [0,1] (high = suspicious), whatever the monitor's direction."""
    lo, hi = monitor.scale
    x = min(max((score - lo) / (hi - lo), 0.0), 1.0)
    return x if monitor.direction == "high=suspicious" else 1.0 - x


class _BudgetAdapter:
    """rl.monitor.run.call's reserve/settle on top of rl.budget.Budget's reserve/commit."""

    def __init__(self, budget: Budget, note: str, model: str | None = None):
        self.b, self.note, self.model = budget, note, model

    def reserve(self, amount):
        self.b.reserve(amount)

    def settle(self, reserved, actual):
        if actual > 0:
            self.b.commit(actual, reserved_usd=reserved, kind=KIND, note=self.note, model=self.model)
        else:
            self.b.release(reserved)


@dataclass
class CotMonitorScorer:
    """`await scorer(episodes) -> {episode_id: [c_turn]}`. judge: async prompt -> str | (text, in_tok, out_tok[, thinking])."""
    judge: object
    budget: Budget
    backend: str = "openrouter"
    judge_model: str = "openai/gpt-oss-20b"
    monitor: str = DEFAULT
    concurrency: int = 8
    max_tokens: int = MAX_TOKENS
    fallback: float = 0.0
    max_fallback_frac: float = 0.5
    out_dir: Path | None = None
    tokenizer: object = None  # default: the episode model's tokenizer (offline)
    stats: dict = field(default_factory=lambda: dict(calls=0, turns=0, scored=0, skipped=0, fallback=0, unparsed=0, errors=0, cost_usd=0.0))

    def __post_init__(self):
        if not 0.0 <= self.fallback <= 1.0:
            raise ValueError(f"fallback must be in [0,1], got {self.fallback}")
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.mon = get(self.monitor)
        self.price = price_for(self.backend if self.backend == "tinker" else "openrouter", self.judge_model)  # a fake backend is priced as openrouter

    async def __call__(self, episodes) -> dict[str, list[float]]:
        self.stats["calls"] += 1
        sem = asyncio.Semaphore(self.concurrency)
        rows: list[dict] = []
        out: dict[str, list[float]] = {}

        async def one(ep, k, view, task):
            base = dict(schema=SCHEMA, episode_id=ep.episode_id, step=ep.step, task=ep.task, turn=k, monitor=self.mon.name, backend=self.backend, judge_model=self.judge_model)
            if view["truncated"] or not view["cot"]:
                return dict(base, c=0.0, score=None, skipped="truncated" if view["truncated"] else "empty_cot", fallback=False, reason=None, cost_usd=0.0)
            prompt = self.mon.format(task, view["cot"], view["tool_calls"])
            it = dict(run=ep.run, log=ep.log_path, sample=ep.sample_id, epoch=ep.epoch, task=ep.task, turn=k, cot=view["cot"], prompt=prompt, side={})
            async with sem:
                try:
                    r = await score_item(it, self.mon, self.judge, _BudgetAdapter(self.budget, f"{self.judge_model} ep {ep.episode_id} turn {k}", self.judge_model), self.price, self.max_tokens, self.judge_model)
                except BudgetExceeded:
                    raise
                except Exception as e:  # judge failure: defined fallback, recorded
                    return dict(base, c=self.fallback, score=None, skipped=None, fallback=True, reason=f"judge_error: {type(e).__name__}: {str(e)[:200]}", cost_usd=0.0,
                                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
            row = dict(base, score=r["score"], skipped=None, raw=r["raw"], reasoning=r["reasoning"], attempts=r["attempts"], tokens=r["tokens"], cost_usd=r["cost_usd"],
                       prompt_sha256=r["prompt_sha256"])
            if r["score"] is None:
                return dict(row, c=self.fallback, fallback=True, reason="unparsed")
            return dict(row, c=normalize(r["score"], self.mon), fallback=False, reason=None)

        jobs, keys = [], []
        for ep in episodes:
            tok = self.tokenizer or _tokenizer_for(ep.model or "openai/gpt-oss-120b")
            task = task_text(ep, tok)
            views = turn_views(ep, tok)
            out[ep.episode_id] = [0.0] * len(views)
            for k, v in enumerate(views):
                jobs.append(one(ep, k, v, task))
                keys.append((ep.episode_id, k))
        tasks = [asyncio.ensure_future(j) for j in jobs]
        try:
            res = await asyncio.gather(*tasks)
        except BaseException:  # BudgetExceeded (or cancellation): stop the other judge calls instead of leaving them spending in the background
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for (eid, k), r in zip(keys, res):
            out[eid][k] = r["c"]
            rows.append(r)
        n_judged = sum(not r["skipped"] for r in rows)
        n_fb = sum(r["fallback"] for r in rows)
        s = self.stats
        s["turns"] += len(rows)
        s["skipped"] += sum(bool(r["skipped"]) for r in rows)
        s["scored"] += sum(not r["skipped"] and not r["fallback"] for r in rows)
        s["fallback"] += n_fb
        s["unparsed"] += sum(r["reason"] == "unparsed" for r in rows)
        s["errors"] += sum(bool(r["reason"]) and r["reason"].startswith("judge_error") for r in rows)
        s["cost_usd"] += sum(r["cost_usd"] for r in rows)
        if self.out_dir is not None:
            p = Path(self.out_dir) / "cot_monitor.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:
                for r in rows:
                    f.write(json.dumps(dict(r, time=time.time())) + "\n")
        if n_judged and n_fb / n_judged > self.max_fallback_frac:
            raise CotMonitorError(f"cot monitor fell back on {n_fb}/{n_judged} judged turns (> {self.max_fallback_frac}); first reason: "
                                  f"{next(r['reason'] for r in rows if r['fallback'])}")
        return out


def live_judge(backend: str, model: str, max_tokens: int = MAX_TOKENS):
    """Build the live judge. Only rl.py calls this, behind --allow-live and a finite budget; never in tests."""
    from .monitor import judges
    if backend == "openrouter":
        return judges.openrouter_judge(model, max_tokens, judges.load_openrouter_key())
    if backend == "tinker":
        return judges.tinker_judge(model, max_tokens)
    raise ValueError(f"unknown judge backend {backend!r}")
