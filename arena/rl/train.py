"""One optimizer step on a Batch: forward_backward (importance_sampling) then optim_step, lp_gap diagnostic, learning-direction check, checkpoints."""

import asyncio
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tinker import types

from .budget import Budget, train_cost_usd
from .datums import Batch

LOSS = "importance_sampling"
TTL_MIN, TTL_MAX = 3600, 10 * 365 * 86400


@dataclass
class TrainHP:
    learning_rate: float = 4e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0
    rank: int = 32
    seed: int = 0


def adam_params(hp: TrainHP) -> types.AdamParams:
    return types.AdamParams(learning_rate=hp.learning_rate, beta1=hp.beta1, beta2=hp.beta2, eps=hp.eps, weight_decay=hp.weight_decay, grad_clip_norm=hp.grad_clip_norm)


@dataclass
class StepMetrics:
    applied: bool = False  # optim_step was awaited
    n_datums: int = 0
    n_positions: int = 0  # positions with advantage != 0
    tokens_total: int = 0
    loss_sum: float | None = None
    lp_gap_mean: float | None = None
    lp_gap_abs_p95: float | None = None
    lp_gap_frac_gt_0_7: float | None = None
    direction_gain: float | None = None
    direction_null_mean: float | None = None
    direction_null_sd: float | None = None
    direction_ok: bool | None = None
    router: dict = field(default_factory=dict)
    optim_metrics: dict = field(default_factory=dict)
    train_usd: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _list(x) -> list[float]:
    return [float(v) for v in (x.tolist() if hasattr(x, "tolist") else list(x))]


async def _result(fut):
    return await fut.result_async()


async def next_sampler(tc):
    """Sampling client for the CURRENT weights. The SDK's async variant blocks the event loop on a synchronous .result(), so run the sync one in a thread (C10e)."""
    return await asyncio.to_thread(tc.save_weights_and_get_sampling_client)


def lp_gap(batch: Batch, trainer_lps: list[list[float]]) -> dict:
    """trainer_logprob - sampler_logprob at positions with advantage != 0, over the whole batch."""
    gaps = []
    for d, tl in zip(batch.datums, trainer_lps):
        f = d.fields
        if len(tl) != len(f["logprobs"]):
            raise ValueError(f"episode {d.episode_id}: trainer returned {len(tl)} logprobs for {len(f['logprobs'])} positions")
        gaps += [t - s for t, s, a in zip(tl, f["logprobs"], f["advantages"]) if a != 0.0]
    if not gaps:
        return dict(mean=None, abs_p95=None, frac_gt=None, n=0)
    ab = sorted(abs(g) for g in gaps)
    return dict(mean=sum(gaps) / len(gaps), abs_p95=ab[min(len(ab) - 1, int(0.95 * len(ab)))], frac_gt=sum(a > 0.7 for a in ab) / len(ab), n=len(gaps))


def _episode_deltas(batch: Batch, before: list[list[float]], after: list[list[float]]) -> list[tuple[float, int]]:
    out = []
    for d, b, a in zip(batch.datums, before, after):
        idx = [j for j, x in enumerate(d.fields["advantages"]) if x != 0.0]
        out.append((sum(a[j] - b[j] for j in idx), len(idx)))
    return out


def direction_gain(batch: Batch, before: list[list[float]], after: list[list[float]], *, n_perm: int = 200, seed: int = 0) -> dict:
    """gain = sum_j A_j (lp_after_j - lp_before_j) / sum_j |A_j| over trained positions; null = the same with episode advantages permuted within task groups."""
    s = _episode_deltas(batch, before, after)
    adv = [d.advantage for d in batch.datums]

    def gain(a):
        den = sum(abs(x) * n for x, (_, n) in zip(a, s))
        return sum(x * ds for x, (ds, _) in zip(a, s)) / den if den else 0.0

    rng, groups = random.Random(seed), {}
    for i, d in enumerate(batch.datums):
        groups.setdefault(d.task, []).append(i)
    nulls = []
    for _ in range(n_perm):
        a = list(adv)
        for ids in groups.values():
            vals = [adv[i] for i in ids]
            rng.shuffle(vals)
            for i, v in zip(ids, vals):
                a[i] = v
        nulls.append(gain(a))
    m = sum(nulls) / len(nulls)
    sd = (sum((x - m) ** 2 for x in nulls) / len(nulls)) ** 0.5
    g = gain(adv)
    return dict(gain=g, null_mean=m, null_sd=sd, ok=g > m + 2 * sd and g > 0)


async def train_step(tc, batch: Batch, hp: TrainHP, *, budget: Budget | None = None, model: str = "openai/gpt-oss-120b", check_direction: bool = False,
                     on_applied=None) -> StepMetrics:
    """Zero datums: no call is made and applied is False. Else reserve the worst-case train cost (forwards count too), then run forward_backward and
    optim_step back to back, await both (on_applied() fires right after), and commit the counted cost. lp_before for the direction check comes from forward_async BEFORE the step (C3).
    If the step raises, every pass already submitted to Tinker is counted at its full cost (its spend is unknown, the submission is the upper bound)."""
    m = StepMetrics(n_datums=len(batch.datums), tokens_total=batch.tokens_total)
    if not batch.datums:
        return m
    t0 = time.time()
    data = batch.tinker_data()
    passes = 3 if check_direction else 1
    worst = train_cost_usd(model, batch.tokens_total) * passes
    if budget:
        budget.reserve(worst)
    actual, submitted, per_pass = 0.0, 0, train_cost_usd(model, batch.tokens_total)
    try:
        before = None
        if check_direction:
            submitted += 1
            out = await _result(await tc.forward_async(data, LOSS))
            before = [_list(o["logprobs"]) for o in out.loss_fn_outputs]
            actual += train_cost_usd(model, batch.tokens_total)
        submitted += 1
        fb_f = await tc.forward_backward_async(data, LOSS)
        opt_f = await tc.optim_step_async(adam_params(hp))
        fb, opt = await _result(fb_f), await _result(opt_f)
        m.applied = True
        if on_applied:
            on_applied()  # called right after optim_step is awaited (the trained.json marker)
        actual += train_cost_usd(model, batch.tokens_total)
        trainer = [_list(o["logprobs"]) for o in fb.loss_fn_outputs]
        g = lp_gap(batch, trainer)
        m.n_positions, m.lp_gap_mean, m.lp_gap_abs_p95, m.lp_gap_frac_gt_0_7 = g["n"], g["mean"], g["abs_p95"], g["frac_gt"]
        fbm = dict(fb.metrics or {})
        m.loss_sum = fbm.get("loss:sum")
        m.router = {k: v for k, v in fbm.items() if k.startswith("e_")}
        m.optim_metrics = dict(getattr(opt, "metrics", None) or {})
        if check_direction:
            submitted += 1
            out = await _result(await tc.forward_async(data, LOSS))
            after = [_list(o["logprobs"]) for o in out.loss_fn_outputs]
            actual += train_cost_usd(model, batch.tokens_total)
            d = direction_gain(batch, before, after, seed=hp.seed)
            m.direction_gain, m.direction_null_mean, m.direction_null_sd, m.direction_ok = d["gain"], d["null_mean"], d["null_sd"], d["ok"]
    except BaseException:
        actual = max(actual, submitted * per_pass)
        raise
    finally:
        m.train_usd, m.seconds = actual, time.time() - t0
        if budget:
            budget.commit(actual, reserved_usd=worst, kind="train", note=f"{batch.tokens_total} tokens")
    return m


def check_ttl(ttl_s: int) -> int:
    if ttl_s is None or not (TTL_MIN <= ttl_s <= TTL_MAX):
        raise ValueError(f"ttl_s must be finite and between {TTL_MIN} and {TTL_MAX}, got {ttl_s!r}")
    return int(ttl_s)


async def save_checkpoint(tc, kind: str, name: str, ttl_s: int, **meta) -> dict:
    """kind 'sampler' (weights for sampling) or 'state' (weights + optimizer, resumable). Returns a checkpoints.json row."""
    ttl_s = check_ttl(ttl_s)
    if kind == "state":
        r = await _result(await tc.save_state_async(name, ttl_seconds=ttl_s, overwrite=False))
    elif kind == "sampler":
        r = await _result(await tc.save_weights_for_sampler_async(name, ttl_seconds=ttl_s))
    else:
        raise ValueError(kind)
    return dict(name=name, kind=kind, path=r.path, ttl_s=ttl_s, expires_at=time.time() + ttl_s, created_at=time.time(), **meta)


async def open_training_client(svc, hp: TrainHP, base_model: str, from_state: str | None = None):
    """A fresh training client, or one resumed from a saved state with its optimizer. Every call needs its own ServiceClient (a failed load poisons one)."""
    if from_state:
        return await svc.create_training_client_from_state_with_optimizer_async(from_state)
    return await svc.create_lora_training_client_async(base_model, rank=hp.rank, seed=hp.seed)


def new_service(run: str, stage: str):
    """Real ServiceClient, key from arena/.env as in run.py. Not used by any test."""
    import tinker
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    return tinker.ServiceClient(user_metadata={"project": "arena-rl", "run": run, "stage": stage})
