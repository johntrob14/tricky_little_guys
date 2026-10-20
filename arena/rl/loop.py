"""Training loop: pick tasks, collect episodes through a pluggable `collect`, build merged datums, train, checkpoint, halt rules, resume.

`collect` is supplied by the integration package (Inspect-based); it must sample with `req.sampler` (the current weights) and return Episodes stamped with
`req.weights_id`. Rewards are read from each episode's Inspect log by the loop (rewards.attach_reward) unless `passed` is already set.
Files under cfg.out_dir: steps.jsonl, checkpoints.json, HALTED.md, step_NNN/{episodes.jsonl, collected.json, trained.json, batch.json}.
"""

import asyncio
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from .budget import Budget, BudgetExceeded, sample_cost_usd
from .classify import Failure, HaltRules as InfraRules
from .cot_monitor import CotMonitorError
from .datums import CTX_LIMIT, Refused, build, write_batch
from .rewardcfg import RewardCfg
from .rewards import attach_reward, compute_composite, compute_rewards, reward_telemetry
from .schema import INFRA, TOOL_FAIL_FLAGS, Episode, append_jsonl, read_jsonl
from .train import StepMetrics, TrainHP, next_sampler, open_training_client, save_checkpoint, train_step

DAY = 86400
CONTEXT_ENDS = {"context", "model_length", "truncated", "wall"}


class HaltError(RuntimeError):
    pass


@dataclass
class CollectRequest:
    step: int
    tasks: list[str]
    n: int  # episodes (epochs) per task
    sampler: Any
    weights_id: str
    out_dir: Path
    deadline_s: float | None
    budget: Budget


Collect = Callable[[CollectRequest], Awaitable[list[Episode]]]


@dataclass
class HaltRules:
    invalid_frac: float = 0.20
    tool_fail_rate: float = 0.30
    empty_reasoning_rate: float = 0.10
    truncated_rate: float = 0.10
    context_end_frac: float = 0.30
    zero_signal_steps: int = 2
    lp_gap_abs: float = 0.10
    lp_gap_warn: float = 0.03
    gate_direction: bool = False
    flag_frac: float = 0.10
    baseline_reward: float | None = None
    baseline_drop: float = 0.3
    baseline_steps: int = 2
    spend_frac: float = 0.8
    consecutive_infra: int = 3
    infra_window: int = 8
    infra_window_frac: float = 0.25
    _infra: InfraRules = field(default_factory=InfraRules)
    _zero: int = 0
    _below: int = 0

    def observe(self, eps: list[Episode]) -> None:
        self._infra.consecutive, self._infra.window, self._infra.max_rate = self.consecutive_infra, self.infra_window, self.infra_window_frac
        self._infra.recent = deque(self._infra.recent, maxlen=self.infra_window)
        for e in eps:
            self._infra.record(Failure(INFRA, e.failure_reason) if e.failure_class == INFRA else None)

    def check(self, line: dict) -> str | None:
        if self._infra.run >= self.consecutive_infra:
            return f"{self._infra.run} consecutive INFRA failures"
        w = self._infra.recent
        if len(w) >= self.infra_window and sum(w) / len(w) > self.infra_window_frac:
            return f"INFRA rate {sum(w)}/{len(w)} over the last {len(w)} episodes"
        n = max(line["n_episodes"], 1)
        if line["n_none"] / n > self.invalid_frac:
            return f"invalid episodes {line['n_none']}/{n} above {self.invalid_frac:.0%}"
        r = line["rates"]
        for key, lim, name in (("tool_fail", self.tool_fail_rate, "tool-call failures"), ("empty_reasoning", self.empty_reasoning_rate, "empty-reasoning turns"),
                               ("truncated", self.truncated_rate, "truncated turns")):
            if r[key] > lim:
                return f"{name} {r[key]:.1%} of turns above {lim:.0%}"
        if r["context_end"] > self.context_end_frac:
            return f"context/truncated/wall endings {r['context_end']:.1%} of episodes above {self.context_end_frac:.0%}"
        if line["flag_rate"] > self.flag_frac:
            return f"flagged episodes {line['flag_rate']:.1%} of valid above {self.flag_frac:.0%}"
        self._zero = self._zero + 1 if line["groups_with_signal"] == 0 else 0
        if self._zero >= self.zero_signal_steps:
            return f"groups_with_signal == 0 for {self._zero} consecutive steps"
        g = line.get("lp_gap_mean")
        if g is not None and abs(g) > self.lp_gap_abs:
            return f"|lp_gap_mean| {abs(g):.4f} above {self.lp_gap_abs}"
        if g is not None and abs(g) > self.lp_gap_warn:
            line.setdefault("warnings", []).append(f"|lp_gap_mean| {abs(g):.4f} above {self.lp_gap_warn}")
        if self.gate_direction and line.get("direction_gain") is not None and line["direction_gain"] <= 0:
            return f"learning-direction gain {line['direction_gain']:.4g} not above 0"
        if self.baseline_reward is not None and line["reward_mean"] is not None:
            self._below = self._below + 1 if line["reward_mean"] < self.baseline_reward - self.baseline_drop else 0
            if self._below >= self.baseline_steps:
                return f"mean train reward {line['reward_mean']:.2f} more than {self.baseline_drop} below baseline {self.baseline_reward:.2f} for {self._below} steps"
        if line["spent_frac"] > self.spend_frac and line["done_frac"] < self.spend_frac:
            return f"spend {line['spent_frac']:.0%} of the cap at {line['done_frac']:.0%} of the steps"
        return None


@dataclass
class LoopCfg:
    run: str
    out_dir: Path
    train_tasks: list[str]
    steps: int
    B: int
    G: int
    ep_worst_usd: float  # per-episode worst case reserved for rollouts and charged in full if collect fails (budget.episode_worst_case_usd)
    model: str = "openai/gpt-oss-120b"
    seed: int = 0
    hp: TrainHP = field(default_factory=TrainHP)
    reward_cfg: RewardCfg = field(default_factory=RewardCfg)  # default is main-only; side/cot/final weights are 0, so the loop is unchanged unless turned on
    adv_norm: str = "none"
    max_len: int = CTX_LIMIT
    check_direction: bool = False  # on step 0 only
    state_every: int = 2
    state_ttl_s: int = 3 * DAY
    final_state_ttl_s: int = 7 * DAY
    final_sampler_ttl_s: int = 30 * DAY
    step_deadline_s: float | None = None
    ep_usd_est: float = 0.03  # before the first step the forecast uses 1.5x these, afterwards the running mean
    train_usd_est: float = 0.02
    halt: HaltRules = field(default_factory=HaltRules)


def pick_tasks(train_active: list[str], B: int, step: int, seed: int) -> list[str]:
    """Shuffled cycle over train_active, B per step, no repeat inside a step unless B exceeds the number of tasks."""
    n = len(train_active)
    out, p = [], step * B
    while len(out) < B:
        if len(set(out)) == n:
            out.append(train_active[len(out) % n])
            continue
        order = list(train_active)
        random.Random(f"{seed}:{p // n}").shuffle(order)
        t = order[p % n]
        p += 1
        if t not in out:
            out.append(t)
    return out


def forecast_step(cfg: LoopCfg, ema_ep_usd: float | None, ema_train_usd: float | None) -> float:
    n = cfg.B * cfg.G
    return n * ((ema_ep_usd if ema_ep_usd is not None else 1.5 * cfg.ep_usd_est) + (ema_train_usd if ema_train_usd is not None else 1.5 * cfg.train_usd_est))


def initial_weights_id(cfg: LoopCfg) -> str:
    return "w0-" + hashlib.sha1(f"{cfg.run}|{cfg.model}|{cfg.hp.rank}|{cfg.hp.seed}".encode()).hexdigest()[:10]


def next_weights_id(wid: str, step: int, episode_ids: list[str]) -> str:
    return "w-" + hashlib.sha1(f"{wid}|{step}|{','.join(sorted(episode_ids))}".encode()).hexdigest()[:10]


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def episode_sample_usd(e: Episode, model: str) -> float:
    u = e.usage or {}
    return sample_cost_usd(model, u.get("prompt_tokens", 0), u.get("cached_tokens", 0), u.get("completion_tokens", 0)) + e.wasted_usd


def step_line(cfg: LoopCfg, step: int, chosen: list[str], eps: list[Episode], rewards: dict, batch, m: StepMetrics, extra: dict) -> dict:
    n_exp = cfg.B * cfg.G
    valid = [e for e in eps if rewards.get(e.episode_id) is not None]
    turns = [t for e in eps if e.failure_class != INFRA for t in e.turns]
    nt = max(len(turns), 1)
    by_task = defaultdict(lambda: [0, 0])
    for e in valid:
        by_task[e.task][1] += 1
        by_task[e.task][0] += int(rewards[e.episode_id] > 0.5)
    ne = max(len(eps), 1)
    return dict(
        schema="arena.rl.step.v1", step=step, tasks=chosen, G=cfg.G, n_episodes=n_exp, n_collected=len(eps), n_valid=len(valid), n_none=n_exp - len(valid),
        n_infra=sum(e.failure_class == INFRA for e in eps), pass_rate=(sum(rewards[e.episode_id] for e in valid) / len(valid)) if valid else None,
        reward_mean=(sum(rewards[e.episode_id] for e in valid) / len(valid)) if valid else None, reward_by_task={k: v for k, v in by_task.items()},
        groups=len(batch.groups), groups_with_signal=sum(g.kept for g in batch.groups), ended=dict(Counter(e.ended for e in eps)),
        turns_mean=sum(len(e.turns) for e in eps) / ne, n_datums=len(batch.datums), n_train_tokens=batch.tokens_total, n_sampled_tokens=batch.tokens_sampled,
        n_truncated_turns_zeroed=sum(d.truncated_turns for d in batch.datums), applied=m.applied if batch.datums else False,
        loss_sum=m.loss_sum, lp_gap_mean=m.lp_gap_mean, lp_gap_abs_p95=m.lp_gap_abs_p95, lp_gap_frac_gt_0_7=m.lp_gap_frac_gt_0_7,
        direction_gain=m.direction_gain, direction_null_mean=m.direction_null_mean, direction_null_sd=m.direction_null_sd, router=m.router, optim_metrics=m.optim_metrics,
        rates=dict(empty_reasoning=sum("empty_reasoning" in t.flags for t in turns) / nt, truncated=sum(t.stop_reason == "length" for t in turns) / nt,
                   tool_fail=sum(bool(TOOL_FAIL_FLAGS & set(t.flags)) for t in turns) / nt, context_end=sum(e.ended in CONTEXT_ENDS for e in eps) / ne),
        flag_rate=(sum(any(f in ("scope_violation", "runner_tampered", "runner_rigged") for t in e.turns for f in t.flags) for e in valid) / len(valid)) if valid else 0.0,
        train_usd=m.train_usd, wasted_usd=sum(e.wasted_usd for e in eps), **extra)


async def run_loop(cfg: LoopCfg, svc, collect: Collect, *, budget: Budget, resume: bool = True, start_step: int | None = None, base_model: str | None = None,
                   cot_score: Callable[[list[Episode]], Awaitable[dict[str, list[float]]]] | None = None) -> dict:
    """Run steps start..cfg.steps-1. Resume step = last saved state's step + 1 (checkpoints.json); any other start_step is refused. A HALT writes
    HALTED.md and raises HaltError (BudgetExceeded included); everything stays resumable from the last state."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpts = _read_json(out / "checkpoints.json", []) if resume else []
    states = [c for c in ckpts if c["kind"] == "state"]
    last = max(states, key=lambda c: c["step"]) if states else None
    derived = last["step"] + 1 if last else 0
    if start_step is not None and start_step != derived:
        raise ValueError(f"start_step {start_step} refused: the last saved state is step {last['step'] if last else None}, resume step is {derived}")
    start = derived
    wid, n_applied = (last["weights_id"], last["n_applied"]) if last else (initial_weights_id(cfg), 0)
    session = hashlib.sha1(os.urandom(8)).hexdigest()[:4]
    for sd in sorted(out.glob("step_*")):  # a marker at or after the resume step describes an update the reloaded weights do not contain
        if int(sd.name.split("_")[1]) >= start and (sd / "trained.json").exists():
            os.replace(sd / "trained.json", sd / "trained.json.stale")
        if int(sd.name.split("_")[1]) > start:  # rows after the resume step were sampled by weights that only a lost update chain could reproduce
            for f in ("collected.json", "episodes.jsonl"):
                if (sd / f).exists():
                    os.replace(sd / f, sd / (f + ".stale"))
    tc = await open_training_client(svc, cfg.hp, base_model or cfg.model, last["path"] if last else None)
    sampler, sampler_wid = None, None
    cfg.halt._infra, cfg.halt._zero, cfg.halt._below = InfraRules(), 0, 0
    ema_ep = ema_train = None
    saved_n = last["n_applied"] if last else 0
    history = []

    def halt(reason: str, step: int):
        (out / "HALTED.md").write_text(f"# HALTED at step {step}\n\n{reason}\n\nbudget: {json.dumps(budget.summary())}\n\nResume with run_loop(resume=True): it restarts from the last saved state.\n")
        raise HaltError(reason)

    if not budget.started:
        try:
            budget.start_stage(forecast_step(cfg, None, None), note=f"run_loop {cfg.run} first-step forecast, steps {start}..{cfg.steps - 1}")
        except BudgetExceeded as e:
            halt(f"BudgetExceeded before step {start}: {e}", start)

    for step in range(start, cfg.steps):
        t0 = time.time()
        sd = out / f"step_{step:03d}"
        sd.mkdir(exist_ok=True)
        chosen = pick_tasks(cfg.train_tasks, cfg.B, step, cfg.seed)
        marker = _read_json(sd / "collected.json")
        reused = bool(marker and marker.get("complete") and marker["weights_id"] == wid and marker["tasks"] == chosen)
        if reused:
            eps = read_jsonl(sd / "episodes.jsonl")
        else:
            for f in ("collected.json", "episodes.jsonl"):
                if (sd / f).exists():
                    os.replace(sd / f, sd / (f + ".stale"))
            try:
                reserve = cfg.ep_worst_usd * cfg.B * cfg.G
                budget.require(forecast_step(cfg, ema_ep, ema_train) + reserve)
                budget.check_total(reserve)
                budget.reserve(reserve)
            except BudgetExceeded as e:
                halt(f"BudgetExceeded before step {step}: {e}", step)
            if sampler is None or sampler_wid != wid:
                sampler, sampler_wid = await next_sampler(tc), wid
            req = CollectRequest(step, chosen, cfg.G, sampler, wid, sd, cfg.step_deadline_s, budget)
            try:
                eps = list(await collect(req))
            except BaseException:
                try:  # spend of a failed collect is unknown: count the reservation as the upper bound
                    budget.commit(reserve, reserved_usd=reserve, note=f"step {step} collect failed, worst-case bound")
                except BudgetExceeded:
                    pass
                raise
            budget.release(reserve)
            eps = [attach_reward(e) if e.passed is None else e for e in eps]
            append_jsonl(sd / "episodes.jsonl", eps)
            spent = sum(episode_sample_usd(e, cfg.model) for e in eps)
            try:
                budget.commit(spent, kind="sample", note=f"step {step} {len(eps)} episodes")
            except BudgetExceeded as e:
                _write_json(sd / "collected.json", dict(step=step, weights_id=wid, tasks=chosen, episode_ids=[x.episode_id for x in eps], complete=True, sampling_usd=spent))
                halt(f"BudgetExceeded after step {step} rollouts: {e}", step)
            _write_json(sd / "collected.json", dict(step=step, weights_id=wid, tasks=chosen, episode_ids=[x.episode_id for x in eps], complete=True, sampling_usd=spent))
            ema_ep = spent / max(cfg.B * cfg.G, 1) if ema_ep is None else 0.5 * ema_ep + 0.5 * spent / max(cfg.B * cfg.G, 1)
        mismatched = [e for e in eps if e.weights_id != wid]
        eps = [e for e in eps if e.weights_id == wid]  # C2: never train on rows another weights sampled
        cfg.halt.observe(eps)
        main_only = cfg.reward_cfg.is_main_only()
        rewards = compute_rewards(eps) if main_only else compute_composite(eps, cfg.reward_cfg)
        cot = None  # analysis-channel CoT penalty, only when the organism's CoT term is on and a per-turn scorer is supplied
        if cfg.reward_cfg.cot_penalty_enabled and cfg.reward_cfg.beta_cot and cot_score is not None:
            try:
                cot = (cfg.reward_cfg.beta_cot, await cot_score(eps))
            except BudgetExceeded as e:
                halt(f"BudgetExceeded in CoT monitor at step {step}: {e}", step)
            except CotMonitorError as e:
                halt(f"CoT monitor failed at step {step}: {e}", step)
        try:
            batch = build(eps, rewards, adv_norm=cfg.adv_norm, allowed_tasks=set(cfg.train_tasks), max_len=cfg.max_len, expect_weights_id=wid, soft=True, cot_penalty=cot)
        except Refused as e:
            halt(f"refused episode at step {step}: {e}", step)
        write_batch(batch, sd)
        wid_after = next_weights_id(wid, step, [d.episode_id for d in batch.datums])

        def mark():
            _write_json(sd / "trained.json", dict(step=step, episode_ids=[d.episode_id for d in batch.datums], optim_step_applied=True, weights_id_before=wid,
                                                  weights_id_after=wid_after, n_applied_after=n_applied + 1, time=time.time()))

        try:
            m = await train_step(tc, batch, cfg.hp, budget=budget, model=cfg.model, check_direction=cfg.check_direction and step == 0, on_applied=mark)
        except BudgetExceeded as e:
            halt(f"BudgetExceeded in training step {step}: {e}", step)
        if m.applied:
            wid, n_applied = wid_after, n_applied + 1
            ema_train = m.train_usd / max(cfg.B * cfg.G, 1)
        ck = None
        final = step == cfg.steps - 1
        meta = dict(step=step, weights_id=wid, n_applied=n_applied, run=cfg.run, model=cfg.model, rank=cfg.hp.rank, lr=cfg.hp.learning_rate)
        if n_applied > saved_n and (final or (step + 1) % cfg.state_every == 0):
            ck = await save_checkpoint(tc, "state", f"{cfg.run}-s{step:03d}-{session}-state", cfg.final_state_ttl_s if final else cfg.state_ttl_s, **meta)
            ckpts.append(ck)
            saved_n = n_applied
            _write_json(out / "checkpoints.json", ckpts)
        if final and n_applied > 0:  # the deliverable sampler is saved even when the last steps applied no update (the latest state then already holds these weights)
            ckpts.append(await save_checkpoint(tc, "sampler", f"{cfg.run}-s{step:03d}-{session}", cfg.final_sampler_ttl_s, **meta))
            _write_json(out / "checkpoints.json", ckpts)
        tele = {} if main_only else dict(reward_telemetry=reward_telemetry(eps, rewards, cfg.reward_cfg), cot_penalty_on=cot is not None)
        line = step_line(cfg, step, chosen, eps, rewards, batch, m, dict(
            weights_id_sampled=marker["weights_id"] if reused else sampler_wid, weights_id_after=wid, n_applied=n_applied, reused_rows=reused, dropped_other_weights=len(mismatched),
            ckpt=ck["path"] if ck else None, cost=budget.summary(), spent_frac=budget.committed / budget.cap if budget.cap else 0.0, done_frac=(step + 1) / cfg.steps,
            seconds=time.time() - t0, warnings=[], stop=None, **tele))
        reason = cfg.halt.check(line)
        line["stop"] = reason
        with open(out / "steps.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
            f.flush()
            os.fsync(f.fileno())
        history.append(line)
        if reason:
            halt(reason, step)
    return dict(run=cfg.run, steps_run=len(history), start_step=start, weights_id=wid, n_applied=n_applied, budget=budget.summary(), checkpoints=ckpts)
