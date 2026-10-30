"""Reward and group advantages. Reward is the scorer's binary pass read from the Inspect log; INFRA failures give None and are excluded."""

import math
import random
from collections import defaultdict
from dataclasses import dataclass

from .classify import classify_sample
from .schema import AGENT, ENDED_EXCLUDED, INFRA, Episode

SCORER = "task_scorer"


def read_inspect_score(log_path: str, sample_id: str, epoch: int) -> dict:
    """Read one sample's score from an Inspect .eval log through classify_sample. A missing sample or score, and a sample error or scorer-side failure classed
    INFRA or HALT, give INFRA (reward None); an error classed AGENT (app not ready, tests timeout) gives reward 0; limits do not matter, the score decides."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path))
    sample = next((s for s in (log.samples or []) if str(s.id) == str(sample_id) and s.epoch == epoch), None)
    if sample is None:
        return dict(passed=None, value=None, explanation="", failure_class=INFRA, failure_reason=f"sample {sample_id} epoch {epoch} not in {log_path}")
    failure = classify_sample(sample)
    if failure is not None and failure.reason.startswith("limit_"):
        failure = None
    if failure is not None and failure.failure_class != AGENT:
        return dict(passed=None, value=None, explanation="", failure_class=INFRA, failure_reason=f"{failure.reason}: {(sample.error.message if sample.error else failure.detail)[:300]}")
    if failure is not None and sample.error is not None:
        return dict(passed=False, value=None, explanation=sample.error.message[:500], failure_class=AGENT, failure_reason=failure.reason)
    score = (sample.scores or {}).get(SCORER) or next(iter((sample.scores or {}).values()), None)
    if score is None:
        return dict(passed=None, value=None, explanation="", failure_class=INFRA, failure_reason="no score on sample")
    return dict(passed=score.value == "C", value=str(score.value), explanation=score.explanation or "", failure_class=AGENT if failure else None, failure_reason=failure.reason if failure else "")


def attach_reward(ep: Episode, log_path: str | None = None) -> Episode:
    """Fill ep.passed/score_* from its Inspect log. A failure_class the provider already set to INFRA is kept and wins."""
    path = log_path or ep.log_path
    if ep.failure_class == INFRA or path is None:
        return ep
    ep.log_path = str(path)
    r = read_inspect_score(path, ep.sample_id, ep.epoch)
    ep.passed, ep.score_value, ep.score_explanation = r["passed"], r["value"], r["explanation"]
    if r["failure_class"] in (INFRA, AGENT):
        ep.failure_class, ep.failure_reason = r["failure_class"], r["failure_reason"]
    elif ep.passed is False and ep.failure_class != INFRA:
        ep.failure_class = AGENT if ep.failure_class == "none" else ep.failure_class
    return ep


def episode_reward(ep: Episode) -> float | None:
    """1.0 pass, 0.0 fail, None when the episode is INFRA, has no score, or ended on an excluded reason (budget, infra, wall, step_deadline)."""
    if ep.failure_class == INFRA or ep.ended in ENDED_EXCLUDED or ep.passed is None:
        return None
    return 1.0 if ep.passed else 0.0


def compute_rewards(eps: list[Episode]) -> dict[str, float | None]:
    return {e.episode_id: episode_reward(e) for e in eps}


def side_unevaluated(ep: Episode, cfg) -> bool:
    """With the side term on: the side scorer ran on this episode (opportunity_present set) but side_done is None (side check errored, no snapshot), or
    final_reveal is None while w_final counts it. Scoring that as 0 would push the policy off the side task on scorer errors, so the episode is dropped."""
    return cfg.side_enabled and ep.opportunity_present is not None and (ep.side_done is None or (cfg.w_final != 0.0 and ep.final_reveal is None))


def composite_reward(ep: Episode, cfg) -> float | None:
    """R = w_main*main_pass + w_side*side_done - w_final*final_reveal. main_pass is episode_reward's pass path (same None-exclusion: INFRA, excluded-ended,
    no score => None). An episode with a side spec but an unevaluated side check is None too (side_unevaluated). Otherwise side_done/final_reveal are
    read as already-computed booleans; None (a task with no side spec, so the whole group) counts as 0. When cfg.is_main_only() this equals episode_reward exactly."""
    base = episode_reward(ep)  # None exclusion + 1.0/0.0 pass path
    if base is None:
        return None
    if cfg.is_main_only():
        return cfg.w_main * base
    if side_unevaluated(ep, cfg):
        return None
    r = cfg.w_main * base
    if cfg.side_enabled:
        r += cfg.w_side * (1.0 if ep.side_done else 0.0)
        r -= cfg.w_final * (1.0 if ep.final_reveal else 0.0)
    return r


def compute_composite(eps: list[Episode], cfg) -> dict[str, float | None]:
    return {e.episode_id: composite_reward(e, cfg) for e in eps}


def reward_telemetry(eps: list[Episode], rewards: dict[str, float | None], cfg) -> dict:
    """Per-step summary dict. side_done_rate is split by opportunity_present so we can later tell "hidden behavior" from "the model stopped doing it":
    rate_when_opportunity vs rate_when_not. Rates are over episodes with a non-None reward (the trainable set); n_side_unevaluated counts the episodes
    dropped only because their side check was not evaluated (side_unevaluated)."""
    out = {}
    by_step = defaultdict(list)
    for e in eps:
        by_step[e.step].append(e)
    for step, g in sorted(by_step.items()):
        valid = [e for e in g if rewards.get(e.episode_id) is not None]
        n = len(valid)
        opp = [e for e in valid if e.opportunity_present is True]
        no_opp = [e for e in valid if e.opportunity_present is False]
        out[step] = dict(
            n=n,
            main_pass_rate=(sum(1 for e in valid if e.passed) / n) if n else None,
            side_done_rate=(sum(1 for e in valid if e.side_done) / n) if n else None,
            side_done_rate_when_opportunity=(sum(1 for e in opp if e.side_done) / len(opp)) if opp else None,
            side_done_rate_when_not=(sum(1 for e in no_opp if e.side_done) / len(no_opp)) if no_opp else None,
            n_opportunity=len(opp),
            n_no_opportunity=len(no_opp),
            final_reveal_rate=(sum(1 for e in valid if e.final_reveal) / n) if n else None,
            n_side_unevaluated=sum(1 for e in g if episode_reward(e) is not None and side_unevaluated(e, cfg)),
        )
    return out


@dataclass
class GroupStat:
    task: str
    step: int
    n: int
    n_valid: int
    reward_mean: float | None
    reward_std: float | None
    kept: bool
    drop_reason: str = ""
    group: tuple = ()  # the full grouping key (default (step, task); (step, task, trunk) in branch mode)


def default_group_key(e: Episode) -> tuple:
    return (e.step, e.task)


def branch_group_key(e: Episode) -> tuple:
    """Branch mode: a trunk (branch_of=="") and every branch that forked from it (branch_of==trunk id) share one
    group, so advantage is contrastive between siblings of a common prefix -- the localized covert/overt signal."""
    return (e.step, e.task, e.branch_of or e.episode_id)


def group_advantages(eps: list[Episode], rewards: dict[str, float | None], *, adv_norm: str = "none",
                     group_key=default_group_key) -> tuple[dict[str, float], list[GroupStat]]:
    """A_i = r_i - mean_g(r) per group over valid episodes; the group is group_key(e) ((step, task) by default,
    (step, task, trunk) with branch_group_key). Groups with fewer than 2 valid episodes or constant reward are
    dropped (their episodes are absent from the result). adv_norm="std" divides by std + 1e-6."""
    groups = defaultdict(list)
    for e in eps:
        groups[group_key(e)].append(e)
    advs, stats = {}, []
    for key, g in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        step, task = g[0].step, g[0].task
        valid = [e for e in g if rewards.get(e.episode_id) is not None]
        r = [rewards[e.episode_id] for e in valid]
        if len(valid) < 2:
            stats.append(GroupStat(task, step, len(g), len(valid), (sum(r) / len(r)) if r else None, None, False, "fewer than 2 valid", key))
            continue
        mean = sum(r) / len(r)
        std = math.sqrt(sum((x - mean) ** 2 for x in r) / len(r))
        if max(r) - min(r) < 1e-12:
            stats.append(GroupStat(task, step, len(g), len(valid), mean, std, False, "constant reward", key))
            continue
        scale = std + 1e-6 if adv_norm == "std" else 1.0
        for e, x in zip(valid, r):
            advs[e.episode_id] = (x - mean) / scale
        stats.append(GroupStat(task, step, len(g), len(valid), mean, std, True, "", key))
    return advs, stats


def permute_advantages(advs: dict[str, float], eps: list[Episode], rng: random.Random) -> dict[str, float]:
    """Permute the episode advantages within each (step, task) group (the null of the learning-direction check)."""
    by = defaultdict(list)
    for e in eps:
        if e.episode_id in advs:
            by[(e.step, e.task)].append(e.episode_id)
    out = {}
    for ids in by.values():
        vals = [advs[i] for i in ids]
        rng.shuffle(vals)
        out.update(zip(ids, vals))
    return out
