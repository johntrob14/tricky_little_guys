"""Reward and group advantages. Reward is the scorer's binary pass read from the Inspect log; INFRA failures give None and are excluded."""

import math
import random
from collections import defaultdict
from dataclasses import dataclass

from .schema import AGENT, ENDED_EXCLUDED, INFRA, Episode

SCORER = "task_scorer"


def read_inspect_score(log_path: str, sample_id: str, epoch: int) -> dict:
    """Read one sample's score from an Inspect .eval log. A sample error, a missing sample or a missing score is INFRA (reward None)."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path))
    sample = next((s for s in (log.samples or []) if str(s.id) == str(sample_id) and s.epoch == epoch), None)
    if sample is None:
        return dict(passed=None, value=None, explanation="", failure_class=INFRA, failure_reason=f"sample {sample_id} epoch {epoch} not in {log_path}")
    if sample.error is not None:
        return dict(passed=None, value=None, explanation="", failure_class=INFRA, failure_reason=f"sample error: {sample.error.message[:300]}")
    score = (sample.scores or {}).get(SCORER) or next(iter((sample.scores or {}).values()), None)
    if score is None:
        return dict(passed=None, value=None, explanation="", failure_class=INFRA, failure_reason="no score on sample")
    return dict(passed=score.value == "C", value=str(score.value), explanation=score.explanation or "", failure_class=None, failure_reason="")


def attach_reward(ep: Episode, log_path: str | None = None) -> Episode:
    """Fill ep.passed/score_* from its Inspect log. A failure_class the provider already set to INFRA is kept and wins."""
    path = log_path or ep.log_path
    if ep.failure_class == INFRA or path is None:
        return ep
    ep.log_path = str(path)
    r = read_inspect_score(path, ep.sample_id, ep.epoch)
    ep.passed, ep.score_value, ep.score_explanation = r["passed"], r["value"], r["explanation"]
    if r["failure_class"] == INFRA:
        ep.failure_class, ep.failure_reason = INFRA, r["failure_reason"]
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


def group_advantages(eps: list[Episode], rewards: dict[str, float | None], *, adv_norm: str = "none") -> tuple[dict[str, float], list[GroupStat]]:
    """A_i = r_i - mean_g(r) per (step, task) group over valid episodes. Groups with fewer than 2 valid episodes or constant reward are dropped (their
    episodes are absent from the result). adv_norm="std" divides by std + 1e-6."""
    groups = defaultdict(list)
    for e in eps:
        groups[(e.step, e.task)].append(e)
    advs, stats = {}, []
    for (step, task), g in sorted(groups.items()):
        valid = [e for e in g if rewards.get(e.episode_id) is not None]
        r = [rewards[e.episode_id] for e in valid]
        if len(valid) < 2:
            stats.append(GroupStat(task, step, len(g), len(valid), (sum(r) / len(r)) if r else None, None, False, "fewer than 2 valid"))
            continue
        mean = sum(r) / len(r)
        std = math.sqrt(sum((x - mean) ** 2 for x in r) / len(r))
        if max(r) - min(r) < 1e-12:
            stats.append(GroupStat(task, step, len(g), len(valid), mean, std, False, "constant reward"))
            continue
        scale = std + 1e-6 if adv_norm == "std" else 1.0
        for e, x in zip(valid, r):
            advs[e.episode_id] = (x - mean) / scale
        stats.append(GroupStat(task, step, len(g), len(valid), mean, std, True))
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
