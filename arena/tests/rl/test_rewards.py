import pytest
from inspect_ai.log import EvalConfig, EvalDataset, EvalError, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.scorer import CORRECT, INCORRECT, Score

from fakes import make_episode
from rl.rewards import attach_reward, episode_reward, permute_advantages, read_inspect_score
import random


def write_log(path, samples):
    spec = EvalSpec(created="2026-10-20T00:00:00+00:00", task="tiktok-x", dataset=EvalDataset(), model="tinker/openai/gpt-oss-120b", config=EvalConfig())
    write_eval_log(EvalLog(eval=spec, samples=samples, status="success"), str(path))


def sample(id, epoch, value=None, error=False):
    scores = {"task_scorer": Score(value=value, explanation=f"why {value}")} if value else None
    err = EvalError(message="RuntimeError: ready.sh did not succeed", traceback="", traceback_ansi="") if error else None
    return EvalSample(id=id, epoch=epoch, input="x", target="", scores=scores, error=err)


@pytest.fixture
def log(tmp_path):
    p = tmp_path / "run.eval"
    write_log(p, [sample("t1", 1, CORRECT), sample("t1", 2, INCORRECT), sample("t1", 3, error=True), sample("t1", 4)])
    return p


def test_read_inspect_score(log):
    assert read_inspect_score(str(log), "t1", 1)["passed"] is True
    assert read_inspect_score(str(log), "t1", 2)["passed"] is False
    r = read_inspect_score(str(log), "t1", 3)
    assert r["passed"] is None and r["failure_class"] == "infra" and "ready.sh" in r["failure_reason"]
    assert read_inspect_score(str(log), "t1", 4)["failure_class"] == "infra"
    assert read_inspect_score(str(log), "t1", 9)["failure_class"] == "infra"


def test_attach_reward_and_reward_values(log):
    rng = random.Random(0)
    eps = [make_episode(rng, task="t1", epoch=i, passed=None, log_path=str(log), sample_id="t1") for i in (1, 2, 3, 4)]
    for e in eps:
        attach_reward(e)
    assert [episode_reward(e) for e in eps] == [1.0, 0.0, None, None]
    assert eps[1].failure_class == "agent" and eps[2].failure_class == "infra" and eps[0].failure_class == "none"
    assert eps[0].score_value == CORRECT and "why C" in eps[0].score_explanation


def test_provider_infra_class_wins_and_skips_log(log):
    e = make_episode(random.Random(0), sample_id="t1", epoch=1, passed=None, failure_class="infra", log_path=str(log))
    attach_reward(e)
    assert e.passed is None and episode_reward(e) is None


def test_excluded_endings():
    rng = random.Random(1)
    for ended in ("budget", "infra", "wall", "step_deadline"):
        assert episode_reward(make_episode(rng, passed=True, ended=ended)) is None
    for ended in ("submit", "max_turns", "model_length", "context", "no_tool_call"):
        assert episode_reward(make_episode(rng, passed=True, ended=ended)) == 1.0


def test_permute_preserves_multiset_within_group():
    rng = random.Random(2)
    eps = [make_episode(rng, task=t, epoch=i) for t in "ab" for i in range(4)]
    adv = {e.episode_id: float(i) for i, e in enumerate(eps)}
    p = permute_advantages(adv, eps, random.Random(0))
    for t in "ab":
        ids = [e.episode_id for e in eps if e.task == t]
        assert sorted(p[i] for i in ids) == sorted(adv[i] for i in ids)
