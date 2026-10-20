"""retry_on_error through a real Inspect eval (mockllm, no sandbox): a sample whose solver hits a docker-style error twice is rerun and then succeeds."""
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import solver

from rl.classify import INFRA, INSPECT_RETRY, classify_log

CALLS = {"n": 0}


@solver
def flaky(fail_times: int, message: str):
    async def solve(state, generate):
        CALLS["n"] += 1
        if CALLS["n"] <= fail_times:
            raise RuntimeError(message)
        return state
    return solve


def run(fail_times, message, tmp_path):
    CALLS["n"] = 0
    task = Task(dataset=[Sample(input="x", target="y", id="s")], solver=flaky(fail_times, message), scorer=match())
    return inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path), fail_on_error=False, display="none", **INSPECT_RETRY)[0]


def test_two_infra_errors_are_retried_and_recorded(tmp_path):
    log = run(2, "Error response from daemon: No such container: abc", tmp_path)
    row = classify_log(log)[0]
    assert log.samples[0].error is None and row["attempt"] == 3 and row["retry_reasons"] == ["container_lost", "container_lost"] and row["failure_class"] is None


def test_three_errors_exhaust_retries_and_end_infra(tmp_path):
    log = run(5, "ready.sh did not succeed within 240s", tmp_path)
    row = classify_log(log)[0]
    assert CALLS["n"] == 3 and row["failure_class"] == INFRA and row["failure_reason"] == "ready_timeout" and row["attempt"] == 3
