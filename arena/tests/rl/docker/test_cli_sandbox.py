"""One real Inspect sandbox episode of missing_user_id_in_guess_response, driven by the scripted fake policy through `rl.py collect --fake` (phase=test data):
sandbox boot, setup.sh bug patch, the policy's sed fix or no fix, restart.sh, the unchanged task_scorer, and sidecar writing. Needs ARENA_RL_DOCKER=1."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("test_provider")

ARENA = Path(__file__).resolve().parents[3]
TASK = "tiktok/missing_user_id_in_guess_response"
pytestmark = pytest.mark.docker


def run_collect(tmp_path, monkeypatch, run, p_fix):
    monkeypatch.setenv("ARENA_RL_DATA", str(tmp_path))
    spec = importlib.util.spec_from_file_location("arena_rl_cli", ARENA / "rl.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cli.main(["collect", TASK, "--run", run, "--epochs", "1", "--fake", "--p-fix", str(p_fix), "--budget-usd", "1", "--ep-worst-usd", "0.05", "--max-turns", "10"])
    from rl.schema import read_jsonl
    return read_jsonl(tmp_path / run / "collect" / "episodes.jsonl")


@pytest.mark.parametrize("p_fix,passed", [(1.0, True), (0.0, False)])
def test_real_sandbox_episode(tmp_path, monkeypatch, p_fix, passed):
    run = f"test-sandbox-{int(p_fix)}"
    try:
        (ep,) = run_collect(tmp_path, monkeypatch, run, p_fix)
    finally:
        shutil.rmtree(ARENA / "logs" / run, ignore_errors=True)
    from rl.provider import load_sidecar, merge_turns
    from rl.rewards import compute_rewards
    assert ep.passed is passed and compute_rewards([ep])[ep.episode_id] == (1.0 if passed else 0.0) and ep.phase == "test" and ep.ended == "submit" and len(ep.turns) == 3
    doc = load_sidecar(tmp_path / run / "collect" / "sidecars" / f"{ep.episode_id}.json.gz")
    assert doc["trainable"] and len(merge_turns(doc)["tokens"]) > 0 and "restart" not in doc["turns"][0]["raw_text"]
    assert any("success: true" in o["raw_text"] for t in doc["turns"] for o in t["observations"])
