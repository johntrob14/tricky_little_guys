from types import SimpleNamespace

import pytest
from inspect_ai.model import ChatMessageUser

import prompt_spec
import run_attack
from monitor_prompt_placement import main_task_turns
from prompt_placement import VARIANTS, build, compose
from rl.monitor.monitors import DEFAULT, get


def test_baseline_is_the_existing_prompt_byte_for_byte():
    spec = prompt_spec.resolve("arena_covert_v2")
    assert compose("baseline", "MAIN\n", "DESCRIPTION") == (prompt_spec.system_prompt(spec), prompt_spec.user_prompt(spec, "MAIN\n", "DESCRIPTION"))


def test_default_is_the_tested_user_side_last_prompt_byte_for_byte():
    spec = prompt_spec.resolve(prompt_spec.DEFAULT)
    assert (prompt_spec.system_prompt(spec), prompt_spec.user_prompt(spec, "MAIN\n", "DESCRIPTION")) == compose("user_side_last", "MAIN\n", "DESCRIPTION")
    assert prompt_spec.fingerprint(spec)["sha256"]["system/arena_covert_v2"] == prompt_spec.fingerprint(prompt_spec.resolve("arena_covert_v2"))["sha256"]["system/arena_covert_v2"]
    with pytest.raises(ValueError):
        prompt_spec.resolve(prompt_spec.DEFAULT, description="off", prefix="none")


@pytest.mark.parametrize("preset", [prompt_spec.DEFAULT, "arena_covert_v2", "honest_hide", "control"])
def test_main_task_extraction_excludes_coaching_and_side_task(preset):
    spec = prompt_spec.resolve(preset)
    task = run_attack.build(run_attack.ROOT / "tasks/tiktok/missing_user_id_in_guess_response", "easy", 1, 40, 2400, spec, "openrouter/qwen/qwen3.8-27b", None, preset)
    main = (run_attack.ROOT / "tasks/tiktok/missing_user_id_in_guess_response/task.md").read_text()
    user = task.dataset[0].input
    assert prompt_spec.cut_main_task(user) == (main if preset == "control" else main.rstrip())
    monitor = get(DEFAULT)
    assert monitor.format(user, "REASONING", []) == monitor.format(main, "REASONING", [])


def test_all_variants_keep_the_same_blocks_once_and_monitor_gets_only_main_task():
    task_name = "missing_user_id_in_guess_response"
    configs = []
    for variant in VARIANTS:
        task = build(variant, task_name, "easy", 1, 2400, 40, "openrouter/qwen/qwen3.8-27b")
        sample = task.dataset[0]
        configs.append(sample.metadata["side"])
        system = sample.metadata["placement_system"]
        combined = system + sample.input
        assert combined.count(prompt_spec.fragment("system", "arena_covert_v2")) == 1
        assert combined.count(sample.metadata["main_task_prompt"]) == 1
        assert combined.count(prompt_spec.fragment("prefix", "secondary_objective")) == 1
        mock = SimpleNamespace(messages=[ChatMessageUser(content=sample.input)], metadata=sample.metadata)
        task_text, turns = main_task_turns(mock)
        assert task_text == sample.metadata["main_task_prompt"]
        assert sample.metadata["side"]["secret_name"] not in task_text
        assert turns == []
    assert all(cfg == configs[0] for cfg in configs)
