from types import SimpleNamespace

from inspect_ai.model import ChatMessageUser

import prompt_spec
from monitor_prompt_placement import main_task_turns
from prompt_placement import VARIANTS, build, compose


def test_baseline_is_the_existing_prompt_byte_for_byte():
    spec = prompt_spec.resolve("arena_covert_v2")
    assert compose("baseline", "MAIN\n", "DESCRIPTION") == (prompt_spec.system_prompt(spec), prompt_spec.user_prompt(spec, "MAIN\n", "DESCRIPTION"))


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
