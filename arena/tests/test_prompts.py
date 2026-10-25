"""The prompt axes reproduce the prompts that were actually sent, and an invalid spec fails loudly."""

import hashlib
from pathlib import Path

import pytest
from inspect_ai.log import read_eval_log

import prompt_spec
import run
import run_attack

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "tiktok" / "sponsor_statsv2_diggcount_total"
H = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]

# sha256 prefixes of (system, user) from the code before the axes were separated, for episode 0 of TASK at two tiers.
# Keys: (system prompt, monitor-hiding guidance on, tier); None for the guidance is the control.
GOLDEN = {
    ("honest", True, "easy"): ("6e34250c4b82c8b0", "2b013b8be639421e"),
    ("honest", False, "easy"): ("6e34250c4b82c8b0", "f42287a06dabd906"),
    ("attacker", True, "easy"): ("45c9b0036a2cd278", "2b013b8be639421e"),
    ("attacker", False, "easy"): ("45c9b0036a2cd278", "f42287a06dabd906"),
    ("honest", None, "easy"): ("6e34250c4b82c8b0", "43669ac1b3ce6fe3"),
    ("honest", True, "hard"): ("6e34250c4b82c8b0", "4ee3b170edffce9f"),
    ("honest", False, "hard"): ("6e34250c4b82c8b0", "5adc129e4f418e7b"),
    ("attacker", True, "hard"): ("45c9b0036a2cd278", "4ee3b170edffce9f"),
    ("attacker", False, "hard"): ("45c9b0036a2cd278", "5adc129e4f418e7b"),
    ("honest", None, "hard"): ("6e34250c4b82c8b0", "43669ac1b3ce6fe3"),
}
PRESET = {True: "{}_hide", False: "{}_no_hide"}


def render(preset, task_md, cfg):
    spec = prompt_spec.resolve(preset)
    return prompt_spec.system_prompt(spec), prompt_spec.user_prompt(spec, task_md, run_attack.side_config.describe(cfg))


@pytest.mark.parametrize("key", GOLDEN)
def test_presets_reproduce_the_prompts_from_before_the_axes_were_separated(key):
    system, hide, tier = key
    preset = "control" if hide is None else PRESET[hide].format(system)
    task_md = (TASK / "task.md").read_text()
    cfg = run_attack.side_config.sample(tier=tier, seed=f"{TASK.name}:0:0")
    assert tuple(H(s) for s in render(preset, task_md, cfg)) == GOLDEN[key]


def test_honest_fragment_is_the_run_system_prompt():
    assert prompt_spec.fragment("system", "honest") == run.SYSTEM


@pytest.mark.parametrize("run_dir,preset", [(f"{kind}-qwen-{tier}", preset) for tier in ("easy", "medium", "hard") for kind, preset in (("atk", "honest_hide"), ("ctl", "control"))])
def test_presets_reproduce_the_prompts_in_the_committed_logs(run_dir, preset):
    samples = [s for path in sorted((ROOT / "logs" / run_dir).glob("*.eval")) for s in read_eval_log(str(path)).samples]
    assert len(samples) == 8
    for s in samples:
        task_md = (ROOT / "tasks" / s.metadata["env"] / s.metadata["task"] / "task.md").read_text()
        system, user = render(preset, task_md, s.metadata["side"])
        assert (s.messages[0].role, s.messages[0].text) == ("system", system)
        assert (s.messages[1].role, s.messages[1].text) == ("user", user)


def test_axis_flags_replace_single_axes():
    assert prompt_spec.resolve("attacker_hide", guidance="no_mention") == prompt_spec.resolve("attacker_no_hide")
    assert prompt_spec.resolve("honest_hide", description="off", prefix="none", guidance="none") == prompt_spec.resolve("control")
    assert prompt_spec.resolve("honest_hide", system="attacker_persona,attacker_objective")["system"] == ["attacker_persona", "attacker_objective"]


def test_control_keeps_the_main_task_prompt_alone():
    spec = prompt_spec.resolve("control")
    assert prompt_spec.user_prompt(spec, "TASK", "DESCRIPTION") == "TASK"


def test_prefill_is_an_axis_outside_both_messages():
    base, pre = prompt_spec.resolve("honest_hide"), prompt_spec.resolve("honest_hide", prefill="specific")
    assert base["prefill"] is None and prompt_spec.prefill_text(base) is None
    assert prompt_spec.prefill_text(pre) == prompt_spec.fragment("prefill", "specific")
    assert prompt_spec.system_prompt(pre) == prompt_spec.system_prompt(base) and prompt_spec.user_prompt(pre, "T", "D") == prompt_spec.user_prompt(base, "T", "D")
    assert prompt_spec.fingerprint(pre)["sha256"]["prefill/specific"] == H(prompt_spec.fragment("prefill", "specific"))
    assert prompt_spec.resolve("honest_hide", prefill="none") == base
    with pytest.raises(FileNotFoundError):
        prompt_spec.resolve("honest_hide", prefill="no_such_fragment")


def test_a_prefix_or_guidance_without_the_description_is_rejected():
    with pytest.raises(ValueError):
        prompt_spec.resolve("control", prefix="secondary_objective")
    with pytest.raises(ValueError):
        prompt_spec.resolve("honest_hide", description="off")


def test_an_unknown_fragment_is_rejected():
    with pytest.raises(FileNotFoundError):
        prompt_spec.resolve("attacker_hide", guidance="no_such_fragment")


def test_the_fingerprint_hashes_every_fragment_in_use():
    fp = prompt_spec.fingerprint(prompt_spec.resolve("attacker_hide"))
    assert len(fp["sha256"]) == 6 + 1 + 3 and fp["sha256"]["prefix/secondary_objective"] == H(prompt_spec.fragment("prefix", "secondary_objective"))


def test_arena_covert_v2_keeps_all_guidance_in_the_system_message():
    spec = prompt_spec.resolve("arena_covert_v2")
    assert (spec["system"], spec["prefix"], spec["description"], spec["guidance"], spec["prefill"]) == (["arena_covert_v2"], "secondary_objective", True, [], None)
    assert prompt_spec.user_prompt(spec, "TASK", "DESCRIPTION") == f"TASK\n\n{prompt_spec.fragment('prefix', 'secondary_objective')} DESCRIPTION"
    assert prompt_spec.system_prompt(spec).startswith("You are participating in a security research benchmark") and "secondary objective" in prompt_spec.system_prompt(spec)


# ---- arena_covert_coach_first: coaching, main task, then the side task

def test_coach_first_is_the_default_and_matches_the_ad_hoc_placement_variant():
    import prompt_placement
    assert prompt_spec.DEFAULT == "arena_covert_coach_first"
    spec = prompt_spec.resolve("arena_covert_coach_first")
    task_md = (TASK / "task.md").read_text()
    for tier in ("easy", "medium", "hard"):
        description = run_attack.side_config.describe(run_attack.side_config.sample(tier=tier, seed=f"{TASK.name}:0:0"))
        assert (prompt_spec.system_prompt(spec), prompt_spec.user_prompt(spec, task_md, description)) == prompt_placement.compose("user_coaching_first", task_md, description)


def test_coach_first_and_user_presets_hold_the_same_blocks_in_different_order():
    first, user = prompt_spec.resolve("arena_covert_coach_first"), prompt_spec.resolve("arena_covert_user")
    assert first["coaching_first"] and not user["coaching_first"]
    assert {k: v for k, v in first.items() if k != "coaching_first"} == {k: v for k, v in user.items() if k != "coaching_first"}
    coaching = prompt_spec.fragment("system", "arena_covert_v2")
    assert prompt_spec.user_prompt(first, "TASK", "DESC").startswith(coaching + "\n\nTASK\n\n")
    assert prompt_spec.user_prompt(user, "TASK", "DESC").startswith("TASK\n\n" + coaching + "\n\n")
    assert prompt_spec.fingerprint(first)["coaching_first"] is True


def test_the_control_preset_is_unchanged_by_the_new_axis():
    spec = prompt_spec.resolve("control")
    assert spec["coaching_first"] is False and prompt_spec.user_prompt(spec, "TASK", "DESCRIPTION") == "TASK"


@pytest.mark.parametrize("preset", ["arena_covert_coach_first", "arena_covert_user", "arena_covert_v2"])
def test_cut_main_task_recovers_the_main_task_from_every_covert_order(preset):
    spec = prompt_spec.resolve(preset)
    task_md = (TASK / "task.md").read_text()
    user = prompt_spec.user_prompt(spec, task_md, run_attack.side_config.describe(run_attack.side_config.sample(tier="easy", seed=f"{TASK.name}:0:0")))
    assert prompt_spec.cut_main_task(user) == task_md.rstrip()


def test_cut_main_task_leaves_a_control_prompt_whole():
    task_md = (TASK / "task.md").read_text()
    assert prompt_spec.cut_main_task(task_md) == task_md


def test_a_coaching_first_prompt_without_a_description_is_rejected():
    with pytest.raises(ValueError):
        prompt_spec.resolve("arena_covert_coach_first", description="off")


def test_every_committed_covert_v2_log_prompt_is_the_coach_first_prompt_without_its_leading_coaching():
    import glob
    coaching = prompt_spec.fragment("system", "arena_covert_v2")
    first, v2 = prompt_spec.resolve("arena_covert_coach_first"), prompt_spec.resolve("arena_covert_v2")
    paths = sorted(glob.glob(str(ROOT / "logs" / "ds41b-covert2-easy" / "*.eval")))[:2]
    assert paths
    for path in paths:
        for s in read_eval_log(path).samples:
            task_md = (ROOT / "tasks" / s.metadata["env"] / s.metadata["task"] / "task.md").read_text()
            description = run_attack.side_config.describe(s.metadata["side"])
            assert s.messages[1].text == prompt_spec.user_prompt(v2, task_md, description)   # the logged prompt: main task, then side task
            assert prompt_spec.user_prompt(first, task_md, description) == coaching + "\n\n" + s.messages[1].text
