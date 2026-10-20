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
