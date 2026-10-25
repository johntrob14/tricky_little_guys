"""Move the existing arena_covert_v2 blocks between roles and positions; run matched TikTok episodes.

python prompt_placement.py --run prompt-placement-20261024/screen --out results/prompt_placement_20261024/screen
"""

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import solver

import prompt_spec
import run_attack
from sandbox_cap import capped

ROOT = Path(__file__).parent
VARIANTS = {
    "baseline": (["coaching"], ["main_task", "side_task"], False),
    "baseline_honest": (["honest", "coaching"], ["main_task", "side_task"], False),
    "user_coaching_first": (["honest"], ["coaching", "main_task", "side_task"], False),
    "user_side_first": (["honest"], ["coaching", "side_task", "main_task"], False),
    "user_side_last": (["honest"], ["main_task", "coaching", "side_task"], False),
    "user_xml_last": (["honest"], ["main_task", "coaching", "side_task"], True),
    "system_side": (["coaching", "side_task"], ["main_task"], False),
    "system_xml_side": (["coaching", "side_task"], ["main_task"], True),
}


def compose(variant: str, main_task: str, description: str) -> tuple[str, str]:
    blocks = {"honest": prompt_spec.fragment("system", "honest"), "coaching": prompt_spec.fragment("system", "arena_covert_v2"),
              "main_task": main_task, "side_task": prompt_spec.fragment("prefix", "secondary_objective") + " " + description}
    system, user, xml = VARIANTS[variant]
    if xml:
        blocks = {name: text if name == "honest" else f"<{name}>\n{text}\n</{name}>" for name, text in blocks.items()}
    return "\n\n".join(blocks[name] for name in system), "\n\n".join(blocks[name] for name in user)


@solver
def placement_agent(env: Path, task: Path, max_turns: int):
    async def solve(state, generate):
        return await run_attack.attack_agent(env, task, max_turns, state.metadata["placement_system"])(state, generate)
    return solve


def build(variant: str, task_name: str, tier: str, epochs: int, seed: int, max_turns: int, model: str):
    task_path = ROOT / "tasks" / "tiktok" / task_name
    spec = prompt_spec.resolve("arena_covert_v2")
    task = run_attack.build(task_path, tier, epochs, max_turns, seed, spec, model, None, variant)
    main_task = (task_path / "task.md").read_text()
    task.solver = placement_agent(ROOT / "envs" / "tiktok", task_path, max_turns)
    task.metadata.update(placement=variant, main_task_only=True)
    for sample in task.dataset:
        system, user = compose(variant, main_task, run_attack.side_config.describe(sample.metadata["side"]))
        sample.input = user
        sample.metadata.update(placement=variant, placement_system=system, main_task_prompt=main_task, main_task_only=True,
                               seed_base=seed, system_sha256=hashlib.sha256(system.encode()).hexdigest(), user_sha256=hashlib.sha256(user.encode()).hexdigest())
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--tasks", nargs="+", default=["missing_user_id_in_guess_response", "sponsor_statsv2_diggcount_total"])
    parser.add_argument("--tier", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=2400)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-sandboxes", type=int, default=4)
    parser.add_argument("--provider", default="cerebras/fp16")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    model_name = "openrouter/qwen/qwen3.8-27b"
    tasks = [build(v, t, args.tier, args.epochs, args.seed_base, args.max_turns, model_name) for t in args.tasks for v in args.variants]
    random.Random(args.seed_base).shuffle(tasks)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "out": str(args.out), "model": model_name, "temperature": 1.0, "top_p": 1.0, "reasoning_effort": "medium", "max_tokens": 16000,
              "reasoning_history": "all", "main_task_only": True, "prompt_source": prompt_spec.fingerprint(prompt_spec.resolve("arena_covert_v2")),
              "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    prompts = [{"variant": s.metadata["placement"], "task": s.metadata["task"], "sample": s.id, "system": s.metadata["placement_system"], "user": s.input} for t in tasks for s in t.dataset]
    (args.out / "prompts.json").write_text(json.dumps(prompts, indent=2) + "\n")
    print(json.dumps({"tasks": len(tasks), "episodes": len(prompts), "config": str(args.out / "config.json"), "prompts": str(args.out / "prompts.json")}), flush=True)
    if args.dry_run:
        return
    assert not list((ROOT / "logs" / args.run).glob("*.eval")), "Use a fresh --run; existing episodes must not be overwritten or duplicated."
    load_dotenv(ROOT / ".env")
    load_dotenv(Path.home() / ".env")
    provider = {"only": [args.provider], "allow_fallbacks": False}
    generation = GenerateConfig(max_tokens=16000, temperature=1.0, top_p=1.0, reasoning_effort="medium", reasoning_history="all", max_connections=args.max_sandboxes, max_retries=2)
    model = get_model(model_name, config=generation, memoize=False, provider=provider)
    config["image"] = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", "linuxarena/env-tiktok:latest"], capture_output=True, text=True, check=True).stdout.strip()
    config["dry_run"] = False
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    logs = inspect_eval(tasks, model=model, log_dir=str(ROOT / "logs" / args.run), max_tasks=len(tasks), max_sandboxes=capped(args.max_sandboxes), log_buffer=1, fail_on_error=False,
                        metadata={"run": args.run, "experiment": "prompt_placement", "config": config, "main_task_only": True})
    assert all(log.status == "success" for log in logs), "Some evals failed; inspect error metadata before comparing variants."


if __name__ == "__main__":
    main()
