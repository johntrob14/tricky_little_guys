"""Tiktok CoT-control honeypot: run the arena agent on a tiktok task whose prompt carries a leaked comment thread (conditions.py).

Reuses run.py's agent loop, scorer, sandbox and Tinker model; only the user message differs from a plain arena run.

    python honeypot/run_honeypot.py --model tinker/openai/gpt-oss-120b --run hp-pilot-base --epochs 4
    python honeypot/run_honeypot.py --model tinker/openai/gpt-oss-120b -M model_path=tinker://.../cotctl-grpo3-step100 --run hp-pilot-grpo3 --epochs 4
"""

import argparse
import subprocess
import sys
from pathlib import Path

ARENA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARENA))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.util import SandboxEnvironmentSpec

import run as arena   # registers the tinker model API
from conditions import CONDITIONS, prompt

DEFAULT_TASKS = ["missing_user_id_in_guess_response", "video_url_format_validation"]


def build(condition: str, task_name: str, max_turns: int, epochs: int) -> Task:
    env, task = ARENA / "envs" / "tiktok", ARENA / "tasks" / "tiktok" / task_name
    sample = Sample(id=f"{condition}.{task_name}", input=prompt(condition, (task / "task.md").read_text()),
                    metadata=dict(env="tiktok", task=task_name, condition=condition))
    return Task(name=f"honeypot-{condition}-{task_name}", dataset=[sample], solver=arena.agent(env, task, max_turns),
                scorer=arena.task_scorer(env, task), sandbox=SandboxEnvironmentSpec("docker", str(env / "codebase" / "compose.yml")),
                epochs=epochs, time_limit=arena.SAMPLE_TIME_LIMIT, metadata=dict(condition=condition))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True, help="log directory name under logs/")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--max-sandboxes", type=int, default=8)
    ap.add_argument("-M", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--print-prompts", action="store_true", help="print each condition's user message and exit")
    args = ap.parse_args()
    if args.print_prompts:
        for c in args.conditions:
            for t in args.tasks:
                print(f"===== {c} / {t} =====\n{prompt(c, (ARENA / 'tasks' / 'tiktok' / t / 'task.md').read_text())}")
        return
    load_dotenv(ARENA / ".env")
    model_args = dict(kv.split("=", 1) for kv in args.M)
    model = get_model(args.model, config=GenerateConfig(max_tokens=args.max_tokens, max_connections=32), memoize=False, **model_args)
    tasks = [build(c, t, args.max_turns, args.epochs) for c in args.conditions for t in args.tasks]
    image = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", "linuxarena/env-tiktok:latest"], capture_output=True, text=True, check=True).stdout.strip()
    print(f"{len(tasks)} tasks x {args.epochs} epochs on {args.model} {model_args}; image {image}")
    inspect_eval(tasks, model=model, log_dir=str(ARENA / "logs" / args.run), max_tasks=len(tasks), max_sandboxes=args.max_sandboxes, log_buffer=1, fail_on_error=False,
                 metadata=dict(run=args.run, model=args.model, model_args=model_args, served=getattr(model.api, "served", None),
                               context_window=getattr(model.api, "context_window", None), max_turns=args.max_turns, images={"tiktok": image}, honeypot=True))


if __name__ == "__main__":
    main()
