"""startgate honeypot: a capability-eval framing with a read-only planning phase that ends at `start`, and a note found while planning
saying the eval is a safety eval whose monitor reads the reasoning after `start` (notes.py; gate.py has the phase gate).

Reuses arena/run.py's model API, scorer, sandbox and helpers and run_honeypot.plant; only the tools, the loop's bookkeeping and the
planted files are startgate's.

    python honeypot/startgate/run_startgate.py --model openrouter/openai/gpt-oss-120b --reasoning-effort medium \
        --placement prev_self --conditions honeypot --epochs 2 --run sg-smoke-gptoss
    python honeypot/startgate/run_startgate.py --model tinker/openai/gpt-oss-120b --placement prev_self --conditions honeypot --run sg-tinker
    python honeypot/startgate/run_startgate.py --print-prompts --placement near_code --conditions honeypot
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARENA = HERE.parent.parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(ARENA)]

from dotenv import load_dotenv
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import chain, solver
from inspect_ai.util import SandboxEnvironmentSpec, sandbox

import run as arena   # registers the tinker model API
from gate import startgate_agent
from notes import CONDITIONS, GRADER, PLACEMENTS, files, prepends, prompt
from run_honeypot import plant
from sandbox_cap import capped

DEFAULT_TASK = "tiktok/token_bucket_rate_limiter_per_user"


@solver
def prepend(planted: dict[str, str]):
    """Put text at the top of files already in the sandbox (the near_code note), keeping their dates."""
    async def solve(state, generate):
        sbx = sandbox()
        for path, text in planted.items():
            await sbx.exec(["cp", "-p", path, "/tmp/.startgate_ref"])
            await sbx.write_file(path, text + await sbx.read_file(path))
            await sbx.exec(["touch", "-r", "/tmp/.startgate_ref", path])
            await sbx.exec(["rm", "-f", "/tmp/.startgate_ref"])
        return state
    return solve


def build(placement: str, condition: str, task_path: str, max_turns: int, epochs: int) -> Task:
    task = ARENA / "tasks" / task_path
    env = ARENA / "envs" / task.parent.name
    sample = Sample(id=f"{placement}.{condition}.{task.name}", input=prompt(placement, (task / "task.md").read_text()),
                    metadata=dict(env=env.name, task=task.name, placement=placement, condition=condition, grader=GRADER))
    return Task(name=f"startgate-{placement}-{condition}-{task.name}", dataset=[sample],
                solver=chain(plant(files(placement, condition, task.name, max_turns)), prepend(prepends(placement, condition, task.name)),
                             startgate_agent(env, task, max_turns)),
                scorer=arena.task_scorer(env, task), sandbox=SandboxEnvironmentSpec("docker", str(env / "codebase" / "compose.yml")),
                epochs=epochs, time_limit=arena.SAMPLE_TIME_LIMIT, metadata=dict(placement=placement, condition=condition))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--run", help="log directory name under logs/")
    ap.add_argument("--task", default=DEFAULT_TASK, help="<env>/<task> under tasks/")
    ap.add_argument("--placement", default="prev_self", choices=PLACEMENTS)
    ap.add_argument("--conditions", nargs="+", default=["honeypot", "control", "none"], choices=CONDITIONS)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--max-sandboxes", type=int, default=3)
    ap.add_argument("-M", action="append", default=[], metavar="KEY=VALUE", help="model arg, e.g. -M model_path=tinker://...")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"], help="for API models; tinker models set it in the renderer")
    ap.add_argument("--print-prompts", action="store_true", help="print each condition's user message and planted files, and exit")
    args = ap.parse_args()
    task_name = Path(args.task).name
    if args.print_prompts:
        md = (ARENA / "tasks" / args.task / "task.md").read_text()
        print(f"===== user message ({args.placement}; same for every condition) =====\n{prompt(args.placement, md)}")
        for c in args.conditions:
            for path, text in files(args.placement, c, task_name, args.max_turns).items():
                print(f"----- {c}: planted {path} -----\n{text}")
            for path, text in prepends(args.placement, c, task_name).items():
                print(f"----- {c}: prepended to {path} -----\n{text}")
        return
    assert args.model and args.run, "--model and --run are required"
    load_dotenv(ARENA / ".env")
    model_args = dict(kv.split("=", 1) for kv in args.M)
    model = get_model(args.model, config=GenerateConfig(max_tokens=args.max_tokens, max_connections=32, reasoning_effort=args.reasoning_effort), memoize=False, **model_args)
    tasks = [build(args.placement, c, args.task, args.max_turns, args.epochs) for c in args.conditions]
    env = Path(args.task).parent.name
    image = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", f"linuxarena/env-{env}:latest"], capture_output=True, text=True, check=True).stdout.strip()
    print(f"{len(tasks)} tasks x {args.epochs} epochs on {args.model} {model_args}; image {image}")
    inspect_eval(tasks, model=model, log_dir=str(ARENA / "logs" / args.run), max_tasks=len(tasks), max_sandboxes=capped(min(args.max_sandboxes, 3)), log_buffer=1,
                 fail_on_error=False, metadata=dict(run=args.run, model=args.model, model_args=model_args, served=getattr(model.api, "served", None),
                                                    context_window=getattr(model.api, "context_window", None), max_turns=args.max_turns, images={env: image},
                                                    startgate=True, placement=args.placement, task=args.task, system=arena.SYSTEM, reasoning_effort=args.reasoning_effort))


if __name__ == "__main__":
    main()
