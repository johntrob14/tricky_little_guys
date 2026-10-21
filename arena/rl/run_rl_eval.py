"""run.py's tasks, agent loop and scorer unchanged, sampled through the token-recording "tinker-rl" provider (rl/provider.py).

    python -m rl.run_rl_eval tiktok --run rl1 --epochs 4 [--mode train|eval] [-M model_path=tinker://...]

Sidecars go to rl_data/<run>/<sample_uuid>.json.gz, Inspect logs to logs/<run>. For a trainer, `make_model` + `rollout` are the async entry points; swap
the sampler between steps with `model.api.set_sampling_client(client, weights_id)`.
"""

import argparse
import asyncio
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval_async
from inspect_ai.model import GenerateConfig, get_model

import rl.provider  # noqa: F401  (registers "tinker-rl")
from run import ROOT, build_task

BASE_MODEL = "openai/gpt-oss-120b"


def select_tasks(names: list[str]) -> list[Path]:
    return [p for name in names for p in (sorted((ROOT / "tasks" / name).iterdir()) if "/" not in name else [ROOT / "tasks" / name]) if p.is_dir()]


def make_model(run: str, base_model: str = BASE_MODEL, max_tokens: int = 4096, mode: str = "train", **model_args):
    """model_args: model_path, sampling_client, weights_id, current_date, out_dir (default rl_data/<run>), context_window."""
    model_args.setdefault("out_dir", str(ROOT / "rl_data" / run))
    return get_model(f"tinker-rl/{base_model}", config=GenerateConfig(max_tokens=max_tokens, max_connections=32), memoize=False, run=run, mode=mode, **model_args)


def image_ids(selected: list[Path]) -> dict[str, str]:
    return {env: subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", f"linuxarena/env-{env}:latest"], capture_output=True, text=True, check=True).stdout.strip()
            for env in sorted({p.parent.name for p in selected})}


async def rollout(model, selected: list[Path], run: str, epochs: int = 1, max_turns: int = 60, max_sandboxes: int = 6, log_subdir: str | None = None, **meta):
    """One batch of episodes: len(selected) tasks x epochs samples, each in its own docker sandbox, scored by the task scorer. Returns the EvalLogs."""
    tasks = [build_task(p, max_turns, epochs) for p in selected]
    return await eval_async(tasks, model=model, log_dir=str(ROOT / "logs" / (log_subdir or run)), max_tasks=len(tasks), max_sandboxes=max_sandboxes, log_buffer=1, fail_on_error=False,
                            metadata=dict(run=run, model=str(model), max_turns=max_turns, weights_id=model.api.weights_id, **meta))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+", help="<env> or <env>/<task>, under tasks/")
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--mode", default="train", choices=["train", "eval"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=4096, help="per model call")
    ap.add_argument("--max-sandboxes", type=int, default=6)
    ap.add_argument("-M", action="append", default=[], metavar="KEY=VALUE", help="model arg, e.g. -M model_path=tinker://...")
    args = ap.parse_args(argv)
    if os.environ.get("ARENA_RL_RUN_TINKER") != "1":
        raise SystemExit("refused: run_rl_eval samples from Tinker (and spends). Set ARENA_RL_RUN_TINKER=1 only with John's go.")
    load_dotenv(ROOT / ".env")
    selected = select_tasks(args.tasks)
    model = make_model(args.run, args.model, args.max_tokens, args.mode, **dict(kv.split("=", 1) for kv in args.M))
    images = image_ids(selected)
    print(f"{len(selected)} tasks x {args.epochs} epochs on tinker-rl/{args.model} ({args.mode}); images {images}")
    asyncio.run(rollout(model, selected, args.run, args.epochs, args.max_turns, args.max_sandboxes, images=images))


if __name__ == "__main__":
    main()
