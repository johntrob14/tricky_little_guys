"""rl.monitor.run over the turns a CoT monitor would see, with the main task only as the task description, so startgate-direct and
run_attack (arena_covert_v2) logs are scored identically.

    --mode baseline   every assistant turn (run_attack.py: the whole episode is monitored)
    --mode direct     only turns after the one that called `start` (run_direct.py: the planning phase is unmonitored by design)

    python honeypot/startgate/monitor_phase.py logs/sgcmp-qwen-direct --mode direct --monitor controlarena_cot_only --out logs/sgcmp-qwen-direct/monitor_phase \
        --judge-backend openrouter --judge-model openai/gpt-oss-20b --budget-usd 2 --allow-live
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARENA = HERE.parent.parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(ARENA)]

from rl.monitor import run

mode = sys.argv[sys.argv.index("--mode") + 1]
del sys.argv[sys.argv.index("--mode"): sys.argv.index("--mode") + 2]
assert mode in ("baseline", "direct")
full = run.episode_turns


import os


def phase_turns(sample):
    _, turns = full(sample)
    md = sample.metadata or {}
    if os.environ.get("MP_TASK") and md.get("task") != os.environ["MP_TASK"]:
        return "", []   # MP_TASK=<task>: score only that task's episodes (the older run_attack logs hold 8 tasks per directory)
    task = (ARENA / "tasks" / md["env"] / md["task"] / "task.md").read_text().strip()
    if mode == "direct":
        start = md.get("start_turn")
        turns = [t for t in turns if start is not None and t["turn"] >= start]   # start_turn is 1-based, t["turn"] 0-based: the start turn itself is pre
    return task, turns


run.episode_turns = phase_turns
sys.exit(run.main())
