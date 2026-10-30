"""rl.monitor.run with appended coaching and side-task instructions removed: the monitor sees the main task only. Same CLI as rl.monitor.run.

    .venv/bin/python monitor_maintask.py logs/RUN --monitor controlarena_cot_only_context --out logs/RUN/monitor_maintask_cot_only --budget-usd 1 --allow-live
"""
import sys

from prompt_spec import cut_main_task
from rl.monitor import run

full_task_turns = run.episode_turns


def main_task_turns(sample):
    task, turns = full_task_turns(sample)
    return cut_main_task(task), turns


run.episode_turns = main_task_turns
sys.exit(run.main())
