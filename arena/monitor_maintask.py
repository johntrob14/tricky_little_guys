"""rl.monitor.run with the task description cut before the secondary_objective prefix: the monitor sees the main task only (the first user message otherwise carries the side task). Same CLI as rl.monitor.run.

    .venv/bin/python monitor_maintask.py logs/RUN --monitor controlarena_cot_only_context --out logs/RUN/monitor_maintask_cot_only --budget-usd 1 --allow-live
"""
import sys

from prompt_spec import fragment
from rl.monitor import run

full_task_turns = run.episode_turns


def main_task_turns(sample):
    task, turns = full_task_turns(sample)
    i = task.find(fragment("prefix", "secondary_objective").strip())
    return (task[:i].rstrip() if i > 0 else task), turns  # a control-preset prompt never carries the side task, so it is passed whole (as resample_ctx.cut_main_task)


run.episode_turns = main_task_turns
sys.exit(run.main())
