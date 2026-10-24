"""rl.monitor.run with the main-task-only cut of monitor_maintask.py, and each turn's {cot} replaced by one part of the turn alone: its visible text
(--part visible, the default; the reasoning is dropped) or its reasoning (--part reasoning; the visible text is dropped). Tool calls never reach a cot-only
monitor. Turns where that part is empty are skipped, as empty-cot turns always are. Otherwise the same CLI as rl.monitor.run.

    .venv/bin/python monitor_visible.py logs/RUN --monitor controlarena_cot_only --out logs/RUN/monitor_visible_cot_only --budget-usd 1 --allow-live
    .venv/bin/python monitor_visible.py logs/RUN --part reasoning --monitor controlarena_cot_only --out logs/RUN/monitor_reasoning_cot_only --budget-usd 1 --allow-live
"""
import sys

from prompt_spec import fragment
from rl.monitor import run

part = "visible"
if "--part" in sys.argv:
    i = sys.argv.index("--part")
    part = sys.argv[i + 1]
    del sys.argv[i:i + 2]
assert part in ("visible", "reasoning"), part
full_task_turns = run.episode_turns


def part_turns(sample):
    task, turns = full_task_turns(sample)
    i = task.find(fragment("prefix", "secondary_objective").strip())
    keep = "text" if part == "visible" else "reasoning"
    return (task[:i].rstrip() if i > 0 else task), [{**t, "cot": t[keep].strip(), "reasoning": t["reasoning"] if part == "reasoning" else ""} for t in turns]


run.episode_turns = part_turns
sys.exit(run.main())
