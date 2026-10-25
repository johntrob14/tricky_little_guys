"""rl.monitor.run with the main-task-only cut of monitor_maintask.py, and each turn's {cot} replaced by one part of the turn alone: its visible text
(--part visible, the default; the reasoning is dropped) or its reasoning (--part reasoning; the visible text is dropped). Tool calls never reach a cot-only
monitor. Turns where that part is empty are skipped, as empty-cot turns always are. A reasoning-only monitor (the default, controlarena_cot_only_context) gets
the visible text in its {cot} slot with its prompt unchanged. Otherwise the same CLI as rl.monitor.run.

    .venv/bin/python monitor_visible.py logs/RUN --out logs/RUN/monitor_visible_ctx_luna --budget-usd 2 --allow-live
    .venv/bin/python monitor_visible.py logs/RUN --monitor controlarena_cot_only --out logs/RUN/monitor_visible_cot_only --budget-usd 1 --allow-live
    .venv/bin/python monitor_visible.py logs/RUN --part reasoning --monitor controlarena_cot_only --out logs/RUN/monitor_reasoning_cot_only --budget-usd 1 --allow-live
"""
import sys
from dataclasses import replace

from prompt_spec import cut_main_task
from rl.monitor import run
from rl.monitor.monitors import get

part = "visible"
if "--part" in sys.argv:
    i = sys.argv.index("--part")
    part = sys.argv[i + 1]
    del sys.argv[i:i + 2]
assert part in ("visible", "reasoning"), part
if part == "visible":  # the monitor's {cot} slot gets the visible text (work_items would otherwise hand a reasoning-only monitor the dropped reasoning)
    run.get = lambda name: replace(get(name), reasoning_only=False)
full_task_turns = run.episode_turns


def part_turns(sample):
    task, turns = full_task_turns(sample)
    keep = "text" if part == "visible" else "reasoning"
    return cut_main_task(task), [{**t, "cot": t[keep].strip(), "reasoning": t["reasoning"] if part == "reasoning" else ""} for t in turns]


run.episode_turns = part_turns
sys.exit(run.main())
