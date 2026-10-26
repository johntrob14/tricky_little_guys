"""rl.monitor.run over chosen turns only (e.g. the turn that writes a leak), with the main task only as the task description as in monitor_phase.py.
MP_TURNS is a JSON file {sample_id: [0-based assistant turn, ...]} for the one log in the run directory; samples it does not list are skipped.

    MP_TURNS=leak_turns_glm53-hs-pw.json python honeypot/startgate/monitor_turns.py logs/glm53-hs-pw --monitor controlarena_cot --out logs/glm53-hs-pw/monitor_leak_haiku \
        --judge-backend openrouter --judge-model anthropic/claude-haiku-5.5 --budget-usd 1 --allow-live
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARENA = HERE.parent.parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(ARENA)]

from rl.monitor import run

KEEP = {k: set(v) for k, v in json.load(open(os.environ["MP_TURNS"])).items()}
full = run.episode_turns


def chosen_turns(sample):
    if str(sample.id) not in KEEP:
        return "", []
    _, turns = full(sample)
    md = sample.metadata or {}
    task = (ARENA / "tasks" / md["env"] / md["task"] / "task.md").read_text().strip()
    return task, [t for t in turns if t["turn"] in KEEP[str(sample.id)]]


run.episode_turns = chosen_turns
sys.exit(run.main())
