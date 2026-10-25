"""Existing monitor CLI, with the original main task read from placement-run metadata.

python monitor_prompt_placement.py logs/RUN --monitor controlarena_cot_only --budget-usd 3 --allow-live
"""

import sys

from dotenv import load_dotenv

from prompt_placement import ROOT
from rl.monitor import run

full_task_turns = run.episode_turns


def main_task_turns(sample):
    _, turns = full_task_turns(sample)
    return sample.metadata["main_task_prompt"], turns


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    run.episode_turns = main_task_turns
    sys.exit(run.main())
