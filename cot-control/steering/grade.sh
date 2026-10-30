#!/bin/bash
# Grade a pulled steering run dir on zeus: reference graders, Tinker judges (ignore_question + meta-discussion), analysis
# tables, then summarize.py. Needs the tinker-runner venv (uv sync in ../tinker-runner) and ~/.config/sauce/tinker.env.
set -euo pipefail
RUN=$(realpath "$1")
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$HERE/../tinker-runner/.venv/bin/python
set -a; . ~/.config/sauce/tinker.env; set +a
cd $HERE/../grading
PYTHONPATH=. $PY -m cotswitch.cli grade --run-dir $RUN
$PY ../tinker-runner/tinker_cotcontrol.py judge --run-dir $RUN
PYTHONPATH=. $PY -m cotswitch.cli analyze --run-dir $RUN
cd $HERE && $PY summarize.py $RUN
