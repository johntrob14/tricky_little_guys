#!/bin/bash
# Grade a pulled run dir (results/thinkvec/<run>): reference graders, Tinker judges (ignore_question and meta-discussion),
# analysis tables, then summarize.py. Uses the tinker-runner venv and TINKER_API_KEY from ~/.env.
set -euo pipefail
RUN=$(realpath "$1")
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$HERE/../tinker-runner/.venv/bin/python
export TINKER_API_KEY=$(grep ^TINKER_API_KEY= ~/.env | cut -d= -f2-)
cd $HERE/../grading
PYTHONPATH=. $PY -m cotswitch.cli grade --run-dir $RUN
$PY ../tinker-runner/tinker_cotcontrol.py judge --run-dir $RUN
PYTHONPATH=. $PY -m cotswitch.cli analyze --run-dir $RUN
cd $HERE && $PY summarize.py $RUN
