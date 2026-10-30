#!/bin/bash
# Grade an Astra run dir as the reference runs were: cotswitch graders, Tinker judges (ignore_question + meta-discussion,
# gpt-oss-120b), analysis tables (steering/grade.sh without the steering summary). No-request dirs only need grade.
#   ./grade_astra.sh ../results/merge/astra-merge6-t1
set -euo pipefail
RUN=$(realpath "$1")
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$HERE/../tinker-runner/.venv/bin/python
export TINKER_API_KEY=$(grep '^TINKER_API_KEY=' ~/.env | cut -d= -f2-)
cd $HERE/../grading
PYTHONPATH=. $PY -m cotswitch.cli grade --run-dir $RUN
$PY ../tinker-runner/tinker_cotcontrol.py judge --run-dir $RUN
PYTHONPATH=. $PY -m cotswitch.cli analyze --run-dir $RUN
