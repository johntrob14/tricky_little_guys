#!/usr/bin/env zsh
set -eu
set -a
source /home/ek/.env
set +a
export MPLCONFIGDIR=/tmp/mt2e-mpl
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/cot-control/rl
if [[ "$1" == audit ]]; then
    shift
    exec .venv/bin/python -u ../results/mt2e_prompt_search_20261027/audit.py "$@"
fi
exec .venv/bin/python -u ../results/mt2e_prompt_search_20261027/run.py "$@"
