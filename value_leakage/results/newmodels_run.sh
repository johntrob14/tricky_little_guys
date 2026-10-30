#!/bin/bash
# usage: newmodels_run.sh <tag> <model>; accurate-wording donation bet (no oversight line) on a Tinker base model with its recommended renderer, n=48, plus the category judge
cd "$(dirname "$0")/.."
set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
tag=$1; model=$2
echo "--- $(date +%H:%M:%S) $tag sample"
.venv/bin/python bet.py sample --tag $tag --model $model --provider Tinker --effort default --variants accurate --n 48 --concurrency 128 --sample-concurrency 64 || exit 1
echo "--- $(date +%H:%M:%S) $tag judge"
.venv/bin/python cot.py judge --tag $tag --sets accurate.None.None.p50 --judges category --concurrency 64 || exit 1
echo "--- $(date +%H:%M:%S) $tag all done"
