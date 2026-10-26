#!/bin/bash
# usage: oversight_analysis_run.sh <tag> <model>; the 2026-10-26 "another AI model / analysis channel" lines on gpt-oss-120b via Tinker, medium renderer, n=48
cd "$(dirname "$0")/.."
set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
tag=$1; model=$2
for s in ${LINES:-model_reads_reasoning model_reads_analysis keyword_analysis}; do
  echo "--- $(date +%H:%M:%S) $tag $s sample"
  .venv/bin/python bet.py sample --tag $tag --model $model --provider Tinker --effort medium --variants accurate --suffix $s --n 48 --concurrency 128 --sample-concurrency 64 || exit 1
  echo "--- $(date +%H:%M:%S) $tag $s judge"
  .venv/bin/python cot.py judge --tag $tag --sets accurate.None.$s.p50 --judges category --concurrency 64 || exit 1
done
echo "--- $(date +%H:%M:%S) $tag all done"
