#!/bin/bash
# usage: oversight_qwen_run.sh <tag> <model>; the four claim-carrying lines on a Qwen3.8-27B model via Tinker, medium renderer, n=32
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/value_leakage
tag=$1; model=$2
for s in None auto_monitor keyword_monitor keyword_discard; do
  suf=$([ $s = None ] && echo "" || echo "--suffix $s")
  echo "--- $(date +%H:%M:%S) $tag $s sample"
  .venv/bin/python bet.py sample --tag $tag --model $model --provider Tinker --effort medium --variants accurate $suf --n 32 --concurrency 128 --sample-concurrency 48 || exit 1
  echo "--- $(date +%H:%M:%S) $tag $s judge"
  .venv/bin/python cot.py judge --tag $tag --sets accurate.None.$s.p50 --judges category --concurrency 64 || exit 1
done
echo "--- $(date +%H:%M:%S) $tag all done"
