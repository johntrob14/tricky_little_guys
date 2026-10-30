#!/bin/bash
# usage: oversight_qwen_run2.sh <tag> <model> <wait_pid>; waits for the in-flight child, judges the no-line cell, then runs two more lines
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/value_leakage
tag=$1; model=$2; wait_pid=$3
while kill -0 $wait_pid 2>/dev/null; do sleep 10; done
echo "--- $(date +%H:%M:%S) $tag None judge"
.venv/bin/python cot.py judge --tag $tag --sets accurate.None.None.p50 --judges category --concurrency 64 || exit 1
for s in auto_monitor keyword_monitor; do
  echo "--- $(date +%H:%M:%S) $tag $s sample"
  .venv/bin/python bet.py sample --tag $tag --model $model --provider Tinker --effort medium --variants accurate --suffix $s --n 32 --concurrency 128 --sample-concurrency 48 || exit 1
  echo "--- $(date +%H:%M:%S) $tag $s judge"
  .venv/bin/python cot.py judge --tag $tag --sets accurate.None.$s.p50 --judges category --concurrency 64 || exit 1
done
echo "--- $(date +%H:%M:%S) $tag all done"
