#!/bin/bash
# usage: [EFFORT=medium] [CELL_LINES="..."] [PREFIX_LINE=1] oversight_mt_run.sh <tag> <model>; the no-line cell plus the 2026-10-24 oversight lines (results/oversight; default all eleven suffix lines and the prefix line) via Tinker, n=48
cd "$(dirname "$0")/.."
tag=$1; model=$2
run() {   # $1 = flag (--suffix/--prefix or empty), $2 = key
  echo "--- $(date +%H:%M:%S) $tag $2 sample"
  .venv/bin/python bet.py sample --tag $tag --model $model --provider Tinker --effort ${EFFORT:-medium} --variants accurate $1 --n 48 --concurrency 128 --sample-concurrency 64 || exit 1
  echo "--- $(date +%H:%M:%S) $tag $2 judge"
  .venv/bin/python cot.py judge --tag $tag --sets accurate.$3.p50 --judges category --concurrency 64 || exit 1
}
run "" None None.None
for s in ${CELL_LINES:-cot_private answer_checked user_reads user_reads_bet friend_reads grader auto_monitor monitor_discard keyword_monitor keyword_discard no_mention_instr}; do run "--suffix $s" $s None.$s; done
[ "${PREFIX_LINE:-1}" = 1 ] && run "--prefix monitor_discard_top" monitor_discard_top monitor_discard_top.None
echo "--- $(date +%H:%M:%S) $tag all done"
