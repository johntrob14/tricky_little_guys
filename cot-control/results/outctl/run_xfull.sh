#!/bin/bash
# Option C: 540 items x {cot, cot_low, expl} x 2 samples for the core five + obf2g; Astra-only Tinker cross-check for Qwen and DeepSeek bases.
# Re-runnable (per-item resume). Grades 3 cells at a time.
F=/home/john/code/tlg-arena-rl/cot-control/results/outctl/xfull
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && . ~/.config/sauce/openrouter.env && set +a
for m in qwen-base-or qwen-opsd-grpo1b ds-base-or ds-rwsft1 ds-obf2g glm-base-or; do for s in cot cot_low expl; do mkdir -p $F/$s/$m
  timeout 10000 uv run python ../results/outctl/xmodel.py $F/$s/$m $m $s 2 --concurrency 32 >> $F/$s/$m.log 2>&1 & done; done
for m in qwen-base-tk ds-base-tk; do for s in cot cot_low expl; do mkdir -p $F/$s/$m
  timeout 10000 uv run python ../results/outctl/xmodel.py $F/$s/$m $m $s 2 --astra-only --concurrency 32 >> $F/$s/$m.log 2>&1 & done; done
wait
for m in qwen-base-or qwen-opsd-grpo1b ds-base-or ds-rwsft1 ds-obf2g glm-base-or qwen-base-tk ds-base-tk; do for s in cot cot_low expl; do
  [ -f $F/$s/$m/graded.json ] || echo "$s $m"; done; done |
  OUTCTL_ITEMS=/home/john/code/tlg-arena-rl/cot-control/results/outctl/all_items_rs.json xargs -P 3 -L 1 bash -c 'uv run python ../results/outctl/astra_more.py '"$F"'/$0/$1 --grade >> '"$F"'/$0/$1.grade.log 2>&1; echo graded $0 $1'
wc -l $F/*/*/gens.jsonl | tail -1; ls $F/*/*/graded.json | wc -l; grep -c FAILED $F/*/*.log | grep -v ":0"
echo xfull done
