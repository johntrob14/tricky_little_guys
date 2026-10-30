#!/bin/bash
# GLM-5.3 base only: resume xfull cells (540 items x 2 samples per arm), then grade. Re-runnable.
F=/home/john/code/tlg-arena-rl/cot-control/results/outctl/xfull
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/openrouter.env && set +a
for s in cot cot_low expl; do timeout 10000 uv run python ../results/outctl/xmodel.py $F/$s/glm-base-or glm-base-or $s 2 --concurrency 6 >> $F/$s/glm-base-or.log 2>&1 & done
wait
for s in cot cot_low expl; do echo "$s $(wc -l < $F/$s/glm-base-or/gens.jsonl) rows"; done
for s in cot cot_low expl; do OUTCTL_ITEMS=/home/john/code/tlg-arena-rl/cot-control/results/outctl/all_items_rs.json uv run python ../results/outctl/astra_more.py $F/$s/glm-base-or --grade >> $F/$s/glm-base-or.grade.log 2>&1 & done
wait; ls $F/*/glm-base-or/graded.json; echo glm done
