#!/bin/bash
# Pilot: every 27th item (20 items), 1 sample, all models x arms; then grade 3 at a time.
P=/home/john/code/tlg-arena-rl/cot-control/results/outctl/xpilot
cd /home/john/code/tlg-arena-rl/cot-control/rl && set -a && . ~/.config/sauce/tinker.env && . ~/.config/sauce/openrouter.env && set +a
M="qwen-base-or qwen-base-tk qwen-opsd-grpo1b ds-base-or ds-base-tk ds-rwsft1 ds-obf2g glm-base-or"
for m in $M; do for s in cot cot_low expl; do mkdir -p $P/$s/$m
  timeout 3000 uv run python ../results/outctl/xmodel.py $P/$s/$m $m $s 1 --limit-every 27 > $P/$s/$m.log 2>&1 & done; done
wait
for m in $M; do for s in cot cot_low expl; do [ -s $P/$s/$m/gens.jsonl ] && echo "$s $m"; done; done |
  OUTCTL_ITEMS=/home/john/code/tlg-arena-rl/cot-control/results/outctl/all_items_rs.json xargs -P 3 -L 1 bash -c 'uv run python ../results/outctl/astra_more.py '"$P"'/$0/$1 --grade >> '"$P"'/$0/$1.grade.log 2>&1'
grep -l -E "FAILED|Traceback" $P/*/*.log
echo pilot done
