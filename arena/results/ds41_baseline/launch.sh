#!/usr/bin/env zsh
# Three lanes of 2 sandboxes (machine cap 6): easy then control on one lane, medium and hard on the others.
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena
M=openrouter/deepseek/deepseek-v4.1-flash-1017
run() { .venv/bin/python run_attack.py tiktok --model $M --epochs 4 --max-sandboxes 2 "$@" }
(run --tier easy --run ds41-covert2-easy > results/ds41_baseline/easy.log 2>&1; run --tier easy --preset control --run ds41-ctl-easy > results/ds41_baseline/ctl.log 2>&1) &
run --tier medium --run ds41-covert2-medium > results/ds41_baseline/medium.log 2>&1 &
run --tier hard --run ds41-covert2-hard > results/ds41_baseline/hard.log 2>&1 &
wait
echo "all cells done $(date)"
