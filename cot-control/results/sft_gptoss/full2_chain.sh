#!/bin/bash
cd /home/john/code/tlg-arena-rl/cot-control/rl
set -a; . ~/.config/sauce/tinker.env; set +a
.venv/bin/python gptoss_rewrite.py --jobs ../results/sft_gptoss/full2/rewrite_jobs.jsonl --out ../results/sft_gptoss/full2/rewrites.jsonl --budget 6 2>&1 | grep -v -e PyTorch -e "model of type" | tail -40
.venv/bin/python gptoss_sft_data.py score --run full2 2>&1 | grep -v -e PyTorch -e "model of type"
.venv/bin/python gptoss_sft_data.py force --run full2 --budget 3 2>&1 | grep -v -e PyTorch -e "model of type" | grep -v "^Request failed" | tail -3
.venv/bin/python gptoss_sft_data.py judge --run full2 --budget 3 2>&1 | grep -v -e PyTorch -e "model of type" | tail -2
.venv/bin/python gptoss_sft_data.py accept --run full2 2>&1 | grep -v -e PyTorch -e "model of type"
.venv/bin/python validity_judge.py --run full2 --budget 2 --seam 2>&1 | grep -v -e PyTorch -e "model of type" | tail -4
echo "== full2 done $(date -u +%H:%M)"
