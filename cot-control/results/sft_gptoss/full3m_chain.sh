#!/bin/bash
# full3m: redo of full3's rejected DAPO misdirect_end long-decoy jobs (inline LaTeX delimiters stripped, wider length slack), then its checks.
set -u
cd /home/john/code/tlg-arena-rl/cot-control/rl
export PYTHONPATH=/home/john/.cache/cotctl_stubs
set -a; . ~/.config/sauce/tinker.env; set +a
R=../results/sft_gptoss
F="grep -v -e PyTorch -e model.of.type"
.venv/bin/python gptoss_rewrite.py --jobs $R/full3m/rewrite_jobs.jsonl --out $R/full3m/rewrites.jsonl --budget 1.5 --long-decoy --seed 3 2>&1 | $F | grep -A8 outcomes
.venv/bin/python gptoss_sft_data.py score --run full3m 2>&1 | $F
.venv/bin/python gptoss_sft_data.py force --run full3m --budget 1 2>&1 | $F | grep -v "^Request failed" | tail -2
.venv/bin/python gptoss_sft_data.py judge --run full3m --budget 1 2>&1 | $F | tail -1
.venv/bin/python gptoss_sft_data.py accept --run full3m 2>&1 | $F
.venv/bin/python validity_judge.py --run full3m --budget 1 2>&1 | $F | tail -3
.venv/bin/python validity_judge.py --run full3m --budget 1 --monitor 2>&1 | $F | tail -10
echo "== full3m done $(date -u +%H:%M)"
