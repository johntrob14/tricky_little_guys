#!/bin/bash
# full3 rebuild: style rewrites keep checks and notation (minimal edits), long-decoy misdirect_end, answer_withheld with dangling-end trim,
# 50% DAPO per type, lenient DAPO source validity; plus the full3p pilot (cover_story map, withheld implicit). Ends with the monitors.
set -u
cd /home/john/code/tlg-arena-rl/cot-control/rl
export PYTHONPATH=/home/john/.cache/cotctl_stubs
set -a; . ~/.config/sauce/tinker.env; set +a
R=../results/sft_gptoss
F="grep -v -e PyTorch -e model.of.type"
echo "== start $(date -u +%H:%M)"
.venv/bin/python gptoss_rewrite.py --jobs $R/full3/rewrite_jobs_main.jsonl --out $R/full3/rewrites.jsonl --budget 10 --long-decoy 2>&1 | $F | tail -40 &
.venv/bin/python gptoss_rewrite.py --jobs $R/full3p/rewrite_jobs.jsonl --out $R/full3p/rewrites.jsonl --budget 1.5 2>&1 | $F | tail -20 &
wait
echo "== rewrites done $(date -u +%H:%M)"
.venv/bin/python gptoss_sft_data.py plain --run full3 2>&1 | $F
for RUN in full3 full3p; do .venv/bin/python gptoss_sft_data.py score --run $RUN 2>&1 | $F; done
.venv/bin/python gptoss_sft_data.py code --run full3 --per-type 80 --seed 7 2>&1 | $F
for RUN in full3 full3p; do
  .venv/bin/python gptoss_sft_data.py force --run $RUN --budget 4 2>&1 | $F | grep -v "^Request failed" | tail -2
  .venv/bin/python gptoss_sft_data.py judge --run $RUN --budget 4 2>&1 | $F | tail -1
  .venv/bin/python gptoss_sft_data.py accept --run $RUN 2>&1 | $F
  .venv/bin/python validity_judge.py --run $RUN --budget 2 --seam 2>&1 | $F | tail -3
  .venv/bin/python validity_judge.py --run $RUN --budget 1 --monitor --monitor-plain 30 2>&1 | $F | tail -30
done
echo "== full3 done $(date -u +%H:%M)"
