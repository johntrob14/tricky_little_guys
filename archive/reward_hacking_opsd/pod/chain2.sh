#!/bin/bash
# Stage 2: train C (split teacher: base-model reasoning in the student's context, thinking-off encourage-prompted
# answer teacher sharpened at T=0.5; anti-hack instructions in 50% of student prompts) -> eval C on eval_all
# -> base/A/B on the rows eval_all gained since their evals (antihack held-out x2, neutral) -> fried diagnostics.
set -x
true && . /workspace/rh_opsd/pod/env.sh
cd /workspace/rh_opsd
M=/root/models/qwen3.6-27b; R=/workspace/rh_opsd/runs
EV="--max-tokens 4096 --grade --gpu-mem 0.9 --max-num-seqs 256"
python -m opsd.train_opsd --model $M --run-name C --runs-dir $R --latest-dir /root/opsd_local_C \
  --train-prompts data/train_C.jsonl --heldout data/heldout_C.jsonl \
  --split-teacher --reasoning-teacher student_ctx --answer-teacher-mode nothink --answer-teacher-temp 0.5 \
  --total-rollouts 768 --round-size 128 --mb-per-step 16 --max-tokens 4096 --eval-every 256 --ckpt-every 768 \
  --lr 5e-6 --gpu-mem 0.85 && echo TRAIN_C_DONE
date -u
tail -n 1068 data/eval_all.jsonl > data/eval_new_rows.jsonl
python -m opsd.eval_gen --model $M --include-base --lora A_single=$R/A_single/ckpt_768 --lora B_multi=$R/B_multi/ckpt_768 \
  --lora C=$R/C/ckpt_768 --input data/eval_new_rows.jsonl --output runs/eval/new_rows.jsonl $EV && echo EVAL_NEWROWS_DONE
python -m opsd.eval_gen --model $M --lora C=$R/C/ckpt_768 --input data/eval_all.jsonl --output runs/eval/C.jsonl $EV && echo EVAL_C_DONE
date -u
bash pod/fried_chain.sh && echo FRIED_DONE
echo CHAIN2_DONE
