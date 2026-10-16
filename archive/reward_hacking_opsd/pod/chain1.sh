#!/bin/bash
# Baseline -> run A (single encourage prompt) -> eval A -> run B (multi) -> eval B. lr 5e-6 (2e-5 saturated in one round in the setup benchmark).
true && . /workspace/rh_opsd/pod/env.sh
cd /workspace/rh_opsd
M=/root/models/qwen3.6-27b
EV="--max-tokens 4096 --grade --gpu-mem 0.9 --max-num-seqs 256"
python -m opsd.eval_gen --model $M --include-base --input data/eval_all.jsonl --output runs/eval/base.jsonl $EV && echo BASE_DONE
for RUN in A_single B_multi; do
  TP=data/train_single.jsonl; [ $RUN = B_multi ] && TP=data/train_multi.jsonl
  python -m opsd.train_opsd --model $M --run-name $RUN --runs-dir /workspace/rh_opsd/runs --latest-dir /root/opsd_local_$RUN \
    --train-prompts $TP --heldout data/heldout_train_dist.jsonl --total-rollouts 768 --round-size 128 --mb-per-step 16 \
    --max-tokens 4096 --eval-every 256 --ckpt-every 256 --lr 5e-6 --gpu-mem 0.85 && echo TRAIN_${RUN}_DONE
  python -m opsd.eval_gen --model $M --lora $RUN=runs/$RUN/ckpt_768 --input data/eval_all.jsonl --output runs/eval/$RUN.jsonl $EV && echo EVAL_${RUN}_DONE
done
echo CHAIN1_DONE
